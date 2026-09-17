"""Train every model on the SAME stratified split with the SAME preprocessing,
evaluate on the untouched test set, persist artefacts, and return a comparison.

Artefacts written:
  saved_models/<disease>/<model>.joblib      fitted Pipeline (preprocessing + model)
  saved_models/<disease>/metadata.json       features, target meaning, split, results
  experiments/<disease>_<timestamp>.json     full run record for later comparison
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import qiskit
import qiskit_machine_learning
import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from backend.config import EXPERIMENTS_DIR, RANDOM_SEED, SAVED_MODELS_DIR, TEST_SIZE, get_spec
from backend.data.loader import load_dataset
from backend.data.splitter import describe_split, stratified_split
from backend.evaluation.metrics import compute_metrics, format_comparison_table, format_confusion
from backend.models.classical import build_classical_pipelines, positive_class_score
from backend.models.quantum import QuantumClassifier, QuantumKernelClassifier
from sklearn.metrics import roc_curve
from backend.preprocessing.pipeline import build_preprocessor, build_quantum_feature_pipeline

DEFAULT_QUANTUM_CONFIG = dict(n_qubits=4, reps=1, entanglement="linear", feature_map="angle",
                              observable="parity", optimizer="cobyla", maxiter=150)


def _roc_points(y_true, y_score, max_points: int = 60) -> dict:
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))
    if len(fpr) > max_points:                       # thin evenly, always keep the end points
        idx = np.unique(np.concatenate([[0], np.linspace(0, len(fpr) - 1, max_points).astype(int), [len(fpr) - 1]]))
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
    return {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr],
            "thresholds": [None if not np.isfinite(t) else round(float(t), 4) for t in thr]}


def _fit_and_eval(name, pipeline, X_train, y_train, X_test, y_test, log):
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    y_score = positive_class_score(pipeline, X_test)
    predict_s = time.perf_counter() - t1
    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics["roc_curve"] = _roc_points(y_test, y_score)
    train_acc = float((pipeline.predict(X_train) == np.asarray(y_train)).mean())
    log(f"  {name:<22} trained in {train_s:6.2f}s  train_acc={train_acc:.3f}  test_acc={metrics['accuracy']:.3f}")
    return metrics, {"train_time_s": round(train_s, 3), "predict_time_s": round(predict_s, 4),
                     "predict_ms_per_sample": round(1000 * predict_s / max(1, len(y_test)), 3), "train_accuracy": round(train_acc, 4)}


def run_benchmark(disease: str, quantum_config: dict | None = None, run_quantum: bool = True,
                  seed: int = RANDOM_SEED, test_size: float = TEST_SIZE, save: bool = True,
                  verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    spec = get_spec(disease)
    qcfg = {**DEFAULT_QUANTUM_CONFIG, **(quantum_config or {})}

    # ---- data
    ds = load_dataset(spec)
    log(ds.summary())
    X_train, X_test, y_train, y_test = stratified_split(ds.X, ds.y, test_size=test_size, seed=seed)
    log(describe_split(y_train, y_test))

    # ---- shared preprocessing (unfitted template; each pipeline fits its own clone on train only)
    preprocessor = build_preprocessor(spec)
    n_prep = clone(preprocessor).fit(X_train).transform(X_train).shape[1]
    log(f"Preprocessed feature count: {n_prep}")

    results, extra, pipelines = {}, {}, {}

    # ---- classical
    log("\nClassical baselines:")
    for name, pipe in build_classical_pipelines(preprocessor, seed).items():
        results[name], extra[name] = _fit_and_eval(name, pipe, X_train, y_train, X_test, y_test, log)
        pipelines[name] = pipe

    # ---- quantum
    quantum_desc = None
    quantum_models = {}
    if run_quantum:
        log("\nQuantum classifiers:")
        reduction = (f"PCA {n_prep}->{qcfg['n_qubits']}" if n_prep > qcfg["n_qubits"] else "none (features <= qubits)")
        # (1) variational quantum classifier
        qmodel = QuantumClassifier(seed=seed, **qcfg)
        quantum_desc = qmodel.describe()
        quantum_desc["feature_reduction"] = reduction
        quantum_desc["angle_scaling"] = "MinMax to [0, pi], fitted on train, clipped on test"
        for k, v in quantum_desc.items():
            log(f"  {k:<20}: {v}")
        qfeatures = build_quantum_feature_pipeline(preprocessor, qcfg["n_qubits"], n_prep, seed)
        qpipe = Pipeline([("prep", qfeatures), ("model", qmodel)])
        name = "quantum_vqc"
        results[name], extra[name] = _fit_and_eval(name, qpipe, X_train, y_train, X_test, y_test, log)
        extra[name]["final_train_loss"] = round(qmodel.loss_history_[-1], 4) if qmodel.loss_history_ else None
        extra[name]["optimizer_evaluations"] = len(qmodel.loss_history_)
        pipelines[name] = qpipe
        quantum_models[name] = {"type": "Variational Quantum Classifier (EstimatorQNN + COBYLA)", **qmodel.describe(),
                                "feature_reduction": reduction, "circuit_text": qmodel.draw(),
                                "loss_history": [round(v, 4) for v in qmodel.loss_history_]}
        # (2) quantum kernel SVM
        kcfg = dict(n_qubits=qcfg["n_qubits"], feature_map="zz", entanglement=qcfg["entanglement"], reps=1, seed=seed)
        kmodel = QuantumKernelClassifier(**kcfg)
        kpipe = Pipeline([("prep", build_quantum_feature_pipeline(preprocessor, qcfg["n_qubits"], n_prep, seed)), ("model", kmodel)])
        name = "quantum_qsvc"
        results[name], extra[name] = _fit_and_eval(name, kpipe, X_train, y_train, X_test, y_test, log)
        pipelines[name] = kpipe
        quantum_models[name] = {"type": "Quantum kernel SVM (fidelity kernel, exact statevector)", **kmodel.describe(),
                                "feature_reduction": reduction, "circuit_text": kmodel.draw()}

    # ---- report
    log("\n" + "=" * 100)
    log(f"BENCHMARK  {spec.display_name}   test n={len(y_test)}  seed={seed}")
    log("=" * 100)
    log(format_comparison_table(results, extra))
    for name, m in results.items():
        log(f"\n{name}\n{format_confusion(m)}")

    record = {
        "disease": disease,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": spec.to_metadata(),
        "split": {"strategy": "stratified_holdout", "test_size": test_size, "seed": seed,
                  "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                  "train_positives": int(y_train.sum()), "test_positives": int(y_test.sum())},
        "preprocessing": {"numeric": "median impute + StandardScaler",
                          "categorical": "most_frequent impute + OneHot(drop_if_binary)",
                          "n_features_after_preprocessing": int(n_prep)},
        "quantum_config": quantum_desc,
        "quantum_models": quantum_models,
        "results": results,
        "training_info": extra,
        "environment": {"python": platform.python_version(), "qiskit": qiskit.__version__,
                        "qiskit_machine_learning": qiskit_machine_learning.__version__,
                        "sklearn": sklearn.__version__},
        "disclaimer": "Research prototype. Not a medical device. Not clinically validated.",
    }

    if save:
        out_dir = SAVED_MODELS_DIR / disease
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, pipe in pipelines.items():
            joblib.dump(pipe, out_dir / f"{name}.joblib")
        (out_dir / "metadata.json").write_text(json.dumps(record, indent=2))
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        exp_path = EXPERIMENTS_DIR / f"{disease}_{record['timestamp_utc'].replace(':', '').replace('-', '')}.json"
        exp_path.write_text(json.dumps(record, indent=2))
        log(f"\nSaved {len(pipelines)} models + metadata -> {out_dir}")
        log(f"Experiment record -> {exp_path}")

    return record
