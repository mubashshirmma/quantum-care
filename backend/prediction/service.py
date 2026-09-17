"""Patient prediction service used by the API.

Loads saved pipelines once (cache keyed by disease+model, invalidated when the
metadata file changes, i.e. after retraining) and never retrains.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from backend.config import SAVED_MODELS_DIR, get_spec
from backend.data.loader import coerce_features
from backend.data.schema import DiseaseSpec
from backend.explainability.explain import explain_prediction
from backend.prediction.form_schema import build_form_schema
from backend.prediction.predictor import Predictor

CRITERIA = ("accuracy", "f1", "roc_auc")
MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
                "svm_rbf": "SVM (RBF kernel)", "xgboost": "XGBoost", "quantum_vqc": "Quantum VQC",
                "quantum_qsvc": "Quantum kernel SVM (QSVC)"}
DISCLAIMER = ("This tool is intended for research and decision-support purposes only and is not a substitute "
              "for professional medical diagnosis.")

_predictors: dict[tuple[str, str], tuple[float, Predictor]] = {}


def metadata_for(disease: str) -> dict | None:
    p = SAVED_MODELS_DIR / disease / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else None


def trained_models(disease: str) -> list[str]:
    d = SAVED_MODELS_DIR / disease
    return sorted(p.stem for p in d.glob("*.joblib")) if d.exists() else []


NO_PROBABILITY = {"svm_rbf", "quantum_qsvc"}


def best_model(meta: dict, criterion: str = "accuracy", prefer_probability: bool = False) -> str:
    """Best trained model by a validated hold-out metric. With prefer_probability, models that
    cannot output a probability are only chosen if nothing else is available."""
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    items = list(meta["results"].items())
    if prefer_probability:
        with_prob = [kv for kv in items if kv[0] not in NO_PROBABILITY]
        items = with_prob or items
    return max(items, key=lambda kv: (kv[1].get(criterion) or 0, kv[1]["accuracy"]))[0]


def models_summary(disease: str, criterion: str = "accuracy", prefer_probability: bool = False) -> dict:
    meta = metadata_for(disease)
    avail = trained_models(disease)
    if not meta or not avail:
        return {"disease": disease, "trained": False, "models": [], "best": None, "criterion": criterion}
    best = best_model(meta, criterion, prefer_probability)
    models = []
    for m in avail:
        r = meta["results"].get(m, {})
        models.append({"name": m, "label": MODEL_LABELS.get(m, m), "is_quantum": m.startswith("quantum"),
                       "is_best": m == best, "metrics": {k: r.get(k) for k in ("accuracy", "precision", "recall",
                                                                              "specificity", "f1", "roc_auc")},
                       "has_probability": m not in NO_PROBABILITY,
                       "explainable": m in ("logistic_regression", "xgboost", "random_forest")})
    return {"disease": disease, "trained": True, "models": models, "best": best, "criterion": criterion,
            "prefer_probability": prefer_probability,
            "evaluated_on": f"hold-out test set, n={meta['split']['n_test']} (seed {meta['split']['seed']})",
            "trained_at": meta["timestamp_utc"], "quantum_config": meta.get("quantum_config")}


def _predictor(disease: str, model: str) -> Predictor:
    meta_path = SAVED_MODELS_DIR / disease / "metadata.json"
    mtime = meta_path.stat().st_mtime
    hit = _predictors.get((disease, model))
    if hit and hit[0] == mtime:
        return hit[1]
    pr = Predictor(disease, model)
    _predictors[(disease, model)] = (mtime, pr)
    return pr


def validate_patient(spec: DiseaseSpec, data: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Returns (clean_data, errors, warnings). Errors block; warnings are advisory."""
    schema = build_form_schema(spec)
    errors, warnings, clean = [], [], {}
    for f in schema["features"]:
        name = f["name"]
        raw = data.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors.append(f"'{f['label']}' is required")
            continue
        if f["type"] == "number":
            try:
                v = float(raw)
            except (TypeError, ValueError):
                errors.append(f"'{f['label']}' must be a number (got {raw!r})")
                continue
            lo, hi = f["range"]["min"], f["range"]["max"]
            span = (hi - lo) or 1.0
            if v < lo - 0.25 * span or v > hi + 0.25 * span:
                warnings.append(f"'{f['label']}' = {v:g} is well outside the training range {lo:g}–{hi:g}; "
                                f"the model has not seen such values and its output is unreliable there")
            elif v < lo or v > hi:
                warnings.append(f"'{f['label']}' = {v:g} is slightly outside the training range {lo:g}–{hi:g}")
            clean[name] = v
        else:
            v = str(raw).strip()
            if v.endswith(".0"):
                v = v[:-2]
            allowed = {o["value"] for o in f["options"]}
            if v not in allowed:
                errors.append(f"'{f['label']}' must be one of {sorted(allowed)} (got {raw!r})")
                continue
            clean[name] = v
    unknown = sorted(set(data) - {f["name"] for f in schema["features"]})
    if unknown:
        warnings.append(f"ignored fields not used by this model: {unknown}")
    return clean, errors, warnings


def _risk_level(prediction: int, prob: float | None) -> str:
    if prob is None:
        return "elevated" if prediction == 1 else "low"
    if prob >= 0.65:
        return "elevated"
    if prob >= 0.35:
        return "borderline"
    return "low"


def _confidence(selected_prob: float | None, per_model: dict) -> tuple[str, str]:
    preds = {r["prediction"] for r in per_model.values()}
    agree = len(preds) == 1
    if selected_prob is None:
        return ("moderate" if agree else "low"), "based on agreement between trained models (no probability for this model)"
    margin = abs(selected_prob - 0.5)
    if agree and margin >= 0.3:
        level = "high"
    elif agree or margin >= 0.3:
        level = "moderate"
    else:
        level = "low"
    return level, (f"probability margin from 0.5 = {margin:.2f}; models {'agree' if agree else 'disagree'} on the class. "
                   "This is model agreement, not a clinical certainty.")


def predict_patient(disease: str, data: dict[str, Any], model: str = "best", criterion: str = "accuracy",
                    explain: bool = True, prefer_probability: bool = False) -> dict:
    spec = get_spec(disease)
    meta = metadata_for(disease)
    avail = trained_models(disease)
    if not meta or not avail:
        raise LookupError(f"No trained models for '{disease}'. Train it from the Dataset / Models page first.")

    clean, errors, warnings = validate_patient(spec, data)
    if errors:
        raise ValueError("; ".join(errors))

    best = best_model(meta, criterion, prefer_probability)
    if model == "best":
        selected = best
    elif model == "all":
        selected = best
    elif model in avail:
        selected = model
    else:
        raise ValueError(f"unknown model '{model}'; trained models: {avail}")
    to_run = avail if model in ("all", "best") else [selected]
    if selected not in to_run:
        to_run = [selected] + to_run

    X = coerce_features(pd.DataFrame([clean]), spec)
    per_model = {}
    for m in to_run:
        pr = _predictor(disease, m)
        out = pr.predict(clean)
        r = meta["results"].get(m, {})
        per_model[m] = {
            "label": MODEL_LABELS.get(m, m), "prediction": out["prediction"], "meaning": out["meaning"],
            "probability": out["score_class_1"] if m not in NO_PROBABILITY else None,
            "score_kind": "uncalibrated (1+<Z>)/2" if m == "quantum_vqc" else ("decision margin" if m in NO_PROBABILITY else "model probability"),
            "score": out["score_class_1"], "score_note": out["score_note"],
            "is_quantum": m.startswith("quantum"), "is_best": m == best,
            "validated_accuracy": r.get("accuracy"), "validated_roc_auc": r.get("roc_auc"),
        }

    sel = per_model[selected]
    prob = sel["probability"]
    conf, conf_basis = _confidence(prob, per_model)
    explanation = None
    if explain:
        explanation = explain_prediction(_predictor(disease, selected).pipeline, X, spec, selected)

    return {
        "disease": disease, "display_name": spec.display_name, "icon": spec.icon,
        "inputs": clean,
        "selected_model": selected, "selected_model_label": MODEL_LABELS.get(selected, selected),
        "best_model": best, "best_criterion": criterion, "prefer_probability": prefer_probability,
        "prediction": {
            "label": sel["prediction"], "meaning": sel["meaning"],
            "probability": prob, "risk_level": _risk_level(sel["prediction"], prob),
            "confidence": conf, "confidence_basis": conf_basis,
            "wording": "Model prediction — estimated risk, not a diagnosis",
        },
        "models": per_model,
        "explanation": explanation,
        "warnings": warnings,
        "target_meaning": {str(k): v for k, v in spec.target_meaning.items()},
        "target_documented": spec.target_documented,
        "disclaimer": DISCLAIMER,
    }
