"""Quantum Lab + preprocessing pipeline views (read-only, from saved artefacts)."""
from __future__ import annotations

import json

import joblib
from fastapi import APIRouter, HTTPException

from backend.config import SAVED_MODELS_DIR, get_spec

router = APIRouter(prefix="/api", tags=["lab"])


def _meta(name: str) -> dict | None:
    p = SAVED_MODELS_DIR / name / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else None


@router.get("/quantum/{name}")
def quantum_lab(name: str):
    try:
        spec = get_spec(name)
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    meta = _meta(name)
    if not meta:
        raise HTTPException(409, "No trained models for this disease yet.")
    models = meta.get("quantum_models")
    if not models:                       # older metadata (before QSVC / circuit text): rebuild from the saved pipeline
        models = {}
        for m in ("quantum_vqc", "quantum_qsvc"):
            p = SAVED_MODELS_DIR / name / f"{m}.joblib"
            if p.exists():
                q = joblib.load(p).named_steps["model"]
                models[m] = {**q.describe(), "circuit_text": q.draw(), "feature_reduction": meta["quantum_config"].get("feature_reduction")}
    for m, info in models.items():
        info["results"] = meta["results"].get(m)
        info["training_info"] = meta["training_info"].get(m)
    return {"disease": name, "display_name": spec.display_name, "icon": spec.icon, "trained_at": meta["timestamp_utc"],
            "environment": meta["environment"], "n_features_after_preprocessing": meta["preprocessing"]["n_features_after_preprocessing"],
            "models": models,
            "classical_reference": {m: r for m, r in meta["results"].items() if not m.startswith("quantum")}}


@router.get("/diseases/{name}/pipeline")
def pipeline_view(name: str):
    try:
        spec = get_spec(name)
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    meta = _meta(name)
    n_num, n_cat = len(spec.numeric_features), len(spec.categorical_features)
    steps = [
        {"step": "Raw data", "detail": f"{spec.path.name}: {n_num} numeric + {n_cat} categorical features, target '{spec.target_column}' "
                                       f"({spec.to_dict()['target_rule_text']})", "leakage_safe": None},
        {"step": "Stratified split", "detail": f"{int((1 - 0.2) * 100)}/20 hold-out, seed 42; test rows never touched by any fit()", "leakage_safe": True},
        {"step": "Missing values", "detail": "numeric: median imputation; categorical: most-frequent" +
                                             (f"; zeros treated as missing in {spec.zero_as_missing}" if spec.zero_as_missing else ""), "leakage_safe": True},
        {"step": "Encoding", "detail": f"one-hot for {n_cat} categorical features (binary columns kept as one column)" if n_cat else "no categorical features", "leakage_safe": True},
        {"step": "Scaling", "detail": "StandardScaler on numeric features", "leakage_safe": True},
    ]
    if meta:
        n_prep = meta["preprocessing"]["n_features_after_preprocessing"]
        q = meta.get("quantum_config") or {}
        steps += [
            {"step": "Feature selection / reduction (quantum branch)", "detail": f"{n_prep} preprocessed features -> {q.get('feature_reduction', 'n/a')}", "leakage_safe": True},
            {"step": "Quantum feature encoding", "detail": f"MinMax to [0, pi] then {q.get('feature_map')} encoding on {q.get('n_qubits')} qubits", "leakage_safe": True},
            {"step": "Models", "detail": "classical: " + ", ".join(m for m in meta["results"] if not m.startswith("quantum")) +
                                        " | quantum: " + ", ".join(m for m in meta["results"] if m.startswith("quantum")), "leakage_safe": None},
        ]
    else:
        steps.append({"step": "Models", "detail": "not trained yet", "leakage_safe": None})
    return {"disease": name, "display_name": spec.display_name, "icon": spec.icon, "steps": steps, "trained": meta is not None}
