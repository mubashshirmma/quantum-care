"""Per-prediction explanations, ONLY where the fitted model mathematically supports one.

  logistic_regression  exact: contribution_j = coef_j * z_j  (z = preprocessed feature), sums to logit - intercept
  xgboost              TreeSHAP via booster.predict(pred_contribs=True); sums to logit - bias
  random_forest        no per-patient method here -> global feature_importances_ (clearly labelled as global)
  svm_rbf, quantum_vqc not available (kernel / circuit models need a dedicated method, see Stage 5)

One-hot columns are summed back into their parent categorical feature so the
output is expressed in the original patient fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.data.schema import DiseaseSpec

UNAVAILABLE_NOTES = {
    "svm_rbf": "RBF-kernel SVM has no per-feature decomposition; a model-agnostic method (SHAP KernelExplainer / "
               "permutation importance) is planned for Stage 5.",
    "quantum_vqc": "Inputs are PCA-mixed before angle encoding, so per-feature attribution needs a dedicated method "
                   "(planned for Stage 5). No explanation is shown rather than an invented one.",
    "quantum_qsvc": "Quantum kernel SVM: the decision depends on fidelities to support vectors in Hilbert space; "
                    "no per-feature decomposition exists without a model-agnostic method (planned for Stage 5).",
}


def _group_columns(feature_names: list[str], spec: DiseaseSpec) -> dict[str, list[int]]:
    """Map each original feature to the indices of its preprocessed columns."""
    groups: dict[str, list[int]] = {f: [] for f in spec.feature_columns}
    cats = sorted(spec.categorical_features, key=len, reverse=True)   # longest prefix first
    for i, col in enumerate(feature_names):
        if col in groups and col in spec.numeric_features:
            groups[col].append(i)
            continue
        for c in cats:
            if col.startswith(c + "_"):
                groups[c].append(i)
                break
    return groups


def _aggregate(values: np.ndarray, groups: dict[str, list[int]], spec: DiseaseSpec) -> list[dict]:
    rows = []
    for f, idx in groups.items():
        if not idx:
            continue
        v = float(np.sum(values[idx]))
        rows.append({"feature": f, "label": spec.feature_labels.get(f, f), "contribution": round(v, 4)})
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows


def explain_prediction(pipeline, X: pd.DataFrame, spec: DiseaseSpec, model_name: str) -> dict:
    base = {"model": model_name, "available": False, "patient_specific": False, "method": None,
            "unit": None, "features": [], "note": None}
    prep = pipeline.named_steps.get("prep")
    model = pipeline.named_steps.get("model")
    if prep is None or model is None or not hasattr(prep, "get_feature_names_out"):
        base["note"] = UNAVAILABLE_NOTES.get(model_name, "No explanation method for this model.")
        return base

    try:
        names = list(prep.get_feature_names_out())
    except Exception:
        base["note"] = UNAVAILABLE_NOTES.get(model_name, "Preprocessed feature names unavailable.")
        return base
    groups = _group_columns(names, spec)
    Z = np.asarray(prep.transform(X), dtype=float)[0]

    cls = type(model).__name__
    if cls == "LogisticRegression":
        contrib = model.coef_[0] * Z
        base.update(available=True, patient_specific=True, method="linear log-odds decomposition (coef × standardised value)",
                    unit="log-odds", baseline=round(float(model.intercept_[0]), 4),
                    features=_aggregate(contrib, groups, spec),
                    note="Exact for logistic regression: contributions plus the baseline equal the model's log-odds.")
    elif cls == "XGBClassifier":
        import xgboost as xgb
        contrib = model.get_booster().predict(xgb.DMatrix(Z.reshape(1, -1)), pred_contribs=True)[0]
        base.update(available=True, patient_specific=True, method="TreeSHAP (xgboost pred_contribs)",
                    unit="log-odds", baseline=round(float(contrib[-1]), 4),
                    features=_aggregate(contrib[:-1], groups, spec),
                    note="SHAP values computed exactly from the trained trees for this patient.")
    elif cls == "RandomForestClassifier":
        imp = np.asarray(model.feature_importances_)
        feats = _aggregate(imp, groups, spec)
        base.update(available=True, patient_specific=False, method="global impurity-based feature importance",
                    unit="importance (sums to 1)", features=feats,
                    note="Global importance of each feature to the forest, NOT specific to this patient. "
                         "Per-patient TreeSHAP for random forests is planned for Stage 5.")
    else:
        base["note"] = UNAVAILABLE_NOTES.get(model_name, f"No explanation method for {cls}.")
    for r in base["features"]:
        r["direction"] = ("toward_present" if r["contribution"] > 0 else "toward_absent") if base["patient_specific"] else "importance"
    return base
