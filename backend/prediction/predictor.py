"""Load a saved pipeline and predict for a new patient WITHOUT retraining."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from backend.config import SAVED_MODELS_DIR, get_spec
from backend.data.loader import coerce_features


class Predictor:
    def __init__(self, disease: str, model_name: str = "logistic_regression"):
        self.spec = get_spec(disease)
        model_dir = SAVED_MODELS_DIR / disease
        path = model_dir / f"{model_name}.joblib"
        if not path.exists():
            available = sorted(p.stem for p in model_dir.glob("*.joblib")) if model_dir.exists() else []
            raise FileNotFoundError(f"No saved model '{model_name}' for '{disease}'. "
                                    f"Available: {available}. Run scripts/train.py first.")
        self.model_name = model_name
        self.pipeline = joblib.load(path)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())

    def required_features(self) -> list[str]:
        return list(self.spec.feature_columns)

    def predict(self, patient: dict) -> dict:
        missing = [f for f in self.spec.feature_columns if f not in patient]
        if missing:
            raise ValueError(f"Patient record missing features: {missing}")
        unknown = [k for k in patient if k not in self.spec.feature_columns]
        X = coerce_features(pd.DataFrame([patient]), self.spec)
        pred = int(self.pipeline.predict(X)[0])
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            score, score_kind = float(self.pipeline.predict_proba(X)[0, 1]), "probability-like"
        else:
            score, score_kind = float(self.pipeline.decision_function(X)[0]), "decision_function (signed distance)"
        return {
            "disease": self.spec.name,
            "model": self.model_name,
            "prediction": pred,
            "meaning": self.spec.target_meaning[pred],
            "score_class_1": round(score, 4),
            "score_note": ("uncalibrated (1+<Z>)/2 from quantum circuit" if self.model_name == "quantum_vqc"
                           else score_kind),
            "ignored_fields": unknown,
            "disclaimer": "Research prototype. Not a medical diagnosis.",
        }
