"""Classical baseline factory. Each model is wrapped in a Pipeline with its own
clone of the shared preprocessor so that fit() on the training set fits both."""
from __future__ import annotations

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier


def classical_estimators(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),  # score via decision_function
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="logloss", random_state=seed, n_jobs=1),
    }


def build_classical_pipelines(preprocessor: ColumnTransformer, seed: int) -> dict[str, Pipeline]:
    return {name: Pipeline([("prep", clone(preprocessor)), ("model", est)])
            for name, est in classical_estimators(seed).items()}


def positive_class_score(pipeline: Pipeline, X):
    """Continuous score for class 1, used for ROC-AUC."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)
