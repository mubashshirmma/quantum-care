"""Preprocessing builders.

Everything here is an *unfitted* scikit-learn object. Fitting happens inside
the training routine on the training fold only, so no statistic (mean, scale,
category list, PCA axes, min/max) ever sees test data.

Two builders:
  build_preprocessor(spec)  -> ColumnTransformer shared by ALL models
  build_quantum_feature_pipeline(preprocessor, n_qubits, n_prep_features)
                            -> preprocessor + reduce-to-n_qubits + scale to [0, pi]
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from backend.data.schema import DiseaseSpec


def build_preprocessor(spec: DiseaseSpec) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # drop='if_binary' keeps sex/fbs/exang as one column instead of two redundant ones
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])
    transformers = []
    if spec.numeric_features:
        transformers.append(("num", numeric, list(spec.numeric_features)))
    if spec.categorical_features:
        transformers.append(("cat", categorical, list(spec.categorical_features)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_quantum_feature_pipeline(preprocessor: ColumnTransformer, n_qubits: int,
                                   n_preprocessed_features: int, seed: int) -> Pipeline:
    """Preprocess -> reduce to n_qubits dims (PCA, only if needed) -> scale to [0, pi].

    Angle encoding uses each feature as a rotation angle, so features must sit in a
    bounded range. MinMaxScaler(clip=True) guarantees [0, pi] on unseen data even
    if a test value lies outside the training range.
    """
    if n_preprocessed_features > n_qubits:
        reducer = PCA(n_components=n_qubits, random_state=seed)
    else:
        reducer = "passthrough"
    return Pipeline([
        ("prep", clone(preprocessor)),
        ("reduce", reducer),
        ("angle", MinMaxScaler(feature_range=(0.0, np.pi), clip=True)),
    ])
