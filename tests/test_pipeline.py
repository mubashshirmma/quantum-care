"""Fast sanity tests: run with  .venv/bin/python -m pytest tests -q  (or python -m unittest)."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from backend.config import DISEASE_REGISTRY, get_spec
from backend.data.loader import coerce_features, load_dataset
from backend.data.splitter import stratified_split
from backend.evaluation.metrics import compute_metrics
from backend.models.quantum import QuantumClassifier
from backend.preprocessing.pipeline import build_preprocessor, build_quantum_feature_pipeline


class LoaderTests(unittest.TestCase):
    def test_all_registered_datasets_load_binary(self):
        for spec in DISEASE_REGISTRY.values():
            ds = load_dataset(spec)
            self.assertEqual(sorted(ds.y.unique()), [0, 1], spec.name)
            self.assertEqual(list(ds.X.columns), spec.feature_columns)

    def test_cleveland_binarisation_matches_docs(self):
        spec = get_spec("heart_disease_cleveland")
        raw = pd.read_csv(spec.path, header=None, names=spec.columns, na_values=["?"])
        ds = load_dataset(spec)
        self.assertEqual(int(ds.y.sum()), int((raw["num"] > 0).sum()))
        self.assertEqual(ds.missing_per_feature["ca"], 4)
        self.assertEqual(ds.missing_per_feature["thal"], 2)

    def test_coerce_features_matches_training_dtypes(self):
        spec = get_spec("heart_disease_cleveland")
        patient = {"age": 63, "sex": 1, "cp": 1.0, "trestbps": 145, "chol": 233, "fbs": "1", "restecg": 2,
                   "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
        X = coerce_features(pd.DataFrame([patient]), spec)
        self.assertEqual(X.loc[0, "cp"], "1")       # 1.0 -> "1"
        self.assertEqual(X.loc[0, "fbs"], "1")      # "1" -> "1"
        self.assertEqual(X["age"].dtype, float)


class SplitAndLeakageTests(unittest.TestCase):
    def test_split_is_stratified_and_disjoint(self):
        ds = load_dataset(get_spec("heart_disease_cleveland"))
        Xtr, Xte, ytr, yte = stratified_split(ds.X, ds.y)
        self.assertEqual(len(set(Xtr.index) & set(Xte.index)), 0)
        self.assertAlmostEqual(ytr.mean(), yte.mean(), delta=0.03)

    def test_preprocessor_statistics_come_from_train_only(self):
        ds = load_dataset(get_spec("heart_disease_cleveland"))
        Xtr, Xte, ytr, yte = stratified_split(ds.X, ds.y)
        prep = clone(build_preprocessor(get_spec("heart_disease_cleveland"))).fit(Xtr)
        scaler = prep.named_transformers_["num"].named_steps["scale"]
        train_means = Xtr[get_spec("heart_disease_cleveland").numeric_features].median()  # imputer uses median
        # StandardScaler mean must equal the mean of the (median-imputed) TRAIN numeric columns
        imputed = Xtr[get_spec("heart_disease_cleveland").numeric_features].fillna(train_means)
        np.testing.assert_allclose(scaler.mean_, imputed.mean().values, rtol=1e-6)

    def test_quantum_features_bounded_in_angle_range(self):
        spec = get_spec("heart_disease_cleveland")
        ds = load_dataset(spec)
        Xtr, Xte, ytr, yte = stratified_split(ds.X, ds.y)
        prep = build_preprocessor(spec)
        n_prep = clone(prep).fit(Xtr).transform(Xtr).shape[1]
        q = build_quantum_feature_pipeline(prep, 4, n_prep, seed=0).fit(Xtr)
        Zte = q.transform(Xte)
        self.assertEqual(Zte.shape[1], 4)
        self.assertGreaterEqual(Zte.min(), 0.0)
        self.assertLessEqual(Zte.max(), np.pi)


class MetricsTests(unittest.TestCase):
    def test_metrics_known_values(self):
        y = [0, 0, 1, 1, 1]; p = [0, 1, 1, 1, 0]
        m = compute_metrics(y, p, [0.1, 0.6, 0.9, 0.8, 0.4])
        self.assertEqual(m["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 2})
        self.assertAlmostEqual(m["accuracy"], 0.6)
        self.assertAlmostEqual(m["specificity"], 0.5)
        self.assertAlmostEqual(m["recall"], 2 / 3, places=4)


class QuantumTests(unittest.TestCase):
    def test_learns_simple_separable_rule(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, np.pi, (40, 3)); y = (X[:, 0] > np.pi / 2).astype(int)
        q = QuantumClassifier(n_qubits=3, reps=1, maxiter=60, seed=1).fit(X, y)
        self.assertGreater((q.predict(X) == y).mean(), 0.8)
        self.assertEqual(q.predict_proba(X).shape, (40, 2))

    def test_rejects_wrong_feature_count(self):
        with self.assertRaises(ValueError):
            QuantumClassifier(n_qubits=2).fit(np.zeros((5, 3)), [0, 1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()


class RegistryAndRuleTests(unittest.TestCase):
    def test_target_rules(self):
        from backend.data.schema import apply_target_rule
        raw = pd.Series([0, 1, 2, 3, 0, 4])
        self.assertEqual(apply_target_rule(raw, {"type": "greater_than", "threshold": 0}).tolist(), [0, 1, 1, 1, 0, 1])
        self.assertEqual(apply_target_rule(pd.Series([0, 1, 1]), {"type": "binary"}).tolist(), [0, 1, 1])
        mixed = pd.Series(["M", "B", "M", "B"])
        self.assertEqual(apply_target_rule(mixed, {"type": "positive_values", "values": ["M"]}).tolist(), [1, 0, 1, 0])
        floats = pd.Series([0.0, 1.0, 1.0])       # 0.0 must match "0"
        self.assertEqual(apply_target_rule(floats, {"type": "positive_values", "values": ["0"]}).tolist(), [1, 0, 0])
        with self.assertRaises(ValueError):
            apply_target_rule(raw, {"type": "nope"})

    def test_spec_json_roundtrip(self):
        from backend.config import PROJECT_ROOT
        from backend.data.schema import DiseaseSpec
        for spec in DISEASE_REGISTRY.values():
            d = spec.to_dict(root=PROJECT_ROOT)
            self.assertEqual(DiseaseSpec.from_dict(d, root=PROJECT_ROOT), spec)
            self.assertIn(d["icon"], ("❤️", "🩸", "🧠", "🎗️", "🧬"))

    def test_spec_rejects_bad_definitions(self):
        from backend.data.schema import DiseaseSpec
        from pathlib import Path
        base = dict(name="x_test", display_name="x", path=Path("/nonexistent.csv"), numeric_features=["a"],
                    categorical_features=[], target_column="t", target_rule={"type": "binary"},
                    target_meaning={0: "no", 1: "yes"})
        DiseaseSpec(**base)  # valid
        with self.assertRaises(ValueError):
            DiseaseSpec(**{**base, "category": "dermatology"})
        with self.assertRaises(ValueError):
            DiseaseSpec(**{**base, "numeric_features": ["t"]})          # target as feature
        with self.assertRaises(ValueError):
            DiseaseSpec(**{**base, "numeric_features": ["a"], "categorical_features": ["a"]})
