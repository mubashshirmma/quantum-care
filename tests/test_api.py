"""API tests against the saved models (requires saved_models/heart_disease_cleveland from scripts/train.py).
Run:  .venv/bin/python -W ignore -m unittest tests.test_api -v"""
import unittest
import warnings

warnings.simplefilter("ignore")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.app import app  # noqa: E402

PATIENT = {"age": 67, "sex": "1", "cp": "4", "trestbps": 160, "chol": 286, "fbs": "0", "restecg": "2",
           "thalach": 108, "exang": "1", "oldpeak": 1.5, "slope": "2", "ca": 3, "thal": "3"}
D = "heart_disease_cleveland"


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(app)
        cls.c.post("/api/auth/login", json={"username": "admin", "password": "admin123"})

    def test_form_schema_matches_trained_features(self):
        f = self.c.get(f"/api/diseases/{D}/form").json()
        self.assertEqual({x["name"] for x in f["features"]}, set(PATIENT))
        num = next(x for x in f["features"] if x["name"] == "age")
        self.assertEqual(num["type"], "number"); self.assertIn("range", num); self.assertEqual(num["unit"], "years")
        cat = next(x for x in f["features"] if x["name"] == "cp")
        self.assertEqual(cat["type"], "select"); self.assertEqual(len(cat["options"]), 4)
        self.assertEqual(cat["options"][0]["label"], "Typical angina")

    def test_models_and_best(self):
        m = self.c.get(f"/api/diseases/{D}/models").json()
        self.assertTrue(m["trained"]); self.assertEqual(len(m["models"]), 6)
        self.assertEqual(sum(x["is_best"] for x in m["models"]), 1)
        m2 = self.c.get(f"/api/diseases/{D}/models?prefer_probability=true").json()
        self.assertNotEqual(m2["best"], "svm_rbf")

    def test_predict_all_models_no_retrain(self):
        r = self.c.post(f"/api/predict/{D}", json={"data": PATIENT, "model": "all", "patient_name": "T"})
        self.assertEqual(r.status_code, 200, r.text); p = r.json()
        self.assertEqual(set(p["models"]), {"logistic_regression", "random_forest", "svm_rbf", "xgboost", "quantum_vqc", "quantum_qsvc"})
        self.assertIn(p["prediction"]["label"], (0, 1)); self.assertEqual(p["patient_name"], "T")
        self.assertIsNone(p["models"]["svm_rbf"]["probability"])
        self.assertTrue(0 <= p["models"]["logistic_regression"]["probability"] <= 1)
        self.assertIn("not a substitute", p["disclaimer"])

    def test_explanations_are_model_specific_not_invented(self):
        e = self.c.post(f"/api/predict/{D}", json={"data": PATIENT, "model": "logistic_regression"}).json()["explanation"]
        self.assertTrue(e["available"] and e["patient_specific"]); self.assertEqual(len(e["features"]), 13)
        e = self.c.post(f"/api/predict/{D}", json={"data": PATIENT, "model": "xgboost"}).json()["explanation"]
        self.assertTrue(e["available"] and e["patient_specific"]); self.assertIn("SHAP", e["method"])
        e = self.c.post(f"/api/predict/{D}", json={"data": PATIENT, "model": "random_forest"}).json()["explanation"]
        self.assertTrue(e["available"]); self.assertFalse(e["patient_specific"])
        for m in ("svm_rbf", "quantum_vqc", "quantum_qsvc"):
            e = self.c.post(f"/api/predict/{D}", json={"data": PATIENT, "model": m}).json()["explanation"]
            self.assertFalse(e["available"]); self.assertEqual(e["features"], []); self.assertTrue(e["note"])

    def test_validation(self):
        self.assertEqual(self.c.post(f"/api/predict/{D}", json={"data": {**PATIENT, "age": "x"}}).status_code, 422)
        self.assertEqual(self.c.post(f"/api/predict/{D}", json={"data": {**PATIENT, "cp": "9"}}).status_code, 422)
        d = dict(PATIENT); del d["chol"]
        self.assertEqual(self.c.post(f"/api/predict/{D}", json={"data": d}).status_code, 422)
        r = self.c.post(f"/api/predict/{D}", json={"data": {**PATIENT, "chol": 900}})
        self.assertEqual(r.status_code, 200); self.assertTrue(r.json()["warnings"])
        self.assertEqual(self.c.post("/api/predict/nope", json={"data": {}}).status_code, 404)
        self.assertEqual(self.c.post(f"/api/predict/{D}", json={"data": PATIENT, "model": "nope"}).status_code, 422)

    def test_frontend_served(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertIn("Patient Analysis", self.c.get("/").text)
        for f in ("/static/js/app.js", "/static/js/patient.js", "/static/css/style.css"):
            self.assertEqual(self.c.get(f).status_code, 200, f)


if __name__ == "__main__":
    unittest.main()
