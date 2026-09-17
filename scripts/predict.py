"""Predict for one patient using a saved model (no retraining).

  .venv/bin/python scripts/predict.py --disease heart_disease_cleveland --model quantum_vqc \
      --patient '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
  .venv/bin/python scripts/predict.py --disease heart_disease_cleveland --features   # list required inputs
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import DISEASE_REGISTRY, get_spec  # noqa: E402
from backend.prediction.predictor import Predictor  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", default="heart_disease_cleveland", choices=sorted(DISEASE_REGISTRY))
    p.add_argument("--model", default="logistic_regression")
    p.add_argument("--patient", help="JSON object of feature -> value")
    p.add_argument("--patient-file", help="path to JSON file with feature -> value")
    p.add_argument("--features", action="store_true", help="print required feature names and exit")
    a = p.parse_args()

    if a.features:
        spec = get_spec(a.disease)
        print(json.dumps({"numeric": spec.numeric_features, "categorical": spec.categorical_features,
                          "target_meaning": {str(k): v for k, v in spec.target_meaning.items()}}, indent=2))
        return
    if not (a.patient or a.patient_file):
        p.error("provide --patient JSON or --patient-file")
    patient = json.loads(Path(a.patient_file).read_text()) if a.patient_file else json.loads(a.patient)
    print(json.dumps(Predictor(a.disease, a.model).predict(patient), indent=2))


if __name__ == "__main__":
    main()
