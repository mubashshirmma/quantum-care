"""Central configuration: paths, reproducibility, and the disease registry.

Registry = built-in specs (defined here) + user-added specs persisted in
datasets/registry.json. register_spec()/unregister_spec() keep both the
in-memory dict and the JSON file in sync.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend.data.schema import DiseaseSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
USER_REGISTRY_PATH = DATASETS_DIR / "registry.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2

_BUILTIN_SPECS = [
    DiseaseSpec(
        name="heart_disease_cleveland",
        display_name="Heart Disease (UCI Cleveland)",
        category="cardiovascular",
        path=DATASETS_DIR / "heart_disease" / "processed.cleveland.data",
        has_header=False,
        columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"],
        numeric_features=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        categorical_features=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
        target_column="num",
        # UCI docs: num 0 = <50% narrowing (absent); 1-4 = >50% narrowing (present)
        target_rule={"type": "greater_than", "threshold": 0},
        target_meaning={0: "disease absent (<50% diameter narrowing)",
                        1: "disease present (>50% diameter narrowing)"},
        target_documented=True,
        na_values=["?"],
        source="https://archive.ics.uci.edu/dataset/45/heart+disease",
        notes="303 rows. 'ca' and 'thal' contain 6 missing values encoded as '?'. "
              "Binarisation num>0 follows the dataset's own documentation.",
        builtin=True,
        # labels/options transcribed from heart-disease.names (UCI documentation)
        feature_labels={"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
                        "chol": "Serum cholesterol", "fbs": "Fasting blood sugar > 120 mg/dl",
                        "restecg": "Resting ECG result", "thalach": "Maximum heart rate achieved",
                        "exang": "Exercise-induced angina", "oldpeak": "ST depression (exercise vs rest)",
                        "slope": "Slope of peak exercise ST segment", "ca": "Major vessels coloured by fluoroscopy",
                        "thal": "Thallium stress test result"},
        feature_units={"age": "years", "trestbps": "mm Hg", "chol": "mg/dl", "thalach": "bpm", "oldpeak": "mm"},
        feature_options={"sex": {"1": "Male", "0": "Female"},
                         "cp": {"1": "Typical angina", "2": "Atypical angina", "3": "Non-anginal pain", "4": "Asymptomatic"},
                         "fbs": {"1": "Yes (>120 mg/dl)", "0": "No"},
                         "restecg": {"0": "Normal", "1": "ST-T wave abnormality", "2": "Left ventricular hypertrophy"},
                         "exang": {"1": "Yes", "0": "No"},
                         "slope": {"1": "Upsloping", "2": "Flat", "3": "Downsloping"},
                         "thal": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"}},
    ),
    DiseaseSpec(
        name="heart_disease_synthetic",
        display_name="Heart Disease (200-row synthetic)",
        category="cardiovascular",
        path=DATASETS_DIR / "heart_disease" / "heart_disease_200_synthetic.csv",
        has_header=True,
        numeric_features=["age", "blood_pressure", "cholesterol", "max_heart_rate"],
        categorical_features=[],
        target_column="target",
        target_rule={"type": "binary"},
        target_meaning={0: "assumed: disease absent", 1: "assumed: disease present"},
        target_documented=False,
        source="local synthetic file, provenance unknown",
        notes="Target semantics are NOT documented; 1=present is an assumption.",
        builtin=True,
        feature_labels={"age": "Age", "blood_pressure": "Blood pressure", "cholesterol": "Cholesterol",
                        "max_heart_rate": "Maximum heart rate"},
        feature_units={"age": "years", "blood_pressure": "mm Hg", "cholesterol": "mg/dl", "max_heart_rate": "bpm"},
    ),
    DiseaseSpec(
        name="diabetes_pima",
        display_name="Diabetes (Pima Indians)",
        category="diabetes",
        path=DATASETS_DIR / "diabetes_pima" / "pima_indians_diabetes.csv",
        has_header=True,
        numeric_features=["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi",
                          "diabetes_pedigree", "age"],
        categorical_features=[],
        target_column="outcome",
        target_rule={"type": "binary"},
        target_meaning={0: "tested negative for diabetes", 1: "tested positive for diabetes (WHO criteria)"},
        target_documented=True,
        source="NIDDK via UCI / jbrownlee Datasets mirror (pima-indians-diabetes.data.csv)",
        notes="768 female patients of Pima Indian heritage, age >= 21. Zeros in glucose, blood_pressure, "
              "skin_thickness, insulin and bmi encode missing values and are imputed from the training split.",
        builtin=True,
        feature_labels={"pregnancies": "Number of pregnancies", "glucose": "Plasma glucose (2 h OGTT)",
                        "blood_pressure": "Diastolic blood pressure", "skin_thickness": "Triceps skin fold thickness",
                        "insulin": "2-hour serum insulin", "bmi": "Body mass index", "diabetes_pedigree": "Diabetes pedigree function",
                        "age": "Age"},
        feature_units={"glucose": "mg/dl", "blood_pressure": "mm Hg", "skin_thickness": "mm", "insulin": "mu U/ml",
                       "bmi": "kg/m²", "age": "years"},
        zero_as_missing=["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"],
    ),
]

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


def _load_user_specs() -> list[DiseaseSpec]:
    if not USER_REGISTRY_PATH.exists():
        return []
    try:
        raw = json.loads(USER_REGISTRY_PATH.read_text())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{USER_REGISTRY_PATH} is not valid JSON: {e}") from e
    return [DiseaseSpec.from_dict({**d, "builtin": False}, root=PROJECT_ROOT) for d in raw]


def _save_user_specs() -> None:
    user = [s.to_dict(root=PROJECT_ROOT) for s in DISEASE_REGISTRY.values() if not s.builtin]
    USER_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_REGISTRY_PATH.write_text(json.dumps(user, indent=2, ensure_ascii=False))


DISEASE_REGISTRY: dict[str, DiseaseSpec] = {s.name: s for s in _BUILTIN_SPECS}
for _s in _load_user_specs():
    DISEASE_REGISTRY.setdefault(_s.name, _s)


def get_spec(disease: str) -> DiseaseSpec:
    try:
        return DISEASE_REGISTRY[disease]
    except KeyError:
        raise KeyError(f"Unknown disease '{disease}'. Known: {sorted(DISEASE_REGISTRY)}") from None


def validate_slug(name: str) -> str:
    if not _SLUG_RE.match(name):
        raise ValueError("name must be a slug: lowercase letters, digits, underscores, 3-64 chars, "
                         "starting with a letter (e.g. 'diabetes_pima')")
    return name


def register_spec(spec: DiseaseSpec) -> DiseaseSpec:
    validate_slug(spec.name)
    if spec.name in DISEASE_REGISTRY:
        raise ValueError(f"disease '{spec.name}' already exists")
    DISEASE_REGISTRY[spec.name] = spec
    _save_user_specs()
    return spec


def unregister_spec(name: str) -> DiseaseSpec:
    spec = get_spec(name)
    if spec.builtin:
        raise ValueError(f"'{name}' is built in and cannot be removed")
    del DISEASE_REGISTRY[name]
    _save_user_specs()
    return spec
