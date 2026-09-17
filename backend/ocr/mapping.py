"""Map extracted clinical findings onto the feature schema of a specific trained disease model.

Each disease declares which canonical finding feeds which model feature and how to convert it.
Anything not mapped stays 'missing' and must be filled by the user; nothing is guessed.
"""
from __future__ import annotations

from backend.data.schema import DiseaseSpec
from backend.ocr.extract import Finding
from backend.prediction.form_schema import build_form_schema

# disease -> feature -> (finding key, converter name)
DISEASE_MAPPINGS: dict[str, dict[str, tuple[str, str]]] = {
    "heart_disease_cleveland": {
        "age": ("age", "number"), "sex": ("sex", "sex_1m_0f"), "trestbps": ("blood_pressure", "systolic"),
        "chol": ("cholesterol", "number"), "thalach": ("heart_rate", "number"), "fbs": ("fasting_blood_sugar", "fbs_gt_120"),
        "cp": ("chest_pain", "category"), "exang": ("exercise_angina", "yes_no_1_0"), "oldpeak": ("st_depression", "number"),
        "slope": ("st_slope", "category"), "restecg": ("rest_ecg", "category"), "ca": ("major_vessels", "number"),
        "thal": ("thallium", "category"),
    },
    "heart_disease_synthetic": {
        "age": ("age", "number"), "blood_pressure": ("blood_pressure", "systolic"), "cholesterol": ("cholesterol", "number"),
        "max_heart_rate": ("heart_rate", "number"),
    },
    "diabetes_pima": {
        "pregnancies": ("pregnancies", "number"), "glucose": ("glucose", "number"), "blood_pressure": ("blood_pressure", "diastolic"),
        "skin_thickness": ("skin_thickness", "number"), "insulin": ("insulin", "number"), "bmi": ("bmi", "number"),
        "diabetes_pedigree": ("diabetes_pedigree", "number"), "age": ("age", "number"),
    },
}

# generic fallback for user-added datasets: match model feature names to finding keys directly
GENERIC = {"age": ("age", "number"), "sex": ("sex", "sex_1m_0f"), "glucose": ("glucose", "number"), "bmi": ("bmi", "number"),
           "insulin": ("insulin", "number"), "cholesterol": ("cholesterol", "number"), "chol": ("cholesterol", "number"),
           "blood_pressure": ("blood_pressure", "systolic"), "heart_rate": ("heart_rate", "number"), "max_heart_rate": ("heart_rate", "number")}


def _convert(kind: str, f: Finding, field: dict):
    v = f.value
    if v is None:
        return None
    if kind == "number":
        return float(v) if isinstance(v, (int, float)) else None
    if kind in ("systolic", "diastolic"):
        return v.get(kind) if isinstance(v, dict) else None
    if kind == "sex_1m_0f":
        if v in ("male", "female"):
            # honour the dataset's own option labels when present
            for o in field.get("options", []):
                if o["label"].lower().startswith(v[0]):
                    return o["value"]
            return "1" if v == "male" else "0"
        return None
    if kind == "fbs_gt_120":
        if isinstance(v, (int, float)):
            return "1" if v > 120 else "0"
        return None
    if kind == "yes_no_1_0":
        if isinstance(v, str):
            return "1" if v.startswith("y") or v.startswith("pos") else "0" if v.startswith("n") or v.startswith("neg") else None
        if isinstance(v, (int, float)):
            return str(int(v)) if int(v) in (0, 1) else None
        return None
    if kind == "category":
        allowed = {o["value"] for o in field.get("options", [])}
        if isinstance(v, (int, float)) and str(int(v)) in allowed:
            return str(int(v))
        if isinstance(v, str):
            for o in field.get("options", []):
                if o["label"].lower() in v or v in o["label"].lower():
                    return o["value"]
        return None
    return None


def map_findings(spec: DiseaseSpec, findings: list[Finding], source: str) -> dict:
    schema = build_form_schema(spec)
    mapping = DISEASE_MAPPINGS.get(spec.name) or {f["name"]: GENERIC[f["name"]] for f in schema["features"] if f["name"] in GENERIC}
    by_key = {f.key: f for f in findings}
    out, used = {}, set()
    for field in schema["features"]:
        name = field["name"]
        entry = {"feature": name, "label": field["label"], "type": field["type"], "value": None, "confidence": None,
                 "source_text": None, "status": "missing", "note": "not found in document"}
        if name in mapping and mapping[name][0] in by_key:
            f = by_key[mapping[name][0]]; used.add(f.key)
            val = _convert(mapping[name][1], f, field)
            entry.update(value=val, confidence=f.confidence, source_text=f.raw, note=f.note)
            if val is None:
                entry["status"] = "needs_review"; entry["note"] = f.note or "value could not be converted to the model's format"
            elif f.needs_review:
                entry["status"] = "low_confidence"
            else:
                entry["status"] = "ok"
            if field["type"] == "number" and isinstance(val, (int, float)):
                lo, hi = field["range"]["min"], field["range"]["max"]
                if val < lo or val > hi:
                    entry["status"] = "needs_review"; entry["note"] = f"{val:g} is outside the training range {lo:g}–{hi:g}"
        out[name] = entry
    unmapped = [f.to_dict() for f in findings if f.key not in used]
    n_ok = sum(1 for e in out.values() if e["status"] == "ok")
    return {"disease": spec.name, "source": source, "features": list(out.values()), "unmapped_findings": unmapped,
            "summary": {"total": len(out), "ok": n_ok, "low_confidence": sum(1 for e in out.values() if e["status"] == "low_confidence"),
                        "needs_review": sum(1 for e in out.values() if e["status"] == "needs_review"),
                        "missing": sum(1 for e in out.values() if e["status"] == "missing")},
            "confidence_available": any(f.confidence is not None for f in findings),
            "note": "Nothing is sent to the model until you confirm every value. Missing fields must be entered manually."}
