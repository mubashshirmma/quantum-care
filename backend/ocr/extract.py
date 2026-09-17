"""Medical information extraction from OCR text lines.

Input : list of {"text": str, "confidence": float|None}   (from NVIDIA OCR, or manual transcription)
Output: list of Finding(key, value, unit, raw, confidence, needs_review)

Confidence is inherited from the OCR detections that produced the value. Manual transcription
has no confidence (None) and is never presented as OCR confidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# canonical clinical keys -> label synonyms (lower-case, matched at line start or before ':' / '=')
SYNONYMS: dict[str, list[str]] = {
    "age": ["age", "patient age", "years"],
    "sex": ["sex", "gender"],
    "blood_pressure": ["blood pressure", "bp", "b.p", "b.p.", "arterial pressure", "resting bp", "resting blood pressure"],
    "systolic_bp": ["systolic", "sbp", "systolic bp"],
    "diastolic_bp": ["diastolic", "dbp", "diastolic bp"],
    "cholesterol": ["cholesterol", "total cholesterol", "serum cholesterol", "chol", "tc"],
    "heart_rate": ["heart rate", "hr", "pulse", "pulse rate", "max heart rate", "maximum heart rate", "thalach", "peak heart rate"],
    "fasting_blood_sugar": ["fasting blood sugar", "fbs", "fasting glucose", "fasting plasma glucose", "fpg"],
    "glucose": ["glucose", "plasma glucose", "blood glucose", "ogtt", "glucose 2h", "2h glucose", "random blood sugar", "rbs"],
    "bmi": ["bmi", "body mass index"],
    "insulin": ["insulin", "serum insulin", "2-hour insulin"],
    "skin_thickness": ["skin thickness", "triceps skin fold", "skinfold", "tsf"],
    "pregnancies": ["pregnancies", "gravida", "no. of pregnancies", "number of pregnancies"],
    "diabetes_pedigree": ["diabetes pedigree", "pedigree function", "dpf"],
    "chest_pain": ["chest pain", "chest pain type", "cp", "angina type"],
    "exercise_angina": ["exercise induced angina", "exercise angina", "exang", "angina on exertion"],
    "st_depression": ["st depression", "oldpeak", "st dep"],
    "st_slope": ["st slope", "slope", "slope of st"],
    "rest_ecg": ["resting ecg", "rest ecg", "ecg", "restecg", "electrocardiogram"],
    "major_vessels": ["major vessels", "vessels colored", "vessels coloured", "ca", "fluoroscopy"],
    "thallium": ["thallium", "thal", "thallium stress", "thalium"],
}

# OCR confusions to flag (letter O for zero, l/I for one, S for 5)
SUSPICIOUS = re.compile(r"(?<=\d)[OoIl|S](?=\d|\b)|(?<=\b)[OoIl](?=\d)")
NUM = r"[-+]?\d+(?:[.,]\d+)?"


@dataclass
class Finding:
    key: str
    value: str | float | None
    unit: str | None
    raw: str
    confidence: float | None
    needs_review: bool
    note: str | None = None

    def to_dict(self):
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("：", ":").strip()).lower()


def _split_label_value(line: str) -> tuple[str, str] | None:
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 .()/-]{0,40}?)\s*[:=\-–]\s*(.+)$", line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^\s*([A-Za-z][A-Za-z .()/-]{1,40}?)\s+(\d.*)$", line)     # "Age 54"
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _match_key(label: str) -> str | None:
    lab = re.sub(r"\([^)]*\)", " ", _norm(label)).strip().rstrip(" .:")
    lab = re.sub(r"\s+", " ", lab)
    best, best_len = None, 0
    for key, syns in SYNONYMS.items():
        for syn in syns:
            if lab == syn or lab.startswith(syn + " ") or lab.endswith(" " + syn) or lab == syn.replace(".", ""):
                if len(syn) > best_len:
                    best, best_len = key, len(syn)
    return best


_CONFUSION = [(r"(?<=\d)[Oo](?=\d|\b)", "0"), (r"\b[Oo](?=\d)", "0"), (r"(?<=\d)[Il|](?=\d|\b)", "1"), (r"\b[Il|](?=\d)", "1"),
              (r"(?<=\d)S(?=\d|\b)", "5"), (r"\bS(?=\d)", "5"), (r"(?<=\d)B(?=\d|\b)", "8")]


def _fix_confusions(v: str) -> str:
    """Only touch letters that sit inside/against digit runs (14O -> 140, 25S -> 255); words are untouched."""
    for pat, rep in _CONFUSION:
        v = re.sub(pat, rep, v)
    return v


def _parse_number(v: str):
    fixed = _fix_confusions(v).replace(",", ".")
    m = re.search(NUM, fixed)
    return float(m.group(0)) if m else None


def extract_findings(lines: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[str] = set()
    for ln in lines:
        text = (ln.get("text") or "").strip()
        conf = ln.get("confidence")
        if not text:
            continue
        # several "Label: value" pairs can share one OCR line
        for chunk in re.split(r"\s{2,}|\s\|\s|;|,(?=\s*[A-Za-z][A-Za-z .]{2,}[:=])", text):
            pair = _split_label_value(chunk)
            if not pair:
                continue
            label, value = pair
            key = _match_key(label)
            if not key or key in seen:
                continue
            f = _finding(key, value, chunk, conf)
            if f:
                findings.append(f); seen.add(key)
    return findings


def _finding(key: str, value: str, raw: str, conf) -> Finding | None:
    suspicious = bool(SUSPICIOUS.search(value))
    low_conf = conf is not None and conf < 0.85
    note = None
    if key == "sex":
        v = _norm(value)
        val = "male" if v.startswith("m") else "female" if v.startswith("f") else None
        return Finding(key, val, None, raw, conf, val is None or low_conf, None if val else "could not read sex")
    if key == "blood_pressure":
        m = re.search(rf"({NUM})\s*/\s*({NUM})", _fix_confusions(value))
        if m:
            return Finding(key, {"systolic": float(m.group(1)), "diastolic": float(m.group(2))}, "mm Hg", raw, conf, suspicious or low_conf,
                           "OCR confusion (O/0, l/1) corrected — please check" if suspicious else None)
        n = _parse_number(value)
        return Finding(key, {"systolic": n, "diastolic": None} if n else None, "mm Hg", raw, conf, True, "single value read; systolic assumed")
    if key in ("chest_pain", "st_slope", "rest_ecg", "thallium", "exercise_angina"):
        n = _parse_number(value)
        return Finding(key, n if n is not None else _norm(value), None, raw, conf, True if n is None else (suspicious or low_conf),
                       "categorical value — confirm the category" )
    n = _parse_number(value)
    unit = None
    um = re.search(r"(mg/?dl|mmol/?l|mm ?hg|bpm|kg/m2|kg/m²|%|mm|years|yrs|mu ?u/ml)", value, re.I)
    if um:
        unit = um.group(1)
    if n is None:
        return Finding(key, None, unit, raw, conf, True, "value not readable")
    return Finding(key, n, unit, raw, conf, suspicious or low_conf,
                   "OCR confusion (O/0, l/1) corrected — please check" if suspicious else ("low OCR confidence" if low_conf else None))
