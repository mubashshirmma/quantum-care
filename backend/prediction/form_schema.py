"""Configuration-driven patient form: one schema per disease, derived from the
DiseaseSpec (labels, units, option names) plus training-data statistics
(numeric ranges, observed categorical values). Cached per dataset file mtime."""
from __future__ import annotations

import numpy as np

from backend.data.loader import load_dataset
from backend.data.schema import DiseaseSpec

_cache: dict[str, tuple[float, dict]] = {}


def _step_for(series) -> float:
    vals = series.dropna()
    if len(vals) and np.all(np.equal(np.mod(vals, 1), 0)):
        return 1
    return 0.1


def build_form_schema(spec: DiseaseSpec) -> dict:
    mtime = spec.path.stat().st_mtime
    hit = _cache.get(spec.name)
    if hit and hit[0] == mtime:
        return hit[1]

    ds = load_dataset(spec)
    fields = []
    for col in spec.numeric_features:
        s = ds.X[col]
        fields.append({
            "name": col, "label": spec.feature_labels.get(col, col.replace("_", " ").capitalize()),
            "type": "number", "required": True, "unit": spec.feature_units.get(col),
            "step": _step_for(s),
            "range": {"min": float(s.min()), "max": float(s.max()), "median": float(s.median())},
            "hint": f"training data: {s.min():g} – {s.max():g}",
        })
    for col in spec.categorical_features:
        counts = ds.X[col].value_counts()
        labels = spec.feature_options.get(col, {})
        options = [{"value": str(v), "label": labels.get(str(v), str(v)), "count": int(n)} for v, n in counts.items()]
        options.sort(key=lambda o: (o["label"] not in labels.values(), o["value"]))
        fields.append({
            "name": col, "label": spec.feature_labels.get(col, col.replace("_", " ").capitalize()),
            "type": "select", "required": True, "options": options,
            "hint": f"{len(options)} categories in training data",
        })
    schema = {
        "disease": spec.name, "display_name": spec.display_name, "icon": spec.icon, "category": spec.category,
        "target_meaning": {str(k): v for k, v in spec.target_meaning.items()},
        "target_documented": spec.target_documented,
        "n_training_rows": int(len(ds.y)),
        "features": fields,
    }
    _cache[spec.name] = (mtime, schema)
    return schema
