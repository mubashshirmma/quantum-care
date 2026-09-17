"""FastAPI backend (Stage 6 slice).

  GET  /api/categories                     icon + label per category
  GET  /api/diseases                       registered datasets (cards)
  GET  /api/diseases/{name}                spec + data summary + benchmark results if trained
  DELETE /api/diseases/{name}              remove a user-added dataset
  POST /api/datasets/inspect               upload CSV -> column profile (to build the add form)
  POST /api/datasets                       upload CSV + spec -> validate + register
  POST /api/train/{name}                   start benchmark in background thread
  GET  /api/train/{name}/status
  GET  /api/diseases/{name}/form           configuration-driven patient form schema
  GET  /api/diseases/{name}/models         trained models + validated metrics + best
  POST /api/predict/{name}                 predict for ONE patient with saved model(s); never retrains
  /api/auth/*                              operator login / session (see auth_routes.py)
  /api/patients*, /api/analyses*           patient registry + OTP registration (see patient_routes.py)
  GET  /                                   frontend

Run:  .venv/bin/uvicorn backend.api.app:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import re
import shutil
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.auth_routes import router as auth_router
from backend.api.lab_routes import router as lab_router
from backend.api.ocr_routes import router as ocr_router
from backend.api.patient_routes import router as patient_router
from backend.auth.middleware import AuthMiddleware
from backend.auth.service import ensure_default_user
from backend.config import (DATASETS_DIR, DISEASE_REGISTRY, PROJECT_ROOT, SAVED_MODELS_DIR,
                            get_spec, register_spec, unregister_spec, validate_slug)
from backend.core.db import init_db
from backend.data.loader import load_dataset
from backend.data.schema import CATEGORIES, DiseaseSpec
from backend.prediction import service as predict_service
from backend.prediction.form_schema import build_form_schema
from backend.training.benchmark import run_benchmark

FRONTEND_DIR = PROJECT_ROOT / "frontend"
MAX_UPLOAD_MB = 50
TARGET_NAME_HINTS = re.compile(r"target|label|class|outcome|diagnosis|disease|result|status|^num$|^y$", re.I)

app = FastAPI(title="Hybrid Quantum ML Disease Platform", version="0.3.0")
init_db()
ensure_default_user()
app.add_middleware(AuthMiddleware)          # every /api/* route except /api/auth/* and /api/health needs a session
app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(ocr_router)
app.include_router(lab_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}



# ----------------------------------------------------------------- helpers
def _metadata(name: str) -> dict | None:
    p = SAVED_MODELS_DIR / name / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else None


def _card(spec: DiseaseSpec) -> dict:
    meta = _metadata(spec.name)
    best = None
    if meta:
        best = max(meta["results"].items(), key=lambda kv: kv[1]["accuracy"])
    return {
        "name": spec.name, "display_name": spec.display_name, "category": spec.category,
        "icon": spec.icon, "builtin": spec.builtin,
        "n_features": len(spec.feature_columns), "target_documented": spec.target_documented,
        "trained": meta is not None,
        "trained_at": meta["timestamp_utc"] if meta else None,
        "best_model": {"name": best[0], "accuracy": best[1]["accuracy"]} if best else None,
        "training": _train_state.get(spec.name, {}).get("state"),
    }


def _read_upload(file: UploadFile) -> bytes:
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
    if not data.strip():
        raise HTTPException(400, "empty file")
    return data


def _profile_columns(df: pd.DataFrame) -> list[dict]:
    out = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(s)
        if TARGET_NAME_HINTS.search(str(col)) and n_unique <= 10:
            role = "target"
        elif is_num and n_unique > 10:
            role = "numeric"
        elif is_num and n_unique <= 10:
            role = "categorical"
        else:
            role = "categorical"
        sample = s.dropna().unique()[:12]
        out.append({
            "name": str(col), "dtype": str(s.dtype), "n_unique": n_unique,
            "n_missing": int(s.isna().sum()),
            "sample_values": [v.item() if hasattr(v, "item") else v for v in sample],
            "suggested_role": role,
        })
    return out


# ----------------------------------------------------------------- routes: read
@app.get("/api/categories")
def categories():
    return [{"key": k, **v} for k, v in CATEGORIES.items()]


@app.get("/api/diseases")
def diseases():
    return [_card(s) for s in DISEASE_REGISTRY.values()]


@app.get("/api/diseases/{name}")
def disease_detail(name: str):
    try:
        spec = get_spec(name)
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    detail = {"spec": spec.to_dict(root=PROJECT_ROOT), "card": _card(spec)}
    try:
        ds = load_dataset(spec)
        counts = ds.y.value_counts().sort_index()
        detail["data"] = {"n_rows": int(len(ds.y)), "n_dropped_missing_target": ds.n_rows_dropped_missing_target,
                          "class_counts": {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))},
                          "missing_per_feature": {k: v for k, v in ds.missing_per_feature.items() if v}}
    except Exception as e:  # dataset file may have been moved/deleted
        detail["data"] = {"error": str(e)}
    detail["benchmark"] = _metadata(name)
    return detail


@app.delete("/api/diseases/{name}")
def disease_delete(name: str):
    try:
        spec = unregister_spec(name)
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    except ValueError as e:
        raise HTTPException(400, str(e))
    # user-added datasets live in their own folder datasets/<name>/ ; remove it and any saved models
    folder = spec.path.parent
    if folder.parent == DATASETS_DIR and folder.name == name:
        shutil.rmtree(folder, ignore_errors=True)
    shutil.rmtree(SAVED_MODELS_DIR / name, ignore_errors=True)
    return {"removed": name}


# ----------------------------------------------------------------- routes: add dataset
@app.post("/api/datasets/inspect")
def inspect_dataset(file: UploadFile = File(...)):
    data = _read_upload(file)
    try:
        df = pd.read_csv(io.BytesIO(data), na_values=["?", "NA", "N/A", ""])
    except Exception as e:
        raise HTTPException(400, f"could not parse CSV: {e}")
    if df.shape[1] < 2:
        raise HTTPException(400, "CSV needs at least one feature column and one target column")
    return {
        "filename": file.filename, "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
        "columns": _profile_columns(df),
        "preview": json.loads(df.head(5).to_json(orient="records")),
    }


@app.post("/api/datasets", status_code=201)
def add_dataset(file: UploadFile = File(...), spec_json: str = Form(...)):
    try:
        body = json.loads(spec_json)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"spec_json is not valid JSON: {e}")

    name = str(body.get("name", "")).strip().lower()
    try:
        validate_slug(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if name in DISEASE_REGISTRY:
        raise HTTPException(409, f"disease '{name}' already exists")

    data = _read_upload(file)
    folder = DATASETS_DIR / name
    folder.mkdir(parents=True, exist_ok=False)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "data.csv").name) or "data.csv"
    csv_path = folder / safe_name
    csv_path.write_bytes(data)

    try:
        tm = body.get("target_meaning") or {}
        spec = DiseaseSpec(
            name=name,
            display_name=str(body.get("display_name") or name.replace("_", " ").title()),
            category=body.get("category", "other"),
            path=csv_path, has_header=True,
            numeric_features=list(body.get("numeric_features", [])),
            categorical_features=list(body.get("categorical_features", [])),
            target_column=body["target_column"],
            target_rule=body["target_rule"],
            target_meaning={0: tm.get("0") or "disease absent", 1: tm.get("1") or "disease present"},
            target_documented=bool(body.get("target_documented", False)),
            na_values=["?", "NA", "N/A"],
            source=str(body.get("source", "user upload")),
            notes=str(body.get("notes", "")),
            builtin=False,
        )
        ds = load_dataset(spec)               # full validation: columns, binary target, dtypes
        if len(ds.y) < 20:
            raise ValueError(f"only {len(ds.y)} usable rows; need at least 20")
        counts = ds.y.value_counts()
        if counts.min() < 5:
            raise ValueError(f"minority class has only {int(counts.min())} rows; need at least 5")
        register_spec(spec)
    except (KeyError, ValueError, TypeError) as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, f"invalid dataset definition: {e}")

    (folder / "README.md").write_text(
        f"# {spec.display_name}\n\nAdded via platform on {datetime.now(timezone.utc):%Y-%m-%d}.\n"
        f"Source: {spec.source}\n\nTarget column `{spec.target_column}`: "
        f"{spec.to_dict()['target_rule_text']}\n0 = {spec.target_meaning[0]}\n1 = {spec.target_meaning[1]}\n"
        f"Documented: {spec.target_documented}\n\n{spec.notes}\n")
    return {"card": _card(spec), "spec": spec.to_dict(root=PROJECT_ROOT),
            "data": {"n_rows": int(len(ds.y)), "class_counts": {str(k): int(v) for k, v in counts.sort_index().items()},
                     "missing_per_feature": {k: v for k, v in ds.missing_per_feature.items() if v}}}


# ----------------------------------------------------------------- routes: training
_train_state: dict[str, dict] = {}
_train_lock = threading.Lock()


def _train_worker(name: str, qcfg: dict):
    try:
        record = run_benchmark(name, quantum_config=qcfg, verbose=False)
        _train_state[name] = {"state": "done", "finished_at": record["timestamp_utc"], "results": record["results"]}
    except Exception as e:
        _train_state[name] = {"state": "failed", "error": f"{type(e).__name__}: {e}",
                              "traceback": traceback.format_exc()[-2000:]}


@app.post("/api/train/{name}", status_code=202)
def train(name: str, n_qubits: int = 4, reps: int = 1, feature_map: str = "angle", maxiter: int = 150):
    try:
        get_spec(name)
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    with _train_lock:
        if _train_state.get(name, {}).get("state") == "running":
            raise HTTPException(409, "training already running for this disease")
        _train_state[name] = {"state": "running", "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    qcfg = dict(n_qubits=n_qubits, reps=reps, feature_map=feature_map, maxiter=maxiter)
    threading.Thread(target=_train_worker, args=(name, qcfg), daemon=True).start()
    return {"name": name, "state": "running", "quantum_config": qcfg}


@app.get("/api/train/{name}/status")
def train_status(name: str):
    return _train_state.get(name, {"state": "idle"})


# ----------------------------------------------------------------- routes: patient prediction
class PredictRequest(BaseModel):
    data: dict = Field(..., description="feature name -> value, as defined by GET /api/diseases/{name}/form")
    model: str = Field("best", description="'best' | 'all' | a trained model name")
    criterion: str = Field("accuracy", description="how 'best' is chosen: accuracy | f1 | roc_auc")
    explain: bool = True
    prefer_probability: bool = Field(False, description="when choosing 'best', skip models with no probability output")
    patient_name: str | None = Field(None, description="echoed back only; never stored server-side")


@app.get("/api/diseases/{name}/form")
def disease_form(name: str):
    try:
        return build_form_schema(get_spec(name))
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))


@app.get("/api/diseases/{name}/models")
def disease_models(name: str, criterion: str = "accuracy", prefer_probability: bool = False):
    try:
        get_spec(name)
        return predict_service.models_summary(name, criterion, prefer_probability)
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/predict/{name}")
def predict(name: str, req: PredictRequest):
    try:
        result = predict_service.predict_patient(name, req.data, model=req.model, criterion=req.criterion,
                                                 explain=req.explain, prefer_probability=req.prefer_probability)
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    except LookupError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    result["patient_name"] = req.patient_name
    result["analysed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


# ----------------------------------------------------------------- frontend
@app.get("/", include_in_schema=False)
def index():
    # never cache the shell: it pins the versioned asset URLs, so a stale copy would load old CSS/JS
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"})


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
