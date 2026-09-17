# Hybrid Quantum-Classical ML Platform for Early Disease Detection

SIH research prototype. Compares strong classical baselines with a Qiskit
variational quantum classifier on the same biomedical data, same split, same
preprocessing. **Not a medical device. Not clinically validated.**

## Status: Hybrid Quantum Health Intelligence Platform (research prototype)
Heart disease (UCI Cleveland) end to end: CSV in, stratified hold-out split,
leakage-safe preprocessing, 4 classical baselines + 1 quantum classifier,
full metric set, saved models, single-patient prediction.
Plus a web application (early Stages 5-8) with two separated workflows:

* **Patient Analysis (main workflow)** — name a patient, pick a trained disease,
  fill the disease-specific form generated from that model's feature
  configuration, and get an instant prediction from the *saved* models (no
  retraining). Result screen: estimated risk/probability, confidence, per-model
  comparison, "Why this prediction?" (real per-patient contributions for
  Logistic Regression and XGBoost, global importance for Random Forest, an
  explicit "not available" for SVM and the quantum model), local patient history.
* **Dataset / Models (research workspace)** — CSV upload wizard, column and
  target definition, dataset info, training, benchmarking by category.

## Setup
**New computer?** See [SETUP.md](SETUP.md) — or run `./setup.sh` then `./start.sh` (macOS/Linux) or double-click `setup.bat` then `start.bat` (Windows).

```bash
~/.pyenv/versions/3.12.8/bin/python -m venv .venv       # any Python >= 3.10 works
.venv/bin/pip install -r requirements.txt
```
Pinned: qiskit 2.5.2, qiskit-machine-learning 0.9.1, scikit-learn 1.9, xgboost 3.4.

## Run
```bash
# train + benchmark everything for a disease (saves models + experiment record)
.venv/bin/python scripts/train.py --disease heart_disease_cleveland
.venv/bin/python scripts/train.py --disease heart_disease_cleveland --n-qubits 6 --reps 2 --feature-map zz

# predict for one patient from saved artefacts (no retraining)
.venv/bin/python scripts/predict.py --disease heart_disease_cleveland --features      # required inputs
.venv/bin/python scripts/predict.py --disease heart_disease_cleveland --model quantum_vqc \
    --patient '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

# tests
.venv/bin/python -m unittest tests.test_pipeline -v

# web app + API  (open http://localhost:8000 ; API docs at /docs)
.venv/bin/uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
.venv/bin/python -W ignore -m unittest tests.test_api tests.test_registry -v   # API + registry tests
```

## What the platform does (v0.4)
* **Patient Analysis** — new/existing patient → disease → disease-specific form (manual entry) *or*
  **NVIDIA OCR** document intake (JPG/PNG/PDF → NVIDIA NIM OCR → medical information extraction →
  mapping onto the model's feature schema → human verification → confirm) → saved trained model →
  prediction → explanation → **Save Analysis** on the patient record.
* **Patient Records** — search by Patient ID / name / phone / email, patient list, profile with
  analysis timeline, JSON export. One permanent record per patient, many analyses.
* **Models & Datasets** — CSV upload wizard, Dataset Explorer (rows, features, ranges, missing values,
  class distribution), Preprocessing pipeline view, trained model cards.
* **Benchmark** — classical vs quantum on the identical split: table with training/inference time,
  per-metric charts, ROC curves, confusion matrices.
* **Quantum Lab** — the two quantum models (VQC, fidelity-kernel QSVC), their **actual circuits**,
  encodings, ansatz, optimizer, backend and training loss.
* Glassmorphism surfaces + neumorphic controls, light medical aesthetic.

### NVIDIA OCR (primary and only OCR engine)
`backend/ocr/nvidia_client.py` calls the NVIDIA NIM OCR API — hosted (`NVIDIA_API_KEY`, model
`nvidia/nemoretriever-ocr-v1`, or `NVIDIA_OCR_MODEL=baidu/paddleocr`) or a self-hosted NIM container
(`NVIDIA_OCR_URL`; needs NVIDIA GPU + driver, nvidia-container-toolkit and an NGC key). No other OCR
engine is used or substituted. When neither is configured the UI shows the status and offers
**manual transcription**, clearly labelled as *not OCR* and carrying no confidence values.
OCR confidence shown in the UI is the value returned by NVIDIA per text detection; misread
characters (O/0, l/1, S/5) are flagged for review, never silently trusted. Raw OCR output never
reaches a model: the operator confirms every field first (`POST /api/ocr/document/{disease}` →
verify → `POST /api/predict/{disease}`).

## Sign-in, patients and verification
The app is protected by an **operator login** (clinician account). Default dev
credentials are `admin` / `admin123`; set `APP_ADMIN_USER` and
`APP_ADMIN_PASSWORD` before real use (the UI shows a banner until you do).

Patient records live in a SQLite database (`data/app.db`). Risk-based verification:

| Action | Verification |
|---|---|
| Register a **new** patient | phone + email checked for duplicates (existing → "Patient Found", phone→A & email→B → conflict, never merged); then email verified with a 6-digit code before the record is created (10 min, single use, 5 attempts, resend cooldown 60 s, per-email/IP rate limits); unique constraints on patient_id, phone, email |
| Search / open record / view history / analyse / save | **none** beyond the operator session (fast path, Patient ID `PAT-YYYY-NNNNNN`) |
| Change a patient's email | code sent to the new address |

Email delivery: set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`,
`SMTP_FROM` to send real mail. Without `SMTP_HOST` the code is logged on the
server and, in **DEV MODE**, returned to the UI (`DEV_SHOW_OTP`, default on only
when no SMTP is configured). A Patient ID is an identifier, not a credential.

Endpoints: `POST /api/auth/login|logout`, `GET /api/auth/me`;
`GET /api/patients?q=`, `GET /api/patients/{id}`, `POST /api/patients/register`,
`POST /api/patients/register/{pending}/verify|resend`, `PATCH .../email`,
`POST /api/patients/{id}/email` (change), `POST/GET /api/patients/{id}/analyses`,
`GET /api/analyses/recent`.

## Datasets (all documented in datasets/*/README.md)
UCI Cleveland heart disease · Pima Indians diabetes (NIDDK) · Wisconsin breast cancer (sklearn) ·
a 200-row synthetic heart file of unknown provenance (kept as smoke test, target assumed).
Each has 4 classical models (LogReg, RF, SVM-RBF, XGBoost) and 2 quantum models (VQC, QSVC).

## Web app pages
| Page | Purpose |
|---|---|
| Dashboard | primary call-to-action **+ New Patient Analysis**, trained diseases, recent analyses |
| Patient Analysis | find patient by ID/name/phone/email **or** register (email OTP once) → record → **+ New Analysis** → disease → dynamic form → model → **Analyze** → result + explanation → **Save to record** |
| Dataset / Models | CSV upload wizard, dataset info, train & benchmark (secondary, research use) |
| Benchmarking | per-disease metric bars + full tables, quantum config and its capacity caveat |
| Patient History | analyses saved to patient records (server-side SQLite), newest first |
| Settings | "best" criterion, prefer-probability option, verification policy summary, change operator password |

Prediction API: `POST /api/predict/{disease}` with
`{"data": {...features...}, "model": "best"|"all"|"<name>", "criterion": "accuracy", "prefer_probability": false, "patient_name": "optional"}`.
Form config: `GET /api/diseases/{disease}/form`. Trained models: `GET /api/diseases/{disease}/models`.

## Web UI: adding a dataset
1. **Add New Dataset** -> choose a CSV with a header row.
2. The server profiles every column (dtype, distinct values, missing) and
   suggests a role. Set each column to *Feature numeric*, *Feature categorical*,
   *Target*, or *Ignore*. Exactly one Target.
3. Say how the raw target maps to disease present (1): already 0/1, value in a
   ticked list (e.g. `"M"`, or `0` when the file encodes 0 = malignant), or
   greater than a threshold (e.g. UCI `num > 0`). Write what 0 and 1 mean and
   whether that is documented by the source or your assumption.
4. Registering runs the full loader validation (columns exist, target strictly
   binary, >= 20 rows, >= 5 per class). The CSV is copied to
   `datasets/<name>/`, the spec is appended to `datasets/registry.json`, and the
   dataset appears under its category. **Train & benchmark** runs the same
   pipeline as `scripts/train.py` in a background thread.
User-added datasets can be removed from the detail panel; built-ins cannot.

## Pipeline
```
CSV --> load + validate (schema, binary target, missing report)
    --> stratified 80/20 split (seed 42)            test set never touched by any fit()
    --> ColumnTransformer  numeric: median impute + StandardScaler
                           categorical: mode impute + OneHot(drop if binary)
    --+--> Logistic Regression / Random Forest / SVM-RBF / XGBoost
      |
      +--> PCA to n_qubits (only if features > qubits) --> MinMax to [0, pi]
           --> RY(x_i) encoding --> real_amplitudes(reps) ansatz --> <Z...Z>
           --> COBYLA on squared error, exact statevector expectation
    --> accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix
    --> saved_models/<disease>/{model}.joblib + metadata.json
    --> experiments/<disease>_<timestamp>.json
```

## Layout
```
backend/
  config.py                 paths, seed, DISEASE_REGISTRY = built-ins + datasets/registry.json
  api/app.py                FastAPI: /api/diseases(+/form,/models), /api/datasets(+/inspect), /api/train, /api/predict, serves frontend/
  api/auth_routes.py, api/patient_routes.py   operator session; patient registry + OTP registration
  auth/, core/, patients/, notifications/    sessions & passwords (stdlib pbkdf2), SQLite, registration rules, email senders
  prediction/service.py     predict with cached saved pipelines, best-model choice, validation, confidence
  prediction/form_schema.py disease-specific patient form config derived from spec + training data
  explainability/explain.py per-patient contributions where the model supports it (LogReg exact, XGBoost TreeSHAP)
  data/schema.py            DiseaseSpec dataclass, CATEGORIES, serialisable target rules
  data/loader.py            load_dataset, coerce_features (shared with predictor)
  data/splitter.py          stratified_split
  preprocessing/pipeline.py build_preprocessor, build_quantum_feature_pipeline
  models/classical.py       4 baselines wrapped with their own preprocessor clone
  models/quantum.py         QuantumClassifier (sklearn-compatible, Qiskit 2.x)
  evaluation/metrics.py     compute_metrics, comparison table
  training/benchmark.py     run_benchmark: fit all, evaluate, persist
  prediction/predictor.py   Predictor: load + predict one patient
scripts/train.py, scripts/predict.py
frontend/index.html, css/style.css              SPA shell (sidebar nav, pages, modals)
frontend/js/app.js                              router, dashboard, dataset workspace + wizard, benchmarking, settings
frontend/js/patient.js                          patient analysis flow, result/explanation screen, local history store
docs/mindmap.html                               collapsible mind map of the whole project (open in any browser, works offline)
datasets/heart_disease/     data + README documenting target semantics
saved_models/, experiments/ generated
tests/test_pipeline.py
```

## Adding a disease in code
Either use the web UI above, or add one `DiseaseSpec` to `_BUILTIN_SPECS` in
`backend/config.py`: file path, numeric/categorical columns, target column, a
JSON-serialisable `target_rule` (`binary` / `greater_than` / `positive_values`)
and what 0/1 mean. Nothing else changes.

## Known limitations (deliberately not hidden)
* Single hold-out split of 61 test rows: metrics move by several points across
  seeds. Stratified K-fold CV is Stage 2 and must precede any claim.
* Quantum model uses 4 qubits on PCA-reduced features; the classical models see
  all 22 preprocessed features. This is a capacity gap, not a like-for-like test,
  and is one of the things Stage 3/4 will study.
* Quantum VQC scores are (1+<Z>)/2, uncalibrated; QSVC and SVM output a decision margin. The UI
  calls every headline number a **model score**, never a clinical probability. The SVM outputs a decision
  score, not a probability (sklearn 1.9 deprecates `probability=True`); use the
  Settings option or pick another model for a percentage.
* Explanations: exact only for Logistic Regression (log-odds decomposition) and
  XGBoost (TreeSHAP). Random Forest shows global importance. SVM and the quantum
  model show none. None of these is clinically validated.
* Noiseless statevector simulation only; no shot noise, no hardware noise.
* No hyperparameter tuning anywhere yet, so all models are at defaults.
* Synthetic dataset target meaning is undocumented; treated as assumption.
