"""Patient registry: search / open records (fast, no OTP), register new patients (email OTP),
change email (OTP, sensitive), save & list analyses. All routes need an operator session
(enforced by AuthMiddleware); Patient IDs are identifiers only."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.patients.registration import RegistrationError, RegistrationService
from backend.patients.store import PatientStore, public_patient

router = APIRouter(prefix="/api", tags=["patients"])
store = PatientStore()
reg = RegistrationService(store)


def _user(request: Request) -> str:
    return getattr(request.state, "user", {}).get("username", "?")


def _ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def _wrap(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except RegistrationError as e:
        raise HTTPException(e.status, {"message": e.detail, **e.extra} if e.extra else e.detail)


class RegisterRequest(BaseModel):
    name: str
    phone: str
    email: str


class CodeRequest(BaseModel):
    code: str


class EmailRequest(BaseModel):
    email: str


class ContactUpdate(BaseModel):
    name: str | None = None
    phone: str | None = None


class SaveAnalysis(BaseModel):
    result: dict
    input_source: str = Field("manual", description="manual | nvidia_ocr | manual_transcription")


# ------------------------------------------------------------------ fast routine access (no OTP)
@router.get("/patients")
def search_patients(q: str = "", limit: int = 20):
    return [public_patient(p) for p in store.search(q, min(limit, 50))]


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    p = store.get(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found.")
    return {"patient": public_patient(p, full=True), "analyses": store.list_analyses(p["patient_id"])}


@router.patch("/patients/{patient_id}")
def update_contact(patient_id: str, body: ContactUpdate):
    """Name/phone edits are routine (no OTP). Email changes go through /email (OTP)."""
    p = store.get(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found.")
    from backend.core.security import normalise_phone
    phone = None
    if body.phone:
        try:
            phone = normalise_phone(body.phone)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"patient": public_patient(store.update_contact(p["patient_id"], body.name, phone), full=True)}


@router.post("/patients/{patient_id}/analyses", status_code=201)
def save_analysis(patient_id: str, body: SaveAnalysis, request: Request):
    p = store.get(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found.")
    if "prediction" not in body.result or "disease" not in body.result:
        raise HTTPException(400, "result must be the response of POST /api/predict/{disease}")
    if body.input_source not in ("manual", "nvidia_ocr", "manual_transcription"):
        raise HTTPException(400, "input_source must be manual, nvidia_ocr or manual_transcription")
    return store.add_analysis(p["patient_id"], body.result, _user(request), body.input_source)


@router.get("/patients/{patient_id}/analyses")
def list_analyses(patient_id: str):
    if not store.get(patient_id):
        raise HTTPException(404, "Patient not found.")
    return store.list_analyses(patient_id.upper())


@router.get("/analyses/recent")
def recent_analyses(limit: int = 20):
    return store.recent_analyses(min(limit, 100))


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    a = store.get_analysis(analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found.")
    return a


@router.delete("/analyses/{analysis_id}")
def delete_analysis(analysis_id: str):
    if not store.get_analysis(analysis_id):
        raise HTTPException(404, "Analysis not found.")
    store.delete_analysis(analysis_id)
    return {"deleted": analysis_id}


@router.get("/patients/{patient_id}/export")
def export_record(patient_id: str):
    from fastapi.responses import JSONResponse
    rec = store.export_record(patient_id)
    if not rec:
        raise HTTPException(404, "Patient not found.")
    return JSONResponse(rec, headers={"Content-Disposition": f'attachment; filename="{rec["patient"]["patient_id"]}.json"'})


@router.get("/patients-stats")
def stats():
    return store.counts()


# ------------------------------------------------------------------ new patient: email verification required
@router.post("/patients/register")
def register(body: RegisterRequest, request: Request):
    return _wrap(reg.start_registration, body.name, body.phone, body.email, _user(request), _ip(request))


@router.post("/patients/check")
def check_duplicates(body: RegisterRequest):
    """Duplicate / conflict check without sending any code."""
    from backend.core.security import normalise_email, normalise_phone
    try:
        phone, email = normalise_phone(body.phone), normalise_email(body.email)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return reg.check_duplicates(phone, email) or {"status": "new"}


@router.post("/patients/register/{pending_id}/verify")
def verify(pending_id: str, body: CodeRequest, request: Request):
    return _wrap(reg.verify, pending_id, body.code, _user(request))


@router.post("/patients/register/{pending_id}/resend")
def resend(pending_id: str, request: Request):
    return _wrap(reg.resend, pending_id, _ip(request))


@router.patch("/patients/register/{pending_id}/email")
def correct_email(pending_id: str, body: EmailRequest, request: Request):
    return _wrap(reg.correct_email, pending_id, body.email, _ip(request))


@router.delete("/patients/register/{pending_id}")
def cancel(pending_id: str):
    reg.cancel(pending_id)
    return {"cancelled": pending_id}


# ------------------------------------------------------------------ sensitive: change email on an existing record
@router.post("/patients/{patient_id}/email")
def start_email_change(patient_id: str, body: EmailRequest, request: Request):
    return _wrap(reg.start_email_change, patient_id.upper(), body.email, _user(request), _ip(request))
