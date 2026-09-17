"""NVIDIA OCR document intake.

  GET  /api/ocr/status                       engine availability, mode, GPU info, requirements
  POST /api/ocr/extract                      file (jpg/png/pdf) -> NVIDIA OCR -> lines with confidence (+ page thumbnails)
  POST /api/ocr/map/{disease}                lines -> findings -> model feature schema (human verification follows in the UI)
  POST /api/ocr/document/{disease}           extract + map in one call
  POST /api/ocr/transcribe/{disease}         typed text (NOT OCR, labelled manual_transcription) -> findings -> mapping
Raw OCR output never reaches a model: the UI requires the operator to confirm every field first,
then calls the normal POST /api/predict/{disease}.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import get_spec
from backend.ocr.documents import SUPPORTED, encode_for_api, load_pages, thumbnail_data_url
from backend.ocr.extract import extract_findings
from backend.ocr.mapping import map_findings
from backend.ocr.nvidia_client import NvidiaOCRClient, NvidiaOCRError

router = APIRouter(prefix="/api/ocr", tags=["ocr"])
MAX_UPLOAD_MB = 20


class LinesRequest(BaseModel):
    lines: list[dict]
    source: str = "nvidia_ocr"


class TranscribeRequest(BaseModel):
    text: str


def _client() -> NvidiaOCRClient:
    return NvidiaOCRClient()          # re-read env each call so keys can be set without restart


def _spec(disease: str):
    try:
        return get_spec(disease)
    except KeyError as e:
        raise HTTPException(404, e.args[0])


def _run_ocr(file: UploadFile) -> dict:
    mime = (file.content_type or "").lower()
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
    if mime not in SUPPORTED:
        name = (file.filename or "").lower()
        mime = "application/pdf" if name.endswith(".pdf") else "image/jpeg" if name.endswith((".jpg", ".jpeg")) else "image/png" if name.endswith(".png") else mime
        if mime not in SUPPORTED:
            raise HTTPException(415, f"unsupported file type '{file.content_type}'. Use JPG, PNG or PDF.")
    client = _client()
    if not client.mode:
        raise HTTPException(503, {"message": client.status()["reason"], "status": client.status()})
    try:
        pages = load_pages(data, mime)
    except Exception as e:
        raise HTTPException(400, f"could not read document: {e}")
    out_pages, all_lines = [], []
    for i, img in enumerate(pages, start=1):
        img_bytes, img_mime = encode_for_api(img)
        try:
            page = client.ocr_image(img_bytes, img_mime, page=i)
        except NvidiaOCRError as e:
            raise HTTPException(502, str(e))
        lines = page.lines
        all_lines.extend(lines)
        out_pages.append({"page": i, "n_lines": len(lines), "lines": lines, "thumbnail": thumbnail_data_url(img)})
    confs = [l["confidence"] for l in all_lines if l["confidence"] is not None]
    return {"engine": "NVIDIA NIM OCR", "model": client.model, "mode": client.mode, "filename": file.filename,
            "n_pages": len(out_pages), "pages": out_pages, "lines": all_lines,
            "text": "\n".join(l["text"] for l in all_lines),
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None}


@router.get("/status")
def status():
    return _client().status()


@router.post("/extract")
def extract(file: UploadFile = File(...)):
    return _run_ocr(file)


@router.post("/map/{disease}")
def map_lines(disease: str, body: LinesRequest):
    spec = _spec(disease)
    findings = extract_findings(body.lines)
    return {"findings": [f.to_dict() for f in findings], "mapping": map_findings(spec, findings, body.source)}


@router.post("/document/{disease}")
def document(disease: str, file: UploadFile = File(...)):
    spec = _spec(disease)
    ocr = _run_ocr(file)
    findings = extract_findings(ocr["lines"])
    return {"ocr": ocr, "findings": [f.to_dict() for f in findings], "mapping": map_findings(spec, findings, "nvidia_ocr")}


@router.post("/transcribe/{disease}")
def transcribe(disease: str, body: TranscribeRequest):
    """Manual transcription path: the operator types/pastes the report text. This is NOT OCR and
    carries no confidence values; it exists so the extraction + mapping + verification flow works
    when NVIDIA OCR is not configured."""
    spec = _spec(disease)
    lines = [{"text": t, "confidence": None} for t in body.text.splitlines() if t.strip()]
    if not lines:
        raise HTTPException(400, "no text provided")
    findings = extract_findings(lines)
    return {"engine": "manual transcription (not OCR)", "lines": lines, "findings": [f.to_dict() for f in findings],
            "mapping": map_findings(spec, findings, "manual_transcription")}
