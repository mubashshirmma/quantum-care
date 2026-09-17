"""NVIDIA OCR client (NIM). This is the PRIMARY and ONLY OCR engine in the platform.

Two deployment modes, both the same NVIDIA NIM OCR microservice API:
  hosted   NVIDIA API Catalog endpoint, needs NVIDIA_API_KEY (free key at https://build.nvidia.com)
           default model: nvidia/nemoretriever-ocr-v1   (alt: baidu/paddleocr, set NVIDIA_OCR_MODEL)
  local    self-hosted NIM container, set NVIDIA_OCR_URL, e.g. http://localhost:8010/v1/infer
           requires: NVIDIA GPU + driver, nvidia-container-toolkit, NGC API key to pull the container:
             docker run --gpus all -p 8010:8000 -e NGC_API_KEY nvcr.io/nim/nvidia/nemoretriever-ocr-v1:latest

No other OCR engine is used or substituted. When NVIDIA OCR is not configured the service
reports itself unavailable and the UI offers *manual transcription* (typed text, clearly
labelled as not OCR) so the extraction/mapping pipeline can still be exercised.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field

import httpx

HOSTED_BASE = "https://ai.api.nvidia.com/v1/cv/"
DEFAULT_MODEL = "nvidia/nemoretriever-ocr-v1"
MAX_INLINE_BYTES = 180_000          # hosted endpoints accept ~180 KB inline images; larger need the NVCF asset API


@dataclass
class Detection:
    text: str
    confidence: float | None
    box: list[list[float]] = field(default_factory=list)   # [[x,y],...] normalised or pixel coords as returned


@dataclass
class OCRPage:
    page: int
    detections: list[Detection]
    raw: dict

    @property
    def lines(self) -> list[dict]:
        return [{"text": d.text, "confidence": d.confidence, "box": d.box} for d in self.detections]


class NvidiaOCRError(Exception):
    pass


class NvidiaOCRClient:
    def __init__(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
        self.local_url = os.environ.get("NVIDIA_OCR_URL")
        self.model = os.environ.get("NVIDIA_OCR_MODEL", DEFAULT_MODEL)
        self.timeout = float(os.environ.get("NVIDIA_OCR_TIMEOUT", "60"))

    # ------------------------------------------------------------------ status
    @property
    def mode(self) -> str | None:
        if self.local_url:
            return "local_nim"
        if self.api_key:
            return "hosted"
        return None

    def status(self) -> dict:
        gpu = _gpu_info()
        st = {"engine": "NVIDIA NIM OCR", "model": self.model, "mode": self.mode, "available": self.mode is not None,
              "endpoint": self.local_url or (HOSTED_BASE + self.model), "gpu": gpu,
              "requirements": {
                  "hosted": "set NVIDIA_API_KEY (build.nvidia.com); internet access to ai.api.nvidia.com",
                  "local": "NVIDIA GPU + driver, nvidia-container-toolkit, NGC_API_KEY, run the nemoretriever-ocr NIM container, set NVIDIA_OCR_URL",
              }}
        if not st["available"]:
            st["reason"] = "No NVIDIA_API_KEY and no NVIDIA_OCR_URL configured. NVIDIA OCR is not running; manual transcription is offered instead."
        return st

    # ------------------------------------------------------------------ inference
    def ocr_image(self, image_bytes: bytes, mime: str = "image/png", page: int = 1) -> OCRPage:
        if not self.mode:
            raise NvidiaOCRError(self.status()["reason"])
        if len(image_bytes) > MAX_INLINE_BYTES and self.mode == "hosted":
            raise NvidiaOCRError(f"image is {len(image_bytes)//1024} KB; hosted NVIDIA endpoint accepts ~{MAX_INLINE_BYTES//1000} KB inline "
                                 f"(the document loader should have downscaled it)")
        b64 = base64.b64encode(image_bytes).decode()
        payload = {"input": [{"type": "image_url", "url": f"data:{mime};base64,{b64}"}]}
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        url = self.local_url or (HOSTED_BASE + self.model)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise NvidiaOCRError(f"NVIDIA OCR request failed: {e}") from e
        if r.status_code == 401:
            raise NvidiaOCRError("NVIDIA API rejected the key (401). Check NVIDIA_API_KEY.")
        if r.status_code == 429:
            raise NvidiaOCRError("NVIDIA API rate limit reached (429). Try again shortly.")
        if r.status_code >= 400:
            raise NvidiaOCRError(f"NVIDIA OCR error {r.status_code}: {r.text[:300]}")
        data = r.json()
        return OCRPage(page=page, detections=self._parse(data), raw=data)

    @staticmethod
    def _parse(data: dict) -> list[Detection]:
        """Handles the NIM OCR response shapes: data[].text_detections[].{text_prediction{text,confidence}, bounding_box{points}}."""
        out: list[Detection] = []
        items = data.get("data") or data.get("predictions") or []
        for item in items:
            dets = item.get("text_detections") or item.get("detections") or []
            for d in dets:
                tp = d.get("text_prediction") or {}
                text = tp.get("text") if isinstance(tp, dict) else d.get("text")
                conf = tp.get("confidence") if isinstance(tp, dict) else d.get("confidence")
                pts = (d.get("bounding_box") or {}).get("points") or d.get("box") or []
                box = [[p.get("x"), p.get("y")] if isinstance(p, dict) else list(p) for p in pts]
                if text:
                    out.append(Detection(text=str(text).strip(), confidence=float(conf) if conf is not None else None, box=box))
        # reading order: top-to-bottom, then left-to-right when boxes are available
        def key(d):
            if d.box:
                ys = [p[1] for p in d.box if p[1] is not None]; xs = [p[0] for p in d.box if p[0] is not None]
                return (round(min(ys) if ys else 0, 2), min(xs) if xs else 0)
            return (0, 0)
        out.sort(key=key)
        return out


def _gpu_info() -> dict:
    """Best-effort local GPU discovery (informational only; hosted mode does not need a local GPU)."""
    import shutil
    import subprocess
    for exe in (shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi"):
        if exe and os.path.exists(exe):
            try:
                out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                if out:
                    name, drv, mem = [x.strip() for x in out.splitlines()[0].split(",")]
                    return {"present": True, "name": name, "driver": drv, "memory": mem,
                            "container_runtime": shutil.which("nvidia-container-toolkit") is not None or shutil.which("nvidia-ctk") is not None}
            except Exception:
                pass
    return {"present": False}
