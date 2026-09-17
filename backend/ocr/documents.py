"""Turn an uploaded medical document (JPG/PNG/PDF) into page images sized for the OCR API."""
from __future__ import annotations

import io

from PIL import Image, ImageOps

SUPPORTED = {"image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"}
MAX_PAGES = 5
TARGET_BYTES = 170_000
MAX_SIDE = 1600


def load_pages(data: bytes, mime: str) -> list[Image.Image]:
    if mime == "application/pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(data)
        pages = []
        for i in range(min(len(pdf), MAX_PAGES)):
            pages.append(pdf[i].render(scale=150 / 72).to_pil().convert("RGB"))
        return pages
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return [img]


def encode_for_api(img: Image.Image) -> tuple[bytes, str]:
    """Downscale + JPEG-compress until under the inline size budget. Returns (bytes, mime)."""
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    for quality in (92, 85, 75, 65, 55, 45):
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= TARGET_BYTES:
            return buf.getvalue(), "image/jpeg"
    # still too big: shrink further
    while buf.tell() > TARGET_BYTES and max(img.size) > 600:
        img = img.resize((int(img.size[0] * 0.8), int(img.size[1] * 0.8)), Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=60, optimize=True)
    return buf.getvalue(), "image/jpeg"


def thumbnail_data_url(img: Image.Image, max_side: int = 700) -> str:
    import base64
    im = img.copy(); im.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
