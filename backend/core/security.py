"""Small, dependency-free security helpers (stdlib only)."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path

from backend.config import PROJECT_ROOT

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
_SECRET_FILE = DATA_DIR / ".secret"


def _load_secret() -> bytes:
    env = os.environ.get("APP_SECRET")
    if env:
        return env.encode()
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_bytes()
    s = secrets.token_bytes(32)
    _SECRET_FILE.write_bytes(s)
    try:
        _SECRET_FILE.chmod(0o600)
    except OSError:
        pass
    return s


SECRET = _load_secret()
_PBKDF2_ITERS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS)
    return f"pbkdf2_sha256${_PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def hash_token(token: str) -> str:
    """Keyed hash so a leaked DB does not reveal usable session tokens / OTP codes."""
    return hmac.new(SECRET, token.encode(), hashlib.sha256).hexdigest()


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def generate_otp(digits: int = 6) -> str:
    return f"{secrets.randbelow(10 ** digits):0{digits}d}"


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")


def normalise_email(email: str) -> str:
    e = (email or "").strip().lower()
    if len(e) > 254 or not _EMAIL_RE.match(e) or ".." in e:
        raise ValueError("Please enter a valid email address (e.g. name@example.com).")
    return e


def normalise_phone(phone: str) -> str:
    digits = re.sub(r"[^\d+]", "", phone or "")
    if len(re.sub(r"\D", "", digits)) < 6:
        raise ValueError("Please enter a valid phone number.")
    return digits


def mask_email(email: str) -> str:
    try:
        local, domain = email.split("@")
        return (local[0] + "•" * max(1, len(local) - 2) + local[-1] if len(local) > 2 else local[0] + "•") + "@" + domain
    except Exception:
        return "•••"


def mask_phone(phone: str) -> str:
    d = re.sub(r"\D", "", phone or "")
    return ("•" * max(0, len(d) - 3) + d[-3:]) if d else ""
