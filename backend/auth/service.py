"""Operator (clinician) accounts + session cookies. Minimal but real: hashed passwords,
random session tokens stored hashed, expiry. Patient IDs are identifiers, never credentials."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from backend.core.db import connect, row_to_dict
from backend.core.security import generate_token, hash_password, hash_token, verify_password

SESSION_HOURS = int(os.environ.get("APP_SESSION_HOURS", "12"))
COOKIE_NAME = "qc_session"
DEFAULT_USER, DEFAULT_PASSWORD = "admin", "admin123"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def using_default_credentials() -> bool:
    return not os.environ.get("APP_ADMIN_PASSWORD")


def ensure_default_user() -> None:
    username = os.environ.get("APP_ADMIN_USER", DEFAULT_USER)
    password = os.environ.get("APP_ADMIN_PASSWORD", DEFAULT_PASSWORD)
    with connect() as conn:
        row = conn.execute("SELECT username FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO users(username, password_hash, role, created_at) VALUES (?,?,?,?)",
                         (username, hash_password(password), "admin", _iso(_now())))
        elif os.environ.get("APP_ADMIN_PASSWORD"):
            conn.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password(password), username))
    if using_default_credentials():
        import logging
        logging.getLogger("qc.auth").warning("Using DEFAULT operator credentials (%s / %s). Set APP_ADMIN_USER / APP_ADMIN_PASSWORD.", username, DEFAULT_PASSWORD)


def login(username: str, password: str, user_agent: str | None = None) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        token = generate_token()
        now = _now()
        conn.execute("INSERT INTO sessions(token_hash, username, created_at, expires_at, user_agent) VALUES (?,?,?,?,?)",
                     (hash_token(token), row["username"], _iso(now), _iso(now + timedelta(hours=SESSION_HOURS)), (user_agent or "")[:200]))
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_iso(now),))
        return token


def user_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    with connect() as conn:
        row = conn.execute("SELECT s.username, s.expires_at, u.role FROM sessions s JOIN users u ON u.username=s.username WHERE s.token_hash=?",
                           (hash_token(token),)).fetchone()
        if row is None or row["expires_at"] < _iso(_now()):
            return None
        return {"username": row["username"], "role": row["role"], "session_expires_at": row["expires_at"]}


def logout(token: str | None) -> None:
    if token:
        with connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (hash_token(token),))


def change_password(username: str, old: str, new: str) -> bool:
    if len(new) < 8:
        raise ValueError("new password must be at least 8 characters")
    with connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
        if row is None or not verify_password(old, row["password_hash"]):
            return False
        conn.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password(new), username))
        conn.execute("DELETE FROM sessions WHERE username=?", (username,))
        return True
