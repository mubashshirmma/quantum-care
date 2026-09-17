"""SQLite persistence (stdlib). One file: data/app.db (override with APP_DB_PATH)."""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from backend.core.security import DATA_DIR

DB_PATH = Path(os.environ.get("APP_DB_PATH", DATA_DIR / "app.db"))
_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'clinician',
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
  user_agent TEXT);
CREATE TABLE IF NOT EXISTS patients (
  patient_id TEXT PRIMARY KEY, name TEXT NOT NULL, phone TEXT NOT NULL, email TEXT NOT NULL,
  email_verified_at TEXT, created_at TEXT NOT NULL, created_by TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email);
CREATE INDEX IF NOT EXISTS idx_patients_phone ON patients(phone);
CREATE TABLE IF NOT EXISTS pending_verifications (
  pending_id TEXT PRIMARY KEY, purpose TEXT NOT NULL, patient_id TEXT, name TEXT, phone TEXT, email TEXT NOT NULL,
  code_hash TEXT NOT NULL, code_expires_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  sends INTEGER NOT NULL DEFAULT 1, last_sent_at TEXT NOT NULL, created_at TEXT NOT NULL, created_by TEXT);
CREATE TABLE IF NOT EXISTS analyses (
  analysis_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
  disease TEXT NOT NULL, model TEXT NOT NULL, risk_level TEXT, probability REAL, prediction INTEGER,
  result_json TEXT NOT NULL, created_at TEXT NOT NULL, created_by TEXT);
CREATE INDEX IF NOT EXISTS idx_analyses_patient ON analyses(patient_id, created_at);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


@contextmanager
def connect():
    with _lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


MIGRATIONS = [
    # (description, sql)  -- each must be idempotent or guarded below
    ("analyses.input_source", "ALTER TABLE analyses ADD COLUMN input_source TEXT NOT NULL DEFAULT 'manual'"),
    ("patients.phone unique", "CREATE UNIQUE INDEX IF NOT EXISTS uq_patients_phone ON patients(phone)"),
    ("patients.email unique", "CREATE UNIQUE INDEX IF NOT EXISTS uq_patients_email ON patients(email)"),
]


def init_db() -> None:
    import logging
    log = logging.getLogger("qc.db")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(analyses)")}
        for desc, sql in MIGRATIONS:
            if desc == "analyses.input_source" and "input_source" in cols:
                continue
            try:
                conn.execute(sql)
            except sqlite3.IntegrityError as e:      # existing duplicate rows: keep running, warn loudly
                log.error("Migration '%s' skipped: %s. Resolve duplicate patients, then restart.", desc, e)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise


def row_to_dict(row) -> dict | None:
    return dict(row) if row is not None else None
