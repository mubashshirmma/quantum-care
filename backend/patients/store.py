"""Patient records, pending verifications, analyses. Pure data access on SQLite."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from backend.core.db import connect, row_to_dict
from backend.core.security import mask_email, mask_phone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def public_patient(row: dict | None, full: bool = False) -> dict | None:
    """full=False masks contact details (lists/search); full=True for the opened record."""
    if not row:
        return None
    p = dict(row)
    p["email_verified"] = bool(p.get("email_verified_at"))
    if not full:
        p["email"] = mask_email(p["email"])
        p["phone"] = mask_phone(p["phone"])
    return p


class PatientStore:
    # ------------------------------------------------------------ patients
    def next_patient_id(self, conn) -> str:
        year = datetime.now(timezone.utc).year
        key = f"patient_seq_{year}"
        conn.execute("INSERT INTO counters(name, value) VALUES (?, 0) ON CONFLICT(name) DO NOTHING", (key,))
        conn.execute("UPDATE counters SET value = value + 1 WHERE name=?", (key,))
        n = conn.execute("SELECT value FROM counters WHERE name=?", (key,)).fetchone()["value"]
        return f"PAT-{year}-{n:06d}"

    def create_patient(self, name: str, phone: str, email: str, email_verified_at: str | None, created_by: str | None) -> dict:
        with connect() as conn:
            pid = self.next_patient_id(conn)
            conn.execute("INSERT INTO patients(patient_id, name, phone, email, email_verified_at, created_at, created_by, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                         (pid, name.strip(), phone, email, email_verified_at, now_iso(), created_by, now_iso()))
            return row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id=?", (pid,)).fetchone())

    def get(self, patient_id: str) -> dict | None:
        with connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id.strip().upper(),)).fetchone())

    def find_by_email(self, email: str) -> dict | None:
        with connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM patients WHERE email=?", (email,)).fetchone())

    def find_by_phone(self, phone: str) -> dict | None:
        with connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM patients WHERE phone=?", (phone,)).fetchone())

    def export_record(self, patient_id: str) -> dict | None:
        p = self.get(patient_id)
        if not p:
            return None
        return {"exported_at": now_iso(), "format": "quantumcare-patient-record/1",
                "patient": public_patient(p, full=True), "analyses": self.list_analyses(p["patient_id"], with_result=True),
                "disclaimer": "Patient information and model outputs are intended for authorized research and "
                              "decision-support use. This system is not a substitute for professional medical diagnosis."}

    def search(self, q: str, limit: int = 20) -> list[dict]:
        q = (q or "").strip()
        if not q:
            with connect() as conn:
                return [row_to_dict(r) for r in conn.execute("SELECT * FROM patients ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]
        like = f"%{q}%"
        digits = re.sub(r"\D", "", q)
        with connect() as conn:
            rows = conn.execute(
                "SELECT * FROM patients WHERE upper(patient_id) LIKE upper(?) OR name LIKE ? OR email LIKE ? "
                "OR (? != '' AND replace(replace(replace(phone,'-',''),' ',''),'+','') LIKE ?) "
                "ORDER BY CASE WHEN upper(patient_id)=upper(?) THEN 0 ELSE 1 END, created_at DESC LIMIT ?",
                (like, like, like, digits, f"%{digits}%", q, limit)).fetchall()
            return [row_to_dict(r) for r in rows]

    def update_email(self, patient_id: str, email: str, verified_at: str) -> dict | None:
        with connect() as conn:
            conn.execute("UPDATE patients SET email=?, email_verified_at=?, updated_at=? WHERE patient_id=?", (email, verified_at, now_iso(), patient_id))
            return row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone())

    def update_contact(self, patient_id: str, name: str | None = None, phone: str | None = None) -> dict | None:
        with connect() as conn:
            if name:
                conn.execute("UPDATE patients SET name=?, updated_at=? WHERE patient_id=?", (name.strip(), now_iso(), patient_id))
            if phone:
                conn.execute("UPDATE patients SET phone=?, updated_at=? WHERE patient_id=?", (phone, now_iso(), patient_id))
            return row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone())

    def counts(self) -> dict:
        with connect() as conn:
            return {"patients": conn.execute("SELECT count(*) c FROM patients").fetchone()["c"],
                    "analyses": conn.execute("SELECT count(*) c FROM analyses").fetchone()["c"]}

    # ------------------------------------------------------------ pending verifications
    def create_pending(self, **f) -> dict:
        pid = uuid.uuid4().hex
        with connect() as conn:
            conn.execute("INSERT INTO pending_verifications(pending_id, purpose, patient_id, name, phone, email, code_hash, code_expires_at, attempts, sends, last_sent_at, created_at, created_by) "
                         "VALUES (?,?,?,?,?,?,?,?,0,1,?,?,?)",
                         (pid, f["purpose"], f.get("patient_id"), f.get("name"), f.get("phone"), f["email"], f["code_hash"], f["code_expires_at"], now_iso(), now_iso(), f.get("created_by")))
            return row_to_dict(conn.execute("SELECT * FROM pending_verifications WHERE pending_id=?", (pid,)).fetchone())

    def get_pending(self, pending_id: str) -> dict | None:
        with connect() as conn:
            return row_to_dict(conn.execute("SELECT * FROM pending_verifications WHERE pending_id=?", (pending_id,)).fetchone())

    def update_pending(self, pending_id: str, **fields) -> dict | None:
        if not fields:
            return self.get_pending(pending_id)
        cols = ", ".join(f"{k}=?" for k in fields)
        with connect() as conn:
            conn.execute(f"UPDATE pending_verifications SET {cols} WHERE pending_id=?", (*fields.values(), pending_id))
            return row_to_dict(conn.execute("SELECT * FROM pending_verifications WHERE pending_id=?", (pending_id,)).fetchone())

    def delete_pending(self, pending_id: str) -> None:
        with connect() as conn:
            conn.execute("DELETE FROM pending_verifications WHERE pending_id=?", (pending_id,))

    def purge_expired_pending(self, older_than_iso: str) -> None:
        with connect() as conn:
            conn.execute("DELETE FROM pending_verifications WHERE code_expires_at < ?", (older_than_iso,))

    # ------------------------------------------------------------ analyses
    def add_analysis(self, patient_id: str, result: dict, created_by: str | None, input_source: str = "manual") -> dict:
        aid = uuid.uuid4().hex
        pred = result.get("prediction", {})
        with connect() as conn:
            conn.execute("INSERT INTO analyses(analysis_id, patient_id, disease, model, risk_level, probability, prediction, result_json, created_at, created_by, input_source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (aid, patient_id, result.get("disease"), result.get("selected_model"), pred.get("risk_level"), pred.get("probability"), pred.get("label"),
                          json.dumps(result), now_iso(), created_by, input_source))
            return self._analysis(conn.execute("SELECT * FROM analyses WHERE analysis_id=?", (aid,)).fetchone())

    def list_analyses(self, patient_id: str, with_result: bool = False) -> list[dict]:
        with connect() as conn:
            rows = conn.execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY created_at DESC", (patient_id,)).fetchall()
            return [self._analysis(r, with_result) for r in rows]

    def get_analysis(self, analysis_id: str) -> dict | None:
        with connect() as conn:
            r = conn.execute("SELECT a.*, p.name AS patient_name FROM analyses a JOIN patients p ON p.patient_id=a.patient_id WHERE analysis_id=?", (analysis_id,)).fetchone()
            return self._analysis(r, True) if r else None

    def recent_analyses(self, limit: int = 20) -> list[dict]:
        with connect() as conn:
            rows = conn.execute("SELECT a.*, p.name AS patient_name FROM analyses a JOIN patients p ON p.patient_id=a.patient_id ORDER BY a.created_at DESC LIMIT ?", (limit,)).fetchall()
            return [self._analysis(r) for r in rows]

    def delete_analysis(self, analysis_id: str) -> None:
        with connect() as conn:
            conn.execute("DELETE FROM analyses WHERE analysis_id=?", (analysis_id,))

    @staticmethod
    def _analysis(row, with_result: bool = False) -> dict:
        d = dict(row)
        result = json.loads(d.pop("result_json"))
        d["display_name"], d["icon"] = result.get("display_name"), result.get("icon")
        if with_result:
            d["result"] = result
        return d
