"""Registration + email verification rules (risk-based: OTP only for creating a record or
changing its email; never for routine access).

  code: 6 digits, stored as keyed hash, valid OTP_TTL_MIN minutes, single use
  attempts: MAX_ATTEMPTS wrong codes invalidate the pending verification
  sends: RESEND_COOLDOWN_S between sends, MAX_SENDS per pending verification
  rate limits: per email and per client IP over a sliding window (in-memory)
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from backend.core.security import generate_otp, hash_token, mask_email, normalise_email, normalise_phone
from backend.notifications.email import dev_show_otp, get_sender, verification_message
from backend.patients.store import PatientStore, now_iso, public_patient

OTP_TTL_MIN = 10
MAX_ATTEMPTS = 5
MAX_SENDS = 5
RESEND_COOLDOWN_S = 60
RATE_WINDOW_S = 15 * 60
RATE_PER_EMAIL = 6
RATE_PER_IP = 30


class RegistrationError(Exception):
    def __init__(self, status: int, detail: str, **extra):
        super().__init__(detail)
        self.status, self.detail, self.extra = status, detail, extra


class _RateLimiter:
    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str, limit: int):
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > RATE_WINDOW_S:
            q.popleft()
        if len(q) >= limit:
            raise RegistrationError(429, "Too many verification emails requested. Please wait a few minutes and try again.")
        q.append(now)


class RegistrationService:
    def __init__(self, store: PatientStore | None = None):
        self.store = store or PatientStore()
        self.sender = get_sender()
        self.limiter = _RateLimiter()

    # ------------------------------------------------------------------ helpers
    def _issue_code(self, pending: dict, ip: str | None) -> dict:
        self.limiter.check(f"email:{pending['email']}", RATE_PER_EMAIL)
        if ip:
            self.limiter.check(f"ip:{ip}", RATE_PER_IP)
        code = generate_otp()
        expires = datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MIN)
        subject, body = verification_message(code, OTP_TTL_MIN, pending["purpose"])
        self.sender.send(pending["email"], subject, body)
        out = {"expires_at": expires.isoformat(timespec="seconds")}
        if dev_show_otp():
            out["dev_code"] = code            # ONLY when no mail server is configured; UI shows a DEV banner
        return {"code_hash": hash_token(code), "code_expires_at": out["expires_at"], **out}

    def _public_pending(self, p: dict, issued: dict | None = None) -> dict:
        return {"pending_id": p["pending_id"], "purpose": p["purpose"], "email_masked": mask_email(p["email"]),
                "expires_at": p["code_expires_at"], "attempts_left": MAX_ATTEMPTS - p["attempts"],
                "sends_left": MAX_SENDS - p["sends"], "resend_after_s": RESEND_COOLDOWN_S,
                "delivery": self.sender.name, "status": "verification_sent",
                **({"dev_code": issued["dev_code"]} if issued and "dev_code" in issued else {})}

    # ------------------------------------------------------------------ new patient
    def check_duplicates(self, phone: str, email: str) -> dict | None:
        """Phone and email are unique per patient. Returns None when both are free, otherwise a status dict:
        exists   - one patient matches (by phone, email or both)  -> open it / add an analysis
        conflict - phone belongs to patient A and email to patient B -> never merged automatically"""
        by_phone, by_email = self.store.find_by_phone(phone), self.store.find_by_email(email)
        if not by_phone and not by_email:
            return None
        if by_phone and by_email and by_phone["patient_id"] != by_email["patient_id"]:
            return {"status": "conflict",
                    "message": "This phone number belongs to one patient record and this email to a different one. "
                               "They were not merged. Open the correct record, or correct the phone/email.",
                    "phone_patient": public_patient(by_phone), "email_patient": public_patient(by_email)}
        p = by_phone or by_email
        matched_by = [k for k, hit in (("phone", by_phone), ("email", by_email)) if hit]
        mismatch = None
        if len(matched_by) == 1:
            mismatch = f"the {'email' if matched_by == ['phone'] else 'phone'} you entered differs from the one on record"
        return {"status": "exists", "patient": public_patient(p), "matched_by": matched_by, "mismatch": mismatch,
                "message": "Patient found. Open the existing record or add a new analysis to it. A new record was not created."}

    def start_registration(self, name: str, phone: str, email: str, created_by: str, ip: str | None) -> dict:
        name = (name or "").strip()
        if len(name) < 2:
            raise RegistrationError(400, "Please enter the patient's name.")
        try:
            email, phone = normalise_email(email), normalise_phone(phone)
        except ValueError as e:
            raise RegistrationError(400, str(e))
        dup = self.check_duplicates(phone, email)
        if dup:
            return dup
        issued = self._issue_code({"email": email, "purpose": "register"}, ip)
        pending = self.store.create_pending(purpose="register", name=name, phone=phone, email=email,
                                            code_hash=issued["code_hash"], code_expires_at=issued["code_expires_at"], created_by=created_by)
        return self._public_pending(pending, issued)

    def _load_pending(self, pending_id: str) -> dict:
        p = self.store.get_pending(pending_id)
        if not p:
            raise RegistrationError(404, "This verification has expired or was cancelled. Please start again.")
        return p

    def verify(self, pending_id: str, code: str, by: str) -> dict:
        p = self._load_pending(pending_id)
        if p["code_expires_at"] < now_iso():
            self.store.delete_pending(pending_id)
            raise RegistrationError(410, "The verification code has expired. Request a new code.", expired=True)
        if p["attempts"] >= MAX_ATTEMPTS:
            self.store.delete_pending(pending_id)
            raise RegistrationError(410, "Too many incorrect attempts. Please start the verification again.", expired=True)
        code = (code or "").strip()
        if hash_token(code) != p["code_hash"]:
            p = self.store.update_pending(pending_id, attempts=p["attempts"] + 1)
            left = MAX_ATTEMPTS - p["attempts"]
            if left <= 0:
                self.store.delete_pending(pending_id)
                raise RegistrationError(410, "Too many incorrect attempts. Please start the verification again.", expired=True)
            raise RegistrationError(400, f"Incorrect code. {left} attempt{'s' if left != 1 else ''} left.", attempts_left=left)
        # success: single use
        self.store.delete_pending(pending_id)
        verified_at = now_iso()
        if p["purpose"] == "register":
            dup = self.check_duplicates(p["phone"], p["email"])      # someone may have registered meanwhile
            if dup:
                raise RegistrationError(409, dup["message"], **{k: v for k, v in dup.items() if k != "message"})
            try:
                patient = self.store.create_patient(p["name"], p["phone"], p["email"], verified_at, by)
            except Exception as e:                                   # unique index is the last line of defence
                raise RegistrationError(409, f"A patient with this phone or email already exists ({type(e).__name__}).")
            return {"status": "verified", "created": True, "patient": public_patient(patient, full=True)}
        if p["purpose"] == "email_change":
            patient = self.store.update_email(p["patient_id"], p["email"], verified_at)
            return {"status": "verified", "created": False, "patient": public_patient(patient, full=True)}
        raise RegistrationError(400, "unknown verification purpose")

    def resend(self, pending_id: str, ip: str | None) -> dict:
        p = self._load_pending(pending_id)
        if p["sends"] >= MAX_SENDS:
            raise RegistrationError(429, "Maximum number of codes sent for this registration. Please start again later.")
        last = datetime.fromisoformat(p["last_sent_at"])
        wait = RESEND_COOLDOWN_S - (datetime.now(timezone.utc) - last).total_seconds()
        if wait > 0:
            raise RegistrationError(429, f"Please wait {int(wait) + 1}s before requesting another code.", retry_after_s=int(wait) + 1)
        issued = self._issue_code(p, ip)
        p = self.store.update_pending(pending_id, code_hash=issued["code_hash"], code_expires_at=issued["code_expires_at"],
                                      attempts=0, sends=p["sends"] + 1, last_sent_at=now_iso())
        return self._public_pending(p, issued)

    def correct_email(self, pending_id: str, new_email: str, ip: str | None) -> dict:
        """User typed the wrong address: replace it and send a fresh code (counts as a send)."""
        p = self._load_pending(pending_id)
        try:
            new_email = normalise_email(new_email)
        except ValueError as e:
            raise RegistrationError(400, str(e))
        if p["sends"] >= MAX_SENDS:
            raise RegistrationError(429, "Maximum number of codes sent for this registration. Please start again later.")
        issued = self._issue_code({**p, "email": new_email}, ip)
        p = self.store.update_pending(pending_id, email=new_email, code_hash=issued["code_hash"], code_expires_at=issued["code_expires_at"],
                                      attempts=0, sends=p["sends"] + 1, last_sent_at=now_iso())
        return self._public_pending(p, issued)

    def cancel(self, pending_id: str) -> None:
        self.store.delete_pending(pending_id)

    # ------------------------------------------------------------------ sensitive change: email on an existing record
    def start_email_change(self, patient_id: str, new_email: str, by: str, ip: str | None) -> dict:
        patient = self.store.get(patient_id)
        if not patient:
            raise RegistrationError(404, "Patient not found.")
        try:
            new_email = normalise_email(new_email)
        except ValueError as e:
            raise RegistrationError(400, str(e))
        if new_email == patient["email"] and patient["email_verified_at"]:
            raise RegistrationError(400, "That is already the verified email on this record.")
        other = self.store.find_by_email(new_email)
        if other and other["patient_id"] != patient["patient_id"]:
            raise RegistrationError(409, "That email is already used by another patient record.")
        issued = self._issue_code({"email": new_email, "purpose": "email_change"}, ip)
        pending = self.store.create_pending(purpose="email_change", patient_id=patient["patient_id"], email=new_email,
                                            code_hash=issued["code_hash"], code_expires_at=issued["code_expires_at"], created_by=by)
        return self._public_pending(pending, issued)
