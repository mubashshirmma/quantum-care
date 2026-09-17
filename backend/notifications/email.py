"""Email delivery abstraction.

  SMTP configured (SMTP_HOST set)  -> SMTPEmailSender
  otherwise                        -> ConsoleEmailSender (prints the message to the server log)

DEV_SHOW_OTP: when true the API also returns the verification code so the flow can be
demonstrated without a mail server. Defaults to true ONLY when no SMTP_HOST is set.
The UI shows a prominent DEV MODE banner whenever a code is returned this way.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger("qc.email")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


class EmailSender:
    name = "base"

    def send(self, to: str, subject: str, body: str) -> None:
        raise NotImplementedError


class ConsoleEmailSender(EmailSender):
    name = "console"

    def send(self, to: str, subject: str, body: str) -> None:
        log.info("\n---- EMAIL (not sent, no SMTP configured) ----\nTo: %s\nSubject: %s\n\n%s\n----------------------------------------------", to, subject, body)


class SMTPEmailSender(EmailSender):
    name = "smtp"

    def __init__(self):
        self.host = os.environ["SMTP_HOST"]
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.user = os.environ.get("SMTP_USER")
        self.password = os.environ.get("SMTP_PASSWORD")
        self.sender = os.environ.get("SMTP_FROM", self.user or "noreply@localhost")
        self.use_tls = os.environ.get("SMTP_TLS", "true").lower() != "false"

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        with smtplib.SMTP(self.host, self.port, timeout=15) as s:
            if self.use_tls:
                s.starttls()
            if self.user:
                s.login(self.user, self.password or "")
            s.send_message(msg)


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


def get_sender() -> EmailSender:
    return SMTPEmailSender() if smtp_configured() else ConsoleEmailSender()


def dev_show_otp() -> bool:
    v = os.environ.get("DEV_SHOW_OTP")
    if v is None:
        return not smtp_configured()
    return v.lower() in ("1", "true", "yes")


def verification_message(code: str, minutes: int, purpose: str) -> tuple[str, str]:
    what = "confirm this email for a new patient record" if purpose == "register" else "confirm the new email address on your patient record"
    return ("Your QuantumCare verification code",
            f"Your verification code is: {code}\n\nUse it to {what}. It expires in {minutes} minutes and can be used once.\n"
            f"If you did not expect this message, you can ignore it.\n")
