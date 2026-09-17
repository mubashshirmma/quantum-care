"""Patient registry, auth and OTP flow tests on a temporary database.
Run:  .venv/bin/python -W ignore -m unittest tests.test_registry -v"""
import os
import re
import tempfile
import unittest
import warnings

warnings.simplefilter("ignore")
_tmp = tempfile.mkdtemp()
os.environ["APP_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["APP_DATA_DIR"] = _tmp

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.app import app  # noqa: E402

PATIENT = {"age": 67, "sex": "1", "cp": "4", "trestbps": 160, "chol": 286, "fbs": "0", "restecg": "2",
           "thalach": 108, "exang": "1", "oldpeak": 1.5, "slope": "2", "ca": 3, "thal": "3"}


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(app)
        cls.c.post("/api/auth/login", json={"username": "admin", "password": "admin123"})

    def _register(self, name, phone, email, **kw):
        r = self.c.post("/api/patients/register", json={"name": name, "phone": phone, "email": email, **kw})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_01_api_requires_session(self):
        anon = TestClient(app)
        self.assertEqual(anon.get("/api/patients").status_code, 401)
        self.assertEqual(anon.get("/api/diseases").status_code, 401)
        self.assertEqual(anon.get("/api/auth/me").json()["authenticated"], False)
        self.assertEqual(anon.post("/api/auth/login", json={"username": "admin", "password": "nope"}).status_code, 401)
        self.assertEqual(anon.get("/").status_code, 200)   # SPA shell is public; it shows the login view

    def test_02_registration_requires_verified_email(self):
        r = self.c.post("/api/patients/register", json={"name": "A B", "phone": "5551234567", "email": "bad"})
        self.assertEqual(r.status_code, 400)
        p = self._register("Alice Test", "+1 555 000 1111", "alice@example.com")
        self.assertEqual(p["status"], "verification_sent"); self.assertIn("dev_code", p)
        self.assertEqual(self.c.get("/api/patients?q=alice").json(), [])          # no record yet
        wrong = self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": "000000"})
        self.assertEqual(wrong.status_code, 400); self.assertEqual(wrong.json()["detail"]["attempts_left"], 4)
        ok = self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": p["dev_code"]}).json()
        self.assertTrue(ok["created"]); self.assertTrue(ok["patient"]["email_verified"])
        self.assertRegex(ok["patient"]["patient_id"], r"^PAT-\d{4}-\d{6}$")
        again = self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": p["dev_code"]})
        self.assertEqual(again.status_code, 404)                                   # single use

    def test_03_duplicate_detection_and_conflict(self):
        r = self.c.post("/api/patients/register", json={"name": "Alice Two", "phone": "5559990000", "email": "alice@example.com"}).json()
        self.assertEqual(r["status"], "exists"); self.assertEqual(r["matched_by"], ["email"]); self.assertIn("•", r["patient"]["email"])
        self.assertIn("phone", r["mismatch"])
        # phone belongs to Alice, email to someone else -> conflict, never merged
        p = self._register("Bob Own", "+1 555 000 2222", "bob@example.com")
        self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": p["dev_code"]})
        r = self.c.post("/api/patients/register", json={"name": "Xavier Conflict", "phone": "+1 555 000 1111", "email": "bob@example.com"}).json()
        self.assertEqual(r["status"], "conflict"); self.assertNotEqual(r["phone_patient"]["patient_id"], r["email_patient"]["patient_id"])
        chk = self.c.post("/api/patients/check", json={"name": "x", "phone": "5550009999", "email": "free@example.com"}).json()
        self.assertEqual(chk["status"], "new")
        # unique index: a second verified record with Bob's phone must be impossible
        p2 = self._register("Bob Clone", "+1 555 000 2222", "clone@example.com")
        self.assertEqual(p2["status"], "exists")

    def test_04_correct_email_and_cooldown(self):
        p = self._register("Carol Typo", "5550002222", "carol@exmaple.com")
        self.assertEqual(self.c.post(f"/api/patients/register/{p['pending_id']}/resend").status_code, 429)
        r = self.c.patch(f"/api/patients/register/{p['pending_id']}/email", json={"email": "carol@example.com"}).json()
        self.assertTrue(r["email_masked"].endswith("@example.com")); self.assertEqual(r["sends_left"], 3)
        ok = self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": r["dev_code"]}).json()
        self.assertEqual(ok["patient"]["email"], "carol@example.com")

    def test_05_attempts_exhausted(self):
        p = self._register("Dan Retry", "5550003333", "dan@example.com")
        codes = [self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": f"{i:06d}"}).status_code for i in range(1, 6)]
        self.assertEqual(codes, [400, 400, 400, 400, 410])
        self.assertEqual(self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": p["dev_code"]}).status_code, 404)

    def test_06_routine_access_is_fast_and_otp_free(self):
        pid = self.c.get("/api/patients?q=alice").json()[0]["patient_id"]
        rec = self.c.get(f"/api/patients/{pid.lower()}").json()
        self.assertEqual(rec["patient"]["email"], "alice@example.com")            # full email on the opened record
        self.assertEqual(len(self.c.get("/api/patients?q=PAT-").json()), 3)
        self.assertEqual(len(self.c.get("/api/patients?q=0001111").json()), 1)  # phone digits
        exp = self.c.get(f"/api/patients/{pid}/export"); self.assertEqual(exp.status_code, 200); self.assertIn("attachment", exp.headers["content-disposition"])
        pred = self.c.post("/api/predict/heart_disease_cleveland", json={"data": PATIENT, "model": "all"}).json()
        r = self.c.post(f"/api/patients/{pid}/analyses", json={"result": pred, "input_source": "manual"}); self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["input_source"], "manual")
        self.assertEqual(self.c.post(f"/api/patients/{pid}/analyses", json={"result": pred, "input_source": "magic"}).status_code, 400)
        self.assertEqual(len(self.c.get(f"/api/patients/{pid}").json()["analyses"]), 1)
        recent = self.c.get("/api/analyses/recent").json(); self.assertEqual(recent[0]["patient_name"], "Alice Test")
        detail = self.c.get(f"/api/analyses/{r.json()['analysis_id']}").json(); self.assertIn("explanation", detail["result"])
        self.assertEqual(self.c.post(f"/api/patients/{pid}/analyses", json={"result": {"foo": 1}}).status_code, 400)

    def test_07_email_change_is_sensitive(self):
        pid = self.c.get("/api/patients?q=alice").json()[0]["patient_id"]
        p = self.c.post(f"/api/patients/{pid}/email", json={"email": "alice.new@example.com"}).json()
        self.assertEqual(p["purpose"], "email_change")
        self.assertEqual(self.c.get(f"/api/patients/{pid}").json()["patient"]["email"], "alice@example.com")
        ok = self.c.post(f"/api/patients/register/{p['pending_id']}/verify", json={"code": p["dev_code"]}).json()
        self.assertFalse(ok["created"]); self.assertEqual(ok["patient"]["email"], "alice.new@example.com")

    def test_08_contact_edit_routine(self):
        pid = self.c.get("/api/patients?q=bob").json()[0]["patient_id"]
        r = self.c.patch(f"/api/patients/{pid}", json={"phone": "+44 20 7946 0000"}).json()
        self.assertIn("7946", r["patient"]["phone"])

    def test_09_logout_locks(self):
        c2 = TestClient(app); c2.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        self.assertEqual(c2.get("/api/patients").status_code, 200)
        c2.post("/api/auth/logout"); self.assertEqual(c2.get("/api/patients").status_code, 401)


if __name__ == "__main__":
    unittest.main()
