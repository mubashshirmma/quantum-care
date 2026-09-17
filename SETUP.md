# QuantumCare — first-time setup on a new computer

Works on **macOS (Apple Silicon or Intel)**, **Windows 10/11**, and **Linux / WSL**.
You need Python 3.10 or newer and an internet connection for the first install (about 600 MB of packages).

The `.venv` folder from another computer does **not** work here — always run the setup script once.

---

## macOS (e.g. MacBook Air M3)

1. Open **Terminal** (Cmd + Space, type "Terminal").
2. Install Homebrew if you don't have it (one line, follow its prompts):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Install Python and the OpenMP runtime XGBoost needs:
   ```bash
   brew install python@3.12 libomp
   ```
4. Go to the project folder and run the setup script:
   ```bash
   cd ~/Downloads/quantom_model        # wherever you copied it
   chmod +x setup.sh start.sh
   ./setup.sh
   ```
   This creates `.venv`, installs all packages, and trains any disease whose models are missing (≈ 2 min).
5. Start the app:
   ```bash
   ./start.sh
   ```
   Your browser opens **http://localhost:8000**. Sign in with `admin` / `admin123` (change it in Settings).
6. To stop: press **Ctrl + C** in Terminal. Next time you only need `./start.sh`.

If macOS asks "allow incoming network connections?", click Allow.

---

## Windows 10 / 11 (no WSL needed)

1. Install Python from https://www.python.org/downloads/ (3.12 recommended).
   **Tick "Add python.exe to PATH"** during installation.
2. Open the project folder in **File Explorer**.
3. Double-click **`setup.bat`**. A window shows progress; it creates `.venv`, installs packages and
   trains missing models (≈ 3 min). Press any key when it says it is finished.
4. Double-click **`start.bat`**. Your browser opens **http://localhost:8000**.
   Sign in with `admin` / `admin123`.
5. To stop: close the black server window or press **Ctrl + C** in it. Next time just run `start.bat`.

If Windows Defender/Firewall asks, choose "Allow" (private networks). If `python` is not found,
reinstall Python with the PATH box ticked, or run the commands from **PowerShell** manually:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

---

## Linux / WSL (Ubuntu)

```bash
sudo apt install -y python3.12 python3.12-venv     # or python3-venv
cd /path/to/quantom_model
chmod +x setup.sh start.sh
./setup.sh
./start.sh
```
Inside WSL, open http://localhost:8000 from the Windows browser.

---

## What to copy from the old computer

| Item | Needed? |
|---|---|
| everything except `.venv/` | yes |
| `saved_models/**/*.joblib` | optional — copy to skip retraining; otherwise `setup` retrains |
| `data/` (patients DB, operator password, `.secret`) | optional — copy to keep the same patients and login; omit for a clean start |

## Optional configuration (environment variables, set before `start`)

| Variable | Purpose |
|---|---|
| `APP_ADMIN_USER`, `APP_ADMIN_PASSWORD` | operator login (default admin / admin123, banner shown until set) |
| `NVIDIA_API_KEY` | enables NVIDIA OCR (hosted NIM) — free key at https://build.nvidia.com |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | send real verification emails (otherwise DEV MODE shows the code on screen) |
| `PORT` | change the port used by `start.sh` / `start.bat` (default 8000) |

Example (macOS/Linux): `APP_ADMIN_PASSWORD=MyStrongPass NVIDIA_API_KEY=nvapi-... ./start.sh`
Example (Windows PowerShell): `$env:APP_ADMIN_PASSWORD="MyStrongPass"; .\start.bat`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python3: command not found` / `python is not recognized` | install Python (see above); on Windows tick "Add to PATH" |
| `Address already in use` | a server is still running — open the browser, or stop it (`fuser -k 8000/tcp` on mac/linux, close the window on Windows) |
| XGBoost import error on macOS (`libomp.dylib`) | `brew install libomp` |
| Page looks old / broken | press Ctrl + Shift + R once (cache) |
| "No trained models" for a disease | `.venv/bin/python scripts/train.py --disease <name>` (mac/linux) or `.venv\Scripts\python scripts\train.py --disease <name>` (Windows) |
| Login fails on a copied `data/` folder | the password travelled with it — use the one you set there, or delete `data/app.db` for a fresh admin/admin123 |

Research prototype: patient information and model outputs are intended for authorized research and
decision-support use. This system is not a substitute for professional medical diagnosis.
