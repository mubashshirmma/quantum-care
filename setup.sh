#!/usr/bin/env bash
# First-time setup for macOS / Linux / WSL: virtual env + packages + trained models.
set -euo pipefail
cd "$(dirname "$0")"
echo "== QuantumCare setup =="

# 1) find a suitable Python (3.10+)
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: Python 3.10+ not found."
  echo "  macOS : brew install python@3.12 libomp"
  echo "  Ubuntu: sudo apt install -y python3.12 python3.12-venv"
  exit 1
fi
echo "Using $($PY --version) at $(command -v $PY)"

# 2) macOS: XGBoost needs the OpenMP runtime
if [ "$(uname)" = "Darwin" ]; then
  if ! [ -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && ! [ -e /usr/local/opt/libomp/lib/libomp.dylib ]; then
    if command -v brew >/dev/null 2>&1; then echo "Installing libomp (needed by XGBoost on macOS)..."; brew install libomp; 
    else echo "WARNING: Homebrew not found; XGBoost needs libomp. Install Homebrew then: brew install libomp"; fi
  fi
fi

# 3) virtual environment + packages
if [ ! -x .venv/bin/python ]; then echo "Creating virtual environment..."; rm -rf .venv; "$PY" -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip -q
echo "Installing packages (this can take a few minutes the first time)..."
.venv/bin/pip install -r requirements.txt -q
.venv/bin/python -c "import qiskit, sklearn, xgboost, fastapi; print('Packages OK: qiskit', qiskit.__version__, '| scikit-learn', sklearn.__version__, '| xgboost', xgboost.__version__)"

# 4) train any disease that has no saved models
for d in heart_disease_cleveland diabetes_pima breast_cancer_wisconsin heart_disease_synthetic; do
  if [ ! -f "saved_models/$d/metadata.json" ]; then
    echo "Training models for $d ..."; .venv/bin/python -W ignore scripts/train.py --disease "$d" >/dev/null && echo "  done"
  else echo "Models for $d already present."; fi
done

echo
echo "== Setup complete =="
echo "Start the app with:  ./start.sh   (then open http://localhost:8000, login admin / admin123)"
