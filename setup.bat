@echo off
REM First-time setup for Windows: virtual env + packages + trained models.
setlocal
cd /d "%~dp0"
echo == QuantumCare setup ==

set PY=
py -3.12 -c "print()" >nul 2>&1 && set PY=py -3.12
if "%PY%"=="" py -3.11 -c "print()" >nul 2>&1 && set PY=py -3.11
if "%PY%"=="" py -3.13 -c "print()" >nul 2>&1 && set PY=py -3.13
if "%PY%"=="" py -3.10 -c "print()" >nul 2>&1 && set PY=py -3.10
if "%PY%"=="" python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set PY=python
if "%PY%"=="" (
  echo ERROR: Python 3.10+ not found. Install from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause & exit /b 1
)
echo Using: %PY%

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv || (echo venv creation failed & pause & exit /b 1)
)
.venv\Scripts\python -m pip install --upgrade pip -q
echo Installing packages (a few minutes the first time)...
.venv\Scripts\pip install -r requirements.txt -q || (echo package install failed & pause & exit /b 1)
.venv\Scripts\python -c "import qiskit, sklearn, xgboost, fastapi; print('Packages OK: qiskit', qiskit.__version__, '| scikit-learn', sklearn.__version__, '| xgboost', xgboost.__version__)"

for %%d in (heart_disease_cleveland diabetes_pima breast_cancer_wisconsin heart_disease_synthetic) do (
  if not exist "saved_models\%%d\metadata.json" (
    echo Training models for %%d ...
    .venv\Scripts\python -W ignore scripts\train.py --disease %%d >nul && echo   done
  ) else (
    echo Models for %%d already present.
  )
)

echo.
echo == Setup complete ==
echo Start the app with start.bat  (browser opens http://localhost:8000, login admin / admin123)
pause
