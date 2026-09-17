@echo off
REM Start QuantumCare on Windows and open the browser.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\uvicorn.exe" (
  echo Environment missing - run setup.bat first.
  pause & exit /b 1
)
if "%PORT%"=="" set PORT=8000
echo Starting QuantumCare on http://localhost:%PORT%   (close this window or press Ctrl+C to stop)
start "" /b cmd /c "timeout /t 2 >nul & start http://localhost:%PORT%"
.venv\Scripts\uvicorn backend.api.app:app --host 0.0.0.0 --port %PORT%
pause
