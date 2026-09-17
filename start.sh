#!/usr/bin/env bash
# Start QuantumCare (macOS / Linux / WSL) and open the browser.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/uvicorn ]; then echo "Environment missing — run ./setup.sh first."; exit 1; fi
PORT="${PORT:-8000}"
URL="http://localhost:$PORT"
echo "Starting QuantumCare on $URL  (Ctrl+C to stop)"
( sleep 2; if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true; elif command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "$URL" >/dev/null 2>&1 || true; fi ) &
exec .venv/bin/uvicorn backend.api.app:app --host 0.0.0.0 --port "$PORT"
