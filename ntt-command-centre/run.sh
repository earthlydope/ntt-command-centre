#!/usr/bin/env bash
#
# ONE COMMAND, BOTH HALVES.
#
#   ./run.sh              start the semantic layer and the front end, then stop
#                         both together on Ctrl-C
#   ./run.sh verify       run the regression harness and exit
#
# Vite serves the front end on :5178 and proxies /api and /healthz to uvicorn on
# :8808, so the browser only ever makes same-origin requests: no CORS to
# configure and no base URL to get wrong.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${NTT_PYTHON:-/Users/vineet/Desktop/NTT/.venv/bin/python}"
API_PORT="${NTT_API_PORT:-8808}"
WEB_PORT="${NTT_WEB_PORT:-5178}"

cd "$ROOT"

if [[ ! -x "$PY" ]]; then
  echo "python not found at $PY — set NTT_PYTHON to a venv with fastapi, uvicorn," >&2
  echo "pandas, numpy and scikit-learn installed" >&2
  exit 1
fi

if [[ "${1:-}" == "verify" ]]; then
  exec "$PY" -m api.scripts.verify
fi

if lsof -nP -iTCP:"$API_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $API_PORT is already in use — stop that process or set NTT_API_PORT" >&2
  exit 1
fi

echo "▸ semantic layer   http://127.0.0.1:$API_PORT"
echo "  3,034 opportunity lines · 36,631 logged changes · 485 findings · closure model v2"
"$PY" -m uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

cleanup() {
  trap - INT TERM EXIT
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Wait for the layer to answer before Vite opens, so the first paint is never
# the "semantic layer did not answer" panel. The first request trains the
# closure model and builds the movement features, so allow generously.
for _ in $(seq 1 80); do
  if curl -fsS "http://127.0.0.1:$API_PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 0.25
done
curl -fsS "http://127.0.0.1:$API_PORT/healthz" >/dev/null || {
  echo "the semantic layer did not come up — see the uvicorn output above" >&2
  exit 1
}
echo
echo "▸ deal intelligence  http://localhost:$WEB_PORT"
echo

cd "$ROOT/web"
[[ -d node_modules ]] || npm install
npm run dev -- --port "$WEB_PORT"
