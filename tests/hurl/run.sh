#!/usr/bin/env bash
# Integration test runner: starts `jigor serve` on a scratch port, runs the
# hurl suite against it, and tears the server down afterwards.
# Usage: bash tests/hurl/run.sh [extra hurl args...]
set -euo pipefail

cd "$(dirname "$0")/../.."

PORT="${JIGOR_TEST_PORT:-8080}"
BASE_URL="http://localhost:${PORT}"

if ! command -v hurl >/dev/null 2>&1; then
    echo "hurl is required (https://hurl.dev)" >&2
    exit 1
fi
if [ ! -x target/debug/jigor ]; then
    cargo build -p jigor-cli
fi

export JIGOR_DEVICE="${JIGOR_DEVICE:-}"

target/debug/jigor serve --host 127.0.0.1 --port "$PORT" >/tmp/jigor-hurl-server.log 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
    if curl -s --max-time 1 "$BASE_URL/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

timeout 300 hurl --test --jobs 1 --variable BASE_URL="$BASE_URL" tests/hurl/*.hurl "$@"