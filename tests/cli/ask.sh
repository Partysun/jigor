#!/usr/bin/env bash
# jigor ask CLI tests: wire JSON in via stdin, wire JSON out.
# Reuses crates/jigor/tests/fixtures/*.json; requires the local ONNX model for the local
# backend (same assumption as the hurl suite).
# Usage: bash tests/cli/ask.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ ! -x target/debug/jigor ]; then
    cargo build -p jigor-cli
fi

JIGOR=target/debug/jigor
REQ=crates/jigor/tests/fixtures/local_noul_request.json

### jigor models lists every provider x model pair
OUT=$("$JIGOR" models)
grep -q "local" <<<"$OUT"
grep -q "openrouter" <<<"$OUT"
grep -q "typesafe/jev-1.13" <<<"$OUT"

### local ask via stdin returns the typed wire shape
OUT=$("$JIGOR" ask <"$REQ")
echo "$OUT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['model'] == 'von-1.0.0', d
assert d['backend'] == 'local', d
a = d['answers']['requires_intervention']
assert a['type'] == 'noul', a
assert 0 < a['noul'] < 1, a
"

### unknown provider is rejected with a hint
if OUT=$("$JIGOR" ask --provider bogus --model jev <"$REQ" 2>&1 || true) && grep -q "jigor models" <<<"$OUT"; then
    :
else
    echo "expected \\"jigor models\\" hint for unknown provider, got: $OUT" >&2
    exit 1
fi

### remote backend: without a key, a clear error; with a key, a live ask
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OUT=$("$JIGOR" ask --model jev <"$REQ" 2>&1 || true)
    if ! grep -q "OPENROUTER_API_KEY" <<<"$OUT"; then
        echo "expected OPENROUTER_API_KEY error, got: $OUT" >&2
        exit 1
    fi
else
    OUT=$("$JIGOR" ask --model jev <"crates/jigor/tests/fixtures/jev_tutorial_request.json")
    echo "$OUT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['backend'] == 'openrouter', d
assert d['model'] == 'typesafe/jev-1.13', d
assert d['answers']['is_bug']['type'] == 'noul', d
u = d['usage']
assert u['input_tokens'] > 0, u
assert u['output_tokens'] > 0, u
assert u['cost'] >= 0, u
"
fi

echo "cli tests ok"