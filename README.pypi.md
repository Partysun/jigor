# jigor — System One decision gateway

Ask System One decision models with `noul`, `choice` and `score` questions:
one JSON wire in, typed JSON answers out. `von` and `laya` run locally as
ONNX — no API key; `jev` runs remote on OpenRouter Decisions.

This package installs the `jigor` console script. It is a Rust binary, not a
Python module — there is nothing to `import`; call it from the shell,
`subprocess`, or run it as a service (`jigor serve`).

## Install

```bash
uvx jigor --help        # one-off, no install
uv tool install jigor   # ~/.local/bin/jigor
pip install jigor       # or: pipx install jigor
```

## Quick start

```bash
uvx jigor ask --model von <<'JSON'
{
  "state": { "error": "Disk volume /var/log at 98% capacity." },
  "questions": {
    "requires_intervention": {
      "type": "noul",
      "instructions": "Does this disk space condition require operational intervention?"
    }
  }
}
JSON
{"answers":{"requires_intervention":{"noul":0.7822,"type":"noul"}},"backend":"local","model":"von-1.0.0"}
```

Every question carries `instructions` plus an optional `criteria` (object or
list) that spells out how to judge the state. Answer shapes are the same on
every backend:

- `noul` → `{"type":"noul","noul":0.7822}` — probability of **yes** in `[0, 1]`
- `choice` → `{"type":"choice","choice":"refund","confidence":0.91,"probabilities":{...}}`
- `score` → `{"type":"score","score":1.0,"confidence":0.7,"probabilities":{...},"legend":{...}}`

```json
{
  "state": "Invoice charged twice for order #4471.",
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Classify the root cause domain.",
      "criteria": { "billing": "Invoices, payments, refunds", "infrastructure": "Database, network, hardware" }
    },
    "severity": {
      "type": "score",
      "instructions": "Assess degradation level.",
      "criteria": ["Nominal operation", "Degraded performance", "Critical risk"]
    }
  }
}
```

## From Python: the HTTP gateway

```bash
jigor serve --host 127.0.0.1 --port 8000
```

```python
import requests

r = requests.post("http://127.0.0.1:8000/v1/systemone", json={
    "model": "von-1.0.0",
    "state": {"error": "Disk volume /var/log at 98% capacity."},
    "questions": {
        "requires_intervention": {
            "type": "noul",
            "instructions": "Does this disk space condition require operational intervention?",
        }
    },
})
r.json()["answers"]["requires_intervention"]["noul"]   # 0.7822
```

`GET /healthz` returns `{"status":"ok"}`; `POST /v1/systemone` takes the same
body as `jigor ask`.

## Models

`--model` takes an alias (`von`, `laya`, `jev`, `jev-latest`) or a full id
(`von-1.0.0`, `typesafe/jev-1.13`); `jigor models` lists every provider ×
model pair. The local ONNX models download from Hugging Face to
`~/.cache/huggingface` on first use and are cached afterwards; the remote
`jev` backend needs `OPENROUTER_API_KEY`.

## Platforms

- **macOS** — `macosx_11_0` universal2 wheel (Apple Silicon + Intel) and
  **Windows x86_64** — prebuilt binary, nothing to compile.
- **Linux** — sdist only: `pip` builds the Rust binary at install time
  (needs [rustup](https://rustup.rs), a few minutes; ONNX Runtime prebuilts
  are downloaded during the build). No manylinux wheel yet — the ort prebuilt
  requires glibc ≥ 2.38.

The wheels carry no Python dependencies (`py3-none-<platform>` tags).

Source, Rust crates and npm packages: https://github.com/Partysun/jigor
