# @zatsepin/jigor — System One decision gateway

Installs the `jigor` binary: a CLI and an HTTP gateway over one System One
wire — `noul`/`choice`/`score` questions in, typed JSON answers out. `von`
and `laya` run locally as ONNX (no API key); `jev` runs remote on OpenRouter
Decisions.

Node ≥ 18 is needed once, at install time, to pick the platform package —
the installed `jigor` is a standalone binary with no Node runtime.

## Install

```bash
npm install --global @zatsepin/jigor
jigor --help
```

One-off, no install:

```bash
npx @zatsepin/jigor models
```

The binary comes from an optional dependency matched to your OS and CPU —
`@zatsepin/jigor-linux-x64-gnu`, `@zatsepin/jigor-linux-arm64-gnu`,
`@zatsepin/jigor-darwin` (universal2: Apple Silicon + Intel) or
`@zatsepin/jigor-win32-x64-msvc` — installed automatically, never a
source build.

## Quick start

```bash
jigor ask --model von <<'JSON'
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

## HTTP gateway

```bash
jigor serve --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "von-1.0.0",
    "state": { "error": "Disk volume /var/log at 98% capacity." },
    "questions": {
      "requires_intervention": {
        "type": "noul",
        "instructions": "Does this disk space condition require operational intervention?"
      }
    }
  }'
# {"answers":{"requires_intervention":{"noul":0.7822,"type":"noul"}},"backend":"local","model":"von-1.0.0"}
```

`GET /healthz` returns `{"status":"ok"}`; `POST /v1/systemone` takes the same
body as `jigor ask`.

## Models

`--model` takes an alias (`von`, `laya`, `jev`, `jev-latest`) or a full id
(`von-1.0.0`, `typesafe/jev-1.13`); `jigor models` lists every provider ×
model pair. The local ONNX models download from Hugging Face to
`~/.cache/huggingface` on first use and are cached afterwards; the remote
`jev` backend needs `OPENROUTER_API_KEY`.

Source, Rust crates and the pip package: https://github.com/Partysun/jigor
