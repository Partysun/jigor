# jigor-cli — the `jigor` binary (CLI + HTTP gateway)

Ask System One decision models from the terminal or over HTTP. `von` and
`laya` run locally as ONNX; `jev` runs remote on OpenRouter Decisions — one
wire protocol across all of them: `noul`/`choice`/`score` questions in,
typed answers out.

```bash
cargo install jigor-cli      # or: npm install --global jigor  /  pip install jigor
jigor serve --host 127.0.0.1 --port 8000   # HTTP gateway: POST /v1/systemone, GET /healthz
jigor ask [--model <id>] [--provider <p>]  # the same wire over stdin, no HTTP layer
jigor models                               # provider x model pairs
```

Model aliases: `von`, `laya`, `jev`, `jev-latest`. The local ONNX models
download to `~/.cache/huggingface` on first use; set `OPENROUTER_API_KEY`
for the remote `jev` backend.

## Example

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
# {"model":"von-1.0.0","backend":"local","answers":{"requires_intervention":{"type":"noul","noul":0.2739}}}
```

## The jigor Rust library

This binary is a thin shell over the
[`jigor`](https://crates.io/crates/jigor) crate — local ONNX backends in
`von`/`laya`, remote `jev` via OpenRouter Decisions, and one `answers()`
call for all three. Use the library directly when you want decisions inside
a Rust program: [crate docs](https://docs.rs/jigor) ·
[full project README](https://github.com/Partysun/jigor#readme).