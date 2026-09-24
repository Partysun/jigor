# Changelog

## [0.1.0] - 2026-09-XX

First release of **jigor** — a System One decision gateway: `noul`,
`choice` and `score` questions in, typed answers out. One wire protocol
across every backend.

### Highlights

- **One interface, three backends** — the shared `Backend::answers` API over
  local `von` / `laya` ONNX models (via ort, models pulled from the HF Hub)
  and remote `jev` on OpenRouter's Decisions API. Switching models is naming
  them (`von`, `laya`, `jev`).
- **CLI + server** — `jigor serve` (HTTP gateway `/v1/systemone`, `/healthz`),
  `jigor ask` (same wire over stdin), `jigor models`.
- **Unified errors** — every failure is a `jigor::Error` variant
  (model load, wire parsing, provider routing, remote responses); no
  foreign error type leaks out of the library.
- **Examples** — `decide`, `bench`, a 61-question tweet viral-score tester
  and an auto-tagger.

### Distribution

- crates.io: `jigor` (library) and `jigor-cli` (binary) from one
  workspace, published together
- PyPI: `jigor` wheel (maturin) wrapping the same binary as a console script
- npm: `jigor` binary packages for linux x64/arm64, macos x64/arm64,
  windows x64
- Tests: unit suites for the wire protocol and backends, hurl + CLI
  integration suites, examples coverage

### Notes

- `JIGOR_DEVICE=cuda` enables CUDA (fallback to CPU); `OPENROUTER_API_KEY`
  enables the remote backend.
- Local models are cached under `~/.cache/huggingface`.

