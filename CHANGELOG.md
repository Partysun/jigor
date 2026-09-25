# Changelog

## [0.1.3] - 2026-09-25

Ask `noul` on laya the way the checkpoint needs it — a neutral-key
two-option choice instead of the `false:`/`true:` label pair
([laya#156](https://github.com/NandhaKishorM/laya/issues/156)) — and
validate `noul` criteria once on the wire.

### Fixes

- laya `noul` now rides the choice path: neutral `A`/`B` keys with the
  yes/no wording as descriptions, the `choice question:` head and the
  `choice:2` calibration bucket. Option A is yes, so `p[0]` is the
  probability of yes; a `criteria` side wins when given, otherwise the
  wording echoes the instructions. Positive input no longer comes back
  0.0 (review probe: 0.0 before, 0.94 after; 9 of 10 labelled cases
  correct, was 7)
- laya serializes non-string state with upstream `json.dumps` semantics
  (it was pretty-printed) and truncates the instruction head to
  `max(8, opt_budget)` instead of `opt_budget + 8`
- `noul` criteria are validated where the payload enters: keys must be
  `true`/`false` (case-insensitive, canonicalized), anything else is a
  payload error instead of a silently dropped side
- von and laya render non-string `noul` criteria values as text instead
  of dropping them

## [0.1.2] - 2026-09-25

Fix the ONNX Runtime API mismatch that panicked every macOS build at
startup, and clarify the install options.

### Fixes

- request ORT API 24 instead of ort's default 27: macOS binaries
  statically link the source-built ONNX Runtime v1.24.2 (Microsoft
  ships no Intel mac prebuilts past 1.24), whose runtime serves only
  API 1..24; the linux/windows 1.28 prebuilts accept 24 as well
- document `cargo install` supported targets (linux x64/arm64, macOS
  Apple Silicon, windows x64 — Intel macs use uvx/pip/npm) and that
  the installed command is `jigor`

## [0.1.1] - 2026-09-25

Ship the dedicated CLI README (`crates/jigor-cli/README.md`) on every
registry — crates.io, PyPI, and the npm meta package — plus release
pipeline fixes.

### Fixes

- npm meta package now copies the canonical README at publish time
  instead of committing a duplicate `npm/README.md`
- linux/arm64 npm build: `CXX_aarch64-unknown-linux-gnu` set via `env`
  (shells reject `-` in `export` names) and the binary is copied from
  `target/aarch64-unknown-linux-gnu/release/`
- idempotent npm and PyPI publishing so pipeline re-runs skip versions
  that already exist instead of failing

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
- npm: `jigor` binary packages for linux x64/arm64, macos (universal2 —
  Intel and Apple Silicon in one binary), windows x64
- Tests: unit suites for the wire protocol and backends, hurl + CLI
  integration suites, examples coverage

### Notes

- `JIGOR_DEVICE=cuda` enables CUDA (fallback to CPU); `OPENROUTER_API_KEY`
  enables the remote backend.
- Local models are cached under `~/.cache/huggingface`.

