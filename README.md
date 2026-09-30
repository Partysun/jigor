# jigor — System One decision gateway (Rust lib)

This is a zero-shot classifier models gateway or runner.

Or, if you prefer marketing terms, "System one decision" models.

This is written in Rust (blazing fast as it should be), one wire protocol across
backends: `von` and `laya` run locally as ONNX (via `ort`); `jev` runs remotely on
OpenRouter's Decisions API.

## Models

- HF `Zatsepin/von-onnx-fp16` (`model.onnx` 794M, `tokenizer/tokenizer.json` 3.5M)
- HF `Mattepiu/laya-onnx` (`laya.onnx` fp32 — matches the python reference
  bit-for-bit; `int8/laya_int8.onnx` via `LAYA_ONNX_FILE`)
- HF `Zatsepin/jeff-qwen3.5-0.8b-onnx` (`model.onnx` fp32 — Jeff, the
  zero-shot decision model from [firelex/jeff](https://github.com/firelex/jeff):
  Qwen3.5-0.8B + readout head. Static sequence length 512; left-padded
  requests, longer prompts error out.)
- OpenRouter Decisions (`typesafe/jev-1.13`, `jaredpalmer/kev-4b`)

## Installation

You can install it via npm, pip, or cargo. Is it enough? If not -> [subscribe](https://zatsepin.dev/subscribe)

```bash
npm install --global @zatsepin/jigor
# `pnpm approve-builds` may be needed
pip install jigor
cargo install jigor-cli --locked
cargo bininstall jigor-cli
```

## Usage as cli

You can use ephemeral execution with npx or uvx.

```bash
npx @zatsepin/jigor ask --model von <<'JSON'
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

If you have OPENROUTER_API_KEY in your env, you can easily run this from your terminal and get a quick response.

```bash
npx @zatsepin/jigor ask --model jev <<'JSON'
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
{"answers":{"requires_intervention":{"noul":0.92,"type":"noul"}},"backend":"openrouter","model":"typesafe/jev-1.13","usage":{"cost":0.000012306,"input_tokens":293,"output_tokens":24}}
```

Feel free to use it with your agents. This README.md is enough to teach them how to use it.

## Usage as lib

One dependency, one entry per backend — `noul`/`choice`/`score` questions
in, typed answers out. Identical on von, laya and any OpenRouter model.

```toml
[dependencies]
jigor = { path = "../jigor" }                             # local checkout
# jigor = { git = "https://github.com/Partysun/jigor" }   # from GitHub
# jigor = "0.1.5"                                         # once published
serde_json = { version = "1.0" }                          # Value, json!
```

```rust
use jigor::{Answer, Question, Backend, Result, VonBackend, choice, noul, score};
use serde_json::json;

fn main() -> Result<()> {
    let mut von = VonBackend::new()?;   // VonBackend::new, LayaBackend::new,
                                        // OpenRouterBackend::for_model(...):
                                        // the same answers() call on all three
    let questions = vec![
        noul("churn", "Is the customer likely to churn?"),
        choice(
            "want",
            "What does the customer want?",
            &["refund", "order status", "technical help"],
        ),
        score("urgency", "How urgent is this?", &["calm", "annoyed", "angry"]),
    ];
    let asks = von.answers(
        &json!("Customer: I was charged twice for order #4471."),
        &questions,
        None,
    )?;

    match asks.get("churn").unwrap() {
        Answer::Noul { probability } => println!("churn: {:.4}", *probability),
        _ => {},
    }
    Ok(())
}
```

Only have a model id? `jigor::ask` resolves aliases and the provider for you:

```rust
let asks = jigor::ask("jev", &state, &questions, None)?;  // OpenRouter jev
let asks = jigor::ask("laya", &state, &questions, None)?; // local laya
// Asks { model, backend, answers, usage }
// usage: tokens + billed cost (USD) on remote asks, None for local backends
```

Errors are one type — `jigor::Error` (carried by
`jigor::Result<T>`): `UnknownModel`/`MissingApiKey`/`Remote` for
routing and OpenRouter responses, `Wire`/`MissingAnswer`/`MissingAnswers`
for malformed question/answer payloads, `Serialization` for JSON text,
`External`/`Internal` for everything else. No foreign error type ever
leaks out of the library.

```rust
use jigor::{Error, Result};

match von.answers(&state, &questions, None) {
    Ok(asks) => { /* typed answers */ }
    Err(Error::Remote { status, message }) => { /* upstream 4xx/5xx */ }
    Err(e) => println!("{e}"),
}
```

## Where models are stored?

The local ONNX models (`von`, `laya`, `jeff`) download to
`~/.cache/huggingface` on first use. `jeff` pulls ~3.2 GB (fp32 weights);
note the answer probabilities are verified to the model's own cross-precision
floor (~1e-2), and like the upstream jeff project the model is English-only,
with no reasoning — it judges options, and the 512-token input cap is fixed
by the exported graph.

Set `OPENROUTER_API_KEY` for the remote `jev` backend.

## Run examples (lib crate, no bin)

```bash
cargo run -p jigor --example decide
cargo run -p jigor --release --example bench
cargo run -p jigor --example decide --offline
```

Expected `decide`:

```
infrastructure
0.428
{'infrastructure': 0.6203, 'billing': 0.1873, 'feature_request': 0.1924}
judge: 0.3586
rate score: 1.02 conf: 0.707 probs: {"1": 0.8109, "2": 0.1036, "0": 0.0855}
fan-out intent: payment_failure 0.539
```

Set `JIGOR_DEVICE=cuda` to try CUDA EP
(`ort` `cuda` feature, falls back to CPU if unavailable).

## Usage as Gateway

```bash
jigor serve --host 0.0.0.0 --port 8000   # HTTP gateway
```

The gateway mirrors the library: noul/choice/score questions in, typed
answers out — `von`/`laya` locally, anything else routed to the OpenRouter
backend selected by the `model` field:

```bash
curl -X POST http://localhost:8000/v1/systemone \
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
# {"model":"von-1.0.0","backend":"local","answers":{"requires_intervention":{"type":"noul","noul":0.2739}}}
```

Route by provider + model pair — `typesafe/jev-1.13` (alias `jev`) and
`jaredpalmer/kev-4b` (alias `kev`) go to OpenRouter, `von-1.0.0` (alias `von`)
stays local; unknown pairs are rejected.
Responses from the OpenRouter backend add `usage` (tokens + cost in USD) next
to `answers`, so each `/v1/systemone` call reports what it cost.

Also: `GET /healthz` returns `{"status":"ok"}`.

## Tweet Tester (example: using the lib)

`examples/tweet.rs` is a complete Tweet Tester written _only_ against the lib
API — it demonstrates how to build a Jev-style tool on top of
`noul`/`choice`/`score` questions through one `answers` interface. All
tweet-specific code lives in the example:

- the 61-question viral-score bank (question set `v1.1`, 8 families
  EMO/CNV/SHR/TIM/CRF/IDN/FMT/ANTI);
- a transparent 0-100 aggregation (`0.65 * content mean + 0.35 * clean
anti-signal`) — scoring semantics: 50 = your account's normal post, above
  50 beats it, below 50 does worse;
- per-family "fired % of N questions" radar stats, top helped/hurt, and
  engagement `counters` (a wire JSON shape mirroring the viral-score API).

```bash
cargo run -p jigor --example tweet -- "We just crossed 10,000 paying customers. Thank you."
cargo run -p jigor --example tweet -- --json "We just crossed 10,000 paying customers."
# {"score":53,"beats_own_normal":0.53,
#  "families":{"EMOTION":{"label":"Emotion","fired":0.11,"total":9},...},
#  "counters":{"likes":{"multiple":1.1,"p75":2.1,"p90":4.6,"breakout_share":0.1,"probability":0.5,"confidence":"normal","own_median":null,"expected":null},...},
#  "helped":[{"id":"k_concrete_numbers","family":"CRAFT","label":"Numbers that carry weight","answer":"Yes","detail":"Stronger than your usual post","effect":0.95}],
#  "hurt":[...],"answers":[...61 items...],"engine":{"model":"von-1.0.0","question_set":"v1.1",...}}
```

The example resolves the backend by model, exactly like `jigor ask`: `--model
von` (default, local ONNX), `--model jev` (OpenRouter) or any other alias.
The 61-question bank, score, radar and counters are identical across
backends, so you can compare the same tweet side by side:

```bash
cargo run -p jigor --example tweet -- --model von "We just crossed 10,000 paying customers. Thank you."
cargo run -p jigor --example tweet -- --model jev "We just crossed 10,000 paying customers. Thank you."
cargo run -p jigor --example tweet -- --model jev --json "Hot take." > jev.json
```

For the milestone tweet the outputs line up: von gives 53/100 (conservative,
CRAFT 35%), jev gives 63/100 (CRAFT 73%). Scores across backends are not
calibrated to each other, but the shape is directly comparable.

Family `fired` is the "X% of its questions fired" radar value. The engagement
`counters` (multiples, p75/p90, probabilities) and the `effect` coefficients
are transparent placeholders for a fitted engagement model — tune the
weights in the example's `counter_json`/`counter_multiple` once you have
paired data. Von is a general decision model, so scores are a signal, not a
forecast; fitting an engagement model on the same answers is the calibration
step.

Question builders: `choice` takes plain option strings (each is both key and
description), `choice_pairs` takes `(key, description)` pairs, `score` takes
ordered level texts, `noul` a plain instruction — and all three kinds mix
freely in one `answers` call.

## Auto-tagger (example: using the lib)

`examples/tagger.rs` shows a `choice` workflow: given a note and a list of
existing tags, one question — "Which tag best matches the content of this
note?" — picks the best tag (or the `None of these fit well` fallback) with a
probability distribution. Also runs on either backend via `--model`/`--provider`.

```bash
cargo run -p jigor --example tagger -- --title "Hiring notes" --tags "work, ideas, personal" "Budget approved for two engineers."
# Best tag: work  (confidence 0.440)
#   work 60%  ·  ideas 16%  ·  personal 13%  ·  None of these fit well 10%
cargo run -p jigor --example tagger -- --model jev --tags "bugs, docs, ship" "Fixed the retry loop that dropped webhook events."
cargo run -p jigor --example tagger -- --json --tags "a, b" "note text"   # wire JSON out
```

## Integration tests

Requires [hurl](https://hurl.dev) and the local ONNX model (downloaded on
first run). Wire fixtures live in `tests/fixtures/` (the OpenRouter Decisions
request/response payloads are the reference for the wire format).

```bash
make test        # unit tests + hurl suite + CLI tests
bash tests/hurl/run.sh   # jigor serve: /v1/systemone (health, noul, choice,
                         # score, fan-out, error paths, backend routing)
bash tests/cli/ask.sh    # jigor ask / jigor models over stdin fixtures
# or against a running server:
hurl --test --variable BASE_URL=http://localhost:8000 tests/hurl/*.hurl
```

## If you like it and want to support it

You can [subscribe](https://zatsepin.dev/subscribe).
I will not spam you, but 100% will share my work with you sometimes. I am lazy, don't worry too much.

Or try one of my apps:

- <https://zatsepin.dev/lubo>
- <https://zatsepin.dev/govor>
