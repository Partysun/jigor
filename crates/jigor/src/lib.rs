//! System One decision backends behind one interface: `noul`/`choice`/`score`
//! questions in, typed answers out — the same shape from every model.
//!
//! Backends implement the shared `Backend` trait, one file per model:
//!
//!   `src/von.rs`   — local VonBackend (sevenreasons/von-onnx-fp16, NLI
//!                    entailment scoring)
//!   `src/laya.rs`  — local LayaBackend (Mattepiu/laya-onnx, marker-scoring
//!                    decision head)
//!   `OpenRouterBackend` — any OpenRouter model on the Decisions wire
//!                    protocol (`typesafe/jev-1.13` today, more later), in
//!                    `lib.rs`
//!
//! Everything else — the wire protocol, the provider/model routing and the
//! shared ort session setup — lives here in `lib.rs`.

// ---- shared ort session setup (von and laya) ------------------------------

pub mod error;
pub use error::{Error, Result};

use anyhow::{Context, Result as AnyhowResult};
use std::collections::HashMap;
use std::path::Path;

/// Try CUDA if `JIGOR_DEVICE=cuda` or `CUDA_VISIBLE_DEVICES` is set,
/// fallback to CPU. Requires `ort` built with the `cuda` feature.
fn use_cuda() -> bool {
    let device = std::env::var("JIGOR_DEVICE").unwrap_or_default();
    device.to_lowercase().contains("cuda") || std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
}

pub(crate) fn init_session<P: AsRef<Path>>(model_path: P) -> AnyhowResult<ort::session::Session> {
    if !use_cuda() {
        return ort::session::Session::builder()?
            .commit_from_file(model_path)
            .context("ort session from model.onnx");
    }
    let builder = ort::session::Session::builder()?;
    // try CUDA provider, fallback to CPU on error (e.g. no CUDA EP in ort-sys)
    let with_cuda = builder.with_execution_providers([ort::ep::CUDA::default().build()]);
    match with_cuda {
        Ok(mut b) => match b.commit_from_file(&model_path) {
            Ok(s) => Ok(s),
            Err(e) => {
                eprintln!("CUDA session failed ({e}), fallback to CPU");
                ort::session::Session::builder()?
                    .commit_from_file(&model_path)
                    .context("ort CPU fallback")
            }
        },
        Err(e) => {
            eprintln!("CUDA EP not available ({e}), fallback to CPU");
            ort::session::Session::builder()?
                .commit_from_file(&model_path)
                .context("ort CPU fallback")
        }
    }
}

/// Temperature-scaled softmax over a slice (shared by von and laya).
pub(crate) fn softmax(scores: &[f32], temp: f32) -> Vec<f32> {
    let t = temp.max(1e-4);
    let scaled: Vec<f32> = scores.iter().map(|s| s / t).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(1e-12)).collect()
}

/// Round to 4 decimals, like the python backends' `round(x, 4)`.
pub(crate) fn round4(x: f32) -> f32 {
    (x * 10000.0).round() / 10000.0
}

/// Tokenizer ids as `i64` (the ort input element type), shared by von/laya.
pub(crate) fn i64_ids(ids: &[u32]) -> Vec<i64> {
    ids.iter().map(|&x| x as i64).collect()
}

/// Copy an ort tensor output into a dense `Array2`, collapsing any extra
/// leading batch dimensions (shared by von and laya).
pub(crate) fn tensor_to_array2(
    output: &ort::value::DynValue,
    name: &str,
) -> AnyhowResult<ndarray::Array2<f32>> {
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .with_context(|| format!("{name} tensor"))?;
    let batch: usize = shape[..shape.len().saturating_sub(1)]
        .iter()
        .map(|&d| d as usize)
        .product();
    let width = shape.last().copied().unwrap_or(1) as usize;
    let mut arr = ndarray::Array2::<f32>::zeros((batch, width));
    for i in 0..batch {
        for j in 0..width {
            arr[[i, j]] = data[i * width + j];
        }
    }
    Ok(arr)
}

/// Which local backend answers a model id ("laya"/"von"). Unknown ids fall
/// back to von (the legacy default), matching `backend_for`.
pub fn local_kind(model: &str) -> &'static str {
    if model.starts_with("laya") {
        "laya"
    } else {
        "von"
    }
}

// ---- backends --------------------------------------------------------------

pub mod laya;
pub mod von;

pub use laya::LayaBackend;
pub use von::VonBackend;
// ============================================================================
// Decision backends — one ask() across System One models.
//
// Every backend speaks the same wire protocol (the TypeSafe primitives,
// documented on OpenRouter as the Decisions API):
//
//   questions: {
//     "<id>": { "type": "noul"|"choice"|"score", "instructions": "...", "criteria": {...} }
//   }
//   -> answers: { "<id>": {"type": ..., "noul"|"choice"|"score"|... } }
//
// Pick a backend by model id (`backend_for`), so switching models or fanning
// out to several at once is a matter of naming the model:
//
//   "von-1.0.0"              -> local VonBackend (this repo, ONNX)
//   "laya-1.0.0"             -> local LayaBackend (this repo, ONNX)
//   "typesafe/jev-1.13"      -> OpenRouter Decisions API (Jev), via ureq
//   "~typesafe/jev-latest"   -> same remote backend, unpinned alias
//
// New backends add a branch in `backend_for`, an entry in `known_providers`
// and an `impl Backend`; the wire shapes stay the same. Local backends can
// be held as long-lived instances (`VonBackend::new()`, `LayaBackend::new()`)
// and asked through the shared `Backend::answers`; callers that only know a
// model id use `jigor::ask` (aliases resolved, provider inferred).
// ============================================================================

use anyhow::bail;
use http::StatusCode;
use serde_json::{Map, Value, json};
use std::time::Duration;
use ureq::Agent;

#[derive(Debug, Clone)]
pub struct Question {
    /// Answer key, chosen by the caller (not sent to the model).
    pub id: String,
    /// "noul" | "choice" | "score"
    pub kind: String,
    /// How to judge the state: a specific question or a statement.
    pub instructions: String,
    /// noul: {"true": .., "false": ..}; choice: {option: desc}; score: [levels].
    pub criteria: Option<Value>,
    /// Local temperature override; ignored by remote backends.
    pub temperature: Option<f32>,
}

/// A yes/no judgment (`type: "noul"`).
pub fn noul(id: &str, instructions: &str) -> Question {
    Question {
        id: id.to_string(),
        kind: "noul".to_string(),
        instructions: instructions.to_string(),
        criteria: None,
        temperature: None,
    }
}

/// One option from a fixed set (`type: "choice"`). Each option string is
/// both the answer key and its description.
pub fn choice(id: &str, instructions: &str, options: &[&str]) -> Question {
    let mut criteria = Map::new();
    for opt in options {
        criteria.insert(opt.to_string(), Value::Null);
    }
    Question {
        id: id.to_string(),
        kind: "choice".to_string(),
        instructions: instructions.to_string(),
        criteria: Some(Value::Object(criteria)),
        temperature: None,
    }
}

/// `choice` with an explicit description per option (`{key: desc}` criteria).
pub fn choice_pairs(id: &str, instructions: &str, pairs: &[(&str, &str)]) -> Question {
    let mut options = HashMap::new();
    for (key, desc) in pairs {
        options.insert(key.to_string(), desc.to_string());
    }
    Question {
        id: id.to_string(),
        kind: "choice".to_string(),
        instructions: instructions.to_string(),
        criteria: Some(json!(options)),
        temperature: None,
    }
}

/// A position on an ordered scale (`type: "score"`).
pub fn score(id: &str, instructions: &str, levels: &[&str]) -> Question {
    let mut list: Vec<Value> = Vec::with_capacity(levels.len());
    for level in levels {
        list.push(json!(level));
    }
    Question {
        id: id.to_string(),
        kind: "score".to_string(),
        instructions: instructions.to_string(),
        criteria: Some(json!(list)),
        temperature: None,
    }
}

/// The typed answer every backend returns for one question.
#[derive(Debug, Clone)]
pub enum Answer {
    /// `{"type": "noul", "noul": p}` — probability of yes in [0, 1].
    Noul { probability: f32 },
    /// `{"type": "choice", "choice", "confidence", "probabilities"}`.
    Choice {
        choice: String,
        confidence: f32,
        probabilities: HashMap<String, f32>,
    },
    /// `{"type": "score", "score", "confidence", "probabilities", "legend"}`.
    Score {
        score: f32,
        confidence: f32,
        probabilities: HashMap<String, f32>,
        legend: HashMap<String, String>,
    },
}

impl Answer {
    pub fn kind(&self) -> &'static str {
        match self {
            Answer::Noul { .. } => "noul",
            Answer::Choice { .. } => "choice",
            Answer::Score { .. } => "score",
        }
    }
}

/// Wire JSON of one answer — the same shape from every backend.
pub fn answer_to_json(a: &Answer) -> Value {
    match a {
        Answer::Noul { probability } => json!({"type": "noul", "noul": probability}),
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            json!({"type": "choice", "choice": choice.clone(), "confidence": confidence, "probabilities": probabilities.clone()})
        }
        Answer::Score {
            score,
            confidence,
            probabilities,
            legend,
        } => {
            json!({"type": "score", "score": score, "confidence": confidence, "probabilities": probabilities.clone(), "legend": legend.clone()})
        }
    }
}

fn f32_map_from(v: &Value) -> AnyhowResult<HashMap<String, f32>> {
    let obj = match v.as_object() {
        Some(m) => m,
        _ => bail!("expected an object of numbers"),
    };
    let mut out = HashMap::new();
    for (key, val) in obj {
        match val.as_f64() {
            Some(n) => {
                out.insert(key.clone(), n as f32);
            }
            _ => bail!("value for \"{key}\" must be a number"),
        }
    }
    Ok(out)
}

fn str_map_from(v: &Value) -> AnyhowResult<HashMap<String, String>> {
    let obj = match v.as_object() {
        Some(m) => m,
        _ => bail!("expected an object of string values"),
    };
    let mut out = HashMap::new();
    for (key, val) in obj {
        match val {
            Value::String(s) => {
                out.insert(key.clone(), s.clone());
            }
            _ => bail!("value for \"{key}\" must be a string"),
        }
    }
    Ok(out)
}

/// Parse one answer object. `hint` is the question type when the object
/// carries no `type` of its own (some local wire shapes omit it).
pub fn answer_from_json(v: &Value, hint: &str) -> Result<Answer> {
    let kind = v.get("type").and_then(Value::as_str).unwrap_or(hint);
    match kind {
        "noul" => match v.get("noul").and_then(Value::as_f64) {
            Some(p) => Ok(Answer::Noul {
                probability: p as f32,
            }),
            _ => Err(Error::Wire {
                message: "noul answer missing numeric \"noul\"".to_string(),
            }),
        },
        "choice" => {
            let picked = v
                .get("choice")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("choice answer missing \"choice\""))?;
            let confidence = v.get("confidence").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let probabilities = match v.get("probabilities") {
                Some(p) => f32_map_from(p)?,
                _ => HashMap::new(),
            };
            Ok(Answer::Choice {
                choice: picked.to_string(),
                confidence,
                probabilities,
            })
        }
        "score" => {
            let value = v
                .get("score")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow::anyhow!("score answer missing \"score\""))?
                as f32;
            let confidence = v.get("confidence").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let probabilities = match v.get("probabilities") {
                Some(p) => f32_map_from(p)?,
                _ => HashMap::new(),
            };
            let legend = match v.get("legend") {
                Some(l) => str_map_from(l)?,
                _ => HashMap::new(),
            };
            Ok(Answer::Score {
                score: value,
                confidence,
                probabilities,
                legend,
            })
        }
        other => Err(Error::Wire {
            message: format!("unknown answer type \"{other}\""),
        }),
    }
}

/// Parse a whole `{"answers": {...}}` response against the questions asked.
/// Works for OpenRouter Decisions responses and for any local responder
/// that follows the same wire shape.
pub fn answers_from_wire(wire: &Value, questions: &[Question]) -> Result<HashMap<String, Answer>> {
    let map = wire.get("answers").and_then(Value::as_object);
    match map {
        Some(m) => {
            let mut out = HashMap::new();
            for q in questions {
                match m.get(&q.id) {
                    Some(item) => {
                        let parsed = answer_from_json(item, &q.kind)?;
                        out.insert(q.id.clone(), parsed);
                    }
                    _ => {
                        return Err(Error::MissingAnswer {
                            question: q.id.clone(),
                        });
                    }
                }
            }
            Ok(out)
        }
        _ => Err(Error::MissingAnswers),
    }
}

/// State is a string or any JSON value; backends judge it as text.
pub fn state_text(state: &Value) -> Result<String> {
    match state {
        Value::String(s) => Ok(s.clone()),
        other => match serde_json::to_string_pretty(other) {
            Ok(s) => Ok(s),
            Err(e) => Err(Error::Serialization(e)),
        },
    }
}
/// One way to ask every backend: `noul`/`choice`/`score` questions in,
/// typed answers out. Identical on von, laya and any OpenRouter System One
/// model — there is no per-model sugar.
pub trait Backend {
    fn answers(
        &mut self,
        state: &Value,
        questions: &[Question],
        temperature: Option<f32>,
    ) -> Result<HashMap<String, Answer>>;
}

pub struct Asks {
    pub model: String,
    /// Resolved provider name ("local" | "openrouter").
    pub backend: String,
    pub answers: HashMap<String, Answer>,
}

/// Remote System One backend: the OpenRouter Decisions API over the tiny
/// `ureq` HTTP client. One instance is bound to one model id (`for_model`),
/// so any OpenRouter model speaking the System One wire protocol works —
/// `typesafe/jev-1.13` today, more can be added to `known_providers`
/// without touching this backend. Point it at any endpoint that implements
/// the same wire protocol (this repo's `von serve` included).
#[derive(Debug, Clone)]
pub struct OpenRouterBackend {
    /// The model id this instance is bound to ("typesafe/jev-1.13", ...).
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub timeout_secs: u32,
}

impl OpenRouterBackend {
    /// Bind to an OpenRouter model id. `OPENROUTER_API_KEY` (required at
    /// ask time), `OPENROUTER_BASE_URL` (defaults to the Decisions API),
    /// `OPENROUTER_TIMEOUT_SECS` (default 30).
    pub fn for_model(model: &str) -> Self {
        OpenRouterBackend {
            model: model.to_string(),
            api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENROUTER_BASE_URL")
                .unwrap_or_else(|_| "https://openrouter.ai/api/alpha/decisions".to_string()),
            timeout_secs: std::env::var("OPENROUTER_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(30),
        }
    }

    /// The wire request body (pure, unit-tested).
    pub fn ask_body(
        &self,
        state: &Value,
        questions: &[Question],
        _temperature: Option<f32>,
    ) -> Value {
        let mut qs = Map::new();
        for q in questions {
            match &q.criteria {
                Some(c) => {
                    qs.insert(q.id.clone(), json!({"type": q.kind, "instructions": q.instructions, "criteria": c.clone()}));
                }
                None => {
                    qs.insert(
                        q.id.clone(),
                        json!({"type": q.kind, "instructions": q.instructions}),
                    );
                }
            }
        }
        json!({"model": self.model, "state": state.clone(), "questions": qs})
    }

    /// One POST to the remote Decisions endpoint, answers parsed and typed.
    pub fn ask(
        &self,
        state: &Value,
        questions: &[Question],
        temperature: Option<f32>,
    ) -> Result<Asks> {
        if self.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let body = self.ask_body(state, questions, temperature);
        let body_str = match serde_json::to_string(&body) {
            Ok(s) => s,
            Err(e) => return Err(Error::Serialization(e)),
        };

        let agent: Agent = Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(self.timeout_secs.max(1) as u64)))
            .build()
            .into();
        let req = agent
            .post(self.base_url.clone())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .config()
            .http_status_as_error(false)
            .build();
        let mut resp = req.send(body_str)?;
        let status = resp.status();
        let text = resp.body_mut().read_to_string()?;
        let payload: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => json!({}),
        };
        if status != StatusCode::OK {
            let message = payload
                .get("error")
                .and_then(Value::as_object)
                .map(|e| {
                    e.get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string()
                })
                .unwrap_or("".to_string());
            return Err(Error::Remote { status, message });
        }
        let resolved = payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&self.model);
        let answers = match answers_from_wire(&payload, questions) {
            Ok(a) => a,
            Err(e) => {
                return Err(Error::Wire {
                    message: format!("openrouter response: {e}"),
                });
            }
        };
        Ok(Asks {
            model: resolved.to_string(),
            backend: "openrouter".to_string(),
            answers,
        })
    }
}

impl Backend for OpenRouterBackend {
    fn answers(
        &mut self,
        state: &Value,
        questions: &[Question],
        temperature: Option<f32>,
    ) -> Result<HashMap<String, Answer>> {
        Ok(self.ask(state, questions, temperature)?.answers)
    }
}

/// Which backend answers a model id. Add future backends here.
pub fn backend_for(model: &str) -> std::result::Result<&'static str, String> {
    if model.starts_with("von") || model.starts_with("laya") {
        Ok("local")
    } else if model.starts_with("typesafe/")
        || model.starts_with("~typesafe/")
        || model.starts_with("openrouter/")
    {
        Ok("openrouter")
    } else {
        Err(format!("no backend for model \"{model}\""))
    }
}

/// Short aliases for full model ids.
pub fn alias_model(model: &str) -> &str {
    match model {
        "von" => "von-1.0.0",
        "laya" => "laya-1.0.0",
        "jev" => "typesafe/jev-1.13",
        "jev-latest" => "~typesafe/jev-latest",
        other => other,
    }
}

/// Ask a local backend, picking the implementation by model id. The provider
/// is only needed to disambiguate when the same model id exists on several
/// providers; today `laya-*` and `von-*` are local-only, so the model id
/// alone decides. Anything else requested against a local provider falls
/// back to von (the legacy default).
fn ask_local(
    model: &str,
    state: &Value,
    questions: &[Question],
    temp: Option<f32>,
) -> AnyhowResult<HashMap<String, Answer>> {
    if local_kind(model) == "laya" {
        let mut laya = LayaBackend::new().context("load laya ONNX model")?;
        Ok(laya.answers(state, questions, temp)?)
    } else {
        let mut von = VonBackend::new().context("load von ONNX model")?;
        Ok(von.answers(state, questions, temp)?)
    }
}

/// One entry point when you only know the model id: aliases resolved,
/// provider inferred, backend constructed, answers typed. The single way
/// to run `noul`/`choice`/`score` questions across every backend.
pub fn ask(
    model: &str,
    state: &Value,
    questions: &[Question],
    temperature: Option<f32>,
) -> Result<Asks> {
    let id = alias_model(model).to_string();
    match backend_for(&id) {
        Ok(backend) => match backend {
            "local" => {
                let answers = ask_local(&id, state, questions, temperature)?;
                Ok(Asks {
                    model: id,
                    backend: "local".to_string(),
                    answers,
                })
            }
            "openrouter" => {
                let remote = OpenRouterBackend::for_model(&id);
                remote.ask(state, questions, temperature)
            }
            _other => Err(Error::UnknownModel { model: id }),
        },
        Err(_) => Err(Error::UnknownModel { model: id }),
    }
}

/// Known providers and their default models. Add future OpenRouter System
/// One models here (same Decisions wire, nothing else to touch).
pub fn known_providers() -> Vec<(String, String)> {
    vec![
        ("local".to_string(), "von-1.0.0".to_string()),
        ("local".to_string(), "laya-1.0.0".to_string()),
        ("openrouter".to_string(), "typesafe/jev-1.13".to_string()),
    ]
}

/// Resolve a (provider, model) pair to (provider, full model id). The model
/// may be an alias ("jev"), a full id ("typesafe/jev-1.13"). Provider is
/// inferred when omitted, or overridden ("--provider openrouter --model jev").
pub fn resolve_model(
    provider: Option<&str>,
    model: &str,
) -> std::result::Result<(String, String), String> {
    let id = alias_model(model).to_string();
    match provider {
        Some(p) => match p {
            "local" => Ok(("local".to_string(), id)),
            "openrouter" => Ok(("openrouter".to_string(), id)),
            other => Err(format!("unknown provider \"{other}\" (try `jigor models`)")),
        },
        None => match backend_for(&id) {
            Ok(b) => Ok((b.to_string(), id)),
            Err(e) => Err(e),
        },
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Load a JSON fixture from `tests/fixtures/` (tests run from the repo
    /// root, like the hurl suite).
    fn fixture_value(name: &str) -> Value {
        let path = format!("tests/fixtures/{name}");
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing fixture: {}", name));
        serde_json::from_str::<Value>(&text).unwrap()
    }

    #[tokio::test]
    async fn backend_for_routes_local_and_openrouter() {
        assert_eq!(backend_for("von-1.0.0").unwrap(), "local");
        assert_eq!(backend_for("von-tiny").unwrap(), "local");
        assert_eq!(backend_for("laya-1.0.0").unwrap(), "local");
        assert_eq!(backend_for("laya-int8").unwrap(), "local");
        assert_eq!(backend_for("typesafe/jev-1.13").unwrap(), "openrouter");
        assert_eq!(backend_for("~typesafe/jev-latest").unwrap(), "openrouter");
        assert!(backend_for("gpt-4o").is_err());
    }

    #[tokio::test]
    async fn aliases_resolve_known_models() {
        assert_eq!(alias_model("von"), "von-1.0.0");
        assert_eq!(alias_model("laya"), "laya-1.0.0");
        assert_eq!(alias_model("jev"), "typesafe/jev-1.13");
        assert_eq!(alias_model("jev-latest"), "~typesafe/jev-latest");
        assert_eq!(alias_model("something-else"), "something-else");
    }

    #[tokio::test]
    async fn ask_body_matches_the_jev_tutorial_wire() {
        let remote = OpenRouterBackend::for_model("typesafe/jev-1.13");
        let questions: Vec<Question> = vec![
            noul("is_bug", "Is the customer reporting a software defect?"),
            choice(
                "team",
                "Which team should own this ticket?",
                &["payments", "frontend", "account"],
            ),
            score(
                "urgency",
                "How urgent is this ticket?",
                &[
                    "Can wait for the next release",
                    "Should be fixed this week",
                    "Blocking revenue right now",
                ],
            ),
        ];
        let fixture = fixture_value("jev_tutorial_request.json");
        let state = fixture.get("state").unwrap().clone();
        let body = remote.ask_body(&state, &questions, None);
        assert_eq!(
            body.get("model").unwrap().clone(),
            json!("typesafe/jev-1.13")
        );
        assert_eq!(body.get("state").unwrap().clone(), state);
        let qs = body.get("questions").unwrap().as_object().unwrap();
        assert_eq!(
            qs.get("is_bug").unwrap().get("type").unwrap().clone(),
            json!("noul")
        );
        assert_eq!(
            qs.get("team").unwrap().get("type").unwrap().clone(),
            json!("choice")
        );
        let team_criteria = qs
            .get("team")
            .unwrap()
            .get("criteria")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(team_criteria.contains_key("payments"));
        let urgency = qs.get("urgency").unwrap();
        assert_eq!(urgency.get("type").unwrap().clone(), json!("score"));
        assert_eq!(
            urgency.get("criteria").unwrap().as_array().unwrap().len(),
            3
        );
    }

    #[tokio::test]
    async fn answers_from_wire_parses_the_jev_tutorial_response() {
        let wire = fixture_value("jev_tutorial_response.json");
        let questions: Vec<Question> = vec![
            noul("is_bug", "Is the customer reporting a software defect?"),
            choice_pairs(
                "team",
                "Which team should own this ticket?",
                &[("payments", "x"), ("frontend", "x"), ("account", "x")],
            ),
            score("urgency", "How urgent is this ticket?", &["a", "b", "c"]),
        ];
        let answers = answers_from_wire(&wire, &questions).unwrap();
        match answers.get("is_bug").unwrap() {
            Answer::Noul { probability } => {
                assert!(*probability > 0.95 && *probability < 1.01, "p=0.96")
            }
            _ => panic!("expected noul"),
        }
        match answers.get("team").unwrap() {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                assert_eq!(choice, "payments");
                assert!(*confidence > 0.6 && *confidence < 0.7, "conf=0.67");
                assert_eq!(probabilities.get("payments").unwrap(), &0.78);
            }
            _ => panic!("expected choice"),
        }
        match answers.get("urgency").unwrap() {
            Answer::Score { score, legend, .. } => {
                assert!(*score > 1.9 && *score < 2.1, "score=1.99");
                assert_eq!(legend.get("2").unwrap(), "Blocking revenue right now");
            }
            _ => panic!("expected score"),
        }
    }

    #[tokio::test]
    async fn choice_builders_plain_and_pairs() {
        let q = choice("team", "Pick a team", &["payments", "frontend"]);
        assert_eq!(q.kind, "choice");
        let binding = q.criteria.unwrap();
        let obj = binding.as_object().unwrap();
        assert!(obj.contains_key("payments"));
        assert!(obj.contains_key("frontend"));
        assert_eq!(obj.get("payments").unwrap().clone(), Value::Null);

        let q = choice_pairs(
            "team",
            "Pick a team",
            &[("payments", "Billing issues."), ("frontend", "UI issues.")],
        );
        let binding = q.criteria.unwrap();
        let obj = binding.as_object().unwrap();
        assert_eq!(
            obj.get("payments").unwrap().clone(),
            json!("Billing issues.")
        );
    }

    #[tokio::test]
    async fn answers_from_wire_rejects_error_payloads() {
        let wire = fixture_value("openrouter_error.json");
        let questions: Vec<Question> = vec![noul(
            "is_bug",
            "Is the customer reporting a software defect?",
        )];
        assert!(answers_from_wire(&wire, &questions).is_err());
    }

    #[tokio::test]
    async fn answer_wire_roundtrip() {
        let n = Answer::Noul { probability: 0.42 };
        let wire = answer_to_json(&n);
        assert_eq!(wire, json!({"type": "noul", "noul": 0.42}));
        match answer_from_json(&wire, "noul").unwrap() {
            Answer::Noul { probability } => assert_eq!(probability, 0.42),
            _ => panic!("expected noul"),
        }

        let mut probs = HashMap::new();
        probs.insert("a".to_string(), 0.7_f32);
        probs.insert("b".to_string(), 0.3_f32);
        let c = Answer::Choice {
            choice: "a".to_string(),
            confidence: 0.4,
            probabilities: probs,
        };
        let wire = answer_to_json(&c);
        match answer_from_json(&wire, "choice").unwrap() {
            Answer::Choice { choice, .. } => assert_eq!(choice, "a"),
            _ => panic!("expected choice"),
        }
        // local-style answer without a "type" falls back to the hint
        match answer_from_json(&json!({"noul": 0.9}), "noul").unwrap() {
            Answer::Noul { probability } => assert_eq!(probability, 0.9),
            _ => panic!("expected noul"),
        }
        // unknown kinds are rejected
        assert!(answer_from_json(&json!({"noul": 0.5}), "text").is_err());
        assert!(answer_from_json(&json!({"type": "noul"}), "noul").is_err());
    }

    #[tokio::test]
    async fn state_text_string_and_object() {
        assert_eq!(state_text(&json!("Disk full")).unwrap(), "Disk full");
        let obj = json!({"error": "Disk volume /var/log at 98% capacity."});
        let text = state_text(&obj).unwrap();
        let parsed: Value = serde_json::from_str::<Value>(&text).unwrap();
        assert_eq!(parsed, obj);
    }

    #[tokio::test]
    async fn ask_answer_kind_labels() {
        assert_eq!(Answer::Noul { probability: 0.5 }.kind(), "noul");
        assert_eq!(
            Answer::Choice {
                choice: "a".to_string(),
                confidence: 0.0,
                probabilities: HashMap::new()
            }
            .kind(),
            "choice"
        );
        assert_eq!(
            Answer::Score {
                score: 1.0,
                confidence: 0.0,
                probabilities: HashMap::new(),
                legend: HashMap::new()
            }
            .kind(),
            "score"
        );
    }
}
