//! `jigor` — System One gateway CLI.
//!
//! One wire protocol, many backends. Pick a model, optionally pin a provider,
//! and every ask returns the same typed answer shapes. `von` and `laya` run
//! as local ONNX backends; `jev` runs on OpenRouter's Decisions API
//! (`typesafe/jev-1.13`). Switch models by model name, or by model name plus
//! provider pair; future backends plug in the same way.
//!
//! ```bash
//! jigor serve --host 0.0.0.0 --port 8000   # HTTP gateway for /v1/systemone
//! echo '{"model":"von-1.0.0","state":"...","questions":{...}}' | jigor ask
//! echo '{"model":"laya","state":"...","questions":{...}}' | jigor ask
//! echo '{"state":"...","questions":{...}}' | jigor ask --provider openrouter --model jev
//! jigor models                              # list provider x model pairs
//! ```

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use http::{HeaderValue, Method, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use jigor::{
    Answer, Asks, Backend, LayaBackend, OpenRouterBackend, Question, VonBackend, alias_model,
    answer_to_json, known_providers, local_kind, resolve_model, usage_to_json,
};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::io::BufRead;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

const MAX_BODY_SIZE: usize = 8 * 1024 * 1024;

enum CliCommand {
    Serve {
        host: String,
        port: u16,
    },
    Ask {
        model: Option<String>,
        provider: Option<String>,
    },
    Models,
}

fn print_usage() {
    println!("jigor serve --host <ip> --port <port>     run the HTTP gateway (POST /v1/systemone)");
    println!("jigor ask [--model <id>] [--provider <p>]  ask the wire JSON piped on stdin");
    println!("jigor models                               list provider x model pairs");
    println!("  model aliases: von, laya, jev, jev-latest     providers: local, openrouter");
}

fn parse_args_from(argv: Vec<String>) -> Result<CliCommand> {
    let mut args = argv.into_iter();
    let cmd = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: jigor serve | ask | models (see --help)"))?
        .to_string();
    match cmd.as_str() {
        "serve" => {
            let mut host = String::from("127.0.0.1");
            let mut port: u16 = 8000;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--host" => {
                        host = args
                            .next()
                            .ok_or_else(|| anyhow::anyhow!("--host requires a value"))?;
                    }
                    "--port" => {
                        let raw = args
                            .next()
                            .ok_or_else(|| anyhow::anyhow!("--port requires a value"))?;
                        port = raw.parse().context("invalid --port")?;
                    }
                    "--help" | "-h" => {
                        print_usage();
                        std::process::exit(0);
                    }
                    other => bail!("unknown argument: {other} (see --help)"),
                }
            }
            Ok(CliCommand::Serve { host, port })
        }
        "ask" => {
            let mut model: Option<String> = None;
            let mut provider: Option<String> = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--model" => {
                        model = Some(
                            args.next()
                                .ok_or_else(|| anyhow::anyhow!("--model requires a value"))?
                                .to_string(),
                        );
                    }
                    "--provider" => {
                        provider = Some(
                            args.next()
                                .ok_or_else(|| anyhow::anyhow!("--provider requires a value"))?
                                .to_string(),
                        );
                    }
                    "--help" | "-h" => {
                        print_usage();
                        std::process::exit(0);
                    }
                    other => bail!("unknown argument: {other} (see --help)"),
                }
            }
            Ok(CliCommand::Ask { model, provider })
        }
        "models" => Ok(CliCommand::Models),
        "--help" | "-h" => {
            print_usage();
            std::process::exit(0);
        }
        other => bail!("unknown command: {other} (see --help)"),
    }
}

fn parse_args() -> Result<CliCommand> {
    parse_args_from(std::env::args().skip(1).collect())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cmd = parse_args()?;
    match cmd {
        CliCommand::Serve { host, port } => run(&host, port).await,
        CliCommand::Ask { model, provider } => run_ask(model, provider),
        CliCommand::Models => run_models(),
    }
}

/// `jigor models` — list the provider x model pairs the gateway can route.
fn run_models() -> Result<()> {
    println!("provider    model id          aliases");
    for (provider, model) in known_providers() {
        let aliases: Vec<&'static str> = ["von", "laya", "jev", "jev-latest"]
            .iter()
            .filter(|a| alias_model(a) == model.as_str())
            .copied()
            .collect();
        let alias_str = if aliases.is_empty() {
            String::new()
        } else {
            format!("({})", aliases.join(", "))
        };
        println!("  {:<9}  {:<18} {}", provider, model, alias_str);
    }
    println!("resolve with: --provider <name> --model <id-or-alias>");
    Ok(())
}

/// Read all of stdin as text (EOF or an empty line ends the ask payload).
fn read_stdin() -> Result<String> {
    let mut buf = String::new();
    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    loop {
        let mut line = String::new();
        match handle.read_line(&mut line) {
            Ok(len) => {
                if len == 0 {
                    break;
                }
                buf = format!("{buf}{line}");
            }
            Err(_) => break,
        }
    }
    Ok(buf)
}

/// Errors from processing a wire request. Shape/routing problems are
/// client errors (400); backend execution failures keep their per-provider
/// statuses (500 local, 502 remote).
#[derive(Debug)]
enum BackendKind {
    Local,
    Remote,
}

#[derive(Debug)]
enum PayloadError {
    Invalid(String),
    Backend { kind: BackendKind, message: String },
}

impl Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayloadError::Invalid(m) => write!(f, "{m}"),
            PayloadError::Backend { message, .. } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for PayloadError {}

/// Run one wire request through the resolved backend: parse model/provider
/// (CLI hints win over wire values), resolve aliases, parse questions and
/// execute. Shared by `jigor ask` and the HTTP gateway; `run_local` decides
/// how local backends are executed (fresh per call vs resident in serve).
fn process_payload(
    payload: &Value,
    model_hint: Option<&str>,
    provider_hint: Option<&str>,
    run_local: impl FnOnce(
        &str,
        &Value,
        &[Question],
        Option<f32>,
    ) -> jigor::Result<HashMap<String, Answer>>,
) -> Result<Asks, PayloadError> {
    let wire_model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("von-1.0.0")
        .to_string();
    let wire_provider = payload.get("provider").and_then(Value::as_str);
    let model = model_hint.unwrap_or(&wire_model);
    let provider = provider_hint.or(wire_provider);
    let (resolved_provider, resolved_model) =
        resolve_model(provider, model).map_err(PayloadError::Invalid)?;
    let state = match payload.get("state") {
        Some(v) => v.clone(),
        None => return Err(PayloadError::Invalid("missing \"state\"".into())),
    };
    let questions = match payload.get("questions") {
        Some(qv) => parse_questions(qv).map_err(PayloadError::Invalid)?,
        None => return Err(PayloadError::Invalid("missing \"questions\" object".into())),
    };
    let global_temp = payload
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|t| t as f32);

    let backend = resolved_provider.clone();
    let (answers, usage) = match resolved_provider.as_str() {
        "local" => (
            run_local(&resolved_model, &state, &questions, global_temp),
            None,
        ),
        "openrouter" => {
            let remote = OpenRouterBackend::for_model(&resolved_model);
            match remote.ask(&state, &questions, global_temp) {
                Ok(asks) => (Ok(asks.answers), asks.usage),
                Err(e) => (Err(e), None),
            }
        }
        other => (
            Err(jigor::Error::internal(format!(
                "no backend for provider \"{other}\" (see `jigor models`)"
            ))),
            None,
        ),
    };
    let answers = answers.map_err(|e| PayloadError::Backend {
        kind: if backend == "local" {
            BackendKind::Local
        } else {
            BackendKind::Remote
        },
        message: e.to_string(),
    })?;
    Ok(Asks {
        model: resolved_model,
        backend,
        answers,
        usage,
    })
}

/// Wire JSON of one response: model, backend, answers, and — for remote
/// backends that report it — tokens and cost (USD) under `usage`.
fn asks_to_json(asks: Asks) -> Value {
    let mut wire = Map::new();
    for (qid, answer) in asks.answers {
        wire.insert(qid.clone(), answer_to_json(&answer));
    }
    let mut out = Map::new();
    out.insert("model".to_string(), Value::String(asks.model));
    out.insert("backend".to_string(), Value::String(asks.backend));
    out.insert("answers".to_string(), Value::Object(wire));
    if let Some(usage) = asks.usage {
        out.insert("usage".to_string(), usage_to_json(&usage));
    }
    Value::Object(out)
}

/// `jigor ask` — run one wire request (JSON on stdin) through the chosen
/// backend, without the HTTP layer. Same wire in, same wire out.
fn run_ask(model_hint: Option<String>, provider_hint: Option<String>) -> Result<()> {
    let buf = read_stdin()?;
    let payload: Value = match serde_json::from_str(&buf) {
        Ok(v) => v,
        Err(e) => bail!("stdin is not a JSON request: {e}"),
    };

    let run_local = |model: &str, state: &Value, questions: &[Question], temp: Option<f32>| {
        jigor::ask(model, state, questions, temp).map(|a| a.answers)
    };
    let asks = process_payload(
        &payload,
        model_hint.as_deref(),
        provider_hint.as_deref(),
        run_local,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", serde_json::to_string(&asks_to_json(asks))?);
    Ok(())
}

/// Parse the wire `questions` object into typed Questions.
fn parse_questions(v: &Value) -> std::result::Result<Vec<Question>, String> {
    let obj = match v.as_object() {
        Some(o) => o,
        _ => return Err("missing \"questions\" object".to_string()),
    };
    let mut out: Vec<Question> = Vec::with_capacity(obj.len());
    for (qid, qv) in obj {
        let q = match qv.as_object() {
            Some(m) => m,
            None => return Err(format!("question \"{qid}\" must be an object")),
        };
        let kind: &str = match q.get("type").and_then(Value::as_str) {
            Some(t) => t,
            None => {
                let has_criteria = q.contains_key("criteria");
                let is_array = q.get("criteria").and_then(Value::as_array).is_some();
                if has_criteria && q.contains_key("instructions") {
                    "choice"
                } else if is_array {
                    "score"
                } else if q.contains_key("instructions") {
                    // criteria-less questions default to noul (mirrors `noul()`)
                    "noul"
                } else {
                    return Err(format!("question \"{qid}\" missing \"type\""));
                }
            }
        };
        if kind != "noul" && kind != "choice" && kind != "score" {
            return Err(format!("question \"{qid}\" has unknown type \"{kind}\""));
        }
        let criteria = match kind {
            "choice" => {
                let c = q.get("criteria");
                if c.is_none() {
                    return Err(format!("question \"{qid}\" needs criteria"));
                }
                match c {
                    Some(Value::Object(m)) => {
                        for (key, val) in m {
                            if !matches!(val, Value::String(_) | Value::Null) {
                                return Err(format!(
                                    "choice \"{key}\" description must be a string or null"
                                ));
                            }
                        }
                    }
                    Some(Value::Array(items)) => {
                        for item in items {
                            if item.as_str().is_none() {
                                return Err("choice names in list form must be strings".to_string());
                            }
                        }
                    }
                    Some(_) => {
                        return Err("choice \"criteria\" must be an object or an array".to_string());
                    }
                    None => {}
                }
                c.cloned()
            }
            "score" => {
                let c = q.get("criteria");
                if c.is_none() {
                    return Err(format!("question \"{qid}\" needs criteria list"));
                }
                match c {
                    Some(Value::Array(items)) => {
                        for item in items {
                            if !matches!(item, Value::String(_) | Value::Object(_)) {
                                return Err(
                                    "score criteria items must be strings or objects".to_string()
                                );
                            }
                        }
                    }
                    Some(_) => return Err(format!("question \"{qid}\" needs criteria list")),
                    None => {}
                }
                c.cloned()
            }
            // noul criteria must key on true/false: canonicalize the case
            // here so no backend ever sees a key it would drop on the floor
            "noul" => jigor::normalize_noul_criteria(q.get("criteria"))
                .map_err(|e| format!("question \"{qid}\": {e}"))?,
            _ => q.get("criteria").cloned(),
        };
        let instructions = q
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let temperature = q
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|t| t as f32);
        out.push(Question {
            id: qid.clone(),
            kind: kind.to_string(),
            instructions,
            criteria,
            temperature,
        });
    }
    Ok(out)
}

/// The local ONNX backends shared by the gateway. `jigor serve` keeps von and
/// laya resident, so local requests never reload a 1.5 GB model. One lock per
/// backend: a long laya run must not block von asks (and vice versa).
struct LocalBackends {
    von: Mutex<VonBackend>,
    laya: Mutex<LayaBackend>,
}

async fn run(host: &str, port: u16) -> Result<()> {
    let addr = format!("{host}:{port}");
    let _ = ort::init().commit();
    let backends = Arc::new(LocalBackends {
        von: Mutex::new(VonBackend::new().context("load von ONNX model")?),
        laya: Mutex::new(LayaBackend::new().context("load laya ONNX model")?),
    });
    println!("jigor serve (system one gateway) listening on http://{addr}");

    let listener = TcpListener::bind(addr.as_str())
        .await
        .with_context(|| format!("cannot bind {addr}"))?;
    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(x) => x,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        let backends = backends.clone();
        tokio::spawn(async move {
            let service = service_fn(move |req: Request<Incoming>| {
                let backends = backends.clone();
                async move {
                    let resp = route(req, backends).await;
                    Ok::<_, hyper::Error>(resp)
                }
            });
            let io = TokioIo::new(stream);
            if let Err(e) = auto::Builder::new(TokioExecutor::new())
                .serve_connection(io, service)
                .await
            {
                eprintln!("connection error: {e}");
            }
        });
    }
}

async fn route(req: Request<Incoming>, backends: Arc<LocalBackends>) -> Response<Full<Bytes>> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/v1/systemone") => handle_systemone(req, backends).await,
        (&Method::GET, "/healthz") => json_response(
            StatusCode::OK,
            &json!({"status": "ok", "model": "von-1.0.0"}),
        ),
        _ => error_response(StatusCode::NOT_FOUND, "not found"),
    }
}

async fn handle_systemone(
    req: Request<Incoming>,
    backends: Arc<LocalBackends>,
) -> Response<Full<Bytes>> {
    // Bound the read while draining: never buffer an unbounded body.
    let mut body = req.into_body();
    let mut collected: Vec<u8> = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = match frame {
            Ok(f) => f,
            Err(e) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    &format!("failed to read body: {e}"),
                );
            }
        };
        if let Ok(data) = frame.into_data() {
            collected.extend_from_slice(&data);
            if collected.len() > MAX_BODY_SIZE {
                return error_response(StatusCode::PAYLOAD_TOO_LARGE, "request body too large");
            }
        }
    }
    let payload: Value = match serde_json::from_slice(&collected) {
        Ok(v) => v,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("invalid JSON: {e}")),
    };

    let run_local = |model: &str, state: &Value, questions: &[Question], temp: Option<f32>| {
        if local_kind(model) == "laya" {
            let mut laya = backends
                .laya
                .lock()
                .map_err(|_| jigor::Error::internal("backend lock poisoned".to_string()))?;
            laya.answers(state, questions, temp)
        } else {
            let mut von = backends
                .von
                .lock()
                .map_err(|_| jigor::Error::internal("backend lock poisoned".to_string()))?;
            von.answers(state, questions, temp)
        }
    };
    match process_payload(&payload, None, None, run_local) {
        Ok(asks) => json_response(StatusCode::OK, &asks_to_json(asks)),
        Err(PayloadError::Invalid(msg)) => error_response(StatusCode::BAD_REQUEST, &msg),
        Err(PayloadError::Backend {
            kind: BackendKind::Local,
            message,
        }) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("scoring failed: {message}"),
        ),
        Err(PayloadError::Backend {
            kind: BackendKind::Remote,
            message,
        }) => error_response(StatusCode::BAD_GATEWAY, &message),
    }
}

fn json_response(status: StatusCode, value: &Value) -> Response<Full<Bytes>> {
    let body = match serde_json::to_string(value) {
        Ok(s) => s,
        Err(e) => format!("{{\"error\":\"serialization failed: {e}\"}}"),
    };
    let mut resp = Response::new(Full::new(Bytes::from(body)));
    *resp.status_mut() = status;
    let _ = resp
        .headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

fn error_response(status: StatusCode, message: &str) -> Response<Full<Bytes>> {
    json_response(status, &json!({"error": message}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[tokio::test]
    async fn parse_args_defaults() {
        let cmd = parse_args_from(args(&["serve"])).unwrap();
        match cmd {
            CliCommand::Serve { host, port } => {
                assert_eq!(host, "127.0.0.1");
                assert_eq!(port, 8000);
            }
            _ => panic!("expected serve"),
        }
    }

    #[tokio::test]
    async fn parse_args_host_port() {
        let cmd = parse_args_from(args(&["serve", "--host", "0.0.0.0", "--port", "9000"])).unwrap();
        match cmd {
            CliCommand::Serve { host, port } => {
                assert_eq!(host, "0.0.0.0");
                assert_eq!(port, 9000);
            }
            _ => panic!("expected serve"),
        }
    }

    #[tokio::test]
    async fn parse_args_missing_value() {
        assert!(parse_args_from(args(&["serve", "--port"])).is_err());
        assert!(parse_args_from(args(&["serve", "--host"])).is_err());
    }

    #[tokio::test]
    async fn parse_args_bad_port() {
        assert!(parse_args_from(args(&["serve", "--port", "abc"])).is_err());
    }

    #[tokio::test]
    async fn parse_args_unknown_flag() {
        assert!(parse_args_from(args(&["serve", "--bogus"])).is_err());
    }

    #[tokio::test]
    async fn parse_args_requires_command() {
        assert!(parse_args_from(args(&["--host", "1.2.3.4"])).is_err());
        assert!(parse_args_from(args(&[])).is_err());
    }

    #[tokio::test]
    async fn parse_args_ask_flags() {
        let cmd =
            parse_args_from(args(&["ask", "--model", "jev", "--provider", "openrouter"])).unwrap();
        match cmd {
            CliCommand::Ask { model, provider } => {
                assert_eq!(model.unwrap(), "jev");
                assert_eq!(provider.unwrap(), "openrouter");
            }
            _ => panic!("expected ask"),
        }
    }

    #[tokio::test]
    async fn parse_args_models() {
        match parse_args_from(args(&["models"])).unwrap() {
            CliCommand::Models => {}
            _ => panic!("expected models"),
        }
        assert!(parse_args_from(args(&["bogus"])).is_err());
    }

    #[tokio::test]
    async fn parse_questions_wire() {
        let v = json!({
            "is_bug": {"type": "noul", "instructions": "Is it a bug?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"payments": "x", "frontend": "y"}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}
        });
        let qs = parse_questions(&v).unwrap();
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0].kind, "noul");
        assert_eq!(qs[1].kind, "choice");
        assert_eq!(qs[2].kind, "score");
        assert!(parse_questions(&json!("nope")).is_err());
        assert!(parse_questions(&json!({"a": {"type": "bogus"}})).is_err());
    }

    #[tokio::test]
    async fn parse_questions_infers_kind_without_type() {
        let v = json!({
            "q": {"instructions": "Is it a bug?"},
            "typed": {"type": "noul", "instructions": "Also a bug?"},
            "s": {"criteria": ["low", "high"]}
        });
        let qs = parse_questions(&v).unwrap();
        assert_eq!(qs.iter().find(|q| q.id == "q").unwrap().kind, "noul");
        assert_eq!(qs.iter().find(|q| q.id == "typed").unwrap().kind, "noul");
        assert_eq!(qs.iter().find(|q| q.id == "s").unwrap().kind, "score");
    }

    #[tokio::test]
    async fn parse_questions_normalizes_noul_criteria() {
        // keys arrive in any case, leave canonicalized for every backend
        let v = json!({
            "is_bug": {
                "type": "noul",
                "instructions": "Is it a bug?",
                "criteria": {"True": "a defect", "False": "working as intended"}
            }
        });
        let qs = parse_questions(&v).unwrap();
        assert_eq!(
            qs[0].criteria.as_ref().unwrap(),
            &json!({"true": "a defect", "false": "working as intended"})
        );

        // no criteria (and null criteria) stay absent
        let v = json!({"is_bug": {"type": "noul", "instructions": "Is it a bug?"}});
        assert!(parse_questions(&v).unwrap()[0].criteria.is_none());
        let v = json!({"is_bug": {"type": "noul", "instructions": "x", "criteria": null}});
        assert!(parse_questions(&v).unwrap()[0].criteria.is_none());

        // anything but true/false, or a non-object, is a payload error
        let v = json!({"is_bug": {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}});
        let err = parse_questions(&v).unwrap_err();
        assert!(err.starts_with("question \"is_bug\":"), "{err}");
        let v = json!({"is_bug": {"type": "noul", "instructions": "x", "criteria": ["a", "b"]}});
        assert!(parse_questions(&v).is_err());
    }

    #[tokio::test]
    async fn process_payload_routes_local_and_remote() {
        // local: the closure is invoked with the resolved model id
        let seen = Arc::new(Mutex::new(String::new()));
        let seen2 = seen.clone();
        let payload = json!({"model": "laya", "state": "x", "questions": {}});
        let asks = process_payload(&payload, None, None, move |model, _, _, _| {
            seen2.lock().unwrap().push_str(model);
            Ok(HashMap::new())
        })
        .unwrap();
        assert_eq!(asks.model, "laya-1.0.0");
        assert_eq!(asks.backend, "local");
        assert_eq!(seen.lock().unwrap().as_str(), "laya-1.0.0");

        // remote: no key -> Backend error before any HTTP
        let prev_key = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe { std::env::set_var("OPENROUTER_API_KEY", "") };
        let payload = json!({"model": "jev", "state": "x", "questions": {}});
        let err = match process_payload(&payload, None, None, |_, _, _, _| unreachable!()) {
            Err(e) => e,
            Ok(_) => panic!("expected a missing-key error"),
        };
        match prev_key {
            Some(k) => unsafe { std::env::set_var("OPENROUTER_API_KEY", k) },
            None => unsafe { std::env::remove_var("OPENROUTER_API_KEY") },
        };
        match err {
            PayloadError::Backend {
                kind: BackendKind::Remote,
                message,
            } => assert!(message.contains("OPENROUTER_API_KEY"), "{message}"),
            other => panic!("expected remote backend error, got {other:?}"),
        }

        // shape errors stay Invalid
        let payload = json!({"questions": {}});
        assert!(matches!(
            process_payload(&payload, None, None, |_, _, _, _| unreachable!()),
            Err(PayloadError::Invalid(_))
        ));
    }

    async fn body_to_value(resp: Response<Full<Bytes>>) -> (StatusCode, Value) {
        let status = resp.status();
        let body = resp.into_body();
        let collected = body.collect().await.unwrap();
        let v = serde_json::from_slice(&collected.to_bytes()).unwrap();
        (status, v)
    }

    #[tokio::test]
    async fn json_response_ok() {
        let (status, v) =
            body_to_value(json_response(StatusCode::OK, &json!({"status": "ok"}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v, json!({"status": "ok"}));
    }

    #[tokio::test]
    async fn json_response_status_and_header() {
        let resp = error_response(StatusCode::NOT_FOUND, "not found");
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );
        let (status, v) = body_to_value(resp).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(v, json!({"error": "not found"}));
    }
}
