//! Unified error type for the whole jigor library.
//!
//! Every failure lands in this one enum — model download and inference
//! (von/laya ONNX), provider routing, wire parsing and remote
//! (OpenRouter) responses. Library functions return `error::Result<T>`;
//! foreign error types convert into it automatically, so the public
//! surface never leaks an external error type.

use anyhow::Error as AnyhowError;
use http::StatusCode;
use serde_json::Error as SerdeJsonError;
use thiserror::Error;

/// All jigor errors.
#[derive(Error, Debug)]
pub enum Error {
    /// A failure boxed by `anyhow` (ort, tokenizers, hf-hub, ureq, io):
    /// the automatic conversion target for every `?` that crosses into a
    /// foreign crate. Model load failures land here too — the underlying
    /// message says which part failed. `{0:#}` prints the full anyhow
    /// chain (context → source), so a failed hub download surfaces the
    /// HTTP status and url instead of only the last `.context(...)` tag.
    #[error("{0:#}")]
    External(#[from] AnyhowError),

    /// No backend answers the model id (see `jigor models`).
    #[error("no backend for model \"{model}\"")]
    UnknownModel { model: String },

    /// A remote (OpenRouter) ask was attempted without an API key.
    #[error("OPENROUTER_API_KEY is not set")]
    MissingApiKey,

    /// The remote Decisions endpoint answered with a non-OK status.
    #[error("OpenRouter responded {status}: {message}")]
    Remote { status: StatusCode, message: String },

    /// A wire response had no `answers` object.
    #[error("response has no \"answers\" object")]
    MissingAnswers,

    /// A wire response was missing one asked question.
    #[error("answer missing for question \"{question}\"")]
    MissingAnswer { question: String },

    /// A question or answer payload did not follow the wire protocol.
    #[error("malformed wire payload: {message}")]
    Wire { message: String },

    /// State could not be serialized to JSON text.
    #[error("serialize state: {0}")]
    Serialization(#[from] SerdeJsonError),

    /// Catch-all for dynamic-message failures.
    #[error("{0}")]
    Internal(String),
}

/// Result type of every jigor library operation.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Build an `Internal` error from a manually formatted message.
    pub fn internal(message: String) -> Self {
        Error::Internal(message)
    }
}

// `ureq` errors reach `OpenRouterBackend::ask` unwrapped (one `?` on the
// HTTP call), so box them through `anyhow::Error` like everything else.
impl From<ureq::Error> for Error {
    fn from(error: ureq::Error) -> Self {
        Error::External(error.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_read_well() {
        assert_eq!(
            format!(
                "{}",
                Error::UnknownModel {
                    model: "gpt-4o".to_string()
                }
            ),
            "no backend for model \"gpt-4o\""
        );
        assert_eq!(
            format!("{}", Error::MissingApiKey),
            "OPENROUTER_API_KEY is not set"
        );
        assert_eq!(
            format!("{}", Error::MissingAnswers),
            "response has no \"answers\" object"
        );
        assert_eq!(
            format!(
                "{}",
                Error::MissingAnswer {
                    question: "is_bug".to_string()
                }
            ),
            "answer missing for question \"is_bug\""
        );
        assert_eq!(
            format!(
                "{}",
                Error::Wire {
                    message: "bad type".to_string()
                }
            ),
            "malformed wire payload: bad type"
        );
        assert_eq!(format!("{}", Error::Internal("boom".to_string())), "boom");
    }

    #[test]
    fn display_remote_and_internal_helper() {
        assert_eq!(
            format!(
                "{}",
                Error::Remote {
                    status: StatusCode::UNAUTHORIZED,
                    message: "bad key".to_string()
                }
            ),
            "OpenRouter responded 401 Unauthorized: bad key"
        );
        // `internal()` builds the catch-all variant for dynamic messages
        assert_eq!(format!("{}", Error::internal("boom".to_string())), "boom");
    }

    #[test]
    fn serialization_wraps_serde_json_failures() {
        // `?` on a `serde_json::Result` lands in `Error::Serialization` via
        // the derived `#[from]` impl, and Display keeps the "serialize state:"
        // prefix of the variant.
        match parse_bad_number() {
            Ok(_) => panic!("expected an error"),
            Err(e) => {
                let text = format!("{e}");
                assert!(text.starts_with("serialize state: "), "got: {text}");
                assert!(
                    text.len() > "serialize state: ".len(),
                    "empty serde message"
                );
            }
        }
    }

    #[test]
    fn anyhow_failures_convert_to_external() {
        // the `?` bridge across anyhow::Result lands the failure in
        // `Error::External`, message preserved.
        match sink_anyhow() {
            Ok(_) => panic!("expected an error"),
            Err(Error::External(inner)) => {
                assert_eq!(format!("{}", inner), "load failed: disk");
            }
            Err(other) => panic!("expected External, got {}", other),
        }
    }

    fn parse_bad_number() -> Result<()> {
        let _ = serde_json::from_str::<u32>("not a number")?;
        Ok(())
    }

    fn fail_anyhow() -> anyhow::Result<()> {
        anyhow::bail!("load failed: disk");
    }

    fn sink_anyhow() -> Result<()> {
        fail_anyhow()?;
        Ok(())
    }
}
