//! Minimal Hugging Face Hub client — ureq/rustls only.
//!
//! Downloads model files into the standard hub v2 cache layout shared with
//! python `huggingface_hub` and the previous `hf-hub` crate, so a cache
//! filled by either tool stays valid:
//!
//! `~/.cache/huggingface/hub/models--<owner>--<name>/` with `refs/main`
//! (resolved commit id) and `snapshots/<id>/<path>` (one tree per commit).
//!
//! Only what the local backends need is implemented: files are requested
//! from `resolve/main`, the resolved commit id is read from the response's
//! `x-repo-commit` header (both the redirect and the final asset carry it),
//! and the file lands in the matching snapshots tree. No TLS stack beyond
//! ureq's rustls, so the OpenSSL/native-tls dependency tree stays out of
//! every build.

use anyhow::Result;
use std::fs::{File, create_dir_all, metadata, read_to_string, write};
use std::path::PathBuf;
use std::time::Duration;
use ureq::Agent;

const HUB_BASE: &str = "https://huggingface.co";
const CACHE_TIMEOUT_SECS: u64 = 600;

fn cache_root() -> PathBuf {
    match std::env::var("HF_HOME") {
        Ok(h) => PathBuf::from(h).join("hub"),
        _ => PathBuf::from(std::env::var("HOME").unwrap_or(".".to_string()))
            .join(".cache/huggingface/hub"),
    }
}

/// The v2 cache dir for a repo: `models--<owner>--<name>`.
fn repo_dir(model_id: &str) -> PathBuf {
    let key = model_id.replace('/', "--");
    cache_root().join(format!("models--{key}"))
}

fn hub_agent() -> Agent {
    Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(CACHE_TIMEOUT_SECS)))
        .build()
        .into()
}

/// The cached commit id from `refs/main`, if any (content-aware: the old
/// `hf-hub` crate wrote short ids, python writes full sha256s).
fn cached_commit(model_id: &str) -> Option<String> {
    let ref_path = repo_dir(model_id).join("refs/main");
    match read_to_string(&ref_path) {
        Ok(s) => {
            let c = s.trim();
            if c.is_empty() {
                None
            } else {
                Some(c.to_string())
            }
        }
        Err(_) => None,
    }
}

/// Path of `path_in_repo` inside a snapshots tree, if already downloaded.
fn cached_file(model_id: &str, commit: &str, path_in_repo: &str) -> Option<PathBuf> {
    let candidate = repo_dir(model_id)
        .join("snapshots")
        .join(commit)
        .join(path_in_repo);
    match metadata(&candidate) {
        Ok(_) => Some(candidate),
        Err(_) => None,
    }
}

/// Download (or reuse from the cache) one file of a public hub repo.
/// Returns the local path; callers keep the `.context(...)` messages.
pub(crate) fn hub_file(model_id: &str, path_in_repo: &str) -> Result<PathBuf> {
    match cached_commit(model_id) {
        Some(c) => match cached_file(model_id, &c, path_in_repo) {
            Some(p) => Ok(p),
            _ => Ok(download_file(model_id, path_in_repo)?),
        },
        _ => Ok(download_file(model_id, path_in_repo)?),
    }
}

fn download_file(model_id: &str, path_in_repo: &str) -> Result<PathBuf> {
    let url = format!("{}/{}/resolve/main/{}", HUB_BASE, model_id, path_in_repo);
    let mut resp = hub_agent().get(url.clone()).call()?;
    let commit = resp
        .headers()
        .get_all("x-repo-commit")
        .iter()
        .filter_map(|h| h.to_str().ok())
        .next()
        .map(|s| s.to_string())
        .unwrap_or("main".to_string());

    let dest = repo_dir(model_id)
        .join("snapshots")
        .join(commit.clone())
        .join(path_in_repo);
    if let Some(parent) = dest.parent() {
        create_dir_all(parent)?
    }
    let mut reader = resp.body_mut().as_reader();
    let mut file = File::create(dest.clone())?;
    std::io::copy(&mut reader, &mut file)?;

    // persist the resolved commit so later files skip the network round
    // trip (best effort — a stale ref only costs a re-download)
    let ref_path = repo_dir(model_id).join("refs/main");
    if let Some(ref_dir) = ref_path.parent() {
        create_dir_all(ref_dir)?
    }
    write(&ref_path, commit.as_bytes())?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cache_paths_are_v2_layout() {
        assert!(
            repo_dir("sevenreasons/von-onnx-fp16")
                .to_string_lossy()
                .ends_with(".cache/huggingface/hub/models--sevenreasons--von-onnx-fp16")
        );
        assert!(
            repo_dir("sevenreasons/von-onnx-fp16")
                .join("snapshots")
                .join("abc")
                .join("tokenizer/tokenizer.json")
                .to_string_lossy()
                .ends_with("snapshots/abc/tokenizer/tokenizer.json")
        );
    }
}
