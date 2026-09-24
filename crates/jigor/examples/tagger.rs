//! Tagger — auto-tags a note with one of its existing tags.
//!
//! Demonstrates a `choice` question on the library's decision backends,
//! mirroring a simple auto-tagger: a note (title + body) and a list of
//! existing tags go in, the best-matching tag comes out with a probability
//! distribution. A "None of these fit well" fallback option keeps the model
//! honest when no existing tag matches — the caller can then create a new
//! tag instead of forcing a bad fit.
//!
//! ```bash
//! cargo run -p jigor --example tagger -- --title "Q4 planning" --tags "work, ideas, personal" "Talked through the roadmap and the Q4 hiring push."
//! cargo run -p jigor --example tagger -- --model jev --tags "bugs, docs, ship" "Fixed the retry loop that dropped webhook events."
//! # laya folds a multi-tag choice into 2-option pairwise ballots automatically,
//! # so any number of tags works out of the box
//! cargo run -p jigor --example tagger -- --model laya --tags "work, ideas, personal" "Talked through the roadmap and the Q4 hiring push."
//! cargo run -p jigor --example tagger -- --json --tags "a, b" "note text"
//! ```

use jigor::{Answer, Asks, Error, Question, Result, answer_to_json, resolve_model};
use serde_json::{Map, Value, json};
use std::collections::HashMap;

/// Fallback option appended to the criteria so the model can say "no tag".
const FALLBACK_TAG: &str = "None of these fit well";
const FALLBACK_DESC: &str = "No existing tag fits this note";

/// The picked tag plus the model's distribution over all options.
#[derive(Debug, Clone)]
pub struct TagChoice {
    pub tag: String,
    pub confidence: f32,
    pub probabilities: HashMap<String, f32>,
}

/// The note the tagger judges: title and body, like the reference client.
pub fn note_state(title: &str, body: &str) -> Value {
    json!({"note_title": title, "note_body": body})
}

/// One choice question: pick the existing tag that best matches the note.
/// Each tag is an option without a description; the fallback carries one.
pub fn tag_questions(tags: &[String], include_fallback: bool) -> Vec<Question> {
    let mut criteria = Map::new();
    for tag in tags {
        criteria.insert(tag.clone(), Value::Null);
    }
    if include_fallback {
        criteria.insert(FALLBACK_TAG.to_string(), json!(FALLBACK_DESC));
    }
    vec![Question {
        id: "best_tag".to_string(),
        kind: "choice".to_string(),
        instructions: "Which tag best matches the content of this note?".to_string(),
        criteria: Some(Value::Object(criteria)),
        temperature: None,
    }]
}

/// Read the `best_tag` answer back out of the typed answers map.
pub fn tag_choice_from(asks: &HashMap<String, Answer>) -> std::result::Result<TagChoice, String> {
    match asks.get("best_tag") {
        Some(Answer::Choice {
            choice,
            confidence,
            probabilities,
        }) => Ok(TagChoice {
            tag: choice.clone(),
            confidence: *confidence,
            probabilities: probabilities.clone(),
        }),
        _ => Err("answer missing or not a choice (question \"best_tag\")".to_string()),
    }
}

/// Ask the tagger question through the chosen backend; same wire in, same
/// typed answers out for `local` (von, laya — picked by model id) and
/// `openrouter` (Jev).
pub fn ask_tagger(
    title: &str,
    body: &str,
    tags: &[String],
    model: &str,
    include_fallback: bool,
) -> Result<Asks> {
    let state = note_state(title, body);
    let questions = tag_questions(tags, include_fallback);
    jigor::ask(model, &state, &questions, None)
}

fn main() -> Result<()> {
    let mut parts: Vec<String> = Vec::new();
    let mut title = String::from("Untitled");
    let mut tags_raw = String::from("");
    let mut as_json = false;
    let mut no_fallback = false;
    let mut model_hint: Option<String> = None;
    let mut provider_hint: Option<String> = None;
    let mut args = std::env::args()
        .skip(1)
        .collect::<Vec<String>>()
        .into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--title" => {
                title = args
                    .next()
                    .ok_or_else(|| Error::internal("--title requires a value".to_string()))?
                    .to_string();
            }
            "--tags" => {
                tags_raw = args
                    .next()
                    .ok_or_else(|| {
                        Error::internal("--tags requires a comma-separated list".to_string())
                    })?
                    .to_string();
            }
            "--model" => {
                model_hint = Some(
                    args.next()
                        .ok_or_else(|| Error::internal("--model requires a value".to_string()))?
                        .to_string(),
                );
            }
            "--provider" => {
                provider_hint = Some(
                    args.next()
                        .ok_or_else(|| Error::internal("--provider requires a value".to_string()))?
                        .to_string(),
                );
            }
            "--json" => as_json = true,
            "--no-fallback" => no_fallback = true,
            "--help" | "-h" => {
                println!(
                    "usage: tagger [--title <t>] --tags <t1,t2,..> [--model <id>] [--provider <p>] [--json] [--no-fallback] \"<note body>\""
                );
                std::process::exit(0);
            }
            _other => parts.push(arg.to_string()),
        }
    }
    let body = parts.join(" ");
    if body.is_empty() || tags_raw.is_empty() {
        return Err(Error::internal(
            "usage: tagger --tags <t1,t2,..> \"<note body>\" (see --help)".to_string(),
        ));
    }

    let mut tags: Vec<String> = Vec::new();
    for raw in tags_raw.split(",") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            tags.push(trimmed.to_string());
        }
    }
    if tags.is_empty() {
        return Err(Error::internal(
            "--tags needs at least one tag name".to_string(),
        ));
    }

    let model = match model_hint {
        Some(m) => m,
        None => "von-1.0.0".to_string(),
    };
    let provider_arg: Option<&str> = match &provider_hint {
        Some(p) => Some(p.as_str()),
        None => None,
    };
    let (_, resolved_model) = match resolve_model(provider_arg, &model) {
        Ok(pair) => pair,
        Err(e) => return Err(Error::internal(e)),
    };

    let asks = ask_tagger(&title, &body, &tags, &resolved_model, !no_fallback)?;
    let picked = match tag_choice_from(&asks.answers) {
        Ok(c) => c,
        Err(e) => return Err(Error::internal(e)),
    };

    if as_json {
        let mut wire = Map::new();
        for (qid, answer) in asks.answers {
            wire.insert(qid.clone(), answer_to_json(&answer));
        }
        let out = json!({"model": asks.model, "backend": asks.backend, "answers": wire});
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("Note: {}", title);
        println!("backend: {}  model: {}", asks.backend, asks.model);
        if picked.tag == FALLBACK_TAG {
            println!("No existing tag fits this note — consider creating a new one.");
            print_probabilities(&picked);
        } else {
            println!(
                "Best tag: {}  (confidence {:.3})",
                picked.tag, picked.confidence
            );
            print_probabilities(&picked);
        }
    }
    Ok(())
}

fn print_probabilities(picked: &TagChoice) {
    let mut options: Vec<(String, f32)> = Vec::with_capacity(picked.probabilities.len());
    for (tag, p) in &picked.probabilities {
        options.push((tag.clone(), *p));
    }
    options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut line: Vec<String> = Vec::with_capacity(options.len());
    for (tag, p) in options {
        line.push(format!("{} {}%", tag, (p * 100.0).round() as u32));
    }
    println!("  {}", line.join("  ·  "));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(names: Vec<&str>) -> Vec<String> {
        names.iter().map(|t| t.to_string()).collect::<Vec<String>>()
    }

    #[tokio::test]
    async fn questions_have_fallback_by_default() {
        let qs = tag_questions(&tags(vec!["work", "ideas"]), true);
        assert_eq!(qs.len(), 1);
        let criteria = qs[0].criteria.as_ref().unwrap().as_object().unwrap();
        assert!(criteria.contains_key("work"));
        assert!(criteria.contains_key("ideas"));
        assert_eq!(criteria.get("work").unwrap().clone(), Value::Null);
        assert_eq!(
            criteria.get(FALLBACK_TAG).unwrap().clone(),
            json!(FALLBACK_DESC)
        );
        assert_eq!(
            qs[0].instructions,
            "Which tag best matches the content of this note?"
        );
        assert_eq!(qs[0].kind, "choice");
    }

    #[tokio::test]
    async fn fallback_can_be_turned_off() {
        let qs = tag_questions(&tags(vec!["work"]), false);
        let criteria = qs[0].criteria.as_ref().unwrap().as_object().unwrap();
        assert!(!criteria.contains_key(FALLBACK_TAG));
    }

    #[tokio::test]
    async fn tag_choice_reads_the_answer_back() {
        let mut probabilities = HashMap::new();
        probabilities.insert("work".to_string(), 0.62_f32);
        probabilities.insert("ideas".to_string(), 0.38_f32);
        let mut asks = HashMap::new();
        asks.insert(
            "best_tag".to_string(),
            Answer::Choice {
                choice: "work".to_string(),
                confidence: 0.24,
                probabilities,
            },
        );
        let picked = tag_choice_from(&asks).unwrap();
        assert_eq!(picked.tag, "work");
        assert_eq!(picked.confidence, 0.24);
        assert_eq!(picked.probabilities.get("work").unwrap(), &0.62);
        assert!(tag_choice_from(&HashMap::new()).is_err());
    }
}
