//! Von ONNX backend — the in-repo System One NLI model.
//!
//! Mirrors the Python `von` using `ort` + `tokenizers`; the model is
//! `Zatsepin/von-onnx-fp16` (FP16, logits [batch,3], entail=0). It is an
//! ONNX export of the NLI-era `wfzyx/von` checkpoint (ModernBERT-large +
//! sequence-classification head, revision `999e01cfff98`) — see
//! `scripts/von/export_onnx.py`, artifact on HF as
//! `Zatsepin/von-onnx-fp16`.

use crate::hub::hub_file;
use anyhow::{Context, Result as AnyhowResult};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::{Answer, Backend, Error, Question, Result, state_text};

/// Decision result for `choice`
#[derive(Debug, Clone)]
struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f32,
    pub probabilities: HashMap<String, f32>,
}

/// A rate criterion, mirroring Python `str` or `{"what": ..., "examples": [...]}`.
#[derive(Debug, Clone)]
enum RateCriterion {
    /// Plain text description (Python `str`).
    Text(String),
    /// Python dict form: `what` description plus optional `examples` list.
    Description { what: String, examples: Vec<String> },
}

/// Von ONNX backend
pub struct VonBackend {
    session: ort::session::Session,
    tokenizer: Tokenizer,
    temperature: f32,
    entail_idx: usize,
    contra_idx: usize,
}

impl VonBackend {
    /// Load model from HF Hub (cached at ~/.cache/huggingface/hub like Python)
    pub fn new() -> Result<Self> {
        Self::new_with_model("Zatsepin/von-onnx-fp16")
    }

    pub fn new_with_model(model_id: &str) -> Result<Self> {
        let model_path = hub_file(model_id, "model.onnx").context("download model.onnx")?;
        let tokenizer_path =
            hub_file(model_id, "tokenizer/tokenizer.json").context("download tokenizer")?;

        // calibration
        let mut temperature = 1.0367_f32;
        if let Ok(calib_path) = hub_file(model_id, "calibration.json")
            && let Ok(s) = std::fs::read_to_string(&calib_path)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&s)
            && let Some(t) = v.get("temperature").and_then(|x| x.as_f64())
        {
            temperature = t as f32;
        }

        let tokenizer = Self::init_tokenizer(&tokenizer_path)?;
        let session = crate::init_session(&model_path)?;
        Ok(Self {
            session,
            tokenizer,
            temperature,
            entail_idx: 0,
            contra_idx: 2,
        })
    }

    /// For offline use with local paths (e.g. cached blobs)
    pub fn new_from_files(
        model_onnx: &str,
        tokenizer_json: &str,
        temperature: f32,
    ) -> Result<Self> {
        let tokenizer = Self::init_tokenizer(tokenizer_json)?;
        let session = crate::init_session(model_onnx)?;
        Ok(Self {
            session,
            tokenizer,
            temperature,
            entail_idx: 0,
            contra_idx: 2,
        })
    }

    /// Load tokenizer, truncate to 512 (LongestFirst) like Python `max_length=512`;
    /// tokenizer-side padding is disabled, we pad the batch manually.
    fn init_tokenizer<P: AsRef<Path>>(path: P) -> AnyhowResult<Tokenizer> {
        use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};
        let mut tokenizer =
            Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }));
        tokenizer.with_padding(None);
        Ok(tokenizer)
    }
}

impl VonBackend {
    fn tokenize_batch(
        &self,
        premises: &[String],
        hypotheses: &[String],
    ) -> AnyhowResult<(Array2<i64>, Array2<i64>)> {
        debug_assert_eq!(premises.len(), hypotheses.len());
        let batch = premises.len();
        let pad_id = self.tokenizer.token_to_id("[PAD]").unwrap_or(50283) as i64;

        let mut encodings: Vec<(Vec<i64>, Vec<i64>)> = Vec::with_capacity(batch);
        let mut max_len = 0usize;
        for (prem, hyp) in premises.iter().zip(hypotheses) {
            let enc = self
                .tokenizer
                .encode((prem.as_str(), hyp.as_str()), true)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            // after truncation, ensure max_length 512 (tokenizer already does LongestFirst)
            let mut ids: Vec<i64> = crate::i64_ids(enc.get_ids());
            let mut mask: Vec<i64> = crate::i64_ids(enc.get_attention_mask());
            // hard truncation to 512 if still longer (tokenizer truncation may not apply to pair correctly)
            if ids.len() > 512 {
                ids.truncate(512);
                mask.truncate(512);
            }
            max_len = max_len.max(ids.len());
            encodings.push((ids, mask));
        }
        // pad to max_len
        let mut input_ids = Array2::<i64>::zeros((batch, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, max_len));
        for (i, (ids, mask)) in encodings.into_iter().enumerate() {
            for (j, &v) in ids.iter().enumerate() {
                input_ids[[i, j]] = v;
            }
            for (j, &v) in mask.iter().enumerate() {
                attention_mask[[i, j]] = v;
            }
            // remaining already 0, but need pad_id for input_ids tail
            for j in ids.len()..max_len {
                input_ids[[i, j]] = pad_id;
            }
        }
        Ok((input_ids, attention_mask))
    }

    fn run_logits(
        &mut self,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
    ) -> AnyhowResult<ndarray::Array2<f32>> {
        use ort::value::Tensor;
        let input_ids_tensor = Tensor::from_array(input_ids)?;
        let attention_mask_tensor = Tensor::from_array(attention_mask)?;
        let outputs = self.session.run(ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor
        })?;
        crate::tensor_to_array2(&outputs["logits"], "von logits")
    }

    /// One batched NLI forward pass: temperature-scaled softmax over the
    /// entailment logits of each "(state, hypothesis)" pair.
    fn nli_probs(&mut self, state: &str, hyps: &[String], temp: f32) -> AnyhowResult<Vec<f32>> {
        let premises = vec![state.to_string(); hyps.len()];
        let (ids, mask) = self.tokenize_batch(&premises, hyps)?;
        let logits = self.run_logits(ids, mask)?;
        let entail: Vec<f32> = (0..logits.nrows())
            .map(|i| logits[[i, self.entail_idx]])
            .collect();
        Ok(crate::softmax(&entail, temp))
    }

    /// One VM's worth of a two-option judgment: two hypotheses (yes/no,
    /// true/false, positive/negative) batched against the same state and
    /// scored by entailment. Returns the (4-decimal) probability of the
    /// first hypothesis holding.
    fn two_way_judge(&mut self, state: &str, hyps: Vec<String>, temp: f32) -> AnyhowResult<f32> {
        let probs = self.nli_probs(state, &hyps, temp)?;
        Ok(crate::round4(probs[0]))
    }

    /// Mirrors `von.decide`
    fn decide(
        &mut self,
        state: &str,
        choices: &HashMap<String, Option<String>>,
        instructions: &str,
        temperature: Option<f32>,
    ) -> AnyhowResult<ChoiceAnswer> {
        let temp = temperature.unwrap_or(self.temperature);
        let options: Vec<String> = choices.keys().cloned().collect();
        if options.is_empty() {
            return Ok(ChoiceAnswer {
                choice: "".into(),
                confidence: 0.0,
                probabilities: HashMap::new(),
            });
        }
        let mut hypotheses = Vec::with_capacity(options.len());
        for opt in &options {
            let desc = choices.get(opt).and_then(|v| v.as_ref());
            let text = if let Some(d) = desc {
                format!("{instructions} {d}")
            } else {
                format!("{instructions} {opt}")
            };
            hypotheses.push(text);
        }
        let probs = self.nli_probs(state, &hypotheses, temp)?;
        // argmax over the entailment scores; softmax is monotonic, so the
        // probability ranking matches the raw-score ranking
        let best_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut prob_map = HashMap::new();
        for (opt, p) in options.iter().zip(probs.iter()) {
            prob_map.insert(opt.clone(), crate::round4(*p));
        }
        let mut sorted: Vec<f32> = prob_map.values().cloned().collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let conf = if sorted.len() > 1 {
            sorted[0] - sorted[1]
        } else {
            sorted[0]
        };
        let conf = (conf * 1000.0).round() / 1000.0;
        let conf = conf.clamp(0.0, 1.0);
        Ok(ChoiceAnswer {
            choice: options[best_idx].clone(),
            confidence: conf,
            probabilities: prob_map,
        })
    }

    /// Mirrors `von.judge` — returns noul probability
    fn judge(
        &mut self,
        state: &str,
        instructions: &str,
        criteria: Option<&HashMap<String, String>>,
        temperature: Option<f32>,
    ) -> AnyhowResult<f32> {
        let temp = temperature.unwrap_or(self.temperature);
        let pos = criteria
            .and_then(|c| c.get("true"))
            .map(|s| s.as_str())
            .unwrap_or("");
        let neg = criteria
            .and_then(|c| c.get("false"))
            .map(|s| s.as_str())
            .unwrap_or("");
        if !pos.is_empty() && !neg.is_empty() {
            let hyps = vec![
                format!("{instructions} {pos}").trim().to_string(),
                format!("{instructions} {neg}").trim().to_string(),
            ];
            return self.two_way_judge(state, hyps, temp);
        }
        let s = instructions.trim();
        let is_question = s.ends_with('?') || {
            let l = s.to_lowercase();
            l.starts_with("is ")
                || l.starts_with("are ")
                || l.starts_with("does ")
                || l.starts_with("do ")
                || l.starts_with("can ")
                || l.starts_with("could ")
                || l.starts_with("should ")
        };
        if is_question {
            let q_clean = s.trim_end_matches('?');
            let hyps = vec![format!("{q_clean}? Yes."), format!("{q_clean}? No.")];
            return self.two_way_judge(state, hyps, temp);
        }
        // declarative: entail vs contra sigmoid
        let hyp = if !pos.is_empty() {
            format!("{s} {pos}").trim().to_string()
        } else {
            s.to_string()
        };
        let (ids, mask) = self.tokenize_batch(&[state.to_string()], &[hyp])?;
        let logits = self.run_logits(ids, mask)?;
        let ent = logits[[0, self.entail_idx]];
        let con = logits[[0, self.contra_idx]];
        let diff = (ent - con) / temp.max(1e-4);
        let mut p = 1.0 / (1.0 + (-diff).exp());
        if !neg.is_empty() && pos.is_empty() {
            p = 1.0 - p;
        }
        p = crate::round4(p);
        Ok(p.clamp(0.0, 1.0))
    }

    /// Mirrors `von.rate`
    #[allow(clippy::type_complexity)]
    fn rate(
        &mut self,
        state: &str,
        criteria: &[RateCriterion],
        instructions: &str,
        temperature: Option<f32>,
    ) -> AnyhowResult<(f32, f32, HashMap<String, f32>, HashMap<String, String>)> {
        let temp = temperature.unwrap_or(self.temperature);
        if criteria.is_empty() {
            return Ok((0.0, 0.0, HashMap::new(), HashMap::new()));
        }
        let mut legend = HashMap::new();
        let mut hyps = Vec::with_capacity(criteria.len());
        let inst_clean = instructions.trim();
        let is_how = inst_clean.to_lowercase().starts_with("how ");
        for (i, item) in criteria.iter().enumerate() {
            let desc = match item {
                RateCriterion::Text(t) => t.trim().to_string(),
                RateCriterion::Description { what, examples } => {
                    let mut d = what.clone();
                    if !examples.is_empty() {
                        d = format!("{d} Examples: {}", examples.join(", "));
                    }
                    d.trim().to_string()
                }
            };
            legend.insert(i.to_string(), desc.clone());
            let h = if is_how {
                format!("The condition is {desc}")
            } else if !inst_clean.is_empty() {
                format!("{inst_clean} {desc}")
            } else {
                desc
            };
            hyps.push(h);
        }
        let probs = self.nli_probs(state, &hyps, temp)?;
        let mut prob_map = HashMap::new();
        for (i, p) in probs.iter().enumerate() {
            prob_map.insert(i.to_string(), crate::round4(*p));
        }
        let score: f32 = probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum();
        let score = (score * 100.0).round() / 100.0;
        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let conf = if sorted.len() > 1 {
            sorted[0] - sorted[1]
        } else {
            sorted[0]
        };
        let conf = (conf * 1000.0).round() / 1000.0;
        Ok((score, conf.clamp(0.0, 1.0), prob_map, legend))
    }
}

// ----------------------------------------------------------------------------
// The System One wire protocol glue for von.
// ----------------------------------------------------------------------------

fn parse_choice_criteria(
    v: &serde_json::Value,
) -> std::result::Result<HashMap<String, Option<String>>, String> {
    match v {
        serde_json::Value::Object(m) => {
            let mut out = HashMap::new();
            for (key, val) in m {
                match val {
                    serde_json::Value::String(s) => {
                        out.insert(key.clone(), Some(s.clone()));
                    }
                    serde_json::Value::Null => {
                        out.insert(key.clone(), None);
                    }
                    _ => {
                        return Err(format!(
                            "choice \"{key}\" description must be a string or null"
                        ));
                    }
                }
            }
            Ok(out)
        }
        serde_json::Value::Array(items) => {
            let mut out = HashMap::new();
            for item in items {
                match item {
                    serde_json::Value::String(s) => {
                        out.insert(s.clone(), None);
                    }
                    _ => return Err("choice names in list form must be strings".to_string()),
                }
            }
            Ok(out)
        }
        _ => Err("choice \"criteria\" must be an object or an array".to_string()),
    }
}

fn parse_score_criteria(v: &serde_json::Value) -> std::result::Result<Vec<RateCriterion>, String> {
    let items = match v.as_array() {
        Some(a) => a,
        _ => return Err("score \"criteria\" must be an array".to_string()),
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            serde_json::Value::String(s) => {
                out.push(RateCriterion::Text(s.clone()));
            }
            serde_json::Value::Object(m) => {
                let what = m
                    .get("what")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let examples: Vec<String> = m
                    .get("examples")
                    .and_then(serde_json::Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(serde_json::Value::as_str)
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                out.push(RateCriterion::Description { what, examples });
            }
            _ => return Err("score criteria items must be strings or objects".to_string()),
        }
    }
    Ok(out)
}

/// The shared `Backend` interface: noul/choice/score questions in, typed
/// answers out (noul -> judge, choice -> decide, score -> rate).
impl Backend for VonBackend {
    fn answers(
        &mut self,
        state: &serde_json::Value,
        questions: &[Question],
        global_temp: Option<f32>,
    ) -> Result<HashMap<String, Answer>> {
        let text = state_text(state)?;
        let mut out = HashMap::new();
        for q in questions {
            let temp = q.temperature.or(global_temp);
            let answer: Answer = match q.kind.as_str() {
                "noul" => {
                    let sides =
                        crate::noul_sides(q.criteria.as_ref()).map_err(|e| Error::Wire {
                            message: format!("question {}: {e}", q.id),
                        })?;
                    let p = self.judge(&text, &q.instructions, Some(&sides), temp)?;
                    Answer::Noul { probability: p }
                }
                "choice" => {
                    let criteria = match &q.criteria {
                        Some(c) => match parse_choice_criteria(c) {
                            Ok(m) => m,
                            Err(e) => {
                                return Err(Error::Wire {
                                    message: format!("question {}: {e}", q.id),
                                });
                            }
                        },
                        None => {
                            return Err(Error::Wire {
                                message: format!("question {}: choice needs criteria", q.id),
                            });
                        }
                    };
                    let ans = self.decide(&text, &criteria, &q.instructions, temp)?;
                    Answer::Choice {
                        choice: ans.choice,
                        confidence: ans.confidence,
                        probabilities: ans.probabilities,
                    }
                }
                "score" => {
                    let criteria = match &q.criteria {
                        Some(c) => match parse_score_criteria(c) {
                            Ok(m) => m,
                            Err(e) => {
                                return Err(Error::Wire {
                                    message: format!("question {}: {e}", q.id),
                                });
                            }
                        },
                        None => {
                            return Err(Error::Wire {
                                message: format!("question {}: score needs criteria", q.id),
                            });
                        }
                    };
                    let (value, conf, probs, legend) =
                        self.rate(&text, &criteria, &q.instructions, temp)?;
                    Answer::Score {
                        score: value,
                        confidence: conf,
                        probabilities: probs,
                        legend,
                    }
                }
                other => {
                    return Err(Error::Wire {
                        message: format!("question {}: unknown kind {other}", q.id),
                    });
                }
            };
            out.insert(q.id.clone(), answer);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[tokio::test]
    async fn parse_choice_criteria_object() {
        let v = json!({"infrastructure": "DB failures", "billing": null});
        let c = parse_choice_criteria(&v).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(
            c.get("infrastructure").unwrap().as_ref().unwrap(),
            "DB failures"
        );
        assert!(c.get("billing").unwrap().is_none());
        assert!(parse_choice_criteria(&json!("not a map")).is_err());
        assert!(parse_choice_criteria(&json!({"a": 42})).is_err());
    }

    #[tokio::test]
    async fn parse_score_criteria_strings_and_dicts() {
        let v = json!([
            "Plain text",
            {"what": "Critical", "examples": ["OOM killer", "latency > 5s"]},
            {"what": "No examples"}
        ]);
        let rc = parse_score_criteria(&v).unwrap();
        assert_eq!(rc.len(), 3);
        match &rc[0] {
            RateCriterion::Text(s) => assert_eq!(s, "Plain text"),
            _ => panic!("expected Text"),
        }
        match &rc[1] {
            RateCriterion::Description { what, examples } => {
                assert_eq!(what, "Critical");
                assert_eq!(
                    examples,
                    &vec!["OOM killer".to_string(), "latency > 5s".to_string()]
                );
            }
            _ => panic!("expected Description"),
        }
        assert!(parse_score_criteria(&json!([42])).is_err());
        assert!(parse_score_criteria(&json!("nope")).is_err());
    }
}
