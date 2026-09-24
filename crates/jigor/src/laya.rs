//! Laya ONNX backend — the local marker-scoring decision model.
//!
//! ONNX export of convaiinnovations/laya (a non-autoregressive System 1
//! decision model: ModernBERT-large + a marker-scoring decision head).
//! Inputs:
//!
//!   input_ids      [batch, seq]    i64
//!   attention_mask [batch, seq]    i64
//!   marker_pos     [batch, markers] i64   positions of each [MASK] token
//!   marker_mask    [batch, markers] bool  (padded to the widest question)
//!   qtype          [batch]         i64    0=choice, 1=score, 2=noul
//!
//! Output: logits [batch, markers]; softmax each question's slice, then
//! calibrate with the checkpoint's fitted per-(type, option count) temps.
//! Mirrors `laya/onnx_agent.py` + `laya/common.py` in the upstream repo:
//!
//!   [CLS] <qtype> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1
//!   ... [SEP] <state> [SEP], 512 tokens max, 192 for the option head.

use crate::hub::hub_file;
use anyhow::{Context, Result as AnyhowResult, bail};
use ndarray::Array2;
use std::collections::HashMap;
use tokenizers::Tokenizer;

use crate::{Answer, Backend, Error, Question, Result, init_session, state_text};
use serde_json::Value;

const LAYA_MAX_LEN: usize = 512;
const LAYA_HEAD_MAX_LEN: usize = 192;
const LAYA_OPTION_TOKENS: usize = 48;
const LAYA_TEMP_MIN: f32 = 0.5;
const LAYA_TEMP_MAX: f32 = 5.0;

/// Fitted calibration temperatures and per-(type, option-count) scales from
/// `convaiinnovations/laya` (rl_agent_config.json); these are the values the
/// ONNX export was fine-tuned with. `rl_agent_config.json` in the ONNX repo
/// is read first when present, so a re-fitted checkpoint wins at load time.
#[allow(clippy::excessive_precision)]
fn default_laya_temps() -> ([f32; 3], HashMap<String, f32>) {
    let temperature = [
        clamp_temperature(1.6369030475616455),
        clamp_temperature(1.2514300346374512),
        clamp_temperature(1.983399510383606),
    ];
    let by_options = HashMap::from([
        (
            "choice:2".to_string(),
            clamp_temperature(1.9063563346862793),
        ),
        (
            "choice:3-5".to_string(),
            clamp_temperature(1.7601518630981445),
        ),
        (
            "choice:6-10".to_string(),
            clamp_temperature(1.0000158548355103),
        ),
        (
            "choice:11+".to_string(),
            clamp_temperature(0.10058280825614929),
        ),
        (
            "score:3-5".to_string(),
            clamp_temperature(1.2514300346374512),
        ),
        ("noul:2".to_string(), clamp_temperature(1.983399510383606)),
    ]);
    (temperature, by_options)
}

/// Clamp like `laya.common.clamp_temperature`: an unusable temperature
/// distorts confidence, so refuse it (defaults to 1.0) and pin the rest to
/// [0.5, 5.0].
fn clamp_temperature(t: f32) -> f32 {
    if !t.is_finite() {
        return 1.0;
    }
    t.clamp(LAYA_TEMP_MIN, LAYA_TEMP_MAX)
}

/// Python `json.dumps(v, separators=(", ", ": "), ensure_ascii=False)` —
/// structural separators keep their spaces, strings are JSON-escaped.
fn py_json(v: &Value) -> String {
    match v {
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        Value::Object(m) => {
            let mut parts = Vec::with_capacity(m.len());
            for (k, val) in m {
                parts.push(format!(
                    "{}: {}",
                    serde_json::to_string(k).unwrap_or_default(),
                    py_json(val)
                ));
            }
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_json).collect();
            format!("[{}]", parts.join(", "))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Render one criterion value as text (upstream `render_criterion`):
/// strings pass through; structured values read as compact JSON.
fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_json(other),
    }
}

/// The `false:`/`true:` half of a noul question (upstream `_resolve_noul_labels`
/// with the default label map).
fn noul_side(criteria: Option<&Value>, key: &str, fallback: &str) -> String {
    match criteria.and_then(|c| c.get(key)) {
        Some(v) if !matches!(v, Value::Null) && !matches!(v, Value::String(s) if s.is_empty()) => {
            render_criterion(v)
        }
        _ => fallback.to_string(),
    }
}

/// One Laya question ready for a batched forward pass: (qtype, ids, markers).
struct LayaItem {
    qtype: i64,
    ids: Vec<i64>,
    markers: Vec<usize>,
}

/// The local ONNX export of `convaiinnovations/laya` from HF
/// (`Mattepiu/laya-onnx`).
pub struct LayaBackend {
    session: ort::session::Session,
    tokenizer: Tokenizer,
    temperature: [f32; 3],
    temperature_by_options: HashMap<String, f32>,
    cls_id: i64,
    sep_id: i64,
    mask_id: i64,
    pad_id: i64,
}

/// qtype input id: 0=choice, 1=score, 2=noul (matches `laya.common.QTYPES`).
fn laya_qtype(kind: &str) -> i64 {
    match kind {
        "choice" => 0,
        "score" => 1,
        _ => 2,
    }
}

/// Per-(question type, option count) temperature bucket: `choice:2`,
/// `choice:3-5`, ..., with a fallback to the plain type temperature.
fn temperature_scale(
    kind: &str,
    k: usize,
    temperature: &[f32; 3],
    temperature_by_options: &HashMap<String, f32>,
) -> f32 {
    let bucket = format!("{}:{}", kind, LayaBackend::size_bucket(k));
    temperature_by_options
        .get(&bucket)
        .copied()
        .unwrap_or_else(|| temperature[laya_qtype(kind) as usize])
}

impl LayaBackend {
    /// Load model and tokenizer from HF Hub (cached like Python), same repo
    /// layout as the python export: `laya.onnx` (+ `laya.onnx.data` at the
    /// root, matched against the python reference bit-for-bit),
    /// `tokenizer.json` alongside. This export bakes in a single-question
    /// batch, so multi-question asks fall back to one forward pass per
    /// question. Set `LAYA_ONNX_FILE` to pick another export (e.g.
    /// `int8/laya_int8.onnx` for CPU, `fp16_onlygpu_unverified/
    /// laya_fp16.onnx` for GPU).
    pub fn new() -> Result<Self> {
        Self::new_with_model("Mattepiu/laya-onnx")
    }

    pub fn new_with_model(model_id: &str) -> Result<Self> {
        let onnx_file = std::env::var("LAYA_ONNX_FILE").unwrap_or_else(|_| "laya.onnx".to_string());
        let model_path = hub_file(model_id, &onnx_file)
            .context("download laya.onnx (LAYA_ONNX_FILE to override)")?;
        if let Some(data) = onnx_file.strip_suffix(".onnx") {
            // external-data exports need their <name>.onnx.data next to them
            let _ = hub_file(model_id, &format!("{data}.onnx.data"));
        }
        let tokenizer_path = hub_file(model_id, "tokenizer.json").context("download tokenizer")?;

        let (mut temperature, mut temperature_by_options) = default_laya_temps();
        if let Ok(cfg_path) = hub_file(model_id, "rl_agent_config.json")
            && let Ok(s) = std::fs::read_to_string(&cfg_path)
            && let Ok(v) = serde_json::from_str::<Value>(&s)
        {
            if let Some(ts) = v.get("temperature").and_then(Value::as_array) {
                for (i, t) in ts.iter().enumerate().take(3) {
                    if let Some(f) = t.as_f64() {
                        temperature[i] = clamp_temperature(f as f32);
                    }
                }
            }
            if let Some(by) = v.get("temperature_by_options").and_then(Value::as_object) {
                for (k, val) in by {
                    if let Some(f) = val.as_f64() {
                        temperature_by_options.insert(k.clone(), clamp_temperature(f as f32));
                    }
                }
            }
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let session = init_session(&model_path)?;
        let cls_id = tokenizer.token_to_id("[CLS]").unwrap_or(1) as i64;
        let sep_id = tokenizer.token_to_id("[SEP]").unwrap_or(2) as i64;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(50284) as i64;
        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0) as i64;
        Ok(Self {
            session,
            tokenizer,
            temperature,
            temperature_by_options,
            cls_id,
            sep_id,
            mask_id,
            pad_id,
        })
    }

    /// For offline use with local paths (e.g. cached blobs); calibration
    /// temperatures stay at the checkpoint's fitted defaults.
    pub fn new_from_files(model_onnx: &str, tokenizer_json: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let session = init_session(model_onnx)?;
        let (temperature, temperature_by_options) = default_laya_temps();
        let cls_id = tokenizer.token_to_id("[CLS]").unwrap_or(1) as i64;
        let sep_id = tokenizer.token_to_id("[SEP]").unwrap_or(2) as i64;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(50284) as i64;
        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0) as i64;
        Ok(Self {
            session,
            tokenizer,
            temperature,
            temperature_by_options,
            cls_id,
            sep_id,
            mask_id,
            pad_id,
        })
    }

    /// Option texts in label-index order, upstream `render_options`.
    /// `noul` semantic order is always [false, true]; score levels are
    /// positional ("level 0", "level 1", ...); choice keys carry their
    /// description ("key: desc" unless the description is empty/null).
    fn render_options(&self, kind: &str, criteria: Option<&Value>) -> AnyhowResult<Vec<String>> {
        match kind {
            "choice" => {
                let m = match criteria {
                    Some(Value::Object(m)) => m,
                    _ => bail!("laya question needs choice criteria (an object)"),
                };
                if m.is_empty() {
                    bail!("laya choice needs at least one option");
                }
                let mut opts = Vec::with_capacity(m.len());
                for (k, v) in m {
                    let opt = match v {
                        Value::Null => k.clone(),
                        Value::String(s) if s.is_empty() => k.clone(),
                        Value::String(s) => format!("{k}: {s}"),
                        other => format!("{k}: {}", render_criterion(other)),
                    };
                    opts.push(opt);
                }
                Ok(opts)
            }
            "score" => {
                let items = match criteria {
                    Some(Value::Array(items)) => items,
                    _ => bail!("laya question needs score criteria (a list)"),
                };
                if items.is_empty() {
                    bail!("laya score needs at least one level");
                }
                Ok(items
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                    .collect())
            }
            "noul" => Ok(vec![
                format!(
                    "false: {}",
                    noul_side(criteria, "false", "no, the statement does not hold")
                ),
                format!(
                    "true: {}",
                    noul_side(criteria, "true", "yes, the statement holds")
                ),
            ]),
            other => bail!("laya question: unknown kind {other}"),
        }
    }

    /// Upstream `build_sequence`:
    /// `[CLS] <qtype> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1
    /// ... [SEP] <state> [SEP]`, each option capped at 48 tokens, the
    /// instruction head sharing a 192-token budget with the options, and the
    /// state filling the remaining room of the 512-token window.
    fn build_sequence(
        &self,
        state: &str,
        kind: &str,
        instructions: &str,
        options: &[String],
    ) -> (Vec<i64>, Vec<usize>) {
        let head = format!("{kind} question: {instructions}").replace("[MASK]", " ");
        let mut head_ids: Vec<i64> = self
            .tokenizer
            .encode(head, false)
            .map(|enc| crate::i64_ids(enc.get_ids()))
            .unwrap_or_default();

        let mut opt_ids: Vec<Vec<i64>> = Vec::with_capacity(options.len());
        for opt in options {
            let text = format!(" {opt}").replace("[MASK]", " ");
            let mut body: Vec<i64> = self
                .tokenizer
                .encode(text, false)
                .map(|enc| crate::i64_ids(enc.get_ids()))
                .unwrap_or_default();
            body.truncate(LAYA_OPTION_TOKENS);
            let mut o = vec![self.mask_id];
            o.extend(body);
            opt_ids.push(o);
        }

        let mut opt_budget = {
            let used: isize = opt_ids.iter().map(|o| o.len() as isize).sum();
            LAYA_HEAD_MAX_LEN as isize - used
        };
        if opt_budget < 16 {
            // options all but ate the head: shrink each so the model still
            // sees something beyond them (upstream per-option fallback)
            let per = ((LAYA_HEAD_MAX_LEN as isize - 16) / options.len().max(1) as isize).max(4);
            for o in opt_ids.iter_mut() {
                o.truncate(per as usize);
            }
            opt_budget = LAYA_HEAD_MAX_LEN as isize
                - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        }
        head_ids.truncate(opt_budget.max(0) as usize + 8);

        let mut ids = vec![self.cls_id];
        ids.extend(head_ids);
        ids.push(self.sep_id);
        let mut markers = Vec::with_capacity(opt_ids.len());
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend(o.iter().cloned());
        }
        ids.push(self.sep_id);

        let room = LAYA_MAX_LEN.saturating_sub(ids.len() + 1);
        let mut st: Vec<i64> = self
            .tokenizer
            .encode(state.replace("[MASK]", " "), false)
            .map(|enc| crate::i64_ids(enc.get_ids()))
            .unwrap_or_default();
        st.truncate(room);
        ids.extend(st);
        ids.push(self.sep_id);
        ids.truncate(LAYA_MAX_LEN);
        markers.retain(|&m| m < LAYA_MAX_LEN);
        (ids, markers)
    }

    /// Option-count size bucket for the calibration map (upstream `temp_bucket`).
    fn size_bucket(k: usize) -> &'static str {
        match k {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        }
    }

    /// Upstream `confidence_from_probs`: 1 - normalized Shannon entropy over
    /// the question's probabilities.
    fn confidence_from_probs(p: &[f32], k: usize) -> f32 {
        if k < 2 {
            return 1.0;
        }
        let mut ent = 0.0;
        for &v in p.iter().take(k) {
            ent -= v * v.max(1e-12).ln();
        }
        (1.0 - ent / (k as f32).ln()).clamp(0.0, 1.0)
    }

    /// One batched forward pass; returns logits [questions, max markers].
    fn run(&mut self, items: &[LayaItem]) -> AnyhowResult<ndarray::Array2<f32>> {
        use ort::value::Tensor;
        let n = items.len();
        let max_len = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
        let kmax = items.iter().map(|it| it.markers.len()).max().unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((n, max_len));
        let mut attention_mask = Array2::<i64>::zeros((n, max_len));
        let mut marker_pos = Array2::<i64>::zeros((n, kmax));
        let mut marker_mask = ndarray::Array2::<bool>::from_elem((n, kmax), false);
        let mut qtype = ndarray::Array1::<i64>::zeros(n);

        for (i, it) in items.iter().enumerate() {
            qtype[i] = it.qtype;
            for (j, &v) in it.ids.iter().enumerate() {
                input_ids[[i, j]] = v;
                attention_mask[[i, j]] = 1;
            }
            for j in it.ids.len()..max_len {
                input_ids[[i, j]] = self.pad_id;
            }
            for (j, &m) in it.markers.iter().enumerate() {
                marker_pos[[i, j]] = m as i64;
                marker_mask[[i, j]] = true;
            }
        }

        let outputs = self.session.run(ort::inputs! {
            "input_ids" => Tensor::from_array(input_ids)?,
            "attention_mask" => Tensor::from_array(attention_mask)?,
            "marker_pos" => Tensor::from_array(marker_pos)?,
            "marker_mask" => Tensor::from_array(marker_mask)?,
            "qtype" => Tensor::from_array(qtype)?,
        })?;
        crate::tensor_to_array2(&outputs[0], "laya logits")
    }
}

/// The shared `Backend` interface: answers every question in one (or
/// several) forward passes (mirrors `ONNXAgent.system_one`). The export
/// scores exactly two option markers per question, so a `choice` with
/// more than two options is folded out of the box into one 2-option
/// ballot per pair and the wins are aggregated back into the same choice
/// distribution.
impl Backend for LayaBackend {
    fn answers(
        &mut self,
        state: &Value,
        questions: &[Question],
        global_temp: Option<f32>,
    ) -> Result<HashMap<String, Answer>> {
        struct Plan {
            kind: String,
            k: usize,
            scale: f32,
            row: Option<usize>,
            pairs: Vec<(usize, usize, usize)>,
        }

        let text = state_text(state)?;
        let mut items: Vec<LayaItem> = Vec::with_capacity(questions.len());
        let mut plans: Vec<Plan> = Vec::with_capacity(questions.len());
        for q in questions {
            let options = self.render_options(&q.kind, q.criteria.as_ref())?;
            let k = options.len();
            let override_temp = q.temperature.or(global_temp);
            let pair_scale = match override_temp {
                Some(t) => t,
                None => {
                    temperature_scale("choice", 2, &self.temperature, &self.temperature_by_options)
                }
            };
            if q.kind == "choice" && k > 2 {
                // pairwise fold: one 2-option ballot per (i, j) pair
                let mut pairs = Vec::with_capacity(k * (k - 1) / 2);
                for (a, b) in choice_pair_indices(k) {
                    let ballot = vec![options[a].clone(), options[b].clone()];
                    let (ids, markers) =
                        self.build_sequence(&text, "choice", &q.instructions, &ballot);
                    items.push(LayaItem {
                        qtype: laya_qtype("choice"),
                        ids,
                        markers,
                    });
                    pairs.push((items.len() - 1, a, b));
                }
                plans.push(Plan {
                    kind: q.kind.clone(),
                    k,
                    scale: pair_scale,
                    row: None,
                    pairs,
                });
                continue;
            }
            let (ids, markers) = self.build_sequence(&text, &q.kind, &q.instructions, &options);
            let mk = markers.len();
            // this export scores exactly two option markers per question;
            // `score` keeps its (2-level) ordinal semantics, `noul` is fixed
            if mk != 2 {
                return Err(Error::Wire {
                    message: format!(
                        "laya question \"{}\": this ONNX export scores exactly 2 options per question, but a {}-option {} was asked",
                        q.id, mk, q.kind
                    ),
                });
            }
            let scale = match override_temp {
                Some(t) => t,
                None => {
                    temperature_scale(&q.kind, mk, &self.temperature, &self.temperature_by_options)
                }
            };
            items.push(LayaItem {
                qtype: laya_qtype(&q.kind),
                ids,
                markers,
            });
            plans.push(Plan {
                kind: q.kind.clone(),
                k: mk,
                scale,
                row: Some(items.len() - 1),
                pairs: Vec::new(),
            });
        }
        let logits = match self.run(&items) {
            Ok(logits) => logits,
            Err(_) if items.len() > 1 => {
                // some exports (the root `laya.onnx`) bake in a single-
                // question batch; answer one forward pass per question there
                let mut rows = ndarray::Array2::<f32>::zeros((items.len(), 2));
                for (i, item) in items.iter().enumerate() {
                    let r = self
                        .run(&[LayaItem {
                            qtype: item.qtype,
                            ids: item.ids.clone(),
                            markers: item.markers.clone(),
                        }])
                        .map_err(|e| anyhow::anyhow!("laya per-question run failed: {e}"))?;
                    rows[[i, 0]] = r[[0, 0]];
                    rows[[i, 1]] = r[[0, 1]];
                }
                rows
            }
            Err(e) => return Err(Error::External(e)),
        };

        let mut out = HashMap::new();
        for (i, q) in questions.iter().enumerate() {
            let plan = &plans[i];
            let answer = if !plan.pairs.is_empty() {
                let mut wins = vec![0.0_f32; plan.k];
                for &(row, a, b) in &plan.pairs {
                    let z: Vec<f32> = (0..2).map(|j| logits[[row, j]] / plan.scale).collect();
                    let p = crate::softmax(&z, 1.0);
                    wins[a] += p[0];
                    wins[b] += p[1];
                }
                let probs = pair_wins_to_probs(&wins, plan.pairs.len());
                choice_answer(&criteria_keys(q.criteria.as_ref()), &probs, plan.k)
            } else {
                let row = plan.row.unwrap();
                let z: Vec<f32> = (0..plan.k).map(|j| logits[[row, j]] / plan.scale).collect();
                let p = crate::softmax(&z, 1.0);
                match plan.kind.as_str() {
                    "choice" => choice_answer(&criteria_keys(q.criteria.as_ref()), &p, plan.k),
                    "score" => {
                        let mut score = 0.0;
                        for (i, &v) in p.iter().enumerate() {
                            score += i as f32 * v;
                        }
                        let mut probs = HashMap::new();
                        let mut legend = HashMap::new();
                        if let Some(criteria) = q.criteria.as_ref().and_then(Value::as_array) {
                            for (i, &v) in p.iter().enumerate() {
                                probs.insert(i.to_string(), crate::round4(v));
                                if let Some(c) = criteria.get(i) {
                                    legend.insert(i.to_string(), render_criterion(c));
                                }
                            }
                        }
                        Answer::Score {
                            score: crate::round4(score),
                            confidence: crate::round4(LayaBackend::confidence_from_probs(
                                &p, plan.k,
                            )),
                            probabilities: probs,
                            legend,
                        }
                    }
                    _ => Answer::Noul {
                        probability: crate::round4(p[1]),
                    },
                }
            };
            out.insert(q.id.clone(), answer);
        }
        Ok(out)
    }
}

/// All (i, j) option pairs, i < j — one 2-option ballot per pair.
fn choice_pair_indices(k: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(k * (k - 1) / 2);
    for a in 0..k {
        for b in (a + 1)..k {
            out.push((a, b));
        }
    }
    out
}

/// Win-count distribution over the options after `pairs` pairwise ballots.
/// Every ballot contributes `p_a + p_b == 1.0` total, so the result sums to
/// 1.0 and reads as the same choice distribution the single-question path
/// reports.
fn pair_wins_to_probs(wins: &[f32], pairs: usize) -> Vec<f32> {
    wins.iter().map(|w| w / pairs.max(1) as f32).collect()
}

/// Choice keys from a question's criteria object, in map order.
fn criteria_keys(criteria: Option<&Value>) -> Vec<String> {
    criteria
        .and_then(Value::as_object)
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

/// Assemble an `Answer::Choice` from a probability distribution over the
/// criteria keys (shared by the single-2-marker and pairwise-fold paths).
fn choice_answer(keys: &[String], probs: &[f32], k: usize) -> Answer {
    let mut best = 0;
    for (bi, &v) in probs.iter().enumerate() {
        if v > probs[best] {
            best = bi;
        }
    }
    let mut prob_map = HashMap::new();
    for (key, &v) in keys.iter().zip(probs.iter()) {
        prob_map.insert(key.clone(), crate::round4(v));
    }
    Answer::Choice {
        choice: keys.get(best).cloned().unwrap_or_default(),
        confidence: crate::round4(LayaBackend::confidence_from_probs(probs, k)),
        probabilities: prob_map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[tokio::test]
    async fn laya_calibration_clamping() {
        // the shipped choice:11+ bucket (0.1006) sharpens ~10x and is refused
        assert_eq!(clamp_temperature(0.1005828), 0.5);
        assert_eq!(clamp_temperature(7.0), 5.0);
        assert_eq!(clamp_temperature(f32::NAN), 1.0);
        assert_eq!(clamp_temperature(f32::INFINITY), 1.0);
        assert_eq!(clamp_temperature(1.9833995), 1.9833995);
    }

    #[tokio::test]
    async fn laya_py_json_matches_python_separators() {
        assert_eq!(py_json(&json!("plain text")), "\"plain text\"");
        assert_eq!(py_json(&json!(2)), "2");
        // object key order follows serde_json's sorted map, so compare
        // semantics: the rendered rubric must round-trip to the same JSON
        let v = json!({"what": "Critical", "examples": ["OOM killer", 2]});
        let rendered = py_json(&v);
        assert!(rendered.contains("\"what\": \"Critical\""));
        assert!(rendered.contains("\"examples\": [\"OOM killer\", 2]"));
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed, v);
        assert_eq!(
            render_criterion(&json!("invoices, payments")),
            "invoices, payments"
        );
        assert_eq!(
            render_criterion(&json!({"a": {"b": "c"}})),
            "{\"a\": {\"b\": \"c\"}}"
        );
    }

    #[tokio::test]
    #[allow(clippy::excessive_precision)]
    async fn laya_temperature_buckets_by_option_count() {
        assert_eq!(LayaBackend::size_bucket(1), "2");
        assert_eq!(LayaBackend::size_bucket(2), "2");
        assert_eq!(LayaBackend::size_bucket(3), "3-5");
        assert_eq!(LayaBackend::size_bucket(5), "3-5");
        assert_eq!(LayaBackend::size_bucket(6), "6-10");
        assert_eq!(LayaBackend::size_bucket(10), "6-10");
        assert_eq!(LayaBackend::size_bucket(11), "11+");
        assert_eq!(laya_qtype("choice"), 0);
        assert_eq!(laya_qtype("score"), 1);
        assert_eq!(laya_qtype("noul"), 2);
        // bucket priorities: overrides first, then the plain type temps
        let (temperature, by) = default_laya_temps();
        assert_eq!(
            temperature,
            [1.6369030475616455, 1.2514300346374512, 1.983399510383606]
        );
        assert_eq!(
            by.get("noul:2").copied().unwrap(),
            clamp_temperature(1.983399510383606)
        );
        assert!(
            !by.contains_key("score:2"),
            "score:2 falls back to temperature"
        );
    }

    #[tokio::test]
    async fn laya_choice_pair_indices_cover_all_pairs() {
        assert_eq!(choice_pair_indices(1), Vec::<(usize, usize)>::new());
        assert_eq!(choice_pair_indices(2), vec![(0, 1)]);
        assert_eq!(choice_pair_indices(3), vec![(0, 1), (0, 2), (1, 2)]);
        assert_eq!(
            choice_pair_indices(4),
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        );
        assert_eq!(choice_pair_indices(6).len(), 15);
    }

    #[tokio::test]
    async fn laya_pair_wins_aggregate_to_a_distribution() {
        // 3 options, ballots: (0,1) 0.7/0.3, (0,2) 0.6/0.4, (1,2) 0.2/0.8
        let wins = vec![0.7 + 0.6, 0.3 + 0.2, 0.4 + 0.8];
        let probs = pair_wins_to_probs(&wins, 3);
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 0.4333333).abs() < 1e-6);
        assert!((probs[1] - 0.1666666).abs() < 1e-6);
        assert!((probs[2] - 0.4).abs() < 1e-6);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // a clean sweep: option 0 beats everyone
        let wins = vec![2.0, 0.5, 0.5];
        let probs = pair_wins_to_probs(&wins, 3);
        assert_eq!(probs[0], 2.0 / 3.0);
    }
}
