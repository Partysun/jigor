//! Sales Pitch Tester — example that builds a live "hype meter" for a
//! dictated sales pitch on top of the library's decision backends (Jev-style
//! `noul`/`choice`/`score` questions).
//!
//! Everything sales-specific lives here as an example of lib usage: a
//! 6-question bank — one pass per sentence (the reference hype meter asks
//! "every sentence · 6 questions each") — run through either backend —
//! `von` (local ONNX) or `jev` (OpenRouter). The first question gates the
//! sentence: only claims about a product or service are scored; an arbitrary
//! remark ("Давай что-нибудь запишем") drops out and cannot rack up a hype
//! index. The rest aggregate into a transparent 0-100 **hype index** with
//! the four hype criteria **max-pooled** across sentences (one hyped line
//! reads hyped — a mean would smear it), the five criteria's "fired %"
//! (rose/radar values), strong points that advise what to keep, and weak
//! points that suggest the fix for each hyped signal — with every point
//! value a health score (high = good, low ~0 = problem).
//! Compare the same pitch across backends by model.
//!
//! The bank judges every sentence on five counts:
//!
//!   0. `is_pitch` — is this a claim about a product/service at all? (gate)
//!   1. `concrete_metric` — an actual figure (good, but NOT part of the
//!      hype index — the reference marks it `index: false`);
//!   2. `buzzwords` — stacked jargon with no meaning (hype);
//!   3. `overpromise` — too-good-to-be-true claims (hype);
//!   4. `urgency_pressure` — artificial urgency/scarcity (hype);
//!   5. `vague_benefit` — a benefit too vague to check (hype).
//!
//! ```bash
//! cargo run -p jigor --example sales -- "We just crossed 12,000 paying teams. Our platform is an AI-native, next-gen solution."
//! cargo run -p jigor --example sales -- --model jev "Founder pricing disappears at midnight, so sign today."
//! cargo run -p jigor --example sales -- --model jev --json "It saves six hours of data entry a week."
//! ```
//!
//! The aggregation is pure and unit-tested below (no model needed); only
//! `ask` touches a backend.

use jigor::{Answer, Asks, Error, Question, Result};
use serde_json::{Map, Value, json};
use std::collections::HashMap;

fn main() -> Result<()> {
    let mut parts: Vec<String> = Vec::new();
    let mut as_json = false;
    let mut model_hint: Option<String> = None;
    let mut args = std::env::args()
        .skip(1)
        .collect::<Vec<String>>()
        .into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => as_json = true,
            "--model" => {
                model_hint = Some(
                    args.next()
                        .ok_or_else(|| Error::internal("--model requires a value".to_string()))?
                        .to_string(),
                );
            }
            "--help" | "-h" => {
                println!("usage: sales [--json] [--model <id-or-alias>] \"<pitch text>\"");
                std::process::exit(0);
            }
            _other => parts.push(arg.to_string()),
        }
    }
    let text = parts.join(" ");
    if text.is_empty() {
        println!("usage: sales [--json] [--model <id-or-alias>] \"<pitch text>\"");
        std::process::exit(1);
    }

    let model = match model_hint {
        Some(m) => m,
        None => "von-1.0.0".to_string(),
    };

    let _ = ort::init().commit();
    let report = score(&text, &model)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string(&sales_report_to_json(&report, &model))?
        );
    } else {
        print_report(&report, &text, &model);
    }
    Ok(())
}

fn print_report(report: &SalesReport, text: &str, model: &str) {
    println!("Pitch: {}", text);
    println!("model: {}", model);
    println!(
        "Hype index: {}/100  ({} sentences; 0 = plain and credible, 100 = hype)",
        report.hype_index, report.sentences
    );
    for (family, stats) in &report.families {
        println!(
            "  {}  {}  fired {:>3}% across {} sentences",
            family,
            stats.label,
            (stats.fired * 100.0).round() as u32,
            stats.total,
        );
    }
    println!("strong points:");
    for p in &report.helped {
        println!(
            "  - {} ({}, {}%)",
            p.label,
            p.answer,
            (p.value * 100.0).round() as u32
        );
    }
    println!("weak points:");
    for p in &report.hurt {
        println!(
            "  - {} ({}, {}%)",
            p.label,
            p.answer,
            (p.value * 100.0).round() as u32
        );
    }
}

/// One question in the sales bank — all `noul` (yes/no with a probability),
/// carrying the true/false criteria sides like the lib's noul gate.
#[derive(Debug, Clone)]
pub struct SalesQuestion {
    pub id: String,
    /// Family key: CONCRETE | BUZZWORDS | OVERPROMISE | URGENCY | VAGUE.
    pub family: String,
    /// Short tooltip label for the UI, also the model `instructions`.
    pub label: String,
    /// The criterion for "true" (the signal firing).
    pub criteria_true: String,
    /// The criterion for "false" (the signal absent).
    pub criteria_false: String,
}

/// A normalized answer for one question of one sentence.
#[derive(Debug, Clone)]
pub struct SalesAnswer {
    pub id: String,
    pub family: String,
    pub label: String,
    /// "Yes"/"No".
    pub answer: String,
    /// Normalized strength in [0, 1] — `probability` for noul.
    pub value: f32,
    pub p: f32,
}

/// Per-family aggregates (the "X% fired" radar values).
#[derive(Debug, Clone)]
pub struct SalesFamilyStats {
    pub label: String,
    /// The strongest signal of this family across the pitch's sentences —
    /// max-pooled, in [0, 1]. One hyped line reads hyped.
    pub fired: f32,
    /// Number of sentences the family was judged on.
    pub total: usize,
}

/// One strong/weak point with the value that earned it.
#[derive(Debug, Clone)]
pub struct SalesPoint {
    /// Why it landed here: "CONCRETE" | "CLEAN" | "HYPE".
    pub family: String,
    /// The actionable advice: what to keep for strong points, what to fix
    /// for weak points — never a repeat of the pitch text, which is already
    /// on the user's screen.
    pub label: String,
    /// The criterion behind the point ("Concrete metric", "Overpromise"…).
    pub answer: String,
    /// Health of this aspect in [0, 1]: high = good for the pitch, low
    /// (~0) = the point is a problem to fix. Weak points carry the inverted
    /// fired strength (1 − signal), so a hyped sentence reads ~0%.
    pub value: f32,
}

/// Aggregated score for one pitch.
#[derive(Debug, Clone)]
pub struct SalesReport {
    /// 0-100 hype index of the whole pitch: mean of the four hype
    /// criteria across every sentence. 0 = plain and credible, 100 =
    /// wall-to-wall hype.
    pub hype_index: u32,
    /// Number of sentences the pitch was split into.
    pub sentences: usize,
    /// The five criteria in canonical order, CONCRETE first.
    pub families: Vec<(String, SalesFamilyStats)>,
    /// Strong points: what to keep (a sentence carried the pitch on a
    /// concrete metric, or read clean).
    pub helped: Vec<SalesPoint>,
    /// Weak points: the fix each hyped sentence needs (or the missing
    /// concrete number), with the signal that fired hardest named.
    pub hurt: Vec<SalesPoint>,
}

fn family_keys() -> Vec<String> {
    vec![
        "CONCRETE".to_string(),
        "BUZZWORDS".to_string(),
        "OVERPROMISE".to_string(),
        "URGENCY".to_string(),
        "VAGUE".to_string(),
    ]
}

fn family_label(family: &str) -> String {
    match family {
        "CONCRETE" => "Concrete metric".to_string(),
        "BUZZWORDS" => "Buzzwords".to_string(),
        "OVERPROMISE" => "Overpromise".to_string(),
        "URGENCY" => "Urgency pressure".to_string(),
        "VAGUE" => "Vague benefit".to_string(),
        other => other.to_string(),
    }
}

/// The four criteria that feed the hype index (the reference marks them
/// `index: true`; concrete_metric is `index: false`).
fn is_hype_family(family: &str) -> bool {
    matches!(family, "BUZZWORDS" | "OVERPROMISE" | "URGENCY" | "VAGUE")
}

fn question(
    id: &str,
    family: &str,
    label: &str,
    criteria_true: &str,
    criteria_false: &str,
) -> SalesQuestion {
    SalesQuestion {
        id: id.to_string(),
        family: family.to_string(),
        label: label.to_string(),
        criteria_true: criteria_true.to_string(),
        criteria_false: criteria_false.to_string(),
    }
}

/// The 6-question bank. All noul with true/false criteria sides — the same
/// wire shape as the lib's noul gate; von's two-way judge is markedly
/// sharper with explicit sides than with a bare instructions string. The
/// first question, `is_pitch`, gates the sentence: only claims about a
/// product or service get scored at all — a random remark ("Давай
/// что-нибудь запишем") is not a pitch and must not rack up a hype index.
pub fn sales_bank() -> Vec<SalesQuestion> {
    vec![
        question(
            "is_pitch",
            "PITCH",
            "Is this a claim about a product or service?",
            "The sentence is a claim about a product or service: what it does, a benefit, a price, availability, or urgency to buy",
            "The sentence is unrelated: a general remark, a question, an instruction, or narrative",
        ),
        question(
            "concrete_metric",
            "CONCRETE",
            "Gives a concrete, measurable number",
            "An actual figure: users, revenue, retention, time saved, price, speed with a number",
            "Claims of traction or quality without a figure",
        ),
        question(
            "buzzwords",
            "BUZZWORDS",
            "Mostly buzzwords with little meaning",
            "Stacked jargon like AI-native, next-gen, disruptive, synergy, paradigm with no concrete function",
            "Says concretely what the product does, even if it mentions AI or cloud",
        ),
        question(
            "overpromise",
            "OVERPROMISE",
            "Promises too-good-to-be-true results",
            "Guaranteed dramatic outcomes, absolute promises, total elimination of problems",
            "Typical ranges, honest limits, certifications or modest claims",
        ),
        question(
            "urgency_pressure",
            "URGENCY",
            "Pushes with artificial urgency",
            "Deadlines, limited spots, fear of missing out, 'sign today'",
            "Neutral timelines, invitations to take time, or plain availability",
        ),
        question(
            "vague_benefit",
            "VAGUE",
            "Describes a benefit too vague to check",
            "Benefits like better lives, unlocked potential, new levels of clarity",
            "A specific, checkable benefit: time saved, a named feature, a removed manual task",
        ),
    ]
}

/// Split a pitch into sentences on `.!?!。！？` (with newlines), keeping
/// decimal points ("2.5x faster" stays one sentence) and attaching a
/// closing quote/bracket to the sentence it belongs to.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let prev_digit = current
            .chars()
            .next_back()
            .is_some_and(|p| p.is_ascii_digit());
        let next_digit = chars.peek().is_some_and(|n| n.is_ascii_digit());
        let decimal_dot = ch == '.' && prev_digit && next_digit;
        current.push(ch);
        let terminator = matches!(ch, '.' | '!' | '?' | '。' | '！' | '？' | '\n');
        if terminator && !decimal_dot {
            // A closing quote or bracket belongs to the finished sentence.
            if let Some(&next) = chars.peek()
                && matches!(next, '"' | '\'' | '»' | ')')
            {
                current.push(next);
                chars.next();
            }
            let s = current.trim();
            if !s.is_empty() {
                out.push(s.to_string());
            }
            current.clear();
        }
    }
    let s = current.trim();
    if !s.is_empty() {
        out.push(s.to_string());
    }
    out
}

/// The 5 questions in their wire form, so any backend can answer them.
/// `noul`, with the true/false criteria sides for the four hype signals —
/// exactly like the lib's gate. `concrete_metric` is asked as a bare
/// yes/no question: its `criteria_true` holds the question text, no sides.
/// The 6 questions in their wire form, so any backend can answer them:
/// `noul` with the true/false criteria sides, exactly like the lib's gate.
fn bank_as_asks() -> Vec<Question> {
    sales_bank()
        .iter()
        .map(|q| Question {
            id: q.id.clone(),
            kind: "noul".to_string(),
            instructions: q.label.clone(),
            criteria: Some(json!({
                "true": q.criteria_true,
                "false": q.criteria_false,
            })),
            temperature: None,
        })
        .collect()
}

/// Probability at which a sentence counts as a pitch claim (see
/// `is_pitch`). Calibrated on jev: arbitrary remarks score 2-4%, real
/// claims 70-95% — a 0.5 split is wide open. (The local von backend is
/// mushier here — 0.45-0.6 near the border — which caps how cleanly it can
/// separate garbage from pitch.)
const PITCH_GATE: f32 = 0.5;

/// Run the whole bank over every sentence through `jigor::ask` and
/// aggregate the report. One `ask()` per sentence. Sentences that do not
/// clear `is_pitch` (a chance remark, a question, narration — not product
/// claims) drop out and never contribute to the index or the points.
pub fn score(text: &str, model: &str) -> Result<SalesReport> {
    let sentences = split_sentences(text);
    let mut judged: Vec<String> = Vec::new();
    let mut per_sentence: Vec<Vec<SalesAnswer>> = Vec::new();
    for s in &sentences {
        let asks = ask_bank(s, model)?;
        let is_claim = match asks.answers.get("is_pitch") {
            Some(Answer::Noul { probability }) => *probability >= PITCH_GATE,
            _ => false,
        };
        if is_claim {
            judged.push(s.clone());
            per_sentence.push(sales_answers(&asks.answers));
        }
    }
    Ok(report_from_sentences(&judged, &per_sentence))
}

/// Run the whole bank through one backend: local runs the ONNX model in
/// process, openrouter posts to the Decisions API (Jev). Returns the same
/// normalized rows either way, so the index is directly comparable.
fn ask_bank(sentence: &str, model: &str) -> Result<Asks> {
    let context = Value::String(sentence.to_string());
    let questions = bank_as_asks();
    jigor::ask(model, &context, &questions, None)
}

/// Map the lib's typed answers back onto the bank rows (label rendering).
pub fn sales_answers(asks: &HashMap<String, Answer>) -> Vec<SalesAnswer> {
    let mut out: Vec<SalesAnswer> = Vec::with_capacity(sales_bank().len());
    for q in sales_bank() {
        if let Some(ans) = asks.get(&q.id)
            && let Answer::Noul { probability } = ans
        {
            out.push(SalesAnswer {
                id: q.id.clone(),
                family: q.family.clone(),
                label: q.label.clone(),
                answer: if *probability >= 0.5 {
                    "Yes".to_string()
                } else {
                    "No".to_string()
                },
                value: *probability,
                p: *probability,
            });
        }
    }
    out
}

/// Per-sentence metrics used by the strong/weak selection.
#[derive(Clone)]
struct Row {
    concrete: f32,
    /// Mean of the four hype criteria in this sentence.
    hype: f32,
    /// The hype signal that fired hardest: (value, family key).
    worst: Option<(f32, String)>,
}

/// The fix a weak point suggests, derived from the signal that fired. The
/// pitch text itself is already on the user's screen, so points carry the
/// actionable advice, not a repeat of the sentence.
fn suggestion_for(family: &str) -> String {
    match family {
        "BUZZWORDS" => "Say concretely what the product does — drop the jargon".to_string(),
        "OVERPROMISE" => "Swap the guarantee for a realistic range or limit".to_string(),
        "URGENCY" => "Drop the artificial deadline; offer a calm next step".to_string(),
        "VAGUE" => "Name a specific, checkable outcome".to_string(),
        _ => "Add a concrete number — users, time saved, price".to_string(),
    }
}

/// The advice a strong point carries: what to keep, not the sentence text.
fn keep_suggestion_for(family: &str) -> String {
    match family {
        "CONCRETE" => "Keep the concrete metric — a real number carries the claim".to_string(),
        _ => "Keep it plain and credible — no hype needed".to_string(),
    }
}

fn row_for(answers: &[SalesAnswer]) -> Row {
    let mut concrete = 0.0_f32;
    let mut hype_sum = 0.0_f32;
    let mut hype_n = 0usize;
    let mut worst: Option<(f32, String)> = None;
    for a in answers {
        if a.family == "CONCRETE" {
            concrete = a.value;
        } else if is_hype_family(&a.family) {
            hype_sum += a.value;
            hype_n += 1;
            let is_worse = match &worst {
                Some((m, _)) => a.value > *m,
                None => true,
            };
            if is_worse {
                worst = Some((a.value, a.family.clone()));
            }
        }
    }
    let hype = if hype_n == 0 {
        0.0
    } else {
        hype_sum / hype_n as f32
    };
    Row {
        concrete,
        hype,
        worst,
    }
}

/// Pure aggregation over normalized answers (no model I/O, unit-testable).
pub fn report_from_sentences(
    sentences: &[String],
    per_sentence: &[Vec<SalesAnswer>],
) -> SalesReport {
    let judged_total = sentences.len();

    let mut families: Vec<(String, SalesFamilyStats)> = Vec::with_capacity(5);
    let mut hype_acc = 0.0_f32;
    let mut hype_n = 0usize;
    for family in family_keys() {
        // Max-pool across sentences: the strongest signal in the pitch sets
        // the family level. One hyped line means the pitch reads hyped — a
        // mean would smear it into middling, exactly the failure the
        // gate + max design fixes (the junk sentence used to outscore the
        // hype pitch on flat von answers).
        let mut strongest = 0.0_f32;
        let mut judged = 0usize;
        for answers in per_sentence {
            for a in answers {
                if a.family == family {
                    strongest = strongest.max(a.value);
                    judged += 1;
                }
            }
        }
        let fired = if judged == 0 {
            0.0
        } else {
            ((strongest) * 10000.0).round() / 10000.0
        };
        families.push((
            family.clone(),
            SalesFamilyStats {
                label: family_label(&family),
                fired,
                total: judged,
            },
        ));
        if is_hype_family(&family) {
            hype_acc += fired;
            hype_n += 1;
        }
    }
    let hype_index = if hype_n == 0 {
        0
    } else {
        ((hype_acc / hype_n as f32) * 100.0).round() as u32
    };

    let rows: Vec<(usize, Row)> = per_sentence
        .iter()
        .take(judged_total)
        .enumerate()
        .map(|(i, answers)| (i, row_for(answers)))
        .collect();

    // Strong points: sentences carrying a concrete metric first, then the
    // cleanest (lowest-hype) sentences to fill the list. Each carries the
    // advice to keep that line, not the sentence text.
    let mut helped: Vec<SalesPoint> = Vec::with_capacity(3);
    let mut helped_used: Vec<usize> = Vec::with_capacity(3);
    let mut by_concrete = rows.clone();
    by_concrete.sort_by(|a, b| b.1.concrete.partial_cmp(&a.1.concrete).unwrap());
    for (i, row) in &by_concrete {
        if helped.len() == 3 {
            break;
        }
        if row.concrete >= 0.6 {
            helped_used.push(*i);
            helped.push(SalesPoint {
                family: "CONCRETE".to_string(),
                label: keep_suggestion_for("CONCRETE"),
                answer: "Concrete metric".to_string(),
                value: row.concrete,
            });
        }
    }
    if helped.len() < 3 {
        let mut by_clean = rows.clone();
        by_clean.sort_by(|a, b| a.1.hype.partial_cmp(&b.1.hype).unwrap());
        for (i, row) in &by_clean {
            if helped.len() == 3 {
                break;
            }
            if row.hype <= 0.4 && !helped_used.contains(i) {
                helped_used.push(*i);
                helped.push(SalesPoint {
                    family: "CLEAN".to_string(),
                    label: keep_suggestion_for("CLEAN"),
                    answer: "Plain and credible".to_string(),
                    value: ((1.0 - row.hype) * 10000.0).round() / 10000.0,
                });
            }
        }
    }

    // Weak points: the most hyped sentences — a sentence counts when at
    // least one hype signal fired hard (worst >= 0.5), ranked by its mean
    // hype; the fired signal becomes the suggested fix. Then the sentences
    // without a concrete number fill the list. Values are *health*: high =
    // good for the pitch, low (~0) = the fix is needed — so a heavily
    // hyped sentence reads ~0%, never a big positive number.
    let mut hurt: Vec<SalesPoint> = Vec::with_capacity(3);
    let mut hurt_used: Vec<usize> = Vec::with_capacity(3);
    let mut by_hype = rows.clone();
    by_hype.sort_by(|a, b| b.1.hype.partial_cmp(&a.1.hype).unwrap());
    for (i, row) in &by_hype {
        if hurt.len() == 3 {
            break;
        }
        let worst_fired = row.worst.as_ref().map(|(v, _)| *v).unwrap_or(0.0);
        if worst_fired >= 0.5
            && let Some((value, family)) = &row.worst
        {
            hurt_used.push(*i);
            hurt.push(SalesPoint {
                family: "HYPE".to_string(),
                label: suggestion_for(family),
                answer: family_label(family),
                value: ((1.0 - *value) * 10000.0).round() / 10000.0,
            });
        }
    }
    if hurt.len() < 3 {
        let mut no_metric = rows.clone();
        no_metric.sort_by(|a, b| a.1.concrete.partial_cmp(&b.1.concrete).unwrap());
        for (i, row) in &no_metric {
            if hurt.len() == 3 {
                break;
            }
            if row.concrete <= 0.4 && !hurt_used.contains(i) {
                hurt_used.push(*i);
                hurt.push(SalesPoint {
                    family: "CONCRETE".to_string(),
                    label: suggestion_for("CONCRETE"),
                    answer: "No concrete metric".to_string(),
                    value: row.concrete,
                });
            }
        }
    }

    SalesReport {
        hype_index,
        sentences: judged_total,
        families,
        helped,
        hurt,
    }
}

fn point_to_json(p: &SalesPoint) -> Value {
    json!({
        "family": p.family,
        "label": p.label,
        "answer": p.answer,
        "value": p.value,
    })
}

/// 0-100 hype index, the five criteria's "fired %" (rose widget), and the
/// strong/weak point lists — strong points name the sentence, weak points
/// suggest the fix.
pub fn sales_report_to_json(report: &SalesReport, model: &str) -> Value {
    let mut families = Map::new();
    for (family, stats) in &report.families {
        families.insert(
            family.clone(),
            json!({"label": stats.label, "fired": stats.fired, "total": stats.total}),
        );
    }
    let engine = json!({
        "model": model,
        "question_set": "v1.1",
        "note": "MVP: transparent heuristic over the 5-question hype bank; the index is the mean of the four hype criteria across every sentence.",
    });
    json!({
        "score": report.hype_index,
        "sentences": report.sentences,
        "families": families,
        "helped": report.helped.iter().map(point_to_json).collect::<Vec<Value>>(),
        "hurt": report.hurt.iter().map(point_to_json).collect::<Vec<Value>>(),
        "engine": engine,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(family: &str, value: f32) -> SalesAnswer {
        let bank = sales_bank();
        let q = bank
            .iter()
            .find(|q| q.family == family)
            .expect("question for family");
        SalesAnswer {
            id: q.id.clone(),
            family: q.family.clone(),
            label: q.label.clone(),
            answer: if value >= 0.5 { "Yes" } else { "No" }.to_string(),
            value,
            p: value,
        }
    }

    /// Build a sentence's answers with per-criterion values.
    fn sentence(
        concrete: f32,
        buzzwords: f32,
        overpromise: f32,
        urgency: f32,
        vague: f32,
    ) -> Vec<SalesAnswer> {
        vec![
            answer("CONCRETE", concrete),
            answer("BUZZWORDS", buzzwords),
            answer("OVERPROMISE", overpromise),
            answer("URGENCY", urgency),
            answer("VAGUE", vague),
        ]
    }

    #[tokio::test]
    async fn bank_has_6_questions_cover_every_family() {
        let bank = sales_bank();
        assert_eq!(bank.len(), 6);
        for family in family_keys() {
            assert!(
                bank.iter().any(|q| q.family == family),
                "family {family} covered"
            );
        }
    }

    #[tokio::test]
    async fn splits_on_punctuation_and_newlines() {
        assert_eq!(
            split_sentences("We save you six hours a week. That's huge!\nAnything else?"),
            vec![
                "We save you six hours a week.".to_string(),
                "That's huge!".to_string(),
                "Anything else?".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn keeps_decimals_and_closing_quotes() {
        let parts = split_sentences("\"Our app cuts costs 2.5x.\" Trust us.");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "\"Our app cuts costs 2.5x.\"");
        assert_eq!(parts[1], "Trust us.");
    }

    #[tokio::test]
    async fn empty_text_has_no_sentences() {
        assert!(split_sentences("   \n ").is_empty());
    }

    #[tokio::test]
    async fn hype_index_is_the_max_of_the_four_hype_criteria() {
        // Signals are max-pooled per family across sentences: one fully
        // hyped sentence alone drives the pitch to 100; a half-hyped, half
        // clean pitch reads 100 too (one hyped line is enough to flag);
        // a single signal alone reads 25.
        let s = vec!["one".to_string(), "two".to_string()];
        let per = vec![
            sentence(1.0, 1.0, 1.0, 1.0, 1.0),
            sentence(0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        assert_eq!(report_from_sentences(&s, &per).hype_index, 100);
        let per = vec![sentence(0.0, 1.0, 0.0, 0.0, 0.0)];
        assert_eq!(report_from_sentences(&s[..1], &per).hype_index, 25);
        let per = vec![sentence(0.8, 0.0, 0.0, 0.0, 0.0)];
        assert_eq!(report_from_sentences(&s[..1], &per).hype_index, 0);
    }

    #[tokio::test]
    async fn concrete_metric_never_moves_the_index() {
        let s = vec!["one".to_string()];
        let per = vec![sentence(1.0, 0.4, 0.4, 0.4, 0.4)];
        let report = report_from_sentences(&s, &per);
        assert_eq!(report.hype_index, 40);
        assert!((family_stats(&report, "CONCRETE").fired - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn strong_points_suggest_what_to_keep() {
        let s = vec![
            "We cut onboarding from nine days to four.".to_string(),
            "This is the total revolution of everything.".to_string(),
            "Take two weeks with the trial.".to_string(),
        ];
        let per = vec![
            sentence(0.95, 0.0, 0.2, 0.1, 0.1),
            sentence(0.0, 0.9, 0.9, 0.3, 0.8),
            sentence(0.2, 0.0, 0.0, 0.0, 0.0),
        ];
        let report = report_from_sentences(&s, &per);
        assert_eq!(report.helped.len(), 2);
        assert_eq!(report.helped[0].family, "CONCRETE");
        assert_eq!(
            report.helped[0].label,
            "Keep the concrete metric — a real number carries the claim"
        );
        assert_eq!(report.helped[1].family, "CLEAN");
        assert_eq!(
            report.helped[1].label,
            "Keep it plain and credible — no hype needed"
        );
    }

    #[tokio::test]
    async fn weak_points_suggest_fixes_and_fill_without_metrics() {
        let s = vec![
            "We guarantee your revenue triples in ninety days.".to_string(),
            "Onboarding takes ten business days.".to_string(),
            "It makes things smoother.".to_string(),
        ];
        let per = vec![
            sentence(0.1, 0.2, 0.95, 0.3, 0.5),
            sentence(1.0, 0.0, 0.0, 0.0, 0.0),
            sentence(0.1, 0.2, 0.1, 0.3, 0.0),
        ];
        let report = report_from_sentences(&s, &per);
        assert_eq!(report.hurt.len(), 2);
        assert_eq!(report.hurt[0].family, "HYPE");
        assert_eq!(report.hurt[0].answer, "Overpromise");
        assert_eq!(
            report.hurt[0].label,
            "Swap the guarantee for a realistic range or limit"
        );
        // Health scale: a 0.95 overpromise shows ~0, not a big number.
        assert_eq!(report.hurt[0].value, 0.05);
        assert_eq!(report.hurt[1].family, "CONCRETE");
        assert_eq!(
            report.hurt[1].label,
            "Add a concrete number — users, time saved, price"
        );
        assert_eq!(report.hurt[1].value, 0.1);
    }

    #[tokio::test]
    async fn one_hard_fired_signal_flags_the_sentence() {
        // Mean hype sits below the threshold, but the overpromise fired
        // hard: the sentence must still land in the weak points.
        let s = vec!["We guarantee a triple in ninety days.".to_string()];
        let per = vec![sentence(0.1, 0.2, 0.95, 0.3, 0.5)];
        let report = report_from_sentences(&s, &per);
        assert_eq!(report.hurt.len(), 1);
        assert_eq!(report.hurt[0].family, "HYPE");
        assert_eq!(report.hurt[0].value, 0.05);
    }

    #[tokio::test]
    async fn a_clean_pitch_gets_no_weak_points() {
        let s = vec![
            "It flags duplicate vendor charges in your bank statements.".to_string(),
            "Reps stop copying call notes by hand.".to_string(),
        ];
        let per = vec![
            sentence(0.45, 0.0, 0.0, 0.0, 0.0),
            sentence(0.5, 0.0, 0.0, 0.0, 0.0),
        ];
        let report = report_from_sentences(&s, &per);
        assert!(report.hurt.is_empty());
        assert!(report.helped.len() >= 1);
    }

    #[tokio::test]
    async fn report_json_shape() {
        let s = vec!["One pitch sentence.".to_string()];
        let per = vec![sentence(0.6, 0.1, 0.2, 0.0, 0.1)];
        let v = sales_report_to_json(&report_from_sentences(&s, &per), "von-1.0.0");
        assert!(v.get("sentences").unwrap().as_u64().unwrap() >= 1);
        let fams = v.get("families").and_then(Value::as_object).unwrap();
        assert!(fams.contains_key("CONCRETE"));
        assert!(fams.contains_key("VAGUE"));
        assert_eq!(
            v.get("engine")
                .unwrap()
                .get("question_set")
                .unwrap()
                .clone(),
            json!("v1.1")
        );
    }

    #[tokio::test]
    async fn bank_roundtrips_through_lib_wire_shapes() {
        let asks = bank_as_asks();
        assert_eq!(asks.len(), 6);
        for q in asks {
            assert_eq!(q.kind, "noul");
            assert!(!q.instructions.is_empty());
            match q.criteria {
                Some(Value::Object(m)) => {
                    assert!(m.contains_key("true"));
                    assert!(m.contains_key("false"));
                }
                // concrete_metric is asked as a plain yes/no question (its
                // example numbers entangle with the sentence as sides).
                None if q.id == "concrete_metric" => {}
                _ => panic!("noul criteria must carry true/false sides"),
            }
        }
    }

    #[tokio::test]
    async fn typed_answers_render_bank_rows() {
        let mut asks = HashMap::new();
        asks.insert("buzzwords".to_string(), Answer::Noul { probability: 0.87 });
        asks.insert(
            "concrete_metric".to_string(),
            Answer::Noul { probability: 0.93 },
        );
        let rows = sales_answers(&asks);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "concrete_metric");
        assert_eq!(rows[0].answer, "Yes");
        assert_eq!(rows[1].id, "buzzwords");
        assert_eq!(rows[1].family, "BUZZWORDS");
    }

    fn family_stats(report: &SalesReport, family: &str) -> SalesFamilyStats {
        for (f, stats) in &report.families {
            if f == family {
                return stats.clone();
            }
        }
        panic!("family {family} missing")
    }
}
