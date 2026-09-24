//! Tweet Tester — example that builds a working viral-score MVP on top of the
//! library's decision backends (Jev-style `noul`/`choice`/`score` questions).
//!
//! Reference used as an example for this library: https://superx.so/tweet-tester (ideas only, reimplemented manually here — no code copied).
//!
//! Everything tweet-specific lives here as an example of lib usage: the
//! 61-question bank (question set "v1.1", 8 families), running the bank
//! through either backend — `von` (local ONNX) or `jev` (OpenRouter) — a
//! transparent 0-100 aggregation (50 = your account's normal post; above
//! beats it, below does worse), the family "fired %" radar values, and
//! engagement counters (placeholders for a fitted model). Compare the same
//! tweet across backends by model.
//!
//! ```bash
//! cargo run -p jigor --example tweet -- "We just crossed 10,000 paying customers. Thank you."
//! cargo run -p jigor --example tweet -- --model jev "We just crossed 10,000 paying customers."
//! cargo run -p jigor --example tweet -- --model jev --json "Hot take."
//! ```

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
                println!("usage: tweet [--json] [--model <id-or-alias>] \"<post text>\"");
                std::process::exit(0);
            }
            _other => parts.push(arg.to_string()),
        }
    }
    let text = parts.join(" ");
    if text.is_empty() {
        println!("usage: tweet [--json] [--model <id-or-alias>] \"<post text>\"");
        std::process::exit(1);
    }

    let model = match model_hint {
        Some(m) => m,
        None => "von-1.0.0".to_string(),
    };

    let _ = ort::init().commit();
    let asks = ask_bank(&text, &model)?;
    let answers = tweet_answers(&asks.answers, &tweet_bank());
    let report = report_from_answers(&answers);

    if as_json {
        let v = tweet_report_to_json(&report, &asks.model);
        println!("{}", serde_json::to_string(&v)?);
    } else {
        print_tweet_report(&report, &text, &asks.backend, &asks.model);
    }
    Ok(())
}

fn print_tweet_report(report: &TweetReport, text: &str, backend: &str, model: &str) {
    println!("Tweet: {}", text);
    println!("backend: {}  model: {}", backend, model);
    println!("Viral score: {}/100  (50 = your normal post)", report.score);
    println!("beats own normal: {:.4}", report.beats_own_normal);
    for (family, stats) in &report.families {
        println!(
            "  {}  {}  fired {:>3}% of {} questions",
            family,
            stats.label,
            (stats.fired * 100.0).round() as u32,
            stats.total,
        );
    }
    println!("helped:");
    for a in &report.helped {
        println!(
            "  - {} ({}, {}%)",
            a.label,
            a.answer,
            (a.value * 100.0).round() as u32
        );
    }
    println!("hurt:");
    for a in &report.hurt {
        println!(
            "  - {} ({}, {}%)",
            a.label,
            a.answer,
            (a.value * 100.0).round() as u32
        );
    }
}

/// Jev-style question kinds: `noul`/`choice`/`score` — one wire protocol
/// across backends.
#[derive(Debug, Clone)]
pub enum TweetCriteria {
    /// Plain yes/no question (`type: "noul"`).
    Noul,
    /// Options with descriptions (`type: "choice"`).
    Choice(HashMap<String, Option<String>>),
    /// Ordered 0..N-1 rubric (`type: "score"`).
    Score(Vec<String>),
}

/// One question in the bank, mirroring the `/v1/systemone` `questions` shape.
#[derive(Debug, Clone)]
pub struct TweetQuestion {
    pub id: String,
    pub family: String,
    /// "noul" | "choice" | "score"
    pub kind: String,
    /// The tooltip label, used as the model `instructions`.
    pub label: String,
    pub criteria: TweetCriteria,
}

/// A normalized answer for one question.
#[derive(Debug, Clone)]
pub struct TweetAnswer {
    pub id: String,
    pub family: String,
    pub kind: String,
    pub label: String,
    /// Human answer: "Yes"/"No", the chosen option description, or "1 of 3".
    pub answer: String,
    /// Normalized strength in [0, 1]: noul p, choice p, score level/max.
    pub value: f32,
    pub p: Option<f32>,
}

/// Per-family aggregates (the "X% of its questions fired" tooltip data).
#[derive(Debug, Clone)]
pub struct FamilyStats {
    pub label: String,
    /// Weighted share of the family's questions that fired, in [0, 1]
    /// (mean positive-direction value, the radar "% fired" number).
    pub fired: f32,
    pub total: usize,
}

/// Aggregated score for one tweet.
#[derive(Debug, Clone)]
pub struct TweetReport {
    /// 0-100 viral score. 50 = your account's normal post: above 50 the
    /// draft reads better than usual, below 50 worse.
    pub score: u32,
    /// 0-1 mirror of the score ("beats own normal").
    pub beats_own_normal: f32,
    /// Families in canonical order, ANTI-SIGNAL last.
    pub families: Vec<(String, FamilyStats)>,
    /// 8 families x 61 answers, so a client can render every tooltip.
    pub answers: Vec<TweetAnswer>,
    /// Top answers that helped (positive families, value >= 0.6).
    pub helped: Vec<TweetAnswer>,
    /// Top answers that hurt (fired anti-signals, then weakest positives).
    pub hurt: Vec<TweetAnswer>,
}

fn family_order() -> Vec<String> {
    vec![
        "EMOTION".to_string(),
        "CONVERSATION".to_string(),
        "SHAREABILITY".to_string(),
        "TIMELINESS".to_string(),
        "CRAFT".to_string(),
        "IDENTITY".to_string(),
        "FORMAT".to_string(),
        "ANTI-SIGNAL".to_string(),
    ]
}

fn family_label(family: &str) -> String {
    match family {
        "EMOTION" => "Emotion".to_string(),
        "CONVERSATION" => "Conversation".to_string(),
        "SHAREABILITY" => "Shareability".to_string(),
        "TIMELINESS" => "Timeliness".to_string(),
        "CRAFT" => "Craft".to_string(),
        "IDENTITY" => "Identity".to_string(),
        "FORMAT" => "Format".to_string(),
        "ANTI-SIGNAL" => "Anti-signal".to_string(),
        other => other.to_string(),
    }
}

fn question(
    id: &str,
    family: &str,
    kind: &str,
    label: &str,
    criteria: TweetCriteria,
) -> TweetQuestion {
    TweetQuestion {
        id: id.to_string(),
        family: family.to_string(),
        kind: kind.to_string(),
        label: label.to_string(),
        criteria,
    }
}

fn choice(pairs: Vec<(&str, &str)>) -> TweetCriteria {
    let mut options = HashMap::new();
    for (key, desc) in pairs {
        options.insert(key.to_string(), Some(desc.to_string()));
    }
    TweetCriteria::Choice(options)
}

fn score_rubric(items: Vec<&str>) -> TweetCriteria {
    TweetCriteria::Score(items.iter().map(|s| s.to_string()).collect())
}

/// The 61-question bank (question set "v1.1").
#[allow(clippy::vec_init_then_push)]
pub fn tweet_bank() -> Vec<TweetQuestion> {
    let mut qs: Vec<TweetQuestion> = Vec::with_capacity(61);

    qs.push(question(
        "e_dominant_emotion",
        "EMOTION",
        "choice",
        "The strongest feeling it produces",
        choice(vec![
            ("none", "No single emotion stands out"),
            (
                "joy_or_pride",
                "Excitement, pride or satisfaction about an achievement",
            ),
            ("humour", "It is a joke, irony or playful"),
            ("anger_annoyance", "Frustration, irritation or outrage"),
            ("sadness", "Loss, disappointment or melancholy"),
            (
                "surprise",
                "A twist that makes the reader go: wait, really?",
            ),
            ("fear_anxiety", "Worry, unease or apprehension"),
            ("motivation", "Energy or inspiration to go do something"),
        ]),
    ));
    qs.push(question(
        "e_milestone_joy",
        "EMOTION",
        "noul",
        "A win you just reached, told with feeling",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "e_humour",
        "EMOTION",
        "score",
        "How funny it is to your audience",
        score_rubric(vec![
            "Not trying to be funny",
            "A light touch of humour",
            "Genuinely funny to the audience",
        ]),
    ));
    qs.push(question(
        "e_self_deprecating",
        "EMOTION",
        "noul",
        "The joke is on you",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "e_indignation",
        "EMOTION",
        "score",
        "How annoyed the post sounds",
        score_rubric(vec![
            "Calm and collected, no irritation",
            "Audibly annoyed",
            "Outright angry or outraged",
        ]),
    ));
    qs.push(question(
        "e_vulnerability",
        "EMOTION",
        "noul",
        "Admits a failure or a worry",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "e_relatable_experience",
        "EMOTION",
        "score",
        "A named experience readers share",
        score_rubric(vec![
            "Nothing most readers share",
            "A common enough situation",
            "A near-universal experience everyone has had",
        ]),
    ));
    qs.push(question(
        "e_surprise_twist",
        "EMOTION",
        "noul",
        "It breaks the reader's expectation",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "e_gushing_no_tension",
        "EMOTION",
        "noul",
        "Praise for someone with nothing at stake",
        TweetCriteria::Noul,
    ));

    qs.push(question(
        "c_question_kind",
        "CONVERSATION",
        "choice",
        "What kind of question it asks",
        choice(vec![
            ("none", "No question is asked"),
            ("open_question", "An open question for answers"),
            ("rhetorical", "A rhetorical or leading question"),
            ("poll_or_choice", "Asks readers to pick an option"),
            ("help_request", "Asks the audience for help or advice"),
            ("invitation", "Invites them to connect or participate"),
        ]),
    ));
    qs.push(question(
        "c_easy_to_answer",
        "CONVERSATION",
        "score",
        "How easy it is to reply to",
        score_rubric(vec![
            "Takes real effort to answer",
            "A quick thought gets you there",
            "Anyone can answer in one breath",
        ]),
    ));
    qs.push(question(
        "c_asks_own_experience",
        "CONVERSATION",
        "noul",
        "Asks readers to report their own experience",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "c_help_request",
        "CONVERSATION",
        "noul",
        "Asks readers what to do",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "c_networking_invitation",
        "CONVERSATION",
        "noul",
        "An invitation to connect",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "c_contestable_claim",
        "CONVERSATION",
        "score",
        "A claim some readers would argue with",
        score_rubric(vec![
            "Nothing to argue with",
            "A claim some readers would question",
            "Half the timeline will disagree",
        ]),
    ));
    qs.push(question(
        "c_identity_challenge",
        "CONVERSATION",
        "noul",
        "Puts a named group on the spot",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "c_reply_tone_forecast",
        "CONVERSATION",
        "choice",
        "The replies it will get",
        choice(vec![
            ("agreement", "Mostly agreement and support"),
            ("debate", "Arguments and pushback"),
            ("jokes", "Puns, jokes and memes"),
            ("answers", "Answers and solutions"),
            ("attacks", "Retaliation and flame"),
            ("none", "Hardly any replies at all"),
        ]),
    ));
    qs.push(question(
        "c_leaves_opening",
        "CONVERSATION",
        "noul",
        "Leaves a gap a reader can fill",
        TweetCriteria::Noul,
    ));

    qs.push(question(
        "s_group_chat",
        "SHAREABILITY",
        "score",
        "Worth showing to a group chat",
        score_rubric(vec![
            "Not something you would forward",
            "You might send it around",
            "Made to be forwarded to the group chat",
        ]),
    ));
    qs.push(question(
        "s_send_to_one_person",
        "SHAREABILITY",
        "noul",
        "Someone would send it to one person",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "s_stands_alone",
        "SHAREABILITY",
        "score",
        "It makes sense on its own",
        score_rubric(vec![
            "Only works inside a thread or reply",
            "Mostly readable on its own",
            "Fully self-contained",
        ]),
    ));
    qs.push(question(
        "s_reference_worthy",
        "SHAREABILITY",
        "score",
        "Worth saving for later",
        score_rubric(vec![
            "Nothing to save",
            "Worth bookmarking for some readers",
            "A keeper: stats, rules or a framework",
        ]),
    ));
    qs.push(question(
        "s_quotable_line",
        "SHAREABILITY",
        "noul",
        "One line can be lifted out",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "s_useful_favour",
        "SHAREABILITY",
        "noul",
        "A concrete use for a kind of reader",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "s_names_accounts",
        "SHAREABILITY",
        "noul",
        "Built around naming several people",
        TweetCriteria::Noul,
    ));

    qs.push(question(
        "t_current_event",
        "TIMELINESS",
        "noul",
        "Hangs on a recent outside event",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "t_event_angle",
        "TIMELINESS",
        "choice",
        "Its angle on the news",
        choice(vec![
            ("not_about_an_event", "Not tied to any outside event"),
            ("live_reaction", "Reacting to something happening now"),
            (
                "analysis_or_explainer",
                "Explaining what just happened or why it matters",
            ),
            ("prediction", "Predicting what happens next"),
            (
                "recap_or_result",
                "Reporting results or a recap after the fact",
            ),
            ("joke_on_event", "Memeing or joking on the news"),
        ]),
    ));
    qs.push(question(
        "t_time_position",
        "TIMELINESS",
        "choice",
        "Where it sits in time",
        choice(vec![
            ("timeless", "No time anchor; true any day"),
            ("before_the_event", "Posted in anticipation of an event"),
            ("during_the_event", "Posted while the event is unfolding"),
            ("after_the_event", "Posted after the event or outcome"),
        ]),
    ));
    qs.push(question(
        "t_names_newsworthy_entity",
        "TIMELINESS",
        "noul",
        "Names a company or person as news",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "t_meme_format",
        "TIMELINESS",
        "noul",
        "Built on a known template or meme",
        TweetCriteria::Noul,
    ));

    qs.push(question(
        "k_specificity",
        "CRAFT",
        "score",
        "How concrete the detail is",
        score_rubric(vec![
            "Vague and generic",
            "Some concrete detail",
            "Sharp, specific detail: a number, a scene, a name",
        ]),
    ));
    qs.push(question(
        "k_first_line_hook",
        "CRAFT",
        "score",
        "How much the first line pulls",
        score_rubric(vec![
            "A flat opening line",
            "The first line makes you pause",
            "The first line alone would stop a scroll",
        ]),
    ));
    qs.push(question(
        "k_payoff_inside",
        "CRAFT",
        "noul",
        "The whole thing is here, not behind a link",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "k_concrete_numbers",
        "CRAFT",
        "noul",
        "Numbers that carry weight",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "k_credential_proof",
        "CRAFT",
        "noul",
        "A result of your own backs the point",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "k_one_point",
        "CRAFT",
        "noul",
        "One clear point or one clear story",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "k_wordiness",
        "CRAFT",
        "noul",
        "Padded with filler",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "k_voice_register",
        "CRAFT",
        "choice",
        "How the voice reads",
        choice(vec![
            ("offhand_casual", "Offhand, side-comment energy"),
            ("conversational", "Talking to a friend"),
            ("professional", "Measured, work voice"),
            ("authoritative", "Speaks with authority and certainty"),
            ("humorous", "Playful and jokey"),
            ("storytelling", "Narrative and scene-driven"),
        ]),
    ));
    qs.push(question(
        "k_micro_anecdote",
        "CRAFT",
        "noul",
        "A short incident with a line that lands",
        TweetCriteria::Noul,
    ));

    qs.push(question(
        "i_first_person_own",
        "IDENTITY",
        "noul",
        "Your own experience is the subject",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "i_role",
        "IDENTITY",
        "choice",
        "Who you are speaking as",
        choice(vec![
            (
                "commentator_on_others",
                "Commenting from the outside on other people's work",
            ),
            ("founder_or_builder", "Speaking as the person building it"),
            ("expert_or_practitioner", "Speaking with hands-on expertise"),
            ("fan_or_enthusiast", "A devoted fan or enthusiast"),
            ("worker_or_insider", "An insider or worker in the industry"),
            ("outsider_or_critic", "A critic not part of the scene"),
        ]),
    ));
    qs.push(question(
        "i_second_person_advice",
        "IDENTITY",
        "noul",
        "Tells the reader what to do",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "i_addresses_group_as_peer",
        "IDENTITY",
        "noul",
        "Speaks to a named group as one of them",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "i_outgroup_target",
        "IDENTITY",
        "noul",
        "Criticises a group or a named target",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "i_ingroup_affirmation",
        "IDENTITY",
        "noul",
        "Praises the group's identity",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "i_self_intro",
        "IDENTITY",
        "noul",
        "An introduction of yourself",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "i_audience_breadth",
        "IDENTITY",
        "choice",
        "How wide the audience is",
        choice(vec![
            ("this_niche_only", "Only this niche cares"),
            ("broad_industry", "The whole industry relates"),
            ("mainstream", "Nearly everyone gets it"),
        ]),
    ));
    qs.push(question(
        "i_self_promotion",
        "IDENTITY",
        "choice",
        "How much it sells",
        choice(vec![
            ("none", "No selling"),
            ("passing_mention", "Merely mentions the product"),
            ("primary_subject", "The product is the subject"),
            ("soft_sale", "Gently steers toward a sale"),
            ("hard_sale", "Full advertorial pitch"),
        ]),
    ));

    qs.push(question(
        "f_post_type",
        "FORMAT",
        "choice",
        "What kind of post it is",
        choice(vec![
            ("commentary_on_quoted_post", "Commentary on a quoted post"),
            ("text_story", "A standalone text story or monologue"),
            ("thread", "A numbered thread"),
            ("announcement", "An announcement or update"),
            ("question", "A question to the timeline"),
            ("hot_take", "A hot take or opinion"),
            ("meme_or_gif", "Meme, gif or visual joke"),
            ("news_or_info", "News or a factual post"),
        ]),
    ));
    qs.push(question(
        "f_quoted_dependence",
        "FORMAT",
        "noul",
        "The words lean on the image or quoted post",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "f_reused_template",
        "FORMAT",
        "noul",
        "A structure readers have seen elsewhere",
        TweetCriteria::Noul,
    ));

    qs.push(question(
        "a_reply_farm",
        "ANTI-SIGNAL",
        "score",
        "Asking for replies as the point",
        score_rubric(vec![
            "Not asking for engagement",
            "Asks for likes or follows",
            "Reply bait is the whole point",
        ]),
    ));
    qs.push(question(
        "a_ai_slop",
        "ANTI-SIGNAL",
        "score",
        "Reads as machine written",
        score_rubric(vec![
            "Reads human-written",
            "Slightly stilted",
            "Smells fully machine-written",
        ]),
    ));
    qs.push(question(
        "a_platitude",
        "ANTI-SIGNAL",
        "noul",
        "A recycled maxim with nothing new",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "a_incentivised_promotion",
        "ANTI-SIGNAL",
        "noul",
        "Promotion of someone else that looks arranged",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "a_bare_announcement",
        "ANTI-SIGNAL",
        "noul",
        "Newswire tone with nobody speaking",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "a_low_effort",
        "ANTI-SIGNAL",
        "noul",
        "Nothing specific to react to",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "a_hype_caption",
        "ANTI-SIGNAL",
        "noul",
        "Excitement words with no substance",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "a_off_platform_push",
        "ANTI-SIGNAL",
        "noul",
        "Sends readers somewhere else",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "a_offensive",
        "ANTI-SIGNAL",
        "noul",
        "Abuse or harassment",
        TweetCriteria::Noul,
    ));
    qs.push(question(
        "a_politics_culture_war",
        "ANTI-SIGNAL",
        "noul",
        "Politics or culture war content",
        TweetCriteria::Noul,
    ));

    qs
}

/// Run the whole bank through one Von backend and aggregate the report.
/// Convert the 61 `TweetQuestion` bank to the lib wire questions, so either
/// backend (local von, remote openrouter/jev) can answer it.
fn bank_as_asks() -> Vec<Question> {
    let mut out: Vec<Question> = Vec::with_capacity(61);
    for q in tweet_bank() {
        out.push(Question {
            id: q.id.clone(),
            kind: q.kind.clone(),
            instructions: q.label.clone(),
            criteria: criteria_value(&q.criteria),
            temperature: None,
        });
    }
    out
}

/// Build the wire `criteria` Value for one bank question.
fn criteria_value(c: &TweetCriteria) -> Option<Value> {
    match c {
        TweetCriteria::Noul => None,
        TweetCriteria::Choice(options) => {
            let mut obj = Map::new();
            for (key, val) in options {
                match val {
                    Some(s) => {
                        obj.insert(key.clone(), json!(s));
                    }
                    None => {
                        obj.insert(key.clone(), Value::Null);
                    }
                }
            }
            Some(Value::Object(obj))
        }
        TweetCriteria::Score(rubric) => {
            let mut list: Vec<Value> = Vec::with_capacity(rubric.len());
            for item in rubric {
                list.push(json!(item));
            }
            Some(Value::Array(list))
        }
    }
}

/// Run the whole bank through one backend: local runs the ONNX model in
/// process, openrouter posts to the Decisions API (Jev). Returns the same
/// normalized rows either way, so the score is directly comparable.
fn ask_bank(text: &str, model: &str) -> Result<Asks> {
    let context = Value::String(text.to_string());
    let questions = bank_as_asks();
    jigor::ask(model, &context, &questions, None)
}

/// Map the lib's typed answers back onto the bank rows (label rendering).
fn tweet_answers(asks: &HashMap<String, Answer>, bank: &[TweetQuestion]) -> Vec<TweetAnswer> {
    let mut out: Vec<TweetAnswer> = Vec::with_capacity(bank.len());
    for q in bank {
        if let Some(ans) = asks.get(&q.id) {
            let row: TweetAnswer = match ans {
                Answer::Noul { probability } => TweetAnswer {
                    id: q.id.clone(),
                    family: q.family.clone(),
                    kind: "noul".to_string(),
                    label: q.label.clone(),
                    answer: if *probability >= 0.5 {
                        "Yes".to_string()
                    } else {
                        "No".to_string()
                    },
                    value: *probability,
                    p: Some(*probability),
                },
                Answer::Choice {
                    choice,
                    confidence,
                    probabilities,
                } => {
                    let mut desc = choice.clone();
                    if let TweetCriteria::Choice(options) = &q.criteria
                        && let Some(opt) = options.get(choice)
                        && let Some(d) = opt.as_ref()
                    {
                        desc = d.to_string();
                    }
                    TweetAnswer {
                        id: q.id.clone(),
                        family: q.family.clone(),
                        kind: "choice".to_string(),
                        label: q.label.clone(),
                        answer: desc,
                        value: probabilities.get(choice).unwrap_or(&0.0).clamp(0.0, 1.0),
                        p: Some(*confidence),
                    }
                }
                Answer::Score { score, legend, .. } => {
                    let total = legend.len();
                    let denom = (total - 1).max(1) as f32;
                    TweetAnswer {
                        id: q.id.clone(),
                        family: q.family.clone(),
                        kind: "score".to_string(),
                        label: q.label.clone(),
                        answer: format!("{} of {}", (*score).round(), total),
                        value: (*score / denom).clamp(0.0, 1.0),
                        p: None,
                    }
                }
            };
            out.push(row);
        }
    }
    out
}

/// Pure aggregation over normalized answers (no model I/O, unit-testable).
pub fn report_from_answers(answers: &[TweetAnswer]) -> TweetReport {
    let mut families: Vec<(String, FamilyStats)> = Vec::with_capacity(8);
    let mut pos_sum = 0.0_f32;
    let mut pos_n = 0usize;
    let mut anti_mean = 0.0_f32;
    for family in family_order() {
        let total = answers.iter().filter(|a| a.family == family).count();
        let mut sum = 0.0_f32;
        for a in answers {
            if a.family == family {
                sum += a.value;
            }
        }
        let fired = if total == 0 {
            0.0
        } else {
            ((sum / total as f32) * 10000.0).round() / 10000.0
        };
        families.push((
            family.clone(),
            FamilyStats {
                label: family_label(&family),
                fired,
                total,
            },
        ));
        if family == "ANTI-SIGNAL" {
            anti_mean = fired;
        } else {
            pos_sum += fired;
            pos_n += 1;
        }
    }

    let mut positives: Vec<TweetAnswer> = Vec::with_capacity(51);
    let mut anti: Vec<TweetAnswer> = Vec::with_capacity(10);
    for a in answers {
        if a.family == "ANTI-SIGNAL" {
            anti.push(a.clone());
        } else {
            positives.push(a.clone());
        }
    }

    let pos_mean = if pos_n == 0 {
        0.5
    } else {
        pos_sum / pos_n as f32
    };
    let score = (((0.65 * pos_mean) + (0.35 * (1.0 - anti_mean))) * 100.0).round() as u32;
    let beats_own_normal = ((score as f32 / 100.0) * 10000.0).round() / 10000.0;

    positives.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
    anti.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

    let mut helped: Vec<TweetAnswer> = Vec::with_capacity(3);
    for a in &positives {
        if a.value >= 0.6 && helped.len() < 3 {
            helped.push(a.clone());
        }
    }

    let mut hurt: Vec<TweetAnswer> = Vec::with_capacity(3);
    for a in &anti {
        if a.value >= 0.5 && hurt.len() < 3 {
            hurt.push(a.clone());
        }
    }
    if hurt.len() < 3 {
        let mut weakest = positives.clone();
        weakest.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
        for a in &weakest {
            if hurt.len() == 3 {
                break;
            }
            let mut seen = false;
            for h in &hurt {
                if h.id == a.id {
                    seen = true;
                    break;
                }
            }
            if !seen && a.value <= 0.4 {
                hurt.push(a.clone());
            }
        }
    }

    TweetReport {
        score,
        beats_own_normal,
        families,
        answers: answers.to_vec(),
        helped,
        hurt,
    }
}

fn tweet_answer_to_json(a: &TweetAnswer) -> Value {
    match a.p {
        Some(p) => {
            json!({"id": a.id, "family": a.family, "kind": a.kind, "label": a.label, "answer": a.answer, "value": a.value, "p": p})
        }
        None => {
            json!({"id": a.id, "family": a.family, "kind": a.kind, "label": a.label, "answer": a.answer, "value": a.value})
        }
    }
}

fn helped_hurt_to_json(a: &TweetAnswer, detail: &str) -> Value {
    match a.p {
        Some(p) => {
            json!({"id": a.id, "family": a.family, "kind": a.kind, "label": a.label, "answer": a.answer, "value": a.value, "p": p, "detail": detail, "effect": a.value})
        }
        None => {
            json!({"id": a.id, "family": a.family, "kind": a.kind, "label": a.label, "answer": a.answer, "value": a.value, "detail": detail, "effect": a.value})
        }
    }
}

/// Engagement multiple at a given score: 50 (your normal post) -> exactly
/// 1.0x; above 50 grows exponentially, below shrinks. The per-counter
/// weights are placeholders for a fitted engagement model — they are
/// tuned so a strong draft lands in the same ballpark (e.g. 79 -> ~3.2x
/// likes, ~2.6x replies, ~2.0x reposts).
fn counter_multiple(score: u32, weight: f32) -> f32 {
    let s = score as f32;
    (weight * (s - 50.0) / 50.0).exp()
}

fn counter_json(score: u32, weight: f32, confidence: &str) -> Value {
    let multiple = counter_multiple(score, weight);
    let probability = 1.0 / (1.0 + (-((score as f32 - 53.0) / 16.0)).exp());
    json!({
        "multiple": (multiple * 100.0).round() / 100.0,
        "p75": (multiple * 1.9 * 100.0).round() / 100.0,
        "p90": (multiple * 4.2 * 100.0).round() / 100.0,
        "breakout_share": (probability * 0.22 * 1000.0).round() / 1000.0,
        "probability": (probability * 10000.0).round() / 10000.0,
        "confidence": confidence,
        "own_median": Value::Null,
        "expected": Value::Null,
    })
}

/// Serialize a report to the `/v1/tweet-score` wire shape (an
/// anonymous-account viral-score response shape).
pub fn tweet_report_to_json(report: &TweetReport, model: &str) -> Value {
    let mut families = Map::new();
    for (family, stats) in &report.families {
        families.insert(
            family.clone(),
            json!({"label": stats.label, "fired": stats.fired, "total": stats.total}),
        );
    }
    let helped: Vec<Value> = report
        .helped
        .iter()
        .map(|a| helped_hurt_to_json(a, "Stronger than your usual post"))
        .collect();
    let hurt: Vec<Value> = report
        .hurt
        .iter()
        .map(|a| {
            helped_hurt_to_json(
                a,
                if a.family == "ANTI-SIGNAL" {
                    "A warning sign fired"
                } else {
                    "Weaker than your usual post"
                },
            )
        })
        .collect();
    let answers: Vec<Value> = report.answers.iter().map(tweet_answer_to_json).collect();

    let mut counters = Map::new();
    counters.insert(
        "likes".to_string(),
        counter_json(report.score, 2.0, "normal"),
    );
    counters.insert(
        "replies".to_string(),
        counter_json(report.score, 1.6, "normal"),
    );
    counters.insert(
        "reposts_and_quotes".to_string(),
        counter_json(report.score, 1.1, "low"),
    );
    counters.insert(
        "views".to_string(),
        counter_json(report.score, 1.4, "normal"),
    );

    let engine = json!({
        "model": model,
        "question_set": "v1.1",
        "note": "MVP: transparent heuristic over the 61-question bank; counters and probabilities are placeholders for a fitted engagement model.",
    });
    json!({
        "score": report.score,
        "beats_own_normal": report.beats_own_normal,
        "families": families,
        "counters": counters,
        "helped": helped,
        "hurt": hurt,
        "answers": answers,
        "engine": engine,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(
        id: &str,
        family: &str,
        kind: &str,
        label: &str,
        value: f32,
        text: &str,
        p: Option<f32>,
    ) -> TweetAnswer {
        TweetAnswer {
            id: id.to_string(),
            family: family.to_string(),
            kind: kind.to_string(),
            label: label.to_string(),
            value,
            answer: text.to_string(),
            p,
        }
    }

    fn uniform(overall: f32) -> Vec<TweetAnswer> {
        let mut out: Vec<TweetAnswer> = Vec::with_capacity(61);
        for q in tweet_bank() {
            out.push(answer(
                &q.id,
                &q.family,
                &q.kind,
                &q.label,
                overall,
                "Yes",
                Some(overall),
            ));
        }
        out
    }

    fn family_stats(report: &TweetReport, family: &str) -> FamilyStats {
        for (f, stats) in &report.families {
            if f == family {
                return stats.clone();
            }
        }
        panic!("family {family} missing")
    }

    #[tokio::test]
    async fn bank_has_61_questions() {
        let bank = tweet_bank();
        assert_eq!(bank.len(), 61);
    }

    #[tokio::test]
    async fn bank_covers_every_family() {
        let bank = tweet_bank();
        for family in family_order() {
            let mut n = 0usize;
            for q in &bank {
                if q.family == family {
                    n += 1;
                }
            }
            assert!(n > 0);
        }
    }

    #[tokio::test]
    async fn neutral_post_scores_midpoint() {
        let report = report_from_answers(&uniform(0.5));
        assert_eq!(report.score, 50);
        assert_eq!(family_stats(&report, "EMOTION").total, 9);
        assert_eq!(family_stats(&report, "ANTI-SIGNAL").total, 10);
    }

    #[tokio::test]
    async fn perfect_content_clean_signal_scores_100() {
        let mut answers = uniform(1.0);
        for a in &mut answers {
            if a.family == "ANTI-SIGNAL" {
                a.value = 0.0;
            }
        }
        assert_eq!(report_from_answers(&answers).score, 100);
    }

    #[tokio::test]
    async fn reply_farm_wipes_the_score() {
        let mut answers = uniform(1.0);
        for a in &mut answers {
            if a.family == "ANTI-SIGNAL" {
                a.value = 1.0;
            }
        }
        assert_eq!(report_from_answers(&answers).score, 65);
    }

    #[tokio::test]
    async fn family_stats_mean_and_fired() {
        let answers: Vec<TweetAnswer> = vec![
            answer(
                "e_humour",
                "EMOTION",
                "score",
                "How funny it is to your audience",
                1.0,
                "2 of 2",
                None,
            ),
            answer(
                "e_vulnerability",
                "EMOTION",
                "noul",
                "Admits a failure or a worry",
                0.0,
                "No",
                Some(0.0),
            ),
        ];
        let stats = family_stats(&report_from_answers(&answers), "EMOTION");
        assert_eq!(stats.fired, 0.5);
        assert_eq!(stats.total, 2);
    }

    #[tokio::test]
    async fn helped_only_strong_positive_signals() {
        let answers: Vec<TweetAnswer> = vec![
            answer(
                "k_credential_proof",
                "CRAFT",
                "noul",
                "A result of your own backs the point",
                0.95,
                "Yes",
                Some(0.95),
            ),
            answer(
                "e_humour",
                "EMOTION",
                "score",
                "How funny it is to your audience",
                0.55,
                "1 of 2",
                None,
            ),
            answer(
                "s_group_chat",
                "SHAREABILITY",
                "score",
                "Worth showing to a group chat",
                0.9,
                "2 of 2",
                None,
            ),
            answer(
                "a_reply_farm",
                "ANTI-SIGNAL",
                "score",
                "Asking for replies as the point",
                0.9,
                "2 of 2",
                None,
            ),
        ];
        let report = report_from_answers(&answers);
        assert_eq!(report.helped.len(), 2);
        assert_eq!(report.helped[0].id, "k_credential_proof");
        assert_eq!(report.helped[1].id, "s_group_chat");
        assert_eq!(report.hurt.len(), 1);
        assert_eq!(report.hurt[0].id, "a_reply_farm");
    }

    #[tokio::test]
    async fn anti_signal_fired_always_lands_in_hurt() {
        let answers: Vec<TweetAnswer> = vec![
            answer(
                "a_ai_slop",
                "ANTI-SIGNAL",
                "score",
                "Reads as machine written",
                0.6,
                "1 of 2",
                None,
            ),
            answer(
                "a_platitude",
                "ANTI-SIGNAL",
                "noul",
                "A recycled maxim with nothing new",
                0.9,
                "Yes",
                Some(0.9),
            ),
            answer(
                "s_group_chat",
                "SHAREABILITY",
                "score",
                "Worth showing to a group chat",
                0.3,
                "0 of 2",
                None,
            ),
        ];
        let hurt_ids = report_from_answers(&answers)
            .hurt
            .iter()
            .map(|a| a.id.clone())
            .collect::<Vec<String>>();
        assert!(hurt_ids.contains(&"a_ai_slop".to_string()));
        assert!(hurt_ids.contains(&"a_platitude".to_string()));
        assert!(hurt_ids.contains(&"s_group_chat".to_string()));
    }

    #[tokio::test]
    async fn report_json_shape() {
        let report = report_from_answers(&uniform(0.6));
        let v = tweet_report_to_json(&report, "von-1.0.0");
        let fams = v.get("families").and_then(Value::as_object).unwrap();
        assert!(fams.contains_key("EMOTION"));
        assert!(fams.contains_key("ANTI-SIGNAL"));
        let answers = v.get("answers").and_then(Value::as_array).unwrap();
        assert_eq!(answers.len(), 61);
        assert_eq!(
            v.get("engine")
                .unwrap()
                .get("question_set")
                .unwrap()
                .clone(),
            json!("v1.1")
        );
        let counters = v.get("counters").and_then(Value::as_object).unwrap();
        assert!(counters.contains_key("likes"));
        assert!(counters.contains_key("reposts_and_quotes"));
        assert_eq!(
            counters
                .get("reposts_and_quotes")
                .unwrap()
                .get("confidence")
                .unwrap()
                .clone(),
            json!("low")
        );
        let beats = v.get("beats_own_normal").unwrap().as_f64().unwrap();
        assert!(beats > 0.52 && beats < 0.54, "53/100");
    }

    #[tokio::test]
    async fn counters_are_1x_at_neutral_score() {
        let report = report_from_answers(&uniform(0.5));
        assert_eq!(report.score, 50);
        let v = tweet_report_to_json(&report, "von-1.0.0");
        let counters = v.get("counters").and_then(Value::as_object).unwrap();
        assert_eq!(
            counters
                .get("likes")
                .unwrap()
                .get("multiple")
                .unwrap()
                .clone(),
            json!(1.0)
        );
        assert_eq!(
            counters
                .get("replies")
                .unwrap()
                .get("multiple")
                .unwrap()
                .clone(),
            json!(1.0)
        );
        let reposts = counters.get("reposts_and_quotes").unwrap();
        assert_eq!(reposts.get("confidence").unwrap().clone(), json!("low"));
    }

    #[tokio::test]
    async fn counters_scale_with_score() {
        let mut answers = uniform(1.0);
        for a in &mut answers {
            if a.family == "ANTI-SIGNAL" {
                a.value = 0.0;
            }
        }
        let report = report_from_answers(&answers);
        assert_eq!(report.score, 100);
        let v = tweet_report_to_json(&report, "von-1.0.0");
        let likes = v.get("counters").unwrap().get("likes").unwrap();
        let multiple = likes.get("multiple").unwrap().as_f64().unwrap();
        assert!(multiple > 7.0 && multiple < 8.0, "exp(2) ~= 7.39");
        assert!(
            likes.get("p90").unwrap().as_f64().unwrap()
                > likes.get("p75").unwrap().as_f64().unwrap()
        );
    }

    #[tokio::test]
    async fn typed_answers_render_bank_rows() {
        // the same typed answers any backend would return, mapped back
        let mut asks = HashMap::new();
        asks.insert(
            "e_milestone_joy".to_string(),
            Answer::Noul { probability: 0.93 },
        );
        let mut probs = HashMap::new();
        probs.insert("none".to_string(), 0.1_f32);
        probs.insert("joy_or_pride".to_string(), 0.9_f32);
        asks.insert(
            "e_dominant_emotion".to_string(),
            Answer::Choice {
                choice: "joy_or_pride".to_string(),
                confidence: 0.8,
                probabilities: probs,
            },
        );
        let mut legend = HashMap::new();
        legend.insert("0".to_string(), "Not trying to be funny".to_string());
        legend.insert("1".to_string(), "A light touch of humour".to_string());
        legend.insert(
            "2".to_string(),
            "Genuinely funny to the audience".to_string(),
        );
        let mut spl = HashMap::new();
        spl.insert("1".to_string(), 1.0_f32);
        asks.insert(
            "e_humour".to_string(),
            Answer::Score {
                score: 1.99,
                confidence: 0.99,
                probabilities: spl,
                legend,
            },
        );

        let rows = tweet_answers(&asks, &tweet_bank());
        assert_eq!(rows.len(), 3);
        // bank order: e_dominant_emotion, e_milestone_joy, e_humour
        match &rows[0] {
            TweetAnswer {
                id,
                kind,
                answer,
                value,
                ..
            } => {
                assert_eq!(id, "e_dominant_emotion");
                assert_eq!(kind, "choice");
                assert_eq!(
                    answer,
                    "Excitement, pride or satisfaction about an achievement"
                );
                assert!(*value > 0.899 && *value < 0.901, "p of chosen option");
            }
            _ => panic!("expected choice row"),
        }
        match &rows[1] {
            TweetAnswer {
                id,
                kind,
                answer,
                value,
                ..
            } => {
                assert_eq!(id, "e_milestone_joy");
                assert_eq!(kind, "noul");
                assert_eq!(answer, "Yes");
                assert!(*value > 0.9 && *value < 0.95, "p=0.93");
            }
            _ => panic!("expected noul row"),
        }
        match &rows[2] {
            TweetAnswer {
                id, answer, value, ..
            } => {
                assert_eq!(id, "e_humour");
                assert_eq!(answer, "2 of 3");
                assert!(*value > 0.994 && *value < 0.996, "1.99/2");
            }
            _ => panic!("expected score row"),
        }
    }

    #[tokio::test]
    async fn bank_roundtrips_through_lib_wire_shapes() {
        let asks = bank_as_asks();
        assert_eq!(asks.len(), 61);
        for q in asks {
            match q.criteria {
                Some(Value::Object(m)) => assert!(m.len() > 0),
                Some(Value::Array(items)) => assert_eq!(items.len(), 3),
                _ => assert_eq!(q.kind, "noul"),
            }
        }
    }
}
