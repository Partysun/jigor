//! Example mirroring `main.py` and `main_branch.py` on the unified
//! `noul`/`choice`/`score` API — one `answers()` call per step.

use jigor::{Answer, Backend, Error, Result, VonBackend, choice_pairs, noul, score};
use serde_json::json;

fn main() -> Result<()> {
    // ort 2.0 auto-initializes; explicit init optional
    let _ = ort::init().commit();

    let mut von = VonBackend::new()?;

    // 1. decide — same as main.py
    let ans = von.answers(
        &json!("Database replication lag on cluster us-west-2 exceeded 45 seconds."),
        &[choice_pairs(
            "root_cause",
            "Classify the root cause domain of this incident.",
            &[
                (
                    "infrastructure",
                    "Database, hardware, network, or server failures",
                ),
                (
                    "billing",
                    "Invoices, payments, refunds, subscription queries",
                ),
                ("feature_request", "Requests for new platform capabilities"),
            ],
        )],
        None,
    )?;
    match ans.get("root_cause").unwrap() {
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            println!("{}", choice);
            println!("{:.3}", confidence);
            // pretty probabilities like Python dict
            println!(
                "{{'infrastructure': {:.4}, 'billing': {:.4}, 'feature_request': {:.4}}}",
                probabilities.get("infrastructure").unwrap_or(&0.0),
                probabilities.get("billing").unwrap_or(&0.0),
                probabilities.get("feature_request").unwrap_or(&0.0)
            );
        }
        other => {
            return Err(Error::internal(format!(
                "expected choice, got {}",
                other.kind()
            )));
        }
    }

    // 2. judge
    let p = von.answers(
        &json!("Connection pool exhausted on port 5432; subsequent handshakes timing out."),
        &[noul(
            "blocking",
            "Is this issue actively blocking customer operations?",
        )],
        None,
    )?;
    match p.get("blocking").unwrap() {
        Answer::Noul { probability } => println!("judge: {:.4}", *probability),
        other => {
            return Err(Error::internal(format!(
                "expected noul, got {}",
                other.kind()
            )));
        }
    }

    // 3. rate
    let r = von.answers(
        &json!("Memory utilization reached 98% with frequent OOM killer invocations."),
        &[score(
            "severity",
            "Assess system degradation level.",
            &[
                "Nominal operation; within acceptable variance",
                "Elevated resource consumption; degraded performance",
                "Critical threshold; immediate risk of service termination",
            ],
        )],
        None,
    )?;
    match r.get("severity").unwrap() {
        Answer::Score {
            score,
            confidence,
            probabilities,
            ..
        } => println!(
            "rate score: {:.2} conf: {:.3} probs: {:?}",
            *score, *confidence, probabilities
        ),
        other => {
            return Err(Error::internal(format!(
                "expected score, got {}",
                other.kind()
            )));
        }
    }

    // 4. fan-out example: several question kinds in one ask
    let intent = von.answers(
        &json!("Payment gateway reports timeout on charge authorizations. Urgent."),
        &[choice_pairs(
            "intent",
            "What is the operational nature of this ticket?",
            &[
                (
                    "payment_failure",
                    "Failures processing charges, gateway timeouts, credit card declines",
                ),
                (
                    "access_issue",
                    "Login, SSO, authentication, or permission errors",
                ),
            ],
        )],
        None,
    )?;
    match intent.get("intent").unwrap() {
        Answer::Choice {
            choice, confidence, ..
        } => {
            println!("fan-out intent: {} {:.3}", choice, *confidence);
        }
        other => {
            return Err(Error::internal(format!(
                "expected choice, got {}",
                other.kind()
            )));
        }
    }

    Ok(())
}
