use jigor::{Backend, Result, VonBackend, choice, noul};
use serde_json::json;
use std::time::Instant;

fn main() -> Result<()> {
    let _ = ort::init().commit();
    let mut von = VonBackend::new()?;

    // the bench workload: a 3-option choice plus a noul, one answers() call
    let questions = vec![
        choice(
            "domain",
            "Classify the root cause domain of this incident.",
            &["infrastructure", "billing", "feature_request"],
        ),
        noul(
            "blocking",
            "Is this issue actively blocking customer operations?",
        ),
    ];
    let state = json!("Database replication lag on cluster us-west-2 exceeded 45 seconds.");

    // warmup
    for _ in 0..3 {
        von.answers(&state, &questions, None)?;
    }

    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        von.answers(&state, &questions, None)?;
    }
    let elapsed = start.elapsed();
    println!(
        "rust von {} asks (choice+noul): {:.3}s avg {:.2}ms {:.2} asks/s",
        n,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / n as f64,
        n as f64 / elapsed.as_secs_f64()
    );

    // latency distribution
    let mut times = Vec::new();
    for _ in 0..20 {
        let t0 = Instant::now();
        von.answers(&state, &questions, None)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let p95 = times[(times.len() as f64 * 0.95) as usize];
    println!(
        "rust latency p50 {:.2}ms p95 {:.2}ms min {:.2} max {:.2}",
        median,
        p95,
        times[0],
        times[times.len() - 1]
    );

    // noul-only: 30 answers() calls with a single yes/no question
    let noul_only = vec![noul(
        "blocking",
        "Is this issue actively blocking customer operations?",
    )];
    let t0 = Instant::now();
    for _ in 0..30 {
        von.answers(&state, &noul_only, None)?;
    }
    let e = t0.elapsed();
    println!(
        "rust judge 30: {:.3}s avg {:.2}ms",
        e.as_secs_f64(),
        e.as_secs_f64() * 1000.0 / 30.0
    );

    Ok(())
}
