//! How much int8 weights change a model: on one text, the model sees each
//! token's prefix and predicts the next, in f32 and then with its weights
//! quantized. Reports next-token agreement between the two and perplexity.
//!
//!     cargo run --release -p inference --example eval_int8 -- [text file] [tokens]
//!
//! The default text is this repository's tokenizer source, which the model has
//! not been tuned on. Needs data/qwen2.5-coder-0.5b/.

mod common;

use inference::cpu::CpuBackend;
use inference::llama::Llama;
use inference::model::Model;
use inference::ops::argmax;
use std::time::Instant;

struct Pass {
    /// Negative log-likelihood of each true next token, in nats.
    nll: Vec<f64>,
    /// The token the model ranks first at each position.
    top1: Vec<u32>,
    secs: f64,
}

/// Feeds `ids` one at a time through the cache (the decode path) and scores
/// the model's prediction for each following token.
fn teacher_forced(model: &impl Model, ids: &[u32]) -> Pass {
    let mut cache = model.new_cache();
    let (mut nll, mut top1) = (Vec::new(), Vec::new());
    let start = Instant::now();
    for pair in ids.windows(2) {
        let logits = model.forward(&pair[..1], &mut cache);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
        nll.push(max as f64 + sum.ln() - logits[pair[1] as usize] as f64);
        top1.push(argmax(&logits));
    }
    Pass {
        nll,
        top1,
        secs: start.elapsed().as_secs_f64(),
    }
}

fn perplexity(nll: &[f64]) -> f64 {
    (nll.iter().sum::<f64>() / nll.len() as f64).exp()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/../tokenizer/src/tokenizer.rs").into());
    let n: usize = args.next().map_or(512, |n| n.parse().unwrap());
    let ids = common::tokenizer(common::QWEN).encode(&std::fs::read_to_string(&path).unwrap());
    let ids = &ids[..n.min(ids.len())];

    let mut model = Llama::from_dir(&common::data_dir(common::QWEN), CpuBackend::default());
    let f32_pass = teacher_forced(&model, ids);
    model.quantize();
    let q_pass = teacher_forced(&model, ids);

    let steps = f32_pass.top1.len();
    let agree = f32_pass
        .top1
        .iter()
        .zip(&q_pass.top1)
        .filter(|(a, b)| a == b)
        .count();
    let (p32, p8) = (perplexity(&f32_pass.nll), perplexity(&q_pass.nll));
    println!("text                 {path}, {} tokens", ids.len());
    println!(
        "next-token agreement {agree} of {steps} positions ({:.1}%) pick the same top-1 token",
        100.0 * agree as f64 / steps as f64
    );
    println!(
        "perplexity           f32 {p32:.3}, int8 {p8:.3} ({:+.2}%)",
        100.0 * (p8 / p32 - 1.0)
    );
    println!(
        "decode tok/s         f32 {:.1}, int8 {:.1} (cached, one token per step)",
        steps as f64 / f32_pass.secs,
        steps as f64 / q_pass.secs
    );
}
