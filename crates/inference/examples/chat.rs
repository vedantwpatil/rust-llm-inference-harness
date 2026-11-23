//! Chat with a Llama-style checkpoint on the CPU backend; the reply streams as
//! it is generated.
//!
//!     cargo run --release -p inference --example chat -- [--int8] [--max-new N] "prompt"
//!
//! Needs data/qwen2.5-coder-0.5b/ (see crates/inference/scripts/fetch_model.py).

mod common;

use inference::cpu::CpuBackend;
use inference::llama::Llama;
use inference::model::Model;
use std::io::Write;
use std::time::Instant;
use tokenizer::chatml;

fn main() {
    let (mut int8, mut max_new, mut prompt) = (false, 256, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--int8" => int8 = true,
            "--max-new" => max_new = args.next().unwrap().parse().unwrap(),
            _ => prompt = Some(arg),
        }
    }
    let prompt = prompt.expect("usage: chat [--int8] [--max-new N] \"prompt\"");

    let tok = common::tokenizer(common::QWEN);
    let mut model = Llama::from_dir(&common::data_dir(common::QWEN), CpuBackend::default());
    if int8 {
        model.quantize();
    }
    let eos = model.config().eos.clone();
    let ids = tok.encode(&chatml(common::SYSTEM, &prompt));

    let start = Instant::now();
    let (mut first, mut generated) = (None, 0usize);
    // Bytes of a character cut between two tokens wait here for the next one.
    let mut pending = Vec::new();
    let mut out = std::io::stdout().lock();
    model.generate_each(&ids, max_new, &eos, &mut |id| {
        first.get_or_insert_with(|| start.elapsed());
        generated += 1;
        if eos.contains(&id) {
            return;
        }
        match tok.decode_stream(id, &mut pending) {
            Ok(text) => write!(out, "{text}").unwrap(),
            Err(_) => {
                pending.clear();
                write!(out, "\u{FFFD}").unwrap();
            }
        }
        out.flush().unwrap();
    });
    let total = start.elapsed();
    let first = first.unwrap_or(total);
    // The first token comes out of prefill, so the rate counts only the rest.
    let rate = match generated {
        0 | 1 => "n/a".to_string(),
        n => format!("{:.1}", (n - 1) as f64 / (total - first).as_secs_f64()),
    };
    eprintln!(
        "\n[{} prompt tokens, first token after {:.0} ms; {generated} tokens generated, {rate} tok/s after the first{}]",
        ids.len(),
        first.as_secs_f64() * 1e3,
        if int8 { ", int8" } else { "" }
    );
}
