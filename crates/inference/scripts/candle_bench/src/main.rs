//! Times `candle` on a Qwen2 GGUF file the way `bench` times this project's
//! engine: median of 7 prefills, then median of 64 single-token decodes.
//! Run through `../compare.py`.
//!
//!     candle_bench <model.gguf> <prompt token ids, comma separated>

use candle_core::quantized::gguf_file;
use candle_core::{Device, Result, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::time::Instant;

const PREFILLS: usize = 7;
const NEW_TOKENS: usize = 64;

fn argmax(logits: &Tensor) -> Result<u32> {
    let row = logits.squeeze(0)?.to_vec1::<f32>()?;
    let best = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1));
    Ok(best.expect("empty logits").0 as u32)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let usage = "usage: candle_bench <model.gguf> <id,id,...>";
    let mut args = std::env::args().skip(1);
    let path = args.next().expect(usage);
    let ids: Vec<u32> = args
        .next()
        .expect(usage)
        .split(',')
        .map(|s| s.parse().expect(usage))
        .collect();

    let device = Device::Cpu;
    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;
    let prompt = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;

    let mut prefill = Vec::new();
    for _ in 0..PREFILLS {
        model.clear_kv_cache();
        let start = Instant::now();
        std::hint::black_box(model.forward(&prompt, 0)?);
        prefill.push(start.elapsed().as_secs_f64());
    }

    model.clear_kv_cache();
    let mut logits = model.forward(&prompt, 0)?;
    let (mut generated, mut steps) = (Vec::new(), Vec::new());
    for step in 0..NEW_TOKENS {
        let next = argmax(&logits)?;
        generated.push(next);
        let x = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        let start = Instant::now();
        logits = model.forward(&x, ids.len() + step)?;
        steps.push(start.elapsed().as_secs_f64());
    }

    println!("prefill_tok_s {}", ids.len() as f64 / median(prefill));
    println!("decode_tok_s {}", 1.0 / median(steps));
    println!("greedy ids {:?}", &generated[..16]);
    Ok(())
}
