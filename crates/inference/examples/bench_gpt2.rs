//! Prefill and decode speed of GPT-2 on the CPU backend, against the memory
//! bandwidth roofline.
//!
//!     cargo run --release -p inference --example bench_gpt2
//!
//! `INFERENCE_THREADS=n` sets the model's thread count (default: all cores).
//! Needs data/gpt2/model.safetensors (see crates/inference/scripts).

use inference::backend::{AttnShape, Backend, Matrix};
use inference::cpu::CpuBackend;
use inference::gpt2::Gpt2;
use inference::kernel::dot;
use inference::model::Model;
use inference::ops::argmax;
use inference::pool::Pool;
use inference::safetensors::load;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;
use tokenizer::Tokenizer;

const PROMPT: &str = "The history of computing is a story of abstraction. Early machines were \
programmed by rewiring them, then by punching cards, then by writing assembly, and finally by \
describing intent in languages that compile down to the instructions a processor understands. \
Each layer hides the one below it, which makes engineers productive and makes performance \
mysterious. A transformer language model is one more layer: text becomes integers, integers \
become vectors, and a stack of attention and feed-forward blocks turns those vectors into a \
probability for every possible next word.";
const NEW_TOKENS: usize = 64;

/// Bytes per second all cores can stream from memory: each thread reads its
/// share of a 1 GiB array, far larger than any cache, with the model's own dot
/// kernel. Passes are timed one by one and the fastest wins: bandwidth is a
/// ceiling, and a pass slowed by a cold or busy core measures that, not memory.
fn memory_bandwidth() -> f64 {
    let pool = Pool::new(std::thread::available_parallelism().map_or(1, |n| n.get()));
    let data = vec![1.0f32; 256 << 20];
    let chunks: Vec<&[f32]> = data.chunks(1 << 16).collect();
    let pass = || {
        let start = Instant::now();
        pool.run(chunks.len(), &|i| {
            std::hint::black_box(dot(chunks[i], chunks[i]));
        });
        start.elapsed().as_secs_f64()
    };
    pass();
    let best = (0..20).map(|_| pass()).fold(f64::INFINITY, f64::min);
    (data.len() * 4) as f64 / best
}

/// Wraps a backend and adds up the time spent in each op: where a token goes.
struct Timed<B> {
    inner: B,
    secs: RefCell<[f64; 8]>,
}

const OPS: [&str; 8] = [
    "matmul",
    "layernorm",
    "gelu",
    "softmax",
    "attention",
    "rmsnorm",
    "swiglu",
    "rope",
];

impl<B: Backend> Timed<B> {
    fn time<R>(&self, op: usize, f: impl FnOnce(&B) -> R) -> R {
        let start = Instant::now();
        let r = f(&self.inner);
        self.secs.borrow_mut()[op] += start.elapsed().as_secs_f64();
        r
    }
}

impl<B: Backend> Backend for Timed<B> {
    fn matmul(&self, x: &[f32], t: usize, w: &Matrix, bias: Option<&[f32]>) -> Vec<f32> {
        self.time(0, |b| b.matmul(x, t, w, bias))
    }
    fn layernorm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        self.time(1, |b| b.layernorm(x, gamma, beta, eps))
    }
    fn gelu(&self, x: &mut [f32]) {
        self.time(2, |b| b.gelu(x))
    }
    fn softmax(&self, x: &mut [f32]) {
        self.time(3, |b| b.softmax(x))
    }
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], shape: AttnShape) -> Vec<f32> {
        self.time(4, |b| b.attention(q, k, v, shape))
    }
    fn rmsnorm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        self.time(5, |b| b.rmsnorm(x, gamma, eps))
    }
    fn swiglu(&self, gate_up: &[f32], width: usize) -> Vec<f32> {
        self.time(6, |b| b.swiglu(gate_up, width))
    }
    fn rope(&self, x: &mut [f32], pos0: usize, n_heads: usize, head_dim: usize, inv_freq: &[f32]) {
        self.time(7, |b| b.rope(x, pos0, n_heads, head_dim, inv_freq))
    }
}

/// Milliseconds per op for one prefill of `ids` and per decoded token, plus
/// how much of each pass the ops account for (the rest is copies, residual
/// adds and allocation).
fn breakdown(tensors: Tensors, ids: &[u32]) {
    let model = Gpt2::from_tensors(
        tensors,
        Timed {
            inner: CpuBackend::default(),
            secs: RefCell::new([0.0; 8]),
        },
    );
    let report = |label: &str, total: f64, secs: [f64; 8], per: f64| {
        let parts: Vec<String> = OPS
            .iter()
            .zip(secs)
            .map(|(name, s)| format!("{name} {:.2}", s * per * 1e3))
            .collect();
        let other = total - secs.iter().sum::<f64>();
        println!(
            "{label:8} {:.2} ms: {}, other {:.2}",
            total * per * 1e3,
            parts.join(", "),
            other * per * 1e3
        );
    };

    let mut cache = model.new_cache();
    let start = Instant::now();
    let mut logits = model.forward(ids, &mut cache);
    let total = start.elapsed().as_secs_f64();
    report(
        "prefill",
        total,
        std::mem::take(&mut *model.backend().secs.borrow_mut()),
        1.0,
    );

    let steps = 32;
    let start = Instant::now();
    for _ in 0..steps {
        logits = model.forward(&[argmax(&logits)], &mut cache);
    }
    let total = start.elapsed().as_secs_f64();
    report(
        "decode",
        total,
        std::mem::take(&mut *model.backend().secs.borrow_mut()),
        1.0 / steps as f64,
    );
}

type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/data/gpt2/");
    let tensors = load(format!("{dir}model.safetensors").as_ref());
    // Every matmul weight is read once per decoded token. Token embedding
    // counts once, as the output projection: the lookup reads a single row.
    let weight_bytes: usize = tensors
        .iter()
        .filter(|(name, _)| {
            name.ends_with("weight") && !name.contains("ln_") && !name.starts_with("wpe")
        })
        .map(|(_, (_, data))| data.len() * 4)
        .sum();
    let model = Gpt2::from_tensors(tensors, CpuBackend::default());
    let tok = Tokenizer::from_gpt2_merges(include_str!("../../tokenizer/data/gpt2/merges.txt"));
    let ids = tok.encode(PROMPT);

    let bandwidth = memory_bandwidth();

    // Prefill: the whole prompt in one pass.
    let mut prefill = Vec::new();
    for _ in 0..7 {
        let mut cache = model.new_cache();
        let start = Instant::now();
        std::hint::black_box(model.forward(&ids, &mut cache));
        prefill.push(start.elapsed().as_secs_f64());
    }
    let prefill_s = median(prefill);

    // Decode: one token at a time against the cache.
    let mut cache = model.new_cache();
    let mut logits = model.forward(&ids, &mut cache);
    let mut steps = Vec::new();
    for _ in 0..NEW_TOKENS {
        let next = argmax(&logits);
        let start = Instant::now();
        logits = model.forward(&[next], &mut cache);
        steps.push(start.elapsed().as_secs_f64());
    }
    let step_s = median(steps);

    // Decode also reads the whole KV cache each token; use the average length.
    let kv_bytes = (cache.bytes_per_token() * (ids.len() + NEW_TOKENS / 2)) as f64;
    let roofline = bandwidth / (weight_bytes as f64 + kv_bytes);
    println!(
        "threads              {}",
        std::env::var("INFERENCE_THREADS").unwrap_or("all".into())
    );
    println!(
        "read per decoded tok {:.0} MB weights + {:.1} MB KV cache",
        weight_bytes as f64 / 1e6,
        kv_bytes / 1e6
    );
    println!(
        "memory bandwidth     {:.1} GB/s (best of 20 streaming passes)",
        bandwidth / 1e9
    );
    println!(
        "prefill              {} tokens in {:.1} ms = {:.0} tok/s",
        ids.len(),
        prefill_s * 1e3,
        ids.len() as f64 / prefill_s
    );
    println!(
        "decode               {:.2} ms/token = {:.1} tok/s (median of {NEW_TOKENS})",
        step_s * 1e3,
        1.0 / step_s
    );
    println!(
        "roofline             {:.1} tok/s, decode reaches {:.0}% of it",
        roofline,
        100.0 * (1.0 / step_s) / roofline
    );
    breakdown(load(format!("{dir}model.safetensors").as_ref()), &ids);
}
