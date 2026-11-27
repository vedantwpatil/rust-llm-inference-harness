use crate::backend::{AttnShape, Backend, Matrix};
use crate::config::LlamaConfig;
use crate::kv_cache::KvCache;
use crate::model::{Model, Trace, note};
use crate::ops::{add_in_place, split_columns};
use crate::safetensors::{Tensors, load, take};
use std::path::Path;

/// One decoder layer. The query, key and value projections are stacked into
/// one matrix at load, and so are the gate and up projections: one matmul (one
/// parallel region, one pass over the activations) instead of three or two.
struct Layer {
    attn_norm: Vec<f32>,
    qkv: Matrix,
    /// Qwen2 has biases on query, key and value; Llama has none.
    qkv_bias: Option<Vec<f32>>,
    o_proj: Matrix,
    mlp_norm: Vec<f32>,
    gate_up: Matrix,
    down: Matrix,
}

impl Layer {
    fn matrices(&self) -> [&Matrix; 4] {
        [&self.qkv, &self.o_proj, &self.gate_up, &self.down]
    }

    fn matrices_mut(&mut self) -> [&mut Matrix; 4] {
        [
            &mut self.qkv,
            &mut self.o_proj,
            &mut self.gate_up,
            &mut self.down,
        ]
    }
}

/// A Llama-style decoder: token embedding only (position lives in the rotary
/// embedding), RMSNorm, grouped-query attention, SwiGLU feed-forward. Covers
/// Llama 3 and Qwen2, whose differences are all in `LlamaConfig` and the
/// optional biases.
pub struct Llama<B: Backend> {
    backend: B,
    cfg: LlamaConfig,
    inv_freq: Vec<f32>,
    embed: Matrix,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    /// `None` when the output projection is the embedding table.
    lm_head: Option<Matrix>,
}

fn linear(t: &mut Tensors, name: &str) -> Matrix {
    let (shape, data) = take(t, name);
    Matrix::new(shape[0], shape[1], data)
}

/// Stacks the rows of weight matrices that share an input width.
fn stacked(t: &mut Tensors, names: &[String]) -> Matrix {
    let (mut rows, mut cols, mut data) = (0, None, Vec::new());
    for name in names {
        let (shape, d) = take(t, name);
        assert_eq!(
            *cols.get_or_insert(shape[1]),
            shape[1],
            "{name} has another width"
        );
        rows += shape[0];
        data.extend(d);
    }
    Matrix::new(rows, cols.unwrap(), data)
}

impl<B: Backend> Llama<B> {
    /// Builds the model from `safetensors::load`'s output, consuming it.
    pub fn from_tensors(mut t: Tensors, cfg: LlamaConfig, backend: B) -> Self {
        let embed = linear(&mut t, "model.embed_tokens.weight");
        let layers = (0..cfg.n_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                let names = |parts: &[&str], suffix: &str| -> Vec<String> {
                    parts.iter().map(|x| format!("{p}.{x}.{suffix}")).collect()
                };
                let qkv_parts = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"];
                let bias_names = names(&qkv_parts, "bias");
                let qkv_bias = t
                    .contains_key(&bias_names[0])
                    .then(|| bias_names.iter().flat_map(|n| take(&mut t, n).1).collect());
                Layer {
                    attn_norm: take(&mut t, &format!("{p}.input_layernorm.weight")).1,
                    qkv: stacked(&mut t, &names(&qkv_parts, "weight")),
                    qkv_bias,
                    o_proj: linear(&mut t, &format!("{p}.self_attn.o_proj.weight")),
                    mlp_norm: take(&mut t, &format!("{p}.post_attention_layernorm.weight")).1,
                    gate_up: stacked(&mut t, &names(&["mlp.gate_proj", "mlp.up_proj"], "weight")),
                    down: linear(&mut t, &format!("{p}.mlp.down_proj.weight")),
                }
            })
            .collect();
        let norm = take(&mut t, "model.norm.weight").1;
        let lm_head = (!cfg.tie_embeddings).then(|| linear(&mut t, "lm_head.weight"));
        Self {
            backend,
            inv_freq: cfg.inv_freq(),
            cfg,
            embed,
            layers,
            norm,
            lm_head,
        }
    }

    /// Loads `config.json` and `model.safetensors` from a checkpoint directory.
    pub fn from_dir(dir: &Path, backend: B) -> Self {
        let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let tensors = load(&dir.join("model.safetensors"));
        Self::from_tensors(tensors, LlamaConfig::from_json(&json), backend)
    }

    /// Stores every weight matrix as int8 with one scale per row, the token
    /// embedding table (also the output projection when tied) included; every
    /// matmul then rounds its activations to int8 as well. Norm weights and
    /// biases stay f32: they are tiny and every output depends on them.
    pub fn quantize(&mut self) {
        let layers = self.layers.iter_mut().flat_map(Layer::matrices_mut);
        for m in layers.chain([&mut self.embed]).chain(self.lm_head.as_mut()) {
            m.quantize();
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    fn output_projection(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    /// Runs `ids` as positions `cache.len()..` and returns the final-norm
    /// output `[ids.len(), hidden]`. Appends the new keys and values to `cache`.
    fn run(&self, ids: &[u32], cache: &mut KvCache, mut trace: Option<&mut Trace>) -> Vec<f32> {
        let (b, c) = (&self.backend, &self.cfg);
        let (t, pos0) = (ids.len(), cache.len());
        assert!(
            pos0 + t <= c.max_positions,
            "context is {} tokens",
            c.max_positions
        );
        let shape = AttnShape {
            t,
            pos0,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            head_dim: c.head_dim,
        };
        let (q_dim, kv_dim) = (c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);

        let mut x = Vec::with_capacity(t * c.hidden);
        for &id in ids {
            self.embed.push_row(id as usize, &mut x);
        }
        note(&mut trace, "embed", &x);

        let last_layer = self.layers.len() - 1;
        for (l, layer) in self.layers.iter().enumerate() {
            let h = b.rmsnorm(&x, &layer.attn_norm, c.rms_eps);
            note(&mut trace, format!("block{l}.ln_1"), &h);

            let mut qkv = b.matmul(&h, t, &layer.qkv, layer.qkv_bias.as_deref());
            // On the last layer of a non-traced prefill, every token's keys and
            // values must still populate the cache, but only the last token's
            // hidden state is needed for logits.
            let prune_last = l == last_layer && t > 1 && trace.is_none();
            let (t_out, a) = if t == 1 {
                // `qkv` is already `[q | k | v]` contiguously: rotate in place
                // without allocating three separate vectors.
                let (q, kv) = qkv.split_at_mut(q_dim);
                let (k, v) = kv.split_at_mut(kv_dim);
                b.rope(q, pos0, c.n_heads, c.head_dim, &self.inv_freq);
                b.rope(k, pos0, c.n_kv_heads, c.head_dim, &self.inv_freq);
                cache.append(l, k, v);
                let (all_k, all_v) = &cache.layers[l];
                (1, b.attention(q, all_k, all_v, shape))
            } else {
                let [mut q, mut k, v]: [Vec<f32>; 3] = split_columns(&qkv, &[q_dim, kv_dim, kv_dim])
                    .try_into()
                    .unwrap();
                b.rope(&mut k, pos0, c.n_kv_heads, c.head_dim, &self.inv_freq);
                cache.append(l, &k, &v);
                let (all_k, all_v) = &cache.layers[l];
                if prune_last {
                    let q_last = &mut q[(t - 1) * q_dim..];
                    b.rope(q_last, pos0 + t - 1, c.n_heads, c.head_dim, &self.inv_freq);
                    let last_shape = AttnShape {
                        t: 1,
                        pos0: pos0 + t - 1,
                        ..shape
                    };
                    (1, b.attention(q_last, all_k, all_v, last_shape))
                } else {
                    b.rope(&mut q, pos0, c.n_heads, c.head_dim, &self.inv_freq);
                    (t, b.attention(&q, all_k, all_v, shape))
                }
            };
            if prune_last {
                x.drain(..(t - 1) * c.hidden);
            }
            let a = b.matmul(&a, t_out, &layer.o_proj, None);
            note(&mut trace, format!("block{l}.attn"), &a);
            add_in_place(&mut x, &a);

            let h = b.rmsnorm(&x, &layer.mlp_norm, c.rms_eps);
            note(&mut trace, format!("block{l}.ln_2"), &h);
            let m = b.matmul(&h, t_out, &layer.gate_up, None);
            let m = b.swiglu(&m, c.intermediate);
            let m = b.matmul(&m, t_out, &layer.down, None);
            note(&mut trace, format!("block{l}.mlp"), &m);
            add_in_place(&mut x, &m);
            note(&mut trace, format!("block.{l}"), &x);
        }
        cache.advance(t);

        let x = b.rmsnorm(&x, &self.norm, c.rms_eps);
        note(&mut trace, "ln_f", &x);
        x
    }
}

impl<B: Backend> Model for Llama<B> {
    fn new_cache(&self) -> KvCache {
        KvCache::new(self.cfg.n_layers, self.cfg.n_kv_heads * self.cfg.head_dim)
    }

    fn forward(&self, ids: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let x = self.run(ids, cache, None);
        let w = self.cfg.hidden;
        self.backend
            .matmul(&x[x.len() - w..], 1, self.output_projection(), None)
    }

    fn weight_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .flat_map(Layer::matrices)
            .map(Matrix::bytes)
            .sum();
        layers + self.output_projection().bytes()
    }

    fn forward_trace(&self, ids: &[u32]) -> Trace {
        let mut trace = Trace::new();
        let x = self.run(ids, &mut self.new_cache(), Some(&mut trace));
        let logits = self
            .backend
            .matmul(&x, ids.len(), self.output_projection(), None);
        trace.insert("logits".into(), logits);
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close_to_scale;
    use crate::cpu::CpuBackend;
    use crate::ops::argmax;
    use crate::reference::{self, Fixture};
    use std::sync::OnceLock;

    const QWEN: &str = "qwen2.5-coder-0.5b";

    fn load_qwen() -> Option<Fixture<Llama<CpuBackend>>> {
        Fixture::load(QWEN, |dir| Llama::from_dir(dir, CpuBackend::default()))
    }

    // Loading Qwen's weights as f32 takes about 2 GB, so the tests that only
    // read the model share one copy instead of each loading its own.
    fn qwen() -> Option<&'static Fixture<Llama<CpuBackend>>> {
        static MODEL: OnceLock<Option<Fixture<Llama<CpuBackend>>>> = OnceLock::new();
        MODEL.get_or_init(load_qwen).as_ref()
    }

    #[test]
    fn qwen_every_layer_matches_hugging_face() {
        if let Some(f) = qwen() {
            reference::check_layers(f, 24, 1e-4);
        }
    }

    #[test]
    fn qwen_greedy_matches_hugging_face_generate() {
        if let Some(f) = qwen() {
            reference::check_greedy(f, &f.model.config().eos);
        }
    }

    #[test]
    fn qwen_cached_decode_equals_recompute() {
        if let Some(f) = qwen() {
            reference::check_cached_decode(f, 1e-4);
        }
    }

    // Quantizing is lossy, so the bar is far looser than 1e-4: the next-token
    // logits of the chat prompt must stay close to Hugging Face's float32 ones
    // and pick the same token.
    #[test]
    fn qwen_int8_logits_stay_close_to_hugging_face() {
        let Some(mut f) = load_qwen() else { return };
        f.model.quantize();
        let ids = f.ids("ids");
        let got = f.model.forward(&ids, &mut f.model.new_cache());
        let all = &f.reference["logits"].1;
        let want = &all[all.len() - got.len()..];
        assert_eq!(argmax(&got), argmax(want));
        assert_close_to_scale(&got, want, 5e-2);
    }

    const TINY: &str = "tiny-llama3";

    fn tiny(rope_scaling: bool) -> Option<Fixture<Llama<CpuBackend>>> {
        Fixture::load(TINY, |dir| {
            let mut model = Llama::from_dir(dir, CpuBackend::new(2));
            if !rope_scaling {
                model.cfg.rope_scaling = None;
                model.inv_freq = model.cfg.inv_freq();
            }
            model
        })
    }

    // A random Llama 3 style model: untied output projection, no biases, and
    // rope scaling that changes the frequencies of every head dimension pair but
    // the first, with a prompt that runs past the original context.
    #[test]
    fn tiny_llama3_matches_hugging_face() {
        if let Some(f) = tiny(true) {
            reference::check_layers(&f, 2, 1e-4);
            reference::check_greedy(&f, &f.model.config().eos);
            reference::check_cached_decode(&f, 1e-4);
        }
    }

    // Without the scaling the same check must fail, or it proves nothing.
    #[test]
    fn rope_scaling_is_what_the_tiny_test_checks() {
        if let Some(f) = tiny(false) {
            let check = std::panic::AssertUnwindSafe(|| reference::check_layers(&f, 2, 1e-4));
            assert!(std::panic::catch_unwind(check).is_err());
        }
    }
}
