use crate::backend::{AttnShape, Backend, Matrix};
use crate::config::LlamaConfig;
use crate::kv_cache::KvCache;
use crate::model::{Model, Trace, note};
use crate::ops::{add_in_place, split_columns};
use crate::safetensors::{Tensors, take};

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

        for (l, layer) in self.layers.iter().enumerate() {
            let h = b.rmsnorm(&x, &layer.attn_norm, c.rms_eps);
            note(&mut trace, format!("block{l}.ln_1"), &h);

            let qkv = b.matmul(&h, t, &layer.qkv, layer.qkv_bias.as_deref());
            let [mut q, mut k, v]: [Vec<f32>; 3] = split_columns(&qkv, &[q_dim, kv_dim, kv_dim])
                .try_into()
                .unwrap();
            b.rope(&mut q, pos0, c.n_heads, c.head_dim, &self.inv_freq);
            b.rope(&mut k, pos0, c.n_kv_heads, c.head_dim, &self.inv_freq);
            cache.append(l, &k, &v);
            let (all_k, all_v) = &cache.layers[l];
            let a = b.attention(&q, all_k, all_v, shape);
            let a = b.matmul(&a, t, &layer.o_proj, None);
            note(&mut trace, format!("block{l}.attn"), &a);
            add_in_place(&mut x, &a);

            let h = b.rmsnorm(&x, &layer.mlp_norm, c.rms_eps);
            note(&mut trace, format!("block{l}.ln_2"), &h);
            let m = b.matmul(&h, t, &layer.gate_up, None);
            let m = b.swiglu(&m, c.intermediate);
            let m = b.matmul(&m, t, &layer.down, None);
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
    use crate::cpu::CpuBackend;
    use crate::reference::{self, Fixture, data_dir};
    use std::sync::OnceLock;

    fn config(name: &str) -> LlamaConfig {
        LlamaConfig::from_json(
            &std::fs::read_to_string(data_dir(name).join("config.json")).unwrap(),
        )
    }

    // Loading Qwen's weights as f32 takes about 2 GB, so the tests of one
    // checkpoint share one copy instead of each loading its own.
    fn qwen() -> Option<&'static Fixture<Llama<CpuBackend>>> {
        static QWEN: OnceLock<Option<Fixture<Llama<CpuBackend>>>> = OnceLock::new();
        QWEN.get_or_init(|| {
            let name = "qwen2.5-coder-0.5b";
            Fixture::load(name, |t| {
                Llama::from_tensors(t, config(name), CpuBackend::default())
            })
        })
        .as_ref()
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

    const TINY: &str = "tiny-llama3";

    fn tiny(cfg: LlamaConfig) -> Option<Fixture<Llama<CpuBackend>>> {
        Fixture::load(TINY, |t| Llama::from_tensors(t, cfg, CpuBackend::new(2)))
    }

    // A random Llama 3 style model: untied output projection, no biases, and
    // rope scaling that changes the frequencies of every head dimension pair but
    // the first, with a prompt that runs past the original context.
    #[test]
    fn tiny_llama3_matches_hugging_face() {
        if let Some(f) = tiny(config(TINY)) {
            reference::check_layers(&f, 2, 1e-4);
            reference::check_greedy(&f, &f.model.config().eos);
            reference::check_cached_decode(&f, 1e-4);
        }
    }

    // Without the scaling the same test must fail, or it proves nothing.
    #[test]
    fn rope_scaling_is_what_the_tiny_test_checks() {
        let cfg = LlamaConfig {
            rope_scaling: None,
            ..config(TINY)
        };
        if let Some(f) = tiny(cfg) {
            let check = std::panic::AssertUnwindSafe(|| reference::check_layers(&f, 2, 1e-4));
            assert!(std::panic::catch_unwind(check).is_err());
        }
    }
}
