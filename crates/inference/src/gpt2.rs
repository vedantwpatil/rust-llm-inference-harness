use crate::backend::{AttnShape, Backend, Matrix};
use crate::kv_cache::KvCache;
use crate::model::{Model, Trace, note};
use crate::ops::{add_in_place, embed, split_columns};
use crate::safetensors::{Tensors, take};

// GPT-2 small. The checkpoint stores no hyperparameters; the layer count,
// width, context and vocabulary come from tensor shapes, these two do not.
const N_HEAD: usize = 12;
const EPS: f32 = 1e-5;

struct Block {
    ln_1: (Vec<f32>, Vec<f32>),
    c_attn: Matrix,
    c_attn_bias: Vec<f32>,
    attn_proj: Matrix,
    attn_proj_bias: Vec<f32>,
    ln_2: (Vec<f32>, Vec<f32>),
    c_fc: Matrix,
    c_fc_bias: Vec<f32>,
    mlp_proj: Matrix,
    mlp_proj_bias: Vec<f32>,
}

pub struct Gpt2<B: Backend> {
    backend: B,
    n_ctx: usize,
    wte: Matrix,
    wpe: Vec<f32>,
    blocks: Vec<Block>,
    ln_f: (Vec<f32>, Vec<f32>),
}

/// Hugging Face `Conv1D` weights are `[in, out]`; `Matrix` wants `[out, in]`.
fn conv1d(tensors: &mut Tensors, name: &str) -> Matrix {
    let (shape, data) = take(tensors, name);
    let (n_in, n_out) = (shape[0], shape[1]);
    let mut t = vec![0.0; data.len()];
    for i in 0..n_in {
        for o in 0..n_out {
            t[o * n_in + i] = data[i * n_out + o];
        }
    }
    Matrix::new(n_out, n_in, t)
}

impl<B: Backend> Gpt2<B> {
    /// Builds the model from `safetensors::load`'s output, consuming it.
    pub fn from_tensors(mut t: Tensors, backend: B) -> Self {
        let (shape, wte) = take(&mut t, "wte.weight");
        let wte = Matrix::new(shape[0], shape[1], wte);
        let (_, wpe) = take(&mut t, "wpe.weight");
        let n_layer = (0..)
            .take_while(|i| t.contains_key(&format!("h.{i}.ln_1.weight")))
            .count();

        let blocks = (0..n_layer)
            .map(|i| {
                let p = format!("h.{i}");
                let mut vec = |name: &str| take(&mut t, &format!("{p}.{name}")).1;
                let ln_1 = (vec("ln_1.weight"), vec("ln_1.bias"));
                let c_attn_bias = vec("attn.c_attn.bias");
                let attn_proj_bias = vec("attn.c_proj.bias");
                let ln_2 = (vec("ln_2.weight"), vec("ln_2.bias"));
                let c_fc_bias = vec("mlp.c_fc.bias");
                let mlp_proj_bias = vec("mlp.c_proj.bias");
                Block {
                    ln_1,
                    c_attn: conv1d(&mut t, &format!("{p}.attn.c_attn.weight")),
                    c_attn_bias,
                    attn_proj: conv1d(&mut t, &format!("{p}.attn.c_proj.weight")),
                    attn_proj_bias,
                    ln_2,
                    c_fc: conv1d(&mut t, &format!("{p}.mlp.c_fc.weight")),
                    c_fc_bias,
                    mlp_proj: conv1d(&mut t, &format!("{p}.mlp.c_proj.weight")),
                    mlp_proj_bias,
                }
            })
            .collect();
        let ln_f = (take(&mut t, "ln_f.weight").1, take(&mut t, "ln_f.bias").1);

        Self {
            backend,
            n_ctx: wpe.len() / wte.cols,
            wte,
            wpe,
            blocks,
            ln_f,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Runs `ids` as positions `cache.len()..` and returns the final-layer-norm
    /// output `[ids.len(), width]`. Appends the new keys and values to `cache`.
    fn run(&self, ids: &[u32], cache: &mut KvCache, mut trace: Option<&mut Trace>) -> Vec<f32> {
        let b = &self.backend;
        let (t, w, pos0) = (ids.len(), self.wte.cols, cache.len());
        assert!(pos0 + t <= self.n_ctx, "context is {} tokens", self.n_ctx);
        let shape = AttnShape {
            t,
            pos0,
            n_heads: N_HEAD,
            n_kv_heads: N_HEAD,
            head_dim: w / N_HEAD,
        };
        // The new tokens sit at positions pos0.., not 0..: skip that many rows of wpe.
        let mut x = embed(self.wte.f32(), &self.wpe[pos0 * w..], ids, w);
        note(&mut trace, "embed", &x);

        for (l, blk) in self.blocks.iter().enumerate() {
            let h = b.layernorm(&x, &blk.ln_1.0, &blk.ln_1.1, EPS);
            note(&mut trace, format!("block{l}.ln_1"), &h);

            // c_attn fuses query, key, value: each row is [q | k | v], w wide each.
            let qkv = b.matmul(&h, t, &blk.c_attn, Some(&blk.c_attn_bias));
            let [q, k, v] = &split_columns(&qkv, &[w; 3])[..] else {
                unreachable!()
            };
            cache.append(l, k, v);
            let (all_k, all_v) = &cache.layers[l];
            let a = b.attention(q, all_k, all_v, shape);
            let a = b.matmul(&a, t, &blk.attn_proj, Some(&blk.attn_proj_bias));
            note(&mut trace, format!("block{l}.attn"), &a);
            add_in_place(&mut x, &a);

            let h = b.layernorm(&x, &blk.ln_2.0, &blk.ln_2.1, EPS);
            note(&mut trace, format!("block{l}.ln_2"), &h);
            let mut m = b.matmul(&h, t, &blk.c_fc, Some(&blk.c_fc_bias));
            b.gelu(&mut m);
            let m = b.matmul(&m, t, &blk.mlp_proj, Some(&blk.mlp_proj_bias));
            note(&mut trace, format!("block{l}.mlp"), &m);
            add_in_place(&mut x, &m);
            note(&mut trace, format!("block.{l}"), &x);
        }
        cache.advance(t);

        let x = b.layernorm(&x, &self.ln_f.0, &self.ln_f.1, EPS);
        note(&mut trace, "ln_f", &x);
        x
    }
}

impl<B: Backend> Model for Gpt2<B> {
    fn new_cache(&self) -> KvCache {
        KvCache::new(self.blocks.len(), self.wte.cols)
    }

    fn forward(&self, ids: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let x = self.run(ids, cache, None);
        let w = self.wte.cols;
        // Output projection is tied to the embedding table: logits = x @ wte^T.
        self.backend.matmul(&x[x.len() - w..], 1, &self.wte, None)
    }

    fn weight_bytes(&self) -> usize {
        let blocks: usize = self
            .blocks
            .iter()
            .flat_map(|b| [&b.c_attn, &b.attn_proj, &b.c_fc, &b.mlp_proj])
            .map(Matrix::bytes)
            .sum();
        blocks + self.wte.bytes()
    }

    fn forward_trace(&self, ids: &[u32]) -> Trace {
        let mut trace = Trace::new();
        let x = self.run(ids, &mut self.new_cache(), Some(&mut trace));
        let logits = self.backend.matmul(&x, ids.len(), &self.wte, None);
        trace.insert("logits".into(), logits);
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::reference::{self, Fixture};
    use crate::safetensors::load;

    const EOS: u32 = 50256;

    // Both files are gitignored: tests skip on a fresh clone.
    fn fixture() -> Option<Fixture<Gpt2<CpuBackend>>> {
        Fixture::load("gpt2", |dir| {
            let tensors = load(&dir.join("model.safetensors"));
            Gpt2::from_tensors(tensors, CpuBackend::default())
        })
    }

    #[test]
    fn every_layer_matches_hugging_face() {
        if let Some(f) = fixture() {
            reference::check_layers(&f, 12, 1e-4);
        }
    }

    #[test]
    fn greedy_matches_hugging_face_generate() {
        if let Some(f) = fixture() {
            reference::check_greedy(&f, &[EOS]);
        }
    }

    #[test]
    fn cached_decode_equals_recompute() {
        if let Some(f) = fixture() {
            reference::check_cached_decode(&f, 1e-5);
        }
    }
}
