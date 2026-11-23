use crate::backend::{AttnShape, Backend, Matrix};
use crate::kv_cache::KvCache;
use crate::ops::{add_in_place, argmax, embed};
use std::collections::HashMap;

// GPT-2 small. The checkpoint stores no hyperparameters; the layer count,
// width, context and vocabulary come from tensor shapes, these two do not.
const N_HEAD: usize = 12;
const EPS: f32 = 1e-5;

/// Named intermediate activations, filled when a forward pass is traced.
pub type Trace = HashMap<String, Vec<f32>>;

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

type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn take(tensors: &mut Tensors, name: &str) -> (Vec<usize>, Vec<f32>) {
    tensors
        .remove(name)
        .unwrap_or_else(|| panic!("checkpoint has no tensor {name}"))
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

    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.blocks.len(), self.wte.cols)
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
        let mut note = |name: String, x: &[f32]| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.insert(name, x.to_vec());
            }
        };

        // The new tokens sit at positions pos0.., not 0..: skip that many rows of wpe.
        let mut x = embed(&self.wte.data, &self.wpe[pos0 * w..], ids, w);
        note("embed".into(), &x);

        for (l, blk) in self.blocks.iter().enumerate() {
            let h = b.layernorm(&x, &blk.ln_1.0, &blk.ln_1.1, EPS);
            note(format!("block{l}.ln_1"), &h);

            // c_attn fuses query, key, value: each row is [q | k | v], w wide each.
            let qkv = b.matmul(&h, t, &blk.c_attn, Some(&blk.c_attn_bias));
            let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
            for row in qkv.chunks_exact(3 * w) {
                q.extend_from_slice(&row[..w]);
                k.extend_from_slice(&row[w..2 * w]);
                v.extend_from_slice(&row[2 * w..]);
            }
            cache.append(l, &k, &v);
            let (all_k, all_v) = &cache.layers[l];
            let a = b.attention(&q, all_k, all_v, shape);
            let a = b.matmul(&a, t, &blk.attn_proj, Some(&blk.attn_proj_bias));
            note(format!("block{l}.attn"), &a);
            add_in_place(&mut x, &a);

            let h = b.layernorm(&x, &blk.ln_2.0, &blk.ln_2.1, EPS);
            note(format!("block{l}.ln_2"), &h);
            let mut m = b.matmul(&h, t, &blk.c_fc, Some(&blk.c_fc_bias));
            b.gelu(&mut m);
            let m = b.matmul(&m, t, &blk.mlp_proj, Some(&blk.mlp_proj_bias));
            note(format!("block{l}.mlp"), &m);
            add_in_place(&mut x, &m);
            note(format!("block.{l}"), &x);
        }
        cache.advance(t);

        let x = b.layernorm(&x, &self.ln_f.0, &self.ln_f.1, EPS);
        note("ln_f".into(), &x);
        x
    }

    /// Logits `[vocab]` for the token after the last of `ids`.
    pub fn forward(&self, ids: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let x = self.run(ids, cache, None);
        let w = self.wte.cols;
        // Output projection is tied to the embedding table: logits = x @ wte^T.
        self.backend.matmul(&x[x.len() - w..], 1, &self.wte, None)
    }

    /// Like `forward` from an empty cache, but returns every intermediate
    /// activation, and logits for every position under "logits".
    pub fn forward_trace(&self, ids: &[u32]) -> Trace {
        let mut trace = Trace::new();
        let x = self.run(ids, &mut self.new_cache(), Some(&mut trace));
        let logits = self.backend.matmul(&x, ids.len(), &self.wte, None);
        trace.insert("logits".into(), logits);
        trace
    }

    /// Greedy decoding: the prompt followed by up to `max_new` argmax tokens,
    /// stopping after `eos` if it is produced.
    pub fn generate_greedy(&self, prompt: &[u32], max_new: usize, eos: Option<u32>) -> Vec<u32> {
        let mut cache = self.new_cache();
        let mut out = prompt.to_vec();
        let mut logits = self.forward(prompt, &mut cache);
        for step in 0..max_new {
            let next = argmax(&logits);
            out.push(next);
            if Some(next) == eos || step + 1 == max_new {
                break;
            }
            logits = self.forward(&[next], &mut cache);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close_to_scale;
    use crate::cpu::CpuBackend;
    use crate::safetensors::load;
    use std::path::Path;

    const EOS: u32 = 50256;

    // Both files are gitignored: tests skip on a fresh clone.
    fn fixtures() -> Option<(Gpt2<CpuBackend>, Tensors)> {
        let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/gpt2/"));
        let (model, reference) = (
            dir.join("model.safetensors"),
            dir.join("reference.safetensors"),
        );
        if !model.exists() || !reference.exists() {
            return None;
        }
        Some((
            Gpt2::from_tensors(load(&model), CpuBackend),
            load(&reference),
        ))
    }

    fn ids_of(reference: &Tensors, key: &str) -> Vec<u32> {
        reference[key].1.iter().map(|&x| x as u32).collect()
    }

    // The done-when bar: every layer boundary against Hugging Face for the same ids.
    #[test]
    fn every_layer_matches_hugging_face() {
        let Some((model, reference)) = fixtures() else {
            return;
        };
        let ids = ids_of(&reference, "ids");
        let trace = model.forward_trace(&ids);
        let mut compared = 0;
        for (name, (_, want)) in &reference {
            if name == "ids" || name == "greedy_ids" {
                continue;
            }
            let got = trace
                .get(name)
                .unwrap_or_else(|| panic!("no trace for {name}"));
            assert_close_to_scale(got, want, 1e-4);
            compared += 1;
        }
        // embed, 4 block-0 sub-layers, 12 blocks, ln_f, logits
        assert_eq!(compared, 1 + 4 + 12 + 1 + 1);
    }

    #[test]
    fn greedy_matches_hugging_face_generate() {
        let Some((model, reference)) = fixtures() else {
            return;
        };
        let prompt = ids_of(&reference, "ids");
        let want = ids_of(&reference, "greedy_ids");
        let got = model.generate_greedy(&prompt, want.len() - prompt.len(), Some(EOS));
        assert_eq!(got, want);
    }

    // The cache is an optimization only: feeding tokens one at a time must give
    // the logits a from-scratch pass over the same prefix gives.
    #[test]
    fn cached_decode_equals_recompute() {
        let Some((model, reference)) = fixtures() else {
            return;
        };
        let ids = ids_of(&reference, "greedy_ids");
        let split = ids_of(&reference, "ids").len();
        let mut cache = model.new_cache();
        let mut cached = vec![model.forward(&ids[..split], &mut cache)];
        for i in split..ids.len() {
            cached.push(model.forward(&ids[i..=i], &mut cache));
        }
        assert_eq!(cache.len(), ids.len());
        for (step, got) in cached.iter().enumerate() {
            let want = model.forward(&ids[..split + step], &mut model.new_cache());
            assert_close_to_scale(got, &want, 1e-5);
        }
    }
}
