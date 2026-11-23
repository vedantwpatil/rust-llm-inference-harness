//! Checks shared by every model's tests: its forward pass against Hugging Face's
//! float32 one on the same ids. `scripts/gen_reference.py` writes the
//! reference file; both it and the model weights are gitignored, so a fixture
//! that cannot be loaded makes the test skip.

use crate::assert_close_to_scale;
use crate::model::Model;
use crate::safetensors::{Tensors, load};
use std::path::{Path, PathBuf};

pub struct Fixture<M> {
    pub model: M,
    pub reference: Tensors,
}

pub fn data_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name)
}

impl<M> Fixture<M> {
    /// Loads `data/<name>/model.safetensors` into a model and
    /// `data/<name>/reference.safetensors` beside it.
    pub fn load(name: &str, build: impl FnOnce(Tensors) -> M) -> Option<Self> {
        let dir = data_dir(name);
        let (model, reference) = (
            dir.join("model.safetensors"),
            dir.join("reference.safetensors"),
        );
        if !model.exists() || !reference.exists() {
            return None;
        }
        Some(Self {
            model: build(load(&model)),
            reference: load(&reference),
        })
    }

    pub fn ids(&self, key: &str) -> Vec<u32> {
        self.reference[key].1.iter().map(|&x| x as u32).collect()
    }
}

/// Every layer boundary against Hugging Face for the same ids, each within
/// `rel` of the tensor's own scale.
pub fn check_layers<M: Model>(f: &Fixture<M>, n_layers: usize, rel: f32) {
    let trace = f.model.forward_trace(&f.ids("ids"));
    let mut compared = 0;
    for (name, (_, want)) in &f.reference {
        if name == "ids" || name == "greedy_ids" {
            continue;
        }
        let got = trace
            .get(name)
            .unwrap_or_else(|| panic!("no trace for {name}"));
        assert_close_to_scale(got, want, rel);
        compared += 1;
    }
    // embed, 4 sub-layers of block 0, every block, final norm, logits
    assert_eq!(compared, 1 + 4 + n_layers + 1 + 1);
}

/// Greedy decoding gives the ids Hugging Face's `generate` gives, sampling off.
pub fn check_greedy<M: Model>(f: &Fixture<M>, eos: &[u32]) {
    let (prompt, want) = (f.ids("ids"), f.ids("greedy_ids"));
    let got = f
        .model
        .generate_greedy(&prompt, want.len() - prompt.len(), eos);
    assert_eq!(got, want);
}

// The cache is an optimization only: feeding tokens one at a time must give
// the logits a from-scratch pass over the same prefix gives.
pub fn check_cached_decode<M: Model>(f: &Fixture<M>, rel: f32) {
    let ids = f.ids("greedy_ids");
    let split = f.ids("ids").len();
    let mut cache = f.model.new_cache();
    let mut cached = vec![f.model.forward(&ids[..split], &mut cache)];
    for i in split..ids.len() {
        cached.push(f.model.forward(&ids[i..=i], &mut cache));
    }
    assert_eq!(cache.len(), ids.len());
    for (step, got) in cached.iter().enumerate() {
        let want = f
            .model
            .forward(&ids[..split + step], &mut f.model.new_cache());
        assert_close_to_scale(got, &want, rel);
    }
}
