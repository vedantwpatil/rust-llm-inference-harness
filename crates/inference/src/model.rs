use crate::kv_cache::KvCache;
use crate::ops::argmax;
use std::collections::HashMap;

/// Named intermediate activations, filled when a forward pass is traced.
pub type Trace = HashMap<String, Vec<f32>>;

/// Records `x` under `name` when a trace was asked for.
pub(crate) fn note(trace: &mut Option<&mut Trace>, name: impl Into<String>, x: &[f32]) {
    if let Some(trace) = trace.as_deref_mut() {
        trace.insert(name.into(), x.to_vec());
    }
}

/// A language model the generation loop can drive.
pub trait Model {
    fn new_cache(&self) -> KvCache;

    /// Runs `ids` as the positions after `cache.len()`, appends their keys and
    /// values to `cache`, and returns the logits `[vocab]` for the token after
    /// the last of `ids`.
    fn forward(&self, ids: &[u32], cache: &mut KvCache) -> Vec<f32>;

    /// Greedy decoding: the prompt followed by up to `max_new` argmax tokens,
    /// stopping after one that is in `eos`.
    fn generate_greedy(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Vec<u32> {
        let mut cache = self.new_cache();
        let mut out = prompt.to_vec();
        let mut logits = self.forward(prompt, &mut cache);
        for step in 0..max_new {
            let next = argmax(&logits);
            out.push(next);
            if eos.contains(&next) || step + 1 == max_new {
                break;
            }
            logits = self.forward(&[next], &mut cache);
        }
        out
    }
}
