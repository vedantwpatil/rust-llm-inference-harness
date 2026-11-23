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

    /// Bytes of weights one decoded token streams from memory: every matrix
    /// once, with the token embedding table counted as the output projection
    /// (the lookup reads one row). Decode speed is at most memory bandwidth
    /// divided by this, plus the cache.
    fn weight_bytes(&self) -> usize;

    /// Runs `ids` as the positions after `cache.len()`, appends their keys and
    /// values to `cache`, and returns the logits `[vocab]` for the token after
    /// the last of `ids`.
    fn forward(&self, ids: &[u32], cache: &mut KvCache) -> Vec<f32>;

    /// Like `forward` from an empty cache, but returns every intermediate
    /// activation under the names `reference.rs` and the reference script
    /// share, and logits for every position under "logits".
    fn forward_trace(&self, ids: &[u32]) -> Trace;

    /// Greedy decoding: the prompt followed by up to `max_new` argmax tokens,
    /// stopping after one that is in `eos`.
    fn generate_greedy(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Vec<u32> {
        let mut out = prompt.to_vec();
        self.generate_each(prompt, max_new, eos, &mut |token| out.push(token));
        out
    }

    /// The same loop, handing each new token to `on_token` as it is chosen
    /// (the stop token included), so a caller can stream the reply.
    fn generate_each(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
        on_token: &mut dyn FnMut(u32),
    ) {
        assert!(!prompt.is_empty(), "the prompt has no tokens");
        let mut cache = self.new_cache();
        let mut logits = self.forward(prompt, &mut cache);
        for step in 0..max_new {
            let next = argmax(&logits);
            on_token(next);
            if eos.contains(&next) || step + 1 == max_new {
                break;
            }
            logits = self.forward(&[next], &mut cache);
        }
    }
}
