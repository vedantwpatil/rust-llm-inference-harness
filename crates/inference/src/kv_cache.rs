/// Keys and values of every token seen so far, one pair of flat buffers per
/// layer, `[position][kv_dim]` row-major. Decode appends one row per layer per
/// token instead of recomputing the whole prefix.
pub struct KvCache {
    kv_dim: usize,
    pub(crate) layers: Vec<(Vec<f32>, Vec<f32>)>,
    len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize) -> Self {
        Self {
            kv_dim,
            layers: vec![(Vec::new(), Vec::new()); n_layers],
            len: 0,
        }
    }

    /// Tokens cached so far: the absolute position of the next token.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes added per token: keys and values, every layer, f32.
    pub fn bytes_per_token(&self) -> usize {
        2 * self.layers.len() * self.kv_dim * 4
    }

    pub(crate) fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), v.len());
        assert_eq!(k.len() % self.kv_dim, 0);
        self.layers[layer].0.extend_from_slice(k);
        self.layers[layer].1.extend_from_slice(v);
    }

    /// Called once after every layer has appended the same `t` rows.
    pub(crate) fn advance(&mut self, t: usize) {
        self.len += t;
        for (k, v) in &self.layers {
            debug_assert_eq!(k.len(), self.len * self.kv_dim);
            debug_assert_eq!(v.len(), self.len * self.kv_dim);
        }
    }
}
