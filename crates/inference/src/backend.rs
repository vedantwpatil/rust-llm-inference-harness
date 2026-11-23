/// A weight matrix stored `[rows = outputs, cols = inputs]`, row-major. Output
/// `o` of `x @ W^T` is the dot product of `x` with the contiguous run
/// `data[o * cols..(o + 1) * cols]`, which is what a matrix-vector kernel
/// wants to stream. Hugging Face's GPT-2 `Conv1D` weights are `[in, out]` and
/// are transposed once at load.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    /// Appends row `i` to `out`: an embedding lookup.
    pub fn push_row(&self, i: usize, out: &mut Vec<f32>) {
        out.extend_from_slice(&self.data[i * self.cols..(i + 1) * self.cols]);
    }
}

/// Shape of one attention call. `t` new tokens (queries) sit at absolute
/// positions `pos0..pos0 + t`; the key/value arrays hold `pos0 + t` positions.
/// Prefill is `pos0 = 0, t = prompt length`, decode is `t = 1`.
#[derive(Clone, Copy)]
pub struct AttnShape {
    pub t: usize,
    pub pos0: usize,
    pub n_heads: usize,
    /// Equal to `n_heads` for multi-head attention, smaller for grouped-query
    /// attention: each key/value head serves `n_heads / n_kv_heads` query heads.
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

/// The tensor ops a model needs. Model code is generic over this trait, so a
/// faster implementation (stage 6: Metal) swaps in with static dispatch and is
/// checked op by op against `CpuBackend`, the correctness reference.
pub trait Backend {
    /// `x` is `[t, w.cols]`, result is `[t, w.rows]`: `x @ W^T + bias`.
    fn matmul(&self, x: &[f32], t: usize, w: &Matrix, bias: Option<&[f32]>) -> Vec<f32>;

    /// Normalizes each `width`-long row of `x` to zero mean and unit variance,
    /// then scales by `gamma` and shifts by `beta`.
    fn layernorm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32>;

    /// Scales each `gamma.len()`-long row of `x` by `1 / sqrt(mean(x^2) + eps)`,
    /// then by `gamma`: no mean subtraction, no bias.
    fn rmsnorm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32>;

    /// GPT-2's tanh approximation of GELU, in place.
    fn gelu(&self, x: &mut [f32]);

    /// SwiGLU gate. `gate_up` holds rows of `[gate | up]`, each `2 * width`
    /// values; returns rows of `silu(gate) * up`, each `width` values.
    fn swiglu(&self, gate_up: &[f32], width: usize) -> Vec<f32>;

    /// Rotary position embedding, in place. `x` is `[t, n_heads * head_dim]`
    /// and row `i` sits at absolute position `pos0 + i`. In every head, element
    /// `j` pairs with element `j + head_dim / 2` ("rotate half", the layout
    /// Hugging Face's Llama weights assume) and the pair is rotated by the angle
    /// `position * inv_freq[j]`.
    fn rope(&self, x: &mut [f32], pos0: usize, n_heads: usize, head_dim: usize, inv_freq: &[f32]);

    /// Softmax over the whole slice, in place.
    fn softmax(&self, x: &mut [f32]);

    /// Causal attention. `q` is `[t, n_heads * head_dim]`, `k` and `v` are
    /// `[pos0 + t, n_kv_heads * head_dim]`. Query `i` attends to key positions
    /// `0..=pos0 + i`. Result is `[t, n_heads * head_dim]`, heads concatenated.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], shape: AttnShape) -> Vec<f32>;
}
