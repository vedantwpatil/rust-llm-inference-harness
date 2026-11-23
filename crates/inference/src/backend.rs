use crate::kernel::{Q8_BLOCK, packed_index, q8_scale, quantize_into};

/// A weight matrix stored `[rows = outputs, cols = inputs]`, row-major. Output
/// `o` of `x @ W^T` is the dot product of `x` with the contiguous run of row
/// `o`, which is what a matrix-vector kernel wants to stream. Hugging Face's
/// GPT-2 `Conv1D` weights are `[in, out]` and are transposed once at load.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub weights: Weights,
}

/// How a matrix's values are stored.
pub enum Weights {
    F32(Vec<f32>),
    /// Int8: row `r` is its values (at `Matrix::q8_index`) times `scales[r]`.
    /// Each row has its own scale, so one large value only costs the precision
    /// of its own row. A matmul against these rounds its activations to int8
    /// too, one scale per `Q8_BLOCK` values.
    Q8 {
        values: Vec<i8>,
        scales: Vec<f32>,
    },
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            rows,
            cols,
            weights: Weights::F32(data),
        }
    }

    /// The values of an unquantized matrix.
    pub fn f32(&self) -> &[f32] {
        match &self.weights {
            Weights::F32(data) => data,
            Weights::Q8 { .. } => panic!("matrix is quantized"),
        }
    }

    /// Where `Weights::Q8` keeps element `(row, col)`. Each group of four rows is
    /// stored block by block (see `packed_index`) so a matmul streams them as
    /// one run; the rows after the last whole group stay plain.
    pub fn q8_index(&self, row: usize, col: usize) -> usize {
        let whole = self.rows / 4 * 4;
        let (first, group) = if row < whole {
            (row / 4 * 4, 4)
        } else {
            (row, 1)
        };
        first * self.cols + packed_index(group, row - first, col, self.cols)
    }

    /// Replaces the values with int8 and one scale per row: the largest
    /// magnitude in a row maps to 127, so no value is clipped and each is off by
    /// at most half a step. Does nothing to a matrix that is already int8.
    pub fn quantize(&mut self) {
        let Weights::F32(data) = &self.weights else {
            return;
        };
        let mut values = vec![0i8; data.len()];
        let mut scales = Vec::with_capacity(self.rows);
        for (r, row) in data.chunks_exact(self.cols).enumerate() {
            let scale = q8_scale(row);
            for (b, block) in row.chunks(Q8_BLOCK).enumerate() {
                let at = self.q8_index(r, b * Q8_BLOCK);
                quantize_into(block, scale, &mut values[at..][..block.len()]);
            }
            scales.push(scale);
        }
        self.weights = Weights::Q8 { values, scales };
    }

    /// Appends row `i` to `out` as f32: an embedding lookup.
    pub fn push_row(&self, i: usize, out: &mut Vec<f32>) {
        let (start, end) = (i * self.cols, (i + 1) * self.cols);
        match &self.weights {
            Weights::F32(data) => out.extend_from_slice(&data[start..end]),
            Weights::Q8 { values, scales } => {
                out.extend((0..self.cols).map(|c| values[self.q8_index(i, c)] as f32 * scales[i]))
            }
        }
    }

    /// Bytes of memory the values take: what a matmul streams from memory.
    pub fn bytes(&self) -> usize {
        match &self.weights {
            Weights::F32(data) => data.len() * 4,
            Weights::Q8 { values, scales } => values.len() + scales.len() * 4,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::data;

    #[test]
    fn quantize_is_within_half_a_step_and_exact_at_the_row_max() {
        let (rows, cols) = (5, 70);
        let mut raw = data(rows * cols, 7);
        raw[cols..2 * cols].fill(0.0); // an all-zero row
        let mut m = Matrix::new(rows, cols, raw.clone());
        let before = m.bytes();
        m.quantize();
        let Weights::Q8 { values, scales } = &m.weights else {
            panic!("not quantized")
        };
        for r in 0..rows {
            let row = &raw[r * cols..(r + 1) * cols];
            let max = row.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            for (c, &v) in row.iter().enumerate() {
                let back = values[m.q8_index(r, c)] as f32 * scales[r];
                assert!(
                    (back - v).abs() <= scales[r] / 2.0 * 1.0001,
                    "row {r} col {c}"
                );
            }
            assert!((0..cols).all(|c| values[m.q8_index(r, c)] != -128));
            assert!(max == 0.0 || (scales[r] * 127.0 - max).abs() <= max * 1e-6);
        }
        assert_eq!(m.bytes(), rows * cols + rows * 4);
        assert!(m.bytes() * 3 < before);
        // an embedding lookup gives the dequantized row
        let mut out = Vec::new();
        m.push_row(3, &mut out);
        let want = &raw[3 * cols..4 * cols];
        assert!(
            out.iter()
                .zip(want)
                .all(|(a, b)| (a - b).abs() <= scales[3] / 2.0 * 1.0001)
        );
    }
}
