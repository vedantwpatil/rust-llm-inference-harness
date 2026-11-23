use crate::backend::{AttnShape, Backend, Matrix, Weights};
use crate::kernel::{dot, dot_q8, tile};
use crate::pool::Pool;
use std::cell::RefCell;

thread_local! {
    // Widened int8 weight rows for the prefill path, one buffer per pool thread
    // so a task never allocates: a 16-row slab of the widest matrix is ~300 KB,
    // and a fresh allocation that size would page-fault on every call.
    static WIDENED: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Plain Rust on the CPU: NEON kernels, a spinning thread pool. The reference
/// every other backend is compared against.
pub struct CpuBackend {
    pool: Pool,
}

impl CpuBackend {
    pub fn new(threads: usize) -> Self {
        Self {
            pool: Pool::new(threads),
        }
    }
}

impl Default for CpuBackend {
    /// `INFERENCE_THREADS` if set, else one thread per core up to 8.
    // shortcut: the cap of 8 is measured on an M3 Max (10 performance + 4
    // efficiency cores), where all 14 threads decode slower than 8 because the
    // efficiency cores straggle; read the performance-core count when another
    // machine matters.
    fn default() -> Self {
        let threads = std::env::var("INFERENCE_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get().min(8)));
        Self::new(threads)
    }
}

pub(crate) fn softmax_in_place(x: &mut [f32]) {
    // Subtracting the max keeps exp() from overflowing; the result is the same.
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// `out[r * t + i] = w_row_r . x_row_i` for every row of `w` (`k` floats each)
/// and every one of the `t` rows of `x`. Rows go four at a time, tokens four at
/// a time when there are that many, so a loaded weight is reused across tokens.
fn matmul_block(w: &[f32], x: &[f32], k: usize, t: usize, out: &mut [f32]) {
    let rows = w.len() / k;
    let mut r = 0;
    while r + 4 <= rows {
        let w4 = &w[r * k..(r + 4) * k];
        let mut i = 0;
        while i + 4 <= t {
            let tl = tile::<4, 4, 1>(w4, &x[i * k..], k);
            for (a, row) in tl.iter().enumerate() {
                out[(r + a) * t + i..][..4].copy_from_slice(row);
            }
            i += 4;
        }
        for i in i..t {
            let tl = tile::<4, 1, 4>(w4, &x[i * k..], k);
            for (a, row) in tl.iter().enumerate() {
                out[(r + a) * t + i] = row[0];
            }
        }
        r += 4;
    }
    for r in r..rows {
        for i in 0..t {
            out[r * t + i] = dot(&w[r * k..(r + 1) * k], &x[i * k..(i + 1) * k]);
        }
    }
}

/// Rows `first..` of `w` (as many as `slab` has room for) against every token:
/// `slab[r * t + i]` is row `first + r` dotted with token `i`.
fn matmul_rows(w: &Matrix, first: usize, x: &[f32], t: usize, slab: &mut [f32]) {
    let (n_in, rows) = (w.cols, slab.len() / t);
    match &w.weights {
        Weights::F32(data) => matmul_block(&data[first * n_in..][..rows * n_in], x, n_in, t, slab),
        // One token: read each int8 row once, straight from memory.
        Weights::Q8 { values, scales } if t == 1 => {
            for (r, out) in slab.iter_mut().enumerate() {
                *out = dot_q8(&values[(first + r) * n_in..][..n_in], x) * scales[first + r];
            }
        }
        // Several tokens reuse each row, so widen it to f32 once and let the
        // f32 tiles do the arithmetic.
        Weights::Q8 { values, scales } => WIDENED.with_borrow_mut(|buf| {
            buf.clear();
            for r in first..first + rows {
                let row = &values[r * n_in..][..n_in];
                buf.extend(row.iter().map(|&q| q as f32 * scales[r]));
            }
            matmul_block(buf, x, n_in, t, slab);
        }),
    }
}

impl Backend for CpuBackend {
    fn matmul(&self, x: &[f32], t: usize, w: &Matrix, bias: Option<&[f32]>) -> Vec<f32> {
        let (n_out, n_in) = (w.rows, w.cols);
        assert_eq!(x.len(), t * n_in);
        if let Some(b) = bias {
            assert_eq!(b.len(), n_out);
        }
        // Each task owns ROWS whole weight rows and every token's dot product
        // with them, so a weight row is read from memory once however large t
        // is.
        const ROWS: usize = 16;

        // One token: `[n_out][1]` is already `[1][n_out]`, so the tasks write
        // the result and the bias directly. A second parallel region would do
        // nothing but cost a fork-join, and a decoded token has ~100 matmuls.
        if t == 1 {
            let mut out = vec![0.0; n_out];
            self.pool.for_each_chunk_mut(&mut out, ROWS, |c, slab| {
                matmul_rows(w, c * ROWS, x, 1, slab);
                if let Some(b) = bias {
                    for (v, b) in slab.iter_mut().zip(&b[c * ROWS..]) {
                        *v += b;
                    }
                }
            });
            return out;
        }

        // Several tokens: the result is `[n_out][t]`; transposed below.
        let mut by_out = vec![0.0; n_out * t];
        self.pool
            .for_each_chunk_mut(&mut by_out, ROWS * t, |c, slab| {
                matmul_rows(w, c * ROWS, x, t, slab)
            });

        // Transpose to `[t][n_out]` and add the bias. A task owns a few whole
        // token rows of `out`, so the writes stay disjoint.
        const TOKENS: usize = 4;
        let mut out = vec![0.0; t * n_out];
        self.pool
            .for_each_chunk_mut(&mut out, TOKENS * n_out, |c, rows| {
                let first = c * TOKENS;
                for o in 0..n_out {
                    let b = bias.map_or(0.0, |b| b[o]);
                    for (j, row) in rows.chunks_exact_mut(n_out).enumerate() {
                        row[o] = by_out[o * t + first + j] + b;
                    }
                }
            });
        out
    }

    fn layernorm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let width = gamma.len();
        assert_eq!(beta.len(), width);
        assert_eq!(x.len() % width, 0);
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            out.extend(
                row.iter()
                    .zip(gamma.iter().zip(beta))
                    .map(|(v, (g, b))| (v - mean) * inv_std * g + b),
            );
        }
        out
    }

    fn rmsnorm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let width = gamma.len();
        assert_eq!(x.len() % width, 0);
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(width) {
            let mean_sq = dot(row, row) / width as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            out.extend(row.iter().zip(gamma).map(|(v, g)| v * scale * g));
        }
        out
    }

    fn swiglu(&self, gate_up: &[f32], width: usize) -> Vec<f32> {
        assert_eq!(gate_up.len() % (2 * width), 0);
        let mut out = vec![0.0; gate_up.len() / 2];
        // Chunks over the flat output, not one per row: a decoded token is a
        // single row and would otherwise run on one thread.
        const CHUNK: usize = 512;
        self.pool.for_each_chunk_mut(&mut out, CHUNK, |c, chunk| {
            let (mut row, mut j) = ((c * CHUNK) / width, (c * CHUNK) % width);
            for v in chunk {
                let g = gate_up[row * 2 * width + j];
                *v = g / (1.0 + (-g).exp()) * gate_up[row * 2 * width + width + j];
                j += 1;
                if j == width {
                    (row, j) = (row + 1, 0);
                }
            }
        });
        out
    }

    fn rope(&self, x: &mut [f32], pos0: usize, n_heads: usize, head_dim: usize, inv_freq: &[f32]) {
        let half = head_dim / 2;
        assert_eq!(inv_freq.len(), half);
        assert_eq!(x.len() % (n_heads * head_dim), 0);
        for (i, row) in x.chunks_exact_mut(n_heads * head_dim).enumerate() {
            let pos = (pos0 + i) as f32;
            // The angles depend on the position and pair index only, so every head shares them.
            let angles: Vec<(f32, f32)> = inv_freq.iter().map(|f| (pos * f).sin_cos()).collect();
            for head in row.chunks_exact_mut(head_dim) {
                let (lo, hi) = head.split_at_mut(half);
                for (j, &(sin, cos)) in angles.iter().enumerate() {
                    let (a, b) = (lo[j], hi[j]);
                    lo[j] = a * cos - b * sin;
                    hi[j] = b * cos + a * sin;
                }
            }
        }
    }

    fn gelu(&self, x: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        // A scalar tanh per element: serial it was a third of prefill time. A
        // chunk this size is also more than one decoded token's 3072, so decode
        // stays a single inline task.
        self.pool.for_each_chunk_mut(x, 4096, |_, chunk| {
            for v in chunk {
                let inner = SQRT_2_OVER_PI * (*v + 0.044715 * *v * *v * *v);
                *v = 0.5 * *v * (1.0 + inner.tanh());
            }
        });
    }

    fn softmax(&self, x: &mut [f32]) {
        softmax_in_place(x);
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], shape: AttnShape) -> Vec<f32> {
        let AttnShape {
            t,
            pos0,
            n_heads,
            n_kv_heads,
            head_dim,
        } = shape;
        let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);
        assert_eq!(q.len(), t * q_dim);
        assert_eq!(k.len(), (pos0 + t) * kv_dim);
        assert_eq!(v.len(), k.len());
        assert_eq!(n_heads % n_kv_heads, 0);
        let group = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out = vec![0.0; t * q_dim];
        // One task per (query, head): chunk `idx` of `out` is that head's slot.
        self.pool.for_each_chunk_mut(&mut out, head_dim, |idx, o| {
            let (i, h) = (idx / n_heads, idx % n_heads);
            let kv_off = (h / group) * head_dim;
            let qv = &q[i * q_dim + h * head_dim..][..head_dim];
            // The causal mask: key j > pos0 + i is never scored, which is the
            // same as a -inf score before softmax.
            let n_keys = pos0 + i + 1;
            let mut scores: Vec<f32> = (0..n_keys)
                .map(|j| dot(qv, &k[j * kv_dim + kv_off..][..head_dim]) * scale)
                .collect();
            softmax_in_place(&mut scores);
            for (j, p) in scores.iter().enumerate() {
                let vj = &v[j * kv_dim + kv_off..][..head_dim];
                for (acc, x) in o.iter_mut().zip(vj) {
                    *acc = p.mul_add(*x, *acc);
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;
    use crate::test_data::data;

    fn cpu() -> CpuBackend {
        CpuBackend::new(3)
    }

    #[test]
    fn matmul_toy() {
        // x [2,3] @ W^T, W [2,3] -> [2,2]
        let x = [1., 2., 3., 4., 5., 6.];
        let w = Matrix::new(2, 3, vec![1., 0., 1., 0., 1., 0.]);
        let got = cpu().matmul(&x, 2, &w, Some(&[10., 20.]));
        assert_eq!(got, [14., 22., 20., 25.]);
    }

    #[test]
    fn matmul_matches_naive_loops() {
        // sizes hit: a k tail, a partial ROWS chunk, rows % 4 != 0, t % 4 != 0, t >= 4
        for (t, n_in, n_out) in [
            (1, 19, 37),
            (5, 19, 37),
            (3, 64, 16),
            (1, 5, 1),
            (9, 40, 18),
            (8, 768, 33),
        ] {
            let x = data(t * n_in, 1);
            let w = Matrix::new(n_out, n_in, data(n_out * n_in, 2));
            let bias = data(n_out, 3);
            let got = cpu().matmul(&x, t, &w, Some(&bias));
            let mut want = vec![0.0f32; t * n_out];
            for tt in 0..t {
                for o in 0..n_out {
                    let mut s = bias[o] as f64;
                    for i in 0..n_in {
                        s += x[tt * n_in + i] as f64 * w.f32()[o * n_in + i] as f64;
                    }
                    want[tt * n_out + o] = s as f32;
                }
            }
            assert_close(&got, &want, 1e-5, 1e-5);
        }
    }

    #[test]
    fn layernorm_toy() {
        // mean 2.5, variance 1.25, so (x - 2.5) / sqrt(1.25)
        let got = cpu().layernorm(&[1., 2., 3., 4.], &[1.; 4], &[0.; 4], 0.0);
        assert_close(
            &got,
            &[-1.3416408, -0.4472136, 0.4472136, 1.3416408],
            1e-6,
            0.0,
        );
        // gamma scales and beta shifts; each row is normalized on its own, so
        // [1, 3] and [10, 14] (different mean and spread) land on the same output
        let got = cpu().layernorm(&[1., 3., 10., 14.], &[2., 2.], &[1., 1.], 0.0);
        assert_close(&got, &[-1., 3., -1., 3.], 1e-6, 0.0);
    }

    #[test]
    fn int8_matmul_equals_f32_matmul_on_the_dequantized_weights() {
        let (n_out, n_in) = (37, 100);
        let mut q = Matrix::new(n_out, n_in, data(n_out * n_in, 4));
        q.quantize();
        let Weights::Q8 { values, scales } = &q.weights else {
            unreachable!()
        };
        let widened: Vec<f32> = (0..n_out * n_in)
            .map(|i| values[i] as f32 * scales[i / n_in])
            .collect();
        let widened = Matrix::new(n_out, n_in, widened);
        let bias = data(n_out, 6);
        // one token takes the direct kernel; 2 and 9 tokens take the widened path
        for t in [1, 2, 9] {
            let x = data(t * n_in, 8);
            let got = cpu().matmul(&x, t, &q, Some(&bias));
            let want = cpu().matmul(&x, t, &widened, Some(&bias));
            assert_close(&got, &want, 1e-5, 1e-4);
        }
    }

    #[test]
    fn rmsnorm_toy() {
        // mean square 12.5, so every value is divided by sqrt(12.5) = 3.5355339
        let got = cpu().rmsnorm(&[3., 4.], &[1., 2.], 0.0);
        assert_close(&got, &[0.8485281, 2.2627417], 1e-6, 0.0);
    }

    #[test]
    fn swiglu_toy() {
        // one row, width 2: silu(1) * 2 and silu(-1) * 3
        let got = cpu().swiglu(&[1., -1., 2., 3.], 2);
        assert_close(&got, &[1.4621172, -0.8068243], 1e-6, 0.0);
        // more rows than one chunk, so the flat-index walk crosses row ends
        let (rows, width) = (3, 700);
        let x = data(rows * 2 * width, 5);
        let got = cpu().swiglu(&x, width);
        for r in 0..rows {
            for j in 0..width {
                let (g, u) = (x[r * 2 * width + j], x[r * 2 * width + width + j]);
                assert!((got[r * width + j] - g / (1.0 + (-g).exp()) * u).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rope_rotates_half_pairs() {
        // head_dim 4: elements 0 and 2 pair, 1 and 3 pair. At position 1 a
        // quarter turn (pi / 2) sends (1, 0) to (0, 1) and leaves the other
        // pair, whose frequency is 0, alone.
        let mut x = [1., 5., 0., 7.];
        cpu().rope(&mut x, 1, 1, 4, &[std::f32::consts::FRAC_PI_2, 0.0]);
        assert_close(&x, &[0., 5., 1., 7.], 1e-6, 0.0);
    }

    #[test]
    fn rope_position_zero_is_identity_and_offset_matches() {
        let (n_heads, head_dim) = (3, 8);
        let inv_freq: Vec<f32> = (0..4).map(|j| 10000f32.powf(-(j as f32) / 4.0)).collect();
        let x = data(4 * n_heads * head_dim, 9);
        let mut all = x.clone();
        cpu().rope(&mut all, 0, n_heads, head_dim, &inv_freq);
        assert_close(
            &all[..n_heads * head_dim],
            &x[..n_heads * head_dim],
            1e-7,
            0.0,
        );
        // The property the KV cache relies on: row 3 rotated alone at pos0 = 3
        // equals row 3 of the full pass.
        let row = n_heads * head_dim;
        let mut last = x[3 * row..].to_vec();
        cpu().rope(&mut last, 3, n_heads, head_dim, &inv_freq);
        assert_close(&last, &all[3 * row..], 1e-7, 0.0);
        // A rotation does not change length.
        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        assert!((norm(&all) - norm(&x)).abs() < 1e-4);
    }

    #[test]
    fn gelu_known_values() {
        let mut x = [0.0, 1.0, -1.0, 10.0, -10.0];
        cpu().gelu(&mut x);
        assert_close(&x, &[0.0, 0.841192, -0.158808, 10.0, 0.0], 1e-5, 0.0);
    }

    #[test]
    fn softmax_is_stable_and_sums_to_one() {
        let mut x = [1000.0, 1000.0, 0.0];
        cpu().softmax(&mut x);
        assert_close(&x, &[0.5, 0.5, 0.0], 1e-6, 0.0);
        let mut y = data(10, 7);
        cpu().softmax(&mut y);
        assert!((y.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    fn shape(t: usize, pos0: usize, n_heads: usize, n_kv_heads: usize) -> AttnShape {
        AttnShape {
            t,
            pos0,
            n_heads,
            n_kv_heads,
            head_dim: 4,
        }
    }

    #[test]
    fn attention_first_token_returns_its_own_value() {
        // Query 0 sees only key 0, so softmax is [1] and the output is v[0].
        let s = shape(3, 0, 2, 2);
        let (q, k, v) = (data(3 * 8, 1), data(3 * 8, 2), data(3 * 8, 3));
        let got = cpu().attention(&q, &k, &v, s);
        assert_close(&got[..8], &v[..8], 1e-6, 0.0);
    }

    #[test]
    fn attention_decode_equals_last_row_of_prefill() {
        // The property the KV cache relies on: one query at pos0 = 4 against
        // cached keys gives the same row as the full 5-token pass.
        for (n_heads, n_kv_heads) in [(2, 2), (4, 2), (4, 1)] {
            let (q_dim, kv_dim) = (n_heads * 4, n_kv_heads * 4);
            let (q, k, v) = (data(5 * q_dim, 1), data(5 * kv_dim, 2), data(5 * kv_dim, 3));
            let full = cpu().attention(&q, &k, &v, shape(5, 0, n_heads, n_kv_heads));
            let last = cpu().attention(&q[4 * q_dim..], &k, &v, shape(1, 4, n_heads, n_kv_heads));
            assert_close(&last, &full[4 * q_dim..], 1e-6, 0.0);
        }
    }

    #[test]
    fn attention_grouped_heads_share_one_kv_head() {
        // Two query heads with identical q and one kv head must give identical outputs.
        let q1 = data(4, 1);
        let q: Vec<f32> = q1.iter().chain(&q1).copied().collect();
        let (k, v) = (data(3 * 4, 2), data(3 * 4, 3));
        let got = cpu().attention(&q[..8], &k, &v, shape(1, 2, 2, 1));
        assert_close(&got[..4], &got[4..], 1e-7, 0.0);
    }
}
