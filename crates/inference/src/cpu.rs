use crate::backend::{AttnShape, Backend, Matrix};
use crate::kernel::{dot, tile};
use crate::pool::Pool;

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

impl Backend for CpuBackend {
    fn matmul(&self, x: &[f32], t: usize, w: &Matrix, bias: Option<&[f32]>) -> Vec<f32> {
        let (n_out, n_in) = (w.rows, w.cols);
        assert_eq!(x.len(), t * n_in);
        if let Some(b) = bias {
            assert_eq!(b.len(), n_out);
        }
        // Each task owns ROWS whole weight rows and every token's dot product
        // with them, so a weight row is read from memory once however large t
        // is. That makes the result `[n_out][t]`; transposed below.
        const ROWS: usize = 16;
        let mut by_out = vec![0.0; n_out * t];
        self.pool
            .for_each_chunk_mut(&mut by_out, ROWS * t, |c, slab| {
                let rows = slab.len() / t;
                matmul_block(&w.data[c * ROWS * n_in..][..rows * n_in], x, n_in, t, slab);
            });

        let mut out = vec![0.0; t * n_out];
        for o in 0..n_out {
            let b = bias.map_or(0.0, |b| b[o]);
            for i in 0..t {
                out[i * n_out + o] = by_out[o * t + i] + b;
            }
        }
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
                        s += x[tt * n_in + i] as f64 * w.data[o * n_in + i] as f64;
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
