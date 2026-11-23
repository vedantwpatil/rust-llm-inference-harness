use crate::backend::{AttnShape, Backend, Matrix};
use rayon::prelude::*;

/// Plain Rust on the CPU, rayon across cores. The reference every other
/// backend is compared against.
pub struct CpuBackend;

/// Dot product with 8 independent accumulators. Float addition is not
/// associative, so LLVM will not reorder one running sum into SIMD lanes; eight
/// separate sums are lanes by construction.
// shortcut: mul_add is one instruction on aarch64 but a slow libm call on
// x86 without target-cpu=native, fine while this targets Apple Silicon.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let mut a8 = a.chunks_exact(8);
    let mut b8 = b.chunks_exact(8);
    for (x, y) in (&mut a8).zip(&mut b8) {
        for l in 0..8 {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in a8.remainder().iter().zip(b8.remainder()) {
        sum = x.mul_add(*y, sum);
    }
    sum
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
        by_out
            .par_chunks_mut(ROWS * t)
            .enumerate()
            .for_each(|(c, slab)| {
                for (r, ys) in slab.chunks_mut(t).enumerate() {
                    let o = c * ROWS + r;
                    let wrow = &w.data[o * n_in..(o + 1) * n_in];
                    for (tt, y) in ys.iter_mut().enumerate() {
                        *y = dot(&x[tt * n_in..(tt + 1) * n_in], wrow);
                    }
                }
            });

        let mut out = vec![0.0; t * n_out];
        for o in 0..n_out {
            let b = bias.map_or(0.0, |b| b[o]);
            for tt in 0..t {
                out[tt * n_out + o] = by_out[o * t + tt] + b;
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
        for v in x.iter_mut() {
            let inner = SQRT_2_OVER_PI * (*v + 0.044715 * *v * *v * *v);
            *v = 0.5 * *v * (1.0 + inner.tanh());
        }
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
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(idx, o)| {
                let (i, h) = (idx / n_heads, idx % n_heads);
                let kv_off = (h / group) * head_dim;
                let qv = &q[i * q_dim + h * head_dim..][..head_dim];
                // The causal mask: key j > pos0 + i is never scored, which is
                // the same as a -inf score before softmax.
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

    // Small deterministic pseudo-random data, no rand dependency.
    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn matmul_toy() {
        // x [2,3] @ W^T, W [2,3] -> [2,2]
        let x = [1., 2., 3., 4., 5., 6.];
        let w = Matrix::new(2, 3, vec![1., 0., 1., 0., 1., 0.]);
        let got = CpuBackend.matmul(&x, 2, &w, Some(&[10., 20.]));
        assert_eq!(got, [14., 22., 20., 25.]);
    }

    #[test]
    fn matmul_matches_naive_loops() {
        // sizes chosen to hit the 8-wide dot remainder, a partial ROWS chunk and t > 1
        for (t, n_in, n_out) in [(1, 19, 37), (5, 19, 37), (3, 64, 16), (1, 5, 1)] {
            let x = data(t * n_in, 1);
            let w = Matrix::new(n_out, n_in, data(n_out * n_in, 2));
            let bias = data(n_out, 3);
            let got = CpuBackend.matmul(&x, t, &w, Some(&bias));
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
        let got = CpuBackend.layernorm(&[1., 2., 3., 4.], &[1.; 4], &[0.; 4], 0.0);
        assert_close(
            &got,
            &[-1.3416408, -0.4472136, 0.4472136, 1.3416408],
            1e-6,
            0.0,
        );
        // gamma scales and beta shifts; each row is normalized on its own, so
        // [1, 3] and [10, 14] (different mean and spread) land on the same output
        let got = CpuBackend.layernorm(&[1., 3., 10., 14.], &[2., 2.], &[1., 1.], 0.0);
        assert_close(&got, &[-1., 3., -1., 3.], 1e-6, 0.0);
    }

    #[test]
    fn gelu_known_values() {
        let mut x = [0.0, 1.0, -1.0, 10.0, -10.0];
        CpuBackend.gelu(&mut x);
        assert_close(&x, &[0.0, 0.841192, -0.158808, 10.0, 0.0], 1e-5, 0.0);
    }

    #[test]
    fn softmax_is_stable_and_sums_to_one() {
        let mut x = [1000.0, 1000.0, 0.0];
        CpuBackend.softmax(&mut x);
        assert_close(&x, &[0.5, 0.5, 0.0], 1e-6, 0.0);
        let mut y = data(10, 7);
        CpuBackend.softmax(&mut y);
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
        let got = CpuBackend.attention(&q, &k, &v, s);
        assert_close(&got[..8], &v[..8], 1e-6, 0.0);
    }

    #[test]
    fn attention_decode_equals_last_row_of_prefill() {
        // The property the KV cache relies on: one query at pos0 = 4 against
        // cached keys gives the same row as the full 5-token pass.
        for (n_heads, n_kv_heads) in [(2, 2), (4, 2), (4, 1)] {
            let (q_dim, kv_dim) = (n_heads * 4, n_kv_heads * 4);
            let (q, k, v) = (data(5 * q_dim, 1), data(5 * kv_dim, 2), data(5 * kv_dim, 3));
            let full = CpuBackend.attention(&q, &k, &v, shape(5, 0, n_heads, n_kv_heads));
            let last =
                CpuBackend.attention(&q[4 * q_dim..], &k, &v, shape(1, 4, n_heads, n_kv_heads));
            assert_close(&last, &full[4 * q_dim..], 1e-6, 0.0);
        }
    }

    #[test]
    fn attention_grouped_heads_share_one_kv_head() {
        // Two query heads with identical q and one kv head must give identical outputs.
        let q1 = data(4, 1);
        let q: Vec<f32> = q1.iter().chain(&q1).copied().collect();
        let (k, v) = (data(3 * 4, 2), data(3 * 4, 3));
        let got = CpuBackend.attention(&q[..8], &k, &v, shape(1, 2, 2, 1));
        assert_close(&got[..4], &got[4..], 1e-7, 0.0);
    }
}
