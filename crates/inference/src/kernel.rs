//! Dot-product microkernels. Everything the model multiplies goes through
//! `tile`, so there is one place to make fast and one place to check.

/// `R` rows of `w` against `T` rows of `x`, each row `k` floats, returning all
/// `R * T` dot products: `out[r][t] = w[r] . x[t]`.
///
/// Each weight value loaded is used `T` times and each activation value `R`
/// times, which is what makes prefill (many tokens) compute-bound instead of
/// load-bound. `U` vector accumulators per output hide the latency of the
/// fused multiply-add: a dependent chain of FMAs issues one per ~4 cycles,
/// while the core can start four per cycle. `R * T * U` is the number of live
/// accumulator registers and should stay at or below 16 (of 32).
// The loop indices are pointer offsets here, not just slice positions.
#[allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn tile<const R: usize, const T: usize, const U: usize>(
    w: &[f32],
    x: &[f32],
    k: usize,
) -> [[f32; T]; R] {
    use std::arch::aarch64::*;
    // `k <= len / R`, not `len >= R * k`: the product can wrap in release builds.
    assert!(k <= w.len() / R && k <= x.len() / T);
    // The vector loop takes whole groups of 4 * U floats; a scalar loop does the rest.
    let main = k / (4 * U) * (4 * U);
    let mut out = [[0.0f32; T]; R];

    // SAFETY: the cfg gate guarantees NEON. Every load reads 4 floats
    // at `row * k + kk + 4 * u` with `kk + 4 * U <= main <= k` and `row < R`
    // (or `< T`), so it lies inside row `row` of `w` (or `x`), and the assert
    // above guarantees those rows exist.
    unsafe {
        let mut acc = [[[vdupq_n_f32(0.0); U]; T]; R];
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut kk = 0;
        while kk < main {
            let mut xv = [[vdupq_n_f32(0.0); U]; T];
            for t in 0..T {
                for u in 0..U {
                    xv[t][u] = vld1q_f32(xp.add(t * k + kk + 4 * u));
                }
            }
            for r in 0..R {
                for u in 0..U {
                    let wv = vld1q_f32(wp.add(r * k + kk + 4 * u));
                    for t in 0..T {
                        acc[r][t][u] = vfmaq_f32(acc[r][t][u], wv, xv[t][u]);
                    }
                }
            }
            kk += 4 * U;
        }
        for r in 0..R {
            for t in 0..T {
                let mut sum = acc[r][t][0];
                for u in 1..U {
                    sum = vaddq_f32(sum, acc[r][t][u]);
                }
                out[r][t] = vaddvq_f32(sum);
            }
        }
    }

    for r in 0..R {
        for t in 0..T {
            for kk in main..k {
                out[r][t] = w[r * k + kk].mul_add(x[t * k + kk], out[r][t]);
            }
        }
    }
    out
}

// shortcut: a scalar loop, not vectorized by hand, so the crate builds off
// Apple Silicon; write an AVX2 or generic-SIMD tile when a second target matters.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub fn tile<const R: usize, const T: usize, const U: usize>(
    w: &[f32],
    x: &[f32],
    k: usize,
) -> [[f32; T]; R] {
    // `k <= len / R`, not `len >= R * k`: the product can wrap in release builds.
    assert!(k <= w.len() / R && k <= x.len() / T);
    let mut out = [[0.0f32; T]; R];
    for r in 0..R {
        for t in 0..T {
            for kk in 0..k {
                out[r][t] = w[r * k + kk].mul_add(x[t * k + kk], out[r][t]);
            }
        }
    }
    out
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    tile::<1, 1, 4>(a, b, a.len())[0][0]
}

/// Dot product of int8 weights with f32 activations, not yet multiplied by the
/// row's scale. Decode is memory-bound, so the point is to read a quarter of the
/// bytes `dot` reads; the widening to f32 happens in registers, 16 weights per
/// step, into four accumulators so the fused multiply-adds overlap.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn dot_q8(w: &[i8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    assert_eq!(w.len(), x.len());
    let main = w.len() / 16 * 16;
    let mut sum;
    // SAFETY: the cfg gate guarantees NEON. Each step loads 16 bytes
    // from `w` and 4 floats four times from `x`, all starting below `main`
    // and ending at most at `main <= len`, which the assert ties to both slices.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        for k in (0..main).step_by(16) {
            let q = vld1q_s8(wp.add(k));
            let (lo, hi) = (vmovl_s8(vget_low_s8(q)), vmovl_high_s8(q));
            let widened = [
                vmovl_s16(vget_low_s16(lo)),
                vmovl_high_s16(lo),
                vmovl_s16(vget_low_s16(hi)),
                vmovl_high_s16(hi),
            ];
            for (u, v) in widened.into_iter().enumerate() {
                acc[u] = vfmaq_f32(acc[u], vcvtq_f32_s32(v), vld1q_f32(xp.add(k + 4 * u)));
            }
        }
        sum = vaddvq_f32(vaddq_f32(
            vaddq_f32(acc[0], acc[1]),
            vaddq_f32(acc[2], acc[3]),
        ));
    }
    for k in main..w.len() {
        sum += w[k] as f32 * x[k];
    }
    sum
}

// shortcut: scalar loop off aarch64, like `tile`; write an AVX2 version when a
// second target matters.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub fn dot_q8(w: &[i8], x: &[f32]) -> f32 {
    assert_eq!(w.len(), x.len());
    w.iter().zip(x).map(|(&q, v)| q as f32 * v).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;
    use crate::test_data::data;

    fn naive<const R: usize, const T: usize>(w: &[f32], x: &[f32], k: usize) -> [[f32; T]; R] {
        let mut out = [[0.0; T]; R];
        for r in 0..R {
            for t in 0..T {
                out[r][t] = (0..k)
                    .map(|i| w[r * k + i] as f64 * x[t * k + i] as f64)
                    .sum::<f64>() as f32;
            }
        }
        out
    }

    fn check<const R: usize, const T: usize, const U: usize>(k: usize) {
        let (w, x) = (data(R * k, 1), data(T * k, 2));
        let got = tile::<R, T, U>(&w, &x, k);
        let want = naive::<R, T>(&w, &x, k);
        for r in 0..R {
            assert_close(&got[r], &want[r], 1e-5, 1e-5);
        }
    }

    #[test]
    fn tiles_match_naive_for_every_shape_used() {
        // k covers: shorter than one vector group, a tail after whole groups, exact multiples
        for k in [0, 3, 4, 15, 16, 19, 64, 100, 768] {
            check::<4, 4, 1>(k);
            check::<4, 1, 4>(k);
            check::<1, 1, 4>(k);
        }
    }

    #[test]
    fn dot_q8_matches_widening_first() {
        // lengths: shorter than one step, a tail after whole steps, exact multiples
        for k in [0, 3, 15, 16, 17, 100, 896] {
            let w: Vec<i8> = data(k, 3).iter().map(|v| (v * 127.0) as i8).collect();
            let x = data(k, 4);
            let want: f64 = w.iter().zip(&x).map(|(&q, &v)| q as f64 * v as f64).sum();
            assert_close(&[dot_q8(&w, &x)], &[want as f32], 1e-5, 1e-4);
        }
    }

    #[test]
    fn dot_toy_and_length_check() {
        assert_eq!(dot(&[1., 2., 3.], &[4., 5., 6.]), 32.0);
        assert!(std::panic::catch_unwind(|| dot(&[1.0], &[1.0, 2.0])).is_err());
    }
}
