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

/// `e^x`, written so that a loop over a slice vectorizes: no branch and no call,
/// where libm's `expf` is one call per value (about 2.5 ns each, which made
/// swiglu and softmax a tenth of a prefill). The relative error is below 3e-7
/// for `x` in -87..=88; outside it `x` is clamped, which gives 1.6e-38 or 1.6e38
/// where `exp` would give 0 or infinity.
#[inline]
pub fn exp(x: f32) -> f32 {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    // ln 2 as a float with its low bits zero (so `n * LN2_HI` is exact for the
    // small integers `n` here) plus the remainder: the reduced argument stays
    // accurate where one constant would round it.
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    let x = x.clamp(-87.0, 88.0);
    // e^x = 2^n * e^r with |r| <= ln 2 / 2, where a degree-6 series is enough.
    let n = (x * LOG2_E).round();
    let r = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let mut p = 1.0f32 / 720.0;
    for c in [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p = p.mul_add(r, c);
    }
    // 2^n is an exponent field: n + 127 in bits 23..31.
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// Activation values that share one int8 scale. A scale per whole token lets
/// one outlier channel crush the precision of every other value in it (top-1
/// agreement with f32 drops 2.5 points on Qwen); a scale per block confines the
/// damage. Longer blocks mean fewer conversions per dot product, shorter ones
/// a better fit to the values.
pub const Q8_BLOCK: usize = 128;

/// The scale that maps the largest magnitude in `row` to 127. An all-zero row
/// has nothing to scale; 1 keeps the division finite.
pub fn q8_scale(row: &[f32]) -> f32 {
    let max = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max == 0.0 { 1.0 } else { max / 127.0 }
}

/// `row` divided by `scale` and rounded to int8, written to `out`.
pub fn quantize_into(row: &[f32], scale: f32, out: &mut [i8]) {
    let inv = 1.0 / scale;
    for (q, v) in out.iter_mut().zip(row) {
        *q = (v * inv).round() as i8;
    }
}

/// Where element `col` of row `r` sits in a group of `rows` int8 weight rows of
/// `k` values each, which are stored block by block: block 0 of every row, then
/// block 1 of every row, and so on. A tile that dots all `rows` rows with one
/// activation row then reads one contiguous run of memory, where plain rows
/// would be `rows` streams `k` bytes apart: one thread streams a packed matrix
/// at 91-94 GB/s and the same bytes as plain rows at 29-68, depending on `k`.
/// One row (`rows == 1`) is the plain layout.
pub fn packed_index(rows: usize, r: usize, col: usize, k: usize) -> usize {
    let (b, within) = (col / Q8_BLOCK, col % Q8_BLOCK);
    // The last block is shorter when `k` is not a multiple of `Q8_BLOCK`.
    let len = Q8_BLOCK.min(k - b * Q8_BLOCK);
    b * rows * Q8_BLOCK + r * len + within
}

/// `tile` for a group of `R` int8 weight rows `w` (`k` values each, laid out as
/// `packed_index` says) against int8 activation rows `x`, where `x_scales`
/// holds `k.div_ceil(Q8_BLOCK)` scales per row. Returns the dot products of the
/// dequantized activations with the *unscaled* weights: the caller multiplies
/// each weight row's own scale in.
///
/// `sdot` multiplies 16 int8 pairs into four int32 sums, four times the
/// multiply-adds of an f32 `fmla`. Sums are int32 within one block, then
/// converted and scaled into f32 accumulators, so a block costs one convert and
/// one fused multiply-add per output beyond the dot products themselves.
#[allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn tile_q8<const R: usize, const T: usize>(
    w: &[i8],
    x: &[i8],
    x_scales: &[f32],
    k: usize,
) -> [[f32; T]; R] {
    use std::arch::aarch64::*;
    let blocks = k.div_ceil(Q8_BLOCK);
    assert!(R * k <= w.len() && k <= x.len() / T && blocks <= x_scales.len() / T);
    let full = k / Q8_BLOCK;
    let mut out = [[0.0f32; T]; R];

    // SAFETY: the cfg gate guarantees NEON. A weight load reads 16 bytes at
    // `(b * R + r) * Q8_BLOCK + s` with `b < full`, `r < R` and
    // `s + 16 <= Q8_BLOCK`, so it ends at most at `full * R * Q8_BLOCK <= R * k`,
    // inside `w` by the assert above. An activation load reads 16 bytes at
    // `t * k + b * Q8_BLOCK + s`, which ends at most at `full * Q8_BLOCK <= k`
    // inside row `t < T` of `x`, which the assert guarantees exists.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); T]; R];
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        for b in 0..full {
            let mut sums = [[vdupq_n_s32(0); T]; R];
            for s in (0..Q8_BLOCK).step_by(16) {
                let at = b * Q8_BLOCK + s;
                let xv: [int8x16_t; T] = std::array::from_fn(|t| vld1q_s8(xp.add(t * k + at)));
                for r in 0..R {
                    let wv = vld1q_s8(wp.add((b * R + r) * Q8_BLOCK + s));
                    for t in 0..T {
                        sums[r][t] = vdotq_s32(sums[r][t], wv, xv[t]);
                    }
                }
            }
            for t in 0..T {
                let scale = x_scales[t * blocks + b];
                for r in 0..R {
                    acc[r][t] = vfmaq_n_f32(acc[r][t], vcvtq_f32_s32(sums[r][t]), scale);
                }
            }
        }
        for r in 0..R {
            for t in 0..T {
                out[r][t] = vaddvq_f32(acc[r][t]);
            }
        }
    }

    if full < blocks {
        for r in 0..R {
            for t in 0..T {
                let at = full * Q8_BLOCK;
                let sum = dot_i8(
                    &w[packed_index(R, r, at, k)..][..k - at],
                    &x[t * k + at..][..k - at],
                );
                out[r][t] += sum as f32 * x_scales[t * blocks + full];
            }
        }
    }
    out
}

// shortcut: scalar loops off aarch64, like `tile`; write an AVX2 version when a
// second target matters.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub fn tile_q8<const R: usize, const T: usize>(
    w: &[i8],
    x: &[i8],
    x_scales: &[f32],
    k: usize,
) -> [[f32; T]; R] {
    let blocks = k.div_ceil(Q8_BLOCK);
    assert!(R * k <= w.len() && k <= x.len() / T && blocks <= x_scales.len() / T);
    let mut out = [[0.0f32; T]; R];
    for r in 0..R {
        for t in 0..T {
            for b in 0..blocks {
                let (from, to) = (b * Q8_BLOCK, ((b + 1) * Q8_BLOCK).min(k));
                let sum = dot_i8(
                    &w[packed_index(R, r, from, k)..][..to - from],
                    &x[t * k + from..t * k + to],
                );
                out[r][t] += sum as f32 * x_scales[t * blocks + b];
            }
        }
    }
    out
}

fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    w.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum()
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
    fn int8_tiles_match_naive() {
        // k covers: shorter than a block, a partial block after whole ones, exact multiples
        for k in [0, 3, 16, 100, Q8_BLOCK, Q8_BLOCK + 3, 3 * Q8_BLOCK, 896] {
            let blocks = k.div_ceil(Q8_BLOCK);
            let w: Vec<i8> = data(4 * k, 3).iter().map(|v| (v * 127.0) as i8).collect();
            let mut packed = vec![0i8; 4 * k];
            for (i, &v) in w.iter().enumerate() {
                packed[packed_index(4, i / k, i % k, k)] = v;
            }
            let x: Vec<i8> = data(4 * k, 4).iter().map(|v| (v * 127.0) as i8).collect();
            let xs: Vec<f32> = data(4 * blocks, 5).iter().map(|v| v.abs() + 0.5).collect();
            let naive = |r: usize, t: usize| -> f32 {
                (0..k)
                    .map(|i| {
                        let scale = xs[t * blocks + i / Q8_BLOCK];
                        w[r * k + i] as f64 * x[t * k + i] as f64 * scale as f64
                    })
                    .sum::<f64>() as f32
            };
            let (a, b) = (
                tile_q8::<4, 4>(&packed, &x, &xs, k),
                tile_q8::<4, 1>(&packed, &x, &xs, k),
            );
            for r in 0..4 {
                for (t, got) in a[r].iter().enumerate() {
                    assert_close(&[*got], &[naive(r, t)], 1e-4, 1e-4);
                }
                assert_close(&[b[r][0]], &[naive(r, 0)], 1e-4, 1e-4);
            }
            // a lone row is stored plain
            let one = tile_q8::<1, 1>(&w[k..], &x, &xs, k);
            assert_close(&[one[0][0]], &[naive(1, 0)], 1e-4, 1e-4);
        }
    }

    #[test]
    fn quantize_keeps_the_max_exact_and_zero_rows_finite() {
        let row = data(70, 5);
        let scale = q8_scale(&row);
        let mut q = vec![0i8; row.len()];
        quantize_into(&row, scale, &mut q);
        let big = row.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let at = row.iter().position(|v| v.abs() == big).unwrap();
        assert_eq!(q[at].unsigned_abs(), 127);
        for (&q, &v) in q.iter().zip(&row) {
            assert!((q as f32 * scale - v).abs() <= 0.5 * scale * (1.0 + 1e-5));
        }
        assert_eq!(q8_scale(&[0.0; 8]), 1.0);
    }

    #[test]
    fn exp_matches_std_over_its_range() {
        let mut worst = 0.0f64;
        for i in -8700..=8800 {
            let x = i as f32 / 100.0;
            let want = (x as f64).exp();
            worst = worst.max((exp(x) as f64 - want).abs() / want);
        }
        assert!(worst < 3e-7, "relative error {worst}");
        assert_eq!(exp(0.0), 1.0);
        // clamped, never infinite or NaN
        assert!(exp(1e9).is_finite() && exp(-1e9) > 0.0 && exp(-1e9) < 1e-37);
        assert!(exp(f32::NAN).is_nan());
    }

    #[test]
    fn dot_toy_and_length_check() {
        assert_eq!(dot(&[1., 2., 3.], &[4., 5., 6.]), 32.0);
        assert!(std::panic::catch_unwind(|| dot(&[1.0], &[1.0, 2.0])).is_err());
    }
}
