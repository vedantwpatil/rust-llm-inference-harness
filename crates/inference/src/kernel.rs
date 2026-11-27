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

/// Fast 4-lane NEON `e^x` using Estrin's scheme for the same degree-6 series
/// as `exp` (3 FMA dependency depth instead of 6).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn vexpq_f32_fast(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0)), vdupq_n_f32(88.0));
        let n = vrndaq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
        let r = vfmsq_f32(
            vfmsq_f32(x, n, vdupq_n_f32(0.693_145_75)),
            n,
            vdupq_n_f32(1.428_606_8e-6),
        );
        let r2 = vmulq_f32(r, r);
        let r4 = vmulq_f32(r2, r2);
        let p01 = vaddq_f32(vdupq_n_f32(1.0), r);
        let p23 = vfmaq_f32(vdupq_n_f32(0.5), r, vdupq_n_f32(1.0 / 6.0));
        let p45 = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), r, vdupq_n_f32(1.0 / 120.0));
        let p456 = vfmaq_f32(p45, r2, vdupq_n_f32(1.0 / 720.0));
        let p = vfmaq_f32(vfmaq_f32(p01, r2, p23), r4, p456);
        let pow2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(
            vcvtaq_s32_f32(n),
            vdupq_n_s32(127),
        )));
        vmulq_f32(p, pow2)
    }
}

/// Fast 4-lane NEON `1 / d` via estimate + 2 Newton-Raphson steps (full f32
/// precision without `fdiv.4s` pipeline stalls).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn vrecipq_f32_fast(d: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let mut r = vrecpeq_f32(d);
        r = vmulq_f32(r, vrecpsq_f32(d, r));
        r = vmulq_f32(r, vrecpsq_f32(d, r));
        r
    }
}

/// `out[i] = silu(gate[i]) * up[i]` for equal-length slices, 4-vector unrolled
/// on aarch64.
pub fn swiglu_slice(gate: &[f32], up: &[f32], out: &mut [std::mem::MaybeUninit<f32>]) {
    let n = out.len();
    assert_eq!(gate.len(), n);
    assert_eq!(up.len(), n);
    let mut i = 0;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use std::arch::aarch64::*;
        let main = n & !15;
        // SAFETY: `i + 16 <= main <= n`, and `out` is written via raw pointers.
        unsafe {
            let (gp, up_p, op) = (gate.as_ptr(), up.as_ptr(), out.as_mut_ptr().cast::<f32>());
            let one = vdupq_n_f32(1.0);
            while i < main {
                let g0 = vld1q_f32(gp.add(i));
                let g1 = vld1q_f32(gp.add(i + 4));
                let g2 = vld1q_f32(gp.add(i + 8));
                let g3 = vld1q_f32(gp.add(i + 12));
                let r0 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(g0))));
                let r1 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(g1))));
                let r2 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(g2))));
                let r3 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(g3))));
                let u0 = vld1q_f32(up_p.add(i));
                let u1 = vld1q_f32(up_p.add(i + 4));
                let u2 = vld1q_f32(up_p.add(i + 8));
                let u3 = vld1q_f32(up_p.add(i + 12));
                vst1q_f32(op.add(i), vmulq_f32(vmulq_f32(g0, r0), u0));
                vst1q_f32(op.add(i + 4), vmulq_f32(vmulq_f32(g1, r1), u1));
                vst1q_f32(op.add(i + 8), vmulq_f32(vmulq_f32(g2, r2), u2));
                vst1q_f32(op.add(i + 12), vmulq_f32(vmulq_f32(g3, r3), u3));
                i += 16;
            }
        }
    }
    for j in i..n {
        let g = gate[j];
        out[j].write(g / (1.0 + exp(-g)) * up[j]);
    }
}

/// GPT-2's tanh GELU approximation in place, 4-vector unrolled on aarch64.
pub fn gelu_slice(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let n = x.len();
    let mut i = 0;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use std::arch::aarch64::*;
        let main = n & !15;
        // SAFETY: `i + 16 <= main <= n`.
        unsafe {
            let p = x.as_mut_ptr();
            let c0 = vdupq_n_f32(2.0 * SQRT_2_OVER_PI);
            let c1 = vdupq_n_f32(0.044_715);
            let one = vdupq_n_f32(1.0);
            while i < main {
                let x0 = vld1q_f32(p.add(i));
                let x1 = vld1q_f32(p.add(i + 4));
                let x2 = vld1q_f32(p.add(i + 8));
                let x3 = vld1q_f32(p.add(i + 12));
                // 0.5 * x * (2 - 2 / (e^(2a) + 1)) = x * (1 - 1 / (e^(2a) + 1)) = x / (1 + e^(-2a))
                // Keeping `2a = 2 * sqrt(2/pi) * x * (1 + 0.044715 * x^2)`:
                let a0 = vmulq_f32(c0, vfmaq_f32(x0, c1, vmulq_f32(vmulq_f32(x0, x0), x0)));
                let a1 = vmulq_f32(c0, vfmaq_f32(x1, c1, vmulq_f32(vmulq_f32(x1, x1), x1)));
                let a2 = vmulq_f32(c0, vfmaq_f32(x2, c1, vmulq_f32(vmulq_f32(x2, x2), x2)));
                let a3 = vmulq_f32(c0, vfmaq_f32(x3, c1, vmulq_f32(vmulq_f32(x3, x3), x3)));
                let r0 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(a0))));
                let r1 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(a1))));
                let r2 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(a2))));
                let r3 = vrecipq_f32_fast(vaddq_f32(one, vexpq_f32_fast(vnegq_f32(a3))));
                vst1q_f32(p.add(i), vmulq_f32(x0, r0));
                vst1q_f32(p.add(i + 4), vmulq_f32(x1, r1));
                vst1q_f32(p.add(i + 8), vmulq_f32(x2, r2));
                vst1q_f32(p.add(i + 12), vmulq_f32(x3, r3));
                i += 16;
            }
        }
    }
    for v in &mut x[i..] {
        let inner = SQRT_2_OVER_PI * (*v + 0.044715 * *v * *v * *v);
        *v = 0.5 * *v * (2.0 - 2.0 / (exp(2.0 * inner) + 1.0));
    }
}

/// Normalizes one row with 4-accumulator NEON mean and variance (avoiding Rust's
/// scalar sequential `iter().sum()`).
pub fn layernorm_row(row: &[f32], gamma: &[f32], beta: &[f32], eps: f32, out: &mut [f32]) {
    let w = row.len();
    assert_eq!(gamma.len(), w);
    assert_eq!(beta.len(), w);
    assert_eq!(out.len(), w);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use std::arch::aarch64::*;
        let main = w & !15;
        // SAFETY: `i + 16 <= main <= w`, and all slices have length `w`.
        unsafe {
            let p = row.as_ptr();
            let (mut s0, mut s1, mut s2, mut s3) = (
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            let mut i = 0;
            while i < main {
                s0 = vaddq_f32(s0, vld1q_f32(p.add(i)));
                s1 = vaddq_f32(s1, vld1q_f32(p.add(i + 4)));
                s2 = vaddq_f32(s2, vld1q_f32(p.add(i + 8)));
                s3 = vaddq_f32(s3, vld1q_f32(p.add(i + 12)));
                i += 16;
            }
            let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            for &v in &row[main..] {
                sum += v;
            }
            let mean = sum / w as f32;
            let vmean = vdupq_n_f32(mean);
            let (mut q0, mut q1, mut q2, mut q3) = (
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            let mut i = 0;
            while i < main {
                let d0 = vsubq_f32(vld1q_f32(p.add(i)), vmean);
                let d1 = vsubq_f32(vld1q_f32(p.add(i + 4)), vmean);
                let d2 = vsubq_f32(vld1q_f32(p.add(i + 8)), vmean);
                let d3 = vsubq_f32(vld1q_f32(p.add(i + 12)), vmean);
                q0 = vfmaq_f32(q0, d0, d0);
                q1 = vfmaq_f32(q1, d1, d1);
                q2 = vfmaq_f32(q2, d2, d2);
                q3 = vfmaq_f32(q3, d3, d3);
                i += 16;
            }
            let mut var_sum = vaddvq_f32(vaddq_f32(vaddq_f32(q0, q1), vaddq_f32(q2, q3)));
            for &v in &row[main..] {
                let d = v - mean;
                var_sum += d * d;
            }
            let inv_std = 1.0 / (var_sum / w as f32 + eps).sqrt();
            let vinv = vdupq_n_f32(inv_std);
            let (gp, bp, op) = (gamma.as_ptr(), beta.as_ptr(), out.as_mut_ptr());
            let mut i = 0;
            while i < main {
                for off in [0, 4, 8, 12] {
                    let norm = vmulq_f32(vsubq_f32(vld1q_f32(p.add(i + off)), vmean), vinv);
                    let y = vfmaq_f32(vld1q_f32(bp.add(i + off)), norm, vld1q_f32(gp.add(i + off)));
                    vst1q_f32(op.add(i + off), y);
                }
                i += 16;
            }
            for j in main..w {
                out[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        let mean = row.iter().sum::<f32>() / w as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ((o, v), (g, b)) in out.iter_mut().zip(row).zip(gamma.iter().zip(beta)) {
            *o = (v - mean) * inv_std * g + b;
        }
    }
}

/// Fast in-place softmax using NEON `exp` and reciprocal multiply (no `fdiv`).
pub fn softmax_slice(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let n = x.len();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use std::arch::aarch64::*;
        let mut i = 0;
        let main = n & !3;
        // SAFETY: `i + 4 <= main <= n`.
        unsafe {
            let p = x.as_mut_ptr();
            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
            while i < main {
                vmax = vmaxq_f32(vmax, vld1q_f32(p.add(i)));
                i += 4;
            }
            let mut max = vmaxvq_f32(vmax);
            for &v in &x[main..] {
                if v > max {
                    max = v;
                }
            }
            let vmax = vdupq_n_f32(max);
            let mut vsum = vdupq_n_f32(0.0);
            let mut i = 0;
            while i < main {
                let e = vexpq_f32_fast(vsubq_f32(vld1q_f32(p.add(i)), vmax));
                vst1q_f32(p.add(i), e);
                vsum = vaddq_f32(vsum, e);
                i += 4;
            }
            let mut sum = vaddvq_f32(vsum);
            for v in &mut x[main..] {
                let e = exp(*v - max);
                *v = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            let vinv = vdupq_n_f32(inv);
            let mut i = 0;
            while i < main {
                vst1q_f32(p.add(i), vmulq_f32(vld1q_f32(p.add(i)), vinv));
                i += 4;
            }
            for v in &mut x[main..] {
                *v *= inv;
            }
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            let e = exp(*v - max);
            *v = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// Accumulates `sum_j weights[j] * v[j]` into `out[0..head_dim]`, keeping the
/// full `head_dim = 64` accumulator in 16 NEON registers on aarch64.
pub fn weighted_sum_rows(weights: &[f32], v: &[f32], stride: usize, out: &mut [f32]) {
    let head_dim = out.len();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if head_dim == 64 {
        use std::arch::aarch64::*;
        // SAFETY: `head_dim == 64` (16 NEON vectors), and caller ensures
        // `v.len() >= (weights.len() - 1) * stride + 64` when non-empty.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 16];
            for (j, &p) in weights.iter().enumerate() {
                let vp = vdupq_n_f32(p);
                let vjp = v[j * stride..][..64].as_ptr();
                for (d, a) in acc.iter_mut().enumerate() {
                    *a = vfmaq_f32(*a, vp, vld1q_f32(vjp.add(d * 4)));
                }
            }
            let op = out.as_mut_ptr();
            for (d, &a) in acc.iter().enumerate() {
                vst1q_f32(op.add(d * 4), a);
            }
            return;
        }
    }
    out.fill(0.0);
    for (j, &p) in weights.iter().enumerate() {
        let vj = &v[j * stride..][..head_dim];
        for (acc, &val) in out.iter_mut().zip(vj) {
            *acc = p.mul_add(val, *acc);
        }
    }
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

/// Fused single-pass quantization of one block (up to `Q8_BLOCK` floats):
/// returns the block scale and writes the rounded int8 values into `out`.
/// Bit-identical to `q8_scale` followed by `quantize_into`, but reads `row`
/// once in L1 and uses NEON vector max/round/narrow on aarch64.
pub fn quantize_block(row: &[f32], out: &mut [i8]) -> f32 {
    assert_eq!(row.len(), out.len());
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if row.len() == Q8_BLOCK {
        use std::arch::aarch64::*;
        // SAFETY: `row` and `out` both have length `Q8_BLOCK` (128, a multiple
        // of 16), and the cfg gate guarantees NEON.
        unsafe {
            let p = row.as_ptr();
            let mut m0 = vdupq_n_f32(0.0);
            let mut m1 = vdupq_n_f32(0.0);
            let mut m2 = vdupq_n_f32(0.0);
            let mut m3 = vdupq_n_f32(0.0);
            for i in (0..Q8_BLOCK).step_by(16) {
                m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(p.add(i))));
                m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(p.add(i + 4))));
                m2 = vmaxq_f32(m2, vabsq_f32(vld1q_f32(p.add(i + 8))));
                m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(p.add(i + 12))));
            }
            let max = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
            if max == 0.0 {
                out.fill(0);
                return 1.0;
            }
            let scale = max / 127.0;
            let inv = vdupq_n_f32(1.0 / scale);
            let dp = out.as_mut_ptr();
            for i in (0..Q8_BLOCK).step_by(16) {
                let a = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(i)), inv));
                let b = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(i + 4)), inv));
                let c = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(i + 8)), inv));
                let d = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(i + 12)), inv));
                let ab = vqmovn_high_s32(vqmovn_s32(a), b);
                let cd = vqmovn_high_s32(vqmovn_s32(c), d);
                let abcd = vqmovn_high_s16(vqmovn_s16(ab), cd);
                vst1q_s8(dp.add(i), abcd);
            }
            return scale;
        }
    }
    let scale = q8_scale(row);
    quantize_into(row, scale, out);
    scale
}

/// Whether this CPU supports ARMv8.6 `i8mm` (`smmla` 2x8x2 int8 matrix multiply).
#[inline]
pub fn use_i8mm() -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        std::arch::is_aarch64_feature_detected!("i8mm")
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        false
    }
}

/// Where element `col` of row `r` (`0..8`) sits in an 8-row `smmla` weight
/// group of width `k`. Full `Q8_BLOCK` (128-element) blocks are laid out as 16
/// steps of 8 elements, 4 row-pairs × 16 bytes per step (1024 contiguous bytes
/// per block), so both 8×8 prefill and 8×1 decode stream sequentially. Any tail
/// `col >= full * Q8_BLOCK` is stored plain row-by-row after the full blocks.
#[inline]
pub fn packed_index_smmla8(r: usize, col: usize, k: usize) -> usize {
    let full = k / Q8_BLOCK;
    let full_cols = full * Q8_BLOCK;
    if col < full_cols {
        let b = col / Q8_BLOCK;
        let within = col % Q8_BLOCK;
        let step = within / 8;
        let lane = within % 8;
        let pair = r / 2;
        let in_pair = r % 2;
        b * (8 * Q8_BLOCK) + (step * 4 + pair) * 16 + in_pair * 8 + lane
    } else {
        let rem = k - full_cols;
        full * (8 * Q8_BLOCK) + r * rem + (col - full_cols)
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
/// converted and scaled into f32 accumulators. With four rows the lanes of the
/// rows' sums are added together first (`vpaddq`), so a block costs one convert
/// and one fused multiply-add per token beyond the dot products themselves, and
/// 16 accumulators fewer stay live: with an f32 accumulator per output, the
/// 16 int32 sums, the 16 accumulators and the operands outnumber the 32
/// registers and the compiler spills.
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
        // Four rows: one vector per token holds that token's four row totals.
        // Otherwise one vector per output, summed across its lanes at the end.
        let mut by_token = [vdupq_n_f32(0.0); T];
        let mut by_output = [[vdupq_n_f32(0.0); T]; R];
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
                if R == 4 {
                    // Add the lanes of the four rows' sums together first: one
                    // convert and one multiply-add instead of four of each, and
                    // 16 fewer live registers than an accumulator per output.
                    let s: [int32x4_t; 4] = std::array::from_fn(|r| sums[r][t]);
                    let (lo, hi) = (vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
                    let totals = vcvtq_f32_s32(vpaddq_s32(lo, hi));
                    by_token[t] = vfmaq_n_f32(by_token[t], totals, scale);
                } else {
                    for r in 0..R {
                        let sum = vcvtq_f32_s32(sums[r][t]);
                        by_output[r][t] = vfmaq_n_f32(by_output[r][t], sum, scale);
                    }
                }
            }
        }
        for t in 0..T {
            if R == 4 {
                let mut rows = [0.0f32; 4];
                vst1q_f32(rows.as_mut_ptr(), by_token[t]);
                for r in 0..R {
                    out[r][t] = rows[r];
                }
            } else {
                for r in 0..R {
                    out[r][t] = vaddvq_f32(by_output[r][t]);
                }
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

pub(crate) fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    w.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum()
}

/// 8 weight rows (laid out by `packed_index_smmla8`) against 8 packed activation
/// rows over the `full = k / Q8_BLOCK` full blocks of `k`, using ARMv8.6 `smmla`.
/// Returns unscaled-weight dot products `out[r][t]`; the caller handles any
/// partial block (`k % Q8_BLOCK`) and multiplies by each weight row's scale.
///
/// Uses all 32 NEON registers: `v0..v15` hold 16 int32 2×2 subtiles (64 outputs
/// with zero horizontal `addp` reductions), while `v16..v23` and `v24..v31`
/// double-buffer two 8-element steps per loop iteration.
///
/// # Safety
/// The CPU must support ARMv8.6 `i8mm` (check `use_i8mm()`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "i8mm")]
pub unsafe fn tile_q8_smmla8_8x8(
    w_blk: &[i8],
    x_blk: &[i8],
    s_blk: &[f32],
    full_blocks: usize,
) -> [[f32; 8]; 8] {
    assert!(full_blocks * 8 * Q8_BLOCK <= w_blk.len());
    assert!(full_blocks * 8 * Q8_BLOCK <= x_blk.len());
    assert!(full_blocks * 16 <= s_blk.len());
    let mut raw = [0.0f32; 64];
    let wp = w_blk.as_ptr();
    let xp = x_blk.as_ptr();
    let sp = s_blk.as_ptr();
    // SAFETY: caller guarantees `i8mm` via `target_feature`, asserts above
    // bound all pointer increments (`1024` bytes per block for `wp` and `xp`,
    // `64` bytes per block for `sp`, `256` bytes in `raw`).
    unsafe {
        std::arch::asm!(
            ".arch_extension i8mm",
            "cbz {b}, 3f",
            "2:",
            "movi v0.16b, #0",  "movi v1.16b, #0",  "movi v2.16b, #0",  "movi v3.16b, #0",
            "movi v4.16b, #0",  "movi v5.16b, #0",  "movi v6.16b, #0",  "movi v7.16b, #0",
            "movi v8.16b, #0",  "movi v9.16b, #0",  "movi v10.16b, #0", "movi v11.16b, #0",
            "movi v12.16b, #0", "movi v13.16b, #0", "movi v14.16b, #0", "movi v15.16b, #0",
            "mov {s}, #8", // 8 iterations x 2 steps x 8 elements = 128 elements
            "4:",
            // Step A
            "ldp q16, q17, [{wp}], #32",
            "ldp q18, q19, [{wp}], #32",
            "ldp q20, q21, [{xp}], #32",
            "ldp q22, q23, [{xp}], #32",
            "smmla v0.4s,  v16.16b, v20.16b",
            "smmla v1.4s,  v16.16b, v21.16b",
            "smmla v2.4s,  v16.16b, v22.16b",
            "smmla v3.4s,  v16.16b, v23.16b",
            "smmla v4.4s,  v17.16b, v20.16b",
            "smmla v5.4s,  v17.16b, v21.16b",
            "smmla v6.4s,  v17.16b, v22.16b",
            "smmla v7.4s,  v17.16b, v23.16b",
            "smmla v8.4s,  v18.16b, v20.16b",
            "smmla v9.4s,  v18.16b, v21.16b",
            "smmla v10.4s, v18.16b, v22.16b",
            "smmla v11.4s, v18.16b, v23.16b",
            "smmla v12.4s, v19.16b, v20.16b",
            "smmla v13.4s, v19.16b, v21.16b",
            "smmla v14.4s, v19.16b, v22.16b",
            "smmla v15.4s, v19.16b, v23.16b",
            // Step B
            "ldp q24, q25, [{wp}], #32",
            "ldp q26, q27, [{wp}], #32",
            "ldp q28, q29, [{xp}], #32",
            "ldp q30, q31, [{xp}], #32",
            "smmla v0.4s,  v24.16b, v28.16b",
            "smmla v1.4s,  v24.16b, v29.16b",
            "smmla v2.4s,  v24.16b, v30.16b",
            "smmla v3.4s,  v24.16b, v31.16b",
            "smmla v4.4s,  v25.16b, v28.16b",
            "smmla v5.4s,  v25.16b, v29.16b",
            "smmla v6.4s,  v25.16b, v30.16b",
            "smmla v7.4s,  v25.16b, v31.16b",
            "smmla v8.4s,  v26.16b, v28.16b",
            "smmla v9.4s,  v26.16b, v29.16b",
            "smmla v10.4s, v26.16b, v30.16b",
            "smmla v11.4s, v26.16b, v31.16b",
            "smmla v12.4s, v27.16b, v28.16b",
            "smmla v13.4s, v27.16b, v29.16b",
            "smmla v14.4s, v27.16b, v30.16b",
            "smmla v15.4s, v27.16b, v31.16b",
            "subs {s}, {s}, #1",
            "b.ne 4b",
            // Scale and accumulate into L1 `raw` once per 256 smmla instructions.
            "ldp q20, q21, [{sp}], #32",
            "ldp q22, q23, [{sp}], #32",
            "scvtf v0.4s, v0.4s",   "scvtf v1.4s, v1.4s",   "scvtf v2.4s, v2.4s",   "scvtf v3.4s, v3.4s",
            "ldp q24, q25, [{out}, #0]",  "ldp q26, q27, [{out}, #32]",
            "fmla v24.4s, v0.4s, v20.4s", "fmla v25.4s, v1.4s, v21.4s",
            "fmla v26.4s, v2.4s, v22.4s", "fmla v27.4s, v3.4s, v23.4s",
            "stp q24, q25, [{out}, #0]",  "stp q26, q27, [{out}, #32]",
            "scvtf v4.4s, v4.4s",   "scvtf v5.4s, v5.4s",   "scvtf v6.4s, v6.4s",   "scvtf v7.4s, v7.4s",
            "ldp q24, q25, [{out}, #64]", "ldp q26, q27, [{out}, #96]",
            "fmla v24.4s, v4.4s, v20.4s", "fmla v25.4s, v5.4s, v21.4s",
            "fmla v26.4s, v6.4s, v22.4s", "fmla v27.4s, v7.4s, v23.4s",
            "stp q24, q25, [{out}, #64]", "stp q26, q27, [{out}, #96]",
            "scvtf v8.4s, v8.4s",   "scvtf v9.4s, v9.4s",   "scvtf v10.4s, v10.4s", "scvtf v11.4s, v11.4s",
            "ldp q24, q25, [{out}, #128]", "ldp q26, q27, [{out}, #160]",
            "fmla v24.4s, v8.4s, v20.4s",  "fmla v25.4s, v9.4s, v21.4s",
            "fmla v26.4s, v10.4s, v22.4s", "fmla v27.4s, v11.4s, v23.4s",
            "stp q24, q25, [{out}, #128]", "stp q26, q27, [{out}, #160]",
            "scvtf v12.4s, v12.4s", "scvtf v13.4s, v13.4s", "scvtf v14.4s, v14.4s", "scvtf v15.4s, v15.4s",
            "ldp q24, q25, [{out}, #192]", "ldp q26, q27, [{out}, #224]",
            "fmla v24.4s, v12.4s, v20.4s", "fmla v25.4s, v13.4s, v21.4s",
            "fmla v26.4s, v14.4s, v22.4s", "fmla v27.4s, v15.4s, v23.4s",
            "stp q24, q25, [{out}, #192]", "stp q26, q27, [{out}, #224]",
            "subs {b}, {b}, #1",
            "b.ne 2b",
            "3:",
            b = inout(reg) full_blocks => _,
            wp = inout(reg) wp => _,
            xp = inout(reg) xp => _,
            sp = inout(reg) sp => _,
            out = in(reg) raw.as_mut_ptr(),
            s = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
    let mut out = [[0.0f32; 8]; 8];
    for p in 0..4 {
        for q in 0..4 {
            let base = (p * 4 + q) * 4;
            out[p * 2][q * 2] = raw[base];
            out[p * 2][q * 2 + 1] = raw[base + 1];
            out[p * 2 + 1][q * 2] = raw[base + 2];
            out[p * 2 + 1][q * 2 + 1] = raw[base + 3];
        }
    }
    out
}

/// Single-token decode over an 8-row `smmla` weight group (`packed_index_smmla8`),
/// streaming 1024-byte blocks sequentially.
///
/// # Safety
/// The CPU must support ARMv8.6 `i8mm` (check `use_i8mm()`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "i8mm")]
pub unsafe fn tile_q8_smmla8_8x1(
    w_blk: &[i8],
    x: &[i8],
    x_scales: &[f32],
    k: usize,
) -> [f32; 8] {
    let blocks = k.div_ceil(Q8_BLOCK);
    let full = k / Q8_BLOCK;
    assert!(8 * k <= w_blk.len() && k <= x.len() && blocks <= x_scales.len());
    let mut out = [0.0f32; 8];
    let wp = w_blk.as_ptr();
    let xp = x.as_ptr();
    let sp = x_scales.as_ptr();
    // SAFETY: caller guarantees `i8mm` via `target_feature`, asserts above
    // bound all loads over the `full` 128-element blocks.
    unsafe {
        std::arch::asm!(
            ".arch_extension i8mm",
            "movi v8.16b, #0",
            "movi v9.16b, #0",
            "cbz {b}, 3f",
            "2:",
            "movi v0.16b, #0",
            "movi v1.16b, #0",
            "movi v2.16b, #0",
            "movi v3.16b, #0",
            "mov {s}, #16",
            "4:",
            "ldp q16, q17, [{wp}], #32",
            "ldp q18, q19, [{wp}], #32",
            "ld1r {{v20.2d}}, [{xp}], #8",
            "smmla v0.4s, v16.16b, v20.16b",
            "smmla v1.4s, v17.16b, v20.16b",
            "smmla v2.4s, v18.16b, v20.16b",
            "smmla v3.4s, v19.16b, v20.16b",
            "subs {s}, {s}, #1",
            "b.ne 4b",
            // Each `v0..v3` holds `[r0·x, r0·x, r1·x, r1·x]`; `uzp1` packs lanes 0 and 2.
            "uzp1 v0.4s, v0.4s, v1.4s",
            "uzp1 v2.4s, v2.4s, v3.4s",
            "ldr s21, [{sp}], #4",
            "scvtf v0.4s, v0.4s",
            "scvtf v2.4s, v2.4s",
            "fmla v8.4s, v0.4s, v21.s[0]",
            "fmla v9.4s, v2.4s, v21.s[0]",
            "subs {b}, {b}, #1",
            "b.ne 2b",
            "3:",
            "stp q8, q9, [{out}]",
            b = inout(reg) full => _,
            wp = inout(reg) wp => _,
            xp = inout(reg) xp => _,
            sp = inout(reg) sp => _,
            out = in(reg) out.as_mut_ptr(),
            s = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v8") _, out("v9") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _,
            options(nostack),
        );
    }
    if full < blocks {
        let at = full * Q8_BLOCK;
        let rem = k - at;
        let scale = x_scales[full];
        for (r, acc) in out.iter_mut().enumerate() {
            let sum = dot_i8(
                &w_blk[full * 8 * Q8_BLOCK + r * rem..][..rem],
                &x[at..][..rem],
            );
            *acc += sum as f32 * scale;
        }
    }
    out
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

    #[test]
    fn neon_elementwise_and_softmax_match_scalar_over_wide_range() {
        let gate: Vec<f32> = (-1000..=1000).map(|i| i as f32 / 10.0).collect();
        let up: Vec<f32> = (0..gate.len()).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
        let mut out = vec![std::mem::MaybeUninit::uninit(); gate.len()];
        swiglu_slice(&gate, &up, &mut out);
        for (i, (&g, &u)) in gate.iter().zip(&up).enumerate() {
            let got = unsafe { out[i].assume_init() };
            let want = (g as f64 / (1.0 + (-g as f64).exp()) * u as f64) as f32;
            let tol = 1e-5f32.max(want.abs() * 2e-6);
            assert!(
                (got - want).abs() <= tol,
                "swiglu mismatch at g={g}: got={got}, want={want}"
            );
        }

        let mut g_in = gate.clone();
        gelu_slice(&mut g_in);
        const SQRT_2_OVER_PI: f64 = 0.797_884_560_8;
        for (&x, &got) in gate.iter().zip(&g_in) {
            let xf = x as f64;
            let want = (0.5 * xf * (1.0 + (SQRT_2_OVER_PI * (xf + 0.044715 * xf * xf * xf)).tanh()))
                as f32;
            let tol = 1e-4f32.max(want.abs() * 1e-5);
            assert!(
                (got - want).abs() <= tol,
                "gelu mismatch at x={x}: got={got}, want={want}"
            );
        }

        let mut sm: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.25).collect();
        let mut sm_ref = sm.clone();
        softmax_slice(&mut sm);
        let max = sm_ref.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = sm_ref.iter().map(|&v| ((v - max) as f64).exp()).sum();
        for (v, &orig) in sm_ref.iter_mut().zip(&sm) {
            let want = (((*v - max) as f64).exp() / sum) as f32;
            assert!((orig - want).abs() < 1e-6, "softmax mismatch: got={orig}, want={want}");
        }
    }

    #[test]
    fn quantize_block_matches_q8_scale_and_quantize_into() {
        for len in [1, 17, Q8_BLOCK] {
            let row = data(len, 9);
            let want_scale = q8_scale(&row);
            let mut want_q = vec![0i8; len];
            quantize_into(&row, want_scale, &mut want_q);
            let mut got_q = vec![0i8; len];
            let got_scale = quantize_block(&row, &mut got_q);
            assert_eq!(got_scale, want_scale);
            assert_eq!(got_q, want_q);
        }
        let zeros = vec![0.0f32; Q8_BLOCK];
        let mut q = vec![7i8; Q8_BLOCK];
        assert_eq!(quantize_block(&zeros, &mut q), 1.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn smmla8_tiles_match_naive() {
        if !use_i8mm() {
            return;
        }
        for k in [0, 3, 16, 100, Q8_BLOCK, Q8_BLOCK + 3, 3 * Q8_BLOCK, 896] {
            let blocks = k.div_ceil(Q8_BLOCK);
            let full = k / Q8_BLOCK;
            let w: Vec<i8> = data(8 * k, 3).iter().map(|v| (v * 127.0) as i8).collect();
            let mut w_packed = vec![0i8; 8 * k];
            for r in 0..8 {
                for c in 0..k {
                    w_packed[packed_index_smmla8(r, c, k)] = w[r * k + c];
                }
            }
            let x: Vec<i8> = data(8 * k, 4).iter().map(|v| (v * 127.0) as i8).collect();
            let xs: Vec<f32> = data(8 * blocks, 5).iter().map(|v| v.abs() + 0.5).collect();
            let mut x_packed = vec![0i8; full * 8 * Q8_BLOCK];
            let mut s_packed = vec![0.0f32; full * 16];
            for b in 0..full {
                for s in 0..(Q8_BLOCK / 8) {
                    for p in 0..4 {
                        let base = b * 8 * Q8_BLOCK + (s * 4 + p) * 16;
                        let col = b * Q8_BLOCK + s * 8;
                        x_packed[base..base + 8].copy_from_slice(&x[(p * 2) * k + col..][..8]);
                        x_packed[base + 8..base + 16]
                            .copy_from_slice(&x[(p * 2 + 1) * k + col..][..8]);
                    }
                }
                for p in 0..4 {
                    let s0 = xs[(p * 2) * blocks + b];
                    let s1 = xs[(p * 2 + 1) * blocks + b];
                    s_packed[b * 16 + p * 4..][..4].copy_from_slice(&[s0, s1, s0, s1]);
                }
            }
            let naive_full = |r: usize, t: usize| -> f32 {
                (0..full * Q8_BLOCK)
                    .map(|i| {
                        w[r * k + i] as f64
                            * x[t * k + i] as f64
                            * xs[t * blocks + i / Q8_BLOCK] as f64
                    })
                    .sum::<f64>() as f32
            };
            let naive_all = |r: usize| -> f32 {
                (0..k)
                    .map(|i| w[r * k + i] as f64 * x[i] as f64 * xs[i / Q8_BLOCK] as f64)
                    .sum::<f64>() as f32
            };
            let got8 = unsafe { tile_q8_smmla8_8x8(&w_packed, &x_packed, &s_packed, full) };
            let got1 = unsafe { tile_q8_smmla8_8x1(&w_packed, &x[..k], &xs[..blocks], k) };
            for r in 0..8 {
                for (t, &v) in got8[r].iter().enumerate() {
                    assert_close(&[v], &[naive_full(r, t)], 1e-4, 1e-4);
                }
                assert_close(&[got1[r]], &[naive_all(r)], 1e-4, 1e-4);
            }
        }
    }
}
