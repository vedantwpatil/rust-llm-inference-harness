pub mod backend;
pub mod cpu;
pub mod gpt2;
pub mod kv_cache;
pub mod ops;
pub mod safetensors;

/// The largest `|got - want|` is at most `rel * max|want|`: error measured
/// against the tensor's own scale. GPT-2's residual stream has entries near
/// 3000 next to entries near 0, and f32 summation order moves every entry by a
/// few units in the 7th digit of the *largest* one, so per-element relative
/// error is meaningless for the small entries.
#[cfg(test)]
pub(crate) fn assert_close_to_scale(got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(
        err <= rel * scale,
        "max error {err} vs tensor scale {scale}"
    );
}

/// `|got - want| <= atol + rtol * |want|` for every element.
#[cfg(test)]
pub(crate) fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= atol + rtol * w.abs(),
            "index {i}: got {g}, want {w}"
        );
    }
}
