pub mod backend;
pub mod cpu;
pub mod ops;
pub mod safetensors;

/// `|got - want| <= atol + rtol * |want|` for every element. Hidden states in
/// later GPT-2 blocks reach the thousands, so a bare absolute 1e-4 would ask
/// f32 for more digits than it has.
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
