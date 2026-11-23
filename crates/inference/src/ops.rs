pub fn embed(wte: &[f32], wpe: &[f32], ids: &[u32], w: usize) -> Vec<f32> {
    let mut final_vec: Vec<f32> = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        let t = id as usize;
        let token_start = t * w;
        let position_start = pos * w;

        let token_row = &wte[token_start..token_start + w];
        let position_row = &wpe[position_start..position_start + w];

        final_vec.extend(
            token_row
                .iter()
                .zip(position_row.iter())
                .map(|(a, b)| a + b),
        );
    }
    final_vec
}

/// The residual connection: `x += y`.
pub fn add_in_place(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Index of the largest value, the first one on a tie (as `torch.argmax`).
pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::load;
    use std::path::Path;

    #[test]
    fn argmax_takes_first_of_ties() {
        assert_eq!(argmax(&[1., 5., 5., 2.]), 1);
        let mut x = [1., 2.];
        add_in_place(&mut x, &[10., 20.]);
        assert_eq!(x, [11., 22.]);
    }

    #[test]
    fn embed_toy() {
        // width 4, three tokens, three positions
        let wte = [
            10., 11., 12., 13., //
            20., 21., 22., 23., //
            30., 31., 32., 33.,
        ];
        let wpe = [
            1., 2., 3., 4., //
            5., 6., 7., 8., //
            9., 10., 11., 12.,
        ];
        assert_eq!(
            embed(&wte, &wpe, &[2, 0], 4),
            [31., 33., 35., 37., 15., 17., 19., 21.]
        );
    }

    #[test]
    fn embed_matches_reference() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/data/gpt2/");
        let model = Path::new(dir).join("model.safetensors");
        let reference = Path::new(dir).join("reference.safetensors");
        // both files are gitignored, so a fresh clone has neither
        if !model.exists() || !reference.exists() {
            return;
        }
        let model = load(&model);
        let reference = load(&reference);

        let (shape, wte) = &model["wte.weight"];
        let (_, wpe) = &model["wpe.weight"];
        // the fixture stores ids as f32
        let ids: Vec<u32> = reference["ids"].1.iter().map(|&x| x as u32).collect();

        let got = embed(wte, wpe, &ids, shape[1]);
        let want = &reference["embed"].1;

        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
