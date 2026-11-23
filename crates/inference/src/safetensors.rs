use std::{collections::HashMap, path::Path};

/// Reads every tensor, widening BF16 to f32.
pub fn load(path: &Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let header_end = 8 + n as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).unwrap();
    let data = &bytes[header_end..];

    let mut tensors = HashMap::new();
    for (name, entry) in header.as_object().unwrap() {
        if name.as_str() == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let offsets = entry["data_offsets"].as_array().unwrap();
        let begin = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;

        let bytes = &data[begin..end];
        let n = shape.iter().product::<usize>();
        let values: Vec<f32> = match entry["dtype"].as_str().unwrap() {
            "F32" => {
                assert_eq!(bytes.len(), n * 4);
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            }
            // bfloat16 is the top 16 bits of an f32, so widening is a shift.
            "BF16" => {
                assert_eq!(bytes.len(), n * 2);
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                    .collect()
            }
            other => panic!("tensor {name} has dtype {other}, only F32 and BF16 are read"),
        };
        tensors.insert(name.clone(), (shape, values));
    }
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_gpt2() {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/gpt2/model.safetensors"
        ));
        // the weights are gitignored, so a fresh clone has none
        if !path.exists() {
            return;
        }
        let tensors = load(path);
        assert_eq!(tensors.len(), 160);

        // Expected values come from the Python `safetensors` library, an
        // independent reader of the same file.
        let (shape, wte) = &tensors["wte.weight"];
        assert_eq!(shape, &[50257, 768]);
        assert_eq!(
            wte[..4],
            [-0.11010301_f32, -0.039266724, 0.033107508, 0.13382645]
        );
        assert_eq!(wte.last(), Some(&0.120678246));
    }

    // The expected values are exact bfloat16 numbers, so they print with many digits.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn loads_bf16_qwen() {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/qwen2.5-coder-0.5b/model.safetensors"
        ));
        if !path.exists() {
            return;
        }
        let tensors = load(path);
        // Expected values from the Python `safetensors` library. bfloat16 values
        // are exactly representable in f32, so equality is exact.
        let (shape, embed) = &tensors["model.embed_tokens.weight"];
        assert_eq!(shape, &[151936, 896]);
        assert_eq!(
            embed[..4],
            [
                -0.0166015625,
                0.025146484375,
                0.0230712890625,
                0.01226806640625
            ]
        );
        assert_eq!(embed.last(), Some(&0.0205078125));
        let (_, bias) = &tensors["model.layers.0.self_attn.q_proj.bias"];
        assert_eq!(
            bias[..3],
            [-0.0299072265625, -0.0079345703125, -0.08544921875]
        );
    }
}
