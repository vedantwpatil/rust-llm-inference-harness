use std::{collections::HashMap, path::Path};

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

        assert_eq!(entry["dtype"].as_str().unwrap(), "F32");
        assert_eq!(end - begin, shape.iter().product::<usize>() * 4);

        let values: Vec<f32> = data[begin..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
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
}
