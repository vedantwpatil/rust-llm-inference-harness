//! What the examples share: where checkpoints live and how to load one.
// Each example uses some of these.
#![allow(dead_code)]

use inference::backend::Backend;
use inference::gpt2::Gpt2;
use inference::llama::Llama;
use inference::model::Model;
use inference::safetensors::load;
use std::path::PathBuf;
use tokenizer::Tokenizer;

pub const QWEN: &str = "qwen2.5-coder-0.5b";
pub const SYSTEM: &str = "You are a helpful coding assistant.";

/// `data/<name>/`, which `scripts/fetch_model.py` fills.
pub fn data_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name)
}

pub fn tokenizer(name: &str) -> Tokenizer {
    if name == "gpt2" {
        return Tokenizer::from_gpt2_merges(include_str!(
            "../../../tokenizer/data/gpt2/merges.txt"
        ));
    }
    let json = std::fs::read_to_string(data_dir(name).join("tokenizer.json")).unwrap();
    Tokenizer::from_tokenizer_json(&json)
}

/// GPT-2 or a Llama-style checkpoint by directory name, optionally int8.
pub fn load_model<B: Backend + 'static>(name: &str, int8: bool, backend: B) -> Box<dyn Model> {
    if name == "gpt2" {
        assert!(!int8, "int8 is implemented for the Llama-style model only");
        let tensors = load(&data_dir(name).join("model.safetensors"));
        return Box::new(Gpt2::from_tensors(tensors, backend));
    }
    let mut model = Llama::from_dir(&data_dir(name), backend);
    if int8 {
        model.quantize();
    }
    Box::new(model)
}
