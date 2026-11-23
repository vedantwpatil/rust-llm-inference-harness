use serde_json::Value;

/// Llama 3.x's long-context rope frequency scaling (`rope_type: "llama3"`).
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_positions: f64,
}

/// The hyperparameters of a Llama-style decoder (Llama, Qwen2), read from the
/// checkpoint's `config.json` instead of assumed.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    /// The output projection reuses the token embedding table.
    pub tie_embeddings: bool,
    pub max_positions: usize,
    /// Token ids that end a reply.
    pub eos: Vec<u32>,
}

impl LlamaConfig {
    pub fn from_json(json: &str) -> Self {
        let c: Value = serde_json::from_str(json).unwrap();
        let int = |key: &str| {
            c[key]
                .as_u64()
                .unwrap_or_else(|| panic!("config.json has no integer {key}")) as usize
        };
        let (hidden, n_heads) = (int("hidden_size"), int("num_attention_heads"));
        let rope_scaling = match &c["rope_scaling"] {
            Value::Null => None,
            r => match r["rope_type"].as_str().or(r["type"].as_str()) {
                Some("llama3") => Some(RopeScaling {
                    factor: r["factor"].as_f64().unwrap(),
                    low_freq_factor: r["low_freq_factor"].as_f64().unwrap(),
                    high_freq_factor: r["high_freq_factor"].as_f64().unwrap(),
                    original_max_positions: r["original_max_position_embeddings"].as_f64().unwrap(),
                }),
                other => panic!("rope_scaling type {other:?} is not supported"),
            },
        };
        let eos = match &c["eos_token_id"] {
            Value::Array(ids) => ids.iter().map(|i| i.as_u64().unwrap() as u32).collect(),
            id => vec![id.as_u64().unwrap() as u32],
        };
        Self {
            hidden,
            n_layers: int("num_hidden_layers"),
            n_heads,
            n_kv_heads: int("num_key_value_heads"),
            // Some checkpoints set it apart from hidden / heads.
            head_dim: c["head_dim"]
                .as_u64()
                .map_or(hidden / n_heads, |d| d as usize),
            intermediate: int("intermediate_size"),
            rms_eps: c["rms_norm_eps"].as_f64().unwrap() as f32,
            rope_theta: c["rope_theta"].as_f64().unwrap(),
            rope_scaling,
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),
            max_positions: int("max_position_embeddings"),
            eos,
        }
    }

    /// `head_dim / 2` rotation frequencies: `theta^(-2j / head_dim)`, then
    /// Llama 3's rescaling for long wavelengths when the config has one.
    pub fn inv_freq(&self) -> Vec<f32> {
        let d = self.head_dim as f64;
        (0..self.head_dim / 2)
            .map(|j| {
                let freq = self.rope_theta.powf(-(2.0 * j as f64) / d);
                self.rope_scaling.as_ref().map_or(freq, |s| s.apply(freq)) as f32
            })
            .collect()
    }
}

impl RopeScaling {
    /// Short wavelengths (high frequency) are kept, long ones are stretched by
    /// `factor`, and the band between blends the two. The blend starts from the
    /// unscaled frequency, as Hugging Face's `_compute_llama3_parameters` does.
    fn apply(&self, freq: f64) -> f64 {
        let wavelen = 2.0 * std::f64::consts::PI / freq;
        let low_wavelen = self.original_max_positions / self.low_freq_factor;
        let high_wavelen = self.original_max_positions / self.high_freq_factor;
        if wavelen < high_wavelen {
            freq
        } else if wavelen > low_wavelen {
            freq / self.factor
        } else {
            let smooth = (self.original_max_positions / wavelen - self.low_freq_factor)
                / (self.high_freq_factor - self.low_freq_factor);
            (1.0 - smooth) * freq / self.factor + smooth * freq
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = r#"{"hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14,
        "num_key_value_heads": 2, "intermediate_size": 4864, "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0, "tie_word_embeddings": true, "max_position_embeddings": 32768,
        "eos_token_id": 151645}"#;

    #[test]
    fn parses_qwen_config() {
        let c = LlamaConfig::from_json(QWEN);
        assert_eq!(
            (c.hidden, c.n_layers, c.n_heads, c.n_kv_heads),
            (896, 24, 14, 2)
        );
        assert_eq!(c.head_dim, 64);
        assert!(c.tie_embeddings && c.rope_scaling.is_none());
        assert_eq!(c.eos, [151645]);
    }

    // Expected values come from Hugging Face's `ROPE_INIT_FUNCTIONS`, computed
    // for Llama 3.2 1B's settings (theta 500000, factor 32, low 1, high 4,
    // original context 8192) and for Qwen's plain theta 1e6. Indices 0..15 are
    // high frequencies (unchanged), 15 to 17 the blended band, 20 and up divided.
    #[test]
    fn inv_freq_matches_hugging_face() {
        let llama3 = LlamaConfig {
            head_dim: 64,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_positions: 8192.0,
            }),
            ..LlamaConfig::from_json(QWEN)
        };
        let got = llama3.inv_freq();
        for (j, want) in [
            (0, 1.0),
            (1, 0.663601279258728),
            (10, 0.016560440883040428),
            (15, 0.0012905480107292533),
            (16, 0.000429556705057621),
            (17, 9.708286233944818e-05),
            (20, 8.570255886297673e-06),
            (31, 9.418306490260875e-08),
        ] {
            assert!(
                (got[j] as f64 / want - 1.0).abs() < 1e-6,
                "j = {j}: {} vs {want}",
                got[j]
            );
        }
        let qwen = LlamaConfig::from_json(QWEN).inv_freq();
        for (j, want) in [
            (1, 0.6493816375732422),
            (16, 0.0010000000474974513),
            (31, 1.5399265294036013e-06),
        ] {
            assert!((qwen[j] as f64 / want - 1.0).abs() < 1e-6, "j = {j}");
        }
    }
}
