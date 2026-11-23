use crate::byte_level::{build_char_to_byte, bytes_to_unicode, prints_cleanly};
use crate::encode::{build_ranks, build_vocab_ids, encode_chunk};
use crate::pretokenize::{GPT2_PATTERN, pre_tokenize};
use crate::train::tokenize_bpe;
use fancy_regex::Regex;
use std::{cmp::Reverse, collections::HashMap, string::FromUtf8Error};

pub const ENDOFTEXT: &str = "<|endoftext|>";

// Everything encode and decode need, built once. An id is an index into
// `vocab`, so the layout is whatever the constructor chose: training puts
// <|endoftext|> first, the GPT-2 loader matches GPT-2's ids.
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub(crate) ranks: HashMap<(String, String), usize>,
    pub(crate) vocab_ids: HashMap<String, u32>,
    pub(crate) byte_symbols: [char; 256],
    pub(crate) char_to_byte: HashMap<char, u8>,
    pub(crate) pattern: Regex,
    // shortcut: linear scan per lookup, fine for a handful of special tokens
    pub(crate) specials: Vec<(String, u32)>,
}

impl Tokenizer {
    pub fn new(
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
        pattern: &str,
        specials: Vec<(String, u32)>,
    ) -> Self {
        // An empty special would match at every position and never advance.
        assert!(specials.iter().all(|(text, _)| !text.is_empty()));
        let byte_symbols = bytes_to_unicode();
        Self {
            ranks: build_ranks(&merges),
            vocab_ids: build_vocab_ids(&vocab),
            char_to_byte: build_char_to_byte(&byte_symbols),
            pattern: Regex::new(pattern).unwrap(),
            byte_symbols,
            vocab,
            merges,
            specials,
        }
    }

    pub fn train(corpus: &[&str], vocab_size: usize) -> Tokenizer {
        let re = Regex::new(GPT2_PATTERN).unwrap();
        let (vocab, merges) = tokenize_bpe(corpus, vocab_size, &re);
        Self::new(
            vocab,
            merges,
            GPT2_PATTERN,
            vec![(ENDOFTEXT.to_string(), 0)],
        )
    }

    pub fn from_gpt2_merges(merges_txt: &str) -> Self {
        let byte_symbols = bytes_to_unicode();
        let mut vocab: Vec<String> = (0..=u8::MAX)
            .filter(|&b| prints_cleanly(b))
            .chain((0..=u8::MAX).filter(|&b| !prints_cleanly(b)))
            .map(|b| byte_symbols[usize::from(b)].to_string())
            .collect();

        let merges: Vec<(String, String)> = merges_txt
            .lines()
            .skip(1)
            .filter(|line| !line.is_empty())
            .map(|line| {
                let (a, b) = line.split_once(' ').unwrap();
                (a.to_string(), b.to_string())
            })
            .collect();

        vocab.extend(merges.iter().map(|(a, b)| format!("{a}{b}")));
        let endoftext_id = vocab.len();
        vocab.push(ENDOFTEXT.to_string());
        Self::new(
            vocab,
            merges,
            GPT2_PATTERN,
            vec![(ENDOFTEXT.to_string(), u32::try_from(endoftext_id).unwrap())],
        )
    }

    /// Builds a tokenizer from a Hugging Face `tokenizer.json` with a byte-level
    /// BPE model (Llama 3, Qwen): the vocabulary, merges, split pattern and
    /// added tokens are all read as data. Every added token is a special here,
    /// as in Hugging Face, which cuts them out of raw text whether or not they
    /// are flagged special.
    // shortcut: the file's normalizer is ignored. Qwen's is NFC, so text with
    // decomposed accents encodes differently from Hugging Face; needs Unicode
    // normalization tables.
    pub fn from_tokenizer_json(json: &str) -> Self {
        let doc: serde_json::Value = serde_json::from_str(json).unwrap();
        let model = &doc["model"];
        assert_eq!(model["type"], "BPE");

        let specials: Vec<(String, u32)> = doc["added_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| {
                let id = u32::try_from(t["id"].as_u64().unwrap()).unwrap();
                (t["content"].as_str().unwrap().to_string(), id)
            })
            .collect();
        let entries: Vec<(&str, usize)> = model["vocab"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(token, id)| (token.as_str(), id.as_u64().unwrap() as usize))
            .chain(specials.iter().map(|(t, id)| (t.as_str(), *id as usize)))
            .collect();
        let mut vocab = vec![String::new(); entries.iter().map(|&(_, id)| id + 1).max().unwrap()];
        for (token, id) in entries {
            vocab[id] = token.to_string();
        }
        // A hole would map the empty string to an id and decode to nothing.
        assert!(
            vocab.iter().all(|t| !t.is_empty()),
            "ids are not contiguous"
        );

        let merges = model["merges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| match m.as_str() {
                Some(pair) => pair.split_once(' ').unwrap(),
                None => (m[0].as_str().unwrap(), m[1].as_str().unwrap()),
            })
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();

        // A plain ByteLevel pre-tokenizer (GPT-2's file) has no Split to read.
        let pattern = split_pattern(&doc["pre_tokenizer"]).unwrap_or(GPT2_PATTERN);
        Self::new(vocab, merges, pattern, specials)
    }

    // Special tokens are cut out of the text first and emitted as their own id,
    // so they are never pre-tokenized or merged. Text between them is encoded
    // normally.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        // Earliest special in `rest`, the longest one if several start together.
        while let Some((at, len, id)) = self
            .specials
            .iter()
            .filter_map(|(special, id)| {
                rest.find(special.as_str())
                    .map(|at| (at, special.len(), *id))
            })
            .min_by_key(|&(at, len, _)| (at, Reverse(len)))
        {
            self.encode_plain(&rest[..at], &mut ids);
            ids.push(id);
            rest = &rest[at + len..];
        }
        self.encode_plain(rest, &mut ids);
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        for chunk in pre_tokenize(&self.pattern, text) {
            for piece in encode_chunk(chunk, &self.ranks, &self.byte_symbols) {
                ids.push(self.vocab_ids[&piece]);
            }
        }
    }

    // Bytes are collected across all tokens and turned into a String once at
    // the end: one token can hold half of a multi-byte character.
    pub fn decode(&self, ids: &[u32]) -> Result<String, FromUtf8Error> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            // A special's text is plain UTF-8, not byte symbols.
            self.token_bytes(id, &mut bytes);
        }
        String::from_utf8(bytes)
    }

    pub fn decode_stream(
        &self,
        id: u32,
        pending: &mut Vec<u8>,
    ) -> Result<String, std::str::Utf8Error> {
        self.token_bytes(id, pending);
        match std::str::from_utf8(pending) {
            Ok(s) => {
                let text = s.to_owned();
                pending.clear();
                Ok(text)
            }
            // `None` means the bytes end mid-character, not that they are invalid.
            Err(e) if e.error_len().is_none() => {
                let done: Vec<u8> = pending.drain(..e.valid_up_to()).collect();
                Ok(String::from_utf8(done).unwrap())
            }
            Err(e) => Err(e),
        }
    }

    fn token_bytes(&self, id: u32, out: &mut Vec<u8>) {
        if let Some((special, _)) = self
            .specials
            .iter()
            .find(|(_, special_id)| *special_id == id)
        {
            out.extend_from_slice(special.as_bytes());
            return;
        }
        for chars in self.vocab[id as usize].chars() {
            out.push(self.char_to_byte[&chars]);
        }
    }
}

/// The regex of the first `Split` pre-tokenizer, looking inside a `Sequence`.
fn split_pattern(pre: &serde_json::Value) -> Option<&str> {
    match pre["type"].as_str()? {
        "Split" => pre["pattern"]["Regex"].as_str(),
        "Sequence" => pre["pretokenizers"]
            .as_array()?
            .iter()
            .find_map(split_pattern),
        _ => None,
    }
}
