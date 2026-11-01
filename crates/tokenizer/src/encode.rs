use crate::byte_level::bytes_to_symbols;
use std::collections::HashMap;

// Created structs to make it easier and more maintainable then remembering the specific position of
// data
struct Index {
    i: usize,
}

struct Rank {
    r: usize,
}

// TODO: Optimize this, currently very slow and will show up in the future
pub fn encode_chunk(
    chunk: &str,
    ranks: &HashMap<(String, String), usize>,
    byte_symbols: &[char; 256],
) -> Vec<String> {
    let mut pieces = bytes_to_symbols(byte_symbols, chunk);
    // One key buffer for every lookup: clone_from reuses its allocation.
    let mut key: (String, String) = (String::new(), String::new());
    loop {
        // index, rank
        let mut best: Option<(usize, usize)> = None;

        for (index, rank) in pieces.windows(2).enumerate() {
            key.0.clone_from(&rank[0]);
            key.1.clone_from(&rank[1]);
            if let Some(&rank) = ranks.get(&key)
                && best.is_none_or(|(_, best_rank)| rank < best_rank)
            {
                best = Some((index, rank));
            }
        }

        match best {
            Some((i, _)) => {
                let first = &pieces[i];
                let second = &pieces[i + 1];
                pieces[i] = format!("{first}{second}");
                pieces.remove(i + 1);
            }
            None => break,
        }
    }
    pieces
}

pub fn build_ranks(merges: &[(String, String)]) -> HashMap<(String, String), usize> {
    merges
        .iter()
        .enumerate()
        .map(|(rank, pair)| (pair.clone(), rank))
        .collect()
}

pub fn build_vocab_ids(vocab: &[String]) -> HashMap<String, usize> {
    vocab
        .iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), id))
        .collect()
}
