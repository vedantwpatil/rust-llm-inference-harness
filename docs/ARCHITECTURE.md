# Architecture

How the code is structured today. Why it is built this way: [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md). Planned crates and stages: [ROADMAP.md](ROADMAP.md#crate-map).

Cargo workspace, one crate per layer:

- `crates/tokenizer`: the library.
- `crates/tokenizer-cli`: demo binary (`main.rs`). `take_input` reads the prompt from a positional arg (no `clap`). Prints compression for word, char, byte, trained BPE and GPT-2 BPE.
- `crates/inference`: stage 2. `safetensors::load` reads GPT-2's weights into `HashMap<String, (Vec<usize>, Vec<f32>)>`: tensor name to (shape, flat row-major values).

## Tokenizer modules

A tokenizer is normalizer, then pre-tokenizer, then model. This one has no normalizer: on a fixed corpus the text can be fixed up front, though a general dataset would want one.

| Module        | Holds                                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline`    | `tokenize_by_char`, `tokenize_by_word`, `tokenize_by_byte`: naive tokenizers the BPE one is compared against                                            |
| `pretokenize` | `pre_tokenize` and `GPT2_PATTERN`: GPT-2's regex split via `fancy-regex`. Merges never cross a chunk                                                     |
| `byte_level`  | `bytes_to_unicode`: GPT-2's byte to printable-char table, `[char; 256]` indexed by byte                                                                  |
| `train`       | `tokenize_bpe` (the trainer), `compute_pair_freqs`, `merge_pairs`                                                                                       |
| `encode`      | `encode_chunk`: replays merges on one chunk, lowest rank first. `build_ranks`, `build_vocab_ids`                                                         |
| `tokenizer`   | `Tokenizer { vocab, merges, ranks, vocab_ids, pattern, specials, .. }`: `train`, `from_gpt2_merges`, `encode`, `decode`, `decode_stream`                 |
| `metrics`     | `bytes_per_token`                                                                                                                                       |

`encode` cuts special tokens out first, then pre-tokenizes and replays merges. `decode` joins bytes across tokens and builds the `String` once at the end. `decode_stream(id, &mut pending)` is the one-id-at-a-time version: the caller owns `pending`, an incomplete UTF-8 tail stays in it. Both share `token_bytes`.

## Data flow

Training:

```text
corpus: Vec<&str>
   │  pre_tokenize (GPT-2 regex, runs on the raw text)
   ▼
word_freqs: HashMap<&str, u32>        chunk -> occurrences in corpus
   │  chunk.bytes() -> bytes_to_unicode table
   ▼
splits: HashMap<&str, Vec<String>>    chunk -> its current symbols
   │
   │  loop until vocab.len() == vocab_size or no pair is left
   │    compute_pair_freqs(&splits, &word_freqs) -> HashMap<(String, String), u32>
   │    best pair = max by (count, then smallest pair)
   │    merge_pairs(a, b, &mut splits)            rewrites every chunk in place
   │    merges.push(best_pair); vocab.push(a + b)
   ▼
(vocab: Vec<String>, merges: Vec<(String, String)>)
```

Encode and decode: pre-split, bytes to unicode, merge by rank; and back through ids to bytes and a UTF-8 hold buffer. Drawn in [ROADMAP.md](ROADMAP.md#stage-1-tokenizer-byte-level-bpe).

## Output contract

`tokenize_bpe(corpus: &[&str], vocab_size: usize)` returns `(vocab, merges)`.

| Index                               | Content                                                     |
| ----------------------------------- | ----------------------------------------------------------- |
| `vocab[0]`                          | `<\|endoftext\|>`, special token, never produced by a merge |
| `vocab[1 + b]` for `b` in `0..=255` | the symbol for byte `b`, which is `bytes_to_unicode()[b]`   |
| `vocab[257 + i]`                    | `merges[i].0 + merges[i].1`, the token merge `i` created    |

- A token's id is its index in `vocab`, so each constructor picks its own id layout. `Tokenizer::from_gpt2_merges` builds GPT-2's (188 printable bytes first, then the 68 shifted ones, then merges, `<|endoftext|>` at 50256), so its ids match tiktoken's. A trained tokenizer keeps the layout above and its ids do not match GPT-2's.
- A merge's rank is its index in `merges`, the order merges were learned. The encoder replays them in that order, so the order is part of the trained artifact. Hence `Vec`, not `HashMap`: training never looks a merge up, and a map would lose the order. The encoder builds `pair -> rank` and `token -> id` maps from the two vectors at load.
- `vocab[257 + i] == format!("{a}{b}")` for `merges[i] == (a, b)`.
- Starting size is 257, so `TOY_VOCAB_SIZE = 280` learns 23 merges.
