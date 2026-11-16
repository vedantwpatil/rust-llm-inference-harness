# Development Notes

Why the tokenizer is built the way it is: reasoning, cost, measured baselines. Structure and data flow: [ARCHITECTURE.md](ARCHITECTURE.md). Plan and open decisions: [ROADMAP.md](ROADMAP.md). Deep dives: [CHARS_VS_BYTES.md](CHARS_VS_BYTES.md), [I32_VS_U32.md](I32_VS_U32.md). Rust mechanics: [RUST_NOTES.md](RUST_NOTES.md).

## Stage 1: byte-level BPE

### Pre-tokenizer

`GPT2_PATTERN` is GPT-2's split regex: contractions, optional-space-prefixed letter, digit and punctuation runs, then whitespace. The trailing `\s+(?!\S)` is a negative lookahead, which the `regex` crate cannot express, hence `fancy-regex`. `pre_tokenize<'a>(re: &Regex, text: &'a str) -> Vec<&'a str>` returns slices of `text`. The explicit `'a` is needed because two reference parameters defeat lifetime elision, see [RUST_NOTES.md](RUST_NOTES.md#lifetimes).

Consequences for training:

- A leading space stays attached to its word (`" the"`), so spaces are part of learned tokens. Hence `Ġt`.
- Pairs are counted with `windows(2)` inside one chunk, so no merge spans two chunks.

### Bytes to unicode

`bytes_to_unicode() -> [char; 256]` is GPT-2's table, indexed by byte.

- 188 bytes print cleanly and map to themselves: `0x21..=0x7E` (`!` to `~`), `0xA1..=0xAC`, `0xAE..=0xFF`.
- The other 68 (control characters, space, `0x7F..=0xA0`, `0xAD`) get `U+0100`, `U+0101`, ... in byte order, ending at `U+0143`: `0x00` is `Ā` (`U+0100`), newline `Ċ` (`U+010A`), space `Ġ` (`U+0120`), `0xAD` `Ń` (`U+0143`).
- The `shifted` counter advancing in byte order makes the table identical to GPT-2's. Any other order is a valid bijection but would not match GPT-2's `vocab.json` and `merges.txt`.

Why bytes instead of characters, and why this table: [CHARS_VS_BYTES.md](CHARS_VS_BYTES.md).

All 256 entries were compared once against a Python port of GPT-2's `bytes_to_unicode`. The tests check five known entries plus bijectivity and printability, and the tiktoken oracle test covers the rest.

### Training loop

State: `word_freqs` (fixed), `splits` (rewritten each merge), `vocab`, `merges`. Each iteration:

1. **Count.** `compute_pair_freqs` adds each chunk's frequency to every adjacent pair of its current symbols. A pair's count is its occurrences in the whole corpus, not the number of chunks containing it.
2. **Stop check.** Empty `pair_freqs` means every chunk is one symbol: break. Without it the loop picks nothing and never grows `vocab`.
3. **Select.** One pass keeps the highest count, ties to the smallest `(String, String)`.
4. **Merge.** `merge_pairs` rewrites every chunk's symbols in place.
5. **Record.** Push the pair onto `merges`, the concatenation onto `vocab`.

`while vocab.len() < vocab_size` stops at the target size.

**Selection is order-independent.** `HashMap` has a per-instance random seed, so iteration order differs between maps and between runs, and "first pair seen with the max count" picks different merges per run. Max under `(count, smaller pair)` is a total order, so the winner does not depend on iteration order. `training_is_deterministic` trains twice and compares. HF's Python course is deterministic only because dicts keep insertion order, so its tie-break differs: on the toy corpus `("e", "r")`, `("i", "s")` and `("Ġ", "a")` all occur 5 times. This repo takes them in pair byte order (`er`, `is`, `Ġa`), the course takes `is` before `er`. Only the first merge, `("Ġ", "t")` with 7 occurrences (` the`, ` tokenization`, ` tokenizer`, ` to`, ` they`, ` trained`, ` tokens`), has no tie. The difference affects trained artifacts only, not encoding with published GPT-2 merges.

**Merge semantics.** `merge_pairs` scans each chunk left to right. On a match at `i` it writes the concatenation to `split[i]`, removes `split[i + 1]`, and does not advance `i`, so the new symbol is compared with its right neighbour next. Matches never overlap: `["a","a","a"]` under `("a","a")` becomes `["aa","a"]`, four `a`s become `["aa","aa"]`. The encoder must use the same left-to-right greedy order, or trained and encode-time merges disagree.

### Cost

Every iteration recounts all pairs from scratch.

- `compute_pair_freqs` is linear in total symbols across unique chunks, but clones each chunk's `Vec<String>` and both strings of every window, so it is allocation heavy.
- `merge_pairs` is linear in the same total, quadratic in chunk length worst case (`Vec::remove` shifts the tail).
- Total is about `merges × symbols`. Fine for the toy corpus, wrong shape for a real one. Standard fix: incremental pair counts, updating only chunks a merge touched, with a heap for the argmax. Not done: correctness comes before speed here.
- Pair counts are `u32`: type choice and overflow behavior in [I32_VS_U32.md](I32_VS_U32.md).

### Decoder

`Tokenizer::token_bytes(id, out: &mut Vec<u8>)` appends one id's bytes to `out`. A special's text is plain UTF-8, so it is looked up first (`specials`, linear scan) and copied whole. Any other id's `vocab` entry is symbol chars, each mapped back to a byte through `char_to_byte`. It appends instead of returning a `Vec` so no allocation happens per token.

- **`decode(&[u32]) -> Result<String, FromUtf8Error>`** calls `token_bytes` for every id into one `Vec<u8>` and runs `String::from_utf8` once at the end. A merged token can end mid-character, so converting per token would fail on valid input. A cut character at the very end is an `Err`.
- **`decode_stream(id, &mut pending) -> Result<String, Utf8Error>`** appends the id's bytes to `pending`, then `std::str::from_utf8(pending)`:
  - `Ok(s)`: everything is complete. Return it, clear `pending`.
  - `Err(e)` with `e.error_len() == None`: the bytes end mid-character. Drain the first `e.valid_up_to()` bytes out of `pending` and return them as text, the tail stays for the next call.
  - `Err(e)` otherwise: invalid bytes, return the error. `pending` is left as is, so the caller clears it.
- **State is the caller's.** `pending` is a parameter, not a field, so `decode_stream` takes `&self` and one `Tokenizer` can serve several streams. A stage 4 `Session` will own its own buffer.
- **End of stream.** A non-empty `pending` is a cut character. There is no flush method: the caller decides whether that is an error.
- **Error type.** `decode_stream` returns `Utf8Error` (from `str::from_utf8` on a slice), `decode` returns `FromUtf8Error` (from `String::from_utf8`, which owns the `Vec`).

### Compression metrics

`main` prints tokens and bytes per token for word, char, byte, trained BPE and GPT-2 BPE. The word row has an unbounded vocab, so it is not comparable with the others. Rows state vocab size and training data: 280 entries from four sentences against GPT-2's 50,257 is a data and vocab gap, not an algorithm gap.

### Benchmarks

`bench_encode_mixed_text` and `bench_encode_one_long_chunk` in `crates/tokenizer/src/tests.rs` are `#[ignore]`d and each checks its token count and id hash against tiktoken. Baseline with the `String`-based `encode_chunk`:

- 60 KB mixed text: 10.4 ms (5.75 MB/s, 2.1M tokens/s).
- One 4000-byte chunk: 137.9 ms (0.03 MB/s), about 200x slower per byte, because cost grows with chunk length squared.
