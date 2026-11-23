# Development Notes

Why the code is built the way it is: reasoning, cost, measured baselines. Structure and data flow: [ARCHITECTURE.md](ARCHITECTURE.md). Plan and open decisions: [ROADMAP.md](ROADMAP.md). Deep dives: [CHARS_VS_BYTES.md](CHARS_VS_BYTES.md), [I32_VS_U32.md](I32_VS_U32.md). Rust mechanics: [RUST_NOTES.md](RUST_NOTES.md).

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

## Stage 2: GPT-2 on the CPU

### Verification

The bar is Hugging Face's float32 forward pass on the same ids, compared at every layer boundary, plus greedy `generate` and cached-versus-recomputed decode. Per-element relative error does not work on this model: the residual stream has entries near 3000 next to entries near 0, and float32 summation order moves every entry by a few units in the seventh digit of the largest one. So the check is `max|got - want| <= rel * max|want|`, per tensor. The layer test uses `rel = 1e-4` and the worst observed error is about `1.4e-5`; cached decode against a from-scratch pass uses `1e-5`. Embedding output matches exactly and the error does not compound across blocks: it saturates.

### Why `Matrix` is `[out, in]`

`y = x @ W^T` then reads one weight row and one activation row, both contiguous, per output: a dot product the kernel can stream. Llama's `nn.Linear` is already `[out, in]`, so the Llama-style model needs no second layout, and int8 scales attach per output row. GPT-2's `Conv1D` is `[in, out]` and pays one transpose at load.

### Speed

Decode reads every weight once per token and does about two operations per byte, so it is limited by memory bandwidth, not arithmetic. Roofline: `tokens/s = bandwidth / bytes read per token`. For GPT-2 that is 494 MB of f32 weights (the tied token embedding counts once, as the output projection) plus the KV cache, about 10 MB at 100 tokens of context.

**The bandwidth a CPU can reach is not the spec sheet.** On an M3 Max (10 performance plus 4 efficiency cores) the specification says 300 GB/s but a streaming read through the CPU cores saturates near 110 to 127 GB/s, and four threads already get most of it. `bench` measures it each run: best of 20 passes over 1 GiB with the model's own `dot`. An average hides cold or contended passes, and bandwidth is a ceiling. Against the spec number the roofline would be about 600 tokens/s and unreachable.

**Why not rayon.** The first version used rayon's `par_chunks_mut`. Decode is about 60 tiny parallel regions per token (four matmuls and one attention per block, plus the output projection), and an empty rayon region measured 16 to 19 µs at 4 threads, about 45 µs at 8 and over 100 µs at 14 (workers asleep). That is a large share of a 5 ms token, and more threads made it worse: the best decode was 160 tokens/s at 4 to 5 threads, 121 at the default. `Pool` replaces it: workers spin about 200 µs after a region, then sleep, so back-to-back regions find them awake. The cost is that idle workers burn CPU for that window.

**Thread counts.** The pool has one thread per performance core (10 on the M3 Max), from `sysctl`, and `INFERENCE_THREADS` overrides it. A region ends when its slowest task does, so an efficiency core that claims a task late stalls everyone: with all 14 cores decode is 17% slower than with 8. A decoded token is bound by memory, which 8 threads already saturate, and with 10 it is 5% slower, so decode regions run on 8 (`DECODE_THREADS`). A prompt is bound by arithmetic and prefills 13% faster on all 10. `Pool::run_width` limits a region to its first `width` threads. The others have to sleep through a run of narrow regions: left to spin they cost decode 2.5%, from two idle cores that spin flat out for the whole token. These numbers are one machine, not a rule.

**The kernel.** `tile<R, T, U>` multiplies `R` weight rows by `T` activation rows with `U` NEON accumulators per output. A fused multiply-add has a latency of a few cycles but the core starts four per cycle, so one accumulator chain leaves the units idle; `U` independent chains fill them. Register budget: `R * T * U` accumulators out of 32 vector registers. Three shapes are used:

| Shape   | Where                         | Why                                                                                                   |
| ------- | ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| `4,4,1` | prefill, 4 tokens at a time   | Each loaded weight is reused across 4 tokens and each activation across 4 rows: compute-bound          |
| `4,1,4` | decode, leftover tokens       | One token, so there is no reuse to tile for; 4 rows with 4 accumulators keep the loads streaming      |
| `1,1,4` | `dot`, leftover weight rows   | The plain dot product, also used by attention                                                          |

On one core with data in cache `4,4,1` reaches about 113 GFLOP/s (the core's ceiling is roughly 128) and `4,1,4` about 64.

**Where a token goes.** `bench` wraps the backend in a timing `Backend` and prints milliseconds per op. That is what found two prefill costs a guess at the matmul would have missed: scalar `gelu` was about a third of prefill, and the serial `[n_out][t]` to `[t][n_out]` transpose in `matmul` about a third of what was left. Both now run on the pool.

**Results**, 103-token prompt, median run, 8 threads except the last row. Roofline from the bandwidth measured in the same run.

| Version                                          | Prefill (tokens/s) | Decode (tokens/s) | Decode share of roofline |
| ------------------------------------------------ | ------------------ | ----------------- | ------------------------ |
| rayon, 8-accumulator `dot`                       | 705                | 121               | about 50%                |
| spinning pool, tiled NEON kernels                | about 1200         | about 177         | 75%                      |
| plus parallel `gelu` and transpose               | about 2600         | about 185 to 193  | about 75%                |
| plus the [performance-ceiling work](#performance-ceiling), prefill on 10 threads | about 3800 | about 215 | about 85% |

Prefill is about 440 GFLOP/s (17.5 GFLOP for the prompt in about 40 ms). Decode matmul runs at about 107 GB/s of the roughly 120 GB/s probe, so what is left of the gap to the roofline is spread thin: `gelu` and attention take about 0.3 ms each of 5.3 ms, and each parallel region still pays a small wake-up and join cost.

**Not done.** The residual adds are serial. Every `Backend` call allocates a fresh `Vec` for its output.

## Stage 3: a Llama-style model and int8

The checkpoint is Qwen2.5-Coder-0.5B-Instruct: 24 layers, width 896, 14 query heads sharing 2 key/value heads, head dimension 64, feed-forward width 4864, vocabulary 151,936, tied output projection, BF16 on disk. It is a coder model (stage 5 needs one) and small enough to iterate on. Nothing in the code is Qwen specific: sizes, rope base and scaling, tied embeddings and projection biases are read from `config.json` and the tensor list, and a randomly initialised Llama with an untied projection and Llama 3 rope scaling passes the same checks.

### Verification

Same method as stage 2 (every layer boundary, greedy `generate`, cached against recomputed decode), run by the same three functions on both models. Float32 results: every layer within about `1.1e-5` of the tensor's scale (bar `1e-4`), greedy ids equal, cached decode within `1e-5`. BF16 widens to float32 by a 16-bit shift, exact, so Rust and PyTorch read the same numbers.

Two things the Qwen checkpoint cannot show, so they have their own tests:

- **Llama 3 rope scaling.** Qwen has none. `tiny-llama3` is built so the three bands (kept, blended, divided) each hold at least one frequency and the prompt runs past the original context. The test fails with the scaling switched off, so it does check it. The frequencies themselves are also compared with values computed by Hugging Face.
- **Rotate-half pairing.** Element `j` pairs with `j + head_dim / 2`, not `2j` with `2j + 1`. The wrong pairing gives plausible text, so a layer-by-layer match is the only protection; block 0's attention output is the first place it would show.

The chat prompt is checked as ids: `chatml` produces the ids Hugging Face's `apply_chat_template` produces. The Qwen tokenizer built from `tokenizer.json` matches Hugging Face's `tokenizers` on the shared oracle cases plus Qwen's own special tokens, in both directions. Cases NFC would change are left out of the oracle, see below.

### Fused projections

Query, key and value are one `[q_dim + 2 * kv_dim, hidden]` matrix and gate and up are one `[2 * intermediate, hidden]` matrix, stacked at load. A decode token then makes 4 matmul calls per layer instead of 7, and each call is a parallel region with a fixed wake-up and join cost ([Speed](#speed)). The price is a split of the result per token, which is a copy of a few kilobytes.

### Int8

Weights are symmetric int8 with one scale per output row (`scale = max|row| / 127`). A matmul against them rounds its activations to int8 as well, one scale per `Q8_BLOCK` (128) values of a token, so the inner loop is the integer dot product `sdot`: 16 multiply-adds per instruction, four times an f32 fused multiply-add, with no widening first. The row scale is applied once per output; the activation scales are the ones that matter for accuracy. A scale per whole token let one outlier channel flatten the rest and cost 2.5 points of top-1 agreement with float32, and a scale per block confines it.

**What it costs in accuracy.** `eval_int8` over 512 tokens of this repository's source: 97.1% of positions have the same top-1 token as float32 and perplexity rises from 12.41 to 12.50 (0.65%). Logits on the verification prompt are within `1.5e-2` of Hugging Face's, relative to the logits' scale. With the top token different at about 1 position in 35, greedy output eventually diverges from float32's; that is the expected shape of quantization error, which is why the `1e-4` bar does not apply.

**What it buys.** Weight bytes per token fall from 1976 MB to 496 MB, and decode from 55 to about 205 tokens/s, 3.7 times for 4 times fewer bytes. Prefill is compute-bound and also gains, from 1000 to about 2500 tokens/s, since `sdot` does four times the work per instruction ([Speed](#speed)).

**Packed layout.** A tile of four weight rows against one activation row reads four streams `k` bytes apart. One thread pulled that from memory at 29 to 68 GB/s depending on `k`, and the same bytes at 91 to 94 GB/s when the four rows of each block sit next to each other. So int8 weights are stored in groups of four rows, block by block (`packed_index`, `Matrix::q8_index`), and rows after the last whole group stay plain. The float32 layout is unchanged.

**What stays float32.** Norm weights and biases (tiny, and their error is not averaged away by a dot product), the activations between ops, and the KV cache. The token embedding table is quantized; the tied output projection reads the same matrix, and it is the largest tensor (136M of the 494M parameters).

### Tokenizer as data

`Tokenizer::from_tokenizer_json` takes the vocabulary, merges, pre-tokenizer regex and added tokens from the file, so the stage 1 encoder runs unchanged on a 151K vocabulary. All added tokens act as specials, as in Hugging Face. The `tokenizer.json` normalizer (Unicode NFC for Qwen) is not implemented: text that is not already NFC can encode to different ids than Hugging Face gives.

### Speed

Same roofline as stage 2: `bandwidth / (weight bytes + KV bytes)`. Prefill pushes 101 tokens through 358M matrix-multiplied parameters (the embedding table is only read for the last row): 72 Gop for the layers. Prefill tokens/s is lower than GPT-2's because each token costs 4 times the arithmetic.

| 10 performance cores, decode on 8        | Prefill (tokens/s) | Decode (tokens/s) | Decode share of its roofline |
| ---------------------------------------- | ------------------ | ----------------- | ---------------------------- |
| GPT-2, float32, 103 tokens               | about 3800         | about 215         | about 85%                    |
| Qwen2.5-Coder-0.5B, float32, 101 tokens  | about 1000         | about 56          | about 86%                    |
| Qwen2.5-Coder-0.5B, int8, 101 tokens     | about 2500         | about 205         | about 80%                    |

Comparison with llama.cpp and mlx-lm is not done: neither is installed here.

### Performance ceiling

Each number below is measured, not estimated: `ceilings` times register-only loops and memory streaming on this chip, `bench` times the model, and the gap between them is accounted for. The `bench` breakdown prints milliseconds per op and, for decode, GB/s per matmul shape.

| Limit (M3 Max)                       | Measured                                                                                           |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Memory read, 2 or more threads       | 125 to 128 GB/s (one thread: 83)                                                                   |
| float32 fused multiply-add           | 120 Gop/s per thread, 1140 on 10                                                                   |
| int8 `sdot`                          | 481 Gop/s per thread, 4570 on 10 (95% scaling). `smmla` measures the same, so it buys nothing      |
| One cold slice streamed per region   | 0.8 MB: 59 to 78 GB/s, 1 MB: 66 to 100, 4.4 MB: 103 to 117, 8.8 MB: 110 to 123 (4 to 10 threads) |
| Empty parallel region                | 0.3 µs on 4 threads, 2.8 on 8, 3.3 on 10                                                           |

**Decode, Qwen int8: 5.0 ms per token against 3.9 ms at the roofline.** The 1.1 ms gap is three parts.

- About 0.5 ms is the small matrices. A region streaming under about 4 MB does not reach full bandwidth however it is cut (the cold-slice rows above): the attention output projection is 0.8 MB and measures 57 GB/s, the fused query, key and value projection is 1.0 MB and measures 74. Summing every shape at the rate the probe gives for its size on 8 threads (70, 79, 115, 120 and 125 GB/s) puts the matmuls at 4.35 ms, against 3.9 ms at full bandwidth.
- About 0.1 ms is the matmuls over their own ceilings, 4.48 ms measured against 4.35: task tails and uneven task sizes. The large shapes are at their rate: 118 GB/s for the 8.8 MB feed-forward matrix and 124 for the 137 MB logits projection.
- About 0.5 ms is everything that is not a matmul: attention 0.30 (24 regions of 14 heads, about 12 µs each, 3 µs of it the region itself), then norms, `swiglu`, `rope` and the residual adds.

Float32 Qwen and GPT-2 decode are the same story at 86% and 85% of their roofline.

**Prefill, Qwen int8: 40 ms for 101 tokens.**

| Step                                                   | Time for the 72 Gop of layer matmuls |
| ------------------------------------------------------ | ------------------------------------ |
| `sdot` peak on 10 threads                              | 15.8 ms                              |
| the 4 by 4 tile on 10 threads, operands in L1          | 25.9 ms (280 Gop/s per thread)       |
| measured, `CpuBackend::matmul`                         | 32.6 ms (2220 Gop/s)                 |

- Peak to tile: a 4 by 4 tile does 128 `sdot` per block of 128 values, and with them 64 loads and about 36 other vector instructions: zeroing the 16 int32 sums, 12 pairwise adds that collapse each token's four row sums into one vector, 4 conversions and 4 multiply-adds that apply the block's scale. A loads-and-`sdot`-only tile reaches 386 Gop/s per thread, so the per-block scale (which is what the accuracy needs) costs a fifth to a quarter.
- Tile to measured: 87 to 89% on the large matrices (2430 to 2480 Gop/s) and 56 to 61% on the two small ones (1560 and 1720). A matmul is three regions (quantize the scales, quantize the values, multiply), 56 to 72 tasks of 16 rows on 10 threads leave a tail, and a 100 µs call has little room for either.
- The rest of the 40 ms: the logits projection 1.1 ms (memory-bound, 124 GB/s), attention 2.6, `swiglu` 1.6 (11.8M `exp` and divisions), `rmsnorm` 0.5, `rope` 0.7, copies and residual adds 1.1. Attention scales to about three quarters of linear on 8 threads. Claiming a task costs about 120 ns on the shared counter and 14 (query, head) pairs per task is the flat optimum, so the rest is per-item cost.

**What got it here.** A vectorized `exp`; parallel norms, `rope` and attention with tasks sized by the data they cover (a task shorter than the ~120 ns it costs to claim slows its region down); matmul strips written straight into the result in place of a `[n_out][t]` transpose; the lane-adding epilogue in `tile_q8` (with an accumulator per output the tile needs more than the 32 vector registers and spills); prefill on every performance core.

**Tried and dropped.**

- `smmla` for `sdot`: the same Gop/s, with a repacking cost.
- `sdot` by element: needs 16 independent accumulators to cover its latency at 4 per cycle, which brings back the epilogue adds.
- A scale per token instead of per block: 2.5 points of top-1 agreement.
- Blocks of 256 values: the tile measured slower, 322 against 352 Gop/s per thread at `k = 4864` where no short tail block is involved.
- Software prefetch from the calling thread: slower, since `prfm` stalls once the miss queue is full.
- Tasks that claim a fixed number of rows instead of bytes in decode: 14 KB tasks reach 80 GB/s, 256 KB ones about 105.
- Thread priority classes: no change in any ceiling.

**Left, each with its bound.** Weight prefetch from a background thread with a redesigned pool: at most 5% of decode. Packing the float32 layout the way int8 is: at most 3% of float32 decode, and a large change. Fusing activation quantization into the op before it: about 1 ms of prefill. A prefill-specific attention kernel: about 0.8 ms. None is a missing piece of understanding, each is a few percent, and what is left is the shape of the memory system and the instruction mix, not the code. Further than this on one chip needs the GPU, which is the Metal stage in [ROADMAP.md](ROADMAP.md).
