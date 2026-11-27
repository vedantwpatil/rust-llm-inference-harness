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
| plus the [performance-ceiling work](#performance-ceiling), prefill on 10 threads | about 4380 | about 225 | about 90% |

Prefill reaches roughly 750 billion floating-point operations per second (17.5 billion operations for the prompt in about 23.5 milliseconds). Decode matrix multiplication runs near 120 to 124 gigabytes per second on large matrices against a 124 to 127 gigabytes per second probe, leaving roughly 0.4 milliseconds per token spread across small-matrix overhead, attention (0.21 milliseconds), normalizations, activations, and residual additions.

**Not done.** The residual additions are serial. Every `Backend` call allocates a fresh `Vec` for its output (though large prefill outputs skip single-thread zero-filling and fault pages in parallel).

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

Weights are symmetric 8-bit integers with one scale per output row (`scale = max|row| / 127`). A matrix multiplication against them rounds its activations to 8-bit integers as well, one scale per `Q8_BLOCK` (128) values of a token. On processors with ARMv8.6 integer matrix multiply (`i8mm`, checked at runtime), the inner loop uses `smmla` (2-by-8 by 8-by-2 matrix multiply-accumulate, 32 multiply-adds per instruction with no pairwise horizontal reduction epilogue); on older 64-bit ARM processors it falls back to `sdot` (16 multiply-adds per instruction). The row scale is applied once per output; the activation scales are the ones that matter for accuracy. A scale per whole token let one outlier channel flatten the rest and cost 2.5 points of top-1 agreement with 32-bit floating-point, and a scale per block confines it.

**What it costs in accuracy.** `eval_int8` over 512 tokens of `crates/tokenizer/src/tokenizer.rs`: 97.7% of positions (499 of 511) pick the same top-1 token as 32-bit floating-point and perplexity rises by 0.38% (12.414 to 12.461). Across other repository files earlier checks stayed within 97.1% to 97.5% agreement and 0.14% to 0.77% perplexity rise. Weight-only 8-bit integer had measured 98.2% and 0.30% on the first file, so rounding activations costs about a point of agreement. Logits on the verification prompt are within `1.5e-2` of Hugging Face's, relative to the logits' scale. With the top token different at roughly 1 position in 40, greedy output eventually diverges from 32-bit floating-point; that is the expected shape of quantization error, which is why the `1e-4` bar does not apply.

**What it buys.** Weight bytes per token fall from 1976 megabytes to 496 megabytes, and decode rises from 57 to about 225 tokens per second — 3.95 times faster for 4 times fewer bytes. Prefill is compute-bound and rises from about 1120 to about 3600 to 3730 tokens per second, roughly 3.3 times faster.

**Packed layout.** A tile of multiple weight rows against one activation row would otherwise read separate streams `k` bytes apart. One thread pulled four plain streams from memory at 29 to 68 gigabytes per second depending on `k`, versus 91 to 94 gigabytes per second packed. With `i8mm`, 8-bit integer weights are stored in groups of 8 rows (`packed_index_smmla8`, 1024 contiguous bytes per 128-element block); without `i8mm`, they pack in groups of 4 rows (`packed_index`). Leftover rows after the last whole group stay plain. The 32-bit floating-point layout is unchanged.

**What stays float32.** Normalization weights and biases (tiny, and their error is not averaged away by a dot product), the activations between operations, and the key and value cache. The token embedding table is quantized; the tied output projection reads the same matrix, and it is the largest tensor (136 million of the 494 million parameters).

### Tokenizer as data

`Tokenizer::from_tokenizer_json` takes the vocabulary, merges, pre-tokenizer regex and added tokens from the file, so the stage 1 encoder runs unchanged on a 151K vocabulary. All added tokens act as specials, as in Hugging Face. The `tokenizer.json` normalizer (Unicode NFC for Qwen) is not implemented: text that is not already NFC can encode to different ids than Hugging Face gives.

### Speed

Same roofline as stage 2: `bandwidth / (weight bytes + key and value cache bytes)`. Prefill pushes 101 tokens through 358 million matrix-multiplied parameters (the embedding table is only read for the last row): 72 billion operations for the layers. Prefill tokens per second is lower than GPT-2's because each token costs 4 times the arithmetic.

| Standalone wall-power run (10 performance cores, decode up to 8) | Prefill (tokens/s) | Decode (tokens/s) | Decode share of its roofline |
| ---------------------------------------------------------------- | ------------------ | ----------------- | ---------------------------- |
| GPT-2, 32-bit floating-point, 103 tokens                         | about 4380         | about 225         | about 90%                    |
| Qwen2.5-Coder-0.5B, 32-bit floating-point, 101 tokens            | about 1120 to 1150 | about 57          | about 88% to 90%             |
| Qwen2.5-Coder-0.5B, 8-bit integer, 101 tokens                    | about 3600 to 3730 | about 225         | about 89% to 90%             |

Head-to-head on Qwen2.5-Coder-0.5B across 3 alternating rounds on wall power (`crates/inference/scripts/compare.py`, median and min to max tokens per second; all 16 greedy 32-bit floating-point tokens match across all three engines). Note that on macOS Homebrew's `llama.cpp` still routes 32-bit floating-point prefill through Apple Accelerate even with Metal disabled, while this engine and `candle` run pure CPU NEON.

| Weights               | Engine      | Prefill                        | Decode                      |
| --------------------- | ----------- | ------------------------------ | --------------------------- |
| 32-bit floating-point | this engine | 928 (819 to 1118)              | 58 (57 to 58)               |
| 32-bit floating-point | llama.cpp   | 1108 (1067 to 1109, 6 threads) | 57 (54 to 58, 10 threads)   |
| 32-bit floating-point | candle      | 451 (442 to 454, 8 threads)    | 25 (25 to 25, 6 threads)    |
| 8-bit integer         | this engine | 2934 (2125 to 3467)            | 212 (180 to 221)            |
| 8-bit integer         | llama.cpp   | 1554 (1463 to 1668, 8 threads) | 189 (184 to 191, 8 threads) |
| 8-bit integer         | candle      | 812 (799 to 836, 8 threads)    | 131 (128 to 147, 6 threads) |

### Performance ceiling

Each number below is measured on wall power, not estimated: `ceilings` times register-only loops and memory streaming on this chip, `bench` times the model, and the gap between them is accounted for. The `bench` breakdown prints milliseconds per operation and, for decode, gigabytes per second per matrix shape.

| Limit (Apple M3 Max, wall power)     | Measured                                                                                              |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Memory read, 2 or more threads       | 123 to 128 gigabytes per second (one thread: 83 to 86)                                                |
| 32-bit floating-point fused multiply-add | 121 billion operations per second per thread, 1130 on 10                                          |
| 8-bit integer `sdot` / `smmla`       | 472 to 480 billion operations per second per thread, 4560 to 4570 on 10 (95% scaling)                 |
| One cold slice streamed per region   | 0.8 MB: 66 to 101 GB/s, 1 MB: 90 to 104, 4.4 MB: 93 to 119, 8.8 MB: 100 to 122 (4 to 10 threads)      |
| Empty parallel region                | 0.58 µs on 2 threads, 0.59 on 4, 0.69 on 6, 0.78 on 8, 0.89 on 10                                     |

**Decode, Qwen 8-bit integer: 4.43 milliseconds per token against 3.97 milliseconds at the roofline (89% to 90%).** The 0.46-millisecond gap is two parts:

- About 0.15 milliseconds is the small matrices. Even with the lock-free pool and static 6-thread splitting, a 0.8-megabyte attention output projection streams at 99 to 103 gigabytes per second and a 1.0-megabyte fused query, key, and value projection at 102 to 105 gigabytes per second, right at the cold-slice ceiling (101 to 104 gigabytes per second). Large matrices sit right on the memory ceiling: 118 to 122 gigabytes per second for the 8.8-megabyte feed-forward matrix and 121 to 124 gigabytes per second for the 137-megabyte logits projection.
- About 0.35 milliseconds is everything that is not a matrix multiplication: attention 0.12 milliseconds (parallelized across all 14 query heads with stack score buffers and register-resident value accumulation), Swish-gated linear unit 0.07, root mean square normalization 0.02, rotary position embedding 0.01, and residual additions plus buffer allocation 0.13.

32-bit floating-point Qwen (88% to 90% of roofline) and GPT-2 (90% of roofline) show the same profile.

**Prefill, Qwen 8-bit integer: 27 to 28 milliseconds for 101 tokens (3600 to 3730 tokens per second).**

- Matrix multiplication takes 24.2 to 25.0 milliseconds total (about 1.1 milliseconds for the memory-bound logits projection, and 23.1 to 23.9 milliseconds for the 72 billion operations across the 24 layers — roughly 3000 billion operations per second, or 66% of the raw register-only `smmla` ceiling including activation quantization, packing, per-block float conversion and scaling, and writing uninitialized output memory).
- Why `smmla` won despite equal raw register peak to `sdot`: an 8-by-8 `smmla` tile computes 64 outputs in 16 accumulators with zero pairwise horizontal additions (`addp`), uses all 32 vector registers with double-buffered loads and zero stack spills, and pairs with fused single-pass quantization and packing (`quantize_block`) into a reusable scratch buffer (`q8_scratch`). Together with removing per-task temporary allocations and pruning the final layer to only the last token after updating the cache, 8-bit integer prefill matrix multiplication fell from 35.9 to 24.2 milliseconds.
- The rest of the 28 milliseconds: tiled 4-by-4 prefill attention 1.23 milliseconds, Swish-gated linear unit 0.93 to 1.13 milliseconds (4-vector Estrin exponential and Newton-Raphson reciprocal), root mean square normalization 0.47 to 0.57 milliseconds, cached rotary position embedding 0.17 milliseconds, and copies plus residual additions 1.15 milliseconds.

**What got it here.** A lock-free zero-allocation `Pool` (single atomic `(epoch, width)` dispatch, cache-line-aligned per-worker completion slots, interactive quality-of-service pinning on macOS worker threads); static splitting for small decode matrices and large contiguous dynamic chunks for big ones; `smmla` 8-by-8 prefill and 8-by-1 decode kernels with fused single-pass NEON block quantization; direct tile writes into uninitialized output memory; 4-vector unrolled NEON activations and 4-accumulator layer normalization; cached rotary position embedding tables; tiled 4-by-4 prefill attention with register-resident value accumulation and split-key long-context decode attention; zero-copy single-token query, key, and value slicing; and final-layer prefill pruning when not tracing.

**Tried and dropped.**

- `sdot` by element: needs 16 independent accumulators to cover its latency at 4 per cycle, which brings back the epilogue adds.
- Hand-written spill-free 4-by-4 `sdot` assembly: Apple M3's out-of-order store forwarding hides compiler spills in LLVM's fully unrolled `tile_q8::<4, 4>`, so `smmla` 8-by-8 was the real win over `sdot`.
- A scale per token instead of per block: costs 2.5 points of top-1 agreement.
- Blocks of 256 values: slower in cache than 128-value blocks (322 against 352 billion operations per second per thread at `k = 4864`).
- Software prefetch from the calling thread: slower, since `prfm` stalls once the miss queue is full.
- Static single-slab partitioning across all 10 performance cores on macOS: any single OS preemption stalls all other threads at the barrier; capping decode at 8 threads with large dynamic chunks (and sizing short elementwise passes to buffer size) keeps bandwidth high without straggler spikes.

**Left, each with its bound.** Pre-packed 32-bit floating-point weights for a 12-by-8 prefill microkernel: about 10% on 32-bit floating-point prefill at the cost of a second layout and double load work (and <= 3% on decode). Half-precision key and value cache: halves attention memory traffic at long contexts (1024+ tokens), negligible at 100 tokens. Further than this on one chip requires the GPU, which is the Metal stage in [ROADMAP.md](ROADMAP.md).
