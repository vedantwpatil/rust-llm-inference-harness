# Architecture

How the code is structured today. Why it is built this way: [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md). Planned crates and stages: [ROADMAP.md](ROADMAP.md#crate-map).

Cargo workspace, one crate per layer:

- `crates/tokenizer`: the library.
- `crates/tokenizer-cli`: demo binary (`main.rs`). `take_input` reads the prompt from a positional arg (no `clap`). Prints compression for word, char, byte, trained BPE and GPT-2 BPE.
- `crates/inference`: the model side. Safetensors loader, a `Backend` trait with a CPU implementation, GPT-2 and Llama-style (Llama, Qwen2) forward passes with a KV cache, int8. Modules in [Inference modules](#inference-modules).

## Tokenizer modules

A tokenizer is normalizer, then pre-tokenizer, then model. This one has no normalizer: on a fixed corpus the text can be fixed up front, though a general dataset would want one.

| Module        | Holds                                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline`    | `tokenize_by_char`, `tokenize_by_word`, `tokenize_by_byte`: naive tokenizers the BPE one is compared against                                            |
| `pretokenize` | `pre_tokenize` and `GPT2_PATTERN`: GPT-2's regex split via `fancy-regex`. Merges never cross a chunk                                                     |
| `byte_level`  | `bytes_to_unicode`: GPT-2's byte to printable-char table, `[char; 256]` indexed by byte                                                                  |
| `train`       | `tokenize_bpe` (the trainer), `compute_pair_freqs`, `merge_pairs`                                                                                       |
| `encode`      | `encode_chunk`: replays merges on one chunk, lowest rank first. `build_ranks`, `build_vocab_ids`                                                         |
| `tokenizer`   | `Tokenizer { vocab, merges, ranks, vocab_ids, pattern, specials, .. }`: `train`, `from_gpt2_merges`, `from_tokenizer_json`, `encode`, `decode`, `decode_stream`. `chatml` |
| `metrics`     | `bytes_per_token`                                                                                                                                       |

`encode` cuts special tokens out first, then pre-tokenizes and replays merges. `decode` joins bytes across tokens and builds the `String` once at the end. `decode_stream(id, &mut pending)` is the one-id-at-a-time version: the caller owns `pending`, an incomplete UTF-8 tail stays in it. Both share `token_bytes`.

`from_tokenizer_json` reads a Hugging Face `tokenizer.json` (Llama 3, Qwen): vocabulary, merges, the `Split` regex of the pre-tokenizer and every added token, so none of them is a constant. Added tokens are all specials, as in Hugging Face. The file's normalizer is ignored. `chatml(system, user)` renders the turn format Qwen was trained on.

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

## Inference modules

`inference` has no tokenizer dependency: it takes ids and returns logits. Demos and benchmarks live in `crates/inference/examples/` and use `tokenizer` as a dev-dependency.

| Module        | Holds                                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `safetensors` | `load`: 8-byte header length, JSON header, raw bytes (F32 or BF16, widened to f32) into `Tensors`, a `HashMap` from tensor name to shape and flat row-major values. `take` removes one by name |
| `backend`     | `Matrix { rows, cols, weights }` with `Weights::{F32, Q8}`, `AttnShape`, the `Backend` trait: `matmul`, `layernorm`, `rmsnorm`, `gelu`, `swiglu`, `rope`, `softmax`, `attention` |
| `cpu`         | `CpuBackend { pool, rope_cache, q8_scratch }`: the reference `Backend`. Parallel across a lock-free `Pool`, matrix multiplications and elementwise operations through `kernel`; prefill uses all performance cores and decode up to 8 |
| `kernel`      | `tile<R, T, U>`: `R` weight rows against `T` activation rows, NEON on aarch64, scalar elsewhere (`dot` is 1 by 1). 8-bit integer kernels: `tile_q8<R, T>` (`sdot`), `tile_q8_smmla8_8x8_store` and `tile_q8_smmla8_8x1` (`smmla` when `i8mm` is available), `quantize_block`. Vectorized `layernorm_row`, `rmsnorm_row`, `swiglu_slice`, `gelu_slice`, `softmax_slice`, `weighted_sum_rows`, `weighted_sum_rows_4_hd64` |
| `pool`        | `Pool`: lock-free persistent fork-join thread pool. `run_threads(width, f)`, `run(n_tasks, f)`, `for_each_chunk_mut(slice, chunk, f)`, and their `_width` forms |
| `kv_cache`    | `KvCache`: per layer, keys and values as flat `[position][kv_dim]` vectors, plus `append_qkv_kv` for direct strided appends and the number of positions held |
| `ops`         | `embed`, `add_in_place`, `argmax`, `split_columns`                                                                                                            |
| `model`       | `Model` trait: `new_cache`, `forward`, `forward_trace`, `weight_bytes`, and the provided greedy loop `generate_each` / `generate_greedy`. `Trace`              |
| `gpt2`        | `Gpt2<B: Backend>`: `from_tensors`, implements `Model`                                                                                                        |
| `config`      | `LlamaConfig::from_json`: sizes, norm epsilon, rope base, tied embeddings, end-of-sequence ids, optional Llama 3 `RopeScaling`. `inv_freq`                    |
| `llama`       | `Llama<B: Backend>`: `from_dir`, `from_tensors`, `quantize`, implements `Model`                                                                               |

### Contracts

- **`Matrix` is `[out, in]`, row-major.** `matmul(x, t, w, bias)` computes `y = x @ w^T + bias` for `t` rows of `x`, so output `o` for one token is a dot product of two contiguous slices. Hugging Face's GPT-2 `Conv1D` weights are `[in, out]` and are transposed once at load.
- **One attention op for prefill, decode and grouped-query attention.** `AttnShape { t, pos0, n_heads, n_kv_heads, head_dim }`: `t` new queries sit at positions `pos0..pos0 + t`, and `k`, `v` hold all `pos0 + t` positions. Query `i` scores keys `0..=pos0 + i`, which is the causal mask. Query head `h` reads key/value head `h / (n_heads / n_kv_heads)`. Full multi-head attention is `n_kv_heads == n_heads`.
- **`KvCache` is appended per layer, then advanced once.** A layer's keys and values are `append`ed during the pass, `advance(t)` runs after the last layer, so `len()` is the position of the next token for every layer in between. Position embeddings are offset by `len()`.
- **`forward_trace` keys match the reference fixture, for every model:** `embed`, `block0.ln_1`, `block0.attn`, `block0.ln_2`, `block0.mlp`, `block.N`, `ln_f`, `logits` (every position). That is what lets one set of checks in `reference.rs` run on any `Model`.
- **`Model` is object safe.** The examples hold a `Box<dyn Model>`; `generate_each` takes `&mut dyn FnMut(u32)` so a caller can stream tokens.
- **`rope` rotates "half" pairs.** In each head, element `j` pairs with element `j + head_dim / 2`, the layout Hugging Face's Llama weights assume. The frequencies come in as data (`LlamaConfig::inv_freq`), so the Llama 3 scaling is not in the op.
- **A quantized `Matrix` is still a `Matrix`.** `matmul` takes the same arguments for both storage kinds; `Matrix::quantize` swaps the storage in place.

### Forward pass

```text
ids ─▶ embed (wte[id] + wpe[pos0 + i]) ─▶ x [t, width]
        │
        │  per block l:
        │    h = layernorm(x)
        │    qkv = matmul(h, c_attn)            [t, 3 * width], split into q | k | v per row
        │    cache.append(l, k, v)
        │    a = attention(q, cache keys, cache values)
        │    x += matmul(a, attn_proj)
        │    x += matmul(gelu(matmul(layernorm(x), c_fc)), mlp_proj)
        ▼
cache.advance(t) ─▶ layernorm ─▶ last row @ wte^T ─▶ logits [vocab]
```

Llama-style block, with the fused projections the loader builds by stacking weight rows:

```text
ids ─▶ embedding rows ─▶ x [t, hidden]
        │
        │  per layer l:
        │    h = rmsnorm(x)
        │    qkv = matmul(h, [q_proj; k_proj; v_proj] (+ bias for Qwen2))   split into q | k | v
        │    rope(q), rope(k)                                               position = cache.len() + row
        │    cache.append(l, k, v)
        │    x += matmul(attention(q, cache), o_proj)                       n_kv_heads < n_heads
        │    gate_up = matmul(rmsnorm(x), [gate_proj; up_proj])
        │    x += matmul(swiglu(gate_up), down_proj)                        silu(gate) * up
        ▼
cache.advance(t) ─▶ rmsnorm ─▶ last row @ (lm_head or embedding)^T ─▶ logits [vocab]
```

Everything that differs between checkpoints is read from `config.json` or the tensor list: sizes, `head_dim`, the rope base and scaling, tied embeddings, and whether the query, key and value projections have biases (Qwen2 yes, Llama no: the loader checks for the tensor).

### Int8

`Weights::Q8 { values: Vec<i8>, scales: Vec<f32> }`: row `r` is `scales[r]` times its values, scale `max|row| / 127`. When the processor supports ARMv8.6 integer matrix multiply (`i8mm`, checked via `use_i8mm`), values are packed in groups of 8 rows (`packed_index_smmla8`, 1024 contiguous bytes per 128-element block laid out as 16 steps of 4 two-by-eight row pairs) so both 8-by-8 direct-store prefill (`tile_q8_smmla8_8x8_store`) and 8-by-1 decode (`tile_q8_smmla8_8x1`) stream sequentially; leftover rows after the last whole group stay plain. On processors without `i8mm`, values pack in groups of 4 rows (`packed_index`) for `tile_q8` (`sdot` or scalar). `Matrix::q8_index` hides the active layout from callers. Normalization weights and biases stay 32-bit floating-point. `Llama::quantize` converts every weight matrix including the token embedding table; its lookup widens one row directly from the active packed layout.

A matrix multiplication against 8-bit integer weights rounds its activations to 8-bit integers once per call, with one scale per `Q8_BLOCK` (128) values of a token (`quantize_block`). On the `smmla` prefill path, quantization and packing into 8-token groups are fused in a single pass using a reusable scratch buffer (`q8_scratch`) and a 1-kilobyte stack buffer per block. Single-token decode quantizes into stack buffers (for widths up to 8192), splits matrices up to 2 megabytes statically across up to 8 performance cores, and divides larger matrices dynamically into an exact multiple of the thread count (3 tasks per thread), writing directly into uninitialized output memory (see [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md)).

### Parallelism

`Pool::new(n)` pins both the calling thread (thread 0) and `n - 1` workers to the interactive quality-of-service class on macOS so active threads stay on performance cores. Starting a region writes the closure pointer into a cache-line-aligned slot and publishes `(epoch, width)` in a single atomic 64-bit word — zero heap allocation and zero mutex acquisition while workers are spinning. Workers spin for 1 millisecond before sleeping on a condition variable (checked via a sleeping bitmask so narrow regions leave unused workers asleep), and signal completion into separate 128-byte-aligned per-worker slots to avoid barrier contention. `run_threads` assigns each participating thread its `(thread_id, width)` for static partitioning; `run_width` layers a stack-allocated atomic counter on top for dynamic work distribution. One region runs at a time; panics on either the caller or any worker are caught and re-raised after every active worker joins. Why this instead of rayon: [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md#stage-2-gpt-2-on-the-cpu).

`CpuBackend::matmul` uses 6-by-4 NEON tiles (`prefill_f32`) for 32-bit floating-point prefill and writes both prefill and decode outputs directly into uninitialized memory through `SharedBuf` (turning the vector into initialized memory once every thread finishes, avoiding single-thread zero-filling and per-task temporary allocations). In `Gpt2` and `Llama`, single-token decode slices the fused query, key, and value projection in place without copying, prefill appends keys and values straight into the key and value cache (`append_qkv_kv`, with rotary position embedding applied in place in `Llama`), and non-traced prefill prunes the final layer after updating the cache so only the last token runs attention, output projection, multilayer perceptron, and final normalization. `rope` caches its `(cos, sin)` tables across layers in `CpuBackend`, `attention` uses 4-by-4 tiled head-major scoring with 4-query batched value accumulation (`weighted_sum_rows_4_hd64`) for prefill and register-resident value accumulation (with split-key merging past 512 keys) for decode, and short elementwise and normalization passes scale their thread count to buffer size (`elem_threads`).

### Verification fixtures

`crates/inference/data/<name>/` is gitignored: `gpt2`, `qwen2.5-coder-0.5b`, `tiny-llama3`. `scripts/fetch_model.py` downloads a checkpoint, `scripts/gen_reference.py <name>` runs Hugging Face in float32 on a fixed prompt and writes `reference.safetensors` (ids, every traced activation, greedy ids). `tiny-llama3` is generated by that script: a random Llama with an untied output projection, no biases and Llama 3 rope scaling, small enough to run in milliseconds. `reference.rs` holds the three checks every model runs against its fixture (layers, greedy, cached decode); tests skip when the files are missing.

### Examples

`crates/inference/examples/`: `bench` (prefill and decode against the bandwidth roofline, `--int8`), `chat` (streams a reply), `eval_int8` (next-token agreement and perplexity, f32 against int8). `common/mod.rs` holds the checkpoint paths and loaders they share.
