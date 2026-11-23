# Architecture

How the code is structured today. Why it is built this way: [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md). Planned crates and stages: [ROADMAP.md](ROADMAP.md#crate-map).

Cargo workspace, one crate per layer:

- `crates/tokenizer`: the library.
- `crates/tokenizer-cli`: demo binary (`main.rs`). `take_input` reads the prompt from a positional arg (no `clap`). Prints compression for word, char, byte, trained BPE and GPT-2 BPE.
- `crates/inference`: the model side. Safetensors loader, a `Backend` trait with a CPU implementation, the GPT-2 forward pass with a KV cache. Modules in [Inference modules](#inference-modules).

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

## Inference modules

`inference` has no tokenizer dependency: it takes ids and returns logits. Demos and benchmarks live in `crates/inference/examples/` and use `tokenizer` as a dev-dependency.

| Module        | Holds                                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `safetensors` | `load`: 8-byte header length, JSON header, raw bytes, into `HashMap<String, (Vec<usize>, Vec<f32>)>` (tensor name to shape and flat row-major values)          |
| `backend`     | `Matrix { rows, cols, data }`, `AttnShape`, the `Backend` trait: `matmul`, `layernorm`, `gelu`, `softmax`, `attention`                                         |
| `cpu`         | `CpuBackend { pool }`: the reference `Backend`. Parallel across a `Pool`, matmul through `kernel::tile`                                                       |
| `kernel`      | `tile<R, T, U>`: `R` weight rows against `T` activation rows, NEON on aarch64, scalar elsewhere. `dot` is the 1 by 1 case                                     |
| `pool`        | `Pool`: persistent fork-join thread pool. `run(n_tasks, f)`, `for_each_chunk_mut(slice, chunk, f)`                                                            |
| `kv_cache`    | `KvCache`: per layer, keys and values as flat `[position][kv_dim]` vectors, plus the number of positions held                                                 |
| `ops`         | `embed`, `add_in_place`, `argmax`                                                                                                                             |
| `gpt2`        | `Gpt2<B: Backend>`: `from_tensors`, `new_cache`, `forward` (logits for the next token), `forward_trace` (every intermediate), `generate_greedy`               |

### Contracts

- **`Matrix` is `[out, in]`, row-major.** `matmul(x, t, w, bias)` computes `y = x @ w^T + bias` for `t` rows of `x`, so output `o` for one token is a dot product of two contiguous slices. Hugging Face's GPT-2 `Conv1D` weights are `[in, out]` and are transposed once at load.
- **One attention op for prefill, decode and grouped-query attention.** `AttnShape { t, pos0, n_heads, n_kv_heads, head_dim }`: `t` new queries sit at positions `pos0..pos0 + t`, and `k`, `v` hold all `pos0 + t` positions. Query `i` scores keys `0..=pos0 + i`, which is the causal mask. Query head `h` reads key/value head `h / (n_heads / n_kv_heads)`. Full multi-head attention is `n_kv_heads == n_heads`.
- **`KvCache` is appended per layer, then advanced once.** A layer's keys and values are `append`ed during the pass, `advance(t)` runs after the last layer, so `len()` is the position of the next token for every layer in between. Position embeddings are offset by `len()`.
- **`forward_trace` keys match the reference fixture:** `embed`, `block0.ln_1`, `block0.attn`, `block0.ln_2`, `block0.mlp`, `block.N`, `ln_f`, `logits` (every position).

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

### Parallelism

`Pool::new(n)` starts `n - 1` workers; the calling thread is the n-th. A region publishes a job (task count, an atomic next-task counter, an atomic done counter) and every thread claims task indices until none are left. Workers spin on an atomic generation counter for a short window after each region, then sleep on a condition variable. A region with one task runs inline. One region runs at a time; a task panic is caught per task and re-raised by `run` after the region ends. Why this instead of rayon: [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md#stage-2-gpt-2-on-the-cpu).

`CpuBackend::matmul` gives each task 16 whole weight rows and every token's dot product with them, so a weight row is read from memory once however many tokens there are. The result is `[n_out][t]` and a second parallel pass transposes it to `[t][n_out]` and adds the bias. `attention` is one task per (query, head).

### Verification fixtures

`crates/inference/data/gpt2/` is gitignored. `scripts/fetch_model.py` downloads a checkpoint, `scripts/gen_reference.py` runs Hugging Face on a fixed prompt and writes `reference.safetensors` (ids, every traced activation, greedy ids). Tests skip when the files are missing.
