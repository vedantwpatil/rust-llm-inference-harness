# Rust LLM Inference Harness

An LLM inference stack built from scratch in Rust: byte-level BPE tokenizer, GPT-2 and Llama-style models on CPU, an inference harness, and a local coding harness on top.

Working on getting gpt-2 and llama style models on cpu currently

For my analysis on tokenizers and how I went about my approach [analysis](./docs/RESEARCH.md)

For the development plan and next steps on this repo look at (AI assisted) [roadmap](/docs/ROADMAP.md)

## Layout

Cargo workspace, one crate per layer of the stack. Today:

- `crates/tokenizer`: the library. Byte-level BPE trainer, encoder, decoder, GPT-2 loader, naive char/word/byte baselines, tests and the tiktoken oracle fixtures.
- `crates/tokenizer-cli`: demo binary, `cargo run -p tokenizer-cli -- "your prompt here"`.
- `crates/inference`: GPT-2 and a Llama-style model (Qwen2.5-Coder) on CPU. Safetensors loader, `Backend` trait with a NEON CPU backend, forward pass with KV cache, greedy decoding, weight-only int8. Examples: `chat` (streams a reply), `bench` (prefill and decode speed against the memory-bandwidth roofline), `eval_int8` (accuracy cost of int8), run with `cargo run --release -p inference --example <name>`. Model files are fetched with `crates/inference/scripts/fetch_model.py`.

`cargo test` runs everything. Speed benchmarks are ignored by default: `cargo test --release bench_ -- --ignored --nocapture`.

## Overview

Plan to implement multiple types of tokenizers and experiment with "outdated" tokenizers before building a [BPE](./docs/RESEARCH.md#bpe-byte-pair-encoding) tokenizer from scratch

The main goal of this is to be able to learn about how the inference pipeline works in a lot of common machine learning pipelines. The way that we'll learn this is by first creating a tokenizer, then creating the respective components around the tokenizer to feed into a model to do inference and then finally use this inference pipeline to create my own harness. Everything should be created from scratch and in rust with writing some bindings to python to be able to call some of the crates for performance testing against other inference pipelines.

Docs are AI assisted with the relevant blog posts linked
