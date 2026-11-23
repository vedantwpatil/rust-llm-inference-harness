"""Regenerates data/<name>/reference.safetensors: what Hugging Face computes in
float32 for a fixed prompt, so each step of the Rust forward pass has an answer
to compare against. Needs `pip install torch transformers safetensors tiktoken`;
run from the repo root with

    python3 crates/inference/scripts/gen_reference.py gpt2
    python3 crates/inference/scripts/gen_reference.py qwen2.5-coder-0.5b
    python3 crates/inference/scripts/gen_reference.py tiny-llama3

(not `-I`: it hides the user site-packages, where torch's dependencies can live).

Each PyTorch model is built from the checkpoint in data/<name>/ (nothing is
downloaded), so Rust and PyTorch read the same weights; BF16 weights are widened
to float32 exactly. `tiny-llama3` has no checkpoint to fetch: a small randomly
initialised Llama whose config turns on Llama 3's rope frequency scaling is
generated here and saved there. The output is a safetensors file so the Rust
side can reuse `load`: token ids are stored as f32 (exact below 2^24). Like the
weights it is gitignored, rerun this script to recreate it.

Tensors written (T = prompt length, batch dimension dropped, W = model width):
  ids          [T]          prompt token ids
  embed        [T, W]       token embedding (plus position embedding for GPT-2)
  block0.ln_1  [T, W]       block 0, output of each sub-layer, before the
  block0.attn  [T, W]       residual add (attn is after the output projection)
  block0.ln_2  [T, W]
  block0.mlp   [T, W]
  block.0..N   [T, W]       output of each full block (after both residuals)
  ln_f         [T, W]       final norm
  logits       [T, vocab]   every position, not just the last
  greedy_ids   [T + new]    HF generate, sampling off, prompt included
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

name = sys.argv[1]
DATA = Path(__file__).resolve().parent.parent / "data" / name

if name == "gpt2":
    import tiktoken

    model = GPT2LMHeadModel._from_config(GPT2Config(), attn_implementation="eager").eval()
    missing, unexpected = model.transformer.load_state_dict(
        load_file(DATA / "model.safetensors"), strict=False
    )
    # newer transformers rebuilds the causal mask itself, so attn.bias is "unexpected"
    assert not missing, missing
    assert all(k.endswith("attn.bias") for k in unexpected), unexpected
    # lm_head must share wte's storage, or the logits would not use the loaded weights
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode("The capital of France is")
    new_tokens, eos = 10, 50256
    decode = enc.decode
    blocks = model.transformer.h
    parts = {"ln_1": "ln_1", "attn": "attn", "ln_2": "ln_2", "mlp": "mlp"}
    final_norm = model.transformer.ln_f
elif name == "tiny-llama3":
    torch.manual_seed(0)
    # 4 query heads share 2 key/value heads; original context 64 puts head
    # dimension pair 0 in the unscaled band, pair 1 in the blended band and the
    # rest in the divided band; the 48-token prompt runs past the original context.
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rms_norm_eps=1e-5, rope_theta=500000.0, tie_word_embeddings=False,
        initializer_range=0.5, eos_token_id=1, bos_token_id=0, pad_token_id=0,
        rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                      "high_freq_factor": 4.0, "original_max_position_embeddings": 64},
    )
    model = LlamaForCausalLM._from_config(config, attn_implementation="eager").eval()
    DATA.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DATA, safe_serialization=True)
    ids = torch.randint(2, 128, (48,)).tolist()
    new_tokens, eos = 8, 1
    decode = str
else:
    model = AutoModelForCausalLM.from_pretrained(
        DATA, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tok = AutoTokenizer.from_pretrained(DATA)
    # explicit system turn, so the Rust `chatml` helper needs no hidden default
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("prompt text     ", repr(text))
    ids = tok(text).input_ids
    new_tokens, eos = 24, model.config.eos_token_id
    decode = tok.decode

if name != "gpt2":
    blocks = model.model.layers
    parts = {
        "ln_1": "input_layernorm", "attn": "self_attn",
        "ln_2": "post_attention_layernorm", "mlp": "mlp",
    }
    final_norm = model.model.norm

saved = {}


def grab(key):
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        saved[key] = t[0].detach()

    return hook


for i, block in enumerate(blocks):
    block.register_forward_hook(grab(f"block.{i}"))
for key, attr in parts.items():
    getattr(blocks[0], attr).register_forward_hook(grab(f"block0.{key}"))
final_norm.register_forward_hook(grab("ln_f"))

x = torch.tensor([ids])

with torch.no_grad():
    out = model(x, output_hidden_states=True)
    tensors = dict(saved)  # snapshot: generate below fires the hooks again
    greedy = model.generate(
        x,
        attention_mask=torch.ones_like(x),
        do_sample=False,
        max_new_tokens=new_tokens,
        eos_token_id=eos,
        pad_token_id=eos if isinstance(eos, int) else eos[0],
        # the checkpoint's generation_config would otherwise sample and penalise repeats
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
    )

tensors["embed"] = out.hidden_states[0][0]
tensors["logits"] = out.logits[0]
tensors["ids"] = torch.tensor(ids)
tensors["greedy_ids"] = greedy[0]

path = DATA / "reference.safetensors"
save_file({k: v.detach().clone().float().contiguous() for k, v in tensors.items()}, path)

print("prompt ids      ", ids)
print("greedy ids      ", greedy[0].tolist())
print("greedy text     ", repr(decode(greedy[0].tolist())))
print("wrote", len(tensors), "tensors to", path)
