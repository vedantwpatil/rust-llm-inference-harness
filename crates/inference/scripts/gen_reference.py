"""Regenerates data/gpt2/reference.safetensors: what Hugging Face's GPT-2
computes for a fixed prompt, so each step of the Rust forward pass has an
answer to compare against. Needs `pip install torch transformers safetensors
tiktoken`; run from the repo root with
`python3 crates/inference/scripts/gen_reference.py` (not `-I`: it hides the
user site-packages, where torch's dependencies can live).

The PyTorch model is built from data/gpt2/model.safetensors itself (nothing is
downloaded), so Rust and PyTorch read the same weights. The output is a
safetensors file so the Rust side can reuse `load`, which only takes F32:
token ids are stored as f32 (exact below 2^24). Like the weights it is
gitignored, rerun this script to recreate it.

Tensors written (T = prompt length, batch dimension dropped):
  ids          [T]          prompt token ids
  embed        [T, 768]     wte[id] + wpe[pos]
  block0.ln_1  [T, 768]     block 0, output of each sub-layer, before the
  block0.attn  [T, 768]     residual add (attn is after c_proj)
  block0.ln_2  [T, 768]
  block0.mlp   [T, 768]
  block.0..11  [T, 768]     output of each full block (after both residuals)
  ln_f         [T, 768]     final LayerNorm
  logits       [T, 50257]   every position, not just the last
  greedy_ids   [T + new]    HF generate, sampling off, prompt included
"""

from pathlib import Path

import tiktoken
import torch
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel

PROMPT = "The capital of France is"
NEW_TOKENS = 10
EOS = 50256

DATA = Path(__file__).resolve().parent.parent / "data" / "gpt2"

model = GPT2LMHeadModel._from_config(GPT2Config(), attn_implementation="eager").eval()
missing, unexpected = model.transformer.load_state_dict(
    load_file(DATA / "model.safetensors"), strict=False
)
# newer transformers rebuilds the causal mask itself, so attn.bias is "unexpected"
assert not missing, missing
assert all(k.endswith("attn.bias") for k in unexpected), unexpected
# lm_head must share wte's storage, or the logits would not use the loaded weights
assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()

saved = {}


def grab(name):
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        saved[name] = t[0].detach()

    return hook


for i, block in enumerate(model.transformer.h):
    block.register_forward_hook(grab(f"block.{i}"))
for sub in ["ln_1", "attn", "ln_2", "mlp"]:
    getattr(model.transformer.h[0], sub).register_forward_hook(grab(f"block0.{sub}"))
model.transformer.ln_f.register_forward_hook(grab("ln_f"))

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode(PROMPT)
x = torch.tensor([ids])

with torch.no_grad():
    out = model(x, output_hidden_states=True)
    tensors = dict(saved)  # snapshot: generate below fires the hooks again
    greedy = model.generate(
        x,
        attention_mask=torch.ones_like(x),
        do_sample=False,
        max_new_tokens=NEW_TOKENS,
        pad_token_id=EOS,
    )

tensors["embed"] = out.hidden_states[0][0]
tensors["logits"] = out.logits[0]
tensors["ids"] = torch.tensor(ids)
tensors["greedy_ids"] = greedy[0]

path = DATA / "reference.safetensors"
save_file({k: v.detach().clone().float().contiguous() for k, v in tensors.items()}, path)

print("prompt ids      ", ids)
print("wte[0][:4]      ", model.transformer.wte.weight[0, :4].tolist())
print("greedy ids      ", greedy[0].tolist())
print("greedy text     ", repr(enc.decode(greedy[0].tolist())))
print("wrote", len(tensors), "tensors to", path)
