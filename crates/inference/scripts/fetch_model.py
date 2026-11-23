"""Downloads a Hugging Face checkpoint into data/<name>/ (gitignored for the
weights): model.safetensors, config.json and tokenizer.json. Needs
`pip install huggingface_hub`; run from the repo root with

    python3 crates/inference/scripts/fetch_model.py Qwen/Qwen2.5-Coder-0.5B-Instruct qwen2.5-coder-0.5b

Sharded checkpoints (model-0000N-of-0000M.safetensors) are not handled: the
Rust loader reads one file. Both models in the roadmap ship a single file.
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo, name = sys.argv[1], sys.argv[2]
dest = Path(__file__).resolve().parent.parent / "data" / name
dest.mkdir(parents=True, exist_ok=True)

for remote, local in [
    ("model.safetensors", "model.safetensors"),
    ("config.json", "config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("generation_config.json", "generation_config.json"),
]:
    try:
        cached = hf_hub_download(repo, remote)
    except Exception as e:  # generation_config.json is optional
        print("skipped", remote, type(e).__name__)
        continue
    shutil.copyfile(cached, dest / local)
    print("wrote", dest / local)
