"""Regenerates data/qwen2.5-coder/oracle_cases.in, the ids Hugging Face's
`tokenizers` gives Qwen2.5-Coder for the shared cases plus Qwen's own special
tokens. Needs `pip install tokenizers` and the checkpoint from
crates/inference/scripts/fetch_model.py; run from crates/tokenizer with
`python3 scripts/gen_qwen_oracle.py`.

Cases NFC would change (decomposed accents) are left out: the Rust loader does
not normalize.
"""

import unicodedata
from pathlib import Path

from oracle_cases import CASES, rust_str
from tokenizers import Tokenizer

TOKENIZER = "../inference/data/qwen2.5-coder-0.5b/tokenizer.json"

QWEN_CASES = CASES + [
    "<|im_start|>user\nWrite a function.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_end|><|im_start|>",
    "<tool_call>{\"name\": \"f\"}</tool_call>",
    "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>",
    "<|im_start",
    "digits 0123456789 split one at a time",
]

tok = Tokenizer.from_file(TOKENIZER)
cases = [t for t in QWEN_CASES if unicodedata.normalize("NFC", t) == t]
lines = ["&["]
for text in cases:
    ids = tok.encode(text, add_special_tokens=False).ids
    assert tok.decode(ids, skip_special_tokens=False) == text, text
    lines.append("    (%s, &[%s])," % (rust_str(text), ", ".join(map(str, ids))))
lines.append("]")

out = Path("data/qwen2.5-coder/oracle_cases.in")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print("wrote %d cases, skipped %d" % (len(cases), len(QWEN_CASES) - len(cases)))
