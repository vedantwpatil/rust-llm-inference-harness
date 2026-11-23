"""Regenerates data/gpt2/oracle_cases.in, the expected GPT-2 ids the Rust
oracle test compares against, and data/gpt2/bench_expected.in, the token count
and id hash the speed benchmarks check their output against. Needs
`pip install tiktoken`; run from crates/tokenizer with
`python3 scripts/gen_oracle.py`.

The output is a Rust expression (`include!`d by src/tests.rs), so there is no
JSON parser on the Rust side. Non-printable-ASCII chars are written as
\\u{..} escapes so the file survives any editor.
"""

import tiktoken
from oracle_cases import CASES, ENDOFTEXT, rust_str

enc = tiktoken.get_encoding("gpt2")
lines = ["&["]
for text in CASES:
    ids = enc.encode(text, allowed_special={ENDOFTEXT})
    assert enc.decode(ids) == text
    lines.append("    (%s, &[%s])," % (rust_str(text), ", ".join(map(str, ids))))
lines.append("]")

with open("data/gpt2/oracle_cases.in", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote %d cases" % len(CASES))


# Speed benchmarks in src/tests.rs encode these two texts and compare the
# number of ids and an FNV-1a hash of them with tiktoken's. BENCH_REPEATS and
# both text rules must match the Rust side.
BENCH_REPEATS = 50
BENCH_TEXTS = ["\n".join(CASES) * BENCH_REPEATS, "a" * 4000]


def fnv(ids):
    h = 0xCBF29CE484222325
    for i in ids:
        h = ((h ^ i) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


expected = []
for text in BENCH_TEXTS:
    ids = enc.encode(text, allowed_special={ENDOFTEXT})
    expected.append("(%d, 0x%x)" % (len(ids), fnv(ids)))
with open("data/gpt2/bench_expected.in", "w") as f:
    f.write("[%s]\n" % ", ".join(expected))
print("wrote bench expectations")
