"""CPU prefill and decode speed of this engine against llama.cpp and candle, on
the same weights and prompt. Run from the repo root on an otherwise idle machine:

    python3 crates/inference/scripts/compare.py [rounds]

Needs llama.cpp (`brew install llama.cpp`) and, in crates/inference/data/gguf/,
the checkpoint converted with llama.cpp's converter (from its source tree, same
release as the installed binaries), once per type in (f32, q8_0):

    python convert_hf_to_gguf.py <repo>/crates/inference/data/qwen2.5-coder-0.5b \\
        --outtype <type> --outfile <repo>/crates/inference/data/gguf/qwen-<type>.gguf

Every engine runs on the CPU (Metal disabled via `--device none -ngl 0`; on
macOS Homebrew's llama.cpp build still routes float32 prefill through Apple
Accelerate, while this engine and candle run pure CPU NEON).
Engines alternate within a round so drift hits all of them alike. llama.cpp and
candle are tried on 6, 8 and 10 threads and keep their best one per phase; this
engine uses its own default policy. Each engine keeps its own defaults
otherwise (llama.cpp stores its key/value cache as float16, this engine as
float32).
"""

import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "crates/inference/scripts"
GGUF = ROOT / "crates/inference/data/gguf"
THREADS = (6, 8, 10)
CPU_ONLY = ["--device", "none", "-ngl", "0"]
# (label, GGUF file suffix, flags of this engine's bench)
FORMATS = [("float32", "f32", []), ("int8", "q8_0", ["--int8"])]


def run(cmd, **env):
    done = subprocess.run(
        [str(c) for c in cmd], cwd=ROOT, env=os.environ | env, capture_output=True, text=True
    )
    if done.returncode:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{done.stderr[-2000:]}")
    return done.stdout


def field(pattern, text):
    return re.search(pattern, text, re.M).group(1)


def tokenize(model, text):
    out = run(["llama-tokenize", "-m", model, "-p", text, "--ids", "--no-bos", "--log-disable"])
    return json.loads(out.strip())


def ours(flags):
    out = run([ROOT / "target/release/examples/bench", "qwen2.5-coder-0.5b", *flags])
    return {
        "prefill": float(field(r"^prefill .* = (\d+) tok/s", out)),
        "decode": float(field(r"^decode .* = ([\d.]+) tok/s", out)),
        "prompt": field(r"^prompt +(.+)$", out),
        "tokens": int(field(r"^prefill +(\d+) tokens", out)),
        "greedy": json.loads(field(r"^greedy ids +(\[.*\])$", out)),
    }


def llama(model, n_prompt):
    """{(phase, threads): tokens per second}."""
    speeds = {}
    for phase, args in [
        ("prefill", ["-p", n_prompt, "-n", 0, "-r", 7]),
        # -d fills the cache to the prompt's length before decode is timed.
        ("decode", ["-p", 0, "-n", 64, "-d", n_prompt, "-r", 3]),
    ]:
        out = run(
            ["llama-bench", "-m", model, *CPU_ONLY, "-t", ",".join(map(str, THREADS)), *args, "-o", "json"]
        )
        for row in json.loads(out):
            speeds[phase, row["n_threads"]] = statistics.median(row["samples_ts"])
    return speeds


def candle(model, ids):
    speeds, greedy = {}, None
    for threads in THREADS:
        out = run(
            [ROOT / "target/candle_bench/release/candle_bench", model, ",".join(map(str, ids))],
            RAYON_NUM_THREADS=str(threads),
        )
        speeds["prefill", threads] = float(field(r"^prefill_tok_s (\S+)", out))
        speeds["decode", threads] = float(field(r"^decode_tok_s (\S+)", out))
        greedy = json.loads(field(r"^greedy ids (\[.*\])$", out))
    return speeds, greedy


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run(["cargo", "build", "--release", "-p", "inference", "--example", "bench"])
    run(
        ["cargo", "build", "--release", "--manifest-path", SCRIPTS / "candle_bench/Cargo.toml"],
        CARGO_TARGET_DIR=str(ROOT / "target/candle_bench"),
        RUSTFLAGS="-C target-cpu=native",
    )

    # {(format, engine, phase, threads): [tokens per second, one per round]}
    speeds, greedy = {}, {}
    for round_ in range(rounds):
        for label, suffix, flags in FORMATS:
            print(f"round {round_ + 1}/{rounds}, {label}", file=sys.stderr, flush=True)
            model = GGUF / f"qwen-{suffix}.gguf"
            mine = ours(flags)
            prompt = mine["prompt"]
            ids = tokenize(model, mine["prompt"])
            if len(ids) != mine["tokens"]:
                sys.exit(f"prompt is {len(ids)} tokens in llama.cpp, {mine['tokens']} here")
            theirs, greedy[label, "candle"] = candle(model, ids)
            greedy[label, "this engine"] = mine["greedy"]
            results = {
                "this engine": {(p, "default"): mine[p] for p in ("prefill", "decode")},
                "llama.cpp": llama(model, mine["tokens"]),
                "candle": theirs,
            }
            for engine, found in results.items():
                for (phase, threads), speed in found.items():
                    speeds.setdefault((label, engine, phase, threads), []).append(speed)

    print(f"\nTokens per second, median (min to max) of {rounds} rounds\n")
    print("| Weights | Engine | Prefill | Decode |\n|---|---|---|---|")
    for label, _, _ in FORMATS:
        for engine in ("this engine", "llama.cpp", "candle"):
            cells = []
            for phase in ("prefill", "decode"):
                runs, threads = max(
                    ((v, k[3]) for k, v in speeds.items() if k[:3] == (label, engine, phase)),
                    key=lambda r: statistics.median(r[0]),
                )
                note = "" if threads == "default" else f", {threads} threads"
                cells.append(f"{statistics.median(runs):.0f} ({min(runs):.0f} to {max(runs):.0f}{note})")
            print(f"| {label} | {engine} | {cells[0]} | {cells[1]} |")

    # Same weights, same prompt: float32 engines should pick the same tokens.
    # int8 formats round differently, so those may part ways.
    model = GGUF / "qwen-f32.gguf"
    # The converted file carries Qwen's default repetition penalty, which
    # llama.cpp applies even at temperature 0: switch it off to compare argmax.
    text = run(
        ["llama-completion", "-m", model, *CPU_ONLY, "-no-cnv", "--no-display-prompt",
         "-p", prompt, "-n", 16, "--temp", "0", "--repeat-penalty", "1.0"]
    )
    greedy["float32", "llama.cpp"] = tokenize(model, text.removesuffix("\n\n"))
    print("\nfirst 16 greedy tokens, float32:")
    for engine in ("llama.cpp", "candle"):
        same = greedy["float32", engine] == greedy["float32", "this engine"]
        print(f"  {engine}: {'same as this engine' if same else 'DIFFERENT ' + str(greedy['float32', engine])}")


if __name__ == "__main__":
    main()
