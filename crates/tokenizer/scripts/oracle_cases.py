"""Texts and helpers shared by the oracle generators (gen_oracle.py for GPT-2,
gen_qwen_oracle.py for Qwen). Run those from crates/tokenizer with
`python3 scripts/<name>.py`.
"""

ENDOFTEXT = "<|endoftext|>"

CASES = [
    "",
    "x",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "This is the hugging face course",
    # contractions, curly apostrophe, stray apostrophes
    "I've been told they'll say we're done, he's gone; she'd go. I'm sure.",
    "rock'n'roll 's 't 're O'Brien’s DON'T",
    # numbers
    "It's 3:45pm on 2024-10-24, call 555-0123. 1234567890 3.14159 1e-10",
    # code
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    'fn main() {\n\tprintln!("{}", 1 + 2);\n}\n',
    "https://example.com/a/b?c=d&e=f#g",
    "camelCaseIdentifier snake_case_name HELLO WORLD",
    # whitespace: leading, trailing, runs, tabs, CRLF
    " ",
    "   ",
    "  leading",
    "trailing  ",
    "a  b   c",
    "\n\n\nthree newlines",
    "tabs\t\there",
    "\r\n windows\r\n",
    "end \n",
    "word \n word",
    # repeats
    "a" * 50,
    "!!!!!!!!!! ????",
    # non-ASCII
    "café naïve résumé",
    "日本語のテキスト",
    "Привет, мир!",
    "नमस्ते",
    "emoji \U0001F642\U0001F680\U0001F44D\U0001F3FD",
    "\U0001D514\U0001D52B\U0001D526\U0001D520\U0001D52C\U0001D521\U0001D522",
    # odd whitespace and control characters
    "a b",
    "x​y",
    "­",
    "\x00\x01 \x7f",
    # long-ish prose
    "Four score and seven years ago our fathers brought forth on this continent, "
    "a new nation, conceived in Liberty, and dedicated to the proposition that "
    "all men are created equal.\n\nNow we are engaged in a great civil war, "
    "testing whether that nation, or any nation so conceived and so dedicated, "
    "can long endure.",
    # special token: recognised and never merged
    ENDOFTEXT,
    "before" + ENDOFTEXT + "after",
    ENDOFTEXT + ENDOFTEXT,
    "line one\n" + ENDOFTEXT + "\nline two",
]


def rust_str(s):
    out = []
    for ch in s:
        if 0x20 <= ord(ch) < 0x7F and ch not in '"\\':
            out.append(ch)
        else:
            out.append("\\u{%x}" % ord(ch))
    return '"' + "".join(out) + '"'
