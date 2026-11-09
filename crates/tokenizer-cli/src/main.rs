use std::{env, fs};
use tokenizer::{
    TOY_CORPUS, TOY_VOCAB_SIZE, Tokenizer, bytes_per_token, tokenize_by_byte, tokenize_by_char,
    tokenize_by_word,
};

fn main() -> Result<(), String> {
    println!("Hello, rust tokenizer!");

    // We need to take in input "prompt"
    let prompt = take_input()?;

    // Tokenize the prompt

    // Could have all of the different types of tokenizers here to see how they change for the same
    // prompt

    let word_by_word = tokenize_by_word(&prompt);
    let char_by_char = tokenize_by_char(&prompt);
    let byte_by_byte = tokenize_by_byte(&prompt);
    // TODO: Load a larger corpus
    let corpus: Vec<&str> = if prompt.len() > 100 {
        vec![prompt.as_str()]
    } else {
        TOY_CORPUS.to_vec()
    };
    let tokenizer = Tokenizer::train(&corpus, TOY_VOCAB_SIZE);

    // DISPLAY
    section("Prompt");
    println!("{prompt:?}  ({} bytes)", prompt.len());

    // Should consider running this into a llm or having some visual/demonstration of how each
    // tokenizer works and the performance of it. That might be something to do after the tokenizers
    // themselves as it would first require creating my own LLM to be able to feed a tokenized prompt
    // into which would be it's own project
    //
    // Some ideas on displays with just the tokenizer itself were to figure out the compresion ratio
    // it was able to determine and printing out the merged tokens to see the patterns it learned
    section("Baselines");
    println!(
        "word {:>4} tokens  {}",
        word_by_word.len(),
        preview(&word_by_word)
    );
    println!(
        "char {:>4} tokens  {}",
        char_by_char.len(),
        preview(&char_by_char)
    );
    println!(
        "byte {:>4} tokens  {}",
        byte_by_byte.len(),
        preview(&byte_by_byte)
    );

    section("BPE: what training learned");
    println!(
        "{} vocab entries, {} merges (rank: pair -> new token)",
        tokenizer.vocab.len(),
        tokenizer.merges.len()
    );
    for (rank, (a, b)) in tokenizer.merges.iter().enumerate() {
        println!("  {rank:>3}: {a:>5} + {b:<5} -> {a}{b}");
    }

    let ids = tokenizer.encode(&prompt);
    section("BPE: encoded prompt");
    println!("{} tokens", ids.len());
    println!("  {:>3}  {:>5}  token", "pos", "id");
    for (pos, &id) in ids.iter().enumerate().take(PREVIEW) {
        println!("  {pos:>3}  {id:>5}  {:?}", tokenizer.vocab[id as usize]);
    }
    if ids.len() > PREVIEW {
        println!("  ... +{} more", ids.len() - PREVIEW);
    }

    let decoded_prompt = tokenizer.decode(&ids).unwrap();
    let verdict = if decoded_prompt == prompt {
        "ok"
    } else {
        "MISMATCH"
    };
    println!("round trip {verdict}: {decoded_prompt:?}");

    // METRICS
    // The trained vocab is tiny, so GPT-2's published merges are the useful
    // comparison. They load as data; without the file that row is skipped.
    let trained_on = if corpus.len() == 1 {
        "the prompt".to_string()
    } else {
        format!("{} corpus lines", corpus.len())
    };
    let mut counts = vec![
        ("word".to_string(), word_by_word.len()),
        ("char".to_string(), char_by_char.len()),
        ("byte".to_string(), byte_by_byte.len()),
        (
            format!(
                "BPE, {} vocab, trained on {trained_on}",
                tokenizer.vocab.len()
            ),
            ids.len(),
        ),
    ];
    match fs::read_to_string(GPT2_MERGES_PATH) {
        Ok(merges_txt) => {
            let gpt2 = Tokenizer::from_gpt2_merges(&merges_txt);
            counts.push((
                format!("BPE, {} vocab, GPT-2's training", gpt2.vocab.len()),
                gpt2.encode(&prompt).len(),
            ));
        }
        Err(e) => println!("(no GPT-2 row: {GPT2_MERGES_PATH}: {e})"),
    }
    section(&format!(
        "Compression of the same {} byte prompt",
        prompt.len()
    ));
    println!("{:>6}  {:>11}  tokenizer", "tokens", "bytes/token");
    for (name, n) in counts {
        let ratio = bytes_per_token(&prompt, n);
        println!("{n:>6}  {ratio:>11.2}  {name}");
    }

    Ok(())
}

fn take_input() -> Result<String, String> {
    env::args()
        .nth(1)
        .ok_or_else(|| "No prompt provided".to_string())
}

// Rows or items shown before a long list is cut with "+N more".
const PREVIEW: usize = 40;

fn section(title: &str) {
    println!(
        "\n── {title} {}",
        "─".repeat(56usize.saturating_sub(title.len()))
    );
}

fn preview<T: std::fmt::Debug>(items: &[T]) -> String {
    let shown = &items[..items.len().min(PREVIEW)];
    match items.len() - shown.len() {
        0 => format!("{shown:?}"),
        cut => format!("{shown:?} ... +{cut} more"),
    }
}

const GPT2_MERGES_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tokenizer/data/gpt2/merges.txt"
);
