//! Run one bounded CPU-only Llama chat request from a supported local GGUF.
//!
//! `cargo run --example llama_prompt -- path/to/model.gguf "hello" 16`

use rustgrad::models::transformer::{LlamaChatMessage, LlamaChatRole, LlamaPromptWorkflow};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: llama_prompt <model.gguf> <prompt> [max_tokens]")?;
    let prompt = args
        .next()
        .ok_or("usage: llama_prompt <model.gguf> <prompt> [max_tokens]")?;
    let max_new_tokens = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    if args.next().is_some() {
        return Err("usage: llama_prompt <model.gguf> <prompt> [max_tokens]".into());
    }

    let workflow = LlamaPromptWorkflow::from_path(path)?;
    let output = workflow.generate_chat(
        &[LlamaChatMessage::new(LlamaChatRole::User, prompt)?],
        max_new_tokens,
    )?;
    println!("{}", output.generation().decoded());
    Ok(())
}
