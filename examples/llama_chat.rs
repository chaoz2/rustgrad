//! Run a bounded two-turn CPU-only Llama conversation from a supported local GGUF.
//!
//! `cargo run --example llama_chat -- path/to/model.gguf "hello" "tell me more" 16`

use rustgrad::models::transformer::LlamaPromptWorkflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: llama_chat <model.gguf> <first_turn> <second_turn> [max_tokens]")?;
    let first_turn = args
        .next()
        .ok_or("usage: llama_chat <model.gguf> <first_turn> <second_turn> [max_tokens]")?;
    let second_turn = args
        .next()
        .ok_or("usage: llama_chat <model.gguf> <first_turn> <second_turn> [max_tokens]")?;
    let max_new_tokens = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    if args.next().is_some() {
        return Err(
            "usage: llama_chat <model.gguf> <first_turn> <second_turn> [max_tokens]".into(),
        );
    }

    let workflow = LlamaPromptWorkflow::from_path(path)?;
    let mut conversation = workflow.conversation();
    for turn in [first_turn, second_turn] {
        let response = conversation.send(turn, max_new_tokens)?;
        println!("{}", response.assistant().content());
    }
    Ok(())
}
