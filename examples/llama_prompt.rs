//! Run one bounded CPU or strict-native Llama chat request from a supported local GGUF.
//!
//! `cargo run --example llama_prompt -- path/to/model.gguf "hello" 16`
//! `cargo run --example llama_prompt -- --native path/to/model.gguf "hello" 16`

use rustgrad::models::transformer::{LlamaChatMessage, LlamaChatRole, LlamaPromptWorkflow};

const USAGE: &str = "usage: llama_prompt [--native] <model.gguf> <prompt> [max_tokens]";

#[derive(Debug)]
struct CliError(&'static str);

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for CliError {}

struct PromptArgs {
    native: bool,
    path: String,
    prompt: String,
    max_new_tokens: usize,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<PromptArgs, CliError> {
    let mut args = args.into_iter();
    let first = args.next().ok_or(CliError(USAGE))?;
    let (native, path) = if first == "--native" {
        (true, args.next().ok_or(CliError(USAGE))?)
    } else {
        (false, first)
    };
    let prompt = args.next().ok_or(CliError(USAGE))?;
    let max_new_tokens = args
        .next()
        .map(|value| value.parse::<usize>().map_err(|_| CliError(USAGE)))
        .transpose()?
        .unwrap_or(16);
    if args.next().is_some() {
        return Err(CliError(USAGE));
    }

    Ok(PromptArgs {
        native,
        path,
        prompt,
        max_new_tokens,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1))?;

    let workflow = LlamaPromptWorkflow::from_path(args.path)?;
    let messages = [LlamaChatMessage::new(LlamaChatRole::User, args.prompt)?];
    if args.native {
        println!(
            "{}",
            workflow
                .generate_chat_native(&messages, args.max_new_tokens)?
                .generation()
                .decoded()
        );
    } else {
        println!(
            "{}",
            workflow
                .generate_chat(&messages, args.max_new_tokens)?
                .generation()
                .decoded()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    #[test]
    fn native_flag_is_explicit_and_cpu_positionals_are_unchanged() {
        let cpu = parse_args(["fixture.gguf", "hello", "4"].map(str::to_owned)).unwrap();
        assert!(!cpu.native);
        assert_eq!(cpu.path, "fixture.gguf");
        assert_eq!(cpu.prompt, "hello");
        assert_eq!(cpu.max_new_tokens, 4);

        let native =
            parse_args(["--native", "fixture.gguf", "hello", "4"].map(str::to_owned)).unwrap();
        assert!(native.native);
        assert_eq!(native.path, "fixture.gguf");
        assert_eq!(native.prompt, "hello");
        assert_eq!(native.max_new_tokens, 4);
    }
}
