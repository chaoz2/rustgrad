//! Generate or verify the machine-readable compatibility ledger.

#[path = "../compatibility_manifest/mod.rs"]
mod compatibility_manifest;

use compatibility_manifest::{MANIFEST_PATH, SOURCE_PATH, render};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let mode = arguments.next();
    if arguments.next().is_some() {
        return Err("usage: compatibility_manifest [--check|--write]".into());
    }

    let source = fs::read_to_string(SOURCE_PATH)?;
    let rendered = render(&source)?;
    match mode.as_deref() {
        None => print!("{rendered}"),
        Some("--write") => fs::write(MANIFEST_PATH, rendered)?,
        Some("--check") => {
            let checked_in = fs::read_to_string(MANIFEST_PATH)?;
            if checked_in != rendered {
                return Err(format!(
                    "{MANIFEST_PATH} is stale; run `cargo run --bin compatibility_manifest -- --write`"
                )
                .into());
            }
        }
        Some(_) => return Err("usage: compatibility_manifest [--check|--write]".into()),
    }
    Ok(())
}
