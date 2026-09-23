//! Compile, train, replay, checkpoint, and resume one owned fixed-shape tiny
//! Transformer on CPU or explicitly selected strict Metal.
//!
//! ## Contract
//!
//! - Right-padded batches keep one static shape.
//! - Ignore-index targets provide both attention validity and exact token-mean
//!   gradient weights across each accumulation window.
//! - Resume authenticates the compiled program, optimizer frontier, tied and
//!   frozen module state, and replay progress before publication.
//!
//! ## Resume modes
//!
//! | Mode | Command | Boundary |
//! |---|---|---|
//! | Reuse one plan | `cargo run --example compiled_transformer_train_resume -- cpu-reuse` | Restores in process without rebuilding the graph or capture. |
//! | Portable file | `cargo run --example compiled_transformer_train_resume -- cpu-file-resume` | Restores a deliberately different initialization from a resource-free program artifact and complete module checkpoint. |
//! | Strict-native file | `cargo run --release --example compiled_transformer_train_resume -- native-cpu-file-resume` | Runs the same file lifecycle through CPU JIT with no fallback. |
//! | Cross-process file | `cross-process-produce <cpu|native-cpu> <directory>`, then `cross-process-consume <cpu|native-cpu> <directory>` | Produces a pending RGAB and authenticates its continuation in a fresh OS process. |
//!
//! ## Replay and evidence modes
//!
//! | Mode | Command | Boundary |
//! |---|---|---|
//! | Interpreter CPU | `cargo run --example compiled_transformer_train_resume -- cpu` | Replays graph-free on the host interpreter. |
//! | Strict-native CPU | `cargo run --release --example compiled_transformer_train_resume -- native-cpu` | Replays the same capture through CPU JIT. |
//! | CPU scoreboard | `cargo run --release --example compiled_transformer_train_resume -- native-cpu-scoreboard` | Emits the bounded strict-native evidence report. |
//! | Strict Metal | `cargo run --release --example compiled_transformer_train_resume -- metal` | Uses the first visible Metal device with no CPU fallback. |

#[path = "compiled_transformer_train_resume/mod.rs"]
mod workflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    workflow::run()
}
