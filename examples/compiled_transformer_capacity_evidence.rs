//! Fixed 29,568-trainable-parameter causal Transformer training acceptance.
//!
//! Uses the same compiled AdamW/dropout, strict-native replay, pending-window
//! checkpoint, and different-initialization resume contract as the stable
//! benchmark, with a longer context and larger vocabulary and hidden state.
//! This is a capacity/correctness probe, not a throughput claim.

#[path = "support/transformer_training_evidence.rs"]
mod evidence;

const BATCH: usize = 4;
const TIME: usize = 32;
const VOCAB: usize = 128;
const EMBEDDING: usize = 32;
const HEADS: usize = 4;
const FEED_FORWARD: usize = 128;
const VALID_LENGTHS: [[usize; BATCH]; 2] = [[32, 28, 20, 8], [32, 20, 12, 0]];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    evidence::run()
}
