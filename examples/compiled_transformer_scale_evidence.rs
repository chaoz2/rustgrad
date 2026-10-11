//! Stable training benchmark geometry; kept unchanged for historical comparisons.

#[path = "support/transformer_training_evidence.rs"]
mod evidence;

const BATCH: usize = 4;
const TIME: usize = 8;
const VOCAB: usize = 16;
const EMBEDDING: usize = 8;
const HEADS: usize = 2;
const FEED_FORWARD: usize = 32;
const VALID_LENGTHS: [[usize; BATCH]; 2] = [[8, 7, 5, 2], [8, 5, 3, 0]];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    evidence::run()
}
