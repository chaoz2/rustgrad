//! Inspect one or more bounded local CIFAR-10 binary batches without network access.
//!
//! `cargo run --example cifar10_local -- data_batch_1.bin data_batch_2.bin`

use rustgrad::{BatchIter, load_cifar10_files};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths = std::env::args().skip(1).collect::<Vec<_>>();
    if paths.is_empty() {
        return Err("usage: cifar10_local <batch.bin> [batch.bin ...]".into());
    }
    let dataset = load_cifar10_files(&paths)?;
    let batches = BatchIter::new(dataset.labels.len(), 32, 0, true, false)?;
    println!(
        "loaded {} CIFAR-10 images of shape [3, 32, 32] in {} deterministic batches",
        dataset.labels.len(),
        batches.count()
    );
    Ok(())
}
