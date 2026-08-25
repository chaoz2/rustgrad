//! Inspect a bounded local MNIST IDX pair before using it in the documented
//! train/resume/evaluate acceptance workflow.
//!
//! `cargo run --example mnist_idx_local -- images.idx3-ubyte labels.idx1-ubyte`

use rustgrad::{BatchIter, load_mnist_idx_files};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let images = args
        .next()
        .ok_or("usage: mnist_idx_local <images> <labels>")?;
    let labels = args
        .next()
        .ok_or("usage: mnist_idx_local <images> <labels>")?;
    if args.next().is_some() {
        return Err("usage: mnist_idx_local <images> <labels>".into());
    }
    let dataset = load_mnist_idx_files(images, labels)?;
    let batches = BatchIter::new(dataset.labels.len(), 32, 0, true, false)?;
    println!(
        "loaded {} images of shape [1, {}, {}] in {} deterministic batches",
        dataset.labels.len(),
        dataset.rows,
        dataset.cols,
        batches.count()
    );
    Ok(())
}
