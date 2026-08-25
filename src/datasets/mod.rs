//! Deterministic local dataset readers and batch ordering.
//!
//! This module performs no network access, caching, or random augmentation.

mod batch;
mod cifar;
mod idx;

pub use batch::BatchIter;
pub use cifar::{Cifar10, parse_cifar10};
pub use idx::{
    MnistIdx, MnistIdxFileError, MnistIdxReadLimits, load_mnist_idx_files,
    load_mnist_idx_files_with_limits, parse_mnist_idx,
};

use crate::{Error, Result};

pub(super) fn bad(reason: impl Into<String>) -> Error {
    Error::Dataset {
        reason: reason.into(),
    }
}

pub(super) fn checked_exact_len(
    actual: usize,
    count: usize,
    record_bytes: usize,
    format: &str,
) -> Result<()> {
    let expected = count
        .checked_mul(record_bytes)
        .ok_or_else(|| bad(format!("{format} byte length overflow")))?;
    if actual != expected {
        return Err(bad(format!(
            "{format} payload length mismatch: expected {expected} bytes for {count} records, got {actual}"
        )));
    }
    Ok(())
}
