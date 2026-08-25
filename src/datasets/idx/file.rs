//! Bounded local filesystem adapter for the exact IDX parser.

use super::{MnistIdx, be32, parse_mnist_idx};
use std::{fmt, path::Path};

/// Limits checked before loading local IDX file contents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MnistIdxReadLimits {
    pub max_file_bytes: usize,
    pub max_items: usize,
    pub max_rows: usize,
    pub max_cols: usize,
}

impl Default for MnistIdxReadLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 256 * 1024 * 1024,
            max_items: 1_000_000,
            max_rows: 16_384,
            max_cols: 16_384,
        }
    }
}

/// File I/O, declared-limit, or exact IDX format rejection.
#[derive(Clone, Debug, PartialEq)]
pub enum MnistIdxFileError {
    Io {
        path: String,
        kind: std::io::ErrorKind,
    },
    Limit {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    Format(crate::Error),
}

impl fmt::Display for MnistIdxFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MNIST IDX file error: {self:?}")
    }
}
impl std::error::Error for MnistIdxFileError {}

/// Reads two local IDX files using conservative default limits.
pub fn load_mnist_idx_files(
    images: impl AsRef<Path>,
    labels: impl AsRef<Path>,
) -> Result<MnistIdx, MnistIdxFileError> {
    load_mnist_idx_files_with_limits(images, labels, MnistIdxReadLimits::default())
}

/// Preflights local file sizes and declared image dimensions before delegating
/// to [`super::parse_mnist_idx`]. It never returns a partially decoded pair.
pub fn load_mnist_idx_files_with_limits(
    images: impl AsRef<Path>,
    labels: impl AsRef<Path>,
    limits: MnistIdxReadLimits,
) -> Result<MnistIdx, MnistIdxFileError> {
    let images = read_limited(images.as_ref(), limits)?;
    let labels = read_limited(labels.as_ref(), limits)?;
    validate_declared(&images, &labels, limits)?;
    parse_mnist_idx(&images, &labels).map_err(MnistIdxFileError::Format)
}

fn read_limited(path: &Path, limits: MnistIdxReadLimits) -> Result<Vec<u8>, MnistIdxFileError> {
    let display = path.display().to_string();
    let metadata = std::fs::metadata(path).map_err(|error| MnistIdxFileError::Io {
        path: display.clone(),
        kind: error.kind(),
    })?;
    let actual = usize::try_from(metadata.len()).map_err(|_| MnistIdxFileError::Limit {
        field: "file_bytes",
        actual: usize::MAX,
        maximum: limits.max_file_bytes,
    })?;
    if actual > limits.max_file_bytes {
        return Err(MnistIdxFileError::Limit {
            field: "file_bytes",
            actual,
            maximum: limits.max_file_bytes,
        });
    }
    std::fs::read(path).map_err(|error| MnistIdxFileError::Io {
        path: display,
        kind: error.kind(),
    })
}

fn validate_declared(
    images: &[u8],
    labels: &[u8],
    limits: MnistIdxReadLimits,
) -> Result<(), MnistIdxFileError> {
    if images.len() < 16 || labels.len() < 8 {
        return Ok(());
    }
    let count = be32(&images[4..8]).map_err(MnistIdxFileError::Format)?;
    let label_count = be32(&labels[4..8]).map_err(MnistIdxFileError::Format)?;
    let rows = be32(&images[8..12]).map_err(MnistIdxFileError::Format)?;
    let cols = be32(&images[12..16]).map_err(MnistIdxFileError::Format)?;
    for (field, actual, maximum) in [
        ("items", count, limits.max_items),
        ("label_items", label_count, limits.max_items),
        ("rows", rows, limits.max_rows),
        ("cols", cols, limits.max_cols),
    ] {
        if actual > maximum {
            return Err(MnistIdxFileError::Limit {
                field,
                actual,
                maximum,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn fixture(count: u32, rows: u32, cols: u32) -> (Vec<u8>, Vec<u8>) {
        let pixels = (count * rows * cols) as usize;
        let mut images = Vec::new();
        images.extend_from_slice(&2051u32.to_be_bytes());
        images.extend_from_slice(&count.to_be_bytes());
        images.extend_from_slice(&rows.to_be_bytes());
        images.extend_from_slice(&cols.to_be_bytes());
        images.extend((0..pixels).map(|value| value as u8));
        let mut labels = Vec::new();
        labels.extend_from_slice(&2049u32.to_be_bytes());
        labels.extend_from_slice(&count.to_be_bytes());
        labels.extend((0..count).map(|value| value as u8));
        (images, labels)
    }

    fn files(images: &[u8], labels: &[u8]) -> (std::path::PathBuf, std::path::PathBuf) {
        let suffix = NEXT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("rustgrad-idx-{suffix}"));
        std::fs::create_dir_all(&root).unwrap();
        let image_path = root.join("images.idx3-ubyte");
        let label_path = root.join("labels.idx1-ubyte");
        std::fs::write(&image_path, images).unwrap();
        std::fs::write(&label_path, labels).unwrap();
        (image_path, label_path)
    }

    #[test]
    fn local_idx_files_are_bounded_and_delegate_exact_layout_validation() {
        let (images, labels) = fixture(2, 2, 2);
        let (image_path, label_path) = files(&images, &labels);
        let loaded = load_mnist_idx_files(&image_path, &label_path).unwrap();
        assert_eq!(loaded.images.to_le_bytes().unwrap(), images[16..]);
        assert_eq!(loaded.labels.to_le_bytes().unwrap(), labels[8..]);

        let limits = MnistIdxReadLimits {
            max_items: 1,
            ..MnistIdxReadLimits::default()
        };
        assert!(matches!(
            load_mnist_idx_files_with_limits(&image_path, &label_path, limits),
            Err(MnistIdxFileError::Limit { field: "items", .. })
        ));
        std::fs::write(&label_path, &labels[..labels.len() - 1]).unwrap();
        assert!(matches!(
            load_mnist_idx_files(&image_path, &label_path),
            Err(MnistIdxFileError::Format(_))
        ));
        let root = image_path.parent().unwrap().to_path_buf();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_idx_files_are_a_valid_zero_batch_dataset() {
        let (images, labels) = fixture(0, 28, 28);
        let (image_path, label_path) = files(&images, &labels);
        let loaded = load_mnist_idx_files(&image_path, &label_path).unwrap();
        assert_eq!(loaded.images.shape().dims(), &[0, 1, 28, 28]);
        let root = image_path.parent().unwrap().to_path_buf();
        std::fs::remove_dir_all(root).unwrap();
    }
}
