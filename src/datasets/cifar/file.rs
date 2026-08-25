//! Bounded local filesystem adapter for canonical CIFAR-10 binary batches.

use super::{Cifar10, RECORD_BYTES, parse_cifar10};
use crate::{DType, TensorData};
use std::{
    fmt,
    path::{Path, PathBuf},
};

/// Limits checked before reading and concatenating local CIFAR-10 batches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cifar10ReadLimits {
    pub max_files: usize,
    pub max_total_bytes: usize,
    pub max_records: usize,
}

impl Default for Cifar10ReadLimits {
    fn default() -> Self {
        Self {
            max_files: 64,
            max_total_bytes: 256 * 1024 * 1024,
            max_records: 1_000_000,
        }
    }
}

/// Local I/O, declared-limit, or canonical-record rejection.
#[derive(Clone, Debug, PartialEq)]
pub enum Cifar10FileError {
    Io {
        path: String,
        kind: std::io::ErrorKind,
    },
    Limit {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    Format {
        path: String,
        error: crate::Error,
    },
}

impl fmt::Display for Cifar10FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CIFAR-10 file error: {self:?}")
    }
}
impl std::error::Error for Cifar10FileError {}

/// Reads local uncompressed CIFAR-10 batches in caller-provided deterministic
/// order using conservative default limits. An empty path slice is a legal
/// zero-record dataset with the same NCHW descriptor as [`parse_cifar10`].
pub fn load_cifar10_files(
    paths: &[impl AsRef<Path>],
) -> std::result::Result<Cifar10, Cifar10FileError> {
    load_cifar10_files_with_limits(paths, Cifar10ReadLimits::default())
}

/// Reads local uncompressed CIFAR-10 batches after preflighting all declared
/// byte and record limits. Paths are neither sorted nor normalized: their
/// input slice order is the exact concatenation order.
pub fn load_cifar10_files_with_limits(
    paths: &[impl AsRef<Path>],
    limits: Cifar10ReadLimits,
) -> std::result::Result<Cifar10, Cifar10FileError> {
    let (plans, total_records) = preflight_paths(paths, limits)?;
    let mut batches = Vec::with_capacity(plans.len());
    for FilePlan {
        path,
        display,
        bytes,
        records,
    } in plans
    {
        let contents = std::fs::read(path).map_err(|error| Cifar10FileError::Io {
            path: display.clone(),
            kind: error.kind(),
        })?;
        if contents.len() != bytes {
            return Err(Cifar10FileError::Format {
                path: display,
                error: crate::datasets::bad("CIFAR-10 file changed while reading"),
            });
        }
        batches.push((display, parse_cifar10(&contents, records)));
    }

    concatenate_batches(batches, total_records, limits.max_total_bytes)
}

struct FilePlan {
    path: PathBuf,
    display: String,
    bytes: usize,
    records: usize,
}

/// Plans every caller-provided path and validates aggregate constraints before
/// the loader begins its separate read/parse pass.
fn preflight_paths(
    paths: &[impl AsRef<Path>],
    limits: Cifar10ReadLimits,
) -> std::result::Result<(Vec<FilePlan>, usize), Cifar10FileError> {
    limited("files", paths.len(), limits.max_files)?;
    let mut total_bytes = 0usize;
    let mut total_records = 0usize;
    let mut plans = Vec::with_capacity(paths.len());
    for path in paths {
        let path = path.as_ref();
        let display = path.display().to_string();
        let metadata = std::fs::metadata(path).map_err(|error| Cifar10FileError::Io {
            path: display.clone(),
            kind: error.kind(),
        })?;
        let bytes = usize::try_from(metadata.len()).map_err(|_| Cifar10FileError::Limit {
            field: "total bytes",
            actual: usize::MAX,
            maximum: limits.max_total_bytes,
        })?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or(Cifar10FileError::Limit {
                field: "total bytes",
                actual: usize::MAX,
                maximum: limits.max_total_bytes,
            })?;
        limited("total bytes", total_bytes, limits.max_total_bytes)?;
        if bytes % RECORD_BYTES != 0 {
            return Err(Cifar10FileError::Format {
                path: display,
                error: crate::datasets::bad(format!(
                    "CIFAR-10 file byte length {bytes} is not a whole {RECORD_BYTES}-byte record"
                )),
            });
        }
        let records = bytes / RECORD_BYTES;
        total_records = total_records
            .checked_add(records)
            .ok_or(Cifar10FileError::Limit {
                field: "records",
                actual: usize::MAX,
                maximum: limits.max_records,
            })?;
        limited("records", total_records, limits.max_records)?;
        plans.push(FilePlan {
            path: path.to_path_buf(),
            display,
            bytes,
            records,
        });
    }

    Ok((plans, total_records))
}

fn concatenate_batches(
    batches: Vec<(String, crate::Result<Cifar10>)>,
    total_records: usize,
    max_total_bytes: usize,
) -> std::result::Result<Cifar10, Cifar10FileError> {
    let image_bytes =
        total_records
            .checked_mul(super::IMAGE_BYTES)
            .ok_or(Cifar10FileError::Limit {
                field: "image bytes",
                actual: usize::MAX,
                maximum: max_total_bytes,
            })?;
    let mut images = Vec::with_capacity(image_bytes);
    let mut labels = Vec::with_capacity(total_records);
    for (path, batch) in batches {
        let batch = batch.map_err(|error| Cifar10FileError::Format { path, error })?;
        images.extend(
            batch
                .images
                .to_le_bytes()
                .map_err(|error| Cifar10FileError::Format {
                    path: "concatenation".to_string(),
                    error,
                })?,
        );
        labels.extend(
            batch
                .labels
                .to_le_bytes()
                .map_err(|error| Cifar10FileError::Format {
                    path: "concatenation".to_string(),
                    error,
                })?,
        );
    }
    Ok(Cifar10 {
        images: TensorData::from_le_bytes(
            [total_records, super::CHANNELS, super::HEIGHT, super::WIDTH],
            DType::U8,
            &images,
        )
        .map_err(|error| Cifar10FileError::Format {
            path: "concatenation".to_string(),
            error,
        })?,
        labels: TensorData::from_le_bytes([total_records], DType::U8, &labels).map_err(
            |error| Cifar10FileError::Format {
                path: "concatenation".to_string(),
                error,
            },
        )?,
    })
}

fn limited(
    field: &'static str,
    actual: usize,
    maximum: usize,
) -> std::result::Result<(), Cifar10FileError> {
    if actual > maximum {
        return Err(Cifar10FileError::Limit {
            field,
            actual,
            maximum,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn record(label: u8, value: u8) -> Vec<u8> {
        let mut record = vec![label];
        record.extend(std::iter::repeat_n(value, RECORD_BYTES - 1));
        record
    }
    fn files(contents: &[&[u8]]) -> (std::path::PathBuf, Vec<std::path::PathBuf>) {
        let root = std::env::temp_dir().join(format!(
            "rustgrad-cifar-files-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let paths = contents
            .iter()
            .enumerate()
            .map(|(index, content)| {
                let path = root.join(format!("batch-{index}.bin"));
                std::fs::write(&path, content).unwrap();
                path
            })
            .collect();
        (root, paths)
    }

    #[test]
    fn local_cifar_files_preserve_input_order_and_limits() {
        let first = record(2, 11);
        let second = record(4, 22);
        let (root, paths) = files(&[&first, &second]);
        let data = load_cifar10_files(&paths).unwrap();
        assert_eq!(data.labels.to_le_bytes().unwrap(), vec![2, 4]);
        assert_eq!(data.images.to_le_bytes().unwrap()[0], 11);
        assert_eq!(
            data.images.to_le_bytes().unwrap()[super::super::IMAGE_BYTES],
            22
        );
        assert!(matches!(
            load_cifar10_files_with_limits(
                &paths,
                Cifar10ReadLimits {
                    max_files: 1,
                    ..Cifar10ReadLimits::default()
                }
            ),
            Err(Cifar10FileError::Limit { field: "files", .. })
        ));
        assert!(matches!(
            load_cifar10_files_with_limits(
                &paths,
                Cifar10ReadLimits {
                    max_records: 1,
                    ..Cifar10ReadLimits::default()
                }
            ),
            Err(Cifar10FileError::Limit {
                field: "records",
                ..
            })
        ));
        assert!(matches!(
            load_cifar10_files_with_limits(
                &paths,
                Cifar10ReadLimits {
                    max_total_bytes: RECORD_BYTES,
                    ..Cifar10ReadLimits::default()
                }
            ),
            Err(Cifar10FileError::Limit {
                field: "total bytes",
                ..
            })
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preflight_rejects_a_later_structural_file_before_the_read_parse_pass() {
        let valid = record(1, 7);
        let malformed = &valid[..valid.len() - 1];
        let (root, paths) = files(&[&valid, malformed]);

        // `preflight_paths` is deliberately pure metadata/limit planning. A
        // later whole-record failure therefore returns before `load_*` reaches
        // its separate filesystem read/parse loop for the valid first path.
        assert!(matches!(
            preflight_paths(&paths, Cifar10ReadLimits::default()),
            Err(Cifar10FileError::Format { path, .. }) if path.ends_with("batch-1.bin")
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn local_cifar_files_reject_incomplete_or_invalid_records_and_allow_empty_input() {
        let invalid = record(10, 0);
        let truncated = &invalid[..invalid.len() - 1];
        let (root, paths) = files(&[truncated]);
        assert!(matches!(
            load_cifar10_files(&paths),
            Err(Cifar10FileError::Format { .. })
        ));
        std::fs::remove_dir_all(root).unwrap();
        let (root, paths) = files(&[&invalid]);
        assert!(matches!(
            load_cifar10_files(&paths),
            Err(Cifar10FileError::Format { .. })
        ));
        std::fs::remove_dir_all(root).unwrap();
        let empty: [&std::path::Path; 0] = [];
        assert_eq!(
            load_cifar10_files(&empty).unwrap().images.shape().dims(),
            &[0, 3, 32, 32]
        );
    }
}
