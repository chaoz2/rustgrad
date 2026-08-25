//! Bounded filesystem adapters around the canonical `.npy` byte codec.
//!
//! This module owns paths and staged replacement only. NPY syntax, dtype, and
//! layout policy remain in [`super::npy`].

use super::{NpyError, NpyReadLimits, decode_npy_with_limits, encode_npy};
use crate::TensorData;
use std::{
    fmt, fs,
    io::{self, Read, Write},
    path::Path,
};

/// A typed local-file failure distinct from an NPY format failure.
#[derive(Debug)]
pub enum NpyFileError {
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
    Format(NpyError),
}

impl fmt::Display for NpyFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => write!(f, "npy file {operation} failed: {kind:?}"),
            Self::Format(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for NpyFileError {}

fn io_error(operation: &'static str, error: io::Error) -> NpyFileError {
    NpyFileError::Io {
        operation,
        kind: error.kind(),
    }
}

/// Loads a bounded local NPY v1/v2 file with [`NpyReadLimits::default`].
pub fn load_npy_file(path: impl AsRef<Path>) -> Result<TensorData, NpyFileError> {
    load_npy_file_with_limits(path, NpyReadLimits::default())
}

/// Loads a local NPY v1/v2 file under explicit file, header, rank, and element
/// limits. The file is read into an independent byte buffer; no mapping or
/// zero-copy tensor ownership is exposed.
pub fn load_npy_file_with_limits(
    path: impl AsRef<Path>,
    limits: NpyReadLimits,
) -> Result<TensorData, NpyFileError> {
    let path = path.as_ref();
    let metadata = fs::metadata(path).map_err(|error| io_error("inspect", error))?;
    if metadata.len() > u64::try_from(limits.max_file_bytes).unwrap_or(u64::MAX) {
        return Err(NpyFileError::Format(NpyError::Limit {
            limit: "file bytes",
            actual: usize::try_from(metadata.len()).unwrap_or(usize::MAX),
            maximum: limits.max_file_bytes,
        }));
    }
    let file = fs::File::open(path).map_err(|error| io_error("open", error))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(limits.max_file_bytes.min(64 << 10))
        .map_err(|_| NpyFileError::Format(NpyError::Codec))?;
    file.take(
        u64::try_from(limits.max_file_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    )
    .read_to_end(&mut bytes)
    .map_err(|error| io_error("read", error))?;
    decode_npy_with_limits(&bytes, limits).map_err(NpyFileError::Format)
}

/// Deterministically encodes `tensor` and atomically replaces `path` after a
/// staged same-directory write. An encoding, create, write, sync, or rename
/// failure leaves an existing target intact.
pub fn save_npy_file(path: impl AsRef<Path>, tensor: &TensorData) -> Result<(), NpyFileError> {
    let path = path.as_ref();
    let bytes = encode_npy(tensor).map_err(NpyFileError::Format)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(NpyFileError::Io {
            operation: "validate path",
            kind: io::ErrorKind::InvalidInput,
        })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temp = None;
    for attempt in 0..128u16 {
        let candidate = parent.join(format!(
            ".{name}.rustgrad-{}-{attempt}.tmp",
            std::process::id()
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                let result = (|| {
                    file.write_all(&bytes)
                        .map_err(|error| io_error("write", error))?;
                    file.sync_all().map_err(|error| io_error("sync", error))
                })();
                if let Err(error) = result {
                    drop(file);
                    let _ = fs::remove_file(&candidate);
                    return Err(error);
                }
                temp = Some(candidate);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error("create staging file", error)),
        }
    }
    let temp = temp.ok_or(NpyFileError::Io {
        operation: "create staging file",
        kind: io::ErrorKind::AlreadyExists,
    })?;
    fs::rename(&temp, path).map_err(|error| {
        let _ = fs::remove_file(&temp);
        io_error("replace", error)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn directory() -> std::path::PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rustgrad-npy-file-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn file_round_trip_is_deterministic_and_preserves_special_bits() {
        let path = directory().join("specials.npy");
        let tensor =
            TensorData::from_le_bytes([2], DType::F32, &[0, 0, 0, 0x80, 0x01, 0, 0xc0, 0x7f])
                .unwrap();
        save_npy_file(&path, &tensor).unwrap();
        let first = fs::read(&path).unwrap();
        save_npy_file(&path, &tensor).unwrap();
        assert_eq!(first, fs::read(&path).unwrap());
        assert_eq!(
            load_npy_file(&path).unwrap().to_le_bytes().unwrap(),
            tensor.to_le_bytes().unwrap()
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn file_limits_and_format_failures_are_typed() {
        let path = directory().join("bounded.npy");
        fs::write(&path, b"not an array").unwrap();
        assert!(matches!(
            load_npy_file(&path),
            Err(NpyFileError::Format(NpyError::Magic))
        ));
        assert!(matches!(
            load_npy_file_with_limits(
                &path,
                NpyReadLimits {
                    max_file_bytes: 2,
                    ..NpyReadLimits::default()
                }
            ),
            Err(NpyFileError::Format(NpyError::Limit {
                limit: "file bytes",
                ..
            }))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_replacement_keeps_the_existing_target_and_cleans_staging() {
        let directory = directory();
        let target = directory.join("target.npy");
        fs::create_dir(&target).unwrap();
        let tensor = TensorData::from_le_bytes([1], DType::U8, &[7]).unwrap();
        assert!(matches!(
            save_npy_file(&target, &tensor),
            Err(NpyFileError::Io {
                operation: "replace",
                ..
            })
        ));
        assert!(target.is_dir());
        assert!(
            !fs::read_dir(&directory)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".target.npy.rustgrad-")
                })
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
