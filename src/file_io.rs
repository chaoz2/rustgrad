use std::{
    fs,
    io::{self, Read, Write},
    path::Path,
};

#[derive(Debug)]
pub(crate) enum ExactFileError {
    Io {
        operation: &'static str,
        source: io::Error,
    },
    Limit {
        actual: u64,
        maximum: usize,
    },
    Allocation,
    InvalidFileName,
    StagingExhausted,
}

pub(crate) fn read_file_bytes_bounded(
    path: impl AsRef<Path>,
    maximum: usize,
) -> Result<Vec<u8>, ExactFileError> {
    read_file_bytes_bounded_after_metadata(path.as_ref(), maximum, || {})
}

fn read_file_bytes_bounded_after_metadata(
    path: &Path,
    maximum: usize,
    after_metadata: impl FnOnce(),
) -> Result<Vec<u8>, ExactFileError> {
    let metadata = fs::metadata(path).map_err(|error| ExactFileError::Io {
        operation: "inspect",
        source: error,
    })?;
    if metadata.len() > u64::try_from(maximum).unwrap_or(u64::MAX) {
        return Err(ExactFileError::Limit {
            actual: metadata.len(),
            maximum,
        });
    }
    after_metadata();
    let file = fs::File::open(path).map_err(|error| ExactFileError::Io {
        operation: "open",
        source: error,
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(maximum.min(64 << 10))
        .map_err(|_| ExactFileError::Allocation)?;
    file.take(u64::try_from(maximum).unwrap_or(u64::MAX).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| ExactFileError::Io {
            operation: "read",
            source: error,
        })?;
    if bytes.len() > maximum {
        return Err(ExactFileError::Limit {
            actual: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            maximum,
        });
    }
    Ok(bytes)
}

pub(crate) fn replace_file_bytes_atomically(
    path: impl AsRef<Path>,
    bytes: &[u8],
) -> Result<(), ExactFileError> {
    let path = path.as_ref();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(ExactFileError::InvalidFileName)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut staging = None;
    for attempt in 0..128u16 {
        let candidate = parent.join(format!(
            ".{file_name}.rustgrad-{}-{attempt}.tmp",
            std::process::id()
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                let result = file
                    .write_all(bytes)
                    .map_err(|error| ExactFileError::Io {
                        operation: "write staging file",
                        source: error,
                    })
                    .and_then(|()| {
                        file.sync_all().map_err(|error| ExactFileError::Io {
                            operation: "sync staging file",
                            source: error,
                        })
                    });
                if let Err(error) = result {
                    drop(file);
                    let _ = fs::remove_file(&candidate);
                    return Err(error);
                }
                drop(file);
                staging = Some(candidate);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(ExactFileError::Io {
                    operation: "create staging file",
                    source: error,
                });
            }
        }
    }
    let staging = staging.ok_or(ExactFileError::StagingExhausted)?;
    fs::rename(&staging, path).map_err(|error| {
        let _ = fs::remove_file(&staging);
        ExactFileError::Io {
            operation: "replace destination",
            source: error,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn bounded_read_rejects_a_file_that_grows_after_metadata() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let ordinal = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rustgrad-growing-file-{}-{ordinal}",
            std::process::id()
        ));
        fs::write(&path, b"1234").unwrap();
        let result = read_file_bytes_bounded_after_metadata(&path, 4, || {
            let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(b"5").unwrap();
            file.sync_all().unwrap();
        });
        assert!(matches!(
            result,
            Err(ExactFileError::Limit {
                actual: 5,
                maximum: 4
            })
        ));
        fs::remove_file(path).unwrap();
    }
}
