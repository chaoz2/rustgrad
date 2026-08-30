//! Checked, owned file-backed byte storage. This is copying I/O, not mmap.
use crate::{DType, Shape, TensorData};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

#[derive(Debug)]
pub enum FileBufferError {
    Io(std::io::Error),
    Bounds,
    Overflow,
    Limit { actual: u64, maximum: usize },
    MisalignedTensorBytes { bytes: usize, itemsize: usize },
    ReadOnly,
    Truncated { expected: usize, actual: usize },
    Tensor(crate::Error),
}
impl From<std::io::Error> for FileBufferError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<crate::Error> for FileBufferError {
    fn from(e: crate::Error) -> Self {
        Self::Tensor(e)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileAccess {
    ReadOnly,
    ReadWrite,
}

/// Resource limit for an owned raw dense tensor file read.
///
/// This bounds only a flat canonical-byte file. It does not add a lazy,
/// mapped, device-backed, or native-endian tensor representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileTensorReadLimits {
    pub max_file_bytes: usize,
}

impl Default for FileTensorReadLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 1 << 30,
        }
    }
}

pub struct FileBuffer {
    file: File,
    len: usize,
    access: FileAccess,
}
impl FileBuffer {
    /// Copies a bounded flat raw dense file into canonical [`TensorData`].
    ///
    /// The shape is inferred as one dimension from the full file length and
    /// `dtype` item width. The input is interpreted as RustGrad's canonical
    /// little-endian dense representation, so raw float bits are retained.
    /// A nonempty file whose length is not a whole number of items is rejected.
    pub fn read_tensor_file(
        path: impl AsRef<Path>,
        dtype: DType,
        limits: FileTensorReadLimits,
    ) -> Result<TensorData, FileBufferError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;
        if metadata.len() > u64::try_from(limits.max_file_bytes).unwrap_or(u64::MAX) {
            return Err(FileBufferError::Limit {
                actual: metadata.len(),
                maximum: limits.max_file_bytes,
            });
        }
        let bytes = usize::try_from(metadata.len()).map_err(|_| FileBufferError::Overflow)?;
        let itemsize = dtype.itemsize();
        if bytes % itemsize != 0 {
            return Err(FileBufferError::MisalignedTensorBytes { bytes, itemsize });
        }
        let mut file = Self::open(path, FileAccess::ReadOnly, bytes)?;
        file.read_tensor([bytes / itemsize], dtype)
    }

    /// Writes canonical tensor bytes through a staged same-directory replace.
    ///
    /// Encoding completes before the target is opened. The staging file is
    /// created exclusively, synced, and removed if writing or replacement
    /// fails; an existing target is changed only by the final rename. This is
    /// owned copying I/O, not a mapped or lazy tensor backing.
    pub fn save_tensor_file(
        path: impl AsRef<Path>,
        tensor: &TensorData,
    ) -> Result<(), FileBufferError> {
        let path = path.as_ref();
        let bytes = tensor.to_le_bytes()?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                FileBufferError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "raw tensor path must have a UTF-8 filename",
                ))
            })?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut staged = None;
        for attempt in 0..128u16 {
            let candidate = parent.join(format!(
                ".{name}.rustgrad-{}-{attempt}.tmp",
                std::process::id()
            ));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(mut file) => {
                    let result = (|| {
                        file.write_all(&bytes)?;
                        file.sync_all()
                    })();
                    if let Err(error) = result {
                        drop(file);
                        let _ = fs::remove_file(&candidate);
                        return Err(FileBufferError::Io(error));
                    }
                    staged = Some(candidate);
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(FileBufferError::Io(error)),
            }
        }
        let staged = staged.ok_or_else(|| {
            FileBufferError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not create unique raw tensor staging file",
            ))
        })?;
        fs::rename(&staged, path).map_err(|error| {
            let _ = fs::remove_file(&staged);
            FileBufferError::Io(error)
        })
    }

    pub fn create(path: impl AsRef<Path>, len: usize) -> Result<Self, FileBufferError> {
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)?;
        file.set_len(u64::try_from(len).map_err(|_| FileBufferError::Overflow)?)?;
        Ok(Self {
            file,
            len,
            access: FileAccess::ReadWrite,
        })
    }
    pub fn open(
        path: impl AsRef<Path>,
        access: FileAccess,
        len: usize,
    ) -> Result<Self, FileBufferError> {
        let file = OpenOptions::new()
            .read(true)
            .write(access == FileAccess::ReadWrite)
            .open(path)?;
        let actual =
            usize::try_from(file.metadata()?.len()).map_err(|_| FileBufferError::Overflow)?;
        if actual < len {
            return Err(FileBufferError::Truncated {
                expected: len,
                actual,
            });
        }
        Ok(Self { file, len, access })
    }
    fn window(&self, offset: usize, len: usize) -> Result<(), FileBufferError> {
        let end = offset.checked_add(len).ok_or(FileBufferError::Overflow)?;
        if end > self.len {
            Err(FileBufferError::Bounds)
        } else {
            Ok(())
        }
    }
    /// Reject a backing file truncated after this owned descriptor was opened.
    ///
    /// The logical extent is fixed at construction time. Without this check, a
    /// logically in-range write after an external truncate could extend the
    /// backing file and silently publish bytes outside the still-valid view.
    fn validate_backing_len(&self) -> Result<(), FileBufferError> {
        let actual =
            usize::try_from(self.file.metadata()?.len()).map_err(|_| FileBufferError::Overflow)?;
        if actual < self.len {
            Err(FileBufferError::Truncated {
                expected: self.len,
                actual,
            })
        } else {
            Ok(())
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn read(&mut self, offset: usize, out: &mut [u8]) -> Result<(), FileBufferError> {
        self.window(offset, out.len())?;
        if !out.is_empty() {
            self.validate_backing_len()?;
        }
        self.file.seek(SeekFrom::Start(offset as u64))?;
        self.file.read_exact(out)?;
        Ok(())
    }
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), FileBufferError> {
        if self.access == FileAccess::ReadOnly {
            return Err(FileBufferError::ReadOnly);
        }
        self.window(offset, bytes.len())?;
        if !bytes.is_empty() {
            self.validate_backing_len()?;
        }
        self.file.seek(SeekFrom::Start(offset as u64))?;
        self.file.write_all(bytes)?;
        Ok(())
    }
    pub fn resize(&mut self, len: usize) -> Result<(), FileBufferError> {
        if self.access == FileAccess::ReadOnly {
            return Err(FileBufferError::ReadOnly);
        }
        self.file
            .set_len(u64::try_from(len).map_err(|_| FileBufferError::Overflow)?)?;
        self.len = len;
        Ok(())
    }
    pub fn sync(&self) -> Result<(), FileBufferError> {
        self.file.sync_all()?;
        Ok(())
    }
    pub fn write_tensor(&mut self, tensor: &TensorData) -> Result<(), FileBufferError> {
        let bytes = tensor.to_le_bytes()?;
        if bytes.len() != self.len {
            return Err(FileBufferError::Bounds);
        }
        self.write(0, &bytes)
    }
    pub fn read_tensor(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<TensorData, FileBufferError> {
        let shape = shape.into();
        let expected = shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or(FileBufferError::Overflow)?;
        if expected != self.len {
            return Err(FileBufferError::Bounds);
        }
        let mut bytes = vec![0; self.len];
        self.read(0, &mut bytes)?;
        Ok(TensorData::from_le_bytes(shape, dtype, &bytes)?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn path(n: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rustgrad-file-runtime-{n}-{}", std::process::id()))
    }
    #[test]
    fn roundtrip_and_bounds() {
        let p = path("a");
        let mut b = FileBuffer::create(&p, 4).unwrap();
        b.write(0, &[1, 2, 3, 4]).unwrap();
        b.sync().unwrap();
        assert!(matches!(b.write(3, &[1, 2],), Err(FileBufferError::Bounds)));
        drop(b);
        let mut b = FileBuffer::open(&p, FileAccess::ReadOnly, 4).unwrap();
        let mut out = [0; 4];
        b.read(0, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        assert!(matches!(b.write(0, &[0]), Err(FileBufferError::ReadOnly)));
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn raw_tensor_file_is_bounded_aligned_and_preserves_float_bits() {
        // This mirrors tinygrad's TestPathTensor contract: a raw byte file
        // becomes a flat typed tensor only when its byte count is exact.
        let p = path("raw-tensor");
        let raw = [0, 0, 0, 0x80, 0x34, 0x12, 0xc0, 0x7f];
        std::fs::write(&p, raw).unwrap();
        let tensor = FileBuffer::read_tensor_file(
            &p,
            DType::F32,
            FileTensorReadLimits {
                max_file_bytes: raw.len(),
            },
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[2]);
        assert_eq!(tensor.to_le_bytes().unwrap(), raw);
        assert!(matches!(
            FileBuffer::read_tensor_file(
                &p,
                DType::F32,
                FileTensorReadLimits {
                    max_file_bytes: raw.len() - 1,
                }
            ),
            Err(FileBufferError::Limit { .. })
        ));
        std::fs::write(&p, [1, 2, 3]).unwrap();
        assert!(matches!(
            FileBuffer::read_tensor_file(&p, DType::I16, FileTensorReadLimits::default()),
            Err(FileBufferError::MisalignedTensorBytes { .. })
        ));
        std::fs::remove_file(p).unwrap();

        let empty = path("raw-tensor-empty");
        std::fs::write(&empty, b"").unwrap();
        assert_eq!(
            FileBuffer::read_tensor_file(&empty, DType::U64, FileTensorReadLimits::default())
                .unwrap()
                .shape()
                .dims(),
            &[0]
        );
        std::fs::remove_file(empty).unwrap();
    }

    #[test]
    fn raw_tensor_file_save_is_staged_exact_and_retryable() {
        let directory =
            std::env::temp_dir().join(format!("rustgrad-raw-tensor-save-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let target = directory.join("tensor.bin");
        let occupied = directory.join(format!(".tensor.bin.rustgrad-{}-0.tmp", std::process::id()));
        std::fs::write(&occupied, b"other writer").unwrap();
        let tensor =
            TensorData::from_le_bytes([2], DType::F32, &[0, 0, 0, 0x80, 0x34, 0x12, 0xc0, 0x7f])
                .unwrap();

        std::fs::create_dir(&target).unwrap();
        assert!(FileBuffer::save_tensor_file(&target, &tensor).is_err());
        assert!(target.is_dir());
        assert_eq!(std::fs::read(&occupied).unwrap(), b"other writer");
        assert!(
            !std::fs::read_dir(&directory)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| { entry.file_name().to_string_lossy().ends_with("-1.tmp") })
        );

        std::fs::remove_dir(&target).unwrap();
        FileBuffer::save_tensor_file(&target, &tensor).unwrap();
        assert_eq!(
            std::fs::read(&target).unwrap(),
            tensor.to_le_bytes().unwrap()
        );
        assert_eq!(
            FileBuffer::read_tensor_file(&target, DType::F32, FileTensorReadLimits::default())
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            tensor.to_le_bytes().unwrap()
        );
        std::fs::remove_file(target).unwrap();

        let empty = directory.join("empty.bin");
        let empty_tensor = TensorData::from_le_bytes([0], DType::U8, &[]).unwrap();
        FileBuffer::save_tensor_file(&empty, &empty_tensor).unwrap();
        assert!(std::fs::read(&empty).unwrap().is_empty());
        assert_eq!(
            FileBuffer::read_tensor_file(&empty, DType::U8, FileTensorReadLimits::default())
                .unwrap()
                .shape()
                .dims(),
            &[0]
        );
        std::fs::remove_file(empty).unwrap();
        std::fs::remove_file(occupied).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn tensor_read_preflights_exact_extent_without_changing_file_bytes() {
        let p = path("tensor-preflight");
        let original = 3.5f32.to_le_bytes();
        let mut buffer = FileBuffer::create(&p, original.len()).unwrap();
        buffer.write(0, &original).unwrap();
        buffer.sync().unwrap();

        assert!(matches!(
            buffer.read_tensor([2], DType::F32),
            Err(FileBufferError::Bounds)
        ));
        assert!(matches!(
            buffer.read_tensor([1], DType::F64),
            Err(FileBufferError::Bounds)
        ));
        let mut after_rejections = [0; 4];
        buffer.read(0, &mut after_rejections).unwrap();
        assert_eq!(after_rejections, original);

        let restored = buffer.read_tensor([1], DType::F32).unwrap();
        assert_eq!(restored.to_le_bytes().unwrap(), original);
        drop(buffer);
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn external_truncation_rejects_reads_and_writes_without_extending_the_file() {
        let p = path("external-truncate");
        let original = [10, 20, 30, 40];
        let mut buffer = FileBuffer::create(&p, original.len()).unwrap();
        buffer.write(0, &original).unwrap();
        buffer.sync().unwrap();

        std::fs::OpenOptions::new()
            .write(true)
            .open(&p)
            .unwrap()
            .set_len(2)
            .unwrap();
        let after_truncate = std::fs::read(&p).unwrap();
        assert_eq!(after_truncate, original[..2]);

        let mut empty = [];
        buffer.read(0, &mut empty).unwrap();
        buffer.write(0, &[]).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), after_truncate);

        let mut read = [0; 2];
        assert!(matches!(
            buffer.read(0, &mut read),
            Err(FileBufferError::Truncated {
                expected: 4,
                actual: 2
            })
        ));
        assert!(matches!(
            buffer.write(0, &[99, 98]),
            Err(FileBufferError::Truncated {
                expected: 4,
                actual: 2
            })
        ));
        assert_eq!(std::fs::read(&p).unwrap(), after_truncate);

        drop(buffer);
        std::fs::remove_file(p).unwrap();
    }
}
