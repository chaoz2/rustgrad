//! Checked, owned file-backed byte storage. This is copying I/O, not mmap.
use crate::{DType, Shape, TensorData};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

#[derive(Debug)]
pub enum FileBufferError {
    Io(std::io::Error),
    Bounds,
    Overflow,
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
pub struct FileBuffer {
    file: File,
    len: usize,
    access: FileAccess,
}
impl FileBuffer {
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
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn read(&mut self, offset: usize, out: &mut [u8]) -> Result<(), FileBufferError> {
        self.window(offset, out.len())?;
        self.file.seek(SeekFrom::Start(offset as u64))?;
        self.file.read_exact(out)?;
        Ok(())
    }
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), FileBufferError> {
        if self.access == FileAccess::ReadOnly {
            return Err(FileBufferError::ReadOnly);
        }
        self.window(offset, bytes.len())?;
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
}
