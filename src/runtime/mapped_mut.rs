//! Thread-confined mutable mapped-file ownership.
//!
//! This is the writable companion to [`super::mapped::MappedTensor`], but it
//! deliberately does not yield `TensorData` aliases or graph inputs. A single
//! owner exposes only checked copy-in/copy-out windows and explicit `sync`.
//! That gives a later disk allocator a defined persistence and retry boundary
//! without weakening the owned `TensorData`/autograd model today.

use crate::{DType, Shape, TensorData};
use std::{
    fs::{File, OpenOptions},
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
};

#[derive(Debug)]
pub enum MutableMappedFileError {
    Io(io::Error),
    UnsupportedPlatform,
    Overflow,
    Bounds,
    ShapeBytes { expected: usize, actual: usize },
    DType { expected: DType, actual: DType },
    Tensor(crate::Error),
}

impl From<io::Error> for MutableMappedFileError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<crate::Error> for MutableMappedFileError {
    fn from(error: crate::Error) -> Self {
        Self::Tensor(error)
    }
}

/// One thread-confined owner of a shared writable mapping.
///
/// It is intentionally neither `Clone` nor `Send`/`Sync`: RustGrad provides
/// no concurrent or aliasing mutation contract for mapped storage. `create`
/// exclusively creates an exact-length file; `open` never grows an existing
/// file; and this bounded first phase has no resize API. Callers must create a
/// new owner for a different byte extent.
#[derive(Debug)]
pub struct MutableMappedFile {
    path: PathBuf,
    file: File,
    bytes: usize,
    #[cfg(unix)]
    ptr: *mut u8,
    // Rc makes ownership explicitly thread-confined without exposing a raw
    // mutable pointer through a public API.
    _thread_confined: PhantomData<Rc<()>>,
}

impl MutableMappedFile {
    /// Creates a new exact-length mapped file. Existing paths are never
    /// opened, truncated, or replaced by this operation.
    pub fn create(path: impl AsRef<Path>, bytes: usize) -> Result<Self, MutableMappedFileError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.set_len(u64::try_from(bytes).map_err(|_| MutableMappedFileError::Overflow)?)?;
        match Self::from_file(path, file, bytes) {
            Ok(owner) => Ok(owner),
            Err(error) => {
                // The owner was never returned. Remove only the file this
                // call created so a corrected request can be retried.
                let _ = std::fs::remove_file(&path);
                Err(error)
            }
        }
    }

    /// Opens an existing file at its current exact byte extent. It never
    /// extends a file, so a failed bounds check cannot change persistent data.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MutableMappedFileError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let bytes = usize::try_from(file.metadata()?.len())
            .map_err(|_| MutableMappedFileError::Overflow)?;
        Self::from_file(path, file, bytes)
    }

    fn from_file(
        path: PathBuf,
        file: File,
        bytes: usize,
    ) -> Result<Self, MutableMappedFileError> {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // Refuse a second in-process or cooperating-process mutable owner
            // before mapping. The lock is released automatically with `file`.
            if unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) } != 0 {
                return Err(MutableMappedFileError::Io(io::Error::last_os_error()));
            }
        }
        #[cfg(unix)]
        let ptr = if bytes == 0 {
            std::ptr::null_mut()
        } else {
            use std::os::fd::AsRawFd;
            // SAFETY: the file is open read/write and the nonzero mapping is
            // owned exclusively by this value until Drop.
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    bytes,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == MAP_FAILED {
                return Err(MutableMappedFileError::Io(io::Error::last_os_error()));
            }
            ptr.cast()
        };
        #[cfg(not(unix))]
        if bytes != 0 {
            return Err(MutableMappedFileError::UnsupportedPlatform);
        }
        Ok(Self {
            path,
            file,
            bytes,
            #[cfg(unix)]
            ptr,
            _thread_confined: PhantomData,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Copies raw bytes into a checked window. Persistence requires an
    /// explicit [`Self::sync`] after all successful writes.
    pub fn write_bytes(
        &mut self,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), MutableMappedFileError> {
        self.window(offset, bytes.len())?;
        if !bytes.is_empty() {
            #[cfg(unix)]
            // SAFETY: `window` validates the destination against the owned
            // mapping, and the source slice is valid for `bytes.len()`.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len()) };
            #[cfg(not(unix))]
            return Err(MutableMappedFileError::UnsupportedPlatform);
        }
        Ok(())
    }

    /// Copies a typed dense tensor into a checked typed element window.
    pub fn write_tensor(
        &mut self,
        offset_elements: usize,
        shape: impl Into<Shape>,
        dtype: DType,
        value: &TensorData,
    ) -> Result<(), MutableMappedFileError> {
        let shape = shape.into();
        if value.dtype() != dtype {
            return Err(MutableMappedFileError::DType {
                expected: dtype,
                actual: value.dtype(),
            });
        }
        let expected = shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or(MutableMappedFileError::Overflow)?;
        let raw = value.to_le_bytes()?;
        if value.shape() != &shape || raw.len() != expected {
            return Err(MutableMappedFileError::ShapeBytes {
                expected,
                actual: raw.len(),
            });
        }
        let offset = offset_elements
            .checked_mul(dtype.itemsize())
            .ok_or(MutableMappedFileError::Overflow)?;
        self.write_bytes(offset, &raw)
    }

    /// Materializes a typed dense window as independent owned CPU storage.
    pub fn read_tensor(
        &self,
        offset_elements: usize,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<TensorData, MutableMappedFileError> {
        let shape = shape.into();
        let len = shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or(MutableMappedFileError::Overflow)?;
        let offset = offset_elements
            .checked_mul(dtype.itemsize())
            .ok_or(MutableMappedFileError::Overflow)?;
        self.window(offset, len)?;
        let bytes = if len == 0 {
            &[]
        } else {
            #[cfg(unix)]
            // SAFETY: `window` bounds this immutable read within the owned
            // mapping; no mutable reference escapes the owner.
            unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) }
            #[cfg(not(unix))]
            {
                return Err(MutableMappedFileError::UnsupportedPlatform);
            }
        };
        Ok(TensorData::from_le_bytes(shape, dtype, bytes)?)
    }

    /// Flushes the entire mapping, then the file metadata/data state. Failed
    /// sync leaves the owner usable for explicit retry.
    pub fn sync(&self) -> Result<(), MutableMappedFileError> {
        if self.bytes != 0 {
            #[cfg(unix)]
            // SAFETY: this is the exact valid mapping range owned by self.
            if unsafe { msync(self.ptr.cast(), self.bytes, MS_SYNC) } != 0 {
                return Err(MutableMappedFileError::Io(io::Error::last_os_error()));
            }
            #[cfg(not(unix))]
            return Err(MutableMappedFileError::UnsupportedPlatform);
        }
        self.file.sync_all()?;
        Ok(())
    }

    fn window(&self, offset: usize, len: usize) -> Result<(), MutableMappedFileError> {
        let end = offset.checked_add(len).ok_or(MutableMappedFileError::Overflow)?;
        (end <= self.bytes)
            .then_some(())
            .ok_or(MutableMappedFileError::Bounds)
    }
}

#[cfg(unix)]
impl Drop for MutableMappedFile {
    fn drop(&mut self) {
        if self.bytes != 0 {
            // SAFETY: this is the exact mapping pair constructed in from_file.
            unsafe { munmap(self.ptr.cast(), self.bytes) };
        }
    }
}

#[cfg(unix)]
const PROT_READ: i32 = 0x1;
#[cfg(unix)]
const PROT_WRITE: i32 = 0x2;
#[cfg(unix)]
const MAP_SHARED: i32 = 0x1;
#[cfg(unix)]
const MAP_FAILED: *mut std::ffi::c_void = !0usize as *mut std::ffi::c_void;
#[cfg(unix)]
const MS_SYNC: i32 = 0x4;
#[cfg(unix)]
const LOCK_EX: i32 = 0x2;
#[cfg(unix)]
const LOCK_NB: i32 = 0x4;

#[cfg(unix)]
unsafe extern "C" {
    fn mmap(
        address: *mut std::ffi::c_void,
        length: usize,
        protection: i32,
        flags: i32,
        fd: std::ffi::c_int,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn msync(address: *mut std::ffi::c_void, length: usize, flags: i32) -> std::ffi::c_int;
    fn munmap(address: *mut std::ffi::c_void, length: usize) -> std::ffi::c_int;
    fn flock(fd: std::ffi::c_int, operation: std::ffi::c_int) -> std::ffi::c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rustgrad-mapped-mut-{name}-{}", std::process::id()))
    }

    #[test]
    fn typed_windows_persist_only_through_explicit_owned_materialization() {
        let path = path("roundtrip");
        let mut file = MutableMappedFile::create(&path, 8).unwrap();
        let values = TensorData::from_scalars(
            [2],
            DType::F32,
            [crate::Scalar::F(-0.0), crate::Scalar::F(f32::NAN as f64)],
        )
        .unwrap();
        file.write_tensor(0, [2], DType::F32, &values).unwrap();
        file.sync().unwrap();
        assert_eq!(file.read_tensor(0, [2], DType::F32).unwrap().to_le_bytes().unwrap(), values.to_le_bytes().unwrap());
        drop(file);
        let reopened = MutableMappedFile::open(&path).unwrap();
        assert_eq!(reopened.read_tensor(0, [2], DType::F32).unwrap().to_le_bytes().unwrap(), values.to_le_bytes().unwrap());
        drop(reopened);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn create_is_exclusive_and_invalid_windows_do_not_mutate() {
        let path = path("bounds");
        let mut file = MutableMappedFile::create(&path, 4).unwrap();
        file.write_bytes(0, &[1, 2, 3, 4]).unwrap();
        assert!(matches!(file.write_bytes(3, &[9, 9]), Err(MutableMappedFileError::Bounds)));
        assert_eq!(file.read_tensor(0, [4], DType::U8).unwrap().to_le_bytes().unwrap(), vec![1, 2, 3, 4]);
        assert!(MutableMappedFile::create(&path, 4).is_err());
        assert!(MutableMappedFile::open(&path).is_err());
        assert!(matches!(
            file.write_tensor(0, [1], DType::I32, &TensorData::new([1], vec![1.]).unwrap()),
            Err(MutableMappedFileError::DType { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }
}
