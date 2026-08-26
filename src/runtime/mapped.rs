//! Read-only mapped tensor sources.
//!
//! This is an ownership foundation for file-backed tensor inputs, not a new
//! `TensorData` storage variant.  A mapped source owns an immutable OS mapping
//! behind `Arc`; every CPU, artifact, or capture consumer must explicitly
//! materialize an independent owned [`TensorData`].  That boundary preserves
//! the existing no-alias CPU/autograd contract while keeping backing identity,
//! byte windows, and cleanup explicit for a later disk-device vertical.

use crate::{DType, Shape, TensorData};
use std::{
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(unix)]
use std::fs::File;

static NEXT_BACKING_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MappedBackingId(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MappedTensorPolicy {
    /// The mapping is immutable; materialization is the only supported CPU,
    /// artifact, or capture boundary.
    CopyToOwned,
}

#[derive(Debug)]
pub enum MappedTensorError {
    Io(io::Error),
    UnsupportedPlatform,
    EmptyMapping,
    Overflow,
    Bounds,
    Misaligned { offset: usize, itemsize: usize },
    ShapeBytes { expected: usize, actual: usize },
    Tensor(crate::Error),
}

impl From<io::Error> for MappedTensorError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<crate::Error> for MappedTensorError {
    fn from(error: crate::Error) -> Self {
        Self::Tensor(error)
    }
}

#[derive(Debug)]
struct MappedBacking {
    id: MappedBackingId,
    path: PathBuf,
    bytes: usize,
    #[cfg(unix)]
    ptr: *const u8,
}

// A backing is an immutable MAP_PRIVATE read mapping.  The pointer is never
// exposed mutably and its lifetime is bound to the final Arc drop.
#[cfg(unix)]
unsafe impl Send for MappedBacking {}
#[cfg(unix)]
unsafe impl Sync for MappedBacking {}

#[cfg(unix)]
impl Drop for MappedBacking {
    fn drop(&mut self) {
        if self.bytes != 0 {
            // SAFETY: `ptr` and `bytes` are exactly the successful mmap pair
            // constructed below, and this is the final Arc owner.
            unsafe { munmap(self.ptr.cast(), self.bytes) };
        }
    }
}

/// A checked immutable typed window into a refcounted file mapping.
#[derive(Clone, Debug)]
pub struct MappedTensor {
    backing: Arc<MappedBacking>,
    offset: usize,
    shape: Shape,
    dtype: DType,
}

impl MappedTensor {
    /// Maps a nonempty file as one exact dense tensor window.
    ///
    /// The source bytes use RustGrad's canonical little-endian representation.
    /// Empty files remain representable via [`Self::open_empty`], avoiding a
    /// platform-dependent zero-length `mmap` call.
    pub fn open(
        path: impl AsRef<Path>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self, MappedTensorError> {
        let path = path.as_ref().to_path_buf();
        let shape = shape.into();
        let bytes = std::fs::metadata(&path)?.len();
        let bytes = usize::try_from(bytes).map_err(|_| MappedTensorError::Overflow)?;
        if bytes == 0 {
            return Err(MappedTensorError::EmptyMapping);
        }
        let expected = shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or(MappedTensorError::Overflow)?;
        if expected != bytes {
            return Err(MappedTensorError::ShapeBytes {
                expected,
                actual: bytes,
            });
        }
        let backing = Arc::new(open_backing(&path, bytes)?);
        Ok(Self {
            backing,
            offset: 0,
            shape,
            dtype,
        })
    }

    /// Creates an explicit empty source without issuing an OS mapping call.
    pub fn open_empty(
        path: impl AsRef<Path>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self, MappedTensorError> {
        let path = path.as_ref().to_path_buf();
        let shape = shape.into();
        let expected = shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or(MappedTensorError::Overflow)?;
        let actual = usize::try_from(std::fs::metadata(&path)?.len())
            .map_err(|_| MappedTensorError::Overflow)?;
        if actual != 0 || expected != 0 {
            return Err(MappedTensorError::ShapeBytes {
                expected,
                actual,
            });
        }
        Ok(Self {
            backing: Arc::new(MappedBacking {
                id: MappedBackingId(NEXT_BACKING_ID.fetch_add(1, Ordering::Relaxed)),
                path,
                bytes: 0,
                #[cfg(unix)]
                ptr: std::ptr::null(),
            }),
            offset: 0,
            shape,
            dtype,
        })
    }

    /// Narrows this mapping to an aligned typed byte window.
    pub fn view(
        &self,
        offset_elements: usize,
        shape: impl Into<Shape>,
    ) -> Result<Self, MappedTensorError> {
        let shape = shape.into();
        let itemsize = self.dtype.itemsize();
        let offset = offset_elements
            .checked_mul(itemsize)
            .and_then(|offset| self.offset.checked_add(offset))
            .ok_or(MappedTensorError::Overflow)?;
        if offset % itemsize != 0 {
            return Err(MappedTensorError::Misaligned { offset, itemsize });
        }
        let bytes = shape
            .numel()?
            .checked_mul(itemsize)
            .ok_or(MappedTensorError::Overflow)?;
        let end = offset.checked_add(bytes).ok_or(MappedTensorError::Overflow)?;
        let parent_bytes = self
            .shape
            .numel()?
            .checked_mul(itemsize)
            .ok_or(MappedTensorError::Overflow)?;
        let parent_end = self
            .offset
            .checked_add(parent_bytes)
            .ok_or(MappedTensorError::Overflow)?;
        if end > parent_end || end > self.backing.bytes {
            return Err(MappedTensorError::Bounds);
        }
        Ok(Self {
            backing: Arc::clone(&self.backing),
            offset,
            shape,
            dtype: self.dtype,
        })
    }

    pub fn backing_id(&self) -> MappedBackingId {
        self.backing.id
    }

    pub fn path(&self) -> &Path {
        &self.backing.path
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn policy(&self) -> MappedTensorPolicy {
        MappedTensorPolicy::CopyToOwned
    }

    /// Copies canonical bytes into an owned CPU tensor. This is the required
    /// graph, artifact, and capture boundary; mapped aliasing is never carried
    /// into `TensorData` or reverse-mode state.
    pub fn materialize_cpu(&self) -> Result<TensorData, MappedTensorError> {
        let bytes = self.bytes()?;
        Ok(TensorData::from_le_bytes(self.shape.clone(), self.dtype, bytes)?)
    }

    /// Explicit name for serialization/capture callers: they receive owned
    /// bytes, never a path, pointer, or mapping lifetime.
    pub fn materialize_for_artifact(&self) -> Result<TensorData, MappedTensorError> {
        self.materialize_cpu()
    }

    fn bytes(&self) -> Result<&[u8], MappedTensorError> {
        let len = self
            .shape
            .numel()?
            .checked_mul(self.dtype.itemsize())
            .ok_or(MappedTensorError::Overflow)?;
        if len == 0 {
            return Ok(&[]);
        }
        #[cfg(unix)]
        {
            // SAFETY: the Arc retains the immutable mapping, and `view`
            // validates offset + len against its mapped byte extent.
            return Ok(unsafe { std::slice::from_raw_parts(self.backing.ptr.add(self.offset), len) });
        }
        #[cfg(not(unix))]
        {
            let _ = len;
            Err(MappedTensorError::UnsupportedPlatform)
        }
    }
}

#[cfg(unix)]
fn open_backing(path: &Path, bytes: usize) -> Result<MappedBacking, MappedTensorError> {
    use std::os::fd::AsRawFd;
    let file = File::open(path)?;
    // SAFETY: arguments are a valid readable fd and nonzero length. The
    // returned pointer is retained only by MappedBacking and unmapped on Drop.
    let ptr = unsafe { mmap(std::ptr::null_mut(), bytes, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
    if ptr == MAP_FAILED {
        return Err(MappedTensorError::Io(io::Error::last_os_error()));
    }
    Ok(MappedBacking {
        id: MappedBackingId(NEXT_BACKING_ID.fetch_add(1, Ordering::Relaxed)),
        path: path.to_path_buf(),
        bytes,
        ptr: ptr.cast(),
    })
}

#[cfg(not(unix))]
fn open_backing(_path: &Path, _bytes: usize) -> Result<MappedBacking, MappedTensorError> {
    Err(MappedTensorError::UnsupportedPlatform)
}

#[cfg(unix)]
const PROT_READ: i32 = 0x1;
#[cfg(unix)]
const MAP_PRIVATE: i32 = 0x2;
#[cfg(unix)]
const MAP_FAILED: *mut std::ffi::c_void = !0usize as *mut std::ffi::c_void;

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
    fn munmap(address: *const std::ffi::c_void, length: usize) -> std::ffi::c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rustgrad-mapped-{name}-{}", std::process::id()))
    }

    #[test]
    fn mapped_windows_share_identity_and_materialize_owned_raw_bits() {
        let path = path("window");
        let raw = [0_u8, 0, 0, 0x80, 0x34, 0x12, 0xc0, 0x7f];
        std::fs::write(&path, raw).unwrap();
        let mapped = MappedTensor::open(&path, [2], DType::F32).unwrap();
        let view = mapped.view(1, [1]).unwrap();
        assert_eq!(mapped.backing_id(), view.backing_id());
        assert_eq!(mapped.policy(), MappedTensorPolicy::CopyToOwned);
        assert_eq!(view.materialize_cpu().unwrap().to_le_bytes().unwrap(), raw[4..]);
        assert_eq!(mapped.materialize_for_artifact().unwrap().to_le_bytes().unwrap(), raw);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_shape_or_window_rejects_before_exposing_tensor_data() {
        let path = path("reject");
        std::fs::write(&path, [0_u8; 4]).unwrap();
        assert!(matches!(
            MappedTensor::open(&path, [2], DType::F32),
            Err(MappedTensorError::ShapeBytes { .. })
        ));
        let mapped = MappedTensor::open(&path, [1], DType::F32).unwrap();
        assert!(matches!(mapped.view(1, [1]), Err(MappedTensorError::Bounds)));
        std::fs::remove_file(path).unwrap();
    }
}
