//! Sealed logical buffers and command-retained physical generations.
use super::{MetalError, dispatch::RawBuffer, resource::DeviceInner};
use crate::DType;
use std::{cell::Cell, rc::Rc};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BufferDesc {
    bytes: usize,
    dtype: Option<DType>,
}

pub(super) struct PhysicalBuffer {
    pub(super) device: Rc<DeviceInner>,
    pub(super) raw: Option<RawBuffer>,
}

impl Drop for PhysicalBuffer {
    fn drop(&mut self) {
        if let Some(raw) = self.raw {
            self.device.dispatch.buffer_release(raw, self.device.owner);
        }
    }
}

struct LogicalBuffer {
    physical: Rc<PhysicalBuffer>,
    desc: BufferDesc,
    generation: u64,
    closed: Cell<bool>,
}

/// Stable, thread-confined Metal buffer identity.
pub struct MetalBuffer {
    inner: Rc<LogicalBuffer>,
}

#[derive(Clone)]
pub(super) struct BufferSnapshot {
    logical: Rc<LogicalBuffer>,
    pub(super) physical: Rc<PhysicalBuffer>,
    generation: u64,
}

impl BufferSnapshot {
    pub(super) fn raw(&self) -> Option<RawBuffer> {
        self.physical.raw
    }

    pub(super) fn validate_current(&self) -> Result<(), MetalError> {
        if self.logical.generation == self.generation {
            Ok(())
        } else {
            Err(MetalError::StaleGeneration {
                expected: self.generation,
                actual: self.logical.generation,
            })
        }
    }
}

impl MetalBuffer {
    pub(super) fn allocate(
        device: Rc<DeviceInner>,
        bytes: usize,
        dtype: Option<DType>,
    ) -> Result<Self, MetalError> {
        device.live()?;
        if bytes > device.info.capabilities.max_buffer_length {
            return Err(MetalError::InvalidArgument(
                "buffer exceeds device maximum length",
            ));
        }
        let raw = if bytes == 0 {
            None
        } else {
            Some(
                device
                    .dispatch
                    .buffer_create(device.raw, bytes, device.owner)?,
            )
        };
        let physical = Rc::new(PhysicalBuffer {
            device: device.clone(),
            raw,
        });
        Ok(Self {
            inner: Rc::new(LogicalBuffer {
                physical,
                desc: BufferDesc { bytes, dtype },
                generation: 1,
                closed: Cell::new(false),
            }),
        })
    }

    /// Returns the logical byte length.
    pub fn len(&self) -> usize {
        self.inner.desc.bytes
    }

    /// Reports whether the logical buffer has zero bytes and no native object.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the optional checked storage dtype.
    pub fn dtype(&self) -> Option<DType> {
        self.inner.desc.dtype
    }

    /// Returns the visible physical-generation identity.
    pub fn generation(&self) -> u64 {
        self.inner.generation
    }

    /// Returns the stable safe owner identity, never a native handle.
    pub fn owner_id(&self) -> u64 {
        self.inner.physical.device.owner
    }

    pub(super) fn snapshot(
        &self,
        device: &Rc<DeviceInner>,
        offset: usize,
        bytes: usize,
        dtype: Option<DType>,
    ) -> Result<BufferSnapshot, MetalError> {
        device.live()?;
        if self.inner.closed.get() {
            return Err(MetalError::Closed("buffer"));
        }
        if !Rc::ptr_eq(device, &self.inner.physical.device) {
            return Err(MetalError::OwnerMismatch);
        }
        if offset.checked_add(bytes).ok_or(MetalError::Overflow)? > self.len() {
            return Err(MetalError::Bounds);
        }
        if let (Some(actual), Some(expected)) = (self.dtype(), dtype)
            && actual != expected
        {
            return Err(MetalError::InvalidBinding(
                "logical buffer dtype mismatch".into(),
            ));
        }
        Ok(BufferSnapshot {
            logical: self.inner.clone(),
            physical: self.inner.physical.clone(),
            generation: self.inner.generation,
        })
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        self.inner.closed.set(true);
    }
}
