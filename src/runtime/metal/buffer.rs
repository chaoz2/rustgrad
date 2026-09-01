//! Sealed logical buffers and command-retained physical generations.
use super::{MetalError, dispatch::RawBuffer, resource::DeviceInner};
use crate::DType;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BufferDesc {
    bytes: usize,
    physical_bytes: usize,
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

struct VisibleGeneration {
    physical: Rc<PhysicalBuffer>,
    generation: u64,
}

struct LogicalBuffer {
    device: Rc<DeviceInner>,
    visible: RefCell<VisibleGeneration>,
    desc: BufferDesc,
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
        let actual = self.logical.visible.borrow().generation;
        if actual == self.generation {
            Ok(())
        } else {
            Err(MetalError::StaleGeneration {
                expected: self.generation,
                actual,
            })
        }
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }
}

impl MetalBuffer {
    pub(super) fn allocate(
        device: Rc<DeviceInner>,
        bytes: usize,
        dtype: Option<DType>,
    ) -> Result<Self, MetalError> {
        Self::allocate_with_handle(device, bytes, dtype, false)
    }

    pub(super) fn allocate_with_handle(
        device: Rc<DeviceInner>,
        bytes: usize,
        dtype: Option<DType>,
        requires_native_handle: bool,
    ) -> Result<Self, MetalError> {
        device.live()?;
        let physical_bytes = if bytes == 0 && requires_native_handle {
            4
        } else {
            bytes
        };
        if physical_bytes > device.info.capabilities.max_buffer_length {
            return Err(MetalError::InvalidArgument(
                "buffer exceeds device maximum length",
            ));
        }
        let raw = if physical_bytes == 0 {
            None
        } else {
            Some(
                device
                    .dispatch
                    .buffer_create(device.raw, physical_bytes, device.owner)?,
            )
        };
        let physical = Rc::new(PhysicalBuffer {
            device: device.clone(),
            raw,
        });
        Ok(Self {
            inner: Rc::new(LogicalBuffer {
                device,
                visible: RefCell::new(VisibleGeneration {
                    physical,
                    generation: 1,
                }),
                desc: BufferDesc {
                    bytes,
                    physical_bytes,
                    dtype,
                },
                closed: Cell::new(false),
            }),
        })
    }

    /// Returns the logical byte length.
    pub fn len(&self) -> usize {
        self.inner.desc.bytes
    }

    /// Reports whether the logical buffer has zero addressable bytes. A private
    /// ABI sentinel may still own a native object.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the optional checked storage dtype.
    pub fn dtype(&self) -> Option<DType> {
        self.inner.desc.dtype
    }

    /// Returns the visible physical-generation identity.
    pub fn generation(&self) -> u64 {
        self.inner.visible.borrow().generation
    }

    /// Returns the stable safe owner identity, never a native handle.
    pub fn owner_id(&self) -> u64 {
        self.inner.device.owner
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
        if !Rc::ptr_eq(device, &self.inner.device) {
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
        let visible = self.inner.visible.borrow();
        Ok(BufferSnapshot {
            logical: self.inner.clone(),
            physical: visible.physical.clone(),
            generation: visible.generation,
        })
    }

    pub(super) fn candidate(&self) -> Result<Rc<PhysicalBuffer>, MetalError> {
        self.inner.device.live()?;
        let raw = if self.inner.desc.physical_bytes == 0 {
            None
        } else {
            Some(self.inner.device.dispatch.buffer_create(
                self.inner.device.raw,
                self.inner.desc.physical_bytes,
                self.inner.device.owner,
            )?)
        };
        Ok(Rc::new(PhysicalBuffer {
            device: self.inner.device.clone(),
            raw,
        }))
    }

    pub(super) fn commit_candidate(
        &self,
        expected: u64,
        candidate: Rc<PhysicalBuffer>,
    ) -> Result<u64, MetalError> {
        self.inner.device.live()?;
        if self.inner.closed.get() {
            return Err(MetalError::Closed("buffer"));
        }
        if !Rc::ptr_eq(&self.inner.device, &candidate.device) {
            return Err(MetalError::OwnerMismatch);
        }
        let mut visible = self.inner.visible.borrow_mut();
        if visible.generation != expected {
            return Err(MetalError::StaleGeneration {
                expected,
                actual: visible.generation,
            });
        }
        let next = expected.checked_add(1).ok_or(MetalError::Overflow)?;
        *visible = VisibleGeneration {
            physical: candidate,
            generation: next,
        };
        Ok(next)
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        self.inner.closed.set(true);
    }
}
