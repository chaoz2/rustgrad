//! Sealed logical buffers and command-retained physical generations.
use super::{WebGpuError, dispatch::RawBuffer, resource::DeviceInner};
use crate::DType;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BufferDesc {
    logical_bytes: usize,
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

/// Stable, typed, thread-confined WebGPU buffer identity.
pub struct WebGpuBuffer {
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

    pub(super) fn validate_current(&self) -> Result<(), WebGpuError> {
        let actual = self.logical.visible.borrow().generation;
        if actual == self.generation {
            Ok(())
        } else {
            Err(WebGpuError::StaleGeneration {
                expected: self.generation,
                actual,
            })
        }
    }

    pub(super) fn same_logical(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.logical, &other.logical)
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }
}

impl WebGpuBuffer {
    pub(super) fn allocate(
        device: Rc<DeviceInner>,
        logical_bytes: usize,
        dtype: Option<DType>,
    ) -> Result<Self, WebGpuError> {
        Self::allocate_with_handle(device, logical_bytes, dtype, false)
    }

    pub(super) fn allocate_with_handle(
        device: Rc<DeviceInner>,
        logical_bytes: usize,
        dtype: Option<DType>,
        requires_native_handle: bool,
    ) -> Result<Self, WebGpuError> {
        device.live()?;
        let physical_bytes = if logical_bytes == 0 && requires_native_handle {
            4
        } else {
            logical_bytes.checked_add(3).ok_or(WebGpuError::Overflow)? / 4 * 4
        };
        if physical_bytes > device.info.capabilities.max_buffer_size {
            return Err(WebGpuError::InvalidArgument(
                "buffer exceeds adapter maximum size",
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
        Ok(Self {
            inner: Rc::new(LogicalBuffer {
                device: device.clone(),
                visible: RefCell::new(VisibleGeneration {
                    physical: Rc::new(PhysicalBuffer { device, raw }),
                    generation: 1,
                }),
                desc: BufferDesc {
                    logical_bytes,
                    physical_bytes,
                    dtype,
                },
                closed: Cell::new(false),
            }),
        })
    }

    /// Returns the logical byte length, excluding required WebGPU padding.
    pub fn len(&self) -> usize {
        self.inner.desc.logical_bytes
    }

    /// Reports whether the logical buffer contains no addressable bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns checked storage dtype metadata when allocated as typed.
    pub fn dtype(&self) -> Option<DType> {
        self.inner.desc.dtype
    }
    /// Returns the private physical allocation size rounded to four bytes.
    pub fn physical_len(&self) -> usize {
        self.inner.desc.physical_bytes
    }

    /// Returns the visible physical-generation identity.
    pub fn generation(&self) -> u64 {
        self.inner.visible.borrow().generation
    }

    /// Returns the stable Rust owner identity, never a native handle.
    pub fn owner_id(&self) -> u64 {
        self.inner.device.owner
    }

    pub(super) fn snapshot(
        &self,
        device: &Rc<DeviceInner>,
        offset: usize,
        bytes: usize,
        dtype: Option<DType>,
    ) -> Result<BufferSnapshot, WebGpuError> {
        device.live()?;
        if self.inner.closed.get() {
            return Err(WebGpuError::Closed("buffer"));
        }
        if !Rc::ptr_eq(device, &self.inner.device) {
            return Err(WebGpuError::OwnerMismatch);
        }
        if offset.checked_add(bytes).ok_or(WebGpuError::Overflow)? > self.len() {
            return Err(WebGpuError::Bounds);
        }
        if let Some(expected) = dtype {
            match self.dtype() {
                Some(actual) if actual == expected => {}
                Some(_) => {
                    return Err(WebGpuError::InvalidBinding(
                        "logical buffer dtype mismatch".into(),
                    ));
                }
                None => {
                    return Err(WebGpuError::InvalidBinding(
                        "typed launch requires typed logical buffer".into(),
                    ));
                }
            }
        }
        let visible = self.inner.visible.borrow();
        Ok(BufferSnapshot {
            logical: self.inner.clone(),
            physical: visible.physical.clone(),
            generation: visible.generation,
        })
    }

    pub(super) fn candidate(&self) -> Result<Rc<PhysicalBuffer>, WebGpuError> {
        self.inner.device.live()?;
        if self.inner.closed.get() {
            return Err(WebGpuError::Closed("buffer"));
        }
        let raw = if self.physical_len() == 0 {
            None
        } else {
            Some(self.inner.device.dispatch.buffer_create(
                self.inner.device.raw,
                self.physical_len(),
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
    ) -> Result<u64, WebGpuError> {
        self.inner.device.live()?;
        if self.inner.closed.get() {
            return Err(WebGpuError::Closed("buffer"));
        }
        if !Rc::ptr_eq(&self.inner.device, &candidate.device) {
            return Err(WebGpuError::OwnerMismatch);
        }
        let mut visible = self.inner.visible.borrow_mut();
        if visible.generation != expected {
            return Err(WebGpuError::StaleGeneration {
                expected,
                actual: visible.generation,
            });
        }
        let next = expected.checked_add(1).ok_or(WebGpuError::Overflow)?;
        *visible = VisibleGeneration {
            physical: candidate,
            generation: next,
        };
        Ok(next)
    }

    #[cfg(test)]
    pub(super) fn replace_generation_for_test(&self) -> Result<u64, WebGpuError> {
        self.inner.device.live()?;
        let raw = if self.physical_len() == 0 {
            None
        } else {
            Some(self.inner.device.dispatch.buffer_create(
                self.inner.device.raw,
                self.physical_len(),
                self.inner.device.owner,
            )?)
        };
        let mut visible = self.inner.visible.borrow_mut();
        let next = visible
            .generation
            .checked_add(1)
            .ok_or(WebGpuError::Overflow)?;
        *visible = VisibleGeneration {
            physical: Rc::new(PhysicalBuffer {
                device: self.inner.device.clone(),
                raw,
            }),
            generation: next,
        };
        Ok(next)
    }
}

impl Drop for WebGpuBuffer {
    fn drop(&mut self) {
        self.inner.closed.set(true);
    }
}
