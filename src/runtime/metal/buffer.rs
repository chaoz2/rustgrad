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

/// Immutable accounting for native Metal buffers owned by one discovered device.
///
/// Byte totals are the requested `MTLBuffer` lengths for successful, nonzero
/// RustGrad-owned allocations. They are not allocator RSS, physical residency,
/// driver overhead, unified-memory pressure, or a per-session metric. The
/// lifetime high-water values span all clones of this discovered device.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalBufferAllocationStats {
    /// Native physical buffers currently owned by RustGrad.
    pub current_physical_buffer_count: usize,
    /// Sum of the requested lengths of current native physical buffers.
    pub current_physical_buffer_bytes: usize,
    /// Largest current physical-buffer count observed during this device lifetime.
    pub lifetime_high_water_physical_buffer_count: usize,
    /// Largest current physical-buffer byte total observed during this device lifetime.
    pub lifetime_high_water_physical_buffer_bytes: usize,
}

impl MetalBufferAllocationStats {
    pub(super) fn checked_allocate(self, bytes: usize) -> Result<Self, MetalError> {
        if bytes == 0 {
            return Err(MetalError::InvalidArgument(
                "physical buffer accounting requires a nonzero length",
            ));
        }
        let current_physical_buffer_count = self
            .current_physical_buffer_count
            .checked_add(1)
            .ok_or(MetalError::Overflow)?;
        let current_physical_buffer_bytes = self
            .current_physical_buffer_bytes
            .checked_add(bytes)
            .ok_or(MetalError::Overflow)?;
        Ok(Self {
            current_physical_buffer_count,
            current_physical_buffer_bytes,
            lifetime_high_water_physical_buffer_count: self
                .lifetime_high_water_physical_buffer_count
                .max(current_physical_buffer_count),
            lifetime_high_water_physical_buffer_bytes: self
                .lifetime_high_water_physical_buffer_bytes
                .max(current_physical_buffer_bytes),
        })
    }

    pub(super) fn checked_release(self, bytes: usize) -> Result<Self, MetalError> {
        if bytes == 0 {
            return Err(MetalError::InvalidArgument(
                "physical buffer accounting requires a nonzero length",
            ));
        }
        let current_physical_buffer_count = self
            .current_physical_buffer_count
            .checked_sub(1)
            .ok_or(MetalError::Overflow)?;
        let current_physical_buffer_bytes = self
            .current_physical_buffer_bytes
            .checked_sub(bytes)
            .ok_or(MetalError::Overflow)?;
        Ok(Self {
            current_physical_buffer_count,
            current_physical_buffer_bytes,
            ..self
        })
    }
}

pub(super) struct PhysicalBuffer {
    pub(super) device: Rc<DeviceInner>,
    pub(super) raw: Option<RawBuffer>,
    accounted_bytes: usize,
}

impl PhysicalBuffer {
    fn allocate(device: Rc<DeviceInner>, bytes: usize) -> Result<Rc<Self>, MetalError> {
        if bytes == 0 {
            return Ok(Rc::new(Self {
                device,
                raw: None,
                accounted_bytes: 0,
            }));
        }
        let raw = device
            .dispatch
            .buffer_create(device.raw, bytes, device.owner)?;
        if let Err(error) = device.record_physical_buffer_allocation(bytes) {
            device.dispatch.buffer_release(raw, device.owner);
            return Err(error);
        }
        Ok(Rc::new(Self {
            device,
            raw: Some(raw),
            accounted_bytes: bytes,
        }))
    }
}

impl Drop for PhysicalBuffer {
    fn drop(&mut self) {
        if let Some(raw) = self.raw {
            self.device.dispatch.buffer_release(raw, self.device.owner);
            let accounting = self
                .device
                .record_physical_buffer_release(self.accounted_bytes);
            debug_assert!(accounting.is_ok(), "physical-buffer accounting underflow");
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
    owners: Cell<usize>,
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
        let physical = PhysicalBuffer::allocate(device.clone(), physical_bytes)?;
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
                owners: Cell::new(1),
            }),
        })
    }

    /// Creates another private owner of the same authenticated logical buffer.
    /// Snapshots remain generation-checked and the buffer closes only after
    /// the last owning handle is dropped.
    pub(super) fn share(&self) -> Result<Self, MetalError> {
        if self.inner.closed.get() {
            return Err(MetalError::Closed("buffer"));
        }
        let owners = self
            .inner
            .owners
            .get()
            .checked_add(1)
            .ok_or(MetalError::Overflow)?;
        self.inner.owners.set(owners);
        Ok(Self {
            inner: self.inner.clone(),
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

    pub(super) fn physical_len(&self) -> usize {
        self.inner.desc.physical_bytes
    }

    pub(super) fn has_native_handle(&self) -> bool {
        self.inner.visible.borrow().physical.raw.is_some()
    }

    pub(super) fn logical_identity(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
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
        PhysicalBuffer::allocate(self.inner.device.clone(), self.inner.desc.physical_bytes)
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
        let owners = self.inner.owners.get();
        debug_assert!(owners != 0);
        let remaining = owners.saturating_sub(1);
        self.inner.owners.set(remaining);
        if remaining == 0 {
            self.inner.closed.set(true);
        }
    }
}

#[cfg(test)]
mod allocation_stats_tests {
    use super::*;

    #[test]
    fn checked_accounting_overflow_and_underflow_leave_snapshots_unchanged() {
        let empty = MetalBufferAllocationStats::default();
        assert_eq!(empty.checked_release(1), Err(MetalError::Overflow));
        assert_eq!(empty, MetalBufferAllocationStats::default());

        let too_few_bytes = MetalBufferAllocationStats {
            current_physical_buffer_count: 1,
            current_physical_buffer_bytes: 0,
            lifetime_high_water_physical_buffer_count: 1,
            lifetime_high_water_physical_buffer_bytes: 0,
        };
        assert_eq!(too_few_bytes.checked_release(1), Err(MetalError::Overflow));

        let full = MetalBufferAllocationStats {
            current_physical_buffer_count: usize::MAX,
            current_physical_buffer_bytes: 7,
            lifetime_high_water_physical_buffer_count: usize::MAX,
            lifetime_high_water_physical_buffer_bytes: 7,
        };
        assert_eq!(full.checked_allocate(1), Err(MetalError::Overflow));
        assert_eq!(
            full,
            MetalBufferAllocationStats {
                current_physical_buffer_count: usize::MAX,
                current_physical_buffer_bytes: 7,
                lifetime_high_water_physical_buffer_count: usize::MAX,
                lifetime_high_water_physical_buffer_bytes: 7,
            }
        );

        let full_bytes = MetalBufferAllocationStats {
            current_physical_buffer_count: 7,
            current_physical_buffer_bytes: usize::MAX,
            lifetime_high_water_physical_buffer_count: 7,
            lifetime_high_water_physical_buffer_bytes: usize::MAX,
        };
        assert_eq!(full_bytes.checked_allocate(1), Err(MetalError::Overflow));
    }
}
