//! Private generation-checked host storage for schedule temporaries.
use crate::{DType, Shape, TensorData};
use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HostBufferDesc {
    pub buffer_id: u64,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
    pub alignment: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostBufferError {
    Overflow,
    OwnerMismatch,
    StaleGeneration { slot: u64, generation: u64 },
    LogicalBounds { requested: usize, capacity: usize },
    IncompatibleDescriptor,
    DoubleRelease { slot: u64 },
    OutstandingBorrow { slot: u64 },
    MissingSlot(u64),
    MissingValue(u64),
}
impl fmt::Display for HostBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "host buffer error: {self:?}")
    }
}
impl std::error::Error for HostBufferError {}

#[derive(Clone)]
pub(crate) struct HostSlotPool {
    inner: Arc<Mutex<PoolState>>,
}
struct PoolState {
    slots: BTreeMap<u64, Slot>,
    next_sentinel: u64,
}
struct Slot {
    generation: u64,
    capacity: usize,
    descriptor: Option<HostBufferDesc>,
    value: Option<TensorData>,
    leased: bool,
    views: usize,
    // This is deliberately private; no pointer or capacity escapes the ABI.
    _physical: Vec<u8>,
}
impl HostSlotPool {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolState {
                slots: BTreeMap::new(),
                next_sentinel: 0,
            })),
        }
    }

    pub(crate) fn lease(
        &self,
        physical_slot: Option<u64>,
        descriptor: HostBufferDesc,
    ) -> Result<HostBufferLease, HostBufferError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = physical_slot.unwrap_or_else(|| {
            let id = state.next_sentinel;
            state.next_sentinel += 1;
            u64::MAX - id
        });
        let is_sentinel = physical_slot.is_none();
        let entry = state.slots.entry(slot).or_insert_with(|| Slot {
            generation: 0,
            capacity: descriptor.bytes,
            descriptor: None,
            value: None,
            leased: false,
            views: 0,
            _physical: if is_sentinel {
                vec![]
            } else {
                vec![0; descriptor.bytes]
            },
        });
        if entry.leased {
            return Err(HostBufferError::OutstandingBorrow { slot });
        }
        if entry.views != 0 {
            return Err(HostBufferError::OutstandingBorrow { slot });
        }
        if !is_sentinel && entry.capacity != descriptor.bytes {
            return Err(HostBufferError::LogicalBounds {
                requested: descriptor.bytes,
                capacity: entry.capacity,
            });
        }
        if let Some(previous) = &entry.descriptor
            && (previous.dtype != descriptor.dtype
                || previous.shape != descriptor.shape
                || previous.alignment != descriptor.alignment
                || previous.bytes != descriptor.bytes)
        {
            return Err(HostBufferError::IncompatibleDescriptor);
        }
        entry.generation = entry
            .generation
            .checked_add(1)
            .ok_or(HostBufferError::Overflow)?;
        entry.descriptor = Some(descriptor.clone());
        entry.value = None;
        entry.leased = true;
        Ok(HostBufferLease {
            inner: self.inner.clone(),
            slot,
            generation: entry.generation,
            descriptor,
            released: false,
        })
    }

    pub(crate) fn physical_slots(&self) -> Result<usize, HostBufferError> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?
            .slots
            .values()
            .filter(|slot| slot.capacity != 0)
            .count())
    }
}

/// Non-cloneable ownership of one logical allocation generation.
pub(crate) struct HostBufferLease {
    inner: Arc<Mutex<PoolState>>,
    slot: u64,
    generation: u64,
    descriptor: HostBufferDesc,
    released: bool,
}
impl HostBufferLease {
    pub(crate) fn slot(&self) -> u64 {
        self.slot
    }
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn write(&self, value: TensorData) -> Result<(), HostBufferError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = live_slot(&mut state, self.slot, self.generation)?;
        if value.shape() != &self.descriptor.shape || value.dtype() != self.descriptor.dtype {
            return Err(HostBufferError::IncompatibleDescriptor);
        }
        slot.value = Some(value);
        Ok(())
    }

    pub(crate) fn view(&self) -> Result<HostBufferView, HostBufferError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = live_slot(&mut state, self.slot, self.generation)?;
        slot.views = slot.views.checked_add(1).ok_or(HostBufferError::Overflow)?;
        Ok(HostBufferView {
            inner: self.inner.clone(),
            slot: self.slot,
            generation: self.generation,
            descriptor: self.descriptor.clone(),
        })
    }

    pub(crate) fn release(&mut self) -> Result<(), HostBufferError> {
        self.release_inner()
    }

    fn release_inner(&mut self) -> Result<(), HostBufferError> {
        if self.released {
            return Err(HostBufferError::DoubleRelease { slot: self.slot });
        }
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = live_slot(&mut state, self.slot, self.generation)?;
        if slot.views != 0 {
            return Err(HostBufferError::OutstandingBorrow { slot: self.slot });
        }
        slot.leased = false;
        slot.value = None;
        self.released = true;
        Ok(())
    }
}
impl Drop for HostBufferLease {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

/// Non-cloneable checked view. It cannot outlive its slot generation.
pub(crate) struct HostBufferView {
    inner: Arc<Mutex<PoolState>>,
    slot: u64,
    generation: u64,
    descriptor: HostBufferDesc,
}
impl HostBufferView {
    pub(crate) fn logical_range(
        &self,
        offset: usize,
        bytes: usize,
    ) -> Result<std::ops::Range<usize>, HostBufferError> {
        let end = offset.checked_add(bytes).ok_or(HostBufferError::Overflow)?;
        if end > self.descriptor.bytes {
            return Err(HostBufferError::LogicalBounds {
                requested: end,
                capacity: self.descriptor.bytes,
            });
        }
        Ok(offset..end)
    }

    pub(crate) fn tensor(&self) -> Result<TensorData, HostBufferError> {
        self.logical_range(0, self.descriptor.bytes)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = live_slot(&mut state, self.slot, self.generation)?;
        slot.value
            .clone()
            .ok_or(HostBufferError::MissingValue(self.descriptor.buffer_id))
    }
}
impl Drop for HostBufferView {
    fn drop(&mut self) {
        if let Ok(mut state) = self.inner.lock()
            && let Some(slot) = state.slots.get_mut(&self.slot)
            && slot.generation == self.generation
            && slot.views != 0
        {
            slot.views -= 1;
        }
    }
}

fn live_slot(
    state: &mut PoolState,
    slot: u64,
    generation: u64,
) -> Result<&mut Slot, HostBufferError> {
    let entry = state
        .slots
        .get_mut(&slot)
        .ok_or(HostBufferError::MissingSlot(slot))?;
    if entry.generation != generation || !entry.leased {
        return Err(HostBufferError::StaleGeneration { slot, generation });
    }
    Ok(entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(id: u64, shape: impl Into<Shape>) -> HostBufferDesc {
        let shape = shape.into();
        HostBufferDesc {
            buffer_id: id,
            dtype: DType::F32,
            bytes: shape.numel().unwrap() * 4,
            shape,
            alignment: 4,
        }
    }
    #[test]
    fn stale_views_and_live_borrows_prevent_reuse() {
        let pool = HostSlotPool::new();
        let mut lease = pool.lease(Some(0), desc(1, [2])).unwrap();
        lease
            .write(TensorData::new([2], vec![1.0, 2.0]).unwrap())
            .unwrap();
        let view = lease.view().unwrap();
        assert!(matches!(
            view.logical_range(4, 8),
            Err(HostBufferError::LogicalBounds { .. })
        ));
        assert!(matches!(
            lease.release(),
            Err(HostBufferError::OutstandingBorrow { .. })
        ));
        drop(view);
        lease.release().unwrap();
        let mut lease = pool.lease(Some(0), desc(2, [2])).unwrap();
        let stale = HostBufferView {
            inner: pool.inner.clone(),
            slot: lease.slot(),
            generation: lease.generation().saturating_sub(1),
            descriptor: desc(2, [2]),
        };
        lease.release().unwrap();
        assert!(matches!(
            stale.tensor(),
            Err(HostBufferError::StaleGeneration { .. })
        ));
    }
}
