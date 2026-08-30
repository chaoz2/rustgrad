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
    /// Portable lane width used only for an exact logical ABI contract.
    pub lanes: usize,
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

/// Read-only liveness accounting for checked host leases. It intentionally
/// exposes neither addresses nor backing capacities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HostPoolStats {
    pub physical_slots: usize,
    pub leased_slots: usize,
    pub live_views: usize,
    pub mutable_windows: usize,
    pub zero_byte_sentinels: usize,
}

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
    mutable_window: bool,
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
            mutable_window: false,
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
        if entry.mutable_window {
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
                || previous.lanes != descriptor.lanes
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
        Ok(self.stats()?.physical_slots)
    }

    pub(crate) fn stats(&self) -> Result<HostPoolStats, HostBufferError> {
        let state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        Ok(HostPoolStats {
            physical_slots: state
                .slots
                .values()
                .filter(|slot| slot.capacity != 0)
                .count(),
            leased_slots: state.slots.values().filter(|slot| slot.leased).count(),
            live_views: state.slots.values().map(|slot| slot.views).sum(),
            mutable_windows: state
                .slots
                .values()
                .filter(|slot| slot.mutable_window)
                .count(),
            zero_byte_sentinels: state
                .slots
                .values()
                .filter(|slot| slot.capacity == 0)
                .count(),
        })
    }

    /// Commits a whole persistent-state transaction. Every lease/value pair is
    /// validated while holding the one pool lock before any visible slot value
    /// changes, so a rejected batch cannot partially update persistent state.
    pub(crate) fn commit(&self, writes: Vec<HostBufferWrite>) -> Result<(), HostBufferError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let mut seen = std::collections::BTreeSet::new();
        for write in &writes {
            if !Arc::ptr_eq(&self.inner, &write.inner) {
                return Err(HostBufferError::OwnerMismatch);
            }
            if !seen.insert(write.slot) {
                return Err(HostBufferError::OutstandingBorrow { slot: write.slot });
            }
            let slot = live_slot(&mut state, write.slot, write.generation)?;
            if slot.views != 0 || slot.mutable_window {
                return Err(HostBufferError::OutstandingBorrow { slot: write.slot });
            }
            if slot.descriptor.as_ref() != Some(&write.descriptor)
                || write.value.shape() != &write.descriptor.shape
                || write.value.dtype() != write.descriptor.dtype
            {
                return Err(HostBufferError::IncompatibleDescriptor);
            }
        }
        // No fallible checks remain after this point. Values are already owned
        // by the transaction, and each slot has an exclusive live lease.
        for write in writes {
            let slot = state
                .slots
                .get_mut(&write.slot)
                .expect("prevalidated live host slot");
            slot.value = Some(write.value);
        }
        Ok(())
    }
}

/// An owned, descriptor-checked value for one pool transaction. This remains
/// crate-private so callers cannot manufacture a slot/generation capability.
pub(crate) struct HostBufferWrite {
    inner: Arc<Mutex<PoolState>>,
    slot: u64,
    generation: u64,
    descriptor: HostBufferDesc,
    value: TensorData,
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

    pub(crate) fn staged_write(
        &self,
        value: TensorData,
    ) -> Result<HostBufferWrite, HostBufferError> {
        if value.shape() != &self.descriptor.shape || value.dtype() != self.descriptor.dtype {
            return Err(HostBufferError::IncompatibleDescriptor);
        }
        Ok(HostBufferWrite {
            inner: self.inner.clone(),
            slot: self.slot,
            generation: self.generation,
            descriptor: self.descriptor.clone(),
            value,
        })
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

    #[allow(dead_code)] // consumed by the effect executor in the next schedule integration.
    pub(crate) fn mutable_window(
        &mut self,
        offset: usize,
        bytes: usize,
    ) -> Result<HostByteWindow, HostBufferError> {
        let range = checked_range(&self.descriptor, offset, bytes)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = live_slot(&mut state, self.slot, self.generation)?;
        if slot.views != 0 || slot.mutable_window {
            return Err(HostBufferError::OutstandingBorrow { slot: self.slot });
        }
        slot.mutable_window = true;
        Ok(HostByteWindow {
            inner: self.inner.clone(),
            slot: self.slot,
            generation: self.generation,
            range,
            mutable: true,
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
    pub(crate) fn logical_bytes(&self) -> usize {
        self.descriptor.bytes
    }
    pub(crate) fn logical_range(
        &self,
        offset: usize,
        bytes: usize,
    ) -> Result<std::ops::Range<usize>, HostBufferError> {
        checked_range(&self.descriptor, offset, bytes)
    }

    pub(crate) fn byte_window(
        &self,
        offset: usize,
        bytes: usize,
    ) -> Result<HostByteWindow, HostBufferError> {
        let range = self.logical_range(offset, bytes)?;
        Ok(HostByteWindow {
            inner: self.inner.clone(),
            slot: self.slot,
            generation: self.generation,
            range,
            mutable: false,
        })
    }

    /// Acquires the sole mutable logical subrange for this live generation.
    /// A view itself counts as one borrow, so mutation is allowed only when no
    /// sibling view exists. The returned window exposes neither bytes nor a
    /// pointer; it is an ownership/liveness proof for a staged effect commit.
    #[allow(dead_code)] // schedule effect execution is the next consumer.
    pub(crate) fn mutable_window(
        &self,
        offset: usize,
        bytes: usize,
    ) -> Result<HostByteWindow, HostBufferError> {
        let range = self.logical_range(offset, bytes)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = live_slot(&mut state, self.slot, self.generation)?;
        if slot.views != 1 || slot.mutable_window {
            return Err(HostBufferError::OutstandingBorrow { slot: self.slot });
        }
        slot.mutable_window = true;
        Ok(HostByteWindow {
            inner: self.inner.clone(),
            slot: self.slot,
            generation: self.generation,
            range,
            mutable: true,
        })
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

/// Checked call-duration byte window. It exposes only a logical range, never
/// backing capacity or a raw pointer; native ABI plumbing must borrow it anew.
pub(crate) struct HostByteWindow {
    inner: Arc<Mutex<PoolState>>,
    slot: u64,
    generation: u64,
    range: std::ops::Range<usize>,
    mutable: bool,
}
impl HostByteWindow {
    pub(crate) fn len(&self) -> usize {
        self.range.len()
    }
    pub(crate) fn validate(&self) -> Result<(), HostBufferError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferError::OwnerMismatch)?;
        let slot = live_slot(&mut state, self.slot, self.generation)?;
        if self.range.end > slot.capacity {
            return Err(HostBufferError::LogicalBounds {
                requested: self.range.end,
                capacity: slot.capacity,
            });
        }
        Ok(())
    }
}
impl Drop for HostByteWindow {
    fn drop(&mut self) {
        if self.mutable
            && let Ok(mut state) = self.inner.lock()
            && let Some(slot) = state.slots.get_mut(&self.slot)
            && slot.generation == self.generation
        {
            slot.mutable_window = false;
        }
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

fn checked_range(
    descriptor: &HostBufferDesc,
    offset: usize,
    bytes: usize,
) -> Result<std::ops::Range<usize>, HostBufferError> {
    let end = offset.checked_add(bytes).ok_or(HostBufferError::Overflow)?;
    if end > descriptor.bytes {
        return Err(HostBufferError::LogicalBounds {
            requested: end,
            capacity: descriptor.bytes,
        });
    }
    Ok(offset..end)
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
            lanes: 4,
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
        assert!(view.mutable_window(0, 4).is_ok());
        drop(view);
        let window = lease.mutable_window(0, 4).unwrap();
        assert_eq!(window.len(), 4);
        window.validate().unwrap();
        assert!(matches!(
            lease.mutable_window(0, 4),
            Err(HostBufferError::OutstandingBorrow { .. })
        ));
        drop(window);
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

    #[test]
    fn transaction_preflights_every_lease_and_reuse_advances_generation() {
        let pool = HostSlotPool::new();
        let mut first = pool.lease(Some(4), desc(4, [2])).unwrap();
        let second = pool.lease(Some(5), desc(5, [2])).unwrap();
        first
            .write(TensorData::new([2], vec![1.0, 2.0]).unwrap())
            .unwrap();
        second
            .write(TensorData::new([2], vec![3.0, 4.0]).unwrap())
            .unwrap();
        let view = second.view().unwrap();
        let first_write = first
            .staged_write(TensorData::new([2], vec![9.0, 9.0]).unwrap())
            .unwrap();
        let second_write = second
            .staged_write(TensorData::new([2], vec![8.0, 8.0]).unwrap())
            .unwrap();
        assert!(matches!(
            pool.commit(vec![first_write, second_write]),
            Err(HostBufferError::OutstandingBorrow { slot: 5 })
        ));
        assert_eq!(
            first.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![1.0, 2.0]
        );
        drop(view);
        let first_write = first
            .staged_write(TensorData::new([2], vec![9.0, 9.0]).unwrap())
            .unwrap();
        let second_write = second
            .staged_write(TensorData::new([2], vec![8.0, 8.0]).unwrap())
            .unwrap();
        pool.commit(vec![first_write, second_write]).unwrap();
        assert_eq!(
            first.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![9.0, 9.0]
        );
        let old_generation = first.generation();
        first.release().unwrap();
        let reused = pool.lease(Some(4), desc(4, [2])).unwrap();
        assert_eq!(reused.slot(), 4);
        assert!(reused.generation() > old_generation);
        let stats = pool.stats().unwrap();
        assert_eq!(stats.physical_slots, 2);
        assert_eq!(stats.leased_slots, 2);
    }

    #[test]
    fn transaction_rejects_cross_pool_staged_write_before_publication() {
        let first_pool = HostSlotPool::new();
        let second_pool = HostSlotPool::new();
        let first = first_pool.lease(Some(7), desc(7, [2])).unwrap();
        let second = second_pool.lease(Some(7), desc(7, [2])).unwrap();
        second
            .write(TensorData::new([2], vec![1.0, 2.0]).unwrap())
            .unwrap();

        let foreign = first
            .staged_write(TensorData::new([2], vec![9.0, 9.0]).unwrap())
            .unwrap();
        assert_eq!(
            second_pool.commit(vec![foreign]),
            Err(HostBufferError::OwnerMismatch)
        );
        assert_eq!(
            second.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![1.0, 2.0]
        );

        let local = second
            .staged_write(TensorData::new([2], vec![8.0, 8.0]).unwrap())
            .unwrap();
        second_pool.commit(vec![local]).unwrap();
        assert_eq!(
            second.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![8.0, 8.0]
        );
    }
}
