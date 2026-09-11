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
    values: [Option<TensorData>; 2],
    active: usize,
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
            values: [None, None],
            active: 0,
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
        entry.values = [None, None];
        entry.active = 0;
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
        // by the transaction, and each slot has an exclusive live lease. Stage
        // every successor into the inactive bank before making any visible.
        let mut flips = Vec::with_capacity(writes.len());
        for write in writes {
            let slot = state
                .slots
                .get_mut(&write.slot)
                .expect("prevalidated live host slot");
            let inactive = 1 - slot.active;
            slot.values[inactive] = Some(write.value);
            flips.push(write.slot);
        }
        for slot in flips {
            let slot = state
                .slots
                .get_mut(&slot)
                .expect("staged host slot remains live");
            slot.active = 1 - slot.active;
        }
        Ok(())
    }

    pub(crate) fn transact_inactive_banks<T, E>(
        &self,
        requests: &[HostBufferBankRequest],
        stage: impl FnOnce(&mut [HostBufferBank<'_>]) -> Result<T, E>,
    ) -> Result<T, HostBufferBankTransactionError<E>> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| HostBufferBankTransactionError::Host(HostBufferError::OwnerMismatch))?;
        let mut requested = BTreeMap::new();
        for (ordinal, request) in requests.iter().enumerate() {
            if !Arc::ptr_eq(&self.inner, &request.inner)
                || requested.insert(request.slot, ordinal).is_some()
            {
                return Err(HostBufferBankTransactionError::Host(
                    HostBufferError::OwnerMismatch,
                ));
            }
            let slot = live_slot(&mut state, request.slot, request.generation)
                .map_err(HostBufferBankTransactionError::Host)?;
            if slot.views != 0 || slot.mutable_window {
                return Err(HostBufferBankTransactionError::Host(
                    HostBufferError::OutstandingBorrow { slot: request.slot },
                ));
            }
            if slot.descriptor.as_ref() != Some(&request.descriptor) {
                return Err(HostBufferBankTransactionError::Host(
                    HostBufferError::IncompatibleDescriptor,
                ));
            }
            let inactive = 1 - slot.active;
            if slot.values[inactive]
                .as_ref()
                .is_none_or(|value| !tensor_matches_descriptor(value, &request.descriptor))
            {
                slot.values[inactive] = Some(
                    TensorData::zeros_with_dtype(
                        request.descriptor.shape.clone(),
                        request.descriptor.dtype,
                    )
                    .map_err(|_| {
                        HostBufferBankTransactionError::Host(
                            HostBufferError::IncompatibleDescriptor,
                        )
                    })?,
                );
            }
        }

        let mut banks = Vec::with_capacity(requests.len());
        for (slot_id, slot) in &mut state.slots {
            let Some(ordinal) = requested.get(slot_id).copied() else {
                continue;
            };
            let (active, inactive) = if slot.active == 0 {
                let (active, inactive) = slot.values.split_at_mut(1);
                (active[0].as_ref(), inactive[0].as_mut())
            } else {
                let (inactive, active) = slot.values.split_at_mut(1);
                (active[0].as_ref(), inactive[0].as_mut())
            };
            banks.push(HostBufferBank {
                ordinal,
                buffer_id: requests[ordinal].descriptor.buffer_id,
                active: active.ok_or_else(|| {
                    HostBufferBankTransactionError::Host(HostBufferError::MissingValue(
                        requests[ordinal].descriptor.buffer_id,
                    ))
                })?,
                inactive: inactive.expect("inactive persistent bank was initialized"),
            });
        }
        if banks.len() != requests.len() {
            return Err(HostBufferBankTransactionError::Host(
                HostBufferError::OwnerMismatch,
            ));
        }
        banks.sort_by_key(|bank| bank.ordinal);
        let value = stage(&mut banks).map_err(HostBufferBankTransactionError::Stage)?;
        if banks.iter().any(|bank| {
            !tensor_matches_descriptor(bank.inactive(), &requests[bank.ordinal].descriptor)
        }) {
            return Err(HostBufferBankTransactionError::Host(
                HostBufferError::IncompatibleDescriptor,
            ));
        }
        drop(banks);
        for request in requests {
            let slot = state
                .slots
                .get_mut(&request.slot)
                .expect("validated inactive persistent bank remains live");
            slot.active = 1 - slot.active;
        }
        Ok(value)
    }

    /// Borrows an authenticated set of active persistent values for exactly
    /// one callback while holding the pool lock. References cannot escape the
    /// callback, and this read-only path never stages or flips a bank.
    pub(crate) fn with_active_tensors<T, E>(
        &self,
        requests: &[HostBufferReadRequest],
        read: impl FnOnce(&[HostBufferRead<'_>]) -> Result<T, E>,
    ) -> Result<T, HostBufferReadError<E>> {
        let state = self
            .inner
            .lock()
            .map_err(|_| HostBufferReadError::Host(HostBufferError::OwnerMismatch))?;
        let mut seen = std::collections::BTreeSet::new();
        for request in requests {
            if !Arc::ptr_eq(&self.inner, &request.inner) || !seen.insert(request.slot) {
                return Err(HostBufferReadError::Host(HostBufferError::OwnerMismatch));
            }
            let slot = state
                .slots
                .get(&request.slot)
                .ok_or(HostBufferError::MissingSlot(request.slot))
                .map_err(HostBufferReadError::Host)?;
            if slot.generation != request.generation || !slot.leased {
                return Err(HostBufferReadError::Host(
                    HostBufferError::StaleGeneration {
                        slot: request.slot,
                        generation: request.generation,
                    },
                ));
            }
            if slot.mutable_window {
                return Err(HostBufferReadError::Host(
                    HostBufferError::OutstandingBorrow { slot: request.slot },
                ));
            }
            if slot.descriptor.as_ref() != Some(&request.descriptor) {
                return Err(HostBufferReadError::Host(
                    HostBufferError::IncompatibleDescriptor,
                ));
            }
            let active = slot.values[slot.active]
                .as_ref()
                .ok_or(HostBufferError::MissingValue(request.descriptor.buffer_id))
                .map_err(HostBufferReadError::Host)?;
            if !tensor_matches_descriptor(active, &request.descriptor) {
                return Err(HostBufferReadError::Host(
                    HostBufferError::IncompatibleDescriptor,
                ));
            }
        }

        let reads = requests
            .iter()
            .enumerate()
            .map(|(ordinal, request)| {
                let slot = state
                    .slots
                    .get(&request.slot)
                    .expect("authenticated active host slot remains registered");
                HostBufferRead {
                    ordinal,
                    buffer_id: request.descriptor.buffer_id,
                    tensor: slot.values[slot.active]
                        .as_ref()
                        .expect("authenticated active host value remains present"),
                }
            })
            .collect::<Vec<_>>();
        read(&reads).map_err(HostBufferReadError::Callback)
    }
}

fn tensor_matches_descriptor(value: &TensorData, descriptor: &HostBufferDesc) -> bool {
    value.shape() == &descriptor.shape
        && value.dtype() == descriptor.dtype
        && value.len().checked_mul(value.dtype().itemsize()) == Some(descriptor.bytes)
}

#[derive(Debug)]
pub(crate) enum HostBufferBankTransactionError<E> {
    Host(HostBufferError),
    Stage(E),
}

#[derive(Debug)]
pub(crate) enum HostBufferReadError<E> {
    Host(HostBufferError),
    Callback(E),
}

pub(crate) struct HostBufferReadRequest {
    inner: Arc<Mutex<PoolState>>,
    slot: u64,
    generation: u64,
    descriptor: HostBufferDesc,
}

/// Non-cloneable active-bank borrow valid only for one pool-locked callback.
pub(crate) struct HostBufferRead<'a> {
    ordinal: usize,
    buffer_id: u64,
    tensor: &'a TensorData,
}

impl<'a> HostBufferRead<'a> {
    pub(crate) fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub(crate) fn buffer_id(&self) -> u64 {
        self.buffer_id
    }

    pub(crate) fn tensor(&self) -> &'a TensorData {
        self.tensor
    }
}

pub(crate) struct HostBufferBankRequest {
    inner: Arc<Mutex<PoolState>>,
    slot: u64,
    generation: u64,
    descriptor: HostBufferDesc,
}

pub(crate) struct HostBufferBank<'a> {
    ordinal: usize,
    buffer_id: u64,
    active: &'a TensorData,
    inactive: &'a mut TensorData,
}

impl HostBufferBank<'_> {
    pub(crate) fn buffer_id(&self) -> u64 {
        self.buffer_id
    }

    pub(crate) fn tensors(&mut self) -> (&TensorData, &mut TensorData) {
        (self.active, &mut *self.inactive)
    }

    pub(crate) fn inactive(&self) -> &TensorData {
        &*self.inactive
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
        let active = slot.active;
        slot.values[active] = Some(value);
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

    pub(crate) fn bank_request(&self) -> HostBufferBankRequest {
        HostBufferBankRequest {
            inner: self.inner.clone(),
            slot: self.slot,
            generation: self.generation,
            descriptor: self.descriptor.clone(),
        }
    }

    pub(crate) fn read_request(&self) -> HostBufferReadRequest {
        HostBufferReadRequest {
            inner: self.inner.clone(),
            slot: self.slot,
            generation: self.generation,
            descriptor: self.descriptor.clone(),
        }
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
        slot.values = [None, None];
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
        let active = slot.active;
        slot.values[active]
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
    fn inactive_bank_transaction_hides_failure_and_flips_once() {
        let pool = HostSlotPool::new();
        let lease = pool.lease(Some(7), desc(7, [2])).unwrap();
        lease
            .write(TensorData::new([2], vec![1.0, 2.0]).unwrap())
            .unwrap();
        let requests = [lease.bank_request()];
        let failed = pool.transact_inactive_banks(&requests, |banks| {
            let (active, inactive) = banks[0].tensors();
            assert_eq!(active.to_vec_f64(), vec![1.0, 2.0]);
            *inactive = TensorData::new([2], vec![9.0, 8.0]).unwrap();
            Err::<(), _>("reject")
        });
        assert!(matches!(
            failed,
            Err(HostBufferBankTransactionError::Stage("reject"))
        ));
        assert_eq!(
            lease.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![1.0, 2.0]
        );

        let malformed = pool.transact_inactive_banks(&requests, |banks| {
            let (_, inactive) = banks[0].tensors();
            *inactive = TensorData::new([1], vec![9.0]).unwrap();
            Ok::<_, ()>(())
        });
        assert!(matches!(
            malformed,
            Err(HostBufferBankTransactionError::Host(
                HostBufferError::IncompatibleDescriptor
            ))
        ));
        assert_eq!(
            lease.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![1.0, 2.0]
        );

        pool.transact_inactive_banks(&requests, |banks| {
            let (active, inactive) = banks[0].tensors();
            assert_eq!(active.to_vec_f64(), vec![1.0, 2.0]);
            *inactive = TensorData::new([2], vec![5.0, 6.0]).unwrap();
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(
            lease.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![5.0, 6.0]
        );
    }

    #[test]
    fn active_tensor_reads_authenticate_complete_requests_and_never_flip() {
        let pool = HostSlotPool::new();
        let lease = pool.lease(Some(7), desc(7, [2])).unwrap();
        lease
            .write(TensorData::new([2], vec![1.0, 2.0]).unwrap())
            .unwrap();
        let request = lease.read_request();
        let rejected = pool.with_active_tensors(std::slice::from_ref(&request), |reads| {
            assert_eq!(reads.len(), 1);
            assert_eq!(reads[0].ordinal(), 0);
            assert_eq!(reads[0].buffer_id(), 7);
            assert_eq!(reads[0].tensor().to_vec_f64(), vec![1.0, 2.0]);
            Err::<(), _>("reject")
        });
        assert!(matches!(
            rejected,
            Err(HostBufferReadError::Callback("reject"))
        ));
        assert_eq!(
            lease.view().unwrap().tensor().unwrap().to_vec_f64(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            pool.with_active_tensors(std::slice::from_ref(&request), |reads| {
                Ok::<_, ()>(reads[0].tensor().to_vec_f64())
            })
            .unwrap(),
            vec![1.0, 2.0]
        );

        assert!(matches!(
            pool.with_active_tensors(&[lease.read_request(), lease.read_request()], |_| {
                Ok::<_, ()>(())
            }),
            Err(HostBufferReadError::Host(HostBufferError::OwnerMismatch))
        ));

        let foreign_pool = HostSlotPool::new();
        assert!(matches!(
            foreign_pool.with_active_tensors(std::slice::from_ref(&request), |_| Ok::<_, ()>(())),
            Err(HostBufferReadError::Host(HostBufferError::OwnerMismatch))
        ));

        let mut malformed = lease.read_request();
        malformed.descriptor.shape = Shape::from([1]);
        malformed.descriptor.bytes = 4;
        assert!(matches!(
            pool.with_active_tensors(std::slice::from_ref(&malformed), |_| Ok::<_, ()>(())),
            Err(HostBufferReadError::Host(
                HostBufferError::IncompatibleDescriptor
            ))
        ));

        let window = lease.view().unwrap();
        let mutable = window.mutable_window(0, 4).unwrap();
        assert!(matches!(
            pool.with_active_tensors(std::slice::from_ref(&request), |_| Ok::<_, ()>(())),
            Err(HostBufferReadError::Host(
                HostBufferError::OutstandingBorrow { slot: 7 }
            ))
        ));
        drop(mutable);
        drop(window);

        let mut stale_lease = lease;
        let stale = stale_lease.read_request();
        stale_lease.release().unwrap();
        let replacement = pool.lease(Some(7), desc(7, [2])).unwrap();
        replacement
            .write(TensorData::new([2], vec![8.0, 9.0]).unwrap())
            .unwrap();
        assert!(matches!(
            pool.with_active_tensors(std::slice::from_ref(&stale), |_| Ok::<_, ()>(())),
            Err(HostBufferReadError::Host(
                HostBufferError::StaleGeneration { slot: 7, .. }
            ))
        ));
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
