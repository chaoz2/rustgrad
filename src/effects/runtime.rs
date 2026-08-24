//! Persistent, generation-checked host ownership for effect states.
use super::{BufferState, EffectError, EffectPlan};
use crate::TensorData;
use crate::host_buffer::{HostBufferDesc, HostBufferError, HostBufferLease, HostSlotPool};
use std::collections::BTreeMap;

#[derive(Debug)]
pub enum RuntimeError {
    Effect(EffectError),
    Host(HostBufferError),
    DuplicateBuffer(u64),
    MissingBuffer(u64),
    StaleState { buffer: u64, version: u64 },
    InjectedFailure(u64),
}
impl From<EffectError> for RuntimeError {
    fn from(value: EffectError) -> Self {
        Self::Effect(value)
    }
}
impl From<HostBufferError> for RuntimeError {
    fn from(value: HostBufferError) -> Self {
        Self::Host(value)
    }
}

struct PersistentStateSlot {
    state: BufferState,
    lease: HostBufferLease,
}

/// Owns persistent effect storage. Logical identities, rather than host
/// addresses or generations, are the stable effect/cache identity.
pub struct EffectRuntime {
    pool: HostSlotPool,
    slots: BTreeMap<u64, PersistentStateSlot>,
}

#[derive(Clone, Debug)]
pub struct PersistentSnapshot {
    pub state: BufferState,
    value: TensorData,
}
impl PersistentSnapshot {
    pub fn tensor(&self) -> &TensorData {
        &self.value
    }
}

impl EffectRuntime {
    pub fn new() -> Self {
        Self {
            pool: HostSlotPool::new(),
            slots: BTreeMap::new(),
        }
    }
    pub fn register(
        &mut self,
        buffer: u64,
        value: TensorData,
    ) -> Result<BufferState, RuntimeError> {
        if self.slots.contains_key(&buffer) {
            return Err(RuntimeError::DuplicateBuffer(buffer));
        }
        let bytes = value
            .len()
            .checked_mul(value.dtype().itemsize())
            .ok_or(HostBufferError::Overflow)?;
        let state = BufferState {
            buffer,
            version: 0,
            shape: value.shape().clone(),
            dtype: value.dtype(),
            bytes,
        };
        let desc = HostBufferDesc {
            buffer_id: buffer,
            dtype: state.dtype,
            shape: state.shape.clone(),
            bytes,
            alignment: state.dtype.itemsize().max(1),
            lanes: 1,
        };
        // Zero-byte values receive pool-private sentinels; they are never a
        // reusable physical allocation.
        let lease = self.pool.lease((bytes != 0).then_some(buffer), desc)?;
        lease.write(value)?;
        self.slots.insert(
            buffer,
            PersistentStateSlot {
                state: state.clone(),
                lease,
            },
        );
        Ok(state)
    }
    pub fn snapshot(&self, state: &BufferState) -> Result<PersistentSnapshot, RuntimeError> {
        let slot = self
            .slots
            .get(&state.buffer)
            .ok_or(RuntimeError::MissingBuffer(state.buffer))?;
        if slot.state != *state {
            return Err(RuntimeError::StaleState {
                buffer: state.buffer,
                version: state.version,
            });
        }
        let view = slot.lease.view()?;
        Ok(PersistentSnapshot {
            state: state.clone(),
            value: view.tensor()?,
        })
    }
    /// Preflights all reads and assignment candidates before writing any live
    /// lease. Candidate construction uses immutable version snapshots.
    pub fn execute(
        &mut self,
        plan: &EffectPlan,
        injected_failure: Option<u64>,
    ) -> Result<Vec<BufferState>, RuntimeError> {
        plan.validate()?;
        let mut snapshots = BTreeMap::new();
        for (buffer, slot) in &self.slots {
            snapshots.insert(
                (*buffer, slot.state.version),
                self.snapshot(&slot.state)?.value,
            );
        }
        let mut candidates = Vec::new();
        for step in &plan.steps {
            if injected_failure == Some(step.id) {
                return Err(RuntimeError::InjectedFailure(step.id));
            }
            let target = snapshots
                .get(&(step.reads[0].buffer, step.reads[0].version))
                .ok_or(RuntimeError::StaleState {
                    buffer: step.reads[0].buffer,
                    version: step.reads[0].version,
                })?
                .clone();
            let source = snapshots
                .get(&(step.reads[1].buffer, step.reads[1].version))
                .ok_or(RuntimeError::StaleState {
                    buffer: step.reads[1].buffer,
                    version: step.reads[1].version,
                })?;
            let mut candidate = target;
            candidate
                .assign_from(source)
                .map_err(|_| EffectError::TransactionFailed { step: step.id })?;
            snapshots.insert((step.write.buffer, step.write.version), candidate.clone());
            candidates.push((step.write.clone(), candidate));
        }
        // All writes are descriptor-checked by the preflight above. No fallible
        // operation remains between the first visible write and completion.
        for (state, value) in &candidates {
            let slot = self
                .slots
                .get_mut(&state.buffer)
                .ok_or(RuntimeError::MissingBuffer(state.buffer))?;
            if slot.state.shape != state.shape
                || slot.state.dtype != state.dtype
                || slot.state.bytes != state.bytes
            {
                return Err(RuntimeError::StaleState {
                    buffer: state.buffer,
                    version: state.version,
                });
            }
            slot.lease.write(value.clone())?;
        }
        for (state, _) in &candidates {
            self.slots
                .get_mut(&state.buffer)
                .expect("preflighted slot")
                .state = state.clone();
        }
        Ok(candidates.into_iter().map(|(state, _)| state).collect())
    }
}
impl Default for EffectRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, EffectGraph, Shape, Storage};

    #[test]
    fn persistent_runtime_stages_failure_and_commits_versions() {
        let mut graph = EffectGraph::default();
        let a = graph
            .insert(
                1,
                TensorData::from_storage([2], Storage::U64(vec![0, 0])).unwrap(),
            )
            .unwrap();
        let b = graph
            .insert(
                2,
                TensorData::from_storage([1], Storage::U64(vec![9])).unwrap(),
            )
            .unwrap();
        let next = graph.assign(&a, &b).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                1,
                TensorData::from_storage([2], Storage::U64(vec![0, 0])).unwrap(),
            )
            .unwrap();
        runtime
            .register(
                2,
                TensorData::from_storage([1], Storage::U64(vec![9])).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            runtime.execute(&graph.plan(), Some(0)),
            Err(RuntimeError::InjectedFailure(0))
        ));
        assert_eq!(
            runtime.snapshot(a.state()).unwrap().tensor().storage(),
            &Storage::U64(vec![0, 0])
        );
        runtime.execute(&graph.plan(), None).unwrap();
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::U64(vec![9, 9])
        );
        assert!(matches!(
            runtime.snapshot(a.state()),
            Err(RuntimeError::StaleState { .. })
        ));
        assert_eq!(next.state().dtype, DType::U64);
        assert_eq!(next.state().shape, Shape::from([2]));
    }
}
