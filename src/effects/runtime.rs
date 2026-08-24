//! Persistent, generation-checked host ownership for effect states.
use super::{BufferState, EffectError, EffectPlan};
use crate::TensorData;
use crate::host_buffer::{
    HostBufferDesc, HostBufferError, HostBufferLease, HostPoolStats, HostSlotPool,
};
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

/// Read-only identity for a persistent logical buffer generation. It is not a
/// raw address and cannot be used to access host storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentSlotIdentity {
    pub slot: u64,
    pub generation: u64,
}

/// Read-only host-pool liveness accounting for effect-runtime diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentRuntimeStats {
    pub physical_slots: usize,
    pub leased_slots: usize,
    pub live_views: usize,
    pub mutable_windows: usize,
    pub zero_byte_sentinels: usize,
}

impl From<HostPoolStats> for PersistentRuntimeStats {
    fn from(stats: HostPoolStats) -> Self {
        Self {
            physical_slots: stats.physical_slots,
            leased_slots: stats.leased_slots,
            live_views: stats.live_views,
            mutable_windows: stats.mutable_windows,
            zero_byte_sentinels: stats.zero_byte_sentinels,
        }
    }
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

    pub fn slot_identity(
        &self,
        state: &BufferState,
    ) -> Result<PersistentSlotIdentity, RuntimeError> {
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
        Ok(PersistentSlotIdentity {
            slot: slot.lease.slot(),
            generation: slot.lease.generation(),
        })
    }

    pub fn stats(&self) -> Result<PersistentRuntimeStats, RuntimeError> {
        Ok(self.pool.stats()?.into())
    }
    /// Preflights all reads and assignment candidates before writing any live
    /// lease. Candidate construction uses immutable version snapshots.
    pub fn execute(
        &mut self,
        plan: &EffectPlan,
        injected_failure: Option<u64>,
    ) -> Result<Vec<BufferState>, RuntimeError> {
        self.execute_with_sources(plan, &BTreeMap::new(), injected_failure)
    }

    /// Internal mixed-schedule boundary: values were already materialized by
    /// a pure schedule and are substituted only for the matching STORE source.
    /// They remain owned transaction inputs; no raw host allocation leaks.
    pub(crate) fn execute_with_sources(
        &mut self,
        plan: &EffectPlan,
        sources: &BTreeMap<u64, TensorData>,
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
        // Validate every persistent target before candidate construction. This
        // leaves no missing-slot or descriptor path after the transaction
        // reaches its pool-wide visible commit.
        for step in &plan.steps {
            let slot = self
                .slots
                .get(&step.write.buffer)
                .ok_or(RuntimeError::MissingBuffer(step.write.buffer))?;
            if slot.state.shape != step.write.shape
                || slot.state.dtype != step.write.dtype
                || slot.state.bytes != step.write.bytes
            {
                return Err(RuntimeError::StaleState {
                    buffer: step.write.buffer,
                    version: step.write.version,
                });
            }
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
            let source = if let Some(source) = sources.get(&step.id) {
                if source.shape() != &step.reads[1].shape || source.dtype() != step.reads[1].dtype {
                    return Err(RuntimeError::Effect(EffectError::DescriptorMismatch {
                        buffer: step.reads[1].buffer,
                        version: step.reads[1].version,
                    }));
                }
                source
            } else {
                snapshots
                    .get(&(step.reads[1].buffer, step.reads[1].version))
                    .ok_or(RuntimeError::StaleState {
                        buffer: step.reads[1].buffer,
                        version: step.reads[1].version,
                    })?
            };
            let mut candidate = target;
            if let Some(view) = &step.target_view {
                candidate.assign_view_from(view, source)
            } else {
                candidate.assign_from(source)
            }
            .map_err(|_| EffectError::TransactionFailed { step: step.id })?;
            snapshots.insert((step.write.buffer, step.write.version), candidate.clone());
            candidates.push((step.write.clone(), candidate));
        }
        // Multiple versions of one buffer commit only their final candidate;
        // all intermediate versions remain immutable transaction snapshots.
        let mut final_values = BTreeMap::new();
        for (state, value) in &candidates {
            final_values.insert(state.buffer, (state.clone(), value.clone()));
        }
        let mut writes = Vec::with_capacity(final_values.len());
        for (buffer, (_, value)) in &final_values {
            let slot = self
                .slots
                .get(buffer)
                .expect("prevalidated persistent target");
            writes.push(slot.lease.staged_write(value.clone())?);
        }
        self.pool.commit(writes)?;
        for (state, _) in final_values.values() {
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

    fn data(shape: impl Into<Shape>, storage: Storage) -> TensorData {
        TensorData::from_storage(shape, storage).unwrap()
    }

    fn exact_storage(left: &Storage, right: &Storage) -> bool {
        match (left, right) {
            (Storage::F32(left), Storage::F32(right)) => left
                .iter()
                .map(|value| value.to_bits())
                .eq(right.iter().map(|value| value.to_bits())),
            (Storage::F64(left), Storage::F64(right)) => left
                .iter()
                .map(|value| value.to_bits())
                .eq(right.iter().map(|value| value.to_bits())),
            _ => left == right,
        }
    }

    fn persistent_matches_detached(target: TensorData, source: TensorData) {
        let mut graph = EffectGraph::default();
        let target_handle = graph.insert(10, target.clone()).unwrap();
        let source_handle = graph.insert(20, source.clone()).unwrap();
        let next = graph.assign(&target_handle, &source_handle).unwrap();
        let detached = graph.execute().unwrap();

        let mut runtime = EffectRuntime::new();
        runtime.register(10, target).unwrap();
        runtime.register(20, source).unwrap();
        runtime.execute(&graph.plan(), None).unwrap();
        assert!(exact_storage(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            detached.values[&10].storage(),
        ));
    }

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

    #[test]
    fn persistent_assignments_preserve_every_dense_storage_variant() {
        let cases = vec![
            (
                "bool",
                data([2, 2], Storage::Bool(vec![false; 4])),
                data([1], Storage::Bool(vec![true])),
            ),
            (
                "i8",
                data([2, 2], Storage::I8(vec![0; 4])),
                data([1], Storage::I8(vec![-7])),
            ),
            (
                "u8",
                data([2, 2], Storage::U8(vec![0; 4])),
                data([1], Storage::U8(vec![9])),
            ),
            (
                "i16",
                data([2, 2], Storage::I16(vec![0; 4])),
                data([1], Storage::I16(vec![-70])),
            ),
            (
                "u16",
                data([2, 2], Storage::U16(vec![0; 4])),
                data([1], Storage::U16(vec![90])),
            ),
            (
                "i32",
                data([2, 2], Storage::I32(vec![0; 4])),
                data([1], Storage::I32(vec![-700])),
            ),
            (
                "u32",
                data([2, 2], Storage::U32(vec![0; 4])),
                data([1], Storage::U32(vec![900])),
            ),
            (
                "i64",
                data([2, 2], Storage::I64(vec![0; 4])),
                data([1], Storage::I64(vec![-7000])),
            ),
            (
                "u64",
                data([2, 2], Storage::U64(vec![0; 4])),
                data([1], Storage::U64(vec![9000])),
            ),
            (
                "f16 raw",
                data([2, 2], Storage::F16(vec![0; 4])),
                data([1], Storage::F16(vec![0x7e55])),
            ),
            (
                "bf16 raw",
                data([2, 2], Storage::BF16(vec![0; 4])),
                data([1], Storage::BF16(vec![0x7fc1])),
            ),
            (
                "f32",
                data([2, 2], Storage::F32(vec![0.0; 4])),
                data([1], Storage::F32(vec![-0.0])),
            ),
            (
                "f64",
                data([2, 2], Storage::F64(vec![0.0; 4])),
                data([1], Storage::F64(vec![f64::NAN])),
            ),
        ];
        for (name, target, source) in cases {
            persistent_matches_detached(target, source);
            assert!(!name.is_empty());
        }
        persistent_matches_detached(
            data([], Storage::I32(vec![0])),
            data([], Storage::I32(vec![7])),
        );
        persistent_matches_detached(
            data([0, 2], Storage::U64(vec![])),
            data([1, 2], Storage::U64(vec![3, 4])),
        );
    }

    #[test]
    fn persistent_chains_and_diamonds_keep_snapshot_reads_and_deterministic_identities() {
        let mut graph = EffectGraph::default();
        let a = graph
            .insert(1, data([2], Storage::I32(vec![0, 0])))
            .unwrap();
        let b = graph
            .insert(2, data([2], Storage::I32(vec![3, 3])))
            .unwrap();
        let c = graph
            .insert(3, data([2], Storage::I32(vec![9, 9])))
            .unwrap();
        let a1 = graph.assign(&a, &b).unwrap();
        let b1 = graph.assign(&b, &a1).unwrap();
        let c1 = graph.assign(&c, &a1).unwrap();
        let a2 = graph.assign(&a1, &c1).unwrap();
        let detached = graph.execute().unwrap();

        let mut runtime = EffectRuntime::new();
        runtime
            .register(1, data([2], Storage::I32(vec![0, 0])))
            .unwrap();
        runtime
            .register(2, data([2], Storage::I32(vec![3, 3])))
            .unwrap();
        runtime
            .register(3, data([2], Storage::I32(vec![9, 9])))
            .unwrap();
        let retained = runtime.snapshot(a.state()).unwrap();
        let before = runtime.slot_identity(a.state()).unwrap();
        let committed = runtime.execute(&graph.plan(), None).unwrap();
        assert_eq!(
            committed,
            vec![
                a1.state().clone(),
                b1.state().clone(),
                c1.state().clone(),
                a2.state().clone()
            ]
        );
        assert_eq!(retained.tensor().storage(), &Storage::I32(vec![0, 0]));
        assert!(matches!(
            runtime.snapshot(a.state()),
            Err(RuntimeError::StaleState { .. })
        ));
        assert_eq!(
            runtime.snapshot(a2.state()).unwrap().tensor().storage(),
            detached.values[&1].storage()
        );
        assert_eq!(
            runtime.snapshot(b1.state()).unwrap().tensor().storage(),
            detached.values[&2].storage()
        );
        assert_eq!(
            runtime.snapshot(c1.state()).unwrap().tensor().storage(),
            detached.values[&3].storage()
        );
        let after = runtime.slot_identity(a2.state()).unwrap();
        assert_eq!(before.slot, after.slot);
        assert_eq!(before.generation, after.generation);
        assert_eq!(runtime.stats().unwrap().leased_slots, 3);
    }

    #[test]
    fn persistent_preflight_rejects_bad_states_without_mutation_and_can_retry() {
        let mut graph = EffectGraph::default();
        let a = graph
            .insert(1, data([2], Storage::I32(vec![1, 2])))
            .unwrap();
        let b = graph.insert(2, data([1], Storage::I32(vec![7]))).unwrap();
        let next = graph.assign(&a, &b).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(1, data([2], Storage::I32(vec![1, 2])))
            .unwrap();
        runtime
            .register(2, data([1], Storage::I32(vec![7])))
            .unwrap();
        let before = runtime.stats().unwrap();
        assert!(matches!(
            runtime.register(1, data([2], Storage::I32(vec![0, 0]))),
            Err(RuntimeError::DuplicateBuffer(1))
        ));
        assert!(matches!(
            runtime.snapshot(&BufferState {
                version: 99,
                ..a.state().clone()
            }),
            Err(RuntimeError::StaleState { .. })
        ));

        let mut malformed = graph.plan();
        malformed.steps[0].reads[0].buffer = 2;
        assert!(matches!(
            runtime.execute(&malformed, None),
            Err(RuntimeError::Effect(EffectError::DescriptorMismatch { .. }))
        ));
        assert_eq!(
            runtime.snapshot(a.state()).unwrap().tensor().storage(),
            &Storage::I32(vec![1, 2])
        );
        assert_eq!(runtime.stats().unwrap(), before);
        assert!(matches!(
            runtime.execute(&graph.plan(), Some(0)),
            Err(RuntimeError::InjectedFailure(0))
        ));
        assert_eq!(
            runtime.snapshot(a.state()).unwrap().tensor().storage(),
            &Storage::I32(vec![1, 2])
        );
        runtime.execute(&graph.plan(), None).unwrap();
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::I32(vec![7, 7])
        );
    }

    #[test]
    fn persistent_registration_tracks_zero_byte_sentinels_without_reusable_slots() {
        let mut runtime = EffectRuntime::new();
        let zero = runtime
            .register(7, data([0, 3], Storage::F16(vec![])))
            .unwrap();
        let nonzero = runtime
            .register(8, data([1], Storage::F16(vec![0x3c00])))
            .unwrap();
        let stats = runtime.stats().unwrap();
        assert_eq!(stats.physical_slots, 1);
        assert_eq!(stats.zero_byte_sentinels, 1);
        assert_eq!(stats.leased_slots, 2);
        assert_eq!(runtime.slot_identity(&zero).unwrap().slot, u64::MAX);
        assert_eq!(runtime.slot_identity(&nonzero).unwrap().slot, 8);
    }
}
