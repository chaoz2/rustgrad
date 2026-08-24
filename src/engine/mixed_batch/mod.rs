//! In-memory batch schema for ordered mixed captures.
//!
//! This module intentionally reuses RGSM and [`crate::EffectBatch`]; it owns
//! no serialization format and never records runtime resource identities.
mod artifact;
mod backend;
mod metal;
mod opencl;
#[cfg(test)]
pub(crate) mod test_support;
mod webgpu;

use super::mixed_capture::CapturedMixedSchedule;
use crate::{
    CapturedReplayExecutor, EffectBatch, EffectBatchEntry, EffectRuntime, ReplayError, TensorData,
};
pub use artifact::MixedBatchArtifactError;
pub use metal::{MetalMixedBatchResult, MetalMixedBatchTrace};
pub use opencl::{OpenClMixedBatchResult, OpenClMixedBatchTrace};
use std::collections::BTreeMap;
pub use webgpu::{WebGpuMixedBatchResult, WebGpuMixedBatchTrace};

/// Ordered, immutable logical batch of decoded mixed captures.
#[derive(Clone, Debug)]
pub struct CapturedMixedBatch {
    captures: Vec<CapturedMixedSchedule>,
    identity: u64,
}

/// Logical strict-native batch trace. Runtime handles, generations, pointers,
/// and current bytes are intentionally absent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMixedBatchTrace {
    pub identity: u64,
    pub batch_identity: u64,
    pub vectorized: bool,
    pub binding_count: usize,
    pub binding_schema_keys: Vec<u64>,
    pub pure_item_cache_keys: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct NativeMixedBatchResult {
    pub committed: Vec<crate::BufferState>,
    pub trace: NativeMixedBatchTrace,
}

impl CapturedMixedBatch {
    /// Validates every constituent RGSM envelope and assigns a stable identity
    /// over its ordered logical bytes. Runtime slots, generations, pointers,
    /// and current storage never participate.
    pub fn new(captures: Vec<CapturedMixedSchedule>) -> Result<Self, ReplayError> {
        if captures.is_empty() {
            return Err(ReplayError::Corrupt("empty mixed batch".into()));
        }
        let mut hash = 0xcbf29ce484222325u64;
        for capture in &captures {
            let bytes = capture.to_bytes()?;
            for byte in bytes {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        Ok(Self {
            captures,
            identity: hash,
        })
    }

    pub fn captures(&self) -> &[CapturedMixedSchedule] {
        &self.captures
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Runs all pure prefixes against detached candidates, then performs the
    /// one visible persistent commit only after every entry staged cleanly.
    pub fn replay(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        injected_failure: Option<crate::EffectBatchStep>,
    ) -> Result<Vec<crate::BufferState>, ReplayError> {
        if inputs.len() != self.captures.len() {
            return Err(ReplayError::Descriptor("mixed batch input count".into()));
        }
        let mut latest = BTreeMap::<u64, crate::BufferState>::new();
        let mut candidates = BTreeMap::<crate::BufferState, TensorData>::new();
        let mut entries: Vec<EffectBatchEntry> = Vec::with_capacity(self.captures.len());
        for (capture, provided) in self.captures.iter().zip(inputs) {
            let mut starts = BTreeMap::new();
            for local in capture.initial_states() {
                let state = latest
                    .get(&local.buffer)
                    .cloned()
                    .unwrap_or_else(|| local.clone());
                if !candidates.contains_key(&state) && !latest.contains_key(&local.buffer) {
                    let value = runtime
                        .snapshot(&state)
                        .map_err(|e| ReplayError::Execute(format!("batch preflight: {e:?}")))?
                        .tensor()
                        .clone();
                    candidates.insert(state.clone(), value);
                }
                starts.insert(local.buffer, state);
            }
            let entry = capture.stage_interpreter(&mut candidates, starts, provided)?;
            for step in &entry.plan.steps {
                let start = entry
                    .starts
                    .get(&step.write.buffer)
                    .ok_or_else(|| ReplayError::Corrupt("batch target start".into()))?;
                latest.insert(
                    step.write.buffer,
                    crate::BufferState {
                        version: start
                            .version
                            .checked_add(step.write.version)
                            .ok_or_else(|| ReplayError::Corrupt("batch version overflow".into()))?,
                        ..step.write.clone()
                    },
                );
            }
            entries.push(entry);
        }
        let batch = EffectBatch::new(entries)
            .map_err(|e| ReplayError::Execute(format!("batch validate: {e:?}")))?;
        runtime
            .execute_batch(&batch, injected_failure)
            .map_err(|e| ReplayError::Execute(format!("batch commit: {e:?}")))
    }

    /// Strictly plans and runs every pure prefix natively before the one
    /// effect-runtime batch commit. Unsupported items fail closed.
    pub fn replay_native(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        executor: &CapturedReplayExecutor,
        vectorized: bool,
        injected_failure: Option<crate::EffectBatchStep>,
    ) -> Result<Vec<crate::BufferState>, ReplayError> {
        Ok(self
            .replay_native_traced(runtime, inputs, executor, vectorized, injected_failure)?
            .committed)
    }

    pub fn replay_native_traced(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        executor: &CapturedReplayExecutor,
        vectorized: bool,
        injected_failure: Option<crate::EffectBatchStep>,
    ) -> Result<NativeMixedBatchResult, ReplayError> {
        if inputs.len() != self.captures.len() {
            return Err(ReplayError::Descriptor("mixed batch input count".into()));
        }
        // Bind every capture first. These bindings are logical schemas and
        // detached input values only; no native item has executed yet.
        let mut latest = BTreeMap::<u64, crate::BufferState>::new();
        let mut candidates = BTreeMap::<crate::BufferState, TensorData>::new();
        let mut bound = Vec::with_capacity(self.captures.len());
        for (capture, provided) in self.captures.iter().zip(inputs) {
            let mut starts = BTreeMap::new();
            for local in capture.initial_states() {
                let state = latest
                    .get(&local.buffer)
                    .cloned()
                    .unwrap_or_else(|| local.clone());
                if !candidates.contains_key(&state) && !latest.contains_key(&local.buffer) {
                    candidates.insert(
                        state.clone(),
                        runtime
                            .snapshot(&state)
                            .map_err(|e| ReplayError::Execute(format!("batch preflight: {e:?}")))?
                            .tensor()
                            .clone(),
                    );
                }
                starts.insert(local.buffer, state);
            }
            for state in &capture.states {
                let start = starts
                    .get(&state.buffer)
                    .ok_or_else(|| ReplayError::Corrupt("batch target start".into()))?;
                latest.insert(
                    state.buffer,
                    crate::BufferState {
                        version: start
                            .version
                            .checked_add(state.version)
                            .ok_or_else(|| ReplayError::Corrupt("batch version overflow".into()))?,
                        ..state.clone()
                    },
                );
            }
            bound.push(super::mixed_capture::BoundMixedCapture::bind(
                capture,
                &candidates,
                starts,
                provided,
            )?);
        }
        // Strict native policy is all-or-nothing at the planning boundary.
        let planned = bound
            .into_iter()
            .map(|bound| bound.plan_native(executor, vectorized))
            .collect::<Result<Vec<_>, _>>()?;
        let pure_item_cache_keys = planned
            .iter()
            .flat_map(|planned| planned.cache_keys())
            .collect::<Vec<_>>();
        let binding_schema_keys = planned
            .iter()
            .map(super::mixed_capture::PlannedBoundMixedCapture::binding_schema_key)
            .collect::<Vec<_>>();
        let binding_count = inputs.iter().map(BTreeMap::len).sum::<usize>();
        let mut entries = Vec::with_capacity(planned.len());
        for planned in planned {
            entries.push(planned.execute_stage(&mut candidates, executor)?);
        }
        let batch = EffectBatch::new(entries)
            .map_err(|e| ReplayError::Execute(format!("batch validate: {e:?}")))?;
        let committed = runtime
            .execute_batch(&batch, injected_failure)
            .map_err(|e| ReplayError::Execute(format!("batch commit: {e:?}")))?;
        let mut identity = self.identity ^ u64::from(vectorized);
        for key in &pure_item_cache_keys {
            identity = (identity ^ *key).wrapping_mul(0x100000001b3);
        }
        for key in &binding_schema_keys {
            identity = (identity ^ *key).wrapping_mul(0x100000001b3);
        }
        Ok(NativeMixedBatchResult {
            committed,
            trace: NativeMixedBatchTrace {
                identity,
                batch_identity: self.identity,
                vectorized,
                binding_count,
                binding_schema_keys,
                pure_item_cache_keys,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::indexing::{StaticIndex, StaticIndexPlan};
    use crate::{
        CapturedSchedule, EffectBatchStep, EffectGraph, Shape, Storage, TensorData,
        schedule_effects,
    };

    fn tensor(shape: impl Into<Shape>, storage: Storage) -> TensorData {
        TensorData::from_storage(shape, storage).unwrap()
    }

    fn capture(graph: &EffectGraph, states: Vec<crate::BufferState>) -> CapturedMixedSchedule {
        let schedule = schedule_effects(graph).unwrap();
        CapturedMixedSchedule::from_parts(
            CapturedSchedule {
                items: schedule.items.clone(),
                inputs: vec![],
                constants: BTreeMap::new(),
                quantized_constants: BTreeMap::new(),
                requested: vec![],
                identity: 0,
                symbolic: None,
                specialized_from: None,
            },
            &schedule,
            states,
        )
        .unwrap()
    }

    #[test]
    fn interpreter_batch_rebases_chain_and_defers_all_visibility() {
        let mut first = EffectGraph::default();
        let base = first
            .insert(1, tensor([3], Storage::F16(vec![1, 2, 3])))
            .unwrap();
        let rhs = first
            .insert(2, tensor([3], Storage::F16(vec![9, 8, 7])))
            .unwrap();
        let next = first.assign(&base, &rhs).unwrap();
        let first_capture = capture(
            &first,
            vec![
                base.state().clone(),
                rhs.state().clone(),
                next.state().clone(),
            ],
        );

        let mut second = EffectGraph::default();
        let base_two = second
            .insert(1, tensor([3], Storage::F16(vec![0, 0, 0])))
            .unwrap();
        let rhs_two = second
            .insert(3, tensor([1], Storage::F16(vec![0x8000])))
            .unwrap();
        let plan = StaticIndexPlan::new(
            Shape::from([3]),
            &[StaticIndex::Advanced {
                shape: Shape::from([2]),
                values: vec![1, 1],
            }],
        )
        .unwrap();
        let final_state = second
            .static_index_assign(&base_two, &rhs_two, plan)
            .unwrap();
        let second_capture = capture(
            &second,
            vec![
                base_two.state().clone(),
                rhs_two.state().clone(),
                final_state.state().clone(),
            ],
        );
        let batch =
            CapturedMixedBatch::new(vec![first_capture.clone(), second_capture.clone()]).unwrap();
        assert_eq!(
            batch.identity(),
            CapturedMixedBatch::new(vec![first_capture, second_capture])
                .unwrap()
                .identity()
        );
        let mut runtime = EffectRuntime::new();
        runtime
            .register(1, tensor([3], Storage::F16(vec![1, 2, 3])))
            .unwrap();
        runtime
            .register(2, tensor([3], Storage::F16(vec![9, 8, 7])))
            .unwrap();
        runtime
            .register(3, tensor([1], Storage::F16(vec![0x8000])))
            .unwrap();
        assert!(
            batch
                .replay(
                    &mut runtime,
                    &[BTreeMap::new(), BTreeMap::new()],
                    Some(EffectBatchStep { entry: 1, step: 0 })
                )
                .is_err()
        );
        assert_eq!(
            runtime.snapshot(base.state()).unwrap().tensor().storage(),
            &Storage::F16(vec![1, 2, 3])
        );
        batch
            .replay(&mut runtime, &[BTreeMap::new(), BTreeMap::new()], None)
            .unwrap();
        let rebased_final = crate::BufferState {
            version: 2,
            ..final_state.state().clone()
        };
        assert_eq!(
            runtime.snapshot(&rebased_final).unwrap().tensor().storage(),
            &Storage::F16(vec![9, 0x8000, 7])
        );
    }

    #[test]
    fn interpreter_batch_rejects_missing_input_count_before_mutation() {
        let mut graph = EffectGraph::default();
        let target = graph
            .insert(8, tensor([], Storage::U64(vec![u64::MAX])))
            .unwrap();
        let source = graph.insert(9, tensor([], Storage::U64(vec![7]))).unwrap();
        let next = graph.assign(&target, &source).unwrap();
        let batch = CapturedMixedBatch::new(vec![capture(
            &graph,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )])
        .unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(8, tensor([], Storage::U64(vec![u64::MAX])))
            .unwrap();
        runtime
            .register(9, tensor([], Storage::U64(vec![7])))
            .unwrap();
        assert!(batch.replay(&mut runtime, &[], None).is_err());
        assert_eq!(
            runtime.snapshot(target.state()).unwrap().tensor().storage(),
            &Storage::U64(vec![u64::MAX])
        );
    }

    #[test]
    fn native_batch_chain_is_atomic_and_trace_is_deterministic() {
        let mut first = EffectGraph::default();
        let base = first
            .insert(20, tensor([2], Storage::F16(vec![1, 2])))
            .unwrap();
        let rhs = first
            .insert(21, tensor([2], Storage::F16(vec![9, 8])))
            .unwrap();
        let next = first.assign(&base, &rhs).unwrap();
        let first_capture = capture(
            &first,
            vec![
                base.state().clone(),
                rhs.state().clone(),
                next.state().clone(),
            ],
        );
        let mut second = EffectGraph::default();
        let base_two = second
            .insert(20, tensor([2], Storage::F16(vec![0, 0])))
            .unwrap();
        let rhs_two = second
            .insert(22, tensor([], Storage::F16(vec![0x8000])))
            .unwrap();
        let end = second.assign(&base_two, &rhs_two).unwrap();
        let second_capture = capture(
            &second,
            vec![
                base_two.state().clone(),
                rhs_two.state().clone(),
                end.state().clone(),
            ],
        );
        let batch = CapturedMixedBatch::new(vec![first_capture, second_capture]).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(20, tensor([2], Storage::F16(vec![1, 2])))
            .unwrap();
        runtime
            .register(21, tensor([2], Storage::F16(vec![9, 8])))
            .unwrap();
        runtime
            .register(22, tensor([], Storage::F16(vec![0x8000])))
            .unwrap();
        let native = crate::CapturedReplayExecutor::default();
        assert!(
            batch
                .replay_native(
                    &mut runtime,
                    &[BTreeMap::new(), BTreeMap::new()],
                    &native,
                    false,
                    Some(EffectBatchStep { entry: 1, step: 0 })
                )
                .is_err()
        );
        assert_eq!(
            runtime.snapshot(base.state()).unwrap().tensor().storage(),
            &Storage::F16(vec![1, 2])
        );
        let first = batch
            .replay_native_traced(
                &mut runtime,
                &[BTreeMap::new(), BTreeMap::new()],
                &native,
                false,
                None,
            )
            .unwrap();
        assert_eq!(
            runtime
                .snapshot(&crate::BufferState {
                    version: 2,
                    ..end.state().clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F16(vec![0x8000, 0x8000])
        );
        assert_eq!(first.trace, first.trace.clone());
    }
}
