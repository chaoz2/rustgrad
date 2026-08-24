//! In-memory batch schema for ordered mixed captures.
//!
//! This module intentionally reuses RGSM and [`crate::EffectBatch`]; it owns
//! no serialization format and never records runtime resource identities.
use super::mixed_capture::CapturedMixedSchedule;
use crate::{EffectBatch, EffectBatchEntry, EffectRuntime, ReplayError, TensorData};
use std::collections::BTreeMap;

/// Ordered, immutable logical batch of decoded mixed captures.
#[derive(Clone, Debug)]
pub struct CapturedMixedBatch {
    captures: Vec<CapturedMixedSchedule>,
    identity: u64,
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
                if !candidates.contains_key(&state) {
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
}
