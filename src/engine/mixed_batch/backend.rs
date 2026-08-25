//! Shared deterministic coordinator for prepared device pure prefixes.
use super::CapturedMixedBatch;
use crate::{
    EffectBatch, EffectBatchEntry, EffectBatchStep, EffectRuntime, ReplayError, ScheduleItem,
    TensorData,
};
use std::collections::BTreeMap;

pub(super) trait PreparedBackend {
    type Prepared;
    fn prepare(&self, items: &[ScheduleItem]) -> Result<Self::Prepared, ReplayError>;
    fn execute(
        &self,
        prepared: &Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), ReplayError>;
    fn keys(&self, prepared: &Self::Prepared) -> Vec<String>;
}

pub(super) fn replay<B: PreparedBackend>(
    batch: &CapturedMixedBatch,
    runtime: &mut EffectRuntime,
    inputs: &[BTreeMap<String, TensorData>],
    backend: &B,
    injected: Option<EffectBatchStep>,
) -> Result<(Vec<crate::BufferState>, Vec<String>), ReplayError> {
    if inputs.len() != batch.captures.len() {
        return Err(ReplayError::Descriptor("mixed batch input count".into()));
    }
    let mut latest = BTreeMap::new();
    let mut candidates = BTreeMap::new();
    let mut bound = Vec::new();
    for (capture, provided) in batch.captures.iter().zip(inputs) {
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
        bound.push(crate::engine::mixed_capture::BoundMixedCapture::bind(
            capture,
            &candidates,
            starts,
            provided,
        )?);
    }
    let prepared = bound
        .iter()
        .map(|capture| {
            let split = capture
                .capture()
                .schedule
                .items
                .iter()
                .position(crate::ScheduleItem::is_effect)
                .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
            backend.prepare(&capture.capture().schedule.items[..split])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut entries: Vec<EffectBatchEntry> = Vec::new();
    let mut keys = Vec::new();
    for (capture, prefix) in bound.iter().zip(&prepared) {
        let mut values = capture.capture().schedule.constants.clone();
        for input in &capture.capture().schedule.inputs {
            values.insert(
                input.desc.id,
                capture
                    .inputs()
                    .get(&input.name)
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing(input.name.clone()))?,
            );
        }
        backend.execute(prefix, &mut values)?;
        keys.extend(backend.keys(prefix));
        entries.push(capture.capture().stage_values(
            &mut candidates,
            capture.starts().clone(),
            super::super::captured_replay::ReplayValues::from_materialized(values),
        )?);
    }
    let effects = EffectBatch::new(entries)
        .map_err(|e| ReplayError::Execute(format!("batch validate: {e:?}")))?;
    Ok((
        runtime
            .execute_batch(&effects, injected)
            .map_err(|e| ReplayError::Execute(format!("batch commit: {e:?}")))?,
        keys,
    ))
}
