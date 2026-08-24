//! Strict hybrid Metal replay: retained Metal pure prefixes, host-atomic effects.
use super::CapturedMixedBatch;
use crate::runtime::metal::{MetalDevice, MetalRenderer, PreparedMetalPrefix};
use crate::{
    EffectBatch, EffectBatchEntry, EffectBatchStep, EffectRuntime, ReplayError, TensorData,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalMixedBatchTrace {
    pub identity: u64,
    pub prepared_cache_keys: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct MetalMixedBatchResult {
    pub committed: Vec<crate::BufferState>,
    pub trace: MetalMixedBatchTrace,
}

impl CapturedMixedBatch {
    /// Executes only prepared static pure prefixes on Metal. Persistent state
    /// remains host-owned and becomes visible only at the one batch commit.
    pub fn replay_metal(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        device: MetalDevice,
        renderer: MetalRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<MetalMixedBatchResult, ReplayError> {
        if inputs.len() != self.captures.len() {
            return Err(ReplayError::Descriptor("mixed batch input count".into()));
        }
        let mut latest = BTreeMap::new();
        let mut candidates = BTreeMap::new();
        let mut bound = Vec::new();
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
                    .ok_or_else(|| {
                        ReplayError::Unsupported("mixed capture has no effects".into())
                    })?;
                PreparedMetalPrefix::prepare(
                    device.clone(),
                    &capture.capture().schedule.items[..split],
                    renderer.clone(),
                )
                .map_err(|e| ReplayError::Execute(format!("Metal prepare: {e:?}")))
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
            prefix
                .execute(&mut values)
                .map_err(|e| ReplayError::Execute(format!("Metal execute: {e:?}")))?;
            keys.extend(prefix.kernel_cache_keys());
            entries.push(capture.capture().stage_values(
                &mut candidates,
                capture.starts().clone(),
                values,
            )?);
        }
        let batch = EffectBatch::new(entries)
            .map_err(|e| ReplayError::Execute(format!("batch validate: {e:?}")))?;
        let committed = runtime
            .execute_batch(&batch, injected_failure)
            .map_err(|e| ReplayError::Execute(format!("batch commit: {e:?}")))?;
        Ok(MetalMixedBatchResult {
            committed,
            trace: MetalMixedBatchTrace {
                identity: self.identity(),
                prepared_cache_keys: keys,
            },
        })
    }
}
