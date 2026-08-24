//! Strict hybrid OpenCL replay: device pure prefixes, host atomic effects.
use super::CapturedMixedBatch;
use crate::runtime::opencl::{OpenClContext, OpenClRenderer, PreparedOpenClPrefix};
use crate::{
    EffectBatch, EffectBatchEntry, EffectBatchStep, EffectRuntime, ReplayError, TensorData,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenClMixedBatchTrace {
    pub identity: u64,
    pub prepared_cache_keys: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct OpenClMixedBatchResult {
    pub committed: Vec<crate::BufferState>,
    pub trace: OpenClMixedBatchTrace,
}

impl CapturedMixedBatch {
    pub fn replay_opencl(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        context: OpenClContext,
        renderer: OpenClRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<OpenClMixedBatchResult, ReplayError> {
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
            let bound_capture = crate::engine::mixed_capture::BoundMixedCapture::bind(
                capture,
                &candidates,
                starts,
                provided,
            )?;
            bound.push(bound_capture);
        }
        let mut prepared = Vec::new();
        for capture in &bound {
            let split = capture
                .capture()
                .schedule
                .items
                .iter()
                .position(crate::ScheduleItem::is_effect)
                .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
            prepared.push(
                PreparedOpenClPrefix::prepare(
                    context.clone(),
                    &capture.capture().schedule.items[..split],
                    renderer.clone(),
                )
                .map_err(|e| ReplayError::Execute(format!("OpenCL prepare: {e:?}")))?,
            );
        }
        let mut entries: Vec<EffectBatchEntry> = Vec::new();
        let mut keys = Vec::new();
        for (bound_capture, prefix) in bound.iter().zip(&prepared) {
            let mut values = bound_capture.capture().schedule.constants.clone();
            for input in &bound_capture.capture().schedule.inputs {
                values.insert(
                    input.desc.id,
                    bound_capture
                        .inputs()
                        .get(&input.name)
                        .cloned()
                        .ok_or_else(|| ReplayError::Missing(input.name.clone()))?,
                );
            }
            prefix
                .execute(&mut values)
                .map_err(|e| ReplayError::Execute(format!("OpenCL execute: {e:?}")))?;
            keys.extend(prefix.kernel_cache_keys());
            entries.push(bound_capture.capture().stage_values(
                &mut candidates,
                bound_capture.starts().clone(),
                values,
            )?);
        }
        let batch = EffectBatch::new(entries)
            .map_err(|e| ReplayError::Execute(format!("batch validate: {e:?}")))?;
        let committed = runtime
            .execute_batch(&batch, injected_failure)
            .map_err(|e| ReplayError::Execute(format!("batch commit: {e:?}")))?;
        Ok(OpenClMixedBatchResult {
            committed,
            trace: OpenClMixedBatchTrace {
                identity: self.identity(),
                prepared_cache_keys: keys,
            },
        })
    }
}
