//! Strict hybrid WebGPU replay: retained pure prefixes, host-atomic effects.
use super::{CapturedMixedBatch, backend};
use crate::runtime::webgpu::{PreparedWebGpuPrefix, WebGpuDevice, WgslRenderer};
use crate::{EffectBatchStep, EffectRuntime, ReplayError, ScheduleItem, TensorData};
use std::collections::BTreeMap;

/// Logical trace for a WebGPU mixed batch. Native resources and current bytes
/// intentionally do not participate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebGpuMixedBatchTrace {
    pub identity: u64,
    pub prepared_cache_keys: Vec<String>,
}

/// Host-committed outcome of a retained WebGPU prefix batch.
#[derive(Clone, Debug)]
pub struct WebGpuMixedBatchResult {
    pub committed: Vec<crate::BufferState>,
    pub trace: WebGpuMixedBatchTrace,
}

struct WebGpuBackend {
    device: WebGpuDevice,
    renderer: WgslRenderer,
}

impl backend::PreparedBackend for WebGpuBackend {
    type Prepared = PreparedWebGpuPrefix;

    fn prepare(&self, items: &[ScheduleItem]) -> Result<Self::Prepared, ReplayError> {
        PreparedWebGpuPrefix::prepare(self.device.clone(), items, self.renderer.clone())
            .map_err(|error| ReplayError::Execute(format!("WebGPU prepare: {error:?}")))
    }

    fn execute(
        &self,
        prepared: &Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), ReplayError> {
        prepared
            .execute(values)
            .map_err(|error| ReplayError::Execute(format!("WebGPU execute: {error:?}")))
    }

    fn keys(&self, prepared: &Self::Prepared) -> Vec<String> {
        prepared.kernel_cache_keys()
    }
}

impl CapturedMixedBatch {
    /// Executes prepared WebGPU pure prefixes into detached values, then makes
    /// persistent effects visible through exactly one host transaction.
    pub fn replay_webgpu(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        device: WebGpuDevice,
        renderer: WgslRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<WebGpuMixedBatchResult, ReplayError> {
        let (committed, prepared_cache_keys) = backend::replay(
            self,
            runtime,
            inputs,
            &WebGpuBackend { device, renderer },
            injected_failure,
        )?;
        Ok(WebGpuMixedBatchResult {
            committed,
            trace: WebGpuMixedBatchTrace {
                identity: self.identity(),
                prepared_cache_keys,
            },
        })
    }
}
