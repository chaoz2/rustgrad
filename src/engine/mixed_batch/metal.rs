//! Strict hybrid Metal replay: retained Metal pure prefixes, host-atomic effects.
use super::{CapturedMixedBatch, backend};
use crate::runtime::metal::{MetalDevice, MetalPrefixPlan, MetalRenderer, PreparedMetalPrefix};
use crate::{EffectBatchStep, EffectRuntime, ReplayError, ScheduleItem, TensorData};
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

struct MetalBackend {
    device: MetalDevice,
    renderer: MetalRenderer,
}
impl backend::PreparedBackend for MetalBackend {
    type Plan = MetalPrefixPlan;
    type Prepared = PreparedMetalPrefix;
    fn plan(
        &self,
        items: &[ScheduleItem],
        retained_outputs: &[u64],
    ) -> Result<Self::Plan, ReplayError> {
        MetalPrefixPlan::plan_for_outputs(items, retained_outputs, self.renderer.clone())
            .map_err(|e| ReplayError::Execute(format!("Metal prepare: {e:?}")))
    }
    fn prepare(&self, plan: Self::Plan) -> Result<Self::Prepared, ReplayError> {
        PreparedMetalPrefix::from_plan(self.device.clone(), plan)
            .map_err(|e| ReplayError::Execute(format!("Metal prepare: {e:?}")))
    }
    fn execute(
        &self,
        prepared: &mut Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), ReplayError> {
        prepared
            .execute(values)
            .map_err(|e| ReplayError::Execute(format!("Metal execute: {e:?}")))
    }
    fn keys(&self, prepared: &Self::Prepared) -> Vec<String> {
        prepared.kernel_cache_keys()
    }
}

impl CapturedMixedBatch {
    pub fn replay_metal_with_rebindings(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        rebindings: &[crate::MixedStateRebinding],
        device: MetalDevice,
        renderer: MetalRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<MetalMixedBatchResult, ReplayError> {
        let mut result = self.rebound(rebindings)?.replay_metal(
            runtime,
            inputs,
            device,
            renderer,
            injected_failure,
        )?;
        result.trace.identity = (result.trace.identity
            ^ super::rebinding_schema_identity(rebindings))
        .wrapping_mul(0x100000001b3);
        Ok(result)
    }

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
        let (committed, prepared_cache_keys) = backend::replay(
            self,
            runtime,
            inputs,
            &MetalBackend { device, renderer },
            injected_failure,
        )?;
        Ok(MetalMixedBatchResult {
            committed,
            trace: MetalMixedBatchTrace {
                identity: self.identity(),
                prepared_cache_keys,
            },
        })
    }
}
