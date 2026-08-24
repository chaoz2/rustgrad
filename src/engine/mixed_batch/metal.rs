//! Strict hybrid Metal replay: retained Metal pure prefixes, host-atomic effects.
use super::{CapturedMixedBatch, backend};
use crate::runtime::metal::{MetalDevice, MetalRenderer, PreparedMetalPrefix};
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
    type Prepared = PreparedMetalPrefix;
    fn prepare(&self, items: &[ScheduleItem]) -> Result<Self::Prepared, ReplayError> {
        PreparedMetalPrefix::prepare(self.device.clone(), items, self.renderer.clone())
            .map_err(|e| ReplayError::Execute(format!("Metal prepare: {e:?}")))
    }
    fn execute(
        &self,
        prepared: &Self::Prepared,
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
