//! Complete owned-module momentum-SGD checkpoints.

use super::{
    CompiledModuleCheckpoint, CompiledModuleCheckpointPayload, CompiledMomentumSgdCheckpoint,
};
use crate::Result;
use std::borrow::Cow;
use std::collections::BTreeSet;

/// Portable state for one complete owned compiled-module momentum-SGD
/// lifecycle.
///
/// The shared deterministic envelope retains the exact optimizer checkpoint,
/// canonical topology, tied aliases, and immutable parameter/buffer values.
pub type CompiledModuleMomentumSgdCheckpoint =
    CompiledModuleCheckpoint<CompiledMomentumSgdCheckpoint>;

impl CompiledModuleCheckpointPayload for CompiledMomentumSgdCheckpoint {
    const MODULE_FORMAT: &'static str = "rustgrad-compiled-module-momentum-sgd-v1";

    fn decode_module_payload(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }

    fn encode_module_payload(&self) -> Result<Cow<'_, [u8]>> {
        Ok(Cow::Owned(self.to_bytes()?))
    }

    fn module_parameter_names(&self) -> Result<BTreeSet<String>> {
        Ok(self.parameters().keys().cloned().collect())
    }
}
