//! Complete owned-module AdamW checkpoints.

use super::{CompiledAdamWCheckpoint, CompiledModuleCheckpoint, CompiledModuleCheckpointPayload};
use crate::Result;
use std::borrow::Cow;
use std::collections::BTreeSet;

/// Portable state for one complete owned compiled-module AdamW lifecycle.
///
/// The embedded [`CompiledAdamWCheckpoint`] bytes are preserved exactly. The
/// shared deterministic envelope adds canonical topology, tied aliases, and
/// deduplicated immutable parameter/buffer values. A v2 envelope also
/// authenticates an attached evaluator's capture identity.
pub type CompiledModuleAdamWCheckpoint = CompiledModuleCheckpoint<CompiledAdamWCheckpoint>;

impl CompiledModuleCheckpointPayload for CompiledAdamWCheckpoint {
    const MODULE_FORMAT: &'static str = "rustgrad-compiled-module-adamw-v1";
    const MODULE_EVALUATION_FORMAT: Option<&'static str> =
        Some("rustgrad-compiled-module-adamw-v2");

    fn decode_module_payload(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes.to_vec())
    }

    fn encode_module_payload(&self) -> Result<Cow<'_, [u8]>> {
        Ok(Cow::Borrowed(self.as_bytes()))
    }

    fn module_parameter_names(&self) -> Result<BTreeSet<String>> {
        Ok(self.decoded().parameters.keys().cloned().collect())
    }

    fn record_module_checkpoint_decode() {
        #[cfg(test)]
        super::program_artifact::record_module_checkpoint_decode();
    }
}
