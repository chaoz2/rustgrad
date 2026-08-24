use super::{LlamaDecoder, LlamaDecoderConfig, LlamaDecoderError};
use crate::TensorData;

/// Deterministic single-sequence KV cache for the supported one-layer decoder.
/// State changes only after logits, keys, and values all execute successfully.
#[derive(Clone, Debug)]
pub struct LlamaKvCache {
    config: LlamaDecoderConfig,
    keys: Option<TensorData>,
    values: Option<TensorData>,
}

impl LlamaKvCache {
    /// Creates an empty cache tied to one exact decoder configuration.
    pub const fn new(config: LlamaDecoderConfig) -> Self {
        Self {
            config,
            keys: None,
            values: None,
        }
    }

    /// Returns the committed prefix length.
    pub fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |keys| keys.shape().dims()[1])
    }

    /// Returns true before the first successful forward or after [`Self::clear`].
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Drops all committed keys and values without changing configuration.
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
    }

    /// Executes one non-empty token chunk and commits the complete prefix cache
    /// only after every graph output succeeds.
    pub fn forward(
        &mut self,
        decoder: &LlamaDecoder,
        tokens: &[u32],
    ) -> Result<TensorData, LlamaDecoderError> {
        if decoder.config() != self.config {
            return Err(LlamaDecoderError::CacheConfigMismatch);
        }
        let plan = decoder.plan_with_past(tokens, self.keys.as_ref(), self.values.as_ref())?;
        let output = plan.execute()?;
        let logits = output.logits().clone();
        let keys = output.keys().clone();
        let values = output.values().clone();
        self.keys = Some(keys);
        self.values = Some(values);
        Ok(logits)
    }
}
