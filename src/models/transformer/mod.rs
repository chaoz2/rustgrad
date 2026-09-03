//! Explicit dense Llama GGUF binding, graph execution, caching, and generation.
//!
//! The fixed state schemas reject heuristic key discovery, while
//! [`LlamaDecoder`] and [`LlamaModel`] compose validated F32 state through
//! RustGrad's graph and CPU semantic oracle. Generation owns transactional
//! graph-produced caches rather than a host-side numerical substitute.

use crate::{
    DType, TensorData,
    gguf::{GgufError, GgufFile},
};
use std::{collections::BTreeMap, error, fmt};

mod batch;
mod batch_generation;
mod cache;
mod chat;
mod conversation;
mod decoder;
mod generation;
mod layer;
mod metal_step;
mod model;
mod native;
mod native_generation;
pub mod serving;
mod workflow;

pub use batch::{LlamaBatchCache, LlamaBatchPlan};
pub use batch_generation::{
    LlamaBatchGeneration, LlamaBatchGenerationError, LlamaBatchGenerator, LlamaBatchSampling,
    LlamaBatchSequence,
};
pub use cache::LlamaKvCache;
pub use chat::{
    LLAMA_SIMPLE_CHAT_TEMPLATE, LlamaChatError, LlamaChatMessage, LlamaChatRole, LlamaChatTemplate,
};
pub use conversation::{LlamaConversation, LlamaConversationError, LlamaConversationTurn};
pub use decoder::{
    LlamaDecoder, LlamaDecoderConfig, LlamaDecoderError, LlamaForwardOutput, LlamaForwardPlan,
};
pub use generation::{LlamaGeneration, LlamaGenerationError, LlamaGenerator, LlamaSampling};
pub use metal_step::{
    LlamaMetalStep, LlamaMetalStepError, LlamaMetalStepPlan, LlamaMetalStepSession,
};
pub use model::{
    LlamaLinearWeight, LlamaModel, LlamaModelCache, LlamaModelConfig, LlamaModelError,
    LlamaModelPlan, LlamaModelState, LlamaQkNorm, LlamaTokenIds,
};
pub use native::{
    LlamaBatchNativeCache, LlamaBatchNativeExecution, LlamaBatchNativePlan, LlamaNativeCache,
    LlamaNativeError, LlamaNativeExecution, LlamaNativeExecutor, LlamaNativePlan,
    LlamaNativeStageKind, LlamaNativeStageTrace,
};
pub use native_generation::{
    LlamaBatchNativeGeneration, LlamaBatchNativeGenerator, LlamaBatchNativeSequence,
    LlamaNativeGeneration, LlamaNativeGenerationError, LlamaNativeGenerationStepTrace,
    LlamaNativeGenerator,
};
pub use serving::{
    LlamaPrefixCacheStats, LlamaRequestId, LlamaRequestStatus, LlamaServingConfig,
    LlamaServingError, LlamaServingGenerationConfig, LlamaServingResult, LlamaServingSampling,
    LlamaServingScheduler, LlamaTokenEvent,
};
pub use workflow::{
    LlamaNativePromptOutput, LlamaNativePromptWorkflowError, LlamaPromptOutput,
    LlamaPromptWorkflow, LlamaPromptWorkflowError,
};

const TOKEN_EMBEDDING: &str = "token_embd.weight";
const OUTPUT_NORM: &str = "output_norm.weight";
const OUTPUT_WEIGHT: &str = "output.weight";
const ROPE_FREQS: &str = "rope_freqs.weight";

const BLOCK_TENSORS: [&str; 9] = [
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
];

/// Validated dimensions for the exact one-layer dense Llama state schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaDecoderSchema {
    vocab_size: usize,
    embedding_dim: usize,
    hidden_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
}

impl LlamaDecoderSchema {
    /// Validates non-zero dimensions, even rotary width, and checked projection
    /// sizes before any GGUF tensor is materialized.
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dim: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
    ) -> Result<Self, LlamaStateError> {
        for (field, value) in [
            ("vocab_size", vocab_size),
            ("embedding_dim", embedding_dim),
            ("hidden_dim", hidden_dim),
            ("query_heads", query_heads),
            ("kv_heads", kv_heads),
            ("head_dim", head_dim),
            ("rope_dim", rope_dim),
        ] {
            if value == 0 {
                return Err(LlamaStateError::InvalidConfig { field });
            }
        }
        if !rope_dim.is_multiple_of(2) || rope_dim > head_dim {
            return Err(LlamaStateError::InvalidConfig { field: "rope_dim" });
        }
        if !query_heads.is_multiple_of(kv_heads) {
            return Err(LlamaStateError::InvalidConfig {
                field: "query_heads",
            });
        }
        query_heads
            .checked_mul(head_dim)
            .ok_or(LlamaStateError::ProjectionOverflow)?;
        kv_heads
            .checked_mul(head_dim)
            .ok_or(LlamaStateError::ProjectionOverflow)?;
        // Every gated feed-forward projection spans the residual and
        // intermediate widths. Validate that extent before GGUF tensors can
        // be materialized for binding.
        embedding_dim
            .checked_mul(hidden_dim)
            .ok_or(LlamaStateError::ProjectionOverflow)?;
        Ok(Self {
            vocab_size,
            embedding_dim,
            hidden_dim,
            query_heads,
            kv_heads,
            head_dim,
            rope_dim,
        })
    }

    /// Returns the vocabulary row count.
    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    /// Returns the residual-stream and embedding width.
    pub const fn embedding_dim(self) -> usize {
        self.embedding_dim
    }

    /// Returns the dense gated feed-forward width.
    pub const fn hidden_dim(self) -> usize {
        self.hidden_dim
    }

    /// Returns the query head count.
    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    /// Returns the key/value head count.
    pub const fn kv_heads(self) -> usize {
        self.kv_heads
    }

    /// Returns each attention head width.
    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    /// Returns the rotary subspace width.
    pub const fn rope_dim(self) -> usize {
        self.rope_dim
    }

    /// Atomically materializes all GGUF tensors to F32, then validates the
    /// complete explicit state inventory and shapes.
    pub fn bind(self, file: &GgufFile<'_>) -> Result<LlamaDecoderState, LlamaStateError> {
        let state = file.materialize_state_f32()?;
        self.bind_materialized(state)
    }

    fn bind_materialized(
        self,
        state: BTreeMap<String, TensorData>,
    ) -> Result<LlamaDecoderState, LlamaStateError> {
        for name in state.keys() {
            if name != TOKEN_EMBEDDING
                && name != OUTPUT_NORM
                && name != OUTPUT_WEIGHT
                && name != ROPE_FREQS
                && !BLOCK_TENSORS.contains(&name.as_str())
            {
                return Err(LlamaStateError::UnexpectedTensor(name.clone()));
            }
        }

        let query_width = self
            .query_heads
            .checked_mul(self.head_dim)
            .ok_or(LlamaStateError::ProjectionOverflow)?;
        let kv_width = self
            .kv_heads
            .checked_mul(self.head_dim)
            .ok_or(LlamaStateError::ProjectionOverflow)?;
        let expected = [
            (TOKEN_EMBEDDING, vec![self.vocab_size, self.embedding_dim]),
            (OUTPUT_NORM, vec![self.embedding_dim]),
            ("blk.0.attn_norm.weight", vec![self.embedding_dim]),
            ("blk.0.attn_q.weight", vec![query_width, self.embedding_dim]),
            ("blk.0.attn_k.weight", vec![kv_width, self.embedding_dim]),
            ("blk.0.attn_v.weight", vec![kv_width, self.embedding_dim]),
            (
                "blk.0.attn_output.weight",
                vec![self.embedding_dim, query_width],
            ),
            ("blk.0.ffn_norm.weight", vec![self.embedding_dim]),
            (
                "blk.0.ffn_gate.weight",
                vec![self.hidden_dim, self.embedding_dim],
            ),
            (
                "blk.0.ffn_up.weight",
                vec![self.hidden_dim, self.embedding_dim],
            ),
            (
                "blk.0.ffn_down.weight",
                vec![self.embedding_dim, self.hidden_dim],
            ),
        ];
        for (name, shape) in expected {
            validate_tensor(&state, name, &shape)?;
        }
        if state.contains_key(OUTPUT_WEIGHT) {
            validate_tensor(
                &state,
                OUTPUT_WEIGHT,
                &[self.vocab_size, self.embedding_dim],
            )?;
        }
        if state.contains_key(ROPE_FREQS) {
            validate_tensor(&state, ROPE_FREQS, &[self.rope_dim / 2])?;
        }

        let output = if state.contains_key(OUTPUT_WEIGHT) {
            LlamaOutputBinding::Explicit
        } else {
            LlamaOutputBinding::TiedToTokenEmbedding
        };
        Ok(LlamaDecoderState {
            schema: self,
            state,
            output,
        })
    }
}

/// How the output projection was resolved from the GGUF state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaOutputBinding {
    Explicit,
    TiedToTokenEmbedding,
}

/// Fully validated one-layer Llama decoder state.
#[derive(Clone, Debug)]
pub struct LlamaDecoderState {
    schema: LlamaDecoderSchema,
    state: BTreeMap<String, TensorData>,
    output: LlamaOutputBinding,
}

impl LlamaDecoderState {
    /// Returns the dimensions proven when this state was bound.
    pub const fn schema(&self) -> LlamaDecoderSchema {
        self.schema
    }

    pub const fn output_binding(&self) -> LlamaOutputBinding {
        self.output
    }

    /// Returns validated stored tensors. A tied output remains represented by
    /// [`Self::output_weight`] rather than a duplicate map entry.
    pub fn tensors(&self) -> &BTreeMap<String, TensorData> {
        &self.state
    }

    /// Resolves either the explicit output projection or tinygrad's fallback
    /// tie to `token_embd.weight`.
    pub fn output_weight(&self) -> &TensorData {
        match self.output {
            LlamaOutputBinding::Explicit => &self.state[OUTPUT_WEIGHT],
            LlamaOutputBinding::TiedToTokenEmbedding => &self.state[TOKEN_EMBEDDING],
        }
    }
}

/// Structured schema/materialization rejection for one-layer Llama state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaStateError {
    InvalidConfig {
        field: &'static str,
    },
    ProjectionOverflow,
    Gguf(GgufError),
    MissingTensor(&'static str),
    UnexpectedTensor(String),
    DTypeMismatch {
        tensor: &'static str,
        actual: DType,
    },
    ShapeMismatch {
        tensor: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

impl fmt::Display for LlamaStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama state error: {self:?}")
    }
}

impl error::Error for LlamaStateError {}

impl From<GgufError> for LlamaStateError {
    fn from(value: GgufError) -> Self {
        Self::Gguf(value)
    }
}

fn validate_tensor(
    state: &BTreeMap<String, TensorData>,
    name: &'static str,
    expected: &[usize],
) -> Result<(), LlamaStateError> {
    let tensor = state
        .get(name)
        .ok_or(LlamaStateError::MissingTensor(name))?;
    if tensor.dtype() != DType::F32 {
        return Err(LlamaStateError::DTypeMismatch {
            tensor: name,
            actual: tensor.dtype(),
        });
    }
    if tensor.shape().dims() != expected {
        return Err(LlamaStateError::ShapeMismatch {
            tensor: name,
            expected: expected.to_vec(),
            actual: tensor.shape().dims().to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod decoder_tests;
#[cfg(test)]
pub(crate) mod model_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod batch_tests;

#[cfg(test)]
mod native_generation_tests;
#[cfg(test)]
mod native_tests;
#[cfg(test)]
mod packed_tests;
#[cfg(test)]
pub(crate) use packed_tests::models as packed_metal_fixture_models;
#[cfg(test)]
mod workflow_tests;
