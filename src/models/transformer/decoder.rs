use super::{
    LlamaDecoderSchema, LlamaDecoderState, OUTPUT_NORM, TOKEN_EMBEDDING,
    layer::{append_dense_layer, embedding, linear, rms_norm},
};
use crate::{Backend, CpuBackend, DType, Error, Graph, NodeId, Scalar, Shape, TensorData};
use std::{collections::HashMap, error, fmt};

const TOKENS_INPUT: &str = "llama.tokens";

/// Runtime configuration for the supported one-layer, bias-free dense Llama
/// decoder graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaDecoderConfig {
    schema: LlamaDecoderSchema,
    max_context: usize,
    norm_eps: f32,
    rope_theta: f64,
}

impl LlamaDecoderConfig {
    /// Validates the state schema plus the runtime context, RMSNorm epsilon,
    /// and rotary frequency base.
    pub fn new(
        schema: LlamaDecoderSchema,
        max_context: usize,
        norm_eps: f32,
        rope_theta: f64,
    ) -> Result<Self, LlamaDecoderError> {
        if max_context == 0 {
            return Err(LlamaDecoderError::InvalidConfig {
                reason: "max_context must be nonzero",
            });
        }
        if !norm_eps.is_finite() || norm_eps < 0.0 {
            return Err(LlamaDecoderError::InvalidConfig {
                reason: "norm_eps must be finite and non-negative",
            });
        }
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(LlamaDecoderError::InvalidConfig {
                reason: "rope_theta must be finite and positive",
            });
        }
        Ok(Self {
            schema,
            max_context,
            norm_eps,
            rope_theta,
        })
    }

    /// Returns the validated state schema.
    pub const fn schema(self) -> LlamaDecoderSchema {
        self.schema
    }

    /// Returns the maximum total cached or full-sequence length.
    pub const fn max_context(self) -> usize {
        self.max_context
    }
}

/// Executable one-layer Llama decoder backed by validated F32 state.
#[derive(Clone, Debug)]
pub struct LlamaDecoder {
    config: LlamaDecoderConfig,
    state: LlamaDecoderState,
}

impl LlamaDecoder {
    /// Couples runtime configuration to state only when both carry the exact
    /// same validated schema.
    pub fn new(
        config: LlamaDecoderConfig,
        state: LlamaDecoderState,
    ) -> Result<Self, LlamaDecoderError> {
        if config.schema != state.schema() {
            return Err(LlamaDecoderError::StateSchemaMismatch);
        }
        Ok(Self { config, state })
    }

    /// Returns the immutable runtime configuration.
    pub const fn config(&self) -> LlamaDecoderConfig {
        self.config
    }

    /// Builds an inspectable full-sequence graph with no prior cache.
    pub fn plan(&self, tokens: &[u32]) -> Result<LlamaForwardPlan, LlamaDecoderError> {
        self.plan_with_past(tokens, None, None)
    }

    /// Executes a full causal sequence through [`CpuBackend`].
    pub fn forward(&self, tokens: &[u32]) -> Result<LlamaForwardOutput, LlamaDecoderError> {
        self.plan(tokens)?.execute()
    }

    pub(super) fn plan_with_past(
        &self,
        tokens: &[u32],
        past_keys: Option<&TensorData>,
        past_values: Option<&TensorData>,
    ) -> Result<LlamaForwardPlan, LlamaDecoderError> {
        if tokens.is_empty() {
            return Err(LlamaDecoderError::EmptyTokens);
        }
        let schema = self.config.schema;
        for &token in tokens {
            if usize::try_from(token).map_or(true, |token| token >= schema.vocab_size) {
                return Err(LlamaDecoderError::TokenOutOfRange {
                    token,
                    vocab_size: schema.vocab_size,
                });
            }
        }
        let past_len = validate_past(schema, past_keys, past_values)?;
        let total_len = past_len
            .checked_add(tokens.len())
            .ok_or(LlamaDecoderError::ContextOverflow)?;
        if total_len > self.config.max_context {
            return Err(LlamaDecoderError::ContextLength {
                requested: total_len,
                maximum: self.config.max_context,
            });
        }

        let mut graph = Graph::new();
        let token_node =
            graph.input_dtype_requires_grad(TOKENS_INPUT, [tokens.len()], DType::I64, false);
        let mut bindings = HashMap::from([(
            TOKENS_INPUT.to_owned(),
            TensorData::from_scalars(
                [tokens.len()],
                DType::I64,
                tokens.iter().map(|token| Scalar::I(i64::from(*token))),
            )?,
        )]);

        let embedding_weight = graph.constant(self.state.state[TOKEN_EMBEDDING].clone());
        let x = embedding(
            &mut graph,
            token_node,
            embedding_weight,
            schema.embedding_dim,
        )?;
        let mut quantized_linears = super::model::QuantizedLinearBindings::new();
        let layer = append_dense_layer(
            &mut graph,
            &mut bindings,
            x,
            super::layer::LayerState::Dense(&self.state.state),
            &mut quantized_linears,
            "blk.0",
            "llama.cache",
            schema,
            tokens.len(),
            past_len,
            total_len,
            self.config.norm_eps,
            self.config.rope_theta,
            super::LlamaQkNorm::None,
            false,
            past_keys,
            past_values,
        )?;

        let final_norm_weight = graph.constant(self.state.state[OUTPUT_NORM].clone());
        let normalized = rms_norm(
            &mut graph,
            layer.output,
            final_norm_weight,
            schema.embedding_dim,
            self.config.norm_eps,
        )?;
        let output_weight = graph.constant(self.state.output_weight().clone());
        let logits = linear(&mut graph, normalized, output_weight)?;
        Ok(LlamaForwardPlan {
            graph,
            bindings,
            logits,
            keys: layer.keys,
            values: layer.values,
        })
    }
}

/// An inspectable fixed-shape graph and its immutable runtime bindings.
#[derive(Debug)]
pub struct LlamaForwardPlan {
    graph: Graph,
    bindings: HashMap<String, TensorData>,
    logits: NodeId,
    keys: NodeId,
    values: NodeId,
}

impl LlamaForwardPlan {
    /// Returns the typed graph before execution.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the all-position logits output node.
    pub const fn logits_node(&self) -> NodeId {
        self.logits
    }

    /// Executes logits and the committed cache values through the CPU semantic
    /// oracle. No cache owner is mutated by this operation.
    pub fn execute(&self) -> Result<LlamaForwardOutput, LlamaDecoderError> {
        let backend = CpuBackend;
        let logits = backend.execute(&self.graph, self.logits, &self.bindings)?;
        let keys = backend.execute(&self.graph, self.keys, &self.bindings)?;
        let values = backend.execute(&self.graph, self.values, &self.bindings)?;
        Ok(LlamaForwardOutput {
            logits,
            keys,
            values,
        })
    }
}

/// Materialized decoder results. Keys and values contain the complete prefix,
/// including any prior cache supplied to the plan.
#[derive(Clone, Debug)]
pub struct LlamaForwardOutput {
    logits: TensorData,
    keys: TensorData,
    values: TensorData,
}

impl LlamaForwardOutput {
    /// Returns logits shaped `[sequence, vocabulary]`.
    pub fn logits(&self) -> &TensorData {
        &self.logits
    }

    /// Returns complete-prefix keys shaped `[kv_heads, sequence, head_dim]`.
    pub fn keys(&self) -> &TensorData {
        &self.keys
    }

    /// Returns complete-prefix values shaped `[kv_heads, sequence, head_dim]`.
    pub fn values(&self) -> &TensorData {
        &self.values
    }
}

/// Structured decoder configuration, graph, input, execution, or cache error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaDecoderError {
    InvalidConfig {
        reason: &'static str,
    },
    StateSchemaMismatch,
    EmptyTokens,
    TokenOutOfRange {
        token: u32,
        vocab_size: usize,
    },
    ContextOverflow,
    ContextLength {
        requested: usize,
        maximum: usize,
    },
    CachePairMismatch,
    CacheConfigMismatch,
    CacheShape {
        expected_heads: usize,
        expected_head_dim: usize,
        keys: Shape,
        values: Shape,
    },
    CacheDType {
        keys: DType,
        values: DType,
    },
    Graph(Error),
}

impl fmt::Display for LlamaDecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama decoder error: {self:?}")
    }
}

impl error::Error for LlamaDecoderError {}

impl From<Error> for LlamaDecoderError {
    fn from(value: Error) -> Self {
        Self::Graph(value)
    }
}

fn validate_past(
    schema: LlamaDecoderSchema,
    keys: Option<&TensorData>,
    values: Option<&TensorData>,
) -> Result<usize, LlamaDecoderError> {
    let (keys, values) = match (keys, values) {
        (None, None) => return Ok(0),
        (Some(keys), Some(values)) => (keys, values),
        _ => return Err(LlamaDecoderError::CachePairMismatch),
    };
    if keys.dtype() != DType::F32 || values.dtype() != DType::F32 {
        return Err(LlamaDecoderError::CacheDType {
            keys: keys.dtype(),
            values: values.dtype(),
        });
    }
    let valid = |shape: &Shape| {
        shape.rank() == 3
            && shape.dims()[0] == schema.kv_heads
            && shape.dims()[2] == schema.head_dim
    };
    if !valid(keys.shape()) || !valid(values.shape()) || keys.shape() != values.shape() {
        return Err(LlamaDecoderError::CacheShape {
            expected_heads: schema.kv_heads,
            expected_head_dim: schema.head_dim,
            keys: keys.shape().clone(),
            values: values.shape().clone(),
        });
    }
    Ok(keys.shape().dims()[1])
}
