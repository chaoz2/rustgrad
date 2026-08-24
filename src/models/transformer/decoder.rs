use super::{LlamaDecoderSchema, LlamaDecoderState, OUTPUT_NORM, TOKEN_EMBEDDING};
use crate::{
    AttentionOptions, Backend, CpuBackend, DType, Error, Graph, NodeId, Scalar, Shape, TensorData,
};
use std::{collections::HashMap, error, fmt};

const TOKENS_INPUT: &str = "llama.tokens";
const PAST_KEYS_INPUT: &str = "llama.cache.keys";
const PAST_VALUES_INPUT: &str = "llama.cache.values";

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
        let mut x = embedding(
            &mut graph,
            token_node,
            embedding_weight,
            schema.embedding_dim,
        )?;

        let attn_norm_weight = graph.constant(self.state.state["blk.0.attn_norm.weight"].clone());
        let normalized = rms_norm(
            &mut graph,
            x,
            attn_norm_weight,
            schema.embedding_dim,
            self.config.norm_eps,
        )?;

        let query_weight = graph.constant(self.state.state["blk.0.attn_q.weight"].clone());
        let query_weight = permute_rope_weight(
            &mut graph,
            query_weight,
            schema.query_heads,
            schema.head_dim,
            schema.rope_dim,
            schema.embedding_dim,
        )?;
        let key_weight = graph.constant(self.state.state["blk.0.attn_k.weight"].clone());
        let key_weight = permute_rope_weight(
            &mut graph,
            key_weight,
            schema.kv_heads,
            schema.head_dim,
            schema.rope_dim,
            schema.embedding_dim,
        )?;
        let value_weight = graph.constant(self.state.state["blk.0.attn_v.weight"].clone());
        let query = linear(&mut graph, normalized, query_weight)?;
        let key = linear(&mut graph, normalized, key_weight)?;
        let value = linear(&mut graph, normalized, value_weight)?;
        let query = heads(
            &mut graph,
            query,
            tokens.len(),
            schema.query_heads,
            schema.head_dim,
        )?;
        let key = heads(
            &mut graph,
            key,
            tokens.len(),
            schema.kv_heads,
            schema.head_dim,
        )?;
        let value = heads(
            &mut graph,
            value,
            tokens.len(),
            schema.kv_heads,
            schema.head_dim,
        )?;
        let query = apply_rope(
            &mut graph,
            query,
            schema.query_heads,
            tokens.len(),
            schema.head_dim,
            schema.rope_dim,
            past_len,
            self.config.rope_theta,
        )?;
        let key = apply_rope(
            &mut graph,
            key,
            schema.kv_heads,
            tokens.len(),
            schema.head_dim,
            schema.rope_dim,
            past_len,
            self.config.rope_theta,
        )?;

        let (all_keys, all_values) = match (past_keys, past_values) {
            (Some(past_keys), Some(past_values)) => {
                let past_key_node = graph.input_dtype_requires_grad(
                    PAST_KEYS_INPUT,
                    past_keys.shape().clone(),
                    DType::F32,
                    false,
                );
                let past_value_node = graph.input_dtype_requires_grad(
                    PAST_VALUES_INPUT,
                    past_values.shape().clone(),
                    DType::F32,
                    false,
                );
                bindings.insert(PAST_KEYS_INPUT.to_owned(), past_keys.clone());
                bindings.insert(PAST_VALUES_INPUT.to_owned(), past_values.clone());
                (
                    graph.concat(vec![past_key_node, key], 1)?,
                    graph.concat(vec![past_value_node, value], 1)?,
                )
            }
            (None, None) => (key, value),
            _ => unreachable!("validate_past rejects half caches"),
        };

        let mask = graph.constant(TensorData::from_scalars(
            [tokens.len(), total_len],
            DType::Bool,
            (0..tokens.len()).flat_map(|row| {
                (0..total_len).map(move |column| Scalar::Bool(column <= past_len + row))
            }),
        )?);
        let attended = graph.scaled_dot_product_attention(
            query,
            all_keys,
            all_values,
            Some(mask),
            AttentionOptions {
                enable_gqa: true,
                ..AttentionOptions::default()
            },
        )?;
        let attended = graph.permute(attended, vec![1, 0, 2])?;
        let attended = graph.reshape(
            attended,
            [tokens.len(), schema.query_heads * schema.head_dim],
        )?;
        let output_weight = graph.constant(self.state.state["blk.0.attn_output.weight"].clone());
        let attended = linear(&mut graph, attended, output_weight)?;
        x = graph.add(x, attended)?;

        let ffn_norm_weight = graph.constant(self.state.state["blk.0.ffn_norm.weight"].clone());
        let normalized = rms_norm(
            &mut graph,
            x,
            ffn_norm_weight,
            schema.embedding_dim,
            self.config.norm_eps,
        )?;
        let gate_weight = graph.constant(self.state.state["blk.0.ffn_gate.weight"].clone());
        let up_weight = graph.constant(self.state.state["blk.0.ffn_up.weight"].clone());
        let down_weight = graph.constant(self.state.state["blk.0.ffn_down.weight"].clone());
        let gate = linear(&mut graph, normalized, gate_weight)?;
        let gate = graph.silu(gate)?;
        let up = linear(&mut graph, normalized, up_weight)?;
        let gated = graph.mul(gate, up)?;
        let down = linear(&mut graph, gated, down_weight)?;
        x = graph.add(x, down)?;

        let final_norm_weight = graph.constant(self.state.state[OUTPUT_NORM].clone());
        let normalized = rms_norm(
            &mut graph,
            x,
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
            keys: all_keys,
            values: all_values,
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

fn embedding(
    graph: &mut Graph,
    tokens: NodeId,
    weight: NodeId,
    embedding_dim: usize,
) -> Result<NodeId, Error> {
    let sequence = graph.shape(tokens)?.dims()[0];
    let indices = graph.reshape(tokens, [sequence, 1])?;
    let indices = graph.expand(indices, [sequence, embedding_dim])?;
    graph.gather(weight, indices, 0)
}

fn rms_norm(
    graph: &mut Graph,
    input: NodeId,
    weight: NodeId,
    dim: usize,
    eps: f32,
) -> Result<NodeId, Error> {
    if graph.shape(input)?.dims().last().copied() != Some(dim) {
        return Err(Error::InvalidReshape {
            from: graph.shape(input)?.clone(),
            to: Shape::new([dim]),
        });
    }
    let squared = graph.square(input)?;
    let mean = graph.reduce(squared, crate::ReduceKind::Mean, Some(vec![-1]), true)?;
    let epsilon = graph.constant(TensorData::scalar(eps));
    let scale = graph.add(mean, epsilon)?;
    let scale = graph.rsqrt(scale)?;
    let normalized = graph.mul(input, scale)?;
    graph.mul(normalized, weight)
}

fn linear(graph: &mut Graph, input: NodeId, weight: NodeId) -> Result<NodeId, Error> {
    let weight = graph.permute(weight, vec![1, 0])?;
    graph.matmul(input, weight)
}

fn heads(
    graph: &mut Graph,
    input: NodeId,
    sequence: usize,
    heads: usize,
    head_dim: usize,
) -> Result<NodeId, Error> {
    let input = graph.reshape(input, [sequence, heads, head_dim])?;
    graph.permute(input, vec![1, 0, 2])
}

fn permute_rope_weight(
    graph: &mut Graph,
    weight: NodeId,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    input_dim: usize,
) -> Result<NodeId, Error> {
    let weight = graph.reshape(weight, [heads, head_dim, input_dim])?;
    let prefix = head_dim - rope_dim;
    let rope = graph.shrink(weight, vec![(0, heads), (prefix, head_dim), (0, input_dim)])?;
    let rope = graph.reshape(rope, [heads, rope_dim / 2, 2, input_dim])?;
    let rope = graph.permute(rope, vec![0, 2, 1, 3])?;
    let rope = graph.reshape(rope, [heads, rope_dim, input_dim])?;
    let weight = if prefix == 0 {
        rope
    } else {
        let unrotated = graph.shrink(weight, vec![(0, heads), (0, prefix), (0, input_dim)])?;
        graph.concat(vec![unrotated, rope], 1)?
    };
    graph.reshape(weight, [heads * head_dim, input_dim])
}

#[allow(clippy::too_many_arguments)]
fn apply_rope(
    graph: &mut Graph,
    input: NodeId,
    heads: usize,
    sequence: usize,
    head_dim: usize,
    rope_dim: usize,
    start_pos: usize,
    theta: f64,
) -> Result<NodeId, Error> {
    let half = rope_dim / 2;
    let first = graph.shrink(input, vec![(0, heads), (0, sequence), (0, half)])?;
    let second = graph.shrink(input, vec![(0, heads), (0, sequence), (half, rope_dim)])?;
    let frequencies = (0..sequence).flat_map(|offset| {
        (0..half).map(move |index| {
            let frequency = 1.0 / theta.powf((2 * index) as f64 / rope_dim as f64);
            (start_pos + offset) as f64 * frequency
        })
    });
    let angles = frequencies.collect::<Vec<_>>();
    let cos = graph.constant(TensorData::from_scalars(
        [sequence, half],
        DType::F32,
        angles.iter().copied().map(|angle| Scalar::F(angle.cos())),
    )?);
    let sin = graph.constant(TensorData::from_scalars(
        [sequence, half],
        DType::F32,
        angles.into_iter().map(|angle| Scalar::F(angle.sin())),
    )?);
    let first_cos = graph.mul(first, cos)?;
    let second_sin = graph.mul(second, sin)?;
    let rotated_first = graph.sub(first_cos, second_sin)?;
    let second_cos = graph.mul(second, cos)?;
    let first_sin = graph.mul(first, sin)?;
    let rotated_second = graph.add(second_cos, first_sin)?;
    let rotated = graph.concat(vec![rotated_first, rotated_second], 2)?;
    if rope_dim == head_dim {
        Ok(rotated)
    } else {
        let tail = graph.shrink(input, vec![(0, heads), (0, sequence), (rope_dim, head_dim)])?;
        graph.concat(vec![rotated, tail], 2)
    }
}
