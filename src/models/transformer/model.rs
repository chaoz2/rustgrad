use super::{
    LlamaDecoderSchema, LlamaOutputBinding, OUTPUT_NORM, OUTPUT_WEIGHT, ROPE_FREQS,
    TOKEN_EMBEDDING,
    layer::{append_dense_layer, embedding, linear, rms_norm},
};
use crate::{
    Backend, CpuBackend, DType, Error, Graph, NodeId, QuantizedMatmulPlan, Scalar, Shape,
    TensorData,
    gguf::{
        GgmlLayout, GgmlType, GgufError, GgufFile, GgufMetadataAccessError, QuantizedRowGatherPlan,
        QuantizedTensorData,
    },
    tokenizer::{SimpleTokenizer, TokenizerError},
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error, fmt,
};

const ARCHITECTURE_KEY: &str = "general.architecture";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";

/// Special token IDs carried alongside the validated GGUF model configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaTokenIds {
    bos: Option<u32>,
    eos: u32,
    eot: Option<u32>,
}

impl LlamaTokenIds {
    /// Returns the optional beginning-of-sequence ID.
    pub const fn bos(self) -> Option<u32> {
        self.bos
    }
    /// Returns the required end-of-sequence ID.
    pub const fn eos(self) -> u32 {
        self.eos
    }
    /// Returns the optional end-of-turn ID.
    pub const fn eot(self) -> Option<u32> {
        self.eot
    }
    /// Returns true for the configured EOS or EOT ID.
    pub const fn is_stop(self, token: u32) -> bool {
        token == self.eos || matches!(self.eot, Some(eot) if token == eot)
    }
}

/// Source-evidenced q/k RMSNorm placement for dense Llama attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaQkNorm {
    None,
    /// One shared-width norm per head, applied after head reshape.
    PerHead,
    /// One full-projection norm, applied before head reshape.
    PerProjection,
}

/// Exact storage retained for one rank-two Llama projection.
#[derive(Clone, Debug)]
pub enum LlamaLinearWeight {
    /// Dense GGUF storage converted once to the model's F32 graph dtype.
    Dense(TensorData),
    /// Audited GGML blocks retained byte-for-byte for native quantized matmul.
    Quantized(QuantizedTensorData),
}

impl LlamaLinearWeight {
    /// Returns the validated logical `[out_features, in_features]` shape.
    pub fn shape(&self) -> &Shape {
        match self {
            Self::Dense(value) => value.shape(),
            Self::Quantized(value) => &value.descriptor().logical_shape,
        }
    }

    /// Returns the packed GGML format, or `None` for a dense projection.
    pub fn quantized_type(&self) -> Option<GgmlType> {
        match self {
            Self::Dense(_) => None,
            Self::Quantized(value) => Some(value.descriptor().ggml_type),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct QuantizedLinearBinding {
    pub(super) tensor: String,
    pub(super) input_name: String,
    pub(super) weight_node: NodeId,
    pub(super) weight: QuantizedTensorData,
}

pub(super) type QuantizedLinearBindings = BTreeMap<usize, QuantizedLinearBinding>;

/// Typed configuration for the supported dense-or-packed GGUF Llama model.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaModelConfig {
    architecture: String,
    layer_count: usize,
    schema: LlamaDecoderSchema,
    max_context: usize,
    norm_eps: f32,
    rope_theta: f64,
    qk_norm: LlamaQkNorm,
    qkv_bias: bool,
    token_ids: LlamaTokenIds,
}

impl LlamaModelConfig {
    /// Derives the exact supported Llama configuration from typed GGUF
    /// metadata and the exact tensor names inspected by the checked-in source.
    /// Unsupported scaling, bias families, experts, MLA, or SSM remain explicit.
    pub fn from_gguf(file: &GgufFile<'_>) -> Result<Self, LlamaModelError> {
        let architecture = required_string(file, ARCHITECTURE_KEY)?;
        if architecture != "llama" {
            return Err(LlamaModelError::UnsupportedArchitecture(
                architecture.to_owned(),
            ));
        }
        let tokenizer = SimpleTokenizer::from_gguf(file)?;
        let vocab_size = file
            .metadata_strings(TOKENS_KEY)?
            .ok_or_else(|| LlamaModelError::MissingMetadata(TOKENS_KEY.to_owned()))?
            .len();
        let prefix = architecture;
        let layer_count = required_usize(file, &format!("{prefix}.block_count"))?;
        let embedding_dim = required_usize(file, &format!("{prefix}.embedding_length"))?;
        let hidden_dim = required_usize(file, &format!("{prefix}.feed_forward_length"))?;
        let query_heads = required_usize(file, &format!("{prefix}.attention.head_count"))?;
        let kv_heads = required_usize(file, &format!("{prefix}.attention.head_count_kv"))?;
        let max_context = required_usize(file, &format!("{prefix}.context_length"))?;
        let default_head_dim =
            embedding_dim
                .checked_div(query_heads)
                .ok_or(LlamaModelError::InvalidConfig {
                    field: "query_heads",
                })?;
        if default_head_dim.checked_mul(query_heads) != Some(embedding_dim) {
            return Err(LlamaModelError::InvalidConfig {
                field: "embedding_length",
            });
        }
        let head_dim = optional_usize(file, &format!("{prefix}.attention.key_length"))?
            .unwrap_or(default_head_dim);
        let value_dim =
            optional_usize(file, &format!("{prefix}.attention.value_length"))?.unwrap_or(head_dim);
        if value_dim != head_dim {
            return Err(LlamaModelError::UnsupportedVariant(
                "value head width differs from key head width",
            ));
        }
        let rope_dim =
            optional_usize(file, &format!("{prefix}.rope.dimension_count"))?.unwrap_or(head_dim);
        if rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
            return Err(LlamaModelError::UnsupportedVariant(
                "invalid partial rotary dimensions",
            ));
        }
        if rope_dim != head_dim && !head_dim.is_multiple_of(2) {
            return Err(LlamaModelError::UnsupportedVariant(
                "partial rotary dimensions with odd key head width",
            ));
        }
        let scaling_prefix = format!("{prefix}.rope.scaling.");
        let frequency_scale = format!("{prefix}.rope.freq_scale");
        if file
            .metadata()
            .iter()
            .any(|entry| entry.key().starts_with(&scaling_prefix) || entry.key() == frequency_scale)
        {
            return Err(LlamaModelError::UnsupportedVariant(
                "RoPE frequency scaling metadata",
            ));
        }
        reject_nonzero(
            file,
            &format!("{prefix}.nextn_predict_layers"),
            "next-token prediction layers",
        )?;
        reject_nonzero(
            file,
            &format!("{prefix}.expert_count"),
            "mixture-of-experts",
        )?;
        reject_nonzero(file, &format!("{prefix}.attention.kv_lora_rank"), "MLA")?;
        reject_nonzero(
            file,
            &format!("{prefix}.attention.q_lora_rank"),
            "query LoRA",
        )?;

        let norm_eps = required_float(file, &format!("{prefix}.attention.layer_norm_rms_epsilon"))?;
        if !norm_eps.is_finite() || norm_eps < 0.0 || norm_eps > f64::from(f32::MAX) {
            return Err(LlamaModelError::InvalidConfig {
                field: "layer_norm_rms_epsilon",
            });
        }
        let norm_eps = norm_eps as f32;
        let rope_theta = required_float(file, &format!("{prefix}.rope.freq_base"))?;
        let schema = LlamaDecoderSchema::new(
            vocab_size,
            embedding_dim,
            hidden_dim,
            query_heads,
            kv_heads,
            head_dim,
            rope_dim,
        )?;
        if layer_count == 0 {
            return Err(LlamaModelError::InvalidConfig {
                field: "block_count",
            });
        }
        if max_context == 0 {
            return Err(LlamaModelError::InvalidConfig {
                field: "context_length",
            });
        }
        if !norm_eps.is_finite() || norm_eps < 0.0 {
            return Err(LlamaModelError::InvalidConfig {
                field: "layer_norm_rms_epsilon",
            });
        }
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(LlamaModelError::InvalidConfig {
                field: "rope.freq_base",
            });
        }
        let query_width =
            query_heads
                .checked_mul(head_dim)
                .ok_or(LlamaModelError::InvalidConfig {
                    field: "query projection",
                })?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .ok_or(LlamaModelError::InvalidConfig {
                field: "key/value projection",
            })?;
        let q_norm = file.tensor("blk.0.attn_q_norm.weight");
        let k_norm = file.tensor("blk.0.attn_k_norm.weight");
        let qk_norm = match (q_norm, k_norm) {
            (None, None) => LlamaQkNorm::None,
            (Some(query), Some(key))
                if query.shape().dims() == [head_dim] && key.shape().dims() == [head_dim] =>
            {
                LlamaQkNorm::PerHead
            }
            (Some(query), Some(key))
                if query_width == kv_width
                    && query.shape().dims() == [query_width]
                    && key.shape().dims() == [kv_width] =>
            {
                LlamaQkNorm::PerProjection
            }
            _ => {
                return Err(LlamaModelError::UnsupportedVariant(
                    "partial or incompatible q/k normalization",
                ));
            }
        };
        let q_bias = file.tensor("blk.0.attn_q.bias").is_some();
        let k_bias = file.tensor("blk.0.attn_k.bias").is_some();
        let v_bias = file.tensor("blk.0.attn_v.bias").is_some();
        if q_bias != k_bias || q_bias != v_bias {
            return Err(LlamaModelError::UnsupportedVariant(
                "partial q/k/v projection bias family",
            ));
        }
        Ok(Self {
            architecture: architecture.to_owned(),
            layer_count,
            schema,
            max_context,
            norm_eps,
            rope_theta,
            qk_norm,
            qkv_bias: q_bias,
            token_ids: LlamaTokenIds {
                bos: tokenizer.bos_id(),
                eos: tokenizer.eos_id(),
                eot: tokenizer.eot_id(),
            },
        })
    }

    /// Returns the validated GGUF architecture name (`llama`).
    pub fn architecture(&self) -> &str {
        &self.architecture
    }
    /// Returns the fixed transformer block count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }
    /// Returns the shared tensor dimensions for every block.
    pub const fn schema(&self) -> LlamaDecoderSchema {
        self.schema
    }
    /// Returns the maximum prompt plus cache length.
    pub const fn max_context(&self) -> usize {
        self.max_context
    }
    /// Returns the RMSNorm epsilon.
    pub const fn norm_eps(&self) -> f32 {
        self.norm_eps
    }
    /// Returns the RoPE frequency base.
    pub const fn rope_theta(&self) -> f64 {
        self.rope_theta
    }
    /// Returns the source-exact q/k normalization placement.
    pub const fn qk_norm(&self) -> LlamaQkNorm {
        self.qk_norm
    }
    /// Returns whether every layer has q/k/v projection biases.
    pub const fn qkv_bias(&self) -> bool {
        self.qkv_bias
    }
    /// Returns BOS/EOS/EOT IDs validated by the tokenizer metadata path.
    pub const fn token_ids(&self) -> LlamaTokenIds {
        self.token_ids
    }
}

/// Atomically bound dense auxiliaries and dense-or-packed rank-two projections.
#[derive(Clone, Debug)]
pub struct LlamaModelState {
    config: LlamaModelConfig,
    embedding: LlamaLinearWeight,
    dense: BTreeMap<String, TensorData>,
    linears: BTreeMap<String, LlamaLinearWeight>,
    output: LlamaOutputBinding,
}

impl LlamaModelState {
    /// Atomically validates and binds every root and `blk.N` tensor. Supported
    /// rank-two projections and the embedding table retain packed GGML bytes;
    /// norms, biases, and optional RoPE auxiliaries become dense F32.
    pub fn bind(config: &LlamaModelConfig, file: &GgufFile<'_>) -> Result<Self, LlamaModelError> {
        let schema = config.schema;
        let query_width = schema.query_heads.checked_mul(schema.head_dim).ok_or(
            LlamaModelError::InvalidConfig {
                field: "query projection",
            },
        )?;
        let kv_width =
            schema
                .kv_heads
                .checked_mul(schema.head_dim)
                .ok_or(LlamaModelError::InvalidConfig {
                    field: "key/value projection",
                })?;
        let expected_capacity = config
            .layer_count
            .checked_mul(
                9 + usize::from(config.qk_norm != LlamaQkNorm::None) * 2
                    + usize::from(config.qkv_bias) * 3,
            )
            .and_then(|count| count.checked_add(3))
            .ok_or_else(|| {
                LlamaModelError::MetadataValueOutOfRange("llama.block_count".to_owned())
            })?;
        let mut expected =
            Vec::with_capacity(expected_capacity.min(file.tensors().len().saturating_add(2)));
        expected.push((
            TOKEN_EMBEDDING.to_owned(),
            vec![schema.vocab_size, schema.embedding_dim],
        ));
        expected.push((OUTPUT_NORM.to_owned(), vec![schema.embedding_dim]));
        for layer in 0..config.layer_count {
            let prefix = format!("blk.{layer}");
            expected.extend([
                (
                    format!("{prefix}.attn_norm.weight"),
                    vec![schema.embedding_dim],
                ),
                (
                    format!("{prefix}.attn_q.weight"),
                    vec![query_width, schema.embedding_dim],
                ),
                (
                    format!("{prefix}.attn_k.weight"),
                    vec![kv_width, schema.embedding_dim],
                ),
                (
                    format!("{prefix}.attn_v.weight"),
                    vec![kv_width, schema.embedding_dim],
                ),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![schema.embedding_dim, query_width],
                ),
                (
                    format!("{prefix}.ffn_norm.weight"),
                    vec![schema.embedding_dim],
                ),
                (
                    format!("{prefix}.ffn_gate.weight"),
                    vec![schema.hidden_dim, schema.embedding_dim],
                ),
                (
                    format!("{prefix}.ffn_up.weight"),
                    vec![schema.hidden_dim, schema.embedding_dim],
                ),
                (
                    format!("{prefix}.ffn_down.weight"),
                    vec![schema.embedding_dim, schema.hidden_dim],
                ),
            ]);
            match config.qk_norm {
                LlamaQkNorm::None => {}
                LlamaQkNorm::PerHead => expected.extend([
                    (
                        format!("{prefix}.attn_q_norm.weight"),
                        vec![schema.head_dim],
                    ),
                    (
                        format!("{prefix}.attn_k_norm.weight"),
                        vec![schema.head_dim],
                    ),
                ]),
                LlamaQkNorm::PerProjection => expected.extend([
                    (format!("{prefix}.attn_q_norm.weight"), vec![query_width]),
                    (format!("{prefix}.attn_k_norm.weight"), vec![kv_width]),
                ]),
            }
            if config.qkv_bias {
                expected.extend([
                    (format!("{prefix}.attn_q.bias"), vec![query_width]),
                    (format!("{prefix}.attn_k.bias"), vec![kv_width]),
                    (format!("{prefix}.attn_v.bias"), vec![kv_width]),
                ]);
            }
        }
        let allowed = expected
            .iter()
            .map(|(name, _)| name.as_str())
            .chain([OUTPUT_WEIGHT, ROPE_FREQS])
            .collect::<HashSet<_>>();
        if let Some(tensor) = file
            .tensors()
            .iter()
            .find(|tensor| !allowed.contains(tensor.name()))
        {
            return Err(LlamaModelError::UnexpectedTensor(tensor.name().to_owned()));
        }
        for (name, shape) in &expected {
            validate_gguf_tensor(file, name, shape)?;
        }
        if file.tensor(OUTPUT_WEIGHT).is_some() {
            validate_gguf_tensor(
                file,
                OUTPUT_WEIGHT,
                &[schema.vocab_size, schema.embedding_dim],
            )?;
        }
        if file.tensor(ROPE_FREQS).is_some() {
            validate_gguf_tensor(file, ROPE_FREQS, &[schema.rope_dim / 2])?;
        }

        let linear_names = expected
            .iter()
            .map(|(name, _)| name.as_str())
            .filter(|name| *name == TOKEN_EMBEDDING || is_projection_weight(name))
            .chain(file.tensor(OUTPUT_WEIGHT).map(|_| OUTPUT_WEIGHT))
            .collect::<HashSet<_>>();
        let mut dense = BTreeMap::new();
        let mut linears = BTreeMap::new();
        for tensor in file.tensors() {
            let name = tensor.name();
            if linear_names.contains(name) {
                let weight = match tensor.layout() {
                    GgmlLayout::Dense { .. } => {
                        LlamaLinearWeight::Dense(file.materialize_f32(name)?)
                    }
                    GgmlLayout::Quantized { .. } => {
                        LlamaLinearWeight::Quantized(file.quantized_tensor(name)?)
                    }
                };
                linears.insert(name.to_owned(), weight);
            } else {
                if matches!(tensor.layout(), GgmlLayout::Quantized { .. }) {
                    return Err(LlamaModelError::UnsupportedPackedTensor(name.to_owned()));
                }
                dense.insert(name.to_owned(), file.materialize_f32(name)?);
            }
        }
        let embedding = linears
            .remove(TOKEN_EMBEDDING)
            .ok_or_else(|| LlamaModelError::MissingTensor(TOKEN_EMBEDDING.to_owned()))?;
        let output = if file.tensor(OUTPUT_WEIGHT).is_some() {
            LlamaOutputBinding::Explicit
        } else {
            LlamaOutputBinding::TiedToTokenEmbedding
        };
        Ok(Self {
            config: config.clone(),
            embedding,
            dense,
            linears,
            output,
        })
    }

    /// Returns the exact configuration used to validate this state.
    pub fn config(&self) -> &LlamaModelConfig {
        &self.config
    }

    /// Returns whether the output projection is explicit or tied.
    pub const fn output_binding(&self) -> LlamaOutputBinding {
        self.output
    }

    /// Returns the validated dense F32 auxiliary inventory.
    pub fn tensors(&self) -> &BTreeMap<String, TensorData> {
        &self.dense
    }

    /// Returns the token embedding storage without materializing packed rows.
    pub const fn embedding_weight(&self) -> &LlamaLinearWeight {
        &self.embedding
    }

    /// Returns the validated rank-two projection inventory without changing
    /// packed storage.
    pub fn linear_weights(&self) -> &BTreeMap<String, LlamaLinearWeight> {
        &self.linears
    }
}

pub(super) fn append_model_linear(
    graph: &mut Graph,
    input: NodeId,
    state: &LlamaModelState,
    name: &str,
    quantized: &mut QuantizedLinearBindings,
) -> Result<NodeId, Error> {
    let weight = if name == TOKEN_EMBEDDING {
        &state.embedding
    } else {
        &state.linears[name]
    };
    match weight {
        LlamaLinearWeight::Dense(value) => {
            let weight = graph.constant(value.clone());
            Ok(linear(graph, input, weight)?)
        }
        LlamaLinearWeight::Quantized(value) => {
            let input_name = format!("llama.packed.{name}");
            let weight_node = graph.input_dtype(
                &input_name,
                value.descriptor().logical_shape.clone(),
                DType::F32,
            );
            let transposed = graph.permute(weight_node, vec![1, 0])?;
            let output = graph.matmul(input, transposed)?;
            quantized.insert(
                output.index(),
                QuantizedLinearBinding {
                    tensor: name.to_owned(),
                    input_name,
                    weight_node,
                    weight: value.clone(),
                },
            );
            Ok(output)
        }
    }
}

pub(super) fn append_model_embedding(
    graph: &mut Graph,
    token_node: NodeId,
    token_data: &TensorData,
    state: &LlamaModelState,
    bindings: &mut HashMap<String, TensorData>,
) -> Result<NodeId, LlamaModelError> {
    match &state.embedding {
        LlamaLinearWeight::Dense(value) => {
            let weight = graph.constant(value.clone());
            let schema = state.config.schema;
            match token_data.shape().dims() {
                [_] => Ok(embedding(graph, token_node, weight, schema.embedding_dim)?),
                [batch, sequence] => {
                    let weight =
                        graph.reshape(weight, [1, schema.vocab_size, schema.embedding_dim])?;
                    let weight =
                        graph.expand(weight, [*batch, schema.vocab_size, schema.embedding_dim])?;
                    let indices = graph.reshape(token_node, [*batch, *sequence, 1])?;
                    let indices =
                        graph.expand(indices, [*batch, *sequence, schema.embedding_dim])?;
                    Ok(graph.gather(weight, indices, 1)?)
                }
                _ => Err(LlamaModelError::InvalidConfig {
                    field: "token shape",
                }),
            }
        }
        LlamaLinearWeight::Quantized(value) => {
            let rows = QuantizedRowGatherPlan::new(value)
                .and_then(|plan| plan.execute(token_data, value))
                .map_err(|error| LlamaModelError::PackedWeight {
                    tensor: TOKEN_EMBEDDING.to_owned(),
                    reason: error.to_string(),
                })?;
            let input_name = "llama.packed.token_embedding.rows";
            let input = graph.input_dtype_requires_grad(
                input_name,
                rows.shape().clone(),
                DType::F32,
                false,
            );
            bindings.insert(input_name.to_owned(), rows);
            Ok(input)
        }
    }
}

/// Executable N-layer Llama model retaining supported packed projections.
#[derive(Clone, Debug)]
pub struct LlamaModel {
    config: LlamaModelConfig,
    state: LlamaModelState,
}

impl LlamaModel {
    /// Couples a configuration to state validated against that exact value.
    pub fn new(config: LlamaModelConfig, state: LlamaModelState) -> Result<Self, LlamaModelError> {
        if config != state.config {
            return Err(LlamaModelError::StateConfigMismatch);
        }
        Ok(Self { config, state })
    }

    /// Constructs the model and source-compatible tokenizer from one GGUF file.
    pub fn from_gguf(file: &GgufFile<'_>) -> Result<(Self, SimpleTokenizer), LlamaModelError> {
        let tokenizer = SimpleTokenizer::from_gguf(file)?;
        let config = LlamaModelConfig::from_gguf(file)?;
        let state = LlamaModelState::bind(&config, file)?;
        Ok((Self::new(config, state)?, tokenizer))
    }

    /// Returns the immutable metadata-derived configuration.
    pub fn config(&self) -> &LlamaModelConfig {
        &self.config
    }

    /// Returns whether final logits use an explicit or tied projection matrix.
    pub const fn output_binding(&self) -> LlamaOutputBinding {
        self.state.output
    }

    pub(super) fn dense_state(&self) -> &BTreeMap<String, TensorData> {
        &self.state.dense
    }

    pub(super) const fn embedding_weight(&self) -> &LlamaLinearWeight {
        &self.state.embedding
    }

    pub(super) fn linear_weights(&self) -> &BTreeMap<String, LlamaLinearWeight> {
        &self.state.linears
    }

    pub(super) const fn model_state(&self) -> &LlamaModelState {
        &self.state
    }

    /// Builds an inspectable all-position full-sequence graph.
    pub fn plan(&self, tokens: &[u32]) -> Result<LlamaModelPlan, LlamaModelError> {
        self.plan_with_past(tokens, None)
    }

    /// Executes an uncached full sequence through the CPU semantic oracle.
    pub fn forward(&self, tokens: &[u32]) -> Result<TensorData, LlamaModelError> {
        self.plan(tokens)?.execute()
    }

    pub(super) fn plan_with_past(
        &self,
        tokens: &[u32],
        past: Option<&[LayerCache]>,
    ) -> Result<LlamaModelPlan, LlamaModelError> {
        if tokens.is_empty() {
            return Err(LlamaModelError::EmptyTokens);
        }
        let schema = self.config.schema;
        for &token in tokens {
            if usize::try_from(token).map_or(true, |token| token >= schema.vocab_size) {
                return Err(LlamaModelError::TokenOutOfRange {
                    token,
                    vocab_size: schema.vocab_size,
                });
            }
        }
        let past_len = validate_cache(&self.config, past)?;
        let total_len = past_len
            .checked_add(tokens.len())
            .ok_or(LlamaModelError::ContextOverflow)?;
        if total_len > self.config.max_context {
            return Err(LlamaModelError::ContextLength {
                requested: total_len,
                maximum: self.config.max_context,
            });
        }

        let mut graph = Graph::new();
        let token_node =
            graph.input_dtype_requires_grad("llama.tokens", [tokens.len()], DType::I64, false);
        let token_data = TensorData::from_scalars(
            [tokens.len()],
            DType::I64,
            tokens.iter().map(|token| Scalar::I(i64::from(*token))),
        )?;
        let mut bindings = HashMap::from([("llama.tokens".to_owned(), token_data.clone())]);
        let mut x = append_model_embedding(
            &mut graph,
            token_node,
            &token_data,
            &self.state,
            &mut bindings,
        )?;
        let mut cache_nodes = Vec::with_capacity(self.config.layer_count);
        let mut quantized_linears = QuantizedLinearBindings::new();
        for layer in 0..self.config.layer_count {
            let previous = past.map(|past| &past[layer]);
            let tensor_prefix = format!("blk.{layer}");
            let cache_prefix = format!("llama.cache.{layer}");
            let built = append_dense_layer(
                &mut graph,
                &mut bindings,
                x,
                super::layer::LayerState::Model(&self.state),
                &mut quantized_linears,
                &tensor_prefix,
                &cache_prefix,
                schema,
                tokens.len(),
                past_len,
                total_len,
                self.config.norm_eps,
                self.config.rope_theta,
                self.config.qk_norm,
                self.config.qkv_bias,
                previous.map(|cache| &cache.keys),
                previous.map(|cache| &cache.values),
            )?;
            x = built.output;
            cache_nodes.push((built.keys, built.values));
        }
        let norm_weight = graph.constant(self.state.dense[OUTPUT_NORM].clone());
        let normalized = rms_norm(
            &mut graph,
            x,
            norm_weight,
            schema.embedding_dim,
            self.config.norm_eps,
        )?;
        let logits = append_model_linear(
            &mut graph,
            normalized,
            &self.state,
            match self.state.output {
                LlamaOutputBinding::Explicit => OUTPUT_WEIGHT,
                LlamaOutputBinding::TiedToTokenEmbedding => TOKEN_EMBEDDING,
            },
            &mut quantized_linears,
        )?;
        let packed_logits_input = quantized_linears
            .contains_key(&logits.index())
            .then_some(normalized);
        Ok(LlamaModelPlan {
            graph,
            bindings,
            logits,
            cache_nodes,
            quantized_linears,
            packed_logits_input,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct LayerCache {
    pub(super) keys: TensorData,
    pub(super) values: TensorData,
}

/// Transactional per-layer cache for one exact N-layer model configuration.
#[derive(Clone, Debug)]
pub struct LlamaModelCache {
    config: LlamaModelConfig,
    layers: Option<Vec<LayerCache>>,
}

impl LlamaModelCache {
    /// Creates an empty cache bound to one exact model configuration.
    pub fn new(config: LlamaModelConfig) -> Self {
        Self {
            config,
            layers: None,
        }
    }
    /// Returns the prefix length shared by every committed layer.
    pub fn len(&self) -> usize {
        self.layers
            .as_ref()
            .map_or(0, |layers| layers[0].keys.shape().dims()[1])
    }
    /// Returns true before the first successful forward or after clearing.
    pub fn is_empty(&self) -> bool {
        self.layers.is_none()
    }
    /// Drops all committed per-layer keys and values.
    pub fn clear(&mut self) {
        self.layers = None;
    }
    /// Executes one token chunk and atomically commits every layer cache only
    /// after logits and all graph-produced keys and values succeed.
    pub fn forward(
        &mut self,
        model: &LlamaModel,
        tokens: &[u32],
    ) -> Result<TensorData, LlamaModelError> {
        if model.config != self.config {
            return Err(LlamaModelError::CacheConfigMismatch);
        }
        let plan = model.plan_with_past(tokens, self.layers.as_deref())?;
        let output = plan.execute_all()?;
        self.layers = Some(output.layers);
        Ok(output.logits)
    }
}

/// Inspectable N-layer graph plus fixed inputs and per-layer cache outputs.
#[derive(Debug)]
pub struct LlamaModelPlan {
    pub(super) graph: Graph,
    pub(super) bindings: HashMap<String, TensorData>,
    pub(super) logits: NodeId,
    pub(super) cache_nodes: Vec<(NodeId, NodeId)>,
    pub(super) quantized_linears: QuantizedLinearBindings,
    pub(super) packed_logits_input: Option<NodeId>,
}

impl LlamaModelPlan {
    /// Returns the typed graph before execution.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the all-position logits node.
    pub const fn logits_node(&self) -> NodeId {
        self.logits
    }

    /// Executes all-position logits through the CPU semantic oracle.
    pub fn execute(&self) -> Result<TensorData, LlamaModelError> {
        let bindings = self.cpu_bindings()?;
        execute_cpu_logits(
            &self.graph,
            &bindings,
            self.logits,
            self.packed_logits_input,
            &self.quantized_linears,
        )
    }

    fn execute_all(&self) -> Result<ModelOutput, LlamaModelError> {
        let backend = CpuBackend;
        let bindings = self.cpu_bindings()?;
        let logits = execute_cpu_logits(
            &self.graph,
            &bindings,
            self.logits,
            self.packed_logits_input,
            &self.quantized_linears,
        )?;
        let mut layers = Vec::with_capacity(self.cache_nodes.len());
        for &(keys, values) in &self.cache_nodes {
            layers.push(LayerCache {
                keys: backend.execute(&self.graph, keys, &bindings)?,
                values: backend.execute(&self.graph, values, &bindings)?,
            });
        }
        Ok(ModelOutput { logits, layers })
    }

    fn cpu_bindings(&self) -> Result<HashMap<String, TensorData>, LlamaModelError> {
        let mut bindings = self.bindings.clone();
        for (&output, binding) in &self.quantized_linears {
            if self.packed_logits_input.is_some() && output == self.logits.index() {
                continue;
            }
            let dense =
                binding
                    .weight
                    .dequantize_f32()
                    .map_err(|error| LlamaModelError::PackedWeight {
                        tensor: binding.tensor.clone(),
                        reason: error.to_string(),
                    })?;
            bindings.insert(binding.input_name.clone(), dense);
        }
        Ok(bindings)
    }
}

pub(super) fn execute_cpu_logits(
    graph: &Graph,
    bindings: &HashMap<String, TensorData>,
    logits: NodeId,
    packed_input: Option<NodeId>,
    quantized: &QuantizedLinearBindings,
) -> Result<TensorData, LlamaModelError> {
    let Some(input_node) = packed_input else {
        return Ok(CpuBackend.execute(graph, logits, bindings)?);
    };
    let binding = &quantized[&logits.index()];
    let activation = CpuBackend.execute(graph, input_node, bindings)?;
    let plan = QuantizedMatmulPlan::new(
        input_node,
        binding.weight_node,
        logits,
        graph.node(input_node)?.shape.clone(),
        binding.weight.descriptor().clone(),
    )
    .map_err(|error| LlamaModelError::PackedWeight {
        tensor: binding.tensor.clone(),
        reason: error.to_string(),
    })?;
    plan.execute(&activation, &binding.weight)
        .map_err(|error| LlamaModelError::PackedWeight {
            tensor: binding.tensor.clone(),
            reason: error.to_string(),
        })
}

struct ModelOutput {
    logits: TensorData,
    layers: Vec<LayerCache>,
}

/// Structured GGUF configuration, state, graph, or cache rejection.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaModelError {
    Metadata(GgufMetadataAccessError),
    Tokenizer(TokenizerError),
    State(GgufError),
    StateSchema(super::LlamaStateError),
    MissingMetadata(String),
    UnsupportedArchitecture(String),
    UnsupportedVariant(&'static str),
    MetadataValueOutOfRange(String),
    InvalidConfig {
        field: &'static str,
    },
    StateConfigMismatch,
    MissingTensor(String),
    UnexpectedTensor(String),
    UnsupportedPackedTensor(String),
    PackedWeight {
        tensor: String,
        reason: String,
    },
    DTypeMismatch {
        tensor: String,
        actual: DType,
    },
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
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
    EmptyBatch,
    EmptyBatchStep,
    BatchSize {
        expected: usize,
        actual: usize,
    },
    BatchTokenOutOfRange {
        row: usize,
        token: u32,
        vocab_size: usize,
    },
    BatchContextLength {
        row: usize,
        requested: usize,
        maximum: usize,
    },
    CacheConfigMismatch,
    CacheLayerCount {
        expected: usize,
        actual: usize,
    },
    CacheLengthMismatch,
    CacheShape {
        layer: usize,
        expected_heads: usize,
        expected_head_dim: usize,
        keys: Shape,
        values: Shape,
    },
    CacheDType {
        layer: usize,
        keys: DType,
        values: DType,
    },
    BatchCacheShape {
        layer: usize,
        expected: Vec<usize>,
        keys: Vec<usize>,
        values: Vec<usize>,
    },
    Graph(Error),
}

impl fmt::Display for LlamaModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama model error: {self:?}")
    }
}
impl error::Error for LlamaModelError {}
impl From<GgufMetadataAccessError> for LlamaModelError {
    fn from(value: GgufMetadataAccessError) -> Self {
        Self::Metadata(value)
    }
}
impl From<TokenizerError> for LlamaModelError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}
impl From<GgufError> for LlamaModelError {
    fn from(value: GgufError) -> Self {
        Self::State(value)
    }
}
impl From<super::LlamaStateError> for LlamaModelError {
    fn from(value: super::LlamaStateError) -> Self {
        Self::StateSchema(value)
    }
}
impl From<Error> for LlamaModelError {
    fn from(value: Error) -> Self {
        Self::Graph(value)
    }
}

fn required_string<'a>(file: &'a GgufFile<'_>, key: &str) -> Result<&'a str, LlamaModelError> {
    file.metadata_string(key)?
        .ok_or_else(|| LlamaModelError::MissingMetadata(key.to_owned()))
}
fn required_usize(file: &GgufFile<'_>, key: &str) -> Result<usize, LlamaModelError> {
    optional_usize(file, key)?.ok_or_else(|| LlamaModelError::MissingMetadata(key.to_owned()))
}
fn optional_usize(file: &GgufFile<'_>, key: &str) -> Result<Option<usize>, LlamaModelError> {
    file.metadata_u64(key)?
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| LlamaModelError::MetadataValueOutOfRange(key.to_owned()))
        })
        .transpose()
}
fn required_float(file: &GgufFile<'_>, key: &str) -> Result<f64, LlamaModelError> {
    file.metadata_f64(key)?
        .ok_or_else(|| LlamaModelError::MissingMetadata(key.to_owned()))
}
fn reject_nonzero(
    file: &GgufFile<'_>,
    key: &str,
    variant: &'static str,
) -> Result<(), LlamaModelError> {
    if file.metadata_u64(key)?.unwrap_or(0) != 0 {
        Err(LlamaModelError::UnsupportedVariant(variant))
    } else {
        Ok(())
    }
}

fn is_projection_weight(name: &str) -> bool {
    name == OUTPUT_WEIGHT
        || [
            ".attn_q.weight",
            ".attn_k.weight",
            ".attn_v.weight",
            ".attn_output.weight",
            ".ffn_gate.weight",
            ".ffn_up.weight",
            ".ffn_down.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn validate_gguf_tensor(
    file: &GgufFile<'_>,
    name: &str,
    expected: &[usize],
) -> Result<(), LlamaModelError> {
    let tensor = file
        .tensor(name)
        .ok_or_else(|| LlamaModelError::MissingTensor(name.to_owned()))?;
    if tensor.shape().dims() != expected {
        return Err(LlamaModelError::ShapeMismatch {
            tensor: name.to_owned(),
            expected: expected.to_vec(),
            actual: tensor.shape().dims().to_vec(),
        });
    }
    Ok(())
}

fn validate_cache(
    config: &LlamaModelConfig,
    past: Option<&[LayerCache]>,
) -> Result<usize, LlamaModelError> {
    let Some(past) = past else {
        return Ok(0);
    };
    if past.len() != config.layer_count {
        return Err(LlamaModelError::CacheLayerCount {
            expected: config.layer_count,
            actual: past.len(),
        });
    }
    let mut length = None;
    for (layer, cache) in past.iter().enumerate() {
        if cache.keys.dtype() != DType::F32 || cache.values.dtype() != DType::F32 {
            return Err(LlamaModelError::CacheDType {
                layer,
                keys: cache.keys.dtype(),
                values: cache.values.dtype(),
            });
        }
        let valid = |shape: &Shape| {
            shape.rank() == 3
                && shape.dims()[0] == config.schema.kv_heads
                && shape.dims()[2] == config.schema.head_dim
        };
        if !valid(cache.keys.shape())
            || !valid(cache.values.shape())
            || cache.keys.shape() != cache.values.shape()
        {
            return Err(LlamaModelError::CacheShape {
                layer,
                expected_heads: config.schema.kv_heads,
                expected_head_dim: config.schema.head_dim,
                keys: cache.keys.shape().clone(),
                values: cache.values.shape().clone(),
            });
        }
        let current = cache.keys.shape().dims()[1];
        if length.is_some_and(|length| length != current) {
            return Err(LlamaModelError::CacheLengthMismatch);
        }
        length = Some(current);
    }
    Ok(length.unwrap_or(0))
}
