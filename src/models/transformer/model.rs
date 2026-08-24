use super::{
    LlamaDecoderSchema, LlamaOutputBinding, OUTPUT_NORM, OUTPUT_WEIGHT, ROPE_FREQS,
    TOKEN_EMBEDDING,
    layer::{append_dense_layer, embedding, linear, rms_norm},
};
use crate::{
    Backend, CpuBackend, DType, Error, Graph, NodeId, Scalar, Shape, TensorData,
    gguf::{GgufError, GgufFile, GgufMetadataAccessError},
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

/// Typed configuration for the supported bias-free dense GGUF Llama model.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaModelConfig {
    architecture: String,
    layer_count: usize,
    schema: LlamaDecoderSchema,
    max_context: usize,
    norm_eps: f32,
    rope_theta: f64,
    token_ids: LlamaTokenIds,
}

impl LlamaModelConfig {
    /// Derives the exact supported dense-Llama configuration from typed GGUF
    /// metadata. Model variants requiring bias, experts, MLA, or SSM remain
    /// explicit errors rather than inferred tensor layouts.
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
        if rope_dim != head_dim {
            return Err(LlamaModelError::UnsupportedVariant(
                "partial rotary dimensions",
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
        Ok(Self {
            architecture: architecture.to_owned(),
            layer_count,
            schema,
            max_context,
            norm_eps,
            rope_theta,
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
    /// Returns BOS/EOS/EOT IDs validated by the tokenizer metadata path.
    pub const fn token_ids(&self) -> LlamaTokenIds {
        self.token_ids
    }
}

/// Atomically materialized and completely name/shape-validated N-layer state.
#[derive(Clone, Debug)]
pub struct LlamaModelState {
    config: LlamaModelConfig,
    state: BTreeMap<String, TensorData>,
    output: LlamaOutputBinding,
}

impl LlamaModelState {
    /// Atomically materializes the complete GGUF state to F32 and validates
    /// every root and `blk.N` name, dtype, and shape against `config`.
    pub fn bind(config: &LlamaModelConfig, file: &GgufFile<'_>) -> Result<Self, LlamaModelError> {
        let state = file.materialize_state_f32()?;
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
            .checked_mul(9)
            .and_then(|count| count.checked_add(3))
            .ok_or_else(|| {
                LlamaModelError::MetadataValueOutOfRange("llama.block_count".to_owned())
            })?;
        let mut expected = Vec::with_capacity(expected_capacity.min(state.len().saturating_add(2)));
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
        }
        let allowed = expected
            .iter()
            .map(|(name, _)| name.as_str())
            .chain([OUTPUT_WEIGHT, ROPE_FREQS])
            .collect::<HashSet<_>>();
        if let Some(name) = state.keys().find(|name| !allowed.contains(name.as_str())) {
            return Err(LlamaModelError::UnexpectedTensor(name.clone()));
        }
        for (name, shape) in &expected {
            validate_tensor(&state, name, shape)?;
        }
        if state.contains_key(OUTPUT_WEIGHT) {
            validate_tensor(
                &state,
                OUTPUT_WEIGHT,
                &[schema.vocab_size, schema.embedding_dim],
            )?;
        }
        if state.contains_key(ROPE_FREQS) {
            validate_tensor(&state, ROPE_FREQS, &[schema.rope_dim / 2])?;
        }
        Ok(Self {
            config: config.clone(),
            output: if state.contains_key(OUTPUT_WEIGHT) {
                LlamaOutputBinding::Explicit
            } else {
                LlamaOutputBinding::TiedToTokenEmbedding
            },
            state,
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

    /// Returns the validated F32 tensor inventory.
    pub fn tensors(&self) -> &BTreeMap<String, TensorData> {
        &self.state
    }

    pub(super) fn output_weight(&self) -> &TensorData {
        match self.output {
            LlamaOutputBinding::Explicit => &self.state[OUTPUT_WEIGHT],
            LlamaOutputBinding::TiedToTokenEmbedding => &self.state[TOKEN_EMBEDDING],
        }
    }
}

/// Executable N-layer dense Llama model whose tensors are dequantized to F32.
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

    pub(super) fn state_map(&self) -> &BTreeMap<String, TensorData> {
        &self.state.state
    }

    pub(super) fn output_weight(&self) -> &TensorData {
        self.state.output_weight()
    }

    /// Builds an inspectable all-position full-sequence graph.
    pub fn plan(&self, tokens: &[u32]) -> Result<LlamaModelPlan, LlamaModelError> {
        self.plan_with_past(tokens, None)
    }

    /// Executes an uncached full sequence through the CPU semantic oracle.
    pub fn forward(&self, tokens: &[u32]) -> Result<TensorData, LlamaModelError> {
        self.plan(tokens)?.execute()
    }

    fn plan_with_past(
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
        let mut bindings = HashMap::from([(
            "llama.tokens".to_owned(),
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
        let mut cache_nodes = Vec::with_capacity(self.config.layer_count);
        for layer in 0..self.config.layer_count {
            let previous = past.map(|past| &past[layer]);
            let tensor_prefix = format!("blk.{layer}");
            let cache_prefix = format!("llama.cache.{layer}");
            let built = append_dense_layer(
                &mut graph,
                &mut bindings,
                x,
                &self.state.state,
                &tensor_prefix,
                &cache_prefix,
                schema,
                tokens.len(),
                past_len,
                total_len,
                self.config.norm_eps,
                self.config.rope_theta,
                previous.map(|cache| &cache.keys),
                previous.map(|cache| &cache.values),
            )?;
            x = built.output;
            cache_nodes.push((built.keys, built.values));
        }
        let norm_weight = graph.constant(self.state.state[OUTPUT_NORM].clone());
        let normalized = rms_norm(
            &mut graph,
            x,
            norm_weight,
            schema.embedding_dim,
            self.config.norm_eps,
        )?;
        let output_weight = graph.constant(self.state.output_weight().clone());
        let logits = linear(&mut graph, normalized, output_weight)?;
        Ok(LlamaModelPlan {
            graph,
            bindings,
            logits,
            cache_nodes,
        })
    }
}

#[derive(Clone, Debug)]
struct LayerCache {
    keys: TensorData,
    values: TensorData,
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
    graph: Graph,
    bindings: HashMap<String, TensorData>,
    logits: NodeId,
    cache_nodes: Vec<(NodeId, NodeId)>,
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
        Ok(CpuBackend.execute(&self.graph, self.logits, &self.bindings)?)
    }

    fn execute_all(&self) -> Result<ModelOutput, LlamaModelError> {
        let backend = CpuBackend;
        let logits = backend.execute(&self.graph, self.logits, &self.bindings)?;
        let mut layers = Vec::with_capacity(self.cache_nodes.len());
        for &(keys, values) in &self.cache_nodes {
            layers.push(LayerCache {
                keys: backend.execute(&self.graph, keys, &self.bindings)?,
                values: backend.execute(&self.graph, values, &self.bindings)?,
            });
        }
        Ok(ModelOutput { logits, layers })
    }
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

fn validate_tensor(
    state: &BTreeMap<String, TensorData>,
    name: &str,
    expected: &[usize],
) -> Result<(), LlamaModelError> {
    let tensor = state
        .get(name)
        .ok_or_else(|| LlamaModelError::MissingTensor(name.to_owned()))?;
    if tensor.dtype() != DType::F32 {
        return Err(LlamaModelError::DTypeMismatch {
            tensor: name.to_owned(),
            actual: tensor.dtype(),
        });
    }
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
