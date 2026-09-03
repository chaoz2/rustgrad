//! Fixed-shape, device-resident dense Llama token execution on Metal.

use super::{
    LlamaLinearWeight, LlamaModel, LlamaOutputBinding, LlamaQkNorm, OUTPUT_NORM, OUTPUT_WEIGHT,
    ROPE_FREQS, TOKEN_EMBEDDING,
    layer::{add_bias, permute_rope_projection, rms_norm},
};
use crate::runtime::metal::{
    MetalAppendStateInferencePlan, MetalDevice, MetalDeviceRunReport, MetalDeviceSession,
    MetalDeviceSessionSummary, MetalError, MetalRenderer, RenderedMetal,
};
use crate::{
    AttentionOptions, CapturedAppendStateInference, CapturedInferenceError, CapturedSchedule,
    DType, Error, ExecutionPlanSummary, Graph, InferenceAppendStateLink, NodeId, ReplayInput,
    Scalar, Shape, TensorData,
    engine::capture::QuantizedCaptureBinding,
    gguf::{GgmlType, QuantizedTensorData},
};
use std::{collections::BTreeMap, error, fmt};

const TOKEN_INPUT: &str = "llama.token";
const POSITION_INPUT: &str = "llama.position";
const ROPE_TABLE: &str = "llama.rope.table";

/// Resource-free deployment of one dense F32, batch-one Llama token body.
///
/// The graph owns fixed-capacity K/V state and is captured exactly once. Token,
/// scalar position are the only per-run host inputs; the row-shaped append
/// index is derived and materialized on device from that position. All GGUF
/// weights and the precomputed RoPE table are immutable named residents.
pub struct LlamaMetalStepPlan {
    inner: MetalAppendStateInferencePlan,
    max_context: usize,
    vocab_size: usize,
    layer_count: usize,
    output_binding: LlamaOutputBinding,
}

/// Persistent device-resident Llama token session whose position advances only
/// after every K/V row and the public logits commit successfully.
pub struct LlamaMetalStepSession {
    inner: MetalDeviceSession,
    max_context: usize,
    vocab_size: usize,
}

/// One successfully committed token invocation.
pub struct LlamaMetalStep {
    logits: TensorData,
    position: usize,
    report: MetalDeviceRunReport,
}

impl LlamaMetalStep {
    /// Returns the `[1, vocab]` F32 logits downloaded for this token.
    pub const fn logits(&self) -> &TensorData {
        &self.logits
    }

    /// Returns the zero-based position consumed by this invocation.
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the exact successful underlying Metal run report.
    pub const fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }

    /// Consumes the step into detached logits and its run report.
    pub fn into_parts(self) -> (TensorData, MetalDeviceRunReport) {
        (self.logits, self.report)
    }
}

/// Dense-Llama planning, binding, graph, or strict Metal failure.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaMetalStepError {
    Graph(Error),
    Capture(CapturedInferenceError),
    Metal(MetalError),
    PackedTensor(String),
    Dimension(&'static str),
    TokenOutOfRange { token: u32, vocab_size: usize },
    ContextExhausted { position: usize, maximum: usize },
}

impl fmt::Display for LlamaMetalStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama Metal step error: {self:?}")
    }
}

impl error::Error for LlamaMetalStepError {}

impl From<Error> for LlamaMetalStepError {
    fn from(value: Error) -> Self {
        Self::Graph(value)
    }
}

impl From<CapturedInferenceError> for LlamaMetalStepError {
    fn from(value: CapturedInferenceError) -> Self {
        Self::Capture(value)
    }
}

impl From<MetalError> for LlamaMetalStepError {
    fn from(value: MetalError) -> Self {
        Self::Metal(value)
    }
}

impl LlamaMetalStepPlan {
    /// Builds, captures, and renders one reusable fixed-capacity token graph.
    /// Packed GGUF tensors and dimensions that cannot be represented by the
    /// I32 runtime ABI reject before any Metal resource is created.
    pub fn new(model: &LlamaModel, renderer: MetalRenderer) -> Result<Self, LlamaMetalStepError> {
        let config = model.config();
        let schema = config.schema();
        if schema.vocab_size() > i32::MAX as usize {
            return Err(LlamaMetalStepError::Dimension("vocabulary exceeds I32"));
        }
        if config.max_context() > i32::MAX as usize {
            return Err(LlamaMetalStepError::Dimension("context exceeds I32"));
        }
        let built = build_step_graph(model)?;
        let host_gathers = if built.packed_embedding {
            &[POSITION_INPUT][..]
        } else {
            &[TOKEN_INPUT, POSITION_INPUT][..]
        };
        let captured = CapturedAppendStateInference::from_graph_residents(
            &built.graph,
            &[built.logits],
            &built.state_links,
            built.initial_state,
            built.residents,
            &built.quantized,
            host_gathers,
        )?;
        let inner = MetalAppendStateInferencePlan::new(captured, renderer)?;
        if inner.summary().fallback_count != 0 {
            return Err(LlamaMetalStepError::Metal(MetalError::Unsupported(
                "Llama token plan admitted a fallback".into(),
            )));
        }
        Ok(Self {
            inner,
            max_context: config.max_context(),
            vocab_size: schema.vocab_size(),
            layer_count: config.layer_count(),
            output_binding: model.output_binding(),
        })
    }

    /// Returns the capture plus exact resident/state payload identity.
    pub const fn deployment_identity(&self) -> u64 {
        self.inner.deployment_identity()
    }

    /// Returns the exact authenticated token-step capture.
    pub fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    /// Returns backend-neutral logical schedule and memory facts.
    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.inner.execution_plan()
    }

    /// Returns deterministic Metal resource and execution planning facts.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    /// Returns exact immutable weight and RoPE resident schemas.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    /// Returns the ordered per-layer K/V state-input schemas.
    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.inner.state_inputs()
    }

    /// Returns the token and scalar-position transient schemas.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    /// Returns every rendered schedule item for inspection.
    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    /// Returns the fixed K/V capacity.
    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Returns the exact GGUF vocabulary row count.
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the transformer layer count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Returns whether logits use an explicit or tied output weight.
    pub const fn output_binding(&self) -> LlamaOutputBinding {
        self.output_binding
    }

    /// Creates persistent resources, uploads every immutable resident once,
    /// and initializes the single physical K/V state bank at position zero.
    pub fn prepare(
        self,
        device: MetalDevice,
    ) -> Result<LlamaMetalStepSession, LlamaMetalStepError> {
        Ok(LlamaMetalStepSession {
            inner: self.inner.prepare(device)?,
            max_context: self.max_context,
            vocab_size: self.vocab_size,
        })
    }
}

impl LlamaMetalStepSession {
    /// Returns the number of tokens atomically committed to device K/V state.
    pub fn position(&self) -> usize {
        self.inner
            .committed_state_position()
            .expect("Llama plans always use append-state sessions")
    }

    /// Returns true after the final valid context position commits.
    pub fn is_full(&self) -> bool {
        self.position() == self.max_context
    }

    /// Returns the strict session for resource, metric, and kernel inspection.
    pub const fn metal_session(&self) -> &MetalDeviceSession {
        &self.inner
    }

    /// Runs exactly one token. Invalid tokens, a full context, and failed
    /// device transactions preserve both position and the prior committed K/V rows.
    pub fn run_token(&mut self, token: u32) -> Result<LlamaMetalStep, LlamaMetalStepError> {
        if token > i32::MAX as u32 || token as usize >= self.vocab_size {
            return Err(LlamaMetalStepError::TokenOutOfRange {
                token,
                vocab_size: self.vocab_size,
            });
        }
        let position = self.position();
        if position >= self.max_context {
            return Err(LlamaMetalStepError::ContextExhausted {
                position,
                maximum: self.max_context,
            });
        }
        let position_value = step_position_input(position)?;
        let inputs = BTreeMap::from([
            (
                TOKEN_INPUT.to_owned(),
                TensorData::from_scalars([1, 1], DType::I32, [Scalar::I(i64::from(token))])?,
            ),
            (POSITION_INPUT.to_owned(), position_value),
        ]);
        let run = self.inner.run(&inputs)?;
        let (mut outputs, report) = run.into_parts();
        debug_assert_eq!(outputs.len(), 1);
        let logits = outputs
            .pop()
            .expect("capture authenticates one Llama output");
        debug_assert_eq!(logits.shape().dims(), [1, self.vocab_size]);
        debug_assert_eq!(logits.dtype(), DType::F32);
        Ok(LlamaMetalStep {
            logits,
            position,
            report,
        })
    }
}

fn step_position_input(position: usize) -> Result<TensorData, LlamaMetalStepError> {
    let position = i32::try_from(position)
        .map_err(|_| LlamaMetalStepError::Dimension("position exceeds I32"))?;
    Ok(TensorData::from_scalars(
        [1],
        DType::I32,
        [Scalar::I(i64::from(position))],
    )?)
}

struct BuiltStepGraph {
    graph: Graph,
    residents: BTreeMap<String, (NodeId, TensorData)>,
    quantized: Vec<QuantizedCaptureBinding>,
    initial_state: BTreeMap<String, TensorData>,
    state_links: Vec<InferenceAppendStateLink>,
    packed_embedding: bool,
    logits: NodeId,
}

fn build_step_graph(model: &LlamaModel) -> Result<BuiltStepGraph, LlamaMetalStepError> {
    let config = model.config();
    let schema = config.schema();
    let mut graph = Graph::new();
    let token = graph.input_dtype_requires_grad(TOKEN_INPUT, [1, 1], DType::I32, false);
    let position = graph.input_dtype_requires_grad(POSITION_INPUT, [1], DType::I32, false);
    let append_index_shape = Shape::new([1, schema.kv_heads(), 1, schema.head_dim()]);
    let append_index = graph.reshape(position, vec![1; append_index_shape.rank()])?;
    let append_index = graph.expand(append_index, append_index_shape)?;
    let mut residents = BTreeMap::new();
    let mut nodes = BTreeMap::new();
    let mut packed = BTreeMap::new();
    insert_weight(
        &mut graph,
        &mut residents,
        &mut nodes,
        &mut packed,
        TOKEN_EMBEDDING,
        model.embedding_weight(),
    )?;
    for (name, value) in model
        .dense_state()
        .iter()
        .filter(|(name, _)| name.as_str() != ROPE_FREQS)
    {
        insert_resident(&mut graph, &mut residents, &mut nodes, name, value)?;
    }
    for (name, weight) in model.linear_weights() {
        insert_weight(
            &mut graph,
            &mut residents,
            &mut nodes,
            &mut packed,
            name,
            weight,
        )?;
    }
    let rope = rope_table(config.max_context(), schema.rope_dim(), config.rope_theta())?;
    insert_resident(&mut graph, &mut residents, &mut nodes, ROPE_TABLE, &rope)?;

    let mut quantized = Vec::new();
    let mut x = lookup_embedding(
        &mut graph,
        nodes[TOKEN_EMBEDDING],
        token,
        schema.vocab_size(),
        schema.embedding_dim(),
    )?;
    let packed_embedding = packed.contains_key(TOKEN_EMBEDDING);
    if let Some(weight) = packed.get(TOKEN_EMBEDDING) {
        quantized.push(QuantizedCaptureBinding::RowGather {
            output: x,
            indices: token,
            weight: nodes[TOKEN_EMBEDDING],
            value: weight.clone(),
        });
    }
    let rope_row = lookup_rope_row(&mut graph, nodes[ROPE_TABLE], position, schema.rope_dim())?;
    let positions = TensorData::from_scalars(
        [config.max_context()],
        DType::I32,
        (0..config.max_context()).map(|value| Scalar::I(value as i64)),
    )?;
    let positions = graph.constant(positions);
    let positions = graph.reshape(positions, [1, 1, 1, config.max_context()])?;
    let position_mask = graph.reshape(position, [1, 1, 1, 1])?;
    let attention_mask = graph.le(positions, position_mask)?;

    let cache_shape = Shape::new([
        1,
        schema.kv_heads(),
        config.max_context(),
        schema.head_dim(),
    ]);
    let mut initial_state = BTreeMap::new();
    let state_count = config
        .layer_count()
        .checked_mul(2)
        .ok_or(LlamaMetalStepError::Dimension("KV state count overflow"))?;
    let mut state_links = Vec::with_capacity(state_count);
    for layer in 0..config.layer_count() {
        let key_name = format!("llama.state.{layer}.key");
        let value_name = format!("llama.state.{layer}.value");
        let past_key =
            graph.input_dtype_requires_grad(&key_name, cache_shape.clone(), DType::F32, false);
        let past_value =
            graph.input_dtype_requires_grad(&value_name, cache_shape.clone(), DType::F32, false);
        initial_state.insert(
            key_name,
            TensorData::zeros_with_dtype(cache_shape.clone(), DType::F32)?,
        );
        initial_state.insert(
            value_name,
            TensorData::zeros_with_dtype(cache_shape.clone(), DType::F32)?,
        );
        let built = StepLayerBuildContext {
            graph: &mut graph,
            nodes: &nodes,
            packed: &packed,
            quantized: &mut quantized,
            config,
            rope_row,
            append_index,
            attention_mask,
        }
        .append(x, layer, past_key, past_value)?;
        x = built.output;
        state_links.extend([
            InferenceAppendStateLink::new(
                past_key,
                built.key,
                position,
                append_index,
                built.key_update,
                2,
            ),
            InferenceAppendStateLink::new(
                past_value,
                built.value,
                position,
                append_index,
                built.value_update,
                2,
            ),
        ]);
    }

    let normalized = rms_norm(
        &mut graph,
        x,
        nodes[OUTPUT_NORM],
        schema.embedding_dim(),
        config.norm_eps(),
    )?;
    let output_name = match model.output_binding() {
        LlamaOutputBinding::Explicit => OUTPUT_WEIGHT,
        LlamaOutputBinding::TiedToTokenEmbedding => TOKEN_EMBEDDING,
    };
    let logits = model_linear(
        &mut graph,
        normalized,
        output_name,
        &nodes,
        &packed,
        &mut quantized,
    )?;
    let logits = graph.reshape(logits, [1, schema.vocab_size()])?;
    Ok(BuiltStepGraph {
        graph,
        residents,
        quantized,
        initial_state,
        state_links,
        packed_embedding,
        logits,
    })
}

fn insert_weight(
    graph: &mut Graph,
    residents: &mut BTreeMap<String, (NodeId, TensorData)>,
    nodes: &mut BTreeMap<String, NodeId>,
    packed: &mut BTreeMap<String, QuantizedTensorData>,
    name: &str,
    weight: &LlamaLinearWeight,
) -> Result<(), LlamaMetalStepError> {
    match weight {
        LlamaLinearWeight::Dense(value) => insert_resident(graph, residents, nodes, name, value),
        LlamaLinearWeight::Quantized(value) => {
            if !matches!(
                value.descriptor().ggml_type,
                GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K
            ) {
                return Err(LlamaMetalStepError::PackedTensor(format!(
                    "{name} ({:?})",
                    value.descriptor().ggml_type
                )));
            }
            value
                .validate()
                .map_err(|error| LlamaMetalStepError::PackedTensor(format!("{name} ({error})")))?;
            if residents.contains_key(name) || packed.contains_key(name) {
                return Err(LlamaMetalStepError::Dimension(
                    "duplicate Llama resident name",
                ));
            }
            let node = graph.input_dtype_requires_grad(
                format!("llama.packed.{name}"),
                value.descriptor().logical_shape.clone(),
                DType::F32,
                false,
            );
            nodes.insert(name.to_owned(), node);
            packed.insert(name.to_owned(), value.clone());
            Ok(())
        }
    }
}

fn insert_resident(
    graph: &mut Graph,
    residents: &mut BTreeMap<String, (NodeId, TensorData)>,
    nodes: &mut BTreeMap<String, NodeId>,
    name: &str,
    value: &TensorData,
) -> Result<(), LlamaMetalStepError> {
    if value.dtype() != DType::F32 {
        return Err(LlamaMetalStepError::Dimension(
            "dense Llama residents must be F32",
        ));
    }
    if residents.contains_key(name) {
        return Err(LlamaMetalStepError::Dimension(
            "duplicate Llama resident name",
        ));
    }
    let node = graph.input_dtype_requires_grad(name, value.shape().clone(), DType::F32, false);
    residents.insert(name.to_owned(), (node, value.clone()));
    nodes.insert(name.to_owned(), node);
    Ok(())
}

// Both scalar sources are host-validated before any driver work. Capture then
// authenticates their value-preserving reshape/expand lineage into raw Gather.
fn lookup_embedding(
    graph: &mut Graph,
    embedding: NodeId,
    token: NodeId,
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<NodeId, Error> {
    let embedding = graph.reshape(embedding, [1, vocab_size, embedding_dim])?;
    let index = graph.reshape(token, [1, 1, 1])?;
    let index = graph.expand(index, [1, 1, embedding_dim])?;
    graph.gather(embedding, index, 1)
}

fn lookup_rope_row(
    graph: &mut Graph,
    table: NodeId,
    position: NodeId,
    rope_dim: usize,
) -> Result<NodeId, Error> {
    let index = graph.reshape(position, [1, 1])?;
    let index = graph.expand(index, [1, rope_dim])?;
    graph.gather(table, index, 0)
}

// Keep the exact materialized row and raw Scatter boundary isolated so the
// append-state capture can authenticate one device-produced dense update.
fn append_cache_row(
    graph: &mut Graph,
    state: NodeId,
    index: NodeId,
    value: NodeId,
) -> Result<(NodeId, NodeId), Error> {
    let update = graph.contiguous(value)?;
    let output = graph.scatter(state, index, update, 2)?;
    Ok((update, output))
}

fn rope_table(
    max_context: usize,
    rope_dim: usize,
    theta: f64,
) -> Result<TensorData, LlamaMetalStepError> {
    let half = rope_dim / 2;
    let mut values = Vec::with_capacity(
        max_context
            .checked_mul(rope_dim)
            .ok_or(LlamaMetalStepError::Dimension("RoPE table overflow"))?,
    );
    for position in 0..max_context {
        let angles = (0..half)
            .map(|index| {
                let frequency = 1.0 / theta.powf((2 * index) as f64 / rope_dim as f64);
                position as f64 * frequency
            })
            .collect::<Vec<_>>();
        values.extend(angles.iter().map(|angle| angle.cos() as f32));
        values.extend(angles.iter().map(|angle| angle.sin() as f32));
    }
    Ok(TensorData::new([max_context, rope_dim], values)?)
}

struct StepLayerNodes {
    output: NodeId,
    key: NodeId,
    value: NodeId,
    key_update: NodeId,
    value_update: NodeId,
}

struct StepLayerBuildContext<'a> {
    graph: &'a mut Graph,
    nodes: &'a BTreeMap<String, NodeId>,
    packed: &'a BTreeMap<String, QuantizedTensorData>,
    quantized: &'a mut Vec<QuantizedCaptureBinding>,
    config: &'a super::LlamaModelConfig,
    rope_row: NodeId,
    append_index: NodeId,
    attention_mask: NodeId,
}

impl StepLayerBuildContext<'_> {
    fn append(
        self,
        mut x: NodeId,
        layer: usize,
        past_key: NodeId,
        past_value: NodeId,
    ) -> Result<StepLayerNodes, LlamaMetalStepError> {
        let Self {
            graph,
            nodes,
            packed,
            quantized,
            config,
            rope_row,
            append_index,
            attention_mask,
        } = self;
        let schema = config.schema();
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        let attn_norm = rms_norm(
            graph,
            x,
            nodes[&name("attn_norm.weight")],
            schema.embedding_dim(),
            config.norm_eps(),
        )?;
        let mut query = model_linear(
            graph,
            attn_norm,
            &name("attn_q.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let mut key = model_linear(
            graph,
            attn_norm,
            &name("attn_k.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let value = model_linear(
            graph,
            attn_norm,
            &name("attn_v.weight"),
            nodes,
            packed,
            quantized,
        )?;
        query = permute_rope_projection(
            graph,
            query,
            schema.query_heads(),
            schema.head_dim(),
            schema.rope_dim(),
            true,
        )?;
        key = permute_rope_projection(
            graph,
            key,
            schema.kv_heads(),
            schema.head_dim(),
            schema.rope_dim(),
            false,
        )?;
        query = add_bias(
            graph,
            query,
            config.qkv_bias().then(|| nodes[&name("attn_q.bias")]),
        )?;
        key = add_bias(
            graph,
            key,
            config.qkv_bias().then(|| nodes[&name("attn_k.bias")]),
        )?;
        let value = add_bias(
            graph,
            value,
            config.qkv_bias().then(|| nodes[&name("attn_v.bias")]),
        )?;
        if config.qk_norm() == LlamaQkNorm::PerProjection {
            query = rms_norm(
                graph,
                query,
                nodes[&name("attn_q_norm.weight")],
                schema.query_heads() * schema.head_dim(),
                config.norm_eps(),
            )?;
            key = rms_norm(
                graph,
                key,
                nodes[&name("attn_k_norm.weight")],
                schema.kv_heads() * schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        query = graph.reshape(query, [1, 1, schema.query_heads(), schema.head_dim()])?;
        query = graph.permute(query, vec![0, 2, 1, 3])?;
        key = graph.reshape(key, [1, 1, schema.kv_heads(), schema.head_dim()])?;
        key = graph.permute(key, vec![0, 2, 1, 3])?;
        let mut value = graph.reshape(value, [1, 1, schema.kv_heads(), schema.head_dim()])?;
        value = graph.permute(value, vec![0, 2, 1, 3])?;
        if config.qk_norm() == LlamaQkNorm::PerHead {
            query = rms_norm(
                graph,
                query,
                nodes[&name("attn_q_norm.weight")],
                schema.head_dim(),
                config.norm_eps(),
            )?;
            key = rms_norm(
                graph,
                key,
                nodes[&name("attn_k_norm.weight")],
                schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        let (query, key) = apply_resident_rope(graph, query, key, rope_row, schema)?;
        let (key_update, next_key) = append_cache_row(graph, past_key, append_index, key)?;
        let (value_update, next_value) = append_cache_row(graph, past_value, append_index, value)?;
        let attended = graph.scaled_dot_product_attention(
            query,
            next_key,
            next_value,
            Some(attention_mask),
            AttentionOptions {
                enable_gqa: true,
                ..AttentionOptions::default()
            },
        )?;
        let attended = graph.permute(attended, vec![0, 2, 1, 3])?;
        let attended = graph.reshape(attended, [1, 1, schema.query_heads() * schema.head_dim()])?;
        let attended = model_linear(
            graph,
            attended,
            &name("attn_output.weight"),
            nodes,
            packed,
            quantized,
        )?;
        x = graph.add(x, attended)?;

        let normalized = rms_norm(
            graph,
            x,
            nodes[&name("ffn_norm.weight")],
            schema.embedding_dim(),
            config.norm_eps(),
        )?;
        let gate = model_linear(
            graph,
            normalized,
            &name("ffn_gate.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let gate = graph.silu(gate)?;
        let up = model_linear(
            graph,
            normalized,
            &name("ffn_up.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let gated = graph.mul(gate, up)?;
        let down = model_linear(
            graph,
            gated,
            &name("ffn_down.weight"),
            nodes,
            packed,
            quantized,
        )?;
        x = graph.add(x, down)?;
        Ok(StepLayerNodes {
            output: x,
            key: next_key,
            value: next_value,
            key_update,
            value_update,
        })
    }
}

fn linear(graph: &mut Graph, input: NodeId, weight: NodeId) -> Result<NodeId, Error> {
    let weight = graph.permute(weight, vec![1, 0])?;
    graph.matmul(input, weight)
}

fn model_linear(
    graph: &mut Graph,
    input: NodeId,
    name: &str,
    nodes: &BTreeMap<String, NodeId>,
    packed: &BTreeMap<String, QuantizedTensorData>,
    quantized: &mut Vec<QuantizedCaptureBinding>,
) -> Result<NodeId, LlamaMetalStepError> {
    let weight = nodes[name];
    let output = linear(graph, input, weight)?;
    if let Some(value) = packed.get(name) {
        quantized.push(QuantizedCaptureBinding::Matmul {
            output,
            activation: input,
            weight,
            value: value.clone(),
        });
    }
    Ok(output)
}

fn apply_resident_rope(
    graph: &mut Graph,
    query: NodeId,
    key: NodeId,
    row: NodeId,
    schema: super::LlamaDecoderSchema,
) -> Result<(NodeId, NodeId), LlamaMetalStepError> {
    let half = schema.rope_dim() / 2;
    let cos = graph.shrink(row, vec![(0, 1), (0, half)])?;
    let sin = graph.shrink(row, vec![(0, 1), (half, schema.rope_dim())])?;
    let cos = graph.reshape(cos, [1, 1, 1, half])?;
    let sin = graph.reshape(sin, [1, 1, 1, half])?;
    Ok((
        rotate(graph, query, cos, sin, schema.rope_dim(), schema.head_dim())?,
        rotate(graph, key, cos, sin, schema.rope_dim(), schema.head_dim())?,
    ))
}

fn rotate(
    graph: &mut Graph,
    input: NodeId,
    cos: NodeId,
    sin: NodeId,
    rope_dim: usize,
    head_dim: usize,
) -> Result<NodeId, Error> {
    let shape = graph.shape(input)?.dims().to_vec();
    let half = rope_dim / 2;
    let mut first_bounds = shape
        .iter()
        .copied()
        .map(|dim| (0, dim))
        .collect::<Vec<_>>();
    first_bounds[3] = (0, half);
    let first = graph.shrink(input, first_bounds)?;
    let mut second_bounds = shape
        .iter()
        .copied()
        .map(|dim| (0, dim))
        .collect::<Vec<_>>();
    second_bounds[3] = (half, rope_dim);
    let second = graph.shrink(input, second_bounds)?;
    let first_cos = graph.mul(first, cos)?;
    let second_sin = graph.mul(second, sin)?;
    let rotated_first = graph.sub(first_cos, second_sin)?;
    let second_cos = graph.mul(second, cos)?;
    let first_sin = graph.mul(first, sin)?;
    let rotated_second = graph.add(second_cos, first_sin)?;
    let mut parts = vec![rotated_first, rotated_second];
    if rope_dim != head_dim {
        let mut tail_bounds = shape
            .iter()
            .copied()
            .map(|dim| (0, dim))
            .collect::<Vec<_>>();
        tail_bounds[3] = (rope_dim, head_dim);
        parts.push(graph.shrink(input, tail_bounds)?);
    }
    graph.concat(parts, 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::metal::MetalCapabilities;
    use crate::{Backend, CpuBackend, Op};
    use std::collections::HashMap;

    fn renderer() -> MetalRenderer {
        MetalRenderer::new(
            8,
            MetalCapabilities {
                max_buffer_length: 1 << 24,
                unified_memory: true,
                family: "MockApple9".into(),
            },
        )
        .unwrap()
    }

    fn assert_close(actual: &TensorData, expected: &TensorData) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), expected.dtype());
        for (&actual, &expected) in actual.values().iter().zip(expected.values()) {
            assert!(
                (actual - expected).abs() <= 3e-5,
                "{actual} differs from {expected}"
            );
        }
    }

    #[test]
    fn dense_gqa_step_graph_uses_raw_gathers_and_owned_append_rows() {
        let (model, _, _) = super::super::model_tests::make_variant_model(4);
        let built = build_step_graph(&model).unwrap();
        assert_eq!(
            (0..built.graph.node_count())
                .filter(|&index| matches!(
                    built.graph.op(NodeId::from_index(index)).unwrap(),
                    Op::Gather { .. }
                ))
                .count(),
            2
        );
        assert_eq!(
            (0..built.graph.node_count())
                .filter(|&index| matches!(
                    built.graph.op(NodeId::from_index(index)).unwrap(),
                    Op::Scatter { add: false, .. }
                ))
                .count(),
            model.config().layer_count() * 2
        );
        assert!(built.state_links.iter().all(|link| {
            matches!(built.graph.op(link.output()).unwrap(), Op::Scatter { .. })
                && matches!(
                    built.graph.op(link.updates()).unwrap(),
                    Op::Contiguous { .. }
                )
                && link.index()
                    == built
                        .state_links
                        .first()
                        .expect("Llama has K/V state")
                        .index()
        }));
        let index = built.state_links[0].index();
        let Op::Expand {
            input: reshaped, ..
        } = built.graph.op(index).unwrap()
        else {
            panic!("append index must be the shared scalar expansion")
        };
        assert!(matches!(
            built.graph.op(*reshaped).unwrap(),
            Op::Reshape { input, .. } if *input == built.state_links[0].position()
        ));
        let mut states = built.initial_state.clone();
        let mut oracle = super::super::LlamaModelCache::new(model.config().clone());
        for (position, token) in [3u32, 4, 5].into_iter().enumerate() {
            let mut bindings = built
                .residents
                .iter()
                .map(|(name, (_, value))| (name.clone(), value.clone()))
                .chain(states.clone())
                .collect::<HashMap<_, _>>();
            bindings.insert(
                TOKEN_INPUT.into(),
                TensorData::from_scalars([1, 1], DType::I32, [Scalar::I(i64::from(token))])
                    .unwrap(),
            );
            bindings.insert(
                POSITION_INPUT.into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(position as i64)]).unwrap(),
            );
            let actual = CpuBackend
                .execute(&built.graph, built.logits, &bindings)
                .unwrap();
            let expected = oracle.forward(&model, &[token]).unwrap();
            assert_close(&actual, &expected);
            states = built
                .state_links
                .iter()
                .map(|link| {
                    let Op::Input { name } = built.graph.op(link.input()).unwrap() else {
                        unreachable!()
                    };
                    (
                        name.clone(),
                        CpuBackend
                            .execute(&built.graph, link.output(), &bindings)
                            .unwrap(),
                    )
                })
                .collect();
        }
    }

    #[test]
    fn scalar_position_is_the_only_host_append_coordinate() {
        let position = step_position_input(7).unwrap();
        assert_eq!(position.dtype(), DType::I32);
        assert_eq!(position.shape().dims(), [1]);
        assert_eq!(position.to_le_bytes().unwrap(), 7i32.to_le_bytes());
    }

    #[test]
    fn plans_tied_and_explicit_outputs_with_exact_resident_state_ownership() {
        let (tied, _, _) = super::super::model_tests::make_model(4);
        let tied_plan = LlamaMetalStepPlan::new(&tied, renderer()).unwrap();
        assert_eq!(
            tied_plan.output_binding(),
            LlamaOutputBinding::TiedToTokenEmbedding
        );
        assert_eq!(
            tied_plan.state_inputs().len(),
            tied.config().layer_count() * 2
        );
        assert_eq!(
            tied_plan.summary().state_pair_count,
            tied.config().layer_count() * 2
        );
        assert_eq!(tied_plan.summary().fallback_count, 0);
        assert_eq!(tied_plan.summary().requested_output_count, 1);
        assert_eq!(tied_plan.summary().state_bank_count, 1);
        assert_eq!(
            tied_plan.summary().append_state_work_items,
            tied.config().layer_count()
                * 2
                * tied.config().schema().kv_heads()
                * tied.config().schema().head_dim()
        );
        assert_eq!(
            tied_plan
                .transient_inputs()
                .iter()
                .map(|input| (
                    input.name.as_str(),
                    input.desc.dtype,
                    input.desc.shape.dims().to_vec(),
                ))
                .collect::<Vec<_>>(),
            [
                (POSITION_INPUT, DType::I32, vec![1]),
                (TOKEN_INPUT, DType::I32, vec![1, 1]),
            ]
        );
        assert_eq!(
            tied_plan
                .resident_inputs()
                .iter()
                .filter(|input| input.name == TOKEN_EMBEDDING)
                .count(),
            1
        );
        assert!(
            tied_plan
                .resident_inputs()
                .iter()
                .any(|input| input.name == ROPE_TABLE)
        );
        assert!(
            tied_plan
                .rendered_items()
                .all(|item| item.extent == 0 || !item.source.is_empty())
        );
        assert!(
            tied_plan
                .capture()
                .items
                .iter()
                .all(|item| item.boundary.is_none())
        );
        assert_eq!(
            tied_plan
                .rendered_items()
                .filter(|item| item.indexed_movement().is_some())
                .count(),
            0
        );
        assert_eq!(
            tied_plan
                .rendered_items()
                .filter(|item| item.source.contains("rg_metal_host_gather_f32_i32"))
                .count(),
            2
        );

        let explicit = super::super::model_tests::make_explicit_model(4);
        let explicit_plan = LlamaMetalStepPlan::new(&explicit, renderer()).unwrap();
        assert_eq!(explicit_plan.output_binding(), LlamaOutputBinding::Explicit);
        assert!(
            explicit_plan
                .resident_inputs()
                .iter()
                .any(|input| input.name == OUTPUT_WEIGHT)
        );
        assert_ne!(
            tied_plan.deployment_identity(),
            explicit_plan.deployment_identity()
        );
    }

    #[test]
    fn authenticated_gathers_change_append_deployment_not_graph_capture_identity() {
        let (model, _, _) = super::super::model_tests::make_variant_model(2);
        let built = build_step_graph(&model).unwrap();
        let unchecked = CapturedAppendStateInference::from_graph_residents(
            &built.graph,
            &[built.logits],
            &built.state_links,
            built.initial_state.clone(),
            built.residents.clone(),
            &built.quantized,
            &[],
        )
        .unwrap();
        let authenticated = CapturedAppendStateInference::from_graph_residents(
            &built.graph,
            &[built.logits],
            &built.state_links,
            built.initial_state,
            built.residents,
            &built.quantized,
            &[TOKEN_INPUT, POSITION_INPUT],
        )
        .unwrap();
        assert_eq!(
            unchecked.capture().identity,
            authenticated.capture().identity
        );
        assert_ne!(
            unchecked.deployment_identity(),
            authenticated.deployment_identity()
        );
    }
}
