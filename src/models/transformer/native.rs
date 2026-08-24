use super::{
    LlamaBatchPlan, LlamaModel, LlamaModelConfig, LlamaModelError, LlamaModelPlan,
    batch::BatchLayerCache,
    model::{LayerCache, QuantizedLinearBindings},
};
use crate::{
    Backend, CapturedBackendPolicy, CapturedItemTrace, CapturedReplayExecutor,
    CapturedReplayOptions, CapturedSchedule, CpuBackend, Graph, ItemBackend, NodeId, Op,
    ReplayError, TensorData,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error, fmt,
};

/// One observable stage kind in strict native Llama replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaNativeStageKind {
    /// A serialized captured schedule replayed under strict `NativeJit`.
    NativeSchedule,
    /// A strict native activation-by-packed-GGML projection.
    QuantizedMatmul {
        tensor: String,
        format: crate::GgmlType,
    },
    /// A typed movement/indexing boundary evaluated without arithmetic ancestors.
    Movement(&'static str),
}

/// Replay evidence for one graph node in topological order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaNativeStageTrace {
    pub node: usize,
    pub kind: LlamaNativeStageKind,
    pub items: Vec<CapturedItemTrace>,
}

/// Strict native logits, graph-produced caches, and the complete stage trace.
#[derive(Clone, Debug)]
pub struct LlamaNativeExecution {
    logits: TensorData,
    layers: Vec<LayerCache>,
    trace: Vec<LlamaNativeStageTrace>,
}

impl LlamaNativeExecution {
    pub fn logits(&self) -> &TensorData {
        &self.logits
    }
    pub fn trace(&self) -> &[LlamaNativeStageTrace] {
        &self.trace
    }
}

/// Fixed-batch strict-native outputs in input row order.
#[derive(Clone, Debug)]
pub struct LlamaBatchNativeExecution {
    rows: Vec<TensorData>,
    native: LlamaNativeExecution,
}

impl LlamaBatchNativeExecution {
    pub fn rows(&self) -> &[TensorData] {
        &self.rows
    }
    pub fn trace(&self) -> &[LlamaNativeStageTrace] {
        self.native.trace()
    }
}

#[derive(Clone, Debug)]
enum PlannedStage {
    Native {
        node: NodeId,
        artifact: Box<CapturedSchedule>,
        bytes: Vec<u8>,
        inputs: Vec<(String, NodeId)>,
    },
    Quantized {
        node: NodeId,
        tensor: String,
        format: crate::GgmlType,
        activation: NodeId,
        activation_name: String,
        artifact: Box<CapturedSchedule>,
        bytes: Vec<u8>,
    },
    Movement {
        node: NodeId,
        kind: &'static str,
    },
}

/// Fixed-shape staged Llama plan containing serialized native schedule artifacts.
#[derive(Clone, Debug)]
pub struct LlamaNativePlan {
    graph: Graph,
    bindings: HashMap<String, TensorData>,
    logits: NodeId,
    cache_nodes: Vec<(NodeId, NodeId)>,
    stages: Vec<PlannedStage>,
}

/// One fixed-shape batch artifact set; differing padded extents compile separately.
#[derive(Clone, Debug)]
pub struct LlamaBatchNativePlan {
    native: LlamaNativePlan,
    chunk_lengths: Vec<usize>,
    next_lengths: Vec<usize>,
}

impl LlamaBatchNativePlan {
    fn compile(plan: LlamaBatchPlan) -> Result<Self, LlamaNativeError> {
        let chunk_lengths = plan.chunk_lengths;
        let next_lengths = plan.next_lengths;
        let native = LlamaNativePlan::compile_parts(
            plan.graph,
            plan.bindings,
            plan.logits,
            plan.cache_nodes,
            plan.quantized_linears,
        )?;
        Ok(Self {
            native,
            chunk_lengths,
            next_lengths,
        })
    }

    pub fn artifacts(&self) -> impl Iterator<Item = &[u8]> {
        self.native.artifacts()
    }

    pub fn execute(
        &self,
        executor: &LlamaNativeExecutor,
    ) -> Result<LlamaBatchNativeExecution, LlamaNativeError> {
        let native = self.native.execute(executor)?;
        let shape = native.logits().shape().dims();
        let (batch, padded, vocab) = (shape[0], shape[1], shape[2]);
        let mut rows = Vec::with_capacity(batch);
        for (row, &length) in self.chunk_lengths.iter().enumerate() {
            let start = row * padded * vocab;
            rows.push(TensorData::new(
                [length, vocab],
                native.logits().values()[start..start + length * vocab].to_vec(),
            )?);
        }
        Ok(LlamaBatchNativeExecution { rows, native })
    }
}

impl LlamaNativePlan {
    fn compile(plan: LlamaModelPlan) -> Result<Self, LlamaNativeError> {
        Self::compile_parts(
            plan.graph,
            plan.bindings,
            plan.logits,
            plan.cache_nodes,
            plan.quantized_linears,
        )
    }

    pub(super) fn compile_parts(
        graph: Graph,
        bindings: HashMap<String, TensorData>,
        logits: NodeId,
        cache_nodes: Vec<(NodeId, NodeId)>,
        quantized_linears: QuantizedLinearBindings,
    ) -> Result<Self, LlamaNativeError> {
        let outputs = std::iter::once(logits)
            .chain(cache_nodes.iter().flat_map(|&(key, value)| [key, value]))
            .collect::<Vec<_>>();
        let reachable = reachable_nodes(&graph, &outputs)?;
        let mut stages = Vec::new();
        let mut ignored_quantized_nodes = BTreeSet::new();
        for (&output, binding) in &quantized_linears {
            ignored_quantized_nodes.insert(binding.weight_node.index());
            if let Op::Matmul { rhs, .. } = graph.op(NodeId::from_index(output))? {
                ignored_quantized_nodes.insert(rhs.index());
            }
        }
        for index in reachable.iter().copied() {
            if ignored_quantized_nodes.contains(&index) {
                continue;
            }
            let node = NodeId::from_index(index);
            let op = graph.op(node)?;
            if matches!(op, Op::Input { .. } | Op::Constant(_)) {
                continue;
            }
            if let Some(binding) = quantized_linears.get(&index) {
                let Op::Matmul { lhs, .. } = op else {
                    return Err(LlamaNativeError::UnsupportedOperation {
                        node: index,
                        operation: op_name(op),
                    });
                };
                let activation_name = format!("llama.packed.activation.{index}");
                let captured = CapturedSchedule::capture_quantized_matmul(
                    activation_name.clone(),
                    *lhs,
                    binding.weight_node,
                    node,
                    graph.shape(*lhs)?.clone(),
                    binding.weight.clone(),
                )?;
                let bytes = captured.to_bytes()?;
                let artifact = CapturedSchedule::from_bytes(&bytes)?;
                stages.push(PlannedStage::Quantized {
                    node,
                    tensor: binding.tensor.clone(),
                    format: binding.weight.descriptor().ggml_type,
                    activation: *lhs,
                    activation_name,
                    artifact: Box::new(artifact),
                    bytes,
                });
            } else if schedulable(op) {
                let (stage_graph, output, inputs) = native_stage_graph(&graph, node, op)?;
                let schedule = crate::schedule(&stage_graph, output).map_err(|error| {
                    LlamaNativeError::StageSchedule {
                        node: index,
                        reason: error.to_string(),
                    }
                })?;
                let captured = CapturedSchedule::capture(&stage_graph, &schedule, &[output])?;
                let bytes = captured.to_bytes()?;
                let artifact = CapturedSchedule::from_bytes(&bytes)?;
                stages.push(PlannedStage::Native {
                    node,
                    artifact: Box::new(artifact),
                    bytes,
                    inputs,
                });
            } else if let Some(kind) = movement_name(op) {
                stages.push(PlannedStage::Movement { node, kind });
            } else {
                return Err(LlamaNativeError::UnsupportedOperation {
                    node: index,
                    operation: op_name(op),
                });
            }
        }
        Ok(Self {
            graph,
            bindings,
            logits,
            cache_nodes,
            stages,
        })
    }

    /// Returns deterministic bytes for every captured arithmetic/reduction/matmul stage.
    pub fn artifacts(&self) -> impl Iterator<Item = &[u8]> {
        self.stages.iter().filter_map(|stage| match stage {
            PlannedStage::Native { bytes, .. } => Some(bytes.as_slice()),
            PlannedStage::Quantized { bytes, .. } => Some(bytes.as_slice()),
            PlannedStage::Movement { .. } => None,
        })
    }

    /// Executes this concrete plan through a caller-owned native compile cache.
    pub fn execute(
        &self,
        executor: &LlamaNativeExecutor,
    ) -> Result<LlamaNativeExecution, LlamaNativeError> {
        executor.execute(self)
    }
}

/// Reusable strict native executor whose compiled C kernels survive decode steps.
#[derive(Default)]
pub struct LlamaNativeExecutor {
    replay: CapturedReplayExecutor,
    #[cfg(test)]
    fail_after_stage: Option<usize>,
}

impl LlamaNativeExecutor {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compile_cache_len(&self) -> usize {
        self.replay.compile_cache_len(false)
    }
    #[cfg(test)]
    pub(super) fn inject_stage_failure(&mut self, stage: Option<usize>) {
        self.fail_after_stage = stage;
    }

    fn execute(&self, plan: &LlamaNativePlan) -> Result<LlamaNativeExecution, LlamaNativeError> {
        let mut values = BTreeMap::<usize, TensorData>::new();
        for index in 0..plan.graph.node_count() {
            let node = NodeId::from_index(index);
            match plan.graph.op(node)? {
                Op::Input { name } => {
                    if let Some(value) = plan.bindings.get(name) {
                        values.insert(index, value.clone());
                    }
                }
                Op::Constant(value) => {
                    values.insert(index, value.clone());
                }
                _ => {}
            }
        }
        let mut trace = Vec::with_capacity(plan.stages.len());
        for stage in &plan.stages {
            #[cfg(test)]
            if self.fail_after_stage == Some(trace.len()) {
                return Err(LlamaNativeError::InjectedStageFailure(trace.len()));
            }
            match stage {
                PlannedStage::Native {
                    node,
                    artifact,
                    inputs,
                    ..
                } => {
                    let mut provided = BTreeMap::new();
                    for (name, original) in inputs {
                        let value = values
                            .get(&original.index())
                            .ok_or(LlamaNativeError::MissingStageValue(original.index()))?;
                        provided.insert(name.clone(), value.clone());
                    }
                    let result = artifact
                        .replay_with_options(
                            &provided,
                            &self.replay,
                            CapturedReplayOptions {
                                backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                            },
                        )
                        .map_err(|error| LlamaNativeError::StageReplay {
                            node: node.index(),
                            reason: error.to_string(),
                        })?;
                    if result
                        .trace
                        .items
                        .iter()
                        .any(|item| item.backend != ItemBackend::NativeJit)
                    {
                        return Err(LlamaNativeError::NonNativeStage(node.index()));
                    }
                    let output = result
                        .outputs
                        .into_iter()
                        .next()
                        .ok_or(LlamaNativeError::MissingStageOutput(node.index()))?;
                    values.insert(node.index(), output);
                    trace.push(LlamaNativeStageTrace {
                        node: node.index(),
                        kind: LlamaNativeStageKind::NativeSchedule,
                        items: result.trace.items,
                    });
                }
                PlannedStage::Quantized {
                    node,
                    tensor,
                    format,
                    activation,
                    activation_name,
                    artifact,
                    ..
                } => {
                    let value = values
                        .get(&activation.index())
                        .cloned()
                        .ok_or(LlamaNativeError::MissingStageValue(activation.index()))?;
                    let provided = BTreeMap::from([(activation_name.clone(), value)]);
                    let result = artifact
                        .replay_with_options(
                            &provided,
                            &self.replay,
                            CapturedReplayOptions {
                                backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                            },
                        )
                        .map_err(|error| LlamaNativeError::StageReplay {
                            node: node.index(),
                            reason: error.to_string(),
                        })?;
                    if result
                        .trace
                        .items
                        .iter()
                        .any(|item| item.backend != ItemBackend::NativeJit)
                    {
                        return Err(LlamaNativeError::NonNativeStage(node.index()));
                    }
                    let output = result
                        .outputs
                        .into_iter()
                        .next()
                        .ok_or(LlamaNativeError::MissingStageOutput(node.index()))?;
                    values.insert(node.index(), output);
                    trace.push(LlamaNativeStageTrace {
                        node: node.index(),
                        kind: LlamaNativeStageKind::QuantizedMatmul {
                            tensor: tensor.clone(),
                            format: *format,
                        },
                        items: result.trace.items,
                    });
                }
                PlannedStage::Movement { node, kind } => {
                    let output = execute_movement(&plan.graph, *node, &values)?;
                    values.insert(node.index(), output);
                    trace.push(LlamaNativeStageTrace {
                        node: node.index(),
                        kind: LlamaNativeStageKind::Movement(kind),
                        items: Vec::new(),
                    });
                }
            }
        }
        let logits = values
            .get(&plan.logits.index())
            .cloned()
            .ok_or(LlamaNativeError::MissingStageOutput(plan.logits.index()))?;
        let layers = plan
            .cache_nodes
            .iter()
            .map(|&(key, value)| {
                Ok(LayerCache {
                    keys: values
                        .get(&key.index())
                        .cloned()
                        .ok_or(LlamaNativeError::MissingStageOutput(key.index()))?,
                    values: values
                        .get(&value.index())
                        .cloned()
                        .ok_or(LlamaNativeError::MissingStageOutput(value.index()))?,
                })
            })
            .collect::<Result<Vec<_>, LlamaNativeError>>()?;
        Ok(LlamaNativeExecution {
            logits,
            layers,
            trace,
        })
    }
}

/// Transactional single-sequence cache for strict native staged execution.
pub struct LlamaNativeCache {
    config: LlamaModelConfig,
    layers: Option<Vec<LayerCache>>,
    executor: LlamaNativeExecutor,
}

/// Transactional fixed-batch cache backed only by strict native staged execution.
pub struct LlamaBatchNativeCache {
    config: LlamaModelConfig,
    batch_size: usize,
    lengths: Vec<usize>,
    layers: Option<Vec<BatchLayerCache>>,
    executor: LlamaNativeExecutor,
}

/// Immutable single-row native KV state used to seed a fixed batch.
#[derive(Clone, Debug)]
pub(super) struct LlamaNativePrefixSnapshot {
    config: LlamaModelConfig,
    layers: Vec<LayerCache>,
    length: usize,
}

impl LlamaNativePrefixSnapshot {
    pub(super) const fn len(&self) -> usize {
        self.length
    }

    pub(super) fn byte_len(&self) -> Result<usize, LlamaNativeError> {
        self.layers.iter().try_fold(0usize, |total, layer| {
            let elements = layer
                .keys
                .len()
                .checked_add(layer.values.len())
                .ok_or(LlamaModelError::ContextOverflow)?;
            total
                .checked_add(
                    elements
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or(LlamaModelError::ContextOverflow)?,
                )
                .ok_or_else(|| LlamaModelError::ContextOverflow.into())
        })
    }
}

impl LlamaBatchNativeCache {
    pub fn new(config: LlamaModelConfig, batch_size: usize) -> Result<Self, LlamaNativeError> {
        if batch_size == 0 {
            return Err(LlamaModelError::EmptyBatch.into());
        }
        Ok(Self {
            config,
            batch_size,
            lengths: vec![0; batch_size],
            layers: None,
            executor: LlamaNativeExecutor::new(),
        })
    }

    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    pub fn clear(&mut self) {
        self.lengths.fill(0);
        self.layers = None;
    }

    pub fn compile_cache_len(&self) -> usize {
        self.executor.compile_cache_len()
    }

    pub(super) fn from_snapshots(
        config: LlamaModelConfig,
        snapshots: &[Option<LlamaNativePrefixSnapshot>],
    ) -> Result<Self, LlamaNativeError> {
        let batch_size = snapshots.len();
        if batch_size == 0 {
            return Err(LlamaModelError::EmptyBatch.into());
        }
        let schema = config.schema();
        let cache_shape = [
            batch_size,
            schema.kv_heads(),
            config.max_context(),
            schema.head_dim(),
        ];
        let row_stride = schema
            .kv_heads()
            .checked_mul(config.max_context())
            .and_then(|value| value.checked_mul(schema.head_dim()))
            .ok_or(LlamaModelError::ContextOverflow)?;
        let mut lengths = Vec::with_capacity(batch_size);
        for snapshot in snapshots.iter().flatten() {
            if snapshot.config != config {
                return Err(LlamaModelError::CacheConfigMismatch.into());
            }
            if snapshot.layers.len() != config.layer_count() {
                return Err(LlamaModelError::CacheLayerCount {
                    expected: config.layer_count(),
                    actual: snapshot.layers.len(),
                }
                .into());
            }
        }
        lengths.extend(
            snapshots
                .iter()
                .map(|snapshot| snapshot.as_ref().map_or(0, LlamaNativePrefixSnapshot::len)),
        );
        let layers = if snapshots.iter().all(Option::is_none) {
            None
        } else {
            let mut layers = Vec::with_capacity(config.layer_count());
            for layer in 0..config.layer_count() {
                let mut keys = vec![0.0; batch_size * row_stride];
                let mut values = vec![0.0; batch_size * row_stride];
                for (row, snapshot) in snapshots.iter().enumerate() {
                    let Some(snapshot) = snapshot else { continue };
                    let row_elements = schema
                        .kv_heads()
                        .checked_mul(snapshot.length)
                        .and_then(|value| value.checked_mul(schema.head_dim()))
                        .ok_or(LlamaModelError::ContextOverflow)?;
                    let expected = [schema.kv_heads(), snapshot.length, schema.head_dim()];
                    let source = &snapshot.layers[layer];
                    if source.keys.shape().dims() != expected
                        || source.values.shape().dims() != expected
                    {
                        return Err(LlamaModelError::CacheLengthMismatch.into());
                    }
                    for head in 0..schema.kv_heads() {
                        let source_start = head * snapshot.length * schema.head_dim();
                        let source_end = source_start + snapshot.length * schema.head_dim();
                        let target_start =
                            row * row_stride + head * config.max_context() * schema.head_dim();
                        let target_end = target_start + snapshot.length * schema.head_dim();
                        keys[target_start..target_end]
                            .copy_from_slice(&source.keys.values()[source_start..source_end]);
                        values[target_start..target_end]
                            .copy_from_slice(&source.values.values()[source_start..source_end]);
                    }
                    debug_assert_eq!(row_elements, source.keys.len());
                }
                layers.push(BatchLayerCache {
                    keys: TensorData::new(cache_shape, keys)?,
                    values: TensorData::new(cache_shape, values)?,
                });
            }
            Some(layers)
        };
        Ok(Self {
            config,
            batch_size,
            lengths,
            layers,
            executor: LlamaNativeExecutor::new(),
        })
    }

    pub(super) fn snapshots(&self) -> Result<Vec<LlamaNativePrefixSnapshot>, LlamaNativeError> {
        let schema = self.config.schema();
        let Some(layers) = &self.layers else {
            return Ok(Vec::new());
        };
        let row_stride = schema.kv_heads() * self.config.max_context() * schema.head_dim();
        (0..self.batch_size)
            .map(|row| {
                let length = self.lengths[row];
                let mut row_layers = Vec::with_capacity(layers.len());
                for layer in layers {
                    let mut keys =
                        Vec::with_capacity(schema.kv_heads() * length * schema.head_dim());
                    let mut values = Vec::with_capacity(keys.capacity());
                    for head in 0..schema.kv_heads() {
                        let start =
                            row * row_stride + head * self.config.max_context() * schema.head_dim();
                        let end = start + length * schema.head_dim();
                        keys.extend_from_slice(&layer.keys.values()[start..end]);
                        values.extend_from_slice(&layer.values.values()[start..end]);
                    }
                    row_layers.push(LayerCache {
                        keys: TensorData::new(
                            [schema.kv_heads(), length, schema.head_dim()],
                            keys,
                        )?,
                        values: TensorData::new(
                            [schema.kv_heads(), length, schema.head_dim()],
                            values,
                        )?,
                    });
                }
                Ok(LlamaNativePrefixSnapshot {
                    config: self.config.clone(),
                    layers: row_layers,
                    length,
                })
            })
            .collect()
    }
    #[cfg(test)]
    pub(super) fn inject_stage_failure(&mut self, stage: Option<usize>) {
        self.executor.fail_after_stage = stage;
    }

    pub fn forward(
        &mut self,
        model: &LlamaModel,
        chunks: &[Vec<u32>],
    ) -> Result<LlamaBatchNativeExecution, LlamaNativeError> {
        if model.config() != &self.config {
            return Err(LlamaModelError::CacheConfigMismatch.into());
        }
        if chunks.len() != self.batch_size {
            return Err(LlamaModelError::BatchSize {
                expected: self.batch_size,
                actual: chunks.len(),
            }
            .into());
        }
        let graph_plan =
            model.plan_batch_with_past(chunks, &self.lengths, self.layers.as_deref())?;
        let plan = LlamaBatchNativePlan::compile(graph_plan)?;
        let execution = plan.execute(&self.executor)?;
        let layers = execution
            .native
            .layers
            .iter()
            .map(|layer| BatchLayerCache {
                keys: layer.keys.clone(),
                values: layer.values.clone(),
            })
            .collect();
        self.layers = Some(layers);
        self.lengths = plan.next_lengths;
        Ok(execution)
    }

    pub(super) fn forward_with_executor(
        &mut self,
        model: &LlamaModel,
        chunks: &[Vec<u32>],
        executor: &LlamaNativeExecutor,
    ) -> Result<LlamaBatchNativeExecution, LlamaNativeError> {
        if model.config() != &self.config {
            return Err(LlamaModelError::CacheConfigMismatch.into());
        }
        if chunks.len() != self.batch_size {
            return Err(LlamaModelError::BatchSize {
                expected: self.batch_size,
                actual: chunks.len(),
            }
            .into());
        }
        let graph_plan =
            model.plan_batch_with_past(chunks, &self.lengths, self.layers.as_deref())?;
        let plan = LlamaBatchNativePlan::compile(graph_plan)?;
        let execution = plan.execute(executor)?;
        self.layers = Some(
            execution
                .native
                .layers
                .iter()
                .map(|layer| BatchLayerCache {
                    keys: layer.keys.clone(),
                    values: layer.values.clone(),
                })
                .collect(),
        );
        self.lengths = plan.next_lengths;
        Ok(execution)
    }
}

impl LlamaNativeCache {
    pub fn new(config: LlamaModelConfig) -> Self {
        Self {
            config,
            layers: None,
            executor: LlamaNativeExecutor::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.layers
            .as_ref()
            .map_or(0, |layers| layers[0].keys.shape().dims()[1])
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_none()
    }
    pub fn clear(&mut self) {
        self.layers = None;
    }
    pub fn compile_cache_len(&self) -> usize {
        self.executor.compile_cache_len()
    }
    #[cfg(test)]
    pub(super) fn inject_stage_failure(&mut self, stage: Option<usize>) {
        self.executor.fail_after_stage = stage;
    }
    pub fn forward(
        &mut self,
        model: &LlamaModel,
        tokens: &[u32],
    ) -> Result<LlamaNativeExecution, LlamaNativeError> {
        if model.config() != &self.config {
            return Err(LlamaNativeError::Model(
                LlamaModelError::CacheConfigMismatch,
            ));
        }
        let graph_plan = model.plan_with_past(tokens, self.layers.as_deref())?;
        let plan = LlamaNativePlan::compile(graph_plan)?;
        let execution = plan.execute(&self.executor)?;
        self.layers = Some(execution.layers.clone());
        Ok(execution)
    }
}

impl LlamaModel {
    /// Captures every supported fixed-shape stage into serialized native artifacts.
    pub fn plan_native(&self, tokens: &[u32]) -> Result<LlamaNativePlan, LlamaNativeError> {
        LlamaNativePlan::compile(self.plan(tokens)?)
    }

    /// Executes an uncached sequence using strict native schedule replay.
    pub fn forward_native(&self, tokens: &[u32]) -> Result<LlamaNativeExecution, LlamaNativeError> {
        self.plan_native(tokens)?
            .execute(&LlamaNativeExecutor::new())
    }

    /// Captures a concrete right-padded fixed batch into strict native artifacts.
    pub fn plan_batch_native(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<LlamaBatchNativePlan, LlamaNativeError> {
        LlamaBatchNativePlan::compile(self.plan_batch(sequences)?)
    }
}

fn reachable_nodes(graph: &Graph, outputs: &[NodeId]) -> Result<BTreeSet<usize>, LlamaNativeError> {
    fn visit(
        graph: &Graph,
        node: NodeId,
        out: &mut BTreeSet<usize>,
    ) -> Result<(), LlamaNativeError> {
        if !out.insert(node.index()) {
            return Ok(());
        }
        for input in op_inputs(graph.op(node)?) {
            visit(graph, input, out)?;
        }
        Ok(())
    }
    let mut out = BTreeSet::new();
    for &node in outputs {
        visit(graph, node, &mut out)?;
    }
    Ok(out)
}

fn schedulable(op: &Op) -> bool {
    matches!(
        op,
        Op::Cast { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Compare { .. }
            | Op::Logical { .. }
            | Op::Select { .. }
            | Op::Reduce { .. }
            | Op::Shrink { .. }
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::Scatter { .. }
            | Op::Matmul { .. }
    )
}

type NativeStageGraph = (Graph, NodeId, Vec<(String, NodeId)>);

fn native_stage_graph(
    source: &Graph,
    original: NodeId,
    op: &Op,
) -> Result<NativeStageGraph, LlamaNativeError> {
    let originals = op_inputs(op);
    let mut graph = Graph::new();
    let mut inputs = Vec::with_capacity(originals.len());
    let mut local = Vec::with_capacity(originals.len());
    for (slot, input) in originals.iter().copied().enumerate() {
        let name = format!("llama.stage.{}.{}", original.index(), slot);
        local.push(graph.input_dtype(&name, source.shape(input)?.clone(), source.dtype(input)?));
        inputs.push((name, input));
    }
    let mapped = match op {
        Op::Cast { dtype, .. } => Op::Cast {
            input: local[0],
            dtype: *dtype,
        },
        Op::Unary { op, .. } => Op::Unary {
            op: *op,
            input: local[0],
        },
        Op::Binary { op, .. } => Op::Binary {
            op: *op,
            lhs: local[0],
            rhs: local[1],
        },
        Op::Compare { op, .. } => Op::Compare {
            op: *op,
            lhs: local[0],
            rhs: local[1],
        },
        Op::Logical { op, rhs, .. } => Op::Logical {
            op: *op,
            lhs: local[0],
            rhs: rhs.map(|_| local[1]),
        },
        Op::Select { .. } => Op::Select {
            condition: local[0],
            on_true: local[1],
            on_false: local[2],
        },
        Op::Reduce {
            kind,
            axes,
            keepdim,
            ..
        } => Op::Reduce {
            input: local[0],
            kind: *kind,
            axes: axes.clone(),
            keepdim: *keepdim,
        },
        Op::Shrink { bounds, .. } => Op::Shrink {
            input: local[0],
            bounds: bounds.clone(),
        },
        Op::Concat { axis, .. } => Op::Concat {
            inputs: local,
            axis: *axis,
        },
        Op::Gather { axis, .. } => Op::Gather {
            input: local[0],
            index: local[1],
            axis: *axis,
        },
        Op::Scatter { axis, add, .. } => Op::Scatter {
            base: local[0],
            index: local[1],
            updates: local[2],
            axis: *axis,
            add: *add,
        },
        Op::Matmul { .. } => Op::Matmul {
            lhs: local[0],
            rhs: local[1],
        },
        _ => {
            return Err(LlamaNativeError::UnsupportedOperation {
                node: original.index(),
                operation: op_name(op),
            });
        }
    };
    let output = NodeId::from_index(graph.nodes.len());
    graph.nodes.push(crate::ir::Node {
        op: mapped,
        shape: source.shape(original)?.clone(),
        dtype: source.dtype(original)?,
        requires_grad: false,
    });
    Ok((graph, output, inputs))
}

fn movement_name(op: &Op) -> Option<&'static str> {
    Some(match op {
        Op::Reshape { .. } => "reshape",
        Op::Permute { .. } => "permute",
        Op::Expand { .. } => "expand",
        _ => return None,
    })
}

fn op_name(op: &Op) -> &'static str {
    movement_name(op).unwrap_or(match op {
        Op::Input { .. } => "input",
        Op::Constant(_) => "constant",
        Op::Random { .. } => "random",
        Op::RandomPermutation { .. } => "random_permutation",
        Op::Detach { .. } => "detach",
        _ => "unsupported",
    })
}

fn op_inputs(op: &Op) -> Vec<NodeId> {
    match op {
        Op::Input { .. } | Op::Constant(_) | Op::Random { .. } | Op::RandomPermutation { .. } => {
            vec![]
        }
        Op::Cast { input, .. }
        | Op::Detach { input }
        | Op::Unary { input, .. }
        | Op::Reduce { input, .. }
        | Op::Reshape { input, .. }
        | Op::Permute { input, .. }
        | Op::Expand { input, .. }
        | Op::Shrink { input, .. } => vec![*input],
        Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } | Op::Matmul { lhs, rhs } => {
            vec![*lhs, *rhs]
        }
        Op::Logical { lhs, rhs, .. } => std::iter::once(*lhs).chain(rhs.iter().copied()).collect(),
        Op::Select {
            condition,
            on_true,
            on_false,
        } => vec![*condition, *on_true, *on_false],
        Op::Concat { inputs, .. } => inputs.clone(),
        Op::Gather { input, index, .. } => vec![*input, *index],
        Op::Scatter {
            base,
            index,
            updates,
            ..
        } => vec![*base, *index, *updates],
        _ => vec![],
    }
}

fn execute_movement(
    graph: &Graph,
    node: NodeId,
    values: &BTreeMap<usize, TensorData>,
) -> Result<TensorData, LlamaNativeError> {
    let op = graph.op(node)?;
    let inputs = op_inputs(op)
        .into_iter()
        .map(|input| {
            values
                .get(&input.index())
                .cloned()
                .ok_or(LlamaNativeError::MissingStageValue(input.index()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut mini = Graph::new();
    let ids = inputs
        .into_iter()
        .map(|value| mini.constant(value))
        .collect::<Vec<_>>();
    let output = match op {
        Op::Reshape { shape, .. } => mini.reshape(ids[0], shape.clone())?,
        Op::Permute { axes, .. } => mini.permute(ids[0], axes.clone())?,
        Op::Expand { shape, .. } => mini.expand(ids[0], shape.clone())?,
        Op::Concat { axis, .. } => mini.concat(ids, *axis)?,
        Op::Gather { axis, .. } => mini.gather(ids[0], ids[1], *axis)?,
        Op::Scatter { axis, add, .. } => {
            if *add {
                mini.scatter_add(ids[0], ids[1], ids[2], *axis)?
            } else {
                mini.scatter(ids[0], ids[1], ids[2], *axis)?
            }
        }
        _ => {
            return Err(LlamaNativeError::UnsupportedOperation {
                node: node.index(),
                operation: op_name(op),
            });
        }
    };
    Ok(CpuBackend.execute(&mini, output, &HashMap::new())?)
}

/// Structured native planning, artifact, binding, execution, or model failure.
#[derive(Debug)]
pub enum LlamaNativeError {
    Model(LlamaModelError),
    Graph(crate::Error),
    Schedule(crate::ScheduleError),
    StageSchedule {
        node: usize,
        reason: String,
    },
    StageReplay {
        node: usize,
        reason: String,
    },
    Replay(ReplayError),
    UnsupportedOperation {
        node: usize,
        operation: &'static str,
    },
    MissingStageValue(usize),
    MissingStageOutput(usize),
    NonNativeStage(usize),
    #[cfg(test)]
    InjectedStageFailure(usize),
}

impl fmt::Display for LlamaNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama native execution error: {self:?}")
    }
}
impl error::Error for LlamaNativeError {}
impl From<LlamaModelError> for LlamaNativeError {
    fn from(value: LlamaModelError) -> Self {
        Self::Model(value)
    }
}
impl From<crate::Error> for LlamaNativeError {
    fn from(value: crate::Error) -> Self {
        Self::Graph(value)
    }
}
impl From<crate::ScheduleError> for LlamaNativeError {
    fn from(value: crate::ScheduleError) -> Self {
        Self::Schedule(value)
    }
}
impl From<ReplayError> for LlamaNativeError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}
