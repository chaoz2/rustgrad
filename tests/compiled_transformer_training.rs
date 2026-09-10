#[cfg(target_os = "macos")]
use rustgrad::MetalSessionTarget;
use rustgrad::nn::{Embedding, LayerNorm, Mode, ModeModuleForward, StateDict, StateKind};
use rustgrad::runtime::metal::{MetalCapabilities, MetalRenderer};
#[cfg(target_os = "macos")]
use rustgrad::runtime::metal::{
    MetalDeviceRunReport, MetalDiscovery, MetalRuntime, MetalScoreboardContext,
};
use rustgrad::{
    Backend, CapturedReplayExecutor, CompareOp, CompiledAdamWCheckpoint, CompiledAdamWConfig,
    CompiledAdamWFlush, CompiledAdamWFlushRuntime, CompiledAdamWPlan, CompiledAdamWRuntime,
    CompiledAdamWStep, CompiledCheckpointRuntime, CompiledDropoutConfig, CompiledDropoutKey,
    CompiledEvaluation, CompiledEvaluationRuntime, CompiledInputBatch, CompiledInputSpec,
    CompiledModuleAdamWPlan, CompiledMultiStepLr, CompiledTrainingRuntime, CompiledTrainingStep,
    CpuBackend, CpuNonFinitePolicy, CpuSessionTarget, DType, Error, Graph, LossOptions,
    MetalCompiledAdamWPlan, Module, NativeCpuSessionTarget, NativeTrainingReport,
    NativeTrainingScoreboard, NodeId, Op, Parameter, Reduction, Result, Scalar, Shape, TensorData,
    TrainingDropoutProvider, TransformerBlock, cross_entropy, load_safetensors, save_safetensors,
};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::{env, fs::OpenOptions, io::Write, path::PathBuf};

const VOCAB: usize = 3;
const EMBEDDING: usize = 2;
const BATCH: usize = 2;
const TIME: usize = 3;
const TOKEN_COUNT: usize = BATCH * TIME;
const ACCUMULATION_STEPS: u64 = 3;
const MAX_GRADIENT_NORM: f32 = 0.25;
const LOSS_MASK: &str = "loss_mask";
const WEIGHT_DECAY_EXCLUSIONS: [&str; 12] = [
    "block.ff1.1",
    "block.ff2.1",
    "block.key.1",
    "block.ln1.0",
    "block.ln1.1",
    "block.ln2.0",
    "block.ln2.1",
    "block.out.1",
    "block.query.1",
    "block.value.1",
    "norm.bias",
    "norm.weight",
];

struct TinyCausalTransformer {
    tokens: Embedding,
    block: TransformerBlock,
    norm: LayerNorm,
    frozen_scale: Parameter,
}

impl TinyCausalTransformer {
    fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            tokens: Embedding::new_static(VOCAB, EMBEDDING, None, seed)?,
            block: TransformerBlock::new_static(EMBEDDING, 1, 4, true, 0.25, seed.wrapping_add(1))?
                .with_causal_attention(true),
            norm: LayerNorm::new_static([EMBEDDING], 1e-5, true)?,
            frozen_scale: Parameter::new(TensorData::scalar(1.0), false),
        })
    }

    fn forward(
        &self,
        graph: &mut Graph,
        tokens: NodeId,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<NodeId> {
        let hidden = self.tokens.forward(graph, tokens)?;
        let hidden = self
            .block
            .forward_training_with_dropout(graph, hidden, dropout)?;
        self.project_logits(graph, hidden)
    }

    fn forward_eval(&self, graph: &mut Graph, tokens: NodeId) -> Result<NodeId> {
        let hidden = self.tokens.forward(graph, tokens)?;
        let hidden = self.block.forward_mode(graph, hidden, Mode::Eval)?.output;
        self.project_logits(graph, hidden)
    }

    fn project_logits(&self, graph: &mut Graph, hidden: NodeId) -> Result<NodeId> {
        let hidden = self.norm.forward(graph, hidden)?;
        let tied_weight = self.tokens.weight.bind(graph)?;
        let tied_weight = graph.permute(tied_weight, [1, 0])?;
        let logits = graph.matmul(hidden, tied_weight)?;
        let frozen_scale = self.frozen_scale.bind(graph)?;
        graph.mul(logits, frozen_scale)
    }
}

impl Module for TinyCausalTransformer {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        let child = |name: &str| {
            if prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{prefix}.{name}")
            }
        };
        self.tokens.visit(&child("tokens"), visitor);
        self.block.visit(&child("block"), visitor);
        self.norm.visit(&child("norm"), visitor);
        visitor(
            child("frozen_scale"),
            &self.frozen_scale,
            StateKind::Parameter,
        );
        visitor(
            child("lm_head.weight"),
            &self.tokens.weight,
            StateKind::Parameter,
        );
    }
}

struct BufferedTinyCausalTransformer {
    transformer: TinyCausalTransformer,
    running_marker: Parameter,
}

impl BufferedTinyCausalTransformer {
    fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            transformer: TinyCausalTransformer::new(seed)?,
            running_marker: Parameter::new(TensorData::scalar(3.0), false),
        })
    }
}

impl Module for BufferedTinyCausalTransformer {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.transformer.visit(prefix, visitor);
        let name = if prefix.is_empty() {
            "running_marker".to_owned()
        } else {
            format!("{prefix}.running_marker")
        };
        visitor(name, &self.running_marker, StateKind::Buffer);
    }
}

fn config() -> CompiledAdamWConfig {
    config_with_max_gradient_norm(Some(MAX_GRADIENT_NORM))
}

#[cfg(target_os = "macos")]
fn frozen_embedding_config() -> CompiledAdamWConfig {
    config().with_frozen_parameters(["tokens.weight"]).unwrap()
}

fn config_with_max_gradient_norm(max_gradient_norm: Option<f32>) -> CompiledAdamWConfig {
    optimizer_config(max_gradient_norm)
        .with_host_token_input("tokens", [BATCH, TIME])
        .unwrap()
        .with_host_token_input("targets", [BATCH, TIME])
        .unwrap()
}

fn optimizer_config(max_gradient_norm: Option<f32>) -> CompiledAdamWConfig {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)
        .unwrap()
        .with_weight_decay_exclusions(WEIGHT_DECAY_EXCLUSIONS)
        .unwrap()
        .with_loss_scale(128.0)
        .unwrap()
        .with_gradient_accumulation(ACCUMULATION_STEPS)
        .unwrap();
    match max_gradient_norm {
        Some(max_gradient_norm) => config.with_max_gradient_norm(max_gradient_norm).unwrap(),
        None => config,
    }
}

fn masked_config() -> CompiledAdamWConfig {
    optimizer_config(Some(MAX_GRADIENT_NORM))
        .with_input_batch::<MaskedTransformerBatch>()
        .unwrap()
}

fn sparse_causal_loss(graph: &mut Graph, logits: NodeId, targets: NodeId) -> Result<NodeId> {
    let flat_logits = graph.reshape(logits, [TOKEN_COUNT, VOCAB])?;
    let log_probabilities = graph.log_softmax(flat_logits, 1, None)?;
    let target_indices = graph.reshape(targets, [TOKEN_COUNT, 1])?;
    let selected = graph.gather(log_probabilities, target_indices, 1)?;
    let selected = graph.reshape(selected, [TOKEN_COUNT])?;
    let losses = graph.neg(selected)?;
    graph.mean_default(losses)
}

fn masked_sparse_causal_loss(
    graph: &mut Graph,
    logits: NodeId,
    targets: NodeId,
    loss_mask: NodeId,
) -> Result<NodeId> {
    let flat_logits = graph.reshape(logits, [TOKEN_COUNT, VOCAB])?;
    let log_probabilities = graph.log_softmax(flat_logits, 1, None)?;
    let target_indices = graph.reshape(targets, [TOKEN_COUNT, 1])?;
    let selected = graph.gather(log_probabilities, target_indices, 1)?;
    let selected = graph.reshape(selected, [TOKEN_COUNT])?;
    let losses = graph.neg(selected)?;
    let loss_mask = graph.reshape(loss_mask, [TOKEN_COUNT])?;
    let weighted = graph.mul(losses, loss_mask)?;
    let numerator = graph.sum_default(weighted)?;
    let denominator = graph.sum_default(loss_mask)?;
    graph.div(numerator, denominator)
}

fn build(
    model: &TinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let logits = model.forward(graph, inputs["tokens"], dropout)?;
    let loss = sparse_causal_loss(graph, logits, inputs["targets"])?;
    Ok((loss, BTreeMap::new()))
}

fn build_buffered(
    model: &BufferedTinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let mut observed = ObservedResidualDropout {
        inner: dropout,
        sites: Vec::new(),
    };
    let logits = model
        .transformer
        .forward(graph, inputs["tokens"], &mut observed)?;
    assert_eq!(observed.sites.len(), 2);
    let loss = masked_sparse_causal_loss(graph, logits, inputs["targets"], inputs[LOSS_MASK])?;
    Ok((
        loss,
        BTreeMap::from([
            ("dropout_0_input".into(), observed.sites[0].0),
            ("dropout_0_output".into(), observed.sites[0].1),
            ("dropout_1_input".into(), observed.sites[1].0),
            ("dropout_1_output".into(), observed.sites[1].1),
        ]),
    ))
}

fn build_with_dropout_observations(
    model: &TinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let mut observed = ObservedResidualDropout {
        inner: dropout,
        sites: Vec::new(),
    };
    let logits = model.forward(graph, inputs["tokens"], &mut observed)?;
    assert_eq!(observed.sites.len(), 2);
    let loss = sparse_causal_loss(graph, logits, inputs["targets"])?;
    Ok((
        loss,
        BTreeMap::from([
            ("dropout_0_input".into(), observed.sites[0].0),
            ("dropout_0_output".into(), observed.sites[0].1),
            ("dropout_1_input".into(), observed.sites[1].0),
            ("dropout_1_output".into(), observed.sites[1].1),
        ]),
    ))
}

fn build_evaluation(
    model: &TinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let logits = model.forward_eval(graph, inputs["tokens"])?;
    let loss = sparse_causal_loss(graph, logits, inputs["targets"])?;
    Ok((loss, BTreeMap::from([("logits".into(), logits)])))
}

fn dropout_config() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([0x1234_5678, 0x9abc_def0]))
}

fn batch_values(replay: u64) -> ([i32; TOKEN_COUNT], [i32; TOKEN_COUNT]) {
    match (replay - 1) % ACCUMULATION_STEPS {
        0 => ([0, 1, 2, 2, 0, 1], [1, 2, 0, 0, 1, 2]),
        1 => ([1, 2, 0, 0, 1, 2], [2, 0, 1, 1, 2, 0]),
        2 => ([2, 0, 1, 1, 2, 0], [0, 1, 2, 2, 0, 1]),
        _ => unreachable!(),
    }
}

fn token_tensor(values: [i32; TOKEN_COUNT]) -> TensorData {
    TensorData::from_scalars(
        Shape::new([BATCH, TIME]),
        DType::I32,
        values.into_iter().map(|value| Scalar::I(i64::from(value))),
    )
    .unwrap()
}

fn batch(replay: u64) -> BTreeMap<String, TensorData> {
    let (tokens, targets) = batch_values(replay);
    BTreeMap::from([
        ("tokens".into(), token_tensor(tokens)),
        ("targets".into(), token_tensor(targets)),
    ])
}

struct MaskedTransformerBatch {
    tokens: TensorData,
    targets: TensorData,
    loss_mask: TensorData,
}

impl MaskedTransformerBatch {
    const SCHEMA: [CompiledInputSpec; 3] = [
        CompiledInputSpec::new(LOSS_MASK, &[BATCH, TIME], DType::F32),
        CompiledInputSpec::host_token("targets", &[BATCH, TIME]),
        CompiledInputSpec::host_token("tokens", &[BATCH, TIME]),
    ];

    fn right_padded(replay: u64, valid_lengths: [usize; BATCH]) -> Self {
        let (mut tokens, mut targets) = batch_values(replay);
        let mut loss_mask = [0.0; TOKEN_COUNT];
        for (row, valid) in valid_lengths.into_iter().enumerate() {
            for column in 0..TIME {
                let index = row * TIME + column;
                if column < valid {
                    loss_mask[index] = 1.0;
                } else {
                    tokens[index] = 0;
                    targets[index] = 0;
                }
            }
        }
        Self::from_values(tokens, targets, loss_mask)
    }

    fn from_values(
        tokens: [i32; TOKEN_COUNT],
        targets: [i32; TOKEN_COUNT],
        loss_mask: [f32; TOKEN_COUNT],
    ) -> Self {
        Self {
            tokens: token_tensor(tokens),
            targets: token_tensor(targets),
            loss_mask: TensorData::new([BATCH, TIME], loss_mask.to_vec()).unwrap(),
        }
    }

    fn validate(&self) -> Result<()> {
        let expected_shape = Shape::new([BATCH, TIME]);
        if self.tokens.shape() != &expected_shape
            || self.tokens.dtype() != DType::I32
            || self.targets.shape() != &expected_shape
            || self.targets.dtype() != DType::I32
            || self.loss_mask.shape() != &expected_shape
            || self.loss_mask.dtype() != DType::F32
        {
            return Err(masked_batch_error(
                "masked causal batch descriptor mismatch",
            ));
        }
        let mut any_valid = false;
        for row in 0..BATCH {
            let mut reached_padding = false;
            for column in 0..TIME {
                let index = row * TIME + column;
                let token = self.tokens.scalar_at(index).as_i64();
                let target = self.targets.scalar_at(index).as_i64();
                if !(0..VOCAB as i64).contains(&token) || !(0..VOCAB as i64).contains(&target) {
                    return Err(masked_batch_error(
                        "masked causal batch tokens and dummy targets must be legal vocabulary indices",
                    ));
                }
                let mask = self.loss_mask.scalar_at(index).as_f64();
                if !mask.is_finite() || (mask != 0.0 && mask != 1.0) {
                    return Err(masked_batch_error(
                        "masked causal batch loss mask must contain finite binary values",
                    ));
                }
                if mask == 0.0 {
                    reached_padding = true;
                } else if reached_padding {
                    return Err(masked_batch_error(
                        "masked causal batch loss mask must describe right padding",
                    ));
                } else {
                    any_valid = true;
                }
            }
        }
        if !any_valid {
            return Err(masked_batch_error(
                "masked causal batch must contain at least one valid target",
            ));
        }
        Ok(())
    }
}

impl CompiledInputBatch for MaskedTransformerBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        self.validate()?;
        Ok(BTreeMap::from([
            (LOSS_MASK.into(), self.loss_mask),
            ("targets".into(), self.targets),
            ("tokens".into(), self.tokens),
        ]))
    }
}

fn masked_batch_error(reason: &str) -> Error {
    Error::SessionTraining {
        reason: reason.into(),
    }
}

fn masked_batch(replay: u64) -> MaskedTransformerBatch {
    const VALID_LENGTHS: [[usize; BATCH]; 3] = [[3, 2], [2, 1], [3, 0]];
    MaskedTransformerBatch::right_padded(
        replay,
        VALID_LENGTHS[((replay - 1) % ACCUMULATION_STEPS) as usize],
    )
}

fn invalid_masked_batches() -> Vec<MaskedTransformerBatch> {
    let (tokens, targets) = batch_values(1);
    let mut batches = vec![
        MaskedTransformerBatch::from_values(tokens, targets, [f32::NAN, 1.0, 0.0, 1.0, 0.0, 0.0]),
        MaskedTransformerBatch::from_values(tokens, targets, [0.5, 1.0, 0.0, 1.0, 0.0, 0.0]),
        MaskedTransformerBatch::from_values(tokens, targets, [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        MaskedTransformerBatch::from_values(
            tokens,
            [1, 2, 3, 0, 0, 0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ),
        MaskedTransformerBatch::from_values(tokens, targets, [0.0; TOKEN_COUNT]),
    ];
    batches.push(MaskedTransformerBatch {
        tokens: token_tensor(tokens),
        targets: token_tensor(targets),
        loss_mask: TensorData::scalar(1.0),
    });
    batches.push(MaskedTransformerBatch {
        tokens: token_tensor(tokens),
        targets: token_tensor(targets),
        loss_mask: token_tensor([1; TOKEN_COUNT]),
    });
    batches
}

fn learning_rate() -> TensorData {
    TensorData::scalar(0.05)
}

fn checkpoint_dropout_block_counter(checkpoint: &CompiledAdamWCheckpoint) -> u64 {
    checkpoint
        .info()
        .dropout_block_counter()
        .expect("compiled Transformer checkpoints retain dropout state")
}

#[cfg(target_os = "macos")]
fn assert_frozen_embedding_checkpoint_inventory(
    checkpoint: &CompiledAdamWCheckpoint,
    expected_parameters: &BTreeMap<String, TensorData>,
    expected_dropout_block_counter: u64,
) {
    const EXPECTED_NAMES: [&str; 18] = [
        "block.ff1.0",
        "block.ff1.1",
        "block.ff2.0",
        "block.ff2.1",
        "block.key.0",
        "block.key.1",
        "block.ln1.0",
        "block.ln1.1",
        "block.ln2.0",
        "block.ln2.1",
        "block.out.0",
        "block.out.1",
        "block.query.0",
        "block.query.1",
        "block.value.0",
        "block.value.1",
        "norm.bias",
        "norm.weight",
    ];
    let expected_names = EXPECTED_NAMES.map(str::to_owned).to_vec();
    assert_eq!(
        expected_parameters.keys().cloned().collect::<Vec<_>>(),
        expected_names
    );
    let (state, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    let parameter_names = serde_json::from_str::<Vec<String>>(&metadata["parameter_names"])
        .expect("checkpoint parameter names must be valid JSON");
    assert_eq!(parameter_names, expected_names);
    assert!(
        !parameter_names
            .iter()
            .any(|name| { matches!(name.as_str(), "tokens.weight" | "lm_head.weight") })
    );

    let mut expected_tensor_names = BTreeSet::from(["dropout_block_counter".to_owned()]);
    for (ordinal, name) in parameter_names.iter().enumerate() {
        let parameter = &expected_parameters[name];
        for family in [
            "parameter",
            "first_moment",
            "second_moment",
            "gradient_accumulator",
        ] {
            let tensor_name = format!("{family}.{ordinal}");
            let tensor = &state[tensor_name.as_str()];
            assert_eq!(tensor.shape(), parameter.shape(), "{tensor_name} shape");
            assert_eq!(tensor.dtype(), parameter.dtype(), "{tensor_name} dtype");
            assert!(expected_tensor_names.insert(tensor_name));
        }
    }
    assert_eq!(
        state.keys().cloned().collect::<BTreeSet<_>>(),
        expected_tensor_names
    );
    let dropout = &state["dropout_block_counter"];
    assert_eq!(dropout.shape(), &Shape::new([]));
    assert_eq!(dropout.dtype(), DType::U64);
    assert_eq!(
        dropout.scalar_at(0).as_u64(),
        expected_dropout_block_counter
    );
}

#[test]
fn sparse_causal_loss_matches_dense_reference_and_analytic_gradient() {
    let logits_values = vec![
        2.0, 1.0, -0.5, -1.0, 0.5, 1.5, 0.25, -0.75, 1.25, 1.75, -0.25, 0.5, -0.5, 2.25, 0.75, 0.0,
        1.0, -1.0,
    ];
    let target_values: [i32; 6] = [0, 2, 1, 0, 1, 2];
    let logits = TensorData::new([BATCH, TIME, VOCAB], logits_values.clone()).unwrap();
    let targets = TensorData::from_scalars(
        [BATCH, TIME],
        DType::I32,
        target_values
            .into_iter()
            .map(|target| Scalar::I(i64::from(target))),
    )
    .unwrap();
    let bindings = HashMap::from([
        ("logits".into(), logits.clone()),
        ("targets".into(), targets.clone()),
    ]);

    let mut sparse_graph = Graph::new();
    let sparse_logits =
        sparse_graph.input_dtype_requires_grad("logits", [BATCH, TIME, VOCAB], DType::F32, true);
    let sparse_targets = sparse_graph.input_dtype("targets", [BATCH, TIME], DType::I32);
    let sparse_loss = sparse_causal_loss(&mut sparse_graph, sparse_logits, sparse_targets).unwrap();
    let sparse_gradient = sparse_graph.grad(sparse_loss, sparse_logits).unwrap();
    let sparse = CpuBackend
        .execute_many(&sparse_graph, &[sparse_loss, sparse_gradient], &bindings)
        .unwrap();

    let mut dense_graph = Graph::new();
    let dense_logits =
        dense_graph.input_dtype_requires_grad("logits", [BATCH, TIME, VOCAB], DType::F32, true);
    let dense_targets = dense_graph.input_dtype("targets", [BATCH, TIME], DType::I32);
    let dense_logits = dense_graph
        .reshape(dense_logits, [TOKEN_COUNT, VOCAB])
        .unwrap();
    let dense_targets = dense_graph.reshape(dense_targets, [TOKEN_COUNT]).unwrap();
    let dense_loss = cross_entropy(
        &mut dense_graph,
        dense_logits,
        dense_targets,
        LossOptions {
            reduction: Reduction::Mean,
            ..LossOptions::default()
        },
    )
    .unwrap();
    let dense_gradient = dense_graph.grad(dense_loss, dense_logits).unwrap();
    let dense = CpuBackend
        .execute_many(&dense_graph, &[dense_loss, dense_gradient], &bindings)
        .unwrap();

    assert!(
        (sparse.outputs[0].scalar_at(0).as_f64() - dense.outputs[0].scalar_at(0).as_f64()).abs()
            < 1e-6
    );
    for index in 0..TOKEN_COUNT * VOCAB {
        assert!(
            (sparse.outputs[1].scalar_at(index).as_f64()
                - dense.outputs[1].scalar_at(index).as_f64())
            .abs()
                < 1e-6
        );
    }

    let mut expected_loss = 0.0f64;
    for (row, target) in target_values.into_iter().enumerate() {
        let row_logits = &logits_values[row * VOCAB..(row + 1) * VOCAB];
        let maximum = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let denominator = row_logits
            .iter()
            .map(|value| (f64::from(*value) - maximum).exp())
            .sum::<f64>();
        expected_loss -= f64::from(row_logits[target as usize]) - maximum - denominator.ln();
        for (class, value) in row_logits.iter().enumerate() {
            let probability = (f64::from(*value) - maximum).exp() / denominator;
            let selected = if class == target as usize { 1.0 } else { 0.0 };
            let expected = (probability - selected) / TOKEN_COUNT as f64;
            let actual = sparse.outputs[1].scalar_at(row * VOCAB + class).as_f64();
            assert!((actual - expected).abs() < 1e-5);
        }
    }
    expected_loss /= TOKEN_COUNT as f64;
    assert!((sparse.outputs[0].scalar_at(0).as_f64() - expected_loss).abs() < 1e-5);
}

struct FixedResidualDropout {
    masks: [TensorData; 2],
    next: usize,
}

impl FixedResidualDropout {
    fn new() -> Self {
        let mask = |values: [bool; BATCH * TIME * EMBEDDING]| {
            TensorData::from_scalars(
                [BATCH, TIME, EMBEDDING],
                DType::Bool,
                values.into_iter().map(Scalar::Bool),
            )
            .unwrap()
        };
        Self::from_masks([
            mask([
                true, false, true, true, false, true, false, true, true, false, true, false,
            ]),
            mask([
                false, true, true, false, true, true, true, true, false, true, false, true,
            ]),
        ])
    }

    fn from_masks(masks: [TensorData; 2]) -> Self {
        Self { masks, next: 0 }
    }
}

struct ObservedResidualDropout<'a> {
    inner: &'a mut dyn TrainingDropoutProvider,
    sites: Vec<(NodeId, NodeId)>,
}

impl TrainingDropoutProvider for ObservedResidualDropout<'_> {
    fn dropout(&mut self, graph: &mut Graph, input: NodeId, probability: f64) -> Result<NodeId> {
        let output = self.inner.dropout(graph, input, probability)?;
        self.sites.push((input, output));
        Ok(output)
    }
}

impl TrainingDropoutProvider for FixedResidualDropout {
    fn dropout(&mut self, graph: &mut Graph, input: NodeId, probability: f64) -> Result<NodeId> {
        assert_eq!(probability.to_bits(), 0.25f64.to_bits());
        assert_eq!(graph.dtype(input)?, DType::F32);
        assert_eq!(graph.shape(input)?, &Shape::new([BATCH, TIME, EMBEDDING]));
        let mask = self
            .masks
            .get(self.next)
            .expect("the maintained block has exactly two residual-dropout sites")
            .clone();
        self.next += 1;
        let mask = graph.constant(mask);
        let mask = graph.contiguous(mask)?;
        let zero = graph.zeros_with_dtype(Shape::new([]), DType::F32)?;
        let kept = graph.select(mask, input, zero)?;
        let denominator = graph.constant(TensorData::scalar(0.75));
        graph.div(kept, denominator)
    }
}

fn module_parameter_state(model: &TinyCausalTransformer) -> BTreeMap<String, (TensorData, u64)> {
    let mut state = BTreeMap::new();
    let mut error = None;
    model.visit("", &mut |name, parameter, _| match parameter.snapshot() {
        Ok(snapshot) => {
            state.insert(name, (snapshot.data, snapshot.version));
        }
        Err(snapshot_error) => error = Some(snapshot_error),
    });
    if let Some(error) = error {
        panic!("the maintained Transformer state must remain readable: {error}");
    }
    state
}

fn observed_dropout_masks(outputs: &BTreeMap<String, TensorData>) -> [TensorData; 2] {
    [0, 1].map(|site| {
        let input = &outputs[&format!("dropout_{site}_input")];
        let output = &outputs[&format!("dropout_{site}_output")];
        assert_eq!(input.shape(), output.shape());
        TensorData::from_scalars(
            input.shape().clone(),
            DType::Bool,
            (0..input.len()).map(|coordinate| {
                let input = input.scalar_at(coordinate).as_f64();
                let output = output.scalar_at(coordinate).as_f64();
                assert_ne!(input, 0.0, "dropout mask observation must be unambiguous");
                let kept = output != 0.0;
                if kept {
                    assert!((output * 0.75 - input).abs() < 1e-6);
                }
                Scalar::Bool(kept)
            }),
        )
        .unwrap()
    })
}

fn maintained_transformer_analytic_gradients(
    model: &TinyCausalTransformer,
    inputs: BTreeMap<String, TensorData>,
    masks: [TensorData; 2],
) -> BTreeMap<String, TensorData> {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [BATCH, TIME], DType::I32);
    let targets = graph.input_dtype("targets", [BATCH, TIME], DType::I32);
    let mut dropout = FixedResidualDropout::from_masks(masks);
    let logits = model.forward(&mut graph, tokens, &mut dropout).unwrap();
    assert_eq!(dropout.next, 2);
    let loss = sparse_causal_loss(&mut graph, logits, targets).unwrap();
    let trainable = model.trainable_parameters().unwrap();
    let targets = trainable
        .iter()
        .map(|(_, parameter)| parameter.node(&graph).unwrap())
        .collect::<Vec<_>>();
    let gradients = graph.gradient_default(loss, &targets).unwrap();
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.extend(inputs);
    let realized = CpuBackend
        .execute_many(&graph, &gradients, &bindings)
        .unwrap();
    trainable
        .into_iter()
        .map(|(name, _)| name)
        .zip(realized.outputs)
        .collect()
}

struct AdamWOracleState {
    parameters: BTreeMap<String, TensorData>,
    first_moments: BTreeMap<String, TensorData>,
    second_moments: BTreeMap<String, TensorData>,
}

struct AdamWOracleUpdate {
    state: AdamWOracleState,
    excluded_decay_counterfactuals: usize,
    included_decay_counterfactuals: usize,
}

fn average_gradient_window(
    gradients: &[BTreeMap<String, TensorData>],
) -> BTreeMap<String, Vec<f32>> {
    assert_eq!(gradients.len(), ACCUMULATION_STEPS as usize);
    let names = gradients[0].keys().collect::<Vec<_>>();
    for gradient in gradients.iter().skip(1) {
        assert_eq!(gradient.keys().collect::<Vec<_>>(), names);
    }
    names
        .into_iter()
        .map(|name| {
            let descriptor = &gradients[0][name];
            assert_eq!(descriptor.dtype(), DType::F32);
            for gradient in gradients.iter().skip(1) {
                assert_eq!(gradient[name].shape(), descriptor.shape());
                assert_eq!(gradient[name].dtype(), DType::F32);
            }
            let averaged = (0..descriptor.len())
                .map(|coordinate| {
                    let first = gradients[0][name].scalar_at(coordinate).as_f64() as f32;
                    let second = gradients[1][name].scalar_at(coordinate).as_f64() as f32;
                    let third = gradients[2][name].scalar_at(coordinate).as_f64() as f32;
                    // Recurrent accumulation stores each F32 sum before the
                    // next replay adds to it.
                    ((first + second) + third) / ACCUMULATION_STEPS as f32
                })
                .collect();
            (name.clone(), averaged)
        })
        .collect()
}

fn clip_gradient_window(
    averaged: &BTreeMap<String, Vec<f32>>,
    max_norm: f32,
) -> (BTreeMap<String, Vec<f32>>, f32) {
    let mut squared_norm = 0.0f32;
    // Production first reduces each canonical parameter in lane order, then
    // adds those subtotals in canonical parameter order.
    for gradient in averaged.values() {
        let parameter_squared_norm = gradient
            .iter()
            .fold(0.0f32, |subtotal, gradient| subtotal + gradient * gradient);
        squared_norm += parameter_squared_norm;
    }
    let gradient_norm = squared_norm.sqrt();
    let scale = max_norm / gradient_norm.max(max_norm);
    let clipped = averaged
        .iter()
        .map(|(name, gradient)| {
            (
                name.clone(),
                gradient.iter().map(|gradient| gradient * scale).collect(),
            )
        })
        .collect();
    (clipped, gradient_norm)
}

fn tensor_from_f32_lanes(
    template: &TensorData,
    lanes: impl IntoIterator<Item = f32>,
) -> TensorData {
    TensorData::from_scalars(
        template.shape().clone(),
        DType::F32,
        lanes.into_iter().map(|value| Scalar::F(f64::from(value))),
    )
    .unwrap()
}

fn apply_adamw_oracle_recurrence(
    previous: &AdamWOracleState,
    gradients: &BTreeMap<String, Vec<f32>>,
    optimizer: &CompiledAdamWConfig,
    optimizer_step: u64,
    learning_rate: f32,
) -> AdamWOracleUpdate {
    assert!(optimizer_step > 0);
    assert_eq!(
        previous.parameters.keys().collect::<Vec<_>>(),
        gradients.keys().collect::<Vec<_>>()
    );
    let beta1 = optimizer.beta1();
    let beta2 = optimizer.beta2();
    let first_correction = 1.0 - beta1.powf(optimizer_step as f32);
    let second_correction = 1.0 - beta2.powf(optimizer_step as f32);
    let decay_factor = 1.0 - learning_rate * optimizer.weight_decay();
    let exclusions = optimizer.weight_decay_exclusions().collect::<BTreeSet<_>>();
    let mut parameters = BTreeMap::new();
    let mut first_moments = BTreeMap::new();
    let mut second_moments = BTreeMap::new();
    let mut excluded_decay_counterfactuals = 0;
    let mut included_decay_counterfactuals = 0;

    for (name, previous_parameter) in &previous.parameters {
        let previous_first = &previous.first_moments[name];
        let previous_second = &previous.second_moments[name];
        let excluded = exclusions.contains(name.as_str());
        let mut next_parameters = Vec::with_capacity(previous_parameter.len());
        let mut next_first_moments = Vec::with_capacity(previous_parameter.len());
        let mut next_second_moments = Vec::with_capacity(previous_parameter.len());
        for (coordinate, gradient) in gradients[name].iter().copied().enumerate() {
            let retained_first = beta1 * previous_first.scalar_at(coordinate).as_f64() as f32;
            let fresh_first = (1.0 - beta1) * gradient;
            let next_first = retained_first + fresh_first;
            let retained_second = beta2 * previous_second.scalar_at(coordinate).as_f64() as f32;
            let gradient_squared = gradient * gradient;
            let fresh_second = (1.0 - beta2) * gradient_squared;
            let next_second = retained_second + fresh_second;
            let corrected_first = next_first / first_correction;
            let corrected_second = next_second / second_correction;
            let root = corrected_second.sqrt();
            let denominator = root + optimizer.eps();
            let normalized = corrected_first / denominator;
            let previous_parameter = previous_parameter.scalar_at(coordinate).as_f64() as f32;
            let decayed = previous_parameter * decay_factor;
            let scaled = learning_rate * normalized;
            let without_decay = previous_parameter - scaled;
            let with_decay = decayed - scaled;
            if with_decay != without_decay {
                if excluded {
                    excluded_decay_counterfactuals += 1;
                } else {
                    included_decay_counterfactuals += 1;
                }
            }
            next_parameters.push(if excluded { without_decay } else { with_decay });
            next_first_moments.push(next_first);
            next_second_moments.push(next_second);
        }
        parameters.insert(
            name.clone(),
            tensor_from_f32_lanes(previous_parameter, next_parameters),
        );
        first_moments.insert(
            name.clone(),
            tensor_from_f32_lanes(previous_first, next_first_moments),
        );
        second_moments.insert(
            name.clone(),
            tensor_from_f32_lanes(previous_second, next_second_moments),
        );
    }
    AdamWOracleUpdate {
        state: AdamWOracleState {
            parameters,
            first_moments,
            second_moments,
        },
        excluded_decay_counterfactuals,
        included_decay_counterfactuals,
    }
}

fn assert_adamw_oracle_state(
    window: u64,
    expected: &AdamWOracleState,
    actual_parameters: &BTreeMap<String, TensorData>,
    actual_first_moments: &BTreeMap<String, TensorData>,
    actual_second_moments: &BTreeMap<String, TensorData>,
) {
    const FIRST_MOMENT_TOLERANCE: f64 = 2e-6;
    const SECOND_MOMENT_TOLERANCE: f64 = 2e-7;
    const PARAMETER_TOLERANCE: f64 = 2e-5;

    let parameter_names = expected.parameters.keys().collect::<Vec<_>>();
    assert_eq!(
        actual_parameters.keys().collect::<Vec<_>>(),
        parameter_names
    );
    assert_eq!(
        actual_first_moments.keys().collect::<Vec<_>>(),
        parameter_names
    );
    assert_eq!(
        actual_second_moments.keys().collect::<Vec<_>>(),
        parameter_names
    );
    let mut coordinates_checked = 0;
    for (name, expected_parameter) in &expected.parameters {
        let expected_first = &expected.first_moments[name];
        let expected_second = &expected.second_moments[name];
        let actual_parameter = &actual_parameters[name];
        let actual_first = &actual_first_moments[name];
        let actual_second = &actual_second_moments[name];
        for (actual, expected) in [
            (actual_parameter, expected_parameter),
            (actual_first, expected_first),
            (actual_second, expected_second),
        ] {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.dtype(), DType::F32);
            assert_eq!(expected.dtype(), DType::F32);
        }
        for coordinate in 0..expected_parameter.len() {
            for (kind, actual, expected, tolerance) in [
                (
                    "first moment",
                    actual_first.scalar_at(coordinate).as_f64(),
                    expected_first.scalar_at(coordinate).as_f64(),
                    FIRST_MOMENT_TOLERANCE,
                ),
                (
                    "second moment",
                    actual_second.scalar_at(coordinate).as_f64(),
                    expected_second.scalar_at(coordinate).as_f64(),
                    SECOND_MOMENT_TOLERANCE,
                ),
                (
                    "parameter",
                    actual_parameter.scalar_at(coordinate).as_f64(),
                    expected_parameter.scalar_at(coordinate).as_f64(),
                    PARAMETER_TOLERANCE,
                ),
            ] {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "window {window} {name}[{coordinate}] {kind} mismatch: actual={actual}, expected={expected}"
                );
            }
            coordinates_checked += 1;
        }
    }
    assert_eq!(coordinates_checked, 64);
}

fn perturbed_parameter_bindings(
    bindings: &HashMap<String, TensorData>,
    input_name: &str,
    value: &TensorData,
    coordinate: usize,
    delta: f64,
) -> HashMap<String, TensorData> {
    let mut perturbed = bindings.clone();
    let replacement = TensorData::from_scalars(
        value.shape().clone(),
        value.dtype(),
        (0..value.len()).map(|index| {
            if index == coordinate {
                Scalar::F(value.scalar_at(index).as_f64() + delta)
            } else {
                value.scalar_at(index)
            }
        }),
    )
    .unwrap();
    assert!(perturbed.insert(input_name.into(), replacement).is_some());
    perturbed
}

fn directionally_perturbed_parameter_bindings(
    bindings: &HashMap<String, TensorData>,
    input_names: &[String],
    directions: &[TensorData],
    scale: f64,
) -> HashMap<String, TensorData> {
    assert_eq!(input_names.len(), directions.len());
    let mut perturbed = bindings.clone();
    for (input_name, direction) in input_names.iter().zip(directions) {
        let value = bindings.get(input_name).unwrap();
        assert_eq!(direction.shape(), value.shape());
        assert_eq!(direction.dtype(), value.dtype());
        let replacement = TensorData::from_scalars(
            value.shape().clone(),
            value.dtype(),
            (0..value.len()).map(|coordinate| {
                Scalar::F(
                    value.scalar_at(coordinate).as_f64()
                        + scale * direction.scalar_at(coordinate).as_f64(),
                )
            }),
        )
        .unwrap();
        assert!(perturbed.insert(input_name.clone(), replacement).is_some());
    }
    perturbed
}

fn canonical_parameter_directions(
    bindings: &HashMap<String, TensorData>,
    input_names: &[String],
) -> Vec<TensorData> {
    const VALUES: [f64; 8] = [
        -0.125, -0.09375, -0.0625, -0.03125, 0.03125, 0.0625, 0.09375, 0.125,
    ];

    let mut offset = 0;
    input_names
        .iter()
        .map(|input_name| {
            let value = bindings.get(input_name).unwrap();
            assert_eq!(value.dtype(), DType::F32);
            let direction = TensorData::from_scalars(
                value.shape().clone(),
                DType::F32,
                (0..value.len()).map(|coordinate| Scalar::F(VALUES[(offset + coordinate) % 8])),
            )
            .unwrap();
            offset += value.len();
            direction
        })
        .collect()
}

fn assert_relu_region_unchanged(base: &TensorData, perturbed: &TensorData, context: &str) {
    assert_eq!(perturbed.shape(), base.shape());
    for coordinate in 0..base.len() {
        let base = base.scalar_at(coordinate).as_f64();
        let perturbed = perturbed.scalar_at(coordinate).as_f64();
        assert!(
            (base > 0.0) == (perturbed > 0.0),
            "finite difference crossed the ReLU kink at {context}[{coordinate}]: {base} -> {perturbed}"
        );
    }
}

fn relu_region_unchanged(base: &TensorData, perturbed: &TensorData) -> bool {
    perturbed.shape() == base.shape()
        && (0..base.len()).all(|coordinate| {
            (base.scalar_at(coordinate).as_f64() > 0.0)
                == (perturbed.scalar_at(coordinate).as_f64() > 0.0)
        })
}

fn unique_relu_input(graph: &Graph, loss: NodeId) -> NodeId {
    let relu_inputs = graph
        .trace(loss)
        .unwrap()
        .steps
        .into_iter()
        .filter_map(|step| {
            let Op::Select {
                condition,
                on_true,
                on_false,
            } = graph.op(step.node).unwrap()
            else {
                return None;
            };
            let Op::Compare {
                op: CompareOp::Lt,
                lhs,
                rhs,
            } = graph.op(*condition).unwrap()
            else {
                return None;
            };
            if lhs != on_false || rhs != on_true {
                return None;
            }
            match graph.op(*lhs).unwrap() {
                Op::Constant(zero)
                    if zero.shape() == &Shape::new([])
                        && zero.dtype() == graph.dtype(*rhs).unwrap()
                        && zero.scalar_at(0).as_f64() == 0.0 =>
                {
                    Some(*rhs)
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(relu_inputs.len(), 1);
    relu_inputs[0]
}

struct MaintainedDerivativeFixture {
    model: TinyCausalTransformer,
    state_before: StateDict,
    parameter_state_before: BTreeMap<String, (TensorData, u64)>,
    graph: Graph,
    loss: NodeId,
    targets: Vec<NodeId>,
    target_input_names: Vec<String>,
    relu_input: NodeId,
    bindings: HashMap<String, TensorData>,
}

fn maintained_derivative_fixture() -> MaintainedDerivativeFixture {
    let model = TinyCausalTransformer::new(7).unwrap();
    let state_before = model.state_dict().unwrap();
    let parameter_state_before = module_parameter_state(&model);
    let mut traversal = BTreeMap::new();
    model.visit("", &mut |name, parameter, kind| {
        traversal.insert(name, (parameter.id(), kind, parameter.is_trainable()));
    });
    assert_eq!(
        traversal["tokens.weight"].0, traversal["lm_head.weight"].0,
        "the output head must be the embedding Parameter"
    );
    assert!(!traversal["frozen_scale"].2);
    assert!(state_before.tensors().contains_key("tokens.weight"));
    assert!(!state_before.tensors().contains_key("lm_head.weight"));

    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [BATCH, TIME], DType::I32);
    let target_tokens = graph.input_dtype("targets", [BATCH, TIME], DType::I32);
    let mut dropout = FixedResidualDropout::new();
    let logits = model.forward(&mut graph, tokens, &mut dropout).unwrap();
    assert_eq!(dropout.next, 2);
    let loss = sparse_causal_loss(&mut graph, logits, target_tokens).unwrap();

    let trainable = model.trainable_parameters().unwrap();
    assert_eq!(trainable.len(), 19);
    assert!(trainable.iter().any(|(name, _)| name == "tokens.weight"));
    assert!(trainable.iter().all(|(name, _)| name != "lm_head.weight"));
    assert!(trainable.iter().all(|(name, _)| name != "frozen_scale"));
    let targets = trainable
        .iter()
        .map(|(_, parameter)| parameter.node(&graph).unwrap())
        .collect::<Vec<_>>();
    let target_input_names = targets
        .iter()
        .map(|target| match graph.op(*target).unwrap() {
            Op::Input { name } => name.clone(),
            op => panic!("bound trainable target %{target} must be an input, got {op:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        targets
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        targets.len(),
        "canonical trainable parameters must map to distinct graph leaves"
    );

    let relu_input = unique_relu_input(&graph, loss);

    // Deliberately keep two feed-forward lanes active and two inactive. LayerNorm
    // bounds the two-element input, so these biases provide a wide fixed region
    // for every parameter perturbation without mutating the module itself.
    let ff1_bias_index = trainable
        .iter()
        .position(|(name, _)| name == "block.ff1.1")
        .unwrap();
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.insert(
        target_input_names[ff1_bias_index].clone(),
        TensorData::new([4], vec![2.0, -2.0, 1.5, -1.5]).unwrap(),
    );
    bindings.extend(batch(1));

    MaintainedDerivativeFixture {
        model,
        state_before,
        parameter_state_before,
        graph,
        loss,
        targets,
        target_input_names,
        relu_input,
        bindings,
    }
}

#[test]
fn maintained_causal_transformer_all_parameter_vjps_match_central_differences() {
    const EPSILON: f64 = 1e-3;
    const RELU_MARGIN: f64 = 0.5;
    const ABSOLUTE_TOLERANCE: f64 = 3e-3;
    const RELATIVE_TOLERANCE: f64 = 3e-3;

    let MaintainedDerivativeFixture {
        model,
        state_before,
        parameter_state_before,
        mut graph,
        loss,
        targets,
        target_input_names,
        relu_input,
        bindings,
    } = maintained_derivative_fixture();

    let trainable = model.trainable_parameters().unwrap();

    // This is the same one-batched-reverse entry point used by compiled AdamW.
    let gradients = graph.gradient_default(loss, &targets).unwrap();
    assert_eq!(gradients.len(), trainable.len());
    let mut analytic_outputs = Vec::with_capacity(2 + gradients.len());
    analytic_outputs.push(loss);
    analytic_outputs.extend(gradients.iter().copied());
    analytic_outputs.push(relu_input);
    let analytic = CpuBackend
        .execute_many(&graph, &analytic_outputs, &bindings)
        .unwrap();
    let base_relu = &analytic.outputs[1 + gradients.len()];
    assert!(
        base_relu
            .to_vec_f64()
            .into_iter()
            .all(|value| value.abs() >= RELU_MARGIN),
        "the finite-difference fixture must stay well away from the ReLU kink"
    );

    let finite_difference_outputs = [loss, relu_input];
    let analytic_gradients = &analytic.outputs[1..1 + gradients.len()];
    let mut coordinates_checked = 0;
    for (((name, parameter), input_name), gradient) in trainable
        .iter()
        .zip(&target_input_names)
        .zip(analytic_gradients)
    {
        let snapshot = parameter.snapshot().unwrap();
        let fixture = bindings.get(input_name).unwrap();
        assert_eq!(fixture.shape(), &snapshot.shape);
        assert_eq!(fixture.dtype(), snapshot.dtype);
        for coordinate in 0..fixture.len() {
            let plus_bindings =
                perturbed_parameter_bindings(&bindings, input_name, fixture, coordinate, EPSILON);
            let minus_bindings =
                perturbed_parameter_bindings(&bindings, input_name, fixture, coordinate, -EPSILON);
            let plus = CpuBackend
                .execute_many(&graph, &finite_difference_outputs, &plus_bindings)
                .unwrap();
            let minus = CpuBackend
                .execute_many(&graph, &finite_difference_outputs, &minus_bindings)
                .unwrap();
            assert_relu_region_unchanged(base_relu, &plus.outputs[1], &format!("{name} + epsilon"));
            assert_relu_region_unchanged(
                base_relu,
                &minus.outputs[1],
                &format!("{name} - epsilon"),
            );
            let numerical = (plus.outputs[0].scalar_at(0).as_f64()
                - minus.outputs[0].scalar_at(0).as_f64())
                / (2.0 * EPSILON);
            let analytic = gradient.scalar_at(coordinate).as_f64();
            assert!(analytic.is_finite() && numerical.is_finite());
            let error = (analytic - numerical).abs();
            let tolerance =
                ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * analytic.abs().max(numerical.abs());
            assert!(
                error <= tolerance,
                "{name}[{coordinate}] VJP mismatch: analytic={analytic}, numerical={numerical}, error={error}, tolerance={tolerance}"
            );
            coordinates_checked += 1;
        }
    }
    assert_eq!(coordinates_checked, 64);
    assert_eq!(model.state_dict().unwrap(), state_before);
    assert_eq!(module_parameter_state(&model), parameter_state_before);
}

#[test]
fn maintained_causal_transformer_all_parameter_hvps_match_central_differences() {
    const EPSILON: f64 = 1e-2;
    const RELU_MARGIN: f64 = 0.5;
    const ABSOLUTE_TOLERANCE: f64 = 1e-2;
    const RELATIVE_TOLERANCE: f64 = 1e-2;

    let MaintainedDerivativeFixture {
        model,
        state_before,
        parameter_state_before,
        mut graph,
        loss,
        targets,
        target_input_names,
        relu_input,
        bindings,
    } = maintained_derivative_fixture();
    let trainable = model.trainable_parameters().unwrap();
    let directions = canonical_parameter_directions(&bindings, &target_input_names);
    assert_eq!(directions.len(), trainable.len());
    assert_eq!(directions.iter().map(TensorData::len).sum::<usize>(), 64);
    assert!(directions.iter().all(|direction| {
        direction
            .to_vec_f64()
            .into_iter()
            .all(|value| value != 0.0 && value.abs() <= 0.125)
    }));

    // Both reverse traversals are batched over the same canonical parameter
    // leaves, matching the compilation path rather than differentiating each
    // parameter or coordinate in isolation.
    let gradients = graph.gradient_default(loss, &targets).unwrap();
    assert_eq!(gradients.len(), trainable.len());
    let mut directional_terms = Vec::with_capacity(gradients.len());
    for (gradient, direction) in gradients.iter().zip(&directions) {
        let direction = graph.constant(direction.clone());
        let weighted = graph.mul(*gradient, direction).unwrap();
        directional_terms.push(graph.sum_all(weighted).unwrap());
    }
    let mut directional_derivative = directional_terms[0];
    for term in directional_terms.into_iter().skip(1) {
        directional_derivative = graph.add(directional_derivative, term).unwrap();
    }
    let hvps = graph
        .gradient_default(directional_derivative, &targets)
        .unwrap();
    assert_eq!(hvps.len(), trainable.len());

    let mut base_outputs = Vec::with_capacity(hvps.len() + gradients.len() + 1);
    base_outputs.extend(hvps.iter().copied());
    base_outputs.extend(gradients.iter().copied());
    base_outputs.push(relu_input);
    let base = CpuBackend
        .execute_many(&graph, &base_outputs, &bindings)
        .unwrap();
    let base_relu = &base.outputs[hvps.len() + gradients.len()];
    assert!(
        base_relu
            .to_vec_f64()
            .into_iter()
            .all(|value| value.abs() >= RELU_MARGIN),
        "the HVP fixture must stay well away from the ReLU kink"
    );

    let plus_bindings = directionally_perturbed_parameter_bindings(
        &bindings,
        &target_input_names,
        &directions,
        EPSILON,
    );
    let minus_bindings = directionally_perturbed_parameter_bindings(
        &bindings,
        &target_input_names,
        &directions,
        -EPSILON,
    );
    let mut finite_difference_outputs = gradients.clone();
    finite_difference_outputs.push(relu_input);
    let plus = CpuBackend
        .execute_many(&graph, &finite_difference_outputs, &plus_bindings)
        .unwrap();
    let minus = CpuBackend
        .execute_many(&graph, &finite_difference_outputs, &minus_bindings)
        .unwrap();
    assert_relu_region_unchanged(
        base_relu,
        plus.outputs.last().unwrap(),
        "+ epsilon * direction",
    );
    assert_relu_region_unchanged(
        base_relu,
        minus.outputs.last().unwrap(),
        "- epsilon * direction",
    );

    let analytic_hvps = &base.outputs[..hvps.len()];
    let mut coordinates_checked = 0;
    for (((name, _), analytic), (plus, minus)) in trainable.iter().zip(analytic_hvps).zip(
        plus.outputs[..gradients.len()]
            .iter()
            .zip(&minus.outputs[..gradients.len()]),
    ) {
        assert_eq!(analytic.shape(), plus.shape());
        assert_eq!(analytic.shape(), minus.shape());
        for coordinate in 0..analytic.len() {
            let analytic = analytic.scalar_at(coordinate).as_f64();
            let numerical = (plus.scalar_at(coordinate).as_f64()
                - minus.scalar_at(coordinate).as_f64())
                / (2.0 * EPSILON);
            assert!(analytic.is_finite() && numerical.is_finite());
            let error = (analytic - numerical).abs();
            let tolerance =
                ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * analytic.abs().max(numerical.abs());
            assert!(
                error <= tolerance,
                "{name}[{coordinate}] HVP mismatch: analytic={analytic}, numerical={numerical}, error={error}, tolerance={tolerance}"
            );
            coordinates_checked += 1;
        }
    }
    assert_eq!(coordinates_checked, 64);
    assert_eq!(model.state_dict().unwrap(), state_before);
    assert_eq!(module_parameter_state(&model), parameter_state_before);
}

#[derive(Clone, Copy)]
struct TransformerGradientProbe {
    parameter: &'static str,
    coordinate: usize,
    boundary: &'static str,
}

fn numerical_transformer_gradient_lanes(
    model: &TinyCausalTransformer,
    inputs: BTreeMap<String, TensorData>,
    masks: [TensorData; 2],
    probes: &[TransformerGradientProbe],
) -> Vec<f64> {
    const EPSILONS: [f64; 4] = [1e-3, 5e-4, 2.5e-4, 1.25e-4];

    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [BATCH, TIME], DType::I32);
    let targets = graph.input_dtype("targets", [BATCH, TIME], DType::I32);
    let mut dropout = FixedResidualDropout::from_masks(masks);
    let logits = model.forward(&mut graph, tokens, &mut dropout).unwrap();
    assert_eq!(dropout.next, 2);
    let loss = sparse_causal_loss(&mut graph, logits, targets).unwrap();
    let relu_input = unique_relu_input(&graph, loss);

    let trainable = model.trainable_parameters().unwrap();
    let parameter_inputs = trainable
        .iter()
        .map(|(name, parameter)| {
            let node = parameter.node(&graph).unwrap();
            let Op::Input { name: input_name } = graph.op(node).unwrap() else {
                panic!("bound trainable parameter {name} must be a graph input");
            };
            (name.clone(), input_name.clone())
        })
        .collect::<BTreeMap<_, _>>();
    assert!(
        probes
            .iter()
            .all(|probe| parameter_inputs.contains_key(probe.parameter))
    );

    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.extend(inputs);
    let base_relu = CpuBackend.execute(&graph, relu_input, &bindings).unwrap();
    probes
        .iter()
        .map(|probe| {
            let input_name = &parameter_inputs[probe.parameter];
            let parameter = &bindings[input_name];
            assert!(
                probe.coordinate < parameter.len(),
                "{} probe {}[{}] is outside {} lanes",
                probe.boundary,
                probe.parameter,
                probe.coordinate,
                parameter.len()
            );
            let outputs = [loss, relu_input];
            let context = format!(
                "{} through {}[{}]",
                probe.boundary, probe.parameter, probe.coordinate
            );
            EPSILONS
                .into_iter()
                .find_map(|epsilon| {
                    let plus_bindings = perturbed_parameter_bindings(
                        &bindings,
                        input_name,
                        parameter,
                        probe.coordinate,
                        epsilon,
                    );
                    let minus_bindings = perturbed_parameter_bindings(
                        &bindings,
                        input_name,
                        parameter,
                        probe.coordinate,
                        -epsilon,
                    );
                    let plus = CpuBackend
                        .execute_many(&graph, &outputs, &plus_bindings)
                        .unwrap();
                    let minus = CpuBackend
                        .execute_many(&graph, &outputs, &minus_bindings)
                        .unwrap();
                    if !relu_region_unchanged(&base_relu, &plus.outputs[1])
                        || !relu_region_unchanged(&base_relu, &minus.outputs[1])
                    {
                        return None;
                    }
                    assert_relu_region_unchanged(
                        &base_relu,
                        &plus.outputs[1],
                        &format!("{context} +"),
                    );
                    assert_relu_region_unchanged(
                        &base_relu,
                        &minus.outputs[1],
                        &format!("{context} -"),
                    );
                    Some(
                        (plus.outputs[0].scalar_at(0).as_f64()
                            - minus.outputs[0].scalar_at(0).as_f64())
                            / (2.0 * epsilon),
                    )
                })
                .unwrap_or_else(|| panic!("no bounded central difference preserved {context}"))
        })
        .collect()
}

fn evaluate(model: &TinyCausalTransformer) -> TensorData {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [BATCH, TIME], DType::I32);
    let logits = model.forward_eval(&mut graph, tokens).unwrap();
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.insert("tokens".into(), batch(1).remove("tokens").unwrap());
    CpuBackend.execute(&graph, logits, &bindings).unwrap()
}

fn evaluate_mean_sparse_loss(model: &TinyCausalTransformer) -> f64 {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [BATCH, TIME], DType::I32);
    let targets = graph.input_dtype("targets", [BATCH, TIME], DType::I32);
    let logits = model.forward_eval(&mut graph, tokens).unwrap();
    let loss = sparse_causal_loss(&mut graph, logits, targets).unwrap();
    let parameter_bindings = model.input_bindings(&graph).unwrap();
    let total = (1..=ACCUMULATION_STEPS)
        .map(|replay| {
            let mut bindings = parameter_bindings.clone();
            bindings.extend(batch(replay));
            CpuBackend
                .execute(&graph, loss, &bindings)
                .unwrap()
                .scalar_at(0)
                .as_f64()
        })
        .sum::<f64>();
    total / ACCUMULATION_STEPS as f64
}

fn evaluate_mean_masked_sparse_loss(model: &TinyCausalTransformer) -> f64 {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [BATCH, TIME], DType::I32);
    let targets = graph.input_dtype("targets", [BATCH, TIME], DType::I32);
    let loss_mask = graph.input_dtype(LOSS_MASK, [BATCH, TIME], DType::F32);
    let logits = model.forward_eval(&mut graph, tokens).unwrap();
    let loss = masked_sparse_causal_loss(&mut graph, logits, targets, loss_mask).unwrap();
    let parameter_bindings = model.input_bindings(&graph).unwrap();
    let total = (1..=ACCUMULATION_STEPS)
        .map(|replay| {
            let mut bindings = parameter_bindings.clone();
            bindings.extend(masked_batch(replay).into_compiled_inputs().unwrap());
            CpuBackend
                .execute(&graph, loss, &bindings)
                .unwrap()
                .scalar_at(0)
                .as_f64()
        })
        .sum::<f64>();
    total / ACCUMULATION_STEPS as f64
}

#[derive(Debug)]
struct ExactResumeEvaluation {
    initial_mean_sparse_loss: f64,
    final_mean_sparse_loss: f64,
}

fn compiled_transformer(model: &TinyCausalTransformer) -> CompiledAdamWPlan {
    CompiledAdamWPlan::compile_module_with_dropout(config(), dropout_config(), model, build)
        .expect("the fixed causal Transformer training program must compile")
}

fn owned_compiled_transformer(
    model: TinyCausalTransformer,
) -> rustgrad::CompiledModuleAdamWPlan<TinyCausalTransformer> {
    CompiledModuleAdamWPlan::compile_with_dropout(config(), dropout_config(), model, build)
        .unwrap()
        .with_evaluation(build_evaluation)
        .unwrap()
}

fn run_exact_resume<R, P>(evaluation_tolerance: f64, mut prepare: P) -> ExactResumeEvaluation
where
    R: CompiledAdamWRuntime + CompiledAdamWFlushRuntime + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<TinyCausalTransformer>,
    ) -> Result<rustgrad::CompiledModuleAdamWSession<TinyCausalTransformer, R>>,
{
    let model = TinyCausalTransformer::new(7).unwrap();
    assert!(model.block.is_causal());
    assert_ne!(batch(1), batch(2));
    assert_ne!(batch(2), batch(3));
    assert_ne!(batch(1), batch(3));
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&model);
    let plan = owned_compiled_transformer(model);
    let capture_identity = plan.capture_identity();
    let mut uninterrupted = prepare(plan).unwrap();
    assert!(
        uninterrupted
            .parameter_snapshots()
            .unwrap()
            .contains_key("tokens.weight")
    );
    assert!(
        !uninterrupted
            .parameter_snapshots()
            .unwrap()
            .contains_key("lm_head.weight"),
        "the tied output head must share the embedding's recurrent state"
    );
    assert_eq!(
        uninterrupted.gradient_accumulation_steps(),
        ACCUMULATION_STEPS
    );
    assert_eq!(uninterrupted.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    let initial_parameters = uninterrupted.parameter_snapshots().unwrap();
    let initial_first_moments = uninterrupted.first_moment_snapshots().unwrap();
    let initial_second_moments = uninterrupted.second_moment_snapshots().unwrap();
    let empty_accumulators = uninterrupted.gradient_accumulator_snapshots().unwrap();

    for replay in 1..=2 {
        let step = uninterrupted.step(batch(replay), learning_rate()).unwrap();
        assert_eq!(step.step(), replay);
        assert_eq!(step.optimizer_step(), 0);
        assert_eq!(step.accumulation_index(), replay);
        assert!(!step.did_update());
    }
    assert_ne!(
        uninterrupted.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators,
        "two distinct microbatches must populate the partial gradient window"
    );
    assert_eq!(
        uninterrupted.parameter_snapshots().unwrap(),
        initial_parameters
    );
    assert_eq!(
        uninterrupted.first_moment_snapshots().unwrap(),
        initial_first_moments
    );
    assert_eq!(
        uninterrupted.second_moment_snapshots().unwrap(),
        initial_second_moments
    );
    let reset = uninterrupted.zero_grad().unwrap();
    assert_eq!(reset.discarded_microbatches(), 2);
    assert_eq!(uninterrupted.step_count(), 2);
    assert_eq!(uninterrupted.optimizer_step().unwrap(), 0);
    assert_eq!(uninterrupted.accumulation_index().unwrap(), 0);
    assert_eq!(
        uninterrupted.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );
    let after_reset = uninterrupted.checkpoint().unwrap();
    assert_eq!(
        uninterrupted.zero_grad().unwrap().discarded_microbatches(),
        0
    );
    assert_eq!(uninterrupted.checkpoint().unwrap(), after_reset);

    for replay in 3..=4 {
        let step = uninterrupted.step(batch(replay), learning_rate()).unwrap();
        assert_eq!(step.step(), replay);
        assert_eq!(step.optimizer_step(), 0);
        assert_eq!(step.accumulation_index(), replay - 2);
        assert!(!step.did_update());
    }
    let saved = uninterrupted.checkpoint().unwrap();
    let checkpoint = CompiledAdamWCheckpoint::from_bytes(saved.into_bytes()).unwrap();
    let (_, checkpoint_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(checkpoint_metadata["format"], "rustgrad-compiled-adamw-v4");
    let checkpoint_info = *checkpoint.info();
    assert_eq!(checkpoint_info.capture_identity(), capture_identity);
    assert_eq!(checkpoint_info.replay_step(), 4);
    assert_eq!(checkpoint_info.optimizer_step(), 0);
    assert_eq!(checkpoint_info.gradient_accumulation_steps(), 3);
    assert_eq!(checkpoint_info.accumulation_index(), 2);
    assert_eq!(checkpoint_info.discarded_microbatches(), 2);
    assert_eq!(checkpoint_info.flushed_window_count(), 0);
    assert_eq!(checkpoint_info.flushed_microbatch_count(), 0);
    assert_eq!(checkpoint_info.flush_capture_identity(), None);
    assert_eq!(checkpoint_info.dropout_block_counter(), Some(48));
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let tied = resumed_model.tokens.weight.clone();
    let frozen = resumed_model.frozen_scale.clone();
    let training_builds = Cell::new(0);
    let evaluation_builds = Cell::new(0);
    let resumed_plan = CompiledModuleAdamWPlan::compile_with_dropout(
        config(),
        dropout_config(),
        resumed_model,
        |model, graph, inputs, dropout| {
            training_builds.set(training_builds.get() + 1);
            build(model, graph, inputs, dropout)
        },
    )
    .unwrap()
    .with_evaluation(|model, graph, inputs| {
        evaluation_builds.set(evaluation_builds.get() + 1);
        build_evaluation(model, graph, inputs)
    })
    .unwrap();
    assert_eq!(training_builds.get(), 1);
    assert_eq!(evaluation_builds.get(), 1);
    let resumed_plan = resumed_plan.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(training_builds.get(), 1);
    assert_eq!(evaluation_builds.get(), 1);
    assert_eq!(resumed_plan.capture_identity(), capture_identity);
    assert_eq!(resumed_plan.step_count(), 4);
    let mut resumed = prepare(resumed_plan).unwrap();
    assert_eq!(resumed.optimizer_step().unwrap(), 0);
    assert_eq!(resumed.accumulation_index().unwrap(), 2);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);

    for replay in (checkpoint_info.replay_step() + 1)..=7 {
        let expected = uninterrupted.step(batch(replay), learning_rate()).unwrap();
        let actual = resumed.step(batch(replay), learning_rate()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.did_update(), expected.did_update());
        let (optimizer_step, accumulation_index, did_update) = match replay {
            5 => (1, 0, true),
            6 => (1, 1, false),
            7 => (1, 2, false),
            _ => unreachable!(),
        };
        assert_eq!(actual.optimizer_step(), optimizer_step);
        assert_eq!(actual.accumulation_index(), accumulation_index);
        assert_eq!(actual.did_update(), did_update);
    }
    let expected_flush = uninterrupted.flush_partial_window(learning_rate()).unwrap();
    let actual_flush = resumed.flush_partial_window(learning_rate()).unwrap();
    assert_eq!(actual_flush.flushed_microbatches(), 2);
    assert_eq!(
        actual_flush.flushed_microbatches(),
        expected_flush.flushed_microbatches()
    );
    assert_eq!(
        actual_flush.optimizer_step(),
        expected_flush.optimizer_step()
    );
    assert_eq!(actual_flush.did_update(), expected_flush.did_update());
    assert!(actual_flush.did_update());

    assert_eq!(resumed.step_count(), 7);
    assert_eq!(resumed.optimizer_step().unwrap(), 2);
    assert_eq!(resumed.accumulation_index().unwrap(), 0);
    assert_eq!(
        resumed.parameter_snapshots().unwrap(),
        uninterrupted.parameter_snapshots().unwrap()
    );
    assert_eq!(
        resumed.first_moment_snapshots().unwrap(),
        uninterrupted.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        resumed.second_moment_snapshots().unwrap(),
        uninterrupted.second_moment_snapshots().unwrap()
    );
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        uninterrupted.gradient_accumulator_snapshots().unwrap()
    );
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );
    let final_checkpoint = resumed.checkpoint().unwrap();
    let final_info = final_checkpoint.info();
    assert_eq!(final_info.replay_step(), 7);
    assert_eq!(final_info.optimizer_step(), 2);
    assert_eq!(final_info.accumulation_index(), 0);
    assert_eq!(final_info.discarded_microbatches(), 2);
    assert_eq!(final_info.flushed_window_count(), 1);
    assert_eq!(final_info.flushed_microbatch_count(), 2);
    assert_eq!(
        final_info.flush_capture_identity(),
        resumed.flush_capture_identity()
    );
    assert_eq!(final_info.dropout_block_counter(), Some(84));
    let before_evaluation = resumed.checkpoint().unwrap();
    let evaluation_identity = resumed.evaluation_capture_identity().unwrap();
    let mut final_mean_sparse_loss = 0.0;
    for replay in 1..=ACCUMULATION_STEPS {
        let evaluated = resumed.evaluate(batch(replay)).unwrap();
        assert_eq!(evaluated.capture_identity(), evaluation_identity);
        assert_eq!(
            evaluated.output("logits").unwrap().shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        final_mean_sparse_loss += evaluated.loss().scalar_at(0).as_f64();
    }
    final_mean_sparse_loss /= ACCUMULATION_STEPS as f64;
    assert_eq!(resumed.checkpoint().unwrap(), before_evaluation);
    let published = resumed.parameter_snapshots().unwrap();
    let tied_version = tied.version().unwrap();
    let frozen_before = frozen.snapshot().unwrap();
    let (uninterrupted_model, uninterrupted_checkpoint) =
        uninterrupted.finish_with_checkpoint().unwrap();
    let (resumed_model, finished_checkpoint) = resumed.finish_with_checkpoint().unwrap();
    assert_eq!(finished_checkpoint, final_checkpoint);
    assert_eq!(finished_checkpoint, uninterrupted_checkpoint);
    let final_restore = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        TinyCausalTransformer::new(7).unwrap(),
        &finished_checkpoint,
        build,
    )
    .unwrap()
    .with_evaluation(build_evaluation)
    .unwrap();
    let final_restore = prepare(final_restore).unwrap();
    assert_eq!(final_restore.checkpoint().unwrap(), finished_checkpoint);
    let final_restored_parameters = final_restore.parameter_snapshots().unwrap();
    let _fresh_module = final_restore.into_module_without_publication();
    let live = resumed_model.state_dict().unwrap();
    assert_eq!(live, uninterrupted_model.state_dict().unwrap());
    for (name, value) in &final_restored_parameters {
        assert_eq!(&live.tensors()[name], value);
    }
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert_eq!(resumed_model.tokens.weight.id(), tied.id());
    assert_eq!(
        resumed_model.tokens.weight.version().unwrap(),
        tied_version + 1
    );
    let frozen_after = resumed_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    let versions_before_eval = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let first_eval = evaluate(&resumed_model);
    let second_eval = evaluate(&resumed_model);
    let published_mean_sparse_loss = evaluate_mean_sparse_loss(&resumed_model);
    assert_eq!(first_eval.shape(), &Shape::new([BATCH, TIME, VOCAB]));
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel().unwrap())
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), versions_before_eval[&name]);
    }
    assert!(
        (final_mean_sparse_loss - published_mean_sparse_loss).abs() <= evaluation_tolerance,
        "compiled evaluation differs from published CPU evaluation: {final_mean_sparse_loss} vs {published_mean_sparse_loss}"
    );
    ExactResumeEvaluation {
        initial_mean_sparse_loss,
        final_mean_sparse_loss,
    }
}

fn metal_renderer() -> MetalRenderer {
    MetalRenderer::new(
        8,
        MetalCapabilities {
            max_buffer_length: 1 << 30,
            unified_memory: true,
            family: "Apple9".into(),
        },
    )
    .unwrap()
}

fn assert_strict_dropout_kernels(plan: &MetalCompiledAdamWPlan) {
    let threefry = plan
        .rendered_items()
        .filter(|item| item.entry.starts_with("rg_metal_threefry_"))
        .collect::<Vec<_>>();
    let bitcasts = plan
        .rendered_items()
        .filter(|item| item.entry == "rg_metal_portable_bitcast")
        .collect::<Vec<_>>();
    assert_eq!(threefry.len(), 2);
    assert_eq!(bitcasts.len(), 4);
    assert!(
        threefry
            .into_iter()
            .chain(bitcasts)
            .all(|item| item.transaction.is_none() && item.indexed_movement().is_none()),
        "compiled dropout kernels must not introduce a status-read transaction"
    );
}

#[test]
fn compiled_transformer_plan_cpu_target_decreases_loss_and_resumes_exactly() {
    let target = CpuSessionTarget::new();
    let evaluation = run_exact_resume(0.0, |plan| {
        plan.prepare(&target).map_err(|error| error.into_parts().1)
    });

    assert!(
        evaluation.final_mean_sparse_loss < evaluation.initial_mean_sparse_loss,
        "compiled causal Transformer eval loss did not decrease: {evaluation:?}"
    );
}

#[test]
fn compiled_transformer_checkpoint_rebase_is_independent_and_owned_failure_is_recoverable() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let plan = compiled_transformer(&model);
    let capture_identity = plan.capture_identity();
    let flush_capture_identity = plan.flush_capture_identity();
    let mut source = plan.prepare_cpu().unwrap();
    for replay in 1..=2 {
        source.step(batch(replay), learning_rate()).unwrap();
    }
    assert_eq!(source.zero_grad().unwrap().discarded_microbatches(), 2);
    for replay in 3..=4 {
        source.step(batch(replay), learning_rate()).unwrap();
    }
    let checkpoint_parameters = source.parameter_snapshots().unwrap();
    let checkpoint = source.checkpoint().unwrap();

    let first = plan.restore_checkpoint(&checkpoint).unwrap();
    let second = plan.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(
        plan.step_count(),
        0,
        "the source plan must remain unchanged"
    );
    for restored in [&first, &second] {
        assert_eq!(restored.capture_identity(), capture_identity);
        assert_eq!(restored.flush_capture_identity(), flush_capture_identity);
        assert_eq!(restored.dropout_config(), Some(dropout_config()));
        assert_eq!(restored.gradient_accumulation_steps(), ACCUMULATION_STEPS);
        assert_eq!(restored.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
        assert_eq!(restored.loss_scale(), 128.0);
        assert_eq!(restored.step_count(), 4);
    }

    let mut first = first.prepare_cpu().unwrap();
    let mut second = second.prepare_cpu().unwrap();
    assert_eq!(first.checkpoint().unwrap(), checkpoint);
    assert_eq!(second.checkpoint().unwrap(), checkpoint);
    assert!(first.zero_grad().unwrap().did_discard());
    assert_ne!(first.checkpoint().unwrap(), checkpoint);
    assert_eq!(second.checkpoint().unwrap(), checkpoint);

    let expected_flush = source.flush_partial_window(learning_rate()).unwrap();
    let actual_flush = second.flush_partial_window(learning_rate()).unwrap();
    assert_eq!(actual_flush.flushed_microbatches(), 2);
    assert_eq!(
        actual_flush.optimizer_step(),
        expected_flush.optimizer_step()
    );
    assert_eq!(second.checkpoint().unwrap(), source.checkpoint().unwrap());

    let (state, mut metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    metadata.insert(
        "capture_identity".into(),
        capture_identity.wrapping_add(1).to_string(),
    );
    let malformed_identity =
        CompiledAdamWCheckpoint::from_bytes(save_safetensors(&state, &metadata).unwrap()).unwrap();
    let owned_model = TinyCausalTransformer::new(41).unwrap();
    let tied = owned_model.tokens.weight.clone();
    let tied_version = tied.version().unwrap();
    let frozen = owned_model.frozen_scale.clone();
    let frozen_before = frozen.snapshot().unwrap();
    assert_ne!(
        owned_model.tokens.weight.value().unwrap(),
        checkpoint_parameters["tokens.weight"],
        "the owned candidate must be initialized differently from the checkpoint"
    );
    let owned = owned_compiled_transformer(owned_model);
    let evaluation_identity = owned.evaluation_capture_identity().unwrap();
    let error = match owned.restore_checkpoint(&malformed_identity) {
        Ok(_) => panic!("a mismatched capture identity restored"),
        Err(error) => error,
    };
    assert!(
        error
            .source_error()
            .to_string()
            .contains("capture identity mismatch")
    );
    assert_eq!(error.plan().capture_identity(), capture_identity);
    assert_eq!(
        error.plan().evaluation_capture_identity(),
        Some(evaluation_identity)
    );
    let restored = error.into_plan().restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(
        restored.evaluation_capture_identity(),
        Some(evaluation_identity)
    );
    let mut restored = restored.prepare(&CpuSessionTarget::new()).unwrap();
    assert_eq!(restored.checkpoint().unwrap(), checkpoint);
    let before_evaluation = restored.checkpoint().unwrap();
    let evaluated = restored.evaluate(batch(1)).unwrap();
    assert_eq!(evaluated.capture_identity(), evaluation_identity);
    assert_eq!(restored.checkpoint().unwrap(), before_evaluation);
    let (restored_model, finished_checkpoint) = restored.finish_with_checkpoint().unwrap();
    assert_eq!(finished_checkpoint, checkpoint);
    assert_eq!(restored_model.tokens.weight.id(), tied.id());
    assert_eq!(
        restored_model.tokens.weight.version().unwrap(),
        tied_version + 1
    );
    assert_eq!(
        restored_model.tokens.weight.value().unwrap(),
        checkpoint_parameters["tokens.weight"]
    );
    let frozen_after = restored_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    assert_eq!(frozen_after.identity, frozen_before.identity);
    assert_eq!(frozen_after.trainable, frozen_before.trainable);
    assert!(
        !restored_model
            .state_dict()
            .unwrap()
            .tensors()
            .contains_key("lm_head.weight")
    );
}

#[test]
fn compiled_transformer_native_cpu_target_is_strict_precompiled_and_resumes_exactly() {
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let evaluation = run_exact_resume(1e-5, |plan| {
        let inspection = plan.inspection()?;
        assert!(inspection.partial_flush().is_some());
        assert!(inspection.evaluation().is_some());
        let session = plan
            .prepare(&target)
            .map_err(|error| error.into_parts().1)?;
        let preparation = session.native_cpu_preparation_report();
        assert!(preparation.main().native_item_count() > 0);
        assert!(preparation.partial_flush().is_some());
        assert!(preparation.evaluation().is_some());
        Ok(session)
    });

    assert!(
        evaluation.final_mean_sparse_loss < evaluation.initial_mean_sparse_loss,
        "strict-native compiled causal Transformer eval loss did not decrease: {evaluation:?}"
    );
}

#[test]
fn compiled_transformer_native_cpu_scoreboard_is_bounded_and_authenticated() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let plan =
        CompiledAdamWPlan::compile_module_with_dropout(config(), dropout_config(), &model, build)
            .unwrap();
    let inspection = plan.inspection().unwrap();
    assert!(inspection.main().1.schedule_item_count > 0);
    assert!(inspection.main().1.peak_logical_bytes > 0);
    assert!(inspection.partial_flush().is_some());
    assert!(inspection.evaluation().is_none());
    assert!(inspection.recurrent_state_count() > 0);
    assert!(inspection.recurrent_state_bytes() > 0);

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let mut session = plan.prepare(&target).unwrap();
    let mut scoreboard = NativeTrainingScoreboard::new(
        inspection.clone(),
        session.preparation_report(),
        Duration::ZERO,
        Duration::ZERO,
    )
    .unwrap();

    let mut malformed = batch(1);
    malformed.remove("targets");
    assert!(session.step(malformed, learning_rate()).is_err());
    for replay in 1..=3 {
        let step = session.step(batch(replay), learning_rate()).unwrap();
        scoreboard.record(step.report()).unwrap();
    }
    let checkpoint = session.checkpoint().unwrap();
    let checkpoint_bytes = checkpoint.as_bytes().len();
    scoreboard
        .observe_checkpoint(&checkpoint, Duration::ZERO)
        .unwrap();
    let report = scoreboard.report().unwrap();
    let bytes = report.to_json_bytes().unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(report.successful_replay_count(), 3);
    assert_eq!(report.steady_replay_wall_time().sample_count, 2);
    assert_eq!(report.main().capture_identity(), inspection.main().0);
    assert_eq!(
        report.main().execution_plan_identity(),
        inspection.main().1.identity
    );
    assert_eq!(
        report.main().logical_schedule_item_count() as usize,
        inspection.main().1.schedule_item_count,
    );
    assert_eq!(
        report.main().peak_logical_temporary_bytes() as usize,
        inspection.main().1.peak_logical_bytes,
    );
    assert_eq!(
        report.schedule_cache_keys().len(),
        report.main().native_item_count() as usize
    );
    assert_eq!(
        report.partial_flush().unwrap().capture_identity(),
        inspection.partial_flush().unwrap().0
    );
    assert_eq!(report.fallback_count(), 0);
    assert_eq!(
        report.recurrent_state_count() as usize,
        inspection.recurrent_state_count()
    );
    assert_eq!(
        report.recurrent_state_bytes() as usize,
        inspection.recurrent_state_bytes()
    );
    assert_eq!(
        report.checkpoint_byte_count().unwrap() as usize,
        checkpoint_bytes
    );
    assert!(
        report
            .steady_microbatches_per_second()
            .is_none_or(|rate| rate.is_finite() && rate > 0.0)
    );
    assert!(json["kernel_launch_count"].is_null());
    assert!(json["host_to_device"].is_null());
    assert!(json["device_to_host"].is_null());
    assert!(json["measured_peak_host_memory_bytes"].is_null());
    assert_eq!(
        NativeTrainingReport::from_json_bytes(&bytes).unwrap(),
        report
    );

    let restored = plan.restore_checkpoint(&checkpoint).unwrap();
    let restored_inspection = restored.inspection().unwrap();
    assert_eq!(restored_inspection.initial_replay_step(), 3);
    assert_eq!(restored_inspection.main(), inspection.main());
    assert_eq!(
        restored_inspection.partial_flush(),
        inspection.partial_flush()
    );
    assert_eq!(
        restored_inspection.recurrent_state_count(),
        inspection.recurrent_state_count()
    );
    assert_eq!(
        restored_inspection.recurrent_state_bytes(),
        inspection.recurrent_state_bytes()
    );
    let cached = restored.prepare(&target).unwrap();
    assert_eq!(cached.preparation_report().main().cache_miss_count(), 0);
    assert_eq!(
        cached.preparation_report().main().cache_hit_count(),
        cached.preparation_report().main().native_item_count()
    );
}

#[test]
fn compiled_evaluation_is_failure_atomic_and_retryable() {
    let plan = owned_compiled_transformer(TinyCausalTransformer::new(7).unwrap());
    let mut session = plan.prepare(&CpuSessionTarget).unwrap();
    for replay in 1..=ACCUMULATION_STEPS {
        session.step(batch(replay), learning_rate()).unwrap();
    }
    let checkpoint = session.checkpoint().unwrap();
    let mut incomplete = batch(1);
    incomplete.remove("targets");
    assert!(session.evaluate(incomplete).is_err());
    assert_eq!(session.checkpoint().unwrap(), checkpoint);
    let first = session.evaluate(batch(1)).unwrap();
    let second = session.evaluate(batch(1)).unwrap();
    assert_eq!(first.loss(), second.loss());
    assert_eq!(first.outputs(), second.outputs());
    assert_eq!(session.checkpoint().unwrap(), checkpoint);
}

#[test]
fn compiled_transformer_active_global_clip_changes_the_first_window_update() {
    let clipped_plan = CompiledModuleAdamWPlan::compile_with_dropout(
        config(),
        dropout_config(),
        TinyCausalTransformer::new(7).unwrap(),
        build,
    )
    .unwrap();
    let unclipped_plan = CompiledModuleAdamWPlan::compile_with_dropout(
        config_with_max_gradient_norm(None),
        dropout_config(),
        TinyCausalTransformer::new(7).unwrap(),
        build,
    )
    .unwrap();
    assert_eq!(
        clipped_plan.dropout_config(),
        unclipped_plan.dropout_config()
    );
    assert_ne!(
        clipped_plan.capture_identity(),
        unclipped_plan.capture_identity(),
        "the global clipping policy must remain capture-authenticated"
    );

    let target = CpuSessionTarget::new();
    let mut clipped = clipped_plan.prepare(&target).unwrap();
    let mut unclipped = unclipped_plan.prepare(&target).unwrap();
    assert_eq!(clipped.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(unclipped.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(clipped.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    assert_eq!(unclipped.max_gradient_norm(), None);
    assert_eq!(clipped.loss_scale(), unclipped.loss_scale());

    for replay in 1..=ACCUMULATION_STEPS {
        let clipped_step = clipped.step(batch(replay), learning_rate()).unwrap();
        let unclipped_step = unclipped.step(batch(replay), learning_rate()).unwrap();
        assert_eq!(clipped_step.loss(), unclipped_step.loss());
        assert_eq!(clipped_step.outputs(), unclipped_step.outputs());
        assert_eq!(clipped_step.step(), replay);
        assert_eq!(unclipped_step.step(), replay);
        let did_update = replay == ACCUMULATION_STEPS;
        assert_eq!(clipped_step.did_update(), did_update);
        assert_eq!(unclipped_step.did_update(), did_update);
    }

    assert_eq!(clipped.optimizer_step().unwrap(), 1);
    assert_eq!(unclipped.optimizer_step().unwrap(), 1);
    assert_eq!(clipped.accumulation_index().unwrap(), 0);
    assert_eq!(unclipped.accumulation_index().unwrap(), 0);
    assert_ne!(
        clipped.first_moment_snapshots().unwrap(),
        unclipped.first_moment_snapshots().unwrap(),
        "the maintained gradient norm must exceed its configured clipping limit"
    );
    assert_ne!(
        clipped.parameter_snapshots().unwrap(),
        unclipped.parameter_snapshots().unwrap(),
        "active clipping must change the maintained Transformer's update"
    );
}

#[test]
fn compiled_transformer_second_window_gradient_inputs_match_numerical_oracle() {
    const NUMERICAL_ABSOLUTE_TOLERANCE: f64 = 6e-3;
    const NUMERICAL_RELATIVE_TOLERANCE: f64 = 3e-3;
    const ADAM_INPUT_TOLERANCE: f64 = 1e-5;
    const FIRST_MOMENT_TOLERANCE: f64 = 7e-4;
    const SECOND_MOMENT_TOLERANCE: f64 = 7e-6;
    const PROBES: [TransformerGradientProbe; 10] = [
        TransformerGradientProbe {
            parameter: "tokens.weight",
            coordinate: 4,
            boundary: "embedding lookup, tied output transpose, gather, and mean",
        },
        TransformerGradientProbe {
            parameter: "block.query.0",
            coordinate: 3,
            boundary: "query projection and attention views",
        },
        TransformerGradientProbe {
            parameter: "block.key.0",
            coordinate: 2,
            boundary: "key projection and causal attention",
        },
        TransformerGradientProbe {
            parameter: "block.value.0",
            coordinate: 1,
            boundary: "value projection and attention reduction",
        },
        TransformerGradientProbe {
            parameter: "block.out.0",
            coordinate: 0,
            boundary: "attention output projection",
        },
        TransformerGradientProbe {
            parameter: "block.ln1.0",
            coordinate: 1,
            boundary: "first LayerNorm affine scale",
        },
        TransformerGradientProbe {
            parameter: "block.ln2.1",
            coordinate: 0,
            boundary: "second LayerNorm affine bias",
        },
        TransformerGradientProbe {
            parameter: "block.ff1.0",
            coordinate: 6,
            boundary: "feed-forward expansion and ReLU",
        },
        TransformerGradientProbe {
            parameter: "block.ff2.0",
            coordinate: 5,
            boundary: "feed-forward contraction",
        },
        TransformerGradientProbe {
            parameter: "norm.bias",
            coordinate: 1,
            boundary: "final LayerNorm affine and logits reduction",
        },
    ];

    // Clipping is disabled only for this focused proof so the completed
    // window's first moment exposes the exact averaged gradient presented to
    // AdamW. The separate clipping tests cover the intervening global policy.
    let optimizer = config_with_max_gradient_norm(None);
    let model = TinyCausalTransformer::new(7).unwrap();
    let tied_identity = model.tokens.weight.id();
    let plan = CompiledAdamWPlan::compile_module_with_dropout(
        optimizer.clone(),
        dropout_config(),
        &model,
        build_with_dropout_observations,
    )
    .unwrap();
    let mut runtime = plan.prepare(&CpuSessionTarget).unwrap();

    for replay in 1..=ACCUMULATION_STEPS {
        let step = runtime.step(batch(replay), learning_rate()).unwrap();
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
    }
    assert_eq!(runtime.step_count(), ACCUMULATION_STEPS);
    assert_eq!(runtime.optimizer_step().unwrap(), 1);
    assert_eq!(runtime.accumulation_index().unwrap(), 0);
    assert_eq!(runtime.dropout_block_counter().unwrap(), Some(36));

    // Freeze an independent forward-only oracle at the exact recurrent
    // parameter frontier after window one. It never calls Graph::gradient or
    // any production reverse-mode helper.
    let frontier_parameters = runtime.parameter_snapshots().unwrap();
    let frontier_first_moments = runtime.first_moment_snapshots().unwrap();
    let frontier_second_moments = runtime.second_moment_snapshots().unwrap();
    let frontier_checkpoint = runtime.checkpoint().unwrap();
    let oracle_model = TinyCausalTransformer::new(7).unwrap();
    oracle_model
        .load_trainable_parameters_exact(&frontier_parameters)
        .unwrap();

    // A checkpoint-identical sibling advances the same replay/dropout cursor,
    // snapshots each raw gradient in the existing accumulator seam, and then
    // discards it before the next replay. This captures all three microbatch
    // gradients without reaching clipping or AdamW and without changing the
    // sequential runtime whose completed-window recurrence is checked below.
    let gradient_probe_model = TinyCausalTransformer::new(7).unwrap();
    let mut gradient_probe_runtime =
        CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
            optimizer.clone(),
            dropout_config(),
            &gradient_probe_model,
            &frontier_checkpoint,
            build_with_dropout_observations,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap();
    let mut masks = Vec::with_capacity(ACCUMULATION_STEPS as usize);
    let mut raw_gradients = Vec::with_capacity(ACCUMULATION_STEPS as usize);
    for replay in ACCUMULATION_STEPS + 1..=2 * ACCUMULATION_STEPS {
        let step = gradient_probe_runtime
            .step(batch(replay), learning_rate())
            .unwrap();
        assert!(!step.did_update());
        masks.push(observed_dropout_masks(step.outputs()));
        assert_eq!(gradient_probe_runtime.accumulation_index().unwrap(), 1);
        raw_gradients.push(
            gradient_probe_runtime
                .gradient_accumulator_snapshots()
                .unwrap(),
        );
        assert!(gradient_probe_runtime.zero_grad().unwrap().did_discard());
    }
    assert_eq!(raw_gradients.len(), ACCUMULATION_STEPS as usize);
    assert_eq!(gradient_probe_runtime.optimizer_step().unwrap(), 1);
    assert_eq!(gradient_probe_runtime.accumulation_index().unwrap(), 0);
    assert_eq!(
        gradient_probe_runtime.dropout_block_counter().unwrap(),
        Some(72)
    );

    for (index, replay) in (ACCUMULATION_STEPS + 1..=2 * ACCUMULATION_STEPS).enumerate() {
        let step = runtime.step(batch(replay), learning_rate()).unwrap();
        assert_eq!(observed_dropout_masks(step.outputs()), masks[index]);
        assert_eq!(step.did_update(), replay == 2 * ACCUMULATION_STEPS);
    }
    assert_eq!(runtime.optimizer_step().unwrap(), 2);
    assert_eq!(runtime.accumulation_index().unwrap(), 0);
    assert_eq!(runtime.dropout_block_counter().unwrap(), Some(72));

    let numerical = masks
        .into_iter()
        .enumerate()
        .map(|(index, masks)| {
            numerical_transformer_gradient_lanes(
                &oracle_model,
                batch(ACCUMULATION_STEPS + 1 + index as u64),
                masks,
                &PROBES,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(numerical.len(), ACCUMULATION_STEPS as usize);
    assert!(
        numerical
            .iter()
            .all(|gradient| gradient.len() == PROBES.len())
    );

    let next_first_moments = runtime.first_moment_snapshots().unwrap();
    let next_second_moments = runtime.second_moment_snapshots().unwrap();
    let beta1 = optimizer.beta1();
    let beta2 = optimizer.beta2();
    for (probe_index, probe) in PROBES.iter().enumerate() {
        let numerical_lanes = [
            numerical[0][probe_index] as f32,
            numerical[1][probe_index] as f32,
            numerical[2][probe_index] as f32,
        ];
        let actual_lanes: [f32; 3] = std::array::from_fn(|replay_index| {
            raw_gradients[replay_index][probe.parameter]
                .scalar_at(probe.coordinate)
                .as_f64() as f32
        });
        for (replay_index, (actual, expected)) in
            actual_lanes.into_iter().zip(numerical_lanes).enumerate()
        {
            let error = (f64::from(actual) - f64::from(expected)).abs();
            let tolerance = NUMERICAL_ABSOLUTE_TOLERANCE
                + NUMERICAL_RELATIVE_TOLERANCE * f64::from(actual.abs().max(expected.abs()));
            assert!(
                error <= tolerance,
                "{} raw gradient {}[{}] at second-window replay {} mismatch: actual={actual}, numerical={expected}, error={error}, tolerance={tolerance}",
                probe.boundary,
                probe.parameter,
                probe.coordinate,
                replay_index + 1
            );
        }

        let numerical_average = ((numerical_lanes[0] + numerical_lanes[1]) + numerical_lanes[2])
            / ACCUMULATION_STEPS as f32;
        let average =
            ((actual_lanes[0] + actual_lanes[1]) + actual_lanes[2]) / ACCUMULATION_STEPS as f32;
        let average_error = (f64::from(average) - f64::from(numerical_average)).abs();
        let average_tolerance = NUMERICAL_ABSOLUTE_TOLERANCE
            + NUMERICAL_RELATIVE_TOLERANCE * f64::from(average.abs().max(numerical_average.abs()));
        assert!(
            average_error <= average_tolerance,
            "{} averaged raw gradient {}[{}] mismatch: captured={average}, numerical={numerical_average}, error={average_error}, tolerance={average_tolerance}",
            probe.boundary,
            probe.parameter,
            probe.coordinate
        );
        let previous_first = frontier_first_moments[probe.parameter]
            .scalar_at(probe.coordinate)
            .as_f64() as f32;
        let actual_first = next_first_moments[probe.parameter]
            .scalar_at(probe.coordinate)
            .as_f64() as f32;
        let captured_adam_input = (actual_first - beta1 * previous_first) / (1.0 - beta1);
        assert!(
            (f64::from(captured_adam_input) - f64::from(average)).abs() <= ADAM_INPUT_TOLERANCE,
            "{} averaged AdamW input {}[{}] mismatch: captured={captured_adam_input}, raw={average}",
            probe.boundary,
            probe.parameter,
            probe.coordinate
        );

        let expected_first = beta1 * previous_first + (1.0 - beta1) * average;
        let previous_second = frontier_second_moments[probe.parameter]
            .scalar_at(probe.coordinate)
            .as_f64() as f32;
        let actual_second = next_second_moments[probe.parameter]
            .scalar_at(probe.coordinate)
            .as_f64() as f32;
        let expected_second = beta2 * previous_second + (1.0 - beta2) * (average * average);
        assert!(
            (f64::from(actual_first) - f64::from(expected_first)).abs() <= FIRST_MOMENT_TOLERANCE,
            "{} first moment {}[{}] mismatch: actual={actual_first}, numerical={expected_first}",
            probe.boundary,
            probe.parameter,
            probe.coordinate
        );
        assert!(
            (f64::from(actual_second) - f64::from(expected_second)).abs()
                <= SECOND_MOMENT_TOLERANCE,
            "{} second moment {}[{}] mismatch: actual={actual_second}, numerical={expected_second}",
            probe.boundary,
            probe.parameter,
            probe.coordinate
        );
    }

    assert_eq!(model.tokens.weight.id(), tied_identity);
    assert!(!frontier_parameters.contains_key("lm_head.weight"));
    assert!(
        runtime
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .all(|value| value.to_vec_f64().into_iter().all(|lane| lane == 0.0))
    );
}

#[test]
fn compiled_transformer_recurrent_adamw_updates_match_analytic_reference() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let state_before = model.state_dict().unwrap();
    let parameter_state_before = module_parameter_state(&model);
    let tied_identity = model.tokens.weight.id();
    let frozen_before = model.frozen_scale.snapshot().unwrap();
    let optimizer = config();
    assert_eq!(optimizer.loss_scale(), 128.0);
    let max_norm = optimizer.max_gradient_norm().unwrap();
    let exclusions = optimizer
        .weight_decay_exclusions()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let expected_exclusions = WEIGHT_DECAY_EXCLUSIONS
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(exclusions, expected_exclusions);

    let plan = CompiledAdamWPlan::compile_module_with_dropout(
        optimizer.clone(),
        dropout_config(),
        &model,
        build_with_dropout_observations,
    )
    .unwrap();
    assert_eq!(plan.dropout_blocks_per_replay(), Some(12));
    let mut runtime = plan.prepare(&CpuSessionTarget).unwrap();
    let initial_parameters = runtime.parameter_snapshots().unwrap();
    assert_eq!(initial_parameters.len(), 19);
    assert!(initial_parameters.contains_key("tokens.weight"));
    assert!(!initial_parameters.contains_key("lm_head.weight"));
    assert!(!initial_parameters.contains_key("frozen_scale"));

    let learning_rate_value = learning_rate().scalar_at(0).as_f64() as f32;
    let mut excluded_decay_counterfactuals = 0;
    let mut included_decay_counterfactuals = 0;
    let oracle_model = TinyCausalTransformer::new(7).unwrap();
    assert_eq!(oracle_model.state_dict().unwrap(), state_before);

    for window in 1..=2 {
        let previous = AdamWOracleState {
            parameters: runtime.parameter_snapshots().unwrap(),
            first_moments: runtime.first_moment_snapshots().unwrap(),
            second_moments: runtime.second_moment_snapshots().unwrap(),
        };
        oracle_model
            .load_trainable_parameters_exact(&previous.parameters)
            .unwrap();
        let first_replay = (window - 1) * ACCUMULATION_STEPS + 1;
        let mut masks = Vec::with_capacity(ACCUMULATION_STEPS as usize);
        for replay in first_replay..first_replay + ACCUMULATION_STEPS {
            let step = runtime.step(batch(replay), learning_rate()).unwrap();
            masks.push(observed_dropout_masks(step.outputs()));
            assert_eq!(step.did_update(), replay == window * ACCUMULATION_STEPS);
        }
        assert_eq!(runtime.step_count(), window * ACCUMULATION_STEPS);
        assert_eq!(runtime.optimizer_step().unwrap(), window);
        assert_eq!(runtime.accumulation_index().unwrap(), 0);
        assert_eq!(runtime.dropout_block_counter().unwrap(), Some(window * 36));

        let gradients = masks
            .into_iter()
            .enumerate()
            .map(|(index, masks)| {
                let replay = first_replay + index as u64;
                maintained_transformer_analytic_gradients(&oracle_model, batch(replay), masks)
            })
            .collect::<Vec<_>>();
        let averaged = average_gradient_window(&gradients);
        let (clipped, gradient_norm) = clip_gradient_window(&averaged, max_norm);
        assert!(gradient_norm.is_finite());
        if window == 1 {
            assert!(
                gradient_norm > max_norm,
                "the maintained first window must activate clipping"
            );
        }
        let update = apply_adamw_oracle_recurrence(
            &previous,
            &clipped,
            &optimizer,
            window,
            learning_rate_value,
        );
        excluded_decay_counterfactuals += update.excluded_decay_counterfactuals;
        included_decay_counterfactuals += update.included_decay_counterfactuals;
        assert_adamw_oracle_state(
            window,
            &update.state,
            &runtime.parameter_snapshots().unwrap(),
            &runtime.first_moment_snapshots().unwrap(),
            &runtime.second_moment_snapshots().unwrap(),
        );
        assert!(
            runtime
                .gradient_accumulator_snapshots()
                .unwrap()
                .values()
                .all(|value| value.to_vec_f64().into_iter().all(|lane| lane == 0.0))
        );
    }

    assert_eq!(runtime.step_count(), 2 * ACCUMULATION_STEPS);
    assert_eq!(runtime.optimizer_step().unwrap(), 2);
    assert_eq!(runtime.accumulation_index().unwrap(), 0);
    assert_eq!(runtime.dropout_block_counter().unwrap(), Some(72));
    assert!(excluded_decay_counterfactuals > 0);
    assert!(included_decay_counterfactuals > 0);
    assert!(!initial_parameters.contains_key("lm_head.weight"));
    assert!(!initial_parameters.contains_key("frozen_scale"));
    assert_eq!(model.tokens.weight.id(), tied_identity);
    let frozen_after = model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    assert_eq!(model.state_dict().unwrap(), state_before);
    assert_eq!(module_parameter_state(&model), parameter_state_before);
}

#[test]
fn owned_compiled_transformer_session_finishes_and_resumes_one_module_lifecycle() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let tied_identity = model.tokens.weight.id();
    let frozen = model.frozen_scale.clone();
    let frozen_before = frozen.snapshot().unwrap();
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&model);
    let plan = owned_compiled_transformer(model);
    let capture_identity = plan.capture_identity();
    let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    for replay in 1..=4 {
        session.step(batch(replay), learning_rate()).unwrap();
    }
    let midpoint = session.parameter_snapshots().unwrap();
    let (model, checkpoint) = session.finish_with_checkpoint().unwrap();
    assert_eq!(model.tokens.weight.id(), tied_identity);
    assert_eq!(
        model.tokens.weight.value().unwrap(),
        midpoint["tokens.weight"]
    );
    assert_eq!(model.tokens.weight.version().unwrap(), 1);
    assert_eq!(
        model.frozen_scale.snapshot().unwrap().data,
        frozen_before.data
    );
    assert_eq!(model.frozen_scale.version().unwrap(), frozen_before.version);
    assert!(
        !model
            .state_dict()
            .unwrap()
            .tensors()
            .contains_key("lm_head.weight")
    );

    let resumed = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        model,
        &checkpoint,
        build,
    )
    .unwrap();
    assert_eq!(resumed.capture_identity(), capture_identity);
    assert_eq!(resumed.step_count(), 4);
    let mut session = resumed.prepare(&CpuSessionTarget::new()).unwrap();
    for replay in 5..=8 {
        session.step(batch(replay), learning_rate()).unwrap();
    }
    let final_parameters = session.parameter_snapshots().unwrap();
    let (model, final_checkpoint) = session.finish_with_checkpoint().unwrap();
    assert_eq!(model.tokens.weight.id(), tied_identity);
    assert_eq!(
        model.tokens.weight.value().unwrap(),
        final_parameters["tokens.weight"]
    );
    assert_eq!(model.tokens.weight.version().unwrap(), 2);
    assert_eq!(
        model.frozen_scale.snapshot().unwrap().data,
        frozen_before.data
    );
    assert_eq!(model.frozen_scale.version().unwrap(), frozen_before.version);
    assert_eq!(final_checkpoint.info().replay_step(), 8);
    assert_eq!(final_checkpoint.info().optimizer_step(), 2);
    let final_mean_sparse_loss = evaluate_mean_sparse_loss(&model);
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "owned compiled causal Transformer eval loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
}

#[test]
fn compiled_transformer_checkpoint_replaces_different_destination_state_exactly() {
    const POLICY_FROZEN: &str = "block.ff1.0";

    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1]).unwrap();
    let policy = masked_config()
        .with_frozen_parameters([POLICY_FROZEN])
        .unwrap()
        .with_captured_multi_step_lr(schedule.clone());
    let masks = (1..=3)
        .map(|replay| masked_batch(replay).loss_mask)
        .collect::<Vec<_>>();
    assert_ne!(masks[0], masks[1]);
    assert_ne!(masks[1], masks[2]);
    assert_eq!(masks[2].to_vec_f64(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    let source = BufferedTinyCausalTransformer::new(7).unwrap();
    let initial_mean_sparse_loss = evaluate_mean_masked_sparse_loss(&source.transformer);
    let source_policy_frozen = source
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .find_map(|(name, parameter)| (name == POLICY_FROZEN).then_some(parameter))
        .unwrap()
        .value()
        .unwrap();
    let builds = Cell::new(0);
    let plan = CompiledAdamWPlan::compile_module_with_dropout(
        policy,
        dropout_config(),
        &source,
        |model, graph, inputs, dropout| {
            builds.set(builds.get() + 1);
            build_buffered(model, graph, inputs, dropout)
        },
    )
    .unwrap();
    assert_eq!(builds.get(), 1);
    assert_eq!(plan.step_count(), 0);
    assert_eq!(plan.captured_multi_step_lr(), Some(&schedule));
    let capture_identity = plan.capture_identity();
    let flush_capture_identity = plan.flush_capture_identity();
    let target =
        CpuSessionTarget::new().with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut uninterrupted = plan.prepare(&target).unwrap();
    assert_eq!(uninterrupted.captured_multi_step_lr(), Some(&schedule));
    assert_eq!(
        uninterrupted.gradient_accumulation_steps(),
        ACCUMULATION_STEPS
    );
    assert_eq!(uninterrupted.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    assert_eq!(
        uninterrupted.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );
    let initial_checkpoint = uninterrupted.checkpoint().unwrap();
    for invalid in invalid_masked_batches() {
        assert!(uninterrupted.step_batch_scheduled(invalid).is_err());
        assert_eq!(uninterrupted.checkpoint().unwrap(), initial_checkpoint);
    }
    let before_wrong_entrypoint = uninterrupted.checkpoint().unwrap();
    assert!(uninterrupted.step_batch(masked_batch(1), 0.05).is_err());
    assert_eq!(uninterrupted.checkpoint().unwrap(), before_wrong_entrypoint);
    let empty_accumulators = uninterrupted.gradient_accumulator_snapshots().unwrap();
    for replay in 1..=2 {
        let step = uninterrupted
            .step_batch_scheduled(masked_batch(replay))
            .unwrap();
        assert!(!step.did_update());
    }
    assert_eq!(
        uninterrupted.zero_grad().unwrap().discarded_microbatches(),
        2
    );
    assert_eq!(uninterrupted.step_count(), 2);
    assert_eq!(uninterrupted.optimizer_step().unwrap(), 0);
    assert_eq!(uninterrupted.accumulation_index().unwrap(), 0);
    assert_eq!(
        uninterrupted.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );
    assert_eq!(
        checkpoint_dropout_block_counter(&uninterrupted.checkpoint().unwrap()),
        24
    );
    for replay in 3..=6 {
        uninterrupted
            .step_batch_scheduled(masked_batch(replay))
            .unwrap();
    }

    let checkpoint = uninterrupted.checkpoint().unwrap();
    let checkpoint_parameters = uninterrupted.parameter_snapshots().unwrap();
    let checkpoint_first_moments = uninterrupted.first_moment_snapshots().unwrap();
    let checkpoint_second_moments = uninterrupted.second_moment_snapshots().unwrap();
    let checkpoint_accumulators = uninterrupted.gradient_accumulator_snapshots().unwrap();
    assert!(
        checkpoint_first_moments
            .values()
            .flat_map(TensorData::to_vec_f64)
            .any(|value| value != 0.0)
    );
    assert!(
        checkpoint_second_moments
            .values()
            .flat_map(TensorData::to_vec_f64)
            .any(|value| value != 0.0)
    );
    assert!(
        checkpoint_accumulators
            .values()
            .flat_map(TensorData::to_vec_f64)
            .any(|value| value != 0.0)
    );
    assert_eq!(checkpoint.info().capture_identity(), capture_identity);
    assert_eq!(checkpoint.info().replay_step(), 6);
    assert_eq!(checkpoint.info().optimizer_step(), 1);
    assert_eq!(checkpoint.info().accumulation_index(), 1);
    assert_eq!(checkpoint.info().discarded_microbatches(), 2);
    assert_eq!(checkpoint.info().dropout_block_counter(), Some(72));

    let destination = BufferedTinyCausalTransformer::new(0xdecafbad).unwrap();
    let tied = destination.transformer.tokens.weight.clone();
    let tied_identity = tied.id();
    assert_ne!(
        tied.value().unwrap(),
        checkpoint_parameters["tokens.weight"]
    );
    let policy_frozen = destination
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .find_map(|(name, parameter)| (name == POLICY_FROZEN).then_some(parameter))
        .unwrap();
    assert_ne!(policy_frozen.value().unwrap(), source_policy_frozen);
    policy_frozen.replace(source_policy_frozen).unwrap();
    let policy_frozen_before = policy_frozen.snapshot().unwrap();
    let inherent_frozen = destination.transformer.frozen_scale.clone();
    let inherent_frozen_before = inherent_frozen.snapshot().unwrap();
    let running_marker = destination.running_marker.clone();
    running_marker.replace(TensorData::scalar(29.0)).unwrap();
    let running_marker_before = running_marker.snapshot().unwrap();

    let mut destination_parameters = BTreeMap::new();
    let mut destination_versions = BTreeMap::new();
    for (ordinal, (name, parameter)) in destination
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .filter(|(name, _)| name != POLICY_FROZEN)
        .enumerate()
    {
        let replacement = TensorData::full_with_dtype(
            parameter.shape().unwrap(),
            Scalar::F(20.0 + ordinal as f64),
            parameter.dtype().unwrap(),
        )
        .unwrap();
        parameter.replace(replacement.clone()).unwrap();
        destination_versions.insert(name.clone(), parameter.version().unwrap());
        assert!(destination_parameters.insert(name, replacement).is_none());
    }
    assert_eq!(
        destination_parameters.keys().collect::<Vec<_>>(),
        checkpoint_parameters.keys().collect::<Vec<_>>()
    );
    for (name, value) in &destination_parameters {
        assert_ne!(
            value, &checkpoint_parameters[name],
            "{name} must be restored"
        );
    }

    let resumed_plan = plan.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(
        builds.get(),
        1,
        "checkpoint restore must not rebuild the graph"
    );
    assert_eq!(
        plan.step_count(),
        0,
        "borrowed source plan must stay unchanged"
    );
    assert_eq!(resumed_plan.capture_identity(), capture_identity);
    assert_eq!(
        resumed_plan.flush_capture_identity(),
        flush_capture_identity
    );
    assert_eq!(resumed_plan.step_count(), 6);
    assert_eq!(resumed_plan.captured_multi_step_lr(), Some(&schedule));
    let mut resumed = resumed_plan.prepare(&target).unwrap();
    assert_eq!(resumed.capture_identity(), capture_identity);
    assert_eq!(resumed.step_count(), 6);
    assert_eq!(resumed.optimizer_step().unwrap(), 1);
    assert_eq!(resumed.accumulation_index().unwrap(), 1);
    assert_eq!(resumed.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    assert_eq!(resumed.captured_multi_step_lr(), Some(&schedule));
    assert_eq!(
        resumed.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );
    let restored_before_wrong_entrypoint = resumed.checkpoint().unwrap();
    assert!(resumed.step_batch(masked_batch(7), 0.05).is_err());
    assert_eq!(
        resumed.checkpoint().unwrap(),
        restored_before_wrong_entrypoint
    );
    assert_eq!(
        checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
        72
    );
    assert_eq!(
        resumed.parameter_snapshots().unwrap(),
        checkpoint_parameters
    );
    assert_eq!(
        resumed.first_moment_snapshots().unwrap(),
        checkpoint_first_moments
    );
    assert_eq!(
        resumed.second_moment_snapshots().unwrap(),
        checkpoint_second_moments
    );
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        checkpoint_accumulators
    );
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    assert_eq!(
        policy_frozen.snapshot().unwrap().data,
        policy_frozen_before.data
    );
    assert_eq!(
        policy_frozen.version().unwrap(),
        policy_frozen_before.version
    );
    assert_eq!(
        running_marker.snapshot().unwrap().data,
        running_marker_before.data
    );
    assert_eq!(
        running_marker.version().unwrap(),
        running_marker_before.version
    );

    for replay in 7..=9 {
        let expected = uninterrupted
            .step_batch_scheduled(masked_batch(replay))
            .unwrap();
        let actual = resumed.step_batch_scheduled(masked_batch(replay)).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.did_update(), expected.did_update());
        assert_eq!(actual.did_update(), replay == 8);
        assert_eq!(actual.capture_identity(), capture_identity);
        assert_eq!(
            checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
            checkpoint_dropout_block_counter(&uninterrupted.checkpoint().unwrap())
        );
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            uninterrupted.parameter_snapshots().unwrap()
        );
        assert_eq!(
            resumed.first_moment_snapshots().unwrap(),
            uninterrupted.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.second_moment_snapshots().unwrap(),
            uninterrupted.second_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.gradient_accumulator_snapshots().unwrap(),
            uninterrupted.gradient_accumulator_snapshots().unwrap()
        );
        assert_eq!(
            resumed.checkpoint().unwrap(),
            uninterrupted.checkpoint().unwrap()
        );
    }
    assert_eq!(resumed.step_count(), 9);
    assert_eq!(resumed.optimizer_step().unwrap(), 2);
    assert_eq!(resumed.accumulation_index().unwrap(), 1);
    assert_eq!(
        checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
        108
    );

    let expected_flush = uninterrupted.flush_partial_window_scheduled().unwrap();
    let actual_flush = resumed.flush_partial_window_scheduled().unwrap();
    assert_eq!(actual_flush.flushed_microbatches(), 1);
    assert_eq!(
        actual_flush.flushed_microbatches(),
        expected_flush.flushed_microbatches()
    );
    assert_eq!(
        actual_flush.optimizer_step(),
        expected_flush.optimizer_step()
    );
    assert_eq!(actual_flush.did_update(), expected_flush.did_update());
    assert_eq!(resumed.optimizer_step().unwrap(), 3);
    assert_eq!(resumed.accumulation_index().unwrap(), 0);
    assert_eq!(resumed.step_count(), 9);
    assert_eq!(
        checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
        108
    );
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );

    let final_parameters = resumed.parameter_snapshots().unwrap();
    let final_checkpoint = resumed.checkpoint().unwrap();
    assert_eq!(final_checkpoint, uninterrupted.checkpoint().unwrap());
    assert!(resumed.publish_parameters(&destination).unwrap().is_clean());
    assert_eq!(resumed.checkpoint().unwrap(), final_checkpoint);
    assert_eq!(destination.transformer.tokens.weight.id(), tied_identity);
    assert_eq!(destination.transformer.tokens.weight.id(), tied.id());
    let mut tied_alias_identity = None;
    destination.visit("", &mut |name, parameter, _| {
        if name == "lm_head.weight" {
            tied_alias_identity = Some(parameter.id());
        }
    });
    assert_eq!(tied_alias_identity, Some(tied_identity));
    for (name, parameter) in destination.trainable_parameters().unwrap() {
        if name == POLICY_FROZEN {
            assert_eq!(
                parameter.snapshot().unwrap().data,
                policy_frozen_before.data
            );
            assert_eq!(parameter.version().unwrap(), policy_frozen_before.version);
            assert_eq!(parameter.id(), policy_frozen_before.identity);
            assert_eq!(parameter.is_trainable(), policy_frozen_before.trainable);
        } else {
            assert_eq!(parameter.value().unwrap(), final_parameters[&name]);
            assert_eq!(
                parameter.version().unwrap(),
                destination_versions[&name] + 1
            );
        }
    }
    let inherent_frozen_after = inherent_frozen.snapshot().unwrap();
    assert_eq!(inherent_frozen_after.data, inherent_frozen_before.data);
    assert_eq!(
        inherent_frozen_after.version,
        inherent_frozen_before.version
    );
    assert_eq!(
        inherent_frozen_after.identity,
        inherent_frozen_before.identity
    );
    assert_eq!(
        inherent_frozen_after.trainable,
        inherent_frozen_before.trainable
    );
    let running_marker_after = running_marker.snapshot().unwrap();
    assert_eq!(running_marker_after.data, running_marker_before.data);
    assert_eq!(running_marker_after.version, running_marker_before.version);
    assert_eq!(
        running_marker_after.identity,
        running_marker_before.identity
    );
    assert_eq!(
        running_marker_after.trainable,
        running_marker_before.trainable
    );
    let final_mean_sparse_loss = evaluate_mean_masked_sparse_loss(&destination.transformer);
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "masked compile-once causal Transformer loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
}

#[test]
fn owned_compiled_transformer_flushes_a_partial_window_and_resumes_exactly() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let frozen_before = model.frozen_scale.snapshot().unwrap();
    let plan = owned_compiled_transformer(model);
    let flush_identity = plan.flush_capture_identity().unwrap();
    let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    let initial_parameters = session.parameter_snapshots().unwrap();
    for replay in 1..=2 {
        let step = session.step(batch(replay), learning_rate()).unwrap();
        assert!(!step.did_update());
    }
    assert_eq!(
        checkpoint_dropout_block_counter(&session.checkpoint().unwrap()),
        24
    );
    assert_eq!(session.parameter_snapshots().unwrap(), initial_parameters);
    let flush = session.flush_partial_window(learning_rate()).unwrap();
    assert!(flush.did_update());
    assert_eq!(flush.flushed_microbatches(), 2);
    assert_eq!(flush.optimizer_step(), 1);
    assert_eq!(session.step_count(), 2);
    assert_eq!(session.optimizer_step().unwrap(), 1);
    assert_eq!(session.accumulation_index().unwrap(), 0);
    assert_eq!(
        checkpoint_dropout_block_counter(&session.checkpoint().unwrap()),
        24
    );
    assert_ne!(session.parameter_snapshots().unwrap(), initial_parameters);
    assert!(
        session
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .all(|value| value.to_vec_f64().into_iter().all(|lane| lane == 0.0))
    );

    let checkpoint = session.checkpoint().unwrap();
    let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], "rustgrad-compiled-adamw-v5");
    let checkpoint_info = checkpoint.info();
    assert_eq!(checkpoint_info.replay_step(), 2);
    assert_eq!(checkpoint_info.optimizer_step(), 1);
    assert_eq!(checkpoint_info.gradient_accumulation_steps(), 3);
    assert_eq!(checkpoint_info.accumulation_index(), 0);
    assert_eq!(checkpoint_info.discarded_microbatches(), 0);
    assert_eq!(checkpoint_info.flushed_window_count(), 1);
    assert_eq!(checkpoint_info.flushed_microbatch_count(), 2);
    assert_eq!(
        checkpoint_info.flush_capture_identity(),
        Some(flush_identity)
    );
    assert_eq!(checkpoint_info.dropout_block_counter(), Some(24));

    let fresh = TinyCausalTransformer::new(7).unwrap();
    let tied_identity = fresh.tokens.weight.id();
    let fresh_frozen = fresh.frozen_scale.clone();
    let resumed = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        fresh,
        &checkpoint,
        build,
    )
    .unwrap();
    assert_eq!(resumed.flush_capture_identity(), Some(flush_identity));
    let mut resumed = resumed.prepare(&CpuSessionTarget::new()).unwrap();
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    assert_eq!(
        checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
        24
    );
    let continued = resumed.step(batch(3), learning_rate()).unwrap();
    assert_eq!(continued.step(), 3);
    assert_eq!(continued.optimizer_step(), 1);
    assert_eq!(continued.accumulation_index(), 1);
    assert_eq!(
        checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
        36
    );
    let final_parameters = resumed.parameter_snapshots().unwrap();
    let (model, finished_checkpoint) = resumed.finish_with_checkpoint().unwrap();
    assert_eq!(model.tokens.weight.id(), tied_identity);
    assert_eq!(
        model.tokens.weight.value().unwrap(),
        final_parameters["tokens.weight"]
    );
    assert!(
        !model
            .state_dict()
            .unwrap()
            .tensors()
            .contains_key("lm_head.weight")
    );
    let frozen_after = fresh_frozen.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    assert_eq!(finished_checkpoint.info().replay_step(), 3);
    assert_eq!(finished_checkpoint.info().optimizer_step(), 1);
    assert_eq!(finished_checkpoint.info().flushed_window_count(), 1);
    assert_eq!(finished_checkpoint.info().dropout_block_counter(), Some(36));
}

#[test]
fn compiled_transformer_dropout_is_keyed_replay_varying_and_zero_grad_is_not_a_draw() {
    let left_model = TinyCausalTransformer::new(7).unwrap();
    let right_model = TinyCausalTransformer::new(7).unwrap();
    let mut left = compiled_transformer(&left_model).prepare_cpu().unwrap();
    let mut right = compiled_transformer(&right_model).prepare_cpu().unwrap();
    let mut replay_losses = Vec::new();

    for replay in 1..=8 {
        let left_step = left.step(batch(replay), TensorData::scalar(0.0)).unwrap();
        let right_step = right.step(batch(replay), TensorData::scalar(0.0)).unwrap();
        assert_eq!(left_step.loss(), right_step.loss());
        assert_eq!(left_step.outputs(), right_step.outputs());
        replay_losses.push(left_step.loss().clone());
        assert_eq!(left.dropout_block_counter().unwrap(), Some(replay * 12));
        assert_eq!(right.dropout_block_counter().unwrap(), Some(replay * 12));
    }
    assert_ne!(replay_losses[0], replay_losses[1]);
    assert_eq!(left.dropout_block_counter().unwrap(), Some(96));
    assert_eq!(right.dropout_block_counter().unwrap(), Some(96));
    let before = left.dropout_block_counter().unwrap();
    assert_eq!(left.zero_grad().unwrap().discarded_microbatches(), 2);
    assert_eq!(left.dropout_block_counter().unwrap(), before);
}

#[test]
fn compiled_transformer_plan_is_strictly_renderable_for_metal() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let policy = config();
    assert_eq!(
        policy.weight_decay_exclusions().collect::<Vec<_>>(),
        WEIGHT_DECAY_EXCLUSIONS
    );
    let exclusions = policy
        .weight_decay_exclusions()
        .collect::<std::collections::BTreeSet<_>>();
    let decayed = model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| !exclusions.contains(name.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        decayed,
        vec![
            "block.ff1.0",
            "block.ff2.0",
            "block.key.0",
            "block.out.0",
            "block.query.0",
            "block.value.0",
            "tokens.weight",
        ]
    );
    let compiled = compiled_transformer(&model);
    assert_eq!(compiled.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(compiled.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    assert_eq!(compiled.loss_scale(), 128.0);
    assert_eq!(compiled.dropout_config(), Some(dropout_config()));
    assert_eq!(compiled.dropout_blocks_per_replay(), Some(12));
    let parameters = compiled
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .parameter_snapshots()
        .unwrap();
    let parameter_count = parameters.len();
    let parameter_bytes = parameters
        .values()
        .map(|value| value.shape().numel().unwrap() * value.dtype().itemsize())
        .sum::<usize>();
    let plan = compiled.metal_plan(metal_renderer()).unwrap();

    assert_eq!(plan.capture_identity(), compiled.capture_identity());
    assert_eq!(plan.loss_scale(), 128.0);
    assert_eq!(plan.step_count(), 0);
    assert_eq!(plan.summary().fallback_count, 0);
    assert_eq!(plan.summary().state_pair_count, parameter_count * 4 + 3);
    assert_eq!(plan.summary().state_bank_count, 2);
    assert_eq!(
        plan.summary().logical_state_bytes,
        parameter_bytes * 4 + 3 * 8
    );
    assert_eq!(plan.summary().logical_state_bytes, 1_048);
    assert_eq!(plan.summary().state_device_bytes, 2_096);
    assert_eq!(plan.summary().requested_output_count, 1);
    assert!(plan.summary().nonzero_item_count > 0);
    assert_strict_dropout_kernels(&plan);
    let authenticated = plan
        .rendered_items()
        .filter(|item| item.entry.starts_with("rg_metal_training_host_"))
        .collect::<Vec<_>>();
    assert_eq!(authenticated.len(), 4);
    assert!(authenticated.iter().all(|item| {
        item.transaction.is_none()
            && item.indexed_movement().is_none()
            && !item.source.contains("rg_status")
    }));
    assert!(
        plan.rendered_items()
            .all(|item| item.transaction.is_none() && item.indexed_movement().is_none())
    );
    assert_eq!(
        plan.rendered_items().len(),
        plan.summary().nonzero_item_count
    );
}

#[test]
fn compiled_transformer_freezing_reduces_the_authenticated_cpu_and_metal_frontier() {
    let policy = config().with_frozen_parameters(["block.ff1.0"]).unwrap();
    assert_eq!(
        policy.frozen_parameters().collect::<Vec<_>>(),
        ["block.ff1.0"]
    );
    let model = TinyCausalTransformer::new(7).unwrap();
    let frozen_weight = model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .find_map(|(name, parameter)| (name == "block.ff1.0").then_some(parameter))
        .unwrap();
    let frozen_before = frozen_weight.snapshot().unwrap();
    let compiled = CompiledAdamWPlan::compile_module_with_dropout(
        policy.clone(),
        dropout_config(),
        &model,
        build,
    )
    .unwrap();
    let mut cpu = compiled.prepare_cpu().unwrap();
    let parameters = cpu.parameter_snapshots().unwrap();
    assert_eq!(parameters.len(), 18);
    assert!(!parameters.contains_key("block.ff1.0"));
    assert!(parameters.contains_key("tokens.weight"));
    assert!(!parameters.contains_key("lm_head.weight"));
    assert!(
        !cpu.first_moment_snapshots()
            .unwrap()
            .contains_key("block.ff1.0")
    );
    assert!(
        !cpu.second_moment_snapshots()
            .unwrap()
            .contains_key("block.ff1.0")
    );
    assert!(
        !cpu.gradient_accumulator_snapshots()
            .unwrap()
            .contains_key("block.ff1.0")
    );
    cpu.step(batch(1), learning_rate()).unwrap();
    let checkpoint = cpu.checkpoint().unwrap();
    let frozen_after = frozen_weight.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    assert_eq!(frozen_after.identity, frozen_before.identity);
    assert!(frozen_after.trainable);

    let fresh = TinyCausalTransformer::new(7).unwrap();
    let resumed = CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
        policy,
        dropout_config(),
        &fresh,
        &checkpoint,
        build,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap();
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    assert!(
        CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
            config(),
            dropout_config(),
            &fresh,
            &checkpoint,
            build,
        )
        .is_err(),
        "restore must authenticate the reduced parameter topology"
    );

    let metal = compiled.metal_plan(metal_renderer()).unwrap();
    assert_eq!(metal.summary().fallback_count, 0);
    assert_eq!(metal.summary().state_pair_count, 75);
    assert_eq!(metal.summary().logical_state_bytes, 920);
    assert_eq!(metal.summary().state_device_bytes, 1_840);
}

#[test]
fn compiled_transformer_frozen_embedding_uses_one_forward_only_host_gather() {
    let policy = config().with_frozen_parameters(["tokens.weight"]).unwrap();
    let model = TinyCausalTransformer::new(7).unwrap();
    let tied_identity = model.tokens.weight.id();
    let frozen_before = model.tokens.weight.snapshot().unwrap();
    let compiled = CompiledAdamWPlan::compile_module_with_dropout(
        policy.clone(),
        dropout_config(),
        &model,
        build,
    )
    .unwrap();
    let mut cpu = compiled.prepare_cpu().unwrap();
    assert_eq!(cpu.parameter_snapshots().unwrap().len(), 18);
    assert!(
        !cpu.parameter_snapshots()
            .unwrap()
            .contains_key("tokens.weight")
    );
    assert!(
        !cpu.first_moment_snapshots()
            .unwrap()
            .contains_key("tokens.weight")
    );
    assert!(
        !cpu.second_moment_snapshots()
            .unwrap()
            .contains_key("tokens.weight")
    );
    assert!(
        !cpu.gradient_accumulator_snapshots()
            .unwrap()
            .contains_key("tokens.weight")
    );
    cpu.step(batch(1), learning_rate()).unwrap();
    let checkpoint = cpu.checkpoint().unwrap();
    assert_eq!(model.tokens.weight.id(), tied_identity);
    let frozen_after = model.tokens.weight.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    assert_eq!(frozen_after.identity, frozen_before.identity);
    assert!(frozen_after.trainable);

    let metal = compiled.metal_plan(metal_renderer()).unwrap();
    assert_eq!(metal.summary().fallback_count, 0);
    assert_eq!(metal.summary().state_pair_count, 75);
    assert_eq!(metal.summary().logical_state_bytes, 952);
    assert_eq!(metal.summary().state_device_bytes, 1_904);
    assert_eq!(
        metal
            .rendered_items()
            .filter(|item| item.entry == "rg_metal_host_gather_fixed_f32_i32")
            .count(),
        1,
        "the frozen embedding must retain exactly its forward Gather"
    );
    assert_eq!(
        metal
            .rendered_items()
            .filter(|item| item.entry.starts_with("rg_metal_training_host_"))
            .count(),
        2,
        "the target loss must retain its trainable Gather/ScatterAdd pair"
    );
    assert!(metal.rendered_items().all(|item| {
        item.transaction.is_none()
            && item.indexed_movement().is_none()
            && !item.source.contains("rg_status")
    }));

    let fresh = TinyCausalTransformer::new(7).unwrap();
    let resumed = CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
        policy,
        dropout_config(),
        &fresh,
        &checkpoint,
        build,
    )
    .unwrap();
    let resumed_metal = resumed.metal_plan(metal_renderer()).unwrap();
    assert_eq!(resumed_metal.summary().fallback_count, 0);
    assert_eq!(resumed_metal.summary().state_pair_count, 75);
    assert_eq!(
        resumed.prepare_cpu().unwrap().checkpoint().unwrap(),
        checkpoint
    );
    let restored_frozen = fresh.tokens.weight.snapshot().unwrap();
    assert_eq!(restored_frozen.data, frozen_before.data);
    assert_eq!(restored_frozen.version, frozen_before.version);
    assert!(restored_frozen.trainable);
}

#[test]
fn protected_live_metal_workflow_runs_the_exact_compiled_training_acceptance() {
    let workflow = include_str!("../.github/workflows/metal-live.yml");
    for required in [
        "RUSTGRAD_METAL_TRAINING_EVIDENCE_PATH:",
        "metal-live-compiled-training-v7.json",
        "Train and resume the compiled causal Transformer on Metal",
        "cargo test --release --test compiled_transformer_training",
        "live_metal_compiled_causal_transformer_training_resumes_exactly",
        "${{ env.RUSTGRAD_METAL_TRAINING_EVIDENCE_PATH }}",
    ] {
        assert!(
            workflow.contains(required),
            "protected live Metal workflow is missing {required:?}"
        );
    }
    for required in [
        "run_gguf:",
        "default: false",
        "type: boolean",
        "if: ${{ inputs.run_gguf }}",
    ] {
        assert!(
            workflow.contains(required),
            "protected live Metal workflow is missing the optional GGUF contract {required:?}"
        );
    }
    let (training_job, gguf_job) = workflow
        .split_once("  live-metal-llama:")
        .expect("workflow must keep a separate GGUF job");
    assert!(!training_job.contains("RUSTGRAD_METAL_LLAMA_"));
    assert!(gguf_job.contains("RUSTGRAD_METAL_LLAMA_GGUF_PATH:"));
    assert_eq!(
        workflow
            .matches("workflow dispatch revision does not match expected_sha")
            .count(),
        2,
        "each runnable Metal job must authenticate its dispatch revision"
    );
    assert_eq!(
        workflow
            .matches("checked-out revision does not match expected_sha")
            .count(),
        2,
        "each runnable Metal job must authenticate its checkout"
    );
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct LiveTrainingTotals {
    kernel_launch_count: usize,
    command_submission_count: usize,
    command_wait_count: usize,
    transient_h2d_calls: usize,
    transient_h2d_bytes: usize,
    retained_d2h_calls: usize,
    retained_d2h_bytes: usize,
    observed_training_invocations: usize,
    device_only_training_invocations: usize,
}

#[cfg(target_os = "macos")]
impl LiveTrainingTotals {
    fn record(
        &mut self,
        report: &MetalDeviceRunReport,
        observed: bool,
        state_pair_count: usize,
        logical_state_bytes: usize,
        state_work_items: usize,
        planned_kernel_count: usize,
        expected_command_count: usize,
        expected_transient_h2d_calls: usize,
        expected_transient_h2d_bytes: usize,
    ) {
        let output_multiplier = if observed { 1 } else { 0 };
        assert_eq!(report.output_count, output_multiplier);
        assert_eq!(report.retained_d2h_calls, output_multiplier);
        assert_eq!(report.retained_d2h_bytes, output_multiplier * 4);
        assert_eq!(report.committed_state_pair_count, state_pair_count);
        assert_eq!(report.committed_state_bytes, logical_state_bytes);
        assert_eq!(report.committed_state_work_items, state_work_items);
        assert_eq!(report.kernel_launch_count, planned_kernel_count);
        assert_eq!(report.command_submission_count, expected_command_count);
        assert_eq!(report.command_wait_count, expected_command_count);
        assert_eq!(report.transient_h2d_calls, expected_transient_h2d_calls);
        assert_eq!(report.transient_h2d_bytes, expected_transient_h2d_bytes);
        self.kernel_launch_count += report.kernel_launch_count;
        self.command_submission_count += report.command_submission_count;
        self.command_wait_count += report.command_wait_count;
        self.transient_h2d_calls += report.transient_h2d_calls;
        self.transient_h2d_bytes += report.transient_h2d_bytes;
        self.retained_d2h_calls += report.retained_d2h_calls;
        self.retained_d2h_bytes += report.retained_d2h_bytes;
        if observed {
            self.observed_training_invocations += 1;
        } else {
            self.device_only_training_invocations += 1;
        }
    }
}

#[cfg(target_os = "macos")]
fn assert_live_scalar_close(label: &str, actual: f64, expected: f64) -> f64 {
    let absolute_error = (actual - expected).abs();
    let tolerance = 1e-5 + 1e-4 * actual.abs().max(expected.abs());
    assert!(
        absolute_error <= tolerance,
        "{label} mismatch: actual={actual}, expected={expected}, error={absolute_error}, tolerance={tolerance}"
    );
    absolute_error
}

#[cfg(target_os = "macos")]
fn assert_live_tensor_maps_close(
    label: &str,
    actual: &BTreeMap<String, TensorData>,
    expected: &BTreeMap<String, TensorData>,
) -> (usize, f64) {
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    let mut lanes = 0;
    let mut max_absolute_error = 0.0_f64;
    for (name, actual) in actual {
        let expected = &expected[name];
        assert_eq!(actual.shape(), expected.shape(), "{label} {name} shape");
        assert_eq!(actual.dtype(), expected.dtype(), "{label} {name} dtype");
        for index in 0..actual.len() {
            let actual = actual.scalar_at(index).as_f64();
            let expected = expected.scalar_at(index).as_f64();
            let absolute_error =
                assert_live_scalar_close(&format!("{label} {name}[{index}]"), actual, expected);
            max_absolute_error = max_absolute_error.max(absolute_error);
            lanes += 1;
        }
    }
    (lanes, max_absolute_error)
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires the manual self-hosted Apple-GPU lane"]
fn live_metal_compiled_causal_transformer_training_resumes_exactly() {
    let expected_sha = env::var("RUSTGRAD_METAL_EXPECTED_SHA")
        .expect("the live lane must provide RUSTGRAD_METAL_EXPECTED_SHA");
    assert!(
        expected_sha.len() == 40
            && expected_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "the live evidence revision must be a lowercase full Git SHA"
    );
    let evidence_path = PathBuf::from(
        env::var_os("RUSTGRAD_METAL_TRAINING_EVIDENCE_PATH")
            .expect("the live lane must provide RUSTGRAD_METAL_TRAINING_EVIDENCE_PATH"),
    );

    let runtime = MetalRuntime::load().expect("the live lane requires the native Metal runtime");
    let mut devices = match runtime
        .discover()
        .expect("native Metal discovery must complete")
    {
        MetalDiscovery::Devices(devices) => devices,
        MetalDiscovery::NoDevices => panic!("the live Metal lane requires a process-visible GPU"),
    };
    assert!(
        !devices.is_empty(),
        "typed discovery returned an empty device set"
    );
    let device = devices.remove(0);
    let device_info = device.info().clone();

    let policy = frozen_embedding_config();
    let model = TinyCausalTransformer::new(7).unwrap();
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&model);
    let initial_module_state = model.state_dict().unwrap();
    let seed = CompiledModuleAdamWPlan::compile_with_dropout(
        policy.clone(),
        dropout_config(),
        model,
        build,
    )
    .unwrap()
    .with_evaluation(build_evaluation)
    .unwrap();
    let capture_identity = seed.capture_identity();
    let cpu_model = TinyCausalTransformer::new(7).unwrap();
    let cpu_seed = CompiledModuleAdamWPlan::compile_with_dropout(
        policy.clone(),
        dropout_config(),
        cpu_model,
        build,
    )
    .unwrap()
    .with_evaluation(build_evaluation)
    .unwrap();
    assert_eq!(cpu_seed.capture_identity(), capture_identity);
    let mut cpu_primary = cpu_seed.prepare(&CpuSessionTarget::new()).unwrap();
    let target = MetalSessionTarget::new(device.clone(), 64)
        .expect("selected device must produce its exact renderer identity")
        .with_scoreboard(
            MetalScoreboardContext::new(
                "tiny-causal-transformer-compiled-adamw",
                expected_sha.clone(),
                "protected live Metal",
            )
            .unwrap(),
        );
    let summary = seed
        .metal_summary(target.renderer().clone())
        .expect("the complete training capture must be entirely Metal-admitted");
    assert_eq!(summary.fallback_count, 0);
    assert!(summary.nonzero_item_count > 0);
    let state_pair_count = summary.state_pair_count;
    let logical_state_bytes = summary.logical_state_bytes;
    assert_eq!(state_pair_count, 75);
    assert_eq!(logical_state_bytes, 952);
    assert_eq!(summary.state_bank_count, 2);
    assert_eq!(summary.state_device_bytes, 1_904);
    let planned_kernel_count = summary.nonzero_item_count;
    let command_count_per_invocation = 1;
    let mut uninterrupted = seed
        .prepare(&target)
        .expect("live Metal preparation must compile, allocate, and upload training state");
    for report in uninterrupted.evaluation_preparation_reports().unwrap() {
        assert_eq!(report.resident_h2d_calls, 0);
        assert_eq!(report.resident_h2d_bytes, 0);
        assert_eq!(report.initial_state_h2d_calls, 0);
        assert_eq!(report.initial_state_h2d_bytes, 0);
    }
    let [evaluation_false_summary, evaluation_true_summary] =
        uninterrupted.evaluation_summaries().unwrap();
    let evaluation_false_summary = (*evaluation_false_summary).clone();
    let evaluation_true_summary = (*evaluation_true_summary).clone();
    assert_eq!(evaluation_false_summary, evaluation_true_summary);
    assert_eq!(evaluation_false_summary.transient_input_names.len(), 2);
    assert_eq!(evaluation_false_summary.transient_input_bytes, 48);
    assert_eq!(evaluation_false_summary.requested_output_count, 2);
    assert_eq!(evaluation_false_summary.fallback_count, 0);
    let evaluation_kernel_count = evaluation_false_summary.nonzero_item_count;
    let evaluation_zero_item_count = evaluation_false_summary.zero_item_count;
    let state_work_items = uninterrupted
        .metal_session()
        .state_inputs()
        .iter()
        .map(|input| input.desc.shape.numel().unwrap())
        .sum::<usize>();
    assert_eq!(state_work_items, 235);
    let deployment_identity = uninterrupted
        .execution_scoreboard_report()
        .unwrap()
        .expect("the prepared owned session must expose its deployment evidence")
        .deployment_identity;
    let authenticated_host_indexed_movement_item_count = uninterrupted
        .metal_session()
        .compiled_kernels()
        .filter(|item| item.entry.starts_with("rg_metal_training_host_"))
        .count();
    assert_eq!(authenticated_host_indexed_movement_item_count, 2);
    let authenticated_frozen_host_gather_item_count = uninterrupted
        .metal_session()
        .compiled_kernels()
        .filter(|item| item.entry == "rg_metal_host_gather_fixed_f32_i32")
        .count();
    assert_eq!(authenticated_frozen_host_gather_item_count, 1);
    assert!(
        uninterrupted
            .metal_session()
            .compiled_kernels()
            .all(|item| { item.transaction.is_none() && item.indexed_movement().is_none() })
    );
    assert_eq!(
        uninterrupted
            .metal_session()
            .compiled_kernels()
            .filter(|item| item.entry.starts_with("rg_metal_threefry_"))
            .count(),
        2
    );
    assert_eq!(
        uninterrupted
            .metal_session()
            .compiled_kernels()
            .filter(|item| item.entry == "rg_metal_portable_bitcast")
            .count(),
        4
    );
    let transient_h2d_calls_per_invocation = uninterrupted.metal_session().transient_inputs().len();
    let token_input = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .find(|input| input.name == "tokens")
        .expect("the fixed token minibatch must remain transient");
    assert_eq!(token_input.desc.shape, Shape::new([BATCH, TIME]));
    assert_eq!(token_input.desc.dtype, DType::I32);
    assert_eq!(token_input.desc.bytes, TOKEN_COUNT * DType::I32.itemsize());
    let target_input = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .find(|input| input.name == "targets")
        .expect("the fixed target minibatch must remain transient");
    assert_eq!(target_input.desc.shape, Shape::new([BATCH, TIME]));
    assert_eq!(target_input.desc.dtype, DType::I32);
    assert_eq!(target_input.desc.bytes, TOKEN_COUNT * DType::I32.itemsize());
    let learning_rate_input = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .find(|input| input.name == "__rustgrad_compiled_training_learning_rate")
        .expect("the scalar learning rate must remain transient");
    assert_eq!(learning_rate_input.desc.shape, Shape::new([]));
    assert_eq!(learning_rate_input.desc.dtype, DType::F32);
    assert_eq!(learning_rate_input.desc.bytes, DType::F32.itemsize());
    let transient_h2d_bytes_per_invocation = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .map(|input| input.desc.bytes)
        .sum::<usize>();
    assert_eq!(transient_h2d_calls_per_invocation, 3);
    assert_eq!(transient_h2d_bytes_per_invocation, 52);
    assert_eq!(uninterrupted.loss_scale(), 128.0);
    assert_eq!(
        uninterrupted.gradient_accumulation_steps(),
        ACCUMULATION_STEPS
    );
    assert_eq!(uninterrupted.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    assert_eq!(uninterrupted.step_count(), 0);
    assert_eq!(uninterrupted.optimizer_step().unwrap(), 0);
    assert_eq!(uninterrupted.accumulation_index().unwrap(), 0);
    let initial_parameters = uninterrupted.parameter_snapshots().unwrap();
    assert_eq!(initial_parameters.len(), 18);
    assert!(!initial_parameters.contains_key("tokens.weight"));
    assert!(!initial_parameters.contains_key("lm_head.weight"));
    assert_eq!(
        cpu_primary.parameter_snapshots().unwrap(),
        initial_parameters
    );
    assert!(
        !uninterrupted
            .first_moment_snapshots()
            .unwrap()
            .contains_key("tokens.weight")
    );
    assert!(
        !uninterrupted
            .second_moment_snapshots()
            .unwrap()
            .contains_key("tokens.weight")
    );
    let empty_accumulators = uninterrupted.gradient_accumulator_snapshots().unwrap();
    assert!(!empty_accumulators.contains_key("tokens.weight"));
    assert_eq!(
        cpu_primary.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );

    let mut totals = LiveTrainingTotals::default();
    for index in 0..4u64 {
        let cpu_result = cpu_primary.step(batch(index + 1), learning_rate()).unwrap();
        if matches!(index, 1 | 2) {
            let result = uninterrupted
                .step_without_host_outputs(batch(index + 1), learning_rate())
                .unwrap();
            assert_eq!(result.step(), index + 1);
            assert_eq!(result.capture_identity(), capture_identity);
            assert_eq!(result.report().successful_invocation, index + 1);
            assert_eq!(result.optimizer_step(), 0);
            assert_eq!(
                result.accumulation_index(),
                if index < 2 { index + 1 } else { index - 1 }
            );
            assert!(!result.did_update());
            totals.record(
                result.report(),
                false,
                state_pair_count,
                logical_state_bytes,
                state_work_items,
                planned_kernel_count,
                command_count_per_invocation,
                transient_h2d_calls_per_invocation,
                transient_h2d_bytes_per_invocation,
            );
        } else {
            let result = uninterrupted
                .step(batch(index + 1), learning_rate())
                .unwrap();
            let actual_loss = result.loss().scalar_at(0).as_f64();
            let expected_loss = cpu_result.loss().scalar_at(0).as_f64();
            assert_live_scalar_close("observed training loss", actual_loss, expected_loss);
            assert_eq!(result.step(), index + 1);
            assert_eq!(result.capture_identity(), capture_identity);
            assert_eq!(result.report().successful_invocation, index + 1);
            assert_eq!(result.optimizer_step(), 0);
            assert_eq!(
                result.accumulation_index(),
                if index < 2 { index + 1 } else { index - 1 }
            );
            assert!(!result.did_update());
            totals.record(
                result.report(),
                true,
                state_pair_count,
                logical_state_bytes,
                state_work_items,
                planned_kernel_count,
                command_count_per_invocation,
                transient_h2d_calls_per_invocation,
                transient_h2d_bytes_per_invocation,
            );
        }
        assert_eq!(cpu_primary.step_count(), uninterrupted.step_count());
        assert_eq!(
            cpu_primary.optimizer_step().unwrap(),
            uninterrupted.optimizer_step().unwrap()
        );
        assert_eq!(
            cpu_primary.accumulation_index().unwrap(),
            uninterrupted.accumulation_index().unwrap()
        );
        if index == 1 {
            let parameters_before_reset = uninterrupted.parameter_snapshots().unwrap();
            let first_moments_before_reset = uninterrupted.first_moment_snapshots().unwrap();
            let second_moments_before_reset = uninterrupted.second_moment_snapshots().unwrap();
            assert_ne!(
                uninterrupted.gradient_accumulator_snapshots().unwrap(),
                empty_accumulators
            );
            let state_epoch = uninterrupted.metal_session().state_epoch();
            let checkpoint_before_reset = uninterrupted.checkpoint().unwrap();
            let dropout_counter_before_reset =
                checkpoint_dropout_block_counter(&checkpoint_before_reset);
            assert_eq!(dropout_counter_before_reset, 24);
            let scoreboard_before_reset = uninterrupted
                .execution_scoreboard_report()
                .unwrap()
                .unwrap();
            let evaluation_before_reset = uninterrupted.evaluate(batch(1)).unwrap();
            let evaluation_loss_before_reset = evaluation_before_reset.loss().clone();
            let evaluation_logits_before_reset =
                evaluation_before_reset.output("logits").unwrap().clone();
            assert_eq!(uninterrupted.checkpoint().unwrap(), checkpoint_before_reset);
            assert_eq!(
                uninterrupted
                    .execution_scoreboard_report()
                    .unwrap()
                    .unwrap(),
                scoreboard_before_reset
            );
            let reset = uninterrupted.zero_grad().unwrap();
            let cpu_reset = cpu_primary.zero_grad().unwrap();
            assert_eq!(reset.discarded_microbatches(), 2);
            assert_eq!(cpu_reset.discarded_microbatches(), 2);
            assert_eq!(uninterrupted.step_count(), 2);
            assert_eq!(uninterrupted.optimizer_step().unwrap(), 0);
            assert_eq!(uninterrupted.accumulation_index().unwrap(), 0);
            assert_ne!(uninterrupted.metal_session().state_epoch(), state_epoch);
            assert_eq!(
                uninterrupted
                    .execution_scoreboard_report()
                    .unwrap()
                    .unwrap(),
                scoreboard_before_reset
            );
            assert_eq!(
                checkpoint_dropout_block_counter(&uninterrupted.checkpoint().unwrap()),
                dropout_counter_before_reset
            );
            let evaluation_after_reset = uninterrupted.evaluate(batch(1)).unwrap();
            assert_eq!(evaluation_after_reset.loss(), &evaluation_loss_before_reset);
            assert_eq!(
                evaluation_after_reset.output("logits").unwrap(),
                &evaluation_logits_before_reset
            );
            assert_eq!(
                uninterrupted
                    .execution_scoreboard_report()
                    .unwrap()
                    .unwrap(),
                scoreboard_before_reset
            );
            assert_eq!(
                uninterrupted.gradient_accumulator_snapshots().unwrap(),
                empty_accumulators
            );
            assert_eq!(
                uninterrupted.parameter_snapshots().unwrap(),
                parameters_before_reset
            );
            assert_eq!(
                uninterrupted.first_moment_snapshots().unwrap(),
                first_moments_before_reset
            );
            assert_eq!(
                uninterrupted.second_moment_snapshots().unwrap(),
                second_moments_before_reset
            );
            let checkpoint_after_reset = uninterrupted.checkpoint().unwrap();
            let state_epoch = uninterrupted.metal_session().state_epoch();
            assert!(!uninterrupted.zero_grad().unwrap().did_discard());
            assert!(!cpu_primary.zero_grad().unwrap().did_discard());
            assert_eq!(uninterrupted.metal_session().state_epoch(), state_epoch);
            assert_eq!(uninterrupted.checkpoint().unwrap(), checkpoint_after_reset);
            assert_eq!(
                uninterrupted
                    .execution_scoreboard_report()
                    .unwrap()
                    .unwrap(),
                scoreboard_before_reset
            );
            assert_eq!(
                checkpoint_dropout_block_counter(&uninterrupted.checkpoint().unwrap()),
                dropout_counter_before_reset
            );
        }
    }
    assert_eq!(uninterrupted.optimizer_step().unwrap(), 0);
    assert_eq!(uninterrupted.accumulation_index().unwrap(), 2);
    let checkpoint = uninterrupted.checkpoint().unwrap();
    let cpu_checkpoint = cpu_primary.checkpoint().unwrap();
    let (partial_parameter_lanes, partial_parameter_error) = assert_live_tensor_maps_close(
        "partial parameters",
        &uninterrupted.parameter_snapshots().unwrap(),
        &cpu_primary.parameter_snapshots().unwrap(),
    );
    let (partial_accumulator_lanes, partial_accumulator_error) = assert_live_tensor_maps_close(
        "partial accumulators",
        &uninterrupted.gradient_accumulator_snapshots().unwrap(),
        &cpu_primary.gradient_accumulator_snapshots().unwrap(),
    );
    assert_eq!(partial_parameter_lanes, 58);
    assert_eq!(partial_accumulator_lanes, 58);
    let (checkpoint_state, checkpoint_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(checkpoint_metadata["format"], "rustgrad-compiled-adamw-v4");
    assert_eq!(checkpoint_metadata["replay_step"], "4");
    assert_eq!(checkpoint_metadata["optimizer_step"], "0");
    assert_eq!(checkpoint_metadata["gradient_accumulation_steps"], "3");
    assert_eq!(checkpoint_metadata["accumulation_index"], "2");
    assert_eq!(checkpoint_metadata["discarded_microbatch_count"], "2");
    assert_frozen_embedding_checkpoint_inventory(&checkpoint, &initial_parameters, 48);
    assert_eq!(
        checkpoint_state["dropout_block_counter"]
            .scalar_at(0)
            .as_u64(),
        48
    );
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let tied = resumed_model.tokens.weight.clone();
    let tied_before = tied.snapshot().unwrap();
    let frozen = resumed_model.frozen_scale.clone();
    let frozen_before = frozen.snapshot().unwrap();
    let before_versions = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let resumed_seed = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        policy.clone(),
        dropout_config(),
        resumed_model,
        &checkpoint,
        build,
    )
    .unwrap()
    .with_evaluation(build_evaluation)
    .unwrap();
    let cpu_resumed_model = TinyCausalTransformer::new(7).unwrap();
    let cpu_tied = cpu_resumed_model.tokens.weight.clone();
    let cpu_tied_before = cpu_tied.snapshot().unwrap();
    let cpu_resumed_seed = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        policy.clone(),
        dropout_config(),
        cpu_resumed_model,
        &cpu_checkpoint,
        build,
    )
    .unwrap()
    .with_evaluation(build_evaluation)
    .unwrap();
    assert_eq!(cpu_resumed_seed.capture_identity(), capture_identity);
    let mut cpu_resumed = cpu_resumed_seed.prepare(&CpuSessionTarget::new()).unwrap();
    assert_eq!(resumed_seed.step_count(), 4);
    let resumed_target = MetalSessionTarget::new(device, 64)
        .expect("selected device must retain its renderer identity")
        .with_scoreboard(
            MetalScoreboardContext::new(
                "tiny-causal-transformer-compiled-adamw-resume",
                expected_sha.clone(),
                "protected live Metal checkpoint resume",
            )
            .unwrap(),
        );
    assert_eq!(resumed_seed.capture_identity(), capture_identity);
    assert_eq!(
        resumed_seed
            .metal_summary(resumed_target.renderer().clone())
            .expect("the checkpoint-restored capture must remain entirely Metal-admitted")
            .fallback_count,
        0
    );
    let mut resumed = resumed_seed
        .prepare(&resumed_target)
        .expect("checkpoint-restored Metal preparation must succeed");
    for report in resumed.evaluation_preparation_reports().unwrap() {
        assert_eq!(report.resident_h2d_calls, 0);
        assert_eq!(report.resident_h2d_bytes, 0);
        assert_eq!(report.initial_state_h2d_calls, 0);
        assert_eq!(report.initial_state_h2d_bytes, 0);
    }
    let [resumed_false_summary, resumed_true_summary] = resumed.evaluation_summaries().unwrap();
    assert_eq!(resumed_false_summary, &evaluation_false_summary);
    assert_eq!(resumed_true_summary, &evaluation_true_summary);
    let resumed_deployment_identity = resumed
        .execution_scoreboard_report()
        .unwrap()
        .expect("the prepared restored session must expose its deployment evidence")
        .deployment_identity;
    assert_ne!(
        resumed_deployment_identity, deployment_identity,
        "the deployment identity must authenticate the checkpoint-restored state bytes"
    );
    assert_eq!(
        resumed.metal_session().transient_inputs(),
        uninterrupted.metal_session().transient_inputs()
    );
    assert_eq!(resumed.loss_scale(), 128.0);
    assert_eq!(resumed.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(resumed.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    assert_eq!(resumed.step_count(), 4);
    assert_eq!(resumed.optimizer_step().unwrap(), 0);
    assert_eq!(resumed.accumulation_index().unwrap(), 2);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    assert_eq!(cpu_resumed.step_count(), 4);
    assert_eq!(cpu_resumed.optimizer_step().unwrap(), 0);
    assert_eq!(cpu_resumed.accumulation_index().unwrap(), 2);
    assert_eq!(cpu_resumed.checkpoint().unwrap(), cpu_checkpoint);

    let scoreboard_before_flush = resumed.execution_scoreboard_report().unwrap().unwrap();
    let epoch_before_flush = resumed.metal_session().state_epoch();
    let dropout_before_flush = checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap());
    let flush_capture_identity = resumed.flush_capture_identity().unwrap();
    assert_eq!(
        cpu_resumed.flush_capture_identity(),
        Some(flush_capture_identity)
    );
    assert_eq!(
        cpu_primary.flush_capture_identity(),
        Some(flush_capture_identity)
    );
    let cpu_primary_flush = cpu_primary.flush_partial_window(learning_rate()).unwrap();
    let cpu_flush = cpu_resumed.flush_partial_window(learning_rate()).unwrap();
    let metal_flush = resumed.flush_partial_window(learning_rate()).unwrap();
    assert!(cpu_primary_flush.did_update());
    assert!(cpu_flush.did_update());
    assert!(metal_flush.did_update());
    assert_eq!(cpu_primary_flush.flushed_microbatches(), 2);
    assert_eq!(cpu_flush.flushed_microbatches(), 2);
    assert_eq!(metal_flush.flushed_microbatches(), 2);
    assert_eq!(cpu_flush.optimizer_step(), 1);
    assert_eq!(metal_flush.optimizer_step(), 1);
    let flush_report = metal_flush.report().unwrap().clone();
    assert_eq!(flush_report.successful_invocation, 1);
    assert_eq!(flush_report.transient_h2d_calls, 1);
    assert_eq!(flush_report.transient_h2d_bytes, 4);
    assert_eq!(flush_report.runtime_control_h2d_calls, 0);
    assert_eq!(flush_report.runtime_control_h2d_bytes, 0);
    assert_eq!(flush_report.retained_d2h_calls, 0);
    assert_eq!(flush_report.retained_d2h_bytes, 0);
    assert_eq!(flush_report.output_count, 0);
    assert!(flush_report.kernel_launch_count > 0);
    assert_eq!(flush_report.command_submission_count, 1);
    assert_eq!(flush_report.command_wait_count, 1);
    assert_eq!(flush_report.committed_state_pair_count, state_pair_count);
    assert_eq!(flush_report.committed_state_bytes, logical_state_bytes);
    assert_eq!(flush_report.committed_state_work_items, state_work_items);
    assert_eq!(resumed.metal_session().state_epoch(), !epoch_before_flush);
    assert_eq!(resumed.step_count(), 4);
    assert_eq!(cpu_resumed.step_count(), 4);
    assert_eq!(resumed.optimizer_step().unwrap(), 1);
    assert_eq!(cpu_resumed.optimizer_step().unwrap(), 1);
    assert_eq!(resumed.accumulation_index().unwrap(), 0);
    assert_eq!(cpu_resumed.accumulation_index().unwrap(), 0);
    assert_eq!(
        cpu_primary.checkpoint().unwrap(),
        cpu_resumed.checkpoint().unwrap()
    );
    assert_eq!(
        checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
        dropout_before_flush
    );
    assert_eq!(
        resumed.execution_scoreboard_report().unwrap().unwrap(),
        scoreboard_before_flush
    );
    let (parameter_lanes, parameter_max_absolute_error) = assert_live_tensor_maps_close(
        "flushed parameters",
        &resumed.parameter_snapshots().unwrap(),
        &cpu_resumed.parameter_snapshots().unwrap(),
    );
    let (first_moment_lanes, first_moment_max_absolute_error) = assert_live_tensor_maps_close(
        "flushed first moments",
        &resumed.first_moment_snapshots().unwrap(),
        &cpu_resumed.first_moment_snapshots().unwrap(),
    );
    let (second_moment_lanes, second_moment_max_absolute_error) = assert_live_tensor_maps_close(
        "flushed second moments",
        &resumed.second_moment_snapshots().unwrap(),
        &cpu_resumed.second_moment_snapshots().unwrap(),
    );
    let (accumulator_lanes, accumulator_max_absolute_error) = assert_live_tensor_maps_close(
        "flushed accumulators",
        &resumed.gradient_accumulator_snapshots().unwrap(),
        &cpu_resumed.gradient_accumulator_snapshots().unwrap(),
    );
    assert_eq!(parameter_lanes, 58);
    assert_eq!(first_moment_lanes, 58);
    assert_eq!(second_moment_lanes, 58);
    assert_eq!(accumulator_lanes, 58);
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );

    let checkpoint_after_flush = resumed.checkpoint().unwrap();
    let (_, flush_metadata) = load_safetensors(checkpoint_after_flush.as_bytes()).unwrap();
    assert_eq!(flush_metadata["format"], "rustgrad-compiled-adamw-v5");
    assert_eq!(flush_metadata["replay_step"], "4");
    assert_eq!(flush_metadata["optimizer_step"], "1");
    assert_eq!(flush_metadata["accumulation_index"], "0");
    assert_eq!(flush_metadata["discarded_microbatch_count"], "2");
    assert_eq!(flush_metadata["flushed_window_count"], "1");
    assert_eq!(flush_metadata["flushed_microbatch_count"], "2");
    assert_frozen_embedding_checkpoint_inventory(&checkpoint_after_flush, &initial_parameters, 48);
    let epoch_after_flush = resumed.metal_session().state_epoch();
    let scoreboard_after_flush = resumed.execution_scoreboard_report().unwrap().unwrap();
    let empty_metal_flush = resumed.flush_partial_window(learning_rate()).unwrap();
    let empty_cpu_primary_flush = cpu_primary.flush_partial_window(learning_rate()).unwrap();
    let empty_cpu_flush = cpu_resumed.flush_partial_window(learning_rate()).unwrap();
    assert!(!empty_metal_flush.did_update());
    assert!(empty_metal_flush.report().is_none());
    assert!(!empty_cpu_flush.did_update());
    assert!(!empty_cpu_primary_flush.did_update());
    assert_eq!(resumed.metal_session().state_epoch(), epoch_after_flush);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint_after_flush);
    assert_eq!(
        resumed.execution_scoreboard_report().unwrap().unwrap(),
        scoreboard_after_flush
    );
    for replay in 5..=7 {
        let cpu_primary_result = cpu_primary.step(batch(replay), learning_rate()).unwrap();
        let cpu_resumed_result = cpu_resumed.step(batch(replay), learning_rate()).unwrap();
        assert_eq!(cpu_primary_result.loss(), cpu_resumed_result.loss());
        let metal_result = resumed
            .step_without_host_outputs(batch(replay), learning_rate())
            .unwrap();
        assert_eq!(metal_result.step(), replay);
        assert_eq!(metal_result.capture_identity(), capture_identity);
        totals.record(
            metal_result.report(),
            false,
            state_pair_count,
            logical_state_bytes,
            state_work_items,
            planned_kernel_count,
            command_count_per_invocation,
            transient_h2d_calls_per_invocation,
            transient_h2d_bytes_per_invocation,
        );
        assert_eq!(
            cpu_primary.checkpoint().unwrap(),
            cpu_resumed.checkpoint().unwrap()
        );
        assert_eq!(cpu_resumed.step_count(), resumed.step_count());
        assert_eq!(
            cpu_resumed.optimizer_step().unwrap(),
            resumed.optimizer_step().unwrap()
        );
        assert_eq!(
            cpu_resumed.accumulation_index().unwrap(),
            resumed.accumulation_index().unwrap()
        );
    }
    assert_eq!(resumed.step_count(), 7);
    assert_eq!(resumed.optimizer_step().unwrap(), 2);
    assert_eq!(resumed.accumulation_index().unwrap(), 0);
    assert_eq!(resumed.metal_session().state_epoch(), epoch_before_flush);
    assert_eq!(
        cpu_primary.checkpoint().unwrap(),
        cpu_resumed.checkpoint().unwrap()
    );
    let final_metal_parameters = resumed.parameter_snapshots().unwrap();
    assert!(
        final_metal_parameters
            .iter()
            .any(|(name, value)| value != &initial_parameters[name])
    );
    let (final_parameter_lanes, final_parameter_max_absolute_error) = assert_live_tensor_maps_close(
        "final parameters",
        &final_metal_parameters,
        &cpu_resumed.parameter_snapshots().unwrap(),
    );
    let (final_first_moment_lanes, final_first_moment_max_absolute_error) =
        assert_live_tensor_maps_close(
            "final first moments",
            &resumed.first_moment_snapshots().unwrap(),
            &cpu_resumed.first_moment_snapshots().unwrap(),
        );
    let (final_second_moment_lanes, final_second_moment_max_absolute_error) =
        assert_live_tensor_maps_close(
            "final second moments",
            &resumed.second_moment_snapshots().unwrap(),
            &cpu_resumed.second_moment_snapshots().unwrap(),
        );
    assert_eq!(final_parameter_lanes, 58);
    assert_eq!(final_first_moment_lanes, 58);
    assert_eq!(final_second_moment_lanes, 58);
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );
    assert_eq!(
        cpu_resumed.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );
    assert_eq!(totals.observed_training_invocations, 2);
    assert_eq!(totals.device_only_training_invocations, 5);
    assert_eq!(totals.command_submission_count, 7);
    assert_eq!(totals.command_wait_count, 7);
    assert_eq!(totals.kernel_launch_count, planned_kernel_count * 7);
    assert_eq!(totals.transient_h2d_calls, 21);
    assert_eq!(totals.transient_h2d_bytes, 364);
    assert_eq!(totals.retained_d2h_calls, 2);
    assert_eq!(totals.retained_d2h_bytes, 8);
    let checkpoint_before_evaluation = resumed.checkpoint().unwrap();
    let scoreboard_before_evaluation = resumed.execution_scoreboard_report().unwrap().unwrap();
    let epoch_before_evaluation = resumed.metal_session().state_epoch();
    let dropout_before_evaluation = checkpoint_dropout_block_counter(&checkpoint_before_evaluation);
    let evaluation_identity = resumed.evaluation_capture_identity().unwrap();
    assert_eq!(
        cpu_resumed.evaluation_capture_identity(),
        Some(evaluation_identity)
    );
    let mut compiled_final_mean_sparse_loss = 0.0;
    let mut cpu_final_mean_sparse_loss = 0.0;
    let mut evaluation_lanes = 0;
    let mut evaluation_max_absolute_error = 0.0_f64;
    let mut evaluation_reports = Vec::new();
    for replay in 1..=ACCUMULATION_STEPS {
        let evaluated = resumed.evaluate(batch(replay)).unwrap();
        let cpu_evaluated = cpu_resumed.evaluate(batch(replay)).unwrap();
        assert_eq!(evaluated.capture_identity(), evaluation_identity);
        assert_eq!(cpu_evaluated.capture_identity(), evaluation_identity);
        assert_eq!(
            evaluated.output("logits").unwrap().shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        let actual_loss = evaluated.loss().scalar_at(0).as_f64();
        let expected_loss = cpu_evaluated.loss().scalar_at(0).as_f64();
        let loss_error = assert_live_scalar_close("evaluation loss", actual_loss, expected_loss);
        evaluation_max_absolute_error = evaluation_max_absolute_error.max(loss_error);
        evaluation_lanes += 1;
        for index in 0..TOKEN_COUNT * VOCAB {
            let actual = evaluated
                .output("logits")
                .unwrap()
                .scalar_at(index)
                .as_f64();
            let expected = cpu_evaluated
                .output("logits")
                .unwrap()
                .scalar_at(index)
                .as_f64();
            let error = assert_live_scalar_close("evaluation logit", actual, expected);
            evaluation_max_absolute_error = evaluation_max_absolute_error.max(error);
            evaluation_lanes += 1;
        }
        compiled_final_mean_sparse_loss += evaluated.loss().scalar_at(0).as_f64();
        cpu_final_mean_sparse_loss += cpu_evaluated.loss().scalar_at(0).as_f64();
        evaluation_reports.push(evaluated.report().clone());
    }
    compiled_final_mean_sparse_loss /= ACCUMULATION_STEPS as f64;
    cpu_final_mean_sparse_loss /= ACCUMULATION_STEPS as f64;
    assert_eq!(evaluation_lanes, 3 * (1 + TOKEN_COUNT * VOCAB));
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint_before_evaluation);
    assert_eq!(
        resumed.execution_scoreboard_report().unwrap().unwrap(),
        scoreboard_before_evaluation
    );
    assert_eq!(
        resumed.metal_session().state_epoch(),
        epoch_before_evaluation
    );
    assert_eq!(
        checkpoint_dropout_block_counter(&resumed.checkpoint().unwrap()),
        dropout_before_evaluation
    );
    let evaluation_retained_bytes = DType::F32.itemsize() * (1 + TOKEN_COUNT * VOCAB);
    assert_eq!(evaluation_retained_bytes, 76);
    for (index, report) in evaluation_reports.iter().enumerate() {
        assert_eq!(report.successful_invocation, index as u64 + 1);
        assert_eq!(report.transient_h2d_calls, 2);
        assert_eq!(report.transient_h2d_bytes, 48);
        assert_eq!(report.runtime_control_h2d_calls, 0);
        assert_eq!(report.runtime_control_h2d_bytes, 0);
        assert_eq!(report.retained_d2h_calls, 2);
        assert_eq!(report.retained_d2h_bytes, evaluation_retained_bytes);
        assert_eq!(report.output_count, 2);
        assert_eq!(report.kernel_launch_count, evaluation_kernel_count);
        assert_eq!(report.zero_item_count, evaluation_zero_item_count);
        assert_eq!(report.command_submission_count, 1);
        assert_eq!(report.command_wait_count, 1);
        assert_eq!(report.committed_state_pair_count, 0);
        assert_eq!(report.committed_state_bytes, 0);
        assert_eq!(report.committed_state_work_items, 0);
        assert_eq!(report.committed_state_position, None);
    }
    let (final_state, final_metadata) =
        load_safetensors(resumed.checkpoint().unwrap().as_bytes()).unwrap();
    assert_eq!(final_metadata["format"], "rustgrad-compiled-adamw-v5");
    assert_eq!(final_metadata["replay_step"], "7");
    assert_eq!(final_metadata["optimizer_step"], "2");
    assert_eq!(final_metadata["accumulation_index"], "0");
    assert_eq!(final_metadata["discarded_microbatch_count"], "2");
    assert_eq!(final_metadata["flushed_window_count"], "1");
    assert_eq!(final_metadata["flushed_microbatch_count"], "2");
    assert_frozen_embedding_checkpoint_inventory(
        &resumed.checkpoint().unwrap(),
        &initial_parameters,
        84,
    );
    assert_eq!(
        final_state["dropout_block_counter"].scalar_at(0).as_u64(),
        84
    );
    let published = resumed.parameter_snapshots().unwrap();
    let published_bytes = published
        .values()
        .map(|value| value.shape().numel().unwrap() * value.dtype().itemsize())
        .sum::<usize>();
    assert_eq!(published.len(), 18);
    assert_eq!(published_bytes, 232);
    assert!(!published.contains_key("tokens.weight"));
    let initial_scoreboard = uninterrupted
        .execution_scoreboard_report()
        .unwrap()
        .expect("live compiled training scoreboard must be enabled");
    let resumed_scoreboard = resumed
        .execution_scoreboard_report()
        .unwrap()
        .expect("live resumed training scoreboard must be enabled");
    assert_eq!(initial_scoreboard.successful_run_count, 4);
    assert_eq!(resumed_scoreboard.successful_run_count, 3);
    assert_eq!(initial_scoreboard.fallback_count, 0);
    assert_eq!(resumed_scoreboard.fallback_count, 0);
    assert!(uninterrupted.scoreboard_recording_error().is_none());
    assert!(resumed.scoreboard_recording_error().is_none());
    assert_eq!(initial_scoreboard.retained_host_api_d2h_calls, 2);
    assert_eq!(initial_scoreboard.retained_host_api_d2h_bytes, 8);
    assert_eq!(resumed_scoreboard.retained_host_api_d2h_calls, 0);
    assert_eq!(resumed_scoreboard.retained_host_api_d2h_bytes, 0);
    assert_eq!(
        initial_scoreboard
            .successful_runs
            .iter()
            .filter(|run| run.output_count == 0)
            .count(),
        2
    );
    assert_eq!(
        resumed_scoreboard
            .successful_runs
            .iter()
            .filter(|run| run.output_count == 0)
            .count(),
        3
    );
    assert!(
        initial_scoreboard
            .successful_runs
            .iter()
            .chain(&resumed_scoreboard.successful_runs)
            .all(
                |run| run.command_submission_count == command_count_per_invocation
                    && run.command_wait_count == command_count_per_invocation
                    && run.committed_state_pair_count == state_pair_count
                    && run.committed_state_bytes == logical_state_bytes
                    && run.committed_state_work_items == state_work_items
                    && ((run.output_count == 0
                        && run.retained_host_api_d2h_calls == 0
                        && run.retained_host_api_d2h_bytes == 0)
                        || (run.output_count == 1
                            && run.retained_host_api_d2h_calls == 1
                            && run.retained_host_api_d2h_bytes == 4))
            )
    );
    let loss_scale = uninterrupted.loss_scale();
    let partial_metal_model = uninterrupted.into_module_without_publication();
    assert_eq!(
        partial_metal_model.state_dict().unwrap(),
        initial_module_state
    );
    let cpu_primary_model = cpu_primary
        .finish()
        .expect("the uninterrupted CPU reference must finish atomically");
    let cpu_resumed_model = cpu_resumed
        .finish()
        .expect("the checkpoint-restored CPU reference must finish atomically");
    assert_eq!(
        cpu_primary_model.state_dict().unwrap(),
        cpu_resumed_model.state_dict().unwrap()
    );
    let resumed_model = resumed
        .finish()
        .expect("the resumed owned module must finish atomically");
    let live = resumed_model.state_dict().unwrap();
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert_eq!(resumed_model.tokens.weight.id(), tied.id());
    let tied_after = resumed_model.tokens.weight.snapshot().unwrap();
    assert_eq!(tied_after.data, tied_before.data);
    assert_eq!(tied_after.version, tied_before.version);
    assert_eq!(tied_after.trainable, tied_before.trainable);
    let cpu_tied_after = cpu_resumed_model.tokens.weight.snapshot().unwrap();
    assert_eq!(cpu_tied_after.data, cpu_tied_before.data);
    assert_eq!(cpu_tied_after.version, cpu_tied_before.version);
    assert_eq!(cpu_tied_after.trainable, cpu_tied_before.trainable);
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        if name == "tokens.weight" {
            assert_eq!(parameter.version().unwrap(), before_versions[&name]);
        } else {
            assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
        }
    }
    let frozen_after = resumed_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    let first_eval = evaluate(&resumed_model);
    let second_eval = evaluate(&resumed_model);
    let final_mean_sparse_loss = evaluate_mean_sparse_loss(&resumed_model);
    let cpu_published_mean_sparse_loss = evaluate_mean_sparse_loss(&cpu_resumed_model);
    assert_eq!(first_eval.shape(), &Shape::new([BATCH, TIME, VOCAB]));
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel().unwrap())
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        if name == "tokens.weight" {
            assert_eq!(parameter.version().unwrap(), before_versions[&name]);
        } else {
            assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
        }
    }
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "live compiled causal Transformer eval loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
    assert_live_scalar_close(
        "compiled Metal versus published loss",
        compiled_final_mean_sparse_loss,
        final_mean_sparse_loss,
    );
    assert_live_scalar_close(
        "compiled Metal versus CPU reference loss",
        compiled_final_mean_sparse_loss,
        cpu_final_mean_sparse_loss,
    );
    assert!(
        cpu_published_mean_sparse_loss < initial_mean_sparse_loss,
        "CPU reference causal Transformer eval loss did not decrease: {initial_mean_sparse_loss} -> {cpu_published_mean_sparse_loss}"
    );
    assert!((cpu_final_mean_sparse_loss - cpu_published_mean_sparse_loss).abs() < 1e-5);
    let (final_module_lanes, final_module_max_absolute_error) = assert_live_tensor_maps_close(
        "finished module parameters",
        live.tensors(),
        cpu_resumed_model.state_dict().unwrap().tensors(),
    );
    assert_eq!(final_module_lanes, 65);

    let device_evidence = serde_json::json!({
        "name": device_info.name,
        "registry_id": device_info.registry_id,
        "family": device_info.capabilities.family,
        "unified_memory": device_info.capabilities.unified_memory,
    });
    let identity_evidence = serde_json::json!({
        "capture_identity": capture_identity,
        "loss_scale": loss_scale,
        "initial_deployment_identity": deployment_identity,
        "resumed_deployment_identity": resumed_deployment_identity,
        "fallback_count": 0,
    });
    let state_evidence = serde_json::json!({
        "state_pair_count": state_pair_count,
        "logical_state_bytes": logical_state_bytes,
        "state_work_items": state_work_items,
        "state_bank_count": 2,
        "state_device_bytes": summary.state_device_bytes,
        "planned_kernel_count": planned_kernel_count,
        "indexed_movement_item_count": 0,
        "authenticated_host_indexed_movement_item_count": authenticated_host_indexed_movement_item_count,
        "authenticated_frozen_host_gather_item_count": authenticated_frozen_host_gather_item_count,
        "command_count_per_invocation": command_count_per_invocation,
    });
    let invocation_evidence = serde_json::json!({
        "primary_training_steps": 7,
        "primary_optimizer_steps": 2,
        "final_accumulation_index": 0,
        "resume_replay_steps": 3,
        "nonempty_flush_invocations": 1,
        "empty_flush_invocations": 1,
        "total_device_invocations": 11,
        "observed_training_invocations": totals.observed_training_invocations,
        "device_only_training_invocations": totals.device_only_training_invocations,
    });
    let checkpoint_evidence = serde_json::json!({
        "checkpoint_resume_step": 4,
        "checkpoint_optimizer_step": 0,
        "checkpoint_accumulation_index": 2,
        "discarded_microbatches": 2,
        "zero_grad_nonempty_calls": 1,
        "zero_grad_empty_calls": 1,
        "checkpoint_dropout_block_counter": 48,
        "final_dropout_block_counter": 84,
        "checkpoint_resume_exact": true,
        "cpu_checkpoint_resume_exact": true,
        "flushed_window_count": 1,
        "flushed_microbatch_count": 2,
        "published_parameter_count": published.len(),
        "published_parameter_bytes": published_bytes,
        "publication_native_read_count": serde_json::Value::Null,
    });
    let loss_evidence = serde_json::json!({
        "initial_eval_mean_sparse_loss": initial_mean_sparse_loss,
        "final_eval_mean_sparse_loss": final_mean_sparse_loss,
        "compiled_in_session_eval_mean_sparse_loss": compiled_final_mean_sparse_loss,
        "cpu_in_session_eval_mean_sparse_loss": cpu_final_mean_sparse_loss,
        "cpu_published_eval_mean_sparse_loss": cpu_published_mean_sparse_loss,
        "compiled_in_session_eval_calls": evaluation_reports.len(),
        "fixed_dataset_loss_decreased": true,
    });
    let accounting_evidence = serde_json::json!({
        "kernel_launch_count": totals.kernel_launch_count,
        "command_submission_count": totals.command_submission_count,
        "command_wait_count": totals.command_wait_count,
        "transient_host_api_h2d_calls": totals.transient_h2d_calls,
        "transient_host_api_h2d_bytes": totals.transient_h2d_bytes,
        "retained_host_api_d2h_calls": totals.retained_d2h_calls,
        "retained_host_api_d2h_bytes": totals.retained_d2h_bytes,
        "device_only_retained_host_api_d2h_calls": 0,
        "device_only_retained_host_api_d2h_bytes": 0,
        "observed_retained_host_api_d2h_calls": totals.retained_d2h_calls,
        "observed_retained_host_api_d2h_bytes": totals.retained_d2h_bytes,
    });
    let flush_evidence = serde_json::json!({
        "flush_capture_identity": flush_capture_identity,
        "flush_transient_host_api_h2d_calls": flush_report.transient_h2d_calls,
        "flush_transient_host_api_h2d_bytes": flush_report.transient_h2d_bytes,
        "flush_retained_host_api_d2h_calls": flush_report.retained_d2h_calls,
        "flush_retained_host_api_d2h_bytes": flush_report.retained_d2h_bytes,
        "flush_output_count": flush_report.output_count,
        "flush_kernel_launch_count": flush_report.kernel_launch_count,
        "flush_command_submission_count": flush_report.command_submission_count,
        "flush_command_wait_count": flush_report.command_wait_count,
        "flush_committed_state_pair_count": flush_report.committed_state_pair_count,
        "flush_committed_state_bytes": flush_report.committed_state_bytes,
        "flush_committed_state_work_items": flush_report.committed_state_work_items,
        "flush_epoch_before": epoch_before_flush,
        "flush_epoch_after": epoch_after_flush,
        "empty_flush_submitted_commands": false,
        "empty_flush_changed_epoch": false,
    });
    let evaluation_evidence = serde_json::json!({
        "evaluation_parameter_resident_h2d_calls": 0,
        "evaluation_parameter_resident_h2d_bytes": 0,
        "evaluation_calls": evaluation_reports.len(),
        "evaluation_transient_host_api_h2d_calls": 2 * evaluation_reports.len(),
        "evaluation_transient_host_api_h2d_bytes": 48 * evaluation_reports.len(),
        "evaluation_retained_host_api_d2h_calls": 2 * evaluation_reports.len(),
        "evaluation_retained_host_api_d2h_bytes": evaluation_retained_bytes * evaluation_reports.len(),
        "evaluation_output_count_per_invocation": 2,
        "evaluation_kernel_launch_count_per_invocation": evaluation_kernel_count,
        "evaluation_zero_item_count_per_invocation": evaluation_zero_item_count,
        "evaluation_command_submission_count": evaluation_reports.len(),
        "evaluation_command_wait_count": evaluation_reports.len(),
        "evaluation_committed_state_pair_count": 0,
        "evaluation_committed_state_bytes": 0,
        "evaluation_committed_state_work_items": 0,
    });
    let agreement_evidence = serde_json::json!({
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 1e-4,
        "partial_parameter_lanes": partial_parameter_lanes,
        "partial_parameter_max_absolute_error": partial_parameter_error,
        "partial_accumulator_lanes": partial_accumulator_lanes,
        "partial_accumulator_max_absolute_error": partial_accumulator_error,
        "post_flush_parameter_lanes": parameter_lanes,
        "post_flush_parameter_max_absolute_error": parameter_max_absolute_error,
        "post_flush_first_moment_lanes": first_moment_lanes,
        "post_flush_first_moment_max_absolute_error": first_moment_max_absolute_error,
        "post_flush_second_moment_lanes": second_moment_lanes,
        "post_flush_second_moment_max_absolute_error": second_moment_max_absolute_error,
        "post_flush_accumulator_lanes": accumulator_lanes,
        "post_flush_accumulator_max_absolute_error": accumulator_max_absolute_error,
        "final_parameter_lanes": final_parameter_lanes,
        "final_parameter_max_absolute_error": final_parameter_max_absolute_error,
        "final_first_moment_lanes": final_first_moment_lanes,
        "final_first_moment_max_absolute_error": final_first_moment_max_absolute_error,
        "final_second_moment_lanes": final_second_moment_lanes,
        "final_second_moment_max_absolute_error": final_second_moment_max_absolute_error,
        "evaluation_lanes": evaluation_lanes,
        "evaluation_max_absolute_error": evaluation_max_absolute_error,
        "finished_module_max_absolute_error": final_module_max_absolute_error,
    });
    let frozen_evidence = serde_json::json!({
        "policy_frozen_parameters": ["tokens.weight"],
        "tied_alias": "lm_head.weight",
        "effective_trainable_parameter_count": published.len(),
        "effective_trainable_parameter_lanes": final_parameter_lanes,
        "frozen_tied_parameter_lanes": tied_after.data.len(),
        "total_trainable_parameter_lanes": final_parameter_lanes + tied_after.data.len(),
        "frozen_recurrent_state_absent": true,
        "frozen_checkpoint_state_absent": true,
        "frozen_module_bytes_unchanged": true,
        "frozen_module_version_unchanged": true,
        "frozen_module_trainable_flag_unchanged": true,
        "original_module_state_unchanged": true,
        "unfrozen_parameter_changed": true,
        "finished_module_lanes": final_module_lanes,
    });
    let scoreboard_evidence = serde_json::json!({
        "initial_scoreboard": initial_scoreboard,
        "resumed_scoreboard": resumed_scoreboard,
    });
    let mut evidence = serde_json::json!({
        "format_version": 7,
        "workload": "tiny-causal-transformer-compiled-adamw",
        "implementation_revision": expected_sha,
        "device": device_evidence,
    });
    let evidence_object = evidence
        .as_object_mut()
        .expect("live evidence root must remain an object");
    for fragment in [
        identity_evidence,
        state_evidence,
        invocation_evidence,
        checkpoint_evidence,
        loss_evidence,
        accounting_evidence,
        flush_evidence,
        evaluation_evidence,
        agreement_evidence,
        frozen_evidence,
        scoreboard_evidence,
    ] {
        let serde_json::Value::Object(fragment) = fragment else {
            unreachable!("live evidence fragment is statically an object")
        };
        evidence_object.extend(fragment);
    }
    evidence_object.insert("weight_decay".into(), policy.weight_decay().into());
    evidence_object.insert(
        "gradient_accumulation_steps".into(),
        ACCUMULATION_STEPS.into(),
    );
    evidence_object.insert("max_gradient_norm".into(), MAX_GRADIENT_NORM.into());
    evidence_object.insert(
        "weight_decay_exclusions".into(),
        serde_json::json!(WEIGHT_DECAY_EXCLUSIONS),
    );
    let encoded = serde_json::to_vec(&evidence).expect("live evidence JSON must serialize");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&evidence_path)
        .expect("live evidence must use a create-new path");
    file.write_all(&encoded)
        .expect("live evidence bytes must be written completely");
    file.sync_all()
        .expect("live evidence must be durable before the test succeeds");
}
