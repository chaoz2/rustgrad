#[cfg(target_os = "macos")]
use rustgrad::MetalSessionTarget;
use rustgrad::nn::{Embedding, LayerNorm, Mode, ModeModuleForward, StateDict, StateKind};
use rustgrad::runtime::metal::{MetalCapabilities, MetalRenderer};
#[cfg(target_os = "macos")]
use rustgrad::runtime::metal::{
    MetalDeviceRunReport, MetalDiscovery, MetalRuntime, MetalScoreboardContext,
};
use rustgrad::{
    Backend, CompareOp, CompiledAdamWCheckpoint, CompiledAdamWConfig, CompiledAdamWPlan,
    CompiledAdamWRuntime, CompiledAdamWStep, CompiledCheckpointRuntime, CompiledDropoutConfig,
    CompiledDropoutKey, CompiledEvaluation, CompiledEvaluationRuntime, CompiledModuleAdamWPlan,
    CompiledTrainingRuntime, CompiledTrainingStep, CpuBackend, CpuSessionTarget, DType, Graph,
    LossOptions, MetalCompiledAdamWPlan, Module, NodeId, Op, Parameter, Reduction, Result, Scalar,
    Shape, TensorData, TrainingDropoutProvider, TransformerBlock, cross_entropy, load_safetensors,
};
use std::collections::{BTreeMap, HashMap};
#[cfg(target_os = "macos")]
use std::{env, fs::OpenOptions, io::Write, path::PathBuf};

const VOCAB: usize = 3;
const EMBEDDING: usize = 2;
const BATCH: usize = 2;
const TIME: usize = 3;
const TOKEN_COUNT: usize = BATCH * TIME;
const ACCUMULATION_STEPS: u64 = 3;
const MAX_GRADIENT_NORM: f32 = 0.25;
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

fn config() -> CompiledAdamWConfig {
    config_with_max_gradient_norm(Some(MAX_GRADIENT_NORM))
}

fn config_with_max_gradient_norm(max_gradient_norm: Option<f32>) -> CompiledAdamWConfig {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)
        .unwrap()
        .with_weight_decay_exclusions(WEIGHT_DECAY_EXCLUSIONS)
        .unwrap()
        .with_loss_scale(128.0)
        .unwrap()
        .with_gradient_accumulation(ACCUMULATION_STEPS)
        .unwrap();
    let config = match max_gradient_norm {
        Some(max_gradient_norm) => config.with_max_gradient_norm(max_gradient_norm).unwrap(),
        None => config,
    };
    config
        .with_host_token_input("tokens", [BATCH, TIME])
        .unwrap()
        .with_host_token_input("targets", [BATCH, TIME])
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

fn batch(replay: u64) -> BTreeMap<String, TensorData> {
    let tensor = |values: [i32; TOKEN_COUNT]| {
        TensorData::from_scalars(
            Shape::new([BATCH, TIME]),
            DType::I32,
            values.into_iter().map(|value| Scalar::I(i64::from(value))),
        )
        .unwrap()
    };
    let (tokens, targets) = match (replay - 1) % ACCUMULATION_STEPS {
        0 => ([0, 1, 2, 2, 0, 1], [1, 2, 0, 0, 1, 2]),
        1 => ([1, 2, 0, 0, 1, 2], [2, 0, 1, 1, 2, 0]),
        2 => ([2, 0, 1, 1, 2, 0], [0, 1, 2, 2, 0, 1]),
        _ => unreachable!(),
    };
    BTreeMap::from([
        ("tokens".into(), tensor(tokens)),
        ("targets".into(), tensor(targets)),
    ])
}

fn learning_rate() -> TensorData {
    TensorData::scalar(0.05)
}

#[cfg(target_os = "macos")]
fn checkpoint_dropout_block_counter(checkpoint: &CompiledAdamWCheckpoint) -> u64 {
    let (state, _) = load_safetensors(checkpoint.as_bytes()).unwrap();
    state["dropout_block_counter"].scalar_at(0).as_u64()
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
        Self {
            masks: [
                mask([
                    true, false, true, true, false, true, false, true, true, false, true, false,
                ]),
                mask([
                    false, true, true, false, true, true, true, true, false, true, false, true,
                ]),
            ],
            next: 0,
        }
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
        relu_input: relu_inputs[0],
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

fn run_exact_resume<R, P>(mut prepare: P) -> ExactResumeEvaluation
where
    R: CompiledAdamWRuntime + CompiledEvaluationRuntime,
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
    let (checkpoint_state, checkpoint_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(checkpoint_metadata["format"], "rustgrad-compiled-adamw-v4");
    assert_eq!(checkpoint_metadata["replay_step"], "4");
    assert_eq!(checkpoint_metadata["optimizer_step"], "0");
    assert_eq!(checkpoint_metadata["gradient_accumulation_steps"], "3");
    assert_eq!(checkpoint_metadata["accumulation_index"], "2");
    assert_eq!(checkpoint_metadata["discarded_microbatch_count"], "2");
    assert_eq!(
        checkpoint_state["dropout_block_counter"]
            .scalar_at(0)
            .as_u64(),
        48
    );
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let tied = resumed_model.tokens.weight.clone();
    let frozen = resumed_model.frozen_scale.clone();
    let resumed_plan = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        resumed_model,
        &checkpoint,
        build,
    )
    .unwrap()
    .with_evaluation(build_evaluation)
    .unwrap();
    assert_eq!(resumed_plan.capture_identity(), capture_identity);
    assert_eq!(resumed_plan.step_count(), 4);
    let mut resumed = prepare(resumed_plan).unwrap();
    assert_eq!(resumed.optimizer_step().unwrap(), 0);
    assert_eq!(resumed.accumulation_index().unwrap(), 2);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);

    for replay in 5..=8 {
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
            8 => (2, 0, true),
            _ => unreachable!(),
        };
        assert_eq!(actual.optimizer_step(), optimizer_step);
        assert_eq!(actual.accumulation_index(), accumulation_index);
        assert_eq!(actual.did_update(), did_update);
    }
    assert_eq!(resumed.step_count(), 8);
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
    let (final_state, final_metadata) =
        load_safetensors(resumed.checkpoint().unwrap().as_bytes()).unwrap();
    assert_eq!(final_metadata["replay_step"], "8");
    assert_eq!(final_metadata["optimizer_step"], "2");
    assert_eq!(final_metadata["accumulation_index"], "0");
    assert_eq!(final_metadata["discarded_microbatch_count"], "2");
    assert_eq!(
        final_state["dropout_block_counter"].scalar_at(0).as_u64(),
        96
    );
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
    let _uninterrupted_model = uninterrupted.finish().unwrap();
    let resumed_model = resumed.finish().unwrap();
    let live = resumed_model.state_dict().unwrap();
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
    assert_eq!(final_mean_sparse_loss, published_mean_sparse_loss);
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
    let evaluation =
        run_exact_resume(|plan| plan.prepare(&target).map_err(|error| error.into_parts().1));

    assert!(
        evaluation.final_mean_sparse_loss < evaluation.initial_mean_sparse_loss,
        "compiled causal Transformer eval loss did not decrease: {evaluation:?}"
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
    let checkpoint = session.checkpoint().unwrap();
    let midpoint = session.parameter_snapshots().unwrap();
    let model = session.finish().unwrap();
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
    let model = session.finish().unwrap();
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
    let final_mean_sparse_loss = evaluate_mean_sparse_loss(&model);
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "owned compiled causal Transformer eval loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
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
fn protected_live_metal_workflow_runs_the_exact_compiled_training_acceptance() {
    let workflow = include_str!("../.github/workflows/metal-live.yml");
    for required in [
        "RUSTGRAD_METAL_TRAINING_EVIDENCE_PATH:",
        "metal-live-compiled-training-v6.json",
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

    let model = TinyCausalTransformer::new(7).unwrap();
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&model);
    let seed = owned_compiled_transformer(model);
    let capture_identity = seed.capture_identity();
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
    assert_eq!(state_pair_count, 79);
    assert_eq!(logical_state_bytes, 1_048);
    assert_eq!(summary.state_bank_count, 2);
    assert_eq!(summary.state_device_bytes, 2_096);
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
    assert_eq!(state_work_items, 259);
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
    assert_eq!(authenticated_host_indexed_movement_item_count, 4);
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
    let empty_accumulators = uninterrupted.gradient_accumulator_snapshots().unwrap();

    let mut totals = LiveTrainingTotals::default();
    for index in 0..4u64 {
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
            assert_eq!(reset.discarded_microbatches(), 2);
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
    let (checkpoint_state, checkpoint_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(checkpoint_metadata["format"], "rustgrad-compiled-adamw-v4");
    assert_eq!(checkpoint_metadata["replay_step"], "4");
    assert_eq!(checkpoint_metadata["optimizer_step"], "0");
    assert_eq!(checkpoint_metadata["gradient_accumulation_steps"], "3");
    assert_eq!(checkpoint_metadata["accumulation_index"], "2");
    assert_eq!(checkpoint_metadata["discarded_microbatch_count"], "2");
    assert_eq!(
        checkpoint_state["dropout_block_counter"]
            .scalar_at(0)
            .as_u64(),
        48
    );
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let tied = resumed_model.tokens.weight.clone();
    let frozen = resumed_model.frozen_scale.clone();
    let tied_version = tied.version().unwrap();
    let frozen_before = frozen.snapshot().unwrap();
    let before_versions = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let resumed_seed = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        resumed_model,
        &checkpoint,
        build,
    )
    .unwrap()
    .with_evaluation(build_evaluation)
    .unwrap();
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

    for resumed_index in 0..4u64 {
        let replay = resumed_index + 5;
        if resumed_index < 3 {
            let expected = uninterrupted
                .step_without_host_outputs(batch(replay), learning_rate())
                .unwrap();
            let actual = resumed
                .step_without_host_outputs(batch(replay), learning_rate())
                .unwrap();
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
            assert_eq!(actual.report().successful_invocation, resumed_index + 1);
            for report in [expected.report(), actual.report()] {
                totals.record(
                    report,
                    false,
                    state_pair_count,
                    logical_state_bytes,
                    state_work_items,
                    planned_kernel_count,
                    command_count_per_invocation,
                    transient_h2d_calls_per_invocation,
                    transient_h2d_bytes_per_invocation,
                );
            }
        } else {
            let expected = uninterrupted.step(batch(replay), learning_rate()).unwrap();
            let actual = resumed.step(batch(replay), learning_rate()).unwrap();
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.step(), expected.step());
            assert_eq!(actual.optimizer_step(), expected.optimizer_step());
            assert_eq!(actual.accumulation_index(), expected.accumulation_index());
            assert_eq!(actual.did_update(), expected.did_update());
            assert_eq!(actual.optimizer_step(), 2);
            assert_eq!(actual.accumulation_index(), 0);
            assert!(actual.did_update());
            assert_eq!(actual.report().successful_invocation, resumed_index + 1);
            for report in [expected.report(), actual.report()] {
                totals.record(
                    report,
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
        }
    }

    assert_eq!(resumed.step_count(), 8);
    assert_eq!(uninterrupted.step_count(), 8);
    assert_eq!(resumed.optimizer_step().unwrap(), 2);
    assert_eq!(uninterrupted.optimizer_step().unwrap(), 2);
    assert_eq!(resumed.accumulation_index().unwrap(), 0);
    assert_eq!(uninterrupted.accumulation_index().unwrap(), 0);
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        uninterrupted.gradient_accumulator_snapshots().unwrap()
    );
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        empty_accumulators
    );
    assert_eq!(totals.observed_training_invocations, 4);
    assert_eq!(totals.device_only_training_invocations, 8);
    assert_eq!(totals.command_submission_count, 12);
    assert_eq!(totals.command_wait_count, 12);
    assert_eq!(totals.kernel_launch_count, planned_kernel_count * 12);
    assert_eq!(
        totals.transient_h2d_calls,
        transient_h2d_calls_per_invocation * 12
    );
    assert_eq!(totals.transient_h2d_calls, 36);
    assert_eq!(
        totals.transient_h2d_bytes,
        transient_h2d_bytes_per_invocation * 12
    );
    assert_eq!(totals.transient_h2d_bytes, 624);
    assert_eq!(totals.retained_d2h_calls, 4);
    assert_eq!(totals.retained_d2h_bytes, 16);
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );
    let checkpoint_before_evaluation = resumed.checkpoint().unwrap();
    let scoreboard_before_evaluation = resumed.execution_scoreboard_report().unwrap().unwrap();
    let epoch_before_evaluation = resumed.metal_session().state_epoch();
    let dropout_before_evaluation = checkpoint_dropout_block_counter(&checkpoint_before_evaluation);
    let evaluation_identity = resumed.evaluation_capture_identity().unwrap();
    let mut compiled_final_mean_sparse_loss = 0.0;
    let mut evaluation_reports = Vec::new();
    for replay in 1..=ACCUMULATION_STEPS {
        let evaluated = resumed.evaluate(batch(replay)).unwrap();
        assert_eq!(evaluated.capture_identity(), evaluation_identity);
        assert_eq!(
            evaluated.output("logits").unwrap().shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        compiled_final_mean_sparse_loss += evaluated.loss().scalar_at(0).as_f64();
        evaluation_reports.push(evaluated.report().clone());
    }
    compiled_final_mean_sparse_loss /= ACCUMULATION_STEPS as f64;
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
    assert_eq!(final_metadata["replay_step"], "8");
    assert_eq!(final_metadata["optimizer_step"], "2");
    assert_eq!(final_metadata["accumulation_index"], "0");
    assert_eq!(final_metadata["discarded_microbatch_count"], "2");
    assert_eq!(
        final_state["dropout_block_counter"].scalar_at(0).as_u64(),
        96
    );
    let published = resumed.parameter_snapshots().unwrap();
    let published_bytes = published
        .values()
        .map(|value| value.shape().numel().unwrap() * value.dtype().itemsize())
        .sum::<usize>();
    assert_eq!(published.len(), 19);
    assert_eq!(published_bytes, 256);
    let initial_scoreboard = uninterrupted
        .execution_scoreboard_report()
        .unwrap()
        .expect("live compiled training scoreboard must be enabled");
    let resumed_scoreboard = resumed
        .execution_scoreboard_report()
        .unwrap()
        .expect("live resumed training scoreboard must be enabled");
    assert_eq!(initial_scoreboard.successful_run_count, 8);
    assert_eq!(resumed_scoreboard.successful_run_count, 4);
    assert_eq!(initial_scoreboard.fallback_count, 0);
    assert_eq!(resumed_scoreboard.fallback_count, 0);
    assert!(uninterrupted.scoreboard_recording_error().is_none());
    assert!(resumed.scoreboard_recording_error().is_none());
    assert_eq!(initial_scoreboard.retained_host_api_d2h_calls, 3);
    assert_eq!(initial_scoreboard.retained_host_api_d2h_bytes, 12);
    assert_eq!(resumed_scoreboard.retained_host_api_d2h_calls, 1);
    assert_eq!(resumed_scoreboard.retained_host_api_d2h_bytes, 4);
    assert_eq!(
        initial_scoreboard
            .successful_runs
            .iter()
            .filter(|run| run.output_count == 0)
            .count(),
        5
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
    let _uninterrupted_model = uninterrupted
        .finish()
        .expect("the uninterrupted owned module must finish atomically");
    let resumed_model = resumed
        .finish()
        .expect("the resumed owned module must finish atomically");
    let live = resumed_model.state_dict().unwrap();
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert_eq!(resumed_model.tokens.weight.id(), tied.id());
    assert_eq!(
        resumed_model.tokens.weight.version().unwrap(),
        tied_version + 1
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
    }
    let frozen_after = resumed_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    let first_eval = evaluate(&resumed_model);
    let second_eval = evaluate(&resumed_model);
    let final_mean_sparse_loss = evaluate_mean_sparse_loss(&resumed_model);
    assert_eq!(first_eval.shape(), &Shape::new([BATCH, TIME, VOCAB]));
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel().unwrap())
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
    }
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "live compiled causal Transformer eval loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
    assert!((compiled_final_mean_sparse_loss - final_mean_sparse_loss).abs() < 1e-5);

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
        "command_count_per_invocation": command_count_per_invocation,
    });
    let invocation_evidence = serde_json::json!({
        "primary_training_steps": 8,
        "primary_optimizer_steps": 2,
        "final_accumulation_index": 0,
        "resume_replay_steps": 4,
        "total_device_invocations": 12,
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
        "final_dropout_block_counter": 96,
        "checkpoint_resume_exact": true,
        "published_parameter_count": published.len(),
        "published_parameter_bytes": published_bytes,
        "publication_native_read_count": serde_json::Value::Null,
    });
    let loss_evidence = serde_json::json!({
        "initial_eval_mean_sparse_loss": initial_mean_sparse_loss,
        "final_eval_mean_sparse_loss": final_mean_sparse_loss,
        "compiled_in_session_eval_mean_sparse_loss": compiled_final_mean_sparse_loss,
        "compiled_in_session_eval_calls": evaluation_reports.len(),
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
    let scoreboard_evidence = serde_json::json!({
        "initial_scoreboard": initial_scoreboard,
        "resumed_scoreboard": resumed_scoreboard,
    });
    let mut evidence = serde_json::json!({
        "format_version": 6,
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
        evaluation_evidence,
        scoreboard_evidence,
    ] {
        let serde_json::Value::Object(fragment) = fragment else {
            unreachable!("live evidence fragment is statically an object")
        };
        evidence_object.extend(fragment);
    }
    evidence_object.insert("weight_decay".into(), config().weight_decay().into());
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
