//! Manual, resource-bounded evidence for a larger fixed-shape compiled
//! Transformer training run on strict-native CPU.
//!
//! This example is intentionally excluded from ordinary CI execution. The
//! `native-transformer-scale-evidence` workflow runs it at an exact reviewed
//! commit and uploads its authenticated scoreboard, objective facts, and host
//! provenance. A separate no-dropout replay checks two dense gradient
//! projections, while a fresh executor restores the replay-three portable
//! bundle through the same isolated durable cache and partitions its warm
//! preparation by program role. Neither affects the cold six-replay timing
//! sample. Timings are observations, never pass/fail thresholds.

use rustgrad::nn::{Embedding, LayerNorm, Mode, StateKind};
use rustgrad::{
    Backend, CapturedReplayExecutor, CompiledAdamWCheckpoint, CompiledAdamWConfig,
    CompiledAdamWGraph, CompiledAdamWIgnoreIndexContext, CompiledAdamWInspection,
    CompiledAdamWResumeBundle, CompiledAdamWRuntime, CompiledCheckpointRestoreRuntime,
    CompiledCheckpointRuntime, CompiledDropoutConfig, CompiledDropoutKey,
    CompiledEvaluationRuntime, CompiledInputBatch, CompiledInputSpec,
    CompiledModuleAdamWCheckpoint, CompiledModuleAdamWPlan, CompiledMultiStepLr,
    CompiledTrainingRuntime, CpuBackend, DType, Error as RustGradError, Graph, LossOptions, Module,
    NativeCpuCompiledAdamW, NativeCpuCompiledAdamWStepResult, NativeCpuProgramPreparationReport,
    NativeCpuSessionTarget, NativeTrainingScoreboard, NodeId, Op, Parameter, ParameterSnapshot,
    Reduction, Result, Scalar, TensorData, TrainingDropoutProvider, TransformerBlock,
    sparse_categorical_cross_entropy,
};
use serde::Serialize;
use serde_json::json;
use std::{
    collections::{BTreeMap, HashMap},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const BATCH: usize = 4;
const TIME: usize = 8;
const VOCAB: usize = 16;
const EMBEDDING: usize = 8;
const HEADS: usize = 2;
const FEED_FORWARD: usize = 32;
const ACCUMULATION_STEPS: u64 = 2;
const REPLAYS: u64 = 6;
const CHECKPOINT_REPLAY: u64 = 3;
const IGNORE_INDEX: i32 = -100;
const TARGETS: &str = "targets";
const TOKENS: &str = "tokens";
const FROZEN_POSITION_WEIGHT: &str = "positions.weight";
const ATTENTION_KEEP_MASK_SHAPE: [usize; 4] = [BATCH, 1, 1, TIME];
const GRADIENT_PROBE_PARAMETER_COUNT: usize = 35;
const GRADIENT_PROBE_COORDINATE_COUNT: usize = 1_888;
const GRADIENT_PROBE_VALID_TOKEN_COUNT: u64 = 22;
const GRADIENT_PROBE_EPSILON: f64 = 4e-2;
const GRADIENT_PROBE_TOLERANCE: f64 = 3e-2;
const GRADIENT_PROBE_DIRECTION_GAIN: f64 = 8.0;
const SCALE_SEED: u64 = 0x5ca1_e000;
const WARM_RESUME_SEED: u64 = 0x5ca1_f000;

struct ScaleTransformer {
    tokens: Embedding,
    positions: Embedding,
    first: TransformerBlock,
    second: TransformerBlock,
    norm: LayerNorm,
}

impl ScaleTransformer {
    fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            tokens: Embedding::new_static(VOCAB, EMBEDDING, None, seed)?,
            positions: Embedding::new_static(TIME, EMBEDDING, None, seed.wrapping_add(1))?,
            first: TransformerBlock::new_static(
                EMBEDDING,
                HEADS,
                FEED_FORWARD,
                true,
                0.1,
                seed.wrapping_add(2),
            )?
            .with_causal_attention(true)
            .with_attention_dropout(0.1)?,
            second: TransformerBlock::new_static(
                EMBEDDING,
                HEADS,
                FEED_FORWARD,
                true,
                0.1,
                seed.wrapping_add(3),
            )?
            .with_causal_attention(true)
            .with_attention_dropout(0.1)?,
            norm: LayerNorm::new_static([EMBEDDING], 1e-5, true)?,
        })
    }

    fn positions(graph: &mut Graph) -> Result<NodeId> {
        let values = (0..BATCH).flat_map(|_| {
            (0..TIME).map(|position| {
                Scalar::I(i64::try_from(position).expect("fixed position fits in i64"))
            })
        });
        Ok(graph.constant(TensorData::from_scalars([BATCH, TIME], DType::I32, values)?))
    }

    fn forward_training(
        &self,
        graph: &mut Graph,
        tokens: NodeId,
        attention_keep_mask: NodeId,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<NodeId> {
        let token_hidden = self.tokens.forward(graph, tokens)?;
        let positions = Self::positions(graph)?;
        let position_hidden = self.positions.forward(graph, positions)?;
        let hidden = graph.add(token_hidden, position_hidden)?;
        let hidden = self
            .first
            .forward_training_with_dropout_and_attention_mask(
                graph,
                hidden,
                attention_keep_mask,
                dropout,
            )?;
        let hidden = self
            .second
            .forward_training_with_dropout_and_attention_mask(
                graph,
                hidden,
                attention_keep_mask,
                dropout,
            )?;
        self.project(graph, hidden)
    }

    fn forward_evaluation(
        &self,
        graph: &mut Graph,
        tokens: NodeId,
        attention_keep_mask: NodeId,
    ) -> Result<NodeId> {
        let token_hidden = self.tokens.forward(graph, tokens)?;
        let positions = Self::positions(graph)?;
        let position_hidden = self.positions.forward(graph, positions)?;
        let hidden = graph.add(token_hidden, position_hidden)?;
        let hidden = self
            .first
            .forward_mode_with_attention_mask(graph, hidden, attention_keep_mask, Mode::Eval)?
            .output;
        let hidden = self
            .second
            .forward_mode_with_attention_mask(graph, hidden, attention_keep_mask, Mode::Eval)?
            .output;
        self.project(graph, hidden)
    }

    fn project(&self, graph: &mut Graph, hidden: NodeId) -> Result<NodeId> {
        let hidden = self.norm.forward(graph, hidden)?;
        let tied_weight = self.tokens.weight.bind(graph)?;
        let tied_weight = graph.permute(tied_weight, [1, 0])?;
        graph.matmul(hidden, tied_weight)
    }
}

impl Module for ScaleTransformer {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        let child = |name: &str| {
            if prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{prefix}.{name}")
            }
        };
        self.tokens.visit(&child("tokens"), visitor);
        self.positions.visit(&child("positions"), visitor);
        self.first.visit(&child("first"), visitor);
        self.second.visit(&child("second"), visitor);
        self.norm.visit(&child("norm"), visitor);
        visitor(
            child("lm_head.weight"),
            &self.tokens.weight,
            StateKind::Parameter,
        );
    }
}

struct ScaleBatch {
    tokens: TensorData,
    targets: TensorData,
}

impl ScaleBatch {
    const SCHEMA: [CompiledInputSpec; 2] = [
        CompiledInputSpec::new(TARGETS, &[BATCH, TIME], DType::I32),
        CompiledInputSpec::host_token(TOKENS, &[BATCH, TIME]),
    ];

    fn valid_lengths(replay: u64) -> [usize; BATCH] {
        let phase = (replay - 1) % ACCUMULATION_STEPS + 1;
        match phase {
            1 => [8, 7, 5, 2],
            2 => [8, 5, 3, 0],
            _ => unreachable!(),
        }
    }

    fn attention_keep_mask(replay: u64) -> Result<TensorData> {
        let valid_lengths = Self::valid_lengths(replay);
        TensorData::from_scalars(
            ATTENTION_KEEP_MASK_SHAPE,
            DType::Bool,
            valid_lengths
                .into_iter()
                .flat_map(|valid| (0..TIME).map(move |column| Scalar::Bool(column < valid))),
        )
    }

    fn new(replay: u64) -> Result<Self> {
        let phase = (replay - 1) % ACCUMULATION_STEPS + 1;
        let valid_lengths = Self::valid_lengths(replay);
        let mut tokens = Vec::with_capacity(BATCH * TIME);
        let mut targets = Vec::with_capacity(BATCH * TIME);
        for (row, valid) in valid_lengths.into_iter().enumerate() {
            for column in 0..TIME {
                if column < valid {
                    let token = i64::try_from(
                        (usize::try_from(phase).expect("fixed replay phase fits in usize")
                            + row * 3
                            + column)
                            % VOCAB,
                    )
                    .expect("fixed token fits in i64");
                    tokens.push(Scalar::I(token));
                    targets.push(Scalar::I(
                        (token + 1) % i64::try_from(VOCAB).expect("fixed vocabulary fits in i64"),
                    ));
                } else {
                    tokens.push(Scalar::I(0));
                    targets.push(Scalar::I(i64::from(IGNORE_INDEX)));
                }
            }
        }
        Ok(Self {
            tokens: TensorData::from_scalars([BATCH, TIME], DType::I32, tokens)?,
            targets: TensorData::from_scalars([BATCH, TIME], DType::I32, targets)?,
        })
    }
}

impl CompiledInputBatch for ScaleBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Ok(BTreeMap::from([
            (TARGETS.into(), self.targets),
            (TOKENS.into(), self.tokens),
        ]))
    }
}

fn losses(graph: &mut Graph, logits: NodeId, targets: NodeId) -> Result<NodeId> {
    sparse_categorical_cross_entropy(
        graph,
        logits,
        targets,
        LossOptions {
            reduction: Reduction::None,
            class_axis: 2,
            ignore_index: Some(i64::from(IGNORE_INDEX)),
            label_smoothing: 0.0,
        },
    )
}

fn build_training(
    model: &ScaleTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model.forward_training(graph, inputs[TOKENS], attention_keep_mask, dropout)?;
    Ok(CompiledAdamWGraph::token_mean(
        losses(graph, logits, inputs[TARGETS])?,
        BTreeMap::new(),
    ))
}

fn build_evaluation(
    model: &ScaleTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
) -> Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model.forward_evaluation(graph, inputs[TOKENS], attention_keep_mask)?;
    Ok(CompiledAdamWGraph::token_mean(
        losses(graph, logits, inputs[TARGETS])?,
        BTreeMap::new(),
    ))
}

fn scale_config() -> Result<CompiledAdamWConfig> {
    Ok(CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(0.25)?
        .with_input_batch::<ScaleBatch>()?
        .with_token_weighted_ignore_index(TARGETS, IGNORE_INDEX)?
        .with_frozen_parameters([FROZEN_POSITION_WEIGHT])?
        .with_captured_multi_step_lr(CompiledMultiStepLr::new(2e-3, 0.5, [1])?)
        .with_clip_report()
        .with_window_loss_report())
}

fn build_gradient_probe(
    model: &ScaleTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
) -> Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model.forward_evaluation(graph, inputs[TOKENS], attention_keep_mask)?;
    Ok(CompiledAdamWGraph::token_mean(
        losses(graph, logits, inputs[TARGETS])?,
        BTreeMap::new(),
    ))
}

struct ScaleGradientParameter {
    name: String,
    input_name: String,
    value: TensorData,
}

struct ScaleGradientOracle {
    graph: Graph,
    numerator: NodeId,
    bindings: HashMap<String, TensorData>,
    parameters: Vec<ScaleGradientParameter>,
}

#[derive(Serialize)]
struct ScaleGradientProjectionEvidence {
    direction_id: &'static str,
    epsilon: f64,
    expected_projection: f64,
    actual_projection: f64,
    absolute_error: f64,
    tolerance: f64,
    mutation_sensitive: bool,
}

#[derive(Serialize)]
struct ScaleGradientEvidence {
    replay: u64,
    accumulation_index: u64,
    did_update: bool,
    dropout_probability: f64,
    valid_token_count: u64,
    active_parameter_count: usize,
    active_coordinate_count: usize,
    tied_output_head_is_canonical_alias: bool,
    frozen_position_is_absent: bool,
    native_preparation_fallback_count: usize,
    native_replay_fallback_count: usize,
    projections: Vec<ScaleGradientProjectionEvidence>,
}

#[derive(Serialize)]
struct ScaleReplayProgressEvidence {
    replay: u64,
    optimizer_step: u64,
    accumulation_index: u64,
    did_update: bool,
}

#[derive(Serialize)]
struct ScaleEvaluationSampleEvidence {
    batch_replay: u64,
    capture_identity: u64,
    token_mean_loss: f64,
    valid_token_count: u64,
    native_fallback_count: usize,
    executed_native_item_count: usize,
    module_dispatch_count: usize,
}

#[derive(Serialize)]
struct ScaleEvaluationFrontierEvidence {
    replay: u64,
    optimizer_step: u64,
    accumulation_index: u64,
    token_mean_loss: f64,
    checkpoint_state_neutral: bool,
    samples: Vec<ScaleEvaluationSampleEvidence>,
}

#[derive(Serialize)]
struct ScaleWarmProgramPreparationEvidence {
    total_wall_time_ns: u64,
    layout_wall_time_ns: u64,
    render_wall_time_ns: u64,
    compiler_wall_time_ns: u64,
    load_wall_time_ns: u64,
    residual_wall_time_ns: u64,
}

#[derive(Serialize)]
struct ScaleWarmPreparationEvidence {
    main: ScaleWarmProgramPreparationEvidence,
    accumulation: ScaleWarmProgramPreparationEvidence,
    partial_flush: ScaleWarmProgramPreparationEvidence,
    zero_grad: ScaleWarmProgramPreparationEvidence,
    evaluation: ScaleWarmProgramPreparationEvidence,
    summed_program_wall_time_ns: u64,
    runtime_overhead_wall_time_ns: u64,
    module_overlap_wall_time_ns: u64,
    render_overlap_wall_time_ns: u64,
    effective_render_wall_time_ns: u64,
    effective_render_fraction: f64,
}

#[derive(Serialize)]
struct ScaleWarmResumeEvidence {
    replay_from: u64,
    replay_to: u64,
    resume_bundle_bytes: usize,
    module_checkpoint_bytes: usize,
    artifact_decode_wall_time_ns: u64,
    owner_restore_wall_time_ns: u64,
    preparation_wall_time_ns: u64,
    preparation: ScaleWarmPreparationEvidence,
    capture_identity: u64,
    evaluation_capture_identity: u64,
    program_count: usize,
    loaded_module_count: usize,
    durable_artifact_cache_hit_count: usize,
    durable_artifact_cache_miss_count: usize,
    render_capsule_hit_count: usize,
    render_capsule_miss_count: usize,
    local_render_job_count: usize,
    compiler_invocation_count: usize,
    linker_invocation_count: usize,
    fallback_count: usize,
    module_visit_count: usize,
    canonical_state_count: usize,
    artifact_checkpoint_authenticated: bool,
    topology_authenticated: bool,
    different_initialization: bool,
    fresh_executor: bool,
    exact_continuation: bool,
    evaluation_state_neutral: bool,
    target_owned_module_published: bool,
}

struct ScaleModuleStateWitness {
    name: String,
    parameter: Parameter,
    kind: StateKind,
    snapshot: ParameterSnapshot,
}

struct ScaleContinuation {
    replay: u64,
    loss: TensorData,
    optimizer_checkpoint: CompiledAdamWCheckpoint,
    module_checkpoint: CompiledModuleAdamWCheckpoint,
}

struct ScaleWarmResumeInput<'a> {
    resume_bundle_path: &'a Path,
    module_checkpoint_path: &'a Path,
    saved_bundle: &'a CompiledAdamWResumeBundle,
    inspection: &'a CompiledAdamWInspection,
    capture_identity: u64,
    evaluation_capture_identity: u64,
    continuation: &'a [ScaleContinuation],
    expected_final_checkpoint: &'a CompiledModuleAdamWCheckpoint,
    expected_final_model: &'a ScaleTransformer,
    expected_final_loss: f64,
}

fn scale_module_state_witness(model: &ScaleTransformer) -> Vec<ScaleModuleStateWitness> {
    let mut states = Vec::new();
    model.visit("", &mut |name, parameter, kind| {
        states.push(ScaleModuleStateWitness {
            name,
            parameter: parameter.clone(),
            kind,
            snapshot: parameter
                .snapshot()
                .expect("the fixed scale module state remains readable"),
        });
    });
    states
}

fn assert_scale_module_witness_unchanged(states: &[ScaleModuleStateWitness]) -> Result<()> {
    for state in states {
        assert_scale_parameter_snapshot_eq(&state.parameter.snapshot()?, &state.snapshot);
    }
    Ok(())
}

fn assert_scale_parameter_snapshot_eq(actual: &ParameterSnapshot, expected: &ParameterSnapshot) {
    assert_eq!(actual.data, expected.data);
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(actual.dtype, expected.dtype);
    assert_eq!(actual.version, expected.version);
    assert_eq!(actual.identity, expected.identity);
    assert_eq!(actual.trainable, expected.trainable);
    assert_eq!(actual.input_name, expected.input_name);
}

fn duration_nanos(duration: std::time::Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).map_err(|_| RustGradError::InvalidIndex)
}

fn checked_duration_sum(
    durations: impl IntoIterator<Item = std::time::Duration>,
) -> Result<std::time::Duration> {
    durations
        .into_iter()
        .try_fold(std::time::Duration::ZERO, |total, duration| {
            total
                .checked_add(duration)
                .ok_or(RustGradError::InvalidIndex)
        })
}

fn warm_program_preparation_evidence(
    report: &NativeCpuProgramPreparationReport,
) -> Result<(
    ScaleWarmProgramPreparationEvidence,
    std::time::Duration,
    std::time::Duration,
)> {
    let phases = report.phases();
    let accounted = checked_duration_sum([
        phases.layout_wall_time(),
        phases.render_wall_time(),
        phases.compiler_process_wall_time(),
        phases.module_load_wall_time(),
        phases.residual_wall_time(),
    ])?;
    assert_eq!(accounted, report.wall_time());
    Ok((
        ScaleWarmProgramPreparationEvidence {
            total_wall_time_ns: duration_nanos(report.wall_time())?,
            layout_wall_time_ns: duration_nanos(phases.layout_wall_time())?,
            render_wall_time_ns: duration_nanos(phases.render_wall_time())?,
            compiler_wall_time_ns: duration_nanos(phases.compiler_process_wall_time())?,
            load_wall_time_ns: duration_nanos(phases.module_load_wall_time())?,
            residual_wall_time_ns: duration_nanos(phases.residual_wall_time())?,
        },
        report.wall_time(),
        phases.render_wall_time(),
    ))
}

fn scale_gradient_oracle(model: &ScaleTransformer) -> Result<ScaleGradientOracle> {
    const ATTENTION_KEEP_MASK: &str = "gradient_probe_attention_keep_mask";

    let mut graph = Graph::new();
    let tokens = graph.input_dtype(TOKENS, [BATCH, TIME], DType::I32);
    let targets = graph.input_dtype(TARGETS, [BATCH, TIME], DType::I32);
    let attention_keep_mask =
        graph.input_dtype(ATTENTION_KEEP_MASK, ATTENTION_KEEP_MASK_SHAPE, DType::Bool);
    let logits = model.forward_evaluation(&mut graph, tokens, attention_keep_mask)?;
    let token_losses = losses(&mut graph, logits, targets)?;
    let numerator = graph.sum_all(token_losses)?;

    let mut parameters = model
        .trainable_parameters()?
        .into_iter()
        .filter(|(name, _)| name != FROZEN_POSITION_WEIGHT)
        .map(|(name, parameter)| {
            let snapshot = parameter.snapshot()?;
            let node = parameter.node(&graph)?;
            let Op::Input { name: input_name } = graph.op(node)? else {
                return Err(RustGradError::SessionTraining {
                    reason: format!(
                        "scale gradient parameter {name:?} is not a direct graph input"
                    ),
                });
            };
            Ok(ScaleGradientParameter {
                name,
                input_name: input_name.clone(),
                value: snapshot.data,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    parameters.sort_by(|left, right| left.name.cmp(&right.name));
    assert_eq!(parameters.len(), GRADIENT_PROBE_PARAMETER_COUNT);
    assert_eq!(
        parameters
            .iter()
            .map(|parameter| parameter.value.len())
            .sum::<usize>(),
        GRADIENT_PROBE_COORDINATE_COUNT
    );
    assert!(
        parameters
            .iter()
            .any(|parameter| parameter.name == "tokens.weight")
    );
    assert!(
        parameters
            .iter()
            .all(|parameter| parameter.name != "lm_head.weight")
    );
    assert!(
        parameters
            .iter()
            .all(|parameter| parameter.name != FROZEN_POSITION_WEIGHT)
    );

    let mut bindings = model.input_bindings(&graph)?;
    for parameter in &parameters {
        assert_eq!(
            bindings.get(&parameter.input_name),
            Some(&parameter.value),
            "scale gradient parameter {:?} must resolve to its exact versioned graph binding",
            parameter.name
        );
    }
    let batch = ScaleBatch::new(1)?;
    bindings.insert(TOKENS.into(), batch.tokens);
    bindings.insert(TARGETS.into(), batch.targets);
    bindings.insert(
        ATTENTION_KEEP_MASK.into(),
        ScaleBatch::attention_keep_mask(1)?,
    );
    Ok(ScaleGradientOracle {
        graph,
        numerator,
        bindings,
        parameters,
    })
}

fn scale_gradient_direction(
    parameters: &[ScaleGradientParameter],
    direction_index: usize,
) -> Result<BTreeMap<String, TensorData>> {
    let raw = |coordinate: usize| -> f64 {
        match direction_index {
            0 => {
                if coordinate.is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            }
            1 => [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0][coordinate % 7],
            _ => unreachable!("the scale probe has exactly two dense directions"),
        }
    };
    let raw_norm = (0..GRADIENT_PROBE_COORDINATE_COUNT)
        .map(raw)
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(raw_norm.is_finite() && raw_norm > 0.0);

    let mut offset = 0_usize;
    let mut direction = BTreeMap::new();
    for parameter in parameters {
        let value = TensorData::from_scalars(
            parameter.value.shape().clone(),
            DType::F32,
            (0..parameter.value.len()).map(|coordinate| {
                Scalar::F(GRADIENT_PROBE_DIRECTION_GAIN * raw(offset + coordinate) / raw_norm)
            }),
        )?;
        offset = offset
            .checked_add(parameter.value.len())
            .expect("the fixed scale gradient inventory cannot overflow");
        assert!(direction.insert(parameter.name.clone(), value).is_none());
    }
    assert_eq!(offset, GRADIENT_PROBE_COORDINATE_COUNT);
    Ok(direction)
}

fn perturbed_scale_bindings(
    oracle: &ScaleGradientOracle,
    direction: &BTreeMap<String, TensorData>,
    epsilon: f64,
) -> Result<HashMap<String, TensorData>> {
    let mut bindings = oracle.bindings.clone();
    let binding_count = bindings.len();
    for parameter in &oracle.parameters {
        let parameter_direction = &direction[&parameter.name];
        let value = TensorData::from_scalars(
            parameter.value.shape().clone(),
            DType::F32,
            (0..parameter.value.len()).map(|coordinate| {
                Scalar::F(
                    parameter.value.scalar_at(coordinate).as_f64()
                        + epsilon * parameter_direction.scalar_at(coordinate).as_f64(),
                )
            }),
        )?;
        let binding = bindings
            .get_mut(&parameter.input_name)
            .expect("the authenticated versioned parameter binding must remain present");
        let original = std::mem::replace(binding, value);
        assert_eq!(&original, &parameter.value);
        assert_ne!(&*binding, &parameter.value);
    }
    assert_eq!(bindings.len(), binding_count);
    Ok(bindings)
}

fn scale_numerator(
    oracle: &ScaleGradientOracle,
    direction: &BTreeMap<String, TensorData>,
    epsilon: f64,
) -> Result<f64> {
    Ok(CpuBackend
        .execute(
            &oracle.graph,
            oracle.numerator,
            &perturbed_scale_bindings(oracle, direction, epsilon)?,
        )?
        .scalar_at(0)
        .as_f64())
}

fn scale_gradient_projection(
    gradients: &BTreeMap<String, TensorData>,
    direction: &BTreeMap<String, TensorData>,
) -> f64 {
    gradients
        .iter()
        .map(|(name, gradient)| {
            let parameter_direction = &direction[name];
            assert_eq!(gradient.shape(), parameter_direction.shape());
            (0..gradient.len())
                .map(|coordinate| {
                    gradient.scalar_at(coordinate).as_f64()
                        * parameter_direction.scalar_at(coordinate).as_f64()
                })
                .sum::<f64>()
        })
        .sum()
}

fn collect_scale_gradient_evidence() -> Result<ScaleGradientEvidence> {
    let oracle_model = ScaleTransformer::new(SCALE_SEED)?;
    let oracle = scale_gradient_oracle(&oracle_model)?;
    let mut traversal = BTreeMap::new();
    oracle_model.visit("", &mut |name, parameter, _| {
        traversal.insert(name, parameter.id());
    });
    assert_eq!(traversal["tokens.weight"], traversal["lm_head.weight"]);

    let compiled_model = ScaleTransformer::new(SCALE_SEED)?;
    assert_eq!(compiled_model.state_dict()?, oracle_model.state_dict()?);
    let plan = CompiledModuleAdamWPlan::compile_graph_with_ignore_index(
        scale_config()?,
        compiled_model,
        build_gradient_probe,
    )
    .map_err(|error| error.into_parts().1)?;
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let mut runtime = plan
        .prepare(&target)
        .map_err(|error| error.into_parts().1)?;
    let preparation = runtime.native_cpu_preparation_report();
    let preparation_fallback_count: usize = [
        Some(preparation.main()),
        preparation.accumulation(),
        preparation.partial_flush(),
        preparation.zero_grad(),
    ]
    .into_iter()
    .flatten()
    .map(|program| program.fallback_count())
    .sum();
    assert_eq!(preparation_fallback_count, 0);
    assert_eq!(
        runtime.parameter_snapshots()?,
        oracle
            .parameters
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.value.clone()))
            .collect::<BTreeMap<_, _>>()
    );
    let step = runtime.step_batch_scheduled(ScaleBatch::new(1)?)?;
    assert!(!step.did_update());
    assert_eq!(step.accumulation_index(), 1);
    assert_eq!(step.loss_weight(), GRADIENT_PROBE_VALID_TOKEN_COUNT);
    assert_eq!(
        runtime.checkpoint()?.info().accumulated_token_count(),
        Some(GRADIENT_PROBE_VALID_TOKEN_COUNT)
    );
    assert_strict_native(&step);
    let gradients = runtime.gradient_accumulator_snapshots()?;
    assert_eq!(gradients.len(), GRADIENT_PROBE_PARAMETER_COUNT);
    assert_eq!(
        gradients.values().map(TensorData::len).sum::<usize>(),
        GRADIENT_PROBE_COORDINATE_COUNT
    );
    assert_eq!(
        gradients.keys().cloned().collect::<Vec<_>>(),
        oracle
            .parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect::<Vec<_>>()
    );

    let mut projections = Vec::new();
    for (direction_index, direction_id) in ["alternating_dense_v1", "seven_phase_dense_v1"]
        .into_iter()
        .enumerate()
    {
        let direction = scale_gradient_direction(&oracle.parameters, direction_index)?;
        let coarse = (scale_numerator(&oracle, &direction, GRADIENT_PROBE_EPSILON)?
            - scale_numerator(&oracle, &direction, -GRADIENT_PROBE_EPSILON)?)
            / (2.0 * GRADIENT_PROBE_EPSILON);
        let fine = (scale_numerator(&oracle, &direction, GRADIENT_PROBE_EPSILON / 2.0)?
            - scale_numerator(&oracle, &direction, -GRADIENT_PROBE_EPSILON / 2.0)?)
            / GRADIENT_PROBE_EPSILON;
        let expected_projection = (4.0 * fine - coarse) / 3.0;
        let actual_projection = scale_gradient_projection(&gradients, &direction);
        let absolute_error = (actual_projection - expected_projection).abs();
        let tolerance = GRADIENT_PROBE_TOLERANCE
            * 1.0f64
                .max(actual_projection.abs())
                .max(expected_projection.abs());
        assert!(
            actual_projection.is_finite()
                && expected_projection.is_finite()
                && expected_projection.abs() > tolerance
                && actual_projection.abs() > tolerance
                && absolute_error <= tolerance,
            "scale gradient direction {direction_id} is not mutation-sensitive or differs: actual={actual_projection}, expected={expected_projection}, error={absolute_error}, tolerance={tolerance}"
        );
        projections.push(ScaleGradientProjectionEvidence {
            direction_id,
            epsilon: GRADIENT_PROBE_EPSILON,
            expected_projection,
            actual_projection,
            absolute_error,
            tolerance,
            mutation_sensitive: true,
        });
    }

    Ok(ScaleGradientEvidence {
        replay: step.step(),
        accumulation_index: step.accumulation_index(),
        did_update: step.did_update(),
        dropout_probability: 0.0,
        valid_token_count: step.loss_weight(),
        active_parameter_count: gradients.len(),
        active_coordinate_count: gradients.values().map(TensorData::len).sum(),
        tied_output_head_is_canonical_alias: true,
        frozen_position_is_absent: !gradients.contains_key(FROZEN_POSITION_WEIGHT),
        native_preparation_fallback_count: preparation_fallback_count,
        native_replay_fallback_count: step.report().fallback_count(),
        projections,
    })
}

fn collect_scale_warm_resume_evidence(
    input: ScaleWarmResumeInput<'_>,
) -> std::result::Result<ScaleWarmResumeEvidence, Box<dyn Error>> {
    let ScaleWarmResumeInput {
        resume_bundle_path,
        module_checkpoint_path,
        saved_bundle,
        inspection,
        capture_identity,
        evaluation_capture_identity,
        continuation,
        expected_final_checkpoint,
        expected_final_model,
        expected_final_loss,
    } = input;
    let decode_started = Instant::now();
    let resume_bundle = CompiledAdamWResumeBundle::load_file(resume_bundle_path)?;
    let artifact_decode_wall_time_ns = duration_nanos(decode_started.elapsed())?;
    let persisted_checkpoint = CompiledModuleAdamWCheckpoint::load_file(module_checkpoint_path)?;
    assert_eq!(&resume_bundle, saved_bundle);
    assert_eq!(resume_bundle.checkpoint(), &persisted_checkpoint);
    assert_eq!(
        resume_bundle.checkpoint().as_bytes(),
        persisted_checkpoint.as_bytes()
    );

    let destination = ScaleTransformer::new(WARM_RESUME_SEED)?;
    let destination_initial = destination.state_dict()?;
    assert_ne!(destination_initial, expected_final_model.state_dict()?);
    assert_ne!(
        destination.positions.weight.value()?,
        expected_final_model.positions.weight.value()?,
        "the policy-frozen position table must come from a different initialization"
    );
    let destination_states = scale_module_state_witness(&destination);
    let destination_tied_identity = destination.tokens.weight.id();
    assert_eq!(destination_states.len(), 37);
    assert_eq!(destination_initial.tensors().len(), 36);
    assert_eq!(
        destination_states
            .iter()
            .find(|state| state.name == "lm_head.weight")
            .expect("the destination exposes the tied output head")
            .snapshot
            .identity,
        destination_tied_identity
    );

    let owner_restore_started = Instant::now();
    let restored_plan =
        CompiledModuleAdamWPlan::restore_from_resume_bundle(destination, &resume_bundle)
            .map_err(|error| error.into_parts().1)?;
    let owner_restore_wall_time_ns = duration_nanos(owner_restore_started.elapsed())?;
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(
        restored_plan.evaluation_capture_identity(),
        Some(evaluation_capture_identity)
    );
    let restored_inspection = restored_plan.inspection()?;
    assert_eq!(restored_inspection.initial_replay_step(), CHECKPOINT_REPLAY);
    assert_eq!(restored_inspection.main(), inspection.main());
    assert_eq!(
        restored_inspection.accumulation(),
        inspection.accumulation()
    );
    assert_eq!(
        restored_inspection.partial_flush(),
        inspection.partial_flush()
    );
    assert_eq!(restored_inspection.zero_grad(), inspection.zero_grad());
    assert_eq!(restored_inspection.evaluation(), inspection.evaluation());
    assert_eq!(
        restored_inspection.recurrent_state_count(),
        inspection.recurrent_state_count()
    );
    assert_eq!(
        restored_inspection.recurrent_state_bytes(),
        inspection.recurrent_state_bytes()
    );
    assert_eq!(
        restored_plan.program_artifact()?.as_bytes(),
        resume_bundle.program_artifact().as_bytes()
    );
    assert_scale_module_witness_unchanged(&destination_states)?;

    let warm_executor = CapturedReplayExecutor::default();
    let warm_target = NativeCpuSessionTarget::new(&warm_executor).vectorized(true);
    let preparation_started = Instant::now();
    let mut resumed = restored_plan
        .prepare(&warm_target)
        .map_err(|error| error.into_parts().1)?;
    let preparation_wall_time = preparation_started.elapsed();
    let preparation_wall_time_ns = duration_nanos(preparation_wall_time)?;
    assert_scale_module_witness_unchanged(&destination_states)?;
    assert_eq!(
        resumed.checkpoint()?,
        resume_bundle.checkpoint().optimizer_checkpoint().clone()
    );

    let preparation = resumed.native_cpu_preparation_report();
    assert_eq!(preparation.compiler_process_count(), 0);
    let main = preparation.main();
    let accumulation = preparation
        .accumulation()
        .expect("the scale workload has an accumulation program");
    let partial_flush = preparation
        .partial_flush()
        .expect("the scale workload has a partial-flush program");
    let zero_grad = preparation
        .zero_grad()
        .expect("the scale workload has a zero-grad program");
    let evaluation = preparation
        .evaluation()
        .expect("the scale workload has an evaluation program");
    let programs = [main, accumulation, partial_flush, zero_grad, evaluation];
    let program_count = programs.len();
    let mut loaded_module_count = 0_usize;
    let mut durable_artifact_cache_hit_count = 0_usize;
    let mut durable_artifact_cache_miss_count = 0_usize;
    let mut compiler_invocation_count = 0_usize;
    let mut linker_invocation_count = 0_usize;
    let mut fallback_count = 0_usize;
    for program in &programs {
        assert!(program.native_item_count() > 0);
        let work = program.work();
        assert_eq!(work.loaded_module_count(), 1);
        assert_eq!(work.durable_artifact_cache_miss_count(), 0);
        assert_eq!(work.durable_artifact_cache_hit_count(), 1);
        assert_eq!(work.compiler_invocation_count(), 0);
        assert_eq!(work.linker_invocation_count(), 0);
        assert_eq!(program.fallback_count(), 0);
        loaded_module_count = loaded_module_count
            .checked_add(work.loaded_module_count())
            .expect("the fixed warm module inventory cannot overflow");
        durable_artifact_cache_hit_count = durable_artifact_cache_hit_count
            .checked_add(work.durable_artifact_cache_hit_count())
            .expect("the fixed durable-hit inventory cannot overflow");
        durable_artifact_cache_miss_count = durable_artifact_cache_miss_count
            .checked_add(work.durable_artifact_cache_miss_count())
            .expect("the fixed durable-miss inventory cannot overflow");
        compiler_invocation_count = compiler_invocation_count
            .checked_add(work.compiler_invocation_count())
            .expect("the fixed compiler inventory cannot overflow");
        linker_invocation_count = linker_invocation_count
            .checked_add(work.linker_invocation_count())
            .expect("the fixed linker inventory cannot overflow");
        fallback_count = fallback_count
            .checked_add(program.fallback_count())
            .expect("the fixed fallback inventory cannot overflow");
    }
    assert_eq!(loaded_module_count, 5);
    assert_eq!(durable_artifact_cache_hit_count, 5);
    assert_eq!(durable_artifact_cache_miss_count, 0);
    let render_capsule_hit_count = preparation.render_capsule_hit_count();
    let render_capsule_miss_count = preparation.render_capsule_miss_count();
    let local_render_job_count = preparation.local_render_job_count();
    assert_eq!(render_capsule_hit_count, 5);
    assert_eq!(render_capsule_miss_count, 0);
    assert_eq!(local_render_job_count, 0);

    let (main_preparation, main_wall, main_render) = warm_program_preparation_evidence(main)?;
    let (accumulation_preparation, accumulation_wall, accumulation_render) =
        warm_program_preparation_evidence(accumulation)?;
    let (partial_flush_preparation, partial_flush_wall, partial_flush_render) =
        warm_program_preparation_evidence(partial_flush)?;
    let (zero_grad_preparation, zero_grad_wall, zero_grad_render) =
        warm_program_preparation_evidence(zero_grad)?;
    let (evaluation_preparation, evaluation_wall, evaluation_render) =
        warm_program_preparation_evidence(evaluation)?;
    let summed_program_wall_time = checked_duration_sum([
        main_wall,
        accumulation_wall,
        partial_flush_wall,
        zero_grad_wall,
        evaluation_wall,
    ])?;
    let summed_render_wall_time = checked_duration_sum([
        main_render,
        accumulation_render,
        partial_flush_render,
        zero_grad_render,
        evaluation_render,
    ])?;
    let module_overlap_wall_time = preparation.parallel_module_overlap_wall_time();
    let render_overlap_wall_time = preparation.parallel_render_overlap_wall_time();
    let effective_program_wall_time = summed_program_wall_time
        .checked_sub(module_overlap_wall_time)
        .and_then(|duration| duration.checked_sub(render_overlap_wall_time))
        .ok_or(RustGradError::InvalidIndex)?;
    let runtime_overhead_wall_time = preparation_wall_time
        .checked_sub(effective_program_wall_time)
        .ok_or(RustGradError::InvalidIndex)?;
    let reconstructed_preparation_wall_time = runtime_overhead_wall_time
        .checked_add(summed_program_wall_time)
        .and_then(|duration| duration.checked_sub(module_overlap_wall_time))
        .and_then(|duration| duration.checked_sub(render_overlap_wall_time))
        .ok_or(RustGradError::InvalidIndex)?;
    assert_eq!(reconstructed_preparation_wall_time, preparation_wall_time);
    let effective_render_wall_time = summed_render_wall_time
        .checked_sub(render_overlap_wall_time)
        .ok_or(RustGradError::InvalidIndex)?;
    assert!(effective_render_wall_time <= preparation_wall_time);
    assert!(!preparation_wall_time.is_zero());
    let effective_render_fraction =
        effective_render_wall_time.as_secs_f64() / preparation_wall_time.as_secs_f64();
    assert!(effective_render_fraction.is_finite());
    let warm_preparation = ScaleWarmPreparationEvidence {
        main: main_preparation,
        accumulation: accumulation_preparation,
        partial_flush: partial_flush_preparation,
        zero_grad: zero_grad_preparation,
        evaluation: evaluation_preparation,
        summed_program_wall_time_ns: duration_nanos(summed_program_wall_time)?,
        runtime_overhead_wall_time_ns: duration_nanos(runtime_overhead_wall_time)?,
        module_overlap_wall_time_ns: duration_nanos(module_overlap_wall_time)?,
        render_overlap_wall_time_ns: duration_nanos(render_overlap_wall_time)?,
        effective_render_wall_time_ns: duration_nanos(effective_render_wall_time)?,
        effective_render_fraction,
    };

    assert_eq!(
        continuation.len(),
        usize::try_from(REPLAYS - CHECKPOINT_REPLAY)?
    );
    for expected in continuation {
        let step = resumed.step_batch_commit_only_scheduled(ScaleBatch::new(expected.replay)?)?;
        assert_strict_native(&step);
        assert_eq!(step.loss(), &expected.loss);
        assert_eq!(resumed.checkpoint()?, expected.optimizer_checkpoint);
        assert_eq!(resumed.module_checkpoint()?, expected.module_checkpoint);
    }
    assert_eq!(resumed.module_checkpoint()?, *expected_final_checkpoint);

    let before_evaluation = resumed.module_checkpoint()?;
    let warm_final_loss = mean_evaluation_loss(&mut resumed, evaluation_capture_identity)?;
    assert_eq!(warm_final_loss.to_bits(), expected_final_loss.to_bits());
    assert_eq!(resumed.module_checkpoint()?, before_evaluation);
    let (resumed_model, resumed_final_checkpoint) = resumed
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(resumed_final_checkpoint, *expected_final_checkpoint);
    assert_eq!(
        resumed_model.state_dict()?,
        expected_final_model.state_dict()?
    );

    let expected_states = scale_module_state_witness(expected_final_model);
    let resumed_states = scale_module_state_witness(&resumed_model);
    assert_eq!(resumed_states.len(), destination_states.len());
    assert_eq!(resumed_states.len(), expected_states.len());
    for ((before, expected), actual) in destination_states
        .iter()
        .zip(&expected_states)
        .zip(&resumed_states)
    {
        assert_eq!(actual.name, before.name);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.kind, before.kind);
        assert_eq!(actual.kind, expected.kind);
        assert_eq!(actual.snapshot.data, expected.snapshot.data);
        assert_eq!(actual.snapshot.identity, before.snapshot.identity);
        assert_eq!(actual.snapshot.trainable, before.snapshot.trainable);
        assert_eq!(
            actual.snapshot.version,
            before
                .snapshot
                .version
                .checked_add(1)
                .expect("successful warm publication cannot overflow a version")
        );
    }
    assert_eq!(resumed_model.tokens.weight.id(), destination_tied_identity);
    assert_eq!(
        resumed_states
            .iter()
            .find(|state| state.name == "lm_head.weight")
            .expect("the finished destination retains the tied output head")
            .snapshot
            .identity,
        destination_tied_identity
    );

    Ok(ScaleWarmResumeEvidence {
        replay_from: CHECKPOINT_REPLAY,
        replay_to: REPLAYS,
        resume_bundle_bytes: resume_bundle.as_bytes().len(),
        module_checkpoint_bytes: persisted_checkpoint.as_bytes().len(),
        artifact_decode_wall_time_ns,
        owner_restore_wall_time_ns,
        preparation_wall_time_ns,
        preparation: warm_preparation,
        capture_identity,
        evaluation_capture_identity,
        program_count,
        loaded_module_count,
        durable_artifact_cache_hit_count,
        durable_artifact_cache_miss_count,
        render_capsule_hit_count,
        render_capsule_miss_count,
        local_render_job_count,
        compiler_invocation_count,
        linker_invocation_count,
        fallback_count,
        module_visit_count: resumed_states.len(),
        canonical_state_count: resumed_model.state_dict()?.tensors().len(),
        artifact_checkpoint_authenticated: true,
        topology_authenticated: true,
        different_initialization: true,
        fresh_executor: true,
        exact_continuation: true,
        evaluation_state_neutral: true,
        target_owned_module_published: true,
    })
}

fn mean_evaluation_loss(
    runtime: &mut rustgrad::CompiledModuleAdamWSession<
        ScaleTransformer,
        NativeCpuCompiledAdamW<'_>,
    >,
    evaluation_capture_identity: u64,
) -> Result<f64> {
    Ok(scale_evaluation_samples(runtime, evaluation_capture_identity)?.0)
}

fn scale_evaluation_samples(
    runtime: &mut rustgrad::CompiledModuleAdamWSession<
        ScaleTransformer,
        NativeCpuCompiledAdamW<'_>,
    >,
    evaluation_capture_identity: u64,
) -> Result<(f64, Vec<ScaleEvaluationSampleEvidence>)> {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0_u64;
    let mut samples = Vec::new();
    for replay in 1..=ACCUMULATION_STEPS {
        let evaluation = runtime.evaluate_batch(ScaleBatch::new(replay)?)?;
        assert_eq!(evaluation.capture_identity(), evaluation_capture_identity);
        assert_eq!(evaluation.report().fallback_count(), 0);
        assert!(evaluation.report().executed_native_item_count() > 0);
        assert!(evaluation.report().module_dispatch_count() > 0);
        let valid_token_count = evaluation.loss_weight();
        assert_eq!(valid_token_count, if replay == 1 { 22 } else { 16 });
        let token_mean_loss = evaluation.loss().scalar_at(0).as_f64();
        assert!(token_mean_loss.is_finite());
        weighted_sum += token_mean_loss * valid_token_count as f64;
        weight_sum = weight_sum
            .checked_add(valid_token_count)
            .expect("the fixed evaluation weight cannot overflow");
        samples.push(ScaleEvaluationSampleEvidence {
            batch_replay: replay,
            capture_identity: evaluation.capture_identity(),
            token_mean_loss,
            valid_token_count,
            native_fallback_count: evaluation.report().fallback_count(),
            executed_native_item_count: evaluation.report().executed_native_item_count(),
            module_dispatch_count: evaluation.report().module_dispatch_count(),
        });
    }
    assert_eq!(weight_sum, 38);
    Ok((weighted_sum / weight_sum as f64, samples))
}

fn evaluate_scale_frontier(
    runtime: &mut rustgrad::CompiledModuleAdamWSession<
        ScaleTransformer,
        NativeCpuCompiledAdamW<'_>,
    >,
    replay: u64,
    optimizer_step: u64,
    evaluation_capture_identity: u64,
) -> Result<ScaleEvaluationFrontierEvidence> {
    let before = runtime.module_checkpoint()?;
    let info = before.optimizer_checkpoint().info();
    assert_eq!(info.replay_step(), replay);
    assert_eq!(info.optimizer_step(), optimizer_step);
    assert_eq!(info.accumulation_index(), 0);
    let (token_mean_loss, samples) =
        scale_evaluation_samples(runtime, evaluation_capture_identity)?;
    assert!(token_mean_loss.is_finite());
    assert_eq!(runtime.module_checkpoint()?, before);
    Ok(ScaleEvaluationFrontierEvidence {
        replay,
        optimizer_step,
        accumulation_index: 0,
        token_mean_loss,
        checkpoint_state_neutral: true,
        samples,
    })
}

fn assert_strict_native(step: &NativeCpuCompiledAdamWStepResult) {
    assert_eq!(step.report().fallback_count(), 0);
    assert!(step.report().executed_native_item_count() > 0);
    assert!(step.report().module_dispatch_count() > 0);
}

fn required_path(name: &str) -> std::result::Result<PathBuf, Box<dyn Error>> {
    let value = env::var_os(name).ok_or_else(|| format!("{name} must name an evidence file"))?;
    Ok(PathBuf::from(value))
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let git_sha = env::var("RUSTGRAD_EVIDENCE_GIT_SHA")?;
    if git_sha.len() != 40
        || !git_sha
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("RUSTGRAD_EVIDENCE_GIT_SHA must be a lowercase full Git SHA".into());
    }
    let scoreboard_path = required_path("RUSTGRAD_LARGER_SCOREBOARD_PATH")?;
    let objective_path = required_path("RUSTGRAD_LARGER_OBJECTIVE_PATH")?;
    let resume_bundle_path = required_path("RUSTGRAD_LARGER_RESUME_BUNDLE_PATH")?;
    let module_checkpoint_path = required_path("RUSTGRAD_LARGER_MODULE_CHECKPOINT_PATH")?;

    let compile_started = Instant::now();
    let plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
        scale_config()?,
        CompiledDropoutConfig::new(CompiledDropoutKey([0x5ca1_e001, 0x5ca1_e002])),
        ScaleTransformer::new(SCALE_SEED)?,
        build_training,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph_and_ignore_index(build_evaluation)
    .map_err(|error| error.into_parts().1)?;
    let compile_wall_time = compile_started.elapsed();
    let capture_identity = plan.capture_identity();
    let evaluation_capture_identity = plan
        .evaluation_capture_identity()
        .expect("the larger Transformer evaluator is attached");
    let program_artifact = plan.program_artifact()?;
    let inspection = plan.inspection()?;
    assert_eq!(inspection.initial_replay_step(), 0);
    assert!(inspection.recurrent_state_count() > 0);
    assert!(inspection.recurrent_state_bytes() > 0);

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let prepare_started = Instant::now();
    let mut runtime = plan
        .prepare(&target)
        .map_err(|error| error.into_parts().1)?;
    let prepare_wall_time = prepare_started.elapsed();
    let preparation = runtime.native_cpu_preparation_report();
    for program in [
        Some(preparation.main()),
        preparation.accumulation(),
        preparation.partial_flush(),
        preparation.zero_grad(),
        preparation.evaluation(),
    ]
    .into_iter()
    .flatten()
    {
        assert!(program.native_item_count() > 0);
        assert_eq!(program.fallback_count(), 0);
    }
    let mut scoreboard = NativeTrainingScoreboard::new(
        inspection.clone(),
        preparation,
        compile_wall_time,
        prepare_wall_time,
    )?;

    let mut evaluation_trajectory = vec![evaluate_scale_frontier(
        &mut runtime,
        0,
        0,
        evaluation_capture_identity,
    )?];
    let initial_loss = evaluation_trajectory[0].token_mean_loss;

    let mut pending_checkpoint = None;
    let mut pending_checkpoint_bytes = None;
    let mut pending_module_checkpoint_bytes = None;
    let mut saved_resume_bundle = None;
    let mut continuation = Vec::new();
    let mut replay_progress = Vec::new();
    for replay in 1..=REPLAYS {
        let step = runtime.step_batch_commit_only_scheduled(ScaleBatch::new(replay)?)?;
        assert_strict_native(&step);
        assert_eq!(step.did_update(), replay % ACCUMULATION_STEPS == 0);
        assert_eq!(step.optimizer_step(), replay / ACCUMULATION_STEPS);
        assert_eq!(step.accumulation_index(), replay % ACCUMULATION_STEPS);
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        scoreboard.record_step(&step)?;
        replay_progress.push(ScaleReplayProgressEvidence {
            replay,
            optimizer_step: step.optimizer_step(),
            accumulation_index: step.accumulation_index(),
            did_update: step.did_update(),
        });
        if step.did_update() {
            evaluation_trajectory.push(evaluate_scale_frontier(
                &mut runtime,
                replay,
                step.optimizer_step(),
                evaluation_capture_identity,
            )?);
        }
        if replay == CHECKPOINT_REPLAY {
            let checkpoint = runtime.checkpoint()?;
            assert_eq!(checkpoint.info().replay_step(), CHECKPOINT_REPLAY);
            assert_eq!(checkpoint.info().optimizer_step(), 1);
            assert_eq!(checkpoint.info().accumulation_index(), 1);
            let module_checkpoint = runtime.module_checkpoint()?;
            assert_eq!(module_checkpoint.optimizer_checkpoint(), &checkpoint);
            assert_eq!(
                module_checkpoint.evaluation_capture_identity(),
                Some(evaluation_capture_identity)
            );
            let resume_bundle = CompiledAdamWResumeBundle::new(
                program_artifact.clone(),
                module_checkpoint.clone(),
            )?;
            resume_bundle.save_file(&resume_bundle_path)?;
            module_checkpoint.save_file(&module_checkpoint_path)?;
            pending_checkpoint_bytes = Some(u64::try_from(checkpoint.as_bytes().len())?);
            pending_module_checkpoint_bytes =
                Some(u64::try_from(module_checkpoint.as_bytes().len())?);
            pending_checkpoint = Some(checkpoint);
            saved_resume_bundle = Some(resume_bundle);
        } else if replay > CHECKPOINT_REPLAY {
            let optimizer_checkpoint = runtime.checkpoint()?;
            let module_checkpoint = runtime.module_checkpoint()?;
            assert_eq!(
                module_checkpoint.optimizer_checkpoint(),
                &optimizer_checkpoint
            );
            continuation.push(ScaleContinuation {
                replay,
                loss: step.loss().clone(),
                optimizer_checkpoint,
                module_checkpoint,
            });
        }
    }
    let terminal_checkpoint_started = Instant::now();
    let uninterrupted_final = runtime.checkpoint()?;
    scoreboard.observe_checkpoint(&uninterrupted_final, terminal_checkpoint_started.elapsed())?;
    assert_eq!(uninterrupted_final.info().replay_step(), REPLAYS);
    assert_eq!(uninterrupted_final.info().optimizer_step(), 3);
    assert_eq!(uninterrupted_final.info().accumulation_index(), 0);
    assert_eq!(evaluation_trajectory.len(), 4);
    let uninterrupted_final_loss = evaluation_trajectory
        .last()
        .expect("the final committed window is evaluated")
        .token_mean_loss;

    let pending_checkpoint = pending_checkpoint.expect("replay three checkpoints a pending window");
    let pending_checkpoint_bytes =
        pending_checkpoint_bytes.expect("the pending checkpoint records its exact byte count");
    let pending_module_checkpoint_bytes = pending_module_checkpoint_bytes
        .expect("the pending module checkpoint records its exact byte count");
    assert_eq!(
        pending_checkpoint_bytes,
        u64::try_from(pending_checkpoint.as_bytes().len())?
    );
    runtime.restore_checkpoint_in_place(&pending_checkpoint)?;
    assert_eq!(runtime.checkpoint()?, pending_checkpoint);
    for expected in &continuation {
        let replay = expected.replay;
        let step = runtime.step_batch_commit_only_scheduled(ScaleBatch::new(replay)?)?;
        assert_strict_native(&step);
        assert_eq!(step.loss(), &expected.loss);
        assert_eq!(runtime.checkpoint()?, expected.optimizer_checkpoint);
        assert_eq!(runtime.module_checkpoint()?, expected.module_checkpoint);
    }
    assert_eq!(runtime.checkpoint()?, uninterrupted_final);
    let before_final_evaluation = runtime.checkpoint()?;
    let final_loss = mean_evaluation_loss(&mut runtime, evaluation_capture_identity)?;
    assert_eq!(runtime.checkpoint()?, before_final_evaluation);
    assert_eq!(final_loss.to_bits(), uninterrupted_final_loss.to_bits());
    assert!(
        final_loss < initial_loss,
        "larger compiled Transformer loss did not decrease: {initial_loss} -> {final_loss}"
    );

    let report = scoreboard.report()?;
    assert_eq!(report.fallback_count(), 0);
    assert_eq!(report.successful_replay_count(), REPLAYS);
    assert_eq!(
        report.recurrent_state_count(),
        u64::try_from(inspection.recurrent_state_count())?
    );
    assert_eq!(
        report.recurrent_state_bytes(),
        u64::try_from(inspection.recurrent_state_bytes())?
    );
    let terminal_checkpoint_bytes = report
        .checkpoint_byte_count()
        .expect("the scoreboard records the terminal checkpoint");
    assert_eq!(
        terminal_checkpoint_bytes,
        u64::try_from(uninterrupted_final.as_bytes().len())?
    );
    fs::write(&scoreboard_path, report.to_json_bytes()?)?;

    let expected_final_module_checkpoint = runtime.module_checkpoint()?;
    assert_eq!(
        expected_final_module_checkpoint.optimizer_checkpoint(),
        &uninterrupted_final
    );
    assert_eq!(
        continuation
            .last()
            .expect("the fixed continuation is nonempty")
            .module_checkpoint,
        expected_final_module_checkpoint
    );
    let (expected_final_model, published_final_module_checkpoint) = runtime
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(
        published_final_module_checkpoint,
        expected_final_module_checkpoint
    );
    let saved_resume_bundle =
        saved_resume_bundle.expect("replay three persists one portable resume bundle");
    let warm_resume_evidence = collect_scale_warm_resume_evidence(ScaleWarmResumeInput {
        resume_bundle_path: &resume_bundle_path,
        module_checkpoint_path: &module_checkpoint_path,
        saved_bundle: &saved_resume_bundle,
        inspection: &inspection,
        capture_identity,
        evaluation_capture_identity,
        continuation: &continuation,
        expected_final_checkpoint: &expected_final_module_checkpoint,
        expected_final_model: &expected_final_model,
        expected_final_loss: final_loss,
    })?;

    // Run the shape-sensitive autograd probe only after the primary scoreboard
    // has been finalized so its additional compilation cannot affect the
    // observed six-replay preparation or execution timings.
    let gradient_evidence = collect_scale_gradient_evidence()?;

    let objective = json!({
        "schema_version": 6,
        "git_sha": git_sha,
        "workload": {
            "batch": BATCH,
            "time": TIME,
            "vocabulary": VOCAB,
            "embedding": EMBEDDING,
            "heads": HEADS,
            "feed_forward": FEED_FORWARD,
            "blocks": 2,
            "compile_count": 1,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "replays": REPLAYS
        },
        "objective": {
            "initial_token_mean_loss": initial_loss,
            "final_token_mean_loss": final_loss,
            "decreased": true,
            "evaluation_trajectory": evaluation_trajectory
        },
        "progress": {
            "replays": replay_progress,
            "pending_resume_checkpoint": {
                "replay": CHECKPOINT_REPLAY,
                "optimizer_step": 1,
                "accumulation_index": 1,
                "bytes": pending_checkpoint_bytes,
                "module_checkpoint_bytes": pending_module_checkpoint_bytes
            },
            "terminal_scoreboard_checkpoint": {
                "replay": REPLAYS,
                "optimizer_step": 3,
                "accumulation_index": 0,
                "bytes": terminal_checkpoint_bytes
            },
            "final_replay": REPLAYS,
            "final_optimizer_step": 3,
            "final_accumulation_index": 0,
            "exact_resume": true
        },
        "native": {
            "fallback_count": report.fallback_count(),
            "successful_replay_count": report.successful_replay_count(),
            "recurrent_state_count": report.recurrent_state_count(),
            "recurrent_state_bytes": report.recurrent_state_bytes()
        },
        "warm_resume": warm_resume_evidence,
        "gradient_probe": gradient_evidence
    });
    let mut objective_bytes = serde_json::to_vec_pretty(&objective)?;
    objective_bytes.push(b'\n');
    fs::write(&objective_path, objective_bytes)?;
    Ok(())
}
