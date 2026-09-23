use rustgrad::nn::{Embedding, LayerNorm, Mode, ModeModuleForward, StateKind};
use rustgrad::runtime::metal::MetalRuntime;
use rustgrad::{
    Backend, CapturedReplayExecutor, CompiledAdamWCheckpoint, CompiledAdamWConfig,
    CompiledAdamWGraph, CompiledAdamWIgnoreIndexContext, CompiledAdamWPlan,
    CompiledAdamWResumeBundle, CompiledAdamWRuntime, CompiledAdamWStep,
    CompiledCheckpointRestoreRuntime, CompiledCheckpointRuntime, CompiledDropoutConfig,
    CompiledDropoutKey, CompiledEvaluation, CompiledEvaluationRuntime, CompiledInputBatch,
    CompiledInputSpec, CompiledModuleAdamWCheckpoint, CompiledModuleAdamWPlan,
    CompiledModuleAdamWSession, CompiledMultiStepLr, CompiledScheduledAdamWRuntime,
    CompiledTrainingRatePolicyRuntime, CompiledTrainingRatePolicyWindowCommitRuntime,
    CompiledTrainingRuntime, CompiledTrainingStep, CompiledTrainingWindowCommit,
    CompiledTrainingWindowCommitRuntime, CompiledTrainingWindowResetRuntime,
    CompiledTrainingWindowRuntime, CompiledTrainingWindowStep, CpuBackend, CpuCompiledAdamW,
    CpuNonFinitePolicy, CpuSessionTarget, DType, Graph, LossOptions, MetalSessionTarget, Module,
    NativeCpuCompiledAdamW, NativeCpuCompiledAdamWStepResult, NativeCpuCompiledEvaluationResult,
    NativeCpuRenderCapsuleProgramRole, NativeCpuSessionTarget, NativeTrainingScoreboard, NodeId,
    Parameter, Reduction, Result, Scalar, Shape, TensorData, TrainingDropoutProvider,
    TransformerBlock, sparse_categorical_cross_entropy,
};
use serde::{Deserialize, Serialize};
use std::{
    cell::Cell,
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

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
const INITIAL_STEPS: usize = 4;
const RESUMED_STEPS: usize = 3;
const POLICY_FROZEN: &str = "block.ff1.0";
const FILE_RESUME_POLICY_FROZEN: &str = "positions.weight";
const LOSS_MASK: &str = "loss_mask";
const FILE_RESUME_IGNORE_INDEX: i32 = -100;
// Per-sample key validity broadcasts across heads and query positions.
const ATTENTION_KEEP_MASK_SHAPE: [usize; 4] = [BATCH, 1, 1, TIME];

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

struct FileResumeTransformer {
    tokens: Embedding,
    positions: Embedding,
    first: TransformerBlock,
    second: TransformerBlock,
    norm: LayerNorm,
    frozen_scale: Parameter,
    running_marker: Parameter,
}

impl FileResumeTransformer {
    fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            tokens: Embedding::new_static(VOCAB, EMBEDDING, None, seed)?,
            positions: Embedding::new_static(TIME, EMBEDDING, None, seed.wrapping_add(1))?,
            first: TransformerBlock::new_static(EMBEDDING, 1, 4, true, 0.25, seed.wrapping_add(2))?
                .with_causal_attention(true)
                .with_attention_dropout(0.25)?,
            second: TransformerBlock::new_static(
                EMBEDDING,
                1,
                4,
                true,
                0.25,
                seed.wrapping_add(3),
            )?
            .with_causal_attention(true)
            .with_attention_dropout(0.25)?,
            norm: LayerNorm::new_static([EMBEDDING], 1e-5, true)?,
            frozen_scale: Parameter::new(TensorData::scalar(1.0), false),
            running_marker: Parameter::new(TensorData::scalar(3.0), false),
        })
    }

    fn forward(
        &self,
        graph: &mut Graph,
        tokens: NodeId,
        attention_keep_mask: NodeId,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<NodeId> {
        let token_hidden = self.tokens.forward(graph, tokens)?;
        let positions = graph.constant(TensorData::from_scalars(
            [BATCH, TIME],
            DType::I32,
            [0, 1, 2, 0, 1, 2].into_iter().map(Scalar::I),
        )?);
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
        self.project_logits(graph, hidden)
    }

    fn forward_eval(
        &self,
        graph: &mut Graph,
        tokens: NodeId,
        attention_keep_mask: NodeId,
    ) -> Result<NodeId> {
        let token_hidden = self.tokens.forward(graph, tokens)?;
        let positions = graph.constant(TensorData::from_scalars(
            [BATCH, TIME],
            DType::I32,
            [0, 1, 2, 0, 1, 2].into_iter().map(Scalar::I),
        )?);
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

impl Module for FileResumeTransformer {
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
            child("frozen_scale"),
            &self.frozen_scale,
            StateKind::Parameter,
        );
        visitor(
            child("running_marker"),
            &self.running_marker,
            StateKind::Buffer,
        );
        visitor(
            child("lm_head.weight"),
            &self.tokens.weight,
            StateKind::Parameter,
        );
    }
}

fn config() -> Result<CompiledAdamWConfig> {
    optimizer_config()?.with_input_batch::<TransformerBatch>()
}

fn optimizer_config() -> Result<CompiledAdamWConfig> {
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_weight_decay_exclusions(WEIGHT_DECAY_EXCLUSIONS)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(MAX_GRADIENT_NORM)
}

fn reuse_config(schedule: CompiledMultiStepLr) -> Result<CompiledAdamWConfig> {
    Ok(optimizer_config()?
        .with_input_batch::<MaskedTransformerBatch>()?
        .with_token_weighted_gradient_accumulation(LOSS_MASK)?
        .with_frozen_parameters([POLICY_FROZEN])?
        .with_captured_multi_step_lr(schedule))
}

fn two_block_config<B: CompiledInputBatch>(
    schedule: CompiledMultiStepLr,
) -> Result<CompiledAdamWConfig> {
    Ok(CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(MAX_GRADIENT_NORM)?
        .with_input_batch::<B>()?
        .with_token_weighted_gradient_accumulation(LOSS_MASK)?
        .with_frozen_parameters([FILE_RESUME_POLICY_FROZEN])?
        .with_captured_multi_step_lr(schedule)
        .with_clip_report()
        .with_window_loss_report())
}

fn file_resume_config(schedule: CompiledMultiStepLr) -> Result<CompiledAdamWConfig> {
    Ok(CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(MAX_GRADIENT_NORM)?
        .with_input_batch::<FileResumeBatch>()?
        .with_token_weighted_ignore_index(
            MaskedTransformerBatch::TARGETS,
            FILE_RESUME_IGNORE_INDEX,
        )?
        .with_frozen_parameters([FILE_RESUME_POLICY_FROZEN])?
        .with_captured_multi_step_lr(schedule)
        .with_clip_report()
        .with_window_loss_report())
}

fn scoreboard_config(schedule: CompiledMultiStepLr) -> Result<CompiledAdamWConfig> {
    two_block_config::<MaskedTransformerBatch>(schedule)
}

fn dropout_config() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([0x1234_5678, 0x9abc_def0]))
}

fn sparse_causal_losses(graph: &mut Graph, logits: NodeId, targets: NodeId) -> Result<NodeId> {
    let flat_logits = graph.reshape(logits, [TOKEN_COUNT, VOCAB])?;
    let log_probabilities = graph.log_softmax(flat_logits, 1, None)?;
    let target_indices = graph.reshape(targets, [TOKEN_COUNT, 1])?;
    let selected = graph.gather(log_probabilities, target_indices, 1)?;
    let selected = graph.reshape(selected, [TOKEN_COUNT])?;
    let losses = graph.neg(selected)?;
    graph.reshape(losses, [BATCH, TIME])
}

fn sparse_causal_loss(graph: &mut Graph, logits: NodeId, targets: NodeId) -> Result<NodeId> {
    let losses = sparse_causal_losses(graph, logits, targets)?;
    graph.mean_default(losses)
}

fn file_resume_losses(graph: &mut Graph, logits: NodeId, targets: NodeId) -> Result<NodeId> {
    sparse_categorical_cross_entropy(
        graph,
        logits,
        targets,
        LossOptions {
            reduction: Reduction::None,
            class_axis: 2,
            ignore_index: Some(i64::from(FILE_RESUME_IGNORE_INDEX)),
            label_smoothing: 0.0,
        },
    )
}

fn masked_sparse_causal_loss(
    graph: &mut Graph,
    logits: NodeId,
    targets: NodeId,
    loss_mask: NodeId,
) -> Result<NodeId> {
    let losses = sparse_causal_losses(graph, logits, targets)?;
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
    let logits = model.forward(graph, inputs[TransformerBatch::TOKENS], dropout)?;
    let loss = sparse_causal_loss(graph, logits, inputs[TransformerBatch::TARGETS])?;
    Ok((loss, BTreeMap::new()))
}

fn build_buffered(
    model: &BufferedTinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let logits =
        model
            .transformer
            .forward(graph, inputs[MaskedTransformerBatch::TOKENS], dropout)?;
    let losses = sparse_causal_losses(graph, logits, inputs[MaskedTransformerBatch::TARGETS])?;
    Ok((losses, BTreeMap::from([("logits".into(), logits)])))
}

fn build_file_resume(
    model: &FileResumeTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model.forward(
        graph,
        inputs[MaskedTransformerBatch::TOKENS],
        attention_keep_mask,
        dropout,
    )?;
    let losses = file_resume_losses(graph, logits, inputs[MaskedTransformerBatch::TARGETS])?;
    Ok(CompiledAdamWGraph::token_mean(
        losses,
        BTreeMap::from([("logits".into(), logits)]),
    ))
}

fn build_file_resume_evaluation(
    model: &FileResumeTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
) -> Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model.forward_eval(
        graph,
        inputs[MaskedTransformerBatch::TOKENS],
        attention_keep_mask,
    )?;
    let losses = file_resume_losses(graph, logits, inputs[MaskedTransformerBatch::TARGETS])?;
    Ok(CompiledAdamWGraph::token_mean(
        losses,
        BTreeMap::from([("logits".into(), logits)]),
    ))
}

fn build_scoreboard(
    model: &FileResumeTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    // Keep the Bool attention policy capture-owned so every varying replay
    // input uses the native borrowed-input path measured by the scoreboard.
    // The second sample's final key is padding in every accepted batch.
    let attention_keep_mask = graph.constant(TensorData::from_scalars(
        ATTENTION_KEEP_MASK_SHAPE,
        DType::Bool,
        [true, true, true, true, true, false]
            .into_iter()
            .map(Scalar::Bool),
    )?);
    let logits = model.forward(
        graph,
        inputs[MaskedTransformerBatch::TOKENS],
        attention_keep_mask,
        dropout,
    )?;
    let losses = sparse_causal_losses(graph, logits, inputs[MaskedTransformerBatch::TARGETS])?;
    Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
}

fn build_evaluation(
    model: &TinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let logits = model.forward_eval(graph, inputs[TransformerBatch::TOKENS])?;
    let loss = sparse_causal_loss(graph, logits, inputs[TransformerBatch::TARGETS])?;
    Ok((loss, BTreeMap::from([("logits".into(), logits)])))
}

fn compile(model: TinyCausalTransformer) -> Result<CompiledModuleAdamWPlan<TinyCausalTransformer>> {
    CompiledModuleAdamWPlan::compile_with_dropout(config()?, dropout_config(), model, build)
        .map_err(|error| error.into_parts().1)?
        .with_evaluation(build_evaluation)
        .map_err(|error| error.into_parts().1)
}

struct TransformerBatch {
    tokens: TensorData,
    targets: TensorData,
}

struct MaskedTransformerBatch {
    tokens: TensorData,
    targets: TensorData,
    loss_mask: TensorData,
}

struct FileResumeBatch {
    masked: MaskedTransformerBatch,
}

impl FileResumeBatch {
    const SCHEMA: [CompiledInputSpec; 2] = [
        CompiledInputSpec::new(MaskedTransformerBatch::TARGETS, &[BATCH, TIME], DType::I32),
        CompiledInputSpec::host_token(MaskedTransformerBatch::TOKENS, &[BATCH, TIME]),
    ];

    fn from_masked(masked: MaskedTransformerBatch) -> Result<Self> {
        assert_eq!(masked.loss_mask.shape(), &Shape::new([BATCH, TIME]));
        assert_eq!(masked.loss_mask.dtype(), DType::F32);
        let validity = masked.loss_mask.to_vec_f64();
        assert!(
            validity
                .iter()
                .all(|value| value.is_finite() && (*value == 0.0 || *value == 1.0))
        );
        let targets = masked.targets.to_vec_f64();
        let sentinel_values = targets
            .into_iter()
            .zip(&validity)
            .map(|(target, keep)| {
                if *keep == 1.0 {
                    target as i64
                } else {
                    i64::from(FILE_RESUME_IGNORE_INDEX)
                }
            })
            .collect::<Vec<_>>();
        let sentinel_targets = TensorData::from_scalars(
            [BATCH, TIME],
            DType::I32,
            sentinel_values.iter().copied().map(Scalar::I),
        )?;
        let masked = MaskedTransformerBatch {
            targets: sentinel_targets,
            ..masked
        };
        Ok(Self { masked })
    }

    fn has_fully_masked_sample(&self) -> bool {
        self.masked
            .targets
            .to_vec_f64()
            .chunks_exact(TIME)
            .any(|row| {
                row.iter()
                    .all(|value| *value as i32 == FILE_RESUME_IGNORE_INDEX)
            })
    }

    fn loss_mask(&self) -> &TensorData {
        &self.masked.loss_mask
    }
}

impl CompiledInputBatch for FileResumeBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Ok(BTreeMap::from([
            (MaskedTransformerBatch::TARGETS.into(), self.masked.targets),
            (MaskedTransformerBatch::TOKENS.into(), self.masked.tokens),
        ]))
    }
}

impl MaskedTransformerBatch {
    const TOKENS: &'static str = "tokens";
    const TARGETS: &'static str = "targets";
    const SCHEMA: [CompiledInputSpec; 3] = [
        CompiledInputSpec::new(LOSS_MASK, &[BATCH, TIME], DType::F32),
        CompiledInputSpec::host_token(Self::TARGETS, &[BATCH, TIME]),
        CompiledInputSpec::host_token(Self::TOKENS, &[BATCH, TIME]),
    ];

    fn right_padded(replay: u64, valid_lengths: [usize; BATCH]) -> Result<Self> {
        if valid_lengths.iter().any(|valid| *valid > TIME) || valid_lengths.iter().all(|v| *v == 0)
        {
            return Err(rustgrad::Error::SessionTraining {
                reason: "masked causal batch needs bounded right-padded valid lengths".into(),
            });
        }
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
        Ok(Self {
            tokens: token_tensor(tokens)?,
            targets: token_tensor(targets)?,
            loss_mask: TensorData::new([BATCH, TIME], loss_mask.to_vec())?,
        })
    }
}

impl CompiledInputBatch for MaskedTransformerBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Ok(BTreeMap::from([
            (LOSS_MASK.into(), self.loss_mask),
            (Self::TARGETS.into(), self.targets),
            (Self::TOKENS.into(), self.tokens),
        ]))
    }
}

impl TransformerBatch {
    const TOKENS: &'static str = "tokens";
    const TARGETS: &'static str = "targets";
    const SCHEMA: [CompiledInputSpec; 2] = [
        CompiledInputSpec::host_token(Self::TOKENS, &[BATCH, TIME]),
        CompiledInputSpec::host_token(Self::TARGETS, &[BATCH, TIME]),
    ];
}

impl CompiledInputBatch for TransformerBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Ok(BTreeMap::from([
            (Self::TOKENS.into(), self.tokens),
            (Self::TARGETS.into(), self.targets),
        ]))
    }
}

fn batch_values(replay: u64) -> ([i32; TOKEN_COUNT], [i32; TOKEN_COUNT]) {
    match (replay - 1) % ACCUMULATION_STEPS {
        0 => ([0, 1, 2, 2, 0, 1], [1, 2, 0, 0, 1, 2]),
        1 => ([1, 2, 0, 0, 1, 2], [2, 0, 1, 1, 2, 0]),
        2 => ([2, 0, 1, 1, 2, 0], [0, 1, 2, 2, 0, 1]),
        _ => unreachable!(),
    }
}

fn token_tensor(values: [i32; TOKEN_COUNT]) -> Result<TensorData> {
    TensorData::from_scalars(
        Shape::new([BATCH, TIME]),
        DType::I32,
        values.into_iter().map(|value| Scalar::I(i64::from(value))),
    )
}

fn batch(replay: u64) -> Result<TransformerBatch> {
    let (tokens, targets) = batch_values(replay);
    Ok(TransformerBatch {
        tokens: token_tensor(tokens)?,
        targets: token_tensor(targets)?,
    })
}

fn masked_batch(replay: u64) -> Result<MaskedTransformerBatch> {
    const VALID_LENGTHS: [[usize; BATCH]; 3] = [[3, 2], [2, 1], [3, 0]];
    MaskedTransformerBatch::right_padded(
        replay,
        VALID_LENGTHS[((replay - 1) % ACCUMULATION_STEPS) as usize],
    )
}

fn file_resume_batch(replay: u64) -> Result<FileResumeBatch> {
    FileResumeBatch::from_masked(masked_batch(replay)?)
}

fn loss_mask_weight(mask: &TensorData) -> u64 {
    mask.to_vec_f64().into_iter().sum::<f64>() as u64
}

fn evaluate(model: &TinyCausalTransformer) -> Result<TensorData> {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype(TransformerBatch::TOKENS, [BATCH, TIME], DType::I32);
    let logits = model.forward_eval(&mut graph, tokens)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert(TransformerBatch::TOKENS.into(), batch(1)?.tokens);
    CpuBackend.execute(&graph, logits, &bindings)
}

fn evaluate_mean_sparse_loss(model: &TinyCausalTransformer) -> Result<f64> {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype(TransformerBatch::TOKENS, [BATCH, TIME], DType::I32);
    let targets = graph.input_dtype(TransformerBatch::TARGETS, [BATCH, TIME], DType::I32);
    let logits = model.forward_eval(&mut graph, tokens)?;
    let loss = sparse_causal_loss(&mut graph, logits, targets)?;
    let parameter_bindings = model.input_bindings(&graph)?;
    let mut total = 0.0;
    for replay in 1..=ACCUMULATION_STEPS {
        let mut bindings = parameter_bindings.clone();
        bindings.extend(batch(replay)?.into_compiled_inputs()?);
        total += CpuBackend
            .execute(&graph, loss, &bindings)?
            .scalar_at(0)
            .as_f64();
    }
    Ok(total / ACCUMULATION_STEPS as f64)
}

fn evaluate_mean_masked_sparse_loss(model: &TinyCausalTransformer) -> Result<f64> {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype(MaskedTransformerBatch::TOKENS, [BATCH, TIME], DType::I32);
    let targets = graph.input_dtype(MaskedTransformerBatch::TARGETS, [BATCH, TIME], DType::I32);
    let loss_mask = graph.input_dtype(LOSS_MASK, [BATCH, TIME], DType::F32);
    let logits = model.forward_eval(&mut graph, tokens)?;
    let loss = masked_sparse_causal_loss(&mut graph, logits, targets, loss_mask)?;
    let parameter_bindings = model.input_bindings(&graph)?;
    let mut weighted_loss_sum = 0.0;
    let mut loss_weight_sum = 0;
    for replay in 1..=ACCUMULATION_STEPS {
        let batch = masked_batch(replay)?;
        let loss_weight = loss_mask_weight(&batch.loss_mask);
        let mut bindings = parameter_bindings.clone();
        bindings.extend(batch.into_compiled_inputs()?);
        let normalized_loss = CpuBackend
            .execute(&graph, loss, &bindings)?
            .scalar_at(0)
            .as_f64();
        weighted_loss_sum += normalized_loss * loss_weight as f64;
        loss_weight_sum += loss_weight;
    }
    Ok(weighted_loss_sum / loss_weight_sum as f64)
}

fn evaluate_mean_file_resume_loss<R, V>(runtime: &mut R, mut validate: V) -> Result<f64>
where
    R: CompiledEvaluationRuntime,
    R::Evaluation: CompiledEvaluation,
    V: FnMut(&R::Evaluation),
{
    let mut weighted_loss_sum = 0.0;
    let mut loss_weight_sum = 0_u64;
    for (replay, expected_weight) in (1..=ACCUMULATION_STEPS).zip([5, 3, 3]) {
        let batch = file_resume_batch(replay)?;
        assert_eq!(loss_mask_weight(batch.loss_mask()), expected_weight);
        let evaluation = runtime.evaluate_batch(batch)?;
        validate(&evaluation);
        assert_eq!(evaluation.loss_weight(), expected_weight);
        assert_eq!(
            evaluation
                .output("logits")
                .expect("the file-resume evaluator exposes logits")
                .shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        weighted_loss_sum +=
            evaluation.loss().scalar_at(0).as_f64() * evaluation.loss_weight() as f64;
        loss_weight_sum += evaluation.loss_weight();
    }
    assert_eq!(loss_weight_sum, 11);
    Ok(weighted_loss_sum / loss_weight_sum as f64)
}

mod checkpoint;
mod scoreboard;

use checkpoint::{
    CrossProcessBackend, run_cpu_file_resume, run_cpu_reuse, run_cross_process_consumer,
    run_cross_process_producer, run_exact_resume, run_native_cpu_file_resume,
};
use scoreboard::run_native_cpu_scoreboard;

pub(crate) fn run() -> std::result::Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let mode = arguments.next().unwrap_or_else(|| "cpu".to_owned());
    match mode.as_str() {
        "native-cpu-scoreboard" => run_native_cpu_scoreboard()?,
        "cpu-reuse" => run_cpu_reuse()?,
        "cpu-file-resume" => run_cpu_file_resume()?,
        "native-cpu-file-resume" => run_native_cpu_file_resume()?,
        "cross-process-produce" | "cross-process-consume" => {
            let backend = arguments
                .next()
                .ok_or_else(|| format!("{mode} requires a `cpu` or `native-cpu` backend"))?;
            let backend = CrossProcessBackend::parse(&backend)?;
            let directory = arguments
                .next()
                .ok_or_else(|| format!("{mode} requires an output directory"))?;
            if let Some(unexpected) = arguments.next() {
                return Err(format!("unexpected {mode} argument {unexpected:?}").into());
            }
            if mode == "cross-process-produce" {
                run_cross_process_producer(backend, Path::new(&directory))?;
            } else {
                run_cross_process_consumer(backend, Path::new(&directory))?;
            }
        }
        "cpu" => {
            let target = CpuSessionTarget::new();
            run_exact_resume("CPU", |plan| {
                plan.prepare(&target).map_err(|error| error.into_parts().1)
            })?;
        }
        "native-cpu" => {
            let executor = CapturedReplayExecutor::default();
            let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
            run_exact_resume("native CPU", |plan| {
                plan.prepare(&target).map_err(|error| error.into_parts().1)
            })?;
        }
        "metal" => {
            let device = MetalRuntime::load()?.device(0)?;
            let target = MetalSessionTarget::new(device, 64)?;
            run_exact_resume("Metal", |plan| {
                assert_eq!(
                    plan.metal_summary(target.renderer().clone())?
                        .fallback_count,
                    0
                );
                plan.prepare(&target).map_err(|error| error.into_parts().1)
            })?;
        }
        other => {
            return Err(format!(
                "unknown target {other:?}; expected `native-cpu-scoreboard`, `cpu-reuse`, `cpu-file-resume`, `native-cpu-file-resume`, `cross-process-produce`, `cross-process-consume`, `cpu`, `native-cpu`, or `metal`"
            )
            .into());
        }
    }
    Ok(())
}
