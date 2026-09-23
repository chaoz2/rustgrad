//! Compile, train, replay, checkpoint, and resume one owned fixed-shape tiny
//! Transformer on CPU or explicitly selected strict Metal.
//!
//! ## Contract
//!
//! - Right-padded batches keep one static shape.
//! - Ignore-index targets provide both attention validity and exact token-mean
//!   gradient weights across each accumulation window.
//! - Resume authenticates the compiled program, optimizer frontier, tied and
//!   frozen module state, and replay progress before publication.
//!
//! ## Resume modes
//!
//! | Mode | Command | Boundary |
//! |---|---|---|
//! | Reuse one plan | `cargo run --example compiled_transformer_train_resume -- cpu-reuse` | Restores in process without rebuilding the graph or capture. |
//! | Portable file | `cargo run --example compiled_transformer_train_resume -- cpu-file-resume` | Restores a deliberately different initialization from a resource-free program artifact and complete module checkpoint. |
//! | Strict-native file | `cargo run --release --example compiled_transformer_train_resume -- native-cpu-file-resume` | Runs the same file lifecycle through CPU JIT with no fallback. |
//! | Cross-process file | `cross-process-produce <cpu|native-cpu> <directory>`, then `cross-process-consume <cpu|native-cpu> <directory>` | Produces a pending RGAB and authenticates its continuation in a fresh OS process. |
//!
//! ## Replay and evidence modes
//!
//! | Mode | Command | Boundary |
//! |---|---|---|
//! | Interpreter CPU | `cargo run --example compiled_transformer_train_resume -- cpu` | Replays graph-free on the host interpreter. |
//! | Strict-native CPU | `cargo run --release --example compiled_transformer_train_resume -- native-cpu` | Replays the same capture through CPU JIT. |
//! | CPU scoreboard | `cargo run --release --example compiled_transformer_train_resume -- native-cpu-scoreboard` | Emits the bounded strict-native evidence report. |
//! | Strict Metal | `cargo run --release --example compiled_transformer_train_resume -- metal` | Uses the first visible Metal device with no CPU fallback. |

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

struct TemporaryCheckpointFile {
    path: PathBuf,
}

impl TemporaryCheckpointFile {
    fn new() -> std::result::Result<Self, Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "rustgrad-compiled-module-resume-{}-{nonce}.rgab",
            std::process::id()
        ));
        Ok(Self { path })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TemporaryCheckpointFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

const CROSS_PROCESS_EVIDENCE_VERSION: u64 = 1;
const CROSS_PROCESS_EVIDENCE_MAX_BYTES: u64 = 64 << 10;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CrossProcessBackend {
    Cpu,
    NativeCpu,
}

impl CrossProcessBackend {
    fn parse(value: &str) -> std::result::Result<Self, Box<dyn Error>> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "native-cpu" => Ok(Self::NativeCpu),
            other => Err(format!(
                "unknown cross-process backend {other:?}; expected `cpu` or `native-cpu`"
            )
            .into()),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::NativeCpu => "native CPU",
        }
    }
}

struct CrossProcessResumePaths {
    pending_bundle: PathBuf,
    terminal_checkpoint: PathBuf,
    evidence: PathBuf,
}

impl CrossProcessResumePaths {
    fn new(directory: &Path) -> std::result::Result<Self, Box<dyn Error>> {
        let metadata = fs::symlink_metadata(directory)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err("cross-process resume directory must be a real directory".into());
        }
        Ok(Self {
            pending_bundle: directory.join("pending.rgab"),
            terminal_checkpoint: directory.join("expected-terminal.safetensors"),
            evidence: directory.join("expected.json"),
        })
    }

    fn require_absent(&self) -> std::result::Result<(), Box<dyn Error>> {
        for path in [
            &self.pending_bundle,
            &self.terminal_checkpoint,
            &self.evidence,
        ] {
            match fs::symlink_metadata(path) {
                Ok(_) => {
                    return Err(format!(
                        "cross-process resume producer refuses existing output {}",
                        path.display()
                    )
                    .into());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CrossProcessCheckpointEvidence {
    capture_identity: u64,
    replay_step: u64,
    optimizer_step: u64,
    gradient_accumulation_steps: u64,
    accumulation_index: u64,
    discarded_microbatches: u64,
    flushed_window_count: u64,
    flushed_microbatch_count: u64,
    flush_capture_identity: Option<u64>,
    dropout_block_counter: Option<u64>,
    accumulated_token_count: Option<u64>,
    accumulated_loss_numerator_bits: Option<u32>,
    window_loss_report_enabled: bool,
    reset_transition_count: u64,
    reset_capture_identity: Option<u64>,
    accumulation_capture_identity: Option<u64>,
}

impl CrossProcessCheckpointEvidence {
    fn new(checkpoint: &CompiledAdamWCheckpoint) -> Self {
        let info = checkpoint.info();
        Self {
            capture_identity: info.capture_identity(),
            replay_step: info.replay_step(),
            optimizer_step: info.optimizer_step(),
            gradient_accumulation_steps: info.gradient_accumulation_steps(),
            accumulation_index: info.accumulation_index(),
            discarded_microbatches: info.discarded_microbatches(),
            flushed_window_count: info.flushed_window_count(),
            flushed_microbatch_count: info.flushed_microbatch_count(),
            flush_capture_identity: info.flush_capture_identity(),
            dropout_block_counter: info.dropout_block_counter(),
            accumulated_token_count: info.accumulated_token_count(),
            accumulated_loss_numerator_bits: info.accumulated_loss_numerator().map(f32::to_bits),
            window_loss_report_enabled: info.window_loss_report_enabled(),
            reset_transition_count: info.reset_transition_count(),
            reset_capture_identity: info.reset_capture_identity(),
            accumulation_capture_identity: info.accumulation_capture_identity(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CrossProcessResumeEvidence {
    format_version: u64,
    backend: CrossProcessBackend,
    pending_bundle_bytes: u64,
    pending_bundle_checksum: u64,
    terminal_checkpoint_bytes: u64,
    terminal_checkpoint_checksum: u64,
    evaluation_capture_identity: u64,
    initial_evaluation_loss_bits: u64,
    terminal_evaluation_loss_bits: u64,
    pending: CrossProcessCheckpointEvidence,
    terminal: CrossProcessCheckpointEvidence,
}

impl CrossProcessResumeEvidence {
    fn canonical_bytes(&self) -> std::result::Result<Vec<u8>, Box<dyn Error>> {
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn save(&self, path: &Path) -> std::result::Result<(), Box<dyn Error>> {
        let bytes = self.canonical_bytes()?;
        if u64::try_from(bytes.len())? > CROSS_PROCESS_EVIDENCE_MAX_BYTES {
            return Err("cross-process resume evidence exceeds its byte bound".into());
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    fn load(path: &Path) -> std::result::Result<Self, Box<dyn Error>> {
        let length = fs::metadata(path)?.len();
        if length > CROSS_PROCESS_EVIDENCE_MAX_BYTES {
            return Err("cross-process resume evidence exceeds its byte bound".into());
        }
        let bytes = fs::read(path)?;
        Self::from_canonical_bytes(&bytes)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> std::result::Result<Self, Box<dyn Error>> {
        if u64::try_from(bytes.len())? > CROSS_PROCESS_EVIDENCE_MAX_BYTES {
            return Err("cross-process resume evidence exceeds its byte bound".into());
        }
        let evidence: Self = serde_json::from_slice(bytes)?;
        if evidence.format_version != CROSS_PROCESS_EVIDENCE_VERSION {
            return Err(format!(
                "unsupported cross-process resume evidence version {}",
                evidence.format_version
            )
            .into());
        }
        if evidence.canonical_bytes()? != bytes {
            return Err("cross-process resume evidence is not canonical".into());
        }
        Ok(evidence)
    }

    fn validate_backend(
        &self,
        expected: CrossProcessBackend,
    ) -> std::result::Result<(), Box<dyn Error>> {
        if self.backend != expected {
            return Err("cross-process resume evidence backend differs".into());
        }
        Ok(())
    }
}

fn cross_process_checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn validate_cross_process_file_binding(
    label: &str,
    bytes: &[u8],
    expected_bytes: u64,
    expected_checksum: u64,
) -> std::result::Result<(), Box<dyn Error>> {
    let actual_bytes = u64::try_from(bytes.len())?;
    if actual_bytes != expected_bytes || cross_process_checksum(bytes) != expected_checksum {
        return Err(format!("cross-process {label} differs from its evidence binding").into());
    }
    Ok(())
}

fn replay_prepared_transformer<R>(runtime: &mut R, replay: u64) -> Result<R::Step>
where
    R: CompiledTrainingWindowRuntime,
{
    runtime.step_batch(batch(replay)?, 0.05)
}

fn replay_prepared_policy_batch<R, B>(runtime: &mut R, batch: B) -> Result<R::Step>
where
    R: CompiledTrainingRatePolicyRuntime,
    B: CompiledInputBatch,
{
    runtime.step_batch_with_rate_policy(batch)
}

fn assert_file_resume_steps_match<S>(actual: &S, expected: &S)
where
    S: CompiledAdamWStep,
{
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());
    assert_eq!(
        actual
            .output("logits")
            .expect("the file-resume capture exposes logits")
            .shape(),
        &Shape::new([BATCH, TIME, VOCAB])
    );
    assert_eq!(actual.step(), expected.step());
    assert_eq!(actual.optimizer_step(), expected.optimizer_step());
    assert_eq!(actual.accumulation_index(), expected.accumulation_index());
    assert_eq!(actual.loss_weight(), expected.loss_weight());
    assert_eq!(actual.did_update(), expected.did_update());
    assert_eq!(actual.clip_report(), expected.clip_report());
    assert_eq!(actual.window_loss_report(), expected.window_loss_report());
}

fn run_exact_resume<R, P>(target_name: &str, mut prepare: P) -> Result<()>
where
    R: CompiledAdamWRuntime + CompiledTrainingWindowCommitRuntime + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<TinyCausalTransformer>,
    ) -> Result<CompiledModuleAdamWSession<TinyCausalTransformer, R>>,
{
    let model = TinyCausalTransformer::new(7)?;
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&model)?;
    let plan = compile(model)?;
    let capture_identity = plan.capture_identity();
    let flush_capture_identity = plan
        .flush_capture_identity()
        .expect("three-step accumulation exposes a partial-flush capture");
    let mut uninterrupted = prepare(plan)?.into_training_session();
    let evaluation_identity = uninterrupted
        .evaluation_capture_identity()
        .expect("evaluation was attached before preparation");
    let parameters = uninterrupted.parameter_snapshots()?;
    assert!(parameters.contains_key("tokens.weight"));
    assert!(
        !parameters.contains_key("lm_head.weight"),
        "the tied output head must share the embedding's recurrent state"
    );
    assert_eq!(uninterrupted.gradient_window_size(), ACCUMULATION_STEPS);
    assert_eq!(
        uninterrupted.runtime().max_gradient_norm(),
        Some(MAX_GRADIENT_NORM)
    );
    let initial_parameters = uninterrupted.parameter_snapshots()?;
    let initial_first_moments = uninterrupted.runtime().first_moment_snapshots()?;
    let initial_second_moments = uninterrupted.runtime().second_moment_snapshots()?;
    let empty_accumulators = uninterrupted.runtime().gradient_accumulator_snapshots()?;

    for replay in 1..=2 {
        let step = replay_prepared_transformer(&mut uninterrupted, replay)?;
        assert_eq!(step.optimizer_step(), 0);
        assert_eq!(step.accumulation_index(), replay);
        assert!(!step.did_update());
        assert_eq!(step.pending_microbatch_count(), replay);
        assert!(!step.did_close_gradient_window());
    }
    assert_ne!(
        uninterrupted.runtime().gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    assert_eq!(uninterrupted.parameter_snapshots()?, initial_parameters);
    assert_eq!(
        uninterrupted.runtime().first_moment_snapshots()?,
        initial_first_moments
    );
    assert_eq!(
        uninterrupted.runtime().second_moment_snapshots()?,
        initial_second_moments
    );
    assert_eq!(
        uninterrupted
            .reset_gradient_window()?
            .discarded_microbatches(),
        2
    );
    assert_eq!(uninterrupted.step_count(), 2);
    assert_eq!(uninterrupted.runtime().optimizer_step()?, 0);
    assert_eq!(uninterrupted.runtime().accumulation_index()?, 0);
    assert_eq!(
        uninterrupted.runtime().gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    let after_reset = uninterrupted.checkpoint()?;
    assert_eq!(
        uninterrupted
            .reset_gradient_window()?
            .discarded_microbatches(),
        0
    );
    assert_eq!(uninterrupted.checkpoint()?, after_reset);

    for replay in 3..=INITIAL_STEPS as u64 {
        let step = replay_prepared_transformer(&mut uninterrupted, replay)?;
        assert_eq!(step.optimizer_step(), 0);
        assert_eq!(step.accumulation_index(), replay - 2);
        assert!(!step.did_update());
    }

    let saved = uninterrupted.checkpoint()?;
    let checkpoint = CompiledAdamWCheckpoint::from_bytes(saved.into_bytes())?;
    let checkpoint_info = checkpoint.info();
    assert_eq!(checkpoint_info.capture_identity(), capture_identity);
    assert_eq!(checkpoint_info.replay_step(), INITIAL_STEPS as u64);
    assert_eq!(checkpoint_info.optimizer_step(), 0);
    assert_eq!(
        checkpoint_info.gradient_accumulation_steps(),
        ACCUMULATION_STEPS
    );
    assert_eq!(checkpoint_info.accumulation_index(), 2);
    assert_eq!(checkpoint_info.discarded_microbatches(), 2);
    assert_eq!(checkpoint_info.flushed_window_count(), 0);
    assert_eq!(checkpoint_info.flushed_microbatch_count(), 0);
    assert_eq!(
        checkpoint_info.flush_capture_identity(),
        Some(flush_capture_identity)
    );
    assert_eq!(
        uninterrupted.partial_window_commit_capture_identity(),
        Some(flush_capture_identity)
    );
    assert_eq!(checkpoint_info.dropout_block_counter(), Some(48));
    let resumed_first_replay = checkpoint_info.replay_step() + 1;
    let resumed_last_replay = checkpoint_info.replay_step() + RESUMED_STEPS as u64;
    let restored_model = TinyCausalTransformer::new(7)?;
    let tied = restored_model.tokens.weight.clone();
    let frozen = restored_model.frozen_scale.clone();
    let restored_plan = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config()?,
        dropout_config(),
        restored_model,
        &checkpoint,
        build,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation(build_evaluation)
    .map_err(|error| error.into_parts().1)?;
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(restored_plan.step_count(), INITIAL_STEPS as u64);
    let mut resumed = prepare(restored_plan)?.into_training_session();
    assert_eq!(resumed.runtime().optimizer_step()?, 0);
    assert_eq!(resumed.runtime().accumulation_index()?, 2);
    assert_eq!(resumed.checkpoint()?, checkpoint);

    for replay in resumed_first_replay..=resumed_last_replay {
        let expected = replay_prepared_transformer(&mut uninterrupted, replay)?;
        let actual = replay_prepared_transformer(&mut resumed, replay)?;
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.did_update(), expected.did_update());
        assert_eq!(
            actual.pending_microbatch_count(),
            expected.pending_microbatch_count()
        );
        assert_eq!(
            actual.did_close_gradient_window(),
            expected.did_close_gradient_window()
        );
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
    let expected_flush = uninterrupted.commit_partial_window(TensorData::scalar(0.05))?;
    let actual_flush = resumed.commit_partial_window(TensorData::scalar(0.05))?;
    assert_eq!(actual_flush.committed_microbatches(), 2);
    assert_eq!(
        actual_flush.committed_microbatches(),
        expected_flush.committed_microbatches()
    );
    assert_eq!(
        actual_flush.did_commit_window(),
        expected_flush.did_commit_window()
    );
    assert!(actual_flush.did_commit_window());
    let after_flush = resumed.checkpoint()?;
    let empty_flush = resumed.commit_partial_window(TensorData::scalar(0.05))?;
    assert_eq!(empty_flush.committed_microbatches(), 0);
    assert!(!empty_flush.did_commit_window());
    assert_eq!(resumed.checkpoint()?, after_flush);
    assert_eq!(resumed.runtime().optimizer_step()?, 2);
    assert_eq!(resumed.runtime().accumulation_index()?, 0);

    assert_eq!(
        resumed.parameter_snapshots()?,
        uninterrupted.parameter_snapshots()?
    );
    assert_eq!(
        resumed.runtime().first_moment_snapshots()?,
        uninterrupted.runtime().first_moment_snapshots()?
    );
    assert_eq!(
        resumed.runtime().second_moment_snapshots()?,
        uninterrupted.runtime().second_moment_snapshots()?
    );
    assert_eq!(
        resumed.runtime().gradient_accumulator_snapshots()?,
        uninterrupted.runtime().gradient_accumulator_snapshots()?
    );
    assert_eq!(
        resumed.runtime().gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let before_evaluation_checkpoint = resumed.checkpoint()?;
    let mut final_mean_sparse_loss = 0.0;
    for replay in 1..=ACCUMULATION_STEPS {
        let evaluated = resumed.evaluate_batch(batch(replay)?)?;
        assert_eq!(evaluated.capture_identity(), evaluation_identity);
        assert_eq!(
            evaluated.output("logits").unwrap().shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        final_mean_sparse_loss += evaluated.loss().scalar_at(0).as_f64();
    }
    final_mean_sparse_loss /= ACCUMULATION_STEPS as f64;
    assert_eq!(resumed.checkpoint()?, before_evaluation_checkpoint);
    let tied_version = tied.version()?;
    let frozen_before = frozen.snapshot()?;
    let runtime_step = resumed.step_count();
    let (uninterrupted_model, uninterrupted_checkpoint) = uninterrupted
        .finish_with_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    let (restored_model, finished_checkpoint) = resumed
        .finish_with_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(finished_checkpoint, before_evaluation_checkpoint);
    assert_eq!(finished_checkpoint, uninterrupted_checkpoint);
    let live = restored_model.state_dict()?;
    assert_eq!(live, uninterrupted_model.state_dict()?);
    assert_eq!(
        restored_model.tokens.weight.value()?,
        live.tensors()["tokens.weight"]
    );
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert_eq!(restored_model.tokens.weight.version()?, tied_version + 1);
    let frozen_after = restored_model.frozen_scale.snapshot()?;
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    let versions_before_eval = restored_model
        .trainable_parameters()?
        .into_iter()
        .map(|(name, parameter)| Ok((name, parameter.version()?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let first_eval = evaluate(&restored_model)?;
    let second_eval = evaluate(&restored_model)?;
    let published_mean_sparse_loss = evaluate_mean_sparse_loss(&restored_model)?;
    assert_eq!(first_eval.shape(), &Shape::new([BATCH, TIME, VOCAB]));
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel()?)
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in restored_model.trainable_parameters()? {
        assert_eq!(parameter.version()?, versions_before_eval[&name]);
    }
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "compiled causal Transformer eval loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
    assert!((final_mean_sparse_loss - published_mean_sparse_loss).abs() < 1e-5);
    println!(
        "{target_name}: capture={capture_identity:016x}, steps={}, eval_mean_sparse_loss={:.6} -> {:.6}, exact_resume=true, published=true",
        runtime_step, initial_mean_sparse_loss, final_mean_sparse_loss
    );
    Ok(())
}

fn run_cpu_reuse() -> Result<()> {
    let source = BufferedTinyCausalTransformer::new(7)?;
    let initial_mean_sparse_loss = evaluate_mean_masked_sparse_loss(&source.transformer)?;
    assert_eq!(
        masked_batch(3)?.loss_mask.to_vec_f64(),
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "the third replay is a right-padded final partial row"
    );
    let source_policy_frozen = source
        .trainable_parameters()?
        .into_iter()
        .find_map(|(name, parameter)| (name == POLICY_FROZEN).then_some(parameter))
        .expect("the maintained Transformer exposes the policy-frozen parameter")
        .value()?;
    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1])?;
    let builds = Cell::new(0);
    let plan = CompiledAdamWPlan::compile_module_graph_with_dropout(
        reuse_config(schedule.clone())?,
        dropout_config(),
        &source,
        |model, graph, inputs, dropout| {
            builds.set(builds.get() + 1);
            let (losses, outputs) = build_buffered(model, graph, inputs, dropout)?;
            Ok(CompiledAdamWGraph::token_mean(losses, outputs))
        },
    )?;
    assert_eq!(builds.get(), 1, "the training graph must compile once");
    let capture_identity = plan.capture_identity();
    assert_eq!(
        plan.token_weighted_gradient_accumulation_mask(),
        Some(LOSS_MASK)
    );
    let target =
        CpuSessionTarget::new().with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut uninterrupted = plan.prepare(&target)?;
    assert_eq!(
        uninterrupted.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );
    assert_eq!(uninterrupted.captured_multi_step_lr(), Some(&schedule));
    assert!(
        !uninterrupted
            .parameter_snapshots()?
            .contains_key(POLICY_FROZEN)
    );

    for replay in 1..=4 {
        let batch = masked_batch(replay)?;
        let loss_weight = loss_mask_weight(&batch.loss_mask);
        let step = uninterrupted.step_batch_scheduled(batch)?;
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
    }
    assert_eq!(uninterrupted.optimizer_step()?, 1);
    assert_eq!(uninterrupted.accumulation_index()?, 1);
    let checkpoint = uninterrupted.checkpoint()?;
    assert_eq!(checkpoint.info().replay_step(), 4);
    assert_eq!(checkpoint.info().optimizer_step(), 1);
    assert_eq!(checkpoint.info().accumulation_index(), 1);
    assert_eq!(checkpoint.info().accumulated_token_count(), Some(5));
    assert_eq!(uninterrupted.dropout_block_counter()?, Some(48));

    let restored_plan = plan.restore_checkpoint(&checkpoint)?;
    assert_eq!(
        builds.get(),
        1,
        "checkpoint restore must not rebuild the graph"
    );
    assert_eq!(
        plan.step_count(),
        0,
        "the borrowed source plan stays reusable"
    );
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(restored_plan.captured_multi_step_lr(), Some(&schedule));
    assert_eq!(restored_plan.step_count(), 4);
    let mut resumed = restored_plan.prepare(&target)?;
    assert_eq!(resumed.captured_multi_step_lr(), Some(&schedule));
    assert_eq!(
        resumed.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );
    assert_eq!(resumed.checkpoint()?, checkpoint);

    let before_wrong_entrypoint = resumed.checkpoint()?;
    assert!(resumed.step_batch(masked_batch(5)?, 0.05).is_err());
    assert_eq!(resumed.checkpoint()?, before_wrong_entrypoint);
    for replay in 5..=6 {
        let loss_weight = loss_mask_weight(&masked_batch(replay)?.loss_mask);
        let expected = uninterrupted.step_batch_scheduled(masked_batch(replay)?)?;
        let actual = resumed.step_batch_scheduled(masked_batch(replay)?)?;
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.loss_weight(), expected.loss_weight());
        assert_eq!(actual.loss_weight(), loss_weight);
        assert_eq!(
            actual
                .output("logits")
                .expect("the reuse capture exposes logits")
                .shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.did_update(), expected.did_update());
        assert_eq!(
            resumed.parameter_snapshots()?,
            uninterrupted.parameter_snapshots()?
        );
        assert_eq!(
            resumed.first_moment_snapshots()?,
            uninterrupted.first_moment_snapshots()?
        );
        assert_eq!(
            resumed.second_moment_snapshots()?,
            uninterrupted.second_moment_snapshots()?
        );
        assert_eq!(
            resumed.gradient_accumulator_snapshots()?,
            uninterrupted.gradient_accumulator_snapshots()?
        );
        assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    }
    assert_eq!(resumed.optimizer_step()?, 2);
    assert_eq!(resumed.accumulation_index()?, 0);
    assert_eq!(resumed.dropout_block_counter()?, Some(72));

    let final_parameters = resumed.parameter_snapshots()?;
    let destination = BufferedTinyCausalTransformer::new(0xdecafbad)?;
    let tied_before = destination.transformer.tokens.weight.snapshot()?;
    let tied_identity = tied_before.identity;
    assert_ne!(
        destination.transformer.tokens.weight.value()?,
        final_parameters["tokens.weight"]
    );
    let policy_frozen = destination
        .trainable_parameters()?
        .into_iter()
        .find_map(|(name, parameter)| (name == POLICY_FROZEN).then_some(parameter))
        .expect("the destination exposes the policy-frozen parameter");
    assert_ne!(policy_frozen.value()?, source_policy_frozen);
    policy_frozen.replace(source_policy_frozen)?;
    let policy_frozen_before = policy_frozen.snapshot()?;
    let inherent_frozen_before = destination.transformer.frozen_scale.snapshot()?;
    destination
        .running_marker
        .replace(TensorData::scalar(29.0))?;
    let running_marker_before = destination.running_marker.snapshot()?;
    let effective_trainables_before = destination
        .trainable_parameters()?
        .into_iter()
        .filter(|(name, _)| name != POLICY_FROZEN)
        .map(|(name, parameter)| Ok((name, parameter.snapshot()?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let final_checkpoint = resumed.checkpoint()?;
    assert!(resumed.publish_parameters(&destination)?.is_clean());
    assert_eq!(resumed.checkpoint()?, final_checkpoint);

    let mut tied_alias = None;
    destination.visit("", &mut |name, parameter, _| {
        if name == "lm_head.weight" {
            tied_alias = Some(parameter.clone());
        }
    });
    let tied_alias = tied_alias.expect("the destination exposes the tied output alias");
    assert_eq!(tied_alias.id(), tied_identity);
    assert_eq!(destination.transformer.tokens.weight.id(), tied_identity);
    assert!(
        !destination
            .state_dict()?
            .tensors()
            .contains_key("lm_head.weight")
    );
    for (name, parameter) in destination.trainable_parameters()? {
        if name != POLICY_FROZEN {
            let before = &effective_trainables_before[&name];
            let after = parameter.snapshot()?;
            assert_eq!(after.data, final_parameters[&name]);
            assert_eq!(after.identity, before.identity);
            assert_eq!(after.trainable, before.trainable);
            assert_eq!(
                after.version,
                before
                    .version
                    .checked_add(1)
                    .expect("successful publication cannot overflow a version")
            );
        }
    }
    let tied_after = destination.transformer.tokens.weight.snapshot()?;
    assert_eq!(tied_after.data, final_parameters["tokens.weight"]);
    assert_eq!(tied_after.identity, tied_before.identity);
    assert_eq!(tied_after.trainable, tied_before.trainable);
    assert_eq!(
        tied_after.version,
        tied_before
            .version
            .checked_add(1)
            .expect("successful publication cannot overflow the tied weight version")
    );
    let tied_alias_after = tied_alias.snapshot()?;
    assert_eq!(tied_alias_after.data, tied_after.data);
    assert_eq!(tied_alias_after.identity, tied_after.identity);
    assert_eq!(tied_alias_after.version, tied_after.version);
    assert_eq!(tied_alias_after.trainable, tied_after.trainable);
    let policy_frozen_after = policy_frozen.snapshot()?;
    assert_eq!(policy_frozen_after.data, policy_frozen_before.data);
    assert_eq!(policy_frozen_after.version, policy_frozen_before.version);
    assert_eq!(policy_frozen_after.identity, policy_frozen_before.identity);
    assert_eq!(
        policy_frozen_after.trainable,
        policy_frozen_before.trainable
    );
    let inherent_frozen_after = destination.transformer.frozen_scale.snapshot()?;
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
    let running_marker_after = destination.running_marker.snapshot()?;
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

    let final_mean_sparse_loss = evaluate_mean_masked_sparse_loss(&destination.transformer)?;
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "compile-once causal Transformer loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
    println!(
        "CPU reuse: capture={capture_identity:016x}, builds={}, checkpoint=(replay=4, optimizer=1, accumulation=1), optimizer_steps={}, eval_mean_sparse_loss={:.6} -> {:.6}, exact_resume=true, published=true",
        builds.get(),
        resumed.optimizer_step()?,
        initial_mean_sparse_loss,
        final_mean_sparse_loss
    );
    Ok(())
}

fn run_cpu_file_resume() -> std::result::Result<(), Box<dyn Error>> {
    let target =
        CpuSessionTarget::new().with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    run_file_resume::<CpuCompiledAdamW, _, _, _, _>(
        "CPU file resume",
        |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
        |_| {},
        |_| {},
        |_| {},
    )
}

fn assert_cross_process_pending_checkpoint(checkpoint: &CompiledAdamWCheckpoint) {
    let info = checkpoint.info();
    assert_eq!(info.replay_step(), 4);
    assert_eq!(info.optimizer_step(), 1);
    assert_eq!(info.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(info.accumulation_index(), 1);
    assert_eq!(info.accumulated_token_count(), Some(5));
    assert!(
        info.dropout_block_counter()
            .is_some_and(|counter| counter > 0)
    );
    assert!(
        info.accumulated_loss_numerator()
            .is_some_and(|value| value.is_finite() && value != 0.0)
    );
    assert!(info.window_loss_report_enabled());
}

fn assert_cross_process_terminal_checkpoint(checkpoint: &CompiledAdamWCheckpoint) {
    let info = checkpoint.info();
    assert_eq!(info.replay_step(), 11);
    assert_eq!(info.optimizer_step(), 4);
    assert_eq!(info.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(info.accumulation_index(), 0);
    assert_eq!(info.discarded_microbatches(), 1);
    assert_eq!(info.flushed_window_count(), 1);
    assert_eq!(info.flushed_microbatch_count(), 1);
    assert_eq!(info.accumulated_token_count(), Some(0));
    assert_eq!(info.accumulated_loss_numerator(), Some(0.0));
    assert_eq!(info.reset_transition_count(), 1);
    assert!(info.window_loss_report_enabled());
}

fn assert_cross_process_zero_accumulators<R>(
    session: &CompiledModuleAdamWSession<FileResumeTransformer, R>,
) -> Result<()>
where
    R: CompiledScheduledAdamWRuntime,
{
    let accumulators = session.gradient_accumulator_snapshots()?;
    assert!(!accumulators.is_empty());
    for (name, accumulator) in accumulators {
        for value in accumulator.values() {
            assert_eq!(
                value.to_bits(),
                0,
                "terminal accumulator {name} must contain positive zero"
            );
        }
    }
    Ok(())
}

fn run_cross_process_file_resume_suffix<R, S, E>(
    session: &mut CompiledModuleAdamWSession<FileResumeTransformer, R>,
    mut validate_step: S,
    mut validate_evaluation: E,
) -> std::result::Result<f64, Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    const WINDOW_LOSS_WEIGHT: u64 = 11;
    for replay in 5..=9 {
        let batch = file_resume_batch(replay)?;
        let loss_weight = loss_mask_weight(batch.loss_mask());
        let step = replay_prepared_policy_batch(session, batch)?;
        validate_step(&step);
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), matches!(replay, 6 | 9));
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        if let Some(report) = step.window_loss_report() {
            assert!(report.is_finite());
            assert_eq!(report.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(report.loss_weight(), WINDOW_LOSS_WEIGHT);
        }
    }

    let partial = replay_prepared_policy_batch(session, file_resume_batch(10)?)?;
    validate_step(&partial);
    assert!(!partial.did_update());
    assert_eq!(partial.accumulation_index(), 1);
    let flush = session.commit_partial_window_with_rate_policy()?;
    assert!(flush.did_commit_window());
    assert_eq!(flush.committed_microbatches(), 1);

    let discard = replay_prepared_policy_batch(session, file_resume_batch(11)?)?;
    validate_step(&discard);
    assert!(!discard.did_update());
    assert_eq!(discard.accumulation_index(), 1);
    let reset = session.reset_gradient_window()?;
    assert_eq!(reset.discarded_microbatches(), 1);
    assert_eq!(session.optimizer_step()?, 4);
    assert_eq!(session.accumulation_index()?, 0);
    assert_cross_process_zero_accumulators(session)?;

    let before_evaluation = session.checkpoint()?;
    let before_counter = before_evaluation.info().dropout_block_counter();
    let before_accumulators = session.gradient_accumulator_snapshots()?;
    let terminal_loss = evaluate_mean_file_resume_loss(session, &mut validate_evaluation)?;
    assert_eq!(session.checkpoint()?, before_evaluation);
    assert_eq!(
        session.checkpoint()?.info().dropout_block_counter(),
        before_counter
    );
    assert_eq!(
        session.gradient_accumulator_snapshots()?,
        before_accumulators
    );
    Ok(terminal_loss)
}

fn produce_cross_process_file_resume<R, P, V, S, E>(
    backend: CrossProcessBackend,
    paths: &CrossProcessResumePaths,
    mut prepare: P,
    mut validate_preparation: V,
    mut validate_step: S,
    mut validate_evaluation: E,
) -> std::result::Result<(), Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<FileResumeTransformer>,
    ) -> Result<CompiledModuleAdamWSession<FileResumeTransformer, R>>,
    V: FnMut(&CompiledModuleAdamWSession<FileResumeTransformer, R>),
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    paths.require_absent()?;
    let schedule = CompiledMultiStepLr::new(1e-3, 0.5, [1])?;
    let config = file_resume_config(schedule)?;
    let source = FileResumeTransformer::new(0x5678)?;
    let builds = Cell::new(0_u64);
    let source_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
        config,
        dropout_config(),
        source,
        |model, graph, inputs, ignore_index, dropout| {
            builds.set(builds.get() + 1);
            build_file_resume(model, graph, inputs, ignore_index, dropout)
        },
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph_and_ignore_index(build_file_resume_evaluation)
    .map_err(|error| error.into_parts().1)?;
    assert_eq!(builds.get(), 1, "the producer training graph compiles once");
    let evaluation_capture_identity = source_plan
        .evaluation_capture_identity()
        .expect("the cross-process evaluator is attached");
    let program_artifact = source_plan.program_artifact()?;
    let mut source = prepare(source_plan)?;
    validate_preparation(&source);

    let before_initial_evaluation = source.checkpoint()?;
    let initial_loss = evaluate_mean_file_resume_loss(&mut source, &mut validate_evaluation)?;
    assert_eq!(source.checkpoint()?, before_initial_evaluation);
    for replay in 1..=4 {
        let batch = file_resume_batch(replay)?;
        let loss_weight = loss_mask_weight(batch.loss_mask());
        let step = replay_prepared_policy_batch(&mut source, batch)?;
        validate_step(&step);
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
    }

    let pending_checkpoint = source.module_checkpoint()?;
    assert_cross_process_pending_checkpoint(pending_checkpoint.optimizer_checkpoint());
    let pending_bundle =
        CompiledAdamWResumeBundle::new(program_artifact, pending_checkpoint.clone())?;
    pending_bundle.save_file(&paths.pending_bundle)?;

    let terminal_loss = run_cross_process_file_resume_suffix(
        &mut source,
        &mut validate_step,
        &mut validate_evaluation,
    )?;
    assert!(
        terminal_loss < initial_loss,
        "cross-process producer loss did not decrease: {initial_loss} -> {terminal_loss}"
    );
    let terminal_checkpoint = source.module_checkpoint()?;
    assert_cross_process_terminal_checkpoint(terminal_checkpoint.optimizer_checkpoint());
    terminal_checkpoint.save_file(&paths.terminal_checkpoint)?;

    let evidence = CrossProcessResumeEvidence {
        format_version: CROSS_PROCESS_EVIDENCE_VERSION,
        backend,
        pending_bundle_bytes: u64::try_from(pending_bundle.as_bytes().len())?,
        pending_bundle_checksum: cross_process_checksum(pending_bundle.as_bytes()),
        terminal_checkpoint_bytes: u64::try_from(terminal_checkpoint.as_bytes().len())?,
        terminal_checkpoint_checksum: cross_process_checksum(terminal_checkpoint.as_bytes()),
        evaluation_capture_identity,
        initial_evaluation_loss_bits: initial_loss.to_bits(),
        terminal_evaluation_loss_bits: terminal_loss.to_bits(),
        pending: CrossProcessCheckpointEvidence::new(pending_checkpoint.optimizer_checkpoint()),
        terminal: CrossProcessCheckpointEvidence::new(terminal_checkpoint.optimizer_checkpoint()),
    };
    evidence.save(&paths.evidence)?;
    assert_eq!(CrossProcessResumeEvidence::load(&paths.evidence)?, evidence);
    println!(
        "{} cross-process producer: replay=4, optimizer=1, accumulation=1, terminal_replay=11, terminal_optimizer=4, loss={initial_loss:.6}->{terminal_loss:.6}",
        backend.label()
    );
    Ok(())
}

fn consume_cross_process_file_resume<R, P, V, S, E>(
    backend: CrossProcessBackend,
    paths: &CrossProcessResumePaths,
    mut prepare: P,
    mut validate_preparation: V,
    validate_step: S,
    validate_evaluation: E,
) -> std::result::Result<(), Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<FileResumeTransformer>,
    ) -> Result<CompiledModuleAdamWSession<FileResumeTransformer, R>>,
    V: FnMut(&CompiledModuleAdamWSession<FileResumeTransformer, R>),
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    let evidence = CrossProcessResumeEvidence::load(&paths.evidence)?;
    evidence.validate_backend(backend)?;
    let pending_bundle = CompiledAdamWResumeBundle::load_file(&paths.pending_bundle)?;
    let terminal_checkpoint = CompiledModuleAdamWCheckpoint::load_file(&paths.terminal_checkpoint)?;
    validate_cross_process_file_binding(
        "pending bundle",
        pending_bundle.as_bytes(),
        evidence.pending_bundle_bytes,
        evidence.pending_bundle_checksum,
    )?;
    validate_cross_process_file_binding(
        "terminal checkpoint",
        terminal_checkpoint.as_bytes(),
        evidence.terminal_checkpoint_bytes,
        evidence.terminal_checkpoint_checksum,
    )?;
    assert_eq!(
        CrossProcessCheckpointEvidence::new(pending_bundle.checkpoint().optimizer_checkpoint()),
        evidence.pending
    );
    assert_eq!(
        CrossProcessCheckpointEvidence::new(terminal_checkpoint.optimizer_checkpoint()),
        evidence.terminal
    );
    assert_eq!(
        terminal_checkpoint.evaluation_capture_identity(),
        Some(evidence.evaluation_capture_identity)
    );
    assert_eq!(
        pending_bundle.checkpoint().evaluation_capture_identity(),
        Some(evidence.evaluation_capture_identity)
    );
    assert_cross_process_pending_checkpoint(pending_bundle.checkpoint().optimizer_checkpoint());
    assert_cross_process_terminal_checkpoint(terminal_checkpoint.optimizer_checkpoint());

    let initial_loss = f64::from_bits(evidence.initial_evaluation_loss_bits);
    let expected_terminal_loss = f64::from_bits(evidence.terminal_evaluation_loss_bits);
    assert!(initial_loss.is_finite());
    assert!(expected_terminal_loss.is_finite());
    assert!(expected_terminal_loss < initial_loss);

    let destination = FileResumeTransformer::new(0x9abc)?;
    destination.frozen_scale.replace(TensorData::scalar(7.0))?;
    destination
        .running_marker
        .replace(TensorData::scalar(29.0))?;
    let destination_tied_identity = destination.tokens.weight.id();
    let mut tied_alias = None;
    let mut destination_states = Vec::new();
    destination.visit("", &mut |name, parameter, kind| {
        if name == "lm_head.weight" {
            tied_alias = Some(parameter.clone());
        }
        destination_states.push((
            name,
            parameter.clone(),
            kind,
            parameter
                .snapshot()
                .expect("the cross-process destination remains readable"),
        ));
    });
    let tied_alias = tied_alias.expect("the destination exposes the tied output head");
    assert_eq!(tied_alias.id(), destination_tied_identity);

    let restored_plan =
        CompiledModuleAdamWPlan::restore_from_resume_bundle(destination, &pending_bundle)
            .map_err(|error| error.into_parts().1)?;
    assert_eq!(
        restored_plan.capture_identity(),
        evidence.pending.capture_identity
    );
    assert_eq!(
        restored_plan.evaluation_capture_identity(),
        Some(evidence.evaluation_capture_identity)
    );
    for (_, parameter, _, before) in &destination_states {
        let after = parameter.snapshot()?;
        assert_eq!(after.data, before.data);
        assert_eq!(after.version, before.version);
        assert_eq!(after.identity, before.identity);
        assert_eq!(after.trainable, before.trainable);
    }

    let mut resumed = prepare(restored_plan)?;
    validate_preparation(&resumed);
    assert_eq!(&resumed.module_checkpoint()?, pending_bundle.checkpoint());
    let terminal_loss =
        run_cross_process_file_resume_suffix(&mut resumed, validate_step, validate_evaluation)?;
    assert_eq!(
        terminal_loss.to_bits(),
        evidence.terminal_evaluation_loss_bits
    );
    assert_eq!(resumed.module_checkpoint()?, terminal_checkpoint);

    let (resumed_model, resumed_checkpoint) = resumed
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(resumed_checkpoint, terminal_checkpoint);
    let mut resumed_states = Vec::new();
    resumed_model.visit("", &mut |name, parameter, kind| {
        resumed_states.push((
            name,
            kind,
            parameter
                .snapshot()
                .expect("the cross-process result remains readable"),
        ));
    });
    assert_eq!(resumed_states.len(), destination_states.len());
    for ((name, _, kind, before), (actual_name, actual_kind, actual)) in
        destination_states.iter().zip(&resumed_states)
    {
        assert_eq!(actual_name, name);
        assert_eq!(actual_kind, kind);
        assert_eq!(actual.identity, before.identity);
        assert_eq!(actual.trainable, before.trainable);
        assert_eq!(
            actual.version,
            before
                .version
                .checked_add(1)
                .expect("successful publication cannot overflow a version")
        );
    }
    assert_eq!(resumed_model.tokens.weight.id(), destination_tied_identity);
    assert_eq!(tied_alias.id(), destination_tied_identity);
    assert_eq!(tied_alias.value()?, resumed_model.tokens.weight.value()?);
    assert!(resumed_model.positions.weight.is_trainable());
    assert!(!resumed_model.frozen_scale.is_trainable());
    assert!(!resumed_model.running_marker.is_trainable());
    assert_ne!(resumed_model.frozen_scale.value()?, TensorData::scalar(7.0));
    assert_ne!(
        resumed_model.running_marker.value()?,
        TensorData::scalar(29.0)
    );
    println!(
        "{} cross-process consumer: exact_terminal=true, different_init=true, published=true, loss={initial_loss:.6}->{terminal_loss:.6}",
        backend.label()
    );
    Ok(())
}

fn assert_native_file_resume_preparation(
    session: &CompiledModuleAdamWSession<FileResumeTransformer, NativeCpuCompiledAdamW<'_>>,
) {
    let preparation = session.native_cpu_preparation_report();
    for program in [
        preparation.main(),
        preparation
            .accumulation()
            .expect("three-step accumulation has a native sibling"),
        preparation
            .partial_flush()
            .expect("partial commit has a native sibling"),
        preparation
            .zero_grad()
            .expect("window reset has a native sibling"),
        preparation
            .evaluation()
            .expect("file resume has a native evaluator"),
    ] {
        assert!(program.native_item_count() > 0);
        assert_eq!(program.fallback_count(), 0);
    }
}

fn assert_native_file_resume_step(step: &NativeCpuCompiledAdamWStepResult) {
    assert!(step.report().native_item_count() > 0);
    assert!(step.report().executed_native_item_count() > 0);
    assert!(step.report().module_dispatch_count() > 0);
    assert_eq!(
        step.report().module_dispatched_native_item_count(),
        step.report().executed_native_item_count()
    );
    assert_eq!(step.report().fallback_count(), 0);
}

fn assert_native_file_resume_evaluation(evaluation: &NativeCpuCompiledEvaluationResult) {
    assert!(evaluation.report().native_item_count() > 0);
    assert!(evaluation.report().executed_native_item_count() > 0);
    assert!(evaluation.report().module_dispatch_count() > 0);
    assert_eq!(
        evaluation.report().module_dispatched_native_item_count(),
        evaluation.report().executed_native_item_count()
    );
    assert_eq!(evaluation.report().fallback_count(), 0);
}

fn run_native_cpu_file_resume() -> std::result::Result<(), Box<dyn Error>> {
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .vectorized(true)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let preparations = Cell::new(0_u64);
    run_file_resume::<NativeCpuCompiledAdamW<'_>, _, _, _, _>(
        "native CPU file resume",
        |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
        |session| {
            assert_native_file_resume_preparation(session);
            let invocation = preparations.get();
            if invocation > 0 {
                assert_eq!(
                    session
                        .native_cpu_preparation_report()
                        .compiler_process_count(),
                    0,
                    "artifact-restored preparation must reuse durable native modules"
                );
            }
            preparations.set(invocation + 1);
        },
        assert_native_file_resume_step,
        assert_native_file_resume_evaluation,
    )
}

fn run_cross_process_producer(
    backend: CrossProcessBackend,
    directory: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let paths = CrossProcessResumePaths::new(directory)?;
    match backend {
        CrossProcessBackend::Cpu => {
            let target = CpuSessionTarget::new()
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            produce_cross_process_file_resume::<CpuCompiledAdamW, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                |_| {},
                |_| {},
                |_| {},
            )
        }
        CrossProcessBackend::NativeCpu => {
            let executor = CapturedReplayExecutor::default();
            let target = NativeCpuSessionTarget::new(&executor)
                .vectorized(true)
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            produce_cross_process_file_resume::<NativeCpuCompiledAdamW<'_>, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                assert_native_file_resume_preparation,
                assert_native_file_resume_step,
                assert_native_file_resume_evaluation,
            )
        }
    }
}

fn run_cross_process_consumer(
    backend: CrossProcessBackend,
    directory: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let paths = CrossProcessResumePaths::new(directory)?;
    match backend {
        CrossProcessBackend::Cpu => {
            let target = CpuSessionTarget::new()
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            consume_cross_process_file_resume::<CpuCompiledAdamW, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                |_| {},
                |_| {},
                |_| {},
            )
        }
        CrossProcessBackend::NativeCpu => {
            let executor = CapturedReplayExecutor::default();
            let target = NativeCpuSessionTarget::new(&executor)
                .vectorized(true)
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            consume_cross_process_file_resume::<NativeCpuCompiledAdamW<'_>, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                assert_native_file_resume_preparation,
                assert_native_file_resume_step,
                assert_native_file_resume_evaluation,
            )
        }
    }
}

fn run_file_resume<R, P, V, S, E>(
    target_name: &str,
    mut prepare: P,
    mut validate_preparation: V,
    mut validate_step: S,
    mut validate_evaluation: E,
) -> std::result::Result<(), Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledCheckpointRestoreRuntime<Checkpoint = CompiledAdamWCheckpoint>
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<FileResumeTransformer>,
    ) -> Result<CompiledModuleAdamWSession<FileResumeTransformer, R>>,
    V: FnMut(&CompiledModuleAdamWSession<FileResumeTransformer, R>),
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    const LAST_REPLAY: u64 = 9;
    const WINDOW_LOSS_WEIGHT: u64 = 11;

    let schedule = CompiledMultiStepLr::new(1e-3, 0.5, [1])?;
    let config = file_resume_config(schedule.clone())?;
    assert!(config.clip_report_enabled());
    assert!(config.window_loss_report_enabled());
    assert_eq!(
        config.token_weighted_ignore_index(),
        Some((MaskedTransformerBatch::TARGETS, FILE_RESUME_IGNORE_INDEX))
    );
    assert_eq!(
        config.inputs().map(|(name, _, _)| name).collect::<Vec<_>>(),
        [
            MaskedTransformerBatch::TARGETS,
            MaskedTransformerBatch::TOKENS
        ],
        "sentinel targets are the only attention-validity source"
    );
    assert!(
        file_resume_batch(3)?.has_fully_masked_sample(),
        "the zero-length row must exercise fully masked attention"
    );
    let source = FileResumeTransformer::new(0x5678)?;
    let source_initial = source.state_dict()?;
    let source_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
        config.clone(),
        dropout_config(),
        source,
        build_file_resume,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph_and_ignore_index(build_file_resume_evaluation)
    .map_err(|error| error.into_parts().1)?;
    let capture_identity = source_plan.capture_identity();
    let evaluation_identity = source_plan
        .evaluation_capture_identity()
        .expect("the compiler-owned token-mean evaluator is attached");
    let program_artifact = source_plan.program_artifact()?;
    assert_eq!(source_plan.captured_multi_step_lr(), Some(&schedule));
    let mut uninterrupted = prepare(source_plan)?;
    validate_preparation(&uninterrupted);
    let before_initial_evaluation = uninterrupted.checkpoint()?;
    let initial_loss =
        evaluate_mean_file_resume_loss(&mut uninterrupted, &mut validate_evaluation)?;
    assert_eq!(uninterrupted.checkpoint()?, before_initial_evaluation);
    for replay in 1..=4 {
        let batch = file_resume_batch(replay)?;
        let loss_weight = loss_mask_weight(batch.loss_mask());
        let step = replay_prepared_policy_batch(&mut uninterrupted, batch)?;
        validate_step(&step);
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        if let Some(report) = step.clip_report() {
            assert!(report.is_finite());
            assert!(report.did_clip().is_some());
        }
        if let Some(report) = step.window_loss_report() {
            assert!(report.is_finite());
            assert_eq!(report.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(report.loss_weight(), WINDOW_LOSS_WEIGHT);
        }
    }
    assert_eq!(uninterrupted.step_count(), 4);
    assert_eq!(uninterrupted.optimizer_step()?, 1);
    assert_eq!(uninterrupted.accumulation_index()?, 1);
    let saved_optimizer_checkpoint = uninterrupted.checkpoint()?;
    let saved_dropout_cursor = saved_optimizer_checkpoint
        .info()
        .dropout_block_counter()
        .expect("attention and residual dropout are captured");
    assert!(saved_dropout_cursor > 0);

    let checkpoint = uninterrupted.module_checkpoint()?;
    assert_eq!(
        checkpoint.optimizer_checkpoint(),
        &saved_optimizer_checkpoint
    );
    assert_eq!(checkpoint.optimizer_checkpoint().info().replay_step(), 4);
    assert_eq!(
        checkpoint
            .optimizer_checkpoint()
            .info()
            .accumulation_index(),
        1
    );
    assert_eq!(
        checkpoint
            .optimizer_checkpoint()
            .info()
            .accumulated_token_count(),
        Some(5)
    );
    let resume_bundle = CompiledAdamWResumeBundle::new(program_artifact, checkpoint.clone())?;
    let resume_file = TemporaryCheckpointFile::new()?;
    resume_bundle.save_file(resume_file.path())?;
    let resume_bundle = CompiledAdamWResumeBundle::load_file(resume_file.path())?;
    assert_eq!(resume_bundle.checkpoint(), &checkpoint);
    let decoded = resume_bundle.checkpoint().clone();
    assert_eq!(decoded, checkpoint);
    assert!(
        decoded
            .optimizer_checkpoint()
            .info()
            .window_loss_report_enabled()
    );
    let pending_loss_numerator = decoded
        .optimizer_checkpoint()
        .info()
        .accumulated_loss_numerator()
        .expect("window-loss reporting checkpoints its pending F32 numerator");
    assert!(pending_loss_numerator.is_finite());
    assert_ne!(pending_loss_numerator, 0.0);

    let destination = FileResumeTransformer::new(0x9abc)?;
    destination.frozen_scale.replace(TensorData::scalar(7.0))?;
    destination
        .running_marker
        .replace(TensorData::scalar(29.0))?;
    let destination_initial = destination.state_dict()?;
    assert_ne!(destination_initial.tensors(), source_initial.tensors());
    assert_ne!(
        destination.positions.weight.value()?,
        source_initial.tensors()[FILE_RESUME_POLICY_FROZEN]
    );
    let destination_tied_identity = destination.tokens.weight.id();
    let mut tied_alias = None;
    let mut destination_states = Vec::new();
    destination.visit("", &mut |name, parameter, kind| {
        if name == "lm_head.weight" {
            tied_alias = Some(parameter.clone());
        }
        destination_states.push((
            name,
            parameter.clone(),
            kind,
            parameter
                .snapshot()
                .expect("the destination state remains readable"),
        ));
    });
    let tied_alias = tied_alias.expect("the destination exposes the tied output head");
    assert_eq!(tied_alias.id(), destination_tied_identity);

    let restored_plan =
        CompiledModuleAdamWPlan::restore_from_resume_bundle(destination, &resume_bundle)
            .map_err(|error| error.into_parts().1)?;
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(
        restored_plan.evaluation_capture_identity(),
        Some(evaluation_identity)
    );
    assert_eq!(restored_plan.captured_multi_step_lr(), Some(&schedule));
    for (_, parameter, _, before) in &destination_states {
        let after = parameter.snapshot()?;
        assert_eq!(after.data, before.data);
        assert_eq!(after.version, before.version);
        assert_eq!(after.identity, before.identity);
        assert_eq!(after.trainable, before.trainable);
    }

    let resumed = prepare(restored_plan)?;
    validate_preparation(&resumed);
    assert_eq!(
        resumed.checkpoint()?,
        decoded.optimizer_checkpoint().clone()
    );
    assert_eq!(
        resumed.checkpoint()?.info().dropout_block_counter(),
        Some(saved_dropout_cursor)
    );
    let mut resumed = resumed.into_training_session();
    let restored_checkpoint = resumed.checkpoint()?;
    assert_eq!(uninterrupted.checkpoint()?, restored_checkpoint);
    let expected_restore_probe =
        replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(5)?)?;
    let actual_restore_probe = replay_prepared_policy_batch(&mut resumed, file_resume_batch(5)?)?;
    validate_step(&expected_restore_probe);
    validate_step(&actual_restore_probe);
    assert_file_resume_steps_match(&actual_restore_probe, &expected_restore_probe);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    uninterrupted.restore_checkpoint_in_place(&restored_checkpoint)?;
    resumed.restore_checkpoint_in_place(&restored_checkpoint)?;
    assert_eq!(resumed.checkpoint()?, restored_checkpoint);
    assert_eq!(uninterrupted.checkpoint()?, restored_checkpoint);
    for replay in 5..=LAST_REPLAY {
        let expected =
            replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(replay)?)?;
        let actual = replay_prepared_policy_batch(&mut resumed, file_resume_batch(replay)?)?;
        validate_step(&expected);
        validate_step(&actual);
        assert_file_resume_steps_match(&actual, &expected);
        assert_eq!(
            actual.clip_report().is_some(),
            actual.accumulation_index() == 0
        );
        assert_eq!(actual.window_loss_report().is_some(), actual.did_update());
        if let Some(report) = actual.window_loss_report() {
            assert!(matches!(replay, 6 | 9));
            assert!(report.is_finite());
            assert_eq!(report.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(report.loss_weight(), WINDOW_LOSS_WEIGHT);
        }
        let actual_checkpoint = resumed.checkpoint()?;
        let expected_checkpoint = uninterrupted.checkpoint()?;
        assert_eq!(
            actual_checkpoint.info().dropout_block_counter(),
            expected_checkpoint.info().dropout_block_counter()
        );
        assert_eq!(actual_checkpoint, expected_checkpoint);
    }
    let expected_partial =
        replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(10)?)?;
    let actual_partial = replay_prepared_policy_batch(&mut resumed, file_resume_batch(10)?)?;
    validate_step(&expected_partial);
    validate_step(&actual_partial);
    assert_file_resume_steps_match(&actual_partial, &expected_partial);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let expected_flush = uninterrupted.commit_partial_window_with_rate_policy()?;
    let actual_flush = resumed.commit_partial_window_with_rate_policy()?;
    assert_eq!(
        actual_flush.did_commit_window(),
        expected_flush.did_commit_window()
    );
    assert_eq!(actual_flush.committed_microbatches(), 1);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let expected_discard =
        replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(11)?)?;
    let actual_discard = replay_prepared_policy_batch(&mut resumed, file_resume_batch(11)?)?;
    validate_step(&expected_discard);
    validate_step(&actual_discard);
    assert_file_resume_steps_match(&actual_discard, &expected_discard);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let expected_reset = uninterrupted.reset_gradient_window()?;
    let actual_reset = resumed.reset_gradient_window()?;
    assert_eq!(actual_reset, expected_reset);
    assert_eq!(actual_reset.discarded_microbatches(), 1);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    assert_eq!(resumed.runtime().optimizer_step()?, 4);
    assert_eq!(resumed.runtime().accumulation_index()?, 0);
    for (_, parameter, _, before) in &destination_states {
        let after = parameter.snapshot()?;
        assert_eq!(after.data, before.data);
        assert_eq!(after.version, before.version);
        assert_eq!(after.identity, before.identity);
        assert_eq!(after.trainable, before.trainable);
    }

    let before_evaluation_checkpoint = resumed.checkpoint()?;
    let before_evaluation_counter = before_evaluation_checkpoint.info().dropout_block_counter();
    let before_evaluation_accumulators = resumed.runtime().gradient_accumulator_snapshots()?;
    let uninterrupted_final_loss =
        evaluate_mean_file_resume_loss(&mut uninterrupted, &mut validate_evaluation)?;
    let final_loss = evaluate_mean_file_resume_loss(&mut resumed, &mut validate_evaluation)?;
    assert_eq!(final_loss.to_bits(), uninterrupted_final_loss.to_bits());
    assert_eq!(resumed.checkpoint()?, before_evaluation_checkpoint);
    assert_eq!(
        resumed.checkpoint()?.info().dropout_block_counter(),
        before_evaluation_counter
    );
    assert_eq!(
        resumed.runtime().gradient_accumulator_snapshots()?,
        before_evaluation_accumulators
    );

    let (uninterrupted_model, uninterrupted_checkpoint) = uninterrupted
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    let (resumed_model, resumed_checkpoint) = resumed
        .finish_with_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(
        &resumed_checkpoint,
        uninterrupted_checkpoint.optimizer_checkpoint()
    );
    assert_eq!(
        resumed_checkpoint.as_bytes(),
        uninterrupted_checkpoint.optimizer_checkpoint().as_bytes()
    );
    assert_eq!(
        resumed_model.state_dict()?,
        uninterrupted_model.state_dict()?
    );
    let mut expected_states = Vec::new();
    uninterrupted_model.visit("", &mut |name, parameter, kind| {
        expected_states.push((
            name,
            kind,
            parameter
                .snapshot()
                .expect("the finished reference state remains readable"),
        ));
    });
    let mut resumed_states = Vec::new();
    resumed_model.visit("", &mut |name, parameter, kind| {
        resumed_states.push((
            name,
            kind,
            parameter
                .snapshot()
                .expect("the finished destination state remains readable"),
        ));
    });
    assert_eq!(resumed_states.len(), destination_states.len());
    assert_eq!(resumed_states.len(), expected_states.len());
    for (
        ((name, _, kind, before), (expected_name, expected_kind, expected)),
        (actual_name, actual_kind, actual),
    ) in destination_states
        .iter()
        .zip(&expected_states)
        .zip(&resumed_states)
    {
        assert_eq!(actual_name, name);
        assert_eq!(expected_name, name);
        assert_eq!(actual_kind, kind);
        assert_eq!(expected_kind, kind);
        assert_eq!(actual.data, expected.data);
        assert_eq!(actual.identity, before.identity);
        assert_eq!(actual.trainable, before.trainable);
        assert_eq!(
            actual.version,
            before
                .version
                .checked_add(1)
                .expect("successful publication cannot overflow a version")
        );
    }
    assert_eq!(resumed_model.tokens.weight.id(), destination_tied_identity);
    assert_eq!(tied_alias.id(), destination_tied_identity);
    assert_eq!(tied_alias.value()?, resumed_model.tokens.weight.value()?);
    assert!(resumed_model.positions.weight.is_trainable());
    assert!(!resumed_model.frozen_scale.is_trainable());
    assert!(!resumed_model.running_marker.is_trainable());
    assert_eq!(
        resumed_model.frozen_scale.value()?,
        uninterrupted_model.frozen_scale.value()?
    );
    assert_eq!(
        resumed_model.running_marker.value()?,
        uninterrupted_model.running_marker.value()?
    );

    assert!(
        final_loss < initial_loss,
        "file-resumed two-block causal Transformer loss did not decrease: {initial_loss} -> {final_loss}"
    );
    println!(
        "{target_name}: capture={capture_identity:016x}, checkpoint=(replay=4, optimizer=1, accumulation=1), optimizer_steps=4, eval_mean_sparse_loss={initial_loss:.6} -> {final_loss:.6}, exact_resume=true, different_init=true, published=true"
    );
    Ok(())
}

fn run_native_cpu_scoreboard() -> std::result::Result<(), Box<dyn Error>> {
    const SAMPLES: u64 = 3;
    const EXPECTED_LOSS_WEIGHTS: [u64; SAMPLES as usize] = [5, 3, 3];
    const EXPECTED_MAIN_MODULE_DISPATCHES: usize = 1;
    const EXPECTED_ACCUMULATION_MODULE_DISPATCHES: usize = 2;
    // The module exposes 37 parameter traversal entries. The tied LM head
    // deduplicates with tokens.weight, and positions.weight is policy-frozen.
    const EXPECTED_ADAMW_UPDATE_GROUPS: usize = 35;
    // Parameter/m1/m2 plus optimizer step are definitionally unchanged.
    // Prepared schedule simplification may prove additional accumulator
    // pass-throughs; index/token/loss/dropout must still receive new banks.
    const GUARANTEED_RETAINED_RECURRENT_STATES: u64 = (EXPECTED_ADAMW_UPDATE_GROUPS * 3 + 1) as u64;
    const REMAINING_RECURRENT_STATES: u64 = (EXPECTED_ADAMW_UPDATE_GROUPS + 4) as u64;
    const MANDATORY_REPLACED_RECURRENT_STATES: u64 = 4;
    const EXPECTED_MAIN_RENDERED_ENTRIES: usize = 787 - EXPECTED_ADAMW_UPDATE_GROUPS * 3;
    const EXPECTED_ACCUMULATION_RENDERED_ENTRIES: usize = 465;
    const EXPECTED_MAIN_EXECUTED_ENTRIES: u64 = 681;
    const EXPECTED_ACCUMULATION_EXECUTED_ENTRIES: u64 = 358;
    const EXPECTED_RECURRENT_STATES: usize = 145;
    const EXPECTED_RECURRENT_BYTES: u64 = 1_924;
    const EXPECTED_SHARED_ACCUMULATION_PREFIX: usize = 319;
    const EXPECTED_COLD_COMPILER_PROCESSES: usize = 6;
    // One accumulator per update group, plus loss numerator, index, and token count.
    const EXPECTED_ZERO_GRAD_ENTRIES: usize = EXPECTED_ADAMW_UPDATE_GROUPS + 3;

    let source = FileResumeTransformer::new(7)?;
    let optimized_parameters = source
        .trainable_parameters()?
        .into_iter()
        .filter(|(name, _)| name != FILE_RESUME_POLICY_FROZEN)
        .collect::<Vec<_>>();
    assert_eq!(optimized_parameters.len(), EXPECTED_ADAMW_UPDATE_GROUPS);
    let optimized_parameter_bytes = optimized_parameters
        .iter()
        .map(|(_, parameter)| {
            parameter
                .value()
                .map(|value| value.len() * value.dtype().itemsize())
        })
        .sum::<Result<usize>>()?;
    let guaranteed_retained_recurrent_bytes =
        u64::try_from(optimized_parameter_bytes * 3 + DType::U64.itemsize())?;
    let remaining_recurrent_bytes = u64::try_from(
        optimized_parameter_bytes + DType::U64.itemsize() * 3 + DType::F32.itemsize(),
    )?;
    let mandatory_replaced_recurrent_bytes =
        u64::try_from(DType::U64.itemsize() * 3 + DType::F32.itemsize())?;
    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1])?;
    let builds = Cell::new(0);
    let compile_started = Instant::now();
    let plan = CompiledAdamWPlan::compile_module_graph_with_dropout(
        scoreboard_config(schedule)?,
        dropout_config(),
        &source,
        |model, graph, inputs, dropout| {
            builds.set(builds.get() + 1);
            build_scoreboard(model, graph, inputs, dropout)
        },
    )?;
    let compile_wall_time = compile_started.elapsed();
    assert_eq!(builds.get(), 1, "the training graph must compile once");
    let inspection = plan.inspection()?;
    let compile_phases = inspection
        .compile_phases()
        .expect("fresh compilation retains phase evidence");
    assert_eq!(compile_phases.compile_count(), 1);
    assert_eq!(
        compile_phases.main_capture().logical_schedule_item_count(),
        Some(inspection.main().1.schedule_item_count)
    );
    assert!(compile_phases.accumulation_capture().is_some());
    assert!(compile_phases.partial_flush().is_some());
    assert!(compile_phases.zero_grad().is_some());
    assert!(compile_phases.evaluation().is_none());
    assert!(compile_phases.measured_wall_time().unwrap() <= compile_wall_time);
    assert_eq!(
        u64::try_from(inspection.recurrent_state_count())?,
        GUARANTEED_RETAINED_RECURRENT_STATES + REMAINING_RECURRENT_STATES
    );
    assert_eq!(
        inspection.recurrent_state_count(),
        EXPECTED_RECURRENT_STATES
    );
    assert_eq!(
        u64::try_from(inspection.recurrent_state_bytes())?,
        EXPECTED_RECURRENT_BYTES
    );

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .vectorized(true)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let prepare_started = Instant::now();
    let mut session = plan.prepare(&target)?;
    let prepare_wall_time = prepare_started.elapsed();
    let main_preparation = session.preparation_report().main();
    assert_eq!(main_preparation.native_item_count(), 787);
    assert_eq!(
        main_preparation.work().rendered_entry_count(),
        EXPECTED_MAIN_RENDERED_ENTRIES
    );
    assert_eq!(main_preparation.work().loaded_module_count(), 1);
    assert_eq!(main_preparation.work().referenced_module_count(), 1);
    assert_eq!(
        main_preparation.work().unique_rendered_entry_count(),
        EXPECTED_MAIN_RENDERED_ENTRIES
    );
    assert_eq!(main_preparation.work().shared_prefix_entry_count(), 0);
    assert!(main_preparation.work().rendered_source_bytes() > 0);
    assert_eq!(main_preparation.work().shared_prefix_source_bytes(), 0);
    assert_eq!(
        main_preparation.work().unique_rendered_source_bytes(),
        main_preparation.work().rendered_source_bytes()
    );
    assert!(main_preparation.work().compiler_invocation_count() <= 3);
    if main_preparation.work().compiler_invocation_count() != 0 {
        assert_eq!(main_preparation.work().combined_compile_link_count(), 0);
        assert_eq!(main_preparation.work().object_compile_count(), 2);
        assert_eq!(main_preparation.work().linker_invocation_count(), 1);
    }
    let accumulation_preparation = session
        .preparation_report()
        .accumulation()
        .expect("scoreboard configuration captures accumulation-only replay");
    assert_ne!(
        accumulation_preparation.capture_identity(),
        main_preparation.capture_identity()
    );
    assert!(accumulation_preparation.native_item_count() < main_preparation.native_item_count());
    assert!(
        accumulation_preparation.work().rendered_entry_count()
            < main_preparation.work().rendered_entry_count()
    );
    assert_eq!(
        accumulation_preparation.work().rendered_entry_count(),
        EXPECTED_ACCUMULATION_RENDERED_ENTRIES
    );
    assert_eq!(accumulation_preparation.work().loaded_module_count(), 1);
    assert_eq!(accumulation_preparation.work().referenced_module_count(), 2);
    assert_eq!(
        accumulation_preparation
            .work()
            .unique_rendered_entry_count(),
        EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert_eq!(
        accumulation_preparation.work().shared_prefix_entry_count(),
        EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert!(accumulation_preparation.work().shared_prefix_source_bytes() > 0);
    assert_eq!(
        accumulation_preparation.work().shared_prefix_source_bytes()
            + accumulation_preparation
                .work()
                .unique_rendered_source_bytes(),
        accumulation_preparation.work().rendered_source_bytes()
    );
    assert_eq!(
        accumulation_preparation.cache_hit_count(),
        EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert_eq!(
        accumulation_preparation.cache_miss_count(),
        EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert!(accumulation_preparation.work().compiler_invocation_count() <= 1);
    let partial_preparation = session
        .preparation_report()
        .partial_flush()
        .expect("scoreboard configuration captures partial flush");
    assert_eq!(partial_preparation.native_item_count(), 357);
    assert_eq!(
        partial_preparation.work().rendered_entry_count(),
        357 - EXPECTED_ADAMW_UPDATE_GROUPS * 3
    );
    assert_eq!(partial_preparation.work().shared_prefix_entry_count(), 0);
    assert_eq!(
        partial_preparation.work().unique_rendered_entry_count(),
        partial_preparation.work().rendered_entry_count()
    );
    assert_eq!(partial_preparation.work().referenced_module_count(), 1);
    assert!(partial_preparation.work().compiler_invocation_count() <= 1);
    assert_eq!(
        session
            .preparation_report()
            .zero_grad()
            .expect("scoreboard configuration captures zero grad")
            .native_item_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    let zero_grad_preparation = session
        .preparation_report()
        .zero_grad()
        .expect("scoreboard configuration captures zero grad");
    assert_eq!(
        zero_grad_preparation.work().rendered_entry_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    assert_eq!(
        zero_grad_preparation.work().unique_rendered_entry_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    assert_eq!(zero_grad_preparation.work().referenced_module_count(), 1);
    assert!(zero_grad_preparation.work().compiler_invocation_count() <= 1);
    assert_eq!(
        [
            main_preparation,
            accumulation_preparation,
            partial_preparation,
            zero_grad_preparation,
        ]
        .into_iter()
        .map(|program| program.work().unique_rendered_entry_count())
        .sum::<usize>(),
        1_118
    );
    assert!(
        session.preparation_report().compiler_process_count() <= EXPECTED_COLD_COMPILER_PROCESSES
    );
    assert!(
        session
            .preparation_report()
            .max_parallel_compiler_process_count()
            <= 2
    );
    assert!((1..=2).contains(&session.preparation_report().max_parallel_render_job_count()));
    if env::var_os("RUSTGRAD_REQUIRE_COLD_NATIVE_SCOREBOARD").is_some() {
        assert_eq!(
            session.preparation_report().compiler_process_count(),
            EXPECTED_COLD_COMPILER_PROCESSES
        );
        assert_eq!(
            session
                .preparation_report()
                .max_parallel_compiler_process_count(),
            2
        );
        assert!(
            session
                .preparation_report()
                .compiler_process_overlap_wall_time()
                > Duration::ZERO
        );
        assert!(
            session
                .preparation_report()
                .parallel_module_overlap_wall_time()
                > Duration::ZERO
        );
        assert_eq!(
            session.preparation_report().max_parallel_render_job_count(),
            2
        );
        assert!(
            session
                .preparation_report()
                .parallel_render_overlap_wall_time()
                > Duration::ZERO
        );
        assert_eq!(main_preparation.work().object_compile_count(), 2);
        assert_eq!(main_preparation.work().linker_invocation_count(), 1);
        for preparation in [
            accumulation_preparation,
            partial_preparation,
            zero_grad_preparation,
        ] {
            assert_eq!(preparation.work().combined_compile_link_count(), 1);
            assert_eq!(preparation.work().object_compile_count(), 0);
            assert_eq!(preparation.work().linker_invocation_count(), 0);
        }
    }
    let main_segmentation = *session.preparation_report().main().dispatch_segmentation();
    let expected_module_dispatches = u64::try_from(EXPECTED_MAIN_MODULE_DISPATCHES)?;
    assert_eq!(
        main_segmentation.segment_count(),
        expected_module_dispatches
    );
    assert_eq!(main_segmentation.dispatch_reached_module_count(), 1);
    assert_eq!(main_segmentation.terminal_segment_count(), 1);
    assert_eq!(main_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(main_segmentation.module_change_count(), 0);
    assert_eq!(main_segmentation.output_slot_alias_count(), 0);
    assert_eq!(main_segmentation.derived_slot_dependency_count(), 0);
    let accumulation_segmentation = *accumulation_preparation.dispatch_segmentation();
    assert_eq!(accumulation_segmentation.terminal_segment_count(), 1);
    assert_eq!(accumulation_segmentation.dispatch_reached_module_count(), 2);
    assert_eq!(accumulation_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(
        accumulation_segmentation.segment_count(),
        u64::try_from(EXPECTED_ACCUMULATION_MODULE_DISPATCHES)?
    );
    assert_eq!(
        accumulation_segmentation.module_change_count(),
        u64::try_from(accumulation_preparation.work().referenced_module_count() - 1)?,
        "the accumulation preparation crosses once from its shared prefix to its suffix module"
    );
    assert_eq!(accumulation_segmentation.output_slot_alias_count(), 0);
    assert_eq!(accumulation_segmentation.derived_slot_dependency_count(), 0);
    let mut scoreboard = NativeTrainingScoreboard::new(
        inspection.clone(),
        session.preparation_report(),
        compile_wall_time,
        prepare_wall_time,
    )?;
    let recurrent_state_bytes = u64::try_from(inspection.recurrent_state_bytes())?;
    assert_eq!(
        recurrent_state_bytes,
        guaranteed_retained_recurrent_bytes + remaining_recurrent_bytes
    );
    let mut stable_accumulation_executed_native_items = None;
    let mut committed_executed_native_items = None;
    for replay in 1..=SAMPLES {
        let batch = masked_batch(replay)?;
        assert_eq!(
            loss_mask_weight(&batch.loss_mask),
            EXPECTED_LOSS_WEIGHTS[(replay - 1) as usize]
        );
        let step = session.step_batch_commit_only_scheduled(batch)?;
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
        assert!(step.outputs().is_empty());
        assert!(step.loss().values()[0].is_finite());
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        if let Some(clip) = step.clip_report() {
            assert!(clip.is_finite());
            assert!(clip.did_clip().is_some());
        }
        if let Some(window) = step.window_loss_report() {
            assert!(window.is_finite());
            assert_eq!(window.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(
                window.loss_weight(),
                EXPECTED_LOSS_WEIGHTS.iter().copied().sum::<u64>()
            );
        }
        let report = step.report();
        assert_eq!(
            report.traffic().materialized_egress_count(),
            if step.did_update() { 5 } else { 1 }
        );
        assert_eq!(
            report.traffic().materialized_egress_bytes(),
            if step.did_update() { 24 } else { 4 }
        );
        let executed = report.executed_native_item_count();
        assert!(executed > 0);
        assert!(executed <= report.native_item_count());
        if step.did_update() {
            assert_eq!(
                u64::try_from(report.module_dispatch_count())?,
                main_segmentation.segment_count(),
                "the commit program must retain its authenticated safe-segment partition"
            );
        }
        assert_eq!(report.module_dispatched_native_item_count(), executed);
        assert!(report.module_dispatch_count() < executed);
        if step.did_update() {
            assert!(committed_executed_native_items.replace(executed).is_none());
        } else if let Some(expected) = stable_accumulation_executed_native_items {
            assert_eq!(executed, expected);
        } else {
            stable_accumulation_executed_native_items = Some(executed);
        }
        scoreboard.record_step(&step)?;
    }

    let checkpoint_started = Instant::now();
    let checkpoint = session.checkpoint()?;
    let checkpoint_wall_time = checkpoint_started.elapsed();
    scoreboard.observe_checkpoint(&checkpoint, checkpoint_wall_time)?;
    let report = scoreboard.report()?;
    let reported_compile = report.compile_phases().expect("v23 reports compile phases");
    assert_eq!(reported_compile.compile_count(), 1);
    assert!(reported_compile.evaluation().is_none());
    let executed_native_items = report
        .main_replay_executed_native_item_count()
        .expect("current native CPU scoreboard reports executed JIT items");
    let expected_main_rendered_entries = u64::try_from(EXPECTED_MAIN_RENDERED_ENTRIES)?;
    let expected_executed_native_items = expected_main_rendered_entries
        .checked_sub(1)
        .expect("the main program has one nonexecuted native entry");
    assert_eq!(executed_native_items, expected_executed_native_items);
    assert_eq!(executed_native_items, EXPECTED_MAIN_EXECUTED_ENTRIES);
    assert_eq!(
        report.main().rendered_entry_count(),
        expected_main_rendered_entries
    );
    assert_eq!(report.main().loaded_module_count(), 1);
    assert_eq!(report.main().referenced_module_count(), 1);
    assert_eq!(
        report.main().unique_rendered_entry_count(),
        expected_main_rendered_entries
    );
    assert_eq!(report.main().shared_prefix_entry_count(), 0);
    let recorded_segmentation = report
        .main()
        .dispatch_segmentation()
        .expect("current scoreboard reports dispatch segmentation");
    assert_eq!(
        recorded_segmentation.segment_count(),
        u64::try_from(EXPECTED_MAIN_MODULE_DISPATCHES)?
    );
    assert_eq!(recorded_segmentation.dispatch_reached_module_count(), 1);
    assert_eq!(recorded_segmentation.terminal_segment_count(), 1);
    assert_eq!(recorded_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(recorded_segmentation.module_change_count(), 0);
    assert_eq!(recorded_segmentation.output_slot_alias_count(), 0);
    assert_eq!(recorded_segmentation.derived_slot_dependency_count(), 0);
    assert!(report.main().compiler_invocation_count() <= 3);
    if report.main().compiler_invocation_count() != 0 {
        assert_eq!(report.main().combined_compile_link_count(), 0);
        assert_eq!(report.main().object_compile_count(), 2);
        assert_eq!(report.main().linker_invocation_count(), 1);
    }
    let accumulation_program = report
        .accumulation()
        .expect("current scoreboard reports accumulation preparation");
    assert_eq!(
        accumulation_program.rendered_entry_count(),
        u64::try_from(EXPECTED_ACCUMULATION_RENDERED_ENTRIES)?
    );
    assert_eq!(accumulation_program.loaded_module_count(), 1);
    assert_eq!(accumulation_program.referenced_module_count(), 2);
    assert_eq!(
        accumulation_program.unique_rendered_entry_count(),
        u64::try_from(
            EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
        )?
    );
    assert_eq!(
        accumulation_program.shared_prefix_entry_count(),
        u64::try_from(EXPECTED_SHARED_ACCUMULATION_PREFIX)?
    );
    assert_eq!(
        accumulation_program.cache_hit_count(),
        u64::try_from(EXPECTED_SHARED_ACCUMULATION_PREFIX)?
    );
    assert_eq!(
        accumulation_program.cache_miss_count(),
        u64::try_from(
            EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
        )?
    );
    assert_eq!(
        accumulation_program.shared_prefix_source_program_index(),
        Some(0)
    );
    assert_eq!(
        accumulation_program.shared_prefix_source_native_identity(),
        Some(report.main().native_identity())
    );
    let accumulation_segmentation = accumulation_program
        .dispatch_segmentation()
        .expect("current scoreboard reports accumulation dispatch segmentation");
    assert_eq!(accumulation_segmentation.terminal_segment_count(), 1);
    assert_eq!(accumulation_segmentation.dispatch_reached_module_count(), 2);
    assert_eq!(accumulation_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(
        accumulation_segmentation.segment_count(),
        u64::try_from(EXPECTED_ACCUMULATION_MODULE_DISPATCHES)?
    );
    assert_eq!(
        accumulation_segmentation.module_change_count(),
        accumulation_program.referenced_module_count() - 1,
        "the accumulation program crosses exactly once from its shared prefix to its suffix module"
    );
    assert_eq!(accumulation_segmentation.output_slot_alias_count(), 0);
    assert_eq!(accumulation_segmentation.derived_slot_dependency_count(), 0);
    let program_prepare_wall_time = [
        Some(report.main()),
        report.accumulation(),
        report.partial_flush(),
        report.zero_grad(),
        report.evaluation(),
    ]
    .into_iter()
    .flatten()
    .fold(Duration::ZERO, |total, program| {
        let timing = program
            .preparation_timing()
            .expect("current native CPU scoreboard reports every program preparation phase");
        total
            .checked_add(
                timing
                    .total()
                    .to_duration()
                    .expect("program preparation total is representable"),
            )
            .expect("program preparation totals fit")
    });
    assert_eq!(
        program_prepare_wall_time
            .checked_sub(
                report
                    .prepare_parallel_module_overlap_wall_time()
                    .expect("current native CPU scoreboard reports parallel module overlap")
                    .to_duration()?,
            )
            .expect("parallel module overlap fits program preparation")
            .checked_sub(
                report
                    .prepare_parallel_render_overlap_wall_time()
                    .expect("current native CPU scoreboard reports parallel render overlap")
                    .to_duration()?,
            )
            .expect("parallel render overlap fits program preparation")
            .checked_add(
                report
                    .prepare_runtime_overhead_wall_time()
                    .expect("current native CPU scoreboard reports whole-prepare overhead")
                    .to_duration()?,
            )
            .expect("whole preparation phases fit"),
        report.prepare_wall_time().to_duration()?
    );
    assert_eq!(report.successful_replay_count(), SAMPLES);
    let step_phases = report
        .step_phases()
        .expect("phase-aware scoreboard reports successful main-step classes");
    assert_eq!(
        step_phases.first().phase(),
        rustgrad::NativeTrainingStepPhase::AccumulationOnly
    );
    assert_eq!(
        step_phases
            .first()
            .native_dispatcher_wall_time()
            .expect("first phase reports native dispatcher time")
            .to_duration()?
            .checked_add(
                step_phases
                    .first()
                    .executor_host_wall_time()
                    .expect("first phase reports executor host time")
                    .to_duration()?
            )
            .expect("first classified executor durations fit"),
        step_phases.first().executor_wall_time().to_duration()?
    );
    assert_eq!(
        step_phases
            .warm_accumulation_only()
            .expect("the second replay remains accumulation-only")
            .sample_count(),
        1
    );
    assert_eq!(
        step_phases
            .warm_optimizer_commit()
            .expect("the third replay commits the accumulated window")
            .sample_count(),
        1
    );
    for phase in [
        step_phases.warm_accumulation_only(),
        step_phases.warm_optimizer_commit(),
    ]
    .into_iter()
    .flatten()
    {
        assert_eq!(
            phase
                .native_dispatcher_total_wall_time()
                .expect("current phase reports native dispatcher time")
                .to_duration()?
                .checked_add(
                    phase
                        .executor_host_total_wall_time()
                        .expect("current phase reports executor host time")
                        .to_duration()?
                )
                .expect("classified executor phase durations fit"),
            phase.executor_total_wall_time().to_duration()?
        );
        assert_eq!(
            phase
                .executor_total_wall_time()
                .to_duration()?
                .checked_add(phase.recurrent_overhead_total_wall_time().to_duration()?)
                .expect("classified warm replay phase durations fit"),
            phase.total_wall_time().to_duration()?
        );
    }
    let executor_timing = report
        .main_replay_executor_wall_time()
        .expect("current native CPU scoreboard reports sealed executor timing");
    let dispatcher_timing = report
        .main_replay_native_dispatcher_wall_time()
        .expect("current native CPU scoreboard reports native dispatcher timing");
    let executor_host_timing = report
        .main_replay_executor_host_wall_time()
        .expect("current native CPU scoreboard reports executor host timing");
    let overhead_timing = report
        .main_replay_recurrent_overhead_wall_time()
        .expect("current native CPU scoreboard reports recurrent overhead timing");
    assert_eq!(
        dispatcher_timing
            .first()
            .to_duration()?
            .checked_add(executor_host_timing.first().to_duration()?)
            .expect("first executor phase durations fit"),
        executor_timing.first().to_duration()?
    );
    assert_eq!(
        dispatcher_timing
            .steady_total()
            .to_duration()?
            .checked_add(executor_host_timing.steady_total().to_duration()?)
            .expect("steady executor phase durations fit"),
        executor_timing.steady_total().to_duration()?
    );
    assert_eq!(
        executor_timing
            .first()
            .to_duration()?
            .checked_add(overhead_timing.first().to_duration()?)
            .expect("first replay phase durations fit"),
        report.first_replay_wall_time().to_duration()?
    );
    assert_eq!(
        executor_timing
            .steady_total()
            .to_duration()?
            .checked_add(overhead_timing.steady_total().to_duration()?)
            .expect("steady replay phase durations fit"),
        report.steady_replay_total_wall_time().to_duration()?
    );
    assert_eq!(
        Some(usize::try_from(executed_native_items)?),
        committed_executed_native_items
    );
    assert!(executed_native_items <= report.main().native_item_count());
    let replay_traffic = report
        .main_replay_traffic()
        .expect("current native CPU scoreboard reports replay traffic");
    assert_eq!(replay_traffic.external_input_import_count(), 0);
    assert_eq!(replay_traffic.external_input_import_bytes(), 0);
    assert_eq!(replay_traffic.materialized_egress_count(), 5);
    assert_eq!(replay_traffic.materialized_egress_bytes(), 24);
    assert_eq!(
        replay_traffic.borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        replay_traffic.borrowed_recurrent_output_bytes(),
        recurrent_state_bytes
    );
    let accumulation = report
        .accumulation()
        .expect("phase-specialized scoreboard reports the accumulation program");
    let accumulation_executed = report
        .accumulation_replay_executed_native_item_count()
        .expect("phase-specialized scoreboard reports accumulation execution");
    assert_eq!(
        accumulation_executed,
        EXPECTED_ACCUMULATION_EXECUTED_ENTRIES
    );
    assert_eq!(
        Some(usize::try_from(accumulation_executed)?),
        stable_accumulation_executed_native_items
    );
    assert!(accumulation_executed < executed_native_items);
    assert_eq!(
        report.accumulation_schedule_cache_keys().len(),
        usize::try_from(accumulation.native_item_count())?
    );
    let accumulation_traffic = report
        .accumulation_replay_traffic()
        .expect("phase-specialized scoreboard reports accumulation traffic");
    assert_eq!(accumulation_traffic.external_input_import_count(), 0);
    assert_eq!(accumulation_traffic.external_input_import_bytes(), 0);
    assert_eq!(accumulation_traffic.materialized_egress_count(), 1);
    assert_eq!(accumulation_traffic.materialized_egress_bytes(), 4);
    assert_eq!(
        accumulation_traffic.borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        accumulation_traffic.borrowed_recurrent_output_bytes()
            + accumulation_traffic.retained_recurrent_state_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        accumulation_traffic.replaced_recurrent_state_bytes(),
        accumulation_traffic.borrowed_recurrent_output_bytes()
    );
    assert_eq!(
        accumulation_traffic.retained_recurrent_state_count()
            + accumulation_traffic.replaced_recurrent_state_count(),
        u64::try_from(inspection.recurrent_state_count())?
    );
    assert!(
        accumulation_traffic.retained_recurrent_state_count()
            >= GUARANTEED_RETAINED_RECURRENT_STATES
    );
    assert!(accumulation_traffic.replaced_recurrent_state_count() <= REMAINING_RECURRENT_STATES);
    assert!(
        accumulation_traffic.replaced_recurrent_state_count()
            >= MANDATORY_REPLACED_RECURRENT_STATES
    );
    assert!(
        accumulation_traffic.retained_recurrent_state_bytes()
            >= guaranteed_retained_recurrent_bytes
    );
    assert!(accumulation_traffic.replaced_recurrent_state_bytes() <= remaining_recurrent_bytes);
    assert!(
        accumulation_traffic.replaced_recurrent_state_bytes() >= mandatory_replaced_recurrent_bytes
    );

    let prepared_retained_recurrent_states = accumulation_traffic.retained_recurrent_state_count();
    let prepared_replaced_recurrent_states = accumulation_traffic.replaced_recurrent_state_count();
    let prepared_retained_recurrent_bytes = accumulation_traffic.retained_recurrent_state_bytes();
    let prepared_replaced_recurrent_bytes = accumulation_traffic.replaced_recurrent_state_bytes();

    let restored = plan.restore_checkpoint(&checkpoint)?;
    let restored_inspection = restored.inspection()?;
    assert_eq!(restored_inspection.main(), inspection.main());
    assert_eq!(
        restored_inspection.accumulation(),
        inspection.accumulation()
    );
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
    assert_eq!(builds.get(), 1, "checkpoint restore must not rebuild");
    let mut restored_session = restored.prepare(&target)?;
    let restored_preparation = restored_session.preparation_report();
    assert_eq!(restored_preparation.compiler_process_count(), 0);
    assert_eq!(
        restored_preparation.max_parallel_compiler_process_count(),
        0
    );
    assert_eq!(
        restored_preparation.parallel_module_overlap_wall_time(),
        Duration::ZERO
    );
    let capsule_diagnostics = restored_preparation.render_capsule_diagnostics();
    assert_eq!(
        capsule_diagnostics
            .iter()
            .map(|diagnostic| diagnostic.role())
            .collect::<Vec<_>>(),
        [
            NativeCpuRenderCapsuleProgramRole::Main,
            NativeCpuRenderCapsuleProgramRole::Accumulation,
            NativeCpuRenderCapsuleProgramRole::PartialFlush,
            NativeCpuRenderCapsuleProgramRole::ZeroGrad,
        ]
    );
    assert_eq!(
        restored_preparation.max_parallel_render_job_count(),
        0,
        "warm render capsule diagnostics: {capsule_diagnostics:#?}"
    );
    assert_eq!(restored_preparation.render_capsule_hit_count(), 4);
    assert_eq!(restored_preparation.render_capsule_miss_count(), 0);
    assert_eq!(restored_preparation.local_render_job_count(), 0);
    assert_eq!(
        restored_preparation.parallel_render_overlap_wall_time(),
        Duration::ZERO
    );
    let restored_programs = [
        Some(restored_preparation.main()),
        restored_preparation.accumulation(),
        restored_preparation.partial_flush(),
        restored_preparation.zero_grad(),
        restored_preparation.evaluation(),
    ];
    assert_eq!(restored_programs.iter().flatten().count(), 4);
    assert!(restored_preparation.evaluation().is_none());
    for program in restored_programs.into_iter().flatten() {
        assert_eq!(program.cache_miss_count(), 0);
        assert_eq!(program.cache_hit_count(), program.native_item_count());
        assert!(program.work().rendered_entry_count() <= program.native_item_count());
        assert_eq!(
            program.work().loaded_module_count(),
            usize::from(program.work().unique_rendered_entry_count() != 0)
        );
        assert_eq!(program.work().durable_artifact_cache_hit_count(), 0);
        assert_eq!(program.work().durable_artifact_cache_miss_count(), 0);
        assert_eq!(program.work().compiler_invocation_count(), 0);
        assert_eq!(
            program.phases().compiler_process_wall_time(),
            Duration::ZERO
        );
    }
    let restored_step = restored_session.step_batch_scheduled(masked_batch(SAMPLES + 1)?)?;
    let restored_report = restored_step.report();
    assert_eq!(restored_report.fallback_count(), 0);
    assert_eq!(
        u64::try_from(restored_report.native_item_count())?,
        accumulation.native_item_count()
    );
    assert_eq!(restored_report.traffic().external_input_import_count(), 0);
    assert_eq!(restored_report.traffic().external_input_import_bytes(), 0);
    assert_eq!(
        restored_report.traffic().borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        restored_report.traffic().borrowed_recurrent_output_bytes()
            + restored_report.traffic().retained_recurrent_state_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        restored_report.traffic().replaced_recurrent_state_bytes(),
        restored_report.traffic().borrowed_recurrent_output_bytes()
    );
    assert_eq!(
        restored_report.traffic().retained_recurrent_state_count()
            + restored_report.traffic().replaced_recurrent_state_count(),
        u64::try_from(restored_inspection.recurrent_state_count())?
    );
    assert_eq!(
        restored_report.traffic().retained_recurrent_state_count(),
        prepared_retained_recurrent_states
    );
    assert_eq!(
        restored_report.traffic().replaced_recurrent_state_count(),
        prepared_replaced_recurrent_states
    );
    assert_eq!(
        restored_report.traffic().retained_recurrent_state_bytes(),
        prepared_retained_recurrent_bytes
    );
    assert_eq!(
        restored_report.traffic().replaced_recurrent_state_bytes(),
        prepared_replaced_recurrent_bytes
    );
    assert_eq!(
        restored_report.executed_native_item_count(),
        stable_accumulation_executed_native_items
            .expect("the scoreboard recorded successful accumulation replays")
    );

    let report_bytes = report.to_json_bytes()?;
    let report_json: serde_json::Value = serde_json::from_slice(&report_bytes)?;
    let compiler_process_count = report_json["prepare_compiler_process_count"]
        .as_u64()
        .expect("current scoreboard reports compiler process count");
    assert_eq!(
        report_json["prepare_compiler_process_timings"]
            .as_array()
            .expect("current scoreboard reports bounded compiler process timings")
            .len(),
        usize::try_from(compiler_process_count)?
    );
    for timing in report_json["prepare_compiler_process_timings"]
        .as_array()
        .expect("current scoreboard reports bounded compiler process timings")
    {
        let source_bytes = timing["rendered_source_bytes"]
            .as_u64()
            .expect("current scoreboard reports translation-unit source bytes");
        assert!((source_bytes == 0) == (timing["process"]["kind"] == "link"));
    }
    let overlaps = report_json["prepare_module_overlaps"]
        .as_array()
        .expect("current scoreboard reports ordered main-program overlaps");
    assert!(
        report_json["main"]["rendered_source_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0)
    );
    assert_eq!(report_json["main"]["shared_prefix_source_bytes"], 0);
    assert_eq!(overlaps.len(), 3);
    for (index, overlap) in overlaps.iter().enumerate() {
        assert_eq!(
            overlap["program_index"],
            u64::try_from(
                index
                    .checked_add(1)
                    .expect("program ordinal remains bounded")
            )?
        );
        for field in [
            "evidence_identity",
            "contiguous_prefix_entry_count",
            "contiguous_prefix_source_bytes",
            "additional_scattered_entry_count",
            "additional_scattered_source_bytes",
        ] {
            assert!(overlap[field].is_u64());
        }
    }
    assert_eq!(
        overlaps[0]["contiguous_prefix_entry_count"],
        u64::try_from(EXPECTED_SHARED_ACCUMULATION_PREFIX)?
    );
    assert!(
        overlaps[0]["contiguous_prefix_source_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0)
    );
    let pair_overlaps = report_json["prepare_program_pair_overlaps"]
        .as_array()
        .expect("current scoreboard reports every ordered earlier/later program pair");
    assert_eq!(pair_overlaps.len(), 6);
    assert_eq!(
        pair_overlaps
            .iter()
            .map(|overlap| (
                overlap["source_program_index"].as_u64().unwrap(),
                overlap["target_program_index"].as_u64().unwrap(),
            ))
            .collect::<Vec<_>>(),
        [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    );
    for (overlap, main_overlap) in pair_overlaps
        .iter()
        .filter(|overlap| overlap["source_program_index"] == 0)
        .zip(overlaps)
    {
        assert_eq!(
            overlap["contiguous_prefix_entry_count"],
            main_overlap["contiguous_prefix_entry_count"]
        );
        assert_eq!(
            overlap["additional_scattered_entry_count"],
            main_overlap["additional_scattered_entry_count"]
        );
    }
    let translation_units = report_json["prepare_translation_units"]
        .as_array()
        .expect("current scoreboard reports the exact translation-unit build plan");
    assert_eq!(translation_units.len(), 5);
    for unit in translation_units {
        assert!(unit["translation_unit_identity"].is_u64());
        assert!(unit["evidence_identity"].is_u64());
        assert!(unit["entry_count"].as_u64().is_some_and(|count| count > 0));
        assert!(
            unit["rendered_source_bytes"]
                .as_u64()
                .is_some_and(|bytes| bytes > 0)
        );
    }
    assert_eq!(
        report_json["prepare_compiler_critical_tail"].is_object(),
        compiler_process_count != 0
    );
    print!("{}", String::from_utf8(report_bytes)?);
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
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

#[cfg(test)]
mod cross_process_evidence_tests {
    use super::*;

    fn checkpoint(replay_step: u64) -> CrossProcessCheckpointEvidence {
        CrossProcessCheckpointEvidence {
            capture_identity: 11,
            replay_step,
            optimizer_step: 1,
            gradient_accumulation_steps: 3,
            accumulation_index: 1,
            discarded_microbatches: 0,
            flushed_window_count: 0,
            flushed_microbatch_count: 0,
            flush_capture_identity: Some(12),
            dropout_block_counter: Some(48),
            accumulated_token_count: Some(5),
            accumulated_loss_numerator_bits: Some(1.25_f32.to_bits()),
            window_loss_report_enabled: true,
            reset_transition_count: 0,
            reset_capture_identity: Some(13),
            accumulation_capture_identity: Some(14),
        }
    }

    fn evidence() -> CrossProcessResumeEvidence {
        CrossProcessResumeEvidence {
            format_version: CROSS_PROCESS_EVIDENCE_VERSION,
            backend: CrossProcessBackend::Cpu,
            pending_bundle_bytes: 3,
            pending_bundle_checksum: cross_process_checksum(b"one"),
            terminal_checkpoint_bytes: 3,
            terminal_checkpoint_checksum: cross_process_checksum(b"two"),
            evaluation_capture_identity: 15,
            initial_evaluation_loss_bits: 2.0_f64.to_bits(),
            terminal_evaluation_loss_bits: 1.0_f64.to_bits(),
            pending: checkpoint(4),
            terminal: checkpoint(11),
        }
    }

    #[test]
    fn cross_process_evidence_round_trips_canonically() {
        let expected = evidence();
        let bytes = expected.canonical_bytes().unwrap();
        assert_eq!(
            CrossProcessResumeEvidence::from_canonical_bytes(&bytes).unwrap(),
            expected
        );
    }

    #[test]
    fn cross_process_evidence_rejects_unknown_version_and_fields() {
        let mut unsupported = evidence();
        unsupported.format_version += 1;
        let bytes = unsupported.canonical_bytes().unwrap();
        assert!(CrossProcessResumeEvidence::from_canonical_bytes(&bytes).is_err());

        let mut value = serde_json::to_value(evidence()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), serde_json::Value::Bool(true));
        let mut bytes = serde_json::to_vec_pretty(&value).unwrap();
        bytes.push(b'\n');
        assert!(CrossProcessResumeEvidence::from_canonical_bytes(&bytes).is_err());
    }

    #[test]
    fn cross_process_evidence_rejects_backend_mismatch() {
        assert!(
            evidence()
                .validate_backend(CrossProcessBackend::NativeCpu)
                .is_err()
        );
    }

    #[test]
    fn cross_process_evidence_rejects_checksum_mismatch() {
        let bytes = b"one";
        assert!(
            validate_cross_process_file_binding(
                "pending bundle",
                bytes,
                u64::try_from(bytes.len()).unwrap(),
                cross_process_checksum(bytes) ^ 1,
            )
            .is_err()
        );
    }
}
