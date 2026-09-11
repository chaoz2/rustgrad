//! Compile, train, checkpoint, and authentically recompile a fresh owned tiny
//! Transformer for portable resume on CPU or strict Metal.
//!
//! Same-process callers may instead retain one `CompiledAdamWPlan` and call
//! `restore_checkpoint` without rebuilding its graph or captures. That CPU
//! path uses fixed-capacity right-padded batches; compilation derives the
//! masked token-mean loss and weights its gradient by each valid-token count
//! across an accumulation window.
//!
//! Run that compile-once, same-process CPU path:
//!
//! ```text
//! cargo run --example compiled_transformer_train_resume -- cpu-reuse
//! ```
//!
//! Run a complete module checkpoint through a file and recompile a deliberately
//! different initialization before exact CPU continuation:
//!
//! ```text
//! cargo run --example compiled_transformer_train_resume -- cpu-file-resume
//! ```
//!
//! Run that complete-module file-resume lifecycle through strict native CPU JIT
//! replay with no fallback:
//!
//! ```text
//! cargo run --release --example compiled_transformer_train_resume -- native-cpu-file-resume
//! ```
//!
//! Emit a bounded strict-native CPU training scoreboard from that same
//! compile-once path:
//!
//! ```text
//! cargo run --release --example compiled_transformer_train_resume -- native-cpu-scoreboard
//! ```
//!
//! Run on the graph-free CPU replay target:
//!
//! ```text
//! cargo run --example compiled_transformer_train_resume -- cpu
//! ```
//!
//! Run the identical capture through strict native CPU JIT replay:
//!
//! ```text
//! cargo run --release --example compiled_transformer_train_resume -- native-cpu
//! ```
//!
//! Run the identical capture on the first visible Metal device, with no CPU fallback:
//!
//! ```text
//! cargo run --release --example compiled_transformer_train_resume -- metal
//! ```

use rustgrad::nn::{Embedding, LayerNorm, Mode, ModeModuleForward, StateKind};
use rustgrad::runtime::metal::MetalRuntime;
use rustgrad::{
    Backend, CapturedReplayExecutor, CompiledAdamWCheckpoint, CompiledAdamWConfig,
    CompiledAdamWFlush, CompiledAdamWFlushRuntime, CompiledAdamWGraph, CompiledAdamWPlan,
    CompiledAdamWRuntime, CompiledAdamWStep, CompiledCheckpointRuntime, CompiledDropoutConfig,
    CompiledDropoutKey, CompiledEvaluation, CompiledEvaluationRuntime, CompiledInputBatch,
    CompiledInputSpec, CompiledModuleAdamWCheckpoint, CompiledModuleAdamWPlan,
    CompiledModuleAdamWSession, CompiledMultiStepLr, CompiledScheduledAdamWRuntime,
    CompiledTrainingRuntime, CompiledTrainingStep, CpuBackend, CpuCompiledAdamW,
    CpuNonFinitePolicy, CpuSessionTarget, DType, Graph, MetalSessionTarget, Module,
    NativeCpuCompiledAdamW, NativeCpuCompiledAdamWStepResult, NativeCpuCompiledEvaluationResult,
    NativeCpuSessionTarget, NativeTrainingScoreboard, NodeId, Parameter, Result, Scalar, Shape,
    TensorData, TrainingDropoutProvider, TransformerBlock,
};
use std::{
    cell::Cell,
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::PathBuf,
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
const ATTENTION_KEEP_MASK: &str = "attention_keep_mask";
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
    two_block_config::<FileResumeBatch>(schedule)
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
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    let logits = model.forward(
        graph,
        inputs[MaskedTransformerBatch::TOKENS],
        inputs[ATTENTION_KEEP_MASK],
        dropout,
    )?;
    let losses = sparse_causal_losses(graph, logits, inputs[MaskedTransformerBatch::TARGETS])?;
    Ok(CompiledAdamWGraph::token_mean(
        losses,
        BTreeMap::from([("logits".into(), logits)]),
    ))
}

fn build_file_resume_evaluation(
    model: &FileResumeTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<CompiledAdamWGraph> {
    let logits = model.forward_eval(
        graph,
        inputs[MaskedTransformerBatch::TOKENS],
        inputs[ATTENTION_KEEP_MASK],
    )?;
    let losses = sparse_causal_losses(graph, logits, inputs[MaskedTransformerBatch::TARGETS])?;
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
    attention_keep_mask: TensorData,
}

impl FileResumeBatch {
    const SCHEMA: [CompiledInputSpec; 4] = [
        CompiledInputSpec::new(ATTENTION_KEEP_MASK, &ATTENTION_KEEP_MASK_SHAPE, DType::Bool),
        CompiledInputSpec::new(LOSS_MASK, &[BATCH, TIME], DType::F32),
        CompiledInputSpec::host_token(MaskedTransformerBatch::TARGETS, &[BATCH, TIME]),
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
        let attention_keep_mask = TensorData::from_scalars(
            ATTENTION_KEEP_MASK_SHAPE,
            DType::Bool,
            validity.iter().map(|value| Scalar::Bool(*value == 1.0)),
        )?;
        let batch = Self {
            masked,
            attention_keep_mask,
        };
        batch.assert_attention_keep_mask();
        Ok(batch)
    }

    fn assert_attention_keep_mask(&self) {
        assert_eq!(
            self.attention_keep_mask.shape(),
            &Shape::new(ATTENTION_KEEP_MASK_SHAPE)
        );
        assert_eq!(self.attention_keep_mask.dtype(), DType::Bool);
        assert_eq!(
            self.attention_keep_mask.to_vec_f64(),
            self.masked.loss_mask.to_vec_f64()
        );
    }

    fn has_fully_masked_sample(&self) -> bool {
        self.attention_keep_mask
            .to_vec_f64()
            .chunks_exact(TIME)
            .any(|row| row.iter().all(|value| *value == 0.0))
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
        let mut inputs = self.masked.into_compiled_inputs()?;
        assert!(
            inputs
                .insert(ATTENTION_KEEP_MASK.into(), self.attention_keep_mask)
                .is_none()
        );
        Ok(inputs)
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
    staged: PathBuf,
}

impl TemporaryCheckpointFile {
    fn new() -> std::result::Result<Self, Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "rustgrad-compiled-module-resume-{}-{nonce}.safetensors",
            std::process::id()
        ));
        let staged = path.with_extension("safetensors.tmp");
        Ok(Self { path, staged })
    }

    fn write_then_read(&self, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        fs::write(&self.staged, bytes)?;
        fs::rename(&self.staged, &self.path)?;
        fs::read(&self.path)
    }
}

impl Drop for TemporaryCheckpointFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.staged);
        let _ = fs::remove_file(&self.path);
    }
}

fn run_exact_resume<R, P>(target_name: &str, mut prepare: P) -> Result<()>
where
    R: CompiledAdamWRuntime + CompiledAdamWFlushRuntime + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<TinyCausalTransformer>,
    ) -> Result<CompiledModuleAdamWSession<TinyCausalTransformer, R>>,
{
    let model = TinyCausalTransformer::new(7)?;
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&model)?;
    let plan = compile(model)?;
    let capture_identity = plan.capture_identity();
    let mut uninterrupted = prepare(plan)?;
    let evaluation_identity = uninterrupted
        .evaluation_capture_identity()
        .expect("evaluation was attached before preparation");
    let parameters = uninterrupted.parameter_snapshots()?;
    assert!(parameters.contains_key("tokens.weight"));
    assert!(
        !parameters.contains_key("lm_head.weight"),
        "the tied output head must share the embedding's recurrent state"
    );
    assert_eq!(
        uninterrupted.gradient_accumulation_steps(),
        ACCUMULATION_STEPS
    );
    assert_eq!(uninterrupted.max_gradient_norm(), Some(MAX_GRADIENT_NORM));
    let initial_parameters = uninterrupted.parameter_snapshots()?;
    let initial_first_moments = uninterrupted.first_moment_snapshots()?;
    let initial_second_moments = uninterrupted.second_moment_snapshots()?;
    let empty_accumulators = uninterrupted.gradient_accumulator_snapshots()?;

    for replay in 1..=2 {
        let step = uninterrupted.step_batch(batch(replay)?, 0.05)?;
        assert_eq!(step.optimizer_step(), 0);
        assert_eq!(step.accumulation_index(), replay);
        assert!(!step.did_update());
    }
    assert_ne!(
        uninterrupted.gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    assert_eq!(uninterrupted.parameter_snapshots()?, initial_parameters);
    assert_eq!(
        uninterrupted.first_moment_snapshots()?,
        initial_first_moments
    );
    assert_eq!(
        uninterrupted.second_moment_snapshots()?,
        initial_second_moments
    );
    assert_eq!(uninterrupted.zero_grad()?.discarded_microbatches(), 2);
    assert_eq!(uninterrupted.step_count(), 2);
    assert_eq!(uninterrupted.optimizer_step()?, 0);
    assert_eq!(uninterrupted.accumulation_index()?, 0);
    assert_eq!(
        uninterrupted.gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    let after_reset = uninterrupted.checkpoint()?;
    assert_eq!(uninterrupted.zero_grad()?.discarded_microbatches(), 0);
    assert_eq!(uninterrupted.checkpoint()?, after_reset);

    for replay in 3..=INITIAL_STEPS as u64 {
        let step = uninterrupted.step_batch(batch(replay)?, 0.05)?;
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
    assert_eq!(checkpoint_info.flush_capture_identity(), None);
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
    let mut resumed = prepare(restored_plan)?;
    assert_eq!(resumed.optimizer_step()?, 0);
    assert_eq!(resumed.accumulation_index()?, 2);
    assert_eq!(resumed.checkpoint()?, checkpoint);

    for replay in resumed_first_replay..=resumed_last_replay {
        let expected = uninterrupted.step_batch(batch(replay)?, 0.05)?;
        let actual = resumed.step_batch(batch(replay)?, 0.05)?;
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
    let expected_flush = uninterrupted.flush_partial_window(TensorData::scalar(0.05))?;
    let actual_flush = resumed.flush_partial_window(TensorData::scalar(0.05))?;
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
    assert_eq!(resumed.optimizer_step()?, 2);
    assert_eq!(resumed.accumulation_index()?, 0);

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
    assert_eq!(
        resumed.gradient_accumulator_snapshots()?,
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

fn assert_native_file_resume_preparation(
    session: &CompiledModuleAdamWSession<FileResumeTransformer, NativeCpuCompiledAdamW<'_>>,
) {
    let preparation = session.native_cpu_preparation_report();
    for program in [
        Some(preparation.main()),
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
    run_file_resume::<NativeCpuCompiledAdamW<'_>, _, _, _, _>(
        "native CPU file resume",
        |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
        assert_native_file_resume_preparation,
        assert_native_file_resume_step,
        assert_native_file_resume_evaluation,
    )
}

fn run_file_resume<R, P, V, S, E>(
    target_name: &str,
    mut prepare: P,
    mut validate_preparation: V,
    mut validate_step: S,
    mut validate_evaluation: E,
) -> std::result::Result<(), Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime + CompiledEvaluationRuntime,
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
    assert!(
        file_resume_batch(3)?.has_fully_masked_sample(),
        "the zero-length row must exercise fully masked attention"
    );
    let source = FileResumeTransformer::new(0x5678)?;
    let source_initial = source.state_dict()?;
    let source_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout(
        config.clone(),
        dropout_config(),
        source,
        build_file_resume,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph(build_file_resume_evaluation)
    .map_err(|error| error.into_parts().1)?;
    let capture_identity = source_plan.capture_identity();
    let evaluation_identity = source_plan
        .evaluation_capture_identity()
        .expect("the compiler-owned token-mean evaluator is attached");
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
        let step = uninterrupted.step_batch_scheduled(batch)?;
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
    let checkpoint_file = TemporaryCheckpointFile::new()?;
    let decoded = CompiledModuleAdamWCheckpoint::from_bytes(
        checkpoint_file.write_then_read(checkpoint.as_bytes())?,
    )?;
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

    let restored_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_from_module_checkpoint(
        config,
        dropout_config(),
        destination,
        &decoded,
        build_file_resume,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph(build_file_resume_evaluation)
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

    let mut resumed = prepare(restored_plan)?;
    validate_preparation(&resumed);
    assert_eq!(
        resumed.checkpoint()?,
        decoded.optimizer_checkpoint().clone()
    );
    assert_eq!(
        resumed.checkpoint()?.info().dropout_block_counter(),
        Some(saved_dropout_cursor)
    );
    for replay in 5..=LAST_REPLAY {
        let expected = uninterrupted.step_batch_scheduled(file_resume_batch(replay)?)?;
        let actual = resumed.step_batch_scheduled(file_resume_batch(replay)?)?;
        validate_step(&expected);
        validate_step(&actual);
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
    assert_eq!(resumed.optimizer_step()?, 3);
    assert_eq!(resumed.accumulation_index()?, 0);
    for (_, parameter, _, before) in &destination_states {
        let after = parameter.snapshot()?;
        assert_eq!(after.data, before.data);
        assert_eq!(after.version, before.version);
        assert_eq!(after.identity, before.identity);
        assert_eq!(after.trainable, before.trainable);
    }

    let before_evaluation_checkpoint = resumed.checkpoint()?;
    let before_evaluation_counter = before_evaluation_checkpoint.info().dropout_block_counter();
    let before_evaluation_accumulators = resumed.gradient_accumulator_snapshots()?;
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
        resumed.gradient_accumulator_snapshots()?,
        before_evaluation_accumulators
    );

    let (uninterrupted_model, uninterrupted_checkpoint) = uninterrupted
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    let (resumed_model, resumed_checkpoint) = resumed
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(resumed_checkpoint, uninterrupted_checkpoint);
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
        "{target_name}: capture={capture_identity:016x}, checkpoint=(replay=4, optimizer=1, accumulation=1), optimizer_steps=3, eval_mean_sparse_loss={initial_loss:.6} -> {final_loss:.6}, exact_resume=true, different_init=true, published=true"
    );
    Ok(())
}

fn run_native_cpu_scoreboard() -> std::result::Result<(), Box<dyn Error>> {
    const SAMPLES: u64 = 3;
    const EXPECTED_LOSS_WEIGHTS: [u64; SAMPLES as usize] = [5, 3, 3];
    const EXPECTED_MODULE_DISPATCHES: usize = 38;
    // The module exposes 37 parameter traversal entries. The tied LM head
    // deduplicates with tokens.weight, and positions.weight is policy-frozen.
    const EXPECTED_ADAMW_UPDATE_GROUPS: usize = 35;
    const EXPECTED_MAIN_RENDERED_ENTRIES: usize = 787 - EXPECTED_ADAMW_UPDATE_GROUPS * 3;
    // One accumulator per update group, plus loss numerator, index, and token count.
    const EXPECTED_ZERO_GRAD_ENTRIES: usize = EXPECTED_ADAMW_UPDATE_GROUPS + 3;

    let source = FileResumeTransformer::new(7)?;
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
    assert!(main_preparation.work().compiler_invocation_count() <= 1);
    let partial_preparation = session
        .preparation_report()
        .partial_flush()
        .expect("scoreboard configuration captures partial flush");
    assert_eq!(partial_preparation.native_item_count(), 357);
    assert_eq!(
        partial_preparation.work().rendered_entry_count(),
        357 - EXPECTED_ADAMW_UPDATE_GROUPS * 3
    );
    assert_eq!(
        session
            .preparation_report()
            .zero_grad()
            .expect("scoreboard configuration captures zero grad")
            .native_item_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    assert_eq!(
        session
            .preparation_report()
            .zero_grad()
            .expect("scoreboard configuration captures zero grad")
            .work()
            .rendered_entry_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    assert!(session.preparation_report().compiler_process_count() <= 3);
    assert!(
        session
            .preparation_report()
            .max_parallel_compiler_process_count()
            <= 2
    );
    if env::var_os("RUSTGRAD_REQUIRE_COLD_NATIVE_SCOREBOARD").is_some() {
        assert_eq!(session.preparation_report().compiler_process_count(), 3);
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
    }
    let mut scoreboard = NativeTrainingScoreboard::new(
        inspection.clone(),
        session.preparation_report(),
        compile_wall_time,
        prepare_wall_time,
    )?;
    let recurrent_state_bytes = u64::try_from(inspection.recurrent_state_bytes())?;
    let mut stable_executed_native_items = None;
    for replay in 1..=SAMPLES {
        let batch = masked_batch(replay)?;
        assert_eq!(
            loss_mask_weight(&batch.loss_mask),
            EXPECTED_LOSS_WEIGHTS[(replay - 1) as usize]
        );
        let step = session.step_batch_scheduled(batch)?;
        let report = step.report();
        let executed = report.executed_native_item_count();
        assert!(executed > 0);
        assert!(executed <= report.native_item_count());
        assert_eq!(
            report.module_dispatch_count(),
            EXPECTED_MODULE_DISPATCHES,
            "the fixed workspace must retain its authenticated safe-segment partition"
        );
        assert_eq!(report.module_dispatched_native_item_count(), executed);
        assert!(report.module_dispatch_count() < executed);
        if let Some(expected) = stable_executed_native_items {
            assert_eq!(executed, expected);
        } else {
            stable_executed_native_items = Some(executed);
        }
        scoreboard.record(step.report())?;
    }

    let checkpoint_started = Instant::now();
    let checkpoint = session.checkpoint()?;
    let checkpoint_wall_time = checkpoint_started.elapsed();
    scoreboard.observe_checkpoint(&checkpoint, checkpoint_wall_time)?;
    let report = scoreboard.report()?;
    let executed_native_items = report
        .main_replay_executed_native_item_count()
        .expect("current native CPU scoreboard reports executed JIT items");
    let expected_main_rendered_entries = u64::try_from(EXPECTED_MAIN_RENDERED_ENTRIES)?;
    let expected_executed_native_items = expected_main_rendered_entries
        .checked_sub(1)
        .expect("the main program has one nonexecuted native entry");
    assert_eq!(executed_native_items, expected_executed_native_items);
    assert_eq!(
        report.main().rendered_entry_count(),
        expected_main_rendered_entries
    );
    assert_eq!(report.main().loaded_module_count(), 1);
    assert!(report.main().compiler_invocation_count() <= 1);
    let program_prepare_wall_time = [
        Some(report.main()),
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
    let executor_timing = report
        .main_replay_executor_wall_time()
        .expect("current native CPU scoreboard reports sealed executor timing");
    let overhead_timing = report
        .main_replay_recurrent_overhead_wall_time()
        .expect("current native CPU scoreboard reports recurrent overhead timing");
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
        Some(executed_native_items as usize),
        stable_executed_native_items
    );
    assert!(executed_native_items <= report.main().native_item_count());
    let replay_traffic = report
        .main_replay_traffic()
        .expect("current native CPU scoreboard reports replay traffic");
    assert_eq!(replay_traffic.external_input_import_count(), 0);
    assert_eq!(replay_traffic.external_input_import_bytes(), 0);
    assert_eq!(
        replay_traffic.borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        replay_traffic.borrowed_recurrent_output_bytes(),
        recurrent_state_bytes
    );

    let restored = plan.restore_checkpoint(&checkpoint)?;
    let restored_inspection = restored.inspection()?;
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
    for program in [
        Some(restored_preparation.main()),
        restored_preparation.partial_flush(),
        restored_preparation.zero_grad(),
        restored_preparation.evaluation(),
    ]
    .into_iter()
    .flatten()
    {
        assert_eq!(program.cache_miss_count(), 0);
        assert_eq!(program.cache_hit_count(), program.native_item_count());
        assert!(program.work().rendered_entry_count() <= program.native_item_count());
        assert_eq!(program.work().loaded_module_count(), 1);
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
        report.main().native_item_count()
    );
    assert_eq!(restored_report.traffic().external_input_import_count(), 0);
    assert_eq!(restored_report.traffic().external_input_import_bytes(), 0);
    assert_eq!(
        restored_report.traffic().borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        restored_report.traffic().borrowed_recurrent_output_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        restored_report.executed_native_item_count(),
        stable_executed_native_items.expect("the scoreboard recorded successful replays")
    );

    print!("{}", String::from_utf8(report.to_json_bytes()?)?);
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    match env::args().nth(1).as_deref().unwrap_or("cpu") {
        "native-cpu-scoreboard" => run_native_cpu_scoreboard()?,
        "cpu-reuse" => run_cpu_reuse()?,
        "cpu-file-resume" => run_cpu_file_resume()?,
        "native-cpu-file-resume" => run_native_cpu_file_resume()?,
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
                "unknown target {other:?}; expected `native-cpu-scoreboard`, `cpu-reuse`, `cpu-file-resume`, `native-cpu-file-resume`, `cpu`, `native-cpu`, or `metal`"
            )
            .into());
        }
    }
    Ok(())
}
