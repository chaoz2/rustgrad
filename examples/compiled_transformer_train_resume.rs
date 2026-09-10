//! Compile, train, checkpoint, and authentically recompile a fresh owned tiny
//! Transformer for portable resume on CPU or strict Metal.
//!
//! Same-process callers may instead retain one `CompiledAdamWPlan` and call
//! `restore_checkpoint` without rebuilding its graph or captures.
//!
//! Run that compile-once, same-process CPU path:
//!
//! ```text
//! cargo run --example compiled_transformer_train_resume -- cpu-reuse
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
    CompiledAdamWFlush, CompiledAdamWFlushRuntime, CompiledAdamWPlan, CompiledAdamWRuntime,
    CompiledAdamWStep, CompiledCheckpointRuntime, CompiledDropoutConfig, CompiledDropoutKey,
    CompiledEvaluation, CompiledEvaluationRuntime, CompiledInputBatch, CompiledInputSpec,
    CompiledModuleAdamWPlan, CompiledModuleAdamWSession, CompiledMultiStepLr,
    CompiledTrainingRuntime, CompiledTrainingStep, CpuBackend, CpuNonFinitePolicy,
    CpuSessionTarget, DType, Graph, MetalSessionTarget, Module, NativeCpuSessionTarget,
    NativeTrainingScoreboard, NodeId, Parameter, Result, Scalar, Shape, TensorData,
    TrainingDropoutProvider, TransformerBlock,
};
use std::{cell::Cell, collections::BTreeMap, env, error::Error, time::Instant};

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

fn config() -> Result<CompiledAdamWConfig> {
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_weight_decay_exclusions(WEIGHT_DECAY_EXCLUSIONS)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(MAX_GRADIENT_NORM)?
        .with_input_batch::<TransformerBatch>()
}

fn reuse_config(schedule: CompiledMultiStepLr) -> Result<CompiledAdamWConfig> {
    Ok(config()?
        .with_frozen_parameters([POLICY_FROZEN])?
        .with_captured_multi_step_lr(schedule))
}

fn dropout_config() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([0x1234_5678, 0x9abc_def0]))
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
    let logits = model
        .transformer
        .forward(graph, inputs[TransformerBatch::TOKENS], dropout)?;
    let loss = sparse_causal_loss(graph, logits, inputs[TransformerBatch::TARGETS])?;
    Ok((loss, BTreeMap::from([("logits".into(), logits)])))
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

fn batch(replay: u64) -> Result<TransformerBatch> {
    let tensor = |values: [i32; TOKEN_COUNT]| {
        TensorData::from_scalars(
            Shape::new([BATCH, TIME]),
            DType::I32,
            values.into_iter().map(|value| Scalar::I(i64::from(value))),
        )
    };
    let (tokens, targets) = match (replay - 1) % ACCUMULATION_STEPS {
        0 => ([0, 1, 2, 2, 0, 1], [1, 2, 0, 0, 1, 2]),
        1 => ([1, 2, 0, 0, 1, 2], [2, 0, 1, 1, 2, 0]),
        2 => ([2, 0, 1, 1, 2, 0], [0, 1, 2, 2, 0, 1]),
        _ => unreachable!(),
    };
    Ok(TransformerBatch {
        tokens: tensor(tokens)?,
        targets: tensor(targets)?,
    })
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
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&source.transformer)?;
    let source_policy_frozen = source
        .trainable_parameters()?
        .into_iter()
        .find_map(|(name, parameter)| (name == POLICY_FROZEN).then_some(parameter))
        .expect("the maintained Transformer exposes the policy-frozen parameter")
        .value()?;
    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1])?;
    let builds = Cell::new(0);
    let plan = CompiledAdamWPlan::compile_module_with_dropout(
        reuse_config(schedule.clone())?,
        dropout_config(),
        &source,
        |model, graph, inputs, dropout| {
            builds.set(builds.get() + 1);
            build_buffered(model, graph, inputs, dropout)
        },
    )?;
    assert_eq!(builds.get(), 1, "the training graph must compile once");
    let capture_identity = plan.capture_identity();
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
        let step = uninterrupted.step_batch_scheduled(batch(replay)?)?;
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
    }
    assert_eq!(uninterrupted.optimizer_step()?, 1);
    assert_eq!(uninterrupted.accumulation_index()?, 1);
    let checkpoint = uninterrupted.checkpoint()?;
    assert_eq!(checkpoint.info().replay_step(), 4);
    assert_eq!(checkpoint.info().optimizer_step(), 1);
    assert_eq!(checkpoint.info().accumulation_index(), 1);
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
    assert!(resumed.step_batch(batch(5)?, 0.05).is_err());
    assert_eq!(resumed.checkpoint()?, before_wrong_entrypoint);
    for replay in 5..=6 {
        let expected = uninterrupted.step_batch_scheduled(batch(replay)?)?;
        let actual = resumed.step_batch_scheduled(batch(replay)?)?;
        assert_eq!(actual.loss(), expected.loss());
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

    let final_mean_sparse_loss = evaluate_mean_sparse_loss(&destination.transformer)?;
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

fn run_native_cpu_scoreboard() -> std::result::Result<(), Box<dyn Error>> {
    const SAMPLES: u64 = 6;

    let source = BufferedTinyCausalTransformer::new(7)?;
    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1])?;
    let builds = Cell::new(0);
    let compile_started = Instant::now();
    let plan = CompiledAdamWPlan::compile_module_with_dropout(
        reuse_config(schedule)?,
        dropout_config(),
        &source,
        |model, graph, inputs, dropout| {
            builds.set(builds.get() + 1);
            build_buffered(model, graph, inputs, dropout)
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
    let mut scoreboard = NativeTrainingScoreboard::new(
        inspection.clone(),
        session.preparation_report(),
        compile_wall_time,
        prepare_wall_time,
    )?;
    for replay in 1..=SAMPLES {
        let step = session.step_batch_scheduled(batch(replay)?)?;
        scoreboard.record(step.report())?;
    }

    let checkpoint_started = Instant::now();
    let checkpoint = session.checkpoint()?;
    let checkpoint_wall_time = checkpoint_started.elapsed();
    scoreboard.observe_checkpoint(&checkpoint, checkpoint_wall_time)?;
    let report = scoreboard.report()?;

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
    let restored_session = restored.prepare(&target)?;
    assert_eq!(
        restored_session
            .preparation_report()
            .main()
            .cache_miss_count(),
        0
    );

    print!("{}", String::from_utf8(report.to_json_bytes()?)?);
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    match env::args().nth(1).as_deref().unwrap_or("cpu") {
        "native-cpu-scoreboard" => run_native_cpu_scoreboard()?,
        "cpu-reuse" => run_cpu_reuse()?,
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
                "unknown target {other:?}; expected `native-cpu-scoreboard`, `cpu-reuse`, `cpu`, `native-cpu`, or `metal`"
            )
            .into());
        }
    }
    Ok(())
}
