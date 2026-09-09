//! Compile a fixed two-row tiny causal Transformer once, then train and resume on CPU or strict Metal.
//!
//! Run on the graph-free CPU replay target:
//!
//! ```text
//! cargo run --example compiled_transformer_train_resume -- cpu
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
    Backend, CompiledAdamWCheckpoint, CompiledAdamWConfig, CompiledAdamWFlush,
    CompiledAdamWFlushRuntime, CompiledAdamWRuntime, CompiledAdamWStep, CompiledCheckpointRuntime,
    CompiledDropoutConfig, CompiledDropoutKey, CompiledEvaluation, CompiledEvaluationRuntime,
    CompiledInputBatch, CompiledInputSpec, CompiledModuleAdamWPlan, CompiledModuleAdamWSession,
    CompiledTrainingRuntime, CompiledTrainingStep, CpuBackend, CpuSessionTarget, DType, Graph,
    MetalSessionTarget, Module, NodeId, Parameter, Result, Scalar, Shape, TensorData,
    TrainingDropoutProvider, TransformerBlock,
};
use std::{collections::BTreeMap, env, error::Error};

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

fn config() -> Result<CompiledAdamWConfig> {
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_weight_decay_exclusions(WEIGHT_DECAY_EXCLUSIONS)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(MAX_GRADIENT_NORM)?
        .with_input_batch::<TransformerBatch>()
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

    for replay in (INITIAL_STEPS as u64 + 1)..=(INITIAL_STEPS + RESUMED_STEPS) as u64 {
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
    let published = resumed.parameter_snapshots()?;
    let tied_version = tied.version()?;
    let frozen_before = frozen.snapshot()?;
    let runtime_step = resumed.step_count();
    let _uninterrupted_model = uninterrupted
        .finish()
        .map_err(|error| error.into_parts().1)?;
    let restored_model = resumed.finish().map_err(|error| error.into_parts().1)?;
    let live = restored_model.state_dict()?;
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert_eq!(
        restored_model.tokens.weight.value()?,
        live.tensors()["tokens.weight"]
    );
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert!(!published.contains_key("lm_head.weight"));
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

fn main() -> std::result::Result<(), Box<dyn Error>> {
    match env::args().nth(1).as_deref().unwrap_or("cpu") {
        "cpu" => {
            let target = CpuSessionTarget::new();
            run_exact_resume("CPU", |plan| {
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
            return Err(format!("unknown target {other:?}; expected `cpu` or `metal`").into());
        }
    }
    Ok(())
}
