//! Compile a tiny causal Transformer once, then train and resume on CPU or strict Metal.
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
    Backend, CompiledAdamWCheckpoint, CompiledAdamWConfig, CompiledAdamWPlan, CompiledAdamWRuntime,
    CompiledDropoutConfig, CompiledDropoutKey, CompiledTrainingStep, CpuBackend, CpuSessionTarget,
    DType, Graph, LossOptions, MetalSessionTarget, Module, NodeId, Parameter, Reduction, Result,
    Scalar, Shape, TensorData, TrainingDropoutProvider, TransformerBlock, cross_entropy,
};
use std::{collections::BTreeMap, env, error::Error};

const VOCAB: usize = 3;
const EMBEDDING: usize = 2;
const TIME: usize = 3;
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
const RESUMED_STEPS: usize = 4;

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
        .with_input("tokens", [1, TIME], DType::I32)?
        .with_input("targets", [1, TIME], DType::I32)
}

fn dropout_config() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([0x1234_5678, 0x9abc_def0]))
}

fn build(
    model: &TinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let logits = model.forward(graph, inputs["tokens"], dropout)?;
    let flat_logits = graph.reshape(logits, [TIME, VOCAB])?;
    let flat_targets = graph.reshape(inputs["targets"], [TIME])?;
    let loss = cross_entropy(
        graph,
        flat_logits,
        flat_targets,
        LossOptions {
            reduction: Reduction::Mean,
            ..LossOptions::default()
        },
    )?;
    Ok((loss, BTreeMap::from([("logits".into(), logits)])))
}

fn compile(model: &TinyCausalTransformer) -> Result<CompiledAdamWPlan> {
    CompiledAdamWPlan::compile_module_with_dropout(config()?, dropout_config(), model, build)
}

fn batch() -> Result<BTreeMap<String, TensorData>> {
    let tensor = |values: [i32; TIME]| {
        TensorData::from_scalars(
            Shape::new([1, TIME]),
            DType::I32,
            values.into_iter().map(|value| Scalar::I(i64::from(value))),
        )
    };
    Ok(BTreeMap::from([
        ("tokens".into(), tensor([0, 1, 2])?),
        ("targets".into(), tensor([1, 2, 0])?),
    ]))
}

fn evaluate(model: &TinyCausalTransformer) -> Result<TensorData> {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [1, TIME], DType::I32);
    let logits = model.forward_eval(&mut graph, tokens)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert("tokens".into(), batch()?.remove("tokens").unwrap());
    CpuBackend.execute(&graph, logits, &bindings)
}

fn run_exact_resume<R, P>(target_name: &str, mut prepare: P) -> Result<()>
where
    R: CompiledAdamWRuntime,
    P: FnMut(&CompiledAdamWPlan) -> Result<R>,
{
    let model = TinyCausalTransformer::new(7)?;
    let plan = compile(&model)?;
    let capture_identity = plan.capture_identity();
    let mut uninterrupted = prepare(&plan)?;
    let parameters = uninterrupted.parameter_snapshots()?;
    assert!(parameters.contains_key("tokens.weight"));
    assert!(
        !parameters.contains_key("lm_head.weight"),
        "the tied output head must share the embedding's recurrent state"
    );

    let mut losses = Vec::with_capacity(INITIAL_STEPS + RESUMED_STEPS);
    for _ in 0..INITIAL_STEPS {
        let step = uninterrupted.step(batch()?, TensorData::scalar(0.05))?;
        losses.push(step.loss().scalar_at(0).as_f64());
    }

    let saved = uninterrupted.checkpoint()?;
    let checkpoint = CompiledAdamWCheckpoint::from_bytes(saved.into_bytes())?;
    let restored_model = TinyCausalTransformer::new(7)?;
    let restored_plan = CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
        config()?,
        dropout_config(),
        &restored_model,
        &checkpoint,
        build,
    )?;
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(restored_plan.step_count(), INITIAL_STEPS as u64);
    let mut resumed = prepare(&restored_plan)?;

    for _ in 0..RESUMED_STEPS {
        let expected = uninterrupted.step(batch()?, TensorData::scalar(0.05))?;
        let actual = resumed.step(batch()?, TensorData::scalar(0.05))?;
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        losses.push(expected.loss().scalar_at(0).as_f64());
    }

    assert!(
        losses.last() < losses.first(),
        "compiled causal Transformer loss did not decrease: {losses:?}"
    );
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
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let published = resumed.parameter_snapshots()?;
    let tied_version = restored_model.tokens.weight.version()?;
    let frozen_before = restored_model.frozen_scale.snapshot()?;
    let runtime_checkpoint = resumed.checkpoint()?;
    let runtime_step = resumed.step_count();
    assert!(resumed.publish_parameters(&restored_model)?.is_clean());
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
    assert_eq!(resumed.step_count(), runtime_step);
    assert_eq!(resumed.checkpoint()?, runtime_checkpoint);
    let versions_before_eval = restored_model
        .trainable_parameters()?
        .into_iter()
        .map(|(name, parameter)| Ok((name, parameter.version()?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let first_eval = evaluate(&restored_model)?;
    let second_eval = evaluate(&restored_model)?;
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel()?)
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in restored_model.trainable_parameters()? {
        assert_eq!(parameter.version()?, versions_before_eval[&name]);
    }
    println!(
        "{target_name}: capture={capture_identity:016x}, steps={}, loss={:.6} -> {:.6}, exact_resume=true, published=true",
        resumed.step_count(),
        losses[0],
        losses.last().expect("eight losses were recorded")
    );
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    match env::args().nth(1).as_deref().unwrap_or("cpu") {
        "cpu" => {
            let target = CpuSessionTarget::new();
            run_exact_resume("CPU", |plan| plan.prepare(&target))?;
        }
        "metal" => {
            let device = MetalRuntime::load()?.device(0)?;
            let target = MetalSessionTarget::new(device, 64)?;
            run_exact_resume("Metal", |plan| {
                let rendered = plan.metal_plan(target.renderer().clone())?;
                assert_eq!(rendered.summary().fallback_count, 0);
                plan.prepare(&target)
            })?;
        }
        other => {
            return Err(format!("unknown target {other:?}; expected `cpu` or `metal`").into());
        }
    }
    Ok(())
}
