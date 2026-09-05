use rustgrad::nn::{Embedding, LayerNorm, StateKind};
use rustgrad::runtime::metal::{MetalCapabilities, MetalRenderer};
use rustgrad::{
    CompiledAdamWConfig, CpuCompiledAdamW, DType, Graph, LossOptions, Mode, ModeModuleForward,
    Module, NodeId, Parameter, Reduction, Result, Scalar, Shape, TensorData, TransformerBlock,
    cross_entropy,
};
use std::collections::BTreeMap;

const VOCAB: usize = 3;
const EMBEDDING: usize = 2;
const TIME: usize = 3;

struct TinyCausalTransformer {
    tokens: Embedding,
    block: TransformerBlock,
    norm: LayerNorm,
}

impl TinyCausalTransformer {
    fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            tokens: Embedding::new_static(VOCAB, EMBEDDING, None, seed)?,
            block: TransformerBlock::new_static(EMBEDDING, 1, 4, true, 0.0, seed.wrapping_add(1))?
                .with_causal_attention(true),
            norm: LayerNorm::new_static([EMBEDDING], 1e-5, true)?,
        })
    }

    fn forward(&self, graph: &mut Graph, tokens: NodeId) -> Result<NodeId> {
        let hidden = self.tokens.forward(graph, tokens)?;
        let hidden = self.block.forward_mode(graph, hidden, Mode::Eval)?.output;
        let hidden = self.norm.forward(graph, hidden)?;
        let tied_weight = self.tokens.weight.bind(graph)?;
        let tied_weight = graph.permute(tied_weight, [1, 0])?;
        graph.matmul(hidden, tied_weight)
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
            child("lm_head.weight"),
            &self.tokens.weight,
            StateKind::Parameter,
        );
    }
}

fn config() -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_input("tokens", [1, TIME], DType::I32)
        .unwrap()
        .with_input("targets", [1, TIME], DType::I32)
        .unwrap()
}

fn build(
    model: &TinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let logits = model.forward(graph, inputs["tokens"])?;
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

fn batch() -> BTreeMap<String, TensorData> {
    let tensor = |values: [i32; TIME]| {
        TensorData::from_scalars(
            Shape::new([1, TIME]),
            DType::I32,
            values.into_iter().map(|value| Scalar::I(i64::from(value))),
        )
        .unwrap()
    };
    BTreeMap::from([
        ("tokens".into(), tensor([0, 1, 2])),
        ("targets".into(), tensor([1, 2, 0])),
    ])
}

fn learning_rate() -> TensorData {
    TensorData::scalar(0.05)
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

#[test]
fn compiled_causal_transformer_training_decreases_loss_and_resumes_exactly() {
    let model = TinyCausalTransformer::new(7).unwrap();
    assert!(model.block.is_causal());
    let mut uninterrupted = CpuCompiledAdamW::compile_module(config(), &model, build).unwrap();
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

    let mut losses = Vec::new();
    for _ in 0..4 {
        losses.push(
            uninterrupted
                .step(batch(), learning_rate())
                .unwrap()
                .loss()
                .scalar_at(0)
                .as_f64(),
        );
    }

    let checkpoint = uninterrupted.checkpoint().unwrap();
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let mut resumed = CpuCompiledAdamW::compile_module_from_checkpoint(
        config(),
        &resumed_model,
        &checkpoint,
        build,
    )
    .unwrap();

    for _ in 0..4 {
        let expected = uninterrupted.step(batch(), learning_rate()).unwrap();
        let actual = resumed.step(batch(), learning_rate()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        losses.push(expected.loss().scalar_at(0).as_f64());
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "compiled causal Transformer loss did not decrease: {losses:?}"
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
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );
}

#[test]
fn causal_transformer_training_capture_is_strictly_renderable_for_metal() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let compiled = CpuCompiledAdamW::compile_module(config(), &model, build).unwrap();
    let parameter_count = compiled.parameter_snapshots().unwrap().len();
    let plan = compiled.metal_plan(metal_renderer()).unwrap();

    assert_eq!(plan.capture_identity(), compiled.capture_identity());
    assert_eq!(plan.step_count(), 0);
    assert_eq!(plan.summary().fallback_count, 0);
    assert_eq!(plan.summary().state_pair_count, parameter_count * 3 + 1);
    assert_eq!(plan.summary().state_bank_count, 2);
    assert_eq!(plan.summary().requested_output_count, 2);
    assert!(plan.summary().nonzero_item_count > 0);
    assert_eq!(
        plan.rendered_items().len(),
        plan.summary().nonzero_item_count
    );
}
