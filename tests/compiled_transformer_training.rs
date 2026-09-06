#[cfg(target_os = "macos")]
use rustgrad::MetalSessionTarget;
use rustgrad::nn::{Embedding, LayerNorm, StateKind};
use rustgrad::runtime::metal::{MetalCapabilities, MetalRenderer};
#[cfg(target_os = "macos")]
use rustgrad::runtime::metal::{MetalDiscovery, MetalRuntime, MetalScoreboardContext};
use rustgrad::{
    CompiledAdamWCheckpoint, CompiledAdamWConfig, CompiledAdamWPlan, CompiledAdamWRuntime,
    CompiledTrainingStep, CpuSessionTarget, DType, Graph, LossOptions, Mode, ModeModuleForward,
    Module, NodeId, Parameter, Reduction, Result, Scalar, Shape, TensorData, TransformerBlock,
    cross_entropy,
};
use std::collections::BTreeMap;
#[cfg(target_os = "macos")]
use std::{env, fs::OpenOptions, io::Write, path::PathBuf};

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
        .with_loss_scale(128.0)
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

fn compiled_transformer(model: &TinyCausalTransformer) -> CompiledAdamWPlan {
    CompiledAdamWPlan::compile_module(config(), model, build)
        .expect("the fixed causal Transformer training program must compile")
}

fn run_exact_resume<R, P>(mut prepare: P) -> Vec<f64>
where
    R: CompiledAdamWRuntime,
    P: FnMut(&CompiledAdamWPlan) -> Result<R>,
{
    let model = TinyCausalTransformer::new(7).unwrap();
    assert!(model.block.is_causal());
    let plan = compiled_transformer(&model);
    let capture_identity = plan.capture_identity();
    let mut uninterrupted = prepare(&plan).unwrap();
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

    let saved = uninterrupted.checkpoint().unwrap();
    let checkpoint = CompiledAdamWCheckpoint::from_bytes(saved.into_bytes()).unwrap();
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let resumed_plan = CompiledAdamWPlan::compile_module_from_checkpoint(
        config(),
        &resumed_model,
        &checkpoint,
        build,
    )
    .unwrap();
    assert_eq!(resumed_plan.capture_identity(), capture_identity);
    assert_eq!(resumed_plan.step_count(), 4);
    let mut resumed = prepare(&resumed_plan).unwrap();

    for _ in 0..4 {
        let expected = uninterrupted.step(batch(), learning_rate()).unwrap();
        let actual = resumed.step(batch(), learning_rate()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        losses.push(expected.loss().scalar_at(0).as_f64());
    }

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
    losses
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
fn compiled_transformer_plan_cpu_target_decreases_loss_and_resumes_exactly() {
    let target = CpuSessionTarget::new();
    let losses = run_exact_resume(|plan| plan.prepare(&target));

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "compiled causal Transformer loss did not decrease: {losses:?}"
    );
}

#[test]
fn compiled_transformer_plan_is_strictly_renderable_for_metal() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let compiled = compiled_transformer(&model);
    assert_eq!(compiled.loss_scale(), 128.0);
    let parameter_count = compiled
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .parameter_snapshots()
        .unwrap()
        .len();
    let plan = compiled.metal_plan(metal_renderer()).unwrap();

    assert_eq!(plan.capture_identity(), compiled.capture_identity());
    assert_eq!(plan.loss_scale(), 128.0);
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

#[test]
fn protected_live_metal_workflow_runs_the_exact_compiled_training_acceptance() {
    let workflow = include_str!("../.github/workflows/metal-live.yml");
    for required in [
        "RUSTGRAD_METAL_TRAINING_EVIDENCE_PATH:",
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
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires the protected self-hosted Apple-GPU lane"]
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
    let seed = compiled_transformer(&model);
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
    let rendered = seed
        .metal_plan(target.renderer().clone())
        .expect("the complete training capture must be entirely Metal-admitted");
    assert_eq!(rendered.capture_identity(), capture_identity);
    assert_eq!(rendered.loss_scale(), 128.0);
    assert_eq!(rendered.summary().fallback_count, 0);
    assert!(rendered.summary().nonzero_item_count > 0);
    let deployment_identity = rendered.deployment_identity();
    let state_pair_count = rendered.summary().state_pair_count;
    let planned_kernel_count = rendered.summary().nonzero_item_count;
    let mut uninterrupted = seed
        .prepare(&target)
        .expect("live Metal preparation must compile, allocate, and upload training state");
    assert_eq!(uninterrupted.loss_scale(), 128.0);
    assert_eq!(uninterrupted.step_count(), 0);
    assert_eq!(uninterrupted.optimizer_step().unwrap(), 0);

    let mut losses = Vec::new();
    let mut kernel_launch_count = 0usize;
    let mut command_submission_count = 0usize;
    let mut transient_h2d_bytes = 0usize;
    let mut retained_d2h_bytes = 0usize;
    for index in 0..4u64 {
        let result = uninterrupted.step(batch(), learning_rate()).unwrap();
        assert_eq!(result.step(), index + 1);
        assert_eq!(result.capture_identity(), capture_identity);
        assert_eq!(result.report().successful_invocation, index + 1);
        assert_eq!(result.report().committed_state_pair_count, state_pair_count);
        assert_eq!(result.report().kernel_launch_count, planned_kernel_count);
        losses.push(result.loss().scalar_at(0).as_f64());
        kernel_launch_count += result.report().kernel_launch_count;
        command_submission_count += result.report().command_submission_count;
        transient_h2d_bytes += result.report().transient_h2d_bytes;
        retained_d2h_bytes += result.report().retained_d2h_bytes;
    }

    let checkpoint = uninterrupted.checkpoint().unwrap();
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let resumed_seed = CompiledAdamWPlan::compile_module_from_checkpoint(
        config(),
        &resumed_model,
        &checkpoint,
        build,
    )
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
    let resumed_rendered = resumed_seed
        .metal_plan(resumed_target.renderer().clone())
        .expect("the checkpoint-restored capture must remain entirely Metal-admitted");
    assert_eq!(resumed_rendered.capture_identity(), capture_identity);
    let resumed_deployment_identity = resumed_rendered.deployment_identity();
    assert_ne!(
        resumed_deployment_identity, deployment_identity,
        "the deployment identity must authenticate the checkpoint-restored state bytes"
    );
    assert_eq!(resumed_rendered.summary().fallback_count, 0);
    let mut resumed = resumed_seed
        .prepare(&resumed_target)
        .expect("checkpoint-restored Metal preparation must succeed");
    assert_eq!(resumed.loss_scale(), 128.0);
    assert_eq!(resumed.step_count(), 4);
    assert_eq!(resumed.optimizer_step().unwrap(), 4);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);

    for resumed_index in 0..4u64 {
        let expected = uninterrupted.step(batch(), learning_rate()).unwrap();
        let actual = resumed.step(batch(), learning_rate()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.report().successful_invocation, resumed_index + 1);
        assert_eq!(actual.report().committed_state_pair_count, state_pair_count);
        assert_eq!(actual.report().kernel_launch_count, planned_kernel_count);
        losses.push(expected.loss().scalar_at(0).as_f64());
        kernel_launch_count += expected.report().kernel_launch_count;
        kernel_launch_count += actual.report().kernel_launch_count;
        command_submission_count += expected.report().command_submission_count;
        command_submission_count += actual.report().command_submission_count;
        transient_h2d_bytes += expected.report().transient_h2d_bytes;
        transient_h2d_bytes += actual.report().transient_h2d_bytes;
        retained_d2h_bytes += expected.report().retained_d2h_bytes;
        retained_d2h_bytes += actual.report().retained_d2h_bytes;
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "live compiled causal Transformer loss did not decrease: {losses:?}"
    );
    assert_eq!(resumed.step_count(), 8);
    assert_eq!(uninterrupted.step_count(), 8);
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );
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

    let evidence = serde_json::json!({
        "format_version": 1,
        "workload": "tiny-causal-transformer-compiled-adamw",
        "implementation_revision": expected_sha,
        "device": {
            "name": device_info.name,
            "registry_id": device_info.registry_id,
            "family": device_info.capabilities.family,
            "unified_memory": device_info.capabilities.unified_memory,
        },
        "capture_identity": capture_identity,
        "loss_scale": uninterrupted.loss_scale(),
        "initial_deployment_identity": deployment_identity,
        "resumed_deployment_identity": resumed_deployment_identity,
        "fallback_count": 0,
        "state_pair_count": state_pair_count,
        "planned_kernel_count": planned_kernel_count,
        "primary_training_steps": 8,
        "resume_replay_steps": 4,
        "total_device_invocations": 12,
        "checkpoint_resume_step": 4,
        "checkpoint_resume_exact": true,
        "initial_loss": losses.first().unwrap(),
        "final_loss": losses.last().unwrap(),
        "kernel_launch_count": kernel_launch_count,
        "command_submission_count": command_submission_count,
        "transient_host_api_h2d_bytes": transient_h2d_bytes,
        "retained_host_api_d2h_bytes": retained_d2h_bytes,
        "initial_scoreboard": initial_scoreboard,
        "resumed_scoreboard": resumed_scoreboard,
    });
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
