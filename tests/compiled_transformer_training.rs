#[cfg(target_os = "macos")]
use rustgrad::CompiledTrainingRuntime;
#[cfg(target_os = "macos")]
use rustgrad::MetalSessionTarget;
use rustgrad::nn::{Embedding, LayerNorm, Mode, ModeModuleForward, StateKind};
use rustgrad::runtime::metal::{MetalCapabilities, MetalRenderer};
#[cfg(target_os = "macos")]
use rustgrad::runtime::metal::{MetalDiscovery, MetalRuntime, MetalScoreboardContext};
use rustgrad::{
    Backend, CompiledAdamWCheckpoint, CompiledAdamWConfig, CompiledAdamWPlan, CompiledAdamWRuntime,
    CompiledDropoutConfig, CompiledDropoutKey, CompiledTrainingStep, CpuBackend, CpuSessionTarget,
    DType, Graph, LossOptions, MetalCompiledAdamWPlan, Module, NodeId, Parameter, Reduction,
    Result, Scalar, Shape, TensorData, TrainingDropoutProvider, TransformerBlock, cross_entropy,
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

fn dropout_config() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([0x1234_5678, 0x9abc_def0]))
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

fn evaluate(model: &TinyCausalTransformer) -> TensorData {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [1, TIME], DType::I32);
    let logits = model.forward_eval(&mut graph, tokens).unwrap();
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.insert("tokens".into(), batch().remove("tokens").unwrap());
    CpuBackend.execute(&graph, logits, &bindings).unwrap()
}

fn compiled_transformer(model: &TinyCausalTransformer) -> CompiledAdamWPlan {
    CompiledAdamWPlan::compile_module_with_dropout(config(), dropout_config(), model, build)
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
    let resumed_plan = CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
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
    let runtime_checkpoint = resumed.checkpoint().unwrap();
    let runtime_progress = (
        resumed.step_count(),
        resumed.optimizer_step().unwrap(),
        resumed.accumulation_index().unwrap(),
    );
    let published = resumed.parameter_snapshots().unwrap();
    let frozen_before = resumed_model.frozen_scale.snapshot().unwrap();
    let before_versions = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert!(
        resumed
            .publish_parameters(&resumed_model)
            .unwrap()
            .is_clean()
    );
    let live = resumed_model.state_dict().unwrap();
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert!(!live.tensors().contains_key("lm_head.weight"));
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
    }
    let frozen_after = resumed_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    assert_eq!(resumed.checkpoint().unwrap(), runtime_checkpoint);
    assert_eq!(
        (
            resumed.step_count(),
            resumed.optimizer_step().unwrap(),
            resumed.accumulation_index().unwrap(),
        ),
        runtime_progress
    );
    let versions_before_eval = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let first_eval = evaluate(&resumed_model);
    let second_eval = evaluate(&resumed_model);
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel().unwrap())
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), versions_before_eval[&name]);
    }
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
    let losses = run_exact_resume(|plan| plan.prepare(&target));

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "compiled causal Transformer loss did not decrease: {losses:?}"
    );
}

#[test]
fn compiled_transformer_dropout_is_keyed_replay_varying_and_zero_grad_is_not_a_draw() {
    let left_model = TinyCausalTransformer::new(7).unwrap();
    let right_model = TinyCausalTransformer::new(7).unwrap();
    let mut left = compiled_transformer(&left_model).prepare_cpu().unwrap();
    let mut right = compiled_transformer(&right_model).prepare_cpu().unwrap();
    let mut replay_outputs = Vec::new();

    for replay in 1..=3 {
        let left_step = left.step(batch(), TensorData::scalar(0.0)).unwrap();
        let right_step = right.step(batch(), TensorData::scalar(0.0)).unwrap();
        assert_eq!(left_step.loss(), right_step.loss());
        assert_eq!(left_step.outputs(), right_step.outputs());
        replay_outputs.push(left_step.outputs().clone());
        assert_eq!(left.dropout_block_counter().unwrap(), Some(replay * 6));
        assert_eq!(right.dropout_block_counter().unwrap(), Some(replay * 6));
    }
    assert_ne!(replay_outputs[0], replay_outputs[1]);
    let before = left.dropout_block_counter().unwrap();
    assert!(!left.zero_grad().unwrap().did_discard());
    assert_eq!(left.dropout_block_counter().unwrap(), before);
}

#[test]
fn compiled_transformer_plan_is_strictly_renderable_for_metal() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let compiled = compiled_transformer(&model);
    assert_eq!(compiled.loss_scale(), 128.0);
    assert_eq!(compiled.dropout_config(), Some(dropout_config()));
    assert_eq!(compiled.dropout_blocks_per_replay(), Some(6));
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
    assert_eq!(plan.summary().state_pair_count, parameter_count * 3 + 2);
    assert_eq!(plan.summary().state_bank_count, 2);
    assert_eq!(plan.summary().requested_output_count, 2);
    assert!(plan.summary().nonzero_item_count > 0);
    assert_strict_dropout_kernels(&plan);
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
    assert_strict_dropout_kernels(&rendered);
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
    let mut command_wait_count = 0usize;
    let mut transient_h2d_bytes = 0usize;
    let mut retained_d2h_bytes = 0usize;
    for index in 0..4u64 {
        let result = uninterrupted.step(batch(), learning_rate()).unwrap();
        assert_eq!(result.step(), index + 1);
        assert_eq!(result.capture_identity(), capture_identity);
        assert_eq!(result.report().successful_invocation, index + 1);
        assert_eq!(result.report().committed_state_pair_count, state_pair_count);
        assert_eq!(result.report().kernel_launch_count, planned_kernel_count);
        assert_eq!(
            result.report().command_submission_count,
            result.report().command_wait_count
        );
        assert!(result.report().command_submission_count > 0);
        losses.push(result.loss().scalar_at(0).as_f64());
        kernel_launch_count += result.report().kernel_launch_count;
        command_submission_count += result.report().command_submission_count;
        command_wait_count += result.report().command_wait_count;
        transient_h2d_bytes += result.report().transient_h2d_bytes;
        retained_d2h_bytes += result.report().retained_d2h_bytes;
    }

    let checkpoint = uninterrupted.checkpoint().unwrap();
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let resumed_seed = CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
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
    assert_strict_dropout_kernels(&resumed_rendered);
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
        assert_eq!(
            expected.report().command_submission_count,
            expected.report().command_wait_count
        );
        assert_eq!(
            actual.report().command_submission_count,
            actual.report().command_wait_count
        );
        assert!(expected.report().command_submission_count > 0);
        assert!(actual.report().command_submission_count > 0);
        losses.push(expected.loss().scalar_at(0).as_f64());
        kernel_launch_count += expected.report().kernel_launch_count;
        kernel_launch_count += actual.report().kernel_launch_count;
        command_submission_count += expected.report().command_submission_count;
        command_submission_count += actual.report().command_submission_count;
        command_wait_count += expected.report().command_wait_count;
        command_wait_count += actual.report().command_wait_count;
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
    assert_eq!(command_submission_count, command_wait_count);
    assert!(command_submission_count >= 12);
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );
    let runtime_checkpoint = resumed.checkpoint().unwrap();
    let runtime_progress = (
        resumed.step_count(),
        resumed.optimizer_step().unwrap(),
        resumed.accumulation_index().unwrap(),
    );
    let published = resumed.parameter_snapshots().unwrap();
    let published_bytes = published
        .values()
        .map(|value| value.shape().numel().unwrap() * value.dtype().itemsize())
        .sum::<usize>();
    assert_eq!(published.len(), 19);
    assert_eq!(published_bytes, 256);
    let frozen_before = resumed_model.frozen_scale.snapshot().unwrap();
    let before_versions = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert!(
        resumed
            .publish_parameters(&resumed_model)
            .unwrap()
            .is_clean()
    );
    let live = resumed_model.state_dict().unwrap();
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert!(!live.tensors().contains_key("lm_head.weight"));
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
    }
    let frozen_after = resumed_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    assert_eq!(resumed.checkpoint().unwrap(), runtime_checkpoint);
    assert_eq!(
        (
            resumed.step_count(),
            resumed.optimizer_step().unwrap(),
            resumed.accumulation_index().unwrap(),
        ),
        runtime_progress
    );
    let first_eval = evaluate(&resumed_model);
    let second_eval = evaluate(&resumed_model);
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel().unwrap())
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
    }
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
        "format_version": 2,
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
        "published_parameter_count": published.len(),
        "published_parameter_bytes": published_bytes,
        "publication_native_read_count": serde_json::Value::Null,
        "initial_loss": losses.first().unwrap(),
        "final_loss": losses.last().unwrap(),
        "kernel_launch_count": kernel_launch_count,
        "command_submission_count": command_submission_count,
        "command_wait_count": command_wait_count,
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
