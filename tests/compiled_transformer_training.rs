#[cfg(target_os = "macos")]
use rustgrad::MetalSessionTarget;
use rustgrad::nn::{Embedding, LayerNorm, Mode, ModeModuleForward, StateKind};
use rustgrad::runtime::metal::{MetalCapabilities, MetalRenderer};
#[cfg(target_os = "macos")]
use rustgrad::runtime::metal::{
    MetalDeviceRunReport, MetalDiscovery, MetalRuntime, MetalScoreboardContext,
};
use rustgrad::{
    Backend, CompiledAdamWCheckpoint, CompiledAdamWConfig, CompiledAdamWPlan, CompiledAdamWRuntime,
    CompiledCheckpointRuntime, CompiledDropoutConfig, CompiledDropoutKey, CompiledModuleAdamWPlan,
    CompiledTrainingRuntime, CompiledTrainingStep, CpuBackend, CpuSessionTarget, DType, Graph,
    LossOptions, MetalCompiledAdamWPlan, Module, NodeId, Parameter, Reduction, Result, Scalar,
    Shape, TensorData, TrainingDropoutProvider, TransformerBlock, cross_entropy,
};
use std::collections::BTreeMap;
#[cfg(target_os = "macos")]
use std::{env, fs::OpenOptions, io::Write, path::PathBuf};

const VOCAB: usize = 3;
const EMBEDDING: usize = 2;
const BATCH: usize = 2;
const TIME: usize = 3;
const TOKEN_COUNT: usize = BATCH * TIME;
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
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)
        .unwrap()
        .with_weight_decay_exclusions(WEIGHT_DECAY_EXCLUSIONS)
        .unwrap()
        .with_loss_scale(128.0)
        .unwrap()
        .with_host_token_input("tokens", [BATCH, TIME])
        .unwrap()
        .with_input("targets", [BATCH, TIME], DType::I32)
        .unwrap()
}

fn build(
    model: &TinyCausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let logits = model.forward(graph, inputs["tokens"], dropout)?;
    let flat_logits = graph.reshape(logits, [TOKEN_COUNT, VOCAB])?;
    let flat_targets = graph.reshape(inputs["targets"], [TOKEN_COUNT])?;
    let loss = cross_entropy(
        graph,
        flat_logits,
        flat_targets,
        LossOptions {
            reduction: Reduction::Mean,
            ..LossOptions::default()
        },
    )?;
    Ok((loss, BTreeMap::new()))
}

fn dropout_config() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([0x1234_5678, 0x9abc_def0]))
}

fn batch() -> BTreeMap<String, TensorData> {
    let tensor = |values: [i32; TOKEN_COUNT]| {
        TensorData::from_scalars(
            Shape::new([BATCH, TIME]),
            DType::I32,
            values.into_iter().map(|value| Scalar::I(i64::from(value))),
        )
        .unwrap()
    };
    BTreeMap::from([
        ("tokens".into(), tensor([0, 1, 2, 2, 0, 1])),
        ("targets".into(), tensor([1, 2, 0, 0, 1, 2])),
    ])
}

fn learning_rate() -> TensorData {
    TensorData::scalar(0.05)
}

fn evaluate(model: &TinyCausalTransformer) -> TensorData {
    let mut graph = Graph::new();
    let tokens = graph.input_dtype("tokens", [BATCH, TIME], DType::I32);
    let logits = model.forward_eval(&mut graph, tokens).unwrap();
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.insert("tokens".into(), batch().remove("tokens").unwrap());
    CpuBackend.execute(&graph, logits, &bindings).unwrap()
}

fn compiled_transformer(model: &TinyCausalTransformer) -> CompiledAdamWPlan {
    CompiledAdamWPlan::compile_module_with_dropout(config(), dropout_config(), model, build)
        .expect("the fixed causal Transformer training program must compile")
}

fn owned_compiled_transformer(
    model: TinyCausalTransformer,
) -> rustgrad::CompiledModuleAdamWPlan<TinyCausalTransformer> {
    CompiledModuleAdamWPlan::compile_with_dropout(config(), dropout_config(), model, build).unwrap()
}

fn run_exact_resume<R, P>(mut prepare: P) -> Vec<f64>
where
    R: CompiledAdamWRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<TinyCausalTransformer>,
    ) -> Result<rustgrad::CompiledModuleAdamWSession<TinyCausalTransformer, R>>,
{
    let model = TinyCausalTransformer::new(7).unwrap();
    assert!(model.block.is_causal());
    let plan = owned_compiled_transformer(model);
    let capture_identity = plan.capture_identity();
    let mut uninterrupted = prepare(plan).unwrap();
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
    let tied = resumed_model.tokens.weight.clone();
    let frozen = resumed_model.frozen_scale.clone();
    let resumed_plan = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        resumed_model,
        &checkpoint,
        build,
    )
    .unwrap();
    assert_eq!(resumed_plan.capture_identity(), capture_identity);
    assert_eq!(resumed_plan.step_count(), 4);
    let mut resumed = prepare(resumed_plan).unwrap();

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
    let published = resumed.parameter_snapshots().unwrap();
    let tied_version = tied.version().unwrap();
    let frozen_before = frozen.snapshot().unwrap();
    let _uninterrupted_model = uninterrupted.finish().unwrap();
    let resumed_model = resumed.finish().unwrap();
    let live = resumed_model.state_dict().unwrap();
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert_eq!(resumed_model.tokens.weight.id(), tied.id());
    assert_eq!(
        resumed_model.tokens.weight.version().unwrap(),
        tied_version + 1
    );
    let frozen_after = resumed_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    let versions_before_eval = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let first_eval = evaluate(&resumed_model);
    let second_eval = evaluate(&resumed_model);
    assert_eq!(first_eval.shape(), &Shape::new([BATCH, TIME, VOCAB]));
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
    let losses =
        run_exact_resume(|plan| plan.prepare(&target).map_err(|error| error.into_parts().1));

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "compiled causal Transformer loss did not decrease: {losses:?}"
    );
}

#[test]
fn owned_compiled_transformer_session_finishes_and_resumes_one_module_lifecycle() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let tied_identity = model.tokens.weight.id();
    let frozen = model.frozen_scale.clone();
    let frozen_before = frozen.snapshot().unwrap();
    let plan = owned_compiled_transformer(model);
    let capture_identity = plan.capture_identity();
    let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    let mut losses = Vec::new();
    for _ in 0..4 {
        losses.push(
            session
                .step(batch(), learning_rate())
                .unwrap()
                .loss()
                .scalar_at(0)
                .as_f64(),
        );
    }
    let checkpoint = session.checkpoint().unwrap();
    let midpoint = session.parameter_snapshots().unwrap();
    let model = session.finish().unwrap();
    assert_eq!(model.tokens.weight.id(), tied_identity);
    assert_eq!(
        model.tokens.weight.value().unwrap(),
        midpoint["tokens.weight"]
    );
    assert_eq!(model.tokens.weight.version().unwrap(), 1);
    assert_eq!(
        model.frozen_scale.snapshot().unwrap().data,
        frozen_before.data
    );
    assert_eq!(model.frozen_scale.version().unwrap(), frozen_before.version);
    assert!(
        !model
            .state_dict()
            .unwrap()
            .tensors()
            .contains_key("lm_head.weight")
    );

    let resumed = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        model,
        &checkpoint,
        build,
    )
    .unwrap();
    assert_eq!(resumed.capture_identity(), capture_identity);
    assert_eq!(resumed.step_count(), 4);
    let mut session = resumed.prepare(&CpuSessionTarget::new()).unwrap();
    for _ in 0..4 {
        losses.push(
            session
                .step(batch(), learning_rate())
                .unwrap()
                .loss()
                .scalar_at(0)
                .as_f64(),
        );
    }
    let final_parameters = session.parameter_snapshots().unwrap();
    let model = session.finish().unwrap();
    assert_eq!(model.tokens.weight.id(), tied_identity);
    assert_eq!(
        model.tokens.weight.value().unwrap(),
        final_parameters["tokens.weight"]
    );
    assert_eq!(model.tokens.weight.version().unwrap(), 2);
    assert_eq!(
        model.frozen_scale.snapshot().unwrap().data,
        frozen_before.data
    );
    assert_eq!(model.frozen_scale.version().unwrap(), frozen_before.version);
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "owned compiled causal Transformer loss did not decrease: {losses:?}"
    );
}

#[test]
fn compiled_transformer_dropout_is_keyed_replay_varying_and_zero_grad_is_not_a_draw() {
    let left_model = TinyCausalTransformer::new(7).unwrap();
    let right_model = TinyCausalTransformer::new(7).unwrap();
    let mut left = compiled_transformer(&left_model).prepare_cpu().unwrap();
    let mut right = compiled_transformer(&right_model).prepare_cpu().unwrap();
    let mut replay_losses = Vec::new();

    for replay in 1..=8 {
        let left_step = left.step(batch(), TensorData::scalar(0.0)).unwrap();
        let right_step = right.step(batch(), TensorData::scalar(0.0)).unwrap();
        assert_eq!(left_step.loss(), right_step.loss());
        assert_eq!(left_step.outputs(), right_step.outputs());
        replay_losses.push(left_step.loss().clone());
        assert_eq!(left.dropout_block_counter().unwrap(), Some(replay * 12));
        assert_eq!(right.dropout_block_counter().unwrap(), Some(replay * 12));
    }
    assert_ne!(replay_losses[0], replay_losses[1]);
    assert_eq!(left.dropout_block_counter().unwrap(), Some(96));
    assert_eq!(right.dropout_block_counter().unwrap(), Some(96));
    let before = left.dropout_block_counter().unwrap();
    assert!(!left.zero_grad().unwrap().did_discard());
    assert_eq!(left.dropout_block_counter().unwrap(), before);
}

#[test]
fn compiled_transformer_plan_is_strictly_renderable_for_metal() {
    let model = TinyCausalTransformer::new(7).unwrap();
    let policy = config();
    assert_eq!(
        policy.weight_decay_exclusions().collect::<Vec<_>>(),
        WEIGHT_DECAY_EXCLUSIONS
    );
    let exclusions = policy
        .weight_decay_exclusions()
        .collect::<std::collections::BTreeSet<_>>();
    let decayed = model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| !exclusions.contains(name.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        decayed,
        vec![
            "block.ff1.0",
            "block.ff2.0",
            "block.key.0",
            "block.out.0",
            "block.query.0",
            "block.value.0",
            "tokens.weight",
        ]
    );
    let compiled = compiled_transformer(&model);
    assert_eq!(compiled.loss_scale(), 128.0);
    assert_eq!(compiled.dropout_config(), Some(dropout_config()));
    assert_eq!(compiled.dropout_blocks_per_replay(), Some(12));
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
    assert_eq!(plan.summary().logical_state_bytes, 784);
    assert_eq!(plan.summary().state_device_bytes, 1_568);
    assert_eq!(plan.summary().requested_output_count, 1);
    assert!(plan.summary().nonzero_item_count > 0);
    assert_strict_dropout_kernels(&plan);
    let authenticated = plan
        .rendered_items()
        .filter(|item| item.entry.starts_with("rg_metal_training_host_"))
        .collect::<Vec<_>>();
    assert_eq!(authenticated.len(), 2);
    assert!(authenticated.iter().all(|item| {
        item.transaction.is_none()
            && item.indexed_movement().is_none()
            && !item.source.contains("rg_status")
    }));
    assert!(
        plan.rendered_items()
            .all(|item| item.transaction.is_none() && item.indexed_movement().is_none())
    );
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
        "metal-live-compiled-training-v4.json",
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
#[derive(Default)]
struct LiveTrainingTotals {
    kernel_launch_count: usize,
    command_submission_count: usize,
    command_wait_count: usize,
    transient_h2d_calls: usize,
    transient_h2d_bytes: usize,
    retained_d2h_calls: usize,
    retained_d2h_bytes: usize,
    observed_training_invocations: usize,
    device_only_training_invocations: usize,
}

#[cfg(target_os = "macos")]
impl LiveTrainingTotals {
    fn record(
        &mut self,
        report: &MetalDeviceRunReport,
        observed: bool,
        state_pair_count: usize,
        logical_state_bytes: usize,
        planned_kernel_count: usize,
        expected_command_count: usize,
        expected_transient_h2d_calls: usize,
        expected_transient_h2d_bytes: usize,
    ) {
        let output_multiplier = if observed { 1 } else { 0 };
        assert_eq!(report.output_count, output_multiplier);
        assert_eq!(report.retained_d2h_calls, output_multiplier);
        assert_eq!(report.retained_d2h_bytes, output_multiplier * 4);
        assert_eq!(report.committed_state_pair_count, state_pair_count);
        assert_eq!(report.committed_state_bytes, logical_state_bytes);
        assert_eq!(report.committed_state_work_items, 194);
        assert_eq!(report.kernel_launch_count, planned_kernel_count);
        assert_eq!(report.command_submission_count, expected_command_count);
        assert_eq!(report.command_wait_count, expected_command_count);
        assert_eq!(report.transient_h2d_calls, expected_transient_h2d_calls);
        assert_eq!(report.transient_h2d_bytes, expected_transient_h2d_bytes);
        self.kernel_launch_count += report.kernel_launch_count;
        self.command_submission_count += report.command_submission_count;
        self.command_wait_count += report.command_wait_count;
        self.transient_h2d_calls += report.transient_h2d_calls;
        self.transient_h2d_bytes += report.transient_h2d_bytes;
        self.retained_d2h_calls += report.retained_d2h_calls;
        self.retained_d2h_bytes += report.retained_d2h_bytes;
        if observed {
            self.observed_training_invocations += 1;
        } else {
            self.device_only_training_invocations += 1;
        }
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
    let seed = owned_compiled_transformer(model);
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
    let summary = seed
        .metal_summary(target.renderer().clone())
        .expect("the complete training capture must be entirely Metal-admitted");
    assert_eq!(summary.fallback_count, 0);
    assert!(summary.nonzero_item_count > 0);
    let state_pair_count = summary.state_pair_count;
    let logical_state_bytes = summary.logical_state_bytes;
    assert_eq!(state_pair_count, 59);
    assert_eq!(logical_state_bytes, 784);
    assert_eq!(summary.state_bank_count, 2);
    assert_eq!(summary.state_device_bytes, 1_568);
    let planned_kernel_count = summary.nonzero_item_count;
    let command_count_per_invocation = 1;
    let mut uninterrupted = seed
        .prepare(&target)
        .expect("live Metal preparation must compile, allocate, and upload training state");
    let deployment_identity = uninterrupted
        .execution_scoreboard_report()
        .unwrap()
        .expect("the prepared owned session must expose its deployment evidence")
        .deployment_identity;
    let authenticated_host_indexed_movement_item_count = uninterrupted
        .metal_session()
        .compiled_kernels()
        .filter(|item| item.entry.starts_with("rg_metal_training_host_"))
        .count();
    assert_eq!(authenticated_host_indexed_movement_item_count, 2);
    assert!(
        uninterrupted
            .metal_session()
            .compiled_kernels()
            .all(|item| { item.transaction.is_none() && item.indexed_movement().is_none() })
    );
    assert_eq!(
        uninterrupted
            .metal_session()
            .compiled_kernels()
            .filter(|item| item.entry.starts_with("rg_metal_threefry_"))
            .count(),
        2
    );
    assert_eq!(
        uninterrupted
            .metal_session()
            .compiled_kernels()
            .filter(|item| item.entry == "rg_metal_portable_bitcast")
            .count(),
        4
    );
    let transient_h2d_calls_per_invocation = uninterrupted.metal_session().transient_inputs().len();
    let token_input = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .find(|input| input.name == "tokens")
        .expect("the fixed token minibatch must remain transient");
    assert_eq!(token_input.desc.shape, Shape::new([BATCH, TIME]));
    assert_eq!(token_input.desc.dtype, DType::I32);
    assert_eq!(token_input.desc.bytes, TOKEN_COUNT * DType::I32.itemsize());
    let target_input = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .find(|input| input.name == "targets")
        .expect("the fixed target minibatch must remain transient");
    assert_eq!(target_input.desc.shape, Shape::new([BATCH, TIME]));
    assert_eq!(target_input.desc.dtype, DType::I32);
    assert_eq!(target_input.desc.bytes, TOKEN_COUNT * DType::I32.itemsize());
    let learning_rate_input = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .find(|input| input.name == "__rustgrad_compiled_training_learning_rate")
        .expect("the scalar learning rate must remain transient");
    assert_eq!(learning_rate_input.desc.shape, Shape::new([]));
    assert_eq!(learning_rate_input.desc.dtype, DType::F32);
    assert_eq!(learning_rate_input.desc.bytes, DType::F32.itemsize());
    let transient_h2d_bytes_per_invocation = uninterrupted
        .metal_session()
        .transient_inputs()
        .iter()
        .map(|input| input.desc.bytes)
        .sum::<usize>();
    assert_eq!(transient_h2d_calls_per_invocation, 3);
    assert_eq!(transient_h2d_bytes_per_invocation, 52);
    assert_eq!(uninterrupted.loss_scale(), 128.0);
    assert_eq!(uninterrupted.step_count(), 0);
    assert_eq!(uninterrupted.optimizer_step().unwrap(), 0);

    let mut losses = Vec::new();
    let mut totals = LiveTrainingTotals::default();
    for index in 0..4u64 {
        if matches!(index, 1 | 2) {
            let result = uninterrupted
                .step_without_host_outputs(batch(), learning_rate())
                .unwrap();
            assert_eq!(result.step(), index + 1);
            assert_eq!(result.capture_identity(), capture_identity);
            assert_eq!(result.report().successful_invocation, index + 1);
            totals.record(
                result.report(),
                false,
                state_pair_count,
                logical_state_bytes,
                planned_kernel_count,
                command_count_per_invocation,
                transient_h2d_calls_per_invocation,
                transient_h2d_bytes_per_invocation,
            );
        } else {
            let result = uninterrupted.step(batch(), learning_rate()).unwrap();
            assert_eq!(result.step(), index + 1);
            assert_eq!(result.capture_identity(), capture_identity);
            assert_eq!(result.report().successful_invocation, index + 1);
            losses.push(result.loss().scalar_at(0).as_f64());
            totals.record(
                result.report(),
                true,
                state_pair_count,
                logical_state_bytes,
                planned_kernel_count,
                command_count_per_invocation,
                transient_h2d_calls_per_invocation,
                transient_h2d_bytes_per_invocation,
            );
        }
    }
    let checkpoint = uninterrupted.checkpoint().unwrap();
    let resumed_model = TinyCausalTransformer::new(7).unwrap();
    let tied = resumed_model.tokens.weight.clone();
    let frozen = resumed_model.frozen_scale.clone();
    let tied_version = tied.version().unwrap();
    let frozen_before = frozen.snapshot().unwrap();
    let before_versions = resumed_model
        .trainable_parameters()
        .unwrap()
        .into_iter()
        .map(|(name, parameter)| (name, parameter.version().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let resumed_seed = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config(),
        dropout_config(),
        resumed_model,
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
    assert_eq!(resumed_seed.capture_identity(), capture_identity);
    assert_eq!(
        resumed_seed
            .metal_summary(resumed_target.renderer().clone())
            .expect("the checkpoint-restored capture must remain entirely Metal-admitted")
            .fallback_count,
        0
    );
    let mut resumed = resumed_seed
        .prepare(&resumed_target)
        .expect("checkpoint-restored Metal preparation must succeed");
    let resumed_deployment_identity = resumed
        .execution_scoreboard_report()
        .unwrap()
        .expect("the prepared restored session must expose its deployment evidence")
        .deployment_identity;
    assert_ne!(
        resumed_deployment_identity, deployment_identity,
        "the deployment identity must authenticate the checkpoint-restored state bytes"
    );
    assert_eq!(
        resumed.metal_session().transient_inputs(),
        uninterrupted.metal_session().transient_inputs()
    );
    assert_eq!(resumed.loss_scale(), 128.0);
    assert_eq!(resumed.step_count(), 4);
    assert_eq!(resumed.optimizer_step().unwrap(), 4);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);

    for resumed_index in 0..4u64 {
        if resumed_index < 3 {
            let expected = uninterrupted
                .step_without_host_outputs(batch(), learning_rate())
                .unwrap();
            let actual = resumed
                .step_without_host_outputs(batch(), learning_rate())
                .unwrap();
            assert_eq!(actual.step(), expected.step());
            assert_eq!(actual.optimizer_step(), expected.optimizer_step());
            assert_eq!(actual.accumulation_index(), expected.accumulation_index());
            assert_eq!(actual.did_update(), expected.did_update());
            assert_eq!(actual.report().successful_invocation, resumed_index + 1);
            for report in [expected.report(), actual.report()] {
                totals.record(
                    report,
                    false,
                    state_pair_count,
                    logical_state_bytes,
                    planned_kernel_count,
                    command_count_per_invocation,
                    transient_h2d_calls_per_invocation,
                    transient_h2d_bytes_per_invocation,
                );
            }
        } else {
            let expected = uninterrupted.step(batch(), learning_rate()).unwrap();
            let actual = resumed.step(batch(), learning_rate()).unwrap();
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.step(), expected.step());
            assert_eq!(actual.report().successful_invocation, resumed_index + 1);
            for report in [expected.report(), actual.report()] {
                totals.record(
                    report,
                    true,
                    state_pair_count,
                    logical_state_bytes,
                    planned_kernel_count,
                    command_count_per_invocation,
                    transient_h2d_calls_per_invocation,
                    transient_h2d_bytes_per_invocation,
                );
            }
            losses.push(expected.loss().scalar_at(0).as_f64());
        }
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "live compiled causal Transformer loss did not decrease: {losses:?}"
    );
    assert_eq!(resumed.step_count(), 8);
    assert_eq!(uninterrupted.step_count(), 8);
    assert_eq!(totals.observed_training_invocations, 4);
    assert_eq!(totals.device_only_training_invocations, 8);
    assert_eq!(totals.command_submission_count, 12);
    assert_eq!(totals.command_wait_count, 12);
    assert_eq!(totals.kernel_launch_count, planned_kernel_count * 12);
    assert_eq!(
        totals.transient_h2d_calls,
        transient_h2d_calls_per_invocation * 12
    );
    assert_eq!(totals.transient_h2d_calls, 36);
    assert_eq!(
        totals.transient_h2d_bytes,
        transient_h2d_bytes_per_invocation * 12
    );
    assert_eq!(totals.transient_h2d_bytes, 624);
    assert_eq!(totals.retained_d2h_calls, 4);
    assert_eq!(totals.retained_d2h_bytes, 16);
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );
    let published = resumed.parameter_snapshots().unwrap();
    let published_bytes = published
        .values()
        .map(|value| value.shape().numel().unwrap() * value.dtype().itemsize())
        .sum::<usize>();
    assert_eq!(published.len(), 19);
    assert_eq!(published_bytes, 256);
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
    assert_eq!(initial_scoreboard.retained_host_api_d2h_calls, 3);
    assert_eq!(initial_scoreboard.retained_host_api_d2h_bytes, 12);
    assert_eq!(resumed_scoreboard.retained_host_api_d2h_calls, 1);
    assert_eq!(resumed_scoreboard.retained_host_api_d2h_bytes, 4);
    assert_eq!(
        initial_scoreboard
            .successful_runs
            .iter()
            .filter(|run| run.output_count == 0)
            .count(),
        5
    );
    assert_eq!(
        resumed_scoreboard
            .successful_runs
            .iter()
            .filter(|run| run.output_count == 0)
            .count(),
        3
    );
    assert!(
        initial_scoreboard
            .successful_runs
            .iter()
            .chain(&resumed_scoreboard.successful_runs)
            .all(
                |run| run.command_submission_count == command_count_per_invocation
                    && run.command_wait_count == command_count_per_invocation
                    && run.committed_state_pair_count == state_pair_count
                    && run.committed_state_bytes == logical_state_bytes
                    && run.committed_state_work_items == 194
                    && ((run.output_count == 0
                        && run.retained_host_api_d2h_calls == 0
                        && run.retained_host_api_d2h_bytes == 0)
                        || (run.output_count == 1
                            && run.retained_host_api_d2h_calls == 1
                            && run.retained_host_api_d2h_bytes == 4))
            )
    );
    let loss_scale = uninterrupted.loss_scale();
    let _uninterrupted_model = uninterrupted
        .finish()
        .expect("the uninterrupted owned module must finish atomically");
    let resumed_model = resumed
        .finish()
        .expect("the resumed owned module must finish atomically");
    let live = resumed_model.state_dict().unwrap();
    for (name, value) in &published {
        assert_eq!(&live.tensors()[name], value);
    }
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert_eq!(resumed_model.tokens.weight.id(), tied.id());
    assert_eq!(
        resumed_model.tokens.weight.version().unwrap(),
        tied_version + 1
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
    }
    let frozen_after = resumed_model.frozen_scale.snapshot().unwrap();
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    let first_eval = evaluate(&resumed_model);
    let second_eval = evaluate(&resumed_model);
    assert_eq!(first_eval.shape(), &Shape::new([BATCH, TIME, VOCAB]));
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel().unwrap())
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in resumed_model.trainable_parameters().unwrap() {
        assert_eq!(parameter.version().unwrap(), before_versions[&name] + 1);
    }

    let device_evidence = serde_json::json!({
        "name": device_info.name,
        "registry_id": device_info.registry_id,
        "family": device_info.capabilities.family,
        "unified_memory": device_info.capabilities.unified_memory,
    });
    let identity_evidence = serde_json::json!({
        "capture_identity": capture_identity,
        "loss_scale": loss_scale,
        "initial_deployment_identity": deployment_identity,
        "resumed_deployment_identity": resumed_deployment_identity,
        "fallback_count": 0,
    });
    let state_evidence = serde_json::json!({
        "state_pair_count": state_pair_count,
        "logical_state_bytes": logical_state_bytes,
        "state_work_items": 194,
        "state_bank_count": 2,
        "state_device_bytes": 1_568,
        "planned_kernel_count": planned_kernel_count,
        "indexed_movement_item_count": 0,
        "authenticated_host_indexed_movement_item_count": authenticated_host_indexed_movement_item_count,
        "command_count_per_invocation": command_count_per_invocation,
    });
    let invocation_evidence = serde_json::json!({
        "primary_training_steps": 8,
        "resume_replay_steps": 4,
        "total_device_invocations": 12,
        "observed_training_invocations": totals.observed_training_invocations,
        "device_only_training_invocations": totals.device_only_training_invocations,
    });
    let checkpoint_evidence = serde_json::json!({
        "checkpoint_resume_step": 4,
        "checkpoint_resume_exact": true,
        "published_parameter_count": published.len(),
        "published_parameter_bytes": published_bytes,
        "publication_native_read_count": serde_json::Value::Null,
    });
    let loss_evidence = serde_json::json!({
        "initial_loss": losses.first().unwrap(),
        "final_loss": losses.last().unwrap(),
    });
    let accounting_evidence = serde_json::json!({
        "kernel_launch_count": totals.kernel_launch_count,
        "command_submission_count": totals.command_submission_count,
        "command_wait_count": totals.command_wait_count,
        "transient_host_api_h2d_calls": totals.transient_h2d_calls,
        "transient_host_api_h2d_bytes": totals.transient_h2d_bytes,
        "retained_host_api_d2h_calls": totals.retained_d2h_calls,
        "retained_host_api_d2h_bytes": totals.retained_d2h_bytes,
        "device_only_retained_host_api_d2h_calls": 0,
        "device_only_retained_host_api_d2h_bytes": 0,
        "observed_retained_host_api_d2h_calls": totals.retained_d2h_calls,
        "observed_retained_host_api_d2h_bytes": totals.retained_d2h_bytes,
    });
    let scoreboard_evidence = serde_json::json!({
        "initial_scoreboard": initial_scoreboard,
        "resumed_scoreboard": resumed_scoreboard,
    });
    let mut evidence = serde_json::json!({
        "format_version": 4,
        "workload": "tiny-causal-transformer-compiled-adamw",
        "implementation_revision": expected_sha,
        "device": device_evidence,
    });
    let evidence_object = evidence
        .as_object_mut()
        .expect("live evidence root must remain an object");
    for fragment in [
        identity_evidence,
        state_evidence,
        invocation_evidence,
        checkpoint_evidence,
        loss_evidence,
        accounting_evidence,
        scoreboard_evidence,
    ] {
        let serde_json::Value::Object(fragment) = fragment else {
            unreachable!("live evidence fragment is statically an object")
        };
        evidence_object.extend(fragment);
    }
    evidence_object.insert("weight_decay".into(), config().weight_decay().into());
    evidence_object.insert(
        "weight_decay_exclusions".into(),
        serde_json::json!(WEIGHT_DECAY_EXCLUSIONS),
    );
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
