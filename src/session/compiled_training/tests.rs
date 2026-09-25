use super::observation::{
    AdamWObservation, CompiledTrainingObservationNode, CompiledTrainingObservationValue,
};
use super::*;
use crate::nn::{ParameterSnapshot, StateKind};
use crate::{
    Backend, CpuBackend, LossOptions, Op, Parameter, SafetensorsFileError, SafetensorsReadLimits,
    cross_entropy,
};
use std::{
    cell::Cell,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

struct TemporaryCheckpointPath {
    path: PathBuf,
}

struct TemporaryCheckpointDirectory {
    path: PathBuf,
}

impl TemporaryCheckpointPath {
    fn new(label: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let ordinal = NEXT.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!(
                "rustgrad-{label}-{}-{ordinal}.safetensors",
                std::process::id()
            )),
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryCheckpointPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl TemporaryCheckpointDirectory {
    fn new(label: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let ordinal = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("rustgrad-{label}-{}-{ordinal}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryCheckpointDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn native_training_program_roles_authenticate_batch_and_plan_assignment() {
    use NativeCpuTrainingProgramRole::{Accumulation, Evaluation, Main, PartialFlush, ZeroGrad};

    let mut batch = NativeCpuTrainingProgramBatch::with_capacity(5);
    batch.push(Main, "main", 10).unwrap();
    batch.push(Accumulation, "accumulation", 20).unwrap();
    batch.push(PartialFlush, "partial_flush", 30).unwrap();
    batch.push(ZeroGrad, "zero_grad", 40).unwrap();
    batch.push(Evaluation, "evaluation", 50).unwrap();
    let (roles, programs) = batch.into_planning_inputs();
    assert_eq!(
        roles,
        [Main, Accumulation, PartialFlush, ZeroGrad, Evaluation]
    );
    assert_eq!(
        programs,
        [
            ("main", 10),
            ("accumulation", 20),
            ("partial_flush", 30),
            ("zero_grad", 40),
            ("evaluation", 50),
        ]
    );

    let plans = NativeCpuTrainingPrograms::from_ordered(roles, vec![1, 2, 3, 4, 5]).unwrap();
    assert_eq!(plans.main, 1);
    assert_eq!(plans.accumulation, Some(2));
    assert_eq!(plans.partial_flush, Some(3));
    assert_eq!(plans.zero_grad, Some(4));
    assert_eq!(plans.evaluation, Some(5));

    let capsule_diagnostic = |program_index| crate::backend::NativeRenderCapsuleDiagnostic {
        program_index,
        load: crate::backend::NativeRenderCapsuleLoadStatus::Hit,
        store: crate::backend::NativeRenderCapsuleStoreStatus::NotAttempted,
    };
    let main_evaluation = NativeCpuRenderCapsuleDiagnostic::from_ordered_native(
        &[Main, Evaluation],
        vec![capsule_diagnostic(0), capsule_diagnostic(1)],
    )
    .unwrap();
    assert_eq!(
        main_evaluation
            .iter()
            .map(NativeCpuRenderCapsuleDiagnostic::role)
            .collect::<Vec<_>>(),
        [
            NativeCpuRenderCapsuleProgramRole::Main,
            NativeCpuRenderCapsuleProgramRole::Evaluation,
        ]
    );
    let accumulating = NativeCpuRenderCapsuleDiagnostic::from_ordered_native(
        &[Main, Accumulation, PartialFlush],
        vec![
            capsule_diagnostic(0),
            capsule_diagnostic(1),
            capsule_diagnostic(2),
        ],
    )
    .unwrap();
    assert_eq!(
        accumulating
            .iter()
            .map(NativeCpuRenderCapsuleDiagnostic::role)
            .collect::<Vec<_>>(),
        [
            NativeCpuRenderCapsuleProgramRole::Main,
            NativeCpuRenderCapsuleProgramRole::Accumulation,
            NativeCpuRenderCapsuleProgramRole::PartialFlush,
        ]
    );
    assert!(
        NativeCpuRenderCapsuleDiagnostic::from_ordered_native(
            &[Main, Evaluation],
            vec![capsule_diagnostic(1), capsule_diagnostic(0)],
        )
        .is_err()
    );
    assert!(
        NativeCpuRenderCapsuleDiagnostic::from_ordered_native(
            &[Main, Evaluation],
            vec![capsule_diagnostic(0)],
        )
        .is_err()
    );

    let mut missing_main = NativeCpuTrainingProgramBatch::with_capacity(1);
    assert!(missing_main.push(Evaluation, (), ()).is_err());
    let mut reordered = NativeCpuTrainingProgramBatch::with_capacity(3);
    reordered.push(Main, (), ()).unwrap();
    reordered.push(PartialFlush, (), ()).unwrap();
    assert!(reordered.push(Accumulation, (), ()).is_err());
    assert!(NativeCpuTrainingPrograms::from_ordered(vec![Main], Vec::<u8>::new()).is_err());
    assert!(NativeCpuTrainingPrograms::from_ordered(vec![Main, Main], vec![1_u8, 2]).is_err());
}

#[test]
fn native_preparation_work_authenticates_exact_module_build_mode() {
    let direct = NativeCpuPreparationWork {
        rendered_entry_count: 512,
        rendered_source_bytes: 5_120,
        loaded_module_count: 1,
        referenced_module_count: 1,
        unique_rendered_entry_count: 512,
        unique_rendered_source_bytes: 5_120,
        shared_prefix_entry_count: 0,
        shared_prefix_source_bytes: 0,
        shared_prefix_source_program: None,
        durable_artifact_cache_hit_count: 0,
        durable_artifact_cache_miss_count: 1,
        combined_compile_link_count: 1,
        object_compile_count: 0,
        linker_invocation_count: 0,
        compiler_invocation_count: 1,
    };
    assert!(direct.validate(512).is_ok());
    let mismatched_source_partition = NativeCpuPreparationWork {
        unique_rendered_source_bytes: 5_119,
        ..direct
    };
    assert!(mismatched_source_partition.validate(512).is_err());
    let oversized_direct = NativeCpuPreparationWork {
        rendered_entry_count: 682,
        unique_rendered_entry_count: 682,
        ..direct
    };
    assert!(oversized_direct.validate(682).is_err());

    let chunked = NativeCpuPreparationWork {
        combined_compile_link_count: 0,
        object_compile_count: 2,
        linker_invocation_count: 1,
        compiler_invocation_count: 3,
        ..oversized_direct
    };
    assert!(chunked.validate(682).is_ok());
    let bounded_chunked = NativeCpuPreparationWork {
        rendered_entry_count: 512,
        unique_rendered_entry_count: 512,
        ..chunked
    };
    assert!(bounded_chunked.validate(512).is_err());
}

#[test]
fn native_preparation_authenticates_strict_dispatch_segmentation() {
    let segmentation = NativeCpuDispatchSegmentation {
        segment_count: 3,
        dispatch_reached_module_count: 2,
        terminal_segment_count: 1,
        non_dispatch_boundary_count: 0,
        module_change_count: 1,
        output_slot_alias_count: 1,
        derived_slot_dependency_count: 0,
    };
    assert!(segmentation.validate(465, 2).is_ok());

    let mut missing_terminal = segmentation;
    missing_terminal.terminal_segment_count = 0;
    missing_terminal.output_slot_alias_count = 2;
    assert!(missing_terminal.validate(465, 2).is_err());

    let mut fallback_boundary = segmentation;
    fallback_boundary.non_dispatch_boundary_count = 1;
    fallback_boundary.output_slot_alias_count = 0;
    assert!(fallback_boundary.validate(465, 2).is_err());

    let mut too_few_module_changes = segmentation;
    too_few_module_changes.module_change_count = 0;
    too_few_module_changes.output_slot_alias_count = 2;
    assert!(too_few_module_changes.validate(465, 2).is_err());

    let mut too_many_module_changes = segmentation;
    too_many_module_changes.module_change_count = 2;
    too_many_module_changes.output_slot_alias_count = 0;
    assert!(too_many_module_changes.validate(465, 2).is_err());

    let all_elided = NativeCpuDispatchSegmentation {
        segment_count: 0,
        dispatch_reached_module_count: 0,
        terminal_segment_count: 0,
        non_dispatch_boundary_count: 0,
        module_change_count: 0,
        output_slot_alias_count: 0,
        derived_slot_dependency_count: 0,
    };
    assert!(all_elided.validate(1, 1).is_ok());

    let elided_only_suffix = NativeCpuDispatchSegmentation {
        segment_count: 1,
        dispatch_reached_module_count: 1,
        terminal_segment_count: 1,
        non_dispatch_boundary_count: 0,
        module_change_count: 0,
        output_slot_alias_count: 0,
        derived_slot_dependency_count: 0,
    };
    assert!(elided_only_suffix.validate(2, 2).is_ok());
}

#[test]
fn compiled_state_aliases_receive_explicit_capture_owners() {
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("state", [], DType::F32, false);
    let constant = graph
        .full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)
        .unwrap();
    let expanded = graph
        .lazy_full_with_dtype(Shape::new([2]), Scalar::I(0), DType::F32)
        .unwrap();
    let one = graph
        .full_with_dtype(Shape::from([]), Scalar::F(1.0), DType::F32)
        .unwrap();
    let computed = graph.add(input, one).unwrap();
    let sources = [input, constant, expanded, computed];
    let owners = materialize_compiled_state_aliases(&mut graph, &sources).unwrap();

    for (source, owner) in sources[..3].iter().zip(&owners[..3]) {
        assert_ne!(owner, source);
        assert!(matches!(
            graph.op(*owner).unwrap(),
            Op::Contiguous { input } if input == source
        ));
    }
    assert_eq!(owners[3], computed);
    let scheduled = schedule_many(&graph, &owners).unwrap();
    let produced = scheduled
        .items
        .iter()
        .flat_map(|item| item.outputs.iter())
        .map(|output| output.id)
        .collect::<BTreeSet<_>>();
    assert!(
        owners
            .iter()
            .all(|owner| produced.contains(&(owner.index() as u64)))
    );
}

#[test]
fn state_dependent_f32_zero_is_exact_and_retains_backend_neutral_scalar_ops() {
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("state", [5], DType::F32, false);
    let output = state_dependent_zero(&mut graph, input).unwrap();
    let Op::Select {
        condition, on_true, ..
    } = graph.op(output).unwrap()
    else {
        panic!("state-dependent zero must retain Select");
    };
    assert_eq!(*on_true, input);
    let Op::Compare {
        op: CompareOp::Lt,
        lhs,
        rhs,
    } = graph.op(*condition).unwrap()
    else {
        panic!("state-dependent zero must retain ordered Lt");
    };
    assert_eq!(*lhs, input);
    assert!(matches!(
        graph.op(*rhs).unwrap(),
        Op::Binary {
            op: crate::BinaryOp::Add,
            lhs: shifted_input,
            ..
        } if *shifted_input == input
    ));

    let schedule = schedule_many(&graph, &[output]).unwrap();
    assert_eq!(schedule.items.len(), 1);
    let kernel = &schedule.items[0].kernel;
    assert!(matches!(kernel.operation(), crate::Operation::Sink));
    let operations = kernel.topological().unwrap();
    assert!(operations.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::GraphBinary(crate::BinaryOp::Add)
    )));
    assert!(operations.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::GraphCompare(CompareOp::Lt)
    )));
    assert!(
        operations
            .iter()
            .any(|node| matches!(node.operation(), crate::Operation::Ternary(_)))
    );

    let values = TensorData::from_storage(
        [5],
        crate::Storage::F32(vec![3.5, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
    )
    .unwrap();
    let actual = CpuBackend
        .execute(&graph, output, &HashMap::from([("state".into(), values)]))
        .unwrap();
    let crate::Storage::F32(values) = actual.storage() else {
        panic!("state-dependent F32 zero changed dtype");
    };
    assert!(
        values
            .iter()
            .all(|value| value.to_bits() == 0.0f32.to_bits())
    );
}

#[test]
fn compiled_dropout_reserves_source_order_blocks_only_for_active_f32_draws() {
    let mut graph = Graph::new();
    let counter = graph.input_dtype_requires_grad("counter", [], DType::U64, false);
    let f32_three = graph.input_dtype_requires_grad("f32_three", [3], DType::F32, true);
    let f32_four = graph.input_dtype_requires_grad("f32_four", [4], DType::F32, true);
    let empty = graph.input_dtype_requires_grad("empty", [0], DType::F32, true);
    let integer = graph.input_dtype_requires_grad("integer", [3], DType::I32, false);
    let mut stream = CompiledDropoutStream::new(
        counter,
        CompiledDropoutConfig::new(CompiledDropoutKey([3, 7])),
    );

    assert_eq!(
        stream.dropout(&mut graph, f32_three, 0.0).unwrap(),
        f32_three
    );
    assert_eq!(stream.dropout(&mut graph, empty, 0.5).unwrap(), empty);
    let all_zero = stream.dropout(&mut graph, f32_three, 1.0).unwrap();
    assert_eq!(graph.shape(all_zero).unwrap(), &Shape::new([3]));
    assert!(stream.dropout(&mut graph, integer, 0.5).is_err());
    let first = stream.dropout(&mut graph, f32_three, 0.5).unwrap();
    let second = stream.dropout(&mut graph, f32_four, 0.25).unwrap();
    assert_eq!(graph.shape(first).unwrap(), &Shape::new([3]));
    assert_eq!(graph.shape(second).unwrap(), &Shape::new([4]));

    let (successor, state) = stream.finish(&mut graph).unwrap();
    assert_eq!(state.blocks_per_replay, 4);
    assert_eq!(state.config.key().words(), [3, 7]);
    assert_eq!(graph.dtype(successor).unwrap(), DType::U64);
    assert!(!graph.requires_grad(successor).unwrap());
    let loss = graph.sum_all(first).unwrap();
    let gradient = graph.gradient_default(loss, &[f32_three]).unwrap()[0];
    let bindings = HashMap::from([
        (
            "counter".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(0)]).unwrap(),
        ),
        (
            "f32_three".into(),
            TensorData::new([3], vec![1.0, 1.0, 1.0]).unwrap(),
        ),
    ]);
    let cpu = CpuBackend;
    assert_eq!(
        cpu.execute(&graph, gradient, &bindings).unwrap(),
        cpu.execute(&graph, first, &bindings).unwrap(),
        "for unit inputs and p=0.5, the VJP is exactly the realized mask/(1-p)"
    );
    assert_eq!(
        (0..graph.node_count())
            .filter(|index| matches!(graph.op(NodeId(*index)).unwrap(), Op::Threefry { .. }))
            .count(),
        2
    );
}

struct TiedFrozenModule {
    shared: Parameter,
    frozen: Parameter,
    buffer: Parameter,
}

impl TiedFrozenModule {
    fn new(frozen: [f32; 2]) -> Self {
        Self {
            shared: Parameter::new(TensorData::new([2], vec![0.25, -0.5]).unwrap(), true),
            frozen: Parameter::new(TensorData::new([2], frozen.to_vec()).unwrap(), false),
            buffer: Parameter::new(TensorData::scalar(3.0), false),
        }
    }
}

impl Module for TiedFrozenModule {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        assert!(prefix.is_empty());
        visitor("shared".into(), &self.shared, StateKind::Parameter);
        visitor("shared_alias".into(), &self.shared, StateKind::Parameter);
        visitor("frozen".into(), &self.frozen, StateKind::Parameter);
        visitor("buffer".into(), &self.buffer, StateKind::Buffer);
    }
}

#[derive(Debug)]
struct FinishRaceModule {
    weight: Parameter,
    finish_visits: Cell<u64>,
    race_after_second_visit: Cell<bool>,
}

struct CheckpointCountingRuntime {
    inner: CpuCompiledAdamW,
    checkpoint_calls: Rc<Cell<u64>>,
}

/// Test-only optimizer facade proving that the shared training capabilities do
/// not require an AdamW implementation from their concrete runtime.
struct OptimizerNeutralRuntimeProbe {
    inner: CpuCompiledAdamW,
}

impl CompiledTrainingRuntime for OptimizerNeutralRuntimeProbe {
    type Step = CompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CompiledTrainingRuntime::step(&mut self.inner, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        CompiledTrainingRuntime::step_count(&self.inner)
    }

    fn capture_identity(&self) -> u64 {
        CompiledTrainingRuntime::capture_identity(&self.inner)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CompiledTrainingRuntime::parameter_snapshots(&self.inner)
    }
}

impl CompiledTrainingWindowResetRuntime for OptimizerNeutralRuntimeProbe {
    fn reset_gradient_window(&mut self) -> Result<CompiledTrainingWindowReset> {
        CompiledTrainingWindowResetRuntime::reset_gradient_window(&mut self.inner)
    }

    fn gradient_window_reset_capture_identity(&self) -> Option<u64> {
        CompiledTrainingWindowResetRuntime::gradient_window_reset_capture_identity(&self.inner)
    }
}

impl CompiledTrainingWindowRuntime for OptimizerNeutralRuntimeProbe {
    fn gradient_window_size(&self) -> u64 {
        CompiledTrainingWindowRuntime::gradient_window_size(&self.inner)
    }

    fn pending_microbatch_count(&self) -> Result<u64> {
        CompiledTrainingWindowRuntime::pending_microbatch_count(&self.inner)
    }
}

impl CompiledTrainingWindowCommitRuntime for OptimizerNeutralRuntimeProbe {
    type WindowCommit = CompiledAdamWFlushResult;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        CompiledTrainingWindowCommitRuntime::commit_partial_window(&mut self.inner, learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        CompiledTrainingWindowCommitRuntime::partial_window_commit_capture_identity(&self.inner)
    }
}

impl CompiledTrainingRatePolicyRuntime for OptimizerNeutralRuntimeProbe {
    fn step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        CompiledTrainingRatePolicyRuntime::step_with_rate_policy(&mut self.inner, inputs)
    }
}

impl CompiledTrainingRatePolicyCommitOnlyRuntime for OptimizerNeutralRuntimeProbe {
    fn commit_step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        CompiledTrainingRatePolicyCommitOnlyRuntime::commit_step_with_rate_policy(
            &mut self.inner,
            inputs,
        )
    }
}

impl CompiledTrainingRatePolicyWindowCommitRuntime for OptimizerNeutralRuntimeProbe {
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        CompiledTrainingRatePolicyWindowCommitRuntime::commit_partial_window_with_rate_policy(
            &mut self.inner,
        )
    }
}

impl CompiledTrainingRuntime for CheckpointCountingRuntime {
    type Step = CompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.inner.step(inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        CompiledTrainingRuntime::publish_parameters(&self.inner, module)
    }
}

impl CompiledCheckpointRuntime for CheckpointCountingRuntime {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.checkpoint_calls
            .set(self.checkpoint_calls.get().saturating_add(1));
        self.inner.checkpoint()
    }
}

impl CompiledAdamWRuntime for CheckpointCountingRuntime {
    fn gradient_accumulation_steps(&self) -> u64 {
        self.inner.gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.inner.max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.inner.loss_scale()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.inner.optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.inner.accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.inner.zero_grad()
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.inner.zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.gradient_accumulator_snapshots()
    }
}

impl FinishRaceModule {
    fn new() -> Self {
        Self {
            weight: Parameter::new(TensorData::new([2], vec![0.25, -0.5]).unwrap(), true),
            finish_visits: Cell::new(0),
            race_after_second_visit: Cell::new(false),
        }
    }

    fn arm_finish_race(&self) {
        self.finish_visits.set(0);
        self.race_after_second_visit.set(true);
    }
}

impl Module for FinishRaceModule {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        assert!(prefix.is_empty());
        let visit = self.finish_visits.get() + 1;
        self.finish_visits.set(visit);
        visitor("weight".into(), &self.weight, StateKind::Parameter);
        if self.race_after_second_visit.get() && visit == 2 {
            self.weight.replace(self.weight.value().unwrap()).unwrap();
            self.race_after_second_visit.set(false);
        }
    }
}

fn build_finish_race(
    module: &FinishRaceModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let weight = module.weight.bind(graph)?;
    let output = graph.mul(weight, inputs["x"])?;
    Ok((graph.sum_all(output)?, BTreeMap::new()))
}

struct FineTuneModule {
    base: Parameter,
    adapter: Parameter,
    frozen: Parameter,
}

impl FineTuneModule {
    fn new() -> Self {
        Self {
            base: Parameter::new(TensorData::new([2], vec![0.25, -0.5]).unwrap(), true),
            adapter: Parameter::new(TensorData::new([2], vec![0.1, 0.2]).unwrap(), true),
            frozen: Parameter::new(TensorData::new([2], vec![1.0, -1.0]).unwrap(), false),
        }
    }
}

impl Module for FineTuneModule {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        assert!(prefix.is_empty());
        visitor("base".into(), &self.base, StateKind::Parameter);
        visitor("base_alias".into(), &self.base, StateKind::Parameter);
        visitor("adapter".into(), &self.adapter, StateKind::Parameter);
        visitor("frozen".into(), &self.frozen, StateKind::Parameter);
    }
}

fn build_fine_tune(
    module: &FineTuneModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let base = module.base.bind(graph)?;
    assert_eq!(module.base.bind(graph)?, base);
    let adapter = module.adapter.bind(graph)?;
    let frozen = module.frozen.bind(graph)?;
    let scaled = graph.mul(inputs["x"], base)?;
    let adapted = graph.add(scaled, adapter)?;
    let output = graph.add(adapted, frozen)?;
    let squared = graph.square(output)?;
    let loss = graph.reduce(squared, crate::ReduceKind::Mean, None, false)?;
    Ok((loss, BTreeMap::from([("output".into(), output)])))
}

fn build_fine_tune_clip(
    module: &FineTuneModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let base = module.base.bind(graph)?;
    assert_eq!(module.base.bind(graph)?, base);
    let adapter = module.adapter.bind(graph)?;
    let scaled = graph.mul(inputs["x"], base)?;
    let output = graph.add(scaled, adapter)?;
    Ok((graph.mean_default(output)?, BTreeMap::new()))
}

fn assert_parameter_snapshot_eq(actual: &ParameterSnapshot, expected: &ParameterSnapshot) {
    assert_eq!(actual.data, expected.data);
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(actual.dtype, expected.dtype);
    assert_eq!(actual.version, expected.version);
    assert_eq!(actual.identity, expected.identity);
    assert_eq!(actual.trainable, expected.trainable);
    assert_eq!(actual.input_name, expected.input_name);
}

fn build_tied_dropout(
    module: &TiedFrozenModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let dropped = dropout.dropout(graph, inputs["x"], 0.5)?;
    let shared = module.shared.bind(graph)?;
    let frozen = module.frozen.bind(graph)?;
    let scaled = graph.mul(dropped, shared)?;
    let output = graph.add(scaled, frozen)?;
    let squared = graph.square(output)?;
    let loss = graph.sum_all(squared)?;
    Ok((loss, BTreeMap::from([("output".into(), output)])))
}

fn build_double_dropout(
    module: &TiedFrozenModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let first = dropout.dropout(graph, inputs["x"], 0.5)?;
    let second = dropout.dropout(graph, first, 0.5)?;
    let shared = module.shared.bind(graph)?;
    let output = graph.mul(second, shared)?;
    let loss = graph.sum_all(output)?;
    Ok((loss, BTreeMap::from([("output".into(), output)])))
}

fn build_tied_dropout_with_input_guard(
    module: &TiedFrozenModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let (loss, outputs) = build_tied_dropout(module, graph, inputs, dropout)?;
    let reciprocal = graph.reciprocal(inputs["x"])?;
    let reciprocal_sum = graph.sum_all(reciprocal)?;
    Ok((graph.add(loss, reciprocal_sum)?, outputs))
}

#[test]
fn compiled_dropout_effect_failure_retry_and_zero_grad_preserve_counter_contract() {
    let config = module_config().with_gradient_accumulation(2).unwrap();
    let key = CompiledDropoutConfig::new(CompiledDropoutKey([11, 13]));
    let module = TiedFrozenModule::new([0.1, -0.2]);
    let reference_module = TiedFrozenModule::new([0.1, -0.2]);
    let mut candidate = CompiledAdamWPlan::compile_module_with_dropout(
        config.clone(),
        key,
        &module,
        build_tied_dropout,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap();
    let mut reference = CompiledAdamWPlan::compile_module_with_dropout(
        config,
        key,
        &reference_module,
        build_tied_dropout,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap();
    let inputs = || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
    let before = candidate.checkpoint().unwrap();
    assert!(
        candidate
            .step_inner(inputs(), TensorData::scalar(0.01), Some(0))
            .is_err()
    );
    assert_eq!(candidate.dropout_block_counter().unwrap(), Some(0));
    assert_eq!(candidate.checkpoint().unwrap(), before);
    let expected = reference.step(inputs(), TensorData::scalar(0.01)).unwrap();
    let actual = candidate.step(inputs(), TensorData::scalar(0.01)).unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());
    assert_eq!(candidate.dropout_block_counter().unwrap(), Some(1));
    assert!(candidate.zero_grad().unwrap().did_discard());
    assert_eq!(candidate.dropout_block_counter().unwrap(), Some(1));
    assert_eq!(candidate.step_count(), 1);
}

#[test]
fn compiled_dropout_checkpoint_v4_requires_exact_restore_policy_and_counter() {
    let key = CompiledDropoutConfig::new(CompiledDropoutKey([23, 29]));
    let module = TiedFrozenModule::new([0.1, -0.2]);
    let mut runtime = CompiledAdamWPlan::compile_module_with_dropout(
        module_config(),
        key,
        &module,
        build_tied_dropout,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap();
    runtime
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let checkpoint = runtime.checkpoint().unwrap();
    let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V4);
    let info = checkpoint.info();
    assert_eq!(info.capture_identity(), runtime.capture_identity());
    assert_eq!(info.replay_step(), 1);
    assert_eq!(info.optimizer_step(), 1);
    assert_eq!(info.gradient_accumulation_steps(), 1);
    assert_eq!(info.accumulation_index(), 0);
    assert_eq!(info.discarded_microbatches(), 0);
    assert_eq!(info.flushed_window_count(), 0);
    assert_eq!(info.flushed_microbatch_count(), 0);
    assert_eq!(info.flush_capture_identity(), None);
    assert_eq!(info.dropout_block_counter(), Some(1));

    let fresh = TiedFrozenModule::new([0.1, -0.2]);
    assert!(
        CompiledAdamWPlan::compile_module_from_checkpoint(
            module_config(),
            &fresh,
            &checkpoint,
            build_tied_frozen,
        )
        .is_err()
    );
    assert!(
        CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
            module_config(),
            CompiledDropoutConfig::new(CompiledDropoutKey([23, 30])),
            &fresh,
            &checkpoint,
            build_tied_dropout,
        )
        .is_err()
    );
    assert!(
        CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
            module_config(),
            key,
            &fresh,
            &checkpoint,
            build_double_dropout,
        )
        .is_err()
    );

    let decoded = decode_adamw_checkpoint(checkpoint.as_bytes()).unwrap();
    let malformed = CompiledAdamWCheckpoint::from_bytes(
        encode_adamw_checkpoint(
            AdamWCheckpointProgress {
                capture_identity: decoded.capture_identity,
                accumulation_capture_identity: decoded.accumulation_capture_identity,
                replay_step: decoded.replay_step,
                optimizer_step: decoded.optimizer_step,
                accumulation_steps: decoded.accumulation_steps,
                accumulation_index: decoded.accumulation_index,
                discarded_microbatches: decoded.discarded_microbatches,
                flushed_window_count: decoded.flushed_window_count,
                flushed_microbatch_count: decoded.flushed_microbatch_count,
                flush_capture_identity: decoded.flush_capture_identity,
                dropout_block_counter: decoded.dropout_block_counter.map(|value| value + 1),
                accumulated_token_count: decoded.accumulated_token_count,
                window_loss_report: decoded.window_loss_report,
                reset_transition_count: decoded.reset_transition_count,
                reset_capture_identity: decoded.reset_capture_identity,
            },
            AdamWCheckpointTensors {
                parameters: decoded.parameters,
                first_moments: decoded.first_moments,
                second_moments: decoded.second_moments,
                gradient_accumulators: decoded.gradient_accumulators,
                accumulated_loss_numerator: decoded.accumulated_loss_numerator,
            },
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
            module_config(),
            key,
            &fresh,
            &malformed,
            build_tied_dropout,
        )
        .is_err()
    );

    let ordinary = CpuCompiledAdamW::compile_module(module_config(), &fresh, build_tied_frozen)
        .unwrap()
        .checkpoint()
        .unwrap();
    let (_, metadata) = load_safetensors(ordinary.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V1);
    let ordinary_info = ordinary.info();
    assert_eq!(ordinary_info.replay_step(), 0);
    assert_eq!(ordinary_info.optimizer_step(), 0);
    assert_eq!(ordinary_info.gradient_accumulation_steps(), 1);
    assert_eq!(ordinary_info.accumulation_index(), 0);
    assert_eq!(ordinary_info.discarded_microbatches(), 0);
    assert_eq!(ordinary_info.flushed_window_count(), 0);
    assert_eq!(ordinary_info.flushed_microbatch_count(), 0);
    assert_eq!(ordinary_info.flush_capture_identity(), None);
    assert_eq!(ordinary_info.dropout_block_counter(), None);
    assert_eq!(ordinary_info.accumulated_token_count(), None);
    assert_eq!(ordinary_info.reset_transition_count(), 0);
    assert_eq!(ordinary_info.reset_capture_identity(), None);
    assert!(
        CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
            module_config(),
            key,
            &fresh,
            &ordinary,
            build_tied_dropout,
        )
        .is_err()
    );
    let mut accumulated = CpuCompiledAdamW::compile_module(
        module_config().with_gradient_accumulation(2).unwrap(),
        &fresh,
        build_tied_frozen,
    )
    .unwrap();
    accumulated
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let current = accumulated.checkpoint().unwrap();
    let (state, metadata) = load_safetensors(current.as_bytes()).unwrap();
    let v2 = CompiledAdamWCheckpoint::from_bytes(
        save_safetensors(
            &state,
            &crate::Metadata::from([
                ("format".into(), ADAMW_CHECKPOINT_FORMAT_V2.into()),
                (
                    "capture_identity".into(),
                    metadata["capture_identity"].clone(),
                ),
                ("replay_step".into(), metadata["replay_step"].clone()),
                ("optimizer_step".into(), metadata["optimizer_step"].clone()),
                (
                    "gradient_accumulation_steps".into(),
                    metadata["gradient_accumulation_steps"].clone(),
                ),
                (
                    "accumulation_index".into(),
                    metadata["accumulation_index"].clone(),
                ),
                (
                    "parameter_names".into(),
                    metadata["parameter_names"].clone(),
                ),
            ]),
        )
        .unwrap(),
    )
    .unwrap();
    accumulated.zero_grad().unwrap();
    let current = accumulated.checkpoint().unwrap();
    let (state, metadata) = load_safetensors(current.as_bytes()).unwrap();
    let v3 = CompiledAdamWCheckpoint::from_bytes(
        save_safetensors(
            &state,
            &crate::Metadata::from([
                ("format".into(), ADAMW_CHECKPOINT_FORMAT_V3.into()),
                (
                    "capture_identity".into(),
                    metadata["capture_identity"].clone(),
                ),
                ("replay_step".into(), metadata["replay_step"].clone()),
                ("optimizer_step".into(), metadata["optimizer_step"].clone()),
                (
                    "gradient_accumulation_steps".into(),
                    metadata["gradient_accumulation_steps"].clone(),
                ),
                (
                    "accumulation_index".into(),
                    metadata["accumulation_index"].clone(),
                ),
                (
                    "discarded_microbatch_count".into(),
                    metadata["discarded_microbatch_count"].clone(),
                ),
                (
                    "parameter_names".into(),
                    metadata["parameter_names"].clone(),
                ),
            ]),
        )
        .unwrap(),
    )
    .unwrap();
    for (legacy, format, accumulation_index, discarded_microbatches) in [
        (v2, ADAMW_CHECKPOINT_FORMAT_V2, 1, 0),
        (v3, ADAMW_CHECKPOINT_FORMAT_V3, 0, 1),
    ] {
        let (_, metadata) = load_safetensors(legacy.as_bytes()).unwrap();
        assert_eq!(metadata["format"], format);
        let info = legacy.info();
        assert_eq!(info.replay_step(), 1);
        assert_eq!(info.optimizer_step(), 0);
        assert_eq!(info.gradient_accumulation_steps(), 2);
        assert_eq!(info.accumulation_index(), accumulation_index);
        assert_eq!(info.discarded_microbatches(), discarded_microbatches);
        assert_eq!(info.reset_transition_count(), 0);
        assert_eq!(info.reset_capture_identity(), None);
        assert_eq!(info.flushed_window_count(), 0);
        assert_eq!(info.flushed_microbatch_count(), 0);
        assert_eq!(info.flush_capture_identity(), None);
        assert_eq!(info.dropout_block_counter(), None);
        assert_eq!(info.accumulated_token_count(), None);
        assert_eq!(info.accumulation_capture_identity(), None);
        let restored = CompiledAdamWPlan::compile_module_from_checkpoint(
            module_config().with_gradient_accumulation(2).unwrap(),
            &fresh,
            &legacy,
            build_tied_frozen,
        )
        .unwrap();
        assert_eq!(restored.step_count(), 1);
        assert!(restored.accumulation_capture_identity().is_some());
        assert!(
            CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
                module_config().with_gradient_accumulation(2).unwrap(),
                key,
                &fresh,
                &legacy,
                build_tied_dropout,
            )
            .is_err()
        );
    }
}

#[test]
fn compiled_dropout_rejects_counter_exhaustion_before_replay() {
    let module = TiedFrozenModule::new([0.1, -0.2]);
    let mut plan = CompiledAdamWPlan::compile_module_with_dropout(
        module_config(),
        CompiledDropoutConfig::new(CompiledDropoutKey([17, 19])),
        &module,
        build_double_dropout,
    )
    .unwrap();
    assert_eq!(plan.dropout_blocks_per_replay(), Some(2));
    let replay_step = u64::MAX / 2;
    let mut values = plan.inner.state_values.clone();
    values.insert(
        RecurrentStateKey::adamw_global(AdamWGlobalState::Step),
        TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(replay_step)]).unwrap(),
    );
    values.insert(
        RecurrentStateKey::dropout_counter(),
        TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(replay_step * 2)])
            .unwrap(),
    );
    plan.inner = plan
        .inner
        .clone()
        .restore_frontier(replay_step, values)
        .unwrap();
    plan.progress = CompiledTrainingWindowProgress {
        replay_step,
        optimizer_step: replay_step,
        accumulation_index: 0,
        discarded_microbatches: 0,
        flushed_window_count: 0,
        flushed_microbatch_count: 0,
        reset_transition_count: 0,
    };
    let mut runtime = plan.prepare_cpu().unwrap();
    let before = runtime.checkpoint().unwrap();
    assert!(
        runtime
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap(),)]),
                TensorData::scalar(0.01),
            )
            .is_err()
    );
    assert_eq!(runtime.checkpoint().unwrap(), before);
}

fn module_config() -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_input("x", [2], DType::F32)
        .unwrap()
}

#[test]
fn compiled_adamw_host_token_input_is_atomic_sorted_and_strict() {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_host_token_input("tokens_b", [2, 3])
        .unwrap()
        .with_host_token_input("tokens_a", [1, 2])
        .unwrap();
    assert_eq!(
        config
            .host_token_inputs()
            .map(|(name, shape)| (name, shape.dims()))
            .collect::<Vec<_>>(),
        [("tokens_a", &[1, 2][..]), ("tokens_b", &[2, 3][..])]
    );
    assert_eq!(
        config
            .inputs()
            .map(|(name, shape, dtype)| (name, shape.dims(), dtype))
            .collect::<Vec<_>>(),
        [
            ("tokens_a", &[1, 2][..], DType::I32),
            ("tokens_b", &[2, 3][..], DType::I32),
        ]
    );

    let base = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0).unwrap();
    assert!(
        base.clone()
            .with_input("tokens", [1, 2], DType::I32)
            .unwrap()
            .with_host_token_input("tokens", [1, 2])
            .is_err()
    );
    assert!(
        base.clone()
            .with_host_token_input("tokens", [1, 2])
            .unwrap()
            .with_input("tokens", [1, 2], DType::I32)
            .is_err()
    );
    for shape in [
        Shape::new([0, 2]),
        Shape::new([2, 0]),
        Shape::new([2]),
        Shape::new([1, 2, 1]),
        Shape::new([usize::MAX, 2]),
    ] {
        assert!(base.clone().with_host_token_input("tokens", shape).is_err());
    }
}

fn build_tied_frozen(
    module: &TiedFrozenModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let shared = module.shared.bind(graph)?;
    assert_eq!(module.shared.bind(graph)?, shared);
    assert_eq!(module.shared.node(graph)?, shared);
    let frozen = module.frozen.bind(graph)?;
    assert!(!graph.requires_grad(frozen)?);
    let scaled = graph.mul(inputs["x"], shared)?;
    let tied = graph.add(scaled, shared)?;
    let output = graph.add(tied, frozen)?;
    let squared = graph.square(output)?;
    let loss = graph.reduce(squared, crate::ReduceKind::Mean, None, false)?;
    Ok((loss, BTreeMap::from([("output".into(), output)])))
}

#[test]
fn recurrent_training_phases_derive_from_one_canonical_mixed_capture() {
    let before = canonical_recurrent_capture_counts();
    let module = TiedFrozenModule::new([1.0, -1.0]);
    let plan = with_canonical_recurrent_reference(|| {
        CompiledAdamWPlan::compile_module_graph(
            module_config().with_gradient_accumulation(3).unwrap(),
            &module,
            |module, graph, inputs| {
                let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                Ok(CompiledAdamWGraph::scalar(loss, outputs))
            },
        )
    })
    .unwrap();
    let counts = canonical_recurrent_capture_delta(before, canonical_recurrent_capture_counts());

    assert_eq!(counts.canonical, 4);
    assert_eq!(counts.reference, 4);
    assert!(plan.inner.accumulation.is_some());
    assert!(plan.partial_flush.is_some());
    assert!(plan.zero_grad.is_some());
}

fn tied_token_mean_config() -> CompiledAdamWConfig {
    module_config()
        .with_input("mask", [2], DType::F32)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_token_weighted_gradient_accumulation("mask")
        .unwrap()
        .with_max_gradient_norm(0.25)
        .unwrap()
        .with_clip_report()
}

fn tied_token_mean_dropout() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([71, 73]))
}

fn build_tied_frozen_token_mean(
    module: &TiedFrozenModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    let shared = module.shared.bind(graph)?;
    assert_eq!(module.shared.bind(graph)?, shared);
    let frozen = module.frozen.bind(graph)?;
    assert!(!graph.requires_grad(frozen)?);
    let scaled = graph.mul(inputs["x"], shared)?;
    let tied = graph.add(scaled, shared)?;
    let output = graph.add(tied, frozen)?;
    let dropped = dropout.dropout(graph, output, 0.25)?;
    let losses = graph.square(dropped)?;
    Ok(CompiledAdamWGraph::token_mean(
        losses,
        BTreeMap::from([("output".into(), dropped)]),
    ))
}

fn tied_token_mean_batch(x: [f32; 2], mask: [f32; 2]) -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        ("mask".into(), TensorData::new([2], mask.to_vec()).unwrap()),
        ("x".into(), TensorData::new([2], x.to_vec()).unwrap()),
    ])
}

fn initial_parameters() -> Vec<TrainingParameterInit> {
    vec![
        TrainingParameterInit::new(
            "w1",
            TensorData::new(
                [2, 4],
                vec![0.20, -0.10, 0.05, 0.30, -0.25, 0.15, 0.40, -0.20],
            )
            .unwrap(),
        )
        .unwrap(),
        TrainingParameterInit::new(
            "w2",
            TensorData::new(
                [4, 2],
                vec![0.10, -0.20, 0.30, 0.05, -0.15, 0.25, 0.20, -0.10],
            )
            .unwrap(),
        )
        .unwrap(),
    ]
}

fn config() -> CompiledMomentumSgdConfig {
    CompiledMomentumSgdConfig::new(0.9)
        .unwrap()
        .with_input("x", [4, 2], DType::F32)
        .unwrap()
        .with_input("target", [4], DType::I64)
        .unwrap()
}

fn build_tinybob(
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    parameters: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let hidden = graph.matmul(inputs["x"], parameters["w1"])?;
    let hidden = graph.relu(hidden)?;
    let logits = graph.matmul(hidden, parameters["w2"])?;
    let loss = cross_entropy(graph, logits, inputs["target"], LossOptions::default())?;
    Ok((loss, BTreeMap::from([("logits".into(), logits)])))
}

fn compiled() -> CpuCompiledMomentumSgd {
    CpuCompiledMomentumSgd::compile(config(), initial_parameters(), build_tinybob).unwrap()
}

fn adamw_config() -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)
        .unwrap()
        .with_input_batch::<TinyBobBatch>()
        .unwrap()
}

fn accumulated_adamw_config(steps: u64) -> CompiledAdamWConfig {
    adamw_config().with_gradient_accumulation(steps).unwrap()
}

fn compiled_adamw() -> CpuCompiledAdamW {
    CpuCompiledAdamW::compile(adamw_config(), initial_parameters(), build_tinybob).unwrap()
}

fn non_finite_config(accumulation_steps: u64) -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(accumulation_steps)
        .unwrap()
        .with_input("x", [], DType::F32)
        .unwrap()
}

fn non_finite_plan(accumulation_steps: u64) -> CompiledAdamWPlan {
    let parameter = TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap();
    CompiledAdamWPlan::compile(
        non_finite_config(accumulation_steps),
        [parameter],
        |graph, inputs, parameters| {
            let radicand = graph.add(parameters["weight"], inputs["x"])?;
            let loss = graph.sqrt(radicand)?;
            Ok((loss, BTreeMap::new()))
        },
    )
    .unwrap()
}

fn non_finite_flush_plan() -> CompiledAdamWPlan {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 2.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_clip_report()
        .with_input("x", [], DType::F32)
        .unwrap();
    let parameter = TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap();
    CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
        let loss = graph.mul(parameters["weight"], inputs["x"])?;
        Ok((loss, BTreeMap::new()))
    })
    .unwrap()
}

#[test]
fn compiled_adamw_contract_is_cloned_across_cpu_preparation_and_snapshot() {
    let plan = non_finite_plan(3);
    let expected = plan.contract.clone();
    let runtime = plan.prepare_cpu().unwrap();
    assert_eq!(runtime.contract, expected);
    assert_eq!(runtime.snapshot_plan().unwrap().contract, expected);
}

fn scalar_batch(value: f32) -> BTreeMap<String, TensorData> {
    BTreeMap::from([("x".into(), TensorData::scalar(value))])
}

fn rejecting_cpu_target() -> ConfiguredCpuSessionTarget {
    CpuSessionTarget.with_non_finite_policy(CpuNonFinitePolicy::RejectTransition)
}

#[test]
fn cpu_non_finite_validation_admits_empty_signed_zero_and_subnormal() {
    let empty = TensorData::new([0], Vec::<f32>::new()).unwrap();
    let finite = TensorData::new([2], vec![-0.0, f32::from_bits(1)]).unwrap();
    assert!(validate_finite_tensors([&empty, &finite], "fixture").is_ok());
    assert!(validate_finite_tensors([&TensorData::scalar(f32::INFINITY)], "fixture",).is_err());
}

#[test]
fn clip_report_extraction_preserves_exact_non_finite_f32_bits() {
    let norm_bits = 0x7f80_0123;
    let scale_bits = 0x7fc0_4567;
    let mut values = [
        TensorData::scalar(f32::from_bits(norm_bits)),
        TensorData::scalar(f32::from_bits(scale_bits)),
    ]
    .into_iter();
    let report = take_compiled_clip_report(&mut values, true).unwrap();
    assert_eq!(report.pre_clip_global_norm().to_bits(), norm_bits);
    assert_eq!(report.applied_scale().to_bits(), scale_bits);
    assert!(!report.is_finite());
    assert_eq!(report.did_clip(), None);
    assert!(values.next().is_none());
}

#[test]
fn window_loss_report_extraction_preserves_exact_f32_bits() {
    let loss_bits = 0x7fc0_4567;
    let mut values = [
        TensorData::scalar(f32::from_bits(loss_bits)),
        TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(11)]).unwrap(),
    ]
    .into_iter();
    let value = take_compiled_window_loss_value(&mut values, true).unwrap();
    let report = CompiledAdamWWindowLossReport::new(value, 3);
    assert_eq!(report.mean_loss().to_bits(), loss_bits);
    assert_eq!(report.loss_weight(), 11);
    assert_eq!(report.microbatch_count(), 3);
    assert!(!report.is_finite());
    assert!(values.next().is_none());
}

#[test]
fn global_norm_commits_through_stacked_typed_f32_sum() {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_clip_report();
    let mut graph = Graph::new();
    let left = graph.input_dtype("left", [2], DType::F32);
    let clipped = clip_gradients_by_global_norm(
        &config,
        &mut graph,
        &BTreeMap::from([("left".into(), left)]),
    )
    .unwrap();
    let committed_gradient = clipped.gradients["left"];
    let gradient_owner = crate::rangeify::computed_view(&graph, committed_gradient)
        .unwrap()
        .source;
    assert!(matches!(
        graph.op(gradient_owner).unwrap(),
        Op::Concat { axis: 0, .. }
    ));
    let norm = clipped.report.unwrap().pre_clip_global_norm;
    let Op::Unary {
        op: crate::UnaryOp::Sqrt,
        input: committed,
    } = graph.op(norm).unwrap()
    else {
        panic!("global norm must end in sqrt");
    };
    let Op::Reduce {
        input: stacked,
        kind: crate::ReduceKind::Sum,
        accumulator: DType::F32,
        ..
    } = graph.op(*committed).unwrap()
    else {
        panic!("global norm must consume a typed F32 sum");
    };
    assert_eq!(graph.dtype(*committed).unwrap(), DType::F32);
    assert!(matches!(
        graph.op(*stacked).unwrap(),
        Op::Concat { axis: 0, .. }
    ));
    let Op::Concat { inputs, .. } = graph.op(*stacked).unwrap() else {
        unreachable!();
    };
    let parameter_sum = crate::rangeify::computed_view(&graph, inputs[0])
        .unwrap()
        .source;
    let Op::Reduce { input: squared, .. } = graph.op(parameter_sum).unwrap() else {
        panic!("stacked norm input must be one parameter reduction");
    };
    assert!(matches!(
        graph.op(*squared).unwrap(),
        Op::Binary {
            op: crate::BinaryOp::Mul,
            lhs,
            rhs,
        } if lhs == &committed_gradient && rhs == &committed_gradient
    ));
}

#[test]
fn guarded_cpu_preparation_rejects_non_finite_initial_and_restored_frontiers() {
    let initial = CompiledAdamWPlan::compile(
        non_finite_config(1),
        [TrainingParameterInit::new("weight", TensorData::scalar(f32::INFINITY)).unwrap()],
        |graph, _, parameters| {
            let loss = graph.square(parameters["weight"])?;
            Ok((loss, BTreeMap::new()))
        },
    )
    .unwrap();
    let error = initial
        .prepare(&rejecting_cpu_target())
        .err()
        .expect("guarded preparation must reject a non-finite initial frontier");
    assert!(
        error
            .to_string()
            .contains("non-finite prepared recurrent state")
    );

    let plan = non_finite_plan(1);
    let mut propagating = plan.prepare(&CpuSessionTarget).unwrap();
    propagating
        .step(scalar_batch(0.0), TensorData::scalar(0.01))
        .unwrap();
    let restored = plan
        .restore_checkpoint(&propagating.checkpoint().unwrap())
        .unwrap();
    let error = restored
        .prepare(&rejecting_cpu_target())
        .err()
        .expect("guarded preparation must reject a non-finite restored frontier");
    assert!(
        error
            .to_string()
            .contains("non-finite prepared recurrent state")
    );

    let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
    let before = interpreted.checkpoint().unwrap();
    assert!(
        interpreted
            .restore_checkpoint_in_place(&propagating.checkpoint().unwrap())
            .is_err()
    );
    assert_eq!(interpreted.checkpoint().unwrap(), before);
    interpreted.restore_checkpoint_in_place(&before).unwrap();
    assert_eq!(interpreted.checkpoint().unwrap(), before);

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native = plan.prepare(&target).unwrap();
    let before = native.checkpoint().unwrap();
    let plan_count = executor.native_item_plan_count();
    assert!(
        native
            .restore_checkpoint_in_place(&propagating.checkpoint().unwrap())
            .is_err()
    );
    assert_eq!(native.checkpoint().unwrap(), before);
    assert_eq!(executor.native_item_plan_count(), plan_count);
    native.restore_checkpoint_in_place(&before).unwrap();
    assert_eq!(native.checkpoint().unwrap(), before);
    assert_eq!(executor.native_item_plan_count(), plan_count);
}

#[test]
fn cpu_non_finite_policy_preserves_default_identity_and_rejects_before_commit() {
    let plan = non_finite_plan(1);
    let legacy = plan.prepare(&CpuSessionTarget).unwrap();
    let explicit = plan
        .prepare(&CpuSessionTarget.with_non_finite_policy(CpuNonFinitePolicy::Propagate))
        .unwrap();
    let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
    assert_eq!(legacy.capture_identity(), plan.capture_identity());
    assert_eq!(explicit.capture_identity(), plan.capture_identity());
    assert_eq!(legacy.checkpoint().unwrap(), explicit.checkpoint().unwrap());
    assert_eq!(legacy.checkpoint().unwrap(), guarded.checkpoint().unwrap());
    assert_eq!(legacy.non_finite_policy(), CpuNonFinitePolicy::Propagate);
    assert_eq!(
        guarded.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );

    let before = guarded.checkpoint().unwrap();
    let error = guarded
        .step(scalar_batch(0.0), TensorData::scalar(0.01))
        .unwrap_err();
    assert!(error.to_string().contains("non-finite recurrent successor"));
    assert_eq!(guarded.checkpoint().unwrap(), before);
    assert_eq!(guarded.step_count(), 0);

    for invalid_rate in [f32::NAN, f32::INFINITY] {
        let error = guarded
            .step(scalar_batch(1.0), TensorData::scalar(invalid_rate))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("non-finite external learning rate")
        );
        assert_eq!(guarded.checkpoint().unwrap(), before);
    }
    let mut reference = plan.prepare(&rejecting_cpu_target()).unwrap();
    let expected = reference
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    let actual = guarded
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(
        guarded.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );
}

#[test]
fn cpu_non_finite_policy_checks_loss_but_not_named_outputs() {
    let parameter = TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap();
    let plan = CompiledAdamWPlan::compile(
        non_finite_config(1),
        [parameter],
        |graph, inputs, parameters| {
            let loss = graph.square(parameters["weight"])?;
            let one = scalar_f32(graph, 1.0)?;
            let diagnostic = graph.div(one, inputs["x"])?;
            Ok((loss, BTreeMap::from([("diagnostic".into(), diagnostic)])))
        },
    )
    .unwrap();
    let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
    let result = guarded
        .step(scalar_batch(0.0), TensorData::scalar(0.01))
        .unwrap();
    assert!(result.output("diagnostic").unwrap().values()[0].is_infinite());

    let mut guarded = non_finite_plan(1).prepare(&rejecting_cpu_target()).unwrap();
    let before = guarded.checkpoint().unwrap();
    let error = guarded
        .step(scalar_batch(f32::NAN), TensorData::scalar(0.01))
        .unwrap_err();
    assert!(error.to_string().contains("non-finite loss"));
    assert_eq!(guarded.checkpoint().unwrap(), before);
}

#[test]
fn non_finite_clip_report_rejects_atomically_and_retries_on_both_cpu_paths() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_clip_report()
        .with_input("x", [], DType::F32)
        .unwrap();
    let plan = CompiledAdamWPlan::compile(
        config,
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, inputs, parameters| {
            Ok((
                graph.mul(parameters["weight"], inputs["x"])?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap();
    let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
    let before = interpreted.checkpoint().unwrap();
    let error = interpreted
        .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
        .expect_err("non-finite interpreted clip report must reject");
    assert!(error.to_string().contains("non-finite clip report"));
    assert_eq!(interpreted.checkpoint().unwrap(), before);
    assert_eq!(interpreted.step_count(), 0);

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native = target.prepare(&plan).unwrap();
    let error = match native.step(scalar_batch(f32::MAX), TensorData::scalar(0.01)) {
        Ok(_) => panic!("non-finite native clip report must reject"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("non-finite clip report"));
    assert_eq!(native.checkpoint().unwrap(), before);
    assert_eq!(native.successful_steps, 0);

    let interpreted_step = interpreted
        .step(scalar_batch(3.0), TensorData::scalar(0.01))
        .unwrap();
    let native_step = native
        .step(scalar_batch(3.0), TensorData::scalar(0.01))
        .unwrap();
    let report = interpreted_step.clip_report().unwrap();
    assert_eq!(report.pre_clip_global_norm(), 3.0);
    assert_eq!(report.applied_scale(), 1.0 / 3.0);
    assert!(report.is_finite());
    assert_eq!(report.did_clip(), Some(true));
    assert_eq!(native_step.clip_report(), Some(report));
    assert_eq!(native.successful_steps, 1);
    assert_eq!(
        native.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );
}

#[test]
fn unreported_global_norm_overflow_matches_both_cpu_paths() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_input("x", [], DType::F32)
        .unwrap();
    let plan = CompiledAdamWPlan::compile(
        config,
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, inputs, parameters| {
            Ok((
                graph.mul(parameters["weight"], inputs["x"])?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap();
    let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native = target.prepare(&plan).unwrap();

    // MAX is finite: its F32 squared norm overflows, scale becomes zero,
    // and the finite gradient clips to zero. Both final frontiers are valid.
    let interpreted_step = interpreted
        .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
        .unwrap();
    let native_step = native
        .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
        .unwrap();
    assert!(interpreted_step.clip_report().is_none());
    assert!(native_step.clip_report().is_none());
    assert_eq!(interpreted_step.optimizer_step(), 1);
    assert_eq!(native_step.optimizer_step(), 1);
    assert_eq!(native.successful_steps, 1);
    assert_eq!(
        native.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );
}

#[test]
fn accumulation_only_clip_report_overflow_is_discarded_before_safe_commit() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_clip_report()
        .with_input("x", [], DType::F32)
        .unwrap();
    let plan = CompiledAdamWPlan::compile(
        config,
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, inputs, parameters| {
            Ok((
                graph.mul(parameters["weight"], inputs["x"])?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap();
    let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native = target.prepare(&plan).unwrap();

    let interpreted_first = interpreted
        .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
        .unwrap();
    let native_first = native
        .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
        .unwrap();
    assert!(interpreted_first.clip_report().is_none());
    assert!(native_first.clip_report().is_none());
    assert_eq!(interpreted_first.accumulation_index(), 1);
    assert_eq!(native_first.accumulation_index(), 1);
    assert_eq!(
        native.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );

    let interpreted_commit = interpreted
        .step(scalar_batch(-f32::MAX), TensorData::scalar(0.01))
        .unwrap();
    let native_commit = native
        .step(scalar_batch(-f32::MAX), TensorData::scalar(0.01))
        .unwrap();
    let report = interpreted_commit.clip_report().unwrap();
    assert_eq!(report.pre_clip_global_norm(), 0.0);
    assert_eq!(report.applied_scale(), 1.0);
    assert!(report.is_finite());
    assert_eq!(report.did_clip(), Some(false));
    assert_eq!(native_commit.clip_report(), Some(report));
    assert_eq!(interpreted_commit.optimizer_step(), 1);
    assert_eq!(native_commit.optimizer_step(), 1);
    assert_eq!(native.successful_steps, 2);
    assert_eq!(
        native.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );
}

#[test]
fn cpu_non_finite_rejection_preserves_partial_dropout_window_and_flush() {
    let config = module_config().with_gradient_accumulation(2).unwrap();
    let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([41, 43]));
    let module = TiedFrozenModule::new([0.1, -0.2]);
    let plan = CompiledAdamWPlan::compile_module_with_dropout(
        config,
        dropout,
        &module,
        build_tied_dropout_with_input_guard,
    )
    .unwrap();
    let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
    let mut reference = plan.prepare(&rejecting_cpu_target()).unwrap();
    let finite = || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
    guarded.step(finite(), TensorData::scalar(0.01)).unwrap();
    reference.step(finite(), TensorData::scalar(0.01)).unwrap();
    let partial = guarded.checkpoint().unwrap();
    assert_eq!(guarded.dropout_block_counter().unwrap(), Some(1));
    let error = guarded
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.0, 2.0]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap_err();
    assert!(error.to_string().contains("non-finite loss"));
    assert_eq!(guarded.checkpoint().unwrap(), partial);
    assert_eq!(guarded.dropout_block_counter().unwrap(), Some(1));
    let expected = reference.step(finite(), TensorData::scalar(0.01)).unwrap();
    let actual = guarded.step(finite(), TensorData::scalar(0.01)).unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());
    assert_eq!(
        guarded.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );

    let flush_plan = CompiledAdamWPlan::compile(
        accumulated_adamw_config(2),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    let mut guarded = flush_plan.prepare(&rejecting_cpu_target()).unwrap();
    let mut reference = flush_plan.prepare(&rejecting_cpu_target()).unwrap();
    guarded.step(batch(), lr()).unwrap();
    reference.step(batch(), lr()).unwrap();
    let partial = guarded.checkpoint().unwrap();
    assert!(
        guarded
            .flush_partial_window(TensorData::scalar(f32::NAN))
            .is_err()
    );
    assert_eq!(guarded.checkpoint().unwrap(), partial);
    assert_eq!(
        guarded.flush_partial_window(lr()).unwrap().optimizer_step(),
        1
    );
    reference.flush_partial_window(lr()).unwrap();
    assert_eq!(
        guarded.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );
}

#[test]
fn non_finite_partial_flush_successors_reject_atomically_on_both_cpu_paths() {
    let plan = non_finite_flush_plan();
    let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
    let mut reference = plan.prepare(&rejecting_cpu_target()).unwrap();
    guarded
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    reference
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    let before = guarded.checkpoint().unwrap();
    let error = guarded
        .flush_partial_window(TensorData::scalar(f32::MAX))
        .unwrap_err();
    assert!(error.to_string().contains("non-finite recurrent successor"));
    assert_eq!(guarded.checkpoint().unwrap(), before);
    assert_eq!(guarded.optimizer_step().unwrap(), 0);
    assert_eq!(guarded.accumulation_index().unwrap(), 1);
    let actual = guarded
        .flush_partial_window(TensorData::scalar(0.01))
        .unwrap();
    let expected = reference
        .flush_partial_window(TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.clip_report(), expected.clip_report());
    assert!(actual.clip_report().is_some());
    assert_eq!(
        guarded.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native = target.prepare(&plan).unwrap();
    let mut native_reference = target.prepare(&plan).unwrap();
    native
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    native_reference
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    let before = native.checkpoint().unwrap();
    let error = native
        .flush_partial_window(TensorData::scalar(f32::MAX))
        .err()
        .expect("non-finite native partial flush must reject");
    assert!(error.to_string().contains("non-finite recurrent successor"));
    assert_eq!(native.checkpoint().unwrap(), before);
    assert_eq!(native.inner.optimizer_step().unwrap(), 0);
    assert_eq!(native.inner.accumulation_index().unwrap(), 1);
    assert_eq!(native.successful_flushes, 0);
    let actual = native
        .flush_partial_window(TensorData::scalar(0.01))
        .unwrap();
    let expected = native_reference
        .flush_partial_window(TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        actual.flushed_microbatches(),
        expected.flushed_microbatches()
    );
    assert_eq!(actual.optimizer_step(), expected.optimizer_step());
    assert_eq!(actual.clip_report(), expected.clip_report());
    assert!(actual.clip_report().is_some());
    assert_eq!(actual.report().unwrap().successful_invocation(), 1);
    assert_eq!(native.successful_flushes, 1);
    assert_eq!(
        native.checkpoint().unwrap(),
        native_reference.checkpoint().unwrap()
    );
}

#[test]
fn native_cpu_non_finite_rejection_is_atomic_and_retryable() {
    let plan = non_finite_plan(1);
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native = target.prepare(&plan).unwrap();
    assert_eq!(
        native.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );
    let prepared_workspace = native.main_replay.workspace_stats();
    assert_eq!(prepared_workspace.borrowed_external_input_bytes, 0);
    let before = native.checkpoint().unwrap();
    assert!(
        native
            .step(scalar_batch(0.0), TensorData::scalar(0.01))
            .is_err()
    );
    assert_eq!(native.checkpoint().unwrap(), before);
    assert_eq!(native.successful_steps, 0);
    let rejected_workspace = native.main_replay.workspace_stats();
    assert_eq!(rejected_workspace.input_import_count, 0);
    assert_eq!(rejected_workspace.borrowed_external_input_bytes, 8);
    assert!(
        native
            .step(scalar_batch(1.0), TensorData::scalar(f32::INFINITY))
            .is_err()
    );
    assert_eq!(native.checkpoint().unwrap(), before);
    assert_eq!(native.successful_steps, 0);
    assert_eq!(native.main_replay.workspace_stats(), rejected_workspace);

    let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
    let expected = interpreted
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    let actual = native
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    assert_cross_engine_tensor_close("guarded retry loss", actual.loss(), expected.loss());
    assert_eq!(actual.report().successful_invocation(), 1);
    assert!(actual.report().executed_native_item_count() > 0);
    assert!(actual.report().executed_native_item_count() <= actual.report().native_item_count());
    assert!(actual.report().skipped_output_clear_count() > 0);
    assert!(actual.report().skipped_output_clear_count() <= actual.report().native_item_count());
    assert!(actual.report().module_dispatch_count() > 0);
    assert_eq!(
        actual.report().module_dispatched_native_item_count(),
        actual.report().executed_native_item_count()
    );
    assert_eq!(actual.report().traffic().external_input_import_count(), 0);
    assert_eq!(actual.report().traffic().external_input_import_bytes(), 0);
    assert_eq!(native.successful_steps, 1);
    assert_native_adamw_state_close(&native, &interpreted);
}

fn assert_cross_engine_tensor_close(label: &str, actual: &TensorData, expected: &TensorData) {
    assert_eq!(actual.shape(), expected.shape(), "{label} shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{label} dtype");
    assert_eq!(actual.dtype(), DType::F32, "{label} comparison dtype");
    for index in 0..actual.len() {
        let actual = actual.scalar_at(index).as_f64();
        let expected = expected.scalar_at(index).as_f64();
        let error = (actual - expected).abs();
        assert!(
            error <= 1e-5,
            "{label}[{index}] mismatch: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn assert_cross_engine_tensor_maps_close(
    label: &str,
    actual: &BTreeMap<String, TensorData>,
    expected: &BTreeMap<String, TensorData>,
) {
    assert_eq!(actual.len(), expected.len(), "{label} key count");
    for (name, expected) in expected {
        let actual = actual
            .get(name)
            .unwrap_or_else(|| panic!("{label} missing {name}"));
        assert_cross_engine_tensor_close(&format!("{label} {name}"), actual, expected);
    }
}

fn assert_native_adamw_state_close(
    native: &NativeCpuCompiledAdamW<'_>,
    interpreted: &CpuCompiledAdamW,
) {
    assert_cross_engine_tensor_maps_close(
        "parameters",
        &native.parameter_snapshots().unwrap(),
        &interpreted.parameter_snapshots().unwrap(),
    );
    assert_cross_engine_tensor_maps_close(
        "first moments",
        &native.first_moment_snapshots().unwrap(),
        &interpreted.first_moment_snapshots().unwrap(),
    );
    assert_cross_engine_tensor_maps_close(
        "second moments",
        &native.second_moment_snapshots().unwrap(),
        &interpreted.second_moment_snapshots().unwrap(),
    );
    assert_cross_engine_tensor_maps_close(
        "gradient accumulators",
        &native.gradient_accumulator_snapshots().unwrap(),
        &interpreted.gradient_accumulator_snapshots().unwrap(),
    );
    let native_checkpoint = native.checkpoint().unwrap();
    let interpreted_checkpoint = interpreted.checkpoint().unwrap();
    assert_eq!(native_checkpoint.info(), interpreted_checkpoint.info());
}

fn native_recurrent_test_counts(native: &NativeCpuCompiledAdamW<'_>) -> (usize, usize) {
    native.inner.inner.runtime.recurrent_test_counts()
}

fn assert_native_run_timing(report: &NativeCpuRunReport) {
    assert_eq!(
        report
            .native_dispatcher_wall_time()
            .checked_add(report.executor_host_wall_time())
            .unwrap(),
        report.executor_wall_time()
    );
    assert_eq!(
        report
            .executor_wall_time()
            .checked_add(report.replay_overhead_wall_time())
            .unwrap(),
        report.wall_time()
    );
}

fn assert_no_hot_phase_capture_work() {
    let counts = crate::engine::mixed_capture::prepared_replay_validation_counts();
    assert_eq!(counts.cursor_projection_preparations, 0);
    assert_eq!(counts.mixed_capture_validations, 0);
    assert_eq!(counts.schedule_rekeys, 0);
    assert_eq!(counts.identity_serializations, 0);
    assert_eq!(counts.recurrent_frontier_plans, 0);
}

#[test]
fn fresh_checkpoint_restore_retains_compile_phase_observation() {
    let plan =
        CompiledAdamWPlan::compile(adamw_config(), initial_parameters(), build_tinybob).unwrap();
    let observation = plan.compile_phases().cloned().unwrap();
    let checkpoint = plan.prepare_cpu().unwrap().checkpoint().unwrap();
    let restored = plan.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(restored.compile_phases(), Some(&observation));
}

#[test]
fn native_cpu_adamw_prepares_strictly_reuses_cache_and_commits_atomically() {
    let plan =
        CompiledAdamWPlan::compile(adamw_config(), initial_parameters(), build_tinybob).unwrap();
    let inspection = plan.inspection().unwrap();
    let compile_phases = inspection
        .compile_phases()
        .expect("fresh compilation retains phase evidence");
    assert_eq!(compile_phases.compile_count(), 1);
    assert!(compile_phases.accumulation_capture().is_none());
    assert!(compile_phases.partial_flush().is_none());
    assert!(compile_phases.zero_grad().is_none());
    assert!(compile_phases.evaluation().is_none());
    let compile_wall_time = compile_phases.measured_wall_time().unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let mut native = target.prepare(&plan).unwrap();
    let mut scoreboard = crate::NativeTrainingScoreboard::new(
        inspection,
        native.preparation_report(),
        compile_wall_time,
        native.preparation_report().main().wall_time(),
    )
    .unwrap();
    assert_eq!(executor.native_item_plan_count(), 1);
    assert_eq!(native.main_replay.structure_validation_count(), 1);
    let workspace = native.main_replay.workspace_stats();
    assert!(workspace.allocation_count > 0);
    assert_eq!(workspace.input_import_count, 0);
    assert_eq!(workspace.intermediate_materialization_count, 0);
    assert_eq!(workspace.borrowed_recurrent_input_bytes, 0);
    assert_eq!(workspace.borrowed_recurrent_output_bytes, 0);
    assert!(workspace.sealed_dispatch_step_count > 0);
    assert!(workspace.sealed_dispatch_segment_count > 0);
    assert!(
        workspace.sealed_derived_materialization_count > 0,
        "the real compiled AdamW workspace must seal derived inputs as dispatcher actions"
    );
    assert_eq!(workspace.dispatch_metadata_build_count, 1);
    assert_eq!(workspace.dispatch_scratch_capacity_growth_count, 0);
    assert!(workspace.dispatch_scratch_is_empty);
    let preparation = native.preparation_report();
    assert_eq!(
        preparation.main().capture_identity(),
        plan.capture_identity()
    );
    assert!(preparation.main().native_item_count() > 0);
    assert_eq!(
        preparation.main().cache_hit_count() + preparation.main().cache_miss_count(),
        preparation.main().native_item_count()
    );
    assert_eq!(
        preparation.main().work().rendered_entry_count(),
        preparation.main().native_item_count()
    );
    assert_eq!(preparation.main().work().loaded_module_count(), 1);
    assert!(preparation.main().work().compiler_invocation_count() <= 1);
    assert!(preparation.compiler_process_count() <= 1);
    assert!(preparation.max_parallel_compiler_process_count() <= 1);
    assert_eq!(
        preparation.parallel_module_overlap_wall_time(),
        Duration::ZERO
    );
    assert!(preparation.max_parallel_render_job_count() <= 1);
    assert_eq!(
        preparation.parallel_render_overlap_wall_time(),
        Duration::ZERO
    );
    let phases = preparation.main().phases();
    assert_eq!(
        phases
            .layout_wall_time()
            .checked_add(phases.render_wall_time())
            .and_then(|elapsed| elapsed.checked_add(phases.compiler_process_wall_time()))
            .and_then(|elapsed| elapsed.checked_add(phases.module_load_wall_time()))
            .and_then(|elapsed| elapsed.checked_add(phases.residual_wall_time())),
        Some(preparation.main().wall_time())
    );
    assert!(preparation.main().cache_miss_count() > 0);
    assert!(preparation.partial_flush().is_none());
    assert!(preparation.zero_grad().is_none());
    assert!(preparation.evaluation().is_none());
    assert!(preparation.recurrent_state_count() > 0);
    assert!(preparation.recurrent_state_bytes() > 0);
    let recurrent_state_bytes = preparation.recurrent_state_bytes();
    let prepared_native_identity = preparation.main().native_identity();

    let cached = target.prepare(&plan).unwrap();
    assert_eq!(executor.native_item_plan_count(), 2);
    assert_eq!(cached.preparation_report().main().cache_miss_count(), 0);
    assert_eq!(
        cached
            .preparation_report()
            .main()
            .work()
            .compiler_invocation_count(),
        0
    );
    assert_eq!(
        cached
            .preparation_report()
            .main()
            .phases()
            .compiler_process_wall_time(),
        Duration::ZERO
    );
    assert_eq!(
        cached
            .preparation_report()
            .main()
            .phases()
            .module_load_wall_time(),
        Duration::ZERO
    );
    assert_eq!(
        cached.preparation_report().main().native_identity(),
        prepared_native_identity
    );

    let mut interpreted = plan.prepare_cpu().unwrap();
    let expected = interpreted.step(batch(), lr()).unwrap();
    let before_replay = native_recurrent_test_counts(&native);
    let actual = native.step(batch(), lr()).unwrap();
    let after_replay = native_recurrent_test_counts(&native);
    assert_eq!(after_replay.0, before_replay.0);
    assert_eq!(after_replay.1, before_replay.1 + 1);
    assert_cross_engine_tensor_close("first loss", actual.loss(), expected.loss());
    assert_cross_engine_tensor_maps_close("first outputs", actual.outputs(), expected.outputs());
    assert_eq!(actual.step(), expected.step());
    assert_eq!(actual.optimizer_step(), expected.optimizer_step());
    assert_eq!(actual.accumulation_index(), expected.accumulation_index());
    assert_eq!(expected.loss_weight(), 1);
    assert_eq!(actual.loss_weight(), 1);
    assert_eq!(actual.report().successful_invocation(), 1);
    assert!(actual.report().first_successful_invocation());
    assert_native_run_timing(actual.report());
    assert_eq!(actual.report().native_identity(), prepared_native_identity);
    assert!(actual.report().executed_native_item_count() > 0);
    assert!(actual.report().executed_native_item_count() <= actual.report().native_item_count());
    let first_executed_native_item_count = actual.report().executed_native_item_count();
    let first_traffic = *actual.report().traffic();
    assert_eq!(first_traffic.external_input_import_count(), 1);
    assert_eq!(first_traffic.external_input_import_bytes(), 32);
    assert!(first_traffic.materialized_egress_count() > 0);
    assert!(first_traffic.materialized_egress_bytes() > 0);
    assert_eq!(
        first_traffic.borrowed_recurrent_input_bytes(),
        u64::try_from(recurrent_state_bytes).unwrap()
    );
    assert_eq!(
        first_traffic.borrowed_recurrent_output_bytes(),
        u64::try_from(recurrent_state_bytes).unwrap()
    );
    let mut malformed_report = actual.report().clone();
    malformed_report.traffic.borrowed_recurrent_output_bytes -= 1;
    assert!(scoreboard.record(&malformed_report).is_err());
    let mut malformed_report = actual.report().clone();
    malformed_report.executed_native_item_count = malformed_report.native_item_count + 1;
    assert!(scoreboard.record(&malformed_report).is_err());
    let mut malformed_report = actual.report().clone();
    malformed_report.skipped_output_clear_count = malformed_report.native_item_count + 1;
    assert!(validate_native_cpu_run_report(&malformed_report).is_err());
    let mut malformed_report = actual.report().clone();
    malformed_report.module_dispatched_native_item_count =
        malformed_report.executed_native_item_count + 1;
    assert!(validate_native_cpu_run_report(&malformed_report).is_err());
    let mut malformed_report = actual.report().clone();
    malformed_report.module_dispatch_count =
        malformed_report.module_dispatched_native_item_count + 1;
    assert!(validate_native_cpu_run_report(&malformed_report).is_err());
    let mut malformed_report = actual.report().clone();
    malformed_report.executor_wall_time = malformed_report
        .wall_time
        .checked_add(Duration::from_nanos(1))
        .unwrap();
    assert!(validate_native_cpu_run_report(&malformed_report).is_err());
    assert!(scoreboard.record(&malformed_report).is_err());
    let mut malformed_report = actual.report().clone();
    malformed_report.native_dispatcher_wall_time = malformed_report
        .executor_wall_time
        .checked_add(Duration::from_nanos(1))
        .unwrap();
    assert!(validate_native_cpu_run_report(&malformed_report).is_err());
    assert!(scoreboard.record(&malformed_report).is_err());
    scoreboard.record(actual.report()).unwrap();
    assert!(scoreboard.record_step(&actual).is_err());
    assert_native_adamw_state_close(&native, &interpreted);
    let first_workspace = native.main_replay.workspace_stats();
    assert_eq!(first_workspace.allocation_count, workspace.allocation_count);
    assert_eq!(first_workspace.input_import_count, 1);
    assert_eq!(first_workspace.borrowed_external_input_bytes, 36);
    assert_eq!(first_workspace.intermediate_materialization_count, 0);
    assert_eq!(
        first_workspace.borrowed_recurrent_input_bytes,
        recurrent_state_bytes
    );
    assert_eq!(
        first_workspace.borrowed_recurrent_output_bytes,
        recurrent_state_bytes
    );

    let before_failure = native.checkpoint().unwrap();
    let before_failure_counts = native_recurrent_test_counts(&native);
    assert!(native.step_inner(batch(), lr(), Some(0)).is_err());
    let after_failure_counts = native_recurrent_test_counts(&native);
    assert_eq!(after_failure_counts.0, before_failure_counts.0);
    assert_eq!(after_failure_counts.1, before_failure_counts.1);
    assert_eq!(native.checkpoint().unwrap(), before_failure);
    let failed_workspace = native.main_replay.workspace_stats();
    assert_eq!(
        failed_workspace.allocation_count,
        workspace.allocation_count
    );
    assert_eq!(failed_workspace.intermediate_materialization_count, 0);
    assert_eq!(failed_workspace.input_import_count, 2);
    assert_eq!(failed_workspace.borrowed_external_input_bytes, 72);
    assert_eq!(
        failed_workspace.borrowed_recurrent_input_bytes,
        first_workspace.borrowed_recurrent_input_bytes + recurrent_state_bytes
    );
    assert_eq!(
        failed_workspace.borrowed_recurrent_output_bytes,
        first_workspace.borrowed_recurrent_output_bytes + recurrent_state_bytes
    );
    let expected = interpreted.step(batch(), lr()).unwrap();
    let before_retry = native_recurrent_test_counts(&native);
    let actual = native.step(batch(), lr()).unwrap();
    let after_retry = native_recurrent_test_counts(&native);
    assert_eq!(after_retry.0, before_retry.0);
    assert_eq!(after_retry.1, before_retry.1 + 1);
    assert_cross_engine_tensor_close("retry loss", actual.loss(), expected.loss());
    assert_cross_engine_tensor_maps_close("retry outputs", actual.outputs(), expected.outputs());
    assert_eq!(expected.loss_weight(), 1);
    assert_eq!(actual.loss_weight(), 1);
    assert_eq!(actual.report().successful_invocation(), 2);
    assert_eq!(
        actual.report().executed_native_item_count(),
        first_executed_native_item_count
    );
    assert_eq!(actual.report().traffic(), &first_traffic);
    let mut changed_execution_report = actual.report().clone();
    changed_execution_report.executed_native_item_count = first_executed_native_item_count - 1;
    assert!(scoreboard.record(&changed_execution_report).is_err());
    scoreboard.record(actual.report()).unwrap();
    assert_eq!(
        scoreboard.report().unwrap().main_replay_traffic().unwrap(),
        &first_traffic
    );
    assert_eq!(
        scoreboard
            .report()
            .unwrap()
            .main_replay_executed_native_item_count(),
        Some(u64::try_from(first_executed_native_item_count).unwrap())
    );
    let raw_json: serde_json::Value =
        serde_json::from_slice(&scoreboard.report().unwrap().to_json_bytes().unwrap()).unwrap();
    assert_eq!(
        raw_json["format_version"],
        crate::NATIVE_TRAINING_REPORT_FORMAT_VERSION
    );
    assert!(raw_json.get("step_phases").is_none());
    assert_eq!(
        raw_json["main_replay_traffic"]["materialized_egress_count"],
        first_traffic.materialized_egress_count()
    );
    assert_eq!(
        raw_json["main_replay_traffic"]["materialized_egress_bytes"],
        first_traffic.materialized_egress_bytes()
    );
    assert_native_adamw_state_close(&native, &interpreted);
    assert_eq!(executor.native_item_plan_count(), 2);
    let retried_workspace = native.main_replay.workspace_stats();
    assert_eq!(
        retried_workspace.allocation_count,
        workspace.allocation_count
    );
    assert_eq!(retried_workspace.input_import_count, 3);
    assert_eq!(retried_workspace.borrowed_external_input_bytes, 108);
    assert_eq!(retried_workspace.intermediate_materialization_count, 0);
    assert_eq!(
        retried_workspace.borrowed_recurrent_input_bytes,
        failed_workspace.borrowed_recurrent_input_bytes + recurrent_state_bytes
    );
    assert_eq!(
        retried_workspace.borrowed_recurrent_output_bytes,
        failed_workspace.borrowed_recurrent_output_bytes + recurrent_state_bytes
    );
    assert_eq!(native.main_replay.structure_validation_count(), 1);
    assert_eq!(retried_workspace.dispatch_metadata_build_count, 1);
    assert_eq!(retried_workspace.dispatch_scratch_capacity_growth_count, 0);
    assert!(retried_workspace.dispatch_scratch_is_empty);
}

#[test]
fn native_cpu_module_dispatch_failures_keep_recurrent_publication_atomic() {
    let plan =
        CompiledAdamWPlan::compile(adamw_config(), initial_parameters(), build_tinybob).unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let item_count = plan
        .prepare(&target)
        .unwrap()
        .preparation_report()
        .main()
        .native_item_count();
    assert!(item_count >= 3);

    for index in [0, item_count / 2, item_count - 1] {
        let mut native = plan.prepare(&target).unwrap();
        let prepared_workspace = native.main_replay.workspace_stats();
        assert_eq!(prepared_workspace.dispatch_metadata_build_count, 1);
        assert_eq!(prepared_workspace.dispatch_scratch_capacity_growth_count, 0);
        assert!(prepared_workspace.dispatch_scratch_is_empty);
        let checkpoint = native.checkpoint().unwrap();
        let cursor = native.inner.inner.cursor.clone();
        let counts = native_recurrent_test_counts(&native);
        native.main_replay.inject_dispatch_failure(index);
        let error = match native.step(batch(), lr()) {
            Ok(_) => panic!("injected dispatcher failure must reject"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains(&format!("native schedule item {index}")),
            "failure must retain the logical schedule ordinal: {error}"
        );
        assert_eq!(native.inner.inner.cursor, cursor);
        assert_eq!(native_recurrent_test_counts(&native), counts);
        assert_eq!(native.checkpoint().unwrap(), checkpoint);
        let rejected_workspace = native.main_replay.workspace_stats();
        assert_eq!(rejected_workspace.dispatch_metadata_build_count, 1);
        assert_eq!(rejected_workspace.dispatch_scratch_capacity_growth_count, 0);
        assert!(rejected_workspace.dispatch_scratch_is_empty);

        let replay = native.step(batch(), lr()).unwrap();
        assert!(replay.report().module_dispatch_count() > 0);
        assert_eq!(
            replay.report().module_dispatched_native_item_count(),
            replay.report().executed_native_item_count()
        );
        let retried_workspace = native.main_replay.workspace_stats();
        assert_eq!(retried_workspace.dispatch_metadata_build_count, 1);
        assert_eq!(retried_workspace.dispatch_scratch_capacity_growth_count, 0);
        assert!(retried_workspace.dispatch_scratch_is_empty);
    }
}

#[test]
fn adamw_lowering_resolves_ordered_recurrent_store_keys() {
    let single = CompiledTrainingWindowTopology::from_validated_parts(1, false, false);
    let accumulating = CompiledTrainingWindowTopology::from_validated_parts(3, false, false);
    assert!(adamw_recurrent_store_group_specs(["weight".to_string()].iter(), single).is_empty());
    let specs = adamw_recurrent_store_group_specs(["weight".to_string()].iter(), accumulating);
    let expected_keys = vec![
        RecurrentStateKey::parameter("weight"),
        RecurrentStateKey::adamw_parameter("weight", AdamWParameterState::FirstMoment),
        RecurrentStateKey::adamw_parameter("weight", AdamWParameterState::SecondMoment),
        RecurrentStateKey::adamw_parameter("weight", AdamWParameterState::GradientAccumulator),
    ];
    assert_eq!(specs[0].members, expected_keys);

    let mut graph = Graph::new();
    let nodes = [
        graph.input("parameter", [2]),
        graph.input("first_moment", [2]),
        graph.input("second_moment", [2]),
        graph.input("accumulator", [2]),
    ];
    let updates = expected_keys
        .iter()
        .cloned()
        .zip(nodes)
        .collect::<BTreeMap<_, _>>();
    let state_buffers = expected_keys
        .iter()
        .cloned()
        .zip([101, 102, 103, 104])
        .collect::<BTreeMap<_, _>>();
    let manifests = resolve_recurrent_store_groups(&specs, &updates, &state_buffers).unwrap();
    assert_eq!(
        manifests[0]
            .members
            .iter()
            .map(|member| (member.output, member.state_buffer))
            .collect::<Vec<_>>(),
        nodes
            .iter()
            .zip([101, 102, 103, 104])
            .map(|(node, buffer)| (node.index() as u64, buffer))
            .collect::<Vec<_>>()
    );

    let mut missing = state_buffers;
    missing.remove(expected_keys.last().unwrap());
    let error = match resolve_recurrent_store_groups(&specs, &updates, &missing) {
        Ok(_) => panic!("missing recurrent store state resolved"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("recurrent store state is absent")
    );
}

#[test]
fn training_window_progress_supports_momentum_style_accumulation_transitions() {
    let steps = 3;
    let initial = CompiledTrainingWindowProgress::INITIAL;
    let first = initial.advance_replay(steps).unwrap();
    let second = first.advance_replay(steps).unwrap();
    assert_eq!((second.replay_step, second.optimizer_step), (2, 0));
    assert_eq!(second.accumulation_index, 2);

    // Admission is pure: a failed optimizer replay does not publish the
    // computed transition, and retry derives the identical commit.
    let rejected_commit = second.advance_replay(steps).unwrap();
    assert_eq!(second.accumulation_index, 2);
    let committed = second.advance_replay(steps).unwrap();
    assert_eq!(committed, rejected_commit);
    assert_eq!(
        (
            committed.replay_step,
            committed.optimizer_step,
            committed.accumulation_index,
        ),
        (3, 1, 0)
    );

    let pending_flush = committed.advance_replay(steps).unwrap();
    let (flushed, flush) = pending_flush.flush_partial(steps).unwrap();
    assert_eq!(flush.flushed_microbatches, 1);
    assert_eq!(flush.optimizer_step, 2);
    assert_eq!(
        (
            flushed.replay_step,
            flushed.optimizer_step,
            flushed.accumulation_index,
        ),
        (4, 2, 0)
    );

    let pending_reset = flushed.advance_replay(steps).unwrap();
    let (reset, transition) = pending_reset.cancel(steps).unwrap();
    assert_eq!(transition.discarded_microbatches, 1);
    let reset = reset.record_reset_transition().unwrap();
    assert_eq!(
        (
            reset.replay_step,
            reset.optimizer_step,
            reset.accumulation_index,
            reset.discarded_microbatches,
            reset.reset_transition_count,
        ),
        (5, 2, 0, 1, 1)
    );
    validate_training_window_progress(reset, steps).unwrap();
}

#[test]
fn adamw_window_topology_matrix_matches_state_phases_checkpoint_and_artifact() {
    #[derive(Clone, Copy, Debug)]
    enum TokenPolicy {
        None,
        ExplicitMask,
        IgnoreIndex,
    }

    for accumulation_steps in [1, 3] {
        for token_policy in [
            TokenPolicy::None,
            TokenPolicy::ExplicitMask,
            TokenPolicy::IgnoreIndex,
        ] {
            for window_loss_report in [false, true] {
                let mut config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
                    .unwrap()
                    .with_gradient_accumulation(accumulation_steps)
                    .unwrap();
                config = match token_policy {
                    TokenPolicy::None => config,
                    TokenPolicy::ExplicitMask => config
                        .with_input("features", [3], DType::F32)
                        .unwrap()
                        .with_input("mask", [3], DType::F32)
                        .unwrap()
                        .with_token_weighted_gradient_accumulation("mask")
                        .unwrap(),
                    TokenPolicy::IgnoreIndex => config
                        .with_input("features", [3], DType::F32)
                        .unwrap()
                        .with_input("targets", [3], DType::I32)
                        .unwrap()
                        .with_token_weighted_ignore_index("targets", -100)
                        .unwrap(),
                };
                if window_loss_report {
                    config = config.with_window_loss_report();
                }
                let topology = CompiledTrainingWindowTopology::from_config(&config);
                let accumulating = accumulation_steps > 1;
                let token_weighted = !matches!(token_policy, TokenPolicy::None);
                assert_eq!(topology.accumulating(), accumulating);
                assert_eq!(
                    topology.retains_token_count(),
                    accumulating && token_weighted
                );
                assert_eq!(topology.retains_window_numerator(), window_loss_report);

                let owner = CompiledModuleAdamWPlan::compile_graph(
                        config,
                        TokenMeanModule::new(),
                        |module, graph, inputs| {
                            let weight = module.weight.bind(graph)?;
                            match token_policy {
                                TokenPolicy::None => Ok(CompiledAdamWGraph::scalar(
                                    graph.square(weight)?,
                                    BTreeMap::new(),
                                )),
                                TokenPolicy::ExplicitMask | TokenPolicy::IgnoreIndex => {
                                    let losses = graph.mul(weight, inputs["features"])?;
                                    Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
                                }
                            }
                        },
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "topology matrix failed for steps={accumulation_steps}, policy={token_policy:?}, report={window_loss_report}: {error}"
                        )
                    });
                let has_state =
                    |key: RecurrentStateKey| owner.plan.inner.state_values.contains_key(&key);
                assert!(has_state(RecurrentStateKey::parameter("weight")));
                assert!(has_state(RecurrentStateKey::adamw_parameter(
                    "weight",
                    AdamWParameterState::FirstMoment,
                )));
                assert!(has_state(RecurrentStateKey::adamw_parameter(
                    "weight",
                    AdamWParameterState::SecondMoment,
                )));
                assert_eq!(
                    has_state(RecurrentStateKey::adamw_parameter(
                        "weight",
                        AdamWParameterState::GradientAccumulator,
                    )),
                    accumulating
                );
                assert!(has_state(RecurrentStateKey::adamw_global(
                    AdamWGlobalState::Step,
                )));
                assert_eq!(
                    has_state(RecurrentStateKey::adamw_global(
                        AdamWGlobalState::AccumulationIndex,
                    )),
                    accumulating
                );
                assert_eq!(
                    has_state(RecurrentStateKey::adamw_global(
                        AdamWGlobalState::AccumulatedTokenCount,
                    )),
                    accumulating && token_weighted
                );
                assert_eq!(
                    has_state(RecurrentStateKey::adamw_global(
                        AdamWGlobalState::AccumulatedLossNumerator,
                    )),
                    window_loss_report
                );
                assert_eq!(owner.plan.inner.accumulation.is_some(), accumulating);
                assert_eq!(owner.plan.partial_flush.is_some(), accumulating);
                assert_eq!(owner.plan.zero_grad.is_some(), accumulating);
                if accumulating {
                    assert!(
                        owner
                            .plan
                            .inner
                            .accumulation
                            .as_ref()
                            .unwrap()
                            .phase()
                            .retains_unchanged()
                    );
                    assert!(
                        !owner
                            .plan
                            .partial_flush
                            .as_ref()
                            .unwrap()
                            .phase()
                            .retains_unchanged()
                    );
                    assert!(
                        !owner
                            .plan
                            .zero_grad
                            .as_ref()
                            .unwrap()
                            .phase()
                            .retains_unchanged()
                    );
                }
                assert_eq!(
                    owner.plan.inner.recurrent_store_groups.len(),
                    accumulating as usize
                );
                assert_eq!(
                    owner
                        .plan
                        .partial_flush
                        .as_ref()
                        .map_or(0, |phase| phase.phase().store_groups().len()),
                    accumulating as usize
                );
                assert_eq!(
                    owner
                        .plan
                        .zero_grad
                        .as_ref()
                        .map_or(0, |phase| phase.phase().store_groups().len()),
                    0
                );

                let checkpoint = owner.plan.prepare_cpu().unwrap().checkpoint().unwrap();
                let decoded = decode_adamw_checkpoint(checkpoint.as_bytes()).unwrap();
                assert_eq!(decoded.gradient_accumulators.is_empty(), !accumulating);
                assert_eq!(
                    decoded.accumulated_token_count.is_some(),
                    accumulating && token_weighted
                );
                assert_eq!(
                    decoded.accumulated_loss_numerator.is_some(),
                    window_loss_report
                );
                assert_eq!(
                    decoded.accumulation_capture_identity.is_some(),
                    accumulating
                );
                assert_eq!(decoded.flush_capture_identity.is_some(), accumulating);

                let artifact = owner.program_artifact().unwrap();
                program_artifact::rewrite_json_for_test(&artifact, |json| {
                    assert_eq!(json["accumulation"].is_null(), !accumulating);
                    assert_eq!(json["partial_flush"].is_null(), !accumulating);
                    assert_eq!(json["zero_grad"].is_null(), !accumulating);
                    assert_eq!(
                        json["main"]["phase"]["adamw_native_updates"]
                            .as_array()
                            .unwrap()
                            .len(),
                        accumulating as usize
                    );
                    if accumulating {
                        assert_eq!(
                            json["partial_flush"]["adamw_native_updates"]
                                .as_array()
                                .unwrap()
                                .len(),
                            1
                        );
                        assert!(
                            json["accumulation"]["adamw_native_updates"]
                                .as_array()
                                .unwrap()
                                .is_empty()
                        );
                        assert!(
                            json["zero_grad"]["adamw_native_updates"]
                                .as_array()
                                .unwrap()
                                .is_empty()
                        );
                    }
                });
            }
        }
    }
}

#[test]
fn native_cpu_adamw_state_update_groups_preserve_logical_failure_atomicity() {
    let plan = CompiledAdamWPlan::compile(
        accumulated_adamw_config(2),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    let partial_flush = plan.partial_flush.as_ref().unwrap();
    for manifest in partial_flush.phase().store_groups() {
        assert_eq!(manifest.members.len(), 4);
        let accumulator = manifest.members.last().unwrap();
        let item = partial_flush
            .phase()
            .capture
            .schedule
            .items
            .iter()
            .find(|item| item.primary_output().id == accumulator.output)
            .unwrap();
        assert!(
            matches!(item.kernel.operation(), crate::Operation::Sink)
                && matches!(
                    item.kernel.sources(),
                    [store, end_range]
                        if matches!(store.operation(), crate::Operation::Store)
                            && matches!(end_range.operation(), crate::Operation::EndRange)
                ),
            "partial-flush accumulator reset must retain a scalar native root"
        );
    }
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let native = plan.prepare(&target).unwrap();
    let main_updates = native.main_replay.recurrent_store_group_indices();
    let main_update_admissions = native
        .main_replay
        .recurrent_store_group_admission_diagnostics();
    let flush_updates = native
        .partial_flush_replay
        .as_ref()
        .unwrap()
        .recurrent_store_group_indices();
    let flush_update_admissions = native
        .partial_flush_replay
        .as_ref()
        .unwrap()
        .recurrent_store_group_admission_diagnostics();
    assert!(
        main_updates.len() == 2
            && main_updates.iter().all(|group| group.len() == 4)
            && flush_updates.len() == 2
            && flush_updates.iter().all(|group| group.len() == 4),
        "AdamW native update admissions:\nmain: {main_update_admissions:#?}\npartial flush: {flush_update_admissions:#?}"
    );
    let preparation = native.preparation_report();
    assert_eq!(
        preparation.main().native_item_count() - preparation.main().work().rendered_entry_count(),
        main_updates.len() * 3
    );
    let flush = preparation.partial_flush().unwrap();
    assert_eq!(
        flush.native_item_count() - flush.work().rendered_entry_count(),
        flush_updates.len() * 3
    );

    for &index in &main_updates[0] {
        let mut native = plan.prepare(&target).unwrap();
        let mut interpreted = plan.prepare_cpu().unwrap();
        let expected = interpreted.step(batch(), lr()).unwrap();
        let actual = native.step(batch(), lr()).unwrap();
        assert_cross_engine_tensor_close(
            "AdamW accumulation before grouped update",
            actual.loss(),
            expected.loss(),
        );
        let checkpoint = native.checkpoint().unwrap();
        let cursor = native.inner.inner.cursor.clone();
        let counts = native_recurrent_test_counts(&native);
        native.main_replay.inject_dispatch_failure(index);
        let error = match native.step(batch(), lr()) {
            Ok(_) => panic!("injected native store-group failure must be reported"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains(&format!("native schedule item {index}"))
        );
        assert_eq!(native.inner.inner.cursor, cursor);
        assert_eq!(native_recurrent_test_counts(&native), counts);
        assert_eq!(native.checkpoint().unwrap(), checkpoint);
        let expected = interpreted.step(batch(), lr()).unwrap();
        let actual = native.step(batch(), lr()).unwrap();
        assert_cross_engine_tensor_close(
            "AdamW state update group retry loss",
            actual.loss(),
            expected.loss(),
        );
        assert_native_adamw_state_close(&native, &interpreted);
    }
}

#[test]
fn native_cpu_adamw_retains_transposed_parameter_storage() {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_input("x", [2, 3], DType::F32)
        .unwrap()
        .with_input("y", [2, 3], DType::F32)
        .unwrap();
    let parameter = TrainingParameterInit::new(
        "weight",
        TensorData::new(
            [4, 3],
            vec![
                0.2, -0.1, 0.3, 0.4, 0.05, -0.2, -0.3, 0.25, 0.1, 0.15, -0.4, 0.35,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
        let weight = graph.permute(parameters["weight"], [1, 0])?;
        let output = graph.matmul(inputs["x"], weight)?;
        let second = graph.matmul(inputs["y"], weight)?;
        let combined = graph.add(output, second)?;
        let loss = graph.sum_all(combined)?;
        Ok((loss, BTreeMap::from([("output".into(), output)])))
    })
    .unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = target.prepare(&plan).unwrap();
    let mut interpreted = plan.prepare_cpu().unwrap();
    let prepared = native.main_replay.workspace_stats();
    let accumulation_prepared = native
        .accumulation_replay
        .as_ref()
        .unwrap()
        .workspace_stats();
    assert!(prepared.retained_transpose_matmul_input_count >= 2);
    assert!(accumulation_prepared.retained_transpose_matmul_input_count >= 2);
    assert_eq!(prepared.affine_matmul_materialization_bytes, 0);
    assert_eq!(accumulation_prepared.affine_matmul_materialization_bytes, 0);
    assert_eq!(native.preparation_report().main().fallback_count(), 0);

    let batch = BTreeMap::from([
        (
            "x".into(),
            TensorData::new([2, 3], vec![1.0, -0.5, 0.25, -1.0, 0.75, 0.5]).unwrap(),
        ),
        (
            "y".into(),
            TensorData::new([2, 3], vec![0.5, 0.25, -1.0, 0.75, -0.5, 1.0]).unwrap(),
        ),
    ]);
    let before = native.checkpoint().unwrap();
    assert!(
        native
            .step_inner(batch.clone(), TensorData::scalar(0.01), Some(0))
            .is_err()
    );
    assert_eq!(native.checkpoint().unwrap(), before);
    assert_eq!(
        native
            .accumulation_replay
            .as_ref()
            .unwrap()
            .workspace_stats()
            .allocation_count,
        accumulation_prepared.allocation_count
    );
    assert_eq!(
        native
            .accumulation_replay
            .as_ref()
            .unwrap()
            .workspace_stats()
            .affine_matmul_materialization_bytes,
        0
    );

    let expected = interpreted
        .step(batch.clone(), TensorData::scalar(0.01))
        .unwrap();
    let actual = native
        .step(batch.clone(), TensorData::scalar(0.01))
        .unwrap();
    assert_cross_engine_tensor_close("retained transpose loss", actual.loss(), expected.loss());
    assert_cross_engine_tensor_maps_close(
        "retained transpose output",
        actual.outputs(),
        expected.outputs(),
    );
    let expected = interpreted
        .step(batch.clone(), TensorData::scalar(0.01))
        .unwrap();
    let actual = native.step(batch, TensorData::scalar(0.01)).unwrap();
    assert_cross_engine_tensor_close(
        "retained transpose update loss",
        actual.loss(),
        expected.loss(),
    );
    assert_native_adamw_state_close(&native, &interpreted);
    let replayed = native.main_replay.workspace_stats();
    assert_eq!(replayed.allocation_count, prepared.allocation_count);
    assert_eq!(replayed.affine_matmul_materialization_bytes, 0);
}

#[test]
fn native_cpu_adamw_materializes_duplicate_transposes_of_recurrent_parameter() {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_input("x", [2, 2], DType::F32)
        .unwrap();
    let parameter = TrainingParameterInit::new(
        "weight",
        TensorData::new([2, 2], vec![0.2, -0.1, 0.3, 0.4]).unwrap(),
    )
    .unwrap();
    let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
        let lhs = graph.permute(parameters["weight"], [1, 0])?;
        let rhs = graph.permute(parameters["weight"], [1, 0])?;
        let product = graph.matmul(lhs, rhs)?;
        let weighted = graph.mul(product, inputs["x"])?;
        let loss = graph.sum_all(weighted)?;
        Ok((loss, BTreeMap::from([("output".into(), product)])))
    })
    .unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = target.prepare(&plan).unwrap();
    let mut interpreted = plan.prepare_cpu().unwrap();
    assert_eq!(native.preparation_report().main().fallback_count(), 0);

    let batch = BTreeMap::from([(
        "x".into(),
        TensorData::new([2, 2], vec![1.0, -0.5, 0.25, 2.0]).unwrap(),
    )]);
    let expected = interpreted
        .step(batch.clone(), TensorData::scalar(0.01))
        .unwrap();
    let actual = native.step(batch, TensorData::scalar(0.01)).unwrap();
    assert_cross_engine_tensor_close(
        "duplicate recurrent transpose loss",
        actual.loss(),
        expected.loss(),
    );
    assert_cross_engine_tensor_maps_close(
        "duplicate recurrent transpose output",
        actual.outputs(),
        expected.outputs(),
    );
    assert_native_adamw_state_close(&native, &interpreted);
    assert!(actual.report().traffic().borrowed_recurrent_input_bytes() > 0);
    assert!(actual.report().traffic().borrowed_recurrent_output_bytes() > 0);
}

#[test]
fn native_cpu_adamw_materializes_transpose_colliding_with_dense_recurrent_parameter() {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_input("x", [2, 2], DType::F32)
        .unwrap();
    let parameter = TrainingParameterInit::new(
        "weight",
        TensorData::new([2, 2], vec![0.2, -0.1, 0.3, 0.4]).unwrap(),
    )
    .unwrap();
    let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
        let transposed = graph.permute(parameters["weight"], [1, 0])?;
        let product = graph.matmul(parameters["weight"], transposed)?;
        let weighted = graph.mul(product, inputs["x"])?;
        let loss = graph.sum_all(weighted)?;
        Ok((loss, BTreeMap::from([("output".into(), product)])))
    })
    .unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = target.prepare(&plan).unwrap();
    let mut interpreted = plan.prepare_cpu().unwrap();
    assert_eq!(native.preparation_report().main().fallback_count(), 0);

    let batch = BTreeMap::from([(
        "x".into(),
        TensorData::new([2, 2], vec![1.0, -0.5, 0.25, 2.0]).unwrap(),
    )]);
    let expected = interpreted
        .step(batch.clone(), TensorData::scalar(0.01))
        .unwrap();
    let actual = native.step(batch, TensorData::scalar(0.01)).unwrap();
    assert_cross_engine_tensor_close(
        "dense recurrent transpose loss",
        actual.loss(),
        expected.loss(),
    );
    assert_cross_engine_tensor_maps_close(
        "dense recurrent transpose output",
        actual.outputs(),
        expected.outputs(),
    );
    assert_native_adamw_state_close(&native, &interpreted);
    assert!(actual.report().traffic().borrowed_recurrent_input_bytes() > 0);
    assert!(actual.report().traffic().borrowed_recurrent_output_bytes() > 0);
}

#[test]
fn native_cpu_adamw_partial_flush_and_zero_grad_match_interpreter() {
    let plan = CompiledAdamWPlan::compile(
        accumulated_adamw_config(3),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    let accumulation_transition = plan.inner.accumulation.as_ref().unwrap();
    assert!(
        accumulation_transition
            .phase()
            .capture
            .schedule
            .inputs
            .iter()
            .all(|input| input.name != LEARNING_RATE_INPUT),
        "the accumulation-only graph must not retain external learning-rate work"
    );
    let reset_transition = plan.zero_grad.as_ref().unwrap();
    assert_eq!(
        reset_transition.phase().state_buffers.len(),
        initial_parameters().len() + 1
    );
    assert!(
        reset_transition
            .phase()
            .state_buffers
            .keys()
            .all(RecurrentStateKey::is_accumulation_reset_state)
    );
    let mut overflow = plan.prepare_cpu().unwrap();
    overflow.step(batch(), lr()).unwrap();
    let overflow_frontier = overflow.inner.plan().unwrap();
    let mut overflow_versions = overflow_frontier.state_versions.clone();
    for key in reset_transition.phase().state_buffers.keys() {
        overflow_versions.insert(key.clone(), u64::MAX);
    }
    overflow
        .inner
        .restore_frontier(
            overflow_frontier.step,
            &overflow_frontier.state_values,
            &overflow_versions,
        )
        .unwrap();
    let before_overflow_cursor = overflow.inner.cursor.clone();
    let before_overflow_progress = overflow.progress;
    let before_overflow_runtime = overflow.inner.runtime.recurrent_test_counts();
    let overflow_error = match overflow.zero_grad() {
        Ok(_) => panic!("overflowing zero-grad transition was accepted"),
        Err(error) => error,
    };
    assert_eq!(
        overflow_error,
        Error::SessionTraining {
            reason: "compiled auxiliary state version would overflow".into(),
        }
    );
    assert_eq!(overflow.inner.cursor, before_overflow_cursor);
    assert_eq!(overflow.progress, before_overflow_progress);
    assert_eq!(
        overflow.inner.runtime.recurrent_test_counts(),
        before_overflow_runtime
    );
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native = target.prepare(&plan).unwrap();
    assert_eq!(executor.native_item_plan_count(), 4);
    let expected_recurrent_state_count = plan.inspection().unwrap().recurrent_state_count();
    let main_layout = native.main_replay.recurrent_bank_layout_evidence();
    let main_buffers = main_layout.buffers;
    let main_inputs = main_layout.input_ordinals;
    assert_eq!(main_buffers.len(), expected_recurrent_state_count);
    assert!(
        main_buffers
            .windows(2)
            .all(|buffers| buffers[0] < buffers[1])
    );
    let reset_buffers = reset_transition
        .phase()
        .state_buffers
        .values()
        .copied()
        .collect::<BTreeSet<_>>();
    let expected_reset_source_ordinals = main_buffers
        .iter()
        .enumerate()
        .filter_map(|(ordinal, buffer)| reset_buffers.contains(buffer).then_some(ordinal))
        .collect::<Vec<_>>();
    assert_eq!(
        reset_transition.phase().cursor_projection.source_ordinals(),
        expected_reset_source_ordinals.as_slice()
    );
    assert!(
        expected_reset_source_ordinals
            .windows(2)
            .any(|ordinals| ordinals[1] > ordinals[0] + 1),
        "zero-grad must exercise noncontiguous source frontier ordinals"
    );
    let mut sorted_main_inputs = main_inputs.clone();
    sorted_main_inputs.sort_unstable();
    assert_eq!(
        sorted_main_inputs,
        (0..expected_recurrent_state_count).collect::<Vec<_>>()
    );
    assert_eq!(
        main_layout.retained,
        vec![false; expected_recurrent_state_count]
    );
    assert_eq!(
        (main_layout.retained_count, main_layout.retained_bytes),
        (0, 0)
    );
    let accumulation_workspace = native
        .accumulation_replay
        .as_ref()
        .unwrap()
        .workspace_stats();
    let accumulation_layout = native
        .accumulation_replay
        .as_ref()
        .unwrap()
        .recurrent_bank_layout_evidence();
    assert_eq!(accumulation_layout.buffers, main_buffers);
    assert_eq!(accumulation_layout.input_ordinals, main_inputs);
    let expected_retained_buffers = accumulation_transition
        .phase()
        .state_buffers
        .iter()
        .filter(|(key, _)| !key.is_accumulation_reset_state())
        .map(|(_, buffer)| *buffer)
        .collect::<BTreeSet<_>>();
    assert!(!expected_retained_buffers.is_empty());
    assert_eq!(
        expected_retained_buffers.len(),
        initial_parameters().len() * 3 + 1
    );
    assert_eq!(
        main_buffers
            .iter()
            .zip(accumulation_layout.retained)
            .filter_map(|(buffer, retained)| retained.then_some(*buffer))
            .collect::<BTreeSet<_>>(),
        expected_retained_buffers
    );
    assert_eq!(
        accumulation_layout.retained_count,
        u64::try_from(expected_retained_buffers.len()).unwrap()
    );
    assert!(accumulation_layout.retained_bytes > 0);
    assert_eq!(
        native
            .accumulation_replay
            .as_ref()
            .unwrap()
            .retained_recurrent_state_count(),
        expected_retained_buffers.len()
    );
    assert_eq!(
        native
            .accumulation_replay
            .as_ref()
            .unwrap()
            .retained_recurrent_state_buffers(),
        expected_retained_buffers
    );
    let flush_workspace = native
        .partial_flush_replay
        .as_ref()
        .unwrap()
        .workspace_stats();
    let reset_workspace = native.zero_grad_replay.as_ref().unwrap().workspace_stats();
    for workspace in [
        native.main_replay.workspace_stats(),
        accumulation_workspace,
        flush_workspace,
        reset_workspace,
    ] {
        assert!(workspace.sealed_dispatch_step_count > 0);
        assert!(workspace.sealed_dispatch_segment_count > 0);
        assert_eq!(workspace.dispatch_metadata_build_count, 1);
        assert_eq!(workspace.dispatch_scratch_capacity_growth_count, 0);
        assert!(workspace.dispatch_scratch_is_empty);
    }
    assert_eq!(native.main_replay.structure_validation_count(), 1);
    assert_eq!(
        native
            .accumulation_replay
            .as_ref()
            .unwrap()
            .structure_validation_count(),
        1
    );
    assert_eq!(
        native
            .partial_flush_replay
            .as_ref()
            .unwrap()
            .structure_validation_count(),
        1
    );
    assert_eq!(
        native
            .zero_grad_replay
            .as_ref()
            .unwrap()
            .structure_validation_count(),
        1
    );
    assert!(flush_workspace.allocation_count > 0);
    assert!(accumulation_workspace.allocation_count > 0);
    assert!(reset_workspace.allocation_count > 0);
    let mut interpreted = plan
        .prepare_cpu_with_non_finite_policy(CpuNonFinitePolicy::RejectTransition)
        .unwrap();
    assert_eq!(
        CompiledTrainingWindowResetRuntime::gradient_window_reset_capture_identity(&native),
        native.zero_grad_capture_identity()
    );
    assert_eq!(
        CompiledTrainingWindowResetRuntime::gradient_window_reset_capture_identity(&interpreted),
        interpreted.zero_grad_capture_identity()
    );
    assert!(native.preparation_report().partial_flush().is_some());
    assert!(native.preparation_report().zero_grad().is_some());
    let accumulation_preparation = native.preparation_report().accumulation().unwrap();
    assert_eq!(
        accumulation_preparation.capture_identity(),
        plan.accumulation_capture_identity().unwrap()
    );
    assert_ne!(
        accumulation_preparation.capture_identity(),
        plan.capture_identity()
    );
    assert!(
        accumulation_preparation.native_item_count()
            < native.preparation_report().main().native_item_count()
    );
    assert!(native.preparation_report().compiler_process_count() <= 6);
    assert!(
        native
            .preparation_report()
            .max_parallel_compiler_process_count()
            <= 2
    );
    for program in [
        Some(native.preparation_report().main()),
        native.preparation_report().accumulation(),
        native.preparation_report().partial_flush(),
        native.preparation_report().zero_grad(),
    ]
    .into_iter()
    .flatten()
    {
        assert!(program.work().rendered_entry_count() <= program.native_item_count());
        assert_eq!(
            program.work().loaded_module_count(),
            usize::from(program.work().unique_rendered_entry_count() != 0)
        );
        assert!(program.work().compiler_invocation_count() <= 3);
    }

    let initial_checkpoint = native.checkpoint().unwrap();
    let initial_interpreted_checkpoint = interpreted.checkpoint().unwrap();
    let invalid_native_learning_rate =
        match native.step_inner(batch(), TensorData::scalar(f32::NAN), None) {
            Ok(_) => panic!("non-finite native accumulation learning rate was accepted"),
            Err(error) => error,
        };
    assert!(
        invalid_native_learning_rate
            .to_string()
            .contains("non-finite")
    );
    let invalid_interpreted_learning_rate = interpreted
        .step_inner(batch(), TensorData::scalar(f32::NAN), None)
        .unwrap_err();
    assert!(
        invalid_interpreted_learning_rate
            .to_string()
            .contains("non-finite")
    );
    assert_eq!(native.checkpoint().unwrap(), initial_checkpoint);
    assert_eq!(
        interpreted.checkpoint().unwrap(),
        initial_interpreted_checkpoint
    );
    assert_eq!(native.successful_steps, 0);
    assert!(native.step_inner(batch(), lr(), Some(0)).is_err());
    assert_eq!(native.checkpoint().unwrap(), initial_checkpoint);
    assert_eq!(native.successful_steps, 0);

    let parameter_values = native.parameter_snapshots().unwrap();
    let first_moments = native.first_moment_snapshots().unwrap();
    let second_moments = native.second_moment_snapshots().unwrap();
    let state_versions = native.inner.inner.plan().unwrap().state_versions;
    crate::engine::mixed_capture::reset_prepared_replay_validation_counts();
    crate::host_buffer::reset_host_bank_transaction_test_counts();
    let actual = native.step(batch(), lr()).unwrap();
    assert_no_hot_phase_capture_work();
    assert_eq!(
        crate::host_buffer::host_bank_transaction_test_counts(),
        crate::host_buffer::HostBankTransactionTestCounts {
            ordered_full_frontier_transactions: 1,
            request_map_builds: 0,
            ordinal_sorts: 0,
        }
    );
    let expected = interpreted.step(batch(), lr()).unwrap();
    assert_core_training_window_step(&actual, 1, false);
    assert_core_training_window_step(&expected, 1, false);
    assert_core_training_window(&native, 3, 1);
    assert_core_training_window(&interpreted, 3, 1);
    assert!(!actual.did_update());
    assert_eq!(actual.capture_identity(), plan.capture_identity());
    assert!(actual.clip_report().is_none());
    let traffic = actual.report().traffic();
    assert_eq!(traffic.retained_recurrent_state_count(), 7);
    assert_eq!(traffic.replaced_recurrent_state_count(), 3);
    assert_eq!(
        traffic
            .retained_recurrent_state_bytes()
            .checked_add(traffic.replaced_recurrent_state_bytes()),
        Some(traffic.borrowed_recurrent_input_bytes())
    );
    assert_eq!(
        traffic.replaced_recurrent_state_bytes(),
        traffic.borrowed_recurrent_output_bytes()
    );
    assert!(actual.report().executed_native_item_count() < actual.report().native_item_count());
    assert_eq!(
        actual.report().capture_identity(),
        plan.accumulation_capture_identity().unwrap()
    );
    assert_eq!(native.parameter_snapshots().unwrap(), parameter_values);
    assert_eq!(native.first_moment_snapshots().unwrap(), first_moments);
    assert_eq!(native.second_moment_snapshots().unwrap(), second_moments);
    let next_versions = native.inner.inner.plan().unwrap().state_versions;
    assert_eq!(next_versions.len(), state_versions.len());
    for (key, version) in state_versions {
        assert_eq!(next_versions[&key], version + 1, "{key:?}");
    }
    assert_cross_engine_tensor_close("cancelled-window loss", actual.loss(), expected.loss());
    assert_cross_engine_tensor_maps_close(
        "cancelled-window outputs",
        actual.outputs(),
        expected.outputs(),
    );
    let before_failed_reset = native.checkpoint().unwrap();
    let before_failed_reset_counts = native_recurrent_test_counts(&native);
    assert_eq!(native.successful_zero_grads, 0);
    crate::engine::mixed_capture::reset_prepared_replay_validation_counts();
    assert!(native.zero_grad_with_injected_failure(0).is_err());
    assert_no_hot_phase_capture_work();
    let after_failed_reset_counts = native_recurrent_test_counts(&native);
    assert_eq!(after_failed_reset_counts.0, before_failed_reset_counts.0);
    assert_eq!(after_failed_reset_counts.1, before_failed_reset_counts.1);
    assert_eq!(native.checkpoint().unwrap(), before_failed_reset);
    assert_eq!(native.successful_zero_grads, 0);
    assert_core_training_window(&native, 3, 1);
    let before_reset = native_recurrent_test_counts(&native);
    crate::engine::mixed_capture::reset_prepared_replay_validation_counts();
    let native_reset = reset_core_training_window(&mut native);
    assert_no_hot_phase_capture_work();
    assert_eq!(
        native_reset,
        reset_adamw_window_compatibility(&mut interpreted)
    );
    assert_core_training_window(&native, 3, 0);
    assert_core_training_window(&interpreted, 3, 0);
    let after_reset = native_recurrent_test_counts(&native);
    assert_eq!(after_reset.0, before_reset.0);
    assert_eq!(after_reset.1, before_reset.1 + 1);
    assert_eq!(native.successful_zero_grads, 1);
    assert!(
        native
            .zero_grad_replay
            .as_ref()
            .unwrap()
            .last_executed_native_item_count()
            > 0
    );
    let reset_replay = native.zero_grad_replay.as_ref().unwrap();
    let reset_dispatch = reset_replay.last_module_dispatch_counts();
    assert!(reset_dispatch.0 > 0);
    assert_eq!(
        reset_dispatch.1,
        reset_replay.last_executed_native_item_count()
    );
    assert_native_adamw_state_close(&native, &interpreted);
    let used_reset_workspace = native.zero_grad_replay.as_ref().unwrap().workspace_stats();
    assert_eq!(
        used_reset_workspace.allocation_count,
        reset_workspace.allocation_count
    );
    assert_eq!(used_reset_workspace.input_import_count, 0);
    assert!(used_reset_workspace.borrowed_recurrent_input_bytes > 0);
    assert_eq!(
        used_reset_workspace.borrowed_recurrent_input_bytes,
        used_reset_workspace.borrowed_recurrent_output_bytes
    );
    assert_eq!(used_reset_workspace.intermediate_materialization_count, 0);
    assert!(used_reset_workspace.skipped_output_clear_count > 0);

    let before_empty_reset = native.checkpoint().unwrap();
    let before_empty_reset_counts = native_recurrent_test_counts(&native);
    let before_empty_reset_workspace = native.zero_grad_replay.as_ref().unwrap().workspace_stats();
    assert!(!reset_core_training_window(&mut native).did_discard());
    assert_eq!(
        native_recurrent_test_counts(&native),
        before_empty_reset_counts
    );
    assert_eq!(native.successful_zero_grads, 1);
    assert_eq!(native.checkpoint().unwrap(), before_empty_reset);
    assert_eq!(
        native.zero_grad_replay.as_ref().unwrap().workspace_stats(),
        before_empty_reset_workspace
    );

    let actual = native.step(batch(), lr()).unwrap();
    let expected = interpreted.step(batch(), lr()).unwrap();
    assert_cross_engine_tensor_close("partial-window loss", actual.loss(), expected.loss());
    assert_cross_engine_tensor_maps_close(
        "partial-window outputs",
        actual.outputs(),
        expected.outputs(),
    );
    let before_failed_flush = native.checkpoint().unwrap();
    let before_failed_flush_counts = native_recurrent_test_counts(&native);
    assert_eq!(native.successful_flushes, 0);
    crate::engine::mixed_capture::reset_prepared_replay_validation_counts();
    assert!(
        native
            .flush_partial_window_with_injected_failure(lr(), 0)
            .is_err()
    );
    assert_no_hot_phase_capture_work();
    let after_failed_flush_counts = native_recurrent_test_counts(&native);
    assert_eq!(after_failed_flush_counts.0, before_failed_flush_counts.0);
    assert_eq!(after_failed_flush_counts.1, before_failed_flush_counts.1);
    assert_eq!(native.checkpoint().unwrap(), before_failed_flush);
    assert_eq!(native.successful_flushes, 0);
    let before_flush = native_recurrent_test_counts(&native);
    crate::engine::mixed_capture::reset_prepared_replay_validation_counts();
    let actual = commit_core_training_window(&mut native);
    assert_no_hot_phase_capture_work();
    let after_flush = native_recurrent_test_counts(&native);
    assert_eq!(after_flush.0, before_flush.0);
    assert_eq!(after_flush.1, before_flush.1 + 1);
    let expected = interpreted.flush_partial_window(lr()).unwrap();
    assert_eq!(actual.committed_microbatches(), 1);
    assert!(actual.did_commit_window());
    assert_eq!(
        actual.flushed_microbatches(),
        expected.flushed_microbatches()
    );
    assert_eq!(actual.optimizer_step(), expected.optimizer_step());
    assert_eq!(actual.report().unwrap().successful_invocation(), 1);
    assert_native_run_timing(actual.report().unwrap());
    assert!(actual.report().unwrap().executed_native_item_count() > 0);
    assert!(
        actual.report().unwrap().executed_native_item_count()
            <= actual.report().unwrap().native_item_count()
    );
    assert!(actual.report().unwrap().module_dispatch_count() > 0);
    assert!(actual.report().unwrap().skipped_output_clear_count() > 0);
    assert_eq!(
        actual
            .report()
            .unwrap()
            .module_dispatched_native_item_count(),
        actual.report().unwrap().executed_native_item_count()
    );
    let flush_replay = native.partial_flush_replay.as_ref().unwrap();
    assert_eq!(
        flush_replay.last_module_dispatch_counts().1,
        flush_replay.last_executed_native_item_count()
    );
    let flush_traffic = actual.report().unwrap().traffic();
    assert_eq!(flush_traffic.external_input_import_count(), 0);
    assert_eq!(flush_traffic.external_input_import_bytes(), 0);
    assert!(flush_traffic.borrowed_recurrent_input_bytes() > 0);
    assert_eq!(
        flush_traffic.borrowed_recurrent_input_bytes(),
        flush_traffic.borrowed_recurrent_output_bytes()
    );
    assert_native_adamw_state_close(&native, &interpreted);

    let before_empty_flush_workspace = native
        .partial_flush_replay
        .as_ref()
        .unwrap()
        .workspace_stats();
    let before_empty_flush_plan_count = executor.native_item_plan_count();
    let before_empty_flush_counts = native_recurrent_test_counts(&native);
    let empty = commit_core_training_window(&mut native);
    assert!(!empty.did_update());
    assert_eq!(empty.committed_microbatches(), 0);
    assert!(!empty.did_commit_window());
    assert!(empty.report().is_none());
    assert_eq!(
        native_recurrent_test_counts(&native),
        before_empty_flush_counts
    );
    assert_eq!(
        executor.native_item_plan_count(),
        before_empty_flush_plan_count
    );
    assert_eq!(
        native
            .partial_flush_replay
            .as_ref()
            .unwrap()
            .workspace_stats(),
        before_empty_flush_workspace
    );
    let used_flush_workspace = native
        .partial_flush_replay
        .as_ref()
        .unwrap()
        .workspace_stats();
    assert_eq!(
        used_flush_workspace.allocation_count,
        flush_workspace.allocation_count
    );
    assert_eq!(used_flush_workspace.input_import_count, 0);
    assert_eq!(used_flush_workspace.borrowed_external_input_bytes, 8);
    assert!(used_flush_workspace.borrowed_recurrent_input_bytes > 0);
    assert_eq!(
        used_flush_workspace.borrowed_recurrent_input_bytes,
        used_flush_workspace.borrowed_recurrent_output_bytes
    );
    assert_eq!(used_flush_workspace.intermediate_materialization_count, 0);
    assert_eq!(native.main_replay.structure_validation_count(), 1);
    assert_eq!(
        native
            .partial_flush_replay
            .as_ref()
            .unwrap()
            .structure_validation_count(),
        1
    );
    assert_eq!(
        native
            .zero_grad_replay
            .as_ref()
            .unwrap()
            .structure_validation_count(),
        1
    );
}

#[test]
fn native_cpu_compiled_multi_step_lr_matches_interpreter() {
    let config = accumulated_adamw_config(3)
        .with_captured_multi_step_lr(CompiledMultiStepLr::new(0.05, 0.5, [1]).unwrap());
    let plan = CompiledAdamWPlan::compile(config, initial_parameters(), build_tinybob).unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = target.prepare(&plan).unwrap();
    let mut interpreted = plan.prepare_cpu().unwrap();

    let before = native.checkpoint().unwrap();
    assert!(native.step(batch(), lr()).is_err());
    assert_eq!(native.checkpoint().unwrap(), before);
    assert!(
        native
            .step_with_learning_rate(batch(), None, CompiledStepOutputSelection::All, Some(0),)
            .is_err()
    );
    assert_eq!(native.checkpoint().unwrap(), before);
    for _ in 0..2 {
        let actual = run_core_rate_policy_step(&mut native);
        let expected = run_core_rate_policy_step(&mut interpreted);
        assert_cross_engine_tensor_close("scheduled loss", actual.loss(), expected.loss());
        assert_cross_engine_tensor_maps_close(
            "scheduled outputs",
            actual.outputs(),
            expected.outputs(),
        );
        assert_eq!(actual.report().traffic().external_input_import_count(), 1);
        assert_eq!(actual.report().traffic().external_input_import_bytes(), 32);
    }
    assert_eq!(
        native
            .main_replay
            .workspace_stats()
            .borrowed_external_input_bytes,
        0
    );
    assert_eq!(
        native
            .accumulation_replay
            .as_ref()
            .unwrap()
            .workspace_stats()
            .borrowed_external_input_bytes,
        96
    );
    let actual = commit_core_rate_policy_training_window(&mut native);
    let expected = interpreted.flush_partial_window_scheduled().unwrap();
    assert_eq!(
        actual.flushed_microbatches(),
        expected.flushed_microbatches()
    );
    assert_eq!(actual.optimizer_step(), expected.optimizer_step());
    assert_native_adamw_state_close(&native, &interpreted);
}

#[test]
fn native_cpu_adamw_evaluation_borrows_active_parameters_and_retries() {
    let plan = CompiledModuleAdamWPlan::compile(
        module_config(),
        TiedFrozenModule::new([0.1, -0.2]),
        build_tied_frozen,
    )
    .unwrap()
    .with_evaluation(build_tied_frozen)
    .unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut session = target.prepare(plan).unwrap();
    assert_eq!(executor.native_item_plan_count(), 2);
    let checkpoint = session.checkpoint().unwrap();
    let workspace = session
        .runtime()
        .evaluation_replay
        .as_ref()
        .unwrap()
        .plan
        .workspace_stats();
    assert!(workspace.allocation_count > 0);
    assert_eq!(workspace.input_import_count, 0);
    assert!(workspace.sealed_dispatch_step_count > 0);
    assert!(workspace.sealed_dispatch_segment_count > 0);
    assert_eq!(workspace.dispatch_metadata_build_count, 1);
    assert_eq!(workspace.dispatch_scratch_capacity_growth_count, 0);
    assert!(workspace.dispatch_scratch_is_empty);
    let before_invalid_counts = native_recurrent_test_counts(session.runtime());

    assert!(session.evaluate(BTreeMap::new()).is_err());
    assert_eq!(session.runtime().successful_evaluations, 0);
    assert_eq!(
        native_recurrent_test_counts(session.runtime()),
        before_invalid_counts
    );
    assert_eq!(
        session
            .runtime()
            .evaluation_replay
            .as_ref()
            .unwrap()
            .plan
            .workspace_stats(),
        workspace
    );
    let binding = session
        .runtime()
        .evaluation_replay
        .as_ref()
        .unwrap()
        .parameter_inputs[0]
        .clone();
    session
        .runtime
        .evaluation_replay
        .as_mut()
        .unwrap()
        .parameter_inputs[0]
        .buffer ^= 1;
    assert!(
        session
            .evaluate(BTreeMap::from([(
                "x".into(),
                TensorData::new([2], vec![1.0, 2.0]).unwrap(),
            )]))
            .is_err()
    );
    session
        .runtime
        .evaluation_replay
        .as_mut()
        .unwrap()
        .parameter_inputs[0] = binding;
    assert_eq!(session.runtime().successful_evaluations, 0);
    assert_eq!(
        native_recurrent_test_counts(session.runtime()),
        before_invalid_counts
    );
    assert_eq!(
        session
            .runtime()
            .evaluation_replay
            .as_ref()
            .unwrap()
            .plan
            .workspace_stats(),
        workspace
    );
    assert_eq!(session.checkpoint().unwrap(), checkpoint);
    let before_evaluation_counts = native_recurrent_test_counts(session.runtime());

    let evaluation = session
        .evaluate(BTreeMap::from([(
            "x".into(),
            TensorData::new([2], vec![1.0, 2.0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(evaluation.report().successful_invocation(), 1);
    assert!(evaluation.report().first_successful_invocation());
    assert_native_run_timing(evaluation.report());
    assert!(evaluation.report().executed_native_item_count() > 0);
    assert!(
        evaluation.report().executed_native_item_count() <= evaluation.report().native_item_count()
    );
    assert!(evaluation.report().module_dispatch_count() > 0);
    assert!(evaluation.report().skipped_output_clear_count() > 0);
    assert_eq!(
        evaluation.report().module_dispatched_native_item_count(),
        evaluation.report().executed_native_item_count()
    );
    assert_eq!(
        evaluation.report().traffic().external_input_import_count(),
        0
    );
    assert_eq!(
        evaluation.report().traffic().external_input_import_bytes(),
        0
    );
    assert_eq!(
        evaluation
            .report()
            .traffic()
            .borrowed_recurrent_input_bytes(),
        8
    );
    assert_eq!(
        evaluation
            .report()
            .traffic()
            .borrowed_recurrent_output_bytes(),
        0
    );
    assert_eq!(
        native_recurrent_test_counts(session.runtime()),
        before_evaluation_counts
    );
    let first_output = evaluation.outputs()["output"].clone();
    assert_eq!(session.checkpoint().unwrap(), checkpoint);
    assert_eq!(executor.native_item_plan_count(), 2);
    let evaluated_workspace = session
        .runtime()
        .evaluation_replay
        .as_ref()
        .unwrap()
        .plan
        .workspace_stats();
    assert_eq!(
        evaluated_workspace.allocation_count,
        workspace.allocation_count
    );
    assert_eq!(evaluated_workspace.input_import_count, 0);
    assert_eq!(evaluated_workspace.borrowed_external_input_bytes, 8);
    assert_eq!(evaluated_workspace.borrowed_recurrent_input_bytes, 8);
    assert_eq!(evaluated_workspace.borrowed_recurrent_output_bytes, 0);
    assert_eq!(evaluated_workspace.intermediate_materialization_count, 0);
    assert_eq!(evaluated_workspace.dispatch_metadata_build_count, 1);
    assert_eq!(
        evaluated_workspace.dispatch_scratch_capacity_growth_count,
        0
    );
    assert!(evaluated_workspace.dispatch_scratch_is_empty);

    session
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let updated = session.checkpoint().unwrap();
    let before_updated_evaluation = native_recurrent_test_counts(session.runtime());
    let updated_evaluation = session
        .evaluate(BTreeMap::from([(
            "x".into(),
            TensorData::new([2], vec![1.0, 2.0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(updated_evaluation.report().successful_invocation(), 2);
    assert_eq!(
        updated_evaluation.report().executed_native_item_count(),
        evaluation.report().executed_native_item_count()
    );
    assert_ne!(updated_evaluation.outputs()["output"], first_output);
    assert_eq!(
        native_recurrent_test_counts(session.runtime()),
        before_updated_evaluation
    );
    assert_eq!(session.checkpoint().unwrap(), updated);
    let updated_workspace = session
        .runtime()
        .evaluation_replay
        .as_ref()
        .unwrap()
        .plan
        .workspace_stats();
    assert_eq!(updated_workspace.borrowed_external_input_bytes, 16);
    assert_eq!(updated_workspace.borrowed_recurrent_input_bytes, 16);
    assert_eq!(updated_workspace.borrowed_recurrent_output_bytes, 0);
    assert_eq!(updated_workspace.dispatch_metadata_build_count, 1);
    assert_eq!(updated_workspace.dispatch_scratch_capacity_growth_count, 0);
    assert!(updated_workspace.dispatch_scratch_is_empty);
}

#[test]
fn native_cpu_adamw_rejects_unsupported_pure_items_during_preparation() {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .with_input("x", [4], DType::F32)
        .unwrap();
    let parameter = TrainingParameterInit::new("weight", TensorData::scalar(2.0)).unwrap();
    let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
        let loss = graph.square(parameters["weight"])?;
        let unsupported = graph.binary(crate::BinaryOp::Atan2, inputs["x"], inputs["x"])?;
        Ok((loss, BTreeMap::from([("unsupported".into(), unsupported)])))
    })
    .unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    assert!(target.prepare(&plan).is_err());
    assert_eq!(executor.compile_cache_len(false), 0);
    assert_eq!(plan.step_count(), 0);
}

fn assert_core_loss_aggregation_weight(step: &impl CompiledTrainingStep, expected: u64) {
    assert_eq!(step.loss_aggregation_weight(), expected);
}

fn assert_core_training_window_step(
    step: &impl CompiledTrainingWindowStep,
    pending_microbatches: u64,
    did_close: bool,
) {
    assert_eq!(step.pending_microbatch_count(), pending_microbatches);
    assert_eq!(step.did_close_gradient_window(), did_close);
}

fn assert_core_training_window<R: CompiledTrainingWindowRuntime>(
    runtime: &R,
    window_size: u64,
    pending_microbatches: u64,
) {
    assert_eq!(runtime.gradient_window_size(), window_size);
    assert_eq!(
        runtime.pending_microbatch_count().unwrap(),
        pending_microbatches
    );
}

fn run_core_training_step<R: CompiledTrainingRuntime>(
    runtime: &mut R,
) -> (TensorData, BTreeMap<String, TensorData>) {
    let identity = runtime.capture_identity();
    let before = runtime.parameter_snapshots().unwrap();
    let step = runtime.step_batch(TinyBobBatch(batch()), 0.05).unwrap();
    assert_eq!(step.step(), 1);
    assert_core_loss_aggregation_weight(&step, 1);
    assert_eq!(step.capture_identity(), identity);
    assert_eq!(step.output("logits"), step.outputs().get("logits"));
    assert_eq!(runtime.step_count(), 1);
    assert_eq!(runtime.capture_identity(), identity);
    assert_ne!(runtime.parameter_snapshots().unwrap(), before);
    (step.loss().clone(), step.outputs().clone())
}

fn run_core_commit_only_step<R: CompiledTrainingCommitOnlyRuntime>(runtime: &mut R) -> TensorData {
    let identity = runtime.capture_identity();
    let before = runtime.parameter_snapshots().unwrap();
    let step = runtime
        .commit_step_batch(TinyBobBatch(batch()), 0.05)
        .unwrap();
    assert_eq!(step.step(), 1);
    assert_core_loss_aggregation_weight(&step, 1);
    assert_eq!(step.capture_identity(), identity);
    assert!(step.outputs().is_empty());
    assert_eq!(runtime.step_count(), 1);
    assert_eq!(runtime.capture_identity(), identity);
    assert_ne!(runtime.parameter_snapshots().unwrap(), before);
    step.loss().clone()
}

fn run_core_rate_policy_step<R: CompiledTrainingRatePolicyRuntime>(runtime: &mut R) -> R::Step {
    runtime
        .step_batch_with_rate_policy(TinyBobBatch(batch()))
        .unwrap()
}

fn run_core_rate_policy_commit_only_step<R: CompiledTrainingRatePolicyCommitOnlyRuntime>(
    runtime: &mut R,
) -> R::Step {
    runtime
        .commit_step_batch_with_rate_policy(TinyBobBatch(batch()))
        .unwrap()
}

#[test]
fn owned_session_forwards_optimizer_neutral_training_capabilities() {
    let schedule = CompiledMultiStepLr::new(0.01, 0.5, [2]).unwrap();
    let plan = CompiledModuleAdamWPlan::compile_graph(
        module_config()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_captured_multi_step_lr(schedule),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    let mut session = session.map_runtime(|inner| OptimizerNeutralRuntimeProbe { inner });
    let input = || BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);

    assert_core_training_window(&session, 2, 0);
    let first =
        CompiledTrainingRatePolicyRuntime::step_with_rate_policy(&mut session, input()).unwrap();
    assert_core_training_window_step(&first, 1, false);
    assert_core_training_window(&session, 2, 1);

    let reset = CompiledTrainingWindowResetRuntime::reset_gradient_window(&mut session).unwrap();
    assert_eq!(reset.discarded_microbatches(), 1);
    assert_core_training_window(&session, 2, 0);

    let committed = CompiledTrainingRatePolicyCommitOnlyRuntime::commit_step_with_rate_policy(
        &mut session,
        input(),
    )
    .unwrap();
    assert!(committed.outputs().is_empty());
    assert_core_training_window(&session, 2, 1);

    let flush =
        CompiledTrainingRatePolicyWindowCommitRuntime::commit_partial_window_with_rate_policy(
            &mut session,
        )
        .unwrap();
    assert_eq!(flush.committed_microbatches(), 1);
    assert_core_training_window(&session, 2, 0);
}

fn reset_core_training_window<R: CompiledTrainingWindowRuntime>(
    runtime: &mut R,
) -> CompiledTrainingWindowReset {
    runtime.reset_gradient_window().unwrap()
}

fn commit_core_training_window<R: CompiledTrainingWindowCommitRuntime>(
    runtime: &mut R,
) -> R::WindowCommit {
    runtime.commit_partial_window(lr()).unwrap()
}

fn commit_core_rate_policy_training_window<R: CompiledTrainingRatePolicyWindowCommitRuntime>(
    runtime: &mut R,
) -> R::WindowCommit {
    runtime.commit_partial_window_with_rate_policy().unwrap()
}

fn reset_adamw_window_compatibility<R: CompiledAdamWRuntime>(
    runtime: &mut R,
) -> CompiledAdamWZeroGradResult {
    runtime.zero_grad().unwrap()
}

fn run_adamw_commit_only_compatibility<R: CompiledAdamWCommitOnlyRuntime>(runtime: &mut R) {
    assert_eq!(runtime.gradient_accumulation_steps(), 1);
    let step = runtime
        .step_batch_commit_only(TinyBobBatch(batch()), 0.05)
        .unwrap();
    assert_eq!(step.optimizer_step(), 1);
    assert!(step.outputs().is_empty());
}

struct TinyBobBatch(BTreeMap<String, TensorData>);

impl TinyBobBatch {
    const SCHEMA: [CompiledInputSpec; 2] = [
        CompiledInputSpec::new("target", &[4], DType::I64),
        CompiledInputSpec::new("x", &[4, 2], DType::F32),
    ];
}

impl CompiledInputBatch for TinyBobBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Ok(self.0)
    }
}

struct RejectedBatch;

impl CompiledInputBatch for RejectedBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &[]
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Err(training("rejected test batch"))
    }
}

#[test]
fn compiled_input_batch_conversion_precedes_recurrent_mutation() {
    let mut runtime = compiled_adamw();
    let checkpoint = runtime.checkpoint().unwrap();
    assert!(runtime.step_batch(RejectedBatch, 0.05).is_err());
    assert_eq!(runtime.step_count(), 0);
    assert_eq!(runtime.checkpoint().unwrap(), checkpoint);
}

#[test]
fn compiled_training_runtime_is_optimizer_neutral() {
    let mut momentum = compiled();
    let mut adamw = compiled_adamw();
    assert!(momentum.inner.recurrent_store_groups.is_empty());
    assert!(momentum.inner.phase_outputs.observations.entries.is_empty());
    assert!(momentum.inner.accumulation.is_none());

    let (momentum_loss, momentum_outputs) = run_core_training_step(&mut momentum);
    let (adamw_loss, adamw_outputs) = run_core_training_step(&mut adamw);
    assert_eq!(momentum_loss.shape(), adamw_loss.shape());
    assert_eq!(
        momentum_outputs.keys().collect::<Vec<_>>(),
        adamw_outputs.keys().collect::<Vec<_>>()
    );
    let checkpoint = adamw.checkpoint().unwrap();
    assert_eq!(
        decode_adamw_checkpoint(checkpoint.as_bytes())
            .unwrap()
            .capture_identity,
        adamw.capture_identity()
    );

    let momentum_loss = run_core_commit_only_step(&mut compiled());
    let adamw_loss = run_core_commit_only_step(&mut compiled_adamw());
    assert_eq!(momentum_loss.shape(), adamw_loss.shape());
    run_adamw_commit_only_compatibility(&mut compiled_adamw());
}

#[test]
fn momentum_commit_only_matches_ordinary_replay_and_retries_atomically() {
    let mut ordinary = compiled();
    let mut committed = compiled();
    let identity = ordinary.capture_identity();
    let ordinary_step = ordinary.step(batch(), lr()).unwrap();
    let committed_step = committed
        .commit_step_batch(TinyBobBatch(batch()), 0.05)
        .unwrap();
    assert_eq!(ordinary_step.loss(), committed_step.loss());
    assert_eq!(ordinary_step.step(), committed_step.step());
    assert_eq!(ordinary_step.capture_identity(), identity);
    assert_eq!(committed_step.capture_identity(), identity);
    assert!(ordinary_step.output("logits").is_some());
    assert!(committed_step.outputs().is_empty());
    assert_eq!(
        ordinary.parameter_snapshots().unwrap(),
        committed.parameter_snapshots().unwrap()
    );
    assert_eq!(
        ordinary.momentum_snapshots().unwrap(),
        committed.momentum_snapshots().unwrap()
    );
    assert_eq!(
        ordinary.parameter_versions().unwrap(),
        committed.parameter_versions().unwrap()
    );
    assert_eq!(
        ordinary.momentum_versions().unwrap(),
        committed.momentum_versions().unwrap()
    );

    let mut retry = compiled();
    let initial_parameters = retry.parameter_snapshots().unwrap();
    let initial_momentum = retry.momentum_snapshots().unwrap();
    let initial_parameter_versions = retry.parameter_versions().unwrap();
    let initial_momentum_versions = retry.momentum_versions().unwrap();
    let mut missing = batch();
    missing.remove("target");
    assert!(retry.commit_step(missing, lr()).is_err());
    assert!(retry.commit_step_inner(batch(), lr(), Some(0)).is_err());
    assert_eq!(retry.step_count(), 0);
    assert_eq!(retry.parameter_snapshots().unwrap(), initial_parameters);
    assert_eq!(retry.momentum_snapshots().unwrap(), initial_momentum);
    assert_eq!(
        retry.parameter_versions().unwrap(),
        initial_parameter_versions
    );
    assert_eq!(
        retry.momentum_versions().unwrap(),
        initial_momentum_versions
    );

    let actual = retry.commit_step(batch(), lr()).unwrap();
    let mut expected = compiled();
    let expected_step = expected.commit_step(batch(), lr()).unwrap();
    assert_eq!(actual.loss(), expected_step.loss());
    assert_eq!(
        retry.parameter_snapshots().unwrap(),
        expected.parameter_snapshots().unwrap()
    );
    assert_eq!(
        retry.momentum_snapshots().unwrap(),
        expected.momentum_snapshots().unwrap()
    );
    assert_eq!(
        retry.parameter_versions().unwrap(),
        expected.parameter_versions().unwrap()
    );
    assert_eq!(
        retry.momentum_versions().unwrap(),
        expected.momentum_versions().unwrap()
    );
}

#[test]
fn adamw_observation_adapter_maps_every_ordered_report_shape() {
    assert_eq!(adamw_observation_schema(false, false).len(), 0);
    assert_eq!(adamw_observation_schema(true, false).len(), 2);
    assert_eq!(adamw_observation_schema(false, true).len(), 2);
    assert_eq!(adamw_observation_schema(true, true).len(), 4);
    let value = |key, value| CompiledTrainingObservationValue { key, value };
    let inner = |observations| CompiledTrainingStepResult {
        loss: TensorData::scalar(1.0),
        loss_aggregation_weight: 1,
        outputs: BTreeMap::new(),
        step: 1,
        capture_identity: 7,
        observations,
    };
    let committed = CompiledTrainingWindowProgress {
        replay_step: 1,
        optimizer_step: 1,
        ..CompiledTrainingWindowProgress::INITIAL
    };
    let window_weight =
        TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(3)]).unwrap();
    let observations = vec![
        value(
            AdamWObservation::ClipNorm.spec().key,
            TensorData::scalar(2.0),
        ),
        value(
            AdamWObservation::ClipScale.spec().key,
            TensorData::scalar(0.5),
        ),
        value(
            AdamWObservation::WindowMean.spec().key,
            TensorData::scalar(1.25),
        ),
        value(AdamWObservation::WindowWeight.spec().key, window_weight),
    ];
    let adapted = adamw_step_result(inner(observations), committed, 1, 3, true, true);
    assert_eq!(adapted.clip_report().unwrap().pre_clip_global_norm(), 2.0);
    assert_eq!(adapted.window_loss_report().unwrap().loss_weight(), 3);
    let no_report = adamw_step_result(inner(Vec::new()), committed, 1, 1, false, false);
    assert!(no_report.clip_report().is_none());
    assert!(no_report.window_loss_report().is_none());
    let pending = CompiledTrainingWindowProgress {
        replay_step: 1,
        accumulation_index: 1,
        ..CompiledTrainingWindowProgress::INITIAL
    };
    let pending = adamw_step_result(inner(Vec::new()), pending, 1, 3, true, true);
    assert!(pending.clip_report().is_none());
    assert!(pending.window_loss_report().is_none());
}

#[test]
fn adamw_auxiliary_output_schema_authenticates_every_report_shape() {
    let values = |clip_report, window_loss_report| {
        let mut values = Vec::new();
        if clip_report {
            values.extend([TensorData::scalar(2.0), TensorData::scalar(0.5)]);
        }
        if window_loss_report {
            values.extend([
                TensorData::scalar(1.25),
                TensorData::scalar_with_dtype(Scalar::U(3), DType::U64),
            ]);
        }
        values
    };
    for (clip_report, window_loss_report) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let schema =
            CompiledAdamWAuxiliaryOutputSchema::from_report_flags(clip_report, window_loss_report);
        assert_eq!(
            schema.report_flags(),
            Some((clip_report, window_loss_report))
        );
        assert_eq!(schema.clip_report_enabled(), clip_report);
        assert_eq!(schema.window_loss_report_enabled(), window_loss_report);
        for policy in [
            CpuNonFinitePolicy::Propagate,
            CpuNonFinitePolicy::RejectTransition,
        ] {
            let decoded = schema
                .validate_and_decode(&values(clip_report, window_loss_report), policy)
                .unwrap();
            assert_eq!(decoded.clip_report.is_some(), clip_report);
            assert_eq!(decoded.window_loss.is_some(), window_loss_report);
            if let Some(report) = decoded.clip_report {
                assert_eq!(report.pre_clip_global_norm(), 2.0);
                assert_eq!(report.applied_scale(), 0.5);
            }
            if let Some(report) = decoded.window_loss {
                assert_eq!(f32::from_bits(report.mean_loss_bits), 1.25);
                assert_eq!(report.loss_weight, 3);
            }
        }
    }
}

#[test]
fn adamw_auxiliary_output_schema_rejects_malformed_outputs_before_commit() {
    let clip = CompiledAdamWAuxiliaryOutputSchema::from_report_flags(true, false);
    let malformed = [
        TensorData::scalar(2.0),
        TensorData::scalar_with_dtype(Scalar::U(1), DType::U64),
    ];
    for policy in [
        CpuNonFinitePolicy::Propagate,
        CpuNonFinitePolicy::RejectTransition,
    ] {
        assert!(clip.validate_and_decode(&malformed, policy).is_err());
        assert!(clip.validate_and_decode(&malformed[..1], policy).is_err());
        assert!(
            CompiledAdamWAuxiliaryOutputSchema::from_report_flags(false, false)
                .validate_and_decode(&[TensorData::scalar(1.0)], policy)
                .is_err()
        );
    }

    let non_finite = [TensorData::scalar(f32::NAN), TensorData::scalar(1.0)];
    assert!(
        clip.validate_and_decode(&non_finite, CpuNonFinitePolicy::Propagate)
            .is_ok()
    );
    assert!(
        clip.validate_and_decode(&non_finite, CpuNonFinitePolicy::RejectTransition)
            .is_err()
    );

    let window = CompiledAdamWAuxiliaryOutputSchema::from_report_flags(false, true);
    let zero_weight = [
        TensorData::scalar(1.0),
        TensorData::scalar_with_dtype(Scalar::U(0), DType::U64),
    ];
    assert!(
        window
            .validate_and_decode(&zero_weight, CpuNonFinitePolicy::Propagate)
            .is_err()
    );
    assert!(
        window
            .validate_and_decode(&zero_weight, CpuNonFinitePolicy::RejectTransition)
            .is_err()
    );
}

#[test]
fn adamw_auxiliary_output_schema_rejects_reordered_nodes_and_flag_mismatch() {
    let mut graph = Graph::new();
    let clip_norm = graph
        .full_with_dtype(Shape::from([]), Scalar::F(2.0), DType::F32)
        .unwrap();
    let clip_scale = graph
        .full_with_dtype(Shape::from([]), Scalar::F(0.5), DType::F32)
        .unwrap();
    let reordered = [
        CompiledTrainingObservationNode {
            spec: AdamWObservation::ClipScale.spec(),
            node: clip_scale,
        },
        CompiledTrainingObservationNode {
            spec: AdamWObservation::ClipNorm.spec(),
            node: clip_norm,
        },
    ];
    assert!(CompiledAdamWAuxiliaryOutputSchema::from_nodes(&graph, &reordered).is_err());

    let mut plan = non_finite_flush_plan();
    plan.partial_flush.as_mut().unwrap().outputs =
        CompiledAdamWAuxiliaryOutputSchema::from_report_flags(false, false);
    assert!(plan.prepare_cpu().is_err());

    let mut plan = non_finite_flush_plan();
    plan.zero_grad.as_mut().unwrap().outputs =
        CompiledAdamWAuxiliaryOutputSchema::from_report_flags(true, false);
    assert!(plan.prepare_cpu().is_err());
}

#[test]
fn pending_adamw_step_admission_is_owned_and_publication_metadata_matches_native() {
    let plan = non_finite_flush_plan();
    let mut interpreted = plan.prepare_cpu().unwrap();
    let before_progress = interpreted.progress;
    let before_checkpoint = interpreted.checkpoint().unwrap();
    let pending = interpreted
        .admit_step(
            scalar_batch(1.0),
            Some(TensorData::scalar(0.01)),
            CompiledStepOutputSelection::All,
            None,
        )
        .unwrap();
    assert_eq!(interpreted.progress, before_progress);
    assert_eq!(interpreted.checkpoint().unwrap(), before_checkpoint);
    assert_eq!(pending.next_progress.replay_step, 1);
    assert_eq!(pending.next_progress.optimizer_step, 0);
    assert_eq!(pending.next_progress.accumulation_index, 1);
    assert_eq!(pending.loss_weight, 1);
    assert_eq!(pending.request.inputs, scalar_batch(1.0));
    assert!(pending.request.learning_rate.is_some());
    assert!(matches!(
        pending.request.output_selection,
        CompiledStepOutputSelection::All
    ));
    assert_eq!(
        pending.request.non_finite_policy,
        CpuNonFinitePolicy::Propagate
    );
    assert!(pending.request.injected_failure.is_none());

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = target.prepare(&plan).unwrap();
    let interpreted_accumulation = interpreted
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    let native_accumulation = native
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        (
            native_accumulation.step(),
            native_accumulation.optimizer_step(),
            native_accumulation.accumulation_index(),
            native_accumulation.loss_weight(),
            native_accumulation.did_update(),
            native_accumulation.clip_report(),
            native_accumulation.window_loss_report(),
        ),
        (
            interpreted_accumulation.step(),
            interpreted_accumulation.optimizer_step(),
            interpreted_accumulation.accumulation_index(),
            interpreted_accumulation.loss_weight(),
            interpreted_accumulation.did_update(),
            interpreted_accumulation.clip_report(),
            interpreted_accumulation.window_loss_report(),
        )
    );

    let interpreted_commit = interpreted
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    let native_commit = native
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        (
            native_commit.step(),
            native_commit.optimizer_step(),
            native_commit.accumulation_index(),
            native_commit.loss_weight(),
            native_commit.did_update(),
            native_commit.clip_report(),
            native_commit.window_loss_report(),
        ),
        (
            interpreted_commit.step(),
            interpreted_commit.optimizer_step(),
            interpreted_commit.accumulation_index(),
            interpreted_commit.loss_weight(),
            interpreted_commit.did_update(),
            interpreted_commit.clip_report(),
            interpreted_commit.window_loss_report(),
        )
    );
    assert_eq!(native.successful_steps, 2);
}

#[test]
fn cpu_step_transaction_rejects_output_layout_before_publication() {
    let plan = non_finite_flush_plan();

    let mut interpreted = plan.prepare_cpu().unwrap();
    let mut interpreted_reference = plan.prepare_cpu().unwrap();
    let initial = interpreted.checkpoint().unwrap();
    let accumulation_owner = Arc::make_mut(
        &mut interpreted
            .inner
            .accumulation
            .as_mut()
            .unwrap()
            .phase
            .capture,
    )
    .schedule
    .requested
    .pop()
    .unwrap();
    let error = match interpreted.step_commit_only(scalar_batch(1.0), TensorData::scalar(0.01)) {
        Ok(_) => panic!("malformed accumulation output layout succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("requested output layout"));
    assert_eq!(interpreted.checkpoint().unwrap(), initial);
    Arc::make_mut(
        &mut interpreted
            .inner
            .accumulation
            .as_mut()
            .unwrap()
            .phase
            .capture,
    )
    .schedule
    .requested
    .push(accumulation_owner);
    let expected = interpreted_reference
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    let actual = interpreted
        .step_commit_only(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(
        interpreted.checkpoint().unwrap(),
        interpreted_reference.checkpoint().unwrap()
    );

    let pending = interpreted.checkpoint().unwrap();
    let main_owner = Arc::make_mut(&mut interpreted.inner.capture)
        .schedule
        .requested
        .pop()
        .unwrap();
    let error = match interpreted.step(scalar_batch(2.0), TensorData::scalar(0.01)) {
        Ok(_) => panic!("malformed main output layout succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("requested output layout"));
    assert_eq!(interpreted.checkpoint().unwrap(), pending);
    Arc::make_mut(&mut interpreted.inner.capture)
        .schedule
        .requested
        .push(main_owner);
    interpreted_reference
        .step(scalar_batch(2.0), TensorData::scalar(0.01))
        .unwrap();
    interpreted
        .step(scalar_batch(2.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        interpreted.checkpoint().unwrap(),
        interpreted_reference.checkpoint().unwrap()
    );

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = target.prepare(&plan).unwrap();
    let mut native_reference = target.prepare(&plan).unwrap();
    let initial = native.checkpoint().unwrap();
    let accumulation_owner = Arc::make_mut(
        &mut native
            .inner
            .inner
            .accumulation
            .as_mut()
            .unwrap()
            .phase
            .capture,
    )
    .schedule
    .requested
    .pop()
    .unwrap();
    let error = match native.step(scalar_batch(1.0), TensorData::scalar(0.01)) {
        Ok(_) => panic!("malformed native accumulation output layout succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("requested output layout"));
    assert_eq!(native.checkpoint().unwrap(), initial);
    assert_eq!(native.successful_steps, 0);
    Arc::make_mut(
        &mut native
            .inner
            .inner
            .accumulation
            .as_mut()
            .unwrap()
            .phase
            .capture,
    )
    .schedule
    .requested
    .push(accumulation_owner);
    native_reference
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    native
        .step(scalar_batch(1.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        native.checkpoint().unwrap(),
        native_reference.checkpoint().unwrap()
    );

    let pending = native.checkpoint().unwrap();
    let main_owner = Arc::make_mut(&mut native.inner.inner.capture)
        .schedule
        .requested
        .pop()
        .unwrap();
    let error = match native.step_commit_only(scalar_batch(2.0), TensorData::scalar(0.01)) {
        Ok(_) => panic!("malformed native main output layout succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("requested output layout"));
    assert_eq!(native.checkpoint().unwrap(), pending);
    assert_eq!(native.successful_steps, 1);
    Arc::make_mut(&mut native.inner.inner.capture)
        .schedule
        .requested
        .push(main_owner);
    native_reference
        .step(scalar_batch(2.0), TensorData::scalar(0.01))
        .unwrap();
    native
        .step_commit_only(scalar_batch(2.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(native.successful_steps, 2);
    assert_eq!(
        native.checkpoint().unwrap(),
        native_reference.checkpoint().unwrap()
    );
}

#[test]
fn compiled_observation_schema_rejects_duplicate_keys_and_wrong_nodes() {
    let mut graph = Graph::new();
    let f32_value = graph
        .full_with_dtype(Shape::from([]), Scalar::F(1.0), DType::F32)
        .unwrap();
    let u64_value = graph
        .full_with_dtype(Shape::from([]), Scalar::U(1), DType::U64)
        .unwrap();
    let duplicate = [
        CompiledTrainingObservationNode {
            spec: AdamWObservation::ClipNorm.spec(),
            node: f32_value,
        },
        CompiledTrainingObservationNode {
            spec: AdamWObservation::ClipNorm.spec(),
            node: f32_value,
        },
    ];
    assert!(CompiledTrainingObservationSchema::from_nodes(&graph, &duplicate).is_err());
    assert!(
        CompiledTrainingObservationSchema::from_nodes(
            &graph,
            &[CompiledTrainingObservationNode {
                spec: AdamWObservation::ClipNorm.spec(),
                node: u64_value,
            }],
        )
        .is_err()
    );
}

#[test]
fn adamw_observation_schema_mismatch_fails_before_runtime_preparation() {
    let mut plan = non_finite_flush_plan();
    plan.inner.phase_outputs.observations.entries.swap(0, 1);
    assert!(plan.prepare_cpu().is_err());
}

#[test]
fn malformed_observation_value_rejects_during_staged_validation() {
    let schema = adamw_observation_schema(true, false);
    let outputs = [
        TensorData::scalar(1.0),
        TensorData::scalar_with_dtype(Scalar::U(2), DType::U64),
        TensorData::scalar(0.5),
    ];
    assert!(
        validate_staged_observations(
            &outputs,
            1,
            &schema,
            true,
            CpuNonFinitePolicy::RejectTransition,
        )
        .is_err()
    );
}

#[test]
fn optimizer_neutral_plan_renders_momentum_through_shared_metal_core() {
    let plan = CompiledTrainingPlan::compile(
        MomentumProgram {
            config: CompiledMomentumSgdConfig::new(0.9).unwrap(),
        },
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, _, parameters| Ok((graph.square(parameters["weight"])?, BTreeMap::new())),
    )
    .unwrap();
    let capture_identity = plan.capture_identity().unwrap();
    let metal = plan
        .metal_plan(
            MetalRenderer::new(
                8,
                crate::runtime::metal::MetalCapabilities {
                    max_buffer_length: 1 << 30,
                    unified_memory: true,
                    family: "Apple9".into(),
                },
            )
            .unwrap(),
            &BTreeMap::new(),
            None,
        )
        .unwrap();

    assert_eq!(metal.program_identity, capture_identity);
    assert_eq!(metal.inner.summary().fallback_count, 0);
    assert!(metal.inner.rendered_items().next().is_some());
    assert_eq!(
        metal
            .state_input_keys
            .values()
            .filter_map(RecurrentStateKey::momentum_parameter_name)
            .collect::<Vec<_>>(),
        ["weight"]
    );
}

#[test]
fn momentum_plan_prepares_independent_cpu_runtimes() {
    let plan =
        CompiledMomentumSgdPlan::compile(config(), initial_parameters(), build_tinybob).unwrap();
    let identity = plan.capture_identity();
    assert_eq!(plan.step_count(), 0);

    let mut first = plan.prepare_cpu().unwrap();
    let second = plan.prepare(&CpuSessionTarget::new()).unwrap();
    assert_eq!(first.capture_identity(), identity);
    assert_eq!(second.capture_identity(), identity);
    assert_eq!(
        first.parameter_snapshots().unwrap(),
        second.parameter_snapshots().unwrap()
    );

    first.step(batch(), lr()).unwrap();
    assert_eq!(first.step_count(), 1);
    assert_eq!(second.step_count(), 0);
    assert_ne!(
        first.parameter_snapshots().unwrap(),
        second.parameter_snapshots().unwrap()
    );
}

#[test]
fn momentum_plan_restores_checkpoint_without_recompiling_or_mutating_source() {
    let plan =
        CompiledMomentumSgdPlan::compile(config(), initial_parameters(), build_tinybob).unwrap();
    let mut trained = plan.prepare_cpu().unwrap();
    trained.step(batch(), lr()).unwrap();
    let checkpoint = trained.checkpoint().unwrap();

    let restored = plan.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(plan.step_count(), 0);
    assert_eq!(restored.step_count(), checkpoint.step());
    assert_eq!(restored.capture_identity(), plan.capture_identity());

    let resumed = restored.prepare_cpu().unwrap();
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);

    let mut foreign = checkpoint.clone();
    foreign.capture_identity ^= 1;
    assert!(plan.restore_checkpoint(&foreign).is_err());
    assert_eq!(plan.step_count(), 0);
}

#[test]
fn adamw_plan_prepares_independent_cpu_runtimes() {
    let plan =
        CompiledAdamWPlan::compile(adamw_config(), initial_parameters(), build_tinybob).unwrap();
    let identity = plan.capture_identity();
    assert_eq!(plan.step_count(), 0);

    let mut first = plan.prepare_cpu().unwrap();
    let second = plan.prepare_cpu().unwrap();
    assert_eq!(first.capture_identity(), identity);
    assert_eq!(second.capture_identity(), identity);
    assert_eq!(
        first.parameter_snapshots().unwrap(),
        second.parameter_snapshots().unwrap()
    );

    first.step(batch(), lr()).unwrap();
    assert_eq!(first.step_count(), 1);
    assert_eq!(second.step_count(), 0);
    assert_ne!(
        first.parameter_snapshots().unwrap(),
        second.parameter_snapshots().unwrap()
    );
}

fn build_two_parameter_linear_loss(
    graph: &mut Graph,
    _inputs: &BTreeMap<String, NodeId>,
    parameters: &BTreeMap<String, NodeId>,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let three = graph.full_with_dtype(Shape::from([]), Scalar::F(3.0), DType::F32)?;
    let four = graph.full_with_dtype(Shape::from([]), Scalar::F(4.0), DType::F32)?;
    let a = graph.mul(parameters["a"], three)?;
    let b = graph.mul(parameters["b"], four)?;
    Ok((graph.add(a, b)?, BTreeMap::new()))
}

fn batch() -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        (
            "target".into(),
            TensorData::from_scalars(
                [4],
                DType::I64,
                [Scalar::I(0), Scalar::I(1), Scalar::I(1), Scalar::I(0)],
            )
            .unwrap(),
        ),
        (
            "x".into(),
            TensorData::new([4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5]).unwrap(),
        ),
    ])
}

fn lr() -> TensorData {
    TensorData::scalar(0.05)
}

fn fresh_oracle_step(
    parameters: &BTreeMap<String, TensorData>,
    momentum: &BTreeMap<String, TensorData>,
) -> (
    TensorData,
    TensorData,
    BTreeMap<String, TensorData>,
    BTreeMap<String, TensorData>,
) {
    let values = batch();
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [4, 2], DType::F32);
    let target = graph.input_dtype("target", [4], DType::I64);
    let learning_rate = graph.input_dtype("lr", Shape::from([]), DType::F32);
    let mut parameter_nodes = BTreeMap::new();
    let mut momentum_nodes = BTreeMap::new();
    let mut bindings = HashMap::from([
        ("x".into(), values["x"].clone()),
        ("target".into(), values["target"].clone()),
        ("lr".into(), lr()),
    ]);
    for name in ["w1", "w2"] {
        let parameter_name = format!("parameter_{name}");
        let momentum_name = format!("momentum_{name}");
        let parameter = graph.input_dtype(
            parameter_name.clone(),
            parameters[name].shape().clone(),
            DType::F32,
        );
        let velocity = graph.input_dtype(
            momentum_name.clone(),
            momentum[name].shape().clone(),
            DType::F32,
        );
        parameter_nodes.insert(name.to_string(), parameter);
        momentum_nodes.insert(name.to_string(), velocity);
        bindings.insert(parameter_name, parameters[name].clone());
        bindings.insert(momentum_name, momentum[name].clone());
    }
    let hidden = graph.matmul(x, parameter_nodes["w1"]).unwrap();
    let hidden = graph.relu(hidden).unwrap();
    let logits = graph.matmul(hidden, parameter_nodes["w2"]).unwrap();
    let loss = cross_entropy(&mut graph, logits, target, LossOptions::default()).unwrap();
    let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
    let gradients = graph.gradient_default(loss, &targets).unwrap();
    let retained = graph
        .full_with_dtype(Shape::from([]), Scalar::F(0.9), DType::F32)
        .unwrap();
    let mut next_parameters = BTreeMap::new();
    let mut next_momentum = BTreeMap::new();
    let mut update_nodes = Vec::new();
    for ((name, parameter), gradient) in parameter_nodes.iter().zip(gradients) {
        let velocity = momentum_nodes[name];
        let velocity = graph
            .mul(retained, velocity)
            .and_then(|value| graph.add(value, gradient))
            .unwrap();
        let update = graph.mul(learning_rate, velocity).unwrap();
        let parameter = graph.sub(*parameter, update).unwrap();
        update_nodes.push((name.clone(), velocity, parameter));
    }
    let cpu = CpuBackend;
    for (name, velocity, parameter) in update_nodes {
        next_momentum.insert(
            name.clone(),
            cpu.execute(&graph, velocity, &bindings).unwrap(),
        );
        next_parameters.insert(name, cpu.execute(&graph, parameter, &bindings).unwrap());
    }
    (
        cpu.execute(&graph, loss, &bindings).unwrap(),
        cpu.execute(&graph, logits, &bindings).unwrap(),
        next_parameters,
        next_momentum,
    )
}

#[test]
fn tinybob_three_step_compiled_replay_matches_fresh_cpu_training() {
    let mut compiled = compiled();
    let identity = compiled.capture_identity();
    let mut parameters = compiled.parameter_snapshots().unwrap();
    let mut momentum = compiled.momentum_snapshots().unwrap();
    for step in 1..=3 {
        let (loss, logits, next_parameters, next_momentum) =
            fresh_oracle_step(&parameters, &momentum);
        let result = compiled.step(batch(), lr()).unwrap();
        assert_eq!(result.loss().storage(), loss.storage());
        assert_eq!(result.output("logits").unwrap().storage(), logits.storage());
        assert_eq!(result.step(), step);
        assert_eq!(result.capture_identity(), identity);
        assert_eq!(compiled.parameter_snapshots().unwrap(), next_parameters);
        assert_eq!(compiled.momentum_snapshots().unwrap(), next_momentum);
        assert_eq!(
            compiled.parameter_versions().unwrap(),
            BTreeMap::from([("w1".into(), step), ("w2".into(), step)])
        );
        assert_eq!(
            compiled.momentum_versions().unwrap(),
            BTreeMap::from([("w1".into(), step), ("w2".into(), step)])
        );
        parameters = next_parameters;
        momentum = next_momentum;
    }
}

#[test]
fn compile_identity_is_stable_and_initial_values_are_detached() {
    let original = initial_parameters();
    let before = original
        .iter()
        .map(|parameter| (parameter.name().to_string(), parameter.value().clone()))
        .collect::<BTreeMap<_, _>>();
    let first = CpuCompiledMomentumSgd::compile(config(), original, build_tinybob).unwrap();
    let second = compiled();
    assert_eq!(first.capture_identity(), second.capture_identity());
    assert_eq!(first.parameter_snapshots().unwrap(), before);
    let mut detached = first.parameter_snapshots().unwrap();
    detached
        .get_mut("w1")
        .unwrap()
        .assign(&TensorData::zeros([2, 4]).unwrap())
        .unwrap();
    assert_ne!(detached, first.parameter_snapshots().unwrap());
}

#[test]
fn step_inputs_exclude_every_persistent_state_binding() {
    let compiled = compiled();
    let external = compiled
        .inner
        .inputs
        .keys()
        .map(String::as_str)
        .chain([LEARNING_RATE_INPUT])
        .collect::<BTreeSet<_>>();
    assert_eq!(
        external,
        BTreeSet::from(["target", "x", LEARNING_RATE_INPUT])
    );
    let persistent = compiled
        .inner
        .capture
        .state_bindings
        .iter()
        .map(|binding| {
            compiled
                .inner
                .capture
                .schedule
                .inputs
                .iter()
                .find(|input| input.node == binding.input_node)
                .unwrap()
                .name
                .as_str()
        })
        .collect::<BTreeSet<_>>();
    assert!(!persistent.is_empty());
    assert!(
        persistent
            .iter()
            .all(|name| name.starts_with(INTERNAL_PREFIX))
    );
    assert!(external.is_disjoint(&persistent));
    let mut consumer_views = BTreeMap::<NodeId, BTreeSet<bool>>::new();
    for binding in &compiled.inner.capture.state_bindings {
        consumer_views
            .entry(binding.input_node)
            .or_default()
            .insert(binding.desc.view.is_some());
    }
    assert!(
        consumer_views
            .values()
            .any(|views| views == &BTreeSet::from([false, true]))
    );
    let pure_items = compiled
        .inner
        .capture
        .schedule
        .items
        .iter()
        .take_while(|item| !item.is_effect())
        .collect::<Vec<_>>();
    assert!(!pure_items.is_empty());
    assert!(pure_items.iter().all(|item| item.boundary.is_none()));
}

#[test]
fn malformed_and_duplicate_inputs_fail_before_state_publication() {
    let duplicate = TrainingParameterInit::new(
        "w1",
        TensorData::zeros_with_dtype([2, 4], DType::F32).unwrap(),
    )
    .unwrap();
    assert!(
        CpuCompiledMomentumSgd::compile(
            config(),
            initial_parameters().into_iter().chain([duplicate]),
            build_tinybob,
        )
        .is_err()
    );
    assert!(
        CompiledMomentumSgdConfig::new(0.9)
            .unwrap()
            .with_input(INTERNAL_PREFIX, [1], DType::F32)
            .is_err()
    );
    assert!(
        CompiledMomentumSgdConfig::new(0.9)
            .unwrap()
            .with_input("x", [1], DType::F32)
            .unwrap()
            .with_input("x", [1], DType::F32)
            .is_err()
    );
    assert!(
        TrainingParameterInit::new(
            "bad",
            TensorData::zeros_with_dtype([1], DType::I32).unwrap(),
        )
        .is_err()
    );

    let mut compiled = compiled();
    let before = compiled.parameter_snapshots().unwrap();
    let mut missing = batch();
    missing.remove("target");
    assert!(compiled.step(missing, lr()).is_err());
    assert!(
        compiled
            .step(batch(), TensorData::new([1], vec![0.05]).unwrap())
            .is_err()
    );
    assert_eq!(compiled.step_count(), 0);
    assert_eq!(compiled.parameter_snapshots().unwrap(), before);
}

#[test]
fn cross_namespace_names_reject_before_the_private_graph_builder_runs() {
    let invoked = std::cell::Cell::new(false);
    let conflicting = CompiledMomentumSgdConfig::new(0.9)
        .unwrap()
        .with_input("w1", [4, 2], DType::F32)
        .unwrap();
    let result = CpuCompiledMomentumSgd::compile(conflicting, initial_parameters(), |_, _, _| {
        invoked.set(true);
        Err(training("builder should not run"))
    });
    assert!(result.is_err());
    assert!(!invoked.get());
}

#[test]
fn injected_and_stale_replay_failures_preserve_runtime_cursor_and_step() {
    let mut compiled = compiled();
    let initial_parameters = compiled.parameter_snapshots().unwrap();
    let initial_momentum = compiled.momentum_snapshots().unwrap();
    let initial_cursor = compiled.inner.cursor.clone();
    assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
    assert_eq!(compiled.step_count(), 0);
    assert_eq!(compiled.inner.cursor, initial_cursor);
    assert_eq!(compiled.parameter_snapshots().unwrap(), initial_parameters);
    assert_eq!(compiled.momentum_snapshots().unwrap(), initial_momentum);

    compiled.step(batch(), lr()).unwrap();
    let advanced = compiled.inner.cursor.clone();
    let advanced_parameters = compiled.parameter_snapshots().unwrap();
    compiled.inner.cursor = initial_cursor;
    assert!(compiled.step(batch(), lr()).is_err());
    assert_eq!(compiled.step_count(), 1);
    compiled.inner.cursor = advanced;
    assert_eq!(compiled.parameter_snapshots().unwrap(), advanced_parameters);
}

#[test]
fn adamw_replays_one_capture_with_graph_owned_state() {
    let mut compiled = compiled_adamw();
    let identity = compiled.capture_identity();
    let zeros = initial_parameters()
        .into_iter()
        .map(|parameter| {
            (
                parameter.name().to_string(),
                TensorData::zeros_with_dtype(parameter.value().shape().clone(), DType::F32)
                    .unwrap(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(compiled.optimizer_step().unwrap(), 0);
    assert_eq!(compiled.first_moment_snapshots().unwrap(), zeros);
    assert_eq!(compiled.second_moment_snapshots().unwrap(), zeros);

    let first = compiled.step(batch(), lr()).unwrap();
    assert_eq!(first.step(), 1);
    assert_eq!(first.capture_identity(), identity);
    assert_eq!(compiled.optimizer_step().unwrap(), 1);
    assert_eq!(
        compiled
            .parameter_versions()
            .unwrap()
            .values()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 1]
    );
    assert_eq!(
        compiled
            .first_moment_versions()
            .unwrap()
            .values()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 1]
    );
    assert_eq!(
        compiled
            .second_moment_versions()
            .unwrap()
            .values()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 1]
    );
    assert_ne!(compiled.first_moment_snapshots().unwrap(), zeros);
    assert_ne!(compiled.second_moment_snapshots().unwrap(), zeros);

    let second = compiled.step(batch(), lr()).unwrap();
    assert_eq!(second.step(), 2);
    assert_eq!(second.capture_identity(), identity);
    assert_eq!(compiled.optimizer_step().unwrap(), 2);
}

#[test]
fn adamw_clips_the_complete_parameter_gradient_set_by_one_global_norm() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap();
    assert_eq!(config.max_gradient_norm(), Some(1.0));
    assert!(!config.clip_report_enabled());
    let parameters = || {
        ["a", "b"].map(|name| TrainingParameterInit::new(name, TensorData::scalar(0.0)).unwrap())
    };
    let unclipped = CpuCompiledAdamW::compile(
        CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0).unwrap(),
        parameters(),
        build_two_parameter_linear_loss,
    )
    .unwrap();
    let mut clipped = CpuCompiledAdamW::compile(
        config.clone(),
        parameters(),
        build_two_parameter_linear_loss,
    )
    .unwrap();
    assert_eq!(clipped.max_gradient_norm(), Some(1.0));
    assert_ne!(clipped.capture_identity(), unclipped.capture_identity());

    let unreported = clipped
        .step(BTreeMap::new(), TensorData::scalar(0.1))
        .unwrap();
    assert!(unreported.clip_report().is_none());
    let moments = clipped.first_moment_snapshots().unwrap();
    let a = moments["a"].scalar_at(0).as_f64();
    let b = moments["b"].scalar_at(0).as_f64();
    assert!((a - 0.6).abs() < 1e-6, "clipped a gradient was {a}");
    assert!((b - 0.8).abs() < 1e-6, "clipped b gradient was {b}");

    let reported_plan = CompiledAdamWPlan::compile(
        config.with_clip_report(),
        parameters(),
        build_two_parameter_linear_loss,
    )
    .unwrap();
    assert!(reported_plan.clip_report_enabled());
    assert_ne!(reported_plan.capture_identity(), clipped.capture_identity());
    let renderer = MetalRenderer::new(
        8,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: 1 << 30,
            unified_memory: true,
            family: "Apple9".into(),
        },
    )
    .unwrap();
    let error = match reported_plan.metal_plan(renderer) {
        Ok(_) => panic!("clip-report plan rendered for Metal"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("clip reporting is currently CPU-only")
    );

    let mut interpreted = reported_plan.prepare_cpu().unwrap();
    let executor = CapturedReplayExecutor::default();
    let mut native = reported_plan
        .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
        .unwrap();
    let interpreted_step = interpreted
        .step(BTreeMap::new(), TensorData::scalar(0.1))
        .unwrap();
    let native_step = native
        .step(BTreeMap::new(), TensorData::scalar(0.1))
        .unwrap();
    let interpreted_report = interpreted_step.clip_report().unwrap();
    let native_report = native_step.clip_report().unwrap();
    assert_eq!(interpreted_report.pre_clip_global_norm(), 5.0);
    assert_eq!(interpreted_report.applied_scale(), 0.2);
    assert_eq!(interpreted_report.did_clip(), Some(true));
    assert_eq!(native_report, interpreted_report);
    let reported_checkpoint = interpreted.checkpoint().unwrap();
    assert_eq!(native.checkpoint().unwrap(), reported_checkpoint);
    let legacy_checkpoint = clipped.checkpoint().unwrap();
    assert!(
        reported_plan
            .restore_checkpoint(&legacy_checkpoint)
            .is_err()
    );
    let (legacy_tensors, mut legacy_metadata) =
        load_safetensors(legacy_checkpoint.as_bytes()).unwrap();
    let (reported_tensors, mut reported_metadata) =
        load_safetensors(reported_checkpoint.as_bytes()).unwrap();
    assert_eq!(reported_tensors, legacy_tensors);
    assert_ne!(
        reported_metadata.remove("capture_identity"),
        legacy_metadata.remove("capture_identity")
    );
    assert_eq!(reported_metadata, legacy_metadata);
}

#[test]
fn adamw_loss_scaling_unscales_before_optimizer_policies() {
    let parameters = || {
        ["a", "b"].map(|name| TrainingParameterInit::new(name, TensorData::scalar(0.0)).unwrap())
    };
    let base = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0).unwrap();
    let unit = CpuCompiledAdamW::compile(
        base.clone().with_loss_scale(1.0).unwrap(),
        parameters(),
        build_two_parameter_linear_loss,
    )
    .unwrap();
    let unscaled =
        CpuCompiledAdamW::compile(base.clone(), parameters(), build_two_parameter_linear_loss)
            .unwrap();
    assert_eq!(unit.loss_scale(), 1.0);
    assert_eq!(unit.capture_identity(), unscaled.capture_identity());

    let scaled_config = base
        .with_loss_scale(128.0)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap();
    assert_eq!(scaled_config.loss_scale(), 128.0);
    let mut scaled =
        CpuCompiledAdamW::compile(scaled_config, parameters(), build_two_parameter_linear_loss)
            .unwrap();
    let mut clipped = CpuCompiledAdamW::compile(
        CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap(),
        parameters(),
        build_two_parameter_linear_loss,
    )
    .unwrap();
    assert_eq!(scaled.loss_scale(), 128.0);
    assert_ne!(scaled.capture_identity(), clipped.capture_identity());

    let scaled_result = scaled
        .step(BTreeMap::new(), TensorData::scalar(0.1))
        .unwrap();
    let clipped_result = clipped
        .step(BTreeMap::new(), TensorData::scalar(0.1))
        .unwrap();
    assert_eq!(scaled_result.loss(), clipped_result.loss());
    assert_eq!(
        scaled.parameter_snapshots().unwrap(),
        clipped.parameter_snapshots().unwrap()
    );
    assert_eq!(
        scaled.first_moment_snapshots().unwrap(),
        clipped.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        scaled.second_moment_snapshots().unwrap(),
        clipped.second_moment_snapshots().unwrap()
    );
}

#[test]
fn adamw_weight_decay_exclusions_preserve_the_complete_optimizer_frontier() {
    let parameters = || {
        [
            TrainingParameterInit::new("a", TensorData::scalar(2.0)).unwrap(),
            TrainingParameterInit::new("b", TensorData::scalar(3.0)).unwrap(),
        ]
    };
    let config = |weight_decay, exclusions: &[&str]| {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, weight_decay)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap();
        config
            .with_weight_decay_exclusions(exclusions.iter().copied())
            .unwrap()
    };
    let compile = |config| {
        CpuCompiledAdamW::compile(config, parameters(), build_two_parameter_linear_loss).unwrap()
    };
    let mut excluded = compile(config(0.1, &["b"]));
    let mut full_decay = compile(config(0.1, &[]));
    let mut no_decay = compile(config(0.0, &[]));

    for runtime in [&mut excluded, &mut full_decay, &mut no_decay] {
        let first = runtime
            .step(BTreeMap::new(), TensorData::scalar(0.1))
            .unwrap();
        assert!(!first.did_update());
        assert_eq!(
            runtime.gradient_accumulator_snapshots().unwrap(),
            BTreeMap::from([
                ("a".into(), TensorData::scalar(3.0)),
                ("b".into(), TensorData::scalar(4.0)),
            ])
        );
        let second = runtime
            .step(BTreeMap::new(), TensorData::scalar(0.1))
            .unwrap();
        assert!(second.did_update());
    }

    let excluded_parameters = excluded.parameter_snapshots().unwrap();
    let full_decay_parameters = full_decay.parameter_snapshots().unwrap();
    let no_decay_parameters = no_decay.parameter_snapshots().unwrap();
    assert_eq!(excluded_parameters["a"], full_decay_parameters["a"]);
    assert_ne!(excluded_parameters["a"], no_decay_parameters["a"]);
    assert_eq!(excluded_parameters["b"], no_decay_parameters["b"]);
    assert_ne!(excluded_parameters["b"], full_decay_parameters["b"]);
    assert_eq!(
        excluded.first_moment_snapshots().unwrap(),
        full_decay.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        excluded.first_moment_snapshots().unwrap(),
        no_decay.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        excluded.second_moment_snapshots().unwrap(),
        full_decay.second_moment_snapshots().unwrap()
    );
    assert_eq!(
        excluded.second_moment_snapshots().unwrap(),
        no_decay.second_moment_snapshots().unwrap()
    );
    assert!(
        excluded
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .all(|value| value == &TensorData::scalar(0.0))
    );

    let checkpoint = excluded.checkpoint().unwrap();
    let resumed = CpuCompiledAdamW::compile_from_checkpoint(
        config(0.1, &["b"]),
        &checkpoint,
        build_two_parameter_linear_loss,
    )
    .unwrap();
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(
            config(0.1, &[]),
            &checkpoint,
            build_two_parameter_linear_loss,
        )
        .is_err()
    );
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(
            config(0.1, &["a"]),
            &checkpoint,
            build_two_parameter_linear_loss,
        )
        .is_err()
    );
}

#[test]
fn adamw_weight_decay_exclusion_names_validate_before_graph_construction() {
    struct NamesModule {
        trainable: Parameter,
        frozen: Parameter,
        buffer: Parameter,
    }

    impl Module for NamesModule {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            assert!(prefix.is_empty());
            visitor("weight".into(), &self.trainable, StateKind::Parameter);
            visitor("weight_alias".into(), &self.trainable, StateKind::Parameter);
            visitor("frozen".into(), &self.frozen, StateKind::Parameter);
            visitor("running".into(), &self.buffer, StateKind::Buffer);
        }
    }

    struct RepeatedCanonicalName(Parameter);

    impl Module for RepeatedCanonicalName {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            assert!(prefix.is_empty());
            visitor("weight".into(), &self.0, StateKind::Parameter);
            visitor("weight".into(), &self.0, StateKind::Parameter);
        }
    }

    let base = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.1).unwrap();
    assert_eq!(base.weight_decay_exclusions().len(), 0);
    assert_eq!(
        base.clone()
            .with_weight_decay_exclusions(["z", "a"])
            .unwrap()
            .weight_decay_exclusions()
            .collect::<Vec<_>>(),
        vec!["a", "z"]
    );
    assert!(
        base.clone()
            .with_weight_decay_exclusions(["weight", "weight"])
            .is_err()
    );
    assert!(
        base.clone()
            .with_weight_decay_exclusions(["weight"])
            .unwrap()
            .with_weight_decay_exclusions(["weight"])
            .is_err()
    );

    let parameters = || [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()];
    let empty = CpuCompiledAdamW::compile(
        base.clone()
            .with_weight_decay_exclusions(std::iter::empty::<&str>())
            .unwrap(),
        parameters(),
        |graph, _, parameters| {
            Ok((
                graph.mul(parameters["weight"], parameters["weight"])?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap();
    let ordinary = CpuCompiledAdamW::compile(base.clone(), parameters(), |graph, _, parameters| {
        assert_eq!(graph.dtype(parameters["weight"])?, DType::F32);
        Ok((
            graph.mul(parameters["weight"], parameters["weight"])?,
            BTreeMap::new(),
        ))
    })
    .unwrap();
    assert_eq!(empty.capture_identity(), ordinary.capture_identity());
    assert_eq!(empty.checkpoint().unwrap(), ordinary.checkpoint().unwrap());

    let repeated = RepeatedCanonicalName(Parameter::new(TensorData::scalar(1.0), true));
    let invoked = std::cell::Cell::new(false);
    let result = CpuCompiledAdamW::compile_module(base.clone(), &repeated, |_, _, _| {
        invoked.set(true);
        Err(training("builder should not run"))
    });
    assert!(result.is_err());
    assert!(!invoked.get());

    let invoked = std::cell::Cell::new(false);
    let result = CpuCompiledAdamW::compile(
        base.clone()
            .with_weight_decay_exclusions(["missing"])
            .unwrap(),
        parameters(),
        |_, _, _| {
            invoked.set(true);
            Err(training("builder should not run"))
        },
    );
    assert!(result.is_err());
    assert!(!invoked.get());

    let module = NamesModule {
        trainable: Parameter::new(TensorData::scalar(1.0), true),
        frozen: Parameter::new(TensorData::scalar(2.0), false),
        buffer: Parameter::new(TensorData::scalar(3.0), false),
    };
    for invalid in ["weight_alias", "frozen", "running", "missing"] {
        let invoked = std::cell::Cell::new(false);
        let result = CpuCompiledAdamW::compile_module(
            base.clone()
                .with_weight_decay_exclusions([invalid])
                .unwrap(),
            &module,
            |_, _, _| {
                invoked.set(true);
                Err(training("builder should not run"))
            },
        );
        assert!(
            result.is_err(),
            "invalid exclusion {invalid:?} was accepted"
        );
        assert!(!invoked.get());
    }
}

#[test]
fn token_weight_policy_validates_static_descriptor_before_compilation() {
    let base = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0).unwrap();
    assert!(base.clone().with_zero_valid_token_microbatches().is_err());
    assert!(
        base.clone()
            .with_token_weighted_gradient_accumulation("mask")
            .is_err()
    );
    assert!(
        base.clone()
            .with_input("mask", [2], DType::F32)
            .unwrap()
            .with_token_weighted_gradient_accumulation("mask")
            .is_ok()
    );
    assert!(
        base.clone()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_input("mask", [2], DType::I32)
            .unwrap()
            .with_token_weighted_gradient_accumulation("mask")
            .is_err()
    );
    assert!(
        base.clone()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_input("mask", [0], DType::F32)
            .unwrap()
            .with_token_weighted_gradient_accumulation("mask")
            .is_err()
    );
    assert!(
        base.with_gradient_accumulation(2)
            .unwrap()
            .with_input("mask", [8_388_609], DType::F32)
            .unwrap()
            .with_token_weighted_gradient_accumulation("mask")
            .is_err()
    );
}

fn token_weighted_config(steps: u64) -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(steps)
        .unwrap()
        .with_input("features", [3], DType::F32)
        .unwrap()
        .with_input("mask", [3], DType::F32)
        .unwrap()
        .with_token_weighted_gradient_accumulation("mask")
        .unwrap()
}

struct TokenMeanModule {
    weight: Parameter,
}

impl TokenMeanModule {
    fn new() -> Self {
        Self {
            weight: Parameter::new(TensorData::scalar(2.0), true),
        }
    }
}

impl Module for TokenMeanModule {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        assert!(prefix.is_empty());
        visitor("weight".into(), &self.weight, StateKind::Parameter);
    }
}

fn token_mean_dropout() -> CompiledDropoutConfig {
    CompiledDropoutConfig::new(CompiledDropoutKey([47, 53]))
}

fn build_token_losses(
    module: &TokenMeanModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
    let observation = dropout.dropout(graph, inputs["features"], 0.5)?;
    let weight = module.weight.bind(graph)?;
    let losses = graph.mul(weight, inputs["features"])?;
    Ok((
        losses,
        BTreeMap::from([("dropout_observation".into(), observation)]),
    ))
}

fn build_token_graph(
    module: &TokenMeanModule,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    let (losses, outputs) = build_token_losses(module, graph, inputs, dropout)?;
    Ok(CompiledAdamWGraph::token_mean(losses, outputs))
}

fn compile_token_weighted_plan(steps: u64) -> CompiledAdamWPlan {
    CompiledAdamWPlan::compile_token_mean_module_with_dropout(
        token_weighted_config(steps),
        token_mean_dropout(),
        &TokenMeanModule::new(),
        build_token_losses,
    )
    .unwrap()
}

fn compile_zero_token_weighted_plan(steps: u64) -> CompiledAdamWPlan {
    CompiledAdamWPlan::compile_token_mean_module_with_dropout(
        token_weighted_config(steps)
            .with_zero_valid_token_microbatches()
            .unwrap()
            .with_window_loss_report(),
        token_mean_dropout(),
        &TokenMeanModule::new(),
        build_token_losses,
    )
    .unwrap()
}

#[test]
fn single_step_token_mean_updates_without_accumulation_state() {
    let plan = compile_zero_token_weighted_plan(1);
    assert_eq!(plan.gradient_accumulation_steps(), 1);
    assert_eq!(plan.accumulation_capture_identity(), None);
    assert_eq!(plan.flush_capture_identity(), None);
    assert_eq!(plan.zero_grad_capture_identity(), None);
    assert_ne!(
        plan.inner.capture.schedule.requested[0], plan.inner.capture.schedule.requested[2],
        "public loss and WindowMean observation require distinct owners"
    );
    let renderer = MetalRenderer::new(
        8,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: 1 << 30,
            unified_memory: true,
            family: "Apple9".into(),
        },
    )
    .unwrap();
    assert!(plan.metal_plan(renderer).is_err());

    let valid = || token_weighted_batch([1.0, 100.0, 3.0], [1.0, 0.0, 1.0]);
    let empty = || token_weighted_batch([7.0, 11.0, 13.0], [0.0; 3]);
    let mut interpreted = plan.prepare_cpu().unwrap();
    assert_core_training_window(&interpreted, 1, 0);
    let initial = interpreted.checkpoint().unwrap();
    assert_eq!(initial.info().accumulated_token_count(), None);
    let error = match interpreted.step(empty(), TensorData::scalar(0.1)) {
        Ok(_) => panic!("zero-token single-step training replay succeeded"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("completed token window must contain at least one valid token")
    );
    assert_eq!(interpreted.checkpoint().unwrap(), initial);

    let first = interpreted.step(valid(), TensorData::scalar(0.1)).unwrap();
    assert!(first.did_update());
    assert_eq!(first.accumulation_index(), 0);
    assert_core_training_window_step(&first, 0, true);
    assert_core_training_window(&interpreted, 1, 0);
    assert_eq!(first.loss_weight(), 2);
    assert_core_loss_aggregation_weight(&first, 2);
    assert_eq!(first.loss().scalar_at(0).as_f64(), 4.0);
    let window = first.window_loss_report().unwrap();
    assert_eq!(f64::from(window.mean_loss()), 4.0);
    assert_eq!(window.loss_weight(), 2);
    assert_eq!(window.microbatch_count(), 1);
    assert!(
        interpreted
            .gradient_accumulator_snapshots()
            .unwrap()
            .is_empty()
    );
    let checkpoint = interpreted.checkpoint().unwrap();
    assert_eq!(checkpoint.info().accumulated_token_count(), None);

    let mut two_tokens = plan.prepare_cpu().unwrap();
    let mut one_token = plan.prepare_cpu().unwrap();
    let two = two_tokens
        .step(
            token_weighted_batch([1.0, 3.0, 100.0], [1.0, 1.0, 0.0]),
            TensorData::scalar(0.1),
        )
        .unwrap();
    let one = one_token
        .step(
            token_weighted_batch([2.0, 100.0, 100.0], [1.0, 0.0, 0.0]),
            TensorData::scalar(0.1),
        )
        .unwrap();
    assert_eq!(two.loss(), one.loss());
    assert_eq!(two.loss_weight(), 2);
    assert_eq!(one.loss_weight(), 1);
    assert_eq!(
        two_tokens.checkpoint().unwrap(),
        one_token.checkpoint().unwrap()
    );

    let mut restored = plan
        .restore_checkpoint(&checkpoint)
        .unwrap()
        .prepare_cpu()
        .unwrap();
    assert_eq!(restored.checkpoint().unwrap(), checkpoint);
    let expected = interpreted.step(valid(), TensorData::scalar(0.1)).unwrap();
    let actual = restored.step(valid(), TensorData::scalar(0.1)).unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.loss_weight(), expected.loss_weight());
    assert_eq!(
        restored.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = plan.prepare(&target).unwrap();
    assert_core_training_window(&native, 1, 0);
    assert!(native.preparation_report().accumulation().is_none());
    assert!(native.preparation_report().partial_flush().is_none());
    assert!(native.preparation_report().zero_grad().is_none());
    let native_initial = native.checkpoint().unwrap();
    let error = match native.step(empty(), TensorData::scalar(0.1)) {
        Ok(_) => panic!("zero-token native single-step training replay succeeded"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("completed token window must contain at least one valid token")
    );
    assert_eq!(native.checkpoint().unwrap(), native_initial);
    let native_step = native.step(valid(), TensorData::scalar(0.1)).unwrap();
    assert!(native_step.did_update());
    assert_core_training_window_step(&native_step, 0, true);
    assert_core_training_window(&native, 1, 0);
    assert_eq!(native_step.loss_weight(), 2);
    assert_core_loss_aggregation_weight(&native_step, 2);
    assert_eq!(native_step.window_loss_report().unwrap().loss_weight(), 2);
    assert_eq!(native_step.report().fallback_count(), 0);
}

fn compile_direct_token_mean_without_dropout(
    config: CompiledAdamWConfig,
    module: &TokenMeanModule,
) -> CompiledAdamWPlan {
    let (mask_input, mask_shape) = token_mean_loss_descriptor(&config).unwrap();
    let allow_zero_valid_token_microbatches = config.allow_zero_valid_token_microbatches;
    let parameter = TrainingParameterInit::new("weight", module.weight.value().unwrap()).unwrap();
    CompiledAdamWPlan::compile_parameters_with_lowered_loss(
        config,
        [parameter],
        |graph, inputs, parameters| {
            let losses = graph.mul(parameters["weight"], inputs["features"])?;
            let loss = lower_token_mean_loss(
                graph,
                losses,
                inputs[mask_input.as_str()],
                &mask_shape,
                allow_zero_valid_token_microbatches,
            )?;
            Ok((loss, BTreeMap::new()))
        },
    )
    .unwrap()
}

#[test]
fn unified_scalar_objective_matches_legacy_capture_replay_and_checkpoint() {
    let module = TiedFrozenModule::new([0.1, -0.2]);
    let legacy =
        CompiledAdamWPlan::compile_module(module_config(), &module, build_tied_frozen).unwrap();
    let unified = CompiledAdamWPlan::compile_module_graph(
        module_config(),
        &module,
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            let built = CompiledAdamWGraph::scalar(loss, outputs);
            assert_eq!(built.objective().node(), loss);
            assert_eq!(built.outputs().len(), 1);
            assert!(built.outputs().contains_key("output"));
            Ok(built)
        },
    )
    .unwrap();
    assert_eq!(unified.capture_identity(), legacy.capture_identity());
    assert_eq!(unified.inspection().unwrap(), legacy.inspection().unwrap());

    let inputs = || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
    let mut legacy_runtime = legacy.prepare_cpu().unwrap();
    let mut unified_runtime = unified.prepare_cpu().unwrap();
    let legacy_step = legacy_runtime
        .step(inputs(), TensorData::scalar(0.01))
        .unwrap();
    let unified_step = unified_runtime
        .step(inputs(), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(unified_step.loss(), legacy_step.loss());
    assert_eq!(unified_step.outputs(), legacy_step.outputs());
    assert_eq!(unified_step.optimizer_step(), legacy_step.optimizer_step());
    assert_eq!(unified_step.loss_weight(), 1);
    let checkpoint = unified_runtime.checkpoint().unwrap();
    assert_eq!(checkpoint, legacy_runtime.checkpoint().unwrap());
    assert_eq!(
        unified
            .restore_checkpoint(&checkpoint)
            .unwrap()
            .prepare_cpu()
            .unwrap()
            .checkpoint()
            .unwrap(),
        checkpoint
    );

    let owned = CompiledModuleAdamWPlan::compile_graph(
        module_config(),
        TiedFrozenModule::new([0.1, -0.2]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::new(
                CompiledAdamWObjective::scalar(loss),
                outputs,
            ))
        },
    )
    .unwrap();
    assert_eq!(owned.capture_identity(), legacy.capture_identity());
    assert_eq!(owned.inspection().unwrap(), legacy.inspection().unwrap());

    let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([61, 67]));
    let legacy_dropout = CompiledAdamWPlan::compile_module_with_dropout(
        module_config(),
        dropout,
        &module,
        build_tied_dropout,
    )
    .unwrap();
    let unified_dropout = CompiledAdamWPlan::compile_module_graph_with_dropout(
        module_config(),
        dropout,
        &module,
        |module, graph, inputs, dropout| {
            let (loss, outputs) = build_tied_dropout(module, graph, inputs, dropout)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    assert_eq!(
        unified_dropout.capture_identity(),
        legacy_dropout.capture_identity()
    );
    assert_eq!(
        unified_dropout.inspection().unwrap(),
        legacy_dropout.inspection().unwrap()
    );
}

#[test]
fn unified_token_mean_objective_matches_legacy_and_rejects_policy_mismatch_atomically() {
    let module = TokenMeanModule::new();
    let direct = compile_direct_token_mean_without_dropout(token_weighted_config(2), &module);
    let unified_without_dropout = CompiledAdamWPlan::compile_module_graph(
        token_weighted_config(2),
        &module,
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
        },
    )
    .unwrap();
    assert_eq!(
        unified_without_dropout.capture_identity(),
        direct.capture_identity()
    );
    assert_eq!(
        unified_without_dropout.inspection().unwrap(),
        direct.inspection().unwrap()
    );
    let first_batch = || token_weighted_batch([1.0, 100.0, 1.0], [1.0, 0.0, 1.0]);
    let mut direct_runtime = direct.prepare_cpu().unwrap();
    let mut unified_without_dropout_runtime = unified_without_dropout.prepare_cpu().unwrap();
    let direct_step = direct_runtime
        .step(first_batch(), TensorData::scalar(0.1))
        .unwrap();
    let unified_without_dropout_step = unified_without_dropout_runtime
        .step(first_batch(), TensorData::scalar(0.1))
        .unwrap();
    assert_eq!(unified_without_dropout_step.loss(), direct_step.loss());
    assert_eq!(
        unified_without_dropout_step.outputs(),
        direct_step.outputs()
    );
    assert_eq!(unified_without_dropout_step.loss_weight(), 2);
    let without_dropout_checkpoint = unified_without_dropout_runtime.checkpoint().unwrap();
    assert_eq!(
        without_dropout_checkpoint,
        direct_runtime.checkpoint().unwrap()
    );
    assert_eq!(
        unified_without_dropout
            .restore_checkpoint(&without_dropout_checkpoint)
            .unwrap()
            .prepare_cpu()
            .unwrap()
            .checkpoint()
            .unwrap(),
        without_dropout_checkpoint
    );

    let legacy = CompiledAdamWPlan::compile_token_mean_module_with_dropout(
        token_weighted_config(2),
        token_mean_dropout(),
        &module,
        build_token_losses,
    )
    .unwrap();
    let unified = CompiledAdamWPlan::compile_module_graph_with_dropout(
        token_weighted_config(2),
        token_mean_dropout(),
        &module,
        build_token_graph,
    )
    .unwrap();
    assert_eq!(unified.capture_identity(), legacy.capture_identity());
    assert_eq!(unified.inspection().unwrap(), legacy.inspection().unwrap());

    let mut legacy_runtime = legacy.prepare_cpu().unwrap();
    let mut unified_runtime = unified.prepare_cpu().unwrap();
    let legacy_step = legacy_runtime
        .step(first_batch(), TensorData::scalar(0.1))
        .unwrap();
    let unified_step = unified_runtime
        .step(first_batch(), TensorData::scalar(0.1))
        .unwrap();
    assert_eq!(unified_step.loss(), legacy_step.loss());
    assert_eq!(unified_step.outputs(), legacy_step.outputs());
    assert_eq!(unified_step.loss_weight(), 2);
    let checkpoint = unified_runtime.checkpoint().unwrap();
    assert_eq!(checkpoint, legacy_runtime.checkpoint().unwrap());
    assert_eq!(
        unified
            .restore_checkpoint(&checkpoint)
            .unwrap()
            .prepare_cpu()
            .unwrap()
            .checkpoint()
            .unwrap(),
        checkpoint
    );

    let owned = CompiledModuleAdamWPlan::compile_graph_with_dropout(
        token_weighted_config(2),
        token_mean_dropout(),
        TokenMeanModule::new(),
        build_token_graph,
    )
    .unwrap();
    assert_eq!(owned.capture_identity(), legacy.capture_identity());
    assert_eq!(owned.inspection().unwrap(), legacy.inspection().unwrap());

    let scalar_before = module.weight.snapshot().unwrap();
    let scalar_mismatch = CompiledAdamWPlan::compile_module_graph(
        token_weighted_config(2),
        &module,
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::scalar(
                graph.sum_all(losses)?,
                BTreeMap::new(),
            ))
        },
    );
    let scalar_error = match scalar_mismatch {
        Ok(_) => panic!("scalar objective compiled with token weighting"),
        Err(error) => error,
    };
    assert!(
        scalar_error
            .to_string()
            .contains("requires the token-mean-loss compile surface")
    );
    assert_parameter_snapshot_eq(&module.weight.snapshot().unwrap(), &scalar_before);

    let ordinary_config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_input("features", [3], DType::F32)
        .unwrap();
    let token_before = module.weight.snapshot().unwrap();
    let token_mismatch = CompiledAdamWPlan::compile_module_graph(
        ordinary_config.clone(),
        &module,
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            Ok(CompiledAdamWGraph::token_mean(
                graph.mul(weight, inputs["features"])?,
                BTreeMap::new(),
            ))
        },
    );
    let token_error = match token_mismatch {
        Ok(_) => panic!("token-mean objective compiled without token weighting"),
        Err(error) => error,
    };
    assert!(
        token_error
            .to_string()
            .contains("token-mean-loss compilation requires token weighting")
    );
    assert_parameter_snapshot_eq(&module.weight.snapshot().unwrap(), &token_before);

    let owned_module = TokenMeanModule::new();
    let owned_before = owned_module.weight.snapshot().unwrap();
    let owned_error = match CompiledModuleAdamWPlan::compile_graph(
        ordinary_config,
        owned_module,
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            Ok(CompiledAdamWGraph::token_mean(
                graph.mul(weight, inputs["features"])?,
                BTreeMap::new(),
            ))
        },
    ) {
        Ok(_) => panic!("token-mean objective compiled without its mask policy"),
        Err(error) => error,
    };
    assert!(
        owned_error
            .source_error()
            .to_string()
            .contains("token-mean-loss compilation requires token weighting")
    );
    let returned = owned_error.into_module();
    assert_parameter_snapshot_eq(&returned.weight.snapshot().unwrap(), &owned_before);
}

#[test]
fn token_mean_loss_surface_owns_normalization_and_rejects_scalar_seams() {
    let module = TokenMeanModule::new();
    let config = token_weighted_config(2);
    let invoked = Cell::new(false);
    let raw = CompiledAdamWPlan::compile(
        config.clone(),
        [TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap()],
        |_, _, _| {
            invoked.set(true);
            Err(training("scalar builder should not run"))
        },
    );
    assert!(raw.is_err());
    assert!(!invoked.get());

    let module_scalar = CompiledAdamWPlan::compile_module(config.clone(), &module, |_, _, _| {
        invoked.set(true);
        Err(training("scalar builder should not run"))
    });
    assert!(module_scalar.is_err());
    assert!(!invoked.get());

    let dropout_scalar = CompiledAdamWPlan::compile_module_with_dropout(
        config.clone(),
        token_mean_dropout(),
        &module,
        |_, _, _, _| {
            invoked.set(true);
            Err(training("scalar builder should not run"))
        },
    );
    assert!(dropout_scalar.is_err());
    assert!(!invoked.get());

    let without_policy = CompiledAdamWPlan::compile_token_mean_module_with_dropout(
        CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_input("features", [3], DType::F32)
            .unwrap()
            .with_input("mask", [3], DType::F32)
            .unwrap(),
        token_mean_dropout(),
        &module,
        |_, _, _, _| {
            invoked.set(true);
            Err(training("token-loss builder should not run"))
        },
    );
    assert!(without_policy.is_err());
    assert!(!invoked.get());

    for wrong_dtype in [false, true] {
        let invalid = CompiledAdamWPlan::compile_token_mean_module_with_dropout(
            config.clone(),
            token_mean_dropout(),
            &module,
            |module, graph, inputs, dropout| {
                let _ = dropout.dropout(graph, inputs["features"], 0.5)?;
                let weight = module.weight.bind(graph)?;
                let losses = graph.mul(weight, inputs["features"])?;
                let losses = if wrong_dtype {
                    graph.cast(losses, DType::I32)?
                } else {
                    graph.sum_all(losses)?
                };
                Ok((losses, BTreeMap::new()))
            },
        );
        assert!(invalid.is_err());
    }

    let plan = compile_token_weighted_plan(2);
    let mut runtime = plan.prepare_cpu().unwrap();
    let first = runtime
        .step(
            token_weighted_batch([1.0, 100.0, 1.0], [1.0, 0.0, 1.0]),
            TensorData::scalar(0.1),
        )
        .unwrap();
    assert_eq!(first.loss().scalar_at(0).as_f64(), 2.0);
    assert_eq!(first.loss_weight(), 2);
}

fn token_weighted_batch(features: [f32; 3], mask: [f32; 3]) -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        (
            "features".into(),
            TensorData::new([3], features.to_vec()).unwrap(),
        ),
        ("mask".into(), TensorData::new([3], mask.to_vec()).unwrap()),
    ])
}

fn ignore_index_config() -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(3)
        .unwrap()
        .with_input("features", [3], DType::F32)
        .unwrap()
        .with_input("targets", [3], DType::I32)
        .unwrap()
        .with_token_weighted_ignore_index("targets", -100)
        .unwrap()
        .with_zero_valid_token_microbatches()
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_window_loss_report()
}

fn ignore_index_batch(features: [f32; 3], targets: [i32; 3]) -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        (
            "features".into(),
            TensorData::new([3], features.to_vec()).unwrap(),
        ),
        (
            "targets".into(),
            TensorData::from_scalars(
                [3],
                DType::I32,
                targets.into_iter().map(|value| Scalar::I(i64::from(value))),
            )
            .unwrap(),
        ),
    ])
}

fn malformed_ignore_index_batch() -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        (
            "features".into(),
            TensorData::new([3], vec![1.0, 100.0, 3.0]).unwrap(),
        ),
        (
            "targets".into(),
            TensorData::new([3], vec![0.0, -100.0, 1.0]).unwrap(),
        ),
    ])
}

fn single_step_ignore_index_config() -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_input("features", [3], DType::F32)
        .unwrap()
        .with_input("targets", [3], DType::I32)
        .unwrap()
        .with_token_weighted_ignore_index("targets", -100)
        .unwrap()
        .with_zero_valid_token_microbatches()
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_clip_report()
        .with_window_loss_report()
}

fn compile_ignore_index_plan() -> CompiledAdamWPlan {
    let module = TokenMeanModule::new();
    CompiledAdamWPlan::compile_module_graph(
        ignore_index_config(),
        &module,
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
        },
    )
    .unwrap()
}

fn compile_ignore_index_context_plan() -> CompiledAdamWPlan {
    let module = TokenMeanModule::new();
    CompiledAdamWPlan::compile_module_graph_with_ignore_index(
        ignore_index_config(),
        &module,
        |module, graph, inputs, ignore_index| {
            assert_eq!(ignore_index.targets(), inputs["targets"]);
            assert_eq!(graph.shape(ignore_index.validity())?, &Shape::new([3]));
            assert_eq!(graph.dtype(ignore_index.validity())?, DType::Bool);
            assert_eq!(graph.shape(ignore_index.weight())?, &Shape::new([3]));
            assert_eq!(graph.dtype(ignore_index.weight())?, DType::F32);
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(
                losses,
                BTreeMap::from([
                    ("ignore_index_validity".into(), ignore_index.validity()),
                    ("ignore_index_weight".into(), ignore_index.weight()),
                ]),
            ))
        },
    )
    .unwrap()
}

#[test]
fn ignore_index_graph_context_shares_weight_and_authenticates_capture() {
    let legacy = compile_ignore_index_plan();
    let context = compile_ignore_index_context_plan();
    assert_ne!(legacy.capture_identity(), context.capture_identity());
    let matching_context = compile_ignore_index_context_plan();
    assert_eq!(
        context.capture_identity(),
        matching_context.capture_identity()
    );
    assert_eq!(
        context
            .inner
            .inputs
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["features", "targets"]
    );

    let mut runtime = context.prepare_cpu().unwrap();
    let initial = runtime.checkpoint().unwrap();
    let error = match runtime.step(malformed_ignore_index_batch(), TensorData::scalar(0.1)) {
        Ok(_) => panic!("malformed ignore-index target unexpectedly replayed"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("input descriptor mismatch"));
    assert_eq!(runtime.checkpoint().unwrap(), initial);

    let step = runtime
        .step(
            ignore_index_batch([1.0, 100.0, 3.0], [0, -100, 1]),
            TensorData::scalar(0.1),
        )
        .unwrap();
    assert_eq!(step.loss_weight(), 2);
    assert_eq!(
        step.outputs()["ignore_index_validity"].to_vec_f64(),
        [1.0, 0.0, 1.0]
    );
    assert_eq!(
        step.outputs()["ignore_index_weight"].to_vec_f64(),
        [1.0, 0.0, 1.0]
    );
    let checkpoint = runtime.checkpoint().unwrap();
    assert!(legacy.restore_checkpoint(&checkpoint).is_err());
    assert!(matching_context.restore_checkpoint(&checkpoint).is_ok());

    let invoked = Cell::new(false);
    let explicit = token_weighted_config(2);
    let error = CompiledAdamWPlan::compile_module_graph_with_ignore_index(
        explicit,
        &TokenMeanModule::new(),
        |_, _, _, _| {
            invoked.set(true);
            Err(training("ignore-index builder must not run"))
        },
    );
    assert!(error.is_err());
    assert!(!invoked.get());

    let repeated_legacy = compile_ignore_index_plan();
    assert_eq!(
        legacy.capture_identity(),
        repeated_legacy.capture_identity()
    );
    let legacy_checkpoint = legacy.prepare_cpu().unwrap().checkpoint().unwrap();
    assert_eq!(
        repeated_legacy.prepare_cpu().unwrap().checkpoint().unwrap(),
        legacy_checkpoint
    );
}

#[test]
fn ignore_index_graph_context_evaluation_is_read_only() {
    let owner = CompiledModuleAdamWPlan::compile_graph_with_ignore_index(
        ignore_index_config(),
        TokenMeanModule::new(),
        |module, graph, inputs, ignore_index| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(
                losses,
                BTreeMap::from([("ignore_index_validity".into(), ignore_index.validity())]),
            ))
        },
    )
    .unwrap()
    .with_evaluation_graph_and_ignore_index(|module, graph, inputs, ignore_index| {
        let weight = module.weight.bind(graph)?;
        let losses = graph.mul(weight, inputs["features"])?;
        Ok(CompiledAdamWGraph::token_mean(
            losses,
            BTreeMap::from([("ignore_index_weight".into(), ignore_index.weight())]),
        ))
    })
    .unwrap();
    let mut session = owner.prepare(&CpuSessionTarget::new()).unwrap();
    let before = session.checkpoint().unwrap();
    assert!(session.evaluate(malformed_ignore_index_batch()).is_err());
    assert_eq!(session.checkpoint().unwrap(), before);
    let evaluation = session
        .evaluate(ignore_index_batch([1.0, 100.0, 3.0], [0, -100, 1]))
        .unwrap();
    assert_eq!(evaluation.loss_weight(), 2);
    assert_eq!(
        evaluation.outputs()["ignore_index_weight"].to_vec_f64(),
        [1.0, 0.0, 1.0]
    );
    assert_eq!(session.checkpoint().unwrap(), before);
}

#[test]
fn single_step_ignore_index_artifact_restores_for_native_evaluation() {
    let builds = Cell::new(0);
    let owner = CompiledModuleAdamWPlan::compile_graph_with_ignore_index(
        single_step_ignore_index_config(),
        TokenMeanModule::new(),
        |module, graph, inputs, ignore_index| {
            builds.set(builds.get() + 1);
            assert_eq!(ignore_index.targets(), inputs["targets"]);
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
        },
    )
    .unwrap()
    .with_evaluation_graph_and_ignore_index(|module, graph, inputs, ignore_index| {
        let weight = module.weight.bind(graph)?;
        let losses = graph.mul(weight, inputs["features"])?;
        Ok(CompiledAdamWGraph::token_mean(
            losses,
            BTreeMap::from([("validity".into(), ignore_index.validity())]),
        ))
    })
    .unwrap();
    assert_eq!(builds.get(), 1);
    assert_eq!(owner.accumulation_capture_identity(), None);
    assert_eq!(owner.flush_capture_identity(), None);
    let artifact = owner.program_artifact().unwrap();
    assert_eq!(owner.program_artifact().unwrap(), artifact);
    assert_eq!(artifact.info().format_version(), 1);
    assert_eq!(artifact.as_bytes()[4], 1);
    program_artifact::rewrite_json_for_test(&artifact, |json| {
        assert!(json.get("metal").is_none());
    });
    let mut source = owner.prepare(&CpuSessionTarget).unwrap();

    let empty = || ignore_index_batch([7.0, 11.0, 13.0], [-100; 3]);
    let before_evaluation = source.checkpoint().unwrap();
    let evaluation = source.evaluate(empty()).unwrap();
    assert_eq!(evaluation.loss().scalar_at(0).as_f64(), 0.0);
    assert_eq!(evaluation.loss_weight(), 0);
    assert_eq!(source.checkpoint().unwrap(), before_evaluation);
    let step = source
        .step(
            ignore_index_batch([1.0, 100.0, 3.0], [0, -100, 1]),
            TensorData::scalar(0.1),
        )
        .unwrap();
    assert!(step.did_update());
    assert_eq!(step.loss_weight(), 2);
    assert_eq!(step.window_loss_report().unwrap().loss_weight(), 2);
    assert!(step.clip_report().is_some());
    let checkpoint = source.module_checkpoint().unwrap();
    assert_eq!(
        checkpoint
            .optimizer_checkpoint()
            .info()
            .accumulated_token_count(),
        None
    );
    let bundle = CompiledAdamWResumeBundle::new(artifact, checkpoint.clone()).unwrap();

    let destination = TokenMeanModule {
        weight: Parameter::new(TensorData::scalar(9.0), true),
    };
    let destination_before = destination.weight.snapshot().unwrap();
    let before_topology = program_artifact::portable_resume_decode_counts();
    let restored =
        CompiledModuleAdamWPlan::restore_from_resume_bundle(destination, &bundle).unwrap();
    let independent = CompiledModuleAdamWPlan::restore_from_resume_bundle(
        TokenMeanModule {
            weight: Parameter::new(TensorData::scalar(11.0), true),
        },
        &bundle,
    )
    .unwrap();
    let after_topology = program_artifact::portable_resume_decode_counts();
    assert_eq!(
        after_topology.topology_seals,
        before_topology.topology_seals + 1
    );
    assert_eq!(
        after_topology.topology_phase_validations,
        before_topology.topology_phase_validations + 1
    );
    assert_eq!(
        after_topology.recurrent_execution_plans,
        before_topology.recurrent_execution_plans + 1
    );
    assert_eq!(
        after_topology.evaluation_execution_plans,
        before_topology.evaluation_execution_plans + 1
    );
    assert_eq!(
        after_topology.cursor_projections,
        before_topology.cursor_projections
    );
    assert_eq!(
        restored.plan.topology_allocations(),
        independent.plan.topology_allocations()
    );
    assert!(restored.plan.topology_allocations().accumulation.is_none());
    assert!(restored.plan.topology_allocations().partial_flush.is_none());
    assert!(restored.plan.topology_allocations().zero_grad.is_none());
    assert_parameter_snapshot_eq(
        &restored.module.weight.snapshot().unwrap(),
        &destination_before,
    );
    assert_eq!(
        builds.get(),
        1,
        "artifact restore must not rerun the builder"
    );
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut restored = restored.prepare(&target).unwrap();
    assert_eq!(
        restored.checkpoint().unwrap(),
        checkpoint.optimizer_checkpoint().clone()
    );
    assert!(
        restored
            .native_cpu_preparation_report()
            .accumulation()
            .is_none()
    );
    assert!(
        restored
            .native_cpu_preparation_report()
            .partial_flush()
            .is_none()
    );
    assert!(
        restored
            .native_cpu_preparation_report()
            .zero_grad()
            .is_none()
    );
    let native = restored
        .step(
            ignore_index_batch([2.0, 200.0, 4.0], [0, -100, 1]),
            TensorData::scalar(0.1),
        )
        .unwrap();
    assert!(native.did_update());
    assert_eq!(native.loss_weight(), 2);
    assert_eq!(native.report().fallback_count(), 0);
    let before_evaluation = restored.checkpoint().unwrap();
    let evaluation = restored.evaluate(empty()).unwrap();
    assert_eq!(evaluation.loss_weight(), 0);
    assert_eq!(evaluation.report().fallback_count(), 0);
    assert_eq!(restored.checkpoint().unwrap(), before_evaluation);
}

#[test]
fn ignore_index_token_weighting_is_unbiased_atomic_and_cpu_portable() {
    let plan = compile_ignore_index_plan();
    assert_eq!(plan.token_weighted_gradient_accumulation_mask(), None);
    assert_eq!(plan.token_weighted_ignore_index(), Some(("targets", -100)));
    let mut interpreted = plan.prepare_cpu().unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = plan.prepare(&target).unwrap();
    assert_core_training_window(&interpreted, 3, 0);
    assert_core_training_window(&native, 3, 0);

    let nonempty = || ignore_index_batch([1.0, 100.0, 3.0], [0, -100, 1]);
    let empty = || ignore_index_batch([100.0, 200.0, 300.0], [-100; 3]);
    let interpreted_first = interpreted
        .step(nonempty(), TensorData::scalar(0.1))
        .unwrap();
    let native_first = native.step(nonempty(), TensorData::scalar(0.1)).unwrap();
    assert_eq!(interpreted_first.loss_weight(), 2);
    assert_eq!(native_first.loss_weight(), 2);
    assert_core_loss_aggregation_weight(&interpreted_first, 2);
    assert_core_loss_aggregation_weight(&native_first, 2);
    assert_core_training_window_step(&interpreted_first, 1, false);
    assert_core_training_window_step(&native_first, 1, false);
    assert_core_training_window(&interpreted, 3, 1);
    assert_core_training_window(&native, 3, 1);
    assert!(!interpreted_first.did_update());
    assert!(!native_first.did_update());
    let interpreted_accumulators = interpreted.gradient_accumulator_snapshots().unwrap();
    let native_accumulators = native.gradient_accumulator_snapshots().unwrap();
    let interpreted_empty = interpreted.step(empty(), TensorData::scalar(0.1)).unwrap();
    let native_empty = native.step(empty(), TensorData::scalar(0.1)).unwrap();
    assert_eq!(interpreted_empty.loss().scalar_at(0).as_f64(), 0.0);
    assert_eq!(native_empty.loss().scalar_at(0).as_f64(), 0.0);
    assert_eq!(interpreted_empty.loss_weight(), 0);
    assert_eq!(native_empty.loss_weight(), 0);
    assert_core_loss_aggregation_weight(&interpreted_empty, 0);
    assert_core_loss_aggregation_weight(&native_empty, 0);
    assert_core_training_window_step(&interpreted_empty, 2, false);
    assert_core_training_window_step(&native_empty, 2, false);
    assert_core_training_window(&interpreted, 3, 2);
    assert_core_training_window(&native, 3, 2);
    assert!(!interpreted_empty.did_update());
    assert!(!native_empty.did_update());
    assert_eq!(
        interpreted.gradient_accumulator_snapshots().unwrap(),
        interpreted_accumulators
    );
    assert_eq!(
        native.gradient_accumulator_snapshots().unwrap(),
        native_accumulators
    );
    assert_eq!(interpreted.accumulation_index().unwrap(), 2);
    assert_eq!(native.accumulation_index().unwrap(), 2);
    let checkpoint = interpreted.checkpoint().unwrap();
    assert_eq!(checkpoint.info().accumulated_token_count(), Some(2));
    let mut restored = plan
        .restore_checkpoint(&checkpoint)
        .unwrap()
        .prepare_cpu()
        .unwrap();

    let commit = ignore_index_batch([7.0, 8.0, 9.0], [-100, 2, -100]);
    let interpreted_step = interpreted
        .step(commit.clone(), TensorData::scalar(0.1))
        .unwrap();
    let restored_step = restored.step(commit, TensorData::scalar(0.1)).unwrap();
    let native_step = native
        .step(
            ignore_index_batch([7.0, 8.0, 9.0], [-100, 2, -100]),
            TensorData::scalar(0.1),
        )
        .unwrap();
    assert!(interpreted_step.did_update());
    assert!(native_step.did_update());
    assert_core_training_window_step(&interpreted_step, 0, true);
    assert_core_training_window_step(&restored_step, 0, true);
    assert_core_training_window_step(&native_step, 0, true);
    assert_core_training_window(&interpreted, 3, 0);
    assert_core_training_window(&restored, 3, 0);
    assert_core_training_window(&native, 3, 0);
    assert_eq!(interpreted_step.loss_weight(), 1);
    assert_eq!(native_step.loss_weight(), 1);
    assert_eq!(restored_step.loss(), interpreted_step.loss());
    assert_eq!(
        restored.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );

    let mut rejected = compile_ignore_index_plan().prepare_cpu().unwrap();
    rejected.step(empty(), TensorData::scalar(0.1)).unwrap();
    rejected.step(empty(), TensorData::scalar(0.1)).unwrap();
    let before = rejected.checkpoint().unwrap();
    assert!(rejected.step(empty(), TensorData::scalar(0.1)).is_err());
    assert_eq!(rejected.checkpoint().unwrap(), before);
    assert_core_training_window(&rejected, 3, 2);
    let retried = rejected.step(nonempty(), TensorData::scalar(0.1)).unwrap();
    assert!(retried.did_update());
    assert_core_training_window_step(&retried, 0, true);
    assert_core_training_window(&rejected, 3, 0);

    let mut native_rejected = compile_ignore_index_plan().prepare(&target).unwrap();
    native_rejected
        .step(empty(), TensorData::scalar(0.1))
        .unwrap();
    native_rejected
        .step(empty(), TensorData::scalar(0.1))
        .unwrap();
    let before = native_rejected.checkpoint().unwrap();
    assert!(
        native_rejected
            .step(empty(), TensorData::scalar(0.1))
            .is_err()
    );
    assert_eq!(native_rejected.checkpoint().unwrap(), before);
    assert_core_training_window(&native_rejected, 3, 2);
    let native_retried = native_rejected
        .step(nonempty(), TensorData::scalar(0.1))
        .unwrap();
    assert!(native_retried.did_update());
    assert_core_training_window_step(&native_retried, 0, true);
    assert_core_training_window(&native_rejected, 3, 0);
}

#[test]
fn ignore_index_policy_rejects_invalid_or_repeated_descriptors() {
    let base = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(3)
        .unwrap();
    assert!(
        base.clone()
            .with_token_weighted_ignore_index("targets", -100)
            .is_err()
    );
    assert!(
        base.clone()
            .with_input("targets", [3], DType::F32)
            .unwrap()
            .with_token_weighted_ignore_index("targets", -100)
            .is_err()
    );
    let configured = base
        .with_input("targets", [3], DType::I32)
        .unwrap()
        .with_input("mask", [3], DType::F32)
        .unwrap()
        .with_token_weighted_ignore_index("targets", -100)
        .unwrap();
    assert!(
        configured
            .clone()
            .with_token_weighted_gradient_accumulation("mask")
            .is_err()
    );
    assert_eq!(
        configured.token_weighted_ignore_index(),
        Some(("targets", -100))
    );
}

fn token_evaluation_config() -> CompiledAdamWConfig {
    CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_input("features", [6], DType::F32)
        .unwrap()
        .with_input("mask", [6], DType::F32)
        .unwrap()
        .with_token_weighted_gradient_accumulation("mask")
        .unwrap()
}

fn compile_token_training_owner() -> CompiledModuleAdamWPlan<TokenMeanModule> {
    CompiledModuleAdamWPlan::compile_graph(
        token_evaluation_config(),
        TokenMeanModule::new(),
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
        },
    )
    .unwrap()
}

fn compile_token_evaluation_plan() -> CompiledModuleAdamWPlan<TokenMeanModule> {
    compile_token_training_owner()
        .with_evaluation_graph(|module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(
                losses,
                BTreeMap::from([("losses".into(), losses)]),
            ))
        })
        .unwrap()
}

fn token_evaluation_batch(mask: [f32; 6]) -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        (
            "features".into(),
            TensorData::new([6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]).unwrap(),
        ),
        ("mask".into(), TensorData::new([6], mask.to_vec()).unwrap()),
    ])
}

#[test]
fn unified_token_mean_evaluation_weights_batches_and_preserves_frontier() {
    let masks = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ];
    let mut interpreted = compile_token_evaluation_plan()
        .prepare(&CpuSessionTarget::new())
        .unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = compile_token_evaluation_plan().prepare(&target).unwrap();
    assert_eq!(
        interpreted.evaluation_capture_identity(),
        native.evaluation_capture_identity()
    );
    let interpreted_checkpoint = interpreted.checkpoint().unwrap();
    let native_checkpoint = native.checkpoint().unwrap();
    let interpreted_accumulators = interpreted.gradient_accumulator_snapshots().unwrap();
    let native_accumulators = native.gradient_accumulator_snapshots().unwrap();
    let native_evaluation_workspace = native
        .runtime()
        .evaluation_replay
        .as_ref()
        .unwrap()
        .plan
        .workspace_stats();
    let native_recurrent_counts = native_recurrent_test_counts(native.runtime());

    for mask in [
        [1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ] {
        assert!(interpreted.evaluate(token_evaluation_batch(mask)).is_err());
        assert!(native.evaluate(token_evaluation_batch(mask)).is_err());
    }
    assert_eq!(native.runtime().successful_evaluations, 0);
    assert_eq!(
        native_recurrent_test_counts(native.runtime()),
        native_recurrent_counts
    );
    assert_eq!(
        native
            .runtime()
            .evaluation_replay
            .as_ref()
            .unwrap()
            .plan
            .workspace_stats(),
        native_evaluation_workspace
    );
    assert_eq!(interpreted.checkpoint().unwrap(), interpreted_checkpoint);
    assert_eq!(native.checkpoint().unwrap(), native_checkpoint);

    let mut interpreted_weighted_loss = 0.0;
    let mut native_weighted_loss = 0.0;
    let mut total_weight = 0_u64;
    for (invocation, (mask, expected_weight)) in masks.into_iter().zip([5, 3, 3]).enumerate() {
        let expected = interpreted.evaluate(token_evaluation_batch(mask)).unwrap();
        let before_native_evaluation = native_recurrent_test_counts(native.runtime());
        let actual = native.evaluate(token_evaluation_batch(mask)).unwrap();
        assert_eq!(
            native_recurrent_test_counts(native.runtime()),
            before_native_evaluation
        );
        assert_eq!(expected.loss_weight(), expected_weight);
        assert_eq!(actual.loss_weight(), expected_weight);
        assert_cross_engine_tensor_close(
            "token-mean evaluation loss",
            actual.loss(),
            expected.loss(),
        );
        assert_cross_engine_tensor_maps_close(
            "token-mean evaluation outputs",
            actual.outputs(),
            expected.outputs(),
        );
        assert_eq!(
            actual.report().successful_invocation(),
            invocation as u64 + 1
        );
        assert_eq!(actual.report().traffic().external_input_import_count(), 0);
        assert_eq!(actual.report().traffic().external_input_import_bytes(), 0);
        assert_eq!(
            actual.report().traffic().borrowed_recurrent_input_bytes(),
            4
        );
        assert_eq!(
            actual.report().traffic().borrowed_recurrent_output_bytes(),
            0
        );
        interpreted_weighted_loss += expected.loss().scalar_at(0).as_f64() * expected_weight as f64;
        native_weighted_loss += actual.loss().scalar_at(0).as_f64() * expected_weight as f64;
        total_weight += expected_weight;
    }
    assert_eq!(total_weight, 11);
    assert!((interpreted_weighted_loss / total_weight as f64 - 54.0 / 11.0).abs() < 1e-6);
    assert!((native_weighted_loss - interpreted_weighted_loss).abs() < 1e-5);
    assert_eq!(interpreted.checkpoint().unwrap(), interpreted_checkpoint);
    assert_eq!(native.checkpoint().unwrap(), native_checkpoint);
    assert_eq!(
        interpreted.gradient_accumulator_snapshots().unwrap(),
        interpreted_accumulators
    );
    assert_eq!(
        native.gradient_accumulator_snapshots().unwrap(),
        native_accumulators
    );
    assert_eq!(interpreted.step_count(), 0);
    assert_eq!(native.step_count(), 0);
    let native_evaluation_workspace = native
        .runtime()
        .evaluation_replay
        .as_ref()
        .unwrap()
        .plan
        .workspace_stats();
    assert_eq!(native_evaluation_workspace.input_import_count, 0);
    assert_eq!(
        native_evaluation_workspace.borrowed_external_input_bytes,
        144
    );
    assert_eq!(
        native_evaluation_workspace.borrowed_recurrent_input_bytes,
        12
    );
    assert_eq!(
        native_evaluation_workspace.borrowed_recurrent_output_bytes,
        0
    );
}

#[test]
fn prepared_native_checkpoint_restore_reuses_programs_and_lifetime_counters() {
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let batch = || token_evaluation_batch([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    let preparation_identity = |runtime: &NativeCpuCompiledAdamW<'_>| {
        let preparation = runtime.preparation_report();
        std::iter::once(preparation.main())
            .chain(preparation.accumulation())
            .chain(preparation.partial_flush())
            .chain(preparation.zero_grad())
            .chain(preparation.evaluation())
            .map(|program| {
                (
                    program.capture_identity(),
                    program.native_identity(),
                    program.native_item_count(),
                    program.cache_hit_count(),
                    program.cache_miss_count(),
                )
            })
            .collect::<Vec<_>>()
    };

    let mut runtime = compile_token_evaluation_plan()
        .prepare(&target)
        .unwrap()
        .into_training_session();
    let preparation = preparation_identity(runtime.runtime());
    let native_plan_count = executor.native_item_plan_count();
    let evaluated = runtime.evaluate(batch()).unwrap();
    assert_eq!(evaluated.report().successful_invocation(), 1);
    let first = runtime.step(batch(), TensorData::scalar(0.01)).unwrap();
    assert!(!first.did_update());
    let pending = runtime.checkpoint().unwrap();
    assert!(
        runtime
            .step(batch(), TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert!(
        !runtime
            .step(batch(), TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert!(runtime.reset_gradient_window().unwrap().did_discard());
    assert_eq!(runtime.runtime().successful_steps, 3);
    assert_eq!(runtime.runtime().successful_zero_grads, 1);

    runtime.restore_checkpoint_in_place(&pending).unwrap();
    assert_eq!(runtime.checkpoint().unwrap(), pending);
    assert_eq!(preparation_identity(runtime.runtime()), preparation);
    assert_eq!(executor.native_item_plan_count(), native_plan_count);
    assert_eq!(runtime.runtime().successful_steps, 3);
    assert_eq!(runtime.runtime().successful_zero_grads, 1);
    assert_eq!(runtime.runtime().successful_evaluations, 1);

    let mut reference = compile_token_evaluation_plan()
        .restore_checkpoint(&pending)
        .unwrap()
        .prepare(&target)
        .unwrap()
        .into_training_session();
    let actual_evaluation = runtime.evaluate(batch()).unwrap();
    let expected_evaluation = reference.evaluate(batch()).unwrap();
    assert_eq!(actual_evaluation.loss(), expected_evaluation.loss());
    assert_eq!(actual_evaluation.outputs(), expected_evaluation.outputs());
    assert_eq!(actual_evaluation.report().successful_invocation(), 2);
    assert_eq!(expected_evaluation.report().successful_invocation(), 1);

    let actual_flush = runtime
        .commit_partial_window(TensorData::scalar(0.01))
        .unwrap();
    let expected_flush = reference
        .commit_partial_window(TensorData::scalar(0.01))
        .unwrap();
    assert!(actual_flush.did_update());
    assert_eq!(
        actual_flush.optimizer_step(),
        expected_flush.optimizer_step()
    );
    assert_eq!(
        runtime.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );

    let actual = runtime.step(batch(), TensorData::scalar(0.01)).unwrap();
    let expected = reference.step(batch(), TensorData::scalar(0.01)).unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());
    assert_eq!(actual.report().successful_invocation(), 4);
    assert_eq!(expected.report().successful_invocation(), 1);
    assert_eq!(
        runtime.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );

    let checkpoint = runtime.checkpoint().unwrap();
    let expected_weight = decode_adamw_checkpoint(checkpoint.as_bytes())
        .unwrap()
        .parameters["weight"]
        .clone();
    let module = runtime.finish().unwrap();
    assert_eq!(module.weight.value().unwrap(), expected_weight);
}

#[test]
fn prepared_native_restore_accepts_frontiers_older_than_preparation() {
    let batch = || token_evaluation_batch([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    let mut source = compile_token_evaluation_plan()
        .prepare(&CpuSessionTarget)
        .unwrap();
    assert!(
        !source
            .step(batch(), TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    let older = source.checkpoint().unwrap();
    assert_eq!(older.info().replay_step(), 1);
    assert_eq!(older.info().optimizer_step(), 0);
    assert_eq!(older.info().accumulation_index(), 1);
    assert!(
        source
            .step(batch(), TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert!(
        !source
            .step(batch(), TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert!(source.zero_grad().unwrap().did_discard());
    assert!(
        !source
            .step(batch(), TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert!(
        source
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    let newer = source.checkpoint().unwrap();
    assert_eq!(newer.info().replay_step(), 4);
    assert_eq!(newer.info().optimizer_step(), 2);
    assert_eq!(newer.info().accumulation_index(), 0);
    assert_eq!(newer.info().reset_transition_count(), 1);
    assert_eq!(newer.info().flushed_window_count(), 1);

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut runtime = compile_token_evaluation_plan()
        .restore_checkpoint(&newer)
        .unwrap()
        .prepare(&target)
        .unwrap();
    let preparation = format!("{:?}", runtime.runtime().preparation_report());
    let native_plan_count = executor.native_item_plan_count();
    let workspace_stats = |runtime: &NativeCpuCompiledAdamW<'_>| {
        std::iter::once(runtime.main_replay.workspace_stats())
            .chain(
                runtime
                    .accumulation_replay
                    .iter()
                    .map(PreparedRecurrentNativeReplay::workspace_stats),
            )
            .chain(
                runtime
                    .partial_flush_replay
                    .iter()
                    .map(PreparedRecurrentNativeReplay::workspace_stats),
            )
            .chain(
                runtime
                    .zero_grad_replay
                    .iter()
                    .map(PreparedRecurrentNativeReplay::workspace_stats),
            )
            .chain(
                runtime
                    .evaluation_replay
                    .iter()
                    .map(|evaluation| evaluation.plan.workspace_stats()),
            )
            .collect::<Vec<_>>()
    };
    runtime.evaluate(batch()).unwrap();
    runtime.step(batch(), TensorData::scalar(0.01)).unwrap();
    let workspaces = workspace_stats(runtime.runtime());

    runtime.restore_checkpoint_in_place(&older).unwrap();
    assert_eq!(runtime.checkpoint().unwrap(), older);
    crate::host_buffer::reset_host_bank_transaction_test_counts();
    assert_eq!(
        format!("{:?}", runtime.runtime().preparation_report()),
        preparation
    );
    assert_eq!(workspace_stats(runtime.runtime()), workspaces);
    assert_eq!(executor.native_item_plan_count(), native_plan_count);
    assert_eq!(runtime.runtime().successful_steps, 1);
    assert_eq!(runtime.runtime().successful_evaluations, 1);

    assert!(runtime.zero_grad().unwrap().did_discard());
    assert_eq!(
        crate::host_buffer::host_bank_transaction_test_counts(),
        crate::host_buffer::HostBankTransactionTestCounts {
            ordered_full_frontier_transactions: 0,
            request_map_builds: 1,
            ordinal_sorts: 1,
        },
        "zero-grad projects only reset states and retains the generic subset transaction"
    );
    let reset_replay = runtime.runtime().zero_grad_replay.as_ref().unwrap();
    assert!(reset_replay.last_executed_native_item_count() > 0);
    let reset_dispatch = reset_replay.last_module_dispatch_counts();
    assert!(reset_dispatch.0 > 0);
    assert_eq!(
        reset_dispatch.1,
        reset_replay.last_executed_native_item_count()
    );
    runtime.restore_checkpoint_in_place(&older).unwrap();
    assert!(
        runtime
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert_eq!(
        crate::host_buffer::host_bank_transaction_test_counts(),
        crate::host_buffer::HostBankTransactionTestCounts {
            ordered_full_frontier_transactions: 1,
            request_map_builds: 1,
            ordinal_sorts: 1,
        },
        "partial flush uses the canonical full-frontier transaction"
    );
    runtime.restore_checkpoint_in_place(&older).unwrap();
    assert_eq!(runtime.runtime().successful_zero_grads, 1);
    assert_eq!(runtime.runtime().successful_flushes, 1);
    assert_eq!(executor.native_item_plan_count(), native_plan_count);

    let mut reference = compile_token_evaluation_plan()
        .restore_checkpoint(&older)
        .unwrap()
        .prepare(&target)
        .unwrap();
    let actual_evaluation = runtime.evaluate(batch()).unwrap();
    let expected_evaluation = reference.evaluate(batch()).unwrap();
    assert_eq!(actual_evaluation.loss(), expected_evaluation.loss());
    assert_eq!(actual_evaluation.outputs(), expected_evaluation.outputs());

    let actual_main = runtime.step(batch(), TensorData::scalar(0.01)).unwrap();
    let expected_main = reference.step(batch(), TensorData::scalar(0.01)).unwrap();
    assert!(actual_main.did_update());
    assert_eq!(actual_main.loss(), expected_main.loss());
    assert_eq!(actual_main.outputs(), expected_main.outputs());
    assert_eq!(
        runtime.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );

    let actual_accumulation = runtime.step(batch(), TensorData::scalar(0.01)).unwrap();
    let expected_accumulation = reference.step(batch(), TensorData::scalar(0.01)).unwrap();
    assert!(!actual_accumulation.did_update());
    assert_eq!(actual_accumulation.loss(), expected_accumulation.loss());
    assert_eq!(
        actual_accumulation.outputs(),
        expected_accumulation.outputs()
    );
    assert_eq!(
        runtime.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );
    assert_eq!(runtime.runtime().successful_steps, 3);
    assert_eq!(runtime.runtime().successful_evaluations, 2);
    assert_eq!(executor.native_item_plan_count(), native_plan_count * 2);
    assert_eq!(
        crate::host_buffer::host_bank_transaction_test_counts(),
        crate::host_buffer::HostBankTransactionTestCounts {
            ordered_full_frontier_transactions: 5,
            request_map_builds: 1,
            ordinal_sorts: 1,
        },
        "restored main, accumulation, and flush use canonical order while reset keeps the subset fallback"
    );
}

#[test]
fn legacy_scalar_evaluation_keeps_unit_weight() {
    let legacy = CompiledModuleAdamWPlan::compile(
        module_config(),
        TiedFrozenModule::new([0.1, -0.2]),
        build_tied_frozen,
    )
    .unwrap()
    .with_evaluation(build_tied_frozen)
    .unwrap();
    let unified = CompiledModuleAdamWPlan::compile_graph(
        module_config(),
        TiedFrozenModule::new([0.1, -0.2]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
        Ok(CompiledAdamWGraph::scalar(loss, outputs))
    })
    .unwrap();
    assert_eq!(legacy.capture_identity(), unified.capture_identity());
    assert_eq!(
        legacy.evaluation_capture_identity(),
        unified.evaluation_capture_identity()
    );
    let mut legacy = legacy.prepare(&CpuSessionTarget::new()).unwrap();
    let mut unified = unified.prepare(&CpuSessionTarget::new()).unwrap();
    let inputs = || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
    let expected = legacy.evaluate(inputs()).unwrap();
    let actual = unified.evaluate(inputs()).unwrap();
    assert_eq!(expected.loss_weight(), 1);
    assert_eq!(actual.loss_weight(), 1);
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());

    let mismatch = compile_token_training_owner().with_evaluation_graph(|module, graph, inputs| {
        let weight = module.weight.bind(graph)?;
        let losses = graph.mul(weight, inputs["features"])?;
        Ok(CompiledAdamWGraph::scalar(
            graph.sum_all(losses)?,
            BTreeMap::new(),
        ))
    });
    let error = match mismatch {
        Ok(_) => panic!("scalar evaluation bypassed token-mean policy"),
        Err(error) => error,
    };
    assert!(
        error
            .source_error()
            .to_string()
            .contains("requires the token-mean-loss compile surface")
    );
}

#[test]
fn token_weighted_accumulation_is_atomic_native_consistent_and_checkpointed() {
    let plan = compile_token_weighted_plan(2);
    assert_eq!(
        plan.token_weighted_gradient_accumulation_mask(),
        Some("mask")
    );
    let renderer = MetalRenderer::new(
        8,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: 1 << 30,
            unified_memory: true,
            family: "Apple9".into(),
        },
    )
    .unwrap();
    let error = match plan.metal_plan(renderer) {
        Ok(_) => panic!("token-weighted accumulation rendered for Metal"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("token-weighted accumulation is currently CPU-only")
    );
    let mut interpreted = plan.prepare_cpu().unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native = target.prepare(&plan).unwrap();
    let initial = interpreted.checkpoint().unwrap();
    for invalid_mask in [[f32::NAN, 1.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]] {
        let batch = token_weighted_batch([1.0, 1.0, 100.0], invalid_mask);
        assert!(
            interpreted
                .step(batch.clone(), TensorData::scalar(0.1))
                .is_err()
        );
        assert!(native.step(batch, TensorData::scalar(0.1)).is_err());
        assert_eq!(interpreted.checkpoint().unwrap(), initial);
        assert_eq!(native.checkpoint().unwrap(), initial);
    }

    let first_batch = token_weighted_batch([1.0, 100.0, 1.0], [1.0, -0.0, 1.0]);
    let interpreted_first = interpreted
        .step(first_batch.clone(), TensorData::scalar(0.1))
        .unwrap();
    assert!(!interpreted_first.did_update());
    assert_eq!(interpreted_first.loss_weight(), 2);
    let native_first = native.step(first_batch, TensorData::scalar(0.1)).unwrap();
    assert_eq!(native_first.loss_weight(), interpreted_first.loss_weight());
    assert_eq!(native_first.report().successful_invocation(), 1);
    assert_eq!(native_first.report().fallback_count(), 0);
    let checkpoint = interpreted.checkpoint().unwrap();
    let native_checkpoint = native.checkpoint().unwrap();
    assert_native_adamw_state_close(&native, &interpreted);
    assert_eq!(checkpoint.info().accumulated_token_count(), Some(2));
    let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V9);
    let (mut tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    tensors.insert(
        "accumulated_token_count".into(),
        TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(0)]).unwrap(),
    );
    let empty_count =
        CompiledAdamWCheckpoint::from_bytes(save_safetensors(&tensors, &metadata).unwrap())
            .unwrap();
    assert!(plan.restore_checkpoint(&empty_count).is_err());
    let (mut tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    tensors.insert(
        "accumulated_token_count".into(),
        TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(4)]).unwrap(),
    );
    let impossible =
        CompiledAdamWCheckpoint::from_bytes(save_safetensors(&tensors, &metadata).unwrap())
            .unwrap();
    assert!(plan.restore_checkpoint(&impossible).is_err());

    let mut restored = plan
        .restore_checkpoint(&checkpoint)
        .unwrap()
        .prepare_cpu()
        .unwrap();
    let native_restored_plan = plan.restore_checkpoint(&native_checkpoint).unwrap();
    let mut native_restored = target.prepare(&native_restored_plan).unwrap();
    let second_batch = token_weighted_batch([3.0, 100.0, 100.0], [1.0, 0.0, 0.0]);
    let interpreted_second = interpreted
        .step(second_batch.clone(), TensorData::scalar(0.1))
        .unwrap();
    let restored_second = restored
        .step(second_batch.clone(), TensorData::scalar(0.1))
        .unwrap();
    let native_second = native
        .step(second_batch.clone(), TensorData::scalar(0.1))
        .unwrap();
    let native_restored_second = native_restored
        .step(second_batch, TensorData::scalar(0.1))
        .unwrap();
    assert_eq!(interpreted_second.loss_weight(), 1);
    assert_eq!(restored_second.loss_weight(), 1);
    assert_eq!(native_second.loss_weight(), 1);
    assert_eq!(native_restored_second.loss_weight(), 1);
    assert_eq!(native_second.report().fallback_count(), 0);
    assert_eq!(
        restored.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );
    assert_eq!(
        native_restored.checkpoint().unwrap(),
        native.checkpoint().unwrap()
    );
    assert_native_adamw_state_close(&native, &interpreted);
    let first_moment = interpreted.first_moment_snapshots().unwrap()["weight"]
        .scalar_at(0)
        .as_f64();
    assert!((first_moment - 5.0 / 3.0).abs() < 1e-6);
    assert_eq!(
        interpreted
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_token_count(),
        Some(0)
    );
}

#[test]
fn zero_token_microbatches_are_unbiased_checkpointed_and_reject_empty_windows() {
    let plan = compile_zero_token_weighted_plan(3);
    assert!(plan.zero_valid_token_microbatches_enabled());
    assert_ne!(
        plan.capture_identity(),
        compile_token_weighted_plan(3).capture_identity()
    );
    let empty = || token_weighted_batch([17.0, -7.0, 13.0], [0.0, 0.0, 0.0]);
    let valid = || token_weighted_batch([3.0, 3.0, 3.0], [1.0, 1.0, 1.0]);
    let learning_rate = || TensorData::scalar(0.1);

    let mut mixed = plan.prepare_cpu().unwrap();
    let first = mixed.step(empty(), learning_rate()).unwrap();
    assert!(!first.did_update());
    assert_eq!(first.loss_weight(), 0);
    assert_eq!(
        first.loss().scalar_at(0).as_f64().to_bits(),
        0.0_f64.to_bits()
    );
    assert!(first.window_loss_report().is_none());
    let first_checkpoint = mixed.checkpoint().unwrap();
    assert_eq!(first_checkpoint.info().accumulated_token_count(), Some(0));
    let mut restored = plan
        .restore_checkpoint(&first_checkpoint)
        .unwrap()
        .prepare_cpu()
        .unwrap();
    assert_eq!(restored.checkpoint().unwrap(), first_checkpoint);

    for runtime in [&mut mixed, &mut restored] {
        assert!(!runtime.step(valid(), learning_rate()).unwrap().did_update());
        let committed = runtime.step(empty(), learning_rate()).unwrap();
        assert!(committed.did_update());
        let report = committed.window_loss_report().unwrap();
        assert_eq!(report.mean_loss(), 6.0);
        assert_eq!(report.loss_weight(), 3);
        assert_eq!(report.microbatch_count(), 3);
    }
    assert_eq!(restored.checkpoint().unwrap(), mixed.checkpoint().unwrap());

    let mut reordered = plan.prepare_cpu().unwrap();
    reordered.step(valid(), learning_rate()).unwrap();
    reordered.step(empty(), learning_rate()).unwrap();
    reordered.step(empty(), learning_rate()).unwrap();
    assert_eq!(reordered.checkpoint().unwrap(), mixed.checkpoint().unwrap());

    let executor = CapturedReplayExecutor::default();
    let mut native = plan
        .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
        .unwrap();
    for batch in [empty(), valid(), empty()] {
        let result = native.step(batch, learning_rate()).unwrap();
        assert_eq!(result.report().fallback_count(), 0);
    }
    assert_native_adamw_state_close(&native, &mixed);

    let mut all_empty = plan.prepare_cpu().unwrap();
    let mut native_all_empty = plan
        .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
        .unwrap();
    for _ in 0..2 {
        assert!(
            !all_empty
                .step(empty(), learning_rate())
                .unwrap()
                .did_update()
        );
        let result = native_all_empty.step(empty(), learning_rate()).unwrap();
        assert!(!result.did_update());
        assert_eq!(result.report().fallback_count(), 0);
    }
    let before = all_empty.checkpoint().unwrap();
    let native_before = native_all_empty.checkpoint().unwrap();
    assert!(all_empty.step(empty(), learning_rate()).is_err());
    assert!(native_all_empty.step(empty(), learning_rate()).is_err());
    assert_eq!(all_empty.checkpoint().unwrap(), before);
    assert_eq!(native_all_empty.checkpoint().unwrap(), native_before);
    let retry = all_empty.step(valid(), learning_rate()).unwrap();
    let native_retry = native_all_empty.step(valid(), learning_rate()).unwrap();
    assert!(retry.did_update());
    assert!(native_retry.did_update());
    assert_eq!(native_retry.report().fallback_count(), 0);
    assert_native_adamw_state_close(&native_all_empty, &all_empty);

    let mut partial = plan.prepare_cpu().unwrap();
    let mut native_partial = plan
        .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
        .unwrap();
    partial.step(empty(), learning_rate()).unwrap();
    native_partial.step(empty(), learning_rate()).unwrap();
    let partial_before = partial.checkpoint().unwrap();
    let native_partial_before = native_partial.checkpoint().unwrap();
    assert!(partial.flush_partial_window(learning_rate()).is_err());
    assert!(
        native_partial
            .flush_partial_window(learning_rate())
            .is_err()
    );
    assert_eq!(partial.checkpoint().unwrap(), partial_before);
    assert_eq!(native_partial.checkpoint().unwrap(), native_partial_before);
    assert_eq!(partial.zero_grad().unwrap().discarded_microbatches(), 1);
    assert_eq!(
        native_partial.zero_grad().unwrap().discarded_microbatches(),
        1
    );
    assert_eq!(
        partial
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_token_count(),
        Some(0)
    );

    let mut evaluation = CompiledModuleAdamWPlan::compile_graph(
        token_evaluation_config()
            .with_zero_valid_token_microbatches()
            .unwrap(),
        TokenMeanModule::new(),
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let weight = module.weight.bind(graph)?;
        let losses = graph.mul(weight, inputs["features"])?;
        Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
    })
    .unwrap()
    .prepare(&CpuSessionTarget::new())
    .unwrap();
    let evaluation_before = evaluation.checkpoint().unwrap();
    let evaluated = evaluation
        .evaluate(token_evaluation_batch([0.0; 6]))
        .unwrap();
    assert_eq!(evaluated.loss_weight(), 0);
    assert_eq!(evaluated.loss().scalar_at(0).as_f64(), 0.0);
    assert_eq!(evaluation.checkpoint().unwrap(), evaluation_before);
}

#[test]
fn token_weighted_partial_flush_and_zero_grad_reset_the_count() {
    let flush_plan = compile_token_weighted_plan(3);
    let mut flushed = flush_plan.prepare_cpu().unwrap();
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut native_flushed = target.prepare(&flush_plan).unwrap();
    let mut complete = compile_token_weighted_plan(2).prepare_cpu().unwrap();
    let batches = [
        token_weighted_batch([1.0, 1.0, 100.0], [1.0, 1.0, 0.0]),
        token_weighted_batch([3.0, 100.0, 100.0], [1.0, 0.0, 0.0]),
    ];
    for batch in &batches {
        flushed
            .step(batch.clone(), TensorData::scalar(0.1))
            .unwrap();
        let native = native_flushed
            .step(batch.clone(), TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(native.report().fallback_count(), 0);
        complete
            .step(batch.clone(), TensorData::scalar(0.1))
            .unwrap();
    }
    assert_eq!(
        flushed
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_token_count(),
        Some(3)
    );
    let native_checkpoint = native_flushed.checkpoint().unwrap();
    let native_restored_plan = flush_plan.restore_checkpoint(&native_checkpoint).unwrap();
    let mut native_restored = target.prepare(&native_restored_plan).unwrap();
    flushed
        .flush_partial_window(TensorData::scalar(0.1))
        .unwrap();
    let native_flush = native_flushed
        .flush_partial_window(TensorData::scalar(0.1))
        .unwrap();
    assert_eq!(
        native_flush
            .report()
            .expect("a nonempty partial window runs")
            .fallback_count(),
        0
    );
    native_restored
        .flush_partial_window(TensorData::scalar(0.1))
        .unwrap();
    assert_eq!(
        flushed.parameter_snapshots().unwrap(),
        complete.parameter_snapshots().unwrap()
    );
    assert_eq!(
        flushed.first_moment_snapshots().unwrap(),
        complete.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        flushed
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_token_count(),
        Some(0)
    );
    assert_eq!(
        native_restored.checkpoint().unwrap(),
        native_flushed.checkpoint().unwrap()
    );
    assert_native_adamw_state_close(&native_flushed, &flushed);

    flushed
        .step(batches[0].clone(), TensorData::scalar(0.1))
        .unwrap();
    native_flushed
        .step(batches[0].clone(), TensorData::scalar(0.1))
        .unwrap();
    assert_eq!(
        flushed
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_token_count(),
        Some(2)
    );
    flushed.zero_grad().unwrap();
    native_flushed.zero_grad().unwrap();
    assert_eq!(flushed.accumulation_index().unwrap(), 0);
    let reset_checkpoint = flushed.checkpoint().unwrap();
    assert_eq!(reset_checkpoint.info().accumulated_token_count(), Some(0));
    assert_eq!(reset_checkpoint.info().reset_transition_count(), 1);
    assert_eq!(
        native_flushed
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_token_count(),
        Some(0)
    );
    assert_native_adamw_state_close(&native_flushed, &flushed);
    let (_, metadata) = load_safetensors(reset_checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V9);
    assert_eq!(metadata["token_weighted_accumulation_present"], "true");
}

#[test]
fn adamw_accumulation_clips_once_after_averaging_the_complete_window() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_loss_scale(128.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_input("scale", [], DType::F32)
        .unwrap();
    let parameter = TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap();
    let mut compiled =
        CpuCompiledAdamW::compile(config, [parameter], |graph, inputs, parameters| {
            Ok((
                graph.mul(parameters["weight"], inputs["scale"])?,
                BTreeMap::new(),
            ))
        })
        .unwrap();

    let input = |scale| BTreeMap::from([("scale".into(), TensorData::scalar(scale))]);
    let first = compiled
        .step(input(100.0), TensorData::scalar(0.1))
        .unwrap();
    assert!(!first.did_update());
    assert_eq!(
        compiled.gradient_accumulator_snapshots().unwrap()["weight"]
            .scalar_at(0)
            .as_f64(),
        100.0
    );

    let second = compiled
        .step(input(-99.0), TensorData::scalar(0.1))
        .unwrap();
    assert!(second.did_update());
    let first_moment = compiled.first_moment_snapshots().unwrap()["weight"]
        .scalar_at(0)
        .as_f64();
    assert!(
        (first_moment - 0.5).abs() < 1e-6,
        "window-average gradient was {first_moment}"
    );
    assert_eq!(
        compiled.gradient_accumulator_snapshots().unwrap()["weight"]
            .scalar_at(0)
            .as_f64(),
        0.0
    );
}

#[test]
fn adamw_clip_report_marks_only_full_or_flushed_windows_on_both_cpu_paths() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_clip_report()
        .with_input("scale", [], DType::F32)
        .unwrap();
    let compile = || {
        CompiledAdamWPlan::compile(
            config.clone(),
            [TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap()],
            |graph, inputs, parameters| {
                Ok((
                    graph.mul(parameters["weight"], inputs["scale"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap()
    };
    let input = |scale| BTreeMap::from([("scale".into(), TensorData::scalar(scale))]);
    let plan = compile();
    let mut interpreted = plan.prepare_cpu().unwrap();
    let executor = CapturedReplayExecutor::default();
    let mut native = plan
        .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
        .unwrap();

    assert!(
        interpreted
            .step(input(3.0), TensorData::scalar(0.1))
            .unwrap()
            .clip_report()
            .is_none()
    );
    assert!(
        native
            .step(input(3.0), TensorData::scalar(0.1))
            .unwrap()
            .clip_report()
            .is_none()
    );
    let interpreted_step = interpreted
        .step(input(5.0), TensorData::scalar(0.1))
        .unwrap();
    let native_step = native.step(input(5.0), TensorData::scalar(0.1)).unwrap();
    let expected = interpreted_step.clip_report().unwrap();
    assert_eq!(expected.pre_clip_global_norm(), 4.0);
    assert_eq!(expected.applied_scale(), 0.25);
    assert_eq!(native_step.clip_report(), Some(expected));
    assert_eq!(
        native.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );

    let mut interpreted_flush = compile().prepare_cpu().unwrap();
    let flush_plan = compile();
    let mut native_flush = flush_plan
        .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
        .unwrap();
    assert!(
        interpreted_flush
            .step(input(4.0), TensorData::scalar(0.1))
            .unwrap()
            .clip_report()
            .is_none()
    );
    assert!(
        native_flush
            .step(input(4.0), TensorData::scalar(0.1))
            .unwrap()
            .clip_report()
            .is_none()
    );
    let interpreted_flush_result = interpreted_flush
        .flush_partial_window(TensorData::scalar(0.1))
        .unwrap();
    let native_flush_result = native_flush
        .flush_partial_window(TensorData::scalar(0.1))
        .unwrap();
    let expected = interpreted_flush_result.clip_report().unwrap();
    assert_eq!(expected.pre_clip_global_norm(), 4.0);
    assert_eq!(expected.applied_scale(), 0.25);
    assert_eq!(native_flush_result.clip_report(), Some(expected));
    assert!(
        interpreted_flush
            .flush_partial_window(TensorData::scalar(0.1))
            .unwrap()
            .clip_report()
            .is_none()
    );
    assert!(
        native_flush
            .flush_partial_window(TensorData::scalar(0.1))
            .unwrap()
            .clip_report()
            .is_none()
    );
    assert_eq!(
        native_flush.checkpoint().unwrap(),
        interpreted_flush.checkpoint().unwrap()
    );
}

#[test]
fn adamw_window_loss_reports_full_and_flushed_windows_on_both_cpu_paths() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_window_loss_report()
        .with_input("scale", [], DType::F32)
        .unwrap();
    let plan = CompiledAdamWPlan::compile(
        config,
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, inputs, parameters| {
            Ok((
                graph.mul(parameters["weight"], inputs["scale"])?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap();
    assert!(plan.window_loss_report_enabled());
    let input = |scale| BTreeMap::from([("scale".into(), TensorData::scalar(scale))]);
    let executor = CapturedReplayExecutor::default();
    let mut interpreted = plan.prepare_cpu().unwrap();
    let mut native = plan
        .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
        .unwrap();

    assert!(
        interpreted
            .step(input(2.0), TensorData::scalar(0.0))
            .unwrap()
            .window_loss_report()
            .is_none()
    );
    assert!(
        native
            .step(input(2.0), TensorData::scalar(0.0))
            .unwrap()
            .window_loss_report()
            .is_none()
    );
    let interpreted_step = interpreted
        .step(input(4.0), TensorData::scalar(0.0))
        .unwrap();
    let native_step = native.step(input(4.0), TensorData::scalar(0.0)).unwrap();
    let report = interpreted_step.window_loss_report().unwrap();
    assert_eq!(report.mean_loss(), 3.0);
    assert_eq!(report.loss_weight(), 2);
    assert_eq!(report.microbatch_count(), 2);
    assert_eq!(native_step.window_loss_report(), Some(report));
    let checkpoint = interpreted.checkpoint().unwrap();
    assert_eq!(native.checkpoint().unwrap(), checkpoint);
    let (mut checkpoint_tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V9);
    assert_eq!(metadata["window_loss_report_enabled"], "true");
    assert!(
        checkpoint_tensors
            .remove("accumulated_loss_numerator")
            .is_some()
    );
    assert!(
        CompiledAdamWCheckpoint::from_bytes(
            save_safetensors(&checkpoint_tensors, &metadata).unwrap()
        )
        .is_err()
    );
    assert_eq!(checkpoint.info().accumulated_loss_numerator(), Some(0.0));
    let without_report = CompiledAdamWPlan::compile(
        CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_input("scale", [], DType::F32)
            .unwrap(),
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, inputs, parameters| {
            Ok((
                graph.mul(parameters["weight"], inputs["scale"])?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap();
    assert!(without_report.restore_checkpoint(&checkpoint).is_err());

    let mut flushed = plan.prepare_cpu().unwrap();
    let partial = flushed.step(input(6.0), TensorData::scalar(0.0)).unwrap();
    assert!(partial.window_loss_report().is_none());
    assert_eq!(
        flushed
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_loss_numerator(),
        Some(6.0)
    );
    assert_eq!(flushed.zero_grad().unwrap().discarded_microbatches(), 1);
    assert_eq!(
        flushed
            .checkpoint()
            .unwrap()
            .info()
            .accumulated_loss_numerator(),
        Some(0.0)
    );
    flushed.step(input(8.0), TensorData::scalar(0.0)).unwrap();
    let flush = flushed
        .flush_partial_window(TensorData::scalar(0.0))
        .unwrap();
    let report = flush.window_loss_report().unwrap();
    assert_eq!(report.mean_loss(), 8.0);
    assert_eq!(report.loss_weight(), 1);
    assert_eq!(report.microbatch_count(), 1);
    assert!(
        flushed
            .flush_partial_window(TensorData::scalar(0.0))
            .unwrap()
            .window_loss_report()
            .is_none()
    );
}

#[test]
fn cpu_commit_only_steps_preserve_adamw_state_and_skip_named_egress() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_clip_report()
        .with_window_loss_report()
        .with_input("scale", [], DType::F32)
        .unwrap()
        .with_input("features", [4], DType::F32)
        .unwrap();
    let plan = CompiledAdamWPlan::compile(
        config,
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, inputs, parameters| {
            let loss = graph.mul(parameters["weight"], inputs["scale"])?;
            let direct = graph.square(inputs["features"])?;
            let shrunk = graph.shrink(direct, [(1, 3)])?;
            let empty = graph.constant(TensorData::new([0], Vec::<f32>::new())?);
            Ok((
                loss,
                BTreeMap::from([
                    ("direct".into(), direct),
                    ("empty".into(), empty),
                    ("shrunk".into(), shrunk),
                ]),
            ))
        },
    )
    .unwrap();
    assert_eq!(
        plan.inner.phase_outputs.named_outputs,
        vec!["direct".to_owned(), "empty".to_owned(), "shrunk".to_owned()]
    );
    assert!(
        plan.inner
            .capture
            .schedule
            .requested_passthroughs
            .is_empty()
    );
    let input = |scale| {
        BTreeMap::from([
            (
                "features".into(),
                TensorData::new([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            ),
            ("scale".into(), TensorData::scalar(scale)),
        ])
    };
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut interpreted = plan.prepare_cpu().unwrap();
    let mut interpreted_commit = plan.prepare_cpu().unwrap();
    let mut native = target.prepare(&plan).unwrap();
    let mut native_commit = target.prepare(&plan).unwrap();
    assert_eq!(native.main_replay.zero_domain_item_count(), 1);
    assert_eq!(native_commit.main_replay.zero_domain_item_count(), 1);
    assert_eq!(
        native
            .accumulation_replay
            .as_ref()
            .unwrap()
            .zero_domain_item_count(),
        1
    );
    assert_eq!(
        native_commit
            .accumulation_replay
            .as_ref()
            .unwrap()
            .zero_domain_item_count(),
        1
    );

    for scale in [2.0, 4.0] {
        let expected = interpreted
            .step(input(scale), TensorData::scalar(0.01))
            .unwrap();
        let actual = interpreted_commit
            .step_commit_only(input(scale), TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert!(actual.outputs().is_empty());
        assert_eq!(expected.outputs().len(), 3);
        assert_eq!(expected.output("direct").unwrap().len(), 4);
        assert_eq!(expected.output("empty").unwrap().len(), 0);
        assert_eq!(expected.output("shrunk").unwrap().len(), 2);
        assert_eq!(
            expected.output("direct").unwrap().values(),
            [1.0, 4.0, 9.0, 16.0]
        );
        assert_eq!(expected.output("shrunk").unwrap().values(), [4.0, 9.0]);
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.clip_report(), expected.clip_report());
        assert_eq!(actual.window_loss_report(), expected.window_loss_report());
        assert_eq!(
            interpreted_commit.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );

        let expected = native.step(input(scale), TensorData::scalar(0.01)).unwrap();
        let actual = native_commit
            .step_commit_only(input(scale), TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert!(actual.outputs().is_empty());
        assert_eq!(expected.outputs().len(), 3);
        assert_eq!(expected.output("direct").unwrap().len(), 4);
        assert_eq!(expected.output("empty").unwrap().len(), 0);
        assert_eq!(expected.output("shrunk").unwrap().len(), 2);
        assert_eq!(
            expected.output("direct").unwrap().values(),
            [1.0, 4.0, 9.0, 16.0]
        );
        assert_eq!(expected.output("shrunk").unwrap().values(), [4.0, 9.0]);
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.clip_report(), expected.clip_report());
        assert_eq!(actual.window_loss_report(), expected.window_loss_report());
        let required_egress_count = if actual.did_update() { 5 } else { 1 };
        let required_egress_bytes = if actual.did_update() { 24 } else { 4 };
        assert_eq!(
            actual.report().traffic().materialized_egress_count(),
            required_egress_count
        );
        assert_eq!(
            actual.report().traffic().materialized_egress_bytes(),
            required_egress_bytes
        );
        assert_eq!(
            expected.report().traffic().materialized_egress_count(),
            required_egress_count + 3
        );
        assert_eq!(
            expected.report().traffic().materialized_egress_bytes(),
            required_egress_bytes + 24
        );
        assert_eq!(
            native_commit.checkpoint().unwrap(),
            native.checkpoint().unwrap()
        );
    }
    assert_eq!(
        interpreted_commit.parameter_snapshots().unwrap(),
        interpreted.parameter_snapshots().unwrap()
    );
    assert_eq!(
        interpreted_commit.first_moment_snapshots().unwrap(),
        interpreted.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        interpreted_commit.second_moment_snapshots().unwrap(),
        interpreted.second_moment_snapshots().unwrap()
    );
    assert_eq!(
        interpreted_commit.gradient_accumulator_snapshots().unwrap(),
        interpreted.gradient_accumulator_snapshots().unwrap()
    );
    assert_eq!(
        native_commit.parameter_snapshots().unwrap(),
        native.parameter_snapshots().unwrap()
    );
    assert_eq!(
        native_commit.first_moment_snapshots().unwrap(),
        native.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        native_commit.second_moment_snapshots().unwrap(),
        native.second_moment_snapshots().unwrap()
    );
    assert_eq!(
        native_commit.gradient_accumulator_snapshots().unwrap(),
        native.gradient_accumulator_snapshots().unwrap()
    );

    let ordinary_accumulation = native
        .accumulation_replay
        .as_ref()
        .unwrap()
        .workspace_stats();
    let commit_accumulation = native_commit
        .accumulation_replay
        .as_ref()
        .unwrap()
        .workspace_stats();
    assert_eq!(
        ordinary_accumulation.last_materialized_egress_count,
        commit_accumulation.last_materialized_egress_count + 3
    );
    assert_eq!(
        ordinary_accumulation.last_materialized_egress_bytes,
        commit_accumulation.last_materialized_egress_bytes + 24
    );
    assert_eq!(commit_accumulation.last_materialized_egress_count, 1);
    assert_eq!(commit_accumulation.last_materialized_egress_bytes, 4);
    let ordinary_main = native.main_replay.workspace_stats();
    let commit_main = native_commit.main_replay.workspace_stats();
    assert_eq!(
        ordinary_main.last_materialized_egress_count,
        commit_main.last_materialized_egress_count + 3
    );
    assert_eq!(
        ordinary_main.last_materialized_egress_bytes,
        commit_main.last_materialized_egress_bytes + 24
    );
    assert_eq!(commit_main.last_materialized_egress_count, 5);
    assert_eq!(commit_main.last_materialized_egress_bytes, 24);

    let mut injected = plan.prepare_cpu().unwrap();
    let mut retry_reference = plan.prepare_cpu().unwrap();
    let before = injected.checkpoint().unwrap();
    assert!(
        injected
            .step_commit_only_inner(input(2.0), TensorData::scalar(0.01), Some(0))
            .is_err()
    );
    assert_eq!(injected.checkpoint().unwrap(), before);
    let expected = retry_reference
        .step(input(2.0), TensorData::scalar(0.01))
        .unwrap();
    let actual = injected
        .step_commit_only(input(2.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert!(actual.outputs().is_empty());
    assert_eq!(
        injected.checkpoint().unwrap(),
        retry_reference.checkpoint().unwrap()
    );

    let mut injected = target.prepare(&plan).unwrap();
    let mut retry_reference = target.prepare(&plan).unwrap();
    let before = injected.checkpoint().unwrap();
    assert!(
        injected
            .step_commit_only_inner(input(2.0), TensorData::scalar(0.01), Some(0))
            .is_err()
    );
    assert_eq!(injected.checkpoint().unwrap(), before);
    assert_eq!(injected.successful_steps, 0);
    let expected = retry_reference
        .step(input(2.0), TensorData::scalar(0.01))
        .unwrap();
    let actual = injected
        .step_commit_only(input(2.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert!(actual.outputs().is_empty());
    assert_eq!(actual.report().successful_invocation(), 1);
    assert_eq!(actual.report().traffic().materialized_egress_count(), 1);
    assert_eq!(actual.report().traffic().materialized_egress_bytes(), 4);
    assert_eq!(
        injected.checkpoint().unwrap(),
        retry_reference.checkpoint().unwrap()
    );

    native.step(input(6.0), TensorData::scalar(0.01)).unwrap();
    native_commit
        .step_commit_only(input(6.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        native.zero_grad().unwrap(),
        native_commit.zero_grad().unwrap()
    );
    assert_eq!(
        native_commit.checkpoint().unwrap(),
        native.checkpoint().unwrap()
    );
    native.step(input(8.0), TensorData::scalar(0.01)).unwrap();
    native_commit
        .step_commit_only(input(8.0), TensorData::scalar(0.01))
        .unwrap();
    let expected = native
        .flush_partial_window(TensorData::scalar(0.01))
        .unwrap();
    let actual = native_commit
        .flush_partial_window(TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        actual.flushed_microbatches(),
        expected.flushed_microbatches()
    );
    assert_eq!(actual.optimizer_step(), expected.optimizer_step());
    assert_eq!(actual.clip_report(), expected.clip_report());
    assert_eq!(actual.window_loss_report(), expected.window_loss_report());
    assert_eq!(
        native_commit.checkpoint().unwrap(),
        native.checkpoint().unwrap()
    );

    let expected = interpreted
        .step(input(6.0), TensorData::scalar(0.01))
        .unwrap();
    let actual = interpreted_commit
        .step_commit_only(input(6.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(
        interpreted_commit.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );
    assert_eq!(
        interpreted.zero_grad().unwrap(),
        interpreted_commit.zero_grad().unwrap()
    );
    assert_eq!(
        interpreted_commit.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );
    interpreted
        .step(input(8.0), TensorData::scalar(0.01))
        .unwrap();
    interpreted_commit
        .step_commit_only(input(8.0), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(
        interpreted
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap(),
        interpreted_commit
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap()
    );
    assert_eq!(
        interpreted_commit.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );

    let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([79, 83]));
    let module = TiedFrozenModule::new([0.1, -0.2]);
    let dropout_plan = CompiledAdamWPlan::compile_module_with_dropout(
        module_config().with_gradient_accumulation(2).unwrap(),
        dropout,
        &module,
        build_tied_dropout_with_input_guard,
    )
    .unwrap();
    let rejecting = rejecting_cpu_target();
    let mut dropout_reference = dropout_plan.prepare(&rejecting).unwrap();
    let mut dropout_commit = dropout_plan.prepare(&rejecting).unwrap();
    let finite = || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
    let invalid = BTreeMap::from([("x".into(), TensorData::new([2], vec![0.0, 2.0]).unwrap())]);
    let before = dropout_commit.checkpoint().unwrap();
    assert!(
        dropout_commit
            .step_commit_only(invalid, TensorData::scalar(0.01))
            .is_err()
    );
    assert_eq!(dropout_commit.checkpoint().unwrap(), before);
    assert_eq!(dropout_commit.dropout_block_counter().unwrap(), Some(0));
    let expected = dropout_reference
        .step(finite(), TensorData::scalar(0.01))
        .unwrap();
    let actual = dropout_commit
        .step_commit_only(finite(), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert!(actual.outputs().is_empty());
    assert_eq!(dropout_commit.dropout_block_counter().unwrap(), Some(1));
    assert_eq!(
        dropout_commit.checkpoint().unwrap(),
        dropout_reference.checkpoint().unwrap()
    );

    let native_target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut native_dropout_reference = native_target.prepare(&dropout_plan).unwrap();
    let mut native_dropout_commit = native_target.prepare(&dropout_plan).unwrap();
    let invalid = BTreeMap::from([("x".into(), TensorData::new([2], vec![0.0, 2.0]).unwrap())]);
    let before = native_dropout_commit.checkpoint().unwrap();
    assert!(
        native_dropout_commit
            .step_commit_only(invalid, TensorData::scalar(0.01))
            .is_err()
    );
    assert_eq!(native_dropout_commit.checkpoint().unwrap(), before);
    assert_eq!(native_dropout_commit.successful_steps, 0);
    assert_eq!(
        native_dropout_commit.dropout_block_counter().unwrap(),
        Some(0)
    );
    let expected = native_dropout_reference
        .step(finite(), TensorData::scalar(0.01))
        .unwrap();
    let actual = native_dropout_commit
        .step_commit_only(finite(), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert!(actual.outputs().is_empty());
    assert_eq!(actual.report().successful_invocation(), 1);
    assert_eq!(
        native_dropout_commit.dropout_block_counter().unwrap(),
        Some(1)
    );
    assert_eq!(
        native_dropout_commit.checkpoint().unwrap(),
        native_dropout_reference.checkpoint().unwrap()
    );
}

#[test]
fn non_finite_completed_window_loss_rejects_atomically_and_retries() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_window_loss_report()
        .with_input("offset", [], DType::F32)
        .unwrap();
    let plan = CompiledAdamWPlan::compile(
        config,
        [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
        |graph, inputs, parameters| {
            Ok((
                graph.add(parameters["weight"], inputs["offset"])?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap();
    let input = |offset| BTreeMap::from([("offset".into(), TensorData::scalar(offset))]);
    let target = rejecting_cpu_target();
    let executor = CapturedReplayExecutor::default();
    let native_target = NativeCpuSessionTarget::new(&executor)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut interpreted = plan.prepare(&target).unwrap();
    let mut native = plan.prepare(&native_target).unwrap();

    assert!(
        interpreted
            .step(input(f32::MAX), TensorData::scalar(0.0))
            .unwrap()
            .window_loss_report()
            .is_none()
    );
    assert!(
        native
            .step(input(f32::MAX), TensorData::scalar(0.0))
            .unwrap()
            .window_loss_report()
            .is_none()
    );
    let interpreted_before = interpreted.checkpoint().unwrap();
    let native_before = native.checkpoint().unwrap();
    assert!(
        interpreted
            .step(input(f32::MAX), TensorData::scalar(0.0))
            .is_err()
    );
    assert!(
        native
            .step(input(f32::MAX), TensorData::scalar(0.0))
            .is_err()
    );
    assert_eq!(interpreted.checkpoint().unwrap(), interpreted_before);
    assert_eq!(native.checkpoint().unwrap(), native_before);

    let interpreted_retry = interpreted
        .step(input(-f32::MAX), TensorData::scalar(0.0))
        .unwrap();
    let native_retry = native
        .step(input(-f32::MAX), TensorData::scalar(0.0))
        .unwrap();
    let report = interpreted_retry.window_loss_report().unwrap();
    assert_eq!(report.mean_loss().to_bits(), 0.0_f32.to_bits());
    assert_eq!(report.loss_weight(), 2);
    assert_eq!(native_retry.window_loss_report(), Some(report));
    assert_eq!(
        native.checkpoint().unwrap(),
        interpreted.checkpoint().unwrap()
    );
}

#[test]
fn adamw_accumulates_recurrent_gradients_and_commits_only_at_window_end() {
    let mut compiled = CpuCompiledAdamW::compile(
        accumulated_adamw_config(2),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    let identity = compiled.capture_identity();
    let initial_parameters = compiled.parameter_snapshots().unwrap();
    let initial_first = compiled.first_moment_snapshots().unwrap();
    let initial_second = compiled.second_moment_snapshots().unwrap();

    let partial = compiled.step(batch(), lr()).unwrap();
    assert_eq!(partial.step(), 1);
    assert_eq!(partial.optimizer_step(), 0);
    assert_eq!(partial.accumulation_index(), 1);
    assert!(!partial.did_update());
    assert_eq!(partial.capture_identity(), identity);
    assert_eq!(compiled.optimizer_step().unwrap(), 0);
    assert_eq!(compiled.accumulation_index().unwrap(), 1);
    assert_eq!(compiled.parameter_snapshots().unwrap(), initial_parameters);
    assert_eq!(compiled.first_moment_snapshots().unwrap(), initial_first);
    assert_eq!(compiled.second_moment_snapshots().unwrap(), initial_second);
    assert!(
        compiled
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .any(|value| value
                != &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32,).unwrap())
    );

    let committed = compiled.step(batch(), lr()).unwrap();
    assert_eq!(committed.step(), 2);
    assert_eq!(committed.optimizer_step(), 1);
    assert_eq!(committed.accumulation_index(), 0);
    assert!(committed.did_update());
    assert_eq!(compiled.optimizer_step().unwrap(), 1);
    assert_eq!(compiled.accumulation_index().unwrap(), 0);
    assert_ne!(compiled.parameter_snapshots().unwrap(), initial_parameters);
    assert_ne!(compiled.first_moment_snapshots().unwrap(), initial_first);
    assert_ne!(compiled.second_moment_snapshots().unwrap(), initial_second);
    assert!(
        compiled
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .all(|value| value
                == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32,).unwrap())
    );
}

#[test]
fn adamw_partial_accumulation_checkpoint_resumes_exactly() {
    let config = accumulated_adamw_config(3);
    let mut uninterrupted =
        CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
    uninterrupted.step(batch(), lr()).unwrap();
    let checkpoint = uninterrupted.checkpoint().unwrap();
    let (state, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V9);
    assert_eq!(metadata["replay_step"], "1");
    assert_eq!(metadata["optimizer_step"], "0");
    assert_eq!(metadata["gradient_accumulation_steps"], "3");
    assert_eq!(metadata["accumulation_index"], "1");
    assert_eq!(
        checkpoint.info().accumulation_capture_identity(),
        uninterrupted
            .inner
            .accumulation
            .as_ref()
            .map(|transition| transition.phase().capture_identity)
    );
    let mut malformed_metadata = metadata;
    let accumulation_identity = checkpoint.info().accumulation_capture_identity().unwrap();
    malformed_metadata.insert(
        "accumulation_capture_identity".into(),
        (accumulation_identity ^ 1).to_string(),
    );
    let malformed =
        CompiledAdamWCheckpoint::from_bytes(save_safetensors(&state, &malformed_metadata).unwrap())
            .unwrap();
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(config.clone(), &malformed, build_tinybob)
            .is_err()
    );

    let mut resumed =
        CpuCompiledAdamW::compile_from_checkpoint(config, &checkpoint, build_tinybob).unwrap();
    assert_eq!(resumed.step_count(), 1);
    assert_eq!(resumed.optimizer_step().unwrap(), 0);
    assert_eq!(resumed.accumulation_index().unwrap(), 1);
    assert_eq!(
        resumed.gradient_accumulator_snapshots().unwrap(),
        uninterrupted.gradient_accumulator_snapshots().unwrap()
    );
    assert_eq!(
        resumed.parameter_snapshots().unwrap(),
        uninterrupted.parameter_snapshots().unwrap()
    );

    for expected_step in 2..=4 {
        let expected = uninterrupted.step(batch(), lr()).unwrap();
        let actual = resumed.step(batch(), lr()).unwrap();
        assert_eq!(actual.step(), expected_step);
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.did_update(), expected.did_update());
    }
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );
}

#[test]
fn adamw_plan_restore_rejects_authenticated_state_schema_mismatch_atomically() {
    let config = accumulated_adamw_config(3);
    let plan = CompiledAdamWPlan::compile(config, initial_parameters(), build_tinybob).unwrap();
    let initial = plan.prepare_cpu().unwrap().checkpoint().unwrap();
    let mut runtime = plan.prepare_cpu().unwrap();
    runtime.step(batch(), lr()).unwrap();
    let decoded = decode_adamw_checkpoint(runtime.checkpoint().unwrap().as_bytes()).unwrap();
    let progress = AdamWCheckpointProgress {
        capture_identity: decoded.capture_identity,
        accumulation_capture_identity: decoded.accumulation_capture_identity,
        replay_step: decoded.replay_step,
        optimizer_step: decoded.optimizer_step,
        accumulation_steps: decoded.accumulation_steps,
        accumulation_index: decoded.accumulation_index,
        discarded_microbatches: decoded.discarded_microbatches,
        flushed_window_count: decoded.flushed_window_count,
        flushed_microbatch_count: decoded.flushed_microbatch_count,
        flush_capture_identity: decoded.flush_capture_identity,
        dropout_block_counter: decoded.dropout_block_counter,
        accumulated_token_count: decoded.accumulated_token_count,
        window_loss_report: decoded.window_loss_report,
        reset_transition_count: decoded.reset_transition_count,
        reset_capture_identity: decoded.reset_capture_identity,
    };
    let mut tensors = AdamWCheckpointTensors {
        parameters: decoded.parameters,
        first_moments: decoded.first_moments,
        second_moments: decoded.second_moments,
        gradient_accumulators: decoded.gradient_accumulators,
        accumulated_loss_numerator: decoded.accumulated_loss_numerator,
    };
    for values in [
        &mut tensors.parameters,
        &mut tensors.first_moments,
        &mut tensors.second_moments,
        &mut tensors.gradient_accumulators,
    ] {
        values.insert(
            "w1".into(),
            TensorData::zeros_with_dtype([4, 2], DType::F32).unwrap(),
        );
    }
    let malformed =
        CompiledAdamWCheckpoint::from_bytes(encode_adamw_checkpoint(progress, tensors).unwrap())
            .unwrap();

    let error = match plan.restore_checkpoint(&malformed) {
        Ok(_) => panic!("an equal-byte state descriptor mismatch restored"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("state descriptor mismatch"));
    assert_eq!(plan.prepare_cpu().unwrap().checkpoint().unwrap(), initial);

    let before = runtime.checkpoint().unwrap();
    let error = runtime.restore_checkpoint_in_place(&malformed).unwrap_err();
    assert!(error.to_string().contains("state descriptor mismatch"));
    assert_eq!(runtime.checkpoint().unwrap(), before);
    runtime.restore_checkpoint_in_place(&before).unwrap();
    assert_eq!(runtime.checkpoint().unwrap(), before);
}

#[test]
fn adamw_partial_flush_matches_a_complete_short_window_and_resumes_exactly() {
    let config = accumulated_adamw_config(3);
    let mut flushed =
        CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
    let mut short = CpuCompiledAdamW::compile(
        accumulated_adamw_config(2),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    assert!(flushed.flush_capture_identity().is_some());
    let transition = flushed.partial_flush.as_ref().unwrap();
    assert_eq!(
        flushed.flush_capture_identity(),
        Some(
            transition
                .phase()
                .capture
                .initial_recurrent_cursor()
                .unwrap()
                .capture_identity()
        )
    );
    assert_ne!(
        flushed.flush_capture_identity(),
        Some(flushed.capture_identity())
    );
    for runtime in [&mut flushed, &mut short] {
        runtime.step(batch(), lr()).unwrap();
        runtime.step(batch(), lr()).unwrap();
    }
    let dropout_before = flushed.dropout_block_counter().unwrap();
    let result = flushed.flush_partial_window(lr()).unwrap();
    assert!(result.did_update());
    assert_eq!(result.flushed_microbatches(), 2);
    assert_eq!(result.optimizer_step(), 1);
    assert_eq!(flushed.step_count(), 2);
    assert_eq!(flushed.optimizer_step().unwrap(), 1);
    assert_eq!(flushed.accumulation_index().unwrap(), 0);
    assert_eq!(flushed.dropout_block_counter().unwrap(), dropout_before);
    assert_eq!(
        flushed.parameter_snapshots().unwrap(),
        short.parameter_snapshots().unwrap()
    );
    assert_eq!(
        flushed.first_moment_snapshots().unwrap(),
        short.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        flushed.second_moment_snapshots().unwrap(),
        short.second_moment_snapshots().unwrap()
    );
    assert!(
        flushed
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .all(|value| value
                == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32).unwrap())
    );

    let checkpoint = flushed.checkpoint().unwrap();
    let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V9);
    assert_eq!(metadata["replay_step"], "2");
    assert_eq!(metadata["optimizer_step"], "1");
    assert_eq!(metadata["flushed_window_count"], "1");
    assert_eq!(metadata["flushed_microbatch_count"], "2");
    assert_eq!(metadata["dropout_state_present"], "false");
    assert_eq!(
        metadata["flush_capture_identity"],
        flushed.flush_capture_identity().unwrap().to_string()
    );
    let info = checkpoint.info();
    assert_eq!(info.capture_identity(), flushed.capture_identity());
    assert_eq!(info.replay_step(), 2);
    assert_eq!(info.optimizer_step(), 1);
    assert_eq!(info.gradient_accumulation_steps(), 3);
    assert_eq!(info.accumulation_index(), 0);
    assert_eq!(info.discarded_microbatches(), 0);
    assert_eq!(info.flushed_window_count(), 1);
    assert_eq!(info.flushed_microbatch_count(), 2);
    assert_eq!(
        info.flush_capture_identity(),
        flushed.flush_capture_identity()
    );
    assert_eq!(info.dropout_block_counter(), None);
    assert_eq!(info.accumulated_token_count(), None);
    assert!(
        flushed
            .parameter_versions()
            .unwrap()
            .values()
            .all(|version| *version == 3)
    );
    assert!(
        flushed
            .inner
            .plan()
            .unwrap()
            .state_versions
            .values()
            .all(|version| *version == 3),
        "parameters, moments, accumulators, and AdamW globals advance once"
    );
    let (state, mut mismatched_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    let mut impossible_progress = mismatched_metadata.clone();
    impossible_progress.insert("replay_step".into(), "5".into());
    impossible_progress.insert("optimizer_step".into(), "2".into());
    impossible_progress.insert("flushed_window_count".into(), "2".into());
    impossible_progress.insert("flushed_microbatch_count".into(), "5".into());
    assert!(
        CompiledAdamWCheckpoint::from_bytes(
            save_safetensors(&state, &impossible_progress).unwrap()
        )
        .is_err(),
        "two partial N=3 windows can retain at most four microbatches"
    );
    mismatched_metadata.insert(
        "flush_capture_identity".into(),
        flushed
            .flush_capture_identity()
            .unwrap()
            .wrapping_add(1)
            .to_string(),
    );
    let mismatched = CompiledAdamWCheckpoint::from_bytes(
        save_safetensors(&state, &mismatched_metadata).unwrap(),
    )
    .unwrap();
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(config.clone(), &mismatched, build_tinybob,)
            .is_err()
    );
    let mut resumed =
        CpuCompiledAdamW::compile_from_checkpoint(config, &checkpoint, build_tinybob).unwrap();
    assert_eq!(
        resumed.parameter_versions().unwrap(),
        flushed.parameter_versions().unwrap()
    );
    assert_eq!(
        resumed.inner.plan().unwrap().state_versions,
        flushed.inner.plan().unwrap().state_versions
    );
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    for _ in 0..3 {
        let expected = flushed.step(batch(), lr()).unwrap();
        let actual = resumed.step(batch(), lr()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
    }
    assert_eq!(resumed.checkpoint().unwrap(), flushed.checkpoint().unwrap());
}

#[test]
fn adamw_partial_flush_empty_validation_and_commit_failure_are_atomic() {
    let mut runtime = CpuCompiledAdamW::compile(
        accumulated_adamw_config(3),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    let initial = runtime.checkpoint().unwrap();
    let noop = runtime.flush_partial_window(lr()).unwrap();
    assert!(!noop.did_update());
    assert_eq!(noop.flushed_microbatches(), 0);
    assert_eq!(runtime.checkpoint().unwrap(), initial);
    assert!(
        runtime
            .flush_partial_window(TensorData::new([1], vec![0.1]).unwrap())
            .is_err()
    );
    assert_eq!(runtime.checkpoint().unwrap(), initial);

    runtime.step(batch(), lr()).unwrap();
    let partial = runtime.checkpoint().unwrap();
    assert!(runtime.flush_partial_window_inner(lr(), Some(0)).is_err());
    assert_eq!(runtime.checkpoint().unwrap(), partial);
    assert_eq!(runtime.accumulation_index().unwrap(), 1);
    assert_eq!(runtime.optimizer_step().unwrap(), 0);
    assert!(runtime.flush_partial_window(lr()).unwrap().did_update());
}

#[test]
fn adamw_partial_flush_preserves_dropout_version_and_restores_split_frontier() {
    let config = module_config().with_gradient_accumulation(3).unwrap();
    let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([41, 43]));
    let module = TiedFrozenModule::new([0.1, -0.2]);
    let mut runtime = CompiledAdamWPlan::compile_module_with_dropout(
        config.clone(),
        dropout,
        &module,
        build_tied_dropout,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap();
    runtime
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    assert_eq!(runtime.dropout_block_counter().unwrap(), Some(1));
    assert!(
        runtime
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert_eq!(runtime.dropout_block_counter().unwrap(), Some(1));
    let versions = runtime.inner.plan().unwrap().state_versions;
    assert_eq!(versions[&RecurrentStateKey::dropout_counter()], 1);
    assert!(
        versions
            .iter()
            .filter(|(key, _)| *key != &RecurrentStateKey::dropout_counter())
            .all(|(_, version)| *version == 2)
    );

    let checkpoint = runtime.checkpoint().unwrap();
    let fresh = TiedFrozenModule::new([0.1, -0.2]);
    let resumed = CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
        config,
        dropout,
        &fresh,
        &checkpoint,
        build_tied_dropout,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap();
    assert_eq!(resumed.inner.plan().unwrap().state_versions, versions);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
}

#[test]
fn adamw_accumulation_failure_preserves_the_partial_frontier() {
    let mut compiled = CpuCompiledAdamW::compile(
        accumulated_adamw_config(2),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    compiled.step(batch(), lr()).unwrap();
    let cursor = compiled.inner.cursor.clone();
    let parameters = compiled.parameter_snapshots().unwrap();
    let first = compiled.first_moment_snapshots().unwrap();
    let second = compiled.second_moment_snapshots().unwrap();
    let accumulators = compiled.gradient_accumulator_snapshots().unwrap();

    assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
    assert_eq!(compiled.step_count(), 1);
    assert_eq!(compiled.optimizer_step().unwrap(), 0);
    assert_eq!(compiled.accumulation_index().unwrap(), 1);
    assert_eq!(compiled.inner.cursor, cursor);
    assert_eq!(compiled.parameter_snapshots().unwrap(), parameters);
    assert_eq!(compiled.first_moment_snapshots().unwrap(), first);
    assert_eq!(compiled.second_moment_snapshots().unwrap(), second);
    assert_eq!(
        compiled.gradient_accumulator_snapshots().unwrap(),
        accumulators
    );

    let cursor = compiled.inner.cursor.clone();
    let malformed = BTreeMap::from([(
        RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex),
        TensorData::scalar(0.0),
    )]);
    let step = compiled.step_count();
    assert!(
        compiled
            .inner
            .replace_state_values(step, malformed)
            .is_err()
    );
    assert_eq!(compiled.inner.cursor, cursor);
    assert_eq!(compiled.parameter_snapshots().unwrap(), parameters);
    assert_eq!(compiled.first_moment_snapshots().unwrap(), first);
    assert_eq!(compiled.second_moment_snapshots().unwrap(), second);
    assert_eq!(
        compiled.gradient_accumulator_snapshots().unwrap(),
        accumulators
    );
}

#[test]
fn adamw_zero_grad_discards_only_the_partial_window_and_resumes_exactly() {
    let config = accumulated_adamw_config(3);
    let mut cancelled =
        CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
    let mut clean =
        CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
    let initial_parameter_values = cancelled.parameter_snapshots().unwrap();
    let initial_first = cancelled.first_moment_snapshots().unwrap();
    let initial_second = cancelled.second_moment_snapshots().unwrap();

    cancelled.step(batch(), lr()).unwrap();
    let partial = cancelled.step(batch(), lr()).unwrap();
    assert_eq!(partial.step(), 2);
    assert_eq!(partial.optimizer_step(), 0);
    assert_eq!(partial.accumulation_index(), 2);
    let parameter_versions = cancelled.parameter_versions().unwrap();
    let first_versions = cancelled.first_moment_versions().unwrap();
    let second_versions = cancelled.second_moment_versions().unwrap();
    let state_versions_before_reset = cancelled.inner.plan().unwrap().state_versions;
    let before_failed_reset = cancelled.checkpoint().unwrap();
    assert!(cancelled.zero_grad_with_injected_failure(0).is_err());
    assert_eq!(cancelled.checkpoint().unwrap(), before_failed_reset);
    let reset = cancelled.zero_grad().unwrap();
    assert!(reset.did_discard());
    assert_eq!(reset.discarded_microbatches(), 2);
    assert_eq!(cancelled.step_count(), 2);
    assert_eq!(cancelled.optimizer_step().unwrap(), 0);
    assert_eq!(cancelled.accumulation_index().unwrap(), 0);
    assert_eq!(
        cancelled.parameter_snapshots().unwrap(),
        initial_parameter_values
    );
    assert_eq!(cancelled.first_moment_snapshots().unwrap(), initial_first);
    assert_eq!(cancelled.second_moment_snapshots().unwrap(), initial_second);
    assert_eq!(cancelled.parameter_versions().unwrap(), parameter_versions);
    assert_eq!(cancelled.first_moment_versions().unwrap(), first_versions);
    assert_eq!(cancelled.second_moment_versions().unwrap(), second_versions);
    let state_versions_after_reset = cancelled.inner.plan().unwrap().state_versions;
    for (key, before) in &state_versions_before_reset {
        let expected = if key.is_accumulation_reset_state() {
            before.checked_add(1).unwrap()
        } else {
            *before
        };
        assert_eq!(state_versions_after_reset[key], expected, "{key:?}");
    }
    assert!(
        cancelled
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .all(|value| value
                == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32).unwrap())
    );

    let checkpoint = cancelled.checkpoint().unwrap();
    let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V9);
    assert_eq!(metadata["replay_step"], "2");
    assert_eq!(metadata["optimizer_step"], "0");
    assert_eq!(metadata["accumulation_index"], "0");
    assert_eq!(metadata["discarded_microbatch_count"], "2");
    assert_eq!(metadata["reset_transition_count"], "1");
    assert_eq!(
        checkpoint.info().reset_capture_identity(),
        cancelled.zero_grad_capture_identity()
    );
    assert_eq!(checkpoint.info().reset_transition_count(), 1);
    let (state, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    let mut malformed_metadata = metadata.clone();
    malformed_metadata.insert("discarded_microbatch_count".into(), "3".into());
    let malformed = save_safetensors(&state, &malformed_metadata).unwrap();
    assert!(CompiledAdamWCheckpoint::from_bytes(malformed).is_err());
    let mut malformed_metadata = metadata.clone();
    malformed_metadata.insert("reset_transition_count".into(), "3".into());
    let malformed = save_safetensors(&state, &malformed_metadata).unwrap();
    assert!(CompiledAdamWCheckpoint::from_bytes(malformed).is_err());
    let mut malformed_metadata = metadata.clone();
    malformed_metadata.insert("reset_capture_identity".into(), "0".into());
    let malformed =
        CompiledAdamWCheckpoint::from_bytes(save_safetensors(&state, &malformed_metadata).unwrap())
            .unwrap();
    assert!(
        CompiledAdamWPlan::compile(config.clone(), initial_parameters(), build_tinybob,)
            .unwrap()
            .restore_checkpoint(&malformed)
            .is_err()
    );
    let mut overflow_metadata = metadata.clone();
    overflow_metadata.insert("replay_step".into(), u64::MAX.to_string());
    overflow_metadata.insert("discarded_microbatch_count".into(), u64::MAX.to_string());
    let overflow =
        CompiledAdamWCheckpoint::from_bytes(save_safetensors(&state, &overflow_metadata).unwrap())
            .unwrap();
    assert!(
        CompiledAdamWPlan::compile(config.clone(), initial_parameters(), build_tinybob,)
            .unwrap()
            .restore_checkpoint(&overflow)
            .is_err()
    );
    let mut zero_discard_metadata = metadata;
    zero_discard_metadata.insert("replay_step".into(), "0".into());
    zero_discard_metadata.insert("discarded_microbatch_count".into(), "0".into());
    let malformed = save_safetensors(&state, &zero_discard_metadata).unwrap();
    assert!(CompiledAdamWCheckpoint::from_bytes(malformed).is_err());
    assert!(
        validate_adamw_progress(
            CompiledTrainingWindowProgress {
                replay_step: 1,
                optimizer_step: 0,
                accumulation_index: 0,
                discarded_microbatches: 1,
                flushed_window_count: 0,
                flushed_microbatch_count: 0,
                reset_transition_count: 0,
            },
            1,
        )
        .is_err()
    );
    let before_noop = checkpoint.clone();
    let noop = cancelled.zero_grad().unwrap();
    assert!(!noop.did_discard());
    assert_eq!(noop.discarded_microbatches(), 0);
    assert_eq!(cancelled.checkpoint().unwrap(), before_noop);

    let mut resumed =
        CpuCompiledAdamW::compile_from_checkpoint(config, &checkpoint, build_tinybob).unwrap();
    assert_eq!(resumed.step_count(), 2);
    assert_eq!(resumed.accumulation_index().unwrap(), 0);
    assert_eq!(
        resumed.inner.plan().unwrap().state_versions,
        cancelled.inner.plan().unwrap().state_versions
    );
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    for _ in 0..3 {
        let expected = cancelled.step(batch(), lr()).unwrap();
        let actual = resumed.step(batch(), lr()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        clean.step(batch(), lr()).unwrap();
    }
    assert_eq!(
        resumed.checkpoint().unwrap(),
        cancelled.checkpoint().unwrap()
    );
    assert_eq!(
        cancelled.parameter_snapshots().unwrap(),
        clean.parameter_snapshots().unwrap()
    );
    assert_eq!(
        cancelled.first_moment_snapshots().unwrap(),
        clean.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        cancelled.second_moment_snapshots().unwrap(),
        clean.second_moment_snapshots().unwrap()
    );
}

#[test]
fn captured_zero_grad_clears_non_finite_accumulators() {
    let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
        .unwrap()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_input("scale", [], DType::F32)
        .unwrap();
    let parameter = TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap();
    let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
        Ok((
            graph.mul(parameters["weight"], inputs["scale"])?,
            BTreeMap::new(),
        ))
    })
    .unwrap();
    let inputs = BTreeMap::from([("scale".into(), TensorData::scalar(f32::INFINITY))]);

    let mut interpreted = plan.prepare_cpu().unwrap();
    interpreted.step(inputs.clone(), lr()).unwrap();
    assert!(
        interpreted.gradient_accumulator_snapshots().unwrap()["weight"]
            .scalar_at(0)
            .as_f64()
            .is_infinite()
    );
    interpreted.zero_grad().unwrap();
    assert_eq!(
        interpreted.gradient_accumulator_snapshots().unwrap()["weight"]
            .scalar_at(0)
            .as_f64(),
        0.0
    );

    let executor = CapturedReplayExecutor::default();
    let mut native = NativeCpuSessionTarget::new(&executor)
        .prepare(&plan)
        .unwrap();
    native.step(inputs, lr()).unwrap();
    native.zero_grad().unwrap();
    assert_eq!(
        native.gradient_accumulator_snapshots().unwrap()["weight"]
            .scalar_at(0)
            .as_f64(),
        0.0
    );
}

#[test]
fn adamw_default_keeps_v1_checkpoint_and_accumulation_one_behavior() {
    let mut compiled = compiled_adamw();
    assert_eq!(compiled.gradient_accumulation_steps(), 1);
    assert_eq!(compiled.max_gradient_norm(), None);
    assert_eq!(compiled.loss_scale(), 1.0);
    assert!(
        compiled
            .gradient_accumulator_snapshots()
            .unwrap()
            .is_empty()
    );
    let result = compiled.step(batch(), lr()).unwrap();
    assert_eq!(result.optimizer_step(), 1);
    assert_eq!(result.accumulation_index(), 0);
    assert!(result.did_update());
    let (_, metadata) = load_safetensors(compiled.checkpoint().unwrap().as_bytes()).unwrap();
    assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V1);
    assert_eq!(metadata["step"], "1");
    let before = compiled.checkpoint().unwrap();
    assert!(!compiled.zero_grad().unwrap().did_discard());
    assert_eq!(compiled.checkpoint().unwrap(), before);
}

#[test]
fn adamw_failure_preserves_parameters_moments_step_and_cursor() {
    let mut compiled = compiled_adamw();
    let parameters = compiled.parameter_snapshots().unwrap();
    let first = compiled.first_moment_snapshots().unwrap();
    let second = compiled.second_moment_snapshots().unwrap();
    let cursor = compiled.inner.cursor.clone();

    assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
    assert_eq!(compiled.step_count(), 0);
    assert_eq!(compiled.optimizer_step().unwrap(), 0);
    assert_eq!(compiled.inner.cursor, cursor);
    assert_eq!(compiled.parameter_snapshots().unwrap(), parameters);
    assert_eq!(compiled.first_moment_snapshots().unwrap(), first);
    assert_eq!(compiled.second_moment_snapshots().unwrap(), second);
}

#[test]
fn owned_module_compile_failures_retain_module_and_preflight_versions() {
    let module = TiedFrozenModule::new([1.0, -1.0]);
    let shared_identity = module.shared.id();
    let error = match CompiledModuleAdamWPlan::compile(module_config(), module, |_, _, _| {
        Err(training("owned graph builder rejected the program"))
    }) {
        Ok(_) => panic!("failing graph builder compiled"),
        Err(error) => error,
    };
    assert!(
        error
            .source_error()
            .to_string()
            .contains("owned graph builder rejected")
    );
    let generic: &CompiledModuleCompileError<_> = &error;
    assert!(format!("{generic:?}").starts_with("CompiledModuleAdamWCompileError"));
    assert!(
        generic
            .to_string()
            .starts_with("owned compiled AdamW compilation failed:")
    );
    let (module, source) = error.into_parts();
    assert!(source.to_string().contains("owned graph builder rejected"));
    assert_eq!(module.shared.id(), shared_identity);

    module.shared.set_version_for_test(u64::MAX).unwrap();
    let error = match CompiledModuleAdamWPlan::compile(module_config(), module, build_tied_frozen) {
        Ok(_) => panic!("unpublishable maximum-version module compiled"),
        Err(error) => error,
    };
    assert!(matches!(
        error.source_error(),
        Error::ParameterVersionOverflow { version: u64::MAX }
    ));
    let module = error.into_module();
    assert_eq!(module.shared.id(), shared_identity);
    assert_eq!(module.shared.version().unwrap(), u64::MAX);

    let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([31, 37]));
    let source = TiedFrozenModule::new([1.0, -1.0]);
    let checkpoint = CompiledAdamWPlan::compile_module_with_dropout(
        module_config(),
        dropout,
        &source,
        build_tied_dropout,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap()
    .checkpoint()
    .unwrap();
    let candidate = TiedFrozenModule::new([1.0, -1.0]);
    let candidate_identity = candidate.shared.id();
    let error = match CompiledModuleAdamWPlan::compile_from_checkpoint(
        module_config(),
        candidate,
        &checkpoint,
        build_tied_frozen,
    ) {
        Ok(_) => panic!("mismatched checkpoint restored"),
        Err(error) => error,
    };
    assert_eq!(error.into_module().shared.id(), candidate_identity);
}

#[test]
fn owned_module_momentum_plan_restores_before_preparation() {
    let config = CompiledMomentumSgdConfig::new(0.9)
        .unwrap()
        .with_input("x", [2], DType::F32)
        .unwrap();
    let source = TiedFrozenModule::new([1.0, -1.0]);
    let source_identity = source.shared.id();
    let plan =
        CompiledModuleMomentumSgdPlan::compile(config.clone(), source, build_tied_frozen).unwrap();
    let capture_identity = plan.capture_identity();
    assert_eq!(plan.step_count(), 0);

    let mut session = plan.prepare(&CpuSessionTarget).unwrap();
    session
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let checkpoint = session.checkpoint().unwrap();
    let trained_source = session.finish().unwrap();
    assert_eq!(trained_source.shared.id(), source_identity);
    assert_eq!(
        trained_source.shared.value().unwrap(),
        checkpoint.parameters()["shared"]
    );

    let destination = TiedFrozenModule::new([1.0, -1.0]);
    let destination_identity = destination.shared.id();
    let destination_before = destination.shared.snapshot().unwrap();
    let plan =
        CompiledModuleMomentumSgdPlan::compile(config, destination, build_tied_frozen).unwrap();
    let mut foreign = checkpoint.clone();
    foreign.capture_identity ^= 1;
    let error = match plan.restore_checkpoint(&foreign) {
        Ok(_) => panic!("foreign momentum checkpoint restored"),
        Err(error) => error,
    };
    assert!(
        error
            .source_error()
            .to_string()
            .contains("capture identity mismatch")
    );
    let generic: &CompiledModulePlanError<_, _> = &error;
    assert!(format!("{generic:?}").starts_with("CompiledModuleMomentumSgdRestoreError"));
    assert!(
        generic
            .to_string()
            .starts_with("owned compiled momentum-SGD checkpoint restore failed:")
    );
    let plan = error.into_plan();
    assert_eq!(plan.capture_identity(), capture_identity);
    assert_eq!(plan.step_count(), 0);

    let plan = plan.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(plan.capture_identity(), capture_identity);
    assert_eq!(plan.step_count(), 1);
    let session = CpuSessionTarget.prepare(plan).unwrap();
    assert_eq!(session.checkpoint().unwrap(), checkpoint);
    let (destination, restored) = session.finish_with_checkpoint().unwrap();
    assert_eq!(restored, checkpoint);
    assert_eq!(destination.shared.id(), destination_identity);
    assert_eq!(
        destination.shared.value().unwrap(),
        checkpoint.parameters()["shared"]
    );
    assert_eq!(
        destination.shared.version().unwrap(),
        destination_before.version + 1
    );
}

#[test]
fn owned_module_momentum_checkpoint_resumes_fresh_identity_atomically() {
    let config = CompiledMomentumSgdConfig::new(0.9)
        .unwrap()
        .with_input("x", [2], DType::F32)
        .unwrap();
    let source = TiedFrozenModule::new([1.0, -1.0]);
    let mut uninterrupted = CompiledModuleTrainingSession::compile_momentum_sgd(
        config.clone(),
        source,
        build_tied_frozen,
    )
    .unwrap();
    let capture_identity = uninterrupted.capture_identity();
    for x in [[0.5, -0.25], [0.75, 0.125]] {
        uninterrupted
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], x.to_vec()).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap();
    }
    let checkpoint = uninterrupted.checkpoint().unwrap();
    assert_eq!(checkpoint.capture_identity(), capture_identity);
    assert_eq!(checkpoint.step(), 2);
    assert_eq!(
        checkpoint.parameters(),
        &uninterrupted.parameter_snapshots().unwrap()
    );
    assert_eq!(
        checkpoint.momenta(),
        &uninterrupted.runtime().momentum_snapshots().unwrap()
    );
    assert_eq!(
        checkpoint.parameter_versions(),
        &uninterrupted.runtime().parameter_versions().unwrap()
    );
    assert_eq!(
        checkpoint.momentum_versions(),
        &uninterrupted.runtime().momentum_versions().unwrap()
    );

    let destination = TiedFrozenModule::new([1.0, -1.0]);
    destination
        .shared
        .replace(TensorData::new([2], vec![9.0, -7.0]).unwrap())
        .unwrap();
    let destination_shared_before = destination.shared.snapshot().unwrap();
    let destination_frozen_before = destination.frozen.snapshot().unwrap();
    let destination_buffer_before = destination.buffer.snapshot().unwrap();
    let destination_shared_identity = destination.shared.id();
    let mut resumed = CompiledModuleTrainingSession::compile_momentum_sgd_from_checkpoint(
        config,
        destination,
        &checkpoint,
        build_tied_frozen,
    )
    .unwrap();
    assert_eq!(resumed.capture_identity(), capture_identity);
    assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    assert_parameter_snapshot_eq(
        &resumed.module.shared.snapshot().unwrap(),
        &destination_shared_before,
    );

    let before_rejection = resumed.checkpoint().unwrap();
    let mut foreign = before_rejection.clone();
    foreign.capture_identity ^= 1;
    assert!(resumed.restore_checkpoint_in_place(&foreign).is_err());
    assert_eq!(resumed.checkpoint().unwrap(), before_rejection);

    let next_inputs =
        BTreeMap::from([("x".into(), TensorData::new([2], vec![-0.5, 0.375]).unwrap())]);
    let expected = uninterrupted
        .step(next_inputs.clone(), TensorData::scalar(0.02))
        .unwrap();
    let actual = resumed.step(next_inputs, TensorData::scalar(0.02)).unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());
    assert_eq!(actual.step(), expected.step());
    assert_eq!(actual.capture_identity(), expected.capture_identity());
    assert_eq!(
        resumed.checkpoint().unwrap(),
        uninterrupted.checkpoint().unwrap()
    );

    let final_checkpoint = resumed.checkpoint().unwrap();
    resumed
        .module
        .frozen
        .replace(TensorData::new([2], vec![7.0, 8.0]).unwrap())
        .unwrap();
    let error = match resumed.finish_with_checkpoint() {
        Ok(_) => panic!("stale momentum module state must reject publication"),
        Err(error) => error,
    };
    let generic: &CompiledModuleTrainingFinishError<_, _> = &error;
    assert!(format!("{generic:?}").starts_with("CompiledModuleTrainingFinishError"));
    assert!(
        generic
            .to_string()
            .starts_with("owned compiled training finalization failed:")
    );
    assert_eq!(generic.session().checkpoint().unwrap(), final_checkpoint);
    generic
        .session()
        .module
        .frozen
        .replace(destination_frozen_before.data.clone())
        .unwrap();
    generic
        .session()
        .module
        .frozen
        .set_version_for_test(destination_frozen_before.version)
        .unwrap();
    let (destination, published) = error.into_session().finish_with_checkpoint().unwrap();
    assert_eq!(published, final_checkpoint);
    assert_eq!(destination.shared.id(), destination_shared_identity);
    assert_eq!(
        destination.shared.value().unwrap(),
        published.parameters()["shared"]
    );
    assert_eq!(
        destination.shared.version().unwrap(),
        destination_shared_before.version + 1
    );
    assert_parameter_snapshot_eq(
        &destination.frozen.snapshot().unwrap(),
        &destination_frozen_before,
    );
    assert_parameter_snapshot_eq(
        &destination.buffer.snapshot().unwrap(),
        &destination_buffer_before,
    );
}

#[test]
fn complete_momentum_module_checkpoint_restores_topology_and_immutable_state() {
    let config = CompiledMomentumSgdConfig::new(0.9)
        .unwrap()
        .with_input("x", [2], DType::F32)
        .unwrap();
    let source = TiedFrozenModule::new([1.0, -1.0]);
    let source_frozen = source.frozen.value().unwrap();
    let source_buffer = source.buffer.value().unwrap();
    let mut source = CompiledModuleTrainingSession::compile_momentum_sgd(
        config.clone(),
        source,
        build_tied_frozen,
    )
    .unwrap();
    source
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let checkpoint = source.module_checkpoint().unwrap();
    assert_eq!(
        checkpoint.optimizer_checkpoint(),
        &source.checkpoint().unwrap()
    );
    assert_eq!(
        CompiledModuleMomentumSgdCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
        checkpoint
    );

    let path = TemporaryCheckpointPath::new("compiled-module-momentum-sgd");
    checkpoint.save_file(path.path()).unwrap();
    assert_eq!(
        CompiledModuleMomentumSgdCheckpoint::load_file(path.path()).unwrap(),
        checkpoint
    );

    let destination = TiedFrozenModule::new([9.0, -7.0]);
    destination
        .frozen
        .replace(TensorData::new([2], vec![8.0, 7.0]).unwrap())
        .unwrap();
    destination.buffer.replace(TensorData::scalar(6.0)).unwrap();
    let destination_shared_identity = destination.shared.id();
    let destination_before = destination.shared.value().unwrap();
    let resumed = CompiledModuleTrainingSession::compile_momentum_sgd_from_module_checkpoint(
        config,
        destination,
        &checkpoint,
        build_tied_frozen,
    )
    .unwrap();
    assert_eq!(resumed.module.shared.value().unwrap(), destination_before);
    assert_eq!(resumed.module_checkpoint().unwrap(), checkpoint);

    let (destination, finished) = resumed.finish_with_module_checkpoint().unwrap();
    assert_eq!(finished, checkpoint);
    assert_eq!(destination.shared.id(), destination_shared_identity);
    assert_eq!(
        destination.shared.value().unwrap(),
        checkpoint.optimizer_checkpoint().parameters()["shared"]
    );
    assert_eq!(destination.frozen.value().unwrap(), source_frozen);
    assert_eq!(destination.buffer.value().unwrap(), source_buffer);

    let (tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(
        metadata["format"],
        "rustgrad-compiled-module-momentum-sgd-v1"
    );
    assert_eq!(metadata["visit_count"], "4");
    assert_eq!(metadata["state_count"], "3");
    assert!(tensors.contains_key("optimizer_checkpoint"));
    assert_eq!(tensors["immutable.1"], source_frozen);
    assert_eq!(tensors["immutable.2"], source_buffer);
}

#[test]
fn momentum_checkpoint_round_trips_deterministic_bytes_and_files() {
    let config = CompiledMomentumSgdConfig::new(0.9)
        .unwrap()
        .with_input("x", [2], DType::F32)
        .unwrap();
    let module = TiedFrozenModule::new([1.0, -1.0]);
    let mut session =
        CompiledModuleTrainingSession::compile_momentum_sgd(config, module, build_tied_frozen)
            .unwrap();
    session
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let checkpoint = session.checkpoint().unwrap();
    let source_parameters = checkpoint.parameters().clone();
    let source_momenta = checkpoint.momenta().clone();

    let bytes = checkpoint.to_bytes().unwrap();
    let restored = CompiledMomentumSgdCheckpoint::from_bytes(&bytes).unwrap();
    assert_eq!(restored, checkpoint);
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    assert_eq!(checkpoint.parameters(), &source_parameters);
    assert_eq!(checkpoint.momenta(), &source_momenta);

    let (tensors, metadata) = load_safetensors(&bytes).unwrap();
    assert_eq!(metadata["format"], "rustgrad-compiled-momentum-sgd-v1");
    assert_eq!(metadata["parameter_count"], "1");
    assert_eq!(metadata["state.0.name"], "shared");
    assert_eq!(
        tensors["state.0.parameter"],
        checkpoint.parameters()["shared"]
    );
    assert_eq!(tensors["state.0.momentum"], checkpoint.momenta()["shared"]);

    let path = TemporaryCheckpointPath::new("compiled-momentum-sgd");
    checkpoint.save_file(path.path()).unwrap();
    assert_eq!(fs::read(path.path()).unwrap(), bytes);
    assert_eq!(
        CompiledMomentumSgdCheckpoint::load_file(path.path()).unwrap(),
        checkpoint
    );
    assert!(matches!(
        CompiledMomentumSgdCheckpoint::load_file_with_limits(
            path.path(),
            SafetensorsReadLimits {
                max_file_bytes: bytes.len() - 1,
            },
        ),
        Err(SafetensorsFileError::Limit { .. })
    ));

    let (tensors, mut unexpected_metadata) = load_safetensors(&bytes).unwrap();
    unexpected_metadata.insert("unexpected".into(), "value".into());
    assert!(
        CompiledMomentumSgdCheckpoint::from_bytes(
            &save_safetensors(&tensors, &unexpected_metadata).unwrap()
        )
        .is_err()
    );

    let (mut mismatched, metadata) = load_safetensors(&bytes).unwrap();
    mismatched.insert("state.0.momentum".into(), TensorData::scalar(0.0));
    assert!(
        CompiledMomentumSgdCheckpoint::from_bytes(
            &save_safetensors(&mismatched, &metadata).unwrap()
        )
        .is_err()
    );

    let (tensors, mut duplicate_metadata) = load_safetensors(&bytes).unwrap();
    duplicate_metadata.insert("parameter_count".into(), "2".into());
    duplicate_metadata.insert("state.1.name".into(), "shared".into());
    duplicate_metadata.insert("state.1.parameter_version".into(), "0".into());
    duplicate_metadata.insert("state.1.momentum_version".into(), "0".into());
    let mut duplicate_tensors = tensors;
    duplicate_tensors.insert(
        "state.1.parameter".into(),
        checkpoint.parameters()["shared"].clone(),
    );
    duplicate_tensors.insert(
        "state.1.momentum".into(),
        checkpoint.momenta()["shared"].clone(),
    );
    assert!(
        CompiledMomentumSgdCheckpoint::from_bytes(
            &save_safetensors(&duplicate_tensors, &duplicate_metadata).unwrap()
        )
        .is_err()
    );
}

#[test]
fn owned_module_adamw_session_seals_replay_and_finishes_atomically() {
    let module = TiedFrozenModule::new([1.0, -1.0]);
    let shared = module.shared.clone();
    let frozen = module.frozen.clone();
    let buffer = module.buffer.clone();
    let shared_before = shared.snapshot().unwrap();
    let frozen_before = frozen.snapshot().unwrap();
    let buffer_before = buffer.snapshot().unwrap();
    let plan =
        CompiledModuleAdamWPlan::compile(module_config(), module, build_tied_frozen).unwrap();
    let capture_identity = plan.capture_identity();
    let session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    let mut session: CompiledModuleTrainingSession<_, _> = session.into_training_session();
    let input = BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
    let step = session.step(input, TensorData::scalar(0.01)).unwrap();
    assert_eq!(step.capture_identity(), capture_identity);
    let published = session.parameter_snapshots().unwrap();
    let expected_checkpoint = session.checkpoint().unwrap();
    let (module, checkpoint) = session.finish_with_checkpoint().unwrap();

    assert_eq!(module.shared.value().unwrap(), published["shared"]);
    assert_eq!(module.shared.version().unwrap(), shared_before.version + 1);
    assert_eq!(module.frozen.value().unwrap(), frozen_before.data);
    assert_eq!(module.frozen.version().unwrap(), frozen_before.version);
    assert_eq!(module.buffer.value().unwrap(), buffer_before.data);
    assert_eq!(module.buffer.version().unwrap(), buffer_before.version);
    assert_eq!(module.shared.id(), shared.id());
    assert_eq!(module.frozen.id(), frozen.id());
    assert_eq!(module.buffer.id(), buffer.id());
    assert_eq!(
        CompiledAdamWCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
        checkpoint
    );
    assert_eq!(checkpoint, expected_checkpoint);

    let stale = TiedFrozenModule::new([1.0, -1.0]);
    let stale_frozen = stale.frozen.clone();
    let stale_frozen_before = stale_frozen.snapshot().unwrap();
    let plan = CompiledModuleAdamWPlan::compile(module_config(), stale, build_tied_frozen).unwrap();
    let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    session
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let runtime_checkpoint = session.checkpoint().unwrap();
    stale_frozen
        .replace(TensorData::new([2], vec![7.0, 8.0]).unwrap())
        .unwrap();
    let error = match session.finish_with_checkpoint() {
        Ok(_) => panic!("stale module state must reject publication"),
        Err(error) => error,
    };
    let compatible: &CompiledModuleAdamWFinishError<_, _> = &error;
    assert!(format!("{compatible:?}").starts_with("CompiledModuleAdamWFinishError"));
    assert!(
        compatible
            .to_string()
            .starts_with("owned compiled AdamW finalization failed:")
    );
    assert_eq!(error.session().step_count(), 1);
    assert_eq!(error.session().checkpoint().unwrap(), runtime_checkpoint);
    assert_eq!(error.session().parameter_snapshots().unwrap().len(), 1);
    assert_eq!(stale_frozen.value().unwrap().values(), &[7.0, 8.0]);
    stale_frozen
        .replace(stale_frozen_before.data.clone())
        .unwrap();
    stale_frozen
        .set_version_for_test(stale_frozen_before.version)
        .unwrap();
    let (stale, retried_checkpoint) = error.into_session().finish_with_checkpoint().unwrap();
    assert_eq!(retried_checkpoint, runtime_checkpoint);
    assert_parameter_snapshot_eq(&stale.frozen.snapshot().unwrap(), &stale_frozen_before);

    let stale_before_prepare = TiedFrozenModule::new([1.0, -1.0]);
    let leaked = stale_before_prepare.shared.clone();
    let plan =
        CompiledModuleAdamWPlan::compile(module_config(), stale_before_prepare, build_tied_frozen)
            .unwrap();
    let capture_identity = plan.capture_identity();
    leaked
        .replace(TensorData::new([2], vec![4.0, 5.0]).unwrap())
        .unwrap();
    let error = match plan.prepare(&CpuSessionTarget::new()) {
        Ok(_) => panic!("stale owned plan prepared"),
        Err(error) => error,
    };
    assert_eq!(error.into_plan().capture_identity(), capture_identity);
}

#[test]
fn program_artifact_rejects_incoherent_zero_token_policies_before_restore() {
    let scalar_plan = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(2).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let scalar_artifact = scalar_plan.program_artifact().unwrap();
    let scalar_checkpoint = scalar_plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    let (bytes, unchecked) = program_artifact::rewrite_json_for_test(&scalar_artifact, |json| {
        json["allow_zero_valid_token_microbatches"] = serde_json::Value::Bool(true);
    });
    let error = CompiledAdamWProgramArtifact::from_bytes(bytes).unwrap_err();
    assert!(error.to_string().contains("policy is inconsistent"));
    let destination = TiedFrozenModule::new([9.0, 10.0]);
    let destination_identity = destination.shared.id();
    let destination_before = destination.shared.snapshot().unwrap();
    let destination = match CompiledModuleAdamWPlan::restore_from_program_artifact(
        destination,
        &unchecked,
        &scalar_checkpoint,
    ) {
        Ok(_) => panic!("zero-token policy without token weighting restored"),
        Err(error) => error.into_module(),
    };
    assert_parameter_snapshot_eq(&destination.shared.snapshot().unwrap(), &destination_before);
    assert_eq!(destination.shared.id(), destination_identity);

    let token_plan = CompiledModuleAdamWPlan::compile_graph(
        token_evaluation_config()
            .with_zero_valid_token_microbatches()
            .unwrap(),
        TokenMeanModule::new(),
        |module, graph, inputs| {
            let weight = module.weight.bind(graph)?;
            let losses = graph.mul(weight, inputs["features"])?;
            Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let weight = module.weight.bind(graph)?;
        let losses = graph.mul(weight, inputs["features"])?;
        Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
    })
    .unwrap();
    let token_artifact = token_plan.program_artifact().unwrap();
    let token_checkpoint = token_plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    let (bytes, unchecked) = program_artifact::rewrite_json_for_test(&token_artifact, |json| {
        json["evaluation"]["allow_zero_valid_token_microbatches"] = serde_json::Value::Bool(false);
    });
    let error = CompiledAdamWProgramArtifact::from_bytes(bytes).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("evaluation artifact schema differs")
    );
    let destination = TokenMeanModule::new();
    let destination_identity = destination.weight.id();
    let destination_before = destination.weight.snapshot().unwrap();
    let destination = match CompiledModuleAdamWPlan::restore_from_program_artifact(
        destination,
        &unchecked,
        &token_checkpoint,
    ) {
        Ok(_) => panic!("mismatched evaluation zero-token policy restored"),
        Err(error) => error.into_module(),
    };
    assert_parameter_snapshot_eq(&destination.weight.snapshot().unwrap(), &destination_before);
    assert_eq!(destination.weight.id(), destination_identity);
}

#[test]
fn program_artifact_rejects_auxiliary_output_flag_mismatch_before_restore() {
    let config = module_config()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_max_gradient_norm(1.0)
        .unwrap()
        .with_clip_report();
    let plan = CompiledModuleAdamWPlan::compile_graph(
        config,
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let artifact = plan.program_artifact().unwrap();
    let checkpoint = plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();

    for (phase, clip_report) in [("partial_flush", false), ("zero_grad", true)] {
        let (bytes, unchecked) = program_artifact::rewrite_json_for_test(&artifact, |json| {
            json[phase]["clip_report"] = serde_json::Value::Bool(clip_report);
        });
        let error = CompiledAdamWProgramArtifact::from_bytes(bytes).unwrap_err();
        assert!(error.to_string().contains("auxiliary observation"));

        let destination = TiedFrozenModule::new([9.0, 10.0]);
        let destination_identity = destination.shared.id();
        let destination_before = destination.shared.snapshot().unwrap();
        let destination = match CompiledModuleAdamWPlan::restore_from_program_artifact(
            destination,
            &unchecked,
            &checkpoint,
        ) {
            Ok(_) => panic!("mismatched auxiliary output flags restored"),
            Err(error) => error.into_module(),
        };
        assert_parameter_snapshot_eq(&destination.shared.snapshot().unwrap(), &destination_before);
        assert_eq!(destination.shared.id(), destination_identity);
    }
}

#[test]
fn compiled_program_artifact_file_io_is_bounded_and_atomic() {
    let plan = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(2).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let artifact = plan.program_artifact().unwrap();
    assert_eq!(plan.program_artifact().unwrap(), artifact);

    let directory = TemporaryCheckpointDirectory::new("compiled-program-artifact-file");
    let path = directory.path().join("program.rgap");
    artifact.save_file(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), artifact.as_bytes());
    assert_eq!(
        CompiledAdamWProgramArtifact::load_file(&path).unwrap(),
        artifact
    );
    assert_eq!(
        CompiledAdamWProgramArtifact::load_file_with_byte_limit(&path, artifact.as_bytes().len(),)
            .unwrap(),
        artifact
    );
    assert!(matches!(
        CompiledAdamWProgramArtifact::load_file_with_byte_limit(
            &path,
            artifact.as_bytes().len() - 1,
        ),
        Err(CompiledAdamWProgramArtifactFileError::Limit { .. })
    ));

    fs::write(&path, b"truncated artifact").unwrap();
    assert!(matches!(
        CompiledAdamWProgramArtifact::load_file(&path),
        Err(CompiledAdamWProgramArtifactFileError::Format(_))
    ));
    let mut corrupt = artifact.as_bytes().to_vec();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    fs::write(&path, corrupt).unwrap();
    assert!(matches!(
        CompiledAdamWProgramArtifact::load_file(&path),
        Err(CompiledAdamWProgramArtifactFileError::Format(_))
    ));
    artifact.save_file(&path).unwrap();

    let oversized = directory.path().join("oversized.rgap");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(268_435_457)
        .unwrap();
    assert!(matches!(
        CompiledAdamWProgramArtifact::load_file_with_byte_limit(&oversized, usize::MAX),
        Err(CompiledAdamWProgramArtifactFileError::Limit {
            actual,
            maximum: 268_435_456,
        }) if actual == 268_435_457
    ));

    let preserved = artifact.as_bytes().to_vec();
    for attempt in 0..128u16 {
        fs::write(
            directory.path().join(format!(
                ".program.rgap.rustgrad-{}-{attempt}.tmp",
                std::process::id()
            )),
            b"another writer",
        )
        .unwrap();
    }
    assert!(matches!(
        artifact.save_file(&path),
        Err(CompiledAdamWProgramArtifactFileError::Io {
            operation: "create unique staging file",
            ..
        })
    ));
    assert_eq!(fs::read(&path).unwrap(), preserved);

    let invalid_path = directory.path().join("invalid.rgap");
    artifact.save_file(&invalid_path).unwrap();
    let (_, unchecked) = program_artifact::rewrite_json_for_test(&artifact, |json| {
        json["gradient_accumulation_steps"] = serde_json::Value::from(1);
    });
    assert!(matches!(
        unchecked.save_file(&invalid_path),
        Err(CompiledAdamWProgramArtifactFileError::Format(_))
    ));
    assert_eq!(fs::read(&invalid_path).unwrap(), artifact.as_bytes());
    assert!(
        fs::read_dir(directory.path())
            .unwrap()
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .all(|name| !name.starts_with(".invalid.rgap.rustgrad-"))
    );

    let alternative = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(3).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap()
    .program_artifact()
    .unwrap();
    assert_ne!(artifact, alternative);
    let concurrent_path = directory.path().join("concurrent.rgap");
    let first = artifact.clone();
    let first_path = concurrent_path.clone();
    let first_writer = std::thread::spawn(move || first.save_file(first_path));
    let second = alternative.clone();
    let second_path = concurrent_path.clone();
    let second_writer = std::thread::spawn(move || second.save_file(second_path));
    first_writer.join().unwrap().unwrap();
    second_writer.join().unwrap().unwrap();
    let concurrent = CompiledAdamWProgramArtifact::load_file(&concurrent_path).unwrap();
    assert!(concurrent == artifact || concurrent == alternative);
    assert!(
        fs::read_dir(directory.path())
            .unwrap()
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .all(|name| !name.starts_with(".concurrent.rgap.rustgrad-"))
    );

    let failed_target = directory.path().join("directory.rgap");
    fs::create_dir(&failed_target).unwrap();
    let occupied = directory.path().join(format!(
        ".directory.rgap.rustgrad-{}-0.tmp",
        std::process::id()
    ));
    fs::write(&occupied, b"another writer").unwrap();
    assert!(matches!(
        artifact.save_file(&failed_target),
        Err(CompiledAdamWProgramArtifactFileError::Io {
            operation: "replace destination",
            ..
        })
    ));
    assert!(failed_target.is_dir());
    assert_eq!(fs::read(&occupied).unwrap(), b"another writer");
    assert_eq!(
        fs::read_dir(directory.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.starts_with(".directory.rgap.rustgrad-"))
            .collect::<Vec<_>>(),
        vec![format!(
            ".directory.rgap.rustgrad-{}-0.tmp",
            std::process::id()
        )]
    );
}

#[test]
fn compiled_resume_bundle_preserves_exact_inner_bytes_and_atomic_file_boundary() {
    let config = module_config()
        .with_gradient_accumulation(2)
        .unwrap()
        .with_window_loss_report();
    let plan = CompiledModuleAdamWPlan::compile_graph(
        config,
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let artifact = plan.program_artifact().unwrap();
    let checkpoint = plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    let bundle = CompiledAdamWResumeBundle::new(artifact.clone(), checkpoint.clone()).unwrap();
    assert_eq!(bundle.format_version(), 1);
    assert_eq!(bundle.program_artifact().as_bytes(), artifact.as_bytes());
    assert_eq!(bundle.checkpoint().as_bytes(), checkpoint.as_bytes());
    assert_eq!(
        CompiledAdamWResumeBundle::new(artifact.clone(), checkpoint.clone()).unwrap(),
        bundle
    );
    assert_eq!(
        CompiledAdamWResumeBundle::from_bytes(bundle.as_bytes().to_vec()).unwrap(),
        bundle
    );

    let mut unsupported = bundle.as_bytes().to_vec();
    unsupported[4] = 2;
    assert!(CompiledAdamWResumeBundle::from_bytes(unsupported).is_err());
    let mut corrupt = bundle.as_bytes().to_vec();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    assert!(CompiledAdamWResumeBundle::from_bytes(corrupt).is_err());
    let mut truncated = bundle.as_bytes().to_vec();
    truncated.pop();
    assert!(CompiledAdamWResumeBundle::from_bytes(truncated).is_err());
    let mut overflowing = bundle.as_bytes().to_vec();
    overflowing[5..13].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(CompiledAdamWResumeBundle::from_bytes(overflowing).is_err());

    let (legacy_tensors, mut legacy_metadata) =
        load_safetensors(checkpoint.optimizer_checkpoint().as_bytes()).unwrap();
    assert_eq!(legacy_metadata["format"], ADAMW_CHECKPOINT_FORMAT_V9);
    legacy_metadata.insert("format".into(), ADAMW_CHECKPOINT_FORMAT_V8.into());
    assert!(
        legacy_metadata
            .remove("accumulation_capture_identity")
            .is_some()
    );
    let legacy_optimizer = CompiledAdamWCheckpoint::from_bytes(
        save_safetensors(&legacy_tensors, &legacy_metadata).unwrap(),
    )
    .unwrap();
    assert_eq!(
        legacy_optimizer.info().accumulation_capture_identity(),
        None
    );
    let decoded_module = checkpoint.decoded();
    let legacy_checkpoint = encode_complete_module_checkpoint(
        &legacy_optimizer,
        decoded_module.evaluation_capture_identity,
        &decoded_module.states,
        &decoded_module.visits,
    )
    .unwrap();
    let legacy_bundle =
        CompiledAdamWResumeBundle::new(artifact.clone(), legacy_checkpoint).unwrap();
    assert_eq!(
        legacy_bundle
            .checkpoint()
            .optimizer_checkpoint()
            .info()
            .accumulation_capture_identity(),
        None
    );

    let evaluated_plan = CompiledModuleAdamWPlan::compile_graph(
        module_config()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_window_loss_report(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
        Ok(CompiledAdamWGraph::scalar(loss, outputs))
    })
    .unwrap();
    let evaluated_checkpoint = evaluated_plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    assert!(CompiledAdamWResumeBundle::new(artifact.clone(), evaluated_checkpoint).is_err());

    let foreign_module_plan = compile_token_training_owner();
    let foreign_module_checkpoint = foreign_module_plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    assert!(CompiledAdamWResumeBundle::new(artifact.clone(), foreign_module_checkpoint).is_err());

    let foreign_plan = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(3).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    assert!(
        CompiledAdamWResumeBundle::new(foreign_plan.program_artifact().unwrap(), checkpoint)
            .is_err()
    );

    let alternative_artifact = foreign_plan.program_artifact().unwrap();
    let alternative_checkpoint = foreign_plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    let alternative =
        CompiledAdamWResumeBundle::new(alternative_artifact, alternative_checkpoint).unwrap();
    assert_ne!(bundle, alternative);

    let directory = TemporaryCheckpointDirectory::new("compiled-resume-bundle-file");
    let path = directory.path().join("resume.rgab");
    bundle.save_file(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), bundle.as_bytes());
    assert_eq!(CompiledAdamWResumeBundle::load_file(&path).unwrap(), bundle);
    assert_eq!(
        CompiledAdamWResumeBundle::load_file_with_byte_limit(&path, bundle.as_bytes().len())
            .unwrap(),
        bundle
    );
    assert!(matches!(
        CompiledAdamWResumeBundle::load_file_with_byte_limit(&path, bundle.as_bytes().len() - 1),
        Err(CompiledAdamWResumeBundleFileError::Limit { .. })
    ));
    let oversized = directory.path().join("oversized.rgab");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(u64::try_from(resume_bundle::MAX_BUNDLE_BYTES).unwrap() + 1)
        .unwrap();
    assert!(matches!(
        CompiledAdamWResumeBundle::load_file_with_byte_limit(&oversized, usize::MAX),
        Err(CompiledAdamWResumeBundleFileError::Limit { actual, maximum })
            if actual == u64::try_from(resume_bundle::MAX_BUNDLE_BYTES).unwrap() + 1
                && maximum == resume_bundle::MAX_BUNDLE_BYTES
    ));

    let preserved = bundle.as_bytes().to_vec();
    for attempt in 0..128u16 {
        fs::write(
            directory.path().join(format!(
                ".resume.rgab.rustgrad-{}-{attempt}.tmp",
                std::process::id()
            )),
            b"another writer",
        )
        .unwrap();
    }
    assert!(matches!(
        bundle.save_file(&path),
        Err(CompiledAdamWResumeBundleFileError::Io {
            operation: "create unique staging file",
            ..
        })
    ));
    assert_eq!(fs::read(&path).unwrap(), preserved);

    let failed_target = directory.path().join("directory.rgab");
    fs::create_dir(&failed_target).unwrap();
    let occupied = directory.path().join(format!(
        ".directory.rgab.rustgrad-{}-0.tmp",
        std::process::id()
    ));
    fs::write(&occupied, b"another writer").unwrap();
    assert!(matches!(
        bundle.save_file(&failed_target),
        Err(CompiledAdamWResumeBundleFileError::Io {
            operation: "replace destination",
            ..
        })
    ));
    assert!(failed_target.is_dir());
    assert_eq!(fs::read(&occupied).unwrap(), b"another writer");
    assert_eq!(
        fs::read_dir(directory.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.starts_with(".directory.rgab.rustgrad-"))
            .collect::<Vec<_>>(),
        vec![format!(
            ".directory.rgab.rustgrad-{}-0.tmp",
            std::process::id()
        )]
    );

    let concurrent_path = directory.path().join("concurrent.rgab");
    let first = bundle.clone();
    let first_path = concurrent_path.clone();
    let first_writer = std::thread::spawn(move || first.save_file(first_path));
    let second = alternative.clone();
    let second_path = concurrent_path.clone();
    let second_writer = std::thread::spawn(move || second.save_file(second_path));
    first_writer.join().unwrap().unwrap();
    second_writer.join().unwrap().unwrap();
    let concurrent = CompiledAdamWResumeBundle::load_file(&concurrent_path).unwrap();
    assert!(concurrent == bundle || concurrent == alternative);
    assert!(
        fs::read_dir(directory.path())
            .unwrap()
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .all(|name| !name.starts_with(".concurrent.rgab.rustgrad-"))
    );
}

#[test]
fn compiled_resume_bundle_seals_one_decode_and_training_topology_for_restore() {
    let plan = CompiledModuleAdamWPlan::compile_graph(
        module_config()
            .with_gradient_accumulation(3)
            .unwrap()
            .with_window_loss_report(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
        Ok(CompiledAdamWGraph::scalar(loss, outputs))
    })
    .unwrap();
    let capture_identity = plan.capture_identity();
    let evaluation_capture_identity = plan.evaluation_capture_identity();
    let owned_restore_probe = plan.plan.clone();
    let public_restore_probe = plan.plan.clone();
    let artifact = plan.program_artifact().unwrap();
    let mut source = plan.prepare(&CpuSessionTarget::new()).unwrap();
    source
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let checkpoint = source.module_checkpoint().unwrap();
    let optimizer_checkpoint = checkpoint.optimizer_checkpoint().clone();

    let owned_allocations = owned_restore_probe.capture_allocations();
    assert!(owned_allocations.main.1 > 0);
    assert!(owned_allocations.accumulation.is_some());
    assert!(owned_allocations.partial_flush.is_some());
    assert!(owned_allocations.zero_grad.is_some());
    assert!(owned_allocations.evaluation.is_some());
    let before_owned = adamw_checkpoint_restore_counts();
    let owned_restore_probe = owned_restore_probe
        .restore_checkpoint_owned(&optimizer_checkpoint)
        .unwrap();
    let after_owned = adamw_checkpoint_restore_counts();
    assert_eq!(owned_restore_probe.capture_allocations(), owned_allocations);
    assert_eq!(
        after_owned.borrowed_plan_clones,
        before_owned.borrowed_plan_clones
    );
    assert_eq!(
        after_owned.consumed_plan_restores,
        before_owned.consumed_plan_restores + 1
    );

    let public_allocations = public_restore_probe.capture_allocations();
    let public_progress = public_restore_probe.progress;
    let before_public = adamw_checkpoint_restore_counts();
    let public_restored = public_restore_probe
        .restore_checkpoint(&optimizer_checkpoint)
        .unwrap();
    let after_public = adamw_checkpoint_restore_counts();
    assert_eq!(
        public_restore_probe.capture_allocations(),
        public_allocations
    );
    assert_eq!(public_restore_probe.progress, public_progress);
    assert_eq!(public_restored.progress.accumulation_index, 1);
    assert_eq!(
        after_public.borrowed_plan_clones,
        before_public.borrowed_plan_clones + 1
    );
    assert_eq!(
        after_public.consumed_plan_restores,
        before_public.consumed_plan_restores
    );

    let incompatible = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(2).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let incompatible_checkpoint = incompatible
        .plan
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .checkpoint()
        .unwrap();
    let before_rejected = adamw_checkpoint_restore_counts();
    let error = match public_restore_probe.restore_checkpoint(&incompatible_checkpoint) {
        Ok(_) => panic!("an incompatible checkpoint was restored"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        Error::SessionTraining {
            reason: "compiled AdamW checkpoint accumulation policy mismatch".into(),
        }
    );
    assert_eq!(
        public_restore_probe.capture_allocations(),
        public_allocations
    );
    assert_eq!(public_restore_probe.progress, public_progress);
    assert_eq!(adamw_checkpoint_restore_counts(), before_rejected);

    let encoded = CompiledAdamWResumeBundle::new(artifact, checkpoint)
        .unwrap()
        .into_bytes();

    let before = program_artifact::portable_resume_decode_counts();
    let bundle = CompiledAdamWResumeBundle::from_bytes(encoded).unwrap();
    let after_load = program_artifact::portable_resume_decode_counts();
    assert_eq!(after_load.program_wire - before.program_wire, 1);
    assert_eq!(after_load.mixed_captures - before.mixed_captures, 4);
    assert_eq!(
        after_load.evaluation_captures - before.evaluation_captures,
        1
    );
    assert_eq!(after_load.module_checkpoints - before.module_checkpoints, 1);
    assert_eq!(after_load.pair_admissions - before.pair_admissions, 1);
    assert_eq!(after_load.topology_seals, before.topology_seals);
    let retained_capture_extents = bundle.program_artifact().retained_capture_extents();
    assert_eq!(retained_capture_extents.len(), 5);
    assert!(
        retained_capture_extents
            .iter()
            .all(|extent| *extent == (0, 0)),
        "admission must release every raw capture allocation after materializing typed captures"
    );

    let cloned = bundle.clone();
    assert!(bundle.shares_admission_with(&cloned));
    let before_restore = adamw_checkpoint_restore_counts();
    let restored = CompiledModuleAdamWPlan::restore_from_resume_bundle(
        TiedFrozenModule::new([9.0, 10.0]),
        &cloned,
    )
    .unwrap();
    let independently_restored = CompiledModuleAdamWPlan::restore_from_resume_bundle(
        TiedFrozenModule::new([11.0, 12.0]),
        &bundle,
    )
    .unwrap();
    let after_restore = adamw_checkpoint_restore_counts();
    let after_topology = program_artifact::portable_resume_decode_counts();
    assert_eq!(
        after_restore.borrowed_plan_clones, before_restore.borrowed_plan_clones,
        "admitted owner restore must not clone its freshly reconstructed program"
    );
    assert_eq!(
        after_restore.consumed_plan_restores,
        before_restore.consumed_plan_restores + 2
    );
    assert_eq!(after_topology.program_wire, after_load.program_wire);
    assert_eq!(after_topology.mixed_captures, after_load.mixed_captures);
    assert_eq!(
        after_topology.evaluation_captures,
        after_load.evaluation_captures
    );
    assert_eq!(
        after_topology.module_checkpoints,
        after_load.module_checkpoints
    );
    assert_eq!(after_topology.pair_admissions, after_load.pair_admissions);
    assert_eq!(after_topology.topology_seals, after_load.topology_seals + 1);
    assert_eq!(
        after_topology.topology_phase_validations,
        after_load.topology_phase_validations + 4
    );
    assert_eq!(
        after_topology.recurrent_execution_plans,
        after_load.recurrent_execution_plans + 4
    );
    assert_eq!(
        after_topology.evaluation_execution_plans,
        after_load.evaluation_execution_plans + 1
    );
    assert_eq!(
        after_topology.cursor_projections,
        after_load.cursor_projections + 3
    );
    assert_eq!(
        restored.plan.topology_allocations(),
        independently_restored.plan.topology_allocations(),
        "restored owners must share only immutable admitted replay topology"
    );
    let mismatched_destination = TiedFrozenModule {
        shared: Parameter::new(TensorData::new([3], vec![7.0, 8.0, 9.0]).unwrap(), true),
        frozen: Parameter::new(TensorData::new([2], vec![3.0, 4.0]).unwrap(), false),
        buffer: Parameter::new(TensorData::scalar(5.0), false),
    };
    let mismatched_before = mismatched_destination.shared.snapshot().unwrap();
    let before_mismatch = program_artifact::portable_resume_decode_counts();
    let mismatched_destination = match CompiledModuleAdamWPlan::restore_from_resume_bundle(
        mismatched_destination,
        &bundle,
    ) {
        Ok(_) => panic!("admitted topology restored into a mismatched module"),
        Err(error) => error.into_module(),
    };
    assert_parameter_snapshot_eq(
        &mismatched_destination.shared.snapshot().unwrap(),
        &mismatched_before,
    );
    assert_eq!(
        program_artifact::portable_resume_decode_counts(),
        before_mismatch,
        "destination rejection must neither mutate the module nor rebuild admitted topology"
    );
    assert_eq!(restored.capture_identity(), capture_identity);
    assert_eq!(
        restored.evaluation_capture_identity(),
        evaluation_capture_identity
    );
    let inspection = independently_restored.inspection().unwrap();
    let mut restored = restored.prepare(&CpuSessionTarget::new()).unwrap();
    let executor = CapturedReplayExecutor::default();
    let independent = independently_restored
        .prepare(&NativeCpuSessionTarget::new(&executor))
        .unwrap();
    let preparation = independent.native_cpu_preparation_report();
    assert_eq!(preparation.main().capture_identity(), inspection.main().0);
    assert_eq!(preparation.main().execution_plan(), inspection.main().1);
    assert_eq!(preparation.main().fallback_count(), 0);
    for (prepared, expected) in [
        (preparation.accumulation(), inspection.accumulation()),
        (preparation.partial_flush(), inspection.partial_flush()),
        (preparation.zero_grad(), inspection.zero_grad()),
        (preparation.evaluation(), inspection.evaluation()),
    ] {
        let prepared = prepared.expect("the N=3 artifact retains every attached phase");
        let expected = expected.expect("the admitted topology retains every attached phase");
        assert_eq!(prepared.capture_identity(), expected.0);
        assert_eq!(prepared.execution_plan(), expected.1);
        assert_eq!(prepared.fallback_count(), 0);
    }
    assert_eq!(restored.checkpoint().unwrap(), optimizer_checkpoint);
    assert_eq!(independent.checkpoint().unwrap(), optimizer_checkpoint);
    restored
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    assert_ne!(restored.checkpoint().unwrap(), optimizer_checkpoint);
    assert_eq!(independent.checkpoint().unwrap(), optimizer_checkpoint);
}

#[test]
fn complete_module_checkpoint_restores_constants_without_mutating_destination() {
    let config = module_config().with_gradient_accumulation(2).unwrap();
    let source = TiedFrozenModule::new([1.0, -1.0]);
    let source_frozen = source.frozen.clone();
    let source_buffer = source.buffer.clone();
    let source_plan =
        CompiledModuleAdamWPlan::compile_graph(config.clone(), source, |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        })
        .unwrap();
    let capture_identity = source_plan.capture_identity();
    let source_inspection = source_plan.inspection().unwrap();
    let source_contract = source_plan.plan.contract.clone();
    let program_artifact = source_plan.program_artifact().unwrap();
    assert_eq!(source_plan.program_artifact().unwrap(), program_artifact);
    assert_eq!(program_artifact.info().format_version(), 2);
    assert_eq!(program_artifact.as_bytes()[4], 2);
    program_artifact::rewrite_json_for_test(&program_artifact, |json| {
        assert!(json["metal"]["main"].as_array().unwrap().len() > 21);
        assert!(!json["metal"]["partial_flush"].is_null());
        assert!(json["metal"]["evaluation"].is_null());
    });
    let (corrupt_recipe_bytes, _) =
        program_artifact::rewrite_json_for_test(&program_artifact, |json| {
            let byte = json["metal"]["main"][13].as_u64().unwrap();
            json["metal"]["main"][13] = serde_json::Value::from(byte ^ 1);
        });
    assert!(CompiledAdamWProgramArtifact::from_bytes(corrupt_recipe_bytes).is_err());
    let artifact_file = TemporaryCheckpointPath::new("compiled-adamw-program-artifact");
    program_artifact.save_file(artifact_file.path()).unwrap();
    let program_artifact = CompiledAdamWProgramArtifact::load_file(artifact_file.path()).unwrap();
    let mut source = source_plan.prepare(&CpuSessionTarget::new()).unwrap();
    let batch = || BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
    source.step(batch(), TensorData::scalar(0.01)).unwrap();
    let checkpoint = source.module_checkpoint().unwrap();
    assert_eq!(
        checkpoint.optimizer_checkpoint(),
        &source.checkpoint().unwrap()
    );
    assert_eq!(checkpoint.evaluation_capture_identity(), None);
    assert_eq!(
        program_artifact.info().capture_identity(),
        checkpoint.optimizer_checkpoint().info().capture_identity()
    );
    let resume_bundle =
        CompiledAdamWResumeBundle::new(program_artifact.clone(), checkpoint.clone()).unwrap();
    let mismatched_destination = TiedFrozenModule {
        shared: Parameter::new(TensorData::new([3], vec![9.0, 10.0, 11.0]).unwrap(), true),
        frozen: Parameter::new(TensorData::new([2], vec![9.0, 10.0]).unwrap(), false),
        buffer: Parameter::new(TensorData::scalar(3.0), false),
    };
    let mismatched_identity = mismatched_destination.shared.id();
    let mismatched_before = mismatched_destination.shared.snapshot().unwrap();
    let mismatched_destination = match CompiledModuleAdamWPlan::restore_from_resume_bundle(
        mismatched_destination,
        &resume_bundle,
    ) {
        Ok(_) => panic!("resume bundle restored into mismatched module topology"),
        Err(error) => error.into_module(),
    };
    assert_parameter_snapshot_eq(
        &mismatched_destination.shared.snapshot().unwrap(),
        &mismatched_before,
    );
    assert_eq!(mismatched_destination.shared.id(), mismatched_identity);
    let artifact_destination = TiedFrozenModule::new([9.0, 10.0]);
    let artifact_destination_before = artifact_destination.shared.snapshot().unwrap();
    let restored =
        CompiledModuleAdamWPlan::restore_from_resume_bundle(artifact_destination, &resume_bundle)
            .unwrap();
    assert_eq!(restored.plan.contract, source_contract);
    assert_eq!(
        restored.program_artifact().unwrap().as_bytes(),
        program_artifact.as_bytes(),
        "flat RGAP bytes must survive the private contract adapter"
    );
    let restored_inspection = restored.inspection().unwrap();
    assert!(
        restored_inspection.compile_phases().is_none(),
        "artifact-derived plans must not synthesize compile observations"
    );
    assert_eq!(source_inspection.initial_replay_step(), 0);
    assert_eq!(
        restored_inspection.initial_replay_step(),
        checkpoint.optimizer_checkpoint().info().replay_step()
    );
    assert_eq!(restored_inspection.main(), source_inspection.main());
    assert_eq!(
        restored_inspection.accumulation(),
        source_inspection.accumulation()
    );
    assert_eq!(
        restored_inspection.partial_flush(),
        source_inspection.partial_flush()
    );
    assert_eq!(
        restored_inspection.zero_grad(),
        source_inspection.zero_grad()
    );
    assert_eq!(
        restored_inspection.evaluation(),
        source_inspection.evaluation()
    );
    assert_eq!(
        restored_inspection.recurrent_state_count(),
        source_inspection.recurrent_state_count()
    );
    assert_eq!(
        restored_inspection.recurrent_state_bytes(),
        source_inspection.recurrent_state_bytes()
    );
    assert_parameter_snapshot_eq(
        &restored.module.shared.snapshot().unwrap(),
        &artifact_destination_before,
    );
    let restored_metal = restored
        .plan
        .metal_plan(
            MetalRenderer::new(
                8,
                crate::runtime::metal::MetalCapabilities {
                    max_buffer_length: 1 << 30,
                    unified_memory: true,
                    family: "Apple9".into(),
                },
            )
            .unwrap(),
        )
        .unwrap();
    assert!(restored_metal.partial_flush.is_some());
    assert_eq!(restored_metal.inner.program_identity, capture_identity);
    let restored = restored.prepare(&CpuSessionTarget::new()).unwrap();
    assert_eq!(
        restored.checkpoint().unwrap(),
        *checkpoint.optimizer_checkpoint()
    );
    let mut corrupt_artifact = program_artifact.as_bytes().to_vec();
    let last = corrupt_artifact.len() - 1;
    corrupt_artifact[last] ^= 1;
    assert!(CompiledAdamWProgramArtifact::from_bytes(corrupt_artifact).is_err());
    let foreign_artifact = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(3).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap()
    .program_artifact()
    .unwrap();
    let mismatch_destination = TiedFrozenModule::new([11.0, 12.0]);
    let mismatch_before = mismatch_destination.shared.snapshot().unwrap();
    let mismatch = match CompiledModuleAdamWPlan::restore_from_program_artifact(
        mismatch_destination,
        &foreign_artifact,
        &checkpoint,
    ) {
        Ok(_) => panic!("foreign program artifact restored"),
        Err(error) => error.into_module(),
    };
    assert_parameter_snapshot_eq(&mismatch.shared.snapshot().unwrap(), &mismatch_before);
    assert_eq!(
        CompiledModuleAdamWCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
        checkpoint
    );
    let module_file = TemporaryCheckpointPath::new("compiled-module-adamw-checkpoint");
    checkpoint.save_file(module_file.path()).unwrap();
    assert_eq!(fs::read(module_file.path()).unwrap(), checkpoint.as_bytes());
    assert_eq!(
        CompiledModuleAdamWCheckpoint::load_file(module_file.path()).unwrap(),
        checkpoint
    );
    assert_eq!(
        CompiledModuleAdamWCheckpoint::load_file_with_limits(
            module_file.path(),
            crate::SafetensorsReadLimits {
                max_file_bytes: checkpoint.as_bytes().len(),
            },
        )
        .unwrap(),
        checkpoint
    );
    assert!(matches!(
        CompiledModuleAdamWCheckpoint::load_file_with_limits(
            module_file.path(),
            crate::SafetensorsReadLimits {
                max_file_bytes: checkpoint.as_bytes().len() - 1,
            },
        ),
        Err(crate::SafetensorsFileError::Limit { .. })
    ));

    let optimizer_file = TemporaryCheckpointPath::new("compiled-adamw-checkpoint");
    checkpoint
        .optimizer_checkpoint()
        .save_file(optimizer_file.path())
        .unwrap();
    assert_eq!(
        CompiledAdamWCheckpoint::load_file(optimizer_file.path()).unwrap(),
        checkpoint.optimizer_checkpoint().clone()
    );
    fs::write(module_file.path(), b"truncated checkpoint").unwrap();
    assert!(matches!(
        CompiledModuleAdamWCheckpoint::load_file_with_limits(
            module_file.path(),
            crate::SafetensorsReadLimits::default(),
        ),
        Err(crate::SafetensorsFileError::Format(_))
    ));
    assert_eq!(source.module_checkpoint().unwrap(), checkpoint);
    let (envelope_tensors, envelope_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    assert_eq!(envelope_tensors.len(), 3);
    assert!(envelope_tensors.contains_key("optimizer_checkpoint"));
    assert_eq!(
        envelope_metadata.get("format").map(String::as_str),
        Some("rustgrad-compiled-module-adamw-v1")
    );

    let schema_destination = TiedFrozenModule::new([4.0, 5.0]);
    let schema_shared_before = schema_destination.shared.snapshot().unwrap();
    let schema_frozen_before = schema_destination.frozen.snapshot().unwrap();
    let schema_buffer_before = schema_destination.buffer.snapshot().unwrap();
    let (schema_tensors, mut schema_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    schema_metadata.insert("unexpected".into(), "field".into());
    assert!(
        CompiledModuleAdamWCheckpoint::from_bytes(
            save_safetensors(&schema_tensors, &schema_metadata).unwrap()
        )
        .is_err()
    );
    assert_parameter_snapshot_eq(
        &schema_destination.shared.snapshot().unwrap(),
        &schema_shared_before,
    );
    assert_parameter_snapshot_eq(
        &schema_destination.frozen.snapshot().unwrap(),
        &schema_frozen_before,
    );
    assert_parameter_snapshot_eq(
        &schema_destination.buffer.snapshot().unwrap(),
        &schema_buffer_before,
    );

    let topology_destination = TiedFrozenModule::new([5.0, 6.0]);
    let topology_shared_before = topology_destination.shared.snapshot().unwrap();
    let topology_frozen_before = topology_destination.frozen.snapshot().unwrap();
    let topology_buffer_before = topology_destination.buffer.snapshot().unwrap();
    let (topology_tensors, mut topology_metadata) =
        load_safetensors(checkpoint.as_bytes()).unwrap();
    topology_metadata.insert("visit.1.name".into(), "renamed_alias".into());
    let topology_checkpoint = CompiledModuleAdamWCheckpoint::from_bytes(
        save_safetensors(&topology_tensors, &topology_metadata).unwrap(),
    )
    .unwrap();
    let topology_error = CompiledModuleAdamWPlan::compile_graph_from_module_checkpoint(
        config.clone(),
        topology_destination,
        &topology_checkpoint,
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .err()
    .expect("mismatched alias topology must reject");
    let topology_destination = topology_error.into_module();
    assert_parameter_snapshot_eq(
        &topology_destination.shared.snapshot().unwrap(),
        &topology_shared_before,
    );
    assert_parameter_snapshot_eq(
        &topology_destination.frozen.snapshot().unwrap(),
        &topology_frozen_before,
    );
    assert_parameter_snapshot_eq(
        &topology_destination.buffer.snapshot().unwrap(),
        &topology_buffer_before,
    );

    let destination = TiedFrozenModule::new([7.0, 8.0]);
    let shared = destination.shared.clone();
    let frozen = destination.frozen.clone();
    let buffer = destination.buffer.clone();
    buffer.replace(TensorData::scalar(-4.0)).unwrap();
    let shared_before = shared.snapshot().unwrap();
    let frozen_before = frozen.snapshot().unwrap();
    let buffer_before = buffer.snapshot().unwrap();
    let restored_plan = CompiledModuleAdamWPlan::compile_graph_from_module_checkpoint(
        config.clone(),
        destination,
        &checkpoint,
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_parameter_snapshot_eq(&shared.snapshot().unwrap(), &shared_before);
    assert_parameter_snapshot_eq(&frozen.snapshot().unwrap(), &frozen_before);
    assert_parameter_snapshot_eq(&buffer.snapshot().unwrap(), &buffer_before);

    let mut resumed = restored_plan.prepare(&CpuSessionTarget::new()).unwrap();
    assert_eq!(
        resumed.checkpoint().unwrap(),
        checkpoint.optimizer_checkpoint().clone()
    );
    let uninterrupted_step = source.step(batch(), TensorData::scalar(0.01)).unwrap();
    let resumed_step = resumed.step(batch(), TensorData::scalar(0.01)).unwrap();
    assert_eq!(resumed_step.loss(), uninterrupted_step.loss());
    assert_eq!(resumed_step.outputs(), uninterrupted_step.outputs());
    assert_eq!(resumed.checkpoint().unwrap(), source.checkpoint().unwrap());
    let source = source.finish().unwrap();
    let destination = resumed.finish().unwrap();
    assert_eq!(
        destination.state_dict().unwrap(),
        source.state_dict().unwrap()
    );
    assert_eq!(destination.shared.id(), shared.id());
    assert!(destination.shared.is_trainable());
    assert!(!destination.frozen.is_trainable());
    assert!(!destination.buffer.is_trainable());
    assert_eq!(
        destination.frozen.value().unwrap(),
        source_frozen.value().unwrap()
    );
    assert_eq!(
        destination.buffer.value().unwrap(),
        source_buffer.value().unwrap()
    );
    assert_eq!(
        destination.shared.version().unwrap(),
        shared_before.version + 1
    );
    assert_eq!(
        destination.frozen.version().unwrap(),
        frozen_before.version + 1
    );
    assert_eq!(
        destination.buffer.version().unwrap(),
        buffer_before.version + 1
    );

    let malformed_destination = TiedFrozenModule::new([9.0, 10.0]);
    let malformed_shared = malformed_destination.shared.snapshot().unwrap();
    let malformed_frozen = malformed_destination.frozen.snapshot().unwrap();
    let malformed_buffer = malformed_destination.buffer.snapshot().unwrap();
    let (mut tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
    let immutable = tensors
        .iter_mut()
        .find(|(name, _)| name.starts_with("immutable."))
        .unwrap();
    *immutable.1 = TensorData::scalar(1.0);
    let malformed =
        CompiledModuleAdamWCheckpoint::from_bytes(save_safetensors(&tensors, &metadata).unwrap())
            .unwrap();
    let error = CompiledModuleAdamWPlan::compile_graph_from_module_checkpoint(
        config,
        malformed_destination,
        &malformed,
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .err()
    .expect("malformed immutable descriptor must reject");
    let malformed_destination = error.into_module();
    assert_parameter_snapshot_eq(
        &malformed_destination.shared.snapshot().unwrap(),
        &malformed_shared,
    );
    assert_parameter_snapshot_eq(
        &malformed_destination.frozen.snapshot().unwrap(),
        &malformed_frozen,
    );
    assert_parameter_snapshot_eq(
        &malformed_destination.buffer.snapshot().unwrap(),
        &malformed_buffer,
    );
}

#[test]
fn portable_v2_program_artifact_reconstructs_strict_metal_training_wrappers() {
    let source = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(3).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
        Ok(CompiledAdamWGraph::scalar(loss, outputs))
    })
    .unwrap();
    let capture_identity = source.capture_identity();
    let artifact = source.program_artifact().unwrap();
    assert_eq!(artifact.info().format_version(), 2);
    assert_eq!(artifact.retained_metal_recipe_extents().len(), 3);
    assert!(
        artifact
            .retained_metal_recipe_extents()
            .into_iter()
            .all(|extent| extent == (0, 0))
    );
    let checkpoint = source
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    let bundle = CompiledAdamWResumeBundle::new(artifact.clone(), checkpoint).unwrap();
    let restored = CompiledModuleAdamWPlan::restore_from_resume_bundle(
        TiedFrozenModule::new([7.0, 8.0]),
        &bundle,
    )
    .unwrap();
    assert_eq!(restored.program_artifact().unwrap(), artifact);
    let rendered = restored
        .plan
        .metal_plan(
            MetalRenderer::new(
                8,
                crate::runtime::metal::MetalCapabilities {
                    max_buffer_length: 1 << 30,
                    unified_memory: true,
                    family: "Apple9".into(),
                },
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(rendered.capture_identity(), capture_identity);
    assert!(rendered.partial_flush.is_some());
    assert!(rendered.inner.evaluation.is_some());
    assert_eq!(rendered.summary().fallback_count, 0);
}

#[test]
fn cpu_only_scheduled_program_artifact_retains_v1_identity_and_round_trip() {
    let schedule = CompiledMultiStepLr::new(0.01, 0.5, [2, 4]).unwrap();
    let source = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_captured_multi_step_lr(schedule),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let artifact = source.program_artifact().unwrap();
    assert_eq!(artifact.info().format_version(), 1);
    assert_eq!(artifact.as_bytes()[4], 1);
    program_artifact::rewrite_json_for_test(&artifact, |json| {
        assert!(json.get("metal").is_none());
    });
    let checkpoint = source
        .prepare(&CpuSessionTarget::new())
        .unwrap()
        .module_checkpoint()
        .unwrap();
    let restored = CompiledModuleAdamWPlan::restore_from_program_artifact(
        TiedFrozenModule::new([9.0, 10.0]),
        &CompiledAdamWProgramArtifact::from_bytes(artifact.as_bytes().to_vec()).unwrap(),
        &checkpoint,
    )
    .unwrap();
    assert_eq!(restored.program_artifact().unwrap(), artifact);
    assert!(
        restored
            .plan
            .metal_plan(
                MetalRenderer::new(
                    8,
                    crate::runtime::metal::MetalCapabilities {
                        max_buffer_length: 1 << 30,
                        unified_memory: true,
                        family: "Apple9".into(),
                    },
                )
                .unwrap(),
            )
            .is_err()
    );
}

#[test]
fn complete_checkpoint_finish_publishes_one_snapshot_and_resumes_exactly() {
    let config = tied_token_mean_config();
    let dropout = tied_token_mean_dropout();
    let source_module = TiedFrozenModule::new([1.0, -1.0]);
    let source_shared = source_module.shared.snapshot().unwrap();
    let source_tied_identity = source_module.shared.id();
    let source_frozen = source_module.frozen.snapshot().unwrap();
    let source_buffer = source_module.buffer.snapshot().unwrap();
    let source_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout(
        config.clone(),
        dropout,
        source_module,
        build_tied_frozen_token_mean,
    )
    .unwrap();
    let capture_identity = source_plan.capture_identity();
    let mut source = source_plan.prepare(&CpuSessionTarget::new()).unwrap();

    let first = source
        .step(
            tied_token_mean_batch([0.5, -0.25], [1.0, 0.0]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    assert_eq!(first.loss_weight(), 1);
    assert!(first.clip_report().is_none());
    let second = source
        .step(
            tied_token_mean_batch([-0.75, 0.25], [1.0, 1.0]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    assert_eq!(second.loss_weight(), 2);
    assert!(second.clip_report().is_some());
    source
        .step(
            tied_token_mean_batch([0.25, 0.75], [0.0, 1.0]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let partial = source.module_checkpoint().unwrap();
    assert_eq!(partial.optimizer_checkpoint().info().optimizer_step(), 1);
    assert_eq!(
        partial.optimizer_checkpoint().info().accumulation_index(),
        1
    );
    assert_eq!(
        partial
            .optimizer_checkpoint()
            .info()
            .accumulated_token_count(),
        Some(1)
    );
    assert!(
        partial
            .optimizer_checkpoint()
            .info()
            .dropout_block_counter()
            .unwrap()
            > 0
    );

    let destination_module = TiedFrozenModule::new([7.0, 8.0]);
    destination_module
        .buffer
        .replace(TensorData::scalar(-4.0))
        .unwrap();
    let destination_shared = destination_module.shared.snapshot().unwrap();
    let destination_frozen = destination_module.frozen.snapshot().unwrap();
    let destination_buffer = destination_module.buffer.snapshot().unwrap();
    let destination_tied_identity = destination_module.shared.id();
    let restored_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_from_module_checkpoint(
        config,
        dropout,
        destination_module,
        &partial,
        build_tied_frozen_token_mean,
    )
    .unwrap();
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    let mut resumed = restored_plan.prepare(&CpuSessionTarget::new()).unwrap();
    assert_eq!(resumed.module_checkpoint().unwrap(), partial);
    assert_parameter_snapshot_eq(
        &resumed.module.shared.snapshot().unwrap(),
        &destination_shared,
    );
    assert_parameter_snapshot_eq(
        &resumed.module.frozen.snapshot().unwrap(),
        &destination_frozen,
    );
    assert_parameter_snapshot_eq(
        &resumed.module.buffer.snapshot().unwrap(),
        &destination_buffer,
    );

    for (x, mask) in [([1.0, -0.5], [1.0, 1.0]), ([-0.5, 0.5], [1.0, 0.0])] {
        let expected = source
            .step(tied_token_mean_batch(x, mask), TensorData::scalar(0.01))
            .unwrap();
        let actual = resumed
            .step(tied_token_mean_batch(x, mask), TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.loss_weight(), expected.loss_weight());
        assert_eq!(actual.clip_report(), expected.clip_report());
        assert_eq!(
            resumed.module_checkpoint().unwrap(),
            source.module_checkpoint().unwrap()
        );
    }

    let expected_optimizer = source.checkpoint().unwrap();
    let expected_complete = source.module_checkpoint().unwrap();
    assert_eq!(expected_optimizer.info().accumulation_index(), 1);
    assert_eq!(expected_optimizer.info().accumulated_token_count(), Some(1));
    let checkpoint_calls = Rc::new(Cell::new(0));
    let source = source.map_runtime(|runtime| CheckpointCountingRuntime {
        inner: runtime,
        checkpoint_calls: Rc::clone(&checkpoint_calls),
    });
    let (source_module, completed) = source.finish_with_module_checkpoint().unwrap();
    assert_eq!(checkpoint_calls.get(), 1);
    assert_eq!(completed, expected_complete);
    assert_eq!(completed.optimizer_checkpoint(), &expected_optimizer);
    assert_eq!(
        CompiledModuleAdamWCheckpoint::from_bytes(completed.as_bytes().to_vec()).unwrap(),
        completed
    );
    assert_eq!(
        source_module.shared.value().unwrap(),
        decode_adamw_checkpoint(completed.optimizer_checkpoint().as_bytes())
            .unwrap()
            .parameters["shared"]
    );
    assert_eq!(source_module.shared.id(), source_tied_identity);
    assert_eq!(
        source_module.shared.version().unwrap(),
        source_shared.version + 1
    );
    assert_parameter_snapshot_eq(&source_module.frozen.snapshot().unwrap(), &source_frozen);
    assert_parameter_snapshot_eq(&source_module.buffer.snapshot().unwrap(), &source_buffer);
    let decoded = completed.decoded();
    assert_eq!(decoded.states.len(), 3);
    assert_eq!(decoded.visits.len(), 4);
    assert_eq!(decoded.visits[0].canonical_name, "shared");
    assert_eq!(decoded.visits[1].canonical_name, "shared");

    let (destination_module, resumed_complete) = resumed.finish_with_module_checkpoint().unwrap();
    assert_eq!(resumed_complete, completed);
    assert_eq!(
        destination_module.state_dict().unwrap(),
        source_module.state_dict().unwrap()
    );
    assert_eq!(destination_module.shared.id(), destination_tied_identity);
    assert!(!destination_module.frozen.is_trainable());
    assert!(!destination_module.buffer.is_trainable());
}

#[test]
fn checkpointed_finish_retains_session_after_late_publication_race() {
    let module = FinishRaceModule::new();
    let weight = module.weight.clone();
    let initial = weight.snapshot().unwrap();
    let plan =
        CompiledModuleAdamWPlan::compile(module_config(), module, build_finish_race).unwrap();
    let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    session
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    let checkpoint = session.module_checkpoint().unwrap();
    session.module.arm_finish_race();

    let error = session.finish_with_module_checkpoint().unwrap_err();
    assert!(matches!(
        error.source_error(),
        Error::ParameterVersionConflict {
            expected: 0,
            actual: 1
        }
    ));
    assert_eq!(
        error.session().checkpoint().unwrap(),
        checkpoint.optimizer_checkpoint().clone()
    );
    assert_eq!(error.session().step_count(), 1);

    weight.set_version_for_test(initial.version).unwrap();
    let (module, retried_checkpoint) = error
        .into_session()
        .finish_with_module_checkpoint()
        .unwrap();
    assert_eq!(retried_checkpoint, checkpoint);
    assert_eq!(module.weight.version().unwrap(), initial.version + 1);
    assert_eq!(
        module.weight.value().unwrap(),
        decode_adamw_checkpoint(checkpoint.optimizer_checkpoint().as_bytes())
            .unwrap()
            .parameters["weight"]
    );
}

#[test]
fn adamw_module_binding_owns_trainable_state_and_preserves_ties_and_freezing() {
    let module = TiedFrozenModule::new([1.0, -1.0]);
    let mut compiled =
        CpuCompiledAdamW::compile_module(module_config(), &module, build_tied_frozen).unwrap();
    assert_eq!(
        compiled
            .parameter_snapshots()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["shared"]
    );

    // Host state is only an initialization source. Once compiled, the
    // recurrent runtime is the sole owner of the trainable value.
    module
        .shared
        .replace(TensorData::new([2], vec![9.0, 9.0]).unwrap())
        .unwrap();
    let input = BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
    let first = compiled
        .step(input.clone(), TensorData::scalar(0.01))
        .unwrap();
    assert_eq!(first.step(), 1);
    assert_eq!(compiled.optimizer_step().unwrap(), 1);
    assert_eq!(compiled.parameter_versions().unwrap()["shared"], 1);

    let runtime_checkpoint = compiled.checkpoint().unwrap();
    let runtime_progress = (compiled.step_count(), compiled.optimizer_step().unwrap());
    let published = compiled.parameter_snapshots().unwrap();
    let host_version = module.shared.version().unwrap();
    let frozen = module.frozen.snapshot().unwrap();
    assert!(compiled.publish_parameters(&module).unwrap().is_clean());
    assert_eq!(module.shared.value().unwrap(), published["shared"]);
    assert_eq!(module.shared.version().unwrap(), host_version + 1);
    assert_eq!(module.frozen.snapshot().unwrap().data, frozen.data);
    assert_eq!(module.frozen.version().unwrap(), frozen.version);
    assert_eq!(compiled.checkpoint().unwrap(), runtime_checkpoint);
    assert_eq!(
        (compiled.step_count(), compiled.optimizer_step().unwrap()),
        runtime_progress
    );

    let checkpoint = compiled.checkpoint().unwrap();
    let resumed_module = TiedFrozenModule::new([1.0, -1.0]);
    let mut resumed = CpuCompiledAdamW::compile_module_from_checkpoint(
        module_config(),
        &resumed_module,
        &checkpoint,
        build_tied_frozen,
    )
    .unwrap();
    assert_eq!(
        resumed.parameter_snapshots().unwrap(),
        compiled.parameter_snapshots().unwrap()
    );
    assert_eq!(
        resumed.first_moment_snapshots().unwrap(),
        compiled.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        resumed.second_moment_snapshots().unwrap(),
        compiled.second_moment_snapshots().unwrap()
    );
    assert_eq!(resumed.step_count(), 1);
    assert_eq!(
        resumed
            .step(input.clone(), TensorData::scalar(0.01))
            .unwrap()
            .step(),
        2
    );

    let accumulation_module = TiedFrozenModule::new([1.0, -1.0]);
    let mut accumulated = CpuCompiledAdamW::compile_module(
        module_config().with_gradient_accumulation(2).unwrap(),
        &accumulation_module,
        build_tied_frozen,
    )
    .unwrap();
    let capture_identity = accumulated.capture_identity();
    let parameter = accumulated.parameter_snapshots().unwrap();
    accumulated.step(input, TensorData::scalar(0.01)).unwrap();
    assert_eq!(accumulated.parameter_snapshots().unwrap(), parameter);
    assert_eq!(
        accumulated
            .gradient_accumulator_snapshots()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["shared"]
    );
    assert!(accumulated.zero_grad().unwrap().did_discard());
    assert_eq!(accumulated.capture_identity(), capture_identity);
    assert_eq!(accumulated.parameter_snapshots().unwrap(), parameter);
    assert!(
        accumulated
            .gradient_accumulator_snapshots()
            .unwrap()
            .values()
            .all(|value| value
                == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32).unwrap())
    );

    let changed_frozen = TiedFrozenModule::new([2.0, -1.0]);
    assert!(
        CpuCompiledAdamW::compile_module_from_checkpoint(
            module_config(),
            &changed_frozen,
            &checkpoint,
            build_tied_frozen,
        )
        .is_err()
    );
}

#[test]
fn adamw_compile_time_freezing_is_canonical_tied_and_raw_fail_closed() {
    let config = module_config().with_frozen_parameters(["base"]).unwrap();
    assert_eq!(config.frozen_parameters().collect::<Vec<_>>(), ["base"]);
    assert!(config.clone().with_frozen_parameters(["base"]).is_err());
    let raw_parameter =
        TrainingParameterInit::new("adapter", TensorData::zeros([2]).unwrap()).unwrap();
    assert!(
        CompiledAdamWPlan::compile(config.clone(), [raw_parameter], |_, _, _| panic!(
            "raw frozen-name policy reached graph construction"
        ),)
        .is_err()
    );

    for name in ["base_alias", "frozen", "missing"] {
        let module = FineTuneModule::new();
        let invalid = module_config().with_frozen_parameters([name]).unwrap();
        assert!(
            CompiledAdamWPlan::compile_module(invalid, &module, |_, _, _| {
                panic!("invalid frozen-name policy reached graph construction")
            })
            .is_err()
        );
    }
    let buffer = TiedFrozenModule::new([1.0, -1.0]);
    assert!(
        CompiledAdamWPlan::compile_module(
            module_config().with_frozen_parameters(["buffer"]).unwrap(),
            &buffer,
            build_tied_frozen,
        )
        .is_err()
    );
    let all_frozen = TiedFrozenModule::new([1.0, -1.0]);
    assert!(
        CompiledAdamWPlan::compile_module(
            module_config().with_frozen_parameters(["shared"]).unwrap(),
            &all_frozen,
            build_tied_frozen,
        )
        .is_err()
    );
    let overlap = FineTuneModule::new();
    assert!(
        CompiledAdamWPlan::compile_module(
            config
                .clone()
                .with_weight_decay_exclusions(["base"])
                .unwrap(),
            &overlap,
            build_fine_tune,
        )
        .is_err()
    );
}

#[test]
fn parameter_override_tracks_source_and_effective_trainability_separately() {
    let parameter = Parameter::new(TensorData::new([2], vec![1.0, 2.0]).unwrap(), true);
    let mut graph = Graph::new();
    let frozen = graph.constant(parameter.value().unwrap());
    assert!(
        graph
            .with_parameter_overrides(
                BTreeMap::from([(parameter.id(), (frozen, false, false))]),
                |graph| parameter.bind(graph),
            )
            .is_err(),
        "effective freezing must not weaken source-trainability authentication"
    );
    let bound = graph
        .with_parameter_overrides(
            BTreeMap::from([(parameter.id(), (frozen, true, false))]),
            |graph| parameter.bind(graph),
        )
        .unwrap();
    assert_eq!(bound, frozen);
    assert!(!graph.requires_grad(bound).unwrap());
    assert!(parameter.is_trainable());
}

#[test]
fn adamw_compile_time_freezing_reduces_state_and_resumes_accumulation_exactly() {
    let config = module_config()
        .with_gradient_accumulation(3)
        .unwrap()
        .with_frozen_parameters(["base"])
        .unwrap();
    let module = FineTuneModule::new();
    let base = module.base.snapshot().unwrap();
    let adapter = module.adapter.snapshot().unwrap();
    let plan = CompiledAdamWPlan::compile_module(config.clone(), &module, build_fine_tune).unwrap();
    let expected_parameter_keys = vec!["adapter".to_owned()];
    assert_eq!(
        plan.inner
            .state_values
            .keys()
            .filter_map(RecurrentStateKey::parameter_name)
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        expected_parameter_keys
    );
    assert!(module.base.is_trainable());
    assert!(module.adapter.is_trainable());
    assert_parameter_snapshot_eq(&module.base.snapshot().unwrap(), &base);
    assert_parameter_snapshot_eq(&module.adapter.snapshot().unwrap(), &adapter);

    let metal = plan
        .metal_plan(
            MetalRenderer::new(
                8,
                crate::runtime::metal::MetalCapabilities {
                    max_buffer_length: 1 << 30,
                    unified_memory: true,
                    family: "Apple9".into(),
                },
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        metal
            .inner
            .state_input_keys
            .values()
            .filter_map(RecurrentStateKey::parameter_name)
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        expected_parameter_keys
    );

    let input = || BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
    let mut uninterrupted = plan.prepare_cpu().unwrap();
    uninterrupted
        .step(input(), TensorData::scalar(0.01))
        .unwrap();
    let checkpoint = uninterrupted.checkpoint().unwrap();
    let decoded = decode_adamw_checkpoint(checkpoint.as_bytes()).unwrap();
    assert_eq!(
        decoded.parameters.keys().cloned().collect::<Vec<_>>(),
        expected_parameter_keys
    );
    assert_eq!(
        decoded.first_moments.keys().cloned().collect::<Vec<_>>(),
        expected_parameter_keys
    );
    assert_eq!(
        decoded.second_moments.keys().cloned().collect::<Vec<_>>(),
        expected_parameter_keys
    );
    assert_eq!(
        decoded
            .gradient_accumulators
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        expected_parameter_keys
    );

    let fresh = FineTuneModule::new();
    let mut resumed = CompiledAdamWPlan::compile_module_from_checkpoint(
        config,
        &fresh,
        &checkpoint,
        build_fine_tune,
    )
    .unwrap()
    .prepare_cpu()
    .unwrap();
    assert!(uninterrupted.zero_grad().unwrap().did_discard());
    assert!(resumed.zero_grad().unwrap().did_discard());
    uninterrupted
        .step(input(), TensorData::scalar(0.01))
        .unwrap();
    resumed.step(input(), TensorData::scalar(0.01)).unwrap();
    assert!(
        uninterrupted
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert!(
        resumed
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap()
            .did_update()
    );
    assert_eq!(
        uninterrupted.checkpoint().unwrap(),
        resumed.checkpoint().unwrap()
    );
    let publication = FineTuneModule::new();
    let publication_base = publication.base.snapshot().unwrap();
    let publication_adapter_version = publication.adapter.version().unwrap();
    assert!(
        uninterrupted
            .publish_parameters(&publication)
            .unwrap()
            .is_clean()
    );
    assert_parameter_snapshot_eq(&publication.base.snapshot().unwrap(), &publication_base);
    assert_eq!(
        publication.adapter.value().unwrap(),
        uninterrupted.parameter_snapshots().unwrap()["adapter"]
    );
    assert_eq!(
        publication.adapter.version().unwrap(),
        publication_adapter_version + 1
    );
    assert_parameter_snapshot_eq(&module.base.snapshot().unwrap(), &base);
    assert_parameter_snapshot_eq(&module.adapter.snapshot().unwrap(), &adapter);
}

#[test]
fn adamw_compile_time_freezing_excludes_tied_gradients_from_global_clipping() {
    let clipped = module_config().with_max_gradient_norm(1.0).unwrap();
    let frozen_config = clipped.clone().with_frozen_parameters(["base"]).unwrap();
    let frozen_module = FineTuneModule::new();
    let frozen_plan =
        CompiledAdamWPlan::compile_module(frozen_config, &frozen_module, build_fine_tune_clip)
            .unwrap();
    let metal = frozen_plan
        .metal_plan(
            MetalRenderer::new(
                8,
                crate::runtime::metal::MetalCapabilities {
                    max_buffer_length: 1 << 30,
                    unified_memory: true,
                    family: "Apple9".into(),
                },
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        metal
            .inner
            .state_input_keys
            .values()
            .filter_map(RecurrentStateKey::parameter_name)
            .collect::<Vec<_>>(),
        ["adapter"]
    );

    let input = || {
        BTreeMap::from([(
            "x".into(),
            TensorData::new([2], vec![1_000.0, -1_000.0]).unwrap(),
        )])
    };
    let mut frozen = frozen_plan.prepare_cpu().unwrap();
    frozen.step(input(), TensorData::scalar(0.01)).unwrap();
    let frozen_moment = frozen.first_moment_snapshots().unwrap()["adapter"].to_vec_f64();
    assert!(
        frozen_moment
            .iter()
            .all(|value| (*value - 0.05).abs() < 1e-6)
    );

    let unfrozen_module = FineTuneModule::new();
    let mut unfrozen =
        CompiledAdamWPlan::compile_module(clipped, &unfrozen_module, build_fine_tune_clip)
            .unwrap()
            .prepare_cpu()
            .unwrap();
    unfrozen.step(input(), TensorData::scalar(0.01)).unwrap();
    let unfrozen_moment = unfrozen.first_moment_snapshots().unwrap()["adapter"].to_vec_f64();
    assert!(unfrozen_moment.iter().all(|value| value.abs() < 1e-3));
    assert_ne!(frozen_moment, unfrozen_moment);
}

#[test]
fn owned_adamw_freezing_publishes_only_the_unfrozen_frontier() {
    let module = FineTuneModule::new();
    let base = module.base.clone();
    let adapter = module.adapter.clone();
    let base_before = base.snapshot().unwrap();
    let adapter_before = adapter.snapshot().unwrap();
    let plan = CompiledModuleAdamWPlan::compile(
        module_config().with_frozen_parameters(["base"]).unwrap(),
        module,
        build_fine_tune,
    )
    .unwrap();
    let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
    session
        .step(
            BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
            TensorData::scalar(0.01),
        )
        .unwrap();
    assert_eq!(
        session
            .parameter_snapshots()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["adapter"]
    );
    let module = session.finish().unwrap();
    assert_parameter_snapshot_eq(&module.base.snapshot().unwrap(), &base_before);
    assert_eq!(module.base.id(), base.id());
    assert!(module.base.is_trainable());
    assert_ne!(module.adapter.value().unwrap(), adapter_before.data);
    assert_eq!(
        module.adapter.version().unwrap(),
        adapter_before.version + 1
    );
    assert_eq!(module.adapter.id(), adapter.id());
    assert!(module.adapter.is_trainable());
}

#[test]
fn adamw_config_rejects_invalid_hyperparameters_before_build() {
    for config in [
        CompiledAdamWConfig::new(-0.1, 0.999, 1e-8, 0.0),
        CompiledAdamWConfig::new(1.0, 0.999, 1e-8, 0.0),
        CompiledAdamWConfig::new(0.9, 1.0, 1e-8, 0.0),
        CompiledAdamWConfig::new(0.9, 0.999, 0.0, 0.0),
        CompiledAdamWConfig::new(0.9, 0.999, f32::NAN, 0.0),
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, -0.1),
    ] {
        assert!(config.is_err());
    }
    assert!(
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(0)
            .is_err()
    );
    for max_norm in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
                .unwrap()
                .with_max_gradient_norm(max_norm)
                .is_err()
        );
    }
    for loss_scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
                .unwrap()
                .with_loss_scale(loss_scale)
                .is_err()
        );
    }
}

#[test]
fn adamw_checkpoint_resume_matches_uninterrupted_replay_exactly() {
    let mut uninterrupted = compiled_adamw();
    let mut saved = compiled_adamw();
    for _ in 0..2 {
        uninterrupted.step(batch(), lr()).unwrap();
        saved.step(batch(), lr()).unwrap();
    }
    let checkpoint = saved.checkpoint().unwrap();
    assert_eq!(checkpoint, saved.checkpoint().unwrap());
    assert_eq!(
        CompiledAdamWCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
        checkpoint
    );

    let mut resumed =
        CpuCompiledAdamW::compile_from_checkpoint(adamw_config(), &checkpoint, build_tinybob)
            .unwrap();
    assert_eq!(resumed.step_count(), 2);
    assert_eq!(resumed.optimizer_step().unwrap(), 2);
    assert_eq!(resumed.capture_identity(), saved.capture_identity());
    assert_eq!(
        resumed.parameter_snapshots().unwrap(),
        saved.parameter_snapshots().unwrap()
    );
    assert_eq!(
        resumed.first_moment_snapshots().unwrap(),
        saved.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        resumed.second_moment_snapshots().unwrap(),
        saved.second_moment_snapshots().unwrap()
    );
    assert_eq!(
        resumed.parameter_versions().unwrap(),
        saved.parameter_versions().unwrap()
    );
    assert_eq!(
        resumed.first_moment_versions().unwrap(),
        saved.first_moment_versions().unwrap()
    );
    assert_eq!(
        resumed.second_moment_versions().unwrap(),
        saved.second_moment_versions().unwrap()
    );

    let expected = uninterrupted.step(batch(), lr()).unwrap();
    let actual = resumed.step(batch(), lr()).unwrap();
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());
    assert_eq!(actual.step(), expected.step());
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
        resumed.parameter_versions().unwrap(),
        uninterrupted.parameter_versions().unwrap()
    );
}

#[test]
fn adamw_checkpoint_rejects_corruption_and_wrong_program_identity() {
    let checkpoint = compiled_adamw().checkpoint().unwrap();
    let mut corrupt = checkpoint.as_bytes().to_vec();
    corrupt.pop();
    assert!(CompiledAdamWCheckpoint::from_bytes(corrupt).is_err());

    let wrong = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.02)
        .unwrap()
        .with_input("x", [4, 2], DType::F32)
        .unwrap()
        .with_input("target", [4], DType::I64)
        .unwrap();
    let wrong_checkpoint =
        CompiledAdamWPlan::compile(wrong.clone(), initial_parameters(), build_tinybob)
            .unwrap()
            .prepare_cpu()
            .unwrap()
            .checkpoint()
            .unwrap();
    assert!(CpuCompiledAdamW::compile_from_checkpoint(wrong, &checkpoint, build_tinybob).is_err());
    let mut runtime = compiled_adamw();
    let before = runtime.checkpoint().unwrap();
    assert!(
        runtime
            .restore_checkpoint_in_place(&wrong_checkpoint)
            .is_err()
    );
    assert_eq!(runtime.checkpoint().unwrap(), before);
    runtime.restore_checkpoint_in_place(&before).unwrap();
    assert_eq!(runtime.checkpoint().unwrap(), before);
    let clipped = adamw_config().with_max_gradient_norm(1.0).unwrap();
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(clipped, &checkpoint, build_tinybob).is_err()
    );
    let scaled = adamw_config().with_loss_scale(128.0).unwrap();
    assert!(CpuCompiledAdamW::compile_from_checkpoint(scaled, &checkpoint, build_tinybob).is_err());

    let mut accumulated = CpuCompiledAdamW::compile(
        accumulated_adamw_config(2),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    accumulated.step(batch(), lr()).unwrap();
    let checkpoint = accumulated.checkpoint().unwrap();
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(adamw_config(), &checkpoint, build_tinybob,)
            .is_err()
    );
}

#[test]
fn compiled_multi_step_lr_validates_immutable_schedule() {
    let schedule = CompiledMultiStepLr::new(0.05, 0.25, [1, 3, 8]).unwrap();
    assert_eq!(schedule.base(), 0.05);
    assert_eq!(schedule.gamma(), 0.25);
    assert_eq!(schedule.milestones(), &[1, 3, 8]);
    assert!(CompiledMultiStepLr::new(f32::NAN, 0.5, []).is_err());
    assert!(CompiledMultiStepLr::new(-0.1, 0.5, []).is_err());
    assert!(CompiledMultiStepLr::new(0.1, f32::INFINITY, []).is_err());
    assert!(CompiledMultiStepLr::new(0.1, -0.5, []).is_err());
    assert!(CompiledMultiStepLr::new(0.1, 0.5, [0]).is_err());
    assert!(CompiledMultiStepLr::new(0.1, 0.5, [1, 1]).is_err());
    assert!(CompiledMultiStepLr::new(0.1, 0.5, [2, 1]).is_err());
    assert!(CompiledMultiStepLr::new(0.1, 0.5, [u64::MAX]).is_err());
    assert!(CompiledMultiStepLr::new(f32::MAX, 2.0, [1]).is_err());
    assert!(CompiledMultiStepLr::new(f32::MAX / 2.0, 1.5, [1, 2]).is_err());
}

#[test]
fn compiled_multi_step_lr_matches_external_updates_flush_and_restore() {
    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1, 2]).unwrap();
    let scheduled_config =
        accumulated_adamw_config(2).with_captured_multi_step_lr(schedule.clone());
    assert_eq!(scheduled_config.captured_multi_step_lr(), Some(&schedule));
    let mut scheduled = CpuCompiledAdamW::compile(
        scheduled_config.clone(),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    let mut external = CpuCompiledAdamW::compile(
        accumulated_adamw_config(2),
        initial_parameters(),
        build_tinybob,
    )
    .unwrap();
    assert_eq!(scheduled.captured_multi_step_lr(), Some(&schedule));
    assert_eq!(external.captured_multi_step_lr(), None);
    assert_ne!(scheduled.capture_identity(), external.capture_identity());
    assert!(
        !scheduled
            .inner
            .capture
            .schedule
            .inputs
            .iter()
            .any(|input| input.name == LEARNING_RATE_INPUT)
    );
    assert!(
        external
            .inner
            .capture
            .schedule
            .inputs
            .iter()
            .any(|input| input.name == LEARNING_RATE_INPUT)
    );
    assert!(
        !scheduled
            .partial_flush
            .as_ref()
            .unwrap()
            .phase()
            .capture
            .schedule
            .inputs
            .iter()
            .any(|input| input.name == LEARNING_RATE_INPUT)
    );
    assert!(
        external
            .partial_flush
            .as_ref()
            .unwrap()
            .phase()
            .capture
            .schedule
            .inputs
            .iter()
            .any(|input| input.name == LEARNING_RATE_INPUT)
    );

    let scheduled_before = scheduled.checkpoint().unwrap();
    assert!(scheduled.step(batch(), TensorData::scalar(0.05)).is_err());
    assert_eq!(scheduled.checkpoint().unwrap(), scheduled_before);
    let external_before = external.checkpoint().unwrap();
    assert!(
        CompiledTrainingRatePolicyRuntime::step_with_rate_policy(&mut external, batch()).is_err()
    );
    assert_eq!(external.checkpoint().unwrap(), external_before);

    for external_rate in [0.05, 0.05, 0.025] {
        let actual = run_core_rate_policy_step(&mut scheduled);
        let expected = external
            .step(batch(), TensorData::scalar(external_rate))
            .unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.did_update(), expected.did_update());
    }
    let scheduled_before_wrong_flush = scheduled.checkpoint().unwrap();
    assert!(
        CompiledTrainingWindowCommitRuntime::commit_partial_window(
            &mut scheduled,
            TensorData::scalar(0.025),
        )
        .is_err()
    );
    assert_eq!(
        scheduled.checkpoint().unwrap(),
        scheduled_before_wrong_flush
    );
    let actual = commit_core_rate_policy_training_window(&mut scheduled);
    let expected = CompiledTrainingWindowCommitRuntime::commit_partial_window(
        &mut external,
        TensorData::scalar(0.025),
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(scheduled.checkpoint().unwrap().info().optimizer_step(), 2);
    assert_eq!(
        scheduled.parameter_snapshots().unwrap(),
        external.parameter_snapshots().unwrap()
    );
    assert_eq!(
        scheduled.first_moment_snapshots().unwrap(),
        external.first_moment_snapshots().unwrap()
    );
    assert_eq!(
        scheduled.second_moment_snapshots().unwrap(),
        external.second_moment_snapshots().unwrap()
    );

    let checkpoint = scheduled.checkpoint().unwrap();
    let mut resumed = CpuCompiledAdamW::compile_from_checkpoint(
        scheduled_config.clone(),
        &checkpoint,
        build_tinybob,
    )
    .unwrap();
    for _ in 0..2 {
        let expected = run_core_rate_policy_step(&mut scheduled);
        let actual = run_core_rate_policy_step(&mut resumed);
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
    }
    assert_eq!(
        resumed.checkpoint().unwrap(),
        scheduled.checkpoint().unwrap()
    );
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(
            accumulated_adamw_config(2),
            &checkpoint,
            build_tinybob,
        )
        .is_err()
    );
    let wrong_schedule = accumulated_adamw_config(2)
        .with_captured_multi_step_lr(CompiledMultiStepLr::new(0.05, 0.5, [1, 3]).unwrap());
    assert!(
        CpuCompiledAdamW::compile_from_checkpoint(wrong_schedule, &checkpoint, build_tinybob)
            .is_err()
    );

    let scheduled_plan =
        CompiledAdamWPlan::compile(scheduled_config, initial_parameters(), build_tinybob).unwrap();
    assert_eq!(scheduled_plan.captured_multi_step_lr(), Some(&schedule));
    let mut observed = scheduled_plan.prepare_cpu().unwrap();
    let mut commit_only = scheduled_plan.prepare_cpu().unwrap();
    let before_rejected = commit_only.checkpoint().unwrap();
    assert!(
        commit_only
            .commit_step_batch_with_rate_policy(RejectedBatch)
            .is_err()
    );
    assert_eq!(commit_only.checkpoint().unwrap(), before_rejected);
    let expected = run_core_rate_policy_step(&mut observed);
    let actual = run_core_rate_policy_commit_only_step(&mut commit_only);
    assert_eq!(actual.loss(), expected.loss());
    assert!(actual.outputs().is_empty());
    assert_eq!(
        commit_only.checkpoint().unwrap(),
        observed.checkpoint().unwrap()
    );
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor);
    let mut observed = target.prepare(&scheduled_plan).unwrap();
    let mut commit_only = target.prepare(&scheduled_plan).unwrap();
    let before_rejected = commit_only.checkpoint().unwrap();
    assert!(
        commit_only
            .commit_step_batch_with_rate_policy(RejectedBatch)
            .is_err()
    );
    assert_eq!(commit_only.checkpoint().unwrap(), before_rejected);
    let expected = run_core_rate_policy_step(&mut observed);
    let actual = run_core_rate_policy_commit_only_step(&mut commit_only);
    assert_eq!(actual.loss(), expected.loss());
    assert!(actual.outputs().is_empty());
    assert_eq!(
        commit_only.checkpoint().unwrap(),
        observed.checkpoint().unwrap()
    );
    let renderer = MetalRenderer::new(
        8,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: 1 << 30,
            unified_memory: true,
            family: "Apple9".into(),
        },
    )
    .unwrap();
    assert!(scheduled_plan.metal_plan(renderer).is_err());
}
