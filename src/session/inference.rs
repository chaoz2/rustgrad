//! Fresh-graph static CPU module inference.
use crate::nn::{ModuleForward, Parameter};
use crate::{
    Backend, CapturedReplayExecutor, CapturedReplayTrace, CapturedSchedule, CompileTrace,
    CpuBackend, DType, Error, ExecutionPlanSummary, Graph, Result, Schedule, TensorData, schedule,
};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
#[derive(Clone, Debug)]
pub struct ModuleInferenceResult {
    output: TensorData,
    trace: CompileTrace,
    parameter_versions: BTreeMap<String, u64>,
}
#[derive(Clone, Debug)]
pub struct NativeModuleInferenceResult {
    output: TensorData,
    trace: CapturedReplayTrace,
    parameter_versions: BTreeMap<String, u64>,
    native_trace: NativeModuleInferenceTrace,
}

/// Immutable, opt-in local observations for one strict native module call.
///
/// Durations are current-thread wall-clock observations, not stable benchmarks
/// or hardware, allocator, RSS, device-memory, or per-kernel measurements.
/// They are deliberately excluded from `identity`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModuleExecutionReport {
    /// Deterministic identity of the static plan and native policy, excluding
    /// local durations and current cache-hit state.
    pub identity: u64,
    /// Canonical, non-executing logical schedule/memory facts. Strict native
    /// replay does not claim to consume this host allocation plan, so reuse is
    /// intentionally disabled in this summary.
    pub execution_plan: ExecutionPlanSummary,
    pub capture_identity: u64,
    pub native_trace_identity: u64,
    pub vectorized: bool,
    pub native_cache_keys: Vec<Option<String>>,
    pub graph_schedule_capture_duration: Duration,
    pub native_prepare_duration: Duration,
    pub native_execute_duration: Duration,
    pub native_item_count: usize,
    pub zero_pruned_item_count: usize,
    pub zero_materialized_item_count: usize,
    pub cache_hit_count: usize,
    pub cache_miss_count: usize,
}

/// The existing detached strict-native result plus opt-in execution
/// observations. The standard inference API intentionally does not construct
/// this report or measure durations.
#[derive(Clone, Debug)]
pub struct ReportedNativeModuleInferenceResult {
    inference: NativeModuleInferenceResult,
    report: NativeModuleExecutionReport,
}

impl ReportedNativeModuleInferenceResult {
    pub fn inference(&self) -> &NativeModuleInferenceResult {
        &self.inference
    }

    pub fn report(&self) -> &NativeModuleExecutionReport {
        &self.report
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModuleInferenceTrace {
    pub identity: u64,
    pub capture_identity: u64,
    pub input_shape: crate::Shape,
    pub input_dtype: DType,
    pub parameter_versions: BTreeMap<String, u64>,
    pub vectorized: bool,
    pub renderer_version: &'static str,
    pub native_cache_keys: Vec<Option<String>>,
}
impl NativeModuleInferenceResult {
    pub fn output(&self) -> &TensorData {
        &self.output
    }
    pub fn trace(&self) -> &CapturedReplayTrace {
        &self.trace
    }
    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }
    pub fn native_trace(&self) -> &NativeModuleInferenceTrace {
        &self.native_trace
    }
}
impl ModuleInferenceResult {
    pub fn output(&self) -> &TensorData {
        &self.output
    }
    pub fn trace(&self) -> &CompileTrace {
        &self.trace
    }
    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }
}

struct NativeModuleInferenceSetup {
    output: crate::NodeId,
    scheduled: Schedule,
    capture: CapturedSchedule,
    bindings: BTreeMap<String, TensorData>,
    input_shape: crate::Shape,
    parameters: Vec<(String, Parameter)>,
}

fn prepare_native_module_inference(
    module: &impl ModuleForward,
    input: TensorData,
) -> Result<NativeModuleInferenceSetup> {
    if input.dtype() != DType::F32 {
        return Err(Error::SessionTraining {
            reason: "module native CPU inference input must have dtype F32".into(),
        });
    }
    let parameters = module.trainable_parameters()?;
    let mut graph = Graph::new();
    let node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
    let output = module.forward(&mut graph, node)?;
    let mut bindings = module.input_bindings(&graph)?;
    let input_shape = input.shape().clone();
    bindings.insert("module_input".into(), input);
    let bindings = bindings.into_iter().collect::<BTreeMap<_, _>>();
    let scheduled = schedule(&graph, output).map_err(|error| Error::SessionTraining {
        reason: error.to_string(),
    })?;
    let capture = CapturedSchedule::capture(&graph, &scheduled, &[output]).map_err(|error| {
        Error::SessionTraining {
            reason: error.to_string(),
        }
    })?;
    Ok(NativeModuleInferenceSetup {
        output,
        scheduled,
        capture,
        bindings,
        input_shape,
        parameters,
    })
}

fn finish_native_module_inference(
    setup: NativeModuleInferenceSetup,
    replay: crate::CapturedReplayResult,
    vectorized: bool,
) -> Result<NativeModuleInferenceResult> {
    let parameter_versions: BTreeMap<String, u64> = setup
        .parameters
        .into_iter()
        .map(|(name, parameter)| Ok((name, parameter.version()?)))
        .collect::<Result<_>>()?;
    let native_cache_keys = replay
        .trace
        .items
        .iter()
        .map(|item| item.native_cache_key.clone())
        .collect::<Vec<_>>();
    let mut bytes = format!(
        "{}:{:?}:{}:{:?}",
        setup.capture.identity, setup.input_shape, vectorized, parameter_versions
    )
    .into_bytes();
    for key in &native_cache_keys {
        bytes.extend_from_slice(key.as_deref().unwrap_or("").as_bytes());
    }
    let identity = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    Ok(NativeModuleInferenceResult {
        output: replay
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::SessionTraining {
                reason: "native inference missing output".into(),
            })?,
        trace: replay.trace,
        parameter_versions: parameter_versions.clone(),
        native_trace: NativeModuleInferenceTrace {
            identity,
            capture_identity: setup.capture.identity,
            input_shape: setup.input_shape,
            input_dtype: DType::F32,
            parameter_versions,
            vectorized,
            renderer_version: crate::cpu_jit::RENDERER_VERSION,
            native_cache_keys,
        },
    })
}
/// Builds and discards one fresh CPU graph for a one-input static module.
pub fn infer_module_cpu(
    module: &impl ModuleForward,
    input: TensorData,
) -> Result<ModuleInferenceResult> {
    if !module.accepts_input_dtype(input.dtype()) {
        return Err(Error::SessionTraining {
            reason: "module CPU inference input dtype is not accepted by the leading module".into(),
        });
    }
    let parameters = module.trainable_parameters()?;
    let mut graph = Graph::new();
    let node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
    let output = module.forward(&mut graph, node)?;
    let mut bindings = module.input_bindings(&graph)?;
    bindings.insert("module_input".into(), input);
    let value = CpuBackend.execute(&graph, output, &bindings)?;
    let parameter_versions: BTreeMap<String, u64> = parameters
        .into_iter()
        .map(|(n, p)| Ok((n, p.version()?)))
        .collect::<Result<_>>()?;
    Ok(ModuleInferenceResult {
        output: value,
        trace: graph.trace(output)?,
        parameter_versions,
    })
}

/// Fresh-graph strict native CPU inference. The caller owns the executor and
/// therefore its deterministic compilation cache; unsupported graphs fail
/// before a native item is executed and never fall back to interpretation.
pub fn infer_module_native_cpu(
    module: &impl ModuleForward,
    input: TensorData,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<NativeModuleInferenceResult> {
    let setup = prepare_native_module_inference(module, input)?;
    let replay = executor
        .replay_pruned_native(&setup.capture, &setup.bindings, vectorized)
        .map_err(|error| Error::SessionTraining {
            reason: error.to_string(),
        })?;
    finish_native_module_inference(setup, replay, vectorized)
}

/// Fresh-graph strict native CPU inference with explicit local timing and
/// structural planning observations. This is not a benchmark or profiler.
pub fn infer_module_native_cpu_with_report(
    module: &impl ModuleForward,
    input: TensorData,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<ReportedNativeModuleInferenceResult> {
    let graph_capture_start = Instant::now();
    let setup = prepare_native_module_inference(module, input)?;
    let graph_schedule_capture_duration = graph_capture_start.elapsed();
    let execution_plan =
        ExecutionPlanSummary::from_schedule(&setup.scheduled, &[setup.output], false).map_err(
            |error| Error::SessionTraining {
                reason: format!("native execution report summary: {error}"),
            },
        )?;

    let prepare_start = Instant::now();
    let prepared = executor
        .prepare_pruned_native(&setup.capture, &setup.bindings, vectorized)
        .map_err(|error| Error::SessionTraining {
            reason: error.to_string(),
        })?;
    let native_prepare_duration = prepare_start.elapsed();
    let zero_pruned_item_count = prepared.zero_pruned_item_count();
    let zero_materialized_item_count = prepared.zero_materialized_item_count();

    let execute_start = Instant::now();
    let replay = executor
        .execute_prepared_pruned_native(&setup.capture, &setup.bindings, &prepared)
        .map_err(|error| Error::SessionTraining {
            reason: error.to_string(),
        })?;
    let native_execute_duration = execute_start.elapsed();
    let inference = finish_native_module_inference(setup, replay, vectorized)?;
    let native_item_count = inference.trace.items.len();
    let cache_items = inference
        .trace
        .items
        .iter()
        .filter(|item| item.native_cache_key.is_some())
        .collect::<Vec<_>>();
    let cache_hit_count = cache_items.iter().filter(|item| item.cache_hit).count();
    let cache_miss_count = cache_items.iter().filter(|item| !item.cache_hit).count();
    let native_cache_keys = inference.native_trace.native_cache_keys.clone();
    let mut bytes = format!(
        "{}:{}:{}:{:?}",
        execution_plan.identity, inference.native_trace.identity, vectorized, native_cache_keys
    )
    .into_bytes();
    bytes.extend_from_slice(&zero_pruned_item_count.to_le_bytes());
    bytes.extend_from_slice(&zero_materialized_item_count.to_le_bytes());
    let identity = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    let report = NativeModuleExecutionReport {
        identity,
        execution_plan,
        capture_identity: inference.native_trace.capture_identity,
        native_trace_identity: inference.native_trace.identity,
        vectorized,
        native_cache_keys,
        graph_schedule_capture_duration,
        native_prepare_duration,
        native_execute_duration,
        native_item_count,
        zero_pruned_item_count,
        zero_materialized_item_count,
        cache_hit_count,
        cache_miss_count,
    };
    Ok(ReportedNativeModuleInferenceResult { inference, report })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{
        AdaptiveAvgPool2d, Conv2d, Flatten, Linear, Module, ModuleForward, Parameter, ReLU,
        Sequential, StateKind,
    };
    use crate::{Conv2dOptions, NodeId, Scalar};

    struct DuplicateTraversal {
        first: Parameter,
        second: Parameter,
    }

    impl Module for DuplicateTraversal {
        fn visit(&self, _: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            visitor("weight".into(), &self.first, StateKind::Parameter);
            visitor("weight".into(), &self.second, StateKind::Parameter);
        }
    }

    impl ModuleForward for DuplicateTraversal {
        fn forward(&self, _: &mut Graph, input: NodeId) -> Result<NodeId> {
            Ok(input)
        }
    }

    struct UnsupportedLater;

    impl Module for UnsupportedLater {
        fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
    }

    impl ModuleForward for UnsupportedLater {
        fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
            let supported = graph.relu(input)?;
            let mask = graph.gt_scalar(supported, Scalar::F(0.0))?;
            graph.masked_select(supported, mask, 2, Scalar::F(0.0))
        }
    }

    fn relu_mlp() -> (Sequential, Parameter) {
        let first = Linear::new_static(2, 2, true, 41).unwrap();
        first
            .weight
            .replace(TensorData::new([2, 2], vec![1., -1., 0.5, 2.]).unwrap())
            .unwrap();
        first
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([2], vec![0.5, -1.]).unwrap())
            .unwrap();
        let second = Linear::new_static(2, 1, true, 42).unwrap();
        second
            .weight
            .replace(TensorData::new([1, 2], vec![3., -2.]).unwrap())
            .unwrap();
        second
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let output_weight = second.weight.clone();
        let mut model = Sequential::default();
        model.push(first);
        model.push(ReLU::new());
        model.push(second);
        (model, output_weight)
    }

    fn configured_cifar_classifier() -> (Sequential, Parameter) {
        let conv = Conv2d::new_static(3, 2, [1, 1], Conv2dOptions::default(), true, 81).unwrap();
        conv.weight
            .replace(TensorData::new([2, 3, 1, 1], vec![1., -1., 0.5, -0.5, 2., 1.]).unwrap())
            .unwrap();
        conv.bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([2], vec![0.25, -0.75]).unwrap())
            .unwrap();
        let linear = Linear::new_static(2, 2, true, 82).unwrap();
        linear
            .weight
            .replace(TensorData::new([2, 2], vec![1., -2., 0.5, 3.]).unwrap())
            .unwrap();
        linear
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([2], vec![0.5, -1.]).unwrap())
            .unwrap();
        let output_weight = linear.weight.clone();
        let mut model = Sequential::default();
        model.push(conv);
        model.push(ReLU::new());
        model.push(AdaptiveAvgPool2d::new([Some(1), Some(1)]));
        model.push(Flatten::new(1));
        model.push(linear);
        (model, output_weight)
    }

    #[test]
    fn strict_native_configured_cifar_matches_cpu_and_preserves_contracts() {
        let (model, output_weight) = configured_cifar_classifier();
        let input = TensorData::new(
            [2, 3, 2, 2],
            (1..=24).map(|value| value as f32 / 8.).collect(),
        )
        .unwrap();
        let original_state = model.state_dict().unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let cold =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, false).unwrap();
        let cache_len = executor.compile_cache_len(false);
        let warm =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, false).unwrap();
        assert_eq!(cold.inference().output(), cpu.output());
        assert_eq!(cold.inference().output(), warm.inference().output());
        assert_eq!(cold.report().identity, warm.report().identity);
        assert_eq!(cache_len, executor.compile_cache_len(false));
        assert_eq!(warm.report().cache_miss_count, 0);
        let vector =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, true).unwrap();
        assert_eq!(vector.inference().output(), cpu.output());
        assert!(vector.report().vectorized);
        assert_ne!(cold.report().identity, vector.report().identity);
        assert!(executor.compile_cache_len(true) > 0);
        assert!(
            cold.inference()
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
        assert!(
            cold.inference()
                .native_trace()
                .parameter_versions
                .keys()
                .eq(["0.bias", "0.weight", "4.bias", "4.weight"])
        );
        assert_eq!(model.state_dict().unwrap(), original_state);

        let wider = TensorData::new([3, 3, 2, 2], vec![0.25; 36]).unwrap();
        let wider_native =
            infer_module_native_cpu(&model, wider.clone(), &executor, false).unwrap();
        assert_eq!(
            wider_native.output(),
            infer_module_cpu(&model, wider).unwrap().output()
        );
        assert_ne!(
            cold.inference().native_trace().identity,
            wider_native.native_trace().identity
        );
        output_weight
            .replace(TensorData::new([2, 2], vec![2., -2., 0.5, 3.]).unwrap())
            .unwrap();
        let changed = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_ne!(
            cold.inference().native_trace().identity,
            changed.native_trace().identity
        );
        assert_ne!(cold.inference().output(), changed.output());

        let empty = TensorData::new([0, 3, 2, 2], Vec::<f32>::new()).unwrap();
        let empty_cpu = infer_module_cpu(&model, empty.clone()).unwrap();
        let before_empty = executor.compile_cache_len(false);
        let empty_native = infer_module_native_cpu(&model, empty, &executor, false).unwrap();
        assert_eq!(empty_native.output(), empty_cpu.output());
        assert_eq!(empty_native.output().shape().dims(), &[0, 2]);
        assert_eq!(before_empty, executor.compile_cache_len(false));
        assert!(
            empty_native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn strict_native_static_conv_contract_matches_cpu_without_mutation() {
        let model = Conv2d::new_static(3, 2, [3, 3], Conv2dOptions::default(), false, 91).unwrap();
        let before = model.state_dict().unwrap();
        let executor = CapturedReplayExecutor::default();
        let input = TensorData::new([1, 3, 3, 3], vec![1.0f32; 27]).unwrap();
        let expected = infer_module_cpu(&model, input.clone()).unwrap();
        let native = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_eq!(native.output(), expected.output());
        assert!(
            native
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
        let cached = executor.compile_cache_len(false);
        assert!(cached > 0);
        let replay = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(replay.output(), expected.output());
        assert_eq!(executor.compile_cache_len(false), cached);
        assert_eq!(model.state_dict().unwrap(), before);
    }

    #[test]
    fn inference_is_fresh_deterministic_and_nonmutating() {
        let model = Linear::new_static(2, 1, true, 1).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let before = model.state_dict().unwrap();
        let first = infer_module_cpu(&model, input.clone()).unwrap();
        let second = infer_module_cpu(&model, input.clone()).unwrap();
        assert_eq!(first.output().to_vec_f64(), vec![9., 19.]);
        assert_eq!(first.output(), second.output());
        assert_eq!(first.trace(), second.trace());
        assert_eq!(before, model.state_dict().unwrap());
        assert!(infer_module_cpu(&model, TensorData::new([1, 3], vec![0.; 3]).unwrap()).is_err());
        assert!(
            infer_module_cpu(
                &model,
                TensorData::from_scalars([1, 2], DType::F64, [crate::Scalar::F(0.); 2]).unwrap()
            )
            .is_err()
        );
        let empty =
            infer_module_cpu(&model, TensorData::new([0, 2], Vec::<f32>::new()).unwrap()).unwrap();
        assert_eq!(empty.output().shape().dims(), &[0, 1]);
    }

    #[test]
    fn strict_native_linear_matches_cpu_and_reuses_caller_cache() {
        let model = Linear::new_static(2, 1, true, 1).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let first = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(cpu.output(), first.output());
        assert_eq!(first.output(), second.output());
        assert_eq!(first.native_trace(), second.native_trace());
        assert!(
            first
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
        assert_eq!(cached, executor.compile_cache_len(false));
        model
            .weight
            .replace(TensorData::new([1, 2], vec![4., 3.]).unwrap())
            .unwrap();
        let changed = infer_module_native_cpu(
            &model,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            &executor,
            false,
        )
        .unwrap();
        assert_ne!(
            first.native_trace().identity,
            changed.native_trace().identity
        );
        assert_ne!(first.output(), changed.output());
        let vector = infer_module_native_cpu(
            &model,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            &executor,
            true,
        )
        .unwrap();
        assert_eq!(
            vector.output(),
            infer_module_cpu(
                &model,
                TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap()
            )
            .unwrap()
            .output()
        );
        assert!(vector.native_trace().vectorized);
        assert!(executor.compile_cache_len(true) > 0);
        let wider = infer_module_native_cpu(
            &model,
            TensorData::new([3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            &executor,
            false,
        )
        .unwrap();
        assert_ne!(
            changed.native_trace().identity,
            wider.native_trace().identity
        );
        assert_eq!(wider.output().shape().dims(), &[3, 1]);
    }

    #[test]
    fn opt_in_native_execution_report_correlates_with_warm_cache_and_static_plan() {
        let model = Linear::new_static(2, 1, true, 61).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let executor = CapturedReplayExecutor::default();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let original_input = input.clone();
        let original_state = model.state_dict().unwrap();

        let cold =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, false).unwrap();
        let cache_len = executor.compile_cache_len(false);
        let warm = infer_module_native_cpu_with_report(&model, input, &executor, false).unwrap();
        assert_eq!(cold.inference().output(), warm.inference().output());
        assert_eq!(cold.report().identity, warm.report().identity);
        assert_eq!(cold.report().execution_plan, warm.report().execution_plan);
        assert_eq!(cache_len, executor.compile_cache_len(false));
        assert_eq!(
            cold.report().native_cache_keys,
            cold.inference().native_trace().native_cache_keys
        );
        assert_eq!(
            cold.report().cache_hit_count + cold.report().cache_miss_count,
            cold.inference()
                .trace()
                .items
                .iter()
                .filter(|item| item.native_cache_key.is_some())
                .count()
        );
        assert_eq!(warm.report().cache_miss_count, 0);
        assert_eq!(
            warm.report().cache_hit_count,
            warm.inference()
                .trace()
                .items
                .iter()
                .filter(|item| item.native_cache_key.is_some())
                .count()
        );
        assert_eq!(
            cold.report().native_item_count,
            cold.inference().trace().items.len()
        );
        assert_eq!(cold.report().execution_plan.requested_outputs.len(), 1);
        let _ = (
            cold.report().graph_schedule_capture_duration,
            cold.report().native_prepare_duration,
            cold.report().native_execute_duration,
        );
        assert_eq!(
            original_input,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap()
        );
        assert_eq!(
            model.state_dict().unwrap().tensors(),
            original_state.tensors()
        );
    }

    #[test]
    fn strict_native_sequential_matches_cpu() {
        let mut model = Sequential::default();
        model.push(Linear::new_static(2, 2, true, 1).unwrap());
        model.push(Linear::new_static(2, 1, true, 2).unwrap());
        let input = TensorData::new([1, 2], vec![1., -2.]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let native = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(cpu.output(), native.output());
        assert!(
            native
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
    }

    #[test]
    fn strict_native_relu_mlp_matches_cpu_and_preserves_strict_contracts() {
        let (model, output_weight) = relu_mlp();
        let input = TensorData::new([2, 2], vec![1., -2., 3., 4.]).unwrap();
        let before = model.state_dict().unwrap();
        let executor = CapturedReplayExecutor::default();

        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let first = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        let scalar_cache = executor.compile_cache_len(false);
        let second = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_eq!(first.output(), cpu.output());
        assert_eq!(first.output(), second.output());
        assert_eq!(first.native_trace(), second.native_trace());
        assert_eq!(scalar_cache, executor.compile_cache_len(false));
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("0.weight")
        );
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("0.bias")
        );
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("2.weight")
        );
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("2.bias")
        );
        assert!(
            first
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );

        let vector = infer_module_native_cpu(&model, input.clone(), &executor, true).unwrap();
        assert_eq!(vector.output(), cpu.output());
        assert!(vector.native_trace().vectorized);
        assert!(executor.compile_cache_len(true) > 0);

        let wider = TensorData::new([3, 2], vec![1., -2., 3., 4., -1., 2.]).unwrap();
        let wider_cpu = infer_module_cpu(&model, wider.clone()).unwrap();
        let wider_native = infer_module_native_cpu(&model, wider, &executor, false).unwrap();
        assert_eq!(wider_native.output(), wider_cpu.output());
        assert_ne!(
            first.native_trace().identity,
            wider_native.native_trace().identity
        );

        output_weight
            .replace(TensorData::new([1, 2], vec![2., 1.]).unwrap())
            .unwrap();
        let changed = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_ne!(
            first.native_trace().identity,
            changed.native_trace().identity
        );
        assert_ne!(first.output(), changed.output());
        assert_eq!(
            model.state_dict().unwrap().tensors().len(),
            before.tensors().len()
        );

        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let empty_cpu = infer_module_cpu(&model, empty.clone()).unwrap();
        let before_empty_cache = executor.compile_cache_len(false);
        let empty_native = infer_module_native_cpu(&model, empty, &executor, false).unwrap();
        assert_eq!(empty_native.output(), empty_cpu.output());
        assert_eq!(empty_native.output().shape().dims(), &[0, 1]);
        assert_eq!(before_empty_cache, executor.compile_cache_len(false));
        assert!(
            empty_native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );

        assert!(
            infer_module_native_cpu(
                &model,
                TensorData::from_scalars([1, 2], DType::F64, [Scalar::F(0.); 2]).unwrap(),
                &executor,
                false,
            )
            .is_err()
        );
        assert!(
            infer_module_native_cpu(
                &model,
                TensorData::new([1, 3], vec![0.; 3]).unwrap(),
                &executor,
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn strict_native_module_rejects_later_unsupported_before_execution() {
        let executor = CapturedReplayExecutor::default();
        let before = executor.compile_cache_len(false);
        assert!(
            infer_module_native_cpu(
                &UnsupportedLater,
                TensorData::new([1, 2], vec![1., -1.]).unwrap(),
                &executor,
                false,
            )
            .is_err()
        );
        assert_eq!(before, executor.compile_cache_len(false));
    }

    #[test]
    fn strict_native_empty_modules_prune_dead_pure_work_without_native_cache_keys() {
        let linear = Linear::new_static(2, 1, true, 17).unwrap();
        let linear_executor = CapturedReplayExecutor::default();
        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let cpu = infer_module_cpu(&linear, empty.clone()).unwrap();
        let native = infer_module_native_cpu(&linear, empty, &linear_executor, false).unwrap();
        assert_eq!(native.output(), cpu.output());
        assert_eq!(native.output().shape().dims(), &[0, 1]);
        assert_eq!(linear_executor.compile_cache_len(false), 0);
        assert!(
            native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );

        let mut sequential = Sequential::default();
        sequential.push(Linear::new_static(2, 2, true, 18).unwrap());
        sequential.push(Linear::new_static(2, 1, true, 19).unwrap());
        let sequential_executor = CapturedReplayExecutor::default();
        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let cpu = infer_module_cpu(&sequential, empty.clone()).unwrap();
        let native =
            infer_module_native_cpu(&sequential, empty, &sequential_executor, false).unwrap();
        assert_eq!(native.output(), cpu.output());
        assert_eq!(native.output().shape().dims(), &[0, 1]);
        assert_eq!(sequential_executor.compile_cache_len(false), 0);
        assert!(
            native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn opt_in_report_keeps_empty_pruning_and_strict_preflight_honest() {
        let linear = Linear::new_static(2, 1, true, 62).unwrap();
        let executor = CapturedReplayExecutor::default();
        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let report = infer_module_native_cpu_with_report(&linear, empty, &executor, false).unwrap();
        assert_eq!(report.inference().output().shape().dims(), &[0, 1]);
        assert!(
            report
                .report()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );
        assert_eq!(report.report().cache_hit_count, 0);
        assert_eq!(report.report().cache_miss_count, 0);
        assert!(report.report().zero_materialized_item_count > 0);
        assert_eq!(executor.compile_cache_len(false), 0);

        let mut sequential = Sequential::default();
        sequential.push(Linear::new_static(2, 2, true, 63).unwrap());
        sequential.push(Linear::new_static(2, 1, true, 64).unwrap());
        let sequential_executor = CapturedReplayExecutor::default();
        let report = infer_module_native_cpu_with_report(
            &sequential,
            TensorData::new([0, 2], Vec::<f32>::new()).unwrap(),
            &sequential_executor,
            false,
        )
        .unwrap();
        assert_eq!(report.inference().output().shape().dims(), &[0, 1]);
        assert!(report.report().zero_pruned_item_count > 0);
        assert_eq!(sequential_executor.compile_cache_len(false), 0);

        let unsupported_executor = CapturedReplayExecutor::default();
        assert!(
            infer_module_native_cpu_with_report(
                &UnsupportedLater,
                TensorData::new([1, 2], vec![1., -1.]).unwrap(),
                &unsupported_executor,
                false,
            )
            .is_err()
        );
        assert_eq!(unsupported_executor.compile_cache_len(false), 0);
    }

    #[test]
    fn inference_rejects_poisoned_or_duplicate_modules_before_execution() {
        let poisoned = Linear::new_static(2, 1, true, 1).unwrap();
        let before = poisoned.bias.as_ref().unwrap().snapshot().unwrap();
        poisoned.weight.poison_for_test();
        assert!(matches!(
            infer_module_cpu(&poisoned, TensorData::new([1, 2], vec![0., 1.]).unwrap()),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert_eq!(
            poisoned.bias.as_ref().unwrap().snapshot().unwrap().data,
            before.data
        );
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            infer_module_native_cpu(
                &poisoned,
                TensorData::new([1, 2], vec![0., 1.]).unwrap(),
                &executor,
                false,
            ),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert_eq!(executor.compile_cache_len(false), 0);

        let duplicate = DuplicateTraversal {
            first: Parameter::new(
                TensorData::from_scalars([1], DType::F32, [Scalar::F(1.)]).unwrap(),
                true,
            ),
            second: Parameter::new(
                TensorData::from_scalars([1], DType::F32, [Scalar::F(2.)]).unwrap(),
                true,
            ),
        };
        let before = (
            duplicate.first.snapshot().unwrap(),
            duplicate.second.snapshot().unwrap(),
        );
        assert!(matches!(
            infer_module_cpu(&duplicate, TensorData::new([1, 1], vec![1.]).unwrap()),
            Err(Error::Serialization { .. })
        ));
        assert_eq!(duplicate.first.snapshot().unwrap().data, before.0.data);
        assert_eq!(duplicate.second.snapshot().unwrap().data, before.1.data);
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            infer_module_native_cpu(
                &duplicate,
                TensorData::new([1, 1], vec![1.]).unwrap(),
                &executor,
                false,
            ),
            Err(Error::Serialization { .. })
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
    }
}
