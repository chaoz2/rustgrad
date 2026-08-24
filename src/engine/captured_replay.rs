//! Graph-independent interpreter/native replay and deterministic batching.
use super::capture::{CapturedSchedule, ReplayError};
use crate::backend::{JitBackendError, PreparedScheduleItem};
use crate::{
    BufferRole, CpuJitBackend, ItemBackend, JitFallback, KernelBindings, KernelBufferDesc,
    ScheduleItem, TensorData,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapturedBackendPolicy {
    Interpreter,
    NativeJit { vectorized: bool },
    JitFallback { vectorized: bool },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapturedReplayOptions {
    pub backend: CapturedBackendPolicy,
}
impl Default for CapturedReplayOptions {
    fn default() -> Self {
        Self {
            backend: CapturedBackendPolicy::Interpreter,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedItemTrace {
    pub invocation: usize,
    pub item: u64,
    pub backend: ItemBackend,
    pub schedule_cache_key: u64,
    pub native_cache_key: Option<String>,
    pub cache_hit: bool,
    pub lanes: usize,
    pub vector_main: usize,
    pub vector_tail: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CapturedReplayTrace {
    pub items: Vec<CapturedItemTrace>,
}

#[derive(Clone, Debug)]
pub struct CapturedReplayResult {
    pub outputs: Vec<TensorData>,
    pub trace: CapturedReplayTrace,
}

#[derive(Clone, Debug)]
pub struct CapturedInvocation {
    bindings: BTreeMap<String, TensorData>,
}
impl CapturedInvocation {
    pub fn bindings(&self) -> &BTreeMap<String, TensorData> {
        &self.bindings
    }
}

#[derive(Clone, Debug)]
pub struct CapturedBatch {
    artifact_identity: u64,
    invocations: Vec<CapturedInvocation>,
}
impl CapturedBatch {
    pub fn new(
        capture: &CapturedSchedule,
        invocations: impl IntoIterator<Item = BTreeMap<String, TensorData>>,
    ) -> Result<Self, ReplayError> {
        let invocations = invocations
            .into_iter()
            .enumerate()
            .map(|(index, bindings)| {
                validate_inputs(capture, &bindings).map_err(|error| ReplayError::Batch {
                    invocation: index,
                    reason: error.to_string(),
                })?;
                Ok(CapturedInvocation { bindings })
            })
            .collect::<Result<Vec<_>, ReplayError>>()?;
        Ok(Self {
            artifact_identity: capture.identity,
            invocations,
        })
    }
    pub fn len(&self) -> usize {
        self.invocations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.invocations.is_empty()
    }
    pub fn invocations(&self) -> &[CapturedInvocation] {
        &self.invocations
    }
}

#[derive(Clone, Debug)]
pub struct CapturedBatchResult {
    pub invocations: Vec<CapturedReplayResult>,
}

pub struct CapturedReplayExecutor {
    scalar: CpuJitBackend,
    vectorized: CpuJitBackend,
}
impl Default for CapturedReplayExecutor {
    fn default() -> Self {
        Self {
            scalar: CpuJitBackend::new(JitFallback::Error),
            vectorized: CpuJitBackend::new(JitFallback::Error).vectorized(true),
        }
    }
}
impl CapturedReplayExecutor {
    pub fn compile_cache_len(&self, vectorized: bool) -> usize {
        self.jit(vectorized).cache_len()
    }

    pub fn replay(
        &self,
        capture: &CapturedSchedule,
        provided: &BTreeMap<String, TensorData>,
        options: CapturedReplayOptions,
    ) -> Result<CapturedReplayResult, ReplayError> {
        crate::schedule::artifact::validate_for_replay(capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        validate_inputs(capture, provided)?;
        let batch = CapturedBatch {
            artifact_identity: capture.identity,
            invocations: vec![CapturedInvocation {
                bindings: provided.clone(),
            }],
        };
        self.replay_batch(capture, &batch, options)?
            .invocations
            .into_iter()
            .next()
            .ok_or_else(|| ReplayError::Corrupt("single replay produced no invocation".into()))
    }

    pub fn replay_batch(
        &self,
        capture: &CapturedSchedule,
        batch: &CapturedBatch,
        options: CapturedReplayOptions,
    ) -> Result<CapturedBatchResult, ReplayError> {
        crate::schedule::artifact::validate_for_replay(capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        if batch.artifact_identity != capture.identity {
            return Err(ReplayError::Corrupt(
                "batch artifact identity mismatch".into(),
            ));
        }
        // All invocations and every native capability are validated before
        // compilation, allocation, or execution.
        for (index, invocation) in batch.invocations.iter().enumerate() {
            validate_inputs(capture, &invocation.bindings).map_err(|error| ReplayError::Batch {
                invocation: index,
                reason: error.to_string(),
            })?;
        }
        let plan = self.plan(capture, options.backend)?;
        let mut invocations = Vec::with_capacity(batch.len());
        for (index, invocation) in batch.invocations.iter().enumerate() {
            invocations.push(execute_invocation(
                capture,
                &invocation.bindings,
                index,
                &plan,
                options.backend,
                self,
            )?);
        }
        Ok(CapturedBatchResult { invocations })
    }

    fn plan(
        &self,
        capture: &CapturedSchedule,
        policy: CapturedBackendPolicy,
    ) -> Result<Vec<PlannedItem>, ReplayError> {
        let (fallback, vectorized) = match policy {
            CapturedBackendPolicy::Interpreter => {
                return Ok(capture
                    .items
                    .iter()
                    .map(|_| PlannedItem::Interpreter)
                    .collect());
            }
            CapturedBackendPolicy::NativeJit { vectorized } => (false, vectorized),
            CapturedBackendPolicy::JitFallback { vectorized } => (true, vectorized),
        };
        let jit = self.jit(vectorized);
        let mut native = Vec::with_capacity(capture.items.len());
        for item in &capture.items {
            match jit.validate_schedule_item(item) {
                Ok(()) => native.push(Ok(())),
                Err(error) if fallback => native.push(Err(error.to_string())),
                Err(error) => return Err(backend_error(error)),
            }
        }
        let mut out = Vec::with_capacity(capture.items.len());
        for (item, capability) in capture.items.iter().zip(native) {
            if let Err(reason) = capability {
                out.push(PlannedItem::Fallback(reason));
                continue;
            }
            match jit.prepare_schedule_item(item) {
                Ok(prepared) => out.push(PlannedItem::Native(prepared)),
                Err(error) if fallback => out.push(PlannedItem::Fallback(error.to_string())),
                Err(error) => return Err(backend_error(error)),
            }
        }
        Ok(out)
    }

    fn jit(&self, vectorized: bool) -> &CpuJitBackend {
        if vectorized {
            &self.vectorized
        } else {
            &self.scalar
        }
    }
}

impl CapturedSchedule {
    /// Replays this concrete artifact with an explicit backend policy and a
    /// caller-owned executor whose native compile cache survives across calls.
    pub fn replay_with_options(
        &self,
        provided: &BTreeMap<String, TensorData>,
        executor: &CapturedReplayExecutor,
        options: CapturedReplayOptions,
    ) -> Result<CapturedReplayResult, ReplayError> {
        executor.replay(self, provided, options)
    }
}

enum PlannedItem {
    Interpreter,
    Native(PreparedScheduleItem),
    Fallback(String),
}

fn execute_invocation(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
    invocation: usize,
    plan: &[PlannedItem],
    policy: CapturedBackendPolicy,
    executor: &CapturedReplayExecutor,
) -> Result<CapturedReplayResult, ReplayError> {
    let mut values = initial_values(capture, provided)?;
    let mut trace = CapturedReplayTrace::default();
    for (item, planned) in capture.items.iter().zip(plan) {
        let output_elements = item
            .output
            .shape
            .numel()
            .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
        let (value, backend, native_key, cache_hit, lanes, main, tail, reason) = match planned {
            PlannedItem::Interpreter => (
                interpret_item(capture, item, &values)?,
                ItemBackend::Interpreter,
                None,
                false,
                1,
                0,
                output_elements,
                "interpreter scalar semantics".into(),
            ),
            PlannedItem::Fallback(reason) => (
                interpret_item(capture, item, &values)?,
                ItemBackend::JitFallback,
                None,
                false,
                1,
                0,
                output_elements,
                reason.clone(),
            ),
            PlannedItem::Native(prepared) => {
                let vectorized = match policy {
                    CapturedBackendPolicy::NativeJit { vectorized }
                    | CapturedBackendPolicy::JitFallback { vectorized } => vectorized,
                    CapturedBackendPolicy::Interpreter => false,
                };
                let (value, execution) = executor
                    .jit(vectorized)
                    .execute_prepared_schedule_item(item, &values, prepared)
                    .map_err(backend_error)?;
                (
                    value,
                    ItemBackend::NativeJit,
                    Some(execution.cache_key),
                    prepared.cache_hit || invocation != 0,
                    execution.vector.lanes,
                    execution.vector_main,
                    execution.vector_tail,
                    execution.vector.reason,
                )
            }
        };
        values.insert(item.output.id, value);
        trace.items.push(CapturedItemTrace {
            invocation,
            item: item.id,
            backend,
            schedule_cache_key: item.cache_key,
            native_cache_key: native_key,
            cache_hit,
            lanes,
            vector_main: main,
            vector_tail: tail,
            reason,
        });
    }
    let outputs = capture
        .requested
        .iter()
        .map(|id| {
            values
                .get(id)
                .cloned()
                .ok_or_else(|| ReplayError::Missing(id.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CapturedReplayResult { outputs, trace })
}

fn validate_inputs(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
) -> Result<(), ReplayError> {
    let expected = capture
        .inputs
        .iter()
        .map(|x| x.name.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(name) = provided.keys().find(|x| !expected.contains(x.as_str())) {
        return Err(ReplayError::Extra(name.clone()));
    }
    for input in &capture.inputs {
        let value = provided
            .get(&input.name)
            .ok_or_else(|| ReplayError::Missing(input.name.clone()))?;
        if value.shape() != &input.desc.shape || value.dtype() != input.desc.dtype {
            return Err(ReplayError::Descriptor(input.name.clone()));
        }
    }
    Ok(())
}

fn initial_values(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
) -> Result<BTreeMap<u64, TensorData>, ReplayError> {
    let mut values = capture.constants.clone();
    for input in &capture.inputs {
        values.insert(
            input.desc.id,
            provided
                .get(&input.name)
                .cloned()
                .ok_or_else(|| ReplayError::Missing(input.name.clone()))?,
        );
    }
    Ok(values)
}

fn interpret_item(
    capture: &CapturedSchedule,
    item: &ScheduleItem,
    values: &BTreeMap<u64, TensorData>,
) -> Result<TensorData, ReplayError> {
    let mut bindings = KernelBindings::default();
    for binding in item.ordered_inputs() {
        let value = values
            .get(&binding.desc.id)
            .cloned()
            .ok_or_else(|| ReplayError::Missing(binding.desc.id.to_string()))?;
        let role = if capture.constants.contains_key(&binding.desc.id) {
            BufferRole::Constant
        } else {
            BufferRole::Input
        };
        let desc = KernelBufferDesc::concrete(
            binding.desc.id,
            role,
            binding.desc.shape.clone(),
            binding.desc.dtype,
            false,
        )
        .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
        bindings
            .insert(&desc, value)
            .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
    }
    crate::kernel::execute_lowered_elementwise(&item.kernel, &bindings)
        .map_err(|e| ReplayError::Execute(e.to_string()))
}

fn backend_error(error: JitBackendError) -> ReplayError {
    match error {
        JitBackendError::Unsupported(reason) => ReplayError::Unsupported(reason),
        other => ReplayError::Backend(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape, UArg};
    use std::collections::HashMap;

    fn captured(graph: &Graph, requested: &[crate::NodeId]) -> CapturedSchedule {
        let schedule = crate::schedule_many(graph, requested).unwrap();
        let capture = CapturedSchedule::capture(graph, &schedule, requested).unwrap();
        CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn deserialized_native_multi_item_replay_matches_oracle_and_hits_cache() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([5]), DType::F32);
        let shared = graph.square(x).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(shared, one).unwrap();
        let right = graph.mul(shared, one).unwrap();
        let capture = captured(&graph, &[left, right]);
        let bindings = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars([5], DType::F32, [-2., -1., 0., 1., 2.].map(Scalar::F))
                .unwrap(),
        )]);
        let oracle_bindings = bindings.clone().into_iter().collect::<HashMap<_, _>>();
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &bindings, options).unwrap();
        let second = executor.replay(&capture, &bindings, options).unwrap();
        for ((actual, again), node) in first.outputs.iter().zip(&second.outputs).zip([left, right])
        {
            let expected = CpuBackend.execute(&graph, node, &oracle_bindings).unwrap();
            assert_eq!(actual.storage(), expected.storage());
            assert_eq!(again.storage(), expected.storage());
        }
        assert!(first.trace.items.iter().all(|x| {
            x.backend == ItemBackend::NativeJit
                && !x.cache_hit
                && x.schedule_cache_key == capture.items[x.item as usize].cache_key
        }));
        assert!(second.trace.items.iter().all(|x| x.cache_hit));
        assert_eq!(executor.compile_cache_len(false), capture.items.len());
    }

    #[test]
    fn native_view_reduction_and_zero_domain_match_interpreter() {
        let executor = CapturedReplayExecutor::default();
        let native = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let interpreter = CapturedReplayOptions::default();

        let mut view_graph = Graph::new();
        let x = view_graph.input_dtype("x", Shape::from([5]), DType::F32);
        let view = view_graph.shrink(x, [(1, 5)]).unwrap();
        let output = view_graph.neg(view).unwrap();
        let view_capture = captured(&view_graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::new([5], vec![0., 1., 2., 3., 4.]).unwrap(),
        )]);
        let view_result = executor
            .replay(
                &view_capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::JitFallback { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            view_result.outputs[0].storage(),
            executor
                .replay(&view_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );
        assert_eq!(view_result.trace.items[0].backend, ItemBackend::JitFallback);

        let mut reduction_graph = Graph::new();
        let x = reduction_graph.input_dtype("x", Shape::from([2, 3]), DType::F32);
        let output = reduction_graph
            .reduce(x, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let reduction_capture = captured(&reduction_graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
        )]);
        assert_eq!(
            executor
                .replay(&reduction_capture, &values, native)
                .unwrap()
                .outputs[0]
                .storage(),
            executor
                .replay(&reduction_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );

        let mut empty_graph = Graph::new();
        let x = empty_graph.input_dtype("x", Shape::from([0]), DType::F32);
        let output = empty_graph.square(x).unwrap();
        let empty_capture = captured(&empty_graph, &[output]);
        let values = BTreeMap::from([("x".into(), TensorData::new([0], vec![]).unwrap())]);
        assert_eq!(
            executor
                .replay(&empty_capture, &values, native)
                .unwrap()
                .outputs[0]
                .storage(),
            executor
                .replay(&empty_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );

        let mut vector_graph = Graph::new();
        let x = vector_graph.input_dtype("x", Shape::from([5]), DType::F32);
        let output = vector_graph.square(x).unwrap();
        let vector_capture = captured(&vector_graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::new([5], vec![-2., -1., 0., 1., 2.]).unwrap(),
        )]);
        let vector = executor
            .replay(
                &vector_capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(
            vector.outputs[0].storage(),
            executor
                .replay(&vector_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );
        assert_eq!(vector.trace.items[0].backend, ItemBackend::NativeJit);
        assert!(vector.trace.items[0].lanes > 1);
        assert_eq!(vector.trace.items[0].vector_main, 4);
        assert_eq!(vector.trace.items[0].vector_tail, 1);
    }

    #[test]
    fn unsupported_native_policy_is_explicit() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2]), DType::F32);
        let output = graph.exp(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([("x".into(), TensorData::new([2], vec![0., 1.]).unwrap())]);
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            executor.replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Unsupported(_))
        ));
        let fallback = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::JitFallback { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(fallback.trace.items[0].backend, ItemBackend::JitFallback);
        assert_eq!(
            fallback.outputs[0].storage(),
            capture.replay(&values).unwrap()[0].storage()
        );
        assert_eq!(executor.compile_cache_len(false), 0);

        let mut vector_graph = Graph::new();
        let x = vector_graph.input("x", Shape::from([4]));
        let one = vector_graph.constant(TensorData::scalar(1.0));
        let output = vector_graph.add(x, one).unwrap();
        let vector_capture = captured(&vector_graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::new([4], vec![1., 2., 3., 4.]).unwrap(),
        )]);
        assert!(matches!(
            executor.replay(
                &vector_capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true }
                }
            ),
            Err(ReplayError::Unsupported(_))
        ));
    }

    #[test]
    fn native_replay_translates_schedule_operand_order_to_native_abi() {
        let mut graph = Graph::new();
        let right = graph.input_dtype("right", Shape::from([2]), DType::F32);
        let left = graph.input_dtype("left", Shape::from([2]), DType::F32);
        let output = graph.sub(left, right).unwrap();
        let capture = captured(&graph, &[output]);
        assert_eq!(capture.items[0].input_bindings[0].input_node, left);
        assert_eq!(capture.items[0].input_bindings[1].input_node, right);
        let values = BTreeMap::from([
            ("left".into(), TensorData::new([2], vec![7., 11.]).unwrap()),
            ("right".into(), TensorData::new([2], vec![2., 3.]).unwrap()),
        ]);
        let result = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(result.outputs[0].values(), &[5., 8.]);
    }

    #[test]
    fn batch_preflight_order_and_owned_outputs_are_deterministic() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.square(x).unwrap();
        let capture = captured(&graph, &[output]);
        let first = BTreeMap::from([("x".into(), TensorData::new([3], vec![1., 2., 3.]).unwrap())]);
        let second =
            BTreeMap::from([("x".into(), TensorData::new([3], vec![4., 5., 6.]).unwrap())]);
        let executor = CapturedReplayExecutor::default();
        let malformed = CapturedBatch::new(
            &capture,
            [
                first.clone(),
                BTreeMap::from([("x".into(), TensorData::scalar(1.0))]),
            ],
        );
        assert!(matches!(
            malformed,
            Err(ReplayError::Batch { invocation: 1, .. })
        ));
        assert_eq!(executor.compile_cache_len(false), 0);

        let batch = CapturedBatch::new(&capture, [first, second]).unwrap();
        let mut wrong_artifact = batch.clone();
        wrong_artifact.artifact_identity ^= 1;
        assert!(matches!(
            executor.replay_batch(
                &capture,
                &wrong_artifact,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Corrupt(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
        let result = executor
            .replay_batch(
                &capture,
                &batch,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(result.invocations[0].outputs[0].values(), &[1., 4., 9.]);
        assert_eq!(result.invocations[1].outputs[0].values(), &[16., 25., 36.]);
        assert_eq!(result.invocations[0].trace.items[0].invocation, 0);
        assert_eq!(result.invocations[1].trace.items[0].invocation, 1);
        assert!(!result.invocations[0].trace.items[0].cache_hit);
        assert!(result.invocations[1].trace.items[0].cache_hit);
        assert_ne!(
            result.invocations[0].outputs[0].values().as_ptr(),
            result.invocations[1].outputs[0].values().as_ptr()
        );
    }

    #[test]
    fn matmul_artifacts_replay_interpreter_native_and_batches() {
        struct Case {
            name: &'static str,
            dtype: DType,
            lhs: Vec<usize>,
            rhs: Vec<usize>,
        }
        let cases = [
            Case {
                name: "dot",
                dtype: DType::F32,
                lhs: vec![3],
                rhs: vec![3],
            },
            Case {
                name: "matvec",
                dtype: DType::F64,
                lhs: vec![2, 3],
                rhs: vec![3],
            },
            Case {
                name: "vecmat",
                dtype: DType::F32,
                lhs: vec![3],
                rhs: vec![3, 2],
            },
            Case {
                name: "broadcast batch",
                dtype: DType::F64,
                lhs: vec![2, 1, 2, 3],
                rhs: vec![1, 4, 3, 2],
            },
            Case {
                name: "zero k",
                dtype: DType::F32,
                lhs: vec![2, 0],
                rhs: vec![0, 3],
            },
        ];
        for case in cases {
            let mut graph = Graph::new();
            let lhs_node = graph.input_dtype("lhs", case.lhs.clone(), case.dtype);
            let rhs_node = graph.input_dtype("rhs", case.rhs.clone(), case.dtype);
            let output = graph.matmul(lhs_node, rhs_node).unwrap();
            let schedule = crate::schedule(&graph, output).unwrap();
            assert_eq!(schedule.items.len(), 1, "{} item count", case.name);
            assert!(
                schedule.items[0].boundary.is_none(),
                "{} boundary",
                case.name
            );
            assert!(matches!(
                schedule.items[0].kernel.kind(),
                crate::UOpKind::Matmul
            ));
            assert_eq!(
                schedule.items[0]
                    .ordered_inputs()
                    .iter()
                    .map(|binding| binding.input_node)
                    .collect::<Vec<_>>(),
                vec![lhs_node, rhs_node],
                "{} ABI",
                case.name
            );
            let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
            let bytes = capture.to_bytes().unwrap();
            let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
            assert_eq!(bytes, decoded.to_bytes().unwrap(), "{} bytes", case.name);
            let lhs = TensorData::from_scalars(
                case.lhs,
                case.dtype,
                (0..graph.shape(lhs_node).unwrap().numel().unwrap())
                    .map(|index| Scalar::F(index as f64 * 0.25 - 1.0)),
            )
            .unwrap();
            let rhs = TensorData::from_scalars(
                case.rhs,
                case.dtype,
                (0..graph.shape(rhs_node).unwrap().numel().unwrap())
                    .map(|index| Scalar::F(index as f64 * -0.125 + 0.75)),
            )
            .unwrap();
            let bindings =
                BTreeMap::from([("lhs".into(), lhs.clone()), ("rhs".into(), rhs.clone())]);
            let oracle = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]),
                )
                .unwrap();
            let executor = CapturedReplayExecutor::default();
            let interpreted = executor
                .replay(&decoded, &bindings, CapturedReplayOptions::default())
                .unwrap();
            let options = CapturedReplayOptions {
                backend: CapturedBackendPolicy::NativeJit { vectorized: false },
            };
            let first = executor.replay(&decoded, &bindings, options).unwrap();
            let second = executor.replay(&decoded, &bindings, options).unwrap();
            assert_eq!(
                interpreted.outputs[0].storage(),
                oracle.storage(),
                "{} interpreter",
                case.name
            );
            assert_eq!(
                first.outputs[0].storage(),
                oracle.storage(),
                "{} native",
                case.name
            );
            assert_eq!(first.trace.items[0].backend, ItemBackend::NativeJit);
            assert!(!first.trace.items[0].cache_hit);
            assert!(second.trace.items[0].cache_hit);
        }

        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 2], DType::F32);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let capture = captured(&graph, &[output]);
        let invocation = |offset: f32| {
            BTreeMap::from([
                (
                    "lhs".into(),
                    TensorData::new([2, 2], vec![offset, 1., 2., 3.]).unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::new([2, 2], vec![1., 2., 3., offset]).unwrap(),
                ),
            ])
        };
        let batch = CapturedBatch::new(&capture, [invocation(4.), invocation(5.)]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let result = executor
            .replay_batch(
                &capture,
                &batch,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            result.invocations[0].outputs[0].values(),
            &[7., 12., 11., 16.]
        );
        assert_eq!(
            result.invocations[1].outputs[0].values(),
            &[8., 15., 11., 19.]
        );
        assert!(!result.invocations[0].trace.items[0].cache_hit);
        assert!(result.invocations[1].trace.items[0].cache_hit);
        assert_eq!(executor.compile_cache_len(false), 1);

        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::F32);
        let bias = graph.input_dtype("bias", [2, 2], DType::F32);
        let squared = graph.square(input).unwrap();
        let product = graph.matmul(squared, rhs).unwrap();
        let output = graph.add(product, bias).unwrap();
        let capture = captured(&graph, &[output]);
        let bindings = BTreeMap::from([
            (
                "input".into(),
                TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::new([2, 2], vec![2., 1., 0., 3.]).unwrap(),
            ),
            (
                "bias".into(),
                TensorData::new([2, 2], vec![1., 1., 1., 1.]).unwrap(),
            ),
        ]);
        let oracle = CpuBackend
            .execute(
                &graph,
                output,
                &bindings.clone().into_iter().collect::<HashMap<_, _>>(),
            )
            .unwrap();
        let executor = CapturedReplayExecutor::default();
        let replay = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(replay.outputs[0].storage(), oracle.storage());
        assert_eq!(replay.trace.items.len(), 3);
        assert!(
            replay
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_eq!(executor.compile_cache_len(false), 3);
    }

    #[test]
    fn matmul_native_dtype_and_artifact_abi_fail_before_compilation() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 2], DType::F16);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::F16);
        let output = graph.matmul(lhs, rhs).unwrap();
        let capture = captured(&graph, &[output]);
        let bindings = BTreeMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([2, 2], DType::F16, [1., 2., 3., 4.].map(Scalar::F))
                    .unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([2, 2], DType::F16, [4., 3., 2., 1.].map(Scalar::F))
                    .unwrap(),
            ),
        ]);
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            executor.replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Unsupported(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
        let fallback = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::JitFallback { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(fallback.trace.items[0].backend, ItemBackend::JitFallback);
        assert_eq!(
            fallback.outputs[0].storage(),
            capture.replay(&bindings).unwrap()[0].storage()
        );

        let mut malformed_abi = capture.clone();
        malformed_abi.items[0].input_bindings.swap(0, 1);
        assert!(matches!(
            executor.replay(
                &malformed_abi,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Corrupt(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);

        let mut malformed_plan = capture;
        let UArg::Matmul(plan) = malformed_plan.items[0].kernel.arg() else {
            panic!("matmul payload missing");
        };
        let mut plan = plan.as_ref().clone();
        plan.output_shape = Shape::from([4]);
        malformed_plan.items[0].kernel = crate::UOp::new(
            crate::UOpKind::Matmul,
            Some(crate::UType::scalar(DType::F16)),
            vec![],
            UArg::Matmul(Box::new(plan)),
        );
        assert!(matches!(
            executor.replay(
                &malformed_plan,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Corrupt(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
    }
}
