//! Deterministic realization of scheduled UOp items.
pub mod capture;
mod captured_replay;
pub(crate) mod dynamic;
mod mixed;
pub mod mixed_batch;
pub mod mixed_capture;
pub mod mixed_rebinding;
mod replay_liveness;
pub(crate) mod symbolic;
pub(crate) mod symbolic_view;
use crate::host_buffer::{HostBufferDesc, HostBufferError, HostBufferLease, HostSlotPool};
use crate::{
    Backend, BufferRole, CpuJitBackend, Graph, JitFallback, KernelBindings, KernelBufferDesc,
    MemoryPlan, NodeId, Op, Schedule, Shape, TensorData,
};
pub use captured_replay::{
    CapturedBackendPolicy, CapturedBatch, CapturedBatchResult, CapturedInvocation,
    CapturedItemTrace, CapturedReplayExecutor, CapturedReplayOptions, CapturedReplayResult,
    CapturedReplayTrace, CapturedSpecialization, CapturedSpecializationTrace,
};
pub use mixed::realize_mixed_effects;
use std::{collections::HashMap, fmt};
pub use symbolic::{SymbolicCaptureSpec, SymbolicGuard, SymbolicParameter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealizationPolicy {
    Interpreter,
    CpuJit { fallback_to_interpreter: bool },
}
/// Whether logical internal allocations can reuse a released exact-compatible
/// slot. The default entry point keeps reuse disabled for compatibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryReuse {
    Disabled,
    Enabled,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RealizationOptions {
    pub backend: RealizationPolicy,
    pub memory_reuse: MemoryReuse,
}
impl Default for RealizationOptions {
    fn default() -> Self {
        Self {
            backend: RealizationPolicy::Interpreter,
            memory_reuse: MemoryReuse::Disabled,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ItemBackend {
    Interpreter,
    NativeJit,
    JitFallback,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ItemTrace {
    pub item: u64,
    pub dependencies: Vec<u64>,
    pub backend: ItemBackend,
    pub cache_key: u64,
    pub materialized_buffer: u64,
    /// Stable schedule item at which this owned buffer has its final consumer.
    /// A future allocator can reuse only after this point.
    pub last_consumer: Option<u64>,
    pub allocation_id: Option<u64>,
    pub physical_slot: Option<u64>,
    pub generation: Option<u64>,
    pub reused_from: Option<u64>,
    pub released_buffers: Vec<u64>,
    pub lanes: usize,
    pub vector_main: usize,
    pub vector_tail: usize,
    pub vector_reason: String,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RealizationTrace {
    pub items: Vec<ItemTrace>,
}
#[derive(Clone, Debug)]
pub struct Realized {
    pub outputs: Vec<TensorData>,
    pub trace: RealizationTrace,
}

/// Executes a normal schedule whose items are exclusively universal
/// STORE/AFTER effect items.  The graph-adjacent state model supplies the
/// immutable versioned snapshots; validation compares the caller's schedule
/// with the canonical lowering before any candidate is staged.
pub fn realize_effects(
    graph: &crate::EffectGraph,
    schedule: &Schedule,
    injected_failure: Option<u64>,
) -> Result<crate::EffectCommit, RealizationError> {
    if schedule.items.iter().any(|item| !item.is_effect()) {
        return Err(RealizationError::Unsupported(
            "mixed pure/effect schedules require a value-to-state binding boundary".into(),
        ));
    }
    let expected = crate::schedule_effects(graph)
        .map_err(|error| RealizationError::Schedule(error.to_string()))?;
    if schedule.items.len() != expected.items.len()
        || schedule
            .items
            .iter()
            .zip(&expected.items)
            .any(|(actual, canonical)| actual.cache_key != canonical.cache_key)
    {
        return Err(RealizationError::Schedule(
            "effect schedule does not match canonical state lowering".into(),
        ));
    }
    if let Some(step) = injected_failure {
        if schedule.items.iter().any(|item| item.id == step) {
            return Err(RealizationError::Execution(format!(
                "injected effect failure at item {step}"
            )));
        }
        return Err(RealizationError::Schedule(format!(
            "injected effect item {step} is absent"
        )));
    }
    graph
        .execute()
        .map_err(|error| RealizationError::Execution(error.to_string()))
}

/// Persistent counterpart of [`realize_effects`]. It validates the same
/// canonical normal schedule, but commits successor versions into the caller's
/// long-lived host-backed effect runtime instead of returning detached bytes.
pub fn realize_effects_persistent(
    runtime: &mut crate::EffectRuntime,
    graph: &crate::EffectGraph,
    schedule: &Schedule,
    injected_failure: Option<u64>,
) -> Result<Vec<crate::BufferState>, RealizationError> {
    if schedule.items.iter().any(|item| !item.is_effect()) {
        return Err(RealizationError::Unsupported(
            "mixed pure/effect schedules require a value-to-state binding boundary".into(),
        ));
    }
    let expected = crate::schedule_effects(graph)
        .map_err(|error| RealizationError::Schedule(error.to_string()))?;
    if schedule.items.len() != expected.items.len()
        || schedule
            .items
            .iter()
            .zip(&expected.items)
            .any(|(actual, canonical)| actual.cache_key != canonical.cache_key)
    {
        return Err(RealizationError::Schedule(
            "effect schedule does not match canonical state lowering".into(),
        ));
    }
    runtime
        .execute(&graph.plan(), injected_failure)
        .map_err(|error| {
            RealizationError::Execution(format!("persistent effect runtime: {error:?}"))
        })
}

/// Concrete, validated shape produced by a dynamic-result realization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeShape(Shape);
impl RuntimeShape {
    pub fn new(expected_rank: usize, shape: Shape) -> Result<Self, RealizationError> {
        if shape.rank() == expected_rank {
            Ok(Self(shape))
        } else {
            Err(RealizationError::Execution(
                "dynamic result rank changed".into(),
            ))
        }
    }
    pub fn shape(&self) -> &Shape {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct DynamicRealized {
    pub output: TensorData,
    pub shape: RuntimeShape,
}

/// First-order dynamic-loss execution result. `gradient` always has the
/// requested static source shape; the loss retains its validated runtime shape.
#[derive(Clone, Debug)]
pub struct DynamicGradient {
    pub loss: DynamicRealized,
    pub gradient: TensorData,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RealizationError {
    Schedule(String),
    MissingBuffer(u64),
    Unsupported(String),
    Execution(String),
    Memory(crate::MemoryPlanError),
    Host(HostBufferError),
}
impl fmt::Display for RealizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "realization error: {self:?}")
    }
}
impl std::error::Error for RealizationError {}

pub fn realize(
    graph: &Graph,
    schedule: &Schedule,
    requested: &[NodeId],
    inputs: &HashMap<String, TensorData>,
    policy: RealizationPolicy,
) -> Result<Realized, RealizationError> {
    realize_with_options(
        graph,
        schedule,
        requested,
        inputs,
        RealizationOptions {
            backend: policy,
            memory_reuse: MemoryReuse::Disabled,
        },
    )
}

pub fn realize_with_options(
    graph: &Graph,
    schedule: &Schedule,
    requested: &[NodeId],
    inputs: &HashMap<String, TensorData>,
    options: RealizationOptions,
) -> Result<Realized, RealizationError> {
    // Preflight the complete output inventory before the planner can lease a
    // slot or the JIT can populate a process-local compile cache.
    schedule
        .validate()
        .map_err(|error| RealizationError::Schedule(error.to_string()))?;
    if schedule.items.iter().any(crate::ScheduleItem::is_effect) {
        return Err(RealizationError::Unsupported(
            "effect schedules must use transactional realize_effects".into(),
        ));
    }
    if schedule.items.iter().any(|item| {
        !item.outputs.is_single() && !matches!(item.kernel.kind(), crate::UOpKind::Sort)
    }) {
        return Err(RealizationError::Unsupported(
            "multi-output schedule items have no executor lowering".into(),
        ));
    }
    let policy = options.backend;
    let plan = MemoryPlan::from_schedule(
        schedule,
        requested,
        options.memory_reuse == MemoryReuse::Enabled,
    )
    .map_err(RealizationError::Memory)?;
    let assignments = plan
        .temporaries
        .iter()
        .map(|entry| (entry.buffer_id, entry))
        .collect::<HashMap<_, _>>();
    let requests = plan
        .requests
        .iter()
        .map(|request| (request.buffer_id, request))
        .collect::<HashMap<_, _>>();
    let requested_buffers = requested
        .iter()
        .map(|node| node.index() as u64)
        .collect::<std::collections::BTreeSet<_>>();
    // Only retained outputs live here. Internal values are reachable solely
    // through non-cloneable, generation-checked pool leases.
    let mut values: HashMap<u64, TensorData> = HashMap::new();
    let mut leases: HashMap<u64, HostBufferLease> = HashMap::new();
    let pool = HostSlotPool::new();
    let mut trace = RealizationTrace::default();
    let jit = matches!(policy, RealizationPolicy::CpuJit { .. })
        .then(|| CpuJitBackend::new(JitFallback::Error));
    for item in &schedule.items {
        if !item.outputs.is_single() && !matches!(item.kernel.kind(), crate::UOpKind::Sort) {
            return Err(RealizationError::Unsupported(format!(
                "item {} has no multi-output executor",
                item.id
            )));
        }
        if item.boundary.is_some() {
            return Err(RealizationError::Unsupported(format!(
                "item {} has boundary {:?}",
                item.id, item.boundary
            )));
        }
        if item
            .dependencies
            .iter()
            .any(|dependency| !trace.items.iter().any(|entry| entry.item == *dependency))
        {
            return Err(RealizationError::Schedule(format!(
                "item {} uses a future dependency",
                item.id
            )));
        }
        let mut backend = ItemBackend::Interpreter;
        let mut lanes = 1;
        let mut vector_main = 0;
        let mut vector_tail = 0;
        let mut vector_reason = "interpreter scalar semantics".to_string();
        let materialized = materialized_values(&leases, &values).map_err(RealizationError::Host)?;
        let sort_pair = if matches!(item.kernel.kind(), crate::UOpKind::Sort) {
            if jit.is_some() {
                return Err(RealizationError::Unsupported(
                    "static sort pairs are CPU-interpreter only".into(),
                ));
            }
            Some(
                interpret_sort_pair(graph, item, inputs, &materialized)
                    .map_err(RealizationError::Execution)?,
            )
        } else {
            None
        };
        let value = if let Some((values, _)) = &sort_pair {
            values.clone()
        } else if let Some(jit) = &jit {
            let native_eligible = item.dependencies.is_empty()
                && item.inputs.iter().all(|buffer| {
                    matches!(
                        graph.op(NodeId::from_index(buffer.id as usize)),
                        Ok(Op::Input { .. } | Op::Constant(_))
                    )
                });
            if native_eligible {
                match jit.execute_native(graph, item.node, inputs) {
                    Ok((value, execution)) => {
                        backend = ItemBackend::NativeJit;
                        lanes = execution.vector.lanes;
                        vector_main = execution.vector_main;
                        vector_tail = execution.vector_tail;
                        vector_reason = execution.vector.reason;
                        value
                    }
                    Err(error)
                        if matches!(
                            policy,
                            RealizationPolicy::CpuJit {
                                fallback_to_interpreter: true
                            }
                        ) =>
                    {
                        backend = ItemBackend::JitFallback;
                        interpret_item(graph, item, inputs, &materialized)
                            .map_err(|e| RealizationError::Execution(format!("{error}; {e}")))?
                    }
                    Err(error) => return Err(RealizationError::Execution(error.to_string())),
                }
            } else if matches!(
                policy,
                RealizationPolicy::CpuJit {
                    fallback_to_interpreter: true
                }
            ) {
                backend = ItemBackend::JitFallback;
                interpret_item(graph, item, inputs, &materialized)
                    .map_err(RealizationError::Execution)?
            } else {
                return Err(RealizationError::Unsupported(format!(
                    "item {} cannot use native CPU JIT with materialized dependencies",
                    item.id
                )));
            }
        } else {
            interpret_item(graph, item, inputs, &materialized)
                .map_err(RealizationError::Execution)?
        };
        let output = item.primary_output();
        let assignment = assignments.get(&output.id);
        let (physical_slot, generation) = if let Some(assignment) = assignment {
            let request = requests
                .get(&output.id)
                .ok_or(RealizationError::MissingBuffer(output.id))?;
            let descriptor = HostBufferDesc {
                buffer_id: request.buffer_id,
                dtype: request.dtype,
                shape: request.shape.clone(),
                bytes: request.bytes,
                alignment: request.alignment,
                lanes: portable_lanes(request.dtype),
            };
            let mut lease = pool
                .lease(assignment.allocation_id, descriptor)
                .map_err(RealizationError::Host)?;
            let output_window = lease
                .mutable_window(0, request.bytes)
                .map_err(RealizationError::Host)?;
            output_window.validate().map_err(RealizationError::Host)?;
            drop(output_window);
            lease.write(value).map_err(RealizationError::Host)?;
            let slot = lease.slot();
            let generation = lease.generation();
            if leases.insert(output.id, lease).is_some() {
                return Err(RealizationError::Schedule(format!(
                    "duplicate live temporary {}",
                    output.id
                )));
            }
            (assignment.allocation_id.map(|_| slot), Some(generation))
        } else {
            values.insert(output.id, value);
            (None, None)
        };
        if let Some((_, indices)) = sort_pair {
            let secondary = item.outputs.iter().nth(1).ok_or_else(|| {
                RealizationError::Schedule("sort indices output is absent".into())
            })?;
            let assignment = assignments.get(&secondary.id);
            if let Some(assignment) = assignment {
                let request = requests
                    .get(&secondary.id)
                    .ok_or(RealizationError::MissingBuffer(secondary.id))?;
                let descriptor = HostBufferDesc {
                    buffer_id: request.buffer_id,
                    dtype: request.dtype,
                    shape: request.shape.clone(),
                    bytes: request.bytes,
                    alignment: request.alignment,
                    lanes: portable_lanes(request.dtype),
                };
                let lease = pool
                    .lease(assignment.allocation_id, descriptor)
                    .map_err(RealizationError::Host)?;
                lease.write(indices).map_err(RealizationError::Host)?;
                if leases.insert(secondary.id, lease).is_some() {
                    return Err(RealizationError::Schedule(format!(
                        "duplicate live temporary {}",
                        secondary.id
                    )));
                }
            } else {
                values.insert(secondary.id, indices);
            }
        }
        let released_buffers = plan
            .temporaries
            .iter()
            .filter(|entry| {
                entry.last_consumer == item.id && !requested_buffers.contains(&entry.buffer_id)
            })
            .map(|entry| entry.buffer_id)
            .collect::<Vec<_>>();
        trace.items.push(ItemTrace {
            item: item.id,
            dependencies: item.dependencies.clone(),
            backend,
            cache_key: item.cache_key,
            materialized_buffer: output.id,
            last_consumer: item.consumers.last().copied(),
            allocation_id: assignment.and_then(|entry| entry.allocation_id),
            physical_slot,
            generation,
            reused_from: assignment.and_then(|entry| entry.reused_from),
            released_buffers: released_buffers.clone(),
            lanes,
            vector_main,
            vector_tail,
            vector_reason,
        });
        for buffer in released_buffers {
            let mut lease = leases
                .remove(&buffer)
                .ok_or(RealizationError::MissingBuffer(buffer))?;
            lease.release().map_err(RealizationError::Host)?;
        }
    }
    let outputs = requested
        .iter()
        .map(|node| {
            values
                .get(&(node.index() as u64))
                .cloned()
                .ok_or(RealizationError::MissingBuffer(node.index() as u64))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if pool.physical_slots().map_err(RealizationError::Host)? != plan.peak_allocations {
        return Err(RealizationError::Schedule(
            "host pool slot count diverges from memory plan".into(),
        ));
    }
    Ok(Realized { outputs, trace })
}

fn portable_lanes(dtype: crate::DType) -> usize {
    if dtype.itemsize() >= 8 {
        1
    } else {
        (16 / dtype.itemsize()).max(1)
    }
}

fn materialized_values(
    leases: &HashMap<u64, HostBufferLease>,
    retained: &HashMap<u64, TensorData>,
) -> Result<HashMap<u64, TensorData>, HostBufferError> {
    let mut values = retained.clone();
    for (buffer, lease) in leases {
        let view = lease.view()?;
        // The interpreter receives only a checked call-duration logical window;
        // the backing slot capacity and pointer remain private to the pool.
        let window = view.byte_window(0, view.logical_bytes())?;
        window.validate()?;
        let _logical_len = window.len();
        values.insert(*buffer, view.tensor()?);
    }
    Ok(values)
}

/// Convenience entry point for the internal lazy path. Scheduling is repeated
/// for each call so symbolic bindings remain concrete in both the schedule
/// descriptors and the executable kernel cache identity.
pub fn realize_graph(
    graph: &Graph,
    requested: &[NodeId],
    inputs: &HashMap<String, TensorData>,
    policy: RealizationPolicy,
) -> Result<Realized, RealizationError> {
    let schedule = crate::schedule_many(graph, requested)
        .map_err(|error| RealizationError::Schedule(error.to_string()))?;
    realize(graph, &schedule, requested, inputs, policy)
}

pub fn realize_graph_with_options(
    graph: &Graph,
    requested: &[NodeId],
    inputs: &HashMap<String, TensorData>,
    options: RealizationOptions,
) -> Result<Realized, RealizationError> {
    let schedule = crate::schedule_many(graph, requested)
        .map_err(|error| RealizationError::Schedule(error.to_string()))?;
    realize_with_options(graph, &schedule, requested, inputs, options)
}

fn interpret_item(
    graph: &Graph,
    item: &crate::ScheduleItem,
    inputs: &HashMap<String, TensorData>,
    values: &HashMap<u64, TensorData>,
) -> Result<TensorData, String> {
    if matches!(
        graph.op(item.node),
        Ok(Op::Reduce { .. } | Op::PrefixScan { .. } | Op::TensorGuard { .. } | Op::Sort { .. })
    ) && item.dependencies.is_empty()
    {
        if matches!(
            graph.op(item.node),
            Ok(Op::PrefixScan { .. } | Op::TensorGuard { .. })
        ) {
            return crate::CpuBackend
                .execute(graph, item.node, inputs)
                .map_err(|error| error.to_string());
        }
        return crate::execute_elementwise(graph, item.node, inputs).map_err(|e| e.to_string());
    }
    if let crate::UArgRef::Movement(plan) = item.kernel.arg() {
        let operands = plan
            .input_operands()
            .into_iter()
            .map(|operand| {
                let id = operand.node.index() as u64;
                if let Some(value) = values.get(&id) {
                    return Ok(value.clone());
                }
                match graph.op(operand.node).map_err(|error| error.to_string())? {
                    Op::Input { name } => inputs
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("missing input {name}")),
                    Op::Constant(value) => Ok(value.clone()),
                    _ => Err(format!("missing materialized buffer {id}")),
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        return plan.execute(&operands).map_err(|error| error.to_string());
    }
    let mut bindings = KernelBindings::default();
    item.validate_input_bindings().map_err(|e| e.to_string())?;
    for binding in item.ordered_inputs() {
        let desc = &binding.desc;
        let id = NodeId::from_index(desc.id as usize);
        let value = if let Some(value) = values.get(&desc.id) {
            value.clone()
        } else {
            match graph.op(id).map_err(|e| e.to_string())? {
                Op::Input { name } => inputs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("missing input {name}"))?,
                Op::Constant(value) => value.clone(),
                _ => return Err(format!("missing materialized buffer {}", desc.id)),
            }
        };
        let role = if matches!(graph.op(id), Ok(Op::Constant(_))) {
            BufferRole::Constant
        } else {
            BufferRole::Input
        };
        let kernel_desc =
            KernelBufferDesc::concrete(desc.id, role, desc.shape.clone(), desc.dtype, false)
                .map_err(|e| e.to_string())?;
        bindings
            .insert(&kernel_desc, value)
            .map_err(|e| e.to_string())?;
    }
    crate::kernel::execute_lowered_elementwise(&item.kernel, &bindings).map_err(|e| e.to_string())
}

fn interpret_sort_pair(
    graph: &Graph,
    item: &crate::ScheduleItem,
    inputs: &HashMap<String, TensorData>,
    values: &HashMap<u64, TensorData>,
) -> Result<(TensorData, TensorData), String> {
    let crate::UArgRef::Sort {
        input,
        input_shape,
        axis,
        descending,
        values: value_node,
        indices: index_node,
        dtype,
    } = item.kernel.arg()
    else {
        return Err("sort item has no typed pair payload".into());
    };
    if item.node != *value_node
        || item.outputs.len() != 2
        || item.primary_output().id != value_node.index() as u64
        || item.outputs.iter().nth(1).is_none_or(|desc| {
            desc.id != index_node.index() as u64
                || desc.shape != *input_shape
                || desc.dtype != crate::DType::I32
        })
    {
        return Err("sort item pair ABI mismatch".into());
    }
    let source = if let Some(value) = values.get(&(input.index() as u64)) {
        value.clone()
    } else {
        match graph.op(*input).map_err(|error| error.to_string())? {
            Op::Input { name } => inputs
                .get(name)
                .cloned()
                .ok_or_else(|| format!("missing input {name}"))?,
            Op::Constant(value) => value.clone(),
            _ => return Err(format!("missing materialized buffer {}", input.index())),
        }
    };
    if source.shape() != input_shape || source.dtype() != *dtype {
        return Err("sort input descriptor mismatch".into());
    }
    crate::backend::cpu::stable_sort_pair(&source, *axis, *descending)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, ReduceKind, Scalar, Shape, TensorData};

    #[test]
    fn multi_output_schedule_rejects_before_interpreter_materialization() {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::from([1]));
        let output = graph.neg(input).unwrap();
        let mut schedule = crate::schedule(&graph, output).unwrap();
        let primary = schedule.items[0].primary_output().clone();
        let mut secondary = primary.clone();
        secondary.id = 99;
        schedule.items[0].outputs = crate::ScheduledOutputs::new(vec![primary, secondary]).unwrap();
        schedule.items[0].output = schedule.items[0].primary_output().clone();
        schedule.items[0].cache_key = crate::schedule::item_cache_key(&schedule.items[0]).unwrap();
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[output],
                &HashMap::from([("input".into(), TensorData::new([1], vec![1.]).unwrap())]),
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Unsupported(_))
        ));
    }

    #[test]
    fn realizes_reduction_boundary_without_recomputing_shared_producers() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2, 3]), DType::F32);
        let y = graph.input_dtype("y", Shape::from([1, 3]), DType::F32);
        let producer = graph.add(x, y).unwrap();
        let sum = graph
            .reduce(producer, ReduceKind::Mean, Some(vec![1]), false)
            .unwrap();
        let two = graph.constant(TensorData::scalar(2.0));
        let output = graph.mul(sum, two).unwrap();
        let inputs = HashMap::from([
            (
                "x".into(),
                TensorData::from_scalars(
                    [2, 3],
                    DType::F32,
                    (0..6).map(|value| Scalar::F(value as f64)),
                )
                .unwrap(),
            ),
            (
                "y".into(),
                TensorData::from_scalars(
                    [1, 3],
                    DType::F32,
                    [Scalar::F(1.0), Scalar::F(2.0), Scalar::F(3.0)],
                )
                .unwrap(),
            ),
        ]);
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        assert_eq!(schedule.items.len(), 2);
        assert_eq!(schedule.items[1].dependencies, vec![schedule.items[0].id]);
        let actual = realize(
            &graph,
            &schedule,
            &[output],
            &inputs,
            RealizationPolicy::Interpreter,
        )
        .unwrap();
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        assert_eq!(actual.outputs[0].storage(), expected.storage());
        assert_eq!(actual.trace.items.len(), 2);
        assert!(
            actual
                .trace
                .items
                .iter()
                .all(|entry| entry.backend == ItemBackend::Interpreter)
        );
        let fallback = realize(
            &graph,
            &schedule,
            &[output],
            &inputs,
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: true,
            },
        )
        .unwrap();
        assert_eq!(fallback.outputs[0].storage(), expected.storage());
        assert_eq!(fallback.trace.items[1].backend, ItemBackend::JitFallback);
    }

    #[test]
    fn diamond_is_materialized_once_and_native_jit_is_explicit() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2]));
        let producer = graph.square(x).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(producer, one).unwrap();
        let right = graph.mul(producer, one).unwrap();
        let inputs = HashMap::from([("x".into(), TensorData::new([2], vec![2.0, 3.0]).unwrap())]);
        let schedule = crate::schedule_many(&graph, &[left, right]).unwrap();
        assert_eq!(schedule.items.len(), 3);
        assert_eq!(schedule.items[0].consumers.len(), 2);
        assert_eq!(schedule.internal_temporaries(&[left, right]).len(), 1);
        let actual = realize(
            &graph,
            &schedule,
            &[left, right],
            &inputs,
            RealizationPolicy::Interpreter,
        )
        .unwrap();
        for (output, expected_node) in actual.outputs.iter().zip([left, right]) {
            assert_eq!(
                output.storage(),
                CpuBackend
                    .execute(&graph, expected_node, &inputs)
                    .unwrap()
                    .storage()
            );
        }

        let direct = crate::schedule(&graph, producer).unwrap();
        let native = realize(
            &graph,
            &direct,
            &[producer],
            &inputs,
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
        )
        .unwrap();
        assert_eq!(native.trace.items[0].backend, ItemBackend::NativeJit);
    }

    #[test]
    fn malformed_dependencies_and_bindings_fail_before_silent_execution() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2]));
        let y = graph.neg(x).unwrap();
        let inputs = HashMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
        let mut schedule = crate::schedule(&graph, y).unwrap();
        schedule.items[0].dependencies.push(99);
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[y],
                &inputs,
                RealizationPolicy::Interpreter
            ),
            Err(RealizationError::Schedule(_))
        ));

        let mut schedule = crate::schedule(&graph, y).unwrap();
        schedule.items[0].inputs[0].shape = Shape::from([3]);
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[y],
                &inputs,
                RealizationPolicy::Interpreter
            ),
            Err(RealizationError::Schedule(_))
        ));
    }

    #[test]
    fn reuse_is_alias_safe_and_reduces_peak_for_reduction_chain() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2, 1]), DType::F32);
        let y = graph.input_dtype("y", Shape::from([2, 1]), DType::F32);
        let first = graph.square(x).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let reduced = graph.add(first, one).unwrap();
        let branch = graph.neg(first).unwrap();
        let later = graph
            .reduce(y, ReduceKind::Sum, Some(vec![1]), true)
            .unwrap();
        let partial = graph.add(reduced, branch).unwrap();
        let output = graph.add(partial, later).unwrap();
        let inputs = HashMap::from([
            (
                "x".into(),
                TensorData::from_scalars([2, 1], DType::F32, [Scalar::F(2.0), Scalar::F(3.0)])
                    .unwrap(),
            ),
            (
                "y".into(),
                TensorData::from_scalars([2, 1], DType::F32, [Scalar::F(5.0), Scalar::F(7.0)])
                    .unwrap(),
            ),
        ]);
        let requested = [branch, reduced, output];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert!(schedule.items.len() >= 5);
        let disabled = crate::MemoryPlan::from_schedule(&schedule, &requested, false).unwrap();
        let enabled = crate::MemoryPlan::from_schedule(&schedule, &requested, true).unwrap();
        assert!(enabled.peak_allocations < disabled.peak_allocations);
        assert!(enabled.peak_bytes < disabled.peak_bytes);
        let actual = realize_with_options(
            &graph,
            &schedule,
            &requested,
            &inputs,
            RealizationOptions {
                backend: RealizationPolicy::Interpreter,
                memory_reuse: MemoryReuse::Enabled,
            },
        )
        .unwrap();
        let no_reuse = realize_with_options(
            &graph,
            &schedule,
            &requested,
            &inputs,
            RealizationOptions::default(),
        )
        .unwrap();
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        assert_eq!(actual.outputs[2].storage(), expected.storage());
        assert_eq!(actual.outputs[2].storage(), no_reuse.outputs[2].storage());
        let jit = realize_with_options(
            &graph,
            &schedule,
            &requested,
            &inputs,
            RealizationOptions {
                backend: RealizationPolicy::CpuJit {
                    fallback_to_interpreter: true,
                },
                memory_reuse: MemoryReuse::Enabled,
            },
        )
        .unwrap();
        assert_eq!(jit.outputs[2].storage(), expected.storage());
        assert!(
            jit.trace
                .items
                .iter()
                .any(|entry| entry.backend == ItemBackend::JitFallback)
        );
        assert!(
            actual
                .trace
                .items
                .iter()
                .any(|entry| entry.reused_from == Some(first.index() as u64))
        );
        let first_lease = actual
            .trace
            .items
            .iter()
            .find(|entry| entry.materialized_buffer == first.index() as u64)
            .unwrap();
        let reused_lease = actual
            .trace
            .items
            .iter()
            .find(|entry| entry.reused_from == Some(first.index() as u64))
            .unwrap();
        assert_eq!(first_lease.physical_slot, reused_lease.physical_slot);
        assert!(reused_lease.generation > first_lease.generation);
        assert!(
            actual
                .trace
                .items
                .iter()
                .any(|entry| entry.released_buffers.contains(&(first.index() as u64)))
        );
    }
}
