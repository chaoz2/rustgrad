//! Deterministic realization of scheduled UOp items.
pub mod capture;
mod captured_replay;
pub(crate) mod dynamic;
mod mixed;
pub mod mixed_batch;
pub mod mixed_capture;
pub mod mixed_rebinding;
mod persistent_inputs;
mod replay_liveness;
pub(crate) mod symbolic;
pub(crate) mod symbolic_view;
use crate::backend::{JitBackendError, PreparedScheduleItem, TensorValueStore};
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
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
};
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

enum PlannedRealizationItem {
    Interpreter,
    ZeroDomain { cache_hit: bool },
    Native(PreparedScheduleItem),
    Fallback(String),
}

struct PlannedRealization {
    jit: Option<CpuJitBackend>,
    items: Vec<PlannedRealizationItem>,
}

fn realization_jit_error(error: JitBackendError) -> RealizationError {
    match error {
        JitBackendError::Unsupported(reason) => RealizationError::Unsupported(reason),
        other => RealizationError::Execution(other.to_string()),
    }
}

fn fallback_execution_error(native_reason: &str, interpreter_reason: String) -> RealizationError {
    RealizationError::Execution(format!("{native_reason}; {interpreter_reason}"))
}

/// Validates the complete pure topology and, for CPU JIT policies, compiles
/// every admitted item before the first item can execute. This is the ordinary
/// realization counterpart of captured replay's typed preparation phase.
fn plan_realization(
    schedule: &Schedule,
    policy: RealizationPolicy,
) -> Result<PlannedRealization, RealizationError> {
    schedule
        .validate()
        .map_err(|error| RealizationError::Schedule(error.to_string()))?;
    let mut prior = BTreeSet::new();
    for item in &schedule.items {
        if item.boundary.is_some() {
            return Err(RealizationError::Unsupported(format!(
                "item {} has boundary {:?}",
                item.id, item.boundary
            )));
        }
        if item
            .dependencies
            .iter()
            .any(|dependency| !prior.contains(dependency))
        {
            return Err(RealizationError::Schedule(format!(
                "item {} uses a future dependency",
                item.id
            )));
        }
        prior.insert(item.id);
    }

    let RealizationPolicy::CpuJit {
        fallback_to_interpreter,
    } = policy
    else {
        return Ok(PlannedRealization {
            jit: None,
            items: schedule
                .items
                .iter()
                .map(|_| PlannedRealizationItem::Interpreter)
                .collect(),
        });
    };

    let jit = CpuJitBackend::new(JitFallback::Error);
    let mut capability = Vec::with_capacity(schedule.items.len());
    for item in &schedule.items {
        if matches!(item.kernel.operation(), crate::Operation::Sort(_)) {
            let reason = "static sort pairs are CPU-interpreter only".to_string();
            if fallback_to_interpreter {
                capability.push(Err(reason));
                continue;
            }
            return Err(RealizationError::Unsupported(reason));
        }
        if !item.quantized_input_bindings.is_empty() {
            let reason =
                "ordinary realization has no caller-owned packed quantized resources".to_string();
            if fallback_to_interpreter {
                capability.push(Err(reason));
                continue;
            }
            return Err(RealizationError::Unsupported(reason));
        }
        let elements = item
            .primary_output()
            .shape
            .numel()
            .map_err(|error| RealizationError::Schedule(error.to_string()))?;
        if elements == 0 {
            match jit.prepare_zero_domain_schedule_item(item) {
                Ok(cache_hit) => capability.push(Ok(cache_hit)),
                Err(error) if fallback_to_interpreter => capability.push(Err(error.to_string())),
                Err(error) => return Err(realization_jit_error(error)),
            }
            continue;
        }
        match jit.validate_schedule_item(item) {
            Ok(()) => capability.push(Ok(false)),
            Err(error) if fallback_to_interpreter => capability.push(Err(error.to_string())),
            Err(error) => return Err(realization_jit_error(error)),
        }
    }

    let mut items = Vec::with_capacity(schedule.items.len());
    for (item, capability) in schedule.items.iter().zip(capability) {
        if matches!(item.kernel.operation(), crate::Operation::Sort(_))
            || !item.quantized_input_bindings.is_empty()
        {
            let Err(reason) = capability else {
                return Err(RealizationError::Schedule(
                    "fallback-only item unexpectedly passed native capability".into(),
                ));
            };
            items.push(PlannedRealizationItem::Fallback(reason));
            continue;
        }
        if item
            .primary_output()
            .shape
            .numel()
            .map_err(|error| RealizationError::Schedule(error.to_string()))?
            == 0
        {
            match capability {
                Ok(cache_hit) => {
                    items.push(PlannedRealizationItem::ZeroDomain { cache_hit });
                }
                Err(reason) => items.push(PlannedRealizationItem::Fallback(reason)),
            }
            continue;
        }
        if let Err(reason) = capability {
            items.push(PlannedRealizationItem::Fallback(reason));
            continue;
        }
        match jit.prepare_schedule_item(item) {
            Ok(prepared) => items.push(PlannedRealizationItem::Native(prepared)),
            Err(error) if fallback_to_interpreter => {
                items.push(PlannedRealizationItem::Fallback(error.to_string()))
            }
            Err(error) => return Err(realization_jit_error(error)),
        }
    }
    Ok(PlannedRealization {
        jit: Some(jit),
        items,
    })
}

/// Ordinary realization historically permits unrelated caller inputs. Validate
/// only the named Graph inputs that the scheduled ABI actually references,
/// before memory planning, native compilation, or item execution begins.
fn validate_realization_inputs(
    graph: &Graph,
    schedule: &Schedule,
    provided: &HashMap<String, TensorData>,
) -> Result<(), RealizationError> {
    let mut expected = BTreeMap::<String, (Shape, crate::DType)>::new();
    for item in &schedule.items {
        for binding in item.ordered_inputs() {
            let Op::Input { name } = graph
                .op(binding.input_node)
                .map_err(|error| RealizationError::Schedule(error.to_string()))?
            else {
                continue;
            };
            let descriptor = (
                graph
                    .shape(binding.input_node)
                    .map_err(|error| RealizationError::Schedule(error.to_string()))?
                    .clone(),
                graph
                    .dtype(binding.input_node)
                    .map_err(|error| RealizationError::Schedule(error.to_string()))?,
            );
            if expected
                .insert(name.clone(), descriptor.clone())
                .is_some_and(|previous| previous != descriptor)
            {
                return Err(RealizationError::Schedule(format!(
                    "input {name} has conflicting descriptors"
                )));
            }
        }
    }
    for (name, (shape, dtype)) in expected {
        let value = provided
            .get(&name)
            .ok_or_else(|| RealizationError::Execution(format!("missing input {name}")))?;
        if value.shape() != &shape || value.dtype() != dtype {
            return Err(RealizationError::Execution(format!(
                "input {name} descriptor mismatch"
            )));
        }
    }
    Ok(())
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
        !item.outputs.is_single() && !matches!(item.kernel.operation(), crate::Operation::Sort(_))
    }) {
        return Err(RealizationError::Unsupported(
            "multi-output schedule items have no executor lowering".into(),
        ));
    }
    let policy = options.backend;
    validate_realization_inputs(graph, schedule, inputs)?;
    let memory_plan = MemoryPlan::from_schedule(
        schedule,
        requested,
        options.memory_reuse == MemoryReuse::Enabled,
    )
    .map_err(RealizationError::Memory)?;
    // Native compilation is still a pre-execution phase, but only begins
    // after the complete input and allocation plans have succeeded.
    let execution = plan_realization(schedule, policy)?;
    let assignments = memory_plan
        .temporaries
        .iter()
        .map(|entry| (entry.buffer_id, entry))
        .collect::<HashMap<_, _>>();
    let requests = memory_plan
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
    for (item, planned) in schedule.items.iter().zip(&execution.items) {
        let mut backend = ItemBackend::Interpreter;
        let mut lanes = 1;
        let mut vector_main = 0;
        let mut vector_tail = 0;
        let mut vector_reason = "interpreter scalar semantics".to_string();
        let materialized = materialized_values(&leases, &values).map_err(RealizationError::Host)?;
        let sort_pair = if matches!(item.kernel.operation(), crate::Operation::Sort(_)) {
            Some(
                interpret_sort_pair(graph, item, inputs, &materialized)
                    .map_err(RealizationError::Execution)?,
            )
        } else {
            None
        };
        let value = if let Some((values, _)) = &sort_pair {
            if let PlannedRealizationItem::Fallback(reason) = planned {
                backend = ItemBackend::JitFallback;
                vector_reason = reason.clone();
            }
            values.clone()
        } else {
            match planned {
                PlannedRealizationItem::Interpreter => {
                    interpret_item(graph, item, inputs, &materialized)
                        .map_err(RealizationError::Execution)?
                }
                PlannedRealizationItem::ZeroDomain { cache_hit } => {
                    backend = ItemBackend::NativeJit;
                    vector_reason = if *cache_hit {
                        "native zero-domain skip (cache hit)".into()
                    } else {
                        "native zero-domain skip".into()
                    };
                    TensorData::zeros_with_dtype(
                        item.primary_output().shape.clone(),
                        item.primary_output().dtype,
                    )
                    .map_err(|error| RealizationError::Execution(error.to_string()))?
                }
                PlannedRealizationItem::Native(prepared) => {
                    let jit = execution.jit.as_ref().ok_or_else(|| {
                        RealizationError::Schedule("native item has no prepared backend".into())
                    })?;
                    let lookup = RealizationJitValues {
                        graph,
                        inputs,
                        materialized: &materialized,
                    };
                    let (value, native) = jit
                        .execute_prepared_schedule_item(item, &lookup, &BTreeMap::new(), prepared)
                        .map_err(realization_jit_error)?;
                    backend = ItemBackend::NativeJit;
                    lanes = native.vector.lanes;
                    vector_main = native.vector_main;
                    vector_tail = native.vector_tail;
                    vector_reason = native.vector.reason;
                    value
                }
                PlannedRealizationItem::Fallback(reason) => {
                    backend = ItemBackend::JitFallback;
                    vector_reason = reason.clone();
                    interpret_item(graph, item, inputs, &materialized)
                        .map_err(|error| fallback_execution_error(reason, error))?
                }
            }
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
        let released_buffers = memory_plan
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
    if pool.physical_slots().map_err(RealizationError::Host)? != memory_plan.peak_allocations {
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

/// Borrowed ordinary-realization lookup for prepared native schedule items.
/// Compiler-created producers come from the memory plan; external inputs and
/// constants remain graph-owned and are never copied into a second binding
/// map merely to satisfy the captured-replay ABI.
struct RealizationJitValues<'a> {
    graph: &'a Graph,
    inputs: &'a HashMap<String, TensorData>,
    materialized: &'a HashMap<u64, TensorData>,
}

impl TensorValueStore for RealizationJitValues<'_> {
    fn tensor(&self, id: u64, context: &str) -> Result<&TensorData, JitBackendError> {
        if let Some(value) = self.materialized.get(&id) {
            return Ok(value);
        }
        let node = NodeId::from_index(usize::try_from(id).map_err(|_| {
            JitBackendError::Binding(format!("{context}: buffer {id} exceeds NodeId range"))
        })?);
        match self
            .graph
            .op(node)
            .map_err(|error| JitBackendError::Binding(error.to_string()))?
        {
            Op::Input { name } => self.inputs.get(name).ok_or_else(|| {
                JitBackendError::Binding(format!("{context}: missing input {name}"))
            }),
            Op::Constant(value) => Ok(value),
            _ => Err(JitBackendError::Binding(format!(
                "{context}: missing materialized buffer {id}"
            ))),
        }
    }
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

fn is_direct_matmul(item: &crate::ScheduleItem) -> bool {
    matches!(
        item.kernel.operation(),
        crate::Operation::Matmul(
            crate::MatmulValue::Serial(_)
                | crate::MatmulValue::Tiled(_)
                | crate::MatmulValue::TensorCore(_)
        )
    )
}

/// Direct plan executors consume logical tensors rather than UOp address
/// expressions. Resolve their consumer-local affine input view here; generic
/// UOp items must retain the physical tensor because their IndexValue::View
/// applies the same mapping during lane evaluation.
pub(crate) fn direct_matmul_input<'a>(
    item: &crate::ScheduleItem,
    binding: &crate::ScheduleInputBinding,
    value: &'a TensorData,
) -> Result<std::borrow::Cow<'a, TensorData>, String> {
    if !is_direct_matmul(item) {
        return Ok(std::borrow::Cow::Borrowed(value));
    }
    match &binding.desc.view {
        Some(view) => value
            .affine_read(view)
            .map(std::borrow::Cow::Owned)
            .map_err(|error| error.to_string()),
        None => Ok(std::borrow::Cow::Borrowed(value)),
    }
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
    if let crate::Operation::Movement(crate::MovementValue::Plan(plan)) = item.kernel.operation() {
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
        let value = direct_matmul_input(item, binding, &value)?.into_owned();
        let role = if matches!(graph.op(id), Ok(Op::Constant(_))) {
            BufferRole::Constant
        } else {
            BufferRole::Input
        };
        let kernel_desc =
            KernelBufferDesc::concrete(desc.id, role, value.shape().clone(), desc.dtype, false)
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
    let crate::Operation::Sort(crate::SortValue {
        input,
        input_shape,
        axis,
        descending,
        values: value_node,
        indices: index_node,
        dtype,
    }) = item.kernel.operation()
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
    use crate::{
        Backend, CpuBackend, DType, Float8Format, Float8Storage, ReduceKind, Scalar, Shape,
        Storage, TensorData,
    };

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
    fn realizes_single_reduction_epilogue_without_intermediate_storage() {
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
        assert_eq!(schedule.items.len(), 1);
        assert!(schedule.items[0].dependencies.is_empty());
        assert!(
            schedule.items[0]
                .inputs
                .iter()
                .all(|input| input.id != sum.index() as u64)
        );
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
        assert_eq!(actual.trace.items.len(), 1);
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
        assert!(
            fallback
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
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
    fn dependent_contiguous_items_execute_through_the_prepared_cpu_jit_path() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let bias = graph.constant(TensorData::scalar(1.0));
        let computed = graph.add(input, bias).unwrap();
        let output = graph.contiguous(computed).unwrap();
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        assert_eq!(schedule.items.len(), 2);
        assert_eq!(schedule.items[1].dependencies, [schedule.items[0].id]);

        let bindings = HashMap::from([(
            "input".into(),
            TensorData::new([2, 3], vec![-3.0, -0.0, 1.0, 2.0, 5.0, f32::INFINITY]).unwrap(),
        )]);
        let actual = realize(
            &graph,
            &schedule,
            &[output],
            &bindings,
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
        )
        .unwrap();
        let expected = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(actual.outputs[0].storage(), expected.storage());
        assert!(
            actual
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
    }

    #[test]
    fn dependent_contiguous_preserves_raw_views_float8_and_empty_geometry() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let permuted = graph.permute(input, [1, 0]).unwrap();
        let output = graph.contiguous(permuted).unwrap();
        let raw = TensorData::from_storage(
            [2, 2],
            Storage::F32(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_0123),
                f32::INFINITY,
                f32::NEG_INFINITY,
            ]),
        )
        .unwrap();
        let bindings = HashMap::from([("input".into(), raw)]);
        let realized = realize_graph(
            &graph,
            &[output],
            &bindings,
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
        )
        .unwrap();
        let Storage::F32(values) = realized.outputs[0].storage() else {
            panic!("F32 contiguous output")
        };
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0x8000_0000, 0x7f80_0000, 0x7fc0_0123, 0xff80_0000]
        );
        assert!(
            realized
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );

        let mut float8 = Graph::new();
        let input = float8.input_dtype("input", [2, 2], DType::F8E4M3);
        let permuted = float8.permute(input, [1, 0]).unwrap();
        let output = float8.contiguous(permuted).unwrap();
        let raw = TensorData::from_storage(
            [2, 2],
            Storage::Float8(Float8Storage::from_raw(
                Float8Format::E4M3,
                vec![0x00, 0x80, 0x7f, 0xff],
            )),
        )
        .unwrap();
        let realized = realize_graph(
            &float8,
            &[output],
            &HashMap::from([("input".into(), raw)]),
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
        )
        .unwrap();
        let Storage::Float8(values) = realized.outputs[0].storage() else {
            panic!("Float8 contiguous output")
        };
        assert_eq!(values.as_raw(), [0x00, 0x7f, 0x80, 0xff]);

        for shape in [Shape::new([]), Shape::new([0, 3])] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", shape.clone(), DType::F32);
            let zero = graph.constant(TensorData::scalar(0.0));
            let computed = graph.add(input, zero).unwrap();
            let output = graph.contiguous(computed).unwrap();
            let value = TensorData::zeros_with_dtype(shape.clone(), DType::F32).unwrap();
            let realized = realize_graph(
                &graph,
                &[output],
                &HashMap::from([("input".into(), value)]),
                RealizationPolicy::CpuJit {
                    fallback_to_interpreter: false,
                },
            )
            .unwrap();
            assert_eq!(realized.outputs[0].shape(), &shape);
            assert!(
                realized
                    .trace
                    .items
                    .iter()
                    .all(|item| item.backend == ItemBackend::NativeJit)
            );
        }
    }

    #[test]
    fn dependent_native_plan_rejects_late_topology_and_descriptor_faults() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let one = graph.constant(TensorData::scalar(1.0));
        let computed = graph.add(input, one).unwrap();
        let output = graph.contiguous(computed).unwrap();
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();

        let mut malformed_edge = schedule.clone();
        malformed_edge.items[1].dependencies.clear();
        malformed_edge.items[1].cache_key =
            crate::schedule::item_cache_key(&malformed_edge.items[1]).unwrap();
        assert!(matches!(
            plan_realization(
                &malformed_edge,
                RealizationPolicy::CpuJit {
                    fallback_to_interpreter: false,
                },
            ),
            Err(RealizationError::Schedule(_))
        ));

        let mut malformed_desc = schedule;
        malformed_desc.items[1].input_bindings[0].desc.dtype = DType::I32;
        malformed_desc.items[1].cache_key =
            crate::schedule::item_cache_key(&malformed_desc.items[1]).unwrap();
        assert!(matches!(
            plan_realization(
                &malformed_desc,
                RealizationPolicy::CpuJit {
                    fallback_to_interpreter: false,
                },
            ),
            Err(RealizationError::Schedule(_))
        ));
    }

    #[test]
    fn fallback_failure_retains_native_and_interpreter_diagnostics() {
        assert_eq!(
            fallback_execution_error(
                "CPU JIT unsupported: operation family",
                "missing materialized buffer 7".into(),
            ),
            RealizationError::Execution(
                "CPU JIT unsupported: operation family; missing materialized buffer 7".into(),
            )
        );
    }

    #[test]
    fn all_external_inputs_are_validated_before_dependent_native_preparation() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F32);
        let y = graph.input_dtype("y", [2], DType::F32);
        let first = graph.square(x).unwrap();
        let second = graph.add(first, y).unwrap();
        let output = graph.contiguous(second).unwrap();
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        assert!(schedule.items.len() >= 2);

        let missing = HashMap::from([("x".into(), TensorData::new([2], vec![2.0, 3.0]).unwrap())]);
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[output],
                &missing,
                RealizationPolicy::CpuJit {
                    fallback_to_interpreter: false,
                },
            ),
            Err(RealizationError::Execution(reason)) if reason == "missing input y"
        ));

        let mismatch = HashMap::from([
            ("x".into(), TensorData::new([2], vec![2.0, 3.0]).unwrap()),
            ("y".into(), TensorData::new([1], vec![1.0]).unwrap()),
            // Preserve ordinary realization's historical extra-input policy.
            ("unused".into(), TensorData::scalar(7.0)),
        ]);
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[output],
                &mismatch,
                RealizationPolicy::CpuJit {
                    fallback_to_interpreter: false,
                },
            ),
            Err(RealizationError::Execution(reason))
                if reason == "input y descriptor mismatch"
        ));
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
        let y = graph.input_dtype("y", Shape::from([2, 2]), DType::F32);
        let first = graph.square(x).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let reduced = graph.add(first, one).unwrap();
        let branch = graph.neg(first).unwrap();
        let later_reduction = graph
            .reduce(y, ReduceKind::Sum, Some(vec![1]), true)
            .unwrap();
        // Keep this memory-planning fixture focused on two materialized
        // lifetimes. The explicit copy is an observable boundary, so reduction
        // epilogue fusion must not erase the temporary whose reuse is tested.
        let later = graph.contiguous(later_reduction).unwrap();
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
                TensorData::from_scalars(
                    [2, 2],
                    DType::F32,
                    [
                        Scalar::F(5.0),
                        Scalar::F(1.0),
                        Scalar::F(7.0),
                        Scalar::F(1.0),
                    ],
                )
                .unwrap(),
            ),
        ]);
        let requested = [branch, reduced, output];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert!(schedule.items.len() >= 4);
        assert!(
            schedule
                .items
                .iter()
                .any(|item| item.node == later_reduction)
        );
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
                .all(|entry| entry.backend == ItemBackend::NativeJit)
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
