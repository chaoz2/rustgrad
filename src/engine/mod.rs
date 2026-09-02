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
mod symbolic_program;
pub(crate) mod symbolic_projected;
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
pub(crate) use symbolic_program::{AuthenticatedSymbolicBody, AuthenticatedSymbolicInvocation};
pub use symbolic_program::{
    CpuSymbolicInvocation, CpuSymbolicProgram, CpuSymbolicResult, CpuSymbolicTrace,
    SymbolicInvocation,
};

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

/// Checked ordered output inventory for one ordinary realization transaction.
///
/// Scheduled values retain their producer-owned buffers. Requested graph
/// inputs and constants have no producer item, so the transaction retains
/// their exact owned storage here. Requested static affine aliases project an
/// immutable source or retained scheduled producer without fabricating output
/// ownership. The ordered IDs deliberately preserve duplicate requests
/// without scheduling or materializing the value twice.
struct RequestedOutputPlan {
    ordered: Vec<u64>,
    retained_sources: HashMap<u64, TensorData>,
    aliases: BTreeMap<u64, crate::RequestedPassthrough>,
}

impl RequestedOutputPlan {
    fn new(
        graph: &Graph,
        schedule: &Schedule,
        requested: &[NodeId],
        inputs: &HashMap<String, TensorData>,
    ) -> Result<Self, RealizationError> {
        let produced = schedule
            .items
            .iter()
            .flat_map(|item| {
                item.outputs
                    .iter()
                    .map(move |output| (output.id, (item, output)))
            })
            .collect::<BTreeMap<_, _>>();
        let passthroughs = schedule
            .requested_passthroughs
            .iter()
            .map(|passthrough| (passthrough.requested.index() as u64, passthrough))
            .collect::<BTreeMap<_, _>>();
        let mut ordered = Vec::with_capacity(requested.len());
        let mut retained_sources = HashMap::new();
        let mut aliases = BTreeMap::new();
        for &node in requested {
            let shape = graph
                .shape(node)
                .map_err(|error| RealizationError::Schedule(error.to_string()))?
                .clone();
            let dtype = graph
                .dtype(node)
                .map_err(|error| RealizationError::Schedule(error.to_string()))?;
            let elements = shape
                .numel()
                .map_err(|error| RealizationError::Schedule(error.to_string()))?;
            let bytes = elements.checked_mul(dtype.itemsize()).ok_or_else(|| {
                RealizationError::Schedule("requested output byte overflow".into())
            })?;
            let id = node.index() as u64;
            ordered.push(id);
            let op = graph
                .op(node)
                .map_err(|error| RealizationError::Schedule(error.to_string()))?;
            if let Some(passthrough) = passthroughs.get(&id) {
                if produced.contains_key(&id) {
                    return Err(RealizationError::Schedule(format!(
                        "requested passthrough {id} is shadowed by a scheduled producer"
                    )));
                }
                passthrough
                    .validate_against_graph(graph)
                    .map_err(|error| RealizationError::Schedule(error.to_string()))?;
                let source = match graph
                    .op(passthrough.source)
                    .map_err(|error| RealizationError::Schedule(error.to_string()))?
                {
                    Op::Input { name } => Some(inputs.get(name).ok_or_else(|| {
                        RealizationError::Execution(format!("missing input {name}"))
                    })?),
                    Op::Constant(value) => Some(value),
                    _ => None,
                };
                if let Some(source) = source {
                    let mut physical = passthrough.desc.clone();
                    physical.view = None;
                    if source.shape() != &physical.shape || source.dtype() != physical.dtype {
                        return Err(RealizationError::Execution(format!(
                            "requested passthrough {id} source descriptor mismatch"
                        )));
                    }
                    retained_sources
                        .entry(passthrough.source.index() as u64)
                        .or_insert_with(|| source.clone());
                } else {
                    let source_id = passthrough.source.index() as u64;
                    let Some((item, output)) = produced.get(&source_id) else {
                        return Err(RealizationError::Schedule(format!(
                            "requested passthrough {id} computed source has no producer"
                        )));
                    };
                    let owns_source = match item.kernel.operation() {
                        crate::Operation::Sort(_) => {
                            canonical_sort_item_owns(graph, item, passthrough.source)?
                        }
                        _ => item.node == passthrough.source,
                    };
                    let mut physical = passthrough.desc.clone();
                    physical.view = None;
                    physical.read_only = false;
                    if !owns_source || *output != &physical {
                        return Err(RealizationError::Schedule(format!(
                            "requested passthrough {id} computed source is inconsistent"
                        )));
                    }
                }
                aliases.insert(id, (*passthrough).clone());
                continue;
            }
            if matches!(op, Op::Input { .. } | Op::Constant(_)) {
                if produced.contains_key(&id) {
                    return Err(RealizationError::Schedule(format!(
                        "requested source {id} is shadowed by a scheduled producer"
                    )));
                }
                let value = match op {
                    Op::Input { name } => inputs.get(name).ok_or_else(|| {
                        RealizationError::Execution(format!("missing input {name}"))
                    })?,
                    Op::Constant(value) => value,
                    _ => unreachable!("source operation was classified above"),
                };
                if value.shape() != &shape || value.dtype() != dtype {
                    return Err(RealizationError::Execution(format!(
                        "requested value {id} descriptor mismatch"
                    )));
                }
                retained_sources.entry(id).or_insert_with(|| value.clone());
                continue;
            }
            if let Some((item, output)) = produced.get(&id) {
                let owns_output = match item.kernel.operation() {
                    crate::Operation::Sort(_) => canonical_sort_item_owns(graph, item, node)?,
                    _ => item.node == node,
                };
                if !owns_output {
                    return Err(RealizationError::Schedule(format!(
                        "requested value {id} has a graph-inconsistent scheduled producer"
                    )));
                }
                if output.shape != shape
                    || output.dtype != dtype
                    || output.bytes != bytes
                    || output.alignment != dtype.itemsize().max(1)
                    || output.view.is_some()
                    || output.read_only
                {
                    return Err(RealizationError::Schedule(format!(
                        "requested output {id} descriptor mismatch"
                    )));
                }
                continue;
            }
            return Err(RealizationError::Schedule(format!(
                "requested value {id} has no scheduled producer"
            )));
        }
        Ok(Self {
            ordered,
            retained_sources,
            aliases,
        })
    }

    fn project(
        &self,
        values: &HashMap<u64, TensorData>,
    ) -> Result<Vec<TensorData>, RealizationError> {
        self.ordered
            .iter()
            .map(|id| {
                if let Some(alias) = self.aliases.get(id) {
                    let source = values
                        .get(&(alias.source.index() as u64))
                        .ok_or(RealizationError::MissingBuffer(alias.source.index() as u64))?;
                    return alias
                        .project(source)
                        .map_err(|error| RealizationError::Execution(error.to_string()));
                }
                values
                    .get(id)
                    .cloned()
                    .ok_or(RealizationError::MissingBuffer(*id))
            })
            .collect()
    }
}

fn canonical_sort_item_owns(
    graph: &Graph,
    item: &crate::ScheduleItem,
    requested: NodeId,
) -> Result<bool, RealizationError> {
    let crate::Operation::Sort(payload) = item.kernel.operation() else {
        return Ok(false);
    };
    let plan =
        crate::CpuStableSortPlan::from_graph(graph, payload.input, payload.values, payload.indices)
            .map_err(|error| {
                RealizationError::Schedule(format!(
                    "requested Sort producer is not canonical: {error}"
                ))
            })?;
    let descriptor_matches =
        |actual: &crate::BufferDesc, expected: &crate::CpuStableSortDescriptor, read_only: bool| {
            actual.id == expected.node.index() as u64
                && actual.shape == expected.shape
                && actual.dtype == expected.dtype
                && actual.bytes == expected.bytes
                && actual.alignment == expected.dtype.itemsize().max(1)
                && actual.read_only == read_only
                && actual.view.is_none()
        };
    let owner_matches = matches!(
        graph.op(item.node),
        Ok(Op::Sort {
            input,
            axis,
            descending,
            pair,
            output: crate::SortOutput::Values,
        }) if *input == plan.source().node
            && *axis == plan.axis()
            && *descending == plan.descending()
            && *pair == plan.pair()
            && item.node == plan.values().node
    );
    let payload_matches = payload.input == plan.source().node
        && payload.input_shape == plan.source().shape
        && payload.axis == plan.axis()
        && payload.descending == plan.descending()
        && payload.values == plan.values().node
        && payload.indices == plan.indices().node
        && payload.dtype == plan.source().dtype;
    let input_matches = item.ordered_inputs().len() == 1
        && item.ordered_inputs()[0].input_node == plan.source().node
        && item.ordered_inputs()[0].abi_index == 0
        && descriptor_matches(&item.ordered_inputs()[0].desc, plan.source(), true);
    let outputs_match = item.outputs.len() == 2
        && descriptor_matches(item.outputs.primary(), plan.values(), false)
        && item
            .outputs
            .iter()
            .nth(1)
            .is_some_and(|output| descriptor_matches(output, plan.indices(), false));
    if !owner_matches || !payload_matches || !input_matches || !outputs_match {
        return Err(RealizationError::Schedule(
            "requested Sort producer diverges from its canonical graph pair".into(),
        ));
    }
    Ok(requested == plan.values().node || requested == plan.indices().node)
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
    let mut requested_plan = RequestedOutputPlan::new(graph, schedule, requested, inputs)?;
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
    let requested_buffers = requested_plan
        .ordered
        .iter()
        .copied()
        .chain(
            requested_plan
                .aliases
                .values()
                .map(|alias| alias.source.index() as u64),
        )
        .collect::<std::collections::BTreeSet<_>>();
    // Only retained outputs live here. Internal values are reachable solely
    // through non-cloneable, generation-checked pool leases.
    let mut values = std::mem::take(&mut requested_plan.retained_sources);
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
    let outputs = requested_plan.project(&values)?;
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
    fn cpu_batch_execution_preserves_order_duplicates_and_source_storage() {
        let mut graph = Graph::new();
        let source = graph.input_dtype("source", Shape::from([2]), DType::BF16);
        let source_f32 = graph.cast(source, DType::F32).unwrap();
        let shared = graph.square(source_f32).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(shared, one).unwrap();
        let right = graph.mul(shared, one).unwrap();
        let empty = graph.input_dtype("empty", Shape::from([0]), DType::U32);
        let raw_source =
            TensorData::from_storage([2], Storage::BF16(vec![0x8000, 0x7fc1])).unwrap();
        let raw_empty = TensorData::from_storage([0], Storage::U32(vec![])).unwrap();
        let inputs = HashMap::from([
            ("source".into(), raw_source.clone()),
            ("empty".into(), raw_empty.clone()),
        ]);
        let direct = CpuBackend
            .execute_many(&graph, &[source, empty, one, source], &inputs)
            .unwrap();
        assert!(direct.trace.items.is_empty());
        assert_eq!(direct.outputs[0].storage(), raw_source.storage());
        assert_eq!(direct.outputs[1].storage(), raw_empty.storage());
        assert_eq!(direct.outputs[0].storage(), direct.outputs[3].storage());
        let requested = [right, source, left, right, empty, one];

        let realized = CpuBackend
            .execute_many(&graph, &requested, &inputs)
            .unwrap();
        assert_eq!(realized.outputs.len(), requested.len());
        assert_eq!(realized.outputs[1].storage(), raw_source.storage());
        assert_eq!(realized.outputs[4].storage(), raw_empty.storage());
        assert_eq!(
            realized.outputs[5].storage(),
            TensorData::scalar(1.0).storage()
        );
        let f32_lane_bits = |value: &TensorData| {
            let Storage::F32(lanes) = value.storage() else {
                panic!("expected F32 batch output")
            };
            lanes.iter().map(|lane| lane.to_bits()).collect::<Vec<_>>()
        };
        let right_bits = f32_lane_bits(&realized.outputs[0]);
        assert_eq!(right_bits, f32_lane_bits(&realized.outputs[3]));
        let single_right = CpuBackend.execute(&graph, right, &inputs).unwrap();
        assert_eq!(right_bits, f32_lane_bits(&single_right));
        assert_eq!(right_bits[0], 0.0f32.to_bits());
        assert!(f32::from_bits(right_bits[1]).is_nan());
        let single_left = CpuBackend.execute(&graph, left, &inputs).unwrap();
        assert_eq!(
            f32_lane_bits(&realized.outputs[2]),
            f32_lane_bits(&single_left)
        );

        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert_eq!(realized.trace.items.len(), schedule.items.len());
        assert_eq!(
            realized
                .trace
                .items
                .iter()
                .filter(|item| item.materialized_buffer == shared.index() as u64)
                .count(),
            1,
            "the shared producer is executed once"
        );

        // Durable capture keeps its established unique-request contract; this
        // runtime-only duplicate projection does not alter artifact identity.
        let captured_requested = [right, source, left, empty, one];
        let captured_schedule = crate::schedule_many(&graph, &captured_requested).unwrap();
        let capture =
            crate::CapturedSchedule::capture(&graph, &captured_schedule, &captured_requested)
                .unwrap();
        let decoded = crate::CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        assert_eq!(decoded.requested, capture.requested);
        assert_eq!(decoded.identity, capture.identity);
    }

    #[test]
    fn source_affine_passthrough_realizes_in_order_without_a_fake_producer() {
        let mut graph = Graph::new();
        let source = graph.input_dtype("source", Shape::from([2, 3]), DType::F32);
        let transposed = graph.permute(source, [1, 0]).unwrap();
        let computed = graph.neg(transposed).unwrap();
        let requested = [transposed, computed, transposed];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert_eq!(schedule.items.len(), 1);
        assert_eq!(schedule.requested_passthroughs.len(), 1);
        assert!(
            schedule
                .items
                .iter()
                .flat_map(|item| item.outputs.iter())
                .all(|output| output.id != transposed.index() as u64)
        );
        let source_value = TensorData::from_storage(
            [2, 3],
            Storage::F32(vec![
                -0.0,
                f32::from_bits(0x7fc0_1234),
                2.0,
                f32::INFINITY,
                -3.0,
                f32::NEG_INFINITY,
            ]),
        )
        .unwrap();
        let bindings = HashMap::from([("source".into(), source_value)]);
        let realized = realize(
            &graph,
            &schedule,
            &requested,
            &bindings,
            RealizationPolicy::Interpreter,
        )
        .unwrap();
        let lane_bits = |value: &TensorData| {
            let Storage::F32(lanes) = value.storage() else {
                panic!("F32 fixture")
            };
            lanes.iter().map(|lane| lane.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(
            lane_bits(&realized.outputs[0]),
            vec![
                (-0.0f32).to_bits(),
                f32::INFINITY.to_bits(),
                0x7fc0_1234,
                (-3.0f32).to_bits(),
                2.0f32.to_bits(),
                f32::NEG_INFINITY.to_bits(),
            ]
        );
        assert_eq!(
            lane_bits(&realized.outputs[0]),
            lane_bits(&realized.outputs[2])
        );
        assert_eq!(realized.trace.items.len(), 1);

        let mut wrong_dtype = schedule.clone();
        wrong_dtype.requested_passthroughs[0].desc.dtype = DType::I32;
        assert!(matches!(
            realize(
                &graph,
                &wrong_dtype,
                &requested,
                &bindings,
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Schedule(_))
        ));

        let mut malformed = schedule;
        malformed.requested_passthroughs[0].source = transposed;
        assert!(matches!(
            realize(
                &graph,
                &malformed,
                &requested,
                &bindings,
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Schedule(_))
        ));
    }

    #[test]
    fn computed_affine_alias_projects_after_its_one_retained_producer() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3, 3], DType::F32);
        let diagonal = graph.diagonal_default(input).unwrap();
        let requested = [diagonal, diagonal];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert_eq!(schedule.items.len(), 2);
        assert_eq!(schedule.requested_passthroughs.len(), 1);
        let source = schedule.requested_passthroughs[0].source;
        assert!(schedule.items.iter().any(|item| item.node == source));
        let memory = crate::MemoryPlan::from_schedule(&schedule, &requested, true).unwrap();
        assert_eq!(memory.temporaries.len(), 1);
        assert_ne!(memory.temporaries[0].buffer_id, source.index() as u64);

        let bindings = HashMap::from([(
            "input".into(),
            TensorData::from_storage(
                [3, 3],
                Storage::F32(vec![
                    -0.0,
                    2.0,
                    3.0,
                    4.0,
                    f32::from_bits(0x7fc0_1234),
                    6.0,
                    7.0,
                    8.0,
                    f32::NEG_INFINITY,
                ]),
            )
            .unwrap(),
        )]);
        let mut realized = realize_with_options(
            &graph,
            &schedule,
            &requested,
            &bindings,
            RealizationOptions {
                backend: RealizationPolicy::Interpreter,
                memory_reuse: MemoryReuse::Enabled,
            },
        )
        .unwrap();
        assert_eq!(realized.trace.items.len(), 2);
        let expected = vec![
            (-0.0f32).to_bits(),
            0x7fc0_1234,
            f32::NEG_INFINITY.to_bits(),
        ];
        for output in &realized.outputs {
            let Storage::F32(values) = output.storage() else {
                panic!("computed diagonal alias must retain F32 storage")
            };
            assert_eq!(
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
        }
        realized.outputs[0]
            .replace(&TensorData::new([3], vec![1.0, 2.0, 3.0]).unwrap())
            .unwrap();
        let Storage::F32(second) = realized.outputs[1].storage() else {
            unreachable!("fixture is F32")
        };
        assert_eq!(second[0].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn requested_source_descriptors_preflight_before_batch_execution() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2]), DType::F32);
        let y = graph.input_dtype("y", Shape::from([2]), DType::F32);
        // This output would fail at execution, but it depends only on x. The
        // separately requested direct y descriptor must reject first.
        let faulting = graph.tensor_guard_distribution(x, 0).unwrap();
        let schedule = crate::schedule_many(&graph, &[faulting, y]).unwrap();
        let invalid_x = TensorData::new([2], vec![1.0, -1.0]).unwrap();
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[faulting, y],
                &HashMap::from([("x".into(), invalid_x.clone())]),
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Execution(reason)) if reason == "missing input y"
        ));
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[faulting, y],
                &HashMap::from([
                    ("x".into(), invalid_x),
                    (
                        "y".into(),
                        TensorData::from_storage([2], Storage::U32(vec![1, 2])).unwrap(),
                    ),
                ]),
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Execution(reason))
                if reason == format!("requested value {} descriptor mismatch", y.index())
        ));

        let computed = graph.neg(x).unwrap();
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[computed],
                &HashMap::from([(
                    "x".into(),
                    TensorData::new([2], vec![1.0, 2.0]).unwrap(),
                )]),
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Schedule(reason))
                if reason.contains("has no scheduled producer")
        ));

        // A structurally valid Store schedule is still not allowed to claim a
        // graph-owned source ID. Requested source classification precedes the
        // producer lookup, so the caller's exact y binding cannot be shadowed.
        let stored = graph.square(x).unwrap();
        let mut shadow = crate::schedule_many(&graph, &[stored]).unwrap();
        let item = &mut shadow.items[0];
        let kernel_sources = item.kernel.sources().to_vec();
        item.kernel = crate::UOp::sink(
            kernel_sources
                .iter()
                .map(|source| {
                    if !matches!(source.operation(), crate::Operation::Store) {
                        return source.clone();
                    }
                    let [index, value] = source.sources() else {
                        panic!("scheduled Store has two sources")
                    };
                    let crate::Operation::Index(crate::IndexValue::Buffer {
                        elements,
                        input_shape,
                        output_shape,
                        ..
                    }) = index.operation()
                    else {
                        panic!("scheduled Store has a dense output index")
                    };
                    let address = crate::UOp::from_operation(
                        crate::Operation::DefineGlobal(crate::AddressValue {
                            space: crate::AddressSpace::Global,
                            name: format!("b{}", y.index()),
                            element: crate::UType::scalar(DType::F32),
                        }),
                        Some(crate::UType::scalar(DType::F32)),
                        vec![],
                    );
                    let index = crate::UOp::from_operation(
                        crate::Operation::Index(crate::IndexValue::Buffer {
                            buffer: y.index() as u64,
                            elements: *elements,
                            input_shape: input_shape.clone(),
                            output_shape: output_shape.clone(),
                            addressing: crate::IndexAddressing::Broadcast,
                        }),
                        Some(crate::UType::scalar(DType::F32)),
                        vec![address, index.sources()[1].clone()],
                    );
                    crate::UOp::from_operation(
                        crate::Operation::Store,
                        None,
                        vec![index, value.clone()],
                    )
                })
                .collect(),
        );
        let mut shadow_output = item.primary_output().clone();
        shadow_output.id = y.index() as u64;
        item.outputs = crate::ScheduledOutputs::single(shadow_output);
        item.cache_key = crate::schedule::item_cache_key(item).unwrap();
        shadow.validate().unwrap();
        assert!(matches!(
            realize(
                &graph,
                &shadow,
                &[y],
                &HashMap::from([
                    ("x".into(), TensorData::new([2], vec![2.0, 3.0]).unwrap()),
                    ("y".into(), TensorData::new([2], vec![5.0, 7.0]).unwrap()),
                ]),
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Schedule(reason))
                if reason == format!(
                    "requested source {} is shadowed by a scheduled producer",
                    y.index()
                )
        ));
    }

    #[test]
    fn requested_sort_secondary_requires_the_canonical_graph_pair() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([3]), DType::F32);
        let (values, indices) = graph.sort(input, 0, false).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            TensorData::new([3], vec![2.0, 1.0, 3.0]).unwrap(),
        )]);
        let canonical = crate::schedule_many(&graph, &[indices]).unwrap();
        let realized = realize(
            &graph,
            &canonical,
            &[indices],
            &bindings,
            RealizationPolicy::Interpreter,
        )
        .unwrap();
        assert_eq!(realized.outputs[0].to_vec_f64(), vec![1.0, 0.0, 2.0]);

        // Keep a structurally valid, rekeyed coupled Sort item, but swap its
        // payload and output descriptors to a different graph pair while the
        // item retains the original canonical values owner.
        let (other_values, other_indices) = graph.sort(input, 0, true).unwrap();
        let other = crate::schedule_many(&graph, &[other_indices]).unwrap();
        let mut tampered = crate::schedule_many(&graph, &[indices]).unwrap();
        tampered.items[0].kernel = other.items[0].kernel.clone();
        tampered.items[0].outputs = other.items[0].outputs.clone();
        tampered.items[0].cache_key = crate::schedule::item_cache_key(&tampered.items[0]).unwrap();
        tampered.validate().unwrap();
        assert_eq!(tampered.items[0].node, values);
        assert_ne!(tampered.items[0].node, other_values);
        assert!(matches!(
            realize(
                &graph,
                &tampered,
                &[other_indices],
                &bindings,
                RealizationPolicy::Interpreter,
            ),
            Err(RealizationError::Schedule(reason))
                if reason.contains("canonical graph pair")
        ));
    }

    #[test]
    fn dependent_contiguous_items_execute_through_the_prepared_cpu_jit_path() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let bias = graph.constant(TensorData::scalar(1.0));
        let computed = graph.add(input, bias).unwrap();
        let output = graph.contiguous(computed).unwrap();
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        assert_eq!(schedule.items.len(), 1);
        assert_eq!(schedule.items[0].node, output);
        assert!(schedule.items[0].dependencies.is_empty());
        assert!(matches!(
            schedule.items[0].kernel.operation(),
            crate::Operation::Sink
        ));

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
        let schedule = crate::schedule_many(&graph, &[computed, output]).unwrap();
        assert_eq!(schedule.items.len(), 2);

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
        let schedule = crate::schedule_many(&graph, &[first, output]).unwrap();
        assert_eq!(schedule.items.len(), 2);

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
