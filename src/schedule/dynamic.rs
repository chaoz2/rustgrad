//! Runtime-sized schedule ABI for exact CPU dynamic cardinality results.
//!
//! This is deliberately separate from the fixed-`Shape` `ScheduleItem` ABI:
//! a dynamic buffer cannot masquerade as a static `BufferDesc`.  It remains a
//! crate-private schedule branch until ordinary schedule/capture artifacts can
//! retain and validate runtime-sized buffers without placeholders.

use crate::ir::{DynamicInput, DynamicOp};
use crate::{
    BinaryOp, BufferDesc, DType, DynamicAllocation, DynamicAllocationError, DynamicAllocationPlan,
    DynamicAllocationTarget, DynamicBinding, DynamicCountStage, DynamicNodeId, Graph, NodeId,
    Schedule, Shape, UnaryOp,
};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

/// Stable logical identity of one runtime-sized result buffer. It is derived
/// from the immutable allocation plan rather than a host allocation or value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeBufferId(pub u64);

/// A buffer with known dtype/rank and static count bindings, but no logical
/// shape or allocation until its preceding count item has completed.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RuntimeBufferDesc {
    pub id: RuntimeBufferId,
    pub dtype: DType,
    pub rank: usize,
    pub count_stage: DynamicCountStage,
    pub bindings: Vec<DynamicBinding>,
    pub plan_identity: u64,
}

/// Explicit items in the exact runtime-sized CPU contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RuntimeScheduleItemKind {
    Count {
        stage: DynamicCountStage,
        bindings: Vec<DynamicBinding>,
    },
    Allocate {
        output: RuntimeBufferDesc,
    },
    MaterializeMaskedSelect {
        output: RuntimeBufferDesc,
    },
    AllocateUnary {
        output: RuntimeBufferDesc,
    },
    DynamicUnary {
        op: UnaryOp,
        input: RuntimeBufferDesc,
        output: RuntimeBufferDesc,
    },
    AllocateBinary {
        output: RuntimeBufferDesc,
    },
    /// A bounded runtime-preserving binary. The static operand is an exact
    /// scalar ABI binding; it is never inferred from a label or descriptor.
    DynamicBinary {
        op: BinaryOp,
        input: RuntimeBufferDesc,
        static_input: StaticScalarBinding,
        output: RuntimeBufferDesc,
    },
    /// The one permitted runtime-to-fixed bridge.  The scalar descriptor is
    /// immutable and never makes a generic fixed item dynamic-capable.
    DynamicReduceSum {
        input: RuntimeBufferDesc,
        output: BufferDesc,
    },
    DynamicReduceMean {
        input: RuntimeBufferDesc,
        output: BufferDesc,
    },
}

/// An immutable item in canonical count-then-allocation order.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RuntimeScheduleItem {
    pub id: u64,
    pub dependencies: Vec<u64>,
    pub kind: RuntimeScheduleItemKind,
    pub cache_key: u64,
}

/// The canonical ordered runtime-buffer schedule. It is not a second planner:
/// construction consumes the graph-owned `DynamicAllocationPlan` and may add
/// one explicitly typed dynamic-capable pure consumer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeSchedule {
    plan: DynamicAllocationPlan,
    pub items: Vec<RuntimeScheduleItem>,
    pub output: RuntimeBufferDesc,
    pub buffers: Vec<RuntimeBufferDesc>,
    pub fixed_output: Option<BufferDesc>,
    pub lifetimes: Vec<crate::memory_plan::RuntimeAllocationLifetime>,
    pub identity: u64,
}

/// One output descriptor in the canonical mixed schedule DAG. A runtime
/// descriptor remains distinct from `BufferDesc` until its count dependency
/// has produced an exact allocation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ScheduledOutputDesc {
    Fixed(BufferDesc),
    Runtime(RuntimeBufferDesc),
}

/// One item in a schedule DAG that may contain existing fixed-shape work and
/// the exact runtime count/allocation pair. Fixed items retain their original
/// logical cache keys verbatim.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum MixedScheduleItemKind {
    Fixed {
        source_item: u64,
    },
    Count {
        stage: DynamicCountStage,
        bindings: Vec<DynamicBinding>,
    },
    Allocate,
    MaterializeMaskedSelect,
    AllocateUnary,
    DynamicUnary {
        op: UnaryOp,
    },
    AllocateBinary,
    DynamicBinary {
        op: BinaryOp,
    },
    DynamicReduceSum,
    DynamicReduceMean,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MixedScheduleItem {
    pub id: u64,
    pub dependencies: Vec<u64>,
    pub consumers: Vec<u64>,
    pub output: ScheduledOutputDesc,
    pub kind: MixedScheduleItemKind,
    pub cache_key: u64,
}

/// A validated scalar static operand. This is deliberately distinct from a
/// runtime descriptor: its shape and bytes are fixed before the count stage.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StaticScalarBinding {
    pub node: NodeId,
    pub descriptor: BufferDesc,
}

/// One ordered source in a runtime-capable consumer ABI.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RuntimeValueSource {
    Runtime {
        source: RuntimeBufferId,
        source_desc: RuntimeBufferDesc,
    },
    StaticScalar(StaticScalarBinding),
}

/// One ordered value edge. It is the sole ABI by which a runtime buffer or
/// declared static scalar may reach a dynamic-capable consumer; no consumer
/// infers operands from a buffer ID, node ordering, or label.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RuntimeValueBinding {
    pub source: RuntimeValueSource,
    pub consumer_item: u64,
    pub abi_index: usize,
}

/// Private canonical DAG envelope joining static `ScheduleItem` records and
/// runtime-sized records. It owns no alternative planner or cache: fixed items
/// retain their ordinary schedule keys and runtime items retain the allocation
/// plan identities from which they were lowered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MixedSchedule {
    runtime: RuntimeSchedule,
    pub items: Vec<MixedScheduleItem>,
    pub runtime_bindings: Vec<RuntimeValueBinding>,
    pub lifetimes: Vec<crate::memory_plan::RuntimeAllocationLifetime>,
    pub identity: u64,
}

/// Runtime allocation metadata remains absent until the count stage has
/// completed. No tensor value or bounded placeholder is stored here.
#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeBufferTable {
    /// Canonical allocation order; lookup never infers a source from map
    /// iteration or a runtime value.
    descriptors: Vec<RuntimeBufferDesc>,
    allocations: BTreeMap<RuntimeBufferId, DynamicAllocation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeScheduleError {
    Plan(DynamicAllocationError),
    InvalidOrdering(&'static str),
    DuplicateBuffer(RuntimeBufferId),
    UnknownBuffer(RuntimeBufferId),
    LiveLookupBeforeAllocation(RuntimeBufferId),
    DuplicateAllocation(RuntimeBufferId),
    StaticConsumerRuntimeInput { consumer: u64, dependency: u64 },
    UnknownItem(u64),
    ExpectedRuntimeOutput(u64),
}

impl fmt::Display for RuntimeScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "runtime schedule error: {self:?}")
    }
}
impl std::error::Error for RuntimeScheduleError {}

/// Builds the only currently supported runtime-sized schedule: a CPU exact
/// `masked_select_dynamic` count followed by allocation.
pub(crate) fn schedule_dynamic(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<MixedSchedule, RuntimeScheduleError> {
    MixedSchedule::from_static_and_runtime(
        &empty_fixed_schedule(),
        schedule_runtime(graph, output)?,
    )
}

/// Lowers the only currently dynamic-capable pure consumer.  The unary result
/// has its own exact allocation: it never aliases the masked-select storage.
pub(crate) fn schedule_dynamic_unary(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<MixedSchedule, RuntimeScheduleError> {
    MixedSchedule::from_static_and_runtime(
        &empty_fixed_schedule(),
        schedule_runtime_unary(graph, output)?,
    )
}

/// Lowers the bounded runtime-preserving binary consumer.  Its left operand
/// is the existing exact rank-one runtime chain and its right operand is one
/// declared static scalar.  This deliberately does not make ordinary fixed
/// binary kernels runtime-capable.
pub(crate) fn schedule_dynamic_binary(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<MixedSchedule, RuntimeScheduleError> {
    MixedSchedule::from_static_and_runtime(
        &empty_fixed_schedule(),
        schedule_runtime_binary(graph, output)?,
    )
}

/// Lowers the sole runtime-to-fixed bridge: an exact rank-one dynamic value
/// into a scalar sum.  It accepts masked select directly or its typed unary
/// consumer, and rejects every other runtime producer structurally.
pub(crate) fn schedule_dynamic_sum(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<MixedSchedule, RuntimeScheduleError> {
    MixedSchedule::from_static_and_runtime(
        &empty_fixed_schedule(),
        schedule_runtime_sum(graph, output)?,
    )
}

/// Lowers the same bounded runtime chain into the canonical fixed scalar mean
/// bridge. No other fixed consumer becomes runtime-capable.
pub(crate) fn schedule_dynamic_mean(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<MixedSchedule, RuntimeScheduleError> {
    MixedSchedule::from_static_and_runtime(
        &empty_fixed_schedule(),
        schedule_runtime_mean(graph, output)?,
    )
}

fn schedule_runtime(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<RuntimeSchedule, RuntimeScheduleError> {
    let plan = graph
        .dynamic_allocation_plan(output)
        .map_err(RuntimeScheduleError::Plan)?;
    plan.validate_target(DynamicAllocationTarget::RuntimeSchedule)
        .map_err(RuntimeScheduleError::Plan)?;
    let runtime_output = RuntimeBufferDesc {
        id: RuntimeBufferId(plan.identity()),
        dtype: plan.output_dtype(),
        rank: plan.output_rank(),
        count_stage: plan.count_stage(),
        bindings: plan.bindings().to_vec(),
        plan_identity: plan.identity(),
    };
    let mut items = vec![
        RuntimeScheduleItem {
            id: 0,
            dependencies: vec![],
            kind: RuntimeScheduleItemKind::Count {
                stage: plan.count_stage(),
                bindings: plan.bindings().to_vec(),
            },
            cache_key: 0,
        },
        RuntimeScheduleItem {
            id: 1,
            dependencies: vec![0],
            kind: RuntimeScheduleItemKind::Allocate {
                output: runtime_output.clone(),
            },
            cache_key: 0,
        },
        RuntimeScheduleItem {
            id: 2,
            dependencies: vec![1],
            kind: RuntimeScheduleItemKind::MaterializeMaskedSelect {
                output: runtime_output.clone(),
            },
            cache_key: 0,
        },
    ];
    for item in &mut items {
        item.cache_key = item_key(item);
    }
    let lifetime = crate::memory_plan::RuntimeAllocationLifetime::new(plan.identity(), 1, 2);
    let mut schedule = RuntimeSchedule {
        plan,
        items,
        output: runtime_output.clone(),
        buffers: vec![runtime_output],
        fixed_output: None,
        lifetimes: vec![lifetime],
        identity: 0,
    };
    schedule.validate()?;
    schedule.identity = schedule_identity(&schedule);
    Ok(schedule)
}

fn schedule_runtime_sum(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<RuntimeSchedule, RuntimeScheduleError> {
    let node = graph
        .dynamic_node(output)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic sum output is absent"))?;
    let DynamicOp::Sum { input } = &node.op else {
        return Err(RuntimeScheduleError::Plan(
            DynamicAllocationError::UnsupportedOutput { output },
        ));
    };
    let input_node = graph
        .dynamic_node(*input)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic sum input is absent"))?;
    let mut schedule = match &input_node.op {
        DynamicOp::MaskedSelect { .. } => schedule_runtime(graph, *input)?,
        DynamicOp::Unary { .. } => schedule_runtime_unary(graph, *input)?,
        DynamicOp::Binary { .. } => schedule_runtime_binary(graph, *input)?,
        _ => {
            return Err(RuntimeScheduleError::Plan(
                DynamicAllocationError::UnsupportedOutput { output },
            ));
        }
    };
    if node.dtype != schedule.output.dtype {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "dynamic sum dtype differs from runtime input",
        ));
    }
    let fixed = BufferDesc {
        id: derived_fixed_sum_id(schedule.output.id, node.dtype),
        shape: Shape::from([]),
        dtype: node.dtype,
        bytes: node.dtype.itemsize(),
        alignment: node.dtype.itemsize().max(1),
        read_only: false,
        view: None,
    };
    let source = schedule.output.clone();
    let id = u64::try_from(schedule.items.len())
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("runtime sum item ID overflows"))?;
    schedule.items.push(RuntimeScheduleItem {
        id,
        dependencies: vec![id - 1],
        kind: RuntimeScheduleItemKind::DynamicReduceSum {
            input: source,
            output: fixed.clone(),
        },
        cache_key: 0,
    });
    for item in &mut schedule.items {
        item.cache_key = item_key(item);
    }
    schedule.fixed_output = Some(fixed);
    for lifetime in &mut schedule.lifetimes {
        *lifetime = crate::memory_plan::RuntimeAllocationLifetime::new(
            lifetime.buffer_id,
            lifetime.allocation_item,
            id,
        );
    }
    schedule.validate()?;
    schedule.identity = schedule_identity(&schedule);
    Ok(schedule)
}

fn schedule_runtime_mean(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<RuntimeSchedule, RuntimeScheduleError> {
    let node = graph
        .dynamic_node(output)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic mean output is absent"))?;
    let DynamicOp::Mean { input } = &node.op else {
        return Err(RuntimeScheduleError::Plan(
            DynamicAllocationError::UnsupportedOutput { output },
        ));
    };
    let input_node = graph
        .dynamic_node(*input)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic mean input is absent"))?;
    let mut schedule = match &input_node.op {
        DynamicOp::MaskedSelect { .. } => schedule_runtime(graph, *input)?,
        DynamicOp::Unary { .. } => schedule_runtime_unary(graph, *input)?,
        DynamicOp::Binary { .. } => schedule_runtime_binary(graph, *input)?,
        _ => {
            return Err(RuntimeScheduleError::Plan(
                DynamicAllocationError::UnsupportedOutput { output },
            ));
        }
    };
    let expected_dtype = if schedule.output.dtype.is_float() {
        schedule.output.dtype
    } else {
        DType::F32
    };
    if node.dtype != expected_dtype {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "dynamic mean dtype differs from canonical reduction policy",
        ));
    }
    let fixed = BufferDesc {
        id: derived_fixed_mean_id(schedule.output.id, node.dtype),
        shape: Shape::from([]),
        dtype: node.dtype,
        bytes: node.dtype.itemsize(),
        alignment: node.dtype.itemsize().max(1),
        read_only: false,
        view: None,
    };
    let source = schedule.output.clone();
    let id = u64::try_from(schedule.items.len())
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("runtime mean item ID overflows"))?;
    schedule.items.push(RuntimeScheduleItem {
        id,
        dependencies: vec![id - 1],
        kind: RuntimeScheduleItemKind::DynamicReduceMean {
            input: source,
            output: fixed.clone(),
        },
        cache_key: 0,
    });
    for item in &mut schedule.items {
        item.cache_key = item_key(item);
    }
    schedule.fixed_output = Some(fixed);
    for lifetime in &mut schedule.lifetimes {
        *lifetime = crate::memory_plan::RuntimeAllocationLifetime::new(
            lifetime.buffer_id,
            lifetime.allocation_item,
            id,
        );
    }
    schedule.validate()?;
    schedule.identity = schedule_identity(&schedule);
    Ok(schedule)
}

fn schedule_runtime_unary(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<RuntimeSchedule, RuntimeScheduleError> {
    let node = graph
        .dynamic_node(output)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic unary output is absent"))?;
    let DynamicOp::Unary { op, input } = &node.op else {
        return Err(RuntimeScheduleError::Plan(
            DynamicAllocationError::UnsupportedOutput { output },
        ));
    };
    let mut schedule = schedule_runtime(graph, *input)?;
    let source = schedule.output.clone();
    let unary_id = derived_buffer_id(source.id, *op, node.dtype);
    if unary_id == source.id {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "runtime unary buffer identity collides",
        ));
    }
    let unary = RuntimeBufferDesc {
        id: unary_id,
        dtype: node.dtype,
        rank: source.rank,
        count_stage: source.count_stage,
        bindings: source.bindings.clone(),
        plan_identity: source.plan_identity,
    };
    schedule.items.push(RuntimeScheduleItem {
        id: 3,
        dependencies: vec![2],
        kind: RuntimeScheduleItemKind::AllocateUnary {
            output: unary.clone(),
        },
        cache_key: 0,
    });
    schedule.items.push(RuntimeScheduleItem {
        id: 4,
        dependencies: vec![2, 3],
        kind: RuntimeScheduleItemKind::DynamicUnary {
            op: *op,
            input: source.clone(),
            output: unary.clone(),
        },
        cache_key: 0,
    });
    for item in &mut schedule.items {
        item.cache_key = item_key(item);
    }
    schedule.output = unary.clone();
    schedule.buffers.push(unary.clone());
    schedule.lifetimes = vec![
        crate::memory_plan::RuntimeAllocationLifetime::new(source.id.0, 1, 4),
        crate::memory_plan::RuntimeAllocationLifetime::new(unary.id.0, 3, 4),
    ];
    schedule.validate()?;
    schedule.identity = schedule_identity(&schedule);
    Ok(schedule)
}

fn schedule_runtime_binary(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<RuntimeSchedule, RuntimeScheduleError> {
    let node = graph
        .dynamic_node(output)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic binary output is absent"))?;
    let DynamicOp::Binary { op, lhs, rhs } = &node.op else {
        return Err(RuntimeScheduleError::Plan(
            DynamicAllocationError::UnsupportedOutput { output },
        ));
    };
    let DynamicInput::Dynamic(input) = lhs else {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "dynamic binary left operand must be a runtime value",
        ));
    };
    let DynamicInput::StaticScalar(static_node) = rhs else {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "dynamic binary right operand must be a static scalar",
        ));
    };
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "dynamic binary operation is not in the bounded CPU subset",
        ));
    }
    let input_node = graph
        .dynamic_node(*input)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic binary input is absent"))?;
    let mut schedule = match &input_node.op {
        DynamicOp::MaskedSelect { .. } => schedule_runtime(graph, *input)?,
        DynamicOp::Unary { .. } => schedule_runtime_unary(graph, *input)?,
        _ => {
            return Err(RuntimeScheduleError::Plan(
                DynamicAllocationError::UnsupportedOutput { output },
            ));
        }
    };
    let static_value = graph
        .node(*static_node)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic binary scalar is absent"))?;
    let static_elements = static_value.shape.numel().map_err(|_| {
        RuntimeScheduleError::InvalidOrdering("dynamic binary scalar shape overflows")
    })?;
    if static_elements != 1
        || node.dtype != DType::F32
        || schedule.output.dtype != DType::F32
        || static_value.dtype != DType::F32
    {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "dynamic binary requires an exact F32 scalar operand",
        ));
    }
    let static_input = StaticScalarBinding {
        node: *static_node,
        descriptor: BufferDesc {
            id: derived_static_scalar_id(*static_node, static_value.dtype),
            shape: static_value.shape.clone(),
            dtype: static_value.dtype,
            bytes: static_value.dtype.itemsize(),
            alignment: static_value.dtype.itemsize().max(1),
            read_only: true,
            view: None,
        },
    };
    let source = schedule.output.clone();
    let allocation_id = u64::try_from(schedule.items.len()).map_err(|_| {
        RuntimeScheduleError::InvalidOrdering("runtime binary allocation item ID overflows")
    })?;
    let binary_id = allocation_id
        .checked_add(1)
        .ok_or(RuntimeScheduleError::InvalidOrdering(
            "runtime binary item ID overflows",
        ))?;
    let binary = RuntimeBufferDesc {
        id: derived_binary_buffer_id(source.id, *op, &static_input, node.dtype),
        dtype: node.dtype,
        rank: source.rank,
        count_stage: source.count_stage,
        bindings: source.bindings.clone(),
        plan_identity: source.plan_identity,
    };
    if binary.id == source.id || schedule.buffers.iter().any(|buffer| buffer.id == binary.id) {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "runtime binary buffer identity collides",
        ));
    }
    let source_last = allocation_id
        .checked_sub(1)
        .ok_or(RuntimeScheduleError::InvalidOrdering(
            "runtime binary source item is absent",
        ))?;
    schedule.items.push(RuntimeScheduleItem {
        id: allocation_id,
        dependencies: vec![source_last],
        kind: RuntimeScheduleItemKind::AllocateBinary {
            output: binary.clone(),
        },
        cache_key: 0,
    });
    schedule.items.push(RuntimeScheduleItem {
        id: binary_id,
        dependencies: vec![source_last, allocation_id],
        kind: RuntimeScheduleItemKind::DynamicBinary {
            op: *op,
            input: source.clone(),
            static_input,
            output: binary.clone(),
        },
        cache_key: 0,
    });
    for item in &mut schedule.items {
        item.cache_key = item_key(item);
    }
    schedule.output = binary.clone();
    schedule.buffers.push(binary.clone());
    for lifetime in &mut schedule.lifetimes {
        *lifetime = crate::memory_plan::RuntimeAllocationLifetime::new(
            lifetime.buffer_id,
            lifetime.allocation_item,
            binary_id,
        );
    }
    schedule
        .lifetimes
        .push(crate::memory_plan::RuntimeAllocationLifetime::new(
            binary.id.0,
            allocation_id,
            binary_id,
        ));
    schedule.validate()?;
    schedule.identity = schedule_identity(&schedule);
    Ok(schedule)
}

fn derived_buffer_id(source: RuntimeBufferId, op: UnaryOp, dtype: DType) -> RuntimeBufferId {
    let mut hasher = DefaultHasher::new();
    "runtime-unary-buffer-v1".hash(&mut hasher);
    source.hash(&mut hasher);
    op.hash(&mut hasher);
    dtype.hash(&mut hasher);
    RuntimeBufferId(hasher.finish())
}

fn derived_binary_buffer_id(
    source: RuntimeBufferId,
    op: BinaryOp,
    static_input: &StaticScalarBinding,
    dtype: DType,
) -> RuntimeBufferId {
    let mut hasher = DefaultHasher::new();
    "runtime-binary-buffer-v1".hash(&mut hasher);
    source.hash(&mut hasher);
    op.hash(&mut hasher);
    static_input.hash(&mut hasher);
    dtype.hash(&mut hasher);
    RuntimeBufferId(hasher.finish())
}

fn derived_static_scalar_id(node: NodeId, dtype: DType) -> u64 {
    let mut hasher = DefaultHasher::new();
    "runtime-static-scalar-v1".hash(&mut hasher);
    node.hash(&mut hasher);
    dtype.hash(&mut hasher);
    hasher.finish()
}

fn derived_fixed_sum_id(source: RuntimeBufferId, dtype: DType) -> u64 {
    derived_fixed_reduction_id("runtime-reduce-sum-buffer-v1", source, dtype)
}

fn derived_fixed_mean_id(source: RuntimeBufferId, dtype: DType) -> u64 {
    derived_fixed_reduction_id("runtime-reduce-mean-buffer-v1", source, dtype)
}

fn derived_fixed_reduction_id(tag: &str, source: RuntimeBufferId, dtype: DType) -> u64 {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    source.hash(&mut hasher);
    dtype.hash(&mut hasher);
    hasher.finish()
}

fn empty_fixed_schedule() -> Schedule {
    Schedule {
        items: vec![],
        value_bindings: vec![],
        state_bindings: vec![],
    }
}

impl RuntimeSchedule {
    pub(crate) fn plan(&self) -> &DynamicAllocationPlan {
        &self.plan
    }

    pub(crate) fn validate(&self) -> Result<(), RuntimeScheduleError> {
        let runtime_item_len = self
            .items
            .len()
            .checked_sub(usize::from(self.fixed_output.is_some()))
            .ok_or(RuntimeScheduleError::InvalidOrdering(
                "runtime fixed-output marker exceeds item count",
            ))?;
        if !(runtime_item_len == 3 || runtime_item_len == 5 || runtime_item_len == 7)
            || self.items[0].id != 0
            || self.items[1].id != 1
            || self.items[2].id != 2
            || !self.items[0].dependencies.is_empty()
            || self.items[1].dependencies.as_slice() != [0]
            || self.items[2].dependencies.as_slice() != [1]
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime schedule must be count then allocation",
            ));
        }
        let RuntimeScheduleItemKind::Count { stage, bindings } = &self.items[0].kind else {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "first runtime item is not a count stage",
            ));
        };
        let RuntimeScheduleItemKind::Allocate { output } = &self.items[1].kind else {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "second runtime item is not an allocation stage",
            ));
        };
        let RuntimeScheduleItemKind::MaterializeMaskedSelect {
            output: materialized,
        } = &self.items[2].kind
        else {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "third runtime item is not a masked-select materialization",
            ));
        };
        if stage != &self.plan.count_stage()
            || bindings != self.plan.bindings()
            || output.plan_identity != self.plan.identity()
            || output.id != RuntimeBufferId(self.plan.identity())
            || output.dtype != self.plan.output_dtype()
            || output.rank != self.plan.output_rank()
            || materialized != output
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime count/allocation ABI mismatch",
            ));
        }
        if self.buffers.first() != Some(output)
            || self.buffers.iter().any(|descriptor| descriptor.rank != 1)
            || self
                .buffers
                .iter()
                .map(|descriptor| descriptor.id)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.buffers.len()
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime buffer descriptors are not an ordered unique rank-one set",
            ));
        }
        let mut expected_buffers = vec![output.clone()];
        let mut allocation_items = vec![1_u64];
        let mut previous_output = output.clone();
        let mut previous_item = 2_u64;
        let mut position = 3_usize;
        let mut saw_binary = false;
        while position < runtime_item_len {
            let allocate =
                self.items
                    .get(position)
                    .ok_or(RuntimeScheduleError::InvalidOrdering(
                        "runtime allocation item is absent",
                    ))?;
            let compute =
                self.items
                    .get(position + 1)
                    .ok_or(RuntimeScheduleError::InvalidOrdering(
                        "runtime consumer item is absent",
                    ))?;
            let is_binary = matches!(
                (&allocate.kind, &compute.kind),
                (
                    RuntimeScheduleItemKind::AllocateBinary { .. },
                    RuntimeScheduleItemKind::DynamicBinary { .. }
                )
            );
            if saw_binary
                || (!is_binary
                    && matches!(&compute.kind, RuntimeScheduleItemKind::DynamicUnary { .. })
                    && position != 3)
            {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime consumer order exceeds the bounded unary-then-binary chain",
                ));
            }
            let produced = match (&allocate.kind, &compute.kind) {
                (
                    RuntimeScheduleItemKind::AllocateUnary { output: allocated },
                    RuntimeScheduleItemKind::DynamicUnary {
                        op,
                        input,
                        output: produced,
                    },
                ) if matches!(op, UnaryOp::Neg | UnaryOp::Square)
                    && input == &previous_output
                    && produced == allocated
                    && allocated.dtype.is_float()
                    && allocated.dtype == previous_output.dtype =>
                {
                    allocated
                }
                (
                    RuntimeScheduleItemKind::AllocateBinary { output: allocated },
                    RuntimeScheduleItemKind::DynamicBinary {
                        op,
                        input,
                        static_input,
                        output: produced,
                    },
                ) if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul)
                    && input == &previous_output
                    && produced == allocated
                    && allocated.dtype == DType::F32
                    && previous_output.dtype == DType::F32
                    && static_input.descriptor.dtype == DType::F32
                    && static_input.descriptor.shape.numel().ok() == Some(1)
                    && static_input.descriptor.bytes == DType::F32.itemsize()
                    && static_input.descriptor.alignment == DType::F32.itemsize().max(1)
                    && static_input.descriptor.read_only
                    && static_input.descriptor.view.is_none()
                    && static_input.descriptor.id
                        == derived_static_scalar_id(
                            static_input.node,
                            static_input.descriptor.dtype,
                        ) =>
                {
                    allocated
                }
                _ => {
                    return Err(RuntimeScheduleError::InvalidOrdering(
                        "runtime dynamic consumer ABI mismatch",
                    ));
                }
            };
            let allocation_id = u64::try_from(position).map_err(|_| {
                RuntimeScheduleError::InvalidOrdering("runtime allocation ID overflows")
            })?;
            let compute_id =
                allocation_id
                    .checked_add(1)
                    .ok_or(RuntimeScheduleError::InvalidOrdering(
                        "runtime consumer ID overflows",
                    ))?;
            if allocate.id != allocation_id
                || compute.id != compute_id
                || allocate.dependencies.as_slice() != [previous_item]
                || compute.dependencies.as_slice() != [previous_item, allocation_id]
                || produced.rank != 1
                || produced.count_stage != output.count_stage
                || produced.bindings != output.bindings
                || produced.plan_identity != output.plan_identity
                || expected_buffers
                    .iter()
                    .any(|buffer| buffer.id == produced.id)
            {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime dynamic allocation/order ABI mismatch",
                ));
            }
            expected_buffers.push(produced.clone());
            allocation_items.push(allocation_id);
            previous_output = produced.clone();
            previous_item = compute_id;
            saw_binary |= is_binary;
            position += 2;
        }
        if self.output != previous_output
            || self.buffers != expected_buffers
            || self.lifetimes.len() != self.buffers.len()
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime output/buffer ABI mismatch",
            ));
        }
        if let Some(fixed) = &self.fixed_output {
            let sum = self
                .items
                .last()
                .ok_or(RuntimeScheduleError::InvalidOrdering(
                    "runtime sum item is absent",
                ))?;
            let (input, output, expected_dtype) = match &sum.kind {
                RuntimeScheduleItemKind::DynamicReduceSum { input, output } => {
                    (input, output, self.output.dtype)
                }
                RuntimeScheduleItemKind::DynamicReduceMean { input, output } => {
                    let dtype = if self.output.dtype.is_float() {
                        self.output.dtype
                    } else {
                        DType::F32
                    };
                    (input, output, dtype)
                }
                _ => {
                    return Err(RuntimeScheduleError::InvalidOrdering(
                        "final runtime item is not a dynamic reduction",
                    ));
                }
            };
            if sum.id != runtime_item_len as u64
                || sum.dependencies.as_slice() != [runtime_item_len as u64 - 1]
                || input != &self.output
                || output != fixed
                || output.shape != Shape::from([])
                || output.dtype != expected_dtype
                || output.bytes != output.dtype.itemsize()
            {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime reduction fixed-output ABI mismatch",
                ));
            }
        }
        if self
            .items
            .iter()
            .any(|item| item.cache_key != item_key(item))
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime item cache identity mismatch",
            ));
        }
        for lifetime in &self.lifetimes {
            lifetime
                .validate()
                .map_err(RuntimeScheduleError::InvalidOrdering)?;
        }
        let final_consumer = if self.fixed_output.is_some() {
            runtime_item_len as u64
        } else {
            runtime_item_len as u64 - 1
        };
        let expected_lifetimes = self
            .buffers
            .iter()
            .zip(allocation_items)
            .map(|(buffer, allocation_item)| {
                crate::memory_plan::RuntimeAllocationLifetime::new(
                    buffer.id.0,
                    allocation_item,
                    final_consumer,
                )
            })
            .collect::<Vec<_>>();
        if self.lifetimes != expected_lifetimes {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime allocation lifetime set mismatch",
            ));
        }
        Ok(())
    }
}

impl MixedSchedule {
    /// Joins a validated fixed schedule with a runtime count/allocation
    /// schedule. The static schedule's item IDs and cache keys are preserved;
    /// runtime item IDs are deterministically placed after them.
    pub(crate) fn from_static_and_runtime(
        fixed: &Schedule,
        runtime: RuntimeSchedule,
    ) -> Result<Self, RuntimeScheduleError> {
        fixed
            .validate()
            .map_err(|_| RuntimeScheduleError::InvalidOrdering("fixed schedule is invalid"))?;
        runtime.validate()?;
        let fixed_count = u64::try_from(fixed.items.len())
            .map_err(|_| RuntimeScheduleError::InvalidOrdering("fixed item count overflows"))?;
        let lifetimes = runtime
            .lifetimes
            .iter()
            .map(|lifetime| {
                Ok(crate::memory_plan::RuntimeAllocationLifetime::new(
                    lifetime.buffer_id,
                    fixed_count.checked_add(lifetime.allocation_item).ok_or(
                        RuntimeScheduleError::InvalidOrdering(
                            "runtime allocation lifetime overflows",
                        ),
                    )?,
                    fixed_count.checked_add(lifetime.final_consumer).ok_or(
                        RuntimeScheduleError::InvalidOrdering(
                            "runtime final-consumer lifetime overflows",
                        ),
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, RuntimeScheduleError>>()?;
        let mut items = fixed
            .items
            .iter()
            .map(|item| MixedScheduleItem {
                id: item.id,
                dependencies: item.dependencies.clone(),
                consumers: item.consumers.clone(),
                output: ScheduledOutputDesc::Fixed(item.primary_output().clone()),
                kind: MixedScheduleItemKind::Fixed {
                    source_item: item.id,
                },
                cache_key: item.cache_key,
            })
            .collect::<Vec<_>>();
        for item in &runtime.items {
            let id =
                fixed_count
                    .checked_add(item.id)
                    .ok_or(RuntimeScheduleError::InvalidOrdering(
                        "runtime item ID overflows",
                    ))?;
            let dependencies = item
                .dependencies
                .iter()
                .map(|dependency| {
                    fixed_count.checked_add(*dependency).ok_or(
                        RuntimeScheduleError::InvalidOrdering("runtime dependency overflows"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (output, kind) = match &item.kind {
                RuntimeScheduleItemKind::Count { stage, bindings } => (
                    // Count produces no allocatable buffer. Its output is the
                    // runtime descriptor it enables, not a scalar placeholder.
                    ScheduledOutputDesc::Runtime(runtime.buffers[0].clone()),
                    MixedScheduleItemKind::Count {
                        stage: *stage,
                        bindings: bindings.clone(),
                    },
                ),
                RuntimeScheduleItemKind::Allocate { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::Allocate,
                ),
                RuntimeScheduleItemKind::MaterializeMaskedSelect { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::MaterializeMaskedSelect,
                ),
                RuntimeScheduleItemKind::AllocateUnary { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::AllocateUnary,
                ),
                RuntimeScheduleItemKind::DynamicUnary { op, output, .. } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::DynamicUnary { op: *op },
                ),
                RuntimeScheduleItemKind::AllocateBinary { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::AllocateBinary,
                ),
                RuntimeScheduleItemKind::DynamicBinary { op, output, .. } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::DynamicBinary { op: *op },
                ),
                RuntimeScheduleItemKind::DynamicReduceSum { output, .. } => (
                    ScheduledOutputDesc::Fixed(output.clone()),
                    MixedScheduleItemKind::DynamicReduceSum,
                ),
                RuntimeScheduleItemKind::DynamicReduceMean { output, .. } => (
                    ScheduledOutputDesc::Fixed(output.clone()),
                    MixedScheduleItemKind::DynamicReduceMean,
                ),
            };
            items.push(MixedScheduleItem {
                id,
                dependencies,
                consumers: vec![],
                output,
                kind,
                cache_key: item.cache_key,
            });
        }
        for index in 0..items.len() {
            let item_id = items[index].id;
            let dependencies = items[index].dependencies.clone();
            for dependency in dependencies {
                let producer = items
                    .get_mut(usize::try_from(dependency).map_err(|_| {
                        RuntimeScheduleError::InvalidOrdering("dependency index overflows")
                    })?)
                    .ok_or(RuntimeScheduleError::InvalidOrdering(
                        "dependency is absent",
                    ))?;
                if !producer.consumers.contains(&item_id) {
                    producer.consumers.push(item_id);
                }
            }
        }
        let mut runtime_bindings = Vec::new();
        for item in &runtime.items {
            let consumer_item =
                fixed_count
                    .checked_add(item.id)
                    .ok_or(RuntimeScheduleError::InvalidOrdering(
                        "runtime binding item ID overflows",
                    ))?;
            match &item.kind {
                RuntimeScheduleItemKind::MaterializeMaskedSelect { output } => runtime_bindings
                    .push(RuntimeValueBinding {
                        source: RuntimeValueSource::Runtime {
                            source: output.id,
                            source_desc: output.clone(),
                        },
                        consumer_item,
                        abi_index: 0,
                    }),
                RuntimeScheduleItemKind::DynamicUnary { input, .. }
                | RuntimeScheduleItemKind::DynamicReduceSum { input, .. }
                | RuntimeScheduleItemKind::DynamicReduceMean { input, .. } => runtime_bindings
                    .push(RuntimeValueBinding {
                        source: RuntimeValueSource::Runtime {
                            source: input.id,
                            source_desc: input.clone(),
                        },
                        consumer_item,
                        abi_index: 0,
                    }),
                RuntimeScheduleItemKind::DynamicBinary {
                    input,
                    static_input,
                    ..
                } => {
                    runtime_bindings.push(RuntimeValueBinding {
                        source: RuntimeValueSource::Runtime {
                            source: input.id,
                            source_desc: input.clone(),
                        },
                        consumer_item,
                        abi_index: 0,
                    });
                    runtime_bindings.push(RuntimeValueBinding {
                        source: RuntimeValueSource::StaticScalar(static_input.clone()),
                        consumer_item,
                        abi_index: 1,
                    });
                }
                RuntimeScheduleItemKind::Count { .. }
                | RuntimeScheduleItemKind::Allocate { .. }
                | RuntimeScheduleItemKind::AllocateUnary { .. }
                | RuntimeScheduleItemKind::AllocateBinary { .. } => {}
            }
        }
        let mut mixed = Self {
            runtime,
            items,
            runtime_bindings,
            lifetimes,
            identity: 0,
        };
        mixed.validate()?;
        mixed.identity = mixed_identity(&mixed);
        Ok(mixed)
    }

    pub(crate) fn runtime(&self) -> &RuntimeSchedule {
        &self.runtime
    }

    /// Centralized descriptor lookup for a mixed DAG consumer. A caller that
    /// requires a runtime allocation must explicitly ask for the runtime form;
    /// fixed descriptors never silently coerce.
    pub(crate) fn runtime_output(
        &self,
        item_id: u64,
    ) -> Result<&RuntimeBufferDesc, RuntimeScheduleError> {
        let item = self
            .items
            .get(usize::try_from(item_id).map_err(|_| RuntimeScheduleError::UnknownItem(item_id))?)
            .ok_or(RuntimeScheduleError::UnknownItem(item_id))?;
        match &item.output {
            ScheduledOutputDesc::Runtime(output) => Ok(output),
            ScheduledOutputDesc::Fixed(_) => {
                Err(RuntimeScheduleError::ExpectedRuntimeOutput(item_id))
            }
        }
    }

    pub(crate) fn validate(&self) -> Result<(), RuntimeScheduleError> {
        self.runtime.validate()?;
        if self.items.len() < self.runtime.items.len() {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "mixed schedule omits runtime count/allocation/materialization items",
            ));
        }
        for (want, item) in self.items.iter().enumerate() {
            if item.id != want as u64 {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "mixed item IDs are not contiguous",
                ));
            }
            for dependency in &item.dependencies {
                let producer = self.items.get(*dependency as usize).ok_or(
                    RuntimeScheduleError::InvalidOrdering("mixed dependency is absent"),
                )?;
                if !producer.consumers.contains(&item.id) {
                    return Err(RuntimeScheduleError::InvalidOrdering(
                        "mixed consumer edge is absent",
                    ));
                }
                if matches!(item.kind, MixedScheduleItemKind::Fixed { .. })
                    && matches!(producer.output, ScheduledOutputDesc::Runtime(_))
                {
                    return Err(RuntimeScheduleError::StaticConsumerRuntimeInput {
                        consumer: item.id,
                        dependency: *dependency,
                    });
                }
            }
        }
        let offset = self
            .items
            .len()
            .checked_sub(self.runtime.items.len())
            .ok_or(RuntimeScheduleError::InvalidOrdering(
                "runtime item offset is absent",
            ))?;
        let mut bound_consumers = BTreeMap::new();
        let expected_binding_count = self
            .runtime
            .items
            .iter()
            .map(|item| match &item.kind {
                RuntimeScheduleItemKind::MaterializeMaskedSelect { .. }
                | RuntimeScheduleItemKind::DynamicUnary { .. }
                | RuntimeScheduleItemKind::DynamicReduceSum { .. }
                | RuntimeScheduleItemKind::DynamicReduceMean { .. } => 1,
                RuntimeScheduleItemKind::DynamicBinary { .. } => 2,
                _ => 0,
            })
            .sum::<usize>();
        for binding in &self.runtime_bindings {
            let consumer = self.items.get(binding.consumer_item as usize).ok_or(
                RuntimeScheduleError::InvalidOrdering("runtime value consumer is absent"),
            )?;
            let runtime_item = self
                .runtime
                .items
                .get(
                    usize::try_from(binding.consumer_item)
                        .ok()
                        .and_then(|id| id.checked_sub(offset))
                        .ok_or(RuntimeScheduleError::InvalidOrdering(
                            "runtime value binding consumer is outside runtime range",
                        ))?,
                )
                .ok_or(RuntimeScheduleError::InvalidOrdering(
                    "runtime value binding source item is absent",
                ))?;
            let valid = match (&runtime_item.kind, &binding.source, binding.abi_index) {
                (
                    RuntimeScheduleItemKind::MaterializeMaskedSelect { output },
                    RuntimeValueSource::Runtime {
                        source,
                        source_desc,
                    },
                    0,
                ) => *source == output.id && source_desc == output,
                (
                    RuntimeScheduleItemKind::DynamicUnary { input, .. }
                    | RuntimeScheduleItemKind::DynamicReduceSum { input, .. }
                    | RuntimeScheduleItemKind::DynamicReduceMean { input, .. },
                    RuntimeValueSource::Runtime {
                        source,
                        source_desc,
                    },
                    0,
                ) => *source == input.id && source_desc == input,
                (
                    RuntimeScheduleItemKind::DynamicBinary { input, .. },
                    RuntimeValueSource::Runtime {
                        source,
                        source_desc,
                    },
                    0,
                ) => *source == input.id && source_desc == input,
                (
                    RuntimeScheduleItemKind::DynamicBinary { static_input, .. },
                    RuntimeValueSource::StaticScalar(source),
                    1,
                ) => source == static_input,
                _ => false,
            };
            let source_is_known = match &binding.source {
                RuntimeValueSource::Runtime {
                    source,
                    source_desc,
                } => self
                    .runtime
                    .buffers
                    .iter()
                    .any(|descriptor| descriptor.id == *source && descriptor == source_desc),
                RuntimeValueSource::StaticScalar(source) => {
                    source.descriptor.shape.numel().ok() == Some(1)
                        && source.descriptor.dtype == DType::F32
                        && source.descriptor.bytes == DType::F32.itemsize()
                        && source.descriptor.alignment == DType::F32.itemsize().max(1)
                        && source.descriptor.read_only
                        && source.descriptor.view.is_none()
                        && source.descriptor.id
                            == derived_static_scalar_id(source.node, source.descriptor.dtype)
                }
            };
            if !valid
                || !source_is_known
                || bound_consumers
                    .insert(
                        (binding.consumer_item, binding.abi_index),
                        binding.source.clone(),
                    )
                    .is_some()
            {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime value binding ABI mismatch",
                ));
            }
            let expected_kind = match &runtime_item.kind {
                RuntimeScheduleItemKind::MaterializeMaskedSelect { .. } => {
                    MixedScheduleItemKind::MaterializeMaskedSelect
                }
                RuntimeScheduleItemKind::DynamicUnary { op, .. } => {
                    MixedScheduleItemKind::DynamicUnary { op: *op }
                }
                RuntimeScheduleItemKind::DynamicBinary { op, .. } => {
                    MixedScheduleItemKind::DynamicBinary { op: *op }
                }
                RuntimeScheduleItemKind::DynamicReduceSum { .. } => {
                    MixedScheduleItemKind::DynamicReduceSum
                }
                RuntimeScheduleItemKind::DynamicReduceMean { .. } => {
                    MixedScheduleItemKind::DynamicReduceMean
                }
                _ => {
                    return Err(RuntimeScheduleError::InvalidOrdering(
                        "non-consumer has a runtime value binding",
                    ));
                }
            };
            if consumer.kind != expected_kind {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime value binding consumer kind mismatch",
                ));
            }
        }
        if self.runtime_bindings.len() != expected_binding_count {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime value binding count mismatch",
            ));
        }
        let count = self
            .items
            .get(offset)
            .ok_or(RuntimeScheduleError::InvalidOrdering(
                "runtime count item is absent",
            ))?;
        let allocation =
            self.items
                .get(offset + 1)
                .ok_or(RuntimeScheduleError::InvalidOrdering(
                    "runtime allocation item is absent",
                ))?;
        let materialization =
            self.items
                .get(offset + 2)
                .ok_or(RuntimeScheduleError::InvalidOrdering(
                    "runtime materialization item is absent",
                ))?;
        if !matches!(count.kind, MixedScheduleItemKind::Count { .. })
            || !matches!(allocation.kind, MixedScheduleItemKind::Allocate)
            || !matches!(
                materialization.kind,
                MixedScheduleItemKind::MaterializeMaskedSelect
            )
            || !count.dependencies.is_empty()
            || allocation.dependencies.as_slice() != [count.id]
            || materialization.dependencies.as_slice() != [allocation.id]
            || count.output != ScheduledOutputDesc::Runtime(self.runtime.buffers[0].clone())
            || allocation.output != ScheduledOutputDesc::Runtime(self.runtime.buffers[0].clone())
            || materialization.output
                != ScheduledOutputDesc::Runtime(self.runtime.buffers[0].clone())
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "mixed runtime count/allocation ABI mismatch",
            ));
        }
        for runtime_item in &self.runtime.items {
            let mixed_item = self.items.get(offset + runtime_item.id as usize).ok_or(
                RuntimeScheduleError::InvalidOrdering("mixed runtime item is absent"),
            )?;
            let expected_dependencies = runtime_item
                .dependencies
                .iter()
                .map(|dependency| offset as u64 + dependency)
                .collect::<Vec<_>>();
            let (expected_output, expected_kind) = match &runtime_item.kind {
                RuntimeScheduleItemKind::Count { stage, bindings } => (
                    ScheduledOutputDesc::Runtime(self.runtime.buffers[0].clone()),
                    MixedScheduleItemKind::Count {
                        stage: *stage,
                        bindings: bindings.clone(),
                    },
                ),
                RuntimeScheduleItemKind::Allocate { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::Allocate,
                ),
                RuntimeScheduleItemKind::MaterializeMaskedSelect { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::MaterializeMaskedSelect,
                ),
                RuntimeScheduleItemKind::AllocateUnary { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::AllocateUnary,
                ),
                RuntimeScheduleItemKind::DynamicUnary { op, output, .. } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::DynamicUnary { op: *op },
                ),
                RuntimeScheduleItemKind::AllocateBinary { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::AllocateBinary,
                ),
                RuntimeScheduleItemKind::DynamicBinary { op, output, .. } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::DynamicBinary { op: *op },
                ),
                RuntimeScheduleItemKind::DynamicReduceSum { output, .. } => (
                    ScheduledOutputDesc::Fixed(output.clone()),
                    MixedScheduleItemKind::DynamicReduceSum,
                ),
                RuntimeScheduleItemKind::DynamicReduceMean { output, .. } => (
                    ScheduledOutputDesc::Fixed(output.clone()),
                    MixedScheduleItemKind::DynamicReduceMean,
                ),
            };
            if mixed_item.id != offset as u64 + runtime_item.id
                || mixed_item.dependencies != expected_dependencies
                || mixed_item.output != expected_output
                || mixed_item.kind != expected_kind
                || mixed_item.cache_key != runtime_item.cache_key
            {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "mixed runtime item ABI mismatch",
                ));
            }
        }
        if self.lifetimes.len() != self.runtime.lifetimes.len() {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "mixed runtime lifetime count mismatch",
            ));
        }
        for (mixed, runtime) in self.lifetimes.iter().zip(&self.runtime.lifetimes) {
            mixed
                .validate()
                .map_err(RuntimeScheduleError::InvalidOrdering)?;
            let expected = crate::memory_plan::RuntimeAllocationLifetime::new(
                runtime.buffer_id,
                offset as u64 + runtime.allocation_item,
                offset as u64 + runtime.final_consumer,
            );
            if *mixed != expected {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "mixed runtime allocation lifetime mismatch",
                ));
            }
        }
        Ok(())
    }
}

impl RuntimeBufferTable {
    pub(crate) fn new(schedule: &RuntimeSchedule) -> Result<Self, RuntimeScheduleError> {
        schedule.validate()?;
        let mut descriptors = Vec::with_capacity(schedule.buffers.len());
        for descriptor in &schedule.buffers {
            if descriptors
                .iter()
                .any(|existing: &RuntimeBufferDesc| existing.id == descriptor.id)
            {
                return Err(RuntimeScheduleError::DuplicateBuffer(descriptor.id));
            }
            descriptors.push(descriptor.clone());
        }
        Ok(Self {
            descriptors,
            allocations: BTreeMap::new(),
        })
    }

    /// Performs the checked allocation stage after the count item. The output
    /// descriptor becomes live only after this returns successfully.
    pub(crate) fn allocate_buffer_after_count(
        &mut self,
        schedule: &RuntimeSchedule,
        id: RuntimeBufferId,
        elements: usize,
    ) -> Result<&DynamicAllocation, RuntimeScheduleError> {
        schedule.validate()?;
        let descriptor = self.descriptor(id)?.clone();
        let position = schedule
            .buffers
            .iter()
            .position(|candidate| candidate.id == id)
            .ok_or(RuntimeScheduleError::UnknownBuffer(id))?;
        if position > 0 {
            let predecessor = schedule.buffers[position - 1].id;
            if !self.allocations.contains_key(&predecessor) {
                return Err(RuntimeScheduleError::LiveLookupBeforeAllocation(
                    predecessor,
                ));
            }
        }
        if self.allocations.contains_key(&id) {
            return Err(RuntimeScheduleError::DuplicateAllocation(id));
        }
        let bytes =
            elements
                .checked_mul(descriptor.dtype.itemsize())
                .ok_or(RuntimeScheduleError::Plan(
                    DynamicAllocationError::AllocationOverflow {
                        elements,
                        dtype: descriptor.dtype,
                    },
                ))?;
        let allocation = DynamicAllocation {
            shape: Shape::from([elements]),
            dtype: descriptor.dtype,
            elements,
            bytes,
        };
        self.allocations.insert(id, allocation);
        self.allocation(id)
    }

    #[cfg(test)]
    pub(crate) fn allocate_output_after_count(
        &mut self,
        schedule: &RuntimeSchedule,
        elements: usize,
    ) -> Result<&DynamicAllocation, RuntimeScheduleError> {
        self.allocate_buffer_after_count(schedule, schedule.buffers[0].id, elements)
    }

    /// Centralized live lookup. A runtime buffer cannot be observed before the
    /// canonical count/allocation dependency has completed.
    pub(crate) fn allocation(
        &self,
        id: RuntimeBufferId,
    ) -> Result<&DynamicAllocation, RuntimeScheduleError> {
        self.descriptor(id)?;
        self.allocations
            .get(&id)
            .ok_or(RuntimeScheduleError::LiveLookupBeforeAllocation(id))
    }

    /// Centralized descriptor lookup for all runtime-buffer consumers.
    pub(crate) fn descriptor(
        &self,
        id: RuntimeBufferId,
    ) -> Result<&RuntimeBufferDesc, RuntimeScheduleError> {
        self.descriptors
            .iter()
            .find(|descriptor| descriptor.id == id)
            .ok_or(RuntimeScheduleError::UnknownBuffer(id))
    }
}

fn item_key(item: &RuntimeScheduleItem) -> u64 {
    let mut hasher = DefaultHasher::new();
    item.id.hash(&mut hasher);
    item.dependencies.hash(&mut hasher);
    item.kind.hash(&mut hasher);
    hasher.finish()
}

fn schedule_identity(schedule: &RuntimeSchedule) -> u64 {
    let mut hasher = DefaultHasher::new();
    schedule.plan.identity().hash(&mut hasher);
    schedule.items.hash(&mut hasher);
    schedule.output.hash(&mut hasher);
    schedule.buffers.hash(&mut hasher);
    schedule.lifetimes.hash(&mut hasher);
    hasher.finish()
}

fn mixed_identity(schedule: &MixedSchedule) -> u64 {
    let mut hasher = DefaultHasher::new();
    schedule.runtime.identity.hash(&mut hasher);
    schedule.items.hash(&mut hasher);
    schedule.runtime_bindings.hash(&mut hasher);
    schedule.lifetimes.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, DynamicAllocationTarget, Graph, Scalar, TensorData};

    fn fixture() -> (Graph, DynamicNodeId) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
        let output = graph.masked_select_dynamic(input, mask).unwrap();
        (graph, output)
    }

    #[test]
    fn exact_runtime_schedule_orders_count_before_allocation() {
        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        assert_eq!(schedule.items[0].dependencies, Vec::<u64>::new());
        assert_eq!(schedule.items[1].dependencies, vec![0]);
        let runtime = schedule.runtime();
        let mut table = RuntimeBufferTable::new(runtime).unwrap();
        assert_eq!(
            table.allocation(runtime.output.id),
            Err(RuntimeScheduleError::LiveLookupBeforeAllocation(
                runtime.output.id
            ))
        );
        assert_eq!(
            table.allocate_output_after_count(runtime, 3).unwrap().shape,
            Shape::from([3])
        );
    }

    #[test]
    fn exact_runtime_schedule_keeps_zero_and_identity_deterministic() {
        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let runtime = schedule.runtime();
        let mut table = RuntimeBufferTable::new(runtime).unwrap();
        let zero = table.allocate_output_after_count(runtime, 0).unwrap();
        assert_eq!(zero.bytes, 0);
        let (equivalent, equivalent_output) = fixture();
        assert_eq!(
            schedule.identity,
            schedule_dynamic(&equivalent, equivalent_output)
                .unwrap()
                .identity
        );
        assert_eq!(
            schedule.runtime().lifetimes.clone(),
            schedule_dynamic(&equivalent, equivalent_output)
                .unwrap()
                .runtime()
                .lifetimes
                .clone()
        );
    }

    #[test]
    fn dynamic_plan_rejects_fixed_and_non_cpu_routes_before_allocation() {
        let (graph, output) = fixture();
        let plan = graph.dynamic_allocation_plan(output).unwrap();
        for target in [
            DynamicAllocationTarget::Schedule,
            DynamicAllocationTarget::Capture,
            DynamicAllocationTarget::Artifact,
            DynamicAllocationTarget::Replay,
            DynamicAllocationTarget::NativeCpuJit,
            DynamicAllocationTarget::Device,
        ] {
            assert!(plan.validate_target(target).is_err());
        }
    }

    #[test]
    fn malformed_ordering_and_overflow_reject_before_allocation() {
        let (graph, output) = fixture();
        let mut schedule = schedule_dynamic(&graph, output).unwrap();
        schedule.items[1].dependencies.clear();
        assert!(matches!(
            schedule.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(_))
        ));

        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let runtime = schedule.runtime();
        let mut table = RuntimeBufferTable::new(runtime).unwrap();
        assert!(matches!(
            table.allocate_output_after_count(runtime, usize::MAX),
            Err(RuntimeScheduleError::Plan(
                DynamicAllocationError::AllocationOverflow { .. }
            ))
        ));
        assert_eq!(
            table.allocation(runtime.output.id),
            Err(RuntimeScheduleError::LiveLookupBeforeAllocation(
                runtime.output.id
            ))
        );
    }

    #[test]
    fn binding_rejection_leaves_runtime_buffer_unallocated() {
        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let input = TensorData::from_scalars([2, 2], DType::F32, [Scalar::F(1.0); 4]).unwrap();
        let wrong_mask =
            TensorData::from_scalars([2, 2], DType::Bool, [Scalar::Bool(true); 4]).unwrap();
        let runtime = schedule.runtime();
        let table = RuntimeBufferTable::new(runtime).unwrap();
        assert!(
            runtime
                .plan()
                .validate_bindings(&input, &wrong_mask)
                .is_err()
        );
        assert_eq!(
            table.allocation(runtime.output.id),
            Err(RuntimeScheduleError::LiveLookupBeforeAllocation(
                runtime.output.id
            ))
        );
    }

    #[test]
    fn fixed_shape_schedules_keep_their_existing_cache_identity() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let output = graph.square(input).unwrap();
        let first = crate::schedule::schedule(&graph, output).unwrap();
        let second = crate::schedule::schedule(&graph, output).unwrap();
        assert_eq!(first.items.len(), 1);
        assert_eq!(first.items[0].cache_key, second.items[0].cache_key);
    }

    #[test]
    fn mixed_dag_preserves_fixed_item_identity_then_orders_runtime_items() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
        let fixed_output = graph.square(input).unwrap();
        let dynamic_output = graph.masked_select_dynamic(input, mask).unwrap();
        let fixed = crate::schedule::schedule(&graph, fixed_output).unwrap();
        let runtime = schedule_runtime(&graph, dynamic_output).unwrap();
        let mixed = MixedSchedule::from_static_and_runtime(&fixed, runtime).unwrap();
        assert_eq!(mixed.items[0].cache_key, fixed.items[0].cache_key);
        assert!(matches!(
            mixed.items[0].output,
            ScheduledOutputDesc::Fixed(_)
        ));
        assert_eq!(mixed.items[1].dependencies, Vec::<u64>::new());
        assert_eq!(mixed.items[2].dependencies, vec![1]);
        assert_eq!(mixed.items[3].dependencies, vec![2]);
        assert_eq!(mixed.lifetimes[0].allocation_item, 2);
        assert_eq!(mixed.lifetimes[0].final_consumer, 3);
        assert!(matches!(
            mixed.items[3].output,
            ScheduledOutputDesc::Runtime(_)
        ));
        assert_eq!(mixed.runtime_output(3).unwrap().rank, 1);
        assert_eq!(
            mixed.runtime_output(0),
            Err(RuntimeScheduleError::ExpectedRuntimeOutput(0))
        );
    }

    #[test]
    fn fixed_consumer_of_runtime_output_rejects_before_allocation() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
        let fixed_output = graph.square(input).unwrap();
        let dynamic_output = graph.masked_select_dynamic(input, mask).unwrap();
        let fixed = crate::schedule::schedule(&graph, fixed_output).unwrap();
        let runtime = schedule_runtime(&graph, dynamic_output).unwrap();
        let mut mixed = MixedSchedule::from_static_and_runtime(&fixed, runtime).unwrap();
        mixed.items[0].dependencies.push(2);
        mixed.items[2].consumers.push(0);
        assert_eq!(
            mixed.validate(),
            Err(RuntimeScheduleError::StaticConsumerRuntimeInput {
                consumer: 0,
                dependency: 2,
            })
        );
    }

    #[test]
    fn runtime_value_binding_is_deterministic_and_rejects_wrong_consumer() {
        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        assert_eq!(schedule.runtime_bindings.len(), 1);
        assert_eq!(schedule.runtime_bindings[0].consumer_item, 2);
        assert_eq!(schedule.runtime_bindings[0].abi_index, 0);
        let mut corrupt = schedule.clone();
        corrupt.runtime_bindings[0].consumer_item = 0;
        assert!(matches!(
            corrupt.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime value binding ABI mismatch"
            ))
        ));
    }

    #[test]
    fn unary_chain_has_two_ordered_exact_buffers_and_distinct_lifetimes() {
        let (mut graph, selected) = fixture();
        let output = graph.dynamic_unary(selected, UnaryOp::Neg).unwrap();
        let schedule = schedule_dynamic_unary(&graph, output).unwrap();
        assert_eq!(schedule.runtime().buffers.len(), 2);
        assert_ne!(
            schedule.runtime().buffers[0].id,
            schedule.runtime().buffers[1].id
        );
        assert_eq!(schedule.items.len(), 5);
        assert_eq!(schedule.items[3].dependencies, vec![2]);
        assert_eq!(schedule.items[4].dependencies, vec![2, 3]);
        assert_eq!(schedule.lifetimes.len(), 2);
        assert_eq!(schedule.lifetimes[0].final_consumer, 4);
        assert_eq!(schedule.lifetimes[1].allocation_item, 3);
        assert_eq!(schedule.runtime_bindings.len(), 2);
        assert_eq!(schedule.runtime_bindings[1].consumer_item, 4);
        let (mut equivalent_graph, equivalent_selected) = fixture();
        let equivalent_output = equivalent_graph
            .dynamic_unary(equivalent_selected, UnaryOp::Neg)
            .unwrap();
        assert_eq!(
            schedule.identity,
            schedule_dynamic_unary(&equivalent_graph, equivalent_output)
                .unwrap()
                .identity
        );
        let mut table = RuntimeBufferTable::new(schedule.runtime()).unwrap();
        let source = schedule.runtime().buffers[0].id;
        let unary = schedule.runtime().buffers[1].id;
        assert_eq!(
            table
                .allocate_buffer_after_count(schedule.runtime(), source, 0)
                .unwrap()
                .bytes,
            0
        );
        assert_eq!(
            table
                .allocate_buffer_after_count(schedule.runtime(), unary, 0)
                .unwrap()
                .bytes,
            0
        );
        assert_eq!(table.allocation(source).unwrap().dtype, DType::F32);
        assert_eq!(table.allocation(unary).unwrap().dtype, DType::F32);
    }

    #[test]
    fn malformed_unary_binding_rejects_before_second_allocation() {
        let (mut graph, selected) = fixture();
        let output = graph.dynamic_unary(selected, UnaryOp::Square).unwrap();
        let mut schedule = schedule_dynamic_unary(&graph, output).unwrap();
        schedule.items[4].dependencies = vec![3];
        assert!(matches!(
            schedule.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(_))
        ));
    }

    #[test]
    fn dynamic_binary_has_ordered_static_scalar_abi_and_distinct_exact_output() {
        let (mut graph, selected) = fixture();
        let scalar = graph.constant(TensorData::scalar(2.0));
        let output = graph
            .dynamic_binary(
                selected,
                crate::DynamicInput::StaticScalar(scalar),
                BinaryOp::Add,
            )
            .unwrap();
        let schedule = schedule_dynamic_binary(&graph, output).unwrap();
        assert_eq!(schedule.items.len(), 5);
        assert!(matches!(
            schedule.items[3].kind,
            MixedScheduleItemKind::AllocateBinary
        ));
        assert!(matches!(
            schedule.items[4].kind,
            MixedScheduleItemKind::DynamicBinary { op: BinaryOp::Add }
        ));
        assert_eq!(schedule.items[3].dependencies, vec![2]);
        assert_eq!(schedule.items[4].dependencies, vec![2, 3]);
        assert_eq!(schedule.runtime().buffers.len(), 2);
        assert_ne!(
            schedule.runtime().buffers[0].id,
            schedule.runtime().buffers[1].id
        );
        assert_eq!(schedule.lifetimes[0].final_consumer, 4);
        assert_eq!(schedule.lifetimes[1].final_consumer, 4);
        assert_eq!(schedule.runtime_bindings.len(), 3);
        assert!(matches!(
            schedule.runtime_bindings[2].source,
            RuntimeValueSource::StaticScalar(_)
        ));
        assert_eq!(schedule.runtime_bindings[2].consumer_item, 4);
        assert_eq!(schedule.runtime_bindings[2].abi_index, 1);
        let (mut equivalent_graph, equivalent_selected) = fixture();
        let equivalent_scalar = equivalent_graph.constant(TensorData::scalar(2.0));
        let equivalent_output = equivalent_graph
            .dynamic_binary(
                equivalent_selected,
                crate::DynamicInput::StaticScalar(equivalent_scalar),
                BinaryOp::Add,
            )
            .unwrap();
        assert_eq!(
            schedule.identity,
            schedule_dynamic_binary(&equivalent_graph, equivalent_output)
                .unwrap()
                .identity
        );
        let mut table = RuntimeBufferTable::new(schedule.runtime()).unwrap();
        let source = schedule.runtime().buffers[0].id;
        let binary = schedule.runtime().buffers[1].id;
        table
            .allocate_buffer_after_count(schedule.runtime(), source, 0)
            .unwrap();
        table
            .allocate_buffer_after_count(schedule.runtime(), binary, 0)
            .unwrap();
        assert_eq!(table.allocation(binary).unwrap().shape, Shape::from([0]));
    }

    #[test]
    fn dynamic_binary_rejects_misordered_static_abi_before_allocation() {
        let (mut graph, selected) = fixture();
        let scalar = graph.constant(TensorData::scalar(2.0));
        let output = graph
            .dynamic_binary(
                selected,
                crate::DynamicInput::StaticScalar(scalar),
                BinaryOp::Mul,
            )
            .unwrap();
        let mut schedule = schedule_dynamic_binary(&graph, output).unwrap();
        schedule.runtime_bindings[2].abi_index = 0;
        assert!(matches!(
            schedule.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime value binding ABI mismatch"
            ))
        ));
    }

    #[test]
    fn dynamic_binary_rejects_non_f32_scalar_before_runtime_allocation() {
        let (mut graph, selected) = fixture();
        let scalar = graph.input_dtype("scalar", [], DType::F64);
        let output = graph
            .dynamic_binary(
                selected,
                crate::DynamicInput::StaticScalar(scalar),
                BinaryOp::Add,
            )
            .unwrap();
        assert!(matches!(
            schedule_dynamic_binary(&graph, output),
            Err(RuntimeScheduleError::InvalidOrdering(
                "dynamic binary requires an exact F32 scalar operand"
            ))
        ));
    }

    #[test]
    fn runtime_sum_is_the_only_runtime_to_fixed_bridge() {
        let (mut graph, selected) = fixture();
        let sum = graph.dynamic_sum(selected).unwrap();
        let schedule = schedule_dynamic_sum(&graph, sum).unwrap();
        assert_eq!(schedule.items.len(), 4);
        assert!(matches!(
            schedule.items[3].kind,
            MixedScheduleItemKind::DynamicReduceSum
        ));
        assert_eq!(schedule.items[3].dependencies, vec![2]);
        assert!(matches!(
            schedule.items[3].output,
            ScheduledOutputDesc::Fixed(_)
        ));
        assert_eq!(schedule.runtime_bindings.len(), 2);
        assert_eq!(schedule.runtime_bindings[1].consumer_item, 3);
        assert_eq!(schedule.lifetimes[0].final_consumer, 3);
        let mut corrupt = schedule.clone();
        corrupt.runtime_bindings[1].source = RuntimeValueSource::Runtime {
            source: RuntimeBufferId(0),
            source_desc: schedule.runtime().output.clone(),
        };
        assert!(matches!(
            corrupt.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(_))
        ));
    }

    #[test]
    fn unary_runtime_sum_extends_both_lifetimes_to_fixed_consumer() {
        let (mut graph, selected) = fixture();
        let unary = graph.dynamic_unary(selected, UnaryOp::Square).unwrap();
        let sum = graph.dynamic_sum(unary).unwrap();
        let schedule = schedule_dynamic_sum(&graph, sum).unwrap();
        assert_eq!(schedule.items.len(), 6);
        assert_eq!(schedule.items[5].dependencies, vec![4]);
        assert_eq!(schedule.lifetimes.len(), 2);
        assert_eq!(schedule.lifetimes[0].final_consumer, 5);
        assert_eq!(schedule.lifetimes[1].final_consumer, 5);
        assert!(matches!(
            schedule.items[5].output,
            ScheduledOutputDesc::Fixed(_)
        ));
    }
}
