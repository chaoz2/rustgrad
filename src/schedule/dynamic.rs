//! Runtime-sized schedule ABI for exact CPU dynamic cardinality results.
//!
//! This is deliberately separate from the fixed-`Shape` `ScheduleItem` ABI:
//! a dynamic buffer cannot masquerade as a static `BufferDesc`.  It remains a
//! crate-private schedule branch until ordinary schedule/capture artifacts can
//! retain and validate runtime-sized buffers without placeholders.

use crate::{
    DynamicAllocation, DynamicAllocationError, DynamicAllocationPlan, DynamicAllocationTarget,
    DynamicBinding, DynamicCountStage, DynamicNodeId, DType, Graph, Shape,
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

/// The two explicit stages needed by the exact runtime-sized CPU contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RuntimeScheduleItemKind {
    Count {
        stage: DynamicCountStage,
        bindings: Vec<DynamicBinding>,
    },
    Allocate {
        output: RuntimeBufferDesc,
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

/// The canonical runtime-sized schedule for one exact dynamic result. It is
/// not a second planner: construction consumes the graph-owned
/// `DynamicAllocationPlan` and execution consumes this single item ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeSchedule {
    plan: DynamicAllocationPlan,
    pub items: Vec<RuntimeScheduleItem>,
    pub output: RuntimeBufferDesc,
    pub identity: u64,
}

/// Runtime allocation metadata remains absent until the count stage has
/// completed. No tensor value or bounded placeholder is stored here.
#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeBufferTable {
    descriptors: BTreeMap<RuntimeBufferId, RuntimeBufferDesc>,
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
    ];
    for item in &mut items {
        item.cache_key = item_key(item);
    }
    let mut schedule = RuntimeSchedule {
        plan,
        items,
        output: runtime_output,
        identity: 0,
    };
    schedule.validate()?;
    schedule.identity = schedule_identity(&schedule);
    Ok(schedule)
}

impl RuntimeSchedule {
    pub(crate) fn plan(&self) -> &DynamicAllocationPlan {
        &self.plan
    }

    pub(crate) fn validate(&self) -> Result<(), RuntimeScheduleError> {
        if self.items.len() != 2
            || self.items[0].id != 0
            || self.items[1].id != 1
            || !self.items[0].dependencies.is_empty()
            || self.items[1].dependencies.as_slice() != [0]
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
        if stage != &self.plan.count_stage()
            || bindings != self.plan.bindings()
            || output != &self.output
            || output.plan_identity != self.plan.identity()
            || output.id != RuntimeBufferId(self.plan.identity())
            || output.dtype != self.plan.output_dtype()
            || output.rank != self.plan.output_rank()
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime count/allocation ABI mismatch",
            ));
        }
        if self.items.iter().any(|item| item.cache_key != item_key(item)) {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime item cache identity mismatch",
            ));
        }
        Ok(())
    }
}

impl RuntimeBufferTable {
    pub(crate) fn new(schedule: &RuntimeSchedule) -> Result<Self, RuntimeScheduleError> {
        schedule.validate()?;
        let mut descriptors = BTreeMap::new();
        if descriptors
            .insert(schedule.output.id, schedule.output.clone())
            .is_some()
        {
            return Err(RuntimeScheduleError::DuplicateBuffer(schedule.output.id));
        }
        Ok(Self {
            descriptors,
            allocations: BTreeMap::new(),
        })
    }

    /// Performs the checked allocation stage after the count item. The output
    /// descriptor becomes live only after this returns successfully.
    pub(crate) fn allocate_output_after_count(
        &mut self,
        schedule: &RuntimeSchedule,
        elements: usize,
    ) -> Result<&DynamicAllocation, RuntimeScheduleError> {
        schedule.validate()?;
        let id = schedule.output.id;
        self.descriptor(id)?;
        if self.allocations.contains_key(&id) {
            return Err(RuntimeScheduleError::DuplicateAllocation(id));
        }
        let allocation = schedule
            .plan
            .allocation_for_count(elements)
            .map_err(RuntimeScheduleError::Plan)?;
        self.allocations.insert(id, allocation);
        self.allocation(id)
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
            .get(&id)
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
        let mut table = RuntimeBufferTable::new(&schedule).unwrap();
        assert_eq!(
            table.allocation(schedule.output.id),
            Err(RuntimeScheduleError::LiveLookupBeforeAllocation(
                schedule.output.id
            ))
        );
        assert_eq!(
            table
                .allocate_output_after_count(&schedule, 3)
                .unwrap()
                .shape,
            Shape::from([3])
        );
    }

    #[test]
    fn exact_runtime_schedule_keeps_zero_and_identity_deterministic() {
        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let mut table = RuntimeBufferTable::new(&schedule).unwrap();
        let zero = table.allocate_output_after_count(&schedule, 0).unwrap();
        assert_eq!(zero.bytes, 0);
        let (equivalent, equivalent_output) = fixture();
        assert_eq!(
            schedule.identity,
            schedule_dynamic(&equivalent, equivalent_output).unwrap().identity
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
        let mut table = RuntimeBufferTable::new(&schedule).unwrap();
        assert!(matches!(
            table.allocate_output_after_count(&schedule, usize::MAX),
            Err(RuntimeScheduleError::Plan(
                DynamicAllocationError::AllocationOverflow { .. }
            ))
        ));
        assert_eq!(
            table.allocation(schedule.output.id),
            Err(RuntimeScheduleError::LiveLookupBeforeAllocation(
                schedule.output.id
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
        let table = RuntimeBufferTable::new(&schedule).unwrap();
        assert!(schedule.plan().validate_bindings(&input, &wrong_mask).is_err());
        assert_eq!(
            table.allocation(schedule.output.id),
            Err(RuntimeScheduleError::LiveLookupBeforeAllocation(
                schedule.output.id
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
}
