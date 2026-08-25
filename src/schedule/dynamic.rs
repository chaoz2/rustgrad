//! Runtime-sized schedule ABI for exact CPU dynamic cardinality results.
//!
//! This is deliberately separate from the fixed-`Shape` `ScheduleItem` ABI:
//! a dynamic buffer cannot masquerade as a static `BufferDesc`.  It remains a
//! crate-private schedule branch until ordinary schedule/capture artifacts can
//! retain and validate runtime-sized buffers without placeholders.

use crate::{
    BufferDesc, DynamicAllocation, DynamicAllocationError, DynamicAllocationPlan,
    DynamicAllocationTarget, DynamicBinding, DynamicCountStage, DynamicNodeId, DType, Graph,
    Schedule, Shape,
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
    Fixed { source_item: u64 },
    Count {
        stage: DynamicCountStage,
        bindings: Vec<DynamicBinding>,
    },
    Allocate,
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

/// Private canonical DAG envelope joining static `ScheduleItem` records and
/// runtime-sized records. It owns no alternative planner or cache: fixed items
/// retain their ordinary schedule keys and runtime items retain the allocation
/// plan identities from which they were lowered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MixedSchedule {
    runtime: RuntimeSchedule,
    pub items: Vec<MixedScheduleItem>,
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
        let mut items = fixed
            .items
            .iter()
            .map(|item| MixedScheduleItem {
                id: item.id,
                dependencies: item.dependencies.clone(),
                consumers: item.consumers.clone(),
                output: ScheduledOutputDesc::Fixed(item.output.clone()),
                kind: MixedScheduleItemKind::Fixed {
                    source_item: item.id,
                },
                cache_key: item.cache_key,
            })
            .collect::<Vec<_>>();
        for item in &runtime.items {
            let id = fixed_count
                .checked_add(item.id)
                .ok_or(RuntimeScheduleError::InvalidOrdering("runtime item ID overflows"))?;
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
                    ScheduledOutputDesc::Runtime(runtime.output.clone()),
                    MixedScheduleItemKind::Count {
                        stage: *stage,
                        bindings: bindings.clone(),
                    },
                ),
                RuntimeScheduleItemKind::Allocate { output } => (
                    ScheduledOutputDesc::Runtime(output.clone()),
                    MixedScheduleItemKind::Allocate,
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
                    .ok_or(RuntimeScheduleError::InvalidOrdering("dependency is absent"))?;
                if !producer.consumers.contains(&item_id) {
                    producer.consumers.push(item_id);
                }
            }
        }
        let mut mixed = Self {
            runtime,
            items,
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
            .get(
                usize::try_from(item_id)
                    .map_err(|_| RuntimeScheduleError::UnknownItem(item_id))?,
            )
            .ok_or(RuntimeScheduleError::UnknownItem(item_id))?;
        match &item.output {
            ScheduledOutputDesc::Runtime(output) => Ok(output),
            ScheduledOutputDesc::Fixed(_) => Err(RuntimeScheduleError::ExpectedRuntimeOutput(item_id)),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), RuntimeScheduleError> {
        self.runtime.validate()?;
        if self.items.len() < 2 {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "mixed schedule omits runtime count/allocation items",
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
            .checked_sub(2)
            .ok_or(RuntimeScheduleError::InvalidOrdering("runtime item offset is absent"))?;
        let count = self.items.get(offset).ok_or(RuntimeScheduleError::InvalidOrdering(
            "runtime count item is absent",
        ))?;
        let allocation = self.items.get(offset + 1).ok_or(
            RuntimeScheduleError::InvalidOrdering("runtime allocation item is absent"),
        )?;
        if !matches!(count.kind, MixedScheduleItemKind::Count { .. })
            || !matches!(allocation.kind, MixedScheduleItemKind::Allocate)
            || !count.dependencies.is_empty()
            || allocation.dependencies.as_slice() != [count.id]
            || count.output != ScheduledOutputDesc::Runtime(self.runtime.output.clone())
            || allocation.output != ScheduledOutputDesc::Runtime(self.runtime.output.clone())
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "mixed runtime count/allocation ABI mismatch",
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

fn mixed_identity(schedule: &MixedSchedule) -> u64 {
    let mut hasher = DefaultHasher::new();
    schedule.runtime.identity.hash(&mut hasher);
    schedule.items.hash(&mut hasher);
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
            table
                .allocate_output_after_count(runtime, 3)
                .unwrap()
                .shape,
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
        assert!(runtime.plan().validate_bindings(&input, &wrong_mask).is_err());
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
        assert!(matches!(
            mixed.items[2].output,
            ScheduledOutputDesc::Runtime(_)
        ));
        assert_eq!(mixed.runtime_output(2).unwrap().rank, 1);
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
}
