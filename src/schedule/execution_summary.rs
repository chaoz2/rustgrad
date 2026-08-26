//! Immutable static schedule and logical-memory inspection.
use super::{BufferDesc, Schedule, ScheduleError, schedule_many};
use crate::{Graph, MemoryPlan, MemoryPlanError, NodeId, UOpKind};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

/// One scheduled operation retained in a static execution summary.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionPlanItemSummary {
    pub item_id: u64,
    /// Ordered producer-owned descriptors. `output` remains the primary
    /// compatibility projection for one-output callers.
    pub outputs: Vec<BufferDesc>,
    pub output: BufferDesc,
    pub operation: UOpKind,
    pub dependencies: Vec<u64>,
}

/// Deterministic logical facts derived from one validated static schedule and
/// its existing host `MemoryPlan`. This is not a profiler or runtime counter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlanSummary {
    pub requested_outputs: Vec<BufferDesc>,
    pub items: Vec<ExecutionPlanItemSummary>,
    pub schedule_item_count: usize,
    pub temporary_allocation_count: usize,
    pub peak_logical_allocations: usize,
    pub peak_logical_bytes: usize,
    pub reuse_enabled: bool,
    pub reuse_count: usize,
    pub zero_domain_item_count: usize,
    pub zero_byte_sentinel_count: usize,
    pub identity: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionPlanSummaryError {
    Schedule(ScheduleError),
    Memory(MemoryPlanError),
    RequestedOutput(NodeId),
}

impl fmt::Display for ExecutionPlanSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "execution plan summary error: {self:?}")
    }
}
impl std::error::Error for ExecutionPlanSummaryError {}

impl ExecutionPlanSummary {
    /// Plans but never executes a concrete static graph request.
    pub fn from_graph(
        graph: &Graph,
        requested: &[NodeId],
        reuse_enabled: bool,
    ) -> Result<Self, ExecutionPlanSummaryError> {
        let schedule =
            schedule_many(graph, requested).map_err(ExecutionPlanSummaryError::Schedule)?;
        Self::from_schedule(&schedule, requested, reuse_enabled)
    }

    /// Summarizes an already constructed schedule without constructing a
    /// second planning path or executing any item.
    pub fn from_schedule(
        schedule: &Schedule,
        requested: &[NodeId],
        reuse_enabled: bool,
    ) -> Result<Self, ExecutionPlanSummaryError> {
        schedule
            .validate()
            .map_err(ExecutionPlanSummaryError::Schedule)?;
        let memory = MemoryPlan::from_schedule(schedule, requested, reuse_enabled)
            .map_err(ExecutionPlanSummaryError::Memory)?;
        let outputs = schedule
            .items
            .iter()
            .flat_map(|item| item.outputs.iter().cloned())
            .map(|output| (output.id, output))
            .collect::<BTreeMap<_, _>>();
        let requested_outputs = requested
            .iter()
            .map(|node| {
                outputs
                    .get(&(node.index() as u64))
                    .cloned()
                    .ok_or(ExecutionPlanSummaryError::RequestedOutput(*node))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let items = schedule
            .items
            .iter()
            .map(|item| ExecutionPlanItemSummary {
                item_id: item.id,
                outputs: item.outputs.iter().cloned().collect(),
                output: item.primary_output().clone(),
                operation: item.kernel.kind().clone(),
                dependencies: item.dependencies.clone(),
            })
            .collect::<Vec<_>>();
        let zero_domain_item_count = items
            .iter()
            .filter(|item| {
                item.outputs.iter().all(|output| {
                    output
                        .shape
                        .numel()
                        .is_ok_and(|elements| elements == 0)
                })
            })
            .count();
        let zero_byte_sentinel_count = memory
            .temporaries
            .iter()
            .filter(|temporary| temporary.allocation_id.is_none())
            .count();
        let reuse_count = memory
            .temporaries
            .iter()
            .filter(|temporary| temporary.reused_from.is_some())
            .count();
        let mut summary = Self {
            requested_outputs,
            schedule_item_count: items.len(),
            items,
            temporary_allocation_count: memory.temporaries.len(),
            peak_logical_allocations: memory.peak_allocations,
            peak_logical_bytes: memory.peak_bytes,
            reuse_enabled,
            reuse_count,
            zero_domain_item_count,
            zero_byte_sentinel_count,
            identity: 0,
        };
        summary.identity = summary.logical_identity();
        Ok(summary)
    }

    fn logical_identity(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.requested_outputs.hash(&mut hasher);
        self.items.hash(&mut hasher);
        self.schedule_item_count.hash(&mut hasher);
        self.temporary_allocation_count.hash(&mut hasher);
        self.peak_logical_allocations.hash(&mut hasher);
        self.peak_logical_bytes.hash(&mut hasher);
        self.reuse_enabled.hash(&mut hasher);
        self.reuse_count.hash(&mut hasher);
        self.zero_domain_item_count.hash(&mut hasher);
        self.zero_byte_sentinel_count.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape, TensorData};

    #[test]
    fn agrees_with_the_canonical_schedule_and_memory_plan() {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::from([2, 2]));
        let squared = graph.square(input).unwrap();
        let output = graph.sum(squared, 1).unwrap();
        let schedule = schedule_many(&graph, &[output]).unwrap();
        for reuse in [false, true] {
            let summary = ExecutionPlanSummary::from_graph(&graph, &[output], reuse).unwrap();
            let memory = MemoryPlan::from_schedule(&schedule, &[output], reuse).unwrap();
            assert_eq!(summary.temporary_allocation_count, memory.temporaries.len());
            assert_eq!(summary.schedule_item_count, schedule.items.len());
            assert_eq!(summary.peak_logical_allocations, memory.peak_allocations);
            assert_eq!(summary.peak_logical_bytes, memory.peak_bytes);
            assert_eq!(
                summary.reuse_count,
                memory
                    .temporaries
                    .iter()
                    .filter(|temporary| temporary.reused_from.is_some())
                    .count()
            );
            assert_eq!(
                summary,
                ExecutionPlanSummary::from_graph(&graph, &[output], reuse).unwrap()
            );
        }
    }

    #[test]
    fn records_empty_output_without_execution() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0, 2], DType::F32);
        let one = graph.constant(TensorData::scalar(1.0));
        let output = graph.add(input, one).unwrap();
        let summary = ExecutionPlanSummary::from_graph(&graph, &[output], true).unwrap();
        assert_eq!(summary.requested_outputs[0].bytes, 0);
        assert_eq!(summary.zero_domain_item_count, 1);
    }
}
