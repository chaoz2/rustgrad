//! Immutable static schedule and logical-memory inspection.
use super::{BufferDesc, Schedule, ScheduleError, ScheduledOutputs, schedule_many};
use crate::{CapturedSchedule, Graph, MemoryPlan, MemoryPlanError, NodeId, Operation};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

/// One scheduled operation retained in a static execution summary.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionPlanItemSummary {
    pub item_id: u64,
    /// Ordered producer-owned descriptors.
    pub outputs: ScheduledOutputs,
    pub operation: Operation,
    pub dependencies: Vec<u64>,
}

impl ExecutionPlanItemSummary {
    /// Canonical first descriptor for one-output inspection paths.
    pub fn primary_output(&self) -> &BufferDesc {
        self.outputs.primary()
    }
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
        let owners = requested
            .iter()
            .map(|node| {
                graph
                    .contiguous_backward_owner(*node)
                    .map_err(|_| ExecutionPlanSummaryError::RequestedOutput(*node))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_schedule(&schedule, &owners, reuse_enabled)
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
        Self::summarize(schedule, requested, requested_outputs, reuse_enabled)
    }

    /// Summarizes one authenticated captured schedule, including direct
    /// input/constant results and source-backed requested aliases. External
    /// passthrough storage is described as an output but never planned as a
    /// temporary allocation.
    pub fn from_capture(
        capture: &CapturedSchedule,
        reuse_enabled: bool,
    ) -> Result<Self, ExecutionPlanSummaryError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|error| captured_summary_error(error.to_string()))?;
        let requested_materializations = super::physical_requested_materializations(
            &capture.items,
            &capture.requested_passthroughs,
            capture.requested.iter().copied(),
        );
        let schedule = Schedule {
            items: capture.items.clone(),
            requested_materializations,
            requested_passthroughs: capture.requested_passthroughs.clone(),
            value_bindings: Vec::new(),
            state_bindings: Vec::new(),
        };
        let passthroughs = capture
            .requested_passthroughs
            .iter()
            .map(|passthrough| (passthrough.requested.index() as u64, passthrough))
            .collect::<BTreeMap<_, _>>();
        let outputs = capture
            .items
            .iter()
            .flat_map(|item| item.outputs.iter().map(|output| (output.id, output)))
            .collect::<BTreeMap<_, _>>();
        let inputs = capture
            .inputs
            .iter()
            .map(|input| (input.node.index() as u64, input))
            .collect::<BTreeMap<_, _>>();
        let requested = capture
            .requested
            .iter()
            .map(|id| {
                let physical = passthroughs.get(id).map_or(*id, |entry| entry.desc.id);
                usize::try_from(physical)
                    .map(NodeId::from_index)
                    .map_err(|_| captured_summary_error("requested ID overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let requested_outputs = capture
            .requested
            .iter()
            .map(|id| {
                captured_requested_output(&outputs, &passthroughs, &inputs, &capture.constants, *id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::summarize(&schedule, &requested, requested_outputs, reuse_enabled)
    }

    fn summarize(
        schedule: &Schedule,
        requested: &[NodeId],
        requested_outputs: Vec<BufferDesc>,
        reuse_enabled: bool,
    ) -> Result<Self, ExecutionPlanSummaryError> {
        schedule
            .validate()
            .map_err(ExecutionPlanSummaryError::Schedule)?;
        let memory = MemoryPlan::from_schedule(schedule, requested, reuse_enabled)
            .map_err(ExecutionPlanSummaryError::Memory)?;
        let items = schedule
            .items
            .iter()
            .map(|item| ExecutionPlanItemSummary {
                item_id: item.id,
                outputs: item.outputs.clone(),
                operation: item.kernel.operation().clone(),
                dependencies: item.dependencies.clone(),
            })
            .collect::<Vec<_>>();
        let zero_domain_item_count = items
            .iter()
            .filter(|item| {
                item.outputs
                    .iter()
                    .all(|output| output.shape.numel().is_ok_and(|elements| elements == 0))
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

fn captured_requested_output(
    outputs: &BTreeMap<u64, &BufferDesc>,
    passthroughs: &BTreeMap<u64, &crate::RequestedPassthrough>,
    inputs: &BTreeMap<u64, &crate::ReplayInput>,
    constants: &BTreeMap<u64, crate::TensorData>,
    requested: u64,
) -> Result<BufferDesc, ExecutionPlanSummaryError> {
    if let Some(output) = outputs.get(&requested) {
        return Ok((**output).clone());
    }
    if let Some(passthrough) = passthroughs.get(&requested) {
        return Ok(passthrough.desc.clone());
    }
    if let Some(input) = inputs.get(&requested) {
        return Ok(input.desc.clone());
    }
    let Some(value) = constants.get(&requested) else {
        return Err(captured_summary_error(format!(
            "requested output {requested} is absent"
        )));
    };
    let dtype = value.dtype();
    let bytes = value
        .to_le_bytes()
        .map_err(|error| captured_summary_error(error.to_string()))?
        .len();
    Ok(BufferDesc {
        id: requested,
        shape: value.shape().clone(),
        dtype,
        bytes,
        alignment: dtype.itemsize().max(1),
        read_only: true,
        view: None,
    })
}

fn captured_summary_error(reason: impl Into<String>) -> ExecutionPlanSummaryError {
    ExecutionPlanSummaryError::Schedule(ScheduleError::Binding(format!(
        "captured execution summary: {}",
        reason.into()
    )))
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

    #[test]
    fn captured_summary_represents_external_duplicate_outputs_without_allocations() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2], DType::F32);
        let constant = graph.constant(TensorData::new([2], vec![3.0, 4.0]).unwrap());
        let transposed = graph.permute(input, [1, 0]).unwrap();
        let requested = [input, constant, input, transposed, transposed];
        let schedule = schedule_many(&graph, &requested).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &requested).unwrap();
        assert_eq!(capture.requested_passthroughs.len(), 1);
        let summary = ExecutionPlanSummary::from_capture(&capture, true).unwrap();

        assert_eq!(summary.schedule_item_count, 0);
        assert_eq!(summary.temporary_allocation_count, 0);
        assert_eq!(summary.peak_logical_allocations, 0);
        assert_eq!(summary.peak_logical_bytes, 0);
        assert_eq!(
            summary
                .requested_outputs
                .iter()
                .map(|output| output.id)
                .collect::<Vec<_>>(),
            [
                input.index() as u64,
                constant.index() as u64,
                input.index() as u64,
                input.index() as u64,
                input.index() as u64,
            ]
        );
        assert_eq!(summary.requested_outputs[3].shape, Shape::from([1, 2]));
        assert_eq!(
            summary.requested_outputs[3]
                .view
                .as_ref()
                .unwrap()
                .logical_shape,
            Shape::from([2, 1])
        );
        assert_eq!(summary.requested_outputs[3], summary.requested_outputs[4]);

        let mut tampered = capture;
        tampered.identity ^= 1;
        assert!(ExecutionPlanSummary::from_capture(&tampered, true).is_err());
    }
}
