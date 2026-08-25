//! Shared materialization seam for the exact runtime-sized mixed schedule.
//!
//! It owns allocation-to-consumer binding only; CPU semantic evaluation still
//! supplies count and values. No alternate executor, placeholder tensor, or
//! capacity policy is introduced here.

use crate::DynamicAllocation;
use crate::schedule::dynamic::{
    MixedSchedule, MixedScheduleItemKind, RuntimeBufferId, RuntimeBufferTable,
    RuntimeScheduleError,
};
use std::{collections::BTreeMap, fmt};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum MixedMaterializationError {
    Schedule(RuntimeScheduleError),
    MissingRuntimeConsumer(u64),
    LifetimeMismatch { consumer: u64, final_consumer: u64 },
}

impl fmt::Display for MixedMaterializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mixed materialization error: {self:?}")
    }
}
impl std::error::Error for MixedMaterializationError {}

/// Centralized allocation/materialization map for a validated mixed DAG.
/// Runtime buffers acquire an exact allocation after their count item and can
/// then be bound only to an explicit dynamic-capable materialization item.
#[derive(Clone, Debug)]
pub(crate) struct MixedMaterializationMap {
    table: RuntimeBufferTable,
    consumers: BTreeMap<u64, RuntimeBufferId>,
}

impl MixedMaterializationMap {
    pub(crate) fn new(schedule: &MixedSchedule) -> Result<Self, MixedMaterializationError> {
        schedule
            .validate()
            .map_err(MixedMaterializationError::Schedule)?;
        let mut consumers = BTreeMap::new();
        for binding in &schedule.runtime_bindings {
            let item = schedule
                .items
                .get(binding.consumer_item as usize)
                .ok_or(MixedMaterializationError::MissingRuntimeConsumer(
                    binding.consumer_item,
                ))?;
            if !matches!(
                item.kind,
                MixedScheduleItemKind::MaterializeMaskedSelect
                    | MixedScheduleItemKind::DynamicUnary { .. }
                    | MixedScheduleItemKind::DynamicReduceSum
            )
                || consumers
                    .insert(binding.consumer_item, binding.source)
                    .is_some()
            {
                return Err(MixedMaterializationError::MissingRuntimeConsumer(
                    binding.consumer_item,
                ));
            }
        }
        Ok(Self {
            table: RuntimeBufferTable::new(schedule.runtime())
                .map_err(MixedMaterializationError::Schedule)?,
            consumers,
        })
    }

    /// Executes only the allocation half of the validated count→allocate
    /// relation. Values remain outside this map until an explicit consumer
    /// obtains the exact descriptor below.
    pub(crate) fn allocate_after_count(
        &mut self,
        schedule: &MixedSchedule,
        elements: usize,
    ) -> Result<DynamicAllocation, MixedMaterializationError> {
        self.table
            .allocate_buffer_after_count(
                schedule.runtime(),
                schedule.runtime().buffers[0].id,
                elements,
            )
            .cloned()
            .map_err(MixedMaterializationError::Schedule)
    }

    /// Binds a live exact allocation to an explicitly typed dynamic-capable
    /// consumer. Any other item is rejected before its caller can materialize
    /// a tensor value.
    pub(crate) fn allocation_for_consumer(
        &self,
        schedule: &MixedSchedule,
        consumer: u64,
    ) -> Result<&DynamicAllocation, MixedMaterializationError> {
        let buffer = self
            .consumers
            .get(&consumer)
            .copied()
            .ok_or(MixedMaterializationError::MissingRuntimeConsumer(consumer))?;
        let lifetime = schedule
            .lifetimes
            .iter()
            .find(|lifetime| lifetime.buffer_id == buffer.0)
            .ok_or(MixedMaterializationError::MissingRuntimeConsumer(consumer))?;
        if lifetime.final_consumer < consumer {
            return Err(MixedMaterializationError::LifetimeMismatch {
                consumer,
                final_consumer: lifetime.final_consumer,
            });
        }
        self.table
            .allocation(buffer)
            .map_err(MixedMaterializationError::Schedule)
    }

    /// Allocates the distinct output of a dynamic-capable item. The item must
    /// already be structurally validated by `MixedSchedule`; no caller may
    /// obtain a hidden or in-place allocation.
    pub(crate) fn allocate_item_output_after_count(
        &mut self,
        schedule: &MixedSchedule,
        item: u64,
        elements: usize,
    ) -> Result<DynamicAllocation, MixedMaterializationError> {
        let item_record = schedule
            .items
            .get(
                usize::try_from(item)
                    .map_err(|_| MixedMaterializationError::MissingRuntimeConsumer(item))?,
            )
            .ok_or(MixedMaterializationError::MissingRuntimeConsumer(item))?;
        if !matches!(item_record.kind, MixedScheduleItemKind::DynamicUnary { .. }) {
            return Err(MixedMaterializationError::MissingRuntimeConsumer(item));
        }
        let descriptor = schedule
            .runtime_output(item)
            .map_err(MixedMaterializationError::Schedule)?;
        self.table
            .allocate_buffer_after_count(schedule.runtime(), descriptor.id, elements)
            .cloned()
            .map_err(MixedMaterializationError::Schedule)
    }

    pub(crate) fn allocation_for_item_output(
        &self,
        schedule: &MixedSchedule,
        item: u64,
    ) -> Result<&DynamicAllocation, MixedMaterializationError> {
        let item_record = schedule
            .items
            .get(
                usize::try_from(item)
                    .map_err(|_| MixedMaterializationError::MissingRuntimeConsumer(item))?,
            )
            .ok_or(MixedMaterializationError::MissingRuntimeConsumer(item))?;
        if !matches!(item_record.kind, MixedScheduleItemKind::DynamicUnary { .. }) {
            return Err(MixedMaterializationError::MissingRuntimeConsumer(item));
        }
        let descriptor = schedule
            .runtime_output(item)
            .map_err(MixedMaterializationError::Schedule)?;
        self.table
            .allocation(descriptor.id)
            .map_err(MixedMaterializationError::Schedule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Graph};
    use crate::schedule::dynamic::{schedule_dynamic, schedule_dynamic_unary};

    fn fixture() -> (Graph, crate::DynamicNodeId) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
        let output = graph.masked_select_dynamic(input, mask).unwrap();
        (graph, output)
    }

    #[test]
    fn exact_allocation_binds_only_to_materialization_consumer() {
        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let mut map = MixedMaterializationMap::new(&schedule).unwrap();
        assert!(matches!(
            map.allocation_for_consumer(&schedule, 2),
            Err(MixedMaterializationError::Schedule(
                RuntimeScheduleError::LiveLookupBeforeAllocation(_)
            ))
        ));
        assert_eq!(map.allocate_after_count(&schedule, 0).unwrap().bytes, 0);
        assert_eq!(map.allocation_for_consumer(&schedule, 2).unwrap().elements, 0);
        assert_eq!(
            map.allocation_for_consumer(&schedule, 0),
            Err(MixedMaterializationError::MissingRuntimeConsumer(0))
        );
    }

    #[test]
    fn nonzero_allocation_observes_the_runtime_final_consumer_lifetime() {
        let (graph, output) = fixture();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let mut map = MixedMaterializationMap::new(&schedule).unwrap();
        let allocation = map.allocate_after_count(&schedule, 3).unwrap();
        assert_eq!(allocation.bytes, 3 * DType::F32.itemsize());
        assert_eq!(schedule.runtime().lifetimes[0].allocation_item, 1);
        assert_eq!(schedule.runtime().lifetimes[0].final_consumer, 2);
        assert_eq!(
            map.allocation_for_consumer(&schedule, 2).unwrap(),
            &allocation
        );
    }

    #[test]
    fn unary_consumer_requires_distinct_source_and_output_allocations() {
        let (mut graph, selected) = fixture();
        let output = graph.dynamic_unary(selected, crate::UnaryOp::Neg).unwrap();
        let schedule = schedule_dynamic_unary(&graph, output).unwrap();
        let mut map = MixedMaterializationMap::new(&schedule).unwrap();
        assert!(matches!(
            map.allocate_item_output_after_count(&schedule, 4, 2),
            Err(MixedMaterializationError::Schedule(
                RuntimeScheduleError::LiveLookupBeforeAllocation(_)
            ))
        ));
        map.allocate_after_count(&schedule, 2).unwrap();
        let output = map.allocate_item_output_after_count(&schedule, 4, 2).unwrap();
        assert_eq!(output.shape, crate::Shape::from([2]));
        assert_eq!(map.allocation_for_consumer(&schedule, 2).unwrap().elements, 2);
        assert_eq!(map.allocation_for_consumer(&schedule, 4).unwrap().elements, 2);
        assert_eq!(map.allocation_for_item_output(&schedule, 4).unwrap(), &output);
    }
}
