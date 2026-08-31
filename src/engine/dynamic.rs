//! Exact allocation seam for runtime-sized instruction schedules.
//!
//! Runtime operands and outputs are owned by the instruction variants in the
//! schedule. This module owns only the allocation table; it cannot invent a
//! side binding or reinterpret an instruction.

use crate::DynamicAllocation;
#[cfg(test)]
use crate::schedule::dynamic::RuntimeBufferId;
use crate::schedule::dynamic::{
    RuntimeBufferDesc, RuntimeBufferTable, RuntimeCount, RuntimeSchedule, RuntimeScheduleError,
};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeMaterializationError {
    Schedule(RuntimeScheduleError),
}

impl fmt::Display for RuntimeMaterializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "runtime materialization error: {self:?}")
    }
}
impl std::error::Error for RuntimeMaterializationError {}

/// Allocation state for one validated runtime instruction DAG.
#[derive(Clone, Debug)]
pub(crate) struct RuntimeMaterializationMap {
    table: RuntimeBufferTable,
}

impl RuntimeMaterializationMap {
    pub(crate) fn new(schedule: &RuntimeSchedule) -> Result<Self, RuntimeMaterializationError> {
        Ok(Self {
            table: RuntimeBufferTable::new(schedule)
                .map_err(RuntimeMaterializationError::Schedule)?,
        })
    }

    pub(crate) fn allocate(
        &mut self,
        schedule: &RuntimeSchedule,
        output: &RuntimeBufferDesc,
        count: RuntimeCount,
    ) -> Result<&DynamicAllocation, RuntimeMaterializationError> {
        self.table
            .allocate_buffer_after_count(schedule, output.id, count)
            .map_err(RuntimeMaterializationError::Schedule)
    }

    pub(crate) fn allocation(
        &self,
        output: &RuntimeBufferDesc,
    ) -> Result<&DynamicAllocation, RuntimeMaterializationError> {
        self.table
            .allocation(output.id)
            .map_err(RuntimeMaterializationError::Schedule)
    }

    #[cfg(test)]
    pub(crate) fn allocation_by_id(
        &self,
        id: RuntimeBufferId,
    ) -> Result<&DynamicAllocation, RuntimeMaterializationError> {
        self.table
            .allocation(id)
            .map_err(RuntimeMaterializationError::Schedule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::dynamic::{RuntimeInstruction, schedule_dynamic};
    use crate::{DType, Graph};

    #[test]
    fn allocations_follow_the_instruction_owned_descriptor() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
        let output = graph.masked_select_dynamic(input, mask).unwrap();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let descriptor = schedule
            .items
            .iter()
            .find_map(|item| match &item.instruction {
                RuntimeInstruction::Allocate { output } => Some(output),
                _ => None,
            })
            .unwrap();
        let mut allocations = RuntimeMaterializationMap::new(&schedule).unwrap();
        assert!(matches!(
            allocations.allocation(descriptor),
            Err(RuntimeMaterializationError::Schedule(
                RuntimeScheduleError::LiveLookupBeforeAllocation(_)
            ))
        ));
        assert_eq!(
            allocations
                .allocate(
                    &schedule,
                    descriptor,
                    RuntimeCount {
                        id: descriptor.count,
                        value: 3,
                    },
                )
                .unwrap()
                .bytes,
            12
        );
        assert_eq!(
            allocations.allocation_by_id(descriptor.id).unwrap().shape,
            crate::Shape::from([3])
        );
    }
}
