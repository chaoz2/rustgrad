//! Descriptor-only construction for ordered whole-buffer effects.

use super::{
    BufferState, EffectError, EffectPlan, EffectStep, ValidatedEffectPlan, validate_buffer_state,
};
use std::collections::{BTreeMap, btree_map::Entry};

/// Incremental descriptor-only builder for whole-buffer replacements.
///
/// It mirrors `EffectGraph::insert` followed by `EffectGraph::assign`, but owns
/// no host payload. The caller controls source order and may interleave its own
/// validation between additions before sealing the complete plan.
#[derive(Debug, Default)]
pub(crate) struct WholeBufferEffectPlanBuilder {
    states: BTreeMap<u64, BufferState>,
    steps: Vec<EffectStep>,
}

impl WholeBufferEffectPlanBuilder {
    pub(crate) fn push(
        mut self,
        destination: BufferState,
        source_buffer: u64,
    ) -> Result<Self, EffectError> {
        match self.states.entry(destination.buffer) {
            Entry::Occupied(_) => {
                return Err(EffectError::DuplicateWrite {
                    buffer: destination.buffer,
                    version: 0,
                });
            }
            Entry::Vacant(entry) => {
                validate_buffer_state(&destination)?;
                entry.insert(destination.clone());
            }
        }

        let source = BufferState {
            buffer: source_buffer,
            version: 0,
            shape: destination.shape.clone(),
            dtype: destination.dtype,
            bytes: destination.bytes,
        };
        match self.states.entry(source_buffer) {
            Entry::Occupied(_) => {
                return Err(EffectError::DuplicateWrite {
                    buffer: source_buffer,
                    version: 0,
                });
            }
            Entry::Vacant(entry) => {
                validate_buffer_state(&source)?;
                entry.insert(source.clone());
            }
        }

        let next = BufferState {
            buffer: destination.buffer,
            version: destination
                .version
                .checked_add(1)
                .ok_or(EffectError::Overflow)?,
            shape: destination.shape.clone(),
            dtype: destination.dtype,
            bytes: destination.bytes,
        };
        let id = u64::try_from(self.steps.len()).map_err(|_| EffectError::Overflow)?;
        self.steps.push(EffectStep {
            id,
            reads: vec![destination, source],
            write: next.clone(),
            target_view: None,
            index_plan: None,
            after: self.steps.last().map(|step| step.id).into_iter().collect(),
        });
        self.states.insert(next.buffer, next);
        Ok(self)
    }

    pub(crate) fn finish(self) -> Result<WholeBufferEffectPlan, EffectError> {
        let validated = ValidatedEffectPlan::new(EffectPlan { steps: self.steps })?;
        let mut states = BTreeMap::new();
        for step in &validated.plan().steps {
            for state in step.reads.iter().chain([&step.write]) {
                states.insert((state.buffer, state.version), state.clone());
            }
        }
        Ok(WholeBufferEffectPlan {
            validated,
            states: states.into_values().collect(),
        })
    }
}

/// One validated whole-buffer plan and its canonical logical-state inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WholeBufferEffectPlan {
    validated: ValidatedEffectPlan,
    states: Vec<BufferState>,
}

impl WholeBufferEffectPlan {
    pub(crate) fn validated(&self) -> &ValidatedEffectPlan {
        &self.validated
    }

    pub(crate) fn states(&self) -> &[BufferState] {
        &self.states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, EffectGraph, Shape, TensorData};

    fn state(buffer: u64, shape: Shape, dtype: DType) -> BufferState {
        let bytes = shape.numel().unwrap() * dtype.itemsize();
        BufferState {
            buffer,
            version: 0,
            shape,
            dtype,
            bytes,
        }
    }

    #[test]
    fn whole_buffer_plan_matches_payload_graph_for_scalar_tensor_and_empty() {
        let cases = [
            (10, 1, Shape::from([]), DType::F32),
            (11, 2, Shape::from([2, 3]), DType::F32),
            (12, 3, Shape::from([0, 4]), DType::U64),
        ];
        let mut builder = WholeBufferEffectPlanBuilder::default();
        let mut graph = EffectGraph::default();
        for (destination, source, shape, dtype) in cases {
            let value = TensorData::zeros_with_dtype(shape.clone(), dtype).unwrap();
            let target = graph.insert(destination, value.clone()).unwrap();
            let source_handle = graph.insert(source, value).unwrap();
            graph.assign(&target, &source_handle).unwrap();
            builder = builder
                .push(state(destination, shape, dtype), source)
                .unwrap();
        }
        let planned = builder.finish().unwrap();
        assert_eq!(planned.validated().plan(), &graph.plan());

        let mut expected = BTreeMap::new();
        for step in graph.plan().steps {
            for state in step.reads.into_iter().chain([step.write]) {
                expected.insert((state.buffer, state.version), state);
            }
        }
        assert_eq!(planned.states(), expected.into_values().collect::<Vec<_>>());
    }

    #[test]
    fn whole_buffer_plan_preserves_duplicate_before_descriptor_error_order() {
        let builder = WholeBufferEffectPlanBuilder::default()
            .push(state(10, Shape::from([2]), DType::F32), 1)
            .unwrap();
        assert_eq!(
            builder
                .push(
                    BufferState {
                        buffer: 10,
                        version: 0,
                        shape: Shape::from([usize::MAX]),
                        dtype: DType::F32,
                        bytes: 0,
                    },
                    2,
                )
                .unwrap_err(),
            EffectError::DuplicateWrite {
                buffer: 10,
                version: 0,
            }
        );
    }

    #[test]
    fn whole_buffer_plan_validates_descriptor_bytes_without_host_payload() {
        assert_eq!(
            WholeBufferEffectPlanBuilder::default()
                .push(
                    BufferState {
                        buffer: 10,
                        version: 0,
                        shape: Shape::from([2]),
                        dtype: DType::F32,
                        bytes: 7,
                    },
                    1,
                )
                .unwrap_err(),
            EffectError::InvalidBytes { buffer: 10 }
        );
        assert_eq!(
            WholeBufferEffectPlanBuilder::default()
                .push(
                    BufferState {
                        buffer: 10,
                        version: 0,
                        shape: Shape::from([usize::MAX, 2]),
                        dtype: DType::F32,
                        bytes: 0,
                    },
                    1,
                )
                .unwrap_err(),
            EffectError::Overflow
        );
    }

    #[test]
    fn whole_buffer_plan_rejects_source_collision_after_destination_admission() {
        let builder = WholeBufferEffectPlanBuilder::default()
            .push(state(10, Shape::from([1]), DType::F32), 1)
            .unwrap();
        assert_eq!(
            builder
                .push(state(11, Shape::from([1]), DType::F32), 1)
                .unwrap_err(),
            EffectError::DuplicateWrite {
                buffer: 1,
                version: 0,
            }
        );
    }

    #[test]
    fn whole_buffer_plan_preserves_collision_insertion_order() {
        let shape = Shape::from([1]);
        let cases = [
            (Vec::new(), 10, 10, 10),
            (vec![(10, 1)], 10, 2, 10),
            (vec![(10, 1)], 11, 1, 1),
            (vec![(10, 1)], 1, 2, 1),
            (vec![(10, 1)], 11, 10, 10),
            (vec![(10, 1)], 1, 10, 1),
        ];
        for (prefix, destination, source, rejected_buffer) in cases {
            let mut builder = WholeBufferEffectPlanBuilder::default();
            for (prefix_destination, prefix_source) in prefix {
                builder = builder
                    .push(
                        state(prefix_destination, shape.clone(), DType::F32),
                        prefix_source,
                    )
                    .unwrap();
            }
            assert_eq!(
                builder
                    .push(state(destination, shape.clone(), DType::F32), source)
                    .unwrap_err(),
                EffectError::DuplicateWrite {
                    buffer: rejected_buffer,
                    version: 0,
                }
            );
        }
    }
}
