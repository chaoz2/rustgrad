//! Descriptor-only sealing of recurrent whole-buffer replacements.

use super::*;
use crate::effects::WholeBufferEffectPlanBuilder;
use crate::schedule::schedule_whole_buffer_effects;

#[derive(Clone, Debug)]
pub(super) struct RecurrentEffectReplacement {
    pub(super) source: NodeId,
    pub(super) destination: BufferState,
}

pub(super) struct RecurrentEffectSeal {
    pub(super) schedule: crate::Schedule,
    pub(super) states: Vec<BufferState>,
}

struct PureOutputIndex<'a> {
    outputs: BTreeMap<u64, (usize, &'a crate::ScheduleItem)>,
}

impl<'a> PureOutputIndex<'a> {
    fn new(schedule: &'a crate::Schedule) -> Self {
        let mut outputs = BTreeMap::new();
        for (position, item) in schedule.items.iter().enumerate() {
            outputs
                .entry(item.primary_output().id)
                .or_insert((position, item));
        }
        Self { outputs }
    }

    fn value_binding(&self, node: NodeId, effect_item: u64) -> Result<crate::ScheduleValueBinding> {
        let (producer_item, producer) = self
            .outputs
            .get(&(node.index() as u64))
            .ok_or_else(|| training("compiled update output is not materialized"))?;
        Ok(crate::ScheduleValueBinding {
            producer_item: u64::try_from(*producer_item)
                .map_err(|_| training("compiled producer index overflow"))?,
            producer_node: node,
            producer_output: producer.primary_output().clone(),
            abi_index: 0,
            effect_item,
            source_position: 0,
        })
    }
}

pub(super) fn seal_recurrent_effects(
    pure: crate::Schedule,
    replacements: impl IntoIterator<Item = RecurrentEffectReplacement>,
) -> Result<RecurrentEffectSeal> {
    let outputs = PureOutputIndex::new(&pure);
    let mut builder = WholeBufferEffectPlanBuilder::default();
    let mut bindings = Vec::new();
    for (ordinal, replacement) in replacements.into_iter().enumerate() {
        if replacement.source.index() as u64 >= STATE_BUFFER_BASE {
            return Err(training(
                "graph node identity overlaps persistent state namespace",
            ));
        }
        builder = builder
            .push(replacement.destination, replacement.source.index() as u64)
            .map_err(effect_error)?;
        let effect_item = u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?;
        bindings.push(outputs.value_binding(replacement.source, effect_item)?);
    }
    let plan = builder.finish().map_err(effect_error)?;
    let effects = schedule_whole_buffer_effects(&plan).map_err(schedule_error)?;
    let states = plan.states().to_vec();
    let schedule =
        crate::combine_mixed_schedules(pure, effects, bindings).map_err(schedule_error)?;
    Ok(RecurrentEffectSeal { schedule, states })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, EffectGraph, Graph, Shape, TensorData, schedule_effects, schedule_many};

    fn state(buffer: u64, shape: Shape, dtype: DType) -> BufferState {
        BufferState {
            buffer,
            version: 0,
            bytes: shape.numel().unwrap() * dtype.itemsize(),
            shape,
            dtype,
        }
    }

    fn reference_value_binding(
        schedule: &crate::Schedule,
        node: NodeId,
        effect_item: u64,
    ) -> Result<crate::ScheduleValueBinding> {
        let (producer_item, producer) = schedule
            .items
            .iter()
            .enumerate()
            .find(|(_, item)| item.primary_output().id == node.index() as u64)
            .ok_or_else(|| training("compiled update output is not materialized"))?;
        Ok(crate::ScheduleValueBinding {
            producer_item: u64::try_from(producer_item)
                .map_err(|_| training("compiled producer index overflow"))?,
            producer_node: node,
            producer_output: producer.primary_output().clone(),
            abi_index: 0,
            effect_item,
            source_position: 0,
        })
    }

    fn reference_seal(
        pure: crate::Schedule,
        replacements: &[(NodeId, u64, TensorData)],
    ) -> Result<RecurrentEffectSeal> {
        let mut effects = EffectGraph::default();
        let mut bindings = Vec::with_capacity(replacements.len());
        for (ordinal, (source, destination_buffer, value)) in replacements.iter().enumerate() {
            if source.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let destination = effects
                .insert(*destination_buffer, value.clone())
                .map_err(effect_error)?;
            let source_handle = effects
                .insert(
                    source.index() as u64,
                    TensorData::zeros_with_dtype(value.shape().clone(), value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source_handle)
                .map_err(effect_error)?;
            let effect_item =
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?;
            bindings.push(reference_value_binding(&pure, *source, effect_item)?);
        }
        let schedule = crate::combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            bindings,
        )
        .map_err(schedule_error)?;
        let plan = effects.plan();
        plan.validate().map_err(effect_error)?;
        let mut states = BTreeMap::new();
        for step in plan.steps {
            for state in step.reads.into_iter().chain([step.write]) {
                states.insert((state.buffer, state.version), state);
            }
        }
        Ok(RecurrentEffectSeal {
            schedule,
            states: states.into_values().collect(),
        })
    }

    fn mixed_bytes(graph: &Graph, unbound: &crate::Schedule, seal: RecurrentEffectSeal) -> Vec<u8> {
        let mut captured = CapturedSchedule::capture(graph, unbound, &[]).unwrap();
        captured.items = seal.schedule.items.clone();
        CapturedMixedSchedule::from_parts(captured, &seal.schedule, seal.states)
            .unwrap()
            .to_bytes()
            .unwrap()
    }

    fn assert_schedule_fields_equal(actual: &crate::Schedule, expected: &crate::Schedule) {
        assert_eq!(
            actual.requested_materializations,
            expected.requested_materializations
        );
        assert_eq!(
            actual.requested_passthroughs,
            expected.requested_passthroughs
        );
        assert_eq!(actual.value_bindings, expected.value_bindings);
        assert_eq!(actual.state_bindings, expected.state_bindings);
        assert_eq!(actual.items.len(), expected.items.len());
        for (actual_item, expected_item) in actual.items.iter().zip(&expected.items) {
            assert_eq!(actual_item.id, expected_item.id);
            assert_eq!(actual_item.node, expected_item.node);
            assert_eq!(actual_item.dependencies, expected_item.dependencies);
            assert_eq!(actual_item.consumers, expected_item.consumers);
            assert_eq!(actual_item.inputs, expected_item.inputs);
            assert_eq!(actual_item.input_bindings, expected_item.input_bindings);
            assert_eq!(
                actual_item.quantized_input_bindings,
                expected_item.quantized_input_bindings
            );
            assert_eq!(
                actual_item.external_materializations,
                expected_item.external_materializations
            );
            assert_eq!(actual_item.outputs, expected_item.outputs);
            assert_eq!(actual_item.kernel, expected_item.kernel);
            assert_eq!(actual_item.boundary, expected_item.boundary);
            assert_eq!(actual_item.cache_key, expected_item.cache_key);
        }
    }

    #[test]
    fn recurrent_effect_seal_matches_payload_oracle_and_mixed_bytes() {
        let mut graph = Graph::new();
        let tensor_input = graph.input_dtype("tensor", [2], DType::F32);
        let tensor = graph.square(tensor_input).unwrap();
        let scalar_input = graph.input_dtype("scalar", [], DType::F32);
        let scalar = graph.square(scalar_input).unwrap();
        let empty_input = graph.input_dtype("empty", [0, 3], DType::F32);
        let empty = graph.square(empty_input).unwrap();
        let unbound = schedule_many(&graph, &[tensor, scalar, empty]).unwrap();

        let state_inputs = BTreeMap::from([
            (
                tensor_input,
                state(STATE_BUFFER_BASE, Shape::from([2]), DType::F32),
            ),
            (
                scalar_input,
                state(STATE_BUFFER_BASE + 1, Shape::from([]), DType::F32),
            ),
            (
                empty_input,
                state(STATE_BUFFER_BASE + 2, Shape::from([0, 3]), DType::F32),
            ),
        ]);
        let bindings = collect_state_bindings(&unbound, &state_inputs).unwrap();
        let pure = bind_schedule_states(unbound.clone(), bindings).unwrap();
        let replacements = vec![
            (
                tensor,
                STATE_BUFFER_BASE,
                TensorData::zeros_with_dtype(Shape::from([2]), DType::F32).unwrap(),
            ),
            (
                scalar,
                STATE_BUFFER_BASE + 1,
                TensorData::zeros_with_dtype(Shape::from([]), DType::F32).unwrap(),
            ),
            (
                empty,
                STATE_BUFFER_BASE + 2,
                TensorData::zeros_with_dtype(Shape::from([0, 3]), DType::F32).unwrap(),
            ),
        ];
        let planned = seal_recurrent_effects(
            pure.clone(),
            replacements
                .iter()
                .map(|(source, destination, value)| RecurrentEffectReplacement {
                    source: *source,
                    destination: state(*destination, value.shape().clone(), value.dtype()),
                }),
        )
        .unwrap();
        let reference = reference_seal(pure, &replacements).unwrap();
        assert_eq!(planned.states, reference.states);
        assert_schedule_fields_equal(&planned.schedule, &reference.schedule);
        assert_eq!(
            mixed_bytes(&graph, &unbound, planned),
            mixed_bytes(&graph, &unbound, reference)
        );
    }

    #[test]
    fn recurrent_effect_seal_preserves_first_primary_output_match() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], DType::F32);
        let rhs = graph.input_dtype("rhs", [3], DType::F32);
        let first = graph.square(lhs).unwrap();
        let second = graph.square(rhs).unwrap();
        let mut pure = schedule_many(&graph, &[first, second]).unwrap();
        let first_desc = pure.items[0].primary_output().clone();
        let later_matching_desc = pure.items[1].primary_output().clone();
        assert_ne!(first_desc.shape, later_matching_desc.shape);
        pure.items[1].outputs = crate::ScheduledOutputs::single(crate::BufferDesc {
            id: first.index() as u64,
            ..later_matching_desc
        });

        let planned = PureOutputIndex::new(&pure).value_binding(first, 0).unwrap();
        let reference = reference_value_binding(&pure, first, 0).unwrap();
        assert_eq!(planned, reference);
        assert_eq!(planned.producer_item, 0);
        assert_eq!(planned.producer_output, first_desc);
        assert_ne!(planned.producer_output.shape, Shape::from([3]));
    }

    #[test]
    fn recurrent_effect_seal_preserves_late_payload_descriptor_error() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let output = graph.square(input).unwrap();
        let pure = schedule_many(&graph, &[output]).unwrap();
        let replacement = RecurrentEffectReplacement {
            source: output,
            destination: state(STATE_BUFFER_BASE, Shape::from([3]), DType::F32),
        };

        assert_eq!(
            seal_recurrent_effects(pure.clone(), [replacement])
                .err()
                .unwrap(),
            Error::SessionTraining {
                reason: "compiled schedule: schedule error: Binding(\"value binding payload descriptor mismatch\")"
                    .into(),
            }
        );
        let reference = reference_seal(
            pure,
            &[(
                output,
                STATE_BUFFER_BASE,
                TensorData::zeros_with_dtype(Shape::from([3]), DType::F32).unwrap(),
            )],
        )
        .err()
        .unwrap();
        assert_eq!(
            reference,
            Error::SessionTraining {
                reason: "compiled schedule: schedule error: Binding(\"value binding payload descriptor mismatch\")"
                    .into(),
            }
        );
    }

    #[test]
    fn recurrent_effect_seal_preserves_cross_record_error_precedence() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [], DType::F32);
        let valid = graph.square(input).unwrap();
        let pure = schedule_many(&graph, &[valid]).unwrap();
        let missing = NodeId::from_index(valid.index() + 17);
        let destination = state(STATE_BUFFER_BASE, Shape::from([]), DType::F32);

        let missing_first = seal_recurrent_effects(
            pure.clone(),
            [
                RecurrentEffectReplacement {
                    source: missing,
                    destination: destination.clone(),
                },
                RecurrentEffectReplacement {
                    source: valid,
                    destination: destination.clone(),
                },
            ],
        )
        .err()
        .unwrap();
        assert_eq!(
            missing_first,
            Error::SessionTraining {
                reason: "compiled update output is not materialized".into(),
            }
        );

        let duplicate_first = seal_recurrent_effects(
            pure,
            [
                RecurrentEffectReplacement {
                    source: valid,
                    destination: destination.clone(),
                },
                RecurrentEffectReplacement {
                    source: missing,
                    destination,
                },
            ],
        )
        .err()
        .unwrap();
        assert_eq!(
            duplicate_first,
            Error::SessionTraining {
                reason: format!(
                    "compiled effect graph: {:?}",
                    crate::EffectError::DuplicateWrite {
                        buffer: STATE_BUFFER_BASE,
                        version: 0,
                    }
                ),
            }
        );
    }

    #[test]
    fn recurrent_effect_seal_checks_namespace_before_descriptor_collisions() {
        let pure = crate::Schedule {
            items: Vec::new(),
            requested_materializations: Vec::new(),
            requested_passthroughs: Vec::new(),
            value_bindings: Vec::new(),
            state_bindings: Vec::new(),
        };
        let error = seal_recurrent_effects(
            pure,
            [RecurrentEffectReplacement {
                source: NodeId::from_index(STATE_BUFFER_BASE as usize),
                destination: state(STATE_BUFFER_BASE, Shape::from([]), DType::F32),
            }],
        )
        .err()
        .unwrap();
        assert_eq!(
            error,
            Error::SessionTraining {
                reason: "graph node identity overlaps persistent state namespace".into(),
            }
        );
    }
}
