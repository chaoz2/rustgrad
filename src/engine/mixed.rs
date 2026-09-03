//! Transactional realization of the explicit pure-value to effect-state edge.
use super::persistent_inputs::bind_persistent_inputs;
use crate::{EffectGraph, EffectRuntime, Graph, Op, ReplayInput, Schedule, TensorData};
use std::collections::{BTreeMap, HashMap};

/// Realizes pure producers into owned temporary values, then commits all
/// affected persistent states through one `EffectRuntime` transaction.
///
/// Only interpreter execution is accepted here. Persistent state-to-pure
/// reads are bound once per logical Graph input before pure execution;
/// native/device execution remains an explicit unsupported boundary.
pub fn realize_mixed_effects(
    runtime: &mut EffectRuntime,
    graph: &Graph,
    effects: &EffectGraph,
    schedule: &Schedule,
    inputs: &HashMap<String, TensorData>,
    injected_failure: Option<u64>,
) -> Result<Vec<crate::BufferState>, super::RealizationError> {
    schedule
        .validate()
        .map_err(|error| super::RealizationError::Schedule(error.to_string()))?;
    if schedule.value_bindings.is_empty() && schedule.state_bindings.is_empty() {
        return Err(super::RealizationError::Unsupported(
            "mixed schedule lacks pure value bindings".into(),
        ));
    }
    let mut pure_items = schedule
        .items
        .iter()
        .take_while(|item| !item.is_effect())
        .cloned()
        .collect::<Vec<_>>();
    if pure_items.is_empty()
        || schedule.items[pure_items.len()..]
            .iter()
            .any(|item| !item.is_effect())
    {
        return Err(super::RealizationError::Unsupported(
            "mixed schedules require an ordered pure then effect DAG".into(),
        ));
    }
    let pure_len = pure_items.len() as u64;
    for item in &mut pure_items {
        item.consumers.retain(|consumer| *consumer < pure_len);
    }
    if pure_items
        .iter()
        .any(|item| item.boundary.is_some() || !item.external_materializations.is_empty())
    {
        return Err(super::RealizationError::Unsupported(
            "mixed pure execution supports only interpreter-lowerable owned outputs".into(),
        ));
    }
    let requested = schedule
        .value_bindings
        .iter()
        .map(|binding| binding.producer_node)
        .collect::<Vec<_>>();
    let pure = Schedule {
        items: pure_items,
        requested_materializations: schedule.requested_materializations.clone(),
        requested_passthroughs: schedule.requested_passthroughs.clone(),
        value_bindings: vec![],
        state_bindings: vec![],
    };
    let mut state_inputs = BTreeMap::new();
    for binding in &schedule.state_bindings {
        let Op::Input { name } = graph
            .op(binding.input_node)
            .map_err(|error| super::RealizationError::Schedule(error.to_string()))?
        else {
            return Err(super::RealizationError::Schedule(
                "state binding input is not graph input".into(),
            ));
        };
        state_inputs
            .entry(binding.input_node)
            .or_insert_with(|| ReplayInput {
                name: name.clone(),
                node: binding.input_node,
                desc: binding.desc.clone(),
            });
    }
    let state_inputs = state_inputs.into_values().collect::<Vec<_>>();
    let provided = inputs
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let pure_inputs = bind_persistent_inputs(
        &state_inputs,
        &schedule.state_bindings,
        &provided,
        |binding| {
            let snapshot = runtime.snapshot(&binding.state).map_err(|error| {
                super::RealizationError::Execution(format!("persistent state read: {error:?}"))
            })?;
            match &binding.view {
                Some(view) => snapshot.tensor().affine_read(view),
                None => Ok(snapshot.tensor().clone()),
            }
            .map_err(|error| {
                super::RealizationError::Execution(format!(
                    "persistent state affine read: {error:?}"
                ))
            })
        },
        |reason| super::RealizationError::Schedule(reason.into()),
        |reason| super::RealizationError::Schedule(reason.into()),
    )?
    .into_iter()
    .collect::<HashMap<_, _>>();
    let realized = super::realize_with_options(
        graph,
        &pure,
        &requested,
        &pure_inputs,
        super::RealizationOptions::default(),
    )?;
    let values = requested
        .into_iter()
        .zip(realized.outputs)
        .collect::<BTreeMap<_, _>>();
    let mut source_values = BTreeMap::new();
    for binding in &schedule.value_bindings {
        let item = &schedule.items[binding.effect_item as usize];
        let step = match item.kernel.operation() {
            crate::Operation::EffectStore(payload) | crate::Operation::After(payload) => {
                payload.step
            }
            _ => {
                return Err(super::RealizationError::Schedule(
                    "effect binding lacks typed payload".into(),
                ));
            }
        };
        source_values.insert(
            step,
            values
                .get(&binding.producer_node)
                .ok_or(super::RealizationError::MissingBuffer(
                    binding.producer_output.id,
                ))?
                .clone(),
        );
    }
    let plan = effects.plan();
    plan.validate()
        .map_err(|error| super::RealizationError::Schedule(error.to_string()))?;
    // This preflight makes versioned base/view lifetimes part of the canonical
    // mixed execution contract before any pure output or persistent mutation.
    crate::AliasLivenessPlan::from_effects(&plan).map_err(super::RealizationError::Memory)?;
    runtime
        .execute_with_sources(&plan, &source_values, injected_failure)
        .map_err(|error| {
            super::RealizationError::Execution(format!(
                "persistent mixed effect runtime: {error:?}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{ScheduleStateBinding, bind_schedule_states};
    use crate::{
        AffineView, BinaryOp, BufferState, DType, EffectGraph, ScheduleValueBinding, Shape,
        Storage, combine_mixed_schedules, schedule, schedule_effects, schedule_many,
    };

    fn normal_and_transposed_state_fixture() -> (
        Graph,
        EffectGraph,
        Schedule,
        EffectRuntime,
        BufferState,
        BufferState,
    ) {
        let mut graph = Graph::new();
        let state_input = graph.input_dtype("state", [2, 3], DType::F32);
        let normal = graph.neg(state_input).unwrap();
        let transposed_view = graph.permute(state_input, [1, 0]).unwrap();
        let transposed = graph.neg(transposed_view).unwrap();
        let pure = schedule_many(&graph, &[normal, transposed]).unwrap();
        let persistent = BufferState {
            buffer: 17,
            version: 0,
            shape: Shape::from([2, 3]),
            dtype: DType::F32,
            bytes: 24,
        };
        let mut state_bindings = Vec::new();
        for item in &pure.items {
            for input in &item.input_bindings {
                if input.input_node == state_input {
                    state_bindings.push(ScheduleStateBinding {
                        state: persistent.clone(),
                        view: None,
                        consumer_item: item.id,
                        consumer_node: item.node,
                        input_node: state_input,
                        desc: input.desc.clone(),
                        abi_index: input.abi_index,
                    });
                }
            }
        }
        assert_eq!(state_bindings.len(), 2);
        assert_ne!(state_bindings[0].desc.view, state_bindings[1].desc.view);
        let pure = bind_schedule_states(pure, state_bindings).unwrap();
        let producer_item = pure
            .items
            .iter()
            .position(|item| item.node == transposed)
            .unwrap();
        let producer_output = pure.items[producer_item].primary_output().clone();

        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                100,
                TensorData::from_storage([3, 2], Storage::F32(vec![0.; 6])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                transposed.index() as u64,
                TensorData::from_storage([3, 2], Storage::F32(vec![0.; 6])).unwrap(),
            )
            .unwrap();
        let next = effects.assign(&target, &source).unwrap();
        let effect = schedule_effects(&effects).unwrap();
        let binding = ScheduleValueBinding {
            producer_item: producer_item as u64,
            producer_node: transposed,
            producer_output,
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        };
        let mixed = combine_mixed_schedules(pure, effect, vec![binding]).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                persistent.buffer,
                TensorData::from_storage([2, 3], Storage::F32(vec![1., 2., 3., 4., 5., 6.]))
                    .unwrap(),
            )
            .unwrap();
        runtime
            .register(
                target.state().buffer,
                TensorData::from_storage([3, 2], Storage::F32(vec![0.; 6])).unwrap(),
            )
            .unwrap();
        (
            graph,
            effects,
            mixed,
            runtime,
            persistent,
            next.state().clone(),
        )
    }

    #[test]
    fn add_is_staged_then_committed_as_one_persistent_transaction() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([1]), DType::F32);
        let y = graph.input_dtype("y", Shape::from([1]), DType::F32);
        let sum = graph.binary(BinaryOp::Add, x, y).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                100,
                TensorData::from_storage([1], Storage::F32(vec![0.0])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                sum.index() as u64,
                TensorData::from_storage([1], Storage::F32(vec![0.0])).unwrap(),
            )
            .unwrap();
        let next = effects.assign(&target, &source).unwrap();
        let effect = schedule_effects(&effects).unwrap();
        let binding = ScheduleValueBinding {
            producer_item: 0,
            producer_node: sum,
            producer_output: pure.items[0].primary_output().clone(),
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        };
        let mixed = combine_mixed_schedules(pure, effect, vec![binding]).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                100,
                TensorData::from_storage([1], Storage::F32(vec![0.0])).unwrap(),
            )
            .unwrap();
        realize_mixed_effects(
            &mut runtime,
            &graph,
            &effects,
            &mixed,
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::from_storage([1], Storage::F32(vec![2.0])).unwrap(),
                ),
                (
                    "y".into(),
                    TensorData::from_storage([1], Storage::F32(vec![3.0])).unwrap(),
                ),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::F32(vec![5.0])
        );
    }

    #[test]
    fn mixed_add_targets_an_injective_base_view() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([1, 2]), DType::F32);
        let y = graph.input_dtype("y", Shape::from([1, 2]), DType::F32);
        let sum = graph.binary(BinaryOp::Add, x, y).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                100,
                TensorData::from_storage([2, 3], Storage::F32(vec![0.; 6])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                sum.index() as u64,
                TensorData::from_storage([1, 2], Storage::F32(vec![0.; 2])).unwrap(),
            )
            .unwrap();
        let view = crate::ViewMap::identity(Shape::from([2, 3]))
            .shrink(&[(1, 2), (1, 3)])
            .unwrap();
        let next = effects.assign_view(&target, &source, view).unwrap();
        let effect = schedule_effects(&effects).unwrap();
        let binding = ScheduleValueBinding {
            producer_item: 0,
            producer_node: sum,
            producer_output: pure.items[0].primary_output().clone(),
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        };
        let mixed = combine_mixed_schedules(pure, effect, vec![binding]).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                100,
                TensorData::from_storage([2, 3], Storage::F32(vec![0.; 6])).unwrap(),
            )
            .unwrap();
        realize_mixed_effects(
            &mut runtime,
            &graph,
            &effects,
            &mixed,
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::from_storage([1, 2], Storage::F32(vec![2., 3.])).unwrap(),
                ),
                (
                    "y".into(),
                    TensorData::from_storage([1, 2], Storage::F32(vec![4., 5.])).unwrap(),
                ),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::F32(vec![0., 0., 0., 0., 6., 8.])
        );
    }

    #[test]
    fn versioned_signed_state_read_feeds_pure_add_then_atomic_effect_commit() {
        let mut graph = Graph::new();
        let state_input = graph.input_dtype("state", [4], DType::F32);
        let bias = graph.input_dtype("bias", [4], DType::F32);
        let sum = graph.binary(BinaryOp::Add, state_input, bias).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let state_input_binding = pure.items[0]
            .input_bindings
            .iter()
            .find(|binding| binding.input_node == state_input)
            .unwrap()
            .clone();

        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                100,
                TensorData::from_storage([4], Storage::F32(vec![0.; 4])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                sum.index() as u64,
                TensorData::from_storage([4], Storage::F32(vec![0.; 4])).unwrap(),
            )
            .unwrap();
        let next = effects.assign(&target, &source).unwrap();
        let state_binding = ScheduleStateBinding {
            state: target.state().clone(),
            view: Some(AffineView::identity(Shape::from([4])).flip(0).unwrap()),
            consumer_item: 0,
            consumer_node: sum,
            input_node: state_input,
            desc: state_input_binding.desc,
            abi_index: state_input_binding.abi_index,
        };
        let pure = bind_schedule_states(pure, vec![state_binding]).unwrap();
        let effect = schedule_effects(&effects).unwrap();
        let binding = ScheduleValueBinding {
            producer_item: 0,
            producer_node: sum,
            producer_output: pure.items[0].primary_output().clone(),
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        };
        let mixed = combine_mixed_schedules(pure, effect, vec![binding]).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                100,
                TensorData::from_storage([4], Storage::F32(vec![1., 2., 3., 4.])).unwrap(),
            )
            .unwrap();
        realize_mixed_effects(
            &mut runtime,
            &graph,
            &effects,
            &mixed,
            &HashMap::from([(
                "bias".into(),
                TensorData::from_storage([4], Storage::F32(vec![10., 20., 30., 40.])).unwrap(),
            )]),
            None,
        )
        .unwrap();
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::F32(vec![14., 23., 32., 41.])
        );
    }

    #[test]
    fn one_state_input_serves_normal_and_transposed_pure_consumers() {
        let (graph, effects, mixed, mut runtime, persistent, next) =
            normal_and_transposed_state_fixture();
        realize_mixed_effects(
            &mut runtime,
            &graph,
            &effects,
            &mixed,
            &HashMap::new(),
            None,
        )
        .unwrap();
        assert_eq!(
            runtime.snapshot(&persistent).unwrap().tensor().storage(),
            &Storage::F32(vec![1., 2., 3., 4., 5., 6.])
        );
        assert_eq!(
            runtime.snapshot(&next).unwrap().tensor().storage(),
            &Storage::F32(vec![-1., -4., -2., -5., -3., -6.])
        );
    }

    #[test]
    fn conflicting_state_consumers_fail_before_pure_execution_or_commit() {
        let (graph, effects, mut mixed, mut runtime, persistent, next) =
            normal_and_transposed_state_fixture();
        mixed.state_bindings[1].state.buffer = 18;
        let error = realize_mixed_effects(
            &mut runtime,
            &graph,
            &effects,
            &mixed,
            &HashMap::new(),
            None,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            super::super::RealizationError::Schedule(message)
                if message == "persistent state input has conflicting consumer bindings"
        ));
        assert_eq!(
            runtime.snapshot(&persistent).unwrap().tensor().storage(),
            &Storage::F32(vec![1., 2., 3., 4., 5., 6.])
        );
        assert!(runtime.snapshot(&next).is_err());

        let (graph, effects, mut mixed, mut runtime, persistent, next) =
            normal_and_transposed_state_fixture();
        mixed.state_bindings[1].view = Some(AffineView::identity(Shape::from([2, 3])));
        let error = realize_mixed_effects(
            &mut runtime,
            &graph,
            &effects,
            &mixed,
            &HashMap::new(),
            None,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            super::super::RealizationError::Schedule(message)
                if message == "persistent state input has conflicting consumer bindings"
        ));
        assert_eq!(
            runtime.snapshot(&persistent).unwrap().tensor().storage(),
            &Storage::F32(vec![1., 2., 3., 4., 5., 6.])
        );
        assert!(runtime.snapshot(&next).is_err());
    }

    #[test]
    fn external_input_cannot_shadow_persistent_state_binding() {
        let (graph, effects, mixed, mut runtime, persistent, next) =
            normal_and_transposed_state_fixture();
        let error = realize_mixed_effects(
            &mut runtime,
            &graph,
            &effects,
            &mixed,
            &HashMap::from([(
                "state".into(),
                TensorData::from_storage([2, 3], Storage::F32(vec![9.; 6])).unwrap(),
            )]),
            None,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            super::super::RealizationError::Schedule(message)
                if message == "external input shadows persistent state binding"
        ));
        assert_eq!(
            runtime.snapshot(&persistent).unwrap().tensor().storage(),
            &Storage::F32(vec![1., 2., 3., 4., 5., 6.])
        );
        assert!(runtime.snapshot(&next).is_err());
    }
}
