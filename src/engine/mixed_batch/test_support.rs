//! Backend-neutral captured-mixed-batch fixtures.
use crate::{
    AffineView, BinaryOp, CapturedMixedSchedule, CapturedSchedule, DType, EffectGraph, Graph,
    ScheduleStateBinding, ScheduleValueBinding, Shape, Storage, TensorData, bind_schedule_states,
    combine_mixed_schedules, schedule, schedule_effects,
};
use std::collections::BTreeMap;

pub(crate) fn data(values: Vec<f32>) -> TensorData {
    TensorData::from_storage([values.len()], Storage::F32(values)).unwrap()
}

pub(crate) fn pure_add_capture(target_id: u64) -> (CapturedMixedSchedule, crate::BufferState) {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2], DType::F32);
    let y = graph.input_dtype("y", [2], DType::F32);
    let sum = graph.binary(BinaryOp::Add, x, y).unwrap();
    let pure = schedule(&graph, sum).unwrap();
    let mut captured = CapturedSchedule::capture(&graph, &pure, &[sum]).unwrap();
    let mut effects = EffectGraph::default();
    let target = effects.insert(target_id, data(vec![0., 0.])).unwrap();
    let source = effects
        .insert(sum.index() as u64, data(vec![0., 0.]))
        .unwrap();
    let next = effects.assign(&target, &source).unwrap();
    let mixed = combine_mixed_schedules(
        pure.clone(),
        schedule_effects(&effects).unwrap(),
        vec![ScheduleValueBinding {
            producer_item: 0,
            producer_node: sum,
            producer_output: pure.items[0].output.clone(),
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        }],
    )
    .unwrap();
    captured.items = mixed.items.clone();
    (
        CapturedMixedSchedule::from_parts(
            captured,
            &mixed,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap(),
        next.state().clone(),
    )
}

pub(crate) fn add_inputs() -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        ("x".into(), data(vec![1., 2.])),
        ("y".into(), data(vec![3., 4.])),
    ])
}

/// A capture whose pure input reads a flipped persistent state. The fixture is
/// backend-neutral; individual runtime suites provide their own dispatch and
/// assertion surface.
pub(crate) fn signed_state_add_capture() -> (CapturedMixedSchedule, crate::BufferState) {
    let mut graph = Graph::new();
    let state = graph.input_dtype("state", [4], DType::F32);
    let bias = graph.input_dtype("bias", [4], DType::F32);
    let sum = graph.binary(BinaryOp::Add, state, bias).unwrap();
    let pure = schedule(&graph, sum).unwrap();
    let input = pure.items[0]
        .input_bindings
        .iter()
        .find(|binding| binding.input_node == state)
        .unwrap()
        .clone();
    let mut captured = CapturedSchedule::capture(&graph, &pure, &[sum]).unwrap();
    let mut effects = EffectGraph::default();
    let target = effects.insert(90, data(vec![0.; 4])).unwrap();
    let source = effects
        .insert(sum.index() as u64, data(vec![0.; 4]))
        .unwrap();
    let next = effects.assign(&target, &source).unwrap();
    let pure = bind_schedule_states(
        pure,
        vec![ScheduleStateBinding {
            state: target.state().clone(),
            view: Some(AffineView::identity(Shape::from([4])).flip(0).unwrap()),
            consumer_item: 0,
            consumer_node: sum,
            input_node: state,
            desc: input.desc,
            abi_index: input.abi_index,
        }],
    )
    .unwrap();
    let mixed = combine_mixed_schedules(
        pure.clone(),
        schedule_effects(&effects).unwrap(),
        vec![ScheduleValueBinding {
            producer_item: 0,
            producer_node: sum,
            producer_output: pure.items[0].output.clone(),
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        }],
    )
    .unwrap();
    captured.items = mixed.items.clone();
    (
        CapturedMixedSchedule::from_parts(
            captured,
            &mixed,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap(),
        next.state().clone(),
    )
}

/// A backend-neutral empty-domain prefix used to ensure dispatch remains a
/// no-op while the versioned host effect transaction still commits.
pub(crate) fn zero_extent_add_capture() -> (CapturedMixedSchedule, crate::BufferState) {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [0], DType::F32);
    let y = graph.input_dtype("y", [0], DType::F32);
    let sum = graph.binary(BinaryOp::Add, x, y).unwrap();
    let pure = schedule(&graph, sum).unwrap();
    let mut captured = CapturedSchedule::capture(&graph, &pure, &[sum]).unwrap();
    let mut effects = EffectGraph::default();
    let target = effects.insert(93, data(vec![])).unwrap();
    let source = effects.insert(sum.index() as u64, data(vec![])).unwrap();
    let next = effects.assign(&target, &source).unwrap();
    let mixed = combine_mixed_schedules(
        pure.clone(),
        schedule_effects(&effects).unwrap(),
        vec![ScheduleValueBinding {
            producer_item: 0,
            producer_node: sum,
            producer_output: pure.items[0].output.clone(),
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        }],
    )
    .unwrap();
    captured.items = mixed.items.clone();
    (
        CapturedMixedSchedule::from_parts(
            captured,
            &mixed,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap(),
        next.state().clone(),
    )
}
