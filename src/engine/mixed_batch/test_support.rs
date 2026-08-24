//! Backend-neutral captured-mixed-batch fixtures.
use crate::{
    BinaryOp, CapturedMixedSchedule, CapturedSchedule, DType, EffectGraph, Graph,
    ScheduleValueBinding, Storage, TensorData, combine_mixed_schedules, schedule, schedule_effects,
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
