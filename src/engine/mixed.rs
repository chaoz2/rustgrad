//! Transactional realization of the explicit pure-value to effect-state edge.
use crate::{EffectGraph, EffectRuntime, Graph, Op, Schedule, TensorData};
use std::collections::{BTreeMap, HashMap};

/// Realizes pure producers into owned temporary values, then commits all
/// affected persistent states through one `EffectRuntime` transaction.
///
/// Only interpreter execution is accepted here. Native/device execution and
/// state-to-pure reads stay explicit unsupported boundaries until their full
/// lifetime and ABI contracts exist.
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
        value_bindings: vec![],
        state_bindings: vec![],
    };
    let mut pure_inputs = inputs.clone();
    for binding in &schedule.state_bindings {
        let Op::Input { name } = graph
            .op(binding.input_node)
            .map_err(|error| super::RealizationError::Schedule(error.to_string()))?
        else {
            return Err(super::RealizationError::Schedule(
                "state binding input is not graph input".into(),
            ));
        };
        let snapshot = runtime.snapshot(&binding.state).map_err(|error| {
            super::RealizationError::Execution(format!("persistent state read: {error:?}"))
        })?;
        let injected = match &binding.view {
            Some(view) => snapshot.tensor().affine_read(view),
            None => Ok(snapshot.tensor().clone()),
        }
        .map_err(|error| {
            super::RealizationError::Execution(format!("persistent state affine read: {error:?}"))
        })?;
        let bytes = injected
            .len()
            .checked_mul(injected.dtype().itemsize())
            .ok_or_else(|| {
                super::RealizationError::Execution("persistent state bytes overflow".into())
            })?;
        if injected.shape() != &binding.desc.shape
            || injected.dtype() != binding.desc.dtype
            || bytes != binding.desc.bytes
        {
            return Err(super::RealizationError::Schedule(
                "persistent state injection descriptor mismatch".into(),
            ));
        }
        pure_inputs.insert(name.clone(), injected);
    }
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
        let step = match item.kernel.arg() {
            crate::UArg::Effect(payload) => payload.step,
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
        AffineView, BinaryOp, DType, EffectGraph, ScheduleValueBinding, Shape, Storage,
        combine_mixed_schedules, schedule, schedule_effects,
    };

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
            producer_output: pure.items[0].output.clone(),
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
            producer_output: pure.items[0].output.clone(),
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
            producer_output: pure.items[0].output.clone(),
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
}
