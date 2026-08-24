//! Strict hybrid OpenCL replay: device pure prefixes, host atomic effects.
use super::CapturedMixedBatch;
use crate::runtime::opencl::{OpenClContext, OpenClRenderer, PreparedOpenClPrefix};
use crate::{
    EffectBatch, EffectBatchEntry, EffectBatchStep, EffectRuntime, ReplayError, TensorData,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenClMixedBatchTrace {
    pub identity: u64,
    pub prepared_cache_keys: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct OpenClMixedBatchResult {
    pub committed: Vec<crate::BufferState>,
    pub trace: OpenClMixedBatchTrace,
}

impl CapturedMixedBatch {
    pub fn replay_opencl(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        context: OpenClContext,
        renderer: OpenClRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<OpenClMixedBatchResult, ReplayError> {
        if inputs.len() != self.captures.len() {
            return Err(ReplayError::Descriptor("mixed batch input count".into()));
        }
        let mut latest = BTreeMap::new();
        let mut candidates = BTreeMap::new();
        let mut bound = Vec::new();
        for (capture, provided) in self.captures.iter().zip(inputs) {
            let mut starts = BTreeMap::new();
            for local in capture.initial_states() {
                let state = latest
                    .get(&local.buffer)
                    .cloned()
                    .unwrap_or_else(|| local.clone());
                if !candidates.contains_key(&state) && !latest.contains_key(&local.buffer) {
                    candidates.insert(
                        state.clone(),
                        runtime
                            .snapshot(&state)
                            .map_err(|e| ReplayError::Execute(format!("batch preflight: {e:?}")))?
                            .tensor()
                            .clone(),
                    );
                }
                starts.insert(local.buffer, state);
            }
            for state in &capture.states {
                let start = starts
                    .get(&state.buffer)
                    .ok_or_else(|| ReplayError::Corrupt("batch target start".into()))?;
                latest.insert(
                    state.buffer,
                    crate::BufferState {
                        version: start
                            .version
                            .checked_add(state.version)
                            .ok_or_else(|| ReplayError::Corrupt("batch version overflow".into()))?,
                        ..state.clone()
                    },
                );
            }
            let bound_capture = crate::engine::mixed_capture::BoundMixedCapture::bind(
                capture,
                &candidates,
                starts,
                provided,
            )?;
            bound.push(bound_capture);
        }
        let mut prepared = Vec::new();
        for capture in &bound {
            let split = capture
                .capture()
                .schedule
                .items
                .iter()
                .position(crate::ScheduleItem::is_effect)
                .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
            prepared.push(
                PreparedOpenClPrefix::prepare(
                    context.clone(),
                    &capture.capture().schedule.items[..split],
                    renderer.clone(),
                )
                .map_err(|e| ReplayError::Execute(format!("OpenCL prepare: {e:?}")))?,
            );
        }
        let mut entries: Vec<EffectBatchEntry> = Vec::new();
        let mut keys = Vec::new();
        for (bound_capture, prefix) in bound.iter().zip(&prepared) {
            let mut values = bound_capture.capture().schedule.constants.clone();
            for input in &bound_capture.capture().schedule.inputs {
                values.insert(
                    input.desc.id,
                    bound_capture
                        .inputs()
                        .get(&input.name)
                        .cloned()
                        .ok_or_else(|| ReplayError::Missing(input.name.clone()))?,
                );
            }
            prefix
                .execute(&mut values)
                .map_err(|e| ReplayError::Execute(format!("OpenCL execute: {e:?}")))?;
            keys.extend(prefix.kernel_cache_keys());
            entries.push(bound_capture.capture().stage_values(
                &mut candidates,
                bound_capture.starts().clone(),
                values,
            )?);
        }
        let batch = EffectBatch::new(entries)
            .map_err(|e| ReplayError::Execute(format!("batch validate: {e:?}")))?;
        let committed = runtime
            .execute_batch(&batch, injected_failure)
            .map_err(|e| ReplayError::Execute(format!("batch commit: {e:?}")))?;
        Ok(OpenClMixedBatchResult {
            committed,
            trace: OpenClMixedBatchTrace {
                identity: self.identity(),
                prepared_cache_keys: keys,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AffineView, BinaryOp, CapturedReplayExecutor, CapturedSchedule, DType, EffectGraph, Graph,
        ScheduleStateBinding, ScheduleValueBinding, Shape, Storage, bind_schedule_states,
        combine_mixed_schedules, schedule, schedule_effects,
    };
    use std::sync::Arc;

    fn data(values: Vec<f32>) -> TensorData {
        TensorData::from_storage([values.len()], Storage::F32(values)).unwrap()
    }

    fn pure_assign(target_id: u64) -> (crate::CapturedMixedSchedule, crate::BufferState) {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F32);
        let y = graph.input_dtype("y", [2], DType::F32);
        let sum = graph.binary(BinaryOp::Add, x, y).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let mut captured = CapturedSchedule::capture(&graph, &pure, &[sum]).unwrap();
        let mut effects = EffectGraph::default();
        let target = effects.insert(target_id, data(vec![0.0, 0.0])).unwrap();
        let source = effects
            .insert(sum.index() as u64, data(vec![0.0, 0.0]))
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
            crate::CapturedMixedSchedule::from_parts(
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

    fn inputs() -> BTreeMap<String, TensorData> {
        BTreeMap::from([
            ("x".into(), data(vec![1.0, 2.0])),
            ("y".into(), data(vec![3.0, 4.0])),
        ])
    }

    fn signed_state_capture() -> (crate::CapturedMixedSchedule, crate::BufferState) {
        let mut graph = Graph::new();
        let state = graph.input_dtype("state", [4], DType::F32);
        let bias = graph.input_dtype("bias", [4], DType::F32);
        let sum = graph.binary(BinaryOp::Add, state, bias).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let input = pure.items[0]
            .input_bindings
            .iter()
            .find(|x| x.input_node == state)
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
            crate::CapturedMixedSchedule::from_parts(
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

    #[test]
    fn replay_opencl_signed_state_input_matches_interpreter_and_native() {
        let (capture, end) = signed_state_capture();
        let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
        let mock = Arc::new(crate::runtime::opencl::tests::MockDispatch::default());
        let (context, _) = crate::runtime::opencl::tests::setup(mock);
        let supplied = BTreeMap::from([("bias".into(), data(vec![10., 20., 30., 40.]))]);
        let mut cl = EffectRuntime::new();
        cl.register(90, data(vec![1., 2., 3., 4.])).unwrap();
        cl.register(2, data(vec![0.; 4])).unwrap();
        batch
            .replay_opencl(
                &mut cl,
                &[supplied.clone()],
                context,
                OpenClRenderer::default(),
                None,
            )
            .unwrap();
        let mut interpreter = EffectRuntime::new();
        interpreter
            .register(90, data(vec![1., 2., 3., 4.]))
            .unwrap();
        interpreter.register(2, data(vec![0.; 4])).unwrap();
        batch
            .replay(&mut interpreter, &[supplied.clone()], None)
            .unwrap();
        let mut native = EffectRuntime::new();
        native.register(90, data(vec![1., 2., 3., 4.])).unwrap();
        native.register(2, data(vec![0.; 4])).unwrap();
        batch
            .replay_native(
                &mut native,
                &[supplied],
                &CapturedReplayExecutor::default(),
                false,
                None,
            )
            .unwrap();
        assert_eq!(
            cl.snapshot(&end).unwrap().tensor().storage(),
            &Storage::F32(vec![14., 23., 32., 41.])
        );
        assert_eq!(
            cl.snapshot(&end).unwrap().tensor().storage(),
            interpreter.snapshot(&end).unwrap().tensor().storage()
        );
        assert_eq!(
            cl.snapshot(&end).unwrap().tensor().storage(),
            native.snapshot(&end).unwrap().tensor().storage()
        );
    }

    #[test]
    fn later_unsupported_capture_prevents_earlier_submission() {
        let (first, end) = pure_assign(91);
        let (mut later, _) = pure_assign(92);
        later.schedule.items[0].boundary = Some(crate::ScheduleBoundary::Unsupported("test"));
        let batch = CapturedMixedBatch::new(vec![first, later]).unwrap();
        let mock = Arc::new(crate::runtime::opencl::tests::MockDispatch::default());
        let (context, _) = crate::runtime::opencl::tests::setup(mock.clone());
        let mut runtime = EffectRuntime::new();
        runtime.register(91, data(vec![9., 9.])).unwrap();
        runtime.register(92, data(vec![8., 8.])).unwrap();
        runtime.register(2, data(vec![0., 0.])).unwrap();
        assert!(
            batch
                .replay_opencl(
                    &mut runtime,
                    &[inputs(), inputs()],
                    context,
                    OpenClRenderer::default(),
                    None
                )
                .is_err()
        );
        assert!(mock.calls().iter().all(|call| call != "kernel_launch"));
        assert_eq!(
            runtime
                .snapshot(&crate::BufferState { version: 0, ..end })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![9., 9.])
        );
    }

    #[test]
    fn replay_opencl_is_atomic_retryable_and_matches_interpreter_and_native() {
        let (first, first_end) = pure_assign(80);
        let (second, second_end) = pure_assign(80);
        let batch = CapturedMixedBatch::new(vec![first.clone(), second.clone()]).unwrap();
        let mock = Arc::new(crate::runtime::opencl::tests::MockDispatch::default());
        let (context, _) = crate::runtime::opencl::tests::setup(mock.clone());
        let renderer = OpenClRenderer::default();
        let mut opencl_runtime = EffectRuntime::new();
        opencl_runtime.register(80, data(vec![9.0, 9.0])).unwrap();
        opencl_runtime.register(2, data(vec![0.0, 0.0])).unwrap();

        mock.set_launch_failure(-5);
        let launch_failure = batch.replay_opencl(
            &mut opencl_runtime,
            &[inputs(), inputs()],
            context.clone(),
            renderer.clone(),
            None,
        );
        assert!(launch_failure.is_err(), "{launch_failure:?}");
        assert_eq!(
            opencl_runtime
                .snapshot(&crate::BufferState {
                    version: 0,
                    ..first_end.clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![9.0, 9.0])
        );
        assert!(mock.calls().iter().any(|call| call == "kernel_launch"));

        let first_result = batch
            .replay_opencl(
                &mut opencl_runtime,
                &[inputs(), inputs()],
                context.clone(),
                renderer.clone(),
                None,
            )
            .unwrap();
        assert_eq!(first_result.trace.identity, batch.identity());
        assert_eq!(first_result.trace.prepared_cache_keys.len(), 2);
        assert_eq!(
            opencl_runtime
                .snapshot(&crate::BufferState {
                    version: 2,
                    ..second_end.clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![4.0, 6.0])
        );

        let (single, end) = pure_assign(81);
        let single_batch = CapturedMixedBatch::new(vec![single]).unwrap();
        let mut effect_failure = EffectRuntime::new();
        effect_failure.register(81, data(vec![9.0, 9.0])).unwrap();
        effect_failure.register(2, data(vec![0.0, 0.0])).unwrap();
        assert!(
            single_batch
                .replay_opencl(
                    &mut effect_failure,
                    &[inputs()],
                    context.clone(),
                    renderer.clone(),
                    Some(EffectBatchStep { entry: 0, step: 0 })
                )
                .is_err()
        );
        assert_eq!(
            effect_failure
                .snapshot(&crate::BufferState {
                    version: 0,
                    ..end.clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![9.0, 9.0])
        );
        single_batch
            .replay_opencl(&mut effect_failure, &[inputs()], context, renderer, None)
            .unwrap();

        let mut interpreter = EffectRuntime::new();
        interpreter.register(81, data(vec![9.0, 9.0])).unwrap();
        interpreter.register(2, data(vec![0.0, 0.0])).unwrap();
        single_batch
            .replay(&mut interpreter, &[inputs()], None)
            .unwrap();
        let mut native_runtime = EffectRuntime::new();
        native_runtime.register(81, data(vec![9.0, 9.0])).unwrap();
        native_runtime.register(2, data(vec![0.0, 0.0])).unwrap();
        single_batch
            .replay_native(
                &mut native_runtime,
                &[inputs()],
                &CapturedReplayExecutor::default(),
                false,
                None,
            )
            .unwrap();
        assert_eq!(
            effect_failure.snapshot(&end).unwrap().tensor().storage(),
            interpreter.snapshot(&end).unwrap().tensor().storage()
        );
        assert_eq!(
            interpreter.snapshot(&end).unwrap().tensor().storage(),
            native_runtime.snapshot(&end).unwrap().tensor().storage()
        );
    }
}
