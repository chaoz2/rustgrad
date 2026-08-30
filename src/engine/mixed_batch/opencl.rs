//! Strict hybrid OpenCL replay: device pure prefixes, host atomic effects.
use super::{CapturedMixedBatch, backend};
use crate::runtime::opencl::{OpenClContext, OpenClRenderer, PreparedOpenClPrefix};
use crate::{EffectBatchStep, EffectRuntime, ReplayError, ScheduleItem, TensorData};
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

struct OpenClBackend {
    context: OpenClContext,
    renderer: OpenClRenderer,
}
impl backend::PreparedBackend for OpenClBackend {
    type Prepared = PreparedOpenClPrefix;
    fn prepare(&self, items: &[ScheduleItem]) -> Result<Self::Prepared, ReplayError> {
        PreparedOpenClPrefix::prepare(self.context.clone(), items, self.renderer)
            .map_err(|e| ReplayError::Execute(format!("OpenCL prepare: {e:?}")))
    }
    fn execute(
        &self,
        prepared: &Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), ReplayError> {
        prepared
            .execute(values)
            .map_err(|e| ReplayError::Execute(format!("OpenCL execute: {e:?}")))
    }
    fn keys(&self, prepared: &Self::Prepared) -> Vec<String> {
        prepared.kernel_cache_keys()
    }
}

impl CapturedMixedBatch {
    pub fn replay_opencl_with_rebindings(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        rebindings: &[crate::MixedStateRebinding],
        context: OpenClContext,
        renderer: OpenClRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<OpenClMixedBatchResult, ReplayError> {
        let mut result = self.rebound(rebindings)?.replay_opencl(
            runtime,
            inputs,
            context,
            renderer,
            injected_failure,
        )?;
        result.trace.identity = (result.trace.identity
            ^ super::rebinding_schema_identity(rebindings))
        .wrapping_mul(0x100000001b3);
        Ok(result)
    }

    pub fn replay_opencl(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        context: OpenClContext,
        renderer: OpenClRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<OpenClMixedBatchResult, ReplayError> {
        let (committed, prepared_cache_keys) = backend::replay(
            self,
            runtime,
            inputs,
            &OpenClBackend { context, renderer },
            injected_failure,
        )?;
        Ok(OpenClMixedBatchResult {
            committed,
            trace: OpenClMixedBatchTrace {
                identity: self.identity(),
                prepared_cache_keys,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapturedReplayExecutor, Storage};
    use std::sync::Arc;

    fn data(values: Vec<f32>) -> TensorData {
        TensorData::from_storage([values.len()], Storage::F32(values)).unwrap()
    }

    fn pure_assign(target_id: u64) -> (crate::CapturedMixedSchedule, crate::BufferState) {
        crate::engine::mixed_batch::test_support::pure_add_capture(target_id)
    }

    fn inputs() -> BTreeMap<String, TensorData> {
        crate::engine::mixed_batch::test_support::add_inputs()
    }

    #[test]
    fn replay_opencl_zero_extent_does_not_submit_and_commits_empty_state() {
        let (capture, end) = crate::engine::mixed_batch::test_support::zero_extent_add_capture();
        let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
        let mock = Arc::new(crate::runtime::opencl::tests::MockDispatch::default());
        let (context, _) = crate::runtime::opencl::tests::setup(mock.clone());
        let mut runtime = EffectRuntime::new();
        runtime.register(93, data(vec![])).unwrap();
        runtime.register(2, data(vec![])).unwrap();
        batch
            .replay_opencl(
                &mut runtime,
                &[BTreeMap::from([
                    ("x".into(), data(vec![])),
                    ("y".into(), data(vec![])),
                ])],
                context,
                OpenClRenderer::default(),
                None,
            )
            .unwrap();
        assert!(mock.calls().iter().all(|call| call != "kernel_launch"));
        assert_eq!(
            runtime.snapshot(&end).unwrap().tensor().storage(),
            &Storage::F32(vec![])
        );
    }

    #[test]
    fn replay_opencl_signed_state_input_matches_interpreter_and_native() {
        let (capture, end) = crate::engine::mixed_batch::test_support::signed_state_add_capture();
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
                std::slice::from_ref(&supplied),
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
            .replay(&mut interpreter, std::slice::from_ref(&supplied), None)
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
        crate::engine::mixed_batch::test_support::mark_first_prefix_unsupported(&mut later, "test");
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
            renderer,
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
                renderer,
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
                    renderer,
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
