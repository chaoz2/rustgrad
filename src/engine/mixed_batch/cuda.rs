//! Retained CUDA/PTX pure-prefix preparation for mixed batches.
use super::{CapturedMixedBatch, backend};
use crate::{
    CudaGraphPrefixPlan, EffectBatchStep, EffectRuntime, PreparedCudaGraphPrefix, PrimaryContext,
    PtxRenderer, ReplayError, ScheduleItem, TensorData,
};
use std::collections::BTreeMap;

fn prepared_identity(
    batch_identity: u64,
    owner: usize,
    renderer: PtxRenderer,
    keys: &[String],
) -> u64 {
    // This is a logical prepared-prefix identity, never a native handle or
    // byte snapshot. The cache itself separately keys the same owner scope.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in batch_identity
        .to_le_bytes()
        .into_iter()
        .chain((owner as u64).to_le_bytes())
        .chain(u64::from(renderer.sm).to_le_bytes())
        .chain(u64::from(renderer.block_size).to_le_bytes())
        .chain(keys.iter().flat_map(|key| key.bytes().chain([0])))
    {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaMixedBatchTrace {
    pub identity: u64,
    pub owner: usize,
    pub prepared_cache_keys: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct CudaMixedBatchResult {
    pub committed: Vec<crate::BufferState>,
    pub trace: CudaMixedBatchTrace,
}

struct CudaBackend {
    primary: PrimaryContext,
    renderer: PtxRenderer,
    cache: crate::ConcurrentPtxCache,
}
impl backend::PreparedBackend for CudaBackend {
    type Plan = CudaGraphPrefixPlan;
    type Prepared = PreparedCudaGraphPrefix;
    fn plan(
        &self,
        items: &[ScheduleItem],
        retained_outputs: &[u64],
    ) -> Result<Self::Plan, ReplayError> {
        CudaGraphPrefixPlan::plan_for_outputs(items, retained_outputs, self.renderer)
            .map_err(|e| ReplayError::Execute(format!("CUDA prepare: {e}")))
    }
    fn prepare(&self, plan: Self::Plan) -> Result<Self::Prepared, ReplayError> {
        PreparedCudaGraphPrefix::from_plan(self.primary.clone(), plan, &self.cache)
            .map_err(|e| ReplayError::Execute(format!("CUDA prepare: {e}")))
    }
    fn execute(
        &self,
        prepared: &mut Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), ReplayError> {
        prepared
            .execute(values)
            .map_err(|e| ReplayError::Execute(format!("CUDA execute: {e}")))
    }
    fn keys(&self, prepared: &Self::Prepared) -> Vec<String> {
        prepared.kernel_cache_keys()
    }
}
impl CapturedMixedBatch {
    pub fn replay_ptx_with_rebindings(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        rebindings: &[crate::MixedStateRebinding],
        primary: PrimaryContext,
        renderer: PtxRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<CudaMixedBatchResult, ReplayError> {
        let mut result = self.rebound(rebindings)?.replay_ptx(
            runtime,
            inputs,
            primary,
            renderer,
            injected_failure,
        )?;
        result.trace.identity = (result.trace.identity
            ^ super::rebinding_schema_identity(rebindings))
        .wrapping_mul(0x100000001b3);
        Ok(result)
    }

    pub fn replay_ptx(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        primary: PrimaryContext,
        renderer: PtxRenderer,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<CudaMixedBatchResult, ReplayError> {
        let owner = primary.identity();
        let (committed, prepared_cache_keys) = backend::replay(
            self,
            runtime,
            inputs,
            &CudaBackend {
                primary,
                renderer,
                cache: crate::ConcurrentPtxCache::new(),
            },
            injected_failure,
        )?;
        Ok(CudaMixedBatchResult {
            committed,
            trace: CudaMixedBatchTrace {
                identity: prepared_identity(self.identity(), owner, renderer, &prepared_cache_keys),
                owner,
                prepared_cache_keys,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Driver, Storage};
    use std::sync::Arc;

    #[test]
    fn ptx_mixed_batch_prepares_prefixes_then_commits_once() {
        let (first, _) = crate::engine::mixed_batch::test_support::pure_add_capture(701);
        let (second, next) = crate::engine::mixed_batch::test_support::pure_add_capture(701);
        let batch = CapturedMixedBatch::new(vec![first, second]).unwrap();
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let inputs = vec![
            crate::engine::mixed_batch::test_support::add_inputs(),
            BTreeMap::from([
                (
                    "x".into(),
                    crate::engine::mixed_batch::test_support::data(vec![5., 6.]),
                ),
                (
                    "y".into(),
                    crate::engine::mixed_batch::test_support::data(vec![7., 8.]),
                ),
            ]),
        ];
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                701,
                crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
            )
            .unwrap();
        runtime
            .register(
                2,
                crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
            )
            .unwrap();
        let result = batch
            .replay_ptx(
                &mut runtime,
                &inputs,
                primary,
                PtxRenderer::new(80).unwrap(),
                None,
            )
            .unwrap();
        assert_eq!(
            result.committed.last(),
            Some(&crate::BufferState {
                version: 2,
                ..next.clone()
            })
        );
        let end = crate::BufferState { version: 2, ..next };
        assert_eq!(
            runtime.snapshot(&end).unwrap().tensor().storage(),
            &Storage::F32(vec![12., 14.])
        );
        assert!(!result.trace.prepared_cache_keys.is_empty());
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| **call == "module_load")
                .count(),
            1,
            "equivalent prefixes share the owner-scoped prepared cache"
        );
    }

    #[test]
    fn ptx_mixed_batch_rejects_later_prefix_before_launch_or_commit() {
        let (first, first_end) = crate::engine::mixed_batch::test_support::pure_add_capture(91);
        let (mut later, _) = crate::engine::mixed_batch::test_support::pure_add_capture(92);
        crate::engine::mixed_batch::test_support::mark_first_prefix_unsupported(&mut later, "test");
        let batch = CapturedMixedBatch::new(vec![first, later]).unwrap();
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut runtime = EffectRuntime::new();
        for (id, values) in [(91, vec![9., 9.]), (92, vec![8., 8.]), (2, vec![0., 0.])] {
            runtime
                .register(id, crate::engine::mixed_batch::test_support::data(values))
                .unwrap();
        }
        assert!(
            batch
                .replay_ptx(
                    &mut runtime,
                    &[
                        crate::engine::mixed_batch::test_support::add_inputs(),
                        crate::engine::mixed_batch::test_support::add_inputs(),
                    ],
                    primary,
                    PtxRenderer::new(80).unwrap(),
                    None,
                )
                .is_err()
        );
        assert!(mock.calls().iter().all(|call| *call != "launch"));
        assert_eq!(
            runtime
                .snapshot(&crate::BufferState {
                    version: 0,
                    ..first_end
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![9., 9.])
        );
    }

    #[test]
    fn ptx_mixed_batch_zero_extent_skips_launch_and_commits() {
        let (capture, end) = crate::engine::mixed_batch::test_support::zero_extent_add_capture();
        let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(93, crate::engine::mixed_batch::test_support::data(vec![]))
            .unwrap();
        runtime
            .register(2, crate::engine::mixed_batch::test_support::data(vec![]))
            .unwrap();
        batch
            .replay_ptx(
                &mut runtime,
                &[BTreeMap::from([
                    (
                        "x".into(),
                        crate::engine::mixed_batch::test_support::data(vec![]),
                    ),
                    (
                        "y".into(),
                        crate::engine::mixed_batch::test_support::data(vec![]),
                    ),
                ])],
                primary,
                PtxRenderer::new(80).unwrap(),
                None,
            )
            .unwrap();
        assert!(mock.calls().iter().all(|call| *call != "launch"));
        assert_eq!(
            runtime.snapshot(&end).unwrap().tensor().storage(),
            &Storage::F32(vec![])
        );
    }

    #[test]
    fn ptx_mixed_batch_signed_state_input_matches_interpreter() {
        let (capture, end) = crate::engine::mixed_batch::test_support::signed_state_add_capture();
        let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
        let supplied = BTreeMap::from([(
            "bias".into(),
            crate::engine::mixed_batch::test_support::data(vec![10., 20., 30., 40.]),
        )]);
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut ptx = EffectRuntime::new();
        let mut interpreter = EffectRuntime::new();
        for runtime in [&mut ptx, &mut interpreter] {
            runtime
                .register(
                    90,
                    crate::engine::mixed_batch::test_support::data(vec![1., 2., 3., 4.]),
                )
                .unwrap();
            runtime
                .register(
                    2,
                    crate::engine::mixed_batch::test_support::data(vec![0.; 4]),
                )
                .unwrap();
        }
        batch
            .replay_ptx(
                &mut ptx,
                std::slice::from_ref(&supplied),
                primary,
                PtxRenderer::new(80).unwrap(),
                None,
            )
            .unwrap();
        batch
            .replay(&mut interpreter, std::slice::from_ref(&supplied), None)
            .unwrap();
        let expected = &Storage::F32(vec![14., 23., 32., 41.]);
        assert_eq!(ptx.snapshot(&end).unwrap().tensor().storage(), expected);
        assert_eq!(
            ptx.snapshot(&end).unwrap().tensor().storage(),
            interpreter.snapshot(&end).unwrap().tensor().storage()
        );
        assert!(mock.calls().contains(&"launch"));
    }

    #[test]
    fn ptx_rebinding_validates_before_mock_launch_and_matches_interpreter() {
        let (capture, end) = crate::engine::mixed_batch::test_support::signed_state_add_capture();
        let batch = CapturedMixedBatch::new(vec![capture.clone()]).unwrap();
        let rebinding = crate::MixedStateRebinding::new(
            capture
                .states
                .iter()
                .map(|state| (state.buffer, state.buffer + 1_000))
                .collect(),
        )
        .unwrap();
        let supplied = BTreeMap::from([(
            "bias".into(),
            crate::engine::mixed_batch::test_support::data(vec![10., 20., 30., 40.]),
        )]);
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut ptx = EffectRuntime::new();
        let mut interpreter = EffectRuntime::new();
        for runtime in [&mut ptx, &mut interpreter] {
            for state in capture.initial_states() {
                runtime
                    .register(
                        state.buffer + 1_000,
                        crate::engine::mixed_batch::test_support::data(vec![1., 2., 3., 4.]),
                    )
                    .unwrap();
            }
        }
        batch
            .replay_ptx_with_rebindings(
                &mut ptx,
                std::slice::from_ref(&supplied),
                std::slice::from_ref(&rebinding),
                primary,
                PtxRenderer::new(80).unwrap(),
                None,
            )
            .unwrap();
        batch
            .replay_with_rebindings(
                &mut interpreter,
                std::slice::from_ref(&supplied),
                std::slice::from_ref(&rebinding),
                None,
            )
            .unwrap();
        let rebound_end = crate::BufferState {
            buffer: end.buffer + 1_000,
            ..end
        };
        assert_eq!(
            ptx.snapshot(&rebound_end).unwrap().tensor().storage(),
            interpreter
                .snapshot(&rebound_end)
                .unwrap()
                .tensor()
                .storage()
        );
        assert!(mock.calls().contains(&"launch"));
    }

    #[test]
    fn ptx_rebinding_failures_reject_before_prepare_or_launch() {
        let (capture, _) = crate::engine::mixed_batch::test_support::pure_add_capture(681);
        let batch = CapturedMixedBatch::new(vec![capture.clone()]).unwrap();
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut runtime = EffectRuntime::new();
        assert!(
            batch
                .replay_ptx_with_rebindings(
                    &mut runtime,
                    &[crate::engine::mixed_batch::test_support::add_inputs()],
                    &[crate::MixedStateRebinding::new(BTreeMap::new()).unwrap()],
                    primary.clone(),
                    PtxRenderer::new(80).unwrap(),
                    None,
                )
                .is_err()
        );
        let rebinding = crate::MixedStateRebinding::new(
            capture
                .states
                .iter()
                .map(|state| (state.buffer, state.buffer + 1_000))
                .collect(),
        )
        .unwrap();
        for state in capture.initial_states() {
            runtime
                .register(
                    state.buffer + 1_000,
                    TensorData::from_storage([1], Storage::F32(vec![0.])).unwrap(),
                )
                .unwrap();
        }
        assert!(
            batch
                .replay_ptx_with_rebindings(
                    &mut runtime,
                    &[crate::engine::mixed_batch::test_support::add_inputs()],
                    std::slice::from_ref(&rebinding),
                    primary,
                    PtxRenderer::new(80).unwrap(),
                    None,
                )
                .is_err()
        );
        assert!(
            mock.calls()
                .iter()
                .all(|call| *call != "module_load" && *call != "launch")
        );

        // A capture always declares its exact predecessor versions.  Once a
        // successful replay advances its target, replaying that same capture
        // must fail during rebinding/runtime preflight, before a retained PTX
        // prefix can be prepared or launched again.
        let stale_mock = Arc::new(crate::cuda::tests::Mock::default());
        let stale_primary = Driver::from_dispatch(stale_mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut stale_runtime = EffectRuntime::new();
        for state in capture.initial_states() {
            stale_runtime
                .register(
                    state.buffer + 1_000,
                    crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
                )
                .unwrap();
        }
        batch
            .replay_ptx_with_rebindings(
                &mut stale_runtime,
                &[crate::engine::mixed_batch::test_support::add_inputs()],
                &[rebinding],
                stale_primary.clone(),
                PtxRenderer::new(80).unwrap(),
                None,
            )
            .unwrap();
        let prepared_or_launched = |calls: &[&'static str]| {
            calls
                .iter()
                .filter(|call| **call == "module_load" || **call == "launch")
                .count()
        };
        let work_before_stale_replay = prepared_or_launched(&stale_mock.calls());
        assert!(
            batch
                .replay_ptx_with_rebindings(
                    &mut stale_runtime,
                    &[crate::engine::mixed_batch::test_support::add_inputs()],
                    &[crate::MixedStateRebinding::new(
                        capture
                            .states
                            .iter()
                            .map(|state| (state.buffer, state.buffer + 1_000))
                            .collect(),
                    )
                    .unwrap()],
                    stale_primary,
                    PtxRenderer::new(80).unwrap(),
                    None,
                )
                .is_err()
        );
        assert_eq!(
            prepared_or_launched(&stale_mock.calls()),
            work_before_stale_replay
        );
    }

    #[test]
    fn ptx_mixed_batch_launch_or_effect_failure_preserves_state_for_retry() {
        let (capture, next) = crate::engine::mixed_batch::test_support::pure_add_capture(95);
        let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                95,
                crate::engine::mixed_batch::test_support::data(vec![9., 9.]),
            )
            .unwrap();
        runtime
            .register(
                2,
                crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
            )
            .unwrap();
        mock.fail_launch_after(0, 1);
        assert!(
            batch
                .replay_ptx(
                    &mut runtime,
                    &[crate::engine::mixed_batch::test_support::add_inputs()],
                    primary.clone(),
                    PtxRenderer::new(80).unwrap(),
                    None,
                )
                .is_err()
        );
        assert_eq!(
            runtime
                .snapshot(&crate::BufferState {
                    version: 0,
                    ..next.clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![9., 9.])
        );
        assert!(
            batch
                .replay_ptx(
                    &mut runtime,
                    &[crate::engine::mixed_batch::test_support::add_inputs()],
                    primary.clone(),
                    PtxRenderer::new(80).unwrap(),
                    Some(EffectBatchStep { entry: 0, step: 0 }),
                )
                .is_err()
        );
        assert_eq!(
            runtime
                .snapshot(&crate::BufferState {
                    version: 0,
                    ..next.clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![9., 9.])
        );
        batch
            .replay_ptx(
                &mut runtime,
                &[crate::engine::mixed_batch::test_support::add_inputs()],
                primary,
                PtxRenderer::new(80).unwrap(),
                None,
            )
            .unwrap();
        assert_eq!(
            runtime.snapshot(&next).unwrap().tensor().storage(),
            &Storage::F32(vec![4., 6.])
        );
    }
}
