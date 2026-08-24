//! Retained CUDA/PTX pure-prefix preparation for mixed batches.
use super::{CapturedMixedBatch, backend};
use crate::{
    EffectBatchStep, EffectRuntime, PrimaryBufferLease, PrimaryContext, PtxBinding, PtxRenderer,
    ReplayError, ScheduleItem, TensorData,
};
use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc};

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

struct PreparedCudaItem {
    kernel: Arc<crate::PrimaryPtxKernel>,
    rendered: crate::RenderedPtx,
    leases: Vec<Option<PrimaryBufferLease>>,
}
struct PreparedCudaPrefix {
    stream: crate::Stream,
    items: Vec<PreparedCudaItem>,
}
impl PreparedCudaPrefix {
    fn prepare(
        primary: PrimaryContext,
        items: &[ScheduleItem],
        renderer: PtxRenderer,
        cache: &crate::ConcurrentPtxCache,
    ) -> Result<Self, crate::PtxError> {
        let stream = primary.stream()?;
        let allocator = primary.allocator();
        let mut prepared = Vec::with_capacity(items.len());
        for item in items {
            if item.boundary.is_some()
                || item.is_effect()
                || !item.quantized_input_bindings.is_empty()
            {
                return Err(crate::PtxError::Unsupported(
                    "pure prefix item is outside CUDA/PTX execution".into(),
                ));
            }
            let rendered = renderer.render(&item.kernel)?;
            rendered.validate_schedule_bindings(item.ordered_inputs())?;
            let kernel = cache.get_or_load(&primary, rendered.clone(), renderer.block_size)?;
            let leases = rendered
                .buffers
                .iter()
                .map(|abi| {
                    let bytes = abi
                        .elements
                        .checked_mul(abi.dtype.itemsize())
                        .ok_or(crate::PtxError::Overflow)?;
                    if bytes == 0 {
                        Ok(None)
                    } else {
                        allocator
                            .allocate(NonZeroUsize::new(bytes).unwrap())
                            .map(Some)
                            .map_err(crate::PtxError::Cuda)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            prepared.push(PreparedCudaItem {
                kernel,
                rendered,
                leases,
            });
        }
        Ok(Self {
            stream,
            items: prepared,
        })
    }
    fn keys(&self) -> Vec<String> {
        self.items
            .iter()
            .map(|item| item.rendered.cache_key.clone())
            .collect()
    }
    fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), crate::PtxError> {
        for item in &self.items {
            let rendered = &item.rendered;
            let output_count = rendered.buffers.iter().filter(|abi| abi.mutable).count();
            if output_count != 1 {
                return Err(crate::PtxError::InvalidBinding(
                    "CUDA mixed prefix needs one output binding".into(),
                ));
            }
            if rendered.extent == 0 {
                let out = rendered
                    .buffers
                    .iter()
                    .find(|abi| abi.mutable)
                    .ok_or_else(|| crate::PtxError::InvalidBinding("missing output".into()))?;
                values.insert(
                    out.id,
                    TensorData::from_scalars(
                        out.source_shape.clone(),
                        out.dtype,
                        std::iter::empty(),
                    )
                    .map_err(|_| crate::PtxError::InvalidBinding("zero output".into()))?,
                );
                continue;
            }
            // Complete host-side ABI validation precedes every upload and
            // launch. This makes a malformed later input fail without
            // partially staging a prefix or touching its detached output.
            for abi in rendered.buffers.iter().filter(|abi| !abi.mutable) {
                let value = values.get(&abi.id).ok_or_else(|| {
                    crate::PtxError::InvalidBinding(format!("missing prefix input {}", abi.id))
                })?;
                if value.dtype() != abi.dtype || value.shape() != &abi.source_shape {
                    return Err(crate::PtxError::InvalidBinding(format!(
                        "prefix input {} dtype/shape mismatch",
                        abi.id
                    )));
                }
                let bytes = value
                    .to_le_bytes()
                    .map_err(|_| crate::PtxError::InvalidBinding("input bytes".into()))?;
                let expected = abi
                    .elements
                    .checked_mul(abi.dtype.itemsize())
                    .ok_or(crate::PtxError::Overflow)?;
                if bytes.len() != expected {
                    return Err(crate::PtxError::InvalidBinding(format!(
                        "prefix input {} byte extent mismatch",
                        abi.id
                    )));
                }
            }
            for (abi, lease) in rendered.buffers.iter().zip(&item.leases) {
                if !abi.mutable {
                    let value = values.get(&abi.id).ok_or_else(|| {
                        crate::PtxError::InvalidBinding(format!("missing prefix input {}", abi.id))
                    })?;
                    let bytes = value
                        .to_le_bytes()
                        .map_err(|_| crate::PtxError::InvalidBinding("input bytes".into()))?;
                    lease
                        .as_ref()
                        .ok_or_else(|| crate::PtxError::InvalidBinding("zero input lease".into()))?
                        .view()?
                        .copy_from(0, &bytes)?;
                }
            }
            let bindings = rendered
                .buffers
                .iter()
                .zip(&item.leases)
                .map(|(abi, lease)| {
                    Ok(PtxBinding {
                        buffer: lease
                            .as_ref()
                            .ok_or_else(|| crate::PtxError::InvalidBinding("missing lease".into()))?
                            .view()?,
                        dtype: abi.dtype,
                        mutable: abi.mutable,
                    })
                })
                .collect::<Result<Vec<_>, crate::PtxError>>()?;
            item.kernel.launch(&self.stream, &bindings, true)?;
            let out_index = rendered
                .buffers
                .iter()
                .position(|abi| abi.mutable)
                .ok_or_else(|| crate::PtxError::InvalidBinding("missing output".into()))?;
            let out = &rendered.buffers[out_index];
            let mut bytes = vec![
                0;
                out.elements
                    .checked_mul(out.dtype.itemsize())
                    .ok_or(crate::PtxError::Overflow)?
            ];
            let output_lease = item.leases[out_index]
                .as_ref()
                .ok_or_else(|| crate::PtxError::InvalidBinding("missing output lease".into()))?;
            output_lease.view()?.copy_to(0, &mut bytes)?;
            values.insert(
                out.id,
                TensorData::from_le_bytes(out.source_shape.clone(), out.dtype, &bytes)
                    .map_err(|_| crate::PtxError::InvalidBinding("output bytes".into()))?,
            );
        }
        Ok(())
    }
}
struct CudaBackend {
    primary: PrimaryContext,
    renderer: PtxRenderer,
    cache: crate::ConcurrentPtxCache,
}
impl backend::PreparedBackend for CudaBackend {
    type Prepared = PreparedCudaPrefix;
    fn prepare(&self, items: &[ScheduleItem]) -> Result<Self::Prepared, ReplayError> {
        PreparedCudaPrefix::prepare(self.primary.clone(), items, self.renderer, &self.cache)
            .map_err(|e| ReplayError::Execute(format!("CUDA prepare: {e}")))
    }
    fn execute(
        &self,
        prepared: &Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), ReplayError> {
        prepared
            .execute(values)
            .map_err(|e| ReplayError::Execute(format!("CUDA execute: {e}")))
    }
    fn keys(&self, prepared: &Self::Prepared) -> Vec<String> {
        prepared.keys()
    }
}
impl CapturedMixedBatch {
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
        later.schedule.items[0].boundary = Some(crate::ScheduleBoundary::Unsupported("test"));
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
