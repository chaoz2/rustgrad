//! Fixed-schema, single-primary-context CUDA graph replay for pure prefixes.

use crate::cuda::PrimaryGraphExec;
use crate::runtime::static_schedule::{
    Sealed, StaticPlanAdapter, StaticRendered, StaticRenderedBuffer, StaticSchedulePlan,
    bind_rendered_buffers,
};
use crate::{
    ConcurrentPtxCache, PrimaryBufferLease, PrimaryContext, PrimaryPtxKernel, PtxBinding, PtxError,
    PtxRenderer, RenderedPtx, ScheduleItem, Stream, TensorData,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    sync::Arc,
};

struct CudaGraphPlanAdapter {
    renderer: PtxRenderer,
}

impl Sealed for CudaGraphPlanAdapter {}

impl StaticPlanAdapter for CudaGraphPlanAdapter {
    type Error = PtxError;
    type Rendered = RenderedPtx;

    fn render(&self, item: &ScheduleItem) -> Result<StaticRendered<Self::Rendered>, Self::Error> {
        let rendered = self.renderer.render(&item.kernel)?;
        rendered.validate_schedule_bindings(item.ordered_inputs())?;
        let buffers = bind_rendered_buffers(
            item,
            rendered.buffers.iter().map(|abi| StaticRenderedBuffer {
                id: abi.id,
                dtype: abi.dtype,
                source_shape: abi.source_shape.clone(),
                elements: abi.elements,
                mutable: abi.mutable,
            }),
            PtxError::InvalidBinding,
            || PtxError::Overflow,
        )?;
        Ok(StaticRendered {
            cache_key: rendered.cache_key.clone(),
            extent: rendered.extent,
            buffers,
            artifact: rendered,
        })
    }

    fn invalid_binding(reason: String) -> Self::Error {
        PtxError::InvalidBinding(reason)
    }

    fn unsupported(reason: String) -> Self::Error {
        PtxError::Unsupported(reason)
    }

    fn overflow() -> Self::Error {
        PtxError::Overflow
    }
}

/// Pure, fully rendered CUDA graph plan. No Driver resource is created here.
pub struct CudaGraphPrefixPlan {
    plan: StaticSchedulePlan<RenderedPtx>,
    renderer: PtxRenderer,
}

impl CudaGraphPrefixPlan {
    pub fn plan(items: &[ScheduleItem], renderer: PtxRenderer) -> Result<Self, PtxError> {
        Self::plan_with_retained(items, None, renderer)
    }

    pub(crate) fn plan_for_outputs(
        items: &[ScheduleItem],
        retained: &[u64],
        renderer: PtxRenderer,
    ) -> Result<Self, PtxError> {
        Self::plan_with_retained(items, Some(retained), renderer)
    }

    fn plan_with_retained(
        items: &[ScheduleItem],
        retained: Option<&[u64]>,
        renderer: PtxRenderer,
    ) -> Result<Self, PtxError> {
        let adapter = CudaGraphPlanAdapter { renderer };
        Ok(Self {
            plan: StaticSchedulePlan::build(&adapter, items, retained)?,
            renderer,
        })
    }

    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.plan.compiled_cache_keys()
    }
}

/// Reusable fixed-pointer CUDA graph. Host inputs may change only within the
/// exact planned dtype/shape/byte schema.
pub struct PreparedCudaGraphPrefix {
    graph: Option<PrimaryGraphExec>,
    stream: Option<Stream>,
    leases: BTreeMap<u64, Arc<PrimaryBufferLease>>,
    buffer_plans: BTreeMap<u64, crate::runtime::static_schedule::StaticBufferPlan>,
    external_inputs: Vec<u64>,
    retained_outputs: Vec<u64>,
    kernel_cache_keys: Vec<String>,
    completion_fence: Option<Arc<crate::PrimaryEventFence>>,
    fence_attached: bool,
    poisoned: bool,
}

impl PreparedCudaGraphPrefix {
    pub fn prepare(
        primary: PrimaryContext,
        items: &[ScheduleItem],
        renderer: PtxRenderer,
        cache: &ConcurrentPtxCache,
    ) -> Result<Self, PtxError> {
        Self::from_plan(primary, CudaGraphPrefixPlan::plan(items, renderer)?, cache)
    }

    pub fn from_plan(
        primary: PrimaryContext,
        planned: CudaGraphPrefixPlan,
        cache: &ConcurrentPtxCache,
    ) -> Result<Self, PtxError> {
        let plan = planned.plan;
        let has_work = plan.items().any(|item| item.extent() != 0);
        if has_work && !primary.supports_graphs() {
            return Err(PtxError::Cuda(crate::CudaError::MissingSymbol(
                "cuStreamBeginCapture",
            )));
        }

        let mut kernels = Vec::<Option<Arc<PrimaryPtxKernel>>>::with_capacity(plan.items().len());
        for item in plan.items() {
            kernels.push(if item.extent() == 0 {
                None
            } else {
                Some(cache.get_or_load(
                    &primary,
                    item.rendered().clone(),
                    planned.renderer.block_size,
                )?)
            });
        }

        let allocated_ids = plan
            .items()
            .zip(&kernels)
            .filter(|(_, kernel)| kernel.is_some())
            .flat_map(|(item, _)| item.buffer_ids().iter().copied())
            .collect::<BTreeSet<_>>();
        let allocator = primary.allocator();
        let mut leases = BTreeMap::new();
        for id in plan
            .buffer_order()
            .iter()
            .filter(|id| allocated_ids.contains(*id))
        {
            let buffer = &plan.buffers()[id];
            let bytes = NonZeroUsize::new(buffer.bytes).ok_or_else(|| {
                PtxError::Unsupported(format!(
                    "nonzero CUDA graph item requires zero-byte buffer {id}"
                ))
            })?;
            leases.insert(*id, Arc::new(allocator.allocate(bytes)?));
        }

        let (stream, graph) = if has_work {
            let stream = primary.stream()?;
            let mut capture = stream.begin_capture()?;
            for lease in leases.values() {
                capture.retain_shared(lease.clone())?;
            }
            for kernel in kernels.iter().flatten() {
                capture.retain_shared(kernel.clone())?;
            }
            for (item, kernel) in plan.items().zip(&kernels) {
                let Some(kernel) = kernel else { continue };
                let bindings = item
                    .buffer_ids()
                    .iter()
                    .zip(&item.rendered().buffers)
                    .map(|(id, abi)| {
                        Ok(PtxBinding {
                            buffer: leases
                                .get(id)
                                .ok_or_else(|| {
                                    PtxError::InvalidBinding(format!(
                                        "CUDA graph buffer {id} is absent"
                                    ))
                                })?
                                .view()?,
                            dtype: abi.dtype,
                            mutable: abi.mutable,
                        })
                    })
                    .collect::<Result<Vec<_>, PtxError>>()?;
                kernel.enqueue_captured(&mut capture, &bindings)?;
            }
            let graph = capture.finish()?.instantiate_primary_owned()?;
            (Some(stream), Some(graph))
        } else {
            (None, None)
        };

        Ok(Self {
            graph,
            stream,
            leases,
            buffer_plans: plan.buffers().clone(),
            external_inputs: plan.external_inputs().to_vec(),
            retained_outputs: plan.retained_outputs().to_vec(),
            kernel_cache_keys: plan.compiled_cache_keys(),
            completion_fence: has_work
                .then(|| primary.event_fence())
                .transpose()?
                .map(Arc::new),
            fence_attached: false,
            poisoned: false,
        })
    }

    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.kernel_cache_keys.clone()
    }

    pub fn execute(&mut self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), PtxError> {
        if self.poisoned {
            return Err(PtxError::Unsupported(
                "CUDA graph replay is poisoned after uncertain completion".into(),
            ));
        }
        let mut uploads = Vec::with_capacity(self.external_inputs.len());
        for id in &self.external_inputs {
            let plan = &self.buffer_plans[id];
            let value = values.get(id).ok_or_else(|| {
                PtxError::InvalidBinding(format!("missing CUDA graph input {id}"))
            })?;
            if value.dtype() != plan.dtype || value.shape() != &plan.source_shape {
                return Err(PtxError::InvalidBinding(format!(
                    "CUDA graph input {id} descriptor mismatch"
                )));
            }
            let bytes = value
                .to_le_bytes()
                .map_err(|_| PtxError::InvalidBinding(format!("CUDA graph input {id} bytes")))?;
            if bytes.len() != plan.bytes {
                return Err(PtxError::InvalidBinding(format!(
                    "CUDA graph input {id} byte length mismatch"
                )));
            }
            uploads.push((*id, bytes));
        }
        let mut downloads = self
            .retained_outputs
            .iter()
            .map(|id| (*id, vec![0; self.buffer_plans[id].bytes]))
            .collect::<Vec<_>>();
        for lease in self.leases.values() {
            lease.execution_metadata()?;
        }

        for (id, bytes) in &uploads {
            if let Some(lease) = self.leases.get(id) {
                lease.view()?.copy_from(0, bytes)?;
            }
        }
        if let (Some(graph), Some(stream)) = (&self.graph, &self.stream) {
            if let Err(error) = graph.launch(stream) {
                let settled = stream.synchronize().is_ok();
                return Err(self.settle_or_poison(settled, error));
            }
            let fence = self
                .completion_fence
                .as_ref()
                .cloned()
                .ok_or_else(|| PtxError::InvalidBinding("CUDA graph has no fence".into()))?;
            if let Err(error) = fence.record(stream) {
                let settled = stream.synchronize().is_ok();
                return Err(self.settle_or_poison(settled, error));
            }
            if !self.fence_attached {
                for lease in self.leases.values() {
                    if let Err(error) = lease.attach_fence(fence.clone()) {
                        let settled = stream.synchronize().is_ok();
                        return Err(self.settle_or_poison(settled, error));
                    }
                }
                self.fence_attached = true;
            }
            if let Err(error) = fence.wait() {
                self.poison();
                return Err(PtxError::Cuda(error));
            }
            for (id, bytes) in &mut downloads {
                if !bytes.is_empty() {
                    self.leases
                        .get(id)
                        .ok_or_else(|| {
                            PtxError::InvalidBinding(format!(
                                "retained CUDA graph output {id} has no lease"
                            ))
                        })?
                        .view()?
                        .copy_to(0, bytes)?;
                }
            }
        }

        let decoded = downloads
            .into_iter()
            .map(|(id, bytes)| {
                let plan = &self.buffer_plans[&id];
                TensorData::from_le_bytes(plan.source_shape.clone(), plan.dtype, &bytes)
                    .map(|value| (id, value))
                    .map_err(|_| PtxError::InvalidBinding(format!("CUDA graph output {id} bytes")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (id, value) in decoded {
            values.insert(id, value);
        }
        Ok(())
    }

    fn settle_or_poison(&mut self, settled: bool, error: crate::CudaError) -> PtxError {
        if !settled {
            self.poison();
        }
        PtxError::Cuda(error)
    }

    fn poison(&mut self) {
        self.poisoned = true;
        for lease in self.leases.values() {
            lease.quarantine();
        }
        if let Some(graph) = self.graph.as_mut() {
            graph.abandon_uncertain();
        }
        if let Some(stream) = self.stream.as_mut() {
            stream.abandon_uncertain();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Driver, Graph, Shape, Storage, schedule_many};

    fn make_primary() -> (Arc<crate::cuda::tests::Mock>, PrimaryContext) {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        (mock, primary)
    }

    fn branch() -> (crate::Schedule, u64, [u64; 2]) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], crate::DType::F32);
        let shared = graph.square(input).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(shared, one).unwrap();
        let right = graph.mul(shared, one).unwrap();
        (
            schedule_many(&graph, &[left, right]).unwrap(),
            input.index() as u64,
            [left.index() as u64, right.index() as u64],
        )
    }

    fn input(id: u64, values: [f32; 2]) -> BTreeMap<u64, TensorData> {
        BTreeMap::from([(
            id,
            TensorData::from_storage(Shape::from([2]), Storage::F32(values.into())).unwrap(),
        )])
    }

    fn prepare_outputs(
        primary: PrimaryContext,
        schedule: &crate::Schedule,
        retained: &[u64],
    ) -> PreparedCudaGraphPrefix {
        let renderer = PtxRenderer::new(80).unwrap();
        PreparedCudaGraphPrefix::from_plan(
            primary,
            CudaGraphPrefixPlan::plan_for_outputs(&schedule.items, retained, renderer).unwrap(),
            &ConcurrentPtxCache::new(),
        )
        .unwrap()
    }

    fn assert_f32(values: &BTreeMap<u64, TensorData>, id: u64, expected: &[f32]) {
        assert_eq!(values[&id].storage(), &Storage::F32(expected.to_vec()));
    }

    #[test]
    fn fixed_graph_uploads_once_retains_intermediates_and_replays_with_new_inputs() {
        let (schedule, external, retained) = branch();
        assert_eq!(schedule.items.len(), 3);
        let (mock, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &retained);
        let capture_calls = mock.calls();
        assert_eq!(
            capture_calls
                .iter()
                .filter(|call| **call == "launch")
                .count(),
            3
        );
        assert!(!capture_calls.contains(&"stream_sync"));
        assert_eq!(
            capture_calls
                .iter()
                .filter(|call| **call == "alloc")
                .count(),
            4,
            "one stable lease is allocated for each logical buffer"
        );

        let mut values = input(external, [2.0, 3.0]);
        prepared.execute(&mut values).unwrap();
        assert_f32(&values, retained[0], &[5.0, 10.0]);
        assert_f32(&values, retained[1], &[4.0, 9.0]);
        let first_calls = mock.calls();
        assert_eq!(
            first_calls.iter().filter(|call| **call == "htod").count(),
            1
        );
        assert_eq!(
            first_calls
                .iter()
                .filter(|call| **call == "graph_launch")
                .count(),
            1
        );
        assert_eq!(
            first_calls.iter().filter(|call| **call == "dtoh").count(),
            2
        );

        let mut second = input(external, [4.0, 5.0]);
        prepared.execute(&mut second).unwrap();
        assert_f32(&second, retained[0], &[17.0, 26.0]);
        assert_f32(&second, retained[1], &[16.0, 25.0]);
        let second_calls = mock.calls();
        assert_eq!(
            second_calls
                .iter()
                .filter(|call| **call == "graph_launch")
                .count(),
            2
        );
        assert_eq!(
            second_calls
                .iter()
                .filter(|call| **call == "event_record")
                .count(),
            2
        );
        assert_eq!(
            second_calls.iter().filter(|call| **call == "alloc").count(),
            4,
            "replay never reallocates a logical buffer"
        );
    }

    #[test]
    fn public_prepare_preserves_all_item_outputs() {
        let (schedule, external, _) = branch();
        let expected = schedule
            .items
            .iter()
            .map(|item| item.primary_output().id)
            .collect::<Vec<_>>();
        let (_, primary) = make_primary();
        let mut prepared = PreparedCudaGraphPrefix::prepare(
            primary,
            &schedule.items,
            PtxRenderer::new(80).unwrap(),
            &ConcurrentPtxCache::new(),
        )
        .unwrap();
        let mut values = input(external, [2.0, 3.0]);
        prepared.execute(&mut values).unwrap();
        assert!(expected.iter().all(|id| values.contains_key(id)));
    }

    #[test]
    fn descriptor_and_read_failures_publish_nothing_and_read_failure_retries() {
        let (schedule, external, retained) = branch();
        let (mock, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &retained);
        let work_before = mock.calls().len();
        let mut wrong = BTreeMap::from([(
            external,
            TensorData::from_storage([2], Storage::I32(vec![2, 3])).unwrap(),
        )]);
        assert!(prepared.execute(&mut wrong).is_err());
        assert_eq!(mock.calls().len(), work_before);

        let mut values = input(external, [2.0, 3.0]);
        let before = values.clone();
        mock.fail_dtoh_after(1, 1);
        assert!(prepared.execute(&mut values).is_err());
        assert_eq!(values, before);
        prepared.execute(&mut values).unwrap();
        assert_f32(&values, retained[0], &[5.0, 10.0]);
        assert_f32(&values, retained[1], &[4.0, 9.0]);
    }

    #[test]
    fn settled_launch_failure_retries_but_uncertain_failure_poisons() {
        let (schedule, external, retained) = branch();
        let (mock, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &retained);
        let mut values = input(external, [2.0, 3.0]);
        let before = values.clone();
        mock.fail_graph_launch_after(0, 1);
        assert!(prepared.execute(&mut values).is_err());
        assert_eq!(values, before);
        prepared.execute(&mut values).unwrap();
        assert_f32(&values, retained[0], &[5.0, 10.0]);

        let quarantine_before = crate::cuda::primary_quarantine_len();
        let (mock, primary) = make_primary();
        let mut poisoned = prepare_outputs(primary, &schedule, &retained);
        let mut values = input(external, [2.0, 3.0]);
        mock.fail_graph_launch_after(0, 1);
        mock.fail_stream_sync_after(0, 1);
        assert!(poisoned.execute(&mut values).is_err());
        let launches = mock
            .calls()
            .iter()
            .filter(|call| **call == "graph_launch")
            .count();
        assert!(poisoned.execute(&mut values).is_err());
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| **call == "graph_launch")
                .count(),
            launches
        );
        drop(poisoned);
        assert!(crate::cuda::primary_quarantine_len() >= quarantine_before + 2);
        assert!(
            mock.calls()
                .iter()
                .all(|call| !matches!(*call, "graph_exec_destroy" | "stream_destroy")),
            "process-reachable quarantine retains graph, stream, and captured owners"
        );
    }

    #[test]
    fn event_record_failure_retries_but_wait_failure_poisons() {
        let (schedule, external, retained) = branch();
        let (mock, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &retained);
        let mut values = input(external, [2.0, 3.0]);
        let before = values.clone();
        mock.set_event_record_result(1);
        assert!(prepared.execute(&mut values).is_err());
        assert_eq!(values, before);
        mock.set_event_record_result(0);
        prepared.execute(&mut values).unwrap();
        assert_f32(&values, retained[0], &[5.0, 10.0]);

        let (mock, primary) = make_primary();
        let mut poisoned = prepare_outputs(primary, &schedule, &retained);
        let mut values = input(external, [2.0, 3.0]);
        let before = values.clone();
        mock.set_event_sync_result(1);
        assert!(poisoned.execute(&mut values).is_err());
        assert_eq!(values, before);
        mock.set_event_sync_result(0);
        let launches = mock
            .calls()
            .iter()
            .filter(|call| **call == "graph_launch")
            .count();
        assert!(poisoned.execute(&mut values).is_err());
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| **call == "graph_launch")
                .count(),
            launches
        );
    }

    #[test]
    fn zero_domain_has_no_graph_stream_module_or_allocation() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0], crate::DType::F32);
        let output = graph.square(input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let (mock, primary) = make_primary();
        let mut prepared = PreparedCudaGraphPrefix::prepare(
            primary,
            &schedule.items,
            PtxRenderer::new(80).unwrap(),
            &ConcurrentPtxCache::new(),
        )
        .unwrap();
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage([0], Storage::F32(vec![])).unwrap(),
        )]);
        prepared.execute(&mut values).unwrap();
        assert_eq!(
            values[&(output.index() as u64)].storage(),
            &Storage::F32(vec![])
        );
        assert!(mock.calls().iter().all(|call| {
            !matches!(
                *call,
                "module_load"
                    | "alloc"
                    | "stream_create"
                    | "event_create"
                    | "capture_begin"
                    | "graph_launch"
            )
        }));
    }
}
