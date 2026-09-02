//! Fixed-schema, single-primary-context CUDA graph replay for pure prefixes.

use crate::cuda::PrimaryGraphExec;
use crate::runtime::static_schedule::{
    Sealed, StaticPlanAdapter, StaticRendered, StaticRenderedBuffer, StaticSchedulePlan,
    StaticSymbolicBackend, StaticSymbolicProgram, bind_rendered_buffers,
};
use crate::{
    CapturedSchedule, ConcurrentPtxCache, PrimaryBufferLease, PrimaryContext, PrimaryPtxKernel,
    PtxBinding, PtxError, PtxRenderer, RenderedPtx, ReplayError, ReplayInput, ScheduleItem, Stream,
    SymbolicInvocation, SymbolicParameter, TensorData,
};
use std::{collections::BTreeMap, fmt, num::NonZeroUsize, sync::Arc};

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
            rendered.buffers.iter().scan(0usize, |ordinal, abi| {
                let output_ordinal = abi.mutable.then(|| {
                    let current = *ordinal;
                    *ordinal += 1;
                    current
                });
                Some(StaticRenderedBuffer {
                    id: abi.id,
                    dtype: abi.dtype,
                    source_shape: abi.source_shape.clone(),
                    elements: abi.elements,
                    output_ordinal,
                })
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
    leases: Vec<Arc<PrimaryBufferLease>>,
    logical_slots: BTreeMap<u64, usize>,
    buffer_plans: BTreeMap<u64, crate::runtime::static_schedule::StaticBufferPlan>,
    external_inputs: Vec<u64>,
    retained_outputs: Vec<u64>,
    kernel_cache_keys: Vec<String>,
    completion_fence: Option<Arc<crate::PrimaryEventFence>>,
    fence_attached: bool,
    poisoned: bool,
}

/// An authenticated captured schedule bound to one prepared CUDA graph prefix.
pub type CapturedCudaGraphPrefix = crate::runtime::CapturedStaticPrefix<PreparedCudaGraphPrefix>;

impl PreparedCudaGraphPrefix {
    pub fn prepare(
        primary: PrimaryContext,
        items: &[ScheduleItem],
        renderer: PtxRenderer,
        cache: &ConcurrentPtxCache,
    ) -> Result<Self, PtxError> {
        Self::from_plan(primary, CudaGraphPrefixPlan::plan(items, renderer)?, cache)
    }

    /// Prepares one authenticated concrete captured schedule for CUDA graph execution.
    pub fn prepare_capture(
        primary: PrimaryContext,
        capture: &crate::CapturedSchedule,
        renderer: PtxRenderer,
        cache: &ConcurrentPtxCache,
    ) -> Result<CapturedCudaGraphPrefix, PtxError> {
        let projection = crate::runtime::static_schedule::CapturedStaticExecution::new(capture)
            .map_err(PtxError::InvalidBinding)?;
        let plan =
            CudaGraphPrefixPlan::plan_for_outputs(&capture.items, projection.retained(), renderer)?;
        let prepared = Self::from_plan(primary, plan, cache)?;
        Ok(crate::runtime::CapturedStaticPrefix::new(
            prepared, projection,
        ))
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

        let allocator = primary.allocator();
        let mut leases = Vec::with_capacity(plan.allocations().slots().len());
        for (slot, allocation) in plan.allocations().slots().iter().enumerate() {
            let bytes = NonZeroUsize::new(allocation.bytes).ok_or_else(|| {
                PtxError::Unsupported(format!(
                    "nonzero CUDA graph item requires zero-byte physical slot {slot}"
                ))
            })?;
            leases.push(Arc::new(allocator.allocate(bytes)?));
        }
        let logical_slots = plan.allocations().logical_slots().clone();

        let (stream, graph) = if has_work {
            let stream = primary.stream()?;
            let mut capture = stream.begin_capture()?;
            for lease in &leases {
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
                            buffer: logical_slots
                                .get(id)
                                .and_then(|slot| leases.get(*slot))
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
            logical_slots,
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
        for lease in &self.leases {
            lease.execution_metadata()?;
        }

        for (id, bytes) in &uploads {
            if let Some(lease) = self
                .logical_slots
                .get(id)
                .and_then(|slot| self.leases.get(*slot))
            {
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
                for lease in &self.leases {
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
                    self.logical_slots
                        .get(id)
                        .and_then(|slot| self.leases.get(*slot))
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
        for lease in &self.leases {
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

impl CapturedCudaGraphPrefix {
    /// Executes the capture transaction and returns detached outputs in request order.
    pub fn execute(
        &mut self,
        inputs: &BTreeMap<String, TensorData>,
    ) -> Result<Vec<TensorData>, PtxError> {
        self.transact_mut(inputs, PtxError::InvalidBinding, |prepared, values| {
            prepared.execute(values)
        })
    }
}

/// Error returned by authenticated symbolic CUDA program construction or execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaSymbolicError {
    Replay(ReplayError),
    Ptx(PtxError),
}

impl fmt::Display for CudaSymbolicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Replay(error) => error.fmt(f),
            Self::Ptx(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CudaSymbolicError {}

impl From<ReplayError> for CudaSymbolicError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<PtxError> for CudaSymbolicError {
    fn from(value: PtxError) -> Self {
        Self::Ptx(value)
    }
}

struct CudaSymbolicBackend {
    primary: PrimaryContext,
    renderer: PtxRenderer,
    cache: Arc<ConcurrentPtxCache>,
}

impl Sealed for CudaSymbolicBackend {}

impl StaticSymbolicBackend for CudaSymbolicBackend {
    type Error = CudaSymbolicError;
    type Plan = CudaGraphPrefixPlan;
    type Prepared = PreparedCudaGraphPrefix;

    fn replay_error(error: ReplayError) -> Self::Error {
        error.into()
    }

    fn invalid_binding(reason: String) -> Self::Error {
        PtxError::InvalidBinding(reason).into()
    }

    fn internal_error(reason: String) -> Self::Error {
        PtxError::Unsupported(reason).into()
    }

    fn plan(
        &self,
        capture: &CapturedSchedule,
        retained: &[u64],
    ) -> Result<Self::Plan, Self::Error> {
        CudaGraphPrefixPlan::plan_for_outputs(&capture.items, retained, self.renderer)
            .map_err(Into::into)
    }

    fn prepare(&self, plan: Self::Plan) -> Result<Self::Prepared, Self::Error> {
        PreparedCudaGraphPrefix::from_plan(self.primary.clone(), plan, &self.cache)
            .map_err(Into::into)
    }

    fn execute(
        &self,
        prepared: &mut Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), Self::Error> {
        prepared.execute(values).map_err(Into::into)
    }

    fn cache_keys(&self, prepared: &Self::Prepared) -> Vec<String> {
        prepared.kernel_cache_keys()
    }
}

/// Successful CUDA symbolic invocation provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaSymbolicTrace {
    body_identity: u64,
    concrete_identity: u64,
    bindings: Vec<(u64, i64)>,
    prepared_now: bool,
    kernel_cache_keys: Vec<String>,
}

impl CudaSymbolicTrace {
    pub fn body_identity(&self) -> u64 {
        self.body_identity
    }

    pub fn concrete_identity(&self) -> u64 {
        self.concrete_identity
    }

    pub fn bindings(&self) -> &[(u64, i64)] {
        &self.bindings
    }

    pub fn prepared_now(&self) -> bool {
        self.prepared_now
    }

    pub fn kernel_cache_keys(&self) -> &[String] {
        &self.kernel_cache_keys
    }
}

/// Detached ordered outputs from one successful CUDA symbolic invocation.
#[derive(Clone, Debug)]
pub struct CudaSymbolicResult {
    outputs: Vec<TensorData>,
    trace: CudaSymbolicTrace,
}

impl CudaSymbolicResult {
    pub fn outputs(&self) -> &[TensorData] {
        &self.outputs
    }

    pub fn into_outputs(self) -> Vec<TensorData> {
        self.outputs
    }

    pub fn into_parts(self) -> (Vec<TensorData>, CudaSymbolicTrace) {
        (self.outputs, self.trace)
    }

    pub fn trace(&self) -> &CudaSymbolicTrace {
        &self.trace
    }
}

/// Owned bounded-symbolic CUDA program with a one-entry last-successful
/// concrete specialization cache. A cache miss prepares device resources only
/// after the complete symbolic invocation and pure PTX/static plan validate.
pub struct CudaSymbolicProgram {
    inner: StaticSymbolicProgram<CudaSymbolicBackend>,
}

impl CudaSymbolicProgram {
    pub fn new(
        primary: PrimaryContext,
        capture: CapturedSchedule,
        renderer: PtxRenderer,
    ) -> Result<Self, CudaSymbolicError> {
        let output_order = (0..capture.requested.len()).collect();
        Self::build(
            primary,
            capture,
            output_order,
            renderer,
            Arc::new(ConcurrentPtxCache::new()),
        )
    }

    /// Creates a program using a caller-owned concurrent PTX cache. The
    /// program retains the cache; compiled modules may therefore be shared
    /// across independently owned symbolic programs on the same context.
    pub fn with_cache(
        primary: PrimaryContext,
        capture: CapturedSchedule,
        renderer: PtxRenderer,
        cache: Arc<ConcurrentPtxCache>,
    ) -> Result<Self, CudaSymbolicError> {
        let output_order = (0..capture.requested.len()).collect();
        Self::build(primary, capture, output_order, renderer, cache)
    }

    /// Creates a program whose detached result selects captured outputs by
    /// position. Repeated positions produce independent owned values.
    pub fn with_output_order(
        primary: PrimaryContext,
        capture: CapturedSchedule,
        output_order: Vec<usize>,
        renderer: PtxRenderer,
    ) -> Result<Self, CudaSymbolicError> {
        Self::build(
            primary,
            capture,
            output_order,
            renderer,
            Arc::new(ConcurrentPtxCache::new()),
        )
    }

    fn build(
        primary: PrimaryContext,
        capture: CapturedSchedule,
        output_order: Vec<usize>,
        renderer: PtxRenderer,
        cache: Arc<ConcurrentPtxCache>,
    ) -> Result<Self, CudaSymbolicError> {
        let body =
            crate::engine::AuthenticatedSymbolicBody::new(capture, output_order, "CUDA symbolic")?;
        Ok(Self {
            inner: StaticSymbolicProgram::new(
                body,
                CudaSymbolicBackend {
                    primary,
                    renderer,
                    cache,
                },
            ),
        })
    }

    pub fn body_identity(&self) -> u64 {
        self.inner.body().capture().identity
    }

    pub fn inputs(&self) -> &[ReplayInput] {
        &self.inner.body().capture().inputs
    }

    pub fn parameters(&self) -> impl ExactSizeIterator<Item = &SymbolicParameter> {
        self.inner.body().schema().parameters().iter()
    }

    pub fn output_count(&self) -> usize {
        self.inner.body().output_order().len()
    }

    pub fn output_order(&self) -> &[usize] {
        self.inner.body().output_order()
    }

    pub fn run(
        &self,
        invocation: SymbolicInvocation,
    ) -> Result<CudaSymbolicResult, CudaSymbolicError> {
        let result = self.inner.run(invocation)?;
        Ok(CudaSymbolicResult {
            outputs: result.outputs,
            trace: CudaSymbolicTrace {
                body_identity: result.body_identity,
                concrete_identity: result.concrete_identity,
                bindings: result.bindings,
                prepared_now: result.prepared_now,
                kernel_cache_keys: result.cache_keys,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CpuSymbolicProgram, Driver, Graph, Shape, Storage, SymbolicCaptureSpec, SymbolicExpr,
        SymbolicShape, schedule_many,
    };

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

    fn reusable_linear() -> (crate::Schedule, u64, [u64; 4]) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], crate::DType::F32);
        let first_value = graph.square(input).unwrap();
        let first = graph.contiguous(first_value).unwrap();
        let second_value = graph.square(first).unwrap();
        let second = graph.contiguous(second_value).unwrap();
        let third_value = graph.square(second).unwrap();
        let third = graph.contiguous(third_value).unwrap();
        let output_value = graph.square(third).unwrap();
        let output = graph.contiguous(output_value).unwrap();
        (
            crate::schedule(&graph, output).unwrap(),
            input.index() as u64,
            [
                first.index() as u64,
                second.index() as u64,
                third.index() as u64,
                output.index() as u64,
            ],
        )
    }

    #[test]
    fn cuda_graph_prefix_plan_accepts_prefix_scan_lane_extent_without_identity_drift() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], crate::DType::F32);
        let output = graph.cumsum(input, 1).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let [item] = schedule.items.as_slice() else {
            panic!("prefix scan must remain one scheduled item")
        };
        let schedule_identity = item.cache_key;
        assert_eq!(
            crate::uop::artifact::encode_schedule_identity(&item.kernel).unwrap()[4],
            18
        );
        let renderer = PtxRenderer::new(80).unwrap();
        let rendered = renderer.render(&item.kernel).unwrap();
        assert_eq!(rendered.buffers.last().unwrap().elements, 6);
        assert_eq!(rendered.extent, 2);
        let rendered_identity = rendered.cache_key.clone();

        let planned = CudaGraphPrefixPlan::plan(&schedule.items, renderer).unwrap();
        assert_eq!(planned.kernel_cache_keys(), vec![rendered_identity]);
        assert_eq!(schedule.items[0].cache_key, schedule_identity);
    }

    #[test]
    fn cuda_graph_prefix_plan_accepts_bitcast_byte_extent_without_identity_drift() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("bytes", [2, 4], crate::DType::U8);
        let output = graph.bitcast(input, crate::DType::U32).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let [item] = schedule.items.as_slice() else {
            panic!("bitcast must remain one scheduled item")
        };
        let schedule_identity = item.cache_key;
        assert_eq!(
            crate::uop::artifact::encode_schedule_identity(&item.kernel).unwrap()[4],
            18
        );
        let renderer = PtxRenderer::new(80).unwrap();
        let rendered = renderer.render(&item.kernel).unwrap();
        assert_eq!(rendered.extent, 8);
        assert_eq!(rendered.buffers[0].elements, 8);
        assert_eq!(rendered.buffers[1].elements, 2);
        let rendered_identity = rendered.cache_key.clone();

        let planned = CudaGraphPrefixPlan::plan(&schedule.items, renderer).unwrap();
        assert_eq!(planned.kernel_cache_keys(), vec![rendered_identity]);
        assert_eq!(schedule.items[0].cache_key, schedule_identity);

        let (_, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &[output.index() as u64]);
        let mut realized = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage([2, 4], Storage::U8(vec![1, 2, 3, 4, 0, 0x80, 0xff, 1]))
                .unwrap(),
        )]);
        prepared.execute(&mut realized).unwrap();
        assert_eq!(
            realized[&(output.index() as u64)].storage(),
            &Storage::U32(vec![0x0403_0201, 0x01ff_8000])
        );
    }

    #[test]
    fn cuda_graph_prefix_plan_executes_portable_pad_without_identity_drift() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], crate::DType::U8);
        let output = graph.pad(input, [(1, 2)], crate::Scalar::U(0x80)).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let [item] = schedule.items.as_slice() else {
            panic!("pad must remain one scheduled item")
        };
        let schedule_identity = item.cache_key;
        assert_eq!(
            crate::uop::artifact::encode_schedule_identity(&item.kernel).unwrap()[4],
            18
        );
        let renderer = PtxRenderer::new(80).unwrap();
        let rendered = renderer.render(&item.kernel).unwrap();
        assert_eq!((rendered.extent, rendered.buffers.len()), (5, 2));
        let rendered_identity = rendered.cache_key.clone();
        let planned = CudaGraphPrefixPlan::plan(&schedule.items, renderer).unwrap();
        assert_eq!(planned.kernel_cache_keys(), vec![rendered_identity]);
        assert_eq!(schedule.items[0].cache_key, schedule_identity);

        let (_, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &[output.index() as u64]);
        let mut realized = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage([2], Storage::U8(vec![0x7f, 0xff])).unwrap(),
        )]);
        prepared.execute(&mut realized).unwrap();
        assert_eq!(
            realized[&(output.index() as u64)].storage(),
            &Storage::U8(vec![0x80, 0x7f, 0xff, 0x80, 0x80])
        );
    }

    #[test]
    fn cuda_graph_executes_both_portable_sort_outputs_without_identity_drift() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], crate::DType::F32);
        let (values, indices) = graph.sort(input, 1, false).unwrap();
        let schedule = schedule_many(&graph, &[values, indices]).unwrap();
        let [item] = schedule.items.as_slice() else {
            panic!("coupled sort must remain one scheduled item")
        };
        let schedule_identity = item.cache_key;
        assert_eq!(
            crate::uop::artifact::encode_schedule_identity(&item.kernel).unwrap()[4],
            18
        );
        let input_value = TensorData::from_storage(
            [2, 3],
            Storage::F32(vec![-0.0, 0.0, f32::NAN, 3.0, 1.0, 1.0]),
        )
        .unwrap();
        let expected = crate::backend::stable_sort_pair(&input_value, 1, false).unwrap();
        let (_, primary) = make_primary();
        let mut prepared = prepare_outputs(
            primary,
            &schedule,
            &[values.index() as u64, indices.index() as u64],
        );
        let mut realized = BTreeMap::from([(input.index() as u64, input_value)]);
        prepared.execute(&mut realized).unwrap();
        assert_eq!(
            realized[&(values.index() as u64)].to_le_bytes().unwrap(),
            expected.0.to_le_bytes().unwrap()
        );
        assert_eq!(
            realized[&(indices.index() as u64)].to_le_bytes().unwrap(),
            expected.1.to_le_bytes().unwrap()
        );
        assert_eq!(schedule.items[0].cache_key, schedule_identity);
    }

    #[test]
    fn cuda_graph_reuses_one_stable_lease_for_disjoint_logical_temporaries() {
        let (schedule, external, outputs) = reusable_linear();
        assert_eq!(schedule.items.len(), 4);
        let (mock, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &[outputs[3]]);
        assert_eq!(
            mock.calls().iter().filter(|call| **call == "alloc").count(),
            4
        );
        assert_eq!(
            prepared.logical_slots[&outputs[0]],
            prepared.logical_slots[&outputs[2]]
        );
        let shared_slot = prepared.logical_slots[&outputs[0]];
        let stable_lease = prepared.leases[shared_slot].clone();

        let mut first = input(external, [1.0, -1.0]);
        prepared.execute(&mut first).unwrap();
        assert_f32(&first, outputs[3], &[1.0, 1.0]);
        let mut second = input(external, [1.0, 1.0]);
        prepared.execute(&mut second).unwrap();
        assert_f32(&second, outputs[3], &[1.0, 1.0]);
        assert!(Arc::ptr_eq(&stable_lease, &prepared.leases[shared_slot]));
        assert_eq!(
            mock.calls().iter().filter(|call| **call == "alloc").count(),
            4
        );
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
    fn captured_static_cuda_graph_projects_named_inputs_and_requested_outputs() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], crate::DType::F32);
        let squared = graph.square(input).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let output = graph.add(squared, one).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let (_, primary) = make_primary();
        let mut prepared = PreparedCudaGraphPrefix::prepare_capture(
            primary,
            &capture,
            PtxRenderer::new(80).unwrap(),
            &ConcurrentPtxCache::new(),
        )
        .unwrap();
        let outputs = prepared
            .execute(&BTreeMap::from([(
                "x".into(),
                TensorData::new([2], vec![2.0, -3.0]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].storage(), &Storage::F32(vec![5.0, 10.0]));
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
            "this branch has four simultaneously live physical slots"
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
            "replay never reallocates a physical slot"
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
    fn computed_reverse_affine_copy_replays_as_one_device_resident_prefix() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [4], crate::DType::I32);
        let squared = graph.square(input).unwrap();
        let reversed = graph
            .stride(
                squared,
                [crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            )
            .unwrap();
        let schedule = crate::schedule(&graph, reversed).unwrap();
        assert_eq!(schedule.items.len(), 2);
        assert!(matches!(
            schedule.items[1].kernel.operation(),
            crate::Operation::Movement(crate::MovementValue::Plan(plan))
                if matches!(&plan.kind, crate::MovementKernelKind::AffineCopy { .. })
        ));

        let (mock, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &[reversed.index() as u64]);
        assert_eq!(prepared.kernel_cache_keys().len(), 2);
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage([4], Storage::I32(vec![1, -2, 3, -4])).unwrap(),
        )]);
        prepared.execute(&mut values).unwrap();
        assert_eq!(
            values[&(reversed.index() as u64)].storage(),
            &Storage::I32(vec![16, 9, 4, 1])
        );
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|call| **call == "htod").count(), 1);
        assert_eq!(
            calls.iter().filter(|call| **call == "graph_launch").count(),
            1
        );
        assert_eq!(calls.iter().filter(|call| **call == "dtoh").count(), 1);
    }

    #[test]
    fn computed_broadcast_affine_copy_keeps_zero_stride_on_cuda_graph() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 1], crate::DType::F32);
        let squared = graph.square(input).unwrap();
        let expanded = graph.expand(squared, [2, 3]).unwrap();
        let schedule = crate::schedule(&graph, expanded).unwrap();
        let movement = schedule
            .items
            .iter()
            .find(|item| item.node == expanded)
            .unwrap();
        let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
            movement.kernel.operation()
        else {
            panic!("computed broadcast needs AffineCopy")
        };
        let crate::MovementKernelKind::AffineCopy { view, .. } = &plan.kind else {
            panic!("computed broadcast needs AffineCopy")
        };
        assert_eq!(view.strides, vec![1, 0]);

        let (_, primary) = make_primary();
        let mut prepared = prepare_outputs(primary, &schedule, &[expanded.index() as u64]);
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage([2, 1], Storage::F32(vec![2.0, -3.0])).unwrap(),
        )]);
        prepared.execute(&mut values).unwrap();
        assert_eq!(
            values[&(expanded.index() as u64)].storage(),
            &Storage::F32(vec![4.0, 4.0, 4.0, 9.0, 9.0, 9.0])
        );
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

    fn projected_reduction_capture() -> CapturedSchedule {
        let extent = SymbolicExpr::variable("extent", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], crate::DType::F32);
        let producer = graph.square(input).unwrap();
        let transposed = graph.permute(producer, [1, 0]).unwrap();
        let flattened = graph.reshape(transposed, [4]).unwrap();
        let output = graph
            .reduce_with_output_dtype(
                flattened,
                crate::ReduceKind::Sum,
                Some(vec![0]),
                false,
                crate::DType::F32,
            )
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.clone().into(), extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap()
    }

    fn symbolic_input(extent: usize) -> SymbolicInvocation {
        SymbolicInvocation::new()
            .with_symbol("extent", extent as i64)
            .with_input(
                "input",
                TensorData::new(
                    [extent, extent],
                    (0..extent * extent)
                        .map(|index| index as f32 - 2.0)
                        .collect(),
                )
                .unwrap(),
            )
    }

    fn matmul_capture() -> CapturedSchedule {
        let rows = SymbolicExpr::variable("rows", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], crate::DType::F32);
        let weight = graph.constant(TensorData::new([2, 2], vec![2.0, -1.0, 0.5, 3.0]).unwrap());
        let output = graph.matmul(input, weight).unwrap();
        let schedule = schedule_many(&graph, &[input, output]).unwrap();
        CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[input, output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![rows.into(), 2usize.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap()
    }

    fn matmul_input(rows: usize) -> SymbolicInvocation {
        SymbolicInvocation::new()
            .with_symbol("rows", rows as i64)
            .with_input(
                "input",
                TensorData::new(
                    [rows, 2],
                    (0..rows * 2)
                        .map(|index| index as f32 * 0.25 - 1.0)
                        .collect(),
                )
                .unwrap(),
            )
    }

    fn sequence_capture() -> CapturedSchedule {
        let time = SymbolicExpr::variable("time", 1, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("tokens", [1, 2, 3, 2], crate::DType::F32);
        let source = graph.square(input).unwrap();
        let transposed = graph.permute(source, [0, 2, 1, 3]).unwrap();
        let merged = graph.reshape(transposed, [1, 3, 4]).unwrap();
        let merged = graph.relu(merged).unwrap();
        let weight = graph.constant(
            TensorData::new(
                [4, 4],
                vec![
                    0.5, -0.25, 0.125, 0.75, -1.0, 0.5, 0.25, -0.125, 0.375, 0.625, -0.5, 0.25,
                    0.125, -0.75, 1.0, 0.5,
                ],
            )
            .unwrap(),
        );
        let projected = graph.matmul(merged, weight).unwrap();
        let squared = graph.square(projected).unwrap();
        let energy = graph
            .reduce_with_output_dtype(
                squared,
                crate::ReduceKind::Sum,
                Some(vec![2]),
                true,
                crate::DType::F32,
            )
            .unwrap();
        let energy = graph.expand(energy, [1, 3, 4]).unwrap();
        let output = graph.add(projected, energy).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![
                    1usize.into(),
                    2usize.into(),
                    time.into(),
                    2usize.into(),
                ]),
            )])),
            &BTreeMap::from([("time".into(), 3)]),
        )
        .unwrap();
        let schema = capture.symbolic.as_ref().unwrap();
        assert!(!schema.projected.is_empty());
        assert!(capture.items.iter().any(|item| {
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|node| matches!(node.operation(), crate::Operation::ReduceInit(_)))
        }));
        assert!(capture.items.iter().any(|item| {
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|node| matches!(node.operation(), crate::Operation::Matmul(_)))
        }));
        capture
    }

    fn sequence_input(time: usize) -> SymbolicInvocation {
        SymbolicInvocation::new()
            .with_symbol("time", time as i64)
            .with_input(
                "tokens",
                TensorData::new(
                    [1, 2, time, 2],
                    (0..time * 4)
                        .map(|index| index as f32 * 0.0625 - 0.375)
                        .collect(),
                )
                .unwrap(),
            )
    }

    fn assert_f32_close(actual: &TensorData, expected: &TensorData, tolerance: f64) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), crate::DType::F32);
        assert_eq!(expected.dtype(), crate::DType::F32);
        for (index, (actual, expected)) in actual
            .to_vec_f64()
            .into_iter()
            .zip(expected.to_vec_f64())
            .enumerate()
        {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "lane {index}: non-finite sequence result"
            );
            assert!(
                (actual - expected).abs() <= tolerance,
                "lane {index}: expected {expected}, got {actual} (tolerance {tolerance})"
            );
        }
    }

    #[test]
    fn symbolic_cuda_matches_cpu_and_caches_only_one_successful_specialization() {
        let capture = projected_reduction_capture();
        let cpu = CpuSymbolicProgram::with_output_order(capture.clone(), vec![0, 0]).unwrap();
        let (mock, primary) = make_primary();
        let cuda = CudaSymbolicProgram::with_output_order(
            primary,
            capture,
            vec![0, 0],
            PtxRenderer::new(80).unwrap(),
        )
        .unwrap();

        for (extent, prepared_now) in [(2, true), (2, false), (3, true), (2, true)] {
            let expected = cpu.run(symbolic_input(extent)).unwrap();
            let actual = cuda.run(symbolic_input(extent)).unwrap();
            assert_eq!(actual.outputs().len(), 2);
            assert_eq!(
                actual.outputs()[0].to_le_bytes().unwrap(),
                expected.outputs()[0].to_le_bytes().unwrap()
            );
            assert_eq!(
                actual.outputs()[1].to_le_bytes().unwrap(),
                actual.outputs()[0].to_le_bytes().unwrap()
            );
            assert_eq!(actual.trace().prepared_now(), prepared_now);
            assert_eq!(actual.trace().body_identity(), cuda.body_identity());
            assert_ne!(actual.trace().concrete_identity(), 0);
            assert!(!actual.trace().kernel_cache_keys().is_empty());
        }
        assert_eq!(cuda.output_order(), [0, 0]);
        assert!(
            mock.calls()
                .iter()
                .filter(|call| **call == "graph_launch")
                .count()
                >= 4
        );
    }

    #[test]
    fn symbolic_cuda_matmul_preserves_requested_source_constant_and_duplicate_order() {
        let capture = matmul_capture();
        assert_eq!(capture.constants.len(), 1);
        assert_eq!(capture.requested.len(), 2);
        assert!(capture.requested_passthroughs.is_empty());
        let cpu = CpuSymbolicProgram::with_output_order(capture.clone(), vec![1, 0, 1]).unwrap();
        let (_, primary) = make_primary();
        let cuda = CudaSymbolicProgram::with_output_order(
            primary,
            capture,
            vec![1, 0, 1],
            PtxRenderer::new(80).unwrap(),
        )
        .unwrap();

        let first = cuda.run(matmul_input(2)).unwrap();
        let expected = cpu.run(matmul_input(2)).unwrap();
        assert!(first.trace().prepared_now());
        assert_eq!(first.outputs().len(), 3);
        for (actual, expected) in first.outputs().iter().zip(expected.outputs()) {
            assert_eq!(
                actual.to_le_bytes().unwrap(),
                expected.to_le_bytes().unwrap()
            );
        }
        assert_eq!(
            first.outputs()[0].to_le_bytes().unwrap(),
            first.outputs()[2].to_le_bytes().unwrap()
        );
        assert_eq!(first.outputs()[1].shape(), &Shape::from([2, 2]));

        let reused = cuda.run(matmul_input(2)).unwrap();
        assert!(!reused.trace().prepared_now());
        assert_eq!(
            reused.trace().concrete_identity(),
            first.trace().concrete_identity()
        );
        let replacement = cuda.run(matmul_input(3)).unwrap();
        assert!(replacement.trace().prepared_now());
        assert_ne!(
            replacement.trace().concrete_identity(),
            first.trace().concrete_identity()
        );
        assert!(cuda.run(matmul_input(2)).unwrap().trace().prepared_now());
    }

    #[test]
    fn symbolic_cuda_runs_one_bounded_sequence_body_across_time_bindings() {
        let capture = sequence_capture();
        let body_identity = capture.identity;
        let cpu = CpuSymbolicProgram::new(capture.clone()).unwrap();
        let (_, primary) = make_primary();
        let cuda =
            CudaSymbolicProgram::new(primary, capture, PtxRenderer::new(80).unwrap()).unwrap();

        let mut identities = BTreeMap::new();
        for (time, prepared_now) in [(1usize, true), (3, true), (3, false), (4, true)] {
            let expected = cpu.run(sequence_input(time)).unwrap();
            let actual = cuda.run(sequence_input(time)).unwrap();
            assert_eq!(actual.outputs()[0].shape(), &Shape::from([1, time, 4]));
            assert_f32_close(&actual.outputs()[0], &expected.outputs()[0], 1e-5);
            assert_eq!(actual.trace().body_identity(), body_identity);
            assert_eq!(actual.trace().prepared_now(), prepared_now);
            if let Some(previous) = identities.insert(time, actual.trace().concrete_identity()) {
                assert_eq!(actual.trace().concrete_identity(), previous);
            }
        }
        assert_ne!(identities[&1], identities[&3]);
        assert_ne!(identities[&3], identities[&4]);
    }

    #[test]
    fn symbolic_cuda_rejects_invocation_and_ptx_capability_before_resources() {
        let capture = projected_reduction_capture();
        let (mock, primary) = make_primary();
        let calls = mock.calls();
        let mut tampered = capture.clone();
        tampered.identity ^= 1;
        assert!(matches!(
            CudaSymbolicProgram::new(primary.clone(), tampered, PtxRenderer::new(80).unwrap()),
            Err(CudaSymbolicError::Replay(ReplayError::Corrupt(_)))
        ));
        assert_eq!(mock.calls(), calls);
        let program =
            CudaSymbolicProgram::new(primary.clone(), capture, PtxRenderer::new(80).unwrap())
                .unwrap();
        let calls = mock.calls();
        assert!(matches!(
            program.run(SymbolicInvocation::new()),
            Err(CudaSymbolicError::Replay(ReplayError::Missing(name))) if name == "extent"
        ));
        assert_eq!(mock.calls(), calls);
        assert!(matches!(
            program.run(
                symbolic_input(2).with_symbol("unexpected", 1)
            ),
            Err(CudaSymbolicError::Replay(ReplayError::Extra(name))) if name == "unexpected"
        ));
        assert_eq!(mock.calls(), calls);
        assert!(matches!(
            program.run(symbolic_input(5)),
            Err(CudaSymbolicError::Replay(ReplayError::Symbolic(_)))
        ));
        assert_eq!(mock.calls(), calls);
        let wrong = SymbolicInvocation::new()
            .with_symbol("extent", 2)
            .with_input("input", TensorData::new([1], vec![1.0]).unwrap());
        assert!(matches!(
            program.run(wrong),
            Err(CudaSymbolicError::Replay(ReplayError::Descriptor(name))) if name == "input"
        ));
        assert_eq!(mock.calls(), calls);
        assert!(matches!(
            program.run(
                symbolic_input(2).with_input(
                    "unexpected",
                    TensorData::new([1], vec![1.0]).unwrap()
                )
            ),
            Err(CudaSymbolicError::Replay(ReplayError::Extra(name))) if name == "unexpected"
        ));
        assert_eq!(mock.calls(), calls);

        let guarded_extent = SymbolicExpr::variable("guarded_extent", 0, 4).unwrap();
        let mut guarded_graph = Graph::new();
        let guarded_input = guarded_graph.input_dtype("guarded", [2], crate::DType::F32);
        let guarded_output = guarded_graph.square(guarded_input).unwrap();
        let guarded_schedule = crate::schedule(&guarded_graph, guarded_output).unwrap();
        let guarded_capture = CapturedSchedule::capture_symbolic(
            &guarded_graph,
            &guarded_schedule,
            &[guarded_output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                guarded_input,
                SymbolicShape::new(vec![guarded_extent.clone().into()]),
            )]))
            .with_guard(crate::SymbolicGuard::divisible(guarded_extent, 2).unwrap()),
            &BTreeMap::from([("guarded_extent".into(), 2)]),
        )
        .unwrap();
        let guarded = CudaSymbolicProgram::new(
            primary.clone(),
            guarded_capture,
            PtxRenderer::new(80).unwrap(),
        )
        .unwrap();
        let calls = mock.calls();
        assert!(matches!(
            guarded.run(
                SymbolicInvocation::new()
                    .with_symbol("guarded_extent", 3)
                    .with_input("guarded", TensorData::new([3], vec![1.0; 3]).unwrap())
            ),
            Err(CudaSymbolicError::Replay(ReplayError::Symbolic(_)))
        ));
        assert_eq!(mock.calls(), calls);

        let extent = SymbolicExpr::variable("extent", 1, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], crate::DType::F32);
        let output = graph.exp(input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let unsupported = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap();
        let unsupported =
            CudaSymbolicProgram::new(primary, unsupported, PtxRenderer::new(80).unwrap()).unwrap();
        let calls = mock.calls();
        assert!(matches!(
            unsupported.run(
                SymbolicInvocation::new()
                    .with_symbol("extent", 2)
                    .with_input("input", TensorData::new([2], vec![1.0, 2.0]).unwrap())
            ),
            Err(CudaSymbolicError::Ptx(PtxError::Unsupported(_)))
        ));
        assert_eq!(mock.calls(), calls);
    }

    #[test]
    fn symbolic_cuda_zero_domain_uses_no_device_resources() {
        let extent = SymbolicExpr::variable("extent", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], crate::DType::F32);
        let output = graph.square(input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap();
        let (mock, primary) = make_primary();
        let program =
            CudaSymbolicProgram::new(primary, capture, PtxRenderer::new(80).unwrap()).unwrap();
        let calls = mock.calls();
        let result = program
            .run(
                SymbolicInvocation::new()
                    .with_symbol("extent", 0)
                    .with_input(
                        "input",
                        TensorData::from_storage([0], Storage::F32(vec![])).unwrap(),
                    ),
            )
            .unwrap();
        assert_eq!(result.outputs()[0].storage(), &Storage::F32(vec![]));
        assert_eq!(mock.calls(), calls);
    }

    #[test]
    fn symbolic_cuda_binds_exact_external_materialized_view_source() {
        let rows = SymbolicExpr::variable("rows", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("unused_source", [2, 3], crate::DType::F32);
        let producer = graph.square(input).unwrap();
        let view = graph.permute(producer, [1, 0]).unwrap();
        let output = graph.contiguous(view).unwrap();
        let schedule =
            crate::schedule_with_external_materializations(&graph, &[output], &[producer]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![rows.into(), 3usize.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        let external_name = format!("@materialized/{}", producer.index());
        assert_eq!(capture.inputs[0].name, external_name);
        let (_, primary) = make_primary();
        let result = CudaSymbolicProgram::new(primary, capture, PtxRenderer::new(80).unwrap())
            .unwrap()
            .run(SymbolicInvocation::new().with_symbol("rows", 1).with_input(
                external_name,
                TensorData::new([1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
            ))
            .unwrap();
        assert_eq!(result.outputs()[0].shape(), &Shape::from([3, 1]));
        assert_eq!(result.outputs()[0].values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn symbolic_cuda_evicts_failed_cached_execution_and_retries_fresh() {
        let capture = projected_reduction_capture();
        let (mock, primary) = make_primary();
        let program =
            CudaSymbolicProgram::new(primary, capture, PtxRenderer::new(80).unwrap()).unwrap();
        assert!(
            program
                .run(symbolic_input(2))
                .unwrap()
                .trace()
                .prepared_now()
        );
        mock.set_allocation_failure(true);
        assert!(program.run(symbolic_input(3)).is_err());
        mock.set_allocation_failure(false);
        assert!(
            !program
                .run(symbolic_input(2))
                .unwrap()
                .trace()
                .prepared_now(),
            "a failed candidate must not replace the last successful specialization"
        );
        mock.fail_graph_launch_after(0, 1);
        assert!(program.run(symbolic_input(3)).is_err());
        assert!(
            !program
                .run(symbolic_input(2))
                .unwrap()
                .trace()
                .prepared_now(),
            "a failed candidate execution must preserve the prior entry"
        );
        mock.fail_graph_launch_after(0, 1);
        mock.fail_stream_sync_after(0, 1);
        assert!(program.run(symbolic_input(2)).is_err());
        let retried = program.run(symbolic_input(2)).unwrap();
        assert!(retried.trace().prepared_now());
    }
}
