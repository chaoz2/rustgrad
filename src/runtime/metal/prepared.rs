//! Retained, nonexecuting Metal preparation for a pure schedule prefix.
use super::{
    MetalBuffer, MetalCache, MetalCommandQueue, MetalDevice, MetalError, MetalPipeline,
    MetalRenderer,
};
use crate::{ScheduleItem, TensorData};
use std::{cell::Cell, collections::BTreeMap, rc::Rc, time::Duration};

use crate::runtime::static_schedule::{
    InitializedStaticSchedule, PreparedStaticSchedule, Sealed, StaticAppendStateLink,
    StaticBufferAllocation, StaticDeviceAdapter, StaticExecutionReport, StaticHostGather,
    StaticHostOutputSelection, StaticPlanAdapter, StaticQuantizedBufferPlan, StaticRendered,
    StaticRenderedBuffer, StaticRenderedQuantizedBuffer, StaticSchedulePlan, StaticStateLink,
    bind_rendered_buffers,
};

struct MetalStaticAdapter {
    device: Option<MetalDevice>,
    renderer: MetalRenderer,
    cache: Option<MetalCache>,
    cache_miss_pipeline_build_wall_time: Rc<Cell<Duration>>,
    append_state: BTreeMap<u64, StaticAppendStateLink>,
    append_span_iota: BTreeMap<u64, StaticAppendStateLink>,
    host_gathers: BTreeMap<u64, StaticHostGather>,
}

impl MetalStaticAdapter {
    fn planner(renderer: MetalRenderer) -> Self {
        Self {
            device: None,
            renderer,
            cache: None,
            cache_miss_pipeline_build_wall_time: Rc::new(Cell::new(Duration::ZERO)),
            append_state: BTreeMap::new(),
            append_span_iota: BTreeMap::new(),
            host_gathers: BTreeMap::new(),
        }
    }

    fn runtime(device: MetalDevice, renderer: MetalRenderer) -> Self {
        let cache = device.cache();
        Self {
            device: Some(device),
            renderer,
            cache: Some(cache),
            cache_miss_pipeline_build_wall_time: Rc::new(Cell::new(Duration::ZERO)),
            append_state: BTreeMap::new(),
            append_span_iota: BTreeMap::new(),
            host_gathers: BTreeMap::new(),
        }
    }

    fn with_append_state(mut self, links: &[StaticAppendStateLink]) -> Result<Self, MetalError> {
        for link in links {
            if self.append_state.insert(link.output, *link).is_some() {
                return Err(MetalError::InvalidBinding(
                    "duplicate Metal append-state output".into(),
                ));
            }
            if let Some(iota) = link.iota {
                self.append_span_iota.entry(iota).or_insert(*link);
            }
        }
        Ok(self)
    }

    fn with_host_gathers(mut self, links: &[StaticHostGather]) -> Result<Self, MetalError> {
        for link in links {
            if self
                .host_gathers
                .insert(link.output, link.clone())
                .is_some()
            {
                return Err(MetalError::InvalidBinding(
                    "duplicate Metal host Gather output".into(),
                ));
            }
        }
        Ok(self)
    }

    fn device(&self) -> Result<&MetalDevice, MetalError> {
        self.device
            .as_ref()
            .ok_or_else(|| MetalError::InvalidBinding("Metal plan has no device".into()))
    }
}

impl Sealed for MetalStaticAdapter {}

impl StaticPlanAdapter for MetalStaticAdapter {
    type Error = MetalError;
    type Rendered = super::RenderedMetal;

    fn render(&self, item: &ScheduleItem) -> Result<StaticRendered<Self::Rendered>, Self::Error> {
        let rendered = match (
            self.append_state.get(&item.outputs.primary().id),
            self.append_span_iota.get(&item.outputs.primary().id),
            self.host_gathers.get(&item.outputs.primary().id),
        ) {
            (Some(_), _, Some(_)) | (_, Some(_), Some(_)) | (Some(_), Some(_), _) => {
                return Err(MetalError::InvalidBinding(
                    "Metal item cannot mix append state, span iota, and host Gather policy".into(),
                ));
            }
            (Some(link), None, None) => self.renderer.render_append_state(&item.kernel, link)?,
            (None, Some(link), None) => self
                .renderer
                .render_authenticated_append_span_iota(&item.kernel, link)?,
            (None, None, Some(link)) => self.renderer.render_host_gather(&item.kernel, link)?,
            (None, None, None) => self.renderer.render(&item.kernel)?,
        };
        rendered.validate_schedule_bindings(item.ordered_inputs())?;
        rendered.validate_quantized_schedule_bindings(&item.quantized_input_bindings)?;
        if rendered.transaction.is_some() {
            return Err(MetalError::Unsupported(
                "guarded Metal prefixes require a staged candidate ABI".into(),
            ));
        }
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
            MetalError::InvalidBinding,
            || MetalError::Overflow,
        )?;
        let quantized_buffers = rendered
            .quantized_buffers
            .iter()
            .map(|abi| StaticRenderedQuantizedBuffer {
                id: abi.id,
                desc: abi.desc.clone(),
            })
            .collect::<Vec<_>>();
        let pointer_ids = rendered
            .pointer_order
            .iter()
            .map(|pointer| match pointer {
                super::renderer::MetalPointerAbi::Dense(index) => buffers[*index].id,
                super::renderer::MetalPointerAbi::Quantized(index) => quantized_buffers[*index].id,
            })
            .collect();
        Ok(StaticRendered {
            cache_key: rendered.cache_key.clone(),
            extent: rendered.extent,
            pointer_ids,
            buffers,
            quantized_buffers,
            artifact: rendered,
        })
    }

    fn invalid_binding(reason: String) -> Self::Error {
        MetalError::InvalidBinding(reason)
    }
    fn unsupported(reason: String) -> Self::Error {
        MetalError::Unsupported(reason)
    }
    fn overflow() -> Self::Error {
        MetalError::Overflow
    }

    fn index_out_of_bounds(axis: usize, index: usize, value: i32, dim: usize) -> Self::Error {
        MetalError::IndexOutOfBounds {
            axis,
            index,
            value,
            dim,
        }
    }
}

impl StaticDeviceAdapter for MetalStaticAdapter {
    type Kernel = Rc<MetalPipeline>;
    type Buffer = MetalBuffer;
    type Queue = MetalCommandQueue;

    fn prepare_zero_extent(&self) -> bool {
        false
    }
    fn compile(&self, rendered: &Self::Rendered) -> Result<Self::Kernel, Self::Error> {
        let loaded = self
            .cache
            .as_ref()
            .ok_or_else(|| MetalError::InvalidBinding("Metal plan has no cache".into()))?
            .load_observed(rendered)?;
        self.cache_miss_pipeline_build_wall_time.set(
            self.cache_miss_pipeline_build_wall_time
                .get()
                .checked_add(loaded.cache_miss_pipeline_build_wall_time)
                .ok_or(MetalError::Overflow)?,
        );
        Ok(loaded.pipeline)
    }
    fn compiled_cache_key(&self, kernel: &Self::Kernel) -> String {
        kernel.rendered().cache_key.clone()
    }
    fn allocate(&self, request: StaticBufferAllocation) -> Result<Self::Buffer, Self::Error> {
        self.device()?.allocate_static(request)
    }
    fn allocate_quantized(
        &self,
        plan: &StaticQuantizedBufferPlan,
    ) -> Result<Self::Buffer, Self::Error> {
        self.device()?.allocate_static_quantized(plan)
    }
    fn create_queue(&self) -> Result<Self::Queue, Self::Error> {
        self.device()?.create_queue()
    }
    fn write(
        &self,
        queue: &Self::Queue,
        buffer: &Self::Buffer,
        bytes: &[u8],
    ) -> Result<(), Self::Error> {
        queue.write(buffer, 0, bytes)
    }
    fn launch_and_wait(
        &self,
        queue: &Self::Queue,
        kernel: &Self::Kernel,
        buffers: &[&Self::Buffer],
    ) -> Result<(), Self::Error> {
        if kernel.rendered().indexed_movement.is_some() {
            return kernel
                .launch_transactional(queue, buffers, self.renderer.local_size)?
                .wait();
        }
        let command = if !kernel.rendered().quantized_buffers.is_empty() {
            kernel.launch_capture_owned_quantized(queue, buffers, self.renderer.local_size)?
        } else {
            kernel.launch(queue, buffers, self.renderer.local_size)?
        };
        if let Some(command) = command {
            command.collect()?;
        }
        Ok(())
    }
    fn read(
        &self,
        queue: &Self::Queue,
        buffer: &Self::Buffer,
        bytes: &mut [u8],
    ) -> Result<(), Self::Error> {
        queue.read(buffer, 0, bytes)
    }
    fn cache_len(&self) -> usize {
        self.cache.as_ref().map_or(0, MetalCache::len)
    }
}

/// A fully rendered, validated pure prefix before any Metal resource is created.
pub struct MetalPrefixPlan {
    plan: StaticSchedulePlan<super::RenderedMetal>,
    renderer: MetalRenderer,
}

impl MetalPrefixPlan {
    /// Performs deterministic renderer, schedule, and physical-buffer validation only.
    pub fn plan(items: &[ScheduleItem], renderer: MetalRenderer) -> Result<Self, MetalError> {
        let adapter = MetalStaticAdapter::planner(renderer.clone());
        let plan = StaticSchedulePlan::build(&adapter, items, None)?;
        if !plan.quantized_buffers().is_empty() {
            return Err(MetalError::Unsupported(
                "packed Metal constants require capture-owned session initialization".into(),
            ));
        }
        Ok(Self { plan, renderer })
    }

    pub(crate) fn plan_for_outputs(
        items: &[ScheduleItem],
        retained: &[u64],
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let adapter = MetalStaticAdapter::planner(renderer.clone());
        let plan = StaticSchedulePlan::build(&adapter, items, Some(retained))?;
        if !plan.quantized_buffers().is_empty() {
            return Err(MetalError::Unsupported(
                "packed Metal constants require capture-owned session initialization".into(),
            ));
        }
        Ok(Self { plan, renderer })
    }

    pub(crate) fn plan_with_output_policy(
        items: &[ScheduleItem],
        host_outputs: &[u64],
        protected_outputs: &[u64],
        state_links: &[StaticStateLink],
        host_gathers: &[StaticHostGather],
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let adapter =
            MetalStaticAdapter::planner(renderer.clone()).with_host_gathers(host_gathers)?;
        Ok(Self {
            plan: StaticSchedulePlan::build_with_output_policy(
                &adapter,
                items,
                host_outputs,
                protected_outputs,
                state_links,
                host_gathers,
            )?,
            renderer,
        })
    }

    pub(crate) fn plan_with_append_policy(
        items: &[ScheduleItem],
        host_outputs: &[u64],
        protected_outputs: &[u64],
        append_state_links: &[StaticAppendStateLink],
        host_gathers: &[StaticHostGather],
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let adapter = MetalStaticAdapter::planner(renderer.clone())
            .with_append_state(append_state_links)?
            .with_host_gathers(host_gathers)?;
        Ok(Self {
            plan: StaticSchedulePlan::build_with_append_policy(
                &adapter,
                items,
                host_outputs,
                protected_outputs,
                append_state_links,
                host_gathers,
            )?,
            renderer,
        })
    }

    pub fn cache_keys(&self) -> Vec<String> {
        self.plan.compiled_cache_keys()
    }

    pub(super) fn rendered_kernels(&self) -> impl ExactSizeIterator<Item = &super::RenderedMetal> {
        self.plan.items().map(|item| item.rendered())
    }

    pub(super) fn item_counts(&self) -> (usize, usize) {
        let nonzero = self.plan.items().filter(|item| item.extent() != 0).count();
        (nonzero, self.plan.items().len() - nonzero)
    }

    pub(super) fn allocation_summary(&self) -> Result<(usize, usize, usize), MetalError> {
        let slots = self.plan.allocations().slots();
        let packed = self.plan.quantized_buffers();
        let bytes = slots.iter().try_fold(0usize, |total, allocation| {
            total
                .checked_add(allocation.physical_bytes())
                .ok_or(MetalError::Overflow)
        })?;
        let allocated_packed_count = packed
            .values()
            .filter(|plan| plan.requires_native_handle)
            .count();
        let bytes = packed
            .values()
            .filter(|plan| plan.requires_native_handle)
            .try_fold(bytes, |total, plan| {
                total
                    .checked_add(if plan.desc.bytes == 0 {
                        4
                    } else {
                        plan.desc.bytes
                    })
                    .ok_or(MetalError::Overflow)
            })?;
        let sentinels = slots
            .iter()
            .filter(|allocation| allocation.bytes == 0 && allocation.requires_native_handle)
            .count();
        let packed_sentinels = packed
            .values()
            .filter(|plan| plan.requires_native_handle && plan.desc.bytes == 0)
            .count();
        Ok((
            slots
                .len()
                .checked_add(allocated_packed_count)
                .ok_or(MetalError::Overflow)?,
            bytes,
            sentinels
                .checked_add(packed_sentinels)
                .ok_or(MetalError::Overflow)?,
        ))
    }

    pub(super) fn external_input_ids(&self) -> &[u64] {
        self.plan.external_inputs()
    }
}

/// A fully validated pure prefix whose logical intermediates remain device-resident.
pub struct PreparedMetalPrefix {
    inner: PreparedStaticSchedule<MetalStaticAdapter>,
    cache_miss_pipeline_build_wall_time: Duration,
}

pub(super) struct InitializedMetalPrefix {
    inner: InitializedStaticSchedule<MetalStaticAdapter>,
}

/// An authenticated captured schedule bound to one prepared Metal prefix.
pub type CapturedMetalPrefix = crate::runtime::CapturedStaticPrefix<PreparedMetalPrefix>;

impl PreparedMetalPrefix {
    pub fn prepare(
        device: MetalDevice,
        items: &[ScheduleItem],
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let plan = MetalPrefixPlan::plan(items, renderer)?;
        Self::from_plan(device, plan)
    }

    /// Prepares one authenticated concrete captured schedule for Metal execution.
    pub fn prepare_capture(
        device: MetalDevice,
        capture: &crate::CapturedSchedule,
        renderer: MetalRenderer,
    ) -> Result<CapturedMetalPrefix, MetalError> {
        let projection = crate::runtime::static_schedule::CapturedStaticExecution::new(capture)
            .map_err(MetalError::InvalidBinding)?;
        let plan =
            MetalPrefixPlan::plan_for_outputs(&capture.items, projection.retained(), renderer)?;
        let prepared = Self::from_plan(device, plan)?;
        Ok(crate::runtime::CapturedStaticPrefix::new(
            prepared, projection,
        ))
    }

    pub fn from_plan(device: MetalDevice, plan: MetalPrefixPlan) -> Result<Self, MetalError> {
        let append_state = plan.plan.append_state_links().to_vec();
        let host_gathers = plan.plan.host_gathers().to_vec();
        let adapter = MetalStaticAdapter::runtime(device, plan.renderer)
            .with_append_state(&append_state)?
            .with_host_gathers(&host_gathers)?;
        let build_wall_time = adapter.cache_miss_pipeline_build_wall_time.clone();
        Ok(Self {
            inner: PreparedStaticSchedule::from_plan(adapter, plan.plan)?,
            cache_miss_pipeline_build_wall_time: build_wall_time.get(),
        })
    }

    pub(super) const fn cache_miss_pipeline_build_wall_time(&self) -> Duration {
        self.cache_miss_pipeline_build_wall_time
    }

    pub fn cache_len(&self) -> usize {
        self.inner.cache_len()
    }
    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.inner.compiled_cache_keys()
    }
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), MetalError> {
        self.inner.execute(values)
    }

    pub(super) fn initialize_resident(
        self,
        values: &BTreeMap<u64, TensorData>,
        resident_ids: &std::collections::BTreeSet<u64>,
        quantized: &BTreeMap<u64, crate::QuantizedTensorData>,
    ) -> Result<(InitializedMetalPrefix, StaticExecutionReport), MetalError> {
        let (inner, report) =
            self.inner
                .initialize_resident_with_quantized(values, resident_ids, quantized)?;
        Ok((InitializedMetalPrefix { inner }, report))
    }
}

impl InitializedMetalPrefix {
    pub(super) fn rendered_kernels(&self) -> impl Iterator<Item = &super::RenderedMetal> {
        self.inner.kernels().map(|kernel| kernel.rendered())
    }

    pub(super) fn execute(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<StaticExecutionReport, MetalError> {
        self.inner.execute(values)
    }

    pub(super) fn execute_stateful(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
        alternate_state_bank: bool,
    ) -> Result<StaticExecutionReport, MetalError> {
        self.inner.execute_stateful(values, alternate_state_bank)
    }

    pub(super) fn execute_append_state(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
        committed_position: usize,
        host_outputs: StaticHostOutputSelection,
    ) -> Result<StaticExecutionReport, MetalError> {
        self.inner
            .execute_append_state(values, committed_position, host_outputs)
    }
}

impl CapturedMetalPrefix {
    /// Executes the capture transaction and returns detached outputs in request order.
    pub fn execute(
        &self,
        inputs: &BTreeMap<String, TensorData>,
    ) -> Result<Vec<TensorData>, MetalError> {
        self.transact(inputs, MetalError::InvalidBinding, |prepared, values| {
            prepared.execute(values)
        })
    }
}
