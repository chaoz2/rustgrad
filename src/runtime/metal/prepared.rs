//! Retained, nonexecuting Metal preparation for a pure schedule prefix.
use super::{
    MetalBuffer, MetalCache, MetalCommandQueue, MetalDevice, MetalError, MetalPipeline,
    MetalRenderer,
};
use crate::{ScheduleItem, TensorData};
use std::{collections::BTreeMap, rc::Rc};

use crate::runtime::static_schedule::{
    PreparedStaticSchedule, Sealed, StaticBufferAllocation, StaticDeviceAdapter, StaticPlanAdapter,
    StaticRendered, StaticRenderedBuffer, StaticSchedulePlan, bind_rendered_buffers,
};

struct MetalStaticAdapter {
    device: Option<MetalDevice>,
    renderer: MetalRenderer,
    cache: Option<MetalCache>,
}

impl MetalStaticAdapter {
    fn planner(renderer: MetalRenderer) -> Self {
        Self {
            device: None,
            renderer,
            cache: None,
        }
    }

    fn runtime(device: MetalDevice, renderer: MetalRenderer) -> Self {
        let cache = device.cache();
        Self {
            device: Some(device),
            renderer,
            cache: Some(cache),
        }
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
        let rendered = self.renderer.render(&item.kernel)?;
        rendered.validate_schedule_bindings(item.ordered_inputs())?;
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
        Ok(StaticRendered {
            cache_key: rendered.cache_key.clone(),
            extent: rendered.extent,
            buffers,
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
}

impl StaticDeviceAdapter for MetalStaticAdapter {
    type Kernel = Rc<MetalPipeline>;
    type Buffer = MetalBuffer;
    type Queue = MetalCommandQueue;

    fn prepare_zero_extent(&self) -> bool {
        false
    }
    fn compile(&self, rendered: &Self::Rendered) -> Result<Self::Kernel, Self::Error> {
        self.cache
            .as_ref()
            .ok_or_else(|| MetalError::InvalidBinding("Metal plan has no cache".into()))?
            .load(rendered)
    }
    fn compiled_cache_key(&self, kernel: &Self::Kernel) -> String {
        kernel.rendered().cache_key.clone()
    }
    fn allocate(&self, request: StaticBufferAllocation) -> Result<Self::Buffer, Self::Error> {
        self.device()?.allocate_static(request)
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
        if let Some(command) = kernel.launch(queue, buffers, self.renderer.local_size)? {
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
        Ok(Self {
            plan: StaticSchedulePlan::build(&adapter, items, None)?,
            renderer,
        })
    }

    pub(crate) fn plan_for_outputs(
        items: &[ScheduleItem],
        retained: &[u64],
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let adapter = MetalStaticAdapter::planner(renderer.clone());
        Ok(Self {
            plan: StaticSchedulePlan::build(&adapter, items, Some(retained))?,
            renderer,
        })
    }

    pub fn cache_keys(&self) -> Vec<String> {
        self.plan.compiled_cache_keys()
    }
}

/// A fully validated pure prefix whose logical intermediates remain device-resident.
pub struct PreparedMetalPrefix {
    inner: PreparedStaticSchedule<MetalStaticAdapter>,
}

impl PreparedMetalPrefix {
    pub fn prepare(
        device: MetalDevice,
        items: &[ScheduleItem],
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let plan = MetalPrefixPlan::plan(items, renderer)?;
        Self::from_plan(device, plan)
    }

    pub fn from_plan(device: MetalDevice, plan: MetalPrefixPlan) -> Result<Self, MetalError> {
        Ok(Self {
            inner: PreparedStaticSchedule::from_plan(
                MetalStaticAdapter::runtime(device, plan.renderer),
                plan.plan,
            )?,
        })
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
}
