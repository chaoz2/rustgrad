//! Retained, nonexecuting WebGPU preparation for a pure schedule prefix.
use super::{
    WebGpuBuffer, WebGpuCache, WebGpuDevice, WebGpuError, WebGpuPipeline, WebGpuQueue, WgslRenderer,
};
use crate::{DType, ScheduleItem, TensorData};
use std::{collections::BTreeMap, rc::Rc};

use crate::runtime::static_schedule::{
    PreparedStaticSchedule, Sealed, StaticDeviceAdapter, StaticPlanAdapter, StaticRendered,
    StaticRenderedBuffer, StaticSchedulePlan, bind_rendered_buffers,
};

struct WebGpuStaticAdapter {
    device: WebGpuDevice,
    renderer: WgslRenderer,
    cache: WebGpuCache,
}

impl WebGpuStaticAdapter {
    fn new(device: WebGpuDevice, renderer: WgslRenderer) -> Self {
        let cache = device.cache();
        Self {
            device,
            renderer,
            cache,
        }
    }
}

impl Sealed for WebGpuStaticAdapter {}

impl StaticPlanAdapter for WebGpuStaticAdapter {
    type Error = WebGpuError;
    type Rendered = super::RenderedWgsl;

    fn render(&self, item: &ScheduleItem) -> Result<StaticRendered<Self::Rendered>, Self::Error> {
        let rendered = self.renderer.render(&item.kernel)?;
        rendered.validate_schedule_bindings(item.ordered_inputs())?;
        if rendered.transaction.is_some() {
            return Err(WebGpuError::Unsupported(
                "guarded WebGPU prefixes require a staged candidate ABI".into(),
            ));
        }
        let buffers = bind_rendered_buffers(
            item,
            rendered.buffers.iter().map(|abi| StaticRenderedBuffer {
                id: abi.id,
                dtype: abi.dtype,
                source_shape: abi.source_shape.clone(),
                elements: abi.elements,
                mutable: abi.mutable,
            }),
            WebGpuError::InvalidBinding,
            || WebGpuError::Overflow,
        )?;
        Ok(StaticRendered {
            cache_key: rendered.cache_key.clone(),
            extent: rendered.extent,
            buffers,
            artifact: rendered,
        })
    }

    fn invalid_binding(reason: String) -> Self::Error {
        WebGpuError::InvalidBinding(reason)
    }
    fn unsupported(reason: String) -> Self::Error {
        WebGpuError::Unsupported(reason)
    }
    fn overflow() -> Self::Error {
        WebGpuError::Overflow
    }
}

impl StaticDeviceAdapter for WebGpuStaticAdapter {
    type Kernel = Rc<WebGpuPipeline>;
    type Buffer = WebGpuBuffer;
    type Queue = WebGpuQueue;

    fn prepare_zero_extent(&self) -> bool {
        true
    }
    fn compile(&self, rendered: &Self::Rendered) -> Result<Self::Kernel, Self::Error> {
        self.cache.load(rendered)
    }
    fn compiled_cache_key(&self, kernel: &Self::Kernel) -> String {
        kernel.rendered().cache_key.clone()
    }
    fn allocate(&self, elements: usize, dtype: DType) -> Result<Self::Buffer, Self::Error> {
        self.device.allocate_typed(elements, dtype)
    }
    fn create_queue(&self) -> Result<Self::Queue, Self::Error> {
        self.device.create_queue()
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
        if let Some(command) = kernel.launch(queue, buffers)? {
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
        self.cache.len()
    }
}

/// A fully validated pure prefix whose logical intermediates remain device-resident.
pub(crate) struct WebGpuPrefixPlan {
    plan: StaticSchedulePlan<super::RenderedWgsl>,
    renderer: WgslRenderer,
}

impl WebGpuPrefixPlan {
    pub(crate) fn plan_for_outputs(
        device: WebGpuDevice,
        items: &[ScheduleItem],
        retained: &[u64],
        renderer: WgslRenderer,
    ) -> Result<Self, WebGpuError> {
        let adapter = WebGpuStaticAdapter::new(device, renderer.clone());
        Ok(Self {
            plan: StaticSchedulePlan::build(&adapter, items, Some(retained))?,
            renderer,
        })
    }
}

/// A fully validated pure prefix whose logical intermediates remain device-resident.
pub struct PreparedWebGpuPrefix {
    inner: PreparedStaticSchedule<WebGpuStaticAdapter>,
}

impl PreparedWebGpuPrefix {
    pub fn prepare(
        device: WebGpuDevice,
        items: &[ScheduleItem],
        renderer: WgslRenderer,
    ) -> Result<Self, WebGpuError> {
        Ok(Self {
            inner: PreparedStaticSchedule::prepare(
                WebGpuStaticAdapter::new(device, renderer),
                items,
            )?,
        })
    }

    pub(crate) fn from_plan(
        device: WebGpuDevice,
        plan: WebGpuPrefixPlan,
    ) -> Result<Self, WebGpuError> {
        Ok(Self {
            inner: PreparedStaticSchedule::from_plan(
                WebGpuStaticAdapter::new(device, plan.renderer),
                plan.plan,
            )?,
        })
    }

    pub fn cache_len(&self) -> usize {
        self.inner.cache_len()
    }
    /// Ordered logical render keys, including zero-domain identities.
    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.inner.compiled_cache_keys()
    }
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), WebGpuError> {
        self.inner.execute(values)
    }
}
