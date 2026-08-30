//! Retained, nonexecuting WebGPU preparation for a pure schedule prefix.
use super::{
    WebGpuBuffer, WebGpuCache, WebGpuDevice, WebGpuError, WebGpuPipeline, WebGpuQueue, WgslRenderer,
};
use crate::{ScheduleItem, TensorData};
use std::{collections::BTreeMap, rc::Rc};

/// A fully validated, compiled, and transient-buffer-allocated pure prefix.
/// Preparing it never submits work; execution uses the retained WebGPU
/// semantic pipeline rather than `CpuBackend`.
pub struct PreparedWebGpuPrefix {
    queue: WebGpuQueue,
    cache: WebGpuCache,
    items: Vec<(ScheduleItem, Rc<WebGpuPipeline>, Vec<WebGpuBuffer>)>,
}

impl PreparedWebGpuPrefix {
    /// Validates and prepares every static pure item before any submission.
    pub fn prepare(
        device: WebGpuDevice,
        items: &[ScheduleItem],
        renderer: WgslRenderer,
    ) -> Result<Self, WebGpuError> {
        if items
            .iter()
            .any(|item| matches!(item.kernel.operation(), crate::Operation::TensorGuard(_)))
        {
            return Err(WebGpuError::Unsupported(
                "tensor guard is CPU-interpreter only".into(),
            ));
        }
        let queue = device.create_queue()?;
        let cache = device.cache();
        let mut prepared = Vec::with_capacity(items.len());
        for item in items {
            if item.boundary.is_some()
                || item.is_effect()
                || !item.quantized_input_bindings.is_empty()
            {
                return Err(WebGpuError::Unsupported(
                    "pure prefix item is outside WebGPU static execution".into(),
                ));
            }
            let rendered = renderer.render(&item.kernel)?;
            rendered.validate_schedule_bindings(item.ordered_inputs())?;
            if rendered.transaction.is_some() {
                return Err(WebGpuError::Unsupported(
                    "guarded WebGPU prefixes require a staged candidate ABI".into(),
                ));
            }
            let pipeline = cache.load(&rendered)?;
            let buffers = pipeline
                .rendered()
                .buffers
                .iter()
                .map(|abi| device.allocate_typed(abi.elements, abi.dtype))
                .collect::<Result<Vec<_>, _>>()?;
            prepared.push((item.clone(), pipeline, buffers));
        }
        Ok(Self {
            queue,
            cache,
            items: prepared,
        })
    }

    /// The cache remains retained to make preparation identity observable in
    /// tests without exposing native objects.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Ordered logical render keys; they exclude resource identity and bytes.
    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.items
            .iter()
            .map(|(_, pipeline, _)| pipeline.rendered().cache_key.clone())
            .collect()
    }

    /// Executes retained prepared plans into detached typed values.
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), WebGpuError> {
        for (_item, pipeline, buffers) in &self.items {
            for (abi, buffer) in pipeline.rendered().buffers.iter().zip(buffers) {
                if !abi.mutable {
                    let value = values.get(&abi.id).ok_or_else(|| {
                        WebGpuError::InvalidBinding(format!("missing prefix input {}", abi.id))
                    })?;
                    self.queue.write(
                        buffer,
                        0,
                        &value
                            .to_le_bytes()
                            .map_err(|_| WebGpuError::InvalidBinding("input bytes".into()))?,
                    )?;
                }
            }
            if let Some(command) =
                pipeline.launch(&self.queue, &buffers.iter().collect::<Vec<_>>())?
            {
                command.collect()?;
            }
            let output = pipeline
                .rendered()
                .buffers
                .last()
                .ok_or_else(|| WebGpuError::InvalidBinding("missing output".into()))?;
            let mut bytes = vec![
                0;
                output
                    .elements
                    .checked_mul(output.dtype.itemsize())
                    .ok_or(WebGpuError::Overflow)?
            ];
            self.queue.read(
                buffers
                    .last()
                    .ok_or_else(|| WebGpuError::InvalidBinding("missing output buffer".into()))?,
                0,
                &mut bytes,
            )?;
            values.insert(
                output.id,
                TensorData::from_le_bytes(output.source_shape.clone(), output.dtype, &bytes)
                    .map_err(|_| WebGpuError::InvalidBinding("output bytes".into()))?,
            );
        }
        Ok(())
    }
}
