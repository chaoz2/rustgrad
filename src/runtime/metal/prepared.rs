//! Retained, nonexecuting Metal preparation for a pure schedule prefix.
use super::{
    MetalBuffer, MetalCache, MetalCommandQueue, MetalDevice, MetalError, MetalPipeline,
    MetalRenderer,
};
use crate::{ScheduleItem, TensorData};
use std::{collections::BTreeMap, rc::Rc};

/// A fully rendered, validated, compiled, and allocated pure prefix. Preparing
/// it has no command submission side effect; execution retains the semantic
/// Metal pipeline rather than consulting the CPU backend.
pub struct PreparedMetalPrefix {
    queue: MetalCommandQueue,
    cache: MetalCache,
    items: Vec<(ScheduleItem, Rc<MetalPipeline>, Vec<MetalBuffer>)>,
}

impl PreparedMetalPrefix {
    pub fn prepare(
        device: MetalDevice,
        items: &[ScheduleItem],
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let queue = device.create_queue()?;
        let cache = device.cache();
        let mut prepared = Vec::with_capacity(items.len());
        for item in items {
            if item.boundary.is_some()
                || item.is_effect()
                || !item.quantized_input_bindings.is_empty()
            {
                return Err(MetalError::Unsupported(
                    "pure prefix item is outside Metal static execution".into(),
                ));
            }
            let rendered = renderer.render(&item.kernel)?;
            rendered.validate_schedule_bindings(item.ordered_inputs())?;
            if rendered.transaction.is_some() {
                return Err(MetalError::Unsupported(
                    "guarded Metal prefixes require a staged candidate ABI".into(),
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
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.items
            .iter()
            .map(|(_, p, _)| p.rendered().cache_key.clone())
            .collect()
    }
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), MetalError> {
        for (_item, pipeline, buffers) in &self.items {
            for (abi, buffer) in pipeline.rendered().buffers.iter().zip(buffers) {
                if !abi.mutable {
                    let value = values.get(&abi.id).ok_or_else(|| {
                        MetalError::InvalidBinding(format!("missing prefix input {}", abi.id))
                    })?;
                    self.queue.write(
                        buffer,
                        0,
                        &value
                            .to_le_bytes()
                            .map_err(|_| MetalError::InvalidBinding("input bytes".into()))?,
                    )?;
                }
            }
            if let Some(command) =
                pipeline.launch(&self.queue, &buffers.iter().collect::<Vec<_>>(), 1)?
            {
                command.collect()?;
            }
            let output = pipeline
                .rendered()
                .buffers
                .last()
                .ok_or_else(|| MetalError::InvalidBinding("missing output".into()))?;
            let mut bytes = vec![
                0;
                output
                    .elements
                    .checked_mul(output.dtype.itemsize())
                    .ok_or(MetalError::Overflow)?
            ];
            self.queue.read(
                buffers
                    .last()
                    .ok_or_else(|| MetalError::InvalidBinding("missing output buffer".into()))?,
                0,
                &mut bytes,
            )?;
            values.insert(
                output.id,
                TensorData::from_le_bytes(output.source_shape.clone(), output.dtype, &bytes)
                    .map_err(|_| MetalError::InvalidBinding("output bytes".into()))?,
            );
        }
        Ok(())
    }
}
