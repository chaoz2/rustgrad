//! Retained, nonexecuting OpenCL preparation for a pure schedule prefix.
use super::{
    OpenClBuffer, OpenClCache, OpenClContext, OpenClError, OpenClKernel, OpenClQueue,
    OpenClRenderer,
};
use crate::{ScheduleItem, TensorData};
use std::{collections::BTreeMap, rc::Rc};

pub struct PreparedOpenClPrefix {
    context: OpenClContext,
    queue: OpenClQueue,
    cache: OpenClCache,
    items: Vec<(ScheduleItem, Rc<OpenClKernel>)>,
}
impl PreparedOpenClPrefix {
    pub fn prepare(
        context: OpenClContext,
        items: &[ScheduleItem],
        renderer: OpenClRenderer,
    ) -> Result<Self, OpenClError> {
        let queue = context.create_queue()?;
        let cache = context.cache();
        let mut prepared = Vec::with_capacity(items.len());
        for item in items {
            if item.boundary.is_some()
                || item.is_effect()
                || !item.quantized_input_bindings.is_empty()
            {
                return Err(OpenClError::Unsupported(
                    "pure prefix item is outside OpenCL static execution".into(),
                ));
            }
            let rendered = renderer.render(&item.kernel)?;
            rendered.validate_schedule_bindings(item.ordered_inputs())?;
            if rendered.transaction.is_some() {
                return Err(OpenClError::Unsupported(
                    "guarded OpenCL prefixes require a staged candidate ABI".into(),
                ));
            }
            let kernel = cache.load(&rendered, "-cl-std=CL1.2", renderer.local_size)?;
            prepared.push((item.clone(), kernel));
        }
        Ok(Self {
            context,
            queue,
            cache,
            items: prepared,
        })
    }
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), OpenClError> {
        for (item, kernel) in &self.items {
            let mut buffers = Vec::<OpenClBuffer>::new();
            for abi in &kernel.rendered().buffers {
                let buffer = self.context.allocate_typed(abi.elements, abi.dtype)?;
                if !abi.mutable {
                    let value = values.get(&abi.id).ok_or_else(|| {
                        OpenClError::InvalidBinding(format!("missing prefix input {}", abi.id))
                    })?;
                    self.queue.write(
                        &buffer,
                        0,
                        &value
                            .to_le_bytes()
                            .map_err(|_| OpenClError::InvalidBinding("input bytes".into()))?,
                    )?;
                }
                buffers.push(buffer);
            }
            if let Some(event) = kernel.launch(&self.queue, &buffers.iter().collect::<Vec<_>>())? {
                event.wait()?;
            }
            let output = kernel
                .rendered()
                .buffers
                .last()
                .ok_or_else(|| OpenClError::InvalidBinding("missing output".into()))?;
            let mut bytes = vec![
                0;
                output
                    .elements
                    .checked_mul(output.dtype.itemsize())
                    .ok_or(OpenClError::Overflow)?
            ];
            self.queue.read(
                buffers
                    .last()
                    .ok_or_else(|| OpenClError::InvalidBinding("missing output buffer".into()))?,
                0,
                &mut bytes,
            )?;
            values.insert(
                output.id,
                TensorData::from_le_bytes(output.source_shape.clone(), output.dtype, &bytes)
                    .map_err(|_| OpenClError::InvalidBinding("output bytes".into()))?,
            );
            let _ = item;
        }
        Ok(())
    }
}
