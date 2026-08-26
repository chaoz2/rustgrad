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
    queue: Option<MetalCommandQueue>,
    cache: MetalCache,
    items: Vec<PreparedMetalItem>,
}

/// Fully rendered pure prefix before any Metal resource is created.
pub struct MetalPrefixPlan {
    items: Vec<PlannedMetalItem>,
}

enum PlannedMetalItem {
    Kernel(Box<super::RenderedMetal>),
    /// A validated pure item whose result has no logical storage. It retains
    /// descriptor identity but never needs a pipeline, buffer, or command.
    ZeroDomain(Box<ScheduleItem>),
}

enum PreparedMetalItem {
    Kernel(Rc<MetalPipeline>, Vec<MetalBuffer>),
    ZeroDomain(Box<ScheduleItem>),
}

impl MetalPrefixPlan {
    /// Performs deterministic renderer and ABI validation only.
    pub fn plan(items: &[ScheduleItem], renderer: MetalRenderer) -> Result<Self, MetalError> {
        let mut planned = Vec::with_capacity(items.len());
        for item in items {
            if item.boundary.is_some()
                || item.is_effect()
                || !item.quantized_input_bindings.is_empty()
                || !item.outputs.is_single()
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
            if item
                .primary_output()
                .shape
                .numel()
                .map_err(|_| MetalError::Overflow)?
                == 0
            {
                planned.push(PlannedMetalItem::ZeroDomain(Box::new(item.clone())));
            } else {
                planned.push(PlannedMetalItem::Kernel(Box::new(rendered)));
            }
        }
        Ok(Self { items: planned })
    }
    pub fn cache_keys(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| match item {
                PlannedMetalItem::Kernel(rendered) => Some(rendered.cache_key.clone()),
                PlannedMetalItem::ZeroDomain(_) => None,
            })
            .collect()
    }
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
    /// Allocates and compiles a previously validated plan.
    pub fn from_plan(device: MetalDevice, plan: MetalPrefixPlan) -> Result<Self, MetalError> {
        let cache = device.cache();
        let queue = if plan
            .items
            .iter()
            .any(|item| matches!(item, PlannedMetalItem::Kernel(..)))
        {
            Some(device.create_queue()?)
        } else {
            None
        };
        let mut prepared = Vec::with_capacity(plan.items.len());
        for item in plan.items {
            match item {
                PlannedMetalItem::Kernel(rendered) => {
                    let pipeline = cache.load(&rendered)?;
                    let buffers = pipeline
                        .rendered()
                        .buffers
                        .iter()
                        .map(|abi| device.allocate_typed(abi.elements, abi.dtype))
                        .collect::<Result<Vec<_>, _>>()?;
                    prepared.push(PreparedMetalItem::Kernel(pipeline, buffers));
                }
                PlannedMetalItem::ZeroDomain(item) => {
                    prepared.push(PreparedMetalItem::ZeroDomain(item));
                }
            }
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
            .filter_map(|item| match item {
                PreparedMetalItem::Kernel(pipeline, _) => {
                    Some(pipeline.rendered().cache_key.clone())
                }
                PreparedMetalItem::ZeroDomain(_) => None,
            })
            .collect()
    }
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), MetalError> {
        for item in &self.items {
            let (pipeline, buffers) = match item {
                PreparedMetalItem::Kernel(pipeline, buffers) => (pipeline, buffers),
                PreparedMetalItem::ZeroDomain(item) => {
                    values.insert(
                        item.primary_output().id,
                        TensorData::zeros_with_dtype(
                            item.primary_output().shape.clone(),
                            item.primary_output().dtype,
                        )
                        .map_err(|_| MetalError::InvalidBinding("zero-domain output".into()))?,
                    );
                    continue;
                }
            };
            let queue = self.queue.as_ref().ok_or_else(|| {
                MetalError::InvalidBinding("kernel prefix has no command queue".into())
            })?;
            for (abi, buffer) in pipeline.rendered().buffers.iter().zip(buffers) {
                if !abi.mutable {
                    let value = values.get(&abi.id).ok_or_else(|| {
                        MetalError::InvalidBinding(format!("missing prefix input {}", abi.id))
                    })?;
                    queue.write(
                        buffer,
                        0,
                        &value
                            .to_le_bytes()
                            .map_err(|_| MetalError::InvalidBinding("input bytes".into()))?,
                    )?;
                }
            }
            if let Some(command) = pipeline.launch(queue, &buffers.iter().collect::<Vec<_>>(), 1)? {
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
            queue.read(
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
