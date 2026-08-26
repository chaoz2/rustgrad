//! Retained, nonexecuting OpenCL preparation for a pure schedule prefix.
use super::{
    OpenClBuffer, OpenClCache, OpenClContext, OpenClError, OpenClKernel, OpenClQueue,
    OpenClRenderer,
};
use crate::{ScheduleItem, TensorData};
use std::{collections::BTreeMap, rc::Rc};

pub struct PreparedOpenClPrefix {
    queue: OpenClQueue,
    cache: OpenClCache,
    items: Vec<(ScheduleItem, Rc<OpenClKernel>, Vec<OpenClBuffer>)>,
}
impl PreparedOpenClPrefix {
    pub fn prepare(
        context: OpenClContext,
        items: &[ScheduleItem],
        renderer: OpenClRenderer,
    ) -> Result<Self, OpenClError> {
        if items
            .iter()
            .any(|item| matches!(item.kernel.kind(), crate::UOpKind::TensorGuard))
        {
            return Err(OpenClError::Unsupported(
                "tensor guard is CPU-interpreter only".into(),
            ));
        }
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
            let buffers = kernel
                .rendered()
                .buffers
                .iter()
                .map(|abi| context.allocate_typed(abi.elements, abi.dtype))
                .collect::<Result<Vec<_>, _>>()?;
            prepared.push((item.clone(), kernel, buffers));
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
    /// Stable rendered-kernel identities retained by this prepared prefix.
    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.items
            .iter()
            .map(|(_, kernel, _)| kernel.cache_key().to_owned())
            .collect()
    }
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), OpenClError> {
        for (item, kernel, buffers) in &self.items {
            for (abi, buffer) in kernel.rendered().buffers.iter().zip(buffers) {
                if !abi.mutable {
                    let value = values.get(&abi.id).ok_or_else(|| {
                        OpenClError::InvalidBinding(format!("missing prefix input {}", abi.id))
                    })?;
                    self.queue.write(
                        buffer,
                        0,
                        &value
                            .to_le_bytes()
                            .map_err(|_| OpenClError::InvalidBinding("input bytes".into()))?,
                    )?;
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryOp, DType, Graph, Scalar, Shape, Storage, TensorData, schedule};
    use std::sync::Arc;

    #[test]
    fn tensor_guard_is_rejected_before_opencl_queue_or_cache_work() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let mut graph = Graph::new();
        let input = graph.constant(
            TensorData::from_scalars([2], DType::F32, [Scalar::F(1.0), Scalar::F(1.0)])
                .unwrap(),
        );
        let guard = graph.tensor_guard_distribution(input, 0).unwrap();
        let items = schedule(&graph, guard).unwrap().items;
        assert!(matches!(
            PreparedOpenClPrefix::prepare(context, &items, OpenClRenderer::default()),
            Err(OpenClError::Unsupported(reason)) if reason.contains("tensor guard")
        ));
        assert!(mock.calls().is_empty());
    }

    #[test]
    fn retained_prefix_preflights_then_executes_the_registered_semantic_kernel() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let mut graph = Graph::new();
        let left = graph.input_dtype("left", [2], DType::F32);
        let right = graph.input_dtype("right", [2], DType::F32);
        let output = graph.binary(BinaryOp::Add, left, right).unwrap();
        let items = schedule(&graph, output).unwrap().items;
        let prefix =
            PreparedOpenClPrefix::prepare(context, &items, OpenClRenderer::default()).unwrap();
        let keys = prefix.kernel_cache_keys();
        assert_eq!(keys, prefix.kernel_cache_keys());
        assert!(
            mock.calls()
                .iter()
                .all(|call| !call.contains("kernel_launch"))
        );

        let mut values = BTreeMap::from([
            (
                left.index() as u64,
                TensorData::from_storage(Shape::from([2]), Storage::F32(vec![1.0, 2.0])).unwrap(),
            ),
            (
                right.index() as u64,
                TensorData::from_storage(Shape::from([2]), Storage::F32(vec![3.0, 4.0])).unwrap(),
            ),
        ]);
        prefix.execute(&mut values).unwrap();
        assert!(
            mock.calls()
                .iter()
                .any(|call| call.contains("kernel_launch"))
        );
        assert_eq!(
            values[&(output.index() as u64)].storage(),
            &Storage::F32(vec![4.0, 6.0])
        );
    }

    #[test]
    fn retained_zero_domain_prefix_never_submits() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0], DType::F32);
        let output = graph.unary(crate::UnaryOp::Neg, input).unwrap();
        let prefix = PreparedOpenClPrefix::prepare(
            context,
            &schedule(&graph, output).unwrap().items,
            OpenClRenderer::default(),
        )
        .unwrap();
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage(Shape::from([0]), Storage::F32(vec![])).unwrap(),
        )]);
        prefix.execute(&mut values).unwrap();
        assert!(
            mock.calls()
                .iter()
                .all(|call| !call.contains("kernel_launch"))
        );
    }
}
