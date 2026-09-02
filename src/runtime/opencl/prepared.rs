//! Retained, nonexecuting OpenCL preparation for a pure schedule prefix.
use super::{
    OpenClBuffer, OpenClCache, OpenClContext, OpenClError, OpenClKernel, OpenClQueue,
    OpenClRenderer,
};
use crate::{ScheduleItem, TensorData};
use std::{collections::BTreeMap, rc::Rc};

use crate::runtime::static_schedule::{
    PreparedStaticSchedule, Sealed, StaticBufferAllocation, StaticDeviceAdapter, StaticPlanAdapter,
    StaticRendered, StaticRenderedBuffer, StaticSchedulePlan, bind_rendered_buffers,
};

struct OpenClStaticAdapter {
    context: OpenClContext,
    renderer: OpenClRenderer,
    cache: OpenClCache,
}

impl OpenClStaticAdapter {
    fn new(context: OpenClContext, renderer: OpenClRenderer) -> Self {
        let cache = context.cache();
        Self {
            context,
            renderer,
            cache,
        }
    }
}

impl Sealed for OpenClStaticAdapter {}

impl StaticPlanAdapter for OpenClStaticAdapter {
    type Error = OpenClError;
    type Rendered = super::RenderedOpenCl;

    fn render(&self, item: &ScheduleItem) -> Result<StaticRendered<Self::Rendered>, Self::Error> {
        let rendered = self.renderer.render(&item.kernel)?;
        rendered.validate_schedule_bindings(item.ordered_inputs())?;
        if rendered.transaction.is_some() {
            return Err(OpenClError::Unsupported(
                "guarded OpenCL prefixes require a staged candidate ABI".into(),
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
            OpenClError::InvalidBinding,
            || OpenClError::Overflow,
        )?;
        Ok(StaticRendered {
            cache_key: rendered.cache_key.clone(),
            extent: rendered.extent,
            buffers,
            artifact: rendered,
        })
    }

    fn invalid_binding(reason: String) -> Self::Error {
        OpenClError::InvalidBinding(reason)
    }
    fn unsupported(reason: String) -> Self::Error {
        OpenClError::Unsupported(reason)
    }
    fn overflow() -> Self::Error {
        OpenClError::Overflow
    }
}

impl StaticDeviceAdapter for OpenClStaticAdapter {
    type Kernel = Rc<OpenClKernel>;
    type Buffer = OpenClBuffer;
    type Queue = OpenClQueue;

    fn prepare_zero_extent(&self) -> bool {
        true
    }
    fn compile(&self, rendered: &Self::Rendered) -> Result<Self::Kernel, Self::Error> {
        self.cache
            .load(rendered, "-cl-std=CL1.2", self.renderer.local_size)
    }
    fn compiled_cache_key(&self, kernel: &Self::Kernel) -> String {
        kernel.cache_key().to_owned()
    }
    fn allocate(&self, request: StaticBufferAllocation) -> Result<Self::Buffer, Self::Error> {
        self.context.allocate_static(request)
    }
    fn create_queue(&self) -> Result<Self::Queue, Self::Error> {
        self.context.create_queue()
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
        if let Some(event) = kernel.launch(queue, buffers)? {
            event.wait()?;
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
pub(crate) struct OpenClPrefixPlan {
    plan: StaticSchedulePlan<super::RenderedOpenCl>,
    renderer: OpenClRenderer,
}

impl OpenClPrefixPlan {
    pub(crate) fn plan_for_outputs(
        context: OpenClContext,
        items: &[ScheduleItem],
        retained: &[u64],
        renderer: OpenClRenderer,
    ) -> Result<Self, OpenClError> {
        let adapter = OpenClStaticAdapter::new(context, renderer);
        Ok(Self {
            plan: StaticSchedulePlan::build(&adapter, items, Some(retained))?,
            renderer,
        })
    }
}

/// A fully validated pure prefix whose logical intermediates remain device-resident.
pub struct PreparedOpenClPrefix {
    inner: PreparedStaticSchedule<OpenClStaticAdapter>,
}

impl PreparedOpenClPrefix {
    pub fn prepare(
        context: OpenClContext,
        items: &[ScheduleItem],
        renderer: OpenClRenderer,
    ) -> Result<Self, OpenClError> {
        Ok(Self {
            inner: PreparedStaticSchedule::prepare(
                OpenClStaticAdapter::new(context, renderer),
                items,
            )?,
        })
    }

    pub(crate) fn from_plan(
        context: OpenClContext,
        plan: OpenClPrefixPlan,
    ) -> Result<Self, OpenClError> {
        Ok(Self {
            inner: PreparedStaticSchedule::from_plan(
                OpenClStaticAdapter::new(context, plan.renderer),
                plan.plan,
            )?,
        })
    }

    pub fn cache_len(&self) -> usize {
        self.inner.cache_len()
    }
    /// Stable rendered identities; zero-domain entries remain observable.
    pub fn kernel_cache_keys(&self) -> Vec<String> {
        self.inner.compiled_cache_keys()
    }
    pub fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), OpenClError> {
        self.inner.execute(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BinaryOp, DType, Graph, Scalar, Shape, Storage, TensorData, schedule, schedule_many,
    };
    use std::sync::Arc;

    #[test]
    fn tensor_guard_is_rejected_before_opencl_queue_or_cache_work() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let calls_before_prepare = mock.calls().len();
        let mut graph = Graph::new();
        let input = graph.constant(
            TensorData::from_scalars([2], DType::F32, [Scalar::F(1.0), Scalar::F(1.0)]).unwrap(),
        );
        let guard = graph.tensor_guard_distribution(input, 0).unwrap();
        let items = schedule(&graph, guard).unwrap().items;
        assert!(matches!(
            PreparedOpenClPrefix::prepare(context.clone(), &items, OpenClRenderer::default()),
            Err(OpenClError::Unsupported(reason)) if reason.contains("tensor guard")
        ));
        assert_eq!(mock.calls().len(), calls_before_prepare);
    }

    #[test]
    fn retained_prefix_executes_the_registered_semantic_kernel() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let mut graph = Graph::new();
        let left = graph.input_dtype("left", [2], DType::F32);
        let right = graph.input_dtype("right", [2], DType::F32);
        let output = graph.binary(BinaryOp::Add, left, right).unwrap();
        let items = schedule(&graph, output).unwrap().items;
        let prefix =
            PreparedOpenClPrefix::prepare(context, &items, OpenClRenderer::default()).unwrap();
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
        assert_eq!(
            values[&(output.index() as u64)].storage(),
            &Storage::F32(vec![4.0, 6.0])
        );
    }

    #[test]
    fn public_branched_prefix_keeps_shared_values_on_device_then_returns_all_outputs() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let shared = graph.square(input).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(shared, one).unwrap();
        let right = graph.mul(shared, one).unwrap();
        let schedule = schedule_many(&graph, &[left, right]).unwrap();
        assert_eq!(schedule.items.len(), 3);
        let prefix =
            PreparedOpenClPrefix::prepare(context, &schedule.items, OpenClRenderer::default())
                .unwrap();
        let before = mock.calls();
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage(Shape::from([2]), Storage::F32(vec![2.0, 3.0])).unwrap(),
        )]);
        prefix.execute(&mut values).unwrap();
        let calls = &mock.calls()[before.len()..];
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("write:"))
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("read:"))
                .count(),
            3
        );
        assert_eq!(
            values[&(shared.index() as u64)].storage(),
            &Storage::F32(vec![4.0, 9.0])
        );
        assert_eq!(
            values[&(left.index() as u64)].storage(),
            &Storage::F32(vec![5.0, 10.0])
        );
        assert_eq!(
            values[&(right.index() as u64)].storage(),
            &Storage::F32(vec![4.0, 9.0])
        );
    }

    #[test]
    fn retained_static_position_chain_stays_device_resident() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let producer = graph.square(input).unwrap();
        let placed = graph
            .scatter_positions(producer, Shape::from([5]), vec![4], vec![-2])
            .unwrap();
        let output = graph.square(placed).unwrap();
        let schedule = schedule(&graph, output).unwrap();
        assert_eq!(schedule.items.len(), 3);
        let plan = OpenClPrefixPlan::plan_for_outputs(
            context.clone(),
            &schedule.items,
            &[output.index() as u64],
            OpenClRenderer::default(),
        )
        .unwrap();
        let prefix = PreparedOpenClPrefix::from_plan(context, plan).unwrap();
        let before = mock.calls();
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage(Shape::from([2]), Storage::F32(vec![2.0, -3.0])).unwrap(),
        )]);
        prefix.execute(&mut values).unwrap();
        let calls = &mock.calls()[before.len()..];
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("write:"))
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("read:"))
                .count(),
            1
        );
        assert!(!values.contains_key(&(producer.index() as u64)));
        assert!(!values.contains_key(&(placed.index() as u64)));
        assert_eq!(
            values[&(output.index() as u64)].storage(),
            &Storage::F32(vec![0.0, 0.0, 81.0, 0.0, 16.0])
        );
    }

    #[test]
    fn unsupported_static_position_preflights_before_opencl_resource_work() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());
        let owner = context.owner_id();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1], DType::U64);
        let output = graph
            .scatter_positions(input, Shape::from([2]), vec![1], vec![1])
            .unwrap();
        let items = schedule(&graph, output).unwrap().items;
        let before = mock.calls();
        assert!(matches!(
            PreparedOpenClPrefix::prepare(context, &items, OpenClRenderer::default()),
            Err(OpenClError::Unsupported(reason)) if reason.contains("64-bit integer")
        ));
        let calls = mock.calls();
        assert_eq!(
            &calls[before.len()..],
            &[format!("context_release:{owner}")]
        );
        let queue_release = calls
            .iter()
            .position(|call| call == &format!("queue_release:{owner}"))
            .unwrap();
        let context_release = calls
            .iter()
            .position(|call| call == &format!("context_release:{owner}"))
            .unwrap();
        assert!(queue_release < context_release);
    }

    #[test]
    fn zero_domain_preparation_preserves_cache_and_empty_reduction_abi() {
        let mock = Arc::new(super::super::tests::MockDispatch::default());
        let (context, _) = super::super::tests::setup(mock.clone());

        let mut empty_graph = Graph::new();
        let empty_input = empty_graph.input_dtype("empty", [0], DType::F32);
        let empty_output = empty_graph.unary(crate::UnaryOp::Neg, empty_input).unwrap();
        let empty_items = schedule(&empty_graph, empty_output).unwrap().items;
        let rendered_empty_key = OpenClRenderer::default()
            .render(&empty_items[0].kernel)
            .unwrap()
            .cache_key;
        let empty =
            PreparedOpenClPrefix::prepare(context.clone(), &empty_items, OpenClRenderer::default())
                .unwrap();
        assert_eq!(empty.cache_len(), 1);
        assert_eq!(empty.kernel_cache_keys().len(), 1);
        assert_ne!(empty.kernel_cache_keys()[0], rendered_empty_key);
        let mut empty_values = BTreeMap::from([(
            empty_input.index() as u64,
            TensorData::from_storage(Shape::from([0]), Storage::F32(vec![])).unwrap(),
        )]);
        empty.execute(&mut empty_values).unwrap();
        assert!(
            mock.calls()
                .iter()
                .all(|call| !call.contains("kernel_launch"))
        );
        assert_eq!(
            empty_values[&(empty_output.index() as u64)].shape(),
            &Shape::from([0])
        );

        let mut reduction_graph = Graph::new();
        let reduction_input = reduction_graph.input_dtype("input", [0, 2], DType::I32);
        let reduction_output = reduction_graph
            .reduce(
                reduction_input,
                crate::ReduceKind::Sum,
                Some(vec![0]),
                false,
            )
            .unwrap();
        let reduction_items = schedule(&reduction_graph, reduction_output).unwrap().items;
        let rendered = OpenClRenderer::default()
            .render(&reduction_items[0].kernel)
            .unwrap();
        assert_eq!(rendered.buffers.len(), 1);
        assert!(rendered.buffers[0].mutable);
        let reduction =
            PreparedOpenClPrefix::prepare(context, &reduction_items, OpenClRenderer::default())
                .unwrap();
        assert_eq!(reduction.kernel_cache_keys().len(), 1);
        let mut reduction_values = BTreeMap::new();
        reduction.execute(&mut reduction_values).unwrap();
        assert_eq!(
            reduction_values[&(reduction_output.index() as u64)].storage(),
            &Storage::I32(vec![0, 0])
        );
    }
}
