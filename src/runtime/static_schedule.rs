//! Shared planning and execution for static single-device schedule prefixes.
//!
//! Renderers remain the owners of operation support. This module owns only the
//! buffer residency and side-effect boundary common to the prepared OpenCL,
//! Metal, and WebGPU paths.

use crate::{DType, Operation, ScheduleItem, Shape, TensorData};
use std::collections::{BTreeMap, BTreeSet};

mod sealed {
    pub trait Sealed {}
}

/// One use of a logical device buffer in a renderer-owned pointer ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StaticBufferUse {
    pub(crate) id: u64,
    pub(crate) dtype: DType,
    pub(crate) source_shape: Shape,
    pub(crate) elements: usize,
    pub(crate) bytes: usize,
    pub(crate) alignment: usize,
    pub(crate) role: StaticBufferRole,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StaticBufferRole {
    Input,
    Output,
}

/// Backend-neutral pointer metadata projected from an existing renderer ABI.
pub(crate) struct StaticRenderedBuffer {
    pub(crate) id: u64,
    pub(crate) dtype: DType,
    pub(crate) source_shape: Shape,
    pub(crate) elements: usize,
    pub(crate) mutable: bool,
}

/// Binds the renderer's exact pointer subset/order to schedule-owned physical
/// descriptors. Consumer-local affine addressing remains in the renderer.
pub(crate) fn bind_rendered_buffers<E>(
    item: &ScheduleItem,
    rendered: impl IntoIterator<Item = StaticRenderedBuffer>,
    invalid: impl Fn(String) -> E,
    overflow: impl Fn() -> E,
) -> Result<Vec<StaticBufferUse>, E> {
    let rendered = rendered.into_iter().collect::<Vec<_>>();
    if rendered.is_empty()
        || rendered.last().is_none_or(|buffer| !buffer.mutable)
        || rendered[..rendered.len() - 1]
            .iter()
            .any(|buffer| buffer.mutable)
    {
        return Err(invalid(
            "rendered ABI requires one final mutable output".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    rendered
        .into_iter()
        .map(|abi| {
            if !seen.insert(abi.id) {
                return Err(invalid(format!(
                    "rendered ABI duplicates logical buffer {}",
                    abi.id
                )));
            }
            let desc = if abi.mutable {
                item.primary_output()
            } else {
                &item
                    .ordered_inputs()
                    .iter()
                    .find(|binding| binding.desc.id == abi.id)
                    .ok_or_else(|| {
                        invalid(format!(
                            "rendered ABI input {} is absent from schedule bindings",
                            abi.id
                        ))
                    })?
                    .desc
            };
            let elements = desc.shape.numel().map_err(|_| overflow())?;
            if abi.id != desc.id
                || abi.dtype != desc.dtype
                || abi.source_shape != desc.shape
                || abi.elements != elements
                || abi.mutable == desc.read_only
            {
                return Err(invalid(format!(
                    "rendered ABI descriptor {} mismatches the schedule",
                    abi.id
                )));
            }
            Ok(StaticBufferUse {
                id: abi.id,
                dtype: abi.dtype,
                source_shape: abi.source_shape,
                elements: abi.elements,
                bytes: desc.bytes,
                alignment: desc.alignment,
                role: if abi.mutable {
                    StaticBufferRole::Output
                } else {
                    StaticBufferRole::Input
                },
            })
        })
        .collect()
}

/// One completely rendered item before any native resource work.
pub(crate) struct StaticRendered<R> {
    pub(crate) artifact: R,
    pub(crate) cache_key: String,
    pub(crate) extent: usize,
    pub(crate) buffers: Vec<StaticBufferUse>,
}

/// Canonical physical storage contract for one logical schedule buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StaticBufferPlan {
    pub(crate) dtype: DType,
    pub(crate) source_shape: Shape,
    pub(crate) elements: usize,
    pub(crate) bytes: usize,
    pub(crate) alignment: usize,
    pub(crate) producer: Option<usize>,
}

struct StaticItemPlan<R> {
    rendered: R,
    cache_key: String,
    extent: usize,
    buffer_ids: Vec<u64>,
}

/// Fully validated schedule/render/buffer graph. Constructing this type is pure
/// with respect to native queues, caches, programs, and buffers.
pub(crate) struct StaticSchedulePlan<R> {
    items: Vec<StaticItemPlan<R>>,
    buffers: BTreeMap<u64, StaticBufferPlan>,
    buffer_order: Vec<u64>,
    external_inputs: Vec<u64>,
    retained_outputs: Vec<u64>,
}

/// Coarse backend resource seam. Operation dispatch deliberately remains in
/// each existing renderer rather than being reconstructed here.
pub(crate) trait StaticDeviceAdapter: sealed::Sealed + Sized {
    type Error;
    type Rendered;
    type Kernel;
    type Buffer;
    type Queue;

    fn render(&self, item: &ScheduleItem) -> Result<StaticRendered<Self::Rendered>, Self::Error>;
    fn invalid_binding(reason: String) -> Self::Error;
    fn unsupported(reason: String) -> Self::Error;
    fn overflow() -> Self::Error;
    /// Preserves the backend's established whole-item zero-domain preparation
    /// policy, including compilation, allocation, and queue participation.
    fn prepare_zero_extent(&self) -> bool;
    fn compile(&self, rendered: &Self::Rendered) -> Result<Self::Kernel, Self::Error>;
    fn compiled_cache_key(&self, kernel: &Self::Kernel) -> String;
    fn allocate(&self, elements: usize, dtype: DType) -> Result<Self::Buffer, Self::Error>;
    fn create_queue(&self) -> Result<Self::Queue, Self::Error>;
    fn write(
        &self,
        queue: &Self::Queue,
        buffer: &Self::Buffer,
        bytes: &[u8],
    ) -> Result<(), Self::Error>;
    fn launch_and_wait(
        &self,
        queue: &Self::Queue,
        kernel: &Self::Kernel,
        buffers: &[&Self::Buffer],
    ) -> Result<(), Self::Error>;
    fn read(
        &self,
        queue: &Self::Queue,
        buffer: &Self::Buffer,
        bytes: &mut [u8],
    ) -> Result<(), Self::Error>;
    fn cache_len(&self) -> usize;
}

pub(crate) use sealed::Sealed;

impl<R> StaticSchedulePlan<R> {
    pub(crate) fn build<A>(
        adapter: &A,
        items: &[ScheduleItem],
        retained: Option<&[u64]>,
    ) -> Result<Self, A::Error>
    where
        A: StaticDeviceAdapter<Rendered = R>,
    {
        let mut planned = Vec::with_capacity(items.len());
        let mut buffers = BTreeMap::<u64, StaticBufferPlan>::new();
        let mut buffer_order = Vec::new();
        let mut producers = BTreeMap::<u64, usize>::new();

        validate_prefix::<A>(items)?;

        for (item_index, item) in items.iter().enumerate() {
            if item.boundary.is_some()
                || item.is_effect()
                || !item.quantized_input_bindings.is_empty()
                || !item.outputs.is_single()
            {
                return Err(A::unsupported(
                    "pure prefix item is outside static single-device execution".into(),
                ));
            }
            let nodes = item
                .kernel
                .topological()
                .map_err(|_| A::invalid_binding("cyclic schedule kernel".into()))?;
            if nodes
                .iter()
                .any(|node| matches!(node.operation(), Operation::TensorGuard(_)))
            {
                return Err(A::unsupported(
                    "tensor guard is CPU-interpreter only".into(),
                ));
            }

            let rendered = adapter.render(item)?;
            if rendered.buffers.is_empty()
                || rendered
                    .buffers
                    .last()
                    .is_none_or(|buffer| buffer.role != StaticBufferRole::Output)
                || rendered.buffers[..rendered.buffers.len() - 1]
                    .iter()
                    .any(|buffer| buffer.role != StaticBufferRole::Input)
            {
                return Err(A::invalid_binding(
                    "static item requires read-only inputs and one final writable output".into(),
                ));
            }
            let output = rendered.buffers.last().expect("checked nonempty");
            let expected = item.primary_output();
            if output.id != expected.id
                || output.dtype != expected.dtype
                || output.source_shape != expected.shape
                || output.elements != expected.shape.numel().map_err(|_| A::overflow())?
                || rendered.extent != output.elements
            {
                return Err(A::invalid_binding(
                    "rendered output mismatches scheduled output".into(),
                ));
            }
            if producers.insert(output.id, item_index).is_some() {
                return Err(A::invalid_binding(format!(
                    "duplicate producer for logical buffer {}",
                    output.id
                )));
            }

            let mut item_ids = BTreeSet::new();
            for use_ in &rendered.buffers {
                if !item_ids.insert(use_.id) {
                    return Err(A::invalid_binding(format!(
                        "duplicate logical buffer {} in one ABI",
                        use_.id
                    )));
                }
                let expected_bytes = use_
                    .elements
                    .checked_mul(use_.dtype.itemsize())
                    .ok_or_else(A::overflow)?;
                if use_.bytes != expected_bytes
                    || use_.alignment == 0
                    || !use_.alignment.is_power_of_two()
                {
                    return Err(A::invalid_binding(format!(
                        "invalid physical descriptor for logical buffer {}",
                        use_.id
                    )));
                }
                let candidate = StaticBufferPlan {
                    dtype: use_.dtype,
                    source_shape: use_.source_shape.clone(),
                    elements: use_.elements,
                    bytes: use_.bytes,
                    alignment: use_.alignment,
                    producer: None,
                };
                match buffers.get_mut(&use_.id) {
                    Some(existing)
                        if existing.dtype == candidate.dtype
                            && existing.source_shape == candidate.source_shape
                            && existing.elements == candidate.elements
                            && existing.bytes == candidate.bytes
                            && existing.alignment == candidate.alignment => {}
                    Some(_) => {
                        return Err(A::invalid_binding(format!(
                            "conflicting storage descriptor for logical buffer {}",
                            use_.id
                        )));
                    }
                    None => {
                        buffer_order.push(use_.id);
                        buffers.insert(use_.id, candidate);
                    }
                }
            }
            planned.push(StaticItemPlan {
                rendered: rendered.artifact,
                cache_key: rendered.cache_key,
                extent: rendered.extent,
                buffer_ids: rendered
                    .buffers
                    .into_iter()
                    .map(|buffer| buffer.id)
                    .collect(),
            });
        }

        for (item_index, item) in planned.iter().enumerate() {
            let source_item = &items[item_index];
            for input in &item.buffer_ids[..item.buffer_ids.len() - 1] {
                if let Some(producer_index) = producers.get(input).copied() {
                    if producer_index >= item_index {
                        return Err(A::invalid_binding(format!(
                            "logical buffer {input} is used before it is produced"
                        )));
                    }
                    let producer_id = items[producer_index].id;
                    if !source_item.dependencies.contains(&producer_id) {
                        return Err(A::invalid_binding(format!(
                            "logical buffer {input} producer is absent from dependencies"
                        )));
                    }
                }
            }
        }

        for (id, producer) in &producers {
            let buffer = buffers.get_mut(id).expect("producer ABI was inserted");
            buffer.producer = Some(*producer);
        }
        let retained_outputs = match retained {
            Some(ids) => {
                let mut unique = BTreeSet::new();
                for id in ids {
                    if !unique.insert(*id) {
                        return Err(A::invalid_binding(format!(
                            "requested logical output {id} is duplicated"
                        )));
                    }
                    if !producers.contains_key(id) {
                        return Err(A::invalid_binding(format!(
                            "requested logical output {id} has no prefix producer"
                        )));
                    }
                }
                ids.to_vec()
            }
            // Public prepared-prefix APIs historically materialize every item
            // output into the caller map. Exact internal consumers pass an
            // explicit retained set through `prepare_for_outputs` instead.
            None => items
                .iter()
                .map(|item| item.primary_output().id)
                .collect::<Vec<_>>(),
        };
        if !items.is_empty() && retained_outputs.is_empty() {
            return Err(A::invalid_binding(
                "static prefix has no terminal requested output".into(),
            ));
        }
        for id in &retained_outputs {
            buffers.get(id).ok_or_else(|| {
                A::invalid_binding(format!("requested logical output {id} is absent"))
            })?;
        }
        let external_inputs = buffer_order
            .iter()
            .copied()
            .filter(|id| buffers[id].producer.is_none())
            .collect();

        Ok(Self {
            items: planned,
            buffers,
            buffer_order,
            external_inputs,
            retained_outputs,
        })
    }

    pub(crate) fn compiled_cache_keys(&self) -> Vec<String> {
        self.items
            .iter()
            .filter(|item| item.extent != 0)
            .map(|item| item.cache_key.clone())
            .collect()
    }
}

struct PreparedStaticItem<K> {
    kernel: Option<K>,
    cache_key: Option<String>,
    extent: usize,
    buffer_ids: Vec<u64>,
}

/// Prepared thread-confined resources for a fully validated static plan.
pub(crate) struct PreparedStaticSchedule<A: StaticDeviceAdapter> {
    adapter: A,
    queue: Option<A::Queue>,
    items: Vec<PreparedStaticItem<A::Kernel>>,
    buffers: BTreeMap<u64, A::Buffer>,
    buffer_plans: BTreeMap<u64, StaticBufferPlan>,
    external_inputs: Vec<u64>,
    retained_outputs: Vec<u64>,
    compiled_cache_keys: Vec<String>,
}

impl<A: StaticDeviceAdapter> PreparedStaticSchedule<A> {
    pub(crate) fn prepare(adapter: A, items: &[ScheduleItem]) -> Result<Self, A::Error> {
        let plan = StaticSchedulePlan::build(&adapter, items, None)?;
        Self::from_plan(adapter, plan)
    }

    pub(crate) fn prepare_for_outputs(
        adapter: A,
        items: &[ScheduleItem],
        retained: &[u64],
    ) -> Result<Self, A::Error> {
        let plan = StaticSchedulePlan::build(&adapter, items, Some(retained))?;
        Self::from_plan(adapter, plan)
    }

    pub(crate) fn from_plan(
        adapter: A,
        plan: StaticSchedulePlan<A::Rendered>,
    ) -> Result<Self, A::Error> {
        let prepare_zero_extent = adapter.prepare_zero_extent();
        let mut prepared_items = Vec::with_capacity(plan.items.len());
        for item in plan.items {
            let kernel = if item.extent != 0 || prepare_zero_extent {
                Some(adapter.compile(&item.rendered)?)
            } else {
                None
            };
            let cache_key = kernel
                .as_ref()
                .map(|kernel| adapter.compiled_cache_key(kernel));
            prepared_items.push(PreparedStaticItem {
                kernel,
                cache_key,
                extent: item.extent,
                buffer_ids: item.buffer_ids,
            });
        }
        let allocated_ids = prepared_items
            .iter()
            .filter(|item| item.kernel.is_some())
            .flat_map(|item| item.buffer_ids.iter().copied())
            .collect::<BTreeSet<_>>();
        let mut buffers = BTreeMap::new();
        for id in plan
            .buffer_order
            .iter()
            .filter(|id| allocated_ids.contains(*id))
        {
            let buffer = &plan.buffers[id];
            buffers.insert(*id, adapter.allocate(buffer.elements, buffer.dtype)?);
        }
        let queue = prepared_items
            .iter()
            .any(|item| item.kernel.is_some())
            .then(|| adapter.create_queue())
            .transpose()?;
        let compiled_cache_keys = prepared_items
            .iter()
            .filter_map(|item| item.cache_key.clone())
            .collect();
        Ok(Self {
            adapter,
            queue,
            items: prepared_items,
            buffers,
            buffer_plans: plan.buffers,
            external_inputs: plan.external_inputs,
            retained_outputs: plan.retained_outputs,
            compiled_cache_keys,
        })
    }

    pub(crate) fn cache_len(&self) -> usize {
        self.adapter.cache_len()
    }

    pub(crate) fn compiled_cache_keys(&self) -> Vec<String> {
        self.compiled_cache_keys.clone()
    }

    pub(crate) fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), A::Error> {
        // Complete all host validation and allocation before the first driver call.
        let mut uploads = Vec::with_capacity(self.external_inputs.len());
        for id in self
            .external_inputs
            .iter()
            .filter(|id| self.buffers.contains_key(*id))
        {
            let plan = &self.buffer_plans[id];
            let value = values
                .get(id)
                .ok_or_else(|| A::invalid_binding(format!("missing prefix input {id}")))?;
            if value.dtype() != plan.dtype || value.shape() != &plan.source_shape {
                return Err(A::invalid_binding(format!(
                    "prefix input {id} descriptor mismatch"
                )));
            }
            let bytes = value
                .to_le_bytes()
                .map_err(|_| A::invalid_binding(format!("prefix input {id} bytes")))?;
            if bytes.len() != plan.bytes {
                return Err(A::invalid_binding(format!(
                    "prefix input {id} byte length mismatch"
                )));
            }
            uploads.push((*id, bytes));
        }
        let mut downloads = self
            .retained_outputs
            .iter()
            .map(|id| (*id, vec![0; self.buffer_plans[id].bytes]))
            .collect::<Vec<_>>();

        if let Some(queue) = &self.queue {
            for (id, bytes) in &uploads {
                self.adapter.write(queue, &self.buffers[id], bytes)?;
            }
            for item in &self.items {
                let Some(kernel) = item.kernel.as_ref() else {
                    if item.extent != 0 {
                        return Err(A::invalid_binding(
                            "nonzero item has no compiled kernel".into(),
                        ));
                    }
                    continue;
                };
                let bindings = item
                    .buffer_ids
                    .iter()
                    .map(|id| {
                        self.buffers.get(id).ok_or_else(|| {
                            A::invalid_binding(format!("logical buffer {id} is absent"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.adapter.launch_and_wait(queue, kernel, &bindings)?;
            }
            for (id, bytes) in &mut downloads {
                if !bytes.is_empty() {
                    let buffer = self.buffers.get(id).ok_or_else(|| {
                        A::invalid_binding(format!(
                            "nonempty retained output {id} has no device allocation"
                        ))
                    })?;
                    self.adapter.read(queue, buffer, bytes)?;
                }
            }
        }

        let decoded = downloads
            .into_iter()
            .map(|(id, bytes)| {
                let plan = &self.buffer_plans[&id];
                TensorData::from_le_bytes(plan.source_shape.clone(), plan.dtype, &bytes)
                    .map(|value| (id, value))
                    .map_err(|_| A::invalid_binding(format!("prefix output {id} bytes")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (id, value) in decoded {
            values.insert(id, value);
        }
        Ok(())
    }
}

fn validate_prefix<A: StaticDeviceAdapter>(items: &[ScheduleItem]) -> Result<(), A::Error> {
    let count = items.len() as u64;
    let mut expected_consumers = BTreeMap::<u64, Vec<u64>>::new();
    for (position, item) in items.iter().enumerate() {
        if item.id != position as u64
            || item.dependencies.windows(2).any(|pair| pair[0] >= pair[1])
            || item
                .dependencies
                .iter()
                .any(|dependency| *dependency >= item.id)
        {
            return Err(A::invalid_binding(
                "static prefix item IDs or dependencies are not canonical".into(),
            ));
        }
        for dependency in &item.dependencies {
            expected_consumers
                .entry(*dependency)
                .or_default()
                .push(item.id);
        }
        for desc in item.inputs.iter().chain(item.outputs.iter()) {
            crate::schedule::validate_buffer_desc(desc)
                .map_err(|error| A::invalid_binding(error.to_string()))?;
        }
        item.validate_input_bindings()
            .map_err(|error| A::invalid_binding(error.to_string()))?;
        item.kernel
            .validate()
            .map_err(|error| A::invalid_binding(error.to_string()))?;
    }
    for item in items {
        if item.consumers.windows(2).any(|pair| pair[0] >= pair[1])
            || item
                .consumers
                .iter()
                .copied()
                .filter(|consumer| *consumer < count)
                .ne(expected_consumers.remove(&item.id).unwrap_or_default())
        {
            return Err(A::invalid_binding(
                "static prefix consumer edges are not canonical".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape, Storage, schedule_many};
    use std::{cell::RefCell, rc::Rc};

    #[derive(Default)]
    struct Calls {
        compile: usize,
        allocate: usize,
        queue: usize,
        write: usize,
        launch: usize,
        read: usize,
        fail_launch_after: Option<usize>,
        fail_read_after: Option<usize>,
    }

    #[derive(Clone)]
    struct FakeAdapter(Rc<RefCell<Calls>>);
    struct FakeRendered;
    struct FakeKernel;
    struct FakeQueue;
    struct FakeBuffer(RefCell<Vec<u8>>);

    impl Sealed for FakeAdapter {}
    impl StaticDeviceAdapter for FakeAdapter {
        type Error = String;
        type Rendered = FakeRendered;
        type Kernel = FakeKernel;
        type Buffer = FakeBuffer;
        type Queue = FakeQueue;

        fn render(
            &self,
            item: &ScheduleItem,
        ) -> Result<StaticRendered<Self::Rendered>, Self::Error> {
            let mut buffers = item
                .ordered_inputs()
                .iter()
                .map(|binding| fake_use(&binding.desc, false))
                .collect::<Result<Vec<_>, _>>()?;
            buffers.push(fake_use(item.primary_output(), true)?);
            Ok(StaticRendered {
                artifact: FakeRendered,
                cache_key: item.cache_key.to_string(),
                extent: item
                    .primary_output()
                    .shape
                    .numel()
                    .map_err(|_| "overflow".to_owned())?,
                buffers,
            })
        }
        fn invalid_binding(reason: String) -> Self::Error {
            reason
        }
        fn unsupported(reason: String) -> Self::Error {
            reason
        }
        fn overflow() -> Self::Error {
            "overflow".into()
        }
        fn prepare_zero_extent(&self) -> bool {
            false
        }
        fn compile(&self, _: &Self::Rendered) -> Result<Self::Kernel, Self::Error> {
            self.0.borrow_mut().compile += 1;
            Ok(FakeKernel)
        }
        fn compiled_cache_key(&self, _: &Self::Kernel) -> String {
            "fake-compiled".into()
        }
        fn allocate(&self, elements: usize, dtype: DType) -> Result<Self::Buffer, Self::Error> {
            self.0.borrow_mut().allocate += 1;
            Ok(FakeBuffer(RefCell::new(vec![
                0;
                elements * dtype.itemsize()
            ])))
        }
        fn create_queue(&self) -> Result<Self::Queue, Self::Error> {
            self.0.borrow_mut().queue += 1;
            Ok(FakeQueue)
        }
        fn write(
            &self,
            _: &Self::Queue,
            buffer: &Self::Buffer,
            bytes: &[u8],
        ) -> Result<(), Self::Error> {
            self.0.borrow_mut().write += 1;
            buffer.0.borrow_mut().copy_from_slice(bytes);
            Ok(())
        }
        fn launch_and_wait(
            &self,
            _: &Self::Queue,
            _: &Self::Kernel,
            buffers: &[&Self::Buffer],
        ) -> Result<(), Self::Error> {
            let mut calls = self.0.borrow_mut();
            calls.launch += 1;
            if let Some(remaining) = calls.fail_launch_after.as_mut() {
                if *remaining == 0 {
                    calls.fail_launch_after = None;
                    return Err("injected launch failure".into());
                }
                *remaining -= 1;
            }
            drop(calls);
            let bytes = buffers
                .first()
                .ok_or_else(|| "missing input".to_owned())?
                .0
                .borrow()
                .clone();
            buffers
                .last()
                .ok_or_else(|| "missing output".to_owned())?
                .0
                .borrow_mut()
                .copy_from_slice(&bytes);
            Ok(())
        }
        fn read(
            &self,
            _: &Self::Queue,
            buffer: &Self::Buffer,
            bytes: &mut [u8],
        ) -> Result<(), Self::Error> {
            let mut calls = self.0.borrow_mut();
            calls.read += 1;
            if let Some(remaining) = calls.fail_read_after.as_mut() {
                if *remaining == 0 {
                    calls.fail_read_after = None;
                    return Err("injected read failure".into());
                }
                *remaining -= 1;
            }
            drop(calls);
            bytes.copy_from_slice(&buffer.0.borrow());
            Ok(())
        }
        fn cache_len(&self) -> usize {
            self.0.borrow().compile
        }
    }

    fn fake_use(desc: &crate::BufferDesc, mutable: bool) -> Result<StaticBufferUse, String> {
        Ok(StaticBufferUse {
            id: desc.id,
            dtype: desc.dtype,
            source_shape: desc.shape.clone(),
            elements: desc.shape.numel().map_err(|_| "overflow".to_owned())?,
            bytes: desc.bytes,
            alignment: desc.alignment,
            role: if mutable {
                StaticBufferRole::Output
            } else {
                StaticBufferRole::Input
            },
        })
    }

    fn branched_schedule() -> (crate::Schedule, u64, [u64; 2]) {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::from([2]));
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

    #[test]
    fn shared_executor_uploads_once_keeps_intermediate_and_downloads_requested_once() {
        let (schedule, input, outputs) = branched_schedule();
        assert_eq!(schedule.items.len(), 3);
        let calls = Rc::new(RefCell::new(Calls::default()));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &outputs,
        )
        .unwrap();
        let mut values = BTreeMap::from([(
            input,
            TensorData::from_storage(Shape::from([2]), Storage::F32(vec![2.0, 3.0])).unwrap(),
        )]);
        prepared.execute(&mut values).unwrap();
        let calls = calls.borrow();
        assert_eq!(calls.write, 1);
        assert_eq!(calls.launch, 3);
        assert_eq!(calls.read, 2);
        assert!(outputs.iter().all(|id| values.contains_key(id)));
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn launch_failure_publishes_nothing_and_retry_reuploads_external_values() {
        let (schedule, input, outputs) = branched_schedule();
        let calls = Rc::new(RefCell::new(Calls {
            fail_launch_after: Some(1),
            ..Calls::default()
        }));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &outputs,
        )
        .unwrap();
        let mut values = BTreeMap::from([(
            input,
            TensorData::from_storage(Shape::from([2]), Storage::F32(vec![2.0, 3.0])).unwrap(),
        )]);
        let before = values.clone();
        assert_eq!(
            prepared.execute(&mut values).unwrap_err(),
            "injected launch failure"
        );
        assert_eq!(values, before);
        prepared.execute(&mut values).unwrap();
        let calls = calls.borrow();
        assert_eq!(calls.write, 2);
        assert_eq!(calls.launch, 5);
        assert_eq!(calls.read, 2);
    }

    #[test]
    fn read_failure_after_an_earlier_download_is_atomic_and_retryable() {
        let (schedule, input, outputs) = branched_schedule();
        let calls = Rc::new(RefCell::new(Calls {
            fail_read_after: Some(1),
            ..Calls::default()
        }));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &outputs,
        )
        .unwrap();
        let mut values = BTreeMap::from([(
            input,
            TensorData::from_storage(Shape::from([2]), Storage::F32(vec![2.0, 3.0])).unwrap(),
        )]);
        let before = values.clone();
        assert_eq!(
            prepared.execute(&mut values).unwrap_err(),
            "injected read failure"
        );
        assert_eq!(values, before);
        prepared.execute(&mut values).unwrap();
        let calls = calls.borrow();
        assert_eq!(calls.write, 2);
        assert_eq!(calls.read, 4);
        assert!(outputs.iter().all(|id| values.contains_key(id)));
    }

    #[test]
    fn malformed_prefix_fails_before_compile_allocation_or_queue_creation() {
        let (mut schedule, _, outputs) = branched_schedule();
        schedule.items[0].consumers.clear();
        let calls = Rc::new(RefCell::new(Calls::default()));
        assert!(
            PreparedStaticSchedule::prepare_for_outputs(
                FakeAdapter(calls.clone()),
                &schedule.items,
                &outputs,
            )
            .is_err()
        );
        let calls = calls.borrow();
        assert_eq!((calls.compile, calls.allocate, calls.queue), (0, 0, 0));
    }

    #[test]
    fn duplicate_producer_use_before_produce_and_conflicting_storage_fail_pre_resource() {
        let (schedule, _, outputs) = branched_schedule();

        let mut duplicate = schedule.clone();
        duplicate.items[2].outputs = duplicate.items[1].outputs.clone();

        let mut future = schedule.clone();
        future.items.swap(0, 1);
        for (position, item) in future.items.iter_mut().enumerate() {
            item.id = position as u64;
            item.dependencies.clear();
            item.consumers.clear();
        }
        future.items[1].consumers.push(2);
        future.items[2].dependencies = vec![1];

        let mut conflicting = schedule.clone();
        let shared = conflicting.items[0].primary_output().id;
        let desc = conflicting.items[2]
            .input_bindings
            .iter_mut()
            .find(|binding| binding.desc.id == shared)
            .expect("branch consumes shared producer");
        desc.desc.alignment *= 2;
        conflicting.items[2]
            .inputs
            .iter_mut()
            .find(|input| input.id == shared)
            .expect("shared input is inventoried")
            .alignment *= 2;

        for (name, items, retained) in [
            ("duplicate", duplicate.items, outputs.to_vec()),
            ("future", future.items, outputs.to_vec()),
            ("conflicting", conflicting.items, outputs.to_vec()),
        ] {
            let calls = Rc::new(RefCell::new(Calls::default()));
            assert!(
                PreparedStaticSchedule::prepare_for_outputs(
                    FakeAdapter(calls.clone()),
                    &items,
                    &retained,
                )
                .is_err(),
                "{name}"
            );
            let calls = calls.borrow();
            assert_eq!(
                (calls.compile, calls.allocate, calls.queue),
                (0, 0, 0),
                "{name}"
            );
        }
    }

    #[test]
    fn affine_consumer_view_reuses_the_producer_physical_identity() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 2], DType::F32);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::F32);
        let bias = graph.input_dtype("bias", [2, 2], DType::F32);
        let product = graph.matmul(lhs, rhs).unwrap();
        let transposed = graph.permute(product, vec![1, 0]).unwrap();
        let output = graph.add(transposed, bias).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        assert!(schedule.items.iter().any(|item| {
            item.ordered_inputs()
                .iter()
                .any(|binding| binding.desc.view.is_some())
        }));
        let adapter = FakeAdapter(Rc::new(RefCell::new(Calls::default())));
        let plan =
            StaticSchedulePlan::build(&adapter, &schedule.items, Some(&[output.index() as u64]))
                .unwrap();
        assert_eq!(
            plan.buffers
                .keys()
                .filter(|id| **id == product.index() as u64)
                .count(),
            1
        );
    }

    #[test]
    fn missing_requested_output_fails_before_resource_work() {
        let (schedule, _, _) = branched_schedule();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let error = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &[u64::MAX],
        )
        .err()
        .expect("missing requested output must fail");
        assert!(error.contains("has no prefix producer"));
        let calls = calls.borrow();
        assert_eq!((calls.compile, calls.allocate, calls.queue), (0, 0, 0));
    }

    #[test]
    fn zero_domain_prefix_allocates_no_queue_and_returns_exact_empty_value() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0], DType::F32);
        let output = graph.unary(crate::UnaryOp::Neg, input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &[output.index() as u64],
        )
        .unwrap();
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage(Shape::from([0]), Storage::F32(vec![])).unwrap(),
        )]);
        prepared.execute(&mut values).unwrap();
        let calls = calls.borrow();
        assert_eq!(
            (
                calls.compile,
                calls.allocate,
                calls.queue,
                calls.write,
                calls.launch,
                calls.read
            ),
            (0, 0, 0, 0, 0, 0)
        );
        assert_eq!(values[&(output.index() as u64)].shape(), &Shape::from([0]));
    }
}
