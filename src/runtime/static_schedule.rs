//! Shared planning and execution for static single-device schedule prefixes.
//!
//! Renderers remain the owners of operation support. This module owns only the
//! buffer residency and side-effect boundary common to the prepared OpenCL,
//! Metal, WebGPU, and fixed-schema CUDA graph paths.

use crate::{
    DType, Operation, ScheduleInputBinding, ScheduleItem, Shape, TensorData,
    memory_plan::{ExactSlotPolicy, ExactSlotRequest, assign_exact_slots},
};
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
    Output(usize),
}

/// Authenticated logical storage and physical launch domains for one item.
/// Most kernels launch one work item per output element; serial PrefixScan
/// and coupled Sort launch one work item per `(row, inner)` lane while
/// retaining the full logical output descriptors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StaticLaunchDomain {
    logical_elements: usize,
    work_items: usize,
}

impl StaticLaunchDomain {
    fn checked(item: &ScheduleItem, logical_elements: usize) -> Result<Self, &'static str> {
        let work_items = match item.kernel.operation() {
            Operation::PrefixScan(value) => {
                let plan = crate::prefix_scan_native::NativePrefixScanPlan::new(value)?;
                if plan.elements != logical_elements {
                    return Err("prefix-scan logical output extent mismatch");
                }
                plan.work_items()
            }
            Operation::Sort(value) => {
                let plan = crate::portable_sort::PortableSortPair::new(value)
                    .map_err(|_| "portable sort launch geometry is invalid")?;
                if plan.elements() != logical_elements {
                    return Err("sort logical output extent mismatch");
                }
                plan.launch_extent()
            }
            _ => logical_elements,
        };
        Ok(Self {
            logical_elements,
            work_items,
        })
    }
}

pub(crate) fn validate_portable_prefix_scan_bindings(
    portable: &crate::prefix_scan_native::PortablePrefixScan<'_>,
    bindings: &[ScheduleInputBinding],
) -> Result<(), crate::prefix_scan_native::PortablePrefixScanError> {
    let plan = portable.plan();
    let bytes = plan
        .elements
        .checked_mul(plan.input_dtype.itemsize())
        .ok_or(crate::prefix_scan_native::PortablePrefixScanError::Overflow)?;
    let [binding] = bindings else {
        return Err(
            crate::prefix_scan_native::PortablePrefixScanError::InvalidBinding(
                "scan requires exactly one dense source binding".into(),
            ),
        );
    };
    if binding.abi_index != 0
        || binding.input_node != portable.value().input
        || binding.desc.id != plan.input
        || binding.desc.shape != portable.value().input_shape
        || binding.desc.dtype != plan.input_dtype
        || binding.desc.bytes != bytes
        || !binding.desc.read_only
        || binding.desc.view.is_some()
    {
        return Err(
            crate::prefix_scan_native::PortablePrefixScanError::InvalidBinding(
                "scan source is not its exact dense descriptor".into(),
            ),
        );
    }
    Ok(())
}

/// Backend-neutral pointer metadata projected from an existing renderer ABI.
pub(crate) struct StaticRenderedBuffer {
    pub(crate) id: u64,
    pub(crate) dtype: DType,
    pub(crate) source_shape: Shape,
    pub(crate) elements: usize,
    pub(crate) output_ordinal: Option<usize>,
}

/// Logical allocation metadata plus the native-handle requirement derived
/// from the complete rendered prefix. A zero-byte buffer keeps its logical
/// descriptor while receiving a private physical sentinel only when a
/// nonempty kernel launch includes that pointer in its ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StaticBufferAllocation {
    pub(crate) elements: usize,
    pub(crate) bytes: usize,
    pub(crate) dtype: DType,
    pub(crate) requires_native_handle: bool,
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
    if rendered.is_empty() {
        return Err(invalid("rendered ABI is empty".into()));
    }
    let mut output_ordinals = BTreeSet::new();
    for buffer in &rendered {
        if let Some(ordinal) = buffer.output_ordinal
            && (!output_ordinals.insert(ordinal) || ordinal >= item.outputs.len())
        {
            return Err(invalid("rendered output ordinal is invalid".into()));
        }
    }
    if output_ordinals.len() != item.outputs.len()
        || !output_ordinals.iter().copied().eq(0..item.outputs.len())
    {
        return Err(invalid(
            "rendered ABI does not bijectively cover scheduled outputs".into(),
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
            let desc = if let Some(ordinal) = abi.output_ordinal {
                item.outputs
                    .iter()
                    .nth(ordinal)
                    .expect("validated output ordinal")
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
                || abi.output_ordinal.is_some() == desc.read_only
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
                role: if let Some(ordinal) = abi.output_ordinal {
                    StaticBufferRole::Output(ordinal)
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

/// Exact within one `StaticPlanAdapter` build; the adapter type is the backend
/// domain, so a slot can never cross renderer/device address spaces.
#[derive(Clone, Debug, Eq, PartialEq)]
struct StaticSlotCompatibility {
    dtype: DType,
    source_shape: Shape,
    bytes: usize,
    alignment: usize,
}

/// Runtime-only physical allocation projection for one validated single-device
/// prefix. Logical IDs remain the renderer ABI; slots own native resources.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct StaticAllocationPlan {
    slots: Vec<StaticBufferAllocation>,
    logical_slots: BTreeMap<u64, usize>,
}

impl StaticAllocationPlan {
    pub(crate) fn slots(&self) -> &[StaticBufferAllocation] {
        &self.slots
    }

    pub(crate) fn logical_slots(&self) -> &BTreeMap<u64, usize> {
        &self.logical_slots
    }

    #[cfg(test)]
    fn peak_bytes(&self) -> usize {
        self.slots.iter().map(|slot| slot.bytes).sum()
    }
}

pub(crate) struct StaticItemPlan<R> {
    rendered: R,
    cache_key: String,
    extent: usize,
    buffer_ids: Vec<u64>,
    input_ids: Vec<u64>,
}

/// Fully validated schedule/render/buffer graph. Constructing this type is pure
/// with respect to native queues, caches, programs, and buffers.
pub(crate) struct StaticSchedulePlan<R> {
    items: Vec<StaticItemPlan<R>>,
    buffers: BTreeMap<u64, StaticBufferPlan>,
    external_inputs: Vec<u64>,
    retained_outputs: Vec<u64>,
    allocations: StaticAllocationPlan,
}

/// Pure renderer/planner seam shared by ordinary device execution and CUDA
/// whole-prefix graph capture.
pub(crate) trait StaticPlanAdapter: sealed::Sealed + Sized {
    type Error;
    type Rendered;

    fn render(&self, item: &ScheduleItem) -> Result<StaticRendered<Self::Rendered>, Self::Error>;
    fn invalid_binding(reason: String) -> Self::Error;
    fn unsupported(reason: String) -> Self::Error;
    fn overflow() -> Self::Error;
}

/// Coarse backend resource seam. Operation dispatch deliberately remains in
/// each existing renderer rather than being reconstructed here.
pub(crate) trait StaticDeviceAdapter: StaticPlanAdapter {
    type Kernel;
    type Buffer;
    type Queue;

    /// Preserves the backend's established whole-item zero-domain preparation
    /// policy, including compilation, allocation, and queue participation.
    fn prepare_zero_extent(&self) -> bool;
    fn compile(&self, rendered: &Self::Rendered) -> Result<Self::Kernel, Self::Error>;
    fn compiled_cache_key(&self, kernel: &Self::Kernel) -> String;
    fn allocate(&self, request: StaticBufferAllocation) -> Result<Self::Buffer, Self::Error>;
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
    pub(crate) fn items(&self) -> impl ExactSizeIterator<Item = &StaticItemPlan<R>> {
        self.items.iter()
    }

    pub(crate) fn buffers(&self) -> &BTreeMap<u64, StaticBufferPlan> {
        &self.buffers
    }

    pub(crate) fn external_inputs(&self) -> &[u64] {
        &self.external_inputs
    }

    pub(crate) fn retained_outputs(&self) -> &[u64] {
        &self.retained_outputs
    }

    pub(crate) fn allocations(&self) -> &StaticAllocationPlan {
        &self.allocations
    }

    pub(crate) fn build<A>(
        adapter: &A,
        items: &[ScheduleItem],
        retained: Option<&[u64]>,
    ) -> Result<Self, A::Error>
    where
        A: StaticPlanAdapter<Rendered = R>,
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
            let input_ids = rendered
                .buffers
                .iter()
                .filter(|buffer| buffer.role == StaticBufferRole::Input)
                .map(|buffer| buffer.id)
                .collect::<Vec<_>>();
            let mut output_ids = vec![None; item.outputs.len()];
            for buffer in &rendered.buffers {
                if let StaticBufferRole::Output(ordinal) = buffer.role
                    && output_ids
                        .get_mut(ordinal)
                        .is_none_or(|slot| slot.replace(buffer.id).is_some())
                {
                    return Err(A::invalid_binding(
                        "static item output ordinal is invalid".into(),
                    ));
                }
            }
            if output_ids.into_iter().collect::<Option<Vec<_>>>().is_none() {
                return Err(A::invalid_binding(
                    "static item requires every scheduled output in its writable ABI".into(),
                ));
            }
            let primary = rendered
                .buffers
                .iter()
                .find(|buffer| buffer.role == StaticBufferRole::Output(0))
                .ok_or_else(|| A::invalid_binding("static primary output is absent".into()))?;
            let launch = StaticLaunchDomain::checked(item, primary.elements)
                .map_err(|reason| A::invalid_binding(reason.into()))?;
            for (ordinal, expected) in item.outputs.iter().enumerate() {
                let output = rendered
                    .buffers
                    .iter()
                    .find(|buffer| buffer.role == StaticBufferRole::Output(ordinal))
                    .expect("validated output ordinal");
                if expected.view.is_some()
                    || output.id != expected.id
                    || output.dtype != expected.dtype
                    || output.source_shape != expected.shape
                    || output.elements != expected.shape.numel().map_err(|_| A::overflow())?
                    || output.elements != launch.logical_elements
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
            }
            if rendered.extent != launch.work_items {
                return Err(A::invalid_binding(
                    "rendered launch extent mismatches scheduled output".into(),
                ));
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
                input_ids,
            });
        }

        for (item_index, item) in planned.iter().enumerate() {
            let source_item = &items[item_index];
            for input in &item.input_ids {
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
                .flat_map(|item| item.outputs.iter().map(|output| output.id))
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
            .collect::<Vec<_>>();
        let allocations = build_static_allocation_plan::<A>(
            &planned,
            &buffers,
            &buffer_order,
            &external_inputs,
            &retained_outputs,
        )?;

        Ok(Self {
            items: planned,
            buffers,
            external_inputs,
            retained_outputs,
            allocations,
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

fn build_static_allocation_plan<A: StaticPlanAdapter>(
    items: &[StaticItemPlan<A::Rendered>],
    buffers: &BTreeMap<u64, StaticBufferPlan>,
    buffer_order: &[u64],
    external_inputs: &[u64],
    retained_outputs: &[u64],
) -> Result<StaticAllocationPlan, A::Error> {
    let required = items
        .iter()
        .filter(|item| item.extent != 0)
        .flat_map(|item| item.buffer_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    let external = external_inputs.iter().copied().collect::<BTreeSet<_>>();
    let retained = retained_outputs.iter().copied().collect::<BTreeSet<_>>();
    let mut requests = buffer_order
        .iter()
        .enumerate()
        .map(|(order, id)| {
            let buffer = &buffers[id];
            let producer_position = buffer.producer.unwrap_or(0);
            let last_consumer_position = items
                .iter()
                .enumerate()
                .filter(|(_, item)| item.input_ids.contains(id))
                .map(|(position, _)| position)
                .max()
                .unwrap_or(producer_position);
            let policy = if !required.contains(id) {
                ExactSlotPolicy::Absent
            } else if buffer.bytes != 0
                && buffer.producer.is_some()
                && !external.contains(id)
                && !retained.contains(id)
            {
                ExactSlotPolicy::Reusable
            } else {
                ExactSlotPolicy::Private
            };
            (
                producer_position,
                order,
                ExactSlotRequest {
                    identity: *id,
                    compatibility: StaticSlotCompatibility {
                        dtype: buffer.dtype,
                        source_shape: buffer.source_shape.clone(),
                        bytes: buffer.bytes,
                        alignment: buffer.alignment,
                    },
                    producer_position,
                    last_consumer_position,
                    policy,
                },
            )
        })
        .collect::<Vec<_>>();
    requests.sort_by_key(|(producer, order, _)| (*producer, *order));
    let assignments = assign_exact_slots(requests.into_iter().map(|(_, _, request)| request));
    let slot_count = assignments
        .iter()
        .filter_map(|assignment| assignment.slot)
        .max()
        .map_or(0usize, |slot| slot as usize + 1);
    let mut slots = vec![None; slot_count];
    let mut logical_slots = BTreeMap::new();
    for assignment in assignments {
        let Some(slot) = assignment.slot else {
            continue;
        };
        let slot = usize::try_from(slot).map_err(|_| A::overflow())?;
        let buffer = &buffers[&assignment.identity];
        let allocation = StaticBufferAllocation {
            elements: buffer.elements,
            bytes: buffer.bytes,
            dtype: buffer.dtype,
            requires_native_handle: true,
        };
        match &slots[slot] {
            Some(existing) if existing != &allocation => {
                return Err(A::invalid_binding(
                    "reused static slot has conflicting physical descriptors".into(),
                ));
            }
            Some(_) => {}
            None => slots[slot] = Some(allocation),
        }
        logical_slots.insert(assignment.identity, slot);
    }
    let slots = slots
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| A::invalid_binding("static allocation slot is vacant".into()))?;
    for item in items.iter().filter(|item| item.extent != 0) {
        let item_slots = item
            .buffer_ids
            .iter()
            .map(|id| logical_slots.get(id).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                A::invalid_binding("nonzero static item has an unallocated logical buffer".into())
            })?;
        if item_slots.iter().copied().collect::<BTreeSet<_>>().len() != item_slots.len() {
            return Err(A::invalid_binding(
                "distinct logical buffers in one static item alias a physical slot".into(),
            ));
        }
    }
    slots.iter().try_fold(0usize, |total, allocation| {
        total.checked_add(allocation.bytes).ok_or_else(A::overflow)
    })?;
    Ok(StaticAllocationPlan {
        slots,
        logical_slots,
    })
}

impl<R> StaticItemPlan<R> {
    pub(crate) fn rendered(&self) -> &R {
        &self.rendered
    }

    pub(crate) fn extent(&self) -> usize {
        self.extent
    }

    pub(crate) fn buffer_ids(&self) -> &[u64] {
        &self.buffer_ids
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
    slots: Vec<A::Buffer>,
    logical_slots: BTreeMap<u64, usize>,
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

    #[cfg(test)]
    fn prepare_for_outputs(
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
        let StaticSchedulePlan {
            items,
            buffers: buffer_plans,
            external_inputs,
            retained_outputs,
            allocations,
            ..
        } = plan;
        let prepare_zero_extent = adapter.prepare_zero_extent();
        let mut prepared_items = Vec::with_capacity(items.len());
        for item in items {
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
        let mut slots = Vec::with_capacity(allocations.slots.len());
        for allocation in &allocations.slots {
            slots.push(adapter.allocate(*allocation)?);
        }
        let queue = prepared_items
            .iter()
            .any(|item| item.extent != 0)
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
            slots,
            logical_slots: allocations.logical_slots,
            buffer_plans,
            external_inputs,
            retained_outputs,
            compiled_cache_keys,
        })
    }

    pub(crate) fn cache_len(&self) -> usize {
        self.adapter.cache_len()
    }

    pub(crate) fn compiled_cache_keys(&self) -> Vec<String> {
        self.compiled_cache_keys.clone()
    }

    fn buffer(&self, id: u64) -> Option<&A::Buffer> {
        self.logical_slots
            .get(&id)
            .and_then(|slot| self.slots.get(*slot))
    }

    pub(crate) fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), A::Error> {
        // Complete all host validation and allocation before the first driver call.
        let mut uploads = Vec::with_capacity(self.external_inputs.len());
        for id in &self.external_inputs {
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
            if self.buffer(*id).is_some() {
                uploads.push((*id, bytes));
            }
        }
        let mut downloads = self
            .retained_outputs
            .iter()
            .map(|id| (*id, vec![0; self.buffer_plans[id].bytes]))
            .collect::<Vec<_>>();

        if let Some(queue) = &self.queue {
            for (id, bytes) in &uploads {
                self.adapter.write(
                    queue,
                    self.buffer(*id).ok_or_else(|| {
                        A::invalid_binding(format!("logical input buffer {id} is absent"))
                    })?,
                    bytes,
                )?;
            }
            for item in &self.items {
                if item.extent == 0 {
                    continue;
                }
                let Some(kernel) = item.kernel.as_ref() else {
                    return Err(A::invalid_binding(
                        "nonzero item has no compiled kernel".into(),
                    ));
                };
                let bindings = item
                    .buffer_ids
                    .iter()
                    .map(|id| {
                        self.buffer(*id).ok_or_else(|| {
                            A::invalid_binding(format!("logical buffer {id} is absent"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.adapter.launch_and_wait(queue, kernel, &bindings)?;
            }
            for (id, bytes) in &mut downloads {
                if !bytes.is_empty() {
                    let buffer = self.buffer(*id).ok_or_else(|| {
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

fn validate_prefix<A: StaticPlanAdapter>(items: &[ScheduleItem]) -> Result<(), A::Error> {
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
        release: usize,
        queue: usize,
        write: usize,
        launch: usize,
        read: usize,
        fail_allocate_after: Option<usize>,
        fail_launch_after: Option<usize>,
        fail_read_after: Option<usize>,
        allocations: Vec<StaticBufferAllocation>,
    }

    #[derive(Clone)]
    struct FakeAdapter(Rc<RefCell<Calls>>);
    struct FakeRendered;
    struct FakeKernel;
    struct FakeQueue;
    struct FakeBuffer {
        bytes: RefCell<Vec<u8>>,
        calls: Rc<RefCell<Calls>>,
    }

    impl Drop for FakeBuffer {
        fn drop(&mut self) {
            self.calls.borrow_mut().release += 1;
        }
    }

    impl Sealed for FakeAdapter {}
    impl StaticPlanAdapter for FakeAdapter {
        type Error = String;
        type Rendered = FakeRendered;

        fn render(
            &self,
            item: &ScheduleItem,
        ) -> Result<StaticRendered<Self::Rendered>, Self::Error> {
            let mut buffers = item
                .ordered_inputs()
                .iter()
                .map(|binding| fake_use(&binding.desc, false))
                .collect::<Result<Vec<_>, _>>()?;
            for (ordinal, output) in item.outputs.iter().enumerate() {
                let mut use_ = fake_use(output, true)?;
                use_.role = StaticBufferRole::Output(ordinal);
                buffers.push(use_);
            }
            let logical = item
                .primary_output()
                .shape
                .numel()
                .map_err(|_| "overflow".to_owned())?;
            Ok(StaticRendered {
                artifact: FakeRendered,
                cache_key: item.cache_key.to_string(),
                extent: StaticLaunchDomain::checked(item, logical)
                    .map_err(str::to_owned)?
                    .work_items,
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
    }
    impl StaticDeviceAdapter for FakeAdapter {
        type Kernel = FakeKernel;
        type Buffer = FakeBuffer;
        type Queue = FakeQueue;

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
        fn allocate(&self, request: StaticBufferAllocation) -> Result<Self::Buffer, Self::Error> {
            let mut calls = self.0.borrow_mut();
            calls.allocate += 1;
            calls.allocations.push(request);
            if let Some(remaining) = calls.fail_allocate_after.as_mut() {
                if *remaining == 0 {
                    calls.fail_allocate_after = None;
                    return Err("injected allocation failure".into());
                }
                *remaining -= 1;
            }
            drop(calls);
            Ok(FakeBuffer {
                bytes: RefCell::new(vec![0; request.elements * request.dtype.itemsize()]),
                calls: self.0.clone(),
            })
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
            buffer.bytes.borrow_mut().copy_from_slice(bytes);
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
                .bytes
                .borrow()
                .clone();
            buffers
                .last()
                .ok_or_else(|| "missing output".to_owned())?
                .bytes
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
            bytes.copy_from_slice(&buffer.bytes.borrow());
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
                StaticBufferRole::Output(0)
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

    fn reusable_linear_schedule() -> (crate::Schedule, u64, [u64; 4]) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let first_value = graph.square(input).unwrap();
        let first = graph.contiguous(first_value).unwrap();
        let second_value = graph.square(first).unwrap();
        let second = graph.contiguous(second_value).unwrap();
        let third_value = graph.square(second).unwrap();
        let third = graph.contiguous(third_value).unwrap();
        let output_value = graph.square(third).unwrap();
        let output = graph.contiguous(output_value).unwrap();
        (
            crate::schedule(&graph, output).unwrap(),
            input.index() as u64,
            [
                first.index() as u64,
                second.index() as u64,
                third.index() as u64,
                output.index() as u64,
            ],
        )
    }

    #[test]
    fn exact_device_slots_reuse_disjoint_linear_temporaries_deterministically() {
        let (schedule, input, outputs) = reusable_linear_schedule();
        assert_eq!(schedule.items.len(), 4);
        let retained = [outputs[3]];
        let adapter = FakeAdapter(Rc::new(RefCell::new(Calls::default())));
        let first = StaticSchedulePlan::build(&adapter, &schedule.items, Some(&retained)).unwrap();
        let second = StaticSchedulePlan::build(&adapter, &schedule.items, Some(&retained)).unwrap();
        assert_eq!(first.allocations, second.allocations);
        let slots = first.allocations.logical_slots();
        assert_eq!(slots[&outputs[0]], slots[&outputs[2]]);
        assert_ne!(slots[&outputs[0]], slots[&outputs[1]]);
        assert_ne!(slots[&input], slots[&outputs[0]]);
        assert_ne!(slots[&outputs[3]], slots[&outputs[0]]);
        assert_eq!(first.allocations.slots().len(), 4);
        assert_eq!(
            first.allocations.peak_bytes(),
            4 * 2 * DType::F32.itemsize()
        );
        for (item, source) in first.items().zip(&schedule.items) {
            let expected_ids = source
                .ordered_inputs()
                .iter()
                .map(|binding| binding.desc.id)
                .chain(std::iter::once(source.primary_output().id))
                .collect::<Vec<_>>();
            assert_eq!(item.buffer_ids(), expected_ids);
            let item_slots = item
                .buffer_ids()
                .iter()
                .map(|id| slots[id])
                .collect::<BTreeSet<_>>();
            assert_eq!(item_slots.len(), item.buffer_ids().len());
        }

        let calls = Rc::new(RefCell::new(Calls::default()));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &retained,
        )
        .unwrap();
        assert_eq!(calls.borrow().allocate, 4);
        let mut values = BTreeMap::from([(
            input,
            TensorData::from_storage(Shape::from([2]), Storage::F32(vec![2.0, -3.0])).unwrap(),
        )]);
        prepared.execute(&mut values).unwrap();
        assert_eq!(
            values[&outputs[3]].storage(),
            &Storage::F32(vec![2.0, -3.0])
        );
    }

    #[test]
    fn public_static_prepare_retains_every_output_and_disables_temporary_reuse() {
        let (schedule, _, outputs) = reusable_linear_schedule();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let prepared =
            PreparedStaticSchedule::prepare(FakeAdapter(calls.clone()), &schedule.items).unwrap();
        assert_eq!(calls.borrow().allocate, 5);
        assert!(
            outputs
                .iter()
                .all(|id| prepared.logical_slots.contains_key(id))
        );
        assert_eq!(
            outputs
                .iter()
                .map(|id| prepared.logical_slots[id])
                .collect::<BTreeSet<_>>()
                .len(),
            outputs.len()
        );
    }

    #[test]
    fn external_inputs_and_retained_output_always_receive_distinct_private_slots() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], DType::F32);
        let rhs = graph.input_dtype("rhs", [2], DType::F32);
        let output = graph.add(lhs, rhs).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let adapter = FakeAdapter(Rc::new(RefCell::new(Calls::default())));
        let plan =
            StaticSchedulePlan::build(&adapter, &schedule.items, Some(&[output.index() as u64]))
                .unwrap();
        let slots = plan.allocations.logical_slots();
        assert_eq!(slots.len(), 3);
        assert_eq!(slots.values().copied().collect::<BTreeSet<_>>().len(), 3);
        assert_ne!(slots[&(lhs.index() as u64)], slots[&(rhs.index() as u64)]);
        assert_ne!(
            slots[&(lhs.index() as u64)],
            slots[&(output.index() as u64)]
        );
    }

    #[test]
    fn allocation_failure_drops_each_completed_physical_slot_once_before_queue_creation() {
        let (schedule, _, outputs) = reusable_linear_schedule();
        let calls = Rc::new(RefCell::new(Calls {
            fail_allocate_after: Some(2),
            ..Calls::default()
        }));
        assert_eq!(
            PreparedStaticSchedule::prepare_for_outputs(
                FakeAdapter(calls.clone()),
                &schedule.items,
                &[outputs[3]],
            )
            .err()
            .as_deref(),
            Some("injected allocation failure")
        );
        let calls = calls.borrow();
        assert_eq!(calls.allocate, 3);
        assert_eq!(calls.release, 2);
        assert_eq!(calls.queue, 0);
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
        assert_eq!(calls.allocate, 4);
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

        let mut aliased_output = schedule.clone();
        let mut output = aliased_output.items[0].primary_output().clone();
        output.view = Some(crate::AffineView::identity(output.shape.clone()));
        aliased_output.items[0].outputs = crate::ScheduledOutputs::single(output);

        for (name, items, retained) in [
            ("duplicate", duplicate.items, outputs.to_vec()),
            ("future", future.items, outputs.to_vec()),
            ("conflicting", conflicting.items, outputs.to_vec()),
            ("aliased-output", aliased_output.items, outputs.to_vec()),
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
        assert!(
            plan.allocations
                .logical_slots()
                .contains_key(&(product.index() as u64))
        );
        assert!(
            !plan
                .allocations
                .logical_slots()
                .contains_key(&(transposed.index() as u64)),
            "consumer-local affine views reuse their base logical slot"
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

    #[test]
    fn populated_zero_contraction_requests_only_private_zero_input_handles() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 0], DType::F32);
        let rhs = graph.input_dtype("rhs", [0, 3], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &[output.index() as u64],
        )
        .unwrap();
        let requests = calls.borrow().allocations.clone();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.bytes == 0 && request.requires_native_handle)
                .count(),
            2
        );
        assert!(requests.iter().any(|request| {
            request.bytes == 6 * DType::F32.itemsize() && request.requires_native_handle
        }));
        assert_eq!(
            [
                lhs.index() as u64,
                rhs.index() as u64,
                output.index() as u64
            ]
            .into_iter()
            .map(|id| prepared.logical_slots[&id])
            .collect::<BTreeSet<_>>()
            .len(),
            3,
            "zero-byte K=0 inputs keep private native-handle sentinels"
        );
        drop(prepared);

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 4], DType::F32);
        let rhs = empty.input_dtype("rhs", [4, 3], DType::F32);
        let output = empty.matmul(lhs, rhs).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &crate::schedule(&empty, output).unwrap().items,
            &[output.index() as u64],
        )
        .unwrap();
        assert!(calls.borrow().allocations.is_empty());
        drop(prepared);
    }

    #[test]
    fn zero_output_validates_all_logical_inputs_without_driver_or_publication() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [0, 4], DType::F32);
        let rhs = graph.input_dtype("rhs", [4, 3], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &[output.index() as u64],
        )
        .unwrap();

        let mut missing = BTreeMap::new();
        let before = missing.clone();
        assert!(
            prepared
                .execute(&mut missing)
                .unwrap_err()
                .contains("missing prefix input")
        );
        assert_eq!(missing, before);

        let mut wrong = BTreeMap::from([
            (
                lhs.index() as u64,
                TensorData::from_storage([0, 5], Storage::F32(Vec::new())).unwrap(),
            ),
            (
                rhs.index() as u64,
                TensorData::from_storage([4, 3], Storage::F32(vec![1.0; 12])).unwrap(),
            ),
        ]);
        let before = wrong.clone();
        assert!(
            prepared
                .execute(&mut wrong)
                .unwrap_err()
                .contains("descriptor mismatch")
        );
        assert_eq!(wrong, before);
        assert!(!wrong.contains_key(&(output.index() as u64)));

        let calls = calls.borrow();
        assert_eq!(
            (
                calls.allocate,
                calls.queue,
                calls.write,
                calls.launch,
                calls.read
            ),
            (0, 0, 0, 0, 0)
        );
    }

    #[test]
    fn coupled_sort_outputs_are_bijective_ordered_and_jointly_retained() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let (values, indices) = graph.sort(input, 1, false).unwrap();
        let schedule = crate::schedule_many(&graph, &[values, indices]).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let plan = StaticSchedulePlan::build(
            &FakeAdapter(calls),
            &schedule.items,
            Some(&[values.index() as u64, indices.index() as u64]),
        )
        .unwrap();
        assert_eq!(plan.items.len(), 1);
        assert_eq!(
            plan.items[0].buffer_ids,
            vec![
                input.index() as u64,
                values.index() as u64,
                indices.index() as u64
            ]
        );
        assert_eq!(plan.items[0].input_ids, vec![input.index() as u64]);
        assert_eq!(
            plan.retained_outputs,
            vec![values.index() as u64, indices.index() as u64]
        );
        assert_eq!(plan.buffers[&(values.index() as u64)].producer, Some(0));
        assert_eq!(plan.buffers[&(indices.index() as u64)].producer, Some(0));
        assert_eq!(plan.allocations.logical_slots.len(), 3);
        assert_eq!(
            plan.allocations
                .logical_slots
                .values()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            3,
            "input and both same-item outputs own distinct physical slots"
        );
    }

    #[test]
    fn coupled_sort_consumes_a_device_resident_producer_with_exact_dependency() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let producer = graph.square(input).unwrap();
        let (values, indices) = graph.sort(producer, 1, true).unwrap();
        let schedule = crate::schedule_many(&graph, &[values, indices]).unwrap();
        assert_eq!(schedule.items.len(), 2);
        let plan = StaticSchedulePlan::build(
            &FakeAdapter(Rc::new(RefCell::new(Calls::default()))),
            &schedule.items,
            Some(&[values.index() as u64, indices.index() as u64]),
        )
        .unwrap();
        assert_eq!(plan.items[1].input_ids, vec![producer.index() as u64]);
        assert_eq!(schedule.items[1].dependencies, vec![schedule.items[0].id]);
        assert_eq!(plan.buffers[&(producer.index() as u64)].producer, Some(0));
        assert!(
            !plan.external_inputs.contains(&(producer.index() as u64)),
            "the sort source stays on device instead of becoming a host ABI"
        );
    }
}
