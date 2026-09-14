//! Retained scratch storage for authenticated fixed-shape native replay.
use super::capture::{CapturedSchedule, ReplayError};
use super::captured_replay::{ReplayValues, backend_error};
use crate::backend::{
    PreparedNativeDispatch, PreparedScheduleDispatch, PreparedScheduleDispatchFailure,
    PreparedScheduleItem, PreparedScheduleSegment,
};
use crate::{CpuJitBackend, ScheduleItem, TensorData};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug, Eq, PartialEq)]
struct BufferKey {
    buffer: u64,
    descriptor: crate::BufferDesc,
    elements: usize,
    direct_view: Option<crate::AffineView>,
}

#[derive(Clone, Debug)]
enum SlotSource {
    Copy(usize),
    Affine {
        source: usize,
        view: crate::AffineView,
    },
}

#[derive(Clone, Debug)]
struct WorkspaceSlot {
    key: BufferKey,
    source: Option<SlotSource>,
}

struct WorkspaceItem {
    slots: Vec<usize>,
    output: usize,
    outputs: Vec<usize>,
    logical_indices: Vec<usize>,
    elided: bool,
    dispatch: Option<PreparedScheduleDispatch>,
    output_initialization: crate::cpu_jit::NativeOutputInitialization,
}

struct WorkspaceOutputAction {
    outputs: Vec<usize>,
    initialization: crate::cpu_jit::NativeOutputInitialization,
}

struct WorkspaceDispatchSegment {
    indices: Vec<usize>,
    prerequisites: Vec<usize>,
    materializations: Vec<crate::cpu_jit::NativeDispatchMaterialization>,
    outputs: Vec<WorkspaceOutputAction>,
    dispatch: PreparedScheduleSegment,
}

enum WorkspaceDispatchStep {
    Segment(WorkspaceDispatchSegment),
    PerItem(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeDispatchSegmentEnd {
    NonDispatchBoundary,
    ModuleChange,
    OutputSlotAlias,
    Terminal,
}

/// Authenticated reasons why the sealed native dispatch tape contains its
/// observed number of module segments. Every segment has exactly one ending:
/// a mutually exclusive split cause or the terminal end of the tape. When
/// conditions coincide, module change precedes output alias. Derived inputs
/// produced inside a segment are sealed as typed pre-entry materializations,
/// so they no longer end a segment. The reached-module inventory excludes
/// rendered entries omitted from the tape by authenticated elision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeDispatchSegmentation {
    pub(crate) segment_count: usize,
    pub(crate) dispatch_reached_module_count: usize,
    pub(crate) terminal_segment_count: usize,
    pub(crate) non_dispatch_boundary_count: usize,
    pub(crate) module_change_count: usize,
    pub(crate) output_slot_alias_count: usize,
    pub(crate) derived_slot_dependency_count: usize,
}

impl NativeDispatchSegmentation {
    fn record(&mut self, end: NativeDispatchSegmentEnd) {
        self.segment_count += 1;
        let count = match end {
            NativeDispatchSegmentEnd::NonDispatchBoundary => &mut self.non_dispatch_boundary_count,
            NativeDispatchSegmentEnd::ModuleChange => &mut self.module_change_count,
            NativeDispatchSegmentEnd::OutputSlotAlias => &mut self.output_slot_alias_count,
            NativeDispatchSegmentEnd::Terminal => &mut self.terminal_segment_count,
        };
        *count += 1;
    }
}

fn dispatch_segment_split(
    crosses_module: bool,
    output_aliases_slot: bool,
) -> Option<NativeDispatchSegmentEnd> {
    if crosses_module {
        Some(NativeDispatchSegmentEnd::ModuleChange)
    } else if output_aliases_slot {
        Some(NativeDispatchSegmentEnd::OutputSlotAlias)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeReplayTraffic {
    pub(crate) external_input_import_count: u64,
    pub(crate) external_input_import_bytes: u64,
    pub(crate) borrowed_recurrent_input_bytes: u64,
    pub(crate) borrowed_recurrent_output_bytes: u64,
    pub(crate) retained_recurrent_state_count: u64,
    pub(crate) retained_recurrent_state_bytes: u64,
    pub(crate) replaced_recurrent_state_count: u64,
    pub(crate) replaced_recurrent_state_bytes: u64,
    pub(crate) materialized_egress_count: u64,
    pub(crate) materialized_egress_bytes: u64,
    pub(crate) executed_native_item_count: usize,
    pub(crate) module_dispatch_count: usize,
    pub(crate) module_dispatched_native_item_count: usize,
    pub(crate) skipped_output_clear_count: usize,
    pub(crate) native_dispatcher_wall_time: Duration,
}

impl NativeReplayTraffic {
    fn checked_native_dispatcher_total(
        &self,
        dispatcher_wall_time: Duration,
    ) -> Result<Duration, ReplayError> {
        self.native_dispatcher_wall_time
            .checked_add(dispatcher_wall_time)
            .ok_or_else(|| ReplayError::Descriptor("native dispatcher duration overflows".into()))
    }
}

/// Private scratch owned by one authenticated prepared native program.
/// Slots keep allocations, not authoritative replay bindings: supported dense
/// inputs borrow caller storage for one invocation, while every mutable derived
/// value is invalidated per run.
pub(super) struct NativeReplayWorkspace {
    buffers: Vec<crate::JitBuffer>,
    slots: Vec<WorkspaceSlot>,
    items: Vec<WorkspaceItem>,
    dispatch_tape: Arc<[WorkspaceDispatchStep]>,
    dispatch_segmentation: NativeDispatchSegmentation,
    dispatch_scratch: crate::cpu_jit::JitScheduleDispatchScratch,
    owners: BTreeMap<u64, usize>,
    inputs: Vec<(String, usize)>,
    immutable: BTreeSet<usize>,
    egress: Vec<(u64, usize, crate::Shape)>,
    valid: Vec<bool>,
    current_traffic: NativeReplayTraffic,
    #[cfg(test)]
    input_import_count: usize,
    #[cfg(test)]
    borrowed_external_input_bytes: usize,
    #[cfg(test)]
    intermediate_materialization_count: usize,
    #[cfg(test)]
    borrowed_recurrent_input_bytes: usize,
    #[cfg(test)]
    borrowed_recurrent_output_bytes: usize,
    #[cfg(test)]
    retained_transpose_matmul_input_count: usize,
    #[cfg(test)]
    affine_matmul_materialization_bytes: usize,
    #[cfg(test)]
    output_clear_count: usize,
    #[cfg(test)]
    skipped_output_clear_count: usize,
    #[cfg(test)]
    dispatch_metadata_build_count: usize,
    #[cfg(test)]
    injected_dispatch_failure: Option<usize>,
}

pub(super) struct NativeReplayBindings<'a> {
    slots: BTreeMap<usize, crate::cpu_jit::BorrowedJitBuffer<'a>>,
}

impl<'a> NativeReplayBindings<'a> {
    pub(super) fn new() -> Self {
        Self {
            slots: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeReplayWorkspaceStats {
    pub(crate) allocation_count: usize,
    pub(crate) input_import_count: usize,
    pub(crate) borrowed_external_input_bytes: usize,
    pub(crate) intermediate_materialization_count: usize,
    pub(crate) borrowed_recurrent_input_bytes: usize,
    pub(crate) borrowed_recurrent_output_bytes: usize,
    pub(crate) retained_transpose_matmul_input_count: usize,
    pub(crate) affine_matmul_materialization_bytes: usize,
    pub(crate) output_clear_count: usize,
    pub(crate) skipped_output_clear_count: usize,
    pub(crate) sealed_dispatch_step_count: usize,
    pub(crate) sealed_dispatch_segment_count: usize,
    pub(crate) sealed_prerequisite_slot_count: usize,
    pub(crate) sealed_derived_materialization_count: usize,
    pub(crate) dispatch_metadata_build_count: usize,
    pub(crate) last_materialized_egress_count: u64,
    pub(crate) last_materialized_egress_bytes: u64,
    pub(crate) dispatch_scratch_capacity_growth_count: usize,
    pub(crate) dispatch_scratch_is_empty: bool,
}

impl NativeReplayWorkspace {
    pub(super) fn new(
        capture: &CapturedSchedule,
        items: &[PreparedNativeDispatch],
    ) -> Result<Self, ReplayError> {
        let mut workspace = Self {
            buffers: Vec::new(),
            slots: Vec::new(),
            items: Vec::with_capacity(items.len()),
            dispatch_tape: Arc::from([]),
            dispatch_segmentation: NativeDispatchSegmentation::default(),
            dispatch_scratch: crate::cpu_jit::JitScheduleDispatchScratch::with_capacity(0, 0, 0, 0),
            owners: BTreeMap::new(),
            inputs: Vec::with_capacity(capture.inputs.len()),
            immutable: BTreeSet::new(),
            egress: Vec::new(),
            valid: Vec::new(),
            current_traffic: NativeReplayTraffic::default(),
            #[cfg(test)]
            input_import_count: 0,
            #[cfg(test)]
            borrowed_external_input_bytes: 0,
            #[cfg(test)]
            intermediate_materialization_count: 0,
            #[cfg(test)]
            borrowed_recurrent_input_bytes: 0,
            #[cfg(test)]
            borrowed_recurrent_output_bytes: 0,
            #[cfg(test)]
            retained_transpose_matmul_input_count: 0,
            #[cfg(test)]
            affine_matmul_materialization_bytes: 0,
            #[cfg(test)]
            output_clear_count: 0,
            #[cfg(test)]
            skipped_output_clear_count: 0,
            #[cfg(test)]
            dispatch_metadata_build_count: 0,
            #[cfg(test)]
            injected_dispatch_failure: None,
        };

        for input in &capture.inputs {
            let elements = input
                .desc
                .shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
            let slot =
                workspace.add_canonical(input.desc.id, input.desc.clone(), elements, false)?;
            workspace.inputs.push((input.name.clone(), slot));
        }
        for (buffer, value) in &capture.constants {
            let elements = value.len();
            let bytes = elements
                .checked_mul(value.dtype().itemsize())
                .ok_or_else(|| {
                    ReplayError::Descriptor("native workspace constant byte overflow".into())
                })?;
            let descriptor = crate::BufferDesc {
                id: *buffer,
                shape: value.shape().clone(),
                dtype: value.dtype(),
                bytes,
                alignment: value.dtype().itemsize().max(1),
                read_only: true,
                view: None,
            };
            let slot = workspace.add_canonical(*buffer, descriptor, elements, false)?;
            workspace.buffers[slot]
                .copy_from_tensor(value)
                .map_err(|error| ReplayError::Backend(error.to_string()))?;
            workspace.immutable.insert(slot);
        }
        let mut covered = BTreeSet::new();
        for prepared in items {
            match prepared {
                PreparedNativeDispatch::Item {
                    logical_index,
                    item: prepared,
                } => {
                    let item = capture.items.get(*logical_index).ok_or_else(|| {
                        ReplayError::Corrupt("prepared native item is out of range".into())
                    })?;
                    if !covered.insert(*logical_index) {
                        return Err(ReplayError::Corrupt(
                            "prepared native logical item repeats".into(),
                        ));
                    }
                    workspace.add_item(
                        *logical_index,
                        item,
                        prepared,
                        &capture.quantized_constants,
                    )?;
                }
                prepared @ PreparedNativeDispatch::ZeroDomain { logical_index, .. } => {
                    let item = capture.items.get(*logical_index).ok_or_else(|| {
                        ReplayError::Corrupt("prepared zero-domain item is out of range".into())
                    })?;
                    let layout = crate::backend::schedule_native_layout(item)
                        .map_err(super::captured_replay::backend_error)?;
                    if !covered.insert(*logical_index)
                        || !prepared.authenticates_layout(*logical_index, item, &layout)
                    {
                        return Err(ReplayError::Corrupt(
                            "prepared zero-domain item mismatch".into(),
                        ));
                    }
                    workspace.add_zero_domain(*logical_index, item)?;
                }
                PreparedNativeDispatch::StoreGroup(group) => {
                    if group
                        .members
                        .iter()
                        .any(|member| !covered.insert(member.logical_index))
                    {
                        return Err(ReplayError::Corrupt(
                            "prepared native logical item repeats".into(),
                        ));
                    }
                    workspace.add_store_group(capture, group)?;
                }
            }
        }
        if covered.len() != capture.items.len() {
            return Err(ReplayError::Corrupt(
                "prepared native logical item is absent".into(),
            ));
        }
        workspace.seal_dispatch_tape()?;

        workspace.valid.resize(workspace.buffers.len(), false);
        let immutable = workspace.immutable.iter().copied().collect::<Vec<_>>();
        let borrowed = NativeReplayBindings::new();
        for slot in immutable {
            if workspace.slots[slot].source.is_none() {
                workspace.valid[slot] = true;
            } else {
                workspace.prepare_slot(slot, &borrowed)?;
            }
        }
        workspace.plan_egress(capture)?;
        Ok(workspace)
    }

    fn add_zero_domain(
        &mut self,
        logical_index: usize,
        item: &ScheduleItem,
    ) -> Result<(), ReplayError> {
        if !item.outputs.is_single() || item.boundary.is_some() || item.is_effect() {
            return Err(ReplayError::Corrupt(
                "prepared zero-domain item is not single-output pure work".into(),
            ));
        }
        let output = item.primary_output();
        let elements = output
            .shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        if elements != 0 {
            return Err(ReplayError::Corrupt(
                "prepared zero-domain item has a nonempty output".into(),
            ));
        }
        let output_slot = self.add_canonical(output.id, output.clone(), elements, true)?;
        // The distinct owner has no payload to initialize. Its validity is
        // invocation-stable even though consumers may execute every replay.
        self.immutable.insert(output_slot);
        self.items.push(WorkspaceItem {
            slots: Vec::new(),
            output: output_slot,
            outputs: vec![output_slot],
            logical_indices: vec![logical_index],
            elided: true,
            dispatch: None,
            output_initialization: crate::cpu_jit::NativeOutputInitialization::NeedsZero,
        });
        Ok(())
    }

    fn add_item(
        &mut self,
        index: usize,
        item: &ScheduleItem,
        prepared: &PreparedScheduleItem,
        quantized: &BTreeMap<u64, crate::QuantizedTensorData>,
    ) -> Result<(), ReplayError> {
        let output = item.primary_output();
        let output_elements = output
            .shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        if let Some(source_buffer) = prepared.elided_output_source(item).map_err(backend_error)? {
            let source = self.owners.get(&source_buffer).copied().ok_or_else(|| {
                ReplayError::Corrupt("elided native transpose source is absent".into())
            })?;
            let source_key = &self.slots[source].key;
            if source_key.elements != output_elements || source_key.descriptor.dtype != output.dtype
            {
                return Err(ReplayError::Corrupt(
                    "elided native transpose descriptor mismatch".into(),
                ));
            }
            if self.owners.insert(output.id, source).is_some() {
                return Err(ReplayError::Corrupt(
                    "elided native transpose output has multiple owners".into(),
                ));
            }
            self.items.push(WorkspaceItem {
                slots: Vec::new(),
                output: source,
                outputs: vec![source],
                logical_indices: vec![index],
                elided: true,
                dispatch: None,
                output_initialization: crate::cpu_jit::NativeOutputInitialization::NeedsZero,
            });
            return Ok(());
        }
        let output_slot = self.add_canonical(output.id, output.clone(), output_elements, true)?;
        let direct_matmul = matches!(
            item.kernel.operation(),
            crate::Operation::Matmul(
                crate::MatmulValue::Serial(_)
                    | crate::MatmulValue::Tiled(_)
                    | crate::MatmulValue::TensorCore(_)
            )
        );
        let mut slots = Vec::with_capacity(prepared.abi().buffers.len());
        for abi in &prepared.abi().buffers {
            if abi.id == output.id {
                if abi.dtype != output.dtype || abi.elements != output_elements || !abi.mutable {
                    return Err(ReplayError::Corrupt(
                        "native workspace output descriptor mismatch".into(),
                    ));
                }
                slots.push(output_slot);
                continue;
            }
            let retained_source = prepared
                .retained_matmul_source(item, abi.id)
                .map_err(backend_error)?;
            slots.push(self.resolve_input(item, abi, direct_matmul, retained_source)?);
        }
        let dispatch = prepared
            .prepare_workspace_dispatch(item, &self.buffers, &slots, quantized)
            .map_err(backend_error)?;
        let output_initialization = prepared
            .output_initialization(item)
            .map_err(backend_error)?;
        self.items.push(WorkspaceItem {
            slots,
            output: output_slot,
            outputs: vec![output_slot],
            logical_indices: vec![index],
            elided: false,
            dispatch,
            output_initialization,
        });
        Ok(())
    }

    fn add_store_group(
        &mut self,
        capture: &CapturedSchedule,
        group: &crate::backend::PreparedNativeStoreGroup,
    ) -> Result<(), ReplayError> {
        let mut outputs = Vec::with_capacity(group.members.len());
        for member in &group.members {
            let logical_index = member.logical_index;
            let logical_item = capture.items.get(logical_index).ok_or_else(|| {
                ReplayError::Corrupt("native store group item is out of range".into())
            })?;
            let output = logical_item.primary_output();
            let elements = output
                .shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
            if output.id != member.output_buffer {
                return Err(ReplayError::Corrupt(
                    "native store group output identity mismatch".into(),
                ));
            }
            outputs.push(self.add_canonical(output.id, output.clone(), elements, true)?);
        }
        let mut slots = Vec::with_capacity(group.abi().buffers.len());
        for abi in &group.abi().buffers {
            if abi.mutable {
                let slot = self.owners.get(&abi.id).copied().ok_or_else(|| {
                    ReplayError::Corrupt("native store group output owner is absent".into())
                })?;
                if !group
                    .members
                    .iter()
                    .any(|member| member.output_buffer == abi.id)
                    || self.slots[slot].key.elements != abi.elements
                    || self.slots[slot].key.descriptor.dtype != abi.dtype
                {
                    return Err(ReplayError::Corrupt(
                        "native store group output descriptor mismatch".into(),
                    ));
                }
                slots.push(slot);
            } else {
                slots.push(
                    self.resolve_group_input(
                        capture,
                        &group
                            .members
                            .iter()
                            .map(|member| member.logical_index)
                            .collect::<Vec<_>>(),
                        abi,
                    )?,
                );
            }
        }
        let dispatch = group
            .prepare_workspace_dispatch(&self.buffers, &slots)
            .map_err(backend_error)?;
        let output = *outputs
            .last()
            .ok_or_else(|| ReplayError::Corrupt("native store group is empty".into()))?;
        self.items.push(WorkspaceItem {
            slots,
            output,
            outputs,
            logical_indices: group
                .members
                .iter()
                .map(|member| member.logical_index)
                .collect(),
            elided: false,
            dispatch: Some(dispatch),
            output_initialization: group.output_initialization(),
        });
        Ok(())
    }

    fn resolve_group_input(
        &mut self,
        capture: &CapturedSchedule,
        members: &[usize],
        abi: &crate::cpu_jit::BufferAbi,
    ) -> Result<usize, ReplayError> {
        let bindings = members
            .iter()
            .flat_map(|index| {
                capture.items[*index]
                    .ordered_inputs()
                    .iter()
                    .filter(move |binding| binding.desc.id == abi.id)
                    .map(move |binding| (*index, binding))
            })
            .collect::<Vec<_>>();
        let Some((member, binding)) = bindings.first().copied() else {
            return Err(ReplayError::Corrupt(format!(
                "native store group input {} has no binding",
                abi.id
            )));
        };
        if bindings
            .iter()
            .any(|(_, candidate)| candidate.desc != binding.desc)
            || binding.desc.view.is_some()
            || abi.mutable
        {
            return Err(ReplayError::Corrupt(
                "native store group input binding mismatch".into(),
            ));
        }
        self.resolve_input(&capture.items[member], abi, false, None)
    }

    fn resolve_input(
        &mut self,
        item: &ScheduleItem,
        abi: &crate::cpu_jit::BufferAbi,
        direct_matmul: bool,
        retained_source: Option<u64>,
    ) -> Result<usize, ReplayError> {
        let binding = item
            .ordered_inputs()
            .iter()
            .find(|binding| binding.desc.id == abi.id)
            .ok_or_else(|| {
                ReplayError::Corrupt(format!("native workspace input {} has no binding", abi.id))
            })?;
        if let Some(source_buffer) = retained_source {
            let source = self.owners.get(&source_buffer).copied().ok_or_else(|| {
                ReplayError::Corrupt(format!(
                    "retained native matmul source {source_buffer} is absent"
                ))
            })?;
            let source_key = &self.slots[source].key;
            if abi.dtype != binding.desc.dtype
                || abi.elements != source_key.elements
                || abi.mutable
                || source_key.descriptor.dtype != binding.desc.dtype
            {
                return Err(ReplayError::Corrupt(format!(
                    "retained native matmul input {} descriptor mismatch",
                    abi.id
                )));
            }
            #[cfg(test)]
            {
                self.retained_transpose_matmul_input_count =
                    self.retained_transpose_matmul_input_count.saturating_add(1);
            }
            return Ok(source);
        }
        let direct_view = direct_matmul.then(|| binding.desc.view.clone()).flatten();
        let source_shape = direct_view
            .as_ref()
            .map(|view| &view.source_shape)
            .unwrap_or(&binding.desc.shape);
        let source_elements = source_shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let mut owner_descriptor = binding.desc.clone();
        owner_descriptor.shape = source_shape.clone();
        owner_descriptor.read_only = false;
        owner_descriptor.view = None;
        let owner_key = BufferKey {
            buffer: abi.id,
            descriptor: owner_descriptor,
            elements: source_elements,
            direct_view: None,
        };
        let source = self
            .slots
            .iter()
            .position(|slot| slot.source.is_none() && slot.key == owner_key)
            .ok_or_else(|| {
                ReplayError::Corrupt(format!(
                    "native workspace input {} has no exact source",
                    abi.id
                ))
            })?;
        let elements = match &direct_view {
            Some(view) => view
                .logical_shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?,
            None => source_elements,
        };
        if abi.dtype != binding.desc.dtype || abi.elements != elements || abi.mutable {
            return Err(ReplayError::Corrupt(format!(
                "native workspace input {} descriptor mismatch",
                abi.id
            )));
        }
        let mut descriptor = binding.desc.clone();
        // Access mode belongs to the kernel ABI, not the retained storage key.
        descriptor.read_only = false;
        let key = BufferKey {
            buffer: abi.id,
            descriptor,
            elements,
            direct_view: direct_view.clone(),
        };
        if self.slots[source].key == key {
            return Ok(source);
        }
        if let Some(slot) = self.slots.iter().position(|slot| slot.key == key) {
            return Ok(slot);
        }
        let derivation = match direct_view {
            Some(view) => SlotSource::Affine { source, view },
            None => SlotSource::Copy(source),
        };
        let slot = self.add_slot(key, false, Some(derivation));
        if self.immutable.contains(&source) {
            self.immutable.insert(slot);
        }
        Ok(slot)
    }

    fn add_canonical(
        &mut self,
        buffer: u64,
        mut descriptor: crate::BufferDesc,
        elements: usize,
        mutable: bool,
    ) -> Result<usize, ReplayError> {
        if self.owners.contains_key(&buffer) {
            return Err(ReplayError::Corrupt(format!(
                "native workspace buffer {buffer} has multiple owners"
            )));
        }
        descriptor.view = None;
        descriptor.read_only = false;
        let slot = self.add_slot(
            BufferKey {
                buffer,
                descriptor,
                elements,
                direct_view: None,
            },
            mutable,
            None,
        );
        self.owners.insert(buffer, slot);
        Ok(slot)
    }

    fn add_slot(&mut self, key: BufferKey, mutable: bool, source: Option<SlotSource>) -> usize {
        let slot = self.buffers.len();
        self.buffers.push(crate::JitBuffer::zeroed(
            key.descriptor.dtype,
            key.elements,
            mutable,
        ));
        self.slots.push(WorkspaceSlot { key, source });
        slot
    }

    fn plan_egress(&mut self, capture: &CapturedSchedule) -> Result<(), ReplayError> {
        let alias_outputs = capture
            .requested_passthroughs
            .iter()
            .map(|alias| alias.requested.index() as u64)
            .collect::<BTreeSet<_>>();
        let mut egress = capture.requested.iter().copied().collect::<BTreeSet<_>>();
        egress.extend(
            capture
                .requested_passthroughs
                .iter()
                .map(|alias| alias.source.index() as u64),
        );
        for buffer in egress {
            if alias_outputs.contains(&buffer) && !self.owners.contains_key(&buffer) {
                continue;
            }
            let slot =
                self.owners.get(&buffer).copied().ok_or_else(|| {
                    ReplayError::Missing(format!("native workspace egress {buffer}"))
                })?;
            let shape = self.slots[slot].key.descriptor.shape.clone();
            self.egress.push((buffer, slot, shape));
        }
        Ok(())
    }

    pub(super) fn begin<'a>(
        &mut self,
        provided: &'a BTreeMap<String, TensorData>,
        bindings: &mut NativeReplayBindings<'a>,
    ) -> Result<(), ReplayError> {
        self.begin_resolved();
        for (name, value) in provided {
            self.bind_external_input(name, value, bindings)?;
        }
        self.finish_inputs()
    }

    pub(super) fn begin_resolved(&mut self) {
        self.current_traffic = NativeReplayTraffic::default();
        self.valid.fill(false);
        for slot in &self.immutable {
            self.valid[*slot] = true;
        }
    }

    pub(super) fn import_input(
        &mut self,
        name: &str,
        value: &TensorData,
    ) -> Result<(), ReplayError> {
        let slot = self
            .inputs
            .iter()
            .find_map(|(input, slot)| (input == name).then_some(*slot))
            .ok_or_else(|| ReplayError::Extra(name.to_owned()))?;
        if self.valid[slot] {
            return Err(ReplayError::Corrupt(format!(
                "native workspace input {name:?} was imported twice"
            )));
        }
        self.buffers[slot]
            .copy_from_tensor(value)
            .map_err(|error| ReplayError::Backend(error.to_string()))?;
        self.valid[slot] = true;
        self.current_traffic.external_input_import_count = self
            .current_traffic
            .external_input_import_count
            .checked_add(1)
            .ok_or_else(|| ReplayError::Descriptor("native input import count overflows".into()))?;
        self.current_traffic.external_input_import_bytes = self
            .current_traffic
            .external_input_import_bytes
            .checked_add(tensor_bytes(value)?)
            .ok_or_else(|| ReplayError::Descriptor("native input import bytes overflow".into()))?;
        #[cfg(test)]
        {
            self.input_import_count += 1;
        }
        Ok(())
    }

    /// Borrows the dense storage families used by fixed-shape CPU training
    /// batches. Other storage keeps the established owned-import fallback.
    pub(super) fn bind_external_input<'a>(
        &mut self,
        name: &str,
        value: &'a TensorData,
        bindings: &mut NativeReplayBindings<'a>,
    ) -> Result<(), ReplayError> {
        if !matches!(value.dtype(), crate::DType::F32 | crate::DType::I32)
            || value.native_dense_ptr().is_none()
        {
            return self.import_input(name, value);
        }
        self.bind_read_input(name, value, bindings, "external")?;
        #[cfg(test)]
        {
            self.borrowed_external_input_bytes = self
                .borrowed_external_input_bytes
                .saturating_add(value.len().saturating_mul(value.dtype().itemsize()));
        }
        Ok(())
    }

    pub(super) fn borrow_recurrent_input<'a>(
        &mut self,
        name: &str,
        value: &'a TensorData,
        bindings: &mut NativeReplayBindings<'a>,
    ) -> Result<(), ReplayError> {
        self.bind_read_input(name, value, bindings, "recurrent")?;
        self.current_traffic.borrowed_recurrent_input_bytes = self
            .current_traffic
            .borrowed_recurrent_input_bytes
            .checked_add(tensor_bytes(value)?)
            .ok_or_else(|| {
                ReplayError::Descriptor("native borrowed recurrent input bytes overflow".into())
            })?;
        #[cfg(test)]
        {
            self.borrowed_recurrent_input_bytes = self
                .borrowed_recurrent_input_bytes
                .saturating_add(value.len().saturating_mul(value.dtype().itemsize()));
        }
        Ok(())
    }

    fn bind_read_input<'a>(
        &mut self,
        name: &str,
        value: &'a TensorData,
        bindings: &mut NativeReplayBindings<'a>,
        kind: &str,
    ) -> Result<(), ReplayError> {
        let slot = self
            .inputs
            .iter()
            .find_map(|(input, slot)| (input == name).then_some(*slot))
            .ok_or_else(|| ReplayError::Extra(name.to_owned()))?;
        if self.valid[slot] || bindings.slots.contains_key(&slot) {
            return Err(ReplayError::Corrupt(format!(
                "native workspace input {name:?} was bound twice"
            )));
        }
        let descriptor = &self.slots[slot].key;
        if value.shape() != &descriptor.descriptor.shape
            || value.dtype() != descriptor.descriptor.dtype
            || value.len() != descriptor.elements
            || value.native_dense_ptr().is_none()
        {
            return Err(ReplayError::Corrupt(format!(
                "native workspace borrowed {kind} input {name:?} descriptor mismatch"
            )));
        }
        bindings
            .slots
            .insert(slot, crate::cpu_jit::BorrowedJitBuffer::Read(value));
        self.valid[slot] = true;
        let mut bound_slots = BTreeSet::from([slot]);
        loop {
            let aliases = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(alias, planned)| match planned.source {
                    Some(SlotSource::Copy(source))
                        if bound_slots.contains(&source) && !self.valid[alias] =>
                    {
                        Some(alias)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            if aliases.is_empty() {
                break;
            }
            for alias in aliases {
                bindings
                    .slots
                    .insert(alias, crate::cpu_jit::BorrowedJitBuffer::Read(value));
                self.valid[alias] = true;
                bound_slots.insert(alias);
            }
        }
        Ok(())
    }

    pub(super) fn borrow_recurrent_output<'a>(
        &mut self,
        buffer: u64,
        value: &'a mut TensorData,
        borrowed: &mut NativeReplayBindings<'a>,
    ) -> Result<(), ReplayError> {
        let slot = self
            .owners
            .get(&buffer)
            .copied()
            .ok_or_else(|| ReplayError::Missing(format!("native workspace output {buffer}")))?;
        if borrowed.slots.contains_key(&slot) {
            return Err(ReplayError::Corrupt(format!(
                "native workspace output {buffer} was bound twice"
            )));
        }
        let descriptor = &self.slots[slot].key;
        if !self.buffers[slot].mutable
            || value.shape() != &descriptor.descriptor.shape
            || value.dtype() != descriptor.descriptor.dtype
            || value.len() != descriptor.elements
            || value.native_dense_mut_ptr().is_none()
        {
            return Err(ReplayError::Corrupt(format!(
                "native workspace borrowed output {buffer} descriptor mismatch"
            )));
        }
        let bytes = tensor_bytes(value)?;
        let next_traffic_bytes = self
            .current_traffic
            .borrowed_recurrent_output_bytes
            .checked_add(bytes)
            .ok_or_else(|| {
                ReplayError::Descriptor("native borrowed recurrent output bytes overflow".into())
            })?;
        #[cfg(test)]
        let test_bytes = usize::try_from(bytes).map_err(|_| {
            ReplayError::Descriptor("native test traffic bytes exceed usize".into())
        })?;
        borrowed
            .slots
            .insert(slot, crate::cpu_jit::BorrowedJitBuffer::Write(value));
        self.current_traffic.borrowed_recurrent_output_bytes = next_traffic_bytes;
        #[cfg(test)]
        {
            self.borrowed_recurrent_output_bytes = self
                .borrowed_recurrent_output_bytes
                .saturating_add(test_bytes);
        }
        Ok(())
    }

    pub(super) fn finish_inputs(&self) -> Result<(), ReplayError> {
        if let Some((name, _)) = self.inputs.iter().find(|(_, slot)| !self.valid[*slot]) {
            return Err(ReplayError::Missing(name.clone()));
        }
        Ok(())
    }

    fn prepare_slot(
        &mut self,
        slot: usize,
        borrowed: &NativeReplayBindings<'_>,
    ) -> Result<(), ReplayError> {
        if self.valid.get(slot).copied().unwrap_or(false) {
            return Ok(());
        }
        let source = self
            .slots
            .get(slot)
            .and_then(|slot| slot.source.clone())
            .ok_or_else(|| ReplayError::Corrupt("native workspace value is unavailable".into()))?;
        let source_slot = match &source {
            SlotSource::Copy(source) | SlotSource::Affine { source, .. } => *source,
        };
        if !self.valid.get(source_slot).copied().unwrap_or(false) {
            return Err(ReplayError::Corrupt(
                "native workspace source is unavailable".into(),
            ));
        }
        #[cfg(test)]
        let affine = matches!(&source, SlotSource::Affine { .. });
        let copied = match borrowed.slots.get(&source_slot) {
            Some(binding) => {
                let target = self.buffers.get_mut(slot).ok_or_else(|| {
                    ReplayError::Corrupt("native workspace target is absent".into())
                })?;
                match source {
                    SlotSource::Copy(_) => Err(crate::JitError::InvalidBuffer(
                        "borrowed dense copy alias was not bound".into(),
                    )),
                    SlotSource::Affine { view, .. } => {
                        target.copy_affine_from_tensor(binding.tensor(), &view)
                    }
                }
            }
            None => {
                let (target, source_buffer) = two_buffers(&mut self.buffers, slot, source_slot)?;
                match source {
                    SlotSource::Copy(_) => target.copy_from_buffer(source_buffer),
                    SlotSource::Affine { view, .. } => {
                        target.copy_affine_from(source_buffer, &view)
                    }
                }
            }
        };
        copied.map_err(|error| ReplayError::Backend(error.to_string()))?;
        #[cfg(test)]
        if affine {
            let bytes = self
                .buffers
                .get(slot)
                .map(|buffer| buffer.bytes().len())
                .unwrap_or(0);
            self.affine_matmul_materialization_bytes = self
                .affine_matmul_materialization_bytes
                .saturating_add(bytes);
        }
        self.valid[slot] = true;
        Ok(())
    }

    fn execute_item(
        &mut self,
        index: usize,
        item: &ScheduleItem,
        backend: &CpuJitBackend,
        quantized: &BTreeMap<u64, crate::QuantizedTensorData>,
        prepared: &PreparedScheduleItem,
        borrowed: &mut NativeReplayBindings<'_>,
    ) -> Result<(), ReplayError> {
        let slot_count = self
            .items
            .get(index)
            .map(|item| item.slots.len())
            .ok_or_else(|| ReplayError::Corrupt("native workspace item is absent".into()))?;
        let output = self.items[index].output;
        if self.items[index].elided {
            if !self.valid.get(output).copied().unwrap_or(false) {
                return Err(ReplayError::Corrupt(
                    "elided native output is unavailable".into(),
                ));
            }
            return Ok(());
        }
        for offset in 0..slot_count {
            let slot = self.items[index].slots[offset];
            if slot != output {
                self.prepare_slot(slot, borrowed)?;
            }
        }
        self.initialize_output(index, borrowed)?;
        let execution = if borrowed.slots.is_empty() {
            backend
                .execute_prepared_schedule_item_in_workspace(
                    item,
                    &mut self.buffers,
                    &self.items[index].slots,
                    None,
                    quantized,
                    prepared,
                )
                .map(|_| ())
        } else {
            backend
                .execute_prepared_schedule_item_in_workspace(
                    item,
                    &mut self.buffers,
                    &self.items[index].slots,
                    Some(&mut borrowed.slots),
                    quantized,
                    prepared,
                )
                .map(|_| ())
        };
        execution.map_err(backend_error)?;
        self.current_traffic.executed_native_item_count = self
            .current_traffic
            .executed_native_item_count
            .checked_add(1)
            .ok_or_else(|| {
                ReplayError::Descriptor("native item execution count overflows".into())
            })?;
        self.valid[output] = true;
        Ok(())
    }

    fn initialize_output(
        &mut self,
        index: usize,
        borrowed: &mut NativeReplayBindings<'_>,
    ) -> Result<(), ReplayError> {
        let item = self.items.get(index).ok_or_else(|| {
            ReplayError::Corrupt("native workspace initialization item is absent".into())
        })?;
        let output_count = item.outputs.len();
        let output_initialization = item.output_initialization;
        for offset in 0..output_count {
            self.valid[self.items[index].outputs[offset]] = false;
        }
        if output_initialization == crate::cpu_jit::NativeOutputInitialization::FullyOverwritten {
            self.current_traffic.skipped_output_clear_count = self
                .current_traffic
                .skipped_output_clear_count
                .checked_add(output_count)
                .ok_or_else(|| {
                    ReplayError::Descriptor("native skipped output clear count overflows".into())
                })?;
            #[cfg(test)]
            {
                self.skipped_output_clear_count =
                    self.skipped_output_clear_count.saturating_add(output_count);
            }
            return Ok(());
        }
        #[cfg(test)]
        {
            self.output_clear_count = self.output_clear_count.saturating_add(1);
        }
        for offset in 0..output_count {
            let output = self.items[index].outputs[offset];
            match borrowed.slots.get_mut(&output) {
                Some(binding) => binding
                    .clear()
                    .map_err(|error| ReplayError::Backend(error.to_string()))?,
                None => self.buffers[output].clear(),
            }
        }
        Ok(())
    }

    fn initialize_output_action(
        &mut self,
        action: &WorkspaceOutputAction,
        borrowed: &mut NativeReplayBindings<'_>,
    ) -> Result<(), ReplayError> {
        for output in &action.outputs {
            self.valid[*output] = false;
        }
        if action.initialization == crate::cpu_jit::NativeOutputInitialization::FullyOverwritten {
            self.current_traffic.skipped_output_clear_count = self
                .current_traffic
                .skipped_output_clear_count
                .checked_add(action.outputs.len())
                .ok_or_else(|| {
                    ReplayError::Descriptor("native skipped output clear count overflows".into())
                })?;
            #[cfg(test)]
            {
                self.skipped_output_clear_count = self
                    .skipped_output_clear_count
                    .saturating_add(action.outputs.len());
            }
            return Ok(());
        }
        #[cfg(test)]
        {
            self.output_clear_count = self.output_clear_count.saturating_add(1);
        }
        for output in &action.outputs {
            match borrowed.slots.get_mut(output) {
                Some(binding) => binding
                    .clear()
                    .map_err(|error| ReplayError::Backend(error.to_string()))?,
                None => self.buffers[*output].clear(),
            }
        }
        Ok(())
    }

    fn execute_dispatch_segment(
        &mut self,
        segment: &WorkspaceDispatchSegment,
        capture: &CapturedSchedule,
        quantized: &BTreeMap<u64, crate::QuantizedTensorData>,
        borrowed: &mut NativeReplayBindings<'_>,
    ) -> Result<(), ReplayError> {
        #[cfg(test)]
        if let Some(index) = self.injected_dispatch_failure.filter(|injected| {
            segment
                .indices
                .iter()
                .any(|index| self.items[*index].logical_indices.contains(injected))
        }) {
            self.injected_dispatch_failure = None;
            let logical = capture.items[index].id;
            return Err(ReplayError::Backend(format!(
                "native schedule item {index} logical {logical}: injected dispatcher failure"
            )));
        }
        segment
            .dispatch
            .authenticate(quantized)
            .map_err(|failure| {
                map_dispatch_failure(&self.items, &segment.indices, capture, failure)
            })?;
        for slot in &segment.prerequisites {
            self.prepare_slot(*slot, borrowed)?;
        }
        for action in &segment.outputs {
            self.initialize_output_action(action, borrowed)?;
        }

        let next_dispatch_count = self
            .current_traffic
            .module_dispatch_count
            .checked_add(1)
            .ok_or_else(|| {
                ReplayError::Descriptor("native module dispatch count overflows".into())
            })?;
        let next_dispatched_items = self
            .current_traffic
            .module_dispatched_native_item_count
            .checked_add(segment.indices.len())
            .ok_or_else(|| {
                ReplayError::Descriptor("native module dispatched item count overflows".into())
            })?;
        let next_executed_items = self
            .current_traffic
            .executed_native_item_count
            .checked_add(segment.indices.len())
            .ok_or_else(|| {
                ReplayError::Descriptor("native item execution count overflows".into())
            })?;
        let execution = segment.dispatch.execute_authenticated(
            &segment.materializations,
            &mut self.buffers,
            (!borrowed.slots.is_empty()).then_some(&mut borrowed.slots),
            &mut self.dispatch_scratch,
        );
        let timing = execution.map_err(|failure| {
            map_dispatch_failure(&self.items, &segment.indices, capture, failure)
        })?;
        let next_dispatcher_wall_time = self
            .current_traffic
            .checked_native_dispatcher_total(timing.dispatcher_wall_time())?;
        for action in &segment.outputs {
            for output in &action.outputs {
                self.valid[*output] = true;
            }
        }
        for materialization in &segment.materializations {
            self.valid[materialization.target()] = true;
        }
        self.current_traffic.module_dispatch_count = next_dispatch_count;
        self.current_traffic.module_dispatched_native_item_count = next_dispatched_items;
        self.current_traffic.executed_native_item_count = next_executed_items;
        self.current_traffic.native_dispatcher_wall_time = next_dispatcher_wall_time;
        Ok(())
    }

    pub(super) fn execute_items(
        &mut self,
        capture: &CapturedSchedule,
        backend: &CpuJitBackend,
        quantized: &BTreeMap<u64, crate::QuantizedTensorData>,
        prepared: &[PreparedNativeDispatch],
        borrowed: &mut NativeReplayBindings<'_>,
    ) -> Result<(), ReplayError> {
        if prepared.len() != self.items.len() {
            return Err(ReplayError::Corrupt(
                "native dispatcher item inventory mismatch".into(),
            ));
        }
        let dispatch_tape = Arc::clone(&self.dispatch_tape);
        for step in dispatch_tape.iter() {
            let is_segment = matches!(step, WorkspaceDispatchStep::Segment(_));
            let index = match step {
                WorkspaceDispatchStep::Segment(segment) => {
                    *segment.indices.first().ok_or_else(|| {
                        ReplayError::Corrupt("native dispatcher segment is empty".into())
                    })?
                }
                WorkspaceDispatchStep::PerItem(index) => *index,
            };
            let logical_index = *self.items[index].logical_indices.first().ok_or_else(|| {
                ReplayError::Corrupt("native dispatch has no logical item".into())
            })?;
            let logical_item = capture.items.get(logical_index).ok_or_else(|| {
                ReplayError::Corrupt("native dispatch logical item is out of range".into())
            })?;
            #[cfg(test)]
            if !is_segment && self.injected_dispatch_failure == Some(logical_index) {
                self.injected_dispatch_failure = None;
                return Err(ReplayError::Backend(format!(
                    "native schedule item {logical_index} logical {}: injected dispatcher failure",
                    logical_item.id
                )));
            }
            let execution = match step {
                WorkspaceDispatchStep::Segment(segment) => {
                    self.execute_dispatch_segment(segment, capture, quantized, borrowed)
                }
                WorkspaceDispatchStep::PerItem(index) => {
                    let prepared_item = prepared[*index].item().ok_or_else(|| {
                        ReplayError::Corrupt("prepared native dispatch kind mismatch".into())
                    })?;
                    self.execute_item(
                        *index,
                        logical_item,
                        backend,
                        quantized,
                        prepared_item,
                        borrowed,
                    )
                }
            };
            execution.map_err(|error| match error {
                ReplayError::Backend(reason) if !is_segment => ReplayError::Backend(format!(
                    "native schedule item {logical_index} logical {}: {reason}",
                    logical_item.id
                )),
                ReplayError::Execute(reason) if !is_segment => ReplayError::Execute(format!(
                    "native schedule item {logical_index} logical {}: {reason}",
                    logical_item.id
                )),
                other => other,
            })?;
        }
        Ok(())
    }

    pub(super) fn materialize(
        &mut self,
        capture: &CapturedSchedule,
        borrowed: &NativeReplayBindings<'_>,
        selected: Option<&BTreeSet<u64>>,
    ) -> Result<ReplayValues, ReplayError> {
        let mut values = ReplayValues::default();
        for (buffer, slot, shape) in &self.egress {
            let wanted = selected.is_none_or(|selected| {
                selected.contains(buffer)
                    || capture.requested_passthroughs.iter().any(|alias| {
                        selected.contains(&(alias.requested.index() as u64))
                            && alias.source.index() as u64 == *buffer
                    })
            });
            if !wanted {
                continue;
            }
            if !self.valid.get(*slot).copied().unwrap_or(false) {
                return Err(ReplayError::Corrupt(format!(
                    "native workspace egress {buffer} is unavailable"
                )));
            }
            let value = match borrowed.slots.get(slot) {
                Some(binding) => binding.tensor().clone(),
                None => self.buffers[*slot]
                    .to_tensor(shape.clone())
                    .map_err(|error| ReplayError::Backend(error.to_string()))?,
            };
            values.insert_tensor(*buffer, value);
        }
        let aliases = capture
            .requested_passthroughs
            .iter()
            .filter(|alias| {
                selected.is_none_or(|selected| selected.contains(&(alias.requested.index() as u64)))
            })
            .cloned()
            .collect::<Vec<_>>();
        values.project_requested_aliases(&aliases)?;
        let mut materialized_egress_count = 0u64;
        let mut materialized_egress_bytes = 0u64;
        for requested in capture
            .requested
            .iter()
            .filter(|requested| selected.is_none_or(|selected| selected.contains(*requested)))
        {
            let value = values.tensor(*requested, "native requested egress")?;
            materialized_egress_count =
                materialized_egress_count.checked_add(1).ok_or_else(|| {
                    ReplayError::Descriptor("native materialized egress count overflows".into())
                })?;
            materialized_egress_bytes = materialized_egress_bytes
                .checked_add(tensor_bytes(value)?)
                .ok_or_else(|| {
                    ReplayError::Descriptor("native materialized egress bytes overflow".into())
                })?;
        }
        self.current_traffic.materialized_egress_count = materialized_egress_count;
        self.current_traffic.materialized_egress_bytes = materialized_egress_bytes;
        Ok(values)
    }

    pub(super) const fn traffic(&self) -> NativeReplayTraffic {
        self.current_traffic
    }

    #[cfg(test)]
    pub(super) const fn last_executed_native_item_count(&self) -> usize {
        self.current_traffic.executed_native_item_count
    }

    #[cfg(test)]
    pub(super) fn poison_outputs(&mut self, byte: u8) {
        for item in &self.items {
            if !item.elided {
                for output in &item.outputs {
                    self.buffers[*output].bytes_mut().fill(byte);
                    self.valid[*output] = false;
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) const fn last_module_dispatch_counts(&self) -> (usize, usize) {
        (
            self.current_traffic.module_dispatch_count,
            self.current_traffic.module_dispatched_native_item_count,
        )
    }

    #[cfg(test)]
    pub(super) fn inject_dispatch_failure(&mut self, index: usize) {
        self.injected_dispatch_failure = Some(index);
    }

    #[cfg(test)]
    pub(super) fn use_per_item_fallback(&mut self, index: usize) -> Result<(), ReplayError> {
        let item = self
            .items
            .get_mut(index)
            .ok_or_else(|| ReplayError::Corrupt("native dispatcher item is absent".into()))?;
        if item.elided || item.logical_indices.len() != 1 || item.dispatch.take().is_none() {
            return Err(ReplayError::Corrupt(
                "native dispatcher fallback item is not dispatchable".into(),
            ));
        }
        self.seal_dispatch_tape()
    }

    fn seal_dispatch_tape(&mut self) -> Result<(), ReplayError> {
        let mut steps = Vec::new();
        let mut segmentation = NativeDispatchSegmentation::default();
        let mut segment = Vec::new();
        let mut segment_slots = BTreeSet::new();
        let mut reached_module_anchors: Vec<usize> = Vec::new();
        for (index, item) in self.items.iter().enumerate() {
            if item.elided {
                // An authenticated retained owner or zero-domain output needs
                // no native call. Leaving it out of the tape also permits
                // dispatchable consumers on both sides to share one module
                // invocation.
                continue;
            }
            if item.dispatch.is_none() {
                if !segment.is_empty() {
                    steps.push(WorkspaceDispatchStep::Segment(seal_workspace_segment(
                        &self.items,
                        &self.slots,
                        std::mem::take(&mut segment),
                    )?));
                    segmentation.record(NativeDispatchSegmentEnd::NonDispatchBoundary);
                    segment_slots.clear();
                }
                steps.push(WorkspaceDispatchStep::PerItem(index));
                continue;
            }
            if !reached_module_anchors.iter().any(|anchor| {
                let reached = self.items[*anchor]
                    .dispatch
                    .as_ref()
                    .expect("dispatch-reached module anchor has a dispatcher");
                item.dispatch
                    .as_ref()
                    .expect("dispatchable workspace item has a dispatcher")
                    .shares_dispatcher_with(reached)
            }) {
                reached_module_anchors.push(index);
            }
            let crosses_module = segment.first().is_some_and(|first| {
                let first = self.items[*first]
                    .dispatch
                    .as_ref()
                    .expect("dispatchable segment entry has a dispatcher");
                !item
                    .dispatch
                    .as_ref()
                    .expect("dispatchable workspace item has a dispatcher")
                    .shares_dispatcher_with(first)
            });
            let output_aliases_slot = item
                .outputs
                .iter()
                .any(|output| segment_slots.contains(output));
            let split = dispatch_segment_split(crosses_module, output_aliases_slot);
            if let Some(split) = split {
                steps.push(WorkspaceDispatchStep::Segment(seal_workspace_segment(
                    &self.items,
                    &self.slots,
                    std::mem::take(&mut segment),
                )?));
                segmentation.record(split);
                segment_slots.clear();
            }
            segment.extend([index]);
            segment_slots.extend(item.slots.iter().copied());
        }
        if !segment.is_empty() {
            steps.push(WorkspaceDispatchStep::Segment(seal_workspace_segment(
                &self.items,
                &self.slots,
                segment,
            )?));
            segmentation.record(NativeDispatchSegmentEnd::Terminal);
        }
        segmentation.dispatch_reached_module_count = reached_module_anchors.len();
        let mut scratch_capacity = (0, 0, 0, 0);
        for step in &steps {
            let WorkspaceDispatchStep::Segment(segment) = step else {
                continue;
            };
            let affine_axes =
                segment
                    .materializations
                    .iter()
                    .try_fold(0usize, |count, action| {
                        count
                            .checked_add(action.affine_axis_count())
                            .ok_or_else(|| {
                                ReplayError::Descriptor(
                                    "native dispatch affine axis count overflows".into(),
                                )
                            })
                    })?;
            scratch_capacity.0 = scratch_capacity.0.max(segment.dispatch.entry_count());
            scratch_capacity.1 = scratch_capacity.1.max(segment.dispatch.pointer_count());
            scratch_capacity.2 = scratch_capacity.2.max(segment.materializations.len());
            scratch_capacity.3 = scratch_capacity.3.max(affine_axes);
        }
        let (entries, pointers, materializations, affine_axes) = scratch_capacity;
        if self.dispatch_tape.is_empty() {
            self.dispatch_scratch = crate::cpu_jit::JitScheduleDispatchScratch::with_capacity(
                entries,
                pointers,
                materializations,
                affine_axes,
            );
        } else {
            self.dispatch_scratch
                .ensure_capacity(entries, pointers, materializations, affine_axes);
        }
        self.dispatch_tape = Arc::from(steps);
        self.dispatch_segmentation = segmentation;
        #[cfg(test)]
        {
            self.dispatch_metadata_build_count =
                self.dispatch_metadata_build_count.saturating_add(1);
        }
        Ok(())
    }

    pub(super) const fn dispatch_segmentation(&self) -> NativeDispatchSegmentation {
        self.dispatch_segmentation
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> NativeReplayWorkspaceStats {
        let sealed_dispatch_step_count = self.dispatch_tape.len();
        let sealed_dispatch_segment_count = self
            .dispatch_tape
            .iter()
            .filter(|step| matches!(step, WorkspaceDispatchStep::Segment(_)))
            .count();
        let sealed_prerequisite_slot_count = self
            .dispatch_tape
            .iter()
            .filter_map(|step| match step {
                WorkspaceDispatchStep::Segment(segment) => Some(segment.prerequisites.len()),
                WorkspaceDispatchStep::PerItem(_) => None,
            })
            .sum();
        let sealed_derived_materialization_count = self
            .dispatch_tape
            .iter()
            .filter_map(|step| match step {
                WorkspaceDispatchStep::Segment(segment) => Some(segment.materializations.len()),
                WorkspaceDispatchStep::PerItem(_) => None,
            })
            .sum();
        NativeReplayWorkspaceStats {
            allocation_count: self.buffers.len(),
            input_import_count: self.input_import_count,
            borrowed_external_input_bytes: self.borrowed_external_input_bytes,
            intermediate_materialization_count: self.intermediate_materialization_count,
            borrowed_recurrent_input_bytes: self.borrowed_recurrent_input_bytes,
            borrowed_recurrent_output_bytes: self.borrowed_recurrent_output_bytes,
            retained_transpose_matmul_input_count: self.retained_transpose_matmul_input_count,
            affine_matmul_materialization_bytes: self.affine_matmul_materialization_bytes,
            output_clear_count: self.output_clear_count,
            skipped_output_clear_count: self.skipped_output_clear_count,
            sealed_dispatch_step_count,
            sealed_dispatch_segment_count,
            sealed_prerequisite_slot_count,
            sealed_derived_materialization_count,
            dispatch_metadata_build_count: self.dispatch_metadata_build_count,
            last_materialized_egress_count: self.current_traffic.materialized_egress_count,
            last_materialized_egress_bytes: self.current_traffic.materialized_egress_bytes,
            dispatch_scratch_capacity_growth_count: self.dispatch_scratch.capacity_growth_count(),
            dispatch_scratch_is_empty: self.dispatch_scratch.is_empty(),
        }
    }
}

fn seal_workspace_segment(
    items: &[WorkspaceItem],
    slots: &[WorkspaceSlot],
    indices: Vec<usize>,
) -> Result<WorkspaceDispatchSegment, ReplayError> {
    if indices.is_empty() {
        return Err(ReplayError::Corrupt(
            "native dispatcher segment is empty".into(),
        ));
    }
    let dispatches = indices
        .iter()
        .map(|index| {
            items
                .get(*index)
                .and_then(|item| item.dispatch.as_ref())
                .ok_or_else(|| {
                    ReplayError::Corrupt("native dispatcher segment entry is absent".into())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dispatch = PreparedScheduleDispatch::seal_segment(&dispatches).map_err(backend_error)?;
    let mut produced = BTreeSet::new();
    let mut materialized = BTreeSet::new();
    let mut seen_prerequisites = BTreeSet::new();
    let mut prerequisites = Vec::new();
    let mut materializations = Vec::new();
    let mut outputs = Vec::with_capacity(indices.len());
    for (entry, index) in indices.iter().enumerate() {
        let item = items.get(*index).ok_or_else(|| {
            ReplayError::Corrupt("native dispatcher segment item is absent".into())
        })?;
        for slot in item
            .slots
            .iter()
            .filter(|slot| !item.outputs.contains(slot))
        {
            if produced.contains(slot) || materialized.contains(slot) {
                continue;
            }
            if derived_slot_depends_on_outputs(*slot, slots, &produced)? {
                append_segment_materializations(
                    *slot,
                    entry,
                    slots,
                    &produced,
                    &mut materialized,
                    &mut BTreeSet::new(),
                    &mut materializations,
                )?;
            } else if seen_prerequisites.insert(*slot) {
                prerequisites.push(*slot);
            }
        }
        produced.extend(item.outputs.iter().copied());
        outputs.push(WorkspaceOutputAction {
            outputs: item.outputs.clone(),
            initialization: item.output_initialization,
        });
    }
    Ok(WorkspaceDispatchSegment {
        indices,
        prerequisites,
        materializations,
        outputs,
        dispatch,
    })
}

fn derived_slot_depends_on_outputs(
    slot: usize,
    slots: &[WorkspaceSlot],
    produced: &BTreeSet<usize>,
) -> Result<bool, ReplayError> {
    let mut current = slot;
    let mut visited = BTreeSet::new();
    loop {
        if produced.contains(&current) {
            return Ok(true);
        }
        if !visited.insert(current) {
            return Err(ReplayError::Corrupt(
                "native workspace derived slot cycle".into(),
            ));
        }
        let Some(source) = slots.get(current).and_then(|slot| slot.source.as_ref()) else {
            return Ok(false);
        };
        current = match source {
            SlotSource::Copy(source) | SlotSource::Affine { source, .. } => *source,
        };
    }
}

fn append_segment_materializations(
    slot: usize,
    before_entry: usize,
    slots: &[WorkspaceSlot],
    produced: &BTreeSet<usize>,
    materialized: &mut BTreeSet<usize>,
    visiting: &mut BTreeSet<usize>,
    actions: &mut Vec<crate::cpu_jit::NativeDispatchMaterialization>,
) -> Result<(), ReplayError> {
    if produced.contains(&slot) || materialized.contains(&slot) {
        return Ok(());
    }
    if !visiting.insert(slot) {
        return Err(ReplayError::Corrupt(
            "native workspace derived slot cycle".into(),
        ));
    }
    let planned = slots
        .get(slot)
        .ok_or_else(|| ReplayError::Corrupt("native workspace derived slot is absent".into()))?;
    let source = planned.source.as_ref().ok_or_else(|| {
        ReplayError::Corrupt("native workspace derived action has no source".into())
    })?;
    let source_slot = match source {
        SlotSource::Copy(source) | SlotSource::Affine { source, .. } => *source,
    };
    if !produced.contains(&source_slot) && !materialized.contains(&source_slot) {
        append_segment_materializations(
            source_slot,
            before_entry,
            slots,
            produced,
            materialized,
            visiting,
            actions,
        )?;
    }
    let source_key = slots
        .get(source_slot)
        .ok_or_else(|| ReplayError::Corrupt("native workspace action source is absent".into()))?;
    if source_key.key.descriptor.dtype != planned.key.descriptor.dtype {
        return Err(ReplayError::Corrupt(
            "native workspace action dtype differs".into(),
        ));
    }
    let action = match source {
        SlotSource::Copy(_) => {
            if source_key.key.elements != planned.key.elements {
                return Err(ReplayError::Corrupt(
                    "native workspace copy action shape differs".into(),
                ));
            }
            crate::cpu_jit::NativeDispatchMaterialization::copy(
                before_entry,
                source_slot,
                slot,
                planned.key.descriptor.dtype,
                planned.key.elements,
            )
        }
        SlotSource::Affine { view, .. } => {
            let source_elements = view
                .source_shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
            let logical_elements = view
                .logical_shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
            if source_key.key.elements != source_elements
                || planned.key.elements != logical_elements
            {
                return Err(ReplayError::Corrupt(
                    "native workspace affine action shape differs".into(),
                ));
            }
            crate::cpu_jit::NativeDispatchMaterialization::affine(
                before_entry,
                source_slot,
                slot,
                planned.key.descriptor.dtype,
                view,
            )
        }
    }
    .map_err(|error| ReplayError::Backend(error.to_string()))?;
    actions.push(action);
    materialized.insert(slot);
    visiting.remove(&slot);
    Ok(())
}

fn map_dispatch_failure(
    items: &[WorkspaceItem],
    indices: &[usize],
    capture: &CapturedSchedule,
    failure: PreparedScheduleDispatchFailure,
) -> ReplayError {
    let index = indices.get(failure.entry).copied().unwrap_or(usize::MAX);
    let logical_index = items
        .get(index)
        .and_then(|item| item.logical_indices.first())
        .copied()
        .unwrap_or(usize::MAX);
    let logical = items
        .get(index)
        .and_then(|item| item.logical_indices.first())
        .and_then(|logical| capture.items.get(*logical))
        .map(|item| item.id.to_string())
        .unwrap_or_else(|| "unknown".into());
    match backend_error(failure.error) {
        ReplayError::Backend(reason) => ReplayError::Backend(format!(
            "native schedule item {logical_index} logical {logical}: {reason}"
        )),
        other => other,
    }
}

fn tensor_bytes(value: &TensorData) -> Result<u64, ReplayError> {
    let bytes = value
        .len()
        .checked_mul(value.dtype().itemsize())
        .ok_or_else(|| ReplayError::Descriptor("native replay tensor bytes overflow".into()))?;
    u64::try_from(bytes)
        .map_err(|_| ReplayError::Descriptor("native replay tensor bytes exceed u64".into()))
}

fn two_buffers(
    buffers: &mut [crate::JitBuffer],
    target: usize,
    source: usize,
) -> Result<(&mut crate::JitBuffer, &crate::JitBuffer), ReplayError> {
    if target == source {
        return Err(ReplayError::Corrupt(
            "native workspace derived slot aliases its source".into(),
        ));
    }
    if target < source {
        let (left, right) = buffers.split_at_mut(source);
        Ok((&mut left[target], &right[0]))
    } else {
        let (left, right) = buffers.split_at_mut(target);
        Ok((&mut right[0], &left[source]))
    }
}

#[cfg(test)]
mod dispatch_segmentation_tests {
    use super::{
        BufferKey, NativeDispatchSegmentEnd, NativeReplayTraffic, SlotSource, WorkspaceSlot,
        append_segment_materializations, derived_slot_depends_on_outputs, dispatch_segment_split,
    };
    use crate::{AffineView, BufferDesc, DType, Shape};
    use std::collections::BTreeSet;
    use std::time::Duration;

    #[test]
    fn split_causes_are_mutually_exclusive_in_runtime_precedence_order() {
        assert_eq!(dispatch_segment_split(false, false), None);
        assert_eq!(
            dispatch_segment_split(false, true),
            Some(NativeDispatchSegmentEnd::OutputSlotAlias)
        );
        assert_eq!(
            dispatch_segment_split(true, true),
            Some(NativeDispatchSegmentEnd::ModuleChange)
        );
    }

    fn slot(buffer: u64, shape: Shape, source: Option<SlotSource>) -> WorkspaceSlot {
        let elements = shape.numel().unwrap();
        WorkspaceSlot {
            key: BufferKey {
                buffer,
                descriptor: BufferDesc {
                    id: buffer,
                    shape,
                    dtype: DType::F32,
                    bytes: elements * DType::F32.itemsize(),
                    alignment: DType::F32.itemsize(),
                    read_only: false,
                    view: None,
                },
                elements,
                direct_view: None,
            },
            source,
        }
    }

    #[test]
    fn typed_materializations_follow_producers_and_deduplicate_targets() {
        let reversed = AffineView {
            source_shape: Shape::new([5]),
            logical_shape: Shape::new([5]),
            strides: vec![-1],
            offset: 4,
        };
        let slots = vec![
            slot(0, Shape::new([5]), None),
            slot(1, Shape::new([5]), Some(SlotSource::Copy(0))),
            slot(
                2,
                Shape::new([5]),
                Some(SlotSource::Affine {
                    source: 1,
                    view: reversed,
                }),
            ),
        ];
        let produced = BTreeSet::from([0]);
        assert!(derived_slot_depends_on_outputs(2, &slots, &produced).unwrap());
        let mut materialized = BTreeSet::new();
        let mut actions = Vec::new();
        append_segment_materializations(
            2,
            3,
            &slots,
            &produced,
            &mut materialized,
            &mut BTreeSet::new(),
            &mut actions,
        )
        .unwrap();
        append_segment_materializations(
            2,
            4,
            &slots,
            &produced,
            &mut materialized,
            &mut BTreeSet::new(),
            &mut actions,
        )
        .unwrap();
        assert_eq!(actions.len(), 2);
        assert!(matches!(
            actions[0],
            crate::cpu_jit::NativeDispatchMaterialization::Copy {
                before_entry: 3,
                source: 0,
                target: 1,
                elements: 5,
                ..
            }
        ));
        assert!(matches!(
            &actions[1],
            crate::cpu_jit::NativeDispatchMaterialization::Affine {
                before_entry: 3,
                source: 1,
                target: 2,
                logical_elements: 5,
                axes,
                ..
            } if axes.len() == 1 && axes[0].reversed
        ));
    }

    #[test]
    fn external_and_recurrent_derived_sources_remain_segment_prerequisites() {
        let slots = vec![
            slot(0, Shape::new([5]), None),
            slot(1, Shape::new([5]), Some(SlotSource::Copy(0))),
        ];
        assert!(!derived_slot_depends_on_outputs(1, &slots, &BTreeSet::new()).unwrap());
    }

    #[test]
    fn affine_materialization_accepts_empty_broadcast_and_tail_geometry() {
        let empty = AffineView {
            source_shape: Shape::new([2, 3]),
            logical_shape: Shape::new([0, 3]),
            strides: vec![3, 1],
            offset: 0,
        };
        let broadcast = AffineView {
            source_shape: Shape::new([1]),
            logical_shape: Shape::new([5]),
            strides: vec![0],
            offset: 0,
        };
        let empty_action =
            crate::cpu_jit::NativeDispatchMaterialization::affine(0, 0, 1, DType::F32, &empty)
                .unwrap();
        let broadcast_action =
            crate::cpu_jit::NativeDispatchMaterialization::affine(1, 0, 1, DType::F32, &broadcast)
                .unwrap();
        assert!(matches!(
            empty_action,
            crate::cpu_jit::NativeDispatchMaterialization::Affine {
                logical_elements: 0,
                ref axes,
                ..
            } if axes.is_empty()
        ));
        assert!(matches!(
            broadcast_action,
            crate::cpu_jit::NativeDispatchMaterialization::Affine {
                logical_elements: 5,
                ref axes,
                ..
            } if axes.is_empty()
        ));
        assert!(
            crate::cpu_jit::NativeDispatchMaterialization::copy(2, 0, 1, DType::F32, 5,).is_ok()
        );
    }

    #[test]
    fn native_dispatcher_timing_accumulates_segments_and_rejects_overflow() {
        let mut traffic = NativeReplayTraffic::default();
        assert_eq!(traffic.native_dispatcher_wall_time, Duration::ZERO);
        traffic.native_dispatcher_wall_time = traffic
            .checked_native_dispatcher_total(Duration::from_nanos(3))
            .unwrap();
        traffic.native_dispatcher_wall_time = traffic
            .checked_native_dispatcher_total(Duration::from_nanos(5))
            .unwrap();
        assert_eq!(traffic.native_dispatcher_wall_time, Duration::from_nanos(8));
        traffic.native_dispatcher_wall_time = Duration::MAX;
        assert!(
            traffic
                .checked_native_dispatcher_total(Duration::from_nanos(1))
                .is_err()
        );
    }
}
