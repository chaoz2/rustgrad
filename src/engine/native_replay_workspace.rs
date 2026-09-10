//! Retained scratch storage for authenticated fixed-shape native replay.
use super::capture::{CapturedSchedule, ReplayError};
use super::captured_replay::{ReplayValues, backend_error};
use crate::backend::PreparedScheduleItem;
use crate::{CpuJitBackend, ScheduleItem, TensorData};
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Clone, Debug)]
struct WorkspaceItem {
    slots: Vec<usize>,
    output: usize,
}

/// Private scratch owned by one authenticated prepared native program.
/// Slots keep allocations, not authoritative replay bindings: every external
/// value is imported and every mutable derived value is invalidated per run.
pub(super) struct NativeReplayWorkspace {
    buffers: Vec<crate::JitBuffer>,
    slots: Vec<WorkspaceSlot>,
    items: Vec<WorkspaceItem>,
    owners: BTreeMap<u64, usize>,
    inputs: Vec<(String, usize)>,
    immutable: BTreeSet<usize>,
    egress: Vec<(u64, usize, crate::Shape)>,
    valid: Vec<bool>,
    #[cfg(test)]
    input_import_count: usize,
    #[cfg(test)]
    intermediate_materialization_count: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeReplayWorkspaceStats {
    pub(crate) allocation_count: usize,
    pub(crate) input_import_count: usize,
    pub(crate) intermediate_materialization_count: usize,
}

impl NativeReplayWorkspace {
    pub(super) fn new(
        capture: &CapturedSchedule,
        items: &[PreparedScheduleItem],
    ) -> Result<Self, ReplayError> {
        let mut workspace = Self {
            buffers: Vec::new(),
            slots: Vec::new(),
            items: Vec::with_capacity(items.len()),
            owners: BTreeMap::new(),
            inputs: Vec::with_capacity(capture.inputs.len()),
            immutable: BTreeSet::new(),
            egress: Vec::new(),
            valid: Vec::new(),
            #[cfg(test)]
            input_import_count: 0,
            #[cfg(test)]
            intermediate_materialization_count: 0,
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
        for (item, prepared) in capture.items.iter().zip(items) {
            workspace.add_item(item, prepared)?;
        }
        if capture.items.len() != items.len() {
            return Err(ReplayError::Corrupt(
                "native workspace item count mismatch".into(),
            ));
        }

        workspace.valid.resize(workspace.buffers.len(), false);
        let immutable = workspace.immutable.iter().copied().collect::<Vec<_>>();
        for slot in immutable {
            if workspace.slots[slot].source.is_none() {
                workspace.valid[slot] = true;
            } else {
                workspace.prepare_slot(slot)?;
            }
        }
        workspace.plan_egress(capture)?;
        Ok(workspace)
    }

    fn add_item(
        &mut self,
        item: &ScheduleItem,
        prepared: &PreparedScheduleItem,
    ) -> Result<(), ReplayError> {
        let output = item.primary_output();
        let output_elements = output
            .shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
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
            slots.push(self.resolve_input(item, abi, direct_matmul)?);
        }
        self.items.push(WorkspaceItem {
            slots,
            output: output_slot,
        });
        Ok(())
    }

    fn resolve_input(
        &mut self,
        item: &ScheduleItem,
        abi: &crate::cpu_jit::BufferAbi,
        direct_matmul: bool,
    ) -> Result<usize, ReplayError> {
        let binding = item
            .ordered_inputs()
            .iter()
            .find(|binding| binding.desc.id == abi.id)
            .ok_or_else(|| {
                ReplayError::Corrupt(format!("native workspace input {} has no binding", abi.id))
            })?;
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

    pub(super) fn begin(
        &mut self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<(), ReplayError> {
        self.valid.fill(false);
        for slot in &self.immutable {
            self.valid[*slot] = true;
        }
        for (name, slot) in &self.inputs {
            let value = provided
                .get(name)
                .ok_or_else(|| ReplayError::Missing(name.clone()))?;
            self.buffers[*slot]
                .copy_from_tensor(value)
                .map_err(|error| ReplayError::Backend(error.to_string()))?;
            self.valid[*slot] = true;
            #[cfg(test)]
            {
                self.input_import_count += 1;
            }
        }
        Ok(())
    }

    fn prepare_slot(&mut self, slot: usize) -> Result<(), ReplayError> {
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
        let (target, source_buffer) = two_buffers(&mut self.buffers, slot, source_slot)?;
        match source {
            SlotSource::Copy(_) => target.copy_from_buffer(source_buffer),
            SlotSource::Affine { view, .. } => target.copy_affine_from(source_buffer, &view),
        }
        .map_err(|error| ReplayError::Backend(error.to_string()))?;
        self.valid[slot] = true;
        Ok(())
    }

    pub(super) fn execute_item(
        &mut self,
        index: usize,
        item: &ScheduleItem,
        backend: &CpuJitBackend,
        quantized: &BTreeMap<u64, crate::QuantizedTensorData>,
        prepared: &PreparedScheduleItem,
    ) -> Result<(), ReplayError> {
        let slot_count = self
            .items
            .get(index)
            .map(|item| item.slots.len())
            .ok_or_else(|| ReplayError::Corrupt("native workspace item is absent".into()))?;
        let output = self.items[index].output;
        for offset in 0..slot_count {
            let slot = self.items[index].slots[offset];
            if slot != output {
                self.prepare_slot(slot)?;
            }
        }
        self.valid[output] = false;
        // Fresh per-item JIT buffers have historically been zero-filled;
        // reductions and scatter-style kernels may rely on that initialization.
        self.buffers[output].clear();
        backend
            .execute_prepared_schedule_item_in_workspace(
                item,
                &mut self.buffers,
                &self.items[index].slots,
                quantized,
                prepared,
            )
            .map_err(backend_error)?;
        self.valid[output] = true;
        Ok(())
    }

    pub(super) fn materialize(
        &self,
        capture: &CapturedSchedule,
    ) -> Result<ReplayValues, ReplayError> {
        let mut values = ReplayValues::default();
        for (buffer, slot, shape) in &self.egress {
            if !self.valid.get(*slot).copied().unwrap_or(false) {
                return Err(ReplayError::Corrupt(format!(
                    "native workspace egress {buffer} is unavailable"
                )));
            }
            values.insert_tensor(
                *buffer,
                self.buffers[*slot]
                    .to_tensor(shape.clone())
                    .map_err(|error| ReplayError::Backend(error.to_string()))?,
            );
        }
        values.project_requested_aliases(&capture.requested_passthroughs)?;
        Ok(values)
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> NativeReplayWorkspaceStats {
        NativeReplayWorkspaceStats {
            allocation_count: self.buffers.len(),
            input_import_count: self.input_import_count,
            intermediate_materialization_count: self.intermediate_materialization_count,
        }
    }
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
