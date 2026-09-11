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
    elided: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeReplayTraffic {
    pub(crate) external_input_import_count: u64,
    pub(crate) external_input_import_bytes: u64,
    pub(crate) borrowed_recurrent_input_bytes: u64,
    pub(crate) borrowed_recurrent_output_bytes: u64,
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
    current_traffic: NativeReplayTraffic,
    #[cfg(test)]
    input_import_count: usize,
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
}

pub(super) struct NativeReplayBorrowedState<'a> {
    slots: BTreeMap<usize, crate::cpu_jit::BorrowedJitBuffer<'a>>,
}

impl<'a> NativeReplayBorrowedState<'a> {
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
    pub(crate) intermediate_materialization_count: usize,
    pub(crate) borrowed_recurrent_input_bytes: usize,
    pub(crate) borrowed_recurrent_output_bytes: usize,
    pub(crate) retained_transpose_matmul_input_count: usize,
    pub(crate) affine_matmul_materialization_bytes: usize,
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
            current_traffic: NativeReplayTraffic::default(),
            #[cfg(test)]
            input_import_count: 0,
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
        let borrowed = NativeReplayBorrowedState::new();
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
                elided: true,
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
        self.items.push(WorkspaceItem {
            slots,
            output: output_slot,
            elided: false,
        });
        Ok(())
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

    pub(super) fn begin(
        &mut self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<(), ReplayError> {
        self.begin_resolved();
        for (name, value) in provided {
            self.import_input(name, value)?;
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

    pub(super) fn borrow_recurrent_input<'a>(
        &mut self,
        name: &str,
        value: &'a TensorData,
        borrowed: &mut NativeReplayBorrowedState<'a>,
    ) -> Result<(), ReplayError> {
        let slot = self
            .inputs
            .iter()
            .find_map(|(input, slot)| (input == name).then_some(*slot))
            .ok_or_else(|| ReplayError::Extra(name.to_owned()))?;
        if self.valid[slot] || borrowed.slots.contains_key(&slot) {
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
                "native workspace borrowed input {name:?} descriptor mismatch"
            )));
        }
        borrowed
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
                borrowed
                    .slots
                    .insert(alias, crate::cpu_jit::BorrowedJitBuffer::Read(value));
                self.valid[alias] = true;
                bound_slots.insert(alias);
            }
        }
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

    pub(super) fn borrow_recurrent_output<'a>(
        &mut self,
        buffer: u64,
        value: &'a mut TensorData,
        borrowed: &mut NativeReplayBorrowedState<'a>,
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
        borrowed: &NativeReplayBorrowedState<'_>,
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
                        "borrowed recurrent copy alias was not bound".into(),
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

    pub(super) fn execute_item(
        &mut self,
        index: usize,
        item: &ScheduleItem,
        backend: &CpuJitBackend,
        quantized: &BTreeMap<u64, crate::QuantizedTensorData>,
        prepared: &PreparedScheduleItem,
        borrowed: &mut NativeReplayBorrowedState<'_>,
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
                    "elided native transpose source is unavailable".into(),
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
        self.valid[output] = false;
        // Fresh per-item JIT buffers have historically been zero-filled;
        // reductions and scatter-style kernels may rely on that initialization.
        match borrowed.slots.get_mut(&output) {
            Some(binding) => binding
                .clear()
                .map_err(|error| ReplayError::Backend(error.to_string()))?,
            None => self.buffers[output].clear(),
        }
        let execution = if borrowed.slots.is_empty() {
            backend.execute_prepared_schedule_item_in_workspace(
                item,
                &mut self.buffers,
                &self.items[index].slots,
                None,
                quantized,
                prepared,
            )
        } else {
            backend.execute_prepared_schedule_item_in_workspace(
                item,
                &mut self.buffers,
                &self.items[index].slots,
                Some(&mut borrowed.slots),
                quantized,
                prepared,
            )
        };
        execution.map_err(backend_error)?;
        self.valid[output] = true;
        Ok(())
    }

    pub(super) fn materialize(
        &self,
        capture: &CapturedSchedule,
        borrowed: &NativeReplayBorrowedState<'_>,
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
        Ok(values)
    }

    pub(super) const fn traffic(&self) -> NativeReplayTraffic {
        self.current_traffic
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> NativeReplayWorkspaceStats {
        NativeReplayWorkspaceStats {
            allocation_count: self.buffers.len(),
            input_import_count: self.input_import_count,
            intermediate_materialization_count: self.intermediate_materialization_count,
            borrowed_recurrent_input_bytes: self.borrowed_recurrent_input_bytes,
            borrowed_recurrent_output_bytes: self.borrowed_recurrent_output_bytes,
            retained_transpose_matmul_input_count: self.retained_transpose_matmul_input_count,
            affine_matmul_materialization_bytes: self.affine_matmul_materialization_bytes,
        }
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
