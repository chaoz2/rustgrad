//! Owned reusable execution for one authenticated bounded-symbolic CPU body.
//!
//! A program keeps the captured body/schema immutable and compiles one native
//! kernel per schedule item. Invocation values are never part of native cache
//! identity. Every run first builds and validates the ordinary concrete
//! specialization, external inventory, and exact memory plan; only then may it
//! publish compiled kernels or allocate execution buffers.

use super::capture::{CapturedSchedule, ReplayError};
use super::captured_replay::{initial_values, validate_inputs};
use crate::{JitBuffer, JitKernel, MemoryPlan, TensorData};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

/// One owned invocation of a bounded symbolic program. Both namespaces are
/// intentionally explicit: tensor names bind captured external storage, while
/// symbolic names bind canonical checked I64 shape parameters.
#[derive(Clone, Debug, Default)]
pub struct SymbolicInvocation {
    symbols: BTreeMap<String, i64>,
    inputs: BTreeMap<String, TensorData>,
}

impl SymbolicInvocation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_symbol(mut self, name: impl Into<String>, value: i64) -> Self {
        self.symbols.insert(name.into(), value);
        self
    }

    pub fn with_input(mut self, name: impl Into<String>, value: TensorData) -> Self {
        self.inputs.insert(name.into(), value);
        self
    }

    pub fn symbols(&self) -> &BTreeMap<String, i64> {
        &self.symbols
    }

    pub fn inputs(&self) -> &BTreeMap<String, TensorData> {
        &self.inputs
    }
}

/// Backward-compatible CPU spelling for the shared symbolic invocation ABI.
pub type CpuSymbolicInvocation = SymbolicInvocation;

/// Authenticated immutable symbolic body shared by CPU and static-device
/// execution. Backend-specific rendering and preparation happen only after
/// [`Self::bind`] has validated the complete invocation.
#[derive(Clone)]
pub(crate) struct AuthenticatedSymbolicBody {
    capture: Arc<CapturedSchedule>,
    output_order: Vec<usize>,
}

pub(crate) struct AuthenticatedSymbolicInvocation {
    pub(crate) concrete: CapturedSchedule,
    pub(crate) canonical: Vec<(u64, i64)>,
    pub(crate) inputs: BTreeMap<String, TensorData>,
}

impl AuthenticatedSymbolicBody {
    pub(crate) fn new(
        capture: CapturedSchedule,
        output_order: Vec<usize>,
        diagnostic_name: &str,
    ) -> Result<Self, ReplayError> {
        crate::schedule::artifact::validate_capture(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        if let Some(position) = output_order
            .iter()
            .find(|position| **position >= capture.requested.len())
        {
            return Err(ReplayError::Descriptor(format!(
                "{diagnostic_name} output position {position} is absent"
            )));
        }
        capture.symbolic.as_ref().ok_or_else(|| {
            ReplayError::Symbolic(format!(
                "{diagnostic_name} program requires a symbolic capture"
            ))
        })?;
        if capture.items.iter().any(|item| !item.outputs.is_single()) {
            return Err(ReplayError::Unsupported(format!(
                "{diagnostic_name} program requires single-output schedule items"
            )));
        }
        if capture.items.iter().any(|item| {
            item.boundary.is_some()
                || item.is_effect()
                || matches!(item.kernel.operation(), crate::Operation::TensorGuard(_))
        }) {
            return Err(ReplayError::Unsupported(format!(
                "{diagnostic_name} program requires an effect-free pure value schedule"
            )));
        }
        if !capture.quantized_constants.is_empty() {
            return Err(ReplayError::Unsupported(format!(
                "{diagnostic_name} program does not own packed resources"
            )));
        }
        Ok(Self {
            capture: Arc::new(capture),
            output_order,
        })
    }

    pub(crate) fn capture(&self) -> &CapturedSchedule {
        &self.capture
    }

    pub(crate) fn schema(&self) -> &crate::engine::symbolic::SymbolicSchema {
        self.capture
            .symbolic
            .as_ref()
            .expect("authenticated symbolic body")
    }

    pub(crate) fn output_order(&self) -> &[usize] {
        &self.output_order
    }

    pub(crate) fn bind(
        &self,
        invocation: SymbolicInvocation,
    ) -> Result<AuthenticatedSymbolicInvocation, ReplayError> {
        let canonical = self.schema().canonical_bindings(&invocation.symbols)?;
        let concrete =
            super::symbolic::specialize_authenticated_capture(&self.capture, &canonical)?;
        validate_inputs(&concrete, &invocation.inputs)?;
        Ok(AuthenticatedSymbolicInvocation {
            concrete,
            canonical,
            inputs: invocation.inputs,
        })
    }
}

/// Successful invocation provenance. Binding values are reportable runtime
/// data, never part of `body_identity` or any `native_cache_keys` entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuSymbolicTrace {
    body_identity: u64,
    bindings: Vec<(u64, i64)>,
    compiled_now: bool,
    native_cache_keys: Vec<String>,
    peak_temporary_allocations: usize,
    peak_temporary_bytes: usize,
}

impl CpuSymbolicTrace {
    pub fn body_identity(&self) -> u64 {
        self.body_identity
    }

    pub fn bindings(&self) -> &[(u64, i64)] {
        &self.bindings
    }

    pub fn compiled_now(&self) -> bool {
        self.compiled_now
    }

    pub fn native_cache_keys(&self) -> &[String] {
        &self.native_cache_keys
    }

    pub fn peak_temporary_bytes(&self) -> usize {
        self.peak_temporary_bytes
    }

    pub fn peak_temporary_allocations(&self) -> usize {
        self.peak_temporary_allocations
    }
}

/// Detached ordered outputs from one successful symbolic invocation.
#[derive(Clone, Debug)]
pub struct CpuSymbolicResult {
    outputs: Vec<TensorData>,
    trace: CpuSymbolicTrace,
}

impl CpuSymbolicResult {
    pub fn outputs(&self) -> &[TensorData] {
        &self.outputs
    }

    pub fn into_outputs(self) -> Vec<TensorData> {
        self.outputs
    }

    pub fn into_parts(self) -> (Vec<TensorData>, CpuSymbolicTrace) {
        (self.outputs, self.trace)
    }

    pub fn trace(&self) -> &CpuSymbolicTrace {
        &self.trace
    }
}

struct PreparedProgram {
    kernels: Vec<JitKernel>,
    cache_keys: Vec<String>,
}

/// Immutable bounded-symbolic body with a lazily published native CPU plan.
/// Construction authenticates and renders the complete pure schedule without
/// compiling or allocating backend resources.
pub struct CpuSymbolicProgram {
    body: AuthenticatedSymbolicBody,
    rendered: Vec<crate::RenderedC>,
    prepared: Mutex<Option<Arc<PreparedProgram>>>,
    compile_count: AtomicUsize,
}

impl CpuSymbolicProgram {
    pub fn new(capture: CapturedSchedule) -> Result<Self, ReplayError> {
        let output_order = (0..capture.requested.len()).collect();
        Self::with_output_order(capture, output_order)
    }

    /// Creates a program whose detached result selects captured outputs by
    /// position. Repeated positions intentionally produce independent owned
    /// values without weakening the artifact's unique requested-ID invariant.
    pub fn with_output_order(
        capture: CapturedSchedule,
        output_order: Vec<usize>,
    ) -> Result<Self, ReplayError> {
        let body = AuthenticatedSymbolicBody::new(capture, output_order, "CPU symbolic")?;
        let capture = body.capture();
        let schema = body.schema();
        let rendered = capture
            .items
            .iter()
            .map(|item| {
                crate::cpu_jit::symbolic_runtime::render(capture.identity, item, schema)
                    .map_err(|error| ReplayError::Unsupported(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        preflight_runtime_abis(capture, &rendered, schema.parameters.len())?;
        Ok(Self {
            body,
            rendered,
            prepared: Mutex::new(None),
            compile_count: AtomicUsize::new(0),
        })
    }

    pub fn body_identity(&self) -> u64 {
        self.body.capture().identity
    }

    pub fn compile_count(&self) -> usize {
        self.compile_count.load(Ordering::Acquire)
    }

    pub fn inputs(&self) -> &[crate::ReplayInput] {
        &self.body.capture().inputs
    }

    pub fn parameters(&self) -> impl ExactSizeIterator<Item = &crate::SymbolicParameter> {
        self.body.schema().parameters().iter()
    }

    pub fn output_count(&self) -> usize {
        self.body.output_order().len()
    }

    pub fn output_order(&self) -> &[usize] {
        self.body.output_order()
    }

    pub fn run(&self, invocation: CpuSymbolicInvocation) -> Result<CpuSymbolicResult, ReplayError> {
        let schema = self.body.schema();
        // The exact order is observable: parameter namespace/range/guards,
        // concrete specialization, external descriptors, memory planning,
        // compilation publication, then private execution.
        let bound = self.body.bind(invocation)?;
        let memory = preflight_memory(&bound.concrete)?;
        preflight_runtime_abis(&bound.concrete, &self.rendered, schema.parameters.len())?;
        let (prepared, compiled_now) = self.prepare()?;
        let symbols = bound
            .canonical
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();
        let execution = execute(
            &bound.concrete,
            self.body.output_order(),
            &bound.inputs,
            &prepared,
            &symbols,
            &memory,
        )?;
        Ok(CpuSymbolicResult {
            outputs: execution.outputs,
            trace: CpuSymbolicTrace {
                body_identity: self.body.capture().identity,
                bindings: bound.canonical,
                compiled_now,
                native_cache_keys: prepared.cache_keys.clone(),
                peak_temporary_allocations: execution.peak_temporary_allocations,
                peak_temporary_bytes: execution.peak_temporary_bytes,
            },
        })
    }

    fn prepare(&self) -> Result<(Arc<PreparedProgram>, bool), ReplayError> {
        let mut prepared = self
            .prepared
            .lock()
            .map_err(|_| ReplayError::Backend("CPU symbolic program lock poisoned".into()))?;
        if let Some(program) = prepared.as_ref() {
            return Ok((program.clone(), false));
        }
        let kernels = self
            .rendered
            .iter()
            .map(JitKernel::load)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ReplayError::Backend(error.to_string()))?;
        let program = Arc::new(PreparedProgram {
            kernels,
            cache_keys: self
                .rendered
                .iter()
                .map(|rendered| rendered.cache_key.clone())
                .collect(),
        });
        // Publish only the complete prepared body. A failed later item may
        // leave ordinary process-wide compiler cache entries for earlier
        // kernels, but it cannot publish this program or increment its local
        // compilation count; a retry remains valid.
        *prepared = Some(program.clone());
        self.compile_count.fetch_add(1, Ordering::Release);
        Ok((program, true))
    }
}

fn preflight_memory(capture: &CapturedSchedule) -> Result<MemoryPlan, ReplayError> {
    let requested = capture
        .requested
        .iter()
        .copied()
        .chain(
            capture
                .requested_passthroughs
                .iter()
                .map(|alias| alias.source.index() as u64),
        )
        .collect::<BTreeSet<_>>();
    let temporaries = capture
        .items
        .iter()
        .flat_map(|item| item.outputs.iter())
        .filter(|output| !requested.contains(&output.id))
        .cloned()
        .collect::<Vec<_>>();
    MemoryPlan::from_temporaries(&capture.items, &temporaries, true)
        .map_err(|error| ReplayError::Descriptor(error.to_string()))
}

fn preflight_runtime_abis(
    capture: &CapturedSchedule,
    rendered: &[crate::RenderedC],
    symbol_count: usize,
) -> Result<(), ReplayError> {
    if rendered.len() != capture.items.len() {
        return Err(ReplayError::Corrupt(
            "CPU symbolic rendered item count mismatch".into(),
        ));
    }
    for (item, rendered) in capture.items.iter().zip(rendered) {
        let output = item.primary_output();
        if item
            .input_bindings
            .iter()
            .any(|binding| binding.desc.id == output.id)
        {
            return Err(ReplayError::Unsupported(format!(
                "CPU symbolic item {} aliases its output",
                item.id
            )));
        }
        let expected_buffers = item
            .input_bindings
            .iter()
            .map(|binding| binding.desc.id)
            .chain(std::iter::once(output.id))
            .collect::<BTreeSet<_>>();
        if rendered.abi.symbol_count != symbol_count
            || !rendered.abi.quantized_buffers.is_empty()
            || rendered.abi.buffers.len() != expected_buffers.len()
        {
            return Err(ReplayError::Corrupt(format!(
                "CPU symbolic item {} ABI inventory mismatch",
                item.id
            )));
        }
        let mut seen = BTreeSet::new();
        for buffer in &rendered.abi.buffers {
            if !seen.insert(buffer.id) {
                return Err(ReplayError::Corrupt(format!(
                    "CPU symbolic item {} repeats buffer {}",
                    item.id, buffer.id
                )));
            }
            let descriptor = if buffer.id == output.id {
                output
            } else {
                &item
                    .input_bindings
                    .iter()
                    .find(|binding| binding.desc.id == buffer.id)
                    .ok_or_else(|| {
                        ReplayError::Corrupt(format!(
                            "CPU symbolic item {} buffer {} is unbound",
                            item.id, buffer.id
                        ))
                    })?
                    .desc
            };
            let elements = descriptor
                .shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
            let mutable = buffer.id == output.id;
            if buffer.dtype != descriptor.dtype
                || buffer.mutable != mutable
                || elements > buffer.elements
                || descriptor.bytes
                    != elements
                        .checked_mul(descriptor.dtype.itemsize())
                        .ok_or_else(|| {
                            ReplayError::Descriptor("CPU symbolic ABI byte extent overflow".into())
                        })?
            {
                return Err(ReplayError::Descriptor(format!(
                    "CPU symbolic item {} buffer {} descriptor mismatch",
                    item.id, buffer.id
                )));
            }
        }
        if seen != expected_buffers {
            return Err(ReplayError::Corrupt(format!(
                "CPU symbolic item {} ABI coverage mismatch",
                item.id
            )));
        }
    }
    Ok(())
}

struct SymbolicExecution {
    outputs: Vec<TensorData>,
    peak_temporary_allocations: usize,
    peak_temporary_bytes: usize,
}

#[derive(Clone, Copy)]
enum TemporaryLocation {
    Slot(u64),
    Zero,
}

#[derive(Clone, Copy)]
struct TakenTemporary {
    buffer_id: u64,
    location: TemporaryLocation,
}

/// Per-invocation realization of one exact logical memory plan. Slots retain
/// private bytes after a value's liveness closes and can only be reassigned by
/// the plan; zero-byte temporaries remain per-value sentinels.
struct SymbolicBufferArena<'a> {
    assignments: BTreeMap<u64, &'a crate::TemporaryAllocation>,
    live: BTreeMap<u64, TemporaryLocation>,
    slots: BTreeMap<u64, JitBuffer>,
    zero_temporaries: BTreeMap<u64, JitBuffer>,
    allocated_slots: BTreeMap<u64, usize>,
}

impl<'a> SymbolicBufferArena<'a> {
    fn new(memory: &'a MemoryPlan) -> Self {
        Self {
            assignments: memory
                .temporaries
                .iter()
                .map(|assignment| (assignment.buffer_id, assignment))
                .collect(),
            live: BTreeMap::new(),
            slots: BTreeMap::new(),
            zero_temporaries: BTreeMap::new(),
            allocated_slots: BTreeMap::new(),
        }
    }

    fn take_input(
        &mut self,
        buffer_id: u64,
    ) -> Result<Option<(JitBuffer, TakenTemporary)>, ReplayError> {
        let Some(location) = self.live.get(&buffer_id).copied() else {
            return Ok(None);
        };
        let mut value = match location {
            TemporaryLocation::Slot(slot) => self.slots.remove(&slot).ok_or_else(|| {
                ReplayError::Corrupt(format!(
                    "CPU symbolic live temporary {buffer_id} has no slot"
                ))
            })?,
            TemporaryLocation::Zero => {
                self.zero_temporaries.remove(&buffer_id).ok_or_else(|| {
                    ReplayError::Corrupt(format!(
                        "CPU symbolic zero temporary {buffer_id} is absent"
                    ))
                })?
            }
        };
        value.mutable = false;
        Ok(Some((
            value,
            TakenTemporary {
                buffer_id,
                location,
            },
        )))
    }

    fn restore_input(
        &mut self,
        temporary: TakenTemporary,
        mut value: JitBuffer,
    ) -> Result<(), ReplayError> {
        value.mutable = true;
        let old = match temporary.location {
            TemporaryLocation::Slot(slot) => self.slots.insert(slot, value),
            TemporaryLocation::Zero => self.zero_temporaries.insert(temporary.buffer_id, value),
        };
        if old.is_some() {
            return Err(ReplayError::Corrupt(format!(
                "CPU symbolic temporary {} was restored twice",
                temporary.buffer_id
            )));
        }
        Ok(())
    }

    fn output_buffer(
        &mut self,
        output: &crate::BufferDesc,
        dtype: crate::DType,
        elements: usize,
    ) -> Result<JitBuffer, ReplayError> {
        let Some(assignment) = self.assignments.get(&output.id).copied() else {
            return Ok(JitBuffer::zeroed(dtype, elements, true));
        };
        let Some(slot) = assignment.allocation_id else {
            return Ok(JitBuffer::zeroed(dtype, elements, true));
        };
        let value = match self.allocated_slots.entry(slot) {
            std::collections::btree_map::Entry::Occupied(_) => {
                self.slots.remove(&slot).ok_or_else(|| {
                    ReplayError::Corrupt(format!(
                        "CPU symbolic temporary slot {slot} is still live"
                    ))
                })?
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(assignment.bytes);
                JitBuffer::zeroed(dtype, elements, true)
            }
        };
        if value.dtype != dtype || value.elements != elements {
            return Err(ReplayError::Corrupt(
                "CPU symbolic reused slot descriptor mismatch".into(),
            ));
        }
        Ok(value)
    }

    fn publish_output(
        &mut self,
        output: &crate::BufferDesc,
        mut value: JitBuffer,
    ) -> Result<Option<TensorData>, ReplayError> {
        let Some(assignment) = self.assignments.get(&output.id).copied() else {
            return value
                .into_tensor(output.shape.clone())
                .map(Some)
                .map_err(|error| ReplayError::Descriptor(error.to_string()));
        };
        value.mutable = true;
        let location = match assignment.allocation_id {
            Some(slot) => {
                if self.slots.insert(slot, value).is_some() {
                    return Err(ReplayError::Corrupt(format!(
                        "CPU symbolic output slot {slot} is already occupied"
                    )));
                }
                TemporaryLocation::Slot(slot)
            }
            None => {
                if self.zero_temporaries.insert(output.id, value).is_some() {
                    return Err(ReplayError::Corrupt(format!(
                        "CPU symbolic zero output {} is already live",
                        output.id
                    )));
                }
                TemporaryLocation::Zero
            }
        };
        if self.live.insert(output.id, location).is_some() {
            return Err(ReplayError::Corrupt(format!(
                "CPU symbolic temporary {} was produced twice",
                output.id
            )));
        }
        Ok(None)
    }

    fn release_after(&mut self, memory: &MemoryPlan, item: u64) -> Result<(), ReplayError> {
        for released in memory
            .temporaries
            .iter()
            .filter(|temporary| temporary.last_consumer == item)
        {
            let location = self.live.remove(&released.buffer_id).ok_or_else(|| {
                ReplayError::Corrupt(format!(
                    "CPU symbolic released temporary {} is absent",
                    released.buffer_id
                ))
            })?;
            match location {
                TemporaryLocation::Slot(slot) => {
                    if !self.slots.contains_key(&slot) {
                        return Err(ReplayError::Corrupt(format!(
                            "CPU symbolic released temporary {} has no slot",
                            released.buffer_id,
                        )));
                    }
                }
                TemporaryLocation::Zero => {
                    self.zero_temporaries
                        .remove(&released.buffer_id)
                        .ok_or_else(|| {
                            ReplayError::Corrupt(format!(
                                "CPU symbolic released zero temporary {} is absent",
                                released.buffer_id
                            ))
                        })?;
                }
            }
        }
        Ok(())
    }

    fn finish(self, memory: &MemoryPlan) -> Result<(usize, usize), ReplayError> {
        if !self.live.is_empty() || !self.zero_temporaries.is_empty() {
            return Err(ReplayError::Corrupt(
                "CPU symbolic temporary liveness did not close".into(),
            ));
        }
        if self.slots.keys().ne(self.allocated_slots.keys()) {
            return Err(ReplayError::Corrupt(
                "CPU symbolic physical slot inventory did not close".into(),
            ));
        }
        let bytes = self
            .allocated_slots
            .values()
            .try_fold(0usize, |total, bytes| {
                total.checked_add(*bytes).ok_or_else(|| {
                    ReplayError::Descriptor("CPU symbolic peak bytes overflow".into())
                })
            })?;
        if self.allocated_slots.len() != memory.peak_allocations || bytes != memory.peak_bytes {
            return Err(ReplayError::Corrupt(
                "CPU symbolic allocation plan was not realized exactly".into(),
            ));
        }
        Ok((self.allocated_slots.len(), bytes))
    }
}

fn execute(
    capture: &CapturedSchedule,
    output_order: &[usize],
    inputs: &BTreeMap<String, TensorData>,
    prepared: &PreparedProgram,
    symbols: &[i64],
    memory: &MemoryPlan,
) -> Result<SymbolicExecution, ReplayError> {
    if prepared.kernels.len() != capture.items.len() {
        return Err(ReplayError::Corrupt(
            "CPU symbolic prepared item count mismatch".into(),
        ));
    }
    let mut values = initial_values(capture, inputs)?;
    let mut arena = SymbolicBufferArena::new(memory);
    for (item, kernel) in capture.items.iter().zip(&prepared.kernels) {
        let output = item.primary_output();
        let mut buffers = Vec::with_capacity(kernel.abi().buffers.len());
        let mut exact_elements = Vec::with_capacity(kernel.abi().buffers.len());
        let mut temporary_inputs = Vec::with_capacity(kernel.abi().buffers.len());
        for buffer in &kernel.abi().buffers {
            if buffer.id == output.id {
                let elements = output
                    .shape
                    .numel()
                    .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
                let output_buffer = arena.output_buffer(output, buffer.dtype, elements)?;
                buffers.push(output_buffer);
                exact_elements.push(elements);
                temporary_inputs.push(None);
            } else if let Some((value, temporary)) = arena.take_input(buffer.id)? {
                exact_elements.push(value.elements);
                buffers.push(value);
                temporary_inputs.push(Some(temporary));
            } else {
                let value = values.tensor(buffer.id, "CPU symbolic input")?;
                buffers.push(JitBuffer::from_tensor(value, false));
                exact_elements.push(value.len());
                temporary_inputs.push(None);
            }
        }
        kernel
            .call_symbolic(&mut buffers, symbols, &exact_elements)
            .map_err(|error| ReplayError::Execute(error.to_string()))?;
        let mut output_buffer = None;
        for ((descriptor, buffer), temporary) in kernel
            .abi()
            .buffers
            .iter()
            .zip(buffers)
            .zip(temporary_inputs)
        {
            if descriptor.id == output.id {
                output_buffer = Some(buffer);
            } else if let Some(temporary) = temporary {
                arena.restore_input(temporary, buffer)?;
            }
        }
        let output_buffer = output_buffer
            .ok_or_else(|| ReplayError::Corrupt("CPU symbolic output ABI is absent".into()))?;
        if let Some(value) = arena.publish_output(output, output_buffer)? {
            values.insert_tensor(output.id, value);
        }
        arena.release_after(memory, item.id)?;
    }
    let requested = output_order
        .iter()
        .map(|position| capture.requested[*position])
        .collect::<Vec<_>>();
    values.project_requested_aliases(&capture.requested_passthroughs)?;
    let outputs = values.requested(&requested)?;
    let (peak_temporary_allocations, peak_temporary_bytes) = arena.finish(memory)?;
    Ok(SymbolicExecution {
        outputs,
        peak_temporary_allocations,
        peak_temporary_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend, CapturedSchedule, CpuBackend, DType, Graph, Scalar, Shape, SymbolicCaptureSpec,
        SymbolicExpr, SymbolicShape,
        nn::{Mode, ModeModuleForward, Module, RMSNorm, TransformerBlock},
    };
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    fn affine_family(
        rows: usize,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [rows, 4], DType::F32);
        let bias = graph.input_dtype("bias", [1, 4], DType::F32);
        let shifted = graph.add(input, bias).unwrap();
        let reshaped = graph.reshape(shifted, [rows, 2, 2]).unwrap();
        let transposed = graph.permute(reshaped, [0, 2, 1]).unwrap();
        let contiguous = graph.contiguous(transposed).unwrap();
        let matrix = graph.reshape(contiguous, [rows, 4]).unwrap();
        let reduced = graph
            .reduce(matrix, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let input_values = TensorData::from_scalars(
            [rows, 4],
            DType::F32,
            (0..rows * 4).map(|index| Scalar::F(index as f64 * 0.25 - 1.0)),
        )
        .unwrap();
        let bias_values = TensorData::new([1, 4], vec![0.5, -0.25, 1.0, -1.5]).unwrap();
        (
            graph,
            input,
            contiguous,
            reduced,
            BTreeMap::from([("input".into(), input_values), ("bias".into(), bias_values)]),
        )
    }

    fn invocation(
        symbols: impl IntoIterator<Item = (&'static str, i64)>,
        inputs: BTreeMap<String, TensorData>,
    ) -> CpuSymbolicInvocation {
        let mut invocation = CpuSymbolicInvocation::new();
        for (name, value) in symbols {
            invocation = invocation.with_symbol(name, value);
        }
        for (name, value) in inputs {
            invocation = invocation.with_input(name, value);
        }
        invocation
    }

    fn projected_family(
        tokens: usize,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2, tokens, 2], DType::F32);
        let producer = graph.square(input).unwrap();
        let permuted = graph.permute(producer, [0, 2, 1, 3]).unwrap();
        let projected = graph.reshape(permuted, [1, tokens, 4]).unwrap();
        let output = graph.relu(projected).unwrap();
        let reduced = graph
            .reduce_with_output_dtype(
                output,
                crate::ReduceKind::Sum,
                Some(vec![2]),
                false,
                DType::F32,
            )
            .unwrap();
        let values = TensorData::new(
            [1, 2, tokens, 2],
            (0..tokens * 4)
                .map(|index| index as f32 * 0.25 - 1.0)
                .collect(),
        )
        .unwrap();
        (
            graph,
            input,
            output,
            reduced,
            BTreeMap::from([("input".into(), values)]),
        )
    }

    #[test]
    fn owned_program_authenticates_and_specializes_projected_permute_reshape() {
        let tokens = SymbolicExpr::variable("tokens", 0, 5).unwrap();
        let (template, input, output, reduced, template_inputs) = projected_family(3);
        let schedule = crate::schedule_many(&template, &[output, reduced]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[output, reduced],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![
                    1usize.into(),
                    2usize.into(),
                    tokens.clone().into(),
                    2usize.into(),
                ]),
            )])),
            &BTreeMap::from([("tokens".into(), 3)]),
        )
        .unwrap();
        let encoded = capture.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), encoded);
        assert!(!decoded.symbolic.as_ref().unwrap().projected.is_empty());

        let mut tampered = decoded.clone();
        tampered
            .symbolic
            .as_mut()
            .unwrap()
            .projected
            .values_mut()
            .next()
            .unwrap()
            .expression =
            crate::projected_index::ProjectedExpr::Constant(SymbolicExpr::constant(0));
        tampered.identity = 0;
        tampered.identity = crate::schedule::artifact::identity(&tampered).unwrap();
        assert!(matches!(
            CpuSymbolicProgram::new(tampered),
            Err(ReplayError::Corrupt(_)) | Err(ReplayError::Symbolic(_))
        ));

        let mut vanishing_oob = decoded.clone();
        let map = vanishing_oob
            .symbolic
            .as_mut()
            .unwrap()
            .projected
            .values_mut()
            .next()
            .unwrap();
        map.expression = crate::projected_index::ProjectedExpr::binary(
            crate::uop::Binary::Add,
            map.expression.clone(),
            crate::projected_index::ProjectedExpr::Constant(
                (tokens - SymbolicExpr::constant(3)) * SymbolicExpr::constant(1_000),
            ),
        )
        .unwrap();
        vanishing_oob.identity = 0;
        vanishing_oob.identity = crate::schedule::artifact::identity(&vanishing_oob).unwrap();
        assert!(CpuSymbolicProgram::new(vanishing_oob).is_err());

        let mut missing_sidecar = decoded.clone();
        missing_sidecar.symbolic.as_mut().unwrap().projected.clear();
        missing_sidecar.identity = 0;
        missing_sidecar.identity = crate::schedule::artifact::identity(&missing_sidecar).unwrap();
        assert!(CpuSymbolicProgram::new(missing_sidecar).is_err());

        let mut wrong_ordinal = decoded.clone();
        let schema = wrong_ordinal.symbolic.as_mut().unwrap();
        let (key, value) = schema.projected.pop_first().unwrap();
        schema
            .projected
            .insert((key.0, key.1.checked_add(1).unwrap()), value);
        wrong_ordinal.identity = 0;
        wrong_ordinal.identity = crate::schedule::artifact::identity(&wrong_ordinal).unwrap();
        assert!(CpuSymbolicProgram::new(wrong_ordinal).is_err());

        let one_tokens = SymbolicExpr::variable("tokens", 0, 5).unwrap();
        let (one_graph, one_input, one_output, one_reduced, _) = projected_family(1);
        let one_schedule = crate::schedule_many(&one_graph, &[one_output, one_reduced]).unwrap();
        let one_capture = CapturedSchedule::capture_symbolic(
            &one_graph,
            &one_schedule,
            &[one_output, one_reduced],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                one_input,
                SymbolicShape::new(vec![
                    1usize.into(),
                    2usize.into(),
                    one_tokens.into(),
                    2usize.into(),
                ]),
            )])),
            &BTreeMap::from([("tokens".into(), 1)]),
        )
        .unwrap();
        assert!(!one_capture.symbolic.as_ref().unwrap().projected.is_empty());
        CpuSymbolicProgram::new(one_capture).unwrap();

        let zero_tokens = SymbolicExpr::variable("tokens", 0, 5).unwrap();
        let (zero_graph, zero_input, zero_output, zero_reduced, _) = projected_family(0);
        let zero_schedule =
            crate::schedule_many(&zero_graph, &[zero_output, zero_reduced]).unwrap();
        assert!(
            CapturedSchedule::capture_symbolic(
                &zero_graph,
                &zero_schedule,
                &[zero_output, zero_reduced],
                &SymbolicCaptureSpec::new(BTreeMap::from([(
                    zero_input,
                    SymbolicShape::new(vec![
                        1usize.into(),
                        2usize.into(),
                        zero_tokens.into(),
                        2usize.into(),
                    ]),
                )])),
                &BTreeMap::from([("tokens".into(), 0)]),
            )
            .is_err()
        );

        let program = CpuSymbolicProgram::new(decoded).unwrap();
        for tokens in [0usize, 1, 3, 5] {
            let (oracle_graph, _, oracle_output, oracle_reduced, inputs) = projected_family(tokens);
            let oracle_bindings = inputs.clone().into_iter().collect::<HashMap<_, _>>();
            let expected_output = CpuBackend
                .execute(&oracle_graph, oracle_output, &oracle_bindings)
                .unwrap();
            let expected_reduced = CpuBackend
                .execute(&oracle_graph, oracle_reduced, &oracle_bindings)
                .unwrap();
            let result = program
                .run(invocation([("tokens", tokens as i64)], inputs))
                .unwrap();
            assert_eq!(
                result.outputs()[0].to_le_bytes().unwrap(),
                expected_output.to_le_bytes().unwrap()
            );
            assert_eq!(
                result.outputs()[1].to_le_bytes().unwrap(),
                expected_reduced.to_le_bytes().unwrap()
            );
        }
        assert_eq!(program.compile_count(), 1);

        assert!(
            program
                .run(invocation([("tokens", 3)], template_inputs))
                .is_ok()
        );
    }

    #[test]
    fn projected_ordinals_distinguish_two_maps_of_one_source() {
        let tokens = SymbolicExpr::variable("tokens", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2, 3, 2], DType::F32);
        let producer = graph.square(input).unwrap();
        let first = graph.permute(producer, [0, 2, 1, 3]).unwrap();
        let first = graph.reshape(first, [1, 3, 4]).unwrap();
        let second = graph.permute(producer, [0, 2, 1, 3]).unwrap();
        let second = graph
            .stride(
                second,
                [
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                ],
            )
            .unwrap();
        let second = graph.reshape(second, [1, 3, 4]).unwrap();
        let output = graph.add(first, second).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![
                    1usize.into(),
                    2usize.into(),
                    tokens.into(),
                    2usize.into(),
                ]),
            )])),
            &BTreeMap::from([("tokens".into(), 3)]),
        )
        .unwrap();
        let projected = &capture.symbolic.as_ref().unwrap().projected;
        assert_eq!(projected.len(), 2);
        let keys = projected.keys().copied().collect::<Vec<_>>();
        assert_eq!(keys[0].0, keys[1].0);
        assert_ne!(keys[0].1, keys[1].1);
        let program = CpuSymbolicProgram::new(capture).unwrap();
        let values = TensorData::new(
            [1, 2, 3, 2],
            (0..12).map(|value| value as f32 - 5.0).collect(),
        )
        .unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("input".into(), values.clone())]),
            )
            .unwrap();
        let result = program
            .run(invocation(
                [("tokens", 3)],
                BTreeMap::from([("input".into(), values)]),
            ))
            .unwrap();
        assert_eq!(
            result.outputs()[0].to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap()
        );

        let empty = TensorData::new([1, 2, 0, 2], Vec::<f32>::new()).unwrap();
        let result = program
            .run(invocation(
                [("tokens", 0)],
                BTreeMap::from([("input".into(), empty)]),
            ))
            .unwrap();
        assert_eq!(result.outputs()[0].shape(), &Shape::from([1, 0, 4]));
        assert!(result.outputs()[0].is_empty());
    }

    #[test]
    fn projected_reduction_preserves_correlated_square_geometry() {
        let extent = SymbolicExpr::variable("extent", 1, 3).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let producer = graph.square(input).unwrap();
        let transposed = graph.permute(producer, [1, 0]).unwrap();
        let flattened = graph.reshape(transposed, [4]).unwrap();
        let output = graph
            .reduce_with_output_dtype(
                flattened,
                crate::ReduceKind::Sum,
                Some(vec![0]),
                false,
                DType::F32,
            )
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.clone().into(), extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap();
        let schema = capture.symbolic.as_ref().unwrap();
        assert_eq!(schema.projected.len(), 1);
        let projected_item = schema.projected.keys().next().unwrap().0;
        assert!(matches!(
            schema.item_domains.get(&projected_item),
            Some(crate::engine::symbolic::SymbolicItemDomain::Reduction { .. })
        ));
        let program = CpuSymbolicProgram::new(capture).unwrap();
        for extent in [1usize, 2, 3] {
            let values = TensorData::new(
                [extent, extent],
                (0..extent * extent)
                    .map(|index| index as f32 * 0.5 - 1.0)
                    .collect(),
            )
            .unwrap();
            let mut oracle = Graph::new();
            let oracle_input = oracle.input_dtype("input", [extent, extent], DType::F32);
            let oracle_producer = oracle.square(oracle_input).unwrap();
            let oracle_transposed = oracle.permute(oracle_producer, [1, 0]).unwrap();
            let oracle_flattened = oracle
                .reshape(oracle_transposed, [extent * extent])
                .unwrap();
            let oracle_output = oracle
                .reduce_with_output_dtype(
                    oracle_flattened,
                    crate::ReduceKind::Sum,
                    Some(vec![0]),
                    false,
                    DType::F32,
                )
                .unwrap();
            let bindings = HashMap::from([("input".into(), values.clone())]);
            let expected = CpuBackend
                .execute(&oracle, oracle_output, &bindings)
                .unwrap();
            let actual = program
                .run(invocation(
                    [("extent", extent as i64)],
                    BTreeMap::from([("input".into(), values)]),
                ))
                .unwrap();
            assert_eq!(
                actual.outputs()[0].to_le_bytes().unwrap(),
                expected.to_le_bytes().unwrap()
            );
        }
        assert_eq!(program.compile_count(), 1);
    }

    #[test]
    fn owned_program_reuses_one_body_for_affine_contiguous_and_reduction() {
        let rows = SymbolicExpr::variable("rows", 0, 8).unwrap();
        let (template, input, contiguous, reduced, template_inputs) = affine_family(3);
        let schedule = crate::schedule_many(&template, &[contiguous, reduced]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[contiguous, reduced],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![rows.into(), 4usize.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 3)]),
        )
        .unwrap();
        let encoded = capture.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), encoded);
        let schema = decoded.symbolic.as_ref().unwrap();
        assert!(decoded.items.iter().all(|item| {
            item.inputs
                .iter()
                .chain(std::iter::once(item.primary_output()))
                .all(|descriptor| schema.buffer_shapes.contains_key(&descriptor.id))
        }));
        let mut tampered = decoded.clone();
        tampered.identity ^= 1;
        assert!(matches!(
            CpuSymbolicProgram::new(tampered),
            Err(ReplayError::Corrupt(_))
        ));
        let mut recomputed_tamper = decoded.clone();
        recomputed_tamper
            .symbolic
            .as_mut()
            .unwrap()
            .buffer_shapes
            .insert(
                input.index() as u64,
                SymbolicShape::new(vec![7usize.into(), 4usize.into()]),
            );
        recomputed_tamper.identity = 0;
        recomputed_tamper.identity =
            crate::schedule::artifact::identity(&recomputed_tamper).unwrap();
        assert!(matches!(
            CpuSymbolicProgram::new(recomputed_tamper),
            Err(ReplayError::Corrupt(_))
        ));
        let mut view_tamper = decoded.clone();
        let view = view_tamper
            .symbolic
            .as_mut()
            .unwrap()
            .views
            .values_mut()
            .next()
            .expect("affine/reduction family carries one authenticated view");
        view.offset = view.offset.clone() + SymbolicExpr::constant(1);
        view_tamper.identity = 0;
        view_tamper.identity = crate::schedule::artifact::identity(&view_tamper).unwrap();
        assert!(matches!(
            CpuSymbolicProgram::new(view_tamper),
            Err(ReplayError::Corrupt(_))
        ));
        let program = CpuSymbolicProgram::with_output_order(decoded, vec![0, 1, 1]).unwrap();
        assert_eq!(program.compile_count(), 0);
        assert_eq!(program.output_count(), 3);
        assert_eq!(
            program
                .inputs()
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec!["bias", "input"]
        );
        let parameters = program.parameters().collect::<Vec<_>>();
        assert_eq!(parameters.len(), 1);
        assert_eq!(parameters[0].variable().name(), "rows");
        assert_eq!(parameters[0].variable().bounds(), (0, 8));

        assert!(matches!(
            program.run(invocation([], template_inputs.clone())),
            Err(ReplayError::Missing(_))
        ));
        assert!(matches!(
            program.run(invocation(
                [("rows", 3), ("unexpected", 1)],
                template_inputs.clone()
            )),
            Err(ReplayError::Extra(_))
        ));
        assert!(matches!(
            program.run(invocation([("rows", 9)], template_inputs.clone())),
            Err(ReplayError::Symbolic(_))
        ));
        assert!(matches!(
            program.run(invocation([("rows", -1)], template_inputs.clone())),
            Err(ReplayError::Symbolic(_))
        ));
        let mut wrong_dtype = template_inputs.clone();
        wrong_dtype.insert(
            "input".into(),
            TensorData::from_scalars([3, 4], DType::I32, [Scalar::I(0); 12]).unwrap(),
        );
        assert!(matches!(
            program.run(invocation([("rows", 3)], wrong_dtype)),
            Err(ReplayError::Descriptor(_))
        ));
        let (_, _, _, _, wrong_inputs) = affine_family(2);
        assert!(matches!(
            program.run(invocation([("rows", 3)], wrong_inputs)),
            Err(ReplayError::Descriptor(_))
        ));
        assert_eq!(program.compile_count(), 0);

        let mut body_identity = None;
        let mut cache_keys = None;
        for (position, rows) in [0usize, 1, 3, 8].into_iter().enumerate() {
            let (oracle_graph, _, oracle_contiguous, oracle_reduced, inputs) = affine_family(rows);
            let oracle_inputs = inputs.clone().into_iter().collect::<HashMap<_, _>>();
            let oracle_contiguous = CpuBackend
                .execute(&oracle_graph, oracle_contiguous, &oracle_inputs)
                .unwrap();
            let oracle_reduced = CpuBackend
                .execute(&oracle_graph, oracle_reduced, &oracle_inputs)
                .unwrap();
            let result = program
                .run(invocation([("rows", rows as i64)], inputs))
                .unwrap();
            assert_eq!(result.outputs().len(), 3);
            assert_eq!(result.outputs()[0].storage(), oracle_contiguous.storage());
            assert_eq!(result.outputs()[1].storage(), oracle_reduced.storage());
            assert_eq!(result.outputs()[2].storage(), oracle_reduced.storage());
            assert_eq!(result.outputs()[1].shape(), &Shape::from([rows]));
            assert_eq!(result.trace().compiled_now(), position == 0);
            assert_eq!(result.trace().bindings().len(), 1);
            assert_eq!(result.trace().bindings()[0].1, rows as i64);
            if let Some(identity) = body_identity {
                assert_eq!(result.trace().body_identity(), identity);
                assert_eq!(
                    result.trace().native_cache_keys(),
                    cache_keys.as_ref().unwrap()
                );
            } else {
                body_identity = Some(result.trace().body_identity());
                cache_keys = Some(result.trace().native_cache_keys().to_vec());
            }
        }
        assert_eq!(program.compile_count(), 1);
    }

    #[test]
    fn fused_reduction_epilogue_view_uses_output_domain_and_broadcast() {
        let rows = SymbolicExpr::variable("rows", 1, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3, 2, 4], DType::F32);
        let scale = graph.input_dtype("scale", [1, 3], DType::F32);
        let viewed_scale = graph.permute(scale, [1, 0]).unwrap();
        let reduced = graph
            .reduce(input, crate::ReduceKind::Sum, Some(vec![2]), false)
            .unwrap();
        let output = graph.mul(reduced, viewed_scale).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([
                (
                    input,
                    SymbolicShape::new(vec![rows.clone().into(), 2usize.into(), 4usize.into()]),
                ),
                (scale, SymbolicShape::new(vec![1usize.into(), rows.into()])),
            ])),
            &BTreeMap::from([("rows".into(), 3)]),
        )
        .unwrap();
        let schema = capture.symbolic.as_ref().unwrap();
        let reduction_item = capture
            .items
            .iter()
            .find(|item| {
                matches!(
                    schema.item_domains.get(&item.id),
                    Some(crate::engine::symbolic::SymbolicItemDomain::Reduction { .. })
                )
            })
            .expect("schedule retains the reduction recurrence");
        let view_item_id = schema
            .views
            .keys()
            .find_map(|(item, buffer)| (*buffer == scale.index() as u64).then_some(*item))
            .expect("scale permutation has an authenticated symbolic view");
        let view_item = capture
            .items
            .iter()
            .find(|item| item.id == view_item_id)
            .unwrap();
        assert_ne!(view_item.id, reduction_item.id);
        assert!(view_item.dependencies.contains(&reduction_item.id));

        let program = CpuSymbolicProgram::new(capture).unwrap();
        for rows in [1usize, 4] {
            let input = TensorData::new([rows, 2, 4], vec![1.0; rows * 8]).unwrap();
            let scale_values = (1..=rows).map(|value| value as f32).collect::<Vec<_>>();
            let scale = TensorData::new([1, rows], scale_values.clone()).unwrap();
            let result = program
                .run(invocation(
                    [("rows", rows as i64)],
                    BTreeMap::from([("input".into(), input), ("scale".into(), scale)]),
                ))
                .unwrap();
            let expected = TensorData::new(
                [rows, 2],
                scale_values
                    .iter()
                    .flat_map(|value| [value * 4.0; 2])
                    .collect(),
            )
            .unwrap();
            assert_eq!(result.outputs()[0].storage(), expected.storage());
        }
    }

    fn matmul_family(
        rows: usize,
        inner: usize,
        dtype: DType,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [rows, inner], dtype);
        let rhs = graph.input_dtype("rhs", [inner, 2], dtype);
        let output = graph.matmul(lhs, rhs).unwrap();
        let lhs_values = TensorData::from_scalars(
            [rows, inner],
            dtype,
            (0..rows * inner).map(|index| Scalar::F(index as f64 * 0.125 - 0.5)),
        )
        .unwrap();
        let rhs_values = TensorData::from_scalars(
            [inner, 2],
            dtype,
            (0..inner * 2).map(|index| Scalar::F(0.75 - index as f64 * 0.0625)),
        )
        .unwrap();
        (
            graph,
            lhs,
            rhs,
            output,
            BTreeMap::from([("lhs".into(), lhs_values), ("rhs".into(), rhs_values)]),
        )
    }

    fn assert_symbolic_matmul(dtype: DType) {
        let rows = SymbolicExpr::variable("rows", 0, 4).unwrap();
        let inner = SymbolicExpr::variable("inner", 0, 5).unwrap();
        let (template, lhs, rhs, output, _) = matmul_family(2, 3, dtype);
        let schedule = crate::schedule(&template, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([
                (
                    lhs,
                    SymbolicShape::new(vec![rows.clone().into(), inner.clone().into()]),
                ),
                (rhs, SymbolicShape::new(vec![inner.into(), 2usize.into()])),
            ])),
            &BTreeMap::from([("rows".into(), 2), ("inner".into(), 3)]),
        )
        .unwrap();
        let program = CpuSymbolicProgram::new(capture).unwrap();
        let (_, _, _, _, mut mismatched) = matmul_family(2, 3, dtype);
        mismatched.insert(
            "rhs".into(),
            TensorData::from_scalars([2, 2], dtype, (0..4).map(|index| Scalar::F(index as f64)))
                .unwrap(),
        );
        assert!(matches!(
            program.run(invocation([("rows", 2), ("inner", 3)], mismatched)),
            Err(ReplayError::Descriptor(_))
        ));
        assert_eq!(program.compile_count(), 0);
        let mut identity = None;
        for (rows, inner) in [(1usize, 0usize), (0, 3), (2, 3), (4, 5)] {
            let (oracle_graph, _, _, oracle_output, inputs) = matmul_family(rows, inner, dtype);
            let oracle = CpuBackend
                .execute(
                    &oracle_graph,
                    oracle_output,
                    &inputs.clone().into_iter().collect::<HashMap<_, _>>(),
                )
                .unwrap();
            let result = program
                .run(invocation(
                    [("rows", rows as i64), ("inner", inner as i64)],
                    inputs,
                ))
                .unwrap();
            assert_eq!(result.outputs()[0].shape(), &Shape::from([rows, 2]));
            assert_eq!(result.outputs()[0].storage(), oracle.storage());
            if let Some(identity) = identity {
                assert_eq!(result.trace().body_identity(), identity);
            } else {
                identity = Some(result.trace().body_identity());
            }
        }
        assert_eq!(program.compile_count(), 1);
    }

    #[test]
    fn owned_program_executes_dynamic_dense_f32_and_f64_matmul() {
        assert_symbolic_matmul(DType::F32);
        assert_symbolic_matmul(DType::F64);
    }

    #[test]
    fn correlated_symbolic_operands_reject_a_shape_mismatch_before_compilation() {
        let extent = SymbolicExpr::variable("extent", 0, 4).unwrap();
        let shape = SymbolicShape::new(vec![extent.into(), 4usize.into()]);
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 4], DType::F32);
        let rhs = graph.input_dtype("rhs", [2, 4], DType::F32);
        let output = graph.add(lhs, rhs).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(lhs, shape.clone()), (rhs, shape)])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap();
        let program = CpuSymbolicProgram::new(capture).unwrap();
        let lhs_data = TensorData::new([3, 4], vec![1.0; 12]).unwrap();
        let mismatched = BTreeMap::from([
            ("lhs".into(), lhs_data.clone()),
            ("rhs".into(), TensorData::new([2, 4], vec![2.0; 8]).unwrap()),
        ]);
        assert!(matches!(
            program.run(invocation([("extent", 3)], mismatched)),
            Err(ReplayError::Descriptor(_))
        ));
        assert_eq!(program.compile_count(), 0);
        let result = program
            .run(invocation(
                [("extent", 3)],
                BTreeMap::from([
                    ("lhs".into(), lhs_data),
                    (
                        "rhs".into(),
                        TensorData::new([3, 4], vec![2.0; 12]).unwrap(),
                    ),
                ]),
            ))
            .unwrap();
        assert_eq!(result.outputs()[0].values(), &[3.0; 12]);
    }

    #[test]
    fn symbolic_matmul_deduplicates_an_aliased_operand_abi() {
        let extent = SymbolicExpr::variable("extent", 0, 3).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let output = graph.matmul(input, input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.clone().into(), extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap();
        let program = CpuSymbolicProgram::new(capture).unwrap();
        let data =
            TensorData::new([3, 3], vec![1.0, 2.0, 3.0, 0.0, -1.0, 2.0, 4.0, 0.5, 1.0]).unwrap();
        let mut oracle_graph = Graph::new();
        let oracle_input = oracle_graph.input_dtype("input", [3, 3], DType::F32);
        let oracle_output = oracle_graph.matmul(oracle_input, oracle_input).unwrap();
        let expected = CpuBackend
            .execute(
                &oracle_graph,
                oracle_output,
                &HashMap::from([("input".into(), data.clone())]),
            )
            .unwrap();
        let result = program
            .run(invocation(
                [("extent", 3)],
                BTreeMap::from([("input".into(), data)]),
            ))
            .unwrap();
        assert_eq!(result.outputs()[0].shape(), &Shape::from([3, 3]));
        assert_eq!(result.outputs()[0].storage(), expected.storage());
    }

    #[test]
    fn owned_program_preserves_scalar_ieee_bits_alongside_zero_geometry() {
        let extent = SymbolicExpr::variable("extent", 0, 2).unwrap();
        let mut graph = Graph::new();
        let vector = graph.input_dtype("vector", [1], DType::F32);
        let scalar = graph.input_dtype("scalar", [], DType::F32);
        let empty = graph.neg(vector).unwrap();
        let scalar_output = graph.neg(scalar).unwrap();
        let schedule = crate::schedule_many(&graph, &[empty, scalar_output]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[empty, scalar_output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                vector,
                SymbolicShape::new(vec![extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 1)]),
        )
        .unwrap();
        let program = CpuSymbolicProgram::new(capture).unwrap();
        let scalar_bits = 0x8000_0000u32;
        let result = program
            .run(invocation(
                [("extent", 0)],
                BTreeMap::from([
                    (
                        "vector".into(),
                        TensorData::from_storage([0], crate::Storage::F32(Vec::new())).unwrap(),
                    ),
                    (
                        "scalar".into(),
                        TensorData::from_storage(
                            [],
                            crate::Storage::F32(vec![f32::from_bits(scalar_bits)]),
                        )
                        .unwrap(),
                    ),
                ]),
            ))
            .unwrap();
        assert_eq!(result.outputs()[0].shape(), &Shape::from([0]));
        let crate::Storage::F32(values) = result.outputs()[1].storage() else {
            unreachable!()
        };
        assert_eq!(values[0].to_bits(), scalar_bits ^ 0x8000_0000);
    }

    #[test]
    fn symbolic_float8_values_use_the_shared_raw_storage_boundary() {
        let extent = SymbolicExpr::variable("extent", 0, 4).unwrap();
        let dtype = DType::F8E4M3;
        let format = crate::Float8Format::E4M3;
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [4], dtype);
        let rhs = graph.input_dtype("rhs", [4], dtype);
        let condition = graph.input_dtype("condition", [4], DType::Bool);
        let sum = graph.binary(crate::BinaryOp::Add, lhs, rhs).unwrap();
        let same_format = graph.cast(lhs, dtype).unwrap();
        let selected = graph.select(condition, lhs, rhs).unwrap();
        let schedule = crate::schedule_many(&graph, &[sum, same_format, selected]).unwrap();
        let symbolic_shape = SymbolicShape::new(vec![extent.into()]);
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[sum, same_format, selected],
            &SymbolicCaptureSpec::new(BTreeMap::from([
                (lhs, symbolic_shape.clone()),
                (rhs, symbolic_shape.clone()),
                (condition, symbolic_shape),
            ])),
            &BTreeMap::from([("extent".into(), 4)]),
        )
        .unwrap();
        let lhs_data = TensorData::from_storage(
            [4],
            crate::Storage::Float8(crate::Float8Storage::from_raw(
                format,
                vec![0xff, 0x80, format.encode(0.5), format.encode(-2.0)],
            )),
        )
        .unwrap();
        let rhs_data = TensorData::from_storage(
            [4],
            crate::Storage::Float8(crate::Float8Storage::from_raw(
                format,
                vec![
                    format.encode(1.0),
                    0x7f,
                    format.encode(0.5),
                    format.encode(2.0),
                ],
            )),
        )
        .unwrap();
        let condition_data =
            TensorData::from_storage([4], crate::Storage::Bool(vec![true, false, true, false]))
                .unwrap();
        let oracle_bindings = HashMap::from([
            ("lhs".into(), lhs_data.clone()),
            ("rhs".into(), rhs_data.clone()),
            ("condition".into(), condition_data.clone()),
        ]);
        let expected = [sum, same_format, selected].map(|output| {
            CpuBackend
                .execute(&graph, output, &oracle_bindings)
                .unwrap()
        });
        let result = CpuSymbolicProgram::new(capture)
            .unwrap()
            .run(invocation(
                [("extent", 4)],
                BTreeMap::from([
                    ("lhs".into(), lhs_data),
                    ("rhs".into(), rhs_data),
                    ("condition".into(), condition_data),
                ]),
            ))
            .unwrap();
        for (actual, expected) in result.outputs().iter().zip(expected) {
            assert_eq!(
                actual.to_le_bytes().unwrap(),
                expected.to_le_bytes().unwrap()
            );
        }
        assert_eq!(
            result.outputs()[1].to_le_bytes().unwrap(),
            oracle_bindings["lhs"].to_le_bytes().unwrap()
        );
    }

    #[test]
    fn symbolic_singleton_reduction_preserves_narrow_raw_storage() {
        let extent = SymbolicExpr::variable("extent", 1, 2).unwrap();
        let cases = [
            (
                DType::F16,
                TensorData::from_storage([1], crate::Storage::F16(vec![0x7e01])).unwrap(),
            ),
            (
                DType::BF16,
                TensorData::from_storage([1], crate::Storage::BF16(vec![0x7fc1])).unwrap(),
            ),
            (
                DType::F8E4M3,
                TensorData::from_storage(
                    [1],
                    crate::Storage::Float8(crate::Float8Storage::from_raw(
                        crate::Float8Format::E4M3,
                        vec![0xff],
                    )),
                )
                .unwrap(),
            ),
        ];
        for (dtype, source) in cases {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2], dtype);
            let output = if dtype.is_float8() {
                // Public Float8 Sum keeps its F32 recurrence but commits back
                // to Float8 storage.
                graph.reduce(input, crate::ReduceKind::Sum, Some(vec![0]), false)
            } else {
                // Exercise the raw same-storage narrow reduction contract.
                graph.reduce_with_output_dtype(
                    input,
                    crate::ReduceKind::Sum,
                    Some(vec![0]),
                    false,
                    dtype,
                )
            }
            .unwrap();
            let schedule = crate::schedule(&graph, output).unwrap();
            let capture = CapturedSchedule::capture_symbolic(
                &graph,
                &schedule,
                &[output],
                &SymbolicCaptureSpec::new(BTreeMap::from([(
                    input,
                    SymbolicShape::new(vec![extent.clone().into()]),
                )])),
                &BTreeMap::from([("extent".into(), 2)]),
            )
            .unwrap();
            let result = CpuSymbolicProgram::new(capture)
                .unwrap()
                .run(invocation(
                    [("extent", 1)],
                    BTreeMap::from([("input".into(), source.clone())]),
                ))
                .unwrap();
            assert_eq!(result.outputs()[0].shape(), &Shape::new([]));
            assert_eq!(
                result.outputs()[0].to_le_bytes().unwrap(),
                source.to_le_bytes().unwrap(),
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn symbolic_argmax_specializes_one_shape_iota_across_axis_extents() {
        let tokens = SymbolicExpr::variable("tokens", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let output = graph.argmax_with_axis(input, Some(1), false).unwrap();
        let iota = (0..graph.node_count())
            .map(crate::NodeId::from_index)
            .find(|node| matches!(graph.op(*node), Ok(crate::Op::ShapeIota { .. })))
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![2usize.into(), tokens.clone().into()]),
            )])),
            &BTreeMap::from([("tokens".into(), 3)]),
        )
        .unwrap();
        assert_eq!(
            capture.symbolic.as_ref().unwrap().buffer_shapes[&(iota.index() as u64)],
            SymbolicShape::new(vec![tokens.into()])
        );
        let encoded = crate::schedule::artifact::encode(&capture).unwrap();
        let decoded = crate::schedule::artifact::decode(&encoded).unwrap();
        let mut tampered = decoded.clone();
        tampered
            .symbolic
            .as_mut()
            .unwrap()
            .buffer_shapes
            .insert(iota.index() as u64, SymbolicShape::new(vec![1usize.into()]));
        tampered.identity = 0;
        tampered.identity = crate::schedule::artifact::identity(&tampered).unwrap();
        assert!(CpuSymbolicProgram::new(tampered).is_err());
        let program = CpuSymbolicProgram::new(decoded).unwrap();

        for (extent, values, expected) in [
            (0usize, vec![], vec![i32::MIN, i32::MIN]),
            (1usize, vec![5.0, -2.0], vec![0, 0]),
            (3, vec![1.0, 5.0, 5.0, 9.0, 2.0, 9.0], vec![1, 0]),
            (4, vec![1.0, 5.0, 5.0, 0.0, 9.0, 2.0, 9.0, 10.0], vec![1, 3]),
        ] {
            let input = TensorData::new([2, extent], values).unwrap();
            let result = program
                .run(invocation(
                    [("tokens", extent as i64)],
                    BTreeMap::from([("input".into(), input)]),
                ))
                .unwrap();
            assert_eq!(
                (0..2)
                    .map(|index| result.outputs()[0].scalar_at(index))
                    .collect::<Vec<_>>(),
                expected
                    .into_iter()
                    .map(|value| Scalar::I(i64::from(value)))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(program.compile_count(), 1);

        let extent = SymbolicExpr::variable("extent", 0, 4).unwrap();
        let mut iota_graph = Graph::new();
        let source = iota_graph.input("source", [2, 1]);
        let iota = iota_graph.shape_iota(source, 1).unwrap();
        let schedule = crate::schedule(&iota_graph, iota).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &iota_graph,
            &schedule,
            &[iota],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                source,
                SymbolicShape::new(vec![2usize.into(), extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 1)]),
        )
        .unwrap();
        let program = CpuSymbolicProgram::new(capture).unwrap();
        for extent in [0usize, 1, 4] {
            let result = program
                .run(invocation([("extent", extent as i64)], BTreeMap::new()))
                .unwrap();
            assert_eq!(
                result.outputs()[0],
                TensorData::from_scalars(
                    [extent],
                    DType::I32,
                    (0..extent).map(|value| Scalar::I(value as i64)),
                )
                .unwrap()
            );
        }

        let oversized = SymbolicExpr::variable("oversized", 1, i64::from(i32::MAX) + 1).unwrap();
        let mut oversized_graph = Graph::new();
        let source = oversized_graph.input("source", [1]);
        let iota = oversized_graph.shape_iota(source, 0).unwrap();
        let schedule = crate::schedule(&oversized_graph, iota).unwrap();
        assert!(matches!(
            CapturedSchedule::capture_symbolic(
                &oversized_graph,
                &schedule,
                &[iota],
                &SymbolicCaptureSpec::new(BTreeMap::from([(
                    source,
                    SymbolicShape::new(vec![oversized.into()]),
                )])),
                &BTreeMap::from([("oversized".into(), 1)]),
            ),
            Err(crate::ReplayError::Unsupported(message))
                if message == "symbolic shape iota exceeds its fixed integer storage"
        ));
    }

    #[test]
    fn symbolic_mean_commits_runtime_cardinality_at_the_work_dtype() {
        let cases = [
            (DType::F16, 2_049usize),
            (DType::BF16, 257usize),
            (DType::F8E5M2, 9usize),
        ];
        for (dtype, count) in cases {
            let extent = SymbolicExpr::variable("extent", 1, count as i64).unwrap();
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [count], dtype);
            let output = graph
                .reduce_with_output_dtype(
                    input,
                    crate::ReduceKind::Mean,
                    Some(vec![0]),
                    false,
                    dtype,
                )
                .unwrap();
            let schedule = crate::schedule(&graph, output).unwrap();
            let capture = CapturedSchedule::capture_symbolic(
                &graph,
                &schedule,
                &[output],
                &SymbolicCaptureSpec::new(BTreeMap::from([(
                    input,
                    SymbolicShape::new(vec![extent.into()]),
                )])),
                &BTreeMap::from([("extent".into(), count as i64)]),
            )
            .unwrap();
            let source = TensorData::from_scalars(
                [count],
                dtype,
                std::iter::repeat_n(Scalar::F(1.0), count),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), source.clone())]),
                )
                .unwrap();
            let actual = CpuSymbolicProgram::new(capture)
                .unwrap()
                .run(invocation(
                    [("extent", count as i64)],
                    BTreeMap::from([("input".into(), source)]),
                ))
                .unwrap();
            assert_eq!(actual.outputs()[0].shape(), &Shape::new([]));
            assert_eq!(
                actual.outputs()[0].to_le_bytes().unwrap(),
                expected.to_le_bytes().unwrap(),
                "{dtype:?} cardinality {count}"
            );
        }
    }

    #[test]
    fn symbolic_execution_realizes_exact_temporary_slot_reuse() {
        let extent = SymbolicExpr::variable("extent", 0, 8).unwrap();
        let mut chain = Graph::new();
        let input = chain.input_dtype("input", [4], DType::F32);
        let mut output = input;
        for _ in 0..6 {
            output = chain.square(output).unwrap();
            output = chain.contiguous(output).unwrap();
        }
        let schedule = crate::schedule(&chain, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &chain,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.clone().into()]),
            )])),
            &BTreeMap::from([("extent".into(), 4)]),
        )
        .unwrap();
        let expected_memory = preflight_memory(&capture).unwrap();
        assert!(expected_memory.peak_allocations < expected_memory.requests.len());
        let chain_result = CpuSymbolicProgram::new(capture)
            .unwrap()
            .run(invocation(
                [("extent", 4)],
                BTreeMap::from([(
                    "input".into(),
                    TensorData::new([4], vec![1.0, 0.5, -1.0, 2.0]).unwrap(),
                )]),
            ))
            .unwrap();
        assert_eq!(
            chain_result.trace().peak_temporary_allocations(),
            expected_memory.peak_allocations
        );
        assert_eq!(
            chain_result.trace().peak_temporary_bytes(),
            expected_memory.peak_bytes
        );

        let mut diamond = Graph::new();
        let input = diamond.input_dtype("input", [4], DType::F32);
        let left_value = diamond.square(input).unwrap();
        let left = diamond.contiguous(left_value).unwrap();
        let right_value = diamond.neg(input).unwrap();
        let right = diamond.contiguous(right_value).unwrap();
        let joined = diamond.add(left, right).unwrap();
        let output = diamond.contiguous(joined).unwrap();
        let schedule = crate::schedule(&diamond, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &diamond,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 4)]),
        )
        .unwrap();
        let expected_memory = preflight_memory(&capture).unwrap();
        assert!(expected_memory.peak_allocations >= 2);
        let diamond_result = CpuSymbolicProgram::new(capture)
            .unwrap()
            .run(invocation(
                [("extent", 4)],
                BTreeMap::from([(
                    "input".into(),
                    TensorData::new([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                )]),
            ))
            .unwrap();
        assert_eq!(
            diamond_result.trace().peak_temporary_allocations(),
            expected_memory.peak_allocations
        );
        assert_eq!(
            diamond_result.trace().peak_temporary_bytes(),
            expected_memory.peak_bytes
        );
    }

    #[test]
    fn symbolic_affine_copy_accepts_an_explicit_materialized_view_source() {
        let rows = SymbolicExpr::variable("rows", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("unused_source", [2, 3], DType::F32);
        let producer = graph.square(input).unwrap();
        let view = graph.permute(producer, [1, 0]).unwrap();
        let output = graph.contiguous(view).unwrap();
        let schedule =
            crate::schedule_with_external_materializations(&graph, &[output], &[producer]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![rows.into(), 3usize.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        let external_name = format!("@materialized/{}", producer.index());
        assert_eq!(
            capture
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec![external_name.as_str()]
        );
        let result = CpuSymbolicProgram::new(capture)
            .unwrap()
            .run(invocation(
                [("rows", 1)],
                BTreeMap::from([(
                    external_name,
                    TensorData::new([1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
                )]),
            ))
            .unwrap();
        assert_eq!(result.outputs()[0].shape(), &Shape::from([3, 1]));
        assert_eq!(result.outputs()[0].values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn invocation_outputs_are_detached_owned_values() {
        let rows = SymbolicExpr::variable("rows", 0, 2).unwrap();
        let mut template = Graph::new();
        let input = template.input_dtype("input", [2], DType::F32);
        let output = template.square(input).unwrap();
        let schedule = crate::schedule(&template, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![rows.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        let program = CpuSymbolicProgram::with_output_order(capture, vec![0, 0]).unwrap();
        let mut result = program
            .run(invocation(
                [("rows", 1)],
                BTreeMap::from([("input".into(), TensorData::new([1], vec![2.0]).unwrap())]),
            ))
            .unwrap()
            .into_outputs();
        assert_eq!(result.len(), 2);
        let before = result[1].clone();
        result[0]
            .replace(&TensorData::new([1], vec![123.0]).unwrap())
            .unwrap();
        assert_eq!(result[1].storage(), before.storage());
    }

    #[test]
    fn symbolic_requested_affine_view_preserves_alias_geometry_and_raw_storage() {
        let rows = SymbolicExpr::variable("rows", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 4], DType::F32);
        let reshaped = graph.reshape(input, [2, 1, 4]).unwrap();
        // Keep the fixed expansion distinct from the rows=2 template binding.
        let expanded = graph.expand(reshaped, [2, 3, 4]).unwrap();
        let permuted = graph.permute(expanded, [1, 0, 2]).unwrap();
        let shrunk = graph.shrink(permuted, [(0, 1), (0, 2), (0, 4)]).unwrap();
        let reversed = graph
            .stride(
                shrunk,
                [
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                ],
            )
            .unwrap();
        let schedule = crate::schedule_many(&graph, &[reversed, reversed]).unwrap();
        assert!(schedule.items.is_empty());
        assert_eq!(schedule.requested_passthroughs.len(), 1);
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[reversed, reversed],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![rows.into(), 4usize.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        assert!(capture.items.is_empty());
        assert_eq!(capture.symbolic.as_ref().unwrap().requested_views.len(), 1);
        let bytes = capture.to_bytes().unwrap();
        let capture = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(capture.to_bytes().unwrap(), bytes);

        let variable = capture.symbolic_parameters()[0].variable().id();
        let empty =
            crate::engine::symbolic::specialize_capture(&capture, &[(variable, 0)]).unwrap();
        let populated =
            crate::engine::symbolic::specialize_capture(&capture, &[(variable, 1)]).unwrap();
        assert_ne!(empty.identity, populated.identity);
        assert_eq!(
            empty.requested_passthroughs[0].desc.shape,
            Shape::from([0, 4])
        );
        assert_eq!(
            empty.requested_passthroughs[0]
                .desc
                .view
                .as_ref()
                .unwrap()
                .logical_shape,
            Shape::from([1, 0, 4])
        );

        let lanes = vec![
            f32::from_bits(0x7fc0_1234),
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let result = CpuSymbolicProgram::new(capture.clone())
            .unwrap()
            .run(invocation(
                [("rows", 1)],
                BTreeMap::from([(
                    "input".into(),
                    TensorData::from_storage([1, 4], crate::Storage::F32(lanes)).unwrap(),
                )]),
            ))
            .unwrap();
        assert_eq!(result.outputs().len(), 2);
        let expected = [
            f32::NEG_INFINITY.to_bits(),
            f32::INFINITY.to_bits(),
            (-0.0f32).to_bits(),
            0x7fc0_1234,
        ];
        for output in result.outputs() {
            let crate::Storage::F32(actual) = output.storage() else {
                panic!("requested view must retain F32 storage")
            };
            assert_eq!(
                actual.iter().map(|lane| lane.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }

        let empty_result = CpuSymbolicProgram::new(capture.clone())
            .unwrap()
            .run(invocation(
                [("rows", 0)],
                BTreeMap::from([(
                    "input".into(),
                    TensorData::from_storage([0, 4], crate::Storage::F32(vec![])).unwrap(),
                )]),
            ))
            .unwrap();
        assert_eq!(empty_result.outputs()[0].shape(), &Shape::from([1, 0, 4]));
        assert!(empty_result.outputs()[0].to_le_bytes().unwrap().is_empty());

        let mut missing = capture.clone();
        missing.symbolic.as_mut().unwrap().requested_views.clear();
        missing.identity = 0;
        missing.identity = crate::schedule::artifact::identity(&missing).unwrap();
        assert!(missing.to_bytes().is_err());

        let mut tampered = capture;
        tampered
            .symbolic
            .as_mut()
            .unwrap()
            .requested_views
            .values_mut()
            .next()
            .unwrap()
            .offset = SymbolicExpr::constant(1);
        tampered.identity = 0;
        tampered.identity = crate::schedule::artifact::identity(&tampered).unwrap();
        assert!(tampered.to_bytes().is_err());
        assert!(CpuSymbolicProgram::new(tampered).is_err());
    }

    #[test]
    fn symbolic_requested_constant_view_resizes_only_exact_splat_storage() {
        let rows = SymbolicExpr::variable("rows", 0, 3).unwrap();
        let mut graph = Graph::new();
        let source = graph.constant(
            TensorData::from_storage(
                [2, 1],
                crate::Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E4M3,
                    vec![0x80; 2],
                )),
            )
            .unwrap(),
        );
        // Keep the fixed expansion distinct from the rows=2 template binding.
        let expanded = graph.expand(source, [2, 3]).unwrap();
        let output = graph.permute(expanded, [1, 0]).unwrap();
        let scalar_source = graph
            .constant(TensorData::from_storage([1], crate::Storage::U16(vec![0x7e01])).unwrap());
        let scalar = graph.reshape(scalar_source, Shape::new([])).unwrap();
        let schedule = crate::schedule_many(&graph, &[output, scalar]).unwrap();
        assert!(schedule.items.is_empty());
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output, scalar],
            &SymbolicCaptureSpec::new(BTreeMap::new())
                .with_constant_shape(source, SymbolicShape::new(vec![rows.into(), 1usize.into()])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        let outputs = CpuSymbolicProgram::new(capture)
            .unwrap()
            .run(invocation([("rows", 1)], BTreeMap::new()))
            .unwrap()
            .into_outputs();
        let output = &outputs[0];
        assert_eq!(output.shape(), &Shape::from([3, 1]));
        assert_eq!(
            output.storage(),
            &crate::Storage::Float8(crate::Float8Storage::from_raw(
                crate::Float8Format::E4M3,
                vec![0x80; 3],
            ))
        );
        assert_eq!(outputs[1].shape(), &Shape::new([]));
        assert_eq!(outputs[1].storage(), &crate::Storage::U16(vec![0x7e01]));
    }

    #[test]
    fn computed_affine_view_keeps_its_logical_shape_before_consumer_broadcast() {
        let columns = SymbolicExpr::variable("columns", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2], DType::F32);
        let rhs = graph.input_dtype("rhs", [2, 3], DType::F32);
        let producer = graph.square(input).unwrap();
        let transposed = graph.permute(producer, [1, 0]).unwrap();
        let output = graph.add(transposed, rhs).unwrap();
        let schedule = crate::schedule_many(&graph, &[producer, output]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[producer, output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                rhs,
                SymbolicShape::new(vec![2usize.into(), columns.into()]),
            )])),
            &BTreeMap::from([("columns".into(), 3)]),
        )
        .unwrap();
        assert!(
            capture
                .symbolic
                .as_ref()
                .unwrap()
                .views
                .values()
                .any(|view| {
                    view.logical_shape == SymbolicShape::new(vec![2usize.into(), 1usize.into()])
                })
        );
        let mut tampered = capture.clone();
        let view = tampered
            .symbolic
            .as_mut()
            .unwrap()
            .views
            .values_mut()
            .next()
            .unwrap();
        view.offset = view.offset.clone() + SymbolicExpr::constant(1);
        tampered.identity = 0;
        tampered.identity = crate::schedule::artifact::identity(&tampered).unwrap();
        assert!(matches!(
            CpuSymbolicProgram::new(tampered),
            Err(ReplayError::Corrupt(_))
        ));

        let input_data = TensorData::new([1, 2], vec![2.0, 3.0]).unwrap();
        let rhs_data = TensorData::new([2, 4], (0..8).map(|value| value as f32).collect()).unwrap();
        let expected =
            TensorData::new([2, 4], vec![4.0, 5.0, 6.0, 7.0, 13.0, 14.0, 15.0, 16.0]).unwrap();
        let result = CpuSymbolicProgram::new(capture)
            .unwrap()
            .run(invocation(
                [("columns", 4)],
                BTreeMap::from([("input".into(), input_data), ("rhs".into(), rhs_data)]),
            ))
            .unwrap();
        assert_eq!(result.outputs()[1].storage(), expected.storage());
    }

    #[test]
    fn execution_failure_publishes_no_result_and_the_program_retries() {
        let rows = SymbolicExpr::variable("rows", 0, 4).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::I32);
        let divisor = graph.input_dtype("divisor", [2], DType::I32);
        let squared = graph.square(input).unwrap();
        let materialized = graph.contiguous(squared).unwrap();
        let output = graph
            .binary(crate::BinaryOp::Div, materialized, divisor)
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let shape = SymbolicShape::new(vec![rows.into()]);
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(input, shape.clone()), (divisor, shape)])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        let program = CpuSymbolicProgram::new(capture).unwrap();
        let input =
            TensorData::from_scalars([2], DType::I32, [Scalar::I(6), Scalar::I(8)]).unwrap();
        let bad = BTreeMap::from([
            ("input".into(), input.clone()),
            (
                "divisor".into(),
                TensorData::from_scalars([2], DType::I32, [Scalar::I(1), Scalar::I(0)]).unwrap(),
            ),
        ]);
        assert!(matches!(
            program.run(invocation([("rows", 2)], bad)),
            Err(ReplayError::Execute(_))
        ));
        assert_eq!(program.compile_count(), 1);
        let good = BTreeMap::from([
            ("input".into(), input),
            (
                "divisor".into(),
                TensorData::from_scalars([2], DType::I32, [Scalar::I(2), Scalar::I(4)]).unwrap(),
            ),
        ]);
        let retried = program.run(invocation([("rows", 2)], good)).unwrap();
        assert_eq!(retried.outputs()[0].to_vec_f64(), vec![18.0, 16.0]);
        assert!(!retried.trace().compiled_now());
        assert_eq!(program.compile_count(), 1);
    }

    fn transformer_family(
        block: &TransformerBlock,
        norm: &RMSNorm,
        time: usize,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("tokens", [1, time, 4], DType::F32);
        let normalized = norm.forward(&mut graph, input).unwrap();
        let output = block
            .forward_mode(&mut graph, normalized, Mode::Eval)
            .unwrap()
            .output;
        let mut bindings = block
            .input_bindings(&graph)
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        bindings.extend(norm.input_bindings(&graph).unwrap());
        bindings.insert(
            "tokens".into(),
            TensorData::from_scalars(
                [1, time, 4],
                DType::F32,
                (0..time * 4).map(|index| Scalar::F(index as f64 * 0.03125 - 0.25)),
            )
            .unwrap(),
        );
        (graph, input, output, bindings)
    }

    fn assert_f32_close(actual: &TensorData, expected: &TensorData, tolerance: f64) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), DType::F32);
        assert_eq!(expected.dtype(), DType::F32);
        for (index, (actual, expected)) in actual
            .to_vec_f64()
            .into_iter()
            .zip(expected.to_vec_f64())
            .enumerate()
        {
            if expected.is_nan() {
                assert!(actual.is_nan(), "lane {index}: expected NaN, got {actual}");
            } else if expected.is_infinite() {
                assert_eq!(actual, expected, "lane {index}: infinity differs");
            } else {
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= tolerance,
                    "lane {index}: expected {expected}, got {actual} (tolerance {tolerance})"
                );
            }
        }
    }

    #[test]
    fn owned_program_runs_one_bounded_transformer_body_across_sequence_lengths() {
        let time = SymbolicExpr::variable("time", 1, 4).unwrap();
        let block = TransformerBlock::new_static(4, 2, 8, true, 0.0, 0x51d).unwrap();
        let norm = RMSNorm::new_static(4, 1e-5, true).unwrap();
        let (template, input, output, _) = transformer_family(&block, &norm, 3);
        let schedule = crate::schedule(&template, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[output],
            &SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![1usize.into(), time.into(), 4usize.into()]),
            )])),
            &BTreeMap::from([("time".into(), 3)]),
        )
        .unwrap();
        let schema = capture.symbolic.as_ref().unwrap();
        assert!(capture.items.iter().any(|item| {
            matches!(
                schema.item_domains.get(&item.id),
                Some(crate::engine::symbolic::SymbolicItemDomain::Reduction { .. })
            ) && !matches!(template.op(item.node), Ok(crate::Op::Reduce { .. }))
        }));
        let program = CpuSymbolicProgram::new(capture).unwrap();
        let cache_keys = program.rendered.len();
        assert!(cache_keys > 1);
        let admitted_inputs = program
            .inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        for (position, time) in [1usize, 3, 4].into_iter().enumerate() {
            let (oracle_graph, _, oracle_output, mut inputs) =
                transformer_family(&block, &norm, time);
            let oracle = CpuBackend
                .execute(
                    &oracle_graph,
                    oracle_output,
                    &inputs.clone().into_iter().collect::<HashMap<_, _>>(),
                )
                .unwrap();
            inputs.retain(|name, _| admitted_inputs.contains(name.as_str()));
            let result = program
                .run(invocation([("time", time as i64)], inputs))
                .unwrap();
            assert_eq!(result.outputs()[0].shape(), &Shape::from([1, time, 4]));
            assert_f32_close(&result.outputs()[0], &oracle, 1e-5);
            assert_eq!(result.trace().native_cache_keys().len(), cache_keys);
            assert_eq!(result.trace().compiled_now(), position == 0);
        }
        assert_eq!(program.compile_count(), 1);
    }
}
