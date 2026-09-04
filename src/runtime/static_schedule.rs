//! Shared planning and execution for static single-device schedule prefixes.
//!
//! Renderers remain the owners of operation support. This module owns only the
//! buffer residency and side-effect boundary common to the prepared OpenCL,
//! Metal, WebGPU, and fixed-schema CUDA graph paths.

use crate::{
    BufferDesc, CapturedSchedule, DType, Operation, QuantizedBufferDesc, QuantizedTensorData,
    ReplayError, ReplayInput, RequestedPassthrough, Scalar, ScheduleInputBinding, ScheduleItem,
    Shape, SymbolicInvocation, TensorData,
    engine::{AuthenticatedSymbolicBody, AuthenticatedSymbolicInvocation},
    memory_plan::{ExactSlotPolicy, ExactSlotRequest, assign_exact_slots},
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

mod sealed {
    pub trait Sealed {}
}

/// Authenticated host ownership around one concrete captured static prefix.
///
/// Rendering, allocation, and execution remain in [`StaticSchedulePlan`] and
/// [`PreparedStaticSchedule`]. This projection only translates the capture's
/// named inputs and owned constants into their exact logical buffer IDs, then
/// restores requested values in capture order after a successful transaction.
#[derive(Clone)]
pub(crate) struct CapturedStaticExecution {
    backing: CapturedStaticBacking,
    passthroughs: BTreeMap<u64, RequestedPassthrough>,
    retained: Vec<u64>,
}

#[derive(Clone)]
enum CapturedStaticBacking {
    Projected {
        inputs: Vec<ReplayInput>,
        constants: BTreeMap<u64, TensorData>,
        quantized_constants: BTreeMap<u64, QuantizedTensorData>,
        requested: Vec<u64>,
    },
    Owned(Arc<CapturedSchedule>),
}

#[derive(Debug)]
pub(crate) enum CapturedStaticAdmissionError {
    Invalid(String),
    Unsupported(String),
}

impl CapturedStaticExecution {
    pub(crate) fn new(capture: &CapturedSchedule) -> Result<Self, String> {
        Self::admit(capture, None).map_err(|error| match error {
            CapturedStaticAdmissionError::Invalid(reason)
            | CapturedStaticAdmissionError::Unsupported(reason) => reason,
        })
    }

    pub(crate) fn from_owned(
        capture: CapturedSchedule,
    ) -> Result<Self, CapturedStaticAdmissionError> {
        let capture = Arc::new(capture);
        Self::admit(&capture, Some(capture.clone()))
    }

    fn admit(
        capture: &CapturedSchedule,
        owned: Option<Arc<CapturedSchedule>>,
    ) -> Result<Self, CapturedStaticAdmissionError> {
        crate::schedule::artifact::validate_capture(capture).map_err(|error| {
            CapturedStaticAdmissionError::Invalid(format!("captured static identity: {error}"))
        })?;
        if capture.is_symbolic() {
            return Err(CapturedStaticAdmissionError::Unsupported(
                "captured static execution requires a concrete artifact".into(),
            ));
        }
        if capture
            .items
            .iter()
            .any(|item| item.is_effect() || item.boundary.is_some())
        {
            return Err(CapturedStaticAdmissionError::Unsupported(
                "captured static execution requires a pure boundary-free prefix".into(),
            ));
        }
        if owned.is_none()
            && (!capture.quantized_constants.is_empty()
                || capture
                    .items
                    .iter()
                    .any(|item| !item.quantized_input_bindings.is_empty()))
        {
            return Err(CapturedStaticAdmissionError::Unsupported(
                "captured static execution does not admit quantized bindings".into(),
            ));
        }
        let produced = capture
            .items
            .iter()
            .flat_map(|item| item.outputs.iter().map(|output| output.id))
            .collect::<BTreeSet<_>>();
        let mut retained_seen = BTreeSet::new();
        let aliases = capture
            .requested_passthroughs
            .iter()
            .map(|alias| (alias.requested.index() as u64, alias.source.index() as u64))
            .collect::<BTreeMap<_, _>>();
        let retained = capture
            .requested
            .iter()
            .filter_map(|id| {
                produced.contains(id).then_some(*id).or_else(|| {
                    aliases
                        .get(id)
                        .copied()
                        .filter(|source| produced.contains(source))
                })
            })
            .filter(|id| retained_seen.insert(*id))
            .collect::<Vec<_>>();
        if !capture.items.is_empty() && retained.is_empty() {
            return Err(CapturedStaticAdmissionError::Invalid(
                "captured static prefix has no produced requested output".into(),
            ));
        }
        Ok(Self {
            passthroughs: capture
                .requested_passthroughs
                .iter()
                .cloned()
                .map(|alias| (alias.requested.index() as u64, alias))
                .collect(),
            backing: match owned {
                Some(capture) => CapturedStaticBacking::Owned(capture),
                None => CapturedStaticBacking::Projected {
                    inputs: capture.inputs.clone(),
                    constants: capture.constants.clone(),
                    quantized_constants: capture.quantized_constants.clone(),
                    requested: capture.requested.clone(),
                },
            },
            retained,
        })
    }

    fn owned_capture(&self) -> Option<&CapturedSchedule> {
        match &self.backing {
            CapturedStaticBacking::Owned(capture) => Some(capture),
            CapturedStaticBacking::Projected { .. } => None,
        }
    }

    pub(crate) fn retained(&self) -> &[u64] {
        &self.retained
    }

    pub(crate) fn retained_for_requested_prefix(&self, count: usize) -> Result<Vec<u64>, String> {
        if count > self.requested().len() {
            return Err("captured static requested prefix is out of range".into());
        }
        let retained = self.retained.iter().copied().collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        Ok(self.requested()[..count]
            .iter()
            .filter_map(|id| {
                retained.contains(id).then_some(*id).or_else(|| {
                    self.passthroughs
                        .get(id)
                        .map(|alias| alias.source.index() as u64)
                        .filter(|source| retained.contains(source))
                })
            })
            .filter(|id| seen.insert(*id))
            .collect())
    }

    pub(crate) fn inputs(&self) -> &[ReplayInput] {
        match &self.backing {
            CapturedStaticBacking::Projected { inputs, .. } => inputs,
            CapturedStaticBacking::Owned(capture) => &capture.inputs,
        }
    }

    fn constants(&self) -> &BTreeMap<u64, TensorData> {
        match &self.backing {
            CapturedStaticBacking::Projected { constants, .. } => constants,
            CapturedStaticBacking::Owned(capture) => &capture.constants,
        }
    }

    fn quantized_constants(&self) -> &BTreeMap<u64, QuantizedTensorData> {
        match &self.backing {
            CapturedStaticBacking::Projected {
                quantized_constants,
                ..
            } => quantized_constants,
            CapturedStaticBacking::Owned(capture) => &capture.quantized_constants,
        }
    }

    fn requested(&self) -> &[u64] {
        match &self.backing {
            CapturedStaticBacking::Projected { requested, .. } => requested,
            CapturedStaticBacking::Owned(capture) => &capture.requested,
        }
    }

    pub(crate) fn constant_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.constants().keys().copied()
    }

    pub(crate) fn quantized_constant_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.quantized_constants().keys().copied()
    }

    fn validate_provided_names(
        inputs: &[ReplayInput],
        provided: &BTreeMap<String, TensorData>,
        context: &str,
    ) -> Result<(), String> {
        let expected = inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(extra) = provided
            .keys()
            .find(|name| !expected.contains(name.as_str()))
        {
            return Err(format!("unexpected {context} input {extra}"));
        }
        if let Some(missing) = inputs
            .iter()
            .find(|input| !provided.contains_key(&input.name))
        {
            return Err(format!("missing {context} input {}", missing.name));
        }
        Ok(())
    }

    fn insert_inputs(
        inputs: &[ReplayInput],
        provided: &BTreeMap<String, TensorData>,
        values: &mut BTreeMap<u64, TensorData>,
        context: &str,
    ) -> Result<(), String> {
        Self::validate_provided_names(inputs, provided, context)?;
        for input in inputs {
            let value = &provided[&input.name];
            if value.shape() != &input.desc.shape || value.dtype() != input.desc.dtype {
                return Err(format!(
                    "{context} input {} descriptor mismatch",
                    input.name
                ));
            }
            let bytes = value
                .to_le_bytes()
                .map_err(|_| format!("{context} input {} bytes", input.name))?;
            if bytes.len() != input.desc.bytes {
                return Err(format!(
                    "{context} input {} byte length mismatch",
                    input.name
                ));
            }
            if values.insert(input.desc.id, value.clone()).is_some() {
                return Err(format!(
                    "{context} input {} aliases owned storage",
                    input.name
                ));
            }
        }
        Ok(())
    }

    fn insert_owned_inputs(
        inputs: &[ReplayInput],
        mut provided: BTreeMap<String, TensorData>,
        values: &mut BTreeMap<u64, TensorData>,
        context: &str,
    ) -> Result<(), String> {
        Self::validate_provided_names(inputs, &provided, context)?;
        for input in inputs {
            let value = provided
                .remove(&input.name)
                .expect("validated resident input name");
            if value.shape() != &input.desc.shape || value.dtype() != input.desc.dtype {
                return Err(format!(
                    "{context} input {} descriptor mismatch",
                    input.name
                ));
            }
            let bytes = value
                .to_le_bytes()
                .map_err(|_| format!("{context} input {} bytes", input.name))?;
            if bytes.len() != input.desc.bytes {
                return Err(format!(
                    "{context} input {} byte length mismatch",
                    input.name
                ));
            }
            if values.insert(input.desc.id, value).is_some() {
                return Err(format!(
                    "{context} input {} aliases owned storage",
                    input.name
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn stage(
        &self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<u64, TensorData>, String> {
        let mut values = self.constants().clone();
        Self::insert_inputs(self.inputs(), provided, &mut values, "captured static")?;
        Ok(values)
    }

    pub(crate) fn project(
        &self,
        values: &BTreeMap<u64, TensorData>,
    ) -> Result<Vec<TensorData>, String> {
        self.project_with_fallback(values, &BTreeMap::new())
    }

    pub(crate) fn project_prefix(
        &self,
        count: usize,
        values: &BTreeMap<u64, TensorData>,
        fallback: &BTreeMap<u64, TensorData>,
    ) -> Result<Vec<TensorData>, String> {
        if count > self.requested().len() {
            return Err("captured static requested prefix is out of range".into());
        }
        self.project_requested(&self.requested()[..count], values, fallback)
    }

    fn project_with_fallback(
        &self,
        values: &BTreeMap<u64, TensorData>,
        fallback: &BTreeMap<u64, TensorData>,
    ) -> Result<Vec<TensorData>, String> {
        self.project_requested(self.requested(), values, fallback)
    }

    fn project_requested(
        &self,
        requested: &[u64],
        values: &BTreeMap<u64, TensorData>,
        fallback: &BTreeMap<u64, TensorData>,
    ) -> Result<Vec<TensorData>, String> {
        let value = |id: u64| {
            values
                .get(&id)
                .or_else(|| fallback.get(&id))
                .or_else(|| self.constants().get(&id))
        };
        requested
            .iter()
            .map(|id| {
                if let Some(alias) = self.passthroughs.get(id) {
                    let source = value(alias.source.index() as u64).ok_or_else(|| {
                        "captured static passthrough source is absent".to_string()
                    })?;
                    return alias
                        .project(source)
                        .map_err(|error| format!("captured static passthrough: {error}"));
                }
                value(*id)
                    .cloned()
                    .ok_or_else(|| format!("captured static requested value {id} is absent"))
            })
            .collect()
    }

    /// Runs one backend transaction around the authenticated host projection.
    /// No requested value is observable unless staging, device execution, and
    /// final ordered projection all succeed.
    pub(crate) fn transact<E>(
        &self,
        provided: &BTreeMap<String, TensorData>,
        invalid_binding: impl Fn(String) -> E,
        execute: impl FnOnce(&mut BTreeMap<u64, TensorData>) -> Result<(), E>,
    ) -> Result<Vec<TensorData>, E> {
        let mut values = self.stage(provided).map_err(&invalid_binding)?;
        execute(&mut values)?;
        self.project(&values).map_err(invalid_binding)
    }
}

/// Runtime-only partition of one authenticated capture's named inputs.
///
/// Constants and explicitly selected inputs are immutable resident values;
/// authenticated runtime controls are session-synthesized, and every remaining
/// named input must be supplied to each invocation. This does not alter captured
/// bytes or schedule identities.
pub(crate) struct StaticLifetimePlan {
    capture: CapturedStaticExecution,
    resident_inputs: Vec<ReplayInput>,
    state_inputs: Vec<ReplayInput>,
    runtime_controls: Vec<ReplayInput>,
    transient_inputs: Vec<ReplayInput>,
    resident_ids: BTreeSet<u64>,
}

#[derive(Debug)]
pub(crate) enum StaticQuantizedGatherError {
    Invalid(String),
    IndexOutOfBounds {
        position: usize,
        value: i32,
        rows: usize,
    },
}

enum CheckedI32IndexError {
    Descriptor,
    IndexOutOfBounds { position: usize, value: i32 },
}

fn validate_i32_index_domain(
    value: &TensorData,
    shape: &Shape,
    extent: usize,
) -> Result<(), CheckedI32IndexError> {
    if value.dtype() != DType::I32 || value.shape() != shape {
        return Err(CheckedI32IndexError::Descriptor);
    }
    for position in 0..value.len() {
        let Scalar::I(raw) = value.scalar_at(position) else {
            return Err(CheckedI32IndexError::Descriptor);
        };
        let selected = i32::try_from(raw).map_err(|_| CheckedI32IndexError::Descriptor)?;
        if !usize::try_from(selected).is_ok_and(|selected| selected < extent) {
            return Err(CheckedI32IndexError::IndexOutOfBounds {
                position,
                value: selected,
            });
        }
    }
    Ok(())
}

impl StaticLifetimePlan {
    pub(crate) fn new(
        capture: CapturedStaticExecution,
        resident_names: &[String],
    ) -> Result<Self, String> {
        Self::new_with_state(capture, resident_names, &[])
    }

    pub(crate) fn new_with_state(
        capture: CapturedStaticExecution,
        resident_names: &[String],
        state_names: &[String],
    ) -> Result<Self, String> {
        Self::new_with_state_and_controls(capture, resident_names, state_names, &[])
    }

    pub(crate) fn new_with_state_and_controls(
        capture: CapturedStaticExecution,
        resident_names: &[String],
        state_names: &[String],
        runtime_controls: &[ReplayInput],
    ) -> Result<Self, String> {
        if capture.owned_capture().is_none() {
            return Err("static lifetime plan requires owned capture backing".into());
        }
        let mut names = BTreeSet::new();
        for name in resident_names {
            if name.is_empty() || !names.insert(name.as_str()) {
                return Err("resident input names must be nonempty and unique".into());
            }
        }
        let mut state = BTreeSet::new();
        for name in state_names {
            if name.is_empty() || !state.insert(name.as_str()) || names.contains(name.as_str()) {
                return Err("state input names must be nonempty, unique, and nonresident".into());
            }
        }
        let known = capture
            .inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(name) = names.iter().find(|name| !known.contains(**name)) {
            return Err(format!("resident input {name} is absent from the capture"));
        }
        if let Some(name) = state.iter().find(|name| !known.contains(**name)) {
            return Err(format!("state input {name} is absent from the capture"));
        }
        let mut control_ids = BTreeSet::new();
        for control in runtime_controls {
            let captured_matches = capture
                .inputs()
                .iter()
                .filter(|input| *input == control)
                .count();
            if captured_matches != 1
                || control.desc.shape.dims() != [1]
                || control.desc.dtype != DType::I32
                || control.desc.bytes != DType::I32.itemsize()
                || control.desc.alignment != DType::I32.itemsize()
                || !control.desc.read_only
                || names.contains(control.name.as_str())
                || state.contains(control.name.as_str())
                || !control_ids.insert(control.desc.id)
            {
                return Err("runtime controls must be exact, unique, and disjoint".into());
            }
        }
        let (resident_inputs, remaining): (Vec<_>, Vec<_>) = capture
            .inputs()
            .iter()
            .cloned()
            .partition(|input| names.contains(input.name.as_str()));
        let (state_inputs, remaining): (Vec<_>, Vec<_>) = remaining
            .into_iter()
            .partition(|input| state.contains(input.name.as_str()));
        let (runtime_controls, transient_inputs): (Vec<_>, Vec<_>) = remaining
            .into_iter()
            .partition(|input| control_ids.contains(&input.desc.id));
        if runtime_controls.len() != control_ids.len() {
            return Err("runtime control inventory is incomplete".into());
        }
        let resident_ids = capture
            .constant_ids()
            .chain(capture.quantized_constant_ids())
            .chain(resident_inputs.iter().map(|input| input.desc.id))
            .chain(state_inputs.iter().map(|input| input.desc.id))
            .collect();
        Ok(Self {
            capture,
            resident_inputs,
            state_inputs,
            runtime_controls,
            transient_inputs,
            resident_ids,
        })
    }

    pub(crate) fn resident_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.resident_inputs.iter().map(|input| input.name.as_str())
    }

    pub(crate) fn resident_inputs(&self) -> &[ReplayInput] {
        &self.resident_inputs
    }

    pub(crate) fn transient_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.transient_inputs
            .iter()
            .map(|input| input.name.as_str())
    }

    pub(crate) fn state_inputs(&self) -> &[ReplayInput] {
        &self.state_inputs
    }

    pub(crate) fn transient_inputs(&self) -> &[ReplayInput] {
        &self.transient_inputs
    }

    pub(crate) fn runtime_controls(&self) -> &[ReplayInput] {
        &self.runtime_controls
    }

    pub(crate) fn resident_ids(&self) -> &BTreeSet<u64> {
        &self.resident_ids
    }

    pub(crate) fn capture(&self) -> &CapturedSchedule {
        self.capture
            .owned_capture()
            .expect("static lifetime plans own their capture")
    }

    pub(crate) fn quantized_constants(&self) -> &BTreeMap<u64, QuantizedTensorData> {
        self.capture.quantized_constants()
    }

    pub(crate) fn validate_quantized_gathers(
        &self,
        values: &BTreeMap<u64, TensorData>,
        allowed_missing: &BTreeSet<u64>,
    ) -> Result<(), StaticQuantizedGatherError> {
        for item in &self.capture().items {
            let Operation::Movement(crate::MovementValue::QuantizedRowGather(plan)) =
                item.kernel.operation()
            else {
                continue;
            };
            let id = plan.indices.index() as u64;
            let Some(indices) = values.get(&id) else {
                if allowed_missing.contains(&id) {
                    continue;
                }
                return Err(StaticQuantizedGatherError::Invalid(format!(
                    "quantized row-gather indices {id} are absent"
                )));
            };
            plan.validate()
                .map_err(|error| StaticQuantizedGatherError::Invalid(error.to_string()))?;
            let rows = plan.weight_desc.logical_shape.dims()[0];
            validate_i32_index_domain(indices, &plan.indices_shape, rows).map_err(|error| {
                match error {
                    CheckedI32IndexError::Descriptor => StaticQuantizedGatherError::Invalid(
                        "quantized row-gather index descriptor mismatch".into(),
                    ),
                    CheckedI32IndexError::IndexOutOfBounds { position, value } => {
                        StaticQuantizedGatherError::IndexOutOfBounds {
                            position,
                            value,
                            rows,
                        }
                    }
                }
            })?;
        }
        Ok(())
    }

    pub(crate) fn stage_resident(
        &self,
        provided: BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<u64, TensorData>, String> {
        let mut values = self.capture.constants().clone();
        CapturedStaticExecution::insert_owned_inputs(
            &self.resident_inputs,
            provided,
            &mut values,
            "resident Metal session",
        )?;
        Ok(values)
    }

    pub(crate) fn stage_initialized(
        &self,
        residents: BTreeMap<String, TensorData>,
        states: BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<u64, TensorData>, String> {
        let mut values = self.capture.constants().clone();
        CapturedStaticExecution::insert_owned_inputs(
            &self.resident_inputs,
            residents,
            &mut values,
            "resident Metal session",
        )?;
        CapturedStaticExecution::insert_owned_inputs(
            &self.state_inputs,
            states,
            &mut values,
            "initial Metal state",
        )?;
        Ok(values)
    }

    pub(crate) fn stage_transient(
        &self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<u64, TensorData>, String> {
        let mut values = BTreeMap::new();
        CapturedStaticExecution::insert_inputs(
            &self.transient_inputs,
            provided,
            &mut values,
            "transient Metal session",
        )?;
        Ok(values)
    }

    pub(crate) fn stage_committed_position(
        &self,
        committed_position: usize,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(usize, usize), String> {
        let [control] = self.runtime_controls.as_slice() else {
            return if self.runtime_controls.is_empty() {
                Ok((0, 0))
            } else {
                Err("sealed append policy requires exactly one runtime control".into())
            };
        };
        if control.desc.shape.dims() != [1]
            || control.desc.dtype != DType::I32
            || control.desc.bytes != DType::I32.itemsize()
            || control.desc.alignment != DType::I32.itemsize()
            || !control.desc.read_only
        {
            return Err("sealed append runtime control descriptor is invalid".into());
        }
        let position = i32::try_from(committed_position)
            .map_err(|_| "sealed append position exceeds I32".to_owned())?;
        let value =
            TensorData::from_scalars([1], DType::I32, [crate::Scalar::I(i64::from(position))])
                .map_err(|error| error.to_string())?;
        if values.insert(control.desc.id, value).is_some() {
            return Err("sealed append runtime control aliases a caller input".into());
        }
        Ok((1, control.desc.bytes))
    }

    pub(crate) fn project(
        &self,
        values: &BTreeMap<u64, TensorData>,
        resident_sources: &BTreeMap<u64, TensorData>,
    ) -> Result<Vec<TensorData>, String> {
        self.capture.project_with_fallback(values, resident_sources)
    }

    pub(crate) fn project_prefix(
        &self,
        count: usize,
        values: &BTreeMap<u64, TensorData>,
        resident_sources: &BTreeMap<u64, TensorData>,
    ) -> Result<Vec<TensorData>, String> {
        self.capture.project_prefix(count, values, resident_sources)
    }

    pub(crate) fn retain_projection_sources(
        &self,
        values: BTreeMap<u64, TensorData>,
    ) -> BTreeMap<u64, TensorData> {
        let mut required = self
            .capture
            .requested()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        required.extend(
            self.capture
                .passthroughs
                .values()
                .map(|alias| alias.source.index() as u64),
        );
        required.retain(|id| !self.capture.constants().contains_key(id));
        values
            .into_iter()
            .filter(|(id, _)| required.contains(id))
            .collect()
    }
}

/// A prepared static-device prefix paired with one authenticated capture ABI.
///
/// Keeping this state in a distinct type makes capture execution impossible on
/// a raw prefix and prevents backend APIs from growing optional runtime modes.
pub struct CapturedStaticPrefix<P> {
    prepared: P,
    capture: CapturedStaticExecution,
}

impl<P> CapturedStaticPrefix<P> {
    pub(crate) fn new(prepared: P, capture: CapturedStaticExecution) -> Self {
        Self { prepared, capture }
    }

    pub(crate) fn transact<E>(
        &self,
        provided: &BTreeMap<String, TensorData>,
        invalid_binding: impl Fn(String) -> E,
        execute: impl FnOnce(&P, &mut BTreeMap<u64, TensorData>) -> Result<(), E>,
    ) -> Result<Vec<TensorData>, E> {
        self.capture.transact(provided, invalid_binding, |values| {
            execute(&self.prepared, values)
        })
    }

    pub(crate) fn transact_mut<E>(
        &mut self,
        provided: &BTreeMap<String, TensorData>,
        invalid_binding: impl Fn(String) -> E,
        execute: impl FnOnce(&mut P, &mut BTreeMap<u64, TensorData>) -> Result<(), E>,
    ) -> Result<Vec<TensorData>, E> {
        self.capture.transact(provided, invalid_binding, |values| {
            execute(&mut self.prepared, values)
        })
    }
}

/// Sealed device adapter for one concrete specialization of an authenticated
/// symbolic body. Planning must be pure; `prepare` is the first method allowed
/// to create device resources.
pub(crate) trait StaticSymbolicBackend: sealed::Sealed {
    type Error;
    type Plan;
    type Prepared;

    fn replay_error(error: ReplayError) -> Self::Error;
    fn invalid_binding(reason: String) -> Self::Error;
    fn internal_error(reason: String) -> Self::Error;
    fn plan(&self, capture: &CapturedSchedule, retained: &[u64])
    -> Result<Self::Plan, Self::Error>;
    fn prepare(&self, plan: Self::Plan) -> Result<Self::Prepared, Self::Error>;
    fn rebind(
        &self,
        _prepared: &mut Self::Prepared,
        plan: Self::Plan,
    ) -> StaticSymbolicRebind<Self::Plan, Self::Error> {
        StaticSymbolicRebind::Prepare(plan)
    }
    fn execute(
        &self,
        prepared: &mut Self::Prepared,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<(), Self::Error>;
    fn cache_keys(&self, prepared: &Self::Prepared) -> Vec<String>;
}

/// Result of attempting to reuse one prepared device executable for a new
/// authenticated concrete specialization. Backends that cannot update an
/// executable return the untouched plan through `Prepare`.
pub(crate) enum StaticSymbolicRebind<P, E> {
    Rebound,
    Prepare(P),
    /// `cached_valid` is true only when the backend failed before mutating the
    /// cached executable. Partial graph-node updates must return false.
    Failed {
        error: E,
        cached_valid: bool,
    },
}

struct CachedStaticSpecialization<P> {
    identity: u64,
    prepared: P,
}

struct PreparedStaticExecution<P> {
    prepared: P,
    outputs: Vec<TensorData>,
    cache_keys: Vec<String>,
}

/// Shared invocation transaction for a bounded symbolic body and one static
/// device backend. The cache intentionally retains only the most recent fully
/// successful concrete specialization.
pub(crate) struct StaticSymbolicProgram<B: StaticSymbolicBackend> {
    body: AuthenticatedSymbolicBody,
    backend: B,
    cached: Mutex<Option<CachedStaticSpecialization<B::Prepared>>>,
}

pub(crate) struct StaticSymbolicRun {
    pub(crate) outputs: Vec<TensorData>,
    pub(crate) body_identity: u64,
    pub(crate) concrete_identity: u64,
    pub(crate) bindings: Vec<(u64, i64)>,
    pub(crate) prepared_now: bool,
    pub(crate) cache_keys: Vec<String>,
}

impl<B: StaticSymbolicBackend> StaticSymbolicProgram<B> {
    pub(crate) fn new(body: AuthenticatedSymbolicBody, backend: B) -> Self {
        Self {
            body,
            backend,
            cached: Mutex::new(None),
        }
    }

    pub(crate) fn body(&self) -> &AuthenticatedSymbolicBody {
        &self.body
    }

    pub(crate) fn run(
        &self,
        invocation: SymbolicInvocation,
    ) -> Result<StaticSymbolicRun, B::Error> {
        let bound = self.body.bind(invocation).map_err(B::replay_error)?;
        self.run_bound(bound)
    }

    fn run_bound(
        &self,
        bound: AuthenticatedSymbolicInvocation,
    ) -> Result<StaticSymbolicRun, B::Error> {
        // Staging authenticates exact descriptors and byte lengths before the
        // pure renderer/static-plan capability gate or any device allocation.
        let projection =
            CapturedStaticExecution::new(&bound.concrete).map_err(B::invalid_binding)?;
        let mut values = projection
            .stage(&bound.inputs)
            .map_err(B::invalid_binding)?;
        let plan = self.backend.plan(&bound.concrete, projection.retained())?;
        let concrete_identity = bound.concrete.identity;
        let mut cached = self
            .cached
            .lock()
            .map_err(|_| B::internal_error("static symbolic cache lock poisoned".into()))?;

        let (outputs, prepared_now, cache_keys) = if let Some(entry) = cached.as_mut() {
            let rebound = if entry.identity != concrete_identity {
                match self.backend.rebind(&mut entry.prepared, plan) {
                    StaticSymbolicRebind::Rebound => true,
                    StaticSymbolicRebind::Prepare(plan) => {
                        let candidate = self.prepare_and_execute(plan, &projection, &mut values)?;
                        *entry = CachedStaticSpecialization {
                            identity: concrete_identity,
                            prepared: candidate.prepared,
                        };
                        return Ok(StaticSymbolicRun {
                            outputs: candidate.outputs,
                            body_identity: self.body.capture().identity,
                            concrete_identity,
                            bindings: bound.canonical,
                            prepared_now: true,
                            cache_keys: candidate.cache_keys,
                        });
                    }
                    StaticSymbolicRebind::Failed {
                        error,
                        cached_valid,
                    } => {
                        if !cached_valid {
                            *cached = None;
                        }
                        return Err(error);
                    }
                }
            } else {
                false
            };
            let result = self
                .backend
                .execute(&mut entry.prepared, &mut values)
                .and_then(|()| projection.project(&values).map_err(B::invalid_binding))
                .and_then(|outputs| self.order_outputs(outputs));
            match result {
                Ok(outputs) => {
                    entry.identity = concrete_identity;
                    let cache_keys = self.backend.cache_keys(&entry.prepared);
                    (outputs, rebound, cache_keys)
                }
                Err(error) => {
                    // A backend failure may have uncertain device
                    // completion. Drop the entry after its backend has
                    // fenced/quarantined as required; a retry prepares a
                    // fresh transaction.
                    *cached = None;
                    return Err(error);
                }
            }
        } else {
            let candidate = self.prepare_and_execute(plan, &projection, &mut values)?;
            *cached = Some(CachedStaticSpecialization {
                identity: concrete_identity,
                prepared: candidate.prepared,
            });
            (candidate.outputs, true, candidate.cache_keys)
        };

        Ok(StaticSymbolicRun {
            outputs,
            body_identity: self.body.capture().identity,
            concrete_identity,
            bindings: bound.canonical,
            prepared_now,
            cache_keys,
        })
    }

    fn prepare_and_execute(
        &self,
        plan: B::Plan,
        projection: &CapturedStaticExecution,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<PreparedStaticExecution<B::Prepared>, B::Error> {
        let mut candidate = self.backend.prepare(plan)?;
        self.backend.execute(&mut candidate, values)?;
        let outputs = projection
            .project(values)
            .map_err(B::invalid_binding)
            .and_then(|outputs| self.order_outputs(outputs))?;
        let cache_keys = self.backend.cache_keys(&candidate);
        Ok(PreparedStaticExecution {
            prepared: candidate,
            outputs,
            cache_keys,
        })
    }

    fn order_outputs(&self, outputs: Vec<TensorData>) -> Result<Vec<TensorData>, B::Error> {
        self.body
            .output_order()
            .iter()
            .map(|position| {
                outputs.get(*position).cloned().ok_or_else(|| {
                    B::invalid_binding(format!(
                        "symbolic requested output position {position} is absent"
                    ))
                })
            })
            .collect()
    }
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
/// Most kernels launch one work item per output element; PrefixScan and
/// coupled Sort launch one item per `(row, inner)` lane, while materializing
/// Bitcast launches one item per raw byte. Logical descriptors remain exact.
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
            Operation::Movement(crate::MovementValue::Plan(plan))
                if matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. }) =>
            {
                let portable = crate::movement_plan::PortableBitcast::new(plan)
                    .map_err(|_| "portable bitcast launch geometry is invalid")?;
                if portable.output_elements() != logical_elements {
                    return Err("bitcast logical output extent mismatch");
                }
                portable.bytes()
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StaticRenderedQuantizedBuffer {
    pub(crate) id: u64,
    pub(crate) desc: QuantizedBufferDesc,
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

impl StaticBufferAllocation {
    pub(crate) fn physical_bytes(self) -> usize {
        if self.bytes == 0 && self.requires_native_handle {
            DType::F32.itemsize()
        } else {
            self.bytes
        }
    }
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
    pub(crate) quantized_buffers: Vec<StaticRenderedQuantizedBuffer>,
    pub(crate) pointer_ids: Vec<u64>,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StaticQuantizedBufferPlan {
    pub(crate) desc: QuantizedBufferDesc,
    pub(crate) requires_native_handle: bool,
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
    dependencies: Vec<usize>,
}

/// Fully validated schedule/render/buffer graph. Constructing this type is pure
/// with respect to native queues, caches, programs, and buffers.
pub(crate) struct StaticSchedulePlan<R> {
    items: Vec<StaticItemPlan<R>>,
    buffers: BTreeMap<u64, StaticBufferPlan>,
    quantized_buffers: BTreeMap<u64, StaticQuantizedBufferPlan>,
    external_inputs: Vec<u64>,
    host_outputs: Vec<u64>,
    protected_outputs: Vec<u64>,
    state_links: Vec<StaticStateLink>,
    append_state_links: Vec<StaticAppendStateLink>,
    host_gathers: Vec<StaticHostGather>,
    allocations: StaticAllocationPlan,
}

/// Runtime-only proof that one internal Gather consumes an affine expansion
/// of a host-validated scalar-or-fixed batch-one I32 transient.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StaticHostGather {
    pub(crate) input: u64,
    pub(crate) input_desc: BufferDesc,
    pub(crate) index: u64,
    pub(crate) output: u64,
    pub(crate) axis: usize,
    pub(crate) axis_extent: usize,
    pub(crate) index_elements: usize,
}

impl StaticHostGather {
    pub(crate) fn input_elements(&self) -> Result<usize, String> {
        self.input_desc
            .shape
            .numel()
            .map_err(|error| error.to_string())
    }
}

fn authenticate_host_index_producer(
    index_item: &ScheduleItem,
    input: u64,
    expected_input_desc: Option<&BufferDesc>,
    index: u64,
    index_shape: &Shape,
    consumers: &[&ScheduleItem],
) -> Result<(), String> {
    index_item
        .kernel
        .validate()
        .map_err(|error| error.to_string())?;
    let [store, end_range] = index_item.kernel.sources() else {
        return Err("host index producer must be one store".into());
    };
    let crate::Operation::Sink = index_item.kernel.operation() else {
        return Err("host index producer must be a sink".into());
    };
    let crate::Operation::Store = store.operation() else {
        return Err("host index producer must contain one store".into());
    };
    let [output_index, value] = store.sources() else {
        return Err("host index store is malformed".into());
    };
    let crate::Operation::Load = value.operation() else {
        return Err("host index producer is not value-preserving".into());
    };
    let [input_index] = value.sources() else {
        return Err("host index load is malformed".into());
    };
    let crate::Operation::EndRange = end_range.operation() else {
        return Err("host index producer has no terminal range".into());
    };
    let [terminal_range] = end_range.sources() else {
        return Err("host index terminal range is malformed".into());
    };
    let crate::Operation::Index(crate::IndexValue::Buffer {
        buffer: output_buffer,
        elements: output_elements,
        input_shape: output_input_shape,
        output_shape: output_output_shape,
        addressing: crate::IndexAddressing::Broadcast,
    }) = output_index.operation()
    else {
        return Err("host index output addressing is invalid".into());
    };
    let crate::Operation::Index(crate::IndexValue::View {
        buffer: input_buffer,
        elements: view_elements,
        input_shape,
        output_shape,
        view,
    }) = input_index.operation()
    else {
        return Err("host index source is not an affine view".into());
    };
    let [output_address, output_range] = output_index.sources() else {
        return Err("host index output addressing is malformed".into());
    };
    let [input_address, input_range] = input_index.sources() else {
        return Err("host index source addressing is malformed".into());
    };
    let crate::Operation::DefineGlobal(output_addressing) = output_address.operation() else {
        return Err("host index output pointer is malformed".into());
    };
    let crate::Operation::DefineGlobal(input_addressing) = input_address.operation() else {
        return Err("host index source pointer is malformed".into());
    };
    let crate::Operation::Range(0) = terminal_range.operation() else {
        return Err("host index range is invalid".into());
    };
    let [extent] = terminal_range.sources() else {
        return Err("host index range extent is malformed".into());
    };
    let crate::Operation::Const(crate::LiteralValue::Int(extent)) = extent.operation() else {
        return Err("host index range extent is not constant".into());
    };
    let [source_binding] = index_item.ordered_inputs() else {
        return Err("host index producer must have one ordered source".into());
    };
    crate::schedule::validate_buffer_desc(&source_binding.desc)
        .map_err(|error| error.to_string())?;
    let input_desc = &source_binding.desc;
    if let Some(expected) = expected_input_desc {
        crate::schedule::validate_buffer_desc(expected).map_err(|error| error.to_string())?;
        // Replay inputs own the physical dense descriptor. Each scheduled
        // materialization owns its local affine view, so one source may feed
        // differently shaped authenticated expansions without making the
        // capture's first retained view authoritative for every consumer.
        if expected.id != input_desc.id
            || expected.shape != input_desc.shape
            || expected.dtype != input_desc.dtype
            || expected.bytes != input_desc.bytes
            || expected.alignment != input_desc.alignment
            || expected.read_only != input_desc.read_only
        {
            return Err("host index physical source is inconsistent".into());
        }
    }
    let Some(captured_view) = input_desc.view.as_ref() else {
        return Err("host index source has no authenticated affine view".into());
    };
    let normalized = captured_view
        .normalized_read()
        .map_err(|error| error.to_string())?;
    let output = index_item.outputs.primary();
    let elements = index_shape.numel().map_err(|error| error.to_string())?;
    let bytes = elements
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| "host index byte extent overflow".to_owned())?;
    let scalar_ty = crate::UType::scalar(DType::I32);
    let range_ty = crate::UType::scalar(DType::I64);
    let input_elements = input_desc
        .shape
        .numel()
        .map_err(|error| error.to_string())?;
    let canonical_fixed_axes = input_elements > 1
        && input_desc.shape.dims() == [1, input_elements]
        && index_shape.rank() == 3
        && index_shape.dims()[..2] == [1, input_elements]
        && normalized.axes.len() == 3
        && normalized.axes[0].stride == 0
        && !normalized.axes[0].reversed
        && normalized.axes[1].stride == 1
        && !normalized.axes[1].reversed
        && normalized.axes[2].stride == 0
        && !normalized.axes[2].reversed;
    let canonical_scalar_axes = input_elements == 1
        && normalized
            .axes
            .iter()
            .all(|axis| axis.stride == 0 && !axis.reversed);
    if index_item.outputs.len() != 1
        || index_item.node.index() as u64 != index
        || output.id != index
        || output.shape != *index_shape
        || output.dtype != DType::I32
        || output.bytes != bytes
        || output.alignment != DType::I32.itemsize()
        || output.view.is_some()
        || output.read_only
        || source_binding.input_node.index() as u64 != input_desc.id
        || source_binding.abi_index != 0
        || input_desc.id != input
        || input_desc.dtype != DType::I32
        || input_elements == 0
        || input_desc.bytes
            != input_elements
                .checked_mul(DType::I32.itemsize())
                .ok_or_else(|| "host Gather input byte extent overflow".to_owned())?
        || input_desc.alignment != DType::I32.itemsize()
        || !input_desc.read_only
        || captured_view.source_shape != input_desc.shape
        || captured_view.logical_shape != *index_shape
        || view != captured_view
        || *input_buffer != input
        || *view_elements != elements
        || input_shape != index_shape
        || output_shape != index_shape
        || *output_buffer != index
        || *output_elements != elements
        || output_input_shape != index_shape
        || output_output_shape != index_shape
        || output_index.ty() != Some(scalar_ty)
        || input_index.ty() != Some(scalar_ty)
        || value.ty() != Some(scalar_ty)
        || output_address.ty() != Some(scalar_ty)
        || input_address.ty() != Some(scalar_ty)
        || terminal_range.ty() != Some(range_ty)
        || *extent
            != i64::try_from(elements).map_err(|_| "host index extent overflow".to_owned())?
        || output_addressing.space != crate::AddressSpace::Global
        || output_addressing.name != format!("b{index}")
        || output_addressing.element != scalar_ty
        || input_addressing.space != crate::AddressSpace::Global
        || input_addressing.name != format!("b{input}")
        || input_addressing.element != scalar_ty
        || !output_range.shares_node_with(terminal_range)
        || !input_range.shares_node_with(terminal_range)
        || index_item.consumers != consumers.iter().map(|item| item.id).collect::<Vec<_>>()
        || consumers
            .iter()
            .any(|item| !item.dependencies.contains(&index_item.id))
        || normalized.offset != 0
        || !(canonical_scalar_axes || canonical_fixed_axes)
    {
        return Err("host Gather index affine provenance is inconsistent".into());
    }
    Ok(())
}

/// Reauthenticates the exact capture-owned affine materialization feeding one
/// raw Gather. This is shared by capture policy construction and static
/// planning; neither boundary trusts graph-local NodeId coincidence.
pub(crate) fn authenticate_host_gather_lineage(
    items: &[ScheduleItem],
    link: &StaticHostGather,
) -> Result<(), String> {
    let gather_items = items
        .iter()
        .filter(|item| item.outputs.iter().any(|output| output.id == link.output))
        .collect::<Vec<_>>();
    let [gather_item] = gather_items.as_slice() else {
        return Err("host Gather output must have one captured owner".into());
    };
    let crate::Operation::Movement(crate::MovementValue::Plan(gather_plan)) =
        gather_item.kernel.operation()
    else {
        return Err("host Gather owner is not a movement plan".into());
    };
    let crate::MovementKernelKind::Gather { index, axis, .. } = &gather_plan.kind else {
        return Err("host Gather owner is not Gather".into());
    };
    let portable = crate::movement_plan::PortableIndexedMovement::new(gather_plan)
        .and_then(|portable| {
            portable.validate_schedule_bindings(gather_item.ordered_inputs())?;
            Ok(portable)
        })
        .map_err(|error| error.to_string())?;
    let gather_output = gather_item.outputs.primary();
    let gather_bytes = gather_plan
        .output_shape
        .numel()
        .map_err(|error| error.to_string())?
        .checked_mul(gather_plan.dtype.itemsize())
        .ok_or_else(|| "host Gather output byte extent overflow".to_owned())?;
    if gather_item.outputs.len() != 1
        || gather_item.node != gather_plan.output
        || gather_plan.dtype != DType::F32
        || gather_output.id != gather_plan.output.index() as u64
        || gather_output.shape != gather_plan.output_shape
        || gather_output.dtype != gather_plan.dtype
        || gather_output.bytes != gather_bytes
        || gather_output.alignment != gather_plan.dtype.itemsize().max(1)
        || gather_output.view.is_some()
        || gather_output.read_only
        || index.node.index() as u64 != link.index
        || *axis != link.axis
        || portable.axis() != link.axis
        || portable.axis_extent() != link.axis_extent
        || portable.index_elements() != link.index_elements
    {
        return Err("host Gather movement geometry is inconsistent".into());
    }

    let index_items = items
        .iter()
        .filter(|item| item.outputs.iter().any(|output| output.id == link.index))
        .collect::<Vec<_>>();
    let [index_item] = index_items.as_slice() else {
        return Err("host Gather index must have one captured producer".into());
    };
    let index_output = index_item.outputs.primary();
    if index_output.shape != index.shape
        || index_output.dtype != index.dtype
        || index_output.id != link.index
    {
        return Err("host Gather affine provenance is inconsistent".into());
    }
    authenticate_host_index_producer(
        index_item,
        link.input,
        Some(&link.input_desc),
        link.index,
        &index_output.shape,
        &[gather_item],
    )
}

/// Reauthenticates one device-produced row index as the exact materialized
/// affine expansion of the append session's scalar position input.
pub(crate) fn authenticate_append_state_index_lineage(
    items: &[ScheduleItem],
    links: &[StaticAppendStateLink],
) -> Result<(), String> {
    let Some(first) = links.first() else {
        return Err("append-state index lineage is absent".into());
    };
    if links.iter().any(|link| {
        link.index != first.index
            || link.position != first.position
            || link.iota != first.iota
            || link.axis != first.axis
            || link.axis_extent != first.axis_extent
            || link.span != first.span
    }) {
        return Err("append-state links do not share one position-derived span".into());
    }
    let index_items = items
        .iter()
        .filter(|item| item.outputs.iter().any(|output| output.id == first.index))
        .collect::<Vec<_>>();
    let [index_item] = index_items.as_slice() else {
        return Err("append-state index must have one captured producer".into());
    };
    let index_output = index_item.outputs.primary();
    let append_outputs = links
        .iter()
        .map(|link| link.output)
        .collect::<BTreeSet<_>>();
    if append_outputs.len() != links.len() {
        return Err("append-state outputs must be unique".into());
    }
    let consumers = items
        .iter()
        .filter(|item| append_outputs.contains(&item.outputs.primary().id))
        .collect::<Vec<_>>();
    if consumers.len() != links.len() {
        return Err("append-state index consumers are incomplete".into());
    }
    if first.span.rows == 1 || first.span.total_elements == 0 {
        return authenticate_host_index_producer(
            index_item,
            first.position,
            None,
            first.index,
            &index_output.shape,
            &consumers,
        );
    }
    authenticate_append_span_producer(items, index_item, first, &consumers)
}

pub(crate) enum AppendSpanEndError {
    Overflow,
    InvalidBinding(String),
}

pub(crate) fn checked_append_span_end(
    committed_position: usize,
    span_rows: usize,
    axis_extent: usize,
) -> Result<usize, AppendSpanEndError> {
    let end = committed_position
        .checked_add(span_rows)
        .ok_or(AppendSpanEndError::Overflow)?;
    let last = end.checked_sub(1).ok_or_else(|| {
        AppendSpanEndError::InvalidBinding("append span must contain at least one row".to_owned())
    })?;
    if end > axis_extent || i32::try_from(last).is_err() {
        return Err(AppendSpanEndError::InvalidBinding(format!(
            "append span {committed_position}..{end} exceeds state extent {axis_extent} or I32 index admission"
        )));
    }
    Ok(end)
}

fn authenticate_append_span_producer(
    items: &[ScheduleItem],
    index_item: &ScheduleItem,
    link: &StaticAppendStateLink,
    consumers: &[&ScheduleItem],
) -> Result<(), String> {
    index_item
        .kernel
        .validate()
        .map_err(|error| error.to_string())?;
    let [store, end_range] = index_item.kernel.sources() else {
        return Err("append span index producer must be one store".into());
    };
    let crate::Operation::Sink = index_item.kernel.operation() else {
        return Err("append span index producer must be a sink".into());
    };
    let crate::Operation::Store = store.operation() else {
        return Err("append span index producer must contain one store".into());
    };
    let [output_index, value] = store.sources() else {
        return Err("append span index store is malformed".into());
    };
    let crate::Operation::GraphBinary(crate::BinaryOp::Add) = value.operation() else {
        return Err("append span index must contain one exact Add".into());
    };
    let [position_load, iota_load] = value.sources() else {
        return Err("append span index Add is malformed".into());
    };
    fn parse_load(load: &crate::UOp) -> Result<(&crate::IndexValue, &crate::UOp), String> {
        let crate::Operation::Load = load.operation() else {
            return Err("append span index operand is not a load".into());
        };
        let [index] = load.sources() else {
            return Err("append span index load is malformed".into());
        };
        let crate::Operation::Index(index_value) = index.operation() else {
            return Err("append span index load has no Index".into());
        };
        Ok((index_value, index))
    }
    let (position_value, position_index) = parse_load(position_load)?;
    let (iota_value, iota_index) = parse_load(iota_load)?;
    let crate::IndexValue::View {
        buffer: position_buffer,
        elements: position_elements,
        input_shape: position_input_shape,
        output_shape: position_output_shape,
        view: position_view,
    } = position_value
    else {
        return Err("append span position is not an affine view".into());
    };
    let crate::IndexValue::View {
        buffer: iota_buffer,
        elements: iota_elements,
        input_shape: iota_input_shape,
        output_shape: iota_output_shape,
        view: iota_view,
    } = iota_value
    else {
        return Err("append span iota is not one affine load".into());
    };
    let output = index_item.outputs.primary();
    let elements = output.shape.numel().map_err(|error| error.to_string())?;
    let position_normalized = position_view
        .normalized_read()
        .map_err(|error| error.to_string())?;
    let ordered_inputs = index_item.ordered_inputs();
    let position_binding = ordered_inputs
        .iter()
        .find(|binding| binding.desc.id == link.position)
        .ok_or_else(|| "append span position binding is absent".to_owned())?;
    let iota_id = link
        .iota
        .ok_or_else(|| "append span ShapeIota identity is absent".to_owned())?;
    let iota_binding = ordered_inputs
        .iter()
        .find(|binding| binding.desc.id == iota_id)
        .ok_or_else(|| "append span ShapeIota binding is absent".to_owned())?;
    crate::schedule::validate_buffer_desc(&position_binding.desc)
        .map_err(|error| error.to_string())?;
    let crate::Operation::Index(crate::IndexValue::Buffer {
        buffer: output_buffer,
        elements: output_elements,
        input_shape: output_input_shape,
        output_shape: output_output_shape,
        addressing: crate::IndexAddressing::Broadcast,
    }) = output_index.operation()
    else {
        return Err("append span output addressing is invalid".into());
    };
    let crate::Operation::EndRange = end_range.operation() else {
        return Err("append span index has no terminal range".into());
    };
    let [terminal_range] = end_range.sources() else {
        return Err("append span terminal range is malformed".into());
    };
    let crate::Operation::Range(0) = terminal_range.operation() else {
        return Err("append span range is invalid".into());
    };
    if index_item.outputs.len() != 1
        || index_item.node.index() as u64 != link.index
        || output.id != link.index
        || output.dtype != DType::I32
        || output.view.is_some()
        || output.read_only
        || *output_buffer != link.index
        || *output_elements != elements
        || output_input_shape != &output.shape
        || output_output_shape != &output.shape
        || *position_buffer != link.position
        || *position_elements != elements
        || position_input_shape != &output.shape
        || position_output_shape != &output.shape
        || ordered_inputs.len() != 2
        || position_binding.desc.id != link.position
        || position_binding.abi_index != 0
        || position_binding.desc.dtype != DType::I32
        || position_binding.desc.shape.numel().ok() != Some(1)
        || position_binding.desc.bytes != DType::I32.itemsize()
        || position_binding.desc.alignment != DType::I32.itemsize()
        || !position_binding.desc.read_only
        || position_binding.desc.view.as_ref() != Some(position_view)
        || position_view.source_shape != position_binding.desc.shape
        || position_view.logical_shape != output.shape
        || position_index.ty() != Some(crate::UType::scalar(DType::I32))
        || iota_load.ty() != Some(crate::UType::scalar(DType::I32))
        || value.ty() != Some(crate::UType::scalar(DType::I32))
        || position_normalized.offset != 0
        || position_normalized
            .axes
            .iter()
            .any(|axis| axis.stride != 0 || axis.reversed)
        || !position_index.sources()[1].shares_node_with(terminal_range)
        || !iota_index.sources()[1].shares_node_with(terminal_range)
        || !output_index.sources()[1].shares_node_with(terminal_range)
        || index_item.consumers != consumers.iter().map(|item| item.id).collect::<Vec<_>>()
        || consumers
            .iter()
            .any(|item| !item.dependencies.contains(&index_item.id))
    {
        return Err("append span index provenance is inconsistent".into());
    }
    authenticate_append_span_iota(
        items,
        index_item,
        iota_id,
        AppendSpanIotaLoad {
            binding: iota_binding,
            view: iota_view,
            buffer: *iota_buffer,
            elements: *iota_elements,
            input_shape: iota_input_shape,
            output_shape: iota_output_shape,
        },
        link,
    )
}

struct AppendSpanIotaLoad<'a> {
    binding: &'a ScheduleInputBinding,
    view: &'a crate::AffineView,
    buffer: u64,
    elements: usize,
    input_shape: &'a Shape,
    output_shape: &'a Shape,
}

fn authenticate_append_span_iota(
    items: &[ScheduleItem],
    index_item: &ScheduleItem,
    iota_id: u64,
    load: AppendSpanIotaLoad<'_>,
    link: &StaticAppendStateLink,
) -> Result<(), String> {
    let normalized = load
        .view
        .normalized_read()
        .map_err(|error| error.to_string())?;
    let source_shape = Shape::from([link.span.rows]);
    let expected_bytes = link
        .span
        .rows
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| "append span ShapeIota byte extent overflow".to_owned())?;
    crate::schedule::validate_buffer_desc(&load.binding.desc).map_err(|error| error.to_string())?;
    if load.buffer != iota_id
        || load.elements != link.span.total_elements
        || load.input_shape != load.output_shape
        || load.output_shape != &load.view.logical_shape
        || load.view.source_shape != source_shape
        || load.binding.input_node.index() as u64 != iota_id
        || load.binding.abi_index != 1
        || load.binding.desc.id != iota_id
        || load.binding.desc.shape != source_shape
        || load.binding.desc.dtype != DType::I32
        || load.binding.desc.bytes != expected_bytes
        || load.binding.desc.alignment != DType::I32.itemsize()
        || !load.binding.desc.read_only
        || load.binding.desc.view.as_ref() != Some(load.view)
        || normalized.offset != 0
        || normalized.axes.len() != load.output_shape.rank()
        || normalized
            .axes
            .iter()
            .enumerate()
            .any(|(axis, map)| map.reversed || map.stride != usize::from(axis == link.axis))
    {
        return Err("append span ShapeIota affine provenance is inconsistent".into());
    }

    let producers = items
        .iter()
        .filter(|item| item.outputs.iter().any(|output| output.id == iota_id))
        .collect::<Vec<_>>();
    let [producer] = producers.as_slice() else {
        return Err("append span ShapeIota must have one captured producer".into());
    };
    producer
        .kernel
        .validate()
        .map_err(|error| error.to_string())?;
    let [store, end_range] = producer.kernel.sources() else {
        return Err("append span ShapeIota producer must be one store".into());
    };
    let crate::Operation::Sink = producer.kernel.operation() else {
        return Err("append span ShapeIota producer must be a sink".into());
    };
    let crate::Operation::Store = store.operation() else {
        return Err("append span ShapeIota producer must contain one store".into());
    };
    let [store_index, stored_value] = store.sources() else {
        return Err("append span ShapeIota store is malformed".into());
    };
    let crate::Operation::Index(crate::IndexValue::Buffer {
        buffer: store_buffer,
        elements: store_elements,
        input_shape: store_input_shape,
        output_shape: store_output_shape,
        addressing: crate::IndexAddressing::Broadcast,
    }) = store_index.operation()
    else {
        return Err("append span ShapeIota output addressing is invalid".into());
    };
    let crate::Operation::EndRange = end_range.operation() else {
        return Err("append span ShapeIota has no terminal range".into());
    };
    let [range] = end_range.sources() else {
        return Err("append span ShapeIota terminal range is malformed".into());
    };
    let crate::Operation::Range(0) = range.operation() else {
        return Err("append span ShapeIota range is invalid".into());
    };
    let [extent] = range.sources() else {
        return Err("append span ShapeIota range extent is malformed".into());
    };
    let crate::Operation::Const(crate::LiteralValue::Int(range_extent)) = extent.operation() else {
        return Err("append span ShapeIota range extent is not an integer literal".into());
    };
    let coordinate_is_exact = match (stored_value.operation(), stored_value.sources()) {
        (crate::Operation::Cast, [coordinate]) => {
            stored_value.ty() == Some(crate::UType::scalar(DType::I32))
                && coordinate.ty() == Some(crate::UType::scalar(DType::I64))
                && coordinate.shares_node_with(range)
        }
        _ => {
            stored_value.ty() == Some(crate::UType::scalar(DType::I32))
                && range.ty() == Some(crate::UType::scalar(DType::I32))
                && stored_value.shares_node_with(range)
        }
    };
    let output = producer.outputs.primary();
    if producer.outputs.len() != 1
        || producer.node.index() as u64 != iota_id
        || output.id != iota_id
        || output.shape != source_shape
        || output.dtype != DType::I32
        || output.bytes != expected_bytes
        || output.alignment != DType::I32.itemsize()
        || output.read_only
        || output.view.is_some()
        || producer.boundary.is_some()
        || !producer.inputs.is_empty()
        || !producer.ordered_inputs().is_empty()
        || !producer.ordered_quantized_inputs().is_empty()
        || !producer.external_materializations.is_empty()
        || !producer.dependencies.is_empty()
        || producer.consumers.as_slice() != [index_item.id]
        || !index_item.dependencies.contains(&producer.id)
        || *store_buffer != iota_id
        || *store_elements != link.span.rows
        || store_input_shape != &source_shape
        || store_output_shape != &source_shape
        || !store_index.sources()[1].shares_node_with(range)
        || usize::try_from(*range_extent).ok() != Some(link.span.rows)
        || !coordinate_is_exact
    {
        return Err("append span ShapeIota producer provenance is inconsistent".into());
    }
    Ok(())
}

/// One authenticated fixed-shape input/output pair whose two private slots
/// alternate ownership between successful static-device invocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StaticStateLink {
    pub(crate) input: u64,
    pub(crate) output: u64,
}

/// Checked geometry for one fixed append span. Totals stay distinct from one
/// row so execution accounting cannot accidentally multiply them twice.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AppendSpanGeometry {
    pub(crate) rows: usize,
    pub(crate) elements_per_row: usize,
    pub(crate) bytes_per_row: usize,
    pub(crate) total_elements: usize,
    pub(crate) total_bytes: usize,
}

/// One authenticated full-buffer logical Scatter output that aliases its
/// exclusively owned input allocation while writing one fixed row span.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StaticAppendStateLink {
    pub(crate) input: u64,
    pub(crate) output: u64,
    pub(crate) position: u64,
    pub(crate) index: u64,
    pub(crate) iota: Option<u64>,
    pub(crate) updates: u64,
    pub(crate) axis: usize,
    pub(crate) axis_extent: usize,
    pub(crate) span: AppendSpanGeometry,
}

struct StaticOutputPolicy<'a> {
    host_outputs: &'a [u64],
    protected_outputs: &'a [u64],
    state_links: &'a [StaticStateLink],
    append_state_links: &'a [StaticAppendStateLink],
    host_gathers: &'a [StaticHostGather],
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
    fn index_out_of_bounds(axis: usize, index: usize, value: i32, dim: usize) -> Self::Error {
        Self::invalid_binding(format!(
            "indexed movement axis {axis} has value {value} at logical index {index}, outside [0, {dim})"
        ))
    }
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
    fn allocate_quantized(
        &self,
        _plan: &StaticQuantizedBufferPlan,
    ) -> Result<Self::Buffer, Self::Error> {
        Err(Self::unsupported(
            "packed static buffers are unsupported by this device adapter".into(),
        ))
    }
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
        &self.host_outputs
    }

    #[cfg(test)]
    pub(crate) fn host_outputs(&self) -> &[u64] {
        &self.host_outputs
    }

    #[cfg(test)]
    pub(crate) fn protected_outputs(&self) -> &[u64] {
        &self.protected_outputs
    }

    pub(crate) fn allocations(&self) -> &StaticAllocationPlan {
        &self.allocations
    }

    pub(crate) fn quantized_buffers(&self) -> &BTreeMap<u64, StaticQuantizedBufferPlan> {
        &self.quantized_buffers
    }

    pub(crate) fn append_state_links(&self) -> &[StaticAppendStateLink] {
        &self.append_state_links
    }

    pub(crate) fn host_gathers(&self) -> &[StaticHostGather] {
        &self.host_gathers
    }

    pub(crate) fn build<A>(
        adapter: &A,
        items: &[ScheduleItem],
        retained: Option<&[u64]>,
    ) -> Result<Self, A::Error>
    where
        A: StaticPlanAdapter<Rendered = R>,
    {
        let outputs = retained.map(|outputs| StaticOutputPolicy {
            host_outputs: outputs,
            protected_outputs: outputs,
            state_links: &[],
            append_state_links: &[],
            host_gathers: &[],
        });
        Self::build_with_outputs(adapter, items, outputs)
    }

    pub(crate) fn build_with_output_policy<A>(
        adapter: &A,
        items: &[ScheduleItem],
        host_outputs: &[u64],
        protected_outputs: &[u64],
        state_links: &[StaticStateLink],
        host_gathers: &[StaticHostGather],
    ) -> Result<Self, A::Error>
    where
        A: StaticPlanAdapter<Rendered = R>,
    {
        Self::build_with_outputs(
            adapter,
            items,
            Some(StaticOutputPolicy {
                host_outputs,
                protected_outputs,
                state_links,
                append_state_links: &[],
                host_gathers,
            }),
        )
    }

    pub(crate) fn build_with_append_policy<A>(
        adapter: &A,
        items: &[ScheduleItem],
        host_outputs: &[u64],
        protected_outputs: &[u64],
        append_state_links: &[StaticAppendStateLink],
        host_gathers: &[StaticHostGather],
    ) -> Result<Self, A::Error>
    where
        A: StaticPlanAdapter<Rendered = R>,
    {
        Self::build_with_outputs(
            adapter,
            items,
            Some(StaticOutputPolicy {
                host_outputs,
                protected_outputs,
                state_links: &[],
                append_state_links,
                host_gathers,
            }),
        )
    }

    fn build_with_outputs<A>(
        adapter: &A,
        items: &[ScheduleItem],
        outputs: Option<StaticOutputPolicy<'_>>,
    ) -> Result<Self, A::Error>
    where
        A: StaticPlanAdapter<Rendered = R>,
    {
        let mut planned = Vec::with_capacity(items.len());
        let mut buffers = BTreeMap::<u64, StaticBufferPlan>::new();
        let mut quantized_buffers = BTreeMap::<u64, StaticQuantizedBufferPlan>::new();
        let mut buffer_order = Vec::new();
        let mut producers = BTreeMap::<u64, usize>::new();
        let append_by_output = outputs
            .as_ref()
            .map(|policy| {
                policy
                    .append_state_links
                    .iter()
                    .map(|link| (link.output, *link))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        if outputs
            .as_ref()
            .is_some_and(|policy| append_by_output.len() != policy.append_state_links.len())
        {
            return Err(A::invalid_binding(
                "static append-state outputs must be unique".into(),
            ));
        }

        validate_prefix::<A>(items)?;
        if let Some(policy) = &outputs
            && !policy.append_state_links.is_empty()
        {
            authenticate_append_state_index_lineage(items, policy.append_state_links)
                .map_err(A::invalid_binding)?;
        }

        for (item_index, item) in items.iter().enumerate() {
            if item.boundary.is_some() || item.is_effect() {
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
            let mut input_ids = rendered
                .buffers
                .iter()
                .filter(|buffer| buffer.role == StaticBufferRole::Input)
                .map(|buffer| buffer.id)
                .collect::<Vec<_>>();
            input_ids.extend(rendered.quantized_buffers.iter().map(|buffer| buffer.id));
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
            let expected_work_items = append_by_output
                .get(&item.outputs.primary().id)
                .map_or(launch.work_items, |link| link.span.total_elements);
            if rendered.extent != expected_work_items {
                return Err(A::invalid_binding(
                    "rendered launch extent mismatches scheduled output".into(),
                ));
            }

            let mut item_ids = BTreeSet::new();
            for use_ in &rendered.buffers {
                if !item_ids.insert(use_.id) || quantized_buffers.contains_key(&use_.id) {
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
            for use_ in &rendered.quantized_buffers {
                if !item_ids.insert(use_.id) || buffers.contains_key(&use_.id) {
                    return Err(A::invalid_binding(format!(
                        "duplicate logical buffer {} in one packed ABI",
                        use_.id
                    )));
                }
                use_.desc
                    .validate_metadata()
                    .map_err(|error| A::invalid_binding(error.to_string()))?;
                let requires_native_handle = rendered.extent != 0;
                match quantized_buffers.get_mut(&use_.id) {
                    Some(existing) if existing.desc == use_.desc => {
                        existing.requires_native_handle |= requires_native_handle;
                    }
                    Some(_) => {
                        return Err(A::invalid_binding(format!(
                            "conflicting packed descriptor for logical buffer {}",
                            use_.id
                        )));
                    }
                    None => {
                        quantized_buffers.insert(
                            use_.id,
                            StaticQuantizedBufferPlan {
                                desc: use_.desc.clone(),
                                requires_native_handle,
                            },
                        );
                    }
                }
            }
            if rendered.pointer_ids.len() != item_ids.len()
                || rendered
                    .pointer_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    != item_ids
            {
                return Err(A::invalid_binding(
                    "rendered pointer order does not cover its dense and packed ABI".into(),
                ));
            }
            planned.push(StaticItemPlan {
                rendered: rendered.artifact,
                cache_key: rendered.cache_key,
                extent: rendered.extent,
                buffer_ids: rendered.pointer_ids,
                input_ids,
                dependencies: item
                    .dependencies
                    .iter()
                    .map(|dependency| usize::try_from(*dependency).map_err(|_| A::overflow()))
                    .collect::<Result<Vec<_>, _>>()?,
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
        let validate_outputs = |ids: &[u64], label: &str| -> Result<Vec<u64>, A::Error> {
            let mut unique = BTreeSet::new();
            for id in ids {
                if !unique.insert(*id) {
                    return Err(A::invalid_binding(format!(
                        "{label} logical output {id} is duplicated"
                    )));
                }
                if !producers.contains_key(id) {
                    return Err(A::invalid_binding(format!(
                        "{label} logical output {id} has no prefix producer"
                    )));
                }
            }
            Ok(ids.to_vec())
        };
        let (host_outputs, protected_outputs, state_links, append_state_links, host_gathers) =
            match outputs {
                Some(policy) => {
                    let host = validate_outputs(policy.host_outputs, "host")?;
                    let protected = validate_outputs(policy.protected_outputs, "protected")?;
                    let protected_set = protected.iter().copied().collect::<BTreeSet<_>>();
                    if let Some(id) = host.iter().find(|id| !protected_set.contains(id)) {
                        return Err(A::invalid_binding(format!(
                            "host logical output {id} is not protected"
                        )));
                    }
                    (
                        host,
                        protected,
                        policy.state_links.to_vec(),
                        policy.append_state_links.to_vec(),
                        policy.host_gathers.to_vec(),
                    )
                }
                // Public prepared-prefix APIs historically materialize every item
                // output into the caller map. Exact internal consumers pass an
                // explicit retained set through `prepare_for_outputs` instead.
                None => {
                    let all = items
                        .iter()
                        .flat_map(|item| item.outputs.iter().map(|output| output.id))
                        .collect::<Vec<_>>();
                    (all.clone(), all, Vec::new(), Vec::new(), Vec::new())
                }
            };
        if !items.is_empty() && protected_outputs.is_empty() {
            return Err(A::invalid_binding(
                "static prefix has no protected output".into(),
            ));
        }
        for id in &protected_outputs {
            buffers.get(id).ok_or_else(|| {
                A::invalid_binding(format!("protected logical output {id} is absent"))
            })?;
        }
        let external_inputs = buffer_order
            .iter()
            .copied()
            .filter(|id| buffers[id].producer.is_none())
            .collect::<Vec<_>>();
        let mut allocation_protected = protected_outputs.clone();
        allocation_protected.extend(state_links.iter().map(|link| link.input));
        allocation_protected.extend(append_state_links.iter().map(|link| link.input));
        let append_aliases = append_state_links
            .iter()
            .map(|link| (link.output, link.input))
            .collect::<BTreeMap<_, _>>();
        let allocations = build_static_allocation_plan::<A>(
            &planned,
            &buffers,
            &buffer_order,
            &external_inputs,
            &allocation_protected,
            &append_aliases,
        )?;

        let mut state_ids = BTreeSet::new();
        for link in &state_links {
            if link.input == link.output
                || !state_ids.insert(link.input)
                || !state_ids.insert(link.output)
            {
                return Err(A::invalid_binding(
                    "static state links must own distinct logical buffers".into(),
                ));
            }
            let input = buffers.get(&link.input).ok_or_else(|| {
                A::invalid_binding(format!("static state input {} is absent", link.input))
            })?;
            let output = buffers.get(&link.output).ok_or_else(|| {
                A::invalid_binding(format!("static state output {} is absent", link.output))
            })?;
            if input.producer.is_some()
                || output.producer.is_none()
                || !external_inputs.contains(&link.input)
                || !protected_outputs.contains(&link.output)
                || host_outputs.contains(&link.output)
                || input.dtype != output.dtype
                || input.source_shape != output.source_shape
                || input.elements != output.elements
                || input.bytes != output.bytes
                || input.alignment != output.alignment
                || (input.bytes != 0
                    && (!allocations.logical_slots.contains_key(&link.input)
                        || !allocations.logical_slots.contains_key(&link.output)
                        || allocations.logical_slots.get(&link.input)
                            == allocations.logical_slots.get(&link.output)))
            {
                return Err(A::invalid_binding(
                    "static state link ownership or descriptor is invalid".into(),
                ));
            }
        }

        for link in &append_state_links {
            if link.input == link.output
                || state_ids.contains(&link.input)
                || state_ids.contains(&link.output)
                || !state_ids.insert(link.input)
                || !state_ids.insert(link.output)
                || link.span.rows == 0
                || link.iota.is_some() != (link.span.rows > 1 && link.span.total_elements > 0)
                || link.span.rows > link.axis_extent
                || link.axis_extent == 0 && link.span.total_elements != 0
                || link.span.total_elements
                    != link
                        .span
                        .rows
                        .checked_mul(link.span.elements_per_row)
                        .ok_or_else(A::overflow)?
                || link.span.bytes_per_row
                    != link
                        .span
                        .elements_per_row
                        .checked_mul(DType::F32.itemsize())
                        .ok_or_else(A::overflow)?
                || link.span.total_bytes
                    != link
                        .span
                        .total_elements
                        .checked_mul(DType::F32.itemsize())
                        .ok_or_else(A::overflow)?
            {
                return Err(A::invalid_binding(
                    "static append-state declaration is inconsistent".into(),
                ));
            }
            let input = buffers.get(&link.input).ok_or_else(|| {
                A::invalid_binding(format!("static append input {} is absent", link.input))
            })?;
            let output = buffers.get(&link.output).ok_or_else(|| {
                A::invalid_binding(format!("static append output {} is absent", link.output))
            })?;
            let index = buffers.get(&link.index).ok_or_else(|| {
                A::invalid_binding(format!("static append index {} is absent", link.index))
            })?;
            let position = buffers.get(&link.position).ok_or_else(|| {
                A::invalid_binding(format!(
                    "static append position {} is absent",
                    link.position
                ))
            })?;
            let updates = buffers.get(&link.updates).ok_or_else(|| {
                A::invalid_binding(format!("static append updates {} is absent", link.updates))
            })?;
            let producer = output
                .producer
                .ok_or_else(|| A::invalid_binding("static append output has no producer".into()))?;
            let update_producer = updates.producer.ok_or_else(|| {
                A::invalid_binding("static append update has no device producer".into())
            })?;
            let index_producer = index.producer.ok_or_else(|| {
                A::invalid_binding("static append index has no device producer".into())
            })?;
            let item = &planned[producer];
            let source_item = &items[producer];
            if input.producer.is_some()
                || !external_inputs.contains(&link.input)
                || !protected_outputs.contains(&link.output)
                || host_outputs.contains(&link.output)
                || position.producer.is_some()
                || !external_inputs.contains(&link.position)
                || index_producer >= producer
                || update_producer >= producer
                || !source_item
                    .dependencies
                    .contains(&items[update_producer].id)
                || !source_item.dependencies.contains(&items[index_producer].id)
                || input.dtype != DType::F32
                || output.dtype != DType::F32
                || index.dtype != DType::I32
                || position.dtype != DType::I32
                || position.source_shape.dims() != [1]
                || position.elements != 1
                || position.bytes != DType::I32.itemsize()
                || position.alignment != DType::I32.itemsize()
                || index.source_shape.dims().get(link.axis) != Some(&link.span.rows)
                || updates.dtype != DType::F32
                || index.source_shape != updates.source_shape
                || input.source_shape != output.source_shape
                || input.elements != output.elements
                || input.bytes != output.bytes
                || input.alignment != output.alignment
                || index.elements != link.span.total_elements
                || updates.elements != link.span.total_elements
                || item.extent != link.span.total_elements
                || source_item.outputs.len() != 1
                || source_item.outputs.primary().id != link.output
                || !item.input_ids.contains(&link.input)
                || !item.input_ids.contains(&link.index)
                || !item.input_ids.contains(&link.updates)
                || allocations.logical_slots.get(&link.input)
                    != allocations.logical_slots.get(&link.output)
            {
                return Err(A::invalid_binding(
                    "static append-state ownership or geometry is invalid".into(),
                ));
            }
        }

        let mut gather_outputs = BTreeSet::new();
        let mut gather_inputs = BTreeSet::new();
        for link in &host_gathers {
            authenticate_host_gather_lineage(items, link).map_err(A::invalid_binding)?;
            if link.axis_extent == 0 && link.index_elements != 0
                || !gather_outputs.insert(link.output)
                || !gather_inputs.insert(link.input)
                || link.input == link.index
                || host_outputs.contains(&link.output)
                || protected_outputs.contains(&link.output)
                || state_ids.contains(&link.input)
                || state_ids.contains(&link.output)
            {
                return Err(A::invalid_binding(
                    "static host Gather declaration is inconsistent".into(),
                ));
            }
            let source = buffers.get(&link.input).ok_or_else(|| {
                A::invalid_binding(format!("host Gather input {} is absent", link.input))
            })?;
            let index = buffers.get(&link.index).ok_or_else(|| {
                A::invalid_binding(format!("host Gather index {} is absent", link.index))
            })?;
            let output = buffers.get(&link.output).ok_or_else(|| {
                A::invalid_binding(format!("host Gather output {} is absent", link.output))
            })?;
            let index_producer = index.producer.ok_or_else(|| {
                A::invalid_binding("host Gather index has no physical producer".into())
            })?;
            let producer = output
                .producer
                .ok_or_else(|| A::invalid_binding("host Gather output has no producer".into()))?;
            let input_elements = link.input_elements().map_err(A::invalid_binding)?;
            let input_bytes = input_elements
                .checked_mul(DType::I32.itemsize())
                .ok_or_else(A::overflow)?;
            let slots_are_distinct = index.bytes == 0
                || match (
                    allocations.logical_slots.get(&link.input),
                    allocations.logical_slots.get(&link.index),
                    allocations.logical_slots.get(&link.output),
                ) {
                    (Some(input), Some(index), Some(output)) => input != index && index != output,
                    _ => false,
                };
            if source.producer.is_some()
                || !external_inputs.contains(&link.input)
                || source.dtype != DType::I32
                || source.elements != input_elements
                || source.bytes != input_bytes
                || index.dtype != DType::I32
                || index.elements != link.index_elements
                || index_producer >= producer
                || !items[producer]
                    .dependencies
                    .contains(&items[index_producer].id)
                || !planned[producer].input_ids.contains(&link.index)
                || items[producer].outputs.len() != 1
                || items[producer].outputs.primary().id != link.output
                || !slots_are_distinct
            {
                return Err(A::invalid_binding(
                    "static host Gather ownership or geometry is invalid".into(),
                ));
            }
        }

        Ok(Self {
            items: planned,
            buffers,
            quantized_buffers,
            external_inputs,
            host_outputs,
            protected_outputs,
            state_links,
            append_state_links,
            host_gathers,
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
    aliases: &BTreeMap<u64, u64>,
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
            let policy = if aliases.contains_key(id) || !required.contains(id) {
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
    for (alias, source) in aliases {
        let Some(slot) = logical_slots.get(source).copied() else {
            if buffers[source].bytes == 0 {
                continue;
            }
            return Err(A::invalid_binding(format!(
                "static alias source {source} has no allocation"
            )));
        };
        if logical_slots.insert(*alias, slot).is_some() {
            return Err(A::invalid_binding(format!(
                "static alias output {alias} already has an allocation"
            )));
        }
    }
    let slots = slots
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| A::invalid_binding("static allocation slot is vacant".into()))?;
    for item in items.iter().filter(|item| item.extent != 0) {
        let dense_ids = item
            .buffer_ids
            .iter()
            .filter(|id| buffers.contains_key(id))
            .copied()
            .collect::<Vec<_>>();
        let item_slots = dense_ids
            .iter()
            .map(|id| logical_slots.get(id).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                A::invalid_binding("nonzero static item has an unallocated logical buffer".into())
            })?;
        for lhs in 0..item_slots.len() {
            for rhs in lhs + 1..item_slots.len() {
                if item_slots[lhs] == item_slots[rhs]
                    && aliases.get(&dense_ids[lhs]) != Some(&dense_ids[rhs])
                    && aliases.get(&dense_ids[rhs]) != Some(&dense_ids[lhs])
                {
                    return Err(A::invalid_binding(
                        "distinct logical buffers in one static item alias a physical slot".into(),
                    ));
                }
            }
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

    pub(crate) fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }
}

struct PreparedStaticItem<K> {
    kernel: Option<K>,
    cache_key: Option<String>,
    extent: usize,
    buffer_ids: Vec<u64>,
}

/// Exact successful host/device activity for one prepared static transaction.
/// Counts describe host API copies and launches, not PCIe traffic or GPU time.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct StaticExecutionReport {
    pub(crate) h2d_calls: usize,
    pub(crate) h2d_bytes: usize,
    pub(crate) d2h_calls: usize,
    pub(crate) d2h_bytes: usize,
    pub(crate) kernel_launches: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StaticHostOutputSelection {
    All,
    None,
}

/// Prepared thread-confined resources for a fully validated static plan.
pub(crate) struct PreparedStaticSchedule<A: StaticDeviceAdapter> {
    adapter: A,
    queue: Option<A::Queue>,
    items: Vec<PreparedStaticItem<A::Kernel>>,
    slots: Vec<A::Buffer>,
    logical_slots: BTreeMap<u64, usize>,
    buffer_plans: BTreeMap<u64, StaticBufferPlan>,
    quantized_buffers: BTreeMap<u64, A::Buffer>,
    quantized_plans: BTreeMap<u64, StaticQuantizedBufferPlan>,
    external_inputs: Vec<u64>,
    host_outputs: Vec<u64>,
    state_links: Vec<StaticStateLink>,
    append_state_links: Vec<StaticAppendStateLink>,
    host_gathers: Vec<StaticHostGather>,
    compiled_cache_keys: Vec<String>,
}

/// Prepared static resources after one authenticated immutable input set has
/// been uploaded successfully. The owned set is the only set skipped later.
pub(crate) struct InitializedStaticSchedule<A: StaticDeviceAdapter> {
    prepared: PreparedStaticSchedule<A>,
    resident_ids: BTreeSet<u64>,
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
            quantized_buffers: quantized_plans,
            external_inputs,
            host_outputs,
            protected_outputs,
            state_links,
            append_state_links,
            host_gathers,
            allocations,
        } = plan;
        if host_outputs
            .iter()
            .any(|id| !protected_outputs.contains(id))
        {
            return Err(A::invalid_binding(
                "static plan host output is not protected".into(),
            ));
        }
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
        let mut quantized_buffers = BTreeMap::new();
        for (id, packed) in &quantized_plans {
            if packed.requires_native_handle {
                quantized_buffers.insert(*id, adapter.allocate_quantized(packed)?);
            }
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
            quantized_buffers,
            quantized_plans,
            external_inputs,
            host_outputs,
            state_links,
            append_state_links,
            host_gathers,
            compiled_cache_keys,
        })
    }

    pub(crate) fn cache_len(&self) -> usize {
        self.adapter.cache_len()
    }

    pub(crate) fn compiled_cache_keys(&self) -> Vec<String> {
        self.compiled_cache_keys.clone()
    }

    pub(crate) fn kernels(&self) -> impl Iterator<Item = &A::Kernel> {
        self.items.iter().filter_map(|item| item.kernel.as_ref())
    }

    fn buffer(&self, id: u64) -> Option<&A::Buffer> {
        self.quantized_buffers.get(&id).or_else(|| {
            self.logical_slots
                .get(&id)
                .and_then(|slot| self.slots.get(*slot))
        })
    }

    fn buffer_for_epoch(&self, id: u64, alternate: bool) -> Option<&A::Buffer> {
        let mapped = self.state_links.iter().find_map(|link| {
            if id == link.input {
                Some(if alternate { link.output } else { link.input })
            } else if id == link.output {
                Some(if alternate { link.input } else { link.output })
            } else {
                None
            }
        });
        let id = mapped.unwrap_or(id);
        let id = self
            .append_state_links
            .iter()
            .find_map(|link| (id == link.output).then_some(link.input))
            .unwrap_or(id);
        self.buffer(id)
    }

    fn validated_uploads(
        &self,
        values: &BTreeMap<u64, TensorData>,
        upload: impl Fn(u64) -> bool,
    ) -> Result<Vec<(u64, Vec<u8>)>, A::Error> {
        let mut uploads = Vec::with_capacity(self.external_inputs.len());
        for id in &self.external_inputs {
            if !upload(*id) {
                continue;
            }
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
            // Native zero-byte sentinels carry pointer identity only. Their
            // logical descriptor is validated above, but adapter writes are
            // intentionally reserved for observable payload bytes.
            if !bytes.is_empty() && self.buffer(*id).is_some() {
                uploads.push((*id, bytes));
            }
        }
        Ok(uploads)
    }

    fn validate_host_gathers(&self, values: &BTreeMap<u64, TensorData>) -> Result<(), A::Error> {
        for link in &self.host_gathers {
            if link.index_elements == 0 {
                continue;
            }
            let value = values.get(&link.input).ok_or_else(|| {
                A::invalid_binding(format!("host Gather input {} is absent", link.input))
            })?;
            let input_elements = link.input_elements().map_err(A::invalid_binding)?;
            let expected_bytes = input_elements
                .checked_mul(DType::I32.itemsize())
                .ok_or_else(A::overflow)?;
            if value.dtype() != DType::I32
                || value.shape() != &self.buffer_plans[&link.input].source_shape
                || value.len() != input_elements
                || self.buffer_plans[&link.input].bytes != expected_bytes
            {
                return Err(A::invalid_binding(
                    "host Gather input descriptor mismatch".into(),
                ));
            }
            validate_i32_index_domain(
                value,
                &self.buffer_plans[&link.input].source_shape,
                link.axis_extent,
            )
            .map_err(|error| match error {
                CheckedI32IndexError::Descriptor => {
                    A::invalid_binding("host Gather input descriptor mismatch".into())
                }
                CheckedI32IndexError::IndexOutOfBounds { position, value } => {
                    A::index_out_of_bounds(link.axis, position, value, link.axis_extent)
                }
            })?;
        }
        Ok(())
    }

    /// Validates and uploads the selected immutable external inputs once. A
    /// failed upload leaves construction unpublished.
    pub(crate) fn initialize_resident_with_quantized(
        self,
        values: &BTreeMap<u64, TensorData>,
        resident_ids: &BTreeSet<u64>,
        quantized: &BTreeMap<u64, QuantizedTensorData>,
    ) -> Result<(InitializedStaticSchedule<A>, StaticExecutionReport), A::Error> {
        if let Some(id) = resident_ids
            .iter()
            .find(|id| !self.external_inputs.contains(id) && !self.quantized_plans.contains_key(id))
        {
            return Err(A::invalid_binding(format!(
                "resident logical buffer {id} is not an external prefix input"
            )));
        }
        if quantized.len() != self.quantized_plans.len()
            || quantized.iter().any(|(id, value)| {
                self.quantized_plans
                    .get(id)
                    .is_none_or(|plan| value.descriptor() != &plan.desc)
            })
        {
            return Err(A::invalid_binding(
                "capture-owned packed constants do not match the static plan".into(),
            ));
        }
        let uploads = self.validated_uploads(values, |id| resident_ids.contains(&id))?;
        let packed_uploads = quantized
            .iter()
            .filter(|(id, value)| {
                self.quantized_plans
                    .get(*id)
                    .is_some_and(|plan| plan.requires_native_handle)
                    && !value.bytes().is_empty()
            })
            .collect::<Vec<_>>();
        let report = if self.queue.is_some() {
            StaticExecutionReport {
                h2d_calls: uploads
                    .len()
                    .checked_add(packed_uploads.len())
                    .ok_or_else(A::overflow)?,
                h2d_bytes: uploads
                    .iter()
                    .map(|(_, bytes)| bytes.len())
                    .chain(packed_uploads.iter().map(|(_, value)| value.bytes().len()))
                    .try_fold(0usize, |total, bytes| {
                        total.checked_add(bytes).ok_or_else(A::overflow)
                    })?,
                ..StaticExecutionReport::default()
            }
        } else {
            StaticExecutionReport::default()
        };
        if let Some(queue) = &self.queue {
            for (id, bytes) in &uploads {
                self.adapter.write(
                    queue,
                    self.buffer(*id).ok_or_else(|| {
                        A::invalid_binding(format!("logical resident buffer {id} is absent"))
                    })?,
                    bytes,
                )?;
            }
            for (id, value) in packed_uploads {
                self.adapter.write(
                    queue,
                    self.buffer(*id).ok_or_else(|| {
                        A::invalid_binding(format!("packed resident buffer {id} is absent"))
                    })?,
                    value.bytes(),
                )?;
            }
        }
        Ok((
            InitializedStaticSchedule {
                prepared: self,
                resident_ids: resident_ids.clone(),
            },
            report,
        ))
    }

    pub(crate) fn execute(&self, values: &mut BTreeMap<u64, TensorData>) -> Result<(), A::Error> {
        self.execute_skipping_residents(values, &BTreeSet::new())?;
        Ok(())
    }

    /// Executes one transaction while preserving an already initialized set
    /// of immutable external buffers.
    fn execute_skipping_residents(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
        resident_ids: &BTreeSet<u64>,
    ) -> Result<StaticExecutionReport, A::Error> {
        self.execute_skipping_residents_at_epoch(
            values,
            resident_ids,
            false,
            StaticHostOutputSelection::All,
        )
    }

    fn execute_skipping_residents_at_epoch(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
        resident_ids: &BTreeSet<u64>,
        alternate_state_bank: bool,
        host_outputs: StaticHostOutputSelection,
    ) -> Result<StaticExecutionReport, A::Error> {
        // Complete all host validation before the first driver call.
        self.validate_host_gathers(values)?;
        let uploads = self.validated_uploads(values, |id| !resident_ids.contains(&id))?;
        let mut downloads = match host_outputs {
            StaticHostOutputSelection::All => self.host_outputs.as_slice(),
            StaticHostOutputSelection::None => &[],
        }
        .iter()
        .map(|id| (*id, vec![0; self.buffer_plans[id].bytes]))
        .collect::<Vec<_>>();
        let report = if self.queue.is_some() {
            StaticExecutionReport {
                h2d_calls: uploads.len(),
                h2d_bytes: uploads.iter().try_fold(0usize, |total, (_, bytes)| {
                    total.checked_add(bytes.len()).ok_or_else(A::overflow)
                })?,
                d2h_calls: downloads
                    .iter()
                    .filter(|(_, bytes)| !bytes.is_empty())
                    .count(),
                d2h_bytes: downloads.iter().try_fold(0usize, |total, (_, bytes)| {
                    total.checked_add(bytes.len()).ok_or_else(A::overflow)
                })?,
                kernel_launches: self.items.iter().filter(|item| item.extent != 0).count(),
            }
        } else {
            StaticExecutionReport::default()
        };

        if let Some(queue) = &self.queue {
            for (id, bytes) in &uploads {
                self.adapter.write(
                    queue,
                    self.buffer_for_epoch(*id, alternate_state_bank)
                        .ok_or_else(|| {
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
                        self.buffer_for_epoch(*id, alternate_state_bank)
                            .ok_or_else(|| {
                                A::invalid_binding(format!("logical buffer {id} is absent"))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.adapter.launch_and_wait(queue, kernel, &bindings)?;
            }
            for (id, bytes) in &mut downloads {
                if !bytes.is_empty() {
                    let buffer = self
                        .buffer_for_epoch(*id, alternate_state_bank)
                        .ok_or_else(|| {
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
        Ok(report)
    }

    fn execute_append_state(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
        resident_ids: &BTreeSet<u64>,
        committed_position: usize,
        host_outputs: StaticHostOutputSelection,
    ) -> Result<StaticExecutionReport, A::Error> {
        if self.append_state_links.is_empty() || !self.state_links.is_empty() {
            return Err(A::invalid_binding(
                "static append-state execution policy is absent".into(),
            ));
        }
        for link in &self.append_state_links {
            checked_append_span_end(committed_position, link.span.rows, link.axis_extent).map_err(
                |error| match error {
                    AppendSpanEndError::Overflow => A::overflow(),
                    AppendSpanEndError::InvalidBinding(reason) => A::invalid_binding(reason),
                },
            )?;
            let position = values.get(&link.position).ok_or_else(|| {
                A::invalid_binding(format!("append position {} is absent", link.position))
            })?;
            if position.dtype() != DType::I32
                || position.shape() != &self.buffer_plans[&link.position].source_shape
            {
                return Err(A::invalid_binding(
                    "append position descriptor mismatch".into(),
                ));
            }
            let expected = i32::try_from(committed_position).map_err(|_| A::overflow())?;
            let bytes = position
                .to_le_bytes()
                .map_err(|_| A::invalid_binding("append position bytes".into()))?;
            if bytes.as_slice() != expected.to_le_bytes().as_slice() {
                return Err(A::invalid_binding(
                    "append position is not the next monotonic position".into(),
                ));
            }
        }
        self.execute_skipping_residents_at_epoch(values, resident_ids, false, host_outputs)
    }
}

impl<A: StaticDeviceAdapter> InitializedStaticSchedule<A> {
    pub(crate) fn kernels(&self) -> impl Iterator<Item = &A::Kernel> {
        self.prepared.kernels()
    }

    pub(crate) fn execute(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
    ) -> Result<StaticExecutionReport, A::Error> {
        self.prepared
            .execute_skipping_residents(values, &self.resident_ids)
    }

    pub(crate) fn execute_stateful(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
        alternate_state_bank: bool,
    ) -> Result<StaticExecutionReport, A::Error> {
        self.prepared.execute_skipping_residents_at_epoch(
            values,
            &self.resident_ids,
            alternate_state_bank,
            StaticHostOutputSelection::All,
        )
    }

    pub(crate) fn execute_append_state(
        &self,
        values: &mut BTreeMap<u64, TensorData>,
        committed_position: usize,
        host_outputs: StaticHostOutputSelection,
    ) -> Result<StaticExecutionReport, A::Error> {
        self.prepared.execute_append_state(
            values,
            &self.resident_ids,
            committed_position,
            host_outputs,
        )
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

    #[test]
    fn append_span_end_authenticates_every_i32_index_before_execution() {
        let i32_extent = i32::MAX as usize + 1;
        assert!(matches!(
            checked_append_span_end(i32_extent - 3, 3, i32_extent),
            Ok(end) if end == i32_extent
        ));
        assert!(matches!(
            checked_append_span_end(i32_extent - 2, 3, usize::MAX),
            Err(AppendSpanEndError::InvalidBinding(reason))
                if reason.contains("I32 index admission")
        ));
        assert!(matches!(
            checked_append_span_end(0, 0, i32_extent),
            Err(AppendSpanEndError::InvalidBinding(_))
        ));
        assert!(matches!(
            checked_append_span_end(usize::MAX, 1, usize::MAX),
            Err(AppendSpanEndError::Overflow)
        ));
    }

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
                pointer_ids: buffers.iter().map(|buffer| buffer.id).collect(),
                buffers,
                quantized_buffers: Vec::new(),
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
    fn bitcast_launch_domain_uses_bytes_while_storage_descriptors_stay_logical() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("bytes", [2, 4], DType::U8);
        let output = graph.bitcast(input, DType::U32).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let plan = StaticSchedulePlan::build(
            &FakeAdapter(calls.clone()),
            &schedule.items,
            Some(&[output.index() as u64]),
        )
        .unwrap();
        let item = plan.items().next().unwrap();
        assert_eq!(item.extent(), 8);
        assert_eq!(plan.buffers[&(input.index() as u64)].elements, 8);
        assert_eq!(plan.buffers[&(output.index() as u64)].elements, 2);

        let prepared = PreparedStaticSchedule::prepare_for_outputs(
            FakeAdapter(calls.clone()),
            &schedule.items,
            &[output.index() as u64],
        )
        .unwrap();
        let mut values = BTreeMap::from([(
            input.index() as u64,
            TensorData::from_storage([2, 4], Storage::U8(vec![1, 2, 3, 4, 0, 0x80, 0xff, 1]))
                .unwrap(),
        )]);
        prepared.execute(&mut values).unwrap();
        assert_eq!(
            values[&(output.index() as u64)].storage(),
            &Storage::U32(vec![0x0403_0201, 0x01ff_8000])
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
            plan.host_outputs,
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

    #[test]
    fn captured_static_projection_authenticates_and_preserves_source_requests() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let transposed = graph.permute(input, [1, 0]).unwrap();
        let constant =
            graph.constant(TensorData::from_storage([1], Storage::F32(vec![-0.0])).unwrap());
        let schedule = schedule_many(&graph, &[constant, transposed]).unwrap();
        let capture =
            crate::CapturedSchedule::capture(&graph, &schedule, &[constant, transposed]).unwrap();
        let projection = CapturedStaticExecution::new(&capture).unwrap();
        assert!(projection.retained().is_empty());
        let source = TensorData::from_storage(
            [2, 2],
            Storage::F32(vec![1.0, 2.0, 3.0, f32::from_bits(0x7fc0_1234)]),
        )
        .unwrap();
        let values = projection
            .stage(&BTreeMap::from([("input".into(), source)]))
            .unwrap();
        let outputs = projection.project(&values).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].to_le_bytes().unwrap(), [0, 0, 0, 128]);
        let expected = TensorData::from_storage(
            [2, 2],
            Storage::F32(vec![1.0, 3.0, 2.0, f32::from_bits(0x7fc0_1234)]),
        )
        .unwrap();
        assert_eq!(
            outputs[1].to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap()
        );

        let mut tampered = capture.clone();
        tampered.identity ^= 1;
        assert!(CapturedStaticExecution::new(&tampered).is_err());
        assert!(
            projection
                .stage(&BTreeMap::new())
                .unwrap_err()
                .contains("missing")
        );
        assert!(
            projection
                .stage(&BTreeMap::from([
                    (
                        "input".into(),
                        TensorData::from_storage([2, 2], Storage::F32(vec![0.0; 4])).unwrap(),
                    ),
                    ("extra".into(), TensorData::scalar(0.0)),
                ]))
                .unwrap_err()
                .contains("unexpected")
        );
    }

    #[test]
    fn captured_static_projection_preserves_external_materialization_ownership() {
        let mut graph = Graph::new();
        let left = graph.input_dtype("left", [1, 2], DType::F32);
        let right = graph.input_dtype("right", [1, 2], DType::F32);
        let addend = graph.input_dtype("addend", [1, 4], DType::F32);
        let joined = graph.concat([left, right], 1).unwrap();
        let output = graph.add(joined, addend).unwrap();
        let schedule =
            crate::schedule_with_external_materializations(&graph, &[output], &[joined]).unwrap();
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let projection = CapturedStaticExecution::new(&capture).unwrap();
        let external_name = format!("@materialized/{}", joined.index());
        let values = projection
            .stage(&BTreeMap::from([
                (
                    "addend".into(),
                    TensorData::new([1, 4], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
                ),
                (
                    external_name,
                    TensorData::new([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ),
            ]))
            .unwrap();
        assert!(values.contains_key(&(joined.index() as u64)));
        assert!(!values.contains_key(&(left.index() as u64)));
        assert!(!values.contains_key(&(right.index() as u64)));
    }

    #[test]
    fn captured_static_projection_deduplicates_only_physical_retention() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], DType::F32);
        let output = graph.square(input).unwrap();
        let schedule = schedule_many(&graph, &[output, output]).unwrap();
        let capture =
            crate::CapturedSchedule::capture(&graph, &schedule, &[output, output]).unwrap();
        let capture = crate::CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        assert_eq!(capture.requested, vec![output.index() as u64; 2]);
        let projection = CapturedStaticExecution::new(&capture).unwrap();
        assert_eq!(projection.retained(), &[output.index() as u64]);
        StaticSchedulePlan::build(
            &FakeAdapter(Rc::new(RefCell::new(Calls::default()))),
            &capture.items,
            Some(projection.retained()),
        )
        .unwrap();
        let value = TensorData::new([2], vec![1.0, 4.0]).unwrap();
        let projected = projection
            .project(&BTreeMap::from([(output.index() as u64, value)]))
            .unwrap();
        assert_eq!(projected.len(), 2);
        assert_eq!(
            projected[0].to_le_bytes().unwrap(),
            projected[1].to_le_bytes().unwrap()
        );
    }

    #[test]
    fn static_output_policy_protects_state_without_host_materialization() {
        let mut graph = Graph::new();
        let state = graph.input_dtype("state", [2], DType::F32);
        let transient = graph.input_dtype("transient", [2], DType::F32);
        let next = graph.add(state, transient).unwrap();
        let output = graph.square(next).unwrap();
        let schedule = schedule_many(&graph, &[output, next]).unwrap();
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[output, next]).unwrap();
        let adapter = FakeAdapter(Rc::new(RefCell::new(Calls::default())));
        let link = StaticStateLink {
            input: state.index() as u64,
            output: next.index() as u64,
        };
        let plan = StaticSchedulePlan::build_with_output_policy(
            &adapter,
            &capture.items,
            &[output.index() as u64],
            &[output.index() as u64, next.index() as u64],
            &[link],
            &[],
        )
        .unwrap();
        assert_eq!(plan.host_outputs(), &[output.index() as u64]);
        assert_eq!(
            plan.protected_outputs(),
            &[output.index() as u64, next.index() as u64]
        );
        assert_ne!(
            plan.allocations().logical_slots()[&(state.index() as u64)],
            plan.allocations().logical_slots()[&(next.index() as u64)]
        );
        assert!(
            StaticSchedulePlan::build_with_output_policy(
                &adapter,
                &capture.items,
                &[next.index() as u64],
                &[output.index() as u64, next.index() as u64],
                &[link],
                &[],
            )
            .is_err()
        );
    }

    #[test]
    fn static_lifetime_plan_partitions_resident_transient_and_constant_inputs() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let weight = graph.input_dtype("weight", [2], DType::F32);
        let constant = graph.constant(TensorData::new([2], vec![1.0, 1.0]).unwrap());
        let sum = graph.add(input, weight).unwrap();
        let output = graph.mul(sum, constant).unwrap();
        let schedule = schedule_many(&graph, &[output, output]).unwrap();
        let capture =
            crate::CapturedSchedule::capture(&graph, &schedule, &[output, output]).unwrap();
        let projection = CapturedStaticExecution::from_owned(capture).unwrap();
        let lifetime = StaticLifetimePlan::new(projection, &["weight".into()]).unwrap();
        assert_eq!(lifetime.resident_names().collect::<Vec<_>>(), ["weight"]);
        assert_eq!(lifetime.transient_names().collect::<Vec<_>>(), ["input"]);
        assert!(lifetime.resident_ids().contains(&(weight.index() as u64)));
        assert!(lifetime.resident_ids().contains(&(constant.index() as u64)));

        assert!(
            lifetime
                .stage_resident(BTreeMap::new())
                .unwrap_err()
                .contains("missing")
        );
        assert!(
            StaticLifetimePlan::new(lifetime.capture.clone(), &["input".into(), "input".into()])
                .err()
                .unwrap()
                .contains("unique")
        );
        assert!(
            StaticLifetimePlan::new(lifetime.capture.clone(), &["unknown".into()])
                .err()
                .unwrap()
                .contains("absent")
        );

        let resident = lifetime
            .stage_resident(BTreeMap::from([(
                "weight".into(),
                TensorData::new([2], vec![2.0, 3.0]).unwrap(),
            )]))
            .unwrap();
        assert!(resident.contains_key(&(weight.index() as u64)));
        assert!(resident.contains_key(&(constant.index() as u64)));
        let values = lifetime
            .stage_transient(&BTreeMap::from([(
                "input".into(),
                TensorData::new([2], vec![5.0, 7.0]).unwrap(),
            )]))
            .unwrap();
        let projected = lifetime
            .project(
                &BTreeMap::from([(
                    output.index() as u64,
                    TensorData::new([2], vec![7.0, 10.0]).unwrap(),
                )]),
                &resident,
            )
            .unwrap();
        assert_eq!(projected.len(), 2);
        assert_eq!(
            projected[0].to_le_bytes().unwrap(),
            projected[1].to_le_bytes().unwrap()
        );
        assert_eq!(values.len(), 1);
    }

    #[test]
    fn static_lifetime_separates_authenticated_runtime_control() {
        let mut graph = Graph::new();
        let position = graph.input_dtype("position", [1], DType::I32);
        let reshaped = graph.reshape(position, [1, 1]).unwrap();
        let expanded = graph.expand(reshaped, [1, 2]).unwrap();
        let output = graph.cast(expanded, DType::F32).unwrap();
        let schedule = schedule_many(&graph, &[output]).unwrap();
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let projection = CapturedStaticExecution::from_owned(capture).unwrap();
        let control = projection.inputs()[0].clone();
        assert_eq!(control.node, position);
        assert!(control.desc.view.is_some());
        let lifetime = StaticLifetimePlan::new_with_state_and_controls(
            projection,
            &[],
            &[],
            std::slice::from_ref(&control),
        )
        .unwrap();
        assert!(lifetime.transient_inputs().is_empty());
        assert_eq!(lifetime.runtime_controls(), std::slice::from_ref(&control));
        let mut values = lifetime.stage_transient(&BTreeMap::new()).unwrap();
        assert_eq!(
            lifetime.stage_committed_position(7, &mut values).unwrap(),
            (1, 4)
        );
        assert_eq!(
            values[&control.desc.id].to_le_bytes().unwrap(),
            7i32.to_le_bytes()
        );
    }

    #[test]
    fn captured_static_projection_retains_computed_alias_source_only() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [4], DType::F32);
        let producer = graph.square(input).unwrap();
        let alias = graph
            .stride(
                producer,
                [crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            )
            .unwrap();
        let schedule = schedule_many(&graph, &[alias, alias]).unwrap();
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[alias, alias]).unwrap();
        let projection = CapturedStaticExecution::new(&capture).unwrap();
        assert_eq!(projection.retained(), &[producer.index() as u64]);
        let plan = StaticSchedulePlan::build(
            &FakeAdapter(Rc::new(RefCell::new(Calls::default()))),
            &capture.items,
            Some(projection.retained()),
        )
        .unwrap();
        assert_eq!(plan.retained_outputs(), &[producer.index() as u64]);
        assert!(!plan.buffers.contains_key(&(alias.index() as u64)));

        let mut values = projection
            .stage(&BTreeMap::from([(
                "x".into(),
                TensorData::new([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )]))
            .unwrap();
        assert!(!values.contains_key(&(alias.index() as u64)));
        values.insert(
            producer.index() as u64,
            TensorData::new([4], vec![1.0, 4.0, 9.0, 16.0]).unwrap(),
        );
        let outputs = projection.project(&values).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].to_vec_f64(), vec![16.0, 9.0, 4.0, 1.0]);
        assert_eq!(outputs[1].storage(), outputs[0].storage());
    }
}
