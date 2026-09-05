//! Persistent, capture-authenticated Metal inference sessions.

use super::{
    MetalDevice, MetalDeviceInfo, MetalError, MetalPrefixPlan, MetalRenderer, PreparedMetalPrefix,
    RenderedMetal, prepared::InitializedMetalPrefix,
};
use crate::{
    CapturedAppendStateInference, CapturedInference, CapturedSchedule, CapturedStatefulInference,
    ExecutionPlanSummary, ReplayInput, TensorData,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    time::{Duration, Instant},
};

use crate::runtime::static_schedule::{
    CapturedStaticExecution, StaticAppendStateLink, StaticExecutionReport, StaticHostGather,
    StaticHostOutputSelection, StaticLifetimePlan, StaticStateLink,
};

/// Resource-free planning controls shared by typed Metal inference facades.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalPlanOptions {
    /// Preferred launch threadgroup width. Exact pipeline limits are checked
    /// during preparation on the selected device.
    pub local_size: usize,
}

impl MetalPlanOptions {
    pub const fn new(local_size: usize) -> Self {
        Self { local_size }
    }
}

impl Default for MetalPlanOptions {
    fn default() -> Self {
        Self { local_size: 64 }
    }
}

/// Deterministic inspection metadata for one concrete Metal session plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceSessionSummary {
    /// Frozen identity of the authenticated captured schedule.
    pub capture_identity: u64,
    /// Named inputs uploaded once during preparation.
    pub resident_input_names: Vec<String>,
    /// Named inputs required on every invocation.
    pub transient_input_names: Vec<String>,
    /// Capture-authenticated inputs synthesized by the session per invocation.
    pub runtime_control_input_names: Vec<String>,
    /// Logical outputs, including ordered duplicate requests and aliases.
    pub requested_output_count: usize,
    /// Capture-owned constants, whether rendered inline or as buffers.
    pub constant_count: usize,
    /// Exact raw tensor payload bytes of capture-owned constants.
    pub constant_bytes: usize,
    /// Capture-owned packed GGUF constants admitted without dequantizing.
    pub quantized_constant_count: usize,
    /// Exact raw packed GGUF payload bytes declared by the capture. Zero-work
    /// owners can remain addressless and therefore need no device allocation.
    pub quantized_constant_bytes: usize,
    /// Declared host payload bytes for resident named inputs.
    pub resident_input_bytes: usize,
    /// Declared host payload bytes for transient named inputs.
    pub transient_input_bytes: usize,
    /// Declared payload bytes synthesized as session runtime controls.
    pub runtime_control_input_bytes: usize,
    /// Physical allocation slots in the static memory plan.
    pub planned_slot_count: usize,
    /// Physical bytes planned for dense tensor slots and capture-owned packed
    /// buffers, including four-byte native sentinels where a nonzero launch
    /// needs an address for logical zero bytes.
    pub planned_device_bytes: usize,
    /// Private native handles planned for logically empty bindings.
    pub zero_byte_sentinel_count: usize,
    /// Schedule items that issue one Metal launch and wait.
    pub nonzero_item_count: usize,
    /// Addressless schedule items skipped without resource work.
    pub zero_item_count: usize,
    /// Ordered renderer/cache identities for nonzero prepared kernels.
    pub rendered_cache_keys: Vec<String>,
    /// This strict path has no CPU fallback branch.
    pub fallback_count: usize,
    /// Number of authenticated recurrent input/output pairs.
    pub state_pair_count: usize,
    /// Bytes in one logical recurrent-state bank.
    pub logical_state_bytes: usize,
    /// Logical bank sets: zero when stateless, one for append-only state, or
    /// two for epoch-swapped state. Empty state remains addressless.
    pub state_bank_count: usize,
    /// Logical payload bytes represented by the selected state-bank policy.
    /// Physical slot bytes remain in `planned_device_bytes`.
    pub state_device_bytes: usize,
    /// Sparse F32 payload bytes written by one successful append invocation.
    /// This is one row for the historical T=1 contract and the complete span
    /// for fixed-span execution.
    pub append_state_row_bytes: usize,
    /// Sparse state elements written by one successful append invocation.
    pub append_state_work_items: usize,
}

/// Successful Metal preparation measurements. Durations are current-thread
/// wall times, not GPU timestamps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDevicePreparationReport {
    /// Host wall-clock duration of pure capture planning and rendering.
    pub planning_wall_time: Duration,
    /// Host wall-clock duration of compilation, allocation, and queue setup.
    pub native_prepare_wall_time: Duration,
    /// Host wall-clock duration spent building cache-miss native libraries and
    /// compute pipelines. Cache hits contribute zero to this field.
    pub cache_miss_pipeline_build_wall_time: Duration,
    /// Host wall-clock duration of immutable resident plus initial-state host
    /// API writes.
    pub initialization_upload_wall_time: Duration,
    /// Number of nonzero rendered kernels requested from the device cache.
    pub pipeline_cache_request_count: usize,
    /// Requests already present in this thread-confined device cache.
    pub pipeline_cache_hit_count: usize,
    /// New cache entries created by this preparation.
    pub pipeline_cache_miss_count: usize,
    /// Host API writes for immutable named inputs, dense constants, and packed
    /// constants; not claimed PCIe transfers.
    pub resident_h2d_calls: usize,
    /// Host API write bytes, not claimed PCIe traffic.
    pub resident_h2d_bytes: usize,
    /// One-time host API writes for initial recurrent state.
    pub initial_state_h2d_calls: usize,
    /// One-time initial recurrent-state bytes written.
    pub initial_state_h2d_bytes: usize,
}

/// Successful per-invocation measurements. Failed invocations return no run
/// report and do not advance the successful invocation index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceRunReport {
    /// One-based count of successfully published invocations.
    pub successful_invocation: u64,
    /// True only for the first successfully published invocation.
    pub first_successful_run: bool,
    /// Host wall-clock duration of validation, device execution, and projection.
    pub run_wall_time: Duration,
    /// Host wall-clock duration of host API copies and per-item launch/wait calls.
    pub synchronous_transaction_wall_time: Duration,
    /// Host API writes for transient inputs only.
    pub transient_h2d_calls: usize,
    /// Bytes passed to transient host API writes.
    pub transient_h2d_bytes: usize,
    /// Host API writes synthesized from authenticated session runtime control.
    pub runtime_control_h2d_calls: usize,
    /// Bytes passed to runtime-control host API writes.
    pub runtime_control_h2d_bytes: usize,
    /// Host API reads for requested materialized outputs only.
    pub retained_d2h_calls: usize,
    /// Bytes passed to retained-output host API reads.
    pub retained_d2h_bytes: usize,
    /// Nonzero schedule items launched, each followed by its own wait.
    pub kernel_launch_count: usize,
    /// Addressless schedule items skipped during this invocation.
    pub zero_item_count: usize,
    /// Logical outputs after ordered duplicate/alias projection.
    pub output_count: usize,
    /// Recurrent pairs atomically committed by this successful invocation.
    pub committed_state_pair_count: usize,
    /// Logical recurrent bytes committed by the epoch flip or row append.
    pub committed_state_bytes: usize,
    /// Sparse recurrent elements committed by this invocation. Double-bank
    /// state reports its full logical element count; stateless runs report 0.
    pub committed_state_work_items: usize,
    /// Next append row after this successful invocation. This is `None` for
    /// stateless and epoch-swapped sessions.
    pub committed_state_position: Option<usize>,
}

/// Detached ordered outputs plus the report committed for that successful run.
pub struct MetalDeviceRun {
    outputs: Vec<TensorData>,
    report: MetalDeviceRunReport,
    session_token: Rc<()>,
}

#[derive(Clone, Copy)]
enum MetalOutputProof {
    BoundedI32 {
        output: usize,
        upper_exclusive: usize,
    },
}

impl MetalDeviceRun {
    /// Returns detached outputs in the capture's requested order.
    pub fn outputs(&self) -> &[TensorData] {
        &self.outputs
    }

    /// Consumes the run and returns its detached ordered outputs.
    pub fn into_outputs(self) -> Vec<TensorData> {
        self.outputs
    }

    /// Consumes the run into its ordered outputs and committed report.
    pub fn into_parts(self) -> (Vec<TensorData>, MetalDeviceRunReport) {
        (self.outputs, self.report)
    }

    /// Returns measurements committed with this successful invocation.
    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }

    pub(super) fn belongs_to(&self, session_token: &Rc<()>) -> bool {
        Rc::ptr_eq(&self.session_token, session_token)
    }
}

/// Resource-free deployment of one owned inference capture to the strict
/// persistent Metal session path.
///
/// Immutable resident values are frozen by the backend-neutral capture. This
/// wrapper adds no execution policy: rendering, allocation, preparation, and
/// repeated execution remain owned by [`MetalDeviceSessionPlan`] and
/// [`MetalDeviceSession`].
pub struct MetalInferencePlan {
    inner: MetalDeviceSessionPlan,
    execution_plan: ExecutionPlanSummary,
    resident_bindings: BTreeMap<String, TensorData>,
    deployment_identity: u64,
}

/// Resource-free deployment of one authenticated fixed-state inference body.
pub struct MetalStatefulInferencePlan {
    inner: MetalDeviceSessionPlan,
    execution_plan: ExecutionPlanSummary,
    resident_bindings: BTreeMap<String, TensorData>,
    initial_state: BTreeMap<String, TensorData>,
    deployment_identity: u64,
}

/// Resource-free deployment of one authenticated append-only state capture.
pub struct MetalAppendStateInferencePlan {
    inner: MetalDeviceSessionPlan,
    execution_plan: ExecutionPlanSummary,
    resident_bindings: BTreeMap<String, TensorData>,
    quantized_input_names: BTreeMap<u64, String>,
    initial_state: BTreeMap<String, TensorData>,
    deployment_identity: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetalSessionStatePolicy {
    None,
    Epoch {
        pair_count: usize,
        bytes: usize,
        work_items: usize,
    },
    Append {
        pair_count: usize,
        span_rows: usize,
        axis_extent: usize,
        row_bytes: usize,
        work_items: usize,
    },
}

struct MetalCapturePolicy {
    resident_input_names: Vec<String>,
    state_input_names: Vec<String>,
    public_output_count: usize,
    state_links: Vec<StaticStateLink>,
    append_state_links: Vec<StaticAppendStateLink>,
    host_gathers: Vec<StaticHostGather>,
    runtime_controls: Vec<ReplayInput>,
}

/// Private proof that one append-state plan may import immutable/state storage
/// from one already initialized append-state session. Ids are program-local;
/// correspondence is authenticated by semantic input names and exact payloads.
pub(crate) struct MetalSharedAppendProof {
    source_capture_identity: u64,
    source_deployment_identity: u64,
    target_capture_identity: u64,
    target_deployment_identity: u64,
    dense: BTreeMap<u64, u64>,
    quantized: BTreeMap<u64, u64>,
}

/// Compares the physical contract for storage shared by two captured programs.
/// `view` is deliberately excluded: it describes each program's use-site
/// indexing and may differ even when both inputs name the same allocation.
fn same_shared_storage_descriptor(left: &crate::BufferDesc, right: &crate::BufferDesc) -> bool {
    left.shape == right.shape
        && left.dtype == right.dtype
        && left.bytes == right.bytes
        && left.alignment == right.alignment
        && left.read_only == right.read_only
}

fn same_tensor_payload(left: &TensorData, right: &TensorData) -> Result<bool, MetalError> {
    if left.shape() != right.shape() || left.dtype() != right.dtype() {
        return Ok(false);
    }
    let left = left.to_le_bytes().map_err(|_| MetalError::Overflow)?;
    let right = right.to_le_bytes().map_err(|_| MetalError::Overflow)?;
    Ok(left == right)
}

fn exact_input_by_name<'a>(
    inputs: &'a [ReplayInput],
    name: &str,
) -> Result<&'a ReplayInput, MetalError> {
    let mut matching = inputs.iter().filter(|input| input.name == name);
    let input = matching.next().ok_or_else(|| {
        MetalError::InvalidBinding(format!("shared Metal input {name} is absent"))
    })?;
    if matching.next().is_some() {
        return Err(MetalError::InvalidBinding(format!(
            "shared Metal input {name} is ambiguous"
        )));
    }
    Ok(input)
}

fn exact_quantized_id_by_name(
    inputs: &BTreeMap<u64, String>,
    name: &str,
) -> Result<u64, MetalError> {
    let mut matching = inputs
        .iter()
        .filter_map(|(id, candidate)| (candidate == name).then_some(*id));
    let id = matching.next().ok_or_else(|| {
        MetalError::InvalidBinding(format!("shared packed Metal input {name} is absent"))
    })?;
    if matching.next().is_some() {
        return Err(MetalError::InvalidBinding(format!(
            "shared packed Metal input {name} is ambiguous"
        )));
    }
    Ok(id)
}

fn static_host_gathers(links: &[crate::session::CapturedHostGather]) -> Vec<StaticHostGather> {
    links
        .iter()
        .map(|link| StaticHostGather {
            input: link.input.desc.id,
            input_desc: link.input.desc.clone(),
            index: link.index,
            output: link.output,
            axis: link.axis,
            axis_extent: link.axis_extent,
            index_elements: link.index_elements,
        })
        .collect()
}

impl MetalAppendStateInferencePlan {
    pub fn new(
        inference: CapturedAppendStateInference,
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let (
            inference,
            public_output_count,
            states,
            initial_state,
            sealed_position,
            deployment_identity,
        ) = inference.into_parts();
        let quantized_input_names = inference.quantized_input_names().clone();
        let (capture, execution_plan, resident_bindings, host_gathers, _) = inference.into_parts();
        let resident_names = resident_bindings.keys().cloned().collect::<Vec<_>>();
        let state_names = states
            .iter()
            .map(|state| state.input.name.clone())
            .collect::<Vec<_>>();
        let append_links = states
            .iter()
            .map(|state| StaticAppendStateLink {
                input: state.input.desc.id,
                output: state.output.id,
                position: state.position.desc.id,
                index: state.index.id,
                iota: state.iota,
                updates: state.updates.id,
                axis: state.link.axis(),
                axis_extent: state.axis_extent,
                span: state.span,
            })
            .collect::<Vec<_>>();
        let inner = MetalDeviceSessionPlan::from_capture_policy(
            capture,
            MetalCapturePolicy {
                resident_input_names: resident_names,
                state_input_names: state_names,
                public_output_count,
                state_links: Vec::new(),
                append_state_links: append_links,
                host_gathers: static_host_gathers(&host_gathers),
                runtime_controls: sealed_position.into_iter().collect(),
            },
            renderer,
        )?;
        Ok(Self {
            inner,
            execution_plan,
            resident_bindings,
            quantized_input_names,
            initial_state,
            deployment_identity,
        })
    }

    pub const fn deployment_identity(&self) -> u64 {
        self.deployment_identity
    }

    pub fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.execution_plan
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    /// Returns the authenticated number of state-axis rows committed by one run.
    pub fn append_span_rows(&self) -> usize {
        self.inner.append_span_rows()
    }

    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.inner.state_inputs()
    }

    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    #[cfg(test)]
    pub(crate) const fn quantized_input_names(&self) -> &BTreeMap<u64, String> {
        &self.quantized_input_names
    }

    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    /// Returns capture-authenticated inputs synthesized by the session.
    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.inner.runtime_control_inputs()
    }

    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    pub fn prepare(self, device: MetalDevice) -> Result<MetalDeviceSession, MetalError> {
        self.inner.prepare_with_state_and_deployment(
            device,
            self.resident_bindings,
            self.initial_state,
            Some(self.deployment_identity),
        )
    }

    pub(crate) fn prepare_shared(
        self,
        device: MetalDevice,
        source: &MetalDeviceSession,
        proof: MetalSharedAppendProof,
    ) -> Result<MetalSharedAppendSession, MetalError> {
        let inner = self.inner.prepare_with_shared_state_and_deployment(
            device,
            self.resident_bindings,
            self.initial_state,
            self.deployment_identity,
            source,
            proof,
        )?;
        Ok(MetalSharedAppendSession { inner })
    }

    /// Authenticates the exact immutable/state inventory this plan may import
    /// from an initialized sibling deployment. Program-local ids are never
    /// compared directly across captures.
    pub(crate) fn authenticate_shared_from(
        &self,
        source: &Self,
    ) -> Result<MetalSharedAppendProof, MetalError> {
        let (
            MetalSessionStatePolicy::Append {
                pair_count: target_pairs,
                axis_extent: target_extent,
                ..
            },
            MetalSessionStatePolicy::Append {
                pair_count: source_pairs,
                axis_extent: source_extent,
                ..
            },
        ) = (self.inner.state_policy, source.inner.state_policy)
        else {
            return Err(MetalError::InvalidBinding(
                "shared Metal plans must both use append state".into(),
            ));
        };
        if target_pairs != source_pairs || target_extent != source_extent {
            return Err(MetalError::InvalidBinding(
                "shared Metal append-state geometry differs".into(),
            ));
        }

        let mut dense = BTreeMap::new();
        for target in self
            .inner
            .resident_inputs()
            .iter()
            .chain(self.inner.state_inputs())
            .filter(|input| self.inner.device_resident_ids.contains(&input.desc.id))
        {
            let source_input =
                exact_input_by_name(source.capture().inputs.as_slice(), &target.name)?;
            if !source
                .inner
                .device_resident_ids
                .contains(&source_input.desc.id)
                || !same_shared_storage_descriptor(&target.desc, &source_input.desc)
            {
                return Err(MetalError::InvalidBinding(format!(
                    "shared Metal input {} descriptor or role differs",
                    target.name
                )));
            }
            let target_value = self
                .resident_bindings
                .get(&target.name)
                .or_else(|| self.initial_state.get(&target.name));
            let source_value = source
                .resident_bindings
                .get(&target.name)
                .or_else(|| source.initial_state.get(&target.name));
            let (Some(target_value), Some(source_value)) = (target_value, source_value) else {
                return Err(MetalError::InvalidBinding(format!(
                    "shared Metal input {} immutable payload differs",
                    target.name
                )));
            };
            if !same_tensor_payload(target_value, source_value)? {
                return Err(MetalError::InvalidBinding(format!(
                    "shared Metal input {} immutable payload differs",
                    target.name
                )));
            }
            dense.insert(target.desc.id, source_input.desc.id);
        }
        if self.inner.state_inputs().len() != source.inner.state_inputs().len()
            || self.inner.state_inputs().iter().any(|target| {
                source
                    .inner
                    .state_inputs()
                    .iter()
                    .filter(|candidate| candidate.name == target.name)
                    .count()
                    != 1
            })
        {
            return Err(MetalError::InvalidBinding(
                "shared Metal state inventory differs".into(),
            ));
        }

        if self.quantized_input_names.len() != self.inner.lifetime.quantized_constants().len()
            || source.quantized_input_names.len()
                != source.inner.lifetime.quantized_constants().len()
        {
            return Err(MetalError::InvalidBinding(
                "shared packed Metal input inventory differs".into(),
            ));
        }
        let mut quantized = BTreeMap::new();
        for (target_id, target_value) in self.inner.lifetime.quantized_constants() {
            let target_name = self.quantized_input_names.get(target_id).ok_or_else(|| {
                MetalError::InvalidBinding(format!(
                    "packed Metal input {target_id} has no semantic name"
                ))
            })?;
            let source_id = exact_quantized_id_by_name(&source.quantized_input_names, target_name)?;
            let source_value = source
                .inner
                .lifetime
                .quantized_constants()
                .get(&source_id)
                .ok_or_else(|| {
                    MetalError::InvalidBinding(format!(
                        "shared packed Metal input {} is not packed in the source",
                        target_name
                    ))
                })?;
            if target_value != source_value {
                return Err(MetalError::InvalidBinding(format!(
                    "shared packed Metal input {} descriptor or payload differs",
                    target_name
                )));
            }
            quantized.insert(*target_id, source_id);
        }

        Ok(MetalSharedAppendProof {
            source_capture_identity: source.capture().identity,
            source_deployment_identity: source.deployment_identity,
            target_capture_identity: self.capture().identity,
            target_deployment_identity: self.deployment_identity,
            dense,
            quantized,
        })
    }
}

impl MetalStatefulInferencePlan {
    pub fn new(
        inference: CapturedStatefulInference,
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let (inference, public_output_count, states, initial_state, deployment_identity) =
            inference.into_parts();
        let (capture, execution_plan, resident_bindings, host_gathers, _) = inference.into_parts();
        let resident_names = resident_bindings.keys().cloned().collect::<Vec<_>>();
        let state_names = states
            .iter()
            .map(|state| state.input.name.clone())
            .collect::<Vec<_>>();
        let state_links = states
            .iter()
            .map(|state| StaticStateLink {
                input: state.input.desc.id,
                output: state.output.id,
            })
            .collect::<Vec<_>>();
        let inner = MetalDeviceSessionPlan::from_capture_policy(
            capture,
            MetalCapturePolicy {
                resident_input_names: resident_names,
                state_input_names: state_names,
                public_output_count,
                state_links,
                append_state_links: Vec::new(),
                host_gathers: static_host_gathers(&host_gathers),
                runtime_controls: Vec::new(),
            },
            renderer,
        )?;
        Ok(Self {
            inner,
            execution_plan,
            resident_bindings,
            initial_state,
            deployment_identity,
        })
    }

    pub const fn deployment_identity(&self) -> u64 {
        self.deployment_identity
    }

    pub fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.execution_plan
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.inner.state_inputs()
    }

    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    /// Returns every rendered item, including addressless zero-work items.
    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    pub fn prepare(self, device: MetalDevice) -> Result<MetalDeviceSession, MetalError> {
        self.inner.prepare_with_state_and_deployment(
            device,
            self.resident_bindings,
            self.initial_state,
            Some(self.deployment_identity),
        )
    }
}

impl MetalInferencePlan {
    /// Renders an owned inference capture without creating a Metal resource.
    pub fn new(inference: CapturedInference, renderer: MetalRenderer) -> Result<Self, MetalError> {
        let (capture, execution_plan, resident_bindings, host_gathers, deployment_identity) =
            inference.into_parts();
        let resident_names = resident_bindings.keys().cloned().collect::<Vec<_>>();
        let inner = MetalDeviceSessionPlan::from_capture_policy(
            capture,
            MetalCapturePolicy {
                resident_input_names: resident_names,
                state_input_names: Vec::new(),
                public_output_count: execution_plan.requested_outputs.len(),
                state_links: Vec::new(),
                append_state_links: Vec::new(),
                host_gathers: static_host_gathers(&host_gathers),
                runtime_controls: Vec::new(),
            },
            renderer,
        )?;
        Ok(Self {
            inner,
            execution_plan,
            resident_bindings,
            deployment_identity,
        })
    }

    /// Returns the identity of the capture plus exact immutable resident
    /// names, descriptors, and payload bytes.
    pub const fn deployment_identity(&self) -> u64 {
        self.deployment_identity
    }

    /// Returns the exact authenticated capture owned by the device plan.
    pub fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    /// Returns backend-neutral logical schedule and memory facts.
    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.execution_plan
    }

    /// Returns typed immutable resident schemas.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    /// Returns typed per-invocation transient schemas.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    /// Returns deterministic Metal resource and execution planning metadata.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    /// Returns every rendered item, including addressless zero-work items.
    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    /// Creates persistent resources and uploads the frozen residents once.
    pub fn prepare(self, device: MetalDevice) -> Result<MetalDeviceSession, MetalError> {
        self.inner.prepare_with_state_and_deployment(
            device,
            self.resident_bindings,
            BTreeMap::new(),
            Some(self.deployment_identity),
        )
    }
}

/// Pure, fully rendered plan for a concrete capture and explicit immutable
/// resident-input policy. Constructing this type creates no Metal resources.
pub struct MetalDeviceSessionPlan {
    lifetime: StaticLifetimePlan,
    prefix: MetalPrefixPlan,
    device_resident_ids: std::collections::BTreeSet<u64>,
    summary: MetalDeviceSessionSummary,
    planning_wall_time: Duration,
    renderer_capabilities: super::MetalCapabilities,
    state_policy: MetalSessionStatePolicy,
}

impl MetalDeviceSessionPlan {
    /// Authenticates and renders a concrete pure capture with an explicit
    /// resident subset of its named inputs. This method creates no resources.
    pub fn from_capture(
        capture: CapturedSchedule,
        resident_input_names: impl IntoIterator<Item = String>,
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let requested_count = capture.requested.len();
        Self::from_capture_policy(
            capture,
            MetalCapturePolicy {
                resident_input_names: resident_input_names.into_iter().collect(),
                state_input_names: Vec::new(),
                public_output_count: requested_count,
                state_links: Vec::new(),
                append_state_links: Vec::new(),
                host_gathers: Vec::new(),
                runtime_controls: Vec::new(),
            },
            renderer,
        )
    }

    fn from_capture_policy(
        capture: CapturedSchedule,
        policy: MetalCapturePolicy,
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let MetalCapturePolicy {
            resident_input_names,
            state_input_names,
            public_output_count,
            state_links,
            append_state_links,
            host_gathers,
            runtime_controls,
        } = policy;
        let planning_start = Instant::now();
        let projection =
            CapturedStaticExecution::from_owned(capture).map_err(|error| match error {
                crate::runtime::static_schedule::CapturedStaticAdmissionError::Invalid(reason) => {
                    MetalError::InvalidBinding(reason)
                }
                crate::runtime::static_schedule::CapturedStaticAdmissionError::Unsupported(
                    reason,
                ) => MetalError::Unsupported(reason),
            })?;
        let host_outputs = projection
            .retained_for_requested_prefix(public_output_count)
            .map_err(MetalError::InvalidBinding)?;
        let mut protected_outputs = host_outputs.clone();
        protected_outputs.extend(state_links.iter().map(|state| state.output));
        protected_outputs.extend(append_state_links.iter().map(|state| state.output));
        let lifetime = if state_input_names.is_empty() && runtime_controls.is_empty() {
            StaticLifetimePlan::new(projection, &resident_input_names)
        } else {
            StaticLifetimePlan::new_with_state_and_controls(
                projection,
                &resident_input_names,
                &state_input_names,
                &runtime_controls,
            )
        }
        .map_err(MetalError::InvalidBinding)?;
        let transient_ids = lifetime
            .transient_inputs()
            .iter()
            .map(|input| input.desc.id)
            .collect::<std::collections::BTreeSet<_>>();
        let runtime_control_ids = lifetime
            .runtime_controls()
            .iter()
            .map(|input| input.desc.id)
            .collect::<std::collections::BTreeSet<_>>();
        if host_gathers.iter().any(|link| {
            !transient_ids.contains(&link.input) && !runtime_control_ids.contains(&link.input)
        }) {
            return Err(MetalError::InvalidBinding(
                "Metal host Gather input is not an exact transient or runtime control".into(),
            ));
        }
        if !state_links.is_empty() && !append_state_links.is_empty() {
            return Err(MetalError::InvalidBinding(
                "Metal session cannot mix epoch and append state".into(),
            ));
        }
        let prefix = if append_state_links.is_empty() {
            MetalPrefixPlan::plan_with_output_policy(
                &lifetime.capture().items,
                &host_outputs,
                &protected_outputs,
                &state_links,
                &host_gathers,
                renderer.clone(),
            )?
        } else {
            MetalPrefixPlan::plan_with_append_policy(
                &lifetime.capture().items,
                &host_outputs,
                &protected_outputs,
                &append_state_links,
                &host_gathers,
                renderer.clone(),
            )?
        };
        let external_inputs = prefix
            .external_input_ids()
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let device_resident_ids = lifetime
            .resident_ids()
            .intersection(&external_inputs)
            .copied()
            .collect();
        let (nonzero_item_count, zero_item_count) = prefix.item_counts();
        let (planned_slot_count, planned_device_bytes, zero_byte_sentinel_count) =
            prefix.allocation_summary()?;
        let constant_bytes =
            lifetime
                .capture()
                .constants
                .values()
                .try_fold(0usize, |total, value| {
                    total
                        .checked_add(value.to_le_bytes().map_err(|_| MetalError::Overflow)?.len())
                        .ok_or(MetalError::Overflow)
                })?;
        let quantized_constant_bytes =
            lifetime
                .quantized_constants()
                .values()
                .try_fold(0usize, |total, value| {
                    total
                        .checked_add(value.bytes().len())
                        .ok_or(MetalError::Overflow)
                })?;
        let resident_input_bytes =
            lifetime
                .resident_inputs()
                .iter()
                .try_fold(0usize, |total, input| {
                    total
                        .checked_add(input.desc.bytes)
                        .ok_or(MetalError::Overflow)
                })?;
        let transient_input_bytes =
            lifetime
                .transient_inputs()
                .iter()
                .try_fold(0usize, |total, input| {
                    total
                        .checked_add(input.desc.bytes)
                        .ok_or(MetalError::Overflow)
                })?;
        let runtime_control_input_bytes =
            lifetime
                .runtime_controls()
                .iter()
                .try_fold(0usize, |total, input| {
                    total
                        .checked_add(input.desc.bytes)
                        .ok_or(MetalError::Overflow)
                })?;
        let logical_state_bytes =
            lifetime
                .state_inputs()
                .iter()
                .try_fold(0usize, |total, input| {
                    total
                        .checked_add(input.desc.bytes)
                        .ok_or(MetalError::Overflow)
                })?;
        let logical_state_elements =
            lifetime
                .state_inputs()
                .iter()
                .try_fold(0usize, |total, input| {
                    total
                        .checked_add(input.desc.shape.numel().map_err(|_| MetalError::Overflow)?)
                        .ok_or(MetalError::Overflow)
                })?;
        let append_state_row_bytes =
            append_state_links.iter().try_fold(0usize, |total, link| {
                total
                    .checked_add(link.span.total_bytes)
                    .ok_or(MetalError::Overflow)
            })?;
        let append_state_work_items =
            append_state_links.iter().try_fold(0usize, |total, link| {
                total
                    .checked_add(link.span.total_elements)
                    .ok_or(MetalError::Overflow)
            })?;
        let state_bank_count = if !append_state_links.is_empty() {
            1
        } else if !state_links.is_empty() {
            2
        } else {
            0
        };
        let state_device_bytes = logical_state_bytes
            .checked_mul(state_bank_count)
            .ok_or(MetalError::Overflow)?;
        let state_policy = if !append_state_links.is_empty() {
            let first = append_state_links[0];
            MetalSessionStatePolicy::Append {
                pair_count: append_state_links.len(),
                span_rows: first.span.rows,
                axis_extent: first.axis_extent,
                row_bytes: append_state_row_bytes,
                work_items: append_state_work_items,
            }
        } else if !state_links.is_empty() {
            MetalSessionStatePolicy::Epoch {
                pair_count: state_links.len(),
                bytes: logical_state_bytes,
                work_items: logical_state_elements,
            }
        } else {
            MetalSessionStatePolicy::None
        };
        let summary = MetalDeviceSessionSummary {
            capture_identity: lifetime.capture().identity,
            resident_input_names: lifetime.resident_names().map(str::to_owned).collect(),
            transient_input_names: lifetime.transient_names().map(str::to_owned).collect(),
            runtime_control_input_names: lifetime
                .runtime_controls()
                .iter()
                .map(|input| input.name.clone())
                .collect(),
            requested_output_count: public_output_count,
            constant_count: lifetime.capture().constants.len(),
            constant_bytes,
            quantized_constant_count: lifetime.quantized_constants().len(),
            quantized_constant_bytes,
            resident_input_bytes,
            transient_input_bytes,
            runtime_control_input_bytes,
            planned_slot_count,
            planned_device_bytes,
            zero_byte_sentinel_count,
            nonzero_item_count,
            zero_item_count,
            rendered_cache_keys: prefix.cache_keys(),
            fallback_count: 0,
            state_pair_count: state_links.len() + append_state_links.len(),
            logical_state_bytes,
            state_bank_count,
            state_device_bytes,
            append_state_row_bytes,
            append_state_work_items,
        };
        let renderer_capabilities = renderer.capabilities.clone();
        Ok(Self {
            lifetime,
            prefix,
            device_resident_ids,
            summary,
            planning_wall_time: planning_start.elapsed(),
            renderer_capabilities,
            state_policy,
        })
    }

    /// Returns the exact authenticated capture owned by this plan.
    pub fn capture(&self) -> &CapturedSchedule {
        self.lifetime.capture()
    }

    /// Returns the typed resident named-input schemas.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.lifetime.resident_inputs()
    }

    /// Returns the fixed state inputs initialized during preparation.
    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.lifetime.state_inputs()
    }

    /// Returns the typed per-invocation named-input schemas.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.lifetime.transient_inputs()
    }

    /// Returns capture-authenticated inputs synthesized by the session.
    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.lifetime.runtime_controls()
    }

    /// Returns deterministic planned resource and execution metadata.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        &self.summary
    }

    fn append_span_rows(&self) -> usize {
        match self.state_policy {
            MetalSessionStatePolicy::Append { span_rows, .. } => span_rows,
            _ => 0,
        }
    }

    /// Returns every inspectable rendered schedule item, including zero-work
    /// items that preparation does not compile.
    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.prefix.rendered_kernels()
    }

    /// Creates persistent Metal resources and uploads immutable named inputs
    /// and buffer-backed constants once.
    pub fn prepare(
        self,
        device: MetalDevice,
        resident_inputs: BTreeMap<String, TensorData>,
    ) -> Result<MetalDeviceSession, MetalError> {
        self.prepare_with_state_and_deployment(device, resident_inputs, BTreeMap::new(), None)
    }

    fn prepare_with_state_and_deployment(
        self,
        device: MetalDevice,
        resident_inputs: BTreeMap<String, TensorData>,
        initial_state: BTreeMap<String, TensorData>,
        inference_deployment_identity: Option<u64>,
    ) -> Result<MetalDeviceSession, MetalError> {
        self.prepare_impl(
            device,
            resident_inputs,
            initial_state,
            inference_deployment_identity,
            None,
        )
    }

    fn prepare_with_shared_state_and_deployment(
        self,
        device: MetalDevice,
        resident_inputs: BTreeMap<String, TensorData>,
        initial_state: BTreeMap<String, TensorData>,
        inference_deployment_identity: u64,
        source: &MetalDeviceSession,
        proof: MetalSharedAppendProof,
    ) -> Result<MetalDeviceSession, MetalError> {
        self.prepare_impl(
            device,
            resident_inputs,
            initial_state,
            Some(inference_deployment_identity),
            Some((source, proof)),
        )
    }

    fn prepare_impl(
        self,
        device: MetalDevice,
        resident_inputs: BTreeMap<String, TensorData>,
        initial_state: BTreeMap<String, TensorData>,
        inference_deployment_identity: Option<u64>,
        shared: Option<(&MetalDeviceSession, MetalSharedAppendProof)>,
    ) -> Result<MetalDeviceSession, MetalError> {
        // Value and capability validation precede cache, compilation,
        // allocation, queue creation, or upload.
        let resident_values = if self.lifetime.state_inputs().is_empty() {
            self.lifetime.stage_resident(resident_inputs)
        } else {
            self.lifetime
                .stage_initialized(resident_inputs, initial_state)
        }
        .map_err(MetalError::InvalidBinding)?;
        let mut transient_ids = self
            .lifetime
            .transient_inputs()
            .iter()
            .map(|input| input.desc.id)
            .collect::<std::collections::BTreeSet<_>>();
        transient_ids.extend(
            self.lifetime
                .runtime_controls()
                .iter()
                .map(|input| input.desc.id),
        );
        self.lifetime
            .validate_quantized_gathers(&resident_values, &transient_ids)
            .map_err(metal_quantized_gather_error)?;
        if device.info().capabilities != self.renderer_capabilities {
            return Err(MetalError::InvalidBinding(
                "Metal session renderer/device capability identity mismatch".into(),
            ));
        }
        if let Some((source, proof)) = &shared
            && (proof.target_capture_identity != self.capture().identity
                || Some(proof.target_deployment_identity) != inference_deployment_identity
                || proof.source_capture_identity != source.capture().identity
                || source.inference_deployment_identity() != Some(proof.source_deployment_identity)
                || source.device_owner_id() != device.owner_id()
                || !matches!(source.state_policy, MetalSessionStatePolicy::Append { .. })
                || source.successful_runs != 0
                || source.committed_state_position != 0)
        {
            return Err(MetalError::InvalidBinding(
                "shared Metal session proof does not belong to these deployments".into(),
            ));
        }
        let device_info = device.info().clone();
        let device_owner_id = device.owner_id();
        let cache = device.cache();
        let mut candidate_keys = std::collections::BTreeSet::new();
        let mut pipeline_cache_hit_count = 0usize;
        let mut pipeline_cache_miss_count = 0usize;
        for rendered in self
            .prefix
            .rendered_kernels()
            .filter(|rendered| rendered.extent != 0)
        {
            if cache.contains_rendered(rendered)
                || !candidate_keys.insert(rendered.cache_key.clone())
            {
                pipeline_cache_hit_count = pipeline_cache_hit_count
                    .checked_add(1)
                    .ok_or(MetalError::Overflow)?;
            } else {
                pipeline_cache_miss_count = pipeline_cache_miss_count
                    .checked_add(1)
                    .ok_or(MetalError::Overflow)?;
            }
        }
        let pipeline_cache_request_count = pipeline_cache_hit_count
            .checked_add(pipeline_cache_miss_count)
            .ok_or(MetalError::Overflow)?;
        if pipeline_cache_request_count != self.summary.nonzero_item_count {
            return Err(MetalError::InvalidBinding(
                "Metal session cache inventory differs from its nonzero schedule".into(),
            ));
        }
        let native_prepare_start = Instant::now();
        let (prepared, imported_dense) = match shared {
            Some((source, proof)) => {
                let resources = source
                    .prepared
                    .share_resources(&proof.dense, &proof.quantized)?;
                let imported_dense = proof.dense.keys().copied().collect::<BTreeSet<_>>();
                let imported_quantized = proof.quantized.keys().copied().collect::<BTreeSet<_>>();
                (
                    PreparedMetalPrefix::from_plan_with_shared(
                        device.clone(),
                        self.prefix,
                        resources,
                        &imported_dense,
                        &imported_quantized,
                    )?,
                    imported_dense,
                )
            }
            None => (
                PreparedMetalPrefix::from_plan(device.clone(), self.prefix)?,
                BTreeSet::new(),
            ),
        };
        let native_prepare_wall_time = native_prepare_start.elapsed();
        let cache_miss_pipeline_build_wall_time = prepared.cache_miss_pipeline_build_wall_time();
        let initialization_upload_start = Instant::now();
        let (prepared, resident_transfer) = prepared.initialize_resident(
            &resident_values,
            &self.device_resident_ids,
            self.lifetime.quantized_constants(),
        )?;
        let initialization_upload_wall_time = initialization_upload_start.elapsed();
        let initial_state_h2d_calls = self
            .lifetime
            .state_inputs()
            .iter()
            .filter(|input| input.desc.bytes != 0 && !imported_dense.contains(&input.desc.id))
            .count();
        let initial_state_h2d_bytes = self
            .lifetime
            .state_inputs()
            .iter()
            .filter(|input| !imported_dense.contains(&input.desc.id))
            .try_fold(0usize, |total, input| {
                total
                    .checked_add(input.desc.bytes)
                    .ok_or(MetalError::Overflow)
            })?;
        let preparation = MetalDevicePreparationReport {
            planning_wall_time: self.planning_wall_time,
            native_prepare_wall_time,
            cache_miss_pipeline_build_wall_time,
            initialization_upload_wall_time,
            pipeline_cache_request_count,
            pipeline_cache_hit_count,
            pipeline_cache_miss_count,
            resident_h2d_calls: resident_transfer
                .h2d_calls
                .checked_sub(initial_state_h2d_calls)
                .ok_or(MetalError::Overflow)?,
            resident_h2d_bytes: resident_transfer
                .h2d_bytes
                .checked_sub(initial_state_h2d_bytes)
                .ok_or(MetalError::Overflow)?,
            initial_state_h2d_calls,
            initial_state_h2d_bytes,
        };
        let resident_sources = self.lifetime.retain_projection_sources(resident_values);
        let public_output_count = self.summary.requested_output_count;
        Ok(MetalDeviceSession {
            lifetime: self.lifetime,
            resident_sources,
            prepared,
            summary: self.summary,
            preparation,
            device_info,
            device_owner_id,
            successful_runs: 0,
            state_epoch: false,
            committed_state_position: 0,
            state_policy: self.state_policy,
            public_output_count,
            session_token: Rc::new(()),
            inference_deployment_identity,
        })
    }
}

/// Thread-confined persistent Metal execution state for one concrete capture.
pub struct MetalDeviceSession {
    lifetime: StaticLifetimePlan,
    resident_sources: BTreeMap<u64, TensorData>,
    prepared: InitializedMetalPrefix,
    summary: MetalDeviceSessionSummary,
    preparation: MetalDevicePreparationReport,
    device_info: MetalDeviceInfo,
    device_owner_id: u64,
    successful_runs: u64,
    state_epoch: bool,
    committed_state_position: usize,
    state_policy: MetalSessionStatePolicy,
    public_output_count: usize,
    session_token: Rc<()>,
    inference_deployment_identity: Option<u64>,
}

/// Private half of a checked two-program append deployment. It deliberately
/// exposes no independent implicit-position run API.
pub(crate) struct MetalSharedAppendSession {
    inner: MetalDeviceSession,
}

impl MetalSharedAppendSession {
    pub(crate) fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    pub(crate) fn preparation_report(&self) -> &MetalDevicePreparationReport {
        self.inner.preparation_report()
    }

    pub(crate) fn compiled_kernels(&self) -> impl Iterator<Item = &RenderedMetal> {
        self.inner.compiled_kernels()
    }

    pub(crate) fn run_without_host_outputs_at(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
        committed_position: usize,
    ) -> Result<MetalDeviceRun, MetalError> {
        self.inner
            .run_without_host_outputs_at(transient_inputs, committed_position)
    }
}

impl MetalDeviceSession {
    /// Returns the exact authenticated capture owned by this session.
    pub fn capture(&self) -> &CapturedSchedule {
        self.lifetime.capture()
    }

    /// Returns the typed resident named-input schemas.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.lifetime.resident_inputs()
    }

    /// Returns the fixed state inputs initialized during preparation.
    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.lifetime.state_inputs()
    }

    /// Returns the typed per-invocation named-input schemas.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.lifetime.transient_inputs()
    }

    /// Returns capture-authenticated inputs synthesized by the session.
    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.lifetime.runtime_controls()
    }

    /// Returns immutable information for the selected Metal device.
    pub fn device_info(&self) -> &MetalDeviceInfo {
        &self.device_info
    }

    /// Returns the stable Rust resource-owner identity of the selected device.
    pub fn device_owner_id(&self) -> u64 {
        self.device_owner_id
    }

    /// Returns deterministic planned resource and execution metadata.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        &self.summary
    }

    /// Returns measurements from the successful preparation transaction.
    pub fn preparation_report(&self) -> &MetalDevicePreparationReport {
        &self.preparation
    }

    /// Returns the inspectable nonzero kernels compiled by this session.
    pub fn compiled_kernels(&self) -> impl Iterator<Item = &RenderedMetal> {
        self.prepared.rendered_kernels()
    }

    /// Returns the number of successfully published invocations.
    pub fn successful_run_count(&self) -> u64 {
        self.successful_runs
    }

    pub(super) fn session_token(&self) -> &Rc<()> {
        &self.session_token
    }

    pub(super) const fn inference_deployment_identity(&self) -> Option<u64> {
        self.inference_deployment_identity
    }

    /// False/true selects which authenticated state bank is currently active.
    pub const fn state_epoch(&self) -> bool {
        self.state_epoch
    }

    /// Returns the next append row for append-state sessions. Other session
    /// policies return `None`.
    pub const fn committed_state_position(&self) -> Option<usize> {
        match self.state_policy {
            MetalSessionStatePolicy::Append { .. } => Some(self.committed_state_position),
            _ => None,
        }
    }

    /// Validates exact transient inputs, executes each nonzero schedule item
    /// with an individual Metal launch/wait, and returns only requested outputs.
    pub fn run(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
    ) -> Result<MetalDeviceRun, MetalError> {
        self.run_with_host_outputs(transient_inputs, StaticHostOutputSelection::All, None)
    }

    #[cfg(test)]
    pub(crate) fn run_without_host_outputs(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
    ) -> Result<MetalDeviceRun, MetalError> {
        self.run_with_host_outputs(transient_inputs, StaticHostOutputSelection::None, None)
    }

    fn run_with_host_outputs(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
        host_outputs: StaticHostOutputSelection,
        output_proof: Option<MetalOutputProof>,
    ) -> Result<MetalDeviceRun, MetalError> {
        self.run_with_host_outputs_at(
            transient_inputs,
            host_outputs,
            self.committed_state_position,
            output_proof,
        )
    }

    pub(crate) fn run_at(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
        committed_position: usize,
    ) -> Result<MetalDeviceRun, MetalError> {
        self.run_append_at(
            transient_inputs,
            StaticHostOutputSelection::All,
            committed_position,
            None,
        )
    }

    pub(crate) fn run_at_requiring_bounded_i32(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
        committed_position: usize,
        output: usize,
        upper_exclusive: usize,
    ) -> Result<MetalDeviceRun, MetalError> {
        if upper_exclusive == 0 {
            return Err(MetalError::InvalidDeviceProof(
                "bounded I32 proof requires a nonempty range",
            ));
        }
        if output >= self.public_output_count {
            return Err(MetalError::InvalidDeviceProof(
                "proof output is outside the public output schema",
            ));
        }
        self.run_append_at(
            transient_inputs,
            StaticHostOutputSelection::All,
            committed_position,
            Some(MetalOutputProof::BoundedI32 {
                output,
                upper_exclusive,
            }),
        )
    }

    pub(crate) fn run_without_host_outputs_at(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
        committed_position: usize,
    ) -> Result<MetalDeviceRun, MetalError> {
        self.run_append_at(
            transient_inputs,
            StaticHostOutputSelection::None,
            committed_position,
            None,
        )
    }

    fn run_append_at(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
        host_outputs: StaticHostOutputSelection,
        committed_position: usize,
        output_proof: Option<MetalOutputProof>,
    ) -> Result<MetalDeviceRun, MetalError> {
        if !matches!(self.state_policy, MetalSessionStatePolicy::Append { .. }) {
            return Err(MetalError::InvalidBinding(
                "explicit Metal position requires append-state inference".into(),
            ));
        }
        self.run_with_host_outputs_at(
            transient_inputs,
            host_outputs,
            committed_position,
            output_proof,
        )
    }

    fn run_with_host_outputs_at(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
        host_outputs: StaticHostOutputSelection,
        committed_position: usize,
        output_proof: Option<MetalOutputProof>,
    ) -> Result<MetalDeviceRun, MetalError> {
        if host_outputs == StaticHostOutputSelection::None
            && (!matches!(self.state_policy, MetalSessionStatePolicy::Append { .. })
                || self.lifetime.runtime_controls().len() != 1)
        {
            return Err(MetalError::InvalidBinding(
                "host-output suppression requires sealed append-state inference".into(),
            ));
        }
        let successful_invocation = self
            .successful_runs
            .checked_add(1)
            .ok_or(MetalError::Overflow)?;
        let next_committed_position = match self.state_policy {
            MetalSessionStatePolicy::Append {
                span_rows,
                axis_extent,
                ..
            } => {
                let end = crate::runtime::static_schedule::checked_append_span_end(
                    committed_position,
                    span_rows,
                    axis_extent,
                )
                .map_err(|error| match error {
                    crate::runtime::static_schedule::AppendSpanEndError::Overflow => {
                        MetalError::Overflow
                    }
                    crate::runtime::static_schedule::AppendSpanEndError::InvalidBinding(reason) => {
                        MetalError::InvalidBinding(reason)
                    }
                })?;
                Some(end)
            }
            _ => None,
        };
        let run_start = Instant::now();
        let mut values = self
            .lifetime
            .stage_transient(transient_inputs)
            .map_err(MetalError::InvalidBinding)?;
        let (runtime_control_h2d_calls, runtime_control_h2d_bytes) = match self.state_policy {
            MetalSessionStatePolicy::Append { .. } => self
                .lifetime
                .stage_committed_position(committed_position, &mut values)
                .map_err(MetalError::InvalidBinding)?,
            _ => (0, 0),
        };
        let (runtime_control_h2d_calls, runtime_control_h2d_bytes) =
            if self.summary.nonzero_item_count == 0 {
                (0, 0)
            } else {
                (runtime_control_h2d_calls, runtime_control_h2d_bytes)
            };
        self.lifetime
            .validate_quantized_gathers(&values, self.lifetime.resident_ids())
            .map_err(metal_quantized_gather_error)?;
        let execute_start = Instant::now();
        let transfer = match self.state_policy {
            MetalSessionStatePolicy::None => self.prepared.execute(&mut values)?,
            MetalSessionStatePolicy::Epoch { .. } => self
                .prepared
                .execute_stateful(&mut values, self.state_epoch)?,
            MetalSessionStatePolicy::Append { .. } => {
                self.prepared
                    .execute_append_state(&mut values, committed_position, host_outputs)?
            }
        };
        let synchronous_transaction_wall_time = execute_start.elapsed();
        let outputs = match host_outputs {
            StaticHostOutputSelection::None => Vec::new(),
            StaticHostOutputSelection::All => match self.state_policy {
                MetalSessionStatePolicy::None => {
                    self.lifetime.project(&values, &self.resident_sources)
                }
                MetalSessionStatePolicy::Epoch { .. } | MetalSessionStatePolicy::Append { .. } => {
                    self.lifetime.project_prefix(
                        self.public_output_count,
                        &values,
                        &self.resident_sources,
                    )
                }
            }
            .map_err(MetalError::InvalidBinding)?,
        };
        if let Some(MetalOutputProof::BoundedI32 {
            output,
            upper_exclusive,
        }) = output_proof
        {
            let proof = outputs
                .get(output)
                .ok_or(MetalError::InvalidDeviceProof("proof output is absent"))?;
            let value = (proof.dtype() == crate::DType::I32 && proof.len() == 1)
                .then(|| proof.scalar_at(0).as_i64())
                .and_then(|value| usize::try_from(value).ok());
            if proof.dtype() != crate::DType::I32
                || proof.len() != 1
                || value.is_none_or(|value| value >= upper_exclusive)
            {
                return Err(MetalError::InvalidDeviceProof(
                    "required I32 output is outside its authenticated range",
                ));
            }
        }
        let report = run_report(RunReportInput {
            successful_invocation,
            run_wall_time: run_start.elapsed(),
            synchronous_transaction_wall_time,
            transfer,
            runtime_control_h2d_calls,
            runtime_control_h2d_bytes,
            zero_item_count: self.summary.zero_item_count,
            output_count: outputs.len(),
            committed_state: match self.state_policy {
                MetalSessionStatePolicy::None => CommittedState::default(),
                MetalSessionStatePolicy::Epoch {
                    pair_count,
                    bytes,
                    work_items,
                } => CommittedState {
                    pair_count,
                    bytes,
                    work_items,
                },
                MetalSessionStatePolicy::Append {
                    pair_count,
                    row_bytes,
                    work_items,
                    ..
                } => CommittedState {
                    pair_count,
                    bytes: row_bytes,
                    work_items,
                },
            },
            committed_state_position: next_committed_position,
        });
        self.successful_runs = successful_invocation;
        match self.state_policy {
            MetalSessionStatePolicy::Epoch { .. } => self.state_epoch = !self.state_epoch,
            MetalSessionStatePolicy::Append { .. } => {
                self.committed_state_position =
                    next_committed_position.expect("append position was preflighted");
            }
            MetalSessionStatePolicy::None => {}
        }
        Ok(MetalDeviceRun {
            outputs,
            report,
            session_token: self.session_token.clone(),
        })
    }
}

struct RunReportInput {
    successful_invocation: u64,
    run_wall_time: Duration,
    synchronous_transaction_wall_time: Duration,
    transfer: StaticExecutionReport,
    runtime_control_h2d_calls: usize,
    runtime_control_h2d_bytes: usize,
    zero_item_count: usize,
    output_count: usize,
    committed_state: CommittedState,
    committed_state_position: Option<usize>,
}

fn run_report(input: RunReportInput) -> MetalDeviceRunReport {
    let RunReportInput {
        successful_invocation,
        run_wall_time,
        synchronous_transaction_wall_time,
        transfer,
        runtime_control_h2d_calls,
        runtime_control_h2d_bytes,
        zero_item_count,
        output_count,
        committed_state,
        committed_state_position,
    } = input;
    MetalDeviceRunReport {
        successful_invocation,
        first_successful_run: successful_invocation == 1,
        run_wall_time,
        synchronous_transaction_wall_time,
        transient_h2d_calls: transfer
            .h2d_calls
            .checked_sub(runtime_control_h2d_calls)
            .expect("runtime controls are a subset of invocation uploads"),
        transient_h2d_bytes: transfer
            .h2d_bytes
            .checked_sub(runtime_control_h2d_bytes)
            .expect("runtime controls are a subset of invocation upload bytes"),
        runtime_control_h2d_calls,
        runtime_control_h2d_bytes,
        retained_d2h_calls: transfer.d2h_calls,
        retained_d2h_bytes: transfer.d2h_bytes,
        kernel_launch_count: transfer.kernel_launches,
        zero_item_count,
        output_count,
        committed_state_pair_count: committed_state.pair_count,
        committed_state_bytes: committed_state.bytes,
        committed_state_work_items: committed_state.work_items,
        committed_state_position,
    }
}

#[derive(Clone, Copy, Default)]
struct CommittedState {
    pair_count: usize,
    bytes: usize,
    work_items: usize,
}

fn metal_quantized_gather_error(
    error: crate::runtime::static_schedule::StaticQuantizedGatherError,
) -> MetalError {
    match error {
        crate::runtime::static_schedule::StaticQuantizedGatherError::Invalid(reason) => {
            MetalError::InvalidBinding(reason)
        }
        crate::runtime::static_schedule::StaticQuantizedGatherError::IndexOutOfBounds {
            position,
            value,
            rows,
        } => MetalError::IndexOutOfBounds {
            axis: 0,
            index: position,
            value,
            dim: rows,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{same_shared_storage_descriptor, same_tensor_payload};
    use crate::{AffineView, BufferDesc, DType, Shape, Storage, TensorData};

    #[test]
    fn shared_storage_authentication_ignores_program_local_views() {
        let physical = BufferDesc {
            id: 7,
            shape: Shape::from([2, 3]),
            dtype: DType::F32,
            bytes: 24,
            alignment: 4,
            read_only: true,
            view: None,
        };
        let mut differently_viewed = physical.clone();
        differently_viewed.id = 19;
        differently_viewed.view = Some(AffineView::identity(Shape::from([2, 3])));
        assert!(same_shared_storage_descriptor(
            &physical,
            &differently_viewed
        ));

        differently_viewed.bytes = 20;
        assert!(!same_shared_storage_descriptor(
            &physical,
            &differently_viewed
        ));
    }

    #[test]
    fn shared_dense_payload_authentication_uses_exact_storage_bits() {
        let payload = TensorData::from_storage(
            [3],
            Storage::F32(vec![
                f32::from_bits(0x0000_0000),
                f32::from_bits(0x7fc0_1234),
                f32::from_bits(0xff80_0000),
            ]),
        )
        .unwrap();
        let identical = payload.clone();
        assert!(same_tensor_payload(&payload, &identical).unwrap());

        let signed_zero = TensorData::from_storage(
            [3],
            Storage::F32(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_1234),
                f32::from_bits(0xff80_0000),
            ]),
        )
        .unwrap();
        assert!(!same_tensor_payload(&payload, &signed_zero).unwrap());

        let different_nan = TensorData::from_storage(
            [3],
            Storage::F32(vec![
                f32::from_bits(0x0000_0000),
                f32::from_bits(0x7fc0_5678),
                f32::from_bits(0xff80_0000),
            ]),
        )
        .unwrap();
        assert!(!same_tensor_payload(&payload, &different_nan).unwrap());
    }
}
