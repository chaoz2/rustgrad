//! Persistent, capture-authenticated Metal inference sessions.

use super::{
    MetalDevice, MetalDeviceInfo, MetalError, MetalPrefixPlan, MetalRenderer, PreparedMetalPrefix,
    RenderedMetal, prepared::InitializedMetalPrefix,
};
use crate::{CapturedInference, CapturedSchedule, ExecutionPlanSummary, ReplayInput, TensorData};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use crate::runtime::static_schedule::{
    CapturedStaticExecution, StaticExecutionReport, StaticLifetimePlan,
};

/// Deterministic inspection metadata for one concrete Metal session plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceSessionSummary {
    /// Frozen identity of the authenticated captured schedule.
    pub capture_identity: u64,
    /// Named inputs uploaded once during preparation.
    pub resident_input_names: Vec<String>,
    /// Named inputs required on every invocation.
    pub transient_input_names: Vec<String>,
    /// Logical outputs, including ordered duplicate requests and aliases.
    pub requested_output_count: usize,
    /// Capture-owned constants, whether rendered inline or as buffers.
    pub constant_count: usize,
    /// Exact raw tensor payload bytes of capture-owned constants.
    pub constant_bytes: usize,
    /// Declared host payload bytes for resident named inputs.
    pub resident_input_bytes: usize,
    /// Declared host payload bytes for transient named inputs.
    pub transient_input_bytes: usize,
    /// Physical allocation slots in the static memory plan.
    pub planned_slot_count: usize,
    /// Physical bytes planned for tensor slots, including four-byte native
    /// sentinels where a nonzero launch needs an address for logical zero bytes.
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
}

/// Successful Metal preparation measurements. Durations are current-thread
/// wall times, not GPU timestamps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDevicePreparationReport {
    /// Host wall-clock duration of pure capture planning and rendering.
    pub planning_wall_time: Duration,
    /// Host wall-clock duration of compilation, allocation, and queue setup.
    pub native_prepare_wall_time: Duration,
    /// Host wall-clock duration of immutable host API writes.
    pub resident_upload_wall_time: Duration,
    /// Number of nonzero rendered kernels requested from the device cache.
    pub pipeline_cache_request_count: usize,
    /// Requests already present in this thread-confined device cache.
    pub pipeline_cache_hit_count: usize,
    /// New cache entries created by this preparation.
    pub pipeline_cache_miss_count: usize,
    /// Host API writes, not claimed PCIe transfers.
    pub resident_h2d_calls: usize,
    /// Host API write bytes, not claimed PCIe traffic.
    pub resident_h2d_bytes: usize,
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
}

/// Detached ordered outputs plus the report committed for that successful run.
pub struct MetalDeviceRun {
    outputs: Vec<TensorData>,
    report: MetalDeviceRunReport,
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

impl MetalInferencePlan {
    /// Renders an owned inference capture without creating a Metal resource.
    pub fn new(inference: CapturedInference, renderer: MetalRenderer) -> Result<Self, MetalError> {
        let (capture, execution_plan, resident_bindings, deployment_identity) =
            inference.into_parts();
        let resident_names = resident_bindings.keys().cloned().collect::<Vec<_>>();
        let inner = MetalDeviceSessionPlan::from_capture(capture, resident_names, renderer)?;
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
        self.inner.prepare(device, self.resident_bindings)
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
}

impl MetalDeviceSessionPlan {
    /// Authenticates and renders a concrete pure capture with an explicit
    /// resident subset of its named inputs. This method creates no resources.
    pub fn from_capture(
        capture: CapturedSchedule,
        resident_input_names: impl IntoIterator<Item = String>,
        renderer: MetalRenderer,
    ) -> Result<Self, MetalError> {
        let planning_start = Instant::now();
        let resident_input_names = resident_input_names.into_iter().collect::<Vec<_>>();
        let projection =
            CapturedStaticExecution::from_owned(capture).map_err(|error| match error {
                crate::runtime::static_schedule::CapturedStaticAdmissionError::Invalid(reason) => {
                    MetalError::InvalidBinding(reason)
                }
                crate::runtime::static_schedule::CapturedStaticAdmissionError::Unsupported(
                    reason,
                ) => MetalError::Unsupported(reason),
            })?;
        let lifetime = StaticLifetimePlan::new(projection, &resident_input_names)
            .map_err(MetalError::InvalidBinding)?;
        let prefix = MetalPrefixPlan::plan_for_outputs(
            &lifetime.capture().items,
            lifetime.retained(),
            renderer.clone(),
        )?;
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
        let resident = lifetime
            .resident_names()
            .collect::<std::collections::BTreeSet<_>>();
        let resident_input_bytes = lifetime
            .capture()
            .inputs
            .iter()
            .filter(|input| resident.contains(input.name.as_str()))
            .try_fold(0usize, |total, input| {
                total
                    .checked_add(input.desc.bytes)
                    .ok_or(MetalError::Overflow)
            })?;
        let transient_input_bytes = lifetime
            .capture()
            .inputs
            .iter()
            .filter(|input| !resident.contains(input.name.as_str()))
            .try_fold(0usize, |total, input| {
                total
                    .checked_add(input.desc.bytes)
                    .ok_or(MetalError::Overflow)
            })?;
        let summary = MetalDeviceSessionSummary {
            capture_identity: lifetime.capture().identity,
            resident_input_names: lifetime.resident_names().map(str::to_owned).collect(),
            transient_input_names: lifetime.transient_names().map(str::to_owned).collect(),
            requested_output_count: lifetime.capture().requested.len(),
            constant_count: lifetime.capture().constants.len(),
            constant_bytes,
            resident_input_bytes,
            transient_input_bytes,
            planned_slot_count,
            planned_device_bytes,
            zero_byte_sentinel_count,
            nonzero_item_count,
            zero_item_count,
            rendered_cache_keys: prefix.cache_keys(),
            fallback_count: 0,
        };
        let renderer_capabilities = renderer.capabilities.clone();
        Ok(Self {
            lifetime,
            prefix,
            device_resident_ids,
            summary,
            planning_wall_time: planning_start.elapsed(),
            renderer_capabilities,
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

    /// Returns the typed per-invocation named-input schemas.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.lifetime.transient_inputs()
    }

    /// Returns deterministic planned resource and execution metadata.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        &self.summary
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
        // Value and capability validation precede cache, compilation,
        // allocation, queue creation, or upload.
        let resident_values = self
            .lifetime
            .stage_resident(resident_inputs)
            .map_err(MetalError::InvalidBinding)?;
        if device.info().capabilities != self.renderer_capabilities {
            return Err(MetalError::InvalidBinding(
                "Metal session renderer/device capability identity mismatch".into(),
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
        let prepared = PreparedMetalPrefix::from_plan(device.clone(), self.prefix)?;
        let native_prepare_wall_time = native_prepare_start.elapsed();
        let resident_upload_start = Instant::now();
        let (prepared, resident_transfer) =
            prepared.initialize_resident(&resident_values, &self.device_resident_ids)?;
        let resident_upload_wall_time = resident_upload_start.elapsed();
        let preparation = MetalDevicePreparationReport {
            planning_wall_time: self.planning_wall_time,
            native_prepare_wall_time,
            resident_upload_wall_time,
            pipeline_cache_request_count,
            pipeline_cache_hit_count,
            pipeline_cache_miss_count,
            resident_h2d_calls: resident_transfer.h2d_calls,
            resident_h2d_bytes: resident_transfer.h2d_bytes,
        };
        let resident_sources = self.lifetime.retain_projection_sources(resident_values);
        Ok(MetalDeviceSession {
            lifetime: self.lifetime,
            resident_sources,
            prepared,
            summary: self.summary,
            preparation,
            device_info,
            device_owner_id,
            successful_runs: 0,
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

    /// Returns the typed per-invocation named-input schemas.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.lifetime.transient_inputs()
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

    /// Validates exact transient inputs, executes each nonzero schedule item
    /// with an individual Metal launch/wait, and returns only requested outputs.
    pub fn run(
        &mut self,
        transient_inputs: &BTreeMap<String, TensorData>,
    ) -> Result<MetalDeviceRun, MetalError> {
        let successful_invocation = self
            .successful_runs
            .checked_add(1)
            .ok_or(MetalError::Overflow)?;
        let run_start = Instant::now();
        let mut values = self
            .lifetime
            .stage_transient(transient_inputs)
            .map_err(MetalError::InvalidBinding)?;
        let execute_start = Instant::now();
        let transfer = self.prepared.execute(&mut values)?;
        let synchronous_transaction_wall_time = execute_start.elapsed();
        let outputs = self
            .lifetime
            .project(&values, &self.resident_sources)
            .map_err(MetalError::InvalidBinding)?;
        let report = run_report(
            successful_invocation,
            run_start.elapsed(),
            synchronous_transaction_wall_time,
            transfer,
            self.summary.zero_item_count,
            outputs.len(),
        );
        self.successful_runs = successful_invocation;
        Ok(MetalDeviceRun { outputs, report })
    }
}

fn run_report(
    successful_invocation: u64,
    run_wall_time: Duration,
    synchronous_transaction_wall_time: Duration,
    transfer: StaticExecutionReport,
    zero_item_count: usize,
    output_count: usize,
) -> MetalDeviceRunReport {
    MetalDeviceRunReport {
        successful_invocation,
        first_successful_run: successful_invocation == 1,
        run_wall_time,
        synchronous_transaction_wall_time,
        transient_h2d_calls: transfer.h2d_calls,
        transient_h2d_bytes: transfer.h2d_bytes,
        retained_d2h_calls: transfer.d2h_calls,
        retained_d2h_bytes: transfer.d2h_bytes,
        kernel_launch_count: transfer.kernel_launches,
        zero_item_count,
        output_count,
    }
}
