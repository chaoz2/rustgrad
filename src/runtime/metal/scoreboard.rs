//! Opt-in host-observed measurements for one persistent Metal inference session.

use super::{
    MetalAppendStateInferencePlan, MetalDeviceInfo, MetalDeviceRun, MetalDeviceSession,
    MetalDeviceSessionSummary, MetalInferencePlan,
};
use crate::{CapturedSchedule, DType, ExecutionPlanSummary, ReplayInput, Shape};
use serde::{Serialize, Serializer};
use std::{collections::BTreeSet, fmt, fs, io, path::Path, rc::Rc, time::Duration};

/// Current deterministic JSON schema emitted by [`MetalSessionScoreboardReport`].
pub const METAL_SESSION_SCOREBOARD_FORMAT_VERSION: u32 = 5;
const MAX_METADATA_BYTES: usize = 1_024;

/// Caller-supplied labels attached to one measurement series.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalScoreboardContext {
    workload: String,
    implementation_revision: String,
    evidence: String,
}

impl MetalScoreboardContext {
    /// Creates bounded, single-line provenance labels for a report.
    pub fn new(
        workload: impl Into<String>,
        implementation_revision: impl Into<String>,
        evidence: impl Into<String>,
    ) -> Result<Self, MetalScoreboardError> {
        let value = Self {
            workload: workload.into(),
            implementation_revision: implementation_revision.into(),
            evidence: evidence.into(),
        };
        validate_label("workload", &value.workload)?;
        validate_label("implementation_revision", &value.implementation_revision)?;
        validate_label("evidence", &value.evidence)?;
        Ok(value)
    }

    /// Returns the caller's workload label.
    pub fn workload(&self) -> &str {
        &self.workload
    }

    /// Returns the caller's implementation revision label.
    pub fn implementation_revision(&self) -> &str {
        &self.implementation_revision
    }

    /// Returns the caller's evidence/provenance label.
    pub fn evidence(&self) -> &str {
        &self.evidence
    }
}

/// Whether one captured input is frozen during preparation or supplied per run.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalScoreboardInputKind {
    /// Immutable module input uploaded during session preparation.
    Resident,
    /// Recurrent input initialized once and updated only by the session's
    /// authenticated state policy.
    State,
    /// Caller input supplied independently to every invocation.
    Transient,
    /// Session-synthesized input authenticated by the captured policy.
    RuntimeControl,
}

/// Stateful policy authenticated by a scoreboard snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalScoreboardStatePolicy {
    /// No recurrent state is owned by the session.
    Stateless,
    /// One fixed-capacity state bank receives one complete row per success.
    Append,
}

/// Exact captured descriptor retained as report evidence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalScoreboardInput {
    /// Exact captured input name.
    pub name: String,
    /// Exact graph node identity authenticated by the capture.
    pub node_id: u64,
    /// Captured static input shape.
    pub shape: Shape,
    /// Captured storage dtype.
    #[serde(serialize_with = "serialize_dtype")]
    pub dtype: DType,
    /// Exact logical payload byte count.
    pub bytes: usize,
    /// Resident or per-invocation ownership policy.
    pub kind: MetalScoreboardInputKind,
}

/// Integer-duration summary over the ordered steady-run host observations.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalHostWallTimeSummary {
    /// Smallest recorded integer duration.
    pub min: Duration,
    /// Nearest-rank 50th percentile.
    pub nearest_rank_p50: Duration,
    /// Nearest-rank 95th percentile.
    pub nearest_rank_p95: Duration,
    /// Largest recorded integer duration.
    pub max: Duration,
}

/// Exact host-observed counters committed by one successful invocation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalScoreboardRun {
    /// One-based successful invocation ordinal.
    pub successful_invocation: u64,
    /// Whether this was the session's first successfully published invocation.
    pub first_successful_run: bool,
    /// Host wall time covering validation, execution, and output projection.
    pub run_wall_time: Duration,
    /// Host wall time covering copies plus command submission and waits.
    pub synchronous_transaction_wall_time: Duration,
    /// Per-invocation transient host API writes.
    pub transient_host_api_h2d_calls: usize,
    /// Per-invocation transient bytes passed to host API writes.
    pub transient_host_api_h2d_bytes: usize,
    /// Per-invocation session-synthesized runtime-control host API writes.
    pub runtime_control_host_api_h2d_calls: usize,
    /// Per-invocation session-synthesized runtime-control bytes written.
    pub runtime_control_host_api_h2d_bytes: usize,
    /// Per-invocation retained-output host API reads.
    pub retained_host_api_d2h_calls: usize,
    /// Per-invocation retained-output bytes passed to host API reads.
    pub retained_host_api_d2h_bytes: usize,
    /// Nonzero schedule items launched by this invocation.
    pub kernel_launch_count: usize,
    /// Metal compute command buffers committed by this invocation; host API
    /// H2D/D2H copy calls are counted separately and excluded.
    pub command_submission_count: usize,
    /// Metal compute command buffers synchronously waited by this invocation;
    /// host API H2D/D2H copy calls are counted separately and excluded.
    pub command_wait_count: usize,
    /// Addressless schedule items skipped by this invocation.
    pub zero_item_count: usize,
    /// Logical outputs published after ordered projection.
    pub output_count: usize,
    /// Recurrent pairs committed atomically with this successful output.
    pub committed_state_pair_count: usize,
    /// Logical recurrent bytes committed by this successful output.
    pub committed_state_bytes: usize,
    /// Sparse recurrent elements committed by this successful output.
    pub committed_state_work_items: usize,
    /// Next append row after this success, or `None` for a stateless session.
    pub committed_state_position: Option<usize>,
}

impl MetalScoreboardRun {
    fn from_report(report: &super::MetalDeviceRunReport) -> Self {
        Self {
            successful_invocation: report.successful_invocation,
            first_successful_run: report.first_successful_run,
            run_wall_time: report.run_wall_time,
            synchronous_transaction_wall_time: report.synchronous_transaction_wall_time,
            transient_host_api_h2d_calls: report.transient_h2d_calls,
            transient_host_api_h2d_bytes: report.transient_h2d_bytes,
            runtime_control_host_api_h2d_calls: report.runtime_control_h2d_calls,
            runtime_control_host_api_h2d_bytes: report.runtime_control_h2d_bytes,
            retained_host_api_d2h_calls: report.retained_d2h_calls,
            retained_host_api_d2h_bytes: report.retained_d2h_bytes,
            kernel_launch_count: report.kernel_launch_count,
            command_submission_count: report.command_submission_count,
            command_wait_count: report.command_wait_count,
            zero_item_count: report.zero_item_count,
            output_count: report.output_count,
            committed_state_pair_count: report.committed_state_pair_count,
            committed_state_bytes: report.committed_state_bytes,
            committed_state_work_items: report.committed_state_work_items,
            committed_state_position: report.committed_state_position,
        }
    }
}

/// Immutable snapshot of one recorder's exact successful-run prefix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalSessionScoreboardReport {
    /// JSON/report schema version.
    pub format_version: u32,
    /// Caller-provided workload, revision, and evidence labels.
    #[serde(flatten)]
    pub context: MetalScoreboardContext,
    /// Exact capture-plus-resident-payload identity.
    pub deployment_identity: u64,
    /// Authenticated captured schedule identity.
    pub capture_identity: u64,
    /// Backend-neutral logical execution-plan identity.
    pub execution_plan_identity: u64,
    /// Handle-free selected-device evidence.
    #[serde(serialize_with = "serialize_device")]
    pub device: MetalDeviceInfo,
    /// Capture-order resident, state, and transient input descriptors.
    pub inputs: Vec<MetalScoreboardInput>,
    /// Stateless or authenticated append-only execution policy.
    pub state_policy: MetalScoreboardStatePolicy,
    /// Number of authenticated recurrent input/output pairs.
    pub state_pair_count: usize,
    /// Bytes in one logical recurrent-state bank.
    pub logical_state_bytes: usize,
    /// Number of logical recurrent bank sets owned by the session.
    pub state_bank_count: usize,
    /// Logical bytes represented by the selected recurrent-bank policy.
    pub state_device_bytes: usize,
    /// Logical recurrent bytes committed by each successful append.
    pub append_state_row_bytes: usize,
    /// Sparse recurrent elements committed by each successful append.
    pub append_state_work_items: usize,
    /// Ordered cache identities of the plan's nonzero rendered kernels.
    pub rendered_cache_keys: Vec<String>,
    /// Logical requested outputs, including ordered duplicates and aliases.
    pub requested_output_count: usize,
    /// Capture-owned dense tensor constants.
    pub captured_constant_count: usize,
    /// Exact raw tensor payload bytes of capture-owned dense constants.
    pub captured_constant_bytes: usize,
    /// Capture-owned packed GGUF constants.
    pub captured_quantized_constant_count: usize,
    /// Exact raw packed GGUF payload bytes of capture-owned constants.
    pub captured_quantized_constant_bytes: usize,
    /// Declared payload bytes for immutable resident inputs.
    pub declared_resident_input_bytes: usize,
    /// Declared payload bytes for each invocation's transient inputs.
    pub declared_transient_input_bytes: usize,
    /// Declared payload bytes for session-synthesized runtime controls.
    pub declared_runtime_control_input_bytes: usize,
    /// Host wall time spent in resource-free planning and rendering.
    pub planning_wall_time: Duration,
    /// Host wall time spent compiling, allocating, and creating the queue.
    pub native_prepare_wall_time: Duration,
    /// Host wall time spent building cache-miss native libraries and pipelines.
    pub cache_miss_pipeline_build_wall_time: Duration,
    /// Host wall time spent issuing immutable resident and initial-state writes.
    pub resident_upload_host_wall_time: Duration,
    /// Host wall time for the first recorded successful run, if any.
    pub first_run_host_wall_time: Option<Duration>,
    /// Invocation-order host wall times after the first successful run.
    pub steady_run_host_wall_times: Vec<Duration>,
    /// Integer nearest-rank summary of the steady-run samples.
    pub steady_run_host_wall_time_summary: Option<MetalHostWallTimeSummary>,
    /// First run's host wall time around copies plus launch/wait calls.
    pub first_synchronous_transaction_host_wall_time: Option<Duration>,
    /// Ordered synchronous transaction host wall times after the first run.
    pub steady_synchronous_transaction_host_wall_times: Vec<Duration>,
    /// Integer nearest-rank summary of steady transaction samples.
    pub steady_synchronous_transaction_host_wall_time_summary: Option<MetalHostWallTimeSummary>,
    /// Backend-neutral schedule item count before zero-domain admission.
    pub logical_schedule_item_count: usize,
    /// Peak simultaneously live logical temporary allocation count.
    pub peak_logical_temporary_allocation_count: usize,
    /// Peak simultaneously live logical temporary payload bytes.
    pub peak_logical_temporary_bytes: usize,
    /// Physical tensor-slot count in the Metal static memory plan.
    pub planned_physical_static_tensor_slot_count: usize,
    /// Planned physical Metal slot bytes, including private zero-byte sentinels.
    pub planned_physical_static_tensor_slot_bytes: usize,
    /// Planned private sentinel handles for logical zero-byte values.
    pub planned_zero_byte_sentinel_count: usize,
    /// Nonzero schedule items expected to launch per run.
    pub planned_kernel_count: usize,
    /// Addressless schedule items expected to skip per run.
    pub planned_zero_item_count: usize,
    /// Kernel-cache lookups made while preparing this session.
    pub pipeline_cache_request_count: usize,
    /// Preparation cache requests already present on this device object.
    pub pipeline_cache_hit_count: usize,
    /// Preparation cache requests newly created on this device object.
    pub pipeline_cache_miss_count: usize,
    /// Host API writes issued once for immutable residents.
    pub resident_host_api_h2d_calls: usize,
    /// Bytes passed to immutable resident host API writes.
    pub resident_host_api_h2d_bytes: usize,
    /// One-time host API writes for initial recurrent state.
    pub initial_state_host_api_h2d_calls: usize,
    /// One-time initial recurrent-state bytes passed to host API writes.
    pub initial_state_host_api_h2d_bytes: usize,
    /// Host API writes issued for all recorded transient inputs.
    pub transient_host_api_h2d_calls: usize,
    /// Bytes passed to recorded transient host API writes.
    pub transient_host_api_h2d_bytes: usize,
    /// Host API writes issued for session-synthesized runtime controls.
    pub runtime_control_host_api_h2d_calls: usize,
    /// Bytes passed to session-synthesized runtime-control writes.
    pub runtime_control_host_api_h2d_bytes: usize,
    /// Host API reads issued for recorded retained outputs.
    pub retained_host_api_d2h_calls: usize,
    /// Bytes passed to recorded retained-output host API reads.
    pub retained_host_api_d2h_bytes: usize,
    /// Resident, initial-state, caller-transient, and runtime-control writes.
    pub host_api_h2d_calls: usize,
    /// Resident, initial-state, caller-transient, and runtime-control bytes.
    pub host_api_h2d_bytes: usize,
    /// Total recorded retained-output host API reads.
    pub host_api_d2h_calls: usize,
    /// Total recorded bytes passed to retained-output host API reads.
    pub host_api_d2h_bytes: usize,
    /// Exact launches across the recorded successful-run prefix.
    pub kernel_launch_count: usize,
    /// Exact Metal compute command-buffer submissions across the successful-run
    /// prefix; host API H2D/D2H copy calls are excluded.
    pub command_submission_count: usize,
    /// Exact synchronous Metal compute command waits across the successful-run
    /// prefix; host API H2D/D2H copy calls are excluded.
    pub command_wait_count: usize,
    /// Addressless item skips across the recorded successful-run prefix.
    pub zero_item_count: usize,
    /// Recurrent pair commits across the recorded successful-run prefix.
    pub committed_state_pair_count: usize,
    /// Logical recurrent bytes committed across the successful-run prefix.
    pub committed_state_bytes: usize,
    /// Sparse recurrent elements committed across the successful-run prefix.
    pub committed_state_work_items: usize,
    /// Next append row after the recorded prefix, or `None` when stateless.
    pub committed_state_position: Option<usize>,
    /// Invocation-order exact counters for every recorded successful run.
    pub successful_runs: Vec<MetalScoreboardRun>,
    /// Length of the recorded consecutive successful-run prefix.
    pub successful_run_count: u64,
    /// Strict execution fallback count, which is always zero for this path.
    pub fallback_count: usize,
}

impl MetalSessionScoreboardReport {
    /// Encodes this observation deterministically as versioned UTF-8 JSON.
    /// Durations use integer `{secs,nanos}` objects rather than floating point.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, MetalScoreboardError> {
        let mut bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| MetalScoreboardError::Json(error.to_string()))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Writes the deterministic JSON representation to `path`.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), MetalScoreboardError> {
        fs::write(path, self.to_json_bytes()?).map_err(|error| MetalScoreboardError::Io {
            operation: "write report",
            kind: error.kind(),
        })
    }
}

/// Failure to construct, bind, record, or serialize a Metal scoreboard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalScoreboardError {
    /// A report label was empty, multiline/control-bearing, or over its bound.
    InvalidMetadata(&'static str),
    /// The recorder was already bound successfully.
    AlreadyBound,
    /// A run or report was requested before a successful bind.
    NotBound,
    /// The session is not a fresh preparation of the snapshotted deployment.
    PlanMismatch,
    /// Token-step evidence requires one committed row per successful invocation.
    UnsupportedAppendSpan { span_rows: usize },
    /// The successful run belongs to another prepared session.
    WrongSession,
    /// A successful run was skipped, repeated, or reordered.
    OutOfOrder { expected: u64, actual: u64 },
    /// A successful run did not commit the authenticated append position.
    StateOutOfOrder {
        expected: Option<usize>,
        actual: Option<usize>,
    },
    /// A successful run's committed recurrent work disagrees with its plan.
    StateCommitMismatch,
    /// Exact counter arithmetic exceeded the host integer representation.
    Overflow,
    /// Deterministic JSON encoding failed.
    Json(String),
    /// Writing the report failed without retaining a platform-specific message.
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
}

impl fmt::Display for MetalScoreboardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMetadata(field) => write!(formatter, "invalid scoreboard {field} label"),
            Self::AlreadyBound => write!(formatter, "Metal scoreboard is already bound"),
            Self::NotBound => write!(formatter, "Metal scoreboard is not bound"),
            Self::PlanMismatch => write!(
                formatter,
                "Metal session does not match the scoreboard inference plan"
            ),
            Self::UnsupportedAppendSpan { span_rows } => write!(
                formatter,
                "Metal token-step scoreboard requires one appended row per successful invocation; got a span of {span_rows} rows"
            ),
            Self::WrongSession => write!(formatter, "Metal run belongs to another session"),
            Self::OutOfOrder { expected, actual } => write!(
                formatter,
                "Metal run {actual} is out of order; expected successful invocation {expected}"
            ),
            Self::StateOutOfOrder { expected, actual } => write!(
                formatter,
                "Metal run committed state position {actual:?}; expected {expected:?}"
            ),
            Self::StateCommitMismatch => write!(
                formatter,
                "Metal run committed state work that disagrees with the scoreboard plan"
            ),
            Self::Overflow => write!(formatter, "Metal scoreboard counter overflow"),
            Self::Json(error) => {
                write!(formatter, "Metal scoreboard JSON encoding failed: {error}")
            }
            Self::Io { operation, kind } => {
                write!(formatter, "Metal scoreboard {operation} failed: {kind:?}")
            }
        }
    }
}

impl std::error::Error for MetalScoreboardError {}

/// Opt-in recorder for the exact successful invocation prefix of one prepared
/// [`MetalDeviceSession`]. It neither submits work nor changes session policy.
pub struct MetalSessionScoreboard {
    context: MetalScoreboardContext,
    deployment_identity: u64,
    capture_identity: u64,
    execution_plan_identity: u64,
    logical_schedule_item_count: usize,
    peak_logical_temporary_allocation_count: usize,
    peak_logical_temporary_bytes: usize,
    plan_summary: MetalDeviceSessionSummary,
    state_policy: MetalScoreboardStatePolicy,
    append_span_rows: usize,
    inputs: Vec<MetalScoreboardInput>,
    binding: Option<BoundScoreboard>,
    runs: Vec<MetalScoreboardRun>,
    totals: RunTotals,
}

struct BoundScoreboard {
    session_token: Rc<()>,
    device: MetalDeviceInfo,
    preparation: super::MetalDevicePreparationReport,
}

struct ScoreboardPlanView<'a> {
    deployment_identity: u64,
    capture: &'a CapturedSchedule,
    execution_plan: &'a ExecutionPlanSummary,
    summary: &'a MetalDeviceSessionSummary,
    resident_inputs: &'a [ReplayInput],
    state_inputs: &'a [ReplayInput],
    state_policy: MetalScoreboardStatePolicy,
}

impl MetalSessionScoreboard {
    /// Snapshots resource-free plan evidence without creating a Metal resource.
    pub fn new(context: MetalScoreboardContext, plan: &MetalInferencePlan) -> Self {
        Self::from_plan(
            context,
            ScoreboardPlanView {
                deployment_identity: plan.deployment_identity(),
                capture: plan.capture(),
                execution_plan: plan.execution_plan(),
                summary: plan.summary(),
                resident_inputs: plan.resident_inputs(),
                state_inputs: &[],
                state_policy: MetalScoreboardStatePolicy::Stateless,
            },
        )
    }

    /// Snapshots one authenticated append-state deployment without creating a
    /// Metal resource. Generation/model orchestration remains caller-owned.
    ///
    /// Token-step callers should use [`Self::try_new_append_state_v4`] so a
    /// multirow plan rejects before preparation. This source-compatible legacy
    /// constructor permits inspection, but its recorder rejects a multirow
    /// session with [`MetalScoreboardError::UnsupportedAppendSpan`] at bind.
    pub fn new_append_state(
        context: MetalScoreboardContext,
        plan: &MetalAppendStateInferencePlan,
    ) -> Self {
        let mut scoreboard = Self::from_plan(
            context,
            ScoreboardPlanView {
                deployment_identity: plan.deployment_identity(),
                capture: plan.capture(),
                execution_plan: plan.execution_plan(),
                summary: plan.summary(),
                resident_inputs: plan.resident_inputs(),
                state_inputs: plan.state_inputs(),
                state_policy: MetalScoreboardStatePolicy::Append,
            },
        );
        scoreboard.append_span_rows = plan.append_span_rows();
        scoreboard
    }

    /// Creates a token-step recorder only when one success commits exactly one row.
    pub fn try_new_append_state_v4(
        context: MetalScoreboardContext,
        plan: &MetalAppendStateInferencePlan,
    ) -> Result<Self, MetalScoreboardError> {
        let span_rows = plan.append_span_rows();
        if span_rows != 1 {
            return Err(MetalScoreboardError::UnsupportedAppendSpan { span_rows });
        }
        Ok(Self::new_append_state(context, plan))
    }

    fn from_plan(context: MetalScoreboardContext, plan: ScoreboardPlanView<'_>) -> Self {
        let resident_names = plan
            .resident_inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        let state_names = plan
            .state_inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        let runtime_control_names = plan
            .summary
            .runtime_control_input_names
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let inputs = plan
            .capture
            .inputs
            .iter()
            .map(|input| {
                scoreboard_input(input, &resident_names, &state_names, &runtime_control_names)
            })
            .collect();
        Self {
            context,
            deployment_identity: plan.deployment_identity,
            capture_identity: plan.capture.identity,
            execution_plan_identity: plan.execution_plan.identity,
            logical_schedule_item_count: plan.execution_plan.schedule_item_count,
            peak_logical_temporary_allocation_count: plan.execution_plan.peak_logical_allocations,
            peak_logical_temporary_bytes: plan.execution_plan.peak_logical_bytes,
            plan_summary: plan.summary.clone(),
            state_policy: plan.state_policy,
            append_span_rows: 0,
            inputs,
            binding: None,
            runs: Vec::new(),
            totals: RunTotals::default(),
        }
    }

    /// Binds this recorder once to the session prepared from its exact inference
    /// deployment. No measurement is recorded by binding.
    pub fn bind(&mut self, session: &MetalDeviceSession) -> Result<(), MetalScoreboardError> {
        if self.state_policy == MetalScoreboardStatePolicy::Append && self.append_span_rows != 1 {
            return Err(MetalScoreboardError::UnsupportedAppendSpan {
                span_rows: self.append_span_rows,
            });
        }
        if self.binding.is_some() {
            return Err(MetalScoreboardError::AlreadyBound);
        }
        let expected_position = match self.state_policy {
            MetalScoreboardStatePolicy::Stateless => None,
            MetalScoreboardStatePolicy::Append => Some(0),
        };
        if session.inference_deployment_identity() != Some(self.deployment_identity)
            || session.capture().identity != self.capture_identity
            || session.summary() != &self.plan_summary
            || session.successful_run_count() != 0
            || session.committed_state_position() != expected_position
        {
            return Err(MetalScoreboardError::PlanMismatch);
        }
        self.binding = Some(BoundScoreboard {
            session_token: session.session_token().clone(),
            device: session.device_info().clone(),
            preparation: session.preparation_report().clone(),
        });
        Ok(())
    }

    /// Records one successful run. The run must be the next unrecorded success
    /// from the exact bound session; failed calls produce no run and no change.
    pub fn record(&mut self, run: &MetalDeviceRun) -> Result<(), MetalScoreboardError> {
        let binding = self
            .binding
            .as_ref()
            .ok_or(MetalScoreboardError::NotBound)?;
        if !run.belongs_to(&binding.session_token) {
            return Err(MetalScoreboardError::WrongSession);
        }
        let expected = u64::try_from(self.runs.len())
            .map_err(|_| MetalScoreboardError::Overflow)?
            .checked_add(1)
            .ok_or(MetalScoreboardError::Overflow)?;
        let actual = run.report().successful_invocation;
        if actual != expected || run.report().first_successful_run != (actual == 1) {
            return Err(MetalScoreboardError::OutOfOrder { expected, actual });
        }
        let recorded = MetalScoreboardRun::from_report(run.report());
        let expected_position = match self.state_policy {
            MetalScoreboardStatePolicy::Stateless => None,
            MetalScoreboardStatePolicy::Append => {
                Some(usize::try_from(expected).map_err(|_| MetalScoreboardError::Overflow)?)
            }
        };
        if recorded.committed_state_position != expected_position {
            return Err(MetalScoreboardError::StateOutOfOrder {
                expected: expected_position,
                actual: recorded.committed_state_position,
            });
        }
        let state_commit_matches = match self.state_policy {
            MetalScoreboardStatePolicy::Stateless => {
                recorded.committed_state_pair_count == 0
                    && recorded.committed_state_bytes == 0
                    && recorded.committed_state_work_items == 0
            }
            MetalScoreboardStatePolicy::Append => {
                recorded.committed_state_pair_count == self.plan_summary.state_pair_count
                    && recorded.committed_state_bytes == self.plan_summary.append_state_row_bytes
                    && recorded.committed_state_work_items
                        == self.plan_summary.append_state_work_items
            }
        };
        if !state_commit_matches {
            return Err(MetalScoreboardError::StateCommitMismatch);
        }
        let mut totals = self.totals.clone();
        totals.add(&recorded)?;
        self.runs.push(recorded);
        self.totals = totals;
        Ok(())
    }

    /// Returns an immutable snapshot of the currently recorded successful prefix.
    pub fn report(&self) -> Result<MetalSessionScoreboardReport, MetalScoreboardError> {
        let binding = self
            .binding
            .as_ref()
            .ok_or(MetalScoreboardError::NotBound)?;
        let preparation = &binding.preparation;
        let totals = RunTotals::checked(&self.runs)?;
        let first_run_host_wall_time = self.runs.first().map(|run| run.run_wall_time);
        let steady_run_host_wall_times = self
            .runs
            .iter()
            .skip(1)
            .map(|run| run.run_wall_time)
            .collect::<Vec<_>>();
        let first_synchronous_transaction_host_wall_time = self
            .runs
            .first()
            .map(|run| run.synchronous_transaction_wall_time);
        let steady_synchronous_transaction_host_wall_times = self
            .runs
            .iter()
            .skip(1)
            .map(|run| run.synchronous_transaction_wall_time)
            .collect::<Vec<_>>();
        let initialization_h2d_calls = preparation
            .resident_h2d_calls
            .checked_add(preparation.initial_state_h2d_calls)
            .ok_or(MetalScoreboardError::Overflow)?;
        let initialization_h2d_bytes = preparation
            .resident_h2d_bytes
            .checked_add(preparation.initial_state_h2d_bytes)
            .ok_or(MetalScoreboardError::Overflow)?;
        let host_api_h2d_calls = initialization_h2d_calls
            .checked_add(totals.transient_h2d_calls)
            .and_then(|total| total.checked_add(totals.runtime_control_h2d_calls))
            .ok_or(MetalScoreboardError::Overflow)?;
        let host_api_h2d_bytes = initialization_h2d_bytes
            .checked_add(totals.transient_h2d_bytes)
            .and_then(|total| total.checked_add(totals.runtime_control_h2d_bytes))
            .ok_or(MetalScoreboardError::Overflow)?;
        let committed_state_position = match self.state_policy {
            MetalScoreboardStatePolicy::Stateless => None,
            MetalScoreboardStatePolicy::Append => Some(
                self.runs
                    .last()
                    .and_then(|run| run.committed_state_position)
                    .unwrap_or(0),
            ),
        };
        Ok(MetalSessionScoreboardReport {
            format_version: METAL_SESSION_SCOREBOARD_FORMAT_VERSION,
            context: self.context.clone(),
            deployment_identity: self.deployment_identity,
            capture_identity: self.capture_identity,
            execution_plan_identity: self.execution_plan_identity,
            device: binding.device.clone(),
            inputs: self.inputs.clone(),
            state_policy: self.state_policy,
            state_pair_count: self.plan_summary.state_pair_count,
            logical_state_bytes: self.plan_summary.logical_state_bytes,
            state_bank_count: self.plan_summary.state_bank_count,
            state_device_bytes: self.plan_summary.state_device_bytes,
            append_state_row_bytes: self.plan_summary.append_state_row_bytes,
            append_state_work_items: self.plan_summary.append_state_work_items,
            rendered_cache_keys: self.plan_summary.rendered_cache_keys.clone(),
            requested_output_count: self.plan_summary.requested_output_count,
            captured_constant_count: self.plan_summary.constant_count,
            captured_constant_bytes: self.plan_summary.constant_bytes,
            captured_quantized_constant_count: self.plan_summary.quantized_constant_count,
            captured_quantized_constant_bytes: self.plan_summary.quantized_constant_bytes,
            declared_resident_input_bytes: self.plan_summary.resident_input_bytes,
            declared_transient_input_bytes: self.plan_summary.transient_input_bytes,
            declared_runtime_control_input_bytes: self.plan_summary.runtime_control_input_bytes,
            planning_wall_time: preparation.planning_wall_time,
            native_prepare_wall_time: preparation.native_prepare_wall_time,
            cache_miss_pipeline_build_wall_time: preparation.cache_miss_pipeline_build_wall_time,
            resident_upload_host_wall_time: preparation.initialization_upload_wall_time,
            first_run_host_wall_time,
            steady_run_host_wall_time_summary: wall_time_summary(&steady_run_host_wall_times),
            steady_run_host_wall_times,
            first_synchronous_transaction_host_wall_time,
            steady_synchronous_transaction_host_wall_time_summary: wall_time_summary(
                &steady_synchronous_transaction_host_wall_times,
            ),
            steady_synchronous_transaction_host_wall_times,
            logical_schedule_item_count: self.logical_schedule_item_count,
            peak_logical_temporary_allocation_count: self.peak_logical_temporary_allocation_count,
            peak_logical_temporary_bytes: self.peak_logical_temporary_bytes,
            planned_physical_static_tensor_slot_count: self.plan_summary.planned_slot_count,
            planned_physical_static_tensor_slot_bytes: self.plan_summary.planned_device_bytes,
            planned_zero_byte_sentinel_count: self.plan_summary.zero_byte_sentinel_count,
            planned_kernel_count: self.plan_summary.nonzero_item_count,
            planned_zero_item_count: self.plan_summary.zero_item_count,
            pipeline_cache_request_count: preparation.pipeline_cache_request_count,
            pipeline_cache_hit_count: preparation.pipeline_cache_hit_count,
            pipeline_cache_miss_count: preparation.pipeline_cache_miss_count,
            resident_host_api_h2d_calls: preparation.resident_h2d_calls,
            resident_host_api_h2d_bytes: preparation.resident_h2d_bytes,
            initial_state_host_api_h2d_calls: preparation.initial_state_h2d_calls,
            initial_state_host_api_h2d_bytes: preparation.initial_state_h2d_bytes,
            transient_host_api_h2d_calls: totals.transient_h2d_calls,
            transient_host_api_h2d_bytes: totals.transient_h2d_bytes,
            runtime_control_host_api_h2d_calls: totals.runtime_control_h2d_calls,
            runtime_control_host_api_h2d_bytes: totals.runtime_control_h2d_bytes,
            retained_host_api_d2h_calls: totals.retained_d2h_calls,
            retained_host_api_d2h_bytes: totals.retained_d2h_bytes,
            host_api_h2d_calls,
            host_api_h2d_bytes,
            host_api_d2h_calls: totals.retained_d2h_calls,
            host_api_d2h_bytes: totals.retained_d2h_bytes,
            kernel_launch_count: totals.kernel_launch_count,
            command_submission_count: totals.command_submission_count,
            command_wait_count: totals.command_wait_count,
            zero_item_count: totals.zero_item_count,
            committed_state_pair_count: totals.committed_state_pair_count,
            committed_state_bytes: totals.committed_state_bytes,
            committed_state_work_items: totals.committed_state_work_items,
            committed_state_position,
            successful_runs: self.runs.clone(),
            successful_run_count: u64::try_from(self.runs.len())
                .map_err(|_| MetalScoreboardError::Overflow)?,
            fallback_count: self.plan_summary.fallback_count,
        })
    }
}

#[derive(Clone, Default)]
struct RunTotals {
    transient_h2d_calls: usize,
    transient_h2d_bytes: usize,
    runtime_control_h2d_calls: usize,
    runtime_control_h2d_bytes: usize,
    retained_d2h_calls: usize,
    retained_d2h_bytes: usize,
    kernel_launch_count: usize,
    command_submission_count: usize,
    command_wait_count: usize,
    zero_item_count: usize,
    committed_state_pair_count: usize,
    committed_state_bytes: usize,
    committed_state_work_items: usize,
}

impl RunTotals {
    fn checked(runs: &[MetalScoreboardRun]) -> Result<Self, MetalScoreboardError> {
        let mut totals = Self::default();
        for run in runs {
            totals.add(run)?;
        }
        Ok(totals)
    }

    fn add(&mut self, report: &MetalScoreboardRun) -> Result<(), MetalScoreboardError> {
        macro_rules! checked_add {
            ($field:ident, $value:expr) => {
                self.$field = self
                    .$field
                    .checked_add($value)
                    .ok_or(MetalScoreboardError::Overflow)?;
            };
        }
        checked_add!(transient_h2d_calls, report.transient_host_api_h2d_calls);
        checked_add!(transient_h2d_bytes, report.transient_host_api_h2d_bytes);
        checked_add!(
            runtime_control_h2d_calls,
            report.runtime_control_host_api_h2d_calls
        );
        checked_add!(
            runtime_control_h2d_bytes,
            report.runtime_control_host_api_h2d_bytes
        );
        checked_add!(retained_d2h_calls, report.retained_host_api_d2h_calls);
        checked_add!(retained_d2h_bytes, report.retained_host_api_d2h_bytes);
        checked_add!(kernel_launch_count, report.kernel_launch_count);
        checked_add!(command_submission_count, report.command_submission_count);
        checked_add!(command_wait_count, report.command_wait_count);
        checked_add!(zero_item_count, report.zero_item_count);
        checked_add!(
            committed_state_pair_count,
            report.committed_state_pair_count
        );
        checked_add!(committed_state_bytes, report.committed_state_bytes);
        checked_add!(
            committed_state_work_items,
            report.committed_state_work_items
        );
        Ok(())
    }
}

fn validate_label(field: &'static str, value: &str) -> Result<(), MetalScoreboardError> {
    if value.is_empty() || value.len() > MAX_METADATA_BYTES || value.chars().any(char::is_control) {
        return Err(MetalScoreboardError::InvalidMetadata(field));
    }
    Ok(())
}

fn scoreboard_input(
    input: &ReplayInput,
    resident_names: &BTreeSet<&str>,
    state_names: &BTreeSet<&str>,
    runtime_control_names: &BTreeSet<&str>,
) -> MetalScoreboardInput {
    MetalScoreboardInput {
        name: input.name.clone(),
        node_id: input.node.index() as u64,
        shape: input.desc.shape.clone(),
        dtype: input.desc.dtype,
        bytes: input.desc.bytes,
        kind: if resident_names.contains(input.name.as_str()) {
            MetalScoreboardInputKind::Resident
        } else if state_names.contains(input.name.as_str()) {
            MetalScoreboardInputKind::State
        } else if runtime_control_names.contains(input.name.as_str()) {
            MetalScoreboardInputKind::RuntimeControl
        } else {
            MetalScoreboardInputKind::Transient
        },
    }
}

fn wall_time_summary(samples: &[Duration]) -> Option<MetalHostWallTimeSummary> {
    if samples.is_empty() {
        return None;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    Some(MetalHostWallTimeSummary {
        min: ordered[0],
        nearest_rank_p50: nearest_rank(&ordered, 50),
        nearest_rank_p95: nearest_rank(&ordered, 95),
        max: ordered[ordered.len() - 1],
    })
}

fn nearest_rank(ordered: &[Duration], percentile: usize) -> Duration {
    let rank =
        (ordered.len() / 100) * percentile + ((ordered.len() % 100) * percentile).div_ceil(100);
    let rank = rank.max(1);
    ordered[rank - 1]
}

fn serialize_dtype<S>(dtype: &DType, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(dtype.canonical_tinygrad_name())
}

fn serialize_device<S>(device: &MetalDeviceInfo, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    #[derive(Serialize)]
    struct Device<'a> {
        name: &'a str,
        registry_id: u64,
        max_buffer_length: usize,
        unified_memory: bool,
        family: &'a str,
    }

    Device {
        name: &device.name,
        registry_id: device.registry_id,
        max_buffer_length: device.capabilities.max_buffer_length,
        unified_memory: device.capabilities.unified_memory,
        family: &device.capabilities.family,
    }
    .serialize(serializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::metal::MetalCapabilities;

    fn fixed_report() -> MetalSessionScoreboardReport {
        let steady = vec![
            Duration::new(0, 9),
            Duration::new(0, 1),
            Duration::new(0, 5),
        ];
        MetalSessionScoreboardReport {
            format_version: METAL_SESSION_SCOREBOARD_FORMAT_VERSION,
            context: MetalScoreboardContext::new("linear", "abc123", "semantic mock").unwrap(),
            deployment_identity: 1,
            capture_identity: 2,
            execution_plan_identity: 3,
            device: MetalDeviceInfo {
                name: "Mock Metal".into(),
                registry_id: 7,
                capabilities: MetalCapabilities {
                    max_buffer_length: 1 << 20,
                    unified_memory: true,
                    family: "mock-family".into(),
                },
            },
            inputs: vec![
                MetalScoreboardInput {
                    name: "features".into(),
                    node_id: 4,
                    shape: Shape::from([1, 4]),
                    dtype: DType::F32,
                    bytes: 16,
                    kind: MetalScoreboardInputKind::Transient,
                },
                MetalScoreboardInput {
                    name: "position".into(),
                    node_id: 5,
                    shape: Shape::from([1]),
                    dtype: DType::I32,
                    bytes: 4,
                    kind: MetalScoreboardInputKind::RuntimeControl,
                },
            ],
            state_policy: MetalScoreboardStatePolicy::Stateless,
            state_pair_count: 0,
            logical_state_bytes: 0,
            state_bank_count: 0,
            state_device_bytes: 0,
            append_state_row_bytes: 0,
            append_state_work_items: 0,
            rendered_cache_keys: vec!["kernel-a".into()],
            requested_output_count: 1,
            captured_constant_count: 0,
            captured_constant_bytes: 0,
            captured_quantized_constant_count: 0,
            captured_quantized_constant_bytes: 0,
            declared_resident_input_bytes: 0,
            declared_transient_input_bytes: 16,
            declared_runtime_control_input_bytes: 4,
            planning_wall_time: Duration::new(0, 11),
            native_prepare_wall_time: Duration::new(0, 12),
            cache_miss_pipeline_build_wall_time: Duration::new(0, 7),
            resident_upload_host_wall_time: Duration::new(0, 13),
            first_run_host_wall_time: Some(Duration::new(0, 20)),
            steady_run_host_wall_times: steady.clone(),
            steady_run_host_wall_time_summary: wall_time_summary(&steady),
            first_synchronous_transaction_host_wall_time: Some(Duration::new(0, 10)),
            steady_synchronous_transaction_host_wall_times: vec![
                Duration::new(0, 4),
                Duration::new(0, 3),
                Duration::new(0, 2),
            ],
            steady_synchronous_transaction_host_wall_time_summary: wall_time_summary(&[
                Duration::new(0, 4),
                Duration::new(0, 3),
                Duration::new(0, 2),
            ]),
            logical_schedule_item_count: 1,
            peak_logical_temporary_allocation_count: 2,
            peak_logical_temporary_bytes: 32,
            planned_physical_static_tensor_slot_count: 2,
            planned_physical_static_tensor_slot_bytes: 32,
            planned_zero_byte_sentinel_count: 0,
            planned_kernel_count: 1,
            planned_zero_item_count: 0,
            pipeline_cache_request_count: 1,
            pipeline_cache_hit_count: 0,
            pipeline_cache_miss_count: 1,
            resident_host_api_h2d_calls: 1,
            resident_host_api_h2d_bytes: 16,
            initial_state_host_api_h2d_calls: 0,
            initial_state_host_api_h2d_bytes: 0,
            transient_host_api_h2d_calls: 4,
            transient_host_api_h2d_bytes: 64,
            runtime_control_host_api_h2d_calls: 4,
            runtime_control_host_api_h2d_bytes: 16,
            retained_host_api_d2h_calls: 4,
            retained_host_api_d2h_bytes: 32,
            host_api_h2d_calls: 9,
            host_api_h2d_bytes: 96,
            host_api_d2h_calls: 4,
            host_api_d2h_bytes: 32,
            kernel_launch_count: 4,
            command_submission_count: 4,
            command_wait_count: 4,
            zero_item_count: 0,
            committed_state_pair_count: 0,
            committed_state_bytes: 0,
            committed_state_work_items: 0,
            committed_state_position: None,
            successful_runs: [(1, 20, 10), (2, 9, 4), (3, 1, 3), (4, 5, 2)]
                .into_iter()
                .map(
                    |(successful_invocation, run_nanos, transaction_nanos)| MetalScoreboardRun {
                        successful_invocation,
                        first_successful_run: successful_invocation == 1,
                        run_wall_time: Duration::new(0, run_nanos),
                        synchronous_transaction_wall_time: Duration::new(0, transaction_nanos),
                        transient_host_api_h2d_calls: 1,
                        transient_host_api_h2d_bytes: 16,
                        runtime_control_host_api_h2d_calls: 1,
                        runtime_control_host_api_h2d_bytes: 4,
                        retained_host_api_d2h_calls: 1,
                        retained_host_api_d2h_bytes: 8,
                        kernel_launch_count: 1,
                        command_submission_count: 1,
                        command_wait_count: 1,
                        zero_item_count: 0,
                        output_count: 1,
                        committed_state_pair_count: 0,
                        committed_state_bytes: 0,
                        committed_state_work_items: 0,
                        committed_state_position: None,
                    },
                )
                .collect(),
            successful_run_count: 4,
            fallback_count: 0,
        }
    }

    #[test]
    fn scoreboard_json_is_deterministic_and_uses_integer_durations() {
        let report = fixed_report();
        assert_eq!(
            report.steady_run_host_wall_time_summary,
            Some(MetalHostWallTimeSummary {
                min: Duration::new(0, 1),
                nearest_rank_p50: Duration::new(0, 5),
                nearest_rank_p95: Duration::new(0, 9),
                max: Duration::new(0, 9),
            })
        );
        let first = report.to_json_bytes().unwrap();
        assert_eq!(first, report.to_json_bytes().unwrap());
        let value: serde_json::Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(value["format_version"], 5);
        assert_eq!(value["workload"], "linear");
        assert_eq!(value["implementation_revision"], "abc123");
        assert_eq!(value["evidence"], "semantic mock");
        assert_eq!(value["planning_wall_time"]["secs"], 0);
        assert_eq!(value["planning_wall_time"]["nanos"], 11);
        assert_eq!(value["inputs"][0]["dtype"], "float");
        assert_eq!(value["inputs"][1]["kind"], "runtime_control");
        assert_eq!(value["deployment_identity"], 1);
        assert_eq!(value["runtime_control_host_api_h2d_calls"], 4);
        assert_eq!(value["command_submission_count"], 4);
        assert_eq!(value["command_wait_count"], 4);
        assert_eq!(
            value["successful_runs"][0]["runtime_control_host_api_h2d_bytes"],
            4
        );
        assert_eq!(value["successful_runs"].as_array().unwrap().len(), 4);
        let path = std::env::temp_dir().join(format!(
            "rustgrad-metal-scoreboard-{}-{}.json",
            std::process::id(),
            report.deployment_identity
        ));
        report.write_json(&path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), first);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn scoreboard_context_rejects_empty_multiline_and_unbounded_labels() {
        assert_eq!(
            MetalScoreboardContext::new("", "revision", "evidence").unwrap_err(),
            MetalScoreboardError::InvalidMetadata("workload")
        );
        assert_eq!(
            MetalScoreboardContext::new("workload", "revision", "line\nbreak").unwrap_err(),
            MetalScoreboardError::InvalidMetadata("evidence")
        );
        assert!(
            MetalScoreboardContext::new("workload", "x".repeat(MAX_METADATA_BYTES + 1), "evidence")
                .is_err()
        );
    }
}
