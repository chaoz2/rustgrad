//! Ordered scoreboard evidence for a two-program Metal Llama workload.

use crate::runtime::metal::{
    METAL_SESSION_SCOREBOARD_FORMAT_VERSION, MetalDeviceRunReport, MetalScoreboardContext,
    MetalScoreboardError, MetalScoreboardRun, MetalSessionScoreboardReport,
};
use serde::Serialize;
use std::{fs, path::Path, time::Duration};

/// Current deterministic JSON schema for [`LlamaMetalExecutionScoreboardReport`].
pub const LLAMA_METAL_EXECUTION_SCOREBOARD_FORMAT_VERSION: u32 = 2;

/// Physical program that committed one Llama workload invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaMetalScoreboardProgram {
    /// Fixed-span state-only prompt ingestion.
    FixedPrefill,
    /// Single-token execution, including prompt tails and decode.
    TokenStep,
}

/// Workload phase attributed by the Llama coordinator, independently of the
/// physical Metal program that executed the invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaMetalScoreboardPhase {
    /// A direct public [`super::LlamaMetalSession::run_token`] invocation.
    Standalone,
    /// Fixed-span or token-step work that ingests prompt rows.
    PromptPrefill,
    /// A generated token fed back through the token-step program.
    SteadyDecode,
}

/// One globally ordered success linked to its real physical-session record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LlamaMetalScoreboardInvocation {
    /// One-based success ordinal across both physical programs.
    pub successful_invocation: u64,
    /// True only for the first success across the complete Llama workload.
    pub first_successful_run: bool,
    /// Physical program that executed this invocation.
    pub program: LlamaMetalScoreboardProgram,
    /// Coordinator-owned workload phase for this invocation.
    pub phase: LlamaMetalScoreboardPhase,
    /// One-based success ordinal in that physical program's own session.
    pub program_successful_invocation: u64,
    /// Rows atomically committed by this invocation.
    pub append_span_rows: usize,
    /// Logical recurrent pairs committed by this invocation.
    pub committed_state_pair_count: usize,
    /// Logical recurrent bytes committed by this invocation.
    pub committed_state_bytes: usize,
    /// Sparse recurrent elements committed by this invocation.
    pub committed_state_work_items: usize,
    /// Exact next shared append position after this invocation.
    pub committed_state_position: usize,
}

impl LlamaMetalScoreboardInvocation {
    pub(super) fn from_report(
        program: LlamaMetalScoreboardProgram,
        phase: LlamaMetalScoreboardPhase,
        program_successful_invocation: u64,
        append_span_rows: usize,
        report: &MetalDeviceRunReport,
    ) -> Result<Self, MetalScoreboardError> {
        Ok(Self {
            successful_invocation: report.successful_invocation,
            first_successful_run: report.first_successful_run,
            program,
            phase,
            program_successful_invocation,
            append_span_rows,
            committed_state_pair_count: report.committed_state_pair_count,
            committed_state_bytes: report.committed_state_bytes,
            committed_state_work_items: report.committed_state_work_items,
            committed_state_position: report
                .committed_state_position
                .ok_or(MetalScoreboardError::StateCommitMismatch)?,
        })
    }
}

/// Checked totals for one Llama scoreboard workload phase.
///
/// These counters are derived from the recorder-owned physical
/// [`MetalScoreboardRun`] records. They exclude preparation, tokenization,
/// sampling, allocator/RSS measurements, and physical-bus transfer claims.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LlamaMetalScoreboardPhaseAggregate {
    /// Prompt, decode-feedback, or standalone rows committed in this phase.
    pub committed_token_count: usize,
    /// Successful physical invocations attributed to this phase.
    pub successful_invocation_count: u64,
    /// Sum of host-observed run wall times for the phase.
    pub host_run_wall_time: Duration,
    /// Sum of synchronous host transaction wall times for the phase.
    pub host_synchronous_transaction_wall_time: Duration,
    /// All-or-none sum of compute-command GPU timestamp intervals.
    pub gpu_command_execution_time: Option<Duration>,
    /// Metal kernel launches in this phase.
    pub kernel_launch_count: usize,
    /// Metal compute command buffers submitted in this phase.
    pub command_submission_count: usize,
    /// Metal compute command buffers synchronously waited in this phase.
    pub command_wait_count: usize,
    /// Per-invocation transient host API writes in this phase.
    pub transient_host_api_h2d_calls: usize,
    /// Transient bytes passed to host API writes in this phase.
    pub transient_host_api_h2d_bytes: usize,
    /// Session-synthesized runtime-control host API writes in this phase.
    pub runtime_control_host_api_h2d_calls: usize,
    /// Runtime-control bytes passed to host API writes in this phase.
    pub runtime_control_host_api_h2d_bytes: usize,
    /// Retained-output host API reads in this phase.
    pub retained_host_api_d2h_calls: usize,
    /// Retained-output bytes passed to host API reads in this phase.
    pub retained_host_api_d2h_bytes: usize,
}

impl LlamaMetalScoreboardPhaseAggregate {
    /// Host-run rows per second. This is not end-to-end or GPU throughput.
    pub fn host_tokens_per_second(&self) -> Option<f64> {
        tokens_per_second(self.committed_token_count, self.host_run_wall_time)
    }

    /// Rows per second over complete compute-command GPU timestamp intervals.
    ///
    /// This excludes host work and copies and returns `None` when timing is
    /// unavailable for any invocation in this phase.
    pub fn gpu_command_tokens_per_second(&self) -> Option<f64> {
        self.gpu_command_execution_time
            .and_then(|duration| tokens_per_second(self.committed_token_count, duration))
    }
}

/// Immutable evidence for one ordered Llama workload over its authentic Metal
/// token-step session and optional authentic fixed-prefill session.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LlamaMetalExecutionScoreboardReport {
    /// JSON/report schema version.
    pub format_version: u32,
    /// Shared caller-provided workload, revision, and evidence labels.
    #[serde(flatten)]
    pub context: MetalScoreboardContext,
    /// Evidence bound to the real token-step physical session.
    pub token_step: MetalSessionScoreboardReport,
    /// Evidence bound to the real fixed-prefill physical session, when enabled.
    pub fixed_prefill: Option<MetalSessionScoreboardReport>,
    /// Global successful invocation order across both component sessions.
    pub successful_runs: Vec<LlamaMetalScoreboardInvocation>,
    /// Total successful invocations across both component sessions.
    pub successful_run_count: u64,
    /// Checked recorder-derived prompt-ingestion totals.
    pub prompt_prefill: LlamaMetalScoreboardPhaseAggregate,
    /// Checked recorder-derived generated-token feedback totals.
    pub steady_decode: LlamaMetalScoreboardPhaseAggregate,
    /// Checked recorder-derived direct `run_token` totals.
    pub standalone: LlamaMetalScoreboardPhaseAggregate,
    /// Exact next shared append position after the recorded prefix.
    pub committed_state_position: usize,
    /// Logical recurrent pair commits across the recorded prefix.
    pub committed_state_pair_count: usize,
    /// Logical recurrent bytes committed across the recorded prefix.
    pub committed_state_bytes: usize,
    /// Sparse recurrent elements committed across the recorded prefix.
    pub committed_state_work_items: usize,
    /// Strict execution fallback count, summed across component sessions.
    pub fallback_count: usize,
}

impl LlamaMetalExecutionScoreboardReport {
    pub(crate) fn new(
        token_step: MetalSessionScoreboardReport,
        fixed_prefill: Option<MetalSessionScoreboardReport>,
        successful_runs: Vec<LlamaMetalScoreboardInvocation>,
    ) -> Result<Self, MetalScoreboardError> {
        if token_step.state_policy != crate::runtime::metal::MetalScoreboardStatePolicy::Append
            || token_step.format_version != METAL_SESSION_SCOREBOARD_FORMAT_VERSION
            || token_step.append_span_rows != 1
            || fixed_prefill.as_ref().is_some_and(|report| {
                report.state_policy != crate::runtime::metal::MetalScoreboardStatePolicy::Append
                    || report.format_version != METAL_SESSION_SCOREBOARD_FORMAT_VERSION
                    || report.append_span_rows <= 1
                    || report.context != token_step.context
                    || report.device != token_step.device
                    || report.state_pair_count != token_step.state_pair_count
                    || report.logical_state_bytes != token_step.logical_state_bytes
                    || report.state_bank_count != token_step.state_bank_count
                    || report.state_device_bytes != token_step.state_device_bytes
            })
        {
            return Err(MetalScoreboardError::PlanMismatch);
        }
        let token_totals = validate_component(&token_step)?;
        let fixed_totals = fixed_prefill.as_ref().map(validate_component).transpose()?;
        let component_phase_totals = match &fixed_totals {
            Some(fixed) => token_totals.phase.checked_add(&fixed.phase)?,
            None => token_totals.phase.clone(),
        };
        let component_count = component_phase_totals.successful_invocation_count;
        if successful_runs.len()
            != usize::try_from(component_count).map_err(|_| MetalScoreboardError::Overflow)?
        {
            return Err(MetalScoreboardError::OutOfOrder {
                expected: component_count,
                actual: u64::try_from(successful_runs.len())
                    .map_err(|_| MetalScoreboardError::Overflow)?,
            });
        }

        let mut token_local = 0usize;
        let mut prefill_local = 0usize;
        let mut committed_state_position = 0usize;
        let mut committed_state_pair_count = 0usize;
        let mut committed_state_bytes = 0usize;
        let mut committed_state_work_items = 0usize;
        let mut prompt_prefill = PhaseAccumulator::default();
        let mut steady_decode = PhaseAccumulator::default();
        let mut standalone = PhaseAccumulator::default();
        for (index, invocation) in successful_runs.iter().enumerate() {
            let expected_global = u64::try_from(index)
                .map_err(|_| MetalScoreboardError::Overflow)?
                .checked_add(1)
                .ok_or(MetalScoreboardError::Overflow)?;
            if invocation.successful_invocation != expected_global
                || invocation.first_successful_run != (expected_global == 1)
            {
                return Err(MetalScoreboardError::OutOfOrder {
                    expected: expected_global,
                    actual: invocation.successful_invocation,
                });
            }
            if !phase_matches_program(invocation.phase, invocation.program) {
                return Err(MetalScoreboardError::PlanMismatch);
            }
            let (component, local_index) = match invocation.program {
                LlamaMetalScoreboardProgram::TokenStep => {
                    let index = token_local;
                    token_local = token_local
                        .checked_add(1)
                        .ok_or(MetalScoreboardError::Overflow)?;
                    (&token_step, index)
                }
                LlamaMetalScoreboardProgram::FixedPrefill => {
                    let component = fixed_prefill
                        .as_ref()
                        .ok_or(MetalScoreboardError::PlanMismatch)?;
                    let index = prefill_local;
                    prefill_local = prefill_local
                        .checked_add(1)
                        .ok_or(MetalScoreboardError::Overflow)?;
                    (component, index)
                }
            };
            let component_run = component.successful_runs.get(local_index).ok_or(
                MetalScoreboardError::OutOfOrder {
                    expected: invocation.program_successful_invocation,
                    actual: 0,
                },
            )?;
            let expected_local = u64::try_from(local_index)
                .map_err(|_| MetalScoreboardError::Overflow)?
                .checked_add(1)
                .ok_or(MetalScoreboardError::Overflow)?;
            let expected_position = committed_state_position
                .checked_add(component.append_span_rows)
                .ok_or(MetalScoreboardError::Overflow)?;
            if invocation.program_successful_invocation != expected_local
                || component_run.successful_invocation != expected_local
                || invocation.append_span_rows != component.append_span_rows
                || invocation.committed_state_pair_count != component_run.committed_state_pair_count
                || invocation.committed_state_bytes != component_run.committed_state_bytes
                || invocation.committed_state_work_items != component_run.committed_state_work_items
                || invocation.committed_state_position != expected_position
                || component_run.committed_state_position != Some(expected_position)
            {
                return Err(MetalScoreboardError::StateCommitMismatch);
            }
            match invocation.phase {
                LlamaMetalScoreboardPhase::PromptPrefill => {
                    prompt_prefill.add(invocation.append_span_rows, component_run)?
                }
                LlamaMetalScoreboardPhase::SteadyDecode => {
                    steady_decode.add(invocation.append_span_rows, component_run)?
                }
                LlamaMetalScoreboardPhase::Standalone => {
                    standalone.add(invocation.append_span_rows, component_run)?
                }
            }
            committed_state_position = expected_position;
            committed_state_pair_count = committed_state_pair_count
                .checked_add(invocation.committed_state_pair_count)
                .ok_or(MetalScoreboardError::Overflow)?;
            committed_state_bytes = committed_state_bytes
                .checked_add(invocation.committed_state_bytes)
                .ok_or(MetalScoreboardError::Overflow)?;
            committed_state_work_items = committed_state_work_items
                .checked_add(invocation.committed_state_work_items)
                .ok_or(MetalScoreboardError::Overflow)?;
        }
        if token_local != token_step.successful_runs.len()
            || fixed_prefill
                .as_ref()
                .is_some_and(|report| prefill_local != report.successful_runs.len())
        {
            return Err(MetalScoreboardError::StateCommitMismatch);
        }
        let prompt_prefill = prompt_prefill.finish();
        let steady_decode = steady_decode.finish();
        let standalone = standalone.finish();
        let phase_totals = prompt_prefill
            .checked_add(&steady_decode)?
            .checked_add(&standalone)?;
        if phase_totals != component_phase_totals {
            return Err(MetalScoreboardError::StateCommitMismatch);
        }
        let fallback_count = token_step
            .fallback_count
            .checked_add(
                fixed_prefill
                    .as_ref()
                    .map_or(0, |report| report.fallback_count),
            )
            .ok_or(MetalScoreboardError::Overflow)?;
        let component_pair_count = checked_component_state_total(
            token_totals.committed_state_pair_count,
            fixed_totals
                .as_ref()
                .map_or(0, |totals| totals.committed_state_pair_count),
        )?;
        let component_bytes = checked_component_state_total(
            token_totals.committed_state_bytes,
            fixed_totals
                .as_ref()
                .map_or(0, |totals| totals.committed_state_bytes),
        )?;
        let component_work_items = checked_component_state_total(
            token_totals.committed_state_work_items,
            fixed_totals
                .as_ref()
                .map_or(0, |totals| totals.committed_state_work_items),
        )?;
        if committed_state_pair_count != component_pair_count
            || committed_state_bytes != component_bytes
            || committed_state_work_items != component_work_items
            || fallback_count != 0
        {
            return Err(MetalScoreboardError::StateCommitMismatch);
        }
        Ok(Self {
            format_version: LLAMA_METAL_EXECUTION_SCOREBOARD_FORMAT_VERSION,
            context: token_step.context.clone(),
            token_step,
            fixed_prefill,
            successful_runs,
            successful_run_count: component_count,
            prompt_prefill,
            steady_decode,
            standalone,
            committed_state_position,
            committed_state_pair_count,
            committed_state_bytes,
            committed_state_work_items,
            fallback_count,
        })
    }

    /// Encodes this observation deterministically as versioned UTF-8 JSON.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, MetalScoreboardError> {
        let mut bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| MetalScoreboardError::Json(error.to_string()))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Writes the deterministic JSON representation to `path`.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), MetalScoreboardError> {
        fs::write(path, self.to_json_bytes()?).map_err(|error| MetalScoreboardError::Io {
            operation: "write Llama report",
            kind: error.kind(),
        })
    }
}

fn phase_matches_program(
    phase: LlamaMetalScoreboardPhase,
    program: LlamaMetalScoreboardProgram,
) -> bool {
    matches!(
        (phase, program),
        (
            LlamaMetalScoreboardPhase::PromptPrefill,
            LlamaMetalScoreboardProgram::FixedPrefill | LlamaMetalScoreboardProgram::TokenStep
        ) | (
            LlamaMetalScoreboardPhase::SteadyDecode | LlamaMetalScoreboardPhase::Standalone,
            LlamaMetalScoreboardProgram::TokenStep
        )
    )
}

fn tokens_per_second(token_count: usize, duration: Duration) -> Option<f64> {
    (token_count != 0 && !duration.is_zero()).then(|| token_count as f64 / duration.as_secs_f64())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidatedComponent {
    phase: LlamaMetalScoreboardPhaseAggregate,
    committed_state_pair_count: usize,
    committed_state_bytes: usize,
    committed_state_work_items: usize,
}

fn validate_component(
    report: &MetalSessionScoreboardReport,
) -> Result<ValidatedComponent, MetalScoreboardError> {
    if report.fallback_count != 0 {
        return Err(MetalScoreboardError::PlanMismatch);
    }
    let run_count =
        u64::try_from(report.successful_runs.len()).map_err(|_| MetalScoreboardError::Overflow)?;
    if report.successful_run_count != run_count {
        return Err(MetalScoreboardError::OutOfOrder {
            expected: run_count,
            actual: report.successful_run_count,
        });
    }
    if report.first_run_host_wall_time
        != report.successful_runs.first().map(|run| run.run_wall_time)
        || report.steady_run_host_wall_times
            != report
                .successful_runs
                .iter()
                .skip(1)
                .map(|run| run.run_wall_time)
                .collect::<Vec<_>>()
        || report.first_synchronous_transaction_host_wall_time
            != report
                .successful_runs
                .first()
                .map(|run| run.synchronous_transaction_wall_time)
        || report.steady_synchronous_transaction_host_wall_times
            != report
                .successful_runs
                .iter()
                .skip(1)
                .map(|run| run.synchronous_transaction_wall_time)
                .collect::<Vec<_>>()
    {
        return Err(MetalScoreboardError::StateCommitMismatch);
    }

    let mut accumulator = PhaseAccumulator::default();
    let mut committed_state_pair_count = 0usize;
    let mut committed_state_bytes = 0usize;
    let mut committed_state_work_items = 0usize;
    let mut terminal_position = 0usize;
    for (index, run) in report.successful_runs.iter().enumerate() {
        let expected = u64::try_from(index)
            .map_err(|_| MetalScoreboardError::Overflow)?
            .checked_add(1)
            .ok_or(MetalScoreboardError::Overflow)?;
        if run.successful_invocation != expected || run.first_successful_run != (expected == 1) {
            return Err(MetalScoreboardError::OutOfOrder {
                expected,
                actual: run.successful_invocation,
            });
        }
        if run.committed_state_pair_count != report.state_pair_count
            || run.committed_state_bytes != report.append_state_row_bytes
            || run.committed_state_work_items != report.append_state_work_items
        {
            return Err(MetalScoreboardError::StateCommitMismatch);
        }
        terminal_position = run
            .committed_state_position
            .ok_or(MetalScoreboardError::StateCommitMismatch)?;
        accumulator.add(report.append_span_rows, run)?;
        committed_state_pair_count = committed_state_pair_count
            .checked_add(run.committed_state_pair_count)
            .ok_or(MetalScoreboardError::Overflow)?;
        committed_state_bytes = committed_state_bytes
            .checked_add(run.committed_state_bytes)
            .ok_or(MetalScoreboardError::Overflow)?;
        committed_state_work_items = committed_state_work_items
            .checked_add(run.committed_state_work_items)
            .ok_or(MetalScoreboardError::Overflow)?;
    }
    let phase = accumulator.finish();
    if report.committed_state_position != Some(terminal_position)
        || report.committed_state_pair_count != committed_state_pair_count
        || report.committed_state_bytes != committed_state_bytes
        || report.committed_state_work_items != committed_state_work_items
        || report.transient_host_api_h2d_calls != phase.transient_host_api_h2d_calls
        || report.transient_host_api_h2d_bytes != phase.transient_host_api_h2d_bytes
        || report.runtime_control_host_api_h2d_calls != phase.runtime_control_host_api_h2d_calls
        || report.runtime_control_host_api_h2d_bytes != phase.runtime_control_host_api_h2d_bytes
        || report.retained_host_api_d2h_calls != phase.retained_host_api_d2h_calls
        || report.retained_host_api_d2h_bytes != phase.retained_host_api_d2h_bytes
        || report.kernel_launch_count != phase.kernel_launch_count
        || report.command_submission_count != phase.command_submission_count
        || report.command_wait_count != phase.command_wait_count
        || report.gpu_command_execution_time != phase.gpu_command_execution_time
    {
        return Err(MetalScoreboardError::StateCommitMismatch);
    }
    Ok(ValidatedComponent {
        phase,
        committed_state_pair_count,
        committed_state_bytes,
        committed_state_work_items,
    })
}

fn checked_component_state_total(left: usize, right: usize) -> Result<usize, MetalScoreboardError> {
    left.checked_add(right)
        .ok_or(MetalScoreboardError::Overflow)
}

#[derive(Default)]
struct PhaseAccumulator {
    committed_token_count: usize,
    successful_invocation_count: u64,
    host_run_wall_time: Duration,
    host_synchronous_transaction_wall_time: Duration,
    gpu_command_execution_time: Duration,
    gpu_timing_unavailable: bool,
    kernel_launch_count: usize,
    command_submission_count: usize,
    command_wait_count: usize,
    transient_host_api_h2d_calls: usize,
    transient_host_api_h2d_bytes: usize,
    runtime_control_host_api_h2d_calls: usize,
    runtime_control_host_api_h2d_bytes: usize,
    retained_host_api_d2h_calls: usize,
    retained_host_api_d2h_bytes: usize,
}

impl PhaseAccumulator {
    fn add(
        &mut self,
        append_span_rows: usize,
        run: &MetalScoreboardRun,
    ) -> Result<(), MetalScoreboardError> {
        macro_rules! checked_add {
            ($field:ident, $value:expr) => {
                self.$field = self
                    .$field
                    .checked_add($value)
                    .ok_or(MetalScoreboardError::Overflow)?;
            };
        }
        checked_add!(committed_token_count, append_span_rows);
        checked_add!(successful_invocation_count, 1);
        checked_add!(host_run_wall_time, run.run_wall_time);
        checked_add!(
            host_synchronous_transaction_wall_time,
            run.synchronous_transaction_wall_time
        );
        checked_add!(kernel_launch_count, run.kernel_launch_count);
        checked_add!(command_submission_count, run.command_submission_count);
        checked_add!(command_wait_count, run.command_wait_count);
        checked_add!(
            transient_host_api_h2d_calls,
            run.transient_host_api_h2d_calls
        );
        checked_add!(
            transient_host_api_h2d_bytes,
            run.transient_host_api_h2d_bytes
        );
        checked_add!(
            runtime_control_host_api_h2d_calls,
            run.runtime_control_host_api_h2d_calls
        );
        checked_add!(
            runtime_control_host_api_h2d_bytes,
            run.runtime_control_host_api_h2d_bytes
        );
        checked_add!(retained_host_api_d2h_calls, run.retained_host_api_d2h_calls);
        checked_add!(retained_host_api_d2h_bytes, run.retained_host_api_d2h_bytes);
        if let Some(duration) = run.gpu_command_execution_time {
            checked_add!(gpu_command_execution_time, duration);
        } else {
            self.gpu_timing_unavailable = true;
        }
        Ok(())
    }

    fn finish(self) -> LlamaMetalScoreboardPhaseAggregate {
        LlamaMetalScoreboardPhaseAggregate {
            committed_token_count: self.committed_token_count,
            successful_invocation_count: self.successful_invocation_count,
            host_run_wall_time: self.host_run_wall_time,
            host_synchronous_transaction_wall_time: self.host_synchronous_transaction_wall_time,
            gpu_command_execution_time: (self.successful_invocation_count != 0
                && !self.gpu_timing_unavailable)
                .then_some(self.gpu_command_execution_time),
            kernel_launch_count: self.kernel_launch_count,
            command_submission_count: self.command_submission_count,
            command_wait_count: self.command_wait_count,
            transient_host_api_h2d_calls: self.transient_host_api_h2d_calls,
            transient_host_api_h2d_bytes: self.transient_host_api_h2d_bytes,
            runtime_control_host_api_h2d_calls: self.runtime_control_host_api_h2d_calls,
            runtime_control_host_api_h2d_bytes: self.runtime_control_host_api_h2d_bytes,
            retained_host_api_d2h_calls: self.retained_host_api_d2h_calls,
            retained_host_api_d2h_bytes: self.retained_host_api_d2h_bytes,
        }
    }
}

impl LlamaMetalScoreboardPhaseAggregate {
    fn checked_add(&self, other: &Self) -> Result<Self, MetalScoreboardError> {
        macro_rules! checked_sum {
            ($field:ident) => {
                self.$field
                    .checked_add(other.$field)
                    .ok_or(MetalScoreboardError::Overflow)?
            };
        }
        let successful_invocation_count = checked_sum!(successful_invocation_count);
        let gpu_command_execution_time = match (
            self.successful_invocation_count,
            self.gpu_command_execution_time,
            other.successful_invocation_count,
            other.gpu_command_execution_time,
        ) {
            (0, _, 0, _) => None,
            (0, _, _, right) => right,
            (_, left, 0, _) => left,
            (_, Some(left), _, Some(right)) => Some(
                left.checked_add(right)
                    .ok_or(MetalScoreboardError::Overflow)?,
            ),
            _ => None,
        };
        Ok(Self {
            committed_token_count: checked_sum!(committed_token_count),
            successful_invocation_count,
            host_run_wall_time: checked_sum!(host_run_wall_time),
            host_synchronous_transaction_wall_time: checked_sum!(
                host_synchronous_transaction_wall_time
            ),
            gpu_command_execution_time,
            kernel_launch_count: checked_sum!(kernel_launch_count),
            command_submission_count: checked_sum!(command_submission_count),
            command_wait_count: checked_sum!(command_wait_count),
            transient_host_api_h2d_calls: checked_sum!(transient_host_api_h2d_calls),
            transient_host_api_h2d_bytes: checked_sum!(transient_host_api_h2d_bytes),
            runtime_control_host_api_h2d_calls: checked_sum!(runtime_control_host_api_h2d_calls),
            runtime_control_host_api_h2d_bytes: checked_sum!(runtime_control_host_api_h2d_bytes),
            retained_host_api_d2h_calls: checked_sum!(retained_host_api_d2h_calls),
            retained_host_api_d2h_bytes: checked_sum!(retained_host_api_d2h_bytes),
        })
    }
}
