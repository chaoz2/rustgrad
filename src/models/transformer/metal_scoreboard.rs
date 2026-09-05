//! Ordered scoreboard evidence for a two-program Metal Llama workload.

use crate::runtime::metal::{
    MetalDeviceRunReport, MetalScoreboardContext, MetalScoreboardError,
    MetalSessionScoreboardReport,
};
use serde::Serialize;
use std::{fs, path::Path};

/// Current deterministic JSON schema for [`LlamaMetalExecutionScoreboardReport`].
pub const LLAMA_METAL_EXECUTION_SCOREBOARD_FORMAT_VERSION: u32 = 1;

/// Physical program that committed one Llama workload invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaMetalScoreboardProgram {
    /// Fixed-span state-only prompt ingestion.
    FixedPrefill,
    /// Single-token execution, including prompt tails and decode.
    TokenStep,
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
        program_successful_invocation: u64,
        append_span_rows: usize,
        report: &MetalDeviceRunReport,
    ) -> Result<Self, MetalScoreboardError> {
        Ok(Self {
            successful_invocation: report.successful_invocation,
            first_successful_run: report.first_successful_run,
            program,
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
            || token_step.append_span_rows != 1
            || fixed_prefill.as_ref().is_some_and(|report| {
                report.state_policy != crate::runtime::metal::MetalScoreboardStatePolicy::Append
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
        let component_count = token_step
            .successful_run_count
            .checked_add(
                fixed_prefill
                    .as_ref()
                    .map_or(0, |report| report.successful_run_count),
            )
            .ok_or(MetalScoreboardError::Overflow)?;
        if token_step.successful_runs.len()
            != usize::try_from(token_step.successful_run_count)
                .map_err(|_| MetalScoreboardError::Overflow)?
            || fixed_prefill.as_ref().is_some_and(|report| {
                usize::try_from(report.successful_run_count) != Ok(report.successful_runs.len())
            })
            || successful_runs.len()
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
        let fallback_count = token_step
            .fallback_count
            .checked_add(
                fixed_prefill
                    .as_ref()
                    .map_or(0, |report| report.fallback_count),
            )
            .ok_or(MetalScoreboardError::Overflow)?;
        let component_pair_count = token_step
            .committed_state_pair_count
            .checked_add(
                fixed_prefill
                    .as_ref()
                    .map_or(0, |report| report.committed_state_pair_count),
            )
            .ok_or(MetalScoreboardError::Overflow)?;
        let component_bytes = token_step
            .committed_state_bytes
            .checked_add(
                fixed_prefill
                    .as_ref()
                    .map_or(0, |report| report.committed_state_bytes),
            )
            .ok_or(MetalScoreboardError::Overflow)?;
        let component_work_items = token_step
            .committed_state_work_items
            .checked_add(
                fixed_prefill
                    .as_ref()
                    .map_or(0, |report| report.committed_state_work_items),
            )
            .ok_or(MetalScoreboardError::Overflow)?;
        if committed_state_pair_count != component_pair_count
            || committed_state_bytes != component_bytes
            || committed_state_work_items != component_work_items
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
