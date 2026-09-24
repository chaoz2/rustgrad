mod validation;

use super::compile_phase::NativeTrainingCompilePhaseReport;
use super::compiler_evidence::{
    NativeTrainingCompilerCriticalTail, NativeTrainingCompilerProcessTiming,
    NativeTrainingModuleOverlap, NativeTrainingProgramPairOverlap, NativeTrainingTranslationUnit,
};
use super::program_report::NativeTrainingProgramReport;
use super::step_phases::NativeTrainingStepPhaseReport;
use super::{
    NativeCpuReplayTraffic, invalid, latency_summary, sum_durations, validate_total_duration,
};
use crate::{BenchmarkDuration, BenchmarkLatencySummary, BenchmarkTransfer, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CheckpointReport {
    pub(super) capture_identity: u64,
    pub(super) replay_step: u64,
    pub(super) byte_count: u64,
    pub(super) wall_time: BenchmarkDuration,
}

/// First-replay and bounded steady-replay wall time for one measured portion
/// of successful strict-native CPU training-step replay.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReplayTiming {
    pub(super) first: BenchmarkDuration,
    pub(super) steady_total: BenchmarkDuration,
    pub(super) steady: BenchmarkLatencySummary,
}

impl NativeTrainingReplayTiming {
    pub const fn first(&self) -> BenchmarkDuration {
        self.first
    }

    pub const fn steady_total(&self) -> BenchmarkDuration {
        self.steady_total
    }

    pub const fn steady(&self) -> &BenchmarkLatencySummary {
        &self.steady
    }

    pub(super) fn from_durations(first: Duration, steady: &[Duration]) -> Result<Self> {
        Ok(Self {
            first: BenchmarkDuration::from_duration(first),
            steady_total: sum_durations(steady)?,
            steady: latency_summary(steady)?,
        })
    }

    fn validate(&self, steady_sample_count: u64) -> Result<()> {
        self.first
            .to_duration()
            .map_err(|_| invalid("invalid native training phase duration"))?;
        if self.steady.sample_count != steady_sample_count
            || self.steady.min > self.steady.nearest_rank_p50
            || self.steady.nearest_rank_p50 > self.steady.nearest_rank_p95
            || self.steady.nearest_rank_p95 > self.steady.max
            || self.steady_total < self.steady.max
        {
            return Err(invalid("invalid native training phase summary"));
        }
        for duration in [
            self.steady_total,
            self.steady.min,
            self.steady.nearest_rank_p50,
            self.steady.nearest_rank_p95,
            self.steady.max,
        ] {
            duration
                .to_duration()
                .map_err(|_| invalid("invalid native training phase duration"))?;
        }
        validate_total_duration(&self.steady, self.steady_total)
    }
}

/// Versioned strict-native CPU compiled-training observation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReport {
    pub(super) format_version: u32,
    pub(super) compile_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) compile_phases: Option<NativeTrainingCompilePhaseReport>,
    pub(super) prepare_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_runtime_overhead_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_parallel_module_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_parallel_render_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_max_parallel_render_job_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_render_capsule_hit_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_render_capsule_miss_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_local_render_job_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_compiler_process_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_max_parallel_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_compiler_process_timings: Option<Vec<NativeTrainingCompilerProcessTiming>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_compiler_critical_tail: Option<NativeTrainingCompilerCriticalTail>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_module_overlaps: Option<Vec<NativeTrainingModuleOverlap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_program_pair_overlaps: Option<Vec<NativeTrainingProgramPairOverlap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepare_translation_units: Option<Vec<NativeTrainingTranslationUnit>>,
    pub(super) initial_replay_step: u64,
    pub(super) successful_replay_count: u64,
    pub(super) main: NativeTrainingProgramReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) accumulation: Option<NativeTrainingProgramReport>,
    pub(super) partial_flush: Option<NativeTrainingProgramReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) zero_grad: Option<NativeTrainingProgramReport>,
    pub(super) evaluation: Option<NativeTrainingProgramReport>,
    pub(super) recurrent_logical_state_count: u64,
    pub(super) recurrent_logical_state_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) main_replay_traffic: Option<NativeCpuReplayTraffic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) main_replay_executed_native_item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) accumulation_replay_traffic: Option<NativeCpuReplayTraffic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) accumulation_replay_executed_native_item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) main_replay_executor_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) main_replay_native_dispatcher_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) main_replay_executor_host_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) main_replay_recurrent_overhead_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) step_phases: Option<NativeTrainingStepPhaseReport>,
    pub(super) first_replay_wall_time: BenchmarkDuration,
    pub(super) steady_replay_total_wall_time: BenchmarkDuration,
    pub(super) steady_replay_wall_time: BenchmarkLatencySummary,
    pub(super) steady_microbatches_per_second: Option<f64>,
    pub(super) schedule_cache_keys: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) accumulation_schedule_cache_keys: Vec<u64>,
    pub(super) checkpoint: Option<CheckpointReport>,
    pub(super) fallback_count: u64,
    pub(super) kernel_launch_count: Option<u64>,
    pub(super) host_to_device: Option<BenchmarkTransfer>,
    pub(super) device_to_host: Option<BenchmarkTransfer>,
    pub(super) measured_peak_host_memory_bytes: Option<u64>,
}

impl NativeTrainingReport {
    pub const fn compile_wall_time(&self) -> BenchmarkDuration {
        self.compile_wall_time
    }

    pub fn compile_phases(&self) -> Option<&NativeTrainingCompilePhaseReport> {
        self.compile_phases.as_ref()
    }

    pub const fn prepare_wall_time(&self) -> BenchmarkDuration {
        self.prepare_wall_time
    }

    /// Whole-prepare time outside the attached native program preparations.
    pub const fn prepare_runtime_overhead_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_runtime_overhead_wall_time
    }

    /// Exact overlap among complete native module compiler/loader jobs.
    pub const fn prepare_parallel_module_overlap_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_parallel_module_overlap_wall_time
    }

    /// Exact overlap among immutable per-program native render jobs.
    pub const fn prepare_parallel_render_overlap_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_parallel_render_overlap_wall_time
    }

    /// Observed maximum concurrency of immutable per-program native renders.
    pub const fn prepare_max_parallel_render_job_count(&self) -> Option<u64> {
        self.prepare_max_parallel_render_job_count
    }

    pub const fn prepare_render_capsule_hit_count(&self) -> Option<u64> {
        self.prepare_render_capsule_hit_count
    }

    pub const fn prepare_render_capsule_miss_count(&self) -> Option<u64> {
        self.prepare_render_capsule_miss_count
    }

    pub const fn prepare_local_render_job_count(&self) -> Option<u64> {
        self.prepare_local_render_job_count
    }

    pub const fn prepare_compiler_process_overlap_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_compiler_process_overlap_wall_time
    }

    pub const fn prepare_compiler_process_count(&self) -> Option<u64> {
        self.prepare_compiler_process_count
    }

    pub const fn prepare_max_parallel_compiler_process_count(&self) -> Option<u64> {
        self.prepare_max_parallel_compiler_process_count
    }

    pub const fn main(&self) -> &NativeTrainingProgramReport {
        &self.main
    }

    pub const fn accumulation(&self) -> Option<&NativeTrainingProgramReport> {
        self.accumulation.as_ref()
    }

    pub const fn partial_flush(&self) -> Option<&NativeTrainingProgramReport> {
        self.partial_flush.as_ref()
    }

    pub const fn evaluation(&self) -> Option<&NativeTrainingProgramReport> {
        self.evaluation.as_ref()
    }

    pub const fn zero_grad(&self) -> Option<&NativeTrainingProgramReport> {
        self.zero_grad.as_ref()
    }

    pub const fn successful_replay_count(&self) -> u64 {
        self.successful_replay_count
    }

    pub const fn steady_replay_wall_time(&self) -> &BenchmarkLatencySummary {
        &self.steady_replay_wall_time
    }

    pub const fn steady_replay_total_wall_time(&self) -> BenchmarkDuration {
        self.steady_replay_total_wall_time
    }

    pub const fn first_replay_wall_time(&self) -> BenchmarkDuration {
        self.first_replay_wall_time
    }

    pub fn schedule_cache_keys(&self) -> &[u64] {
        &self.schedule_cache_keys
    }

    pub const fn steady_microbatches_per_second(&self) -> Option<f64> {
        self.steady_microbatches_per_second
    }

    pub const fn recurrent_state_count(&self) -> u64 {
        self.recurrent_logical_state_count
    }

    pub const fn recurrent_state_bytes(&self) -> u64 {
        self.recurrent_logical_state_bytes
    }

    pub const fn main_replay_traffic(&self) -> Option<&NativeCpuReplayTraffic> {
        self.main_replay_traffic.as_ref()
    }

    /// Stable number of prepared CPU JIT items actually invoked by each
    /// successfully published optimizer-commit replay.
    pub const fn main_replay_executed_native_item_count(&self) -> Option<u64> {
        self.main_replay_executed_native_item_count
    }

    pub const fn accumulation_replay_traffic(&self) -> Option<&NativeCpuReplayTraffic> {
        self.accumulation_replay_traffic.as_ref()
    }

    pub const fn accumulation_replay_executed_native_item_count(&self) -> Option<u64> {
        self.accumulation_replay_executed_native_item_count
    }

    pub fn accumulation_schedule_cache_keys(&self) -> &[u64] {
        &self.accumulation_schedule_cache_keys
    }

    /// Wall time inside the sealed native executor for successful training
    /// steps across their authenticated phase-specific programs.
    pub const fn main_replay_executor_wall_time(&self) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_executor_wall_time.as_ref()
    }

    /// Time inside authenticated native schedule-module dispatcher calls.
    pub const fn main_replay_native_dispatcher_wall_time(
        &self,
    ) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_native_dispatcher_wall_time.as_ref()
    }

    /// Checked executor remainder outside native dispatcher calls, including
    /// workspace work and any conservative per-entry execution.
    pub const fn main_replay_executor_host_wall_time(&self) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_executor_host_wall_time.as_ref()
    }

    /// Checked end-to-end remainder outside the sealed native executor across
    /// the authenticated phase-specific programs. This covers recurrent
    /// staging, validation, and atomic commit/publication.
    pub const fn main_replay_recurrent_overhead_wall_time(
        &self,
    ) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_recurrent_overhead_wall_time.as_ref()
    }

    /// First replay and warm replay timings classified solely by whether the
    /// successful training step committed an optimizer update.
    pub const fn step_phases(&self) -> Option<&NativeTrainingStepPhaseReport> {
        self.step_phases.as_ref()
    }

    pub const fn checkpoint_byte_count(&self) -> Option<u64> {
        match &self.checkpoint {
            Some(checkpoint) => Some(checkpoint.byte_count),
            None => None,
        }
    }

    pub const fn fallback_count(&self) -> u64 {
        self.fallback_count
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| invalid(format!("JSON encoding failed: {error}")))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let report: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("JSON decoding failed: {error}")))?;
        report.validate()?;
        Ok(report)
    }

    pub(super) fn validate(&self) -> Result<()> {
        validation::validate(self)
    }
}
