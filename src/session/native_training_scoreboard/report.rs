use super::compile_phase::NativeTrainingCompilePhaseReport;
use super::compiler_evidence::{
    CompilerProcessValidationContext, NativeTrainingCompilerCriticalTail,
    NativeTrainingCompilerProcessTiming, NativeTrainingModuleOverlap,
    NativeTrainingProgramPairOverlap, NativeTrainingTranslationUnit,
    validate_compiler_process_evidence, validate_module_overlaps, validate_program_pair_overlaps,
    validate_translation_units,
};
use super::program_report::{
    NativeCompilerAggregate, NativeRenderAggregate, NativeTrainingProgramReport,
};
use super::step_phases::{NativeTrainingStepPhaseReport, ReplayTimingPartition};
use super::{
    MAX_REPLAY_SAMPLES, NATIVE_TRAINING_REPORT_FORMAT_V2, NATIVE_TRAINING_REPORT_FORMAT_V3,
    NATIVE_TRAINING_REPORT_FORMAT_V4, NATIVE_TRAINING_REPORT_FORMAT_V5,
    NATIVE_TRAINING_REPORT_FORMAT_V6, NATIVE_TRAINING_REPORT_FORMAT_V7,
    NATIVE_TRAINING_REPORT_FORMAT_V8, NATIVE_TRAINING_REPORT_FORMAT_V9,
    NATIVE_TRAINING_REPORT_FORMAT_V10, NATIVE_TRAINING_REPORT_FORMAT_V11,
    NATIVE_TRAINING_REPORT_FORMAT_V12, NATIVE_TRAINING_REPORT_FORMAT_V13,
    NATIVE_TRAINING_REPORT_FORMAT_V14, NATIVE_TRAINING_REPORT_FORMAT_V15,
    NATIVE_TRAINING_REPORT_FORMAT_V16, NATIVE_TRAINING_REPORT_FORMAT_V17,
    NATIVE_TRAINING_REPORT_FORMAT_V18, NATIVE_TRAINING_REPORT_FORMAT_V19,
    NATIVE_TRAINING_REPORT_FORMAT_V20, NATIVE_TRAINING_REPORT_FORMAT_V21,
    NATIVE_TRAINING_REPORT_FORMAT_V22, NATIVE_TRAINING_REPORT_FORMAT_VERSION,
    NativeCpuReplayTraffic, count, invalid, latency_summary, rate_from_total, sum_durations,
    validate_phase_partition, validate_total_duration,
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
        if !matches!(
            self.format_version,
            1 | NATIVE_TRAINING_REPORT_FORMAT_V2
                | NATIVE_TRAINING_REPORT_FORMAT_V3
                | NATIVE_TRAINING_REPORT_FORMAT_V4
                | NATIVE_TRAINING_REPORT_FORMAT_V5
                | NATIVE_TRAINING_REPORT_FORMAT_V6
                | NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_V9
                | NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11
                | NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION
        ) {
            return Err(invalid("unsupported native training report version"));
        }
        if self.format_version == 1 && self.zero_grad.is_some() {
            return Err(invalid(
                "legacy native training report has zero-grad evidence",
            ));
        }
        if self.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V2
            && self.partial_flush.is_some() != self.zero_grad.is_some()
        {
            return Err(invalid(
                "native training auxiliary program inventory differs",
            ));
        }
        for duration in [
            self.compile_wall_time,
            self.prepare_wall_time,
            self.first_replay_wall_time,
            self.steady_replay_total_wall_time,
            self.steady_replay_wall_time.min,
            self.steady_replay_wall_time.nearest_rank_p50,
            self.steady_replay_wall_time.nearest_rank_p95,
            self.steady_replay_wall_time.max,
        ] {
            duration
                .to_duration()
                .map_err(|_| invalid("invalid native training duration"))?;
        }
        match (self.format_version, &self.compile_phases) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V22, None) => {}
            (NATIVE_TRAINING_REPORT_FORMAT_VERSION, Some(phases)) => phases.validate(
                self.compile_wall_time,
                &self.main,
                self.accumulation.as_ref(),
                self.partial_flush.as_ref(),
                self.zero_grad.as_ref(),
                self.evaluation.as_ref(),
            )?,
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V22, Some(_)) => {
                return Err(invalid("legacy native training report has compile phases"));
            }
            _ => return Err(invalid("native training compile phases are absent")),
        }
        self.main.validate(self.format_version)?;
        if self.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V14
            && (self.main.shared_prefix_entry_count() != 0
                || self.main.shared_prefix_source_program_index.is_some()
                || self.main.shared_prefix_source_native_identity.is_some())
        {
            return Err(invalid(
                "native main program cannot reference an earlier prefix module",
            ));
        }
        match (
            self.format_version,
            &self.accumulation,
            &self.accumulation_replay_traffic,
            self.accumulation_replay_executed_native_item_count,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V10, None, None, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                None,
                None,
                None,
            ) => {}
            (NATIVE_TRAINING_REPORT_FORMAT_V11, Some(program), Some(traffic), Some(executed))
                if traffic.borrowed_recurrent_input_bytes()
                    == self.recurrent_logical_state_bytes
                    && traffic.borrowed_recurrent_output_bytes()
                        == self.recurrent_logical_state_bytes
                    && traffic.retained_recurrent_state_count() == 0
                    && traffic.retained_recurrent_state_bytes() == 0
                    && traffic.replaced_recurrent_state_count() == 0
                    && traffic.replaced_recurrent_state_bytes() == 0
                    && executed <= program.rendered_entry_count
                    && program.capture_identity != self.main.capture_identity =>
            {
                program.validate(self.format_version)?;
                if program.vectorized != self.main.vectorized {
                    return Err(invalid("native program vectorization policy differs"));
                }
            }
            (
                NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(program),
                Some(traffic),
                Some(executed),
            ) if traffic.borrowed_recurrent_input_bytes() == self.recurrent_logical_state_bytes
                && traffic
                    .borrowed_recurrent_output_bytes()
                    .checked_add(traffic.retained_recurrent_state_bytes())
                    == Some(self.recurrent_logical_state_bytes)
                && traffic.retained_recurrent_state_count()
                    <= self.recurrent_logical_state_count
                && ((traffic.retained_recurrent_state_count() == 0
                    && traffic.retained_recurrent_state_bytes() == 0
                    && traffic.replaced_recurrent_state_count() == 0
                    && traffic.replaced_recurrent_state_bytes() == 0)
                    || (traffic
                        .retained_recurrent_state_count()
                        .checked_add(traffic.replaced_recurrent_state_count())
                        == Some(self.recurrent_logical_state_count)
                        && traffic.replaced_recurrent_state_bytes()
                            == traffic.borrowed_recurrent_output_bytes()))
                && executed <= program.rendered_entry_count
                && program.capture_identity != self.main.capture_identity =>
            {
                program.validate(self.format_version)?;
                if program.vectorized != self.main.vectorized {
                    return Err(invalid("native program vectorization policy differs"));
                }
            }
            _ => return Err(invalid("native accumulation replay inventory differs")),
        }
        match (self.format_version, &self.main_replay_traffic) {
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V3
                | NATIVE_TRAINING_REPORT_FORMAT_V4
                | NATIVE_TRAINING_REPORT_FORMAT_V5
                | NATIVE_TRAINING_REPORT_FORMAT_V6
                | NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_V9
                | NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11,
                Some(traffic),
            ) if traffic.borrowed_recurrent_input_bytes() == self.recurrent_logical_state_bytes
                && traffic.borrowed_recurrent_output_bytes()
                    == self.recurrent_logical_state_bytes
                && traffic.retained_recurrent_state_count() == 0
                && traffic.retained_recurrent_state_bytes() == 0
                && traffic.replaced_recurrent_state_count() == 0
                && traffic.replaced_recurrent_state_bytes() == 0 => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(traffic),
            ) if traffic.borrowed_recurrent_input_bytes() == self.recurrent_logical_state_bytes
                && traffic.borrowed_recurrent_output_bytes()
                    == self.recurrent_logical_state_bytes
                && traffic.retained_recurrent_state_count() == 0
                && traffic.retained_recurrent_state_bytes() == 0
                && traffic.replaced_recurrent_state_count() == 0
                && traffic.replaced_recurrent_state_bytes() == 0 => {}
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, Some(_)) => {
                return Err(invalid("legacy native training report has replay traffic"));
            }
            _ => return Err(invalid("native training replay traffic differs")),
        }
        for traffic in self
            .main_replay_traffic
            .iter()
            .chain(&self.accumulation_replay_traffic)
        {
            match self.format_version {
                1..=NATIVE_TRAINING_REPORT_FORMAT_V12
                    if traffic.materialized_egress_count() == 0
                        && traffic.materialized_egress_bytes() == 0 => {}
                NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION
                    if traffic.materialized_egress_count() != 0
                        && traffic.materialized_egress_bytes() != 0 => {}
                1..=NATIVE_TRAINING_REPORT_FORMAT_V12 => {
                    return Err(invalid("legacy native report has CPU egress evidence"));
                }
                NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION => {
                    return Err(invalid("native CPU egress evidence is absent"));
                }
                _ => unreachable!("format version was validated"),
            }
        }
        match (
            self.format_version,
            self.main_replay_executed_native_item_count,
        ) {
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_V3, None) => {}
            (NATIVE_TRAINING_REPORT_FORMAT_V4, Some(executed))
                if executed <= self.main.native_item_count => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V5
                | NATIVE_TRAINING_REPORT_FORMAT_V6
                | NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_V9
                | NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11,
                Some(executed),
            ) if executed <= self.main.rendered_entry_count => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(executed),
            ) if executed <= self.main.rendered_entry_count => {}
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_V3, Some(_)) => {
                return Err(invalid("legacy native training report has execution count"));
            }
            _ => return Err(invalid("native training execution count differs")),
        }
        let mut prior_programs = vec![&self.main];
        for program in self
            .accumulation
            .iter()
            .chain(&self.partial_flush)
            .chain(&self.zero_grad)
            .chain(&self.evaluation)
        {
            program.validate(self.format_version)?;
            if program.vectorized != self.main.vectorized {
                return Err(invalid("native program vectorization policy differs"));
            }
            if let Some(source_index) = program.shared_prefix_source_program_index {
                let source_index = usize::try_from(source_index)
                    .map_err(|_| invalid("native shared-prefix source index overflows"))?;
                let source = prior_programs.get(source_index).copied().ok_or_else(|| {
                    invalid("native shared-prefix source index is not an earlier program")
                })?;
                if program.shared_prefix_source_native_identity != Some(source.native_identity) {
                    return Err(invalid(
                        "native shared-prefix source identity differs from its program index",
                    ));
                }
                if source.rendered_entry_count == 0
                    || source.shared_prefix_entry_count() != 0
                    || program.shared_prefix_entry_count() > source.rendered_entry_count
                {
                    return Err(invalid(
                        "native shared prefix exceeds its source program inventory",
                    ));
                }
            }
            prior_programs.push(program);
        }
        match (self.format_version, &self.prepare_module_overlaps) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(overlaps),
            ) => {
                validate_module_overlaps(overlaps, &prior_programs)?;
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, Some(_)) => {
                return Err(invalid("legacy native report has module overlap evidence"));
            }
            _ => return Err(invalid("native module overlap evidence is absent")),
        }
        match (
            self.format_version,
            &self.prepare_program_pair_overlaps,
            &self.prepare_translation_units,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V20, None, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(overlaps),
                Some(translation_units),
            ) => {
                validate_program_pair_overlaps(
                    overlaps,
                    &prior_programs,
                    self.prepare_module_overlaps
                        .as_deref()
                        .ok_or_else(|| invalid("native main overlap evidence is absent"))?,
                )?;
                validate_translation_units(
                    translation_units,
                    &prior_programs,
                    self.prepare_compiler_process_timings
                        .as_deref()
                        .ok_or_else(|| invalid("native compiler process evidence is absent"))?,
                )?;
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V20, _, _) => {
                return Err(invalid("legacy native report has v21 module evidence"));
            }
            _ => return Err(invalid("native v21 module evidence is absent")),
        }
        let parallel_evidence = match self.format_version {
            1..=NATIVE_TRAINING_REPORT_FORMAT_V7 => {
                if self.prepare_compiler_process_overlap_wall_time.is_some()
                    || self.prepare_compiler_process_count.is_some()
                    || self.prepare_max_parallel_compiler_process_count.is_some()
                {
                    return Err(invalid(
                        "legacy native report has parallel compiler evidence",
                    ));
                }
                None
            }
            NATIVE_TRAINING_REPORT_FORMAT_V8
            | NATIVE_TRAINING_REPORT_FORMAT_V9
            | NATIVE_TRAINING_REPORT_FORMAT_V10
            | NATIVE_TRAINING_REPORT_FORMAT_V11
            | NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION => {
                let compiler_overlap = self
                    .prepare_compiler_process_overlap_wall_time
                    .ok_or_else(|| invalid("native compiler overlap timing is absent"))?
                    .as_nanos()
                    .map_err(|_| invalid("invalid native compiler overlap duration"))?;
                let compiler_count = self
                    .prepare_compiler_process_count
                    .ok_or_else(|| invalid("native compiler process count is absent"))?;
                let max_parallel = self
                    .prepare_max_parallel_compiler_process_count
                    .ok_or_else(|| invalid("native compiler concurrency is absent"))?;
                let aggregate = NativeCompilerAggregate::from_programs(
                    std::iter::once(&self.main)
                        .chain(self.accumulation.iter())
                        .chain(self.partial_flush.iter())
                        .chain(&self.zero_grad)
                        .chain(&self.evaluation),
                    self.format_version,
                )?;
                let internal_compiler_overlap = aggregate.internal_overlap()?;
                let overlap_is_valid = if self.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V15 {
                    let active_union = aggregate
                        .cumulative_wall_sum
                        .checked_sub(compiler_overlap)
                        .ok_or_else(|| {
                            invalid("native compiler overlap exceeds cumulative work")
                        })?;
                    compiler_overlap >= internal_compiler_overlap
                        && aggregate.effective_wall_max <= active_union
                        && active_union <= aggregate.effective_wall_sum
                        && compiler_overlap <= active_union
                } else {
                    let maximum_compiler_overlap = aggregate
                        .effective_wall_sum
                        .checked_sub(aggregate.effective_wall_max)
                        .ok_or_else(|| invalid("native compiler process overlap underflows"))?;
                    compiler_overlap <= maximum_compiler_overlap
                };
                if compiler_count != aggregate.process_count
                    || max_parallel > 2
                    || max_parallel > compiler_count
                    || (compiler_count == 0) != (max_parallel == 0)
                    || (compiler_overlap == 0) != (max_parallel <= 1)
                    || !overlap_is_valid
                {
                    return Err(invalid("native parallel compiler evidence differs"));
                }
                Some((
                    compiler_overlap,
                    aggregate.module_job_count,
                    internal_compiler_overlap,
                ))
            }
            _ => unreachable!("format version was validated"),
        };
        match (
            self.format_version,
            &self.prepare_compiler_process_timings,
            &self.prepare_compiler_critical_tail,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V18, None, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(timings),
                claimed_tail,
            ) => {
                let process_count = self
                    .prepare_compiler_process_count
                    .ok_or_else(|| invalid("native compiler process count is absent"))?;
                let max_parallel = self
                    .prepare_max_parallel_compiler_process_count
                    .ok_or_else(|| invalid("native compiler concurrency is absent"))?;
                let overlap = self
                    .prepare_compiler_process_overlap_wall_time
                    .ok_or_else(|| invalid("native compiler overlap timing is absent"))?;
                let derived_tail = validate_compiler_process_evidence(
                    timings,
                    claimed_tail.as_ref(),
                    &prior_programs,
                    CompilerProcessValidationContext {
                        format_version: self.format_version,
                        process_count,
                        max_parallel,
                        claimed_overlap: overlap,
                        prepare_wall_time: self.prepare_wall_time,
                    },
                )?;
                if derived_tail.is_some() != claimed_tail.is_some() {
                    return Err(invalid("native compiler critical-tail presence differs"));
                }
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V18, _, _) => {
                return Err(invalid(
                    "legacy native report has compiler critical-path evidence",
                ));
            }
            _ => return Err(invalid("native compiler critical-path evidence is absent")),
        }
        match (
            self.format_version,
            self.prepare_render_capsule_hit_count,
            self.prepare_render_capsule_miss_count,
            self.prepare_local_render_job_count,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V21, None, None, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V22 | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(_),
                Some(_),
                Some(_),
            ) => {}
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V21, _, _, _) => {
                return Err(invalid("legacy native report has render capsule evidence"));
            }
            _ => return Err(invalid("native render capsule evidence is absent")),
        }
        let render_overlap = match (
            self.format_version,
            self.prepare_parallel_render_overlap_wall_time,
            self.prepare_max_parallel_render_job_count,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V15, None, None) => 0,
            (
                NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22,
                Some(overlap),
                Some(max_parallel),
            ) => {
                let overlap = overlap
                    .as_nanos()
                    .map_err(|_| invalid("invalid native render overlap duration"))?;
                let aggregate = NativeRenderAggregate::from_programs(
                    std::iter::once(&self.main)
                        .chain(self.accumulation.iter())
                        .chain(self.partial_flush.iter())
                        .chain(&self.zero_grad)
                        .chain(&self.evaluation),
                )?;
                let maximum_overlap = aggregate
                    .wall_sum
                    .checked_sub(aggregate.wall_max)
                    .ok_or_else(|| invalid("native render overlap underflows"))?;
                let no_render_jobs = aggregate.wall_sum == 0;
                if (no_render_jobs && (max_parallel != 0 || overlap != 0))
                    || (!no_render_jobs && max_parallel == 0)
                    || max_parallel > 2
                    || max_parallel > aggregate.job_count
                    || overlap > maximum_overlap
                    || (!no_render_jobs && (overlap == 0) != (max_parallel == 1))
                {
                    return Err(invalid("native parallel render evidence differs"));
                }
                overlap
            }
            (NATIVE_TRAINING_REPORT_FORMAT_VERSION, Some(overlap), Some(max_parallel)) => {
                let overlap = overlap
                    .as_nanos()
                    .map_err(|_| invalid("invalid native render overlap duration"))?;
                let programs = std::iter::once(&self.main)
                    .chain(self.accumulation.iter())
                    .chain(self.partial_flush.iter())
                    .chain(&self.zero_grad)
                    .chain(&self.evaluation)
                    .collect::<Vec<_>>();
                let program_count = count(programs.len(), "native render program")?;
                let aggregate = NativeRenderAggregate::from_programs(programs)?;
                let maximum_overlap = aggregate
                    .wall_sum
                    .checked_sub(aggregate.wall_max)
                    .ok_or_else(|| invalid("native render overlap underflows"))?;
                let hit_count = self
                    .prepare_render_capsule_hit_count
                    .ok_or_else(|| invalid("native render capsule hit count is absent"))?;
                let miss_count = self
                    .prepare_render_capsule_miss_count
                    .ok_or_else(|| invalid("native render capsule miss count is absent"))?;
                let local_render_count = self
                    .prepare_local_render_job_count
                    .ok_or_else(|| invalid("native local render job count is absent"))?;
                let zero_timing = max_parallel == 0 && overlap == 0 && aggregate.wall_sum == 0;
                let timing_partition_matches = if local_render_count == 0 || aggregate.wall_sum == 0
                {
                    zero_timing
                } else {
                    max_parallel != 0 && (overlap == 0) == (max_parallel == 1)
                };
                if hit_count.checked_add(miss_count) != Some(program_count)
                    || miss_count != local_render_count
                    || !timing_partition_matches
                    || max_parallel > 2
                    || max_parallel > local_render_count
                    || overlap > maximum_overlap
                {
                    return Err(invalid("native render capsule evidence differs"));
                }
                overlap
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V15, _, _) => {
                return Err(invalid("legacy native report has parallel render evidence"));
            }
            _ => return Err(invalid("native parallel render evidence is absent")),
        };
        match (
            self.format_version,
            self.prepare_runtime_overhead_wall_time,
            self.prepare_parallel_module_overlap_wall_time,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, None, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_V9
                | NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11
                | NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(overhead),
                overlap,
            ) => {
                let overlap = match (self.format_version, overlap) {
                    (NATIVE_TRAINING_REPORT_FORMAT_V7, None) => 0,
                    (
                        NATIVE_TRAINING_REPORT_FORMAT_V8
                        | NATIVE_TRAINING_REPORT_FORMAT_V9
                        | NATIVE_TRAINING_REPORT_FORMAT_V10
                        | NATIVE_TRAINING_REPORT_FORMAT_V11
                        | NATIVE_TRAINING_REPORT_FORMAT_V12
                        | NATIVE_TRAINING_REPORT_FORMAT_V13
                        | NATIVE_TRAINING_REPORT_FORMAT_V14
                        | NATIVE_TRAINING_REPORT_FORMAT_V15
                        | NATIVE_TRAINING_REPORT_FORMAT_V16
                        | NATIVE_TRAINING_REPORT_FORMAT_V17
                        | NATIVE_TRAINING_REPORT_FORMAT_V18
                        | NATIVE_TRAINING_REPORT_FORMAT_V19
                        | NATIVE_TRAINING_REPORT_FORMAT_V20
                        | NATIVE_TRAINING_REPORT_FORMAT_V21
                        | NATIVE_TRAINING_REPORT_FORMAT_V22
                        | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                        Some(overlap),
                    ) => overlap
                        .as_nanos()
                        .map_err(|_| invalid("invalid native prepare overlap duration"))?,
                    _ => return Err(invalid("native prepare overlap timing differs")),
                };
                if let Some((compiler_overlap, module_job_count, internal_overlap)) =
                    parallel_evidence
                {
                    let cross_program_compiler_overlap = compiler_overlap
                        .checked_sub(internal_overlap)
                        .ok_or_else(|| invalid("native internal compiler overlap exceeds total"))?;
                    if cross_program_compiler_overlap > overlap {
                        return Err(invalid(
                            "native compiler overlap exceeds parallel module overlap",
                        ));
                    }
                    if overlap > 0 && module_job_count < 2 {
                        return Err(invalid(
                            "native parallel module overlap lacks two module jobs",
                        ));
                    }
                }
                let prepare_total = self
                    .prepare_wall_time
                    .as_nanos()
                    .map_err(|_| invalid("invalid native prepare duration"))?;
                let mut programs = std::iter::once(&self.main)
                    .chain(self.accumulation.iter())
                    .chain(self.partial_flush.iter())
                    .chain(&self.zero_grad)
                    .chain(&self.evaluation);
                let program_total = programs.try_fold(0u128, |total, program| {
                    let timing = program
                        .preparation_timing
                        .as_ref()
                        .ok_or_else(|| invalid("native program preparation timing is absent"))?;
                    timing
                        .total
                        .as_nanos()
                        .map_err(|_| invalid("invalid native program preparation duration"))?
                        .checked_add(total)
                        .ok_or_else(|| invalid("native program preparation duration overflows"))
                })?;
                let effective_program_total = program_total
                    .checked_sub(overlap)
                    .and_then(|total| total.checked_sub(render_overlap))
                    .ok_or_else(|| invalid("native prepare overlap exceeds program duration"))?;
                let partitioned = overhead
                    .as_nanos()
                    .map_err(|_| invalid("invalid native prepare overhead duration"))?
                    .checked_add(effective_program_total)
                    .ok_or_else(|| invalid("native prepare duration overflows"))?;
                if partitioned != prepare_total {
                    return Err(invalid("native prepare phases do not partition total"));
                }
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, _, _) => {
                return Err(invalid("legacy native report has preparation timing"));
            }
            _ => return Err(invalid("native prepare timing differs")),
        }
        if self.successful_replay_count < 2
            || self.successful_replay_count > MAX_REPLAY_SAMPLES as u64
            || self.steady_replay_wall_time.sample_count != self.successful_replay_count - 1
            || self.steady_replay_wall_time.min > self.steady_replay_wall_time.nearest_rank_p50
            || self.steady_replay_wall_time.nearest_rank_p50
                > self.steady_replay_wall_time.nearest_rank_p95
            || self.steady_replay_wall_time.nearest_rank_p95 > self.steady_replay_wall_time.max
            || self.steady_replay_total_wall_time < self.steady_replay_wall_time.max
        {
            return Err(invalid("invalid native training replay summary"));
        }
        validate_total_duration(
            &self.steady_replay_wall_time,
            self.steady_replay_total_wall_time,
        )?;
        match (
            self.format_version,
            &self.main_replay_executor_wall_time,
            &self.main_replay_recurrent_overhead_wall_time,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V5, None, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V6
                | NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_V9
                | NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11
                | NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(executor),
                Some(overhead),
            ) => {
                let steady_count = self.successful_replay_count - 1;
                executor.validate(steady_count)?;
                overhead.validate(steady_count)?;
                validate_phase_partition(
                    self.first_replay_wall_time,
                    executor.first,
                    overhead.first,
                    "first replay",
                )?;
                validate_phase_partition(
                    self.steady_replay_total_wall_time,
                    executor.steady_total,
                    overhead.steady_total,
                    "steady replay",
                )?;
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V5, _, _) => {
                return Err(invalid("legacy native training report has replay phases"));
            }
            _ => return Err(invalid("native training replay phases differ")),
        }
        match (
            self.format_version,
            &self.main_replay_native_dispatcher_wall_time,
            &self.main_replay_executor_host_wall_time,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V17, None, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(native_dispatcher),
                Some(executor_host),
            ) => {
                let steady_count = self.successful_replay_count - 1;
                native_dispatcher.validate(steady_count)?;
                executor_host.validate(steady_count)?;
                let executor = self
                    .main_replay_executor_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("native replay executor timing is absent"))?;
                validate_phase_partition(
                    executor.first,
                    native_dispatcher.first,
                    executor_host.first,
                    "first replay executor",
                )?;
                validate_phase_partition(
                    executor.steady_total,
                    native_dispatcher.steady_total,
                    executor_host.steady_total,
                    "steady replay executor",
                )?;
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V17, _, _) => {
                return Err(invalid(
                    "legacy native training report has dispatcher timing",
                ));
            }
            _ => return Err(invalid("native training dispatcher timing differs")),
        }
        match (self.format_version, &self.step_phases) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V9, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11
                | NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14
                | NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17,
                Some(phases),
            ) => phases.validate(
                self.successful_replay_count,
                self.first_replay_wall_time,
                self.steady_replay_total_wall_time,
                ReplayTimingPartition {
                    executor: self
                        .main_replay_executor_wall_time
                        .as_ref()
                        .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                    native_dispatcher: None,
                    executor_host: None,
                    overhead: self
                        .main_replay_recurrent_overhead_wall_time
                        .as_ref()
                        .ok_or_else(|| invalid("classified replay overhead timing is absent"))?,
                },
            )?,
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V9, Some(_)) => {
                return Err(invalid("legacy native training report has step phases"));
            }
            (
                NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(phases),
            ) => phases.validate(
                self.successful_replay_count,
                self.first_replay_wall_time,
                self.steady_replay_total_wall_time,
                ReplayTimingPartition {
                    executor: self
                        .main_replay_executor_wall_time
                        .as_ref()
                        .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                    native_dispatcher: self.main_replay_native_dispatcher_wall_time.as_ref(),
                    executor_host: self.main_replay_executor_host_wall_time.as_ref(),
                    overhead: self
                        .main_replay_recurrent_overhead_wall_time
                        .as_ref()
                        .ok_or_else(|| invalid("classified replay overhead timing is absent"))?,
                },
            )?,
            (
                NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                None,
            ) if self.accumulation.is_none() => {}
            _ => return Err(invalid("native training step phases differ")),
        }
        let expected_rate = rate_from_total(
            self.steady_replay_wall_time.sample_count,
            self.steady_replay_total_wall_time,
        )?;
        if self.steady_microbatches_per_second.map(f64::to_bits) != expected_rate.map(f64::to_bits)
            || self.schedule_cache_keys.len()
                != usize::try_from(self.main.native_item_count)
                    .map_err(|_| invalid("native item count overflows usize"))?
            || match &self.accumulation {
                Some(program) => {
                    self.accumulation_schedule_cache_keys.len()
                        != usize::try_from(program.native_item_count).map_err(|_| {
                            invalid("native accumulation item count overflows usize")
                        })?
                }
                None => !self.accumulation_schedule_cache_keys.is_empty(),
            }
        {
            return Err(invalid("invalid native training replay inventory"));
        }
        if self.fallback_count != 0
            || self.kernel_launch_count.is_some()
            || self.host_to_device.is_some()
            || self.device_to_host.is_some()
            || self.measured_peak_host_memory_bytes.is_some()
        {
            return Err(invalid("native CPU availability fields are inconsistent"));
        }
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .wall_time
                .to_duration()
                .map_err(|_| invalid("invalid checkpoint duration"))?;
            let expected_step = self
                .initial_replay_step
                .checked_add(self.successful_replay_count)
                .ok_or_else(|| invalid("replay step overflows"))?;
            if checkpoint.byte_count == 0
                || checkpoint.capture_identity != self.main.capture_identity
                || checkpoint.replay_step != expected_step
            {
                return Err(invalid("checkpoint does not match recorded replays"));
            }
        }
        Ok(())
    }
}
