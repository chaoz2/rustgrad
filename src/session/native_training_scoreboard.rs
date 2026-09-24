//! Validated observational evidence for strict-native CPU compiled training.
//!
//! The runtime remains the source of preparation and replay facts. This module
//! only checks and aggregates detached reports; it does not time calls, execute
//! programs, or infer unavailable device and allocator measurements.

mod compiler_evidence;
mod inspection;
mod program_report;
mod step_phases;

#[cfg(test)]
use compiler_evidence::NativeTrainingCompilerProcessKind;
use compiler_evidence::{
    CompilerProcessValidationContext, NativeTrainingCompilerCriticalTail,
    NativeTrainingCompilerProcessTiming, NativeTrainingModuleOverlap,
    NativeTrainingProgramPairOverlap, NativeTrainingTranslationUnit, compiler_process_evidence,
    validate_compiler_process_evidence, validate_module_overlaps, validate_program_pair_overlaps,
    validate_translation_units,
};
use inspection::ProgramInspection;
pub use inspection::{
    CompiledAdamWInspection, CompiledTrainingCompileObservation,
    CompiledTrainingCompilePhaseObservation, NativeTrainingPreparationTiming,
};
pub use program_report::NativeTrainingProgramReport;
use program_report::{NativeCompilerAggregate, NativeRenderAggregate};
use step_phases::ReplayTimingPartition;
pub use step_phases::{
    NativeTrainingFirstStepReport, NativeTrainingStepPhase, NativeTrainingStepPhaseReport,
    NativeTrainingWarmStepReport,
};

#[cfg(test)]
use super::NativeCpuDispatchSegmentation;
use super::{
    CompiledAdamWCheckpoint, NativeCpuCompiledAdamWPreparationReport,
    NativeCpuCompiledAdamWStepResult, NativeCpuProgramPreparationReport, NativeCpuReplayTraffic,
    NativeCpuRunReport,
};
#[cfg(test)]
use crate::ExecutionPlanSummary;
use crate::{BenchmarkDuration, BenchmarkLatencySummary, BenchmarkTransfer, Error, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const NATIVE_TRAINING_REPORT_FORMAT_V2: u32 = 2;
const NATIVE_TRAINING_REPORT_FORMAT_V3: u32 = 3;
const NATIVE_TRAINING_REPORT_FORMAT_V4: u32 = 4;
const NATIVE_TRAINING_REPORT_FORMAT_V5: u32 = 5;
const NATIVE_TRAINING_REPORT_FORMAT_V6: u32 = 6;
const NATIVE_TRAINING_REPORT_FORMAT_V7: u32 = 7;
const NATIVE_TRAINING_REPORT_FORMAT_V8: u32 = 8;
const NATIVE_TRAINING_REPORT_FORMAT_V9: u32 = 9;
const NATIVE_TRAINING_REPORT_FORMAT_V10: u32 = 10;
const NATIVE_TRAINING_REPORT_FORMAT_V11: u32 = 11;
const NATIVE_TRAINING_REPORT_FORMAT_V12: u32 = 12;
const NATIVE_TRAINING_REPORT_FORMAT_V13: u32 = 13;
const NATIVE_TRAINING_REPORT_FORMAT_V14: u32 = 14;
const NATIVE_TRAINING_REPORT_FORMAT_V15: u32 = 15;
const NATIVE_TRAINING_REPORT_FORMAT_V16: u32 = 16;
const NATIVE_TRAINING_REPORT_FORMAT_V17: u32 = 17;
const NATIVE_TRAINING_REPORT_FORMAT_V18: u32 = 18;
const NATIVE_TRAINING_REPORT_FORMAT_V19: u32 = 19;
const NATIVE_TRAINING_REPORT_FORMAT_V20: u32 = 20;
const NATIVE_TRAINING_REPORT_FORMAT_V21: u32 = 21;
const NATIVE_TRAINING_REPORT_FORMAT_V22: u32 = 22;
pub const NATIVE_TRAINING_REPORT_FORMAT_VERSION: u32 = 23;
const MAX_REPLAY_SAMPLES: usize = 10_000;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointReport {
    capture_identity: u64,
    replay_step: u64,
    byte_count: u64,
    wall_time: BenchmarkDuration,
}

/// First-replay and bounded steady-replay wall time for one measured portion
/// of successful strict-native CPU training-step replay.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReplayTiming {
    first: BenchmarkDuration,
    steady_total: BenchmarkDuration,
    steady: BenchmarkLatencySummary,
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

    fn from_durations(first: Duration, steady: &[Duration]) -> Result<Self> {
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

#[derive(Clone, Copy, Debug)]
struct ReplayTiming {
    total: Duration,
    executor: Duration,
    native_dispatcher: Duration,
    executor_host: Duration,
    overhead: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayRecordingMode {
    Unset,
    Raw,
    Phased,
}

/// One typed backend-neutral compilation phase in the portable training
/// scoreboard. Exactly one inventory kind is present: graph nodes for graph
/// construction phases or logical schedule items for captured programs.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingCompilePhase {
    wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    graph_node_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logical_schedule_item_count: Option<u64>,
}

impl NativeTrainingCompilePhase {
    fn from_observation(observation: CompiledTrainingCompilePhaseObservation) -> Result<Self> {
        Ok(Self {
            wall_time: BenchmarkDuration::from_duration(observation.wall_time()),
            graph_node_count: observation
                .graph_node_count()
                .map(|value| count(value, "compiled graph node"))
                .transpose()?,
            logical_schedule_item_count: observation
                .logical_schedule_item_count()
                .map(|value| count(value, "compiled logical schedule item"))
                .transpose()?,
        })
    }

    fn duration(self) -> Result<Duration> {
        self.wall_time
            .to_duration()
            .map_err(|_| invalid("invalid compiled training phase duration"))
    }

    fn validate_graph(self) -> Result<()> {
        self.duration()?;
        if self.graph_node_count.is_none() || self.logical_schedule_item_count.is_some() {
            return Err(invalid("compiled graph phase inventory differs"));
        }
        Ok(())
    }

    fn validate_schedule(self, expected_items: u64) -> Result<()> {
        self.duration()?;
        if self.graph_node_count.is_some()
            || self.logical_schedule_item_count != Some(expected_items)
        {
            return Err(invalid("compiled capture phase inventory differs"));
        }
        Ok(())
    }

    pub const fn wall_time(&self) -> BenchmarkDuration {
        self.wall_time
    }

    pub const fn graph_node_count(&self) -> Option<u64> {
        self.graph_node_count
    }

    pub const fn logical_schedule_item_count(&self) -> Option<u64> {
        self.logical_schedule_item_count
    }
}

/// Exact partition of one caller-observed backend-neutral training compile.
/// Residual time contains wrapper work outside the timed compiler phases; the
/// checked sum of all phase durations and residual equals `compile_wall_time`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingCompilePhaseReport {
    compile_count: u64,
    objective_forward: NativeTrainingCompilePhase,
    autograd: NativeTrainingCompilePhase,
    optimizer_lowering: NativeTrainingCompilePhase,
    main_capture: NativeTrainingCompilePhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation_capture: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    partial_flush: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zero_grad: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    evaluation: Option<NativeTrainingCompilePhase>,
    residual_wall_time: BenchmarkDuration,
}

impl NativeTrainingCompilePhaseReport {
    fn from_observation(
        observation: &CompiledTrainingCompileObservation,
        compile_wall_time: Duration,
    ) -> Result<Self> {
        let objective_forward =
            NativeTrainingCompilePhase::from_observation(observation.objective_forward())?;
        let autograd = NativeTrainingCompilePhase::from_observation(observation.autograd())?;
        let optimizer_lowering =
            NativeTrainingCompilePhase::from_observation(observation.optimizer_lowering())?;
        let main_capture =
            NativeTrainingCompilePhase::from_observation(observation.main_capture())?;
        let accumulation_capture = observation
            .accumulation_capture()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let partial_flush = observation
            .partial_flush()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let zero_grad = observation
            .zero_grad()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let evaluation = observation
            .evaluation()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let measured = observation
            .measured_wall_time()
            .ok_or_else(|| invalid("compiled training phase duration overflows"))?;
        let residual = compile_wall_time
            .checked_sub(measured)
            .ok_or_else(|| invalid("compiled training phases exceed compile wall time"))?;
        Ok(Self {
            compile_count: observation.compile_count(),
            objective_forward,
            autograd,
            optimizer_lowering,
            main_capture,
            accumulation_capture,
            partial_flush,
            zero_grad,
            evaluation,
            residual_wall_time: BenchmarkDuration::from_duration(residual),
        })
    }

    fn validate(
        &self,
        compile_wall_time: BenchmarkDuration,
        main: &NativeTrainingProgramReport,
        accumulation: Option<&NativeTrainingProgramReport>,
        partial_flush: Option<&NativeTrainingProgramReport>,
        zero_grad: Option<&NativeTrainingProgramReport>,
        evaluation: Option<&NativeTrainingProgramReport>,
    ) -> Result<()> {
        if self.compile_count != 1 {
            return Err(invalid("compiled training compile count differs"));
        }
        self.objective_forward.validate_graph()?;
        self.autograd.validate_graph()?;
        self.optimizer_lowering.validate_graph()?;
        let objective_nodes = self.objective_forward.graph_node_count.unwrap_or(0);
        let autograd_nodes = self.autograd.graph_node_count.unwrap_or(0);
        let optimizer_nodes = self.optimizer_lowering.graph_node_count.unwrap_or(0);
        if objective_nodes == 0
            || objective_nodes > autograd_nodes
            || autograd_nodes > optimizer_nodes
        {
            return Err(invalid("compiled graph phase inventory order differs"));
        }
        self.main_capture
            .validate_schedule(main.logical_schedule_item_count)?;
        for (phase, program) in [
            (self.accumulation_capture, accumulation),
            (self.partial_flush, partial_flush),
            (self.zero_grad, zero_grad),
            (self.evaluation, evaluation),
        ] {
            match (phase, program) {
                (Some(phase), Some(program)) => {
                    phase.validate_schedule(program.logical_schedule_item_count)?
                }
                (None, None) => {}
                _ => return Err(invalid("compiled phase program inventory differs")),
            }
        }
        let total = [
            Some(self.objective_forward),
            Some(self.autograd),
            Some(self.optimizer_lowering),
            Some(self.main_capture),
            self.accumulation_capture,
            self.partial_flush,
            self.zero_grad,
            self.evaluation,
        ]
        .into_iter()
        .flatten()
        .try_fold(Duration::ZERO, |total, phase| {
            total
                .checked_add(phase.duration()?)
                .ok_or_else(|| invalid("compiled training phase duration overflows"))
        })?
        .checked_add(
            self.residual_wall_time
                .to_duration()
                .map_err(|_| invalid("invalid compiled training residual duration"))?,
        )
        .ok_or_else(|| invalid("compiled training phase duration overflows"))?;
        if total
            != compile_wall_time
                .to_duration()
                .map_err(|_| invalid("invalid native training compile duration"))?
        {
            return Err(invalid("compiled training phase partition differs"));
        }
        Ok(())
    }

    pub const fn compile_count(&self) -> u64 {
        self.compile_count
    }

    pub const fn objective_forward(&self) -> NativeTrainingCompilePhase {
        self.objective_forward
    }

    pub const fn autograd(&self) -> NativeTrainingCompilePhase {
        self.autograd
    }

    pub const fn optimizer_lowering(&self) -> NativeTrainingCompilePhase {
        self.optimizer_lowering
    }

    pub const fn main_capture(&self) -> NativeTrainingCompilePhase {
        self.main_capture
    }

    pub const fn accumulation_capture(&self) -> Option<NativeTrainingCompilePhase> {
        self.accumulation_capture
    }

    pub const fn partial_flush(&self) -> Option<NativeTrainingCompilePhase> {
        self.partial_flush
    }

    pub const fn zero_grad(&self) -> Option<NativeTrainingCompilePhase> {
        self.zero_grad
    }

    pub const fn evaluation(&self) -> Option<NativeTrainingCompilePhase> {
        self.evaluation
    }

    pub const fn residual_wall_time(&self) -> BenchmarkDuration {
        self.residual_wall_time
    }
}

/// Versioned strict-native CPU compiled-training observation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReport {
    format_version: u32,
    compile_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compile_phases: Option<NativeTrainingCompilePhaseReport>,
    prepare_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_runtime_overhead_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_parallel_module_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_parallel_render_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_max_parallel_render_job_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_render_capsule_hit_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_render_capsule_miss_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_local_render_job_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_max_parallel_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_timings: Option<Vec<NativeTrainingCompilerProcessTiming>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_critical_tail: Option<NativeTrainingCompilerCriticalTail>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_module_overlaps: Option<Vec<NativeTrainingModuleOverlap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_program_pair_overlaps: Option<Vec<NativeTrainingProgramPairOverlap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_translation_units: Option<Vec<NativeTrainingTranslationUnit>>,
    initial_replay_step: u64,
    successful_replay_count: u64,
    main: NativeTrainingProgramReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation: Option<NativeTrainingProgramReport>,
    partial_flush: Option<NativeTrainingProgramReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zero_grad: Option<NativeTrainingProgramReport>,
    evaluation: Option<NativeTrainingProgramReport>,
    recurrent_logical_state_count: u64,
    recurrent_logical_state_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_traffic: Option<NativeCpuReplayTraffic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_executed_native_item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation_replay_traffic: Option<NativeCpuReplayTraffic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation_replay_executed_native_item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_executor_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_native_dispatcher_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_executor_host_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_recurrent_overhead_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    step_phases: Option<NativeTrainingStepPhaseReport>,
    first_replay_wall_time: BenchmarkDuration,
    steady_replay_total_wall_time: BenchmarkDuration,
    steady_replay_wall_time: BenchmarkLatencySummary,
    steady_microbatches_per_second: Option<f64>,
    schedule_cache_keys: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accumulation_schedule_cache_keys: Vec<u64>,
    checkpoint: Option<CheckpointReport>,
    fallback_count: u64,
    kernel_launch_count: Option<u64>,
    host_to_device: Option<BenchmarkTransfer>,
    device_to_host: Option<BenchmarkTransfer>,
    measured_peak_host_memory_bytes: Option<u64>,
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

    fn validate(&self) -> Result<()> {
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

/// Bounded collector for successful strict-native CPU training-step replays.
pub struct NativeTrainingScoreboard {
    compile_wall_time: Duration,
    compile_phases: NativeTrainingCompilePhaseReport,
    prepare_wall_time: Duration,
    prepare_runtime_overhead_wall_time: Duration,
    prepare_parallel_module_overlap_wall_time: Duration,
    prepare_parallel_render_overlap_wall_time: Duration,
    prepare_max_parallel_render_job_count: u64,
    prepare_render_capsule_hit_count: u64,
    prepare_render_capsule_miss_count: u64,
    prepare_local_render_job_count: u64,
    prepare_compiler_process_overlap_wall_time: Duration,
    prepare_compiler_process_count: u64,
    prepare_max_parallel_compiler_process_count: u64,
    prepare_compiler_process_timings: Vec<NativeTrainingCompilerProcessTiming>,
    prepare_compiler_critical_tail: Option<NativeTrainingCompilerCriticalTail>,
    prepare_module_overlaps: Vec<NativeTrainingModuleOverlap>,
    prepare_program_pair_overlaps: Vec<NativeTrainingProgramPairOverlap>,
    prepare_translation_units: Vec<NativeTrainingTranslationUnit>,
    inspection: CompiledAdamWInspection,
    main: NativeTrainingProgramReport,
    accumulation: Option<NativeTrainingProgramReport>,
    partial_flush: Option<NativeTrainingProgramReport>,
    zero_grad: Option<NativeTrainingProgramReport>,
    evaluation: Option<NativeTrainingProgramReport>,
    recording_mode: ReplayRecordingMode,
    replay_timings: Vec<ReplayTiming>,
    replay_step_phases: Vec<NativeTrainingStepPhase>,
    schedule_cache_keys: Option<Vec<u64>>,
    main_replay_traffic: Option<NativeCpuReplayTraffic>,
    main_replay_executed_native_item_count: Option<u64>,
    accumulation_schedule_cache_keys: Option<Vec<u64>>,
    accumulation_replay_traffic: Option<NativeCpuReplayTraffic>,
    accumulation_replay_executed_native_item_count: Option<u64>,
    checkpoint: Option<CheckpointReport>,
}

impl NativeTrainingScoreboard {
    /// Starts a bounded observation from one complete strict-native
    /// preparation. `compile_wall_time` is caller-observed around the fresh
    /// plan construction and must contain its measured compile phases;
    /// `prepare_wall_time` similarly encloses every attached program's
    /// measured preparation time.
    pub fn new(
        inspection: CompiledAdamWInspection,
        preparation: &NativeCpuCompiledAdamWPreparationReport,
        compile_wall_time: Duration,
        prepare_wall_time: Duration,
    ) -> Result<Self> {
        let mut prior_native_identities = Vec::new();
        let main = NativeTrainingProgramReport::new(
            &inspection.main,
            preparation.main(),
            &prior_native_identities,
        )?;
        prior_native_identities.push(main.native_identity());
        let accumulation = matching_program(
            "accumulation",
            inspection.accumulation.as_ref(),
            preparation.accumulation(),
            &prior_native_identities,
        )?;
        if let Some(program) = &accumulation {
            prior_native_identities.push(program.native_identity());
        }
        let partial_flush = matching_program(
            "partial flush",
            inspection.partial_flush.as_ref(),
            preparation.partial_flush(),
            &prior_native_identities,
        )?;
        if let Some(program) = &partial_flush {
            prior_native_identities.push(program.native_identity());
        }
        let zero_grad = matching_program(
            "zero grad",
            inspection.zero_grad.as_ref(),
            preparation.zero_grad(),
            &prior_native_identities,
        )?;
        if let Some(program) = &zero_grad {
            prior_native_identities.push(program.native_identity());
        }
        let evaluation = matching_program(
            "evaluation",
            inspection.evaluation.as_ref(),
            preparation.evaluation(),
            &prior_native_identities,
        )?;
        let compile_phases = NativeTrainingCompilePhaseReport::from_observation(
            inspection
                .compile_phases()
                .ok_or_else(|| invalid("compiled training phase observation is absent"))?,
            compile_wall_time,
        )?;
        let programs = std::iter::once(&main)
            .chain(accumulation.iter())
            .chain(partial_flush.iter())
            .chain(zero_grad.iter())
            .chain(evaluation.iter())
            .collect::<Vec<_>>();
        let (prepare_compiler_process_timings, prepare_compiler_critical_tail) =
            compiler_process_evidence(preparation, &programs, prepare_wall_time)?;
        let prepare_module_overlaps = preparation
            .module_overlaps()
            .iter()
            .map(|overlap| NativeTrainingModuleOverlap::from_preparation(overlap, &programs))
            .collect::<Result<Vec<_>>>()?;
        validate_module_overlaps(&prepare_module_overlaps, &programs)?;
        let prepare_program_pair_overlaps = preparation
            .program_pair_overlaps()
            .iter()
            .map(|overlap| NativeTrainingProgramPairOverlap::from_preparation(overlap, &programs))
            .collect::<Result<Vec<_>>>()?;
        let prepare_translation_units = preparation
            .translation_units()
            .iter()
            .map(|unit| NativeTrainingTranslationUnit::from_preparation(unit, &programs))
            .collect::<Result<Vec<_>>>()?;
        validate_program_pair_overlaps(
            &prepare_program_pair_overlaps,
            &programs,
            &prepare_module_overlaps,
        )?;
        validate_translation_units(
            &prepare_translation_units,
            &programs,
            &prepare_compiler_process_timings,
        )?;
        if inspection.recurrent_state_count != preparation.recurrent_state_count()
            || inspection.recurrent_state_bytes != preparation.recurrent_state_bytes()
        {
            return Err(invalid("plan and prepared recurrent state differ"));
        }
        let program_prepare_wall_time = std::iter::once(preparation.main())
            .chain(preparation.accumulation())
            .chain(preparation.partial_flush())
            .chain(preparation.zero_grad())
            .chain(preparation.evaluation())
            .try_fold(Duration::ZERO, |total, program| {
                total
                    .checked_add(program.wall_time())
                    .ok_or_else(|| invalid("native program preparation duration overflows"))
            })?;
        let prepare_parallel_module_overlap_wall_time =
            preparation.parallel_module_overlap_wall_time();
        let prepare_parallel_render_overlap_wall_time =
            preparation.parallel_render_overlap_wall_time();
        let prepare_max_parallel_render_job_count = count(
            preparation.max_parallel_render_job_count(),
            "parallel native render job",
        )?;
        let prepare_render_capsule_hit_count = count(
            preparation.render_capsule_hit_count(),
            "native render capsule hit",
        )?;
        let prepare_render_capsule_miss_count = count(
            preparation.render_capsule_miss_count(),
            "native render capsule miss",
        )?;
        let prepare_local_render_job_count = count(
            preparation.local_render_job_count(),
            "native local render job",
        )?;
        let prepare_compiler_process_count = count(
            preparation.compiler_process_count(),
            "native compiler process",
        )?;
        let prepare_max_parallel_compiler_process_count = count(
            preparation.max_parallel_compiler_process_count(),
            "parallel native compiler process",
        )?;
        let effective_program_prepare_wall_time = program_prepare_wall_time
            .checked_sub(prepare_parallel_module_overlap_wall_time)
            .and_then(|time| time.checked_sub(prepare_parallel_render_overlap_wall_time))
            .ok_or_else(|| invalid("native parallel work overlap exceeds program time"))?;
        let prepare_runtime_overhead_wall_time = prepare_wall_time
            .checked_sub(effective_program_prepare_wall_time)
            .ok_or_else(|| invalid("native program preparation exceeds whole prepare time"))?;
        Ok(Self {
            compile_wall_time,
            compile_phases,
            prepare_wall_time,
            prepare_runtime_overhead_wall_time,
            prepare_parallel_module_overlap_wall_time,
            prepare_parallel_render_overlap_wall_time,
            prepare_max_parallel_render_job_count,
            prepare_render_capsule_hit_count,
            prepare_render_capsule_miss_count,
            prepare_local_render_job_count,
            prepare_compiler_process_overlap_wall_time: preparation
                .compiler_process_overlap_wall_time(),
            prepare_compiler_process_count,
            prepare_max_parallel_compiler_process_count,
            prepare_compiler_process_timings,
            prepare_compiler_critical_tail,
            prepare_module_overlaps,
            prepare_program_pair_overlaps,
            prepare_translation_units,
            inspection,
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
            recording_mode: ReplayRecordingMode::Unset,
            replay_timings: Vec::new(),
            replay_step_phases: Vec::new(),
            schedule_cache_keys: None,
            main_replay_traffic: None,
            main_replay_executed_native_item_count: None,
            accumulation_schedule_cache_keys: None,
            accumulation_replay_traffic: None,
            accumulation_replay_executed_native_item_count: None,
            checkpoint: None,
        })
    }

    /// Records one unclassified report returned by a committed native training
    /// step. A scoreboard cannot mix this raw mode with [`Self::record_step`].
    pub fn record(&mut self, report: &NativeCpuRunReport) -> Result<()> {
        if self.recording_mode == ReplayRecordingMode::Phased {
            return Err(invalid("cannot mix raw and classified replay samples"));
        }
        if self.accumulation.is_some() {
            return Err(invalid(
                "phase-specialized replay requires classified step recording",
            ));
        }
        let phase = NativeTrainingStepPhase::OptimizerCommit;
        let (timing, executed) = self.validate_replay(report, phase)?;
        self.commit_replay(report, timing, executed, phase);
        self.recording_mode = ReplayRecordingMode::Raw;
        Ok(())
    }

    /// Records and classifies one successful compiled AdamW training step using
    /// only its authenticated public `did_update` result. Partial flushes are
    /// deliberately outside this training-step scoreboard.
    pub fn record_step(&mut self, step: &NativeCpuCompiledAdamWStepResult) -> Result<()> {
        if self.recording_mode == ReplayRecordingMode::Raw {
            return Err(invalid("cannot mix raw and classified replay samples"));
        }
        let phase = if step.did_update() {
            NativeTrainingStepPhase::OptimizerCommit
        } else {
            NativeTrainingStepPhase::AccumulationOnly
        };
        let (timing, executed) = self.validate_replay(step.report(), phase)?;
        self.commit_replay(step.report(), timing, executed, phase);
        self.replay_step_phases.push(phase);
        self.recording_mode = ReplayRecordingMode::Phased;
        Ok(())
    }

    fn validate_replay(
        &self,
        report: &NativeCpuRunReport,
        phase: NativeTrainingStepPhase,
    ) -> Result<(ReplayTiming, u64)> {
        if self.replay_timings.len() >= MAX_REPLAY_SAMPLES {
            return Err(invalid("native training replay sample limit exceeded"));
        }
        let expected_invocation = self.replay_timings.len() as u64 + 1;
        let expected = match phase {
            NativeTrainingStepPhase::AccumulationOnly => self
                .accumulation
                .as_ref()
                .ok_or_else(|| invalid("accumulation replay program is absent"))?,
            NativeTrainingStepPhase::OptimizerCommit => &self.main,
        };
        let (expected_cache_keys, expected_traffic, expected_executed) = match phase {
            NativeTrainingStepPhase::AccumulationOnly => (
                &self.accumulation_schedule_cache_keys,
                self.accumulation_replay_traffic,
                self.accumulation_replay_executed_native_item_count,
            ),
            NativeTrainingStepPhase::OptimizerCommit => (
                &self.schedule_cache_keys,
                self.main_replay_traffic,
                self.main_replay_executed_native_item_count,
            ),
        };
        if report.capture_identity() != expected.capture_identity
            || report.native_identity() != expected.native_identity
            || report.is_vectorized() != expected.vectorized
            || count(report.native_item_count(), "native item")? != expected.native_item_count
            || report.executed_native_item_count()
                > usize::try_from(expected.rendered_entry_count)
                    .map_err(|_| invalid("native rendered entry count overflows usize"))?
            || count(report.module_dispatch_count(), "native module dispatch")?
                != expected
                    .dispatch_segmentation
                    .as_ref()
                    .ok_or_else(|| invalid("native dispatch segmentation evidence is absent"))?
                    .segment_count
            || report.fallback_count() != 0
            || report.successful_invocation() != expected_invocation
            || report.schedule_cache_keys().len() != report.native_item_count()
        {
            return Err(invalid(
                "native replay report does not match the scoreboard",
            ));
        }
        if let Some(expected) = expected_cache_keys
            && expected != report.schedule_cache_keys()
        {
            return Err(invalid("native replay cache keys changed"));
        }
        let recurrent_state_bytes = count(
            self.inspection.recurrent_state_bytes,
            "recurrent state byte",
        )?;
        let has_recurrent_inventory = report.traffic().retained_recurrent_state_count() != 0
            || report.traffic().retained_recurrent_state_bytes() != 0
            || report.traffic().replaced_recurrent_state_count() != 0
            || report.traffic().replaced_recurrent_state_bytes() != 0;
        if report.traffic().borrowed_recurrent_input_bytes() != recurrent_state_bytes
            || report
                .traffic()
                .borrowed_recurrent_output_bytes()
                .checked_add(report.traffic().retained_recurrent_state_bytes())
                != Some(recurrent_state_bytes)
            || report.traffic().retained_recurrent_state_count()
                > count(self.inspection.recurrent_state_count, "recurrent state")?
            || (has_recurrent_inventory
                && (report
                    .traffic()
                    .retained_recurrent_state_count()
                    .checked_add(report.traffic().replaced_recurrent_state_count())
                    != Some(count(
                        self.inspection.recurrent_state_count,
                        "recurrent state",
                    )?)
                    || report.traffic().replaced_recurrent_state_bytes()
                        != report.traffic().borrowed_recurrent_output_bytes()))
        {
            return Err(invalid(
                "native replay traffic does not match recurrent state",
            ));
        }
        if let Some(expected) = expected_traffic
            && expected != *report.traffic()
        {
            return Err(invalid("native replay traffic changed"));
        }
        let executed = count(report.executed_native_item_count(), "executed native item")?;
        if let Some(expected) = expected_executed
            && expected != executed
        {
            return Err(invalid("native replay execution count changed"));
        }
        let total = report.wall_time();
        let executor = report.executor_wall_time();
        let native_dispatcher = report.native_dispatcher_wall_time();
        let executor_host = executor
            .checked_sub(native_dispatcher)
            .ok_or_else(|| invalid("native dispatcher time exceeds executor time"))?;
        let overhead = total
            .checked_sub(executor)
            .ok_or_else(|| invalid("native executor time exceeds replay time"))?;
        Ok((
            ReplayTiming {
                total,
                executor,
                native_dispatcher,
                executor_host,
                overhead,
            },
            executed,
        ))
    }

    fn commit_replay(
        &mut self,
        report: &NativeCpuRunReport,
        timing: ReplayTiming,
        executed: u64,
        phase: NativeTrainingStepPhase,
    ) {
        let (cache_keys, traffic, executed_count) = match phase {
            NativeTrainingStepPhase::AccumulationOnly => (
                &mut self.accumulation_schedule_cache_keys,
                &mut self.accumulation_replay_traffic,
                &mut self.accumulation_replay_executed_native_item_count,
            ),
            NativeTrainingStepPhase::OptimizerCommit => (
                &mut self.schedule_cache_keys,
                &mut self.main_replay_traffic,
                &mut self.main_replay_executed_native_item_count,
            ),
        };
        if cache_keys.is_none() {
            *cache_keys = Some(report.schedule_cache_keys().to_vec());
        }
        if traffic.is_none() {
            *traffic = Some(*report.traffic());
        }
        if executed_count.is_none() {
            *executed_count = Some(executed);
        }
        self.replay_timings.push(timing);
    }

    pub fn observe_checkpoint(
        &mut self,
        checkpoint: &CompiledAdamWCheckpoint,
        wall_time: Duration,
    ) -> Result<()> {
        let info = checkpoint.info();
        let replay_count = self.replay_timings.len() as u64;
        let expected_step = self
            .inspection
            .initial_replay_step
            .checked_add(replay_count)
            .ok_or_else(|| invalid("checkpoint replay step overflows"))?;
        let expected_accumulation_identity = self
            .accumulation
            .as_ref()
            .map(|program| program.capture_identity);
        if info.capture_identity() != self.main.capture_identity
            || info.replay_step() != expected_step
            || info.accumulation_capture_identity() != expected_accumulation_identity
        {
            return Err(invalid("checkpoint does not match recorded replays"));
        }
        self.checkpoint = Some(CheckpointReport {
            capture_identity: info.capture_identity(),
            replay_step: info.replay_step(),
            byte_count: count(checkpoint.as_bytes().len(), "checkpoint byte")?,
            wall_time: BenchmarkDuration::from_duration(wall_time),
        });
        Ok(())
    }

    pub fn report(&self) -> Result<NativeTrainingReport> {
        let Some((first, steady)) = self.replay_timings.split_first() else {
            return Err(invalid("native training scoreboard has no replay"));
        };
        if steady.is_empty() {
            return Err(invalid("native training scoreboard has no steady replay"));
        }
        let totals = steady.iter().map(|timing| timing.total).collect::<Vec<_>>();
        let executors = steady
            .iter()
            .map(|timing| timing.executor)
            .collect::<Vec<_>>();
        let overheads = steady
            .iter()
            .map(|timing| timing.overhead)
            .collect::<Vec<_>>();
        let native_dispatchers = steady
            .iter()
            .map(|timing| timing.native_dispatcher)
            .collect::<Vec<_>>();
        let executor_hosts = steady
            .iter()
            .map(|timing| timing.executor_host)
            .collect::<Vec<_>>();
        let (steady_replay_total_wall_time, steady_microbatches_per_second) = rate(&totals)?;
        let step_phases = match self.recording_mode {
            ReplayRecordingMode::Raw => None,
            ReplayRecordingMode::Phased => Some(NativeTrainingStepPhaseReport::from_timings(
                &self.replay_timings,
                &self.replay_step_phases,
            )?),
            ReplayRecordingMode::Unset => {
                return Err(invalid("native training scoreboard has no replay mode"));
            }
        };
        let report = NativeTrainingReport {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            compile_wall_time: BenchmarkDuration::from_duration(self.compile_wall_time),
            compile_phases: Some(self.compile_phases.clone()),
            prepare_wall_time: BenchmarkDuration::from_duration(self.prepare_wall_time),
            prepare_runtime_overhead_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_runtime_overhead_wall_time,
            )),
            prepare_parallel_module_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_parallel_module_overlap_wall_time,
            )),
            prepare_parallel_render_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_parallel_render_overlap_wall_time,
            )),
            prepare_max_parallel_render_job_count: Some(self.prepare_max_parallel_render_job_count),
            prepare_render_capsule_hit_count: Some(self.prepare_render_capsule_hit_count),
            prepare_render_capsule_miss_count: Some(self.prepare_render_capsule_miss_count),
            prepare_local_render_job_count: Some(self.prepare_local_render_job_count),
            prepare_compiler_process_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_compiler_process_overlap_wall_time,
            )),
            prepare_compiler_process_count: Some(self.prepare_compiler_process_count),
            prepare_max_parallel_compiler_process_count: Some(
                self.prepare_max_parallel_compiler_process_count,
            ),
            prepare_compiler_process_timings: Some(self.prepare_compiler_process_timings.clone()),
            prepare_compiler_critical_tail: self.prepare_compiler_critical_tail.clone(),
            prepare_module_overlaps: Some(self.prepare_module_overlaps.clone()),
            prepare_program_pair_overlaps: Some(self.prepare_program_pair_overlaps.clone()),
            prepare_translation_units: Some(self.prepare_translation_units.clone()),
            initial_replay_step: self.inspection.initial_replay_step,
            successful_replay_count: self.replay_timings.len() as u64,
            main: self.main.clone(),
            accumulation: self.accumulation.clone(),
            partial_flush: self.partial_flush.clone(),
            zero_grad: self.zero_grad.clone(),
            evaluation: self.evaluation.clone(),
            recurrent_logical_state_count: count(
                self.inspection.recurrent_state_count,
                "recurrent state",
            )?,
            recurrent_logical_state_bytes: count(
                self.inspection.recurrent_state_bytes,
                "recurrent state byte",
            )?,
            main_replay_traffic: self.main_replay_traffic,
            main_replay_executed_native_item_count: self.main_replay_executed_native_item_count,
            accumulation_replay_traffic: self.accumulation_replay_traffic,
            accumulation_replay_executed_native_item_count: self
                .accumulation_replay_executed_native_item_count,
            main_replay_executor_wall_time: Some(NativeTrainingReplayTiming::from_durations(
                first.executor,
                &executors,
            )?),
            main_replay_native_dispatcher_wall_time: Some(
                NativeTrainingReplayTiming::from_durations(
                    first.native_dispatcher,
                    &native_dispatchers,
                )?,
            ),
            main_replay_executor_host_wall_time: Some(NativeTrainingReplayTiming::from_durations(
                first.executor_host,
                &executor_hosts,
            )?),
            main_replay_recurrent_overhead_wall_time: Some(
                NativeTrainingReplayTiming::from_durations(first.overhead, &overheads)?,
            ),
            step_phases,
            first_replay_wall_time: BenchmarkDuration::from_duration(first.total),
            steady_replay_total_wall_time,
            steady_replay_wall_time: latency_summary(&totals)?,
            steady_microbatches_per_second,
            schedule_cache_keys: self.schedule_cache_keys.clone().unwrap_or_default(),
            accumulation_schedule_cache_keys: self
                .accumulation_schedule_cache_keys
                .clone()
                .unwrap_or_default(),
            checkpoint: self.checkpoint.clone(),
            fallback_count: 0,
            kernel_launch_count: None,
            host_to_device: None,
            device_to_host: None,
            measured_peak_host_memory_bytes: None,
        };
        report.validate()?;
        Ok(report)
    }
}

fn matching_program(
    label: &str,
    inspection: Option<&ProgramInspection>,
    preparation: Option<&NativeCpuProgramPreparationReport>,
    prior_native_identities: &[u64],
) -> Result<Option<NativeTrainingProgramReport>> {
    match (inspection, preparation) {
        (None, None) => Ok(None),
        (Some(inspection), Some(preparation)) => {
            NativeTrainingProgramReport::new(inspection, preparation, prior_native_identities)
                .map(Some)
        }
        _ => Err(invalid(format!(
            "plan and prepared {label} presence differ"
        ))),
    }
}

fn latency_summary(durations: &[Duration]) -> Result<BenchmarkLatencySummary> {
    let mut ordered = durations.to_vec();
    ordered.sort_unstable();
    let nearest_rank = |percentile: usize| {
        let rank = (ordered.len() * percentile).div_ceil(100);
        ordered[rank.saturating_sub(1)]
    };
    Ok(BenchmarkLatencySummary {
        sample_count: count(ordered.len(), "steady replay")?,
        min: BenchmarkDuration::from_duration(ordered[0]),
        nearest_rank_p50: BenchmarkDuration::from_duration(nearest_rank(50)),
        nearest_rank_p95: BenchmarkDuration::from_duration(nearest_rank(95)),
        max: BenchmarkDuration::from_duration(ordered[ordered.len() - 1]),
    })
}

fn rate(durations: &[Duration]) -> Result<(BenchmarkDuration, Option<f64>)> {
    let total = sum_durations(durations)?;
    Ok((total, rate_from_total(durations.len() as u64, total)?))
}

fn sum_durations(durations: &[Duration]) -> Result<BenchmarkDuration> {
    let total = durations
        .iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total
                .checked_add(*duration)
                .ok_or_else(|| invalid("replay duration overflows"))
        })?;
    Ok(BenchmarkDuration::from_duration(total))
}

fn rate_from_total(sample_count: u64, total: BenchmarkDuration) -> Result<Option<f64>> {
    let total = total
        .to_duration()
        .map_err(|_| invalid("invalid steady replay total duration"))?;
    Ok((!total.is_zero()).then(|| sample_count as f64 / total.as_secs_f64()))
}

fn validate_total_duration(
    summary: &BenchmarkLatencySummary,
    total: BenchmarkDuration,
) -> Result<()> {
    let remaining = u128::from(summary.sample_count - 1);
    let min = summary
        .min
        .as_nanos()
        .map_err(|_| invalid("invalid steady replay minimum duration"))?;
    let max = summary
        .max
        .as_nanos()
        .map_err(|_| invalid("invalid steady replay maximum duration"))?;
    let total = total
        .as_nanos()
        .map_err(|_| invalid("invalid steady replay total duration"))?;
    let lower = remaining
        .checked_mul(min)
        .and_then(|rest| max.checked_add(rest))
        .ok_or_else(|| invalid("steady replay duration lower bound overflows"))?;
    let upper = remaining
        .checked_mul(max)
        .and_then(|rest| min.checked_add(rest))
        .ok_or_else(|| invalid("steady replay duration upper bound overflows"))?;
    if !(lower..=upper).contains(&total) {
        return Err(invalid("steady replay total is inconsistent with summary"));
    }
    Ok(())
}

fn validate_phase_partition(
    total: BenchmarkDuration,
    executor: BenchmarkDuration,
    overhead: BenchmarkDuration,
    label: &str,
) -> Result<()> {
    let total = total
        .as_nanos()
        .map_err(|_| invalid(format!("invalid {label} total duration")))?;
    let partitioned = executor
        .as_nanos()
        .map_err(|_| invalid(format!("invalid {label} executor duration")))?
        .checked_add(
            overhead
                .as_nanos()
                .map_err(|_| invalid(format!("invalid {label} overhead duration")))?,
        )
        .ok_or_else(|| invalid(format!("{label} phase duration overflows")))?;
    if partitioned != total {
        return Err(invalid(format!("{label} phases do not partition total")));
    }
    Ok(())
}

fn count(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid(format!("{label} count overflows u64")))
}

fn invalid(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: format!("native training scoreboard: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
