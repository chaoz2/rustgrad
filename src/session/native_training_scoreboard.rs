//! Validated observational evidence for strict-native CPU compiled training.
//!
//! The runtime remains the source of preparation and replay facts. This module
//! only checks and aggregates detached reports; it does not time calls, execute
//! programs, or infer unavailable device and allocator measurements.

use super::{
    CompiledAdamWCheckpoint, NativeCpuCompiledAdamWPreparationReport,
    NativeCpuProgramPreparationReport, NativeCpuReplayTraffic, NativeCpuRunReport,
};
use crate::{
    BenchmarkDuration, BenchmarkLatencySummary, BenchmarkTransfer, Error, ExecutionPlanSummary,
    Result,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const NATIVE_TRAINING_REPORT_FORMAT_V2: u32 = 2;
const NATIVE_TRAINING_REPORT_FORMAT_V3: u32 = 3;
const NATIVE_TRAINING_REPORT_FORMAT_V4: u32 = 4;
const NATIVE_TRAINING_REPORT_FORMAT_V5: u32 = 5;
const NATIVE_TRAINING_REPORT_FORMAT_V6: u32 = 6;
const NATIVE_TRAINING_REPORT_FORMAT_V7: u32 = 7;
const NATIVE_TRAINING_REPORT_FORMAT_V8: u32 = 8;
pub const NATIVE_TRAINING_REPORT_FORMAT_VERSION: u32 = 9;
const MAX_REPLAY_SAMPLES: usize = 10_000;

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProgramInspection {
    capture_identity: u64,
    execution_plan: ExecutionPlanSummary,
}

/// Exact host wall-time partition for preparing one strict-native CPU program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingPreparationTiming {
    total: BenchmarkDuration,
    layout: BenchmarkDuration,
    render: BenchmarkDuration,
    compiler_process: BenchmarkDuration,
    module_load: BenchmarkDuration,
    residual: BenchmarkDuration,
}

impl NativeTrainingPreparationTiming {
    fn from_preparation(preparation: &NativeCpuProgramPreparationReport) -> Self {
        let phases = preparation.phases();
        Self {
            total: BenchmarkDuration::from_duration(preparation.wall_time()),
            layout: BenchmarkDuration::from_duration(phases.layout_wall_time()),
            render: BenchmarkDuration::from_duration(phases.render_wall_time()),
            compiler_process: BenchmarkDuration::from_duration(phases.compiler_process_wall_time()),
            module_load: BenchmarkDuration::from_duration(phases.module_load_wall_time()),
            residual: BenchmarkDuration::from_duration(phases.residual_wall_time()),
        }
    }

    fn validate(&self, work: &NativeTrainingProgramReport) -> Result<()> {
        let total = self
            .total
            .as_nanos()
            .map_err(|_| invalid("invalid native preparation total duration"))?;
        let partitioned = [
            self.layout,
            self.render,
            self.compiler_process,
            self.module_load,
            self.residual,
        ]
        .into_iter()
        .try_fold(0u128, |total, duration| {
            duration
                .as_nanos()
                .map_err(|_| invalid("invalid native preparation phase duration"))?
                .checked_add(total)
                .ok_or_else(|| invalid("native preparation phase duration overflows"))
        })?;
        if partitioned != total
            || (work.compiler_invocation_count == 0
                && self.compiler_process != BenchmarkDuration::from_duration(Duration::ZERO))
            || (work.loaded_module_count == 0
                && self.module_load != BenchmarkDuration::from_duration(Duration::ZERO))
        {
            return Err(invalid("native preparation phases do not match work"));
        }
        Ok(())
    }

    pub const fn total(&self) -> BenchmarkDuration {
        self.total
    }

    pub const fn layout(&self) -> BenchmarkDuration {
        self.layout
    }

    pub const fn render(&self) -> BenchmarkDuration {
        self.render
    }

    pub const fn compiler_process(&self) -> BenchmarkDuration {
        self.compiler_process
    }

    pub const fn module_load(&self) -> BenchmarkDuration {
        self.module_load
    }

    pub const fn residual(&self) -> BenchmarkDuration {
        self.residual
    }
}

/// Immutable logical work and recurrent-state facts for one compiled AdamW
/// plan. Inspection prepares no target and exposes no capture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAdamWInspection {
    initial_replay_step: u64,
    main: ProgramInspection,
    partial_flush: Option<ProgramInspection>,
    zero_grad: Option<ProgramInspection>,
    evaluation: Option<ProgramInspection>,
    recurrent_state_count: usize,
    recurrent_state_bytes: usize,
}

impl CompiledAdamWInspection {
    pub(crate) fn new(
        initial_replay_step: u64,
        main: (u64, ExecutionPlanSummary),
        partial_flush: Option<(u64, ExecutionPlanSummary)>,
        zero_grad: Option<(u64, ExecutionPlanSummary)>,
        evaluation: Option<(u64, ExecutionPlanSummary)>,
        recurrent_state: (usize, usize),
    ) -> Self {
        let program = |(capture_identity, execution_plan)| ProgramInspection {
            capture_identity,
            execution_plan,
        };
        Self {
            initial_replay_step,
            main: program(main),
            partial_flush: partial_flush.map(program),
            zero_grad: zero_grad.map(program),
            evaluation: evaluation.map(program),
            recurrent_state_count: recurrent_state.0,
            recurrent_state_bytes: recurrent_state.1,
        }
    }

    pub const fn initial_replay_step(&self) -> u64 {
        self.initial_replay_step
    }

    pub const fn main(&self) -> (u64, &ExecutionPlanSummary) {
        (self.main.capture_identity, &self.main.execution_plan)
    }

    pub fn partial_flush(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.partial_flush
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
    }

    pub fn evaluation(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.evaluation
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
    }

    pub fn zero_grad(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.zero_grad
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
    }

    pub const fn recurrent_state_count(&self) -> usize {
        self.recurrent_state_count
    }

    pub const fn recurrent_state_bytes(&self) -> usize {
        self.recurrent_state_bytes
    }
}

/// Static logical work, identity, and cache facts for one prepared pure
/// training program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingProgramReport {
    capture_identity: u64,
    native_identity: u64,
    vectorized: bool,
    execution_plan_identity: u64,
    logical_schedule_item_count: u64,
    peak_logical_temporary_allocation_count: u64,
    peak_logical_temporary_bytes: u64,
    native_item_count: u64,
    cache_hit_count: u64,
    cache_miss_count: u64,
    #[serde(default)]
    rendered_entry_count: u64,
    #[serde(default)]
    loaded_module_count: u64,
    #[serde(default)]
    durable_artifact_cache_hit_count: u64,
    #[serde(default)]
    durable_artifact_cache_miss_count: u64,
    #[serde(default)]
    compiler_invocation_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preparation_timing: Option<NativeTrainingPreparationTiming>,
}

impl NativeTrainingProgramReport {
    fn new(
        inspection: &ProgramInspection,
        preparation: &NativeCpuProgramPreparationReport,
    ) -> Result<Self> {
        if inspection.capture_identity != preparation.capture_identity()
            || &inspection.execution_plan != preparation.execution_plan()
        {
            return Err(invalid("plan and native preparation differ"));
        }
        let plan = &inspection.execution_plan;
        Ok(Self {
            capture_identity: preparation.capture_identity(),
            native_identity: preparation.native_identity(),
            vectorized: preparation.is_vectorized(),
            execution_plan_identity: plan.identity,
            logical_schedule_item_count: count(plan.schedule_item_count, "schedule item")?,
            peak_logical_temporary_allocation_count: count(
                plan.peak_logical_allocations,
                "peak logical allocation",
            )?,
            peak_logical_temporary_bytes: count(
                plan.peak_logical_bytes,
                "peak logical temporary byte",
            )?,
            native_item_count: count(preparation.native_item_count(), "native item")?,
            cache_hit_count: count(preparation.cache_hit_count(), "cache hit")?,
            cache_miss_count: count(preparation.cache_miss_count(), "cache miss")?,
            rendered_entry_count: count(
                preparation.work().rendered_entry_count(),
                "rendered entry",
            )?,
            loaded_module_count: count(preparation.work().loaded_module_count(), "loaded module")?,
            durable_artifact_cache_hit_count: count(
                preparation.work().durable_artifact_cache_hit_count(),
                "durable artifact cache hit",
            )?,
            durable_artifact_cache_miss_count: count(
                preparation.work().durable_artifact_cache_miss_count(),
                "durable artifact cache miss",
            )?,
            compiler_invocation_count: count(
                preparation.work().compiler_invocation_count(),
                "compiler invocation",
            )?,
            preparation_timing: Some(NativeTrainingPreparationTiming::from_preparation(
                preparation,
            )),
        })
    }

    fn validate(&self, format_version: u32) -> Result<()> {
        if self
            .cache_hit_count
            .checked_add(self.cache_miss_count)
            .ok_or_else(|| invalid("native program cache count overflows"))?
            != self.native_item_count
        {
            return Err(invalid("native program cache inventory differs"));
        }
        let preparation = [
            self.rendered_entry_count,
            self.loaded_module_count,
            self.durable_artifact_cache_hit_count,
            self.durable_artifact_cache_miss_count,
            self.compiler_invocation_count,
        ];
        if format_version < NATIVE_TRAINING_REPORT_FORMAT_V5 {
            if preparation.into_iter().any(|value| value != 0) {
                return Err(invalid(
                    "legacy native program has module preparation evidence",
                ));
            }
        } else {
            let durable_access_count = self
                .durable_artifact_cache_hit_count
                .checked_add(self.durable_artifact_cache_miss_count)
                .ok_or_else(|| invalid("durable artifact cache count overflows"))?;
            let expected_modules = u64::from(self.rendered_entry_count != 0);
            let rendered_inventory_is_valid = if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V8
            {
                self.rendered_entry_count == self.native_item_count
            } else {
                self.rendered_entry_count <= self.native_item_count
                    && (self.rendered_entry_count == 0) == (self.native_item_count == 0)
            };
            if !rendered_inventory_is_valid
                || self.loaded_module_count != expected_modules
                || durable_access_count > self.loaded_module_count
                || self.compiler_invocation_count != self.durable_artifact_cache_miss_count
            {
                return Err(invalid(
                    "native program module preparation evidence differs",
                ));
            }
        }
        match (format_version, &self.preparation_timing) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(timing),
            ) => timing.validate(self)?,
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, Some(_)) => {
                return Err(invalid(
                    "legacy native program has preparation phase timing",
                ));
            }
            _ => return Err(invalid("native program preparation timing differs")),
        }
        Ok(())
    }

    pub const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub const fn native_identity(&self) -> u64 {
        self.native_identity
    }

    pub const fn is_vectorized(&self) -> bool {
        self.vectorized
    }

    pub const fn execution_plan_identity(&self) -> u64 {
        self.execution_plan_identity
    }

    pub const fn logical_schedule_item_count(&self) -> u64 {
        self.logical_schedule_item_count
    }

    pub const fn peak_logical_temporary_allocation_count(&self) -> u64 {
        self.peak_logical_temporary_allocation_count
    }

    pub const fn peak_logical_temporary_bytes(&self) -> u64 {
        self.peak_logical_temporary_bytes
    }

    /// Logical schedule-item coverage used by the cache inventory.
    pub const fn native_item_count(&self) -> u64 {
        self.native_item_count
    }

    /// Logical schedule items covered by process-local cache hits.
    pub const fn cache_hit_count(&self) -> u64 {
        self.cache_hit_count
    }

    /// Logical schedule items covered by process-local cache misses.
    pub const fn cache_miss_count(&self) -> u64 {
        self.cache_miss_count
    }

    /// Physical compiled entries emitted for the logical schedule inventory.
    pub const fn rendered_entry_count(&self) -> u64 {
        self.rendered_entry_count
    }

    pub const fn loaded_module_count(&self) -> u64 {
        self.loaded_module_count
    }

    pub const fn durable_artifact_cache_hit_count(&self) -> u64 {
        self.durable_artifact_cache_hit_count
    }

    pub const fn durable_artifact_cache_miss_count(&self) -> u64 {
        self.durable_artifact_cache_miss_count
    }

    pub const fn compiler_invocation_count(&self) -> u64 {
        self.compiler_invocation_count
    }

    pub const fn preparation_timing(&self) -> Option<&NativeTrainingPreparationTiming> {
        self.preparation_timing.as_ref()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointReport {
    capture_identity: u64,
    replay_step: u64,
    byte_count: u64,
    wall_time: BenchmarkDuration,
}

/// First-replay and bounded steady-replay wall time for one measured portion
/// of successful strict-native CPU main replay.
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
    overhead: Duration,
}

/// Versioned strict-native CPU compiled-training observation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReport {
    format_version: u32,
    compile_wall_time: BenchmarkDuration,
    prepare_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_runtime_overhead_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_parallel_module_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_max_parallel_compiler_process_count: Option<u64>,
    initial_replay_step: u64,
    successful_replay_count: u64,
    main: NativeTrainingProgramReport,
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
    main_replay_executor_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_recurrent_overhead_wall_time: Option<NativeTrainingReplayTiming>,
    first_replay_wall_time: BenchmarkDuration,
    steady_replay_total_wall_time: BenchmarkDuration,
    steady_replay_wall_time: BenchmarkLatencySummary,
    steady_microbatches_per_second: Option<f64>,
    schedule_cache_keys: Vec<u64>,
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
    /// successfully published main replay.
    pub const fn main_replay_executed_native_item_count(&self) -> Option<u64> {
        self.main_replay_executed_native_item_count
    }

    /// Wall time inside the sealed native executor for committed main replays.
    pub const fn main_replay_executor_wall_time(&self) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_executor_wall_time.as_ref()
    }

    /// Checked end-to-end remainder outside the sealed native executor. This
    /// covers recurrent staging, validation, and atomic commit/publication.
    pub const fn main_replay_recurrent_overhead_wall_time(
        &self,
    ) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_recurrent_overhead_wall_time.as_ref()
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
        self.main.validate(self.format_version)?;
        match (self.format_version, &self.main_replay_traffic) {
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V3
                | NATIVE_TRAINING_REPORT_FORMAT_V4
                | NATIVE_TRAINING_REPORT_FORMAT_V5
                | NATIVE_TRAINING_REPORT_FORMAT_V6
                | NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(traffic),
            ) if traffic.borrowed_recurrent_input_bytes() == self.recurrent_logical_state_bytes
                && traffic.borrowed_recurrent_output_bytes()
                    == self.recurrent_logical_state_bytes => {}
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, Some(_)) => {
                return Err(invalid("legacy native training report has replay traffic"));
            }
            _ => return Err(invalid("native training replay traffic differs")),
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
                | NATIVE_TRAINING_REPORT_FORMAT_V8,
                Some(executed),
            ) if executed <= self.main.rendered_entry_count => {}
            (NATIVE_TRAINING_REPORT_FORMAT_VERSION, Some(executed))
                if executed <= self.main.rendered_entry_count => {}
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_V3, Some(_)) => {
                return Err(invalid("legacy native training report has execution count"));
            }
            _ => return Err(invalid("native training execution count differs")),
        }
        for program in self
            .partial_flush
            .iter()
            .chain(&self.zero_grad)
            .chain(&self.evaluation)
        {
            program.validate(self.format_version)?;
            if program.vectorized != self.main.vectorized {
                return Err(invalid("native program vectorization policy differs"));
            }
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
            NATIVE_TRAINING_REPORT_FORMAT_V8 | NATIVE_TRAINING_REPORT_FORMAT_VERSION => {
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
                let (expected_count, module_job_count, compiler_time_sum, compiler_time_max) =
                    std::iter::once(&self.main)
                        .chain(self.partial_flush.iter())
                        .chain(&self.zero_grad)
                        .chain(&self.evaluation)
                        .try_fold(
                            (0u64, 0u64, 0u128, 0u128),
                            |(compiler_total, job_total, time_sum, time_max), program| {
                                let compiler_total = compiler_total
                                    .checked_add(program.compiler_invocation_count)
                                    .ok_or_else(|| {
                                        invalid("native compiler process count overflows")
                                    })?;
                                let jobs = program
                                    .durable_artifact_cache_hit_count
                                    .checked_add(program.durable_artifact_cache_miss_count)
                                    .ok_or_else(|| invalid("native module job count overflows"))?;
                                let job_total = job_total
                                    .checked_add(jobs)
                                    .ok_or_else(|| invalid("native module job count overflows"))?;
                                let compiler_time = program
                                    .preparation_timing
                                    .as_ref()
                                    .ok_or_else(|| {
                                        invalid("native program preparation timing is absent")
                                    })?
                                    .compiler_process
                                    .as_nanos()
                                    .map_err(|_| {
                                        invalid("invalid native compiler process duration")
                                    })?;
                                let time_sum =
                                    time_sum.checked_add(compiler_time).ok_or_else(|| {
                                        invalid("native compiler process duration overflows")
                                    })?;
                                Ok((
                                    compiler_total,
                                    job_total,
                                    time_sum,
                                    time_max.max(compiler_time),
                                ))
                            },
                        )?;
                let maximum_compiler_overlap = compiler_time_sum
                    .checked_sub(compiler_time_max)
                    .ok_or_else(|| invalid("native compiler process overlap underflows"))?;
                if compiler_count != expected_count
                    || max_parallel > 2
                    || max_parallel > compiler_count
                    || (compiler_count == 0) != (max_parallel == 0)
                    || (compiler_overlap == 0) != (max_parallel <= 1)
                    || compiler_overlap > maximum_compiler_overlap
                {
                    return Err(invalid("native parallel compiler evidence differs"));
                }
                Some((compiler_overlap, module_job_count))
            }
            _ => unreachable!("format version was validated"),
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
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(overhead),
                overlap,
            ) => {
                let overlap = match (self.format_version, overlap) {
                    (NATIVE_TRAINING_REPORT_FORMAT_V7, None) => 0,
                    (
                        NATIVE_TRAINING_REPORT_FORMAT_V8 | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                        Some(overlap),
                    ) => overlap
                        .as_nanos()
                        .map_err(|_| invalid("invalid native prepare overlap duration"))?,
                    _ => return Err(invalid("native prepare overlap timing differs")),
                };
                if let Some((compiler_overlap, module_job_count)) = parallel_evidence {
                    if compiler_overlap > overlap {
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
        let expected_rate = rate_from_total(
            self.steady_replay_wall_time.sample_count,
            self.steady_replay_total_wall_time,
        )?;
        if self.steady_microbatches_per_second.map(f64::to_bits) != expected_rate.map(f64::to_bits)
            || self.schedule_cache_keys.len()
                != usize::try_from(self.main.native_item_count)
                    .map_err(|_| invalid("native item count overflows usize"))?
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

/// Bounded collector for successful strict-native CPU main replays.
pub struct NativeTrainingScoreboard {
    compile_wall_time: Duration,
    prepare_wall_time: Duration,
    prepare_runtime_overhead_wall_time: Duration,
    prepare_parallel_module_overlap_wall_time: Duration,
    prepare_compiler_process_overlap_wall_time: Duration,
    prepare_compiler_process_count: u64,
    prepare_max_parallel_compiler_process_count: u64,
    inspection: CompiledAdamWInspection,
    main: NativeTrainingProgramReport,
    partial_flush: Option<NativeTrainingProgramReport>,
    zero_grad: Option<NativeTrainingProgramReport>,
    evaluation: Option<NativeTrainingProgramReport>,
    replay_timings: Vec<ReplayTiming>,
    schedule_cache_keys: Option<Vec<u64>>,
    main_replay_traffic: Option<NativeCpuReplayTraffic>,
    main_replay_executed_native_item_count: Option<u64>,
    checkpoint: Option<CheckpointReport>,
}

impl NativeTrainingScoreboard {
    /// Starts a bounded observation from one complete strict-native
    /// preparation. `prepare_wall_time` is caller-observed around the whole
    /// target preparation and must contain every attached program's measured
    /// preparation time.
    pub fn new(
        inspection: CompiledAdamWInspection,
        preparation: &NativeCpuCompiledAdamWPreparationReport,
        compile_wall_time: Duration,
        prepare_wall_time: Duration,
    ) -> Result<Self> {
        let main = NativeTrainingProgramReport::new(&inspection.main, preparation.main())?;
        let partial_flush = matching_program(
            "partial flush",
            inspection.partial_flush.as_ref(),
            preparation.partial_flush(),
        )?;
        let zero_grad = matching_program(
            "zero grad",
            inspection.zero_grad.as_ref(),
            preparation.zero_grad(),
        )?;
        let evaluation = matching_program(
            "evaluation",
            inspection.evaluation.as_ref(),
            preparation.evaluation(),
        )?;
        if inspection.recurrent_state_count != preparation.recurrent_state_count()
            || inspection.recurrent_state_bytes != preparation.recurrent_state_bytes()
        {
            return Err(invalid("plan and prepared recurrent state differ"));
        }
        let program_prepare_wall_time = std::iter::once(preparation.main())
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
            .ok_or_else(|| invalid("native parallel module overlap exceeds program time"))?;
        let prepare_runtime_overhead_wall_time = prepare_wall_time
            .checked_sub(effective_program_prepare_wall_time)
            .ok_or_else(|| invalid("native program preparation exceeds whole prepare time"))?;
        Ok(Self {
            compile_wall_time,
            prepare_wall_time,
            prepare_runtime_overhead_wall_time,
            prepare_parallel_module_overlap_wall_time,
            prepare_compiler_process_overlap_wall_time: preparation
                .compiler_process_overlap_wall_time(),
            prepare_compiler_process_count,
            prepare_max_parallel_compiler_process_count,
            inspection,
            main,
            partial_flush,
            zero_grad,
            evaluation,
            replay_timings: Vec::new(),
            schedule_cache_keys: None,
            main_replay_traffic: None,
            main_replay_executed_native_item_count: None,
            checkpoint: None,
        })
    }

    /// Records one report returned by a committed native training step. Failed
    /// or rejected steps expose no report and therefore cannot become samples.
    pub fn record(&mut self, report: &NativeCpuRunReport) -> Result<()> {
        if self.replay_timings.len() >= MAX_REPLAY_SAMPLES {
            return Err(invalid("native training replay sample limit exceeded"));
        }
        let expected_invocation = self.replay_timings.len() as u64 + 1;
        if report.capture_identity() != self.main.capture_identity
            || report.native_identity() != self.main.native_identity
            || report.is_vectorized() != self.main.vectorized
            || count(report.native_item_count(), "native item")? != self.main.native_item_count
            || report.executed_native_item_count()
                > usize::try_from(self.main.rendered_entry_count)
                    .map_err(|_| invalid("native rendered entry count overflows usize"))?
            || report.fallback_count() != 0
            || report.successful_invocation() != expected_invocation
            || report.schedule_cache_keys().len() != report.native_item_count()
        {
            return Err(invalid(
                "native replay report does not match the scoreboard",
            ));
        }
        if let Some(expected) = &self.schedule_cache_keys
            && expected != report.schedule_cache_keys()
        {
            return Err(invalid("native replay cache keys changed"));
        }
        let recurrent_state_bytes = count(
            self.inspection.recurrent_state_bytes,
            "recurrent state byte",
        )?;
        if report.traffic().borrowed_recurrent_input_bytes() != recurrent_state_bytes
            || report.traffic().borrowed_recurrent_output_bytes() != recurrent_state_bytes
        {
            return Err(invalid(
                "native replay traffic does not match recurrent state",
            ));
        }
        if let Some(expected) = self.main_replay_traffic
            && expected != *report.traffic()
        {
            return Err(invalid("native replay traffic changed"));
        }
        let executed = count(report.executed_native_item_count(), "executed native item")?;
        if let Some(expected) = self.main_replay_executed_native_item_count
            && expected != executed
        {
            return Err(invalid("native replay execution count changed"));
        }
        let total = report.wall_time();
        let executor = report.executor_wall_time();
        let overhead = total
            .checked_sub(executor)
            .ok_or_else(|| invalid("native executor time exceeds replay time"))?;
        if self.schedule_cache_keys.is_none() {
            self.schedule_cache_keys = Some(report.schedule_cache_keys().to_vec());
        }
        if self.main_replay_traffic.is_none() {
            self.main_replay_traffic = Some(*report.traffic());
        }
        if self.main_replay_executed_native_item_count.is_none() {
            self.main_replay_executed_native_item_count = Some(executed);
        }
        self.replay_timings.push(ReplayTiming {
            total,
            executor,
            overhead,
        });
        Ok(())
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
        if info.capture_identity() != self.main.capture_identity
            || info.replay_step() != expected_step
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
        let (steady_replay_total_wall_time, steady_microbatches_per_second) = rate(&totals)?;
        let report = NativeTrainingReport {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            compile_wall_time: BenchmarkDuration::from_duration(self.compile_wall_time),
            prepare_wall_time: BenchmarkDuration::from_duration(self.prepare_wall_time),
            prepare_runtime_overhead_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_runtime_overhead_wall_time,
            )),
            prepare_parallel_module_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_parallel_module_overlap_wall_time,
            )),
            prepare_compiler_process_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_compiler_process_overlap_wall_time,
            )),
            prepare_compiler_process_count: Some(self.prepare_compiler_process_count),
            prepare_max_parallel_compiler_process_count: Some(
                self.prepare_max_parallel_compiler_process_count,
            ),
            initial_replay_step: self.inspection.initial_replay_step,
            successful_replay_count: self.replay_timings.len() as u64,
            main: self.main.clone(),
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
            main_replay_executor_wall_time: Some(NativeTrainingReplayTiming::from_durations(
                first.executor,
                &executors,
            )?),
            main_replay_recurrent_overhead_wall_time: Some(
                NativeTrainingReplayTiming::from_durations(first.overhead, &overheads)?,
            ),
            first_replay_wall_time: BenchmarkDuration::from_duration(first.total),
            steady_replay_total_wall_time,
            steady_replay_wall_time: latency_summary(&totals)?,
            steady_microbatches_per_second,
            schedule_cache_keys: self.schedule_cache_keys.clone().unwrap_or_default(),
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
) -> Result<Option<NativeTrainingProgramReport>> {
    match (inspection, preparation) {
        (None, None) => Ok(None),
        (Some(inspection), Some(preparation)) => {
            NativeTrainingProgramReport::new(inspection, preparation).map(Some)
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
mod tests {
    use super::*;

    fn zero_duration() -> BenchmarkDuration {
        BenchmarkDuration::from_duration(Duration::ZERO)
    }

    fn zero_replay_timing() -> NativeTrainingReplayTiming {
        NativeTrainingReplayTiming {
            first: zero_duration(),
            steady_total: zero_duration(),
            steady: BenchmarkLatencySummary {
                sample_count: 1,
                min: zero_duration(),
                nearest_rank_p50: zero_duration(),
                nearest_rank_p95: zero_duration(),
                max: zero_duration(),
            },
        }
    }

    fn zero_preparation_timing() -> NativeTrainingPreparationTiming {
        NativeTrainingPreparationTiming {
            total: zero_duration(),
            layout: zero_duration(),
            render: zero_duration(),
            compiler_process: zero_duration(),
            module_load: zero_duration(),
            residual: zero_duration(),
        }
    }

    fn set_single_steady_replay_duration(
        report: &mut NativeTrainingReport,
        elapsed: BenchmarkDuration,
    ) {
        let summary = BenchmarkLatencySummary {
            sample_count: 1,
            min: elapsed,
            nearest_rank_p50: elapsed,
            nearest_rank_p95: elapsed,
            max: elapsed,
        };
        report.steady_replay_total_wall_time = elapsed;
        report.steady_replay_wall_time = summary.clone();
        let executor = report
            .main_replay_executor_wall_time
            .as_mut()
            .expect("current test report has executor timing");
        executor.steady_total = elapsed;
        executor.steady = summary;
        report.steady_microbatches_per_second = rate_from_total(1, elapsed).unwrap();
    }

    fn remove_module_preparation(json: &mut serde_json::Value) {
        for program in ["main", "partial_flush", "zero_grad", "evaluation"] {
            let Some(program) = json[program].as_object_mut() else {
                continue;
            };
            for field in [
                "rendered_entry_count",
                "loaded_module_count",
                "durable_artifact_cache_hit_count",
                "durable_artifact_cache_miss_count",
                "compiler_invocation_count",
            ] {
                program.remove(field);
            }
        }
    }

    fn remove_replay_phase_timing(json: &mut serde_json::Value) {
        json.as_object_mut()
            .unwrap()
            .remove("main_replay_executor_wall_time");
        json.as_object_mut()
            .unwrap()
            .remove("main_replay_recurrent_overhead_wall_time");
    }

    fn remove_preparation_phase_timing(json: &mut serde_json::Value) {
        json.as_object_mut()
            .unwrap()
            .remove("prepare_runtime_overhead_wall_time");
        json.as_object_mut()
            .unwrap()
            .remove("prepare_parallel_module_overlap_wall_time");
        for field in [
            "prepare_compiler_process_overlap_wall_time",
            "prepare_compiler_process_count",
            "prepare_max_parallel_compiler_process_count",
        ] {
            json.as_object_mut().unwrap().remove(field);
        }
        for program in ["main", "partial_flush", "zero_grad", "evaluation"] {
            if let Some(program) = json[program].as_object_mut() {
                program.remove("preparation_timing");
            }
        }
    }

    fn zero_report() -> NativeTrainingReport {
        NativeTrainingReport {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            compile_wall_time: zero_duration(),
            prepare_wall_time: zero_duration(),
            prepare_runtime_overhead_wall_time: Some(zero_duration()),
            prepare_parallel_module_overlap_wall_time: Some(zero_duration()),
            prepare_compiler_process_overlap_wall_time: Some(zero_duration()),
            prepare_compiler_process_count: Some(1),
            prepare_max_parallel_compiler_process_count: Some(1),
            initial_replay_step: 0,
            successful_replay_count: 2,
            main: NativeTrainingProgramReport {
                capture_identity: 7,
                native_identity: 11,
                vectorized: true,
                execution_plan_identity: 13,
                logical_schedule_item_count: 2,
                peak_logical_temporary_allocation_count: 1,
                peak_logical_temporary_bytes: 4,
                native_item_count: 2,
                cache_hit_count: 0,
                cache_miss_count: 2,
                rendered_entry_count: 2,
                loaded_module_count: 1,
                durable_artifact_cache_hit_count: 0,
                durable_artifact_cache_miss_count: 1,
                compiler_invocation_count: 1,
                preparation_timing: Some(zero_preparation_timing()),
            },
            partial_flush: None,
            zero_grad: None,
            evaluation: None,
            recurrent_logical_state_count: 4,
            recurrent_logical_state_bytes: 16,
            main_replay_traffic: Some(NativeCpuReplayTraffic::new(2, 12, 16, 16)),
            main_replay_executed_native_item_count: Some(1),
            main_replay_executor_wall_time: Some(zero_replay_timing()),
            main_replay_recurrent_overhead_wall_time: Some(zero_replay_timing()),
            first_replay_wall_time: zero_duration(),
            steady_replay_wall_time: BenchmarkLatencySummary {
                sample_count: 1,
                min: zero_duration(),
                nearest_rank_p50: zero_duration(),
                nearest_rank_p95: zero_duration(),
                max: zero_duration(),
            },
            steady_replay_total_wall_time: zero_duration(),
            steady_microbatches_per_second: None,
            schedule_cache_keys: vec![17, 19],
            checkpoint: Some(CheckpointReport {
                capture_identity: 7,
                replay_step: 2,
                byte_count: 32,
                wall_time: zero_duration(),
            }),
            fallback_count: 0,
            kernel_launch_count: None,
            host_to_device: None,
            device_to_host: None,
            measured_peak_host_memory_bytes: None,
        }
    }

    #[test]
    fn zero_durations_and_unavailable_cpu_measurements_round_trip() {
        let report = zero_report();
        let bytes = report.to_json_bytes().unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            json["format_version"],
            NATIVE_TRAINING_REPORT_FORMAT_VERSION
        );
        for field in [
            "kernel_launch_count",
            "host_to_device",
            "device_to_host",
            "measured_peak_host_memory_bytes",
            "steady_microbatches_per_second",
        ] {
            assert!(json[field].is_null());
        }
        assert_eq!(
            json["main_replay_traffic"]["borrowed_recurrent_input_bytes"],
            16
        );
        assert_eq!(
            json["main_replay_traffic"]["borrowed_recurrent_output_bytes"],
            16
        );
        assert_eq!(json["main_replay_executed_native_item_count"], 1);
        assert_eq!(json["main"]["preparation_timing"]["layout"]["secs"], 0);
        assert_eq!(json["prepare_runtime_overhead_wall_time"]["nanos"], 0);
        assert_eq!(
            json["prepare_parallel_module_overlap_wall_time"]["nanos"],
            0
        );
        assert_eq!(json["prepare_compiler_process_count"], 1);
        assert_eq!(json["prepare_max_parallel_compiler_process_count"], 1);
        assert_eq!(json["main_replay_executor_wall_time"]["first"]["secs"], 0);
        assert_eq!(
            json["main_replay_recurrent_overhead_wall_time"]["steady"]["sample_count"],
            1
        );
        assert_eq!(
            NativeTrainingReport::from_json_bytes(&bytes).unwrap(),
            report
        );
    }

    #[test]
    fn legacy_report_without_zero_grad_evidence_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(1);
        json.as_object_mut().unwrap().remove("zero_grad");
        json.as_object_mut().unwrap().remove("main_replay_traffic");
        json.as_object_mut()
            .unwrap()
            .remove("main_replay_executed_native_item_count");
        remove_module_preparation(&mut json);
        remove_replay_phase_timing(&mut json);
        remove_preparation_phase_timing(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.zero_grad().is_none());
    }

    #[test]
    fn version_two_report_without_replay_traffic_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V2);
        json.as_object_mut().unwrap().remove("main_replay_traffic");
        json.as_object_mut()
            .unwrap()
            .remove("main_replay_executed_native_item_count");
        remove_module_preparation(&mut json);
        remove_replay_phase_timing(&mut json);
        remove_preparation_phase_timing(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.main_replay_traffic().is_none());
    }

    #[test]
    fn version_three_report_without_execution_count_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V3);
        json.as_object_mut()
            .unwrap()
            .remove("main_replay_executed_native_item_count");
        remove_module_preparation(&mut json);
        remove_replay_phase_timing(&mut json);
        remove_preparation_phase_timing(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.main_replay_executed_native_item_count().is_none());
        assert!(report.main_replay_traffic().is_some());
    }

    #[test]
    fn version_four_report_without_module_preparation_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V4);
        remove_module_preparation(&mut json);
        remove_replay_phase_timing(&mut json);
        remove_preparation_phase_timing(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(report.main().rendered_entry_count(), 0);
    }

    #[test]
    fn version_five_report_without_replay_phase_timing_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V5);
        remove_replay_phase_timing(&mut json);
        remove_preparation_phase_timing(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.main_replay_executor_wall_time().is_none());
        assert!(report.main_replay_recurrent_overhead_wall_time().is_none());
    }

    #[test]
    fn version_six_report_without_preparation_phase_timing_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V6);
        remove_preparation_phase_timing(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.main().preparation_timing().is_none());
        assert!(report.prepare_runtime_overhead_wall_time().is_none());
        assert!(report.main_replay_executor_wall_time().is_some());
    }

    #[test]
    fn version_seven_report_without_parallel_module_overlap_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V7);
        json.as_object_mut()
            .unwrap()
            .remove("prepare_parallel_module_overlap_wall_time");
        for field in [
            "prepare_compiler_process_overlap_wall_time",
            "prepare_compiler_process_count",
            "prepare_max_parallel_compiler_process_count",
        ] {
            json.as_object_mut().unwrap().remove(field);
        }
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.prepare_parallel_module_overlap_wall_time().is_none());
        assert!(report.prepare_runtime_overhead_wall_time().is_some());
    }

    #[test]
    fn version_eight_report_with_exact_native_inventory_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V8);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(report.main().native_item_count(), 2);
        assert_eq!(report.main().rendered_entry_count(), 2);
        assert_eq!(report.main_replay_executed_native_item_count(), Some(1));
    }

    #[test]
    fn legacy_reports_reject_reduced_rendered_and_executed_inventories() {
        for version in [
            NATIVE_TRAINING_REPORT_FORMAT_V5,
            NATIVE_TRAINING_REPORT_FORMAT_V6,
            NATIVE_TRAINING_REPORT_FORMAT_V7,
            NATIVE_TRAINING_REPORT_FORMAT_V8,
        ] {
            let mut json = serde_json::to_value(zero_report()).unwrap();
            json["format_version"] = serde_json::json!(version);
            if version == NATIVE_TRAINING_REPORT_FORMAT_V5 {
                remove_replay_phase_timing(&mut json);
            }
            if version <= NATIVE_TRAINING_REPORT_FORMAT_V6 {
                remove_preparation_phase_timing(&mut json);
            } else if version == NATIVE_TRAINING_REPORT_FORMAT_V7 {
                json.as_object_mut()
                    .unwrap()
                    .remove("prepare_parallel_module_overlap_wall_time");
                for field in [
                    "prepare_compiler_process_overlap_wall_time",
                    "prepare_compiler_process_count",
                    "prepare_max_parallel_compiler_process_count",
                ] {
                    json.as_object_mut().unwrap().remove(field);
                }
            }
            json["main_replay_executed_native_item_count"] = serde_json::json!(2);
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_ok(),
                "legacy v{version} exact native inventory did not decode"
            );
            json["main"]["rendered_entry_count"] = serde_json::json!(1);
            json["main_replay_executed_native_item_count"] = serde_json::json!(1);
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
                "legacy v{version} accepted reduced rendered/executed inventories"
            );
        }
    }

    #[test]
    fn current_report_accepts_grouped_physical_native_inventory() {
        let mut report = zero_report();
        report.main.logical_schedule_item_count = 4;
        report.main.native_item_count = 4;
        report.main.cache_miss_count = 4;
        report.main.rendered_entry_count = 1;
        report.schedule_cache_keys.extend([23, 29]);
        let bytes = report.to_json_bytes().unwrap();
        let decoded = NativeTrainingReport::from_json_bytes(&bytes).unwrap();
        assert_eq!(
            decoded.format_version,
            NATIVE_TRAINING_REPORT_FORMAT_VERSION
        );
        assert_eq!(decoded.main().native_item_count(), 4);
        assert_eq!(decoded.main().rendered_entry_count(), 1);
        assert_eq!(decoded.main_replay_executed_native_item_count(), Some(1));
    }

    #[test]
    fn current_report_authenticates_preparation_phase_partition() {
        let mut report = zero_report();
        report.prepare_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(10));
        report.prepare_runtime_overhead_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(3)));
        report.prepare_parallel_module_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(2)));
        report.prepare_compiler_process_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
        report.prepare_compiler_process_count = Some(2);
        report.prepare_max_parallel_compiler_process_count = Some(2);
        let timing = report.main.preparation_timing.as_mut().unwrap();
        timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(7));
        timing.render = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        timing.compiler_process = BenchmarkDuration::from_duration(Duration::from_nanos(3));
        timing.module_load = BenchmarkDuration::from_duration(Duration::from_nanos(1));
        timing.residual = BenchmarkDuration::from_duration(Duration::from_nanos(1));
        let mut evaluation = report.main.clone();
        evaluation.preparation_timing = Some(NativeTrainingPreparationTiming {
            total: BenchmarkDuration::from_duration(Duration::from_nanos(2)),
            layout: zero_duration(),
            render: zero_duration(),
            compiler_process: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
            module_load: zero_duration(),
            residual: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
        });
        report.evaluation = Some(evaluation);
        assert!(report.validate().is_ok());
        let valid = report.clone();

        report.main.preparation_timing.as_mut().unwrap().residual =
            BenchmarkDuration::from_duration(Duration::from_nanos(2));
        assert!(report.validate().is_err());

        let mut report = valid.clone();
        report.prepare_runtime_overhead_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(6)));
        assert!(report.validate().is_err());

        let mut report = valid.clone();
        report.prepare_parallel_module_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(8)));
        assert!(report.validate().is_err());

        let mut report = valid.clone();
        report.prepare_compiler_process_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(2)));
        assert!(report.validate().is_err());

        let mut report = valid.clone();
        let main = report.main.preparation_timing.as_mut().unwrap();
        main.compiler_process = zero_duration();
        main.residual = BenchmarkDuration::from_duration(Duration::from_nanos(4));
        let evaluation = report
            .evaluation
            .as_mut()
            .unwrap()
            .preparation_timing
            .as_mut()
            .unwrap();
        evaluation.compiler_process = zero_duration();
        evaluation.residual = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        assert!(report.validate().is_err());

        let mut report = valid;
        report.prepare_compiler_process_overlap_wall_time = Some(zero_duration());
        assert!(report.validate().is_err());

        let mut report = zero_report();
        report.prepare_parallel_module_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
        report.main.preparation_timing.as_mut().unwrap().total =
            BenchmarkDuration::from_duration(Duration::from_nanos(1));
        report.main.preparation_timing.as_mut().unwrap().residual =
            BenchmarkDuration::from_duration(Duration::from_nanos(1));
        assert!(report.validate().is_err());

        let mut report = zero_report();
        report.prepare_compiler_process_count = Some(0);
        assert!(report.validate().is_err());

        let mut report = zero_report();
        report.prepare_max_parallel_compiler_process_count = Some(3);
        assert!(report.validate().is_err());

        let mut report = zero_report();
        report.prepare_compiler_process_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
        assert!(report.validate().is_err());

        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["main"]
            .as_object_mut()
            .unwrap()
            .remove("preparation_timing");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );
    }

    #[test]
    fn current_report_authenticates_replay_phase_partition() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json.as_object_mut()
            .unwrap()
            .remove("main_replay_executor_wall_time");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut report = zero_report();
        let total = BenchmarkDuration::from_duration(Duration::from_nanos(11));
        let executor = BenchmarkDuration::from_duration(Duration::from_nanos(7));
        let overhead = BenchmarkDuration::from_duration(Duration::from_nanos(4));
        report.first_replay_wall_time = total;
        report
            .main_replay_executor_wall_time
            .as_mut()
            .unwrap()
            .first = executor;
        report
            .main_replay_recurrent_overhead_wall_time
            .as_mut()
            .unwrap()
            .first = overhead;
        assert!(report.validate().is_ok());

        report
            .main_replay_recurrent_overhead_wall_time
            .as_mut()
            .unwrap()
            .first = BenchmarkDuration::from_duration(Duration::from_nanos(5));
        assert!(report.validate().is_err());
    }

    #[test]
    fn current_report_authenticates_module_preparation_evidence() {
        let report = zero_report();
        assert_eq!(report.main().rendered_entry_count(), 2);
        assert_eq!(report.main().loaded_module_count(), 1);
        assert_eq!(report.main().durable_artifact_cache_miss_count(), 1);
        assert_eq!(report.main().compiler_invocation_count(), 1);

        for field in [
            "rendered_entry_count",
            "loaded_module_count",
            "durable_artifact_cache_miss_count",
            "compiler_invocation_count",
        ] {
            let mut json = serde_json::to_value(zero_report()).unwrap();
            json["main"][field] = serde_json::json!(0);
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
                "tampered {field} must reject"
            );
        }
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["main"]["durable_artifact_cache_hit_count"] = serde_json::json!(1);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "one loaded artifact cannot be both a durable hit and miss"
        );

        for (hit_count, reason) in [(1, "durable hit"), (0, "in-memory reuse")] {
            let mut json = serde_json::to_value(zero_report()).unwrap();
            json["main"]["durable_artifact_cache_hit_count"] = serde_json::json!(hit_count);
            json["main"]["durable_artifact_cache_miss_count"] = serde_json::json!(0);
            json["main"]["compiler_invocation_count"] = serde_json::json!(0);
            json["prepare_compiler_process_count"] = serde_json::json!(0);
            json["prepare_max_parallel_compiler_process_count"] = serde_json::json!(0);
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_ok(),
                "{reason} module preparation must remain representable"
            );
        }
    }

    #[test]
    fn current_report_requires_exact_recurrent_traffic() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json.as_object_mut().unwrap().remove("main_replay_traffic");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut report = zero_report();
        report.main_replay_traffic = Some(NativeCpuReplayTraffic::new(2, 12, 16, 15));
        assert!(report.validate().is_err());
    }

    #[test]
    fn current_report_requires_bounded_execution_count() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json.as_object_mut()
            .unwrap()
            .remove("main_replay_executed_native_item_count");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut report = zero_report();
        report.main_replay_executed_native_item_count = Some(3);
        assert!(report.validate().is_err());

        let mut report = zero_report();
        report.main.native_item_count = 3;
        report.main.cache_miss_count = 3;
        report.schedule_cache_keys.push(23);
        report.main_replay_executed_native_item_count = Some(3);
        assert!(report.validate().is_err());
    }

    #[test]
    fn current_report_rejects_incomplete_auxiliary_inventory() {
        let mut report = zero_report();
        report.zero_grad = Some(report.main.clone());
        assert!(report.validate().is_err());
    }

    #[test]
    fn report_rejects_invented_cpu_measurements_and_throughput() {
        let mut report = zero_report();
        report.kernel_launch_count = Some(0);
        assert!(report.validate().is_err());

        let mut report = zero_report();
        let elapsed = BenchmarkDuration::from_duration(Duration::from_nanos(10));
        set_single_steady_replay_duration(&mut report, elapsed);
        assert!(report.validate().is_ok());
        let mut json = serde_json::to_value(report).unwrap();
        json["steady_microbatches_per_second"] = serde_json::json!(1.0);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );
    }

    #[test]
    fn positive_derived_rate_round_trips_exactly_and_rejects_tampering() {
        let mut report = zero_report();
        let elapsed = BenchmarkDuration::from_duration(Duration::from_nanos(63));
        set_single_steady_replay_duration(&mut report, elapsed);

        let bytes = report.to_json_bytes().unwrap();
        assert_eq!(
            NativeTrainingReport::from_json_bytes(&bytes).unwrap(),
            report
        );
        let mut json = serde_json::to_value(&report).unwrap();
        json["steady_microbatches_per_second"] = serde_json::json!(1.0);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );
    }

    #[test]
    fn report_rejects_infeasible_steady_totals() {
        let mut report = zero_report();
        let positive = BenchmarkDuration::from_duration(Duration::from_nanos(10));
        report.steady_replay_total_wall_time = positive;
        report.steady_microbatches_per_second = rate_from_total(1, positive).unwrap();
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&report).unwrap()).is_err()
        );

        let min = BenchmarkDuration::from_duration(Duration::from_nanos(10));
        let median = BenchmarkDuration::from_duration(Duration::from_nanos(20));
        let max = BenchmarkDuration::from_duration(Duration::from_nanos(30));
        report.successful_replay_count = 4;
        report.checkpoint = None;
        report.steady_replay_wall_time = BenchmarkLatencySummary {
            sample_count: 3,
            min,
            nearest_rank_p50: median,
            nearest_rank_p95: max,
            max,
        };
        for nanos in [40, 80] {
            let total = BenchmarkDuration::from_duration(Duration::from_nanos(nanos));
            report.steady_replay_total_wall_time = total;
            report.steady_microbatches_per_second = rate_from_total(3, total).unwrap();
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&report).unwrap())
                    .is_err()
            );
        }
    }
}
