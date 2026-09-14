//! Validated observational evidence for strict-native CPU compiled training.
//!
//! The runtime remains the source of preparation and replay facts. This module
//! only checks and aggregates detached reports; it does not time calls, execute
//! programs, or infer unavailable device and allocator measurements.

mod step_phases;

pub use step_phases::{
    NativeTrainingFirstStepReport, NativeTrainingStepPhase, NativeTrainingStepPhaseReport,
    NativeTrainingWarmStepReport,
};

use super::{
    CompiledAdamWCheckpoint, NativeCpuCompiledAdamWPreparationReport,
    NativeCpuCompiledAdamWStepResult, NativeCpuDispatchSegmentation,
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
const NATIVE_TRAINING_REPORT_FORMAT_V9: u32 = 9;
const NATIVE_TRAINING_REPORT_FORMAT_V10: u32 = 10;
const NATIVE_TRAINING_REPORT_FORMAT_V11: u32 = 11;
const NATIVE_TRAINING_REPORT_FORMAT_V12: u32 = 12;
const NATIVE_TRAINING_REPORT_FORMAT_V13: u32 = 13;
const NATIVE_TRAINING_REPORT_FORMAT_V14: u32 = 14;
const NATIVE_TRAINING_REPORT_FORMAT_V15: u32 = 15;
const NATIVE_TRAINING_REPORT_FORMAT_V16: u32 = 16;
pub const NATIVE_TRAINING_REPORT_FORMAT_VERSION: u32 = 17;
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compiler_process_total: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    linker_process: Option<BenchmarkDuration>,
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
            compiler_process_total: Some(BenchmarkDuration::from_duration(
                phases.compiler_process_total_wall_time(),
            )),
            linker_process: Some(BenchmarkDuration::from_duration(
                phases.linker_process_wall_time(),
            )),
            module_load: BenchmarkDuration::from_duration(phases.module_load_wall_time()),
            residual: BenchmarkDuration::from_duration(phases.residual_wall_time()),
        }
    }

    fn validate(&self, work: &NativeTrainingProgramReport, format_version: u32) -> Result<()> {
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
        let extended_timing_is_present =
            self.compiler_process_total.is_some() || self.linker_process.is_some();
        if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V14 && extended_timing_is_present {
            return Err(invalid(
                "legacy native preparation has chunk compiler timing",
            ));
        }
        let compiler_process = self
            .compiler_process
            .as_nanos()
            .map_err(|_| invalid("invalid native compiler process duration"))?;
        let compiler_process_total = match (
            format_version,
            self.compiler_process_total,
            self.linker_process,
        ) {
            (
                NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(total),
                Some(linker),
            ) => {
                let total = total
                    .as_nanos()
                    .map_err(|_| invalid("invalid native cumulative compiler duration"))?;
                let linker = linker
                    .as_nanos()
                    .map_err(|_| invalid("invalid native linker duration"))?;
                if total < compiler_process
                    || (work.compiler_invocation_count() <= 1 && total != compiler_process)
                    || linker > compiler_process
                    || (work.linker_invocation_count() == 0 && linker != 0)
                {
                    return Err(invalid("native chunk compiler timing differs"));
                }
                total
            }
            (
                NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                _,
                _,
            ) => {
                return Err(invalid("native chunk compiler timing is absent"));
            }
            (_, None, None) => compiler_process,
            _ => unreachable!("legacy extended compiler timing was rejected"),
        };
        if partitioned != total
            || (work.compiler_invocation_count == 0
                && self.compiler_process != BenchmarkDuration::from_duration(Duration::ZERO))
            || (work.compiler_invocation_count == 0 && compiler_process_total != 0)
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

    pub const fn compiler_process_total(&self) -> Option<BenchmarkDuration> {
        self.compiler_process_total
    }

    pub const fn linker_process(&self) -> Option<BenchmarkDuration> {
        self.linker_process
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
    accumulation: Option<ProgramInspection>,
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
        accumulation: Option<(u64, ExecutionPlanSummary)>,
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
            accumulation: accumulation.map(program),
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

    pub fn accumulation(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.accumulation
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    referenced_module_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unique_rendered_entry_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix_entry_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix_source_program_index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix_source_native_identity: Option<u64>,
    #[serde(default)]
    durable_artifact_cache_hit_count: u64,
    #[serde(default)]
    durable_artifact_cache_miss_count: u64,
    #[serde(default)]
    compiler_invocation_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    combined_compile_link_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    object_compile_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    linker_invocation_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preparation_timing: Option<NativeTrainingPreparationTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dispatch_segmentation: Option<NativeCpuDispatchSegmentation>,
}

impl NativeTrainingProgramReport {
    fn new(
        inspection: &ProgramInspection,
        preparation: &NativeCpuProgramPreparationReport,
        prior_native_identities: &[u64],
    ) -> Result<Self> {
        if inspection.capture_identity != preparation.capture_identity()
            || &inspection.execution_plan != preparation.execution_plan()
        {
            return Err(invalid("plan and native preparation differ"));
        }
        let plan = &inspection.execution_plan;
        let shared_prefix_source_program_index = preparation
            .work()
            .shared_prefix_source_program()
            .map(|source| count(source, "native shared-prefix source program"))
            .transpose()?;
        let shared_prefix_source_native_identity = preparation
            .work()
            .shared_prefix_source_program()
            .map(|source| {
                prior_native_identities
                    .get(source)
                    .copied()
                    .ok_or_else(|| invalid("native shared-prefix source program is not earlier"))
            })
            .transpose()?;
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
            referenced_module_count: Some(count(
                preparation.work().referenced_module_count(),
                "referenced module",
            )?),
            unique_rendered_entry_count: Some(count(
                preparation.work().unique_rendered_entry_count(),
                "unique rendered entry",
            )?),
            shared_prefix_entry_count: Some(count(
                preparation.work().shared_prefix_entry_count(),
                "shared prefix entry",
            )?),
            shared_prefix_source_program_index,
            shared_prefix_source_native_identity,
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
            combined_compile_link_count: Some(count(
                preparation.work().combined_compile_link_count(),
                "combined compiler invocation",
            )?),
            object_compile_count: Some(count(
                preparation.work().object_compile_count(),
                "object compiler invocation",
            )?),
            linker_invocation_count: Some(count(
                preparation.work().linker_invocation_count(),
                "linker invocation",
            )?),
            preparation_timing: Some(NativeTrainingPreparationTiming::from_preparation(
                preparation,
            )),
            dispatch_segmentation: Some(*preparation.dispatch_segmentation()),
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
        let prefix_evidence_is_present = self.referenced_module_count.is_some()
            || self.unique_rendered_entry_count.is_some()
            || self.shared_prefix_entry_count.is_some()
            || self.shared_prefix_source_program_index.is_some()
            || self.shared_prefix_source_native_identity.is_some();
        let chunk_compiler_evidence_is_present = self.combined_compile_link_count.is_some()
            || self.object_compile_count.is_some()
            || self.linker_invocation_count.is_some();
        let referenced_module_count = self.referenced_module_count.unwrap_or(0);
        if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V16 {
            if self.dispatch_segmentation.is_some() {
                return Err(invalid(
                    "legacy native program has dispatch segmentation evidence",
                ));
            }
        } else {
            self.dispatch_segmentation
                .as_ref()
                .ok_or_else(|| invalid("native dispatch segmentation evidence is absent"))?
                .authenticates(self.rendered_entry_count, referenced_module_count)
                .then_some(())
                .ok_or_else(|| invalid("native dispatch segmentation evidence differs"))?;
        }
        if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V14 && chunk_compiler_evidence_is_present
        {
            return Err(invalid("legacy native program has chunk compiler evidence"));
        }
        let unique_rendered_entry_count = self.unique_rendered_entry_count.unwrap_or(0);
        let shared_prefix_entry_count = self.shared_prefix_entry_count.unwrap_or(0);
        let preparation = [
            self.rendered_entry_count,
            self.loaded_module_count,
            self.durable_artifact_cache_hit_count,
            self.durable_artifact_cache_miss_count,
            self.compiler_invocation_count,
        ];
        if format_version < NATIVE_TRAINING_REPORT_FORMAT_V5 {
            if preparation.into_iter().any(|value| value != 0) || prefix_evidence_is_present {
                return Err(invalid(
                    "legacy native program has module preparation evidence",
                ));
            }
        } else if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V13 {
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
            if prefix_evidence_is_present {
                return Err(invalid("legacy native program has prefix-module evidence"));
            }
        } else {
            if self.referenced_module_count.is_none()
                || self.unique_rendered_entry_count.is_none()
                || self.shared_prefix_entry_count.is_none()
            {
                return Err(invalid("native program prefix-module evidence is absent"));
            }
            let durable_access_count = self
                .durable_artifact_cache_hit_count
                .checked_add(self.durable_artifact_cache_miss_count)
                .ok_or_else(|| invalid("durable artifact cache count overflows"))?;
            let rendered_inventory_is_valid = self.rendered_entry_count <= self.native_item_count
                && (self.rendered_entry_count == 0) == (self.native_item_count == 0);
            let prefix_partition = self
                .unique_rendered_entry_count()
                .checked_add(self.shared_prefix_entry_count())
                .ok_or_else(|| invalid("native prefix entry count overflows"))?;
            if !rendered_inventory_is_valid
                || prefix_partition != self.rendered_entry_count
                || self.cache_hit_count < shared_prefix_entry_count
                || self.loaded_module_count != u64::from(unique_rendered_entry_count != 0)
                || (self.rendered_entry_count == 0) != (referenced_module_count == 0)
                || (shared_prefix_entry_count == 0)
                    != self.shared_prefix_source_program_index.is_none()
                || (shared_prefix_entry_count == 0)
                    != self.shared_prefix_source_native_identity.is_none()
                || referenced_module_count
                    != self
                        .loaded_module_count
                        .checked_add(u64::from(shared_prefix_entry_count != 0))
                        .ok_or_else(|| invalid("native module reference count overflows"))?
                || durable_access_count > self.loaded_module_count
            {
                return Err(invalid(
                    "native program prefix-module preparation evidence differs",
                ));
            }
            if format_version == NATIVE_TRAINING_REPORT_FORMAT_V14 {
                if self.compiler_invocation_count != self.durable_artifact_cache_miss_count {
                    return Err(invalid(
                        "native program prefix-module preparation evidence differs",
                    ));
                }
            } else {
                let (Some(combined), Some(objects), Some(linker)) = (
                    self.combined_compile_link_count,
                    self.object_compile_count,
                    self.linker_invocation_count,
                ) else {
                    return Err(invalid("native program chunk compiler evidence is absent"));
                };
                let compiler_invocations = combined
                    .checked_add(objects)
                    .and_then(|count| count.checked_add(linker))
                    .ok_or_else(|| invalid("native compiler invocation count overflows"))?;
                let compiler_mode_is_valid = if self.durable_artifact_cache_miss_count == 0 {
                    compiler_invocations == 0
                } else {
                    crate::cpu_jit::NativeScheduleModuleBuildMode::for_unique_rendered_entry_count(
                        unique_rendered_entry_count,
                    )
                    .is_some_and(|mode| mode.process_inventory() == (combined, objects, linker))
                };
                if compiler_invocations != self.compiler_invocation_count
                    || objects > unique_rendered_entry_count
                    || !compiler_mode_is_valid
                {
                    return Err(invalid("native program chunk compiler evidence differs"));
                }
            }
        }
        match (format_version, &self.preparation_timing) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, None) => {}
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
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(timing),
            ) => timing.validate(self, format_version)?,
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

    pub const fn referenced_module_count(&self) -> u64 {
        match self.referenced_module_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn unique_rendered_entry_count(&self) -> u64 {
        match self.unique_rendered_entry_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn shared_prefix_entry_count(&self) -> u64 {
        match self.shared_prefix_entry_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn shared_prefix_source_program_index(&self) -> Option<u64> {
        self.shared_prefix_source_program_index
    }

    pub const fn shared_prefix_source_native_identity(&self) -> Option<u64> {
        self.shared_prefix_source_native_identity
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

    pub const fn combined_compile_link_count(&self) -> u64 {
        match self.combined_compile_link_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn object_compile_count(&self) -> u64 {
        match self.object_compile_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn linker_invocation_count(&self) -> u64 {
        match self.linker_invocation_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn preparation_timing(&self) -> Option<&NativeTrainingPreparationTiming> {
        self.preparation_timing.as_ref()
    }

    pub const fn dispatch_segmentation(&self) -> Option<&NativeCpuDispatchSegmentation> {
        self.dispatch_segmentation.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NativeCompilerAggregate {
    process_count: u64,
    module_job_count: u64,
    effective_wall_sum: u128,
    effective_wall_max: u128,
    cumulative_wall_sum: u128,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NativeRenderAggregate {
    job_count: u64,
    wall_sum: u128,
    wall_max: u128,
}

impl NativeRenderAggregate {
    fn from_programs<'a>(
        programs: impl IntoIterator<Item = &'a NativeTrainingProgramReport>,
    ) -> Result<Self> {
        programs
            .into_iter()
            .try_fold(Self::default(), |aggregate, program| {
                let render = program
                    .preparation_timing
                    .as_ref()
                    .ok_or_else(|| invalid("native program preparation timing is absent"))?
                    .render
                    .as_nanos()
                    .map_err(|_| invalid("invalid native render duration"))?;
                Ok(Self {
                    job_count: aggregate
                        .job_count
                        .checked_add(1)
                        .ok_or_else(|| invalid("native render job count overflows"))?,
                    wall_sum: aggregate
                        .wall_sum
                        .checked_add(render)
                        .ok_or_else(|| invalid("native render duration overflows"))?,
                    wall_max: aggregate.wall_max.max(render),
                })
            })
    }
}

impl NativeCompilerAggregate {
    fn from_programs<'a>(
        programs: impl IntoIterator<Item = &'a NativeTrainingProgramReport>,
        format_version: u32,
    ) -> Result<Self> {
        programs
            .into_iter()
            .try_fold(Self::default(), |sum, program| {
                let process_count = sum
                    .process_count
                    .checked_add(program.compiler_invocation_count)
                    .ok_or_else(|| invalid("native compiler process count overflows"))?;
                let jobs = program
                    .durable_artifact_cache_hit_count
                    .checked_add(program.durable_artifact_cache_miss_count)
                    .ok_or_else(|| invalid("native module job count overflows"))?;
                let module_job_count = sum
                    .module_job_count
                    .checked_add(jobs)
                    .ok_or_else(|| invalid("native module job count overflows"))?;
                let timing = program
                    .preparation_timing
                    .as_ref()
                    .ok_or_else(|| invalid("native program preparation timing is absent"))?;
                let effective = timing
                    .compiler_process
                    .as_nanos()
                    .map_err(|_| invalid("invalid native compiler process duration"))?;
                let cumulative = if format_version >= NATIVE_TRAINING_REPORT_FORMAT_V15 {
                    timing
                        .compiler_process_total
                        .ok_or_else(|| invalid("native cumulative compiler timing is absent"))?
                        .as_nanos()
                        .map_err(|_| invalid("invalid native cumulative compiler duration"))?
                } else {
                    effective
                };
                Ok(Self {
                    process_count,
                    module_job_count,
                    effective_wall_sum: sum
                        .effective_wall_sum
                        .checked_add(effective)
                        .ok_or_else(|| invalid("native compiler process duration overflows"))?,
                    effective_wall_max: sum.effective_wall_max.max(effective),
                    cumulative_wall_sum: sum
                        .cumulative_wall_sum
                        .checked_add(cumulative)
                        .ok_or_else(|| invalid("native cumulative compiler duration overflows"))?,
                })
            })
    }

    fn internal_overlap(self) -> Result<u128> {
        self.cumulative_wall_sum
            .checked_sub(self.effective_wall_sum)
            .ok_or_else(|| invalid("native internal compiler overlap underflows"))
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
    overhead: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayRecordingMode {
    Unset,
    Raw,
    Phased,
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
    prepare_parallel_render_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_max_parallel_render_job_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_max_parallel_compiler_process_count: Option<u64>,
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
        let render_overlap = match (
            self.format_version,
            self.prepare_parallel_render_overlap_wall_time,
            self.prepare_max_parallel_render_job_count,
        ) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V15, None, None) => 0,
            (
                NATIVE_TRAINING_REPORT_FORMAT_V16 | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
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
                if max_parallel == 0
                    || max_parallel > 2
                    || max_parallel > aggregate.job_count
                    || overlap > maximum_overlap
                    || (overlap == 0) != (max_parallel == 1)
                {
                    return Err(invalid("native parallel render evidence differs"));
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
        match (self.format_version, &self.step_phases) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V9, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11
                | NATIVE_TRAINING_REPORT_FORMAT_V12
                | NATIVE_TRAINING_REPORT_FORMAT_V13
                | NATIVE_TRAINING_REPORT_FORMAT_V14,
                Some(phases),
            ) => phases.validate(
                self.successful_replay_count,
                self.first_replay_wall_time,
                self.steady_replay_total_wall_time,
                self.main_replay_executor_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                self.main_replay_recurrent_overhead_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay overhead timing is absent"))?,
            )?,
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V9, Some(_)) => {
                return Err(invalid("legacy native training report has step phases"));
            }
            (
                NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(phases),
            ) => phases.validate(
                self.successful_replay_count,
                self.first_replay_wall_time,
                self.steady_replay_total_wall_time,
                self.main_replay_executor_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                self.main_replay_recurrent_overhead_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay overhead timing is absent"))?,
            )?,
            (
                NATIVE_TRAINING_REPORT_FORMAT_V15
                | NATIVE_TRAINING_REPORT_FORMAT_V16
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
    prepare_wall_time: Duration,
    prepare_runtime_overhead_wall_time: Duration,
    prepare_parallel_module_overlap_wall_time: Duration,
    prepare_parallel_render_overlap_wall_time: Duration,
    prepare_max_parallel_render_job_count: u64,
    prepare_compiler_process_overlap_wall_time: Duration,
    prepare_compiler_process_count: u64,
    prepare_max_parallel_compiler_process_count: u64,
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
    /// preparation. `prepare_wall_time` is caller-observed around the whole
    /// target preparation and must contain every attached program's measured
    /// preparation time.
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
            prepare_wall_time,
            prepare_runtime_overhead_wall_time,
            prepare_parallel_module_overlap_wall_time,
            prepare_parallel_render_overlap_wall_time,
            prepare_max_parallel_render_job_count,
            prepare_compiler_process_overlap_wall_time: preparation
                .compiler_process_overlap_wall_time(),
            prepare_compiler_process_count,
            prepare_max_parallel_compiler_process_count,
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
        let overhead = total
            .checked_sub(executor)
            .ok_or_else(|| invalid("native executor time exceeds replay time"))?;
        Ok((
            ReplayTiming {
                total,
                executor,
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
            compiler_process_total: None,
            linker_process: None,
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
        let phase = report
            .step_phases
            .as_mut()
            .unwrap()
            .warm_optimizer_commit
            .as_mut()
            .unwrap();
        phase.total_wall_time = elapsed;
        phase.wall_time = report.steady_replay_wall_time.clone();
        phase.steps_per_second = report.steady_microbatches_per_second;
        phase.executor_total_wall_time = elapsed;
        phase.executor_wall_time = report.steady_replay_wall_time.clone();
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

    fn remove_step_phases(json: &mut serde_json::Value) {
        json.as_object_mut().unwrap().remove("step_phases");
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

    fn remove_prefix_module_evidence(json: &mut serde_json::Value) {
        for program in [
            "main",
            "accumulation",
            "partial_flush",
            "zero_grad",
            "evaluation",
        ] {
            let Some(program) = json[program].as_object_mut() else {
                continue;
            };
            for field in [
                "referenced_module_count",
                "unique_rendered_entry_count",
                "shared_prefix_entry_count",
                "shared_prefix_source_program_index",
                "shared_prefix_source_native_identity",
            ] {
                program.remove(field);
            }
        }
    }

    fn remove_chunk_compiler_evidence(json: &mut serde_json::Value) {
        for program in [
            "main",
            "accumulation",
            "partial_flush",
            "zero_grad",
            "evaluation",
        ] {
            let Some(program) = json[program].as_object_mut() else {
                continue;
            };
            for field in [
                "combined_compile_link_count",
                "object_compile_count",
                "linker_invocation_count",
            ] {
                program.remove(field);
            }
            if let Some(timing) = program["preparation_timing"].as_object_mut() {
                timing.remove("compiler_process_total");
                timing.remove("linker_process");
            }
        }
    }

    fn remove_parallel_render_evidence(json: &mut serde_json::Value) {
        json.as_object_mut()
            .unwrap()
            .remove("prepare_parallel_render_overlap_wall_time");
        json.as_object_mut()
            .unwrap()
            .remove("prepare_max_parallel_render_job_count");
    }

    fn remove_dispatch_segmentation_evidence(json: &mut serde_json::Value) {
        for program in [
            "main",
            "accumulation",
            "partial_flush",
            "zero_grad",
            "evaluation",
        ] {
            if let Some(program) = json[program].as_object_mut() {
                program.remove("dispatch_segmentation");
            }
        }
    }

    fn zero_report() -> NativeTrainingReport {
        NativeTrainingReport {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_V10,
            compile_wall_time: zero_duration(),
            prepare_wall_time: zero_duration(),
            prepare_runtime_overhead_wall_time: Some(zero_duration()),
            prepare_parallel_module_overlap_wall_time: Some(zero_duration()),
            prepare_parallel_render_overlap_wall_time: None,
            prepare_max_parallel_render_job_count: None,
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
                referenced_module_count: None,
                unique_rendered_entry_count: None,
                shared_prefix_entry_count: None,
                shared_prefix_source_program_index: None,
                shared_prefix_source_native_identity: None,
                durable_artifact_cache_hit_count: 0,
                durable_artifact_cache_miss_count: 1,
                compiler_invocation_count: 1,
                combined_compile_link_count: None,
                object_compile_count: None,
                linker_invocation_count: None,
                preparation_timing: Some(zero_preparation_timing()),
                dispatch_segmentation: None,
            },
            accumulation: None,
            partial_flush: None,
            zero_grad: None,
            evaluation: None,
            recurrent_logical_state_count: 4,
            recurrent_logical_state_bytes: 16,
            main_replay_traffic: Some(NativeCpuReplayTraffic::new(2, 12, 16, 16)),
            main_replay_executed_native_item_count: Some(1),
            accumulation_replay_traffic: None,
            accumulation_replay_executed_native_item_count: None,
            main_replay_executor_wall_time: Some(zero_replay_timing()),
            main_replay_recurrent_overhead_wall_time: Some(zero_replay_timing()),
            step_phases: Some(NativeTrainingStepPhaseReport {
                first: NativeTrainingFirstStepReport {
                    phase: NativeTrainingStepPhase::AccumulationOnly,
                    total_wall_time: zero_duration(),
                    executor_wall_time: zero_duration(),
                    recurrent_overhead_wall_time: zero_duration(),
                },
                warm_accumulation_only: None,
                warm_optimizer_commit: Some(NativeTrainingWarmStepReport {
                    total_wall_time: zero_duration(),
                    wall_time: BenchmarkLatencySummary {
                        sample_count: 1,
                        min: zero_duration(),
                        nearest_rank_p50: zero_duration(),
                        nearest_rank_p95: zero_duration(),
                        max: zero_duration(),
                    },
                    steps_per_second: None,
                    executor_total_wall_time: zero_duration(),
                    executor_wall_time: BenchmarkLatencySummary {
                        sample_count: 1,
                        min: zero_duration(),
                        nearest_rank_p50: zero_duration(),
                        nearest_rank_p95: zero_duration(),
                        max: zero_duration(),
                    },
                    recurrent_overhead_total_wall_time: zero_duration(),
                    recurrent_overhead_wall_time: BenchmarkLatencySummary {
                        sample_count: 1,
                        min: zero_duration(),
                        nearest_rank_p50: zero_duration(),
                        nearest_rank_p95: zero_duration(),
                        max: zero_duration(),
                    },
                }),
            }),
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
            accumulation_schedule_cache_keys: Vec::new(),
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

    fn phase_specialized_report() -> NativeTrainingReport {
        let mut report = zero_report();
        report.format_version = NATIVE_TRAINING_REPORT_FORMAT_VERSION;
        report.prepare_parallel_render_overlap_wall_time = Some(zero_duration());
        report.prepare_max_parallel_render_job_count = Some(1);
        report.main_replay_traffic = report
            .main_replay_traffic
            .map(|traffic| traffic.with_materialized_egress(5, 24));
        report.main.referenced_module_count = Some(1);
        report.main.unique_rendered_entry_count = Some(2);
        report.main.shared_prefix_entry_count = Some(0);
        report.main.combined_compile_link_count = Some(1);
        report.main.object_compile_count = Some(0);
        report.main.linker_invocation_count = Some(0);
        report.main.dispatch_segmentation = Some(NativeCpuDispatchSegmentation {
            segment_count: 1,
            dispatch_reached_module_count: 1,
            terminal_segment_count: 1,
            non_dispatch_boundary_count: 0,
            module_change_count: 0,
            output_slot_alias_count: 0,
            derived_slot_dependency_count: 0,
        });
        let timing = report.main.preparation_timing.as_mut().unwrap();
        timing.compiler_process_total = Some(timing.compiler_process);
        timing.linker_process = Some(zero_duration());
        let mut accumulation = report.main.clone();
        accumulation.capture_identity = 8;
        accumulation.native_identity = 12;
        accumulation.execution_plan_identity = 14;
        accumulation.cache_hit_count = 1;
        accumulation.cache_miss_count = 1;
        accumulation.referenced_module_count = Some(2);
        accumulation.unique_rendered_entry_count = Some(1);
        accumulation.shared_prefix_entry_count = Some(1);
        accumulation.shared_prefix_source_program_index = Some(0);
        accumulation.shared_prefix_source_native_identity = Some(report.main.native_identity);
        accumulation.dispatch_segmentation = Some(NativeCpuDispatchSegmentation {
            segment_count: 2,
            dispatch_reached_module_count: 2,
            terminal_segment_count: 1,
            non_dispatch_boundary_count: 0,
            module_change_count: 1,
            output_slot_alias_count: 0,
            derived_slot_dependency_count: 0,
        });
        report.accumulation = Some(accumulation);
        report.accumulation_replay_traffic = Some(
            NativeCpuReplayTraffic::new(2, 12, 16, 8)
                .with_recurrent_inventory(2, 8, 2, 8)
                .with_materialized_egress(1, 4),
        );
        report.accumulation_replay_executed_native_item_count =
            report.main_replay_executed_native_item_count;
        report.accumulation_schedule_cache_keys = vec![23, 29];
        report.prepare_compiler_process_count = Some(2);
        report
    }

    fn set_main_unique_rendered_entry_count(report: &mut NativeTrainingReport, count: u64) {
        let count_usize = usize::try_from(count).unwrap();
        report.main.logical_schedule_item_count = count;
        report.main.native_item_count = count;
        report.main.cache_hit_count = 0;
        report.main.cache_miss_count = count;
        report.main.rendered_entry_count = count;
        report.main.unique_rendered_entry_count = Some(count);
        report.schedule_cache_keys = (0..count_usize)
            .map(|index| u64::try_from(index).unwrap() + 101)
            .collect();
    }

    #[test]
    fn phase_specialized_report_round_trips_and_authenticates_both_programs() {
        let report = phase_specialized_report();
        let bytes = report.to_json_bytes().unwrap();
        let decoded = NativeTrainingReport::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded, report);
        assert_eq!(
            decoded.format_version,
            NATIVE_TRAINING_REPORT_FORMAT_VERSION
        );
        assert_eq!(decoded.accumulation().unwrap().capture_identity(), 8);
        assert_eq!(decoded.accumulation_schedule_cache_keys(), [23, 29]);

        let mut json = serde_json::to_value(&report).unwrap();
        json["accumulation_schedule_cache_keys"] = serde_json::json!([23]);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut json = serde_json::to_value(&report).unwrap();
        json["accumulation_replay_executed_native_item_count"] = serde_json::json!(3);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut json = serde_json::to_value(&report).unwrap();
        let main_identity = json["main"]["capture_identity"].clone();
        json["accumulation"]["capture_identity"] = main_identity;
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut json = serde_json::to_value(&report).unwrap();
        json["accumulation_replay_traffic"]["borrowed_recurrent_output_bytes"] =
            serde_json::json!(15);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut json = serde_json::to_value(&report).unwrap();
        json["accumulation_replay_traffic"]["retained_recurrent_state_count"] =
            serde_json::json!(1);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        for field in ["materialized_egress_count", "materialized_egress_bytes"] {
            let mut json = serde_json::to_value(&report).unwrap();
            json["main_replay_traffic"][field] = serde_json::json!(0);
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
            );

            let mut json = serde_json::to_value(&report).unwrap();
            json["main_replay_traffic"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
            );
        }

        let mut json = serde_json::to_value(&report).unwrap();
        let traffic = json["main_replay_traffic"].as_object_mut().unwrap();
        traffic.remove("materialized_egress_count");
        traffic.remove("materialized_egress_bytes");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );
    }

    #[test]
    fn phase_specialized_v11_report_decodes_without_recurrent_retention_inventory() {
        let report = phase_specialized_report();
        let mut json = serde_json::to_value(report).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V11);
        remove_dispatch_segmentation_evidence(&mut json);
        remove_parallel_render_evidence(&mut json);
        remove_prefix_module_evidence(&mut json);
        remove_chunk_compiler_evidence(&mut json);
        for phase in ["main_replay_traffic", "accumulation_replay_traffic"] {
            let traffic = json[phase].as_object_mut().unwrap();
            traffic.remove("retained_recurrent_state_count");
            traffic.remove("retained_recurrent_state_bytes");
            traffic.remove("replaced_recurrent_state_count");
            traffic.remove("replaced_recurrent_state_bytes");
            traffic.remove("materialized_egress_count");
            traffic.remove("materialized_egress_bytes");
        }
        json["accumulation_replay_traffic"]["borrowed_recurrent_output_bytes"] =
            serde_json::json!(16);
        let decoded =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V11);
        assert_eq!(
            decoded
                .accumulation_replay_traffic()
                .unwrap()
                .retained_recurrent_state_count(),
            0
        );
    }

    #[test]
    fn phase_specialized_v12_report_decodes_without_egress_evidence() {
        let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V12);
        remove_dispatch_segmentation_evidence(&mut json);
        remove_parallel_render_evidence(&mut json);
        remove_prefix_module_evidence(&mut json);
        remove_chunk_compiler_evidence(&mut json);
        for phase in ["main_replay_traffic", "accumulation_replay_traffic"] {
            let traffic = json[phase].as_object_mut().unwrap();
            traffic.remove("materialized_egress_count");
            traffic.remove("materialized_egress_bytes");
        }
        let decoded =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V12);
        assert_eq!(
            decoded
                .main_replay_traffic()
                .unwrap()
                .materialized_egress_count(),
            0
        );
        assert_eq!(
            decoded
                .accumulation_replay_traffic()
                .unwrap()
                .materialized_egress_bytes(),
            0
        );
    }

    #[test]
    fn version_thirteen_report_decodes_without_prefix_module_evidence() {
        let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V13);
        remove_dispatch_segmentation_evidence(&mut json);
        remove_parallel_render_evidence(&mut json);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v13 cannot claim v14 prefix-module evidence"
        );
        for program in ["main", "accumulation"] {
            let program = json[program].as_object_mut().unwrap();
            for field in [
                "referenced_module_count",
                "unique_rendered_entry_count",
                "shared_prefix_entry_count",
            ] {
                program.insert(field.into(), serde_json::json!(0));
            }
            program.remove("shared_prefix_source_program_index");
            program.remove("shared_prefix_source_native_identity");
        }
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v13 rejects even zero-valued v14 prefix-module fields"
        );
        remove_prefix_module_evidence(&mut json);
        remove_chunk_compiler_evidence(&mut json);
        for program in ["main", "accumulation"] {
            let program = json[program].as_object_mut().unwrap();
            program.insert("loaded_module_count".into(), serde_json::json!(1));
        }
        let decoded =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V13);
        assert_eq!(decoded.main().referenced_module_count(), 0);
        assert_eq!(
            decoded.accumulation().unwrap().shared_prefix_entry_count(),
            0
        );
    }

    #[test]
    fn version_fourteen_preserves_prefix_evidence_and_rejects_v15_compiler_fields() {
        let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V14);
        remove_dispatch_segmentation_evidence(&mut json);
        remove_parallel_render_evidence(&mut json);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v14 cannot claim v15 chunk compiler evidence"
        );
        for program in ["main", "accumulation"] {
            json[program]["combined_compile_link_count"] = serde_json::json!(0);
            json[program]["object_compile_count"] = serde_json::json!(0);
            json[program]["linker_invocation_count"] = serde_json::json!(0);
        }
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v14 rejects even zero-valued v15 chunk compiler fields"
        );
        remove_chunk_compiler_evidence(&mut json);
        let decoded =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V14);
        assert_eq!(decoded.main().referenced_module_count(), 1);
        assert_eq!(
            decoded.accumulation().unwrap().shared_prefix_entry_count(),
            1
        );
        assert_eq!(decoded.main().object_compile_count(), 0);
    }

    #[test]
    fn current_report_authenticates_chunked_compile_and_link_inventory() {
        let mut report = phase_specialized_report();
        set_main_unique_rendered_entry_count(&mut report, 682);
        report.main.combined_compile_link_count = Some(0);
        report.main.object_compile_count = Some(2);
        report.main.linker_invocation_count = Some(1);
        report.main.compiler_invocation_count = 3;
        report.prepare_compiler_process_count = Some(4);
        report.prepare_max_parallel_compiler_process_count = Some(2);
        report.prepare_compiler_process_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
        report.prepare_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        let timing = report.main.preparation_timing.as_mut().unwrap();
        timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        timing.compiler_process = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        timing.compiler_process_total =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(3)));
        timing.linker_process = Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
        assert!(report.validate().is_ok());

        let valid = report.clone();

        let mut direct = phase_specialized_report();
        set_main_unique_rendered_entry_count(&mut direct, 512);
        assert!(direct.validate().is_ok());
        set_main_unique_rendered_entry_count(&mut direct, 682);
        assert!(
            direct.validate().is_err(),
            "an oversized unique suffix cannot claim combined compilation"
        );

        let mut small_chunked = valid.clone();
        set_main_unique_rendered_entry_count(&mut small_chunked, 512);
        assert!(
            small_chunked.validate().is_err(),
            "a bounded unique suffix cannot claim chunked compilation"
        );

        report.main.combined_compile_link_count = Some(1);
        assert!(report.validate().is_err(), "compiler modes cannot be mixed");

        let mut report = valid.clone();
        report.main.linker_invocation_count = Some(0);
        assert!(
            report.validate().is_err(),
            "chunked compilation must link once"
        );

        let mut report = valid.clone();
        report
            .accumulation
            .as_mut()
            .unwrap()
            .preparation_timing
            .as_mut()
            .unwrap()
            .compiler_process_total =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
        assert!(
            report.validate().is_err(),
            "one compiler process has no internal overlap"
        );

        let mut json = serde_json::to_value(valid).unwrap();
        json["main"]
            .as_object_mut()
            .unwrap()
            .remove("object_compile_count");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v15 requires complete chunk compiler evidence"
        );
    }

    #[test]
    fn version_fifteen_decodes_without_parallel_render_evidence() {
        let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V15);
        remove_dispatch_segmentation_evidence(&mut json);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v15 cannot claim v16 parallel render evidence"
        );
        remove_parallel_render_evidence(&mut json);
        let decoded =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V15);
        assert!(
            decoded
                .prepare_parallel_render_overlap_wall_time()
                .is_none()
        );
        assert!(decoded.prepare_max_parallel_render_job_count().is_none());
    }

    #[test]
    fn version_sixteen_rejects_dispatch_segmentation_even_when_zero() {
        let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V16);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v16 cannot claim v17 dispatch segmentation evidence"
        );
        for program in ["main", "accumulation"] {
            json[program]["dispatch_segmentation"] = serde_json::json!({
                "segment_count": 0,
                "dispatch_reached_module_count": 0,
                "terminal_segment_count": 0,
                "non_dispatch_boundary_count": 0,
                "module_change_count": 0,
                "output_slot_alias_count": 0,
                "derived_slot_dependency_count": 0,
            });
        }
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "v16 rejects explicit zero-valued v17 evidence"
        );
        remove_dispatch_segmentation_evidence(&mut json);
        let decoded =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V16);
        assert!(decoded.main().dispatch_segmentation().is_none());
    }

    #[test]
    fn current_report_authenticates_dispatch_segmentation_partition() {
        let report = phase_specialized_report();
        let segmentation = report.main().dispatch_segmentation().unwrap();
        assert_eq!(segmentation.segment_count(), 1);
        assert_eq!(segmentation.dispatch_reached_module_count(), 1);
        assert_eq!(segmentation.terminal_segment_count(), 1);
        assert_eq!(segmentation.non_dispatch_boundary_count(), 0);
        assert_eq!(segmentation.module_change_count(), 0);
        assert_eq!(segmentation.output_slot_alias_count(), 0);
        assert_eq!(segmentation.derived_slot_dependency_count(), 0);
        let accumulation = report
            .accumulation()
            .unwrap()
            .dispatch_segmentation()
            .unwrap();
        assert_eq!(accumulation.segment_count(), 2);
        assert_eq!(accumulation.dispatch_reached_module_count(), 2);
        assert_eq!(accumulation.terminal_segment_count(), 1);
        assert_eq!(accumulation.non_dispatch_boundary_count(), 0);
        assert_eq!(accumulation.module_change_count(), 1);
        assert_eq!(report.accumulation().unwrap().referenced_module_count(), 2);
        assert!(report.validate().is_ok());

        let all_elided = NativeCpuDispatchSegmentation {
            segment_count: 0,
            dispatch_reached_module_count: 0,
            terminal_segment_count: 0,
            non_dispatch_boundary_count: 0,
            module_change_count: 0,
            output_slot_alias_count: 0,
            derived_slot_dependency_count: 0,
        };
        assert!(
            all_elided.authenticates(1, 1),
            "structural validation permits rendered work omitted by authenticated elision"
        );

        let elided_only_suffix = NativeCpuDispatchSegmentation {
            segment_count: 1,
            dispatch_reached_module_count: 1,
            terminal_segment_count: 1,
            non_dispatch_boundary_count: 0,
            module_change_count: 0,
            output_slot_alias_count: 0,
            derived_slot_dependency_count: 0,
        };
        assert!(
            elided_only_suffix.authenticates(2, 2),
            "a referenced suffix module may contain only elided entries"
        );

        let mut missing_terminal = report.clone();
        let segmentation = missing_terminal
            .main
            .dispatch_segmentation
            .as_mut()
            .unwrap();
        segmentation.terminal_segment_count = 0;
        segmentation.output_slot_alias_count = 1;
        assert!(
            missing_terminal.validate().is_err(),
            "partition-preserving evidence must retain one terminal segment"
        );

        let mut fallback_boundary = report.clone();
        let segmentation = fallback_boundary
            .accumulation
            .as_mut()
            .unwrap()
            .dispatch_segmentation
            .as_mut()
            .unwrap();
        segmentation.non_dispatch_boundary_count = 1;
        segmentation.module_change_count = 0;
        assert!(
            fallback_boundary.validate().is_err(),
            "strict-native evidence cannot replace a module change with a fallback boundary"
        );

        let mut missing_module_change = report.clone();
        let segmentation = missing_module_change
            .accumulation
            .as_mut()
            .unwrap()
            .dispatch_segmentation
            .as_mut()
            .unwrap();
        segmentation.module_change_count = 0;
        segmentation.derived_slot_dependency_count = 1;
        assert!(
            missing_module_change.validate().is_err(),
            "two referenced modules require one module-change segment while preserving the partition"
        );

        let mut one_module_two_segments = report.clone();
        let segmentation = one_module_two_segments
            .main
            .dispatch_segmentation
            .as_mut()
            .unwrap();
        segmentation.segment_count = 2;
        segmentation.output_slot_alias_count = 1;
        assert!(one_module_two_segments.validate().is_ok());
        let segmentation = one_module_two_segments
            .main
            .dispatch_segmentation
            .as_mut()
            .unwrap();
        segmentation.dispatch_reached_module_count = 2;
        segmentation.module_change_count = 1;
        segmentation.output_slot_alias_count = 0;
        assert!(
            one_module_two_segments.validate().is_err(),
            "one referenced module cannot claim a partition-preserving module change"
        );

        let mut empty_tape_with_reached_module = report.clone();
        empty_tape_with_reached_module.main.dispatch_segmentation =
            Some(NativeCpuDispatchSegmentation {
                dispatch_reached_module_count: 1,
                ..all_elided
            });
        assert!(
            empty_tape_with_reached_module.validate().is_err(),
            "an empty dispatch tape cannot claim a reached module"
        );

        let mut missing = serde_json::to_value(&report).unwrap();
        missing["main"]
            .as_object_mut()
            .unwrap()
            .remove("dispatch_segmentation");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&missing).unwrap()).is_err(),
            "v17 requires dispatch segmentation evidence"
        );

        let mut missing_reached_modules = serde_json::to_value(&report).unwrap();
        missing_reached_modules["main"]["dispatch_segmentation"]
            .as_object_mut()
            .unwrap()
            .remove("dispatch_reached_module_count");
        assert!(
            NativeTrainingReport::from_json_bytes(
                &serde_json::to_vec(&missing_reached_modules).unwrap()
            )
            .is_err(),
            "v17 requires the dispatch-reached module inventory"
        );
    }

    #[test]
    fn current_report_authenticates_parallel_render_overlap_and_bound() {
        let mut report = phase_specialized_report();
        {
            let timing = report.main.preparation_timing.as_mut().unwrap();
            timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(2));
            timing.render = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        }
        {
            let timing = report
                .accumulation
                .as_mut()
                .unwrap()
                .preparation_timing
                .as_mut()
                .unwrap();
            timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(2));
            timing.render = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        }
        report.prepare_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(3));
        report.prepare_parallel_render_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
        report.prepare_max_parallel_render_job_count = Some(2);
        assert!(report.validate().is_ok());

        let valid = report.clone();
        report.prepare_max_parallel_render_job_count = Some(3);
        assert!(report.validate().is_err());

        let mut report = valid.clone();
        report.prepare_parallel_render_overlap_wall_time =
            Some(BenchmarkDuration::from_duration(Duration::from_nanos(3)));
        assert!(report.validate().is_err());

        let mut report = valid;
        report.prepare_parallel_render_overlap_wall_time = None;
        assert!(report.validate().is_err());
    }

    #[test]
    fn current_report_authenticates_exact_prefix_source_and_partition() {
        let report = phase_specialized_report();
        assert_eq!(report.main().unique_rendered_entry_count(), 2);
        assert_eq!(report.main().shared_prefix_entry_count(), 0);
        let accumulation = report.accumulation().unwrap();
        assert_eq!(accumulation.unique_rendered_entry_count(), 1);
        assert_eq!(accumulation.shared_prefix_entry_count(), 1);
        assert_eq!(accumulation.referenced_module_count(), 2);
        assert_eq!(accumulation.shared_prefix_source_program_index(), Some(0));
        assert_eq!(
            accumulation.shared_prefix_source_native_identity(),
            Some(report.main().native_identity())
        );
        assert!(report.validate().is_ok());

        let mut equal_identity = report.clone();
        let main_native_identity = equal_identity.main.native_identity;
        equal_identity
            .accumulation
            .as_mut()
            .unwrap()
            .native_identity = main_native_identity;
        assert!(
            equal_identity.validate().is_ok(),
            "the earlier program index disambiguates equal native identities"
        );

        let mut full_prefix = report.clone();
        let source_segmentation = full_prefix.main.dispatch_segmentation.unwrap();
        let accumulation_native_identity = {
            let accumulation = full_prefix.accumulation.as_mut().unwrap();
            accumulation.cache_hit_count = 2;
            accumulation.cache_miss_count = 0;
            accumulation.unique_rendered_entry_count = Some(0);
            accumulation.shared_prefix_entry_count = Some(2);
            accumulation.loaded_module_count = 0;
            accumulation.referenced_module_count = Some(1);
            accumulation.durable_artifact_cache_miss_count = 0;
            accumulation.compiler_invocation_count = 0;
            accumulation.combined_compile_link_count = Some(0);
            accumulation.dispatch_segmentation = Some(source_segmentation);
            accumulation.native_identity()
        };
        full_prefix.prepare_compiler_process_count = Some(1);
        assert!(
            full_prefix.validate().is_ok(),
            "a full exact prefix has no suffix compilation or module load"
        );
        assert_eq!(
            full_prefix
                .accumulation
                .as_ref()
                .unwrap()
                .dispatch_segmentation,
            full_prefix.main.dispatch_segmentation,
            "a full prefix reuses the source program's sealed segmentation"
        );

        for (field, value) in [
            ("unique_rendered_entry_count", serde_json::json!(0)),
            ("shared_prefix_entry_count", serde_json::json!(2)),
            ("referenced_module_count", serde_json::json!(0)),
            ("shared_prefix_source_program_index", serde_json::json!(1)),
            (
                "shared_prefix_source_native_identity",
                serde_json::json!(accumulation_native_identity),
            ),
        ] {
            let mut json = serde_json::to_value(&report).unwrap();
            json["accumulation"][field] = value;
            assert!(
                NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
                "tampered {field} must reject"
            );
        }

        let mut impossible = report;
        impossible.main.rendered_entry_count = 1;
        impossible.main.unique_rendered_entry_count = Some(1);
        let source_segmentation = impossible.main.dispatch_segmentation.unwrap();
        let accumulation = impossible.accumulation.as_mut().unwrap();
        accumulation.cache_hit_count = 2;
        accumulation.cache_miss_count = 0;
        accumulation.unique_rendered_entry_count = Some(0);
        accumulation.shared_prefix_entry_count = Some(2);
        accumulation.loaded_module_count = 0;
        accumulation.referenced_module_count = Some(1);
        accumulation.durable_artifact_cache_miss_count = 0;
        accumulation.compiler_invocation_count = 0;
        accumulation.combined_compile_link_count = Some(0);
        accumulation.dispatch_segmentation = Some(source_segmentation);
        impossible.prepare_compiler_process_count = Some(1);
        assert!(
            impossible
                .accumulation
                .as_ref()
                .unwrap()
                .validate(NATIVE_TRAINING_REPORT_FORMAT_VERSION)
                .is_ok(),
            "the per-program partition is otherwise coherent"
        );
        assert!(
            impossible.validate().is_err(),
            "a shared prefix cannot exceed the named source program"
        );
    }

    #[test]
    fn current_report_prefix_source_indices_compact_absent_optional_programs() {
        let mut report = phase_specialized_report();
        let evaluation = report.accumulation.take();
        report.evaluation = evaluation;
        report.accumulation_replay_traffic = None;
        report.accumulation_replay_executed_native_item_count = None;
        report.accumulation_schedule_cache_keys.clear();
        report.step_phases = None;
        assert!(report.validate().is_ok());

        report
            .evaluation
            .as_mut()
            .unwrap()
            .shared_prefix_source_program_index = Some(1);
        assert!(
            report.validate().is_err(),
            "absent optional programs cannot leave a hole in the source ordinal"
        );
    }

    #[test]
    fn zero_durations_and_unavailable_cpu_measurements_round_trip() {
        let report = zero_report();
        let bytes = report.to_json_bytes().unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["format_version"], NATIVE_TRAINING_REPORT_FORMAT_V10);
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
        assert_eq!(json["step_phases"]["first"]["phase"], "accumulation_only");
        assert!(json["step_phases"]["warm_accumulation_only"].is_null());
        assert_eq!(
            json["step_phases"]["warm_optimizer_commit"]["wall_time"]["sample_count"],
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
        remove_step_phases(&mut json);
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
        remove_step_phases(&mut json);
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
        remove_step_phases(&mut json);
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
        remove_step_phases(&mut json);
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
        remove_step_phases(&mut json);
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
        remove_step_phases(&mut json);
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
        remove_step_phases(&mut json);
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
        remove_step_phases(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(report.main().native_item_count(), 2);
        assert_eq!(report.main().rendered_entry_count(), 2);
        assert_eq!(report.main_replay_executed_native_item_count(), Some(1));
    }

    #[test]
    fn version_nine_report_without_step_phases_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V9);
        remove_step_phases(&mut json);
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.step_phases().is_none());
        assert_eq!(report.main().rendered_entry_count(), 2);
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
            remove_step_phases(&mut json);
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
    fn single_program_report_accepts_grouped_physical_native_inventory() {
        let mut report = zero_report();
        report.main.logical_schedule_item_count = 4;
        report.main.native_item_count = 4;
        report.main.cache_miss_count = 4;
        report.main.rendered_entry_count = 1;
        report.schedule_cache_keys.extend([23, 29]);
        let bytes = report.to_json_bytes().unwrap();
        let decoded = NativeTrainingReport::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V10);
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
            compiler_process_total: None,
            linker_process: None,
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
        let classified = &mut report.step_phases.as_mut().unwrap().first;
        classified.total_wall_time = total;
        classified.executor_wall_time = executor;
        classified.recurrent_overhead_wall_time = overhead;
        assert!(report.validate().is_ok());

        report
            .main_replay_recurrent_overhead_wall_time
            .as_mut()
            .unwrap()
            .first = BenchmarkDuration::from_duration(Duration::from_nanos(5));
        assert!(report.validate().is_err());
    }

    #[test]
    fn current_report_requires_complete_classified_step_phases() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json.as_object_mut().unwrap().remove("step_phases");
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut report = zero_report();
        report
            .step_phases
            .as_mut()
            .unwrap()
            .warm_optimizer_commit
            .as_mut()
            .unwrap()
            .executor_wall_time
            .sample_count = 2;
        assert!(report.validate().is_err());

        let mut report = zero_report();
        report
            .step_phases
            .as_mut()
            .unwrap()
            .warm_optimizer_commit
            .as_mut()
            .unwrap()
            .recurrent_overhead_total_wall_time =
            BenchmarkDuration::from_duration(Duration::from_nanos(1));
        assert!(report.validate().is_err());

        let mut legacy = serde_json::to_value(zero_report()).unwrap();
        legacy["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V9);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err()
        );
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
    fn current_program_distinguishes_zero_domain_items_from_missing_rendered_work() {
        let mut program = phase_specialized_report().main;
        program.rendered_entry_count = 1;
        program.unique_rendered_entry_count = Some(1);
        assert!(
            program
                .validate(NATIVE_TRAINING_REPORT_FORMAT_VERSION)
                .is_ok(),
            "one rendered entry plus one zero-domain item is valid"
        );

        program.rendered_entry_count = 0;
        program.loaded_module_count = 0;
        program.referenced_module_count = Some(0);
        program.unique_rendered_entry_count = Some(0);
        program.durable_artifact_cache_miss_count = 0;
        program.compiler_invocation_count = 0;
        assert!(
            program
                .validate(NATIVE_TRAINING_REPORT_FORMAT_VERSION)
                .is_err(),
            "nonempty logical native inventory cannot claim no rendered work"
        );
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
