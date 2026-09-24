//! Validated observational evidence for strict-native CPU compiled training.
//!
//! The runtime remains the source of preparation and replay facts. This module
//! only checks and aggregates detached reports; it does not time calls, execute
//! programs, or infer unavailable device and allocator measurements.

mod compile_phase;
mod compiler_evidence;
mod inspection;
mod program_report;
mod report;
mod step_phases;

pub use compile_phase::{NativeTrainingCompilePhase, NativeTrainingCompilePhaseReport};
#[cfg(test)]
use compiler_evidence::NativeTrainingCompilerProcessKind;
use compiler_evidence::{
    NativeTrainingCompilerCriticalTail, NativeTrainingCompilerProcessTiming,
    NativeTrainingModuleOverlap, NativeTrainingProgramPairOverlap, NativeTrainingTranslationUnit,
    compiler_process_evidence, validate_module_overlaps, validate_program_pair_overlaps,
    validate_translation_units,
};
use inspection::ProgramInspection;
pub use inspection::{
    CompiledAdamWInspection, CompiledTrainingCompileObservation,
    CompiledTrainingCompilePhaseObservation, NativeTrainingPreparationTiming,
};
pub use program_report::NativeTrainingProgramReport;
use report::CheckpointReport;
pub use report::{NativeTrainingReplayTiming, NativeTrainingReport};
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
use crate::{BenchmarkDuration, BenchmarkLatencySummary, Error, Result};
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
