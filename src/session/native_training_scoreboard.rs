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
pub const NATIVE_TRAINING_REPORT_FORMAT_VERSION: u32 = 3;
const MAX_REPLAY_SAMPLES: usize = 10_000;

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProgramInspection {
    capture_identity: u64,
    execution_plan: ExecutionPlanSummary,
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
        })
    }

    fn validate(&self) -> Result<()> {
        if self
            .cache_hit_count
            .checked_add(self.cache_miss_count)
            .ok_or_else(|| invalid("native program cache count overflows"))?
            != self.native_item_count
        {
            return Err(invalid("native program cache inventory differs"));
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

    pub const fn native_item_count(&self) -> u64 {
        self.native_item_count
    }

    pub const fn cache_hit_count(&self) -> u64 {
        self.cache_hit_count
    }

    pub const fn cache_miss_count(&self) -> u64 {
        self.cache_miss_count
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

/// Versioned strict-native CPU compiled-training observation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReport {
    format_version: u32,
    compile_wall_time: BenchmarkDuration,
    prepare_wall_time: BenchmarkDuration,
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
            1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_VERSION
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
        self.main.validate()?;
        match (self.format_version, &self.main_replay_traffic) {
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, None) => {}
            (NATIVE_TRAINING_REPORT_FORMAT_VERSION, Some(traffic))
                if traffic.borrowed_recurrent_input_bytes()
                    == self.recurrent_logical_state_bytes
                    && traffic.borrowed_recurrent_output_bytes()
                        == self.recurrent_logical_state_bytes => {}
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, Some(_)) => {
                return Err(invalid("legacy native training report has replay traffic"));
            }
            _ => return Err(invalid("native training replay traffic differs")),
        }
        for program in self
            .partial_flush
            .iter()
            .chain(&self.zero_grad)
            .chain(&self.evaluation)
        {
            program.validate()?;
            if program.vectorized != self.main.vectorized {
                return Err(invalid("native program vectorization policy differs"));
            }
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
    inspection: CompiledAdamWInspection,
    main: NativeTrainingProgramReport,
    partial_flush: Option<NativeTrainingProgramReport>,
    zero_grad: Option<NativeTrainingProgramReport>,
    evaluation: Option<NativeTrainingProgramReport>,
    durations: Vec<Duration>,
    schedule_cache_keys: Option<Vec<u64>>,
    main_replay_traffic: Option<NativeCpuReplayTraffic>,
    checkpoint: Option<CheckpointReport>,
}

impl NativeTrainingScoreboard {
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
        Ok(Self {
            compile_wall_time,
            prepare_wall_time,
            inspection,
            main,
            partial_flush,
            zero_grad,
            evaluation,
            durations: Vec::new(),
            schedule_cache_keys: None,
            main_replay_traffic: None,
            checkpoint: None,
        })
    }

    /// Records one report returned by a committed native training step. Failed
    /// or rejected steps expose no report and therefore cannot become samples.
    pub fn record(&mut self, report: &NativeCpuRunReport) -> Result<()> {
        if self.durations.len() >= MAX_REPLAY_SAMPLES {
            return Err(invalid("native training replay sample limit exceeded"));
        }
        let expected_invocation = self.durations.len() as u64 + 1;
        if report.capture_identity() != self.main.capture_identity
            || report.native_identity() != self.main.native_identity
            || report.is_vectorized() != self.main.vectorized
            || count(report.native_item_count(), "native item")? != self.main.native_item_count
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
        if self.schedule_cache_keys.is_none() {
            self.schedule_cache_keys = Some(report.schedule_cache_keys().to_vec());
        }
        if self.main_replay_traffic.is_none() {
            self.main_replay_traffic = Some(*report.traffic());
        }
        self.durations.push(report.wall_time());
        Ok(())
    }

    pub fn observe_checkpoint(
        &mut self,
        checkpoint: &CompiledAdamWCheckpoint,
        wall_time: Duration,
    ) -> Result<()> {
        let info = checkpoint.info();
        let replay_count = self.durations.len() as u64;
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
        let Some((&first, steady)) = self.durations.split_first() else {
            return Err(invalid("native training scoreboard has no replay"));
        };
        if steady.is_empty() {
            return Err(invalid("native training scoreboard has no steady replay"));
        }
        let (steady_replay_total_wall_time, steady_microbatches_per_second) = rate(steady)?;
        let report = NativeTrainingReport {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            compile_wall_time: BenchmarkDuration::from_duration(self.compile_wall_time),
            prepare_wall_time: BenchmarkDuration::from_duration(self.prepare_wall_time),
            initial_replay_step: self.inspection.initial_replay_step,
            successful_replay_count: self.durations.len() as u64,
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
            first_replay_wall_time: BenchmarkDuration::from_duration(first),
            steady_replay_total_wall_time,
            steady_replay_wall_time: latency_summary(steady)?,
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
    let total = durations
        .iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total
                .checked_add(*duration)
                .ok_or_else(|| invalid("steady replay duration overflows"))
        })?;
    let total = BenchmarkDuration::from_duration(total);
    Ok((total, rate_from_total(durations.len() as u64, total)?))
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

    fn zero_report() -> NativeTrainingReport {
        NativeTrainingReport {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            compile_wall_time: zero_duration(),
            prepare_wall_time: zero_duration(),
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
            },
            partial_flush: None,
            zero_grad: None,
            evaluation: None,
            recurrent_logical_state_count: 4,
            recurrent_logical_state_bytes: 16,
            main_replay_traffic: Some(NativeCpuReplayTraffic::new(2, 12, 16, 16)),
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
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.zero_grad().is_none());
    }

    #[test]
    fn version_two_report_without_replay_traffic_still_decodes() {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V2);
        json.as_object_mut().unwrap().remove("main_replay_traffic");
        let report =
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(report.main_replay_traffic().is_none());
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
        report.steady_replay_total_wall_time = elapsed;
        report.steady_replay_wall_time.min = elapsed;
        report.steady_replay_wall_time.nearest_rank_p50 = elapsed;
        report.steady_replay_wall_time.nearest_rank_p95 = elapsed;
        report.steady_replay_wall_time.max = elapsed;
        report.steady_microbatches_per_second = rate_from_total(1, elapsed).unwrap();
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
        report.steady_replay_total_wall_time = elapsed;
        report.steady_replay_wall_time.min = elapsed;
        report.steady_replay_wall_time.nearest_rank_p50 = elapsed;
        report.steady_replay_wall_time.nearest_rank_p95 = elapsed;
        report.steady_replay_wall_time.max = elapsed;
        report.steady_microbatches_per_second = rate_from_total(1, elapsed).unwrap();

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
