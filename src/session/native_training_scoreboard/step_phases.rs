use super::{
    ReplayTiming, invalid, latency_summary, rate, rate_from_total, sum_durations,
    validate_phase_partition, validate_total_duration,
};
use crate::{BenchmarkDuration, BenchmarkLatencySummary, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Observable outcome of one successful compiled-training main replay.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTrainingStepPhase {
    /// The replay retained gradients without advancing the optimizer.
    AccumulationOnly,
    /// The replay completed its window and committed an optimizer update.
    OptimizerCommit,
}

/// Phase timings for the first successful compiled-training main replay.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingFirstStepReport {
    pub(super) phase: NativeTrainingStepPhase,
    pub(super) total_wall_time: BenchmarkDuration,
    pub(super) executor_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) native_dispatcher_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) executor_host_wall_time: Option<BenchmarkDuration>,
    pub(super) recurrent_overhead_wall_time: BenchmarkDuration,
}

impl NativeTrainingFirstStepReport {
    pub const fn phase(&self) -> NativeTrainingStepPhase {
        self.phase
    }

    pub const fn total_wall_time(&self) -> BenchmarkDuration {
        self.total_wall_time
    }

    pub const fn executor_wall_time(&self) -> BenchmarkDuration {
        self.executor_wall_time
    }

    pub const fn native_dispatcher_wall_time(&self) -> Option<BenchmarkDuration> {
        self.native_dispatcher_wall_time
    }

    pub const fn executor_host_wall_time(&self) -> Option<BenchmarkDuration> {
        self.executor_host_wall_time
    }

    pub const fn recurrent_overhead_wall_time(&self) -> BenchmarkDuration {
        self.recurrent_overhead_wall_time
    }

    fn validate(&self, has_dispatch_partition: bool) -> Result<()> {
        validate_phase_partition(
            self.total_wall_time,
            self.executor_wall_time,
            self.recurrent_overhead_wall_time,
            "first classified replay",
        )?;
        match (
            has_dispatch_partition,
            self.native_dispatcher_wall_time,
            self.executor_host_wall_time,
        ) {
            (false, None, None) => Ok(()),
            (true, Some(dispatcher), Some(host)) => validate_phase_partition(
                self.executor_wall_time,
                dispatcher,
                host,
                "first classified executor",
            ),
            _ => Err(invalid("classified first dispatcher timing differs")),
        }
    }
}

/// Timing summary for one class of successful warm compiled-training replays.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingWarmStepReport {
    pub(super) total_wall_time: BenchmarkDuration,
    pub(super) wall_time: BenchmarkLatencySummary,
    pub(super) steps_per_second: Option<f64>,
    pub(super) executor_total_wall_time: BenchmarkDuration,
    pub(super) executor_wall_time: BenchmarkLatencySummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) native_dispatcher_total_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) native_dispatcher_wall_time: Option<BenchmarkLatencySummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) executor_host_total_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) executor_host_wall_time: Option<BenchmarkLatencySummary>,
    pub(super) recurrent_overhead_total_wall_time: BenchmarkDuration,
    pub(super) recurrent_overhead_wall_time: BenchmarkLatencySummary,
}

impl NativeTrainingWarmStepReport {
    pub const fn sample_count(&self) -> u64 {
        self.wall_time.sample_count
    }

    pub const fn total_wall_time(&self) -> BenchmarkDuration {
        self.total_wall_time
    }

    pub const fn wall_time(&self) -> &BenchmarkLatencySummary {
        &self.wall_time
    }

    pub const fn steps_per_second(&self) -> Option<f64> {
        self.steps_per_second
    }

    pub const fn executor_total_wall_time(&self) -> BenchmarkDuration {
        self.executor_total_wall_time
    }

    pub const fn executor_wall_time(&self) -> &BenchmarkLatencySummary {
        &self.executor_wall_time
    }

    pub const fn native_dispatcher_total_wall_time(&self) -> Option<BenchmarkDuration> {
        self.native_dispatcher_total_wall_time
    }

    pub const fn native_dispatcher_wall_time(&self) -> Option<&BenchmarkLatencySummary> {
        self.native_dispatcher_wall_time.as_ref()
    }

    pub const fn executor_host_total_wall_time(&self) -> Option<BenchmarkDuration> {
        self.executor_host_total_wall_time
    }

    pub const fn executor_host_wall_time(&self) -> Option<&BenchmarkLatencySummary> {
        self.executor_host_wall_time.as_ref()
    }

    pub const fn recurrent_overhead_total_wall_time(&self) -> BenchmarkDuration {
        self.recurrent_overhead_total_wall_time
    }

    pub const fn recurrent_overhead_wall_time(&self) -> &BenchmarkLatencySummary {
        &self.recurrent_overhead_wall_time
    }

    fn from_timings(timings: &[ReplayTiming]) -> Result<Option<Self>> {
        if timings.is_empty() {
            return Ok(None);
        }
        let totals = timings
            .iter()
            .map(|timing| timing.total)
            .collect::<Vec<_>>();
        let executors = timings
            .iter()
            .map(|timing| timing.executor)
            .collect::<Vec<_>>();
        let overheads = timings
            .iter()
            .map(|timing| timing.overhead)
            .collect::<Vec<_>>();
        let dispatchers = timings
            .iter()
            .map(|timing| timing.native_dispatcher)
            .collect::<Vec<_>>();
        let hosts = timings
            .iter()
            .map(|timing| timing.executor_host)
            .collect::<Vec<_>>();
        let (total_wall_time, steps_per_second) = rate(&totals)?;
        Ok(Some(Self {
            total_wall_time,
            wall_time: latency_summary(&totals)?,
            steps_per_second,
            executor_total_wall_time: sum_durations(&executors)?,
            executor_wall_time: latency_summary(&executors)?,
            native_dispatcher_total_wall_time: Some(sum_durations(&dispatchers)?),
            native_dispatcher_wall_time: Some(latency_summary(&dispatchers)?),
            executor_host_total_wall_time: Some(sum_durations(&hosts)?),
            executor_host_wall_time: Some(latency_summary(&hosts)?),
            recurrent_overhead_total_wall_time: sum_durations(&overheads)?,
            recurrent_overhead_wall_time: latency_summary(&overheads)?,
        }))
    }

    fn validate(&self, has_dispatch_partition: bool) -> Result<()> {
        let count = self.wall_time.sample_count;
        if count == 0
            || self.executor_wall_time.sample_count != count
            || self.recurrent_overhead_wall_time.sample_count != count
        {
            return Err(invalid("classified warm replay counts differ"));
        }
        for (summary, total) in [
            (&self.wall_time, self.total_wall_time),
            (&self.executor_wall_time, self.executor_total_wall_time),
            (
                &self.recurrent_overhead_wall_time,
                self.recurrent_overhead_total_wall_time,
            ),
        ] {
            for duration in [
                total,
                summary.min,
                summary.nearest_rank_p50,
                summary.nearest_rank_p95,
                summary.max,
            ] {
                duration
                    .to_duration()
                    .map_err(|_| invalid("invalid classified warm replay duration"))?;
            }
            if summary.min > summary.nearest_rank_p50
                || summary.nearest_rank_p50 > summary.nearest_rank_p95
                || summary.nearest_rank_p95 > summary.max
                || total < summary.max
            {
                return Err(invalid("invalid classified warm replay summary"));
            }
            validate_total_duration(summary, total)?;
        }
        validate_phase_partition(
            self.total_wall_time,
            self.executor_total_wall_time,
            self.recurrent_overhead_total_wall_time,
            "classified warm replay",
        )?;
        match (
            has_dispatch_partition,
            self.native_dispatcher_total_wall_time,
            &self.native_dispatcher_wall_time,
            self.executor_host_total_wall_time,
            &self.executor_host_wall_time,
        ) {
            (false, None, None, None, None) => {}
            (true, Some(dispatcher_total), Some(dispatcher), Some(host_total), Some(host)) => {
                for (summary, total) in [(dispatcher, dispatcher_total), (host, host_total)] {
                    if summary.sample_count != count
                        || summary.min > summary.nearest_rank_p50
                        || summary.nearest_rank_p50 > summary.nearest_rank_p95
                        || summary.nearest_rank_p95 > summary.max
                        || total < summary.max
                    {
                        return Err(invalid("classified warm dispatcher counts differ"));
                    }
                    for duration in [
                        total,
                        summary.min,
                        summary.nearest_rank_p50,
                        summary.nearest_rank_p95,
                        summary.max,
                    ] {
                        duration
                            .to_duration()
                            .map_err(|_| invalid("invalid classified warm dispatcher duration"))?;
                    }
                    validate_total_duration(summary, total)?;
                }
                validate_phase_partition(
                    self.executor_total_wall_time,
                    dispatcher_total,
                    host_total,
                    "classified warm executor",
                )?;
            }
            _ => return Err(invalid("classified warm dispatcher timing differs")),
        }
        let expected_rate = rate_from_total(count, self.total_wall_time)?;
        if self.steps_per_second.map(f64::to_bits) != expected_rate.map(f64::to_bits) {
            return Err(invalid("classified warm replay rate differs"));
        }
        Ok(())
    }
}

/// Authenticated first and warm phase partition for successful main replays.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingStepPhaseReport {
    pub(super) first: NativeTrainingFirstStepReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) warm_accumulation_only: Option<NativeTrainingWarmStepReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) warm_optimizer_commit: Option<NativeTrainingWarmStepReport>,
}

/// Global timing summaries against which classified phase totals are checked.
pub(super) struct ReplayTimingPartition<'a> {
    pub(super) executor: &'a super::NativeTrainingReplayTiming,
    pub(super) native_dispatcher: Option<&'a super::NativeTrainingReplayTiming>,
    pub(super) executor_host: Option<&'a super::NativeTrainingReplayTiming>,
    pub(super) overhead: &'a super::NativeTrainingReplayTiming,
}

impl NativeTrainingStepPhaseReport {
    pub const fn first(&self) -> &NativeTrainingFirstStepReport {
        &self.first
    }

    pub const fn warm_accumulation_only(&self) -> Option<&NativeTrainingWarmStepReport> {
        self.warm_accumulation_only.as_ref()
    }

    pub const fn warm_optimizer_commit(&self) -> Option<&NativeTrainingWarmStepReport> {
        self.warm_optimizer_commit.as_ref()
    }

    pub(super) fn from_timings(
        timings: &[ReplayTiming],
        phases: &[NativeTrainingStepPhase],
    ) -> Result<Self> {
        let Some((first, warm)) = timings.split_first() else {
            return Err(invalid("classified scoreboard has no replay"));
        };
        let Some((&first_phase, warm_phases)) = phases.split_first() else {
            return Err(invalid("classified scoreboard has no replay phase"));
        };
        if warm.len() != warm_phases.len() {
            return Err(invalid("classified replay timing and phase counts differ"));
        }
        let accumulation = warm
            .iter()
            .zip(warm_phases)
            .filter_map(|(timing, phase)| {
                (*phase == NativeTrainingStepPhase::AccumulationOnly).then_some(*timing)
            })
            .collect::<Vec<_>>();
        let commits = warm
            .iter()
            .zip(warm_phases)
            .filter_map(|(timing, phase)| {
                (*phase == NativeTrainingStepPhase::OptimizerCommit).then_some(*timing)
            })
            .collect::<Vec<_>>();
        Ok(Self {
            first: NativeTrainingFirstStepReport {
                phase: first_phase,
                total_wall_time: BenchmarkDuration::from_duration(first.total),
                executor_wall_time: BenchmarkDuration::from_duration(first.executor),
                native_dispatcher_wall_time: Some(BenchmarkDuration::from_duration(
                    first.native_dispatcher,
                )),
                executor_host_wall_time: Some(BenchmarkDuration::from_duration(
                    first.executor_host,
                )),
                recurrent_overhead_wall_time: BenchmarkDuration::from_duration(first.overhead),
            },
            warm_accumulation_only: NativeTrainingWarmStepReport::from_timings(&accumulation)?,
            warm_optimizer_commit: NativeTrainingWarmStepReport::from_timings(&commits)?,
        })
    }

    pub(super) fn validate(
        &self,
        successful_replay_count: u64,
        first_total: BenchmarkDuration,
        steady_total: BenchmarkDuration,
        timing: ReplayTimingPartition<'_>,
    ) -> Result<()> {
        let ReplayTimingPartition {
            executor,
            native_dispatcher,
            executor_host,
            overhead,
        } = timing;
        if native_dispatcher.is_some() != executor_host.is_some() {
            return Err(invalid("classified dispatcher timing partition differs"));
        }
        let has_dispatch_partition = native_dispatcher.is_some() && executor_host.is_some();
        self.first.validate(has_dispatch_partition)?;
        for phase in [
            self.warm_accumulation_only.as_ref(),
            self.warm_optimizer_commit.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            phase.validate(has_dispatch_partition)?;
        }
        if self.first.total_wall_time != first_total
            || self.first.executor_wall_time != executor.first
            || self.first.native_dispatcher_wall_time
                != native_dispatcher.map(|timing| timing.first)
            || self.first.executor_host_wall_time != executor_host.map(|timing| timing.first)
            || self.first.recurrent_overhead_wall_time != overhead.first
        {
            return Err(invalid("classified first replay differs"));
        }
        let warm_count = self
            .warm_accumulation_only
            .as_ref()
            .map_or(0, NativeTrainingWarmStepReport::sample_count)
            .checked_add(
                self.warm_optimizer_commit
                    .as_ref()
                    .map_or(0, NativeTrainingWarmStepReport::sample_count),
            )
            .ok_or_else(|| invalid("classified warm replay count overflows"))?;
        if warm_count != successful_replay_count - 1 {
            return Err(invalid("classified warm replay count differs"));
        }
        let sum = |select: fn(&NativeTrainingWarmStepReport) -> BenchmarkDuration| {
            [
                self.warm_accumulation_only.as_ref(),
                self.warm_optimizer_commit.as_ref(),
            ]
            .into_iter()
            .flatten()
            .try_fold(Duration::ZERO, |total, phase| {
                total
                    .checked_add(
                        select(phase)
                            .to_duration()
                            .map_err(|_| invalid("invalid classified replay duration"))?,
                    )
                    .ok_or_else(|| invalid("classified replay duration overflows"))
            })
            .map(BenchmarkDuration::from_duration)
        };
        let optional_sum = |select: fn(
            &NativeTrainingWarmStepReport,
        ) -> Option<BenchmarkDuration>|
         -> Result<Option<BenchmarkDuration>> {
            if !has_dispatch_partition {
                return Ok(None);
            }
            [
                self.warm_accumulation_only.as_ref(),
                self.warm_optimizer_commit.as_ref(),
            ]
            .into_iter()
            .flatten()
            .try_fold(Duration::ZERO, |total, phase| {
                total
                    .checked_add(
                        select(phase)
                            .ok_or_else(|| invalid("classified dispatcher timing is absent"))?
                            .to_duration()
                            .map_err(|_| invalid("invalid classified replay duration"))?,
                    )
                    .ok_or_else(|| invalid("classified replay duration overflows"))
            })
            .map(BenchmarkDuration::from_duration)
            .map(Some)
        };
        let dispatcher_total = optional_sum(|phase| phase.native_dispatcher_total_wall_time)?;
        let host_total = optional_sum(|phase| phase.executor_host_total_wall_time)?;
        if sum(NativeTrainingWarmStepReport::total_wall_time)? != steady_total
            || sum(NativeTrainingWarmStepReport::executor_total_wall_time)? != executor.steady_total
            || dispatcher_total != native_dispatcher.map(|timing| timing.steady_total)
            || host_total != executor_host.map(|timing| timing.steady_total)
            || sum(NativeTrainingWarmStepReport::recurrent_overhead_total_wall_time)?
                != overhead.steady_total
        {
            return Err(invalid("classified warm replay partition differs"));
        }
        Ok(())
    }
}
