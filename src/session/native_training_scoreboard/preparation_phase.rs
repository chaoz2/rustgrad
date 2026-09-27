//! Checked host-finalization timing for strict-native training preparation.

use super::invalid;
use crate::session::compiled_training::{
    NativeCpuCompiledTrainingPreparationReport, NativeCpuPreparationFinalizationPhases,
    NativeCpuProgramFinalizationPhases,
};
use crate::{BenchmarkDuration, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Host intervals outside one program's existing layout/render/compiler
/// preparation partition.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingProgramFinalizationReport {
    pre_layout_admission_wall_time: BenchmarkDuration,
    workspace_construction_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recurrent_finalization_wall_time: Option<BenchmarkDuration>,
}

impl NativeTrainingProgramFinalizationReport {
    fn from_observation(observation: &NativeCpuProgramFinalizationPhases) -> Self {
        Self {
            pre_layout_admission_wall_time: BenchmarkDuration::from_duration(
                observation.pre_layout_admission_wall_time(),
            ),
            workspace_construction_wall_time: BenchmarkDuration::from_duration(
                observation.workspace_construction_wall_time(),
            ),
            recurrent_finalization_wall_time: observation
                .recurrent_finalization_wall_time()
                .map(BenchmarkDuration::from_duration),
        }
    }

    fn measured_wall_time(self) -> Result<Duration> {
        [
            Some(self.pre_layout_admission_wall_time),
            Some(self.workspace_construction_wall_time),
            self.recurrent_finalization_wall_time,
        ]
        .into_iter()
        .flatten()
        .try_fold(Duration::ZERO, |total, duration| {
            let duration = duration
                .to_duration()
                .map_err(|_| invalid("invalid native preparation finalization duration"))?;
            total
                .checked_add(duration)
                .ok_or_else(|| invalid("native preparation finalization duration overflows"))
        })
    }

    /// Input and item-boundary admission before native layout planning.
    pub const fn pre_layout_admission_wall_time(&self) -> BenchmarkDuration {
        self.pre_layout_admission_wall_time
    }

    /// Fresh call-local native workspace construction after module preparation.
    pub const fn workspace_construction_wall_time(&self) -> BenchmarkDuration {
        self.workspace_construction_wall_time
    }

    /// Recurrent replay sealing for stateful roles; absent for evaluation.
    pub const fn recurrent_finalization_wall_time(&self) -> Option<BenchmarkDuration> {
        self.recurrent_finalization_wall_time
    }
}

/// Checked instrumented prefix and outer remainder for native training
/// preparation. The unattributed interval is explicit and is not assigned to
/// a named operation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingPreparationFinalizationReport {
    instrumented_wall_time: BenchmarkDuration,
    outer_remainder_wall_time: BenchmarkDuration,
    bootstrap_wall_time: BenchmarkDuration,
    main: NativeTrainingProgramFinalizationReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation: Option<NativeTrainingProgramFinalizationReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    partial_flush: Option<NativeTrainingProgramFinalizationReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zero_grad: Option<NativeTrainingProgramFinalizationReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    evaluation: Option<NativeTrainingProgramFinalizationReport>,
    report_input_assembly_wall_time: BenchmarkDuration,
    unattributed_wall_time: BenchmarkDuration,
}

pub(super) struct PreparationProgramPresence {
    pub(super) accumulation: bool,
    pub(super) partial_flush: bool,
    pub(super) zero_grad: bool,
    pub(super) evaluation: bool,
}

impl NativeTrainingPreparationFinalizationReport {
    /// Converts the authenticated runtime observation into its canonical wire
    /// form and validates it against the independently timed caller interval.
    pub fn from_preparation(
        preparation: &NativeCpuCompiledTrainingPreparationReport,
        prepare_wall_time: Duration,
    ) -> Result<Self> {
        let program_wall_time = std::iter::once(preparation.main())
            .chain(preparation.accumulation())
            .chain(preparation.partial_flush())
            .chain(preparation.zero_grad())
            .chain(preparation.evaluation())
            .try_fold(Duration::ZERO, |total, program| {
                total
                    .checked_add(program.wall_time())
                    .ok_or_else(|| invalid("native program preparation duration overflows"))
            })?;
        let effective_program_wall_time = program_wall_time
            .checked_sub(preparation.parallel_module_overlap_wall_time())
            .and_then(|duration| {
                duration.checked_sub(preparation.parallel_render_overlap_wall_time())
            })
            .ok_or_else(|| invalid("native parallel work overlap exceeds program time"))?;
        let prepare_runtime_overhead_wall_time = prepare_wall_time
            .checked_sub(effective_program_wall_time)
            .ok_or_else(|| invalid("native program preparation exceeds whole prepare time"))?;
        Self::from_observation(
            preparation.finalization_phases(),
            prepare_wall_time,
            prepare_runtime_overhead_wall_time,
            PreparationProgramPresence {
                accumulation: preparation.accumulation().is_some(),
                partial_flush: preparation.partial_flush().is_some(),
                zero_grad: preparation.zero_grad().is_some(),
                evaluation: preparation.evaluation().is_some(),
            },
        )
    }

    fn from_observation(
        observation: &NativeCpuPreparationFinalizationPhases,
        prepare_wall_time: Duration,
        prepare_runtime_overhead_wall_time: Duration,
        presence: PreparationProgramPresence,
    ) -> Result<Self> {
        let outer_remainder_wall_time = prepare_wall_time
            .checked_sub(observation.instrumented_wall_time())
            .ok_or_else(|| invalid("instrumented native preparation exceeds caller time"))?;
        let report = Self {
            instrumented_wall_time: BenchmarkDuration::from_duration(
                observation.instrumented_wall_time(),
            ),
            outer_remainder_wall_time: BenchmarkDuration::from_duration(outer_remainder_wall_time),
            bootstrap_wall_time: BenchmarkDuration::from_duration(
                observation.bootstrap_wall_time(),
            ),
            main: NativeTrainingProgramFinalizationReport::from_observation(observation.main()),
            accumulation: observation
                .accumulation()
                .map(NativeTrainingProgramFinalizationReport::from_observation),
            partial_flush: observation
                .partial_flush()
                .map(NativeTrainingProgramFinalizationReport::from_observation),
            zero_grad: observation
                .zero_grad()
                .map(NativeTrainingProgramFinalizationReport::from_observation),
            evaluation: observation
                .evaluation()
                .map(NativeTrainingProgramFinalizationReport::from_observation),
            report_input_assembly_wall_time: BenchmarkDuration::from_duration(
                observation.report_input_assembly_wall_time(),
            ),
            unattributed_wall_time: BenchmarkDuration::from_duration(
                observation.unattributed_wall_time(),
            ),
        };
        report.validate(
            BenchmarkDuration::from_duration(prepare_wall_time),
            BenchmarkDuration::from_duration(prepare_runtime_overhead_wall_time),
            presence,
        )?;
        Ok(report)
    }

    pub(super) fn validate(
        &self,
        prepare_wall_time: BenchmarkDuration,
        prepare_runtime_overhead_wall_time: BenchmarkDuration,
        presence: PreparationProgramPresence,
    ) -> Result<()> {
        if self.accumulation.is_some() != presence.accumulation
            || self.partial_flush.is_some() != presence.partial_flush
            || self.zero_grad.is_some() != presence.zero_grad
            || self.evaluation.is_some() != presence.evaluation
            || self.main.recurrent_finalization_wall_time.is_none()
            || self
                .accumulation
                .is_some_and(|phase| phase.recurrent_finalization_wall_time.is_none())
            || self
                .partial_flush
                .is_some_and(|phase| phase.recurrent_finalization_wall_time.is_none())
            || self
                .zero_grad
                .is_some_and(|phase| phase.recurrent_finalization_wall_time.is_none())
            || self
                .evaluation
                .is_some_and(|phase| phase.recurrent_finalization_wall_time.is_some())
        {
            return Err(invalid(
                "native preparation finalization program inventory differs",
            ));
        }
        let instrumented = self
            .instrumented_wall_time
            .to_duration()
            .map_err(|_| invalid("invalid instrumented preparation duration"))?;
        let outer_remainder = self
            .outer_remainder_wall_time
            .to_duration()
            .map_err(|_| invalid("invalid outer preparation remainder duration"))?;
        let prepare = prepare_wall_time
            .to_duration()
            .map_err(|_| invalid("invalid caller preparation duration"))?;
        if instrumented.checked_add(outer_remainder) != Some(prepare) {
            return Err(invalid("instrumented and outer preparation times differ"));
        }
        let measured_program_finalization = std::iter::once(self.main)
            .chain(self.accumulation)
            .chain(self.partial_flush)
            .chain(self.zero_grad)
            .chain(self.evaluation)
            .try_fold(Duration::ZERO, |total, phase| {
                total
                    .checked_add(phase.measured_wall_time()?)
                    .ok_or_else(|| invalid("native preparation finalization duration overflows"))
            })?;
        let overhead = prepare_runtime_overhead_wall_time
            .to_duration()
            .map_err(|_| invalid("invalid native preparation overhead duration"))?;
        let accounted_overhead = [
            outer_remainder,
            self.bootstrap_wall_time
                .to_duration()
                .map_err(|_| invalid("invalid native preparation bootstrap duration"))?,
            measured_program_finalization,
            self.report_input_assembly_wall_time
                .to_duration()
                .map_err(|_| invalid("invalid report-input assembly duration"))?,
            self.unattributed_wall_time
                .to_duration()
                .map_err(|_| invalid("invalid unattributed preparation duration"))?,
        ]
        .into_iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total
                .checked_add(duration)
                .ok_or_else(|| invalid("native preparation finalization duration overflows"))
        })?;
        if accounted_overhead != overhead {
            return Err(invalid(
                "native preparation finalization does not partition overhead",
            ));
        }
        Ok(())
    }

    /// Prefix from CPU bootstrap start through report-input assembly.
    pub const fn instrumented_wall_time(&self) -> BenchmarkDuration {
        self.instrumented_wall_time
    }

    /// Caller-observed remainder after the instrumented prefix, including
    /// report closing, runtime construction, and any owner wrapper work.
    pub const fn outer_remainder_wall_time(&self) -> BenchmarkDuration {
        self.outer_remainder_wall_time
    }

    /// CPU state construction and checkpoint-frontier restore time.
    pub const fn bootstrap_wall_time(&self) -> BenchmarkDuration {
        self.bootstrap_wall_time
    }

    /// Finalization intervals for the optimizer-commit program.
    pub const fn main(&self) -> &NativeTrainingProgramFinalizationReport {
        &self.main
    }

    /// Finalization intervals for the accumulation-only program, when present.
    pub const fn accumulation(&self) -> Option<&NativeTrainingProgramFinalizationReport> {
        self.accumulation.as_ref()
    }

    /// Finalization intervals for partial-window commit, when present.
    pub const fn partial_flush(&self) -> Option<&NativeTrainingProgramFinalizationReport> {
        self.partial_flush.as_ref()
    }

    /// Finalization intervals for compiled `zero_grad`, when present.
    pub const fn zero_grad(&self) -> Option<&NativeTrainingProgramFinalizationReport> {
        self.zero_grad.as_ref()
    }

    /// Workspace intervals for evaluation, which has no recurrent finalization.
    pub const fn evaluation(&self) -> Option<&NativeTrainingProgramFinalizationReport> {
        self.evaluation.as_ref()
    }

    /// Role tuple and recurrent-inventory assembly before report construction.
    pub const fn report_input_assembly_wall_time(&self) -> BenchmarkDuration {
        self.report_input_assembly_wall_time
    }

    /// Instrumented-prefix time not assigned to an explicit named stage.
    pub const fn unattributed_wall_time(&self) -> BenchmarkDuration {
        self.unattributed_wall_time
    }
}

#[cfg(test)]
pub(super) fn zero_preparation_finalization(
    accumulation: bool,
) -> NativeTrainingPreparationFinalizationReport {
    let program = |recurrent| NativeTrainingProgramFinalizationReport {
        pre_layout_admission_wall_time: BenchmarkDuration::from_duration(Duration::ZERO),
        workspace_construction_wall_time: BenchmarkDuration::from_duration(Duration::ZERO),
        recurrent_finalization_wall_time: recurrent
            .then_some(BenchmarkDuration::from_duration(Duration::ZERO)),
    };
    NativeTrainingPreparationFinalizationReport {
        instrumented_wall_time: BenchmarkDuration::from_duration(Duration::ZERO),
        outer_remainder_wall_time: BenchmarkDuration::from_duration(Duration::ZERO),
        bootstrap_wall_time: BenchmarkDuration::from_duration(Duration::ZERO),
        main: program(true),
        accumulation: accumulation.then_some(program(true)),
        partial_flush: None,
        zero_grad: None,
        evaluation: None,
        report_input_assembly_wall_time: BenchmarkDuration::from_duration(Duration::ZERO),
        unattributed_wall_time: BenchmarkDuration::from_duration(Duration::ZERO),
    }
}

#[cfg(test)]
pub(super) fn set_zero_stage_finalization_partition(
    report: &mut NativeTrainingPreparationFinalizationReport,
    prepare_wall_time: Duration,
    overhead_wall_time: Duration,
) {
    report.instrumented_wall_time = BenchmarkDuration::from_duration(prepare_wall_time);
    report.outer_remainder_wall_time = BenchmarkDuration::from_duration(Duration::ZERO);
    report.bootstrap_wall_time = BenchmarkDuration::from_duration(Duration::ZERO);
    report.report_input_assembly_wall_time = BenchmarkDuration::from_duration(Duration::ZERO);
    report.unattributed_wall_time = BenchmarkDuration::from_duration(overhead_wall_time);
}

#[cfg(test)]
pub(super) fn move_accumulation_finalization_to_evaluation(
    report: &mut NativeTrainingPreparationFinalizationReport,
) {
    let accumulation = report
        .accumulation
        .take()
        .expect("test preparation has accumulation finalization");
    report.evaluation = Some(NativeTrainingProgramFinalizationReport {
        recurrent_finalization_wall_time: None,
        ..accumulation
    });
}
