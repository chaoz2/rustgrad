//! Host-finalization observations outside native program preparation phases.

use super::*;

/// Actual host intervals outside one program's existing layout/render/compiler
/// preparation partition. Recurrent finalization is absent for evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCpuProgramFinalizationPhases {
    pre_layout_admission_wall_time: Duration,
    workspace_construction_wall_time: Duration,
    recurrent_finalization_wall_time: Option<Duration>,
}

impl NativeCpuProgramFinalizationPhases {
    pub(super) const fn from_intervals(
        pre_layout_admission_wall_time: Duration,
        workspace_construction_wall_time: Duration,
        recurrent_finalization_wall_time: Option<Duration>,
    ) -> Self {
        Self {
            pre_layout_admission_wall_time,
            workspace_construction_wall_time,
            recurrent_finalization_wall_time,
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
            total
                .checked_add(duration)
                .ok_or_else(|| training("compiled native CPU finalization duration overflows"))
        })
    }

    /// Time spent validating capture inputs and item boundaries before layout.
    pub const fn pre_layout_admission_wall_time(&self) -> Duration {
        self.pre_layout_admission_wall_time
    }

    /// Time spent constructing fresh call-local workspace ownership.
    pub const fn workspace_construction_wall_time(&self) -> Duration {
        self.workspace_construction_wall_time
    }

    /// Time spent sealing recurrent banks and output projections. Evaluation
    /// has no recurrent finalization and returns `None`.
    pub const fn recurrent_finalization_wall_time(&self) -> Option<Duration> {
        self.recurrent_finalization_wall_time
    }
}

pub(super) struct NativeCpuPreparationFinalizationObservation {
    pub(super) instrumented_wall_time: Duration,
    pub(super) bootstrap_wall_time: Duration,
    pub(super) programs: NativeCpuTrainingPrograms<NativeCpuProgramFinalizationPhases>,
    pub(super) report_input_assembly_wall_time: Duration,
}

/// Disjoint finalization observations for one strict-native CPU training
/// preparation. Existing per-program preparation phases remain authoritative
/// for layout, render, compiler, and module loading work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCpuPreparationFinalizationPhases {
    instrumented_wall_time: Duration,
    bootstrap_wall_time: Duration,
    render_batch_wall_time: Duration,
    render_batch_orchestration_wall_time: Duration,
    main: NativeCpuProgramFinalizationPhases,
    accumulation: Option<NativeCpuProgramFinalizationPhases>,
    partial_flush: Option<NativeCpuProgramFinalizationPhases>,
    zero_grad: Option<NativeCpuProgramFinalizationPhases>,
    evaluation: Option<NativeCpuProgramFinalizationPhases>,
    report_input_assembly_wall_time: Duration,
    unattributed_wall_time: Duration,
}

impl NativeCpuPreparationFinalizationPhases {
    pub(super) fn from_observation(
        observation: NativeCpuPreparationFinalizationObservation,
        reports: &NativeCpuTrainingPrograms<NativeCpuProgramPreparationReport>,
        parallel_module_overlap_wall_time: Duration,
        parallel_render_overlap_wall_time: Duration,
        render_batch_wall_time: Duration,
        render_batch_orchestration_wall_time: Duration,
    ) -> Result<Self> {
        let NativeCpuTrainingPrograms {
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
        } = observation.programs;
        if accumulation.is_some() != reports.accumulation.is_some()
            || partial_flush.is_some() != reports.partial_flush.is_some()
            || zero_grad.is_some() != reports.zero_grad.is_some()
            || evaluation.is_some() != reports.evaluation.is_some()
            || main.recurrent_finalization_wall_time.is_none()
            || accumulation.is_some_and(|phase| phase.recurrent_finalization_wall_time.is_none())
            || partial_flush.is_some_and(|phase| phase.recurrent_finalization_wall_time.is_none())
            || zero_grad.is_some_and(|phase| phase.recurrent_finalization_wall_time.is_none())
            || evaluation.is_some_and(|phase| phase.recurrent_finalization_wall_time.is_some())
        {
            return Err(training(
                "compiled native CPU finalization program inventory differs",
            ));
        }
        let program_wall_time = std::iter::once(&reports.main)
            .chain(reports.accumulation.iter())
            .chain(reports.partial_flush.iter())
            .chain(reports.zero_grad.iter())
            .chain(reports.evaluation.iter())
            .try_fold(Duration::ZERO, |total, report| {
                total
                    .checked_add(report.wall_time)
                    .ok_or_else(|| training("compiled native CPU program duration overflows"))
            })?;
        let effective_program_wall_time = program_wall_time
            .checked_sub(parallel_module_overlap_wall_time)
            .and_then(|duration| duration.checked_sub(parallel_render_overlap_wall_time))
            .ok_or_else(|| training("compiled native CPU parallel overlap exceeds program time"))?;
        let render_wall_time = std::iter::once(&reports.main)
            .chain(reports.accumulation.iter())
            .chain(reports.partial_flush.iter())
            .chain(reports.zero_grad.iter())
            .chain(reports.evaluation.iter())
            .try_fold(Duration::ZERO, |total, report| {
                total
                    .checked_add(report.phases().render_wall_time())
                    .ok_or_else(|| training("compiled native CPU render duration overflows"))
            })?;
        let effective_render_wall_time = render_wall_time
            .checked_sub(parallel_render_overlap_wall_time)
            .ok_or_else(|| training("compiled native CPU render overlap exceeds render time"))?;
        if effective_render_wall_time.checked_add(render_batch_orchestration_wall_time)
            != Some(render_batch_wall_time)
        {
            return Err(training(
                "compiled native CPU render batch phases differ from batch time",
            ));
        }
        let measured_finalization_wall_time = std::iter::once(main)
            .chain(accumulation)
            .chain(partial_flush)
            .chain(zero_grad)
            .chain(evaluation)
            .try_fold(Duration::ZERO, |total, phase| {
                total
                    .checked_add(phase.measured_wall_time()?)
                    .ok_or_else(|| training("compiled native CPU finalization duration overflows"))
            })?;
        let accounted = [
            effective_program_wall_time,
            observation.bootstrap_wall_time,
            measured_finalization_wall_time,
            observation.report_input_assembly_wall_time,
            render_batch_orchestration_wall_time,
        ]
        .into_iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total
                .checked_add(duration)
                .ok_or_else(|| training("compiled native CPU finalization duration overflows"))
        })?;
        let unattributed_wall_time = observation
            .instrumented_wall_time
            .checked_sub(accounted)
            .ok_or_else(|| {
                training("compiled native CPU finalization phases exceed instrumented time")
            })?;
        Ok(Self {
            instrumented_wall_time: observation.instrumented_wall_time,
            bootstrap_wall_time: observation.bootstrap_wall_time,
            render_batch_wall_time,
            render_batch_orchestration_wall_time,
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
            report_input_assembly_wall_time: observation.report_input_assembly_wall_time,
            unattributed_wall_time,
        })
    }

    /// Time from CPU runtime bootstrap start through report-input assembly,
    /// stopping before final report and runtime construction.
    pub const fn instrumented_wall_time(&self) -> Duration {
        self.instrumented_wall_time
    }

    /// Time spent constructing and restoring the interpreter-owned CPU state
    /// used by strict-native replay.
    pub const fn bootstrap_wall_time(&self) -> Duration {
        self.bootstrap_wall_time
    }

    /// Complete wall time of capsule admission, local rendering, and capsule
    /// publication for the ordered native-program batch.
    pub const fn render_batch_wall_time(&self) -> Duration {
        self.render_batch_wall_time
    }

    /// Render-batch time outside the exact union of local renderer intervals.
    pub const fn render_batch_orchestration_wall_time(&self) -> Duration {
        self.render_batch_orchestration_wall_time
    }

    /// Finalization intervals for the optimizer-commit program.
    pub const fn main(&self) -> &NativeCpuProgramFinalizationPhases {
        &self.main
    }

    /// Finalization intervals for the accumulation-only program, when present.
    pub const fn accumulation(&self) -> Option<&NativeCpuProgramFinalizationPhases> {
        self.accumulation.as_ref()
    }

    /// Finalization intervals for partial-window commit, when present.
    pub const fn partial_flush(&self) -> Option<&NativeCpuProgramFinalizationPhases> {
        self.partial_flush.as_ref()
    }

    /// Finalization intervals for compiled `zero_grad`, when present.
    pub const fn zero_grad(&self) -> Option<&NativeCpuProgramFinalizationPhases> {
        self.zero_grad.as_ref()
    }

    /// Workspace intervals for evaluation, which has no recurrent finalization.
    pub const fn evaluation(&self) -> Option<&NativeCpuProgramFinalizationPhases> {
        self.evaluation.as_ref()
    }

    /// Time spent assembling role reports and recurrent inventory supplied to
    /// final preparation-report construction.
    pub const fn report_input_assembly_wall_time(&self) -> Duration {
        self.report_input_assembly_wall_time
    }

    /// Remaining instrumented time after effective program preparation and
    /// every explicitly measured stage.
    pub const fn unattributed_wall_time(&self) -> Duration {
        self.unattributed_wall_time
    }
}
