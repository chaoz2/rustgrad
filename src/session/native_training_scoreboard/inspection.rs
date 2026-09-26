//! Backend-neutral compiled-training inspection and preparation timing.
//!
//! These values describe immutable compiled topology and caller-observed
//! construction phases. They neither prepare a backend nor execute a program.

use super::super::NativeCpuProgramPreparationReport;
use super::program_report::NativeTrainingProgramReport;
use super::{
    NATIVE_TRAINING_REPORT_FORMAT_V14, NATIVE_TRAINING_REPORT_FORMAT_V15,
    NATIVE_TRAINING_REPORT_FORMAT_V16, NATIVE_TRAINING_REPORT_FORMAT_V17,
    NATIVE_TRAINING_REPORT_FORMAT_V18, NATIVE_TRAINING_REPORT_FORMAT_V19,
    NATIVE_TRAINING_REPORT_FORMAT_V20, NATIVE_TRAINING_REPORT_FORMAT_V21,
    NATIVE_TRAINING_REPORT_FORMAT_V22, NATIVE_TRAINING_REPORT_FORMAT_VERSION, invalid,
};
use crate::{BenchmarkDuration, ExecutionPlanSummary, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ProgramInspection {
    pub(super) capture_identity: u64,
    pub(super) execution_plan: ExecutionPlanSummary,
}

/// Exact host wall-time partition for preparing one strict-native CPU program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingPreparationTiming {
    pub(super) total: BenchmarkDuration,
    pub(super) layout: BenchmarkDuration,
    pub(super) render: BenchmarkDuration,
    pub(super) compiler_process: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) compiler_process_total: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) linker_process: Option<BenchmarkDuration>,
    pub(super) module_load: BenchmarkDuration,
    pub(super) residual: BenchmarkDuration,
}

impl NativeTrainingPreparationTiming {
    pub(super) fn from_preparation(preparation: &NativeCpuProgramPreparationReport) -> Self {
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

    pub(super) fn validate(
        &self,
        work: &NativeTrainingProgramReport,
        format_version: u32,
    ) -> Result<()> {
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
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
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
                | NATIVE_TRAINING_REPORT_FORMAT_V17
                | NATIVE_TRAINING_REPORT_FORMAT_V18
                | NATIVE_TRAINING_REPORT_FORMAT_V19
                | NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
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

/// Immutable logical work and recurrent-state facts for one compiled training
/// plan. Inspection prepares no target and exposes no capture.
///
/// Equality compares only that logical topology. Optional compile-phase
/// observations describe one construction and remain independently
/// inspectable, but do not make equivalent plans unequal.
#[derive(Clone, Debug)]
pub struct CompiledTrainingInspection {
    pub(super) initial_replay_step: u64,
    pub(super) main: ProgramInspection,
    pub(super) accumulation: Option<ProgramInspection>,
    pub(super) partial_flush: Option<ProgramInspection>,
    pub(super) zero_grad: Option<ProgramInspection>,
    pub(super) evaluation: Option<ProgramInspection>,
    pub(super) recurrent_state_count: usize,
    pub(super) recurrent_state_bytes: usize,
    pub(super) compile_phases: Option<CompiledTrainingCompileObservation>,
}

impl PartialEq for CompiledTrainingInspection {
    fn eq(&self, other: &Self) -> bool {
        self.initial_replay_step == other.initial_replay_step
            && self.main == other.main
            && self.accumulation == other.accumulation
            && self.partial_flush == other.partial_flush
            && self.zero_grad == other.zero_grad
            && self.evaluation == other.evaluation
            && self.recurrent_state_count == other.recurrent_state_count
            && self.recurrent_state_bytes == other.recurrent_state_bytes
    }
}

impl Eq for CompiledTrainingInspection {}

/// Source-compatible AdamW name for optimizer-neutral training inspection.
pub type CompiledAdamWInspection = CompiledTrainingInspection;

/// One backend-neutral compiled-training construction phase observed before
/// target preparation. Counts describe the immutable graph or schedule at the
/// end of the phase; neither durations nor counts enter program identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledTrainingCompilePhaseObservation {
    wall_time: Duration,
    graph_node_count: Option<usize>,
    logical_schedule_item_count: Option<usize>,
}

impl CompiledTrainingCompilePhaseObservation {
    pub(crate) const fn graph(wall_time: Duration, graph_node_count: usize) -> Self {
        Self {
            wall_time,
            graph_node_count: Some(graph_node_count),
            logical_schedule_item_count: None,
        }
    }

    pub(crate) const fn schedule(wall_time: Duration, logical_schedule_item_count: usize) -> Self {
        Self {
            wall_time,
            graph_node_count: None,
            logical_schedule_item_count: Some(logical_schedule_item_count),
        }
    }

    pub fn wall_time(&self) -> Duration {
        self.wall_time
    }

    pub const fn graph_node_count(&self) -> Option<usize> {
        self.graph_node_count
    }

    pub const fn logical_schedule_item_count(&self) -> Option<usize> {
        self.logical_schedule_item_count
    }
}

/// Typed observation of one successful backend-neutral training compilation.
/// The caller-observed wrapper residual is derived later by the scoreboard so
/// configuration, module traversal, and optional evaluator work remain visible.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledTrainingCompileObservation {
    objective_forward: CompiledTrainingCompilePhaseObservation,
    autograd: CompiledTrainingCompilePhaseObservation,
    optimizer_lowering: CompiledTrainingCompilePhaseObservation,
    main_capture: CompiledTrainingCompilePhaseObservation,
    accumulation_capture: Option<CompiledTrainingCompilePhaseObservation>,
    partial_flush: Option<CompiledTrainingCompilePhaseObservation>,
    zero_grad: Option<CompiledTrainingCompilePhaseObservation>,
    evaluation: Option<CompiledTrainingCompilePhaseObservation>,
}

impl CompiledTrainingCompileObservation {
    pub(crate) const fn new(
        objective_forward: CompiledTrainingCompilePhaseObservation,
        autograd: CompiledTrainingCompilePhaseObservation,
        optimizer_lowering: CompiledTrainingCompilePhaseObservation,
        main_capture: CompiledTrainingCompilePhaseObservation,
        accumulation_capture: Option<CompiledTrainingCompilePhaseObservation>,
    ) -> Self {
        Self {
            objective_forward,
            autograd,
            optimizer_lowering,
            main_capture,
            accumulation_capture,
            partial_flush: None,
            zero_grad: None,
            evaluation: None,
        }
    }

    pub(crate) fn set_auxiliary(
        &mut self,
        partial_flush: Option<CompiledTrainingCompilePhaseObservation>,
        zero_grad: Option<CompiledTrainingCompilePhaseObservation>,
    ) {
        self.partial_flush = partial_flush;
        self.zero_grad = zero_grad;
    }

    pub(crate) fn set_evaluation(&mut self, evaluation: CompiledTrainingCompilePhaseObservation) {
        self.evaluation = Some(evaluation);
    }

    pub const fn compile_count(&self) -> u64 {
        1
    }

    pub const fn objective_forward(&self) -> CompiledTrainingCompilePhaseObservation {
        self.objective_forward
    }

    pub const fn autograd(&self) -> CompiledTrainingCompilePhaseObservation {
        self.autograd
    }

    pub const fn optimizer_lowering(&self) -> CompiledTrainingCompilePhaseObservation {
        self.optimizer_lowering
    }

    pub const fn main_capture(&self) -> CompiledTrainingCompilePhaseObservation {
        self.main_capture
    }

    pub const fn accumulation_capture(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.accumulation_capture
    }

    pub const fn partial_flush(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.partial_flush
    }

    pub const fn zero_grad(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.zero_grad
    }

    pub const fn evaluation(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.evaluation
    }

    /// Checked sum of the instrumented compile phases. The enclosing caller
    /// wall time may be larger; the scoreboard records that remainder.
    pub fn measured_wall_time(&self) -> Option<Duration> {
        [
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
            total.checked_add(phase.wall_time())
        })
    }
}

impl CompiledTrainingInspection {
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
            compile_phases: None,
        }
    }

    pub(crate) fn with_compile_phases(
        mut self,
        compile_phases: Option<CompiledTrainingCompileObservation>,
    ) -> Self {
        self.compile_phases = compile_phases;
        self
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

    pub fn compile_phases(&self) -> Option<&CompiledTrainingCompileObservation> {
        self.compile_phases.as_ref()
    }
}
