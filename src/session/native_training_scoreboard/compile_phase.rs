//! Portable compile-phase observations for compiled training.
//!
//! This module converts backend-neutral construction observations into the
//! versioned scoreboard partition and validates that partition against the
//! compiled program inventory.

use super::inspection::{
    CompiledTrainingCompileObservation, CompiledTrainingCompilePhaseObservation,
    CompiledTrainingRecurrentCaptureObservation,
};
use super::program_report::NativeTrainingProgramReport;
use super::{count, invalid};
use crate::{BenchmarkDuration, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// One typed backend-neutral compilation phase in the portable training
/// scoreboard. Exactly one inventory kind is present: graph nodes for graph
/// construction phases or logical schedule items for captured programs.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingCompilePhase {
    pub(super) wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) graph_node_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) logical_schedule_item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) recurrent_capture: Option<NativeTrainingRecurrentCaptureReport>,
}

/// Disjoint host-wall attribution inside one recurrent phase capture.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingRecurrentCaptureReport {
    pub(super) alias_planning_wall_time: BenchmarkDuration,
    pub(super) preview_schedule_count: u64,
    pub(super) final_schedule_wall_time: BenchmarkDuration,
    pub(super) pure_capture_binding_wall_time: BenchmarkDuration,
    pub(super) effect_assembly_sealing_wall_time: BenchmarkDuration,
    pub(super) recurrent_authentication_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cursor_projection_wall_time: Option<BenchmarkDuration>,
    pub(super) residual_wall_time: BenchmarkDuration,
    pub(super) recurrent_state_count: u64,
}

impl NativeTrainingRecurrentCaptureReport {
    fn from_observation(observation: CompiledTrainingRecurrentCaptureObservation) -> Result<Self> {
        Ok(Self {
            alias_planning_wall_time: BenchmarkDuration::from_duration(
                observation.alias_planning_wall_time(),
            ),
            preview_schedule_count: count(
                observation.preview_schedule_count(),
                "recurrent preview schedule",
            )?,
            final_schedule_wall_time: BenchmarkDuration::from_duration(
                observation.final_schedule_wall_time(),
            ),
            pure_capture_binding_wall_time: BenchmarkDuration::from_duration(
                observation.pure_capture_binding_wall_time(),
            ),
            effect_assembly_sealing_wall_time: BenchmarkDuration::from_duration(
                observation.effect_assembly_sealing_wall_time(),
            ),
            recurrent_authentication_wall_time: BenchmarkDuration::from_duration(
                observation.recurrent_authentication_wall_time(),
            ),
            cursor_projection_wall_time: observation
                .cursor_projection_wall_time()
                .map(BenchmarkDuration::from_duration),
            residual_wall_time: BenchmarkDuration::from_duration(observation.residual_wall_time()),
            recurrent_state_count: count(
                observation.recurrent_state_count(),
                "recurrent capture state",
            )?,
        })
    }

    fn validate(
        self,
        phase_wall_time: Duration,
        expected_preview_schedule_count: u64,
        cursor_projection: bool,
        expected_state_count: u64,
    ) -> Result<()> {
        if self.preview_schedule_count != expected_preview_schedule_count
            || self.cursor_projection_wall_time.is_some() != cursor_projection
            || self.recurrent_state_count != expected_state_count
        {
            return Err(invalid("compiled recurrent capture inventory differs"));
        }
        let total = [
            Some(self.alias_planning_wall_time),
            Some(self.final_schedule_wall_time),
            Some(self.pure_capture_binding_wall_time),
            Some(self.effect_assembly_sealing_wall_time),
            Some(self.recurrent_authentication_wall_time),
            self.cursor_projection_wall_time,
            Some(self.residual_wall_time),
        ]
        .into_iter()
        .flatten()
        .try_fold(Duration::ZERO, |total, duration| {
            let duration = duration
                .to_duration()
                .map_err(|_| invalid("invalid recurrent capture duration"))?;
            total
                .checked_add(duration)
                .ok_or_else(|| invalid("compiled recurrent capture duration overflows"))
        })?;
        if total != phase_wall_time {
            return Err(invalid(
                "compiled recurrent capture timing partition differs",
            ));
        }
        Ok(())
    }

    /// Host wall time spent planning public/state aliases and their preview schedules.
    pub const fn alias_planning_wall_time(&self) -> BenchmarkDuration {
        self.alias_planning_wall_time
    }

    /// Exact number of preview schedule passes used by alias planning.
    pub const fn preview_schedule_count(&self) -> u64 {
        self.preview_schedule_count
    }

    /// Host wall time spent building the final pure schedule.
    pub const fn final_schedule_wall_time(&self) -> BenchmarkDuration {
        self.final_schedule_wall_time
    }

    /// Host wall time spent capturing the pure schedule and binding recurrent states.
    pub const fn pure_capture_binding_wall_time(&self) -> BenchmarkDuration {
        self.pure_capture_binding_wall_time
    }

    /// Host wall time spent assembling effects and sealing the mixed capture.
    pub const fn effect_assembly_sealing_wall_time(&self) -> BenchmarkDuration {
        self.effect_assembly_sealing_wall_time
    }

    /// Host wall time spent authenticating the canonical recurrent capture.
    pub const fn recurrent_authentication_wall_time(&self) -> BenchmarkDuration {
        self.recurrent_authentication_wall_time
    }

    /// Host wall time spent projecting an auxiliary cursor from the main capture.
    pub const fn cursor_projection_wall_time(&self) -> Option<BenchmarkDuration> {
        self.cursor_projection_wall_time
    }

    /// Remaining phase time outside the explicitly measured recurrent stages.
    pub const fn residual_wall_time(&self) -> BenchmarkDuration {
        self.residual_wall_time
    }

    /// Number of logical recurrent states authenticated by this capture.
    pub const fn recurrent_state_count(&self) -> u64 {
        self.recurrent_state_count
    }
}

impl NativeTrainingCompilePhase {
    pub(super) fn from_observation(
        observation: CompiledTrainingCompilePhaseObservation,
    ) -> Result<Self> {
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
            recurrent_capture: observation
                .recurrent_capture()
                .map(NativeTrainingRecurrentCaptureReport::from_observation)
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
        if self.graph_node_count.is_none()
            || self.logical_schedule_item_count.is_some()
            || self.recurrent_capture.is_some()
        {
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

    /// Recurrent capture substage partition for a training phase, when recorded.
    pub const fn recurrent_capture(&self) -> Option<NativeTrainingRecurrentCaptureReport> {
        self.recurrent_capture
    }
}

/// Exact partition of one caller-observed backend-neutral training compile.
/// Residual time contains wrapper work outside the timed compiler phases; the
/// checked sum of all phase durations and residual equals `compile_wall_time`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingCompilePhaseReport {
    pub(super) compile_count: u64,
    pub(super) objective_forward: NativeTrainingCompilePhase,
    pub(super) autograd: NativeTrainingCompilePhase,
    pub(super) optimizer_lowering: NativeTrainingCompilePhase,
    pub(super) main_capture: NativeTrainingCompilePhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) accumulation_capture: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) partial_flush: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) zero_grad: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) evaluation: Option<NativeTrainingCompilePhase>,
    pub(super) residual_wall_time: BenchmarkDuration,
}

pub(super) struct CompileProgramInventory<'a> {
    pub(super) main: &'a NativeTrainingProgramReport,
    pub(super) accumulation: Option<&'a NativeTrainingProgramReport>,
    pub(super) partial_flush: Option<&'a NativeTrainingProgramReport>,
    pub(super) zero_grad: Option<&'a NativeTrainingProgramReport>,
    pub(super) evaluation: Option<&'a NativeTrainingProgramReport>,
}

pub(super) struct CompilePhaseValidationContext<'a> {
    pub(super) require_recurrent_capture: bool,
    pub(super) compile_wall_time: BenchmarkDuration,
    pub(super) recurrent_state_count: u64,
    pub(super) programs: CompileProgramInventory<'a>,
}

impl NativeTrainingCompilePhaseReport {
    pub(super) fn from_observation(
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

    pub(super) fn validate(&self, context: CompilePhaseValidationContext<'_>) -> Result<()> {
        let CompilePhaseValidationContext {
            require_recurrent_capture,
            compile_wall_time,
            recurrent_state_count,
            programs:
                CompileProgramInventory {
                    main,
                    accumulation,
                    partial_flush,
                    zero_grad,
                    evaluation,
                },
        } = context;
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
        let recurrent_phases = [
            self.main_capture,
            self.accumulation_capture.unwrap_or(self.main_capture),
            self.partial_flush.unwrap_or(self.main_capture),
            self.zero_grad.unwrap_or(self.main_capture),
        ];
        if !require_recurrent_capture {
            if recurrent_phases
                .into_iter()
                .any(|phase| phase.recurrent_capture.is_some())
                || self
                    .evaluation
                    .is_some_and(|phase| phase.recurrent_capture.is_some())
            {
                return Err(invalid(
                    "legacy compiled phase has recurrent capture evidence",
                ));
            }
        } else {
            let main_state_count = main
                .recurrent_state_count
                .ok_or_else(|| invalid("main recurrent-state inventory is absent"))?;
            if main_state_count != recurrent_state_count {
                return Err(invalid("main recurrent-state inventory differs"));
            }
            let main = self
                .main_capture
                .recurrent_capture
                .ok_or_else(|| invalid("main recurrent capture evidence is absent"))?;
            main.validate(self.main_capture.duration()?, 2, false, main_state_count)?;
            match (self.accumulation_capture, accumulation) {
                (Some(phase), Some(program)) => {
                    let state_count = program.recurrent_state_count.ok_or_else(|| {
                        invalid("accumulation recurrent-state inventory is absent")
                    })?;
                    let capture = phase.recurrent_capture.ok_or_else(|| {
                        invalid("accumulation recurrent capture evidence is absent")
                    })?;
                    capture.validate(phase.duration()?, 3, true, state_count)?;
                }
                (None, None) => {}
                _ => return Err(invalid("accumulation recurrent-state inventory differs")),
            }
            match (self.partial_flush, partial_flush) {
                (Some(phase), Some(program)) => {
                    let state_count = program.recurrent_state_count.ok_or_else(|| {
                        invalid("partial-flush recurrent-state inventory is absent")
                    })?;
                    let capture = phase.recurrent_capture.ok_or_else(|| {
                        invalid("partial-flush recurrent capture evidence is absent")
                    })?;
                    capture.validate(phase.duration()?, 3, true, state_count)?;
                }
                (None, None) => {}
                _ => return Err(invalid("partial-flush recurrent-state inventory differs")),
            }
            match (self.zero_grad, zero_grad) {
                (Some(phase), Some(program)) => {
                    let state_count = program
                        .recurrent_state_count
                        .ok_or_else(|| invalid("zero-grad recurrent-state inventory is absent"))?;
                    phase
                        .recurrent_capture
                        .ok_or_else(|| invalid("zero-grad recurrent capture evidence is absent"))?
                        .validate(phase.duration()?, 1, true, state_count)?;
                }
                (None, None) => {}
                _ => return Err(invalid("zero-grad recurrent-state inventory differs")),
            }
            if self
                .evaluation
                .is_some_and(|phase| phase.recurrent_capture.is_some())
                || evaluation.is_some_and(|program| program.recurrent_state_count.is_some())
            {
                return Err(invalid("evaluation has recurrent capture evidence"));
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
