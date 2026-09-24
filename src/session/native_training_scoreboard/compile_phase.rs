//! Portable compile-phase observations for compiled training.
//!
//! This module converts backend-neutral construction observations into the
//! versioned scoreboard partition and validates that partition against the
//! compiled program inventory.

use super::inspection::{
    CompiledTrainingCompileObservation, CompiledTrainingCompilePhaseObservation,
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

    pub(super) fn validate(
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
