use super::{
    CompiledAdamWFlushResult, CompiledTrainingWindowReset, CompiledTrainingWindowTopology, training,
};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledTrainingWindowProgress {
    pub(super) replay_step: u64,
    pub(super) optimizer_step: u64,
    pub(super) accumulation_index: u64,
    pub(super) discarded_microbatches: u64,
    pub(super) flushed_window_count: u64,
    pub(super) flushed_microbatch_count: u64,
    pub(super) reset_transition_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledTrainingWindowFlush {
    pub(super) flushed_microbatches: u64,
    pub(super) optimizer_step: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompiledTrainingWindowProgressError {
    ReplayStepOverflow,
    AccumulationIndexOverflow,
    OptimizerStepOverflow,
    DiscardedMicrobatchCountOverflow,
    ResetTransitionCountOverflow,
    PartialFlushCountOverflow,
    FlushedMicrobatchCountOverflow,
    InvalidAccumulationProgress,
    InvalidDiscardedProgress,
    InvalidResetProgress,
    InvalidFlushProgress,
    ProgressUnderflow,
    ProgressOverflow,
    ReplayOptimizerDiverged,
}

pub(super) fn adamw_window_progress_error(error: CompiledTrainingWindowProgressError) -> Error {
    training(match error {
        CompiledTrainingWindowProgressError::ReplayStepOverflow => {
            "compiled training step overflow"
        }
        CompiledTrainingWindowProgressError::AccumulationIndexOverflow => {
            "compiled AdamW accumulation index overflow"
        }
        CompiledTrainingWindowProgressError::OptimizerStepOverflow => {
            "compiled AdamW optimizer step overflow"
        }
        CompiledTrainingWindowProgressError::DiscardedMicrobatchCountOverflow => {
            "compiled AdamW discarded microbatch count overflow"
        }
        CompiledTrainingWindowProgressError::ResetTransitionCountOverflow => {
            "compiled AdamW reset transition count overflow"
        }
        CompiledTrainingWindowProgressError::PartialFlushCountOverflow => {
            "compiled AdamW partial flush count overflow"
        }
        CompiledTrainingWindowProgressError::FlushedMicrobatchCountOverflow => {
            "compiled AdamW flushed microbatch count overflow"
        }
        CompiledTrainingWindowProgressError::InvalidAccumulationProgress => {
            "compiled AdamW checkpoint accumulation progress is invalid"
        }
        CompiledTrainingWindowProgressError::InvalidDiscardedProgress => {
            "compiled AdamW checkpoint discarded progress is invalid"
        }
        CompiledTrainingWindowProgressError::InvalidResetProgress => {
            "compiled AdamW checkpoint reset progress is invalid"
        }
        CompiledTrainingWindowProgressError::InvalidFlushProgress => {
            "compiled AdamW checkpoint flushed progress is invalid"
        }
        CompiledTrainingWindowProgressError::ProgressUnderflow => {
            "compiled AdamW checkpoint progress underflows"
        }
        CompiledTrainingWindowProgressError::ProgressOverflow => {
            "compiled AdamW checkpoint progress overflows"
        }
        CompiledTrainingWindowProgressError::ReplayOptimizerDiverged => {
            "compiled AdamW checkpoint replay and optimizer progress diverged"
        }
    })
}

pub(super) fn adamw_window_progress<T>(
    result: std::result::Result<T, CompiledTrainingWindowProgressError>,
) -> Result<T> {
    result.map_err(adamw_window_progress_error)
}

impl CompiledTrainingWindowFlush {
    pub(super) const fn did_update(self) -> bool {
        self.flushed_microbatches != 0
    }

    pub(super) const fn into_adamw_result(self) -> CompiledAdamWFlushResult {
        CompiledAdamWFlushResult {
            flushed_microbatches: self.flushed_microbatches,
            optimizer_step: self.optimizer_step,
            clip_report: None,
            window_loss_report: None,
        }
    }
}

impl CompiledTrainingWindowProgress {
    pub(super) const INITIAL: Self = Self {
        replay_step: 0,
        optimizer_step: 0,
        accumulation_index: 0,
        discarded_microbatches: 0,
        flushed_window_count: 0,
        flushed_microbatch_count: 0,
        reset_transition_count: 0,
    };

    pub(super) fn advance_replay(
        self,
        accumulation_steps: u64,
    ) -> std::result::Result<Self, CompiledTrainingWindowProgressError> {
        let replay_step = self
            .replay_step
            .checked_add(1)
            .ok_or(CompiledTrainingWindowProgressError::ReplayStepOverflow)?;
        let next_index = self
            .accumulation_index
            .checked_add(1)
            .ok_or(CompiledTrainingWindowProgressError::AccumulationIndexOverflow)?;
        let (optimizer_step, accumulation_index) = if next_index == accumulation_steps {
            (
                self.optimizer_step
                    .checked_add(1)
                    .ok_or(CompiledTrainingWindowProgressError::OptimizerStepOverflow)?,
                0,
            )
        } else {
            (self.optimizer_step, next_index)
        };
        let next = Self {
            replay_step,
            optimizer_step,
            accumulation_index,
            ..self
        };
        validate_training_window_progress(next, accumulation_steps)?;
        Ok(next)
    }

    pub(super) fn cancel(
        self,
        accumulation_steps: u64,
    ) -> std::result::Result<(Self, CompiledTrainingWindowReset), CompiledTrainingWindowProgressError>
    {
        if self.accumulation_index == 0 {
            return Ok((
                self,
                CompiledTrainingWindowReset {
                    discarded_microbatches: 0,
                },
            ));
        }
        let discarded = self.accumulation_index;
        let next = Self {
            accumulation_index: 0,
            discarded_microbatches: self
                .discarded_microbatches
                .checked_add(discarded)
                .ok_or(CompiledTrainingWindowProgressError::DiscardedMicrobatchCountOverflow)?,
            ..self
        };
        validate_training_window_progress(next, accumulation_steps)?;
        Ok((
            next,
            CompiledTrainingWindowReset {
                discarded_microbatches: discarded,
            },
        ))
    }

    pub(super) fn record_reset_transition(
        mut self,
    ) -> std::result::Result<Self, CompiledTrainingWindowProgressError> {
        self.reset_transition_count = self
            .reset_transition_count
            .checked_add(1)
            .ok_or(CompiledTrainingWindowProgressError::ResetTransitionCountOverflow)?;
        Ok(self)
    }

    pub(super) fn flush_partial(
        self,
        accumulation_steps: u64,
    ) -> std::result::Result<(Self, CompiledTrainingWindowFlush), CompiledTrainingWindowProgressError>
    {
        if self.accumulation_index == 0 {
            return Ok((
                self,
                CompiledTrainingWindowFlush {
                    flushed_microbatches: 0,
                    optimizer_step: self.optimizer_step,
                },
            ));
        }
        let flushed_microbatches = self.accumulation_index;
        let next = Self {
            optimizer_step: self
                .optimizer_step
                .checked_add(1)
                .ok_or(CompiledTrainingWindowProgressError::OptimizerStepOverflow)?,
            accumulation_index: 0,
            flushed_window_count: self
                .flushed_window_count
                .checked_add(1)
                .ok_or(CompiledTrainingWindowProgressError::PartialFlushCountOverflow)?,
            flushed_microbatch_count: self
                .flushed_microbatch_count
                .checked_add(flushed_microbatches)
                .ok_or(CompiledTrainingWindowProgressError::FlushedMicrobatchCountOverflow)?,
            ..self
        };
        validate_training_window_progress(next, accumulation_steps)?;
        Ok((
            next,
            CompiledTrainingWindowFlush {
                flushed_microbatches,
                optimizer_step: next.optimizer_step,
            },
        ))
    }
}

pub(super) fn validate_adamw_progress(
    progress: CompiledTrainingWindowProgress,
    accumulation_steps: u64,
) -> Result<()> {
    validate_training_window_progress(progress, accumulation_steps)
        .map_err(adamw_window_progress_error)
}

pub(super) fn validate_training_window_progress(
    progress: CompiledTrainingWindowProgress,
    accumulation_steps: u64,
) -> std::result::Result<(), CompiledTrainingWindowProgressError> {
    let CompiledTrainingWindowProgress {
        replay_step,
        optimizer_step,
        accumulation_index,
        discarded_microbatches,
        flushed_window_count,
        flushed_microbatch_count,
        reset_transition_count,
    } = progress;
    if accumulation_steps == 0 {
        return Err(CompiledTrainingWindowProgressError::InvalidAccumulationProgress);
    }
    let topology =
        CompiledTrainingWindowTopology::from_validated_parts(accumulation_steps, false, false);
    if accumulation_index >= accumulation_steps {
        return Err(CompiledTrainingWindowProgressError::InvalidAccumulationProgress);
    }
    if !topology.accumulating()
        && (discarded_microbatches != 0
            || flushed_window_count != 0
            || flushed_microbatch_count != 0
            || reset_transition_count != 0)
    {
        return Err(CompiledTrainingWindowProgressError::InvalidDiscardedProgress);
    }
    if reset_transition_count > discarded_microbatches {
        return Err(CompiledTrainingWindowProgressError::InvalidResetProgress);
    }
    let maximum_flushed_microbatches = flushed_window_count
        .checked_mul(
            accumulation_steps
                .checked_sub(1)
                .ok_or(CompiledTrainingWindowProgressError::ProgressUnderflow)?,
        )
        .ok_or(CompiledTrainingWindowProgressError::ProgressOverflow)?;
    if flushed_window_count > optimizer_step
        || (flushed_window_count == 0) != (flushed_microbatch_count == 0)
        || (flushed_window_count != 0
            && (flushed_microbatch_count < flushed_window_count
                || flushed_microbatch_count > maximum_flushed_microbatches))
    {
        return Err(CompiledTrainingWindowProgressError::InvalidFlushProgress);
    }
    let complete_windows = optimizer_step - flushed_window_count;
    let expected_replay = complete_windows
        .checked_mul(accumulation_steps)
        .and_then(|step| step.checked_add(flushed_microbatch_count))
        .and_then(|step| step.checked_add(accumulation_index))
        .and_then(|step| step.checked_add(discarded_microbatches))
        .ok_or(CompiledTrainingWindowProgressError::ProgressOverflow)?;
    if replay_step != expected_replay {
        return Err(CompiledTrainingWindowProgressError::ReplayOptimizerDiverged);
    }
    Ok(())
}
