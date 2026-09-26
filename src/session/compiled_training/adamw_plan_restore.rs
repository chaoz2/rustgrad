//! AdamW checkpoint-frontier authentication and portable restoration.

use super::*;

struct ValidatedAdamWCheckpointFrontier {
    replay_step: u64,
    values: BTreeMap<RecurrentStateKey, TensorData>,
    versions: BTreeMap<RecurrentStateKey, u64>,
    progress: CompiledTrainingWindowProgress,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AdamWCheckpointRestoreCounts {
    pub(super) borrowed_plan_clones: usize,
    pub(super) consumed_plan_restores: usize,
}

#[cfg(test)]
std::thread_local! {
    static ADAMW_CHECKPOINT_RESTORE_COUNTS: std::cell::Cell<AdamWCheckpointRestoreCounts> =
        const { std::cell::Cell::new(AdamWCheckpointRestoreCounts {
            borrowed_plan_clones: 0,
            consumed_plan_restores: 0,
        }) };
}

#[cfg(test)]
fn record_adamw_checkpoint_restore(update: impl FnOnce(&mut AdamWCheckpointRestoreCounts)) {
    ADAMW_CHECKPOINT_RESTORE_COUNTS.with(|counts| {
        let mut next = counts.get();
        update(&mut next);
        counts.set(next);
    });
}

#[cfg(test)]
pub(super) fn adamw_checkpoint_restore_counts() -> AdamWCheckpointRestoreCounts {
    ADAMW_CHECKPOINT_RESTORE_COUNTS.with(std::cell::Cell::get)
}

impl CompiledAdamWPlan {
    /// Returns an independent plan whose recurrent frontier is restored from
    /// one checkpoint without rebuilding the Graph, derivatives, schedules,
    /// captures, accumulation/partial-flush transitions, or attached evaluation
    /// program.
    ///
    /// The checkpoint must authenticate this exact compiled program and its
    /// accumulation, dropout, frozen-parameter, input, clipping, loss-scaling,
    /// and partial-flush policies. The source plan remains unchanged on both
    /// success and failure, and each returned plan may be prepared or restored
    /// independently.
    pub fn restore_checkpoint(&self, checkpoint: &CompiledAdamWCheckpoint) -> Result<Self> {
        let decoded = checkpoint.decoded();
        let frontier = self.validate_checkpoint_frontier(decoded)?;
        #[cfg(test)]
        record_adamw_checkpoint_restore(|counts| counts.borrowed_plan_clones += 1);
        self.clone().apply_checkpoint_frontier(frontier)
    }

    pub(super) fn restore_checkpoint_owned(
        self,
        checkpoint: &CompiledAdamWCheckpoint,
    ) -> Result<Self> {
        let frontier = self.validate_checkpoint_frontier(checkpoint.decoded())?;
        #[cfg(test)]
        record_adamw_checkpoint_restore(|counts| counts.consumed_plan_restores += 1);
        self.apply_checkpoint_frontier(frontier)
    }

    fn validate_checkpoint_frontier(
        &self,
        decoded: &DecodedAdamWCheckpoint,
    ) -> Result<ValidatedAdamWCheckpointFrontier> {
        let topology = CompiledTrainingWindowTopology::from_contract(&self.contract);
        if self.contract.gradient_accumulation_steps != decoded.accumulation_steps {
            return Err(training(
                "compiled AdamW checkpoint accumulation policy mismatch",
            ));
        }
        if self.contract.window_loss_report != decoded.window_loss_report {
            return Err(training(
                "compiled AdamW checkpoint window-loss reporting policy mismatch",
            ));
        }
        match (
            topology.retains_token_count(),
            decoded.accumulated_token_count,
        ) {
            (true, Some(count)) => validate_retained_token_count(
                &self.inner.inputs,
                self.contract
                    .token_weight_policy
                    .as_ref()
                    .expect("retained token counts require token weighting"),
                decoded.accumulation_index,
                count,
                self.contract.allow_zero_valid_token_microbatches,
            )?,
            (false, None) => {}
            _ => {
                return Err(training(
                    "compiled AdamW checkpoint token-weighting policy mismatch",
                ));
            }
        }
        if self.capture_identity() != decoded.capture_identity {
            return Err(training(
                "compiled AdamW checkpoint capture identity mismatch",
            ));
        }
        if decoded.accumulation_capture_identity.is_some()
            && self.accumulation_capture_identity() != decoded.accumulation_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint accumulation capture identity mismatch",
            ));
        }
        if decoded.flush_capture_identity.is_some()
            && self.flush_capture_identity() != decoded.flush_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint partial flush capture identity mismatch",
            ));
        }
        if decoded.reset_capture_identity.is_some()
            && self.zero_grad_capture_identity() != decoded.reset_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint zero-grad capture identity mismatch",
            ));
        }
        match (self.contract.dropout, decoded.dropout_block_counter) {
            (None, None) => {}
            (None, Some(_)) => {
                return Err(training(
                    "compiled AdamW dropout checkpoint requires dropout restore",
                ));
            }
            (Some(_), None) => {
                return Err(training(
                    "compiled AdamW checkpoint has no dropout block counter",
                ));
            }
            (Some(dropout), Some(counter)) => {
                let expected = decoded
                    .replay_step
                    .checked_mul(dropout.blocks_per_replay)
                    .ok_or_else(|| training("compiled dropout counter progress overflows"))?;
                if counter != expected {
                    return Err(training(
                        "compiled dropout counter and replay progress diverged",
                    ));
                }
            }
        }

        let mut values = decoded
            .parameters
            .iter()
            .map(|(name, value)| (RecurrentStateKey::parameter(name.clone()), value.clone()))
            .collect::<BTreeMap<_, _>>();
        for (name, value) in &decoded.first_moments {
            values.insert(
                adamw_parameter_key(name, AdamWParameterState::FirstMoment),
                value.clone(),
            );
        }
        for (name, value) in &decoded.second_moments {
            values.insert(
                adamw_parameter_key(name, AdamWParameterState::SecondMoment),
                value.clone(),
            );
        }
        for (name, value) in &decoded.gradient_accumulators {
            values.insert(
                adamw_parameter_key(name, AdamWParameterState::GradientAccumulator),
                value.clone(),
            );
        }
        values.insert(
            adamw_global_key(AdamWGlobalState::Step),
            TensorData::from_scalars(
                Shape::from([]),
                DType::U64,
                [Scalar::U(decoded.optimizer_step)],
            )?,
        );
        if topology.accumulating() {
            values.insert(
                adamw_global_key(AdamWGlobalState::AccumulationIndex),
                TensorData::from_scalars(
                    Shape::from([]),
                    DType::U64,
                    [Scalar::U(decoded.accumulation_index)],
                )?,
            );
        }
        if let Some(count) = decoded.accumulated_token_count {
            values.insert(
                adamw_global_key(AdamWGlobalState::AccumulatedTokenCount),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(count)])?,
            );
        }
        if let Some(numerator) = &decoded.accumulated_loss_numerator {
            values.insert(
                adamw_global_key(AdamWGlobalState::AccumulatedLossNumerator),
                numerator.clone(),
            );
        }
        if let Some(counter) = decoded.dropout_block_counter {
            values.insert(
                RecurrentStateKey::dropout_counter(),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(counter)])?,
            );
        }

        let progress = CompiledTrainingWindowProgress {
            replay_step: decoded.replay_step,
            optimizer_step: decoded.optimizer_step,
            accumulation_index: decoded.accumulation_index,
            discarded_microbatches: decoded.discarded_microbatches,
            flushed_window_count: decoded.flushed_window_count,
            flushed_microbatch_count: decoded.flushed_microbatch_count,
            reset_transition_count: decoded.reset_transition_count,
        };
        let optimizer_version = decoded
            .replay_step
            .checked_add(decoded.flushed_window_count)
            .ok_or_else(|| training("compiled AdamW checkpoint state version overflows"))?;
        let reset_version = optimizer_version
            .checked_add(decoded.reset_transition_count)
            .ok_or_else(|| training("compiled AdamW checkpoint reset state version overflows"))?;
        let versions = values
            .keys()
            .cloned()
            .map(|key| {
                let version = if self.inner.workload_buffers.contains_key(&key) {
                    decoded.replay_step
                } else if is_adamw_accumulation_reset_state(&key) {
                    reset_version
                } else {
                    optimizer_version
                };
                (key, version)
            })
            .collect();

        Ok(ValidatedAdamWCheckpointFrontier {
            replay_step: decoded.replay_step,
            values,
            versions,
            progress,
        })
    }

    fn apply_checkpoint_frontier(self, frontier: ValidatedAdamWCheckpointFrontier) -> Result<Self> {
        let Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract,
            progress: _,
            evaluation,
            compile_phases,
        } = self;
        let inner = inner.restore_frontier_with_versions(
            frontier.replay_step,
            frontier.values,
            frontier.versions,
        )?;
        let partial_flush = partial_flush
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        let zero_grad = zero_grad
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract,
            progress: frontier.progress,
            evaluation,
            compile_phases,
        })
    }

    /// Compatibility constructor that compiles an exact program and then
    /// restores its portable AdamW frontier before runtime preparation.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        if !config.frozen_parameters.is_empty() {
            return Err(training(
                "compiled AdamW raw parameters cannot resolve frozen parameter names",
            ));
        }
        let decoded = checkpoint.decoded();
        let parameters = decoded
            .parameters
            .iter()
            .map(|(name, value)| TrainingParameterInit::new(name.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()?;
        Self::compile(config, parameters, build)?.restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor that compiles a module-bound program and then
    /// restores its portable frontier without preparing a runtime.
    pub fn compile_module_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_module(config, module, build)?.restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor that compiles the explicit residual-dropout
    /// program and then restores its optimizer and Threefry-counter frontier.
    pub fn compile_module_with_dropout_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_module_with_dropout(config, dropout, module, build)?
            .restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor for a compiler-owned token-mean loss that
    /// restores its portable optimizer and Threefry-counter frontier.
    pub fn compile_token_mean_module_with_dropout_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_token_mean_module_with_dropout(config, dropout, module, build)?
            .restore_checkpoint(checkpoint)
    }
}
