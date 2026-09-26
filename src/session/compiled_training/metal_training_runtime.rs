use super::*;

/// Resource-free Metal rendering of one compiled AdamW plan. Preparing it
/// uploads the plan's parameter, moment, and optimizer-step frontier into the
/// existing epoch-swapped Metal runtime.
pub struct MetalCompiledAdamWPlan {
    pub(super) inner: MetalCompiledTrainingPlan,
    pub(super) accumulation_capture_identity: Option<u64>,
    pub(super) partial_flush: Option<MetalFixedStateTransitionPlan>,
    pub(super) progress: CompiledTrainingWindowProgress,
    pub(super) flush_capture_identity: Option<u64>,
    pub(super) contract: MetalAdamWContract,
}

/// Device-resident AdamW training session backed by one fixed Metal capture.
/// Parameters and optimizer slots remain in the double-buffered device state
/// frontier between calls. Batch inputs and learning rate cross the host
/// boundary on every step; requested outputs cross only when the caller uses
/// the observed [`MetalCompiledAdamW::step`] path.
pub struct MetalCompiledAdamW {
    inner: MetalCompiledTrainingProgram,
    accumulation_capture_identity: Option<u64>,
    partial_flush: Option<MetalFixedStateTransitionSession>,
    progress: CompiledTrainingWindowProgress,
    flush_capture_identity: Option<u64>,
    contract: MetalAdamWContract,
}

/// One committed Metal AdamW step plus its exact device execution report.
pub struct MetalCompiledAdamWStepResult {
    inner: CompiledAdamWStepResult,
    report: MetalDeviceRunReport,
}

/// One committed Metal AdamW step whose loss and named outputs remained on the
/// device. The exact replay and optimizer progress plus device report remain
/// available without manufacturing an observed [`CompiledTrainingStep`].
pub struct MetalCompiledAdamWCommitResult {
    progress: CompiledTrainingWindowProgress,
    capture_identity: u64,
    report: MetalDeviceRunReport,
}

/// One committed strict-Metal partial-window flush and its exact device report.
pub struct MetalCompiledAdamWFlushResult {
    inner: CompiledAdamWFlushResult,
    report: Option<MetalDeviceRunReport>,
}

impl MetalCompiledAdamWFlushResult {
    pub fn flushed_microbatches(&self) -> u64 {
        self.inner.flushed_microbatches()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    /// Exact device report for a committed update. Empty flushes execute no
    /// device invocation and therefore return `None`.
    pub fn report(&self) -> Option<&MetalDeviceRunReport> {
        self.report.as_ref()
    }
}

impl CompiledAdamWFlush for MetalCompiledAdamWFlushResult {
    fn flushed_microbatches(&self) -> u64 {
        MetalCompiledAdamWFlushResult::flushed_microbatches(self)
    }

    fn did_update(&self) -> bool {
        MetalCompiledAdamWFlushResult::did_update(self)
    }

    fn optimizer_step(&self) -> u64 {
        MetalCompiledAdamWFlushResult::optimizer_step(self)
    }
}

impl MetalCompiledAdamWCommitResult {
    pub fn step(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn optimizer_step(&self) -> u64 {
        self.progress.optimizer_step
    }

    pub fn accumulation_index(&self) -> u64 {
        self.progress.accumulation_index
    }

    pub fn did_update(&self) -> bool {
        self.progress.accumulation_index == 0
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl MetalCompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    pub fn accumulation_index(&self) -> u64 {
        self.inner.accumulation_index()
    }

    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl CompiledTrainingStep for MetalCompiledAdamWStepResult {
    fn loss(&self) -> &TensorData {
        MetalCompiledAdamWStepResult::loss(self)
    }

    fn loss_aggregation_weight(&self) -> u64 {
        MetalCompiledAdamWStepResult::loss_weight(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        MetalCompiledAdamWStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        MetalCompiledAdamWStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        MetalCompiledAdamWStepResult::capture_identity(self)
    }
}

impl CompiledAdamWStep for MetalCompiledAdamWStepResult {
    fn optimizer_step(&self) -> u64 {
        MetalCompiledAdamWStepResult::optimizer_step(self)
    }

    fn accumulation_index(&self) -> u64 {
        MetalCompiledAdamWStepResult::accumulation_index(self)
    }

    fn loss_weight(&self) -> u64 {
        MetalCompiledAdamWStepResult::loss_weight(self)
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for MetalSessionTarget {
    type Session = MetalCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        let rendered = plan.metal_plan(self.renderer().clone())?;
        match self.scoreboard_context() {
            Some(context) => {
                rendered.prepare_with_scoreboard(self.device().clone(), context.clone())
            }
            None => rendered.prepare(self.device().clone()),
        }
    }
}

impl MetalCompiledAdamWPlan {
    pub fn deployment_identity(&self) -> u64 {
        self.inner.inner.deployment_identity()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.program_identity
    }

    /// Stable identity of the exact mixed transition executed by CPU and
    /// represented by this strict-Metal state-only plan.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.contract.gradient_accumulation_steps
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.contract.max_gradient_norm
    }

    pub fn loss_scale(&self) -> f32 {
        self.contract.loss_scale
    }

    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.contract.dropout.map(|dropout| dropout.config)
    }

    pub fn dropout_blocks_per_replay(&self) -> Option<u64> {
        self.contract
            .dropout
            .map(|dropout| dropout.blocks_per_replay)
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.inner.summary()
    }

    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.inner.rendered_items()
    }

    /// Creates all native resources and uploads the captured recurrent
    /// frontier once. No training step is executed during preparation.
    pub fn prepare(self, device: MetalDevice) -> Result<MetalCompiledAdamW> {
        self.prepare_inner(device, None)
    }

    /// Creates the persistent training session and binds an epoch-state
    /// scoreboard before the first step can execute.
    pub fn prepare_with_scoreboard(
        self,
        device: MetalDevice,
        context: MetalScoreboardContext,
    ) -> Result<MetalCompiledAdamW> {
        let recorder = MetalSessionScoreboard::new_epoch_state(context, &self.inner.inner);
        self.prepare_inner(device, Some(recorder))
    }

    fn prepare_inner(
        self,
        device: MetalDevice,
        recorder: Option<MetalSessionScoreboard>,
    ) -> Result<MetalCompiledAdamW> {
        let inner = self.inner.prepare(device.clone(), recorder)?;
        let partial_flush = self
            .partial_flush
            .map(|plan| {
                plan.prepare(device, &inner.session)
                    .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledAdamW {
            inner,
            accumulation_capture_identity: self.accumulation_capture_identity,
            partial_flush,
            progress: self.progress,
            flush_capture_identity: self.flush_capture_identity,
            contract: self.contract,
        })
    }
}

impl MetalCompiledAdamW {
    fn prepare_step(
        &self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<(CompiledTrainingWindowProgress, BTreeMap<String, TensorData>)> {
        let inputs = self.inner.prepare_inputs(inputs, learning_rate)?;
        let next = adamw_window_progress(
            self.progress
                .advance_replay(self.contract.gradient_accumulation_steps),
        )?;
        if let Some(dropout) = self.contract.dropout {
            expected_dropout_counter(dropout, next.replay_step)?;
        }
        Ok((next, inputs))
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWStepResult> {
        let (next, provided) = self.prepare_step(inputs, learning_rate)?;
        let MetalCompiledTrainingRun {
            loss,
            outputs,
            report,
        } = self.inner.run(&provided)?;
        self.progress = next;
        let inner = adamw_step_result(
            CompiledTrainingStepResult {
                loss,
                loss_aggregation_weight: 1,
                outputs,
                step: self.progress.replay_step,
                capture_identity: self.inner.program_identity,
                observations: Vec::new(),
            },
            self.progress,
            1,
            self.contract.gradient_accumulation_steps,
            false,
            false,
        );
        Ok(MetalCompiledAdamWStepResult { inner, report })
    }

    /// Executes and commits the identical captured training program while
    /// leaving its loss and named outputs on the device. Batch inputs and the
    /// learning rate are still staged, the complete inactive state bank is
    /// produced, and successful replay/optimizer progress advances normally.
    /// Use [`Self::step`] whenever the caller needs to observe loss or outputs.
    pub fn step_without_host_outputs(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWCommitResult> {
        let (next, provided) = self.prepare_step(inputs, learning_rate)?;
        let report = self.inner.run_without_host_outputs(&provided)?;
        self.progress = next;
        Ok(MetalCompiledAdamWCommitResult {
            progress: self.progress,
            capture_identity: self.inner.program_identity,
            report,
        })
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<MetalCompiledEvaluationResult> {
        self.inner.evaluate(inputs)
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }

    /// Returns preparation evidence for both stateless evaluators sharing the
    /// training session's physical parameter banks. Imported trainable state
    /// contributes zero resident or initial-state uploads.
    pub fn evaluation_preparation_reports(
        &self,
    ) -> Option<[&crate::runtime::metal::MetalDevicePreparationReport; 2]> {
        self.inner
            .evaluation
            .as_ref()
            .map(|(evaluation, _, _)| evaluation.preparation_reports())
    }

    pub fn evaluation_summaries(&self) -> Option<[&MetalDeviceSessionSummary; 2]> {
        self.inner
            .evaluation
            .as_ref()
            .map(|(evaluation, _, _)| evaluation.summaries())
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.contract.gradient_accumulation_steps
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.contract.max_gradient_norm
    }

    pub fn loss_scale(&self) -> f32 {
        self.contract.loss_scale
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.program_identity
    }

    pub fn metal_session(&self) -> &MetalDeviceSession {
        &self.inner.session
    }

    /// Returns the opt-in successful-step recorder, when preparation enabled it.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.inner
            .scoreboard
            .as_ref()
            .map(MetalScoreboardObserver::recorder)
    }

    /// Returns a deterministic snapshot of all successfully observed steps.
    pub fn execution_scoreboard_report(
        &self,
    ) -> std::result::Result<Option<MetalSessionScoreboardReport>, MetalScoreboardError> {
        self.execution_scoreboard()
            .map(MetalSessionScoreboard::report)
            .transpose()
    }

    /// Returns the first fail-soft measurement error, if recording froze.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.inner
            .scoreboard
            .as_ref()
            .and_then(MetalScoreboardObserver::first_error)
    }

    /// Downloads every currently committed recurrent value once and returns
    /// it under the optimizer's semantic state keys.
    fn state_snapshots(&self) -> Result<BTreeMap<RecurrentStateKey, TensorData>> {
        self.inner.state_snapshots()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(self.state_snapshots()?, AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(self.state_snapshots()?, AdamWParameterState::SecondMoment)
    }

    pub fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(
            self.state_snapshots()?,
            AdamWParameterState::GradientAccumulator,
        )
    }

    pub fn accumulation_index(&self) -> Result<u64> {
        Ok(self.progress.accumulation_index)
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        Ok(self.progress.optimizer_step)
    }

    /// Explicit diagnostic download of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.contract
            .dropout
            .map(|_| {
                Ok(self
                    .state_snapshots()?
                    .get(&RecurrentStateKey::dropout_counter())
                    .ok_or_else(|| training("compiled Metal dropout counter is absent"))?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()
    }

    /// Clears a retained partial window entirely inside the epoch-swapped
    /// device frontier. No training run or host gradient download is performed.
    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        let (next, reset) = adamw_window_progress(
            self.progress
                .cancel(self.contract.gradient_accumulation_steps),
        )?;
        if !reset.did_discard() {
            return Ok(reset);
        }
        let state_inputs = self
            .inner
            .session
            .state_inputs()
            .iter()
            .map(|input| (input.name.as_str(), &input.desc))
            .collect::<BTreeMap<_, _>>();
        let mut replacements = BTreeMap::new();
        for (input, key) in &self.inner.state_input_keys {
            if is_adamw_accumulation_reset_state(key) {
                let desc = state_inputs
                    .get(input.as_str())
                    .ok_or_else(|| training("compiled Metal reset state is absent"))?;
                replacements.insert(
                    input.clone(),
                    TensorData::zeros_with_dtype(desc.shape.clone(), desc.dtype)?,
                );
            }
        }
        let expected = self
            .inner
            .state_input_keys
            .values()
            .filter(|key| is_adamw_accumulation_reset_state(key))
            .count();
        if replacements.len() != expected
            || !replacements
                .values()
                .any(|value| value.dtype() == DType::U64)
        {
            return Err(training("compiled Metal reset state inventory mismatch"));
        }
        self.inner
            .session
            .replace_fixed_state(replacements)
            .map_err(metal_training_error)?;
        self.progress = next;
        Ok(reset)
    }

    /// Commits a retained partial window through the separately rendered
    /// state-only capture while sharing the live epoch banks and queue.
    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWFlushResult> {
        validate_step_inputs(&BTreeMap::new(), &BTreeMap::new(), &learning_rate)?;
        let (next, flush) = adamw_window_progress(
            self.progress
                .flush_partial(self.contract.gradient_accumulation_steps),
        )?;
        if !flush.did_update() {
            return Ok(MetalCompiledAdamWFlushResult {
                inner: flush.into_adamw_result(),
                report: None,
            });
        }
        let result = flush.into_adamw_result();
        let transition = self
            .partial_flush
            .as_mut()
            .ok_or_else(|| training("compiled Metal partial flush transition is absent"))?;
        let inputs = BTreeMap::from([(LEARNING_RATE_INPUT.to_owned(), learning_rate)]);
        let run = transition
            .run(&mut self.inner.session, &inputs)
            .map_err(metal_training_error)?;
        let (outputs, report) = run.into_parts();
        debug_assert!(outputs.is_empty());
        self.progress = next;
        Ok(MetalCompiledAdamWFlushResult {
            inner: result,
            report: Some(report),
        })
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    /// Preparation evidence for the state-only transition. Imported recurrent
    /// state must contribute zero initialization uploads.
    #[cfg(test)]
    pub(crate) fn flush_preparation_report(
        &self,
    ) -> Option<&crate::runtime::metal::MetalDevicePreparationReport> {
        self.partial_flush
            .as_ref()
            .map(MetalFixedStateTransitionSession::preparation_report)
    }

    /// Downloads one coherent active state bank and encodes the same portable
    /// checkpoint format accepted by [`CpuCompiledAdamW::compile_from_checkpoint`].
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        let states = self.state_snapshots()?;
        let topology = CompiledTrainingWindowTopology::from_validated_parts(
            self.contract.gradient_accumulation_steps,
            false,
            false,
        );
        let optimizer_step = states
            .get(&adamw_global_key(AdamWGlobalState::Step))
            .ok_or_else(|| training("compiled Metal optimizer step is absent"))?
            .scalar_at(0)
            .as_u64();
        let accumulation_index = if !topology.accumulating() {
            0
        } else {
            states
                .get(&adamw_global_key(AdamWGlobalState::AccumulationIndex))
                .ok_or_else(|| training("compiled Metal accumulation index is absent"))?
                .scalar_at(0)
                .as_u64()
        };
        validate_adamw_progress(self.progress, self.contract.gradient_accumulation_steps)?;
        if optimizer_step != self.progress.optimizer_step
            || accumulation_index != self.progress.accumulation_index
        {
            return Err(training("compiled Metal AdamW progress state mismatch"));
        }
        let dropout_block_counter = self
            .contract
            .dropout
            .map(|dropout| {
                let counter = states
                    .get(&RecurrentStateKey::dropout_counter())
                    .ok_or_else(|| training("compiled Metal dropout counter is absent"))?
                    .scalar_at(0)
                    .as_u64();
                if counter != expected_dropout_counter(dropout, self.progress.replay_step)? {
                    return Err(training(
                        "compiled Metal dropout counter and replay progress diverged",
                    ));
                }
                Ok(counter)
            })
            .transpose()?;
        let parameters = metal_parameter_snapshots(&states);
        let first_moments =
            metal_adamw_state_snapshots(states.clone(), AdamWParameterState::FirstMoment)?;
        let second_moments =
            metal_adamw_state_snapshots(states.clone(), AdamWParameterState::SecondMoment)?;
        let gradient_accumulators =
            metal_adamw_state_snapshots(states, AdamWParameterState::GradientAccumulator)?;
        CompiledAdamWCheckpoint::from_bytes(encode_adamw_checkpoint(
            AdamWCheckpointProgress {
                capture_identity: self.inner.program_identity,
                accumulation_capture_identity: self.accumulation_capture_identity,
                replay_step: self.progress.replay_step,
                optimizer_step: self.progress.optimizer_step,
                accumulation_steps: self.contract.gradient_accumulation_steps,
                accumulation_index: self.progress.accumulation_index,
                discarded_microbatches: self.progress.discarded_microbatches,
                flushed_window_count: self.progress.flushed_window_count,
                flushed_microbatch_count: self.progress.flushed_microbatch_count,
                flush_capture_identity: self.flush_capture_identity,
                dropout_block_counter,
                accumulated_token_count: None,
                window_loss_report: false,
                reset_transition_count: 0,
                reset_capture_identity: None,
            },
            AdamWCheckpointTensors {
                parameters,
                first_moments,
                second_moments,
                gradient_accumulators,
                accumulated_loss_numerator: None,
            },
        )?)
    }
}

impl CompiledTrainingRuntime for MetalCompiledAdamW {
    type Step = MetalCompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        MetalCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        MetalCompiledAdamW::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        MetalCompiledAdamW::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::parameter_snapshots(self)
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.parameter_snapshots()?,
            &self.contract.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for MetalCompiledAdamW {
    type Evaluation = MetalCompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        MetalCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::evaluation_capture_identity(self)
    }
}

impl CompiledCheckpointRuntime for MetalCompiledAdamW {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        MetalCompiledAdamW::checkpoint(self)
    }
}

impl CompiledAdamWRuntime for MetalCompiledAdamW {
    fn gradient_accumulation_steps(&self) -> u64 {
        MetalCompiledAdamW::gradient_accumulation_steps(self)
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        MetalCompiledAdamW::max_gradient_norm(self)
    }

    fn loss_scale(&self) -> f32 {
        MetalCompiledAdamW::loss_scale(self)
    }

    fn optimizer_step(&self) -> Result<u64> {
        MetalCompiledAdamW::optimizer_step(self)
    }

    fn accumulation_index(&self) -> Result<u64> {
        MetalCompiledAdamW::accumulation_index(self)
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        MetalCompiledAdamW::zero_grad(self)
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::first_moment_snapshots(self)
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::second_moment_snapshots(self)
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::gradient_accumulator_snapshots(self)
    }
}

impl CompiledTrainingWindowResetRuntime for MetalCompiledAdamW {
    fn reset_gradient_window(&mut self) -> Result<CompiledTrainingWindowReset> {
        MetalCompiledAdamW::zero_grad(self)
    }
}

impl CompiledTrainingWindowRuntime for MetalCompiledAdamW {
    fn gradient_window_size(&self) -> u64 {
        MetalCompiledAdamW::gradient_accumulation_steps(self)
    }

    fn pending_microbatch_count(&self) -> Result<u64> {
        MetalCompiledAdamW::accumulation_index(self)
    }
}

impl CompiledTrainingWindowCommitRuntime for MetalCompiledAdamW {
    type WindowCommit = MetalCompiledAdamWFlushResult;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        MetalCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::flush_capture_identity(self)
    }
}

impl CompiledAdamWFlushRuntime for MetalCompiledAdamW {
    type Flush = MetalCompiledAdamWFlushResult;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        CompiledTrainingWindowCommitRuntime::commit_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::flush_capture_identity(self)
    }
}

fn metal_parameter_snapshots(
    states: &BTreeMap<RecurrentStateKey, TensorData>,
) -> BTreeMap<String, TensorData> {
    states
        .iter()
        .filter_map(|(key, value)| {
            key.parameter_name()
                .map(|name| (name.to_owned(), value.clone()))
        })
        .collect()
}

fn metal_adamw_state_snapshots(
    states: BTreeMap<RecurrentStateKey, TensorData>,
    state: AdamWParameterState,
) -> Result<BTreeMap<String, TensorData>> {
    Ok(states
        .into_iter()
        .filter_map(|(key, value)| {
            parameter_for_adamw_state(&key, state).map(|name| (name.to_owned(), value))
        })
        .collect())
}
