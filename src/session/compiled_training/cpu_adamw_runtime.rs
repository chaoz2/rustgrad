//! Interpreter and strict-native CPU AdamW execution.

use super::*;

pub(super) struct PreparedNativeCpuProgram {
    pub(super) report: NativeCpuProgramPreparationReport,
    pub(super) replay: PreparedRecurrentNativeReplay,
}

pub(super) struct PreparedNativeCpuEvaluation {
    pub(super) report: NativeCpuProgramPreparationReport,
    pub(super) plan: PlannedNativeItems,
    pub(super) parameter_inputs: Vec<PreparedNativeEvaluationParameterInput>,
}

pub(super) struct NativeCpuEvaluationPreparation {
    pub(super) inputs: Option<BTreeMap<String, TensorData>>,
    pub(super) parameter_inputs: Vec<PreparedNativeEvaluationParameterInput>,
    pub(super) residual_wall_time: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreparedNativeEvaluationParameterInput {
    pub(super) parameter: String,
    pub(super) input: String,
    pub(super) buffer: u64,
    pub(super) shape: Shape,
    pub(super) dtype: DType,
    pub(super) bytes: usize,
}

impl PreparedNativeCpuEvaluation {
    pub(super) fn validate(
        &self,
        capture_identity: u64,
        capture: &CapturedSchedule,
        parameter_buffers: &BTreeMap<String, u64>,
    ) -> Result<()> {
        self.plan
            .validate_structure(capture)
            .map_err(replay_error)?;
        let native_identity = native_cpu_identity(
            capture_identity,
            self.plan.vectorized(),
            self.plan.schedule_cache_keys().iter().copied(),
        );
        if self.report.capture_identity != capture_identity
            || self.report.native_identity != native_identity
            || self.report.native_item_count != self.plan.item_count()
            || self.report.cache_hit_count != self.plan.cache_hit_count()
            || self.report.cache_miss_count != self.plan.cache_miss_count()
            || self.report.work
                != NativeCpuPreparationWork::from_module(self.plan.module_preparation())
            || self.report.vectorized != self.plan.vectorized()
            || capture.items.iter().map(|item| item.cache_key).ne(self
                .plan
                .schedule_cache_keys()
                .iter()
                .copied())
        {
            return Err(training(
                "compiled native CPU evaluation preparation identity mismatch",
            ));
        }
        self.report.validate_work()?;
        if parameter_buffers.len() != self.parameter_inputs.len() {
            return Err(training(
                "compiled native CPU evaluation parameter mapping mismatch",
            ));
        }
        let mut parameters = BTreeSet::new();
        let mut inputs = BTreeSet::new();
        let mut buffers = BTreeSet::new();
        for binding in &self.parameter_inputs {
            let input = capture
                .inputs
                .iter()
                .find(|input| input.name == binding.input)
                .ok_or_else(|| {
                    training("compiled native CPU evaluation parameter input is absent")
                })?;
            if !parameters.insert(binding.parameter.as_str())
                || parameter_buffers.get(&binding.parameter) != Some(&binding.buffer)
                || !inputs.insert(binding.input.as_str())
                || !buffers.insert(binding.buffer)
                || input.desc.shape != binding.shape
                || input.desc.dtype != binding.dtype
                || input.desc.bytes != binding.bytes
            {
                return Err(training(
                    "compiled native CPU evaluation parameter mapping mismatch",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn active_parameter_states(
        &self,
        frontier: &[BufferState],
    ) -> Result<Vec<BufferState>> {
        let mut frontier_by_buffer = BTreeMap::new();
        for state in frontier {
            if frontier_by_buffer.insert(state.buffer, state).is_some() {
                return Err(training(
                    "compiled native CPU recurrent frontier contains duplicate buffers",
                ));
            }
        }
        self.parameter_inputs
            .iter()
            .map(|binding| {
                let state = frontier_by_buffer.get(&binding.buffer).ok_or_else(|| {
                    training("compiled native CPU evaluation parameter state is absent")
                })?;
                if state.shape != binding.shape
                    || state.dtype != binding.dtype
                    || state.bytes != binding.bytes
                {
                    return Err(training(
                        "compiled native CPU evaluation parameter state descriptor mismatch",
                    ));
                }
                Ok((*state).clone())
            })
            .collect()
    }
}

/// One compiled AdamW training program with recurrent first/second moments, a
/// graph-owned step counter, and capture-authenticated gradient policies.
pub struct CpuCompiledAdamW {
    pub(super) inner: CpuCompiledTrainingProgram,
    pub(super) partial_flush: Option<CompiledAdamWAuxiliaryPlan>,
    pub(super) zero_grad: Option<CompiledAdamWAuxiliaryPlan>,
    pub(super) contract: CompiledAdamWContract,
    pub(super) progress: CompiledTrainingWindowProgress,
    pub(super) evaluation: Option<CpuCompiledEvaluation>,
    pub(super) non_finite_policy: CpuNonFinitePolicy,
}

/// Strict-native CPU AdamW session prepared from the same authenticated plan
/// as [`CpuCompiledAdamW`]. Optimizer, progress, checkpoint, accumulation,
/// dropout, and evaluation ownership remain in the shared CPU core; only pure
/// schedule execution is replaced with strict native JIT replay.
pub struct NativeCpuCompiledAdamW<'a> {
    pub(super) inner: CpuCompiledAdamW,
    pub(super) executor: &'a CapturedReplayExecutor,
    pub(super) main_replay: PreparedRecurrentNativeReplay,
    pub(super) accumulation_replay: Option<PreparedRecurrentNativeReplay>,
    pub(super) partial_flush_replay: Option<PreparedRecurrentNativeReplay>,
    pub(super) zero_grad_replay: Option<PreparedRecurrentNativeReplay>,
    pub(super) evaluation_replay: Option<PreparedNativeCpuEvaluation>,
    pub(super) preparation: NativeCpuCompiledAdamWPreparationReport,
    pub(super) successful_steps: u64,
    pub(super) successful_flushes: u64,
    pub(super) successful_zero_grads: u64,
    pub(super) successful_evaluations: u64,
}

pub(super) struct PendingAdamWStep {
    pub(super) request: CompiledStepReplayRequest,
    pub(super) next_progress: CompiledTrainingWindowProgress,
    pub(super) loss_weight: u64,
}

impl CpuCompiledAdamW {
    pub(super) fn admit_step(
        &self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<PendingAdamWStep> {
        validate_training_inputs(&self.inner.inputs, &inputs)?;
        let loss_weight = validate_token_weight(
            &inputs,
            self.contract.token_weight_policy.as_ref(),
            self.contract.allow_zero_valid_token_microbatches,
        )?;
        let next_progress = adamw_window_progress(
            self.progress
                .advance_replay(self.contract.gradient_accumulation_steps),
        )?;
        self.validate_completed_token_window(next_progress, loss_weight)?;
        if let Some(dropout) = self.contract.dropout {
            expected_dropout_counter(dropout, next_progress.replay_step)?;
        }
        Ok(PendingAdamWStep {
            request: CompiledStepReplayRequest {
                inputs,
                learning_rate,
                non_finite_policy: self.non_finite_policy,
                output_selection: selection,
                injected_failure,
            },
            next_progress,
            loss_weight,
        })
    }

    fn publish_step(
        &mut self,
        next_progress: CompiledTrainingWindowProgress,
        loss_weight: u64,
        mut result: CompiledTrainingStepResult,
    ) -> CompiledAdamWStepResult {
        result.step = next_progress.replay_step;
        self.progress = next_progress;
        adamw_step_result(
            result,
            next_progress,
            loss_weight,
            self.contract.gradient_accumulation_steps,
            self.contract.clip_report,
            self.contract.window_loss_report,
        )
    }

    fn validate_completed_token_window(
        &self,
        next: CompiledTrainingWindowProgress,
        loss_weight: u64,
    ) -> Result<()> {
        if !self.contract.allow_zero_valid_token_microbatches
            || self.contract.token_weight_policy.is_none()
            || next.accumulation_index != 0
        {
            return Ok(());
        }
        let topology = CompiledTrainingWindowTopology::from_contract(&self.contract);
        let retained = if topology.retains_token_count() {
            self.inner
                .global_snapshot(AdamWGlobalState::AccumulatedTokenCount)?
                .scalar_at(0)
                .as_u64()
        } else {
            0
        };
        let total = retained
            .checked_add(loss_weight)
            .ok_or_else(|| training("compiled AdamW completed token count overflows"))?;
        if total == 0 {
            return Err(training(
                "compiled AdamW completed token window must contain at least one valid token",
            ));
        }
        Ok(())
    }

    fn validate_partial_token_window(&self) -> Result<()> {
        if !self.contract.allow_zero_valid_token_microbatches
            || self.contract.token_weight_policy.is_none()
        {
            return Ok(());
        }
        let retained = self
            .inner
            .global_snapshot(AdamWGlobalState::AccumulatedTokenCount)?
            .scalar_at(0)
            .as_u64();
        if retained == 0 {
            return Err(training(
                "compiled AdamW partial token window must contain at least one valid token",
            ));
        }
        Ok(())
    }

    pub fn compile<F>(
        config: CompiledAdamWConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile(config, parameters, build)?.prepare_cpu()
    }

    /// Compiles an ordinary module forward against optimizer-owned recurrent
    /// parameter state.
    ///
    /// The builder receives only declared batch inputs: calls to
    /// [`crate::nn::Parameter::bind`] inside `module` resolve automatically to
    /// the compiled state frontier. Frozen parameters and buffers are captured
    /// as immutable constants, while tied parameter handles share one graph
    /// node and one AdamW state tuple. Names selected by
    /// [`CompiledAdamWConfig::with_frozen_parameters`] receive that same
    /// constant treatment without changing their host trainable flags.
    pub fn compile_module<M, F>(config: CompiledAdamWConfig, module: &M, build: F) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile_module(config, module, build)?.prepare_cpu()
    }

    /// Recompiles an exact program and restores its saved recurrent frontier.
    /// The build/configuration must reproduce the checkpoint's capture
    /// identity; all state is validated before the fresh runtime is replaced.
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
        CompiledAdamWPlan::compile_from_checkpoint(config, checkpoint, build)?.prepare_cpu()
    }

    /// Recompiles a module-bound program and restores its exact AdamW state.
    /// The module topology, frozen values, builder, and input descriptors must
    /// reproduce the authenticated capture identity before any restored state
    /// becomes visible.
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
        CompiledAdamWPlan::compile_module_from_checkpoint(config, module, checkpoint, build)?
            .prepare_cpu()
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            None,
        )
    }

    /// Commits one external-rate replay while omitting graph-named outputs.
    ///
    /// Loss and enabled clip/window reports remain captured, validated, and
    /// returned. Only the user-named output range is excluded from CPU egress;
    /// recurrent state and checkpoint identity are unchanged.
    pub fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            None,
        )
    }

    /// Replays one batch using the immutable MultiStep rate captured in the
    /// program. This method accepts no host learning-rate value.
    pub fn step_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::All, None)
    }

    /// Scheduled-rate counterpart of [`Self::step_commit_only`].
    pub fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::CommitOnly, None)
    }

    fn step_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        let PendingAdamWStep {
            request,
            next_progress,
            loss_weight,
        } = self.admit_step(inputs, learning_rate, selection, injected_failure)?;
        let result = if next_progress.accumulation_index == 0 {
            self.inner.step_inner_with_learning_rate(request, true)?
        } else {
            let transition = self
                .inner
                .accumulation
                .clone()
                .ok_or_else(|| training("compiled accumulation replay is absent"))?;
            self.inner
                .step_accumulation_inner_with_learning_rate(&transition, request)?
        };
        Ok(self.publish_step(next_progress, loss_weight, result))
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<CompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    pub fn step_batch_commit_only<B>(
        &mut self,
        batch: B,
        learning_rate: f32,
    ) -> Result<CompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }

    pub fn step_batch_commit_only_scheduled<B>(
        &mut self,
        batch: B,
    ) -> Result<CompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only_scheduled(batch.into_compiled_inputs()?)
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledEvaluationResult> {
        let evaluation = self
            .evaluation
            .as_ref()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        evaluation
            .plan
            .evaluate(inputs, self.inner.parameter_snapshots()?)
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation
            .as_ref()
            .map(|evaluation| evaluation.plan.capture_identity)
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.contract.gradient_accumulation_steps
    }

    pub fn token_weighted_gradient_accumulation_mask(&self) -> Option<&str> {
        match &self.contract.token_weight_policy {
            Some(CompiledTokenWeightPolicy::ExplicitMask(name)) => Some(name),
            _ => None,
        }
    }

    /// I32 target input and sentinel used for compiler-owned token weighting.
    pub fn token_weighted_ignore_index(&self) -> Option<(&str, i32)> {
        match &self.contract.token_weight_policy {
            Some(CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            }) => Some((target_input, *value)),
            _ => None,
        }
    }

    pub fn zero_valid_token_microbatches_enabled(&self) -> bool {
        self.contract.allow_zero_valid_token_microbatches
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.contract.max_gradient_norm
    }

    pub fn clip_report_enabled(&self) -> bool {
        self.contract.clip_report
    }

    /// Whether completed-window loss aggregation is captured and reported.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.contract.window_loss_report
    }

    pub fn loss_scale(&self) -> f32 {
        self.contract.loss_scale
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        match &self.contract.learning_rate {
            CompiledLearningRatePolicy::External => None,
            CompiledLearningRatePolicy::MultiStep(schedule) => Some(schedule),
        }
    }

    /// CPU-only admission policy selected when this runtime was prepared.
    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.non_finite_policy
    }

    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.contract
            .dropout
            .map(|_| {
                Ok(self
                    .inner
                    .workload_snapshot(&RecurrentStateKey::dropout_counter())?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        Ok(self.progress.optimizer_step)
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::SecondMoment)
    }

    /// Partial F32 gradient sums retained between microbatches. The map is
    /// empty when accumulation is disabled (`steps == 1`).
    pub fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::GradientAccumulator)
    }

    /// Number of microbatches currently retained toward the next update.
    pub fn accumulation_index(&self) -> Result<u64> {
        Ok(self.progress.accumulation_index)
    }

    /// Atomically clears a retained partial accumulation window. Parameters,
    /// moments, optimizer progress, and successful replay count are preserved.
    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_inner(None)
    }

    fn zero_grad_inner(
        &mut self,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWZeroGradResult> {
        let (next, reset) = adamw_window_progress(
            self.progress
                .cancel(self.contract.gradient_accumulation_steps),
        )?;
        if !reset.did_discard() {
            return Ok(reset);
        }
        let next = adamw_window_progress(next.record_reset_transition())?;
        let transition = self
            .zero_grad
            .as_ref()
            .ok_or_else(|| training("compiled AdamW zero-grad capture is absent"))?;
        let reports = self.inner.replay_auxiliary_transition(
            transition,
            None,
            CpuNonFinitePolicy::Propagate,
            injected_failure,
        )?;
        debug_assert!(reports.clip_report.is_none());
        debug_assert!(reports.window_loss.is_none());
        self.progress = next;
        Ok(reset)
    }

    #[cfg(test)]
    pub(super) fn zero_grad_with_injected_failure(
        &mut self,
        injected_failure: u64,
    ) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_inner(Some(injected_failure))
    }

    /// Atomically commits a nonempty partial accumulation window through its
    /// separately authenticated state-only capture. Replay/dropout progress
    /// and all workload state remain unchanged.
    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWFlushResult> {
        self.contract.learning_rate.require_external()?;
        self.flush_partial_window_with_learning_rate(Some(learning_rate), None)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<CompiledAdamWFlushResult> {
        self.contract.learning_rate.require_scheduled()?;
        self.flush_partial_window_with_learning_rate(None, None)
    }

    fn flush_partial_window_with_learning_rate(
        &mut self,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWFlushResult> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let (next, flush) = adamw_window_progress(
            self.progress
                .flush_partial(self.contract.gradient_accumulation_steps),
        )?;
        if !flush.did_update() {
            return Ok(flush.into_adamw_result());
        }
        let mut result = flush.into_adamw_result();
        self.validate_partial_token_window()?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, self.non_finite_policy)?;
        }
        let transition = self
            .partial_flush
            .as_ref()
            .ok_or_else(|| training("compiled AdamW partial flush capture is absent"))?;
        let reports = self.inner.replay_auxiliary_transition(
            transition,
            learning_rate,
            self.non_finite_policy,
            injected_failure,
        )?;
        result.clip_report = reports.clip_report;
        result.window_loss_report = reports
            .window_loss
            .map(|value| CompiledAdamWWindowLossReport::new(value, result.flushed_microbatches));
        self.progress = next;
        Ok(result)
    }

    /// Stable identity of the state-only flush capture, when accumulation is
    /// enabled.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.partial_flush
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.zero_grad
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn first_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner
            .adamw_state_versions(AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner
            .adamw_state_versions(AdamWParameterState::SecondMoment)
    }

    pub(super) fn snapshot_plan(&self) -> Result<CompiledAdamWPlan> {
        let inner = self.inner.plan()?;
        let partial_flush = self
            .partial_flush
            .clone()
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        let zero_grad = self
            .zero_grad
            .clone()
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        Ok(CompiledAdamWPlan {
            inner,
            partial_flush,
            zero_grad,
            program_identity: self.capture_identity(),
            contract: self.contract.clone(),
            progress: self.progress,
            evaluation: self
                .evaluation
                .as_ref()
                .map(|evaluation| evaluation.plan.clone()),
            compile_phases: None,
        })
    }

    fn restored_candidate(&self, checkpoint: &CompiledAdamWCheckpoint) -> Result<Self> {
        self.snapshot_plan()?
            .restore_checkpoint(checkpoint)?
            .prepare_cpu_with_non_finite_policy(self.non_finite_policy)
    }

    /// Renders the identical loss/backward/AdamW capture for Metal, seeded
    /// from this session's currently committed recurrent state. Planning is
    /// resource-free; unsupported kernels fail before a device is touched.
    pub fn metal_plan(&self, renderer: MetalRenderer) -> Result<MetalCompiledAdamWPlan> {
        self.snapshot_plan()?.metal_plan(renderer)
    }

    /// Captures parameter values, both moment sets, the graph-owned optimizer
    /// step, and the exact compiled capture identity into deterministic bytes.
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        let topology = CompiledTrainingWindowTopology::from_contract(&self.contract);
        validate_adamw_progress(self.progress, self.contract.gradient_accumulation_steps)?;
        validate_cpu_adamw_state(
            &self.inner,
            self.progress,
            self.contract.gradient_accumulation_steps,
        )?;
        let dropout_block_counter = self
            .contract
            .dropout
            .map(|dropout| {
                let counter = self
                    .inner
                    .workload_snapshot(&RecurrentStateKey::dropout_counter())?
                    .scalar_at(0)
                    .as_u64();
                if counter != expected_dropout_counter(dropout, self.progress.replay_step)? {
                    return Err(training(
                        "compiled CPU dropout counter and replay progress diverged",
                    ));
                }
                Ok(counter)
            })
            .transpose()?;
        let accumulated_token_count = topology
            .retains_token_count()
            .then_some(())
            .map(|_| {
                Ok(self
                    .inner
                    .global_snapshot(AdamWGlobalState::AccumulatedTokenCount)?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()?;
        if let (Some(policy), Some(count)) = (
            self.contract.token_weight_policy.as_ref(),
            accumulated_token_count,
        ) {
            validate_retained_token_count(
                &self.inner.inputs,
                policy,
                self.progress.accumulation_index,
                count,
                self.contract.allow_zero_valid_token_microbatches,
            )?;
        }
        let accumulated_loss_numerator = topology
            .retains_window_numerator()
            .then(|| {
                self.inner
                    .global_snapshot(AdamWGlobalState::AccumulatedLossNumerator)
            })
            .transpose()?;
        let bytes = encode_adamw_checkpoint(
            AdamWCheckpointProgress {
                capture_identity: self.capture_identity(),
                accumulation_capture_identity: self
                    .inner
                    .accumulation
                    .as_ref()
                    .map(|transition| transition.phase().capture_identity),
                replay_step: self.progress.replay_step,
                optimizer_step: self.progress.optimizer_step,
                accumulation_steps: self.contract.gradient_accumulation_steps,
                accumulation_index: self.progress.accumulation_index,
                discarded_microbatches: self.progress.discarded_microbatches,
                flushed_window_count: self.progress.flushed_window_count,
                flushed_microbatch_count: self.progress.flushed_microbatch_count,
                flush_capture_identity: self.flush_capture_identity(),
                dropout_block_counter,
                accumulated_token_count,
                window_loss_report: self.contract.window_loss_report,
                reset_transition_count: self.progress.reset_transition_count,
                reset_capture_identity: (self.progress.reset_transition_count != 0)
                    .then(|| self.zero_grad_capture_identity())
                    .flatten(),
            },
            AdamWCheckpointTensors {
                parameters: self.parameter_snapshots()?,
                first_moments: self.first_moment_snapshots()?,
                second_moments: self.second_moment_snapshots()?,
                gradient_accumulators: self.gradient_accumulator_snapshots()?,
                accumulated_loss_numerator,
            },
        )?;
        CompiledAdamWCheckpoint::from_bytes(bytes)
    }

    #[cfg(test)]
    pub(super) fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            injected_failure,
        )
    }

    #[cfg(test)]
    pub(super) fn step_commit_only_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            injected_failure,
        )
    }

    #[cfg(test)]
    pub(super) fn flush_partial_window_inner(
        &mut self,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWFlushResult> {
        self.flush_partial_window_with_learning_rate(Some(learning_rate), injected_failure)
    }
}

impl<'a> NativeCpuCompiledAdamW<'a> {
    pub(super) fn prepare(
        inner: CpuCompiledAdamW,
        executor: &'a CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<Self> {
        let external_learning_rate = matches!(
            &inner.contract.learning_rate,
            CompiledLearningRatePolicy::External
        );
        let program_count = 1
            + usize::from(inner.inner.accumulation.is_some())
            + usize::from(inner.partial_flush.is_some())
            + usize::from(inner.zero_grad.is_some())
            + usize::from(inner.evaluation.is_some());
        let mut drafts = NativeCpuTrainingProgramDrafts::new();
        let (mut main_preparation, main_residual) = inner
            .inner
            .preflight_native(vectorized, external_learning_rate)?;
        {
            let (pure, inputs) = main_preparation.pure_and_inputs();
            let draft = executor
                .preflight_native_items_with_store_groups(
                    pure,
                    inputs,
                    &inner.inner.recurrent_store_groups,
                )
                .map_err(replay_error)?;
            drafts.insert(NativeCpuTrainingProgramRole::Main, draft)?;
        }
        main_preparation.release_input_witnesses();
        let accumulation_preparation = match inner.inner.accumulation.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) = inner
                    .inner
                    .preflight_native_accumulation(transition, vectorized)?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items_with_recurrent_retention(
                            pure,
                            inputs,
                            preparation.retained_recurrent_states(),
                        )
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::Accumulation, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let partial_flush_preparation = match inner.partial_flush.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) =
                    inner.inner.preflight_native_auxiliary_transition(
                        transition.phase(),
                        vectorized,
                        external_learning_rate,
                    )?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items_with_store_groups(
                            pure,
                            inputs,
                            transition.phase().store_groups(),
                        )
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::PartialFlush, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let zero_grad_preparation = match inner.zero_grad.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) = inner
                    .inner
                    .preflight_native_auxiliary_transition(transition.phase(), vectorized, false)?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items(pure, inputs)
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::ZeroGrad, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let evaluation_preparation = match inner.evaluation.as_ref() {
            Some(evaluation) => {
                let mut preparation = evaluation.plan.preflight_native(
                    inner.parameter_snapshots()?,
                    &inner.inner.parameter_buffers,
                )?;
                let draft = executor
                    .preflight_native_items(
                        evaluation.plan.inference.capture(),
                        preparation
                            .inputs
                            .as_ref()
                            .expect("native evaluation input witnesses are present"),
                    )
                    .map_err(replay_error)?;
                drafts.insert(NativeCpuTrainingProgramRole::Evaluation, draft)?;
                preparation.inputs = None;
                Some(preparation)
            }
            None => None,
        };
        let mut programs = NativeCpuTrainingProgramBatch::with_capacity(program_count);
        programs.push(
            NativeCpuTrainingProgramRole::Main,
            main_preparation.pure(),
            drafts.take(NativeCpuTrainingProgramRole::Main)?,
        )?;
        if let Some((preparation, _)) = &accumulation_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::Accumulation,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::Accumulation)?,
            )?;
        }
        if let Some((preparation, _)) = &partial_flush_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::PartialFlush,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::PartialFlush)?,
            )?;
        }
        if let Some((preparation, _)) = &zero_grad_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::ZeroGrad,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::ZeroGrad)?,
            )?;
        }
        if let Some(evaluation) = inner.evaluation.as_ref() {
            programs.push(
                NativeCpuTrainingProgramRole::Evaluation,
                evaluation.plan.inference.capture(),
                drafts.take(NativeCpuTrainingProgramRole::Evaluation)?,
            )?;
        }
        if !drafts.is_empty() {
            return Err(training(
                "compiled native CPU planning draft inventory is excessive",
            ));
        }
        let (roles, programs) = programs.into_planning_inputs();
        let (plans, compilation) = executor
            .plan_native_item_drafts(programs, vectorized)
            .map_err(replay_error)?;
        let render_capsule_diagnostics = NativeCpuRenderCapsuleDiagnostic::from_ordered_native(
            &roles,
            compilation.render_capsule_diagnostics,
        )?;
        let NativeCpuTrainingPrograms {
            main: main_plan,
            accumulation: accumulation_plan,
            partial_flush: partial_flush_plan,
            zero_grad: zero_grad_plan,
            evaluation: evaluation_plan,
        } = NativeCpuTrainingPrograms::from_ordered(roles, plans)?;
        let main = inner
            .inner
            .finish_native(main_preparation, main_plan, main_residual)?;
        let accumulation = match (
            inner.inner.accumulation.as_ref(),
            accumulation_preparation,
            accumulation_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => Some(
                inner
                    .inner
                    .finish_native_accumulation(transition, preparation, plan, residual)?,
            ),
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU accumulation preparation differs",
                ));
            }
        };
        let partial_flush = match (
            inner.partial_flush.as_ref(),
            partial_flush_preparation,
            partial_flush_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => {
                Some(inner.inner.finish_native_auxiliary_transition(
                    transition.phase(),
                    preparation,
                    plan,
                    residual,
                )?)
            }
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU partial-flush preparation differs",
                ));
            }
        };
        let zero_grad = match (
            inner.zero_grad.as_ref(),
            zero_grad_preparation,
            zero_grad_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => {
                Some(inner.inner.finish_native_auxiliary_transition(
                    transition.phase(),
                    preparation,
                    plan,
                    residual,
                )?)
            }
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU zero-grad preparation differs",
                ));
            }
        };
        let evaluation = match (
            inner.evaluation.as_ref(),
            evaluation_preparation,
            evaluation_plan,
        ) {
            (Some(evaluation), Some(preparation), Some(plan)) => Some(
                evaluation
                    .plan
                    .finish_native(preparation, &inner.inner.parameter_buffers, plan)?,
            ),
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU evaluation preparation differs",
                ));
            }
        };
        let (recurrent_state_count, recurrent_state_bytes) = checked_recurrent_state_extent(
            inner
                .inner
                .cursor
                .frontier()
                .iter()
                .map(|state| state.bytes),
        )?;
        let PreparedNativeCpuProgram {
            report: main_report,
            replay: main_replay,
        } = main;
        let (accumulation_report, accumulation_replay) = accumulation
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let (partial_flush_report, partial_flush_replay) = partial_flush
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let (zero_grad_report, zero_grad_replay) = zero_grad
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let evaluation_report = evaluation.as_ref().map(|prepared| prepared.report.clone());
        Ok(Self {
            inner,
            executor,
            main_replay,
            accumulation_replay,
            partial_flush_replay,
            zero_grad_replay,
            evaluation_replay: evaluation,
            preparation: NativeCpuCompiledAdamWPreparationReport {
                main: main_report,
                accumulation: accumulation_report,
                partial_flush: partial_flush_report,
                zero_grad: zero_grad_report,
                evaluation: evaluation_report,
                recurrent_state_count,
                recurrent_state_bytes,
                render_capsule_hit_count: compilation.render_capsule_hit_count,
                render_capsule_miss_count: compilation.render_capsule_miss_count,
                local_render_job_count: compilation.local_render_job_count,
                parallel_render_overlap_wall_time: compilation.parallel_render_overlap_wall_time,
                max_parallel_render_job_count: compilation.max_parallel_render_job_count,
                parallel_module_overlap_wall_time: compilation.parallel_work_overlap_wall_time,
                compiler_process_overlap_wall_time: compilation.compiler_process_overlap_wall_time,
                compiler_process_count: compilation.compiler_process_count,
                max_parallel_compiler_process_count: compilation
                    .max_parallel_compiler_process_count,
                compiler_process_timings: compilation
                    .compiler_process_timings
                    .into_iter()
                    .map(NativeCpuCompilerProcessTiming::from_native)
                    .collect(),
                module_overlaps: compilation
                    .module_overlaps
                    .into_iter()
                    .map(NativeCpuModuleOverlap::from_native)
                    .collect(),
                program_pair_overlaps: compilation
                    .program_pair_overlaps
                    .into_iter()
                    .map(NativeCpuProgramPairOverlap::from_native)
                    .collect(),
                translation_units: compilation
                    .translation_units
                    .into_iter()
                    .map(NativeCpuTranslationUnitEvidence::from_native)
                    .collect(),
                render_capsule_diagnostics,
            },
            successful_steps: 0,
            successful_flushes: 0,
            successful_zero_grads: 0,
            successful_evaluations: 0,
        })
    }

    pub fn preparation_report(&self) -> &NativeCpuCompiledAdamWPreparationReport {
        &self.preparation
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            None,
        )
    }

    /// Strict-native CPU replay that omits only graph-named output egress.
    /// Loss and enabled report scalars retain their ordinary validation and
    /// result semantics.
    pub fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            None,
        )
    }

    pub fn step_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::All, None)
    }

    pub fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::CommitOnly, None)
    }

    pub(super) fn step_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        let PendingAdamWStep {
            request,
            next_progress,
            loss_weight,
        } = self
            .inner
            .admit_step(inputs, learning_rate, selection, injected_failure)?;
        let successful_invocation = self
            .successful_steps
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU run count overflow"))?;
        let (result, mut report) = if next_progress.accumulation_index == 0 {
            self.inner.inner.step_native_inner_with_learning_rate(
                request,
                true,
                NativeReplayContext::new(self.executor, &mut self.main_replay),
            )?
        } else {
            let transition = self
                .inner
                .inner
                .accumulation
                .clone()
                .ok_or_else(|| training("compiled accumulation replay is absent"))?;
            let replay = self
                .accumulation_replay
                .as_mut()
                .ok_or_else(|| training("compiled native CPU accumulation replay is absent"))?;
            self.inner
                .inner
                .step_accumulation_native_inner_with_learning_rate(
                    &transition,
                    request,
                    NativeReplayContext::new(self.executor, replay),
                )?
        };
        report.successful_invocation = successful_invocation;
        let inner = self.inner.publish_step(next_progress, loss_weight, result);
        self.successful_steps = successful_invocation;
        Ok(NativeCpuCompiledAdamWStepResult { inner, report })
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<NativeCpuCompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    pub fn step_batch_commit_only<B>(
        &mut self,
        batch: B,
        learning_rate: f32,
    ) -> Result<NativeCpuCompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }

    pub fn step_batch_commit_only_scheduled<B>(
        &mut self,
        batch: B,
    ) -> Result<NativeCpuCompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only_scheduled(batch.into_compiled_inputs()?)
    }

    #[cfg(test)]
    pub(super) fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            injected_failure,
        )
    }

    #[cfg(test)]
    pub(super) fn step_commit_only_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            injected_failure,
        )
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledEvaluationResult> {
        let NativeCpuCompiledAdamW {
            inner,
            executor,
            evaluation_replay,
            successful_evaluations,
            ..
        } = self;
        let successful_invocation = (*successful_evaluations)
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU evaluation count overflow"))?;
        let executor = *executor;
        let evaluation = inner
            .evaluation
            .as_ref()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        let prepared = evaluation_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU evaluation preparation is absent"))?;
        let program = &mut inner.inner;
        let (loss_weight, parameter_states) = evaluation.plan.preflight_native_borrowed(
            &inputs,
            program.cursor.frontier(),
            prepared,
            &program.parameter_buffers,
        )?;
        let evaluated = program
            .runtime
            .with_active_state_tensors(&parameter_states, |reads| {
                evaluation.plan.evaluate_native_borrowed(
                    &inputs,
                    reads,
                    loss_weight,
                    executor,
                    prepared,
                )
            });
        let (inner, mut report) = match evaluated {
            Ok(evaluated) => evaluated,
            Err(RecurrentTransactionError::Runtime(error)) => return Err(runtime_error(error)),
            Err(RecurrentTransactionError::Stage(error)) => return Err(error),
            Err(RecurrentTransactionError::Contract(reason)) => return Err(training(reason)),
        };
        report.successful_invocation = successful_invocation;
        *successful_evaluations = successful_invocation;
        Ok(NativeCpuCompiledEvaluationResult { inner, report })
    }

    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.inner.contract.learning_rate.require_external()?;
        self.flush_partial_window_impl(Some(learning_rate), None)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.inner.contract.learning_rate.require_scheduled()?;
        self.flush_partial_window_impl(None, None)
    }

    fn flush_partial_window_impl(
        &mut self,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let (next, flush) = adamw_window_progress(
            self.inner
                .progress
                .flush_partial(self.inner.contract.gradient_accumulation_steps),
        )?;
        if !flush.did_update() {
            return Ok(NativeCpuCompiledAdamWFlushResult {
                inner: flush.into_adamw_result(),
                report: None,
            });
        }
        let mut result = flush.into_adamw_result();
        self.inner.validate_partial_token_window()?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, self.inner.non_finite_policy)?;
        }
        let successful_invocation = self
            .successful_flushes
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU flush count overflow"))?;
        let transition = self
            .inner
            .partial_flush
            .as_ref()
            .ok_or_else(|| training("compiled AdamW partial flush capture is absent"))?;
        let prepared = self
            .partial_flush_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU partial flush preparation is absent"))?;
        let (reports, report) = self.inner.inner.replay_auxiliary_transition_native(
            transition,
            learning_rate,
            self.inner.non_finite_policy,
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            injected_failure,
        )?;
        result.clip_report = reports.clip_report;
        result.window_loss_report = reports
            .window_loss
            .map(|value| CompiledAdamWWindowLossReport::new(value, result.flushed_microbatches));
        self.inner.progress = next;
        self.successful_flushes = successful_invocation;
        Ok(NativeCpuCompiledAdamWFlushResult {
            inner: result,
            report: Some(report),
        })
    }

    #[cfg(test)]
    pub(super) fn flush_partial_window_with_injected_failure(
        &mut self,
        learning_rate: TensorData,
        injected_failure: u64,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.flush_partial_window_impl(Some(learning_rate), Some(injected_failure))
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }

    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.inner.zero_grad_capture_identity()
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.inner.captured_multi_step_lr()
    }

    /// CPU-only admission policy selected when this runtime was prepared.
    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.inner.non_finite_policy
    }

    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.inner.dropout_block_counter()
    }

    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        self.inner.checkpoint()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_impl(None)
    }

    fn zero_grad_impl(
        &mut self,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWZeroGradResult> {
        let (next, reset) = adamw_window_progress(
            self.inner
                .progress
                .cancel(self.inner.contract.gradient_accumulation_steps),
        )?;
        if !reset.did_discard() {
            return Ok(reset);
        }
        let next = adamw_window_progress(next.record_reset_transition())?;
        let successful_invocation = self
            .successful_zero_grads
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU zero-grad count overflow"))?;
        let transition = self
            .inner
            .zero_grad
            .as_ref()
            .ok_or_else(|| training("compiled AdamW zero-grad capture is absent"))?;
        let prepared = self
            .zero_grad_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU zero-grad preparation is absent"))?;
        let (reports, _) = self.inner.inner.replay_auxiliary_transition_native(
            transition,
            None,
            CpuNonFinitePolicy::Propagate,
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            injected_failure,
        )?;
        debug_assert!(reports.clip_report.is_none());
        debug_assert!(reports.window_loss.is_none());
        self.inner.progress = next;
        self.successful_zero_grads = successful_invocation;
        Ok(reset)
    }

    #[cfg(test)]
    pub(super) fn zero_grad_with_injected_failure(
        &mut self,
        injected_failure: u64,
    ) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_impl(Some(injected_failure))
    }
}

impl CompiledTrainingRuntime for CpuCompiledAdamW {
    type Step = CompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        CpuCompiledAdamW::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        CpuCompiledAdamW::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::parameter_snapshots(self)
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.parameter_snapshots()?,
            &self.contract.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for CpuCompiledAdamW {
    type Evaluation = CompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        CpuCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::evaluation_capture_identity(self)
    }
}

impl CompiledCheckpointRuntime for CpuCompiledAdamW {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        CpuCompiledAdamW::checkpoint(self)
    }
}

impl CompiledCheckpointRestoreRuntime for CpuCompiledAdamW {
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        let restored = self.restored_candidate(checkpoint)?;
        *self = restored;
        Ok(())
    }
}

impl CompiledAdamWRuntime for CpuCompiledAdamW {
    fn gradient_accumulation_steps(&self) -> u64 {
        CpuCompiledAdamW::gradient_accumulation_steps(self)
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        CpuCompiledAdamW::max_gradient_norm(self)
    }

    fn loss_scale(&self) -> f32 {
        CpuCompiledAdamW::loss_scale(self)
    }

    fn window_loss_report_enabled(&self) -> bool {
        CpuCompiledAdamW::window_loss_report_enabled(self)
    }

    fn optimizer_step(&self) -> Result<u64> {
        CpuCompiledAdamW::optimizer_step(self)
    }

    fn accumulation_index(&self) -> Result<u64> {
        CpuCompiledAdamW::accumulation_index(self)
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        CpuCompiledAdamW::zero_grad(self)
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::zero_grad_capture_identity(self)
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::first_moment_snapshots(self)
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::second_moment_snapshots(self)
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::gradient_accumulator_snapshots(self)
    }
}

impl CompiledAdamWCommitOnlyRuntime for CpuCompiledAdamW {
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingCommitOnlyRuntime for CpuCompiledAdamW {
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingWindowCommitRuntime for CpuCompiledAdamW {
    type WindowCommit = CompiledAdamWFlushResult;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        CpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::flush_capture_identity(self)
    }
}

impl CompiledAdamWFlushRuntime for CpuCompiledAdamW {
    type Flush = CompiledAdamWFlushResult;

    fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWFlushResult> {
        CompiledTrainingWindowCommitRuntime::commit_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::flush_capture_identity(self)
    }
}

impl CompiledTrainingRatePolicyWindowCommitRuntime for CpuCompiledAdamW {
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        CpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWRuntime for CpuCompiledAdamW {
    type ScheduledFlush = CompiledAdamWFlushResult;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        CpuCompiledAdamW::captured_multi_step_lr(self)
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        CpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        CpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWCommitOnlyRuntime for CpuCompiledAdamW {
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only_scheduled(self, inputs)
    }
}

impl CompiledTrainingRuntime for NativeCpuCompiledAdamW<'_> {
    type Step = NativeCpuCompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.inner.parameter_snapshots()?,
            &self.inner.contract.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for NativeCpuCompiledAdamW<'_> {
    type Evaluation = NativeCpuCompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        NativeCpuCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }
}

impl CompiledCheckpointRuntime for NativeCpuCompiledAdamW<'_> {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.inner.checkpoint()
    }
}

impl CompiledCheckpointRestoreRuntime for NativeCpuCompiledAdamW<'_> {
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        let restored = self.inner.restored_candidate(checkpoint)?;
        self.inner = restored;
        Ok(())
    }
}

impl CompiledAdamWRuntime for NativeCpuCompiledAdamW<'_> {
    fn gradient_accumulation_steps(&self) -> u64 {
        self.inner.gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.inner.max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.inner.loss_scale()
    }

    fn window_loss_report_enabled(&self) -> bool {
        self.inner.window_loss_report_enabled()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.inner.optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.inner.accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        NativeCpuCompiledAdamW::zero_grad(self)
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.inner.zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.gradient_accumulator_snapshots()
    }
}

impl CompiledAdamWCommitOnlyRuntime for NativeCpuCompiledAdamW<'_> {
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingCommitOnlyRuntime for NativeCpuCompiledAdamW<'_> {
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingWindowCommitRuntime for NativeCpuCompiledAdamW<'_> {
    type WindowCommit = NativeCpuCompiledAdamWFlushResult;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        NativeCpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }
}

impl CompiledAdamWFlushRuntime for NativeCpuCompiledAdamW<'_> {
    type Flush = NativeCpuCompiledAdamWFlushResult;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        CompiledTrainingWindowCommitRuntime::commit_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }
}

impl CompiledTrainingRatePolicyWindowCommitRuntime for NativeCpuCompiledAdamW<'_> {
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        NativeCpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWRuntime for NativeCpuCompiledAdamW<'_> {
    type ScheduledFlush = NativeCpuCompiledAdamWFlushResult;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.inner.captured_multi_step_lr()
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        NativeCpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWCommitOnlyRuntime for NativeCpuCompiledAdamW<'_> {
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only_scheduled(self, inputs)
    }
}

pub(super) fn adamw_step_result(
    mut inner: CompiledTrainingStepResult,
    progress: CompiledTrainingWindowProgress,
    loss_weight: u64,
    gradient_accumulation_steps: u64,
    clip_report_enabled: bool,
    window_loss_report_enabled: bool,
) -> CompiledAdamWStepResult {
    inner.loss_aggregation_weight = loss_weight;
    let expected = if progress.accumulation_index == 0 {
        adamw_observation_schema(clip_report_enabled, window_loss_report_enabled)
    } else {
        CompiledTrainingObservationSchema::default()
    };
    debug_assert_eq!(inner.observations.len(), expected.len());
    debug_assert!(
        inner
            .observations
            .iter()
            .map(|observation| observation.key)
            .eq(expected.entries.iter().map(|spec| spec.key))
    );
    for (observation, spec) in inner.observations.iter().zip(&expected.entries) {
        debug_assert!(
            validate_observation_value_descriptor(&observation.value, spec.constraint).is_ok()
        );
    }
    let mut observations = std::mem::take(&mut inner.observations)
        .into_iter()
        .map(|observation| observation.value);
    let clip_report = take_compiled_clip_report(
        &mut observations,
        clip_report_enabled && progress.accumulation_index == 0,
    );
    let window_loss = take_compiled_window_loss_value(
        &mut observations,
        window_loss_report_enabled && progress.accumulation_index == 0,
    );
    debug_assert!(observations.next().is_none());
    let window_loss_report = window_loss
        .map(|value| CompiledAdamWWindowLossReport::new(value, gradient_accumulation_steps));
    CompiledAdamWStepResult {
        inner,
        optimizer_step: progress.optimizer_step,
        accumulation_index: progress.accumulation_index,
        clip_report,
        window_loss_report,
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for CpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu()
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for ConfiguredCpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu_with_non_finite_policy(self.non_finite_policy())
    }
}

impl<'executor> SessionTarget<&CompiledAdamWPlan> for NativeCpuSessionTarget<'executor> {
    type Session = NativeCpuCompiledAdamW<'executor>;
    type Error = Error;

    fn prepare(&self, plan: &CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_native_cpu(self)
    }
}
