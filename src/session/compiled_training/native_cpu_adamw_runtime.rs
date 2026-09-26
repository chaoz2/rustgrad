//! Strict-native CPU AdamW replay over the shared interpreter-owned state.

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

impl<'a> NativeCpuCompiledAdamW<'a> {
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
        let output_schema = transition.outputs.clone();
        let non_finite_policy = self.inner.non_finite_policy;
        let (reports, report) = self.inner.inner.replay_recurrent_phase_native(
            transition.phase(),
            RecurrentPhaseReplayRequest {
                learning_rate,
                non_finite_policy,
                injected_failure,
            },
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            move |outputs| output_schema.validate_and_decode(outputs, non_finite_policy),
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
        let output_schema = transition.outputs.clone();
        let (reports, _) = self.inner.inner.replay_recurrent_phase_native(
            transition.phase(),
            RecurrentPhaseReplayRequest {
                learning_rate: None,
                non_finite_policy: CpuNonFinitePolicy::Propagate,
                injected_failure,
            },
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            move |outputs| {
                output_schema.validate_and_decode(outputs, CpuNonFinitePolicy::Propagate)
            },
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
