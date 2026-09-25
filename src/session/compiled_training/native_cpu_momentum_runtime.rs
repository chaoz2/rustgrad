//! Strict-native CPU momentum-SGD replay over the shared recurrent-state core.

use super::*;

/// Strict-native CPU momentum-SGD prepared from the same resource-free plan as
/// [`CpuCompiledMomentumSgd`]. Parameters, momentum slots, checkpointing, and
/// publication remain owned by the shared CPU runtime; only pure schedule
/// execution is replaced with fail-closed native JIT replay.
pub struct NativeCpuCompiledMomentumSgd<'a> {
    pub(super) inner: CpuCompiledMomentumSgd,
    executor: &'a CapturedReplayExecutor,
    main_replay: PreparedRecurrentNativeReplay,
    preparation: NativeCpuCompiledTrainingPreparationReport,
    successful_steps: u64,
}

impl<'a> NativeCpuCompiledMomentumSgd<'a> {
    pub(super) fn prepare(
        inner: CpuCompiledMomentumSgd,
        executor: &'a CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<Self> {
        let (mut main_preparation, main_residual) =
            inner.inner.preflight_native(vectorized, true)?;
        let main_draft = {
            let (pure, inputs) = main_preparation.pure_and_inputs();
            executor
                .preflight_native_items_with_store_groups(
                    pure,
                    inputs,
                    &inner.inner.recurrent_store_groups,
                )
                .map_err(replay_error)?
        };
        main_preparation.release_input_witnesses();

        let roles = vec![NativeCpuTrainingProgramRole::Main];
        let (plans, compilation) = executor
            .plan_native_item_drafts(vec![(main_preparation.pure(), main_draft)], vectorized)
            .map_err(replay_error)?;
        let NativeCpuTrainingPrograms {
            main: main_plan,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
        } = NativeCpuTrainingPrograms::from_ordered(roles.clone(), plans)?;
        if accumulation.is_some()
            || partial_flush.is_some()
            || zero_grad.is_some()
            || evaluation.is_some()
        {
            return Err(training(
                "compiled native CPU momentum-SGD program inventory is excessive",
            ));
        }

        let PreparedNativeCpuProgram {
            report: main_report,
            replay: main_replay,
        } = inner
            .inner
            .finish_native(main_preparation, main_plan, main_residual)?;
        let (recurrent_state_count, recurrent_state_bytes) = checked_recurrent_state_extent(
            inner
                .inner
                .cursor
                .frontier()
                .iter()
                .map(|state| state.bytes),
        )?;
        let preparation = NativeCpuCompiledTrainingPreparationReport::from_compilation(
            &roles,
            NativeCpuTrainingPrograms {
                main: main_report,
                accumulation: None,
                partial_flush: None,
                zero_grad: None,
                evaluation: None,
            },
            (recurrent_state_count, recurrent_state_bytes),
            compilation,
        )?;

        Ok(Self {
            inner,
            executor,
            main_replay,
            preparation,
            successful_steps: 0,
        })
    }

    pub fn preparation_report(&self) -> &NativeCpuCompiledMomentumSgdPreparationReport {
        &self.preparation
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledMomentumSgdStepResult> {
        self.step_with_output_selection(
            inputs,
            learning_rate,
            CompiledStepOutputSelection::All,
            None,
        )
    }

    /// Commits one native replay while omitting only graph-named outputs.
    pub fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledMomentumSgdStepResult> {
        self.step_with_output_selection(
            inputs,
            learning_rate,
            CompiledStepOutputSelection::CommitOnly,
            None,
        )
    }

    fn step_with_output_selection(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        output_selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledMomentumSgdStepResult> {
        let successful_invocation = self
            .successful_steps
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU run count overflow"))?;
        let request = CompiledStepReplayRequest {
            inputs,
            learning_rate: Some(learning_rate),
            non_finite_policy: self.inner.non_finite_policy,
            output_selection,
            injected_failure,
        };
        let (inner, mut report) = self.inner.inner.step_native_inner_with_learning_rate(
            request,
            true,
            NativeReplayContext::new(self.executor, &mut self.main_replay),
        )?;
        report.successful_invocation = successful_invocation;
        self.successful_steps = successful_invocation;
        Ok(NativeCpuCompiledTrainingStepResult { inner, report })
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.inner.non_finite_policy()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn momentum_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.momentum_snapshots()
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn momentum_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.momentum_versions()
    }

    pub fn checkpoint(&self) -> Result<CompiledMomentumSgdCheckpoint> {
        self.inner.checkpoint()
    }

    pub fn restore_checkpoint_in_place(
        &mut self,
        checkpoint: &CompiledMomentumSgdCheckpoint,
    ) -> Result<()> {
        self.inner.restore_checkpoint_in_place(checkpoint)
    }

    #[cfg(test)]
    pub(super) fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledMomentumSgdStepResult> {
        self.step_with_output_selection(
            inputs,
            learning_rate,
            CompiledStepOutputSelection::All,
            injected_failure,
        )
    }
}

impl CompiledTrainingRuntime for NativeCpuCompiledMomentumSgd<'_> {
    type Step = NativeCpuCompiledMomentumSgdStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledMomentumSgd::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        NativeCpuCompiledMomentumSgd::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        NativeCpuCompiledMomentumSgd::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        NativeCpuCompiledMomentumSgd::parameter_snapshots(self)
    }
}

impl CompiledTrainingCommitOnlyRuntime for NativeCpuCompiledMomentumSgd<'_> {
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledMomentumSgd::commit_step(self, inputs, learning_rate)
    }
}

impl CompiledCheckpointRuntime for NativeCpuCompiledMomentumSgd<'_> {
    type Checkpoint = CompiledMomentumSgdCheckpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        NativeCpuCompiledMomentumSgd::checkpoint(self)
    }
}

impl CompiledCheckpointRestoreRuntime for NativeCpuCompiledMomentumSgd<'_> {
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        NativeCpuCompiledMomentumSgd::restore_checkpoint_in_place(self, checkpoint)
    }
}
