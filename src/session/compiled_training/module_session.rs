//! Owned-module compiled training session lifecycle.

use super::*;

impl<M, R> CompiledModuleTrainingSession<M, R> {
    /// Read-only access to optimizer- or backend-specific diagnostics while the
    /// neutral owner retains exclusive control of module publication.
    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    /// Discards the compiled runtime frontier and returns the sealed host
    /// module without publishing any trained parameter values.
    pub fn into_module_without_publication(self) -> M {
        self.module
    }
}

impl<M: Module> CompiledModuleTrainingSession<M, CpuCompiledMomentumSgd> {
    fn prepare_momentum_plan(
        plan: CompiledModuleMomentumSgdPlan<M>,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>> {
        match CpuSessionTarget.prepare(plan) {
            Ok(session) => Ok(session),
            Err(error) => {
                let (plan, source) = error.into_parts();
                Err(CompiledModuleMomentumSgdCompileError {
                    module: plan.into_module(),
                    source,
                })
            }
        }
    }

    /// Compiles CPU momentum-SGD while taking exclusive ownership of the
    /// module for the complete replay and publication lifecycle.
    pub fn compile_momentum_sgd<F>(
        config: CompiledMomentumSgdConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let plan = CompiledModuleMomentumSgdPlan::compile(config, module, build)?;
        Self::prepare_momentum_plan(plan)
    }

    /// Compiles against a fresh module identity and restores an authenticated
    /// parameter/momentum frontier before returning the owned session.
    pub fn compile_momentum_sgd_from_checkpoint<F>(
        config: CompiledMomentumSgdConfig,
        module: M,
        checkpoint: &CompiledMomentumSgdCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let plan = CompiledModuleMomentumSgdPlan::compile_from_checkpoint(
            config, module, checkpoint, build,
        )?;
        Self::prepare_momentum_plan(plan)
    }
}

impl<M, R> CompiledModuleAdamWSession<M, R> {
    /// Read-only access to the retained AdamW runtime for diagnostics.
    pub fn runtime(&self) -> &R {
        self.training.runtime()
    }

    /// Removes the AdamW compatibility surface while retaining the exact owned
    /// module, runtime, seal, and prepared-session frontier.
    pub fn into_training_session(self) -> CompiledModuleTrainingSession<M, R> {
        self.training
    }

    pub fn into_module_without_publication(self) -> M {
        self.training.into_module_without_publication()
    }
}

impl<M: Module, R: CompiledScheduledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.training.runtime.captured_multi_step_lr()
    }

    pub fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<R::Step> {
        self.training.runtime.step_scheduled(inputs)
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training.runtime.step_batch_scheduled(batch)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<R::ScheduledFlush> {
        self.training.runtime.flush_partial_window_scheduled()
    }
}

impl<M: Module, R: CompiledAdamWCommitOnlyRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<R::Step> {
        self.training
            .runtime
            .step_commit_only(inputs, learning_rate)
    }

    pub fn step_batch_commit_only<B>(&mut self, batch: B, learning_rate: f32) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training
            .runtime
            .step_batch_commit_only(batch, learning_rate)
    }
}

impl<M: Module, R: CompiledTrainingCommitOnlyRuntime> CompiledModuleTrainingSession<M, R> {
    pub fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<R::Step> {
        self.runtime.commit_step(inputs, learning_rate)
    }

    pub fn commit_step_batch<B>(&mut self, batch: B, learning_rate: f32) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.runtime.commit_step_batch(batch, learning_rate)
    }
}

impl<M: Module, R: CompiledTrainingCommitOnlyRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<R::Step> {
        self.training.runtime.commit_step(inputs, learning_rate)
    }

    pub fn commit_step_batch<B>(&mut self, batch: B, learning_rate: f32) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training
            .runtime
            .commit_step_batch(batch, learning_rate)
    }
}

impl<M: Module, R: CompiledScheduledAdamWCommitOnlyRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<R::Step> {
        self.training.runtime.step_commit_only_scheduled(inputs)
    }

    pub fn step_batch_commit_only_scheduled<B>(&mut self, batch: B) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training
            .runtime
            .step_batch_commit_only_scheduled(batch)
    }
}

impl<M: Module, R: CompiledTrainingRuntime> CompiledModuleTrainingSession<M, R> {
    /// Atomically publishes the runtime's exact trainable frontier and returns
    /// the owned module. The complete module topology, identities, versions,
    /// descriptors, and frozen/buffer bytes must still match the compile seal.
    /// A failure retains the intact session and can be recovered with
    /// [`CompiledModuleTrainingFinishError::into_session`].
    pub fn finish(self) -> std::result::Result<M, CompiledModuleTrainingFinishError<M, R>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let parameters = match self.runtime.parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { module, .. } = self;
        Ok(module)
    }
}

impl<M: Module, R> CompiledModuleTrainingSession<M, R>
where
    R: CompiledCheckpointRuntime,
    R::Checkpoint: CompiledCheckpointParameterSnapshot,
{
    /// Atomically publishes and returns the exact checkpointed training
    /// frontier.
    ///
    /// The module seal is validated before snapshot work. One coherent
    /// checkpoint snapshot supplies both the returned resumable state and the
    /// parameter values published into the owned module, so a backend never
    /// performs a second parameter-only read. A checkpoint, decode, or
    /// publication failure retains the intact session for inspection or retry.
    pub fn finish_with_checkpoint(
        self,
    ) -> std::result::Result<(M, R::Checkpoint), CompiledModuleTrainingFinishError<M, R>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let checkpoint = match self.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let parameters = match checkpoint.checkpoint_parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { module, .. } = self;
        Ok((module, checkpoint))
    }
}

impl<M: Module, R: CompiledTrainingRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn finish(self) -> std::result::Result<M, CompiledModuleAdamWFinishError<M, R>> {
        match self.training.finish() {
            Ok(module) => Ok(module),
            Err(error) => {
                let (training, source) = error.into_parts();
                Err(CompiledModuleAdamWFinishError {
                    session: Box::new(Self { training }),
                    source,
                })
            }
        }
    }
}

impl<M: Module, R: CompiledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    /// Source-compatible AdamW finalization over the optimizer-neutral owner.
    pub fn finish_with_checkpoint(
        self,
    ) -> std::result::Result<(M, CompiledAdamWCheckpoint), CompiledModuleAdamWFinishError<M, R>>
    {
        match self.training.finish_with_checkpoint() {
            Ok(finished) => Ok(finished),
            Err(error) => {
                let (training, source) = error.into_parts();
                Err(CompiledModuleAdamWFinishError {
                    session: Box::new(Self { training }),
                    source,
                })
            }
        }
    }
}

impl<M: Module, R: CompiledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    /// Snapshots the exact optimizer frontier together with the owned
    /// module's canonical immutable state and topology.
    ///
    /// This does not publish into or release the sealed host module. The
    /// embedded optimizer checkpoint is reused byte-for-byte across v1--v9.
    /// Programs without the private accumulation sibling retain their existing
    /// v1--v8 formats; multi-replay accumulation emits v9. The module envelope
    /// remains v1 when no evaluator is attached and uses v2 only to authenticate
    /// an attached evaluator's capture identity.
    pub fn module_checkpoint(&self) -> Result<CompiledModuleAdamWCheckpoint> {
        self.training
            .seal
            .validate_unchanged(&self.training.module)?;
        let optimizer = self.training.runtime.checkpoint()?;
        self.training
            .seal
            .validate_unchanged(&self.training.module)?;
        let (states, visits) = self.training.seal.checkpoint_inventory();
        encode_module_adamw_checkpoint(
            &optimizer,
            self.training.evaluation_capture_identity,
            &states,
            &visits,
        )
    }

    /// Atomically publishes and returns one complete module checkpoint built
    /// from the exact AdamW snapshot used for publication.
    ///
    /// The checkpoint retains canonical module topology, ties, frozen
    /// parameters, and buffers in addition to the optimizer frontier. The
    /// runtime is checkpointed exactly once; encoding and publication both use
    /// that same snapshot. A seal, checkpoint, encoding, decode, or publication
    /// failure retains the intact session in [`CompiledModuleAdamWFinishError`]
    /// for inspection or retry.
    pub fn finish_with_module_checkpoint(
        self,
    ) -> std::result::Result<(M, CompiledModuleAdamWCheckpoint), CompiledModuleAdamWFinishError<M, R>>
    {
        if let Err(source) = self.training.seal.validate_unchanged(&self.training.module) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let optimizer = match self.training.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let (states, visits) = self.training.seal.checkpoint_inventory();
        let checkpoint = match encode_module_adamw_checkpoint(
            &optimizer,
            self.training.evaluation_capture_identity,
            &states,
            &visits,
        ) {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let parameters = match decode_adamw_checkpoint(optimizer.as_bytes()) {
            Ok(decoded) => decoded.parameters,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self
            .training
            .seal
            .publish(&self.training.module, &parameters)
        {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { training } = self;
        let CompiledModuleTrainingSession { module, .. } = training;
        Ok((module, checkpoint))
    }
}

impl<M> CompiledModuleAdamWSession<M, CpuCompiledAdamW> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.training.runtime.dropout_block_counter()
    }
}

impl<'a, M> CompiledModuleAdamWSession<M, NativeCpuCompiledAdamW<'a>> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.training.runtime.dropout_block_counter()
    }

    /// Returns strict-native CPU preparation evidence without exposing the
    /// sealed module or mutable runtime internals.
    pub fn native_cpu_preparation_report(&self) -> &NativeCpuCompiledAdamWPreparationReport {
        self.training.runtime.preparation_report()
    }
}

impl<M: Module> CompiledModuleAdamWSession<M, MetalCompiledAdamW> {
    /// Strict Metal replay that commits the complete device state frontier
    /// without downloading loss or named outputs.
    pub fn step_without_host_outputs(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWCommitResult> {
        self.training
            .runtime
            .step_without_host_outputs(inputs, learning_rate)
    }

    /// Returns the sealed runtime's read-only Metal session evidence without
    /// exposing the owned module or mutable backend internals.
    pub fn metal_session(&self) -> &MetalDeviceSession {
        self.training.runtime.metal_session()
    }

    /// Returns the opt-in successful-step recorder attached during target
    /// preparation, when present.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.training.runtime.execution_scoreboard()
    }

    /// Snapshots the owned Metal runtime's successfully recorded prefix.
    pub fn execution_scoreboard_report(
        &self,
    ) -> std::result::Result<Option<MetalSessionScoreboardReport>, MetalScoreboardError> {
        self.training.runtime.execution_scoreboard_report()
    }

    /// Returns the first fail-soft scoreboard recording error, when recording
    /// has frozen.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.training.runtime.scoreboard_recording_error()
    }

    /// Returns preparation evidence for the two read-only active-bank
    /// evaluators, when evaluation was attached before preparation.
    pub fn evaluation_preparation_reports(
        &self,
    ) -> Option<[&crate::runtime::metal::MetalDevicePreparationReport; 2]> {
        self.training.runtime.evaluation_preparation_reports()
    }

    /// Returns deterministic resource/execution summaries for both read-only
    /// physical-bank evaluators.
    pub fn evaluation_summaries(&self) -> Option<[&MetalDeviceSessionSummary; 2]> {
        self.training.runtime.evaluation_summaries()
    }
}

impl<M, R> CompiledAdamWRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWRuntime,
{
    fn gradient_accumulation_steps(&self) -> u64 {
        self.training.runtime.gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.training.runtime.max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.training.runtime.loss_scale()
    }

    fn window_loss_report_enabled(&self) -> bool {
        self.training.runtime.window_loss_report_enabled()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.training.runtime.optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.training.runtime.accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.training.runtime.zero_grad()
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.training.runtime.zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.training.runtime.first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.training.runtime.second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.training.runtime.gradient_accumulator_snapshots()
    }
}

impl<M, R> CompiledAdamWCommitOnlyRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWCommitOnlyRuntime,
{
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.training
            .runtime
            .step_commit_only(inputs, learning_rate)
    }
}

impl<M, R> CompiledTrainingCommitOnlyRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingCommitOnlyRuntime,
{
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.runtime.commit_step(inputs, learning_rate)
    }
}

impl<M, R> CompiledTrainingWindowResetRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingWindowResetRuntime,
{
    fn reset_gradient_window(&mut self) -> Result<CompiledTrainingWindowReset> {
        self.runtime.reset_gradient_window()
    }

    fn gradient_window_reset_capture_identity(&self) -> Option<u64> {
        self.runtime.gradient_window_reset_capture_identity()
    }
}

impl<M, R> CompiledTrainingWindowRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingWindowRuntime,
{
    fn gradient_window_size(&self) -> u64 {
        self.runtime.gradient_window_size()
    }

    fn pending_microbatch_count(&self) -> Result<u64> {
        self.runtime.pending_microbatch_count()
    }
}

impl<M, R> CompiledTrainingWindowCommitRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingWindowCommitRuntime,
{
    type WindowCommit = R::WindowCommit;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        self.runtime.commit_partial_window(learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        self.runtime.partial_window_commit_capture_identity()
    }
}

impl<M, R> CompiledTrainingRatePolicyRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingRatePolicyRuntime,
{
    fn step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.runtime.step_with_rate_policy(inputs)
    }
}

impl<M, R> CompiledTrainingRatePolicyCommitOnlyRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingRatePolicyCommitOnlyRuntime,
{
    fn commit_step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.runtime.commit_step_with_rate_policy(inputs)
    }
}

impl<M, R> CompiledTrainingRatePolicyWindowCommitRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingRatePolicyWindowCommitRuntime,
{
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        self.runtime.commit_partial_window_with_rate_policy()
    }
}

impl<M, R> CompiledTrainingCommitOnlyRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingCommitOnlyRuntime,
{
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.training.runtime.commit_step(inputs, learning_rate)
    }
}

impl<M, R> CompiledTrainingWindowCommitRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingWindowCommitRuntime + CompiledAdamWRuntime,
{
    type WindowCommit = R::WindowCommit;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        self.training.runtime.commit_partial_window(learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        self.training
            .runtime
            .partial_window_commit_capture_identity()
    }
}

impl<M, R> CompiledAdamWFlushRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWFlushRuntime,
{
    type Flush = R::Flush;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        self.training.runtime.flush_partial_window(learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        self.training.runtime.flush_capture_identity()
    }
}

impl<M, R> CompiledTrainingRatePolicyWindowCommitRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingRatePolicyWindowCommitRuntime + CompiledScheduledAdamWRuntime,
{
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        self.training
            .runtime
            .commit_partial_window_with_rate_policy()
    }
}

impl<M, R> CompiledScheduledAdamWRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledScheduledAdamWRuntime,
{
    type ScheduledFlush = R::ScheduledFlush;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.training.runtime.captured_multi_step_lr()
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        self.training.runtime.step_scheduled(inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        self.training.runtime.flush_partial_window_scheduled()
    }
}

impl<M, R> CompiledScheduledAdamWCommitOnlyRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledScheduledAdamWCommitOnlyRuntime,
{
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.training.runtime.step_commit_only_scheduled(inputs)
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for CpuSessionTarget {
    type Session = CompiledModuleAdamWSession<M, CpuCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan.plan.prepare_cpu() {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for ConfiguredCpuSessionTarget {
    type Session = CompiledModuleAdamWSession<M, CpuCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan
            .plan
            .prepare_cpu_with_non_finite_policy(self.non_finite_policy())
        {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
    }
}

impl<'executor, M: Module> SessionTarget<CompiledModuleAdamWPlan<M>>
    for NativeCpuSessionTarget<'executor>
{
    type Session = CompiledModuleAdamWSession<M, NativeCpuCompiledAdamW<'executor>>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan.plan.prepare_native_cpu(self) {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for MetalSessionTarget {
    type Session = CompiledModuleAdamWSession<M, MetalCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match <Self as SessionTarget<&CompiledAdamWPlan>>::prepare(self, &plan.plan) {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
    }
}
