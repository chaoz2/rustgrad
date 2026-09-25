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

    /// Source-compatible identity conversion for AdamW callers. The AdamW
    /// session name aliases this optimizer-neutral owner directly.
    pub fn into_training_session(self) -> Self {
        self
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
                Err(momentum_sgd_compile_error(plan.into_module(), source))
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

    /// Recompiles a fresh owned module from a complete checkpoint containing
    /// optimizer state, immutable module values, ties, and traversal topology.
    pub fn compile_momentum_sgd_from_module_checkpoint<F>(
        config: CompiledMomentumSgdConfig,
        module: M,
        checkpoint: &CompiledModuleMomentumSgdCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let plan = CompiledModuleMomentumSgdPlan::compile_from_module_checkpoint(
            config, module, checkpoint, build,
        )?;
        Self::prepare_momentum_plan(plan)
    }
}

impl<M: Module, R: CompiledScheduledAdamWRuntime> CompiledModuleTrainingSession<M, R> {
    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.runtime.captured_multi_step_lr()
    }

    pub fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<R::Step> {
        self.runtime.step_scheduled(inputs)
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.runtime.step_batch_scheduled(batch)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<R::ScheduledFlush> {
        self.runtime.flush_partial_window_scheduled()
    }
}

impl<M: Module, R: CompiledAdamWCommitOnlyRuntime> CompiledModuleTrainingSession<M, R> {
    pub fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<R::Step> {
        self.runtime.step_commit_only(inputs, learning_rate)
    }

    pub fn step_batch_commit_only<B>(&mut self, batch: B, learning_rate: f32) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.runtime.step_batch_commit_only(batch, learning_rate)
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

impl<M: Module, R: CompiledScheduledAdamWCommitOnlyRuntime> CompiledModuleTrainingSession<M, R> {
    pub fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<R::Step> {
        self.runtime.step_commit_only_scheduled(inputs)
    }

    pub fn step_batch_commit_only_scheduled<B>(&mut self, batch: B) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.runtime.step_batch_commit_only_scheduled(batch)
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
            return Err(CompiledModuleTrainingFinishError::new(self, source));
        }
        let parameters = match self.runtime.parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError::new(self, source));
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleTrainingFinishError::new(self, source));
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
            return Err(CompiledModuleTrainingFinishError::new(self, source));
        }
        let checkpoint = match self.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError::new(self, source));
            }
        };
        let parameters = match checkpoint.checkpoint_parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError::new(self, source));
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleTrainingFinishError::new(self, source));
        }
        let Self { module, .. } = self;
        Ok((module, checkpoint))
    }
}

impl<M: Module, R> CompiledModuleTrainingSession<M, R>
where
    R: CompiledCheckpointRuntime,
    R::Checkpoint: CompiledModuleCheckpointPayload,
{
    /// Snapshots the exact optimizer frontier together with the owned
    /// module's canonical immutable state and topology.
    ///
    /// This does not publish into or release the sealed host module. The
    /// embedded optimizer checkpoint retains its optimizer-specific exact
    /// bytes; the surrounding envelope is shared across optimizer runtimes.
    pub fn module_checkpoint(&self) -> Result<CompiledModuleCheckpoint<R::Checkpoint>> {
        self.seal.validate_unchanged(&self.module)?;
        let optimizer = self.runtime.checkpoint()?;
        self.seal.validate_unchanged(&self.module)?;
        let (states, visits) = self.seal.checkpoint_inventory();
        encode_complete_module_checkpoint(
            &optimizer,
            self.evaluation_capture_identity,
            &states,
            &visits,
        )
    }

    /// Atomically publishes and returns one complete module checkpoint built
    /// from the exact optimizer snapshot used for publication.
    ///
    /// The checkpoint retains canonical module topology, ties, frozen
    /// parameters, and buffers in addition to the optimizer frontier. The
    /// runtime is checkpointed exactly once; encoding and publication both use
    /// that same snapshot. A seal, checkpoint, encoding, decode, or publication
    /// failure retains the intact session for inspection or retry.
    pub fn finish_with_module_checkpoint(
        self,
    ) -> CompiledModuleCheckpointFinishResult<M, R, R::Checkpoint> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleTrainingFinishError::new(self, source));
        }
        let optimizer = match self.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError::new(self, source));
            }
        };
        let (states, visits) = self.seal.checkpoint_inventory();
        let checkpoint = match encode_complete_module_checkpoint(
            &optimizer,
            self.evaluation_capture_identity,
            &states,
            &visits,
        ) {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError::new(self, source));
            }
        };
        let parameters = match optimizer.checkpoint_parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError::new(self, source));
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleTrainingFinishError::new(self, source));
        }
        let Self { module, .. } = self;
        Ok((module, checkpoint))
    }
}

impl<M> CompiledModuleTrainingSession<M, CpuCompiledAdamW> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.runtime.dropout_block_counter()
    }
}

impl<'a, M> CompiledModuleTrainingSession<M, NativeCpuCompiledAdamW<'a>> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.runtime.dropout_block_counter()
    }

    /// Returns strict-native CPU preparation evidence without exposing the
    /// sealed module or mutable runtime internals.
    pub fn native_cpu_preparation_report(&self) -> &NativeCpuCompiledAdamWPreparationReport {
        self.runtime.preparation_report()
    }
}

impl<M: Module> CompiledModuleTrainingSession<M, MetalCompiledAdamW> {
    /// Strict Metal replay that commits the complete device state frontier
    /// without downloading loss or named outputs.
    pub fn step_without_host_outputs(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWCommitResult> {
        self.runtime
            .step_without_host_outputs(inputs, learning_rate)
    }

    /// Returns the sealed runtime's read-only Metal session evidence without
    /// exposing the owned module or mutable backend internals.
    pub fn metal_session(&self) -> &MetalDeviceSession {
        self.runtime.metal_session()
    }

    /// Returns the opt-in successful-step recorder attached during target
    /// preparation, when present.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.runtime.execution_scoreboard()
    }

    /// Snapshots the owned Metal runtime's successfully recorded prefix.
    pub fn execution_scoreboard_report(
        &self,
    ) -> std::result::Result<Option<MetalSessionScoreboardReport>, MetalScoreboardError> {
        self.runtime.execution_scoreboard_report()
    }

    /// Returns the first fail-soft scoreboard recording error, when recording
    /// has frozen.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.runtime.scoreboard_recording_error()
    }

    /// Returns preparation evidence for the two read-only active-bank
    /// evaluators, when evaluation was attached before preparation.
    pub fn evaluation_preparation_reports(
        &self,
    ) -> Option<[&crate::runtime::metal::MetalDevicePreparationReport; 2]> {
        self.runtime.evaluation_preparation_reports()
    }

    /// Returns deterministic resource/execution summaries for both read-only
    /// physical-bank evaluators.
    pub fn evaluation_summaries(&self) -> Option<[&MetalDeviceSessionSummary; 2]> {
        self.runtime.evaluation_summaries()
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
            return Err(adamw_prepare_error(plan, source));
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan.plan.prepare_cpu() {
            Ok(runtime) => runtime,
            Err(source) => return Err(adamw_prepare_error(plan, source)),
        };
        let CompiledModuleTrainingPlan {
            module,
            seal,
            plan: _,
            attachment: _,
        } = plan;
        Ok(CompiledModuleTrainingSession::adamw(
            module,
            runtime,
            seal,
            evaluation_capture_identity,
        ))
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
            return Err(adamw_prepare_error(plan, source));
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan
            .plan
            .prepare_cpu_with_non_finite_policy(self.non_finite_policy())
        {
            Ok(runtime) => runtime,
            Err(source) => return Err(adamw_prepare_error(plan, source)),
        };
        let CompiledModuleTrainingPlan {
            module,
            seal,
            plan: _,
            attachment: _,
        } = plan;
        Ok(CompiledModuleTrainingSession::adamw(
            module,
            runtime,
            seal,
            evaluation_capture_identity,
        ))
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
            return Err(adamw_prepare_error(plan, source));
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan.plan.prepare_native_cpu(self) {
            Ok(runtime) => runtime,
            Err(source) => return Err(adamw_prepare_error(plan, source)),
        };
        let CompiledModuleTrainingPlan {
            module,
            seal,
            plan: _,
            attachment: _,
        } = plan;
        Ok(CompiledModuleTrainingSession::adamw(
            module,
            runtime,
            seal,
            evaluation_capture_identity,
        ))
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
            return Err(adamw_prepare_error(plan, source));
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match <Self as SessionTarget<&CompiledAdamWPlan>>::prepare(self, &plan.plan) {
            Ok(runtime) => runtime,
            Err(source) => return Err(adamw_prepare_error(plan, source)),
        };
        let CompiledModuleTrainingPlan {
            module,
            seal,
            plan: _,
            attachment: _,
        } = plan;
        Ok(CompiledModuleTrainingSession::adamw(
            module,
            runtime,
            seal,
            evaluation_capture_identity,
        ))
    }
}
