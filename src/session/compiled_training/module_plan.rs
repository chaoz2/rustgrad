//! Owned-module AdamW plan compilation and authenticated preparation.

use super::*;

impl<M: Module> CompiledModuleTrainingPlan<M, CompiledAdamWPlan, Option<u64>> {
    fn build_owned<F>(
        module: M,
        frozen_parameters: &BTreeSet<String>,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M) -> Result<CompiledAdamWPlan>,
    {
        let result: Result<(CompiledAdamWPlan, CompiledModuleSeal)> = (|| {
            let seal = CompiledModuleSeal::capture(&module, frozen_parameters)?;
            let plan = build(&module)?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal))
        })();
        match result {
            Ok((plan, seal)) => Ok(Self {
                module,
                plan,
                seal,
                attachment: None,
            }),
            Err(source) => Err(adamw_compile_error(module, source)),
        }
    }

    fn build_owned_from_module_checkpoint<F>(
        config: &CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, ModuleParameterPlan) -> Result<CompiledAdamWPlan>,
    {
        let result: Result<(CompiledAdamWPlan, CompiledModuleSeal, Option<u64>)> = (|| {
            let decoded = checkpoint.decoded();
            let required_evaluation_capture_identity = decoded.evaluation_capture_identity;
            let mut seal = CompiledModuleSeal::capture(&module, &config.frozen_parameters)?;
            let immutable_values =
                seal.apply_module_checkpoint(decoded, &decoded.optimizer.decoded().parameters)?;
            let parameter_plan = ModuleParameterPlan::new(&module, &config.frozen_parameters)?
                .with_immutable_values(&immutable_values)?;
            let plan = build(&module, parameter_plan)?
                .restore_checkpoint(checkpoint.optimizer_checkpoint())?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal, required_evaluation_capture_identity))
        })();
        match result {
            Ok((plan, seal, required_evaluation_capture_identity)) => Ok(Self {
                module,
                plan,
                seal,
                attachment: required_evaluation_capture_identity,
            }),
            Err(source) => Err(adamw_compile_error(module, source)),
        }
    }

    fn authenticate_restored_evaluation(&self, evaluation: &CompiledEvaluationPlan) -> Result<()> {
        if let Some(expected) = self.attachment
            && evaluation.capture_identity != expected
        {
            return Err(training(
                "compiled module checkpoint evaluation capture identity mismatch",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_ready_for_preparation(&self) -> Result<()> {
        self.seal.validate_unchanged(&self.module)?;
        if self.attachment.is_some() {
            return Err(training(
                "compiled module checkpoint requires its authenticated evaluation capture",
            ));
        }
        Ok(())
    }

    /// Compiles AdamW from, and takes ownership of, one exact module value.
    pub fn compile<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module(config, module, build)
        })
    }

    /// Compiles the explicit recurrent-dropout workload while taking ownership
    /// of its exact module value.
    pub fn compile_with_dropout<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_with_dropout(config, dropout, module, build)
        })
    }

    /// Compiles and owns a module through the unified explicit objective
    /// facade.
    pub fn compile_graph<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph(config, module, build)
        })
    }

    /// Owned-module counterpart of
    /// [`CompiledAdamWPlan::compile_module_graph_with_ignore_index`].
    pub fn compile_graph_with_ignore_index<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph_with_ignore_index(config, module, build)
        })
    }

    /// Compiles and owns a recurrent-dropout module through the unified
    /// explicit objective facade.
    pub fn compile_graph_with_dropout<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph_with_dropout(config, dropout, module, build)
        })
    }

    /// Owned-module dropout counterpart of
    /// [`CompiledAdamWPlan::compile_module_graph_with_dropout_and_ignore_index`].
    pub fn compile_graph_with_dropout_and_ignore_index<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph_with_dropout_and_ignore_index(
                config, dropout, module, build,
            )
        })
    }

    /// Recompiles a unified owned module program from a complete module
    /// checkpoint without first mutating the destination module.
    ///
    /// Saved frozen parameters and buffers are used as capture constants.
    /// Destination topology, ties, kinds, and source trainability must match;
    /// optimizer and immutable values are published together only by finish. A
    /// v2 checkpoint carrying an evaluator identity must attach that exact
    /// evaluator before target preparation.
    pub fn compile_graph_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                CompiledAdamWPlan::compile_module_graph_parameters(
                    objective_config,
                    module,
                    parameter_plan,
                    build,
                )
            },
        )
    }

    /// Ignore-index-context counterpart of
    /// [`Self::compile_graph_from_module_checkpoint`].
    pub fn compile_graph_with_ignore_index_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                CompiledAdamWPlan::compile_module_graph_with_ignore_index_parameters(
                    objective_config,
                    module,
                    parameter_plan,
                    build,
                )
            },
        )
    }

    /// Recompiles a recurrent-dropout unified owned module program from a
    /// complete module checkpoint without mutating the destination module.
    /// A v2 checkpoint carrying an evaluator identity must attach that exact
    /// evaluator before target preparation.
    pub fn compile_graph_with_dropout_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                let parameters = parameter_plan.initial_parameters()?;
                let lower_config = objective_config.clone();
                CompiledAdamWPlan::compile_module_with_dropout_parameters(
                    objective_config,
                    dropout,
                    module,
                    parameter_plan,
                    parameters,
                    move |module, graph, inputs, dropout| {
                        let built = build(module, graph, inputs, dropout)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective(
                            &lower_config,
                            graph,
                            inputs,
                            objective,
                        )?;
                        Ok((loss, outputs))
                    },
                )
            },
        )
    }

    /// Ignore-index-context counterpart of
    /// [`Self::compile_graph_with_dropout_from_module_checkpoint`].
    pub fn compile_graph_with_dropout_and_ignore_index_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                let parameters = parameter_plan.initial_parameters()?;
                let lower_config = objective_config.clone();
                CompiledAdamWPlan::compile_module_with_dropout_parameters_inner(
                    objective_config,
                    dropout,
                    module,
                    parameter_plan,
                    parameters,
                    move |module, graph, inputs, dropout| {
                        let nodes = lower_ignore_index_nodes(&lower_config, graph, inputs)?;
                        let built = build(module, graph, inputs, nodes, dropout)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective_with_ignore_index_nodes(
                            &lower_config,
                            graph,
                            objective,
                            nodes,
                        )?;
                        Ok((loss, outputs, Some(nodes.weight)))
                    },
                )
            },
        )
    }

    /// Compatibility constructor that compiles an owned module program and
    /// then restores its authenticated AdamW frontier before preparation.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile(config, module, build).and_then(|plan| {
            plan.restore_checkpoint(checkpoint).map_err(|error| {
                let (plan, source) = error.into_parts();
                adamw_compile_error(plan.module, source)
            })
        })
    }

    /// Compatibility constructor that compiles the owned recurrent-dropout
    /// workload and then restores its complete optimizer/dropout frontier.
    pub fn compile_with_dropout_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_with_dropout(config, dropout, module, build).and_then(|plan| {
            plan.restore_checkpoint(checkpoint).map_err(|error| {
                let (plan, source) = error.into_parts();
                adamw_compile_error(plan.module, source)
            })
        })
    }

    /// Restores a checkpoint onto this already compiled owned program without
    /// rebuilding the Graph, derivatives, schedules, captures, partial-flush
    /// transition, or attached evaluation program.
    ///
    /// The returned owner contains an independent restored plan while keeping
    /// the same sealed module value. Failure retains this complete owner for
    /// inspection, retry, or preparation of its unchanged frontier.
    pub fn restore_checkpoint(
        mut self,
        checkpoint: &CompiledAdamWCheckpoint,
    ) -> std::result::Result<Self, CompiledModuleAdamWRestoreError<M>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(adamw_restore_error(self, source));
        }
        match self.plan.restore_checkpoint(checkpoint) {
            Ok(plan) => {
                self.plan = plan;
                Ok(self)
            }
            Err(source) => Err(adamw_restore_error(self, source)),
        }
    }

    /// Attaches one read-only evaluation capture to this exact owned plan.
    /// The evaluator reuses the training input schema and live canonical
    /// trainable frontier; frozen parameters and buffers remain capture-owned
    /// constants. When fresh-module restoration requires a v2-authenticated
    /// evaluator, a different capture identity rejects without consuming the
    /// plan. Failure retains the unconsumed plan for retry or recovery.
    pub fn with_evaluation<F>(
        mut self,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWEvaluationError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let result = (|| {
            if self.plan.evaluation.is_some() {
                return Err(training("compiled evaluation is already attached"));
            }
            self.seal.validate_unchanged(&self.module)?;
            let parameter_plan = self.seal.parameter_plan(&self.module)?;
            let started = Instant::now();
            let evaluation = CompiledEvaluationPlan::compile_with_parameter_plan(
                &self.module,
                &self.plan,
                parameter_plan,
                build,
            )?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                evaluation.inference.execution_plan().schedule_item_count,
            );
            self.seal.validate_unchanged(&self.module)?;
            self.authenticate_restored_evaluation(&evaluation)?;
            Ok((evaluation, phase))
        })();
        match result {
            Ok((evaluation, phase)) => {
                self.plan.evaluation = Some(evaluation);
                if let Some(observation) = &mut self.plan.compile_phases {
                    observation.set_evaluation(phase);
                }
                self.attachment = None;
                Ok(self)
            }
            Err(source) => Err(adamw_evaluation_error(self, source)),
        }
    }

    /// Attaches a read-only evaluator using the training plan's authenticated
    /// scalar or compiler-owned token-mean objective policy.
    ///
    /// Token-mean evaluation reuses the configured F32 mask input, owns masked
    /// normalization inside the captured graph, and reports the exact validated
    /// token count for weighted aggregation. The zero-token training opt-in also
    /// admits an empty evaluation batch with exact-zero loss and weight. Invalid
    /// masks fail before replay. The legacy [`Self::with_evaluation`] scalar
    /// surface remains behavior-compatible, including on token-weighted plans.
    /// A v2-authenticated fresh-module restore accepts only the saved evaluator
    /// identity and retains the plan for retry on mismatch.
    pub fn with_evaluation_graph<F>(
        mut self,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWEvaluationError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let result = (|| {
            if self.plan.evaluation.is_some() {
                return Err(training("compiled evaluation is already attached"));
            }
            self.seal.validate_unchanged(&self.module)?;
            let parameter_plan = self.seal.parameter_plan(&self.module)?;
            let started = Instant::now();
            let evaluation = CompiledEvaluationPlan::compile_graph_with_parameter_plan(
                &self.module,
                &self.plan,
                parameter_plan,
                build,
            )?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                evaluation.inference.execution_plan().schedule_item_count,
            );
            self.seal.validate_unchanged(&self.module)?;
            self.authenticate_restored_evaluation(&evaluation)?;
            Ok((evaluation, phase))
        })();
        match result {
            Ok((evaluation, phase)) => {
                self.plan.evaluation = Some(evaluation);
                if let Some(observation) = &mut self.plan.compile_phases {
                    observation.set_evaluation(phase);
                }
                self.attachment = None;
                Ok(self)
            }
            Err(source) => Err(adamw_evaluation_error(self, source)),
        }
    }

    /// Attaches a read-only token-mean evaluator while exposing the exact
    /// compiler-owned ignore-index nodes to its graph builder.
    pub fn with_evaluation_graph_and_ignore_index<F>(
        mut self,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWEvaluationError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let result = (|| {
            if self.plan.evaluation.is_some() {
                return Err(training("compiled evaluation is already attached"));
            }
            self.seal.validate_unchanged(&self.module)?;
            let parameter_plan = self.seal.parameter_plan(&self.module)?;
            let started = Instant::now();
            let evaluation =
                CompiledEvaluationPlan::compile_graph_with_ignore_index_parameter_plan(
                    &self.module,
                    &self.plan,
                    parameter_plan,
                    build,
                )?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                evaluation.inference.execution_plan().schedule_item_count,
            );
            self.seal.validate_unchanged(&self.module)?;
            self.authenticate_restored_evaluation(&evaluation)?;
            Ok((evaluation, phase))
        })();
        match result {
            Ok((evaluation, phase)) => {
                self.plan.evaluation = Some(evaluation);
                if let Some(observation) = &mut self.plan.compile_phases {
                    observation.set_evaluation(phase);
                }
                self.attachment = None;
                Ok(self)
            }
            Err(source) => Err(adamw_evaluation_error(self, source)),
        }
    }

    pub fn capture_identity(&self) -> u64 {
        self.plan.capture_identity()
    }

    /// Returns the private accumulation-only sibling capture identity when
    /// gradient accumulation is enabled for this owned plan.
    pub fn accumulation_capture_identity(&self) -> Option<u64> {
        self.plan.accumulation_capture_identity()
    }

    /// Returns the owned plan's immutable logical work and recurrent-state
    /// inspection without exposing its sealed module.
    pub fn inspection(&self) -> Result<CompiledAdamWInspection> {
        self.plan.inspection()
    }

    pub fn step_count(&self) -> u64 {
        self.plan.step_count()
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.plan.flush_capture_identity()
    }

    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.plan.dropout_config()
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.plan.captured_multi_step_lr()
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.plan
            .evaluation
            .as_ref()
            .map(|evaluation| evaluation.capture_identity)
    }

    /// Inspects strict Metal admission without exposing an independently
    /// preparable runtime path or releasing the owned module. Resource
    /// preparation still consumes this owner through [`Self::prepare`].
    pub fn metal_summary(&self, renderer: MetalRenderer) -> Result<MetalDeviceSessionSummary> {
        Ok(self.plan.metal_plan(renderer)?.summary().clone())
    }
}
