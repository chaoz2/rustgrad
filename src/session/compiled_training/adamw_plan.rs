//! AdamW graph compilation, checkpoint restoration, and target preparation.

use super::*;

impl CompiledAdamWPlan {
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
        if !config.frozen_parameters.is_empty() {
            return Err(training(
                "compiled AdamW raw parameters cannot resolve frozen parameter names",
            ));
        }
        Self::compile_parameters(config, parameters, build)
    }

    fn compile_parameters<F>(
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
        reject_token_weighted_scalar_loss(&config)?;
        Self::compile_parameters_with_lowered_loss(config, parameters, build)
    }

    pub(super) fn compile_parameters_with_lowered_loss<F>(
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
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        validate_weight_decay_exclusion_names(
            &config,
            parameters.iter().map(TrainingParameterInit::name),
        )?;
        let (inner, compile_phases) = CompiledTrainingPlan::compile_observed(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            build,
        )?;
        Self::from_compiled_inner(config, inner, compile_phases)
    }

    fn compile_parameters_with_ignore_index<F>(
        config: CompiledAdamWConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>, NodeId)>,
    {
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        validate_weight_decay_exclusion_names(
            &config,
            parameters.iter().map(TrainingParameterInit::name),
        )?;
        let (inner, compile_phases) = CompiledTrainingPlan::compile_with_token_weight_observed(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            build,
        )?;
        Self::from_compiled_inner(config, inner, compile_phases)
    }

    fn from_compiled_inner(
        config: CompiledAdamWConfig,
        inner: CompiledTrainingPlan,
        mut compile_phases: CompiledTrainingCompileObservation,
    ) -> Result<Self> {
        let topology = CompiledTrainingWindowTopology::from_config(&config);
        let (partial_flush, partial_flush_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_partial_flush(&inner, &config)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        let (zero_grad, zero_grad_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_zero_grad(&inner, topology)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        compile_phases.set_auxiliary(partial_flush_phase, zero_grad_phase);
        let program_identity = inner.capture_identity()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract: CompiledAdamWContract::from_config(&config, None),
            progress: CompiledTrainingWindowProgress::INITIAL,
            evaluation: None,
            compile_phases: Some(compile_phases),
        })
    }

    /// Compiles an ordinary module forward without preparing a runtime.
    pub fn compile_module<M, F>(config: CompiledAdamWConfig, module: &M, build: F) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan =
            Self::compile_parameters(config, parameters, |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| build(module, graph, inputs),
                )
            })?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    /// Compiles a module through one explicit scalar-or-token-mean objective
    /// facade without preparing a runtime.
    ///
    /// The objective must agree with the configuration: ordinary configs
    /// accept [`CompiledAdamWObjective::Scalar`], while token-weighted configs
    /// accept [`CompiledAdamWObjective::TokenMean`]. The selected objective is
    /// lowered through the same capture path as the compatibility constructors.
    pub fn compile_module_graph<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        Self::compile_module_graph_parameters(config, module, parameter_plan, build)
    }

    /// Compiles an ignore-index token-mean module while exposing the exact
    /// compiler-owned target validity nodes to the graph builder.
    ///
    /// This opt-in surface is available only for
    /// [`CompiledAdamWConfig::with_token_weighted_ignore_index`]. Existing graph
    /// constructors retain their capture topology and bytes.
    pub fn compile_module_graph_with_ignore_index<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        Self::compile_module_graph_with_ignore_index_parameters(
            config,
            module,
            parameter_plan,
            build,
        )
    }

    pub(super) fn compile_module_graph_parameters<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan = Self::compile_parameters_with_lowered_loss(
            config,
            parameters,
            |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| {
                        let built = build(module, graph, inputs)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective(
                            &objective_config,
                            graph,
                            inputs,
                            objective,
                        )?;
                        Ok((loss, outputs))
                    },
                )
            },
        )?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    pub(super) fn compile_module_graph_with_ignore_index_parameters<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan = Self::compile_parameters_with_ignore_index(
            config,
            parameters,
            |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| {
                        let nodes = lower_ignore_index_nodes(&objective_config, graph, inputs)?;
                        let built = build(module, graph, inputs, nodes)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective_with_ignore_index_nodes(
                            &objective_config,
                            graph,
                            objective,
                            nodes,
                        )?;
                        Ok((loss, outputs, nodes.weight))
                    },
                )
            },
        )?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    /// Compiles module-bound AdamW with one device-resident Threefry block
    /// counter shared by the module's explicit residual-dropout calls.
    pub fn compile_module_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
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
        reject_token_weighted_scalar_loss(&config)?;
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            build,
        )
    }

    /// Compiles a module with recurrent dropout through the unified explicit
    /// scalar-or-token-mean objective facade.
    pub fn compile_module_graph_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let built = build(module, graph, inputs, dropout)?;
                let (objective, outputs) = built.into_parts();
                let loss =
                    lower_compiled_adamw_objective(&objective_config, graph, inputs, objective)?;
                Ok((loss, outputs))
            },
        )
    }

    /// Dropout counterpart of [`Self::compile_module_graph_with_ignore_index`].
    pub fn compile_module_graph_with_dropout_and_ignore_index<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        Self::compile_module_with_dropout_parameters_inner(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let nodes = lower_ignore_index_nodes(&objective_config, graph, inputs)?;
                let built = build(module, graph, inputs, nodes, dropout)?;
                let (objective, outputs) = built.into_parts();
                let loss = lower_compiled_adamw_objective_with_ignore_index_nodes(
                    &objective_config,
                    graph,
                    objective,
                    nodes,
                )?;
                Ok((loss, outputs, Some(nodes.weight)))
            },
        )
    }

    /// Compiles module-bound AdamW and derives its scalar differentiation root
    /// from fixed-shape per-token F32 losses and the configured token mask.
    ///
    /// The returned loss node must have exactly the mask input's descriptor.
    /// Capture owns `sum(mask * losses) / sum(mask)` as both the public loss and
    /// differentiation root, while the existing replay guard rejects an empty
    /// or malformed mask before recurrent state can advance.
    pub fn compile_token_mean_module_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
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
        let (mask_input, mask_shape) = token_mean_loss_descriptor(&config)?;
        let allow_zero_valid_token_microbatches = config.allow_zero_valid_token_microbatches;
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let (losses, outputs) = build(module, graph, inputs, dropout)?;
                let loss = lower_token_mean_loss(
                    graph,
                    losses,
                    inputs[mask_input.as_str()],
                    &mask_shape,
                    allow_zero_valid_token_microbatches,
                )?;
                Ok((loss, outputs))
            },
        )
    }

    pub(super) fn compile_module_with_dropout_parameters<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
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
        Self::compile_module_with_dropout_parameters_inner(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let (loss, outputs) = build(module, graph, inputs, dropout)?;
                Ok((loss, outputs, None))
            },
        )
    }

    pub(super) fn compile_module_with_dropout_parameters_inner<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>, Option<NodeId>)>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let topology = CompiledTrainingWindowTopology::from_config(&config);
        let workload = StateSpec::dropout_counter()?;
        let mut dropout_state = None;
        let mut frozen_parameter_nodes = BTreeSet::new();
        let (mut inner, mut compile_phases) = CompiledTrainingPlan::compile_with_workload_observed(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            Some(workload),
            |graph, inputs, parameters, counter| {
                let counter =
                    counter.ok_or_else(|| training("compiled dropout state is absent"))?;
                let mut provider = CompiledDropoutStream::new(counter, dropout);
                let (loss, outputs, token_weight) = parameter_plan
                    .lower_with_frozen_parameter_nodes(
                        graph,
                        parameters,
                        &mut frozen_parameter_nodes,
                        |graph| build(module, graph, inputs, &mut provider),
                    )?;
                let (successor, state) = provider.finish(graph)?;
                dropout_state = Some(state);
                Ok((loss, outputs, Some(successor), token_weight))
            },
        )?;
        inner.frozen_parameter_nodes = frozen_parameter_nodes;
        let dropout = dropout_state
            .ok_or_else(|| training("compiled dropout configuration produced no state"))?;
        let (partial_flush, partial_flush_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_partial_flush(&inner, &config)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        let (zero_grad, zero_grad_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_zero_grad(&inner, topology)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        compile_phases.set_auxiliary(partial_flush_phase, zero_grad_phase);
        let program_identity = inner.capture_identity()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract: CompiledAdamWContract::from_config(&config, Some(dropout)),
            progress: CompiledTrainingWindowProgress::INITIAL,
            evaluation: None,
            compile_phases: Some(compile_phases),
        })
    }

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
                RecurrentStateKey::adamw_parameter(name.clone(), AdamWParameterState::FirstMoment),
                value.clone(),
            );
        }
        for (name, value) in &decoded.second_moments {
            values.insert(
                RecurrentStateKey::adamw_parameter(name.clone(), AdamWParameterState::SecondMoment),
                value.clone(),
            );
        }
        for (name, value) in &decoded.gradient_accumulators {
            values.insert(
                RecurrentStateKey::adamw_parameter(
                    name.clone(),
                    AdamWParameterState::GradientAccumulator,
                ),
                value.clone(),
            );
        }
        values.insert(
            RecurrentStateKey::adamw_global(AdamWGlobalState::Step),
            TensorData::from_scalars(
                Shape::from([]),
                DType::U64,
                [Scalar::U(decoded.optimizer_step)],
            )?,
        );
        if topology.accumulating() {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex),
                TensorData::from_scalars(
                    Shape::from([]),
                    DType::U64,
                    [Scalar::U(decoded.accumulation_index)],
                )?,
            );
        }
        if let Some(count) = decoded.accumulated_token_count {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedTokenCount),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(count)])?,
            );
        }
        if let Some(numerator) = &decoded.accumulated_loss_numerator {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator),
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
                } else if key.is_accumulation_reset_state() {
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

    #[cfg(test)]
    pub(super) fn capture_allocations(&self) -> AdamWPlanCaptureAllocations {
        AdamWPlanCaptureAllocations {
            main: captured_schedule_allocation(&self.inner.capture.schedule),
            accumulation: self
                .inner
                .accumulation
                .as_ref()
                .map(|plan| captured_schedule_allocation(&plan.phase().capture.schedule)),
            partial_flush: self
                .partial_flush
                .as_ref()
                .map(|plan| captured_schedule_allocation(&plan.phase().capture.schedule)),
            zero_grad: self
                .zero_grad
                .as_ref()
                .map(|plan| captured_schedule_allocation(&plan.phase().capture.schedule)),
            evaluation: self
                .evaluation
                .as_ref()
                .map(|plan| captured_schedule_allocation(plan.inference.capture())),
        }
    }

    #[cfg(test)]
    pub(super) fn topology_allocations(&self) -> AdamWPlanTopologyAllocations {
        let phase = |phase: &CompiledRecurrentPhasePlan| {
            (
                Arc::as_ptr(&phase.capture) as usize,
                phase.recurrent_capture.execution_plan_allocation_identity(),
                Arc::as_ptr(&phase.cursor_projection) as usize,
            )
        };
        AdamWPlanTopologyAllocations {
            main: (
                Arc::as_ptr(&self.inner.capture) as usize,
                self.inner
                    .recurrent_capture
                    .execution_plan_allocation_identity(),
            ),
            accumulation: self
                .inner
                .accumulation
                .as_ref()
                .map(|plan| phase(plan.phase())),
            partial_flush: self.partial_flush.as_ref().map(|plan| phase(plan.phase())),
            zero_grad: self.zero_grad.as_ref().map(|plan| phase(plan.phase())),
            evaluation: self
                .evaluation
                .as_ref()
                .map(|plan| plan.inference.topology_allocation_identities()),
        }
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

    /// Prepares graph-free CPU replay from this plan's exact frontier.
    pub fn prepare_cpu(&self) -> Result<CpuCompiledAdamW> {
        self.prepare_cpu_with_non_finite_policy(CpuNonFinitePolicy::Propagate)
    }

    pub(super) fn prepare_cpu_with_non_finite_policy(
        &self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> Result<CpuCompiledAdamW> {
        validate_adamw_observation_schema(
            &self.inner.phase_outputs.observations,
            self.contract.clip_report,
            self.contract.window_loss_report,
        )?;
        if let Some(partial_flush) = &self.partial_flush {
            partial_flush.outputs.validate_report_flags(
                self.contract.clip_report,
                self.contract.window_loss_report,
            )?;
        }
        if let Some(zero_grad) = &self.zero_grad {
            zero_grad.outputs.validate_report_flags(false, false)?;
        }
        Ok(CpuCompiledAdamW {
            inner: self
                .inner
                .prepare_cpu_with_non_finite_policy(non_finite_policy)?,
            partial_flush: self.partial_flush.clone(),
            zero_grad: self.zero_grad.clone(),
            contract: self.contract.clone(),
            progress: self.progress,
            evaluation: self
                .evaluation
                .clone()
                .map(|plan| CpuCompiledEvaluation { plan }),
            non_finite_policy,
        })
    }

    /// Prepares strict-native CPU replay and compiles every attached pure
    /// program before exposing mutable session state.
    pub fn prepare_native_cpu<'a>(
        &self,
        target: &NativeCpuSessionTarget<'a>,
    ) -> Result<NativeCpuCompiledAdamW<'a>> {
        let inner = self.prepare_cpu_with_non_finite_policy(target.non_finite_policy())?;
        NativeCpuCompiledAdamW::prepare(inner, target.executor(), target.is_vectorized())
    }

    /// Prepares this authenticated plan through a concrete session target.
    ///
    /// The target's associated session and error keep backend-specific
    /// diagnostics statically available without a runtime backend enum or CPU
    /// fallback.
    pub fn prepare<'a, T>(
        &'a self,
        target: &T,
    ) -> std::result::Result<
        <T as SessionTarget<&'a Self>>::Session,
        <T as SessionTarget<&'a Self>>::Error,
    >
    where
        T: SessionTarget<&'a Self>,
    {
        target.prepare(self)
    }

    /// Renders the compiled program for strict Metal admission without
    /// creating device resources.
    pub fn metal_plan(&self, renderer: MetalRenderer) -> Result<MetalCompiledAdamWPlan> {
        let contract = self.contract.metal()?;
        let inner = self.inner.metal_plan(
            renderer.clone(),
            &self.contract.host_token_inputs,
            self.evaluation.clone(),
        )?;
        let partial_flush = self
            .partial_flush
            .as_ref()
            .map(|transition| {
                let initial_state = transition
                    .state_input_keys
                    .iter()
                    .map(|(input, key)| {
                        let value = self.inner.state_values.get(key).cloned().ok_or_else(|| {
                            training("compiled partial-flush state value is absent")
                        })?;
                        Ok((input.clone(), value))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                MetalFixedStateTransitionPlan::new(
                    transition
                        .phase()
                        .recurrent_capture
                        .stateful(initial_state)?,
                    renderer,
                    &inner.inner,
                )
                .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledAdamWPlan {
            inner,
            accumulation_capture_identity: self.accumulation_capture_identity(),
            partial_flush,
            progress: self.progress,
            flush_capture_identity: self.flush_capture_identity(),
            contract,
        })
    }

    pub fn capture_identity(&self) -> u64 {
        self.program_identity
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

    /// Returns immutable logical work and recurrent-state facts without
    /// preparing a runtime or exposing the raw mixed capture.
    pub fn inspection(&self) -> Result<CompiledAdamWInspection> {
        let recurrent_state = checked_recurrent_state_extent(
            self.inner
                .state_values
                .values()
                .map(checked_bytes)
                .collect::<Result<Vec<_>>>()?,
        )?;
        let main = (
            self.capture_identity(),
            self.inner.recurrent_capture.execution_plan().clone(),
        );
        let accumulation = self.inner.accumulation.as_ref().map(|transition| {
            (
                transition.phase().capture_identity,
                transition
                    .phase()
                    .recurrent_capture
                    .execution_plan()
                    .clone(),
            )
        });
        let partial_flush = self.partial_flush.as_ref().map(|transition| {
            (
                transition.capture_identity(),
                transition
                    .phase()
                    .recurrent_capture
                    .execution_plan()
                    .clone(),
            )
        });
        let zero_grad = self.zero_grad.as_ref().map(|transition| {
            (
                transition.capture_identity(),
                transition
                    .phase()
                    .recurrent_capture
                    .execution_plan()
                    .clone(),
            )
        });
        let evaluation = self.evaluation.as_ref().map(|evaluation| {
            (
                evaluation.capture_identity,
                evaluation.inference.execution_plan().clone(),
            )
        });
        Ok(CompiledAdamWInspection::new(
            self.step_count(),
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
            recurrent_state,
        )
        .with_compile_phases(self.compile_phases.clone()))
    }

    /// Backend-neutral graph/autograd/lowering/capture observations retained
    /// by the freshly compiled plan. Restored runtime snapshots deliberately
    /// carry no synthetic compilation evidence.
    pub fn compile_phases(&self) -> Option<&CompiledTrainingCompileObservation> {
        self.compile_phases.as_ref()
    }

    /// Returns the explicit compiled dropout policy, when present.
    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.contract.dropout.map(|dropout| dropout.config)
    }

    /// Number of Threefry U64 blocks reserved by each successful replay.
    pub fn dropout_blocks_per_replay(&self) -> Option<u64> {
        self.contract
            .dropout
            .map(|dropout| dropout.blocks_per_replay)
    }

    /// Stable identity of the private accumulation-only sibling capture.
    pub fn accumulation_capture_identity(&self) -> Option<u64> {
        self.inner
            .accumulation
            .as_ref()
            .map(|transition| transition.phase().capture_identity)
    }

    /// Stable identity of the state-only flush capture, when accumulation is
    /// enabled.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.partial_flush
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    /// Stable identity of the captured state-only accumulation reset.
    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.zero_grad
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }
}
