//! Optimizer-neutral CPU compiled-training replay and recurrent state.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum CompiledStepOutputSelection {
    All,
    CommitOnly,
}

impl CompiledStepOutputSelection {
    pub(super) const fn includes_named_outputs(self) -> bool {
        matches!(self, Self::All)
    }
}

pub(super) struct CompiledStepReplayRequest {
    pub(super) inputs: BTreeMap<String, TensorData>,
    pub(super) learning_rate: Option<TensorData>,
    pub(super) non_finite_policy: CpuNonFinitePolicy,
    pub(super) output_selection: CompiledStepOutputSelection,
    pub(super) injected_failure: Option<u64>,
}

struct AdmittedTrainingStep {
    inputs: BTreeMap<String, TensorData>,
    learning_rate: Option<TensorData>,
    transaction: TrainingStepTransaction,
}

struct TrainingStepTransaction {
    next_step: u64,
    non_finite_policy: CpuNonFinitePolicy,
    output_selection: CompiledStepOutputSelection,
    injected_failure: Option<u64>,
}

struct AuthenticatedTrainingStep {
    transaction: TrainingStepTransaction,
    selected_requested: Option<Vec<u64>>,
    named_output_count: usize,
    observations: Option<CompiledTrainingObservationSchema>,
    include_observations: bool,
    validate_commit_observations: bool,
}

struct CompletedTrainingStep {
    next_step: u64,
    result: CompiledTrainingStepResult,
}

impl AdmittedTrainingStep {
    fn into_main_bindings(mut self) -> (BTreeMap<String, TensorData>, TrainingStepTransaction) {
        if let Some(learning_rate) = self.learning_rate {
            self.inputs
                .insert(LEARNING_RATE_INPUT.to_string(), learning_rate);
        }
        (self.inputs, self.transaction)
    }

    fn into_accumulation_bindings(self) -> (BTreeMap<String, TensorData>, TrainingStepTransaction) {
        (self.inputs, self.transaction)
    }
}

impl TrainingStepTransaction {
    fn authenticate_outputs(
        self,
        capture: &CapturedMixedSchedule,
        outputs: &CompiledTrainingPhaseOutputSchema,
        include_observations: bool,
        validate_commit_observations: bool,
    ) -> Result<AuthenticatedTrainingStep> {
        let expected = outputs.selected_len(true, include_observations)?;
        if capture.schedule.requested.len() != expected {
            return Err(training(
                "compiled requested output layout does not match its authenticated capture",
            ));
        }
        let selected_requested = if self.output_selection.includes_named_outputs() {
            None
        } else {
            let mut selected =
                Vec::with_capacity(outputs.selected_len(false, include_observations)?);
            selected.push(capture.schedule.requested[0]);
            selected.extend(
                capture
                    .schedule
                    .requested
                    .iter()
                    .skip(1 + outputs.named_outputs.len())
                    .copied(),
            );
            Some(selected)
        };
        Ok(AuthenticatedTrainingStep {
            transaction: self,
            selected_requested,
            named_output_count: outputs.named_outputs.len(),
            observations: include_observations.then(|| outputs.observations.clone()),
            include_observations,
            validate_commit_observations,
        })
    }
}

impl AuthenticatedTrainingStep {
    fn next_step(&self) -> u64 {
        self.transaction.next_step
    }

    fn selected_requested(&self) -> Option<&[u64]> {
        self.selected_requested.as_deref()
    }

    fn injected_failure(&self) -> Option<u64> {
        self.transaction.injected_failure
    }

    fn validate_transition<'a>(
        &self,
        values: &[TensorData],
        successors: impl IntoIterator<Item = &'a TensorData>,
    ) -> std::result::Result<(), String> {
        validate_staged_transition(values, successors, self.transaction.non_finite_policy, true)?;
        if let Some(observations) = &self.observations {
            validate_staged_observations(
                values,
                1 + usize::from(self.transaction.output_selection.includes_named_outputs())
                    * self.named_output_count,
                observations,
                self.validate_commit_observations,
                self.transaction.non_finite_policy,
            )?;
        }
        Ok(())
    }

    fn complete(
        self,
        values: Vec<TensorData>,
        outputs: &CompiledTrainingPhaseOutputSchema,
        capture_identity: u64,
    ) -> CompletedTrainingStep {
        debug_assert_eq!(
            values.len(),
            outputs
                .selected_len(
                    self.transaction.output_selection.includes_named_outputs(),
                    self.include_observations,
                )
                .expect("compiled output schema was authenticated before replay")
        );
        let next_step = self.transaction.next_step;
        let outputs = outputs.take(
            values,
            self.transaction.output_selection,
            self.include_observations,
        );
        CompletedTrainingStep {
            next_step,
            result: CompiledTrainingStepResult {
                loss: outputs.loss,
                loss_aggregation_weight: 1,
                outputs: outputs.named_outputs,
                step: next_step,
                capture_identity,
                observations: outputs.observations,
            },
        }
    }
}

/// One static CPU training program with runtime-owned recurrent state.
pub(super) struct CpuCompiledTrainingProgram {
    pub(super) capture: Arc<CapturedMixedSchedule>,
    pub(super) recurrent_capture: Arc<CompiledRecurrentCapture>,
    pub(super) runtime: EffectRuntime,
    pub(super) cursor: MixedReplayCursor,
    pub(super) inputs: BTreeMap<String, (Shape, DType)>,
    pub(super) phase_outputs: CompiledTrainingPhaseOutputSchema,
    pub(super) parameter_buffers: BTreeMap<String, u64>,
    pub(super) optimizer_buffers: BTreeMap<RecurrentStateKey, u64>,
    pub(super) workload_buffers: BTreeMap<RecurrentStateKey, u64>,
    pub(super) state_input_buffers: BTreeMap<String, u64>,
    pub(super) state_input_keys: BTreeMap<String, RecurrentStateKey>,
    pub(super) recurrent_store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
    pub(super) frozen_parameter_nodes: BTreeSet<NodeId>,
    pub(super) step: u64,
    pub(super) accumulation: Option<CompiledTrainingSiblingPlan>,
}

struct CpuAuxiliaryReplay {
    cursor: ProjectedRecurrentCursor,
    provided: BTreeMap<String, TensorData>,
}

impl CpuCompiledTrainingProgram {
    pub(super) fn preflight_native(
        &self,
        vectorized: bool,
        external_learning_rate: bool,
    ) -> Result<(RecurrentNativePreparation, Duration)> {
        let started = Instant::now();
        let mut provided = zero_inputs(&self.inputs)?;
        if external_learning_rate {
            provided.insert(
                LEARNING_RATE_INPUT.to_owned(),
                TensorData::zeros_with_dtype(Shape::from([]), DType::F32)?,
            );
        }
        let preparation = self
            .capture
            .preflight_recurrent_native(&self.runtime, &self.cursor, &provided, vectorized)
            .map_err(replay_error)?;
        Ok((preparation, started.elapsed()))
    }

    pub(super) fn finish_native(
        &self,
        preparation: RecurrentNativePreparation,
        plan: PlannedNativeItems,
        residual_wall_time: Duration,
    ) -> Result<PreparedNativeCpuProgram> {
        let replay = preparation.finish(plan).map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let wall_time = native_preparation_wall_time(trace.module, residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: self.capture_identity(),
            native_identity: trace.replay.identity,
            vectorized: trace.replay.vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            work: NativeCpuPreparationWork::from_module(trace.module),
            phases: NativeCpuPreparationPhases::from_module(trace.module, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                trace.dispatch_segmentation,
            )?,
            execution_plan: self.recurrent_capture.execution_plan().clone(),
            wall_time,
        };
        report.validate_work()?;
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    fn admit_training_step(
        &self,
        request: CompiledStepReplayRequest,
    ) -> Result<AdmittedTrainingStep> {
        let CompiledStepReplayRequest {
            inputs,
            learning_rate,
            non_finite_policy,
            output_selection,
            injected_failure,
        } = request;
        validate_training_inputs(&self.inputs, &inputs)?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, non_finite_policy)?;
        }
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| training("compiled training step overflow"))?;
        Ok(AdmittedTrainingStep {
            inputs,
            learning_rate,
            transaction: TrainingStepTransaction {
                next_step,
                non_finite_policy,
                output_selection,
                injected_failure,
            },
        })
    }

    fn publish_main_step(
        &mut self,
        completed: CompletedTrainingStep,
    ) -> CompiledTrainingStepResult {
        self.step = completed.next_step;
        completed.result
    }

    fn publish_accumulation_step(
        &mut self,
        completed: CompletedTrainingStep,
        cursor: ProjectedRecurrentCursor,
    ) -> CompiledTrainingStepResult {
        cursor.publish(&mut self.cursor);
        self.step = completed.next_step;
        completed.result
    }

    /// Executes one graph-free replay and atomically publishes every recurrent
    /// successor. The learning rate is an explicit rank-zero F32 input.
    pub(super) fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_inner(inputs, learning_rate, None)
    }

    pub(super) fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_with_output_selection(
            inputs,
            learning_rate,
            CompiledStepOutputSelection::All,
            injected_failure,
        )
    }

    pub(super) fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_commit_only_inner(inputs, learning_rate, None)
    }

    pub(super) fn step_commit_only_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_with_output_selection(
            inputs,
            learning_rate,
            CompiledStepOutputSelection::CommitOnly,
            injected_failure,
        )
    }

    fn step_with_output_selection(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        output_selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_inner_with_learning_rate(
            CompiledStepReplayRequest {
                inputs,
                learning_rate: Some(learning_rate),
                non_finite_policy: CpuNonFinitePolicy::Propagate,
                output_selection,
                injected_failure,
            },
            true,
        )
    }

    pub(super) fn step_inner_with_learning_rate(
        &mut self,
        request: CompiledStepReplayRequest,
        validate_commit_observations: bool,
    ) -> Result<CompiledTrainingStepResult> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_main_bindings();
        let transaction = transaction.authenticate_outputs(
            &self.capture,
            &self.phase_outputs,
            true,
            validate_commit_observations,
        )?;
        let replay = self
            .capture
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                &mut self.cursor,
                &provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| transaction.validate_transition(outputs, successors),
            )
            .map_err(replay_error)?;
        let completed = transaction.complete(
            replay.outputs,
            &self.phase_outputs,
            self.cursor.capture_identity(),
        );
        Ok(self.publish_main_step(completed))
    }

    pub(super) fn step_native_inner_with_learning_rate(
        &mut self,
        request: CompiledStepReplayRequest,
        validate_commit_observations: bool,
        native: NativeReplayContext<'_>,
    ) -> Result<(CompiledTrainingStepResult, NativeCpuRunReport)> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_main_bindings();
        let started = Instant::now();
        let transaction = transaction.authenticate_outputs(
            &self.capture,
            &self.phase_outputs,
            true,
            validate_commit_observations,
        )?;
        let next_step = transaction.next_step();
        let replay = native
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                &mut self.cursor,
                &provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| {
                    transaction.validate_transition(outputs, successors.iter().copied())
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let executor_wall_time = replay.executor_wall_time;
        let replay = replay.replay;
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            self.capture_identity(),
            native,
            traffic,
            executor_wall_time,
            next_step,
            started.elapsed(),
        );
        let completed = transaction.complete(
            replay.outputs,
            &self.phase_outputs,
            self.cursor.capture_identity(),
        );
        Ok((self.publish_main_step(completed), report))
    }

    pub(super) fn step_accumulation_inner_with_learning_rate(
        &mut self,
        transition: &CompiledTrainingSiblingPlan,
        request: CompiledStepReplayRequest,
    ) -> Result<CompiledTrainingStepResult> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_accumulation_bindings();
        let mut prepared =
            self.prepare_phase_replay(&transition.phase().cursor_projection, provided)?;
        let transaction = transaction.authenticate_outputs(
            &transition.phase().capture,
            &self.phase_outputs,
            false,
            false,
        )?;
        let replay = transition
            .phase()
            .capture
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| transaction.validate_transition(outputs, successors),
            )
            .map_err(replay_error)?;
        let completed =
            transaction.complete(replay.outputs, &self.phase_outputs, self.capture_identity());
        Ok(self.publish_accumulation_step(completed, prepared.cursor))
    }

    pub(super) fn step_accumulation_native_inner_with_learning_rate(
        &mut self,
        transition: &CompiledTrainingSiblingPlan,
        request: CompiledStepReplayRequest,
        native: NativeReplayContext<'_>,
    ) -> Result<(CompiledTrainingStepResult, NativeCpuRunReport)> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_accumulation_bindings();
        let started = Instant::now();
        let mut prepared =
            self.prepare_phase_replay(&transition.phase().cursor_projection, provided)?;
        let transaction = transaction.authenticate_outputs(
            &transition.phase().capture,
            &self.phase_outputs,
            false,
            false,
        )?;
        let replay = native
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| {
                    transaction.validate_transition(outputs, successors.iter().copied())
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let executor_wall_time = replay.executor_wall_time;
        let replay = replay.replay;
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            transition.phase().capture_identity,
            native,
            traffic,
            executor_wall_time,
            0,
            started.elapsed(),
        );
        let completed =
            transaction.complete(replay.outputs, &self.phase_outputs, self.capture_identity());
        Ok((
            self.publish_accumulation_step(completed, prepared.cursor),
            report,
        ))
    }

    pub(super) fn step_count(&self) -> u64 {
        self.step
    }

    pub(super) fn capture_identity(&self) -> u64 {
        self.cursor.capture_identity()
    }

    pub(super) fn plan(&self) -> Result<CompiledTrainingPlan> {
        let state_frontier = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                let buffer = self
                    .state_input_buffers
                    .get(input)
                    .ok_or_else(|| training("compiled runtime state buffer is absent"))?;
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((key.clone(), (value, state.version)))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(CompiledTrainingPlan {
            capture: self.capture.clone(),
            recurrent_capture: self.recurrent_capture.clone(),
            inputs: self.inputs.clone(),
            phase_outputs: self.phase_outputs.clone(),
            parameter_buffers: self.parameter_buffers.clone(),
            optimizer_buffers: self.optimizer_buffers.clone(),
            workload_buffers: self.workload_buffers.clone(),
            state_input_buffers: self.state_input_buffers.clone(),
            state_input_keys: self.state_input_keys.clone(),
            state_values: state_frontier
                .iter()
                .map(|(key, (value, _))| (key.clone(), value.clone()))
                .collect(),
            state_versions: state_frontier
                .into_iter()
                .map(|(key, (_, version))| (key, version))
                .collect(),
            recurrent_store_groups: self.recurrent_store_groups.clone(),
            frozen_parameter_nodes: self.frozen_parameter_nodes.clone(),
            step: self.step,
            accumulation: self.accumulation.clone(),
        })
    }

    /// Returns independent owned parameter snapshots in canonical name order.
    pub(super) fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.snapshots(&self.parameter_buffers)
    }

    /// Current logical parameter versions. Every successful step advances all
    /// parameter and optimizer-state buffers exactly once.
    pub(super) fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.versions(&self.parameter_buffers)
    }

    pub(super) fn adamw_state_snapshots(
        &self,
        state: AdamWParameterState,
    ) -> Result<BTreeMap<String, TensorData>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.parameter_for_adamw_state(state)
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.snapshots(&buffers)
    }

    pub(super) fn adamw_state_versions(
        &self,
        state: AdamWParameterState,
    ) -> Result<BTreeMap<String, u64>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.parameter_for_adamw_state(state)
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.versions(&buffers)
    }

    pub(super) fn momentum_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.momentum_parameter_name()
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.snapshots(&buffers)
    }

    pub(super) fn momentum_versions(&self) -> Result<BTreeMap<String, u64>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.momentum_parameter_name()
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.versions(&buffers)
    }

    pub(super) fn global_snapshot(&self, state: AdamWGlobalState) -> Result<TensorData> {
        let buffer = self
            .optimizer_buffers
            .get(&RecurrentStateKey::adamw_global(state))
            .ok_or_else(|| training("compiled global optimizer state is absent"))?;
        let state = self.current_state(*buffer)?;
        Ok(self
            .runtime
            .snapshot(state)
            .map_err(runtime_error)?
            .tensor()
            .clone())
    }

    pub(super) fn workload_snapshot(&self, key: &RecurrentStateKey) -> Result<TensorData> {
        let buffer = self
            .workload_buffers
            .get(key)
            .ok_or_else(|| training("compiled workload state is absent"))?;
        let state = self.current_state(*buffer)?;
        Ok(self
            .runtime
            .snapshot(state)
            .map_err(runtime_error)?
            .tensor()
            .clone())
    }

    pub(super) fn restore_frontier(
        &mut self,
        step: u64,
        values: &BTreeMap<RecurrentStateKey, TensorData>,
        versions: &BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<()> {
        let buffers = self
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(
                self.optimizer_buffers
                    .iter()
                    .map(|(name, buffer)| (name.clone(), *buffer)),
            )
            .chain(
                self.workload_buffers
                    .iter()
                    .map(|(name, buffer)| (name.clone(), *buffer)),
            )
            .collect::<BTreeMap<_, _>>();
        if values.len() != buffers.len() || values.keys().ne(buffers.keys()) {
            return Err(training("compiled checkpoint state names mismatch"));
        }
        if versions.len() != buffers.len() || versions.keys().ne(buffers.keys()) {
            return Err(training("compiled checkpoint state versions mismatch"));
        }

        let mut snapshots = Vec::with_capacity(buffers.len());
        for (name, buffer) in buffers {
            let value = &values[&name];
            let current = self.current_state(buffer)?;
            if value.shape() != &current.shape || value.dtype() != current.dtype {
                return Err(training("compiled checkpoint state descriptor mismatch"));
            }
            checked_bytes(value)?;
            let mut state = current.clone();
            state.version = versions[&name];
            snapshots.push((state, value.clone()));
        }

        let frontier = snapshots
            .iter()
            .map(|(state, _)| state.clone())
            .collect::<Vec<_>>();
        let cursor = MixedReplayCursor::resume(&self.capture, frontier).map_err(replay_error)?;
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_snapshots(snapshots)
            .map_err(runtime_error)?;
        self.runtime = runtime;
        self.cursor = cursor;
        self.step = step;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn replace_state_values(
        &mut self,
        step: u64,
        replacements: BTreeMap<RecurrentStateKey, TensorData>,
    ) -> Result<()> {
        let plan = self.plan()?;
        let mut values = plan.state_values;
        for (key, value) in replacements {
            let current = values
                .get(&key)
                .ok_or_else(|| training("compiled replacement state is absent"))?;
            if value.shape() != current.shape() || value.dtype() != current.dtype() {
                return Err(training("compiled replacement state descriptor mismatch"));
            }
            values.insert(key, value);
        }
        self.restore_frontier(step, &values, &plan.state_versions)
    }

    fn prepare_auxiliary_replay(
        &self,
        transition: &CompiledRecurrentPhasePlan,
        learning_rate: Option<TensorData>,
    ) -> Result<CpuAuxiliaryReplay> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let mut provided = BTreeMap::new();
        if let Some(learning_rate) = learning_rate {
            provided.insert(LEARNING_RATE_INPUT.to_owned(), learning_rate);
        }
        self.prepare_phase_replay(&transition.cursor_projection, provided)
    }

    fn prepare_phase_replay(
        &self,
        projection: &PreparedRecurrentCursorProjection,
        provided: BTreeMap<String, TensorData>,
    ) -> Result<CpuAuxiliaryReplay> {
        Ok(CpuAuxiliaryReplay {
            cursor: projection
                .project(&self.cursor)
                .map_err(cursor_projection_error)?,
            provided,
        })
    }

    pub(super) fn replay_auxiliary_transition(
        &mut self,
        transition: &CompiledAdamWAuxiliaryPlan,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWAuxiliaryReports> {
        let mut prepared = self.prepare_auxiliary_replay(transition.phase(), learning_rate)?;
        let output_schema = transition.outputs.clone();
        let mut reports = None;
        let _replay = transition
            .phase()
            .capture
            .replay_recurrent_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(outputs, successors, non_finite_policy, false)?;
                    reports = Some(output_schema.validate_and_decode(outputs, non_finite_policy)?);
                    Ok(())
                },
            )
            .map_err(replay_error)?;
        let reports = reports.expect("compiled auxiliary outputs were authenticated before commit");
        #[cfg(debug_assertions)]
        {
            let mut committed = _replay.committed.clone();
            committed.sort_by_key(|state| state.buffer);
            debug_assert_eq!(committed, prepared.cursor.cursor().frontier());
        }
        prepared.cursor.publish(&mut self.cursor);
        Ok(reports)
    }

    pub(super) fn preflight_native_auxiliary_transition(
        &self,
        transition: &CompiledRecurrentPhasePlan,
        vectorized: bool,
        external_learning_rate: bool,
    ) -> Result<(RecurrentNativePreparation, Duration)> {
        debug_assert!(!transition.retains_unchanged());
        let started = Instant::now();
        let learning_rate = external_learning_rate
            .then(|| TensorData::zeros_with_dtype(Shape::from([]), DType::F32))
            .transpose()?;
        let prepared = self.prepare_auxiliary_replay(transition, learning_rate)?;
        let preparation = transition
            .capture
            .preflight_recurrent_native(
                &self.runtime,
                prepared.cursor.cursor(),
                &prepared.provided,
                vectorized,
            )
            .map_err(replay_error)?;
        Ok((preparation, started.elapsed()))
    }

    pub(super) fn preflight_native_accumulation(
        &self,
        transition: &CompiledTrainingSiblingPlan,
        vectorized: bool,
    ) -> Result<(RecurrentNativePreparation, Duration)> {
        debug_assert!(transition.phase().retains_unchanged());
        let started = Instant::now();
        let prepared = self.prepare_phase_replay(
            &transition.phase().cursor_projection,
            zero_inputs(&self.inputs)?,
        )?;
        let preparation = transition
            .phase()
            .capture
            .preflight_recurrent_native_retaining_unchanged(
                &self.runtime,
                prepared.cursor.cursor(),
                &prepared.provided,
                vectorized,
            )
            .map_err(replay_error)?;
        Ok((preparation, started.elapsed()))
    }

    pub(super) fn finish_native_auxiliary_transition(
        &self,
        transition: &CompiledRecurrentPhasePlan,
        preparation: RecurrentNativePreparation,
        plan: PlannedNativeItems,
        residual_wall_time: Duration,
    ) -> Result<PreparedNativeCpuProgram> {
        let replay = preparation.finish(plan).map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let wall_time = native_preparation_wall_time(trace.module, residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: transition.capture_identity(),
            native_identity: trace.replay.identity,
            vectorized: trace.replay.vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            work: NativeCpuPreparationWork::from_module(trace.module),
            phases: NativeCpuPreparationPhases::from_module(trace.module, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                trace.dispatch_segmentation,
            )?,
            execution_plan: transition.recurrent_capture.execution_plan().clone(),
            wall_time,
        };
        report.validate_work()?;
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    pub(super) fn finish_native_accumulation(
        &self,
        transition: &CompiledTrainingSiblingPlan,
        preparation: RecurrentNativePreparation,
        plan: PlannedNativeItems,
        residual_wall_time: Duration,
    ) -> Result<PreparedNativeCpuProgram> {
        let replay = preparation.finish(plan).map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let wall_time = native_preparation_wall_time(trace.module, residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: transition.phase().capture_identity,
            native_identity: trace.replay.identity,
            vectorized: trace.replay.vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            work: NativeCpuPreparationWork::from_module(trace.module),
            phases: NativeCpuPreparationPhases::from_module(trace.module, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                trace.dispatch_segmentation,
            )?,
            execution_plan: transition
                .phase()
                .recurrent_capture
                .execution_plan()
                .clone(),
            wall_time,
        };
        report.validate_work()?;
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    pub(super) fn replay_auxiliary_transition_native(
        &mut self,
        transition: &CompiledAdamWAuxiliaryPlan,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        native: NativeReplayContext<'_>,
        successful_invocation: u64,
        injected_failure: Option<u64>,
    ) -> Result<(CompiledAdamWAuxiliaryReports, NativeCpuRunReport)> {
        let started = Instant::now();
        let mut prepared = self.prepare_auxiliary_replay(transition.phase(), learning_rate)?;
        let output_schema = transition.outputs.clone();
        let mut reports = None;
        let replay = native
            .replay_recurrent_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(
                        outputs,
                        successors.iter().copied(),
                        non_finite_policy,
                        false,
                    )?;
                    reports = Some(output_schema.validate_and_decode(outputs, non_finite_policy)?);
                    Ok(())
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let executor_wall_time = replay.executor_wall_time;
        let replay = replay.replay;
        let reports = reports.expect("compiled auxiliary outputs were authenticated before commit");
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            transition.capture_identity(),
            native,
            traffic,
            executor_wall_time,
            successful_invocation,
            started.elapsed(),
        );
        #[cfg(debug_assertions)]
        {
            let mut committed = replay.committed.clone();
            committed.sort_by_key(|state| state.buffer);
            debug_assert_eq!(committed, prepared.cursor.cursor().frontier());
        }
        prepared.cursor.publish(&mut self.cursor);
        Ok((reports, report))
    }

    fn snapshots(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, TensorData>> {
        buffers
            .iter()
            .map(|(name, buffer)| {
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((name.clone(), value))
            })
            .collect()
    }

    fn versions(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, u64>> {
        buffers
            .iter()
            .map(|(name, buffer)| Ok((name.clone(), self.current_state(*buffer)?.version)))
            .collect()
    }

    fn current_state(&self, buffer: u64) -> Result<&BufferState> {
        self.cursor
            .frontier()
            .iter()
            .find(|state| state.buffer == buffer)
            .ok_or_else(|| training("compiled persistent state is absent"))
    }
}
