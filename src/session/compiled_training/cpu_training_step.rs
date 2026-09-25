//! Atomic CPU replay for one compiled training step.

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

impl CpuCompiledTrainingProgram {
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
}
