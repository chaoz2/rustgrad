use super::*;

/// Optimizer-neutral strict-Metal rendering of one compiled training program.
/// Optimizer facades retain only their policy and progress around this shared
/// recurrent execution core.
pub(super) struct MetalCompiledTrainingPlan {
    pub(super) inner: MetalStatefulInferencePlan,
    pub(super) inputs: BTreeMap<String, (Shape, DType)>,
    pub(super) output_names: Vec<String>,
    pub(super) state_input_keys: BTreeMap<String, RecurrentStateKey>,
    pub(super) program_identity: u64,
    pub(super) evaluation: Option<(MetalFixedStateReadPlan, Vec<String>, u64)>,
}

/// Optimizer-neutral owner of one prepared strict-Metal training program.
pub(super) struct MetalCompiledTrainingProgram {
    pub(super) session: MetalDeviceSession,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    pub(super) state_input_keys: BTreeMap<String, RecurrentStateKey>,
    pub(super) program_identity: u64,
    pub(super) scoreboard: Option<MetalScoreboardObserver>,
    pub(super) evaluation: Option<(MetalFixedStateReadSession, Vec<String>, u64)>,
}

pub(super) struct MetalCompiledTrainingRun {
    pub(super) loss: TensorData,
    pub(super) outputs: BTreeMap<String, TensorData>,
    pub(super) report: MetalDeviceRunReport,
}

impl MetalCompiledTrainingPlan {
    pub(super) fn prepare(
        self,
        device: MetalDevice,
        recorder: Option<MetalSessionScoreboard>,
    ) -> Result<MetalCompiledTrainingProgram> {
        let session = self
            .inner
            .prepare(device.clone())
            .map_err(metal_training_error)?;
        let evaluation = self
            .evaluation
            .map(|(plan, output_names, capture_identity)| {
                plan.prepare(device.clone(), &session)
                    .map(|session| (session, output_names, capture_identity))
                    .map_err(metal_training_error)
            })
            .transpose()?;
        let scoreboard = recorder
            .map(|recorder| {
                MetalScoreboardObserver::bind(recorder, &session)
                    .map_err(|error| training(format!("compiled Metal scoreboard: {error}")))
            })
            .transpose()?;
        Ok(MetalCompiledTrainingProgram {
            session,
            inputs: self.inputs,
            output_names: self.output_names,
            state_input_keys: self.state_input_keys,
            program_identity: self.program_identity,
            scoreboard,
            evaluation,
        })
    }
}

impl MetalCompiledTrainingProgram {
    pub(super) fn prepare_inputs(
        &self,
        mut inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<BTreeMap<String, TensorData>> {
        validate_step_inputs(&self.inputs, &inputs, &learning_rate)?;
        inputs.insert(LEARNING_RATE_INPUT.into(), learning_rate);
        Ok(inputs)
    }

    fn observe_committed_step(&mut self, run: &MetalDeviceRun) {
        if let Some(scoreboard) = &mut self.scoreboard {
            scoreboard.observe(run);
        }
    }

    pub(super) fn run(
        &mut self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<MetalCompiledTrainingRun> {
        let run = self.session.run(provided).map_err(metal_training_error)?;
        self.observe_committed_step(&run);
        let (outputs, report) = run.into_parts();
        debug_assert_eq!(outputs.len(), 1 + self.output_names.len());
        let mut outputs = outputs.into_iter();
        let loss = outputs
            .next()
            .expect("compiled Metal output cardinality was authenticated before preparation");
        let outputs = self.output_names.iter().cloned().zip(outputs).collect();
        Ok(MetalCompiledTrainingRun {
            loss,
            outputs,
            report,
        })
    }

    pub(super) fn run_without_host_outputs(
        &mut self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<MetalDeviceRunReport> {
        let run = self
            .session
            .run_epoch_without_host_outputs(provided)
            .map_err(metal_training_error)?;
        debug_assert!(run.outputs().is_empty());
        debug_assert_eq!(run.report().output_count, 0);
        debug_assert_eq!(run.report().retained_d2h_calls, 0);
        debug_assert_eq!(run.report().retained_d2h_bytes, 0);
        self.observe_committed_step(&run);
        let (_, report) = run.into_parts();
        Ok(report)
    }

    pub(super) fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<MetalCompiledEvaluationResult> {
        validate_evaluation_inputs(&self.inputs, &inputs)?;
        let (evaluation, output_names, capture_identity) = self
            .evaluation
            .as_mut()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        let run = evaluation
            .run(self.session.state_epoch(), &inputs)
            .map_err(metal_training_error)?;
        let (values, report) = run.into_parts();
        let inner = evaluation_result(values, output_names, 1, *capture_identity)?;
        Ok(MetalCompiledEvaluationResult { inner, report })
    }

    pub(super) fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation
            .as_ref()
            .map(|(_, _, capture_identity)| *capture_identity)
    }

    pub(super) fn state_snapshots(&self) -> Result<BTreeMap<RecurrentStateKey, TensorData>> {
        let snapshots = self
            .session
            .state_snapshots()
            .map_err(metal_training_error)?;
        if snapshots.len() != self.state_input_keys.len()
            || snapshots.keys().ne(self.state_input_keys.keys())
        {
            return Err(training("compiled Metal state inventory mismatch"));
        }
        snapshots
            .into_iter()
            .map(|(input, value)| {
                let key = self
                    .state_input_keys
                    .get(&input)
                    .cloned()
                    .ok_or_else(|| training("compiled Metal state key is absent"))?;
                Ok((key, value))
            })
            .collect()
    }

    pub(super) fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        let state_inputs = self
            .session
            .state_inputs()
            .iter()
            .map(|input| (input.name.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        if state_inputs.len() != self.state_input_keys.len()
            || state_inputs
                .keys()
                .copied()
                .ne(self.state_input_keys.keys().map(String::as_str))
        {
            return Err(training("compiled Metal state inventory mismatch"));
        }
        let expected = self
            .state_input_keys
            .iter()
            .filter_map(|(input, key)| {
                key.parameter_name()
                    .map(|name| (input.clone(), name.to_owned()))
            })
            .collect::<BTreeMap<_, _>>();
        if expected.is_empty() {
            return Err(training("compiled Metal parameter inventory is empty"));
        }
        let requested = expected
            .keys()
            .map(|name| state_inputs[name.as_str()].desc.id)
            .collect::<BTreeSet<_>>();
        if requested.len() != expected.len() {
            return Err(training("compiled Metal parameter state identities repeat"));
        }
        let snapshots = self
            .session
            .state_snapshot_subset(&requested)
            .map_err(metal_training_error)?;
        if snapshots.len() != expected.len() || snapshots.keys().ne(expected.keys()) {
            return Err(training(
                "compiled Metal parameter snapshot inventory mismatch",
            ));
        }
        snapshots
            .into_iter()
            .map(|(input, value)| {
                let name = expected
                    .get(&input)
                    .cloned()
                    .ok_or_else(|| training("compiled Metal parameter snapshot is unknown"))?;
                Ok((name, value))
            })
            .collect()
    }
}
