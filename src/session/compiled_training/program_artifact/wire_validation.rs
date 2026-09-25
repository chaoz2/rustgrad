//! Wire admission and structural validation for portable compiled-training artifacts.

use super::*;

impl ProgramWire {
    pub(super) fn discard_capture_bytes(&mut self) {
        drop(std::mem::take(&mut self.main.phase.capture));
        if let Some(phase) = &mut self.accumulation {
            drop(std::mem::take(&mut phase.capture));
        }
        if let Some(phase) = &mut self.partial_flush {
            drop(std::mem::take(&mut phase.capture));
        }
        if let Some(phase) = &mut self.zero_grad {
            drop(std::mem::take(&mut phase.capture));
        }
        if let Some(evaluation) = &mut self.evaluation {
            drop(std::mem::take(&mut evaluation.capture));
        }
        if let Some(metal) = &mut self.metal {
            drop(std::mem::take(&mut metal.main));
            if let Some(recipe) = &mut metal.partial_flush {
                drop(std::mem::take(recipe));
            }
            if let Some(recipe) = &mut metal.evaluation {
                drop(std::mem::take(recipe));
            }
        }
    }

    fn info(
        &self,
        identity: u64,
        captures: &ProgramCaptures,
    ) -> Result<CompiledTrainingProgramArtifactInfo> {
        let capture_identity = captures
            .main
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .capture_identity();
        let phase_identity = |capture: &CapturedMixedSchedule| -> Result<u64> {
            Ok(capture
                .initial_recurrent_cursor()
                .map_err(replay_error)?
                .capture_identity())
        };
        let evaluation_capture_identity = captures
            .evaluation
            .as_ref()
            .map(|capture| {
                (capture.identity
                    == self
                        .evaluation
                        .as_ref()
                        .expect("capture has wire")
                        .capture_identity)
                    .then_some(capture.identity)
                    .ok_or_else(|| training("compiled evaluation artifact identity mismatch"))
            })
            .transpose()?;
        Ok(CompiledTrainingProgramArtifactInfo {
            format_version: self.format_version,
            optimizer: self.optimizer(),
            identity,
            capture_identity,
            accumulation_capture_identity: captures
                .accumulation
                .as_ref()
                .map(|capture| phase_identity(capture.as_ref()))
                .transpose()?,
            flush_capture_identity: captures
                .partial_flush
                .as_ref()
                .map(|capture| phase_identity(capture.as_ref()))
                .transpose()?,
            zero_grad_capture_identity: captures
                .zero_grad
                .as_ref()
                .map(|capture| phase_identity(capture.as_ref()))
                .transpose()?,
            evaluation_capture_identity,
        })
    }

    fn validate_format_inventory(&self) -> Result<()> {
        if self.format_version == LEGACY_FORMAT_VERSION {
            if self.metal.is_some() || !matches!(&self.optimizer, OptimizerPolicyWire::AdamW { .. })
            {
                return Err(training("compiled v1 program artifact inventory differs"));
            }
        } else if self.format_version == METAL_FORMAT_VERSION {
            let metal = self
                .metal
                .as_ref()
                .ok_or_else(|| training("compiled v2 program artifact Metal recipe is absent"))?;
            if !matches!(&self.optimizer, OptimizerPolicyWire::AdamW { .. })
                || metal.partial_flush.is_some() != self.partial_flush.is_some()
                || metal.evaluation.is_some() != self.evaluation.is_some()
            {
                return Err(training(
                    "compiled program artifact Metal recipe inventory differs",
                ));
            }
        } else if self.format_version == OPTIMIZER_FORMAT_VERSION {
            if !matches!(&self.optimizer, OptimizerPolicyWire::MomentumSgd { .. })
                || self.metal.is_some()
                || self.accumulation.is_some()
                || self.partial_flush.is_some()
                || self.zero_grad.is_some()
                || self.evaluation.is_some()
            {
                return Err(training(
                    "compiled v3 momentum-SGD artifact inventory differs",
                ));
            }
        } else {
            return Err(training("compiled program artifact version is unsupported"));
        }
        Ok(())
    }

    fn validate_metal_host_policy(&self, captures: &ProgramCaptures) -> Result<()> {
        if self.format_version != METAL_FORMAT_VERSION {
            return Ok(());
        }
        let expected_names = self
            .host_token_inputs
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let main = captures
            .metal_main
            .as_ref()
            .expect("v2 validation requires a main Metal recipe");
        let expected_main_policy = if expected_names.is_empty() {
            PortableInferenceHostPolicy::None
        } else {
            PortableInferenceHostPolicy::Training
        };
        let expected_evaluation_policy = if expected_names.is_empty() {
            PortableInferenceHostPolicy::None
        } else {
            PortableInferenceHostPolicy::FixedGathers
        };
        if main.host_policy() != expected_main_policy
            || main.host_input_names() != expected_names
            || captures.metal_partial_flush.as_ref().is_some_and(|recipe| {
                recipe.host_policy() != PortableInferenceHostPolicy::None
                    || !recipe.host_input_names().is_empty()
            })
            || captures.metal_evaluation.as_ref().is_some_and(|recipe| {
                recipe.host_policy() != expected_evaluation_policy
                    || recipe.host_input_names() != expected_names
            })
        {
            return Err(training(
                "compiled program artifact Metal host policy differs",
            ));
        }
        Ok(())
    }

    pub(super) fn validate(
        &self,
    ) -> Result<(CompiledTrainingProgramArtifactInfo, ProgramCaptures)> {
        self.validate_format_inventory()?;
        let canonical = serde_json::to_vec(self)
            .map_err(|error| training(format!("compiled program artifact validation: {error}")))?;
        let captures = ProgramCaptures::decode(self)?;
        self.validate_metal_host_policy(&captures)?;
        let info = self.info(checksum(&canonical), &captures)?;
        validate_module_wire(&self.module, &self.frozen_parameters)?;

        self.validate_policy(&info)?;
        let main = self.validate_main(captures.main.as_ref())?;
        self.validate_auxiliary_programs(&main, &captures)?;
        self.validate_evaluation(captures.main.as_ref(), captures.evaluation.as_deref())?;
        self.validate_sibling_identities(&info)?;
        self.validate_optimizer_policy()?;
        Ok((info, captures))
    }

    fn optimizer(&self) -> CompiledTrainingOptimizer {
        match &self.optimizer {
            OptimizerPolicyWire::AdamW { .. } => CompiledTrainingOptimizer::AdamW,
            OptimizerPolicyWire::MomentumSgd { .. } => CompiledTrainingOptimizer::MomentumSgd,
        }
    }

    fn validate_policy(&self, info: &CompiledTrainingProgramArtifactInfo) -> Result<()> {
        if info.capture_identity == 0 || self.gradient_accumulation_steps == 0 {
            return Err(training("compiled program artifact policy is inconsistent"));
        }
        let topology = CompiledTrainingWindowTopology::from_validated_parts(
            self.gradient_accumulation_steps,
            self.token_weight_policy.is_some(),
            self.window_loss_report,
        );
        if topology.accumulating() != self.accumulation.is_some()
            || topology.accumulating() != self.partial_flush.is_some()
            || topology.accumulating() != self.zero_grad.is_some()
            || self.allow_zero_valid_token_microbatches && self.token_weight_policy.is_none()
            || self.loss_scale_bits == 0
            || !f32::from_bits(self.loss_scale_bits).is_finite()
            || f32::from_bits(self.loss_scale_bits) <= 0.0
            || self.clip_report && self.max_gradient_norm_bits.is_none()
            || self.max_gradient_norm_bits.is_some_and(|bits| {
                let value = f32::from_bits(bits);
                !value.is_finite() || value <= 0.0
            })
            || self
                .dropout
                .is_some_and(|dropout| dropout.blocks_per_replay == 0)
        {
            return Err(training("compiled program artifact policy is inconsistent"));
        }
        Ok(())
    }

    fn validate_main_inventory(&self) -> Result<()> {
        if self.main.phase.clip_report != self.clip_report
            || self.main.phase.window_loss_report != self.window_loss_report
            || self.main.parameter_buffers.is_empty()
            || self.main.output_names.iter().collect::<BTreeSet<_>>().len()
                != self.main.output_names.len()
        {
            return Err(training(
                "compiled program artifact main inventory is inconsistent",
            ));
        }
        Ok(())
    }

    fn validate_parameter_policy(&self) -> Result<()> {
        let trainable_parameters = self
            .module
            .states
            .iter()
            .filter(|state| {
                state.kind == "parameter" && state.source_trainable && !state.policy_frozen
            })
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        if self
            .main
            .parameter_buffers
            .keys()
            .map(String::as_str)
            .ne(trainable_parameters)
            || match &self.optimizer {
                OptimizerPolicyWire::AdamW { adamw } => adamw
                    .weight_decay_exclusions
                    .iter()
                    .any(|name| !self.main.parameter_buffers.contains_key(name)),
                OptimizerPolicyWire::MomentumSgd { .. } => false,
            }
        {
            return Err(training(
                "compiled program artifact parameter policy differs",
            ));
        }
        Ok(())
    }

    fn validate_main_output_schema(&self, main_capture: &CapturedMixedSchedule) -> Result<()> {
        let expected_requested = 1usize
            .checked_add(self.main.output_names.len())
            .and_then(|count| count.checked_add(usize::from(self.clip_report) * 2))
            .and_then(|count| count.checked_add(usize::from(self.window_loss_report) * 2))
            .ok_or_else(|| training("compiled program artifact output count overflows"))?;
        if main_capture.schedule.requested.len() != expected_requested {
            return Err(training(
                "compiled program artifact requested output schema differs",
            ));
        }
        Ok(())
    }

    fn expected_main_states(&self) -> Result<ValidatedMain> {
        let expected_main_states = self
            .main
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(decode_key_map(&self.main.optimizer_buffers)?)
            .chain(decode_key_map(&self.main.workload_buffers)?)
            .collect::<BTreeMap<_, _>>();
        let expected_flush_states = self
            .main
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(decode_key_map(&self.main.optimizer_buffers)?)
            .collect::<BTreeMap<_, _>>();
        let expected_zero_grad_states = expected_flush_states
            .iter()
            .filter(|(key, _)| key.is_accumulation_reset_state())
            .map(|(key, buffer)| (key.clone(), *buffer))
            .collect::<BTreeMap<_, _>>();
        Ok(ValidatedMain {
            states: expected_main_states,
            flush_states: expected_flush_states,
            zero_grad_states: expected_zero_grad_states,
        })
    }

    fn validate_main_state_schema(
        &self,
        captured_states: &BTreeMap<RecurrentStateKey, u64>,
        expected: &ValidatedMain,
    ) -> Result<()> {
        if captured_states != &expected.states
            || self
                .main
                .state_input_buffers
                .keys()
                .ne(self.main.state_input_keys.keys())
        {
            return Err(training("compiled program artifact state schema differs"));
        }
        let input_keys = decode_input_key_map(&self.main.state_input_keys)?;
        if input_keys.iter().any(|(input, key)| {
            self.main.state_input_buffers.get(input) != expected.states.get(key)
        }) || self.main.state_input_buffers.len() != input_keys.len()
        {
            return Err(training(
                "compiled program artifact state input mapping differs",
            ));
        }
        Ok(())
    }

    fn validate_host_token_inputs(&self) -> Result<()> {
        if self.host_token_inputs.iter().any(|(name, shape)| {
            self.main.inputs.get(name) != Some(&(shape.clone(), DType::I32))
                || shape.rank() != 2
                || shape.dims().contains(&0)
        }) {
            return Err(training(
                "compiled program artifact host-token policy differs",
            ));
        }
        Ok(())
    }

    fn validate_main_inputs(&self, main_capture: &CapturedMixedSchedule) -> Result<()> {
        let captured_inputs = main_capture
            .schedule
            .inputs
            .iter()
            .filter(|input| {
                !self.main.state_input_keys.contains_key(&input.name)
                    && input.name != LEARNING_RATE_INPUT
            })
            .map(|input| {
                (
                    input.name.clone(),
                    (input.desc.shape.clone(), input.desc.dtype),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let main_has_learning_rate = main_capture
            .schedule
            .inputs
            .iter()
            .any(|input| input.name == LEARNING_RATE_INPUT);
        if captured_inputs != self.main.inputs
            || main_has_learning_rate != matches!(&self.learning_rate, LearningRateWire::External)
        {
            return Err(training("compiled program artifact input schema differs"));
        }
        Ok(())
    }

    fn validate_main(&self, main_capture: &CapturedMixedSchedule) -> Result<ValidatedMain> {
        self.validate_main_inventory()?;
        let captured_states = validate_phase_capture(&self.main.phase, main_capture)?;
        self.validate_parameter_policy()?;
        self.validate_main_output_schema(main_capture)?;
        let expected_states = self.expected_main_states()?;
        self.validate_main_state_schema(&captured_states, &expected_states)?;
        self.validate_host_token_inputs()?;
        self.validate_main_inputs(main_capture)?;
        if let Some(policy) = &self.token_weight_policy {
            let policy = CompiledTokenWeightPolicy::from(policy);
            validate_token_weight_policy(
                &self.main.inputs,
                &policy,
                self.gradient_accumulation_steps,
            )?;
        }
        let topology = CompiledTrainingWindowTopology::from_validated_parts(
            self.gradient_accumulation_steps,
            self.token_weight_policy.is_some(),
            self.window_loss_report,
        );
        let native_manifests = if topology.accumulating() {
            NativeManifestExpectation::AdamW {
                parameters: &self.main.parameter_buffers,
                states: &expected_states.states,
            }
        } else {
            NativeManifestExpectation::None
        };
        validate_native_manifests(&self.main.phase, main_capture, native_manifests)?;
        Ok(expected_states)
    }

    fn validate_accumulation(
        &self,
        main: &ValidatedMain,
        phase: &PhaseWire,
        capture: &CapturedMixedSchedule,
    ) -> Result<()> {
        if phase.clip_report || phase.window_loss_report {
            return Err(training("compiled accumulation artifact exposes reports"));
        }
        let states = validate_phase_capture(phase, capture)?;
        if states != main.states
            || capture.schedule.requested.len() != 1 + self.main.output_names.len()
            || phase_external_inputs(capture, phase).ne(self.main.inputs.keys().cloned())
        {
            return Err(training(
                "compiled accumulation artifact frontier differs from main",
            ));
        }
        validate_native_manifests(phase, capture, NativeManifestExpectation::None)
    }

    fn validate_zero_grad(
        &self,
        main: &ValidatedMain,
        phase: &PhaseWire,
        capture: &CapturedMixedSchedule,
    ) -> Result<()> {
        let outputs = CompiledAdamWAuxiliaryOutputSchema::from_report_flags(
            phase.clip_report,
            phase.window_loss_report,
        );
        outputs.validate_report_flags(false, false)?;
        if phase.clip_report || phase.window_loss_report || !phase.adamw_native_updates.is_empty() {
            return Err(training(
                "compiled zero-grad artifact exposes update outputs",
            ));
        }
        let states = validate_phase_capture(phase, capture)?;
        if states != main.zero_grad_states
            || !capture.schedule.requested.is_empty()
            || phase_external_inputs(capture, phase).next().is_some()
        {
            return Err(training("compiled zero-grad artifact state schema differs"));
        }
        validate_native_manifests(phase, capture, NativeManifestExpectation::None)
    }

    fn validate_partial_flush(
        &self,
        main: &ValidatedMain,
        phase: &PhaseWire,
        capture: &CapturedMixedSchedule,
    ) -> Result<()> {
        let outputs = CompiledAdamWAuxiliaryOutputSchema::from_report_flags(
            phase.clip_report,
            phase.window_loss_report,
        );
        outputs.validate_report_flags(self.clip_report, self.window_loss_report)?;
        let states = validate_phase_capture(phase, capture)?;
        let expected_outputs = outputs.observations.len();
        let has_learning_rate = capture
            .schedule
            .inputs
            .iter()
            .any(|input| input.name == LEARNING_RATE_INPUT);
        if states != main.flush_states
            || capture.schedule.requested.len() != expected_outputs
            || phase_external_inputs(capture, phase).next().is_some()
            || has_learning_rate != matches!(&self.learning_rate, LearningRateWire::External)
        {
            return Err(training("compiled flush artifact state schema differs"));
        }
        validate_native_manifests(
            phase,
            capture,
            NativeManifestExpectation::AdamW {
                parameters: &self.main.parameter_buffers,
                states: &states,
            },
        )
    }

    fn validate_auxiliary_programs(
        &self,
        main: &ValidatedMain,
        captures: &ProgramCaptures,
    ) -> Result<()> {
        if let (Some(phase), Some(capture)) = (&self.accumulation, &captures.accumulation) {
            self.validate_accumulation(main, phase, capture.as_ref())?;
        }
        if let (Some(phase), Some(capture)) = (&self.zero_grad, &captures.zero_grad) {
            self.validate_zero_grad(main, phase, capture.as_ref())?;
        }
        if let (Some(phase), Some(capture)) = (&self.partial_flush, &captures.partial_flush) {
            self.validate_partial_flush(main, phase, capture.as_ref())?;
        }
        Ok(())
    }

    fn validate_evaluation(
        &self,
        main_capture: &CapturedMixedSchedule,
        capture: Option<&CapturedSchedule>,
    ) -> Result<()> {
        if let (Some(evaluation), Some(capture)) = (&self.evaluation, capture) {
            let parameter_inputs = evaluation
                .parameter_inputs
                .values()
                .cloned()
                .collect::<BTreeSet<_>>();
            let expected_inputs = evaluation
                .inputs
                .keys()
                .cloned()
                .chain(parameter_inputs.iter().cloned())
                .collect::<BTreeSet<_>>();
            let captured_inputs = capture
                .inputs
                .iter()
                .map(|input| input.name.clone())
                .collect::<BTreeSet<_>>();
            let main_descriptors = main_capture
                .initial_recurrent_cursor()
                .map_err(replay_error)?
                .frontier()
                .iter()
                .map(|state| (state.buffer, (state.shape.clone(), state.dtype)))
                .collect::<BTreeMap<_, _>>();
            if evaluation
                .parameter_inputs
                .keys()
                .ne(self.main.parameter_buffers.keys())
                || evaluation.inputs != self.main.inputs
                || parameter_inputs.len() != evaluation.parameter_inputs.len()
                || expected_inputs != captured_inputs
                || capture.requested.len() != 1 + evaluation.output_names.len()
                || evaluation
                    .output_names
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != evaluation.output_names.len()
                || evaluation.parameter_inputs.iter().any(|(parameter, name)| {
                    let Some(input) = capture.inputs.iter().find(|input| input.name == *name)
                    else {
                        return true;
                    };
                    let Some(buffer) = self.main.parameter_buffers.get(parameter) else {
                        return true;
                    };
                    main_descriptors.get(buffer)
                        != Some(&(input.desc.shape.clone(), input.desc.dtype))
                })
                || evaluation
                    .loss_weight_policy
                    .as_ref()
                    .is_some_and(|policy| {
                        Some(CompiledTokenWeightPolicy::from(policy))
                            != self
                                .token_weight_policy
                                .as_ref()
                                .map(CompiledTokenWeightPolicy::from)
                    })
                || evaluation.allow_zero_valid_token_microbatches
                    != self.allow_zero_valid_token_microbatches
            {
                return Err(training("compiled evaluation artifact schema differs"));
            }
        }
        Ok(())
    }

    fn validate_sibling_identities(
        &self,
        info: &CompiledTrainingProgramArtifactInfo,
    ) -> Result<()> {
        let sibling_identities = [
            info.accumulation_capture_identity,
            info.flush_capture_identity,
            info.zero_grad_capture_identity,
            info.evaluation_capture_identity,
        ]
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>();
        if sibling_identities.contains(&info.capture_identity)
            || sibling_identities.len()
                != usize::from(self.accumulation.is_some())
                    + usize::from(self.partial_flush.is_some())
                    + usize::from(self.zero_grad.is_some())
                    + usize::from(self.evaluation.is_some())
        {
            return Err(training(
                "compiled program artifact sibling identities alias",
            ));
        }
        Ok(())
    }

    fn validate_optimizer_policy(&self) -> Result<()> {
        if self.optimizer() == CompiledTrainingOptimizer::MomentumSgd {
            let optimizer_states = decode_key_map(&self.main.optimizer_buffers)?;
            let expected = self
                .main
                .parameter_buffers
                .keys()
                .map(RecurrentStateKey::momentum)
                .collect::<BTreeSet<_>>();
            if optimizer_states.keys().cloned().collect::<BTreeSet<_>>() != expected
                || !self.main.workload_buffers.is_empty()
                || self.gradient_accumulation_steps != 1
                || self.token_weight_policy.is_some()
                || self.allow_zero_valid_token_microbatches
                || self.max_gradient_norm_bits.is_some()
                || self.clip_report
                || self.window_loss_report
                || self.loss_scale_bits != 1.0f32.to_bits()
                || self.dropout.is_some()
                || !self.host_token_inputs.is_empty()
                || !self.frozen_parameters.is_empty()
                || !matches!(&self.learning_rate, LearningRateWire::External)
            {
                return Err(training(
                    "compiled momentum-SGD artifact policy is inconsistent",
                ));
            }
            return Ok(());
        }
        let OptimizerPolicyWire::AdamW { adamw: policy } = &self.optimizer else {
            return Err(training("compiled program artifact AdamW policy is absent"));
        };
        let mut adamw = CompiledAdamWConfig::new(
            f32::from_bits(policy.beta1_bits),
            f32::from_bits(policy.beta2_bits),
            f32::from_bits(policy.eps_bits),
            f32::from_bits(policy.weight_decay_bits),
        )?;
        adamw =
            adamw.with_weight_decay_exclusions(policy.weight_decay_exclusions.iter().cloned())?;
        if adamw.weight_decay_exclusions != policy.weight_decay_exclusions {
            return Err(training("compiled program artifact AdamW policy differs"));
        }
        Ok(())
    }
}

fn phase_external_inputs<'a>(
    capture: &'a CapturedMixedSchedule,
    phase: &'a PhaseWire,
) -> impl Iterator<Item = String> + 'a {
    let state_inputs = phase
        .state_input_keys
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    capture
        .schedule
        .inputs
        .iter()
        .filter(move |input| {
            !state_inputs.contains(input.name.as_str()) && input.name != LEARNING_RATE_INPUT
        })
        .map(|input| input.name.clone())
}

enum NativeManifestExpectation<'a> {
    None,
    AdamW {
        parameters: &'a BTreeMap<String, u64>,
        states: &'a BTreeMap<RecurrentStateKey, u64>,
    },
}

fn validate_native_manifests(
    phase: &PhaseWire,
    capture: &CapturedMixedSchedule,
    expectation: NativeManifestExpectation<'_>,
) -> Result<()> {
    let NativeManifestExpectation::AdamW { parameters, states } = expectation else {
        if !phase.adamw_native_updates.is_empty() {
            return Err(training(
                "compiled program artifact has unexpected native updates",
            ));
        }
        return Ok(());
    };
    if phase.adamw_native_updates.len() != parameters.len() {
        return Err(training(
            "compiled program artifact native update count differs",
        ));
    }
    let replacements = capture.value_bindings.iter().try_fold(
        BTreeMap::<u64, u64>::new(),
        |mut replacements, binding| {
            let item = capture
                .schedule
                .items
                .get(binding.effect_item as usize)
                .ok_or_else(|| training("compiled program artifact effect item is absent"))?;
            let state_buffer = match item.kernel.operation() {
                crate::Operation::EffectStore(payload) | crate::Operation::After(payload) => {
                    payload.target.buffer
                }
                _ => {
                    return Err(training(
                        "compiled program artifact replacement effect is invalid",
                    ));
                }
            };
            if replacements
                .insert(binding.producer_output.id, state_buffer)
                .is_some()
            {
                return Err(training(
                    "compiled program artifact replacement output repeats",
                ));
            }
            Ok(replacements)
        },
    )?;
    for ((name, parameter_buffer), manifest) in parameters.iter().zip(&phase.adamw_native_updates) {
        let state_buffer = |state| {
            states
                .get(&RecurrentStateKey::adamw_parameter(name, state))
                .copied()
                .ok_or_else(|| training("compiled program artifact AdamW state is absent"))
        };
        let expected = [
            (0, *parameter_buffer),
            (1, state_buffer(AdamWParameterState::FirstMoment)?),
            (2, state_buffer(AdamWParameterState::SecondMoment)?),
            (3, state_buffer(AdamWParameterState::GradientAccumulator)?),
        ];
        if manifest
            .members
            .iter()
            .zip(expected)
            .any(|(member, (role, buffer))| {
                member.role != role
                    || member.state_buffer != buffer
                    || replacements.get(&member.output) != Some(&buffer)
            })
        {
            return Err(training(
                "compiled program artifact native update mapping differs",
            ));
        }
    }
    Ok(())
}

fn validate_module_wire(module: &ModuleWire, frozen: &BTreeSet<String>) -> Result<()> {
    let states = module
        .states
        .iter()
        .map(|state| (state.name.as_str(), state))
        .collect::<BTreeMap<_, _>>();
    let visits = module
        .visits
        .iter()
        .map(|visit| visit.name.as_str())
        .collect::<BTreeSet<_>>();
    let policy_frozen = module
        .states
        .iter()
        .filter(|state| state.policy_frozen)
        .map(|state| state.name.clone())
        .collect::<BTreeSet<_>>();
    if states.is_empty()
        || states.len() != module.states.len()
        || visits.len() != module.visits.len()
        || policy_frozen != *frozen
        || module.visits.iter().any(|visit| {
            !states.contains_key(visit.canonical_name.as_str())
                || !visits.contains(visit.canonical_name.as_str())
        })
    {
        return Err(training(
            "compiled program artifact module topology is inconsistent",
        ));
    }
    for state in &module.states {
        validate_user_name(&state.name, "compiled artifact module state")?;
        checked_descriptor(&state.shape, state.dtype)?;
        if !matches!(state.kind.as_str(), "parameter" | "buffer")
            || state.policy_frozen && (state.kind != "parameter" || !state.source_trainable)
            || !module
                .visits
                .iter()
                .any(|visit| visit.name == state.name && visit.canonical_name == state.name)
        {
            return Err(training(
                "compiled program artifact module state is inconsistent",
            ));
        }
    }
    Ok(())
}
