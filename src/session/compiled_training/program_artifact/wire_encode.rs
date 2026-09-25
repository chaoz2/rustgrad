//! Encoding of an authenticated compiled-training plan into its RGAP wire schema.

use super::*;

fn key_map(map: &BTreeMap<RecurrentStateKey, u64>) -> BTreeMap<String, u64> {
    map.iter()
        .map(|(key, buffer)| (key.canonical_name().to_owned(), *buffer))
        .collect()
}

fn input_key_map(map: &BTreeMap<String, RecurrentStateKey>) -> BTreeMap<String, String> {
    map.iter()
        .map(|(input, key)| (input.clone(), key.canonical_name().to_owned()))
        .collect()
}

fn manifest_wire(
    manifest: &crate::engine::RecurrentStoreGroupManifest,
) -> Result<AdamWManifestWire> {
    let [parameter, first_moment, second_moment, accumulator] = manifest.members.as_slice() else {
        return Err(training(
            "compiled program artifact native AdamW update inventory differs",
        ));
    };
    let member = |role, member: &crate::engine::RecurrentStoreGroupMember| AdamWMemberWire {
        role,
        output: member.output,
        state_buffer: member.state_buffer,
    };
    // Legacy AdamW RGAP payloads retain this historical fixed role/order wire;
    // adapt the optimizer-neutral runtime group without serializing engine-only
    // grouping metadata.
    let members = [
        member(0, parameter),
        member(1, first_moment),
        member(2, second_moment),
        member(3, accumulator),
    ];
    Ok(AdamWManifestWire { members })
}

fn phase_wire(
    capture: &CapturedMixedSchedule,
    state_buffers: &BTreeMap<RecurrentStateKey, u64>,
    state_input_keys: &BTreeMap<String, RecurrentStateKey>,
    native_updates: &[crate::engine::RecurrentStoreGroupManifest],
    clip_report: bool,
    window_loss_report: bool,
) -> Result<PhaseWire> {
    Ok(PhaseWire {
        capture: capture.to_bytes().map_err(replay_error)?,
        state_buffers: key_map(state_buffers),
        state_input_keys: input_key_map(state_input_keys),
        adamw_native_updates: native_updates
            .iter()
            .map(manifest_wire)
            .collect::<Result<_>>()?,
        clip_report,
        window_loss_report,
    })
}

fn auxiliary_wire(plan: &CompiledAdamWAuxiliaryPlan) -> Result<PhaseWire> {
    let (clip_report, window_loss_report) = plan
        .outputs
        .report_flags()
        .ok_or_else(|| training("compiled AdamW auxiliary observation schema is not canonical"))?;
    phase_wire(
        &plan.phase().capture,
        &plan.phase().state_buffers,
        &plan.state_input_keys,
        plan.phase().store_groups(),
        clip_report,
        window_loss_report,
    )
}

pub(super) fn module_wire(seal: &CompiledModuleSeal) -> ModuleWire {
    let (states, visits) = seal.checkpoint_inventory();
    ModuleWire {
        states: states
            .into_iter()
            .map(|state| {
                let sealed = seal
                    .states
                    .values()
                    .find(|sealed| sealed.name == state.name)
                    .expect("checkpoint inventory came from the seal");
                ModuleStateWire {
                    name: state.name,
                    kind: match state.kind {
                        ModuleCheckpointStateKind::Parameter => "parameter",
                        ModuleCheckpointStateKind::Buffer => "buffer",
                    }
                    .into(),
                    source_trainable: state.source_trainable,
                    policy_frozen: state.policy_frozen,
                    shape: sealed.snapshot.shape.clone(),
                    dtype: sealed.snapshot.dtype,
                }
            })
            .collect(),
        visits: visits
            .into_iter()
            .map(|visit| ModuleVisitWire {
                name: visit.name,
                canonical_name: visit.canonical_name,
            })
            .collect(),
    }
}

struct ProgramWireEncoder<'a, M> {
    owner: &'a CompiledModuleAdamWPlan<M>,
}

impl<'a, M> ProgramWireEncoder<'a, M> {
    fn new(owner: &'a CompiledModuleAdamWPlan<M>) -> Self {
        Self { owner }
    }

    fn plan(&self) -> &CompiledAdamWPlan {
        &self.owner.plan
    }

    fn main(&self) -> &CompiledTrainingPlan {
        &self.plan().inner
    }

    fn validate_observations(&self) -> Result<()> {
        validate_adamw_observation_schema(
            &self.main().phase_outputs.observations,
            self.plan().contract.clip_report,
            self.plan().contract.window_loss_report,
        )
    }

    fn main_state_buffers(&self) -> BTreeMap<RecurrentStateKey, u64> {
        let main = self.main();
        main.parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(
                main.optimizer_buffers
                    .iter()
                    .map(|(key, buffer)| (key.clone(), *buffer)),
            )
            .chain(
                main.workload_buffers
                    .iter()
                    .map(|(key, buffer)| (key.clone(), *buffer)),
            )
            .collect()
    }

    fn main_wire(&self, state_buffers: &BTreeMap<RecurrentStateKey, u64>) -> Result<MainWire> {
        let plan = self.plan();
        let main = self.main();
        Ok(MainWire {
            phase: phase_wire(
                &main.capture,
                state_buffers,
                &main.state_input_keys,
                &main.recurrent_store_groups,
                plan.contract.clip_report,
                plan.contract.window_loss_report,
            )?,
            inputs: main.inputs.clone(),
            output_names: main.phase_outputs.named_outputs.clone(),
            parameter_buffers: main.parameter_buffers.clone(),
            optimizer_buffers: key_map(&main.optimizer_buffers),
            workload_buffers: key_map(&main.workload_buffers),
            state_input_buffers: main.state_input_buffers.clone(),
            state_input_keys: input_key_map(&main.state_input_keys),
        })
    }

    fn accumulation_wire(&self) -> Result<Option<PhaseWire>> {
        let main = self.main();
        main.accumulation
            .as_ref()
            .map(|phase| {
                phase_wire(
                    &phase.phase().capture,
                    &phase.phase().state_buffers,
                    &main.state_input_keys,
                    &[],
                    false,
                    false,
                )
            })
            .transpose()
    }

    fn evaluation_wire(&self) -> Result<Option<EvaluationWire>> {
        self.plan()
            .evaluation
            .as_ref()
            .map(|evaluation| {
                Ok(EvaluationWire {
                    capture: evaluation
                        .inference
                        .capture()
                        .to_bytes()
                        .map_err(replay_error)?,
                    inputs: evaluation.inputs.clone(),
                    output_names: evaluation.output_names.clone(),
                    parameter_inputs: evaluation.parameter_inputs.clone(),
                    loss_weight_policy: evaluation.loss_weight_policy.as_ref().map(Into::into),
                    allow_zero_valid_token_microbatches: evaluation
                        .allow_zero_valid_token_microbatches,
                    capture_identity: evaluation.capture_identity,
                })
            })
            .transpose()
    }

    fn metal_wire(&self) -> Result<Option<MetalProgramWire>> {
        let plan = self.plan();
        if plan.contract.metal().is_err() {
            return Ok(None);
        }
        let main = self.main();
        let main = main
            .recurrent_capture
            .portable_training_recipe(
                &plan.contract.host_token_inputs,
                &main.frozen_parameter_nodes,
            )?
            .to_bytes()
            .map_err(captured_inference_error)?;
        let partial_flush = plan
            .partial_flush
            .as_ref()
            .map(|transition| {
                transition
                    .phase()
                    .recurrent_capture
                    .portable_recipe(PortableInferenceHostPolicy::None)?
                    .to_bytes()
                    .map_err(captured_inference_error)
            })
            .transpose()?;
        let evaluation = plan
            .evaluation
            .as_ref()
            .map(|evaluation| {
                evaluation
                    .inference
                    .portable_recipe()?
                    .to_bytes()
                    .map_err(captured_inference_error)
            })
            .transpose()?;
        Ok(Some(MetalProgramWire {
            main,
            partial_flush,
            evaluation,
        }))
    }

    fn learning_rate_wire(&self) -> LearningRateWire {
        match &self.plan().contract.learning_rate {
            CompiledLearningRatePolicy::External => LearningRateWire::External,
            CompiledLearningRatePolicy::MultiStep(schedule) => LearningRateWire::MultiStep {
                base_bits: schedule.base.to_bits(),
                gamma_bits: schedule.gamma.to_bits(),
                milestones: schedule.milestones.clone(),
            },
        }
    }

    fn adamw_wire(&self) -> AdamWPolicyWire {
        let optimizer = &self.plan().contract.optimizer;
        AdamWPolicyWire {
            beta1_bits: optimizer.beta1.to_bits(),
            beta2_bits: optimizer.beta2.to_bits(),
            eps_bits: optimizer.eps.to_bits(),
            weight_decay_bits: optimizer.weight_decay.to_bits(),
            weight_decay_exclusions: optimizer.weight_decay_exclusions.clone(),
        }
    }

    fn encode(self) -> Result<ProgramWire> {
        self.validate_observations()?;
        let main_state_buffers = self.main_state_buffers();
        let accumulation = self.accumulation_wire()?;
        let evaluation = self.evaluation_wire()?;
        let metal = self.metal_wire()?;
        let module = module_wire(&self.owner.seal);
        let main = self.main_wire(&main_state_buffers)?;
        let partial_flush = self
            .plan()
            .partial_flush
            .as_ref()
            .map(auxiliary_wire)
            .transpose()?;
        let zero_grad = self
            .plan()
            .zero_grad
            .as_ref()
            .map(auxiliary_wire)
            .transpose()?;
        let plan = self.plan();
        Ok(ProgramWire {
            format_version: if metal.is_some() {
                METAL_FORMAT_VERSION
            } else {
                LEGACY_FORMAT_VERSION
            },
            module,
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
            gradient_accumulation_steps: plan.contract.gradient_accumulation_steps,
            token_weight_policy: plan.contract.token_weight_policy.as_ref().map(Into::into),
            allow_zero_valid_token_microbatches: plan.contract.allow_zero_valid_token_microbatches,
            max_gradient_norm_bits: plan.contract.max_gradient_norm.map(f32::to_bits),
            clip_report: plan.contract.clip_report,
            window_loss_report: plan.contract.window_loss_report,
            loss_scale_bits: plan.contract.loss_scale.to_bits(),
            dropout: plan.contract.dropout.map(|dropout| DropoutWire {
                key: dropout.config.key().words(),
                blocks_per_replay: dropout.blocks_per_replay,
            }),
            host_token_inputs: plan.contract.host_token_inputs.clone(),
            frozen_parameters: plan.contract.frozen_parameters.clone(),
            learning_rate: self.learning_rate_wire(),
            optimizer: OptimizerPolicyWire::AdamW {
                adamw: self.adamw_wire(),
            },
            metal,
        })
    }
}

pub(super) fn program_wire<M>(owner: &CompiledModuleAdamWPlan<M>) -> Result<ProgramWire> {
    ProgramWireEncoder::new(owner).encode()
}

pub(super) fn momentum_program_wire<M>(
    owner: &CompiledModuleMomentumSgdPlan<M>,
) -> Result<ProgramWire> {
    let main = &owner.plan.inner;
    let state_buffers = main
        .parameter_buffers
        .iter()
        .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
        .chain(
            main.optimizer_buffers
                .iter()
                .map(|(key, buffer)| (key.clone(), *buffer)),
        )
        .chain(
            main.workload_buffers
                .iter()
                .map(|(key, buffer)| (key.clone(), *buffer)),
        )
        .collect::<BTreeMap<_, _>>();
    Ok(ProgramWire {
        format_version: OPTIMIZER_FORMAT_VERSION,
        module: module_wire(&owner.seal),
        main: MainWire {
            phase: phase_wire(
                &main.capture,
                &state_buffers,
                &main.state_input_keys,
                &main.recurrent_store_groups,
                false,
                false,
            )?,
            inputs: main.inputs.clone(),
            output_names: main.phase_outputs.named_outputs.clone(),
            parameter_buffers: main.parameter_buffers.clone(),
            optimizer_buffers: key_map(&main.optimizer_buffers),
            workload_buffers: key_map(&main.workload_buffers),
            state_input_buffers: main.state_input_buffers.clone(),
            state_input_keys: input_key_map(&main.state_input_keys),
        },
        accumulation: None,
        partial_flush: None,
        zero_grad: None,
        evaluation: None,
        gradient_accumulation_steps: 1,
        token_weight_policy: None,
        allow_zero_valid_token_microbatches: false,
        max_gradient_norm_bits: None,
        clip_report: false,
        window_loss_report: false,
        loss_scale_bits: 1.0f32.to_bits(),
        dropout: None,
        host_token_inputs: BTreeMap::new(),
        frozen_parameters: BTreeSet::new(),
        learning_rate: LearningRateWire::External,
        optimizer: OptimizerPolicyWire::MomentumSgd {
            momentum_sgd: MomentumSgdPolicyWire {},
        },
        metal: None,
    })
}
