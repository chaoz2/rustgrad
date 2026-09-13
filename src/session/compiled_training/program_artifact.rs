use super::*;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"RGAP";
const FORMAT_VERSION: u8 = 1;
const MAX_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAdamWProgramArtifact {
    bytes: Vec<u8>,
    info: CompiledAdamWProgramArtifactInfo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWProgramArtifactInfo {
    identity: u64,
    capture_identity: u64,
    accumulation_capture_identity: Option<u64>,
    flush_capture_identity: Option<u64>,
    zero_grad_capture_identity: Option<u64>,
    evaluation_capture_identity: Option<u64>,
}

impl CompiledAdamWProgramArtifactInfo {
    pub fn format_version(&self) -> u8 {
        FORMAT_VERSION
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub fn accumulation_capture_identity(&self) -> Option<u64> {
        self.accumulation_capture_identity
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.zero_grad_capture_identity
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation_capture_identity
    }
}

impl CompiledAdamWProgramArtifact {
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        let wire = decode(&bytes)?;
        let info = wire.validate()?;
        Ok(Self { bytes, info })
    }

    pub fn info(&self) -> &CompiledAdamWProgramArtifactInfo {
        &self.info
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

pub struct CompiledModuleAdamWArtifactRestoreError<M> {
    module: M,
    source: Error,
}

impl<M> CompiledModuleAdamWArtifactRestoreError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_module(self) -> M {
        self.module
    }

    pub fn into_parts(self) -> (M, Error) {
        (self.module, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleAdamWArtifactRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWArtifactRestoreError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleAdamWArtifactRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "compiled AdamW program artifact restore failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleAdamWArtifactRestoreError<M> {}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProgramWire {
    module: ModuleWire,
    main: MainWire,
    accumulation: Option<PhaseWire>,
    partial_flush: Option<PhaseWire>,
    zero_grad: Option<PhaseWire>,
    evaluation: Option<EvaluationWire>,
    gradient_accumulation_steps: u64,
    token_weight_policy: Option<TokenWeightWire>,
    allow_zero_valid_token_microbatches: bool,
    max_gradient_norm_bits: Option<u32>,
    clip_report: bool,
    window_loss_report: bool,
    loss_scale_bits: u32,
    dropout: Option<DropoutWire>,
    host_token_inputs: BTreeMap<String, Shape>,
    frozen_parameters: BTreeSet<String>,
    learning_rate: LearningRateWire,
    adamw: AdamWPolicyWire,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MainWire {
    phase: PhaseWire,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    parameter_buffers: BTreeMap<String, u64>,
    optimizer_buffers: BTreeMap<String, u64>,
    workload_buffers: BTreeMap<String, u64>,
    state_input_buffers: BTreeMap<String, u64>,
    state_input_keys: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PhaseWire {
    capture: Vec<u8>,
    state_buffers: BTreeMap<String, u64>,
    state_input_keys: BTreeMap<String, String>,
    adamw_native_updates: Vec<AdamWManifestWire>,
    clip_report: bool,
    window_loss_report: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EvaluationWire {
    capture: Vec<u8>,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    parameter_inputs: BTreeMap<String, String>,
    loss_weight_policy: Option<TokenWeightWire>,
    allow_zero_valid_token_microbatches: bool,
    capture_identity: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModuleWire {
    states: Vec<ModuleStateWire>,
    visits: Vec<ModuleVisitWire>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModuleStateWire {
    name: String,
    kind: String,
    source_trainable: bool,
    policy_frozen: bool,
    shape: Shape,
    dtype: DType,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModuleVisitWire {
    name: String,
    canonical_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
enum TokenWeightWire {
    ExplicitMask(String),
    IgnoreIndex { target_input: String, value: i32 },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
enum LearningRateWire {
    External,
    MultiStep {
        base_bits: u32,
        gamma_bits: u32,
        milestones: Vec<u64>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AdamWPolicyWire {
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    weight_decay_bits: u32,
    weight_decay_exclusions: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct DropoutWire {
    key: [u32; 2],
    blocks_per_replay: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AdamWManifestWire {
    members: [AdamWMemberWire; 4],
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct AdamWMemberWire {
    role: u8,
    output: u64,
    state_buffer: u64,
}

struct ValidatedMain {
    capture: CapturedMixedSchedule,
    states: BTreeMap<RecurrentStateKey, u64>,
    flush_states: BTreeMap<RecurrentStateKey, u64>,
    zero_grad_states: BTreeMap<RecurrentStateKey, u64>,
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn encode(wire: &ProgramWire) -> Result<Vec<u8>> {
    let payload = serde_json::to_vec(wire)
        .map_err(|error| training(format!("compiled program artifact encode: {error}")))?;
    let length = u64::try_from(payload.len())
        .map_err(|_| training("compiled program artifact length overflows"))?;
    let total = payload
        .len()
        .checked_add(21)
        .ok_or_else(|| training("compiled program artifact length overflows"))?;
    if total > MAX_ARTIFACT_BYTES {
        return Err(training("compiled program artifact exceeds byte limit"));
    }
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(MAGIC);
    bytes.push(FORMAT_VERSION);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<ProgramWire> {
    if bytes.len() < 21 || bytes.len() > MAX_ARTIFACT_BYTES || &bytes[..4] != MAGIC {
        return Err(training("compiled program artifact header is invalid"));
    }
    if bytes[4] != FORMAT_VERSION {
        return Err(training("compiled program artifact version is unsupported"));
    }
    let payload_len = u64::from_le_bytes(
        bytes[5..13]
            .try_into()
            .map_err(|_| training("compiled program artifact length is invalid"))?,
    );
    let payload_len = usize::try_from(payload_len)
        .map_err(|_| training("compiled program artifact length overflows"))?;
    let payload_end = 13usize
        .checked_add(payload_len)
        .ok_or_else(|| training("compiled program artifact length overflows"))?;
    if payload_end.checked_add(8) != Some(bytes.len()) {
        return Err(training("compiled program artifact length is invalid"));
    }
    let expected = u64::from_le_bytes(
        bytes[payload_end..]
            .try_into()
            .map_err(|_| training("compiled program artifact checksum is invalid"))?,
    );
    if checksum(&bytes[..payload_end]) != expected {
        return Err(training("compiled program artifact checksum mismatch"));
    }
    serde_json::from_slice(&bytes[13..payload_end])
        .map_err(|error| training(format!("compiled program artifact payload: {error}")))
}

#[cfg(test)]
pub(super) fn rewrite_json_for_test<F>(
    artifact: &CompiledAdamWProgramArtifact,
    rewrite: F,
) -> (Vec<u8>, CompiledAdamWProgramArtifact)
where
    F: FnOnce(&mut serde_json::Value),
{
    let wire = decode(artifact.as_bytes()).expect("valid test artifact");
    let mut json = serde_json::to_value(wire).expect("serializable test artifact");
    rewrite(&mut json);
    let wire = serde_json::from_value(json).expect("well-formed rewritten test artifact");
    let bytes = encode(&wire).expect("bounded rewritten test artifact");
    let unchecked = CompiledAdamWProgramArtifact {
        bytes: bytes.clone(),
        info: *artifact.info(),
    };
    (bytes, unchecked)
}

impl ProgramWire {
    fn info(&self, identity: u64) -> Result<CompiledAdamWProgramArtifactInfo> {
        let main =
            CapturedMixedSchedule::from_bytes(&self.main.phase.capture).map_err(replay_error)?;
        let capture_identity = main
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .capture_identity();
        let phase_identity = |phase: &PhaseWire| -> Result<u64> {
            Ok(CapturedMixedSchedule::from_bytes(&phase.capture)
                .map_err(replay_error)?
                .initial_recurrent_cursor()
                .map_err(replay_error)?
                .capture_identity())
        };
        let evaluation_capture_identity = self
            .evaluation
            .as_ref()
            .map(|wire| {
                CapturedSchedule::from_bytes(&wire.capture)
                    .map_err(replay_error)
                    .and_then(|capture| {
                        (capture.identity == wire.capture_identity)
                            .then_some(capture.identity)
                            .ok_or_else(|| {
                                training("compiled evaluation artifact identity mismatch")
                            })
                    })
            })
            .transpose()?;
        Ok(CompiledAdamWProgramArtifactInfo {
            identity,
            capture_identity,
            accumulation_capture_identity: self
                .accumulation
                .as_ref()
                .map(phase_identity)
                .transpose()?,
            flush_capture_identity: self
                .partial_flush
                .as_ref()
                .map(phase_identity)
                .transpose()?,
            zero_grad_capture_identity: self.zero_grad.as_ref().map(phase_identity).transpose()?,
            evaluation_capture_identity,
        })
    }

    fn validate(&self) -> Result<CompiledAdamWProgramArtifactInfo> {
        let canonical = serde_json::to_vec(self)
            .map_err(|error| training(format!("compiled program artifact validation: {error}")))?;
        let info = self.info(checksum(&canonical))?;
        validate_module_wire(&self.module, &self.frozen_parameters)?;

        self.validate_policy(&info)?;
        let main = self.validate_main()?;
        self.validate_auxiliary_programs(&main)?;
        self.validate_evaluation(&main.capture)?;
        self.validate_sibling_identities(&info)?;
        self.validate_adamw_policy()?;
        Ok(info)
    }

    fn validate_policy(&self, info: &CompiledAdamWProgramArtifactInfo) -> Result<()> {
        if info.capture_identity == 0
            || self.gradient_accumulation_steps == 0
            || (self.gradient_accumulation_steps > 1) != self.accumulation.is_some()
            || (self.gradient_accumulation_steps > 1) != self.partial_flush.is_some()
            || (self.gradient_accumulation_steps > 1) != self.zero_grad.is_some()
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

    fn validate_main(&self) -> Result<ValidatedMain> {
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
        let (main_capture, main_states) = decode_phase(&self.main.phase)?;
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
            || self
                .adamw
                .weight_decay_exclusions
                .iter()
                .any(|name| !self.main.parameter_buffers.contains_key(name))
        {
            return Err(training(
                "compiled program artifact parameter policy differs",
            ));
        }
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
        if main_states != expected_main_states
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
            self.main.state_input_buffers.get(input) != expected_main_states.get(key)
        }) || self.main.state_input_buffers.len() != input_keys.len()
        {
            return Err(training(
                "compiled program artifact state input mapping differs",
            ));
        }
        if self.host_token_inputs.iter().any(|(name, shape)| {
            self.main.inputs.get(name) != Some(&(shape.clone(), DType::I32))
                || shape.rank() != 2
                || shape.dims().contains(&0)
        }) {
            return Err(training(
                "compiled program artifact host-token policy differs",
            ));
        }
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
        if let Some(policy) = &self.token_weight_policy {
            let policy = CompiledTokenWeightPolicy::from(policy);
            validate_token_weighted_accumulation(
                &self.main.inputs,
                &policy,
                self.gradient_accumulation_steps,
            )?;
        }
        validate_native_manifests(
            &self.main.phase,
            &main_capture,
            &self.main.parameter_buffers,
            &expected_main_states,
        )?;
        Ok(ValidatedMain {
            capture: main_capture,
            states: expected_main_states,
            flush_states: expected_flush_states,
            zero_grad_states: expected_zero_grad_states,
        })
    }

    fn validate_auxiliary_programs(&self, main: &ValidatedMain) -> Result<()> {
        if let Some(phase) = &self.accumulation {
            if phase.clip_report || phase.window_loss_report {
                return Err(training("compiled accumulation artifact exposes reports"));
            }
            let (capture, states) = decode_phase(phase)?;
            if states != main.states
                || capture.schedule.requested.len() != 1 + self.main.output_names.len()
                || phase_external_inputs(&capture, phase).ne(self.main.inputs.keys().cloned())
            {
                return Err(training(
                    "compiled accumulation artifact frontier differs from main",
                ));
            }
            validate_native_manifests(phase, &capture, &BTreeMap::new(), &BTreeMap::new())?;
        }
        if let Some(phase) = &self.zero_grad {
            if phase.clip_report
                || phase.window_loss_report
                || !phase.adamw_native_updates.is_empty()
            {
                return Err(training(
                    "compiled zero-grad artifact exposes update outputs",
                ));
            }
            let (capture, states) = decode_phase(phase)?;
            if states != main.zero_grad_states
                || !capture.schedule.requested.is_empty()
                || phase_external_inputs(&capture, phase).next().is_some()
            {
                return Err(training("compiled zero-grad artifact state schema differs"));
            }
            validate_native_manifests(phase, &capture, &BTreeMap::new(), &BTreeMap::new())?;
        }
        if let Some(phase) = &self.partial_flush {
            if phase.clip_report != self.clip_report
                || phase.window_loss_report != self.window_loss_report
            {
                return Err(training("compiled flush artifact report policy differs"));
            }
            let (capture, states) = decode_phase(phase)?;
            let expected_outputs =
                usize::from(self.clip_report) * 2 + usize::from(self.window_loss_report) * 2;
            let has_learning_rate = capture
                .schedule
                .inputs
                .iter()
                .any(|input| input.name == LEARNING_RATE_INPUT);
            if states != main.flush_states
                || capture.schedule.requested.len() != expected_outputs
                || phase_external_inputs(&capture, phase).next().is_some()
                || has_learning_rate != matches!(&self.learning_rate, LearningRateWire::External)
            {
                return Err(training("compiled flush artifact state schema differs"));
            }
            validate_native_manifests(phase, &capture, &self.main.parameter_buffers, &states)?;
        }
        Ok(())
    }

    fn validate_evaluation(&self, main_capture: &CapturedMixedSchedule) -> Result<()> {
        if let Some(evaluation) = &self.evaluation {
            let capture =
                CapturedSchedule::from_bytes(&evaluation.capture).map_err(replay_error)?;
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

    fn validate_sibling_identities(&self, info: &CompiledAdamWProgramArtifactInfo) -> Result<()> {
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

    fn validate_adamw_policy(&self) -> Result<()> {
        let mut adamw = CompiledAdamWConfig::new(
            f32::from_bits(self.adamw.beta1_bits),
            f32::from_bits(self.adamw.beta2_bits),
            f32::from_bits(self.adamw.eps_bits),
            f32::from_bits(self.adamw.weight_decay_bits),
        )?;
        adamw = adamw
            .with_weight_decay_exclusions(self.adamw.weight_decay_exclusions.iter().cloned())?;
        if adamw.weight_decay_exclusions != self.adamw.weight_decay_exclusions {
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

fn validate_native_manifests(
    phase: &PhaseWire,
    capture: &CapturedMixedSchedule,
    parameters: &BTreeMap<String, u64>,
    states: &BTreeMap<RecurrentStateKey, u64>,
) -> Result<()> {
    if parameters.is_empty() {
        if !phase.adamw_native_updates.is_empty() {
            return Err(training(
                "compiled program artifact has unexpected native updates",
            ));
        }
        return Ok(());
    }
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

fn decode_key_map(map: &BTreeMap<String, u64>) -> Result<BTreeMap<RecurrentStateKey, u64>> {
    map.iter()
        .map(|(key, buffer)| Ok((RecurrentStateKey::from_canonical(key)?, *buffer)))
        .collect()
}

fn decode_input_key_map(
    map: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, RecurrentStateKey>> {
    map.iter()
        .map(|(input, key)| Ok((input.clone(), RecurrentStateKey::from_canonical(key)?)))
        .collect()
}

impl From<&CompiledTokenWeightPolicy> for TokenWeightWire {
    fn from(policy: &CompiledTokenWeightPolicy) -> Self {
        match policy {
            CompiledTokenWeightPolicy::ExplicitMask(name) => Self::ExplicitMask(name.clone()),
            CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            } => Self::IgnoreIndex {
                target_input: target_input.clone(),
                value: *value,
            },
        }
    }
}

impl From<&TokenWeightWire> for CompiledTokenWeightPolicy {
    fn from(policy: &TokenWeightWire) -> Self {
        match policy {
            TokenWeightWire::ExplicitMask(name) => Self::ExplicitMask(name.clone()),
            TokenWeightWire::IgnoreIndex {
                target_input,
                value,
            } => Self::IgnoreIndex {
                target_input: target_input.clone(),
                value: *value,
            },
        }
    }
}

fn manifest_wire(manifest: &crate::engine::AdamWNativeUpdateManifest) -> AdamWManifestWire {
    AdamWManifestWire {
        members: manifest.members.map(|member| AdamWMemberWire {
            role: match member.role {
                crate::engine::AdamWNativeUpdateRole::Parameter => 0,
                crate::engine::AdamWNativeUpdateRole::FirstMoment => 1,
                crate::engine::AdamWNativeUpdateRole::SecondMoment => 2,
                crate::engine::AdamWNativeUpdateRole::GradientAccumulator => 3,
            },
            output: member.output,
            state_buffer: member.state_buffer,
        }),
    }
}

fn decode_manifest(wire: &AdamWManifestWire) -> Result<crate::engine::AdamWNativeUpdateManifest> {
    let members = wire.members.map(|member| {
        let role = match member.role {
            0 => crate::engine::AdamWNativeUpdateRole::Parameter,
            1 => crate::engine::AdamWNativeUpdateRole::FirstMoment,
            2 => crate::engine::AdamWNativeUpdateRole::SecondMoment,
            3 => crate::engine::AdamWNativeUpdateRole::GradientAccumulator,
            _ => crate::engine::AdamWNativeUpdateRole::Parameter,
        };
        crate::engine::AdamWNativeUpdateSuccessor {
            role,
            output: member.output,
            state_buffer: member.state_buffer,
        }
    });
    if wire.members.iter().any(|member| member.role > 3) {
        return Err(training(
            "compiled program artifact native AdamW role is invalid",
        ));
    }
    Ok(crate::engine::AdamWNativeUpdateManifest { members })
}

fn phase_wire(
    capture: &CapturedMixedSchedule,
    state_buffers: &BTreeMap<RecurrentStateKey, u64>,
    state_input_keys: &BTreeMap<String, RecurrentStateKey>,
    native_updates: &[crate::engine::AdamWNativeUpdateManifest],
    clip_report: bool,
    window_loss_report: bool,
) -> Result<PhaseWire> {
    Ok(PhaseWire {
        capture: capture.to_bytes().map_err(replay_error)?,
        state_buffers: key_map(state_buffers),
        state_input_keys: input_key_map(state_input_keys),
        adamw_native_updates: native_updates.iter().map(manifest_wire).collect(),
        clip_report,
        window_loss_report,
    })
}

fn auxiliary_wire(plan: &CompiledAdamWAuxiliaryPlan) -> Result<PhaseWire> {
    phase_wire(
        &plan.capture,
        &plan.state_buffers,
        &plan.state_input_keys,
        &plan.adamw_native_updates,
        plan.clip_report,
        plan.window_loss_report,
    )
}

fn module_wire(seal: &CompiledModuleSeal) -> ModuleWire {
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

fn program_wire<M>(owner: &CompiledModuleAdamWPlan<M>) -> Result<ProgramWire> {
    let plan = &owner.plan;
    let main = &plan.inner;
    let main_state_buffers = main
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
        .collect();
    let accumulation = main
        .accumulation
        .as_ref()
        .map(|phase| {
            phase_wire(
                &phase.capture,
                &phase.state_buffers,
                &main.state_input_keys,
                &[],
                false,
                false,
            )
        })
        .transpose()?;
    let evaluation = plan
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
                allow_zero_valid_token_microbatches: evaluation.allow_zero_valid_token_microbatches,
                capture_identity: evaluation.capture_identity,
            })
        })
        .transpose()?;
    Ok(ProgramWire {
        module: module_wire(&owner.seal),
        main: MainWire {
            phase: phase_wire(
                &main.capture,
                &main_state_buffers,
                &main.state_input_keys,
                &main.adamw_native_updates,
                main.clip_report,
                main.window_loss_report,
            )?,
            inputs: main.inputs.clone(),
            output_names: main.output_names.clone(),
            parameter_buffers: main.parameter_buffers.clone(),
            optimizer_buffers: key_map(&main.optimizer_buffers),
            workload_buffers: key_map(&main.workload_buffers),
            state_input_buffers: main.state_input_buffers.clone(),
            state_input_keys: input_key_map(&main.state_input_keys),
        },
        accumulation,
        partial_flush: plan
            .partial_flush
            .as_ref()
            .map(auxiliary_wire)
            .transpose()?,
        zero_grad: plan.zero_grad.as_ref().map(auxiliary_wire).transpose()?,
        evaluation,
        gradient_accumulation_steps: plan.gradient_accumulation_steps,
        token_weight_policy: plan.token_weight_policy.as_ref().map(Into::into),
        allow_zero_valid_token_microbatches: plan.allow_zero_valid_token_microbatches,
        max_gradient_norm_bits: plan.max_gradient_norm.map(f32::to_bits),
        clip_report: plan.clip_report,
        window_loss_report: plan.window_loss_report,
        loss_scale_bits: plan.loss_scale.to_bits(),
        dropout: plan.dropout.map(|dropout| DropoutWire {
            key: dropout.config.key().words(),
            blocks_per_replay: dropout.blocks_per_replay,
        }),
        host_token_inputs: plan.host_token_inputs.clone(),
        frozen_parameters: plan.frozen_parameters.clone(),
        learning_rate: match &plan.learning_rate {
            CompiledLearningRatePolicy::External => LearningRateWire::External,
            CompiledLearningRatePolicy::MultiStep(schedule) => LearningRateWire::MultiStep {
                base_bits: schedule.base.to_bits(),
                gamma_bits: schedule.gamma.to_bits(),
                milestones: schedule.milestones.clone(),
            },
        },
        adamw: AdamWPolicyWire {
            beta1_bits: plan.adamw_policy.beta1.to_bits(),
            beta2_bits: plan.adamw_policy.beta2.to_bits(),
            eps_bits: plan.adamw_policy.eps.to_bits(),
            weight_decay_bits: plan.adamw_policy.weight_decay.to_bits(),
            weight_decay_exclusions: plan.adamw_policy.weight_decay_exclusions.clone(),
        },
    })
}

impl<M: Module> CompiledModuleAdamWPlan<M> {
    /// Serializes the resource-free CPU executable plan separately from tensor state.
    pub fn program_artifact(&self) -> Result<CompiledAdamWProgramArtifact> {
        self.validate_ready_for_preparation()?;
        let wire = program_wire(self)?;
        let bytes = encode(&wire)?;
        CompiledAdamWProgramArtifact::from_bytes(bytes)
    }

    /// Restores a differently initialized owned module from one executable
    /// artifact and its separately persisted complete-module checkpoint.
    /// Decoding, topology authentication, and frontier restoration all finish
    /// before a runtime is prepared or the destination module can be published.
    pub fn restore_from_program_artifact(
        module: M,
        artifact: &CompiledAdamWProgramArtifact,
        checkpoint: &CompiledModuleAdamWCheckpoint,
    ) -> std::result::Result<Self, CompiledModuleAdamWArtifactRestoreError<M>> {
        let result = restore_owner(&module, artifact, checkpoint);
        match result {
            Ok((plan, seal)) => Ok(Self {
                module,
                plan,
                seal,
                required_evaluation_capture_identity: None,
            }),
            Err(source) => Err(CompiledModuleAdamWArtifactRestoreError { module, source }),
        }
    }
}

fn decode_phase(
    wire: &PhaseWire,
) -> Result<(CapturedMixedSchedule, BTreeMap<RecurrentStateKey, u64>)> {
    let capture = CapturedMixedSchedule::from_bytes(&wire.capture).map_err(replay_error)?;
    let state_buffers = decode_key_map(&wire.state_buffers)?;
    let frontier = capture
        .initial_recurrent_cursor()
        .map_err(replay_error)?
        .frontier()
        .iter()
        .map(|state| state.buffer)
        .collect::<BTreeSet<_>>();
    if state_buffers.len() != frontier.len()
        || state_buffers.values().copied().collect::<BTreeSet<_>>() != frontier
    {
        return Err(training(
            "compiled program artifact recurrent frontier differs",
        ));
    }
    let input_names = capture
        .schedule
        .inputs
        .iter()
        .map(|input| (input.node, input.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let captured_inputs = capture.state_bindings.iter().try_fold(
        BTreeMap::<String, u64>::new(),
        |mut inputs, binding| {
            let name = input_names.get(&binding.input_node).ok_or_else(|| {
                training("compiled program artifact recurrent input owner is absent")
            })?;
            if let Some(previous) = inputs.insert((*name).to_owned(), binding.state.buffer)
                && previous != binding.state.buffer
            {
                return Err(training(
                    "compiled program artifact recurrent input aliases buffers",
                ));
            }
            Ok(inputs)
        },
    )?;
    let input_keys = decode_input_key_map(&wire.state_input_keys)?;
    if captured_inputs.keys().ne(input_keys.keys())
        || input_keys
            .iter()
            .any(|(name, key)| state_buffers.get(key) != captured_inputs.get(name))
    {
        return Err(training(
            "compiled program artifact recurrent input schema differs",
        ));
    }
    Ok((capture, state_buffers))
}

fn decode_auxiliary(wire: &PhaseWire) -> Result<CompiledAdamWAuxiliaryPlan> {
    let (capture, state_buffers) = decode_phase(wire)?;
    let capture_identity = capture
        .initial_recurrent_cursor()
        .map_err(replay_error)?
        .capture_identity();
    let recurrent_capture = CompiledRecurrentCapture::from_artifact(&capture)?;
    Ok(CompiledAdamWAuxiliaryPlan {
        capture,
        recurrent_capture,
        state_buffers,
        state_input_keys: decode_input_key_map(&wire.state_input_keys)?,
        adamw_native_updates: wire
            .adamw_native_updates
            .iter()
            .map(decode_manifest)
            .collect::<Result<_>>()?,
        capture_identity,
        clip_report: wire.clip_report,
        window_loss_report: wire.window_loss_report,
    })
}

fn checkpoint_module_wire(decoded: &DecodedModuleAdamWCheckpoint) -> Result<ModuleWire> {
    let optimizer = decode_adamw_checkpoint(decoded.optimizer.as_bytes())?;
    let states = decoded
        .states
        .iter()
        .map(|state| {
            let value = state
                .value
                .as_ref()
                .or_else(|| optimizer.parameters.get(&state.name))
                .ok_or_else(|| training("compiled module checkpoint state value is absent"))?;
            Ok(ModuleStateWire {
                name: state.name.clone(),
                kind: match state.kind {
                    ModuleCheckpointStateKind::Parameter => "parameter",
                    ModuleCheckpointStateKind::Buffer => "buffer",
                }
                .into(),
                source_trainable: state.source_trainable,
                policy_frozen: state.policy_frozen,
                shape: value.shape().clone(),
                dtype: value.dtype(),
            })
        })
        .collect::<Result<_>>()?;
    Ok(ModuleWire {
        states,
        visits: decoded
            .visits
            .iter()
            .map(|visit| ModuleVisitWire {
                name: visit.name.clone(),
                canonical_name: visit.canonical_name.clone(),
            })
            .collect(),
    })
}

fn zero_frontier(
    capture: &CapturedMixedSchedule,
    state_buffers: &BTreeMap<RecurrentStateKey, u64>,
) -> Result<(
    BTreeMap<RecurrentStateKey, TensorData>,
    BTreeMap<RecurrentStateKey, u64>,
)> {
    let cursor = capture.initial_recurrent_cursor().map_err(replay_error)?;
    let descriptors = cursor
        .frontier()
        .iter()
        .map(|state| (state.buffer, state))
        .collect::<BTreeMap<_, _>>();
    let values = state_buffers
        .iter()
        .map(|(key, buffer)| {
            let state = descriptors
                .get(buffer)
                .ok_or_else(|| training("compiled artifact state descriptor is absent"))?;
            Ok((
                key.clone(),
                TensorData::zeros_with_dtype(state.shape.clone(), state.dtype)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let versions = values.keys().cloned().map(|key| (key, 0)).collect();
    Ok((values, versions))
}

fn restore_owner<M: Module>(
    module: &M,
    artifact: &CompiledAdamWProgramArtifact,
    checkpoint: &CompiledModuleAdamWCheckpoint,
) -> Result<(CompiledAdamWPlan, CompiledModuleSeal)> {
    let wire = decode(artifact.as_bytes())?;
    if wire.validate()? != *artifact.info() {
        return Err(training("compiled program artifact info differs"));
    }
    let decoded_module = decode_module_adamw_checkpoint(checkpoint.as_bytes())?;
    if checkpoint_module_wire(&decoded_module)? != wire.module {
        return Err(training("compiled program artifact module schema mismatch"));
    }
    let mut seal = CompiledModuleSeal::capture(module, &wire.frozen_parameters)?;
    let _immutable_values = seal.apply_module_checkpoint(&decoded_module)?;
    if module_wire(&seal) != wire.module {
        return Err(training(
            "compiled program artifact destination module mismatch",
        ));
    }

    let (capture, main_state_buffers) = decode_phase(&wire.main.phase)?;
    let parameter_buffers = wire.main.parameter_buffers.clone();
    let optimizer_buffers = decode_key_map(&wire.main.optimizer_buffers)?;
    let workload_buffers = decode_key_map(&wire.main.workload_buffers)?;
    let expected_state_buffers = parameter_buffers
        .iter()
        .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
        .chain(
            optimizer_buffers
                .iter()
                .map(|(key, buffer)| (key.clone(), *buffer)),
        )
        .chain(
            workload_buffers
                .iter()
                .map(|(key, buffer)| (key.clone(), *buffer)),
        )
        .collect::<BTreeMap<_, _>>();
    if expected_state_buffers != main_state_buffers {
        return Err(training("compiled program artifact state map mismatch"));
    }
    let (state_values, state_versions) = zero_frontier(&capture, &main_state_buffers)?;
    let state_input_keys = decode_input_key_map(&wire.main.state_input_keys)?;
    let recurrent_capture = CompiledRecurrentCapture::from_artifact(&capture)?;
    let accumulation = wire
        .accumulation
        .as_ref()
        .map(|phase| {
            let (capture, state_buffers) = decode_phase(phase)?;
            let capture_identity = capture
                .initial_recurrent_cursor()
                .map_err(replay_error)?
                .capture_identity();
            Ok(CompiledAdamWAccumulationPlan {
                recurrent_capture: CompiledRecurrentCapture::from_artifact(&capture)?,
                capture,
                state_buffers,
                capture_identity,
            })
        })
        .transpose()?;
    let inner = CompiledTrainingPlan {
        capture,
        recurrent_capture,
        inputs: wire.main.inputs.clone(),
        output_names: wire.main.output_names.clone(),
        clip_report: wire.main.phase.clip_report,
        window_loss_report: wire.main.phase.window_loss_report,
        parameter_buffers,
        optimizer_buffers,
        workload_buffers,
        state_input_buffers: wire.main.state_input_buffers.clone(),
        state_input_keys,
        state_values,
        state_versions,
        adamw_native_updates: wire
            .main
            .phase
            .adamw_native_updates
            .iter()
            .map(decode_manifest)
            .collect::<Result<_>>()?,
        frozen_parameter_nodes: BTreeSet::new(),
        step: 0,
        accumulation,
    };
    let evaluation = wire
        .evaluation
        .as_ref()
        .map(|evaluation| {
            let capture =
                CapturedSchedule::from_bytes(&evaluation.capture).map_err(replay_error)?;
            if capture.identity != evaluation.capture_identity {
                return Err(training("compiled evaluation artifact identity mismatch"));
            }
            Ok(CompiledEvaluationPlan {
                inference: CompiledEvaluationCapture::from_artifact(capture)?,
                inputs: evaluation.inputs.clone(),
                output_names: evaluation.output_names.clone(),
                parameter_inputs: evaluation.parameter_inputs.clone(),
                loss_weight_policy: evaluation.loss_weight_policy.as_ref().map(Into::into),
                allow_zero_valid_token_microbatches: evaluation.allow_zero_valid_token_microbatches,
                capture_identity: evaluation.capture_identity,
            })
        })
        .transpose()?;
    if decoded_module.evaluation_capture_identity
        != evaluation
            .as_ref()
            .map(|evaluation| evaluation.capture_identity)
    {
        return Err(training(
            "compiled program artifact evaluation identity mismatch",
        ));
    }
    let learning_rate = match &wire.learning_rate {
        LearningRateWire::External => CompiledLearningRatePolicy::External,
        LearningRateWire::MultiStep {
            base_bits,
            gamma_bits,
            milestones,
        } => CompiledLearningRatePolicy::MultiStep(CompiledMultiStepLr::new(
            f32::from_bits(*base_bits),
            f32::from_bits(*gamma_bits),
            milestones.clone(),
        )?),
    };
    let program_identity = inner.capture_identity()?;
    let plan = CompiledAdamWPlan {
        program_identity,
        inner,
        partial_flush: wire
            .partial_flush
            .as_ref()
            .map(decode_auxiliary)
            .transpose()?,
        zero_grad: wire.zero_grad.as_ref().map(decode_auxiliary).transpose()?,
        gradient_accumulation_steps: wire.gradient_accumulation_steps,
        token_weight_policy: wire.token_weight_policy.as_ref().map(Into::into),
        allow_zero_valid_token_microbatches: wire.allow_zero_valid_token_microbatches,
        max_gradient_norm: wire.max_gradient_norm_bits.map(f32::from_bits),
        clip_report: wire.clip_report,
        window_loss_report: wire.window_loss_report,
        loss_scale: f32::from_bits(wire.loss_scale_bits),
        progress: AdamWProgress::INITIAL,
        dropout: wire.dropout.map(|dropout| CompiledDropoutState {
            config: CompiledDropoutConfig::new(CompiledDropoutKey(dropout.key)),
            blocks_per_replay: dropout.blocks_per_replay,
        }),
        host_token_inputs: wire.host_token_inputs.clone(),
        frozen_parameters: wire.frozen_parameters.clone(),
        evaluation,
        learning_rate,
        adamw_policy: CompiledAdamWPolicy {
            beta1: f32::from_bits(wire.adamw.beta1_bits),
            beta2: f32::from_bits(wire.adamw.beta2_bits),
            eps: f32::from_bits(wire.adamw.eps_bits),
            weight_decay: f32::from_bits(wire.adamw.weight_decay_bits),
            weight_decay_exclusions: wire.adamw.weight_decay_exclusions.clone(),
        },
    }
    .restore_checkpoint(checkpoint.optimizer_checkpoint())?;
    seal.validate_unchanged(module)?;
    Ok((plan, seal))
}
