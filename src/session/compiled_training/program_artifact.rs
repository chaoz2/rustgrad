use super::adamw_contract::{CompiledAdamWContract, CompiledAdamWPolicy};
use super::module_adamw_checkpoint::{DecodedModuleAdamWCheckpoint, ModuleCheckpointStateKind};
use super::*;
use crate::file_io::{ExactFileError, read_file_bytes_bounded, replace_file_bytes_atomically};
use serde::{Deserialize, Serialize};
use std::{
    io,
    path::Path,
    sync::{Arc, OnceLock},
};

mod wire_validation;

const MAGIC: &[u8; 4] = b"RGAP";
const LEGACY_FORMAT_VERSION: u8 = 1;
const FORMAT_VERSION: u8 = 2;
const MAX_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct PortableResumeDecodeCounts {
    pub(super) program_wire: usize,
    pub(super) mixed_captures: usize,
    pub(super) evaluation_captures: usize,
    pub(super) module_checkpoints: usize,
    pub(super) pair_admissions: usize,
    pub(super) topology_seals: usize,
    pub(super) topology_phase_validations: usize,
    pub(super) recurrent_execution_plans: usize,
    pub(super) evaluation_execution_plans: usize,
    pub(super) cursor_projections: usize,
}

#[cfg(test)]
thread_local! {
    static PORTABLE_RESUME_DECODE_COUNTS: std::cell::Cell<PortableResumeDecodeCounts> =
        std::cell::Cell::new(PortableResumeDecodeCounts::default());
}

#[cfg(test)]
fn update_decode_counts(update: impl FnOnce(&mut PortableResumeDecodeCounts)) {
    PORTABLE_RESUME_DECODE_COUNTS.with(|counts| {
        let mut next = counts.get();
        update(&mut next);
        counts.set(next);
    });
}

#[cfg(test)]
pub(super) fn portable_resume_decode_counts() -> PortableResumeDecodeCounts {
    PORTABLE_RESUME_DECODE_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn record_module_checkpoint_decode() {
    update_decode_counts(|counts| counts.module_checkpoints += 1);
}

#[cfg(test)]
pub(super) fn record_recurrent_execution_plan() {
    update_decode_counts(|counts| counts.recurrent_execution_plans += 1);
}

#[cfg(test)]
pub(super) fn record_evaluation_execution_plan() {
    update_decode_counts(|counts| counts.evaluation_execution_plans += 1);
}

#[derive(Clone)]
pub struct CompiledAdamWProgramArtifact {
    bytes: Vec<u8>,
    info: CompiledAdamWProgramArtifactInfo,
    admitted: Option<Arc<AdmittedProgramArtifact>>,
}

impl fmt::Debug for CompiledAdamWProgramArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledAdamWProgramArtifact")
            .field("bytes", &self.bytes)
            .field("info", &self.info)
            .finish()
    }
}

impl PartialEq for CompiledAdamWProgramArtifact {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.info == other.info
    }
}

impl Eq for CompiledAdamWProgramArtifact {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWProgramArtifactInfo {
    format_version: u8,
    identity: u64,
    capture_identity: u64,
    accumulation_capture_identity: Option<u64>,
    flush_capture_identity: Option<u64>,
    zero_grad_capture_identity: Option<u64>,
    evaluation_capture_identity: Option<u64>,
}

/// A local compiled-program artifact file failure.
#[derive(Debug)]
pub enum CompiledAdamWProgramArtifactFileError {
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
    Limit {
        actual: u64,
        maximum: usize,
    },
    Format(Error),
}

impl fmt::Display for CompiledAdamWProgramArtifactFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => {
                write!(
                    formatter,
                    "compiled program artifact file {operation} failed: {kind:?}"
                )
            }
            Self::Limit { actual, maximum } => write!(
                formatter,
                "compiled program artifact file has {actual} bytes, exceeding byte limit {maximum}"
            ),
            Self::Format(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompiledAdamWProgramArtifactFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Io { .. } | Self::Limit { .. } => None,
        }
    }
}

fn artifact_file_error(error: ExactFileError) -> CompiledAdamWProgramArtifactFileError {
    match error {
        ExactFileError::Io { operation, source } => CompiledAdamWProgramArtifactFileError::Io {
            operation,
            kind: source.kind(),
        },
        ExactFileError::Limit { actual, maximum } => {
            CompiledAdamWProgramArtifactFileError::Limit { actual, maximum }
        }
        ExactFileError::Allocation => CompiledAdamWProgramArtifactFileError::Io {
            operation: "allocate read buffer",
            kind: io::ErrorKind::OutOfMemory,
        },
        ExactFileError::InvalidFileName => CompiledAdamWProgramArtifactFileError::Io {
            operation: "validate path",
            kind: io::ErrorKind::InvalidInput,
        },
        ExactFileError::StagingExhausted => CompiledAdamWProgramArtifactFileError::Io {
            operation: "create unique staging file",
            kind: io::ErrorKind::AlreadyExists,
        },
    }
}

impl CompiledAdamWProgramArtifactInfo {
    pub fn format_version(&self) -> u8 {
        self.format_version
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
        let mut wire = decode(&bytes)?;
        let (info, captures) = wire.validate()?;
        wire.discard_capture_bytes();
        Ok(Self {
            bytes,
            info,
            admitted: Some(Arc::new(AdmittedProgramArtifact::new(wire, captures))),
        })
    }

    /// Loads and validates one local RGAP artifact under its intrinsic byte bound.
    pub fn load_file(
        path: impl AsRef<Path>,
    ) -> std::result::Result<Self, CompiledAdamWProgramArtifactFileError> {
        Self::load_file_with_byte_limit(path, MAX_ARTIFACT_BYTES)
    }

    /// Loads and validates one local RGAP artifact under the smaller of the
    /// caller's byte limit and the format's intrinsic byte bound.
    pub fn load_file_with_byte_limit(
        path: impl AsRef<Path>,
        maximum: usize,
    ) -> std::result::Result<Self, CompiledAdamWProgramArtifactFileError> {
        let bytes = read_file_bytes_bounded(path, maximum.min(MAX_ARTIFACT_BYTES))
            .map_err(artifact_file_error)?;
        Self::from_bytes(bytes).map_err(CompiledAdamWProgramArtifactFileError::Format)
    }

    /// Atomically replaces `path` with these exact validated artifact bytes
    /// after syncing a uniquely created same-directory staging file.
    pub fn save_file(
        &self,
        path: impl AsRef<Path>,
    ) -> std::result::Result<(), CompiledAdamWProgramArtifactFileError> {
        let wire =
            decode(self.as_bytes()).map_err(CompiledAdamWProgramArtifactFileError::Format)?;
        let (info, _) = wire
            .validate()
            .map_err(CompiledAdamWProgramArtifactFileError::Format)?;
        if info != self.info {
            return Err(CompiledAdamWProgramArtifactFileError::Format(training(
                "compiled program artifact info differs",
            )));
        }
        replace_file_bytes_atomically(path, self.as_bytes()).map_err(artifact_file_error)
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

    fn admitted(&self) -> Option<&Arc<AdmittedProgramArtifact>> {
        self.admitted.as_ref()
    }

    #[cfg(test)]
    pub(super) fn retained_capture_extents(&self) -> Vec<(usize, usize)> {
        let Some(admitted) = &self.admitted else {
            return Vec::new();
        };
        let mut extents = vec![(
            admitted.wire.main.phase.capture.len(),
            admitted.wire.main.phase.capture.capacity(),
        )];
        extents.extend(
            [
                admitted.wire.accumulation.as_ref(),
                admitted.wire.partial_flush.as_ref(),
                admitted.wire.zero_grad.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(|phase| (phase.capture.len(), phase.capture.capacity())),
        );
        extents.extend(
            admitted
                .wire
                .evaluation
                .as_ref()
                .map(|evaluation| (evaluation.capture.len(), evaluation.capture.capacity())),
        );
        extents
    }

    #[cfg(test)]
    pub(super) fn retained_metal_recipe_extents(&self) -> Vec<(usize, usize)> {
        let Some(metal) = self
            .admitted
            .as_ref()
            .and_then(|admitted| admitted.wire.metal.as_ref())
        else {
            return Vec::new();
        };
        std::iter::once((metal.main.len(), metal.main.capacity()))
            .chain(
                [metal.partial_flush.as_ref(), metal.evaluation.as_ref()]
                    .into_iter()
                    .flatten()
                    .map(|recipe| (recipe.len(), recipe.capacity())),
            )
            .collect()
    }

    #[cfg(test)]
    pub(super) fn shares_admission_with(&self, other: &Self) -> bool {
        self.admitted
            .as_ref()
            .zip(other.admitted.as_ref())
            .is_some_and(|(left, right)| Arc::ptr_eq(left, right))
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
    #[serde(skip)]
    format_version: u8,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metal: Option<MetalProgramWire>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MetalProgramWire {
    main: Vec<u8>,
    partial_flush: Option<Vec<u8>>,
    evaluation: Option<Vec<u8>>,
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
    states: BTreeMap<RecurrentStateKey, u64>,
    flush_states: BTreeMap<RecurrentStateKey, u64>,
    zero_grad_states: BTreeMap<RecurrentStateKey, u64>,
}

#[derive(Clone, Debug)]
struct ProgramCaptures {
    main: Arc<CapturedMixedSchedule>,
    accumulation: Option<Arc<CapturedMixedSchedule>>,
    partial_flush: Option<Arc<CapturedMixedSchedule>>,
    zero_grad: Option<Arc<CapturedMixedSchedule>>,
    evaluation: Option<Arc<CapturedSchedule>>,
    metal_main: Option<PortableCapturedInferenceRecipe>,
    metal_partial_flush: Option<PortableCapturedInferenceRecipe>,
    metal_evaluation: Option<PortableCapturedInferenceRecipe>,
    metal_main_recurrent: Option<CompiledRecurrentCapture>,
    metal_partial_flush_recurrent: Option<CompiledRecurrentCapture>,
    metal_evaluation_capture: Option<CompiledEvaluationCapture>,
}

struct AdmittedProgramArtifact {
    wire: ProgramWire,
    captures: ProgramCaptures,
    training_topology: OnceLock<Result<Arc<AdmittedTrainingTopology>>>,
}

/// Immutable resource-free replay structure derived at most once per admitted
/// artifact. Mutable tensor values, logical versions, module seals, backend
/// resources, and replay cursors remain destination-owned.
struct AdmittedTrainingTopology {
    plan: CompiledAdamWPlan,
}

impl fmt::Debug for AdmittedProgramArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmittedProgramArtifact")
            .field("wire", &self.wire)
            .field("captures", &self.captures)
            .field(
                "training_topology_sealed",
                &self.training_topology.get().is_some(),
            )
            .finish()
    }
}

impl AdmittedProgramArtifact {
    fn new(wire: ProgramWire, captures: ProgramCaptures) -> Self {
        Self {
            wire,
            captures,
            training_topology: OnceLock::new(),
        }
    }

    fn training_topology(
        &self,
        checkpoint: &DecodedModuleAdamWCheckpoint,
    ) -> Result<Arc<AdmittedTrainingTopology>> {
        self.training_topology
            .get_or_init(|| {
                #[cfg(test)]
                update_decode_counts(|counts| counts.topology_seals += 1);
                seal_admitted_training_topology(&self.wire, &self.captures, checkpoint)
                    .map(Arc::new)
            })
            .clone()
    }
}

impl ProgramCaptures {
    fn decode_phase(phase: &PhaseWire) -> Result<Arc<CapturedMixedSchedule>> {
        #[cfg(test)]
        update_decode_counts(|counts| counts.mixed_captures += 1);
        let capture = CapturedMixedSchedule::from_bytes(&phase.capture).map_err(replay_error)?;
        capture.initial_recurrent_cursor().map_err(replay_error)?;
        Ok(Arc::new(capture))
    }

    fn decode_evaluation(evaluation: &EvaluationWire) -> Result<Arc<CapturedSchedule>> {
        #[cfg(test)]
        update_decode_counts(|counts| counts.evaluation_captures += 1);
        let capture = CapturedSchedule::from_bytes(&evaluation.capture).map_err(replay_error)?;
        (capture.identity == evaluation.capture_identity)
            .then_some(Arc::new(capture))
            .ok_or_else(|| training("compiled evaluation artifact identity mismatch"))
    }

    fn decode_metal_recipe(bytes: &[u8]) -> Result<PortableCapturedInferenceRecipe> {
        PortableCapturedInferenceRecipe::from_bytes(bytes).map_err(captured_inference_error)
    }

    fn decode(wire: &ProgramWire) -> Result<Self> {
        let main = Self::decode_phase(&wire.main.phase)?;
        let accumulation = wire
            .accumulation
            .as_ref()
            .map(Self::decode_phase)
            .transpose()?;
        let partial_flush = wire
            .partial_flush
            .as_ref()
            .map(Self::decode_phase)
            .transpose()?;
        let zero_grad = wire
            .zero_grad
            .as_ref()
            .map(Self::decode_phase)
            .transpose()?;
        let evaluation = wire
            .evaluation
            .as_ref()
            .map(Self::decode_evaluation)
            .transpose()?;
        let metal_main = wire
            .metal
            .as_ref()
            .map(|metal| Self::decode_metal_recipe(&metal.main))
            .transpose()?;
        let metal_partial_flush = wire
            .metal
            .as_ref()
            .and_then(|metal| metal.partial_flush.as_ref())
            .map(|recipe| Self::decode_metal_recipe(recipe))
            .transpose()?;
        let metal_evaluation = wire
            .metal
            .as_ref()
            .and_then(|metal| metal.evaluation.as_ref())
            .map(|recipe| Self::decode_metal_recipe(recipe))
            .transpose()?;
        let metal_main_recurrent = metal_main
            .clone()
            .map(|recipe| CompiledRecurrentCapture::from_artifact(main.as_ref(), Some(recipe)))
            .transpose()?;
        let metal_partial_flush_recurrent = metal_partial_flush
            .clone()
            .zip(partial_flush.as_ref())
            .map(|(recipe, capture)| {
                CompiledRecurrentCapture::from_artifact(capture.as_ref(), Some(recipe))
            })
            .transpose()?;
        let metal_evaluation_capture = metal_evaluation
            .clone()
            .zip(evaluation.as_ref())
            .map(|(recipe, capture)| {
                CompiledEvaluationCapture::from_artifact(capture.clone(), Some(recipe))
            })
            .transpose()?;
        Ok(Self {
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
            metal_main,
            metal_partial_flush,
            metal_evaluation,
            metal_main_recurrent,
            metal_partial_flush_recurrent,
            metal_evaluation_capture,
        })
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn encode(wire: &ProgramWire) -> Result<Vec<u8>> {
    if !matches!(wire.format_version, LEGACY_FORMAT_VERSION | FORMAT_VERSION) {
        return Err(training("compiled program artifact version is unsupported"));
    }
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
    bytes.push(wire.format_version);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<ProgramWire> {
    #[cfg(test)]
    update_decode_counts(|counts| counts.program_wire += 1);
    if bytes.len() < 21 || bytes.len() > MAX_ARTIFACT_BYTES || &bytes[..4] != MAGIC {
        return Err(training("compiled program artifact header is invalid"));
    }
    if !matches!(bytes[4], LEGACY_FORMAT_VERSION | FORMAT_VERSION) {
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
    let mut wire: ProgramWire = serde_json::from_slice(&bytes[13..payload_end])
        .map_err(|error| training(format!("compiled program artifact payload: {error}")))?;
    wire.format_version = bytes[4];
    Ok(wire)
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
    let format_version = wire.format_version;
    let mut json = serde_json::to_value(wire).expect("serializable test artifact");
    rewrite(&mut json);
    let mut wire: ProgramWire =
        serde_json::from_value(json).expect("well-formed rewritten test artifact");
    wire.format_version = format_version;
    let bytes = encode(&wire).expect("bounded rewritten test artifact");
    let unchecked = CompiledAdamWProgramArtifact {
        bytes: bytes.clone(),
        info: *artifact.info(),
        admitted: None,
    };
    (bytes, unchecked)
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
    // RGAP remains an AdamW artifact: adapt the optimizer-neutral runtime
    // group back to its historical fixed role/order wire without serializing
    // engine-only grouping metadata.
    let members = [
        member(0, parameter),
        member(1, first_moment),
        member(2, second_moment),
        member(3, accumulator),
    ];
    Ok(AdamWManifestWire { members })
}

fn decode_manifest(wire: &AdamWManifestWire) -> Result<crate::engine::RecurrentStoreGroupManifest> {
    if wire
        .members
        .iter()
        .enumerate()
        .any(|(role, member)| usize::from(member.role) != role)
    {
        return Err(training(
            "compiled program artifact native AdamW role is invalid",
        ));
    }
    let members = wire
        .members
        .iter()
        .map(|member| crate::engine::RecurrentStoreGroupMember {
            output: member.output,
            state_buffer: member.state_buffer,
        })
        .collect();
    Ok(crate::engine::RecurrentStoreGroupManifest { members })
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
                FORMAT_VERSION
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
            adamw: self.adamw_wire(),
            metal,
        })
    }
}

fn program_wire<M>(owner: &CompiledModuleAdamWPlan<M>) -> Result<ProgramWire> {
    ProgramWireEncoder::new(owner).encode()
}

impl<M: Module> CompiledModuleAdamWPlan<M> {
    /// Serializes the resource-free executable plan separately from tensor state.
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

    /// Restores from one atomically persisted executable-and-state bundle.
    ///
    /// The bundle retains the exact existing program-artifact and checkpoint
    /// formats. Their ordinary restore validator remains authoritative, and all
    /// validation completes before runtime preparation or module publication.
    pub fn restore_from_resume_bundle(
        module: M,
        bundle: &CompiledAdamWResumeBundle,
    ) -> std::result::Result<Self, CompiledModuleAdamWArtifactRestoreError<M>> {
        let result = restore_owner_from_admitted(&module, bundle.admitted());
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

fn validate_phase_capture(
    wire: &PhaseWire,
    capture: &CapturedMixedSchedule,
) -> Result<BTreeMap<RecurrentStateKey, u64>> {
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
    Ok(state_buffers)
}

fn decode_auxiliary(
    main: &CapturedMixedSchedule,
    wire: &PhaseWire,
    capture: Arc<CapturedMixedSchedule>,
    admitted_recurrent: Option<CompiledRecurrentCapture>,
) -> Result<CompiledAdamWAuxiliaryPlan> {
    #[cfg(test)]
    update_decode_counts(|counts| counts.topology_phase_validations += 1);
    let state_buffers = validate_phase_capture(wire, capture.as_ref())?;
    let outputs = CompiledAdamWAuxiliaryOutputSchema::from_report_flags(
        wire.clip_report,
        wire.window_loss_report,
    );
    outputs.validate_report_flags(wire.clip_report, wire.window_loss_report)?;
    let cursor_projection = PreparedRecurrentCursorProjection::prepare(
        main,
        capture.as_ref(),
        state_buffers.values().copied(),
    )
    .map_err(replay_error)?;
    #[cfg(test)]
    update_decode_counts(|counts| counts.cursor_projections += 1);
    let capture_identity = cursor_projection.target_capture_identity();
    let recurrent_capture = match admitted_recurrent {
        Some(recurrent) => recurrent,
        None => CompiledRecurrentCapture::from_artifact(capture.as_ref(), None)?,
    };
    Ok(CompiledAdamWAuxiliaryPlan {
        phase: CompiledRecurrentPhasePlan {
            capture,
            recurrent_capture,
            state_buffers,
            cursor_projection: Arc::new(cursor_projection),
            capture_identity,
            admission: CompiledRecurrentPhaseAdmission::Replace {
                store_groups: wire
                    .adamw_native_updates
                    .iter()
                    .map(decode_manifest)
                    .collect::<Result<_>>()?,
            },
        },
        state_input_keys: decode_input_key_map(&wire.state_input_keys)?,
        outputs,
    })
}

fn checkpoint_module_wire(decoded: &DecodedModuleAdamWCheckpoint) -> Result<ModuleWire> {
    let optimizer = decoded.optimizer.decoded();
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

#[derive(Clone, Debug)]
pub(super) struct AdmittedArtifactCheckpointPair {
    artifact: Arc<AdmittedProgramArtifact>,
    checkpoint: Arc<DecodedModuleAdamWCheckpoint>,
}

fn decode_admitted_artifact_checkpoint_pair(
    artifact: &CompiledAdamWProgramArtifact,
    checkpoint: &CompiledModuleAdamWCheckpoint,
) -> Result<AdmittedArtifactCheckpointPair> {
    #[cfg(test)]
    update_decode_counts(|counts| counts.pair_admissions += 1);
    let admitted = match artifact.admitted() {
        Some(admitted) => admitted.clone(),
        None => {
            let mut wire = decode(artifact.as_bytes())?;
            let (artifact_info, captures) = wire.validate()?;
            if artifact_info != *artifact.info() {
                return Err(training("compiled program artifact info differs"));
            }
            wire.discard_capture_bytes();
            Arc::new(AdmittedProgramArtifact::new(wire, captures))
        }
    };
    let wire = &admitted.wire;
    let artifact_info = *artifact.info();
    let decoded_module = checkpoint.decoded_arc().clone();
    if checkpoint_module_wire(decoded_module.as_ref())? != wire.module {
        return Err(training("compiled program artifact module schema mismatch"));
    }
    if decoded_module.evaluation_capture_identity != artifact_info.evaluation_capture_identity {
        return Err(training(
            "compiled program artifact evaluation identity mismatch",
        ));
    }
    let checkpoint_info = checkpoint.optimizer_checkpoint().info();
    let topology = CompiledTrainingWindowTopology::from_validated_parts(
        wire.gradient_accumulation_steps,
        wire.token_weight_policy.is_some(),
        wire.window_loss_report,
    );
    if checkpoint_info.capture_identity() != artifact_info.capture_identity
        || checkpoint_info.gradient_accumulation_steps() != wire.gradient_accumulation_steps
        || checkpoint_info.window_loss_report_enabled() != wire.window_loss_report
        || topology.retains_token_count() != checkpoint_info.accumulated_token_count().is_some()
        || wire.dropout.is_some() != checkpoint_info.dropout_block_counter().is_some()
    {
        return Err(training(
            "compiled program artifact checkpoint policy mismatch",
        ));
    }
    if checkpoint_info.accumulation_capture_identity().is_some()
        && checkpoint_info.accumulation_capture_identity()
            != artifact_info.accumulation_capture_identity
    {
        return Err(training(
            "compiled program artifact accumulation identity mismatch",
        ));
    }
    if checkpoint_info.flush_capture_identity().is_some()
        && checkpoint_info.flush_capture_identity() != artifact_info.flush_capture_identity
    {
        return Err(training(
            "compiled program artifact partial flush identity mismatch",
        ));
    }
    if checkpoint_info.reset_capture_identity().is_some()
        && checkpoint_info.reset_capture_identity() != artifact_info.zero_grad_capture_identity
    {
        return Err(training(
            "compiled program artifact zero-grad identity mismatch",
        ));
    }
    Ok(AdmittedArtifactCheckpointPair {
        artifact: admitted,
        checkpoint: decoded_module,
    })
}

pub(super) fn admit_artifact_checkpoint_pair(
    artifact: &CompiledAdamWProgramArtifact,
    checkpoint: &CompiledModuleAdamWCheckpoint,
) -> Result<Arc<AdmittedArtifactCheckpointPair>> {
    decode_admitted_artifact_checkpoint_pair(artifact, checkpoint).map(Arc::new)
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
    let admitted = decode_admitted_artifact_checkpoint_pair(artifact, checkpoint)?;
    restore_owner_from_admitted(module, &admitted)
}

fn restore_owner_from_admitted<M: Module>(
    module: &M,
    admitted: &AdmittedArtifactCheckpointPair,
) -> Result<(CompiledAdamWPlan, CompiledModuleSeal)> {
    let wire = &admitted.artifact.wire;
    let decoded_module = admitted.checkpoint.as_ref();
    let mut seal = CompiledModuleSeal::capture(module, &wire.frozen_parameters)?;
    let _immutable_values = seal.apply_module_checkpoint(decoded_module)?;
    if module_wire(&seal) != wire.module {
        return Err(training(
            "compiled program artifact destination module mismatch",
        ));
    }
    let plan = admitted
        .artifact
        .training_topology(decoded_module)?
        .plan
        .clone()
        .restore_checkpoint_owned(&decoded_module.optimizer)?;
    seal.validate_unchanged(module)?;
    Ok((plan, seal))
}

fn seal_admitted_training_topology(
    wire: &ProgramWire,
    captures: &ProgramCaptures,
    decoded_module: &DecodedModuleAdamWCheckpoint,
) -> Result<AdmittedTrainingTopology> {
    let capture = captures.main.clone();
    #[cfg(test)]
    update_decode_counts(|counts| counts.topology_phase_validations += 1);
    let main_state_buffers = validate_phase_capture(&wire.main.phase, capture.as_ref())?;
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
    let recurrent_capture = Arc::new(match &captures.metal_main_recurrent {
        Some(recurrent) => recurrent.clone(),
        None => CompiledRecurrentCapture::from_artifact(capture.as_ref(), None)?,
    });
    let accumulation = wire
        .accumulation
        .as_ref()
        .zip(captures.accumulation.as_ref())
        .map(|(phase, admitted_capture)| {
            let phase_capture = admitted_capture.clone();
            #[cfg(test)]
            update_decode_counts(|counts| counts.topology_phase_validations += 1);
            let state_buffers = validate_phase_capture(phase, phase_capture.as_ref())?;
            let cursor_projection = PreparedRecurrentCursorProjection::prepare(
                capture.as_ref(),
                phase_capture.as_ref(),
                state_buffers.values().copied(),
            )
            .map_err(replay_error)?;
            #[cfg(test)]
            update_decode_counts(|counts| counts.cursor_projections += 1);
            let capture_identity = cursor_projection.target_capture_identity();
            Ok(CompiledTrainingSiblingPlan {
                phase: CompiledRecurrentPhasePlan {
                    recurrent_capture: CompiledRecurrentCapture::from_artifact(
                        phase_capture.as_ref(),
                        None,
                    )?,
                    capture: phase_capture,
                    state_buffers,
                    cursor_projection: Arc::new(cursor_projection),
                    capture_identity,
                    admission: CompiledRecurrentPhaseAdmission::RetainUnchanged,
                },
            })
        })
        .transpose()?;
    let inner = CompiledTrainingPlan {
        capture,
        recurrent_capture,
        inputs: wire.main.inputs.clone(),
        phase_outputs: CompiledTrainingPhaseOutputSchema {
            loss: CompiledTrainingLossOutput::ScalarF32,
            named_outputs: wire.main.output_names.clone(),
            observations: adamw_observation_schema(
                wire.main.phase.clip_report,
                wire.main.phase.window_loss_report,
            ),
        },
        parameter_buffers,
        optimizer_buffers,
        workload_buffers,
        state_input_buffers: wire.main.state_input_buffers.clone(),
        state_input_keys,
        state_values,
        state_versions,
        recurrent_store_groups: wire
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
            let capture = captures
                .evaluation
                .as_ref()
                .expect("admitted evaluation capture is present")
                .clone();
            if capture.identity != evaluation.capture_identity {
                return Err(training("compiled evaluation artifact identity mismatch"));
            }
            Ok(CompiledEvaluationPlan {
                inference: match &captures.metal_evaluation_capture {
                    Some(inference) => inference.clone(),
                    None => CompiledEvaluationCapture::from_artifact(capture, None)?,
                },
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
    let partial_flush = wire
        .partial_flush
        .as_ref()
        .zip(captures.partial_flush.as_ref())
        .map(|(phase, capture)| {
            decode_auxiliary(
                inner.capture.as_ref(),
                phase,
                capture.clone(),
                captures.metal_partial_flush_recurrent.clone(),
            )
        })
        .transpose()?;
    let zero_grad = wire
        .zero_grad
        .as_ref()
        .zip(captures.zero_grad.as_ref())
        .map(|(phase, capture)| {
            decode_auxiliary(inner.capture.as_ref(), phase, capture.clone(), None)
        })
        .transpose()?;
    let plan = CompiledAdamWPlan {
        program_identity,
        inner,
        partial_flush,
        zero_grad,
        contract: CompiledAdamWContract {
            gradient_accumulation_steps: wire.gradient_accumulation_steps,
            token_weight_policy: wire.token_weight_policy.as_ref().map(Into::into),
            allow_zero_valid_token_microbatches: wire.allow_zero_valid_token_microbatches,
            max_gradient_norm: wire.max_gradient_norm_bits.map(f32::from_bits),
            clip_report: wire.clip_report,
            window_loss_report: wire.window_loss_report,
            loss_scale: f32::from_bits(wire.loss_scale_bits),
            dropout: wire.dropout.map(|dropout| CompiledDropoutState {
                config: CompiledDropoutConfig::new(CompiledDropoutKey(dropout.key)),
                blocks_per_replay: dropout.blocks_per_replay,
            }),
            host_token_inputs: wire.host_token_inputs.clone(),
            frozen_parameters: wire.frozen_parameters.clone(),
            learning_rate,
            optimizer: CompiledAdamWPolicy {
                beta1: f32::from_bits(wire.adamw.beta1_bits),
                beta2: f32::from_bits(wire.adamw.beta2_bits),
                eps: f32::from_bits(wire.adamw.eps_bits),
                weight_decay: f32::from_bits(wire.adamw.weight_decay_bits),
                weight_decay_exclusions: wire.adamw.weight_decay_exclusions.clone(),
            },
        },
        progress: CompiledTrainingWindowProgress::INITIAL,
        evaluation,
        compile_phases: None,
    };
    Ok(AdmittedTrainingTopology { plan })
}
