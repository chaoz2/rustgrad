use super::adamw_contract::{CompiledAdamWContract, CompiledAdamWPolicy};
use super::module_checkpoint::{DecodedModuleCheckpoint, ModuleCheckpointStateKind};
use super::*;
use crate::file_io::{ExactFileError, read_file_bytes_bounded, replace_file_bytes_atomically};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    io,
    path::Path,
    sync::{Arc, OnceLock},
};

mod wire_decode;
mod wire_encode;
mod wire_validation;

use self::wire_decode::{
    decode_auxiliary, decode_input_key_map, decode_key_map, decode_manifest,
    validate_phase_capture, zero_frontier,
};
use self::wire_encode::{module_wire, momentum_program_wire, program_wire};

const MAGIC: &[u8; 4] = b"RGAP";
const LEGACY_FORMAT_VERSION: u8 = 1;
const METAL_FORMAT_VERSION: u8 = 2;
const OPTIMIZER_FORMAT_VERSION: u8 = 3;
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

/// Bounded, checksummed resource-free executable for one compiled optimizer.
///
/// Tensor state is persisted separately in a compatible complete-module
/// checkpoint. [`Self::info`] identifies the optimizer before a caller selects
/// the corresponding typed restore entrypoint.
#[derive(Clone)]
pub struct CompiledTrainingProgramArtifact {
    bytes: Vec<u8>,
    info: CompiledTrainingProgramArtifactInfo,
    admitted: Option<Arc<AdmittedProgramArtifact>>,
}

impl fmt::Debug for CompiledTrainingProgramArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledTrainingProgramArtifact")
            .field("bytes", &self.bytes)
            .field("info", &self.info)
            .finish()
    }
}

impl PartialEq for CompiledTrainingProgramArtifact {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.info == other.info
    }
}

impl Eq for CompiledTrainingProgramArtifact {}

/// Stable identity metadata decoded and validated from an RGAP envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledTrainingProgramArtifactInfo {
    format_version: u8,
    optimizer: CompiledTrainingOptimizer,
    identity: u64,
    capture_identity: u64,
    accumulation_capture_identity: Option<u64>,
    flush_capture_identity: Option<u64>,
    zero_grad_capture_identity: Option<u64>,
    evaluation_capture_identity: Option<u64>,
}

/// Optimizer policy embedded in a compiled-training program artifact.
///
/// This is deliberately an explicit enum rather than a string discriminator:
/// callers can inspect the executable contract without parsing private wire
/// data, while optimizer-specific plans keep their concrete Rust types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledTrainingOptimizer {
    AdamW,
    MomentumSgd,
}

/// Source-compatible AdamW name for the optimizer-neutral RGAP envelope.
pub type CompiledAdamWProgramArtifact = CompiledTrainingProgramArtifact;

/// Momentum-SGD compatibility name for the shared RGAP executable envelope.
pub type CompiledMomentumSgdProgramArtifact = CompiledTrainingProgramArtifact;

/// Source-compatible AdamW name for optimizer-neutral RGAP metadata.
pub type CompiledAdamWProgramArtifactInfo = CompiledTrainingProgramArtifactInfo;

/// A local compiled-program artifact file failure.
#[derive(Debug)]
pub enum CompiledTrainingProgramArtifactFileError {
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

/// Source-compatible AdamW name for local RGAP file failures.
pub type CompiledAdamWProgramArtifactFileError = CompiledTrainingProgramArtifactFileError;

/// Momentum-SGD compatibility name for local RGAP file failures.
pub type CompiledMomentumSgdProgramArtifactFileError = CompiledTrainingProgramArtifactFileError;

impl fmt::Display for CompiledTrainingProgramArtifactFileError {
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

impl std::error::Error for CompiledTrainingProgramArtifactFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Io { .. } | Self::Limit { .. } => None,
        }
    }
}

fn artifact_file_error(error: ExactFileError) -> CompiledTrainingProgramArtifactFileError {
    match error {
        ExactFileError::Io { operation, source } => CompiledTrainingProgramArtifactFileError::Io {
            operation,
            kind: source.kind(),
        },
        ExactFileError::Limit { actual, maximum } => {
            CompiledTrainingProgramArtifactFileError::Limit { actual, maximum }
        }
        ExactFileError::Allocation => CompiledTrainingProgramArtifactFileError::Io {
            operation: "allocate read buffer",
            kind: io::ErrorKind::OutOfMemory,
        },
        ExactFileError::InvalidFileName => CompiledTrainingProgramArtifactFileError::Io {
            operation: "validate path",
            kind: io::ErrorKind::InvalidInput,
        },
        ExactFileError::StagingExhausted => CompiledTrainingProgramArtifactFileError::Io {
            operation: "create unique staging file",
            kind: io::ErrorKind::AlreadyExists,
        },
    }
}

impl CompiledTrainingProgramArtifactInfo {
    pub fn format_version(&self) -> u8 {
        self.format_version
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub fn optimizer(&self) -> CompiledTrainingOptimizer {
        self.optimizer
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

impl CompiledTrainingProgramArtifact {
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
    ) -> std::result::Result<Self, CompiledTrainingProgramArtifactFileError> {
        Self::load_file_with_byte_limit(path, MAX_ARTIFACT_BYTES)
    }

    /// Loads and validates one local RGAP artifact under the smaller of the
    /// caller's byte limit and the format's intrinsic byte bound.
    pub fn load_file_with_byte_limit(
        path: impl AsRef<Path>,
        maximum: usize,
    ) -> std::result::Result<Self, CompiledTrainingProgramArtifactFileError> {
        let bytes = read_file_bytes_bounded(path, maximum.min(MAX_ARTIFACT_BYTES))
            .map_err(artifact_file_error)?;
        Self::from_bytes(bytes).map_err(CompiledTrainingProgramArtifactFileError::Format)
    }

    /// Atomically replaces `path` with these exact validated artifact bytes
    /// after syncing a uniquely created same-directory staging file.
    pub fn save_file(
        &self,
        path: impl AsRef<Path>,
    ) -> std::result::Result<(), CompiledTrainingProgramArtifactFileError> {
        let wire =
            decode(self.as_bytes()).map_err(CompiledTrainingProgramArtifactFileError::Format)?;
        let (info, _) = wire
            .validate()
            .map_err(CompiledTrainingProgramArtifactFileError::Format)?;
        if info != self.info {
            return Err(CompiledTrainingProgramArtifactFileError::Format(training(
                "compiled program artifact info differs",
            )));
        }
        replace_file_bytes_atomically(path, self.as_bytes()).map_err(artifact_file_error)
    }

    pub fn info(&self) -> &CompiledTrainingProgramArtifactInfo {
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

/// Recoverable failure while pairing a momentum-SGD executable artifact with
/// a complete-module checkpoint.
pub struct CompiledModuleMomentumSgdArtifactRestoreError<M> {
    module: M,
    source: Error,
}

impl<M> CompiledModuleMomentumSgdArtifactRestoreError<M> {
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

impl<M> fmt::Debug for CompiledModuleMomentumSgdArtifactRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleMomentumSgdArtifactRestoreError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleMomentumSgdArtifactRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "compiled momentum-SGD program artifact restore failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleMomentumSgdArtifactRestoreError<M> {}

#[derive(Clone, Debug)]
struct ProgramWire {
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
    optimizer: OptimizerPolicyWire,
    metal: Option<MetalProgramWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramWireSerde {
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
    #[serde(default)]
    adamw: Option<AdamWPolicyWire>,
    #[serde(default)]
    momentum_sgd: Option<MomentumSgdPolicyWire>,
    #[serde(default)]
    metal: Option<MetalProgramWire>,
}

#[derive(Serialize)]
struct ProgramWireSerdeRef<'a> {
    module: &'a ModuleWire,
    main: &'a MainWire,
    accumulation: &'a Option<PhaseWire>,
    partial_flush: &'a Option<PhaseWire>,
    zero_grad: &'a Option<PhaseWire>,
    evaluation: &'a Option<EvaluationWire>,
    gradient_accumulation_steps: u64,
    token_weight_policy: &'a Option<TokenWeightWire>,
    allow_zero_valid_token_microbatches: bool,
    max_gradient_norm_bits: Option<u32>,
    clip_report: bool,
    window_loss_report: bool,
    loss_scale_bits: u32,
    dropout: Option<DropoutWire>,
    host_token_inputs: &'a BTreeMap<String, Shape>,
    frozen_parameters: &'a BTreeSet<String>,
    learning_rate: &'a LearningRateWire,
    #[serde(skip_serializing_if = "Option::is_none")]
    adamw: Option<&'a AdamWPolicyWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    momentum_sgd: Option<&'a MomentumSgdPolicyWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metal: Option<&'a MetalProgramWire>,
}

impl Serialize for ProgramWire {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let (adamw, momentum_sgd) = match &self.optimizer {
            OptimizerPolicyWire::AdamW { adamw } => (Some(adamw), None),
            OptimizerPolicyWire::MomentumSgd { momentum_sgd } => (None, Some(momentum_sgd)),
        };
        ProgramWireSerdeRef {
            module: &self.module,
            main: &self.main,
            accumulation: &self.accumulation,
            partial_flush: &self.partial_flush,
            zero_grad: &self.zero_grad,
            evaluation: &self.evaluation,
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            token_weight_policy: &self.token_weight_policy,
            allow_zero_valid_token_microbatches: self.allow_zero_valid_token_microbatches,
            max_gradient_norm_bits: self.max_gradient_norm_bits,
            clip_report: self.clip_report,
            window_loss_report: self.window_loss_report,
            loss_scale_bits: self.loss_scale_bits,
            dropout: self.dropout,
            host_token_inputs: &self.host_token_inputs,
            frozen_parameters: &self.frozen_parameters,
            learning_rate: &self.learning_rate,
            adamw,
            momentum_sgd,
            metal: self.metal.as_ref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ProgramWire {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProgramWireSerde::deserialize(deserializer)?;
        let optimizer = match (wire.adamw, wire.momentum_sgd) {
            (Some(adamw), None) => OptimizerPolicyWire::AdamW { adamw },
            (None, Some(momentum_sgd)) => OptimizerPolicyWire::MomentumSgd { momentum_sgd },
            _ => {
                return Err(serde::de::Error::custom(
                    "compiled program artifact optimizer policy is inconsistent",
                ));
            }
        };
        Ok(Self {
            format_version: 0,
            module: wire.module,
            main: wire.main,
            accumulation: wire.accumulation,
            partial_flush: wire.partial_flush,
            zero_grad: wire.zero_grad,
            evaluation: wire.evaluation,
            gradient_accumulation_steps: wire.gradient_accumulation_steps,
            token_weight_policy: wire.token_weight_policy,
            allow_zero_valid_token_microbatches: wire.allow_zero_valid_token_microbatches,
            max_gradient_norm_bits: wire.max_gradient_norm_bits,
            clip_report: wire.clip_report,
            window_loss_report: wire.window_loss_report,
            loss_scale_bits: wire.loss_scale_bits,
            dropout: wire.dropout,
            host_token_inputs: wire.host_token_inputs,
            frozen_parameters: wire.frozen_parameters,
            learning_rate: wire.learning_rate,
            optimizer,
            metal: wire.metal,
        })
    }
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

#[derive(Clone, Debug)]
enum OptimizerPolicyWire {
    AdamW { adamw: AdamWPolicyWire },
    MomentumSgd { momentum_sgd: MomentumSgdPolicyWire },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MomentumSgdPolicyWire {}

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
enum AdmittedTrainingTopology {
    AdamW(Box<CompiledAdamWPlan>),
    MomentumSgd(Box<CompiledMomentumSgdPlan>),
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

    fn training_topology(&self) -> Result<Arc<AdmittedTrainingTopology>> {
        self.training_topology
            .get_or_init(|| {
                #[cfg(test)]
                update_decode_counts(|counts| counts.topology_seals += 1);
                seal_admitted_training_topology(&self.wire, &self.captures).map(Arc::new)
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
    if !matches!(
        wire.format_version,
        LEGACY_FORMAT_VERSION | METAL_FORMAT_VERSION | OPTIMIZER_FORMAT_VERSION
    ) {
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
    if !matches!(
        bytes[4],
        LEGACY_FORMAT_VERSION | METAL_FORMAT_VERSION | OPTIMIZER_FORMAT_VERSION
    ) {
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
    let unchecked = CompiledTrainingProgramArtifact {
        bytes: bytes.clone(),
        info: *artifact.info(),
        admitted: None,
    };
    (bytes, unchecked)
}

impl<M: Module> CompiledModuleTrainingPlan<M, CompiledAdamWPlan, Option<u64>> {
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
                attachment: None,
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
                attachment: None,
            }),
            Err(source) => Err(CompiledModuleAdamWArtifactRestoreError { module, source }),
        }
    }
}

impl<M: Module> CompiledModuleTrainingPlan<M, CompiledMomentumSgdPlan> {
    /// Serializes the resource-free momentum-SGD executable separately from
    /// its complete-module checkpoint.
    pub fn program_artifact(&self) -> Result<CompiledMomentumSgdProgramArtifact> {
        self.seal.validate_unchanged(&self.module)?;
        let wire = momentum_program_wire(self)?;
        let bytes = encode(&wire)?;
        let artifact = CompiledTrainingProgramArtifact::from_bytes(bytes)?;
        if artifact.info().optimizer() != CompiledTrainingOptimizer::MomentumSgd {
            return Err(training(
                "compiled momentum-SGD program artifact optimizer differs",
            ));
        }
        Ok(artifact)
    }

    /// Restores a differently initialized module without rebuilding the
    /// momentum-SGD graph, derivatives, schedules, or captures.
    pub fn restore_from_program_artifact(
        module: M,
        artifact: &CompiledMomentumSgdProgramArtifact,
        checkpoint: &CompiledModuleMomentumSgdCheckpoint,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdArtifactRestoreError<M>> {
        match restore_momentum_owner(&module, artifact, checkpoint) {
            Ok((plan, seal)) => Ok(Self {
                module,
                plan,
                seal,
                attachment: (),
            }),
            Err(source) => Err(CompiledModuleMomentumSgdArtifactRestoreError { module, source }),
        }
    }

    /// Restores from one atomically persisted executable-and-state bundle
    /// without rebuilding the momentum-SGD training program.
    pub fn restore_from_resume_bundle(
        module: M,
        bundle: &CompiledMomentumSgdResumeBundle,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdArtifactRestoreError<M>> {
        match restore_momentum_owner_from_admitted(&module, bundle.admitted()) {
            Ok((plan, seal)) => Ok(Self {
                module,
                plan,
                seal,
                attachment: (),
            }),
            Err(source) => Err(CompiledModuleMomentumSgdArtifactRestoreError { module, source }),
        }
    }
}

fn checkpoint_module_wire<C>(
    decoded: &DecodedModuleCheckpoint<C>,
    optimizer_parameters: &BTreeMap<String, TensorData>,
) -> Result<ModuleWire> {
    let states = decoded
        .states
        .iter()
        .map(|state| {
            let value = state
                .value
                .as_ref()
                .or_else(|| optimizer_parameters.get(&state.name))
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
pub(super) struct AdmittedArtifactCheckpointPair<C> {
    artifact: Arc<AdmittedProgramArtifact>,
    checkpoint: Arc<DecodedModuleCheckpoint<C>>,
}

fn admitted_artifact(
    artifact: &CompiledTrainingProgramArtifact,
) -> Result<Arc<AdmittedProgramArtifact>> {
    match artifact.admitted() {
        Some(admitted) => Ok(admitted.clone()),
        None => {
            let mut wire = decode(artifact.as_bytes())?;
            let (artifact_info, captures) = wire.validate()?;
            if artifact_info != *artifact.info() {
                return Err(training("compiled program artifact info differs"));
            }
            wire.discard_capture_bytes();
            Ok(Arc::new(AdmittedProgramArtifact::new(wire, captures)))
        }
    }
}

fn decode_admitted_artifact_checkpoint_pair(
    artifact: &CompiledAdamWProgramArtifact,
    checkpoint: &CompiledModuleAdamWCheckpoint,
) -> Result<AdmittedArtifactCheckpointPair<CompiledAdamWCheckpoint>> {
    #[cfg(test)]
    update_decode_counts(|counts| counts.pair_admissions += 1);
    let admitted = admitted_artifact(artifact)?;
    let wire = &admitted.wire;
    let artifact_info = *artifact.info();
    if artifact_info.optimizer != CompiledTrainingOptimizer::AdamW {
        return Err(training(
            "compiled program artifact optimizer differs from checkpoint",
        ));
    }
    let decoded_module = checkpoint.decoded_arc().clone();
    if checkpoint_module_wire(
        decoded_module.as_ref(),
        &decoded_module.optimizer.decoded().parameters,
    )? != wire.module
    {
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
) -> Result<Arc<AdmittedArtifactCheckpointPair<CompiledAdamWCheckpoint>>> {
    decode_admitted_artifact_checkpoint_pair(artifact, checkpoint).map(Arc::new)
}

fn decode_admitted_momentum_artifact_checkpoint_pair(
    artifact: &CompiledMomentumSgdProgramArtifact,
    checkpoint: &CompiledModuleMomentumSgdCheckpoint,
) -> Result<AdmittedArtifactCheckpointPair<CompiledMomentumSgdCheckpoint>> {
    #[cfg(test)]
    update_decode_counts(|counts| counts.pair_admissions += 1);
    let admitted = admitted_artifact(artifact)?;
    if artifact.info().optimizer() != CompiledTrainingOptimizer::MomentumSgd {
        return Err(training(
            "compiled program artifact optimizer differs from checkpoint",
        ));
    }
    let wire = &admitted.wire;
    let decoded = checkpoint.decoded_arc().clone();
    if checkpoint_module_wire(decoded.as_ref(), decoded.optimizer.parameters())? != wire.module {
        return Err(training("compiled program artifact module schema mismatch"));
    }
    if decoded.evaluation_capture_identity.is_some()
        || decoded.optimizer.capture_identity() != artifact.info().capture_identity()
    {
        return Err(training(
            "compiled program artifact checkpoint policy mismatch",
        ));
    }
    Ok(AdmittedArtifactCheckpointPair {
        artifact: admitted,
        checkpoint: decoded,
    })
}

pub(super) fn admit_momentum_artifact_checkpoint_pair(
    artifact: &CompiledMomentumSgdProgramArtifact,
    checkpoint: &CompiledModuleMomentumSgdCheckpoint,
) -> Result<Arc<AdmittedArtifactCheckpointPair<CompiledMomentumSgdCheckpoint>>> {
    decode_admitted_momentum_artifact_checkpoint_pair(artifact, checkpoint).map(Arc::new)
}

fn restore_owner<M: Module>(
    module: &M,
    artifact: &CompiledAdamWProgramArtifact,
    checkpoint: &CompiledModuleAdamWCheckpoint,
) -> Result<(CompiledAdamWPlan, CompiledModuleSeal)> {
    let admitted = decode_admitted_artifact_checkpoint_pair(artifact, checkpoint)?;
    restore_owner_from_admitted(module, &admitted)
}

fn restore_momentum_owner<M: Module>(
    module: &M,
    artifact: &CompiledMomentumSgdProgramArtifact,
    checkpoint: &CompiledModuleMomentumSgdCheckpoint,
) -> Result<(CompiledMomentumSgdPlan, CompiledModuleSeal)> {
    let admitted = decode_admitted_momentum_artifact_checkpoint_pair(artifact, checkpoint)?;
    restore_momentum_owner_from_admitted(module, &admitted)
}

fn restore_momentum_owner_from_admitted<M: Module>(
    module: &M,
    admitted: &AdmittedArtifactCheckpointPair<CompiledMomentumSgdCheckpoint>,
) -> Result<(CompiledMomentumSgdPlan, CompiledModuleSeal)> {
    let wire = &admitted.artifact.wire;
    let decoded = admitted.checkpoint.as_ref();
    let mut seal = CompiledModuleSeal::capture(module, &BTreeSet::new())?;
    let _immutable_values =
        seal.apply_module_checkpoint(decoded, decoded.optimizer.parameters())?;
    if module_wire(&seal) != wire.module {
        return Err(training(
            "compiled program artifact destination module mismatch",
        ));
    }
    let topology = admitted.artifact.training_topology()?;
    let plan = match topology.as_ref() {
        AdmittedTrainingTopology::MomentumSgd(plan) => plan
            .as_ref()
            .clone()
            .restore_checkpoint_owned(&decoded.optimizer)?,
        AdmittedTrainingTopology::AdamW(_) => {
            return Err(training(
                "compiled program artifact optimizer differs from checkpoint",
            ));
        }
    };
    seal.validate_unchanged(module)?;
    Ok((plan, seal))
}

fn restore_owner_from_admitted<M: Module>(
    module: &M,
    admitted: &AdmittedArtifactCheckpointPair<CompiledAdamWCheckpoint>,
) -> Result<(CompiledAdamWPlan, CompiledModuleSeal)> {
    let wire = &admitted.artifact.wire;
    let decoded_module = admitted.checkpoint.as_ref();
    let mut seal = CompiledModuleSeal::capture(module, &wire.frozen_parameters)?;
    let _immutable_values = seal.apply_module_checkpoint(
        decoded_module,
        &decoded_module.optimizer.decoded().parameters,
    )?;
    if module_wire(&seal) != wire.module {
        return Err(training(
            "compiled program artifact destination module mismatch",
        ));
    }
    let plan = admitted.artifact.training_topology()?;
    let plan = match plan.as_ref() {
        AdmittedTrainingTopology::AdamW(plan) => plan
            .as_ref()
            .clone()
            .restore_checkpoint_owned(&decoded_module.optimizer)?,
        AdmittedTrainingTopology::MomentumSgd(_) => {
            return Err(training(
                "compiled program artifact optimizer differs from checkpoint",
            ));
        }
    };
    seal.validate_unchanged(module)?;
    Ok((plan, seal))
}

fn seal_admitted_training_topology(
    wire: &ProgramWire,
    captures: &ProgramCaptures,
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
    if matches!(&wire.optimizer, OptimizerPolicyWire::MomentumSgd { .. }) {
        return CompiledMomentumSgdPlan::from_inner(inner)
            .map(Box::new)
            .map(AdmittedTrainingTopology::MomentumSgd);
    }
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
    let OptimizerPolicyWire::AdamW { adamw } = &wire.optimizer else {
        return Err(training("compiled program artifact AdamW policy is absent"));
    };
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
                beta1: f32::from_bits(adamw.beta1_bits),
                beta2: f32::from_bits(adamw.beta2_bits),
                eps: f32::from_bits(adamw.eps_bits),
                weight_decay: f32::from_bits(adamw.weight_decay_bits),
                weight_decay_exclusions: adamw.weight_decay_exclusions.clone(),
            },
        },
        progress: CompiledTrainingWindowProgress::INITIAL,
        evaluation,
        compile_phases: None,
    };
    Ok(AdmittedTrainingTopology::AdamW(Box::new(plan)))
}
