use super::{
    CompiledTrainingWindowProgress, CompiledTrainingWindowTopology, MAX_EXACT_F32_INTEGER_COUNT,
    checked_bytes, training, validate_adamw_progress, validate_user_name,
};
use crate::safetensors::{read_safetensors_file_bytes_with_limits, save_safetensors_file_bytes};
use crate::{
    DType, Metadata, Result, SafetensorsFileError, SafetensorsReadLimits, Scalar, Shape, StateDict,
    TensorData, load_safetensors, save_safetensors,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

pub(super) const ADAMW_CHECKPOINT_FORMAT_V1: &str = "rustgrad-compiled-adamw-v1";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V2: &str = "rustgrad-compiled-adamw-v2";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V3: &str = "rustgrad-compiled-adamw-v3";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V4: &str = "rustgrad-compiled-adamw-v4";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V5: &str = "rustgrad-compiled-adamw-v5";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V6: &str = "rustgrad-compiled-adamw-v6";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V7: &str = "rustgrad-compiled-adamw-v7";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V8: &str = "rustgrad-compiled-adamw-v8";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V9: &str = "rustgrad-compiled-adamw-v9";

macro_rules! define_adamw_checkpoint_formats {
    ($($variant:ident => $wire:ident),+ $(,)?) => {
        #[repr(u8)]
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum AdamWCheckpointFormat {
            $($variant),+
        }

        impl AdamWCheckpointFormat {
            fn parse(wire: &str) -> Result<Self> {
                match wire {
                    $($wire => Ok(Self::$variant),)+
                    _ => Err(training("compiled AdamW checkpoint format mismatch")),
                }
            }

            const fn wire_name(self) -> &'static str {
                match self {
                    $(Self::$variant => $wire,)+
                }
            }
        }
    };
}

define_adamw_checkpoint_formats! {
    V1 => ADAMW_CHECKPOINT_FORMAT_V1,
    V2 => ADAMW_CHECKPOINT_FORMAT_V2,
    V3 => ADAMW_CHECKPOINT_FORMAT_V3,
    V4 => ADAMW_CHECKPOINT_FORMAT_V4,
    V5 => ADAMW_CHECKPOINT_FORMAT_V5,
    V6 => ADAMW_CHECKPOINT_FORMAT_V6,
    V7 => ADAMW_CHECKPOINT_FORMAT_V7,
    V8 => ADAMW_CHECKPOINT_FORMAT_V8,
    V9 => ADAMW_CHECKPOINT_FORMAT_V9,
}

impl AdamWCheckpointFormat {
    fn metadata_fields(self) -> BTreeSet<&'static str> {
        let mut fields = BTreeSet::from(["format", "capture_identity", "parameter_names"]);
        if self == Self::V1 {
            fields.insert("step");
            return fields;
        }
        fields.extend([
            "replay_step",
            "optimizer_step",
            "gradient_accumulation_steps",
            "accumulation_index",
        ]);
        if self.stores_discarded_progress() {
            fields.insert("discarded_microbatch_count");
        }
        if self.stores_flush_progress() {
            fields.extend([
                "flush_capture_identity",
                "flushed_window_count",
                "flushed_microbatch_count",
                "dropout_state_present",
            ]);
        }
        if self.stores_optional_identities() {
            fields.extend([
                "flush_capture_identity_present",
                "reset_capture_identity",
                "reset_transition_count",
                "token_weighted_accumulation_present",
            ]);
        }
        if self.stores_reset_presence() {
            fields.extend([
                "reset_capture_identity_present",
                "window_loss_report_enabled",
            ]);
        }
        if self == Self::V9 {
            fields.insert("accumulation_capture_identity");
        }
        fields
    }

    const fn stores_discarded_progress(self) -> bool {
        self as u8 >= Self::V3 as u8
    }

    const fn stores_flush_progress(self) -> bool {
        self as u8 >= Self::V5 as u8
    }

    const fn stores_optional_identities(self) -> bool {
        self as u8 >= Self::V7 as u8
    }

    const fn stores_reset_presence(self) -> bool {
        self as u8 >= Self::V8 as u8
    }

    const fn stores_dropout_presence(self) -> bool {
        self as u8 >= Self::V5 as u8
    }

    const fn stores_token_weight_presence(self) -> bool {
        self as u8 >= Self::V7 as u8
    }

    const fn stores_window_loss_policy(self) -> bool {
        self as u8 >= Self::V8 as u8
    }
}

/// Deterministic, portable state for one exact compiled AdamW program.
///
/// The safetensors payload contains parameter and moment tensors plus any
/// partial gradient sums. String metadata authenticates the format, ordered
/// parameter names, compiled capture identity, replay/optimizer progress, and
/// accumulation policy, including discarded partial-window progress when
/// present. Fixed clipping and loss-scaling policy is authenticated by the
/// capture identity rather than duplicated as mutable checkpoint state. It
/// never serializes executable code, graphs, runtime slots, or host pointers.
/// Dropout-bearing v4 adds its recurrent U64 block counter. A v5 checkpoint
/// records progress consumed by explicit CPU partial-window flushes and
/// whether that same counter is present. Multi-step token weighting uses v6
/// for its recurrent valid-token count; `N=1` has no retained count. A v7
/// checkpoint records captured zero-grad transition history and its auxiliary
/// capture identity; opt-in window-loss reporting uses v8 for its recurrent
/// F32 numerator. A v9 checkpoint authenticates the private accumulation-only
/// sibling capture.
/// Legacy v1--v8 bytes remain accepted unchanged.
#[derive(Clone)]
pub struct CompiledAdamWCheckpoint {
    bytes: Vec<u8>,
    info: CompiledAdamWCheckpointInfo,
    decoded: Arc<DecodedAdamWCheckpoint>,
}

impl std::fmt::Debug for CompiledAdamWCheckpoint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledAdamWCheckpoint")
            .field("bytes", &self.bytes)
            .field("info", &self.info)
            .finish()
    }
}

impl PartialEq for CompiledAdamWCheckpoint {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.info == other.info
    }
}

impl Eq for CompiledAdamWCheckpoint {}

impl CompiledAdamWCheckpoint {
    /// Validates and owns deterministic checkpoint bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        let decoded = decode_adamw_checkpoint(&bytes)?;
        let info = CompiledAdamWCheckpointInfo::from_decoded(&decoded);
        Ok(Self {
            bytes,
            info,
            decoded: Arc::new(decoded),
        })
    }

    /// Loads and validates a local checkpoint under the default safetensors
    /// file-size bound.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self> {
        match Self::load_file_with_limits(path, SafetensorsReadLimits::default()) {
            Ok(checkpoint) => Ok(checkpoint),
            Err(SafetensorsFileError::Format(error)) => Err(error),
            Err(error) => Err(training(error.to_string())),
        }
    }

    /// Loads and validates a local checkpoint under an explicit byte bound.
    pub fn load_file_with_limits(
        path: impl AsRef<Path>,
        limits: SafetensorsReadLimits,
    ) -> std::result::Result<Self, SafetensorsFileError> {
        let bytes = read_safetensors_file_bytes_with_limits(path, limits)?;
        Self::from_bytes(bytes).map_err(SafetensorsFileError::Format)
    }

    /// Atomically replaces `path` with these exact checkpoint bytes after
    /// syncing a uniquely created staging file.
    pub fn save_file(&self, path: impl AsRef<Path>) -> Result<()> {
        save_safetensors_file_bytes(path, self.as_bytes())
    }

    /// Returns validated program identity and training progress without exposing
    /// or reparsing the checkpoint's wire representation.
    pub fn info(&self) -> &CompiledAdamWCheckpointInfo {
        &self.info
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn decoded(&self) -> &DecodedAdamWCheckpoint {
        &self.decoded
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Validated, read-only progress carried by a compiled AdamW checkpoint.
///
/// Legacy checkpoints expose zero or `None` for progress fields that
/// predate their wire formats. These values are suitable for selecting the next
/// workload batch or learning-rate schedule before recompiling the saved
/// program; executable capture and tensor payloads remain private checkpoint
/// details.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWCheckpointInfo {
    capture_identity: u64,
    replay_step: u64,
    optimizer_step: u64,
    gradient_accumulation_steps: u64,
    accumulation_index: u64,
    discarded_microbatches: u64,
    flushed_window_count: u64,
    flushed_microbatch_count: u64,
    flush_capture_identity: Option<u64>,
    dropout_block_counter: Option<u64>,
    accumulated_token_count: Option<u64>,
    accumulated_loss_numerator_bits: Option<u32>,
    window_loss_report: bool,
    reset_transition_count: u64,
    reset_capture_identity: Option<u64>,
    accumulation_capture_identity: Option<u64>,
}

impl CompiledAdamWCheckpointInfo {
    fn from_decoded(decoded: &DecodedAdamWCheckpoint) -> Self {
        Self {
            capture_identity: decoded.capture_identity,
            replay_step: decoded.replay_step,
            optimizer_step: decoded.optimizer_step,
            gradient_accumulation_steps: decoded.accumulation_steps,
            accumulation_index: decoded.accumulation_index,
            discarded_microbatches: decoded.discarded_microbatches,
            flushed_window_count: decoded.flushed_window_count,
            flushed_microbatch_count: decoded.flushed_microbatch_count,
            flush_capture_identity: decoded.flush_capture_identity,
            dropout_block_counter: decoded.dropout_block_counter,
            accumulated_token_count: decoded.accumulated_token_count,
            accumulated_loss_numerator_bits: decoded
                .accumulated_loss_numerator
                .as_ref()
                .map(|value| value.values()[0].to_bits()),
            window_loss_report: decoded.window_loss_report,
            reset_transition_count: decoded.reset_transition_count,
            reset_capture_identity: decoded.reset_capture_identity,
            accumulation_capture_identity: decoded.accumulation_capture_identity,
        }
    }

    /// Stable identity of the compiled training capture required for restore.
    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    /// Number of successfully committed training replays.
    pub fn replay_step(&self) -> u64 {
        self.replay_step
    }

    /// Number of committed full or explicitly flushed AdamW updates.
    pub fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    /// Fixed number of microbatches in each complete accumulation window.
    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }

    /// Number of retained microbatches in the current partial window.
    pub fn accumulation_index(&self) -> u64 {
        self.accumulation_index
    }

    /// Total retained microbatches discarded by successful `zero_grad` calls.
    pub fn discarded_microbatches(&self) -> u64 {
        self.discarded_microbatches
    }

    /// Total partial accumulation windows committed by explicit flushes.
    pub fn flushed_window_count(&self) -> u64 {
        self.flushed_window_count
    }

    /// Total microbatches consumed by explicit partial-window flushes.
    pub fn flushed_microbatch_count(&self) -> u64 {
        self.flushed_microbatch_count
    }

    /// Authenticated auxiliary flush identity when v5 progress is present.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    /// Next Threefry block counter for checkpoints carrying dropout state.
    pub fn dropout_block_counter(&self) -> Option<u64> {
        self.dropout_block_counter
    }

    /// Retained valid-token count for an opt-in weighted partial window.
    pub fn accumulated_token_count(&self) -> Option<u64> {
        self.accumulated_token_count
    }

    /// Retained weighted-loss numerator for an opt-in partial window.
    pub fn accumulated_loss_numerator(&self) -> Option<f32> {
        self.accumulated_loss_numerator_bits.map(f32::from_bits)
    }

    /// Whether this checkpoint carries the opt-in recurrent window-loss lane.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.window_loss_report
    }

    /// Number of nonempty windows discarded by the captured reset transition.
    pub fn reset_transition_count(&self) -> u64 {
        self.reset_transition_count
    }

    /// Authenticated state-only reset capture identity when v7 history exists.
    pub fn reset_capture_identity(&self) -> Option<u64> {
        self.reset_capture_identity
    }

    /// Authenticated private accumulation-only sibling capture identity.
    pub fn accumulation_capture_identity(&self) -> Option<u64> {
        self.accumulation_capture_identity
    }
}

#[derive(Clone, Debug)]
pub(super) struct DecodedAdamWCheckpoint {
    pub(super) capture_identity: u64,
    pub(super) replay_step: u64,
    pub(super) optimizer_step: u64,
    pub(super) accumulation_steps: u64,
    pub(super) accumulation_index: u64,
    pub(super) discarded_microbatches: u64,
    pub(super) flushed_window_count: u64,
    pub(super) flushed_microbatch_count: u64,
    pub(super) flush_capture_identity: Option<u64>,
    pub(super) dropout_block_counter: Option<u64>,
    pub(super) accumulated_token_count: Option<u64>,
    pub(super) accumulated_loss_numerator: Option<TensorData>,
    pub(super) window_loss_report: bool,
    pub(super) reset_transition_count: u64,
    pub(super) reset_capture_identity: Option<u64>,
    pub(super) accumulation_capture_identity: Option<u64>,
    pub(super) parameters: BTreeMap<String, TensorData>,
    pub(super) first_moments: BTreeMap<String, TensorData>,
    pub(super) second_moments: BTreeMap<String, TensorData>,
    pub(super) gradient_accumulators: BTreeMap<String, TensorData>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AdamWCheckpointProgress {
    pub(super) capture_identity: u64,
    pub(super) replay_step: u64,
    pub(super) optimizer_step: u64,
    pub(super) accumulation_steps: u64,
    pub(super) accumulation_index: u64,
    pub(super) discarded_microbatches: u64,
    pub(super) flushed_window_count: u64,
    pub(super) flushed_microbatch_count: u64,
    pub(super) flush_capture_identity: Option<u64>,
    pub(super) dropout_block_counter: Option<u64>,
    pub(super) accumulated_token_count: Option<u64>,
    pub(super) window_loss_report: bool,
    pub(super) reset_transition_count: u64,
    pub(super) reset_capture_identity: Option<u64>,
    pub(super) accumulation_capture_identity: Option<u64>,
}

pub(super) struct AdamWCheckpointTensors {
    pub(super) parameters: BTreeMap<String, TensorData>,
    pub(super) first_moments: BTreeMap<String, TensorData>,
    pub(super) second_moments: BTreeMap<String, TensorData>,
    pub(super) gradient_accumulators: BTreeMap<String, TensorData>,
    pub(super) accumulated_loss_numerator: Option<TensorData>,
}

impl AdamWCheckpointFormat {
    fn for_progress(
        progress: AdamWCheckpointProgress,
        topology: CompiledTrainingWindowTopology,
    ) -> Self {
        if progress.accumulation_capture_identity.is_some() {
            Self::V9
        } else if progress.window_loss_report {
            Self::V8
        } else if progress.reset_transition_count != 0 {
            Self::V7
        } else if progress.accumulated_token_count.is_some() {
            Self::V6
        } else if progress.flushed_window_count != 0 {
            Self::V5
        } else if progress.dropout_block_counter.is_some() {
            Self::V4
        } else if progress.discarded_microbatches != 0 {
            Self::V3
        } else if topology.accumulating() {
            Self::V2
        } else {
            Self::V1
        }
    }

    fn flush_identity_presence(
        self,
        progress: AdamWCheckpointProgress,
        topology: CompiledTrainingWindowTopology,
    ) -> bool {
        match self {
            Self::V5 | Self::V6 | Self::V9 => true,
            Self::V7 => {
                progress.accumulated_token_count.is_some() || progress.flushed_window_count != 0
            }
            Self::V8 => topology.accumulating(),
            Self::V1 | Self::V2 | Self::V3 | Self::V4 => false,
        }
    }

    const fn missing_flush_identity_message(self) -> &'static str {
        match self {
            Self::V5 => "compiled AdamW flushed checkpoint capture identity is absent",
            Self::V7 => "compiled AdamW reset checkpoint flush identity is absent",
            Self::V8 => "compiled AdamW window-loss checkpoint flush identity is absent",
            Self::V9 => "compiled AdamW v9 checkpoint flush identity is absent",
            Self::V6 => "compiled AdamW token-weighted checkpoint flush identity is absent",
            Self::V1 | Self::V2 | Self::V3 | Self::V4 => {
                "compiled AdamW checkpoint has unexpected flush identity requirement"
            }
        }
    }

    fn encode_metadata(
        self,
        progress: AdamWCheckpointProgress,
        topology: CompiledTrainingWindowTopology,
        parameter_names: String,
    ) -> Result<Metadata> {
        let mut metadata = Metadata::new();
        metadata.insert("format".into(), self.wire_name().into());
        metadata.insert(
            "capture_identity".into(),
            progress.capture_identity.to_string(),
        );
        if self == Self::V1 {
            metadata.insert("step".into(), progress.optimizer_step.to_string());
            metadata.insert("parameter_names".into(), parameter_names);
            return Ok(metadata);
        }

        metadata.insert("replay_step".into(), progress.replay_step.to_string());
        metadata.insert("optimizer_step".into(), progress.optimizer_step.to_string());
        metadata.insert(
            "gradient_accumulation_steps".into(),
            progress.accumulation_steps.to_string(),
        );
        metadata.insert(
            "accumulation_index".into(),
            progress.accumulation_index.to_string(),
        );
        if self.stores_discarded_progress() {
            metadata.insert(
                "discarded_microbatch_count".into(),
                progress.discarded_microbatches.to_string(),
            );
        }
        if self.stores_flush_progress() {
            let flush_identity_present = self.flush_identity_presence(progress, topology);
            let flush_capture_identity = if flush_identity_present {
                progress
                    .flush_capture_identity
                    .ok_or_else(|| training(self.missing_flush_identity_message()))?
            } else {
                0
            };
            metadata.insert(
                "flush_capture_identity".into(),
                flush_capture_identity.to_string(),
            );
            metadata.insert(
                "flushed_window_count".into(),
                progress.flushed_window_count.to_string(),
            );
            metadata.insert(
                "flushed_microbatch_count".into(),
                progress.flushed_microbatch_count.to_string(),
            );
            metadata.insert(
                "dropout_state_present".into(),
                progress.dropout_block_counter.is_some().to_string(),
            );
            if self.stores_optional_identities() {
                metadata.insert(
                    "flush_capture_identity_present".into(),
                    flush_identity_present.to_string(),
                );
                metadata.insert(
                    "reset_capture_identity".into(),
                    progress.reset_capture_identity.unwrap_or(0).to_string(),
                );
                metadata.insert(
                    "reset_transition_count".into(),
                    progress.reset_transition_count.to_string(),
                );
                metadata.insert(
                    "token_weighted_accumulation_present".into(),
                    progress.accumulated_token_count.is_some().to_string(),
                );
            }
            if self.stores_reset_presence() {
                metadata.insert(
                    "reset_capture_identity_present".into(),
                    progress.reset_capture_identity.is_some().to_string(),
                );
            }
            if self.stores_window_loss_policy() {
                metadata.insert(
                    "window_loss_report_enabled".into(),
                    progress.window_loss_report.to_string(),
                );
            }
        }
        if self == Self::V9 {
            metadata.insert(
                "accumulation_capture_identity".into(),
                progress
                    .accumulation_capture_identity
                    .expect("v9 accumulation identity was selected")
                    .to_string(),
            );
        }
        metadata.insert("parameter_names".into(), parameter_names);
        debug_assert_eq!(
            metadata.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            self.metadata_fields(),
        );
        Ok(metadata)
    }
}

fn validate_adamw_checkpoint(
    progress: AdamWCheckpointProgress,
    tensors: &AdamWCheckpointTensors,
) -> Result<CompiledTrainingWindowTopology> {
    validate_adamw_checkpoint_maps(
        &tensors.parameters,
        &tensors.first_moments,
        &tensors.second_moments,
    )?;
    validate_adamw_progress(
        CompiledTrainingWindowProgress {
            replay_step: progress.replay_step,
            optimizer_step: progress.optimizer_step,
            accumulation_index: progress.accumulation_index,
            discarded_microbatches: progress.discarded_microbatches,
            flushed_window_count: progress.flushed_window_count,
            flushed_microbatch_count: progress.flushed_microbatch_count,
            reset_transition_count: progress.reset_transition_count,
        },
        progress.accumulation_steps,
    )?;
    let topology = CompiledTrainingWindowTopology::from_validated_parts(
        progress.accumulation_steps,
        progress.accumulated_token_count.is_some(),
        progress.window_loss_report,
    );
    if topology.accumulating() {
        validate_gradient_accumulators(&tensors.parameters, &tensors.gradient_accumulators)?;
        match progress.accumulation_capture_identity {
            Some(identity) if identity != progress.capture_identity => {}
            Some(_) => {
                return Err(training(
                    "compiled AdamW accumulation capture identity is not distinct",
                ));
            }
            None => {
                return Err(training(
                    "compiled AdamW accumulation capture identity is absent",
                ));
            }
        }
    } else if !tensors.gradient_accumulators.is_empty()
        || progress.accumulation_capture_identity.is_some()
    {
        return Err(training(
            "compiled AdamW checkpoint has unexpected gradient accumulators",
        ));
    }
    if let Some(count) = progress.accumulated_token_count {
        if !topology.retains_token_count()
            || count > MAX_EXACT_F32_INTEGER_COUNT
            || (progress.accumulation_index == 0 && count != 0)
        {
            return Err(training(
                "compiled AdamW checkpoint accumulated token count is invalid",
            ));
        }
        if progress.flush_capture_identity.is_none() {
            return Err(training(
                "compiled AdamW token-weighted checkpoint flush identity is absent",
            ));
        }
    }
    match (
        topology.retains_window_numerator(),
        &tensors.accumulated_loss_numerator,
    ) {
        (false, None) => {}
        (true, Some(value)) if value.shape() == &Shape::from([]) && value.dtype() == DType::F32 => {
            checked_bytes(value)?;
        }
        _ => {
            return Err(training(
                "compiled AdamW checkpoint window-loss numerator is invalid",
            ));
        }
    }
    if progress.reset_transition_count != 0 {
        if progress.reset_capture_identity.is_none()
            || progress.reset_transition_count > progress.discarded_microbatches
        {
            return Err(training(
                "compiled AdamW checkpoint reset progress is invalid",
            ));
        }
    } else if progress.reset_capture_identity.is_some() {
        return Err(training(
            "compiled AdamW checkpoint has unexpected reset identity",
        ));
    }
    Ok(topology)
}

fn encode_adamw_tensors(
    topology: CompiledTrainingWindowTopology,
    tensors: AdamWCheckpointTensors,
    progress: AdamWCheckpointProgress,
) -> Result<(StateDict, Vec<String>)> {
    let AdamWCheckpointTensors {
        mut parameters,
        mut first_moments,
        mut second_moments,
        mut gradient_accumulators,
        accumulated_loss_numerator,
    } = tensors;
    let names = parameters.keys().cloned().collect::<Vec<_>>();
    let mut state = StateDict::default();
    for (ordinal, name) in names.iter().enumerate() {
        state.insert(
            format!("parameter.{ordinal}"),
            parameters
                .remove(name)
                .expect("checkpoint parameter names were validated"),
        );
        state.insert(
            format!("first_moment.{ordinal}"),
            first_moments
                .remove(name)
                .expect("checkpoint first-moment names were validated"),
        );
        state.insert(
            format!("second_moment.{ordinal}"),
            second_moments
                .remove(name)
                .expect("checkpoint second-moment names were validated"),
        );
        if topology.accumulating() {
            state.insert(
                format!("gradient_accumulator.{ordinal}"),
                gradient_accumulators
                    .remove(name)
                    .expect("checkpoint gradient-accumulator names were validated"),
            );
        }
    }
    if let Some(counter) = progress.dropout_block_counter {
        state.insert(
            "dropout_block_counter".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(counter)])?,
        );
    }
    if let Some(count) = progress.accumulated_token_count {
        state.insert(
            "accumulated_token_count".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(count)])?,
        );
    }
    if let Some(numerator) = accumulated_loss_numerator {
        state.insert("accumulated_loss_numerator".into(), numerator);
    }
    Ok((state, names))
}

pub(super) fn encode_adamw_checkpoint(
    progress: AdamWCheckpointProgress,
    tensors: AdamWCheckpointTensors,
) -> Result<Vec<u8>> {
    let topology = validate_adamw_checkpoint(progress, &tensors)?;
    let (tensors, names) = encode_adamw_tensors(topology, tensors, progress)?;
    let parameter_names = serde_json::to_string(&names)
        .map_err(|error| training(format!("checkpoint names: {error}")))?;
    let format = AdamWCheckpointFormat::for_progress(progress, topology);
    let metadata = format.encode_metadata(progress, topology, parameter_names)?;
    save_safetensors(&tensors, &metadata)
}
struct DecodedAdamWMetadata {
    format: AdamWCheckpointFormat,
    capture_identity: u64,
    accumulation_steps: u64,
    progress: CompiledTrainingWindowProgress,
    flush_capture_identity: Option<u64>,
    reset_capture_identity: Option<u64>,
    parameter_names: Vec<String>,
}

impl DecodedAdamWMetadata {
    fn parse(metadata: &Metadata) -> Result<Self> {
        let format = AdamWCheckpointFormat::parse(
            metadata
                .get("format")
                .ok_or_else(|| training("compiled AdamW checkpoint format is absent"))?,
        )?;
        if metadata.keys().map(String::as_str).collect::<BTreeSet<_>>() != format.metadata_fields()
        {
            return Err(training("compiled AdamW checkpoint metadata mismatch"));
        }
        let capture_identity = metadata["capture_identity"]
            .parse::<u64>()
            .map_err(|_| training("compiled AdamW checkpoint capture identity is invalid"))?;
        let (
            replay_step,
            optimizer_step,
            accumulation_steps,
            accumulation_index,
            discarded_microbatches,
            flushed_window_count,
            flushed_microbatch_count,
        ) = if format == AdamWCheckpointFormat::V1 {
            let step = metadata["step"]
                .parse::<u64>()
                .map_err(|_| training("compiled AdamW checkpoint step is invalid"))?;
            (step, step, 1, 0, 0, 0, 0)
        } else {
            (
                parse_checkpoint_u64(metadata, "replay_step")?,
                parse_checkpoint_u64(metadata, "optimizer_step")?,
                parse_checkpoint_u64(metadata, "gradient_accumulation_steps")?,
                parse_checkpoint_u64(metadata, "accumulation_index")?,
                if format.stores_discarded_progress() {
                    parse_checkpoint_u64(metadata, "discarded_microbatch_count")?
                } else {
                    0
                },
                if format.stores_flush_progress() {
                    parse_checkpoint_u64(metadata, "flushed_window_count")?
                } else {
                    0
                },
                if format.stores_flush_progress() {
                    parse_checkpoint_u64(metadata, "flushed_microbatch_count")?
                } else {
                    0
                },
            )
        };
        let flush_capture_identity = format.parse_flush_identity(metadata)?;
        let reset_transition_count = if format.stores_optional_identities() {
            parse_checkpoint_u64(metadata, "reset_transition_count")?
        } else {
            0
        };
        let reset_capture_identity = format.parse_reset_identity(metadata)?;
        let progress = CompiledTrainingWindowProgress {
            replay_step,
            optimizer_step,
            accumulation_index,
            discarded_microbatches,
            flushed_window_count,
            flushed_microbatch_count,
            reset_transition_count,
        };
        format.validate_legacy_progress(progress, reset_capture_identity)?;
        validate_adamw_progress(progress, accumulation_steps)?;

        let parameter_names = serde_json::from_str::<Vec<String>>(&metadata["parameter_names"])
            .map_err(|_| training("compiled AdamW checkpoint parameter names are invalid"))?;
        validate_checkpoint_parameter_names(&parameter_names)?;
        Ok(Self {
            format,
            capture_identity,
            accumulation_steps,
            progress,
            flush_capture_identity,
            reset_capture_identity,
            parameter_names,
        })
    }
}

impl AdamWCheckpointFormat {
    fn parse_flush_identity(self, metadata: &Metadata) -> Result<Option<u64>> {
        if matches!(self, Self::V5 | Self::V6) {
            return parse_checkpoint_u64(metadata, "flush_capture_identity").map(Some);
        }
        if !self.stores_optional_identities() {
            return Ok(None);
        }
        parse_optional_checkpoint_identity(
            metadata,
            "flush_capture_identity",
            "flush_capture_identity_present",
            "compiled AdamW checkpoint flush-identity flag is invalid",
            "compiled AdamW checkpoint has unexpected flush identity",
        )
    }

    fn parse_reset_identity(self, metadata: &Metadata) -> Result<Option<u64>> {
        if self == Self::V7 {
            return parse_checkpoint_u64(metadata, "reset_capture_identity").map(Some);
        }
        if !self.stores_reset_presence() {
            return Ok(None);
        }
        parse_optional_checkpoint_identity(
            metadata,
            "reset_capture_identity",
            "reset_capture_identity_present",
            "compiled AdamW checkpoint reset-identity flag is invalid",
            "compiled AdamW checkpoint has unexpected reset identity",
        )
    }

    fn validate_legacy_progress(
        self,
        progress: CompiledTrainingWindowProgress,
        reset_capture_identity: Option<u64>,
    ) -> Result<()> {
        if self == Self::V3 && progress.discarded_microbatches == 0 {
            return Err(training(
                "compiled AdamW v3 checkpoint discarded progress is invalid",
            ));
        }
        if self == Self::V5 && progress.flushed_window_count == 0 {
            return Err(training(
                "compiled AdamW v5 checkpoint flushed progress is invalid",
            ));
        }
        if self.stores_optional_identities() {
            if progress.reset_transition_count != 0 {
                if reset_capture_identity.is_none()
                    || progress.reset_transition_count > progress.discarded_microbatches
                {
                    return Err(training(
                        "compiled AdamW checkpoint reset progress is invalid",
                    ));
                }
            } else if reset_capture_identity.is_some() {
                return Err(training(
                    "compiled AdamW checkpoint has unexpected reset identity",
                ));
            }
        }
        Ok(())
    }
}

struct DecodedAdamWRecurrentState {
    topology: CompiledTrainingWindowTopology,
    dropout_block_counter: Option<u64>,
    accumulated_token_count: Option<u64>,
    accumulated_loss_numerator: Option<TensorData>,
}

fn decode_adamw_recurrent_state(
    decoded: &DecodedAdamWMetadata,
    metadata: &Metadata,
    tensors: &mut StateDict,
) -> Result<DecodedAdamWRecurrentState> {
    let has_dropout = if decoded.format.stores_dropout_presence() {
        parse_checkpoint_bool(
            metadata,
            "dropout_state_present",
            "compiled AdamW checkpoint dropout-state flag is invalid",
        )?
    } else {
        decoded.format == AdamWCheckpointFormat::V4
    };
    let dropout_block_counter = has_dropout
        .then(|| {
            take_checkpoint_scalar_u64(
                tensors,
                "dropout_block_counter",
                "compiled AdamW dropout counter is absent",
                "compiled AdamW dropout counter descriptor mismatch",
            )
        })
        .transpose()?;
    let has_accumulated_token_count = if decoded.format.stores_token_weight_presence() {
        parse_checkpoint_bool(
            metadata,
            "token_weighted_accumulation_present",
            "compiled AdamW checkpoint token-weighted flag is invalid",
        )?
    } else {
        decoded.format == AdamWCheckpointFormat::V6
    };
    let token_topology = CompiledTrainingWindowTopology::from_validated_parts(
        decoded.accumulation_steps,
        has_accumulated_token_count,
        false,
    );
    let accumulated_token_count = has_accumulated_token_count
        .then(|| {
            take_checkpoint_scalar_u64(
                tensors,
                "accumulated_token_count",
                "compiled AdamW accumulated token count is absent",
                "compiled AdamW accumulated token count descriptor mismatch",
            )
        })
        .transpose()?;
    if let Some(count) = accumulated_token_count
        && (!token_topology.retains_token_count()
            || count > MAX_EXACT_F32_INTEGER_COUNT
            || (decoded.progress.accumulation_index == 0 && count != 0))
    {
        return Err(training(
            "compiled AdamW checkpoint accumulated token count is invalid",
        ));
    }
    let window_loss_report = if decoded.format.stores_window_loss_policy() {
        match metadata["window_loss_report_enabled"].as_str() {
            "true" => true,
            "false" if decoded.format == AdamWCheckpointFormat::V9 => false,
            _ => {
                return Err(training(
                    "compiled AdamW checkpoint window-loss flag is invalid",
                ));
            }
        }
    } else {
        false
    };
    let topology = CompiledTrainingWindowTopology::from_validated_parts(
        decoded.accumulation_steps,
        has_accumulated_token_count,
        window_loss_report,
    );
    let accumulated_loss_numerator = window_loss_report
        .then(|| {
            take_checkpoint_scalar_f32(
                tensors,
                "accumulated_loss_numerator",
                "compiled AdamW accumulated loss numerator is absent",
                "compiled AdamW accumulated loss numerator descriptor mismatch",
            )
        })
        .transpose()?;
    let expects_flush_identity = if decoded.format == AdamWCheckpointFormat::V9 {
        topology.accumulating()
    } else {
        (topology.retains_window_numerator() && topology.accumulating())
            || accumulated_token_count.is_some()
            || decoded.progress.flushed_window_count != 0
    };
    if decoded.format.stores_optional_identities()
        && (decoded.flush_capture_identity.is_some() != expects_flush_identity)
    {
        return Err(training(
            "compiled AdamW checkpoint flush identity presence is invalid",
        ));
    }
    Ok(DecodedAdamWRecurrentState {
        topology,
        dropout_block_counter,
        accumulated_token_count,
        accumulated_loss_numerator,
    })
}

struct DecodedAdamWParameters {
    parameters: BTreeMap<String, TensorData>,
    first_moments: BTreeMap<String, TensorData>,
    second_moments: BTreeMap<String, TensorData>,
    gradient_accumulators: BTreeMap<String, TensorData>,
}

fn decode_adamw_parameters(
    names: Vec<String>,
    topology: CompiledTrainingWindowTopology,
    mut tensors: StateDict,
) -> Result<DecodedAdamWParameters> {
    let mut parameters = BTreeMap::new();
    let mut first_moments = BTreeMap::new();
    let mut second_moments = BTreeMap::new();
    let mut gradient_accumulators = BTreeMap::new();
    for (ordinal, name) in names.into_iter().enumerate() {
        let parameter = tensors
            .remove(&format!("parameter.{ordinal}"))
            .ok_or_else(|| training("compiled AdamW checkpoint parameter is absent"))?;
        let first = tensors
            .remove(&format!("first_moment.{ordinal}"))
            .ok_or_else(|| training("compiled AdamW checkpoint first moment is absent"))?;
        let second = tensors
            .remove(&format!("second_moment.{ordinal}"))
            .ok_or_else(|| training("compiled AdamW checkpoint second moment is absent"))?;
        let accumulator = topology
            .accumulating()
            .then(|| {
                tensors
                    .remove(&format!("gradient_accumulator.{ordinal}"))
                    .ok_or_else(|| {
                        training("compiled AdamW checkpoint gradient accumulator is absent")
                    })
            })
            .transpose()?;
        parameters.insert(name.clone(), parameter);
        first_moments.insert(name.clone(), first);
        second_moments.insert(name.clone(), second);
        if let Some(accumulator) = accumulator {
            gradient_accumulators.insert(name, accumulator);
        }
    }
    if !tensors.is_empty() {
        return Err(training("compiled AdamW checkpoint tensor set mismatch"));
    }
    validate_adamw_checkpoint_maps(&parameters, &first_moments, &second_moments)?;
    if topology.accumulating() {
        validate_gradient_accumulators(&parameters, &gradient_accumulators)?;
    }
    Ok(DecodedAdamWParameters {
        parameters,
        first_moments,
        second_moments,
        gradient_accumulators,
    })
}

pub(super) fn decode_adamw_checkpoint(bytes: &[u8]) -> Result<DecodedAdamWCheckpoint> {
    let (mut tensors, metadata) = load_safetensors(bytes)?;
    let decoded = DecodedAdamWMetadata::parse(&metadata)?;
    let recurrent = decode_adamw_recurrent_state(&decoded, &metadata, &mut tensors)?;
    let DecodedAdamWMetadata {
        format,
        capture_identity,
        accumulation_steps,
        progress,
        flush_capture_identity,
        reset_capture_identity,
        parameter_names,
        ..
    } = decoded;
    let parameters = decode_adamw_parameters(parameter_names, recurrent.topology, tensors)?;
    let accumulation_capture_identity = (format == AdamWCheckpointFormat::V9)
        .then(|| parse_checkpoint_u64(&metadata, "accumulation_capture_identity"))
        .transpose()?;
    if format == AdamWCheckpointFormat::V9 {
        if !recurrent.topology.accumulating() || accumulation_capture_identity.is_none() {
            return Err(training(
                "compiled AdamW v9 checkpoint accumulation identity differs",
            ));
        }
        if accumulation_capture_identity == Some(capture_identity) {
            return Err(training(
                "compiled AdamW checkpoint accumulation capture identity is not distinct",
            ));
        }
    }
    let CompiledTrainingWindowProgress {
        replay_step,
        optimizer_step,
        accumulation_index,
        discarded_microbatches,
        flushed_window_count,
        flushed_microbatch_count,
        reset_transition_count,
    } = progress;
    Ok(DecodedAdamWCheckpoint {
        capture_identity,
        replay_step,
        optimizer_step,
        accumulation_steps,
        accumulation_index,
        discarded_microbatches,
        flushed_window_count,
        flushed_microbatch_count,
        flush_capture_identity,
        dropout_block_counter: recurrent.dropout_block_counter,
        accumulated_token_count: recurrent.accumulated_token_count,
        accumulated_loss_numerator: recurrent.accumulated_loss_numerator,
        window_loss_report: recurrent.topology.retains_window_numerator(),
        reset_transition_count,
        reset_capture_identity,
        accumulation_capture_identity,
        parameters: parameters.parameters,
        first_moments: parameters.first_moments,
        second_moments: parameters.second_moments,
        gradient_accumulators: parameters.gradient_accumulators,
    })
}

fn validate_checkpoint_parameter_names(names: &[String]) -> Result<()> {
    if names.is_empty() {
        return Err(training("compiled AdamW checkpoint has no parameters"));
    }
    let mut unique = BTreeSet::new();
    for name in names {
        validate_user_name(name, "checkpoint parameter")?;
        if !unique.insert(name) {
            return Err(training("compiled AdamW checkpoint parameter names repeat"));
        }
    }
    Ok(())
}

fn parse_checkpoint_bool(metadata: &Metadata, name: &str, invalid: &'static str) -> Result<bool> {
    match metadata[name].as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(training(invalid)),
    }
}

fn parse_optional_checkpoint_identity(
    metadata: &Metadata,
    identity_name: &str,
    presence_name: &str,
    invalid_presence: &'static str,
    unexpected_identity: &'static str,
) -> Result<Option<u64>> {
    let present = parse_checkpoint_bool(metadata, presence_name, invalid_presence)?;
    let identity = parse_checkpoint_u64(metadata, identity_name)?;
    if !present && identity != 0 {
        return Err(training(unexpected_identity));
    }
    Ok(present.then_some(identity))
}

fn take_checkpoint_scalar_u64(
    tensors: &mut StateDict,
    name: &str,
    absent: &'static str,
    descriptor_mismatch: &'static str,
) -> Result<u64> {
    let value = tensors.remove(name).ok_or_else(|| training(absent))?;
    if value.shape() != &Shape::from([]) || value.dtype() != DType::U64 {
        return Err(training(descriptor_mismatch));
    }
    Ok(value.scalar_at(0).as_u64())
}

fn take_checkpoint_scalar_f32(
    tensors: &mut StateDict,
    name: &str,
    absent: &'static str,
    descriptor_mismatch: &'static str,
) -> Result<TensorData> {
    let value = tensors.remove(name).ok_or_else(|| training(absent))?;
    if value.shape() != &Shape::from([]) || value.dtype() != DType::F32 {
        return Err(training(descriptor_mismatch));
    }
    checked_bytes(&value)?;
    Ok(value)
}

fn parse_checkpoint_u64(metadata: &Metadata, name: &str) -> Result<u64> {
    metadata[name]
        .parse::<u64>()
        .map_err(|_| training(format!("compiled AdamW checkpoint {name} is invalid")))
}

fn validate_gradient_accumulators(
    parameters: &BTreeMap<String, TensorData>,
    accumulators: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if parameters.keys().ne(accumulators.keys()) {
        return Err(training(
            "compiled AdamW checkpoint gradient accumulator names mismatch",
        ));
    }
    for (name, parameter) in parameters {
        let accumulator = &accumulators[name];
        if accumulator.dtype() != DType::F32 || accumulator.shape() != parameter.shape() {
            return Err(training(
                "compiled AdamW checkpoint gradient accumulator descriptor mismatch",
            ));
        }
        checked_bytes(accumulator)?;
    }
    Ok(())
}

fn validate_adamw_checkpoint_maps(
    parameters: &BTreeMap<String, TensorData>,
    first_moments: &BTreeMap<String, TensorData>,
    second_moments: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if parameters.is_empty()
        || parameters.keys().ne(first_moments.keys())
        || parameters.keys().ne(second_moments.keys())
    {
        return Err(training("compiled AdamW checkpoint state names mismatch"));
    }
    for (name, parameter) in parameters {
        validate_user_name(name, "checkpoint parameter")?;
        let first = &first_moments[name];
        let second = &second_moments[name];
        if parameter.dtype() != DType::F32
            || first.dtype() != DType::F32
            || second.dtype() != DType::F32
            || first.shape() != parameter.shape()
            || second.shape() != parameter.shape()
        {
            return Err(training("compiled AdamW checkpoint descriptor mismatch"));
        }
        checked_bytes(parameter)?;
        checked_bytes(first)?;
        checked_bytes(second)?;
    }
    Ok(())
}

#[cfg(test)]
mod format_tests {
    use super::*;

    #[test]
    fn checkpoint_format_table_round_trips_and_declares_schema_growth() {
        let formats = [
            (AdamWCheckpointFormat::V1, ADAMW_CHECKPOINT_FORMAT_V1, 4),
            (AdamWCheckpointFormat::V2, ADAMW_CHECKPOINT_FORMAT_V2, 7),
            (AdamWCheckpointFormat::V3, ADAMW_CHECKPOINT_FORMAT_V3, 8),
            (AdamWCheckpointFormat::V4, ADAMW_CHECKPOINT_FORMAT_V4, 8),
            (AdamWCheckpointFormat::V5, ADAMW_CHECKPOINT_FORMAT_V5, 12),
            (AdamWCheckpointFormat::V6, ADAMW_CHECKPOINT_FORMAT_V6, 12),
            (AdamWCheckpointFormat::V7, ADAMW_CHECKPOINT_FORMAT_V7, 16),
            (AdamWCheckpointFormat::V8, ADAMW_CHECKPOINT_FORMAT_V8, 18),
            (AdamWCheckpointFormat::V9, ADAMW_CHECKPOINT_FORMAT_V9, 19),
        ];

        for (format, wire_name, metadata_field_count) in formats {
            assert_eq!(format.wire_name(), wire_name);
            assert_eq!(AdamWCheckpointFormat::parse(wire_name).unwrap(), format);
            assert_eq!(format.metadata_fields().len(), metadata_field_count);
        }
        assert!(AdamWCheckpointFormat::parse("rustgrad-compiled-adamw-v10").is_err());
        assert_eq!(
            AdamWCheckpointFormat::V9.metadata_fields(),
            BTreeSet::from([
                "format",
                "capture_identity",
                "accumulation_capture_identity",
                "flush_capture_identity",
                "flush_capture_identity_present",
                "reset_capture_identity",
                "reset_capture_identity_present",
                "replay_step",
                "optimizer_step",
                "gradient_accumulation_steps",
                "accumulation_index",
                "discarded_microbatch_count",
                "flushed_window_count",
                "flushed_microbatch_count",
                "reset_transition_count",
                "dropout_state_present",
                "token_weighted_accumulation_present",
                "window_loss_report_enabled",
                "parameter_names",
            ]),
        );
    }
}
