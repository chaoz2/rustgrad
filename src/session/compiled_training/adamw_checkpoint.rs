use super::{
    AdamWProgress, MAX_EXACT_F32_INTEGER_COUNT, checked_bytes, training, validate_adamw_progress,
    validate_user_name,
};
use crate::safetensors::{read_safetensors_file_bytes_with_limits, save_safetensors_file_bytes};
use crate::{
    DType, Metadata, Result, SafetensorsFileError, SafetensorsReadLimits, Scalar, Shape, StateDict,
    TensorData, load_safetensors, save_safetensors,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(super) const ADAMW_CHECKPOINT_FORMAT_V1: &str = "rustgrad-compiled-adamw-v1";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V2: &str = "rustgrad-compiled-adamw-v2";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V3: &str = "rustgrad-compiled-adamw-v3";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V4: &str = "rustgrad-compiled-adamw-v4";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V5: &str = "rustgrad-compiled-adamw-v5";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V6: &str = "rustgrad-compiled-adamw-v6";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V7: &str = "rustgrad-compiled-adamw-v7";
pub(super) const ADAMW_CHECKPOINT_FORMAT_V8: &str = "rustgrad-compiled-adamw-v8";

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
/// whether that same counter is present. Opt-in token-weighted accumulation
/// uses v6 for its recurrent valid-token count. A v7 checkpoint records
/// captured zero-grad transition history and its auxiliary capture identity;
/// opt-in window-loss reporting uses v8 for its recurrent F32 numerator.
/// Legacy v1--v7 bytes remain accepted unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAdamWCheckpoint {
    bytes: Vec<u8>,
    info: CompiledAdamWCheckpointInfo,
}

impl CompiledAdamWCheckpoint {
    /// Validates and owns deterministic checkpoint bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        let decoded = decode_adamw_checkpoint(&bytes)?;
        let info = CompiledAdamWCheckpointInfo::from_decoded(&decoded);
        Ok(Self { bytes, info })
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
}

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
}

pub(super) struct AdamWCheckpointTensors {
    pub(super) parameters: BTreeMap<String, TensorData>,
    pub(super) first_moments: BTreeMap<String, TensorData>,
    pub(super) second_moments: BTreeMap<String, TensorData>,
    pub(super) gradient_accumulators: BTreeMap<String, TensorData>,
    pub(super) accumulated_loss_numerator: Option<TensorData>,
}

pub(super) fn encode_adamw_checkpoint(
    progress: AdamWCheckpointProgress,
    tensors: AdamWCheckpointTensors,
) -> Result<Vec<u8>> {
    let AdamWCheckpointProgress {
        capture_identity,
        replay_step,
        optimizer_step,
        accumulation_steps,
        accumulation_index,
        discarded_microbatches,
        flushed_window_count,
        flushed_microbatch_count,
        flush_capture_identity,
        dropout_block_counter,
        accumulated_token_count,
        window_loss_report,
        reset_transition_count,
        reset_capture_identity,
    } = progress;
    let AdamWCheckpointTensors {
        parameters,
        first_moments,
        second_moments,
        gradient_accumulators,
        accumulated_loss_numerator,
    } = tensors;
    validate_adamw_checkpoint_maps(&parameters, &first_moments, &second_moments)?;
    validate_adamw_progress(
        AdamWProgress {
            replay_step,
            optimizer_step,
            accumulation_index,
            discarded_microbatches,
            flushed_window_count,
            flushed_microbatch_count,
            reset_transition_count,
        },
        accumulation_steps,
    )?;
    if accumulation_steps == 1 {
        if !gradient_accumulators.is_empty() {
            return Err(training(
                "compiled AdamW checkpoint has unexpected gradient accumulators",
            ));
        }
    } else {
        validate_gradient_accumulators(&parameters, &gradient_accumulators)?;
    }
    if let Some(count) = accumulated_token_count {
        if accumulation_steps <= 1
            || count > MAX_EXACT_F32_INTEGER_COUNT
            || count < accumulation_index
            || (count == 0) != (accumulation_index == 0)
        {
            return Err(training(
                "compiled AdamW checkpoint accumulated token count is invalid",
            ));
        }
        if flush_capture_identity.is_none() {
            return Err(training(
                "compiled AdamW token-weighted checkpoint flush identity is absent",
            ));
        }
    }
    match (window_loss_report, &accumulated_loss_numerator) {
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
    if reset_transition_count != 0 {
        if reset_capture_identity.is_none() || reset_transition_count > discarded_microbatches {
            return Err(training(
                "compiled AdamW checkpoint reset progress is invalid",
            ));
        }
    } else if reset_capture_identity.is_some() {
        return Err(training(
            "compiled AdamW checkpoint has unexpected reset identity",
        ));
    }
    let names = parameters.keys().cloned().collect::<Vec<_>>();
    let mut tensors = StateDict::default();
    for (ordinal, name) in names.iter().enumerate() {
        tensors.insert(format!("parameter.{ordinal}"), parameters[name].clone());
        tensors.insert(
            format!("first_moment.{ordinal}"),
            first_moments[name].clone(),
        );
        tensors.insert(
            format!("second_moment.{ordinal}"),
            second_moments[name].clone(),
        );
        if accumulation_steps > 1 {
            tensors.insert(
                format!("gradient_accumulator.{ordinal}"),
                gradient_accumulators[name].clone(),
            );
        }
    }
    if let Some(counter) = dropout_block_counter {
        tensors.insert(
            "dropout_block_counter".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(counter)])?,
        );
    }
    if let Some(count) = accumulated_token_count {
        tensors.insert(
            "accumulated_token_count".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(count)])?,
        );
    }
    if let Some(numerator) = accumulated_loss_numerator {
        tensors.insert("accumulated_loss_numerator".into(), numerator);
    }
    let parameter_names = serde_json::to_string(&names)
        .map_err(|error| training(format!("checkpoint names: {error}")))?;
    let metadata = if window_loss_report {
        let flush_identity_present = accumulation_steps > 1;
        let authenticated_flush_identity = if flush_identity_present {
            Some(flush_capture_identity.ok_or_else(|| {
                training("compiled AdamW window-loss checkpoint flush identity is absent")
            })?)
        } else {
            None
        };
        let reset_identity_present = reset_transition_count != 0;
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V8.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            (
                "flush_capture_identity".into(),
                authenticated_flush_identity.unwrap_or(0).to_string(),
            ),
            (
                "flush_capture_identity_present".into(),
                flush_identity_present.to_string(),
            ),
            (
                "reset_capture_identity".into(),
                reset_capture_identity.unwrap_or(0).to_string(),
            ),
            (
                "reset_capture_identity_present".into(),
                reset_identity_present.to_string(),
            ),
            ("replay_step".into(), replay_step.to_string()),
            ("optimizer_step".into(), optimizer_step.to_string()),
            (
                "gradient_accumulation_steps".into(),
                accumulation_steps.to_string(),
            ),
            ("accumulation_index".into(), accumulation_index.to_string()),
            (
                "discarded_microbatch_count".into(),
                discarded_microbatches.to_string(),
            ),
            (
                "flushed_window_count".into(),
                flushed_window_count.to_string(),
            ),
            (
                "flushed_microbatch_count".into(),
                flushed_microbatch_count.to_string(),
            ),
            (
                "reset_transition_count".into(),
                reset_transition_count.to_string(),
            ),
            (
                "dropout_state_present".into(),
                dropout_block_counter.is_some().to_string(),
            ),
            (
                "token_weighted_accumulation_present".into(),
                accumulated_token_count.is_some().to_string(),
            ),
            ("window_loss_report_enabled".into(), "true".into()),
            ("parameter_names".into(), parameter_names),
        ])
    } else if reset_transition_count != 0 {
        let flush_identity_present = accumulated_token_count.is_some() || flushed_window_count != 0;
        let authenticated_flush_identity = if flush_identity_present {
            Some(flush_capture_identity.ok_or_else(|| {
                training("compiled AdamW reset checkpoint flush identity is absent")
            })?)
        } else {
            None
        };
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V7.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            (
                "flush_capture_identity".into(),
                authenticated_flush_identity.unwrap_or(0).to_string(),
            ),
            (
                "flush_capture_identity_present".into(),
                flush_identity_present.to_string(),
            ),
            (
                "reset_capture_identity".into(),
                reset_capture_identity
                    .expect("reset checkpoint identity was validated")
                    .to_string(),
            ),
            ("replay_step".into(), replay_step.to_string()),
            ("optimizer_step".into(), optimizer_step.to_string()),
            (
                "gradient_accumulation_steps".into(),
                accumulation_steps.to_string(),
            ),
            ("accumulation_index".into(), accumulation_index.to_string()),
            (
                "discarded_microbatch_count".into(),
                discarded_microbatches.to_string(),
            ),
            (
                "flushed_window_count".into(),
                flushed_window_count.to_string(),
            ),
            (
                "flushed_microbatch_count".into(),
                flushed_microbatch_count.to_string(),
            ),
            (
                "reset_transition_count".into(),
                reset_transition_count.to_string(),
            ),
            (
                "dropout_state_present".into(),
                dropout_block_counter.is_some().to_string(),
            ),
            (
                "token_weighted_accumulation_present".into(),
                accumulated_token_count.is_some().to_string(),
            ),
            ("parameter_names".into(), parameter_names),
        ])
    } else if accumulated_token_count.is_some() {
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V6.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            (
                "flush_capture_identity".into(),
                flush_capture_identity
                    .expect("token-weighted checkpoint flush identity was validated")
                    .to_string(),
            ),
            ("replay_step".into(), replay_step.to_string()),
            ("optimizer_step".into(), optimizer_step.to_string()),
            (
                "gradient_accumulation_steps".into(),
                accumulation_steps.to_string(),
            ),
            ("accumulation_index".into(), accumulation_index.to_string()),
            (
                "discarded_microbatch_count".into(),
                discarded_microbatches.to_string(),
            ),
            (
                "flushed_window_count".into(),
                flushed_window_count.to_string(),
            ),
            (
                "flushed_microbatch_count".into(),
                flushed_microbatch_count.to_string(),
            ),
            (
                "dropout_state_present".into(),
                dropout_block_counter.is_some().to_string(),
            ),
            ("parameter_names".into(), parameter_names),
        ])
    } else if flushed_window_count != 0 {
        let flush_capture_identity = flush_capture_identity.ok_or_else(|| {
            training("compiled AdamW flushed checkpoint capture identity is absent")
        })?;
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V5.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            (
                "flush_capture_identity".into(),
                flush_capture_identity.to_string(),
            ),
            ("replay_step".into(), replay_step.to_string()),
            ("optimizer_step".into(), optimizer_step.to_string()),
            (
                "gradient_accumulation_steps".into(),
                accumulation_steps.to_string(),
            ),
            ("accumulation_index".into(), accumulation_index.to_string()),
            (
                "discarded_microbatch_count".into(),
                discarded_microbatches.to_string(),
            ),
            (
                "flushed_window_count".into(),
                flushed_window_count.to_string(),
            ),
            (
                "flushed_microbatch_count".into(),
                flushed_microbatch_count.to_string(),
            ),
            (
                "dropout_state_present".into(),
                dropout_block_counter.is_some().to_string(),
            ),
            ("parameter_names".into(), parameter_names),
        ])
    } else if dropout_block_counter.is_some() {
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V4.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            ("replay_step".into(), replay_step.to_string()),
            ("optimizer_step".into(), optimizer_step.to_string()),
            (
                "gradient_accumulation_steps".into(),
                accumulation_steps.to_string(),
            ),
            ("accumulation_index".into(), accumulation_index.to_string()),
            (
                "discarded_microbatch_count".into(),
                discarded_microbatches.to_string(),
            ),
            ("parameter_names".into(), parameter_names),
        ])
    } else if discarded_microbatches != 0 {
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V3.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            ("replay_step".into(), replay_step.to_string()),
            ("optimizer_step".into(), optimizer_step.to_string()),
            (
                "gradient_accumulation_steps".into(),
                accumulation_steps.to_string(),
            ),
            ("accumulation_index".into(), accumulation_index.to_string()),
            (
                "discarded_microbatch_count".into(),
                discarded_microbatches.to_string(),
            ),
            ("parameter_names".into(), parameter_names),
        ])
    } else if accumulation_steps == 1 {
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V1.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            ("step".into(), optimizer_step.to_string()),
            ("parameter_names".into(), parameter_names),
        ])
    } else {
        Metadata::from([
            ("format".into(), ADAMW_CHECKPOINT_FORMAT_V2.into()),
            ("capture_identity".into(), capture_identity.to_string()),
            ("replay_step".into(), replay_step.to_string()),
            ("optimizer_step".into(), optimizer_step.to_string()),
            (
                "gradient_accumulation_steps".into(),
                accumulation_steps.to_string(),
            ),
            ("accumulation_index".into(), accumulation_index.to_string()),
            ("parameter_names".into(), parameter_names),
        ])
    };
    save_safetensors(&tensors, &metadata)
}

pub(super) fn decode_adamw_checkpoint(bytes: &[u8]) -> Result<DecodedAdamWCheckpoint> {
    let (state, metadata) = load_safetensors(bytes)?;
    let format = metadata
        .get("format")
        .ok_or_else(|| training("compiled AdamW checkpoint format is absent"))?;
    let expected_metadata = match format.as_str() {
        ADAMW_CHECKPOINT_FORMAT_V1 => {
            BTreeSet::from(["format", "capture_identity", "step", "parameter_names"])
        }
        ADAMW_CHECKPOINT_FORMAT_V2 => BTreeSet::from([
            "format",
            "capture_identity",
            "replay_step",
            "optimizer_step",
            "gradient_accumulation_steps",
            "accumulation_index",
            "parameter_names",
        ]),
        ADAMW_CHECKPOINT_FORMAT_V3 => BTreeSet::from([
            "format",
            "capture_identity",
            "replay_step",
            "optimizer_step",
            "gradient_accumulation_steps",
            "accumulation_index",
            "discarded_microbatch_count",
            "parameter_names",
        ]),
        ADAMW_CHECKPOINT_FORMAT_V4 => BTreeSet::from([
            "format",
            "capture_identity",
            "replay_step",
            "optimizer_step",
            "gradient_accumulation_steps",
            "accumulation_index",
            "discarded_microbatch_count",
            "parameter_names",
        ]),
        ADAMW_CHECKPOINT_FORMAT_V5 => BTreeSet::from([
            "format",
            "capture_identity",
            "flush_capture_identity",
            "replay_step",
            "optimizer_step",
            "gradient_accumulation_steps",
            "accumulation_index",
            "discarded_microbatch_count",
            "flushed_window_count",
            "flushed_microbatch_count",
            "dropout_state_present",
            "parameter_names",
        ]),
        ADAMW_CHECKPOINT_FORMAT_V6 => BTreeSet::from([
            "format",
            "capture_identity",
            "flush_capture_identity",
            "replay_step",
            "optimizer_step",
            "gradient_accumulation_steps",
            "accumulation_index",
            "discarded_microbatch_count",
            "flushed_window_count",
            "flushed_microbatch_count",
            "dropout_state_present",
            "parameter_names",
        ]),
        ADAMW_CHECKPOINT_FORMAT_V7 => BTreeSet::from([
            "format",
            "capture_identity",
            "flush_capture_identity",
            "flush_capture_identity_present",
            "reset_capture_identity",
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
            "parameter_names",
        ]),
        ADAMW_CHECKPOINT_FORMAT_V8 => BTreeSet::from([
            "format",
            "capture_identity",
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
        _ => return Err(training("compiled AdamW checkpoint format mismatch")),
    };
    if metadata.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_metadata {
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
        flush_capture_identity,
        reset_transition_count,
        reset_capture_identity,
    ) = if format == ADAMW_CHECKPOINT_FORMAT_V1 {
        let step = metadata["step"]
            .parse::<u64>()
            .map_err(|_| training("compiled AdamW checkpoint step is invalid"))?;
        (step, step, 1, 0, 0, 0, 0, None, 0, None)
    } else {
        (
            parse_checkpoint_u64(&metadata, "replay_step")?,
            parse_checkpoint_u64(&metadata, "optimizer_step")?,
            parse_checkpoint_u64(&metadata, "gradient_accumulation_steps")?,
            parse_checkpoint_u64(&metadata, "accumulation_index")?,
            if matches!(
                format.as_str(),
                ADAMW_CHECKPOINT_FORMAT_V3
                    | ADAMW_CHECKPOINT_FORMAT_V4
                    | ADAMW_CHECKPOINT_FORMAT_V5
                    | ADAMW_CHECKPOINT_FORMAT_V6
                    | ADAMW_CHECKPOINT_FORMAT_V7
                    | ADAMW_CHECKPOINT_FORMAT_V8
            ) {
                parse_checkpoint_u64(&metadata, "discarded_microbatch_count")?
            } else {
                0
            },
            if matches!(
                format.as_str(),
                ADAMW_CHECKPOINT_FORMAT_V5
                    | ADAMW_CHECKPOINT_FORMAT_V6
                    | ADAMW_CHECKPOINT_FORMAT_V7
                    | ADAMW_CHECKPOINT_FORMAT_V8
            ) {
                parse_checkpoint_u64(&metadata, "flushed_window_count")?
            } else {
                0
            },
            if matches!(
                format.as_str(),
                ADAMW_CHECKPOINT_FORMAT_V5
                    | ADAMW_CHECKPOINT_FORMAT_V6
                    | ADAMW_CHECKPOINT_FORMAT_V7
                    | ADAMW_CHECKPOINT_FORMAT_V8
            ) {
                parse_checkpoint_u64(&metadata, "flushed_microbatch_count")?
            } else {
                0
            },
            if matches!(
                format.as_str(),
                ADAMW_CHECKPOINT_FORMAT_V5 | ADAMW_CHECKPOINT_FORMAT_V6
            ) {
                Some(parse_checkpoint_u64(&metadata, "flush_capture_identity")?)
            } else if matches!(
                format.as_str(),
                ADAMW_CHECKPOINT_FORMAT_V7 | ADAMW_CHECKPOINT_FORMAT_V8
            ) {
                let present = match metadata["flush_capture_identity_present"].as_str() {
                    "true" => true,
                    "false" => false,
                    _ => {
                        return Err(training(
                            "compiled AdamW checkpoint flush-identity flag is invalid",
                        ));
                    }
                };
                let identity = parse_checkpoint_u64(&metadata, "flush_capture_identity")?;
                if !present && identity != 0 {
                    return Err(training(
                        "compiled AdamW checkpoint has unexpected flush identity",
                    ));
                }
                present.then_some(identity)
            } else {
                None
            },
            if matches!(
                format.as_str(),
                ADAMW_CHECKPOINT_FORMAT_V7 | ADAMW_CHECKPOINT_FORMAT_V8
            ) {
                parse_checkpoint_u64(&metadata, "reset_transition_count")?
            } else {
                0
            },
            if format == ADAMW_CHECKPOINT_FORMAT_V7 {
                Some(parse_checkpoint_u64(&metadata, "reset_capture_identity")?)
            } else if format == ADAMW_CHECKPOINT_FORMAT_V8 {
                let present = match metadata["reset_capture_identity_present"].as_str() {
                    "true" => true,
                    "false" => false,
                    _ => {
                        return Err(training(
                            "compiled AdamW checkpoint reset-identity flag is invalid",
                        ));
                    }
                };
                let identity = parse_checkpoint_u64(&metadata, "reset_capture_identity")?;
                if !present && identity != 0 {
                    return Err(training(
                        "compiled AdamW checkpoint has unexpected reset identity",
                    ));
                }
                present.then_some(identity)
            } else {
                None
            },
        )
    };
    if format == ADAMW_CHECKPOINT_FORMAT_V3 && discarded_microbatches == 0 {
        return Err(training(
            "compiled AdamW v3 checkpoint discarded progress is invalid",
        ));
    }
    if format == ADAMW_CHECKPOINT_FORMAT_V5 && flushed_window_count == 0 {
        return Err(training(
            "compiled AdamW v5 checkpoint flushed progress is invalid",
        ));
    }
    if matches!(
        format.as_str(),
        ADAMW_CHECKPOINT_FORMAT_V7 | ADAMW_CHECKPOINT_FORMAT_V8
    ) {
        if reset_transition_count != 0 {
            if reset_capture_identity.is_none() || reset_transition_count > discarded_microbatches {
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
    validate_adamw_progress(
        AdamWProgress {
            replay_step,
            optimizer_step,
            accumulation_index,
            discarded_microbatches,
            flushed_window_count,
            flushed_microbatch_count,
            reset_transition_count,
        },
        accumulation_steps,
    )?;
    let names = serde_json::from_str::<Vec<String>>(&metadata["parameter_names"])
        .map_err(|_| training("compiled AdamW checkpoint parameter names are invalid"))?;
    if names.is_empty() {
        return Err(training("compiled AdamW checkpoint has no parameters"));
    }
    let mut unique = BTreeSet::new();
    for name in &names {
        validate_user_name(name, "checkpoint parameter")?;
        if !unique.insert(name.clone()) {
            return Err(training("compiled AdamW checkpoint parameter names repeat"));
        }
    }

    let mut tensors = state;
    let has_dropout = if matches!(
        format.as_str(),
        ADAMW_CHECKPOINT_FORMAT_V5
            | ADAMW_CHECKPOINT_FORMAT_V6
            | ADAMW_CHECKPOINT_FORMAT_V7
            | ADAMW_CHECKPOINT_FORMAT_V8
    ) {
        match metadata["dropout_state_present"].as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(training(
                    "compiled AdamW checkpoint dropout-state flag is invalid",
                ));
            }
        }
    } else {
        format == ADAMW_CHECKPOINT_FORMAT_V4
    };
    let dropout_block_counter = if has_dropout {
        let counter = tensors
            .remove("dropout_block_counter")
            .ok_or_else(|| training("compiled AdamW dropout counter is absent"))?;
        if counter.shape() != &Shape::from([]) || counter.dtype() != DType::U64 {
            return Err(training(
                "compiled AdamW dropout counter descriptor mismatch",
            ));
        }
        Some(counter.scalar_at(0).as_u64())
    } else {
        None
    };
    let has_accumulated_token_count = if matches!(
        format.as_str(),
        ADAMW_CHECKPOINT_FORMAT_V7 | ADAMW_CHECKPOINT_FORMAT_V8
    ) {
        match metadata["token_weighted_accumulation_present"].as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(training(
                    "compiled AdamW checkpoint token-weighted flag is invalid",
                ));
            }
        }
    } else {
        format == ADAMW_CHECKPOINT_FORMAT_V6
    };
    let accumulated_token_count = if has_accumulated_token_count {
        let count = tensors
            .remove("accumulated_token_count")
            .ok_or_else(|| training("compiled AdamW accumulated token count is absent"))?;
        if count.shape() != &Shape::from([]) || count.dtype() != DType::U64 {
            return Err(training(
                "compiled AdamW accumulated token count descriptor mismatch",
            ));
        }
        let count = count.scalar_at(0).as_u64();
        if accumulation_steps <= 1
            || count > MAX_EXACT_F32_INTEGER_COUNT
            || count < accumulation_index
            || (count == 0) != (accumulation_index == 0)
        {
            return Err(training(
                "compiled AdamW checkpoint accumulated token count is invalid",
            ));
        }
        Some(count)
    } else {
        None
    };
    let window_loss_report = if format == ADAMW_CHECKPOINT_FORMAT_V8 {
        match metadata["window_loss_report_enabled"].as_str() {
            "true" => true,
            _ => {
                return Err(training(
                    "compiled AdamW checkpoint window-loss flag is invalid",
                ));
            }
        }
    } else {
        false
    };
    let accumulated_loss_numerator = if window_loss_report {
        let numerator = tensors
            .remove("accumulated_loss_numerator")
            .ok_or_else(|| training("compiled AdamW accumulated loss numerator is absent"))?;
        if numerator.shape() != &Shape::from([]) || numerator.dtype() != DType::F32 {
            return Err(training(
                "compiled AdamW accumulated loss numerator descriptor mismatch",
            ));
        }
        checked_bytes(&numerator)?;
        Some(numerator)
    } else {
        None
    };
    if matches!(
        format.as_str(),
        ADAMW_CHECKPOINT_FORMAT_V7 | ADAMW_CHECKPOINT_FORMAT_V8
    ) && (flush_capture_identity.is_some()
        != ((window_loss_report && accumulation_steps > 1)
            || accumulated_token_count.is_some()
            || flushed_window_count != 0))
    {
        return Err(training(
            "compiled AdamW checkpoint flush identity presence is invalid",
        ));
    }
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
        let accumulator = if accumulation_steps > 1 {
            Some(
                tensors
                    .remove(&format!("gradient_accumulator.{ordinal}"))
                    .ok_or_else(|| {
                        training("compiled AdamW checkpoint gradient accumulator is absent")
                    })?,
            )
        } else {
            None
        };
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
    if accumulation_steps > 1 {
        validate_gradient_accumulators(&parameters, &gradient_accumulators)?;
    }
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
        dropout_block_counter,
        accumulated_token_count,
        accumulated_loss_numerator,
        window_loss_report,
        reset_transition_count,
        reset_capture_identity,
        parameters,
        first_moments,
        second_moments,
        gradient_accumulators,
    })
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
