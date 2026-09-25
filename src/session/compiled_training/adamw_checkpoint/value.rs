use super::{DecodedAdamWCheckpoint, decode_adamw_checkpoint};
use crate::safetensors::{read_safetensors_file_bytes_with_limits, save_safetensors_file_bytes};
use crate::{Result, SafetensorsFileError, SafetensorsReadLimits};
use std::path::Path;
use std::sync::Arc;

use super::super::training;

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

    pub(in crate::session::compiled_training) fn decoded(&self) -> &DecodedAdamWCheckpoint {
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
