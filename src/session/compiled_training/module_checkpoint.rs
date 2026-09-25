//! Optimizer-neutral complete-module checkpoint envelopes.

use super::{CompiledCheckpointParameterSnapshot, checked_bytes, training};
use crate::safetensors::{read_safetensors_file_bytes_with_limits, save_safetensors_file_bytes};
use crate::{DType, Metadata, Result, StateDict, TensorData, load_safetensors, save_safetensors};
use crate::{SafetensorsFileError, SafetensorsReadLimits};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

const OPTIMIZER_TENSOR: &str = "optimizer_checkpoint";
const MAX_MODULE_STATE_COUNT: usize = 1 << 20;

/// Optimizer checkpoint payload supported by a complete owned-module
/// checkpoint.
///
/// Implementations retain their own optimizer wire format. The shared module
/// envelope adds only canonical traversal topology, tied aliases, immutable
/// values, and an optional evaluator identity.
pub trait CompiledModuleCheckpointPayload: CompiledCheckpointParameterSnapshot {
    /// Complete-module envelope format without an attached evaluator.
    const MODULE_FORMAT: &'static str;

    /// Optional envelope format used when an evaluator is attached.
    const MODULE_EVALUATION_FORMAT: Option<&'static str> = None;

    /// Decodes the exact embedded optimizer payload.
    fn decode_module_payload(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;

    /// Returns the exact optimizer payload embedded in the module envelope.
    fn encode_module_payload(&self) -> Result<Cow<'_, [u8]>>;

    /// Returns the canonical trainable inventory embedded in the payload.
    fn module_parameter_names(&self) -> Result<BTreeSet<String>> {
        Ok(self.checkpoint_parameter_snapshots()?.into_keys().collect())
    }

    #[doc(hidden)]
    fn record_module_checkpoint_decode() {}
}

/// Portable complete-module checkpoint around one optimizer payload.
///
/// The deterministic safetensors envelope preserves canonical module
/// topology, tied aliases, frozen parameters, and buffers without serializing
/// executable graphs, schedules, runtime resources, or host identities.
#[derive(Clone)]
pub struct CompiledModuleCheckpoint<C> {
    bytes: Vec<u8>,
    decoded: Arc<DecodedModuleCheckpoint<C>>,
}

impl<C: std::fmt::Debug> std::fmt::Debug for CompiledModuleCheckpoint<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledModuleCheckpoint")
            .field("bytes", &self.bytes)
            .field("optimizer", &self.decoded.optimizer)
            .field(
                "evaluation_capture_identity",
                &self.decoded.evaluation_capture_identity,
            )
            .finish()
    }
}

impl<C> PartialEq for CompiledModuleCheckpoint<C> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl<C> Eq for CompiledModuleCheckpoint<C> {}

impl<C: CompiledModuleCheckpointPayload> CompiledModuleCheckpoint<C> {
    /// Validates and owns deterministic complete-module checkpoint bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        C::record_module_checkpoint_decode();
        let decoded = decode_module_checkpoint(&bytes, module_formats::<C>(), |optimizer_bytes| {
            let optimizer = C::decode_module_payload(optimizer_bytes)?;
            let parameter_names = optimizer.module_parameter_names()?;
            Ok((optimizer, parameter_names))
        })?;
        Ok(Self {
            bytes,
            decoded: Arc::new(decoded),
        })
    }

    /// Loads and validates a local complete-module checkpoint under the
    /// default safetensors file-size bound.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self> {
        match Self::load_file_with_limits(path, SafetensorsReadLimits::default()) {
            Ok(checkpoint) => Ok(checkpoint),
            Err(SafetensorsFileError::Format(error)) => Err(error),
            Err(error) => Err(training(error.to_string())),
        }
    }

    /// Loads and validates a local complete-module checkpoint under an
    /// explicit byte bound.
    pub fn load_file_with_limits(
        path: impl AsRef<Path>,
        limits: SafetensorsReadLimits,
    ) -> std::result::Result<Self, SafetensorsFileError> {
        let bytes = read_safetensors_file_bytes_with_limits(path, limits)?;
        Self::from_bytes(bytes).map_err(SafetensorsFileError::Format)
    }

    /// Atomically replaces `path` with these exact checkpoint bytes.
    pub fn save_file(&self, path: impl AsRef<Path>) -> Result<()> {
        save_safetensors_file_bytes(path, self.as_bytes())
    }

    /// Returns the unchanged embedded optimizer checkpoint.
    pub fn optimizer_checkpoint(&self) -> &C {
        &self.decoded.optimizer
    }

    /// Returns the evaluator capture identity when the envelope requires one.
    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.decoded.evaluation_capture_identity
    }

    /// Returns the exact deterministic complete-module checkpoint bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the checkpoint and returns its exact deterministic bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub(super) fn decoded(&self) -> &DecodedModuleCheckpoint<C> {
        &self.decoded
    }

    pub(super) fn decoded_arc(&self) -> &Arc<DecodedModuleCheckpoint<C>> {
        &self.decoded
    }
}

#[derive(Clone, Copy)]
pub(super) struct ModuleCheckpointFormats {
    base: &'static str,
    with_evaluation: Option<&'static str>,
}

impl ModuleCheckpointFormats {
    pub(super) const fn new(base: &'static str, with_evaluation: Option<&'static str>) -> Self {
        Self {
            base,
            with_evaluation,
        }
    }

    fn encode(self, evaluation_capture_identity: Option<u64>) -> Result<&'static str> {
        match evaluation_capture_identity {
            None => Ok(self.base),
            Some(_) => self.with_evaluation.ok_or_else(|| {
                training("compiled module checkpoint does not support evaluation state")
            }),
        }
    }

    fn has_evaluation(self, format: Option<&str>) -> Result<bool> {
        if format == Some(self.base) {
            return Ok(false);
        }
        if let Some(with_evaluation) = self.with_evaluation
            && format == Some(with_evaluation)
        {
            return Ok(true);
        }
        Err(training("compiled module checkpoint format mismatch"))
    }
}

fn module_formats<C: CompiledModuleCheckpointPayload>() -> ModuleCheckpointFormats {
    ModuleCheckpointFormats::new(C::MODULE_FORMAT, C::MODULE_EVALUATION_FORMAT)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ModuleCheckpointStateKind {
    Parameter,
    Buffer,
}

impl ModuleCheckpointStateKind {
    fn wire(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::Buffer => "buffer",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "parameter" => Ok(Self::Parameter),
            "buffer" => Ok(Self::Buffer),
            _ => Err(training("compiled module checkpoint state kind is invalid")),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ModuleCheckpointState {
    pub(super) name: String,
    pub(super) kind: ModuleCheckpointStateKind,
    pub(super) source_trainable: bool,
    pub(super) policy_frozen: bool,
    pub(super) value: Option<TensorData>,
}

impl ModuleCheckpointState {
    pub(super) fn trainable(&self) -> bool {
        self.kind == ModuleCheckpointStateKind::Parameter
            && self.source_trainable
            && !self.policy_frozen
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ModuleCheckpointVisit {
    pub(super) name: String,
    pub(super) canonical_name: String,
}

#[derive(Clone, Debug)]
pub(super) struct DecodedModuleCheckpoint<C> {
    pub(super) optimizer: C,
    pub(super) evaluation_capture_identity: Option<u64>,
    pub(super) states: Vec<ModuleCheckpointState>,
    pub(super) visits: Vec<ModuleCheckpointVisit>,
}

fn state_metadata_key(index: usize, field: &str) -> String {
    format!("state.{index}.{field}")
}

fn visit_metadata_key(index: usize, field: &str) -> String {
    format!("visit.{index}.{field}")
}

fn immutable_tensor_key(index: usize) -> String {
    format!("immutable.{index}")
}

fn parse_count(metadata: &Metadata, key: &str) -> Result<usize> {
    let count = metadata
        .get(key)
        .ok_or_else(|| training("compiled module checkpoint count is absent"))?
        .parse::<usize>()
        .map_err(|_| training("compiled module checkpoint count is invalid"))?;
    if count > MAX_MODULE_STATE_COUNT {
        return Err(training("compiled module checkpoint count exceeds limit"));
    }
    Ok(count)
}

fn parse_bool(metadata: &Metadata, key: &str) -> Result<bool> {
    match metadata.get(key).map(String::as_str) {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        _ => Err(training("compiled module checkpoint boolean is invalid")),
    }
}

pub(super) fn encode_module_checkpoint(
    formats: ModuleCheckpointFormats,
    optimizer_bytes: &[u8],
    evaluation_capture_identity: Option<u64>,
    states: &[ModuleCheckpointState],
    visits: &[ModuleCheckpointVisit],
) -> Result<Vec<u8>> {
    if states.len() > MAX_MODULE_STATE_COUNT || visits.len() > MAX_MODULE_STATE_COUNT {
        return Err(training("compiled module checkpoint count exceeds limit"));
    }
    let mut tensors = StateDict::from([(
        OPTIMIZER_TENSOR.to_owned(),
        TensorData::from_le_bytes([optimizer_bytes.len()], DType::U8, optimizer_bytes)?,
    )]);
    let mut metadata = Metadata::from([
        (
            "format".to_owned(),
            formats.encode(evaluation_capture_identity)?.to_owned(),
        ),
        ("state_count".to_owned(), states.len().to_string()),
        ("visit_count".to_owned(), visits.len().to_string()),
    ]);
    if let Some(identity) = evaluation_capture_identity {
        metadata.insert(
            "evaluation_capture_identity".to_owned(),
            identity.to_string(),
        );
    }
    for (index, state) in states.iter().enumerate() {
        metadata.insert(state_metadata_key(index, "name"), state.name.clone());
        metadata.insert(
            state_metadata_key(index, "kind"),
            state.kind.wire().to_owned(),
        );
        metadata.insert(
            state_metadata_key(index, "source_trainable"),
            state.source_trainable.to_string(),
        );
        metadata.insert(
            state_metadata_key(index, "policy_frozen"),
            state.policy_frozen.to_string(),
        );
        metadata.insert(
            state_metadata_key(index, "immutable"),
            state.value.is_some().to_string(),
        );
        if let Some(value) = &state.value {
            checked_bytes(value)?;
            tensors.insert(immutable_tensor_key(index), value.clone());
        }
    }
    for (index, visit) in visits.iter().enumerate() {
        metadata.insert(visit_metadata_key(index, "name"), visit.name.clone());
        metadata.insert(
            visit_metadata_key(index, "canonical"),
            visit.canonical_name.clone(),
        );
    }
    save_safetensors(&tensors, &metadata)
}

pub(super) fn encode_complete_module_checkpoint<C: CompiledModuleCheckpointPayload>(
    optimizer: &C,
    evaluation_capture_identity: Option<u64>,
    states: &[ModuleCheckpointState],
    visits: &[ModuleCheckpointVisit],
) -> Result<CompiledModuleCheckpoint<C>> {
    let optimizer_bytes = optimizer.encode_module_payload()?;
    let bytes = encode_module_checkpoint(
        module_formats::<C>(),
        optimizer_bytes.as_ref(),
        evaluation_capture_identity,
        states,
        visits,
    )?;
    CompiledModuleCheckpoint::from_bytes(bytes)
}

pub(super) fn decode_module_checkpoint<C>(
    bytes: &[u8],
    formats: ModuleCheckpointFormats,
    decode_optimizer: impl FnOnce(&[u8]) -> Result<(C, BTreeSet<String>)>,
) -> Result<DecodedModuleCheckpoint<C>> {
    let (mut tensors, metadata) = load_safetensors(bytes)?;
    let has_evaluation = formats.has_evaluation(metadata.get("format").map(String::as_str))?;
    let evaluation_capture_identity = has_evaluation
        .then(|| {
            metadata
                .get("evaluation_capture_identity")
                .ok_or_else(|| {
                    training("compiled module checkpoint evaluation identity is absent")
                })?
                .parse::<u64>()
                .map_err(|_| training("compiled module checkpoint evaluation identity is invalid"))
        })
        .transpose()?;
    let state_count = parse_count(&metadata, "state_count")?;
    let visit_count = parse_count(&metadata, "visit_count")?;
    if state_count == 0 || visit_count < state_count {
        return Err(training("compiled module checkpoint topology is empty"));
    }
    let optimizer_tensor = tensors
        .remove(OPTIMIZER_TENSOR)
        .ok_or_else(|| training("compiled module checkpoint optimizer state is absent"))?;
    if optimizer_tensor.dtype() != DType::U8 || optimizer_tensor.shape().rank() != 1 {
        return Err(training(
            "compiled module checkpoint optimizer tensor is invalid",
        ));
    }
    let (optimizer, optimizer_parameters) = decode_optimizer(&optimizer_tensor.to_le_bytes()?)?;

    let mut expected_metadata = BTreeSet::from([
        "format".to_owned(),
        "state_count".to_owned(),
        "visit_count".to_owned(),
    ]);
    if evaluation_capture_identity.is_some() {
        expected_metadata.insert("evaluation_capture_identity".to_owned());
    }
    let mut states = Vec::with_capacity(state_count);
    let mut state_names = BTreeSet::new();
    let mut trainable_names = BTreeSet::new();
    for index in 0..state_count {
        for field in [
            "name",
            "kind",
            "source_trainable",
            "policy_frozen",
            "immutable",
        ] {
            expected_metadata.insert(state_metadata_key(index, field));
        }
        let name = metadata
            .get(&state_metadata_key(index, "name"))
            .cloned()
            .ok_or_else(|| training("compiled module checkpoint state name is absent"))?;
        if !state_names.insert(name.clone()) {
            return Err(training("compiled module checkpoint state names repeat"));
        }
        let kind = ModuleCheckpointStateKind::parse(
            metadata
                .get(&state_metadata_key(index, "kind"))
                .ok_or_else(|| training("compiled module checkpoint state kind is absent"))?,
        )?;
        let source_trainable =
            parse_bool(&metadata, &state_metadata_key(index, "source_trainable"))?;
        let policy_frozen = parse_bool(&metadata, &state_metadata_key(index, "policy_frozen"))?;
        let immutable = parse_bool(&metadata, &state_metadata_key(index, "immutable"))?;
        if policy_frozen && (kind != ModuleCheckpointStateKind::Parameter || !source_trainable) {
            return Err(training(
                "compiled module checkpoint frozen policy is invalid",
            ));
        }
        let trainable =
            kind == ModuleCheckpointStateKind::Parameter && source_trainable && !policy_frozen;
        if immutable == trainable {
            return Err(training(
                "compiled module checkpoint state mutability is invalid",
            ));
        }
        let value = immutable
            .then(|| {
                tensors
                    .remove(&immutable_tensor_key(index))
                    .ok_or_else(|| training("compiled module checkpoint immutable value is absent"))
            })
            .transpose()?;
        if let Some(value) = &value {
            checked_bytes(value)?;
        } else {
            trainable_names.insert(name.clone());
            if !optimizer_parameters.contains(&name) {
                return Err(training(
                    "compiled module checkpoint trainable state is absent",
                ));
            }
        }
        states.push(ModuleCheckpointState {
            name,
            kind,
            source_trainable,
            policy_frozen,
            value,
        });
    }
    if optimizer_parameters != trainable_names {
        return Err(training(
            "compiled module checkpoint trainable inventory mismatch",
        ));
    }

    let mut visits = Vec::with_capacity(visit_count);
    let mut visit_names = BTreeSet::new();
    let mut first_visits = Vec::new();
    for index in 0..visit_count {
        for field in ["name", "canonical"] {
            expected_metadata.insert(visit_metadata_key(index, field));
        }
        let name = metadata
            .get(&visit_metadata_key(index, "name"))
            .cloned()
            .ok_or_else(|| training("compiled module checkpoint visit name is absent"))?;
        let canonical_name = metadata
            .get(&visit_metadata_key(index, "canonical"))
            .cloned()
            .ok_or_else(|| training("compiled module checkpoint canonical name is absent"))?;
        if !visit_names.insert(name.clone()) || !state_names.contains(&canonical_name) {
            return Err(training(
                "compiled module checkpoint visit topology is invalid",
            ));
        }
        if !visits
            .iter()
            .any(|visit: &ModuleCheckpointVisit| visit.canonical_name == canonical_name)
        {
            if name != canonical_name {
                return Err(training(
                    "compiled module checkpoint alias precedes canonical state",
                ));
            }
            first_visits.push(canonical_name.clone());
        }
        visits.push(ModuleCheckpointVisit {
            name,
            canonical_name,
        });
    }
    if first_visits
        != states
            .iter()
            .map(|state| state.name.clone())
            .collect::<Vec<_>>()
    {
        return Err(training(
            "compiled module checkpoint canonical order is invalid",
        ));
    }
    if metadata.keys().cloned().collect::<BTreeSet<_>>() != expected_metadata || !tensors.is_empty()
    {
        return Err(training("compiled module checkpoint schema mismatch"));
    }
    Ok(DecodedModuleCheckpoint {
        optimizer,
        evaluation_capture_identity,
        states,
        visits,
    })
}
