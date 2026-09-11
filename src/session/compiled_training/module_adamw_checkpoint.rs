use super::{CompiledAdamWCheckpoint, checked_bytes, decode_adamw_checkpoint, training};
use crate::{DType, Metadata, Result, StateDict, TensorData, load_safetensors, save_safetensors};
use std::collections::BTreeSet;

const MODULE_ADAMW_CHECKPOINT_FORMAT_V1: &str = "rustgrad-compiled-module-adamw-v1";
const MODULE_ADAMW_CHECKPOINT_FORMAT_V2: &str = "rustgrad-compiled-module-adamw-v2";
const OPTIMIZER_TENSOR: &str = "optimizer_checkpoint";
const MAX_MODULE_STATE_COUNT: usize = 1 << 20;

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
pub(super) struct DecodedModuleAdamWCheckpoint {
    pub(super) optimizer: CompiledAdamWCheckpoint,
    pub(super) evaluation_capture_identity: Option<u64>,
    pub(super) states: Vec<ModuleCheckpointState>,
    pub(super) visits: Vec<ModuleCheckpointVisit>,
}

/// Portable state for one complete owned compiled-module AdamW lifecycle.
///
/// The embedded [`CompiledAdamWCheckpoint`] bytes are preserved exactly. The
/// surrounding deterministic safetensors envelope adds only canonical module
/// topology and deduplicated immutable parameter/buffer values. When an
/// evaluator is attached, v2 also authenticates its capture identity so fresh
/// recompilation cannot silently attach a different read-only program.
/// Executable graphs, runtime resources, host identities, and versions are not
/// serialized. A lifecycle without evaluation retains its exact v1 bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledModuleAdamWCheckpoint {
    bytes: Vec<u8>,
    optimizer: CompiledAdamWCheckpoint,
    evaluation_capture_identity: Option<u64>,
}

impl CompiledModuleAdamWCheckpoint {
    /// Validates and owns deterministic complete-module checkpoint bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        let decoded = decode_module_adamw_checkpoint(&bytes)?;
        Ok(Self {
            bytes,
            optimizer: decoded.optimizer,
            evaluation_capture_identity: decoded.evaluation_capture_identity,
        })
    }

    /// Returns the unchanged embedded optimizer checkpoint.
    pub fn optimizer_checkpoint(&self) -> &CompiledAdamWCheckpoint {
        &self.optimizer
    }

    /// Required evaluator capture identity for a v2 checkpoint.
    ///
    /// `None` denotes a v1 envelope with no authenticated evaluator requirement,
    /// including checkpoints written by older versions.
    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation_capture_identity
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
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

pub(super) fn encode_module_adamw_checkpoint(
    optimizer: &CompiledAdamWCheckpoint,
    evaluation_capture_identity: Option<u64>,
    states: &[ModuleCheckpointState],
    visits: &[ModuleCheckpointVisit],
) -> Result<CompiledModuleAdamWCheckpoint> {
    if states.len() > MAX_MODULE_STATE_COUNT || visits.len() > MAX_MODULE_STATE_COUNT {
        return Err(training("compiled module checkpoint count exceeds limit"));
    }
    let optimizer_bytes = optimizer.as_bytes();
    let mut tensors = StateDict::from([(
        OPTIMIZER_TENSOR.to_owned(),
        TensorData::from_le_bytes([optimizer_bytes.len()], DType::U8, optimizer_bytes)?,
    )]);
    let format = if evaluation_capture_identity.is_some() {
        MODULE_ADAMW_CHECKPOINT_FORMAT_V2
    } else {
        MODULE_ADAMW_CHECKPOINT_FORMAT_V1
    };
    let mut metadata = Metadata::from([
        ("format".to_owned(), format.to_owned()),
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
    let bytes = save_safetensors(&tensors, &metadata)?;
    CompiledModuleAdamWCheckpoint::from_bytes(bytes)
}

pub(super) fn decode_module_adamw_checkpoint(bytes: &[u8]) -> Result<DecodedModuleAdamWCheckpoint> {
    let (mut tensors, metadata) = load_safetensors(bytes)?;
    let evaluation_capture_identity = match metadata.get("format").map(String::as_str) {
        Some(MODULE_ADAMW_CHECKPOINT_FORMAT_V1) => None,
        Some(MODULE_ADAMW_CHECKPOINT_FORMAT_V2) => Some(
            metadata
                .get("evaluation_capture_identity")
                .ok_or_else(|| {
                    training("compiled module checkpoint evaluation identity is absent")
                })?
                .parse::<u64>()
                .map_err(|_| {
                    training("compiled module checkpoint evaluation identity is invalid")
                })?,
        ),
        _ => return Err(training("compiled module checkpoint format mismatch")),
    };
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
    let optimizer = CompiledAdamWCheckpoint::from_bytes(optimizer_tensor.to_le_bytes()?)?;
    let optimizer_parameters = decode_adamw_checkpoint(optimizer.as_bytes())?.parameters;

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
            if !optimizer_parameters.contains_key(&name) {
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
    if optimizer_parameters.keys().collect::<BTreeSet<_>>()
        != trainable_names.iter().collect::<BTreeSet<_>>()
    {
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
    Ok(DecodedModuleAdamWCheckpoint {
        optimizer,
        evaluation_capture_identity,
        states,
        visits,
    })
}
