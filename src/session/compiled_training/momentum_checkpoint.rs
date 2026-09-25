//! Portable momentum-SGD recurrent-state checkpoints.

use super::{checked_bytes, training, validate_user_name};
use crate::safetensors::{read_safetensors_file_bytes_with_limits, save_safetensors_file_bytes};
use crate::{
    DType, Metadata, Result, SafetensorsFileError, SafetensorsReadLimits, StateDict, TensorData,
    load_safetensors, save_safetensors,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const MOMENTUM_SGD_CHECKPOINT_FORMAT_V1: &str = "rustgrad-compiled-momentum-sgd-v1";
const MAX_PARAMETER_COUNT: usize = 1 << 20;

/// Exact persistent frontier of one compiled CPU momentum-SGD program.
///
/// The checkpoint is independent of the runtime's host module identity. A
/// matching program may validate it against a freshly initialized module,
/// restore parameter and momentum values with their logical versions, and
/// continue replay without publishing into that module until finalization.
/// Its deterministic safetensors encoding contains only detached F32 state,
/// logical versions, replay progress, and the authenticated capture identity;
/// executable graphs, schedules, runtime resources, and module identities are
/// deliberately absent.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledMomentumSgdCheckpoint {
    pub(super) capture_identity: u64,
    pub(super) step: u64,
    pub(super) parameters: BTreeMap<String, TensorData>,
    pub(super) momenta: BTreeMap<String, TensorData>,
    pub(super) parameter_versions: BTreeMap<String, u64>,
    pub(super) momentum_versions: BTreeMap<String, u64>,
}

impl CompiledMomentumSgdCheckpoint {
    /// Validates and decodes one deterministic checkpoint payload.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        decode_momentum_sgd_checkpoint(bytes)
    }

    /// Encodes this detached frontier deterministically.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        encode_momentum_sgd_checkpoint(self)
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
        Self::from_bytes(&bytes).map_err(SafetensorsFileError::Format)
    }

    /// Atomically replaces `path` with a deterministic checkpoint payload.
    pub fn save_file(&self, path: impl AsRef<Path>) -> Result<()> {
        save_safetensors_file_bytes(path, &self.to_bytes()?)
    }

    pub const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub const fn step(&self) -> u64 {
        self.step
    }

    pub fn parameters(&self) -> &BTreeMap<String, TensorData> {
        &self.parameters
    }

    pub fn momenta(&self) -> &BTreeMap<String, TensorData> {
        &self.momenta
    }

    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }

    pub fn momentum_versions(&self) -> &BTreeMap<String, u64> {
        &self.momentum_versions
    }
}

fn metadata_key(index: usize, field: &str) -> String {
    format!("state.{index}.{field}")
}

fn tensor_key(index: usize, field: &str) -> String {
    format!("state.{index}.{field}")
}

fn parse_u64(metadata: &Metadata, key: &str, reason: &'static str) -> Result<u64> {
    metadata
        .get(key)
        .ok_or_else(|| training(reason))?
        .parse::<u64>()
        .map_err(|_| training(reason))
}

fn validate_checkpoint(checkpoint: &CompiledMomentumSgdCheckpoint) -> Result<()> {
    let parameter_count = checkpoint.parameters.len();
    if parameter_count == 0 {
        return Err(training(
            "compiled momentum-SGD checkpoint parameter inventory is empty",
        ));
    }
    if parameter_count > MAX_PARAMETER_COUNT {
        return Err(training(
            "compiled momentum-SGD checkpoint parameter count exceeds limit",
        ));
    }
    if checkpoint.parameters.keys().ne(checkpoint.momenta.keys())
        || checkpoint
            .parameters
            .keys()
            .ne(checkpoint.parameter_versions.keys())
        || checkpoint
            .parameters
            .keys()
            .ne(checkpoint.momentum_versions.keys())
    {
        return Err(training(
            "compiled momentum-SGD checkpoint state names mismatch",
        ));
    }
    for (name, parameter) in &checkpoint.parameters {
        validate_user_name(name, "parameter")?;
        let momentum = &checkpoint.momenta[name];
        if parameter.dtype() != DType::F32
            || momentum.dtype() != DType::F32
            || parameter.shape() != momentum.shape()
        {
            return Err(training(
                "compiled momentum-SGD checkpoint state descriptor mismatch",
            ));
        }
        checked_bytes(parameter)?;
        checked_bytes(momentum)?;
    }
    Ok(())
}

fn encode_momentum_sgd_checkpoint(checkpoint: &CompiledMomentumSgdCheckpoint) -> Result<Vec<u8>> {
    validate_checkpoint(checkpoint)?;
    let mut tensors = StateDict::new();
    let mut metadata = Metadata::from([
        (
            "format".to_owned(),
            MOMENTUM_SGD_CHECKPOINT_FORMAT_V1.to_owned(),
        ),
        (
            "capture_identity".to_owned(),
            checkpoint.capture_identity.to_string(),
        ),
        ("step".to_owned(), checkpoint.step.to_string()),
        (
            "parameter_count".to_owned(),
            checkpoint.parameters.len().to_string(),
        ),
    ]);
    for (index, (name, parameter)) in checkpoint.parameters.iter().enumerate() {
        metadata.insert(metadata_key(index, "name"), name.clone());
        metadata.insert(
            metadata_key(index, "parameter_version"),
            checkpoint.parameter_versions[name].to_string(),
        );
        metadata.insert(
            metadata_key(index, "momentum_version"),
            checkpoint.momentum_versions[name].to_string(),
        );
        tensors.insert(tensor_key(index, "parameter"), parameter.clone());
        tensors.insert(
            tensor_key(index, "momentum"),
            checkpoint.momenta[name].clone(),
        );
    }
    save_safetensors(&tensors, &metadata)
}

fn decode_momentum_sgd_checkpoint(bytes: &[u8]) -> Result<CompiledMomentumSgdCheckpoint> {
    let (mut tensors, metadata) = load_safetensors(bytes)?;
    if metadata.get("format").map(String::as_str) != Some(MOMENTUM_SGD_CHECKPOINT_FORMAT_V1) {
        return Err(training("compiled momentum-SGD checkpoint format mismatch"));
    }
    let capture_identity = parse_u64(
        &metadata,
        "capture_identity",
        "compiled momentum-SGD checkpoint capture identity is invalid",
    )?;
    let step = parse_u64(
        &metadata,
        "step",
        "compiled momentum-SGD checkpoint step is invalid",
    )?;
    let parameter_count = metadata
        .get("parameter_count")
        .ok_or_else(|| training("compiled momentum-SGD checkpoint parameter count is absent"))?
        .parse::<usize>()
        .map_err(|_| training("compiled momentum-SGD checkpoint parameter count is invalid"))?;
    if parameter_count == 0 {
        return Err(training(
            "compiled momentum-SGD checkpoint parameter inventory is empty",
        ));
    }
    if parameter_count > MAX_PARAMETER_COUNT {
        return Err(training(
            "compiled momentum-SGD checkpoint parameter count exceeds limit",
        ));
    }

    let mut expected_metadata = BTreeSet::from([
        "format".to_owned(),
        "capture_identity".to_owned(),
        "step".to_owned(),
        "parameter_count".to_owned(),
    ]);
    let mut names = BTreeSet::new();
    let mut previous_name = None::<String>;
    let mut parameters = BTreeMap::new();
    let mut momenta = BTreeMap::new();
    let mut parameter_versions = BTreeMap::new();
    let mut momentum_versions = BTreeMap::new();
    for index in 0..parameter_count {
        for field in ["name", "parameter_version", "momentum_version"] {
            expected_metadata.insert(metadata_key(index, field));
        }
        let name = metadata
            .get(&metadata_key(index, "name"))
            .cloned()
            .ok_or_else(|| training("compiled momentum-SGD checkpoint state name is absent"))?;
        validate_user_name(&name, "parameter")?;
        if !names.insert(name.clone())
            || previous_name
                .as_ref()
                .is_some_and(|previous| previous >= &name)
        {
            return Err(training(
                "compiled momentum-SGD checkpoint state order is invalid",
            ));
        }
        previous_name = Some(name.clone());
        let parameter = tensors
            .remove(&tensor_key(index, "parameter"))
            .ok_or_else(|| training("compiled momentum-SGD checkpoint parameter is absent"))?;
        let momentum = tensors
            .remove(&tensor_key(index, "momentum"))
            .ok_or_else(|| training("compiled momentum-SGD checkpoint momentum is absent"))?;
        if parameter.dtype() != DType::F32
            || momentum.dtype() != DType::F32
            || parameter.shape() != momentum.shape()
        {
            return Err(training(
                "compiled momentum-SGD checkpoint state descriptor mismatch",
            ));
        }
        checked_bytes(&parameter)?;
        checked_bytes(&momentum)?;
        let parameter_version = parse_u64(
            &metadata,
            &metadata_key(index, "parameter_version"),
            "compiled momentum-SGD checkpoint parameter version is invalid",
        )?;
        let momentum_version = parse_u64(
            &metadata,
            &metadata_key(index, "momentum_version"),
            "compiled momentum-SGD checkpoint momentum version is invalid",
        )?;
        parameters.insert(name.clone(), parameter);
        momenta.insert(name.clone(), momentum);
        parameter_versions.insert(name.clone(), parameter_version);
        momentum_versions.insert(name, momentum_version);
    }
    if metadata.keys().cloned().collect::<BTreeSet<_>>() != expected_metadata || !tensors.is_empty()
    {
        return Err(training("compiled momentum-SGD checkpoint schema mismatch"));
    }
    let checkpoint = CompiledMomentumSgdCheckpoint {
        capture_identity,
        step,
        parameters,
        momenta,
        parameter_versions,
        momentum_versions,
    };
    validate_checkpoint(&checkpoint)?;
    Ok(checkpoint)
}
