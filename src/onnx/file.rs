//! Bounded local ONNX plus named NPY workflow adapters.
//!
//! Parsing/lowering remains in the parent module and NPY parsing/writing
//! remains in `interop::host`; this module owns only filesystem limits, named
//! path validation, and orchestration.

use super::{NativeOnnxInferenceResult, OnnxModel, import_onnx};
use crate::{
    CapturedReplayExecutor, Error, TensorData,
    interop::host::{NpyFileError, NpyReadLimits, load_npy_file_with_limits, save_npy_file},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

/// Explicit bound for a local ONNX model file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnnxReadLimits {
    pub max_model_bytes: usize,
}
impl Default for OnnxReadLimits {
    fn default() -> Self {
        Self {
            max_model_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Typed failure while obtaining a local ONNX model.
#[derive(Debug)]
pub enum OnnxFileError {
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
    Limit {
        actual: usize,
        maximum: usize,
    },
    Model(Error),
}
impl fmt::Display for OnnxFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => write!(f, "ONNX file {operation} failed: {kind:?}"),
            Self::Limit { actual, maximum } => {
                write!(f, "ONNX file has {actual} bytes, exceeding limit {maximum}")
            }
            Self::Model(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for OnnxFileError {}

fn io_error(operation: &'static str, error: io::Error) -> OnnxFileError {
    OnnxFileError::Io {
        operation,
        kind: error.kind(),
    }
}

/// Imports a local static ONNX model using the default 32 MiB bound.
pub fn load_onnx_file(path: impl AsRef<Path>) -> Result<OnnxModel, OnnxFileError> {
    load_onnx_file_with_limits(path, OnnxReadLimits::default())
}

/// Imports a local static ONNX model under an explicit byte limit. A metadata
/// preflight and a capped stream read both reject oversized files before they
/// reach the canonical protobuf parser/lowerer.
pub fn load_onnx_file_with_limits(
    path: impl AsRef<Path>,
    limits: OnnxReadLimits,
) -> Result<OnnxModel, OnnxFileError> {
    let path = path.as_ref();
    let metadata = fs::metadata(path).map_err(|error| io_error("inspect", error))?;
    if metadata.len() > u64::try_from(limits.max_model_bytes).unwrap_or(u64::MAX) {
        return Err(OnnxFileError::Limit {
            actual: usize::try_from(metadata.len()).unwrap_or(usize::MAX),
            maximum: limits.max_model_bytes,
        });
    }
    let file = fs::File::open(path).map_err(|error| io_error("open", error))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(limits.max_model_bytes.min(64 << 10))
        .map_err(|_| {
            OnnxFileError::Model(Error::ModelIo {
                reason: "ONNX input allocation failed".into(),
            })
        })?;
    file.take(
        u64::try_from(limits.max_model_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    )
    .read_to_end(&mut bytes)
    .map_err(|error| io_error("read", error))?;
    if bytes.len() > limits.max_model_bytes {
        return Err(OnnxFileError::Limit {
            actual: bytes.len(),
            maximum: limits.max_model_bytes,
        });
    }
    import_onnx(&bytes).map_err(OnnxFileError::Model)
}

/// A deterministic name-to-path map. Construction rejects empty or duplicate
/// names before any input read or output write starts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NamedPaths(BTreeMap<String, PathBuf>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NamedPathsError {
    EmptyName,
    DuplicateName(String),
}
impl fmt::Display for NamedPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "named paths error: {self:?}")
    }
}
impl std::error::Error for NamedPathsError {}
impl NamedPaths {
    pub fn new(
        entries: impl IntoIterator<Item = (String, PathBuf)>,
    ) -> Result<Self, NamedPathsError> {
        let mut paths = BTreeMap::new();
        for (name, path) in entries {
            if name.is_empty() {
                return Err(NamedPathsError::EmptyName);
            }
            if paths.insert(name.clone(), path).is_some() {
                return Err(NamedPathsError::DuplicateName(name));
            }
        }
        Ok(Self(paths))
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Path)> {
        self.0
            .iter()
            .map(|(name, path)| (name.as_str(), path.as_path()))
    }
    fn names(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }
}

/// Limits for the composed local ONNX/NPY workflow.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OnnxWorkflowLimits {
    pub onnx: OnnxReadLimits,
    pub npy: NpyReadLimits,
}

/// Typed local workflow failure without hiding parser, NPY, or graph errors.
#[derive(Debug)]
pub enum OnnxWorkflowError {
    Model(OnnxFileError),
    Input { name: String, error: NpyFileError },
    Run(Error),
    Native(Error),
    MissingInput(String),
    UnexpectedInput(String),
    NoOutputsSelected,
    UnknownOutput(String),
    DuplicateOutputPath(PathBuf),
    Output { name: String, error: NpyFileError },
}
impl fmt::Display for OnnxWorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ONNX workflow error: {self:?}")
    }
}
impl std::error::Error for OnnxWorkflowError {}

/// Imports `model_path`, loads exact named NPY inputs, executes the existing
/// static CPU model, and stages selected named NPY outputs. All name and output
/// path checks precede input reads and model execution.
pub fn run_onnx_files(
    model_path: impl AsRef<Path>,
    inputs: &NamedPaths,
    outputs: &NamedPaths,
    limits: OnnxWorkflowLimits,
) -> Result<BTreeMap<String, TensorData>, OnnxWorkflowError> {
    let model =
        load_onnx_file_with_limits(model_path, limits.onnx).map_err(OnnxWorkflowError::Model)?;
    validate_input_names(&model, inputs)?;
    validate_outputs(&model, outputs)?;
    let mut tensors = BTreeMap::new();
    for (name, path) in inputs.iter() {
        let value = load_npy_file_with_limits(path, limits.npy).map_err(|error| {
            OnnxWorkflowError::Input {
                name: name.into(),
                error,
            }
        })?;
        tensors.insert(name.into(), value);
    }
    let values = model.run_named(&tensors).map_err(OnnxWorkflowError::Run)?;
    for (name, path) in outputs.iter() {
        save_npy_file(path, &values[name]).map_err(|error| OnnxWorkflowError::Output {
            name: name.into(),
            error,
        })?;
    }
    Ok(values)
}

/// Imports one bounded static ONNX model, reads its exact named NPY input,
/// executes the strict native one-input/one-output subset, then atomically
/// stages the sole selected NPY output. No output write begins until strict
/// native schedule/capture/preflight/execution has succeeded.
pub fn run_onnx_files_native(
    model_path: impl AsRef<Path>,
    inputs: &NamedPaths,
    outputs: &NamedPaths,
    limits: OnnxWorkflowLimits,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<NativeOnnxInferenceResult, OnnxWorkflowError> {
    let model =
        load_onnx_file_with_limits(model_path, limits.onnx).map_err(OnnxWorkflowError::Model)?;
    validate_input_names(&model, inputs)?;
    validate_native_output(&model, outputs)?;
    let mut tensors = BTreeMap::new();
    for (name, path) in inputs.iter() {
        let value = load_npy_file_with_limits(path, limits.npy).map_err(|error| {
            OnnxWorkflowError::Input {
                name: name.into(),
                error,
            }
        })?;
        tensors.insert(name.into(), value);
    }
    let result = model
        .run_native_static(&tensors, executor, vectorized)
        .map_err(OnnxWorkflowError::Native)?;
    let output_path = outputs
        .iter()
        .next()
        .map(|(_, path)| path)
        .ok_or(OnnxWorkflowError::NoOutputsSelected)?;
    save_npy_file(output_path, result.output()).map_err(|error| OnnxWorkflowError::Output {
        name: result.output_name().into(),
        error,
    })?;
    Ok(result)
}

fn validate_input_names(model: &OnnxModel, inputs: &NamedPaths) -> Result<(), OnnxWorkflowError> {
    for expected in model.inputs() {
        if !inputs.0.contains_key(expected) {
            return Err(OnnxWorkflowError::MissingInput(expected.into()));
        }
    }
    for actual in inputs.names() {
        if !model.inputs().any(|expected| expected == actual) {
            return Err(OnnxWorkflowError::UnexpectedInput(actual.clone()));
        }
    }
    Ok(())
}

fn validate_outputs(model: &OnnxModel, outputs: &NamedPaths) -> Result<(), OnnxWorkflowError> {
    if outputs.0.is_empty() {
        return Err(OnnxWorkflowError::NoOutputsSelected);
    }
    let mut destinations = BTreeSet::new();
    for (name, path) in outputs.iter() {
        if !model.outputs().any(|output| output == name) {
            return Err(OnnxWorkflowError::UnknownOutput(name.into()));
        }
        if !destinations.insert(path.to_path_buf()) {
            return Err(OnnxWorkflowError::DuplicateOutputPath(path.to_path_buf()));
        }
    }
    Ok(())
}

fn validate_native_output(
    model: &OnnxModel,
    outputs: &NamedPaths,
) -> Result<(), OnnxWorkflowError> {
    validate_outputs(model, outputs)?;
    if model.outputs().count() != 1 || outputs.0.len() != 1 {
        return Err(OnnxWorkflowError::Native(Error::ModelIo {
            reason: "strict native ONNX requires selecting exactly one sole output".into(),
        }));
    }
    Ok(())
}
