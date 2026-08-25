//! Bounded local ONNX plus named NPY workflow adapters.
//!
//! Parsing/lowering remains in the parent module and NPY parsing/writing
//! remains in `interop::host`; this module owns only filesystem limits, named
//! path validation, and orchestration.

use super::{NativeOnnxInferenceResult, NativeOnnxManyInferenceResult, OnnxModel, import_onnx};
use crate::{
    CapturedReplayExecutor, Error, TensorData,
    interop::host::{
        NpyFileError, NpyReadLimits, encode_npy, load_npy_file_with_limits, save_npy_file,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
static FAIL_BATCH_REPLACEMENT: AtomicUsize = AtomicUsize::new(0);

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
    Input {
        name: String,
        error: NpyFileError,
    },
    Run(Error),
    Native(Error),
    MissingInput(String),
    UnexpectedInput(String),
    NoOutputsSelected,
    UnknownOutput(String),
    DuplicateOutputPath(PathBuf),
    Output {
        name: String,
        error: NpyFileError,
    },
    /// A replacement failed and rollback or staging cleanup also reported
    /// filesystem failures. The primary failure is retained alongside every
    /// cleanup failure rather than being silently discarded.
    OutputTransaction {
        name: String,
        error: NpyFileError,
        cleanup: Vec<BatchFilesystemFailure>,
    },
    /// All replacements completed, but obsolete rollback artifacts could not
    /// be removed. The visible outputs are valid; the cleanup failure remains
    /// explicit for the caller.
    OutputCleanup {
        cleanup: Vec<BatchFilesystemFailure>,
    },
}
impl fmt::Display for OnnxWorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ONNX workflow error: {self:?}")
    }
}
impl std::error::Error for OnnxWorkflowError {}

/// One filesystem failure encountered while rolling back or cleaning up a
/// multi-output replacement transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchFilesystemFailure {
    pub path: PathBuf,
    pub operation: &'static str,
    pub kind: io::ErrorKind,
}

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

/// Executes selected exact named NPY inputs once through strict native replay,
/// then commits all selected NPY outputs as one same-directory rollback batch.
pub fn run_onnx_files_native_many(
    model_path: impl AsRef<Path>,
    inputs: &NamedPaths,
    outputs: &NamedPaths,
    limits: OnnxWorkflowLimits,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<NativeOnnxManyInferenceResult, OnnxWorkflowError> {
    let model =
        load_onnx_file_with_limits(model_path, limits.onnx).map_err(OnnxWorkflowError::Model)?;
    validate_input_names(&model, inputs)?;
    validate_outputs(&model, outputs)?;
    let tensors = inputs
        .iter()
        .map(|(name, path)| {
            load_npy_file_with_limits(path, limits.npy)
                .map(|value| (name.into(), value))
                .map_err(|error| OnnxWorkflowError::Input {
                    name: name.into(),
                    error,
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let selected = outputs.names().cloned().collect();
    let result = model
        .run_native_static_many(&tensors, &selected, executor, vectorized)
        .map_err(OnnxWorkflowError::Native)?;
    save_npy_batch(outputs, result.outputs())?;
    Ok(result)
}

fn save_npy_batch(
    outputs: &NamedPaths,
    values: &BTreeMap<String, TensorData>,
) -> Result<(), OnnxWorkflowError> {
    let mut staged = Vec::new();
    for (name, path) in outputs.iter() {
        let bytes = encode_npy(&values[name]).map_err(|error| OnnxWorkflowError::Output {
            name: name.into(),
            error: NpyFileError::Format(error),
        })?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let file = path
            .file_name()
            .and_then(|x| x.to_str())
            .ok_or(OnnxWorkflowError::Output {
                name: name.into(),
                error: NpyFileError::Io {
                    operation: "validate path",
                    kind: io::ErrorKind::InvalidInput,
                },
            })?;
        if path.exists() && !path.is_file() {
            return Err(batch_output_error(
                name,
                "replace",
                io::ErrorKind::InvalidInput,
                cleanup_staged(&staged),
            ));
        }
        let temp = parent.join(format!(
            ".{file}.rustgrad-batch-{}-{}.tmp",
            std::process::id(),
            staged.len()
        ));
        let mut handle = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| {
                batch_output_error(
                    name,
                    "create staging file",
                    e.kind(),
                    cleanup_staged(&staged),
                )
            })?;
        if let Err(e) = handle.write_all(&bytes).and_then(|_| handle.sync_all()) {
            let mut cleanup = remove_batch_file(&temp, "remove staging file");
            cleanup.extend(cleanup_staged(&staged));
            return Err(batch_output_error(name, "write", e.kind(), cleanup));
        }
        staged.push((name.to_owned(), path.to_path_buf(), temp));
    }
    preflight_batch_backups(&staged)?;
    let mut committed = Vec::new();
    for (name, path, temp) in &staged {
        let backup = path.with_extension(format!("rustgrad-batch-{}.bak", committed.len()));
        let had_old = path.exists();
        if had_old && let Err(e) = fs::rename(path, &backup) {
            let mut cleanup = rollback_batch(&committed);
            cleanup.extend(cleanup_staged(&staged));
            return Err(batch_output_error(name, "backup", e.kind(), cleanup));
        }
        if let Err(e) = fs::rename(temp, path) {
            let mut cleanup = Vec::new();
            if had_old {
                cleanup.extend(rename_batch_file(&backup, path, "restore current backup"));
            }
            cleanup.extend(rollback_batch(&committed));
            cleanup.extend(cleanup_staged(&staged));
            return Err(batch_output_error(name, "replace", e.kind(), cleanup));
        }
        #[cfg(test)]
        if FAIL_BATCH_REPLACEMENT.load(Ordering::Relaxed) == committed.len() + 1 {
            FAIL_BATCH_REPLACEMENT.store(0, Ordering::Relaxed);
            let mut cleanup = Vec::new();
            if had_old {
                cleanup.extend(rename_batch_file(&backup, path, "restore injected backup"));
            }
            cleanup.extend(rollback_batch(&committed));
            cleanup.extend(cleanup_staged(&staged));
            return Err(batch_output_error(
                name,
                "injected replace",
                io::ErrorKind::Other,
                cleanup,
            ));
        }
        committed.push((path.clone(), backup, had_old));
    }
    let mut cleanup = Vec::new();
    for (_, backup, had_old) in committed {
        if had_old {
            cleanup.extend(remove_batch_file(&backup, "remove committed backup"));
        }
    }
    if !cleanup.is_empty() {
        return Err(OnnxWorkflowError::OutputCleanup { cleanup });
    }
    Ok(())
}

fn preflight_batch_backups(staged: &[(String, PathBuf, PathBuf)]) -> Result<(), OnnxWorkflowError> {
    for (index, (name, path, _)) in staged.iter().enumerate() {
        let backup = path.with_extension(format!("rustgrad-batch-{index}.bak"));
        if backup.exists() {
            return Err(batch_output_error(
                name,
                "reserve backup",
                io::ErrorKind::AlreadyExists,
                cleanup_staged(staged),
            ));
        }
    }
    Ok(())
}

fn batch_output_error(
    name: &str,
    operation: &'static str,
    kind: io::ErrorKind,
    cleanup: Vec<BatchFilesystemFailure>,
) -> OnnxWorkflowError {
    let error = NpyFileError::Io { operation, kind };
    if cleanup.is_empty() {
        OnnxWorkflowError::Output {
            name: name.into(),
            error,
        }
    } else {
        OnnxWorkflowError::OutputTransaction {
            name: name.into(),
            error,
            cleanup,
        }
    }
}

fn rollback_batch(committed: &[(PathBuf, PathBuf, bool)]) -> Vec<BatchFilesystemFailure> {
    let mut cleanup = Vec::new();
    for (path, backup, had_old) in committed.iter().rev() {
        cleanup.extend(remove_batch_file(path, "remove replaced output"));
        if *had_old {
            cleanup.extend(rename_batch_file(backup, path, "restore backup"));
        }
    }
    cleanup
}

fn cleanup_staged(staged: &[(String, PathBuf, PathBuf)]) -> Vec<BatchFilesystemFailure> {
    let mut cleanup = Vec::new();
    for (_, _, temp) in staged {
        cleanup.extend(remove_batch_file(temp, "remove staging file"));
    }
    cleanup
}

fn remove_batch_file(path: &Path, operation: &'static str) -> Vec<BatchFilesystemFailure> {
    match fs::remove_file(path) {
        Ok(()) => Vec::new(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => vec![BatchFilesystemFailure {
            path: path.to_path_buf(),
            operation,
            kind: error.kind(),
        }],
    }
}

fn rename_batch_file(
    from: &Path,
    _to: &Path,
    operation: &'static str,
) -> Vec<BatchFilesystemFailure> {
    match fs::rename(from, _to) {
        Ok(()) => Vec::new(),
        Err(error) => vec![BatchFilesystemFailure {
            path: from.to_path_buf(),
            operation,
            kind: error.kind(),
        }],
    }
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

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::DType;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rustgrad-onnx-batch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        path
    }
    #[test]
    fn second_replacement_failure_restores_every_target_and_retries() {
        let d = dir();
        let a = d.join("a.npy");
        let b = d.join("b.npy");
        fs::write(&a, b"old-a").unwrap();
        fs::write(&b, b"old-b").unwrap();
        let paths =
            NamedPaths::new(vec![("a".into(), a.clone()), ("b".into(), b.clone())]).unwrap();
        let values = BTreeMap::from([
            (
                "a".into(),
                TensorData::from_le_bytes([1], DType::U8, &[1]).unwrap(),
            ),
            (
                "b".into(),
                TensorData::from_le_bytes([1], DType::U8, &[2]).unwrap(),
            ),
        ]);
        FAIL_BATCH_REPLACEMENT.store(2, Ordering::Relaxed);
        assert!(save_npy_batch(&paths, &values).is_err());
        assert_eq!(fs::read(&a).unwrap(), b"old-a");
        assert_eq!(fs::read(&b).unwrap(), b"old-b");
        assert!(!fs::read_dir(&d).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .contains("rustgrad-batch")
        }));
        save_npy_batch(&paths, &values).unwrap();
        assert_eq!(
            crate::interop::host::load_npy_file(&a).unwrap(),
            values["a"]
        );
        assert_eq!(
            crate::interop::host::load_npy_file(&b).unwrap(),
            values["b"]
        );
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn composite_transaction_errors_preserve_primary_and_cleanup_failures() {
        let cleanup = vec![
            BatchFilesystemFailure {
                path: PathBuf::from("first.tmp"),
                operation: "remove staging file",
                kind: io::ErrorKind::PermissionDenied,
            },
            BatchFilesystemFailure {
                path: PathBuf::from("second.bak"),
                operation: "restore backup",
                kind: io::ErrorKind::Other,
            },
        ];
        let error = batch_output_error(
            "second",
            "replace",
            io::ErrorKind::AlreadyExists,
            cleanup.clone(),
        );
        match error {
            OnnxWorkflowError::OutputTransaction {
                name,
                error: NpyFileError::Io { operation, kind },
                cleanup: actual,
            } => {
                assert_eq!(name, "second");
                assert_eq!(operation, "replace");
                assert_eq!(kind, io::ErrorKind::AlreadyExists);
                assert_eq!(actual, cleanup);
            }
            other => panic!("expected composite transaction error, got {other:?}"),
        }
    }

    #[test]
    fn orphaned_backup_rejects_before_any_replacement() {
        let d = dir();
        let output = d.join("output.npy");
        let backup = output.with_extension("rustgrad-batch-0.bak");
        fs::write(&output, b"old-output").unwrap();
        fs::write(&backup, b"orphaned-backup").unwrap();
        let paths = NamedPaths::new(vec![("output".into(), output.clone())]).unwrap();
        let values = BTreeMap::from([(
            "output".into(),
            TensorData::from_le_bytes([1], DType::U8, &[7]).unwrap(),
        )]);

        let error = save_npy_batch(&paths, &values).unwrap_err();
        match error {
            OnnxWorkflowError::Output {
                name,
                error:
                    NpyFileError::Io {
                        operation: "reserve backup",
                        kind: io::ErrorKind::AlreadyExists,
                    },
            } => assert_eq!(name, "output"),
            other => panic!("expected backup preflight error, got {other:?}"),
        }
        assert_eq!(fs::read(&output).unwrap(), b"old-output");
        assert_eq!(fs::read(&backup).unwrap(), b"orphaned-backup");
        assert_eq!(
            fs::read_dir(&d)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
                .count(),
            0
        );
        fs::remove_dir_all(d).unwrap();
    }
}
