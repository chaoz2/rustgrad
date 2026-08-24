use super::{
    FuzzComparison, FuzzFailureArtifact, FuzzPath, FuzzReplayStatus, MAX_FUZZ_ARTIFACT_FILE_BYTES,
    regression_cases, replay_failure, run_case,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Explicit mutation policy for regression corpus reconciliation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FuzzCorpusMode {
    /// Inventory and classify without changing the filesystem.
    Check,
    /// Atomically add newly observed failure artifacts.
    Write,
    /// Atomically add failures and remove artifacts proven resolved.
    WriteAndPruneResolved,
}

/// Lifecycle classification for one deterministic corpus identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FuzzCorpusState {
    Reproduced,
    New,
    Changed,
    Resolved,
    Unsupported,
}

/// One inventoried or newly observed failure identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FuzzCorpusRecord {
    pub identity: u64,
    pub previous_identity: Option<u64>,
    pub state: FuzzCorpusState,
}

/// Complete lifecycle accounting for one regression corpus pass.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FuzzCorpusReport {
    pub regressions: usize,
    pub inventoried: usize,
    pub current_failures: usize,
    pub unresolved: usize,
    pub reproduced: usize,
    pub new: usize,
    pub changed: usize,
    pub resolved: usize,
    pub unsupported: usize,
    pub written: usize,
    pub pruned: usize,
    pub records: Vec<FuzzCorpusRecord>,
}

impl FuzzCorpusReport {
    /// Whether no current, changed, unsupported, or unpruned stale state remains.
    pub fn is_clean(&self) -> bool {
        self.current_failures == 0
            && self.unresolved == 0
            && self.new == 0
            && self.changed == 0
            && self.unsupported == 0
            && self.resolved == self.pruned
    }
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> String {
    format!("{operation} {}: {error}", path.display())
}

fn require_exact_directory(directory: &Path) -> Result<(), String> {
    let metadata =
        fs::symlink_metadata(directory).map_err(|error| io_error("inspect", directory, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "corpus path must be an exact directory: {}",
            directory.display()
        ));
    }
    Ok(())
}

/// Reads one artifact without ever allocating beyond the artifact decoder cap.
pub fn read_failure_artifact(path: &Path) -> Result<FuzzFailureArtifact, String> {
    let file = File::open(path).map_err(|error| io_error("open", path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error("inspect", path, error))?
        .len();
    if length > MAX_FUZZ_ARTIFACT_FILE_BYTES as u64 {
        return Err(format!(
            "artifact {} exceeds {} byte cap",
            path.display(),
            MAX_FUZZ_ARTIFACT_FILE_BYTES
        ));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(MAX_FUZZ_ARTIFACT_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error("read", path, error))?;
    if bytes.len() > MAX_FUZZ_ARTIFACT_FILE_BYTES {
        return Err(format!(
            "artifact {} exceeds {} byte cap",
            path.display(),
            MAX_FUZZ_ARTIFACT_FILE_BYTES
        ));
    }
    FuzzFailureArtifact::from_bytes(&bytes).map_err(|error| error.to_string())
}

/// Writes an artifact through a same-directory temporary file and atomic rename.
/// Returns `false` when the exact artifact was already present.
pub fn write_failure_artifact_atomic(
    directory: &Path,
    artifact: &FuzzFailureArtifact,
) -> Result<bool, String> {
    artifact.validate().map_err(|error| error.to_string())?;
    fs::create_dir_all(directory).map_err(|error| io_error("create", directory, error))?;
    require_exact_directory(directory)?;
    let final_path = directory.join(format!("failure-{:016x}.rgfz", artifact.identity));
    match fs::symlink_metadata(&final_path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(format!(
                    "artifact destination is not a regular file: {}",
                    final_path.display()
                ));
            }
            let existing = read_failure_artifact(&final_path)?;
            if existing == *artifact {
                return Ok(false);
            }
            return Err(format!(
                "artifact identity collision at {}",
                final_path.display()
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect", &final_path, error)),
    }

    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp_path = directory.join(format!(
        ".failure-{:016x}-{}-{sequence}.tmp",
        artifact.identity,
        std::process::id()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)
            .map_err(|error| io_error("create", &temp_path, error))?;
        let bytes = artifact.to_bytes().map_err(|error| error.to_string())?;
        file.write_all(&bytes)
            .map_err(|error| io_error("write", &temp_path, error))?;
        file.sync_all()
            .map_err(|error| io_error("sync", &temp_path, error))?;
        fs::rename(&temp_path, &final_path)
            .map_err(|error| io_error("rename", &final_path, error))?;
        Ok(true)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn inventory(directory: &Path) -> Result<Vec<(PathBuf, FuzzFailureArtifact)>, String> {
    if !directory.exists() {
        return Ok(vec![]);
    }
    require_exact_directory(directory)?;
    let mut paths = fs::read_dir(directory)
        .map_err(|error| io_error("list", directory, error))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| io_error("list", directory, error))?;
    paths.sort();
    let mut entries = Vec::new();
    let mut identities = BTreeSet::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("rgfz") {
            continue;
        }
        let metadata =
            fs::symlink_metadata(&path).map_err(|error| io_error("inspect", &path, error))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!(
                "corpus artifact must be a direct regular file: {}",
                path.display()
            ));
        }
        let artifact = read_failure_artifact(&path)?;
        if !identities.insert(artifact.identity) {
            return Err(format!(
                "duplicate artifact identity {:016x} in {}",
                artifact.identity,
                directory.display()
            ));
        }
        entries.push((path, artifact));
    }
    Ok(entries)
}

type FailureKey = (u64, u64, FuzzPath);

fn failure_key(artifact: &FuzzFailureArtifact) -> FailureKey {
    (artifact.seed, artifact.case_index, artifact.actual_path)
}

/// Inventories existing identities and reconciles them with the fixed
/// regression cases under an explicit filesystem mutation policy.
pub fn reconcile_regression_corpus(
    directory: &Path,
    mode: FuzzCorpusMode,
) -> Result<FuzzCorpusReport, String> {
    let existing = inventory(directory)?;
    let mut report = FuzzCorpusReport {
        regressions: regression_cases().len(),
        inventoried: existing.len(),
        ..FuzzCorpusReport::default()
    };
    let mut by_key = BTreeMap::new();
    for (position, (_, artifact)) in existing.iter().enumerate() {
        if by_key.insert(failure_key(artifact), position).is_some() {
            return Err("corpus contains multiple artifacts for one failure key".into());
        }
    }

    let mut matched = BTreeSet::new();
    for (index, case) in regression_cases().iter().enumerate() {
        for comparison in run_case(0xfeed, index as u64, case, false)? {
            match comparison {
                FuzzComparison::Match { .. } => {}
                FuzzComparison::Unsupported { path, reason } => {
                    return Err(format!(
                        "regression interpreter coverage failure on {path:?}: {reason}"
                    ));
                }
                FuzzComparison::Failure(failure) => {
                    report.current_failures += 1;
                    report.unresolved += 1;
                    let key = failure_key(&failure);
                    match by_key.get(&key).copied() {
                        Some(position) => {
                            matched.insert(position);
                            if existing[position].1.identity == failure.identity {
                                report.reproduced += 1;
                                report.records.push(FuzzCorpusRecord {
                                    identity: failure.identity,
                                    previous_identity: None,
                                    state: FuzzCorpusState::Reproduced,
                                });
                            } else {
                                report.changed += 1;
                                report.records.push(FuzzCorpusRecord {
                                    identity: failure.identity,
                                    previous_identity: Some(existing[position].1.identity),
                                    state: FuzzCorpusState::Changed,
                                });
                            }
                        }
                        None => {
                            report.new += 1;
                            report.records.push(FuzzCorpusRecord {
                                identity: failure.identity,
                                previous_identity: None,
                                state: FuzzCorpusState::New,
                            });
                        }
                    }
                    if mode != FuzzCorpusMode::Check
                        && write_failure_artifact_atomic(directory, &failure)?
                    {
                        report.written += 1;
                    }
                }
            }
        }
    }

    for (position, (path, artifact)) in existing.iter().enumerate() {
        if matched.contains(&position) {
            continue;
        }
        match replay_failure(artifact)? {
            FuzzReplayStatus::Reproduced => {
                report.reproduced += 1;
                report.unresolved += 1;
                report.records.push(FuzzCorpusRecord {
                    identity: artifact.identity,
                    previous_identity: None,
                    state: FuzzCorpusState::Reproduced,
                });
            }
            FuzzReplayStatus::Resolved => {
                report.resolved += 1;
                report.records.push(FuzzCorpusRecord {
                    identity: artifact.identity,
                    previous_identity: None,
                    state: FuzzCorpusState::Resolved,
                });
                if mode == FuzzCorpusMode::WriteAndPruneResolved {
                    fs::remove_file(path).map_err(|error| io_error("prune", path, error))?;
                    report.pruned += 1;
                }
            }
            FuzzReplayStatus::Changed => {
                report.changed += 1;
                report.records.push(FuzzCorpusRecord {
                    identity: artifact.identity,
                    previous_identity: None,
                    state: FuzzCorpusState::Changed,
                });
            }
            FuzzReplayStatus::Unsupported { .. } => {
                report.unsupported += 1;
                report.records.push(FuzzCorpusRecord {
                    identity: artifact.identity,
                    previous_identity: None,
                    state: FuzzCorpusState::Unsupported,
                });
            }
        }
    }
    Ok(report)
}
