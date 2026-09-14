use super::{CompiledAdamWProgramArtifact, CompiledModuleAdamWCheckpoint, training};
use crate::file_io::{ExactFileError, read_file_bytes_bounded, replace_file_bytes_atomically};
use crate::{Error, Result};
use std::{fmt, io, path::Path};

const MAGIC: &[u8; 4] = b"RGAB";
const FORMAT_VERSION: u8 = 1;
const HEADER_BYTES: usize = 21;
const CHECKSUM_BYTES: usize = 8;
pub(super) const MAX_BUNDLE_BYTES: usize = (1 << 30) + (256 << 20) + HEADER_BYTES + CHECKSUM_BYTES;

/// One deterministic file payload containing an exact compiled AdamW program
/// artifact and its exact complete-module checkpoint.
///
/// The outer envelope does not reinterpret either inner format. It authenticates
/// that they form one compatible restore pair and replaces a local destination
/// atomically, so a saved resume point cannot expose artifacts from two writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAdamWResumeBundle {
    bytes: Vec<u8>,
    program_artifact: CompiledAdamWProgramArtifact,
    checkpoint: CompiledModuleAdamWCheckpoint,
}

/// A local compiled AdamW resume-bundle file failure.
#[derive(Debug)]
pub enum CompiledAdamWResumeBundleFileError {
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

impl fmt::Display for CompiledAdamWResumeBundleFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => write!(
                formatter,
                "compiled AdamW resume-bundle file {operation} failed: {kind:?}"
            ),
            Self::Limit { actual, maximum } => write!(
                formatter,
                "compiled AdamW resume-bundle file has {actual} bytes, exceeding byte limit {maximum}"
            ),
            Self::Format(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompiledAdamWResumeBundleFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Io { .. } | Self::Limit { .. } => None,
        }
    }
}

fn file_error(error: ExactFileError) -> CompiledAdamWResumeBundleFileError {
    match error {
        ExactFileError::Io { operation, source } => CompiledAdamWResumeBundleFileError::Io {
            operation,
            kind: source.kind(),
        },
        ExactFileError::Limit { actual, maximum } => {
            CompiledAdamWResumeBundleFileError::Limit { actual, maximum }
        }
        ExactFileError::Allocation => CompiledAdamWResumeBundleFileError::Io {
            operation: "allocate read buffer",
            kind: io::ErrorKind::OutOfMemory,
        },
        ExactFileError::InvalidFileName => CompiledAdamWResumeBundleFileError::Io {
            operation: "validate path",
            kind: io::ErrorKind::InvalidInput,
        },
        ExactFileError::StagingExhausted => CompiledAdamWResumeBundleFileError::Io {
            operation: "create unique staging file",
            kind: io::ErrorKind::AlreadyExists,
        },
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn encode(
    program_artifact: &CompiledAdamWProgramArtifact,
    checkpoint: &CompiledModuleAdamWCheckpoint,
) -> Result<Vec<u8>> {
    super::program_artifact::admit_artifact_checkpoint_pair(program_artifact, checkpoint)?;
    let program_len = u64::try_from(program_artifact.as_bytes().len())
        .map_err(|_| training("compiled AdamW resume-bundle program length overflows"))?;
    let checkpoint_len = u64::try_from(checkpoint.as_bytes().len())
        .map_err(|_| training("compiled AdamW resume-bundle checkpoint length overflows"))?;
    let total = program_artifact
        .as_bytes()
        .len()
        .checked_add(checkpoint.as_bytes().len())
        .and_then(|total| total.checked_add(HEADER_BYTES + CHECKSUM_BYTES))
        .ok_or_else(|| training("compiled AdamW resume-bundle length overflows"))?;
    if total > MAX_BUNDLE_BYTES {
        return Err(training("compiled AdamW resume-bundle exceeds byte limit"));
    }
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(MAGIC);
    bytes.push(FORMAT_VERSION);
    bytes.extend_from_slice(&program_len.to_le_bytes());
    bytes.extend_from_slice(&checkpoint_len.to_le_bytes());
    bytes.extend_from_slice(program_artifact.as_bytes());
    bytes.extend_from_slice(checkpoint.as_bytes());
    bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<(CompiledAdamWProgramArtifact, CompiledModuleAdamWCheckpoint)> {
    if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES
        || bytes.len() > MAX_BUNDLE_BYTES
        || &bytes[..4] != MAGIC
    {
        return Err(training("compiled AdamW resume-bundle header is invalid"));
    }
    if bytes[4] != FORMAT_VERSION {
        return Err(training(
            "compiled AdamW resume-bundle version is unsupported",
        ));
    }
    let program_len = u64::from_le_bytes(
        bytes[5..13]
            .try_into()
            .map_err(|_| training("compiled AdamW resume-bundle program length is invalid"))?,
    );
    let checkpoint_len = u64::from_le_bytes(
        bytes[13..21]
            .try_into()
            .map_err(|_| training("compiled AdamW resume-bundle checkpoint length is invalid"))?,
    );
    let program_len = usize::try_from(program_len)
        .map_err(|_| training("compiled AdamW resume-bundle program length overflows"))?;
    let checkpoint_len = usize::try_from(checkpoint_len)
        .map_err(|_| training("compiled AdamW resume-bundle checkpoint length overflows"))?;
    let program_end = HEADER_BYTES
        .checked_add(program_len)
        .ok_or_else(|| training("compiled AdamW resume-bundle length overflows"))?;
    let checkpoint_end = program_end
        .checked_add(checkpoint_len)
        .ok_or_else(|| training("compiled AdamW resume-bundle length overflows"))?;
    if checkpoint_end.checked_add(CHECKSUM_BYTES) != Some(bytes.len()) {
        return Err(training("compiled AdamW resume-bundle length is invalid"));
    }
    let expected = u64::from_le_bytes(
        bytes[checkpoint_end..]
            .try_into()
            .map_err(|_| training("compiled AdamW resume-bundle checksum is invalid"))?,
    );
    if checksum(&bytes[..checkpoint_end]) != expected {
        return Err(training("compiled AdamW resume-bundle checksum mismatch"));
    }
    let program_artifact =
        CompiledAdamWProgramArtifact::from_bytes(bytes[HEADER_BYTES..program_end].to_vec())?;
    let checkpoint =
        CompiledModuleAdamWCheckpoint::from_bytes(bytes[program_end..checkpoint_end].to_vec())?;
    super::program_artifact::admit_artifact_checkpoint_pair(&program_artifact, &checkpoint)?;
    Ok((program_artifact, checkpoint))
}

impl CompiledAdamWResumeBundle {
    /// Creates one validated deterministic envelope without changing either
    /// inner artifact's bytes.
    pub fn new(
        program_artifact: CompiledAdamWProgramArtifact,
        checkpoint: CompiledModuleAdamWCheckpoint,
    ) -> Result<Self> {
        let bytes = encode(&program_artifact, &checkpoint)?;
        Ok(Self {
            bytes,
            program_artifact,
            checkpoint,
        })
    }

    /// Validates and owns deterministic resume-bundle bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        let (program_artifact, checkpoint) = decode(&bytes)?;
        Ok(Self {
            bytes,
            program_artifact,
            checkpoint,
        })
    }

    /// Loads and validates one bundle under its intrinsic byte bound.
    pub fn load_file(
        path: impl AsRef<Path>,
    ) -> std::result::Result<Self, CompiledAdamWResumeBundleFileError> {
        Self::load_file_with_byte_limit(path, MAX_BUNDLE_BYTES)
    }

    /// Loads and validates one bundle under the smaller of a caller-supplied
    /// byte bound and the format's intrinsic bound.
    pub fn load_file_with_byte_limit(
        path: impl AsRef<Path>,
        maximum: usize,
    ) -> std::result::Result<Self, CompiledAdamWResumeBundleFileError> {
        let bytes =
            read_file_bytes_bounded(path, maximum.min(MAX_BUNDLE_BYTES)).map_err(file_error)?;
        Self::from_bytes(bytes).map_err(CompiledAdamWResumeBundleFileError::Format)
    }

    /// Atomically replaces `path` with this exact validated bundle after
    /// syncing a uniquely created same-directory staging file.
    pub fn save_file(
        &self,
        path: impl AsRef<Path>,
    ) -> std::result::Result<(), CompiledAdamWResumeBundleFileError> {
        replace_file_bytes_atomically(path, self.as_bytes()).map_err(file_error)
    }

    pub fn program_artifact(&self) -> &CompiledAdamWProgramArtifact {
        &self.program_artifact
    }

    pub fn format_version(&self) -> u8 {
        FORMAT_VERSION
    }

    pub fn checkpoint(&self) -> &CompiledModuleAdamWCheckpoint {
        &self.checkpoint
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
