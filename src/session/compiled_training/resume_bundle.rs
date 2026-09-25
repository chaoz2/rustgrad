use super::{
    CompiledAdamWCheckpoint, CompiledModuleCheckpoint, CompiledModuleCheckpointPayload,
    CompiledMomentumSgdCheckpoint, CompiledTrainingProgramArtifact, training,
};
use crate::file_io::{ExactFileError, read_file_bytes_bounded, replace_file_bytes_atomically};
use crate::{Error, Result};
use std::sync::Arc;
use std::{fmt, io, path::Path};

const MAGIC: &[u8; 4] = b"RGAB";
const FORMAT_VERSION: u8 = 1;
const HEADER_BYTES: usize = 21;
const CHECKSUM_BYTES: usize = 8;
pub(super) const MAX_BUNDLE_BYTES: usize = (1 << 30) + (256 << 20) + HEADER_BYTES + CHECKSUM_BYTES;

/// One deterministic file payload containing an exact compiled training
/// program artifact and its optimizer-typed complete-module checkpoint.
///
/// The outer envelope does not reinterpret either inner format. It authenticates
/// that they form one compatible restore pair and replaces a local destination
/// atomically, so a saved resume point cannot expose artifacts from two writes.
#[derive(Clone)]
pub struct CompiledTrainingResumeBundle<C> {
    bytes: Vec<u8>,
    program_artifact: CompiledTrainingProgramArtifact,
    checkpoint: CompiledModuleCheckpoint<C>,
    admitted: Arc<super::program_artifact::AdmittedArtifactCheckpointPair<C>>,
}

/// Source-compatible AdamW name for the optimizer-typed bundle.
pub type CompiledAdamWResumeBundle = CompiledTrainingResumeBundle<CompiledAdamWCheckpoint>;

/// Momentum-SGD bundle containing one RGAP v3 program and its complete module
/// checkpoint.
pub type CompiledMomentumSgdResumeBundle =
    CompiledTrainingResumeBundle<CompiledMomentumSgdCheckpoint>;

type PairAdmission<C> =
    fn(
        &CompiledTrainingProgramArtifact,
        &CompiledModuleCheckpoint<C>,
    ) -> Result<Arc<super::program_artifact::AdmittedArtifactCheckpointPair<C>>>;

impl<C: fmt::Debug> fmt::Debug for CompiledTrainingResumeBundle<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledTrainingResumeBundle")
            .field("bytes", &self.bytes)
            .field("program_artifact", &self.program_artifact)
            .field("checkpoint", &self.checkpoint)
            .finish()
    }
}

impl<C> PartialEq for CompiledTrainingResumeBundle<C> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
            && self.program_artifact == other.program_artifact
            && self.checkpoint == other.checkpoint
    }
}

impl<C> Eq for CompiledTrainingResumeBundle<C> {}

/// A local compiled-training resume-bundle file failure.
#[derive(Debug)]
pub enum CompiledTrainingResumeBundleFileError {
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

/// Source-compatible AdamW name for bundle file failures.
pub type CompiledAdamWResumeBundleFileError = CompiledTrainingResumeBundleFileError;

/// Momentum-SGD compatibility name for bundle file failures.
pub type CompiledMomentumSgdResumeBundleFileError = CompiledTrainingResumeBundleFileError;

impl fmt::Display for CompiledTrainingResumeBundleFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => write!(
                formatter,
                "compiled training resume-bundle file {operation} failed: {kind:?}"
            ),
            Self::Limit { actual, maximum } => write!(
                formatter,
                "compiled training resume-bundle file has {actual} bytes, exceeding byte limit {maximum}"
            ),
            Self::Format(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompiledTrainingResumeBundleFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Io { .. } | Self::Limit { .. } => None,
        }
    }
}

fn file_error(error: ExactFileError) -> CompiledTrainingResumeBundleFileError {
    match error {
        ExactFileError::Io { operation, source } => CompiledTrainingResumeBundleFileError::Io {
            operation,
            kind: source.kind(),
        },
        ExactFileError::Limit { actual, maximum } => {
            CompiledTrainingResumeBundleFileError::Limit { actual, maximum }
        }
        ExactFileError::Allocation => CompiledTrainingResumeBundleFileError::Io {
            operation: "allocate read buffer",
            kind: io::ErrorKind::OutOfMemory,
        },
        ExactFileError::InvalidFileName => CompiledTrainingResumeBundleFileError::Io {
            operation: "validate path",
            kind: io::ErrorKind::InvalidInput,
        },
        ExactFileError::StagingExhausted => CompiledTrainingResumeBundleFileError::Io {
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

fn encode<C: CompiledModuleCheckpointPayload>(
    program_artifact: &CompiledTrainingProgramArtifact,
    checkpoint: &CompiledModuleCheckpoint<C>,
) -> Result<Vec<u8>> {
    let program_len = u64::try_from(program_artifact.as_bytes().len())
        .map_err(|_| training("compiled training resume-bundle program length overflows"))?;
    let checkpoint_len = u64::try_from(checkpoint.as_bytes().len())
        .map_err(|_| training("compiled training resume-bundle checkpoint length overflows"))?;
    let total = program_artifact
        .as_bytes()
        .len()
        .checked_add(checkpoint.as_bytes().len())
        .and_then(|total| total.checked_add(HEADER_BYTES + CHECKSUM_BYTES))
        .ok_or_else(|| training("compiled training resume-bundle length overflows"))?;
    if total > MAX_BUNDLE_BYTES {
        return Err(training(
            "compiled training resume-bundle exceeds byte limit",
        ));
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

fn decode<C: CompiledModuleCheckpointPayload>(
    bytes: &[u8],
    admit: PairAdmission<C>,
) -> Result<(
    CompiledTrainingProgramArtifact,
    CompiledModuleCheckpoint<C>,
    Arc<super::program_artifact::AdmittedArtifactCheckpointPair<C>>,
)> {
    if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES
        || bytes.len() > MAX_BUNDLE_BYTES
        || &bytes[..4] != MAGIC
    {
        return Err(training(
            "compiled training resume-bundle header is invalid",
        ));
    }
    if bytes[4] != FORMAT_VERSION {
        return Err(training(
            "compiled training resume-bundle version is unsupported",
        ));
    }
    let program_len = u64::from_le_bytes(
        bytes[5..13]
            .try_into()
            .map_err(|_| training("compiled training resume-bundle program length is invalid"))?,
    );
    let checkpoint_len =
        u64::from_le_bytes(bytes[13..21].try_into().map_err(|_| {
            training("compiled training resume-bundle checkpoint length is invalid")
        })?);
    let program_len = usize::try_from(program_len)
        .map_err(|_| training("compiled training resume-bundle program length overflows"))?;
    let checkpoint_len = usize::try_from(checkpoint_len)
        .map_err(|_| training("compiled training resume-bundle checkpoint length overflows"))?;
    let program_end = HEADER_BYTES
        .checked_add(program_len)
        .ok_or_else(|| training("compiled training resume-bundle length overflows"))?;
    let checkpoint_end = program_end
        .checked_add(checkpoint_len)
        .ok_or_else(|| training("compiled training resume-bundle length overflows"))?;
    if checkpoint_end.checked_add(CHECKSUM_BYTES) != Some(bytes.len()) {
        return Err(training(
            "compiled training resume-bundle length is invalid",
        ));
    }
    let expected = u64::from_le_bytes(
        bytes[checkpoint_end..]
            .try_into()
            .map_err(|_| training("compiled training resume-bundle checksum is invalid"))?,
    );
    if checksum(&bytes[..checkpoint_end]) != expected {
        return Err(training(
            "compiled training resume-bundle checksum mismatch",
        ));
    }
    let program_artifact =
        CompiledTrainingProgramArtifact::from_bytes(bytes[HEADER_BYTES..program_end].to_vec())?;
    let checkpoint =
        CompiledModuleCheckpoint::<C>::from_bytes(bytes[program_end..checkpoint_end].to_vec())?;
    let admitted = admit(&program_artifact, &checkpoint)?;
    Ok((program_artifact, checkpoint, admitted))
}

macro_rules! impl_resume_bundle {
    ($payload:ty, $admit:path) => {
        impl CompiledTrainingResumeBundle<$payload> {
            /// Creates one validated deterministic envelope without changing
            /// either inner artifact's bytes.
            pub fn new(
                program_artifact: CompiledTrainingProgramArtifact,
                checkpoint: CompiledModuleCheckpoint<$payload>,
            ) -> Result<Self> {
                let admitted = $admit(&program_artifact, &checkpoint)?;
                let bytes = encode(&program_artifact, &checkpoint)?;
                Ok(Self {
                    bytes,
                    program_artifact,
                    checkpoint,
                    admitted,
                })
            }

            /// Validates and owns deterministic resume-bundle bytes.
            pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
                let bytes = bytes.into();
                let (program_artifact, checkpoint, admitted) = decode(&bytes, $admit)?;
                Ok(Self {
                    bytes,
                    program_artifact,
                    checkpoint,
                    admitted,
                })
            }

            /// Loads and validates one bundle under its intrinsic byte bound.
            pub fn load_file(
                path: impl AsRef<Path>,
            ) -> std::result::Result<Self, CompiledTrainingResumeBundleFileError> {
                Self::load_file_with_byte_limit(path, MAX_BUNDLE_BYTES)
            }

            /// Loads and validates one bundle under the smaller of a
            /// caller-supplied byte bound and the format's intrinsic bound.
            pub fn load_file_with_byte_limit(
                path: impl AsRef<Path>,
                maximum: usize,
            ) -> std::result::Result<Self, CompiledTrainingResumeBundleFileError> {
                let bytes = read_file_bytes_bounded(path, maximum.min(MAX_BUNDLE_BYTES))
                    .map_err(file_error)?;
                Self::from_bytes(bytes).map_err(CompiledTrainingResumeBundleFileError::Format)
            }
        }
    };
}

impl_resume_bundle!(
    CompiledAdamWCheckpoint,
    super::program_artifact::admit_artifact_checkpoint_pair
);
impl_resume_bundle!(
    CompiledMomentumSgdCheckpoint,
    super::program_artifact::admit_momentum_artifact_checkpoint_pair
);

impl<C> CompiledTrainingResumeBundle<C> {
    /// Atomically replaces `path` with this exact validated bundle after
    /// syncing a uniquely created same-directory staging file.
    pub fn save_file(
        &self,
        path: impl AsRef<Path>,
    ) -> std::result::Result<(), CompiledTrainingResumeBundleFileError> {
        replace_file_bytes_atomically(path, self.as_bytes()).map_err(file_error)
    }

    pub fn program_artifact(&self) -> &CompiledTrainingProgramArtifact {
        &self.program_artifact
    }

    pub fn format_version(&self) -> u8 {
        FORMAT_VERSION
    }

    pub fn checkpoint(&self) -> &CompiledModuleCheckpoint<C> {
        &self.checkpoint
    }

    pub(super) fn admitted(
        &self,
    ) -> &Arc<super::program_artifact::AdmittedArtifactCheckpointPair<C>> {
        &self.admitted
    }

    #[cfg(test)]
    pub(super) fn shares_admission_with(&self, other: &Self) -> bool
    where
        C: CompiledModuleCheckpointPayload,
    {
        Arc::ptr_eq(&self.admitted, &other.admitted)
            && self
                .program_artifact
                .shares_admission_with(&other.program_artifact)
            && Arc::ptr_eq(
                self.checkpoint.decoded_arc(),
                other.checkpoint.decoded_arc(),
            )
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
