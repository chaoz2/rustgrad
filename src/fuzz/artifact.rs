use super::{FuzzCase, FuzzTensor};
use crate::TensorData;
use serde::{Deserialize, Serialize};
use std::fmt;

const MAGIC: &[u8; 4] = b"RGFZ";
const VERSION: u16 = 1;
const MAX_ARTIFACT_BYTES: usize = 1 << 20;
const MAX_TEXT_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzPath {
    CpuOracle,
    CapturedInterpreter,
    NativeScalar,
    NativeVector,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzComparisonPolicy {
    ExactBytes,
    FloatTolerance {
        absolute_bits: u64,
        relative_bits: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum FuzzOutcome {
    Value { tensor: FuzzTensor },
    Error { class: String, detail: String },
}

impl FuzzOutcome {
    pub(super) fn value(tensor: &TensorData) -> Self {
        Self::Value {
            tensor: FuzzTensor::from_tensor(tensor),
        }
    }
    pub(super) fn validate(&self) -> Result<(), FuzzArtifactError> {
        match self {
            Self::Value { tensor } => tensor.validate().map_err(FuzzArtifactError::Invalid),
            Self::Error { class, detail }
                if class.is_empty() || class.len() > 128 || detail.len() > MAX_TEXT_BYTES =>
            {
                Err(FuzzArtifactError::Invalid(
                    "invalid error outcome text".into(),
                ))
            }
            Self::Error { .. } => Ok(()),
        }
    }
}

/// Versioned, deterministic persisted semantic mismatch.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FuzzFailureArtifact {
    pub version: u16,
    pub seed: u64,
    pub case_index: u64,
    pub case: FuzzCase,
    pub expected_path: FuzzPath,
    pub actual_path: FuzzPath,
    pub policy: FuzzComparisonPolicy,
    pub expected: FuzzOutcome,
    pub actual: FuzzOutcome,
    pub identity: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FuzzArtifactError {
    TooLarge,
    Magic,
    Version(u16),
    Truncated,
    Trailing,
    Checksum,
    Json(String),
    Invalid(String),
    Identity,
}

impl fmt::Display for FuzzArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fuzz artifact error: {self:?}")
    }
}
impl std::error::Error for FuzzArtifactError {}

fn identity(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn checksum(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

impl FuzzFailureArtifact {
    pub fn new(
        seed: u64,
        case_index: u64,
        case: FuzzCase,
        actual_path: FuzzPath,
        policy: FuzzComparisonPolicy,
        expected: FuzzOutcome,
        actual: FuzzOutcome,
    ) -> Result<Self, FuzzArtifactError> {
        let mut artifact = Self {
            version: VERSION,
            seed,
            case_index,
            case,
            expected_path: FuzzPath::CpuOracle,
            actual_path,
            policy,
            expected,
            actual,
            identity: 0,
        };
        artifact.validate_without_identity()?;
        artifact.identity = artifact.expected_identity()?;
        Ok(artifact)
    }

    fn canonical_payload(&self) -> Result<Vec<u8>, FuzzArtifactError> {
        serde_json::to_vec(self).map_err(|error| FuzzArtifactError::Json(error.to_string()))
    }

    fn expected_identity(&self) -> Result<u64, FuzzArtifactError> {
        let mut canonical = self.clone();
        canonical.identity = 0;
        Ok(identity(&canonical.canonical_payload()?))
    }

    fn validate_without_identity(&self) -> Result<(), FuzzArtifactError> {
        if self.version != VERSION {
            return Err(FuzzArtifactError::Version(self.version));
        }
        self.case.validate().map_err(FuzzArtifactError::Invalid)?;
        self.expected.validate()?;
        self.actual.validate()?;
        if self.expected_path != FuzzPath::CpuOracle || self.actual_path == FuzzPath::CpuOracle {
            return Err(FuzzArtifactError::Invalid(
                "invalid differential path pair".into(),
            ));
        }
        if let FuzzComparisonPolicy::FloatTolerance {
            absolute_bits,
            relative_bits,
        } = self.policy
        {
            let absolute = f64::from_bits(absolute_bits);
            let relative = f64::from_bits(relative_bits);
            if self.actual_path == FuzzPath::CapturedInterpreter
                || !absolute.is_finite()
                || absolute < 0.0
                || !relative.is_finite()
                || relative < 0.0
            {
                return Err(FuzzArtifactError::Invalid(
                    "invalid floating comparison policy".into(),
                ));
            }
            if !matches!(&self.expected, FuzzOutcome::Value { tensor } if matches!(tensor.dtype, crate::DType::F32 | crate::DType::F64))
            {
                return Err(FuzzArtifactError::Invalid(
                    "floating policy requires floating expected value".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), FuzzArtifactError> {
        self.validate_without_identity()?;
        if self.identity == 0 || self.identity != self.expected_identity()? {
            return Err(FuzzArtifactError::Identity);
        }
        Ok(())
    }

    /// Encodes a bounded checksummed JSON payload in the `RGFZ` v1 envelope.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FuzzArtifactError> {
        self.validate()?;
        let payload = self.canonical_payload()?;
        if payload.len() > MAX_ARTIFACT_BYTES {
            return Err(FuzzArtifactError::TooLarge);
        }
        let mut out = Vec::with_capacity(payload.len() + 14);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&checksum(&payload).to_le_bytes());
        Ok(out)
    }

    /// Decodes and fully validates an `RGFZ` artifact before exposing its case.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FuzzArtifactError> {
        if bytes.len() > MAX_ARTIFACT_BYTES + 14 {
            return Err(FuzzArtifactError::TooLarge);
        }
        if bytes.len() < 14 {
            return Err(FuzzArtifactError::Truncated);
        }
        if &bytes[..4] != MAGIC {
            return Err(FuzzArtifactError::Magic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().expect("fixed width"));
        if version != VERSION {
            return Err(FuzzArtifactError::Version(version));
        }
        let length = u32::from_le_bytes(bytes[6..10].try_into().expect("fixed width")) as usize;
        let end = 10usize
            .checked_add(length)
            .ok_or(FuzzArtifactError::TooLarge)?;
        if end.checked_add(4).ok_or(FuzzArtifactError::TooLarge)? > bytes.len() {
            return Err(FuzzArtifactError::Truncated);
        }
        if end + 4 != bytes.len() {
            return Err(FuzzArtifactError::Trailing);
        }
        let payload = &bytes[10..end];
        let stored_checksum =
            u32::from_le_bytes(bytes[end..end + 4].try_into().expect("fixed width"));
        if checksum(payload) != stored_checksum {
            return Err(FuzzArtifactError::Checksum);
        }
        let artifact: Self = serde_json::from_slice(payload)
            .map_err(|error| FuzzArtifactError::Json(error.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }
}
