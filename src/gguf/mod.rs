//! Bounded, in-memory GGUF container parsing.
//!
//! The reader preserves typed metadata and validated tensor byte ranges. Dense
//! scalar GGML layouts can be materialized into [`TensorData`]. The audited
//! Q4_0, Q8_0, Q4_K, and Q6_K layouts additionally support checked F32
//! dequantization; other quantized layouts remain opaque.

use crate::{DType, TensorData};
use std::{collections::BTreeMap, error, fmt};

mod metadata;
mod quantization;
mod reader;
mod tensor;

pub use metadata::{
    GgufMetadata, GgufMetadataAccessError, GgufMetadataExpectation, GgufMetadataType,
    GgufMetadataValue,
};
pub use tensor::{GgmlLayout, GgmlType, GgufTensor};

/// GGUF container versions evidenced by the checked-in tinygrad reader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GgufVersion {
    V2,
    V3,
}

impl GgufVersion {
    pub const fn raw(self) -> u32 {
        match self {
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }
}

/// Resource bounds applied before allocating from untrusted GGUF counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufLimits {
    pub max_metadata_entries: usize,
    pub max_tensors: usize,
    pub max_string_bytes: usize,
    pub max_array_elements: usize,
    pub max_metadata_values: usize,
    pub max_array_depth: usize,
    pub max_rank: usize,
}

impl Default for GgufLimits {
    fn default() -> Self {
        Self {
            max_metadata_entries: 131_072,
            max_tensors: 1_000_000,
            max_string_bytes: 64 * 1024 * 1024,
            max_array_elements: 16 * 1024 * 1024,
            max_metadata_values: 32 * 1024 * 1024,
            max_array_depth: 8,
            max_rank: 4,
        }
    }
}

/// Structured reason why a GGUF container was rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufErrorKind {
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u32),
    LimitExceeded {
        field: &'static str,
        value: u64,
        limit: usize,
    },
    InvalidUtf8 {
        field: &'static str,
    },
    InvalidMetadataKey(String),
    EmptyName {
        field: &'static str,
    },
    UnknownMetadataType(u32),
    InvalidBoolean(u8),
    ArrayNestingLimit,
    DuplicateMetadata(String),
    DuplicateTensor(String),
    InvalidAlignment(u64),
    InvalidAlignmentType(GgufMetadataType),
    InvalidPadding {
        section: &'static str,
    },
    InvalidRank {
        tensor: String,
        rank: u32,
    },
    InvalidDimension {
        tensor: String,
        axis: usize,
        value: u64,
    },
    ShapeOverflow {
        tensor: String,
    },
    UnknownTensorType(u32),
    BlockElementMismatch {
        tensor: String,
        elements: usize,
        block_elements: usize,
    },
    MisalignedTensorOffset {
        tensor: String,
        offset: u64,
        alignment: usize,
    },
    TensorRangeOutOfBounds {
        tensor: String,
    },
    OverlappingTensors {
        first: String,
        second: String,
    },
    TrailingData {
        bytes: usize,
    },
    TensorNotFound(String),
    QuantizedMaterialization {
        tensor: String,
        kind: GgmlType,
    },
}

/// A GGUF parse/materialization error with its byte position when known.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufError {
    kind: GgufErrorKind,
    offset: usize,
}

impl GgufError {
    pub fn kind(&self) -> &GgufErrorKind {
        &self.kind
    }

    pub const fn offset(&self) -> usize {
        self.offset
    }

    pub(super) const fn new(kind: GgufErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GGUF error at byte {}: {:?}", self.offset, self.kind)
    }
}

impl error::Error for GgufError {}

/// A validated GGUF view borrowing the original in-memory bytes.
pub struct GgufFile<'a> {
    bytes: &'a [u8],
    version: GgufVersion,
    alignment: usize,
    data_offset: usize,
    metadata: Vec<GgufMetadata>,
    tensors: Vec<GgufTensor>,
}

impl fmt::Debug for GgufFile<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("data_offset", &self.data_offset)
            .field("metadata", &self.metadata)
            .field("tensors", &self.tensors)
            .finish_non_exhaustive()
    }
}

impl<'a> GgufFile<'a> {
    pub const fn version(&self) -> GgufVersion {
        self.version
    }

    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    pub const fn data_offset(&self) -> usize {
        self.data_offset
    }

    pub fn metadata(&self) -> &[GgufMetadata] {
        &self.metadata
    }

    pub fn metadata_value(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata
            .iter()
            .find(|entry| entry.key() == key)
            .map(GgufMetadata::value)
    }

    /// Looks up a `STRING` metadata value without losing its wire type.
    pub fn metadata_string(&self, key: &str) -> Result<Option<&str>, GgufMetadataAccessError> {
        metadata::lookup_string(self.metadata_value(key), key)
    }

    /// Looks up a `BOOL` metadata value.
    pub fn metadata_bool(&self, key: &str) -> Result<Option<bool>, GgufMetadataAccessError> {
        metadata::lookup_bool(self.metadata_value(key), key)
    }

    /// Looks up any non-negative integer metadata scalar as `u64`.
    pub fn metadata_u64(&self, key: &str) -> Result<Option<u64>, GgufMetadataAccessError> {
        metadata::lookup_u64(self.metadata_value(key), key)
    }

    /// Looks up an `F32` or `F64` metadata scalar as `f64`.
    pub fn metadata_f64(&self, key: &str) -> Result<Option<f64>, GgufMetadataAccessError> {
        metadata::lookup_f64(self.metadata_value(key), key)
    }

    /// Looks up a homogeneous `ARRAY<STRING>` metadata value.
    pub fn metadata_strings(
        &self,
        key: &str,
    ) -> Result<Option<Vec<&str>>, GgufMetadataAccessError> {
        metadata::lookup_strings(self.metadata_value(key), key)
    }

    /// Looks up a homogeneous integer array and converts every item to `i64`.
    pub fn metadata_integers(
        &self,
        key: &str,
    ) -> Result<Option<Vec<i64>>, GgufMetadataAccessError> {
        metadata::lookup_integers(self.metadata_value(key), key)
    }

    pub fn tensors(&self) -> &[GgufTensor] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|tensor| tensor.name() == name)
    }

    /// Returns the exact validated bytes for either a dense or quantized tensor.
    pub fn tensor_bytes(&self, name: &str) -> Result<&'a [u8], GgufError> {
        let tensor = self.tensor(name).ok_or_else(|| {
            GgufError::new(
                GgufErrorKind::TensorNotFound(name.to_owned()),
                self.data_offset,
            )
        })?;
        Ok(&self.bytes[tensor.raw_range()])
    }

    /// Materializes a dense scalar GGML tensor without numeric conversion.
    /// Quantized layouts return a structured error and remain available through
    /// [`Self::tensor_bytes`].
    pub fn materialize_dense(&self, name: &str) -> Result<TensorData, GgufError> {
        let tensor = self.tensor(name).ok_or_else(|| {
            GgufError::new(
                GgufErrorKind::TensorNotFound(name.to_owned()),
                self.data_offset,
            )
        })?;
        tensor::materialize_dense(tensor, &self.bytes[tensor.raw_range()])
    }

    /// Materializes dense storage or an audited quantized layout as F32.
    /// Dense values use RustGrad's explicit numeric cast policy rather than a
    /// raw bitcast.
    pub fn materialize_f32(&self, name: &str) -> Result<TensorData, GgufError> {
        let tensor = self.tensor(name).ok_or_else(|| {
            GgufError::new(
                GgufErrorKind::TensorNotFound(name.to_owned()),
                self.data_offset,
            )
        })?;
        match tensor.layout() {
            GgmlLayout::Dense { .. } => Ok(self.materialize_dense(name)?.cast(DType::F32)),
            GgmlLayout::Quantized { .. } => {
                quantization::materialize_f32(tensor, &self.bytes[tensor.raw_range()])
            }
        }
    }

    /// Materializes every tensor into a deterministic name map of F32 values.
    ///
    /// Tensors are validated and converted in file order, so the first error
    /// is stable. The map is returned only after every tensor succeeds; an
    /// unsupported quantized layout cannot produce a partial state.
    pub fn materialize_state_f32(&self) -> Result<BTreeMap<String, TensorData>, GgufError> {
        let mut state = BTreeMap::new();
        for tensor in &self.tensors {
            let value = self.materialize_f32(tensor.name())?;
            state.insert(tensor.name().to_owned(), value);
        }
        Ok(state)
    }
}

/// Parses a GGUF container with conservative default resource limits.
pub fn read_gguf(bytes: &[u8]) -> Result<GgufFile<'_>, GgufError> {
    read_gguf_with_limits(bytes, GgufLimits::default())
}

/// Parses a GGUF container with caller-selected resource limits.
pub fn read_gguf_with_limits(bytes: &[u8], limits: GgufLimits) -> Result<GgufFile<'_>, GgufError> {
    reader::parse(bytes, limits)
}

#[cfg(test)]
mod quantization_tests;
#[cfg(test)]
mod tests;
