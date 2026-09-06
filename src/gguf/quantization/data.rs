//! Portable exact-byte ownership for audited GGML block-quantized tensors.

use super::blocks::{
    BlockDecodeError, decode_iq2_s_block, decode_iq3_s_block, decode_iq3_xxs_block,
    decode_iq4_xs_block, decode_mxfp4_block, decode_q1_0_block, decode_q4_0_block,
    decode_q4_1_block, decode_q4_k_block, decode_q5_0_block, decode_q5_1_block, decode_q5_k_block,
    decode_q6_k_block, decode_q8_0_block,
};
use crate::{GgmlLayout, GgmlType, Shape, TensorData};
use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::{Deref, Range},
    sync::Arc,
};

/// Descriptor for a read-only packed GGML buffer. It deliberately has no
/// `DType`: block quantization is a physical encoding, not a scalar dtype.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QuantizedBufferDesc {
    pub ggml_type: GgmlType,
    pub logical_shape: Shape,
    pub block_elements: usize,
    pub block_bytes: usize,
    pub bytes: usize,
    pub alignment: usize,
    pub identity: u64,
}

impl QuantizedBufferDesc {
    pub(crate) fn validate_metadata(&self) -> Result<(), QuantizedError> {
        if self.alignment != 1 || self.logical_shape.rank() != 2 || self.identity == 0 {
            return Err(QuantizedError::InvalidGeometry);
        }
        let GgmlLayout::Quantized {
            block_elements,
            block_bytes,
        } = self.ggml_type.layout()
        else {
            return Err(QuantizedError::UnsupportedType(self.ggml_type));
        };
        if !matches!(
            self.ggml_type,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Mxfp4
                | GgmlType::Q1_0
                | GgmlType::Iq4Xs
                | GgmlType::Iq3Xxs
                | GgmlType::Iq3S
                | GgmlType::Iq2S
                | GgmlType::Q8_0
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
        ) || self.block_elements != block_elements
            || self.block_bytes != block_bytes
        {
            return Err(QuantizedError::InvalidGeometry);
        }
        let rows = self.logical_shape.dims()[0];
        let columns = self.logical_shape.dims()[1];
        if !columns.is_multiple_of(block_elements)
            || rows
                .checked_mul(columns / block_elements)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                != Some(self.bytes)
        {
            return Err(QuantizedError::InvalidGeometry);
        }
        Ok(())
    }
}

/// One immutable byte owner plus the exact range occupied by a packed tensor.
#[derive(Clone)]
struct SharedByteRange {
    owner: Arc<Vec<u8>>,
    range: Range<usize>,
}

impl SharedByteRange {
    fn from_vec(bytes: Vec<u8>) -> Self {
        let range = 0..bytes.len();
        Self {
            owner: Arc::new(bytes),
            range,
        }
    }

    fn new(owner: Arc<Vec<u8>>, range: Range<usize>) -> Result<Self, QuantizedError> {
        owner
            .get(range.clone())
            .ok_or(QuantizedError::InvalidGeometry)?;
        Ok(Self { owner, range })
    }

    fn as_slice(&self) -> &[u8] {
        &self.owner[self.range.clone()]
    }

    #[cfg(test)]
    fn shares_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }

    #[cfg(test)]
    fn owner_count(&self) -> usize {
        Arc::strong_count(&self.owner)
    }
}

impl Deref for SharedByteRange {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl fmt::Debug for SharedByteRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl PartialEq for SharedByteRange {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for SharedByteRange {}

impl PartialOrd for SharedByteRange {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SharedByteRange {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Hash for SharedByteRange {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

/// Immutable exact GGML bytes plus their validated portable descriptor.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QuantizedTensorData {
    desc: QuantizedBufferDesc,
    bytes: SharedByteRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizedError {
    UnsupportedType(GgmlType),
    InvalidRank,
    InvalidGeometry,
    InvalidAlignment,
    Length { expected: usize, actual: usize },
    Overflow,
    NonFinite,
}

impl fmt::Display for QuantizedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "quantized tensor error: {self:?}")
    }
}

impl std::error::Error for QuantizedError {}

impl QuantizedTensorData {
    /// Validates an owned packed tensor with portable byte alignment one.
    pub fn new(
        ggml_type: GgmlType,
        logical_shape: Shape,
        bytes: Vec<u8>,
    ) -> Result<Self, QuantizedError> {
        Self::from_aligned_bytes(ggml_type, logical_shape, bytes, 1, 0)
    }

    /// Validates source alignment before retaining portable byte-aligned
    /// storage. Source offsets never participate in artifact identity.
    pub fn from_aligned_bytes(
        ggml_type: GgmlType,
        logical_shape: Shape,
        bytes: Vec<u8>,
        alignment: usize,
        source_offset: usize,
    ) -> Result<Self, QuantizedError> {
        Self::from_byte_range(
            ggml_type,
            logical_shape,
            SharedByteRange::from_vec(bytes),
            alignment,
            source_offset,
        )
    }

    pub(crate) fn from_shared_aligned_bytes(
        ggml_type: GgmlType,
        logical_shape: Shape,
        owner: Arc<Vec<u8>>,
        range: Range<usize>,
        alignment: usize,
    ) -> Result<Self, QuantizedError> {
        let source_offset = range.start;
        let bytes = SharedByteRange::new(owner, range)?;
        Self::from_byte_range(ggml_type, logical_shape, bytes, alignment, source_offset)
    }

    fn from_byte_range(
        ggml_type: GgmlType,
        logical_shape: Shape,
        bytes: SharedByteRange,
        alignment: usize,
        source_offset: usize,
    ) -> Result<Self, QuantizedError> {
        if alignment == 0
            || !alignment.is_power_of_two()
            || !source_offset.is_multiple_of(alignment)
        {
            return Err(QuantizedError::InvalidAlignment);
        }
        if logical_shape.rank() != 2 {
            return Err(QuantizedError::InvalidRank);
        }
        let GgmlLayout::Quantized {
            block_elements,
            block_bytes,
        } = ggml_type.layout()
        else {
            return Err(QuantizedError::UnsupportedType(ggml_type));
        };
        if !matches!(
            ggml_type,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Mxfp4
                | GgmlType::Q1_0
                | GgmlType::Iq4Xs
                | GgmlType::Iq3Xxs
                | GgmlType::Iq3S
                | GgmlType::Iq2S
                | GgmlType::Q8_0
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
        ) {
            return Err(QuantizedError::UnsupportedType(ggml_type));
        }
        let [rows, columns]: [usize; 2] = logical_shape
            .dims()
            .try_into()
            .map_err(|_| QuantizedError::InvalidRank)?;
        if !columns.is_multiple_of(block_elements) {
            return Err(QuantizedError::InvalidGeometry);
        }
        let expected = rows
            .checked_mul(columns / block_elements)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(QuantizedError::Overflow)?;
        if bytes.len() != expected {
            return Err(QuantizedError::Length {
                expected,
                actual: bytes.len(),
            });
        }
        let mut desc = QuantizedBufferDesc {
            ggml_type,
            logical_shape,
            block_elements,
            block_bytes,
            bytes: expected,
            alignment: 1,
            identity: 0,
        };
        desc.identity = identity(&desc, &bytes);
        let value = Self { desc, bytes };
        value.validate()?;
        Ok(value)
    }

    pub fn descriptor(&self) -> &QuantizedBufferDesc {
        &self.desc
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    #[cfg(test)]
    pub(crate) fn shares_byte_owner(&self, other: &Self) -> bool {
        self.bytes.shares_owner(&other.bytes)
    }

    #[cfg(test)]
    pub(crate) fn byte_owner_count(&self) -> usize {
        self.bytes.owner_count()
    }

    pub fn validate(&self) -> Result<(), QuantizedError> {
        let rebuilt = Self::from_aligned_bytes_unchecked_identity(
            self.desc.ggml_type,
            self.desc.logical_shape.clone(),
            self.bytes.clone(),
            self.desc.alignment,
        )?;
        if rebuilt.desc != self.desc {
            return Err(QuantizedError::InvalidGeometry);
        }
        // Fail before execution when scales decode to NaN or infinity.
        for block in self.bytes.chunks_exact(self.desc.block_bytes) {
            decode(self.desc.ggml_type, block)?;
        }
        Ok(())
    }

    fn from_aligned_bytes_unchecked_identity(
        ggml_type: GgmlType,
        logical_shape: Shape,
        bytes: SharedByteRange,
        alignment: usize,
    ) -> Result<Self, QuantizedError> {
        if alignment == 0
            || !alignment.is_power_of_two()
            || alignment != 1
            || logical_shape.rank() != 2
        {
            return Err(QuantizedError::InvalidAlignment);
        }
        let GgmlLayout::Quantized {
            block_elements,
            block_bytes,
        } = ggml_type.layout()
        else {
            return Err(QuantizedError::UnsupportedType(ggml_type));
        };
        if !matches!(
            ggml_type,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Mxfp4
                | GgmlType::Q1_0
                | GgmlType::Iq4Xs
                | GgmlType::Iq3Xxs
                | GgmlType::Iq3S
                | GgmlType::Iq2S
                | GgmlType::Q8_0
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
        ) {
            return Err(QuantizedError::UnsupportedType(ggml_type));
        }
        let rows = logical_shape.dims()[0];
        let columns = logical_shape.dims()[1];
        if !columns.is_multiple_of(block_elements) {
            return Err(QuantizedError::InvalidGeometry);
        }
        let expected = rows
            .checked_mul(columns / block_elements)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(QuantizedError::Overflow)?;
        if bytes.len() != expected {
            return Err(QuantizedError::Length {
                expected,
                actual: bytes.len(),
            });
        }
        let mut desc = QuantizedBufferDesc {
            ggml_type,
            logical_shape,
            block_elements,
            block_bytes,
            bytes: expected,
            alignment,
            identity: 0,
        };
        desc.identity = identity(&desc, &bytes);
        Ok(Self { desc, bytes })
    }

    pub(crate) fn decode_block(&self, block: usize) -> Result<Vec<f32>, QuantizedError> {
        let start = block
            .checked_mul(self.desc.block_bytes)
            .ok_or(QuantizedError::Overflow)?;
        let end = start
            .checked_add(self.desc.block_bytes)
            .ok_or(QuantizedError::Overflow)?;
        decode(
            self.desc.ggml_type,
            self.bytes
                .get(start..end)
                .ok_or(QuantizedError::InvalidGeometry)?,
        )
    }

    /// Materializes a dense oracle value. Native quantized matmul never calls
    /// this method; it exists for loader compatibility and differential tests.
    pub fn dequantize_f32(&self) -> Result<TensorData, QuantizedError> {
        let mut values = Vec::with_capacity(
            self.desc
                .logical_shape
                .numel()
                .map_err(|_| QuantizedError::Overflow)?,
        );
        for block in 0..self.bytes.len() / self.desc.block_bytes {
            values.extend(self.decode_block(block)?);
        }
        TensorData::new(self.desc.logical_shape.clone(), values)
            .map_err(|_| QuantizedError::InvalidGeometry)
    }
}

fn decode(kind: GgmlType, block: &[u8]) -> Result<Vec<f32>, QuantizedError> {
    let values = match kind {
        GgmlType::Q4_0 => decode_q4_0_block(block)?.to_vec(),
        GgmlType::Q4_1 => decode_q4_1_block(block)?.to_vec(),
        GgmlType::Q5_0 => decode_q5_0_block(block)?.to_vec(),
        GgmlType::Q5_1 => decode_q5_1_block(block)?.to_vec(),
        GgmlType::Mxfp4 => decode_mxfp4_block(block)?.to_vec(),
        GgmlType::Q1_0 => decode_q1_0_block(block)?.to_vec(),
        GgmlType::Iq4Xs => decode_iq4_xs_block(block)?.to_vec(),
        GgmlType::Iq3Xxs => decode_iq3_xxs_block(block)?.to_vec(),
        GgmlType::Iq3S => decode_iq3_s_block(block)?.to_vec(),
        GgmlType::Iq2S => decode_iq2_s_block(block)?.to_vec(),
        GgmlType::Q8_0 => decode_q8_0_block(block)?.to_vec(),
        GgmlType::Q4K => decode_q4_k_block(block)?.to_vec(),
        GgmlType::Q5K => decode_q5_k_block(block)?.to_vec(),
        GgmlType::Q6K => decode_q6_k_block(block)?.to_vec(),
        _ => return Err(QuantizedError::UnsupportedType(kind)),
    };
    Ok(values)
}

impl From<BlockDecodeError> for QuantizedError {
    fn from(error: BlockDecodeError) -> Self {
        match error {
            BlockDecodeError::Length { expected, actual } => Self::Length { expected, actual },
            BlockDecodeError::NonFinite => Self::NonFinite,
        }
    }
}

fn identity(desc: &QuantizedBufferDesc, bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut write = |data: &[u8]| {
        for byte in data {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    };
    write(&desc.ggml_type.raw().to_le_bytes());
    for dim in desc.logical_shape.dims() {
        write(&(*dim as u64).to_le_bytes());
    }
    for value in [
        desc.block_elements,
        desc.block_bytes,
        desc.bytes,
        desc.alignment,
    ] {
        write(&(value as u64).to_le_bytes());
    }
    write(bytes);
    hash
}
