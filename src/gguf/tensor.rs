use super::{GgufError, GgufErrorKind};
use crate::{DType, Shape, Storage, TensorData};
use std::ops::Range;

/// GGML storage types evidenced by tinygrad's checked-in GGUF loader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Iq3Xxs,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    Mxfp4,
    Q1_0,
}

impl GgmlType {
    pub(super) fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            18 => Self::Iq3Xxs,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            39 => Self::Mxfp4,
            41 => Self::Q1_0,
            _ => return None,
        })
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Iq3Xxs => 18,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::BF16 => 30,
            Self::Mxfp4 => 39,
            Self::Q1_0 => 41,
        }
    }

    pub const fn layout(self) -> GgmlLayout {
        match self {
            Self::F32 => GgmlLayout::Dense { dtype: DType::F32 },
            Self::F16 => GgmlLayout::Dense { dtype: DType::F16 },
            Self::I8 => GgmlLayout::Dense { dtype: DType::I8 },
            Self::I16 => GgmlLayout::Dense { dtype: DType::I16 },
            Self::I32 => GgmlLayout::Dense { dtype: DType::I32 },
            Self::I64 => GgmlLayout::Dense { dtype: DType::I64 },
            Self::F64 => GgmlLayout::Dense { dtype: DType::F64 },
            Self::BF16 => GgmlLayout::Dense { dtype: DType::BF16 },
            Self::Q4_0 => GgmlLayout::Quantized {
                block_elements: 32,
                block_bytes: 18,
            },
            Self::Q4_1 => GgmlLayout::Quantized {
                block_elements: 32,
                block_bytes: 20,
            },
            Self::Q5_0 => GgmlLayout::Quantized {
                block_elements: 32,
                block_bytes: 22,
            },
            Self::Q5_1 => GgmlLayout::Quantized {
                block_elements: 32,
                block_bytes: 24,
            },
            Self::Q8_0 => GgmlLayout::Quantized {
                block_elements: 32,
                block_bytes: 34,
            },
            Self::Q4K => GgmlLayout::Quantized {
                block_elements: 256,
                block_bytes: 144,
            },
            Self::Q5K => GgmlLayout::Quantized {
                block_elements: 256,
                block_bytes: 176,
            },
            Self::Q6K => GgmlLayout::Quantized {
                block_elements: 256,
                block_bytes: 210,
            },
            Self::Iq3Xxs => GgmlLayout::Quantized {
                block_elements: 256,
                block_bytes: 98,
            },
            Self::Iq3S => GgmlLayout::Quantized {
                block_elements: 256,
                block_bytes: 110,
            },
            Self::Iq2S => GgmlLayout::Quantized {
                block_elements: 256,
                block_bytes: 82,
            },
            Self::Iq4Xs => GgmlLayout::Quantized {
                block_elements: 256,
                block_bytes: 136,
            },
            Self::Mxfp4 => GgmlLayout::Quantized {
                block_elements: 32,
                block_bytes: 17,
            },
            Self::Q1_0 => GgmlLayout::Quantized {
                block_elements: 128,
                block_bytes: 18,
            },
        }
    }
}

/// Physical scalar or block-quantized payload layout.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GgmlLayout {
    Dense {
        dtype: DType,
    },
    Quantized {
        block_elements: usize,
        block_bytes: usize,
    },
}

/// A validated tensor inventory entry and exact source-byte range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTensor {
    pub(super) name: String,
    pub(super) dimensions: Vec<usize>,
    pub(super) shape: Shape,
    pub(super) elements: usize,
    pub(super) kind: GgmlType,
    pub(super) relative_offset: u64,
    pub(super) raw_range: Range<usize>,
}

impl GgufTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Dimensions in GGUF storage order (innermost first).
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Logical RustGrad shape (the reverse of GGUF storage-order dimensions).
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub const fn ggml_type(&self) -> GgmlType {
        self.kind
    }

    pub const fn layout(&self) -> GgmlLayout {
        self.kind.layout()
    }

    pub const fn relative_offset(&self) -> u64 {
        self.relative_offset
    }

    pub fn raw_range(&self) -> Range<usize> {
        self.raw_range.clone()
    }

    pub fn byte_len(&self) -> usize {
        self.raw_range.len()
    }

    pub const fn elements(&self) -> usize {
        self.elements
    }
}

pub(super) fn byte_len(
    tensor: &str,
    elements: usize,
    kind: GgmlType,
    offset: usize,
) -> Result<usize, GgufError> {
    match kind.layout() {
        GgmlLayout::Dense { dtype } => elements.checked_mul(dtype.itemsize()).ok_or_else(|| {
            GgufError::new(
                GgufErrorKind::ShapeOverflow {
                    tensor: tensor.to_owned(),
                },
                offset,
            )
        }),
        GgmlLayout::Quantized {
            block_elements,
            block_bytes,
        } => {
            if !elements.is_multiple_of(block_elements) {
                return Err(GgufError::new(
                    GgufErrorKind::BlockElementMismatch {
                        tensor: tensor.to_owned(),
                        elements,
                        block_elements,
                    },
                    offset,
                ));
            }
            (elements / block_elements)
                .checked_mul(block_bytes)
                .ok_or_else(|| {
                    GgufError::new(
                        GgufErrorKind::ShapeOverflow {
                            tensor: tensor.to_owned(),
                        },
                        offset,
                    )
                })
        }
    }
}

pub(super) fn materialize_dense(
    tensor: &GgufTensor,
    bytes: &[u8],
) -> Result<TensorData, GgufError> {
    let dtype = match tensor.layout() {
        GgmlLayout::Dense { dtype } => dtype,
        GgmlLayout::Quantized { .. } => {
            return Err(GgufError::new(
                GgufErrorKind::QuantizedMaterialization {
                    tensor: tensor.name.clone(),
                    kind: tensor.kind,
                },
                tensor.raw_range.start,
            ));
        }
    };
    let storage = match dtype {
        DType::I8 => Storage::I8(bytes.iter().map(|&value| value as i8).collect()),
        DType::F16 => Storage::F16(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        ),
        DType::BF16 => Storage::BF16(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        ),
        DType::I16 => Storage::I16(
            bytes
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        ),
        DType::I32 => Storage::I32(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
        ),
        DType::F32 => Storage::F32(
            bytes
                .chunks_exact(4)
                .map(|chunk| {
                    f32::from_bits(u32::from_le_bytes(
                        chunk.try_into().expect("four-byte chunk"),
                    ))
                })
                .collect(),
        ),
        DType::I64 => Storage::I64(
            bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
                .collect(),
        ),
        DType::F64 => Storage::F64(
            bytes
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_bits(u64::from_le_bytes(
                        chunk.try_into().expect("eight-byte chunk"),
                    ))
                })
                .collect(),
        ),
        _ => unreachable!("GGML dense type mapping is exhaustive"),
    };
    TensorData::from_storage(tensor.shape.clone(), storage).map_err(|_| {
        GgufError::new(
            GgufErrorKind::TensorRangeOutOfBounds {
                tensor: tensor.name.clone(),
            },
            tensor.raw_range.start,
        )
    })
}
