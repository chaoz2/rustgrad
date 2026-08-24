//! Exact blockwise row lookup from a rank-two GGML packed tensor.

use super::{QuantizedBufferDesc, QuantizedTensorData};
use crate::{DType, Scalar, Shape, TensorData};
use std::fmt;

/// Immutable row-gather geometry tied to one exact packed buffer identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QuantizedRowGatherPlan {
    weight: QuantizedBufferDesc,
}

/// Structured rejection before or during packed row lookup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizedRowGatherError {
    Descriptor(String),
    WeightMismatch,
    InvalidIndexDType(DType),
    NegativeIndex {
        position: usize,
        index: i64,
    },
    IndexOutOfBounds {
        position: usize,
        index: u64,
        rows: usize,
    },
    Overflow,
    Output(String),
}

impl fmt::Display for QuantizedRowGatherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "quantized row gather error: {self:?}")
    }
}

impl std::error::Error for QuantizedRowGatherError {}

impl QuantizedRowGatherPlan {
    /// Binds lookup geometry and owner identity without decoding packed blocks.
    pub fn new(weight: &QuantizedTensorData) -> Result<Self, QuantizedRowGatherError> {
        weight
            .descriptor()
            .validate_metadata()
            .map_err(|error| QuantizedRowGatherError::Descriptor(error.to_string()))?;
        Ok(Self {
            weight: weight.descriptor().clone(),
        })
    }

    /// Returns the exact packed owner descriptor accepted by this plan.
    pub const fn weight_descriptor(&self) -> &QuantizedBufferDesc {
        &self.weight
    }

    /// Materializes only the selected rows. The complete index tensor is
    /// validated before packed payload validation or row decoding, so malformed
    /// input has no partial observable result.
    pub fn execute(
        &self,
        indices: &TensorData,
        weight: &QuantizedTensorData,
    ) -> Result<TensorData, QuantizedRowGatherError> {
        if weight.descriptor() != &self.weight {
            return Err(QuantizedRowGatherError::WeightMismatch);
        }
        if !indices.dtype().is_integer() {
            return Err(QuantizedRowGatherError::InvalidIndexDType(indices.dtype()));
        }
        let rows = self.weight.logical_shape.dims()[0];
        let columns = self.weight.logical_shape.dims()[1];
        let selected = (0..indices.len())
            .map(|position| match indices.scalar_at(position) {
                Scalar::I(index) if index < 0 => {
                    Err(QuantizedRowGatherError::NegativeIndex { position, index })
                }
                Scalar::I(index) => {
                    usize::try_from(index).map_err(|_| QuantizedRowGatherError::IndexOutOfBounds {
                        position,
                        index: index as u64,
                        rows,
                    })
                }
                Scalar::U(index) => {
                    usize::try_from(index).map_err(|_| QuantizedRowGatherError::IndexOutOfBounds {
                        position,
                        index,
                        rows,
                    })
                }
                _ => unreachable!("integer dtype has integer scalar storage"),
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (position, &index) in selected.iter().enumerate() {
            if index >= rows {
                return Err(QuantizedRowGatherError::IndexOutOfBounds {
                    position,
                    index: index as u64,
                    rows,
                });
            }
        }
        weight
            .validate()
            .map_err(|error| QuantizedRowGatherError::Descriptor(error.to_string()))?;

        let mut output_shape = indices.shape().dims().to_vec();
        output_shape.push(columns);
        let output_len = indices
            .len()
            .checked_mul(columns)
            .ok_or(QuantizedRowGatherError::Overflow)?;
        let blocks_per_row = columns / self.weight.block_elements;
        let mut output = Vec::with_capacity(output_len);
        for row in selected {
            let first_block = row
                .checked_mul(blocks_per_row)
                .ok_or(QuantizedRowGatherError::Overflow)?;
            for block in first_block..first_block + blocks_per_row {
                output.extend(
                    weight
                        .decode_block(block)
                        .map_err(|error| QuantizedRowGatherError::Descriptor(error.to_string()))?,
                );
            }
        }
        TensorData::new(Shape::from(output_shape), output)
            .map_err(|error| QuantizedRowGatherError::Output(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GgmlType, Storage};

    fn block(kind: GgmlType) -> Vec<u8> {
        match kind {
            GgmlType::Q4_0 => {
                let mut block = 1.0f32.to_bits().to_le_bytes()[2..].to_vec();
                block.extend([0x39; 16]);
                block
            }
            GgmlType::Q8_0 => {
                let mut block = 0.25f32.to_bits().to_le_bytes()[2..].to_vec();
                block.extend((0..32).map(|value| value as i8 as u8));
                block
            }
            GgmlType::Q4K => {
                let mut block = vec![0; 144];
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                block[4..16].fill(0x21);
                block[16..].fill(0x73);
                block
            }
            GgmlType::Q6K => {
                let mut block = vec![0u8; 210];
                block[..128].fill(0x95);
                block[128..192].fill(0x02);
                for (index, value) in block[192..208].iter_mut().enumerate() {
                    *value = (index as i8 - 8) as u8;
                }
                block[208..].copy_from_slice(&0x3400u16.to_le_bytes());
                block
            }
            _ => unreachable!(),
        }
    }

    fn fixture(kind: GgmlType) -> QuantizedTensorData {
        let crate::GgmlLayout::Quantized {
            block_elements: elements,
            ..
        } = kind.layout()
        else {
            unreachable!()
        };
        let mut bytes = Vec::new();
        for row in 0..3 {
            let mut row_block = block(kind);
            row_block[row + 2] ^= (row + 1) as u8;
            bytes.extend(row_block);
        }
        QuantizedTensorData::new(kind, Shape::from([3, elements]), bytes).unwrap()
    }

    #[test]
    fn all_audited_formats_gather_batches_repeats_and_match_dense_control() {
        for kind in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
            let weight = fixture(kind);
            let indices =
                TensorData::from_scalars([2, 3], DType::I64, [2, 0, 2, 1, 1, 0].map(Scalar::I))
                    .unwrap();
            let output = QuantizedRowGatherPlan::new(&weight)
                .unwrap()
                .execute(&indices, &weight)
                .unwrap();
            assert_eq!(
                output.shape().dims(),
                &[2, 3, weight.descriptor().block_elements]
            );
            let dense = weight.dequantize_f32().unwrap();
            let columns = weight.descriptor().block_elements;
            for (position, row) in [2usize, 0, 2, 1, 1, 0].into_iter().enumerate() {
                assert_eq!(
                    &output.values()[position * columns..(position + 1) * columns],
                    &dense.values()[row * columns..(row + 1) * columns]
                );
            }
        }
    }

    #[test]
    fn indices_and_exact_owner_are_preflighted() {
        let weight = fixture(GgmlType::Q4_0);
        let plan = QuantizedRowGatherPlan::new(&weight).unwrap();
        let negative =
            TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(-1)]).unwrap();
        assert_eq!(
            plan.execute(&negative, &weight).unwrap_err(),
            QuantizedRowGatherError::NegativeIndex {
                position: 1,
                index: -1
            }
        );
        let outside = TensorData::from_storage([1], Storage::U64(vec![u64::MAX])).unwrap();
        assert!(matches!(
            plan.execute(&outside, &weight),
            Err(QuantizedRowGatherError::IndexOutOfBounds { position: 0, .. })
        ));
        let float = TensorData::new([1], vec![0.0]).unwrap();
        assert_eq!(
            plan.execute(&float, &weight).unwrap_err(),
            QuantizedRowGatherError::InvalidIndexDType(DType::F32)
        );

        let mut changed_bytes = weight.bytes().to_vec();
        changed_bytes[2] ^= 1;
        let changed = QuantizedTensorData::new(
            GgmlType::Q4_0,
            weight.descriptor().logical_shape.clone(),
            changed_bytes,
        )
        .unwrap();
        assert_eq!(
            plan.execute(
                &TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
                &changed,
            )
            .unwrap_err(),
            QuantizedRowGatherError::WeightMismatch
        );
    }
}
