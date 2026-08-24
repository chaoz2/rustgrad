//! Exact blockwise row lookup from a rank-two GGML packed tensor.

use super::{QuantizedBufferDesc, QuantizedTensorData};
use crate::{DType, NodeId, Scalar, Shape, TensorData};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

/// Immutable row-gather geometry tied to one exact packed buffer identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QuantizedRowGatherPlan {
    pub indices: NodeId,
    pub weight: NodeId,
    pub output: NodeId,
    pub indices_shape: Shape,
    pub indices_dtype: DType,
    pub weight_desc: QuantizedBufferDesc,
    pub output_shape: Shape,
    pub output_dtype: DType,
    pub cache_key: u64,
}

/// Structured rejection before or during packed row lookup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizedRowGatherError {
    Descriptor(String),
    InvalidIdentity,
    InvalidGeometry,
    InvalidDType,
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
    pub fn new(
        indices: NodeId,
        weight_node: NodeId,
        output: NodeId,
        indices_shape: Shape,
        indices_dtype: DType,
        weight: &QuantizedTensorData,
    ) -> Result<Self, QuantizedRowGatherError> {
        weight
            .descriptor()
            .validate_metadata()
            .map_err(|error| QuantizedRowGatherError::Descriptor(error.to_string()))?;
        let mut output_dims = indices_shape.dims().to_vec();
        output_dims.push(weight.descriptor().logical_shape.dims()[1]);
        let mut plan = Self {
            indices,
            weight: weight_node,
            output,
            indices_shape,
            indices_dtype,
            weight_desc: weight.descriptor().clone(),
            output_shape: Shape::from(output_dims),
            output_dtype: DType::F32,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        plan.validate()?;
        Ok(plan)
    }

    /// Returns the exact packed owner descriptor accepted by this plan.
    pub const fn weight_descriptor(&self) -> &QuantizedBufferDesc {
        &self.weight_desc
    }

    pub fn validate(&self) -> Result<(), QuantizedRowGatherError> {
        if self.indices == self.weight || self.indices == self.output || self.weight == self.output
        {
            return Err(QuantizedRowGatherError::InvalidIdentity);
        }
        self.weight_desc
            .validate_metadata()
            .map_err(|error| QuantizedRowGatherError::Descriptor(error.to_string()))?;
        if !self.indices_dtype.is_integer() || self.output_dtype != DType::F32 {
            return Err(QuantizedRowGatherError::InvalidDType);
        }
        let index_elements = self
            .indices_shape
            .numel()
            .map_err(|_| QuantizedRowGatherError::Overflow)?;
        index_elements
            .checked_mul(self.weight_desc.logical_shape.dims()[1])
            .ok_or(QuantizedRowGatherError::Overflow)?;
        let mut output_dims = self.indices_shape.dims().to_vec();
        output_dims.push(self.weight_desc.logical_shape.dims()[1]);
        if self.output_shape.dims() != output_dims || self.cache_key != self.expected_cache_key() {
            return Err(QuantizedRowGatherError::InvalidGeometry);
        }
        Ok(())
    }

    fn expected_cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.indices.hash(&mut hasher);
        self.weight.hash(&mut hasher);
        self.output.hash(&mut hasher);
        self.indices_shape.hash(&mut hasher);
        self.indices_dtype.hash(&mut hasher);
        self.weight_desc.hash(&mut hasher);
        self.output_shape.hash(&mut hasher);
        self.output_dtype.hash(&mut hasher);
        hasher.finish()
    }

    /// Validates every runtime index without allocating or decoding a block.
    pub fn preflight_indices(
        &self,
        indices: &TensorData,
    ) -> Result<Vec<usize>, QuantizedRowGatherError> {
        self.validate()?;
        if indices.shape() != &self.indices_shape {
            return Err(QuantizedRowGatherError::InvalidGeometry);
        }
        if indices.dtype() != self.indices_dtype {
            return Err(QuantizedRowGatherError::InvalidIndexDType(indices.dtype()));
        }
        checked_indices(indices, self.weight_desc.logical_shape.dims()[0])
    }

    /// Materializes only the selected rows. The complete index tensor is
    /// validated before packed payload validation or row decoding, so malformed
    /// input has no partial observable result.
    pub fn execute(
        &self,
        indices: &TensorData,
        weight: &QuantizedTensorData,
    ) -> Result<TensorData, QuantizedRowGatherError> {
        self.validate()?;
        if weight.descriptor() != &self.weight_desc {
            return Err(QuantizedRowGatherError::WeightMismatch);
        }
        let selected = self.preflight_indices(indices)?;
        weight
            .validate()
            .map_err(|error| QuantizedRowGatherError::Descriptor(error.to_string()))?;

        let columns = self.weight_desc.logical_shape.dims()[1];
        let output_len = indices
            .len()
            .checked_mul(columns)
            .ok_or(QuantizedRowGatherError::Overflow)?;
        let blocks_per_row = columns / self.weight_desc.block_elements;
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
        TensorData::new(self.output_shape.clone(), output)
            .map_err(|error| QuantizedRowGatherError::Output(error.to_string()))
    }
}

fn checked_indices(
    indices: &TensorData,
    rows: usize,
) -> Result<Vec<usize>, QuantizedRowGatherError> {
    if !indices.dtype().is_integer() {
        return Err(QuantizedRowGatherError::InvalidIndexDType(indices.dtype()));
    }
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
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapturedBackendPolicy, CapturedBatch, CapturedReplayExecutor, CapturedReplayOptions,
        CapturedSchedule, CpuJit, GgmlType, ItemBackend, Storage, UArg,
    };
    use std::collections::BTreeMap;

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
            let output = QuantizedRowGatherPlan::new(
                NodeId::from_index(0),
                NodeId::from_index(1),
                NodeId::from_index(2),
                indices.shape().clone(),
                indices.dtype(),
                &weight,
            )
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
        let plan = QuantizedRowGatherPlan::new(
            NodeId::from_index(0),
            NodeId::from_index(1),
            NodeId::from_index(2),
            Shape::from([2]),
            DType::I64,
            &weight,
        )
        .unwrap();
        let negative =
            TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(-1)]).unwrap();
        assert_eq!(
            plan.execute(&negative, &weight).unwrap_err(),
            QuantizedRowGatherError::NegativeIndex {
                position: 1,
                index: -1
            }
        );
        let outside = TensorData::from_storage([2], Storage::I64(vec![0, 3])).unwrap();
        assert!(matches!(
            plan.execute(&outside, &weight),
            Err(QuantizedRowGatherError::IndexOutOfBounds { position: 1, .. })
        ));
        let float = TensorData::new([2], vec![0.0, 1.0]).unwrap();
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
                &TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(1)]).unwrap(),
                &changed,
            )
            .unwrap_err(),
            QuantizedRowGatherError::WeightMismatch
        );
    }

    #[test]
    fn captured_interpreter_and_strict_native_decode_only_selected_rows() {
        for kind in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
            let weight = fixture(kind);
            let capture = CapturedSchedule::capture_quantized_row_gather(
                "indices",
                NodeId::from_index(10),
                NodeId::from_index(11),
                NodeId::from_index(12),
                Shape::from([2, 3]),
                DType::I64,
                weight.clone(),
            )
            .unwrap();
            let bytes = capture.to_bytes().unwrap();
            let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.to_bytes().unwrap(), bytes);
            assert!(decoded.constants.is_empty());
            assert_eq!(decoded.quantized_constants[&11], weight);
            let UArg::QuantizedRowGather(plan) = decoded.items[0].kernel.arg() else {
                panic!("quantized row gather payload")
            };
            assert_eq!(plan.indices_shape, Shape::from([2, 3]));
            let rendered = CpuJit::render(&decoded.items[0].kernel).unwrap();
            assert_eq!(rendered.abi.pointer_order.len(), 3);
            assert_eq!(rendered.abi.quantized_buffers.len(), 1);
            assert!(!rendered.source.contains("float rg_weight["));

            let indices =
                TensorData::from_scalars([2, 3], DType::I64, [2, 0, 2, 1, 1, 0].map(Scalar::I))
                    .unwrap();
            let bindings = BTreeMap::from([("indices".into(), indices)]);
            let interpreter = decoded.replay(&bindings).unwrap().remove(0);
            let executor = CapturedReplayExecutor::default();
            let options = CapturedReplayOptions {
                backend: CapturedBackendPolicy::NativeJit { vectorized: false },
            };
            let first = executor.replay(&decoded, &bindings, options).unwrap();
            let second = executor.replay(&decoded, &bindings, options).unwrap();
            assert_eq!(first.outputs[0], interpreter);
            assert_eq!(second.outputs[0], interpreter);
            assert_eq!(first.trace.items[0].backend, ItemBackend::NativeJit);
            assert!(!first.trace.items[0].cache_hit);
            assert!(second.trace.items[0].cache_hit);
            assert_eq!(
                first.trace.items[0].packed_weight_bytes,
                weight.bytes().len()
            );
            let batch = CapturedBatch::new(&decoded, [bindings.clone(), bindings]).unwrap();
            let batch = executor.replay_batch(&decoded, &batch, options).unwrap();
            assert_eq!(batch.invocations[0].outputs[0], interpreter);
            assert_eq!(batch.invocations[1].outputs[0], interpreter);
        }
    }

    #[test]
    fn gather_batch_preflight_rejects_all_indices_before_native_compile() {
        let capture = CapturedSchedule::capture_quantized_row_gather(
            "indices",
            NodeId::from_index(20),
            NodeId::from_index(21),
            NodeId::from_index(22),
            Shape::from([2]),
            DType::I64,
            fixture(GgmlType::Q4_0),
        )
        .unwrap();
        let good = BTreeMap::from([(
            "indices".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(2)]).unwrap(),
        )]);
        let bad = BTreeMap::from([(
            "indices".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(3)]).unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        assert!(CapturedBatch::new(&capture, [good, bad]).is_err());
        assert_eq!(executor.compile_cache_len(false), 0);

        let mut malformed = capture.clone();
        malformed.items[0].quantized_input_bindings[0].abi_index = 7;
        assert!(malformed.to_bytes().is_err());
        let mut missing = capture;
        missing.quantized_constants.clear();
        assert!(missing.to_bytes().is_err());
    }

    #[test]
    fn empty_index_domains_replay_without_touching_packed_rows() {
        let capture = CapturedSchedule::capture_quantized_row_gather(
            "indices",
            NodeId::from_index(30),
            NodeId::from_index(31),
            NodeId::from_index(32),
            Shape::from([0, 2]),
            DType::U32,
            fixture(GgmlType::Q8_0),
        )
        .unwrap();
        let bindings = BTreeMap::from([(
            "indices".into(),
            TensorData::from_storage([0, 2], Storage::U32(vec![])).unwrap(),
        )]);
        let interpreter = capture.replay(&bindings).unwrap().remove(0);
        assert_eq!(interpreter.shape(), &Shape::from([0, 2, 32]));
        let native = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(native.outputs[0], interpreter);
    }

    #[test]
    fn scalar_indices_cover_every_integer_storage_dtype_natively() {
        for dtype in [
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
        ] {
            let capture = CapturedSchedule::capture_quantized_row_gather(
                "index",
                NodeId::from_index(40),
                NodeId::from_index(41),
                NodeId::from_index(42),
                Shape::from([]),
                dtype,
                fixture(GgmlType::Q4_0),
            )
            .unwrap();
            let scalar = if matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
                Scalar::I(1)
            } else {
                Scalar::U(1)
            };
            let bindings =
                BTreeMap::from([("index".into(), TensorData::scalar_with_dtype(scalar, dtype))]);
            let interpreter = capture.replay(&bindings).unwrap().remove(0);
            let native = CapturedReplayExecutor::default()
                .replay(
                    &capture,
                    &bindings,
                    CapturedReplayOptions {
                        backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                    },
                )
                .unwrap();
            assert_eq!(native.outputs[0], interpreter, "{dtype:?}");
            assert_eq!(interpreter.shape(), &Shape::from([32]));
        }
    }
}
