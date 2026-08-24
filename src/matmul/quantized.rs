//! Dense-F32 activation by exact GGML block-quantized weight planning.

use crate::{DType, NodeId, QuantizedBufferDesc, QuantizedTensorData, Shape, TensorData};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

/// Llama linear weights are `[out_features, in_features]`; execution computes
/// `activation * weight.transpose()` without transposing packed bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QuantizedMatmulOrientation {
    OutputByInput,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QuantizedMatmulPlan {
    pub activation: NodeId,
    pub weight: NodeId,
    pub output: NodeId,
    pub activation_shape: Shape,
    pub weight_desc: QuantizedBufferDesc,
    pub output_shape: Shape,
    pub activation_dtype: DType,
    pub output_dtype: DType,
    pub orientation: QuantizedMatmulOrientation,
    pub batch_shape: Vec<usize>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub activation_vector: bool,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizedMatmulError {
    InvalidIdentity,
    InvalidGeometry,
    InvalidDType,
    Descriptor(String),
    Overflow,
}

impl fmt::Display for QuantizedMatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "quantized matmul error: {self:?}")
    }
}

impl std::error::Error for QuantizedMatmulError {}

impl QuantizedMatmulPlan {
    pub fn new(
        activation: NodeId,
        weight: NodeId,
        output: NodeId,
        activation_shape: Shape,
        weight_desc: QuantizedBufferDesc,
    ) -> Result<Self, QuantizedMatmulError> {
        weight_desc
            .validate_metadata()
            .map_err(|error| QuantizedMatmulError::Descriptor(error.to_string()))?;
        let geometry = geometry(&activation_shape, &weight_desc.logical_shape)?;
        let mut plan = Self {
            activation,
            weight,
            output,
            activation_shape,
            weight_desc,
            output_shape: geometry.output_shape,
            activation_dtype: DType::F32,
            output_dtype: DType::F32,
            orientation: QuantizedMatmulOrientation::OutputByInput,
            batch_shape: geometry.batch_shape,
            m: geometry.m,
            n: geometry.n,
            k: geometry.k,
            activation_vector: geometry.activation_vector,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), QuantizedMatmulError> {
        if self.activation == self.weight
            || self.activation == self.output
            || self.weight == self.output
        {
            return Err(QuantizedMatmulError::InvalidIdentity);
        }
        self.weight_desc
            .validate_metadata()
            .map_err(|error| QuantizedMatmulError::Descriptor(error.to_string()))?;
        let geometry = geometry(&self.activation_shape, &self.weight_desc.logical_shape)?;
        if self.output_shape != geometry.output_shape
            || self.batch_shape != geometry.batch_shape
            || self.m != geometry.m
            || self.n != geometry.n
            || self.k != geometry.k
            || self.activation_vector != geometry.activation_vector
            || self.orientation != QuantizedMatmulOrientation::OutputByInput
        {
            return Err(QuantizedMatmulError::InvalidGeometry);
        }
        if self.activation_dtype != DType::F32 || self.output_dtype != DType::F32 {
            return Err(QuantizedMatmulError::InvalidDType);
        }
        self.activation_shape
            .numel()
            .and_then(|_| self.output_shape.numel())
            .map_err(|_| QuantizedMatmulError::Overflow)?;
        if self.cache_key != self.expected_cache_key() {
            return Err(QuantizedMatmulError::InvalidGeometry);
        }
        Ok(())
    }

    pub fn abi_nodes(&self) -> [NodeId; 3] {
        [self.activation, self.weight, self.output]
    }

    /// Independent blockwise reference. It decodes exactly one packed block at
    /// a time and never creates a full dense weight tensor.
    pub fn execute(
        &self,
        activation: &TensorData,
        weight: &QuantizedTensorData,
    ) -> Result<TensorData, QuantizedMatmulError> {
        self.validate()?;
        weight
            .validate()
            .map_err(|error| QuantizedMatmulError::Descriptor(error.to_string()))?;
        if activation.shape() != &self.activation_shape
            || activation.dtype() != DType::F32
            || weight.descriptor() != &self.weight_desc
        {
            return Err(QuantizedMatmulError::InvalidGeometry);
        }
        let output_len = self
            .output_shape
            .numel()
            .map_err(|_| QuantizedMatmulError::Overflow)?;
        let rows = if self.n == 0 { 0 } else { output_len / self.n };
        let blocks_per_row = if self.k == 0 {
            0
        } else {
            self.k / self.weight_desc.block_elements
        };
        let crate::Storage::F32(activations) = activation.storage() else {
            return Err(QuantizedMatmulError::InvalidDType);
        };
        let mut output = Vec::with_capacity(output_len);
        for row in 0..rows {
            for column in 0..self.n {
                let mut accumulator = 0.0f64;
                for block_in_row in 0..blocks_per_row {
                    let block = column
                        .checked_mul(blocks_per_row)
                        .and_then(|base| base.checked_add(block_in_row))
                        .ok_or(QuantizedMatmulError::Overflow)?;
                    let decoded = weight
                        .decode_block(block)
                        .map_err(|error| QuantizedMatmulError::Descriptor(error.to_string()))?;
                    let activation_base = row
                        .checked_mul(self.k)
                        .and_then(|base| {
                            base.checked_add(block_in_row * self.weight_desc.block_elements)
                        })
                        .ok_or(QuantizedMatmulError::Overflow)?;
                    for (lane, quant) in decoded.iter().enumerate() {
                        accumulator +=
                            f64::from(activations[activation_base + lane]) * f64::from(*quant);
                    }
                }
                output.push(accumulator as f32);
            }
        }
        TensorData::new(self.output_shape.clone(), output)
            .map_err(|_| QuantizedMatmulError::InvalidGeometry)
    }

    fn expected_cache_key(&self) -> u64 {
        let mut plan = self.clone();
        plan.cache_key = 0;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        plan.hash(&mut hasher);
        hasher.finish()
    }
}

struct Geometry {
    output_shape: Shape,
    batch_shape: Vec<usize>,
    m: usize,
    n: usize,
    k: usize,
    activation_vector: bool,
}

fn geometry(activation: &Shape, weight: &Shape) -> Result<Geometry, QuantizedMatmulError> {
    if activation.rank() == 0 || weight.rank() != 2 {
        return Err(QuantizedMatmulError::InvalidGeometry);
    }
    let k = *activation
        .dims()
        .last()
        .ok_or(QuantizedMatmulError::InvalidGeometry)?;
    let n = weight.dims()[0];
    if weight.dims()[1] != k {
        return Err(QuantizedMatmulError::InvalidGeometry);
    }
    let activation_vector = activation.rank() == 1;
    let m = if activation_vector {
        1
    } else {
        activation.dims()[activation.rank() - 2]
    };
    let batch_shape = if activation.rank() <= 2 {
        Vec::new()
    } else {
        activation.dims()[..activation.rank() - 2].to_vec()
    };
    let mut output = activation.dims()[..activation.rank() - 1].to_vec();
    output.push(n);
    let output_shape = Shape::new(output);
    output_shape
        .numel()
        .map_err(|_| QuantizedMatmulError::Overflow)?;
    Ok(Geometry {
        output_shape,
        batch_shape,
        m,
        n,
        k,
        activation_vector,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapturedBackendPolicy, CapturedBatch, CapturedReplayExecutor, CapturedReplayOptions,
        CapturedSchedule, CpuJit, GgmlType, UArg,
    };
    use std::collections::BTreeMap;

    fn half(bits: u16) -> [u8; 2] {
        bits.to_le_bytes()
    }

    fn q4_0_block(scale: u16, low: u8, high: u8) -> Vec<u8> {
        let mut out = half(scale).to_vec();
        out.extend(std::iter::repeat_n((low & 15) | ((high & 15) << 4), 16));
        out
    }

    fn q8_0_block(scale: u16) -> Vec<u8> {
        let mut out = half(scale).to_vec();
        out.extend((-16i8..16).map(|value| value as u8));
        out
    }

    fn q4_k_block() -> Vec<u8> {
        let scales = [1u8, 2, 3, 4, 17, 33, 49, 63];
        let mins = [0u8, 1, 2, 3, 16, 32, 48, 62];
        let mut packed = [0u8; 12];
        for lane in 0..4 {
            packed[lane] = scales[lane] | ((scales[4 + lane] >> 4) << 6);
            packed[4 + lane] = mins[lane] | ((mins[4 + lane] >> 4) << 6);
            packed[8 + lane] = (scales[4 + lane] & 15) | ((mins[4 + lane] & 15) << 4);
        }
        let mut out = Vec::with_capacity(144);
        out.extend(half(0x3800)); // d = 0.5
        out.extend(half(0x3400)); // dmin = 0.25
        out.extend(packed);
        for pair in 0..4 {
            let low = (pair * 2 + 1) as u8;
            let high = (pair * 2 + 2) as u8;
            out.extend(std::iter::repeat_n(low | (high << 4), 32));
        }
        out
    }

    fn q6_k_block() -> Vec<u8> {
        let mut out = vec![0u8; 210];
        for index in 0..256 {
            let raw = ((index * 29 + 7) & 63) as u8;
            let half_index = index / 128;
            let within = index % 128;
            out[half_index * 64 + within % 64] |= (raw & 15) << ((within / 64) * 4);
            out[128 + half_index * 32 + within % 32] |= ((raw >> 4) & 3) << ((within / 32) * 2);
        }
        for (index, scale) in (-8i8..8).enumerate() {
            out[192 + index] = scale as u8;
        }
        out[208..].copy_from_slice(&half(0x0001)); // smallest half subnormal
        out
    }

    fn fixture(kind: GgmlType, rows: usize, blocks_per_row: usize) -> QuantizedTensorData {
        let block = match kind {
            GgmlType::Q4_0 => q4_0_block(0x3c00, 3, 14),
            GgmlType::Q8_0 => q8_0_block(0x3800),
            GgmlType::Q4K => q4_k_block(),
            GgmlType::Q6K => q6_k_block(),
            _ => unreachable!(),
        };
        let crate::GgmlLayout::Quantized { block_elements, .. } = kind.layout() else {
            unreachable!()
        };
        let bytes = std::iter::repeat_n(block, rows * blocks_per_row)
            .flatten()
            .collect();
        QuantizedTensorData::new(
            kind,
            Shape::from([rows, blocks_per_row * block_elements]),
            bytes,
        )
        .unwrap()
    }

    fn dense_oracle(activation: &TensorData, weight: &QuantizedTensorData) -> TensorData {
        let crate::Storage::F32(a) = activation.storage() else {
            unreachable!()
        };
        let dense = weight.dequantize_f32().unwrap();
        let crate::Storage::F32(w) = dense.storage() else {
            unreachable!()
        };
        let n = weight.descriptor().logical_shape.dims()[0];
        let k = weight.descriptor().logical_shape.dims()[1];
        let rows = if k == 0 {
            activation.len()
        } else {
            a.len() / k
        };
        let mut out = Vec::with_capacity(rows * n);
        for row in 0..rows {
            for column in 0..n {
                let mut acc = 0.0f64;
                for inner in 0..k {
                    acc += f64::from(a[row * k + inner]) * f64::from(w[column * k + inner]);
                }
                out.push(acc as f32);
            }
        }
        let mut shape = activation.shape().dims()[..activation.shape().rank() - 1].to_vec();
        shape.push(n);
        TensorData::new(Shape::new(shape), out).unwrap()
    }

    #[test]
    fn exact_block_layouts_cover_nibbles_high_planes_and_subnormal_scale() {
        let q4 = fixture(GgmlType::Q4_0, 1, 1).dequantize_f32().unwrap();
        let crate::Storage::F32(values) = q4.storage() else {
            unreachable!()
        };
        assert_eq!(&values[..16], &[-5.0; 16]);
        assert_eq!(&values[16..], &[6.0; 16]);

        let q8 = fixture(GgmlType::Q8_0, 1, 1).dequantize_f32().unwrap();
        let crate::Storage::F32(values) = q8.storage() else {
            unreachable!()
        };
        assert_eq!(values[0], -8.0);
        assert_eq!(values[31], 7.5);

        let q4k = fixture(GgmlType::Q4K, 1, 1).dequantize_f32().unwrap();
        let crate::Storage::F32(values) = q4k.storage() else {
            unreachable!()
        };
        assert_eq!(values[0], 0.5);
        assert_eq!(values[128], 38.5);
        assert_eq!(values[255], 236.5);

        let q6k = fixture(GgmlType::Q6K, 1, 1).dequantize_f32().unwrap();
        let crate::Storage::F32(values) = q6k.storage() else {
            unreachable!()
        };
        assert!(values.iter().any(|value| *value != 0.0));
        assert!(values.iter().all(|value| value.is_finite()));
        assert_ne!(values[31], values[32], "high two-bit plane boundary");
    }

    #[test]
    fn blockwise_reference_matches_dense_oracle_for_vector_and_batches() {
        for kind in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
            let weight = fixture(kind, 3, 2);
            let k = weight.descriptor().logical_shape.dims()[1];
            for shape in [Shape::from([k]), Shape::from([2, 2, k])] {
                let len = shape.numel().unwrap();
                let activation = TensorData::new(
                    shape.clone(),
                    (0..len)
                        .map(|index| ((index % 13) as f32 - 6.0) / 7.0)
                        .collect(),
                )
                .unwrap();
                let plan = QuantizedMatmulPlan::new(
                    NodeId::from_index(10),
                    NodeId::from_index(11),
                    NodeId::from_index(12),
                    shape,
                    weight.descriptor().clone(),
                )
                .unwrap();
                assert_eq!(
                    plan.execute(&activation, &weight).unwrap(),
                    dense_oracle(&activation, &weight)
                );
            }
        }
    }

    #[test]
    fn artifact_native_cache_and_batch_keep_weight_packed() {
        for kind in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
            let weight = fixture(kind, 2, 2);
            let k = weight.descriptor().logical_shape.dims()[1];
            let capture = CapturedSchedule::capture_quantized_matmul(
                "activation",
                NodeId::from_index(20),
                NodeId::from_index(21),
                NodeId::from_index(22),
                Shape::from([2, k]),
                weight.clone(),
            )
            .unwrap();
            let bytes = capture.to_bytes().unwrap();
            assert_eq!(bytes, capture.to_bytes().unwrap());
            let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.to_bytes().unwrap(), bytes);
            assert!(decoded.constants.is_empty());
            assert_eq!(decoded.quantized_constants[&21], weight);
            assert!(
                crate::Schedule {
                    items: decoded.items.clone(),
                    value_bindings: vec![],
                }
                .internal_temporaries(&[NodeId::from_index(22)])
                .is_empty()
            );
            let UArg::QuantizedMatmul(plan) = decoded.items[0].kernel.arg() else {
                panic!("quantized matmul payload")
            };
            let rendered = CpuJit::render(&decoded.items[0].kernel).unwrap();
            assert_eq!(rendered.abi.buffers.len(), 2);
            assert_eq!(rendered.abi.quantized_buffers.len(), 1);
            assert_eq!(rendered.abi.quantized_buffers[0].desc, plan.weight_desc);
            assert!(!rendered.source.contains("float rg_weight["));

            let activation = TensorData::new(
                [2, k],
                (0..2 * k)
                    .map(|index| ((index % 9) as f32 - 4.0) / 5.0)
                    .collect(),
            )
            .unwrap();
            let bindings = BTreeMap::from([("activation".into(), activation.clone())]);
            let interpreter = decoded.replay(&bindings).unwrap().remove(0);
            let executor = CapturedReplayExecutor::default();
            let options = CapturedReplayOptions {
                backend: CapturedBackendPolicy::NativeJit { vectorized: false },
            };
            let first = executor.replay(&decoded, &bindings, options).unwrap();
            let second = executor.replay(&decoded, &bindings, options).unwrap();
            assert_eq!(first.outputs[0], interpreter);
            assert_eq!(second.outputs[0], interpreter);
            assert!(!first.trace.items[0].cache_hit);
            assert!(second.trace.items[0].cache_hit);
            assert_eq!(
                first.trace.items[0].packed_weight_bytes,
                weight.bytes().len()
            );
            let batch = CapturedBatch::new(&decoded, [bindings.clone(), bindings]).unwrap();
            let replayed = executor.replay_batch(&decoded, &batch, options).unwrap();
            assert_eq!(replayed.invocations[0].outputs[0], interpreter);
            assert_eq!(replayed.invocations[1].outputs[0], interpreter);

            let mut corrupt = bytes;
            let middle = corrupt.len() / 2;
            corrupt[middle] ^= 0x80;
            assert!(CapturedSchedule::from_bytes(&corrupt).is_err());
            if kind == GgmlType::Q4_0 {
                let mut missing = decoded.clone();
                missing.quantized_constants.clear();
                assert!(missing.to_bytes().is_err());
                let mut extra = decoded.clone();
                extra.quantized_constants.insert(999, weight.clone());
                assert!(extra.to_bytes().is_err());
                let mut bad_abi = decoded.clone();
                bad_abi.items[0].quantized_input_bindings[0].abi_index = 7;
                assert!(bad_abi.to_bytes().is_err());
            }
        }
    }

    #[test]
    fn malformed_geometry_alignment_lengths_and_zero_domains_fail_closed() {
        let block = q4_0_block(0x3c00, 8, 8);
        assert!(
            QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([1, 31]), block.clone()).is_err()
        );
        assert!(
            QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([1, 32]), block[..17].to_vec())
                .is_err()
        );
        assert!(
            QuantizedTensorData::from_aligned_bytes(
                GgmlType::Q4_0,
                Shape::from([1, 32]),
                block.clone(),
                3,
                0
            )
            .is_err()
        );
        assert!(QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([1, 1, 32]), block).is_err());

        let weight = fixture(GgmlType::Q4_0, 3, 1);
        assert!(
            QuantizedMatmulPlan::new(
                NodeId::from_index(1),
                NodeId::from_index(2),
                NodeId::from_index(3),
                Shape::from([33]),
                weight.descriptor().clone(),
            )
            .is_err()
        );
        let mut changed = q4_0_block(0x3c00, 7, 9);
        changed.extend(q4_0_block(0x3c00, 7, 9));
        changed.extend(q4_0_block(0x3c00, 7, 9));
        let changed =
            QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([3, 32]), changed).unwrap();
        assert_ne!(weight.descriptor().identity, changed.descriptor().identity);

        let zero_weight =
            QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([3, 0]), vec![]).unwrap();
        let capture = CapturedSchedule::capture_quantized_matmul(
            "activation",
            NodeId::from_index(30),
            NodeId::from_index(31),
            NodeId::from_index(32),
            Shape::from([2, 0]),
            zero_weight,
        )
        .unwrap();
        let input = TensorData::new([2, 0], vec![]).unwrap();
        let bindings = BTreeMap::from([("activation".into(), input)]);
        let interpreter = capture.replay(&bindings).unwrap().remove(0);
        assert_eq!(interpreter.shape(), &Shape::from([2, 3]));
        let crate::Storage::F32(values) = interpreter.storage() else {
            unreachable!()
        };
        assert_eq!(values, &vec![0.0; 6]);
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

        for (case, activation_shape, weight) in [
            (
                "zero-m",
                Shape::from([0, 32]),
                fixture(GgmlType::Q4_0, 3, 1),
            ),
            (
                "zero-n",
                Shape::from([2, 32]),
                QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([0, 32]), vec![]).unwrap(),
            ),
        ] {
            let capture = CapturedSchedule::capture_quantized_matmul(
                "activation",
                NodeId::from_index(40),
                NodeId::from_index(41),
                NodeId::from_index(42),
                activation_shape.clone(),
                weight,
            )
            .unwrap();
            let input = TensorData::new(
                activation_shape.clone(),
                vec![0.25; activation_shape.numel().unwrap()],
            )
            .unwrap();
            let bindings = BTreeMap::from([("activation".into(), input)]);
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
            assert_eq!(native.outputs[0], interpreter, "{case}");
            assert_eq!(interpreter.len(), 0, "{case}");
        }

        let capture = CapturedSchedule::capture_quantized_matmul(
            "activation",
            NodeId::from_index(50),
            NodeId::from_index(51),
            NodeId::from_index(52),
            Shape::from([1, 32]),
            fixture(GgmlType::Q4_0, 2, 1),
        )
        .unwrap();
        let good = BTreeMap::from([(
            "activation".into(),
            TensorData::new([1, 32], vec![1.0; 32]).unwrap(),
        )]);
        let bad = BTreeMap::from([(
            "activation".into(),
            TensorData::new([2, 32], vec![1.0; 64]).unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        assert!(CapturedBatch::new(&capture, [good, bad]).is_err());
        assert_eq!(
            executor.compile_cache_len(false),
            0,
            "batch rejected before compile"
        );
    }
}
