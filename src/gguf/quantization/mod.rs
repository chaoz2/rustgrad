//! Tensor-level materialization for audited GGML quantized layouts.

use self::blocks::{BlockDecodeError, decode_q4_k_block, decode_q6_k_block};
use super::{GgmlType, GgufError, GgufErrorKind, GgufTensor};
use crate::{TensorData, tensor::f16_to_f32};

pub(super) mod blocks;

pub(super) fn materialize_f32(tensor: &GgufTensor, bytes: &[u8]) -> Result<TensorData, GgufError> {
    let mut values = Vec::with_capacity(tensor.elements());
    let decoded = match tensor.ggml_type() {
        GgmlType::Q4_0 => decode_blocks(bytes, 18, &mut values, |block, values| {
            let d = half(&block[..2]);
            for &packed in &block[2..] {
                values.push((f32::from(packed & 15) - 8.) * d);
            }
            for &packed in &block[2..] {
                values.push((f32::from(packed >> 4) - 8.) * d);
            }
            finite(values)
        }),
        GgmlType::Q8_0 => decode_blocks(bytes, 34, &mut values, |block, values| {
            let d = half(&block[..2]);
            values.extend(block[2..].iter().map(|&q| f32::from(q as i8) * d));
            finite(values)
        }),
        GgmlType::Q4K => decode_blocks(bytes, 144, &mut values, |block, values| {
            values.extend(decode_q4_k_block(block)?);
            Ok(())
        }),
        GgmlType::Q6K => decode_blocks(bytes, 210, &mut values, |block, values| {
            values.extend(decode_q6_k_block(block)?);
            Ok(())
        }),
        _ => return Err(materialization_error(tensor)),
    };
    if decoded.is_err() || values.len() != tensor.elements() {
        return Err(materialization_error(tensor));
    }
    TensorData::new(tensor.shape().clone(), values).map_err(|_| materialization_error(tensor))
}

fn decode_blocks(
    bytes: &[u8],
    block_bytes: usize,
    values: &mut Vec<f32>,
    mut decode: impl FnMut(&[u8], &mut Vec<f32>) -> Result<(), BlockDecodeError>,
) -> Result<(), BlockDecodeError> {
    if !bytes.len().is_multiple_of(block_bytes) {
        return Err(BlockDecodeError::Length {
            expected: bytes.len().next_multiple_of(block_bytes),
            actual: bytes.len(),
        });
    }
    for block in bytes.chunks_exact(block_bytes) {
        decode(block, values)?;
    }
    finite(values)
}

fn half(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn finite(values: &[f32]) -> Result<(), BlockDecodeError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(BlockDecodeError::NonFinite)
    }
}

fn materialization_error(tensor: &GgufTensor) -> GgufError {
    GgufError::new(
        GgufErrorKind::QuantizedMaterialization {
            tensor: tensor.name().to_owned(),
            kind: tensor.ggml_type(),
        },
        tensor.raw_range().start,
    )
}
