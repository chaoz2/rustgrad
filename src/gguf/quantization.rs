//! Exact source-evidenced GGML Q4_0 and Q8_0 block decoding.

use super::{GgmlType, GgufError, GgufErrorKind, GgufTensor};
use crate::TensorData;

pub(super) fn materialize_f32(tensor: &GgufTensor, bytes: &[u8]) -> Result<TensorData, GgufError> {
    let mut values = Vec::with_capacity(tensor.elements());
    match tensor.ggml_type() {
        GgmlType::Q4_0 => {
            for block in bytes.chunks_exact(18) {
                let d = f16(block[0], block[1]);
                for &packed in &block[2..] {
                    values.push((f32::from(packed & 15) - 8.) * d);
                }
                for &packed in &block[2..] {
                    values.push((f32::from(packed >> 4) - 8.) * d);
                }
            }
        }
        GgmlType::Q8_0 => {
            for block in bytes.chunks_exact(34) {
                let d = f16(block[0], block[1]);
                for &q in &block[2..] {
                    values.push(f32::from(q as i8) * d);
                }
            }
        }
        kind => {
            return Err(GgufError::new(
                GgufErrorKind::QuantizedMaterialization {
                    tensor: tensor.name().to_owned(),
                    kind,
                },
                tensor.raw_range().start,
            ));
        }
    }
    if values.len() != tensor.elements() || values.iter().any(|x| !x.is_finite()) {
        return Err(GgufError::new(
            GgufErrorKind::QuantizedMaterialization {
                tensor: tensor.name().to_owned(),
                kind: tensor.ggml_type(),
            },
            tensor.raw_range().start,
        ));
    }
    TensorData::new(tensor.shape().clone(), values).map_err(|_| {
        GgufError::new(
            GgufErrorKind::QuantizedMaterialization {
                tensor: tensor.name().to_owned(),
                kind: tensor.ggml_type(),
            },
            tensor.raw_range().start,
        )
    })
}

fn f16(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let fraction = u32::from(bits & 1023);
    f32::from_bits(if exponent == 0 {
        sign
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | (u32::from(exponent + 112) << 23) | (fraction << 13)
    })
}
