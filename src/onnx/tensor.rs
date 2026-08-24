//! Checked TensorProto dtype and owned-storage decoding.

use super::{
    bad,
    wire::{Msg, one_varint, var},
};
use crate::{DType, Result, Shape, TensorData};

pub(super) fn onnx_dtype(x: u64) -> Result<DType> {
    match x {
        1 => Ok(DType::F32),
        11 => Ok(DType::F64),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        9 => Ok(DType::Bool),
        10 => Ok(DType::F16),
        16 => Ok(DType::BF16),
        2 => Ok(DType::U8),
        3 => Ok(DType::I8),
        5 => Ok(DType::I16),
        4 => Ok(DType::U16),
        12 => Ok(DType::U32),
        13 => Ok(DType::U64),
        _ => Err(bad("unsupported ONNX dtype")),
    }
}

pub(super) fn tensor(m: Msg<'_>) -> Result<(String, TensorData)> {
    let name = m
        .string(8)?
        .ok_or_else(|| bad("ONNX initializer lacks name"))?
        .to_owned();
    if name.is_empty() {
        return Err(bad("empty ONNX initializer name"));
    }
    Ok((name, tensor_data(m)?))
}
pub(super) fn tensor_data(m: Msg<'_>) -> Result<TensorData> {
    if !m.bytes(13)?.is_empty() {
        return Err(bad("ONNX external tensor data is unsupported"));
    }
    let dtype = onnx_dtype(one_varint(&m, 2, "tensor dtype")?)?;
    let dims = m.packed(1)?;
    let shape = Shape::new(
        dims.into_iter()
            .map(|x| {
                usize::try_from(i64::try_from(x).map_err(|_| bad("negative ONNX dimension"))?)
                    .map_err(|_| bad("ONNX dimension overflow"))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let raw = m.bytes(9)?;
    let typed = typed_tensor_bytes(&m, dtype, shape.numel()?)?;
    if !raw.is_empty() && !typed.is_empty() {
        return Err(bad("ONNX tensor raw_data conflicts with typed data"));
    }
    let data = match (raw.as_slice(), typed.as_slice()) {
        ([x], []) => x.to_vec(),
        ([], [x]) => x.clone(),
        ([], []) => return Err(bad("ONNX tensor lacks data")),
        _ => return Err(bad("duplicate ONNX tensor data field")),
    };
    TensorData::from_le_bytes(shape, dtype, &data)
        .map_err(|error| bad(format!("invalid ONNX tensor data: {error}")))
}
fn typed_tensor_bytes(m: &Msg<'_>, dtype: DType, count: usize) -> Result<Vec<Vec<u8>>> {
    let f = m.fields()?;
    let fields: Vec<_> = f
        .iter()
        .filter(|(i, _, _)| matches!(*i, 4 | 5 | 7 | 10 | 11))
        .collect();
    if fields.is_empty() {
        return Ok(vec![]);
    }
    let (mut out, field) = (
        Vec::new(),
        match dtype {
            DType::F32 => 4,
            DType::F64 => 10,
            DType::I64 => 7,
            DType::U64 => 11,
            DType::I32
            | DType::U8
            | DType::I8
            | DType::I16
            | DType::U16
            | DType::U32
            | DType::Bool
            | DType::F16
            | DType::BF16 => 5,
        },
    );
    for (i, w, b) in fields {
        if *i != field {
            return Err(bad("typed field incompatible with dtype"));
        }
        if matches!(dtype, DType::F32 | DType::F64) {
            if *w != if dtype == DType::F32 { 5 } else { 1 } {
                return Err(bad("typed float wire"));
            }
            out.extend_from_slice(b);
            continue;
        }
        let mut vals = Vec::new();
        if *w == 0 {
            let mut at = 0;
            vals.push(var(b, &mut at)?)
        } else if *w == 2 {
            let mut at = 0;
            while at < b.len() {
                vals.push(var(b, &mut at)?)
            }
        } else {
            return Err(bad("typed integer wire"));
        }
        for v in vals {
            match dtype {
                DType::I32 => out.extend_from_slice(&(v as u32 as i32).to_le_bytes()),
                DType::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
                DType::U8 => out.push(u8::try_from(v).map_err(|_| bad("u8 range"))?),
                DType::I8 => out.push(i8::try_from(v as i64).map_err(|_| bad("i8 range"))? as u8),
                DType::I16 => out.extend_from_slice(
                    &(i16::try_from(v as i64).map_err(|_| bad("i16 range"))?).to_le_bytes(),
                ),
                DType::U16 => out.extend_from_slice(
                    &u16::try_from(v)
                        .map_err(|_| bad("u16 range"))?
                        .to_le_bytes(),
                ),
                DType::U32 => out.extend_from_slice(
                    &u32::try_from(v)
                        .map_err(|_| bad("u32 range"))?
                        .to_le_bytes(),
                ),
                DType::U64 => out.extend_from_slice(&v.to_le_bytes()),
                DType::Bool => out.push(if v == 0 {
                    0
                } else if v == 1 {
                    1
                } else {
                    return Err(bad("bool range"));
                }),
                DType::F16 | DType::BF16 => out.extend_from_slice(
                    &(u16::try_from(v).map_err(|_| bad("half range"))?).to_le_bytes(),
                ),
                _ => {}
            }
        }
    }
    if out.len()
        != count
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("typed overflow"))?
    {
        return Err(bad("typed count mismatch"));
    }
    Ok(vec![out])
}
