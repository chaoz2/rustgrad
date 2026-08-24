//! Exact raw-storage encoding used by executable schedule artifacts.
use super::{Storage, TensorData};
use crate::uop::artifact::{
    ArtifactError, Reader, Writer, dtype, dtype_tag, read_shape, write_shape,
};

const MAX_ELEMENTS: usize = 64 << 20;

pub(crate) fn encode_into(w: &mut Writer, tensor: &TensorData) -> Result<(), ArtifactError> {
    write_shape(w, tensor.shape())?;
    w.u8(dtype_tag(tensor.dtype()))?;
    if tensor.len() > MAX_ELEMENTS {
        return Err(ArtifactError::Format("tensor element limit"));
    }
    w.u64(tensor.len() as u64)?;
    match tensor.storage() {
        Storage::Bool(xs) => {
            for x in xs {
                w.bool(*x)?;
            }
        }
        Storage::I8(xs) => {
            for x in xs {
                w.u8(*x as u8)?;
            }
        }
        Storage::U8(xs) => w.bytes(xs)?,
        Storage::I16(xs) => {
            for x in xs {
                w.u16(*x as u16)?;
            }
        }
        Storage::U16(xs) | Storage::F16(xs) | Storage::BF16(xs) => {
            for x in xs {
                w.u16(*x)?;
            }
        }
        Storage::I32(xs) => {
            for x in xs {
                w.u32(*x as u32)?;
            }
        }
        Storage::U32(xs) => {
            for x in xs {
                w.u32(*x)?;
            }
        }
        Storage::I64(xs) => {
            for x in xs {
                w.i64(*x)?;
            }
        }
        Storage::U64(xs) => {
            for x in xs {
                w.u64(*x)?;
            }
        }
        Storage::F32(xs) => {
            for x in xs {
                w.u32(x.to_bits())?;
            }
        }
        Storage::F64(xs) => {
            for x in xs {
                w.u64(x.to_bits())?;
            }
        }
    }
    Ok(())
}

pub(crate) fn decode_from(r: &mut Reader<'_>) -> Result<TensorData, ArtifactError> {
    let shape = read_shape(r)?;
    let dtype = dtype(r.u8()?)?;
    let count = usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("tensor count"))?;
    if count > MAX_ELEMENTS || shape.numel().ok() != Some(count) {
        return Err(ArtifactError::Format("tensor shape"));
    }
    let raw_bytes = count
        .checked_mul(dtype.itemsize())
        .ok_or(ArtifactError::Format("tensor byte count"))?;
    if raw_bytes > r.remaining() {
        return Err(ArtifactError::Format("truncated tensor"));
    }
    let storage = match dtype {
        crate::DType::Bool => {
            Storage::Bool((0..count).map(|_| r.bool()).collect::<Result<_, _>>()?)
        }
        crate::DType::I8 => Storage::I8(
            (0..count)
                .map(|_| r.u8().map(|x| x as i8))
                .collect::<Result<_, _>>()?,
        ),
        crate::DType::U8 => Storage::U8(r.take(count)?.to_vec()),
        crate::DType::I16 => Storage::I16(
            (0..count)
                .map(|_| r.u16().map(|x| x as i16))
                .collect::<Result<_, _>>()?,
        ),
        crate::DType::U16 => Storage::U16((0..count).map(|_| r.u16()).collect::<Result<_, _>>()?),
        crate::DType::I32 => Storage::I32(
            (0..count)
                .map(|_| r.u32().map(|x| x as i32))
                .collect::<Result<_, _>>()?,
        ),
        crate::DType::U32 => Storage::U32((0..count).map(|_| r.u32()).collect::<Result<_, _>>()?),
        crate::DType::I64 => Storage::I64((0..count).map(|_| r.i64()).collect::<Result<_, _>>()?),
        crate::DType::U64 => Storage::U64((0..count).map(|_| r.u64()).collect::<Result<_, _>>()?),
        crate::DType::F16 => Storage::F16((0..count).map(|_| r.u16()).collect::<Result<_, _>>()?),
        crate::DType::BF16 => Storage::BF16((0..count).map(|_| r.u16()).collect::<Result<_, _>>()?),
        crate::DType::F32 => Storage::F32(
            (0..count)
                .map(|_| r.u32().map(f32::from_bits))
                .collect::<Result<_, _>>()?,
        ),
        crate::DType::F64 => Storage::F64(
            (0..count)
                .map(|_| r.u64().map(f64::from_bits))
                .collect::<Result<_, _>>()?,
        ),
    };
    TensorData::from_storage(shape, storage).map_err(|_| ArtifactError::Format("tensor data"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    #[test]
    fn raw_bits_round_trip_for_every_storage_dtype() {
        let cases = vec![
            Storage::Bool(vec![false, true]),
            Storage::I8(vec![i8::MIN, i8::MAX]),
            Storage::U8(vec![0, u8::MAX]),
            Storage::I16(vec![i16::MIN, i16::MAX]),
            Storage::U16(vec![0, u16::MAX]),
            Storage::I32(vec![i32::MIN, i32::MAX]),
            Storage::U32(vec![0, u32::MAX]),
            Storage::I64(vec![i64::MIN, i64::MAX]),
            Storage::U64(vec![0, u64::MAX]),
            Storage::F16(vec![0x8000, 0x7e01]),
            Storage::BF16(vec![0x8000, 0x7fc1]),
            Storage::F32(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_1234),
            ]),
            Storage::F64(vec![
                f64::from_bits(0x8000_0000_0000_0000),
                f64::from_bits(0x7ff8_0000_0000_1234),
            ]),
        ];
        for storage in cases {
            let tensor = TensorData::from_storage(Shape::from([2]), storage).unwrap();
            let mut w = Writer::new();
            encode_into(&mut w, &tensor).unwrap();
            let mut r = Reader::new(&w.out);
            let decoded = decode_from(&mut r).unwrap();
            assert_eq!(decoded.dtype(), tensor.dtype());
            let mut again = Writer::new();
            encode_into(&mut again, &decoded).unwrap();
            assert_eq!(again.out, w.out);
            assert!(r.done());
        }
        assert_eq!(DType::Bool.bits(), 1);
    }
}
