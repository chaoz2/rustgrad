use super::dtype::DType;
use super::scalar::{
    Scalar, bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, scalar_to_i8, scalar_to_i16,
    scalar_to_i32, scalar_to_u8, scalar_to_u16, scalar_to_u32,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Storage {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::U8(_) => DType::U8,
            Self::I16(_) => DType::I16,
            Self::U16(_) => DType::U16,
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::U64(_) => DType::U64,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn scalar(&self, index: usize) -> Scalar {
        match self {
            Self::Bool(v) => Scalar::Bool(v[index]),
            Self::I8(v) => Scalar::I(v[index] as i64),
            Self::U8(v) => Scalar::U(v[index] as u64),
            Self::I16(v) => Scalar::I(v[index] as i64),
            Self::U16(v) => Scalar::U(v[index] as u64),
            Self::I32(v) => Scalar::I(v[index] as i64),
            Self::U32(v) => Scalar::U(v[index] as u64),
            Self::I64(v) => Scalar::I(v[index]),
            Self::U64(v) => Scalar::U(v[index]),
            Self::F16(v) => Scalar::F(f16_to_f32(v[index]) as f64),
            Self::BF16(v) => Scalar::F(bf16_to_f32(v[index]) as f64),
            Self::F32(v) => Scalar::F(v[index] as f64),
            Self::F64(v) => Scalar::F(v[index]),
        }
    }

    pub fn from_scalars(dtype: DType, values: impl IntoIterator<Item = Scalar>) -> Self {
        let values: Vec<_> = values.into_iter().collect();
        match dtype {
            DType::Bool => Self::Bool(values.iter().map(|x| x.as_bool()).collect()),
            DType::I8 => Self::I8(values.iter().map(|x| scalar_to_i8(*x)).collect()),
            DType::U8 => Self::U8(values.iter().map(|x| scalar_to_u8(*x)).collect()),
            DType::I16 => Self::I16(values.iter().map(|x| scalar_to_i16(*x)).collect()),
            DType::U16 => Self::U16(values.iter().map(|x| scalar_to_u16(*x)).collect()),
            DType::I32 => Self::I32(values.iter().map(|x| scalar_to_i32(*x)).collect()),
            DType::U32 => Self::U32(values.iter().map(|x| scalar_to_u32(*x)).collect()),
            DType::I64 => Self::I64(values.iter().map(|x| x.as_i64()).collect()),
            DType::U64 => Self::U64(values.iter().map(|x| x.as_u64()).collect()),
            DType::F16 => Self::F16(
                values
                    .iter()
                    .map(|x| f32_to_f16(x.as_f64() as f32))
                    .collect(),
            ),
            DType::BF16 => Self::BF16(
                values
                    .iter()
                    .map(|x| f32_to_bf16(x.as_f64() as f32))
                    .collect(),
            ),
            DType::F32 => Self::F32(values.iter().map(|x| x.as_f64() as f32).collect()),
            DType::F64 => Self::F64(values.iter().map(|x| x.as_f64()).collect()),
        }
    }
}
