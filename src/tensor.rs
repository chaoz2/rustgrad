use crate::{Error, Result};
use std::fmt;

mod creation;

/// Scalar element types understood by RustGrad's IR.
///
/// `F16` and `BF16` storage uses IEEE bit patterns. This keeps the storage
/// boundary lossless even on targets without native half precision arithmetic.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub enum DType {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DTypeCategory {
    Bool,
    Signed,
    Unsigned,
    Float,
}

impl DType {
    pub const fn category(self) -> DTypeCategory {
        match self {
            Self::Bool => DTypeCategory::Bool,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => DTypeCategory::Signed,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => DTypeCategory::Unsigned,
            Self::F16 | Self::BF16 | Self::F32 | Self::F64 => DTypeCategory::Float,
        }
    }
    pub const fn bits(self) -> u8 {
        match self {
            Self::Bool => 1,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
        }
    }
    pub const fn itemsize(self) -> usize {
        (self.bits() as usize).div_ceil(8)
    }
    pub const fn is_float(self) -> bool {
        matches!(self.category(), DTypeCategory::Float)
    }
    pub const fn is_integer(self) -> bool {
        matches!(
            self.category(),
            DTypeCategory::Signed | DTypeCategory::Unsigned
        )
    }
    /// A compact, deterministic promotion lattice for supported scalar dtypes.
    /// It follows tinygrad's widening intent; fp8/weak/pointer dtypes are not
    /// implemented yet.
    pub fn promote(self, other: Self) -> Self {
        use DType::*;
        if self == other {
            return self;
        }
        if self.is_float() || other.is_float() {
            return match (self, other) {
                (F64, _) | (_, F64) => F64,
                (F32, _) | (_, F32) => F32,
                (F16, BF16) | (BF16, F16) => F32,
                (F16, _) | (_, F16) => F16,
                _ => BF16,
            };
        }
        if self == Bool {
            return other;
        }
        if other == Bool {
            return self;
        }
        let signed = matches!(self.category(), DTypeCategory::Signed);
        let other_signed = matches!(other.category(), DTypeCategory::Signed);
        if signed == other_signed {
            return integer_dtype(signed, self.bits().max(other.bits()));
        }
        let (signed_bits, unsigned_bits) = if signed {
            (self.bits(), other.bits())
        } else {
            (other.bits(), self.bits())
        };
        if signed_bits > unsigned_bits {
            integer_dtype(true, signed_bits)
        } else if unsigned_bits < 64 {
            integer_dtype(true, (unsigned_bits * 2).min(64))
        } else {
            F64
        }
    }
}
const fn integer_dtype(signed: bool, bits: u8) -> DType {
    match (signed, bits) {
        (true, 0..=8) => DType::I8,
        (false, 0..=8) => DType::U8,
        (true, 9..=16) => DType::I16,
        (false, 9..=16) => DType::U16,
        (true, 17..=32) => DType::I32,
        (false, 17..=32) => DType::U32,
        (true, _) => DType::I64,
        (false, _) => DType::U64,
    }
}

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

#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    Bool(bool),
    I(i64),
    U(u64),
    F(f64),
}
impl Scalar {
    pub fn as_bool(self) -> bool {
        match self {
            Self::Bool(x) => x,
            Self::I(x) => x != 0,
            Self::U(x) => x != 0,
            Self::F(x) => x != 0.0,
        }
    }
    pub fn as_i64(self) -> i64 {
        match self {
            Self::Bool(x) => x as i64,
            Self::I(x) => x,
            Self::U(x) => x as i64,
            Self::F(x) => x as i64,
        }
    }
    pub fn as_u64(self) -> u64 {
        match self {
            Self::Bool(x) => x as u64,
            Self::I(x) => x as u64,
            Self::U(x) => x,
            Self::F(x) => x as u64,
        }
    }
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Bool(x) => x as u8 as f64,
            Self::I(x) => x as f64,
            Self::U(x) => x as f64,
            Self::F(x) => x,
        }
    }
}
fn scalar_to_i8(value: Scalar) -> i8 {
    match value {
        Scalar::F(x) => x as i8,
        _ => value.as_i64() as i8,
    }
}
fn scalar_to_u8(value: Scalar) -> u8 {
    match value {
        Scalar::F(x) => x as u8,
        _ => value.as_u64() as u8,
    }
}
fn scalar_to_i16(value: Scalar) -> i16 {
    match value {
        Scalar::F(x) => x as i16,
        _ => value.as_i64() as i16,
    }
}
fn scalar_to_u16(value: Scalar) -> u16 {
    match value {
        Scalar::F(x) => x as u16,
        _ => value.as_u64() as u16,
    }
}
fn scalar_to_i32(value: Scalar) -> i32 {
    match value {
        Scalar::F(x) => x as i32,
        _ => value.as_i64() as i32,
    }
}
fn scalar_to_u32(value: Scalar) -> u32 {
    match value {
        Scalar::F(x) => x as u32,
        _ => value.as_u64() as u32,
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Shape(Vec<usize>);
impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    pub fn numel(&self) -> Result<usize> {
        self.0.iter().try_fold(1usize, |n, dim| {
            n.checked_mul(*dim)
                .ok_or_else(|| Error::ShapeOverflow(self.clone()))
        })
    }
    pub fn without_axis(&self, axis: usize) -> Option<Self> {
        if axis >= self.rank() {
            None
        } else {
            let mut dims = self.0.clone();
            dims.remove(axis);
            Some(Self(dims))
        }
    }
    pub fn broadcast_with(&self, other: &Self) -> Result<Self> {
        let rank = self.rank().max(other.rank());
        let mut output = Vec::with_capacity(rank);
        for offset in (0..rank).rev() {
            let lhs = self
                .0
                .get(self.rank().wrapping_sub(1 + offset))
                .copied()
                .unwrap_or(1);
            let rhs = other
                .0
                .get(other.rank().wrapping_sub(1 + offset))
                .copied()
                .unwrap_or(1);
            if lhs != rhs && lhs != 1 && rhs != 1 {
                return Err(Error::BroadcastMismatch {
                    lhs: self.clone(),
                    rhs: other.clone(),
                });
            }
            output.push(if lhs == 0 || rhs == 0 {
                0
            } else {
                lhs.max(rhs)
            });
        }
        Ok(Self(output))
    }
    pub(crate) fn contiguous_strides(&self) -> Vec<usize> {
        let mut stride = 1;
        let mut strides = vec![0; self.rank()];
        for (index, dim) in self.0.iter().enumerate().rev() {
            strides[index] = stride;
            stride *= dim;
        }
        strides
    }
}
impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self(value.to_vec())
    }
}
impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self(value)
    }
}
impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    shape: Shape,
    storage: Storage,
}
impl TensorData {
    pub fn new(shape: impl Into<Shape>, values: Vec<f32>) -> Result<Self> {
        Self::from_storage(shape, Storage::F32(values))
    }
    pub fn from_storage(shape: impl Into<Shape>, storage: Storage) -> Result<Self> {
        let shape = shape.into();
        let expected = shape.numel()?;
        if storage.len() != expected {
            return Err(Error::InvalidData {
                shape,
                expected,
                actual: storage.len(),
            });
        }
        Ok(Self { shape, storage })
    }
    pub fn from_scalars(
        shape: impl Into<Shape>,
        dtype: DType,
        values: impl IntoIterator<Item = Scalar>,
    ) -> Result<Self> {
        Self::from_storage(shape, Storage::from_scalars(dtype, values))
    }
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: Shape::new(Vec::new()),
            storage: Storage::F32(vec![value]),
        }
    }
    pub fn scalar_with_dtype(value: Scalar, dtype: DType) -> Self {
        Self {
            shape: Shape::new(Vec::new()),
            storage: Storage::from_scalars(dtype, [value]),
        }
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }
    pub fn storage(&self) -> &Storage {
        &self.storage
    }
    pub fn scalar_at(&self, index: usize) -> Scalar {
        self.storage.scalar(index)
    }
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
    pub fn values(&self) -> &[f32] {
        match &self.storage {
            Storage::F32(values) => values,
            _ => panic!("values() is only available for f32 TensorData; use scalar_at or storage"),
        }
    }
    pub fn cast(&self, dtype: DType) -> Self {
        Self {
            shape: self.shape.clone(),
            storage: Storage::from_scalars(dtype, (0..self.len()).map(|i| self.scalar_at(i))),
        }
    }
    pub fn to_vec_f64(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| self.scalar_at(i).as_f64())
            .collect()
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
fn f32_to_bf16(value: f32) -> u16 {
    ((value
        .to_bits()
        .wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1)))
        >> 16) as u16
}
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Normalize the half subnormal before placing it in the f32
            // significand. The old leading-zero shortcut underflowed every
            // nonzero half subnormal by eleven binary orders.
            let mut mant = mant;
            let mut exponent = -14i32;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exponent -= 1;
            }
            sign | (((exponent + 127) as u32) << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f800000 | (mant << 13)
    } else {
        sign | (((exp as u32) + 112) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let raw_exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7fffff;
    if raw_exponent == 0xff {
        return sign
            | 0x7c00
            | if mantissa == 0 {
                0
            } else {
                ((mantissa >> 13) as u16) | 1
            };
    }
    let exponent = raw_exponent as i32 - 127 + 15;
    if exponent <= 0 {
        if exponent < -10 {
            sign
        } else {
            let shift = (14 - exponent) as u32;
            sign | round_shift_right_ties_even(mantissa | 0x800000, shift) as u16
        }
    } else if exponent >= 31 {
        sign | 0x7c00
    } else {
        let rounded = round_shift_right_ties_even(mantissa, 13);
        if rounded == 0x400 {
            if exponent == 30 {
                sign | 0x7c00
            } else {
                sign | (((exponent + 1) as u16) << 10)
            }
        } else {
            sign | ((exponent as u16) << 10) | rounded as u16
        }
    }
}

fn round_shift_right_ties_even(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dtype_metadata_and_promotion() {
        assert_eq!(DType::F16.itemsize(), 2);
        assert_eq!(DType::I8.promote(DType::U8), DType::I16);
        assert_eq!(DType::I32.promote(DType::F32), DType::F32);
        assert_eq!(DType::U64.promote(DType::I64), DType::F64);
    }
    #[test]
    fn storage_preserves_integer_and_bool_values() {
        let x = TensorData::from_scalars(
            [3],
            DType::U64,
            [Scalar::U(u64::MAX), Scalar::U(1), Scalar::U(0)],
        )
        .unwrap();
        assert_eq!(x.storage(), &Storage::U64(vec![u64::MAX, 1, 0]));
        assert_eq!(
            x.cast(DType::Bool).storage(),
            &Storage::Bool(vec![true, true, false])
        );
    }
    #[test]
    fn casts_are_deterministic_and_half_storage_is_lossless() {
        let x = TensorData::from_scalars(
            [3],
            DType::F64,
            [Scalar::F(-1.9), Scalar::F(300.0), Scalar::F(f64::NAN)],
        )
        .unwrap();
        assert_eq!(x.cast(DType::U8).storage(), &Storage::U8(vec![0, 255, 0]));
        let half = TensorData::from_scalars([1], DType::F16, [Scalar::F(1.5)]).unwrap();
        assert_eq!(half.storage(), &Storage::F16(vec![0x3e00]));
        assert_eq!(half.to_vec_f64(), vec![1.5]);
    }
}
