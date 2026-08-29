use crate::{Error, Result};

use super::{scalar::Scalar, storage::Storage};

/// Scalar element types understood by RustGrad's IR.
///
/// `F16` and `BF16` storage uses IEEE bit patterns. This keeps the storage
/// boundary lossless even on targets without native half precision arithmetic.
#[derive(
    Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, serde::Deserialize, serde::Serialize,
)]
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

    pub const fn is_signed_integer(self) -> bool {
        matches!(self.category(), DTypeCategory::Signed)
    }

    pub const fn is_unsigned(self) -> bool {
        matches!(self.category(), DTypeCategory::Unsigned)
    }

    pub const fn is_bool(self) -> bool {
        matches!(self.category(), DTypeCategory::Bool)
    }

    /// Returns the source-visible lower bound in this dtype's scalar family.
    pub const fn min(self) -> Scalar {
        match self {
            Self::Bool => Scalar::Bool(false),
            Self::I8 => Scalar::I(i8::MIN as i64),
            Self::U8 => Scalar::U(u8::MIN as u64),
            Self::I16 => Scalar::I(i16::MIN as i64),
            Self::U16 => Scalar::U(u16::MIN as u64),
            Self::I32 => Scalar::I(i32::MIN as i64),
            Self::U32 => Scalar::U(u32::MIN as u64),
            Self::I64 => Scalar::I(i64::MIN),
            Self::U64 => Scalar::U(u64::MIN),
            Self::F16 | Self::BF16 | Self::F32 | Self::F64 => Scalar::F(f64::NEG_INFINITY),
        }
    }

    /// Returns the source-visible upper bound in this dtype's scalar family.
    pub const fn max(self) -> Scalar {
        match self {
            Self::Bool => Scalar::Bool(true),
            Self::I8 => Scalar::I(i8::MAX as i64),
            Self::U8 => Scalar::U(u8::MAX as u64),
            Self::I16 => Scalar::I(i16::MAX as i64),
            Self::U16 => Scalar::U(u16::MAX as u64),
            Self::I32 => Scalar::I(i32::MAX as i64),
            Self::U32 => Scalar::U(u32::MAX as u64),
            Self::I64 => Scalar::I(i64::MAX),
            Self::U64 => Scalar::U(u64::MAX),
            Self::F16 | Self::BF16 | Self::F32 | Self::F64 => Scalar::F(f64::INFINITY),
        }
    }

    /// Returns the IEEE exponent and mantissa widths for a supported float dtype.
    pub const fn finfo(self) -> Result<(u8, u8)> {
        match self {
            Self::F16 => Ok((5, 10)),
            Self::BF16 => Ok((8, 7)),
            Self::F32 => Ok((8, 23)),
            Self::F64 => Ok((11, 52)),
            _ => Err(Error::InvalidDTypeFinfo { dtype: self }),
        }
    }

    /// Commits a concrete scalar through this dtype's established storage path.
    ///
    /// Integer lanes follow their storage-width casts, Bool uses scalar
    /// truthiness, and float lanes round to their storage width. NaNs are
    /// canonicalized before that storage conversion; negative zero is retained.
    pub fn commit_scalar(self, value: Scalar) -> Scalar {
        let value = match value {
            Scalar::F(value) if value.is_nan() => Scalar::F(f64::NAN),
            value => value,
        };
        Storage::from_scalars(self, [value]).scalar(0)
    }

    /// Returns the concrete float work dtype used by tinygrad's source helpers.
    pub const fn least_upper_float(self) -> Self {
        if self.is_float() { self } else { Self::F32 }
    }

    /// Returns whether every value in `self` is representable by `target`.
    pub const fn can_losslessly_cast_to(self, target: Self) -> bool {
        use DType::*;
        if self == target || self == Bool {
            return true;
        }
        match target {
            F64 => matches!(self, F32 | F16 | BF16 | U32 | U16 | U8 | I32 | I16 | I8),
            F32 => matches!(self, F16 | BF16 | U16 | U8 | I16 | I8),
            F16 => matches!(self, U8 | I8),
            U64 => matches!(self, U32 | U16 | U8),
            U32 => matches!(self, U16 | U8),
            U16 => matches!(self, U8),
            I64 => matches!(self, U32 | U16 | U8 | I32 | I16 | I8),
            I32 => matches!(self, U16 | U8 | I16 | I8),
            I16 => matches!(self, U8 | I8),
            _ => false,
        }
    }

    /// Returns tinygrad's concrete default accumulation storage for Sum.
    pub const fn sum_accumulator_dtype(self) -> Self {
        match self {
            Self::Bool | Self::I8 | Self::I16 | Self::I32 => Self::I32,
            Self::U8 | Self::U16 | Self::U32 => Self::U32,
            Self::I64 => Self::I64,
            Self::U64 => Self::U64,
            Self::F16 | Self::BF16 | Self::F32 => Self::F32,
            Self::F64 => Self::F64,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReductionDType;

    fn assert_scalar_eq(actual: Scalar, expected: Scalar) {
        match (actual, expected) {
            (Scalar::Bool(actual), Scalar::Bool(expected)) => assert_eq!(actual, expected),
            (Scalar::I(actual), Scalar::I(expected)) => assert_eq!(actual, expected),
            (Scalar::U(actual), Scalar::U(expected)) => assert_eq!(actual, expected),
            (Scalar::F(actual), Scalar::F(expected)) => {
                assert_eq!(actual.to_bits(), expected.to_bits())
            }
            (actual, expected) => panic!("scalar kind mismatch: {actual:?} != {expected:?}"),
        }
    }

    #[test]
    fn dtype_metadata_and_promotion() {
        assert_eq!(DType::F16.itemsize(), 2);
        assert_eq!(DType::I8.promote(DType::U8), DType::I16);
        assert_eq!(DType::I32.promote(DType::F32), DType::F32);
        assert_eq!(DType::U64.promote(DType::I64), DType::F64);
    }

    #[test]
    fn dtype_category_bounds_and_finfo_cover_every_local_dtype() {
        let signed = [DType::I8, DType::I16, DType::I32, DType::I64];
        let unsigned = [DType::U8, DType::U16, DType::U32, DType::U64];
        let floats = [DType::F16, DType::BF16, DType::F32, DType::F64];
        for dtype in signed {
            assert!(dtype.is_signed_integer());
            assert!(!dtype.is_unsigned());
            assert!(!dtype.is_bool());
            assert!(matches!(dtype.min(), Scalar::I(_)));
            assert!(matches!(dtype.max(), Scalar::I(_)));
        }
        for dtype in unsigned {
            assert!(!dtype.is_signed_integer());
            assert!(dtype.is_unsigned());
            assert!(!dtype.is_bool());
            assert!(matches!(dtype.min(), Scalar::U(0)));
            assert!(matches!(dtype.max(), Scalar::U(_)));
        }
        assert_scalar_eq(DType::Bool.min(), Scalar::Bool(false));
        assert_scalar_eq(DType::Bool.max(), Scalar::Bool(true));
        assert!(DType::Bool.is_bool());
        assert_scalar_eq(DType::I8.min(), Scalar::I(i8::MIN as i64));
        assert_scalar_eq(DType::I64.max(), Scalar::I(i64::MAX));
        assert_scalar_eq(DType::U8.max(), Scalar::U(u8::MAX as u64));
        assert_scalar_eq(DType::U64.max(), Scalar::U(u64::MAX));
        for dtype in floats {
            assert!(dtype.min().as_f64().is_infinite());
            assert!(dtype.min().as_f64().is_sign_negative());
            assert!(dtype.max().as_f64().is_infinite());
            assert!(dtype.max().as_f64().is_sign_positive());
        }
        assert_eq!(DType::F16.finfo(), Ok((5, 10)));
        assert_eq!(DType::BF16.finfo(), Ok((8, 7)));
        assert_eq!(DType::F32.finfo(), Ok((8, 23)));
        assert_eq!(DType::F64.finfo(), Ok((11, 52)));
        assert!(matches!(
            DType::I32.finfo(),
            Err(Error::InvalidDTypeFinfo { dtype: DType::I32 })
        ));
    }

    #[test]
    fn scalar_commitment_uses_existing_storage_width_conversions() {
        assert_scalar_eq(DType::Bool.commit_scalar(Scalar::I(-1)), Scalar::Bool(true));
        assert_scalar_eq(DType::Bool.commit_scalar(Scalar::F(-0.0)), Scalar::Bool(false));
        assert_scalar_eq(DType::I8.commit_scalar(Scalar::I(257)), Scalar::I(1));
        assert_scalar_eq(
            DType::U8.commit_scalar(Scalar::I(-1)),
            Scalar::U(u8::MAX as u64),
        );
        assert_scalar_eq(
            DType::I16.commit_scalar(Scalar::U(u16::MAX as u64)),
            Scalar::I(-1),
        );
        assert_scalar_eq(
            DType::U16.commit_scalar(Scalar::I(-1)),
            Scalar::U(u16::MAX as u64),
        );
        assert_scalar_eq(
            DType::I32.commit_scalar(Scalar::U(u32::MAX as u64)),
            Scalar::I(-1),
        );
        assert_scalar_eq(
            DType::U32.commit_scalar(Scalar::I(-1)),
            Scalar::U(u32::MAX as u64),
        );
        assert_scalar_eq(DType::I64.commit_scalar(Scalar::U(u64::MAX)), Scalar::I(-1));
        assert_scalar_eq(DType::U64.commit_scalar(Scalar::I(-1)), Scalar::U(u64::MAX));

        let f16 = DType::F16.commit_scalar(Scalar::F(1.0006)).as_f64();
        assert_eq!(f16, 1.000_976_562_5);
        assert_scalar_eq(DType::BF16.commit_scalar(Scalar::F(1.003)), Scalar::F(1.0));
        assert_scalar_eq(
            DType::F32.commit_scalar(Scalar::F(1.1)),
            Scalar::F(1.1_f32 as f64)
        );
        assert_scalar_eq(DType::F64.commit_scalar(Scalar::F(1.1)), Scalar::F(1.1));
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let zero = dtype.commit_scalar(Scalar::F(-0.0)).as_f64();
            assert_eq!(zero.to_bits(), (-0.0f64).to_bits());
            assert!(dtype.commit_scalar(Scalar::F(f64::from_bits(0x7ff8_0000_0000_1234)))
                .as_f64()
                .is_nan());
        }
    }

    #[test]
    fn concrete_float_lub_and_lossless_cast_matrix_match_tinygrad() {
        const DTYPES: [DType; 13] = [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ];
        const LOSSLESS: [[bool; 13]; 13] = [
            [true, true, true, true, true, true, true, true, true, true, true, true, true],
            [false, true, false, true, false, true, false, true, false, true, false, true, true],
            [false, false, true, true, true, true, true, true, true, true, false, true, true],
            [false, false, false, true, false, true, false, true, false, false, false, true, true],
            [false, false, false, false, true, true, true, true, true, false, false, true, true],
            [false, false, false, false, false, true, false, true, false, false, false, false, true],
            [false, false, false, false, false, false, true, true, true, false, false, false, true],
            [false, false, false, false, false, false, false, true, false, false, false, false, false],
            [false, false, false, false, false, false, false, false, true, false, false, false, false],
            [false, false, false, false, false, false, false, false, false, true, false, true, true],
            [false, false, false, false, false, false, false, false, false, false, true, true, true],
            [false, false, false, false, false, false, false, false, false, false, false, true, true],
            [false, false, false, false, false, false, false, false, false, false, false, false, true],
        ];
        for (source_index, source) in DTYPES.into_iter().enumerate() {
            assert_eq!(
                source.least_upper_float(),
                if source.is_float() { source } else { DType::F32 }
            );
            for (target_index, target) in DTYPES.into_iter().enumerate() {
                assert_eq!(
                    source.can_losslessly_cast_to(target),
                    LOSSLESS[source_index][target_index],
                    "{source:?} -> {target:?}"
                );
            }
        }
        assert_eq!(DType::I64.least_upper_float(), DType::F32);
        assert_eq!(DType::U64.least_upper_float(), DType::F32);
    }

    #[test]
    fn sum_accumulator_dtype_drives_the_ir_default_pair() {
        let cases = [
            (DType::Bool, DType::I32, DType::I32),
            (DType::I8, DType::I32, DType::I32),
            (DType::U8, DType::U32, DType::U32),
            (DType::I16, DType::I32, DType::I32),
            (DType::U16, DType::U32, DType::U32),
            (DType::I32, DType::I32, DType::I32),
            (DType::U32, DType::U32, DType::U32),
            (DType::I64, DType::I64, DType::I64),
            (DType::U64, DType::U64, DType::U64),
            (DType::F16, DType::F32, DType::F16),
            (DType::BF16, DType::F32, DType::BF16),
            (DType::F32, DType::F32, DType::F32),
            (DType::F64, DType::F64, DType::F64),
        ];
        for (input, accumulator, output) in cases {
            assert_eq!(input.sum_accumulator_dtype(), accumulator);
            assert_eq!(
                ReductionDType::sum_default(input),
                ReductionDType::new(accumulator, output)
            );
        }
    }
}
