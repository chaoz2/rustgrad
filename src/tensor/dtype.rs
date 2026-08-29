use crate::{Error, Result};

use super::{
    scalar::{Scalar, bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16},
    storage::Storage,
};

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

    /// Converts a source scalar into this dtype's host storage scalar format.
    ///
    /// F16 carries its rounded floating value, while BF16 carries its low
    /// sixteen storage bits. All other concrete dtypes retain the input scalar
    /// exactly, matching tinygrad's storage helper boundary.
    pub fn to_storage_scalar(self, value: Scalar) -> Scalar {
        match self {
            Self::F16 => Scalar::F(f16_to_f32(f32_to_f16(value.as_f64() as f32)) as f64),
            Self::BF16 => Scalar::U(f32_to_bf16(value.as_f64() as f32) as u64),
            _ => value,
        }
    }

    /// Converts this dtype's host storage scalar format back to a source scalar.
    pub fn from_storage_scalar(self, value: Scalar) -> Scalar {
        match self {
            Self::BF16 => Scalar::F(bf16_to_f32(value.as_u64() as u16) as f64),
            _ => value,
        }
    }

    /// Reinterprets one concrete host scalar through equal-width storage.
    ///
    /// This is deliberately independent of TensorData's byte parser: Bool
    /// follows the source struct-unpack rule, where every nonzero byte is true.
    pub fn bitcast_scalar(self, value: Scalar, output: Self) -> Result<Scalar> {
        if self.itemsize() != output.itemsize() {
            return Err(Error::BitcastItemsizeMismatch {
                input: self,
                output,
            });
        }
        let packed = self.pack_storage_scalar(value);
        Ok(output.unpack_storage_scalar(packed))
    }

    fn pack_storage_scalar(self, value: Scalar) -> [u8; 8] {
        let value = self.to_storage_scalar(value);
        let mut packed = [0; 8];
        match self {
            Self::Bool => packed[0] = u8::from(value.as_bool()),
            Self::I8 => packed[0] = value.as_i64() as i8 as u8,
            Self::U8 => packed[0] = value.as_u64() as u8,
            Self::I16 => packed[..2].copy_from_slice(&(value.as_i64() as i16).to_le_bytes()),
            Self::U16 => packed[..2].copy_from_slice(&(value.as_u64() as u16).to_le_bytes()),
            Self::I32 => packed[..4].copy_from_slice(&(value.as_i64() as i32).to_le_bytes()),
            Self::U32 => packed[..4].copy_from_slice(&(value.as_u64() as u32).to_le_bytes()),
            Self::I64 => packed.copy_from_slice(&value.as_i64().to_le_bytes()),
            Self::U64 => packed.copy_from_slice(&value.as_u64().to_le_bytes()),
            Self::F16 => packed[..2]
                .copy_from_slice(&f32_to_f16(value.as_f64() as f32).to_le_bytes()),
            Self::BF16 => packed[..2].copy_from_slice(&(value.as_u64() as u16).to_le_bytes()),
            Self::F32 => packed[..4].copy_from_slice(&(value.as_f64() as f32).to_le_bytes()),
            Self::F64 => packed.copy_from_slice(&value.as_f64().to_le_bytes()),
        }
        packed
    }

    fn unpack_storage_scalar(self, packed: [u8; 8]) -> Scalar {
        let value = match self {
            Self::Bool => Scalar::Bool(packed[0] != 0),
            Self::I8 => Scalar::I((packed[0] as i8) as i64),
            Self::U8 => Scalar::U(packed[0] as u64),
            Self::I16 => Scalar::I(i16::from_le_bytes([packed[0], packed[1]]) as i64),
            Self::U16 => Scalar::U(u16::from_le_bytes([packed[0], packed[1]]) as u64),
            Self::I32 => Scalar::I(
                i32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]) as i64,
            ),
            Self::U32 => Scalar::U(
                u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]) as u64,
            ),
            Self::I64 => Scalar::I(i64::from_le_bytes(packed)),
            Self::U64 => Scalar::U(u64::from_le_bytes(packed)),
            Self::F16 => Scalar::F(
                f16_to_f32(u16::from_le_bytes([packed[0], packed[1]])) as f64,
            ),
            Self::BF16 => Scalar::U(u16::from_le_bytes([packed[0], packed[1]]) as u64),
            Self::F32 => Scalar::F(
                f32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]) as f64,
            ),
            Self::F64 => Scalar::F(f64::from_le_bytes(packed)),
        };
        self.from_storage_scalar(value)
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

    /// Parses one supported tinygrad concrete dtype name or alias.
    ///
    /// Names are ASCII-case-insensitive but deliberately not whitespace
    /// normalized. This is distinct from safetensors' uppercase wire tags.
    pub fn parse_tinygrad_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "bool" => Ok(Self::Bool),
            "int8" | "char" => Ok(Self::I8),
            "uint8" | "uchar" => Ok(Self::U8),
            "int16" | "short" => Ok(Self::I16),
            "uint16" | "ushort" => Ok(Self::U16),
            "int32" | "int" => Ok(Self::I32),
            "uint32" | "uint" => Ok(Self::U32),
            "int64" | "long" => Ok(Self::I64),
            "uint64" | "ulong" => Ok(Self::U64),
            "float16" | "half" => Ok(Self::F16),
            "bfloat16" => Ok(Self::BF16),
            "float32" | "float" => Ok(Self::F32),
            "float64" | "double" => Ok(Self::F64),
            _ => Err(Error::InvalidTinygradDTypeName {
                name: name.to_owned(),
            }),
        }
    }

    /// Returns tinygrad's inverse-dtype dictionary spelling for this dtype.
    pub const fn canonical_tinygrad_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "char",
            Self::U8 => "uchar",
            Self::I16 => "short",
            Self::U16 => "ushort",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::I64 => "long",
            Self::U64 => "ulong",
            Self::F16 => "half",
            Self::BF16 => "bfloat16",
            Self::F32 => "float",
            Self::F64 => "double",
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

    #[test]
    fn storage_scalar_helpers_match_half_and_bfloat16_source_formats() {
        let value = Scalar::F(1.0006);
        assert_scalar_eq(
            DType::F16.to_storage_scalar(value),
            Scalar::F(1.000_976_562_5),
        );
        assert_scalar_eq(
            DType::F16.from_storage_scalar(Scalar::F(1.000_976_562_5)),
            Scalar::F(1.000_976_562_5),
        );

        let bf16_storage = DType::BF16.to_storage_scalar(Scalar::F(1.003));
        assert_scalar_eq(bf16_storage, Scalar::U(0x3f80));
        assert_scalar_eq(
            DType::BF16.from_storage_scalar(bf16_storage),
            Scalar::F(1.0),
        );

        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F32,
            DType::F64,
        ] {
            let value = Scalar::I(-7);
            assert_scalar_eq(dtype.to_storage_scalar(value), value);
            assert_scalar_eq(dtype.from_storage_scalar(value), value);
        }
    }

    #[test]
    fn scalar_bitcast_accepts_every_equal_width_dtype_pair() {
        const BYTE: [DType; 3] = [DType::Bool, DType::I8, DType::U8];
        const WORD: [DType; 4] = [DType::I16, DType::U16, DType::F16, DType::BF16];
        const DWORD: [DType; 3] = [DType::I32, DType::U32, DType::F32];
        const QWORD: [DType; 3] = [DType::I64, DType::U64, DType::F64];
        for (dtypes, value) in [
            (&BYTE[..], Scalar::U(0xff)),
            (&WORD[..], Scalar::U(0x8001)),
            (&DWORD[..], Scalar::U(0x8000_0001)),
            (&QWORD[..], Scalar::U(0x8000_0000_0000_0001)),
        ] {
            for input in dtypes {
                for output in dtypes {
                    assert!(
                        input.bitcast_scalar(value, *output).is_ok(),
                        "{input:?} -> {output:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_bitcast_preserves_representative_raw_lanes_and_special_floats() {
        assert_scalar_eq(
            DType::I8.bitcast_scalar(Scalar::I(-1), DType::U8).unwrap(),
            Scalar::U(0xff),
        );
        assert_scalar_eq(
            DType::U8.bitcast_scalar(Scalar::U(0x80), DType::I8).unwrap(),
            Scalar::I(i8::MIN as i64),
        );
        assert_scalar_eq(
            DType::I16.bitcast_scalar(Scalar::I(-2), DType::U16).unwrap(),
            Scalar::U(0xfffe),
        );
        assert_scalar_eq(
            DType::U16.bitcast_scalar(Scalar::U(0x8000), DType::I16).unwrap(),
            Scalar::I(i16::MIN as i64),
        );
        assert_scalar_eq(
            DType::I32
                .bitcast_scalar(Scalar::I(i32::MIN as i64), DType::U32)
                .unwrap(),
            Scalar::U(0x8000_0000),
        );
        assert_scalar_eq(
            DType::U64
                .bitcast_scalar(Scalar::U(0x8000_0000_0000_0000), DType::I64)
                .unwrap(),
            Scalar::I(i64::MIN),
        );

        assert_scalar_eq(
            DType::F16
                .bitcast_scalar(Scalar::F(f16_to_f32(0x7e01) as f64), DType::U16)
                .unwrap(),
            Scalar::U(0x7e01),
        );
        assert_scalar_eq(
            DType::BF16
                .bitcast_scalar(Scalar::F(bf16_to_f32(0x7fc1) as f64), DType::U16)
                .unwrap(),
            Scalar::U(0x7fc1),
        );
        assert_scalar_eq(
            DType::U32
                .bitcast_scalar(Scalar::U(0x8000_0000), DType::F32)
                .unwrap(),
            Scalar::F((-0.0f32) as f64),
        );
        assert!(DType::U32
            .bitcast_scalar(Scalar::U(0x7fc0_1234), DType::F32)
            .unwrap()
            .as_f64()
            .is_nan());
        assert!(DType::U64
            .bitcast_scalar(Scalar::U(0x7ff8_0000_0000_1234), DType::F64)
            .unwrap()
            .as_f64()
            .is_nan());
        assert!(DType::U64
            .bitcast_scalar(Scalar::U(0x7ff0_0000_0000_0000), DType::F64)
            .unwrap()
            .as_f64()
            .is_infinite());
    }

    #[test]
    fn scalar_bitcast_uses_struct_bool_truthiness_and_rejects_mismatched_widths() {
        assert_scalar_eq(
            DType::U8.bitcast_scalar(Scalar::U(2), DType::Bool).unwrap(),
            Scalar::Bool(true),
        );
        assert_scalar_eq(
            DType::U8.bitcast_scalar(Scalar::U(0), DType::Bool).unwrap(),
            Scalar::Bool(false),
        );
        assert!(matches!(
            DType::F32.bitcast_scalar(Scalar::F(1.0), DType::U16),
            Err(Error::BitcastItemsizeMismatch {
                input: DType::F32,
                output: DType::U16,
            })
        ));
    }

    #[test]
    fn tinygrad_dtype_names_cover_every_supported_alias_and_canonical_spelling() {
        let aliases = [
            (DType::Bool, &["bool"][..]),
            (DType::I8, &["int8", "char"][..]),
            (DType::U8, &["uint8", "uchar"][..]),
            (DType::I16, &["int16", "short"][..]),
            (DType::U16, &["uint16", "ushort"][..]),
            (DType::I32, &["int32", "int"][..]),
            (DType::U32, &["uint32", "uint"][..]),
            (DType::I64, &["int64", "long"][..]),
            (DType::U64, &["uint64", "ulong"][..]),
            (DType::F16, &["float16", "half"][..]),
            (DType::BF16, &["bfloat16"][..]),
            (DType::F32, &["float32", "float"][..]),
            (DType::F64, &["float64", "double"][..]),
        ];
        let canonical = [
            "bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong",
            "half", "bfloat16", "float", "double",
        ];

        for ((dtype, names), canonical) in aliases.into_iter().zip(canonical) {
            for &name in names {
                assert_eq!(DType::parse_tinygrad_name(name).unwrap(), dtype, "{name}");
                assert_eq!(
                    DType::parse_tinygrad_name(&name.to_ascii_uppercase()).unwrap(),
                    dtype,
                    "{name} case-folding"
                );
            }
            assert_eq!(dtype.canonical_tinygrad_name(), canonical);
            assert_eq!(DType::parse_tinygrad_name(canonical).unwrap(), dtype);
        }
    }

    #[test]
    fn tinygrad_dtype_name_parser_rejects_whitespace_and_unsupported_surfaces() {
        for name in [
            " float",
            "float ",
            "float\t",
            "",
            "weakint",
            "weakfloat",
            "void",
            "fp8e4m3",
            "float8_e5m2",
            "ptr",
            "pointer",
            "image",
            "custom",
            "F32",
        ] {
            let result = DType::parse_tinygrad_name(name);
            assert!(matches!(
                result,
                Err(Error::InvalidTinygradDTypeName { name: rejected }) if rejected == name
            ));
        }
        let error = DType::parse_tinygrad_name(" void").unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported tinygrad dtype name \" void\""
        );
    }
}
