/// Scalar element types understood by RustGrad's IR.
///
/// `F16` and `BF16` storage uses IEEE bit patterns. This keeps the storage
/// boundary lossless even on targets without native half precision arithmetic.
use super::Float8Format;
use core::fmt;
use core::str::FromStr;
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
    F8E4M3,
    F8E5M2,
    F8E4M3FNUZ,
    F8E5M2FNUZ,
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
            Self::F8E4M3
            | Self::F8E5M2
            | Self::F8E4M3FNUZ
            | Self::F8E5M2FNUZ
            | Self::F16
            | Self::BF16
            | Self::F32
            | Self::F64 => DTypeCategory::Float,
        }
    }

    pub const fn bits(self) -> u8 {
        match self {
            Self::Bool => 1,
            Self::I8
            | Self::U8
            | Self::F8E4M3
            | Self::F8E5M2
            | Self::F8E4M3FNUZ
            | Self::F8E5M2FNUZ => 8,
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

    pub const fn float8_format(self) -> Option<Float8Format> {
        match self {
            Self::F8E4M3 => Some(Float8Format::E4M3),
            Self::F8E5M2 => Some(Float8Format::E5M2),
            Self::F8E4M3FNUZ => Some(Float8Format::E4M3FNUZ),
            Self::F8E5M2FNUZ => Some(Float8Format::E5M2FNUZ),
            _ => None,
        }
    }

    /// Whether this is one of the distinct raw float8 transport formats.
    pub const fn is_float8(self) -> bool {
        self.float8_format().is_some()
    }

    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::F8E4M3 => "float8_e4m3",
            Self::F8E5M2 => "float8_e5m2",
            Self::F8E4M3FNUZ => "float8_e4m3fnuz",
            Self::F8E5M2FNUZ => "float8_e5m2fnuz",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
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
                (F8E4M3, F8E4M3) => F8E4M3,
                (F8E5M2, F8E5M2) => F8E5M2,
                (F8E4M3FNUZ, F8E4M3FNUZ) => F8E4M3FNUZ,
                (F8E5M2FNUZ, F8E5M2FNUZ) => F8E5M2FNUZ,
                (F8E4M3 | F8E5M2 | F8E4M3FNUZ | F8E5M2FNUZ, _)
                | (_, F8E4M3 | F8E5M2 | F8E4M3FNUZ | F8E5M2FNUZ) => F16,
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

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.stable_name())
    }
}
impl FromStr for DType {
    type Err = ();
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        [
            Self::Bool,
            Self::I8,
            Self::U8,
            Self::I16,
            Self::U16,
            Self::I32,
            Self::U32,
            Self::I64,
            Self::U64,
            Self::F8E4M3,
            Self::F8E5M2,
            Self::F8E4M3FNUZ,
            Self::F8E5M2FNUZ,
            Self::F16,
            Self::BF16,
            Self::F32,
            Self::F64,
        ]
        .into_iter()
        .find(|dtype| dtype.stable_name() == value)
        .ok_or(())
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

    #[test]
    fn dtype_metadata_and_promotion() {
        assert_eq!(DType::F16.itemsize(), 2);
        assert_eq!(DType::I8.promote(DType::U8), DType::I16);
        assert_eq!(DType::I32.promote(DType::F32), DType::F32);
        assert_eq!(DType::U64.promote(DType::I64), DType::F64);
    }
}
