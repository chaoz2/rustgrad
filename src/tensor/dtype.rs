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
        // tinygrad's weak-float node reaches every fp8 family, while each
        // family only reaches F16/BF16. Therefore an exact/bool operand plus
        // one concrete fp8 family retains that family; only a second floating
        // family forces widening.
        if self.is_float8() && !other.is_float() {
            return self;
        }
        if other.is_float8() && !self.is_float() {
            return other;
        }
        if self.is_float() || other.is_float() {
            return match (self, other) {
                (F64, _) | (_, F64) => F64,
                (F32, _) | (_, F32) => F32,
                (F16, BF16) | (BF16, F16) => F32,
                (F8E4M3 | F8E5M2 | F8E4M3FNUZ | F8E5M2FNUZ, BF16)
                | (BF16, F8E4M3 | F8E5M2 | F8E4M3FNUZ | F8E5M2FNUZ) => BF16,
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
        let alias = match value {
            "int8" | "char" => Some(Self::I8),
            "uint8" | "uchar" => Some(Self::U8),
            "int16" | "short" => Some(Self::I16),
            "uint16" | "ushort" => Some(Self::U16),
            "int32" | "int" => Some(Self::I32),
            "uint32" | "uint" => Some(Self::U32),
            "int64" | "long" => Some(Self::I64),
            "uint64" | "ulong" => Some(Self::U64),
            "float16" | "half" => Some(Self::F16),
            "bfloat16" => Some(Self::BF16),
            "float32" | "float" => Some(Self::F32),
            "float64" | "double" => Some(Self::F64),
            _ => None,
        };
        if let Some(dtype) = alias {
            return Ok(dtype);
        }
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
        let fp8s = [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ];
        for fp8 in fp8s {
            assert_eq!(fp8.promote(DType::Bool), fp8);
            assert_eq!(DType::I32.promote(fp8), fp8);
            assert_eq!(fp8.promote(DType::F16), DType::F16);
            assert_eq!(fp8.promote(DType::BF16), DType::BF16);
            assert_eq!(fp8.promote(DType::F32), DType::F32);
            assert_eq!(fp8.promote(DType::F64), DType::F64);
        }
        assert_eq!(DType::F8E4M3.promote(DType::F8E5M2), DType::F16);
    }

    #[test]
    fn dtype_parser_accepts_tinygrad_attribute_names_and_aliases() {
        let cases = [
            ("int8", DType::I8), ("char", DType::I8), ("uint8", DType::U8),
            ("int16", DType::I16), ("uint16", DType::U16), ("int32", DType::I32),
            ("uint32", DType::U32), ("int64", DType::I64), ("uint64", DType::U64),
            ("float16", DType::F16), ("half", DType::F16), ("bfloat16", DType::BF16),
            ("float32", DType::F32), ("float", DType::F32), ("float64", DType::F64),
            ("double", DType::F64),
        ];
        for (name, dtype) in cases { assert_eq!(name.parse(), Ok(dtype), "{name}"); }
        assert_eq!("Float32".parse::<DType>(), Err(()));
    }
}
