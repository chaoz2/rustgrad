/// A scalar value used at dense tensor conversion boundaries.
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

pub(super) fn scalar_to_i8(value: Scalar) -> i8 {
    match value {
        Scalar::F(x) => x as i8,
        _ => value.as_i64() as i8,
    }
}

pub(super) fn scalar_to_u8(value: Scalar) -> u8 {
    match value {
        Scalar::F(x) => x as u8,
        _ => value.as_u64() as u8,
    }
}

pub(super) fn scalar_to_i16(value: Scalar) -> i16 {
    match value {
        Scalar::F(x) => x as i16,
        _ => value.as_i64() as i16,
    }
}

pub(super) fn scalar_to_u16(value: Scalar) -> u16 {
    match value {
        Scalar::F(x) => x as u16,
        _ => value.as_u64() as u16,
    }
}

pub(super) fn scalar_to_i32(value: Scalar) -> i32 {
    match value {
        Scalar::F(x) => x as i32,
        _ => value.as_i64() as i32,
    }
}

pub(super) fn scalar_to_u32(value: Scalar) -> u32 {
    match value {
        Scalar::F(x) => x as u32,
        _ => value.as_u64() as u32,
    }
}

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let upper = (bits >> 16) as u16;
    if bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0 {
        // Preserve the sign and every payload bit representable in BF16.  A
        // NaN whose payload is entirely below the BF16 cutoff needs the low
        // BF16 payload bit set so truncation cannot turn it into infinity.
        return if upper & 0x007f == 0 {
            upper | 1
        } else {
            upper
        };
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
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

pub(super) fn f32_to_f16(value: f32) -> u16 {
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
    use super::f32_to_bf16;

    #[test]
    fn bf16_rounding_preserves_nan_and_non_nan_raw_contracts() {
        let cases = [
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            (0x0000_0001, 0x0000),
            (0x007f_ffff, 0x0080),
            (0x3f80_8000, 0x3f80),
            (0x3f81_8000, 0x3f82),
            (0xbf80_8000, 0xbf80),
            (0x7f80_0000, 0x7f80),
            (0xff80_0000, 0xff80),
            (0x7f80_0001, 0x7f81),
            (0x7f80_7fff, 0x7f81),
            (0x7f81_0000, 0x7f81),
            (0x7fc0_0000, 0x7fc0),
            (0x7fff_ffff, 0x7fff),
            (0xff80_0001, 0xff81),
            (0xffff_ffff, 0xffff),
        ];
        for (input, expected) in cases {
            assert_eq!(
                f32_to_bf16(f32::from_bits(input)),
                expected,
                "{input:#010x}"
            );
        }
    }
}
