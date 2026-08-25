//! Exact host codecs for tinygrad's four float8 storage formats.
//!
//! This is a raw-storage prerequisite, not a `TensorData` dtype or a graph
//! execution promise. In particular, no backend silently treats these bytes as
//! `U8`; callers must select a [`Float8Format`] explicitly.

/// The float8 families defined by checked-in tinygrad `dtype.py`.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
pub enum Float8Format {
    E4M3,
    E5M2,
    E4M3FNUZ,
    E5M2FNUZ,
}

#[derive(Clone, Copy)]
struct Config {
    bias: i32,
    sig_bits: u32,
    mant_mask: u64,
    min_denorm_half: u64,
    overflow_threshold: u64,
    max_normal: u8,
    min_normal: u64,
}

impl Float8Format {
    /// The distinct public `DType` corresponding to this raw format.
    pub const fn dtype(self) -> super::DType {
        match self {
            Self::E4M3 => super::DType::F8E4M3,
            Self::E5M2 => super::DType::F8E5M2,
            Self::E4M3FNUZ => super::DType::F8E4M3FNUZ,
            Self::E5M2FNUZ => super::DType::F8E5M2FNUZ,
        }
    }
    const fn config(self) -> Config {
        match self {
            Self::E4M3 => Config {
                bias: 7,
                sig_bits: 4,
                mant_mask: 0x7,
                min_denorm_half: 0x3F50_0000_0000_0000,
                overflow_threshold: 0x407D_0000_0000_0000,
                max_normal: 0x7E,
                min_normal: 0x3F90_0000_0000_0000,
            },
            Self::E5M2 => Config {
                bias: 15,
                sig_bits: 3,
                mant_mask: 0x3,
                min_denorm_half: 0x3EE0_0000_0000_0000,
                overflow_threshold: 0x40ED_FFFF_FFFF_FFFF,
                max_normal: 0x7B,
                min_normal: 0x3F10_0000_0000_0000,
            },
            Self::E4M3FNUZ => Config {
                bias: 8,
                sig_bits: 4,
                mant_mask: 0x7,
                min_denorm_half: 0x3F40_0000_0000_0000,
                overflow_threshold: 0x406E_FFFF_FFFF_FFFF,
                max_normal: 0x7F,
                min_normal: 0x3F80_0000_0000_0000,
            },
            Self::E5M2FNUZ => Config {
                bias: 16,
                sig_bits: 3,
                mant_mask: 0x3,
                min_denorm_half: 0x3ED0_0000_0000_0000,
                overflow_threshold: 0x40ED_FFFF_FFFF_FFFF,
                max_normal: 0x7F,
                min_normal: 0x3F00_0000_0000_0000,
            },
        }
    }

    const fn is_fnuz(self) -> bool {
        matches!(self, Self::E4M3FNUZ | Self::E5M2FNUZ)
    }

    /// Converts an f64 using tinygrad's IEEE round-to-nearest-even FP8 codec.
    /// FNUZ formats canonicalize both zero signs to `0x00` and non-finites to
    /// `0x80`; E4M3 uses its terminal NaN encodings and E5M2 retains infinity.
    pub fn encode(self, value: f64) -> u8 {
        if self.is_fnuz() && !value.is_finite() {
            return 0x80;
        }
        if self.is_fnuz() && value == 0.0 {
            return 0;
        }
        let sign = if value.is_sign_negative() { 0x80 } else { 0 };
        if self == Self::E4M3 && !value.is_finite() {
            return if sign == 0 { 0x7f } else { 0xff };
        }
        if self == Self::E5M2 && !value.is_finite() {
            return sign | if value.is_infinite() { 0x7c } else { 0x7f };
        }
        let config = self.config();
        let bits = value.to_bits();
        let abs = bits & 0x7fff_ffff_ffff_ffff;
        let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + config.bias;
        let mut mantissa = (bits >> (53 - config.sig_bits)) & config.mant_mask;
        let half_ulp = 1u64 << (52 - config.sig_bits);
        let result = if abs <= config.min_denorm_half {
            0
        } else if abs > config.overflow_threshold {
            config.max_normal as u64
        } else if abs >= config.min_normal {
            let mut result = ((exponent as u64) << (config.sig_bits - 1)) | mantissa;
            let round_bits = bits & ((half_ulp << 1) - 1);
            if round_bits > half_ulp || (round_bits == half_ulp && mantissa & 1 != 0) {
                result += 1;
            }
            result
        } else {
            let shift = (1 - exponent) as u32;
            mantissa |= 1 << (config.sig_bits - 1);
            let mut result = mantissa >> shift;
            let half = half_ulp << shift;
            let round_bits = (bits | (1 << 52)) & ((half << 1) - 1);
            if round_bits > half || (round_bits == half && result & 1 != 0) {
                result += 1;
            }
            result
        };
        if self.is_fnuz() && result == 0 {
            0
        } else {
            result as u8 | sign
        }
    }

    /// Decodes one raw float8 payload exactly according to tinygrad's format
    /// policy. Raw payloads are never canonicalized by this function.
    pub fn decode(self, raw: u8) -> f64 {
        if self.is_fnuz() && raw == 0x80 {
            return f64::NAN;
        }
        if raw & 0x7f == 0 {
            return if raw & 0x80 == 0 { 0.0 } else { -0.0 };
        }
        let config = self.config();
        let mantissa_bits = config.sig_bits - 1;
        let exponent_bits = 8 - config.sig_bits;
        let exponent_max = (1 << exponent_bits) - 1;
        let mantissa_max = (1 << mantissa_bits) - 1;
        let sign = raw >> 7;
        let exponent = (raw >> mantissa_bits) & exponent_max;
        let mantissa = raw & mantissa_max;
        if !self.is_fnuz() && exponent == exponent_max {
            if self == Self::E5M2 {
                let value = if mantissa == 0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                };
                return if sign == 0 { value } else { -value };
            }
            if mantissa == mantissa_max {
                return f64::NAN;
            }
        }
        let value = if exponent == 0 {
            (mantissa as f64 / (mantissa_max + 1) as f64) * 2f64.powi(1 - config.bias)
        } else {
            (1.0 + mantissa as f64 / (mantissa_max + 1) as f64)
                * 2f64.powi(exponent as i32 - config.bias)
        };
        if sign == 0 { value } else { -value }
    }
}

/// Explicitly tagged, raw-bit-preserving float8 host storage. `from_raw`
/// retains every byte (including noncanonical NaN payloads); `from_f64` uses
/// the format's numeric conversion contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Float8Storage {
    format: Float8Format,
    bytes: Vec<u8>,
}

impl Float8Storage {
    pub fn from_raw(format: Float8Format, bytes: Vec<u8>) -> Self {
        Self { format, bytes }
    }
    pub fn from_f64(format: Float8Format, values: impl IntoIterator<Item = f64>) -> Self {
        Self {
            format,
            bytes: values
                .into_iter()
                .map(|value| format.encode(value))
                .collect(),
        }
    }
    pub fn format(&self) -> Float8Format {
        self.format
    }
    pub fn as_raw(&self) -> &[u8] {
        &self.bytes
    }
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
    pub fn values(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.bytes.iter().map(|raw| self.format.decode(*raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, TensorData};

    #[test]
    fn source_checked_vectors_cover_all_float8_families() {
        let cases = [
            (
                Float8Format::E4M3,
                &[0x00, 0x80, 0x01, 0x38, 0x7e, 0x7f][..],
                &[0.0, -0.0, 0.001953125, 1.0, 448.0, f64::NAN][..],
            ),
            (
                Float8Format::E5M2,
                &[0x00, 0x80, 0x01, 0x3c, 0x7b, 0x7c][..],
                &[0.0, -0.0, 0.0000152587890625, 1.0, 57344.0, f64::INFINITY][..],
            ),
            (
                Float8Format::E4M3FNUZ,
                &[0x00, 0x01, 0x40, 0x7f, 0x80][..],
                &[0.0, 0.0009765625, 1.0, 240.0, f64::NAN][..],
            ),
            (
                Float8Format::E5M2FNUZ,
                &[0x00, 0x04, 0x40, 0x7f, 0x80][..],
                &[0.0, 0.000030517578125, 1.0, 57344.0, f64::NAN][..],
            ),
        ];
        for (format, raw, expected) in cases {
            for (&raw, &expected) in raw.iter().zip(expected) {
                let decoded = format.decode(raw);
                if expected.is_nan() {
                    assert!(decoded.is_nan(), "{format:?} {raw:#04x}");
                } else {
                    assert_eq!(
                        decoded.to_bits(),
                        expected.to_bits(),
                        "{format:?} {raw:#04x}"
                    );
                }
            }
        }
    }

    #[test]
    fn encoding_rounds_ties_saturates_and_applies_special_policies() {
        let cases = [
            (Float8Format::E4M3, 1.0625, 0x38),
            (Float8Format::E4M3, 1.1875, 0x3a),
            (Float8Format::E4M3, f64::INFINITY, 0x7f),
            (Float8Format::E5M2, 1.125, 0x3c),
            (Float8Format::E5M2, 1.375, 0x3e),
            (Float8Format::E5M2, f64::INFINITY, 0x7c),
            (Float8Format::E4M3FNUZ, -0.0, 0x00),
            (Float8Format::E4M3FNUZ, f64::INFINITY, 0x80),
            (Float8Format::E5M2FNUZ, -0.0, 0x00),
            (Float8Format::E5M2FNUZ, f64::NEG_INFINITY, 0x80),
        ];
        for (format, value, expected) in cases {
            assert_eq!(format.encode(value), expected, "{format:?} {value}");
        }
        assert_eq!(Float8Format::E4M3.encode(1e9), 0x7e);
        assert_eq!(Float8Format::E5M2.encode(1e9), 0x7b);
    }

    #[test]
    fn raw_storage_preserves_payloads_and_numeric_construction_is_distinct() {
        let raw = Float8Storage::from_raw(Float8Format::E4M3, vec![0x7f, 0xff, 0x80]);
        assert_eq!(raw.as_raw(), [0x7f, 0xff, 0x80]);
        assert!(raw.values().next().unwrap().is_nan());
        let numeric = Float8Storage::from_f64(Float8Format::E4M3FNUZ, [-0.0, f64::NAN]);
        assert_eq!(numeric.as_raw(), [0x00, 0x80]);
        assert!(Float8Storage::from_raw(Float8Format::E4M3, vec![]).is_empty());
    }

    #[test]
    fn every_raw_payload_round_trips_through_typed_tensor_bytes() {
        let payload = (0_u8..=u8::MAX).collect::<Vec<_>>();
        for format in [
            Float8Format::E4M3,
            Float8Format::E5M2,
            Float8Format::E4M3FNUZ,
            Float8Format::E5M2FNUZ,
        ] {
            let tensor = TensorData::from_storage(
                Shape::from([payload.len()]),
                crate::Storage::Float8(Float8Storage::from_raw(format, payload.clone())),
            )
            .unwrap();
            let bytes = tensor.to_le_bytes().unwrap();
            assert_eq!(bytes, payload, "{format:?}");
            let decoded =
                TensorData::from_le_bytes(tensor.shape().clone(), format.dtype(), &bytes).unwrap();
            assert_eq!(decoded, tensor, "{format:?}");
            assert!(TensorData::from_le_bytes(Shape::from([255]), format.dtype(), &bytes).is_err());
            let empty = TensorData::from_storage(
                Shape::from([0]),
                crate::Storage::Float8(Float8Storage::from_raw(format, vec![])),
            )
            .unwrap();
            assert_eq!(empty.to_le_bytes().unwrap(), Vec::<u8>::new());
        }
    }
}
