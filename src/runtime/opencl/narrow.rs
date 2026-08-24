//! Exact raw-storage conversion helpers for OpenCL narrow floats.
use crate::DType;

pub(super) const F16_SOURCE: &str = r#"static float rg_f16_to_f32(ushort h) {
  uint sign = ((uint)(h & (ushort)0x8000u)) << 16;
  uint exponent = ((uint)h >> 10) & 31u;
  uint mantissa = (uint)h & 1023u;
  uint out;
  if (exponent == 0u) {
    if (mantissa == 0u) out = sign;
    else {
      int unbiased = -14;
      while ((mantissa & 1024u) == 0u) { mantissa <<= 1; --unbiased; }
      out = sign | ((uint)(unbiased + 127) << 23) | ((mantissa & 1023u) << 13);
    }
  } else if (exponent == 31u) {
    out = sign | 0x7f800000u | (mantissa << 13);
  } else {
    out = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  return as_float(out);
}
static ushort rg_f32_to_f16(float x) {
  uint bits = as_uint(x);
  ushort sign = (ushort)((bits >> 16) & 0x8000u);
  uint raw_exponent = (bits >> 23) & 255u;
  uint mantissa = bits & 0x7fffffu;
  if (raw_exponent == 255u)
    return (ushort)(sign | (ushort)0x7c00u | (ushort)(mantissa == 0u ? 0u : ((mantissa >> 13) | 1u)));
  int exponent = (int)raw_exponent - 112;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    uint shift = (uint)(14 - exponent);
    uint truncated = (mantissa | 0x800000u) >> shift;
    uint remainder = (mantissa | 0x800000u) & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    return (ushort)(sign | (ushort)(truncated + (uint)(remainder > halfway || (remainder == halfway && (truncated & 1u)))));
  }
  if (exponent >= 31) return (ushort)(sign | (ushort)0x7c00u);
  uint rounded = mantissa >> 13;
  uint remainder = mantissa & 0x1fffu;
  rounded += (uint)(remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u)));
  if (rounded == 0x400u) {
    if (exponent == 30) return (ushort)(sign | (ushort)0x7c00u);
    ++exponent;
    rounded = 0u;
  }
  return (ushort)(sign | (ushort)((uint)exponent << 10) | (ushort)rounded);
}"#;

pub(super) const BF16_SOURCE: &str = r#"static float rg_bf16_to_f32(ushort bits) {
  return as_float(((uint)bits) << 16);
}
static ushort rg_f32_to_bf16(float x) {
  uint bits = as_uint(x);
  return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}"#;

pub(super) fn is_narrow(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

pub(super) fn decode(dtype: DType, raw: impl AsRef<str>) -> Option<String> {
    let raw = raw.as_ref();
    match dtype {
        DType::F16 => Some(format!("((double)rg_f16_to_f32({raw}))")),
        DType::BF16 => Some(format!("((double)rg_bf16_to_f32({raw}))")),
        _ => None,
    }
}

pub(super) fn encode(dtype: DType, value: impl AsRef<str>) -> Option<String> {
    let value = value.as_ref();
    match dtype {
        DType::F16 => Some(format!("rg_f32_to_f16((float)({value}))")),
        DType::BF16 => Some(format!("rg_f32_to_bf16((float)({value}))")),
        _ => None,
    }
}

/// Quantizes a narrow cast at the cast site, then returns its exact decoded
/// value for any fused consumer that follows it.
pub(super) fn quantize(dtype: DType, value: impl AsRef<str>) -> Option<String> {
    encode(dtype, value).and_then(|raw| decode(dtype, raw))
}
