//! Exact software narrow-float storage helpers for WGSL.
use crate::DType;

/// Software narrow-float conversion and packed-storage ABI version.
pub const WEBGPU_NARROW_ABI_VERSION: u32 = 1;

pub(super) const SOURCE: &str = r#"fn rg_f16_to_f32(h: u32) -> f32 {
  let sign: u32 = (h & 0x8000u) << 16u;
  let raw_exponent: u32 = (h >> 10u) & 31u;
  var mantissa: u32 = h & 1023u;
  var out: u32;
  if (raw_exponent == 0u) {
    if (mantissa == 0u) {
      out = sign;
    } else {
      var exponent: i32 = -14i;
      while ((mantissa & 1024u) == 0u) {
        mantissa = mantissa << 1u;
        exponent = exponent - 1i;
      }
      out = sign | (u32(exponent + 127i) << 23u) | ((mantissa & 1023u) << 13u);
    }
  } else if (raw_exponent == 31u) {
    out = sign | 0x7f800000u | (mantissa << 13u);
  } else {
    out = sign | ((raw_exponent + 112u) << 23u) | (mantissa << 13u);
  }
  return bitcast<f32>(out);
}

fn rg_f32_to_f16(value: f32) -> u32 {
  let bits: u32 = bitcast<u32>(value);
  let sign: u32 = (bits >> 16u) & 0x8000u;
  let raw_exponent: u32 = (bits >> 23u) & 255u;
  let mantissa: u32 = bits & 0x7fffffu;
  if (raw_exponent == 255u) {
    return sign | 0x7c00u | select((mantissa >> 13u) | 1u, 0u, mantissa == 0u);
  }
  var exponent: i32 = i32(raw_exponent) - 112i;
  if (exponent <= 0i) {
    if (exponent < -10i) { return sign; }
    let shift: u32 = u32(14i - exponent);
    let significant: u32 = mantissa | 0x800000u;
    let truncated: u32 = significant >> shift;
    let remainder: u32 = significant & ((1u << shift) - 1u);
    let halfway: u32 = 1u << (shift - 1u);
    let increment: u32 = select(0u, 1u, remainder > halfway || (remainder == halfway && (truncated & 1u) != 0u));
    return sign | (truncated + increment);
  }
  if (exponent >= 31i) { return sign | 0x7c00u; }
  var rounded: u32 = mantissa >> 13u;
  let remainder: u32 = mantissa & 0x1fffu;
  rounded = rounded + select(0u, 1u, remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0u));
  if (rounded == 0x400u) {
    if (exponent == 30i) { return sign | 0x7c00u; }
    exponent = exponent + 1i;
    rounded = 0u;
  }
  return sign | (u32(exponent) << 10u) | rounded;
}

fn rg_bf16_to_f32(bits: u32) -> f32 {
  return bitcast<f32>((bits & 0xffffu) << 16u);
}

fn rg_f32_to_bf16(value: f32) -> u32 {
  let bits: u32 = bitcast<u32>(value);
  let upper: u32 = bits >> 16u;
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
    if ((upper & 0x7fu) == 0u) { return upper | 1u; }
    return upper;
  }
  return (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
}"#;

pub(super) fn is_narrow(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

pub(super) fn decode(dtype: DType, raw: impl AsRef<str>) -> Option<String> {
    let raw = raw.as_ref();
    match dtype {
        DType::F16 => Some(format!("rg_f16_to_f32({raw})")),
        DType::BF16 => Some(format!("rg_bf16_to_f32({raw})")),
        _ => None,
    }
}

pub(super) fn encode(dtype: DType, value: impl AsRef<str>) -> Option<String> {
    let value = value.as_ref();
    match dtype {
        DType::F16 => Some(format!("rg_f32_to_f16({value})")),
        DType::BF16 => Some(format!("rg_f32_to_bf16({value})")),
        _ => None,
    }
}

pub(super) fn quantize(dtype: DType, value: impl AsRef<str>) -> Option<String> {
    encode(dtype, value).and_then(|raw| decode(dtype, raw))
}
