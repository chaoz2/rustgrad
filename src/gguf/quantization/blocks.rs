//! Pure checked GGML K-block bit-layout decoding.

use crate::tensor::f16_to_f32;

const K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;
const MXFP4_BLOCK_BYTES: usize = 17;
const Q1_0_BLOCK_BYTES: usize = 18;
const MXFP4_LUT: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0,
    -12.0,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockDecodeError {
    Length { expected: usize, actual: usize },
    NonFinite,
}

/// Decodes one GGML Q4_K block: two half scales, eight packed six-bit
/// scale/min pairs, then eight groups of 32 four-bit values.
pub(crate) fn decode_q4_k_block(block: &[u8]) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
    require_len(block, Q4_K_BLOCK_BYTES)?;
    let d = half(&block[..2]);
    let dmin = half(&block[2..4]);
    if !d.is_finite() || !dmin.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }

    let (scales, mins) = unpack_k_scales_mins(&block[4..16]);

    let quants = &block[16..];
    let mut out = [0.; K_BLOCK_ELEMENTS];
    for group in 0..8 {
        let source_group = group / 2;
        let shift = (group % 2) * 4;
        for lane in 0..32 {
            let q = (quants[source_group * 32 + lane] >> shift) & 0x0f;
            out[group * 32 + lane] =
                d * f32::from(scales[group]) * f32::from(q) - dmin * f32::from(mins[group]);
        }
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q5_K block. Q5_K shares Q4_K's two half block scales and
/// eight packed scale/min pairs, adding a source-lane high-bit plane.
pub(crate) fn decode_q5_k_block(block: &[u8]) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
    require_len(block, Q5_K_BLOCK_BYTES)?;
    let d = half(&block[..2]);
    let dmin = half(&block[2..4]);
    if !d.is_finite() || !dmin.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let (scales, mins) = unpack_k_scales_mins(&block[4..16]);
    let high_bits = &block[16..48];
    let quants = &block[48..];
    let mut out = [0.; K_BLOCK_ELEMENTS];
    for group in 0..8 {
        let source_group = group / 2;
        let shift = (group % 2) * 4;
        for lane in 0..32 {
            let nibble = (quants[source_group * 32 + lane] >> shift) & 0x0f;
            let high = (high_bits[lane] >> group) & 1;
            let quant = nibble + 16 * high;
            out[group * 32 + lane] =
                d * f32::from(scales[group]) * f32::from(quant) - dmin * f32::from(mins[group]);
        }
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q6_K block: low four-bit planes, high two-bit planes,
/// sixteen signed scales, and a trailing half block scale.
pub(crate) fn decode_q6_k_block(block: &[u8]) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
    require_len(block, Q6_K_BLOCK_BYTES)?;
    let d = half(&block[208..]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }

    let mut out = [0.; K_BLOCK_ELEMENTS];
    for (index, value) in out.iter_mut().enumerate() {
        let half_index = index / 128;
        let within_half = index % 128;
        let low_shift = (within_half / 64) * 4;
        let low = (block[half_index * 64 + within_half % 64] >> low_shift) & 0x0f;
        let high_shift = (within_half / 32) * 2;
        let high = ((block[128 + half_index * 32 + within_half % 32] >> high_shift) & 0x03) << 4;
        let quant = i32::from(low | high) - 32;
        let scale = i32::from(block[192 + index / 16] as i8);
        *value = d * (quant * scale) as f32;
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q4_0 block. GGML stores the low nibbles for values
/// 0..16 followed by the high nibbles for values 16..32.
pub(crate) fn decode_q4_0_block(block: &[u8]) -> Result<[f32; 32], BlockDecodeError> {
    require_len(block, 18)?;
    let d = half(&block[..2]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let mut out = [0.0; 32];
    for (lane, &packed) in block[2..].iter().enumerate() {
        out[lane] = (f32::from(packed & 0x0f) - 8.0) * d;
        out[16 + lane] = (f32::from(packed >> 4) - 8.0) * d;
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q4_1 block: little-endian half scale and minimum,
/// followed by low nibbles for values 0..16 and high nibbles for 16..32.
pub(crate) fn decode_q4_1_block(block: &[u8]) -> Result<[f32; 32], BlockDecodeError> {
    require_len(block, 20)?;
    let d = half(&block[..2]);
    let m = half(&block[2..4]);
    if !d.is_finite() || !m.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let mut out = [0.0; 32];
    for (lane, &packed) in block[4..].iter().enumerate() {
        out[lane] = f32::from(packed & 0x0f) * d + m;
        out[16 + lane] = f32::from(packed >> 4) * d + m;
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q5_0 block: a little-endian half scale, four bytes of
/// source-position high bits, and low/high nibble halves for 32 values.
pub(crate) fn decode_q5_0_block(block: &[u8]) -> Result<[f32; 32], BlockDecodeError> {
    require_len(block, 22)?;
    let d = half(&block[..2]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let mut out = [0.0; 32];
    for (lane, &packed) in block[6..].iter().enumerate() {
        for (output_lane, nibble) in [(lane, packed & 0x0f), (16 + lane, packed >> 4)] {
            let high = (block[2 + output_lane / 8] >> (output_lane % 8)) & 1;
            out[output_lane] = d * (f32::from(nibble + 16 * high) - 16.0);
        }
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q5_1 block: little-endian half scale and minimum, four
/// source-position high-bit bytes, and low/high nibble halves for 32 values.
pub(crate) fn decode_q5_1_block(block: &[u8]) -> Result<[f32; 32], BlockDecodeError> {
    require_len(block, 24)?;
    let d = half(&block[..2]);
    let m = half(&block[2..4]);
    if !d.is_finite() || !m.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let mut out = [0.0; 32];
    for (lane, &packed) in block[8..].iter().enumerate() {
        for (output_lane, nibble) in [(lane, packed & 0x0f), (16 + lane, packed >> 4)] {
            let high = (block[4 + output_lane / 8] >> (output_lane % 8)) & 1;
            out[output_lane] = f32::from(nibble + 16 * high) * d + m;
        }
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML MXFP4 block: a shared raw F32 exponent followed by
/// sixteen packed FP4 lookup codes in low-then-high lane order.
pub(crate) fn decode_mxfp4_block(block: &[u8]) -> Result<[f32; 32], BlockDecodeError> {
    require_len(block, MXFP4_BLOCK_BYTES)?;
    let exponent = block[0];
    let scale = f32::from_bits(match exponent {
        0 => 0x0020_0000,
        1 => 0x0040_0000,
        _ => u32::from(exponent - 1) << 23,
    });
    let mut out = [0.0; 32];
    for (lane, &packed) in block[1..].iter().enumerate() {
        out[lane] = MXFP4_LUT[usize::from(packed & 0x0f)] * scale;
        out[16 + lane] = MXFP4_LUT[usize::from(packed >> 4)] * scale;
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q1_0 block: a little-endian half scale followed by a
/// transposed little-bit-order 16-byte plane.
pub(crate) fn decode_q1_0_block(block: &[u8]) -> Result<[f32; 128], BlockDecodeError> {
    require_len(block, Q1_0_BLOCK_BYTES)?;
    let d = half(&block[..2]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let mut out = [0.0; 128];
    for bit in 0..8 {
        for byte in 0..16 {
            let sign = if (block[2 + byte] >> bit) & 1 == 0 {
                -1.0
            } else {
                1.0
            };
            out[bit * 16 + byte] = d * sign;
        }
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML Q8_0 block: one little-endian half scale followed by
/// exactly 32 signed eight-bit quants.
pub(crate) fn decode_q8_0_block(block: &[u8]) -> Result<[f32; 32], BlockDecodeError> {
    require_len(block, 34)?;
    let d = half(&block[..2]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let mut out = [0.0; 32];
    for (value, &quant) in out.iter_mut().zip(&block[2..]) {
        *value = f32::from(quant as i8) * d;
    }
    finite(&out)?;
    Ok(out)
}

fn half(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn unpack_k_scales_mins(packed: &[u8]) -> ([u8; 8], [u8; 8]) {
    debug_assert_eq!(packed.len(), 12);
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for lane in 0..4 {
        scales[lane] = packed[lane] & 0x3f;
        mins[lane] = packed[4 + lane] & 0x3f;
        scales[4 + lane] = (packed[8 + lane] & 0x0f) | ((packed[lane] >> 6) << 4);
        mins[4 + lane] = (packed[8 + lane] >> 4) | ((packed[4 + lane] >> 6) << 4);
    }
    (scales, mins)
}

fn require_len(bytes: &[u8], expected: usize) -> Result<(), BlockDecodeError> {
    if bytes.len() == expected {
        Ok(())
    } else {
        Err(BlockDecodeError::Length {
            expected,
            actual: bytes.len(),
        })
    }
}

fn finite(values: &[f32]) -> Result<(), BlockDecodeError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(BlockDecodeError::NonFinite)
    }
}
