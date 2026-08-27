//! Pure checked GGML K-block bit-layout decoding.

use crate::tensor::f16_to_f32;

const K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;
const MXFP4_BLOCK_BYTES: usize = 17;
const Q1_0_BLOCK_BYTES: usize = 18;
const IQ4_XS_BLOCK_BYTES: usize = 136;
const IQ3_XXS_BLOCK_BYTES: usize = 98;
const MXFP4_LUT: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0,
    -12.0,
];
const IQ4_XS_LUT: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0,
    53.0, 69.0, 89.0, 113.0,
];
// tinygrad/runtime/autogen/ggml_common.py: iq3xxs_grid. Each word encodes
// four source-order u8 grid values in little-byte order.
const IQ3_XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414, 0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404, 0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34, 0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c, 0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c,
    0x0c141c04, 0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c,
    0x0c3e1414, 0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434, 0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e, 0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c,
    0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24, 0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c, 0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414,
    0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434,
    0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c, 0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434,
    0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14, 0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c, 0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424,
    0x340c140c, 0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14, 0x3e042c14,
    0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
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

/// Decodes one GGML IQ4_XS block: a shared half scale, eight signed packed
/// group scales, and eight low-then-high-nibble groups using IQ4's nonlinear
/// lookup table.
pub(crate) fn decode_iq4_xs_block(
    block: &[u8],
) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
    require_len(block, IQ4_XS_BLOCK_BYTES)?;
    let d = half(&block[..2]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales = unpack_iq4_xs_scales(scales_h, &block[4..8]);

    let mut out = [0.0; K_BLOCK_ELEMENTS];
    for group in 0..8 {
        let packed = &block[8 + group * 16..8 + (group + 1) * 16];
        let scale = f32::from(scales[group]);
        for (lane, &value) in packed.iter().enumerate() {
            out[group * 32 + lane] = d * scale * IQ4_XS_LUT[usize::from(value & 0x0f)];
            out[group * 32 + 16 + lane] = d * scale * IQ4_XS_LUT[usize::from(value >> 4)];
        }
    }
    finite(&out)?;
    Ok(out)
}

/// Decodes one GGML IQ3_XXS block: a shared half scale, 64 four-value grid
/// selectors, and eight packed scale/sign words.
pub(crate) fn decode_iq3_xxs_block(
    block: &[u8],
) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
    require_len(block, IQ3_XXS_BLOCK_BYTES)?;
    let d = half(&block[..2]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }

    let mut out = [0.0; K_BLOCK_ELEMENTS];
    for group in 0..8 {
        let word_offset = 66 + group * 4;
        let word = u32::from_le_bytes([
            block[word_offset],
            block[word_offset + 1],
            block[word_offset + 2],
            block[word_offset + 3],
        ]);
        let scale = d * (f32::from((word >> 28) as u8) + 0.5) * 0.5;
        for selector_lane in 0..4 {
            let selector = ((word >> (7 * selector_lane)) & 0x7f) as u8;
            let signs = iq3_xxs_even_signs(selector);
            for byte_lane in 0..8 {
                let selector_index = group * 8 + selector_lane * 2 + byte_lane / 4;
                let grid_lane = byte_lane % 4;
                let grid = ((IQ3_XXS_GRID[usize::from(block[2 + selector_index])]
                    >> (8 * grid_lane))
                    & 0xff) as u8;
                let sign = if (signs >> byte_lane) & 1 == 0 { 1.0 } else { -1.0 };
                out[group * 32 + selector_lane * 8 + byte_lane] = scale * f32::from(grid) * sign;
            }
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

fn unpack_iq4_xs_scales(scales_h: u16, scales_l: &[u8]) -> [i8; 8] {
    debug_assert_eq!(scales_l.len(), 4);
    let mut scales = [0i8; 8];
    for group in 0..8 {
        let nibble = (scales_l[group / 2] >> ((group % 2) * 4)) & 0x0f;
        let high = ((scales_h >> (2 * group)) & 0x03) as u8;
        scales[group] = (nibble | (high << 4)) as i8 - 32;
    }
    scales
}

fn iq3_xxs_even_signs(selector: u8) -> u8 {
    debug_assert!(selector < 128);
    selector | if selector.count_ones() % 2 == 0 { 0 } else { 0x80 }
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
