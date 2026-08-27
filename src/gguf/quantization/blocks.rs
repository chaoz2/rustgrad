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
const IQ3_S_BLOCK_BYTES: usize = 110;
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
// tinygrad/runtime/autogen/ggml_common.py: iq3s_grid. Each word encodes four
// source-order u8 grid values in little-byte order.
const IQ3_S_GRID: [u32; 512] = [
    0x01010101,0x01010103,0x01010105,0x0101010b,0x0101010f,0x01010301,0x01010303,0x01010305,0x01010309,0x0101030d,0x01010501,0x01010503,0x0101050b,0x01010707,0x01010901,0x01010905,0x0101090b,0x0101090f,0x01010b03,0x01010b07,0x01010d01,0x01010d05,0x01010f03,0x01010f09,0x01010f0f,0x01030101,0x01030103,0x01030105,0x01030109,0x01030301,0x01030303,0x0103030b,0x01030501,0x01030507,0x0103050f,0x01030703,0x0103070b,0x01030909,0x01030d03,0x01030d0b,0x01030f05,0x01050101,0x01050103,0x0105010b,0x0105010f,0x01050301,0x01050307,0x0105030d,0x01050503,0x0105050b,0x01050701,0x01050709,0x01050905,0x0105090b,0x0105090f,0x01050b03,0x01050b07,0x01050f01,0x01050f07,0x01070107,0x01070303,0x0107030b,0x01070501,0x01070505,0x01070703,0x01070707,0x0107070d,0x01070909,0x01070b01,0x01070b05,0x01070d0f,0x01070f03,0x01070f0b,0x01090101,0x01090307,0x0109030f,0x01090503,0x01090509,0x01090705,0x01090901,0x01090907,0x01090b03,0x01090f01,0x010b0105,0x010b0109,0x010b0501,0x010b0505,0x010b050d,0x010b0707,0x010b0903,0x010b090b,0x010b090f,0x010b0d0d,0x010b0f07,0x010d010d,0x010d0303,0x010d0307,0x010d0703,0x010d0b05,0x010d0f03,0x010f0101,0x010f0105,0x010f0109,0x010f0501,0x010f0505,0x010f050d,0x010f0707,0x010f0b01,0x010f0b09,0x03010101,0x03010103,0x03010105,0x03010109,0x03010301,0x03010303,0x03010307,0x0301030b,0x0301030f,0x03010501,0x03010505,0x03010703,0x03010709,0x0301070d,0x03010b09,0x03010b0d,0x03010d03,0x03010f05,0x03030101,
    0x03030103,0x03030107,0x0303010d,0x03030301,0x03030309,0x03030503,0x03030701,0x03030707,0x03030903,0x03030b01,0x03030b05,0x03030f01,0x03030f0d,0x03050101,0x03050305,0x0305030b,0x0305030f,0x03050501,0x03050509,0x03050705,0x03050901,0x03050907,0x03050b0b,0x03050d01,0x03050f05,0x03070103,0x03070109,0x0307010f,0x03070301,0x03070307,0x03070503,0x0307050f,0x03070701,0x03070709,0x03070903,0x03070d05,0x03070f01,0x03090107,0x0309010b,0x03090305,0x03090309,0x03090703,0x03090707,0x03090905,0x0309090d,0x03090b01,0x03090b09,0x030b0103,0x030b0301,0x030b0307,0x030b0503,0x030b0701,0x030b0705,0x030b0b03,0x030d0501,0x030d0509,0x030d050f,0x030d0909,0x030d090d,0x030f0103,0x030f0107,0x030f0301,0x030f0305,0x030f0503,0x030f070b,0x030f0903,0x030f0d05,0x030f0f01,0x05010101,0x05010103,0x05010107,0x0501010b,0x0501010f,0x05010301,0x05010305,0x05010309,0x0501030d,0x05010503,0x05010507,0x0501050f,0x05010701,0x05010705,0x05010903,0x05010907,0x0501090b,0x05010b01,0x05010b05,0x05010d0f,0x05010f01,0x05010f07,0x05010f0b,0x05030101,0x05030105,0x05030301,0x05030307,0x0503030f,0x05030505,0x0503050b,0x05030703,0x05030709,0x05030905,0x05030b03,0x05050103,0x05050109,0x0505010f,0x05050503,0x05050507,0x05050701,0x0505070f,0x05050903,0x05050b07,0x05050b0f,0x05050f03,0x05050f09,0x05070101,0x05070105,0x0507010b,0x05070303,0x05070505,0x05070509,0x05070703,0x05070707,0x05070905,0x05070b01,0x05070d0d,0x05090103,0x0509010f,0x05090501,
    0x05090507,0x05090705,0x0509070b,0x05090903,0x05090f05,0x05090f0b,0x050b0109,0x050b0303,0x050b0505,0x050b070f,0x050b0901,0x050b0b07,0x050b0f01,0x050d0101,0x050d0105,0x050d010f,0x050d0503,0x050d0b0b,0x050d0d03,0x050f010b,0x050f0303,0x050f050d,0x050f0701,0x050f0907,0x050f0b01,0x07010105,0x07010303,0x07010307,0x0701030b,0x0701030f,0x07010505,0x07010703,0x07010707,0x0701070b,0x07010905,0x07010909,0x0701090f,0x07010b03,0x07010d07,0x07010f03,0x07030103,0x07030107,0x0703010b,0x07030309,0x07030503,0x07030507,0x07030901,0x07030d01,0x07030f05,0x07030f0d,0x07050101,0x07050305,0x07050501,0x07050705,0x07050709,0x07050b01,0x07070103,0x07070301,0x07070309,0x07070503,0x07070507,0x0707050f,0x07070701,0x07070903,0x07070907,0x0707090f,0x07070b0b,0x07070f07,0x07090107,0x07090303,0x0709030d,0x07090505,0x07090703,0x07090b05,0x07090d01,0x07090d09,0x070b0103,0x070b0301,0x070b0305,0x070b050b,0x070b0705,0x070b0909,0x070b0b0d,0x070b0f07,0x070d030d,0x070d0903,0x070f0103,0x070f0107,0x070f0501,0x070f0505,0x070f070b,0x09010101,0x09010109,0x09010305,0x09010501,0x09010509,0x0901050f,0x09010705,0x09010903,0x09010b01,0x09010f01,0x09030105,0x0903010f,0x09030303,0x09030307,0x09030505,0x09030701,0x0903070b,0x09030907,0x09030b03,0x09030b0b,0x09050103,0x09050107,0x09050301,0x0905030b,0x09050503,0x09050707,0x09050901,0x09050b0f,0x09050d05,0x09050f01,0x09070109,0x09070303,0x09070307,0x09070501,0x09070505,0x09070703,0x0907070b,
    0x09090101,0x09090105,0x09090509,0x0909070f,0x09090901,0x09090f03,0x090b010b,0x090b010f,0x090b0503,0x090b0d05,0x090d0307,0x090d0709,0x090d0d01,0x090f0301,0x090f030b,0x090f0701,0x090f0907,0x090f0b03,0x0b010105,0x0b010301,0x0b010309,0x0b010505,0x0b010901,0x0b010909,0x0b01090f,0x0b010b05,0x0b010d0d,0x0b010f09,0x0b030103,0x0b030107,0x0b03010b,0x0b030305,0x0b030503,0x0b030705,0x0b030f05,0x0b050101,0x0b050303,0x0b050507,0x0b050701,0x0b05070d,0x0b050b07,0x0b070105,0x0b07010f,0x0b070301,0x0b07050f,0x0b070909,0x0b070b03,0x0b070d0b,0x0b070f07,0x0b090103,0x0b090109,0x0b090501,0x0b090705,0x0b09090d,0x0b0b0305,0x0b0b050d,0x0b0b0b03,0x0b0b0b07,0x0b0d0905,0x0b0f0105,0x0b0f0109,0x0b0f0505,0x0d010303,0x0d010307,0x0d01030b,0x0d010703,0x0d010707,0x0d010d01,0x0d030101,0x0d030501,0x0d03050f,0x0d030d09,0x0d050305,0x0d050709,0x0d050905,0x0d050b0b,0x0d050d05,0x0d050f01,0x0d070101,0x0d070309,0x0d070503,0x0d070901,0x0d09050b,0x0d090907,0x0d090d05,0x0d0b0101,0x0d0b0107,0x0d0b0709,0x0d0b0d01,0x0d0d010b,0x0d0d0901,0x0d0f0303,0x0d0f0307,0x0f010101,0x0f010109,0x0f01010f,0x0f010501,0x0f010505,0x0f01070d,0x0f010901,0x0f010b09,0x0f010d05,0x0f030105,0x0f030303,0x0f030509,0x0f030907,0x0f03090b,0x0f050103,0x0f050109,0x0f050301,0x0f05030d,0x0f050503,0x0f050701,0x0f050b03,0x0f070105,0x0f070705,0x0f07070b,0x0f070b07,0x0f090103,0x0f09010b,0x0f090307,0x0f090501,0x0f090b01,0x0f0b0505,0x0f0b0905,0x0f0d0105,0x0f0d0703,0x0f0f0101,
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

/// Decodes one GGML IQ3_S block: 64 low selector bytes, an eight-byte high
/// selector plane, a separate sign plane, and eight packed group scales.
pub(crate) fn decode_iq3_s_block(
    block: &[u8],
) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
    require_len(block, IQ3_S_BLOCK_BYTES)?;
    let d = half(&block[..2]);
    if !d.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }

    let mut out = [0.0; K_BLOCK_ELEMENTS];
    for group in 0..8 {
        let packed_scale = block[106 + group / 2];
        let scale = 1 + 2 * ((packed_scale >> ((group % 2) * 4)) & 0x0f);
        for segment in 0..4 {
            let signs = block[74 + group * 4 + segment];
            for lane in 0..8 {
                let selector_position = group * 8 + segment * 2 + lane / 4;
                let high = (block[66 + selector_position / 8] >> (selector_position % 8)) & 1;
                let selector = usize::from(block[2 + selector_position]) | (usize::from(high) << 8);
                let grid = ((IQ3_S_GRID[selector] >> (8 * (lane % 4))) & 0xff) as u8;
                let sign = if (signs >> lane) & 1 == 0 { 1.0 } else { -1.0 };
                out[group * 32 + segment * 8 + lane] =
                    d * f32::from(scale) * f32::from(grid) * sign;
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
