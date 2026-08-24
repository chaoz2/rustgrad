//! Pure checked GGML K-block bit-layout decoding.

use crate::tensor::f16_to_f32;

const K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::gguf) enum BlockDecodeError {
    Length { expected: usize, actual: usize },
    NonFinite,
}

/// Decodes one GGML Q4_K block: two half scales, eight packed six-bit
/// scale/min pairs, then eight groups of 32 four-bit values.
pub(in crate::gguf) fn decode_q4_k_block(
    block: &[u8],
) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
    require_len(block, Q4_K_BLOCK_BYTES)?;
    let d = half(&block[..2]);
    let dmin = half(&block[2..4]);
    if !d.is_finite() || !dmin.is_finite() {
        return Err(BlockDecodeError::NonFinite);
    }

    let packed = &block[4..16];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for lane in 0..4 {
        scales[lane] = packed[lane] & 0x3f;
        mins[lane] = packed[4 + lane] & 0x3f;
        scales[4 + lane] = (packed[8 + lane] & 0x0f) | ((packed[lane] >> 6) << 4);
        mins[4 + lane] = (packed[8 + lane] >> 4) | ((packed[4 + lane] >> 6) << 4);
    }

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

/// Decodes one GGML Q6_K block: low four-bit planes, high two-bit planes,
/// sixteen signed scales, and a trailing half block scale.
pub(in crate::gguf) fn decode_q6_k_block(
    block: &[u8],
) -> Result<[f32; K_BLOCK_ELEMENTS], BlockDecodeError> {
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

fn half(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
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
