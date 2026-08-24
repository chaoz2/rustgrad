//! Deterministic Threefry 2x32 random primitives shared by graph creation and
//! CPU realization. This module is pure: stream allocation belongs at the IR
//! boundary, keeping replay independent of backend scheduling.

pub mod plan;

const PARITY: u32 = 0x1BD1_1BDA;
const ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];

/// Evaluates the Random123/Threefry 2x32 permutation used by tinygrad.
pub(crate) fn threefry2x32(key: [u32; 2], counter: [u32; 2]) -> [u32; 2] {
    let keys = [key[0], key[1], key[0] ^ key[1] ^ PARITY];
    let mut x0 = counter[0].wrapping_add(keys[0]);
    let mut x1 = counter[1].wrapping_add(keys[1]);
    for round in 0..20 {
        x0 = x0.wrapping_add(x1);
        x1 = x1.rotate_left(ROTATIONS[round % ROTATIONS.len()]) ^ x0;
        if round % 4 == 3 {
            let injection = round / 4 + 1;
            x0 = x0.wrapping_add(keys[injection % 3]);
            x1 = x1
                .wrapping_add(keys[(injection + 1) % 3])
                .wrapping_add(injection as u32);
        }
    }
    [x0, x1]
}

/// Advances a little-endian two-word counter by `words`, returning the start.
pub(crate) fn reserve(counter: &mut [u32; 2], words: u64) -> [u32; 2] {
    let start = *counter;
    let low = counter[0].wrapping_add(words as u32);
    counter[1] = counter[1]
        .wrapping_add((words >> 32) as u32)
        .wrapping_add((low < counter[0]) as u32);
    counter[0] = low;
    start
}

/// Maximum word count in one tinygrad Threefry dispatch. Larger requests are
/// split before constructing their count tensors.
pub(crate) const MAX_CHUNK_WORDS: u64 = u32::MAX as u64;

/// Returns the size of a chunk in a bounded Threefry request without
/// allocating its words. This is used by the stream planner and boundary
/// tests; CPU realization only asks for materializable dense tensors.
pub(crate) fn chunk_words(total: u64, chunk: u64) -> Option<u64> {
    let start = chunk.checked_mul(MAX_CHUNK_WORDS)?;
    (start < total).then(|| (total - start).min(MAX_CHUNK_WORDS))
}

/// Produces the exact packed word layout used by tinygrad's `random_bits`:
/// one derived key per chunk, followed by all low lanes then high lanes. The
/// output is subsequently reinterpreted at the requested storage width.
pub(crate) fn words(key: [u32; 2], counter: [u32; 2], count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let total = count as u64;
    let mut chunk = 0;
    while let Some(size) = chunk_words(total, chunk) {
        let offset = chunk * MAX_CHUNK_WORDS;
        let mut chunk_counter = counter;
        reserve(&mut chunk_counter, offset);
        let derived_key = threefry2x32(key, chunk_counter);
        let pairs = size.div_ceil(2) as u32;
        for lane in 0..pairs {
            out.push(threefry2x32(derived_key, [lane, lane.wrapping_add(pairs)])[0]);
        }
        for lane in 0..pairs {
            if out.len() == count {
                break;
            }
            out.push(threefry2x32(derived_key, [lane, lane.wrapping_add(pairs)])[1]);
        }
        chunk += 1;
    }
    out
}

/// Exact tinygrad-style float construction from a Threefry word.
pub(crate) fn uniform_word(word: u32) -> f32 {
    f32::from_bits((word >> 9) | 0x3F80_0000) - 1.0
}

pub(crate) fn uniform_f16_bits(word: u16) -> u16 {
    (word >> 6) | 0x3C00
}
pub(crate) fn uniform_bf16_bits(word: u16) -> u16 {
    (word >> 9) | 0x3F80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_threefry_vector_and_counter_carry_match() {
        let expected = [
            2221762175, 1752107825, 653745012, 1967534793, 1395205442, 3840423848, 2159346757,
            603508235, 3319473678, 3363866483, 3544324138, 1436466838, 2169858556, 2570072943,
            2387150698, 3678370550, 2911697663, 403244401, 2560861638, 1692360114,
        ];
        let mut lows = Vec::new();
        let mut highs = Vec::new();
        for index in 0..10 {
            let pair = threefry2x32([0, 1337], [index, index + 10]);
            lows.push(pair[0]);
            highs.push(pair[1]);
        }
        let actual = [lows, highs].concat();
        // The checked-in tinygrad reference uses the same key/counter layout.
        assert_eq!(actual, expected);
        assert_eq!(uniform_f16_bits(0x6667), 0x3d99);
        let mut counter = [u32::MAX - 5, 0];
        assert_eq!(reserve(&mut counter, 10), [u32::MAX - 5, 0]);
        assert_eq!(counter, [4, 1]);
    }

    #[test]
    fn chunk_planning_handles_the_u32_boundary_without_allocating() {
        assert_eq!(
            chunk_words(MAX_CHUNK_WORDS - 1, 0),
            Some(MAX_CHUNK_WORDS - 1)
        );
        assert_eq!(chunk_words(MAX_CHUNK_WORDS, 0), Some(MAX_CHUNK_WORDS));
        assert_eq!(chunk_words(MAX_CHUNK_WORDS + 1, 0), Some(MAX_CHUNK_WORDS));
        assert_eq!(chunk_words(MAX_CHUNK_WORDS + 1, 1), Some(1));
        assert_eq!(chunk_words(MAX_CHUNK_WORDS + 1, 2), None);
    }

    #[test]
    fn narrow_float_bit_construction_matches_checked_in_source_words() {
        let raw: Vec<_> = (0..4)
            .map(|index| threefry2x32([0, 1337], [index, index + 10])[0])
            .collect();
        let halves = raw
            .into_iter()
            .flat_map(|word| [word as u16, (word >> 16) as u16]);
        let f16: Vec<_> = halves.clone().map(uniform_f16_bits).collect();
        let bf16: Vec<_> = halves.map(uniform_bf16_bits).collect();
        assert_eq!(
            f16,
            [
                0x3d99, 0x3e11, 0x3c2c, 0x3da1, 0x3d6d, 0x3c9b, 0x3ccb, 0x3dd5
            ]
        );
        assert_eq!(
            bf16,
            [
                0x3fb3, 0x3fc2, 0x3f85, 0x3fb4, 0x3fad, 0x3f93, 0x3f99, 0x3fba
            ]
        );
    }
}
