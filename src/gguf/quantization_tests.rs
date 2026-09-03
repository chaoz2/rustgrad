use super::quantization::blocks::{
    BlockDecodeError, decode_iq2_s_block, decode_iq3_s_block, decode_iq3_xxs_block,
    decode_iq4_xs_block, decode_mxfp4_block, decode_q1_0_block, decode_q4_1_block,
    decode_q4_k_block, decode_q5_0_block, decode_q5_1_block, decode_q5_k_block, decode_q6_k_block,
};
use crate::{GgmlType, QuantizedError, QuantizedTensorData, Shape};

fn half_bits(value: f32) -> [u8; 2] {
    match value {
        0.5 => 0x3800u16.to_le_bytes(),
        1.0 => 0x3c00u16.to_le_bytes(),
        2.0 => 0x4000u16.to_le_bytes(),
        _ => panic!("fixture only encodes exact half values"),
    }
}

#[test]
fn q4_1_decodes_affine_low_high_lanes_and_checked_packed_extent() {
    let mut block = [0u8; 20];
    block[..2].copy_from_slice(&half_bits(2.0));
    block[2..4].copy_from_slice(&half_bits(0.5));
    for (lane, packed) in block[4..].iter_mut().enumerate() {
        *packed = ((15 - lane as u8) << 4) | lane as u8;
    }

    let decoded = decode_q4_1_block(&block).unwrap();
    let expected = (0..16)
        .chain((0..16).rev())
        .map(|quant| quant as f32 * 2.0 + 0.5)
        .collect::<Vec<_>>();
    assert_eq!(decoded.as_slice(), expected);

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Q4_1, Shape::from([2, 32]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 32);
    assert_eq!(packed.descriptor().block_bytes, 20);
    assert_eq!(packed.descriptor().bytes, 40);
    let shared = packed.clone();
    assert!(std::ptr::eq(
        packed.bytes().as_ptr(),
        shared.bytes().as_ptr()
    ));
    shared.validate().unwrap();
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(materialized.values().len(), 64);
    assert_eq!(&materialized.values()[..32], expected.as_slice());
    assert_eq!(&materialized.values()[32..], expected.as_slice());
}

#[test]
fn q4_1_rejects_bad_block_lengths_nonfinite_fields_and_overflow() {
    assert_eq!(
        decode_q4_1_block(&[0; 19]),
        Err(BlockDecodeError::Length {
            expected: 20,
            actual: 19,
        })
    );
    assert_eq!(
        decode_q4_1_block(&[0; 21]),
        Err(BlockDecodeError::Length {
            expected: 20,
            actual: 21,
        })
    );

    let mut nonfinite_scale = [0u8; 20];
    nonfinite_scale[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert_eq!(
        decode_q4_1_block(&nonfinite_scale),
        Err(BlockDecodeError::NonFinite)
    );
    let mut nonfinite_minimum = [0u8; 20];
    nonfinite_minimum[2..4].copy_from_slice(&0x7e00u16.to_le_bytes());
    assert_eq!(
        decode_q4_1_block(&nonfinite_minimum),
        Err(BlockDecodeError::NonFinite)
    );

    assert_eq!(
        QuantizedTensorData::new(GgmlType::Q4_1, Shape::from([usize::MAX, 32]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn q5_0_decodes_source_order_high_bits_nibble_halves_and_packed_extent() {
    let mut block = [0u8; 22];
    block[..2].copy_from_slice(&half_bits(0.5));
    block[2..6].copy_from_slice(&[0x01, 0x02, 0x04, 0x08]);
    for (lane, packed) in block[6..].iter_mut().enumerate() {
        *packed = ((15 - lane as u8) << 4) | lane as u8;
    }

    let expected = [
        0.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, 4.5, -3.0, -2.5, -2.0, -1.5, -1.0,
        -0.5, -0.5, -1.0, 6.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, 2.0, -6.5, -7.0,
        -7.5, -8.0,
    ];
    assert_eq!(decode_q5_0_block(&block).unwrap(), expected);

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Q5_0, Shape::from([2, 32]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 32);
    assert_eq!(packed.descriptor().block_bytes, 22);
    assert_eq!(packed.descriptor().bytes, 44);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(materialized.values().len(), 64);
    assert_eq!(&materialized.values()[..32], expected.as_slice());
    assert_eq!(&materialized.values()[32..], expected.as_slice());
}

#[test]
fn q5_0_rejects_bad_block_lengths_nonfinite_scale_and_overflow() {
    assert_eq!(
        decode_q5_0_block(&[0; 21]),
        Err(BlockDecodeError::Length {
            expected: 22,
            actual: 21,
        })
    );
    assert_eq!(
        decode_q5_0_block(&[0; 23]),
        Err(BlockDecodeError::Length {
            expected: 22,
            actual: 23,
        })
    );

    let mut nonfinite = [0u8; 22];
    nonfinite[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert_eq!(
        decode_q5_0_block(&nonfinite),
        Err(BlockDecodeError::NonFinite)
    );

    assert_eq!(
        QuantizedTensorData::new(GgmlType::Q5_0, Shape::from([usize::MAX, 32]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn q5_1_decodes_source_order_high_bits_nibble_halves_and_packed_extent() {
    let mut block = [0u8; 24];
    block[..2].copy_from_slice(&half_bits(0.5));
    block[2..4].copy_from_slice(&half_bits(2.0));
    block[4..8].copy_from_slice(&[0x01, 0x02, 0x04, 0x08]);
    for (lane, packed) in block[8..].iter_mut().enumerate() {
        *packed = ((15 - lane as u8) << 4) | lane as u8;
    }

    let expected = [
        10.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 14.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.5, 9.0,
        16.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 12.0, 3.5, 3.0, 2.5, 2.0,
    ];
    assert_eq!(decode_q5_1_block(&block).unwrap(), expected);

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Q5_1, Shape::from([2, 32]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 32);
    assert_eq!(packed.descriptor().block_bytes, 24);
    assert_eq!(packed.descriptor().bytes, 48);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(materialized.values().len(), 64);
    assert_eq!(&materialized.values()[..32], expected.as_slice());
    assert_eq!(&materialized.values()[32..], expected.as_slice());
}

#[test]
fn q5_1_rejects_bad_block_lengths_nonfinite_fields_and_overflow() {
    assert_eq!(
        decode_q5_1_block(&[0; 23]),
        Err(BlockDecodeError::Length {
            expected: 24,
            actual: 23,
        })
    );
    assert_eq!(
        decode_q5_1_block(&[0; 25]),
        Err(BlockDecodeError::Length {
            expected: 24,
            actual: 25,
        })
    );

    let mut nonfinite_scale = [0u8; 24];
    nonfinite_scale[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert_eq!(
        decode_q5_1_block(&nonfinite_scale),
        Err(BlockDecodeError::NonFinite)
    );
    let mut nonfinite_minimum = [0u8; 24];
    nonfinite_minimum[2..4].copy_from_slice(&0x7e00u16.to_le_bytes());
    assert_eq!(
        decode_q5_1_block(&nonfinite_minimum),
        Err(BlockDecodeError::NonFinite)
    );

    assert_eq!(
        QuantizedTensorData::new(GgmlType::Q5_1, Shape::from([usize::MAX, 32]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn mxfp4_decodes_two_nibble_halves_signed_zero_and_repeated_blocks() {
    let mut block = [0u8; 17];
    block[0] = 128;
    for (lane, packed) in block[1..].iter_mut().enumerate() {
        *packed = ((15 - lane as u8) << 4) | lane as u8;
    }
    let lut: [f32; 16] = [
        0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
    ];
    let expected = (0..16)
        .map(|code| lut[code])
        .chain((0..16).rev().map(|code| lut[code]))
        .collect::<Vec<_>>();
    let decoded = decode_mxfp4_block(&block).unwrap();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(decoded[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(decoded[8].to_bits(), (-0.0f32).to_bits());
    assert_eq!(decoded[23].to_bits(), (-0.0f32).to_bits());

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Mxfp4, Shape::from([2, 32]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 32);
    assert_eq!(packed.descriptor().block_bytes, 17);
    assert_eq!(packed.descriptor().bytes, 34);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(&materialized.values()[..32], expected.as_slice());
    assert_eq!(&materialized.values()[32..], expected.as_slice());
}

#[test]
fn mxfp4_handles_raw_exponent_boundaries_and_rejects_overflow() {
    for (exponent, bits) in [(0, 0x0020_0000), (1, 0x0040_0000), (2, 0x0080_0000)] {
        let mut block = [0u8; 17];
        block[0] = exponent;
        block[1] = 1;
        assert_eq!(decode_mxfp4_block(&block).unwrap()[0].to_bits(), bits);
    }
    let mut overflowing = [0u8; 17];
    overflowing[0] = 255;
    overflowing[1] = 7;
    assert_eq!(
        decode_mxfp4_block(&overflowing),
        Err(BlockDecodeError::NonFinite)
    );
    assert_eq!(
        decode_mxfp4_block(&[0; 16]),
        Err(BlockDecodeError::Length {
            expected: 17,
            actual: 16,
        })
    );
    assert_eq!(
        decode_mxfp4_block(&[0; 18]),
        Err(BlockDecodeError::Length {
            expected: 17,
            actual: 18,
        })
    );
    assert_eq!(
        QuantizedTensorData::new(GgmlType::Mxfp4, Shape::from([usize::MAX, 32]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn q1_0_decodes_transposed_bits_signed_zero_and_repeated_blocks() {
    let mut block = [0u8; 18];
    block[..2].copy_from_slice(&half_bits(2.0));
    for (byte, payload) in block[2..].iter_mut().enumerate() {
        *payload = 1 << (byte % 8);
    }
    let expected = (0..8)
        .flat_map(|bit| (0..16).map(move |byte| if byte % 8 == bit { 2.0 } else { -2.0 }))
        .collect::<Vec<f32>>();
    let decoded = decode_q1_0_block(&block).unwrap();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(
        &decoded[..18],
        &[
            2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, 2.0,
        ]
    );

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Q1_0, Shape::from([2, 128]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 128);
    assert_eq!(packed.descriptor().block_bytes, 18);
    assert_eq!(packed.descriptor().bytes, 36);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(&materialized.values()[..128], expected.as_slice());
    assert_eq!(&materialized.values()[128..], expected.as_slice());
}

#[test]
fn q1_0_handles_scale_signs_and_rejects_invalid_scale_lengths_and_extent() {
    let mut negative = [0u8; 18];
    negative[..2].copy_from_slice(&0xc000u16.to_le_bytes());
    negative[2] = 1;
    let decoded = decode_q1_0_block(&negative).unwrap();
    assert_eq!(decoded[0], -2.0);
    assert_eq!(decoded[1], 2.0);

    let mut negative_zero = [0u8; 18];
    negative_zero[..2].copy_from_slice(&0x8000u16.to_le_bytes());
    negative_zero[2] = 1;
    let decoded = decode_q1_0_block(&negative_zero).unwrap();
    assert_eq!(decoded[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(decoded[1].to_bits(), 0.0f32.to_bits());

    for bits in [0x7c00u16, 0x7e00u16] {
        let mut nonfinite = [0u8; 18];
        nonfinite[..2].copy_from_slice(&bits.to_le_bytes());
        assert_eq!(
            decode_q1_0_block(&nonfinite),
            Err(BlockDecodeError::NonFinite)
        );
    }
    assert_eq!(
        decode_q1_0_block(&[0; 17]),
        Err(BlockDecodeError::Length {
            expected: 18,
            actual: 17,
        })
    );
    assert_eq!(
        decode_q1_0_block(&[0; 19]),
        Err(BlockDecodeError::Length {
            expected: 18,
            actual: 19,
        })
    );
    assert_eq!(
        QuantizedTensorData::new(GgmlType::Q1_0, Shape::from([usize::MAX, 128]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn iq4_xs_decodes_all_group_scales_and_nibble_halves() {
    let mut block = [0u8; 136];
    block[..2].copy_from_slice(&half_bits(1.0));
    let raw_scales = [0u8, 9, 18, 27, 36, 45, 54, 63];
    let mut scales_h = 0u16;
    for (group, &raw) in raw_scales.iter().enumerate() {
        scales_h |= u16::from(raw >> 4) << (2 * group);
        block[4 + group / 2] |= (raw & 0x0f) << ((group % 2) * 4);
    }
    block[2..4].copy_from_slice(&scales_h.to_le_bytes());
    for group in 0..8 {
        for lane in 0..16 {
            let low = (lane % 16) as u8;
            let high = (15 - lane) as u8;
            block[8 + group * 16 + lane] = low | (high << 4);
        }
    }

    let lut = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];
    let expected = raw_scales
        .iter()
        .flat_map(|&raw| {
            let scale = f32::from(raw as i8 - 32);
            (0..16)
                .map(move |code| scale * lut[code])
                .chain((0..16).rev().map(move |code| scale * lut[code]))
        })
        .collect::<Vec<_>>();
    let decoded = decode_iq4_xs_block(&block).unwrap();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(decoded.len(), 256);

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Iq4Xs, Shape::from([2, 256]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 256);
    assert_eq!(packed.descriptor().block_bytes, 136);
    assert_eq!(packed.descriptor().bytes, 272);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(&materialized.values()[..256], expected.as_slice());
    assert_eq!(&materialized.values()[256..], expected.as_slice());
}

#[test]
fn iq4_xs_preserves_signed_zero_and_rejects_invalid_fields_lengths_and_extent() {
    let mut signed_zero = [0u8; 136];
    signed_zero[..2].copy_from_slice(&0x8000u16.to_le_bytes());
    let decoded = decode_iq4_xs_block(&signed_zero).unwrap();
    assert_eq!(decoded[0].to_bits(), (-0.0f32).to_bits());

    let mut zero_scale = [0u8; 136];
    zero_scale[..2].copy_from_slice(&half_bits(1.0));
    zero_scale[2..4].copy_from_slice(&0x0002u16.to_le_bytes());
    zero_scale[4] = 0x00;
    zero_scale[8] = 8;
    assert_eq!(
        decode_iq4_xs_block(&zero_scale).unwrap()[0].to_bits(),
        0.0f32.to_bits()
    );

    assert_eq!(
        decode_iq4_xs_block(&[0; 135]),
        Err(BlockDecodeError::Length {
            expected: 136,
            actual: 135,
        })
    );
    assert_eq!(
        decode_iq4_xs_block(&[0; 137]),
        Err(BlockDecodeError::Length {
            expected: 136,
            actual: 137,
        })
    );
    for bits in [0x7c00u16, 0x7e00u16] {
        let mut nonfinite = [0u8; 136];
        nonfinite[..2].copy_from_slice(&bits.to_le_bytes());
        assert_eq!(
            decode_iq4_xs_block(&nonfinite),
            Err(BlockDecodeError::NonFinite)
        );
    }
    assert_eq!(
        QuantizedTensorData::new(GgmlType::Iq4Xs, Shape::from([usize::MAX, 256]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn iq3_xxs_decodes_grid_boundaries_scale_words_signs_and_repeated_blocks() {
    let mut block = [0u8; 98];
    block[..2].copy_from_slice(&half_bits(1.0));
    let selectors = [0u8, 1, 254, 255, 0, 1, 254, 255];
    for group in 0..8 {
        block[2 + group * 8..2 + (group + 1) * 8].copy_from_slice(&selectors);
    }
    let sign_fields = [0u8, 1, 0x3f, 0x7f];
    for group in 0..8 {
        let word = u32::from(group as u8) << 28
            | u32::from(sign_fields[0])
            | (u32::from(sign_fields[1]) << 7)
            | (u32::from(sign_fields[2]) << 14)
            | (u32::from(sign_fields[3]) << 21);
        block[66 + group * 4..66 + (group + 1) * 4].copy_from_slice(&word.to_le_bytes());
    }

    let selected_grids = [
        (0u8, 0x0404_0404u32),
        (1u8, 0x0404_0414u32),
        (254u8, 0x3e2c_1424u32),
        (255u8, 0x3e34_1c04u32),
    ];
    let expected = (0..8)
        .flat_map(|group| {
            let scale = (group as f32 + 0.5) * 0.5;
            (0..4).flat_map(move |sign_field| {
                let selector = sign_fields[sign_field];
                let parity = if selector.count_ones() % 2 == 0 {
                    0
                } else {
                    0x80
                };
                let signs = selector | parity;
                (0..8).map(move |lane| {
                    let grid_selector = selectors[sign_field * 2 + lane / 4];
                    let grid_word = selected_grids
                        .iter()
                        .find_map(|&(index, word)| (index == grid_selector).then_some(word))
                        .unwrap();
                    let grid = ((grid_word >> (8 * (lane % 4))) & 0xff) as f32;
                    let sign = if (signs >> lane) & 1 == 0 { 1.0 } else { -1.0 };
                    scale * grid * sign
                })
            })
        })
        .collect::<Vec<_>>();
    let decoded = decode_iq3_xxs_block(&block).unwrap();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(decoded.len(), 256);

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Iq3Xxs, Shape::from([2, 256]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 256);
    assert_eq!(packed.descriptor().block_bytes, 98);
    assert_eq!(packed.descriptor().bytes, 196);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(&materialized.values()[..256], expected.as_slice());
    assert_eq!(&materialized.values()[256..], expected.as_slice());
}

#[test]
fn iq3_xxs_rejects_invalid_fields_lengths_and_packed_extent() {
    assert_eq!(
        decode_iq3_xxs_block(&[0; 97]),
        Err(BlockDecodeError::Length {
            expected: 98,
            actual: 97,
        })
    );
    assert_eq!(
        decode_iq3_xxs_block(&[0; 99]),
        Err(BlockDecodeError::Length {
            expected: 98,
            actual: 99,
        })
    );
    for bits in [0x7c00u16, 0x7e00u16] {
        let mut nonfinite = [0u8; 98];
        nonfinite[..2].copy_from_slice(&bits.to_le_bytes());
        assert_eq!(
            decode_iq3_xxs_block(&nonfinite),
            Err(BlockDecodeError::NonFinite)
        );
    }
    let mut largest_finite = [0u8; 98];
    largest_finite[..2].copy_from_slice(&0x7bffu16.to_le_bytes());
    largest_finite[2..66].fill(255);
    largest_finite[66..98].fill(0xff);
    assert!(
        decode_iq3_xxs_block(&largest_finite)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(
        QuantizedTensorData::new(GgmlType::Iq3Xxs, Shape::from([usize::MAX, 256]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn iq3_s_decodes_grid_high_plane_scales_signs_and_repeated_blocks() {
    let mut block = [0u8; 110];
    block[..2].copy_from_slice(&half_bits(1.0));
    let low_selectors = [0u8, 1, 254, 255, 0, 1, 254, 255];
    for group in 0..8 {
        block[2 + group * 8..2 + (group + 1) * 8].copy_from_slice(&low_selectors);
        block[66 + group] = 0xcc;
    }
    let sign_bytes = [0x00u8, 0xff, 0x55, 0xaa];
    for group in 0..8 {
        block[74 + group * 4..74 + (group + 1) * 4].copy_from_slice(&sign_bytes);
        block[106 + group / 2] |= (group as u8) << ((group % 2) * 4);
    }

    let selected_grids = [
        (0usize, 0x0101_0101u32),
        (1usize, 0x0101_0103u32),
        (510usize, 0x0f0d_0703u32),
        (511usize, 0x0f0f_0101u32),
    ];
    let expected = (0..8)
        .flat_map(|group| {
            let scale = 1.0 + 2.0 * group as f32;
            (0..4).flat_map(move |segment| {
                let signs = sign_bytes[segment];
                (0..8).map(move |lane| {
                    let position = segment * 2 + lane / 4;
                    let high = if position == 2 || position == 3 || position == 6 || position == 7 {
                        1usize
                    } else {
                        0
                    };
                    let selector = usize::from(low_selectors[position]) | (high << 8);
                    let grid_word = selected_grids
                        .iter()
                        .find_map(|&(index, word)| (index == selector).then_some(word))
                        .unwrap();
                    let grid = ((grid_word >> (8 * (lane % 4))) & 0xff) as f32;
                    let sign = if (signs >> lane) & 1 == 0 { 1.0 } else { -1.0 };
                    scale * grid * sign
                })
            })
        })
        .collect::<Vec<_>>();
    let decoded = decode_iq3_s_block(&block).unwrap();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(decoded.len(), 256);

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Iq3S, Shape::from([2, 256]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 256);
    assert_eq!(packed.descriptor().block_bytes, 110);
    assert_eq!(packed.descriptor().bytes, 220);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(&materialized.values()[..256], expected.as_slice());
    assert_eq!(&materialized.values()[256..], expected.as_slice());
}

#[test]
fn iq3_s_preserves_signed_zero_and_rejects_invalid_fields_lengths_and_extent() {
    let mut signed_zero = [0u8; 110];
    signed_zero[..2].copy_from_slice(&0x8000u16.to_le_bytes());
    signed_zero[74] = 1;
    let decoded = decode_iq3_s_block(&signed_zero).unwrap();
    assert_eq!(decoded[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(decoded[1].to_bits(), (-0.0f32).to_bits());

    assert_eq!(
        decode_iq3_s_block(&[0; 109]),
        Err(BlockDecodeError::Length {
            expected: 110,
            actual: 109,
        })
    );
    assert_eq!(
        decode_iq3_s_block(&[0; 111]),
        Err(BlockDecodeError::Length {
            expected: 110,
            actual: 111,
        })
    );
    for bits in [0x7c00u16, 0x7e00u16] {
        let mut nonfinite = [0u8; 110];
        nonfinite[..2].copy_from_slice(&bits.to_le_bytes());
        assert_eq!(
            decode_iq3_s_block(&nonfinite),
            Err(BlockDecodeError::NonFinite)
        );
    }
    let mut largest_finite = [0u8; 110];
    largest_finite[..2].copy_from_slice(&0x7bffu16.to_le_bytes());
    largest_finite[2..66].fill(0xff);
    largest_finite[66..74].fill(0xff);
    largest_finite[74..106].fill(0xff);
    largest_finite[106..110].fill(0xff);
    assert!(
        decode_iq3_s_block(&largest_finite)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(
        QuantizedTensorData::new(GgmlType::Iq3S, Shape::from([usize::MAX, 256]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn iq2_s_decodes_selectors_high_plane_scales_signs_and_repeated_blocks() {
    let mut block = [0u8; 82];
    block[..2].copy_from_slice(&half_bits(1.0));
    let lows = [0u8, 1, 254, 255];
    for position in 0..32 {
        block[2 + position] = lows[position % 4];
        block[34 + position] = [0x00, 0xff, 0x55, 0xaa][position % 4];
        block[66 + position / 4] = if position < 4 { 0xf0 } else { 0xe4 };
    }
    for group in 0..16 {
        block[74 + group / 2] |= (group as u8) << ((group % 2) * 4);
    }
    let grids = [
        (0usize, 0x0808_0808_0808_0808u64),
        (1usize, 0x0808_0808_0808_082bu64),
        (257usize, 0x0819_0819_1919_2b08u64),
        (766usize, 0x192b_0808_1908_0808u64),
        (1022usize, 0x2b2b_2b2b_2b08_2b08u64),
        (1023usize, 0x2b2b_2b2b_2b2b_2b2bu64),
    ];
    let expected = (0..16)
        .flat_map(|group| {
            (0..2).flat_map(move |segment| {
                let position = group * 2 + segment;
                let high = if position < 4 {
                    [0usize, 0, 3, 3][position]
                } else {
                    position % 4
                };
                let selector = usize::from(lows[position % 4]) | (high << 8);
                let word = grids
                    .iter()
                    .find_map(|&(index, word)| (index == selector).then_some(word))
                    .unwrap();
                let signs = [0x00u8, 0xff, 0x55, 0xaa][position % 4];
                (0..8).map(move |lane| {
                    let grid = ((word >> (lane * 8)) & 0xff) as f32;
                    let sign = if (signs >> lane) & 1 == 0 { 1.0 } else { -1.0 };
                    (group as f32 + 0.5) * 0.25 * grid * sign
                })
            })
        })
        .collect::<Vec<_>>();
    let decoded = decode_iq2_s_block(&block).unwrap();
    assert_eq!(decoded.as_slice(), expected);
    let packed = QuantizedTensorData::new(
        GgmlType::Iq2S,
        Shape::from([2, 256]),
        block.into_iter().chain(block).collect(),
    )
    .unwrap();
    assert_eq!(packed.descriptor().bytes, 164);
    assert_eq!(
        packed.dequantize_f32().unwrap().values(),
        expected.repeat(2)
    );
}

#[test]
fn iq2_s_preserves_signed_zero_and_rejects_invalid_fields_lengths_and_extent() {
    let mut signed_zero = [0u8; 82];
    signed_zero[..2].copy_from_slice(&0x8000u16.to_le_bytes());
    signed_zero[34] = 1;
    let decoded = decode_iq2_s_block(&signed_zero).unwrap();
    // The first sign bit negates the shared -0 scale, while the second lane
    // retains it. Keep the IEEE payload expectation lane-exact.
    assert_eq!(decoded[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(decoded[1].to_bits(), (-0.0f32).to_bits());
    for (length, expected) in [(81, 82), (83, 82)] {
        assert_eq!(
            decode_iq2_s_block(&vec![0; length]),
            Err(BlockDecodeError::Length {
                expected,
                actual: length
            })
        );
    }
    for bits in [0x7c00u16, 0x7e00u16] {
        let mut nonfinite = [0u8; 82];
        nonfinite[..2].copy_from_slice(&bits.to_le_bytes());
        assert_eq!(
            decode_iq2_s_block(&nonfinite),
            Err(BlockDecodeError::NonFinite)
        );
    }
    let mut largest_finite = [0xffu8; 82];
    largest_finite[..2].copy_from_slice(&0x7bffu16.to_le_bytes());
    assert!(
        decode_iq2_s_block(&largest_finite)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(
        QuantizedTensorData::new(GgmlType::Iq2S, Shape::from([usize::MAX, 256]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn q5_k_decodes_all_groups_high_planes_and_checked_packed_extent() {
    let mut block = [0u8; 176];
    block[..2].copy_from_slice(&half_bits(1.0));
    block[2..4].copy_from_slice(&half_bits(0.5));
    block[4..16].copy_from_slice(&[
        0x41, 0x82, 0xc3, 0x04, 0x45, 0x86, 0xc7, 0x08, 0xa9, 0xb8, 0xc7, 0xd6,
    ]);
    for (lane, high_bits) in block[16..48].iter_mut().enumerate() {
        *high_bits = 1 << (lane % 8);
    }
    for (lane, packed) in block[48..80].iter_mut().enumerate() {
        let low = lane as u8 % 16;
        *packed = ((15 - low) << 4) | low;
    }
    block[80..112].fill(0x21);
    block[112..144].fill(0x43);
    block[144..176].fill(0x65);

    let scales = [1., 2., 3., 4., 25., 40., 55., 6.];
    let mins = [5., 6., 7., 8., 26., 43., 60., 13.];
    let group_quants = [
        (0..32).map(|i| (i % 16) as f32).collect::<Vec<_>>(),
        (0..32).map(|i| (15 - i % 16) as f32).collect(),
        vec![1.; 32],
        vec![2.; 32],
        vec![3.; 32],
        vec![4.; 32],
        vec![5.; 32],
        vec![6.; 32],
    ];
    let expected = group_quants
        .iter()
        .enumerate()
        .flat_map(|(group, values)| {
            values.iter().enumerate().map(move |(lane, &nibble)| {
                let high = if lane % 8 == group { 16.0 } else { 0.0 };
                scales[group] * (nibble + high) - 0.5 * mins[group]
            })
        })
        .collect::<Vec<_>>();
    let decoded = decode_q5_k_block(&block).unwrap();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(decoded.len(), 256);
    assert_eq!(&decoded[..4], &[13.5, -1.5, -0.5, 0.5]);
    assert_eq!(&decoded[32..36], &[27., 57., 23., 21.]);
    assert_eq!(&decoded[224..228], &[29.5, 29.5, 29.5, 29.5]);

    let bytes = block.into_iter().chain(block).collect();
    let packed = QuantizedTensorData::new(GgmlType::Q5K, Shape::from([2, 256]), bytes).unwrap();
    assert_eq!(packed.descriptor().block_elements, 256);
    assert_eq!(packed.descriptor().block_bytes, 176);
    assert_eq!(packed.descriptor().bytes, 352);
    let materialized = packed.dequantize_f32().unwrap();
    assert_eq!(materialized.values().len(), 512);
    assert_eq!(&materialized.values()[..256], expected.as_slice());
    assert_eq!(&materialized.values()[256..], expected.as_slice());
}

#[test]
fn q5_k_rejects_bad_blocks_nonfinite_fields_and_packed_extent_overflow() {
    assert_eq!(
        decode_q5_k_block(&[0; 175]),
        Err(BlockDecodeError::Length {
            expected: 176,
            actual: 175,
        })
    );
    assert_eq!(
        decode_q5_k_block(&[0; 177]),
        Err(BlockDecodeError::Length {
            expected: 176,
            actual: 177,
        })
    );
    let mut nonfinite_scale = [0u8; 176];
    nonfinite_scale[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert_eq!(
        decode_q5_k_block(&nonfinite_scale),
        Err(BlockDecodeError::NonFinite)
    );
    let mut nonfinite_minimum = [0u8; 176];
    nonfinite_minimum[2..4].copy_from_slice(&0x7e00u16.to_le_bytes());
    assert_eq!(
        decode_q5_k_block(&nonfinite_minimum),
        Err(BlockDecodeError::NonFinite)
    );

    let mut largest_finite = [0u8; 176];
    largest_finite[..2].copy_from_slice(&0x7bffu16.to_le_bytes());
    largest_finite[2..4].copy_from_slice(&0x7bffu16.to_le_bytes());
    largest_finite[4..].fill(0xff);
    assert!(
        decode_q5_k_block(&largest_finite)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );

    assert_eq!(
        QuantizedTensorData::new(GgmlType::Q5K, Shape::from([usize::MAX, 256]), vec![]),
        Err(QuantizedError::Overflow)
    );
}

#[test]
fn q4_k_decodes_packed_scale_min_boundaries_and_group_order() {
    let mut block = [0u8; 144];
    block[..2].copy_from_slice(&half_bits(1.0));
    block[2..4].copy_from_slice(&half_bits(0.5));
    block[4..16].copy_from_slice(&[
        0x41, 0x82, 0xc3, 0x04, 0x45, 0x86, 0xc7, 0x08, 0xa9, 0xb8, 0xc7, 0xd6,
    ]);
    for (lane, packed) in block[16..48].iter_mut().enumerate() {
        let low = lane as u8 % 16;
        *packed = ((15 - low) << 4) | low;
    }
    block[48..80].fill(0x21);
    block[80..112].fill(0x43);
    block[112..144].fill(0x65);

    let decoded = decode_q4_k_block(&block).unwrap();
    let scales = [1., 2., 3., 4., 25., 40., 55., 6.];
    let mins = [5., 6., 7., 8., 26., 43., 60., 13.];
    let group_quants = [
        (0..32).map(|i| (i % 16) as f32).collect::<Vec<_>>(),
        (0..32).map(|i| (15 - i % 16) as f32).collect(),
        vec![1.; 32],
        vec![2.; 32],
        vec![3.; 32],
        vec![4.; 32],
        vec![5.; 32],
        vec![6.; 32],
    ];
    let expected = group_quants
        .iter()
        .enumerate()
        .flat_map(|(group, values)| {
            values
                .iter()
                .map(move |q| scales[group] * q - 0.5 * mins[group])
        })
        .collect::<Vec<_>>();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(&decoded[..4], &[-2.5, -1.5, -0.5, 0.5]);
    assert_eq!(&decoded[128..132], &[62., 62., 62., 62.]);
    assert_eq!(&decoded[252..], &[29.5, 29.5, 29.5, 29.5]);
}

#[test]
fn q6_k_decodes_high_planes_signed_scales_and_flatten_order() {
    let mut block = [0u8; 210];
    block[..128].fill(0xf0);
    block[128..160].fill(0xe4);
    block[160..192].fill(0x1b);
    for (index, scale) in (-8i8..=7).enumerate() {
        block[192 + index] = scale as u8;
    }
    block[208..].copy_from_slice(&half_bits(0.5));

    let decoded = decode_q6_k_block(&block).unwrap();
    let quant_groups = [
        -32, -32, -16, -16, 15, 15, 31, 31, 16, 16, 0, 0, -1, -1, -17, -17,
    ];
    let expected = quant_groups
        .iter()
        .enumerate()
        .flat_map(|(scale_index, &quant)| {
            std::iter::repeat_n(0.5 * quant as f32 * (scale_index as i32 - 8) as f32, 16)
        })
        .collect::<Vec<_>>();
    assert_eq!(decoded.as_slice(), expected);
    assert_eq!(&decoded[..16], &[128.; 16]);
    assert_eq!(&decoded[112..128], &[-15.5; 16]);
    assert_eq!(&decoded[128..144], &[0.; 16]);
    assert_eq!(&decoded[240..], &[-59.5; 16]);
}

#[test]
fn k_block_decoders_reject_wrong_lengths_and_nonfinite_scales() {
    assert_eq!(
        decode_q4_k_block(&[0; 143]),
        Err(BlockDecodeError::Length {
            expected: 144,
            actual: 143,
        })
    );
    assert_eq!(
        decode_q4_k_block(&[0; 145]),
        Err(BlockDecodeError::Length {
            expected: 144,
            actual: 145,
        })
    );
    assert_eq!(
        decode_q6_k_block(&[0; 209]),
        Err(BlockDecodeError::Length {
            expected: 210,
            actual: 209,
        })
    );
    assert_eq!(
        decode_q6_k_block(&[0; 211]),
        Err(BlockDecodeError::Length {
            expected: 210,
            actual: 211,
        })
    );

    let mut q4 = [0u8; 144];
    q4[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert_eq!(decode_q4_k_block(&q4), Err(BlockDecodeError::NonFinite));
    let mut q6 = [0u8; 210];
    q6[208..].copy_from_slice(&0x7e00u16.to_le_bytes());
    assert_eq!(decode_q6_k_block(&q6), Err(BlockDecodeError::NonFinite));

    let mut subnormal = [0u8; 144];
    subnormal[..2].copy_from_slice(&1u16.to_le_bytes());
    subnormal[4] = 1;
    subnormal[16] = 1;
    assert_eq!(
        decode_q4_k_block(&subnormal).unwrap()[0],
        f32::from_bits(0x3380_0000)
    );
}
