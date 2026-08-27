use super::quantization::blocks::{
    BlockDecodeError, decode_q4_1_block, decode_q4_k_block, decode_q5_0_block, decode_q5_1_block,
    decode_q5_k_block, decode_q6_k_block,
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
        0.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, 4.5, -3.0, -2.5, -2.0, -1.5,
        -1.0, -0.5, -0.5, -1.0, 6.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, 2.0,
        -6.5, -7.0, -7.5, -8.0,
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
    assert_eq!(decode_q5_0_block(&nonfinite), Err(BlockDecodeError::NonFinite));

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
        10.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 14.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
        9.5, 9.0, 16.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 12.0, 3.5, 3.0, 2.5, 2.0,
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
    assert!(decode_q5_k_block(&largest_finite)
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));

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
