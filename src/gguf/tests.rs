use super::*;
use crate::{DType, Shape, Storage};

#[derive(Clone)]
struct TensorFixture<'a> {
    name: &'a str,
    dimensions: &'a [u64],
    kind: u32,
    offset: u64,
    data: &'a [u8],
}

fn push_string(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value);
}

fn metadata_u32(key: &str, value: u32) -> Vec<u8> {
    metadata_raw(key, 4, &value.to_le_bytes())
}

fn metadata_raw(key: &str, value_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    push_string(&mut out, key.as_bytes());
    out.extend_from_slice(&value_type.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn metadata_string(key: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    push_string(&mut out, key.as_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut out, value.as_bytes());
    out
}

fn metadata_bool(key: &str, value: u8) -> Vec<u8> {
    let mut out = Vec::new();
    push_string(&mut out, key.as_bytes());
    out.extend_from_slice(&7u32.to_le_bytes());
    out.push(value);
    out
}

fn metadata_u32_array(key: &str, values: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    push_string(&mut out, key.as_bytes());
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn fixture(
    version: u32,
    metadata: &[Vec<u8>],
    tensors: &[TensorFixture<'_>],
    alignment: usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for entry in metadata {
        out.extend_from_slice(entry);
    }
    for tensor in tensors {
        push_string(&mut out, tensor.name.as_bytes());
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in tensor.dimensions {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&tensor.kind.to_le_bytes());
        out.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    if !tensors.is_empty() {
        out.resize(out.len().next_multiple_of(alignment), 0);
    }
    let data_start = out.len();
    let required = tensors
        .iter()
        .map(|tensor| tensor.offset as usize + tensor.data.len())
        .max()
        .unwrap_or(0);
    out.resize(data_start + required, 0);
    for tensor in tensors {
        let start = data_start + tensor.offset as usize;
        out[start..start + tensor.data.len()].copy_from_slice(tensor.data);
    }
    out
}

fn assert_kind(bytes: &[u8], expected: GgufErrorKind) {
    let error = read_gguf(bytes).expect_err("fixture must be rejected");
    assert_eq!(error.kind(), &expected, "error was {error}");
}

#[test]
fn q4_0_and_q8_0_materialize_source_evidenced_block_order() {
    // tinygrad/llm/gguf.py: Q4_0 is f16 d then 16 low/high-nibble bytes;
    // Q4_1 adds a little-endian f16 minimum and uses q * d + m;
    // Q8_0 is f16 d then 32 signed bytes.
    let mut q4 = vec![0x00, 0x3c]; // d = 1
    q4.extend((0..16).map(|i| (15 - i) << 4 | i));
    let mut q41 = vec![0x00, 0x40, 0x00, 0x38]; // d = 2, m = 0.5
    q41.extend((0..16).map(|i| (15 - i) << 4 | i));
    let mut q5 = vec![0x00, 0x38, 0x01, 0x02, 0x04, 0x08]; // d = 0.5, high bits
    q5.extend((0..16).map(|i| (15 - i) << 4 | i));
    let mut q51 = vec![0x00, 0x38, 0x00, 0x40, 0x01, 0x02, 0x04, 0x08]; // d = 0.5, m = 2
    q51.extend((0..16).map(|i| (15 - i) << 4 | i));
    let mut q8 = vec![0x00, 0x38]; // d = 0.5
    q8.extend([0x80, 0xff, 0, 1, 127]);
    q8.resize(34, 0);
    let mut q4_two = q4.clone();
    q4_two.extend_from_slice(&q4);
    let mut q41_two = q41.clone();
    q41_two.extend_from_slice(&q41);
    let mut q5_two = q5.clone();
    q5_two.extend_from_slice(&q5);
    let mut q51_two = q51.clone();
    q51_two.extend_from_slice(&q51);
    let bytes = fixture(
        3,
        &[metadata_u32("general.alignment", 32)],
        &[
            TensorFixture {
                name: "q4",
                dimensions: &[32],
                kind: 2,
                offset: 0,
                data: &q4,
            },
            TensorFixture {
                name: "q8",
                dimensions: &[32],
                kind: 8,
                offset: 32,
                data: &q8,
            },
            TensorFixture {
                name: "q4-two",
                dimensions: &[64],
                kind: 2,
                offset: 96,
                data: &q4_two,
            },
            TensorFixture {
                name: "q41-two",
                dimensions: &[64],
                kind: 3,
                offset: 160,
                data: &q41_two,
            },
            TensorFixture {
                name: "q5-two",
                dimensions: &[64],
                kind: 6,
                offset: 224,
                data: &q5_two,
            },
            TensorFixture {
                name: "q51-two",
                dimensions: &[64],
                kind: 7,
                offset: 288,
                data: &q51_two,
            },
        ],
        32,
    );
    let file = read_gguf(&bytes).unwrap();
    let q4 = file.materialize_f32("q4").unwrap();
    assert_eq!(
        q4.values(),
        &[
            -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 7., 6., 5., 4.,
            3., 2., 1., 0., -1., -2., -3., -4., -5., -6., -7., -8.,
        ]
    );
    let q8 = file.materialize_f32("q8").unwrap();
    assert_eq!(&q8.values()[..5], &[-64., -0.5, 0., 0.5, 63.5]);
    let q4_two = file.materialize_f32("q4-two").unwrap();
    assert_eq!(&q4_two.values()[..32], q4.values());
    assert_eq!(&q4_two.values()[32..], q4.values());
    let q41_two = file.materialize_f32("q41-two").unwrap();
    assert_eq!(
        &q41_two.values()[..32],
        &[
            0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5, 24.5, 26.5,
            28.5, 30.5, 30.5, 28.5, 26.5, 24.5, 22.5, 20.5, 18.5, 16.5, 14.5, 12.5, 10.5, 8.5,
            6.5, 4.5, 2.5, 0.5,
        ]
    );
    assert_eq!(&q41_two.values()[32..], &q41_two.values()[..32]);
    let q5_two = file.materialize_f32("q5-two").unwrap();
    assert_eq!(
        &q5_two.values()[..32],
        &[
            0.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, 4.5, -3.0, -2.5, -2.0, -1.5,
            -1.0, -0.5, -0.5, -1.0, 6.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, 2.0,
            -6.5, -7.0, -7.5, -8.0,
        ]
    );
    assert_eq!(&q5_two.values()[32..], &q5_two.values()[..32]);
    let q51_two = file.materialize_f32("q51-two").unwrap();
    assert_eq!(
        &q51_two.values()[..32],
        &[
            10.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 14.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
            9.5, 9.0, 16.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 12.0, 3.5, 3.0, 2.5, 2.0,
        ]
    );
    assert_eq!(&q51_two.values()[32..], &q51_two.values()[..32]);
}

#[test]
fn q5_k_materializes_repeated_gguf_blocks() {
    let mut block = [0u8; 176];
    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
    block[4..16].copy_from_slice(&[
        0x41, 0x82, 0xc3, 0x04, 0x45, 0x86, 0xc7, 0x08, 0xa9, 0xb8, 0xc7, 0xd6,
    ]);
    for (lane, high_bits) in block[16..48].iter_mut().enumerate() {
        *high_bits = 1 << (lane % 8);
    }
    block[48..80].fill(0x10);
    block[80..112].fill(0x32);
    block[112..144].fill(0x54);
    block[144..176].fill(0x76);
    let mut packed = block.to_vec();
    packed.extend_from_slice(&block);
    let bytes = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "q5k",
            dimensions: &[512],
            kind: 13,
            offset: 0,
            data: &packed,
        }],
        32,
    );
    let materialized = read_gguf(&bytes).unwrap().materialize_f32("q5k").unwrap();
    assert_eq!(materialized.shape(), &Shape::from([512]));
    assert_eq!(materialized.values().len(), 512);
    assert_eq!(&materialized.values()[..256], &materialized.values()[256..]);
}

#[test]
fn mxfp4_materializes_checked_in_reference_block() {
    let block = [
        0x7a, 0x29, 0xab, 0x61, 0x10, 0x21, 0x02, 0x4a, 0x15, 0xca, 0x05, 0x01, 0x9b, 0x39,
        0x0b, 0x0b, 0x1c,
    ];
    let bytes = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "mxfp4",
            dimensions: &[32],
            kind: 39,
            offset: 0,
            data: &block,
        }],
        32,
    );
    let materialized = read_gguf(&bytes).unwrap().materialize_f32("mxfp4").unwrap();
    assert_eq!(
        materialized.values(),
        &[
            -0.015625, -0.046875, 0.015625, 0.0, 0.015625, 0.03125, -0.03125, 0.09375,
            -0.03125, 0.09375, 0.015625, -0.046875, -0.015625, -0.046875, -0.046875, -0.0625,
            0.03125, -0.03125, 0.125, 0.015625, 0.03125, 0.0, 0.0625, 0.015625, -0.0625, 0.0,
            0.0, -0.015625, 0.046875, 0.0, 0.0, 0.015625,
        ]
    );
}

#[test]
fn rank_two_quantized_weight_can_remain_exact_packed_storage() {
    let mut block = vec![0x00, 0x3c];
    block.extend(std::iter::repeat_n(0xe3, 16));
    let mut packed = block.clone();
    packed.extend_from_slice(&block);
    let bytes = fixture(
        3,
        &[metadata_u32("general.alignment", 32)],
        &[TensorFixture {
            name: "linear.weight",
            dimensions: &[32, 2],
            kind: 2,
            offset: 0,
            data: &packed,
        }],
        32,
    );
    let file = read_gguf(&bytes).unwrap();
    let weight = file.quantized_tensor("linear.weight").unwrap();
    assert_eq!(weight.descriptor().logical_shape, Shape::from([2, 32]));
    assert_eq!(weight.descriptor().ggml_type, GgmlType::Q4_0);
    assert_eq!(weight.descriptor().alignment, 1);
    assert_eq!(weight.bytes(), packed);
    assert_eq!(
        weight.dequantize_f32().unwrap(),
        file.materialize_f32("linear.weight").unwrap()
    );
    assert_eq!(weight, file.quantized_tensor("linear.weight").unwrap());
}

#[test]
fn whole_file_f32_state_is_complete_deterministic_and_atomic() {
    let mut q4_block = [0u8; 144];
    q4_block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    q4_block[4..8].fill(1);
    q4_block[12..16].fill(1);
    q4_block[16..].fill(0x21);
    let mut q4_two = q4_block.to_vec();
    let mut second = q4_block;
    second[..2].copy_from_slice(&0x4000u16.to_le_bytes());
    q4_two.extend_from_slice(&second);

    let mut q6 = [0u8; 210];
    q6[192..208].fill(1);
    q6[208..].copy_from_slice(&0x3800u16.to_le_bytes());
    let dense = [2i16.to_le_bytes(), (-3i16).to_le_bytes()].concat();
    let bytes = fixture(
        3,
        &[],
        &[
            TensorFixture {
                name: "z_dense",
                dimensions: &[2],
                kind: 25,
                offset: 544,
                data: &dense,
            },
            TensorFixture {
                name: "a_q4k",
                dimensions: &[512],
                kind: 12,
                offset: 0,
                data: &q4_two,
            },
            TensorFixture {
                name: "m_q6k",
                dimensions: &[256],
                kind: 14,
                offset: 320,
                data: &q6,
            },
        ],
        32,
    );
    let file = read_gguf(&bytes).unwrap();
    let state = file.materialize_state_f32().unwrap();
    assert_eq!(
        state.keys().map(String::as_str).collect::<Vec<_>>(),
        ["a_q4k", "m_q6k", "z_dense"]
    );
    assert!(state.values().all(|tensor| tensor.dtype() == DType::F32));
    assert_eq!(state["z_dense"].values(), &[2., -3.]);
    assert_eq!(state["a_q4k"].shape(), &Shape::from([512]));
    assert_eq!(&state["a_q4k"].values()[..32], &[1.; 32]);
    assert_eq!(&state["a_q4k"].values()[32..64], &[2.; 32]);
    assert_eq!(&state["a_q4k"].values()[256..288], &[2.; 32]);
    assert_eq!(&state["a_q4k"].values()[288..320], &[4.; 32]);
    assert_eq!(state["m_q6k"].values(), &[-16.; 256]);

    let unsupported = [0u8; 176];
    let unsupported_bytes = fixture(
        3,
        &[],
        &[
            TensorFixture {
                name: "unsupported-first",
                dimensions: &[256],
                kind: 13,
                offset: 0,
                data: &unsupported,
            },
            TensorFixture {
                name: "would-succeed",
                dimensions: &[2],
                kind: 25,
                offset: 192,
                data: &dense,
            },
        ],
        32,
    );
    let error = read_gguf(&unsupported_bytes)
        .unwrap()
        .materialize_state_f32()
        .unwrap_err();
    assert_eq!(
        error.kind(),
        &GgufErrorKind::QuantizedMaterialization {
            tensor: "unsupported-first".into(),
            kind: GgmlType::Q5K,
        }
    );
}

#[test]
fn k_block_tensor_geometry_rejects_partial_blocks() {
    let q4 = [0u8; 144];
    let bytes = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "q4k",
            dimensions: &[257],
            kind: 12,
            offset: 0,
            data: &q4,
        }],
        32,
    );
    assert_kind(
        &bytes,
        GgufErrorKind::BlockElementMismatch {
            tensor: "q4k".into(),
            elements: 257,
            block_elements: 256,
        },
    );

    let q6 = [0u8; 210];
    let misaligned = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "q6k",
            dimensions: &[256],
            kind: 14,
            offset: 1,
            data: &q6,
        }],
        32,
    );
    assert_kind(
        &misaligned,
        GgufErrorKind::MisalignedTensorOffset {
            tensor: "q6k".into(),
            offset: 1,
            alignment: 32,
        },
    );
}

#[test]
fn metadata_and_dense_tensor_inventory_preserve_order_and_bits() {
    let f32_bits = [
        1.0f32.to_bits(),
        (-0.0f32).to_bits(),
        0x7fc1_2345,
        f32::INFINITY.to_bits(),
    ];
    let f32_data: Vec<_> = f32_bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let f16_bits = [0x3c00u16, 0x8000];
    let f16_data: Vec<_> = f16_bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let metadata = vec![
        metadata_u32("general.alignment", 32),
        metadata_string("model.name", "tiny fixture"),
        metadata_u32_array("token.ids", &[7, 9, 11]),
        metadata_bool("model.enabled", 1),
    ];
    let tensors = [
        TensorFixture {
            name: "matrix",
            dimensions: &[2, 2],
            kind: 0,
            offset: 0,
            data: &f32_data,
        },
        TensorFixture {
            name: "half",
            dimensions: &[2],
            kind: 1,
            offset: 32,
            data: &f16_data,
        },
    ];
    let bytes = fixture(3, &metadata, &tensors, 32);
    let file = read_gguf(&bytes).unwrap();

    assert_eq!(file.version(), GgufVersion::V3);
    assert_eq!(file.alignment(), 32);
    assert_eq!(
        file.metadata()
            .iter()
            .map(GgufMetadata::key)
            .collect::<Vec<_>>(),
        [
            "general.alignment",
            "model.name",
            "token.ids",
            "model.enabled"
        ]
    );
    assert_eq!(
        file.metadata_value("model.name"),
        Some(&GgufMetadataValue::String("tiny fixture".into()))
    );
    assert_eq!(
        file.metadata_value("token.ids"),
        Some(&GgufMetadataValue::Array {
            element_type: GgufMetadataType::U32,
            values: vec![
                GgufMetadataValue::U32(7),
                GgufMetadataValue::U32(9),
                GgufMetadataValue::U32(11),
            ],
        })
    );
    assert_eq!(
        file.tensors()
            .iter()
            .map(GgufTensor::name)
            .collect::<Vec<_>>(),
        ["matrix", "half"]
    );
    let matrix = file.tensor("matrix").unwrap();
    assert_eq!(matrix.dimensions(), &[2, 2]);
    assert_eq!(matrix.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(matrix.layout(), GgmlLayout::Dense { dtype: DType::F32 });
    assert_eq!(file.tensor_bytes("matrix").unwrap(), f32_data);
    match file.materialize_dense("matrix").unwrap().storage() {
        Storage::F32(values) => assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            f32_bits
        ),
        storage => panic!("unexpected storage {storage:?}"),
    }
    assert_eq!(
        file.materialize_dense("half").unwrap().storage(),
        &Storage::F16(f16_bits.to_vec())
    );

    let version_two = fixture(2, &metadata, &tensors, 32);
    assert_eq!(read_gguf(&version_two).unwrap().version(), GgufVersion::V2);
}

#[test]
fn metadata_scalar_types_preserve_typed_wire_values() {
    let entries = vec![
        metadata_raw("value.u8", 0, &[0xfe]),
        metadata_raw("value.i8", 1, &[0xff]),
        metadata_raw("value.u16", 2, &0xfedcu16.to_le_bytes()),
        metadata_raw("value.i16", 3, &(-123i16).to_le_bytes()),
        metadata_u32("value.u32", 0xfeed_beef),
        metadata_raw("value.i32", 5, &(-456i32).to_le_bytes()),
        metadata_raw("value.f32", 6, &0x7fc1_2345u32.to_le_bytes()),
        metadata_bool("value.bool", 1),
        metadata_string("value.string", "hello"),
        metadata_raw("value.u64", 10, &u64::MAX.to_le_bytes()),
        metadata_raw("value.i64", 11, &i64::MIN.to_le_bytes()),
        metadata_raw("value.f64", 12, &(-0.0f64).to_bits().to_le_bytes()),
    ];
    let bytes = fixture(3, &entries, &[], 32);
    let file = read_gguf(&bytes).unwrap();
    assert_eq!(
        file.metadata_value("value.u8"),
        Some(&GgufMetadataValue::U8(0xfe))
    );
    assert_eq!(
        file.metadata_value("value.i8"),
        Some(&GgufMetadataValue::I8(-1))
    );
    assert_eq!(
        file.metadata_value("value.u16"),
        Some(&GgufMetadataValue::U16(0xfedc))
    );
    assert_eq!(
        file.metadata_value("value.i16"),
        Some(&GgufMetadataValue::I16(-123))
    );
    assert_eq!(
        file.metadata_value("value.u32"),
        Some(&GgufMetadataValue::U32(0xfeed_beef))
    );
    assert_eq!(
        file.metadata_value("value.i32"),
        Some(&GgufMetadataValue::I32(-456))
    );
    match file.metadata_value("value.f32") {
        Some(GgufMetadataValue::F32(value)) => assert_eq!(value.to_bits(), 0x7fc1_2345),
        value => panic!("unexpected f32 metadata {value:?}"),
    }
    assert_eq!(
        file.metadata_value("value.bool"),
        Some(&GgufMetadataValue::Bool(true))
    );
    assert_eq!(
        file.metadata_value("value.string"),
        Some(&GgufMetadataValue::String("hello".into()))
    );
    assert_eq!(
        file.metadata_value("value.u64"),
        Some(&GgufMetadataValue::U64(u64::MAX))
    );
    assert_eq!(
        file.metadata_value("value.i64"),
        Some(&GgufMetadataValue::I64(i64::MIN))
    );
    match file.metadata_value("value.f64") {
        Some(GgufMetadataValue::F64(value)) => assert_eq!(value.to_bits(), (-0.0f64).to_bits()),
        value => panic!("unexpected f64 metadata {value:?}"),
    }
    assert!(file.metadata_f64("value.f32").unwrap().unwrap().is_nan());
    assert_eq!(
        file.metadata_f64("value.f64").unwrap().unwrap().to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(
        file.metadata_f64("value.string").unwrap_err(),
        GgufMetadataAccessError::TypeMismatch {
            key: "value.string".to_owned(),
            expected: GgufMetadataExpectation::Float,
            actual: GgufMetadataType::String,
        }
    );
}

#[test]
fn quantized_tensor_remains_an_opaque_validated_block_payload() {
    let block: Vec<u8> = (0..18).collect();
    let bytes = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "q4",
            dimensions: &[32],
            kind: 2,
            offset: 0,
            data: &block,
        }],
        32,
    );
    let file = read_gguf(&bytes).unwrap();
    let tensor = file.tensor("q4").unwrap();
    assert_eq!(
        tensor.layout(),
        GgmlLayout::Quantized {
            block_elements: 32,
            block_bytes: 18,
        }
    );
    assert_eq!(tensor.elements(), 32);
    assert_eq!(tensor.byte_len(), 18);
    assert_eq!(file.tensor_bytes("q4").unwrap(), block);
    assert_eq!(
        file.materialize_dense("q4").unwrap_err().kind(),
        &GgufErrorKind::QuantizedMaterialization {
            tensor: "q4".into(),
            kind: GgmlType::Q4_0,
        }
    );
}

#[test]
fn source_evidenced_tensor_type_inventory_is_exact() {
    let dense = [
        (0, GgmlType::F32, DType::F32),
        (1, GgmlType::F16, DType::F16),
        (24, GgmlType::I8, DType::I8),
        (25, GgmlType::I16, DType::I16),
        (26, GgmlType::I32, DType::I32),
        (27, GgmlType::I64, DType::I64),
        (28, GgmlType::F64, DType::F64),
        (30, GgmlType::BF16, DType::BF16),
    ];
    for (raw, kind, dtype) in dense {
        assert_eq!(GgmlType::from_raw(raw), Some(kind), "dense type {raw}");
        assert_eq!(kind.raw(), raw, "dense type {raw}");
        assert_eq!(
            kind.layout(),
            GgmlLayout::Dense { dtype },
            "dense type {raw}"
        );
    }
    let quantized = [
        (2, GgmlType::Q4_0, 32, 18),
        (3, GgmlType::Q4_1, 32, 20),
        (6, GgmlType::Q5_0, 32, 22),
        (7, GgmlType::Q5_1, 32, 24),
        (8, GgmlType::Q8_0, 32, 34),
        (12, GgmlType::Q4K, 256, 144),
        (13, GgmlType::Q5K, 256, 176),
        (14, GgmlType::Q6K, 256, 210),
        (18, GgmlType::Iq3Xxs, 256, 98),
        (21, GgmlType::Iq3S, 256, 110),
        (22, GgmlType::Iq2S, 256, 82),
        (23, GgmlType::Iq4Xs, 256, 136),
        (39, GgmlType::Mxfp4, 32, 17),
        (41, GgmlType::Q1_0, 128, 18),
    ];
    for (raw, kind, block_elements, block_bytes) in quantized {
        assert_eq!(GgmlType::from_raw(raw), Some(kind), "quantized type {raw}");
        assert_eq!(kind.raw(), raw, "quantized type {raw}");
        assert_eq!(
            kind.layout(),
            GgmlLayout::Quantized {
                block_elements,
                block_bytes,
            },
            "quantized type {raw}"
        );
    }
    for unknown in [4, 5, 9, 10, 11, 15, 16, 17, 19, 20, 29, 31, 40, u32::MAX] {
        assert_eq!(GgmlType::from_raw(unknown), None, "unknown type {unknown}");
    }
}

#[test]
fn malformed_headers_metadata_and_tensor_tables_fail_structurally() {
    let data = 1.0f32.to_le_bytes();
    let base = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "x",
            dimensions: &[1],
            kind: 0,
            offset: 0,
            data: &data,
        }],
        32,
    );

    let mut bad_magic = base.clone();
    bad_magic[0] = b'B';
    assert_kind(&bad_magic, GgufErrorKind::InvalidMagic);

    let mut bad_version = base.clone();
    bad_version[4..8].copy_from_slice(&4u32.to_le_bytes());
    assert_kind(&bad_version, GgufErrorKind::UnsupportedVersion(4));

    let mut bad_count = base.clone();
    bad_count[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_kind(
        &bad_count,
        GgufErrorKind::LimitExceeded {
            field: "tensor count",
            value: u64::MAX,
            limit: GgufLimits::default().max_tensors,
        },
    );

    let mut truncated = base.clone();
    truncated.pop();
    assert_kind(
        &truncated,
        GgufErrorKind::TensorRangeOutOfBounds { tensor: "x".into() },
    );

    let mut trailing = base.clone();
    trailing.resize(trailing.len().next_multiple_of(32), 0);
    assert!(read_gguf(&trailing).is_ok(), "canonical trailing padding");
    trailing.push(0);
    assert_kind(&trailing, GgufErrorKind::TrailingData { bytes: 1 });

    let mut nonzero_header_padding = base.clone();
    nonzero_header_padding[57] = 1;
    assert_kind(
        &nonzero_header_padding,
        GgufErrorKind::InvalidPadding { section: "header" },
    );

    let mut nonzero_leading_tensor_padding = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "x",
            dimensions: &[1],
            kind: 0,
            offset: 32,
            data: &data,
        }],
        32,
    );
    let leading_data_offset = read_gguf(&nonzero_leading_tensor_padding)
        .unwrap()
        .data_offset();
    nonzero_leading_tensor_padding[leading_data_offset] = 1;
    assert_kind(
        &nonzero_leading_tensor_padding,
        GgufErrorKind::InvalidPadding { section: "tensor" },
    );

    let mut nonzero_trailing_padding = base.clone();
    nonzero_trailing_padding.push(1);
    assert_kind(
        &nonzero_trailing_padding,
        GgufErrorKind::InvalidPadding {
            section: "trailing",
        },
    );

    let mut bad_rank = base.clone();
    bad_rank[33..37].copy_from_slice(&0u32.to_le_bytes());
    assert_kind(
        &bad_rank,
        GgufErrorKind::InvalidRank {
            tensor: "x".into(),
            rank: 0,
        },
    );

    let mut excessive_rank = base.clone();
    excessive_rank[33..37].copy_from_slice(&5u32.to_le_bytes());
    assert_kind(
        &excessive_rank,
        GgufErrorKind::InvalidRank {
            tensor: "x".into(),
            rank: 5,
        },
    );

    let mut bad_dimension = base.clone();
    bad_dimension[37..45].copy_from_slice(&0u64.to_le_bytes());
    assert_kind(
        &bad_dimension,
        GgufErrorKind::InvalidDimension {
            tensor: "x".into(),
            axis: 0,
            value: 0,
        },
    );

    let mut unknown_type = base.clone();
    unknown_type[45..49].copy_from_slice(&99u32.to_le_bytes());
    assert_kind(&unknown_type, GgufErrorKind::UnknownTensorType(99));

    let mut misaligned = base.clone();
    misaligned[49..57].copy_from_slice(&1u64.to_le_bytes());
    assert_kind(
        &misaligned,
        GgufErrorKind::MisalignedTensorOffset {
            tensor: "x".into(),
            offset: 1,
            alignment: 32,
        },
    );

    let mut out_of_bounds = base.clone();
    out_of_bounds[49..57].copy_from_slice(&32u64.to_le_bytes());
    assert_kind(
        &out_of_bounds,
        GgufErrorKind::TensorRangeOutOfBounds { tensor: "x".into() },
    );

    let block_mismatch = fixture(
        3,
        &[],
        &[TensorFixture {
            name: "q",
            dimensions: &[31],
            kind: 2,
            offset: 0,
            data: &[],
        }],
        32,
    );
    assert_kind(
        &block_mismatch,
        GgufErrorKind::BlockElementMismatch {
            tensor: "q".into(),
            elements: 31,
            block_elements: 32,
        },
    );

    let duplicate_tensors = fixture(
        3,
        &[],
        &[
            TensorFixture {
                name: "x",
                dimensions: &[1],
                kind: 0,
                offset: 0,
                data: &data,
            },
            TensorFixture {
                name: "x",
                dimensions: &[1],
                kind: 0,
                offset: 32,
                data: &data,
            },
        ],
        32,
    );
    assert_kind(
        &duplicate_tensors,
        GgufErrorKind::DuplicateTensor("x".into()),
    );

    let overlapping = fixture(
        3,
        &[],
        &[
            TensorFixture {
                name: "a",
                dimensions: &[16],
                kind: 0,
                offset: 0,
                data: &[0; 64],
            },
            TensorFixture {
                name: "b",
                dimensions: &[1],
                kind: 0,
                offset: 32,
                data: &data,
            },
        ],
        32,
    );
    assert_kind(
        &overlapping,
        GgufErrorKind::OverlappingTensors {
            first: "a".into(),
            second: "b".into(),
        },
    );
}

#[test]
fn malformed_metadata_limits_types_and_alignment_fail_structurally() {
    let duplicate = fixture(3, &[metadata_u32("a", 1), metadata_u32("a", 2)], &[], 32);
    assert_kind(&duplicate, GgufErrorKind::DuplicateMetadata("a".into()));

    let invalid_key = fixture(3, &[metadata_u32("Bad.Key", 1)], &[], 32);
    assert_kind(
        &invalid_key,
        GgufErrorKind::InvalidMetadataKey("Bad.Key".into()),
    );

    let invalid_alignment = fixture(3, &[metadata_u32("general.alignment", 3)], &[], 32);
    assert_kind(&invalid_alignment, GgufErrorKind::InvalidAlignment(3));

    let wrong_alignment_type = fixture(3, &[metadata_string("general.alignment", "32")], &[], 32);
    assert_kind(
        &wrong_alignment_type,
        GgufErrorKind::InvalidAlignmentType(GgufMetadataType::String),
    );

    let valid_non_power_of_two_alignment = fixture(
        3,
        &[metadata_u32("general.alignment", 24)],
        &[TensorFixture {
            name: "x",
            dimensions: &[1],
            kind: 0,
            offset: 0,
            data: &1.0f32.to_le_bytes(),
        }],
        24,
    );
    assert_eq!(
        read_gguf(&valid_non_power_of_two_alignment)
            .unwrap()
            .alignment(),
        24
    );

    let invalid_bool = fixture(3, &[metadata_bool("flag", 2)], &[], 32);
    assert_kind(&invalid_bool, GgufErrorKind::InvalidBoolean(2));

    let mut unknown = Vec::new();
    unknown.extend_from_slice(b"GGUF");
    unknown.extend_from_slice(&3u32.to_le_bytes());
    unknown.extend_from_slice(&0u64.to_le_bytes());
    unknown.extend_from_slice(&1u64.to_le_bytes());
    push_string(&mut unknown, b"x");
    unknown.extend_from_slice(&99u32.to_le_bytes());
    assert_kind(&unknown, GgufErrorKind::UnknownMetadataType(99));

    let invalid_utf8 = {
        let mut entry = Vec::new();
        push_string(&mut entry, &[0xff]);
        entry.extend_from_slice(&4u32.to_le_bytes());
        entry.extend_from_slice(&1u32.to_le_bytes());
        fixture(3, &[entry], &[], 32)
    };
    assert_kind(
        &invalid_utf8,
        GgufErrorKind::InvalidUtf8 {
            field: "metadata key",
        },
    );

    let long_string = fixture(3, &[metadata_string("long", "12345")], &[], 32);
    let limits = GgufLimits {
        max_string_bytes: 4,
        ..GgufLimits::default()
    };
    assert_eq!(
        read_gguf_with_limits(&long_string, limits)
            .unwrap_err()
            .kind(),
        &GgufErrorKind::LimitExceeded {
            field: "metadata string",
            value: 5,
            limit: 4,
        }
    );

    let array = fixture(3, &[metadata_u32_array("items", &[1, 2])], &[], 32);
    let limits = GgufLimits {
        max_array_elements: 1,
        ..GgufLimits::default()
    };
    assert_eq!(
        read_gguf_with_limits(&array, limits).unwrap_err().kind(),
        &GgufErrorKind::LimitExceeded {
            field: "array elements",
            value: 2,
            limit: 1,
        }
    );
}
