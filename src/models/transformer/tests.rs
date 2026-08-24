use super::*;
use crate::{Storage, gguf::GgufErrorKind};

const SCHEMA: LlamaDecoderSchema = LlamaDecoderSchema {
    vocab_size: 5,
    embedding_dim: 4,
    hidden_dim: 6,
    query_heads: 2,
    kv_heads: 1,
    head_dim: 2,
    rope_dim: 2,
};

#[derive(Clone)]
struct FixtureTensor {
    name: &'static str,
    shape: Vec<usize>,
    kind: u32,
    bytes: Vec<u8>,
}

fn f32_tensor(name: &'static str, shape: &[usize]) -> FixtureTensor {
    let elements = shape.iter().product::<usize>();
    let mut bytes = Vec::with_capacity(elements * 4);
    for index in 0..elements {
        bytes.extend_from_slice(&((index as f32 + 1.0) / 16.0).to_le_bytes());
    }
    FixtureTensor {
        name,
        shape: shape.to_vec(),
        kind: 0,
        bytes,
    }
}

fn state_tensors(explicit_output: bool) -> Vec<FixtureTensor> {
    let mut tensors = vec![
        f32_tensor(TOKEN_EMBEDDING, &[5, 4]),
        f32_tensor(OUTPUT_NORM, &[4]),
        f32_tensor("blk.0.attn_norm.weight", &[4]),
        f32_tensor("blk.0.attn_q.weight", &[4, 4]),
        f32_tensor("blk.0.attn_k.weight", &[2, 4]),
        f32_tensor("blk.0.attn_v.weight", &[2, 4]),
        f32_tensor("blk.0.attn_output.weight", &[4, 4]),
        f32_tensor("blk.0.ffn_norm.weight", &[4]),
        f32_tensor("blk.0.ffn_gate.weight", &[6, 4]),
        f32_tensor("blk.0.ffn_up.weight", &[6, 4]),
        f32_tensor("blk.0.ffn_down.weight", &[4, 6]),
        f32_tensor(ROPE_FREQS, &[1]),
    ];
    if explicit_output {
        tensors.push(f32_tensor(OUTPUT_WEIGHT, &[5, 4]));
    }
    tensors
}

fn push_string(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn align(value: usize, alignment: usize) -> usize {
    value.next_multiple_of(alignment)
}

fn gguf_fixture(tensors: &[FixtureTensor]) -> Vec<u8> {
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut data_len = 0;
    for tensor in tensors {
        data_len = align(data_len, 32);
        offsets.push(data_len);
        data_len += tensor.bytes.len();
    }

    let mut output = b"GGUF".to_vec();
    output.extend_from_slice(&3u32.to_le_bytes());
    output.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    output.extend_from_slice(&0u64.to_le_bytes());
    for (tensor, offset) in tensors.iter().zip(&offsets) {
        push_string(&mut output, tensor.name);
        output.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dimension in tensor.shape.iter().rev() {
            output.extend_from_slice(&(*dimension as u64).to_le_bytes());
        }
        output.extend_from_slice(&tensor.kind.to_le_bytes());
        output.extend_from_slice(&(*offset as u64).to_le_bytes());
    }
    output.resize(align(output.len(), 32), 0);
    let data_offset = output.len();
    for (tensor, offset) in tensors.iter().zip(offsets) {
        output.resize(data_offset + offset, 0);
        output.extend_from_slice(&tensor.bytes);
    }
    output
}

#[test]
fn binds_exact_state_and_resolves_tied_or_explicit_output() {
    let tied_bytes = gguf_fixture(&state_tensors(false));
    let tied_file = crate::gguf::read_gguf(&tied_bytes).unwrap();
    let tied = SCHEMA.bind(&tied_file).unwrap();
    assert_eq!(
        tied.output_binding(),
        LlamaOutputBinding::TiedToTokenEmbedding
    );
    assert!(!tied.tensors().contains_key(OUTPUT_WEIGHT));
    assert_eq!(tied.output_weight(), &tied.tensors()[TOKEN_EMBEDDING]);

    let explicit_bytes = gguf_fixture(&state_tensors(true));
    let explicit_file = crate::gguf::read_gguf(&explicit_bytes).unwrap();
    let explicit = SCHEMA.bind(&explicit_file).unwrap();
    assert_eq!(explicit.output_binding(), LlamaOutputBinding::Explicit);
    assert_eq!(explicit.output_weight(), &explicit.tensors()[OUTPUT_WEIGHT]);
}

#[test]
fn gguf_bound_state_executes_the_public_decoder_path() {
    let bytes = gguf_fixture(&state_tensors(false));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let state = SCHEMA.bind(&file).unwrap();
    let config = LlamaDecoderConfig::new(SCHEMA, 4, 1e-5, 10.0).unwrap();
    let decoder = LlamaDecoder::new(config, state).unwrap();
    let output = decoder.forward(&[1, 2]).unwrap();
    assert_eq!(output.logits().shape().dims(), &[2, 5]);
    assert!(
        output
            .logits()
            .values()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn rejects_missing_unexpected_and_misshaped_names_without_discovery() {
    let mut missing = state_tensors(false);
    missing.retain(|tensor| tensor.name != "blk.0.attn_k.weight");
    let bytes = gguf_fixture(&missing);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        SCHEMA.bind(&file).unwrap_err(),
        LlamaStateError::MissingTensor("blk.0.attn_k.weight")
    );

    let mut unexpected = state_tensors(false);
    unexpected.push(f32_tensor("blk.1.attn_norm.weight", &[4]));
    let bytes = gguf_fixture(&unexpected);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        SCHEMA.bind(&file).unwrap_err(),
        LlamaStateError::UnexpectedTensor("blk.1.attn_norm.weight".to_owned())
    );

    let mut misshaped = state_tensors(false);
    let query = misshaped
        .iter_mut()
        .find(|tensor| tensor.name == "blk.0.attn_q.weight")
        .unwrap();
    *query = f32_tensor("blk.0.attn_q.weight", &[2, 8]);
    let bytes = gguf_fixture(&misshaped);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        SCHEMA.bind(&file).unwrap_err(),
        LlamaStateError::ShapeMismatch {
            tensor: "blk.0.attn_q.weight",
            expected: vec![4, 4],
            actual: vec![2, 8],
        }
    );
}

#[test]
fn materialization_is_atomic_and_dtype_validation_is_explicit() {
    let mut tensors = state_tensors(false);
    tensors.push(FixtureTensor {
        name: "unsupported.weight",
        shape: vec![32],
        kind: 3,
        bytes: vec![0; 20],
    });
    let bytes = gguf_fixture(&tensors);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        SCHEMA.bind(&file).unwrap_err(),
        LlamaStateError::Gguf(error)
            if matches!(error.kind(), GgufErrorKind::QuantizedMaterialization { tensor, .. } if tensor == "unsupported.weight")
    ));

    let bytes = gguf_fixture(&state_tensors(false));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let mut materialized = file.materialize_state_f32().unwrap();
    materialized.insert(
        OUTPUT_NORM.to_owned(),
        TensorData::from_storage([4], Storage::I32(vec![1, 2, 3, 4])).unwrap(),
    );
    assert_eq!(
        SCHEMA.bind_materialized(materialized).unwrap_err(),
        LlamaStateError::DTypeMismatch {
            tensor: OUTPUT_NORM,
            actual: DType::I32,
        }
    );
}

#[test]
fn schema_configuration_rejects_zero_odd_rope_and_overflow() {
    assert_eq!(
        LlamaDecoderSchema::new(0, 4, 6, 2, 1, 2, 2).unwrap_err(),
        LlamaStateError::InvalidConfig {
            field: "vocab_size"
        }
    );
    assert_eq!(
        LlamaDecoderSchema::new(5, 4, 6, 2, 1, 2, 1).unwrap_err(),
        LlamaStateError::InvalidConfig { field: "rope_dim" }
    );
    assert_eq!(
        LlamaDecoderSchema::new(5, 4, 6, 3, 2, 2, 2).unwrap_err(),
        LlamaStateError::InvalidConfig {
            field: "query_heads"
        }
    );
    assert_eq!(
        LlamaDecoderSchema::new(5, 4, 6, usize::MAX, 1, 2, 2).unwrap_err(),
        LlamaStateError::ProjectionOverflow
    );
}
