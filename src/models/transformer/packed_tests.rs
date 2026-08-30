use super::{
    LlamaBatchGenerator, LlamaBatchNativeCache, LlamaBatchNativeGenerator, LlamaBatchSampling,
    LlamaLinearWeight, LlamaModel, LlamaModelError, LlamaNativeCache, LlamaNativeGenerator,
    LlamaNativeStageKind, LlamaSampling,
    serving::{
        LlamaRequestStatus, LlamaServingConfig, LlamaServingGenerationConfig, LlamaServingSampling,
        LlamaServingScheduler,
    },
};
use crate::{
    GgmlLayout, GgmlType, QuantizedTensorData, Shape, TensorData, tokenizer::SimpleTokenizer,
};
use std::collections::{BTreeMap, BTreeSet};

const VOCAB: usize = 12;
const DIM: usize = 256;
const HIDDEN: usize = 256;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 128;
const ROPE_DIM: usize = 64;
const LAYERS: usize = 2;

#[derive(Clone)]
enum FixtureTensor {
    Dense(TensorData),
    Packed(QuantizedTensorData),
    Raw {
        shape: Shape,
        kind: GgmlType,
        bytes: Vec<u8>,
    },
}

impl FixtureTensor {
    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(value) => value.shape(),
            Self::Packed(value) => &value.descriptor().logical_shape,
            Self::Raw { shape, .. } => shape,
        }
    }

    fn kind(&self) -> GgmlType {
        match self {
            Self::Dense(_) => GgmlType::F32,
            Self::Packed(value) => value.descriptor().ggml_type,
            Self::Raw { kind, .. } => *kind,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Dense(value) => value
                .values()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            Self::Packed(value) => value.bytes().to_vec(),
            Self::Raw { bytes, .. } => bytes.clone(),
        }
    }
}

fn dense(shape: impl Into<Shape>, salt: usize, scale: f32) -> TensorData {
    let shape = shape.into();
    TensorData::new(
        shape.clone(),
        (0..shape.numel().unwrap())
            .map(|index| (((index * 17 + salt * 13) % 31) as f32 - 15.0) * scale)
            .collect(),
    )
    .unwrap()
}

fn block(kind: GgmlType) -> Vec<u8> {
    match kind {
        GgmlType::Q4_0 => {
            let mut out = 0x3000u16.to_le_bytes().to_vec();
            out.extend(std::iter::repeat_n(0x3d, 16));
            out
        }
        GgmlType::Q8_0 => {
            let mut out = 0x2800u16.to_le_bytes().to_vec();
            out.extend((-16i8..16).map(|value| value as u8));
            out
        }
        GgmlType::Q4K => {
            let scales = [1u8, 2, 3, 4, 17, 33, 49, 63];
            let mins = [0u8, 1, 2, 3, 16, 32, 48, 62];
            let mut packed = [0u8; 12];
            for lane in 0..4 {
                packed[lane] = scales[lane] | ((scales[4 + lane] >> 4) << 6);
                packed[4 + lane] = mins[lane] | ((mins[4 + lane] >> 4) << 6);
                packed[8 + lane] = (scales[4 + lane] & 15) | ((mins[4 + lane] & 15) << 4);
            }
            let mut out = Vec::with_capacity(144);
            out.extend(0x2800u16.to_le_bytes());
            out.extend(0x2400u16.to_le_bytes());
            out.extend(packed);
            for pair in 0..4 {
                let low = (pair * 2 + 1) as u8;
                let high = (pair * 2 + 2) as u8;
                out.extend(std::iter::repeat_n(low | (high << 4), 32));
            }
            out
        }
        GgmlType::Q6K => {
            let mut out = vec![0u8; 210];
            for index in 0..256 {
                let raw = ((index * 29 + 7) & 63) as u8;
                let half = index / 128;
                let within = index % 128;
                out[half * 64 + within % 64] |= (raw & 15) << ((within / 64) * 4);
                out[128 + half * 32 + within % 32] |= ((raw >> 4) & 3) << ((within / 32) * 2);
            }
            for (index, scale) in (-8i8..8).enumerate() {
                out[192 + index] = scale as u8;
            }
            out[208..].copy_from_slice(&0x1800u16.to_le_bytes());
            out
        }
        _ => unreachable!(),
    }
}

fn packed(kind: GgmlType, shape: [usize; 2]) -> QuantizedTensorData {
    let GgmlLayout::Quantized { block_elements, .. } = kind.layout() else {
        unreachable!()
    };
    let one = block(kind);
    QuantizedTensorData::new(
        kind,
        Shape::from(shape),
        std::iter::repeat_n(one, shape[0] * shape[1] / block_elements)
            .flatten()
            .collect(),
    )
    .unwrap()
}

fn packed_state() -> BTreeMap<String, FixtureTensor> {
    let mut state = BTreeMap::from([
        (
            super::TOKEN_EMBEDDING.to_owned(),
            FixtureTensor::Packed(packed(GgmlType::Q4K, [VOCAB, DIM])),
        ),
        (
            super::OUTPUT_NORM.to_owned(),
            FixtureTensor::Dense(TensorData::new([DIM], vec![1.0; DIM]).unwrap()),
        ),
    ]);
    let formats = [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K];
    let mut format_index = 0;
    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            state.insert(
                format!("{prefix}.{suffix}"),
                FixtureTensor::Dense(TensorData::new([DIM], vec![1.0; DIM]).unwrap()),
            );
        }
        for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            state.insert(
                format!("{prefix}.{suffix}"),
                FixtureTensor::Dense(TensorData::new([HEAD_DIM], vec![1.0; HEAD_DIM]).unwrap()),
            );
        }
        for (suffix, width) in [
            ("attn_q.bias", HEADS * HEAD_DIM),
            ("attn_k.bias", KV_HEADS * HEAD_DIM),
            ("attn_v.bias", KV_HEADS * HEAD_DIM),
        ] {
            state.insert(
                format!("{prefix}.{suffix}"),
                FixtureTensor::Dense(dense([width], layer + width, 0.0002)),
            );
        }
        for (suffix, shape) in [
            ("attn_q.weight", [HEADS * HEAD_DIM, DIM]),
            ("attn_k.weight", [KV_HEADS * HEAD_DIM, DIM]),
            ("attn_v.weight", [KV_HEADS * HEAD_DIM, DIM]),
            ("attn_output.weight", [DIM, HEADS * HEAD_DIM]),
            ("ffn_gate.weight", [HIDDEN, DIM]),
            ("ffn_up.weight", [HIDDEN, DIM]),
            ("ffn_down.weight", [DIM, HIDDEN]),
        ] {
            let kind = formats[format_index % formats.len()];
            format_index += 1;
            state.insert(
                format!("{prefix}.{suffix}"),
                FixtureTensor::Packed(packed(kind, shape)),
            );
        }
    }
    state
}

fn dense_control(state: &BTreeMap<String, FixtureTensor>) -> BTreeMap<String, FixtureTensor> {
    state
        .iter()
        .map(|(name, value)| {
            let value = match value {
                FixtureTensor::Dense(value) => FixtureTensor::Dense(value.clone()),
                FixtureTensor::Packed(value) => {
                    FixtureTensor::Dense(value.dequantize_f32().unwrap())
                }
                FixtureTensor::Raw { .. } => panic!("raw malformed fixture has no control"),
            };
            (name.clone(), value)
        })
        .collect()
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn metadata(out: &mut Vec<u8>) {
    static TOKENS: [&str; VOCAB] = [
        "<s>",
        "</s>",
        "<|im_end|>",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
    ];
    static TYPES: [i32; VOCAB] = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let strings = [
        ("general.architecture", "llama"),
        ("tokenizer.ggml.pre", "llama3"),
    ];
    for (key, value) in strings {
        push_string(out, key);
        out.extend_from_slice(&8u32.to_le_bytes());
        push_string(out, value);
    }
    push_string(out, "tokenizer.ggml.tokens");
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&(TOKENS.len() as u64).to_le_bytes());
    for token in TOKENS {
        push_string(out, token);
    }
    push_string(out, "tokenizer.ggml.token_type");
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&5u32.to_le_bytes());
    out.extend_from_slice(&(TYPES.len() as u64).to_le_bytes());
    for token_type in TYPES {
        out.extend_from_slice(&token_type.to_le_bytes());
    }
    for (key, value) in [
        ("tokenizer.ggml.bos_token_id", 0),
        ("tokenizer.ggml.eos_token_id", 1),
        ("tokenizer.ggml.eot_token_id", 2),
        ("llama.block_count", LAYERS as u32),
        ("llama.embedding_length", DIM as u32),
        ("llama.feed_forward_length", HIDDEN as u32),
        ("llama.attention.head_count", HEADS as u32),
        ("llama.attention.head_count_kv", KV_HEADS as u32),
        ("llama.attention.key_length", HEAD_DIM as u32),
        ("llama.attention.value_length", HEAD_DIM as u32),
        ("llama.rope.dimension_count", ROPE_DIM as u32),
        ("llama.context_length", 12),
    ] {
        push_string(out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
    push_string(out, "tokenizer.ggml.add_bos_token");
    out.extend_from_slice(&7u32.to_le_bytes());
    out.push(0);
    for (key, value) in [
        ("llama.attention.layer_norm_rms_epsilon", 1e-5f32),
        ("llama.rope.freq_base", 10.0f32),
    ] {
        push_string(out, key);
        out.extend_from_slice(&6u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn fixture(state: &BTreeMap<String, FixtureTensor>) -> Vec<u8> {
    let metadata_count = 19u64;
    let mut offsets = Vec::with_capacity(state.len());
    let mut data_len = 0usize;
    for tensor in state.values() {
        data_len = data_len.next_multiple_of(32);
        offsets.push(data_len);
        data_len += tensor.bytes().len();
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(state.len() as u64).to_le_bytes());
    out.extend_from_slice(&metadata_count.to_le_bytes());
    metadata(&mut out);
    for ((name, tensor), offset) in state.iter().zip(&offsets) {
        push_string(&mut out, name);
        out.extend_from_slice(&(tensor.shape().rank() as u32).to_le_bytes());
        for dimension in tensor.shape().dims().iter().rev() {
            out.extend_from_slice(&(*dimension as u64).to_le_bytes());
        }
        out.extend_from_slice(&tensor.kind().raw().to_le_bytes());
        out.extend_from_slice(&(*offset as u64).to_le_bytes());
    }
    out.resize(out.len().next_multiple_of(32), 0);
    let data_offset = out.len();
    for ((_, tensor), offset) in state.iter().zip(offsets) {
        out.resize(data_offset + offset, 0);
        out.extend(tensor.bytes());
    }
    out
}

fn models() -> (LlamaModel, SimpleTokenizer, LlamaModel, SimpleTokenizer) {
    let state = packed_state();
    let packed_bytes = fixture(&state);
    let packed_file = crate::gguf::read_gguf(&packed_bytes).unwrap();
    let (packed_model, packed_tokenizer) = LlamaModel::from_gguf(&packed_file).unwrap();
    let dense_bytes = fixture(&dense_control(&state));
    let dense_file = crate::gguf::read_gguf(&dense_bytes).unwrap();
    let (dense_model, dense_tokenizer) = LlamaModel::from_gguf(&dense_file).unwrap();
    (packed_model, packed_tokenizer, dense_model, dense_tokenizer)
}

fn assert_close(actual: &TensorData, expected: &TensorData) {
    assert_eq!(actual.shape(), expected.shape());
    // Packed kernels partition dequantized products differently from the
    // independently materialized dense control. Keep the allowance below one
    // 2^-10 F32 accumulation step at this fixture's activation scale.
    for (index, (actual, expected)) in actual.values().iter().zip(expected.values()).enumerate() {
        let difference = (actual - expected).abs();
        assert!(
            difference <= 1e-3,
            "index {index}: {actual} != {expected}, difference={difference}"
        );
    }
}

#[test]
fn mixed_packed_two_layer_native_matches_independently_dense_control_and_caches() {
    let (packed, _, dense, _) = models();
    assert_eq!(packed.linear_weights().len(), LAYERS * 7);
    assert!(
        packed
            .linear_weights()
            .values()
            .all(|weight| matches!(weight, LlamaLinearWeight::Quantized(_)))
    );
    let tokens = [3, 4, 5];
    let control = dense.forward(&tokens).unwrap();
    let direct = packed.forward(&tokens).unwrap();
    assert_close(&direct, &control);
    let native = packed.forward_native(&tokens).unwrap();
    assert_close(native.logits(), &control);
    let formats = native
        .trace()
        .iter()
        .filter_map(|stage| match stage.kind {
            LlamaNativeStageKind::QuantizedMatmul { format, .. } => Some(format),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        formats,
        BTreeSet::from([GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K])
    );
    assert_eq!(
        native
            .trace()
            .iter()
            .filter(|stage| matches!(stage.kind, LlamaNativeStageKind::QuantizedMatmul { .. }))
            .count(),
        LAYERS * 7 + 1
    );
    assert_eq!(
        native
            .trace()
            .iter()
            .filter(|stage| matches!(stage.kind, LlamaNativeStageKind::QuantizedRowGather { .. }))
            .count(),
        1
    );
    assert!(native.trace().iter().all(|stage| match stage.kind {
        LlamaNativeStageKind::QuantizedMatmul { .. }
        | LlamaNativeStageKind::QuantizedRowGather { .. } => {
            stage.items.iter().all(|item| item.packed_weight_bytes > 0)
        }
        LlamaNativeStageKind::NativeSchedule | LlamaNativeStageKind::Movement(_) => {
            stage.items.iter().all(|item| item.packed_weight_bytes == 0)
        }
    }));

    let mut token_cache = LlamaNativeCache::new(packed.config().clone());
    let mut incremental = Vec::new();
    for token in tokens {
        incremental.extend_from_slice(
            token_cache
                .forward(&packed, &[token])
                .unwrap()
                .logits()
                .values(),
        );
    }
    assert_close(
        &TensorData::new([tokens.len(), VOCAB], incremental).unwrap(),
        &control,
    );
    let mut chunk_cache = LlamaNativeCache::new(packed.config().clone());
    let first = chunk_cache.forward(&packed, &[3, 4]).unwrap();
    let second = chunk_cache.forward(&packed, &[5]).unwrap();
    let mut chunked = first.logits().values().to_vec();
    chunked.extend_from_slice(second.logits().values());
    assert_close(
        &TensorData::new([tokens.len(), VOCAB], chunked).unwrap(),
        &control,
    );

    let rows = vec![tokens.to_vec(), vec![6, 3]];
    let mut batch = LlamaBatchNativeCache::new(packed.config().clone(), 2).unwrap();
    let actual = batch.forward(&packed, &rows).unwrap();
    let expected = dense.forward_batch(&rows).unwrap();
    for (actual, expected) in actual.rows().iter().zip(expected) {
        assert_close(actual, &expected);
    }
}

#[test]
fn packed_generation_serving_identity_rollback_and_malformed_binding_are_explicit() {
    let (packed_model, tokenizer, dense, dense_tokenizer) = models();
    let prompt = [3, 4];
    let direct = super::LlamaGenerator::new(&dense, &dense_tokenizer)
        .generate_ids(&prompt, 2, LlamaSampling::Greedy)
        .unwrap();
    let native = LlamaNativeGenerator::new(&packed_model, &tokenizer)
        .generate_ids(&prompt, 2, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(native.generated_ids(), direct.generated_ids());

    let uniforms = vec![0.07; 2 * VOCAB];
    let direct_tape = super::LlamaGenerator::new(&dense, &dense_tokenizer)
        .generate_ids(
            &prompt,
            2,
            LlamaSampling::GumbelMax {
                temperature: 0.8,
                uniforms: &uniforms,
            },
        )
        .unwrap();
    let native_tape = LlamaNativeGenerator::new(&packed_model, &tokenizer)
        .generate_ids(
            &prompt,
            2,
            LlamaSampling::GumbelMax {
                temperature: 0.8,
                uniforms: &uniforms,
            },
        )
        .unwrap();
    assert_eq!(native_tape.generated_ids(), direct_tape.generated_ids());

    let batch_uniforms = vec![0.13; 2 * 2 * VOCAB];
    let dense_batch = LlamaBatchGenerator::new(&dense, &dense_tokenizer, 2)
        .unwrap()
        .generate_ids(
            &[vec![3, 4], vec![3, 5, 6]],
            2,
            LlamaBatchSampling::GumbelMax {
                temperature: 0.7,
                uniforms: &batch_uniforms,
            },
        )
        .unwrap();
    let packed_batch = LlamaBatchNativeGenerator::new(&packed_model, &tokenizer, 2)
        .unwrap()
        .generate_ids(
            &[vec![3, 4], vec![3, 5, 6]],
            2,
            LlamaBatchSampling::GumbelMax {
                temperature: 0.7,
                uniforms: &batch_uniforms,
            },
        )
        .unwrap();
    for (actual, expected) in packed_batch.sequences().iter().zip(dense_batch.sequences()) {
        assert_eq!(actual.generated_ids(), expected.generated_ids());
    }

    let mut scheduler = LlamaServingScheduler::new(
        &packed_model,
        &tokenizer,
        LlamaServingConfig::new(2, 8, 1 << 24).unwrap(),
    );
    let seed = scheduler
        .submit_ids(
            vec![3, 4],
            LlamaServingGenerationConfig::new(1, LlamaServingSampling::Greedy),
        )
        .unwrap();
    while scheduler.pending() != 0 {
        scheduler.step().unwrap();
    }
    assert!(scheduler.result(seed).is_some());
    let retry_a = scheduler
        .submit_ids(
            vec![3, 4, 5],
            LlamaServingGenerationConfig::new(1, LlamaServingSampling::Greedy),
        )
        .unwrap();
    let retry_b = scheduler
        .submit_ids(
            vec![3, 4, 6],
            LlamaServingGenerationConfig::new(
                1,
                LlamaServingSampling::GumbelMax {
                    temperature: 0.7,
                    uniforms: vec![0.13; VOCAB],
                },
            ),
        )
        .unwrap();
    let before = scheduler.prefix_stats();
    scheduler.inject_stage_failure(Some(0));
    assert!(scheduler.step().is_err());
    assert_eq!(scheduler.status(retry_a), Some(LlamaRequestStatus::Queued));
    assert_eq!(scheduler.status(retry_b), Some(LlamaRequestStatus::Queued));
    assert_eq!(scheduler.prefix_stats(), before);
    scheduler.inject_stage_failure(None);
    while scheduler.pending() != 0 {
        scheduler.step().unwrap();
    }
    assert!(scheduler.prefix_stats().hits >= 1);

    let mut changed_state = packed_state();
    let FixtureTensor::Packed(changed) = &changed_state[super::TOKEN_EMBEDDING] else {
        unreachable!()
    };
    let mut bytes = changed.bytes().to_vec();
    bytes[2] ^= 1;
    changed_state.insert(
        super::TOKEN_EMBEDDING.to_owned(),
        FixtureTensor::Packed(
            QuantizedTensorData::new(
                changed.descriptor().ggml_type,
                changed.descriptor().logical_shape.clone(),
                bytes,
            )
            .unwrap(),
        ),
    );
    let changed_bytes = fixture(&changed_state);
    let changed_file = crate::gguf::read_gguf(&changed_bytes).unwrap();
    let (changed, changed_tokenizer) = LlamaModel::from_gguf(&changed_file).unwrap();
    let generation = scheduler.prefix_stats().generation;
    assert!(scheduler.rebind(&changed, &changed_tokenizer).unwrap());
    assert_eq!(scheduler.prefix_stats().entries, 0);
    assert_eq!(scheduler.prefix_stats().generation, generation + 1);

    for kind in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
        let mut tied_state = packed_state();
        tied_state.insert(
            super::TOKEN_EMBEDDING.to_owned(),
            FixtureTensor::Packed(packed(kind, [VOCAB, DIM])),
        );
        let bytes = fixture(&tied_state);
        let file = crate::gguf::read_gguf(&bytes).unwrap();
        let (tied, _) = LlamaModel::from_gguf(&file).unwrap();
        assert_eq!(
            tied.output_binding(),
            super::LlamaOutputBinding::TiedToTokenEmbedding
        );
        assert_eq!(tied.embedding_weight().quantized_type(), Some(kind));
        assert!(!tied.dense_state().contains_key(super::TOKEN_EMBEDDING));
        assert!(!tied.linear_weights().contains_key(super::TOKEN_EMBEDDING));

        let control_bytes = fixture(&dense_control(&tied_state));
        let control_file = crate::gguf::read_gguf(&control_bytes).unwrap();
        let (control, _) = LlamaModel::from_gguf(&control_file).unwrap();
        let tokens = [3, 3, 5];
        let expected = control.forward(&tokens).unwrap();
        let plan = tied.plan(&tokens).unwrap();
        let packed_input = plan
            .packed_logits_input
            .expect("tied packed output must use the blockwise direct plan");
        assert_eq!(
            plan.graph.node(packed_input).unwrap().shape.dims(),
            &[3, DIM]
        );
        let output = &plan.quantized_linears[&plan.logits.index()];
        let LlamaLinearWeight::Quantized(embedding) = tied.embedding_weight() else {
            unreachable!()
        };
        assert_eq!(
            output.weight.descriptor().identity,
            embedding.descriptor().identity
        );
        assert_close(&plan.execute().unwrap(), &expected);
        let native = tied.forward_native(&tokens).unwrap();
        assert_close(native.logits(), &expected);
        assert!(native.trace().iter().any(|stage| matches!(
            &stage.kind,
            LlamaNativeStageKind::QuantizedMatmul { tensor, format }
                if tensor == super::TOKEN_EMBEDDING && *format == kind
        )));
    }

    let mut wrong_embedding = packed_state();
    wrong_embedding.insert(
        super::TOKEN_EMBEDDING.to_owned(),
        FixtureTensor::Packed(packed(GgmlType::Q4_0, [VOCAB, DIM * 2])),
    );
    let bytes = fixture(&wrong_embedding);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        LlamaModel::from_gguf(&file),
        Err(LlamaModelError::ShapeMismatch { tensor, .. })
            if tensor == super::TOKEN_EMBEDDING
    ));

    let mut unsupported_embedding = packed_state();
    unsupported_embedding.insert(
        super::TOKEN_EMBEDDING.to_owned(),
        FixtureTensor::Raw {
            shape: Shape::from([VOCAB, DIM]),
            kind: GgmlType::Q5_0,
            bytes: vec![0; VOCAB * (DIM / 32) * 22],
        },
    );
    let bytes = fixture(&unsupported_embedding);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        LlamaModel::from_gguf(&file),
        Err(LlamaModelError::State(_))
    ));

    let mut malformed_embedding = packed_state();
    let mut malformed_bytes = vec![0; VOCAB * (DIM / 32) * 18];
    malformed_bytes[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    malformed_embedding.insert(
        super::TOKEN_EMBEDDING.to_owned(),
        FixtureTensor::Raw {
            shape: Shape::from([VOCAB, DIM]),
            kind: GgmlType::Q4_0,
            bytes: malformed_bytes,
        },
    );
    let bytes = fixture(&malformed_embedding);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        LlamaModel::from_gguf(&file),
        Err(LlamaModelError::State(_))
    ));

    let mut wrong_orientation = packed_state();
    wrong_orientation.insert(
        "blk.0.attn_k.weight".to_owned(),
        FixtureTensor::Packed(packed(GgmlType::Q4_0, [DIM, KV_HEADS * HEAD_DIM])),
    );
    let bytes = fixture(&wrong_orientation);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        LlamaModel::from_gguf(&file),
        Err(LlamaModelError::ShapeMismatch { tensor, .. }) if tensor == "blk.0.attn_k.weight"
    ));

    let mut unsupported = packed_state();
    unsupported.insert(
        "blk.0.attn_q.weight".to_owned(),
        FixtureTensor::Raw {
            shape: Shape::from([HEADS * HEAD_DIM, DIM]),
            kind: GgmlType::Q5_0,
            bytes: vec![0; HEADS * HEAD_DIM * (DIM / 32) * 22],
        },
    );
    let bytes = fixture(&unsupported);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        LlamaModel::from_gguf(&file),
        Err(LlamaModelError::State(_))
    ));

    let mut malformed_block = packed_state();
    let mut malformed_bytes = vec![0; HEADS * HEAD_DIM * (DIM / 32) * 18];
    malformed_bytes[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    malformed_block.insert(
        "blk.0.attn_q.weight".to_owned(),
        FixtureTensor::Raw {
            shape: Shape::from([HEADS * HEAD_DIM, DIM]),
            kind: GgmlType::Q4_0,
            bytes: malformed_bytes,
        },
    );
    let bytes = fixture(&malformed_block);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        LlamaModel::from_gguf(&file),
        Err(LlamaModelError::State(_))
    ));

    let mut missing = packed_state();
    missing.remove("blk.1.ffn_down.weight");
    let bytes = fixture(&missing);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::MissingTensor("blk.1.ffn_down.weight".to_owned())
    );
}
