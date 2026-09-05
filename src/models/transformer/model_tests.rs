use super::*;
use crate::{DType, Op, TensorData, tokenizer::SimpleTokenizer};
use std::collections::BTreeMap;

pub(super) const VOCAB: usize = 12;
const DIM: usize = 8;
const HIDDEN: usize = 10;
const QUERY_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const ROPE_DIM: usize = 4;
const LAYERS: usize = 2;
const EPS: f32 = 1e-5;
const THETA: f64 = 10.0;

fn values(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * 17 + salt * 11) % 29) as f32 - 14.0) * scale)
        .collect()
}

fn tensor(shape: &[usize], salt: usize, scale: f32) -> TensorData {
    TensorData::new(shape.to_vec(), values(shape.iter().product(), salt, scale)).unwrap()
}

fn fixed_state() -> BTreeMap<String, TensorData> {
    let mut state = BTreeMap::from([
        (TOKEN_EMBEDDING.to_owned(), tensor(&[VOCAB, DIM], 1, 0.018)),
        (
            OUTPUT_NORM.to_owned(),
            TensorData::new([DIM], (0..DIM).map(|i| 0.92 + i as f32 * 0.015).collect()).unwrap(),
        ),
    ]);
    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        let salt = layer * 9;
        state.insert(
            format!("{prefix}.attn_norm.weight"),
            TensorData::new(
                [DIM],
                (0..DIM)
                    .map(|i| 1.03 - i as f32 * 0.01 + layer as f32 * 0.005)
                    .collect(),
            )
            .unwrap(),
        );
        state.insert(
            format!("{prefix}.attn_q.weight"),
            tensor(&[QUERY_HEADS * HEAD_DIM, DIM], salt + 2, 0.012),
        );
        state.insert(
            format!("{prefix}.attn_k.weight"),
            tensor(&[KV_HEADS * HEAD_DIM, DIM], salt + 3, 0.014),
        );
        state.insert(
            format!("{prefix}.attn_v.weight"),
            tensor(&[KV_HEADS * HEAD_DIM, DIM], salt + 4, 0.013),
        );
        state.insert(
            format!("{prefix}.attn_output.weight"),
            tensor(&[DIM, QUERY_HEADS * HEAD_DIM], salt + 5, 0.011),
        );
        state.insert(
            format!("{prefix}.ffn_norm.weight"),
            TensorData::new(
                [DIM],
                (0..DIM)
                    .map(|i| 0.97 + i as f32 * 0.008 + layer as f32 * 0.004)
                    .collect(),
            )
            .unwrap(),
        );
        state.insert(
            format!("{prefix}.ffn_gate.weight"),
            tensor(&[HIDDEN, DIM], salt + 6, 0.010),
        );
        state.insert(
            format!("{prefix}.ffn_up.weight"),
            tensor(&[HIDDEN, DIM], salt + 7, 0.009),
        );
        state.insert(
            format!("{prefix}.ffn_down.weight"),
            tensor(&[DIM, HIDDEN], salt + 8, 0.010),
        );
    }
    state
}

#[derive(Clone)]
enum Metadata<'a> {
    String(&'a str),
    U32(u32),
    F32(f32),
    Bool(bool),
    Strings(&'a [&'a str]),
    I32s(&'a [i32]),
}

fn push_string(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn fixture(state: &BTreeMap<String, TensorData>, metadata: &[(&str, Metadata<'_>)]) -> Vec<u8> {
    let mut offsets = Vec::with_capacity(state.len());
    let mut data_len = 0usize;
    for tensor in state.values() {
        data_len = data_len.next_multiple_of(32);
        offsets.push(data_len);
        data_len += tensor.len() * 4;
    }
    let mut output = b"GGUF".to_vec();
    output.extend_from_slice(&3u32.to_le_bytes());
    output.extend_from_slice(&(state.len() as u64).to_le_bytes());
    output.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        push_string(&mut output, key);
        match value {
            Metadata::String(value) => {
                output.extend_from_slice(&8u32.to_le_bytes());
                push_string(&mut output, value);
            }
            Metadata::U32(value) => {
                output.extend_from_slice(&4u32.to_le_bytes());
                output.extend_from_slice(&value.to_le_bytes());
            }
            Metadata::F32(value) => {
                output.extend_from_slice(&6u32.to_le_bytes());
                output.extend_from_slice(&value.to_le_bytes());
            }
            Metadata::Bool(value) => {
                output.extend_from_slice(&7u32.to_le_bytes());
                output.push(u8::from(*value));
            }
            Metadata::Strings(values) => {
                output.extend_from_slice(&9u32.to_le_bytes());
                output.extend_from_slice(&8u32.to_le_bytes());
                output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in *values {
                    push_string(&mut output, value);
                }
            }
            Metadata::I32s(values) => {
                output.extend_from_slice(&9u32.to_le_bytes());
                output.extend_from_slice(&5u32.to_le_bytes());
                output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in *values {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
    for ((name, tensor), offset) in state.iter().zip(&offsets) {
        push_string(&mut output, name);
        output.extend_from_slice(&(tensor.shape().rank() as u32).to_le_bytes());
        for dimension in tensor.shape().dims().iter().rev() {
            output.extend_from_slice(&(*dimension as u64).to_le_bytes());
        }
        output.extend_from_slice(&0u32.to_le_bytes());
        output.extend_from_slice(&(*offset as u64).to_le_bytes());
    }
    output.resize(output.len().next_multiple_of(32), 0);
    let data_offset = output.len();
    for ((_, tensor), offset) in state.iter().zip(offsets) {
        output.resize(data_offset + offset, 0);
        for value in tensor.values() {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    output
}

fn metadata(context: u32) -> Vec<(&'static str, Metadata<'static>)> {
    static TOKENS: [&str; VOCAB] = [
        "<s>",
        "</s>",
        "<|im_end|>",
        "a",
        "b",
        "c",
        "d",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "\n\n",
        "user",
        "assistant",
    ];
    static TYPES: [i32; VOCAB] = [3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3];
    vec![
        ("general.architecture", Metadata::String("llama")),
        ("tokenizer.ggml.tokens", Metadata::Strings(&TOKENS)),
        ("tokenizer.ggml.token_type", Metadata::I32s(&TYPES)),
        ("tokenizer.ggml.pre", Metadata::String("llama3")),
        ("tokenizer.ggml.add_bos_token", Metadata::Bool(false)),
        ("tokenizer.ggml.bos_token_id", Metadata::U32(0)),
        ("tokenizer.ggml.eos_token_id", Metadata::U32(1)),
        ("tokenizer.ggml.eot_token_id", Metadata::U32(2)),
        ("llama.block_count", Metadata::U32(LAYERS as u32)),
        ("llama.embedding_length", Metadata::U32(DIM as u32)),
        ("llama.feed_forward_length", Metadata::U32(HIDDEN as u32)),
        (
            "llama.attention.head_count",
            Metadata::U32(QUERY_HEADS as u32),
        ),
        (
            "llama.attention.head_count_kv",
            Metadata::U32(KV_HEADS as u32),
        ),
        ("llama.attention.key_length", Metadata::U32(HEAD_DIM as u32)),
        (
            "llama.attention.value_length",
            Metadata::U32(HEAD_DIM as u32),
        ),
        ("llama.rope.dimension_count", Metadata::U32(ROPE_DIM as u32)),
        ("llama.context_length", Metadata::U32(context)),
        ("llama.attention.layer_norm_rms_epsilon", Metadata::F32(EPS)),
        ("llama.rope.freq_base", Metadata::F32(THETA as f32)),
    ]
}

pub(crate) fn make_model(
    context: u32,
) -> (LlamaModel, SimpleTokenizer, BTreeMap<String, TensorData>) {
    let state = fixed_state();
    let bytes = serialized_model_with_template(context, None);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, tokenizer) = LlamaModel::from_gguf(&file).unwrap();
    (model, tokenizer, state)
}

pub(super) fn make_explicit_model(context: u32) -> LlamaModel {
    let mut state = fixed_state();
    state.insert(OUTPUT_WEIGHT.to_owned(), tensor(&[VOCAB, DIM], 37, 0.013));
    let bytes = fixture(&state, &metadata(context));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    LlamaModel::from_gguf(&file).unwrap().0
}

pub(crate) fn make_variant_model(
    context: u32,
) -> (LlamaModel, SimpleTokenizer, BTreeMap<String, TensorData>) {
    make_variant_model_with_state_delta(context, 0.0)
}

pub(super) fn make_variant_model_with_state_delta(
    context: u32,
    embedding_delta: f32,
) -> (LlamaModel, SimpleTokenizer, BTreeMap<String, TensorData>) {
    let mut state = fixed_state();
    if embedding_delta != 0.0 {
        let mut values = state[TOKEN_EMBEDDING].values().to_vec();
        values[0] += embedding_delta;
        state.insert(
            TOKEN_EMBEDDING.to_owned(),
            TensorData::new([VOCAB, DIM], values).unwrap(),
        );
    }
    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        state.insert(
            format!("{prefix}.attn_q_norm.weight"),
            TensorData::new(
                [HEAD_DIM],
                (0..HEAD_DIM)
                    .map(|index| 0.91 + layer as f32 * 0.02 + index as f32 * 0.03)
                    .collect(),
            )
            .unwrap(),
        );
        state.insert(
            format!("{prefix}.attn_k_norm.weight"),
            TensorData::new(
                [HEAD_DIM],
                (0..HEAD_DIM)
                    .map(|index| 1.07 - layer as f32 * 0.01 - index as f32 * 0.02)
                    .collect(),
            )
            .unwrap(),
        );
        for (suffix, width, salt) in [
            ("attn_q.bias", QUERY_HEADS * HEAD_DIM, 3),
            ("attn_k.bias", KV_HEADS * HEAD_DIM, 5),
            ("attn_v.bias", KV_HEADS * HEAD_DIM, 7),
        ] {
            state.insert(
                format!("{prefix}.{suffix}"),
                TensorData::new(
                    [width],
                    (0..width)
                        .map(|index| ((index + layer * 3 + salt) as f32 - 8.0) * 0.004)
                        .collect(),
                )
                .unwrap(),
            );
        }
    }
    let mut metadata = metadata(context);
    let rope = metadata
        .iter_mut()
        .find(|(key, _)| *key == "llama.rope.dimension_count")
        .unwrap();
    rope.1 = Metadata::U32(2);
    let bytes = fixture(&state, &metadata);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, tokenizer) = LlamaModel::from_gguf(&file).unwrap();
    (model, tokenizer, state)
}

pub(crate) fn serialized_model_with_template(context: u32, template: Option<&str>) -> Vec<u8> {
    let state = fixed_state();
    let mut metadata = metadata(context);
    if let Some(template) = template {
        metadata.push(("tokenizer.chat_template", Metadata::String(template)));
    }
    fixture(&state, &metadata)
}

pub(crate) fn serialized_model_with_template_and_eos(
    context: u32,
    template: Option<&str>,
    eos: u32,
) -> Vec<u8> {
    let state = fixed_state();
    let mut metadata = metadata(context);
    metadata
        .iter_mut()
        .find(|(key, _)| *key == "tokenizer.ggml.eos_token_id")
        .unwrap()
        .1 = Metadata::U32(eos);
    if let Some(template) = template {
        metadata.push(("tokenizer.chat_template", Metadata::String(template)));
    }
    fixture(&state, &metadata)
}

pub(super) fn serialized_model_with_numeric_template(context: u32) -> Vec<u8> {
    let state = fixed_state();
    let mut metadata = metadata(context);
    metadata.push(("tokenizer.chat_template", Metadata::U32(7)));
    fixture(&state, &metadata)
}

#[test]
fn gguf_configuration_binds_every_fixed_layer_and_executes_inspectable_graph() {
    let (model, tokenizer, state) = make_model(8);
    assert_eq!(model.config().architecture(), "llama");
    assert_eq!(model.config().layer_count(), 2);
    assert_eq!(model.config().schema().query_heads(), 2);
    assert_eq!(model.config().schema().kv_heads(), 1);
    assert_eq!(model.config().token_ids().bos(), None);
    assert_eq!(model.config().token_ids().eos(), 1);
    assert_eq!(model.config().token_ids().eot(), Some(2));
    assert_eq!(tokenizer.encode("abc").unwrap(), vec![3, 4, 5]);

    let tokens = [3, 4, 5];
    let plan = model.plan(&tokens).unwrap();
    assert_eq!(
        plan.graph().shape(plan.logits_node()).unwrap().dims(),
        &[3, VOCAB]
    );
    let trace = plan.graph().trace(plan.logits_node()).unwrap();
    assert!(
        trace
            .steps
            .iter()
            .filter(|step| step.operation.starts_with("matmul("))
            .count()
            > LAYERS * 7
    );
    assert!(matches!(
        plan.graph().op(plan.logits_node()).unwrap(),
        Op::Matmul { .. }
    ));
    let actual = model.forward(&tokens).unwrap();
    let expected = reference_logits(&tokens, &state);
    assert_close(&actual, &expected, 3e-5);
}

#[test]
fn partial_rope_per_head_qk_norm_and_qkv_bias_match_independent_reference_and_cache() {
    let (model, tokenizer, state) = make_variant_model(10);
    assert_eq!(model.config().schema().rope_dim(), 2);
    assert_eq!(model.config().qk_norm(), LlamaQkNorm::PerHead);
    assert!(model.config().qkv_bias());
    let tokens = [3, 4, 5];
    let full = model.forward(&tokens).unwrap();
    assert_close(&full, &reference_variant_logits(&tokens, &state), 5e-5);

    let mut token_cache = LlamaModelCache::new(model.config().clone());
    let mut collected = Vec::new();
    for token in tokens {
        collected.extend_from_slice(token_cache.forward(&model, &[token]).unwrap().values());
    }
    assert_close(
        &TensorData::new([3, VOCAB], collected).unwrap(),
        &full,
        5e-5,
    );
    let mut chunk_cache = LlamaModelCache::new(model.config().clone());
    let first = chunk_cache.forward(&model, &[3, 4]).unwrap();
    let second = chunk_cache.forward(&model, &[5]).unwrap();
    let mut chunked = first.values().to_vec();
    chunked.extend_from_slice(second.values());
    assert_close(&TensorData::new([3, VOCAB], chunked).unwrap(), &full, 5e-5);
    let generated = LlamaGenerator::new(&model, &tokenizer)
        .generate_text("a", 1, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(generated.prompt_ids(), &[3]);
    assert_eq!(generated.generated_ids().len(), 1);

    let mut explicit_state = state;
    explicit_state.insert(OUTPUT_WEIGHT.to_owned(), tensor(&[VOCAB, DIM], 91, 0.017));
    let mut explicit_metadata = metadata(10);
    explicit_metadata
        .iter_mut()
        .find(|(key, _)| *key == "llama.rope.dimension_count")
        .unwrap()
        .1 = Metadata::U32(2);
    let bytes = fixture(&explicit_state, &explicit_metadata);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (explicit, _) = LlamaModel::from_gguf(&file).unwrap();
    assert_eq!(explicit.output_binding(), LlamaOutputBinding::Explicit);
    assert_ne!(explicit.forward(&tokens).unwrap(), full);
}

#[test]
fn full_projection_qk_norm_requires_equal_projection_widths_and_runs_before_heads() {
    let mut state = fixed_state();
    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        state.insert(
            format!("{prefix}.attn_k.weight"),
            tensor(&[QUERY_HEADS * HEAD_DIM, DIM], 40 + layer, 0.012),
        );
        state.insert(
            format!("{prefix}.attn_v.weight"),
            tensor(&[QUERY_HEADS * HEAD_DIM, DIM], 50 + layer, 0.011),
        );
        state.insert(
            format!("{prefix}.attn_q_norm.weight"),
            TensorData::new([QUERY_HEADS * HEAD_DIM], vec![1.0; QUERY_HEADS * HEAD_DIM]).unwrap(),
        );
        state.insert(
            format!("{prefix}.attn_k_norm.weight"),
            TensorData::new([QUERY_HEADS * HEAD_DIM], vec![1.0; QUERY_HEADS * HEAD_DIM]).unwrap(),
        );
    }
    let mut metadata = metadata(8);
    metadata
        .iter_mut()
        .find(|(key, _)| *key == "llama.attention.head_count_kv")
        .unwrap()
        .1 = Metadata::U32(QUERY_HEADS as u32);
    let bytes = fixture(&state, &metadata);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, _) = LlamaModel::from_gguf(&file).unwrap();
    assert_eq!(model.config().qk_norm(), LlamaQkNorm::PerProjection);
    let full = model.forward(&[3, 4, 5]).unwrap();
    let mut cache = LlamaModelCache::new(model.config().clone());
    let mut incremental = Vec::new();
    for token in [3, 4, 5] {
        incremental.extend_from_slice(cache.forward(&model, &[token]).unwrap().values());
    }
    assert_close(
        &TensorData::new([3, VOCAB], incremental).unwrap(),
        &full,
        5e-5,
    );
}

#[test]
fn variant_metadata_and_tensor_families_fail_closed() {
    let base = fixed_state();
    let mut partial_norm = base.clone();
    partial_norm.insert(
        "blk.0.attn_q_norm.weight".to_owned(),
        TensorData::new([HEAD_DIM], vec![1.0; HEAD_DIM]).unwrap(),
    );
    let bytes = fixture(&partial_norm, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("partial or incompatible q/k normalization")
    );

    let mut unequal_projection_norm = fixed_state();
    unequal_projection_norm.insert(
        "blk.0.attn_q_norm.weight".to_owned(),
        TensorData::new([QUERY_HEADS * HEAD_DIM], vec![1.0; QUERY_HEADS * HEAD_DIM]).unwrap(),
    );
    unequal_projection_norm.insert(
        "blk.0.attn_k_norm.weight".to_owned(),
        TensorData::new([KV_HEADS * HEAD_DIM], vec![1.0; KV_HEADS * HEAD_DIM]).unwrap(),
    );
    let bytes = fixture(&unequal_projection_norm, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("partial or incompatible q/k normalization")
    );

    let mut mixed_norm_layers = base.clone();
    for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
        mixed_norm_layers.insert(
            format!("blk.0.{suffix}"),
            TensorData::new([HEAD_DIM], vec![1.0; HEAD_DIM]).unwrap(),
        );
    }
    mixed_norm_layers.insert(
        "blk.1.attn_q_norm.weight".to_owned(),
        TensorData::new([HEAD_DIM], vec![1.0; HEAD_DIM]).unwrap(),
    );
    let bytes = fixture(&mixed_norm_layers, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::MissingTensor("blk.1.attn_k_norm.weight".to_owned())
    );

    let mut bad_norm = base.clone();
    bad_norm.insert(
        "blk.0.attn_q_norm.weight".to_owned(),
        TensorData::new([3], vec![1.0; 3]).unwrap(),
    );
    bad_norm.insert(
        "blk.0.attn_k_norm.weight".to_owned(),
        TensorData::new([3], vec![1.0; 3]).unwrap(),
    );
    let bytes = fixture(&bad_norm, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("partial or incompatible q/k normalization")
    );

    let mut partial_bias = base.clone();
    partial_bias.insert(
        "blk.0.attn_q.bias".to_owned(),
        TensorData::new([QUERY_HEADS * HEAD_DIM], vec![0.0; QUERY_HEADS * HEAD_DIM]).unwrap(),
    );
    let bytes = fixture(&partial_bias, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("partial q/k/v projection bias family")
    );

    let mut mixed_layers = base.clone();
    for (suffix, width) in [
        ("attn_q.bias", QUERY_HEADS * HEAD_DIM),
        ("attn_k.bias", KV_HEADS * HEAD_DIM),
        ("attn_v.bias", KV_HEADS * HEAD_DIM),
    ] {
        mixed_layers.insert(
            format!("blk.0.{suffix}"),
            TensorData::new([width], vec![0.0; width]).unwrap(),
        );
    }
    let bytes = fixture(&mixed_layers, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::MissingTensor("blk.1.attn_q.bias".to_owned())
    );

    let mut bad_bias_shape = fixed_state();
    for layer in 0..LAYERS {
        for (suffix, width) in [
            ("attn_q.bias", QUERY_HEADS * HEAD_DIM),
            ("attn_k.bias", KV_HEADS * HEAD_DIM),
            ("attn_v.bias", KV_HEADS * HEAD_DIM),
        ] {
            let actual = if layer == 0 && suffix == "attn_q.bias" {
                width - 1
            } else {
                width
            };
            bad_bias_shape.insert(
                format!("blk.{layer}.{suffix}"),
                TensorData::new([actual], vec![0.0; actual]).unwrap(),
            );
        }
    }
    let bytes = fixture(&bad_bias_shape, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::ShapeMismatch {
            tensor: "blk.0.attn_q.bias".to_owned(),
            expected: vec![QUERY_HEADS * HEAD_DIM],
            actual: vec![QUERY_HEADS * HEAD_DIM - 1],
        }
    );

    let mut unsupported_bias = base;
    unsupported_bias.insert(
        "output.bias".to_owned(),
        TensorData::new([VOCAB], vec![0.0; VOCAB]).unwrap(),
    );
    let bytes = fixture(&unsupported_bias, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnexpectedTensor("output.bias".to_owned())
    );

    let mut scaled = metadata(8);
    scaled.push(("llama.rope.scaling.factor", Metadata::F32(2.0)));
    let bytes = fixture(&fixed_state(), &scaled);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("RoPE frequency scaling metadata")
    );

    let mut odd_rope = metadata(8);
    odd_rope
        .iter_mut()
        .find(|(key, _)| *key == "llama.rope.dimension_count")
        .unwrap()
        .1 = Metadata::U32(3);
    let bytes = fixture(&fixed_state(), &odd_rope);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("invalid partial rotary dimensions")
    );

    let mut zero_rope = metadata(8);
    zero_rope
        .iter_mut()
        .find(|(key, _)| *key == "llama.rope.dimension_count")
        .unwrap()
        .1 = Metadata::U32(0);
    let bytes = fixture(&fixed_state(), &zero_rope);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("invalid partial rotary dimensions")
    );
}

#[test]
fn two_layer_full_sequence_equals_token_and_chunk_incremental_cache() {
    let (model, _, _) = make_model(8);
    let tokens = [3, 4, 5];
    let full = model.forward(&tokens).unwrap();
    let mut cache = LlamaModelCache::new(model.config().clone());
    let mut incremental = Vec::new();
    for (position, token) in tokens.into_iter().enumerate() {
        let logits = cache.forward(&model, &[token]).unwrap();
        assert_eq!(cache.len(), position + 1);
        incremental.extend_from_slice(logits.values());
    }
    assert_close(
        &full,
        &TensorData::new([3, VOCAB], incremental).unwrap(),
        3e-5,
    );

    cache.clear();
    cache.forward(&model, &[3, 4]).unwrap();
    let suffix = cache.forward(&model, &[5]).unwrap();
    assert_close(
        &suffix,
        &TensorData::new([1, VOCAB], full.values()[2 * VOCAB..].to_vec()).unwrap(),
        3e-5,
    );
}

#[test]
fn generation_is_deterministic_stops_and_commits_cache_atomically() {
    let (model, tokenizer, _) = make_model(8);
    let mut generator = LlamaGenerator::new(&model, &tokenizer);
    let first = generator
        .generate_text("abc", 2, LlamaSampling::Greedy)
        .unwrap();
    let second = generator
        .generate_text("abc", 2, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.prompt_ids(), &[3, 4, 5]);
    assert_eq!(
        first.decoded(),
        tokenizer.decode(first.generated_ids()).unwrap()
    );
    assert_eq!(generator.cache_len(), 4);

    let mut tape = vec![1e-9; VOCAB];
    tape[1] = 0.9;
    let stopped = generator
        .generate_text(
            "a",
            1,
            LlamaSampling::GumbelMax {
                temperature: 1e6,
                uniforms: &tape,
            },
        )
        .unwrap();
    assert_eq!(stopped.generated_ids(), &[1]);
    assert!(stopped.stopped());
    assert_eq!(stopped.decoded(), "</s>");

    let before = generator.cache_len();
    assert_eq!(
        generator
            .generate_text("abc", 6, LlamaSampling::Greedy)
            .unwrap_err(),
        LlamaGenerationError::ContextLength {
            requested: 9,
            maximum: 8
        }
    );
    assert_eq!(generator.cache_len(), before);
    assert_eq!(
        generator
            .generate_text(
                "a",
                2,
                LlamaSampling::GumbelMax {
                    temperature: 1.0,
                    uniforms: &[0.5]
                }
            )
            .unwrap_err(),
        LlamaGenerationError::UniformTapeLength {
            required: 24,
            actual: 1
        }
    );
    assert_eq!(generator.cache_len(), before);
}

#[test]
fn malformed_metadata_variants_layers_and_cache_are_typed() {
    let state = fixed_state();
    let mut wrong_type = metadata(8);
    let block = wrong_type
        .iter_mut()
        .find(|(key, _)| *key == "llama.block_count")
        .unwrap();
    *block = ("llama.block_count", Metadata::String("2"));
    let bytes = fixture(&state, &wrong_type);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModelConfig::from_gguf(&file).unwrap_err(),
        LlamaModelError::Metadata(crate::gguf::GgufMetadataAccessError::TypeMismatch {
            key: "llama.block_count".to_owned(),
            expected: crate::gguf::GgufMetadataExpectation::UnsignedInteger,
            actual: crate::gguf::GgufMetadataType::String,
        })
    );

    let mut wrong_arch = metadata(8);
    wrong_arch[0] = ("general.architecture", Metadata::String("qwen2"));
    let bytes = fixture(&state, &wrong_arch);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModelConfig::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedArchitecture("qwen2".to_owned())
    );

    let mut expert = metadata(8);
    expert.push(("llama.expert_count", Metadata::U32(8)));
    let bytes = fixture(&state, &expert);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModelConfig::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnsupportedVariant("mixture-of-experts")
    );

    let mut missing = state.clone();
    missing.remove("blk.1.attn_k.weight");
    let bytes = fixture(&missing, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::MissingTensor("blk.1.attn_k.weight".to_owned())
    );

    let mut misshaped = state.clone();
    misshaped.insert("blk.1.attn_k.weight".to_owned(), tensor(&[2, 16], 99, 0.01));
    let bytes = fixture(&misshaped, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::ShapeMismatch {
            tensor: "blk.1.attn_k.weight".to_owned(),
            expected: vec![4, 8],
            actual: vec![2, 16],
        }
    );

    let mut unexpected = state;
    unexpected.insert(
        "blk.1.attn_q.bias".to_owned(),
        TensorData::new([QUERY_HEADS * HEAD_DIM], vec![0.0; QUERY_HEADS * HEAD_DIM]).unwrap(),
    );
    let bytes = fixture(&unexpected, &metadata(8));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaModel::from_gguf(&file).unwrap_err(),
        LlamaModelError::UnexpectedTensor("blk.1.attn_q.bias".to_owned())
    );

    let (model, _, _) = make_model(4);
    let mut cache = LlamaModelCache::new(model.config().clone());
    cache.forward(&model, &[3, 4]).unwrap();
    let before = cache.len();
    assert_eq!(
        cache.forward(&model, &[5, 6, 3]).unwrap_err(),
        LlamaModelError::ContextLength {
            requested: 5,
            maximum: 4
        }
    );
    assert_eq!(cache.len(), before);
    let (other, _, _) = make_model(5);
    assert_eq!(
        cache.forward(&other, &[5]).unwrap_err(),
        LlamaModelError::CacheConfigMismatch
    );
    assert_eq!(cache.len(), before);
}

pub(super) fn reference_logits(tokens: &[u32], state: &BTreeMap<String, TensorData>) -> TensorData {
    reference_logits_with_variant(tokens, state, ROPE_DIM, false, false)
}

pub(super) fn reference_variant_logits(
    tokens: &[u32],
    state: &BTreeMap<String, TensorData>,
) -> TensorData {
    reference_logits_with_variant(tokens, state, 2, true, true)
}

fn reference_logits_with_variant(
    tokens: &[u32],
    state: &BTreeMap<String, TensorData>,
    rope_dim: usize,
    qk_norm: bool,
    qkv_bias: bool,
) -> TensorData {
    let matrix = |name: &str| state[name].values();
    let mut hidden = tokens
        .iter()
        .map(|token| {
            matrix(TOKEN_EMBEDDING)[*token as usize * DIM..(*token as usize + 1) * DIM]
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        let normalized = hidden
            .iter()
            .map(|row| rms_ref(row, matrix(&format!("{prefix}.attn_norm.weight"))))
            .collect::<Vec<_>>();
        let mut queries = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (position, row) in normalized.iter().enumerate() {
            let mut query_heads = projected_heads_variant(
                row,
                matrix(&format!("{prefix}.attn_q.weight")),
                qkv_bias.then(|| matrix(&format!("{prefix}.attn_q.bias"))),
                QUERY_HEADS,
                rope_dim,
                ProjectionKind::Query,
            );
            let mut key_heads = projected_heads_variant(
                row,
                matrix(&format!("{prefix}.attn_k.weight")),
                qkv_bias.then(|| matrix(&format!("{prefix}.attn_k.bias"))),
                KV_HEADS,
                rope_dim,
                ProjectionKind::Key,
            );
            if qk_norm {
                let query_norm = matrix(&format!("{prefix}.attn_q_norm.weight"));
                let key_norm = matrix(&format!("{prefix}.attn_k_norm.weight"));
                query_heads = query_heads
                    .iter()
                    .map(|head| rms_ref(head, query_norm))
                    .collect();
                key_heads = key_heads
                    .iter()
                    .map(|head| rms_ref(head, key_norm))
                    .collect();
            }
            queries.push(
                query_heads
                    .into_iter()
                    .map(|head| rope_ref_variant(head, position, rope_dim))
                    .collect::<Vec<_>>(),
            );
            keys.push(
                key_heads
                    .into_iter()
                    .map(|head| rope_ref_variant(head, position, rope_dim))
                    .collect::<Vec<_>>(),
            );
            values.push(projected_heads_variant(
                row,
                matrix(&format!("{prefix}.attn_v.weight")),
                qkv_bias.then(|| matrix(&format!("{prefix}.attn_v.bias"))),
                KV_HEADS,
                rope_dim,
                ProjectionKind::Value,
            ));
        }
        for position in 0..tokens.len() {
            let mut flattened = Vec::with_capacity(QUERY_HEADS * HEAD_DIM);
            for query_head in 0..QUERY_HEADS {
                let kv_head = query_head / (QUERY_HEADS / KV_HEADS);
                let scores = (0..=position)
                    .map(|key_position| {
                        dot(&queries[position][query_head], &keys[key_position][kv_head])
                            / (HEAD_DIM as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let total: f64 = exp.iter().sum();
                for component in 0..HEAD_DIM {
                    flattened.push(
                        (0..=position)
                            .map(|key_position| {
                                exp[key_position] / total * values[key_position][kv_head][component]
                            })
                            .sum(),
                    );
                }
            }
            let attention = project(
                &flattened,
                matrix(&format!("{prefix}.attn_output.weight")),
                DIM,
                QUERY_HEADS * HEAD_DIM,
            );
            for (hidden, attention) in hidden[position].iter_mut().zip(attention) {
                *hidden += attention;
            }
            let normalized = rms_ref(
                &hidden[position],
                matrix(&format!("{prefix}.ffn_norm.weight")),
            );
            let gate = project(
                &normalized,
                matrix(&format!("{prefix}.ffn_gate.weight")),
                HIDDEN,
                DIM,
            )
            .into_iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
            let up = project(
                &normalized,
                matrix(&format!("{prefix}.ffn_up.weight")),
                HIDDEN,
                DIM,
            );
            let gated = gate
                .iter()
                .zip(up)
                .map(|(gate, up)| gate * up)
                .collect::<Vec<_>>();
            let down = project(
                &gated,
                matrix(&format!("{prefix}.ffn_down.weight")),
                DIM,
                HIDDEN,
            );
            for (hidden, down) in hidden[position].iter_mut().zip(down) {
                *hidden += down;
            }
        }
    }
    let logits = hidden.into_iter().flat_map(|row| {
        project(
            &rms_ref(&row, matrix(OUTPUT_NORM)),
            matrix(TOKEN_EMBEDDING),
            VOCAB,
            DIM,
        )
    });
    TensorData::new(
        [tokens.len(), VOCAB],
        logits.map(|value| value as f32).collect(),
    )
    .unwrap()
}

fn rms_ref(input: &[f64], weight: &[f32]) -> Vec<f64> {
    let mean = input.iter().map(|value| value * value).sum::<f64>() / input.len() as f64;
    let scale = 1.0 / (mean + f64::from(EPS)).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * f64::from(*weight))
        .collect()
}

fn project(input: &[f64], weight: &[f32], rows: usize, columns: usize) -> Vec<f64> {
    (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| input[column] * f64::from(weight[row * columns + column]))
                .sum()
        })
        .collect()
}

#[derive(Clone, Copy)]
enum ProjectionKind {
    Query,
    Key,
    Value,
}

fn projected_heads_variant(
    input: &[f64],
    weight: &[f32],
    bias: Option<&[f32]>,
    heads: usize,
    rope_dim: usize,
    kind: ProjectionKind,
) -> Vec<Vec<f64>> {
    (0..heads)
        .map(|head| {
            (0..HEAD_DIM)
                .map(|output| {
                    let source = match kind {
                        ProjectionKind::Value => output,
                        ProjectionKind::Key => {
                            if output < HEAD_DIM / 2 {
                                output * 2
                            } else {
                                (output - HEAD_DIM / 2) * 2 + 1
                            }
                        }
                        ProjectionKind::Query => {
                            let prefix = HEAD_DIM - rope_dim;
                            if output < prefix {
                                output
                            } else {
                                let rotary = output - prefix;
                                if rotary < rope_dim / 2 {
                                    prefix + rotary * 2
                                } else {
                                    prefix + (rotary - rope_dim / 2) * 2 + 1
                                }
                            }
                        }
                    };
                    let row = head * HEAD_DIM + source;
                    let projected: f64 = (0..DIM)
                        .map(|column| input[column] * f64::from(weight[row * DIM + column]))
                        .sum();
                    projected + bias.map_or(0.0, |bias| f64::from(bias[head * HEAD_DIM + output]))
                })
                .collect()
        })
        .collect()
}

fn rope_ref_variant(input: Vec<f64>, position: usize, rope_dim: usize) -> Vec<f64> {
    let half = rope_dim / 2;
    let mut output = vec![0.0; HEAD_DIM];
    for index in 0..half {
        let angle = position as f64 / THETA.powf((2 * index) as f64 / rope_dim as f64);
        let (cos, sin) = (angle.cos(), angle.sin());
        output[index] = input[index] * cos - input[index + half] * sin;
        output[index + half] = input[index + half] * cos + input[index] * sin;
    }
    output[rope_dim..HEAD_DIM].copy_from_slice(&input[rope_dim..HEAD_DIM]);
    output
}
fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum()
}
pub(super) fn assert_close(actual: &TensorData, expected: &TensorData, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), DType::F32);
    for index in 0..actual.len() {
        let difference =
            (actual.scalar_at(index).as_f64() - expected.scalar_at(index).as_f64()).abs();
        assert!(
            difference <= tolerance,
            "index {index}: actual={} expected={} difference={difference}",
            actual.scalar_at(index).as_f64(),
            expected.scalar_at(index).as_f64()
        );
    }
}
