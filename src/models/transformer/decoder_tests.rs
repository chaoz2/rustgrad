use super::*;
use crate::{
    DType, Op, TensorData,
    tokenizer::{SimpleTokenizer, TokenizerConfig},
};
use std::collections::BTreeMap;

const VOCAB: usize = 7;
const DIM: usize = 8;
const HIDDEN: usize = 10;
const QUERY_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const ROPE_DIM: usize = 4;
const EPS: f32 = 1e-5;
const THETA: f64 = 10.0;

fn schema() -> LlamaDecoderSchema {
    LlamaDecoderSchema::new(
        VOCAB,
        DIM,
        HIDDEN,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        ROPE_DIM,
    )
    .unwrap()
}

fn values(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let centered = ((index * 17 + salt * 11) % 29) as f32 - 14.0;
            centered * scale
        })
        .collect()
}

fn tensor(shape: &[usize], salt: usize, scale: f32) -> TensorData {
    TensorData::new(shape.to_vec(), values(shape.iter().product(), salt, scale)).unwrap()
}

fn fixed_state() -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        (TOKEN_EMBEDDING.to_owned(), tensor(&[VOCAB, DIM], 1, 0.018)),
        (
            OUTPUT_NORM.to_owned(),
            TensorData::new([DIM], (0..DIM).map(|i| 0.92 + i as f32 * 0.015).collect()).unwrap(),
        ),
        (
            "blk.0.attn_norm.weight".to_owned(),
            TensorData::new([DIM], (0..DIM).map(|i| 1.03 - i as f32 * 0.01).collect()).unwrap(),
        ),
        (
            "blk.0.attn_q.weight".to_owned(),
            tensor(&[QUERY_HEADS * HEAD_DIM, DIM], 2, 0.012),
        ),
        (
            "blk.0.attn_k.weight".to_owned(),
            tensor(&[KV_HEADS * HEAD_DIM, DIM], 3, 0.014),
        ),
        (
            "blk.0.attn_v.weight".to_owned(),
            tensor(&[KV_HEADS * HEAD_DIM, DIM], 4, 0.013),
        ),
        (
            "blk.0.attn_output.weight".to_owned(),
            tensor(&[DIM, QUERY_HEADS * HEAD_DIM], 5, 0.011),
        ),
        (
            "blk.0.ffn_norm.weight".to_owned(),
            TensorData::new([DIM], (0..DIM).map(|i| 0.97 + i as f32 * 0.008).collect()).unwrap(),
        ),
        (
            "blk.0.ffn_gate.weight".to_owned(),
            tensor(&[HIDDEN, DIM], 6, 0.01),
        ),
        (
            "blk.0.ffn_up.weight".to_owned(),
            tensor(&[HIDDEN, DIM], 7, 0.009),
        ),
        (
            "blk.0.ffn_down.weight".to_owned(),
            tensor(&[DIM, HIDDEN], 8, 0.01),
        ),
    ])
}

fn make_decoder(max_context: usize) -> (LlamaDecoder, BTreeMap<String, TensorData>) {
    let raw = fixed_state();
    let state = schema().bind_materialized(raw.clone()).unwrap();
    let config = LlamaDecoderConfig::new(schema(), max_context, EPS, THETA).unwrap();
    (LlamaDecoder::new(config, state).unwrap(), raw)
}

#[test]
fn graph_logits_match_independent_dense_reference_and_are_inspectable() {
    let (decoder, state) = make_decoder(8);
    let tokens = [1, 3, 2];
    let plan = decoder.plan(&tokens).unwrap();
    assert_eq!(
        plan.graph().shape(plan.logits_node()).unwrap().dims(),
        &[tokens.len(), VOCAB]
    );
    let trace = plan.graph().trace(plan.logits_node()).unwrap();
    for operation in ["gather(", "matmul(", "Mean(", "where(", "exp("] {
        assert!(
            trace
                .steps
                .iter()
                .any(|step| step.operation.starts_with(operation)),
            "missing {operation} from decoder trace"
        );
    }
    assert!(matches!(
        plan.graph().op(plan.logits_node()).unwrap(),
        Op::Matmul { .. }
    ));

    let actual = plan.execute().unwrap();
    let expected = reference_logits(&tokens, &state);
    assert_close(actual.logits(), &expected, 2e-5);
}

#[test]
fn full_sequence_equals_incremental_cache_including_nonzero_rope_positions() {
    let (decoder, _) = make_decoder(8);
    let tokens = [1, 3, 2];
    let full = decoder.forward(&tokens).unwrap();
    let mut cache = LlamaKvCache::new(decoder.config());
    let mut incremental = Vec::new();
    for (position, token) in tokens.into_iter().enumerate() {
        let logits = cache.forward(&decoder, &[token]).unwrap();
        assert_eq!(cache.len(), position + 1);
        incremental.extend_from_slice(logits.values());
    }
    let incremental = TensorData::new([3, VOCAB], incremental).unwrap();
    assert_close(full.logits(), &incremental, 2e-5);

    cache.clear();
    assert!(cache.is_empty());
    let prefix = cache.forward(&decoder, &[1, 3]).unwrap();
    assert_eq!(prefix.shape().dims(), &[2, VOCAB]);
    let suffix = cache.forward(&decoder, &[2]).unwrap();
    assert_close(
        &suffix,
        &TensorData::new([1, VOCAB], full.logits().values()[2 * VOCAB..].to_vec()).unwrap(),
        2e-5,
    );
}

#[test]
fn tokenizer_ids_drive_the_graph_and_cache_failures_are_non_mutating() {
    let tokenizer = SimpleTokenizer::new(
        [
            ("a".to_owned(), 1),
            ("b".to_owned(), 3),
            ("c".to_owned(), 2),
        ],
        [],
        TokenizerConfig::default(),
    )
    .unwrap();
    let tokens = tokenizer.encode("abc").unwrap();
    assert_eq!(tokens, vec![1, 3, 2]);
    let (decoder, _) = make_decoder(3);
    let logits = decoder.forward(&tokens).unwrap();
    assert_eq!(logits.logits().shape().dims(), &[3, VOCAB]);

    let mut cache = LlamaKvCache::new(decoder.config());
    cache.forward(&decoder, &[1, 3]).unwrap();
    let before = cache.len();
    assert_eq!(
        cache.forward(&decoder, &[2, 1]).unwrap_err(),
        LlamaDecoderError::ContextLength {
            requested: 4,
            maximum: 3,
        }
    );
    assert_eq!(cache.len(), before);
}

#[test]
fn decoder_validation_rejects_invalid_runtime_inputs_and_schema_mismatch() {
    assert_eq!(
        LlamaDecoderConfig::new(schema(), 0, EPS, THETA).unwrap_err(),
        LlamaDecoderError::InvalidConfig {
            reason: "max_context must be nonzero"
        }
    );
    assert_eq!(
        LlamaDecoderConfig::new(schema(), 8, -1.0, THETA).unwrap_err(),
        LlamaDecoderError::InvalidConfig {
            reason: "norm_eps must be finite and non-negative"
        }
    );
    let (decoder, _) = make_decoder(3);
    assert_eq!(
        decoder.plan(&[]).unwrap_err(),
        LlamaDecoderError::EmptyTokens
    );
    assert_eq!(
        decoder.plan(&[VOCAB as u32]).unwrap_err(),
        LlamaDecoderError::TokenOutOfRange {
            token: VOCAB as u32,
            vocab_size: VOCAB,
        }
    );

    let other_schema = LlamaDecoderSchema::new(8, DIM, HIDDEN, 2, 1, 4, 4).unwrap();
    let other_state = other_schema
        .bind_materialized({
            let mut state = fixed_state();
            state.insert(TOKEN_EMBEDDING.to_owned(), tensor(&[8, DIM], 9, 0.01));
            state
        })
        .unwrap();
    assert_eq!(
        LlamaDecoder::new(decoder.config(), other_state).unwrap_err(),
        LlamaDecoderError::StateSchemaMismatch
    );

    let (other_decoder, _) = make_decoder(4);
    let mut cache = LlamaKvCache::new(decoder.config());
    assert_eq!(
        cache.forward(&other_decoder, &[1]).unwrap_err(),
        LlamaDecoderError::CacheConfigMismatch
    );
}

fn reference_logits(tokens: &[u32], state: &BTreeMap<String, TensorData>) -> TensorData {
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
    let normalized = hidden
        .iter()
        .map(|row| rms_ref(row, matrix("blk.0.attn_norm.weight")))
        .collect::<Vec<_>>();
    let mut queries = Vec::new();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (position, row) in normalized.iter().enumerate() {
        let q = projected_heads(row, matrix("blk.0.attn_q.weight"), QUERY_HEADS, true);
        let k = projected_heads(row, matrix("blk.0.attn_k.weight"), KV_HEADS, true);
        let v = projected_heads(row, matrix("blk.0.attn_v.weight"), KV_HEADS, false);
        queries.push(
            q.into_iter()
                .map(|head| rope_ref(head, position))
                .collect::<Vec<_>>(),
        );
        keys.push(
            k.into_iter()
                .map(|head| rope_ref(head, position))
                .collect::<Vec<_>>(),
        );
        values.push(v);
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
            matrix("blk.0.attn_output.weight"),
            DIM,
            QUERY_HEADS * HEAD_DIM,
        );
        for (hidden, attention) in hidden[position].iter_mut().zip(attention) {
            *hidden += attention;
        }
        let normalized = rms_ref(&hidden[position], matrix("blk.0.ffn_norm.weight"));
        let gate = project(&normalized, matrix("blk.0.ffn_gate.weight"), HIDDEN, DIM)
            .into_iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let up = project(&normalized, matrix("blk.0.ffn_up.weight"), HIDDEN, DIM);
        let gated = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| gate * up)
            .collect::<Vec<_>>();
        let down = project(&gated, matrix("blk.0.ffn_down.weight"), DIM, HIDDEN);
        for (hidden, down) in hidden[position].iter_mut().zip(down) {
            *hidden += down;
        }
    }
    let logits = hidden.into_iter().flat_map(|row| {
        let normalized = rms_ref(&row, matrix(OUTPUT_NORM));
        project(&normalized, matrix(TOKEN_EMBEDDING), VOCAB, DIM)
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

fn projected_heads(
    input: &[f64],
    weight: &[f32],
    heads: usize,
    permute_rope: bool,
) -> Vec<Vec<f64>> {
    (0..heads)
        .map(|head| {
            (0..HEAD_DIM)
                .map(|output| {
                    let source = if permute_rope {
                        if output < ROPE_DIM / 2 {
                            output * 2
                        } else {
                            (output - ROPE_DIM / 2) * 2 + 1
                        }
                    } else {
                        output
                    };
                    let row = head * HEAD_DIM + source;
                    (0..DIM)
                        .map(|column| input[column] * f64::from(weight[row * DIM + column]))
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn rope_ref(input: Vec<f64>, position: usize) -> Vec<f64> {
    let half = ROPE_DIM / 2;
    let mut output = vec![0.0; HEAD_DIM];
    for index in 0..half {
        let angle = position as f64 / THETA.powf((2 * index) as f64 / ROPE_DIM as f64);
        let (cos, sin) = (angle.cos(), angle.sin());
        output[index] = input[index] * cos - input[index + half] * sin;
        output[index + half] = input[index + half] * cos + input[index] * sin;
    }
    output[ROPE_DIM..HEAD_DIM].copy_from_slice(&input[ROPE_DIM..HEAD_DIM]);
    output
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn assert_close(actual: &TensorData, expected: &TensorData, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), DType::F32);
    for index in 0..actual.len() {
        let difference =
            (actual.scalar_at(index).as_f64() - expected.scalar_at(index).as_f64()).abs();
        assert!(
            difference <= tolerance,
            "index {index}: actual={} expected={} difference={difference}",
            actual.scalar_at(index).as_f64(),
            expected.scalar_at(index).as_f64(),
        );
    }
}
