use super::LlamaDecoderSchema;
use crate::{AttentionOptions, DType, Error, Graph, NodeId, Scalar, Shape, TensorData};
use std::collections::{BTreeMap, HashMap};

pub(super) struct DenseLayerNodes {
    pub(super) output: NodeId,
    pub(super) keys: NodeId,
    pub(super) values: NodeId,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_dense_layer(
    graph: &mut Graph,
    bindings: &mut HashMap<String, TensorData>,
    mut x: NodeId,
    state: &BTreeMap<String, TensorData>,
    tensor_prefix: &str,
    cache_prefix: &str,
    schema: LlamaDecoderSchema,
    sequence: usize,
    past_len: usize,
    total_len: usize,
    norm_eps: f32,
    rope_theta: f64,
    past_keys: Option<&TensorData>,
    past_values: Option<&TensorData>,
) -> Result<DenseLayerNodes, Error> {
    let weight = |suffix: &str| state[&format!("{tensor_prefix}.{suffix}")].clone();
    let attn_norm = graph.constant(weight("attn_norm.weight"));
    let normalized = rms_norm(graph, x, attn_norm, schema.embedding_dim, norm_eps)?;
    let query_weight = graph.constant(weight("attn_q.weight"));
    let query_weight = permute_rope_weight(
        graph,
        query_weight,
        schema.query_heads,
        schema.head_dim,
        schema.rope_dim,
        schema.embedding_dim,
    )?;
    let key_weight = graph.constant(weight("attn_k.weight"));
    let key_weight = permute_rope_weight(
        graph,
        key_weight,
        schema.kv_heads,
        schema.head_dim,
        schema.rope_dim,
        schema.embedding_dim,
    )?;
    let value_weight = graph.constant(weight("attn_v.weight"));
    let query = linear(graph, normalized, query_weight)?;
    let query = heads(graph, query, sequence, schema.query_heads, schema.head_dim)?;
    let key = linear(graph, normalized, key_weight)?;
    let key = heads(graph, key, sequence, schema.kv_heads, schema.head_dim)?;
    let value = linear(graph, normalized, value_weight)?;
    let value = heads(graph, value, sequence, schema.kv_heads, schema.head_dim)?;
    let query = apply_rope(
        graph,
        query,
        schema.query_heads,
        sequence,
        schema.head_dim,
        schema.rope_dim,
        past_len,
        rope_theta,
    )?;
    let key = apply_rope(
        graph,
        key,
        schema.kv_heads,
        sequence,
        schema.head_dim,
        schema.rope_dim,
        past_len,
        rope_theta,
    )?;
    let (keys, values) = match (past_keys, past_values) {
        (Some(past_keys), Some(past_values)) => {
            let key_name = format!("{cache_prefix}.keys");
            let value_name = format!("{cache_prefix}.values");
            let key_node = graph.input_dtype_requires_grad(
                &key_name,
                past_keys.shape().clone(),
                DType::F32,
                false,
            );
            let value_node = graph.input_dtype_requires_grad(
                &value_name,
                past_values.shape().clone(),
                DType::F32,
                false,
            );
            bindings.insert(key_name, past_keys.clone());
            bindings.insert(value_name, past_values.clone());
            (
                graph.concat(vec![key_node, key], 1)?,
                graph.concat(vec![value_node, value], 1)?,
            )
        }
        (None, None) => (key, value),
        _ => unreachable!("callers validate complete cache pairs"),
    };
    let mask = graph.constant(TensorData::from_scalars(
        [sequence, total_len],
        DType::Bool,
        (0..sequence).flat_map(|row| {
            (0..total_len).map(move |column| Scalar::Bool(column <= past_len + row))
        }),
    )?);
    let attended = graph.scaled_dot_product_attention(
        query,
        keys,
        values,
        Some(mask),
        AttentionOptions {
            enable_gqa: true,
            ..AttentionOptions::default()
        },
    )?;
    let attended = graph.permute(attended, vec![1, 0, 2])?;
    let attended = graph.reshape(attended, [sequence, schema.query_heads * schema.head_dim])?;
    let output_weight = graph.constant(weight("attn_output.weight"));
    let attended = linear(graph, attended, output_weight)?;
    x = graph.add(x, attended)?;

    let ffn_norm = graph.constant(weight("ffn_norm.weight"));
    let normalized = rms_norm(graph, x, ffn_norm, schema.embedding_dim, norm_eps)?;
    let gate_weight = graph.constant(weight("ffn_gate.weight"));
    let up_weight = graph.constant(weight("ffn_up.weight"));
    let down_weight = graph.constant(weight("ffn_down.weight"));
    let gate_linear = linear(graph, normalized, gate_weight)?;
    let gate = graph.silu(gate_linear)?;
    let up = linear(graph, normalized, up_weight)?;
    let gated = graph.mul(gate, up)?;
    let down = linear(graph, gated, down_weight)?;
    x = graph.add(x, down)?;
    Ok(DenseLayerNodes {
        output: x,
        keys,
        values,
    })
}

pub(super) fn embedding(
    graph: &mut Graph,
    tokens: NodeId,
    weight: NodeId,
    embedding_dim: usize,
) -> Result<NodeId, Error> {
    let sequence = graph.shape(tokens)?.dims()[0];
    let indices = graph.reshape(tokens, [sequence, 1])?;
    let indices = graph.expand(indices, [sequence, embedding_dim])?;
    graph.gather(weight, indices, 0)
}

pub(super) fn rms_norm(
    graph: &mut Graph,
    input: NodeId,
    weight: NodeId,
    dim: usize,
    eps: f32,
) -> Result<NodeId, Error> {
    if graph.shape(input)?.dims().last().copied() != Some(dim) {
        return Err(Error::InvalidReshape {
            from: graph.shape(input)?.clone(),
            to: Shape::new([dim]),
        });
    }
    let squared = graph.square(input)?;
    let mean = graph.reduce(squared, crate::ReduceKind::Mean, Some(vec![-1]), true)?;
    let epsilon = graph.constant(TensorData::scalar(eps));
    let scale = graph.add(mean, epsilon)?;
    let scale = graph.rsqrt(scale)?;
    let normalized = graph.mul(input, scale)?;
    graph.mul(normalized, weight)
}

pub(super) fn linear(graph: &mut Graph, input: NodeId, weight: NodeId) -> Result<NodeId, Error> {
    let weight = graph.permute(weight, vec![1, 0])?;
    graph.matmul(input, weight)
}

fn heads(
    graph: &mut Graph,
    input: NodeId,
    sequence: usize,
    heads: usize,
    head_dim: usize,
) -> Result<NodeId, Error> {
    let input = graph.reshape(input, [sequence, heads, head_dim])?;
    graph.permute(input, vec![1, 0, 2])
}

fn permute_rope_weight(
    graph: &mut Graph,
    weight: NodeId,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    input_dim: usize,
) -> Result<NodeId, Error> {
    let weight = graph.reshape(weight, [heads, head_dim, input_dim])?;
    let prefix = head_dim - rope_dim;
    let rope = graph.shrink(weight, vec![(0, heads), (prefix, head_dim), (0, input_dim)])?;
    let rope = graph.reshape(rope, [heads, rope_dim / 2, 2, input_dim])?;
    let rope = graph.permute(rope, vec![0, 2, 1, 3])?;
    let rope = graph.reshape(rope, [heads, rope_dim, input_dim])?;
    let weight = if prefix == 0 {
        rope
    } else {
        let unrotated = graph.shrink(weight, vec![(0, heads), (0, prefix), (0, input_dim)])?;
        graph.concat(vec![unrotated, rope], 1)?
    };
    graph.reshape(weight, [heads * head_dim, input_dim])
}

#[allow(clippy::too_many_arguments)]
fn apply_rope(
    graph: &mut Graph,
    input: NodeId,
    heads: usize,
    sequence: usize,
    head_dim: usize,
    rope_dim: usize,
    start_pos: usize,
    theta: f64,
) -> Result<NodeId, Error> {
    let half = rope_dim / 2;
    let first = graph.shrink(input, vec![(0, heads), (0, sequence), (0, half)])?;
    let second = graph.shrink(input, vec![(0, heads), (0, sequence), (half, rope_dim)])?;
    let frequencies = (0..sequence).flat_map(|offset| {
        (0..half).map(move |index| {
            let frequency = 1.0 / theta.powf((2 * index) as f64 / rope_dim as f64);
            (start_pos + offset) as f64 * frequency
        })
    });
    let angles = frequencies.collect::<Vec<_>>();
    let cos = graph.constant(TensorData::from_scalars(
        [sequence, half],
        DType::F32,
        angles.iter().copied().map(|angle| Scalar::F(angle.cos())),
    )?);
    let sin = graph.constant(TensorData::from_scalars(
        [sequence, half],
        DType::F32,
        angles.into_iter().map(|angle| Scalar::F(angle.sin())),
    )?);
    let first_cos = graph.mul(first, cos)?;
    let second_sin = graph.mul(second, sin)?;
    let rotated_first = graph.sub(first_cos, second_sin)?;
    let second_cos = graph.mul(second, cos)?;
    let first_sin = graph.mul(first, sin)?;
    let rotated_second = graph.add(second_cos, first_sin)?;
    let rotated = graph.concat(vec![rotated_first, rotated_second], 2)?;
    if rope_dim == head_dim {
        Ok(rotated)
    } else {
        let tail = graph.shrink(input, vec![(0, heads), (0, sequence), (rope_dim, head_dim)])?;
        graph.concat(vec![rotated, tail], 2)
    }
}
