//! Source-faithful static BERT encoder-layer composition.

use super::{
    LayerNorm, Linear, Mode, ModeForwardOutput, Module, Parameter, PendingModeEffects, StateKind,
    regularization::validate_dropout_probability, state::join,
};
use crate::{
    AmbientAttentionOptions, AttentionOptions, DType, Error, Graph, NodeId, RandomStream, Result,
    Scalar, Shape, TrainingContext,
};

// Checked-in BERT uses the deliberately rounded source literal, not SQRT_2.
const BERT_GELU_ERF_DIVISOR: f32 = f32::from_bits(0x3fb5_04d5);

/// Static dimensions and dropout controls for one BERT encoder layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BertEncoderLayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub attention_dropout: f64,
    pub hidden_dropout: f64,
}

impl BertEncoderLayerConfig {
    pub const fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        attention_dropout: f64,
        hidden_dropout: f64,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
        }
    }
}

#[derive(Clone, Copy)]
enum BertDropout {
    Eval,
    Seeded([u64; 3]),
    Ambient([Option<RandomStream>; 3]),
}

struct BertGeometry {
    batch: usize,
    time: usize,
    attention_head_size: usize,
    all_head_size: usize,
}

/// One checked-in tinygrad BERT encoder layer.
///
/// The layer accepts `[batch, time, hidden]` states and one Bool or numeric
/// attention mask broadcastable to `[batch, heads, time, time]`.
/// It preserves source post-LayerNorm ordering, the original BERT erf-GELU
/// constant, and the intermediate contiguous boundary. Explicit mode uses
/// deterministic module-owned seeds; [`Self::forward_ambient`] reads the
/// scoped [`TrainingContext`] and reserves all active dropout streams in one
/// graph transaction.
///
/// This is one encoder layer, not a tokenizer, embedding stack, pooler,
/// checkpoint-name translator, classifier head, or complete BERT model.
pub struct BertEncoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_dense: Linear,
    attention_norm: LayerNorm,
    intermediate_dense: Linear,
    output_dense: Linear,
    output_norm: LayerNorm,
    config: BertEncoderLayerConfig,
    dropout_seeds: [u64; 3],
}

impl BertEncoderLayer {
    pub(crate) fn validate_config(config: BertEncoderLayerConfig) -> Result<()> {
        if config.hidden_size == 0
            || config.intermediate_size == 0
            || config.num_attention_heads == 0
        {
            return Err(Error::InvalidAttention {
                reason: "BERT dimensions must be nonzero",
            });
        }
        validate_dropout_probability(config.attention_dropout)?;
        validate_dropout_probability(config.hidden_dropout)
    }

    /// Creates graph-independent deterministic parameters for one encoder layer.
    pub fn new_static(config: BertEncoderLayerConfig, seed: u64) -> Result<Self> {
        Self::validate_config(config)?;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        Ok(Self {
            query: Linear::new_static(config.hidden_size, all_head_size, true, seed)?,
            key: Linear::new_static(
                config.hidden_size,
                all_head_size,
                true,
                seed.wrapping_add(2),
            )?,
            value: Linear::new_static(
                config.hidden_size,
                all_head_size,
                true,
                seed.wrapping_add(4),
            )?,
            attention_dense: Linear::new_static(
                config.hidden_size,
                config.hidden_size,
                true,
                seed.wrapping_add(6),
            )?,
            attention_norm: LayerNorm::new_static([config.hidden_size], 1e-12, true)?,
            intermediate_dense: Linear::new_static(
                config.hidden_size,
                config.intermediate_size,
                true,
                seed.wrapping_add(8),
            )?,
            output_dense: Linear::new_static(
                config.intermediate_size,
                config.hidden_size,
                true,
                seed.wrapping_add(10),
            )?,
            output_norm: LayerNorm::new_static([config.hidden_size], 1e-12, true)?,
            config,
            dropout_seeds: [
                seed.wrapping_add(12),
                seed.wrapping_add(13),
                seed.wrapping_add(14),
            ],
        })
    }

    pub const fn config(&self) -> BertEncoderLayerConfig {
        self.config
    }

    fn project(&self, graph: &mut Graph, projection: &Linear, input: NodeId) -> Result<NodeId> {
        projection.forward_source(graph, input)
    }

    fn geometry(
        &self,
        graph: &Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
    ) -> Result<BertGeometry> {
        let shape = graph.shape(hidden_states)?;
        if shape.rank() != 3 || shape.dims()[2] != self.config.hidden_size {
            return Err(Error::InvalidAttention {
                reason: "BERT hidden states must have shape [batch, time, hidden]",
            });
        }
        let batch = shape.dims()[0];
        let time = shape.dims()[1];
        let attention_head_size = self.config.hidden_size / self.config.num_attention_heads;
        let all_head_size = self.config.num_attention_heads * attention_head_size;
        let score_shape = Shape::new([batch, self.config.num_attention_heads, time, time]);
        score_shape.numel()?;
        let mask_shape = graph.shape(attention_mask)?;
        if !mask_shape
            .broadcast_with(&score_shape)
            .is_ok_and(|shape| shape == score_shape)
        {
            return Err(Error::InvalidAttention {
                reason: "BERT attention mask must broadcast to attention scores",
            });
        }
        Ok(BertGeometry {
            batch,
            time,
            attention_head_size,
            all_head_size,
        })
    }

    fn active_dropout_slots(&self) -> [bool; 3] {
        [
            self.config.attention_dropout > 0.0 && self.config.attention_dropout < 1.0,
            self.config.hidden_dropout > 0.0 && self.config.hidden_dropout < 1.0,
            self.config.hidden_dropout > 0.0 && self.config.hidden_dropout < 1.0,
        ]
    }

    pub(crate) fn ambient_dropout_requests(
        &self,
        hidden_shape: &Shape,
    ) -> Result<Vec<(Shape, DType)>> {
        if hidden_shape.rank() != 3 || hidden_shape.dims()[2] != self.config.hidden_size {
            return Err(Error::InvalidAttention {
                reason: "BERT hidden states must have shape [batch, time, hidden]",
            });
        }
        let batch = hidden_shape.dims()[0];
        let time = hidden_shape.dims()[1];
        let shapes = [
            Shape::new([batch, self.config.num_attention_heads, time, time]),
            hidden_shape.clone(),
            hidden_shape.clone(),
        ];
        shapes[0].numel()?;
        Ok(self
            .active_dropout_slots()
            .into_iter()
            .zip(shapes)
            .filter(|(active, _)| *active)
            .map(|(_, shape)| (shape, DType::F32))
            .collect())
    }

    pub(crate) fn lower_explicit(
        &self,
        graph: &mut Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
        mode: Mode,
    ) -> Result<NodeId> {
        let dropout = match mode {
            Mode::Eval => BertDropout::Eval,
            Mode::Training => BertDropout::Seeded(self.dropout_seeds),
        };
        self.lower(graph, hidden_states, attention_mask, dropout)
    }

    pub(crate) fn lower_ambient_reserved(
        &self,
        graph: &mut Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
        streams: &[RandomStream],
    ) -> Result<NodeId> {
        let active = self.active_dropout_slots();
        if streams.len() != active.into_iter().filter(|active| *active).count() {
            return Err(Error::InvalidRandom {
                reason: "BERT ambient dropout stream count mismatch",
            });
        }
        let mut next = streams.iter().copied();
        let mut reserved = [None; 3];
        for (slot, active) in active.into_iter().enumerate() {
            if active {
                reserved[slot] = next.next();
            }
        }
        self.lower(
            graph,
            hidden_states,
            attention_mask,
            BertDropout::Ambient(reserved),
        )
    }

    fn heads(
        &self,
        graph: &mut Graph,
        projection: &Linear,
        input: NodeId,
        geometry: &BertGeometry,
    ) -> Result<NodeId> {
        let projected = self.project(graph, projection, input)?;
        let projected = graph.reshape(
            projected,
            [
                geometry.batch,
                geometry.time,
                self.config.num_attention_heads,
                geometry.attention_head_size,
            ],
        )?;
        graph.permute(projected, [0, 2, 1, 3])
    }

    fn apply_hidden_dropout(
        &self,
        graph: &mut Graph,
        input: NodeId,
        slot: usize,
        dropout: BertDropout,
    ) -> Result<NodeId> {
        match dropout {
            BertDropout::Eval => Ok(input),
            BertDropout::Seeded(seeds) => {
                graph.dropout(input, self.config.hidden_dropout, true, Some(seeds[slot]))
            }
            BertDropout::Ambient(streams) => match self.config.hidden_dropout {
                0.0 => Ok(input),
                1.0 => graph.zeros_like(input, None),
                _ => graph.lower_ambient_dropout(
                    input,
                    self.config.hidden_dropout,
                    streams[slot].expect("active BERT dropout stream was reserved"),
                ),
            },
        }
    }

    fn attention(
        &self,
        graph: &mut Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
        geometry: &BertGeometry,
        dropout: BertDropout,
    ) -> Result<NodeId> {
        let query = self.heads(graph, &self.query, hidden_states, geometry)?;
        let key = self.heads(graph, &self.key, hidden_states, geometry)?;
        let value = self.heads(graph, &self.value, hidden_states, geometry)?;
        let attended = match dropout {
            BertDropout::Eval => graph.scaled_dot_product_attention(
                query,
                key,
                value,
                Some(attention_mask),
                AttentionOptions {
                    dropout_p: self.config.attention_dropout,
                    ..AttentionOptions::default()
                },
            )?,
            BertDropout::Seeded(seeds) => graph.scaled_dot_product_attention(
                query,
                key,
                value,
                Some(attention_mask),
                AttentionOptions {
                    dropout_p: self.config.attention_dropout,
                    training: true,
                    dropout_seed: Some(seeds[0]),
                    ..AttentionOptions::default()
                },
            )?,
            BertDropout::Ambient(streams)
                if self.config.attention_dropout > 0.0 && self.config.attention_dropout < 1.0 =>
            {
                graph.scaled_dot_product_attention_with_stream(
                    query,
                    key,
                    value,
                    Some(attention_mask),
                    AmbientAttentionOptions {
                        dropout_p: self.config.attention_dropout,
                        ..AmbientAttentionOptions::default()
                    },
                    streams[0].expect("active BERT attention stream was reserved"),
                )?
            }
            BertDropout::Ambient(_) => graph.scaled_dot_product_attention(
                query,
                key,
                value,
                Some(attention_mask),
                AttentionOptions {
                    dropout_p: self.config.attention_dropout,
                    training: true,
                    ..AttentionOptions::default()
                },
            )?,
        };
        let attended = graph.permute(attended, [0, 2, 1, 3])?;
        let attended = graph.reshape(
            attended,
            [geometry.batch, geometry.time, geometry.all_head_size],
        )?;
        self.project(graph, &self.attention_dense, attended)
    }

    fn bert_gelu(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let half = graph.mul_scalar(input, Scalar::F(0.5))?;
        let scaled = graph.div_scalar(input, Scalar::F(f64::from(BERT_GELU_ERF_DIVISOR)))?;
        let error = graph.erf(scaled)?;
        let shifted = graph.add_scalar(error, Scalar::F(1.0))?;
        let activated = graph.mul(half, shifted)?;
        graph.contiguous(activated)
    }

    fn lower(
        &self,
        graph: &mut Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
        dropout: BertDropout,
    ) -> Result<NodeId> {
        let geometry = self.geometry(graph, hidden_states, attention_mask)?;
        let attended = self.attention(graph, hidden_states, attention_mask, &geometry, dropout)?;
        let attended = self.apply_hidden_dropout(graph, attended, 1, dropout)?;
        let attention_residual = graph.add(attended, hidden_states)?;
        let attention_output = self.attention_norm.forward(graph, attention_residual)?;

        let intermediate = self.project(graph, &self.intermediate_dense, attention_output)?;
        let intermediate = self.bert_gelu(graph, intermediate)?;
        let output = self.project(graph, &self.output_dense, intermediate)?;
        let output = self.apply_hidden_dropout(graph, output, 2, dropout)?;
        let output = graph.add(output, attention_output)?;
        self.output_norm.forward(graph, output)
    }

    /// Composes the layer under an explicit deterministic mode.
    pub fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let output = self.lower_explicit(&mut candidate, hidden_states, attention_mask, mode)?;
        *graph = candidate;
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }

    /// Composes the layer under the scoped ambient training context.
    pub fn forward_ambient<'a>(
        &'a self,
        graph: &mut Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
    ) -> Result<ModeForwardOutput<'a>> {
        self.geometry(graph, hidden_states, attention_mask)?;
        if !TrainingContext::is_training() {
            return self.forward_mode(graph, hidden_states, attention_mask, Mode::Eval);
        }
        let requests = self.ambient_dropout_requests(graph.shape(hidden_states)?)?;
        let output = if requests.is_empty() {
            let mut candidate = graph.clone();
            let output = self.lower(
                &mut candidate,
                hidden_states,
                attention_mask,
                BertDropout::Seeded(self.dropout_seeds),
            )?;
            *graph = candidate;
            output
        } else {
            graph.with_implicit_uniform_streams(requests, 0, |candidate, streams| {
                self.lower_ambient_reserved(candidate, hidden_states, attention_mask, streams)
            })?
        };
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }
}

impl Module for BertEncoderLayer {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.query
            .visit(&join(prefix, "attention.self.query"), visitor);
        self.key.visit(&join(prefix, "attention.self.key"), visitor);
        self.value
            .visit(&join(prefix, "attention.self.value"), visitor);
        self.attention_dense
            .visit(&join(prefix, "attention.output.dense"), visitor);
        self.attention_norm
            .visit(&join(prefix, "attention.output.LayerNorm"), visitor);
        self.intermediate_dense
            .visit(&join(prefix, "intermediate.dense"), visitor);
        self.output_dense
            .visit(&join(prefix, "output.dense"), visitor);
        self.output_norm
            .visit(&join(prefix, "output.LayerNorm"), visitor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend, CapturedReplayExecutor, CapturedReplayOptions, CpuBackend, ModuleStateDict, Op,
        TensorData, nn::CastPolicy,
    };
    use std::collections::BTreeMap;

    fn config(attention_dropout: f64, hidden_dropout: f64) -> BertEncoderLayerConfig {
        BertEncoderLayerConfig::new(4, 8, 2, attention_dropout, hidden_dropout)
    }

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    fn bool_data(shape: impl Into<Shape>, values: &[bool]) -> TensorData {
        TensorData::from_scalars(shape, DType::Bool, values.iter().copied().map(Scalar::Bool))
            .unwrap()
    }

    fn random_streams(graph: &Graph) -> Vec<RandomStream> {
        (0..graph.node_count())
            .filter_map(|index| match graph.op(NodeId(index)).unwrap() {
                Op::Random { stream, .. } => Some(*stream),
                _ => None,
            })
            .collect()
    }

    fn zero_projections(layer: &BertEncoderLayer) {
        let tensors = layer
            .state_dict()
            .unwrap()
            .into_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let replacement = if name.ends_with("LayerNorm.weight") {
                    TensorData::ones(tensor.shape().clone()).unwrap()
                } else {
                    TensorData::zeros(tensor.shape().clone()).unwrap()
                };
                (name, replacement)
            })
            .collect::<BTreeMap<_, _>>();
        assert!(
            layer
                .load_state_dict(&ModuleStateDict::from(tensors), true, CastPolicy::Exact,)
                .unwrap()
                .is_clean()
        );
    }

    fn identity_attention(layer: &BertEncoderLayer) {
        let tensors = layer
            .state_dict()
            .unwrap()
            .into_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let replacement = if name.ends_with("LayerNorm.weight") {
                    TensorData::ones(tensor.shape().clone()).unwrap()
                } else if matches!(
                    name.as_str(),
                    "attention.self.query.weight"
                        | "attention.self.key.weight"
                        | "attention.self.value.weight"
                        | "attention.output.dense.weight"
                ) {
                    let mut values = vec![0.0; 16];
                    for diagonal in 0..4 {
                        values[diagonal * 4 + diagonal] = 1.0;
                    }
                    TensorData::new([4, 4], values).unwrap()
                } else {
                    TensorData::zeros(tensor.shape().clone()).unwrap()
                };
                (name, replacement)
            })
            .collect::<BTreeMap<_, _>>();
        assert!(
            layer
                .load_state_dict(&ModuleStateDict::from(tensors), true, CastPolicy::Exact,)
                .unwrap()
                .is_clean()
        );
    }

    #[test]
    fn bert_encoder_layer_validates_config_and_declares_source_state() {
        assert!(BertEncoderLayer::new_static(config(0.1, 0.1), 1).is_ok());
        assert!(matches!(
            BertEncoderLayer::new_static(config(f64::NAN, 0.1), 1),
            Err(Error::UnsupportedDropout { .. })
        ));

        let layer = crate::BertEncoderLayer::new_static(config(0.1, 0.2), 3).unwrap();
        assert_eq!(layer.config(), config(0.1, 0.2));
        assert_eq!(
            layer
                .state_dict()
                .unwrap()
                .tensors()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec![
                "attention.output.LayerNorm.bias",
                "attention.output.LayerNorm.weight",
                "attention.output.dense.bias",
                "attention.output.dense.weight",
                "attention.self.key.bias",
                "attention.self.key.weight",
                "attention.self.query.bias",
                "attention.self.query.weight",
                "attention.self.value.bias",
                "attention.self.value.weight",
                "intermediate.dense.bias",
                "intermediate.dense.weight",
                "output.LayerNorm.bias",
                "output.LayerNorm.weight",
                "output.dense.bias",
                "output.dense.weight",
            ]
        );

        // Checked-in BERT floors the per-head size instead of rejecting the
        // constructor. Its hidden-sized output projection then rejects this
        // non-divisible forward without publishing the rehearsed graph.
        let nondivisible =
            BertEncoderLayer::new_static(BertEncoderLayerConfig::new(5, 8, 2, 0.0, 0.0), 5)
                .unwrap();
        assert_eq!(
            nondivisible.state_dict().unwrap().tensors()["attention.self.query.weight"].shape(),
            &Shape::new([4, 5])
        );
        let mut graph = Graph::new();
        let hidden = graph.input("hidden", [1, 2, 5]);
        let mask = graph.input("mask", [1, 1, 1, 2]);
        let before = graph.node_count();
        assert!(matches!(
            nondivisible.forward_mode(&mut graph, hidden, mask, Mode::Eval),
            Err(Error::InvalidMatmul { .. })
        ));
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn bert_encoder_layer_preserves_post_norm_gelu_mask_and_gradients() {
        let layer = BertEncoderLayer::new_static(config(0.0, 0.0), 7).unwrap();
        zero_projections(&layer);
        let mut graph = Graph::new();
        let hidden = graph.input("hidden", [2, 2, 4]);
        let mask = graph.input("mask", [2, 1, 1, 2]);
        let first = layer.attention_norm.forward(&mut graph, hidden).unwrap();
        let expected = layer.output_norm.forward(&mut graph, first).unwrap();
        let query_weight = layer.query.weight.bind(&mut graph).unwrap();
        let forward = layer
            .forward_mode(&mut graph, hidden, mask, Mode::Eval)
            .unwrap();
        assert!(forward.pending.is_empty());
        let output = forward.output;
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 2, 4]));
        assert_eq!(
            (0..graph.node_count())
                .filter(|index| matches!(graph.op(NodeId(*index)), Ok(Op::Matmul { .. })))
                .count(),
            2,
            "only attention's score/value products remain raw Matmul"
        );
        assert!((0..graph.node_count()).any(|index| {
            let Ok(Op::Contiguous { input }) = graph.op(NodeId(index)) else {
                return false;
            };
            matches!(graph.op(*input), Ok(Op::Binary { .. }))
        }));
        assert!((0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId(index)),
                Ok(Op::Unary {
                    op: crate::UnaryOp::Erf,
                    ..
                })
            )
        }));
        assert!((0..graph.node_count()).any(|index| {
            matches!(
                graph.op(NodeId(index)),
                Ok(Op::Unary {
                    op: crate::UnaryOp::Sign,
                    ..
                })
            )
        }));
        assert!((0..graph.node_count()).any(|index| {
            matches!(
                graph.op(NodeId(index)),
                Ok(Op::Constant(data))
                    if data.len() == 1
                        && data.to_vec_f64()[0] == f64::from(BERT_GELU_ERF_DIVISOR)
            )
        }));
        let squared = graph.square(output).unwrap();
        let loss = graph.sum_all(squared).unwrap();
        let gradients = graph
            .gradient_default(loss, &[hidden, query_weight])
            .unwrap();
        assert!(
            gradients
                .iter()
                .all(|gradient| !matches!(graph.op(*gradient), Ok(Op::Constant(_))))
        );
        let mut bindings = layer.input_bindings(&graph).unwrap();
        bindings.insert(
            "hidden".into(),
            data(
                [2, 2, 4],
                &[
                    1.0, -2.0, 3.0, 0.5, -1.0, 4.0, 2.0, -3.0, 0.5, 1.5, -2.5, 3.5, 4.0, -1.0, 0.0,
                    2.0,
                ],
            ),
        );
        bindings.insert(
            "mask".into(),
            data([2, 1, 1, 2], &[0.0, -10000.0, -10000.0, 0.0]),
        );
        let output_data = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(
            output_data,
            CpuBackend.execute(&graph, expected, &bindings).unwrap()
        );
        for gradient in gradients {
            assert!(
                CpuBackend
                    .execute(&graph, gradient, &bindings)
                    .unwrap()
                    .to_vec_f64()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }

        let schedule = crate::schedule(&graph, output).unwrap();
        assert!(schedule.items.iter().all(|item| item.boundary.is_none()));
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let replay = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &bindings.into_iter().collect(),
                CapturedReplayOptions::default(),
            )
            .unwrap();
        assert_eq!(replay.outputs, vec![output_data]);
    }

    #[test]
    fn bert_encoder_layer_uses_the_additive_attention_mask() {
        let layer = BertEncoderLayer::new_static(config(0.0, 0.0), 9).unwrap();
        identity_attention(&layer);
        let mut graph = Graph::new();
        let hidden = graph.input("hidden", [1, 2, 4]);
        let mask = graph.input("mask", [1, 1, 1, 2]);
        let output = layer
            .forward_mode(&mut graph, hidden, mask, Mode::Eval)
            .unwrap()
            .output;
        let mut bindings = layer.input_bindings(&graph).unwrap();
        bindings.insert(
            "hidden".into(),
            data([1, 2, 4], &[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        );
        bindings.insert("mask".into(), data([1, 1, 1, 2], &[0.0, 0.0]));
        let unmasked = CpuBackend.execute(&graph, output, &bindings).unwrap();
        bindings.insert("mask".into(), data([1, 1, 1, 2], &[0.0, -10000.0]));
        let masked = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_ne!(unmasked, masked);
    }

    #[test]
    fn bert_encoder_layer_promotes_nonfloat_hidden_and_accepts_bool_mask() {
        let layer = BertEncoderLayer::new_static(config(0.0, 0.0), 10).unwrap();
        zero_projections(&layer);
        let mut graph = Graph::new();
        let hidden = graph.input_dtype("hidden", [1, 2, 4], DType::I32);
        let mask = graph.input_dtype("mask", [1, 1, 1, 2], DType::Bool);
        let output = layer
            .forward_mode(&mut graph, hidden, mask, Mode::Eval)
            .unwrap()
            .output;
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([1, 2, 4]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        assert_eq!(
            (0..graph.node_count())
                .filter(|index| matches!(graph.op(NodeId(*index)), Ok(Op::Matmul { .. })))
                .count(),
            2,
            "source Linear projections must remain Dot compositions"
        );

        let mut bindings = layer.input_bindings(&graph).unwrap();
        bindings.insert(
            "hidden".into(),
            TensorData::from_scalars(
                [1, 2, 4],
                DType::I32,
                [1, 2, 3, 4, -4, -3, -2, -1].into_iter().map(Scalar::I),
            )
            .unwrap(),
        );
        bindings.insert("mask".into(), bool_data([1, 1, 1, 2], &[true, true]));
        assert!(
            CpuBackend
                .execute(&graph, output, &bindings)
                .unwrap()
                .to_vec_f64()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn bert_encoder_layer_ambient_dropout_is_ordered_and_atomic() {
        let _random_lock = Graph::lock_implicit_random_tests();
        Graph::manual_seed(31);
        let layer = BertEncoderLayer::new_static(config(0.5, 0.5), 11).unwrap();
        let mut graph = Graph::new();
        let hidden = graph.input("hidden", [1, 2, 4]);
        let mask = graph.input("mask", [1, 1, 1, 2]);

        layer.forward_ambient(&mut graph, hidden, mask).unwrap();
        assert!(random_streams(&graph).is_empty());
        let before = graph.node_count();
        let malformed_mask = graph.input("bad_mask", [3]);
        let after_input = graph.node_count();
        {
            let _training = TrainingContext::training();
            assert!(matches!(
                layer.forward_ambient(&mut graph, hidden, malformed_mask),
                Err(Error::InvalidAttention { .. })
            ));
        }
        assert_eq!(graph.node_count(), after_input);
        assert_eq!(after_input, before + 1);

        let output = {
            let _training = TrainingContext::training();
            let forward = layer.forward_ambient(&mut graph, hidden, mask).unwrap();
            assert!(forward.pending.is_empty());
            forward.output
        };
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([1, 2, 4]));
        let streams = random_streams(&graph);
        assert_eq!(streams.len(), 3);
        assert_eq!(streams[0].counter, [0, 0]);
        assert_eq!(streams[1].counter, [8, 0]);
        assert_eq!(streams[2].counter, [16, 0]);

        let mut explicit_training = Graph::new();
        let explicit_hidden = explicit_training.input("hidden", [1, 2, 4]);
        let explicit_mask = explicit_training.input("mask", [1, 1, 1, 2]);
        layer
            .forward_mode(
                &mut explicit_training,
                explicit_hidden,
                explicit_mask,
                Mode::Training,
            )
            .unwrap();
        assert_eq!(random_streams(&explicit_training).len(), 3);

        let no_draw_layer = BertEncoderLayer::new_static(config(1.0, 1.0), 12).unwrap();
        let mut no_draw_graph = Graph::new();
        let no_draw_hidden = no_draw_graph.input("hidden", [1, 2, 4]);
        let no_draw_mask = no_draw_graph.input("mask", [1, 1, 1, 2]);
        {
            let _training = TrainingContext::training();
            no_draw_layer
                .forward_ambient(&mut no_draw_graph, no_draw_hidden, no_draw_mask)
                .unwrap();
        }
        assert!(random_streams(&no_draw_graph).is_empty());

        let mut explicit_eval = Graph::new();
        let hidden = explicit_eval.input("hidden", [1, 2, 4]);
        let mask = explicit_eval.input("mask", [1, 1, 1, 2]);
        let _training = TrainingContext::training();
        layer
            .forward_mode(&mut explicit_eval, hidden, mask, Mode::Eval)
            .unwrap();
        assert!(random_streams(&explicit_eval).is_empty());
    }

    #[test]
    fn bert_encoder_layer_preserves_empty_batch_without_rng_advance() {
        let _random_lock = Graph::lock_implicit_random_tests();
        Graph::manual_seed(37);
        let layer = BertEncoderLayer::new_static(config(0.5, 0.5), 17).unwrap();

        let mut eval_graph = Graph::new();
        let hidden = eval_graph.input("hidden", [0, 2, 4]);
        let mask = eval_graph.input("mask", [0, 1, 1, 2]);
        let output = layer
            .forward_mode(&mut eval_graph, hidden, mask, Mode::Eval)
            .unwrap()
            .output;
        assert_eq!(eval_graph.shape(output).unwrap(), &Shape::new([0, 2, 4]));
        let mut bindings = layer.input_bindings(&eval_graph).unwrap();
        bindings.insert("hidden".into(), TensorData::zeros([0, 2, 4]).unwrap());
        bindings.insert("mask".into(), TensorData::zeros([0, 1, 1, 2]).unwrap());
        assert_eq!(
            CpuBackend.execute(&eval_graph, output, &bindings).unwrap(),
            TensorData::zeros([0, 2, 4]).unwrap()
        );

        let mut training_graph = Graph::new();
        let hidden = training_graph.input("hidden", [0, 2, 4]);
        let mask = training_graph.input("mask", [0, 1, 1, 2]);
        let output = {
            let _training = TrainingContext::training();
            layer
                .forward_ambient(&mut training_graph, hidden, mask)
                .unwrap()
                .output
        };
        assert_eq!(
            training_graph.shape(output).unwrap(),
            &Shape::new([0, 2, 4])
        );
        assert_eq!(random_streams(&training_graph).len(), 3);
        let next = training_graph.rand_implicit([1], DType::F32).unwrap();
        let Op::Random { stream, .. } = training_graph.op(next).unwrap() else {
            panic!("expected ambient random source");
        };
        assert_eq!(stream.counter, [0, 0]);
    }

    #[test]
    fn bert_encoder_layer_rejects_malformed_inputs_without_publication() {
        let layer = BertEncoderLayer::new_static(config(0.0, 0.0), 13).unwrap();
        for (hidden_shape, mask_shape) in [
            (Shape::new([2, 4]), Shape::new([1])),
            (Shape::new([1, 2, 3]), Shape::new([1, 1, 1, 2])),
            (Shape::new([1, 2, 4]), Shape::new([3])),
        ] {
            let mut graph = Graph::new();
            let hidden = graph.input_dtype("hidden", hidden_shape, DType::F32);
            let mask = graph.input_dtype("mask", mask_shape, DType::F32);
            let before = graph.node_count();
            assert!(
                layer
                    .forward_mode(&mut graph, hidden, mask, Mode::Eval)
                    .is_err()
            );
            assert_eq!(graph.node_count(), before);
        }
    }
}
