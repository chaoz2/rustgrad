//! Checked-in BERT embeddings, encoder stack, and base-model composition.

use super::{
    BertEncoderLayer, BertEncoderLayerConfig, Embedding, LayerNorm, Mode, ModeForwardOutput,
    Module, PendingModeEffects, StateKind, regularization::validate_dropout_probability,
    state::join,
};
use crate::{DType, Error, Graph, NodeId, RandomStream, Result, Scalar, Shape, TrainingContext};

/// Static configuration shared by the checked-in BERT base-model composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BertModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub type_vocab_size: usize,
    pub vocab_size: usize,
    pub attention_dropout: f64,
    pub hidden_dropout: f64,
}

impl BertModelConfig {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        hidden_size: usize,
        intermediate_size: usize,
        max_position_embeddings: usize,
        num_attention_heads: usize,
        num_hidden_layers: usize,
        type_vocab_size: usize,
        vocab_size: usize,
        attention_dropout: f64,
        hidden_dropout: f64,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
            type_vocab_size,
            vocab_size,
            attention_dropout,
            hidden_dropout,
        }
    }

    fn layer(self) -> BertEncoderLayerConfig {
        BertEncoderLayerConfig::new(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.attention_dropout,
            self.hidden_dropout,
        )
    }
}

#[derive(Clone, Copy)]
enum EmbeddingDropout {
    Eval,
    Seeded(u64),
    Ambient(RandomStream),
}

/// Word, position, and token-type embeddings followed by LayerNorm and dropout.
pub struct BertEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    hidden_size: usize,
    hidden_dropout: f64,
    dropout_seed: u64,
}

impl BertEmbeddings {
    fn validate_config(config: BertModelConfig) -> Result<()> {
        validate_dropout_probability(config.hidden_dropout)?;
        if config.hidden_size == 0
            || config.vocab_size == 0
            || config.max_position_embeddings == 0
            || config.type_vocab_size == 0
        {
            return Err(Error::InvalidRandom {
                reason: "BERT embedding dimensions must be nonzero",
            });
        }
        for vocabulary in [
            config.vocab_size,
            config.max_position_embeddings,
            config.type_vocab_size,
        ] {
            let shape = Shape::new([vocabulary, config.hidden_size]);
            shape.numel()?;
            vocabulary
                .checked_add(config.hidden_size)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        }
        Shape::new([config.hidden_size]).numel()?;
        Ok(())
    }

    pub fn new_static(config: BertModelConfig, seed: u64) -> Result<Self> {
        Self::validate_config(config)?;
        Ok(Self {
            word_embeddings: Embedding::new_static(
                config.vocab_size,
                config.hidden_size,
                None,
                seed,
            )?,
            position_embeddings: Embedding::new_static(
                config.max_position_embeddings,
                config.hidden_size,
                None,
                seed.wrapping_add(2),
            )?,
            token_type_embeddings: Embedding::new_static(
                config.type_vocab_size,
                config.hidden_size,
                None,
                seed.wrapping_add(4),
            )?,
            layer_norm: LayerNorm::new_static([config.hidden_size], 1e-12, true)?,
            hidden_size: config.hidden_size,
            hidden_dropout: config.hidden_dropout,
            dropout_seed: seed.wrapping_add(6),
        })
    }

    fn output_shape(
        &self,
        graph: &Graph,
        input_ids: NodeId,
        token_type_ids: NodeId,
    ) -> Result<Shape> {
        let input_shape = graph.shape(input_ids)?;
        if input_shape.rank() != 2 {
            return Err(Error::InvalidAttention {
                reason: "BERT input ids must have shape [batch, time]",
            });
        }
        if !graph.dtype(input_ids)?.is_integer() || !graph.dtype(token_type_ids)?.is_integer() {
            return Err(Error::InvalidIndexDType {
                op: "BERT embeddings",
                actual: if !graph.dtype(input_ids)?.is_integer() {
                    graph.dtype(input_ids)?
                } else {
                    graph.dtype(token_type_ids)?
                },
            });
        }
        let token_shape = graph.shape(token_type_ids)?;
        if !token_shape
            .broadcast_with(input_shape)
            .is_ok_and(|shape| shape == *input_shape)
        {
            return Err(Error::InvalidAttention {
                reason: "BERT token-type ids must broadcast to input ids",
            });
        }
        let output = Shape::new([
            input_shape.dims()[0],
            input_shape.dims()[1],
            self.hidden_size,
        ]);
        output.numel()?;
        Ok(output)
    }

    fn ambient_request(&self, output_shape: &Shape) -> Option<(Shape, DType)> {
        (self.hidden_dropout > 0.0 && self.hidden_dropout < 1.0)
            .then(|| (output_shape.clone(), DType::F32))
    }

    fn apply_dropout(
        &self,
        graph: &mut Graph,
        input: NodeId,
        dropout: EmbeddingDropout,
    ) -> Result<NodeId> {
        match dropout {
            EmbeddingDropout::Eval => Ok(input),
            EmbeddingDropout::Seeded(seed) => {
                graph.dropout(input, self.hidden_dropout, true, Some(seed))
            }
            EmbeddingDropout::Ambient(stream) => {
                graph.lower_ambient_dropout(input, self.hidden_dropout, stream)
            }
        }
    }

    fn lower(
        &self,
        graph: &mut Graph,
        input_ids: NodeId,
        token_type_ids: NodeId,
        dropout: EmbeddingDropout,
    ) -> Result<NodeId> {
        let output_shape = self.output_shape(graph, input_ids, token_type_ids)?;
        let input_shape = graph.shape(input_ids)?.clone();
        let time = i64::try_from(input_shape.dims()[1])
            .map_err(|_| Error::ShapeOverflow(input_shape.clone()))?;
        let position_ids = graph.lazy_arange_default_int(0, time, 1)?;
        let position_ids = graph.reshape(position_ids, [1, input_shape.dims()[1]])?;
        let position_ids = graph.expand(position_ids, input_shape)?;
        let words = self.word_embeddings.forward(graph, input_ids)?;
        let positions = self.position_embeddings.forward(graph, position_ids)?;
        let token_types = self.token_type_embeddings.forward(graph, token_type_ids)?;
        let embeddings = graph.add(words, positions)?;
        let embeddings = graph.add(embeddings, token_types)?;
        debug_assert_eq!(graph.shape(embeddings)?, &output_shape);
        let embeddings = self.layer_norm.forward(graph, embeddings)?;
        self.apply_dropout(graph, embeddings, dropout)
    }

    pub fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input_ids: NodeId,
        token_type_ids: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let dropout = match mode {
            Mode::Eval => EmbeddingDropout::Eval,
            Mode::Training => EmbeddingDropout::Seeded(self.dropout_seed),
        };
        let output = self.lower(&mut candidate, input_ids, token_type_ids, dropout)?;
        *graph = candidate;
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }

    pub fn forward_ambient<'a>(
        &'a self,
        graph: &mut Graph,
        input_ids: NodeId,
        token_type_ids: NodeId,
    ) -> Result<ModeForwardOutput<'a>> {
        if !TrainingContext::is_training() {
            return self.forward_mode(graph, input_ids, token_type_ids, Mode::Eval);
        }
        let output_shape = self.output_shape(graph, input_ids, token_type_ids)?;
        let output = if let Some(request) = self.ambient_request(&output_shape) {
            graph.with_implicit_uniform_streams(vec![request], 0, |candidate, streams| {
                let stream = streams.first().copied().ok_or(Error::InvalidRandom {
                    reason: "BERT ambient dropout stream count mismatch",
                })?;
                self.lower(
                    candidate,
                    input_ids,
                    token_type_ids,
                    EmbeddingDropout::Ambient(stream),
                )
            })?
        } else {
            let mut candidate = graph.clone();
            let output = self.lower(
                &mut candidate,
                input_ids,
                token_type_ids,
                EmbeddingDropout::Seeded(self.dropout_seed),
            )?;
            *graph = candidate;
            output
        };
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }
}

impl Module for BertEmbeddings {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &super::Parameter, StateKind)) {
        self.word_embeddings
            .visit(&join(prefix, "word_embeddings"), visitor);
        self.position_embeddings
            .visit(&join(prefix, "position_embeddings"), visitor);
        self.token_type_embeddings
            .visit(&join(prefix, "token_type_embeddings"), visitor);
        self.layer_norm.visit(&join(prefix, "LayerNorm"), visitor);
    }
}

/// Ordered stack of source BERT encoder layers.
pub struct BertEncoder {
    pub layers: Vec<BertEncoderLayer>,
}

impl BertEncoder {
    pub fn new_static(
        layer_config: BertEncoderLayerConfig,
        num_hidden_layers: usize,
        seed: u64,
    ) -> Result<Self> {
        if num_hidden_layers > 0 {
            BertEncoderLayer::validate_config(layer_config)?;
        }
        let layers = (0..num_hidden_layers)
            .map(|index| {
                BertEncoderLayer::new_static(
                    layer_config,
                    seed.wrapping_add((index as u64).wrapping_mul(16)),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn lower_explicit(
        &self,
        graph: &mut Graph,
        mut hidden_states: NodeId,
        attention_mask: NodeId,
        mode: Mode,
    ) -> Result<NodeId> {
        for layer in &self.layers {
            hidden_states = layer.lower_explicit(graph, hidden_states, attention_mask, mode)?;
        }
        Ok(hidden_states)
    }

    fn ambient_requests(&self, hidden_shape: &Shape) -> Result<Vec<(Shape, DType)>> {
        let mut requests = Vec::new();
        for layer in &self.layers {
            requests.extend(layer.ambient_dropout_requests(hidden_shape)?);
        }
        Ok(requests)
    }

    fn lower_ambient_reserved(
        &self,
        graph: &mut Graph,
        mut hidden_states: NodeId,
        attention_mask: NodeId,
        streams: &[RandomStream],
    ) -> Result<NodeId> {
        let hidden_shape = graph.shape(hidden_states)?.clone();
        let mut offset = 0usize;
        for layer in &self.layers {
            let count = layer.ambient_dropout_requests(&hidden_shape)?.len();
            let end = offset.checked_add(count).ok_or(Error::InvalidRandom {
                reason: "BERT ambient dropout stream count overflow",
            })?;
            let selected = streams.get(offset..end).ok_or(Error::InvalidRandom {
                reason: "BERT ambient dropout stream count mismatch",
            })?;
            hidden_states =
                layer.lower_ambient_reserved(graph, hidden_states, attention_mask, selected)?;
            offset = end;
        }
        if offset != streams.len() {
            return Err(Error::InvalidRandom {
                reason: "BERT ambient dropout stream count mismatch",
            });
        }
        Ok(hidden_states)
    }

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

    pub fn forward_ambient<'a>(
        &'a self,
        graph: &mut Graph,
        hidden_states: NodeId,
        attention_mask: NodeId,
    ) -> Result<ModeForwardOutput<'a>> {
        if !TrainingContext::is_training() {
            return self.forward_mode(graph, hidden_states, attention_mask, Mode::Eval);
        }
        let requests = self.ambient_requests(graph.shape(hidden_states)?)?;
        let output = if requests.is_empty() {
            let mut candidate = graph.clone();
            let output = self.lower_explicit(
                &mut candidate,
                hidden_states,
                attention_mask,
                Mode::Training,
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

impl Module for BertEncoder {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &super::Parameter, StateKind)) {
        for (index, layer) in self.layers.iter().enumerate() {
            layer.visit(&join(prefix, &format!("layer.{index}")), visitor);
        }
    }
}

/// Checked-in BERT base model: embeddings, additive mask conversion, and encoder.
pub struct BertModel {
    pub embeddings: BertEmbeddings,
    pub encoder: BertEncoder,
    config: BertModelConfig,
}

impl BertModel {
    pub fn new_static(config: BertModelConfig, seed: u64) -> Result<Self> {
        BertEmbeddings::validate_config(config)?;
        if config.num_hidden_layers > 0 {
            BertEncoderLayer::validate_config(config.layer())?;
        }
        Ok(Self {
            embeddings: BertEmbeddings::new_static(config, seed)?,
            encoder: BertEncoder::new_static(
                config.layer(),
                config.num_hidden_layers,
                seed.wrapping_add(8),
            )?,
            config,
        })
    }

    pub const fn config(&self) -> BertModelConfig {
        self.config
    }

    fn attention_mask(&self, graph: &mut Graph, attention_mask: NodeId) -> Result<NodeId> {
        let mask = graph.unsqueeze(attention_mask, 1)?;
        let mask = graph.unsqueeze(mask, 2)?;
        let mask = graph.scalar_sub(Scalar::F(1.0), mask)?;
        graph.mul_scalar(mask, Scalar::F(-10_000.0))
    }

    fn lower_explicit(
        &self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        token_type_ids: NodeId,
        mode: Mode,
    ) -> Result<NodeId> {
        let mask = self.attention_mask(graph, attention_mask)?;
        let embedding_dropout = match mode {
            Mode::Eval => EmbeddingDropout::Eval,
            Mode::Training => EmbeddingDropout::Seeded(self.embeddings.dropout_seed),
        };
        let hidden = self
            .embeddings
            .lower(graph, input_ids, token_type_ids, embedding_dropout)?;
        self.encoder.lower_explicit(graph, hidden, mask, mode)
    }

    fn ambient_requests(
        &self,
        graph: &Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        token_type_ids: NodeId,
    ) -> Result<Vec<(Shape, DType)>> {
        let mut preflight = graph.clone();
        self.attention_mask(&mut preflight, attention_mask)?;
        let hidden_shape = self
            .embeddings
            .output_shape(graph, input_ids, token_type_ids)?;
        let mut requests = Vec::new();
        if let Some(request) = self.embeddings.ambient_request(&hidden_shape) {
            requests.push(request);
        }
        requests.extend(self.encoder.ambient_requests(&hidden_shape)?);
        Ok(requests)
    }

    fn lower_ambient_reserved(
        &self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        token_type_ids: NodeId,
        streams: &[RandomStream],
    ) -> Result<NodeId> {
        let mask = self.attention_mask(graph, attention_mask)?;
        let hidden_shape = self
            .embeddings
            .output_shape(graph, input_ids, token_type_ids)?;
        let embedding_streams =
            usize::from(self.embeddings.ambient_request(&hidden_shape).is_some());
        let embedding_dropout = if embedding_streams == 1 {
            EmbeddingDropout::Ambient(streams.first().copied().ok_or(Error::InvalidRandom {
                reason: "BERT ambient dropout stream count mismatch",
            })?)
        } else {
            EmbeddingDropout::Seeded(self.embeddings.dropout_seed)
        };
        let hidden = self
            .embeddings
            .lower(graph, input_ids, token_type_ids, embedding_dropout)?;
        let remaining = streams
            .get(embedding_streams..)
            .ok_or(Error::InvalidRandom {
                reason: "BERT ambient dropout stream count mismatch",
            })?;
        self.encoder
            .lower_ambient_reserved(graph, hidden, mask, remaining)
    }

    pub fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        token_type_ids: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let output = self.lower_explicit(
            &mut candidate,
            input_ids,
            attention_mask,
            token_type_ids,
            mode,
        )?;
        *graph = candidate;
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }

    pub fn forward_ambient<'a>(
        &'a self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        token_type_ids: NodeId,
    ) -> Result<ModeForwardOutput<'a>> {
        if !TrainingContext::is_training() {
            return self.forward_mode(graph, input_ids, attention_mask, token_type_ids, Mode::Eval);
        }
        let requests = self.ambient_requests(graph, input_ids, attention_mask, token_type_ids)?;
        let output = if requests.is_empty() {
            let mut candidate = graph.clone();
            let output = self.lower_explicit(
                &mut candidate,
                input_ids,
                attention_mask,
                token_type_ids,
                Mode::Training,
            )?;
            *graph = candidate;
            output
        } else {
            graph.with_implicit_uniform_streams(requests, 0, |candidate, streams| {
                self.lower_ambient_reserved(
                    candidate,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    streams,
                )
            })?
        };
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }
}

impl Module for BertModel {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &super::Parameter, StateKind)) {
        self.embeddings.visit(&join(prefix, "embeddings"), visitor);
        self.encoder.visit(&join(prefix, "encoder"), visitor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend, CapturedReplayExecutor, CapturedReplayOptions, CpuBackend, ModuleStateDict, Op,
        TensorData, manual_seed, nn::CastPolicy,
    };
    use std::collections::BTreeMap;

    fn config(layers: usize, attention_dropout: f64, hidden_dropout: f64) -> BertModelConfig {
        BertModelConfig::new(4, 8, 8, 2, layers, 2, 6, attention_dropout, hidden_dropout)
    }

    fn ids(shape: impl Into<Shape>, values: impl IntoIterator<Item = i64>) -> TensorData {
        TensorData::from_scalars(shape, DType::I32, values.into_iter().map(Scalar::I)).unwrap()
    }

    fn zero_state(model: &BertModel) {
        let tensors = model
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
            model
                .load_state_dict(&ModuleStateDict::from(tensors), true, CastPolicy::Exact,)
                .unwrap()
                .is_clean()
        );
    }

    fn bindings(model: &BertModel, graph: &Graph) -> std::collections::HashMap<String, TensorData> {
        let mut bindings = model.input_bindings(graph).unwrap();
        bindings.insert("input_ids".into(), ids([1, 2], [1, 2]));
        bindings.insert("attention_mask".into(), ids([1, 2], [1, 1]));
        bindings.insert("token_type_ids".into(), ids([1, 2], [0, 1]));
        bindings
    }

    fn random_streams(graph: &Graph) -> Vec<RandomStream> {
        (0..graph.node_count())
            .filter_map(|index| match graph.op(NodeId(index)).unwrap() {
                Op::Random { stream, .. } => Some(*stream),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn bert_model_preserves_source_state_names_stack_geometry_and_capture() {
        let model = BertModel::new_static(config(2, 0.0, 0.0), 11).unwrap();
        zero_state(&model);
        let state = model.state_dict().unwrap();
        assert_eq!(state.tensors().len(), 37);
        for name in [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "embeddings.LayerNorm.weight",
            "embeddings.LayerNorm.bias",
            "encoder.layer.0.attention.self.query.weight",
            "encoder.layer.0.attention.output.LayerNorm.bias",
            "encoder.layer.1.intermediate.dense.weight",
            "encoder.layer.1.output.LayerNorm.weight",
        ] {
            assert!(state.tensors().contains_key(name), "missing state {name}");
        }

        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::I32);
        let token_type_ids = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        let word_weight = model
            .embeddings
            .word_embeddings
            .weight
            .bind(&mut graph)
            .unwrap();
        let forward = model
            .forward_mode(
                &mut graph,
                input_ids,
                attention_mask,
                token_type_ids,
                Mode::Eval,
            )
            .unwrap();
        assert!(forward.pending.is_empty());
        assert_eq!(graph.shape(forward.output).unwrap(), &Shape::new([1, 2, 4]));
        assert_eq!(graph.dtype(forward.output).unwrap(), DType::F32);
        assert_eq!(
            (0..graph.node_count())
                .filter(|index| matches!(graph.op(NodeId(*index)), Ok(Op::Matmul { .. })))
                .count(),
            4,
            "only the two score/value attention products per layer remain raw Matmul"
        );
        assert_eq!(
            (0..graph.node_count())
                .filter(|index| matches!(graph.op(NodeId(*index)), Ok(Op::Gather { .. })))
                .count(),
            3
        );
        assert!((0..graph.node_count()).any(|index| {
            matches!(
                graph.op(NodeId(index)),
                Ok(Op::Constant(data))
                    if data.len() == 1 && data.to_vec_f64()[0] == -10_000.0
            )
        }));

        let loss = graph.sum_all(forward.output).unwrap();
        let gradient = graph.gradient_default(loss, &[word_weight]).unwrap()[0];
        assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([6, 4]));

        let bindings = bindings(&model, &graph);
        let expected = TensorData::zeros([1, 2, 4]).unwrap();
        assert_eq!(
            CpuBackend
                .execute(&graph, forward.output, &bindings)
                .unwrap(),
            expected
        );
        assert!(
            CpuBackend
                .execute(&graph, gradient, &bindings)
                .unwrap()
                .to_vec_f64()
                .iter()
                .all(|value| value.is_finite())
        );

        let schedule = crate::schedule(&graph, forward.output).unwrap();
        assert!(schedule.items.iter().all(|item| item.boundary.is_none()));
        let capture =
            crate::CapturedSchedule::capture(&graph, &schedule, &[forward.output]).unwrap();
        let replay = CapturedReplayExecutor::default()
            .replay(&capture, &bindings, CapturedReplayOptions::default())
            .unwrap();
        assert_eq!(replay.outputs, vec![expected]);
    }

    #[test]
    fn bert_embeddings_and_zero_layer_encoder_keep_source_boundaries() {
        let model = BertModel::new_static(config(0, 0.0, 0.0), 13).unwrap();
        assert!(model.encoder.layers.is_empty());
        assert_eq!(model.state_dict().unwrap().tensors().len(), 5);
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [2, 3], DType::I16);
        let attention_mask = graph.input_dtype("attention_mask", [2, 3], DType::Bool);
        let token_type_ids = graph.input_dtype("token_type_ids", [3], DType::U8);
        let output = model
            .forward_mode(
                &mut graph,
                input_ids,
                attention_mask,
                token_type_ids,
                Mode::Eval,
            )
            .unwrap()
            .output;
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3, 4]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        assert_eq!(
            (0..graph.node_count())
                .filter(|index| matches!(graph.op(NodeId(*index)), Ok(Op::Gather { .. })))
                .count(),
            3
        );

        let mut empty = Graph::new();
        let input_ids = empty.input_dtype("empty_ids", [0, 2], DType::I32);
        let attention_mask = empty.input_dtype("empty_mask", [0, 2], DType::I32);
        let token_type_ids = empty.input_dtype("empty_types", [1, 2], DType::I32);
        let output = model
            .forward_mode(
                &mut empty,
                input_ids,
                attention_mask,
                token_type_ids,
                Mode::Eval,
            )
            .unwrap()
            .output;
        let mut values = model.input_bindings(&empty).unwrap();
        values.insert("empty_ids".into(), ids([0, 2], []));
        values.insert("empty_mask".into(), ids([0, 2], []));
        values.insert("empty_types".into(), ids([1, 2], [0, 0]));
        assert_eq!(
            CpuBackend.execute(&empty, output, &values).unwrap(),
            TensorData::zeros([0, 2, 4]).unwrap()
        );

        let layered = BertModel::new_static(config(1, 0.0, 0.0), 14).unwrap();
        zero_state(&layered);
        let mut zero_time = Graph::new();
        let input_ids = zero_time.input_dtype("zero_time_ids", [1, 0], DType::I32);
        let attention_mask = zero_time.input_dtype("zero_time_mask", [1, 0], DType::I32);
        let token_type_ids = zero_time.input_dtype("zero_time_types", [1, 0], DType::I32);
        let output = layered
            .forward_mode(
                &mut zero_time,
                input_ids,
                attention_mask,
                token_type_ids,
                Mode::Eval,
            )
            .unwrap()
            .output;
        let mut values = layered.input_bindings(&zero_time).unwrap();
        values.insert("zero_time_ids".into(), ids([1, 0], []));
        values.insert("zero_time_mask".into(), ids([1, 0], []));
        values.insert("zero_time_types".into(), ids([1, 0], []));
        assert_eq!(
            CpuBackend.execute(&zero_time, output, &values).unwrap(),
            TensorData::zeros([1, 0, 4]).unwrap()
        );
    }

    #[test]
    fn bert_model_ambient_dropout_is_one_source_ordered_transaction() {
        let _lock = Graph::lock_implicit_random_tests();
        let model = BertModel::new_static(config(2, 0.25, 0.5), 17).unwrap();
        manual_seed(71);
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::I32);
        let token_type_ids = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        let _training = TrainingContext::training();
        let output = model
            .forward_ambient(&mut graph, input_ids, attention_mask, token_type_ids)
            .unwrap();
        assert!(output.pending.is_empty());
        let streams = random_streams(&graph);
        assert_eq!(streams.len(), 7, "embedding then three draws per layer");

        manual_seed(71);
        let mut malformed = Graph::new();
        let input_ids = malformed.input_dtype("input_ids", [1, 2], DType::I32);
        let bad_mask = malformed.input_dtype("attention_mask", [3], DType::I32);
        let token_type_ids = malformed.input_dtype("token_type_ids", [1, 2], DType::I32);
        let before = malformed.node_count();
        assert!(
            model
                .forward_ambient(&mut malformed, input_ids, bad_mask, token_type_ids)
                .is_err()
        );
        assert_eq!(malformed.node_count(), before);

        let valid_mask = malformed.input_dtype("valid_mask", [1, 2], DType::I32);
        model
            .forward_ambient(&mut malformed, input_ids, valid_mask, token_type_ids)
            .unwrap();
        assert_eq!(random_streams(&malformed), streams);
    }

    #[test]
    fn bert_model_eval_and_unit_dropout_reserve_no_ambient_randomness() {
        let _lock = Graph::lock_implicit_random_tests();
        let model = BertModel::new_static(config(1, 1.0, 1.0), 19).unwrap();
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::Bool);
        let token_type_ids = graph.input_dtype("token_type_ids", [], DType::I32);
        {
            let _training = TrainingContext::training();
            model
                .forward_ambient(&mut graph, input_ids, attention_mask, token_type_ids)
                .unwrap();
        }
        assert!(random_streams(&graph).is_empty());

        let mut eval = Graph::new();
        let input_ids = eval.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = eval.input_dtype("attention_mask", [1, 2], DType::Bool);
        let token_type_ids = eval.input_dtype("token_type_ids", [], DType::I32);
        model
            .forward_ambient(&mut eval, input_ids, attention_mask, token_type_ids)
            .unwrap();
        assert!(random_streams(&eval).is_empty());
    }

    #[test]
    fn bert_model_rejects_malformed_descriptors_without_graph_publication() {
        assert!(BertModel::new_static(config(1, 0.0, -0.1), 23).is_err());
        let model = BertModel::new_static(config(1, 0.0, 0.0), 23).unwrap();
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::I32);
        let float_types = graph.input_dtype("token_type_ids", [1, 2], DType::F32);
        let before = graph.node_count();
        assert!(
            model
                .forward_mode(
                    &mut graph,
                    input_ids,
                    attention_mask,
                    float_types,
                    Mode::Eval,
                )
                .is_err()
        );
        assert_eq!(graph.node_count(), before);

        let token_type_ids = graph.input_dtype("valid_types", [1, 2], DType::I32);
        let bad_mask = graph.input_dtype("bad_mask", [3], DType::I32);
        let before = graph.node_count();
        assert!(
            model
                .forward_mode(&mut graph, input_ids, bad_mask, token_type_ids, Mode::Eval,)
                .is_err()
        );
        assert_eq!(graph.node_count(), before);
    }
}
