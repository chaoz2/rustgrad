//! Checked-in BERT pretraining heads and complete model composition.

use super::{
    BertModel, BertModelConfig, LayerNorm, Linear, Mode, Module, Parameter, StateKind,
    bert::bert_gelu, state::join,
};
use crate::{
    DType, Error, Graph, NodeId, Reduction, Result, Scalar, Shape, TensorData, TrainingContext,
};

/// Ordered logits returned by [`BertForPretraining`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BertPretrainingOutput {
    prediction_logits: NodeId,
    seq_relationship_logits: NodeId,
}

impl BertPretrainingOutput {
    pub const fn prediction_logits(self) -> NodeId {
        self.prediction_logits
    }

    pub const fn seq_relationship_logits(self) -> NodeId {
        self.seq_relationship_logits
    }

    pub const fn into_tuple(self) -> (NodeId, NodeId) {
        (self.prediction_logits, self.seq_relationship_logits)
    }
}

/// Ordered metrics returned by [`BertForPretraining::accuracy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BertPretrainingAccuracy {
    masked_lm_accuracy: NodeId,
    next_sentence_accuracy: NodeId,
    masked_lm_loss: NodeId,
    next_sentence_loss: NodeId,
}

impl BertPretrainingAccuracy {
    pub const fn masked_lm_accuracy(self) -> NodeId {
        self.masked_lm_accuracy
    }

    pub const fn next_sentence_accuracy(self) -> NodeId {
        self.next_sentence_accuracy
    }

    pub const fn masked_lm_loss(self) -> NodeId {
        self.masked_lm_loss
    }

    pub const fn next_sentence_loss(self) -> NodeId {
        self.next_sentence_loss
    }

    pub const fn into_tuple(self) -> (NodeId, NodeId, NodeId, NodeId) {
        (
            self.masked_lm_accuracy,
            self.next_sentence_accuracy,
            self.masked_lm_loss,
            self.next_sentence_loss,
        )
    }
}

/// Dense, exact erf-GELU, and LayerNorm transform used by the MLM head.
pub struct BertPredictionHeadTransform {
    pub dense: Linear,
    pub layer_norm: LayerNorm,
}

impl BertPredictionHeadTransform {
    pub fn new_static(hidden_size: usize, seed: u64) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_static(hidden_size, hidden_size, true, seed)?,
            layer_norm: LayerNorm::new_static([hidden_size], 1e-12, true)?,
        })
    }

    fn lower(&self, graph: &mut Graph, hidden_states: NodeId) -> Result<NodeId> {
        let hidden_states = self.dense.forward_source(graph, hidden_states)?;
        let hidden_states = bert_gelu(graph, hidden_states)?;
        self.layer_norm.forward(graph, hidden_states)
    }

    pub fn forward(&self, graph: &mut Graph, hidden_states: NodeId) -> Result<NodeId> {
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, hidden_states)?;
        *graph = candidate;
        Ok(output)
    }
}

impl Module for BertPredictionHeadTransform {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.dense.visit(&join(prefix, "dense"), visitor);
        self.layer_norm.visit(&join(prefix, "LayerNorm"), visitor);
    }
}

/// Masked-language-model projection with a tied word-embedding weight.
pub struct BertLMPredictionHead {
    pub transform: BertPredictionHeadTransform,
    pub embedding_weight: Parameter,
    pub bias: Parameter,
    hidden_size: usize,
    vocab_size: usize,
}

impl BertLMPredictionHead {
    pub fn new_static(
        hidden_size: usize,
        vocab_size: usize,
        embedding_weight: Parameter,
        seed: u64,
    ) -> Result<Self> {
        let expected = Shape::new([vocab_size, hidden_size]);
        let snapshot = embedding_weight.snapshot()?;
        if snapshot.shape != expected {
            return Err(Error::InvalidMatmul {
                lhs: snapshot.shape,
                rhs: expected,
            });
        }
        Ok(Self {
            transform: BertPredictionHeadTransform::new_static(hidden_size, seed)?,
            embedding_weight,
            bias: Parameter::new(TensorData::zeros([vocab_size])?, true),
            hidden_size,
            vocab_size,
        })
    }

    fn lower(&self, graph: &mut Graph, hidden_states: NodeId) -> Result<NodeId> {
        if graph.shape(hidden_states)?.dims().last().copied() != Some(self.hidden_size) {
            return Err(Error::InvalidMatmul {
                lhs: graph.shape(hidden_states)?.clone(),
                rhs: Shape::new([self.hidden_size, self.vocab_size]),
            });
        }
        let hidden_states = self.transform.lower(graph, hidden_states)?;
        let weight = self.embedding_weight.bind(graph)?;
        let weight = graph.permute(weight, [1, 0])?;
        let logits = graph.matmul_tinygrad_default(hidden_states, weight)?;
        let bias = self.bias.bind(graph)?;
        graph.add(logits, bias)
    }

    pub fn forward(&self, graph: &mut Graph, hidden_states: NodeId) -> Result<NodeId> {
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, hidden_states)?;
        *graph = candidate;
        Ok(output)
    }
}

impl Module for BertLMPredictionHead {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.transform.visit(&join(prefix, "transform"), visitor);
        visitor(
            join(prefix, "embedding_weight"),
            &self.embedding_weight,
            StateKind::Parameter,
        );
        visitor(join(prefix, "bias"), &self.bias, StateKind::Parameter);
    }
}

/// Source BERT pooler: first sequence position, dense projection, then tanh.
pub struct BertPooler {
    pub dense: Linear,
    hidden_size: usize,
}

impl BertPooler {
    pub fn new_static(hidden_size: usize, seed: u64) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_static(hidden_size, hidden_size, true, seed)?,
            hidden_size,
        })
    }

    fn lower(&self, graph: &mut Graph, hidden_states: NodeId) -> Result<NodeId> {
        let shape = graph.shape(hidden_states)?.clone();
        if shape.rank() != 3 || shape.dims()[2] != self.hidden_size || shape.dims()[1] == 0 {
            return Err(Error::InvalidAttention {
                reason: "BERT pooler requires [batch, nonempty time, hidden] states",
            });
        }
        let first = graph.shrink(
            hidden_states,
            vec![(0, shape.dims()[0]), (0, 1), (0, self.hidden_size)],
        )?;
        let first = graph.squeeze(first, Some(1))?;
        let pooled = self.dense.forward_source(graph, first)?;
        graph.tanh(pooled)
    }

    pub fn forward(&self, graph: &mut Graph, hidden_states: NodeId) -> Result<NodeId> {
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, hidden_states)?;
        *graph = candidate;
        Ok(output)
    }
}

impl Module for BertPooler {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.dense.visit(&join(prefix, "dense"), visitor);
    }
}

/// MLM and next-sentence prediction heads in checked-in source order.
pub struct BertPreTrainingHeads {
    pub predictions: BertLMPredictionHead,
    pub pooler: BertPooler,
    pub seq_relationship: Linear,
}

impl BertPreTrainingHeads {
    pub fn new_static(
        hidden_size: usize,
        vocab_size: usize,
        embedding_weight: Parameter,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            predictions: BertLMPredictionHead::new_static(
                hidden_size,
                vocab_size,
                embedding_weight,
                seed,
            )?,
            pooler: BertPooler::new_static(hidden_size, seed.wrapping_add(2))?,
            seq_relationship: Linear::new_static(hidden_size, 2, true, seed.wrapping_add(4))?,
        })
    }

    fn gather_masked(
        graph: &mut Graph,
        sequence_output: NodeId,
        masked_lm_positions: NodeId,
    ) -> Result<NodeId> {
        let sequence_shape = graph.shape(sequence_output)?.clone();
        if sequence_shape.rank() != 3 {
            return Err(Error::InvalidAttention {
                reason: "BERT sequence output must have shape [batch, time, hidden]",
            });
        }
        let positions_shape = graph.shape(masked_lm_positions)?.clone();
        if positions_shape.rank() != 2 {
            return Err(Error::InvalidAttention {
                reason: "BERT masked positions must be rank two",
            });
        }
        let time = sequence_shape.dims()[1];
        let end = i64::try_from(time).map_err(|_| Error::ShapeOverflow(sequence_shape.clone()))?;
        let counter = graph.lazy_arange_default_int(0, end, 1)?;
        let counter = graph.reshape(counter, [1, 1, time])?;
        let counter = graph.expand(
            counter,
            [positions_shape.dims()[0], positions_shape.dims()[1], time],
        )?;
        let positions = graph.unsqueeze(masked_lm_positions, 2)?;
        let positions = graph.expand(
            positions,
            [positions_shape.dims()[0], positions_shape.dims()[1], time],
        )?;
        let onehot = graph.eq(counter, positions)?;
        graph.matmul_tinygrad_default(onehot, sequence_output)
    }

    fn lower(
        &self,
        graph: &mut Graph,
        sequence_output: NodeId,
        masked_lm_positions: NodeId,
    ) -> Result<BertPretrainingOutput> {
        let gathered = Self::gather_masked(graph, sequence_output, masked_lm_positions)?;
        let prediction_logits = self.predictions.lower(graph, gathered)?;
        let pooled = self.pooler.lower(graph, sequence_output)?;
        let seq_relationship_logits = self.seq_relationship.forward_source(graph, pooled)?;
        Ok(BertPretrainingOutput {
            prediction_logits,
            seq_relationship_logits,
        })
    }

    pub fn forward(
        &self,
        graph: &mut Graph,
        sequence_output: NodeId,
        masked_lm_positions: NodeId,
    ) -> Result<BertPretrainingOutput> {
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, sequence_output, masked_lm_positions)?;
        *graph = candidate;
        Ok(output)
    }
}

impl Module for BertPreTrainingHeads {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.predictions
            .visit(&join(prefix, "predictions"), visitor);
        self.pooler.visit(&join(prefix, "pooler"), visitor);
        self.seq_relationship
            .visit(&join(prefix, "seq_relationship"), visitor);
    }
}

/// Checked-in BERT base model with tied MLM and next-sentence heads.
///
/// Forward lowering returns `(prediction_logits, seq_relationship_logits)` in
/// source order. Explicit and ambient paths rehearse the entire graph before
/// publication; ambient training reserves exactly the base model's dropout
/// streams. The loss and accuracy methods retain the checked-in model's live
/// masked-weight comparison, denominator residual, and metric ordering.
///
/// This type does not download or translate checkpoints and does not include
/// tokenization, vocabulary files, an optimizer, or a training loop.
pub struct BertForPretraining {
    pub bert: BertModel,
    pub cls: BertPreTrainingHeads,
}

impl BertForPretraining {
    pub fn new_static(config: BertModelConfig, seed: u64) -> Result<Self> {
        BertModel::validate_config(config)?;
        let bert = BertModel::new_static(config, seed)?;
        let tied = bert.embeddings.word_embeddings.weight.clone();
        let cls = BertPreTrainingHeads::new_static(
            config.hidden_size,
            config.vocab_size,
            tied,
            seed.wrapping_sub(2),
        )?;
        Ok(Self { bert, cls })
    }

    pub const fn config(&self) -> BertModelConfig {
        self.bert.config()
    }

    fn lower_explicit(
        &self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        masked_lm_positions: NodeId,
        token_type_ids: NodeId,
        mode: Mode,
    ) -> Result<BertPretrainingOutput> {
        let sequence_output =
            self.bert
                .lower_explicit(graph, input_ids, attention_mask, token_type_ids, mode)?;
        self.cls.lower(graph, sequence_output, masked_lm_positions)
    }

    pub fn forward_mode(
        &self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        masked_lm_positions: NodeId,
        token_type_ids: NodeId,
        mode: Mode,
    ) -> Result<BertPretrainingOutput> {
        let mut candidate = graph.clone();
        let output = self.lower_explicit(
            &mut candidate,
            input_ids,
            attention_mask,
            masked_lm_positions,
            token_type_ids,
            mode,
        )?;
        *graph = candidate;
        Ok(output)
    }

    pub fn forward_ambient(
        &self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        masked_lm_positions: NodeId,
        token_type_ids: NodeId,
    ) -> Result<BertPretrainingOutput> {
        if !TrainingContext::is_training() {
            return self.forward_mode(
                graph,
                input_ids,
                attention_mask,
                masked_lm_positions,
                token_type_ids,
                Mode::Eval,
            );
        }
        let requests =
            self.bert
                .ambient_requests(graph, input_ids, attention_mask, token_type_ids)?;
        if requests.is_empty() {
            return self.forward_mode(
                graph,
                input_ids,
                attention_mask,
                masked_lm_positions,
                token_type_ids,
                Mode::Training,
            );
        }
        graph.with_implicit_uniform_streams(requests, 0, |candidate, streams| {
            let sequence_output = self.bert.lower_ambient_reserved(
                candidate,
                input_ids,
                attention_mask,
                token_type_ids,
                streams,
            )?;
            self.cls
                .lower(candidate, sequence_output, masked_lm_positions)
        })
    }

    fn masked_lm_loss(
        graph: &mut Graph,
        predictions: NodeId,
        labels: NodeId,
        ignore_index: NodeId,
    ) -> Result<NodeId> {
        let prediction_shape = graph.shape(predictions)?.clone();
        if prediction_shape.rank() == 0 {
            return Err(Error::InvalidAttention {
                reason: "BERT MLM logits require a final vocabulary axis",
            });
        }
        let classes = *prediction_shape.dims().last().expect("rank checked");
        let mut label_dims = prediction_shape.dims().to_vec();
        label_dims.pop();
        let label_shape = Shape::new(label_dims);
        if graph.shape(labels)? != &label_shape {
            return Err(Error::InvalidAttention {
                reason: "BERT MLM labels must equal logits without the vocabulary axis",
            });
        }
        let log_probs = graph.log_softmax(predictions, -1, Some(DType::F32))?;
        let loss_mask = graph.ne(labels, ignore_index)?;
        let labels_numel = label_shape.numel()?;
        let class_end =
            i64::try_from(classes).map_err(|_| Error::ShapeOverflow(prediction_shape.clone()))?;
        let y_counter = graph.lazy_arange_default_int(0, class_end, 1)?;
        let y_counter = graph.reshape(y_counter, [1, classes])?;
        let y_counter = graph.expand(y_counter, [labels_numel, classes])?;
        let labels = graph.reshape(labels, [labels_numel, 1])?;
        let labels = graph.expand(labels, [labels_numel, classes])?;
        let y = graph.eq(y_counter, labels)?;
        let mask = graph.reshape(loss_mask, [labels_numel, 1])?;
        let mask = graph.expand(mask, [labels_numel, classes])?;
        let y = graph.mul(y, mask)?;
        let y = graph.reshape(y, prediction_shape)?;
        let weighted = graph.mul(log_probs, y)?;
        let numerator = graph.sum_default(weighted)?;
        let numerator = graph.neg(numerator)?;
        let denominator = graph.sum_default(loss_mask)?;
        let denominator = graph.add_scalar(denominator, Scalar::F(1e-5))?;
        graph.div(numerator, denominator)
    }

    pub fn loss(
        &self,
        graph: &mut Graph,
        output: BertPretrainingOutput,
        masked_lm_ids: NodeId,
        masked_lm_weights: NodeId,
        next_sentence_labels: NodeId,
    ) -> Result<NodeId> {
        let mut candidate = graph.clone();
        let masked_lm_loss = Self::masked_lm_loss(
            &mut candidate,
            output.prediction_logits,
            masked_lm_ids,
            masked_lm_weights,
        )?;
        let next_sentence_loss = candidate.binary_crossentropy_logits(
            output.seq_relationship_logits,
            next_sentence_labels,
            Reduction::Mean,
            None,
        )?;
        let loss = candidate.add(masked_lm_loss, next_sentence_loss)?;
        *graph = candidate;
        Ok(loss)
    }

    pub fn accuracy(
        &self,
        graph: &mut Graph,
        output: BertPretrainingOutput,
        masked_lm_ids: NodeId,
        masked_lm_weights: NodeId,
        next_sentence_labels: NodeId,
    ) -> Result<BertPretrainingAccuracy> {
        let mut candidate = graph.clone();
        let zero = candidate.constant(TensorData::scalar_with_dtype(
            Scalar::I(0),
            candidate.dtype(masked_lm_ids)?,
        ));
        let valid = candidate.ne(masked_lm_ids, zero)?;
        let masked_predictions =
            candidate.argmax_with_axis(output.prediction_logits, Some(-1), false)?;
        let masked_correct = candidate.eq(masked_predictions, masked_lm_ids)?;
        let masked_correct = candidate.mul(masked_correct, valid)?;
        let masked_correct = candidate.cast(masked_correct, DType::F32)?;
        let valid_float = candidate.cast(valid, DType::F32)?;
        let masked_correct = candidate.sum_default(masked_correct)?;
        let valid_float = candidate.sum_default(valid_float)?;
        let masked_lm_accuracy = candidate.div(masked_correct, valid_float)?;
        let masked_lm_loss = Self::masked_lm_loss(
            &mut candidate,
            output.prediction_logits,
            masked_lm_ids,
            masked_lm_weights,
        )?;
        let seq_predictions =
            candidate.argmax_with_axis(output.seq_relationship_logits, Some(-1), false)?;
        let seq_correct = candidate.eq(seq_predictions, next_sentence_labels)?;
        let seq_correct = candidate.cast(seq_correct, DType::F32)?;
        let next_sentence_accuracy = candidate.mean_default(seq_correct)?;
        let next_sentence_loss = candidate.binary_crossentropy_logits(
            output.seq_relationship_logits,
            next_sentence_labels,
            Reduction::Mean,
            None,
        )?;
        let next_sentence_loss = candidate.cast(next_sentence_loss, DType::F32)?;
        let result = BertPretrainingAccuracy {
            masked_lm_accuracy,
            next_sentence_accuracy,
            masked_lm_loss,
            next_sentence_loss,
        };
        *graph = candidate;
        Ok(result)
    }
}

impl Module for BertForPretraining {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.bert.visit(&join(prefix, "bert"), visitor);
        self.cls.visit(&join(prefix, "cls"), visitor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend, CapturedReplayExecutor, CapturedReplayOptions, CpuBackend, ModuleStateDict, Op,
        Storage, nn::CastPolicy,
    };
    use std::collections::{BTreeMap, HashMap};

    fn config(layers: usize, attention_dropout: f64, hidden_dropout: f64) -> BertModelConfig {
        BertModelConfig::new(4, 8, 8, 2, layers, 2, 6, attention_dropout, hidden_dropout)
    }

    fn ids(shape: impl Into<Shape>, values: impl IntoIterator<Item = i64>) -> TensorData {
        TensorData::from_scalars(shape, DType::I32, values.into_iter().map(Scalar::I)).unwrap()
    }

    fn controlled_state(model: &BertForPretraining) {
        let state = model
            .state_dict()
            .unwrap()
            .into_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let value = if name.ends_with("LayerNorm.weight") {
                    TensorData::ones(tensor.shape().clone()).unwrap()
                } else if name == "cls.predictions.bias" {
                    TensorData::new([6], vec![-2.0, 3.0, 1.0, 0.0, -1.0, -3.0]).unwrap()
                } else if name == "cls.seq_relationship.bias" {
                    TensorData::new([2], vec![-1.0, 2.0]).unwrap()
                } else {
                    TensorData::zeros(tensor.shape().clone()).unwrap()
                };
                (name, value)
            })
            .collect::<BTreeMap<_, _>>();
        assert!(
            model
                .load_state_dict(&ModuleStateDict::from(state), true, CastPolicy::Exact)
                .unwrap()
                .is_clean()
        );
    }

    fn bindings(model: &BertForPretraining, graph: &Graph) -> HashMap<String, TensorData> {
        let mut values = model.input_bindings(graph).unwrap();
        values.insert("input_ids".into(), ids([1, 2], [1, 2]));
        values.insert("attention_mask".into(), ids([1, 2], [1, 1]));
        values.insert("masked_lm_positions".into(), ids([1, 2], [1, 0]));
        values.insert("token_type_ids".into(), ids([1, 2], [0, 1]));
        values
    }

    fn random_streams(graph: &Graph) -> Vec<crate::RandomStream> {
        (0..graph.node_count())
            .filter_map(|index| match graph.op(NodeId(index)).unwrap() {
                Op::Random { stream, .. } => Some(*stream),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn bert_pretraining_ties_source_state_and_preserves_forward_tuple() {
        let model = BertForPretraining::new_static(config(1, 0.0, 0.0), 11).unwrap();
        controlled_state(&model);
        assert_eq!(
            model.bert.embeddings.word_embeddings.weight.id(),
            model.cls.predictions.embedding_weight.id(),
            "the MLM projection must share the embedding parameter identity"
        );
        let mut visited = Vec::new();
        model.visit("", &mut |name, parameter, _| {
            visited.push((name, parameter.id()))
        });
        let embedding = visited
            .iter()
            .find(|(name, _)| name == "bert.embeddings.word_embeddings.weight")
            .unwrap();
        let tied = visited
            .iter()
            .find(|(name, _)| name == "cls.predictions.embedding_weight")
            .unwrap();
        assert_eq!(embedding.1, tied.1);
        assert_eq!(visited.last().unwrap().0, "cls.seq_relationship.bias");
        assert!(
            !model
                .state_dict()
                .unwrap()
                .tensors()
                .contains_key("cls.predictions.embedding_weight"),
            "the established state dictionary emits a tied identity at its first name"
        );

        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::I32);
        let masked = graph.input_dtype("masked_lm_positions", [1, 2], DType::I32);
        let types = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        let output = model
            .forward_mode(
                &mut graph,
                input_ids,
                attention_mask,
                masked,
                types,
                Mode::Eval,
            )
            .unwrap();
        assert_eq!(
            graph.shape(output.prediction_logits()).unwrap(),
            &Shape::new([1, 2, 6])
        );
        assert_eq!(
            graph.shape(output.seq_relationship_logits()).unwrap(),
            &Shape::new([1, 2])
        );
        let tied_node = model
            .bert
            .embeddings
            .word_embeddings
            .weight
            .node(&graph)
            .unwrap();
        let forward_loss = graph.sum_all(output.prediction_logits()).unwrap();
        let tied_gradient = graph.grad(forward_loss, tied_node).unwrap();
        assert_eq!(graph.shape(tied_gradient).unwrap(), &Shape::new([6, 4]));
        let values = bindings(&model, &graph);
        let outputs = [output.prediction_logits(), output.seq_relationship_logits()];
        let realized = CpuBackend.execute_many(&graph, &outputs, &values).unwrap();
        assert_eq!(realized.outputs.len(), 2);
        assert_eq!(
            realized.outputs[0],
            TensorData::new([1, 2, 6], [-2.0, 3.0, 1.0, 0.0, -1.0, -3.0].repeat(2),).unwrap()
        );
        assert_eq!(
            realized.outputs[1],
            TensorData::new([1, 2], vec![-1.0, 2.0]).unwrap()
        );

        let schedule = crate::schedule_many(&graph, &outputs).unwrap();
        assert!(schedule.items.iter().all(|item| item.boundary.is_none()));
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &outputs).unwrap();
        let replay = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &values.clone().into_iter().collect(),
                CapturedReplayOptions::default(),
            )
            .unwrap();
        assert_eq!(replay.outputs, realized.outputs);
        assert!(
            CpuBackend
                .execute(&graph, tied_gradient, &values)
                .unwrap()
                .to_vec_f64()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn bert_pretraining_loss_accuracy_and_vjp_retain_source_formulas() {
        let model = BertForPretraining::new_static(config(0, 0.0, 0.0), 13).unwrap();
        let mut graph = Graph::new();
        let prediction_logits = graph.input("prediction_logits", [1, 2, 3]);
        let seq_relationship_logits = graph.input("seq_relationship_logits", [1, 2]);
        let masked_lm_ids = graph.input_dtype("masked_lm_ids", [1, 2], DType::I32);
        let masked_lm_weights = graph.input_dtype("masked_lm_weights", [1, 2], DType::F32);
        let next_sentence_labels = graph.input("next_sentence_labels", [1]);
        let output = BertPretrainingOutput {
            prediction_logits,
            seq_relationship_logits,
        };
        let loss = model
            .loss(
                &mut graph,
                output,
                masked_lm_ids,
                masked_lm_weights,
                next_sentence_labels,
            )
            .unwrap();
        let accuracy = model
            .accuracy(
                &mut graph,
                output,
                masked_lm_ids,
                masked_lm_weights,
                next_sentence_labels,
            )
            .unwrap();
        let gradient = graph.grad(loss, prediction_logits).unwrap();
        assert_eq!(graph.shape(loss).unwrap(), &Shape::new([]));
        assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([1, 2, 3]));
        let values = HashMap::from([
            (
                "prediction_logits".into(),
                TensorData::new([1, 2, 3], vec![0.0, 4.0, -1.0, 3.0, 2.0, 1.0]).unwrap(),
            ),
            (
                "seq_relationship_logits".into(),
                TensorData::new([1, 2], vec![0.0, 2.0]).unwrap(),
            ),
            ("masked_lm_ids".into(), ids([1, 2], [1, 0])),
            (
                "masked_lm_weights".into(),
                TensorData::new([1, 2], vec![0.0, 0.0]).unwrap(),
            ),
            (
                "next_sentence_labels".into(),
                TensorData::new([1], vec![1.0]).unwrap(),
            ),
        ]);
        let requested = [
            accuracy.masked_lm_accuracy(),
            accuracy.next_sentence_accuracy(),
            accuracy.masked_lm_loss(),
            accuracy.next_sentence_loss(),
            loss,
            gradient,
        ];
        // The source metrics deliberately retain raw ArgReduce and the VJP
        // retains its dedicated first-order reduction node. Those are CPU
        // semantic-oracle operations, not scheduled/captured kernels; realize
        // every requested surface without changing either graph composition.
        let realized = requested
            .into_iter()
            .map(|output| CpuBackend.execute(&graph, output, &values))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(realized[0].storage(), &Storage::F32(vec![1.0]));
        assert_eq!(realized[1].storage(), &Storage::F32(vec![1.0]));
        assert!(
            realized[2..5]
                .iter()
                .flat_map(TensorData::to_vec_f64)
                .all(f64::is_finite)
        );
        assert!(realized[5].to_vec_f64().iter().all(|x| x.is_finite()));

        let invalid_labels = graph.input("invalid_next_sentence_labels", [3]);
        let before = graph.node_count();
        assert!(
            model
                .loss(
                    &mut graph,
                    output,
                    masked_lm_ids,
                    masked_lm_weights,
                    invalid_labels,
                )
                .is_err()
        );
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn bert_masked_gather_preserves_repeats_out_of_range_zero_and_vjp() {
        let mut graph = Graph::new();
        let sequence = graph.input("sequence", [1, 3, 2]);
        let positions = graph.input_dtype("positions", [1, 4], DType::I32);
        let gathered =
            BertPreTrainingHeads::gather_masked(&mut graph, sequence, positions).unwrap();
        assert_eq!(graph.shape(gathered).unwrap(), &Shape::new([1, 4, 2]));
        assert!(
            (0..graph.node_count())
                .all(|node| !matches!(graph.op(NodeId(node)).unwrap(), Op::Matmul { .. }))
        );
        let loss = graph.sum_all(gathered).unwrap();
        let gradient = graph.grad(loss, sequence).unwrap();
        let values = HashMap::from([
            (
                "sequence".into(),
                TensorData::new([1, 3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            ),
            ("positions".into(), ids([1, 4], [2, 0, 2, 7])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gathered, &values).unwrap(),
            TensorData::new([1, 4, 2], vec![5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 0.0, 0.0],).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &values).unwrap(),
            TensorData::new([1, 3, 2], vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0]).unwrap()
        );
    }

    #[test]
    fn bert_lm_prediction_uses_source_typed_matmul() {
        let embedding_weight = Parameter::new(
            TensorData::from_scalars(
                [3, 2],
                DType::I16,
                [1, 2, 3, 4, 5, 6].into_iter().map(Scalar::I),
            )
            .unwrap(),
            true,
        );
        let head = BertLMPredictionHead::new_static(2, 3, embedding_weight, 13).unwrap();
        let mut graph = Graph::new();
        let hidden = graph.input_dtype("hidden", [1, 2], DType::I16);
        let logits = head.forward(&mut graph, hidden).unwrap();
        assert_eq!(graph.shape(logits).unwrap(), &Shape::new([1, 3]));
        assert!(
            (0..graph.node_count())
                .all(|node| !matches!(graph.op(NodeId(node)).unwrap(), Op::Matmul { .. }))
        );
        assert!((0..graph.node_count()).any(|node| matches!(
            graph.op(NodeId(node)).unwrap(),
            Op::Reduce {
                kind: crate::ReduceKind::Sum,
                ..
            }
        )));
    }

    #[test]
    fn bert_pretraining_empty_masked_axis_and_malformed_calls_are_atomic() {
        let model = BertForPretraining::new_static(config(0, 0.0, 0.0), 17).unwrap();
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [0, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [0, 2], DType::Bool);
        let masked = graph.input_dtype("masked_lm_positions", [0, 0], DType::I32);
        let types = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        let output = model
            .forward_mode(
                &mut graph,
                input_ids,
                attention_mask,
                masked,
                types,
                Mode::Eval,
            )
            .unwrap();
        assert_eq!(
            graph.shape(output.prediction_logits()).unwrap(),
            &Shape::new([0, 0, 6])
        );
        assert_eq!(
            graph.shape(output.seq_relationship_logits()).unwrap(),
            &Shape::new([0, 2])
        );
        let mut values = model.input_bindings(&graph).unwrap();
        values.insert("input_ids".into(), ids([0, 2], []));
        values.insert(
            "attention_mask".into(),
            TensorData::from_scalars([0, 2], DType::Bool, []).unwrap(),
        );
        values.insert("masked_lm_positions".into(), ids([0, 0], []));
        values.insert("token_type_ids".into(), ids([1, 2], [0, 0]));
        let outputs = [output.prediction_logits(), output.seq_relationship_logits()];
        let realized = CpuBackend.execute_many(&graph, &outputs, &values).unwrap();
        assert_eq!(realized.outputs[0], TensorData::zeros([0, 0, 6]).unwrap());
        assert_eq!(realized.outputs[1], TensorData::zeros([0, 2]).unwrap());

        let mut empty_masked = Graph::new();
        let input_ids = empty_masked.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = empty_masked.input_dtype("attention_mask", [1, 2], DType::Bool);
        let masked = empty_masked.input_dtype("masked_lm_positions", [1, 0], DType::I32);
        let types = empty_masked.input_dtype("token_type_ids", [1, 2], DType::I32);
        let output = model
            .forward_mode(
                &mut empty_masked,
                input_ids,
                attention_mask,
                masked,
                types,
                Mode::Eval,
            )
            .unwrap();
        assert_eq!(
            empty_masked.shape(output.prediction_logits()).unwrap(),
            &Shape::new([1, 0, 6])
        );

        let mut zero_time = Graph::new();
        let input_ids = zero_time.input_dtype("input_ids", [1, 0], DType::I32);
        let attention_mask = zero_time.input_dtype("attention_mask", [1, 0], DType::Bool);
        let masked = zero_time.input_dtype("masked_lm_positions", [1, 0], DType::I32);
        let types = zero_time.input_dtype("token_type_ids", [1, 0], DType::I32);
        let before = zero_time.node_count();
        assert!(
            model
                .forward_mode(
                    &mut zero_time,
                    input_ids,
                    attention_mask,
                    masked,
                    types,
                    Mode::Eval,
                )
                .is_err()
        );
        assert_eq!(zero_time.node_count(), before);

        let mut malformed = Graph::new();
        let input_ids = malformed.input_dtype("input_ids", [1, 2], DType::I32);
        let bad_mask = malformed.input_dtype("bad_mask", [3], DType::Bool);
        let bad_positions = malformed.input_dtype("bad_positions", [2], DType::I32);
        let types = malformed.input_dtype("token_type_ids", [1, 2], DType::I32);
        let before = malformed.node_count();
        assert!(
            model
                .forward_mode(
                    &mut malformed,
                    input_ids,
                    bad_mask,
                    bad_positions,
                    types,
                    Mode::Eval,
                )
                .is_err()
        );
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn bert_pretraining_ambient_dropout_failure_retries_same_streams() {
        let _lock = Graph::lock_implicit_random_tests();
        let model = BertForPretraining::new_static(config(1, 0.25, 0.5), 19).unwrap();
        Graph::manual_seed(71);
        let _training = TrainingContext::training();
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::Bool);
        let masked = graph.input_dtype("masked_lm_positions", [1, 1], DType::I32);
        let types = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        model
            .forward_ambient(&mut graph, input_ids, attention_mask, masked, types)
            .unwrap();
        let expected = random_streams(&graph);
        assert_eq!(expected.len(), 4);

        Graph::manual_seed(71);
        let mut retry = Graph::new();
        let input_ids = retry.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = retry.input_dtype("attention_mask", [1, 2], DType::Bool);
        let bad_positions = retry.input_dtype("bad_positions", [1], DType::I32);
        let types = retry.input_dtype("token_type_ids", [1, 2], DType::I32);
        let before = retry.node_count();
        assert!(
            model
                .forward_ambient(&mut retry, input_ids, attention_mask, bad_positions, types,)
                .is_err()
        );
        assert_eq!(retry.node_count(), before);
        let masked = retry.input_dtype("masked_lm_positions", [1, 1], DType::I32);
        model
            .forward_ambient(&mut retry, input_ids, attention_mask, masked, types)
            .unwrap();
        assert_eq!(random_streams(&retry), expected);
    }
}
