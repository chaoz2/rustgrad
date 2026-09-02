//! Checked-in BERT question-answering model composition.

use super::{
    BertModel, BertModelConfig, Linear, Mode, ModeForwardOutput, Module, PendingModeEffects,
    StateKind, state::join,
};
use crate::{Error, Graph, NodeId, Result, Shape, TrainingContext};

/// The checked-in tinygrad BERT base model plus its two-logit question-answering head.
///
/// Inputs use the base model's static `[batch, time]` ID and attention-mask
/// contract. The result preserves the source's literal
/// `chunk(2, -1) -> reshape(-1, 1) -> stack()` ordering and therefore has
/// shape `[2, batch * time, 1]`, with start logits before end logits.
/// Explicit and ambient-mode lowering rehearse the whole base-plus-head graph
/// before publication. Ambient training also reserves all BERT dropout draws
/// in that same transaction; the head itself consumes no randomness.
///
/// This composition does not download or translate pretrained checkpoints and
/// does not include tokenization, vocabulary handling, span loss, or decoding.
pub struct BertForQuestionAnswering {
    pub bert: BertModel,
    pub qa_outputs: Linear,
}

impl BertForQuestionAnswering {
    /// Creates graph-independent deterministic parameters for the complete QA model.
    pub fn new_static(config: BertModelConfig, seed: u64) -> Result<Self> {
        BertModel::validate_config(config)?;
        Ok(Self {
            bert: BertModel::new_static(config, seed)?,
            qa_outputs: Linear::new_static(config.hidden_size, 2, true, seed.wrapping_sub(2))?,
        })
    }

    pub const fn config(&self) -> BertModelConfig {
        self.bert.config()
    }

    fn lower_head(&self, graph: &mut Graph, sequence_output: NodeId) -> Result<NodeId> {
        let logits = self.qa_outputs.forward_source(graph, sequence_output)?;
        let chunks = graph.chunk(logits, 2, -1)?;
        let [start_logits, end_logits] = chunks.as_slice() else {
            return Err(Error::InvalidAttention {
                reason: "BERT QA projection must produce two logits",
            });
        };
        let flattened = graph.shape(*start_logits)?.numel()?;
        let start_logits = graph.reshape(*start_logits, Shape::new([flattened, 1]))?;
        let end_logits = graph.reshape(*end_logits, Shape::new([flattened, 1]))?;
        graph.stack_default([start_logits, end_logits])
    }

    fn lower_explicit(
        &self,
        graph: &mut Graph,
        input_ids: NodeId,
        attention_mask: NodeId,
        token_type_ids: NodeId,
        mode: Mode,
    ) -> Result<NodeId> {
        let sequence_output =
            self.bert
                .lower_explicit(graph, input_ids, attention_mask, token_type_ids, mode)?;
        self.lower_head(graph, sequence_output)
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
        let requests =
            self.bert
                .ambient_requests(graph, input_ids, attention_mask, token_type_ids)?;
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
                let sequence_output = self.bert.lower_ambient_reserved(
                    candidate,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    streams,
                )?;
                self.lower_head(candidate, sequence_output)
            })?
        };
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }
}

impl Module for BertForQuestionAnswering {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &super::Parameter, StateKind)) {
        self.bert.visit(&join(prefix, "bert"), visitor);
        self.qa_outputs.visit(&join(prefix, "qa_outputs"), visitor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend, CapturedReplayExecutor, CapturedReplayOptions, CpuBackend, DType, ModuleStateDict,
        Op, Scalar, TensorData, nn::CastPolicy,
    };
    use std::collections::{BTreeMap, HashMap};

    fn config(layers: usize, attention_dropout: f64, hidden_dropout: f64) -> BertModelConfig {
        BertModelConfig::new(4, 8, 8, 2, layers, 2, 6, attention_dropout, hidden_dropout)
    }

    fn ids(shape: impl Into<Shape>, values: impl IntoIterator<Item = i64>) -> TensorData {
        TensorData::from_scalars(shape, DType::I32, values.into_iter().map(Scalar::I)).unwrap()
    }

    fn source_control_state(model: &BertForQuestionAnswering) {
        let tensors = model
            .state_dict()
            .unwrap()
            .into_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let replacement = if name == "qa_outputs.bias" {
                    TensorData::new([2], vec![2.0, -3.0]).unwrap()
                } else if name.ends_with("LayerNorm.weight") {
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

    fn bindings(
        model: &BertForQuestionAnswering,
        graph: &Graph,
        shape: [usize; 2],
    ) -> HashMap<String, TensorData> {
        let mut bindings = model.input_bindings(graph).unwrap();
        let elements = shape[0] * shape[1];
        bindings.insert(
            "input_ids".into(),
            ids(
                shape,
                (0..elements).map(|index| i64::try_from(index % 6).unwrap()),
            ),
        );
        bindings.insert(
            "attention_mask".into(),
            ids(shape, std::iter::repeat_n(1, elements)),
        );
        bindings.insert(
            "token_type_ids".into(),
            ids(shape, std::iter::repeat_n(0, elements)),
        );
        bindings
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
    fn bert_qa_preserves_source_state_output_order_cpu_capture_and_vjp() {
        let model = BertForQuestionAnswering::new_static(config(1, 0.0, 0.0), 11).unwrap();
        source_control_state(&model);

        let mut state_names = Vec::new();
        model.visit("", &mut |name, _, _| state_names.push(name));
        assert_eq!(state_names.len(), 23);
        assert_eq!(
            state_names.first().unwrap(),
            "bert.embeddings.word_embeddings.weight"
        );
        assert_eq!(state_names[state_names.len() - 2], "qa_outputs.weight");
        assert_eq!(state_names[state_names.len() - 1], "qa_outputs.bias");
        for name in [
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.intermediate.dense.bias",
            "bert.encoder.layer.0.output.LayerNorm.weight",
            "qa_outputs.weight",
            "qa_outputs.bias",
        ] {
            assert!(state_names.iter().any(|candidate| candidate == name));
        }

        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::I32);
        let token_type_ids = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        let word_weight = model
            .bert
            .embeddings
            .word_embeddings
            .weight
            .bind(&mut graph)
            .unwrap();
        let qa_bias = model
            .qa_outputs
            .bias
            .as_ref()
            .unwrap()
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
        assert_eq!(graph.shape(forward.output).unwrap(), &Shape::new([2, 2, 1]));
        assert_eq!(graph.dtype(forward.output).unwrap(), DType::F32);
        assert_eq!(
            (0..graph.node_count())
                .filter(|index| matches!(graph.op(NodeId(*index)), Ok(Op::Matmul { .. })))
                .count(),
            2,
            "the QA projection uses source Linear; only attention products are raw Matmul"
        );

        let loss = graph.sum_all(forward.output).unwrap();
        let gradients = graph
            .gradient_default(loss, &[qa_bias, word_weight])
            .unwrap();
        let values = bindings(&model, &graph, [1, 2]);
        let expected = TensorData::new([2, 2, 1], vec![2.0, 2.0, -3.0, -3.0]).unwrap();
        assert_eq!(
            CpuBackend.execute(&graph, forward.output, &values).unwrap(),
            expected
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradients[0], &values).unwrap(),
            TensorData::new([2], vec![2.0, 2.0]).unwrap()
        );
        assert_eq!(graph.shape(gradients[1]).unwrap(), &Shape::new([6, 4]));
        assert!(
            CpuBackend
                .execute(&graph, gradients[1], &values)
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
            .replay(
                &capture,
                &values.into_iter().collect(),
                CapturedReplayOptions::default(),
            )
            .unwrap();
        assert_eq!(replay.outputs, vec![expected]);
    }

    #[test]
    fn bert_qa_flattens_batch_time_and_keeps_empty_domains_zero_work() {
        let model = BertForQuestionAnswering::new_static(config(0, 0.0, 0.0), 13).unwrap();
        source_control_state(&model);

        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [2, 3], DType::I16);
        let attention_mask = graph.input_dtype("attention_mask", [2, 3], DType::Bool);
        let token_type_ids = graph.input_dtype("token_type_ids", [], DType::U8);
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
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 6, 1]));

        let mut empty = Graph::new();
        let input_ids = empty.input_dtype("input_ids", [0, 2], DType::I32);
        let attention_mask = empty.input_dtype("attention_mask", [0, 2], DType::I32);
        let token_type_ids = empty.input_dtype("token_type_ids", [1, 2], DType::I32);
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
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([2, 0, 1]));
        let mut values = model.input_bindings(&empty).unwrap();
        values.insert("input_ids".into(), ids([0, 2], []));
        values.insert("attention_mask".into(), ids([0, 2], []));
        values.insert("token_type_ids".into(), ids([1, 2], [0, 0]));
        assert_eq!(
            CpuBackend.execute(&empty, output, &values).unwrap(),
            TensorData::zeros([2, 0, 1]).unwrap()
        );

        let mut zero_time = Graph::new();
        let input_ids = zero_time.input_dtype("input_ids", [1, 0], DType::I32);
        let attention_mask = zero_time.input_dtype("attention_mask", [1, 0], DType::Bool);
        let token_type_ids = zero_time.input_dtype("token_type_ids", [], DType::I32);
        let output = model
            .forward_mode(
                &mut zero_time,
                input_ids,
                attention_mask,
                token_type_ids,
                Mode::Eval,
            )
            .unwrap()
            .output;
        assert_eq!(zero_time.shape(output).unwrap(), &Shape::new([2, 0, 1]));
        let mut values = model.input_bindings(&zero_time).unwrap();
        values.insert("input_ids".into(), ids([1, 0], []));
        values.insert(
            "attention_mask".into(),
            TensorData::from_scalars([1, 0], DType::Bool, []).unwrap(),
        );
        values.insert("token_type_ids".into(), ids([], [0]));
        assert_eq!(
            CpuBackend.execute(&zero_time, output, &values).unwrap(),
            TensorData::zeros([2, 0, 1]).unwrap()
        );
    }

    #[test]
    fn bert_qa_ambient_dropout_and_graph_publication_are_one_transaction() {
        let _lock = Graph::lock_implicit_random_tests();
        let model = BertForQuestionAnswering::new_static(config(1, 0.25, 0.5), 17).unwrap();
        Graph::manual_seed(71);
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::I32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::I32);
        let token_type_ids = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        let _training = TrainingContext::training();
        let output = model
            .forward_ambient(&mut graph, input_ids, attention_mask, token_type_ids)
            .unwrap();
        assert!(output.pending.is_empty());
        let expected_streams = random_streams(&graph);
        assert_eq!(
            expected_streams.len(),
            4,
            "embedding then three layer draws"
        );

        Graph::manual_seed(71);
        let mut retry = Graph::new();
        let input_ids = retry.input_dtype("input_ids", [1, 2], DType::I32);
        let bad_mask = retry.input_dtype("bad_mask", [3], DType::I32);
        let token_type_ids = retry.input_dtype("token_type_ids", [1, 2], DType::I32);
        let before = retry.node_count();
        assert!(
            model
                .forward_ambient(&mut retry, input_ids, bad_mask, token_type_ids)
                .is_err()
        );
        assert_eq!(retry.node_count(), before);

        let attention_mask = retry.input_dtype("attention_mask", [1, 2], DType::I32);
        model
            .forward_ambient(&mut retry, input_ids, attention_mask, token_type_ids)
            .unwrap();
        assert_eq!(random_streams(&retry), expected_streams);
    }

    #[test]
    fn bert_qa_rejects_invalid_model_and_inputs_atomically() {
        let mut invalid = config(1, 0.0, 0.0);
        invalid.hidden_size = 0;
        assert!(BertForQuestionAnswering::new_static(invalid, 19).is_err());

        let model = BertForQuestionAnswering::new_static(config(1, 0.0, 0.0), 19).unwrap();
        let mut graph = Graph::new();
        let input_ids = graph.input_dtype("input_ids", [1, 2], DType::F32);
        let attention_mask = graph.input_dtype("attention_mask", [1, 2], DType::I32);
        let token_type_ids = graph.input_dtype("token_type_ids", [1, 2], DType::I32);
        let before = graph.node_count();
        assert!(
            model
                .forward_mode(
                    &mut graph,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    Mode::Eval,
                )
                .is_err()
        );
        assert_eq!(graph.node_count(), before);
    }
}
