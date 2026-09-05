//! Source-ordered static Transformer block composition.

use super::{
    LayerNorm, Mode, ModeForwardOutput, ModeModuleForward, Module, ModuleForward, Parameter,
    PendingModeEffects, ReLU, StateKind, init::uniform,
    regularization::validate_dropout_probability, state::join,
};
use crate::{
    AttentionOptions, DType, Error, Graph, NodeId, RandomStream, Result, Shape, TensorData,
    TrainingContext,
};

#[derive(Clone, Copy)]
enum ResidualDropout {
    Eval,
    Seeded([u64; 2]),
    Ambient([RandomStream; 2]),
}

/// The checked source stores each Transformer projection as a `(weight, bias)`
/// pair with `[input, output]` weight orientation. Keep that local state
/// contract instead of silently transposing it into the public `nn::Linear`
/// module's `[output, input]` convention.
struct TransformerProjection {
    weight: Parameter,
    bias: Parameter,
    input: usize,
    output: usize,
}

impl TransformerProjection {
    fn new(input: usize, output: usize, seed: u64) -> Result<Self> {
        let shape = Shape::new([input, output]);
        let elements = shape.numel()?;
        let bound = 1.0 / (elements as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(uniform(shape, -bound, bound, seed)?, true),
            bias: Parameter::new(TensorData::zeros([output])?, true),
            input,
            output,
        })
    }

    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.dims().last().copied() != Some(self.input) {
            return Err(Error::InvalidMatmul {
                lhs: graph.shape(input)?.clone(),
                rhs: Shape::new([self.input, self.output]),
            });
        }
        let weight = self.weight.bind(graph)?;
        let bias = self.bias.bind(graph)?;
        graph.linear(input, weight, Some(bias), None)
    }
}

impl Module for TransformerProjection {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        visitor(join(prefix, "0"), &self.weight, StateKind::Parameter);
        visitor(join(prefix, "1"), &self.bias, StateKind::Parameter);
    }
}

/// Checked-in tinygrad's reusable static Transformer block.
///
/// The block accepts `[batch, time, embedding]`, owns query/key/value/output
/// projections plus a two-layer feed-forward network, and preserves the
/// source's pre- or post-normalization order. `A` is an ordinary module so a
/// caller-supplied activation retains its own typed traversal and graph
/// composition; [`ReLU`] is the source default and [`super::ActivationFn`]
/// adapts a stateless closure. Projection and LayerNorm affine state uses the
/// source tuple paths (`query.0`, `query.1`, `ln1.0`, and peers).
///
/// Dropout mode remains explicit through [`ModeModuleForward::forward_mode`]
/// or scoped through [`ModeModuleForward::forward_ambient`]. No state change is
/// hidden in either route.
pub struct TransformerBlock<A = ReLU> {
    query: TransformerProjection,
    key: TransformerProjection,
    value: TransformerProjection,
    out: TransformerProjection,
    ff1: TransformerProjection,
    ff2: TransformerProjection,
    ln1: LayerNorm,
    ln2: LayerNorm,
    activation: A,
    embedding_dim: usize,
    num_heads: usize,
    head_size: usize,
    feed_forward_dim: usize,
    prenorm: bool,
    is_causal: bool,
    dropout: f64,
    dropout_seeds: [u64; 2],
}

impl TransformerBlock<ReLU> {
    /// Creates a deterministic graph-independent block with source-default
    /// ReLU activation and affine LayerNorm parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new_static(
        embedding_dim: usize,
        num_heads: usize,
        feed_forward_dim: usize,
        prenorm: bool,
        dropout: f64,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static_with_activation(
            embedding_dim,
            num_heads,
            feed_forward_dim,
            prenorm,
            dropout,
            seed,
            ReLU,
        )
    }
}

impl<A: ModuleForward> TransformerBlock<A> {
    /// Creates the same block with an arbitrary typed activation module.
    ///
    /// Stateful activation parameters, when present, are exposed below the
    /// `act` traversal prefix rather than erased behind a runtime kind tag.
    #[allow(clippy::too_many_arguments)]
    pub fn new_static_with_activation(
        embedding_dim: usize,
        num_heads: usize,
        feed_forward_dim: usize,
        prenorm: bool,
        dropout: f64,
        seed: u64,
        activation: A,
    ) -> Result<Self> {
        if embedding_dim == 0
            || num_heads == 0
            || feed_forward_dim == 0
            || !embedding_dim.is_multiple_of(num_heads)
        {
            return Err(Error::InvalidAttention {
                reason: "TransformerBlock dimensions must be nonzero and embedding must divide heads",
            });
        }
        validate_dropout_probability(dropout)?;
        Ok(Self {
            query: TransformerProjection::new(embedding_dim, embedding_dim, seed)?,
            key: TransformerProjection::new(embedding_dim, embedding_dim, seed.wrapping_add(1))?,
            value: TransformerProjection::new(embedding_dim, embedding_dim, seed.wrapping_add(2))?,
            out: TransformerProjection::new(embedding_dim, embedding_dim, seed.wrapping_add(3))?,
            ff1: TransformerProjection::new(embedding_dim, feed_forward_dim, seed.wrapping_add(4))?,
            ff2: TransformerProjection::new(feed_forward_dim, embedding_dim, seed.wrapping_add(5))?,
            ln1: LayerNorm::new_static([embedding_dim], 1e-5, true)?,
            ln2: LayerNorm::new_static([embedding_dim], 1e-5, true)?,
            activation,
            embedding_dim,
            num_heads,
            head_size: embedding_dim / num_heads,
            feed_forward_dim,
            prenorm,
            is_causal: false,
            dropout,
            dropout_seeds: [seed.wrapping_add(6), seed.wrapping_add(7)],
        })
    }

    pub const fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub const fn feed_forward_dim(&self) -> usize {
        self.feed_forward_dim
    }

    pub const fn prenorm(&self) -> bool {
        self.prenorm
    }

    /// Selects causal attention for autoregressive training and decoding.
    ///
    /// The default remains bidirectional to preserve the existing reusable
    /// block contract. Enabling this option feeds the same explicit causal
    /// mask policy into every forward mode; it does not depend on ambient
    /// process state.
    pub fn with_causal_attention(mut self, is_causal: bool) -> Self {
        self.is_causal = is_causal;
        self
    }

    pub const fn is_causal(&self) -> bool {
        self.is_causal
    }

    pub const fn dropout(&self) -> f64 {
        self.dropout
    }

    pub const fn activation(&self) -> &A {
        &self.activation
    }

    fn geometry(&self, graph: &Graph, input: NodeId) -> Result<(usize, usize)> {
        let shape = graph.shape(input)?;
        shape.numel()?;
        if shape.rank() != 3 || shape.dims()[0] == 0 || shape.dims()[2] != self.embedding_dim {
            return Err(Error::InvalidReshape {
                from: shape.clone(),
                to: Shape::new([1, 1, self.embedding_dim]),
            });
        }
        Ok((shape.dims()[0], shape.dims()[1]))
    }

    fn attention(
        &self,
        graph: &mut Graph,
        input: NodeId,
        batch: usize,
        time: usize,
    ) -> Result<NodeId> {
        let heads = |graph: &mut Graph, projection: &TransformerProjection| -> Result<NodeId> {
            let projected = projection.forward(graph, input)?;
            let projected =
                graph.reshape(projected, [batch, time, self.num_heads, self.head_size])?;
            graph.permute(projected, vec![0, 2, 1, 3])
        };
        let query = heads(graph, &self.query)?;
        let key = heads(graph, &self.key)?;
        let value = heads(graph, &self.value)?;
        let attended = graph.scaled_dot_product_attention(
            query,
            key,
            value,
            None,
            AttentionOptions {
                is_causal: self.is_causal,
                ..AttentionOptions::default()
            },
        )?;
        let attended = graph.permute(attended, vec![0, 2, 1, 3])?;
        // Flattening heads after the time/head transpose is not one affine
        // view of the attention result when `time > 1`. Materialize that
        // source reshape boundary once so the following source-Linear Dot
        // reads ordinary contiguous `[batch, time, embedding]` storage.
        let attended = graph.contiguous(attended)?;
        let attended = graph.reshape(attended, [batch, time, self.embedding_dim])?;
        self.out.forward(graph, attended)
    }

    fn feed_forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let hidden = self.ff1.forward(graph, input)?;
        let hidden = self.activation.forward(graph, hidden)?;
        self.ff2.forward(graph, hidden)
    }

    fn apply_dropout(
        &self,
        graph: &mut Graph,
        input: NodeId,
        branch: usize,
        mode: ResidualDropout,
    ) -> Result<NodeId> {
        match mode {
            ResidualDropout::Eval => Ok(input),
            ResidualDropout::Seeded(seeds) => {
                graph.dropout(input, self.dropout, true, Some(seeds[branch]))
            }
            ResidualDropout::Ambient(streams) => {
                graph.lower_ambient_dropout(input, self.dropout, streams[branch])
            }
        }
    }

    fn lower(&self, graph: &mut Graph, input: NodeId, dropout: ResidualDropout) -> Result<NodeId> {
        let (batch, time) = self.geometry(graph, input)?;
        if self.prenorm {
            let normalized = self.ln1.forward(graph, input)?;
            let attended = self.attention(graph, normalized, batch, time)?;
            let attended = self.apply_dropout(graph, attended, 0, dropout)?;
            let residual = graph.add(input, attended)?;
            let normalized = self.ln2.forward(graph, residual)?;
            let feed_forward = self.feed_forward(graph, normalized)?;
            let feed_forward = self.apply_dropout(graph, feed_forward, 1, dropout)?;
            graph.add(residual, feed_forward)
        } else {
            let attended = self.attention(graph, input, batch, time)?;
            let attended = self.apply_dropout(graph, attended, 0, dropout)?;
            let residual = graph.add(input, attended)?;
            let residual = self.ln1.forward(graph, residual)?;
            let feed_forward = self.feed_forward(graph, residual)?;
            let feed_forward = self.apply_dropout(graph, feed_forward, 1, dropout)?;
            let output = graph.add(residual, feed_forward)?;
            self.ln2.forward(graph, output)
        }
    }
}

impl<A: ModuleForward> Module for TransformerBlock<A> {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.query.visit(&join(prefix, "query"), visitor);
        self.key.visit(&join(prefix, "key"), visitor);
        self.value.visit(&join(prefix, "value"), visitor);
        self.out.visit(&join(prefix, "out"), visitor);
        self.ff1.visit(&join(prefix, "ff1"), visitor);
        self.ff2.visit(&join(prefix, "ff2"), visitor);
        if let Some(weight) = &self.ln1.weight {
            visitor(join(prefix, "ln1.0"), weight, StateKind::Parameter);
        }
        if let Some(bias) = &self.ln1.bias {
            visitor(join(prefix, "ln1.1"), bias, StateKind::Parameter);
        }
        if let Some(weight) = &self.ln2.weight {
            visitor(join(prefix, "ln2.0"), weight, StateKind::Parameter);
        }
        if let Some(bias) = &self.ln2.bias {
            visitor(join(prefix, "ln2.1"), bias, StateKind::Parameter);
        }
        self.activation.visit(&join(prefix, "act"), visitor);
    }
}

impl<A: ModuleForward> ModeModuleForward for TransformerBlock<A> {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let dropout = match mode {
            Mode::Eval => ResidualDropout::Eval,
            Mode::Training => ResidualDropout::Seeded(self.dropout_seeds),
        };
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, input, dropout)?;
        *graph = candidate;
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }

    fn forward_ambient<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
    ) -> Result<ModeForwardOutput<'a>> {
        validate_dropout_probability(self.dropout)?;
        let training = TrainingContext::is_training();
        let output = if training && self.dropout > 0.0 && self.dropout < 1.0 {
            let shape = graph.shape(input)?.clone();
            graph.with_implicit_uniform_streams(
                vec![(shape.clone(), DType::F32), (shape, DType::F32)],
                0,
                |candidate, streams| {
                    self.lower(
                        candidate,
                        input,
                        ResidualDropout::Ambient([streams[0], streams[1]]),
                    )
                },
            )?
        } else {
            let dropout = if training {
                ResidualDropout::Seeded(self.dropout_seeds)
            } else {
                ResidualDropout::Eval
            };
            let mut candidate = graph.clone();
            let output = self.lower(&mut candidate, input, dropout)?;
            *graph = candidate;
            output
        };
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActivationFn, Backend, CapturedReplayExecutor, CapturedReplayOptions, CpuBackend, Error,
        ModuleStateDict, Op, TensorData, nn::CastPolicy,
    };
    use std::collections::BTreeMap;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    fn zero_projections(block: &TransformerBlock) {
        let tensors: BTreeMap<String, TensorData> = block
            .state_dict()
            .unwrap()
            .into_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let replacement = if matches!(name.as_str(), "ln1.0" | "ln2.0") {
                    TensorData::ones(tensor.shape().clone()).unwrap()
                } else {
                    TensorData::zeros(tensor.shape().clone()).unwrap()
                };
                (name, replacement)
            })
            .collect();
        assert!(
            block
                .load_state_dict(&ModuleStateDict::from(tensors), true, CastPolicy::Exact)
                .unwrap()
                .is_clean()
        );
    }

    fn random_streams(graph: &Graph) -> Vec<RandomStream> {
        (0..graph.node_count())
            .filter_map(|index| match graph.op(NodeId(index)).unwrap() {
                Op::Random { stream, .. } => Some(*stream),
                _ => None,
            })
            .collect()
    }

    fn assert_replayable_schedule(graph: &Graph, schedule: &crate::Schedule) {
        let unsupported = schedule
            .items
            .iter()
            .filter_map(|item| {
                item.boundary.as_ref().map(|boundary| {
                    (
                        item.id,
                        item.node,
                        graph.op(item.node).unwrap().clone(),
                        boundary.clone(),
                    )
                })
            })
            .collect::<Vec<_>>();
        assert!(
            unsupported.is_empty(),
            "unsupported Transformer schedule items: {unsupported:?}"
        );
    }

    #[test]
    fn transformer_block_validates_configuration_and_declares_source_state() {
        assert!(matches!(
            TransformerBlock::new_static(6, 4, 8, false, 0.1, 1),
            Err(Error::InvalidAttention { .. })
        ));
        assert!(matches!(
            TransformerBlock::new_static(4, 2, 8, false, f64::NAN, 1),
            Err(Error::UnsupportedDropout { .. })
        ));

        let block = TransformerBlock::new_static(4, 2, 8, false, 0.1, 1).unwrap();
        assert_eq!(block.embedding_dim(), 4);
        assert_eq!(block.num_heads(), 2);
        assert_eq!(block.feed_forward_dim(), 8);
        assert!(!block.prenorm());
        assert!(!block.is_causal());
        assert_eq!(block.dropout(), 0.1);
        let state = block.state_dict().unwrap();
        assert_eq!(state.tensors()["ff1.0"].shape(), &Shape::new([4, 8]));
        assert_eq!(state.tensors()["ff2.0"].shape(), &Shape::new([8, 4]));
        assert_eq!(
            state
                .tensors()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec![
                "ff1.0", "ff1.1", "ff2.0", "ff2.1", "key.0", "key.1", "ln1.0", "ln1.1", "ln2.0",
                "ln2.1", "out.0", "out.1", "query.0", "query.1", "value.0", "value.1",
            ]
        );

        let custom = TransformerBlock::new_static_with_activation(
            4,
            2,
            8,
            true,
            0.0,
            2,
            LayerNorm::new_static([8], 1e-5, true).unwrap(),
        )
        .unwrap();
        let keys = custom
            .state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert!(keys.iter().any(|key| key == "act.weight"));
        assert!(keys.iter().any(|key| key == "act.bias"));

        let mut declared = Vec::new();
        block.visit("", &mut |name, _, _| declared.push(name));
        assert_eq!(
            declared,
            vec![
                "query.0", "query.1", "key.0", "key.1", "value.0", "value.1", "out.0", "out.1",
                "ff1.0", "ff1.1", "ff2.0", "ff2.1", "ln1.0", "ln1.1", "ln2.0", "ln2.1",
            ]
        );

        let closure = TransformerBlock::new_static_with_activation(
            4,
            2,
            8,
            false,
            0.0,
            3,
            ActivationFn::new(|graph: &mut Graph, input| graph.square(input)),
        )
        .unwrap();
        assert!(
            closure
                .state_dict()
                .unwrap()
                .tensors()
                .keys()
                .all(|key| !key.starts_with("act"))
        );

        let causal = TransformerBlock::new_static(4, 2, 8, true, 0.0, 4)
            .unwrap()
            .with_causal_attention(true);
        assert!(causal.is_causal());

        let mut narrow_graph = Graph::new();
        let narrow = narrow_graph.input_dtype("narrow", [1, 1, 4], DType::F16);
        closure.ff1.forward(&mut narrow_graph, narrow).unwrap();
        assert!((0..narrow_graph.node_count()).all(|index| {
            !matches!(narrow_graph.op(NodeId(index)).unwrap(), Op::Matmul { .. })
        }));
    }

    #[test]
    fn transformer_block_preserves_source_pre_and_post_norm_order_and_gradients() {
        let input_value = data([1, 2, 4], &[1.0, -2.0, 3.0, 0.5, -1.0, 4.0, 2.0, -3.0]);
        for prenorm in [false, true] {
            let block = TransformerBlock::new_static(4, 2, 8, prenorm, 0.0, 7).unwrap();
            zero_projections(&block);
            let mut graph = Graph::new();
            let input = graph.input("input", [1, 2, 4]);
            let query_weight = block.query.weight.bind(&mut graph).unwrap();
            let expected = if prenorm {
                input
            } else {
                let first = block.ln1.forward(&mut graph, input).unwrap();
                block.ln2.forward(&mut graph, first).unwrap()
            };
            let output = block
                .forward_mode(&mut graph, input, Mode::Eval)
                .unwrap()
                .output;
            assert!((0..graph.node_count()).any(|index| {
                let Ok(Op::Reshape { input, .. }) = graph.op(NodeId(index)) else {
                    return false;
                };
                let Ok(Op::Contiguous { input }) = graph.op(*input) else {
                    return false;
                };
                matches!(graph.op(*input), Ok(Op::Permute { .. }))
            }));
            let loss = graph.sum_all(output).unwrap();
            let gradients = graph
                .gradient_default(loss, &[input, query_weight])
                .unwrap();
            let mut bindings = block.input_bindings(&graph).unwrap();
            bindings.insert("input".into(), input_value.clone());
            let actual = CpuBackend.execute(&graph, output, &bindings).unwrap();
            let reference = CpuBackend.execute(&graph, expected, &bindings).unwrap();
            assert_eq!(actual, reference);
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
            assert_replayable_schedule(&graph, &schedule);
            let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
            let replay = CapturedReplayExecutor::default()
                .replay(
                    &capture,
                    &bindings.clone().into_iter().collect::<BTreeMap<_, _>>(),
                    CapturedReplayOptions::default(),
                )
                .unwrap();
            assert_eq!(replay.outputs, vec![actual]);
        }
    }

    #[test]
    fn ambient_transformer_dropout_reserves_both_branches_atomically() {
        struct LateFailure;
        impl Module for LateFailure {
            fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
        }
        impl ModuleForward for LateFailure {
            fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
                let _staged = graph.square(input)?;
                Err(Error::InvalidRandom {
                    reason: "injected Transformer activation failure",
                })
            }
        }

        let _random_lock = Graph::lock_implicit_random_tests();
        Graph::manual_seed(19);
        let failing =
            TransformerBlock::new_static_with_activation(4, 2, 8, false, 0.5, 3, LateFailure)
                .unwrap();
        let mut failed_graph = Graph::new();
        let failed_input = failed_graph.input("input", [1, 2, 4]);
        let before = failed_graph.node_count();
        {
            let _training = TrainingContext::training();
            assert!(matches!(
                failing.forward_ambient(&mut failed_graph, failed_input),
                Err(Error::InvalidRandom {
                    reason: "injected Transformer activation failure"
                })
            ));
        }
        assert_eq!(failed_graph.node_count(), before);

        let block = TransformerBlock::new_static(4, 2, 8, false, 0.5, 5).unwrap();
        let mut training_graph = Graph::new();
        let training_input = training_graph.input("input", [1, 2, 4]);
        let training_output = {
            let _training = TrainingContext::training();
            block
                .forward_ambient(&mut training_graph, training_input)
                .unwrap()
                .output
        };
        let streams = random_streams(&training_graph);
        assert_eq!(streams.len(), 2);
        assert_eq!(streams[0].counter, [0, 0]);
        assert_eq!(streams[1].counter, [8, 0]);

        let mut bindings = block.input_bindings(&training_graph).unwrap();
        bindings.insert(
            "input".into(),
            data([1, 2, 4], &[1.0, -2.0, 3.0, 0.5, -1.0, 4.0, 2.0, -3.0]),
        );
        assert!(
            CpuBackend
                .execute(&training_graph, training_output, &bindings)
                .unwrap()
                .to_vec_f64()
                .iter()
                .all(|value| value.is_finite())
        );
        let expected = CpuBackend
            .execute(&training_graph, training_output, &bindings)
            .unwrap();
        let schedule = crate::schedule(&training_graph, training_output).unwrap();
        assert_replayable_schedule(&training_graph, &schedule);
        let capture =
            crate::CapturedSchedule::capture(&training_graph, &schedule, &[training_output])
                .unwrap();
        let replay = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &bindings.clone().into_iter().collect::<BTreeMap<_, _>>(),
                CapturedReplayOptions::default(),
            )
            .unwrap();
        assert_eq!(replay.outputs, vec![expected]);

        let mut eval_graph = Graph::new();
        let eval_input = eval_graph.input("input", [1, 2, 4]);
        block.forward_ambient(&mut eval_graph, eval_input).unwrap();
        assert!(random_streams(&eval_graph).is_empty());

        let next = failed_graph.rand_implicit([1], DType::F32).unwrap();
        let Op::Random { stream, .. } = failed_graph.op(next).unwrap() else {
            panic!("expected ambient random source")
        };
        assert_eq!(stream.counter, [16, 0]);
    }

    #[test]
    fn transformer_ambient_mode_nests_isolates_threads_and_never_overrides_explicit_mode() {
        let _random_lock = Graph::lock_implicit_random_tests();
        Graph::manual_seed(23);
        let count = || {
            let block = TransformerBlock::new_static(4, 2, 8, false, 0.5, 17).unwrap();
            let mut graph = Graph::new();
            let input = graph.input("input", [1, 1, 4]);
            block.forward_ambient(&mut graph, input).unwrap();
            random_streams(&graph).len()
        };

        assert_eq!(count(), 0);
        let training = TrainingContext::training();
        assert_eq!(count(), 2);
        let evaluation = TrainingContext::evaluation();
        assert_eq!(count(), 0);
        drop(evaluation);
        assert_eq!(count(), 2);
        assert_eq!(std::thread::spawn(count).join().unwrap(), 0);

        let block = TransformerBlock::new_static(4, 2, 8, false, 0.5, 19).unwrap();
        let mut explicit_eval = Graph::new();
        let eval_input = explicit_eval.input("input", [1, 1, 4]);
        block
            .forward_mode(&mut explicit_eval, eval_input, Mode::Eval)
            .unwrap();
        assert!(random_streams(&explicit_eval).is_empty());
        drop(training);

        let _evaluation = TrainingContext::evaluation();
        let mut explicit_training = Graph::new();
        let training_input = explicit_training.input("input", [1, 1, 4]);
        block
            .forward_mode(&mut explicit_training, training_input, Mode::Training)
            .unwrap();
        assert_eq!(random_streams(&explicit_training).len(), 2);
    }

    #[test]
    fn transformer_empty_time_matches_source_and_input_failures_publish_nothing() {
        let block = TransformerBlock::new_static(4, 2, 8, false, 0.0, 11).unwrap();
        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("input", [1, 0, 4]);
        let output = block
            .forward_mode(&mut empty_graph, empty, Mode::Eval)
            .unwrap()
            .output;
        assert_eq!(empty_graph.shape(output).unwrap(), &Shape::new([1, 0, 4]));

        for shape in [
            Shape::new([2, 4]),
            Shape::new([0, 2, 4]),
            Shape::new([1, 2, 3]),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", shape, DType::F32);
            let before = graph.node_count();
            assert!(block.forward_mode(&mut graph, input, Mode::Eval).is_err());
            assert_eq!(graph.node_count(), before);
        }
    }
}
