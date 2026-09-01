//! Recurrent neural-network cells.

use super::{
    Mode, Module, Parameter, StateKind,
    init::uniform,
    regularization::{apply_dropout, validate_dropout_probability},
    state::join,
};
use crate::{DType, Error, Graph, NodeId, Result, Shape, TensorData};

/// A compositional dense LSTM cell with tinygrad-compatible gate order
/// `(input, forget, cell, output)` and parameter names.
pub struct LSTMCell {
    pub weight_ih: Parameter,
    pub weight_hh: Parameter,
    pub bias_ih: Option<Parameter>,
    pub bias_hh: Option<Parameter>,
    input_size: usize,
    hidden_size: usize,
}
impl LSTMCell {
    /// Creates graph-independent LSTM cell state with tinygrad-compatible
    /// gate order and initialization.
    pub fn new_static(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        if input_size == 0 || hidden_size == 0 {
            return Err(Error::InvalidRandom {
                reason: "LSTM sizes must be nonzero",
            });
        }
        let b = 1.0 / (hidden_size as f32).sqrt();
        let gates = hidden_size
            .checked_mul(4)
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([hidden_size])))?;
        Ok(Self {
            weight_ih: Parameter::new(uniform(Shape::new([gates, input_size]), -b, b, seed)?, true),
            weight_hh: Parameter::new(
                uniform(
                    Shape::new([gates, hidden_size]),
                    -b,
                    b,
                    seed.wrapping_add(1),
                )?,
                true,
            ),
            bias_ih: bias.then(|| {
                Parameter::new(TensorData::zeros(Shape::new([gates])).expect("valid"), true)
            }),
            bias_hh: bias.then(|| {
                Parameter::new(TensorData::zeros(Shape::new([gates])).expect("valid"), true)
            }),
            input_size,
            hidden_size,
        })
    }

    /// Legacy graph-taking constructor retained for source compatibility.
    pub fn new(
        _: &mut Graph,
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static(input_size, hidden_size, bias, seed)
    }
    pub fn forward(
        &self,
        graph: &mut Graph,
        input: NodeId,
        state: Option<(NodeId, NodeId)>,
    ) -> Result<(NodeId, NodeId)> {
        let x = graph.shape(input)?.clone();
        if x.rank() != 2 || x.dims()[1] != self.input_size {
            return Err(Error::InvalidMatmul {
                lhs: x,
                rhs: Shape::new([self.input_size, self.hidden_size * 4]),
            });
        }
        let (h, c) = state.unwrap_or((
            graph.zeros_with_dtype(
                Shape::new([x.dims()[0], self.hidden_size]),
                graph.dtype(input)?,
            )?,
            graph.zeros_with_dtype(
                Shape::new([x.dims()[0], self.hidden_size]),
                graph.dtype(input)?,
            )?,
        ));
        for node in [h, c] {
            if graph.shape(node)?.dims() != [x.dims()[0], self.hidden_size] {
                return Err(Error::InvalidMatmul {
                    lhs: graph.shape(node)?.clone(),
                    rhs: Shape::new([x.dims()[0], self.hidden_size]),
                });
            }
        }
        let wi = self.weight_ih.bind(graph)?;
        let wi = graph.permute(wi, vec![1, 0])?;
        let wh = self.weight_hh.bind(graph)?;
        let wh = graph.permute(wh, vec![1, 0])?;
        let input_gates = graph.matmul(input, wi)?;
        let hidden_gates = graph.matmul(h, wh)?;
        let mut gates = graph.add(input_gates, hidden_gates)?;
        if let Some(b) = &self.bias_ih {
            let b = b.bind(graph)?;
            gates = graph.add(gates, b)?;
        }
        if let Some(b) = &self.bias_hh {
            let b = b.bind(graph)?;
            gates = graph.add(gates, b)?;
        }
        let gate = |g: &mut Graph, start: usize| {
            g.shrink(
                gates,
                vec![(0, x.dims()[0]), (start, start + self.hidden_size)],
            )
        };
        let gi = gate(graph, 0)?;
        let gf = gate(graph, self.hidden_size)?;
        let gz = gate(graph, self.hidden_size * 2)?;
        let go = gate(graph, self.hidden_size * 3)?;
        let i = graph.sigmoid(gi)?;
        let f = graph.sigmoid(gf)?;
        let z = graph.tanh(gz)?;
        let o = graph.sigmoid(go)?;
        let fc = graph.mul(f, c)?;
        let iz = graph.mul(i, z)?;
        let next_c = graph.add(fc, iz)?;
        let tanh_c = graph.tanh(next_c)?;
        let next_h = graph.mul(o, tanh_c)?;
        Ok((next_h, next_c))
    }
}
impl Module for LSTMCell {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(p, "weight_ih"), &self.weight_ih, StateKind::Parameter);
        v(join(p, "weight_hh"), &self.weight_hh, StateKind::Parameter);
        if let Some(x) = &self.bias_ih {
            v(join(p, "bias_ih"), x, StateKind::Parameter)
        }
        if let Some(x) = &self.bias_hh {
            v(join(p, "bias_hh"), x, StateKind::Parameter)
        }
    }
}

/// The recurrent state of a stacked [`LSTM`].
///
/// Both tensors have shape `[layers, batch, hidden_size]`. Keeping hidden and
/// cell state separate avoids the source model's packed `[layers, 2*batch,
/// hidden_size]` representation at the public Rust API boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LSTMState {
    hidden: NodeId,
    cell: NodeId,
}

impl LSTMState {
    /// Creates an explicit carried state from graph nodes.
    pub const fn new(hidden: NodeId, cell: NodeId) -> Self {
        Self { hidden, cell }
    }

    pub const fn hidden(self) -> NodeId {
        self.hidden
    }

    pub const fn cell(self) -> NodeId {
        self.cell
    }
}

/// Sequence and final state returned by [`LSTM::forward`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LSTMOutput {
    sequence: NodeId,
    state: LSTMState,
}

impl LSTMOutput {
    /// The last-layer hidden sequence with shape `[time, batch, hidden_size]`.
    pub const fn sequence(self) -> NodeId {
        self.sequence
    }

    /// The final hidden and cell state for every layer.
    pub const fn state(self) -> LSTMState {
        self.state
    }
}

#[derive(Clone, Debug)]
struct LSTMForwardPlan {
    time: usize,
    batch: usize,
    dropout_seeds: Vec<Vec<Option<u64>>>,
    initial_state: Option<LSTMState>,
}

/// A static, source-ordered stacked LSTM sequence composition.
///
/// Input and output sequences use `[time, batch, features]`; hidden and cell
/// state use `[layers, batch, hidden_size]`. Parameters are graph-independent
/// and traverse as `cells.{layer}.*`. Forward publication is transactional:
/// every descriptor and the complete composition are rehearsed on a cloned
/// graph before the live graph changes.
pub struct LSTM {
    cells: Vec<LSTMCell>,
    input_size: usize,
    hidden_size: usize,
    dropout: f64,
    seed: u64,
}

impl LSTM {
    /// Creates a graph-independent stack with source-compatible first/interior
    /// layer dropout and an undropped final layer when `layers > 1`.
    pub fn new_static(
        input_size: usize,
        hidden_size: usize,
        layers: usize,
        dropout: f64,
        seed: u64,
    ) -> Result<Self> {
        if input_size == 0 || hidden_size == 0 || layers == 0 {
            return Err(Error::InvalidRandom {
                reason: "LSTM sizes and layer count must be nonzero",
            });
        }
        validate_dropout_probability(dropout)?;
        let mut cells = Vec::with_capacity(layers);
        for layer in 0..layers {
            let layer_seed = u64::try_from(layer)
                .ok()
                .and_then(|layer| layer.checked_mul(2))
                .and_then(|offset| seed.checked_add(offset))
                .ok_or(Error::InvalidRandom {
                    reason: "LSTM layer seed range overflows u64",
                })?;
            cells.push(LSTMCell::new_static(
                if layer == 0 { input_size } else { hidden_size },
                hidden_size,
                true,
                layer_seed,
            )?);
        }
        Ok(Self {
            cells,
            input_size,
            hidden_size,
            dropout,
            seed,
        })
    }

    pub const fn input_size(&self) -> usize {
        self.input_size
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn layers(&self) -> usize {
        self.cells.len()
    }

    pub const fn dropout(&self) -> f64 {
        self.dropout
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Borrows one cell for typed parameter inspection or replacement.
    pub fn cell(&self, layer: usize) -> Option<&LSTMCell> {
        self.cells.get(layer)
    }

    fn layer_dropout(&self, layer: usize) -> f64 {
        if layer == 0 || layer + 1 < self.cells.len() {
            self.dropout
        } else {
            0.0
        }
    }

    fn plan(
        &self,
        graph: &Graph,
        input: NodeId,
        state: Option<LSTMState>,
        mode: Mode,
    ) -> Result<LSTMForwardPlan> {
        validate_dropout_probability(self.dropout)?;
        if self.cells.is_empty() || self.input_size == 0 || self.hidden_size == 0 {
            return Err(Error::InvalidRandom {
                reason: "LSTM descriptor is empty",
            });
        }
        let shape = graph.shape(input)?.clone();
        if shape.rank() != 3 || shape.dims()[2] != self.input_size {
            return Err(Error::InvalidMatmul {
                lhs: shape,
                rhs: Shape::new([self.input_size, self.hidden_size]),
            });
        }
        if graph.dtype(input)? != DType::F32 {
            return Err(Error::InvalidElementwiseDType {
                op: "stacked LSTM",
                actual: graph.dtype(input)?,
            });
        }
        let time = shape.dims()[0];
        let batch = shape.dims()[1];
        if time == 0 {
            return Err(Error::InvalidRandom {
                reason: "LSTM sequence length must be nonzero",
            });
        }
        if let Some(state) = state {
            let expected = Shape::new([self.cells.len(), batch, self.hidden_size]);
            for node in [state.hidden, state.cell] {
                if graph.shape(node)? != &expected {
                    return Err(Error::ShapeMismatch {
                        op: "stacked LSTM state",
                        lhs: graph.shape(node)?.clone(),
                        rhs: expected.clone(),
                    });
                }
                if graph.dtype(node)? != DType::F32 {
                    return Err(Error::InvalidElementwiseDType {
                        op: "stacked LSTM state",
                        actual: graph.dtype(node)?,
                    });
                }
            }
        }
        let mut dropout_seeds = vec![vec![None; self.cells.len()]; time];
        if mode == Mode::Training && self.dropout != 0.0 && batch != 0 {
            for (timestep, seeds) in dropout_seeds.iter_mut().enumerate() {
                for (layer, slot) in seeds.iter_mut().enumerate() {
                    if self.layer_dropout(layer) == 0.0 {
                        continue;
                    }
                    let invocation = timestep
                        .checked_mul(self.cells.len())
                        .and_then(|value| value.checked_add(layer))
                        .and_then(|value| u64::try_from(value).ok())
                        .and_then(|value| self.seed.checked_add(value))
                        .ok_or(Error::InvalidRandom {
                            reason: "LSTM dropout seed range overflows u64",
                        })?;
                    *slot = Some(invocation);
                }
            }
        }
        Ok(LSTMForwardPlan {
            time,
            batch,
            dropout_seeds,
            initial_state: state,
        })
    }

    fn lower(
        &self,
        graph: &mut Graph,
        input: NodeId,
        plan: &LSTMForwardPlan,
    ) -> Result<LSTMOutput> {
        let state_shape = Shape::new([self.cells.len(), plan.batch, self.hidden_size]);
        let state = match plan.initial_state {
            Some(state) => state,
            None => LSTMState::new(
                graph.zeros_with_dtype(state_shape.clone(), DType::F32)?,
                graph.zeros_with_dtype(state_shape, DType::F32)?,
            ),
        };
        let mut hidden = state.hidden;
        let mut cell = state.cell;
        let mut outputs = Vec::with_capacity(plan.time);
        for (timestep, dropout_seeds) in plan.dropout_seeds.iter().enumerate() {
            let timestep_input = graph.shrink(
                input,
                [
                    (timestep, timestep + 1),
                    (0, plan.batch),
                    (0, self.input_size),
                ],
            )?;
            let mut layer_input = graph.reshape(timestep_input, [plan.batch, self.input_size])?;
            let mut next_hidden = Vec::with_capacity(self.cells.len());
            let mut next_cell = Vec::with_capacity(self.cells.len());
            for (layer, lstm_cell) in self.cells.iter().enumerate() {
                let prior_hidden = graph.shrink(
                    hidden,
                    [(layer, layer + 1), (0, plan.batch), (0, self.hidden_size)],
                )?;
                let prior_hidden = graph.reshape(prior_hidden, [plan.batch, self.hidden_size])?;
                let prior_cell = graph.shrink(
                    cell,
                    [(layer, layer + 1), (0, plan.batch), (0, self.hidden_size)],
                )?;
                let prior_cell = graph.reshape(prior_cell, [plan.batch, self.hidden_size])?;
                let (mut later_hidden, later_cell) =
                    lstm_cell.forward(graph, layer_input, Some((prior_hidden, prior_cell)))?;
                if let Some(seed) = dropout_seeds[layer] {
                    later_hidden =
                        apply_dropout(graph, later_hidden, self.layer_dropout(layer), seed)?;
                }
                layer_input = later_hidden;
                next_hidden.push(later_hidden);
                next_cell.push(later_cell);
            }
            outputs.push(layer_input);
            hidden = graph.stack_default(next_hidden)?;
            cell = graph.stack_default(next_cell)?;
        }
        let sequence = graph.stack_default(outputs)?;
        debug_assert_eq!(
            graph.shape(sequence).ok().map(Shape::dims),
            Some(&[plan.time, plan.batch, self.hidden_size][..])
        );
        Ok(LSTMOutput {
            sequence,
            state: LSTMState::new(hidden, cell),
        })
    }

    /// Composes one static sequence and atomically publishes it to `graph`.
    ///
    /// `state=None` creates zero hidden and cell state. Training mode applies
    /// deterministic inverted dropout with per-time/per-layer derived seeds;
    /// evaluation never creates random nodes.
    pub fn forward(
        &self,
        graph: &mut Graph,
        input: NodeId,
        state: Option<LSTMState>,
        mode: Mode,
    ) -> Result<LSTMOutput> {
        let plan = self.plan(graph, input, state, mode)?;
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, input, &plan)?;
        *graph = candidate;
        Ok(output)
    }
}

impl Module for LSTM {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        for (layer, cell) in self.cells.iter().enumerate() {
            cell.visit(&join(prefix, &format!("cells.{layer}")), visitor);
        }
    }
}
