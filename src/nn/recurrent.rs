//! Recurrent neural-network cells.

use super::{Module, Parameter, StateKind, init::uniform, state::join};
use crate::{Error, Graph, NodeId, Result, Shape, TensorData};

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
    pub fn new(
        _graph: &mut Graph,
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
