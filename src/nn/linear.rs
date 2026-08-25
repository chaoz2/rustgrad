//! Fully connected module composition.

use super::{Module, ModuleForward, Parameter, StateKind, init::uniform, state::join};
use crate::{Error, Graph, NodeId, Result, Shape};

pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_features: usize,
    pub out_features: usize,
}
impl Linear {
    /// Creates deterministic, graph-independent host parameters.
    ///
    /// Bind the resulting module only when constructing a forward graph. This
    /// is the preferred constructor for static CPU module workflows.
    pub fn new_static(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        if in_features == 0 {
            return Err(Error::InvalidRandom {
                reason: "Linear in_features must be nonzero",
            });
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(
                uniform(Shape::new([out_features, in_features]), -bound, bound, seed)?,
                true,
            ),
            bias: bias.then(|| {
                Parameter::new(
                    uniform(
                        Shape::new([out_features]),
                        -bound,
                        bound,
                        seed.wrapping_add(1),
                    )
                    .expect("validated shape"),
                    true,
                )
            }),
            in_features,
            out_features,
        })
    }

    /// Legacy construction spelling retained for source compatibility.
    ///
    /// Parameters have no graph ownership; `graph` is deliberately ignored and
    /// graph binding still happens only in [`Self::forward`].
    pub fn new(
        _graph: &mut Graph,
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static(in_features, out_features, bias, seed)
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.dims().last().copied() != Some(self.in_features) {
            return Err(Error::InvalidMatmul {
                lhs: graph.shape(input)?.clone(),
                rhs: Shape::new([self.out_features, self.in_features]),
            });
        }
        let weight = self.weight.bind(graph)?;
        let weight = graph.permute(weight, vec![1, 0])?;
        let output = graph.matmul(input, weight)?;
        if let Some(bias) = &self.bias {
            let bias = bias.bind(graph)?;
            graph.add(output, bias)
        } else {
            Ok(output)
        }
    }
}
impl Module for Linear {
    fn visit(&self, prefix: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(prefix, "weight"), &self.weight, StateKind::Parameter);
        if let Some(b) = &self.bias {
            v(join(prefix, "bias"), b, StateKind::Parameter)
        }
    }
}
impl ModuleForward for Linear {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}
