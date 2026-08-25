//! Stateless shape-only module adapters.

use super::{Module, ModuleForward, Parameter, StateKind};
use crate::{Error, Graph, NodeId, Result, Shape};

/// Flattens every dimension from `start_dim` through the final static axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flatten {
    start_dim: usize,
}

impl Flatten {
    pub const fn new(start_dim: usize) -> Self {
        Self { start_dim }
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let input_shape = graph.shape(input)?.clone();
        if self.start_dim >= input_shape.rank() {
            return Err(Error::InvalidReshape {
                from: input_shape,
                to: Shape::new([]),
            });
        }
        let flattened = input_shape.dims()[self.start_dim..]
            .iter()
            .try_fold(1usize, |product, &dim| product.checked_mul(dim))
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        let mut output = input_shape.dims()[..self.start_dim].to_vec();
        output.push(flattened);
        graph.reshape(input, Shape::new(output))
    }
}

impl Module for Flatten {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for Flatten {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, TensorData};

    #[test]
    fn flatten_preserves_batch_and_rejects_invalid_static_axis() {
        let module = Flatten::new(1);
        assert!(module.state_dict().unwrap().tensors().is_empty());
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3, 1, 1], DType::F32);
        let output = module.forward(&mut graph, input).unwrap();
        let value = CpuBackend
            .execute(
                &graph,
                output,
                &std::collections::HashMap::from([(
                    "input".into(),
                    TensorData::new([2, 3, 1, 1], vec![0.; 6]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(value.shape().dims(), &[2, 3]);
        let invalid = Flatten::new(4);
        assert!(invalid.forward(&mut graph, input).is_err());
    }
}
