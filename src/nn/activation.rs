//! Stateless activation module adapters.

use super::{Module, ModuleForward, Parameter, StateKind};
use crate::{Graph, NodeId, Result};

/// A stateless rectified-linear activation for one-input static compositions.
///
/// It owns no parameters, buffers, or execution mode. Its graph semantics are
/// exactly [`Graph::relu`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ReLU;

impl ReLU {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.relu(input)
    }
}

impl Module for ReLU {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for ReLU {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, TensorData};

    #[test]
    fn relu_is_stateless_and_delegates_to_the_graph() {
        let module = ReLU::new();
        assert!(module.state_dict().unwrap().tensors().is_empty());
        assert!(module.trainable_parameters().unwrap().is_empty());
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3], DType::F32);
        let output = module.forward(&mut graph, input).unwrap();
        let value = CpuBackend
            .execute(
                &graph,
                output,
                &std::collections::HashMap::from([(
                    "input".into(),
                    TensorData::new([3], vec![-1., 0., 2.]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(value.to_vec_f64(), [0., 0., 2.]);
    }
}
