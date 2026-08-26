//! Stateless inference-output module adapters.

use super::{Module, ModuleForward, Parameter, StateKind};
use crate::{Graph, NodeId, Result};

/// A stateless first-tie argmax over one checked signed axis.
///
/// The output uses the existing Graph I32 index contract. This is an
/// inference-output adapter, not a differentiable training layer.
#[derive(Clone, Copy, Debug)]
pub struct Argmax {
    axis: isize,
    keepdim: bool,
}

impl Argmax {
    /// Creates an argmax that removes the selected axis.
    pub const fn new(axis: isize) -> Self {
        Self {
            axis,
            keepdim: false,
        }
    }

    /// Creates an argmax that retains the selected axis with extent one.
    pub const fn keepdim(axis: isize) -> Self {
        Self {
            axis,
            keepdim: true,
        }
    }

    pub const fn axis(self) -> isize {
        self.axis
    }

    pub const fn retains_axis(self) -> bool {
        self.keepdim
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.argmax(input, Some(self.axis), self.keepdim)
    }
}

impl Module for Argmax {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for Argmax {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Sequential};
    use crate::{Backend, CpuBackend, DType, Storage, TensorData, infer_module_cpu};

    #[test]
    fn argmax_is_stateless_and_matches_the_direct_graph_path() {
        let module = Argmax::new(-1);
        assert!(module.state_dict().unwrap().tensors().is_empty());
        assert!(module.trainable_parameters().unwrap().is_empty());

        let input = TensorData::new([2, 3], vec![1., 3., 3., 2., 0., -1.]).unwrap();
        let mut module_graph = Graph::new();
        let module_input = module_graph.input("input", [2, 3]);
        let module_output = module.forward(&mut module_graph, module_input).unwrap();
        let mut direct_graph = Graph::new();
        let direct_input = direct_graph.input("input", [2, 3]);
        let direct_output = direct_graph.argmax(direct_input, Some(-1), false).unwrap();
        let bindings = std::collections::HashMap::from([("input".into(), input)]);
        let output = CpuBackend
            .execute(&module_graph, module_output, &bindings)
            .unwrap();
        assert_eq!(output.storage(), &Storage::I32(vec![1, 0]));
        assert_eq!(
            output,
            CpuBackend
                .execute(&direct_graph, direct_output, &bindings)
                .unwrap()
        );
        assert_eq!(
            module_graph.trace(module_output).unwrap(),
            direct_graph.trace(direct_output).unwrap()
        );
        assert!(Argmax::keepdim(1).retains_axis());
    }

    #[test]
    fn argmax_composes_for_static_cpu_classifier_inference() {
        let source_linear = Linear::new_static(2, 3, true, 131).unwrap();
        let mut source = Sequential::default();
        source.push(source_linear);
        source.push(Argmax::new(-1));
        let target_linear = Linear::new_static(2, 3, true, 137).unwrap();
        let mut target = Sequential::default();
        target.push(target_linear);
        target.push(Argmax::new(-1));
        let state = source.state_dict().unwrap();
        target.load_state_dict_strict(&state).unwrap();
        assert_eq!(target.state_dict().unwrap(), state);
        assert_eq!(
            target
                .state_dict()
                .unwrap()
                .tensors()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["0.bias", "0.weight"]
        );

        let input = TensorData::new([2, 2], vec![1., -1., 0.5, 2.]).unwrap();
        let first = infer_module_cpu(&target, input.clone()).unwrap();
        let second = infer_module_cpu(&target, input).unwrap();
        assert_eq!(first.output(), second.output());
        assert_eq!(first.trace(), second.trace());
        assert_eq!(first.output().dtype(), DType::I32);
        assert_eq!(first.output().shape().dims(), &[2]);

        let before = target.state_dict().unwrap();
        let invalid = Argmax::new(2);
        assert!(
            infer_module_cpu(&invalid, TensorData::new([1, 2], vec![0., 1.]).unwrap()).is_err()
        );
        assert_eq!(target.state_dict().unwrap(), before);
    }
}
