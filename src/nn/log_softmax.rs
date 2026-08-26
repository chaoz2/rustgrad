//! Stateless log-softmax module adapter.

use super::{Module, ModuleForward, Parameter, StateKind};
use crate::{DType, Graph, NodeId, Result};

/// A stateless stable log-softmax over one checked signed axis.
///
/// `dtype`, when selected with [`Self::with_dtype`], uses the existing graph
/// calculation/output-dtype policy. The default preserves the input dtype.
#[derive(Clone, Copy, Debug)]
pub struct LogSoftmax {
    axis: isize,
    dtype: Option<DType>,
}

impl LogSoftmax {
    /// Creates log-softmax using the input dtype for its graph calculation.
    pub const fn new(axis: isize) -> Self {
        Self { axis, dtype: None }
    }

    /// Creates log-softmax using the existing graph calculation/output dtype.
    pub const fn with_dtype(axis: isize, dtype: DType) -> Self {
        Self {
            axis,
            dtype: Some(dtype),
        }
    }

    pub const fn axis(self) -> isize {
        self.axis
    }

    pub const fn dtype(self) -> Option<DType> {
        self.dtype
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.log_softmax(input, self.axis, self.dtype)
    }
}

impl Module for LogSoftmax {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for LogSoftmax {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Sequential};
    use crate::{Backend, CpuBackend, TensorData, infer_module_cpu};

    #[test]
    fn log_softmax_is_stateless_and_matches_the_direct_graph_path() {
        let module = LogSoftmax::new(-1);
        assert!(module.state_dict().unwrap().tensors().is_empty());
        assert!(module.trainable_parameters().unwrap().is_empty());

        let input = TensorData::new([2, 3], vec![-1., 0., 1., 2., 1., 0.]).unwrap();
        let mut module_graph = Graph::new();
        let module_input = module_graph.input("input", [2, 3]);
        let module_output = module.forward(&mut module_graph, module_input).unwrap();
        let mut direct_graph = Graph::new();
        let direct_input = direct_graph.input("input", [2, 3]);
        let direct_output = direct_graph.log_softmax(direct_input, -1, None).unwrap();
        let bindings = std::collections::HashMap::from([("input".into(), input)]);
        assert_eq!(
            CpuBackend
                .execute(&module_graph, module_output, &bindings)
                .unwrap(),
            CpuBackend
                .execute(&direct_graph, direct_output, &bindings)
                .unwrap()
        );
        assert_eq!(
            module_graph.trace(module_output).unwrap(),
            direct_graph.trace(direct_output).unwrap()
        );
        assert_eq!(LogSoftmax::with_dtype(1, DType::F32).dtype(), Some(DType::F32));
    }

    #[test]
    fn log_softmax_composes_in_static_cpu_inference_without_state_mutation() {
        let source_linear = Linear::new_static(2, 2, true, 111).unwrap();
        let mut source = Sequential::default();
        source.push(source_linear);
        source.push(LogSoftmax::new(-1));
        let target_linear = Linear::new_static(2, 2, true, 117).unwrap();
        let mut target = Sequential::default();
        target.push(target_linear);
        target.push(LogSoftmax::new(-1));
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
        assert_eq!(first.output().shape().dims(), &[2, 2]);
        for row in first.output().to_vec_f64().chunks_exact(2) {
            assert!((row[0].exp() + row[1].exp() - 1.0).abs() < 1e-6);
        }
        let before = target.state_dict().unwrap();
        let invalid = LogSoftmax::new(2);
        assert!(
            infer_module_cpu(&invalid, TensorData::new([1, 2], vec![0., 1.]).unwrap()).is_err()
        );
        assert_eq!(target.state_dict().unwrap(), before);
    }
}
