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

/// A stateless sigmoid-linear-unit adapter for static compositions.
///
/// Its graph semantics are exactly [`Graph::silu`], the tinygrad-compatible
/// `input * sigmoid(input)` composition.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiLU;

impl SiLU {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.silu(input)
    }
}

impl Module for SiLU {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for SiLU {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// A stateless sigmoid adapter for one-input static compositions.
///
/// Its graph semantics are exactly [`Graph::sigmoid`], retaining the
/// established inspectable tinygrad-style composition and reverse mode.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sigmoid;

impl Sigmoid {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.sigmoid(input)
    }
}

impl Module for Sigmoid {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for Sigmoid {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Sequential};
    use crate::optim::{LearningRateScheduler, Optimizer, SgdConfig};
    use crate::{
        Backend, CpuBackend, CpuModuleTrainer, DType, ModuleCrossEntropy, Scalar, TensorData,
    };

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

    #[test]
    fn silu_is_stateless_and_matches_direct_graph_composition() {
        let module = SiLU::new();
        assert!(module.state_dict().unwrap().tensors().is_empty());
        assert!(module.trainable_parameters().unwrap().is_empty());

        let input = TensorData::new([3], vec![-1., 0., 1.]).unwrap();
        let mut module_graph = Graph::new();
        let module_input = module_graph.input_dtype("input", [3], DType::F32);
        let module_output = module.forward(&mut module_graph, module_input).unwrap();

        let mut direct_graph = Graph::new();
        let direct_input = direct_graph.input_dtype("input", [3], DType::F32);
        let direct_output = direct_graph.silu(direct_input).unwrap();

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
    }

    #[test]
    fn silu_composes_in_static_sequential_cpu_training() {
        let mut model = Sequential::default();
        model.push(Linear::new_static(2, 3, true, 81).unwrap());
        model.push(SiLU::new());
        model.push(Linear::new_static(3, 2, true, 82).unwrap());
        assert_eq!(
            model
                .state_dict()
                .unwrap()
                .tensors()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["0.bias", "0.weight", "2.bias", "2.weight"]
        );

        let input = TensorData::new([2, 2], vec![1., -1., 0.5, 2.]).unwrap();
        let mut graph = Graph::new();
        let node = graph.input_dtype("input", input.shape().clone(), DType::F32);
        let output = model.forward(&mut graph, node).unwrap();
        let mut bindings = model.input_bindings(&graph).unwrap();
        bindings.insert("input".into(), input.clone());
        assert_eq!(
            CpuBackend
                .execute(&graph, output, &bindings)
                .unwrap()
                .shape()
                .dims(),
            &[2, 2]
        );

        let mut optimizer = Optimizer::sgd_for_module(&model, SgdConfig::default()).unwrap();
        let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.).unwrap();
        let before = model.state_dict().unwrap();
        let mut trainer = CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )
        .unwrap();
        let step = trainer
            .train_step(
                input,
                TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(1)]).unwrap(),
            )
            .unwrap();
        assert_eq!(step.logits().shape().dims(), &[2, 2]);
        assert_eq!(step.optimizer_step(), 1);
        assert_eq!(step.scheduler_epoch(), 1);
        assert_ne!(model.state_dict().unwrap(), before);
    }

    #[test]
    fn sigmoid_is_stateless_and_matches_direct_graph_composition() {
        let module = Sigmoid::new();
        assert!(module.state_dict().unwrap().tensors().is_empty());
        assert!(module.trainable_parameters().unwrap().is_empty());

        let input = TensorData::new([3], vec![-1., 0., 1.]).unwrap();
        let mut module_graph = Graph::new();
        let module_input = module_graph.input_dtype("input", [3], DType::F32);
        let module_output = module.forward(&mut module_graph, module_input).unwrap();

        let mut direct_graph = Graph::new();
        let direct_input = direct_graph.input_dtype("input", [3], DType::F32);
        let direct_output = direct_graph.sigmoid(direct_input).unwrap();

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
    }

    #[test]
    fn sigmoid_composes_in_static_sequential_cpu_training() {
        let mut model = Sequential::default();
        model.push(Linear::new_static(2, 3, true, 91).unwrap());
        model.push(Sigmoid::new());
        model.push(Linear::new_static(3, 2, true, 92).unwrap());
        assert_eq!(
            model
                .state_dict()
                .unwrap()
                .tensors()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["0.bias", "0.weight", "2.bias", "2.weight"]
        );

        let input = TensorData::new([2, 2], vec![1., -1., 0.5, 2.]).unwrap();
        let mut graph = Graph::new();
        let node = graph.input_dtype("input", input.shape().clone(), DType::F32);
        let output = model.forward(&mut graph, node).unwrap();
        let mut bindings = model.input_bindings(&graph).unwrap();
        bindings.insert("input".into(), input.clone());
        assert_eq!(
            CpuBackend
                .execute(&graph, output, &bindings)
                .unwrap()
                .shape()
                .dims(),
            &[2, 2]
        );

        let mut optimizer = Optimizer::sgd_for_module(&model, SgdConfig::default()).unwrap();
        let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.).unwrap();
        let before = model.state_dict().unwrap();
        let mut trainer = CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )
        .unwrap();
        let step = trainer
            .train_step(
                input,
                TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(1)]).unwrap(),
            )
            .unwrap();
        assert_eq!(step.logits().shape().dims(), &[2, 2]);
        assert_eq!(step.optimizer_step(), 1);
        assert_eq!(step.scheduler_epoch(), 1);
        assert_ne!(model.state_dict().unwrap(), before);
    }
}
