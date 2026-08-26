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

/// The checked GELU forms already supported by [`Graph::gelu`].
///
/// This finite configuration deliberately maps to the graph's canonical
/// representation rather than accepting a runtime approximation string.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GeluApproximation {
    /// tinygrad-compatible tanh approximation.
    #[default]
    Tanh,
    /// Exact error-function form.
    Exact,
}

impl GeluApproximation {
    const fn graph_approximation(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::Exact => "none",
        }
    }
}

/// A stateless Gaussian-error linear-unit adapter for static compositions.
///
/// [`Self::new`] uses tinygrad's tanh approximation. [`Self::exact`] selects
/// the existing exact error-function graph composition.
#[derive(Clone, Copy, Debug, Default)]
pub struct GELU {
    approximation: GeluApproximation,
}

impl GELU {
    /// Creates the tinygrad-compatible tanh approximation.
    pub const fn new() -> Self {
        Self::tanh()
    }

    pub const fn tanh() -> Self {
        Self {
            approximation: GeluApproximation::Tanh,
        }
    }

    /// Creates the exact error-function form.
    pub const fn exact() -> Self {
        Self {
            approximation: GeluApproximation::Exact,
        }
    }

    /// Creates GELU using one of the finite supported graph forms.
    pub const fn with_approximation(approximation: GeluApproximation) -> Self {
        Self { approximation }
    }

    pub const fn approximation(self) -> GeluApproximation {
        self.approximation
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.gelu(input, self.approximation.graph_approximation())
    }
}

impl Module for GELU {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for GELU {
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
    fn gelu_is_stateless_and_matches_direct_graph_forms() {
        let input = TensorData::new([3], vec![-1., 0., 1.]).unwrap();
        for (module, approximate) in [(GELU::new(), "tanh"), (GELU::exact(), "none")] {
            assert!(module.state_dict().unwrap().tensors().is_empty());
            assert!(module.trainable_parameters().unwrap().is_empty());

            let mut module_graph = Graph::new();
            let module_input = module_graph.input_dtype("input", [3], DType::F32);
            let module_output = module.forward(&mut module_graph, module_input).unwrap();

            let mut direct_graph = Graph::new();
            let direct_input = direct_graph.input_dtype("input", [3], DType::F32);
            let direct_output = direct_graph.gelu(direct_input, approximate).unwrap();

            let bindings = std::collections::HashMap::from([("input".into(), input.clone())]);
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
        assert_eq!(GELU::default().approximation(), GeluApproximation::Tanh);
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
    fn gelu_composes_in_static_sequential_cpu_training() {
        let mut model = Sequential::default();
        model.push(Linear::new_static(2, 3, true, 71).unwrap());
        model.push(GELU::new());
        model.push(Linear::new_static(3, 2, true, 72).unwrap());
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
            CpuBackend.execute(&graph, output, &bindings).unwrap().shape().dims(),
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
                TensorData::from_scalars(
                    [2],
                    DType::I64,
                    [Scalar::I(0), Scalar::I(1)],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(step.logits().shape().dims(), &[2, 2]);
        assert_eq!(step.optimizer_step(), 1);
        assert_eq!(step.scheduler_epoch(), 1);
        assert_ne!(model.state_dict().unwrap(), before);
    }
}
