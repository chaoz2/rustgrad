//! Deterministic heterogeneous module traversal.

use super::{
    Mode, ModeForwardOutput, ModeModuleForward, Module, ModuleForward, Parameter,
    PendingModeEffects, StateKind, state::join,
};
use crate::{DType, Graph, NodeId, Result};

/// A deterministic heterogeneous container for one-input, one-output modules.
///
/// Entries use [`ModuleForward`], rather than a type-name switch, so each
/// component owns its graph composition. Multi-input or stateful signatures
/// remain explicit and are intentionally not accepted here.
#[derive(Default)]
pub struct Sequential {
    modules: Vec<Box<dyn ModuleForward>>,
}
impl Sequential {
    /// Appends a statically configured single-input module.
    pub fn push(&mut self, module: impl ModuleForward + 'static) {
        self.modules.push(Box::new(module));
    }

    /// Composes its entries in insertion order.
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        self.modules
            .iter()
            .try_fold(input, |value, module| module.forward(graph, value))
    }
}
impl ModuleForward for Sequential {
    fn accepts_input_dtype(&self, dtype: DType) -> bool {
        self.modules.first().map_or(dtype == DType::F32, |module| {
            module.accepts_input_dtype(dtype)
        })
    }

    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}
impl Module for Sequential {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        for (i, module) in self.modules.iter().enumerate() {
            module.visit(&join(p, &i.to_string()), v)
        }
    }
}

/// Deterministic one-input composition with an explicit caller-selected mode.
///
/// Ordinary [`ModuleForward`] leaves participate through the state-free
/// [`ModeModuleForward`] blanket implementation, while `BatchNorm` returns its
/// pending running-stat work in insertion order.  This does not implement
/// `ModuleForward`: callers must choose training or evaluation explicitly.
#[derive(Default)]
pub struct ModeSequential {
    modules: Vec<Box<dyn ModeModuleForward>>,
}

impl ModeSequential {
    /// Appends either a stateless `ModuleForward` leaf or a module with an
    /// explicit mode-aware forward contract.
    pub fn push(&mut self, module: impl ModeModuleForward + 'static) {
        self.modules.push(Box::new(module));
    }

    /// Composes entries and aggregates pending effects in declared order.
    pub fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let mut output = input;
        let mut pending = PendingModeEffects::empty();
        for module in &self.modules {
            let later = module.forward_mode(graph, output, mode)?;
            output = later.output;
            pending.append(later.pending);
        }
        Ok(ModeForwardOutput { output, pending })
    }
}

impl ModeModuleForward for ModeSequential {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        Self::forward_mode(self, graph, input, mode)
    }
}

impl Module for ModeSequential {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        for (i, module) in self.modules.iter().enumerate() {
            module.visit(&join(p, &i.to_string()), v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{BatchNorm, Linear, ReLU};
    use crate::{Backend, CpuBackend, DType, Graph, TensorData};

    #[test]
    fn mode_sequential_mixes_stateless_modules_and_aggregates_batchnorm_effects() {
        let mut init = Graph::new();
        let mut modules = ModeSequential::default();
        modules.push(Linear::new_static(2, 2, true, 301).unwrap());
        modules.push(BatchNorm::new(&mut init, 2, 1e-5, true, true, 0.1).unwrap());
        modules.push(ReLU);

        assert_eq!(
            modules
                .state_dict()
                .unwrap()
                .tensors()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                "0.bias",
                "0.weight",
                "1.bias",
                "1.num_batches_tracked",
                "1.running_mean",
                "1.running_var",
                "1.weight",
            ]
        );

        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 2], DType::F32);
        let training = modules
            .forward_mode(&mut graph, input, Mode::Training)
            .unwrap();
        assert_eq!(training.pending.batchnorm_stat_nodes().len(), 1);
        let values = modules.input_bindings(&graph).unwrap();
        let mut values = values;
        values.insert(
            "x".into(),
            TensorData::new([2, 2], vec![1., -2., 3., 4.]).unwrap(),
        );
        let output = CpuBackend
            .execute(&graph, training.output, &values)
            .unwrap();
        assert_eq!(output.shape().dims(), &[2, 2]);

        let mut eval_graph = Graph::new();
        let eval_input = eval_graph.input_dtype("x", [2, 2], DType::F32);
        let eval = modules
            .forward_mode(&mut eval_graph, eval_input, Mode::Eval)
            .unwrap();
        assert!(eval.pending.is_empty());
    }
}
