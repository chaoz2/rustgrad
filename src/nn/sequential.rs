//! Deterministic heterogeneous module traversal.

use super::{Module, ModuleForward, Parameter, StateKind, state::join};
use crate::{Graph, NodeId, Result};

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
