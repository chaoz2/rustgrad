//! Deterministic heterogeneous module traversal.

use super::{Module, Parameter, StateKind, state::join};

/// A deterministic traversal-only heterogeneous container. Forward composition
/// remains explicit because Rust cannot erase differing module call signatures.
#[derive(Default)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}
impl Sequential {
    pub fn push(&mut self, module: impl Module + 'static) {
        self.modules.push(Box::new(module));
    }
}
impl Module for Sequential {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        for (i, module) in self.modules.iter().enumerate() {
            module.visit(&join(p, &i.to_string()), v)
        }
    }
}
