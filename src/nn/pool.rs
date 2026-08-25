//! Stateless pooling module adapters.

use super::{Module, ModuleForward, Parameter, StateKind};
use crate::{Graph, NodeId, Result};

/// Stateless 2D max-pooling module. Index-returning calls use the typed
/// specialized method because a regular `Module` forward has one tensor output.
#[derive(Clone, Copy, Debug)]
pub struct MaxPool2d {
    pub options: crate::Pool2dOptions,
}
impl MaxPool2d {
    pub fn new(options: crate::Pool2dOptions) -> Self {
        Self { options }
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.max_pool2d(input, self.options)
    }
    pub fn forward_with_indices(
        &self,
        graph: &mut Graph,
        input: NodeId,
    ) -> Result<crate::ir::pool::MaxPool2dOutput> {
        graph.max_pool2d_with_indices(input, self.options)
    }
}
impl Module for MaxPool2d {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}
impl ModuleForward for MaxPool2d {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// Stateless 2D average-pooling module.
#[derive(Clone, Copy, Debug)]
pub struct AvgPool2d {
    pub options: crate::Pool2dOptions,
}
impl AvgPool2d {
    pub fn new(options: crate::Pool2dOptions) -> Self {
        Self { options }
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.avg_pool2d(input, self.options)
    }
}
impl Module for AvgPool2d {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveAvgPool2d {
    pub output_size: [Option<usize>; 2],
}
impl AdaptiveAvgPool2d {
    pub fn new(output_size: [Option<usize>; 2]) -> Self {
        Self { output_size }
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.adaptive_avg_pool2d(input, self.output_size)
    }
}
impl Module for AdaptiveAvgPool2d {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}
impl ModuleForward for AdaptiveAvgPool2d {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveMaxPool2d {
    pub output_size: [Option<usize>; 2],
}
impl AdaptiveMaxPool2d {
    pub fn new(output_size: [Option<usize>; 2]) -> Self {
        Self { output_size }
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        graph.adaptive_max_pool2d(input, self.output_size)
    }
}
impl Module for AdaptiveMaxPool2d {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}
