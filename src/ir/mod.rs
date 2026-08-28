//! Typed frontend graph facade.
//!
//! Public IR vocabulary lives in [`types`], graph storage and generic lifecycle
//! live in [`graph`], and pure checked shape propagation lives in [`shape`].

mod attention;
mod creation;
mod dynamic;
mod elementwise;
#[cfg(test)]
mod elementwise_tests;

mod graph;
pub mod indexing;
pub mod pool;
pub mod rearrange;
mod reduce;
mod shape;
mod types;

pub(crate) use dynamic::{DynamicNode, DynamicOp};
pub(crate) use elementwise::{source_lub, source_weak_scalar_dtype};
pub use dynamic::{DynamicNodeId, DynamicOutputShape};
pub use graph::Graph;
pub(crate) use graph::Node;
pub(crate) use shape::*;
pub use types::*;
