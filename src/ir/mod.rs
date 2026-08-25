//! Typed frontend graph facade.
//!
//! Public IR vocabulary lives in [`types`], graph storage and generic lifecycle
//! live in [`graph`], and pure checked shape propagation lives in [`shape`].

mod attention;
mod creation;
mod dynamic;
mod elementwise;

mod graph;
pub mod indexing;
pub mod pool;
pub mod rearrange;
mod reduce;
mod shape;
mod types;

pub use dynamic::{DynamicInput, DynamicNodeId, DynamicOutputShape};
pub(crate) use dynamic::{DynamicNode, DynamicOp};
pub use graph::Graph;
pub(crate) use graph::Node;
pub use rearrange::SplitSizes;
pub(crate) use shape::*;
pub use types::*;
