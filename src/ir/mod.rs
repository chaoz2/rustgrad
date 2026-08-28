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
mod interpolate;
pub mod indexing;
pub mod pool;
pub mod rearrange;
mod reduce;
mod shape;
mod source_gather;
mod types;

pub(crate) use dynamic::{DynamicNode, DynamicOp};
pub(crate) use creation::{one_hot_bool_plan, one_hot_plan};
pub(crate) use attention::validate_log_softmax_plan;
pub(crate) use elementwise::{logsigmoid_plan, source_lub, source_weak_scalar_dtype};
pub use dynamic::{DynamicNodeId, DynamicOutputShape};
pub use graph::{Graph, PadMode, ScatterMode, ScatterReduceKind, ScatterSource};
pub(crate) use graph::Node;
pub(crate) use shape::*;
pub(crate) use source_gather::{lower_source_gather, source_gather, source_gather_plan, SourceGatherPlan};
pub use types::*;
