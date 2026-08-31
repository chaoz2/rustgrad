//! Typed frontend graph facade.
//!
//! Public IR vocabulary lives in [`types`], graph storage and generic lifecycle
//! live in [`graph`], and pure checked shape propagation lives in [`shape`].

mod attention;
mod convolution;
mod creation;
mod dynamic;
mod elementwise;
#[cfg(test)]
mod elementwise_tests;

mod graph;
pub mod indexing;
mod interpolate;
mod multinomial;
pub mod pool;
pub mod rearrange;
mod reduce;
mod shape;
mod source_gather;
mod types;

pub(crate) use attention::validate_log_softmax_plan;
pub use convolution::{ConvolutionSpec, SpatialWindow, SpatialWindowError};
pub use creation::PendingRandomReservation;
pub(crate) use creation::{one_hot_bool_plan, one_hot_plan};
pub use dynamic::{
    DynamicAllocation, DynamicAllocationError, DynamicAllocationPlan, DynamicAllocationTarget,
    DynamicBinding, DynamicCountStage, DynamicInput, DynamicNodeId, DynamicOutputShape,
};
pub(crate) use dynamic::{DynamicNode, DynamicOperation, dynamic_reduction_dtypes};
pub(crate) use elementwise::{logsigmoid_plan, source_lub, source_weak_scalar_dtype};
pub(crate) use graph::Node;
pub use graph::{
    Graph, GraphSequentialTransform, PadMode, ScatterMode, ScatterReduceKind, ScatterSource,
};
pub(crate) use multinomial::MultinomialPlan;
pub use rearrange::SplitSizes;
pub(crate) use shape::*;
pub(crate) use source_gather::{
    SourceGatherPlan, lower_source_gather, source_gather, source_gather_plan,
};
pub use types::*;
