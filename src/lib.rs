//! RustGrad is an inspectable tensor compiler inspired by tinygrad and built
//! around Rust's explicit ownership and backend traits.

pub mod autograd;
pub mod backend;
pub mod einsum;
pub mod error;
mod index;
pub mod ir;
pub mod tensor;
pub mod trace;

pub use backend::{Backend, CpuBackend};
pub use einsum::{EinsumLabel, EinsumPlan};
pub use error::{Error, Result};
pub use ir::{
    AttentionOptions, BinaryOp, CompareOp, Conv2dOptions, Graph, LogicalOp, NodeId, Op, RandomKind,
    ReduceKind, Slice, UnaryOp,
};
pub use tensor::{DType, DTypeCategory, Scalar, Shape, Storage, TensorData};
pub use trace::{CompileTrace, TraceStep};

#[cfg(test)]
mod attention_tests;
#[cfg(test)]
mod conv2d_tests;
#[cfg(test)]
mod creation_random_tests;
#[cfg(test)]
mod einsum_tests;
#[cfg(test)]
mod reduction_tests;
#[cfg(test)]
mod special_functions_tests;
