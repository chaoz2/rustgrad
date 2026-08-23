//! RustGrad is an inspectable tensor compiler inspired by tinygrad and built
//! around Rust's explicit ownership and backend traits.

pub mod autograd;
pub mod backend;
pub mod einsum;
pub mod error;
mod index;
pub mod ir;
pub mod kernel;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod safetensors;
pub mod schedule;
pub mod symbolic;
pub mod symbolic_shape;
pub mod tensor;
pub mod trace;
pub mod uop;

pub use backend::{Backend, CpuBackend};
pub use einsum::{EinsumLabel, EinsumPlan};
pub use error::{Error, Result};
pub use ir::pool::MaxPool2dOutput;
pub use ir::{
    AttentionOptions, BinaryOp, CompareOp, Conv2dOptions, ConvTranspose1dOptions,
    ConvTranspose2dOptions, Graph, LogicalOp, NodeId, Op, Pool2dOptions, PoolOptions, RandomKind,
    ReduceKind, Slice, UnaryOp,
};
pub use kernel::{
    BufferRole, IterationPlan, KernelBindings, KernelBufferDesc, KernelShape, execute_elementwise,
    lower_graph_elementwise,
};
pub use loss::{
    LossOptions, Reduction, binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy,
    nll_loss, sparse_categorical_cross_entropy,
};
pub use nn::{
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, BatchNorm, BatchNorm2d, BatchNormOutput, CastPolicy,
    ConvTranspose1d, ConvTranspose2d, GroupNorm, InstanceNorm, LoadReport, Mode, Module, Parameter,
    ParameterSnapshot, PendingBatchNormStats, StateDict as ModuleStateDict,
};
pub use optim::{AdamConfig, Gradient, Optimizer, OptimizerKind, ParameterGroup, SgdConfig};
pub use safetensors::{
    Metadata, StateDict, load_safetensors, load_safetensors_file, save_safetensors,
    save_safetensors_file,
};
pub use schedule::{BufferDesc, Schedule, ScheduleBoundary, ScheduleError, ScheduleItem, schedule};
pub use symbolic::{
    Bounds as SymbolicBounds, Simplified as SimplifiedSymbolicExpr, SymbolicError, SymbolicExpr,
    SymbolicVar,
};
pub use symbolic_shape::{SymbolicDim, SymbolicShape};
pub use tensor::{DType, DTypeCategory, Scalar, Shape, Storage, TensorData};
pub use trace::{CompileTrace, TraceStep};
pub use uop::{AddressSpace, Binary as UBinary, UArg, UOp, UOpError, UOpKind, UPat, UType};

#[cfg(test)]
mod attention_tests;
#[cfg(test)]
mod conv2d_tests;
#[cfg(test)]
mod creation_random_tests;
#[cfg(test)]
mod einsum_tests;
#[cfg(test)]
mod rearrange_tests;
#[cfg(test)]
mod reduction_tests;
#[cfg(test)]
mod schedule_tests;
#[cfg(test)]
mod special_functions_tests;
#[cfg(test)]
mod symbolic_tests;
#[cfg(test)]
mod uop_tests;
