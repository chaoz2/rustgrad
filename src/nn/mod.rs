//! Explicit module traversal and graph-independent, versioned parameters.
//!
//! A [`Parameter`] owns only host state. [`Parameter::bind`] snapshots that state
//! into a graph-local input leaf, and [`Module::input_bindings`] retrieves the
//! values captured by that graph. Replacing a parameter never mutates an
//! existing graph or changes the values its leaves observe.

mod activation;
mod conv;
mod embedding;
mod init;
mod linear;
mod log_softmax;
mod norm;
mod parameter;
mod pool;
mod recurrent;
mod regularization;
mod sequential;
mod shape;
mod softmax;
mod state;

pub use activation::{GELU, GeluApproximation, ReLU, SiLU, Sigmoid};
pub use conv::{Conv1d, Conv1dOptions, Conv2d, ConvTranspose1d, ConvTranspose2d};
pub use embedding::Embedding;
pub use linear::Linear;
pub use log_softmax::LogSoftmax;
pub use norm::{
    BatchNorm, BatchNorm2d, BatchNormOutput, GroupNorm, InstanceNorm, LayerNorm, LayerNorm2d,
    PendingBatchNormStats, RMSNorm,
};
pub use parameter::{Parameter, ParameterId, ParameterSnapshot};
pub use pool::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AvgPool1d,
    AvgPool2d, MaxPool1d, MaxPool2d,
};
pub use recurrent::LSTMCell;
pub use regularization::{Dropout, ModeDropout};
pub use sequential::{ModeSequential, Sequential};
pub use shape::Flatten;
pub use softmax::Softmax;
pub use state::{
    CastPolicy, LoadReport, Mode, ModeForwardOutput, ModeModuleForward, Module, ModuleForward,
    PendingModeEffects, RealizedBatchNormStats, StateDict, StateKind, StrictStateLoadLimits,
};

pub(crate) use parameter::{ParameterRestore, restore_parameters};

#[cfg(test)]
mod conv_tests;
#[cfg(test)]
mod layer_tests;
#[cfg(test)]
mod norm_tests;
#[cfg(test)]
mod pool_tests;
#[cfg(test)]
mod recurrent_tests;
#[cfg(test)]
mod state_tests;
