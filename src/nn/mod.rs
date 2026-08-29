//! Explicit module traversal and graph-independent, versioned parameters.
//!
//! A [`Parameter`] owns only host state. [`Parameter::bind`] snapshots that state
//! into a graph-local input leaf, and [`Module::input_bindings`] retrieves the
//! values captured by that graph. Replacing a parameter never mutates an
//! existing graph or changes the values its leaves observe.

mod conv;
mod embedding;
mod init;
mod linear;
mod norm;
mod parameter;
mod pool;
mod recurrent;
mod regularization;
mod sequential;
mod state;

pub use conv::{Conv1d, Conv1dOptions, Conv2d, ConvTranspose1d, ConvTranspose2d};
pub use embedding::Embedding;
pub use linear::Linear;
pub use norm::{
    BatchNorm, BatchNorm2d, BatchNormOutput, GroupNorm, InstanceNorm, LayerNorm, LayerNorm2d,
    PendingBatchNormStats, RMSNorm,
};
pub use parameter::{Parameter, ParameterId, ParameterSnapshot};
pub use pool::{AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool2d, MaxPool2d};
pub use recurrent::LSTMCell;
pub use regularization::Dropout;
pub use sequential::Sequential;
pub use state::{CastPolicy, LoadReport, Mode, Module, StateDict, StateKind, get_parameters};

pub(crate) use parameter::{ParameterRestore, next_version, restore_parameters};

#[cfg(test)]
mod layer_tests;
#[cfg(test)]
mod norm_tests;
#[cfg(test)]
mod recurrent_tests;
#[cfg(test)]
mod state_tests;
