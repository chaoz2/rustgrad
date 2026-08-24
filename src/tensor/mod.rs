//! Dense tensor values, scalar types, shapes, and owned storage.

mod data;
mod dtype;
mod scalar;
mod shape;
mod storage;

pub(crate) mod artifact;
mod creation;

pub use data::TensorData;
pub use dtype::{DType, DTypeCategory};
pub use scalar::Scalar;
pub use shape::Shape;
pub use storage::Storage;

pub(crate) use scalar::{bf16_to_f32, f16_to_f32};
