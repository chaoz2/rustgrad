//! Lifetime-safe, dtype-agnostic host tensor descriptors.
//!
//! These types validate host byte layouts and logical addressing only. They do
//! not materialize `TensorData`, provide mutable destination access, or expose
//! a raw pointer/backing-capacity ABI.

mod layout;
mod view;

pub use layout::{HostInteropError, HostTensorLayout, LogicalByteRange};
pub use view::{BorrowedHostTensor, OwnedHostTensor};

#[cfg(test)]
mod tests;
