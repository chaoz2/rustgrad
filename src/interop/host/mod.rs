//! Lifetime-safe, dtype-agnostic host tensor descriptors.
//!
//! Host byte streams are always canonical little-endian, matching
//! [`TensorData::to_le_bytes`](crate::TensorData::to_le_bytes). These bridges
//! copy through validated descriptors; they are not a native-endian ABI or a
//! compute-time zero-copy facility.

mod copy;
mod layout;
mod npy;
mod view;

pub use layout::{HostInteropError, HostTensorLayout, LogicalByteRange};
pub use npy::{NpyError, decode_npy, encode_npy};
pub use view::{BorrowedHostTensor, MutableBorrowedHostTensor, OwnedHostTensor};

#[cfg(test)]
mod tests;
