use super::{HostInteropError, HostTensorLayout, LogicalByteRange};
use std::sync::Arc;

/// A lifetime-bound immutable host view. Its construction validates the exact
/// byte layout; no raw pointer or physical-capacity API is exposed.
pub struct BorrowedHostTensor<'a> {
    bytes: &'a [u8],
    layout: HostTensorLayout,
}

impl<'a> BorrowedHostTensor<'a> {
    pub fn new(bytes: &'a [u8], layout: HostTensorLayout) -> Result<Self, HostInteropError> {
        layout.validate_read(bytes.len())?;
        Ok(Self { bytes, layout })
    }
    pub fn layout(&self) -> &HostTensorLayout {
        &self.layout
    }
    pub fn logical_byte_range(&self, index: usize) -> Result<LogicalByteRange, HostInteropError> {
        self.layout.logical_byte_range(self.bytes.len(), index)
    }
}

/// An immutable ref-counted host view. Cloning retains the same bytes and
/// descriptor without copying; callers can only inspect logical ranges.
#[derive(Clone)]
pub struct OwnedHostTensor {
    bytes: Arc<[u8]>,
    layout: HostTensorLayout,
}

impl OwnedHostTensor {
    pub fn new(bytes: Arc<[u8]>, layout: HostTensorLayout) -> Result<Self, HostInteropError> {
        layout.validate_read(bytes.len())?;
        Ok(Self { bytes, layout })
    }
    pub fn layout(&self) -> &HostTensorLayout {
        &self.layout
    }
    pub fn as_borrowed(&self) -> BorrowedHostTensor<'_> {
        BorrowedHostTensor {
            bytes: &self.bytes,
            layout: self.layout.clone(),
        }
    }
    pub fn logical_byte_range(&self, index: usize) -> Result<LogicalByteRange, HostInteropError> {
        self.layout.logical_byte_range(self.bytes.len(), index)
    }
}
