//! Exact canonical-little-endian copies between validated host layouts and
//! dense [`TensorData`](crate::TensorData) values.

use super::{BorrowedHostTensor, HostInteropError, MutableBorrowedHostTensor, OwnedHostTensor};
use crate::TensorData;

fn materialize(
    bytes: &[u8],
    view: &BorrowedHostTensor<'_>,
) -> Result<TensorData, HostInteropError> {
    let count = view.layout.logical_len()?;
    let width = view.layout.element_width();
    let byte_len = count.checked_mul(width).ok_or(HostInteropError::Overflow)?;
    let mut staged = Vec::new();
    staged
        .try_reserve_exact(byte_len)
        .map_err(|_| HostInteropError::Codec)?;
    for index in 0..count {
        let range = view
            .layout
            .logical_byte_range(bytes.len(), index)?
            .as_range();
        staged.extend_from_slice(&bytes[range]);
    }
    TensorData::from_le_bytes(view.layout.shape().clone(), view.layout.dtype(), &staged)
        .map_err(|_| HostInteropError::Codec)
}

impl BorrowedHostTensor<'_> {
    /// Copies the logical host view into owned dense storage in row-major
    /// logical order. Float and narrow-float payloads are not widened.
    pub fn to_tensor_data(&self) -> Result<TensorData, HostInteropError> {
        materialize(self.bytes, self)
    }
}

impl OwnedHostTensor {
    /// Copies this Arc-backed logical view into independent owned dense storage.
    pub fn to_tensor_data(&self) -> Result<TensorData, HostInteropError> {
        self.as_borrowed().to_tensor_data()
    }
}

impl TensorData {
    /// Copies canonical little-endian dense bytes into an injective validated
    /// host destination. Validation and source encoding finish before any
    /// destination byte is committed, so an error leaves it unchanged.
    pub fn copy_to_host(
        &self,
        destination: &mut MutableBorrowedHostTensor<'_>,
    ) -> Result<(), HostInteropError> {
        if self.dtype() != destination.layout.dtype() {
            return Err(HostInteropError::DTypeMismatch {
                source: self.dtype(),
                destination: destination.layout.dtype(),
            });
        }
        if self.shape() != destination.layout.shape() {
            return Err(HostInteropError::ShapeMismatch);
        }
        destination.layout.validate_write(destination.bytes.len())?;
        let count = destination.layout.logical_len()?;
        if self.len() != count {
            return Err(HostInteropError::ShapeMismatch);
        }
        let width = destination.layout.element_width();
        let source = self.to_le_bytes().map_err(|_| HostInteropError::Codec)?;
        let expected = count.checked_mul(width).ok_or(HostInteropError::Overflow)?;
        if source.len() != expected {
            return Err(HostInteropError::Codec);
        }
        let ranges = (0..count)
            .map(|index| {
                destination
                    .layout
                    .logical_byte_range(destination.bytes.len(), index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Every fallible operation is complete. The following commits only
        // disjoint, element-sized ranges validated above.
        for (index, range) in ranges.into_iter().enumerate() {
            let start = index * width;
            destination.bytes[range.as_range()].copy_from_slice(&source[start..start + width]);
        }
        Ok(())
    }
}
