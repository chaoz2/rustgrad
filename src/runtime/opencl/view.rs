//! Pure validation and address lowering for static OpenCL buffer views.
use super::OpenClError;
use crate::{DType, Shape, ViewMap};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct OpenClViewAccess {
    pub source_shape: Shape,
    pub logical_shape: Shape,
    pub offset: usize,
}

impl OpenClViewAccess {
    pub fn new(view: &ViewMap, dtype: DType) -> Result<Self, OpenClError> {
        if view.logical_shape.rank() != view.strides.len() {
            return Err(OpenClError::Unsupported("view rank/stride mismatch".into()));
        }
        let source_elements = view
            .source_shape
            .numel()
            .map_err(|_| OpenClError::Overflow)?;
        let logical_elements = view
            .logical_shape
            .numel()
            .map_err(|_| OpenClError::Overflow)?;
        let byte_offset = view
            .offset
            .checked_mul(dtype.itemsize())
            .ok_or(OpenClError::Overflow)?;
        if byte_offset % dtype.itemsize().max(1) != 0 {
            return Err(OpenClError::Unsupported("misaligned view offset".into()));
        }
        if logical_elements != 0 {
            let last = view
                .element_offset(logical_elements - 1)
                .map_err(|_| OpenClError::Unsupported("view exceeds source storage".into()))?;
            if last >= source_elements {
                return Err(OpenClError::Unsupported(
                    "view exceeds source storage".into(),
                ));
            }
        } else if view.offset > source_elements {
            return Err(OpenClError::Unsupported(
                "empty view offset exceeds source storage".into(),
            ));
        }

        // A row-major logical run may be shifted within larger source storage,
        // but no dimension may introduce a gap. Size-one axes are irrelevant.
        let mut expected = 1usize;
        for axis in (0..view.logical_shape.rank()).rev() {
            let dim = view.logical_shape.dims()[axis];
            if dim > 1 && view.strides[axis] != expected {
                return Err(OpenClError::Unsupported(
                    "non-contiguous static view".into(),
                ));
            }
            expected = expected.checked_mul(dim).ok_or(OpenClError::Overflow)?;
        }
        Ok(Self {
            source_shape: view.source_shape.clone(),
            logical_shape: view.logical_shape.clone(),
            offset: view.offset,
        })
    }

    pub fn expression(&self, logical: String) -> String {
        if self.logical_shape.numel().ok() == Some(1) {
            format!("{}ul", self.offset)
        } else if self.offset == 0 {
            logical
        } else {
            format!("({}ul + ({logical}))", self.offset)
        }
    }
}
