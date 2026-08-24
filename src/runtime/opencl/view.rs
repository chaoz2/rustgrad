//! Pure validation and address lowering for static OpenCL buffer views.
use super::OpenClError;
use crate::{DType, Shape, ViewMap};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct OpenClViewAccess {
    pub source_shape: Shape,
    pub logical_shape: Shape,
    pub strides: Vec<usize>,
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

        Ok(Self {
            source_shape: view.source_shape.clone(),
            logical_shape: view.logical_shape.clone(),
            strides: view.strides.clone(),
            offset: view.offset,
        })
    }

    pub fn expression(&self, logical: String) -> String {
        if self.logical_shape.numel().ok() == Some(1) {
            return format!("{}ul", self.offset);
        }
        let logical_strides = self.logical_shape.contiguous_strides();
        let mut terms = Vec::new();
        if self.offset != 0 {
            terms.push(format!("{}ul", self.offset));
        }
        for ((dim, stride), logical_stride) in self
            .logical_shape
            .dims()
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .zip(logical_strides)
        {
            if dim > 1 && stride != 0 {
                terms.push(format!(
                    "((({logical}) / {}ul) % {dim}ul) * {stride}ul",
                    logical_stride
                ));
            }
        }
        if terms.is_empty() {
            "0ul".into()
        } else {
            format!("({})", terms.join(" + "))
        }
    }
}
