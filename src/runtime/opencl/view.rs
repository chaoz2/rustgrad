//! Pure validation and address lowering for static OpenCL buffer views.
use super::OpenClError;
use crate::{AffineView, DType, Shape};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct OpenClViewAccess {
    pub source_shape: Shape,
    pub logical_shape: Shape,
    pub strides: Vec<i64>,
    pub offset: i64,
}

impl OpenClViewAccess {
    pub fn new(view: &AffineView, dtype: DType) -> Result<Self, OpenClError> {
        if view.logical_shape.rank() != view.strides.len() {
            return Err(OpenClError::Unsupported("view rank/stride mismatch".into()));
        }
        view.validate_read()
            .map_err(|_| OpenClError::Unsupported("invalid signed affine read map".into()))?;
        let _byte_extent = view
            .source_shape
            .numel()
            .map_err(|_| OpenClError::Overflow)?
            .checked_mul(dtype.itemsize())
            .ok_or(OpenClError::Overflow)?;

        Ok(Self {
            source_shape: view.source_shape.clone(),
            logical_shape: view.logical_shape.clone(),
            strides: view.strides.clone(),
            offset: view.offset,
        })
    }

    pub fn expression(&self, logical: String) -> String {
        if self.offset >= 0 && self.strides.iter().all(|stride| *stride >= 0) {
            return self.unsigned_expression(logical);
        }
        self.signed_expression(logical)
    }

    fn unsigned_expression(&self, logical: String) -> String {
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

    fn signed_expression(&self, logical: String) -> String {
        let logical_strides = self.logical_shape.contiguous_strides();
        let mut terms = vec![format!("{}l", self.offset)];
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
                    "((((long)({logical}) / {logical_stride}l) % {dim}l) * {stride}l)"
                ));
            }
        }
        // `new` validates every logical lane against the source extent. The
        // cast therefore occurs only after the signed address is known valid.
        format!("((ulong)({}))", terms.join(" + "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn signed_affine_view_lowers_without_unsigned_reinterpretation() {
        let view = AffineView {
            source_shape: Shape::from([4]),
            logical_shape: Shape::from([4]),
            strides: vec![-1],
            offset: 3,
        };
        let access = OpenClViewAccess::new(&view, DType::F32).unwrap();
        assert!(access.expression("gid".into()).contains("(long)(gid)"));
    }
}
