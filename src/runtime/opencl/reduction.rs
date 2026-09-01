//! Correctness-first serial reduction geometry for OpenCL C.
use super::{OpenClCapabilities, OpenClError};
use crate::{DType, Operation, Shape, UOp};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct OpenClReduction {
    pub plan: crate::reduction_native::NativeReductionPlan,
}

impl OpenClReduction {
    pub(super) fn producer<'a>(&self, finalize: &'a UOp) -> Result<&'a UOp, OpenClError> {
        finalize
            .sources()
            .first()
            .and_then(|update| update.sources().get(1))
            .ok_or_else(|| OpenClError::Unsupported("malformed reduction value".into()))
    }

    pub fn from_finalize(finalize: &UOp) -> Result<Self, OpenClError> {
        if !matches!(finalize.operation(), Operation::ReduceFinalize) {
            return Err(OpenClError::Unsupported(
                "reduction value lacks ReduceFinalize".into(),
            ));
        }
        let (plan, _) = crate::reduction_native::NativeReductionPlan::from_finalize(finalize)
            .map_err(|reason| OpenClError::Unsupported(reason.into()))?;
        Ok(Self { plan })
    }

    pub fn validate_dtype(
        &self,
        dtype: DType,
        capabilities: OpenClCapabilities,
    ) -> Result<(), OpenClError> {
        if dtype != self.plan.output_dtype {
            return Err(OpenClError::Unsupported(
                "OpenCL reduction output dtype is inconsistent".into(),
            ));
        }
        if self.plan.source_dtype.is_float8() || self.plan.output_dtype.is_float8() {
            return Err(OpenClError::Unsupported(
                "Float8 reduction is outside the OpenCL exact subset".into(),
            ));
        }
        if matches!(self.plan.source_dtype, DType::F16 | DType::BF16) && !capabilities.fp64 {
            return Err(OpenClError::Unsupported(
                "exact narrow-float source decoding requires fp64 capability".into(),
            ));
        }
        Ok(())
    }

    pub fn required_capabilities(&self, dtype: DType) -> OpenClCapabilities {
        OpenClCapabilities {
            int64: matches!(self.plan.source_dtype, DType::I64 | DType::U64)
                || matches!(dtype, DType::I64 | DType::U64),
            fp64: matches!(
                self.plan.source_dtype,
                DType::F16 | DType::BF16 | DType::F64
            ) || dtype == DType::F64,
        }
    }

    /// Row-major source address for output index `gid` and reduction index `r`.
    pub fn input_offset_expression(&self) -> Result<String, OpenClError> {
        let input_strides = self.plan.geometry.input.contiguous_strides();
        let output_strides = self.plan.geometry.output.contiguous_strides();
        let reduction_dims = self
            .plan
            .geometry
            .axes
            .iter()
            .map(|axis| self.plan.geometry.input.dims()[*axis])
            .collect::<Vec<_>>();
        let reduction_strides = Shape::new(reduction_dims).contiguous_strides();
        let mut terms = Vec::new();
        let mut output_axis = 0usize;
        let mut reduction_axis = 0usize;
        for (axis, input_stride) in input_strides.iter().copied().enumerate() {
            let (coord, dim) = if self.plan.geometry.axes.binary_search(&axis).is_ok() {
                let dim = self.plan.geometry.input.dims()[axis];
                let stride = reduction_strides[reduction_axis];
                reduction_axis += 1;
                (format!("((r / {stride}ul) % {dim}ul)"), dim)
            } else {
                let dim = self.plan.geometry.output.dims()[output_axis];
                let stride = output_strides[output_axis];
                output_axis += 1;
                (format!("((gid / {stride}ul) % {dim}ul)"), dim)
            };
            if dim > 1 {
                terms.push(format!("({coord} * {input_stride}ul)"));
            }
            if self.plan.geometry.keepdim && self.plan.geometry.axes.binary_search(&axis).is_ok() {
                output_axis += 1;
            }
        }
        Ok(if terms.is_empty() {
            "0ul".into()
        } else {
            terms.join(" + ")
        })
    }
}
