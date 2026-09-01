//! Correctness-first serial reduction geometry for OpenCL C.
use super::{OpenClCapabilities, OpenClError};
use crate::{DType, Shape};

pub(super) fn validate_dtype(
    plan: &crate::reduction_native::NativeReductionPlan,
    dtype: DType,
    capabilities: OpenClCapabilities,
) -> Result<(), OpenClError> {
    if dtype != plan.output_dtype {
        return Err(OpenClError::Unsupported(
            "OpenCL reduction output dtype is inconsistent".into(),
        ));
    }
    if plan.source_dtype.is_float8() || plan.output_dtype.is_float8() {
        return Err(OpenClError::Unsupported(
            "Float8 reduction is outside the OpenCL exact subset".into(),
        ));
    }
    if matches!(plan.source_dtype, DType::F16 | DType::BF16) && !capabilities.fp64 {
        return Err(OpenClError::Unsupported(
            "exact narrow-float source decoding requires fp64 capability".into(),
        ));
    }
    Ok(())
}

pub(super) fn required_capabilities(
    plan: &crate::reduction_native::NativeReductionPlan,
    dtype: DType,
) -> OpenClCapabilities {
    OpenClCapabilities {
        int64: matches!(plan.source_dtype, DType::I64 | DType::U64)
            || matches!(dtype, DType::I64 | DType::U64),
        fp64: matches!(plan.source_dtype, DType::F16 | DType::BF16 | DType::F64)
            || dtype == DType::F64,
    }
}

/// Row-major source address for output index `gid` and reduction index `r`.
pub(super) fn input_offset_expression(
    plan: &crate::reduction_native::NativeReductionPlan,
) -> Result<String, OpenClError> {
    let input_strides = plan.geometry.input.contiguous_strides();
    let output_strides = plan.geometry.output.contiguous_strides();
    let reduction_dims = plan
        .geometry
        .axes
        .iter()
        .map(|axis| plan.geometry.input.dims()[*axis])
        .collect::<Vec<_>>();
    let reduction_strides = Shape::new(reduction_dims).contiguous_strides();
    let mut terms = Vec::new();
    let mut output_axis = 0usize;
    let mut reduction_axis = 0usize;
    for (axis, input_stride) in input_strides.iter().copied().enumerate() {
        let (coord, dim) = if plan.geometry.axes.binary_search(&axis).is_ok() {
            let dim = plan.geometry.input.dims()[axis];
            let stride = reduction_strides[reduction_axis];
            reduction_axis += 1;
            (format!("((r / {stride}ul) % {dim}ul)"), dim)
        } else {
            let dim = plan.geometry.output.dims()[output_axis];
            let stride = output_strides[output_axis];
            output_axis += 1;
            (format!("((gid / {stride}ul) % {dim}ul)"), dim)
        };
        if dim > 1 {
            terms.push(format!("({coord} * {input_stride}ul)"));
        }
        if plan.geometry.keepdim && plan.geometry.axes.binary_search(&axis).is_ok() {
            output_axis += 1;
        }
    }
    Ok(if terms.is_empty() {
        "0ul".into()
    } else {
        terms.join(" + ")
    })
}
