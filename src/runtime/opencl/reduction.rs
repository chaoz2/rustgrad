//! Correctness-first serial reduction geometry for OpenCL C.
use super::{OpenClCapabilities, OpenClError};
use crate::{DType, ReduceKind, Shape, UArg, UOp, UOpKind};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct OpenClReduction {
    pub input: Shape,
    pub output: Shape,
    pub axes: Vec<usize>,
    pub keepdim: bool,
    pub kind: ReduceKind,
    pub reduction_len: usize,
}

impl OpenClReduction {
    pub fn from_finalize(finalize: &UOp) -> Result<Self, OpenClError> {
        if !matches!(finalize.kind(), UOpKind::ReduceFinalize) {
            return Err(OpenClError::Unsupported(
                "reduction value lacks ReduceFinalize".into(),
            ));
        }
        let init = finalize
            .sources()
            .first()
            .and_then(|update| update.sources().first())
            .ok_or_else(|| OpenClError::Unsupported("malformed reduction chain".into()))?;
        let UArg::Reduction {
            input_shape,
            output_shape,
            axes,
            keepdim,
            kind,
            mean,
        } = init.arg()
        else {
            return Err(OpenClError::Unsupported(
                "reduction lacks typed geometry".into(),
            ));
        };
        if *mean != matches!(kind, ReduceKind::Mean)
            || axes.windows(2).any(|pair| pair[0] >= pair[1])
            || axes.iter().any(|axis| *axis >= input_shape.rank())
        {
            return Err(OpenClError::Unsupported(
                "invalid normalized reduction geometry".into(),
            ));
        }
        let mut reduction_len = 1usize;
        for axis in axes {
            reduction_len = reduction_len
                .checked_mul(input_shape.dims()[*axis])
                .ok_or(OpenClError::Overflow)?;
        }
        Ok(Self {
            input: input_shape.clone(),
            output: output_shape.clone(),
            axes: axes.clone(),
            keepdim: *keepdim,
            kind: *kind,
            reduction_len,
        })
    }

    pub fn validate_dtype(
        &self,
        dtype: DType,
        capabilities: OpenClCapabilities,
    ) -> Result<(), OpenClError> {
        if !matches!(self.kind, ReduceKind::Sum | ReduceKind::Mean) {
            return Err(OpenClError::Unsupported(format!(
                "OpenCL serial reduction does not implement {:?}",
                self.kind
            )));
        }
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(OpenClError::Unsupported(format!(
                "OpenCL exact serial {:?} does not implement {dtype:?}",
                self.kind
            )));
        }
        // RustGrad's CPU oracle accumulates floating reductions in f64, even
        // when storage is F32. Requiring fp64 avoids a silent precision split.
        if !capabilities.fp64 {
            return Err(OpenClError::Unsupported(
                "exact floating reduction requires fp64 capability".into(),
            ));
        }
        Ok(())
    }

    /// Row-major source address for output index `gid` and reduction index `r`.
    pub fn input_offset_expression(&self) -> Result<String, OpenClError> {
        let input_strides = self.input.contiguous_strides();
        let output_strides = self.output.contiguous_strides();
        let reduction_dims = self
            .axes
            .iter()
            .map(|axis| self.input.dims()[*axis])
            .collect::<Vec<_>>();
        let reduction_strides = Shape::new(reduction_dims).contiguous_strides();
        let mut terms = Vec::new();
        let mut output_axis = 0usize;
        let mut reduction_axis = 0usize;
        for (axis, input_stride) in input_strides.iter().copied().enumerate() {
            let (coord, dim) = if self.axes.binary_search(&axis).is_ok() {
                let dim = self.input.dims()[axis];
                let stride = reduction_strides[reduction_axis];
                reduction_axis += 1;
                (format!("((r / {stride}ul) % {dim}ul)"), dim)
            } else {
                let dim = self.output.dims()[output_axis];
                let stride = output_strides[output_axis];
                output_axis += 1;
                (format!("((gid / {stride}ul) % {dim}ul)"), dim)
            };
            if dim > 1 {
                terms.push(format!("({coord} * {input_stride}ul)"));
            }
            if self.keepdim && self.axes.binary_search(&axis).is_ok() {
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
