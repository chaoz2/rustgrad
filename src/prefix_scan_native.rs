use crate::{DType, PrefixScanKind, PrefixScanOutput, PrefixScanValue, Scalar};

/// Renderer-facing, fully checked geometry and dtype contract for one
/// source-literal inclusive prefix scan. This is derived from the canonical
/// UOp payload; it does not introduce a second operation taxonomy.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct NativePrefixScanPlan {
    pub(crate) input: u64,
    pub(crate) output: u64,
    pub(crate) axis: usize,
    pub(crate) kind: PrefixScanKind,
    pub(crate) result: PrefixScanOutput,
    pub(crate) input_dtype: DType,
    pub(crate) output_dtype: DType,
    pub(crate) work_dtype: DType,
    pub(crate) elements: usize,
    pub(crate) rows: usize,
    pub(crate) axis_len: usize,
    pub(crate) inner: usize,
    pub(crate) index_sentinel: usize,
    pub(crate) scalar_identity: bool,
    work_items: usize,
}

impl NativePrefixScanPlan {
    pub(crate) fn new(value: &PrefixScanValue) -> Result<Self, &'static str> {
        if value.input == value.destination {
            return Err("native prefix scan requires distinct input and output buffers");
        }
        let work_dtype = work_dtype(value.input_dtype, value.kind, value.dtype);
        if value.input_shape != value.output_shape
            || (value.input_shape.rank() == 0 && value.axis != 0)
            || (value.input_shape.rank() != 0 && value.axis >= value.input_shape.rank())
            || crate::ir::prefix_scan_output_dtype(value.input_dtype, value.kind, value.output)
                != Some(value.dtype)
        {
            return Err("native prefix scan descriptor is inconsistent");
        }
        let elements = value
            .input_shape
            .numel()
            .map_err(|_| "native prefix scan element count overflows")?;
        elements
            .checked_mul(value.input_dtype.itemsize())
            .ok_or("native prefix scan input byte count overflows")?;
        elements
            .checked_mul(value.dtype.itemsize())
            .ok_or("native prefix scan output byte count overflows")?;
        let scalar_identity = value.input_shape.rank() == 0;
        let (axis_len, inner) = if scalar_identity {
            (1, 1)
        } else {
            let axis_len = value.input_shape.dims()[value.axis];
            let inner = value.input_shape.dims()[value.axis + 1..]
                .iter()
                .try_fold(1usize, |product, dim| product.checked_mul(*dim))
                .ok_or("native prefix scan inner domain overflows")?;
            (axis_len, inner)
        };
        let index_sentinel = if value.input_shape.rank() == 0 {
            0
        } else {
            axis_len
        };
        if value.output == PrefixScanOutput::Indices && index_sentinel > i32::MAX as usize {
            return Err("native prefix scan index sentinel exceeds I32");
        }
        let row_width = axis_len
            .checked_mul(inner)
            .ok_or("native prefix scan row width overflows")?;
        let rows = if row_width == 0 {
            0
        } else {
            elements / row_width
        };
        if rows.checked_mul(row_width) != Some(elements) {
            return Err("native prefix scan geometry is not dense");
        }
        let work_items = rows
            .checked_mul(inner)
            .ok_or("native prefix scan work domain overflows")?;
        Ok(Self {
            input: value.input.index() as u64,
            output: value.destination.index() as u64,
            axis: value.axis,
            kind: value.kind,
            result: value.output,
            input_dtype: value.input_dtype,
            output_dtype: value.dtype,
            work_dtype,
            elements,
            rows,
            axis_len,
            inner,
            index_sentinel,
            scalar_identity,
            work_items,
        })
    }

    pub(crate) fn work_items(&self) -> usize {
        self.work_items
    }

    pub(crate) fn identity(&self) -> Scalar {
        scan_identity(self.input_dtype, self.work_dtype, self.kind)
    }
}

/// Returns the exact source-visible scan identity after committing it through
/// the dtype that owns the recurrence. In particular, Float8 extrema commit
/// their infinities through the source format before the first comparison.
pub(crate) fn scan_identity(input: DType, work: DType, kind: PrefixScanKind) -> Scalar {
    match kind {
        PrefixScanKind::Sum => work.commit_scalar(Scalar::I(0)),
        PrefixScanKind::Product => work.commit_scalar(Scalar::I(1)),
        PrefixScanKind::Max => input.commit_scalar(input.min()),
        PrefixScanKind::Min => input.commit_scalar(input.max()),
    }
}

/// Tinygrad derives cumulative-extrema indices from equality against the
/// already-computed prefix value. The state starts at the axis-length sentinel,
/// moves to a strict new winner, otherwise records only the first equal source
/// lane, and therefore remains the sentinel when equality has no witness (NaN).
pub(crate) fn first_match_index(
    current: usize,
    sentinel: usize,
    coordinate: usize,
    strictly_wins: bool,
    candidate_matches_value: bool,
) -> usize {
    if strictly_wins || (current == sentinel && candidate_matches_value) {
        coordinate
    } else {
        current
    }
}

pub(crate) fn work_dtype(input: DType, kind: PrefixScanKind, output: DType) -> DType {
    match kind {
        PrefixScanKind::Sum
            if matches!(
                input,
                DType::F8E4M3
                    | DType::F8E5M2
                    | DType::F8E4M3FNUZ
                    | DType::F8E5M2FNUZ
                    | DType::F16
                    | DType::BF16
                    | DType::F32
            ) =>
        {
            DType::F32
        }
        PrefixScanKind::Sum => output,
        PrefixScanKind::Product | PrefixScanKind::Max | PrefixScanKind::Min => input,
    }
}
