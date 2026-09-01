use crate::{DType, PrefixScanKind, PrefixScanOutput, PrefixScanValue, Scalar};
use std::fmt;

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PortablePrefixScanError {
    InvalidPlan(String),
    InvalidBinding(String),
    Unsupported(&'static str),
    Overflow,
}

impl fmt::Display for PortablePrefixScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(f, "invalid prefix-scan payload: {reason}"),
            Self::InvalidBinding(reason) => write!(f, "invalid prefix-scan binding: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported prefix scan: {reason}"),
            Self::Overflow => f.write_str("prefix-scan geometry overflow"),
        }
    }
}

impl std::error::Error for PortablePrefixScanError {}

/// Checked common projection for correctness-first static accelerator scans.
///
/// The existing `PrefixScanKind` and `NativePrefixScanPlan` remain the only
/// semantic taxonomy. Portable renderers consume this view solely to share the
/// exact dense two-buffer ABI, 32-bit indexing proof, recurrence dtype and
/// scalar/empty-domain contract before formatting backend syntax.
pub(crate) struct PortablePrefixScan<'a> {
    value: &'a PrefixScanValue,
    plan: NativePrefixScanPlan,
}

impl<'a> PortablePrefixScan<'a> {
    pub(crate) fn new(value: &'a PrefixScanValue) -> Result<Self, PortablePrefixScanError> {
        let plan = NativePrefixScanPlan::new(value)
            .map_err(|reason| PortablePrefixScanError::InvalidPlan(reason.into()))?;
        if !matches!(
            plan.input_dtype,
            DType::Bool | DType::I32 | DType::U32 | DType::F32
        ) || !matches!(
            plan.output_dtype,
            DType::Bool | DType::I32 | DType::U32 | DType::F32
        ) {
            return Err(PortablePrefixScanError::Unsupported(
                "portable scan requires Bool/I32/U32/F32 storage",
            ));
        }
        if [
            plan.elements,
            plan.rows,
            plan.axis_len,
            plan.inner,
            plan.work_items(),
        ]
        .into_iter()
        .any(|extent| extent > u32::MAX as usize)
        {
            return Err(PortablePrefixScanError::Unsupported(
                "portable scan requires a 32-bit dense domain",
            ));
        }
        Ok(Self { value, plan })
    }

    pub(crate) fn value(&self) -> &'a PrefixScanValue {
        self.value
    }

    pub(crate) fn plan(&self) -> &NativePrefixScanPlan {
        &self.plan
    }

    pub(crate) fn launch_extent(&self) -> usize {
        self.plan.work_items()
    }

    pub(crate) fn strict_operator(&self) -> Option<&'static str> {
        match self.plan.kind {
            PrefixScanKind::Max => Some(">"),
            PrefixScanKind::Min => Some("<"),
            PrefixScanKind::Sum | PrefixScanKind::Product => None,
        }
    }

    pub(crate) fn arithmetic_operator(&self) -> Option<&'static str> {
        match self.plan.kind {
            PrefixScanKind::Sum => Some("+"),
            PrefixScanKind::Product => Some("*"),
            PrefixScanKind::Max | PrefixScanKind::Min => None,
        }
    }
}

/// Backend syntax hooks for the shared serial portable-scan program.
///
/// The coordinator below owns the recurrence, first-witness rule, and loop
/// topology. Implementations only format storage-language syntax.
pub(crate) trait PortablePrefixScanDialect {
    fn scalar_body(
        &self,
        plan: &NativePrefixScanPlan,
    ) -> Result<Vec<String>, PortablePrefixScanError>;
    fn domain(&self, plan: &NativePrefixScanPlan) -> Vec<String>;
    fn identity(
        &self,
        plan: &NativePrefixScanPlan,
    ) -> Result<&'static str, PortablePrefixScanError>;
    fn accumulator(&self, plan: &NativePrefixScanPlan, identity: &str) -> String;
    fn index(&self, plan: &NativePrefixScanPlan) -> String;
    fn loop_open(&self, plan: &NativePrefixScanPlan) -> String;
    fn offset(&self, plan: &NativePrefixScanPlan) -> String;
    fn load(&self, plan: &NativePrefixScanPlan) -> Result<String, PortablePrefixScanError>;
    fn strict(&self, plan: &NativePrefixScanPlan, operator: &str) -> String;
    fn equal_before(&self) -> String;
    fn update_extrema(&self) -> String;
    fn update_first_index(&self, plan: &NativePrefixScanPlan) -> String;
    fn arithmetic(
        &self,
        plan: &NativePrefixScanPlan,
        operator: &str,
    ) -> Result<String, PortablePrefixScanError>;
    fn store(&self, plan: &NativePrefixScanPlan) -> Vec<String>;
    fn loop_close(&self) -> String;
}

/// Emits one work-item's serial scan body from the checked shared plan.
pub(crate) fn emit_portable_prefix_scan_body(
    portable: &PortablePrefixScan<'_>,
    dialect: &impl PortablePrefixScanDialect,
) -> Result<Vec<String>, PortablePrefixScanError> {
    let plan = portable.plan();
    if portable.launch_extent() == 0 {
        return Ok(Vec::new());
    }
    if plan.scalar_identity {
        return dialect.scalar_body(plan);
    }
    let identity = dialect.identity(plan)?;
    let mut lines = dialect.domain(plan);
    lines.push(dialect.accumulator(plan, identity));
    if plan.result == PrefixScanOutput::Indices {
        lines.push(dialect.index(plan));
    }
    lines.extend([
        dialect.loop_open(plan),
        dialect.offset(plan),
        dialect.load(plan)?,
    ]);
    if let Some(operator) = portable.strict_operator() {
        // Equality is observed against the pre-update accumulator. This makes
        // the index state an explicit first source witness while unordered
        // candidates (notably NaN) retain the sentinel.
        lines.extend([
            dialect.strict(plan, operator),
            dialect.equal_before(),
            dialect.update_extrema(),
        ]);
        if plan.result == PrefixScanOutput::Indices {
            lines.push(dialect.update_first_index(plan));
        }
    } else {
        lines.push(
            dialect.arithmetic(
                plan,
                portable
                    .arithmetic_operator()
                    .expect("arithmetic scan kind"),
            )?,
        );
    }
    lines.extend(dialect.store(plan));
    lines.push(dialect.loop_close());
    Ok(lines)
}
