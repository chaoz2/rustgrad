//! Checked backend-neutral projection for tinygrad's coupled stable Sort pair.

#[cfg(test)]
use crate::TensorData;
use crate::{DType, ScheduleInputBinding, SortValue};
use std::fmt;

const MAX_PORTABLE_SORT_AXIS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PortableSortCompare {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) left_takes_larger: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum PortableSortStep {
    Swap { left: usize, right: usize },
    Compare(PortableSortCompare),
}

/// Backend syntax hooks for the shared serial bitonic/occurrence-count program.
///
/// The coordinator owns every semantic step and its order. Implementations
/// only spell declarations, loads, comparisons, and stores in one target
/// language; they cannot substitute a host or backend-specific ordering.
pub(crate) trait PortableSortDialect {
    fn domain(&self, plan: &PortableSortPair<'_>) -> Vec<String>;
    fn storage(&self, plan: &PortableSortPair<'_>) -> Result<Vec<String>, PortableSortError>;
    fn load_original(
        &self,
        plan: &PortableSortPair<'_>,
        lane: usize,
    ) -> Result<Vec<String>, PortableSortError>;
    fn pad_work(
        &self,
        plan: &PortableSortPair<'_>,
        lane: usize,
    ) -> Result<String, PortableSortError>;
    fn swap(&self, plan: &PortableSortPair<'_>, left: usize, right: usize) -> Vec<String>;
    fn compare(&self, plan: &PortableSortPair<'_>, step: PortableSortCompare) -> Vec<String>;
    fn count_original_open(&self, plan: &PortableSortPair<'_>) -> Vec<String>;
    fn count_original_step(&self) -> String;
    fn count_original_close(&self) -> Vec<String>;
    fn count_sorted_open(&self, plan: &PortableSortPair<'_>) -> Vec<String>;
    fn count_sorted_step(&self) -> String;
    fn count_sorted_close(&self) -> Vec<String>;
    fn reconstruct_open(&self, plan: &PortableSortPair<'_>) -> Vec<String>;
    fn reconstruct_step(&self) -> String;
    fn reconstruct_store(
        &self,
        plan: &PortableSortPair<'_>,
    ) -> Result<Vec<String>, PortableSortError>;
    fn reconstruct_close(&self) -> Vec<String>;
}

/// Emits one serial work item's complete coupled values/indices program.
pub(crate) fn emit_portable_sort_body(
    plan: &PortableSortPair<'_>,
    dialect: &impl PortableSortDialect,
) -> Result<Vec<String>, PortableSortError> {
    if plan.launch_extent() == 0 {
        return Ok(Vec::new());
    }
    let mut lines = dialect.domain(plan);
    lines.extend(dialect.storage(plan)?);
    for lane in 0..plan.axis_len() {
        lines.extend(dialect.load_original(plan, lane)?);
    }
    for lane in plan.axis_len()..plan.padded_len() {
        lines.push(dialect.pad_work(plan, lane)?);
    }
    for &step in plan.steps() {
        lines.extend(match step {
            PortableSortStep::Swap { left, right } => dialect.swap(plan, left, right),
            PortableSortStep::Compare(compare) => dialect.compare(plan, compare),
        });
    }
    lines.extend(dialect.count_original_open(plan));
    lines.push(dialect.count_original_step());
    lines.extend(dialect.count_original_close());
    lines.extend(dialect.count_sorted_open(plan));
    lines.push(dialect.count_sorted_step());
    lines.extend(dialect.count_sorted_close());
    lines.extend(dialect.reconstruct_open(plan));
    lines.push(dialect.reconstruct_step());
    lines.extend(dialect.reconstruct_store(plan)?);
    lines.extend(dialect.reconstruct_close());
    Ok(lines)
}

/// Shared C-family spelling used by OpenCL C and MSL. Both targets expose
/// dense byte/32-bit arrays and differ only in the already-validated scalar
/// type and infinity literal supplied by their renderer.
pub(crate) struct CLikePortableSortDialect {
    pub(crate) scalar_type: &'static str,
    pub(crate) padding: String,
}

impl PortableSortDialect for CLikePortableSortDialect {
    fn domain(&self, plan: &PortableSortPair<'_>) -> Vec<String> {
        vec![
            format!(
                "  const uint rg_row = (uint)(gid / (ulong){}ul);",
                plan.inner()
            ),
            format!(
                "  const uint rg_inner = (uint)(gid % (ulong){}ul);",
                plan.inner()
            ),
        ]
    }

    fn storage(&self, plan: &PortableSortPair<'_>) -> Result<Vec<String>, PortableSortError> {
        let lanes = plan.padded_len().max(1);
        let axis = plan.axis_len().max(1);
        Ok(vec![
            format!("  {} rg_original[{axis}];", self.scalar_type),
            format!("  {} rg_work[{lanes}];", self.scalar_type),
            format!("  int rg_original_count[{axis}];"),
            format!("  int rg_sorted_count[{axis}];"),
        ])
    }

    fn load_original(
        &self,
        plan: &PortableSortPair<'_>,
        lane: usize,
    ) -> Result<Vec<String>, PortableSortError> {
        Ok(vec![
            format!(
                "  rg_original[{lane}] = b0[((ulong)rg_row * (ulong){}ul + (ulong){lane}ul) * (ulong){}ul + (ulong)rg_inner];",
                plan.axis_len(),
                plan.inner()
            ),
            format!("  rg_work[{lane}] = rg_original[{lane}];"),
        ])
    }

    fn pad_work(
        &self,
        _plan: &PortableSortPair<'_>,
        lane: usize,
    ) -> Result<String, PortableSortError> {
        Ok(format!("  rg_work[{lane}] = {};", self.padding))
    }

    fn swap(&self, _plan: &PortableSortPair<'_>, left: usize, right: usize) -> Vec<String> {
        vec![
            "  {".into(),
            format!("    const {} rg_swap = rg_work[{left}];", self.scalar_type),
            format!("    rg_work[{left}] = rg_work[{right}];"),
            format!("    rg_work[{right}] = rg_swap;"),
            "  }".into(),
        ]
    }

    fn compare(&self, _plan: &PortableSortPair<'_>, step: PortableSortCompare) -> Vec<String> {
        let left = step.left;
        let right = step.right;
        let (first, second) = if step.left_takes_larger {
            ("rg_larger", "rg_smaller")
        } else {
            ("rg_smaller", "rg_larger")
        };
        vec![
            "  {".into(),
            format!("    const {} rg_left = rg_work[{left}];", self.scalar_type),
            format!(
                "    const {} rg_right = rg_work[{right}];",
                self.scalar_type
            ),
            format!(
                "    const {} rg_larger = rg_right > rg_left ? rg_right : rg_left;",
                self.scalar_type
            ),
            format!(
                "    const {} rg_smaller = rg_right < rg_left ? rg_right : rg_left;",
                self.scalar_type
            ),
            format!("    rg_work[{left}] = {first};"),
            format!("    rg_work[{right}] = {second};"),
            "  }".into(),
        ]
    }

    fn count_original_open(&self, plan: &PortableSortPair<'_>) -> Vec<String> {
        vec![
            format!(
                "  for (uint rg_i = 0u; rg_i < {}u; ++rg_i) {{",
                plan.axis_len()
            ),
            "    int rg_count = 0;".into(),
            "    for (uint rg_j = 0u; rg_j <= rg_i; ++rg_j) {".into(),
        ]
    }

    fn count_original_step(&self) -> String {
        "      if (rg_original[rg_j] == rg_original[rg_i]) ++rg_count;".into()
    }

    fn count_original_close(&self) -> Vec<String> {
        vec![
            "    }".into(),
            "    rg_original_count[rg_i] = rg_count;".into(),
            "  }".into(),
        ]
    }

    fn count_sorted_open(&self, plan: &PortableSortPair<'_>) -> Vec<String> {
        vec![
            format!(
                "  for (uint rg_i = 0u; rg_i < {}u; ++rg_i) {{",
                plan.axis_len()
            ),
            "    int rg_count = 0;".into(),
            "    for (uint rg_j = 0u; rg_j <= rg_i; ++rg_j) {".into(),
        ]
    }

    fn count_sorted_step(&self) -> String {
        "      if (rg_work[rg_j] == rg_work[rg_i]) ++rg_count;".into()
    }

    fn count_sorted_close(&self) -> Vec<String> {
        vec![
            "    }".into(),
            "    rg_sorted_count[rg_i] = rg_count;".into(),
            "  }".into(),
        ]
    }

    fn reconstruct_open(&self, plan: &PortableSortPair<'_>) -> Vec<String> {
        vec![
            format!(
                "  for (uint rg_out = 0u; rg_out < {}u; ++rg_out) {{",
                plan.axis_len()
            ),
            "    int rg_index = 0;".into(),
            format!(
                "    for (uint rg_in = 0u; rg_in < {}u; ++rg_in) {{",
                plan.axis_len()
            ),
        ]
    }

    fn reconstruct_step(&self) -> String {
        "      if (rg_original[rg_in] == rg_work[rg_out] && rg_original_count[rg_in] == rg_sorted_count[rg_out]) rg_index += (int)rg_in;".into()
    }

    fn reconstruct_store(
        &self,
        plan: &PortableSortPair<'_>,
    ) -> Result<Vec<String>, PortableSortError> {
        Ok(vec![
            "    }".into(),
            format!(
                "    const ulong rg_offset = ((ulong)rg_row * (ulong){}ul + (ulong)rg_out) * (ulong){}ul + (ulong)rg_inner;",
                plan.axis_len(),
                plan.inner()
            ),
            "    b1[rg_offset] = rg_work[rg_out];".into(),
            "    b2[rg_offset] = rg_index;".into(),
        ])
    }

    fn reconstruct_close(&self) -> Vec<String> {
        vec!["  }".into()]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PortableSortError {
    InvalidPlan(String),
    InvalidBinding(String),
    Unsupported(&'static str),
    Overflow,
}

impl fmt::Display for PortableSortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(f, "invalid sort payload: {reason}"),
            Self::InvalidBinding(reason) => write!(f, "invalid sort binding: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported portable sort: {reason}"),
            Self::Overflow => f.write_str("portable sort geometry overflow"),
        }
    }
}

impl std::error::Error for PortableSortError {}

/// One fully checked dense paired-output sort. `SortValue` remains the sole
/// semantic taxonomy; this projection owns only portable launch geometry and
/// the exact ordered comparator network consumed by renderers.
#[derive(Clone, Debug)]
pub(crate) struct PortableSortPair<'a> {
    value: &'a SortValue,
    elements: usize,
    axis_len: usize,
    inner: usize,
    padded_len: usize,
    work_items: usize,
    steps: Vec<PortableSortStep>,
}

impl<'a> PortableSortPair<'a> {
    pub(crate) fn new(value: &'a SortValue) -> Result<Self, PortableSortError> {
        if value.input == value.values
            || value.input == value.indices
            || value.values == value.indices
            || value.input_shape.rank() == 0
            || value.axis >= value.input_shape.rank()
        {
            return Err(PortableSortError::InvalidPlan(
                "sort requires one dense input and two distinct outputs".into(),
            ));
        }
        if !matches!(
            value.dtype,
            DType::Bool | DType::I32 | DType::U32 | DType::F32
        ) {
            return Err(PortableSortError::Unsupported(
                "storage must be Bool/I32/U32/F32",
            ));
        }
        let elements = value
            .input_shape
            .numel()
            .map_err(|_| PortableSortError::Overflow)?;
        let axis_len = value.input_shape.dims()[value.axis];
        if axis_len > MAX_PORTABLE_SORT_AXIS || axis_len > i32::MAX as usize {
            return Err(PortableSortError::Unsupported(
                "axis exceeds the bounded portable comparator domain",
            ));
        }
        let inner = value.input_shape.dims()[value.axis + 1..]
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim))
            .ok_or(PortableSortError::Overflow)?;
        let row_width = axis_len
            .checked_mul(inner)
            .ok_or(PortableSortError::Overflow)?;
        let rows = if row_width == 0 {
            0
        } else {
            elements / row_width
        };
        if rows.checked_mul(row_width) != Some(elements) {
            return Err(PortableSortError::InvalidPlan(
                "sort geometry is not dense".into(),
            ));
        }
        let work_items = rows.checked_mul(inner).ok_or(PortableSortError::Overflow)?;
        if [elements, rows, axis_len, inner, work_items]
            .into_iter()
            .any(|extent| extent > u32::MAX as usize)
        {
            return Err(PortableSortError::Unsupported(
                "portable sort requires a 32-bit dense domain",
            ));
        }
        let padded_len = if axis_len <= 1 {
            axis_len
        } else {
            axis_len
                .checked_next_power_of_two()
                .ok_or(PortableSortError::Overflow)?
        };
        let steps = tinygrad_network(padded_len, value.descending)?;
        Ok(Self {
            value,
            elements,
            axis_len,
            inner,
            padded_len,
            work_items,
            steps,
        })
    }

    pub(crate) fn value(&self) -> &'a SortValue {
        self.value
    }

    pub(crate) fn elements(&self) -> usize {
        self.elements
    }

    pub(crate) fn axis_len(&self) -> usize {
        self.axis_len
    }

    pub(crate) fn inner(&self) -> usize {
        self.inner
    }

    pub(crate) fn padded_len(&self) -> usize {
        self.padded_len
    }

    pub(crate) fn launch_extent(&self) -> usize {
        self.work_items
    }

    pub(crate) fn steps(&self) -> &[PortableSortStep] {
        &self.steps
    }

    pub(crate) fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), PortableSortError> {
        let [binding] = bindings else {
            return Err(PortableSortError::InvalidBinding(
                "sort requires exactly one dense input".into(),
            ));
        };
        let bytes = self
            .elements
            .checked_mul(self.value.dtype.itemsize())
            .ok_or(PortableSortError::Overflow)?;
        if binding.abi_index != 0
            || binding.input_node != self.value.input
            || binding.desc.id != self.value.input.index() as u64
            || binding.desc.shape != self.value.input_shape
            || binding.desc.dtype != self.value.dtype
            || binding.desc.bytes != bytes
            || !binding.desc.read_only
            || binding.desc.view.is_some()
        {
            return Err(PortableSortError::InvalidBinding(
                "sort input is not its exact dense descriptor".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn execute(
        &self,
        input: &TensorData,
    ) -> Result<(TensorData, TensorData), PortableSortError> {
        if input.shape() != &self.value.input_shape || input.dtype() != self.value.dtype {
            return Err(PortableSortError::InvalidBinding(
                "sort input value descriptor mismatch".into(),
            ));
        }
        crate::backend::stable_sort_pair(input, self.value.axis, self.value.descending)
            .map_err(|error| PortableSortError::InvalidBinding(error.to_string()))
    }
}

fn tinygrad_network(
    padded_len: usize,
    descending: bool,
) -> Result<Vec<PortableSortStep>, PortableSortError> {
    if padded_len <= 1 {
        return Ok(Vec::new());
    }
    let stages = padded_len.trailing_zeros() as usize;
    let mut steps = Vec::new();
    for stage in 1..=stages {
        if stage != stages {
            append_tinygrad_flip(&mut steps, padded_len, stages, stage);
        }
        for substage in (0..stage).rev() {
            let partner_bit = stages - substage - 1;
            let mask = 1usize << (stages - partner_bit - 1);
            for left in 0..padded_len {
                if left & mask == 0 {
                    steps.push(PortableSortStep::Compare(PortableSortCompare {
                        left,
                        right: left | mask,
                        left_takes_larger: descending,
                    }));
                }
            }
        }
        if stage != stages {
            append_tinygrad_flip(&mut steps, padded_len, stages, stage);
        }
    }
    Ok(steps)
}

fn append_tinygrad_flip(
    steps: &mut Vec<PortableSortStep>,
    padded_len: usize,
    stages: usize,
    stage: usize,
) {
    let crossover_bit = stages - stage - 1;
    let crossover_mask = 1usize << (stages - crossover_bit - 1);
    let flip_start_bit = stages - (stage + 1);
    let flip_mask = (1usize << (stages - flip_start_bit)) - 1;
    let mut seen = vec![false; padded_len];
    for lane in 0..padded_len {
        if lane & crossover_mask == 0 || seen[lane] {
            continue;
        }
        let partner = lane ^ (flip_mask & !crossover_mask);
        seen[lane] = true;
        seen[partner] = true;
        if lane != partner {
            steps.push(PortableSortStep::Swap {
                left: lane,
                right: partner,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodeId, Scalar, Shape, Storage};

    fn ordered_pair(dtype: DType, left: Scalar, right: Scalar) -> (Scalar, Scalar) {
        let right_is_larger = match dtype {
            DType::Bool => right.as_bool() && !left.as_bool(),
            DType::I32 => right.as_i64() > left.as_i64(),
            DType::U32 => right.as_u64() > left.as_u64(),
            DType::F32 => right.as_f64() > left.as_f64(),
            _ => unreachable!(),
        };
        let right_is_smaller = match dtype {
            DType::Bool => !right.as_bool() && left.as_bool(),
            DType::I32 => right.as_i64() < left.as_i64(),
            DType::U32 => right.as_u64() < left.as_u64(),
            DType::F32 => right.as_f64() < left.as_f64(),
            _ => unreachable!(),
        };
        (
            if right_is_larger { right } else { left },
            if right_is_smaller { right } else { left },
        )
    }

    fn projected_values(plan: &PortableSortPair<'_>, input: &TensorData) -> TensorData {
        let mut output = vec![Scalar::I(0); plan.elements()];
        let padding = if plan.value().descending {
            plan.value().dtype.min()
        } else {
            plan.value().dtype.max()
        };
        for row in 0..(plan.launch_extent() / plan.inner()) {
            for inner in 0..plan.inner() {
                let mut work = vec![padding; plan.padded_len()];
                for (lane, slot) in work.iter_mut().enumerate().take(plan.axis_len()) {
                    *slot = input.scalar_at((row * plan.axis_len() + lane) * plan.inner() + inner);
                }
                for &step in plan.steps() {
                    match step {
                        PortableSortStep::Swap { left, right } => work.swap(left, right),
                        PortableSortStep::Compare(compare) => {
                            let (larger, smaller) = ordered_pair(
                                plan.value().dtype,
                                work[compare.left],
                                work[compare.right],
                            );
                            let (left, right) = if compare.left_takes_larger {
                                (larger, smaller)
                            } else {
                                (smaller, larger)
                            };
                            work[compare.left] = left;
                            work[compare.right] = right;
                        }
                    }
                }
                for lane in 0..plan.axis_len() {
                    output[(row * plan.axis_len() + lane) * plan.inner() + inner] = work[lane];
                }
            }
        }
        TensorData::from_scalars(plan.value().input_shape.clone(), plan.value().dtype, output)
            .unwrap()
    }

    #[test]
    fn checked_network_matches_tinygrad_oracle_bits_and_indices() {
        let value = SortValue {
            input: NodeId::from_index(0),
            input_shape: Shape::new([2, 3, 2]),
            axis: 1,
            descending: false,
            values: NodeId::from_index(1),
            indices: NodeId::from_index(2),
            dtype: DType::F32,
        };
        let input = TensorData::from_storage(
            value.input_shape.clone(),
            Storage::F32(vec![
                -0.0,
                3.0,
                0.0,
                f32::NAN,
                f32::NAN,
                -1.0,
                f32::INFINITY,
                2.0,
                f32::NEG_INFINITY,
                2.0,
                f32::from_bits(0x7fc0_1234),
                2.0,
            ]),
        )
        .unwrap();
        let plan = PortableSortPair::new(&value).unwrap();
        let projected = projected_values(&plan, &input).to_le_bytes().unwrap();
        let (oracle_values, oracle_indices) = plan.execute(&input).unwrap();
        assert_eq!(projected, oracle_values.to_le_bytes().unwrap());
        assert_eq!(oracle_indices.dtype(), DType::I32);
        assert_eq!(oracle_indices.shape(), &value.input_shape);
    }

    #[test]
    fn checked_projection_rejects_dtype_alias_and_accepts_empty_axis() {
        let value = SortValue {
            input: NodeId::from_index(4),
            input_shape: Shape::new([3]),
            axis: 0,
            descending: true,
            values: NodeId::from_index(5),
            indices: NodeId::from_index(6),
            dtype: DType::U32,
        };
        let plan = PortableSortPair::new(&value).unwrap();
        assert_eq!(plan.launch_extent(), 1);
        let mut alias = value.clone();
        alias.indices = alias.values;
        assert!(matches!(
            PortableSortPair::new(&alias),
            Err(PortableSortError::InvalidPlan(_))
        ));
        let mut unsupported = value.clone();
        unsupported.dtype = DType::F64;
        assert!(matches!(
            PortableSortPair::new(&unsupported),
            Err(PortableSortError::Unsupported(_))
        ));
        let mut empty_axis = value;
        empty_axis.input_shape = Shape::new([2, 0, 3]);
        empty_axis.axis = 1;
        let empty = PortableSortPair::new(&empty_axis).unwrap();
        assert_eq!((empty.elements(), empty.launch_extent()), (0, 0));
    }
}
