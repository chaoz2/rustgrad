//! Immutable contracts and pure execution for materializing movement kernels.
use crate::{
    AffineView, DType, Graph, NodeId, Op, Scalar, Shape, Storage, TensorData, index::DenseIndex,
};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MovementOperand {
    pub node: NodeId,
    pub shape: Shape,
    pub dtype: DType,
}

/// One injective static coordinate placement shared by a movement adjoint and
/// its reverse read. Construction proves the complete Cartesian map in O(rank)
/// from per-axis endpoints; executors never rediscover bounds policy.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct StaticPositionMap {
    pub(crate) input_shape: Shape,
    pub(crate) output_shape: Shape,
    pub(crate) starts: Vec<i64>,
    pub(crate) steps: Vec<i64>,
}

impl StaticPositionMap {
    pub(crate) fn new(
        input_shape: Shape,
        output_shape: Shape,
        starts: &[isize],
        steps: &[isize],
    ) -> Result<Self, MovementPlanError> {
        let starts = starts
            .iter()
            .map(|value| i64::try_from(*value).map_err(|_| MovementPlanError::Overflow))
            .collect::<Result<Vec<_>, _>>()?;
        let steps = steps
            .iter()
            .map(|value| i64::try_from(*value).map_err(|_| MovementPlanError::Overflow))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_i64(input_shape, output_shape, starts, steps)
    }

    pub(crate) fn from_i64(
        input_shape: Shape,
        output_shape: Shape,
        starts: Vec<i64>,
        steps: Vec<i64>,
    ) -> Result<Self, MovementPlanError> {
        let map = Self {
            input_shape,
            output_shape,
            starts,
            steps,
        };
        map.validate()?;
        Ok(map)
    }

    pub(crate) fn validate(&self) -> Result<(), MovementPlanError> {
        let rank = self.input_shape.rank();
        if self.output_shape.rank() != rank || self.starts.len() != rank || self.steps.len() != rank
        {
            return Err(MovementPlanError::InvalidGeometry);
        }
        self.input_shape
            .numel()
            .and_then(|_| self.output_shape.numel())
            .map_err(|_| MovementPlanError::Overflow)?;
        for axis in 0..rank {
            let input = self.input_shape.dims()[axis];
            let step = i128::from(self.steps[axis]);
            if step == 0 {
                return Err(MovementPlanError::InvalidGeometry);
            }
            if input == 0 {
                continue;
            }
            let output = i128::try_from(self.output_shape.dims()[axis])
                .map_err(|_| MovementPlanError::Overflow)?;
            let start = i128::from(self.starts[axis]);
            let final_coordinate = start
                .checked_add(
                    i128::try_from(input - 1)
                        .map_err(|_| MovementPlanError::Overflow)?
                        .checked_mul(step)
                        .ok_or(MovementPlanError::Overflow)?,
                )
                .ok_or(MovementPlanError::Overflow)?;
            if start.min(final_coordinate) < 0 || start.max(final_coordinate) >= output {
                return Err(MovementPlanError::InvalidGeometry);
            }
        }
        Ok(())
    }

    pub(crate) fn read_view(&self) -> Result<AffineView, MovementPlanError> {
        self.validate()?;
        if self
            .input_shape
            .numel()
            .map_err(|_| MovementPlanError::Overflow)?
            == 0
        {
            let view = AffineView {
                source_shape: self.output_shape.clone(),
                logical_shape: self.input_shape.clone(),
                offset: 0,
                strides: vec![0; self.input_shape.rank()],
            };
            view.validate_read()
                .map_err(|_| MovementPlanError::InvalidGeometry)?;
            return Ok(view);
        }
        let mut dense_stride = 1i128;
        let mut strides = vec![0i64; self.input_shape.rank()];
        let mut offset = 0i128;
        for axis in (0..self.input_shape.rank()).rev() {
            offset = offset
                .checked_add(
                    i128::from(self.starts[axis])
                        .checked_mul(dense_stride)
                        .ok_or(MovementPlanError::Overflow)?,
                )
                .ok_or(MovementPlanError::Overflow)?;
            // A singleton axis never contributes to the address. Canonicalize
            // it before multiplying so an otherwise irrelevant i64::MIN step
            // cannot manufacture an overflow in the reverse read projection.
            strides[axis] = if self.input_shape.dims()[axis] <= 1 {
                0
            } else {
                i64::try_from(
                    i128::from(self.steps[axis])
                        .checked_mul(dense_stride)
                        .ok_or(MovementPlanError::Overflow)?,
                )
                .map_err(|_| MovementPlanError::Overflow)?
            };
            dense_stride = dense_stride
                .checked_mul(
                    i128::try_from(self.output_shape.dims()[axis])
                        .map_err(|_| MovementPlanError::Overflow)?,
                )
                .ok_or(MovementPlanError::Overflow)?;
        }
        let view = AffineView {
            source_shape: self.output_shape.clone(),
            logical_shape: self.input_shape.clone(),
            offset: i64::try_from(offset).map_err(|_| MovementPlanError::Overflow)?,
            strides,
        };
        view.validate_read()
            .map_err(|_| MovementPlanError::InvalidGeometry)?;
        Ok(view)
    }

    fn destination_offsets(&self) -> Result<Vec<usize>, MovementExecutionError> {
        self.validate()?;
        let input = dense(&self.input_shape)?;
        let output = dense(&self.output_shape)?;
        let mut offsets = Vec::with_capacity(input.len());
        for linear in 0..input.len() {
            let coords = input
                .coords(linear)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            let destination = coords
                .iter()
                .enumerate()
                .map(|(axis, coordinate)| {
                    i128::from(self.starts[axis])
                        .checked_add(
                            i128::try_from(*coordinate)
                                .map_err(|_| MovementExecutionError::Overflow)?
                                .checked_mul(i128::from(self.steps[axis]))
                                .ok_or(MovementExecutionError::Overflow)?,
                        )
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or(MovementExecutionError::InvalidGeometry)
                })
                .collect::<Result<Vec<_>, _>>()?;
            offsets.push(
                output
                    .offset(&destination)
                    .map_err(|_| MovementExecutionError::InvalidGeometry)?,
            );
        }
        Ok(offsets)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MovementKernelKind {
    /// A source or pure computed producer viewed through a checked static
    /// affine read map and copied into owned dense storage.
    AffineCopy {
        input: MovementOperand,
        view: AffineView,
    },
    /// Concrete constant padding. `fill_bits` is already converted to the
    /// output storage dtype, so all executors share one raw-payload contract.
    Pad {
        input: MovementOperand,
        padding: Vec<(usize, usize)>,
        fill_bits: u64,
    },
    Concat {
        inputs: Vec<MovementOperand>,
        axis: usize,
    },
    Gather {
        input: MovementOperand,
        index: MovementOperand,
        axis: usize,
    },
    Scatter {
        base: MovementOperand,
        index: MovementOperand,
        updates: MovementOperand,
        axis: usize,
        add: bool,
    },
    /// Dense raw-byte reinterpretation. Input and output may use different
    /// element widths, but their total byte extents are identical.
    Bitcast { input: MovementOperand },
    /// Explicit dense owned copy with an unchanged descriptor.
    Contiguous { input: MovementOperand },
    /// Zero-initialize owned dense storage, then place every input coordinate
    /// through one checked injective static map using raw storage bytes.
    /// Appending this variant preserves the derived-hash discriminants—and
    /// therefore released cache keys—of every historical movement kind.
    ScatterPositions {
        input: MovementOperand,
        starts: Vec<i64>,
        steps: Vec<i64>,
    },
}

/// Fully validated materializing movement geometry and ordered pointer ABI.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MovementKernelPlan {
    pub kind: MovementKernelKind,
    pub output: NodeId,
    pub output_shape: Shape,
    pub dtype: DType,
    pub cache_key: u64,
}

/// Borrowed, fully checked projection of the existing raw movement-copy plans.
/// `view == None` is the dense Contiguous identity map; an AffineCopy retains
/// its canonical proven read map. Renderers keep only syntax and capability
/// policy locally.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RawCopyView<'a> {
    plan: &'a MovementKernelPlan,
    input: &'a MovementOperand,
    view: Option<&'a AffineView>,
    input_elements: usize,
    input_bytes: usize,
    elements: usize,
    bytes: usize,
}

/// One active term in the checked row-major output-to-source address map.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawCopyAxis {
    pub(crate) output_axis: usize,
    pub(crate) dimension: usize,
    pub(crate) divisor: usize,
    pub(crate) stride: usize,
    pub(crate) reversed: bool,
}

/// Renderer-neutral unsigned address expression for one raw affine copy.
/// Backends only spell these already-proved terms in their source language.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RawCopyAddress {
    pub(crate) offset: usize,
    pub(crate) axes: Vec<RawCopyAxis>,
}

impl<'a> RawCopyView<'a> {
    pub(crate) fn plan(self) -> &'a MovementKernelPlan {
        self.plan
    }

    pub(crate) fn input(self) -> &'a MovementOperand {
        self.input
    }

    pub(crate) fn address(self) -> Result<Option<RawCopyAddress>, MovementPlanError> {
        let Some(view) = self.view else {
            return Ok(None);
        };
        let normalized = view
            .normalized_read()
            .map_err(|_| MovementPlanError::InvalidGeometry)?;
        let mut axes = Vec::new();
        for (output_axis, (&dimension, normalized_axis)) in self
            .plan
            .output_shape
            .dims()
            .iter()
            .zip(&normalized.axes)
            .enumerate()
        {
            if self.elements == 0 || dimension <= 1 || normalized_axis.stride == 0 {
                continue;
            }
            let divisor = self.plan.output_shape.dims()[output_axis + 1..]
                .iter()
                .try_fold(1usize, |product, next| product.checked_mul(*next))
                .ok_or(MovementPlanError::Overflow)?;
            axes.push(RawCopyAxis {
                output_axis,
                dimension,
                divisor,
                stride: normalized_axis.stride,
                reversed: normalized_axis.reversed,
            });
        }
        Ok(Some(RawCopyAddress {
            offset: normalized.offset,
            axes,
        }))
    }

    pub(crate) fn input_elements(self) -> usize {
        self.input_elements
    }

    pub(crate) fn input_bytes(self) -> usize {
        self.input_bytes
    }

    pub(crate) fn elements(self) -> usize {
        self.elements
    }

    pub(crate) fn bytes(self) -> usize {
        self.bytes
    }

    pub(crate) fn width(self) -> usize {
        self.plan.dtype.itemsize()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MovementPlanError {
    NotMovement,
    InvalidGeometry,
    UnsupportedDType,
    Overflow,
}

impl fmt::Display for MovementPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "movement plan error: {self:?}")
    }
}
impl std::error::Error for MovementPlanError {}

/// A fail-closed error from graph-independent movement execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MovementExecutionError {
    InvalidPlan(MovementPlanError),
    OperandCount {
        expected: usize,
        actual: usize,
    },
    OperandDescriptor {
        position: usize,
        expected_shape: Shape,
        actual_shape: Shape,
        expected_dtype: DType,
        actual_dtype: DType,
    },
    IndexOutOfBounds {
        axis: usize,
        index: i64,
        dim: usize,
    },
    Overflow,
    InvalidGeometry,
}

impl fmt::Display for MovementExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "movement execution error: {self:?}")
    }
}
impl std::error::Error for MovementExecutionError {}

impl From<MovementPlanError> for MovementExecutionError {
    fn from(value: MovementPlanError) -> Self {
        Self::InvalidPlan(value)
    }
}

impl MovementOperand {
    fn from_graph(graph: &Graph, node: NodeId) -> Result<Self, MovementPlanError> {
        Ok(Self {
            node,
            shape: graph
                .shape(node)
                .map_err(|_| MovementPlanError::InvalidGeometry)?
                .clone(),
            dtype: graph
                .dtype(node)
                .map_err(|_| MovementPlanError::UnsupportedDType)?,
        })
    }
}

impl MovementKernelPlan {
    /// Returns the renderer-facing raw copy projection after revalidating the
    /// canonical plan and both checked byte extents. Other movement kinds stay
    /// distinct and fail closed at each renderer boundary.
    pub(crate) fn raw_copy(&self) -> Result<Option<RawCopyView<'_>>, MovementPlanError> {
        let (input, view) = match &self.kind {
            MovementKernelKind::AffineCopy { input, view } => (input, Some(view)),
            MovementKernelKind::Contiguous { input } => (input, None),
            _ => return Ok(None),
        };
        self.validate()?;
        let input_elements = input
            .shape
            .numel()
            .map_err(|_| MovementPlanError::Overflow)?;
        let input_bytes = input_elements
            .checked_mul(input.dtype.itemsize())
            .ok_or(MovementPlanError::Overflow)?;
        let elements = self
            .output_shape
            .numel()
            .map_err(|_| MovementPlanError::Overflow)?;
        let bytes = elements
            .checked_mul(self.dtype.itemsize())
            .ok_or(MovementPlanError::Overflow)?;
        Ok(Some(RawCopyView {
            plan: self,
            input,
            view,
            input_elements,
            input_bytes,
            elements,
            bytes,
        }))
    }

    pub fn from_graph(graph: &Graph, output: NodeId) -> Result<Self, MovementPlanError> {
        let kind = match graph
            .op(output)
            .map_err(|_| MovementPlanError::InvalidGeometry)?
        {
            Op::Pad {
                input,
                padding,
                fill,
            } => {
                let input = MovementOperand::from_graph(graph, *input)?;
                MovementKernelKind::Pad {
                    fill_bits: scalar_bits(input.dtype, *fill),
                    input,
                    padding: padding.clone(),
                }
            }
            Op::Concat { inputs, axis } => MovementKernelKind::Concat {
                inputs: inputs
                    .iter()
                    .map(|node| MovementOperand::from_graph(graph, *node))
                    .collect::<Result<Vec<_>, _>>()?,
                axis: *axis,
            },
            Op::Gather { input, index, axis } => MovementKernelKind::Gather {
                input: MovementOperand::from_graph(graph, *input)?,
                index: MovementOperand::from_graph(graph, *index)?,
                axis: *axis,
            },
            Op::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => MovementKernelKind::Scatter {
                base: MovementOperand::from_graph(graph, *base)?,
                index: MovementOperand::from_graph(graph, *index)?,
                updates: MovementOperand::from_graph(graph, *updates)?,
                axis: *axis,
                add: *add,
            },
            Op::ScatterPositions {
                input,
                shape,
                starts,
                steps,
            } => {
                let input = MovementOperand::from_graph(graph, *input)?;
                let map =
                    StaticPositionMap::new(input.shape.clone(), shape.clone(), starts, steps)?;
                MovementKernelKind::ScatterPositions {
                    input,
                    starts: map.starts,
                    steps: map.steps,
                }
            }
            Op::ScatterPositionsVjp {
                cotangent,
                input_shape,
                starts,
                steps,
            } => {
                let input = MovementOperand::from_graph(graph, *cotangent)?;
                let map = StaticPositionMap::new(
                    input_shape.clone(),
                    input.shape.clone(),
                    starts,
                    steps,
                )?;
                MovementKernelKind::AffineCopy {
                    input,
                    view: map.read_view()?,
                }
            }
            Op::Bitcast { input, .. } => MovementKernelKind::Bitcast {
                input: MovementOperand::from_graph(graph, *input)?,
            },
            Op::Contiguous { input } => MovementKernelKind::Contiguous {
                input: MovementOperand::from_graph(graph, *input)?,
            },
            _ => return Err(MovementPlanError::NotMovement),
        };
        let mut plan = Self {
            kind,
            output,
            output_shape: graph
                .shape(output)
                .map_err(|_| MovementPlanError::InvalidGeometry)?
                .clone(),
            dtype: graph
                .dtype(output)
                .map_err(|_| MovementPlanError::UnsupportedDType)?,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        plan.validate()?;
        Ok(plan)
    }

    /// Selects the executable movement plan used by the schedule compiler.
    ///
    /// A contiguous boundary over an affine view writes that view directly
    /// into the boundary's fresh dense output. Other contiguous inputs retain
    /// the explicit dense-copy plan, including ordinary computed producers.
    pub(crate) fn from_scheduled_graph(
        graph: &Graph,
        output: NodeId,
    ) -> Result<Self, MovementPlanError> {
        match Self::from_contiguous_affine_view(graph, output) {
            Err(MovementPlanError::NotMovement) => Self::from_graph(graph, output),
            result => result,
        }
    }

    fn from_contiguous_affine_view(
        graph: &Graph,
        output: NodeId,
    ) -> Result<Self, MovementPlanError> {
        let input = match graph
            .op(output)
            .map_err(|_| MovementPlanError::InvalidGeometry)?
        {
            Op::Contiguous { input }
                if matches!(
                    graph.op(*input),
                    Ok(Op::Shrink { .. }
                        | Op::Reshape { .. }
                        | Op::Permute { .. }
                        | Op::Expand { .. }
                        | Op::Stride { .. })
                ) =>
            {
                *input
            }
            Op::Contiguous { .. } => return Err(MovementPlanError::NotMovement),
            _ => return Err(MovementPlanError::NotMovement),
        };
        let rangeified = match crate::rangeify::static_view(graph, input) {
            Ok(view) => view,
            Err(crate::rangeify::RangeifyError::Invalid) => {
                return Err(MovementPlanError::InvalidGeometry);
            }
            Err(crate::rangeify::RangeifyError::Unsupported(_)) => {
                match crate::rangeify::computed_view(graph, input) {
                    Ok(view) => view,
                    Err(crate::rangeify::RangeifyError::Invalid) => {
                        return Err(MovementPlanError::InvalidGeometry);
                    }
                    Err(crate::rangeify::RangeifyError::Unsupported(_)) => {
                        return Err(MovementPlanError::NotMovement);
                    }
                }
            }
        };
        let input = MovementOperand::from_graph(graph, rangeified.source)?;
        let output_shape = graph
            .shape(output)
            .map_err(|_| MovementPlanError::InvalidGeometry)?
            .clone();
        let dtype = graph
            .dtype(output)
            .map_err(|_| MovementPlanError::UnsupportedDType)?;
        let mut plan = Self {
            kind: MovementKernelKind::AffineCopy {
                input,
                view: rangeified.view,
            },
            output,
            output_shape,
            dtype,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        plan.validate()?;
        Ok(plan)
    }

    /// Materializes the narrow static computed-view boundary used when a
    /// reduction or contraction requires a dense owned operand. Source-backed
    /// views remain load addressing and never acquire this copy plan.
    pub(crate) fn from_computed_affine_view(
        graph: &Graph,
        output: NodeId,
    ) -> Result<Self, MovementPlanError> {
        let rangeified = crate::rangeify::computed_view(graph, output)
            .map_err(|_| MovementPlanError::InvalidGeometry)?;
        let input = MovementOperand::from_graph(graph, rangeified.source)?;
        let output_shape = graph
            .shape(output)
            .map_err(|_| MovementPlanError::InvalidGeometry)?
            .clone();
        let dtype = graph
            .dtype(output)
            .map_err(|_| MovementPlanError::UnsupportedDType)?;
        let mut plan = Self {
            kind: MovementKernelKind::AffineCopy {
                input,
                view: rangeified.view,
            },
            output,
            output_shape,
            dtype,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), MovementPlanError> {
        self.output_shape
            .numel()
            .map_err(|_| MovementPlanError::Overflow)?
            .checked_mul(self.dtype.itemsize())
            .ok_or(MovementPlanError::Overflow)?;
        if self.cache_key != self.expected_cache_key() {
            return Err(MovementPlanError::InvalidGeometry);
        }
        match &self.kind {
            MovementKernelKind::AffineCopy { input, view } => {
                input
                    .shape
                    .numel()
                    .map_err(|_| MovementPlanError::Overflow)?
                    .checked_mul(input.dtype.itemsize())
                    .ok_or(MovementPlanError::Overflow)?;
                if input.dtype != self.dtype
                    || view.source_shape != input.shape
                    || view.logical_shape != self.output_shape
                    || view.validate_read().is_err()
                {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Pad {
                input,
                padding,
                fill_bits,
            } => {
                input
                    .shape
                    .numel()
                    .map_err(|_| MovementPlanError::Overflow)?;
                if padding.len() != input.shape.rank() || self.dtype != input.dtype {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                let expected = input
                    .shape
                    .dims()
                    .iter()
                    .zip(padding)
                    .map(|(dim, (before, after))| {
                        dim.checked_add(*before)
                            .and_then(|x| x.checked_add(*after))
                            .ok_or(MovementPlanError::Overflow)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if self.output_shape.dims() != expected || !valid_bits(self.dtype, *fill_bits) {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Concat { inputs, axis } => {
                let first = inputs.first().ok_or(MovementPlanError::InvalidGeometry)?;
                if *axis >= first.shape.rank() || self.output_shape.rank() != first.shape.rank() {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                let mut axis_total = 0usize;
                let mut dtype = first.dtype;
                for input in inputs {
                    input
                        .shape
                        .numel()
                        .map_err(|_| MovementPlanError::Overflow)?;
                    if input.shape.rank() != first.shape.rank()
                        || input
                            .shape
                            .dims()
                            .iter()
                            .zip(first.shape.dims())
                            .enumerate()
                            .any(|(dim, (actual, expected))| dim != *axis && actual != expected)
                    {
                        return Err(MovementPlanError::InvalidGeometry);
                    }
                    axis_total = axis_total
                        .checked_add(input.shape.dims()[*axis])
                        .ok_or(MovementPlanError::Overflow)?;
                    dtype = dtype.promote(input.dtype);
                }
                let mut expected = first.shape.dims().to_vec();
                expected[*axis] = axis_total;
                if self.output_shape.dims() != expected || self.dtype != dtype {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Gather { input, index, axis } => {
                validate_index_geometry(input, index, *axis)?;
                if self.dtype != input.dtype || self.output_shape != index.shape {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => {
                validate_index_geometry(base, index, *axis)?;
                if self.output_shape != base.shape
                    || self.dtype != base.dtype.promote(updates.dtype)
                    || updates.shape.rank() != index.shape.rank()
                    || updates
                        .shape
                        .dims()
                        .iter()
                        .zip(index.shape.dims())
                        .any(|(update, index)| update < index)
                {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                if *add && !matches!(self.dtype, DType::F32 | DType::F64) {
                    return Err(MovementPlanError::UnsupportedDType);
                }
            }
            MovementKernelKind::ScatterPositions {
                input,
                starts,
                steps,
            } => {
                input
                    .shape
                    .numel()
                    .map_err(|_| MovementPlanError::Overflow)?
                    .checked_mul(input.dtype.itemsize())
                    .ok_or(MovementPlanError::Overflow)?;
                if input.dtype != self.dtype {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                StaticPositionMap::from_i64(
                    input.shape.clone(),
                    self.output_shape.clone(),
                    starts.clone(),
                    steps.clone(),
                )?;
            }
            MovementKernelKind::Bitcast { input } => {
                let input_itemsize = input.dtype.itemsize();
                let output_itemsize = self.dtype.itemsize();
                let mut expected_dims = input.shape.dims().to_vec();
                if input_itemsize != output_itemsize {
                    let last = expected_dims
                        .last_mut()
                        .ok_or(MovementPlanError::InvalidGeometry)?;
                    let final_bytes = last
                        .checked_mul(input_itemsize)
                        .ok_or(MovementPlanError::Overflow)?;
                    if final_bytes % output_itemsize != 0 {
                        return Err(MovementPlanError::InvalidGeometry);
                    }
                    *last = final_bytes / output_itemsize;
                }
                let input_bytes = input
                    .shape
                    .numel()
                    .map_err(|_| MovementPlanError::Overflow)?
                    .checked_mul(input_itemsize)
                    .ok_or(MovementPlanError::Overflow)?;
                let output_bytes = self
                    .output_shape
                    .numel()
                    .map_err(|_| MovementPlanError::Overflow)?
                    .checked_mul(output_itemsize)
                    .ok_or(MovementPlanError::Overflow)?;
                if input_bytes != output_bytes
                    || input.dtype == self.dtype
                    || self.output_shape.dims() != expected_dims
                {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Contiguous { input } => {
                if input.shape != self.output_shape || input.dtype != self.dtype {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                input
                    .shape
                    .numel()
                    .map_err(|_| MovementPlanError::Overflow)?
                    .checked_mul(input.dtype.itemsize())
                    .ok_or(MovementPlanError::Overflow)?;
            }
        }
        if self
            .input_operands()
            .iter()
            .any(|operand| operand.node == self.output)
        {
            return Err(MovementPlanError::InvalidGeometry);
        }
        Ok(())
    }

    /// Rebinds the shape-bearing descriptors in a captured movement plan.
    ///
    /// This deliberately retains the plan's operation-specific metadata and
    /// routes the rebuilt value through the ordinary validator and cache-key
    /// derivation. Affine copies require a separately specialized view and
    /// differing-width bitcasts require a symbolic byte/divisibility proof;
    /// neither can be reconstructed from buffer shapes alone.
    pub(crate) fn specialize_shapes(
        &self,
        operand_shapes: &std::collections::BTreeMap<NodeId, Shape>,
        output_shape: Shape,
    ) -> Result<Self, MovementPlanError> {
        let operand = |value: &MovementOperand| {
            Ok(MovementOperand {
                node: value.node,
                shape: operand_shapes
                    .get(&value.node)
                    .cloned()
                    .ok_or(MovementPlanError::InvalidGeometry)?,
                dtype: value.dtype,
            })
        };
        let kind = match &self.kind {
            MovementKernelKind::AffineCopy { .. } => {
                return Err(MovementPlanError::InvalidGeometry);
            }
            MovementKernelKind::Pad {
                input,
                padding,
                fill_bits,
            } => MovementKernelKind::Pad {
                input: operand(input)?,
                padding: padding.clone(),
                fill_bits: *fill_bits,
            },
            MovementKernelKind::Concat { inputs, axis } => MovementKernelKind::Concat {
                inputs: inputs.iter().map(operand).collect::<Result<Vec<_>, _>>()?,
                axis: *axis,
            },
            MovementKernelKind::Gather { input, index, axis } => MovementKernelKind::Gather {
                input: operand(input)?,
                index: operand(index)?,
                axis: *axis,
            },
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => MovementKernelKind::Scatter {
                base: operand(base)?,
                index: operand(index)?,
                updates: operand(updates)?,
                axis: *axis,
                add: *add,
            },
            MovementKernelKind::ScatterPositions { .. } => {
                return Err(MovementPlanError::InvalidGeometry);
            }
            MovementKernelKind::Bitcast { input } => {
                if input.dtype.itemsize() != self.dtype.itemsize() {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                MovementKernelKind::Bitcast {
                    input: operand(input)?,
                }
            }
            MovementKernelKind::Contiguous { input } => MovementKernelKind::Contiguous {
                input: operand(input)?,
            },
        };
        let mut specialized = Self {
            kind,
            output: self.output,
            output_shape,
            dtype: self.dtype,
            cache_key: 0,
        };
        specialized.cache_key = specialized.expected_cache_key();
        specialized.validate()?;
        Ok(specialized)
    }

    pub(crate) fn specialize_affine_view(
        &self,
        input_shape: Shape,
        output_shape: Shape,
        view: AffineView,
    ) -> Result<Self, MovementPlanError> {
        let MovementKernelKind::AffineCopy { input, .. } = &self.kind else {
            return Err(MovementPlanError::InvalidGeometry);
        };
        let mut specialized = Self {
            kind: MovementKernelKind::AffineCopy {
                input: MovementOperand {
                    node: input.node,
                    shape: input_shape,
                    dtype: input.dtype,
                },
                view,
            },
            output: self.output,
            output_shape,
            dtype: self.dtype,
            cache_key: 0,
        };
        specialized.cache_key = specialized.expected_cache_key();
        specialized.validate()?;
        Ok(specialized)
    }

    pub fn input_operands(&self) -> Vec<&MovementOperand> {
        match &self.kind {
            MovementKernelKind::AffineCopy { input, .. } => vec![input],
            MovementKernelKind::Pad { input, .. } => vec![input],
            MovementKernelKind::Concat { inputs, .. } => inputs.iter().collect(),
            MovementKernelKind::Gather { input, index, .. } => vec![input, index],
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                ..
            } => vec![base, index, updates],
            MovementKernelKind::ScatterPositions { input, .. } => vec![input],
            MovementKernelKind::Bitcast { input } => vec![input],
            MovementKernelKind::Contiguous { input } => vec![input],
        }
    }

    /// Executes this validated plan against operands in [`Self::input_operands`]
    /// order. All descriptors and every data-dependent index are checked before
    /// output storage is allocated or cloned.
    pub fn execute(&self, operands: &[TensorData]) -> Result<TensorData, MovementExecutionError> {
        self.validate()?;
        let expected = self.input_operands();
        if operands.len() != expected.len() {
            return Err(MovementExecutionError::OperandCount {
                expected: expected.len(),
                actual: operands.len(),
            });
        }
        for (position, (value, desc)) in operands.iter().zip(expected).enumerate() {
            if value.shape() != &desc.shape || value.dtype() != desc.dtype {
                return Err(MovementExecutionError::OperandDescriptor {
                    position,
                    expected_shape: desc.shape.clone(),
                    actual_shape: value.shape().clone(),
                    expected_dtype: desc.dtype,
                    actual_dtype: value.dtype(),
                });
            }
        }
        match &self.kind {
            MovementKernelKind::AffineCopy { view, .. } => {
                let len = self
                    .output_shape
                    .numel()
                    .map_err(|_| MovementExecutionError::Overflow)?;
                let offsets = (0..len)
                    .map(|linear| {
                        view.element_offset(linear)
                            .map_err(|_| MovementExecutionError::InvalidGeometry)
                            .and_then(|offset| {
                                usize::try_from(offset)
                                    .map_err(|_| MovementExecutionError::InvalidGeometry)
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                TensorData::from_storage(
                    self.output_shape.clone(),
                    select_raw(operands[0].storage(), &offsets),
                )
                .map_err(|_| MovementExecutionError::InvalidGeometry)
            }
            MovementKernelKind::Pad {
                input,
                padding,
                fill_bits,
            } => self.execute_pad(&operands[0], input, padding, *fill_bits),
            MovementKernelKind::Concat { inputs, axis } => {
                self.execute_concat(operands, inputs, *axis)
            }
            MovementKernelKind::Gather { input, index, axis } => {
                self.execute_gather(&operands[0], &operands[1], input, index, *axis)
            }
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => self.execute_scatter(
                &operands[0],
                &operands[1],
                &operands[2],
                base,
                index,
                updates,
                *axis,
                *add,
            ),
            MovementKernelKind::ScatterPositions {
                input,
                starts,
                steps,
            } => {
                let map = StaticPositionMap::from_i64(
                    input.shape.clone(),
                    self.output_shape.clone(),
                    starts.clone(),
                    steps.clone(),
                )?;
                let bytes = self
                    .output_shape
                    .numel()
                    .map_err(|_| MovementExecutionError::Overflow)?
                    .checked_mul(self.dtype.itemsize())
                    .ok_or(MovementExecutionError::Overflow)?;
                let zero = TensorData::from_le_bytes(
                    self.output_shape.clone(),
                    self.dtype,
                    &vec![0; bytes],
                )
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
                let destinations = map.destination_offsets()?;
                let placement = destinations
                    .into_iter()
                    .enumerate()
                    .map(|(source, destination)| (destination, source))
                    .collect::<Vec<_>>();
                let storage = scatter_raw(zero.storage(), operands[0].storage(), &placement)?;
                TensorData::from_storage(self.output_shape.clone(), storage)
                    .map_err(|_| MovementExecutionError::InvalidGeometry)
            }
            MovementKernelKind::Bitcast { .. } => operands[0]
                .bitcast_with_shape(self.output_shape.clone(), self.dtype)
                .map_err(|_| MovementExecutionError::InvalidGeometry),
            MovementKernelKind::Contiguous { .. } => Ok(operands[0].clone()),
        }
    }

    fn execute_pad(
        &self,
        input: &TensorData,
        desc: &MovementOperand,
        padding: &[(usize, usize)],
        fill_bits: u64,
    ) -> Result<TensorData, MovementExecutionError> {
        let source = dense(&desc.shape)?;
        let output = dense(&self.output_shape)?;
        let mut offsets = Vec::with_capacity(output.len());
        for linear in 0..output.len() {
            let coords = output
                .coords(linear)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            let inside =
                coords.iter().zip(padding).zip(desc.shape.dims()).all(
                    |((coord, (before, _)), dim)| *coord >= *before && *coord - *before < *dim,
                );
            offsets.push(if inside {
                let input_coords = coords
                    .iter()
                    .zip(padding)
                    .map(|(coord, (before, _))| coord - before)
                    .collect::<Vec<_>>();
                Some(
                    source
                        .offset(&input_coords)
                        .map_err(|_| MovementExecutionError::InvalidGeometry)?,
                )
            } else {
                None
            });
        }
        let fill = scalar_data_from_bits(self.dtype, fill_bits)?;
        input
            .pad_raw_offsets(self.output_shape.clone(), &offsets, &fill)
            .map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    fn execute_concat(
        &self,
        operands: &[TensorData],
        inputs: &[MovementOperand],
        axis: usize,
    ) -> Result<TensorData, MovementExecutionError> {
        let output = dense(&self.output_shape)?;
        let source_indices = inputs
            .iter()
            .map(|input| dense(&input.shape))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ends = Vec::with_capacity(inputs.len());
        let mut total = 0usize;
        for input in inputs {
            total = total
                .checked_add(input.shape.dims()[axis])
                .ok_or(MovementExecutionError::Overflow)?;
            ends.push(total);
        }
        let mut map = Vec::with_capacity(output.len());
        for linear in 0..output.len() {
            let mut coords = output
                .coords(linear)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            let operand = ends
                .iter()
                .position(|end| coords[axis] < *end)
                .ok_or(MovementExecutionError::InvalidGeometry)?;
            let prior = operand.checked_sub(1).map_or(0, |position| ends[position]);
            coords[axis] -= prior;
            let offset = source_indices[operand]
                .offset(&coords)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            map.push((operand, offset));
        }
        let storage = if operands.iter().all(|value| value.dtype() == self.dtype) {
            select_many_raw(self.dtype, operands, &map)?
        } else {
            Storage::from_scalars(
                self.dtype,
                map.iter()
                    .map(|(operand, offset)| operands[*operand].scalar_at(*offset)),
            )
        };
        TensorData::from_storage(self.output_shape.clone(), storage)
            .map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    fn execute_gather(
        &self,
        input: &TensorData,
        index: &TensorData,
        input_desc: &MovementOperand,
        index_desc: &MovementOperand,
        axis: usize,
    ) -> Result<TensorData, MovementExecutionError> {
        let map = indexed_map(input_desc, index_desc, index, axis)?;
        let offsets = map
            .iter()
            .map(|(destination, _)| *destination)
            .collect::<Vec<_>>();
        let storage = select_raw(input.storage(), &offsets);
        TensorData::from_storage(self.output_shape.clone(), storage)
            .map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_scatter(
        &self,
        base_value: &TensorData,
        index_value: &TensorData,
        updates_value: &TensorData,
        base: &MovementOperand,
        index: &MovementOperand,
        updates: &MovementOperand,
        axis: usize,
        add: bool,
    ) -> Result<TensorData, MovementExecutionError> {
        let destinations = indexed_map(base, index, index_value, axis)?;
        let update_index = dense(&updates.shape)?;
        let index_index = dense(&index.shape)?;
        let mut map = Vec::with_capacity(destinations.len());
        for (destination, linear) in destinations {
            let coords = index_index
                .coords(linear)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            let source = update_index
                .offset(&coords)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            map.push((destination, source));
        }
        let storage =
            if !add && base_value.dtype() == self.dtype && updates_value.dtype() == self.dtype {
                scatter_raw(base_value.storage(), updates_value.storage(), &map)?
            } else {
                let mut values = (0..base_value.len())
                    .map(|position| base_value.scalar_at(position))
                    .collect::<Vec<_>>();
                for (destination, source) in map {
                    let update = updates_value.scalar_at(source);
                    values[destination] = if add {
                        Scalar::F(values[destination].as_f64() + update.as_f64())
                    } else {
                        update
                    };
                }
                Storage::from_scalars(self.dtype, values)
            };
        TensorData::from_storage(self.output_shape.clone(), storage)
            .map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    pub(crate) fn expected_cache_key(&self) -> u64 {
        if let MovementKernelKind::ScatterPositions {
            input,
            starts,
            steps,
        } = &self.kind
        {
            fn push_u64(bytes: &mut Vec<u8>, value: u64) {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fn push_text(bytes: &mut Vec<u8>, value: &str) {
                push_u64(bytes, value.len() as u64);
                bytes.extend_from_slice(value.as_bytes());
            }
            let mut bytes = b"rustgrad-static-position-map-v1".to_vec();
            push_u64(&mut bytes, input.node.index() as u64);
            push_u64(&mut bytes, input.shape.rank() as u64);
            for dimension in input.shape.dims() {
                push_u64(&mut bytes, *dimension as u64);
            }
            push_text(&mut bytes, input.dtype.canonical_tinygrad_name());
            push_u64(&mut bytes, self.output.index() as u64);
            push_u64(&mut bytes, self.output_shape.rank() as u64);
            for dimension in self.output_shape.dims() {
                push_u64(&mut bytes, *dimension as u64);
            }
            push_text(&mut bytes, self.dtype.canonical_tinygrad_name());
            push_u64(&mut bytes, starts.len() as u64);
            for value in starts {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            push_u64(&mut bytes, steps.len() as u64);
            for value in steps {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            return bytes.into_iter().fold(0xcbf29ce484222325u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            });
        }
        let mut plan = self.clone();
        plan.cache_key = 0;
        let mut hasher = DefaultHasher::new();
        plan.hash(&mut hasher);
        hasher.finish()
    }
}

fn scalar_bits(dtype: DType, value: Scalar) -> u64 {
    let data = TensorData::scalar_with_dtype(value, dtype);
    match data.storage() {
        Storage::Bool(v) => u64::from(v[0]),
        Storage::I8(v) => v[0] as u8 as u64,
        Storage::U8(v) => v[0] as u64,
        Storage::I16(v) => v[0] as u16 as u64,
        Storage::U16(v) | Storage::F16(v) | Storage::BF16(v) => v[0] as u64,
        Storage::I32(v) => v[0] as u32 as u64,
        Storage::U32(v) => v[0] as u64,
        Storage::I64(v) => v[0] as u64,
        Storage::U64(v) => v[0],
        Storage::Float8(v) => v.as_raw()[0] as u64,
        Storage::F32(v) => v[0].to_bits() as u64,
        Storage::F64(v) => v[0].to_bits(),
    }
}
fn valid_bits(dtype: DType, bits: u64) -> bool {
    dtype.bits() == 64 || bits >> dtype.bits() == 0
}
fn scalar_data_from_bits(dtype: DType, bits: u64) -> Result<TensorData, MovementExecutionError> {
    if dtype == DType::Bool {
        return Ok(TensorData::scalar_with_dtype(
            Scalar::Bool(bits != 0),
            dtype,
        ));
    }
    let bytes = bits.to_le_bytes();
    TensorData::from_le_bytes([], dtype, &bytes[..dtype.itemsize()])
        .map_err(|_| MovementExecutionError::InvalidGeometry)
}

fn dense(shape: &Shape) -> Result<DenseIndex, MovementExecutionError> {
    DenseIndex::new(shape.clone()).map_err(|error| match error {
        crate::Error::ShapeOverflow(_) => MovementExecutionError::Overflow,
        _ => MovementExecutionError::InvalidGeometry,
    })
}

fn indexed_map(
    input: &MovementOperand,
    index: &MovementOperand,
    index_value: &TensorData,
    axis: usize,
) -> Result<Vec<(usize, usize)>, MovementExecutionError> {
    let input_index = dense(&input.shape)?;
    let index_index = dense(&index.shape)?;
    let mut mapped = Vec::with_capacity(index_index.len());
    for linear in 0..index_index.len() {
        let mut coords = index_index
            .coords(linear)
            .map_err(|_| MovementExecutionError::InvalidGeometry)?;
        coords[axis] = checked_index(
            index_value.scalar_at(linear),
            axis,
            input.shape.dims()[axis],
        )?;
        let destination = input_index
            .offset(&coords)
            .map_err(|_| MovementExecutionError::InvalidGeometry)?;
        mapped.push((destination, linear));
    }
    Ok(mapped)
}

fn checked_index(value: Scalar, axis: usize, dim: usize) -> Result<usize, MovementExecutionError> {
    let signed = match value {
        Scalar::I(value) => value,
        Scalar::U(value) => {
            i64::try_from(value).map_err(|_| MovementExecutionError::IndexOutOfBounds {
                axis,
                index: i64::MAX,
                dim,
            })?
        }
        _ => return Err(MovementExecutionError::InvalidGeometry),
    };
    let index = usize::try_from(signed).map_err(|_| MovementExecutionError::IndexOutOfBounds {
        axis,
        index: signed,
        dim,
    })?;
    if index >= dim {
        return Err(MovementExecutionError::IndexOutOfBounds {
            axis,
            index: signed,
            dim,
        });
    }
    Ok(index)
}

fn select_raw(storage: &Storage, offsets: &[usize]) -> Storage {
    macro_rules! selected {
        ($values:expr, $variant:ident) => {
            Storage::$variant(offsets.iter().map(|offset| $values[*offset]).collect())
        };
    }
    match storage {
        Storage::Bool(values) => selected!(values, Bool),
        Storage::I8(values) => selected!(values, I8),
        Storage::U8(values) => selected!(values, U8),
        Storage::Float8(values) => Storage::Float8(crate::Float8Storage::from_raw(
            values.format(),
            offsets
                .iter()
                .map(|offset| values.as_raw()[*offset])
                .collect(),
        )),
        Storage::I16(values) => selected!(values, I16),
        Storage::U16(values) => selected!(values, U16),
        Storage::I32(values) => selected!(values, I32),
        Storage::U32(values) => selected!(values, U32),
        Storage::I64(values) => selected!(values, I64),
        Storage::U64(values) => selected!(values, U64),
        Storage::F16(values) => selected!(values, F16),
        Storage::BF16(values) => selected!(values, BF16),
        Storage::F32(values) => selected!(values, F32),
        Storage::F64(values) => selected!(values, F64),
    }
}

fn select_many_raw(
    dtype: DType,
    operands: &[TensorData],
    map: &[(usize, usize)],
) -> Result<Storage, MovementExecutionError> {
    macro_rules! selected {
        ($variant:ident) => {{
            let mut output = Vec::with_capacity(map.len());
            for (operand, offset) in map {
                let Storage::$variant(values) = operands[*operand].storage() else {
                    return Err(MovementExecutionError::InvalidGeometry);
                };
                output.push(values[*offset]);
            }
            Storage::$variant(output)
        }};
    }
    Ok(match dtype {
        dtype if dtype.is_float8() => {
            let format = dtype.float8_format().expect("float8 dtype");
            let mut output = Vec::with_capacity(map.len());
            for (operand, offset) in map {
                let Storage::Float8(values) = operands[*operand].storage() else {
                    return Err(MovementExecutionError::InvalidGeometry);
                };
                if values.format() != format {
                    return Err(MovementExecutionError::InvalidGeometry);
                }
                output.push(values.as_raw()[*offset]);
            }
            Storage::Float8(crate::Float8Storage::from_raw(format, output))
        }
        DType::Bool => selected!(Bool),
        DType::I8 => selected!(I8),
        DType::U8 => selected!(U8),
        DType::I16 => selected!(I16),
        DType::U16 => selected!(U16),
        DType::I32 => selected!(I32),
        DType::U32 => selected!(U32),
        DType::I64 => selected!(I64),
        DType::U64 => selected!(U64),
        DType::F16 => selected!(F16),
        DType::BF16 => selected!(BF16),
        DType::F32 => selected!(F32),
        DType::F64 => selected!(F64),
        DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ => {
            unreachable!("float8 handled by the transport guard")
        }
    })
}

fn scatter_raw(
    base: &Storage,
    updates: &Storage,
    map: &[(usize, usize)],
) -> Result<Storage, MovementExecutionError> {
    macro_rules! scattered {
        ($variant:ident) => {{
            let Storage::$variant(mut output) = base.clone() else {
                return Err(MovementExecutionError::InvalidGeometry);
            };
            let Storage::$variant(updates) = updates else {
                return Err(MovementExecutionError::InvalidGeometry);
            };
            for (destination, source) in map {
                output[*destination] = updates[*source];
            }
            Storage::$variant(output)
        }};
    }
    Ok(match base {
        Storage::Float8(base) => {
            let Storage::Float8(updates) = updates else {
                return Err(MovementExecutionError::InvalidGeometry);
            };
            if base.format() != updates.format() {
                return Err(MovementExecutionError::InvalidGeometry);
            }
            let mut output = base.as_raw().to_vec();
            for (destination, source) in map {
                output[*destination] = updates.as_raw()[*source];
            }
            Storage::Float8(crate::Float8Storage::from_raw(base.format(), output))
        }
        Storage::Bool(_) => scattered!(Bool),
        Storage::I8(_) => scattered!(I8),
        Storage::U8(_) => scattered!(U8),
        Storage::I16(_) => scattered!(I16),
        Storage::U16(_) => scattered!(U16),
        Storage::I32(_) => scattered!(I32),
        Storage::U32(_) => scattered!(U32),
        Storage::I64(_) => scattered!(I64),
        Storage::U64(_) => scattered!(U64),
        Storage::F16(_) => scattered!(F16),
        Storage::BF16(_) => scattered!(BF16),
        Storage::F32(_) => scattered!(F32),
        Storage::F64(_) => scattered!(F64),
    })
}

fn validate_index_geometry(
    input: &MovementOperand,
    index: &MovementOperand,
    axis: usize,
) -> Result<(), MovementPlanError> {
    input
        .shape
        .numel()
        .and_then(|_| index.shape.numel())
        .map_err(|_| MovementPlanError::Overflow)?;
    if !index.dtype.is_integer()
        || axis >= input.shape.rank()
        || input.shape.rank() != index.shape.rank()
        || input
            .shape
            .dims()
            .iter()
            .zip(index.shape.dims())
            .enumerate()
            .any(|(dim, (input, index))| dim != axis && index > input)
    {
        return Err(MovementPlanError::InvalidGeometry);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend};
    use std::collections::HashMap;

    fn assert_storage_bits_eq(actual: &Storage, expected: &Storage) {
        match (actual, expected) {
            (Storage::F32(actual), Storage::F32(expected)) => assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            ),
            (Storage::F64(actual), Storage::F64(expected)) => assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            ),
            _ => assert_eq!(actual, expected),
        }
    }

    fn plan_result(graph: &Graph, output: NodeId, operands: &[TensorData]) -> TensorData {
        MovementKernelPlan::from_graph(graph, output)
            .unwrap()
            .execute(operands)
            .unwrap()
    }

    #[test]
    fn concat_preserves_raw_storage_for_every_dtype_and_zero_width() {
        let fixtures = vec![
            (Storage::Bool(vec![false]), Storage::Bool(vec![true])),
            (Storage::I8(vec![i8::MIN]), Storage::I8(vec![i8::MAX])),
            (Storage::U8(vec![0]), Storage::U8(vec![u8::MAX])),
            (Storage::I16(vec![i16::MIN]), Storage::I16(vec![i16::MAX])),
            (Storage::U16(vec![0]), Storage::U16(vec![u16::MAX])),
            (Storage::I32(vec![i32::MIN]), Storage::I32(vec![i32::MAX])),
            (Storage::U32(vec![0]), Storage::U32(vec![u32::MAX])),
            (Storage::I64(vec![i64::MIN]), Storage::I64(vec![i64::MAX])),
            (Storage::U64(vec![0]), Storage::U64(vec![u64::MAX])),
            (Storage::F16(vec![0x8000]), Storage::F16(vec![0x7e01])),
            (Storage::BF16(vec![0x8000]), Storage::BF16(vec![0x7fc1])),
            (
                Storage::F32(vec![f32::from_bits(0x8000_0000)]),
                Storage::F32(vec![f32::from_bits(0x7fc0_0001)]),
            ),
            (
                Storage::F64(vec![f64::from_bits(0x8000_0000_0000_0000)]),
                Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_0001)]),
            ),
        ];
        for (left_storage, right_storage) in fixtures {
            let dtype = left_storage.dtype();
            let left = TensorData::from_storage([1, 1], left_storage).unwrap();
            let right = TensorData::from_storage([1, 1], right_storage).unwrap();
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 1], dtype);
            let output = graph.concat([lhs, rhs], 1).unwrap();
            let result = plan_result(&graph, output, &[left.clone(), right.clone()]);
            let expected = match (left.storage(), right.storage()) {
                (Storage::Bool(a), Storage::Bool(b)) => Storage::Bool([a.as_slice(), b].concat()),
                (Storage::I8(a), Storage::I8(b)) => Storage::I8([a.as_slice(), b].concat()),
                (Storage::U8(a), Storage::U8(b)) => Storage::U8([a.as_slice(), b].concat()),
                (Storage::I16(a), Storage::I16(b)) => Storage::I16([a.as_slice(), b].concat()),
                (Storage::U16(a), Storage::U16(b)) => Storage::U16([a.as_slice(), b].concat()),
                (Storage::I32(a), Storage::I32(b)) => Storage::I32([a.as_slice(), b].concat()),
                (Storage::U32(a), Storage::U32(b)) => Storage::U32([a.as_slice(), b].concat()),
                (Storage::I64(a), Storage::I64(b)) => Storage::I64([a.as_slice(), b].concat()),
                (Storage::U64(a), Storage::U64(b)) => Storage::U64([a.as_slice(), b].concat()),
                (Storage::F16(a), Storage::F16(b)) => Storage::F16([a.as_slice(), b].concat()),
                (Storage::BF16(a), Storage::BF16(b)) => Storage::BF16([a.as_slice(), b].concat()),
                (Storage::F32(a), Storage::F32(b)) => Storage::F32([a.as_slice(), b].concat()),
                (Storage::F64(a), Storage::F64(b)) => Storage::F64([a.as_slice(), b].concat()),
                _ => unreachable!(),
            };
            assert_storage_bits_eq(result.storage(), &expected);
            assert_eq!(result.shape(), &Shape::from([1, 2]));
        }

        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 0], DType::I32);
        let rhs = graph.input_dtype("rhs", [2, 3], DType::I32);
        let output = graph.concat([lhs, rhs], 1).unwrap();
        let empty = TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap();
        let values =
            TensorData::from_storage([2, 3], Storage::I32(vec![1, 2, 3, 4, 5, 6])).unwrap();
        assert_eq!(
            plan_result(&graph, output, &[empty, values.clone()]),
            values
        );

        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [0, 2], DType::U8);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::U8);
        let output = graph.concat([lhs, rhs], 0).unwrap();
        let empty = TensorData::from_storage([0, 2], Storage::U8(vec![])).unwrap();
        let values = TensorData::from_storage([2, 2], Storage::U8(vec![1, 2, 3, 4])).unwrap();
        assert_eq!(
            plan_result(&graph, output, &[empty, values.clone()]),
            values
        );
    }

    #[test]
    fn mixed_concat_and_every_integer_gather_match_cpu_oracle() {
        let mut concat_graph = Graph::new();
        let lhs = concat_graph.input_dtype("lhs", [1, 2], DType::I8);
        let rhs = concat_graph.input_dtype("rhs", [1, 1], DType::U8);
        let concat = concat_graph.concat([lhs, rhs], 1).unwrap();
        let left = TensorData::from_storage([1, 2], Storage::I8(vec![-2, 3])).unwrap();
        let right = TensorData::from_storage([1, 1], Storage::U8(vec![250])).unwrap();
        let oracle = CpuBackend
            .execute(
                &concat_graph,
                concat,
                &HashMap::from([("lhs".into(), left.clone()), ("rhs".into(), right.clone())]),
            )
            .unwrap();
        assert_eq!(plan_result(&concat_graph, concat, &[left, right]), oracle);

        for dtype in [
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2, 3], DType::F16);
            let index = graph.input_dtype("index", [2, 2], dtype);
            let output = graph.gather(input, index, 1).unwrap();
            let input_value = TensorData::from_storage(
                [2, 3],
                Storage::F16(vec![0x8000, 0x7e01, 0x3c00, 0x4000, 0x4200, 0x4400]),
            )
            .unwrap();
            let index_value = TensorData::from_scalars(
                [2, 2],
                dtype,
                [Scalar::I(2), Scalar::I(0), Scalar::I(1), Scalar::I(1)],
            )
            .unwrap();
            let oracle = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        ("input".into(), input_value.clone()),
                        ("index".into(), index_value.clone()),
                    ]),
                )
                .unwrap();
            assert_eq!(
                plan_result(&graph, output, &[input_value, index_value]),
                oracle,
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn gather_and_scatter_preflight_indices_and_preserve_row_major_semantics() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 3], DType::I32);
        let index = graph.input_dtype("index", [1, 2], DType::I64);
        let gather = graph.gather(input, index, 1).unwrap();
        let plan = MovementKernelPlan::from_graph(&graph, gather).unwrap();
        let input_value = TensorData::from_storage([1, 3], Storage::I32(vec![1, 2, 3])).unwrap();
        for selected in [-1, 3] {
            let index_value =
                TensorData::from_storage([1, 2], Storage::I64(vec![0, selected])).unwrap();
            assert!(matches!(
                plan.execute(&[input_value.clone(), index_value]),
                Err(MovementExecutionError::IndexOutOfBounds {
                    axis: 1,
                    index,
                    dim: 3
                }) if index == selected
            ));
        }
        assert!(matches!(
            plan.execute(std::slice::from_ref(&input_value)),
            Err(MovementExecutionError::OperandCount {
                expected: 2,
                actual: 1
            })
        ));
        let wrong = TensorData::from_storage([1, 2], Storage::I32(vec![0, 1])).unwrap();
        assert!(matches!(
            plan.execute(&[input_value, wrong]),
            Err(MovementExecutionError::OperandDescriptor { position: 1, .. })
        ));

        let mut unsigned_graph = Graph::new();
        let input = unsigned_graph.input_dtype("input", [1, 3], DType::I32);
        let index = unsigned_graph.input_dtype("index", [1, 1], DType::U64);
        let gather = unsigned_graph.gather(input, index, 1).unwrap();
        let unsigned_plan = MovementKernelPlan::from_graph(&unsigned_graph, gather).unwrap();
        assert!(matches!(
            unsigned_plan.execute(&[
                TensorData::from_storage([1, 3], Storage::I32(vec![1, 2, 3])).unwrap(),
                TensorData::from_storage([1, 1], Storage::U64(vec![u64::MAX])).unwrap()
            ]),
            Err(MovementExecutionError::IndexOutOfBounds {
                axis: 1,
                index: i64::MAX,
                dim: 3
            })
        ));

        let mut empty_graph = Graph::new();
        let input = empty_graph.input_dtype("input", [2, 0], DType::I32);
        let index = empty_graph.input_dtype("index", [2, 0], DType::U64);
        let output = empty_graph.gather(input, index, 1).unwrap();
        let empty_input = TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap();
        let empty_index = TensorData::from_storage([2, 0], Storage::U64(vec![])).unwrap();
        assert_eq!(
            plan_result(&empty_graph, output, &[empty_input, empty_index]).shape(),
            &Shape::from([2, 0])
        );

        let mut scatter_graph = Graph::new();
        let base = scatter_graph.input_dtype("base", [1, 3], DType::F16);
        let index = scatter_graph.input_dtype("index", [1, 3], DType::I32);
        let updates = scatter_graph.input_dtype("updates", [1, 3], DType::F16);
        let output = scatter_graph.scatter(base, index, updates, 1).unwrap();
        let result = plan_result(
            &scatter_graph,
            output,
            &[
                TensorData::from_storage([1, 3], Storage::F16(vec![0x8000, 0x3c00, 0x7e01]))
                    .unwrap(),
                TensorData::from_storage([1, 3], Storage::I32(vec![1, 1, 1])).unwrap(),
                TensorData::from_storage([1, 3], Storage::F16(vec![0x4000, 0x4200, 0x7e55]))
                    .unwrap(),
            ],
        );
        assert_eq!(
            result.storage(),
            &Storage::F16(vec![0x8000, 0x7e55, 0x7e01])
        );

        let mut empty_scatter = Graph::new();
        let base = empty_scatter.input_dtype("base", [2, 0], DType::F32);
        let index = empty_scatter.input_dtype("index", [2, 0], DType::I32);
        let updates = empty_scatter.input_dtype("updates", [2, 0], DType::F32);
        let output = empty_scatter.scatter_add(base, index, updates, 1).unwrap();
        assert_eq!(
            plan_result(
                &empty_scatter,
                output,
                &[
                    TensorData::from_storage([2, 0], Storage::F32(vec![])).unwrap(),
                    TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap(),
                    TensorData::from_storage([2, 0], Storage::F32(vec![])).unwrap(),
                ]
            )
            .shape(),
            &Shape::from([2, 0])
        );
    }

    #[test]
    fn scatter_add_f32_and_f64_match_cpu_row_major_accumulation() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let base = graph.input_dtype("base", [1, 2], dtype);
            let index = graph.input_dtype("index", [1, 3], DType::I32);
            let updates = graph.input_dtype("updates", [1, 3], dtype);
            let output = graph.scatter_add(base, index, updates, 1).unwrap();
            let base_value =
                TensorData::from_scalars([1, 2], dtype, [Scalar::F(1.0), Scalar::F(-2.0)]).unwrap();
            let index_value =
                TensorData::from_storage([1, 3], Storage::I32(vec![0, 0, 1])).unwrap();
            let updates_value = TensorData::from_scalars(
                [1, 3],
                dtype,
                [Scalar::F(0.25), Scalar::F(0.5), Scalar::F(4.0)],
            )
            .unwrap();
            let inputs = HashMap::from([
                ("base".into(), base_value.clone()),
                ("index".into(), index_value.clone()),
                ("updates".into(), updates_value.clone()),
            ]);
            let oracle = CpuBackend.execute(&graph, output, &inputs).unwrap();
            assert_eq!(
                plan_result(&graph, output, &[base_value, index_value, updates_value]),
                oracle,
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn computed_affine_copy_is_dense_exact_and_rejects_malformed_maps() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let producer = graph.relu(input).unwrap();
        let viewed = graph.reshape(producer, [1, 4]).unwrap();
        let plan = MovementKernelPlan::from_computed_affine_view(&graph, viewed).unwrap();
        let value = TensorData::new([2, 2], vec![-1.0f32, 2.0, 3.0, -4.0]).unwrap();
        let produced = CpuBackend
            .execute(&graph, producer, &HashMap::from([("input".into(), value)]))
            .unwrap();
        let copied = plan.execute(std::slice::from_ref(&produced)).unwrap();
        let oracle = CpuBackend
            .execute(
                &graph,
                viewed,
                &HashMap::from([(
                    "input".into(),
                    TensorData::new([2, 2], vec![-1.0f32, 2.0, 3.0, -4.0]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(copied.storage(), oracle.storage());

        let mut malformed = plan.clone();
        let MovementKernelKind::AffineCopy { view, .. } = &mut malformed.kind else {
            unreachable!();
        };
        view.offset = 4;
        malformed.cache_key = malformed.expected_cache_key();
        assert_eq!(
            malformed.validate(),
            Err(MovementPlanError::InvalidGeometry)
        );

        let mut stale = plan.clone();
        stale.output_shape = Shape::from([4, 1]);
        assert_eq!(stale.validate(), Err(MovementPlanError::InvalidGeometry));

        let mut stale_cache = plan.clone();
        stale_cache.cache_key ^= 1;
        assert_eq!(
            stale_cache.validate(),
            Err(MovementPlanError::InvalidGeometry)
        );

        let mut input_bytes_overflow = plan.clone();
        input_bytes_overflow.output_shape = Shape::from([0]);
        let MovementKernelKind::AffineCopy { input, view } = &mut input_bytes_overflow.kind else {
            unreachable!();
        };
        input.shape = Shape::from([usize::MAX]);
        view.source_shape = input.shape.clone();
        view.logical_shape = Shape::from([0]);
        view.strides = vec![1];
        view.offset = 0;
        input_bytes_overflow.cache_key = input_bytes_overflow.expected_cache_key();
        assert_eq!(
            input_bytes_overflow.validate(),
            Err(MovementPlanError::Overflow)
        );

        let mut output_bytes_overflow = plan.clone();
        output_bytes_overflow.output_shape = Shape::from([usize::MAX]);
        let MovementKernelKind::AffineCopy { input, view } = &mut output_bytes_overflow.kind else {
            unreachable!();
        };
        input.shape = Shape::from([1]);
        view.source_shape = input.shape.clone();
        view.logical_shape = output_bytes_overflow.output_shape.clone();
        view.strides = vec![0];
        view.offset = 0;
        output_bytes_overflow.cache_key = output_bytes_overflow.expected_cache_key();
        assert_eq!(
            output_bytes_overflow.validate(),
            Err(MovementPlanError::Overflow)
        );

        let mut alias = plan;
        alias.output = producer;
        alias.cache_key = alias.expected_cache_key();
        assert_eq!(alias.validate(), Err(MovementPlanError::InvalidGeometry));
    }

    #[test]
    fn computed_affine_copy_accepts_broadcast_and_signed_read_maps() {
        let mut broadcast = Graph::new();
        let input = broadcast.input_dtype("input", [2, 1], DType::I32);
        let producer = broadcast.square(input).unwrap();
        let viewed = broadcast.expand(producer, [2, 3]).unwrap();
        let plan = MovementKernelPlan::from_computed_affine_view(&broadcast, viewed).unwrap();
        let MovementKernelKind::AffineCopy { view, .. } = &plan.kind else {
            panic!("affine copy")
        };
        assert_eq!(view.strides, vec![1, 0]);
        let value = TensorData::from_storage([2, 1], Storage::I32(vec![7, 11])).unwrap();
        assert_eq!(
            plan.execute(&[value]).unwrap().storage(),
            &Storage::I32(vec![7, 7, 7, 11, 11, 11])
        );

        let mut reverse = Graph::new();
        let input = reverse.input_dtype("input", [4], DType::I32);
        let producer = reverse.square(input).unwrap();
        let viewed = reverse
            .stride(
                producer,
                [crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            )
            .unwrap();
        let plan = MovementKernelPlan::from_computed_affine_view(&reverse, viewed).unwrap();
        let MovementKernelKind::AffineCopy { view, .. } = &plan.kind else {
            panic!("affine copy")
        };
        assert_eq!(view.offset, 3);
        assert_eq!(view.strides, vec![-1]);
        let value = TensorData::from_storage([4], Storage::I32(vec![1, 4, 9, 16])).unwrap();
        assert_eq!(
            plan.execute(&[value]).unwrap().storage(),
            &Storage::I32(vec![16, 9, 4, 1])
        );
    }

    #[test]
    fn scheduled_contiguous_uses_affine_sources_and_dense_computed_fallback() {
        let mut source = Graph::new();
        let input = source.input_dtype("input", [2, 3], DType::F32);
        let view = source.permute(input, [1, 0]).unwrap();
        let output = source.contiguous(view).unwrap();
        let plan = MovementKernelPlan::from_scheduled_graph(&source, output).unwrap();
        let MovementKernelKind::AffineCopy {
            input: operand,
            view,
        } = &plan.kind
        else {
            panic!("source-backed contiguous affine copy")
        };
        assert_eq!(operand.node, input);
        assert_eq!(view.logical_shape, Shape::from([3, 2]));
        assert_eq!(view.strides, [1, 3]);
        let copy = plan.raw_copy().unwrap().unwrap();
        assert_eq!(copy.input_elements(), 6);
        assert_eq!(copy.input_bytes(), 24);
        assert_eq!(copy.elements(), 6);
        assert_eq!(copy.bytes(), 24);
        assert_eq!(copy.width(), 4);
        let address = copy.address().unwrap().unwrap();
        assert_eq!(address.offset, 0);
        assert_eq!(
            address.axes,
            [
                RawCopyAxis {
                    output_axis: 0,
                    dimension: 3,
                    divisor: 2,
                    stride: 1,
                    reversed: false,
                },
                RawCopyAxis {
                    output_axis: 1,
                    dimension: 2,
                    divisor: 1,
                    stride: 3,
                    reversed: false,
                }
            ]
        );

        let mut expanded = Graph::new();
        let input = expanded.input_dtype("input", [2, 1], DType::I32);
        let view = expanded.expand(input, [2, 3]).unwrap();
        let output = expanded.contiguous(view).unwrap();
        let plan = MovementKernelPlan::from_scheduled_graph(&expanded, output).unwrap();
        assert_eq!(
            plan.execute(&[TensorData::from_storage([2, 1], Storage::I32(vec![7, 11])).unwrap()])
                .unwrap()
                .storage(),
            &Storage::I32(vec![7, 7, 7, 11, 11, 11])
        );

        let mut reversed = Graph::new();
        let input = reversed.input_dtype("input", [4], DType::I32);
        let view = reversed
            .stride(
                input,
                [crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            )
            .unwrap();
        let output = reversed.contiguous(view).unwrap();
        let plan = MovementKernelPlan::from_scheduled_graph(&reversed, output).unwrap();
        assert_eq!(
            plan.execute(&[TensorData::from_storage([4], Storage::I32(vec![1, 2, 3, 4])).unwrap()])
                .unwrap()
                .storage(),
            &Storage::I32(vec![4, 3, 2, 1])
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 3], DType::F32);
        let view = empty.permute(input, [1, 0]).unwrap();
        let output = empty.contiguous(view).unwrap();
        let plan = MovementKernelPlan::from_scheduled_graph(&empty, output).unwrap();
        assert_eq!(
            plan.execute(&[TensorData::from_storage([0, 3], Storage::F32(vec![])).unwrap()])
                .unwrap()
                .shape(),
            &Shape::from([3, 0])
        );

        let mut computed = Graph::new();
        let input = computed.input_dtype("input", [2, 3], DType::F32);
        let producer = computed.square(input).unwrap();
        let view = computed.permute(producer, [1, 0]).unwrap();
        let output = computed.contiguous(view).unwrap();
        let plan = MovementKernelPlan::from_scheduled_graph(&computed, output).unwrap();
        assert!(matches!(
            &plan.kind,
            MovementKernelKind::AffineCopy { input, view }
                if input.node == producer
                    && view.logical_shape == Shape::from([3, 2])
                    && view.strides == [1, 3]
        ));

        let output = computed.contiguous(producer).unwrap();
        let plan = MovementKernelPlan::from_scheduled_graph(&computed, output).unwrap();
        assert!(matches!(
            &plan.kind,
            MovementKernelKind::Contiguous { input } if input.node == producer
        ));
        let copy = plan.raw_copy().unwrap().unwrap();
        assert_eq!(copy.input_elements(), 6);
        assert_eq!(copy.elements(), 6);
        assert!(copy.address().unwrap().is_none());

        let mut non_affine = Graph::new();
        let input = non_affine.input_dtype("input", [2, 3], DType::F32);
        let permuted = non_affine.permute(input, [1, 0]).unwrap();
        let reshaped = non_affine.reshape(permuted, [6]).unwrap();
        let output = non_affine.contiguous(reshaped).unwrap();
        let plan = MovementKernelPlan::from_scheduled_graph(&non_affine, output).unwrap();
        assert!(matches!(
            &plan.kind,
            MovementKernelKind::Contiguous { input } if input.node == reshaped
        ));

        let mut malformed = Graph::new();
        let input = malformed.input_dtype("input", [2, 3], DType::F32);
        let view = malformed.permute(input, [1, 0]).unwrap();
        let output = malformed.contiguous(view).unwrap();
        malformed.nodes[view.index()].op = Op::Permute {
            input,
            axes: vec![0, 0],
        };
        assert_eq!(
            MovementKernelPlan::from_scheduled_graph(&malformed, output),
            Err(MovementPlanError::InvalidGeometry)
        );
        assert!(matches!(
            crate::schedule(&malformed, output),
            Err(crate::ScheduleError::Binding(_))
        ));
    }

    #[test]
    fn pad_plan_canonicalizes_typed_fill_and_geometry_before_execution() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2], DType::F16);
        let output = graph.pad(input, [(1, 0), (0, 2)], Scalar::F(-0.0)).unwrap();
        let plan = MovementKernelPlan::from_graph(&graph, output).unwrap();
        assert_eq!(plan.output_shape, Shape::from([2, 4]));
        let MovementKernelKind::Pad {
            padding, fill_bits, ..
        } = &plan.kind
        else {
            panic!("pad plan")
        };
        assert_eq!(padding.as_slice(), [(1, 0), (0, 2)]);
        assert_eq!(*fill_bits, 0x8000);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn bitcast_plan_reinterprets_exact_bytes_and_validates_descriptors() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 4], DType::U8);
        let output = graph.bitcast(input, DType::U32).unwrap();
        let plan = MovementKernelPlan::from_graph(&graph, output).unwrap();
        assert_eq!(plan.output_shape, Shape::from([2, 1]));
        assert_eq!(plan.dtype, DType::U32);
        assert!(matches!(
            &plan.kind,
            MovementKernelKind::Bitcast { input: operand }
                if operand.node == input
                    && operand.shape == Shape::from([2, 4])
                    && operand.dtype == DType::U8
        ));
        let source =
            TensorData::from_storage([2, 4], Storage::U8(vec![1, 2, 3, 4, 5, 6, 7, 8])).unwrap();
        let result = plan.execute(&[source]).unwrap();
        assert_eq!(result.shape(), &Shape::from([2, 1]));
        assert_eq!(
            result.storage(),
            &Storage::U32(vec![0x0403_0201, 0x0807_0605])
        );

        let mut malformed = plan.clone();
        malformed.output_shape = Shape::from([1, 2]);
        malformed.cache_key = malformed.expected_cache_key();
        assert_eq!(
            malformed.validate(),
            Err(MovementPlanError::InvalidGeometry)
        );
    }

    #[test]
    fn static_position_map_proves_reverse_gap_empty_and_invalid_geometry() {
        let map =
            StaticPositionMap::from_i64(Shape::from([3]), Shape::from([7]), vec![5], vec![-2])
                .unwrap();
        let view = map.read_view().unwrap();
        assert_eq!(view.offset, 5);
        assert_eq!(view.strides, [-2]);
        assert_eq!(map.destination_offsets().unwrap(), [5, 3, 1]);

        let empty = StaticPositionMap::from_i64(
            Shape::from([0, 4]),
            Shape::from([0, 9]),
            vec![-1, 8],
            vec![-1, -2],
        )
        .unwrap();
        let view = empty.read_view().unwrap();
        assert_eq!(view.offset, 0);
        assert_eq!(view.strides, [0, 0]);
        assert!(empty.destination_offsets().unwrap().is_empty());
        assert_eq!(
            StaticPositionMap::from_i64(
                Shape::from([0, 4]),
                Shape::from([0, 9]),
                vec![-1, 8],
                vec![0, -2],
            ),
            Err(MovementPlanError::InvalidGeometry)
        );

        let scalar =
            StaticPositionMap::from_i64(Shape::from([]), Shape::from([]), vec![], vec![]).unwrap();
        assert_eq!(scalar.destination_offsets().unwrap(), [0]);

        let singleton_extreme = StaticPositionMap::from_i64(
            Shape::from([1]),
            Shape::from([2]),
            vec![1],
            vec![i64::MIN],
        )
        .unwrap();
        let singleton_view = singleton_extreme.read_view().unwrap();
        assert_eq!(singleton_view.offset, 1);
        assert_eq!(singleton_view.strides, [0]);

        let mut empty_plan = MovementKernelPlan {
            kind: MovementKernelKind::ScatterPositions {
                input: MovementOperand {
                    node: NodeId::from_index(0),
                    shape: empty.input_shape.clone(),
                    dtype: DType::F32,
                },
                starts: empty.starts.clone(),
                steps: empty.steps.clone(),
            },
            output: NodeId::from_index(1),
            output_shape: empty.output_shape.clone(),
            dtype: DType::F32,
            cache_key: 0,
        };
        empty_plan.cache_key = empty_plan.expected_cache_key();
        let value = TensorData::from_storage([0, 4], Storage::F32(vec![])).unwrap();
        assert_eq!(
            empty_plan.execute(&[value]).unwrap().shape(),
            &Shape::from([0, 9])
        );

        assert_eq!(
            StaticPositionMap::from_i64(Shape::from([2]), Shape::from([3]), vec![0], vec![0]),
            Err(MovementPlanError::InvalidGeometry)
        );
        assert_eq!(
            StaticPositionMap::from_i64(Shape::from([2]), Shape::from([3]), vec![2], vec![1]),
            Err(MovementPlanError::InvalidGeometry)
        );
        assert_eq!(
            StaticPositionMap::from_i64(Shape::from([2]), Shape::from([3]), vec![-1], vec![1]),
            Err(MovementPlanError::InvalidGeometry)
        );
        assert_eq!(
            StaticPositionMap::from_i64(
                Shape::from([usize::MAX, 2]),
                Shape::from([usize::MAX, 2]),
                vec![0, 0],
                vec![1, 1],
            ),
            Err(MovementPlanError::Overflow)
        );

        let mut byte_overflow = MovementKernelPlan {
            kind: MovementKernelKind::ScatterPositions {
                input: MovementOperand {
                    node: NodeId::from_index(0),
                    shape: Shape::from([0]),
                    dtype: DType::F64,
                },
                starts: vec![-1],
                steps: vec![-1],
            },
            output: NodeId::from_index(1),
            output_shape: Shape::from([usize::MAX]),
            dtype: DType::F64,
            cache_key: 0,
        };
        byte_overflow.cache_key = byte_overflow.expected_cache_key();
        assert_eq!(byte_overflow.validate(), Err(MovementPlanError::Overflow));
    }

    #[test]
    fn static_position_movement_preserves_every_storage_width_and_reverse_vjp() {
        for dtype in DType::ALL {
            let width = dtype.itemsize();
            let input_bytes = match dtype {
                DType::Bool => vec![1, 0],
                DType::F16 => [0x7e01u16, 0x8000]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                DType::BF16 => [0x7fc1u16, 0x8000]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                DType::F32 => [0x7fc0_0001u32, 0x8000_0000]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
                DType::F64 => [0x7ff8_0000_0000_0001u64, 0x8000_0000_0000_0000]
                    .into_iter()
                    .flat_map(u64::to_le_bytes)
                    .collect(),
                DType::F8E4M3 | DType::F8E4M3FNUZ | DType::F8E5M2 | DType::F8E5M2FNUZ => {
                    vec![0x81, 0x7f]
                }
                _ => (0..2 * width)
                    .map(|index| 0x81u8.wrapping_add(index as u8))
                    .collect::<Vec<_>>(),
            };
            let input = TensorData::from_le_bytes([2], dtype, &input_bytes).unwrap();
            let operand = MovementOperand {
                node: NodeId::from_index(0),
                shape: Shape::from([2]),
                dtype,
            };
            let mut plan = MovementKernelPlan {
                kind: MovementKernelKind::ScatterPositions {
                    input: operand.clone(),
                    starts: vec![4],
                    steps: vec![-2],
                },
                output: NodeId::from_index(1),
                output_shape: Shape::from([5]),
                dtype,
                cache_key: 0,
            };
            plan.cache_key = plan.expected_cache_key();
            let placed = plan.execute(std::slice::from_ref(&input)).unwrap();
            let placed_bytes = placed.to_le_bytes().unwrap();
            assert_eq!(&placed_bytes[4 * width..5 * width], &input_bytes[..width]);
            assert_eq!(&placed_bytes[2 * width..3 * width], &input_bytes[width..]);
            for lane in [0, 1, 3] {
                assert_eq!(
                    &placed_bytes[lane * width..(lane + 1) * width],
                    vec![0; width]
                );
            }

            let map =
                StaticPositionMap::from_i64(Shape::from([2]), Shape::from([5]), vec![4], vec![-2])
                    .unwrap();
            let mut reverse = MovementKernelPlan {
                kind: MovementKernelKind::AffineCopy {
                    input: MovementOperand {
                        node: NodeId::from_index(1),
                        shape: Shape::from([5]),
                        dtype,
                    },
                    view: map.read_view().unwrap(),
                },
                output: NodeId::from_index(2),
                output_shape: Shape::from([2]),
                dtype,
                cache_key: 0,
            };
            reverse.cache_key = reverse.expected_cache_key();
            assert_eq!(
                reverse.execute(&[placed]).unwrap().to_le_bytes().unwrap(),
                input_bytes
            );
        }
    }

    #[test]
    fn static_position_graph_admission_is_atomic() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let before = graph.node_count();
        assert!(
            graph
                .scatter_positions(input, Shape::from([2]), vec![0], vec![0])
                .is_err()
        );
        assert_eq!(graph.node_count(), before);
        assert!(
            graph
                .scatter_positions(input, Shape::from([2]), vec![1], vec![1])
                .is_err()
        );
        assert_eq!(graph.node_count(), before);

        let mut source_overflow = Graph::new();
        let input = source_overflow.input_dtype("input", [usize::MAX], DType::F64);
        let before = source_overflow.node_count();
        assert!(matches!(
            source_overflow.scatter_positions(input, Shape::from([usize::MAX]), vec![0], vec![1],),
            Err(crate::Error::ShapeOverflow(_))
        ));
        assert_eq!(source_overflow.node_count(), before);

        let mut output_overflow = Graph::new();
        let input = output_overflow.input_dtype("input", [0], DType::F64);
        let before = output_overflow.node_count();
        assert!(matches!(
            output_overflow
                .scatter_positions(input, Shape::from([usize::MAX]), vec![-1], vec![-1],),
            Err(crate::Error::ShapeOverflow(_))
        ));
        assert_eq!(output_overflow.node_count(), before);

        let mut vjp_overflow = Graph::new();
        let cotangent = vjp_overflow.input_dtype("cotangent", [0], DType::F64);
        let before = vjp_overflow.node_count();
        assert!(matches!(
            vjp_overflow.scatter_positions_vjp(
                cotangent,
                Shape::from([usize::MAX]),
                vec![0],
                vec![1],
            ),
            Err(crate::Error::ShapeOverflow(_))
        ));
        assert_eq!(vjp_overflow.node_count(), before);
    }
}
