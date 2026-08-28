use super::{
    shape::{normalize_axes, reduction_shape, unary_dtype},
    Graph, NodeId,
};
use crate::{
    DType, Error, ReduceKind, ReductionDType, Result, Scalar, Shape, TensorData,
    UnaryOp, VarianceCorrection,
};

struct MeanPlan {
    axes: Vec<isize>,
    output_shape: Shape,
    sum_dtypes: ReductionDType,
    division_dtype: DType,
    output_dtype: DType,
    divisor: TensorData,
}

enum MaxLowering {
    Identity,
    IdentityValue(Scalar),
    Reduce,
}

struct MaxPlan {
    axes: Vec<isize>,
    output_shape: Shape,
    dtype: DType,
    lowering: MaxLowering,
}

struct MinPlan {
    axes: Vec<isize>,
    output_shape: Shape,
    dtype: DType,
    lowering: MaxLowering,
}

struct VariancePlan {
    axes: Vec<isize>,
    mean_shape: Shape,
    output_shape: Shape,
    accumulation: DType,
    division_dtype: DType,
    output_dtype: DType,
    mean_divisor: TensorData,
    divisor: TensorData,
}

struct AllPlan {
    axes: Vec<isize>,
    output_shape: Shape,
}

enum AnyLowering {
    Identity,
    IdentityValue,
    Reduce,
}

struct AnyPlan {
    axes: Vec<isize>,
    output_shape: Shape,
    lowering: AnyLowering,
}

struct ArgmaxPlan {
    flatten: bool,
    work_shape: Shape,
    axis: isize,
    keepdim: bool,
    output_shape: Shape,
    first_bounds: Vec<(usize, usize)>,
    sentinel: TensorData,
    empty: bool,
}

enum ArgminInverse {
    Negate,
    LogicalNot,
    BitwiseNot(TensorData),
}

struct ArgminPlan {
    argmax: ArgmaxPlan,
    inverse: ArgminInverse,
}

fn max_reduction_identity(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 => Scalar::I(i8::MIN.into()),
        DType::U8 => Scalar::U(0),
        DType::I16 => Scalar::I(i16::MIN.into()),
        DType::U16 => Scalar::U(0),
        DType::I32 => Scalar::I(i32::MIN.into()),
        DType::U32 => Scalar::U(0),
        DType::I64 => Scalar::I(i64::MIN),
        DType::U64 => Scalar::U(0),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(f64::NEG_INFINITY),
    }
}

fn min_reduction_identity(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(true),
        DType::I8 => Scalar::I(i8::MAX.into()),
        DType::U8 => Scalar::U(u8::MAX.into()),
        DType::I16 => Scalar::I(i16::MAX.into()),
        DType::U16 => Scalar::U(u16::MAX.into()),
        DType::I32 => Scalar::I(i32::MAX.into()),
        DType::U32 => Scalar::U(u32::MAX.into()),
        DType::I64 => Scalar::I(i64::MAX),
        DType::U64 => Scalar::U(u64::MAX),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(f64::INFINITY),
    }
}

fn max_plan(
    input: NodeId,
    shape: &Shape,
    dtype: DType,
    axes: Option<Vec<isize>>,
    keepdim: bool,
) -> Result<MaxPlan> {
    let extent = |shape: &Shape| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let axes = if shape.rank() == 0 {
        if axes.as_ref().is_some_and(|axes| {
            axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0))
        }) {
            return Err(Error::InvalidReductionAxes { node: input, axes: vec![usize::MAX], rank: 0 });
        }
        Vec::new()
    } else {
        normalize_axes(input, shape.rank(), axes)?
    };
    let output_shape = reduction_shape(shape, &axes, keepdim);
    extent(shape)?;
    extent(&output_shape)?;
    let lowering = if axes.is_empty() {
        MaxLowering::Identity
    } else if output_shape.numel()? > 0 && axes.iter().any(|axis| shape.dims()[*axis] == 0) {
        MaxLowering::IdentityValue(max_reduction_identity(dtype))
    } else {
        MaxLowering::Reduce
    };
    Ok(MaxPlan {
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        output_shape,
        dtype,
        lowering,
    })
}

fn min_plan(
    input: NodeId,
    shape: &Shape,
    dtype: DType,
    axes: Option<Vec<isize>>,
    keepdim: bool,
) -> Result<MinPlan> {
    let extent = |shape: &Shape| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let axes = if shape.rank() == 0 {
        if axes.as_ref().is_some_and(|axes| {
            axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0))
        }) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes: vec![usize::MAX],
                rank: 0,
            });
        }
        Vec::new()
    } else {
        normalize_axes(input, shape.rank(), axes)?
    };
    let output_shape = reduction_shape(shape, &axes, keepdim);
    extent(shape)?;
    extent(&output_shape)?;
    let lowering = if axes.is_empty() {
        MaxLowering::Identity
    } else if output_shape.numel()? > 0 && axes.iter().any(|axis| shape.dims()[*axis] == 0) {
        MaxLowering::IdentityValue(min_reduction_identity(dtype))
    } else {
        MaxLowering::Reduce
    };
    Ok(MinPlan {
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        output_shape,
        dtype,
        lowering,
    })
}

fn variance_plan(
    input: NodeId,
    shape: &Shape,
    input_dtype: DType,
    axes: Option<Vec<isize>>,
    keepdim: bool,
    correction: Option<VarianceCorrection>,
) -> Result<VariancePlan> {
    let axes = if shape.rank() == 0 {
        if axes.as_ref().is_some_and(|axes| {
            axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0))
        }) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes: vec![usize::MAX],
                rank: 0,
            });
        }
        Vec::new()
    } else {
        normalize_axes(input, shape.rank(), axes)?
    };
    let count = axes.iter().try_fold(1usize, |count, axis| {
        count
            .checked_mul(shape.dims()[*axis])
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    })?;
    let denominator = variance_denominator(
        count,
        correction.unwrap_or(VarianceCorrection::UNBIASED),
        shape,
    )?;
    let accumulation = ReductionDType::sum_default(input_dtype).accumulator;
    let division_dtype = if accumulation.is_float() {
        accumulation
    } else {
        DType::F32
    };
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let mean_shape = reduction_shape(shape, &axes, true);
    let output_shape = reduction_shape(shape, &axes, keepdim);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(shape, input_dtype)?;
    extent(shape, accumulation)?; // mean input cast
    extent(&mean_shape, accumulation)?; // mean sum
    extent(&mean_shape, division_dtype)?; // mean reciprocal/multiply
    extent(&mean_shape, output_dtype)?; // mean output cast
    extent(shape, output_dtype)?; // centered and squared values
    extent(shape, accumulation)?; // variance numerator cast
    extent(&output_shape, accumulation)?; // variance sum
    extent(&output_shape, division_dtype)?; // variance reciprocal/multiply
    extent(&output_shape, output_dtype)?; // final cast

    let mean_divisor = TensorData::scalar_with_dtype(Scalar::F(count as f64), division_dtype);
    let divisor = TensorData::scalar_with_dtype(Scalar::F(denominator as f64), division_dtype);
    if mean_divisor.dtype() != division_dtype
        || divisor.dtype() != division_dtype
        || mean_shape.broadcast_with(mean_divisor.shape())? != mean_shape
        || output_shape.broadcast_with(divisor.shape())? != output_shape
        || input_dtype.promote(output_dtype) != output_dtype
        || accumulation.promote(division_dtype) != division_dtype
        || unary_dtype(UnaryOp::Square, output_dtype) != output_dtype
        || unary_dtype(UnaryOp::Reciprocal, division_dtype) != division_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "variance scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(VariancePlan {
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        mean_shape,
        output_shape,
        accumulation,
        division_dtype,
        output_dtype,
        mean_divisor,
        divisor,
    })
}

fn all_plan(
    input: NodeId,
    shape: &Shape,
    input_dtype: DType,
    axes: Option<Vec<isize>>,
    keepdim: bool,
) -> Result<AllPlan> {
    let axes = if shape.rank() == 0 {
        if axes.as_ref().is_some_and(|axes| {
            axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0))
        }) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes: vec![usize::MAX],
                rank: 0,
            });
        }
        Vec::new()
    } else {
        normalize_axes(input, shape.rank(), axes)?
    };
    let output_shape = reduction_shape(shape, &axes, keepdim);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(shape, input_dtype)?;
    extent(shape, DType::Bool)?; // bool cast
    extent(&output_shape, DType::Bool)?; // Product result
    Ok(AllPlan {
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        output_shape,
    })
}

fn any_plan(
    input: NodeId,
    shape: &Shape,
    input_dtype: DType,
    axes: Option<Vec<isize>>,
    keepdim: bool,
) -> Result<AnyPlan> {
    let axes = if shape.rank() == 0 {
        if axes.as_ref().is_some_and(|axes| {
            axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0))
        }) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes: vec![usize::MAX],
                rank: 0,
            });
        }
        Vec::new()
    } else {
        normalize_axes(input, shape.rank(), axes)?
    };
    let output_shape = reduction_shape(shape, &axes, keepdim);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(shape, input_dtype)?;
    extent(shape, DType::Bool)?; // bool cast
    extent(&output_shape, DType::Bool)?; // Max result or false identity
    let lowering = if axes.is_empty() {
        AnyLowering::Identity
    } else if output_shape.numel()? > 0 && axes.iter().any(|axis| shape.dims()[*axis] == 0) {
        AnyLowering::IdentityValue
    } else {
        AnyLowering::Reduce
    };
    Ok(AnyPlan {
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        output_shape,
        lowering,
    })
}

fn argmax_plan(
    input: NodeId,
    shape: &Shape,
    dtype: DType,
    axis: Option<isize>,
    keepdim: bool,
) -> Result<ArgmaxPlan> {
    let input_numel = shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let (flatten, work_shape, axis, keepdim) = match axis {
        None => (true, Shape::new([input_numel]), 0usize, false),
        Some(axis) => {
            if shape.rank() == 0 {
                return Err(Error::InvalidAxis {
                    node: input,
                    axis: usize::MAX,
                    rank: 0,
                });
            }
            let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
            (false, shape.clone(), axis, keepdim)
        }
    };
    work_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(work_shape.clone()))?;
    let output_shape = reduction_shape(&work_shape, &[axis], keepdim);
    let output_numel = output_shape.numel()?;
    output_numel
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
    let axis_extent = work_shape.dims()[axis];
    let axis_extent = i64::try_from(axis_extent)
        .map_err(|_| Error::ShapeOverflow(work_shape.clone()))?;
    let empty = axis_extent == 0 && output_numel > 0;
    let first_bounds = if axis_extent == 0 {
        Vec::new()
    } else {
        work_shape
            .dims()
            .iter()
            .enumerate()
            .map(|(dimension, &extent)| {
                if dimension == axis {
                    (0, 1)
                } else {
                    (0, extent)
                }
            })
            .collect()
    };
    if !empty {
        let first_shape = Shape::new(
            first_bounds
                .iter()
                .map(|(start, end)| end - start)
                .collect::<Vec<_>>(),
        );
        let first_result_shape = if keepdim {
            first_shape
        } else {
            Shape::new(
                first_shape
                    .dims()
                    .iter()
                    .enumerate()
                    .filter_map(|(dimension, &extent)| (dimension != axis).then_some(extent))
                    .collect::<Vec<_>>(),
            )
        };
        if first_result_shape != output_shape {
            return Err(Error::InvalidData {
                shape: output_shape.clone(),
                expected: output_numel,
                actual: first_result_shape.numel()?,
            });
        }
        first_result_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(first_result_shape.clone()))?;
        first_result_shape
            .numel()?
            .checked_mul(DType::Bool.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(first_result_shape.clone()))?;
    }
    let sentinel = TensorData::scalar_with_dtype(Scalar::I(axis_extent), DType::I32);
    if output_shape.broadcast_with(sentinel.shape())? != output_shape {
        return Err(Error::InvalidElementwiseDType {
            op: "argmax sentinel promotion",
            actual: DType::I32,
        });
    }
    let axis = isize::try_from(axis).map_err(|_| Error::ShapeOverflow(work_shape.clone()))?;
    Ok(ArgmaxPlan {
        flatten,
        work_shape,
        axis,
        keepdim,
        output_shape,
        first_bounds,
        sentinel,
        empty,
    })
}

fn argmin_plan(
    input: NodeId,
    shape: &Shape,
    dtype: DType,
    axis: Option<isize>,
    keepdim: bool,
) -> Result<ArgminPlan> {
    // Tensor.argmin is literally `self._inverse().argmax(...)`: floats are
    // negated, while Bool and integral storage are bitwise-inverted. Plan the
    // inversion before returning the ArgMax construction plan so the later
    // primitive sequence cannot expose a partial graph.
    let argmax = argmax_plan(input, shape, dtype, axis, keepdim)?;
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let inverse = if dtype.is_float() {
        ArgminInverse::Negate
    } else if dtype == DType::Bool {
        ArgminInverse::LogicalNot
    } else {
        let inverse = match dtype {
            DType::U8 => Scalar::U(u64::from(u8::MAX)),
            DType::U16 => Scalar::U(u64::from(u16::MAX)),
            DType::U32 => Scalar::U(u64::from(u32::MAX)),
            DType::U64 => Scalar::U(u64::MAX),
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(-1),
            DType::Bool | DType::F16 | DType::BF16 | DType::F32 | DType::F64 => unreachable!(),
        };
        let inverse = TensorData::scalar_with_dtype(inverse, dtype);
        if inverse.dtype() != dtype
            || shape.broadcast_with(inverse.shape())? != *shape
            || dtype.promote(inverse.dtype()) != dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "argmin inverse promotion",
                actual: dtype,
            });
        }
        ArgminInverse::BitwiseNot(inverse)
    };
    Ok(ArgminPlan { argmax, inverse })
}

fn mean_plan(
    input: NodeId,
    shape: &Shape,
    input_dtype: DType,
    axes: Option<Vec<isize>>,
    keepdim: bool,
) -> Result<MeanPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let axes = if shape.rank() == 0 {
        if axes.as_ref().is_some_and(|axes| {
            axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0))
        }) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes: vec![usize::MAX],
                rank: 0,
            });
        }
        Vec::new()
    } else {
        normalize_axes(input, shape.rank(), axes)?
    };
    let count = axes.iter().try_fold(1usize, |count, axis| {
        count
            .checked_mul(shape.dims()[*axis])
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    })?;
    let output_shape = reduction_shape(shape, &axes, keepdim);
    let sum_dtypes = ReductionDType::sum_default(input_dtype);
    // The source explicitly casts to Sum's accumulator then true-divides.
    // Integer accumulators are lifted to F32 by Tensor.div; floating
    // accumulators retain their concrete source width.
    let division_dtype = if sum_dtypes.accumulator.is_float() {
        sum_dtypes.accumulator
    } else {
        DType::F32
    };
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    extent(shape, input_dtype)?;
    extent(shape, sum_dtypes.accumulator)?;
    extent(&output_shape, sum_dtypes.accumulator)?;
    extent(&output_shape, division_dtype)?;
    extent(&output_shape, division_dtype)?; // reciprocal/multiply result
    extent(&output_shape, output_dtype)?;
    let divisor = TensorData::scalar_with_dtype(Scalar::F(count as f64), division_dtype);
    if divisor.dtype() != division_dtype
        || output_shape.broadcast_with(divisor.shape())? != output_shape
        || division_dtype.promote(divisor.dtype()) != division_dtype
        || division_dtype.promote(division_dtype) != division_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "mean scalar promotion",
            actual: division_dtype,
        });
    }
    Ok(MeanPlan {
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        output_shape,
        sum_dtypes: ReductionDType::new(sum_dtypes.accumulator, sum_dtypes.accumulator),
        division_dtype,
        output_dtype,
        divisor,
    })
}

impl Graph {
    /// Runs a Sum or Product through an explicit, source-validated
    /// accumulator/output dtype contract.
    ///
    /// The whole contract, axis list, and source extent are validated before
    /// the accumulator cast, reduction, or final narrowing cast is appended.
    pub fn reduce_with_dtypes(
        &mut self,
        input: NodeId,
        kind: ReduceKind,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        dtypes: ReductionDType,
    ) -> Result<NodeId> {
        let (shape, input_dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        let axes = normalize_axes(input, shape.rank(), axes)?;
        if !valid_reduction_dtypes(kind, input_dtype, dtypes) {
            return Err(Error::InvalidElementwiseDType {
                op: "reduce_with_dtypes",
                actual: dtypes.accumulator,
            });
        }
        let accumulator = if input_dtype == dtypes.accumulator {
            input
        } else {
            self.cast(input, dtypes.accumulator)?
        };
        let reduced = self.reduce(
            accumulator,
            kind,
            Some(axes.into_iter().map(|axis| axis as isize).collect()),
            keepdim,
        )?;
        if dtypes.output == dtypes.accumulator {
            Ok(reduced)
        } else {
            self.cast(reduced, dtypes.output)
        }
    }

    fn boolean_reduction_input(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
    ) -> Result<(NodeId, Vec<usize>)> {
        let (rank, dtype) = {
            let source = self.node(input)?;
            (source.shape.rank(), source.dtype)
        };
        let axes = normalize_axes(input, rank, axes)?;
        let boolean = if dtype == DType::Bool {
            input
        } else {
            self.cast(input, DType::Bool)?
        };
        Ok((boolean, axes))
    }

    /// Product reduction over optional signed axes.
    pub fn prod(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        self.reduce(input, crate::ReduceKind::Product, axes, keepdim)
    }

    /// Cumulative sum along one signed axis.
    ///
    /// This is the checked static composition used by tinygrad's `cumsum`:
    /// each inclusive prefix is reduced with tinygrad's default Sum
    /// accumulator/output contract, then the prefix results are concatenated.
    /// Axis, source extent, every prefix bound, and the empty/scalar result
    /// dtype are all resolved before the first movement or reduction node.
    pub fn cumsum(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        let dtypes = ReductionDType::sum_default(dtype);
        if shape.rank() == 0 {
            if !matches!(axis, -1 | 0) {
                return Err(Error::InvalidAxis {
                    node: input,
                    axis: usize::MAX,
                    rank: 0,
                });
            }
            return self.cast(input, dtypes.output);
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        if shape.dims().contains(&0) {
            return self.cast(input, dtypes.output);
        }

        let prefixes = (0..shape.dims()[axis])
            .map(|end| {
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &extent)| {
                        if dimension == axis {
                            (0, end + 1)
                        } else {
                            (0, extent)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let values = prefixes
            .into_iter()
            .map(|bounds| {
                let prefix = self.shrink(input, bounds)?;
                self.reduce_with_dtypes(
                    prefix,
                    crate::ReduceKind::Sum,
                    Some(vec![axis as isize]),
                    true,
                    dtypes,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.concat(values, axis)
    }

    /// Cumulative product along one signed axis.
    ///
    /// This is the checked static composition used by tinygrad's `cumprod`:
    /// each inclusive prefix is reduced with tinygrad's default Product
    /// accumulator/output contract, then the prefix results are concatenated.
    /// Axis, source extent, and every prefix bound are resolved before the
    /// first movement or reduction node.
    pub fn cumprod(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        let dtypes = ReductionDType::product_default(dtype);
        if shape.rank() == 0 {
            if !matches!(axis, -1 | 0) {
                return Err(Error::InvalidAxis {
                    node: input,
                    axis: usize::MAX,
                    rank: 0,
                });
            }
            return Ok(input);
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        if shape.dims().contains(&0) {
            return Ok(input);
        }

        let prefixes = (0..shape.dims()[axis])
            .map(|end| {
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &extent)| {
                        if dimension == axis {
                            (0, end + 1)
                        } else {
                            (0, extent)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let values = prefixes
            .into_iter()
            .map(|bounds| {
                let prefix = self.shrink(input, bounds)?;
                self.reduce_with_dtypes(
                    prefix,
                    crate::ReduceKind::Product,
                    Some(vec![axis as isize]),
                    true,
                    dtypes,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.concat(values, axis)
    }

    /// Boolean all-reduction over optional signed axes.
    ///
    /// This is the checked `bool().prod(...)` composition used by tinygrad:
    /// every nonzero value (including NaN) is true, and an empty product is
    /// true. Axis normalization completes before a cast or reduction node is
    /// appended.
    pub fn all(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let (shape, dtype) = {
            let input_node = self.node(input)?;
            (input_node.shape.clone(), input_node.dtype)
        };
        let plan = all_plan(input, &shape, dtype, axes, keepdim)?;
        let boolean = if dtype == DType::Bool {
            input
        } else {
            self.cast(input, DType::Bool)?
        };
        let output = self.reduce(
            boolean,
            crate::ReduceKind::Product,
            Some(plan.axes),
            keepdim,
        )?;
        debug_assert_eq!(self.shape(output).expect("all preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("all preflighted"), DType::Bool);
        Ok(output)
    }

    /// Boolean any-reduction over optional signed axes.
    ///
    /// This is tinygrad's literal `bool().max(...)` composition. A populated
    /// output with an empty reduced domain receives its typed false identity
    /// before raw Max would reject the empty domain.
    pub fn any(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let (shape, dtype) = {
            let input_node = self.node(input)?;
            (input_node.shape.clone(), input_node.dtype)
        };
        let plan = any_plan(input, &shape, dtype, axes, keepdim)?;
        let boolean = if dtype == DType::Bool {
            input
        } else {
            self.cast(input, DType::Bool)?
        };
        let output = match plan.lowering {
            AnyLowering::Identity => boolean,
            AnyLowering::IdentityValue => {
                self.full_with_dtype(plan.output_shape.clone(), Scalar::Bool(false), DType::Bool)?
            }
            AnyLowering::Reduce => self.reduce(
                boolean,
                crate::ReduceKind::Max,
                Some(plan.axes),
                keepdim,
            )?,
        };
        debug_assert_eq!(self.shape(output).expect("any preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("any preflighted"), DType::Bool);
        Ok(output)
    }

    /// Source-faithful public tinygrad-style ArgMax.
    ///
    /// `None` flattens and ignores `keepdim`; an explicit axis uses the
    /// first-tie ArgReduce path with tinygrad's leading-NaN and empty-axis
    /// sentinels. The legacy [`Self::argmax`] remains the raw ArgReduce API.
    pub fn argmax_with_axis(
        &mut self,
        input: NodeId,
        axis: Option<isize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let (shape, dtype) = {
            let input_node = self.node(input)?;
            (input_node.shape.clone(), input_node.dtype)
        };
        let plan = argmax_plan(input, &shape, dtype, axis, keepdim)?;
        let output = if plan.empty {
            self.full_with_dtype(plan.output_shape.clone(), Scalar::I(i32::MIN.into()), DType::I32)?
        } else {
            let source = if plan.flatten {
                self.reshape(input, plan.work_shape.clone())?
            } else {
                input
            };
            let indices = self.argmax(source, Some(plan.axis), plan.keepdim)?;
            let first = self.shrink(source, plan.first_bounds)?;
            let first = if plan.keepdim {
                first
            } else {
                self.squeeze(first, Some(plan.axis))?
            };
            let leading_nan = self.isnan(first)?;
            let sentinel = self.constant(plan.sentinel);
            self.select(leading_nan, sentinel, indices)?
        };
        debug_assert_eq!(self.shape(output).expect("argmax preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("argmax preflighted"), DType::I32);
        Ok(output)
    }

    /// Source-faithful public tinygrad-style ArgMin.
    ///
    /// tinygrad defines this as `inverse(input).argmax(...)`: floats negate,
    /// Bool logically negates, and integer storage is bitwise-inverted before
    /// the first-tie ArgMax path. `None` flattens and ignores `keepdim`.
    /// The legacy [`Self::argmin`] remains the raw ArgReduce API.
    pub fn argmin_with_axis(
        &mut self,
        input: NodeId,
        axis: Option<isize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let (shape, dtype) = {
            let input_node = self.node(input)?;
            (input_node.shape.clone(), input_node.dtype)
        };
        let plan = argmin_plan(input, &shape, dtype, axis, keepdim)?;
        let output = if plan.argmax.empty {
            self.full_with_dtype(
                plan.argmax.output_shape.clone(),
                Scalar::I(i32::MIN.into()),
                DType::I32,
            )?
        } else {
            let inverse = match plan.inverse {
                ArgminInverse::Negate => self.neg(input)?,
                ArgminInverse::LogicalNot => self.logical_not(input)?,
                ArgminInverse::BitwiseNot(value) => {
                    let value = self.constant(value);
                    self.bit_xor(input, value)?
                }
            };
            let source = if plan.argmax.flatten {
                self.reshape(inverse, plan.argmax.work_shape.clone())?
            } else {
                inverse
            };
            let indices = self.argmax(source, Some(plan.argmax.axis), plan.argmax.keepdim)?;
            let first = self.shrink(source, plan.argmax.first_bounds)?;
            let first = if plan.argmax.keepdim {
                first
            } else {
                self.squeeze(first, Some(plan.argmax.axis))?
            };
            let leading_nan = self.isnan(first)?;
            let sentinel = self.constant(plan.argmax.sentinel);
            self.select(leading_nan, sentinel, indices)?
        };
        debug_assert_eq!(self.shape(output).expect("argmin preflighted"), &plan.argmax.output_shape);
        debug_assert_eq!(self.dtype(output).expect("argmin preflighted"), DType::I32);
        Ok(output)
    }

    /// Reduces multiple axes. Axes refer to the original input rank.
    pub fn sum_axes(
        &mut self,
        input: NodeId,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        let mut axes = axes.into_iter().collect::<Vec<_>>();
        axes.sort_unstable();
        if axes.iter().any(|axis| *axis >= rank) || axes.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes,
                rank,
            });
        }

        let mut output = input;
        for axis in axes.into_iter().rev() {
            output = self.sum(output, axis)?;
        }
        Ok(output)
    }

    pub fn sum_all(&mut self, input: NodeId) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        self.sum_axes(input, 0..rank)
    }

    pub fn mean(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let count = *shape.dims().get(axis).ok_or(Error::InvalidAxis {
            node: input,
            axis,
            rank: shape.rank(),
        })?;
        let sum = self.sum(input, axis)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }

    pub fn mean_axes(
        &mut self,
        input: NodeId,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let mut axes = axes.into_iter().collect::<Vec<_>>();
        axes.sort_unstable();
        if axes.iter().any(|axis| *axis >= shape.rank())
            || axes.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes,
                rank: shape.rank(),
            });
        }
        let count = axes
            .iter()
            .try_fold(1usize, |count, axis| count.checked_mul(shape.dims()[*axis]))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let sum = self.sum_axes(input, axes)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }

    pub fn mean_all(&mut self, input: NodeId) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let count = shape.numel()?;
        let sum = self.sum_all(input)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }

    /// Source-faithful public tinygrad-style mean over signed optional axes.
    ///
    /// Unlike the legacy single-axis conveniences, this accepts `None` for
    /// all axes, a signed axis list, and a keepdim result. It preflights the
    /// entire cast → typed Sum → reciprocal/multiply → final cast sequence
    /// before adding a graph node or scalar constant.
    pub fn mean_with_axes(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = mean_plan(input, &input_node.shape, input_node.dtype, axes, keepdim)?;
        let sum = self.reduce_with_dtypes(
            input,
            ReduceKind::Sum,
            Some(plan.axes),
            keepdim,
            plan.sum_dtypes,
        )?;
        let numerator = if self.dtype(sum)? == plan.division_dtype {
            sum
        } else {
            self.cast(sum, plan.division_dtype)?
        };
        let divisor = self.constant(plan.divisor);
        let reciprocal = self.reciprocal(divisor)?;
        let divided = self.mul(numerator, reciprocal)?;
        let output = if plan.output_dtype == plan.division_dtype {
            divided
        } else {
            self.cast(divided, plan.output_dtype)?
        };
        debug_assert_eq!(self.shape(output).expect("Mean preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("Mean preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.mean()` defaults: all axes, with reduced
    /// dimensions omitted.
    pub fn mean_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.mean_with_axes(input, None, false)
    }

    /// Source-faithful public tinygrad-style Max over signed optional axes.
    /// It preserves dtype and uses a typed dtype-min constant only for a
    /// populated output whose reduced domain is empty.
    pub fn max_with_axes(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = max_plan(input, &input_node.shape, input_node.dtype, axes, keepdim)?;
        let output = match plan.lowering {
            MaxLowering::Identity => input,
            MaxLowering::IdentityValue(value) => {
                self.full_with_dtype(plan.output_shape.clone(), value, plan.dtype)?
            }
            MaxLowering::Reduce => self.reduce(input, ReduceKind::Max, Some(plan.axes), keepdim)?,
        };
        debug_assert_eq!(self.shape(output).expect("Max preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("Max preflighted"), plan.dtype);
        Ok(output)
    }

    /// Source-faithful public tinygrad-style Min over signed optional axes.
    /// tinygrad spells Min as inverse-Max-inverse; the shared typed Min
    /// reducer preserves the same first candidate, NaN, and tie semantics.
    pub fn min_with_axes(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = min_plan(input, &input_node.shape, input_node.dtype, axes, keepdim)?;
        let output = match plan.lowering {
            MaxLowering::Identity => input,
            MaxLowering::IdentityValue(value) => {
                self.full_with_dtype(plan.output_shape.clone(), value, plan.dtype)?
            }
            MaxLowering::Reduce => self.reduce(input, ReduceKind::Min, Some(plan.axes), keepdim)?,
        };
        debug_assert_eq!(self.shape(output).expect("Min preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("Min preflighted"), plan.dtype);
        Ok(output)
    }

    /// Variance with tinygrad's signed `correction` contract.
    ///
    /// The numerator is accumulated through the source dtype's Sum
    /// accumulator, while the public result is the floating input dtype (or
    /// F32 for nonfloating inputs).  As in tinygrad, the denominator is
    /// `max(n - correction, 0)`, including for empty reductions.
    pub fn var(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: Option<VarianceCorrection>,
    ) -> Result<NodeId> {
        let (shape, input_dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let plan = variance_plan(input, &shape, input_dtype, axes, keepdim, correction)?;
        let accumulation_contract = ReductionDType::new(plan.accumulation, plan.accumulation);

        // tinygrad literally builds `self - self.mean(keepdim=true)`, squares
        // that source-width result, then casts only the squares for its Sum.
        // The plan fixes both divisors at the division width, so F16/BF16
        // counts are rounded by F32 reciprocal/multiply rather than output
        // storage.
        let mean_sum = self.reduce_with_dtypes(
            input,
            ReduceKind::Sum,
            Some(plan.axes.clone()),
            true,
            accumulation_contract,
        )?;
        let mean_numerator = if self.dtype(mean_sum)? == plan.division_dtype {
            mean_sum
        } else {
            self.cast(mean_sum, plan.division_dtype)?
        };
        let mean_divisor = self.constant(plan.mean_divisor);
        let mean_reciprocal = self.reciprocal(mean_divisor)?;
        let mean = self.mul(mean_numerator, mean_reciprocal)?;
        let mean = if self.dtype(mean)? == plan.output_dtype {
            mean
        } else {
            self.cast(mean, plan.output_dtype)?
        };
        let deviations = self.sub(input, mean)?;
        let squares = self.square(deviations)?;
        let numerator = self.reduce_with_dtypes(
            squares,
            ReduceKind::Sum,
            Some(plan.axes),
            keepdim,
            accumulation_contract,
        )?;
        let variance_numerator = if self.dtype(numerator)? == plan.division_dtype {
            numerator
        } else {
            self.cast(numerator, plan.division_dtype)?
        };
        let divisor = self.constant(plan.divisor);
        let reciprocal = self.reciprocal(divisor)?;
        let variance = self.mul(variance_numerator, reciprocal)?;
        let output = if self.dtype(variance)? == plan.output_dtype {
            variance
        } else {
            self.cast(variance, plan.output_dtype)?
        };
        debug_assert_eq!(self.shape(output).expect("variance preflighted"), &plan.output_shape);
        debug_assert_eq!(self.shape(mean).expect("variance preflighted"), &plan.mean_shape);
        debug_assert_eq!(self.dtype(output).expect("variance preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.var()` defaults. `None` maps to the
    /// source correction default, `VarianceCorrection::UNBIASED` (one).
    pub fn var_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.var(input, None, false, None)
    }

    /// Returns tinygrad's literal `(var(...), mean(...))` pair.
    ///
    /// Both independently observable result descriptors and every reduction
    /// plan are checked before either branch can append its constants or
    /// nodes.  This intentionally does not share the variance helper's
    /// internal mean: checked-in tinygrad spells `var_mean` as two public
    /// calls, so the mean result retains its own source-typed reduction path.
    pub fn var_mean(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: Option<VarianceCorrection>,
    ) -> Result<(NodeId, NodeId)> {
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let variance_plan = variance_plan(
            input,
            &shape,
            dtype,
            axes.clone(),
            keepdim,
            correction,
        )?;
        let mean_plan = mean_plan(input, &shape, dtype, axes.clone(), keepdim)?;

        let variance = self.var(input, axes.clone(), keepdim, correction)?;
        let mean = self.mean_with_axes(input, axes, keepdim)?;
        debug_assert_eq!(
            self.shape(variance).expect("var_mean preflighted"),
            &variance_plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(variance).expect("var_mean preflighted"),
            variance_plan.output_dtype
        );
        debug_assert_eq!(
            self.shape(mean).expect("var_mean preflighted"),
            &mean_plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(mean).expect("var_mean preflighted"),
            mean_plan.output_dtype
        );
        Ok((variance, mean))
    }

    /// Checked-in tinygrad's `Tensor.var_mean()` defaults, retaining its
    /// observable `(variance, mean)` result order.
    pub fn var_mean_default(&mut self, input: NodeId) -> Result<(NodeId, NodeId)> {
        self.var_mean(input, None, false, None)
    }

    /// Standard deviation, defined exactly as `sqrt(var(...))`.
    pub fn std(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: Option<VarianceCorrection>,
    ) -> Result<NodeId> {
        let variance = self.var(input, axes, keepdim, correction)?;
        self.sqrt(variance)
    }

    /// Checked-in tinygrad's `Tensor.std()` defaults.
    pub fn std_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.std(input, None, false, None)
    }

    /// Returns tinygrad's literal `(std(...), mean(...))` pair.
    ///
    /// The standard-deviation branch remains `var(...).sqrt()` and the mean
    /// remains a separate public reduction, exactly as in tinygrad.  Planning
    /// both branches first keeps malformed axes, descriptors, and byte facts
    /// from publishing a partial pair.
    pub fn std_mean(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: Option<VarianceCorrection>,
    ) -> Result<(NodeId, NodeId)> {
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let variance_plan = variance_plan(
            input,
            &shape,
            dtype,
            axes.clone(),
            keepdim,
            correction,
        )?;
        let mean_plan = mean_plan(input, &shape, dtype, axes.clone(), keepdim)?;

        // `variance_plan` establishes a floating result descriptor.  The
        // public sqrt helper is homogeneous for that descriptor and validates
        // its same-shape, same-dtype output before it appends its raw unary.
        debug_assert!(variance_plan.output_dtype.is_float());
        let standard_deviation = self.std(input, axes.clone(), keepdim, correction)?;
        let mean = self.mean_with_axes(input, axes, keepdim)?;
        debug_assert_eq!(
            self.shape(standard_deviation).expect("std_mean preflighted"),
            &variance_plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(standard_deviation).expect("std_mean preflighted"),
            variance_plan.output_dtype
        );
        debug_assert_eq!(
            self.shape(mean).expect("std_mean preflighted"),
            &mean_plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(mean).expect("std_mean preflighted"),
            mean_plan.output_dtype
        );
        Ok((standard_deviation, mean))
    }

    /// Checked-in tinygrad's `Tensor.std_mean()` defaults, retaining its
    /// observable `(standard_deviation, mean)` result order.
    pub fn std_mean_default(&mut self, input: NodeId) -> Result<(NodeId, NodeId)> {
        self.std_mean(input, None, false, None)
    }

    /// L2-normalizes a floating tensor along one signed axis.
    ///
    /// This is the closed `p = 2` form of tinygrad's public `normalize`
    /// composition: `x / max(sqrt(sum(abs(x)^2, keepdim=true)), eps)`. The
    /// axis, source extent, and dtype are all validated before any cast,
    /// reduction, constant, or elementwise node is appended. `eps` remains a
    /// plain floating scalar because tinygrad applies no finiteness or sign
    /// validation before its `maximum` composition.
    pub fn normalize_l2(&mut self, input: NodeId, axis: isize, eps: f64) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        if !dtype.is_float() {
            return Err(Error::InvalidElementwiseDType {
                op: "normalize_l2",
                actual: dtype,
            });
        }
        shape.numel()?;
        let axes = normalize_axes(input, shape.rank(), Some(vec![axis]))?;
        let accumulation = ReductionDType::sum_default(dtype);
        let normalized_axes = Some(axes.into_iter().map(|axis| axis as isize).collect());

        // For real dtypes, `square` is the exact closed p=2 instance of
        // tinygrad's `abs().pow(2.0)` composition, including its normal
        // floating nonfinite propagation.
        let squares = self.square(input)?;
        let sum = self.reduce_with_dtypes(
            squares,
            ReduceKind::Sum,
            normalized_axes,
            true,
            accumulation,
        )?;
        let magnitude = self.sqrt(sum)?;
        let epsilon = self.constant(TensorData::scalar_with_dtype(Scalar::F(eps), dtype));
        let denominator = self.maximum(magnitude, epsilon)?;
        self.div(input, denominator)
    }
}

fn variance_denominator(
    count: usize,
    correction: VarianceCorrection,
    shape: &crate::Shape,
) -> Result<usize> {
    let correction = correction.value();
    if correction >= 0 {
        let correction = usize::try_from(correction).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        Ok(count.saturating_sub(correction))
    } else {
        let magnitude = usize::try_from(correction.unsigned_abs())
            .map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        count
            .checked_add(magnitude)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    }
}

fn valid_reduction_dtypes(kind: ReduceKind, input: DType, dtypes: ReductionDType) -> bool {
    match kind {
        ReduceKind::Sum => {
            dtypes == ReductionDType::sum_default(input)
                || dtypes.accumulator == dtypes.output
        }
        ReduceKind::Product => dtypes.accumulator == dtypes.output,
        ReduceKind::Mean | ReduceKind::Max | ReduceKind::Min => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Shape};
    use std::collections::HashMap;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn multi_axis_sum_and_mean_are_composable_and_differentiable() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2, 2]);
        let sum = graph.sum_axes(x, [0, 2]).unwrap();
        let mean = graph.mean_all(x).unwrap();
        let dx = graph.grad(mean, x).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            data([2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]),
        )]);

        assert_eq!(
            CpuBackend.execute(&graph, sum, &inputs).unwrap(),
            data([2], &[14., 22.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, mean, &inputs).unwrap(),
            TensorData::scalar(4.5)
        );
        assert_eq!(
            CpuBackend.execute(&graph, dx, &inputs).unwrap(),
            data([2, 2, 2], &[0.125; 8])
        );
    }

    #[test]
    fn rejects_duplicate_reduction_axes() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        assert_eq!(
            graph.sum_axes(x, [1, 1]),
            Err(Error::InvalidReductionAxes {
                node: x,
                axes: vec![1, 1],
                rank: 2
            })
        );
    }

    #[test]
    fn mean_all_preflights_total_extent_before_sum_lowering() {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [usize::MAX, 2]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.mean_all(input),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut valid = Graph::new();
        let input = valid.input("input", [2]);
        let output = valid.mean_all(input).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &valid,
                    output,
                    &HashMap::from([("input".into(), data([2], &[2., 6.]))]),
                )
                .unwrap(),
            TensorData::scalar(4.)
        );
    }

    #[test]
    fn generalized_reductions_normalize_axes_keep_dimensions_and_arg_ties() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let sum = graph
            .reduce(x, crate::ReduceKind::Sum, Some(vec![-1]), true)
            .unwrap();
        let product = graph
            .reduce(x, crate::ReduceKind::Product, Some(vec![0]), false)
            .unwrap();
        let maximum = graph
            .reduce(x, crate::ReduceKind::Max, None, false)
            .unwrap();
        let minimum = graph
            .reduce(x, crate::ReduceKind::Min, Some(vec![1]), false)
            .unwrap();
        let argmax = graph.argmax(x, Some(-1), false).unwrap();
        let inputs = HashMap::from([("x".into(), data([2, 3], &[1., 3., 3., 2., 0., -1.]))]);
        assert_eq!(
            CpuBackend.execute(&graph, sum, &inputs).unwrap(),
            data([2, 1], &[7., 1.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, product, &inputs).unwrap(),
            data([3], &[2., 0., -3.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, maximum, &inputs).unwrap(),
            TensorData::scalar(3.)
        );
        assert_eq!(
            CpuBackend.execute(&graph, minimum, &inputs).unwrap(),
            data([2], &[1., -1.])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, argmax, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I32(vec![1, 0])
        );
        assert!(graph.trace(argmax).unwrap().to_string().contains("argmax"));
    }

    #[test]
    fn prod_forwards_signed_axes_to_validated_product_reduction() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let original_nodes = graph.node_count();
        assert!(matches!(
            graph.prod(input, Some(vec![-1, 1]), false),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(graph.node_count(), original_nodes);

        let product = graph.prod(input, Some(vec![0, -1]), true).unwrap();
        let gradient = graph.grad(product, input).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            data([2, 3], &[1., 2., 3., 4., 5., 6.]),
        )]);
        assert_eq!(graph.shape(product).unwrap(), &Shape::new([1, 1]));
        assert_eq!(
            CpuBackend.execute(&graph, product, &bindings).unwrap(),
            data([1, 1], &[720.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            data([2, 3], &[720., 360., 240., 180., 144., 120.])
        );

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [2, 0]);
        let reduced = empty_graph.prod(empty, Some(vec![-1]), false).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &empty_graph,
                    reduced,
                    &HashMap::from([("empty".into(), data([2, 0], &[]))]),
                )
                .unwrap(),
            data([2], &[1., 1.])
        );
    }

    #[test]
    fn all_is_boolean_nondifferentiable_and_preflights_axes() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let original_nodes = graph.node_count();
        assert!(matches!(
            graph.all(input, Some(vec![-1, 1]), false),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(graph.node_count(), original_nodes);

        let all = graph.all(input, Some(vec![0, -1]), true).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            data([2, 2], &[1., -2., f32::NAN, 4.]),
        )]);
        let output = CpuBackend.execute(&graph, all, &bindings).unwrap();
        assert_eq!(graph.shape(all).unwrap(), &Shape::new([1, 1]));
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![1.]);
        assert!(matches!(graph.grad(all, input), Err(Error::NoGradient(_))));

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [2, 0]);
        let reduced = empty_graph.all(empty, Some(vec![-1]), false).unwrap();
        let output = CpuBackend
            .execute(
                &empty_graph,
                reduced,
                &HashMap::from([("empty".into(), data([2, 0], &[]))]),
            )
            .unwrap();
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![1., 1.]);

        let mut special = Graph::new();
        let input = special.input_dtype("input", [2, 2], DType::F64);
        let reduced = special.all(input, Some(vec![-1]), false).unwrap();
        let output = CpuBackend
            .execute(
                &special,
                reduced,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars(
                        [2, 2],
                        DType::F64,
                        [
                            Scalar::F(-0.0),
                            Scalar::F(f64::NAN),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(1.0),
                        ],
                    )
                    .unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(output.to_vec_f64(), vec![0., 1.]);

        for dtype in [
            DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
            DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32,
            DType::F64,
        ] {
            let mut typed = Graph::new();
            let input = typed.input_dtype("input", [], dtype);
            let output = typed.all(input, None, false).unwrap();
            assert_eq!(typed.dtype(output).unwrap(), DType::Bool);
        }

        let mut scalar = Graph::new();
        let input = scalar.input("input", []);
        let negative_axis = scalar.all(input, Some(vec![-1]), false).unwrap();
        let zero_axis = scalar.all(input, Some(vec![0]), false).unwrap();
        assert_eq!(scalar.shape(negative_axis).unwrap(), &Shape::new([]));
        assert_eq!(scalar.shape(zero_axis).unwrap(), &Shape::new([]));

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 2]);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.all(input, None, false),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn any_is_boolean_nondifferentiable_and_preflights_axes() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let original_nodes = graph.node_count();
        assert!(matches!(
            graph.any(input, Some(vec![-1, 1]), false),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(graph.node_count(), original_nodes);

        let any = graph.any(input, Some(vec![0, -1]), true).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            data([2, 2], &[0., 0., f32::NAN, 0.]),
        )]);
        let output = CpuBackend.execute(&graph, any, &bindings).unwrap();
        assert_eq!(graph.shape(any).unwrap(), &Shape::new([1, 1]));
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![1.]);
        assert!(matches!(graph.grad(any, input), Err(Error::NoGradient(_))));

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [2, 0]);
        let reduced = empty_graph.any(empty, Some(vec![-1]), false).unwrap();
        let output = CpuBackend
            .execute(
                &empty_graph,
                reduced,
                &HashMap::from([("empty".into(), data([2, 0], &[]))]),
            )
            .unwrap();
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![0., 0.]);

        let mut special = Graph::new();
        let input = special.input_dtype("input", [2, 2], DType::F64);
        let reduced = special.any(input, Some(vec![-1]), false).unwrap();
        assert!(special.nodes.iter().any(|node| {
            matches!(
                &node.op,
                crate::Op::Reduce {
                    kind: ReduceKind::Max,
                    ..
                }
            )
        }));
        let output = CpuBackend
            .execute(
                &special,
                reduced,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars(
                        [2, 2],
                        DType::F64,
                        [
                            Scalar::F(-0.0),
                            Scalar::F(0.0),
                            Scalar::F(f64::NAN),
                            Scalar::F(f64::INFINITY),
                        ],
                    )
                    .unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(output.to_vec_f64(), vec![0., 1.]);

        for dtype in [
            DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
            DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32, DType::F64,
        ] {
            let mut typed = Graph::new();
            let input = typed.input_dtype("input", [], dtype);
            let output = typed.any(input, None, false).unwrap();
            assert_eq!(typed.dtype(output).unwrap(), DType::Bool);
        }

        let mut scalar = Graph::new();
        let input = scalar.input("input", []);
        let negative_axis = scalar.any(input, Some(vec![-1]), false).unwrap();
        let zero_axis = scalar.any(input, Some(vec![0]), false).unwrap();
        assert_eq!(scalar.shape(negative_axis).unwrap(), &Shape::new([]));
        assert_eq!(scalar.shape(zero_axis).unwrap(), &Shape::new([]));

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 2]);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.any(input, None, false),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn argmax_with_axis_matches_tinygrad_flatten_and_nan_sentinels() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3, 3], DType::F64);
        let output = graph.argmax_with_axis(input, Some(-1), false).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            TensorData::from_scalars(
                [3, 3],
                DType::F64,
                [
                    Scalar::F(f64::NAN), Scalar::F(2.0), Scalar::F(3.0),
                    Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(-1.0),
                    Scalar::F(1.0), Scalar::F(f64::NAN), Scalar::F(3.0),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.to_vec_f64(), vec![3., 0., 2.]);
        assert!(matches!(graph.grad(output, input), Err(Error::NoGradient(_))));

        let mut flattened = Graph::new();
        let input = flattened.input("input", [2, 2]);
        let output = flattened.argmax_with_axis(input, None, true).unwrap();
        assert_eq!(flattened.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(flattened.dtype(output).unwrap(), DType::I32);
        let values = CpuBackend
            .execute(
                &flattened,
                output,
                &HashMap::from([("input".into(), data([2, 2], &[1., 4., 4., 0.]))]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![1.]);

        let mut scalar = Graph::new();
        let input = scalar.input("input", []);
        let output = scalar.argmax_with_axis(input, None, false).unwrap();
        assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(scalar.dtype(output).unwrap(), DType::I32);
        let values = CpuBackend
            .execute(
                &scalar,
                output,
                &HashMap::from([("input".into(), data([], &[7.]))]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![0.]);

        let mut wide = Graph::new();
        let input = wide.input_dtype("input", [2], DType::U64);
        let output = wide.argmax_with_axis(input, Some(0), false).unwrap();
        let values = CpuBackend
            .execute(
                &wide,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::U64,
                        [Scalar::U(1_u64 << 53), Scalar::U((1_u64 << 53) + 1)],
                    )
                    .unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![1.]);

        let mut empty = Graph::new();
        let input = empty.input("input", [2, 0]);
        let output = empty.argmax_with_axis(input, Some(1), false).unwrap();
        let values = CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), data([2, 0], &[]))]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![i32::MIN as f64; 2]);

        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed.argmax_with_axis(input, Some(-3), false).is_err());
        assert_eq!(malformed.node_count(), nodes);
    }

    #[test]
    fn argmin_with_axis_uses_tinygrad_inverse_and_argmax_sentinels() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3, 3], DType::F64);
        let output = graph.argmin_with_axis(input, Some(-1), false).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            TensorData::from_scalars(
                [3, 3],
                DType::F64,
                [
                    Scalar::F(f64::NAN), Scalar::F(2.0), Scalar::F(-3.0),
                    Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(1.0),
                    Scalar::F(3.0), Scalar::F(f64::NAN), Scalar::F(-1.0),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.to_vec_f64(), vec![3., 0., 2.]);
        assert!(matches!(graph.grad(output, input), Err(Error::NoGradient(_))));

        let mut flattened = Graph::new();
        let input = flattened.input("input", [2, 2]);
        let output = flattened.argmin_with_axis(input, None, true).unwrap();
        assert_eq!(flattened.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(flattened.dtype(output).unwrap(), DType::I32);
        let values = CpuBackend
            .execute(
                &flattened,
                output,
                &HashMap::from([("input".into(), data([2, 2], &[1., -4., -4., 0.]))]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![1.]);

        let mut scalar = Graph::new();
        let input = scalar.input("input", []);
        let output = scalar.argmin_with_axis(input, None, false).unwrap();
        assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(scalar.dtype(output).unwrap(), DType::I32);
        let values = CpuBackend
            .execute(
                &scalar,
                output,
                &HashMap::from([("input".into(), data([], &[7.]))]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![0.]);

        let mut boolean = Graph::new();
        let input = boolean.input_dtype("input", [2], DType::Bool);
        let output = boolean.argmin_with_axis(input, Some(0), false).unwrap();
        let values = CpuBackend
            .execute(
                &boolean,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(false)],
                    )
                    .unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![1.]);

        let mut integer = Graph::new();
        let input = integer.input_dtype("input", [2], DType::I64);
        let output = integer.argmin_with_axis(input, Some(0), false).unwrap();
        let values = CpuBackend
            .execute(
                &integer,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::I64,
                        [Scalar::I(i64::MIN), Scalar::I(-1)],
                    )
                    .unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![0.]);

        let mut unsigned = Graph::new();
        let input = unsigned.input_dtype("input", [2], DType::U64);
        let output = unsigned.argmin_with_axis(input, Some(0), false).unwrap();
        let values = CpuBackend
            .execute(
                &unsigned,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::U64,
                        [Scalar::U(1_u64 << 53), Scalar::U((1_u64 << 53) + 1)],
                    )
                    .unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![0.]);

        let mut empty = Graph::new();
        let input = empty.input("input", [2, 0]);
        let output = empty.argmin_with_axis(input, Some(1), false).unwrap();
        let values = CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), data([2, 0], &[]))]),
            )
            .unwrap();
        assert_eq!(values.to_vec_f64(), vec![i32::MIN as f64; 2]);

        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed.argmin_with_axis(input, Some(-3), false).is_err());
        assert_eq!(malformed.node_count(), nodes);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 2]);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.argmin_with_axis(input, None, false),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn cumsum_matches_tinygrad_signed_axis_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let cumulative = graph.cumsum(input, -1).unwrap();
        let loss = graph.sum_all(cumulative).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([(
            "input".into(),
            TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, cumulative, &inputs).unwrap().to_vec_f64(),
            vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs).unwrap().to_vec_f64(),
            vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0]
        );

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::I8);
        let cumulative = scalar.cumsum(input, 0).unwrap();
        assert_eq!(scalar.dtype(cumulative).unwrap(), DType::I32);
        assert_eq!(
            CpuBackend
                .execute(
                    &scalar,
                    cumulative,
                    &HashMap::from([(
                        "input".into(),
                        TensorData::from_scalars([], DType::I8, [Scalar::I(5)]).unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![5.0]
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0], DType::F16);
        let cumulative = empty.cumsum(input, -1).unwrap();
        assert_eq!(empty.dtype(cumulative).unwrap(), DType::F16);
        assert!(CpuBackend
            .execute(
                &empty,
                cumulative,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64()
            .is_empty());

        let mut invalid = Graph::new();
        let input = invalid.input("input", [2]);
        let before_nodes = invalid.node_count();
        assert!(invalid.cumsum(input, 1).is_err());
        assert_eq!(invalid.node_count(), before_nodes);
    }

    #[test]
    fn cumprod_matches_tinygrad_signed_axis_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let cumulative = graph.cumprod(input, -1).unwrap();
        let loss = graph.sum_all(cumulative).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([(
            "input".into(),
            TensorData::new([2, 3], vec![2.0, 3.0, 4.0, 2.0, 0.0, -3.0]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, cumulative, &inputs).unwrap().to_vec_f64(),
            vec![2.0, 6.0, 24.0, 2.0, 0.0, 0.0]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs).unwrap().to_vec_f64(),
            vec![16.0, 10.0, 6.0, 1.0, -4.0, 0.0]
        );

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::I8);
        let cumulative = scalar.cumprod(input, 0).unwrap();
        assert_eq!(scalar.dtype(cumulative).unwrap(), DType::I8);
        assert_eq!(
            CpuBackend
                .execute(
                    &scalar,
                    cumulative,
                    &HashMap::from([(
                        "input".into(),
                        TensorData::from_scalars([], DType::I8, [Scalar::I(-5)]).unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![-5.0]
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0], DType::F16);
        let cumulative = empty.cumprod(input, -1).unwrap();
        assert_eq!(empty.dtype(cumulative).unwrap(), DType::F16);
        assert!(CpuBackend
            .execute(
                &empty,
                cumulative,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64()
            .is_empty());

        let mut invalid = Graph::new();
        let input = invalid.input("input", [2]);
        let before_nodes = invalid.node_count();
        assert!(invalid.cumprod(input, 1).is_err());
        assert_eq!(invalid.node_count(), before_nodes);
    }

    #[test]
    fn typed_reduction_accumulation_preflights_and_preserves_source_contracts() {
        let mut malformed = Graph::new();
        let input = malformed.input_dtype("input", [2, 2], DType::F16);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                Some(vec![-1, 1]),
                false,
                ReductionDType::sum_default(DType::F16),
            ),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                Some(vec![isize::MIN]),
                false,
                ReductionDType::sum_default(DType::F16),
            ),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                Some(vec![-1]),
                false,
                ReductionDType::new(DType::F64, DType::F32),
            ),
            Err(Error::InvalidElementwiseDType { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut graph = Graph::new();
        let narrow = graph.input_dtype("narrow", [2, 2], DType::F16);
        let narrowed_sum = graph
            .reduce_with_dtypes(
                narrow,
                ReduceKind::Sum,
                Some(vec![-1]),
                false,
                ReductionDType::sum_default(DType::F16),
            )
            .unwrap();
        assert_eq!(graph.dtype(narrowed_sum).unwrap(), DType::F16);
        let reduced = match &graph.nodes[narrowed_sum.index()].op {
            crate::Op::Cast {
                input,
                dtype: DType::F16,
            } => *input,
            op => panic!("expected final F16 cast, got {op:?}"),
        };
        assert_eq!(graph.dtype(reduced).unwrap(), DType::F32);
        let narrow_data = TensorData::from_scalars(
            [2, 2],
            DType::F16,
            [1.5, 2.25, 3.5, 4.75].map(crate::Scalar::F),
        )
        .unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    narrowed_sum,
                    &HashMap::from([("narrow".into(), narrow_data.clone())]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![3.75, 8.25]
        );
        let narrow_loss = graph.sum_all(narrowed_sum).unwrap();
        let narrow_gradient = graph.grad(narrow_loss, narrow).unwrap();
        assert_eq!(graph.dtype(narrow_gradient).unwrap(), DType::F16);
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    narrow_gradient,
                    &HashMap::from([("narrow".into(), narrow_data)]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![1.; 4]
        );

        let bfloat = graph.input_dtype("bfloat", [2], DType::BF16);
        let bfloat_sum = graph
            .reduce_with_dtypes(
                bfloat,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::BF16),
            )
            .unwrap();
        let reduced = match &graph.nodes[bfloat_sum.index()].op {
            crate::Op::Cast {
                input,
                dtype: DType::BF16,
            } => *input,
            op => panic!("expected final BF16 cast, got {op:?}"),
        };
        assert_eq!(graph.dtype(reduced).unwrap(), DType::F32);

        let single = graph.input("single", [2]);
        let single_sum = graph
            .reduce_with_dtypes(
                single,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::F32),
            )
            .unwrap();
        assert_eq!(graph.dtype(single_sum).unwrap(), DType::F32);
        let widened_product = graph
            .reduce_with_dtypes(
                single,
                ReduceKind::Product,
                None,
                false,
                ReductionDType::new(DType::F64, DType::F64),
            )
            .unwrap();
        assert_eq!(graph.dtype(widened_product).unwrap(), DType::F64);

        let wide = graph.input_dtype("wide", [2], DType::F64);
        let wide_sum = graph
            .reduce_with_dtypes(
                wide,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::F64),
            )
            .unwrap();
        assert_eq!(graph.dtype(wide_sum).unwrap(), DType::F64);

        let integer = graph.input_dtype("integer", [2], DType::I8);
        let integer_sum = graph
            .reduce_with_dtypes(
                integer,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::I8),
            )
            .unwrap();
        assert_eq!(graph.dtype(integer_sum).unwrap(), DType::I32);

        let legacy = graph.input_dtype("legacy", [2], DType::F16);
        let legacy_sum = graph.reduce(legacy, ReduceKind::Sum, None, false).unwrap();
        assert_eq!(graph.dtype(legacy_sum).unwrap(), DType::F16);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 2]);
        let original_nodes = overflow.node_count();
        assert!(matches!(
            overflow.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::F32),
            ),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), original_nodes);
    }

    #[test]
    fn variance_and_std_match_tinygrad_correction_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [3]);
        let default_variance = graph.var(input, None, false, None).unwrap();
        let population_variance = graph
            .var(
                input,
                None,
                false,
                Some(VarianceCorrection::new(0)),
            )
            .unwrap();
        let negative_correction = graph
            .var(
                input,
                None,
                false,
                Some(VarianceCorrection::new(-1)),
            )
            .unwrap();
        let standard_deviation = graph
            .std(
                input,
                Some(vec![-1]),
                true,
                Some(VarianceCorrection::new(0)),
            )
            .unwrap();
        let default_gradient = graph.grad(default_variance, input).unwrap();
        let gradient = graph.grad(population_variance, input).unwrap();
        let inputs = HashMap::from([("input".into(), data([3], &[1., 3., 5.]))]);

        assert_eq!(
            CpuBackend.execute(&graph, default_variance, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![4.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, population_variance, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![(8.0f32 / 3.0) as f64]
        );
        assert_eq!(
            CpuBackend.execute(&graph, negative_correction, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![2.]
        );
        assert_eq!(graph.shape(standard_deviation).unwrap(), &Shape::new([1]));
        assert_eq!(
            CpuBackend
                .execute(&graph, standard_deviation, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![(8.0f32 / 3.0).sqrt() as f64]
        );
        assert_eq!(
            CpuBackend.execute(&graph, default_gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![-2., 0., 2.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![(-4.0f32 / 3.0) as f64, 0., (4.0f32 / 3.0) as f64]
        );

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", [2], DType::F16);
        let bf16 = narrow.input_dtype("bf16", [2], DType::BF16);
        let f16_variance = narrow.var(f16, None, false, None).unwrap();
        let bf16_variance = narrow.var(bf16, None, false, None).unwrap();
        assert_eq!(narrow.dtype(f16_variance).unwrap(), DType::F16);
        assert_eq!(narrow.dtype(bf16_variance).unwrap(), DType::BF16);
        assert!(narrow.nodes.iter().any(|node| {
            matches!(&node.op, crate::Op::Reduce { kind: ReduceKind::Sum, .. })
                && node.dtype == DType::F32
        }));
        let f16_data = TensorData::from_scalars([2], DType::F16, [Scalar::F(1.5), Scalar::F(2.5)])
            .unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &narrow,
                    f16_variance,
                    &HashMap::from([("f16".into(), f16_data)]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![0.5]
        );

        let mut integer = Graph::new();
        let values = integer.input_dtype("values", [2], DType::I32);
        let variance = integer
            .var(values, None, false, Some(VarianceCorrection::new(0)))
            .unwrap();
        assert_eq!(integer.dtype(variance).unwrap(), DType::F32);
        assert_eq!(
            CpuBackend
                .execute(
                    &integer,
                    variance,
                    &HashMap::from([(
                        "values".into(),
                        TensorData::from_scalars(
                            [2],
                            DType::I32,
                            [Scalar::I(1), Scalar::I(3)],
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![1.]
        );

        // The Python counts are weak integers. For narrow outputs they join
        // the F32 accumulator at division, rather than first rounding to F16.
        let mut narrow_count = Graph::new();
        let values = narrow_count.input_dtype("values", [2051], DType::F16);
        narrow_count.var(values, None, false, None).unwrap();
        let constants = narrow_count
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                crate::Op::Constant(data) if data.dtype() == DType::F32 => {
                    Some(data.scalar_at(0).as_f64())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(constants.contains(&2051.));
        assert!(constants.contains(&2050.));

        let mut scalar = Graph::new();
        let value = scalar.input("value", []);
        let negative_axis = scalar.var(value, Some(vec![-1]), false, None).unwrap();
        let zero_axis = scalar.var(value, Some(vec![0]), false, None).unwrap();
        assert_eq!(scalar.shape(negative_axis).unwrap(), &Shape::new([]));
        assert_eq!(scalar.shape(zero_axis).unwrap(), &Shape::new([]));
    }

    #[test]
    fn var_mean_matches_tinygrad_literal_pair_dtype_axes_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let (variance, mean) = graph
            .var_mean(
                input,
                Some(vec![-1]),
                true,
                Some(VarianceCorrection::new(0)),
            )
            .unwrap();
        assert_eq!(graph.shape(variance).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.shape(mean).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(variance).unwrap(), DType::F32);
        assert_eq!(graph.dtype(mean).unwrap(), DType::F32);

        let loss = graph
            .add(graph.sum_all(variance).unwrap(), graph.sum_all(mean).unwrap())
            .unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([("input".into(), data([2, 2], &[1., 3., 5., 7.]))]);
        assert_eq!(
            CpuBackend.execute(&graph, variance, &inputs).unwrap().to_vec_f64(),
            vec![1., 1.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, mean, &inputs).unwrap().to_vec_f64(),
            vec![2., 6.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs).unwrap().to_vec_f64(),
            vec![-0.5, 1.5, -0.5, 1.5]
        );

        let mut integer = Graph::new();
        let input = integer.input_dtype("input", [2], DType::I16);
        let (variance, mean) = integer.var_mean(input, None, false, None).unwrap();
        assert_eq!(integer.dtype(variance).unwrap(), DType::F32);
        assert_eq!(integer.dtype(mean).unwrap(), DType::F32);

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 2], DType::BF16);
        let (variance, mean) = empty.var_mean(input, Some(vec![0]), false, None).unwrap();
        assert_eq!(empty.shape(variance).unwrap(), &Shape::new([2]));
        assert_eq!(empty.shape(mean).unwrap(), &Shape::new([2]));
    }

    #[test]
    fn var_mean_preflights_both_source_reductions_before_graph_growth() {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 3]);
        let nodes = malformed.node_count();
        assert!(malformed
            .var_mean(input, Some(vec![0, -2]), false, None)
            .is_err());
        assert_eq!(malformed.node_count(), nodes);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX]);
        let nodes = overflow.node_count();
        assert!(overflow
            .var_mean(
                input,
                None,
                false,
                Some(VarianceCorrection::new(-1)),
            )
            .is_err());
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn std_mean_matches_tinygrad_literal_pair_dtype_axes_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let (standard_deviation, mean) = graph
            .std_mean(
                input,
                Some(vec![-1]),
                true,
                Some(VarianceCorrection::new(0)),
            )
            .unwrap();
        assert_eq!(graph.shape(standard_deviation).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.shape(mean).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(standard_deviation).unwrap(), DType::F32);
        assert_eq!(graph.dtype(mean).unwrap(), DType::F32);

        let loss = graph
            .add(
                graph.sum_all(standard_deviation).unwrap(),
                graph.sum_all(mean).unwrap(),
            )
            .unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([("input".into(), data([2, 2], &[1., 3., 5., 7.]))]);
        assert_eq!(
            CpuBackend
                .execute(&graph, standard_deviation, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![1., 1.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, mean, &inputs).unwrap().to_vec_f64(),
            vec![2., 6.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs).unwrap().to_vec_f64(),
            vec![0., 1., 0., 1.]
        );

        let mut integer = Graph::new();
        let input = integer.input_dtype("input", [2], DType::U16);
        let (standard_deviation, mean) = integer.std_mean(input, None, false, None).unwrap();
        assert_eq!(integer.dtype(standard_deviation).unwrap(), DType::F32);
        assert_eq!(integer.dtype(mean).unwrap(), DType::F32);

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::F16);
        let (standard_deviation, mean) = scalar.std_mean(input, Some(vec![-1]), false, None).unwrap();
        assert_eq!(scalar.shape(standard_deviation).unwrap(), &Shape::new([]));
        assert_eq!(scalar.shape(mean).unwrap(), &Shape::new([]));
    }

    #[test]
    fn std_mean_preflights_both_source_reductions_before_graph_growth() {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 3]);
        let nodes = malformed.node_count();
        assert!(malformed
            .std_mean(input, Some(vec![0, -2]), false, None)
            .is_err());
        assert_eq!(malformed.node_count(), nodes);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX]);
        let nodes = overflow.node_count();
        assert!(overflow
            .std_mean(
                input,
                None,
                false,
                Some(VarianceCorrection::new(-1)),
            )
            .is_err());
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn statistics_default_wrappers_keep_tinygrad_defaults_pair_order_and_atomicity() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mean = graph.mean_default(input).unwrap();
        let variance = graph.var_default(input).unwrap();
        let standard_deviation = graph.std_default(input).unwrap();
        let (pair_variance, pair_mean) = graph.var_mean_default(input).unwrap();
        let (pair_standard_deviation, pair_standard_mean) = graph.std_mean_default(input).unwrap();
        for output in [
            mean, variance, standard_deviation, pair_variance, pair_mean,
            pair_standard_deviation, pair_standard_mean,
        ] {
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
            assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        }
        // The paired forms stay literal `(var/std, mean)`, rather than
        // substituting variance's internal keepdim mean for the public mean.
        assert_ne!(pair_variance, pair_mean);
        assert_ne!(pair_standard_deviation, pair_standard_mean);
        assert!((0..graph.node_count()).any(|index| matches!(
            graph.op(NodeId(index)).unwrap(), crate::Op::Unary { op: UnaryOp::Square, .. }
        )));
        assert!(matches!(
            graph.op(pair_standard_deviation).unwrap(),
            crate::Op::Unary { op: UnaryOp::Sqrt, .. }
        ));
        let loss = graph.add(graph.sum_all(pair_variance).unwrap(), graph.sum_all(pair_mean).unwrap()).unwrap();
        assert!(graph.grad(loss, input).is_ok());

        let mut nonfloat = Graph::new();
        let input = nonfloat.input_dtype("input", [], DType::I16);
        let (variance, mean) = nonfloat.var_mean_default(input).unwrap();
        let (standard_deviation, std_mean) = nonfloat.std_mean_default(input).unwrap();
        for output in [
            nonfloat.mean_default(input).unwrap(), variance, mean,
            nonfloat.var_default(input).unwrap(), standard_deviation, std_mean,
            nonfloat.std_default(input).unwrap(),
        ] {
            assert_eq!(nonfloat.shape(output).unwrap(), &Shape::new([]));
            assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
        }

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 2], DType::BF16);
        let (variance, mean) = empty.var_mean_default(input).unwrap();
        let (standard_deviation, std_mean) = empty.std_mean_default(input).unwrap();
        for output in [
            empty.mean_default(input).unwrap(), variance, mean,
            empty.var_default(input).unwrap(), standard_deviation, std_mean,
            empty.std_default(input).unwrap(),
        ] {
            assert_eq!(empty.shape(output).unwrap(), &Shape::new([]));
            assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
        }

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(overflow.mean_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
        assert!(overflow.var_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
        assert!(overflow.std_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
        assert!(overflow.var_mean_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
        assert!(overflow.std_mean_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn variance_preflights_axes_extents_and_preserves_empty_zero_denominator_policy() {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 3]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.var(input, Some(vec![-1, 1]), false, None),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.var(input, Some(vec![isize::MIN]), false, None),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX]);
        let original_nodes = overflow.node_count();
        assert!(matches!(
            overflow.var(
                input,
                None,
                false,
                Some(VarianceCorrection::new(-1)),
            ),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), original_nodes);

        let mut empty = Graph::new();
        let input = empty.input("input", [0]);
        let variance = empty.var(input, None, false, None).unwrap();
        let output = CpuBackend
            .execute(
                &empty,
                variance,
                &HashMap::from([("input".into(), data([0], &[]))]),
            )
            .unwrap()
            .to_vec_f64();
        assert!(output[0].is_nan());

        let mut singleton = Graph::new();
        let input = singleton.input("input", []);
        let variance = singleton.var(input, None, false, None).unwrap();
        let output = CpuBackend
            .execute(
                &singleton,
                variance,
                &HashMap::from([("input".into(), TensorData::scalar(7.))]),
            )
            .unwrap()
            .to_vec_f64();
        assert!(output[0].is_nan());
    }

    #[test]
    fn l2_normalize_matches_the_closed_tinygrad_default_composition() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let normalized = graph.normalize_l2(input, -1, 1e-12).unwrap();
        let loss = graph.sum_all(normalized).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([("input".into(), data([2, 2], &[3., 4., 5., 12.]))]);

        assert_eq!(graph.shape(normalized).unwrap(), &Shape::new([2, 2]));
        assert_eq!(
            CpuBackend.execute(&graph, normalized, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![
                0.6f32 as f64,
                0.8f32 as f64,
                (5.0f32 / 13.0) as f64,
                (12.0f32 / 13.0) as f64,
            ]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![
                0.032f32 as f64,
                -0.024f32 as f64,
                (84.0f32 / 2197.0) as f64,
                (-35.0f32 / 2197.0) as f64,
            ]
        );

        let mut narrow = Graph::new();
        let input = narrow.input_dtype("input", [2], DType::F16);
        let normalized = narrow.normalize_l2(input, 0, 1e-12).unwrap();
        assert_eq!(narrow.dtype(normalized).unwrap(), DType::F16);
        assert!(narrow.nodes.iter().any(|node| {
            matches!(&node.op, crate::Op::Reduce { kind: ReduceKind::Sum, .. })
                && node.dtype == DType::F32
        }));
        assert_eq!(
            CpuBackend
                .execute(
                    &narrow,
                    normalized,
                    &HashMap::from([(
                        "input".into(),
                        TensorData::from_scalars(
                            [2],
                            DType::F16,
                            [Scalar::F(3.), Scalar::F(4.)],
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![0.60009765625, 0.7998046875]
        );
    }

    #[test]
    fn l2_normalize_preflights_dtype_axis_and_empty_scalar_boundaries() {
        let mut malformed = Graph::new();
        let integer = malformed.input_dtype("integer", [2], DType::I32);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.normalize_l2(integer, 0, 1e-12),
            Err(Error::InvalidElementwiseDType { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let floating = malformed.input("floating", [2]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.normalize_l2(floating, 1, 1e-12),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let scalar = malformed.input("scalar", []);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.normalize_l2(scalar, 0, 1e-12),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut empty = Graph::new();
        let input = empty.input("input", [2, 0]);
        let normalized = empty.normalize_l2(input, -1, 1e-12).unwrap();
        assert_eq!(empty.shape(normalized).unwrap(), &Shape::new([2, 0]));
        assert_eq!(
            CpuBackend
                .execute(
                    &empty,
                    normalized,
                    &HashMap::from([("input".into(), data([2, 0], &[]))]),
                )
                .unwrap()
                .to_vec_f64(),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn mean_with_axes_matches_tinygrad_typed_mean_and_preflights() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F16);
        let output = graph.mean_with_axes(input, Some(vec![-1]), true).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        assert!(graph.nodes.iter().any(|node| {
            matches!(&node.op, crate::Op::Reduce { kind: ReduceKind::Sum, .. })
                && node.dtype == DType::F32
        }));
        let loss = graph.sum_all(output).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            TensorData::from_scalars(
                [2, 2],
                DType::F16,
                [Scalar::F(3.), Scalar::F(5.), Scalar::F(7.), Scalar::F(9.)],
            )
            .unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
            vec![4., 8.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
            vec![0.5; 4]
        );

        let mut all = Graph::new();
        let x = all.input("input", [2, 2]);
        let output = all.mean_with_axes(x, None, false).unwrap();
        assert_eq!(all.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(
            CpuBackend
                .execute(&all, output, &HashMap::from([("input".into(), data([2, 2], &[1., 2., 3., 4.]))]))
                .unwrap()
                .to_vec_f64(),
            vec![2.5]
        );

        for dtype in [
            DType::Bool,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
        ] {
            let mut promoted = Graph::new();
            let x = promoted.input_dtype("input", [], dtype);
            let output = promoted.mean_with_axes(x, None, false).unwrap();
            assert_eq!(promoted.dtype(output).unwrap(), DType::F32);
        }
        let mut scalar = Graph::new();
        let x = scalar.input("input", []);
        let output = scalar.mean_with_axes(x, Some(vec![-1]), false).unwrap();
        assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));

        let mut multiple = Graph::new();
        let x = multiple.input("input", [2, 3]);
        let output = multiple.mean_with_axes(x, Some(vec![0, -1]), true).unwrap();
        assert_eq!(multiple.shape(output).unwrap(), &Shape::new([1, 1]));

        let mut legacy = Graph::new();
        let x = legacy.input("input", [2, 2]);
        let old = legacy.mean(x, 1).unwrap();
        let new = legacy.mean_with_axes(x, Some(vec![1]), false).unwrap();
        assert_eq!(legacy.shape(old).unwrap(), legacy.shape(new).unwrap());

        let mut empty = Graph::new();
        let x = empty.input("input", [2, 0]);
        let output = empty.mean_with_axes(x, Some(vec![1]), false).unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([2]));
        let values = CpuBackend
            .execute(&empty, output, &HashMap::from([("input".into(), data([2, 0], &[]))]))
            .unwrap()
            .to_vec_f64();
        assert!(values.iter().all(|value| value.is_nan()));

        let mut malformed = Graph::new();
        let x = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed.mean_with_axes(x, Some(vec![0, -2]), false).is_err());
        assert_eq!(malformed.node_count(), nodes);

        let mut overflow = Graph::new();
        let x = overflow.input("input", [usize::MAX, 2]);
        let nodes = overflow.node_count();
        assert!(overflow.mean_with_axes(x, None, false).is_err());
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn max_with_axes_matches_tinygrad_shapes_and_empty_identity() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F64);
        let output = graph.max_with_axes(input, Some(vec![-1]), true).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F64);
        let loss = graph.sum_all(output).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            TensorData::from_scalars(
                [2, 2],
                DType::F64,
                [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(f64::NAN), Scalar::F(3.)],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        assert!(values.scalar_at(1).as_f64().is_nan());
        let gradients = CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64();
        assert_eq!(&gradients[..2], &[0.5, 0.5]);

        for dtype in [
            DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64, DType::U8,
            DType::U16, DType::U32, DType::U64, DType::F16, DType::BF16, DType::F32,
            DType::F64,
        ] {
            let mut typed = Graph::new();
            let x = typed.input_dtype("input", [], dtype);
            let output = typed.max_with_axes(x, None, false).unwrap();
            assert_eq!(typed.dtype(output).unwrap(), dtype);
        }
        let mut empty = Graph::new();
        let x = empty.input_dtype("input", [2, 0], DType::F32);
        let output = empty.max_with_axes(x, Some(vec![1]), false).unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([2]));
        let values = CpuBackend
            .execute(&empty, output, &HashMap::from([("input".into(), data([2, 0], &[]))]))
            .unwrap()
            .to_vec_f64();
        assert!(values.iter().all(|value| value.is_infinite() && value.is_sign_negative()));

        let mut malformed = Graph::new();
        let x = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed.max_with_axes(x, Some(vec![0, -2]), false).is_err());
        assert_eq!(malformed.node_count(), nodes);
    }

    #[test]
    fn min_with_axes_matches_tinygrad_inverse_max_and_empty_identity() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F64);
        let output = graph.min_with_axes(input, Some(vec![-1]), true).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F64);
        let loss = graph.sum_all(output).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            TensorData::from_scalars(
                [2, 2],
                DType::F64,
                [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(f64::NAN), Scalar::F(-3.)],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        assert!(values.scalar_at(1).as_f64().is_nan());
        let gradients = CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64();
        assert_eq!(&gradients[..2], &[0.5, 0.5]);

        for dtype in [
            DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64, DType::U8,
            DType::U16, DType::U32, DType::U64, DType::F16, DType::BF16, DType::F32,
            DType::F64,
        ] {
            let mut typed = Graph::new();
            let x = typed.input_dtype("input", [], dtype);
            let output = typed.min_with_axes(x, None, false).unwrap();
            assert_eq!(typed.dtype(output).unwrap(), dtype);
        }
        let mut empty = Graph::new();
        let x = empty.input_dtype("input", [2, 0], DType::F32);
        let output = empty.min_with_axes(x, Some(vec![1]), false).unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([2]));
        let values = CpuBackend
            .execute(&empty, output, &HashMap::from([("input".into(), data([2, 0], &[]))]))
            .unwrap()
            .to_vec_f64();
        assert!(values.iter().all(|value| value.is_infinite() && value.is_sign_positive()));

        let mut scalar = Graph::new();
        let x = scalar.input("input", []);
        let output = scalar.min_with_axes(x, Some(vec![-1]), false).unwrap();
        assert_eq!(output, x);

        let mut malformed = Graph::new();
        let x = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed.min_with_axes(x, Some(vec![0, -2]), false).is_err());
        assert_eq!(malformed.node_count(), nodes);
    }
}
