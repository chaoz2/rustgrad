use super::{
    creation::{lazy_arange_default_int_plan, LazyArangePlan},
    elementwise::{source_lub, source_weak_scalar_dtype},
    shape::{normalize_axes, reduction_shape, unary_dtype},
    Graph, NodeId,
};
use crate::{
    DType, Error, ReduceKind, ReductionDType, Result, Scalar, Shape, TensorData, UnaryOp,
    VarianceCorrection,
};

struct MeanPlan {
    axes: Vec<isize>,
    output_shape: Shape,
    sum_dtypes: ReductionDType,
    division_dtype: DType,
    output_dtype: DType,
    divisor: TensorData,
}

/// Pure descriptor for tinygrad's literal `Tensor.layernorm` composition.
struct LayerNormPlan {
    axes: Vec<isize>,
    mean_shape: Shape,
    mean_dtype: DType,
    centered_shape: Shape,
    centered_dtype: DType,
    variance_shape: Shape,
    variance_dtype: DType,
    epsilon: TensorData,
    output_shape: Shape,
    output_dtype: DType,
}

enum NormalizeLowering {
    Zero {
        sum: ReductionDType,
    },
    Pow {
        pow_dtype: DType,
        sum: ReductionDType,
        exponent: TensorData,
        reciprocal_exponent: TensorData,
    },
}

struct NormalizePlan {
    axes: Vec<isize>,
    denominator_shape: Shape,
    denominator_dtype: DType,
    epsilon: TensorData,
    output_shape: Shape,
    output_dtype: DType,
    lowering: NormalizeLowering,
}

fn normalize_output_dtype(input: DType, denominator: DType) -> DType {
    let division = source_lub(input, denominator);
    let dividend = if division.is_float() {
        division
    } else {
        DType::F32
    };
    let reciprocal = unary_dtype(UnaryOp::Reciprocal, division);
    source_lub(dividend, reciprocal)
}

fn normalize_plan(
    graph: &Graph,
    input: NodeId,
    p: f64,
    dim: isize,
    eps: f64,
) -> Result<NormalizePlan> {
    let source = graph.node(input)?;
    let shape = source.shape.clone();
    let dtype = source.dtype;
    let axes = normalize_axes(input, shape.rank(), Some(vec![dim]))?
        .into_iter()
        .map(|axis| axis as isize)
        .collect::<Vec<_>>();
    let output_shape = reduction_shape(
        &shape,
        &axes.iter().map(|axis| *axis as usize).collect::<Vec<_>>(),
        true,
    );
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&shape, dtype)?;
    let (denominator_dtype, lowering) = if p == 0.0 {
        let sum = ReductionDType::sum_default(DType::Bool);
        for (candidate, storage) in [
            (&shape, DType::Bool),
            (&shape, sum.accumulator),
            (&output_shape, sum.accumulator),
            (&output_shape, sum.output),
        ] {
            extent(candidate, storage)?;
        }
        (sum.output, NormalizeLowering::Zero { sum })
    } else {
        // `abs().pow(p)`: Python `p: float` weak-commits at a floating
        // storage width, so this path never sends a live integer Pow to raw
        // IR/backends. The final root exponent is another independently
        // committed Python float, `1/p`.
        let exponent_dtype = source_weak_scalar_dtype(dtype, Scalar::F(p));
        let pow_dtype = source_lub(dtype, exponent_dtype);
        let exponent = TensorData::scalar_with_dtype(Scalar::F(p), exponent_dtype);
        let sum = ReductionDType::sum_default(pow_dtype);
        let reciprocal_value = 1.0 / p;
        let reciprocal_dtype = source_weak_scalar_dtype(sum.output, Scalar::F(reciprocal_value));
        let denominator_dtype = source_lub(sum.output, reciprocal_dtype);
        let reciprocal_exponent =
            TensorData::scalar_with_dtype(Scalar::F(reciprocal_value), reciprocal_dtype);
        for (candidate, storage) in [
            (&shape, dtype), // abs
            (exponent.shape(), exponent.dtype()),
            (&shape, pow_dtype), // first Pow cast/result
            (&shape, sum.accumulator),
            (&output_shape, sum.accumulator),
            (&output_shape, sum.output),
            (reciprocal_exponent.shape(), reciprocal_exponent.dtype()),
            (&output_shape, denominator_dtype), // second Pow cast/result
        ] {
            extent(candidate, storage)?;
        }
        if !pow_dtype.is_float()
            || !denominator_dtype.is_float()
            || exponent.shape() != &Shape::new([])
            || reciprocal_exponent.shape() != &Shape::new([])
            || source_lub(dtype, exponent_dtype) != pow_dtype
            || source_lub(sum.output, reciprocal_dtype) != denominator_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "normalize scalar pow promotion",
                actual: denominator_dtype,
            });
        }
        (
            denominator_dtype,
            NormalizeLowering::Pow {
                pow_dtype,
                sum,
                exponent,
                reciprocal_exponent,
            },
        )
    };
    let epsilon_dtype = source_weak_scalar_dtype(denominator_dtype, Scalar::F(eps));
    let maximum_dtype = source_lub(denominator_dtype, epsilon_dtype);
    let epsilon = TensorData::scalar_with_dtype(Scalar::F(eps), epsilon_dtype);
    let final_dtype = normalize_output_dtype(dtype, maximum_dtype);
    for (candidate, storage) in [
        (epsilon.shape(), epsilon.dtype()),
        (&output_shape, maximum_dtype),
        (&shape, source_lub(dtype, maximum_dtype)),
        (
            &output_shape,
            unary_dtype(UnaryOp::Reciprocal, source_lub(dtype, maximum_dtype)),
        ),
        (&shape, final_dtype),
    ] {
        extent(candidate, storage)?;
    }
    if epsilon.shape() != &Shape::new([])
        || epsilon.dtype() != epsilon_dtype
        || output_shape.broadcast_with(epsilon.shape())? != output_shape
        || source_lub(denominator_dtype, epsilon_dtype) != maximum_dtype
        || shape.broadcast_with(&output_shape)? != shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "normalize epsilon promotion",
            actual: maximum_dtype,
        });
    }
    Ok(NormalizePlan {
        axes,
        denominator_shape: output_shape,
        denominator_dtype: maximum_dtype,
        epsilon,
        output_shape: shape,
        output_dtype: final_dtype,
        lowering,
    })
}

fn layernorm_plan(
    graph: &Graph,
    input: NodeId,
    axes: Vec<isize>,
    eps: f64,
) -> Result<LayerNormPlan> {
    let source = graph.node(input)?;
    let input_shape = source.shape.clone();
    let input_dtype = source.dtype;
    // Source literal: `y = self - self.mean(axis, keepdim=True)`.
    let mean = mean_plan(input, &input_shape, input_dtype, Some(axes), true)?;
    let centered_shape = input_shape.broadcast_with(&mean.output_shape)?;
    let centered_dtype = source_lub(input_dtype, mean.output_dtype);
    // It then reuses that exact `y` for `y*y`, followed by a second typed
    // mean over the same normalized axes.
    let variance = mean_plan(
        input,
        &centered_shape,
        centered_dtype,
        Some(mean.axes.clone()),
        true,
    )?;
    let epsilon_dtype = source_weak_scalar_dtype(variance.output_dtype, Scalar::F(eps));
    let variance_epsilon_dtype = source_lub(variance.output_dtype, epsilon_dtype);
    let epsilon = TensorData::scalar_with_dtype(Scalar::F(eps), epsilon_dtype);
    let output_shape = centered_shape.broadcast_with(&variance.output_shape)?;
    let output_dtype = source_lub(centered_dtype, variance_epsilon_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    // The two MeanPlan calls above prove their complete typed reductions,
    // including zero-count divisors. These are the cross-stage descriptors
    // that source composition adds: center, square, epsilon add, rsqrt, and
    // final multiply.
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (&mean.output_shape, mean.output_dtype),
        (&centered_shape, centered_dtype),
        (&centered_shape, centered_dtype), // `y*y`
        (&variance.output_shape, variance.output_dtype),
        (epsilon.shape(), epsilon.dtype()),
        (&variance.output_shape, variance_epsilon_dtype),
        (&output_shape, variance_epsilon_dtype), // rsqrt broadcast
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if input_shape.broadcast_with(&mean.output_shape)? != centered_shape
        || centered_shape.broadcast_with(&variance.output_shape)? != output_shape
        || epsilon.shape() != &Shape::new([])
        || epsilon.dtype() != epsilon_dtype
        || variance.output_shape.broadcast_with(epsilon.shape())? != variance.output_shape
        || source_lub(input_dtype, mean.output_dtype) != centered_dtype
        || source_lub(variance.output_dtype, epsilon_dtype) != variance_epsilon_dtype
        || source_lub(centered_dtype, variance_epsilon_dtype) != output_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "layernorm source promotion",
            actual: output_dtype,
        });
    }
    Ok(LayerNormPlan {
        axes: mean.axes,
        mean_shape: mean.output_shape,
        mean_dtype: mean.output_dtype,
        centered_shape,
        centered_dtype,
        variance_shape: variance.output_shape,
        variance_dtype: variance.output_dtype,
        epsilon,
        output_shape,
        output_dtype,
    })
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

struct CumExtremaPlan {
    axis: usize,
    scalar: bool,
    extent: usize,
    prefixes: Vec<Vec<(usize, usize)>>,
    ascending: Option<LazyArangePlan>,
    descending: Option<LazyArangePlan>,
    index_offset: Option<TensorData>,
}

/// Descriptor-only contract for tinygrad's `Tensor.cumprod(axis)`.
///
/// Tinygrad implements the scan through `_cumalu(Ops.MUL)`: every inclusive
/// prefix retains the input storage width, is reduced with Product, and the
/// resulting singleton-axis lanes are concatenated.  RustGrad represents the
/// same concrete composition with checked Shrink/Product/Concat nodes.  Keep
/// all of those descriptors here so a malformed late prefix cannot publish
/// an earlier movement or reduction node.
struct CumprodPlan {
    axis: Option<usize>,
    prefixes: Vec<Vec<(usize, usize)>>,
    dtypes: ReductionDType,
}

fn cumulative_product_plan(
    input: NodeId,
    shape: &Shape,
    dtype: DType,
    axis: isize,
) -> Result<CumprodPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    let dtypes = ReductionDType::product_default(dtype);
    extent(shape, dtype)?;
    if !valid_reduction_dtypes(ReduceKind::Product, dtype, dtypes) {
        return Err(Error::InvalidElementwiseDType {
            op: "cumprod product storage",
            actual: dtypes.output,
        });
    }
    if shape.rank() == 0 {
        // `_split_cumalu` resolves the axis before its rank-zero identity
        // branch; `_resolve_dim` admits exactly -1 and 0 at rank zero.
        if !matches!(axis, -1 | 0) {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: 0,
            });
        }
        return Ok(CumprodPlan {
            axis: None,
            prefixes: Vec::new(),
            dtypes,
        });
    }
    let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
    // `_split_cumalu` is an identity for every zero-extent descriptor after
    // signed-axis resolution. The source extent above is still checked.
    if shape.dims().contains(&0) {
        return Ok(CumprodPlan {
            axis: None,
            prefixes: Vec::new(),
            dtypes,
        });
    }

    let mut prefixes = Vec::with_capacity(shape.dims()[axis]);
    for end in 1..=shape.dims()[axis] {
        let bounds = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(dimension, &dimension_extent)| {
                if dimension == axis {
                    (0, end)
                } else {
                    (0, dimension_extent)
                }
            })
            .collect::<Vec<_>>();
        let mut prefix_dims = shape.dims().to_vec();
        prefix_dims[axis] = end;
        let prefix_shape = Shape::new(prefix_dims);
        let mut reduced_dims = prefix_shape.dims().to_vec();
        reduced_dims[axis] = 1;
        let reduced_shape = Shape::new(reduced_dims);
        // Product has no widening/narrowing boundary, but validate both the
        // accumulator and final descriptor explicitly so this remains true
        // if the shared dtype representation evolves.
        extent(&prefix_shape, dtype)?;
        extent(&prefix_shape, dtypes.accumulator)?;
        extent(&reduced_shape, dtypes.accumulator)?;
        extent(&reduced_shape, dtypes.output)?;
        prefixes.push(bounds);
    }
    // The concatenated singleton lanes reconstruct the original descriptor.
    extent(shape, dtypes.output)?;
    Ok(CumprodPlan {
        axis: Some(axis),
        prefixes,
        dtypes,
    })
}

/// Concrete whole-operation plan for tinygrad's stable
/// `Tensor.logcumsumexp(axis)`.  The source builds the cumulative maxima and
/// lower-triangle predicate explicitly, rather than using a scan primitive.
struct LogCumSumExpPlan {
    axis: usize,
    transposed_shape: Shape,
    matrix_shape: Shape,
    source_dtype: DType,
    exp_source_dtype: DType,
    exp_work_dtype: DType,
    sum_dtypes: ReductionDType,
    log_dtype: DType,
    output_dtype: DType,
    cumulative_max: CumExtremaPlan,
    range: LazyArangePlan,
    minimum: TensorData,
}

/// tinygrad's concrete dtype lattice has one notable exception to RustGrad's
/// raw promotion: the I64/U64 join is the default float, not F64.
fn reduction_source_lub(lhs: DType, rhs: DType) -> DType {
    if matches!(
        (lhs, rhs),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

fn logcumsumexp_plan(
    input: NodeId,
    shape: &Shape,
    dtype: DType,
    axis: isize,
) -> Result<LogCumSumExpPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(shape, dtype)?;
    let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
    let mut dimensions = shape.dims().to_vec();
    dimensions.swap(axis, shape.rank() - 1);
    let transposed_shape = Shape::new(dimensions);
    let axis_extent = *transposed_shape.dims().last().expect("non-scalar shape");
    let mut matrix_dimensions = transposed_shape.dims().to_vec();
    matrix_dimensions.insert(matrix_dimensions.len() - 1, axis_extent);
    let matrix_shape = Shape::new(matrix_dimensions);
    let mask_shape = Shape::new([axis_extent, axis_extent]);
    let range = lazy_arange_default_int_plan(
        0,
        i64::try_from(axis_extent).map_err(|_| Error::ShapeOverflow(transposed_shape.clone()))?,
        1,
    )?;
    let cumulative_max = cumulative_extrema_plan(input, &transposed_shape, dtype, -1)?;

    // `exp` is literally a source-width cast, F32-or-wider scale/multiply,
    // EXP2, then a source-width result.  `log` is LOG2 followed by a
    // source-width ln(2) multiply.  Model those internal boundaries here so
    // an invalid late stage cannot leave the graph partially published.
    let exp_source_dtype = unary_dtype(UnaryOp::Exp, dtype);
    let exp_work_dtype = exp_source_dtype.promote(DType::F32);
    let sum_dtypes = ReductionDType::sum_default(exp_source_dtype);
    let log_dtype = unary_dtype(UnaryOp::Log2, sum_dtypes.output);
    let output_dtype = reduction_source_lub(log_dtype, dtype);
    let minimum = TensorData::scalar_with_dtype(max_reduction_identity(dtype), dtype);

    for (descriptor, storage) in [
        (shape, dtype),
        (&transposed_shape, dtype),
        (&matrix_shape, dtype), // both unsqueezes, subtraction, Select result
        (&matrix_shape, DType::Bool),
        (&mask_shape, DType::Bool),
        (&range.shape, range.dtype),
        (&mask_shape, range.dtype), // the two reshaped range operands
        (&matrix_shape, exp_source_dtype),
        (&matrix_shape, exp_work_dtype),
        (&matrix_shape, sum_dtypes.accumulator),
        (&transposed_shape, sum_dtypes.accumulator),
        (&transposed_shape, sum_dtypes.output),
        (&transposed_shape, log_dtype),
        (&transposed_shape, output_dtype),
    ] {
        extent(descriptor, storage)?;
    }
    extent(minimum.shape(), minimum.dtype())?;
    // Scalars published by Exp and Log, plus the source-typed Select fallback.
    extent(&Shape::new([]), exp_work_dtype)?;
    extent(&Shape::new([]), log_dtype)?;
    if minimum.dtype() != dtype
        || transposed_shape.broadcast_with(minimum.shape())? != transposed_shape
        || matrix_shape.broadcast_with(&mask_shape)? != matrix_shape
        || matrix_shape.broadcast_with(minimum.shape())? != matrix_shape
        || reduction_source_lub(dtype, dtype) != dtype
        || reduction_source_lub(log_dtype, dtype) != output_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "logcumsumexp source promotion",
            actual: output_dtype,
        });
    }
    Ok(LogCumSumExpPlan {
        axis,
        transposed_shape,
        matrix_shape,
        source_dtype: dtype,
        exp_source_dtype,
        exp_work_dtype,
        sum_dtypes,
        log_dtype,
        output_dtype,
        cumulative_max,
        range,
        minimum,
    })
}

fn cumulative_extrema_plan(
    input: NodeId,
    shape: &Shape,
    dtype: DType,
    axis: isize,
) -> Result<CumExtremaPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(shape, dtype)?;
    extent(shape, DType::I32)?;
    if shape.rank() == 0 {
        if !matches!(axis, -1 | 0) {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: 0,
            });
        }
        return Ok(CumExtremaPlan {
            axis: 0,
            scalar: true,
            extent: 1,
            prefixes: Vec::new(),
            ascending: None,
            descending: None,
            index_offset: None,
        });
    }
    let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
    let axis_extent = shape.dims()[axis];
    let axis_i64 = i64::try_from(axis_extent).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
    let ascending = lazy_arange_default_int_plan(0, axis_i64, 1)?;
    let descending = lazy_arange_default_int_plan(axis_i64, 0, -1)?;
    debug_assert_eq!(ascending.dtype, descending.dtype);
    let index_dtype = descending.dtype;
    let mut transposed = shape.dims().to_vec();
    transposed.swap(axis, shape.rank() - 1);
    let transposed_shape = Shape::new(transposed);
    let mut matrix = transposed_shape.dims().to_vec();
    matrix.push(axis_extent);
    let matrix_shape = Shape::new(matrix);
    let prefixes = (0..axis_extent)
        .map(|end| {
            shape
                .dims()
                .iter()
                .enumerate()
                .map(|(dimension, &dimension_extent)| {
                    if dimension == axis {
                        (0, end + 1)
                    } else {
                        (0, dimension_extent)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Validate every source, movement, predicate, typed-range, reduction,
    // cast, and pair-output extent before any prefix node is appended.
    extent(&transposed_shape, dtype)?;
    extent(&matrix_shape, DType::Bool)?;
    extent(&matrix_shape, index_dtype)?;
    extent(&transposed_shape, index_dtype)?;
    extent(&transposed_shape, DType::I32)?;
    for bounds in &prefixes {
        let prefix = Shape::new(
            bounds
                .iter()
                .map(|&(start, end)| end - start)
                .collect::<Vec<_>>(),
        );
        extent(&prefix, dtype)?;
    }
    Ok(CumExtremaPlan {
        axis,
        scalar: false,
        extent: axis_extent,
        prefixes,
        ascending: Some(ascending),
        descending: Some(descending),
        index_offset: Some(TensorData::scalar_with_dtype(
            Scalar::I(axis_i64),
            index_dtype,
        )),
    })
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
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(f64::NEG_INFINITY),
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
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(f64::INFINITY),
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
        if axes
            .as_ref()
            .is_some_and(|axes| axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0)))
        {
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
        if axes
            .as_ref()
            .is_some_and(|axes| axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0)))
        {
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
        if axes
            .as_ref()
            .is_some_and(|axes| axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0)))
        {
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
        if axes
            .as_ref()
            .is_some_and(|axes| axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0)))
        {
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
        if axes
            .as_ref()
            .is_some_and(|axes| axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0)))
        {
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
    let axis_extent =
        i64::try_from(axis_extent).map_err(|_| Error::ShapeOverflow(work_shape.clone()))?;
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
            DType::Bool
            | DType::F8E4M3
            | DType::F8E5M2
            | DType::F8E4M3FNUZ
            | DType::F8E5M2FNUZ
            | DType::F16
            | DType::BF16
            | DType::F32
            | DType::F64 => unreachable!(),
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
        if axes
            .as_ref()
            .is_some_and(|axes| axes.len() > 1 || axes.iter().any(|axis| !matches!(axis, -1 | 0)))
        {
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
        let output_shape = reduction_shape(&shape, &axes, keepdim);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        // Preflight every descriptor in the cast -> reduce -> final-cast
        // contract before publishing its first node.
        extent(&shape, input_dtype)?;
        extent(&shape, dtypes.accumulator)?;
        extent(&output_shape, dtypes.accumulator)?;
        extent(&output_shape, dtypes.output)?;
        let accumulator = if input_dtype == dtypes.accumulator {
            input
        } else {
            self.cast(input, dtypes.accumulator)?
        };
        let reduced = self.reduce_with_output_dtype(
            accumulator,
            kind,
            Some(axes.into_iter().map(|axis| axis as isize).collect()),
            keepdim,
            dtypes.accumulator,
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

    /// Checked-in tinygrad's typed `Tensor.sum(axis, keepdim, dtype)`.
    /// An explicit dtype is both the reduction accumulator and result;
    /// otherwise Sum uses its source accumulator policy and narrows only the
    /// narrow floating result back to its original storage dtype.
    pub fn sum_with_options(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        let input_dtype = self.dtype(input)?;
        let dtypes = dtype
            .map(|dtype| ReductionDType::new(dtype, dtype))
            .unwrap_or_else(|| ReductionDType::sum_default(input_dtype));
        self.reduce_with_dtypes(input, ReduceKind::Sum, axes, keepdim, dtypes)
    }

    /// Checked-in tinygrad's default `Tensor.sum()` surface.
    pub fn sum_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.sum_with_options(input, None, false, None)
    }

    /// Checked-in tinygrad's typed `Tensor.prod(axis, keepdim, dtype)`.
    /// Product never applies Sum's default widening/narrowing policy.
    pub fn prod_with_options(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        let input_dtype = self.dtype(input)?;
        let dtypes = dtype
            .map(|dtype| ReductionDType::new(dtype, dtype))
            .unwrap_or_else(|| ReductionDType::product_default(input_dtype));
        self.reduce_with_dtypes(input, ReduceKind::Product, axes, keepdim, dtypes)
    }

    /// Checked-in tinygrad's default `Tensor.prod()` surface.
    pub fn prod_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.prod_with_options(input, None, false, None)
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

    /// Checked-in tinygrad's `Tensor.cumsum()` default: cumulative Sum along
    /// the leading axis.
    pub fn cumsum_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.cumsum(input, 0)
    }

    /// Tinygrad's literal stable cumulative log-sum-exp.  A scalar is an
    /// identity before its axis is inspected; non-scalars transpose the scan
    /// axis trailing, detach the cumulative maximum, and use a graph-resident
    /// lower triangle to exclude future terms with the source dtype's minimum.
    pub fn logcumsumexp(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        // This is deliberately before signed-axis validation: tinygrad's
        // source returns a rank-zero input directly, including for otherwise
        // invalid axis values.
        if shape.rank() == 0 {
            return Ok(input);
        }
        let plan = logcumsumexp_plan(input, &shape, dtype, axis)?;

        let transposed = self.transpose(input, plan.axis as isize, -1)?;
        let (cumulative_max, _indices) = self.lower_cummax(transposed, &plan.cumulative_max)?;
        let cumulative_max = self.detach(cumulative_max)?;
        let values = self.unsqueeze(transposed, -2)?;
        let maxima = self.unsqueeze(cumulative_max, -1)?;
        let shifted = self.sub(values, maxima)?;

        // `ones(n, n, dtype=bool).tril()` without a dense n² payload.
        let row_range = self.lower_lazy_arange(plan.range.clone())?;
        let row = self.reshape(row_range, Shape::new([plan.range.shape.dims()[0], 1]))?;
        let column_range = self.lower_lazy_arange(plan.range.clone())?;
        let column = self.reshape(column_range, Shape::new([1, plan.range.shape.dims()[0]]))?;
        let lower = self.ge(row, column)?;
        let minimum = self.constant(plan.minimum);
        let masked = self.select(lower, shifted, minimum)?;
        let exponentiated = self.exp(masked)?;
        let summed = self.reduce_with_dtypes(
            exponentiated,
            ReduceKind::Sum,
            Some(vec![-1]),
            false,
            plan.sum_dtypes,
        )?;
        let logged = self.log(summed)?;
        let output = self.add(logged, cumulative_max)?;
        let output = self.transpose(output, -1, plan.axis as isize)?;
        debug_assert_eq!(
            self.shape(output).expect("logcumsumexp preflighted"),
            &shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("logcumsumexp preflighted"),
            plan.output_dtype
        );
        debug_assert_eq!(
            self.shape(transposed).expect("logcumsumexp preflighted"),
            &plan.transposed_shape
        );
        debug_assert_eq!(
            self.shape(masked).expect("logcumsumexp preflighted"),
            &plan.matrix_shape
        );
        debug_assert_eq!(
            self.dtype(masked).expect("logcumsumexp preflighted"),
            plan.source_dtype
        );
        debug_assert_eq!(
            self.dtype(exponentiated).expect("logcumsumexp preflighted"),
            plan.exp_source_dtype
        );
        debug_assert_eq!(
            self.dtype(logged).expect("logcumsumexp preflighted"),
            plan.log_dtype
        );
        debug_assert_eq!(
            plan.exp_source_dtype.promote(DType::F32),
            plan.exp_work_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.logcumsumexp()` default leading axis.
    pub fn logcumsumexp_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.logcumsumexp(input, 0)
    }

    fn lower_cummax(&mut self, input: NodeId, plan: &CumExtremaPlan) -> Result<(NodeId, NodeId)> {
        if plan.scalar {
            let indices = self.lazy_full_with_dtype([], Scalar::I(0), DType::I32)?;
            return Ok((input, indices));
        }
        let values = if plan.extent == 0 {
            input
        } else {
            let prefixes = plan
                .prefixes
                .iter()
                .map(|bounds| {
                    let prefix = self.shrink(input, bounds.clone())?;
                    self.max_with_axes(prefix, Some(vec![plan.axis as isize]), true)
                })
                .collect::<Result<Vec<_>>>()?;
            self.concat(prefixes, plan.axis)?
        };

        // Literal checked-in tinygrad index path: transpose to trailing axis,
        // equality with the cumulative values, upper-triangle mask, descending
        // source-default range, Max, Neg/Add(n), I32 cast, then transpose back.
        let x_t = self.transpose(input, plan.axis as isize, -1)?;
        let values_t = self.transpose(values, plan.axis as isize, -1)?;
        let x_t = self.unsqueeze(x_t, -1)?;
        let values_t = self.unsqueeze(values_t, -2)?;
        let equality = self.eq(x_t, values_t)?;
        let ascending = plan.ascending.as_ref().expect("non-scalar plan");
        let row_range = self.lower_lazy_arange(ascending.clone())?;
        let row = self.reshape(row_range, Shape::new([plan.extent, 1]))?;
        let column_range = self.lower_lazy_arange(ascending.clone())?;
        let column = self.reshape(column_range, Shape::new([1, plan.extent]))?;
        let upper = self.le(row, column)?;
        let matched = self.mul(equality, upper)?;
        let descending =
            self.lower_lazy_arange(plan.descending.clone().expect("non-scalar plan"))?;
        let descending = self.reshape(descending, Shape::new([plan.extent, 1]))?;
        let candidates = self.mul(matched, descending)?;
        let maximum = self.max_with_axes(candidates, Some(vec![-2]), false)?;
        let negative = self.neg(maximum)?;
        let index_offset = self.constant(plan.index_offset.clone().expect("non-scalar plan"));
        let indices_t = self.add(negative, index_offset)?;
        let indices_t = self.cast(indices_t, DType::I32)?;
        let indices = self.transpose(indices_t, -1, plan.axis as isize)?;
        debug_assert_eq!(
            self.shape(values).expect("cummax preflighted"),
            self.shape(input).expect("cummax preflighted")
        );
        debug_assert_eq!(
            self.dtype(values).expect("cummax preflighted"),
            self.dtype(input).expect("cummax preflighted")
        );
        debug_assert_eq!(
            self.shape(indices).expect("cummax preflighted"),
            self.shape(input).expect("cummax preflighted")
        );
        debug_assert_eq!(self.dtype(indices).expect("cummax preflighted"), DType::I32);
        Ok((values, indices))
    }

    /// Checked-in tinygrad's literal `Tensor.cummax(axis) -> (values, indices)`.
    pub fn cummax(&mut self, input: NodeId, axis: isize) -> Result<(NodeId, NodeId)> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let plan = cumulative_extrema_plan(input, &shape, dtype, axis)?;
        self.lower_cummax(input, &plan)
    }

    /// Checked-in tinygrad's `Tensor.cummax()` default leading axis.
    pub fn cummax_default(&mut self, input: NodeId) -> Result<(NodeId, NodeId)> {
        self.cummax(input, 0)
    }

    /// Checked-in tinygrad's literal inverse → CumMax → inverse CumMin pair.
    pub fn cummin(&mut self, input: NodeId, axis: isize) -> Result<(NodeId, NodeId)> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let plan = cumulative_extrema_plan(input, &shape, dtype, axis)?;
        let inverse = if dtype.is_float() {
            self.neg(input)?
        } else {
            self.bitwise_not(input)?
        };
        let (values, indices) = self.lower_cummax(inverse, &plan)?;
        let values = if dtype.is_float() {
            self.neg(values)?
        } else {
            self.bitwise_not(values)?
        };
        Ok((values, indices))
    }

    /// Checked-in tinygrad's `Tensor.cummin()` default leading axis.
    pub fn cummin_default(&mut self, input: NodeId) -> Result<(NodeId, NodeId)> {
        self.cummin(input, 0)
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
        let plan = cumulative_product_plan(input, &shape, dtype, axis)?;
        let Some(axis) = plan.axis else {
            return Ok(input);
        };
        let dtypes = plan.dtypes;

        let values = plan
            .prefixes
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
        debug_assert_eq!(
            self.shape(output).expect("all preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(self.dtype(output).expect("all preflighted"), DType::Bool);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.all()` defaults: all axes, with reduced
    /// dimensions omitted.
    pub fn all_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.all(input, None, false)
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
            AnyLowering::Reduce => {
                self.reduce(boolean, crate::ReduceKind::Max, Some(plan.axes), keepdim)?
            }
        };
        debug_assert_eq!(
            self.shape(output).expect("any preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(self.dtype(output).expect("any preflighted"), DType::Bool);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.any()` defaults: all axes, with reduced
    /// dimensions omitted.
    pub fn any_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.any(input, None, false)
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
            self.full_with_dtype(
                plan.output_shape.clone(),
                Scalar::I(i32::MIN.into()),
                DType::I32,
            )?
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
        debug_assert_eq!(
            self.shape(output).expect("argmax preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(self.dtype(output).expect("argmax preflighted"), DType::I32);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.argmax()` defaults: flatten all source
    /// dimensions and omit reduced dimensions from the scalar I32 result.
    pub fn argmax_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.argmax_with_axis(input, None, false)
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
        debug_assert_eq!(
            self.shape(output).expect("argmin preflighted"),
            &plan.argmax.output_shape
        );
        debug_assert_eq!(self.dtype(output).expect("argmin preflighted"), DType::I32);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.argmin()` defaults: source-width inverse
    /// followed by flattened first-tie ArgMax.
    pub fn argmin_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.argmin_with_axis(input, None, false)
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
        debug_assert_eq!(
            self.shape(output).expect("Mean preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("Mean preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.mean()` defaults: all axes, with reduced
    /// dimensions omitted.
    pub fn mean_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.mean_with_axes(input, None, false)
    }

    /// Checked-in tinygrad `Tensor.layernorm(axis, eps)` over signed axes.
    /// This is deliberately the public scalar-free tensor composition, not
    /// the affine `nn::LayerNorm` module or ONNX multi-output operator.
    pub fn layernorm_with_axes(
        &mut self,
        input: NodeId,
        axes: Vec<isize>,
        eps: f64,
    ) -> Result<NodeId> {
        let plan = layernorm_plan(self, input, axes, eps)?;
        let mean = self.mean_with_axes(input, Some(plan.axes.clone()), true)?;
        let centered = self.sub(input, mean)?;
        let squared = self.mul(centered, centered)?;
        let variance = self.mean_with_axes(squared, Some(plan.axes.clone()), true)?;
        let epsilon = self.constant(plan.epsilon);
        let invstd = self.add(variance, epsilon)?;
        let invstd = self.rsqrt(invstd)?;
        let output = self.mul(centered, invstd)?;
        debug_assert_eq!(
            self.shape(mean).expect("layernorm preflighted"),
            &plan.mean_shape
        );
        debug_assert_eq!(
            self.dtype(mean).expect("layernorm preflighted"),
            plan.mean_dtype
        );
        debug_assert_eq!(
            self.shape(centered).expect("layernorm preflighted"),
            &plan.centered_shape
        );
        debug_assert_eq!(
            self.dtype(centered).expect("layernorm preflighted"),
            plan.centered_dtype
        );
        debug_assert_eq!(
            self.shape(variance).expect("layernorm preflighted"),
            &plan.variance_shape
        );
        debug_assert_eq!(
            self.dtype(variance).expect("layernorm preflighted"),
            plan.variance_dtype
        );
        debug_assert_eq!(
            self.shape(output).expect("layernorm preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("layernorm preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Signed single-axis convenience matching tinygrad's integer `axis`
    /// argument. Use [`Self::layernorm_with_axes`] for tuple-style axes.
    pub fn layernorm(&mut self, input: NodeId, axis: isize, eps: f64) -> Result<NodeId> {
        self.layernorm_with_axes(input, vec![axis], eps)
    }

    /// tinygrad's omitted LayerNorm arguments: final axis and `1e-5` eps.
    pub fn layernorm_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.layernorm(input, -1, 1e-5)
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
        debug_assert_eq!(
            self.shape(output).expect("Max preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(self.dtype(output).expect("Max preflighted"), plan.dtype);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.max()` defaults: all axes, with reduced
    /// dimensions omitted.
    pub fn max_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.max_with_axes(input, None, false)
    }

    /// Source-faithful public tinygrad-style Min over signed optional axes.
    /// tinygrad spells Min literally as inverse-Max-inverse. Floating values
    /// invert with Neg; Bool/integer values invert with bitwise-not.
    pub fn min_with_axes(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let (input_shape, input_dtype) = {
            let input_node = self.node(input)?;
            (input_node.shape.clone(), input_node.dtype)
        };
        let plan = min_plan(input, &input_shape, input_dtype, axes.clone(), keepdim)?;
        // Preflight the exact source middle stage before either inverse can
        // publish. Both inverses preserve the concrete descriptor, so this
        // validates all source/intermediate/output extents atomically.
        let max_plan = max_plan(input, &input_shape, input_dtype, axes.clone(), keepdim)?;
        debug_assert_eq!(plan.output_shape, max_plan.output_shape);
        debug_assert_eq!(plan.dtype, max_plan.dtype);
        let inverse = if input_dtype.is_float() {
            self.neg(input)?
        } else {
            self.bitwise_not(input)?
        };
        let maximum = self.max_with_axes(inverse, axes, keepdim)?;
        let output = if input_dtype.is_float() {
            self.neg(maximum)?
        } else {
            self.bitwise_not(maximum)?
        };
        debug_assert_eq!(
            self.shape(output).expect("Min preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(self.dtype(output).expect("Min preflighted"), plan.dtype);
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.min()` defaults: all axes, with reduced
    /// dimensions omitted.
    pub fn min_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.min_with_axes(input, None, false)
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
        debug_assert_eq!(
            self.shape(output).expect("variance preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.shape(mean).expect("variance preflighted"),
            &plan.mean_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("variance preflighted"),
            plan.output_dtype
        );
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
        let variance_plan = variance_plan(input, &shape, dtype, axes.clone(), keepdim, correction)?;
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
        let variance_plan = variance_plan(input, &shape, dtype, axes.clone(), keepdim, correction)?;
        let mean_plan = mean_plan(input, &shape, dtype, axes.clone(), keepdim)?;

        // `variance_plan` establishes a floating result descriptor.  The
        // public sqrt helper is homogeneous for that descriptor and validates
        // its same-shape, same-dtype output before it appends its raw unary.
        debug_assert!(variance_plan.output_dtype.is_float());
        let standard_deviation = self.std(input, axes.clone(), keepdim, correction)?;
        let mean = self.mean_with_axes(input, axes, keepdim)?;
        debug_assert_eq!(
            self.shape(standard_deviation)
                .expect("std_mean preflighted"),
            &variance_plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(standard_deviation)
                .expect("std_mean preflighted"),
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

    /// Checked-in tinygrad `Tensor.normalize(p, dim, eps)`.
    ///
    /// `p` is a Python float, so each literal Pow scalar is committed at the
    /// source floating work width before the raw homogeneous Pow node. This
    /// leaves the broader live-integer Pow contract untouched.
    pub fn normalize(&mut self, input: NodeId, p: f64, dim: isize, eps: f64) -> Result<NodeId> {
        let plan = normalize_plan(self, input, p, dim, eps)?;
        let denominator = match plan.lowering {
            NormalizeLowering::Zero { sum } => {
                let nonzero = self.ne_scalar(input, Scalar::I(0))?;
                self.reduce_with_dtypes(
                    nonzero,
                    ReduceKind::Sum,
                    Some(plan.axes.clone()),
                    true,
                    sum,
                )?
            }
            NormalizeLowering::Pow {
                pow_dtype,
                sum,
                exponent,
                reciprocal_exponent,
            } => {
                let absolute = self.abs(input)?;
                let base = if self.dtype(absolute)? == pow_dtype {
                    absolute
                } else {
                    self.cast(absolute, pow_dtype)?
                };
                let exponent = self.constant(exponent);
                let powers = self.binary(crate::BinaryOp::Pow, base, exponent)?;
                let summed = self.reduce_with_dtypes(
                    powers,
                    ReduceKind::Sum,
                    Some(plan.axes.clone()),
                    true,
                    sum,
                )?;
                let base = if self.dtype(summed)? == plan.denominator_dtype {
                    summed
                } else {
                    self.cast(summed, plan.denominator_dtype)?
                };
                let exponent = self.constant(reciprocal_exponent);
                self.binary(crate::BinaryOp::Pow, base, exponent)?
            }
        };
        // Keep denominator as the ordered lhs of `den.maximum(eps)`. The
        // scalar helper performs the source-LUB cast needed by p==0's I32
        // Bool-count before its F32 weak epsilon is published.
        let _ = plan.epsilon;
        let denominator = self.maximum_scalar(denominator, Scalar::F(eps))?;
        let output = self.div(input, denominator)?;
        debug_assert_eq!(
            self.shape(denominator).expect("normalize preflighted"),
            &plan.denominator_shape
        );
        debug_assert_eq!(
            self.dtype(denominator).expect("normalize preflighted"),
            plan.denominator_dtype
        );
        debug_assert_eq!(
            self.shape(output).expect("normalize preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("normalize preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// tinygrad's omitted normalize arguments: `p=2`, `dim=1`, `eps=1e-12`.
    pub fn normalize_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.normalize(input, 2.0, 1, 1e-12)
    }
}

fn variance_denominator(
    count: usize,
    correction: VarianceCorrection,
    shape: &crate::Shape,
) -> Result<usize> {
    let correction = correction.value();
    if correction >= 0 {
        let correction =
            usize::try_from(correction).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
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
            dtypes == ReductionDType::sum_default(input) || dtypes.accumulator == dtypes.output
        }
        ReduceKind::Product => dtypes.accumulator == dtypes.output,
        ReduceKind::Mean
        | ReduceKind::Max
        | ReduceKind::Min
        | ReduceKind::Any
        | ReduceKind::All => false,
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
        let bindings = HashMap::from([("input".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.]))]);
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
        let bindings = HashMap::from([("input".into(), data([2, 2], &[1., -2., f32::NAN, 4.]))]);
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
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
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
        let bindings = HashMap::from([("input".into(), data([2, 2], &[0., 0., f32::NAN, 0.]))]);
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
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
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
    fn boolean_reduction_defaults_keep_tinygrad_truthiness_roots_and_atomicity() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F64);
        let any = graph.any_default(input).unwrap();
        let all = graph.all_default(input).unwrap();
        assert_eq!(graph.shape(any).unwrap(), &Shape::new([]));
        assert_eq!(graph.shape(all).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(any).unwrap(), DType::Bool);
        assert_eq!(graph.dtype(all).unwrap(), DType::Bool);
        assert!(matches!(
            graph.op(any).unwrap(),
            crate::Op::Reduce {
                kind: ReduceKind::Max,
                ..
            }
        ));
        assert!(matches!(
            graph.op(all).unwrap(),
            crate::Op::Reduce {
                kind: ReduceKind::Product,
                ..
            }
        ));
        assert!(matches!(graph.grad(any, input), Err(Error::NoGradient(_))));
        assert!(matches!(graph.grad(all, input), Err(Error::NoGradient(_))));
        let bindings = HashMap::from([(
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
        )]);
        assert_eq!(
            CpuBackend
                .execute(&graph, any, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![1.]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, all, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![0.]
        );

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::I32);
        let any = scalar.any_default(input).unwrap();
        let all = scalar.all_default(input).unwrap();
        assert_eq!(scalar.shape(any).unwrap(), &Shape::new([]));
        assert_eq!(scalar.shape(all).unwrap(), &Shape::new([]));
        assert_eq!(scalar.dtype(any).unwrap(), DType::Bool);
        assert_eq!(scalar.dtype(all).unwrap(), DType::Bool);

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 2], DType::F32);
        let any = empty.any_default(input).unwrap();
        let all = empty.all_default(input).unwrap();
        let bindings = HashMap::from([("input".into(), data([0, 2], &[]))]);
        assert_eq!(
            CpuBackend
                .execute(&empty, any, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![0.]
        );
        assert_eq!(
            CpuBackend
                .execute(&empty, all, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![1.]
        );

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(overflow.any_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
        assert!(overflow.all_default(input).is_err());
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
                    Scalar::F(f64::NAN),
                    Scalar::F(2.0),
                    Scalar::F(3.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(-1.0),
                    Scalar::F(1.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(3.0),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.to_vec_f64(), vec![3., 0., 2.]);
        assert!(matches!(
            graph.grad(output, input),
            Err(Error::NoGradient(_))
        ));

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
                    Scalar::F(f64::NAN),
                    Scalar::F(2.0),
                    Scalar::F(-3.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(1.0),
                    Scalar::F(3.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(-1.0),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.to_vec_f64(), vec![3., 0., 2.]);
        assert!(matches!(
            graph.grad(output, input),
            Err(Error::NoGradient(_))
        ));

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
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(i64::MIN), Scalar::I(-1)])
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
    fn argextrema_default_wrappers_preserve_flattened_i32_sentinel_structure() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F64);
        let maximum = graph.argmax_default(input).unwrap();
        let minimum = graph.argmin_default(input).unwrap();
        for output in [maximum, minimum] {
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
            assert_eq!(graph.dtype(output).unwrap(), DType::I32);
            // The explicit source plan keeps first-tie ArgReduce and its
            // leading-NaN sentinel as a Select over the flattened operand.
            assert!(matches!(
                graph.op(output).unwrap(),
                crate::Op::Select { .. }
            ));
            assert!(matches!(
                graph.grad(output, input),
                Err(Error::NoGradient(_))
            ));
        }
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Reshape { shape, .. } if shape == &Shape::new([6])
        )));
        assert!(graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                crate::Op::Constant(data) => Some((data.dtype(), data.scalar_at(0))),
                _ => None,
            })
            .any(|(dtype, value)| dtype == DType::I32 && value.as_i64() == 6));
        // ArgMin is source-literal inverse → ArgMax, not raw ArgMin.
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Unary { op: UnaryOp::Neg, input: source } if *source == input
        )));

        let scalar = graph.input_dtype("scalar", [], DType::U64);
        let scalar_max = graph.argmax_default(scalar).unwrap();
        let scalar_min = graph.argmin_default(scalar).unwrap();
        assert_eq!(graph.shape(scalar_max).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(scalar_min).unwrap(), DType::I32);

        let mut empty = Graph::new();
        let empty_input = empty.input_dtype("empty", [0, 3], DType::I16);
        let empty_max = empty.argmax_default(empty_input).unwrap();
        let empty_min = empty.argmin_default(empty_input).unwrap();
        assert_eq!(empty.shape(empty_max).unwrap(), &Shape::new([]));
        assert_eq!(empty.shape(empty_min).unwrap(), &Shape::new([]));
        assert_eq!(empty.dtype(empty_max).unwrap(), DType::I32);
        assert_eq!(empty.dtype(empty_min).unwrap(), DType::I32);

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("overflow", [usize::MAX, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.argmax_default(input),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
        assert!(matches!(
            overflow.argmin_default(input),
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
            CpuBackend
                .execute(&graph, cumulative, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
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
    fn cumsum_default_uses_the_leading_typed_sum_prefix_contract() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3, 2], DType::F16);
        let output = graph.cumsum_default(input).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([3, 2]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        let crate::Op::Concat { inputs, axis } = graph.op(output).unwrap() else {
            panic!("expected cumsum prefix concatenation");
        };
        assert_eq!(*axis, 0);
        assert_eq!(inputs.len(), 3);
        for prefix in inputs {
            let reduced = match graph.op(*prefix).unwrap() {
                crate::Op::Cast {
                    input,
                    dtype: DType::F16,
                } => *input,
                op => panic!("expected default F16 prefix narrowing, got {op:?}"),
            };
            assert!(matches!(
                graph.op(reduced).unwrap(),
                crate::Op::Reduce {
                    kind: ReduceKind::Sum,
                    axes,
                    keepdim: true,
                    ..
                } if axes == &vec![0]
            ));
            assert_eq!(graph.dtype(reduced).unwrap(), DType::F32);
        }

        let scalar = graph.input_dtype("scalar", [], DType::I8);
        let scalar_output = graph.cumsum_default(scalar).unwrap();
        assert_eq!(graph.dtype(scalar_output).unwrap(), DType::I32);

        let mut empty = Graph::new();
        let empty_input = empty.input_dtype("empty", [0], DType::BF16);
        let empty_output = empty.cumsum_default(empty_input).unwrap();
        assert_eq!(empty.shape(empty_output).unwrap(), &Shape::from([0]));
        assert_eq!(empty.dtype(empty_output).unwrap(), DType::BF16);

        let mut overflow = Graph::new();
        let overflow_input = overflow.input("input", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(matches!(
            overflow.cumsum_default(overflow_input),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn cumextrema_defaults_keep_tinygrad_pair_structure_and_descriptors() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let (values, indices) = graph.cummax_default(input).unwrap();
        assert_eq!(graph.shape(values).unwrap(), &Shape::from([2, 3]));
        assert_eq!(graph.dtype(values).unwrap(), DType::F32);
        assert_eq!(graph.shape(indices).unwrap(), &Shape::from([2, 3]));
        assert_eq!(graph.dtype(indices).unwrap(), DType::I32);
        assert!(graph.requires_grad(values).unwrap());
        assert!(!graph.requires_grad(indices).unwrap());
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Compare {
                op: crate::CompareOp::Eq,
                ..
            }
        )));
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Reduce {
                kind: ReduceKind::Max,
                ..
            }
        )));
        assert!(graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                crate::Op::Constant(data) => Some(data.len()),
                _ => None,
            })
            .all(|len| len == 1));

        let (minimum, minimum_indices) = graph.cummin_default(input).unwrap();
        assert_eq!(graph.dtype(minimum).unwrap(), DType::F32);
        assert_eq!(graph.dtype(minimum_indices).unwrap(), DType::I32);
        assert!(matches!(
            graph.op(minimum).unwrap(),
            crate::Op::Unary {
                op: crate::UnaryOp::Neg,
                ..
            }
        ));

        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let value = graph.input_dtype(format!("{dtype:?}_input"), [2], dtype);
            let (maximum, indices) = graph.cummax(value, -1).unwrap();
            assert_eq!(graph.dtype(maximum).unwrap(), dtype);
            assert_eq!(graph.dtype(indices).unwrap(), DType::I32);
        }

        let scalar = graph.input_dtype("scalar", [], DType::I16);
        let (scalar_values, scalar_indices) = graph.cummax_default(scalar).unwrap();
        assert_eq!(scalar_values, scalar);
        assert_eq!(graph.shape(scalar_indices).unwrap(), &Shape::from([]));
        assert_eq!(graph.dtype(scalar_indices).unwrap(), DType::I32);

        let mut empty = Graph::new();
        let input = empty.input_dtype("empty", [0, 2], DType::BF16);
        let (values, indices) = empty.cummax_default(input).unwrap();
        assert_eq!(empty.shape(values).unwrap(), &Shape::from([0, 2]));
        assert_eq!(empty.shape(indices).unwrap(), &Shape::from([0, 2]));
        assert_eq!(empty.dtype(indices).unwrap(), DType::I32);

        let mut invalid = Graph::new();
        let input = invalid.input("input", [2]);
        let before = invalid.node_count();
        assert!(invalid.cummin(input, 1).is_err());
        assert_eq!(invalid.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(matches!(
            overflow.cummax_default(input),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn logcumsumexp_keeps_the_literal_detached_cummax_and_lazy_triangle() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F16);
        let output = graph.logcumsumexp_default(input).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 3]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        assert!(graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, crate::Op::Detach { .. })));
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Compare {
                op: crate::CompareOp::Ge,
                ..
            }
        )));
        assert!(graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, crate::Op::Select { .. })));
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Reduce { kind: ReduceKind::Sum, axes, keepdim: false, .. }
                if axes == &vec![2]
        )));
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Reduce {
                kind: ReduceKind::Max,
                ..
            }
        )));
        // Both the CumMax indices and the stabilization mask are composed
        // from scalar-backed lazy ranges; no dense control tensor is allowed.
        assert!(graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                crate::Op::Constant(data) => Some(data.len()),
                _ => None,
            })
            .all(|len| len == 1));
        let loss = graph.sum_all(output).unwrap();
        assert!(graph.grad(loss, input).is_ok());

        let nonfloat = graph.input_dtype("nonfloat", [2], DType::I16);
        let nonfloat_output = graph.logcumsumexp(nonfloat, -1).unwrap();
        assert_eq!(graph.dtype(nonfloat_output).unwrap(), DType::F32);

        let scalar = graph.input_dtype("scalar", [], DType::F64);
        // Scalar identity happens before axis validation in the source.
        assert_eq!(graph.logcumsumexp(scalar, isize::MIN).unwrap(), scalar);

        let mut empty = Graph::new();
        let empty_input = empty.input_dtype("empty", [2, 0], DType::BF16);
        let empty_output = empty.logcumsumexp(empty_input, -1).unwrap();
        assert_eq!(empty.shape(empty_output).unwrap(), &Shape::from([2, 0]));
        assert_eq!(empty.dtype(empty_output).unwrap(), DType::BF16);

        let mut invalid = Graph::new();
        let invalid_input = invalid.input("invalid", [2, 3]);
        let before = invalid.node_count();
        assert!(invalid.logcumsumexp(invalid_input, 2).is_err());
        assert_eq!(invalid.node_count(), before);

        let mut overflow = Graph::new();
        let overflow_input = overflow.input_dtype("overflow", [usize::MAX, 2], DType::F32);
        let before = overflow.node_count();
        assert!(matches!(
            overflow.logcumsumexp_default(overflow_input),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn cumprod_matches_tinygrad_signed_axis_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let cumulative = graph.cumprod(input, -1).unwrap();
        assert!(matches!(
            graph.op(cumulative).unwrap(),
            crate::Op::Concat { axis: 1, .. }
        ));
        let loss = graph.sum_all(cumulative).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([(
            "input".into(),
            TensorData::new([2, 3], vec![2.0, 3.0, 4.0, 2.0, 0.0, -3.0]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend
                .execute(&graph, cumulative, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![2.0, 6.0, 24.0, 2.0, 0.0, 0.0]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
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

        // Product has no accumulator widening: source Bool/integer/floating
        // storage is retained in every checked prefix reduction.
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let mut typed = Graph::new();
            let input = typed.input_dtype("typed", [2, 2], dtype);
            let output = typed.cumprod(input, -1).unwrap();
            assert_eq!(typed.shape(output).unwrap(), &Shape::from([2, 2]));
            assert_eq!(typed.dtype(output).unwrap(), dtype);
            assert!(matches!(
                typed.op(output).unwrap(),
                crate::Op::Concat { axis: 1, .. }
            ));
        }

        // The descriptor-first plan rejects byte overflow before a Shrink or
        // Product reduction can be appended.
        let mut overflow = Graph::new();
        let input = overflow.input_dtype("overflow", [usize::MAX, 2], DType::F64);
        let before_nodes = overflow.node_count();
        assert!(matches!(
            overflow.cumprod(input, 0),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before_nodes);
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
    fn typed_sum_and_product_options_keep_their_requested_storage_contracts() {
        let mut graph = Graph::new();
        let narrow = graph.input_dtype("narrow", [2, 3], DType::F16);
        let default_sum = graph
            .sum_with_options(narrow, Some(vec![-1, 0]), true, None)
            .unwrap();
        assert_eq!(graph.shape(default_sum).unwrap(), &Shape::from([1, 1]));
        assert_eq!(graph.dtype(default_sum).unwrap(), DType::F16);
        let default_reduction = match graph.op(default_sum).unwrap() {
            crate::Op::Cast {
                input,
                dtype: DType::F16,
            } => *input,
            op => panic!("expected source-default F16 narrowing cast, got {op:?}"),
        };
        assert!(matches!(
            graph.op(default_reduction).unwrap(),
            crate::Op::Reduce {
                kind: ReduceKind::Sum,
                axes,
                keepdim: true,
                ..
            } if axes == &vec![0, 1]
        ));
        assert_eq!(graph.dtype(default_reduction).unwrap(), DType::F32);

        let integers = graph.input_dtype("integers", [2, 2], DType::I8);
        let narrow_sum = graph
            .sum_with_options(integers, Some(vec![1]), false, Some(DType::I8))
            .unwrap();
        assert_eq!(graph.dtype(narrow_sum).unwrap(), DType::I8);
        assert!(matches!(
            graph.op(narrow_sum).unwrap(),
            crate::Op::Reduce {
                kind: ReduceKind::Sum,
                ..
            }
        ));

        let widened_sum = graph
            .sum_with_options(integers, None, false, Some(DType::F64))
            .unwrap();
        assert_eq!(graph.dtype(widened_sum).unwrap(), DType::F64);
        let widened_sum_input = match graph.op(widened_sum).unwrap() {
            crate::Op::Reduce { input, .. } => *input,
            op => panic!("expected F64 Sum reduction, got {op:?}"),
        };
        assert!(matches!(
            graph.op(widened_sum_input).unwrap(),
            crate::Op::Cast {
                dtype: DType::F64,
                ..
            }
        ));

        let default_product = graph.prod_default(integers).unwrap();
        assert_eq!(graph.dtype(default_product).unwrap(), DType::I8);
        assert!(matches!(
            graph.op(default_product).unwrap(),
            crate::Op::Reduce {
                kind: ReduceKind::Product,
                ..
            }
        ));
        let explicit_product = graph
            .prod_with_options(integers, Some(vec![-1]), true, Some(DType::U16))
            .unwrap();
        assert_eq!(graph.shape(explicit_product).unwrap(), &Shape::from([2, 1]));
        assert_eq!(graph.dtype(explicit_product).unwrap(), DType::U16);

        let scalar = graph.input_dtype("scalar", [], DType::BF16);
        let scalar_sum = graph.sum_default(scalar).unwrap();
        let scalar_product = graph.prod_default(scalar).unwrap();
        assert_eq!(graph.dtype(scalar_sum).unwrap(), DType::BF16);
        assert_eq!(graph.dtype(scalar_product).unwrap(), DType::BF16);

        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let input = graph.input_dtype(format!("{dtype:?}_input"), [2], dtype);
            let sum = graph.sum_default(input).unwrap();
            let product = graph.prod_default(input).unwrap();
            let explicit = graph
                .sum_with_options(input, None, false, Some(DType::F64))
                .unwrap();
            assert_eq!(
                graph.dtype(sum).unwrap(),
                ReductionDType::sum_default(dtype).output
            );
            assert_eq!(graph.dtype(product).unwrap(), dtype);
            assert_eq!(graph.dtype(explicit).unwrap(), DType::F64);
        }

        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 2]);
        let before = malformed.node_count();
        assert!(matches!(
            malformed.sum_with_options(input, Some(vec![0, 0]), false, Some(DType::I8)),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(matches!(
            overflow.prod_with_options(input, None, false, Some(DType::U64)),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn variance_and_std_match_tinygrad_correction_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [3]);
        let default_variance = graph.var(input, None, false, None).unwrap();
        let population_variance = graph
            .var(input, None, false, Some(VarianceCorrection::new(0)))
            .unwrap();
        let negative_correction = graph
            .var(input, None, false, Some(VarianceCorrection::new(-1)))
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
            CpuBackend
                .execute(&graph, default_variance, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![4.]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, population_variance, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![(8.0f32 / 3.0) as f64]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, negative_correction, &inputs)
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
            CpuBackend
                .execute(&graph, default_gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![-2., 0., 2.]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &inputs)
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
            matches!(
                &node.op,
                crate::Op::Reduce {
                    kind: ReduceKind::Sum,
                    ..
                }
            ) && node.dtype == DType::F32
        }));
        let f16_data =
            TensorData::from_scalars([2], DType::F16, [Scalar::F(1.5), Scalar::F(2.5)]).unwrap();
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
                        TensorData::from_scalars([2], DType::I32, [Scalar::I(1), Scalar::I(3)],)
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

        let variance_loss = graph.sum_all(variance).unwrap();
        let mean_loss = graph.sum_all(mean).unwrap();
        let loss = graph.add(variance_loss, mean_loss).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([("input".into(), data([2, 2], &[1., 3., 5., 7.]))]);
        assert_eq!(
            CpuBackend
                .execute(&graph, variance, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![1., 1.]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, mean, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![2., 6.]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
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
            .var_mean(input, None, false, Some(VarianceCorrection::new(-1)),)
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
        assert_eq!(
            graph.shape(standard_deviation).unwrap(),
            &Shape::new([2, 1])
        );
        assert_eq!(graph.shape(mean).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(standard_deviation).unwrap(), DType::F32);
        assert_eq!(graph.dtype(mean).unwrap(), DType::F32);

        let standard_deviation_loss = graph.sum_all(standard_deviation).unwrap();
        let mean_loss = graph.sum_all(mean).unwrap();
        let loss = graph.add(standard_deviation_loss, mean_loss).unwrap();
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
            CpuBackend
                .execute(&graph, mean, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![2., 6.]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![0., 1., 0., 1.]
        );

        let mut integer = Graph::new();
        let input = integer.input_dtype("input", [2], DType::U16);
        let (standard_deviation, mean) = integer.std_mean(input, None, false, None).unwrap();
        assert_eq!(integer.dtype(standard_deviation).unwrap(), DType::F32);
        assert_eq!(integer.dtype(mean).unwrap(), DType::F32);

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::F16);
        let (standard_deviation, mean) =
            scalar.std_mean(input, Some(vec![-1]), false, None).unwrap();
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
            .std_mean(input, None, false, Some(VarianceCorrection::new(-1)),)
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
            mean,
            variance,
            standard_deviation,
            pair_variance,
            pair_mean,
            pair_standard_deviation,
            pair_standard_mean,
        ] {
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
            assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        }
        // The paired forms stay literal `(var/std, mean)`, rather than
        // substituting variance's internal keepdim mean for the public mean.
        assert_ne!(pair_variance, pair_mean);
        assert_ne!(pair_standard_deviation, pair_standard_mean);
        assert!((0..graph.node_count()).any(|index| matches!(
            graph.op(NodeId(index)).unwrap(),
            crate::Op::Unary {
                op: UnaryOp::Square,
                ..
            }
        )));
        assert!(matches!(
            graph.op(pair_standard_deviation).unwrap(),
            crate::Op::Unary {
                op: UnaryOp::Sqrt,
                ..
            }
        ));
        let variance_loss = graph.sum_all(pair_variance).unwrap();
        let mean_loss = graph.sum_all(pair_mean).unwrap();
        let loss = graph.add(variance_loss, mean_loss).unwrap();
        assert!(graph.grad(loss, input).is_ok());

        let mut nonfloat = Graph::new();
        let input = nonfloat.input_dtype("input", [], DType::I16);
        let (variance, mean) = nonfloat.var_mean_default(input).unwrap();
        let (standard_deviation, std_mean) = nonfloat.std_mean_default(input).unwrap();
        for output in [
            nonfloat.mean_default(input).unwrap(),
            variance,
            mean,
            nonfloat.var_default(input).unwrap(),
            standard_deviation,
            std_mean,
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
            empty.mean_default(input).unwrap(),
            variance,
            mean,
            empty.var_default(input).unwrap(),
            standard_deviation,
            std_mean,
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
            overflow.var(input, None, false, Some(VarianceCorrection::new(-1)),),
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
            CpuBackend
                .execute(&graph, normalized, &inputs)
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
            CpuBackend
                .execute(&graph, gradient, &inputs)
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
            matches!(
                &node.op,
                crate::Op::Reduce {
                    kind: ReduceKind::Sum,
                    ..
                }
            ) && node.dtype == DType::F32
        }));
        assert_eq!(
            CpuBackend
                .execute(
                    &narrow,
                    normalized,
                    &HashMap::from([(
                        "input".into(),
                        TensorData::from_scalars([2], DType::F16, [Scalar::F(3.), Scalar::F(4.)],)
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
    fn normalize_uses_float_scalar_pow_or_exact_zero_branch() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::I32);
        let output = graph.normalize(input, 3.0, -1, 1e-12).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.op,
            crate::Op::Binary {
                op: crate::BinaryOp::Pow,
                ..
            }
        ) && node.dtype == DType::F32));
        let default = graph.normalize_default(input).unwrap();
        assert_eq!(graph.shape(default).unwrap(), &Shape::new([2, 3]));

        let mut differentiable = Graph::new();
        let input = differentiable.input_dtype("input", [2], DType::F32);
        let output = differentiable.normalize(input, 1.0, 0, 1e-12).unwrap();
        assert!(differentiable.grad(output, input).is_ok());

        let mut zero = Graph::new();
        let input = zero.input_dtype("input", [], DType::Bool);
        let output = zero.normalize(input, -0.0, -1, f64::NAN).unwrap();
        assert_eq!(zero.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(zero.dtype(output).unwrap(), DType::F32);
        assert!((0..zero.node_count()).all(|index| !matches!(
            zero.op(NodeId(index)).unwrap(),
            crate::Op::Binary {
                op: crate::BinaryOp::Pow,
                ..
            }
        )));

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [2, 0], DType::F16);
        let output = empty
            .normalize(input, f64::INFINITY, -1, f64::NEG_INFINITY)
            .unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([2, 0]));

        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed.normalize(input, 2.0, 2, 1e-12).is_err());
        assert_eq!(malformed.node_count(), nodes);

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.normalize(input, 2.0, -1, 1e-12),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn mean_with_axes_matches_tinygrad_typed_mean_and_preflights() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F16);
        let output = graph.mean_with_axes(input, Some(vec![-1]), true).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        assert!(graph.nodes.iter().any(|node| {
            matches!(
                &node.op,
                crate::Op::Reduce {
                    kind: ReduceKind::Sum,
                    ..
                }
            ) && node.dtype == DType::F32
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
            CpuBackend
                .execute(&graph, output, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![4., 8.]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![0.5; 4]
        );

        let mut all = Graph::new();
        let x = all.input("input", [2, 2]);
        let output = all.mean_with_axes(x, None, false).unwrap();
        assert_eq!(all.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(
            CpuBackend
                .execute(
                    &all,
                    output,
                    &HashMap::from([("input".into(), data([2, 2], &[1., 2., 3., 4.]))])
                )
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
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), data([2, 0], &[]))]),
            )
            .unwrap()
            .to_vec_f64();
        assert!(values.iter().all(|value| value.is_nan()));

        let mut malformed = Graph::new();
        let x = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed
            .mean_with_axes(x, Some(vec![0, -2]), false)
            .is_err());
        assert_eq!(malformed.node_count(), nodes);

        let mut overflow = Graph::new();
        let x = overflow.input("input", [usize::MAX, 2]);
        let nodes = overflow.node_count();
        assert!(overflow.mean_with_axes(x, None, false).is_err());
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn layernorm_reuses_centered_literal_and_preflights_all_stages() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3, 4], DType::F16);
        let output = graph
            .layernorm_with_axes(input, vec![-2, -1], 1e-5)
            .unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3, 4]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        let crate::Op::Binary {
            op: crate::BinaryOp::Mul,
            lhs,
            ..
        } = graph.op(output).unwrap()
        else {
            unreachable!()
        };
        // The final multiply reuses the same centered tensor that was squared
        // for variance, rather than recomputing source-minus-mean.
        let centered = *lhs;
        assert!(graph.nodes.iter().any(|node| {
            matches!(&node.op, crate::Op::Binary { op: crate::BinaryOp::Mul, lhs, rhs } if *lhs == centered && *rhs == centered)
        }));
        assert!(graph.grad(output, input).is_ok());
        let default = graph.layernorm_default(input).unwrap();
        assert_eq!(graph.shape(default).unwrap(), &Shape::new([2, 3, 4]));

        let mut nonfloat = Graph::new();
        let input = nonfloat.input_dtype("input", [], DType::I32);
        let output = nonfloat.layernorm(input, -1, f64::NAN).unwrap();
        assert_eq!(nonfloat.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [2, 0], DType::F32);
        let output = empty.layernorm(input, -1, 1e-5).unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([2, 0]));

        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed
            .layernorm_with_axes(input, vec![0, -2], 1e-5)
            .is_err());
        assert_eq!(malformed.node_count(), nodes);

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.layernorm(input, -1, 1e-5),
            Err(Error::ShapeOverflow(_))
        ));
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
                [
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(3.),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        assert!(values.scalar_at(1).as_f64().is_nan());
        let gradients = CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64();
        assert_eq!(&gradients[..2], &[0.5, 0.5]);

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
            DType::F16,
            DType::BF16,
            DType::F32,
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
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), data([2, 0], &[]))]),
            )
            .unwrap()
            .to_vec_f64();
        assert!(values
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_negative()));

        let mut malformed = Graph::new();
        let x = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed
            .max_with_axes(x, Some(vec![0, -2]), false)
            .is_err());
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
                [
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(-3.),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        assert!(values.scalar_at(1).as_f64().is_nan());
        let gradients = CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64();
        assert_eq!(&gradients[..2], &[0.5, 0.5]);

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
            DType::F16,
            DType::BF16,
            DType::F32,
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
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), data([2, 0], &[]))]),
            )
            .unwrap()
            .to_vec_f64();
        assert!(values
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_positive()));

        let mut scalar = Graph::new();
        let x = scalar.input("input", []);
        let output = scalar.min_with_axes(x, Some(vec![-1]), false).unwrap();
        assert_eq!(output, x);

        let mut malformed = Graph::new();
        let x = malformed.input("input", [2, 2]);
        let nodes = malformed.node_count();
        assert!(malformed
            .min_with_axes(x, Some(vec![0, -2]), false)
            .is_err());
        assert_eq!(malformed.node_count(), nodes);
    }

    #[test]
    fn extrema_default_wrappers_keep_literal_min_structure_and_atomicity() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F64);
        let maximum = graph.max_default(input).unwrap();
        let minimum = graph.min_default(input).unwrap();
        assert_eq!(graph.shape(maximum).unwrap(), &Shape::new([]));
        assert_eq!(graph.shape(minimum).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(maximum).unwrap(), DType::F64);
        assert_eq!(graph.dtype(minimum).unwrap(), DType::F64);
        let crate::Op::Reduce {
            kind: ReduceKind::Max,
            input: max_input,
            ..
        } = graph.op(maximum).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(*max_input, input);
        // tinygrad's Min is exactly `(-x).max(...)._inverse()`, not raw Min.
        let crate::Op::Unary {
            op: UnaryOp::Neg,
            input: min_max,
        } = graph.op(minimum).unwrap()
        else {
            unreachable!()
        };
        let crate::Op::Reduce {
            kind: ReduceKind::Max,
            input: min_inverse,
            ..
        } = graph.op(*min_max).unwrap()
        else {
            unreachable!()
        };
        assert!(matches!(
            graph.op(*min_inverse).unwrap(),
            crate::Op::Unary { op: UnaryOp::Neg, input: source } if *source == input
        ));
        let maximum_loss = graph.sum_all(maximum).unwrap();
        let minimum_loss = graph.sum_all(minimum).unwrap();
        let loss = graph.add(maximum_loss, minimum_loss).unwrap();
        assert!(graph.grad(loss, input).is_ok());

        let mut specials = Graph::new();
        let input = specials.input_dtype("input", [2], DType::F64);
        let maximum = specials.max_default(input).unwrap();
        let minimum = specials.min_default(input).unwrap();
        let max_gradient = specials.grad(maximum, input).unwrap();
        let min_gradient = specials.grad(minimum, input).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            TensorData::from_scalars([2], DType::F64, [Scalar::F(-0.0), Scalar::F(0.0)]).unwrap(),
        )]);
        for output in [maximum, minimum] {
            assert_eq!(
                CpuBackend
                    .execute(&specials, output, &bindings)
                    .unwrap()
                    .scalar_at(0)
                    .as_f64()
                    .to_bits(),
                (-0.0f64).to_bits(),
            );
        }
        assert_eq!(
            CpuBackend
                .execute(&specials, max_gradient, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![0.5, 0.5]
        );
        assert_eq!(
            CpuBackend
                .execute(&specials, min_gradient, &bindings)
                .unwrap()
                .to_vec_f64(),
            vec![0.5, 0.5]
        );

        let mut nonfloat = Graph::new();
        let input = nonfloat.input_dtype("input", [], DType::I32);
        let maximum = nonfloat.max_default(input).unwrap();
        let minimum = nonfloat.min_default(input).unwrap();
        assert_eq!(nonfloat.dtype(maximum).unwrap(), DType::I32);
        assert_eq!(nonfloat.dtype(minimum).unwrap(), DType::I32);
        assert!(matches!(
            nonfloat.op(minimum).unwrap(),
            crate::Op::Binary {
                op: crate::BinaryOp::BitXor,
                ..
            }
        ));

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 2], DType::F16);
        let maximum = empty.max_default(input).unwrap();
        let minimum = empty.min_default(input).unwrap();
        assert_eq!(empty.shape(maximum).unwrap(), &Shape::new([]));
        assert_eq!(empty.shape(minimum).unwrap(), &Shape::new([]));
        assert_eq!(empty.dtype(maximum).unwrap(), DType::F16);
        assert_eq!(empty.dtype(minimum).unwrap(), DType::F16);

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(overflow.max_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
        assert!(overflow.min_default(input).is_err());
        assert_eq!(overflow.node_count(), nodes);
    }
}
