//! Graph lowering for the validated static ONNX subset.

use super::{
    bad,
    schema::{
        attrs, axes_usize, const_i64, conv_pads, conv_pair, conv_same_padding, onnx_pool_options,
        packed_i64, scalar_f32, scalar_i64, strict_typed_packed_i64_attr,
        strict_typed_scalar_i64_attr, strict_typed_string_attr, typed_scalar_f32_attr,
        typed_scalar_i64_attr,
    },
    tensor::{onnx_dtype, tensor_data},
    wire::{var, Msg},
};
use crate::{
    ir::reduction_shape, Conv2dOptions, DType, Graph, NodeId, ReduceKind, ReductionDType, Result,
    Scalar, Shape, Slice, TensorData,
};
use std::collections::BTreeMap;

fn prelu_dtype(x: DType, slope: DType) -> DType {
    // tinygrad's weak binary lowering resolves the only supported lattice
    // disagreement, U64 mixed with I64, at its default F32 width. RustGrad's
    // generic promotion intentionally chooses F64 for that pair.
    if matches!(
        (x, slope),
        (DType::U64, DType::I64) | (DType::I64, DType::U64)
    ) {
        DType::F32
    } else {
        x.promote(slope)
    }
}

/// One source-order step of tinygrad's variadic ONNX `Max` fold.  Tensor
/// maximum first casts both operands to their least-upper dtype, then applies
/// ordered `lhs < rhs ? rhs : lhs` selection.  Keeping those resolved facts
/// separate from construction prevents a later malformed operand from
/// publishing a partial prefix of the fold.
struct VariadicMaxFold {
    input: NodeId,
    dtype: DType,
    shape: Shape,
}

struct VariadicMaxPlan {
    first: NodeId,
    folds: Vec<VariadicMaxFold>,
    output_shape: Shape,
    output_dtype: DType,
}

fn variadic_max_dtype(lhs: DType, rhs: DType) -> DType {
    // This is the same checked-in tinygrad least-upper-dtype exception used
    // by PRelu: mixed I64/U64 falls to default F32 rather than RustGrad's
    // broader generic F64 lattice.
    if matches!(
        (lhs, rhs),
        (DType::U64, DType::I64) | (DType::I64, DType::U64)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

/// One source-order stage of tinygrad's variadic ONNX `Sum` fold. Tensor.add
/// commits both operands to its source LUB before every storage-width ADD.
struct VariadicSumFold {
    input: NodeId,
    dtype: DType,
    shape: Shape,
}

struct VariadicSumPlan {
    first: NodeId,
    folds: Vec<VariadicSumFold>,
    output_shape: Shape,
    output_dtype: DType,
}

fn variadic_sum_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<VariadicSumPlan> {
    if ins.is_empty() {
        return Err(bad("Sum requires at least one input"));
    }
    if !attrs.is_empty() {
        return Err(bad("Sum does not accept attributes"));
    }
    let input = |index: usize| {
        ins.get(index)
            .and_then(|name| values.get(*name))
            .copied()
            .ok_or_else(|| bad("missing ONNX node input"))
    };
    let first = input(0)?;
    let mut output_shape = g.shape(first)?.clone();
    let mut output_dtype = g.dtype(first)?;
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Sum input byte extent overflow"))?;

    let mut folds = Vec::with_capacity(ins.len().saturating_sub(1));
    for index in 1..ins.len() {
        let right = input(index)?;
        let right_shape = g.shape(right)?.clone();
        let right_dtype = g.dtype(right)?;
        right_shape
            .numel()?
            .checked_mul(right_dtype.itemsize())
            .ok_or_else(|| bad("Sum input byte extent overflow"))?;
        let dtype = prelu_dtype(output_dtype, right_dtype);
        let shape = output_shape.broadcast_with(&right_shape)?;
        for (stage_shape, what) in [
            (&output_shape, "left cast"),
            (&right_shape, "right cast"),
            (&shape, "output"),
        ] {
            stage_shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| bad(format!("Sum {what} byte extent overflow")))?;
        }
        folds.push(VariadicSumFold {
            input: right,
            dtype,
            shape: shape.clone(),
        });
        output_shape = shape;
        output_dtype = dtype;
    }
    Ok(VariadicSumPlan {
        first,
        folds,
        output_shape,
        output_dtype,
    })
}

fn lower_variadic_sum_plan(g: &mut Graph, plan: VariadicSumPlan) -> Result<NodeId> {
    let mut sum = plan.first;
    for fold in plan.folds {
        let lhs = if g.dtype(sum).expect("Sum lhs preflighted") == fold.dtype {
            sum
        } else {
            g.cast(sum, fold.dtype)?
        };
        let rhs = if g.dtype(fold.input).expect("Sum rhs preflighted") == fold.dtype {
            fold.input
        } else {
            g.cast(fold.input, fold.dtype)?
        };
        sum = g.add(lhs, rhs)?;
        debug_assert_eq!(g.shape(sum).expect("Sum shape preflighted"), &fold.shape);
        debug_assert_eq!(g.dtype(sum).expect("Sum dtype preflighted"), fold.dtype);
    }
    debug_assert_eq!(
        g.shape(sum).expect("Sum output shape preflighted"),
        &plan.output_shape
    );
    debug_assert_eq!(
        g.dtype(sum).expect("Sum output dtype preflighted"),
        plan.output_dtype
    );
    Ok(sum)
}

/// Descriptor for tinygrad's variadic ONNX `Mean`: its complete source graph
/// is a left-folded Sum followed by true division by weak `len(data_0)`.
struct VariadicMeanPlan {
    sum: VariadicSumPlan,
    division_dtype: DType,
    divisor: TensorData,
    output_shape: Shape,
    output_dtype: DType,
}

fn variadic_mean_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<VariadicMeanPlan> {
    let sum = variadic_sum_plan(g, ins, attrs, values)?;
    // The weak integer count is committed by true division: a floating sum
    // retains its width, while Bool/integer sums lift their dividend to F32.
    let division_dtype = if sum.output_dtype.is_float() {
        sum.output_dtype
    } else {
        DType::F32
    };
    let divisor = TensorData::scalar_with_dtype(Scalar::F(ins.len() as f64), division_dtype);
    let output_shape = sum.output_shape.broadcast_with(divisor.shape())?;
    let output_dtype = prelu_dtype(sum.output_dtype, divisor.dtype());
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Mean {what} byte extent overflow")))
    };
    extent(&sum.output_shape, division_dtype, "sum cast")?;
    extent(divisor.shape(), divisor.dtype(), "divisor")?;
    extent(divisor.shape(), division_dtype, "reciprocal")?;
    extent(&output_shape, output_dtype, "output")?;
    if divisor.dtype() != division_dtype
        || sum.output_shape.broadcast_with(divisor.shape())? != sum.output_shape
        || output_shape != sum.output_shape
        || output_dtype != division_dtype
    {
        return Err(bad("Mean divisor source promotion mismatch"));
    }
    Ok(VariadicMeanPlan {
        sum,
        division_dtype,
        divisor,
        output_shape,
        output_dtype,
    })
}

fn variadic_max_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<VariadicMaxPlan> {
    if ins.is_empty() {
        return Err(bad("Max requires at least one input"));
    }
    if !attrs.is_empty() {
        return Err(bad("Max does not accept attributes"));
    }
    let input = |index: usize| {
        ins.get(index)
            .and_then(|name| values.get(*name))
            .copied()
            .ok_or_else(|| bad("missing ONNX node input"))
    };
    let first = input(0)?;
    let mut output_shape = g.shape(first)?.clone();
    let mut output_dtype = g.dtype(first)?;
    output_shape.numel()?;
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Max input byte extent overflow"))?;

    let mut folds = Vec::with_capacity(ins.len().saturating_sub(1));
    for index in 1..ins.len() {
        let right = input(index)?;
        let right_shape = g.shape(right)?.clone();
        let right_dtype = g.dtype(right)?;
        right_shape.numel()?;
        right_shape
            .numel()?
            .checked_mul(right_dtype.itemsize())
            .ok_or_else(|| bad("Max input byte extent overflow"))?;

        let dtype = variadic_max_dtype(output_dtype, right_dtype);
        let shape = output_shape.broadcast_with(&right_shape)?;
        shape.numel()?;
        for what in ["cast", "selection"] {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| bad(format!("Max {what} byte extent overflow")))?;
        }
        shape
            .numel()?
            .checked_mul(DType::Bool.itemsize())
            .ok_or_else(|| bad("Max comparison byte extent overflow"))?;
        folds.push(VariadicMaxFold {
            input: right,
            dtype,
            shape: shape.clone(),
        });
        output_shape = shape;
        output_dtype = dtype;
    }
    Ok(VariadicMaxPlan {
        first,
        folds,
        output_shape,
        output_dtype,
    })
}

/// One source-order step of tinygrad's variadic ONNX `Min` fold.  Although
/// Tensor.minimum is implemented through negated/bias-transformed Max rather
/// than the public Max helper, its observable ordered selection is
/// `lhs > rhs ? rhs : lhs`: equality and unordered comparisons retain lhs.
struct VariadicMinFold {
    input: NodeId,
    dtype: DType,
    shape: Shape,
}

struct VariadicMinPlan {
    first: NodeId,
    folds: Vec<VariadicMinFold>,
    output_shape: Shape,
    output_dtype: DType,
}

fn variadic_min_dtype(lhs: DType, rhs: DType) -> DType {
    // Match Tensor.minimum's `_broadcasted` least-upper dtype, including the
    // checked-in default-F32 resolution for the mixed I64/U64 pair.
    if matches!(
        (lhs, rhs),
        (DType::U64, DType::I64) | (DType::I64, DType::U64)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

fn variadic_min_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<VariadicMinPlan> {
    if ins.is_empty() {
        return Err(bad("Min requires at least one input"));
    }
    if !attrs.is_empty() {
        return Err(bad("Min does not accept attributes"));
    }
    let input = |index: usize| {
        ins.get(index)
            .and_then(|name| values.get(*name))
            .copied()
            .ok_or_else(|| bad("missing ONNX node input"))
    };
    let first = input(0)?;
    let mut output_shape = g.shape(first)?.clone();
    let mut output_dtype = g.dtype(first)?;
    output_shape.numel()?;
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Min input byte extent overflow"))?;

    let mut folds = Vec::with_capacity(ins.len().saturating_sub(1));
    for index in 1..ins.len() {
        let right = input(index)?;
        let right_shape = g.shape(right)?.clone();
        let right_dtype = g.dtype(right)?;
        right_shape.numel()?;
        right_shape
            .numel()?
            .checked_mul(right_dtype.itemsize())
            .ok_or_else(|| bad("Min input byte extent overflow"))?;

        let dtype = variadic_min_dtype(output_dtype, right_dtype);
        let shape = output_shape.broadcast_with(&right_shape)?;
        shape.numel()?;
        for what in ["cast", "selection"] {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| bad(format!("Min {what} byte extent overflow")))?;
        }
        shape
            .numel()?
            .checked_mul(DType::Bool.itemsize())
            .ok_or_else(|| bad("Min comparison byte extent overflow"))?;
        folds.push(VariadicMinFold {
            input: right,
            dtype,
            shape: shape.clone(),
        });
        output_shape = shape;
        output_dtype = dtype;
    }
    Ok(VariadicMinPlan {
        first,
        folds,
        output_shape,
        output_dtype,
    })
}

/// Fully resolved one stage of tinygrad's `Tensor.clamp`: each bound is
/// independently promoted and broadcast with the value produced by the prior
/// stage, so Min and Max need not broadcast with one another directly.
struct ClipStage {
    bound: NodeId,
    dtype: DType,
    shape: Shape,
}

struct ClipPlan {
    min: Option<ClipStage>,
    max: Option<ClipStage>,
    output_shape: Shape,
    output_dtype: DType,
}

fn clip_dtype(value: DType, bound: DType) -> DType {
    // `Tensor.clamp` reaches `_broadcasted` for each strict comparison and
    // select value pair, including tinygrad's I64/U64 default-F32 exception.
    if matches!(
        (value, bound),
        (DType::U64, DType::I64) | (DType::I64, DType::U64)
    ) {
        DType::F32
    } else {
        value.promote(bound)
    }
}

fn clip_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<ClipPlan> {
    if !(1..=3).contains(&ins.len()) {
        return Err(bad("Clip requires data and up to two bounds"));
    }
    if !attrs.is_empty() {
        return Err(bad("Clip does not accept attributes"));
    }
    let mut shape = g.shape(input)?.clone();
    let mut dtype = g.dtype(input)?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Clip {what} byte extent overflow")))
    };
    extent(&shape, dtype, "input")?;

    let mut stage = |index: usize, what: &str| -> Result<Option<ClipStage>> {
        let Some(name) = ins.get(index).filter(|name| !name.is_empty()) else {
            return Ok(None);
        };
        let bound = values
            .get(*name)
            .copied()
            .ok_or_else(|| bad("missing ONNX node input"))?;
        let bound_shape = g.shape(bound)?.clone();
        let bound_dtype = g.dtype(bound)?;
        extent(&bound_shape, bound_dtype, what)?;
        let stage_dtype = clip_dtype(dtype, bound_dtype);
        let stage_shape = shape.broadcast_with(&bound_shape)?;
        extent(&stage_shape, stage_dtype, what)?;
        extent(&stage_shape, DType::Bool, "comparison")?;
        shape = stage_shape.clone();
        dtype = stage_dtype;
        Ok(Some(ClipStage {
            bound,
            dtype: stage_dtype,
            shape: stage_shape,
        }))
    };
    let min = stage(1, "minimum")?;
    let max = stage(2, "maximum")?;
    drop(stage);
    Ok(ClipPlan {
        min,
        max,
        output_shape: shape,
        output_dtype: dtype,
    })
}

/// Source descriptor for ONNX `Abs`, which tinygrad lowers literally as
/// `x * x.sign()`.  Keep it separate from Graph::abs: the product preserves
/// negative zero and signed-integer wrapping that a unary absolute helper
/// intentionally does not expose.
struct AbsPlan {
    shape: Shape,
    dtype: DType,
}

fn abs_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<AbsPlan> {
    if ins.len() != 1 {
        return Err(bad("Abs requires exactly one input"));
    }
    if !attrs.is_empty() {
        return Err(bad("Abs does not accept attributes"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    for what in ["input", "sign", "output"] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Abs {what} byte extent overflow")))?;
    }
    Ok(AbsPlan { shape, dtype })
}

/// Source descriptor for ONNX `Neg`.  The existing Graph unary is exact for
/// tinygrad's Bool logical-not, integer wrapping multiplication by -1, and
/// floating sign flip; the importer still needs to validate its descriptor
/// before appending that node.
struct NegPlan {
    shape: Shape,
    dtype: DType,
}

fn neg_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<NegPlan> {
    if ins.len() != 1 {
        return Err(bad("Neg requires exactly one input"));
    }
    if !attrs.is_empty() {
        return Err(bad("Neg does not accept attributes"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    for what in ["input", "output"] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Neg {what} byte extent overflow")))?;
    }
    Ok(NegPlan { shape, dtype })
}

/// Source descriptor for ONNX `Relu`. Graph's public ReLU is tinygrad's
/// literal strict `(x > 0).where(x, 0)` graph; prove the importer input and
/// output descriptor before delegating so malformed nodes cannot publish
/// partial graph state.
struct ReluPlan {
    shape: Shape,
    dtype: DType,
}

fn relu_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<ReluPlan> {
    if ins.len() != 1 {
        return Err(bad("Relu requires exactly one input"));
    }
    if !attrs.is_empty() {
        return Err(bad("Relu does not accept attributes"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    for what in ["input", "output"] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Relu {what} byte extent overflow")))?;
    }
    Ok(ReluPlan { shape, dtype })
}

/// Data-only contract for tinygrad's ONNX OneHot adapter.  The adapter is not
/// the public `Tensor.one_hot` helper: it casts arbitrary indices to I32,
/// adjusts negative indices once, and selects from a live `[off, on, ..]`
/// tensor after forming the equality mask.  Keep every fallible descriptor,
/// scalar, movement, and broadcast fact here so malformed input cannot append
/// a partial graph.
struct OneHotPlan {
    axis: isize,
    index_zero: TensorData,
    index_depth: TensorData,
    classes: TensorData,
    class_shape: Shape,
    off_bounds: Vec<(usize, usize)>,
    on_bounds: Vec<(usize, usize)>,
    result_shape: Shape,
    result_dtype: DType,
}

/// tinygrad resolves the OneHot depth through Python's `int` after accepting
/// a scalar or a singleton sequence.  Rust's primitive float casts saturate
/// NaNs/infinities, so keep that conversion explicit rather than reusing the
/// I32/I64-only axis helper.
fn static_one_hot_depth(constants: &BTreeMap<String, TensorData>, name: &str) -> Result<i64> {
    let value = constants
        .get(name)
        .ok_or_else(|| bad("OneHot depth must be a constant initializer"))?;
    let shape = value.shape();
    shape.numel()?;
    if !(shape.rank() == 0 || (shape.rank() == 1 && shape.dims() == &[1])) || value.len() != 1 {
        return Err(bad(
            "OneHot depth must be a scalar or length-one rank-1 tensor",
        ));
    }
    match value.scalar_at(0) {
        Scalar::Bool(value) => Ok(i64::from(value)),
        Scalar::I(value) => Ok(value),
        Scalar::U(value) => {
            i64::try_from(value).map_err(|_| bad("OneHot depth is not representable by arange"))
        }
        Scalar::F(value) => {
            // Python rejects non-finite float-to-int conversion.  The upper
            // bound is exclusive because `i64::MAX as f64` rounds to 2^63.
            let value = value.trunc();
            if !value.is_finite() || value < i64::MIN as f64 || value >= 9_223_372_036_854_775_808.0
            {
                return Err(bad("OneHot depth is not representable by arange"));
            }
            Ok(value as i64)
        }
    }
}

fn one_hot_plan(
    g: &Graph,
    indices: NodeId,
    values: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<OneHotPlan> {
    if attrs.keys().any(|key| key != "axis") {
        return Err(bad("unsupported OneHot attribute"));
    }
    let indices_shape = g.shape(indices)?.clone();
    indices_shape.numel()?;
    let values_shape = g.shape(values)?.clone();
    values_shape.numel()?;
    if values_shape.rank() == 0 || values_shape.dims()[0] < 2 {
        return Err(bad("OneHot values must have first extent at least two"));
    }
    let raw_depth = static_one_hot_depth(constants, ins[1])?;
    if raw_depth < -1 {
        // `arange` itself returns empty for every negative endpoint, but the
        // subsequent shape uses the endpoint literally.  Only -1 is its
        // reshape inference sentinel, producing a zero class extent.
        return Err(bad(
            "OneHot depth below -1 is unsupported by source reshape",
        ));
    }
    let classes = if raw_depth <= 0 {
        0usize
    } else {
        usize::try_from(raw_depth).map_err(|_| bad("OneHot depth is not representable by shape"))?
    };
    let rank = indices_shape.rank();
    let output_rank = rank
        .checked_add(1)
        .ok_or_else(|| bad("OneHot output rank overflow"))?;
    let output_rank_i64 =
        i64::try_from(output_rank).map_err(|_| bad("OneHot output rank overflow"))?;
    let raw_axis = attrs
        .get("axis")
        .map(|raw| scalar_i64(raw))
        .transpose()?
        .unwrap_or(-1);
    if raw_axis < -output_rank_i64 || raw_axis >= output_rank_i64 {
        return Err(bad("invalid OneHot axis"));
    }
    let axis = if raw_axis < 0 {
        raw_axis
            .checked_add(output_rank_i64)
            .ok_or_else(|| bad("invalid OneHot axis"))?
    } else {
        raw_axis
    };
    let axis = usize::try_from(axis).map_err(|_| bad("OneHot axis overflow"))?;

    let mut mask_dims = indices_shape.dims().to_vec();
    mask_dims.insert(axis, classes);
    let mask_shape = Shape::new(mask_dims);
    mask_shape.numel()?;
    let mut class_dims = vec![1; output_rank];
    class_dims[axis] = classes;
    let class_shape = Shape::new(class_dims);
    class_shape.numel()?;
    if class_shape.broadcast_with(&mask_shape)? != mask_shape {
        return Err(bad("OneHot class range cannot broadcast to indices"));
    }

    // tinygrad creates this Python integer as a weak scalar, then promotes it
    // to the concrete I32 index width.  Retain Rust's defined wrapping cast
    // for depths outside I32, which is observable on the negative branch.
    let index_depth =
        TensorData::scalar_with_dtype(Scalar::I(i64::from(raw_depth as i32)), DType::I32);
    let index_zero = TensorData::scalar_with_dtype(Scalar::I(0), DType::I32);
    let classes_data = TensorData::arange(0, raw_depth.max(0), 1)?;
    if classes_data.shape() != &Shape::new([classes]) || classes_data.dtype() != DType::I64 {
        return Err(bad("OneHot class range does not match validated shape"));
    }

    let mut off_bounds = Vec::with_capacity(values_shape.rank());
    let mut on_bounds = Vec::with_capacity(values_shape.rank());
    off_bounds.push((0, 1));
    on_bounds.push((1, 2));
    for &extent in &values_shape.dims()[1..] {
        off_bounds.push((0, extent));
        on_bounds.push((0, extent));
    }
    let value_dims = values_shape.dims()[1..].to_vec();
    let value_shape = Shape::new(value_dims);
    value_shape.numel()?;
    let result_shape = mask_shape.broadcast_with(&value_shape)?;
    result_shape.numel()?;
    let result_dtype = g.dtype(values)?;

    let axis = isize::try_from(axis).map_err(|_| bad("OneHot axis overflow"))?;
    Ok(OneHotPlan {
        axis,
        index_zero,
        index_depth,
        classes: classes_data,
        class_shape,
        off_bounds,
        on_bounds,
        result_shape,
        result_dtype,
    })
}

/// Complete source-level construction plan for tinygrad's Hardmax adapter.
/// Unlike a normal argmax, a leading NaN leaves tinygrad's equality mask
/// empty and therefore produces an all-zero class slice.  The sentinel below
/// preserves that observable result while retaining the existing CPU
/// ArgReduce implementation for normal first-tie selection.
struct HardmaxPlan {
    axis: isize,
    empty: bool,
    first_bounds: Vec<(usize, usize)>,
    sentinel: TensorData,
    classes: TensorData,
    class_shape: Shape,
    output_shape: Shape,
    output_dtype: DType,
}

/// Complete source-level construction plan for tinygrad's ONNX ArgMax
/// adapter. The public Graph ArgReduce is first-tie only, so the importer
/// owns both the reversed last-tie form and tinygrad's leading-NaN sentinel.
/// A zero reduction axis is also represented here as its fully known I64
/// result, avoiding any change to Graph's explicit empty-reduction policy.
struct ArgMaxPlan {
    axis: isize,
    keepdims: bool,
    select_last: bool,
    first_bounds: Vec<(usize, usize)>,
    sentinel: TensorData,
    last_offset: Option<TensorData>,
    empty_axis_result: Option<TensorData>,
}

fn argmax_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<ArgMaxPlan> {
    if attrs
        .keys()
        .any(|key| !matches!(key.as_str(), "axis" | "keepdims" | "select_last_index"))
    {
        return Err(bad("unsupported ArgMax attribute"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    let input_numel = shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("ArgMax input byte extent overflow"))?;
    let rank = shape.rank();
    if rank == 0 {
        // tinygrad's adapter passes an explicit axis to Tensor.argmax, which
        // resolves that axis before its scalar special case.
        return Err(bad("ArgMax does not support scalar input"));
    }
    let rank_i64 = i64::try_from(rank).map_err(|_| bad("ArgMax rank overflow"))?;
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(0);
    let axis = if raw_axis < 0 {
        raw_axis
            .checked_add(rank_i64)
            .ok_or_else(|| bad("invalid ArgMax axis"))?
    } else {
        raw_axis
    };
    if axis < 0 || axis >= rank_i64 {
        return Err(bad("invalid ArgMax axis"));
    }
    let axis = usize::try_from(axis).map_err(|_| bad("invalid ArgMax axis"))?;
    let keepdims = strict_typed_scalar_i64_attr(n, "keepdims")?.unwrap_or(1) != 0;
    let select_last = strict_typed_scalar_i64_attr(n, "select_last_index")?.unwrap_or(0) != 0;
    let output_shape = reduction_shape(&shape, &[axis], keepdims);
    let output_numel = output_shape.numel()?;
    output_numel
        .checked_mul(DType::I64.itemsize())
        .ok_or_else(|| bad("ArgMax output byte extent overflow"))?;

    let axis_extent = shape.dims()[axis];
    let axis_extent_i64 = i64::try_from(axis_extent)
        .map_err(|_| bad("ArgMax axis extent exceeds tinygrad arange range"))?;
    if axis_extent == 0 {
        // In tinygrad, `eq(max(empty))` and the reverse arange are empty;
        // their I32 Max identity is then subtracted from the axis extent and
        // finally cast to I64. Preserve that observable normal/last sentinel
        // without weakening Graph::argmax's EmptyReduction contract.
        let value = if select_last {
            i64::from(i32::MAX)
        } else {
            i64::from(i32::MIN)
        };
        let data = TensorData::from_scalars(
            output_shape,
            DType::I64,
            std::iter::repeat(Scalar::I(value)).take(output_numel),
        )?;
        return Ok(ArgMaxPlan {
            axis: isize::try_from(axis).map_err(|_| bad("ArgMax axis overflow"))?,
            keepdims,
            select_last,
            first_bounds: Vec::new(),
            sentinel: TensorData::scalar_with_dtype(Scalar::I(0), DType::I32),
            last_offset: None,
            empty_axis_result: Some(data),
        });
    }

    let arg_shape = output_shape.clone();
    arg_shape
        .numel()?
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| bad("ArgMax index byte extent overflow"))?;
    let mut first_bounds = Vec::with_capacity(rank);
    for (dimension, &extent) in shape.dims().iter().enumerate() {
        first_bounds.push(if dimension == axis {
            (0, 1)
        } else {
            (0, extent)
        });
    }
    let first_shape = Shape::new(
        first_bounds
            .iter()
            .map(|(start, end)| end - start)
            .collect::<Vec<_>>(),
    );
    first_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("ArgMax first-lane byte extent overflow"))?;
    let first_result_shape = if keepdims {
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
    if first_result_shape != arg_shape {
        return Err(bad("ArgMax first-lane shape does not match argmax"));
    }
    first_result_shape
        .numel()?
        .checked_mul(DType::Bool.itemsize())
        .ok_or_else(|| bad("ArgMax NaN mask byte extent overflow"))?;

    // `Tensor.arange` promotes beyond the default I32 range, but ArgMax
    // explicitly casts its result to I32 before ONNX's final I64 cast.
    let sentinel = TensorData::scalar_with_dtype(Scalar::I(axis_extent_i64), DType::I32);
    if arg_shape.broadcast_with(sentinel.shape())? != arg_shape {
        return Err(bad("ArgMax NaN sentinel cannot broadcast to indices"));
    }
    let last_offset = if select_last {
        let offset = axis_extent_i64
            .checked_sub(1)
            .ok_or_else(|| bad("ArgMax last-index offset overflow"))?;
        let data = TensorData::scalar_with_dtype(Scalar::I(offset), DType::I32);
        if arg_shape.broadcast_with(data.shape())? != arg_shape {
            return Err(bad("ArgMax last-index offset cannot broadcast"));
        }
        Some(data)
    } else {
        None
    };
    Ok(ArgMaxPlan {
        axis: isize::try_from(axis).map_err(|_| bad("ArgMax axis overflow"))?,
        keepdims,
        select_last,
        first_bounds,
        sentinel,
        last_offset,
        empty_axis_result: None,
    })
}

// Tinygrad defines ONNX ArgMin as `ArgMax(-x)`, rather than using its tensor
// argmin helper. The descriptor, first-lane sentinel, and empty-axis results
// are therefore exactly the ArgMax plan; lower the required negation only
// after this shared source-level preflight has succeeded.
fn argmin_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<ArgMaxPlan> {
    argmax_plan(g, input, n, attrs)
}

fn hardmax_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<HardmaxPlan> {
    if attrs.keys().any(|key| key != "axis") {
        return Err(bad("unsupported Hardmax attribute"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    let input_numel = shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Hardmax input byte extent overflow"))?;
    let rank = shape.rank();
    if rank == 0 {
        // tinygrad's explicit `argmax(axis=-1)` path indexes the scalar
        // shape after resolving its axis, rather than using argmax(None).
        return Err(bad("Hardmax does not support scalar input"));
    }
    let rank_i64 = i64::try_from(rank).map_err(|_| bad("Hardmax rank overflow"))?;
    // ONNX declares axis as AttributeProto::INT. Decode the original
    // AttributeProto rather than the normalized raw bytes so another value
    // field cannot masquerade as a signed axis.
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(-1);
    if raw_axis < -rank_i64 || raw_axis >= rank_i64 {
        return Err(bad("invalid Hardmax axis"));
    }
    let axis = if raw_axis < 0 {
        raw_axis
            .checked_add(rank_i64)
            .ok_or_else(|| bad("invalid Hardmax axis"))?
    } else {
        raw_axis
    };
    let axis = usize::try_from(axis).map_err(|_| bad("Hardmax axis overflow"))?;
    let axis_extent = shape.dims()[axis];
    let sentinel =
        i32::try_from(axis_extent).map_err(|_| bad("Hardmax axis extent exceeds I32 indices"))?;

    // ArgReduce removes the axis, while the checked first-lane Shrink then
    // Squeeze reaches exactly the same descriptor before isnan/select.
    let arg_shape = Shape::new(
        shape
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(dimension, &extent)| (dimension != axis).then_some(extent))
            .collect::<Vec<_>>(),
    );
    arg_shape.numel()?;
    arg_shape
        .numel()?
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| bad("Hardmax ArgMax index byte extent overflow"))?;
    let mut first_bounds = Vec::with_capacity(rank);
    for (dimension, &extent) in shape.dims().iter().enumerate() {
        first_bounds.push(if dimension == axis {
            (0, 1)
        } else {
            (0, extent)
        });
    }
    let first_shape = Shape::new(
        first_bounds
            .iter()
            .map(|(start, end)| end - start)
            .collect::<Vec<_>>(),
    );
    first_shape.numel()?;
    first_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Hardmax first-lane byte extent overflow"))?;
    let squeezed_shape = Shape::new(
        first_shape
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(dimension, &extent)| (dimension != axis).then_some(extent))
            .collect::<Vec<_>>(),
    );
    if squeezed_shape != arg_shape {
        return Err(bad("Hardmax first-lane shape does not match argmax"));
    }
    squeezed_shape
        .numel()?
        .checked_mul(DType::Bool.itemsize())
        .ok_or_else(|| bad("Hardmax leading-NaN mask byte extent overflow"))?;
    let sentinel_data = TensorData::scalar_with_dtype(Scalar::I(i64::from(sentinel)), DType::I32);
    if arg_shape.broadcast_with(sentinel_data.shape())? != arg_shape {
        return Err(bad("Hardmax NaN sentinel cannot broadcast to argmax"));
    }
    let mut restored_dims = arg_shape.dims().to_vec();
    restored_dims.insert(axis, 1);
    let restored_index_shape = Shape::new(restored_dims);
    restored_index_shape.numel()?;
    restored_index_shape
        .numel()?
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| bad("Hardmax restored index byte extent overflow"))?;

    let mut class_dims = vec![1; rank];
    class_dims[axis] = axis_extent;
    let class_shape = Shape::new(class_dims);
    class_shape.numel()?;
    class_shape
        .numel()?
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| bad("Hardmax class range byte extent overflow"))?;
    if class_shape.broadcast_with(&restored_index_shape)? != shape {
        return Err(bad("Hardmax classes cannot broadcast to input"));
    }
    let classes = TensorData::arange(
        0,
        i64::try_from(axis_extent).map_err(|_| bad("Hardmax axis extent exceeds arange range"))?,
        1,
    )?
    .cast(DType::I32);
    if classes.shape() != &Shape::new([axis_extent]) || classes.dtype() != DType::I32 {
        return Err(bad("Hardmax class range does not match validated axis"));
    }
    // The final compare restores the original shape and bool-to-source cast
    // preserves the exact input dtype.
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Hardmax output byte extent overflow"))?;
    Ok(HardmaxPlan {
        axis: isize::try_from(axis).map_err(|_| bad("Hardmax axis overflow"))?,
        empty: input_numel == 0,
        first_bounds,
        sentinel: sentinel_data,
        classes,
        class_shape,
        output_shape: shape,
        output_dtype: dtype,
    })
}

/// Data-only descriptor plan for ONNX Shrink activation.  This deliberately
/// has no connection to the movement `Graph::shrink` API: tinygrad defines
/// the activation as two strict-mask products whose IEEE multiplication is
/// observable for NaNs, infinities, and negative lambda overlaps.
struct ShrinkActivationPlan {
    work_dtype: DType,
    output_dtype: DType,
    narrow: bool,
    negative_lambda: TensorData,
    lambda: TensorData,
    bias: TensorData,
    output_shape: Shape,
}

/// Fully constructed data-only EyeLike result. The input's values never take
/// part in tinygrad's adapter; preserving its unusual rectangular padding
/// means the entire result can be validated before its one constant node.
struct EyeLikePlan {
    data: TensorData,
}

/// Complete descriptor-only plan for tinygrad's SpaceToDepth rearrange.  The
/// channel factor deliberately follows `(h1, w1, c)`, not the more common
/// `(c, h1, w1)` convention, so retain both movement shapes explicitly.
struct SpaceToDepthPlan {
    first_shape: Shape,
    output_shape: Shape,
    identity: bool,
}

/// The two literal branches in tinygrad's DepthToSpace adapter. Any UTF-8
/// mode other than exactly `CRD` follows the default DCR branch.
enum DepthToSpaceMode {
    Dcr,
    Crd,
}

struct DepthToSpacePlan {
    first_shape: Shape,
    permutation: [usize; 6],
    output_shape: Shape,
    identity: bool,
}

/// Complete source-level movement plan for CenterCropPad. Tinygrad assigns
/// per-axis entries through Python `zip`, so duplicate axes deliberately
/// overwrite prior plans and unpaired values are invisible.
struct CenterCropPadPlan {
    shrink: Option<Vec<(usize, usize)>>,
    padding: Option<Vec<(usize, usize)>>,
    fill: Scalar,
}

/// Data-only construction contract for tinygrad's LRN adapter.  Its channel
/// average is deliberately not `Graph::avg_pool2d`: tinygrad widens the mean
/// accumulator for narrow floats and performs true F32 division for integral
/// inputs.  Keeping the view windows here also ensures a malformed request
/// cannot publish a Pad or partial reduction chain.
struct LrnPlan {
    input_dtype: DType,
    reshaped: Shape,
    padding: Vec<(usize, usize)>,
    windows: Vec<Vec<Slice>>,
    sum_dtypes: ReductionDType,
    pool_dtype: DType,
    output_dtype: DType,
    narrow_pool: bool,
    divisor: TensorData,
    alpha: TensorData,
    beta: TensorData,
    bias: TensorData,
    output_shape: Shape,
    empty: bool,
}

/// Source-complete plan for tinygrad's generic ONNX `FastGelu`: the optional
/// second input is a live bias, then the result takes Tensor.gelu's tanh path.
/// It deliberately does not describe the unrelated QuickGELU helper.
struct FastGeluPlan {
    input: NodeId,
    bias: Option<NodeId>,
    gelu_input_shape: Shape,
    gelu_input_dtype: DType,
    output_dtype: DType,
}

/// Complete descriptor closure for tinygrad's `BiasGelu(x + bias, approximate)`.
struct BiasGeluPlan {
    add: AddPlan,
    mode: String,
    output_dtype: DType,
}

struct EluPlan {
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    zero: TensorData,
    one: TensorData,
    alpha: TensorData,
    empty: bool,
}

struct SeluPlan {
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    zero: TensorData,
    one: TensorData,
    alpha: TensorData,
    gamma: TensorData,
    empty: bool,
}

struct SwishPlan {
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    alpha: TensorData,
    one: TensorData,
    neg_inv_ln2: TensorData,
    empty: bool,
}

struct ModPlan {
    fmod: bool,
    shape: Shape,
    dtype: DType,
}

/// The importer has a single-value environment, whereas tinygrad's Dropout
/// always returns `(data, bool_mask)`.  This plan therefore admits only the
/// source identity path when the ONNX node requests its first output alone.
/// Ratio and seed are semantically dead there, but supplied controls still
/// need static descriptor validation before X is republished.
struct DropoutPlan {
    shape: Shape,
    dtype: DType,
}

/// Complete static descriptor plan for the first output of tinygrad's
/// LayerNormalization adapter. The source always computes its statistics in
/// F32, then restores X's storage dtype before applying live scale and bias.
struct LayerNormalizationPlan {
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    axes: Vec<isize>,
    sum_dtypes: ReductionDType,
    count: TensorData,
    epsilon: TensorData,
    empty: bool,
}

struct MeanVarianceNormalizationPlan {
    input_dtype: DType,
    work_dtype: DType,
    sum_dtype: DType,
    shape: Shape,
    axes: Vec<isize>,
    count: TensorData,
    epsilon: TensorData,
    empty: bool,
}

struct LpNormalizationPlan {
    input_dtype: DType,
    output_dtype: DType,
    denominator_dtype: DType,
    shape: Shape,
    axes: Vec<isize>,
    sum_dtypes: ReductionDType,
    l1: bool,
    empty: bool,
}

/// Fully resolved live-operand contract for tinygrad's ONNX RMSNormalization.
struct RmsNormalizationPlan {
    output_dtype: DType,
    shape: Shape,
    axes: Vec<isize>,
    count: TensorData,
    epsilon: TensorData,
}

struct EinsumPlan {
    equation: String,
    inputs: Vec<NodeId>,
    output_shape: Shape,
    output_dtype: DType,
}

fn einsum_plan(
    g: &Graph,
    inputs: &[NodeId],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<EinsumPlan> {
    if inputs.is_empty() || attrs.keys().any(|key| key != "equation") {
        return Err(bad("unsupported Einsum input or attribute"));
    }
    // Tensor.einsum removes literal spaces before parsing; retain its entire
    // grammar by forwarding the normalized equation to Graph::einsum.
    let equation = strict_typed_string_attr(n, "equation")?
        .ok_or_else(|| bad("Einsum requires equation"))?
        .replace(' ', "");
    if equation.is_empty() {
        return Err(bad("Einsum equation is empty"));
    }
    let mut output_dtype = DType::Bool;
    for input in inputs {
        let shape = g.shape(*input)?.clone();
        let dtype = g.dtype(*input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Einsum input byte extent overflow"))?;
        output_dtype = output_dtype.promote(dtype);
    }
    // Parse against the complete static shape inventory before Graph::einsum
    // appends its single Einsum node.
    let shapes = inputs
        .iter()
        .map(|input| Ok(g.shape(*input)?.clone()))
        .collect::<Result<Vec<_>>>()?;
    let parsed = crate::EinsumPlan::parse(&equation, &shapes)?;
    let output_shape = parsed.output_shape();
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Einsum output byte extent overflow"))?;
    Ok(EinsumPlan {
        equation,
        inputs: inputs.to_vec(),
        output_shape,
        output_dtype,
    })
}

fn lp_normalization_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<LpNormalizationPlan> {
    if attrs.keys().any(|key| key != "axis" && key != "p") {
        return Err(bad("unsupported LpNormalization attribute"));
    }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(&format!("LpNormalization {what} byte extent overflow")))
    };
    // The source branches begin with either `x * sign(x)` or `x * x`; both
    // retain the original descriptor.  Tensor.sum then materializes its
    // full-shaped accumulator cast before reducing and may narrow only after
    // the reduction.  Prove that complete chain before either branch node is
    // appended so a narrow-storage accumulator overflow remains atomic.
    extent(&shape, input_dtype, "input/base")?;
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(-1);
    let rank = i64::try_from(shape.rank()).map_err(|_| bad("LpNormalization rank overflow"))?;
    let axes = if rank == 0 {
        if !matches!(raw_axis, -1 | 0) {
            return Err(bad("invalid LpNormalization scalar axis"));
        }
        Vec::new()
    } else {
        let axis = if raw_axis < 0 {
            raw_axis
                .checked_add(rank)
                .ok_or_else(|| bad("invalid LpNormalization axis"))?
        } else {
            raw_axis
        };
        if axis < 0 || axis >= rank {
            return Err(bad("invalid LpNormalization axis"));
        }
        vec![isize::try_from(axis).map_err(|_| bad("invalid LpNormalization axis"))?]
    };
    // Tinygrad only distinguishes p == 1; any other INT takes its square
    // branch, including zero, negative, and otherwise nonstandard values.
    let l1 = strict_typed_scalar_i64_attr(n, "p")?.unwrap_or(2) == 1;
    let sum_dtypes = ReductionDType::sum_default(input_dtype);
    let denominator_dtype = if l1 {
        sum_dtypes.output
    } else if sum_dtypes.output.is_float() {
        sum_dtypes.output
    } else {
        DType::F32
    };
    let output_dtype = input_dtype.promote(if denominator_dtype.is_float() {
        denominator_dtype
    } else {
        DType::F32
    });
    extent(&shape, sum_dtypes.accumulator, "Sum accumulator")?;
    extent(&shape, output_dtype, "output")?;
    let mut denom_dims = shape.dims().to_vec();
    for axis in &axes {
        denom_dims[*axis as usize] = 1;
    }
    let denominator_shape = Shape::new(denom_dims);
    extent(
        &denominator_shape,
        sum_dtypes.accumulator,
        "Sum reduction accumulator",
    )?;
    extent(&denominator_shape, sum_dtypes.output, "Sum output")?;
    extent(&denominator_shape, denominator_dtype, "denominator")?;
    let reciprocal_dtype = if denominator_dtype.is_float() {
        denominator_dtype
    } else {
        DType::F32
    };
    extent(&denominator_shape, reciprocal_dtype, "reciprocal")?;
    if denominator_shape.broadcast_with(&shape)? != shape
        || input_dtype.promote(reciprocal_dtype) != output_dtype
    {
        return Err(bad("LpNormalization promotion mismatch"));
    }
    Ok(LpNormalizationPlan {
        input_dtype,
        output_dtype,
        denominator_dtype,
        shape,
        axes,
        sum_dtypes,
        l1,
        empty: numel == 0,
    })
}

fn rms_normalization_plan(
    g: &Graph,
    input: NodeId,
    scale: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<RmsNormalizationPlan> {
    if attrs
        .keys()
        .any(|key| !matches!(key.as_str(), "axis" | "epsilon" | "stash_type"))
    {
        return Err(bad("unsupported RMSNormalization attribute"));
    }
    if strict_typed_scalar_i64_attr(n, "stash_type")?.unwrap_or(1) != 1 {
        return Err(bad("only RMSNormalization stash_type=1 is supported"));
    }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let scale_shape = g.shape(scale)?.clone();
    let scale_dtype = g.dtype(scale)?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(&format!("RMSNormalization {what} byte extent overflow")))
    };
    extent(&shape, input_dtype, "input")?;
    extent(&scale_shape, scale_dtype, "scale")?;
    let rank = i64::try_from(shape.rank()).map_err(|_| bad("RMSNormalization rank overflow"))?;
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(-1);
    let axis = if rank == 0 {
        if !matches!(raw_axis, -1 | 0) {
            return Err(bad("invalid RMSNormalization scalar axis"));
        }
        0usize
    } else {
        let normalized = if raw_axis < 0 {
            raw_axis
                .checked_add(rank)
                .ok_or_else(|| bad("invalid RMSNormalization axis"))?
        } else {
            raw_axis
        };
        if normalized < 0 || normalized >= rank {
            return Err(bad("invalid RMSNormalization axis"));
        }
        usize::try_from(normalized).map_err(|_| bad("invalid RMSNormalization axis"))?
    };
    let axes = if rank == 0 {
        Vec::new()
    } else {
        (axis..shape.rank()).map(|i| i as isize).collect::<Vec<_>>()
    };
    let mut statistic_dims = shape.dims().to_vec();
    for dimension in statistic_dims.iter_mut().skip(axis) {
        *dimension = 1;
    }
    let statistic_shape = Shape::new(statistic_dims);
    let count = if rank == 0 {
        1
    } else {
        shape.dims()[axis..]
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| bad("RMSNormalization normalized extent overflow"))?
    };
    let epsilon = typed_scalar_f32_attr(n, "epsilon")?.unwrap_or(1e-5);
    let count = TensorData::scalar_with_dtype(Scalar::F(count as f64), DType::F32);
    let epsilon = TensorData::scalar_with_dtype(Scalar::F(f64::from(epsilon)), DType::F32);
    // X is explicitly cast before square/mean. These are separate F32 storage
    // boundaries, followed by source-order X*norm then live-scale multiply.
    extent(&shape, DType::F32, "F32 cast/square")?;
    extent(&statistic_shape, DType::F32, "mean/add/rsqrt")?;
    for scalar in [&count, &epsilon] {
        if scalar.dtype() != DType::F32
            || statistic_shape.broadcast_with(scalar.shape())? != statistic_shape
        {
            return Err(bad("RMSNormalization scalar promotion mismatch"));
        }
    }
    if statistic_shape.broadcast_with(&shape)? != shape
        || shape.broadcast_with(&scale_shape)? != shape
    {
        return Err(bad("RMSNormalization scale cannot broadcast to X"));
    }
    let normalized_dtype = input_dtype.promote(DType::F32);
    let output_dtype = normalized_dtype.promote(scale_dtype);
    extent(&shape, normalized_dtype, "X times norm")?;
    extent(&shape, output_dtype, "output")?;
    Ok(RmsNormalizationPlan {
        output_dtype,
        shape,
        axes,
        count,
        epsilon,
    })
}

fn mean_variance_normalization_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<MeanVarianceNormalizationPlan> {
    // The checked-in tinygrad adapter exposes the control as singular `axis`
    // even though other reduction operators use `axes`.
    if attrs.keys().any(|key| key != "axis") {
        return Err(bad("unsupported MeanVarianceNormalization attribute"));
    }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("MeanVarianceNormalization input byte extent overflow"))?;
    let rank =
        i64::try_from(shape.rank()).map_err(|_| bad("MeanVarianceNormalization rank overflow"))?;
    let raw_axes = strict_typed_packed_i64_attr(n, "axis")?.unwrap_or_else(|| vec![0, 2, 3]);
    let mut seen = std::collections::BTreeSet::new();
    let mut axes = Vec::with_capacity(raw_axes.len());
    for raw in raw_axes {
        let axis = if raw < 0 {
            raw.checked_add(rank)
                .ok_or_else(|| bad("invalid MeanVarianceNormalization axis"))?
        } else {
            raw
        };
        if axis < 0 || axis >= rank {
            return Err(bad("invalid MeanVarianceNormalization axis"));
        }
        let axis =
            usize::try_from(axis).map_err(|_| bad("invalid MeanVarianceNormalization axis"))?;
        if !seen.insert(axis) {
            return Err(bad("duplicate MeanVarianceNormalization axis"));
        }
        axes.push(axis);
    }
    let count = axes
        .iter()
        .try_fold(1usize, |count, axis| count.checked_mul(shape.dims()[*axis]))
        .ok_or_else(|| bad("MeanVarianceNormalization reduction extent overflow"))?;
    let sum_dtype = ReductionDType::sum_default(input_dtype).accumulator;
    let work_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    numel
        .checked_mul(work_dtype.itemsize())
        .ok_or_else(|| bad("MeanVarianceNormalization output byte extent overflow"))?;
    let mean_shape = Shape::new(
        shape
            .dims()
            .iter()
            .enumerate()
            .map(|(i, dim)| if seen.contains(&i) { 1 } else { *dim })
            .collect::<Vec<_>>(),
    );
    mean_shape
        .numel()?
        .checked_mul(work_dtype.itemsize())
        .ok_or_else(|| bad("MeanVarianceNormalization mean byte extent overflow"))?;
    mean_shape
        .numel()?
        .checked_mul(sum_dtype.itemsize())
        .ok_or_else(|| bad("MeanVarianceNormalization sum byte extent overflow"))?;
    let count = TensorData::scalar_with_dtype(Scalar::F(count as f64), sum_dtype);
    let epsilon = TensorData::scalar_with_dtype(Scalar::F(1e-9), work_dtype);
    if count.dtype() != sum_dtype
        || epsilon.dtype() != work_dtype
        || mean_shape.broadcast_with(count.shape())? != mean_shape
        || mean_shape.broadcast_with(epsilon.shape())? != mean_shape
    {
        return Err(bad("MeanVarianceNormalization scalar promotion mismatch"));
    }
    Ok(MeanVarianceNormalizationPlan {
        input_dtype,
        work_dtype,
        sum_dtype,
        shape,
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        count,
        epsilon,
        empty: numel == 0,
    })
}

fn layer_normalization_plan(
    g: &Graph,
    input: NodeId,
    scale: NodeId,
    bias: Option<NodeId>,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<LayerNormalizationPlan> {
    if attrs
        .keys()
        .any(|key| !matches!(key.as_str(), "axis" | "epsilon" | "stash_type"))
    {
        return Err(bad("unsupported LayerNormalization attribute"));
    }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("LayerNormalization input byte extent overflow"))?;
    let rank = i64::try_from(shape.rank()).map_err(|_| bad("LayerNormalization rank overflow"))?;
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(-1);
    if rank == 0 || raw_axis < -rank || raw_axis >= rank {
        return Err(bad("invalid LayerNormalization axis"));
    }
    let axis = if raw_axis < 0 {
        raw_axis
            .checked_add(rank)
            .ok_or_else(|| bad("invalid LayerNormalization axis"))?
    } else {
        raw_axis
    };
    let axis = usize::try_from(axis).map_err(|_| bad("invalid LayerNormalization axis"))?;
    // The local source asserts this exact stash dtype rather than changing
    // execution precision based on it.
    if strict_typed_scalar_i64_attr(n, "stash_type")?.unwrap_or(1) != 1 {
        return Err(bad("only LayerNormalization stash_type=1 is supported"));
    }
    let epsilon = typed_scalar_f32_attr(n, "epsilon")?.unwrap_or(1e-5);

    let scale_shape = g.shape(scale)?.clone();
    let scale_dtype = g.dtype(scale)?;
    scale_shape
        .numel()?
        .checked_mul(scale_dtype.itemsize())
        .ok_or_else(|| bad("LayerNormalization scale byte extent overflow"))?;
    if shape.broadcast_with(&scale_shape)? != shape {
        return Err(bad("LayerNormalization scale cannot broadcast to X"));
    }
    let mut output_dtype = input_dtype.promote(scale_dtype);
    if let Some(bias) = bias {
        let bias_shape = g.shape(bias)?.clone();
        let bias_dtype = g.dtype(bias)?;
        bias_shape
            .numel()?
            .checked_mul(bias_dtype.itemsize())
            .ok_or_else(|| bad("LayerNormalization bias byte extent overflow"))?;
        if shape.broadcast_with(&bias_shape)? != shape {
            return Err(bad("LayerNormalization bias cannot broadcast to X"));
        }
        output_dtype = output_dtype.promote(bias_dtype);
    }
    numel
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("LayerNormalization output byte extent overflow"))?;
    let count = shape.dims()[axis..]
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| bad("LayerNormalization normalized extent overflow"))?;
    let count = TensorData::scalar_with_dtype(Scalar::F(count as f64), DType::F32);
    let epsilon = TensorData::scalar_with_dtype(Scalar::F(f64::from(epsilon)), DType::F32);
    let mean_shape = Shape::new(
        shape
            .dims()
            .iter()
            .enumerate()
            .map(|(i, dim)| if i < axis { *dim } else { 1 })
            .collect::<Vec<_>>(),
    );
    mean_shape
        .numel()?
        .checked_mul(DType::F32.itemsize())
        .ok_or_else(|| bad("LayerNormalization statistic byte extent overflow"))?;
    for scalar in [&count, &epsilon] {
        if scalar.dtype() != DType::F32 || mean_shape.broadcast_with(scalar.shape())? != mean_shape
        {
            return Err(bad("LayerNormalization scalar promotion mismatch"));
        }
    }
    let axes = (axis..shape.rank()).map(|i| i as isize).collect();
    Ok(LayerNormalizationPlan {
        input_dtype,
        output_dtype,
        shape,
        axes,
        sum_dtypes: ReductionDType::new(DType::F32, DType::F32),
        count,
        epsilon,
        empty: numel == 0,
    })
}

fn dropout_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<DropoutPlan> {
    if attrs.keys().any(|key| key != "seed") {
        return Err(bad("unsupported Dropout attribute"));
    }
    // ONNX's seed is an INT attribute. It is ignored by tinygrad before the
    // training branch, but malformed aliases must not sneak through the
    // inference-only adapter.
    let _ = strict_typed_scalar_i64_attr(n, "seed")?;
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Dropout input byte extent overflow"))?;

    if let Some(name) = ins.get(1).filter(|name| !name.is_empty()) {
        let ratio = constants
            .get(*name)
            .ok_or_else(|| bad("Dropout ratio must be constant"))?;
        // `_get_python_const` passes this value through, but `dropout_7`
        // never reads it when training_mode is false. Retain that exact
        // identity behavior for every statically representable descriptor.
        ratio
            .shape()
            .numel()?
            .checked_mul(ratio.dtype().itemsize())
            .ok_or_else(|| bad("Dropout ratio byte extent overflow"))?;
    }
    if let Some(name) = ins.get(2).filter(|name| !name.is_empty()) {
        let training = constants
            .get(*name)
            .ok_or_else(|| bad("Dropout training_mode must be constant"))?;
        training
            .shape()
            .numel()?
            .checked_mul(training.dtype().itemsize())
            .ok_or_else(|| bad("Dropout training_mode byte extent overflow"))?;
        // This is the closed ONNX Bool-scalar inference subset. A rank-one
        // `[false]` is a truthy Python list in tinygrad and must not be
        // mistaken for inference.
        if training.dtype() != DType::Bool
            || training.shape().rank() != 0
            || training.len() != 1
            || training.scalar_at(0).as_bool()
        {
            return Err(bad(
                "only inference Dropout with scalar training_mode=false is supported",
            ));
        }
    }
    Ok(DropoutPlan { shape, dtype })
}

struct GlobalAveragePoolPlan {
    axes: Vec<isize>,
    sum_dtypes: ReductionDType,
    work_dtype: DType,
    output_dtype: DType,
    divisor: TensorData,
    output_shape: Shape,
}

struct SoftplusPlan {
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    beta: TensorData,
}

/// Source-level plan for `Tensor.softsign()`. Its `abs` is not RustGrad's
/// hardware-style UnaryOp::Abs: tinygrad spells it `x * x.sign()`, preserving
/// negative zero and signed-integer wrapping before reciprocal-based true
/// division promotes exact storage to F32.
struct SoftsignPlan {
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    one: TensorData,
    empty: bool,
}

fn softsign_plan(
    g: &Graph,
    input: NodeId,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<SoftsignPlan> {
    if !attrs.is_empty() {
        return Err(bad("unsupported Softsign attribute"));
    }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("Softsign input byte extent overflow"))?;

    // `1 + x.abs()` stays at X's concrete storage dtype. Tensor.div then
    // lowers literally to `x * reciprocal(denominator)`, so exact storage
    // becomes F32 only at reciprocal for Bool/integer inputs.
    let reciprocal_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let output_dtype = input_dtype.promote(reciprocal_dtype);
    numel
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Softsign output byte extent overflow"))?;
    let one = TensorData::scalar_with_dtype(Scalar::I(1), input_dtype);
    if one.dtype() != input_dtype
        || shape.broadcast_with(one.shape())? != shape
        || input_dtype.promote(input_dtype) != input_dtype
        || input_dtype.promote(reciprocal_dtype) != output_dtype
    {
        return Err(bad("Softsign scalar promotion mismatch"));
    }
    Ok(SoftsignPlan {
        input_dtype,
        output_dtype,
        shape,
        one,
        empty: numel == 0,
    })
}

fn softplus_plan(
    g: &Graph,
    input: NodeId,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<SoftplusPlan> {
    if !attrs.is_empty() {
        return Err(bad("unsupported Softplus attribute"));
    }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("Softplus input byte extent overflow"))?;

    // The generic ONNX dispatcher calls `Tensor.softplus()` without an
    // argument.  Its Python default is a weak `1.0`, committed by the first
    // `x * beta` to X's float storage width, or to default F32 for exact
    // storage.  Keep that default as a concrete scalar only after proving the
    // complete public composition below.
    let beta_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let beta = TensorData::scalar_with_dtype(Scalar::F(1.0), beta_dtype);
    let beta_shape = beta.shape().clone();
    beta_shape
        .numel()?
        .checked_mul(beta.dtype().itemsize())
        .ok_or_else(|| bad("Softplus beta byte extent overflow"))?;
    let source_promote = |lhs: DType, rhs: DType| prelu_dtype(lhs, rhs);
    let scaled_shape = shape.broadcast_with(&beta_shape)?;
    let scaled_dtype = source_promote(input_dtype, beta_dtype);
    let log_dtype = if scaled_dtype.is_float() {
        scaled_dtype
    } else {
        DType::F32
    };
    let inverse_dtype = if beta_dtype.is_float() {
        beta_dtype
    } else {
        DType::F32
    };
    let output_shape = scaled_shape.broadcast_with(&beta_shape)?;
    let output_dtype = source_promote(log_dtype, inverse_dtype);
    for (extent_shape, dtype, what) in [
        (&scaled_shape, scaled_dtype, "scaled input"),
        (&scaled_shape, log_dtype, "logaddexp input"),
        (&beta_shape, inverse_dtype, "reciprocal beta"),
        (&output_shape, output_dtype, "output"),
    ] {
        extent_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Softplus {what} byte extent overflow")))?;
    }
    let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), log_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), inverse_dtype);
    if beta.dtype() != beta_dtype
        || scaled_shape != shape
        || output_shape != shape
        || zero.dtype() != log_dtype
        || one.dtype() != inverse_dtype
        || scaled_shape.broadcast_with(zero.shape())? != scaled_shape
        || beta_shape.broadcast_with(one.shape())? != beta_shape
        || source_promote(input_dtype, beta_dtype) != scaled_dtype
        || source_promote(log_dtype, zero.dtype()) != log_dtype
        || source_promote(inverse_dtype, one.dtype()) != inverse_dtype
        || source_promote(log_dtype, inverse_dtype) != output_dtype
    {
        return Err(bad("Softplus scalar promotion mismatch"));
    }
    Ok(SoftplusPlan {
        input_dtype,
        output_dtype,
        shape,
        beta,
    })
}

fn global_average_pool_plan(g: &Graph, input: NodeId) -> Result<GlobalAveragePoolPlan> {
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("GlobalAveragePool input byte extent overflow"))?;
    let axes = (2..shape.rank())
        .map(|axis| axis as isize)
        .collect::<Vec<_>>();
    let count = shape.dims()[2..]
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| bad("GlobalAveragePool divisor overflow"))?;
    let sum_dtypes = ReductionDType::sum_default(input_dtype);
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let work_dtype = if input_dtype.is_float() {
        sum_dtypes.accumulator
    } else {
        DType::F32
    };
    let mut output_dims = shape.dims().to_vec();
    for dim in output_dims.iter_mut().skip(2) {
        *dim = 1;
    }
    let output_shape = Shape::new(output_dims);
    // `Tensor.mean` casts to its sum accumulator before the reduction, even
    // when GlobalAveragePool's trailing spatial axis tuple is empty.  Validate
    // every same-shaped stage before either a cast, reduction, or divisor
    // constant can be published.
    output_shape
        .numel()?
        .checked_mul(sum_dtypes.accumulator.itemsize())
        .ok_or_else(|| bad("GlobalAveragePool accumulator byte extent overflow"))?;
    output_shape
        .numel()?
        .checked_mul(work_dtype.itemsize())
        .ok_or_else(|| bad("GlobalAveragePool division byte extent overflow"))?;
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("GlobalAveragePool output byte extent overflow"))?;
    let divisor = TensorData::scalar_with_dtype(Scalar::F(count as f64), work_dtype);
    divisor
        .shape()
        .numel()?
        .checked_mul(divisor.dtype().itemsize())
        .ok_or_else(|| bad("GlobalAveragePool divisor byte extent overflow"))?;
    if output_shape.broadcast_with(divisor.shape())? != output_shape
        || output_dtype.promote(output_dtype) != output_dtype
    {
        return Err(bad("GlobalAveragePool scalar promotion mismatch"));
    }
    Ok(GlobalAveragePoolPlan {
        axes,
        sum_dtypes,
        work_dtype,
        output_dtype,
        divisor,
        output_shape,
    })
}

fn mod_plan(
    g: &Graph,
    lhs: NodeId,
    rhs: NodeId,
    rhs_name: &str,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ModPlan> {
    if attrs.keys().any(|key| key != "fmod") {
        return Err(bad("unsupported Mod attribute"));
    }
    let fmod = strict_typed_scalar_i64_attr(n, "fmod")?.unwrap_or(0) != 0;
    let lhs_shape = g.shape(lhs)?.clone();
    let rhs_shape = g.shape(rhs)?.clone();
    let lhs_dtype = g.dtype(lhs)?;
    let rhs_dtype = g.dtype(rhs)?;
    lhs_shape
        .numel()?
        .checked_mul(lhs_dtype.itemsize())
        .ok_or_else(|| bad("Mod lhs byte extent overflow"))?;
    rhs_shape
        .numel()?
        .checked_mul(rhs_dtype.itemsize())
        .ok_or_else(|| bad("Mod rhs byte extent overflow"))?;
    // Tensor.mod/fmod first use `_broadcasted`, whose I64/U64 meet is
    // tinygrad's default F32 rather than RustGrad's generic F64 fallback.
    // Keep the import plan's descriptor and byte checks aligned with the
    // source casts that Graph::modulo/fmod will emit.
    let dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    let shape = lhs_shape.broadcast_with(&rhs_shape)?;
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Mod output byte extent overflow"))?;
    if dtype.is_integer() {
        if let Some(value) = constants.get(rhs_name) {
            if value.dtype().is_integer()
                && (0..value.len()).any(|i| value.scalar_at(i).as_i64() == 0)
            {
                return Err(bad("Mod integer divisor constant contains zero"));
            }
        }
    }
    Ok(ModPlan { fmod, shape, dtype })
}

fn swish_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<SwishPlan> {
    if attrs.keys().any(|key| key != "alpha") {
        return Err(bad("unsupported Swish attribute"));
    }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1.0);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Swish {what} byte extent overflow")))
            .map(|_| ())
    };
    extent(&shape, input_dtype, "input")?;
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    // The source is `x * (x * alpha).sigmoid()`: after its nonfloat cast,
    // each named elementwise sigmoid stage has the same output descriptor.
    // Resolve them all before any cast or constant is published.
    for what in [
        "cast/work",
        "inner multiply",
        "sigmoid exponent",
        "Exp2",
        "sigmoid denominator",
        "reciprocal",
        "outer multiply/output",
    ] {
        extent(&shape, output_dtype, what)?;
    }
    let alpha = TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), output_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
    let neg_inv_ln2 =
        TensorData::scalar_with_dtype(Scalar::F(-1.0 / std::f64::consts::LN_2), output_dtype);
    for scalar in [&alpha, &one, &neg_inv_ln2] {
        extent(scalar.shape(), scalar.dtype(), "scalar")?;
        if scalar.dtype() != output_dtype || shape.broadcast_with(scalar.shape())? != shape {
            return Err(bad("Swish scalar promotion mismatch"));
        }
    }
    if output_dtype.promote(output_dtype) != output_dtype {
        return Err(bad("Swish output promotion mismatch"));
    }
    Ok(SwishPlan {
        input_dtype,
        output_dtype,
        shape,
        alpha,
        one,
        neg_inv_ln2,
        empty: numel == 0,
    })
}

fn selu_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<SeluPlan> {
    if attrs.keys().any(|key| key != "alpha" && key != "gamma") {
        return Err(bad("unsupported Selu attribute"));
    }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1.67326);
    let gamma = typed_scalar_f32_attr(n, "gamma")?.unwrap_or(1.0507);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("Selu input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    numel
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Selu output byte extent overflow"))?;
    let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), output_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
    let alpha = TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), output_dtype);
    let gamma = TensorData::scalar_with_dtype(Scalar::F(f64::from(gamma)), output_dtype);
    for scalar in [&zero, &one, &alpha, &gamma] {
        if scalar.dtype() != output_dtype || shape.broadcast_with(scalar.shape())? != shape {
            return Err(bad("Selu scalar promotion mismatch"));
        }
    }
    if output_dtype.promote(output_dtype) != output_dtype {
        return Err(bad("Selu output promotion mismatch"));
    }
    Ok(SeluPlan {
        input_dtype,
        output_dtype,
        shape,
        zero,
        one,
        alpha,
        gamma,
        empty: numel == 0,
    })
}

fn elu_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<EluPlan> {
    if attrs.keys().any(|key| key != "alpha") {
        return Err(bad("unsupported Elu attribute"));
    }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1.0);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("Elu input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    numel
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Elu output byte extent overflow"))?;
    let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), output_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
    let alpha = TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), output_dtype);
    for scalar in [&zero, &one, &alpha] {
        if scalar.dtype() != output_dtype || shape.broadcast_with(scalar.shape())? != shape {
            return Err(bad("Elu scalar promotion mismatch"));
        }
    }
    if output_dtype.promote(output_dtype) != output_dtype {
        return Err(bad("Elu output promotion mismatch"));
    }
    Ok(EluPlan {
        input_dtype,
        output_dtype,
        shape,
        zero,
        one,
        alpha,
        empty: numel == 0,
    })
}

fn lrn_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<LrnPlan> {
    if attrs
        .keys()
        .any(|key| key != "size" && key != "alpha" && key != "beta" && key != "bias")
    {
        return Err(bad("unsupported LRN attribute"));
    }
    let raw_size =
        strict_typed_scalar_i64_attr(n, "size")?.ok_or_else(|| bad("LRN requires size"))?;
    let size = usize::try_from(raw_size).map_err(|_| bad("LRN size must be positive"))?;
    if size == 0 {
        return Err(bad("LRN size must be positive"));
    }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1e-4);
    let beta = typed_scalar_f32_attr(n, "beta")?.unwrap_or(0.75);
    let bias = typed_scalar_f32_attr(n, "bias")?.unwrap_or(1.0);

    let input_shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    if input_shape.rank() != 4 {
        return Err(bad("LRN requires rank-four NCHW input"));
    }
    let input_numel = input_shape.numel()?;
    input_numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("LRN input byte extent overflow"))?;
    let [batch, channels, height, width]: [usize; 4] = input_shape
        .dims()
        .try_into()
        .expect("rank-four input preflighted");
    let flattened = height
        .checked_mul(width)
        .ok_or_else(|| bad("LRN spatial extent overflow"))?;
    let reshaped = Shape::new([batch, 1, channels, flattened]);
    if reshaped.numel()? != input_numel {
        return Err(bad("LRN reshape changes element count"));
    }
    reshaped
        .numel()?
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("LRN reshape byte extent overflow"))?;

    let before = (size - 1) / 2;
    let after = size / 2;
    let padded_channels = channels
        .checked_add(before)
        .and_then(|value| value.checked_add(after))
        .ok_or_else(|| bad("LRN channel padding overflow"))?;
    let padded = Shape::new([batch, 1, padded_channels, flattened]);
    padded.numel()?;
    padded
        .numel()?
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("LRN padded byte extent overflow"))?;

    let sum_dtypes = ReductionDType::sum_default(input_dtype);
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let pool_dtype = if input_dtype.is_float() {
        sum_dtypes.accumulator
    } else {
        DType::F32
    };
    let narrow_pool = matches!(input_dtype, DType::F16 | DType::BF16);
    let window_shape = Shape::new([batch, 1, channels, flattened]);
    window_shape.numel()?;
    let stack_shape = Shape::new([batch, 1, channels, flattened, size]);
    stack_shape.numel()?;
    stack_shape
        .numel()?
        .checked_mul(sum_dtypes.accumulator.itemsize())
        .ok_or_else(|| bad("LRN stacked byte extent overflow"))?;
    let reduced_shape = window_shape.clone();
    reduced_shape.numel()?;
    reduced_shape
        .numel()?
        .checked_mul(sum_dtypes.accumulator.itemsize())
        .ok_or_else(|| bad("LRN reduced byte extent overflow"))?;
    let pooled_shape = input_shape.clone();
    pooled_shape.numel()?;
    pooled_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("LRN output byte extent overflow"))?;

    let scalar_shape = Shape::new([]);
    if pooled_shape.broadcast_with(&scalar_shape)? != pooled_shape {
        return Err(bad("LRN scalar cannot broadcast to pooled tensor"));
    }
    if output_dtype.promote(output_dtype) != output_dtype
        || pool_dtype.promote(pool_dtype) != pool_dtype
    {
        return Err(bad("LRN scalar promotion mismatch"));
    }
    let divisor = TensorData::scalar_with_dtype(Scalar::F(size as f64), pool_dtype);
    let alpha = TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), output_dtype);
    let beta = TensorData::scalar_with_dtype(Scalar::F(f64::from(beta)), output_dtype);
    let bias = TensorData::scalar_with_dtype(Scalar::F(f64::from(bias)), output_dtype);
    if input_numel == 0 {
        return Ok(LrnPlan {
            input_dtype,
            reshaped,
            padding: vec![(0, 0), (0, 0), (before, after), (0, 0)],
            windows: Vec::new(),
            sum_dtypes,
            pool_dtype,
            output_dtype,
            narrow_pool,
            divisor,
            alpha,
            beta,
            bias,
            output_shape: input_shape,
            empty: true,
        });
    }

    // Stride bounds are checked now, including their isize representation,
    // before `Graph::pad` can append the first node.
    isize::try_from(size - 1).map_err(|_| bad("LRN window offset overflow"))?;
    let mut windows = Vec::with_capacity(size);
    for offset in 0..size {
        let end = offset
            .checked_add(channels)
            .ok_or_else(|| bad("LRN window extent overflow"))?;
        if end > padded_channels {
            return Err(bad("LRN window exceeds padded channels"));
        }
        let start = isize::try_from(offset).map_err(|_| bad("LRN window offset overflow"))?;
        let end = isize::try_from(end).map_err(|_| bad("LRN window extent overflow"))?;
        windows.push(vec![
            Slice {
                start: None,
                stop: None,
                step: 1,
            },
            Slice {
                start: None,
                stop: None,
                step: 1,
            },
            Slice {
                start: Some(start),
                stop: Some(end),
                step: 1,
            },
            Slice {
                start: None,
                stop: None,
                step: 1,
            },
        ]);
    }

    Ok(LrnPlan {
        input_dtype,
        reshaped,
        padding: vec![(0, 0), (0, 0), (before, after), (0, 0)],
        windows,
        sum_dtypes,
        pool_dtype,
        output_dtype,
        narrow_pool,
        divisor,
        alpha,
        beta,
        bias,
        output_shape: input_shape,
        empty: input_numel == 0,
    })
}

fn fast_gelu_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<FastGeluPlan> {
    if !(1..=2).contains(&ins.len()) || !attrs.is_empty() {
        return Err(bad(
            "FastGelu requires one input, optional bias, and no attributes",
        ));
    }
    let input = values
        .get(ins[0])
        .copied()
        .ok_or_else(|| bad("missing ONNX FastGelu input"))?;
    let input_shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("FastGelu {what} byte extent overflow")))
            .map(|_| ())
    };
    extent(&input_shape, input_dtype, "input")?;

    let bias = ins
        .get(1)
        .filter(|name| !name.is_empty())
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX FastGelu bias"))
        })
        .transpose()?;
    let (gelu_input_shape, gelu_input_dtype) = if let Some(bias) = bias {
        let bias_shape = g.shape(bias)?.clone();
        let bias_dtype = g.dtype(bias)?;
        extent(&bias_shape, bias_dtype, "bias")?;
        let shape = input_shape.broadcast_with(&bias_shape)?;
        let dtype = prelu_dtype(input_dtype, bias_dtype);
        // Add materializes both source-LUB casts and its storage-width result.
        extent(&input_shape, dtype, "input cast")?;
        extent(&bias_shape, dtype, "bias cast")?;
        extent(&shape, dtype, "add output")?;
        (shape, dtype)
    } else {
        (input_shape, input_dtype)
    };
    let output_dtype = if gelu_input_dtype.is_float() {
        gelu_input_dtype
    } else {
        DType::F32
    };
    // Graph::gelu's tanh plan owns seven typed weak constants and fourteen
    // source-width intermediates.  Prove every one before Add or GELU can
    // publish a cast, constant, or ALU node.
    for _ in 0..14 {
        extent(&gelu_input_shape, output_dtype, "tanh intermediate")?;
    }
    for value in [
        0.5,
        1.0,
        2.0,
        (2.0 / std::f64::consts::PI).sqrt(),
        0.044_715,
        3.0,
        -1.0 / std::f64::consts::LN_2,
    ] {
        let scalar = TensorData::scalar_with_dtype(Scalar::F(value), output_dtype);
        extent(scalar.shape(), scalar.dtype(), "tanh scalar")?;
        if scalar.dtype() != output_dtype
            || gelu_input_shape.broadcast_with(scalar.shape())? != gelu_input_shape
        {
            return Err(bad("FastGelu scalar promotion mismatch"));
        }
    }
    Ok(FastGeluPlan {
        input,
        bias,
        gelu_input_shape,
        gelu_input_dtype,
        output_dtype,
    })
}

fn bias_gelu_plan(
    g: &Graph,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<BiasGeluPlan> {
    if ins.len() != 2 || attrs.keys().any(|key| key != "approximate") {
        return Err(bad("BiasGelu requires two inputs and only approximate"));
    }
    // `BiasGelu` delegates to the ONNX Gelu handler, whose omitted optional
    // string selects exact Erf rather than Tensor.gelu's public tanh default.
    let mode = strict_typed_string_attr(n, "approximate")?.unwrap_or_else(|| "none".into());
    if mode != "none" && mode != "tanh" {
        return Err(bad("unsupported BiasGelu approximation"));
    }
    let add = add_plan(g, ins, &BTreeMap::new(), values)?;
    let output_dtype = if add.output_dtype.is_float() {
        add.output_dtype
    } else {
        DType::F32
    };
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("BiasGelu {what} byte extent overflow")))
            .map(|_| ())
    };
    // `Graph::gelu` owns the literal arithmetic, but it runs after Add. Close
    // its cast, every source-width intermediate, and weak constants here so a
    // late GELU overflow cannot leave a published Add behind.
    extent(&add.output_shape, add.output_dtype, "add input")?;
    let operations = if mode == "none" { 6 } else { 14 };
    for _ in 0..operations {
        extent(&add.output_shape, output_dtype, "GELU intermediate")?;
    }
    for value in if mode == "none" {
        vec![0.5, 1.0, std::f64::consts::SQRT_2]
    } else {
        vec![
            0.5,
            1.0,
            2.0,
            (2.0 / std::f64::consts::PI).sqrt(),
            0.044_715,
            3.0,
            -1.0 / std::f64::consts::LN_2,
        ]
    } {
        let scalar = TensorData::scalar_with_dtype(Scalar::F(value), output_dtype);
        extent(scalar.shape(), scalar.dtype(), "GELU scalar")?;
        if scalar.dtype() != output_dtype
            || add.output_shape.broadcast_with(scalar.shape())? != add.output_shape
        {
            return Err(bad("BiasGelu scalar promotion mismatch"));
        }
    }
    Ok(BiasGeluPlan {
        add,
        mode,
        output_dtype,
    })
}

fn center_crop_pad_zero(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(0),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U(0),
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(0.0),
    }
}

fn center_crop_pad_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<CenterCropPadPlan> {
    if attrs.keys().any(|key| key != "axes") {
        return Err(bad("unsupported CenterCropPad attribute"));
    }
    let input_shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    let input_numel = input_shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("CenterCropPad input byte extent overflow"))?;
    let shape_data = constants
        .get(ins[1])
        .ok_or_else(|| bad("CenterCropPad shape must be a constant initializer"))?;
    if shape_data.dtype() != DType::I64 || shape_data.shape().rank() != 1 {
        return Err(bad("CenterCropPad shape must be a rank-one I64 constant"));
    }
    shape_data.shape().numel()?;
    let targets = const_i64(constants, ins[1])?;
    let rank = input_shape.rank();
    let default_axes = || {
        (0..rank)
            .map(|axis| i64::try_from(axis).map_err(|_| bad("CenterCropPad rank overflow")))
            .collect::<Result<Vec<_>>>()
    };
    let axes = match strict_typed_packed_i64_attr(n, "axes")? {
        None => default_axes()?,
        Some(values) if values.is_empty() => default_axes()?,
        Some(values) => values,
    };

    let mut bounds = input_shape
        .dims()
        .iter()
        .map(|&dimension| (0, dimension))
        .collect::<Vec<_>>();
    let mut padding = vec![(0usize, 0usize); rank];
    for (&target, &raw_axis) in targets.iter().zip(&axes) {
        let axis = if raw_axis < 0 {
            raw_axis
                .checked_add(i64::try_from(rank).map_err(|_| bad("CenterCropPad rank overflow"))?)
                .ok_or_else(|| bad("invalid CenterCropPad axis"))?
        } else {
            raw_axis
        };
        let axis = usize::try_from(axis)
            .ok()
            .filter(|&axis| axis < rank)
            .ok_or_else(|| bad("invalid CenterCropPad axis"))?;
        let target = usize::try_from(target)
            .map_err(|_| bad("CenterCropPad target extent must be nonnegative"))?;
        let source = input_shape.dims()[axis];
        // Every iteration overwrites the previous entry for this axis, just
        // as the source's `shrink_arg[x] = ...` / `pad_arg[x] = ...` does.
        bounds[axis] = (0, source);
        padding[axis] = (0, 0);
        if target < source {
            let start = source / 2 - (target / 2 + target % 2);
            let end = (source / 2)
                .checked_add(target / 2)
                .ok_or_else(|| bad("CenterCropPad crop extent overflow"))?;
            bounds[axis] = (start, end);
        } else if target > source {
            let difference = target - source;
            padding[axis] = (difference / 2, difference / 2 + difference % 2);
        }
    }

    let shrink_shape = Shape::new(
        bounds
            .iter()
            .map(|(start, end)| end - start)
            .collect::<Vec<_>>(),
    );
    let shrink_numel = shrink_shape.numel()?;
    shrink_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("CenterCropPad shrink byte extent overflow"))?;
    let output_shape = Shape::new(
        shrink_shape
            .dims()
            .iter()
            .zip(&padding)
            .map(|(dimension, (before, after))| {
                dimension
                    .checked_add(*before)
                    .and_then(|value| value.checked_add(*after))
                    .ok_or_else(|| bad("CenterCropPad output extent overflow"))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let output_numel = output_shape.numel()?;
    output_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("CenterCropPad output byte extent overflow"))?;
    let shrink = (bounds
        != input_shape
            .dims()
            .iter()
            .map(|&dimension| (0, dimension))
            .collect::<Vec<_>>())
    .then_some(bounds);
    let padding = padding
        .iter()
        .any(|&(before, after)| before != 0 || after != 0)
        .then_some(padding);
    Ok(CenterCropPadPlan {
        shrink,
        padding,
        fill: center_crop_pad_zero(dtype),
    })
}

fn depth_to_space_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<DepthToSpacePlan> {
    if attrs.keys().any(|key| key != "blocksize" && key != "mode") {
        return Err(bad("unsupported DepthToSpace attribute"));
    }
    let raw_blocksize = strict_typed_scalar_i64_attr(n, "blocksize")?
        .ok_or_else(|| bad("DepthToSpace requires blocksize"))?;
    let blocksize = usize::try_from(raw_blocksize)
        .map_err(|_| bad("DepthToSpace blocksize must be positive"))?;
    if blocksize == 0 {
        return Err(bad("DepthToSpace blocksize must be positive"));
    }
    let mode = match strict_typed_string_attr(n, "mode")?.as_deref() {
        Some("CRD") => DepthToSpaceMode::Crd,
        _ => DepthToSpaceMode::Dcr,
    };

    let input_shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    if input_shape.rank() != 4 {
        return Err(bad("DepthToSpace requires rank-four NCHW input"));
    }
    let [batch, channels, height, width]: [usize; 4] = input_shape
        .dims()
        .try_into()
        .expect("rank-four input preflighted");
    // tinygrad's `reshape` infers Cout by dividing through B*H*W*s*s. It
    // therefore rejects these otherwise representable empty domains before
    // it can construct the rearrange views.
    if batch == 0 || height == 0 || width == 0 {
        return Err(bad(
            "DepthToSpace source reshape rejects empty batch or spatial extent",
        ));
    }
    let input_numel = input_shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("DepthToSpace input byte extent overflow"))?;
    let block_area = blocksize
        .checked_mul(blocksize)
        .ok_or_else(|| bad("DepthToSpace block area overflow"))?;
    if channels % block_area != 0 {
        return Err(bad(
            "DepthToSpace channels must be divisible by blocksize squared",
        ));
    }
    let output_channels = channels / block_area;
    let output_height = height
        .checked_mul(blocksize)
        .ok_or_else(|| bad("DepthToSpace output height overflow"))?;
    let output_width = width
        .checked_mul(blocksize)
        .ok_or_else(|| bad("DepthToSpace output width overflow"))?;

    let (first_shape, permutation) = match mode {
        // b (h1 w1 c) h w -> b c (h h1) (w w1)
        DepthToSpaceMode::Dcr => (
            Shape::new([batch, blocksize, blocksize, output_channels, height, width]),
            [0, 3, 4, 1, 5, 2],
        ),
        // b (c h1 w1) h w -> b c (h h1) (w w1)
        DepthToSpaceMode::Crd => (
            Shape::new([batch, output_channels, blocksize, blocksize, height, width]),
            [0, 1, 4, 2, 5, 3],
        ),
    };
    let first_numel = first_shape.numel()?;
    if first_numel != input_numel {
        return Err(bad(
            "DepthToSpace intermediate reshape changes element count",
        ));
    }
    first_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("DepthToSpace intermediate byte extent overflow"))?;
    let mut sorted = permutation;
    sorted.sort_unstable();
    if sorted != [0, 1, 2, 3, 4, 5] {
        return Err(bad("invalid DepthToSpace permutation"));
    }

    let output_shape = Shape::new([batch, output_channels, output_height, output_width]);
    let output_numel = output_shape.numel()?;
    if output_numel != input_numel {
        return Err(bad("DepthToSpace output reshape changes element count"));
    }
    output_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("DepthToSpace output byte extent overflow"))?;
    Ok(DepthToSpacePlan {
        first_shape,
        permutation,
        output_shape,
        identity: blocksize == 1,
    })
}

fn space_to_depth_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<SpaceToDepthPlan> {
    if attrs.keys().any(|key| key != "blocksize") {
        return Err(bad("unsupported SpaceToDepth attribute"));
    }
    let raw_blocksize = strict_typed_scalar_i64_attr(n, "blocksize")?
        .ok_or_else(|| bad("SpaceToDepth requires blocksize"))?;
    let blocksize = usize::try_from(raw_blocksize)
        .map_err(|_| bad("SpaceToDepth blocksize must be positive"))?;
    if blocksize == 0 {
        return Err(bad("SpaceToDepth blocksize must be positive"));
    }

    let input_shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    if input_shape.rank() != 4 {
        return Err(bad("SpaceToDepth requires rank-four NCHW input"));
    }
    let input_numel = input_shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("SpaceToDepth input byte extent overflow"))?;
    let [batch, channels, height, width]: [usize; 4] = input_shape
        .dims()
        .try_into()
        .expect("rank-four input preflighted");
    if height % blocksize != 0 || width % blocksize != 0 {
        return Err(bad(
            "SpaceToDepth spatial dimensions must be divisible by blocksize",
        ));
    }
    let reduced_height = height / blocksize;
    let reduced_width = width / blocksize;
    let expanded_channels = channels
        .checked_mul(blocksize)
        .and_then(|value| value.checked_mul(blocksize))
        .ok_or_else(|| bad("SpaceToDepth channel extent overflow"))?;

    // tinygrad: b c (h h1) (w w1) -> b (h1 w1 c) h w
    let first_shape = Shape::new([
        batch,
        channels,
        reduced_height,
        blocksize,
        reduced_width,
        blocksize,
    ]);
    let first_numel = first_shape.numel()?;
    if first_numel != input_numel {
        return Err(bad(
            "SpaceToDepth intermediate reshape changes element count",
        ));
    }
    first_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("SpaceToDepth intermediate byte extent overflow"))?;
    let permutation = [0usize, 3, 5, 1, 2, 4];
    let mut sorted = permutation;
    sorted.sort_unstable();
    if sorted != [0, 1, 2, 3, 4, 5] {
        return Err(bad("invalid SpaceToDepth permutation"));
    }

    let output_shape = Shape::new([batch, expanded_channels, reduced_height, reduced_width]);
    let output_numel = output_shape.numel()?;
    if output_numel != input_numel {
        return Err(bad("SpaceToDepth output reshape changes element count"));
    }
    output_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("SpaceToDepth output byte extent overflow"))?;

    Ok(SpaceToDepthPlan {
        first_shape,
        output_shape,
        identity: blocksize == 1,
    })
}

fn eye_like_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<EyeLikePlan> {
    if attrs.keys().any(|key| key != "dtype" && key != "k") {
        return Err(bad("unsupported EyeLike attribute"));
    }
    let shape = g.shape(input)?.clone();
    if shape.rank() != 2 {
        return Err(bad("EyeLike requires rank-two input"));
    }
    shape
        .numel()?
        .checked_mul(g.dtype(input)?.itemsize())
        .ok_or_else(|| bad("EyeLike input byte extent overflow"))?;
    let rows = shape.dims()[0];
    let columns = shape.dims()[1];
    let rows_i64 = i64::try_from(rows).map_err(|_| bad("EyeLike row extent overflow"))?;
    let columns_i64 = i64::try_from(columns).map_err(|_| bad("EyeLike column extent overflow"))?;
    let dtype = match strict_typed_scalar_i64_attr(n, "dtype")? {
        Some(code) => onnx_dtype(u64::try_from(code).map_err(|_| bad("unsupported ONNX dtype"))?)?,
        None => g.dtype(input)?,
    };
    let k = strict_typed_scalar_i64_attr(n, "k")?.unwrap_or(0);
    let output_shape = Shape::new([rows, columns]);
    let output_numel = output_shape.numel()?;
    output_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("EyeLike output byte extent overflow"))?;

    let common = rows.min(columns);
    if rows != columns {
        let common_i64 = i64::try_from(common).map_err(|_| bad("EyeLike extent overflow"))?;
        let lower = common_i64
            .checked_neg()
            .ok_or_else(|| bad("EyeLike diagonal overflow"))?;
        let upper = rows_i64.max(columns_i64);
        if k < lower || k > upper {
            return Err(bad("EyeLike diagonal crops beyond rectangular identity"));
        }
    }

    // `Tensor.eye(min(shape))` is returned unchanged for square inputs, so k
    // is intentionally ignored there. For rectangles, tinygrad pads only the
    // larger dimension: wide matrices use col-row=k, tall use row-col=k.
    let data = TensorData::from_scalars(
        output_shape,
        dtype,
        (0..output_numel).map(|flat| {
            let row = flat / columns;
            let column = flat % columns;
            let on_diagonal = if rows == columns {
                row == column
            } else if rows < columns {
                i64::try_from(column).expect("preflighted column extent")
                    - i64::try_from(row).expect("preflighted row extent")
                    == k
            } else {
                i64::try_from(row).expect("preflighted row extent")
                    - i64::try_from(column).expect("preflighted column extent")
                    == k
            };
            Scalar::I(on_diagonal as i64)
        }),
    )?;
    Ok(EyeLikePlan { data })
}

fn shrink_activation_plan(
    g: &Graph,
    input: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<ShrinkActivationPlan> {
    if attrs.keys().any(|key| key != "bias" && key != "lambd") {
        return Err(bad("unsupported Shrink attribute"));
    }
    // Keep the FLOAT field/type validation separate from the raw attribute
    // map: an INT/STRING/TENSOR payload must not masquerade as a scalar.
    let bias = typed_scalar_f32_attr(n, "bias")?.unwrap_or(0.0);
    let lambd = typed_scalar_f32_attr(n, "lambd")?.unwrap_or(0.5);
    let input_shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let extent = |dtype: DType, what: &str| {
        input_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Shrink {what} byte extent overflow")))
    };
    extent(input_dtype, "input")?;
    let output_dtype = match input_dtype {
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => input_dtype,
        _ => DType::F32,
    };
    let work_dtype = if input_dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    if work_dtype.promote(work_dtype) != work_dtype {
        return Err(bad("Shrink scalar arithmetic promotion mismatch"));
    }
    let narrow = matches!(output_dtype, DType::F16 | DType::BF16);
    let scalar_shape = Shape::new([]);
    let comparison_shape = input_shape.broadcast_with(&scalar_shape)?;
    comparison_shape.numel()?;
    let branch_shape = input_shape.broadcast_with(&scalar_shape)?;
    branch_shape.numel()?;
    if comparison_shape != input_shape || branch_shape != input_shape {
        return Err(bad("Shrink scalar broadcast does not preserve input shape"));
    }
    // Each narrow arithmetic branch is rounded before its mask product. The
    // Bool mask then promotes to the narrow branch width, and the final Add
    // stays at that same width.  Other inputs work directly at F32/F64.
    let branch_dtype = output_dtype;
    if DType::Bool.promote(branch_dtype) != output_dtype
        || output_dtype.promote(output_dtype) != output_dtype
    {
        return Err(bad("Shrink branch promotion mismatch"));
    }
    let product_shape = comparison_shape.broadcast_with(&branch_shape)?;
    product_shape.numel()?;
    let output_shape = product_shape.broadcast_with(&product_shape)?;
    output_shape.numel()?;
    if output_shape != input_shape {
        return Err(bad("Shrink result shape does not preserve input"));
    }
    // The literal graph contains a work cast, two Bool predicates, two
    // arithmetic branches, optional narrow casts, two mask products, and
    // the final Add. Prove every dense descriptor before its first node.
    extent(work_dtype, "work")?;
    extent(DType::Bool, "lower predicate")?;
    extent(DType::Bool, "upper predicate")?;
    extent(work_dtype, "lower branch")?;
    extent(work_dtype, "upper branch")?;
    extent(output_dtype, "lower product")?;
    extent(output_dtype, "upper product")?;
    extent(output_dtype, "output")?;
    Ok(ShrinkActivationPlan {
        work_dtype,
        output_dtype,
        narrow,
        // Unary negation happens on the source FLOAT payload before weak
        // promotion, preserving signed zero and every IEEE special payload.
        negative_lambda: TensorData::scalar_with_dtype(Scalar::F(f64::from(-lambd)), work_dtype),
        lambda: TensorData::scalar_with_dtype(Scalar::F(f64::from(lambd)), work_dtype),
        bias: TensorData::scalar_with_dtype(Scalar::F(f64::from(bias)), work_dtype),
        output_shape,
    })
}

/// Fully validated static contract for tinygrad's ONNX CumSum adapter.  The
/// source adapter resolves one constant axis, optionally reverses it, shifts
/// it through `pad(...).shrink(...)` for exclusivity, then applies `cumsum`.
/// Keep all movement and prefix-reduction facts here so no graph node is
/// appended before an invalid input is rejected.
struct CumSumPlan {
    axis: isize,
    reverse: bool,
    exclusive: bool,
    padding: Option<Vec<(usize, usize)>>,
    shrink: Option<Vec<(usize, usize)>>,
    fill: Scalar,
}

fn static_i32_i64_scalar(
    constants: &BTreeMap<String, TensorData>,
    name: &str,
    operator: &str,
) -> Result<i64> {
    let value = constants
        .get(name)
        .ok_or_else(|| bad(format!("{operator} value must be a constant initializer")))?;
    if !matches!(value.dtype(), DType::I32 | DType::I64) {
        return Err(bad(format!("{operator} value must be I32 or I64")));
    }
    let shape = value.shape();
    shape.numel()?;
    if !(shape.rank() == 0 || (shape.rank() == 1 && shape.dims() == &[1])) || value.len() != 1 {
        return Err(bad(format!(
            "{operator} value must be a scalar or length-one rank-1 tensor"
        )));
    }
    Ok(value.scalar_at(0).as_i64())
}

/// Static rank-one control arrays for tinygrad's ONNX Slice adapter.  The
/// adapter receives Python lists, so checked I32 and I64 initializers are the
/// complete locally representable integer source surface.
fn static_slice_control(
    constants: &BTreeMap<String, TensorData>,
    name: &str,
    control: &str,
) -> Result<Vec<i64>> {
    let value = constants
        .get(name)
        .ok_or_else(|| bad(format!("Slice {control} must be a constant initializer")))?;
    if !matches!(value.dtype(), DType::I32 | DType::I64) {
        return Err(bad(format!("Slice {control} must be I32 or I64")));
    }
    let shape = value.shape();
    let numel = shape.numel()?;
    numel
        .checked_mul(value.dtype().itemsize())
        .ok_or_else(|| bad(format!("Slice {control} byte extent overflow")))?;
    if shape.rank() != 1 {
        return Err(bad(format!("Slice {control} must be rank one")));
    }
    Ok((0..value.len())
        .map(|index| value.scalar_at(index).as_i64())
        .collect())
}

/// Fully preflighted static ONNX Reshape descriptor.  tinygrad receives its
/// shape as a Python list and implements `allowzero` by substituting either a
/// source extent or a literal zero before its ordinary concrete reshape.
struct ReshapePlan {
    output_shape: Shape,
    dtype: DType,
    identity: bool,
}

/// Fully resolved two-dimensional descriptor for tinygrad's ONNX Flatten:
/// `x.reshape(prod(x.shape[:axis]), -1)`. Python prefix slicing clamps any
/// signed axis rather than using Graph::flatten's rank-preserving API.
struct FlattenPlan {
    output_shape: Shape,
    dtype: DType,
    identity: bool,
}

/// Fully resolved tinygrad ONNX Transpose descriptor.  The adapter uses
/// `perm or reversed(range(rank))`, so an explicitly empty INTS attribute has
/// the same source meaning as an omitted permutation.
struct TransposePlan {
    axes: Vec<usize>,
    output_shape: Shape,
    dtype: DType,
    identity: bool,
}

/// Fully resolved tinygrad ONNX Squeeze descriptor.  An omitted optional axes
/// input calls `squeeze()` (all singleton dimensions); an explicit empty list
/// is the identity because tinygrad folds no per-axis squeeze operations.
struct SqueezePlan {
    output_shape: Shape,
    dtype: DType,
    identity: bool,
}

/// Fully resolved tinygrad ONNX Unsqueeze descriptor. The adapter sorts raw
/// host-list axes ascending, then resolves each one after the prior inserted
/// singleton has changed the rank.
struct UnsqueezePlan {
    output_shape: Shape,
    dtype: DType,
    identity: bool,
}

fn unsqueeze_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<UnsqueezePlan> {
    if ins.len() != 2 || !attrs.is_empty() || ins[1].is_empty() {
        return Err(bad(
            "Unsqueeze requires a static axes input and no attributes",
        ));
    }
    let source_shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    source_shape.numel()?;
    source_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Unsqueeze input byte extent overflow"))?;
    let mut axes = static_slice_control(constants, ins[1], "axes")?;
    axes.sort_unstable();
    let mut dims = source_shape.dims().to_vec();
    for raw_axis in axes {
        let rank = dims
            .len()
            .checked_add(1)
            .and_then(|rank| i64::try_from(rank).ok())
            .ok_or_else(|| bad("Unsqueeze rank overflow"))?;
        let axis = if raw_axis < 0 {
            raw_axis
                .checked_add(rank)
                .ok_or_else(|| bad("invalid Unsqueeze axis"))?
        } else {
            raw_axis
        };
        if axis < 0 || axis >= rank {
            return Err(bad("invalid Unsqueeze axis"));
        }
        dims.insert(
            usize::try_from(axis).map_err(|_| bad("invalid Unsqueeze axis"))?,
            1,
        );
    }
    let output_shape = Shape::new(dims);
    output_shape.numel()?;
    output_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Unsqueeze output byte extent overflow"))?;
    Ok(UnsqueezePlan {
        identity: output_shape == source_shape,
        output_shape,
        dtype,
    })
}

fn squeeze_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<SqueezePlan> {
    if !(1..=2).contains(&ins.len()) || !attrs.is_empty() {
        return Err(bad(
            "Squeeze requires one optional axes input and no attributes",
        ));
    }
    let source_shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    source_shape.numel()?;
    source_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Squeeze input byte extent overflow"))?;
    let output_shape = if ins.len() == 1 || ins[1].is_empty() {
        Shape::new(
            source_shape
                .dims()
                .iter()
                .copied()
                .filter(|&extent| extent != 1)
                .collect::<Vec<_>>(),
        )
    } else {
        let mut axes = static_slice_control(constants, ins[1], "axes")?;
        axes.sort_unstable_by(|left, right| right.cmp(left));
        let mut dims = source_shape.dims().to_vec();
        for raw_axis in axes {
            let rank = i64::try_from(dims.len()).map_err(|_| bad("Squeeze rank overflow"))?;
            let axis = if raw_axis < 0 {
                raw_axis
                    .checked_add(rank)
                    .ok_or_else(|| bad("invalid Squeeze axis"))?
            } else {
                raw_axis
            };
            if axis < 0 || axis >= rank {
                return Err(bad("invalid Squeeze axis"));
            }
            let axis = usize::try_from(axis).map_err(|_| bad("invalid Squeeze axis"))?;
            if dims[axis] == 1 {
                dims.remove(axis);
            }
        }
        Shape::new(dims)
    };
    output_shape.numel()?;
    output_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Squeeze output byte extent overflow"))?;
    Ok(SqueezePlan {
        identity: output_shape == source_shape,
        output_shape,
        dtype,
    })
}

fn transpose_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<TransposePlan> {
    if ins.len() != 1 || attrs.keys().any(|name| name != "perm") {
        return Err(bad("Transpose requires one input and only perm"));
    }
    let source_shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    source_shape.numel()?;
    source_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Transpose input byte extent overflow"))?;
    let raw_axes = strict_typed_packed_i64_attr(n, "perm")?;
    let raw_axes = raw_axes.filter(|axes| !axes.is_empty()).unwrap_or_else(|| {
        (0..source_shape.rank())
            .rev()
            .map(|axis| axis as i64)
            .collect()
    });
    if raw_axes.len() != source_shape.rank() {
        return Err(bad("Transpose permutation must match input rank"));
    }
    let axes = axes_usize(&raw_axes, source_shape.rank())?;
    let mut sorted = axes.clone();
    sorted.sort_unstable();
    if sorted != (0..source_shape.rank()).collect::<Vec<_>>() {
        return Err(bad("invalid Transpose permutation"));
    }
    let output_shape = Shape::new(
        axes.iter()
            .map(|&axis| source_shape.dims()[axis])
            .collect::<Vec<_>>(),
    );
    output_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Transpose output byte extent overflow"))?;
    Ok(TransposePlan {
        identity: axes.iter().copied().eq(0..source_shape.rank()),
        axes,
        output_shape,
        dtype,
    })
}

/// Complete source-level descriptor for ONNX `Concat`.  tinygrad's `cat`
/// first resolves one common stack dtype, then either flattens a stack or
/// accumulates padded inputs.  Resolve that dtype and every descriptor before
/// creating the casts or the final concat node so a malformed later input
/// cannot publish a prefix of the operation.
struct ConcatPlan {
    inputs: Vec<NodeId>,
    axis: usize,
    output_shape: Shape,
    output_dtype: DType,
    identity: bool,
    lowering: ConcatLowering,
}

/// `Tensor.cat` has two source-level routes. Equal axis extents use one
/// all-input `stack(...).flatten(...)`; otherwise every input is padded into
/// the final shape and `usum` folds those padded values from left to right.
/// The latter is observably not a raw concatenation for narrow mixed dtypes
/// and signed zero, so retain it as a literal graph composition.
enum ConcatLowering {
    Stack,
    PadSum { paddings: Vec<Vec<(usize, usize)>> },
}

fn concat_dtype(dtypes: &[DType]) -> DType {
    debug_assert!(!dtypes.is_empty());
    // `Tensor.stack` uses tinygrad's all-input least-upper lattice, rather
    // than a left-associated binary fold.  The only non-associative corner in
    // RustGrad's supported lattice is I64/U64: tinygrad's weak-float bridge
    // resolves to F32 when no floating input is present, but permits a narrow
    // floating input to select that narrow storage width.
    let has = |dtype| dtypes.contains(&dtype);
    if has(DType::F64) {
        DType::F64
    } else if has(DType::F32) || (has(DType::F16) && has(DType::BF16)) {
        DType::F32
    } else if has(DType::F16) {
        DType::F16
    } else if has(DType::BF16) {
        DType::BF16
    } else if has(DType::I64) && has(DType::U64) {
        DType::F32
    } else {
        dtypes[1..].iter().copied().fold(dtypes[0], DType::promote)
    }
}

fn concat_plan(
    g: &Graph,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<ConcatPlan> {
    if ins.is_empty() || attrs.keys().any(|name| name != "axis") {
        return Err(bad("Concat requires inputs and only an axis attribute"));
    }
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?
        .ok_or_else(|| bad("Concat requires an axis attribute"))?;
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let first_shape = g.shape(inputs[0])?.clone();
    let rank = first_shape.rank();
    let axis = axes_usize(&[raw_axis], rank)?[0];
    let mut axis_extent = 0usize;
    let mut dtypes = Vec::with_capacity(inputs.len());
    let mut shapes = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let shape = g.shape(*input)?.clone();
        let dtype = g.dtype(*input)?;
        let numel = shape.numel()?;
        numel
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Concat input byte extent overflow"))?;
        if shape.rank() != rank
            || shape.dims().iter().enumerate().any(|(dimension, extent)| {
                dimension != axis && *extent != first_shape.dims()[dimension]
            })
        {
            return Err(bad("Concat input shapes disagree outside axis"));
        }
        axis_extent = axis_extent
            .checked_add(shape.dims()[axis])
            .ok_or_else(|| bad("Concat axis extent overflow"))?;
        dtypes.push(dtype);
        shapes.push(shape);
    }
    let mut output_dims = first_shape.dims().to_vec();
    output_dims[axis] = axis_extent;
    let output_shape = Shape::new(output_dims);
    let equal_axis_extents = shapes
        .iter()
        .all(|shape| shape.dims()[axis] == first_shape.dims()[axis]);
    let (output_dtype, lowering) = if equal_axis_extents {
        let output_dtype = concat_dtype(&dtypes);
        output_shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| bad("Concat output byte extent overflow"))?;
        // `Tensor.stack` promotes every input together before flattening.
        for input in &inputs {
            g.shape(*input)?
                .numel()?
                .checked_mul(output_dtype.itemsize())
                .ok_or_else(|| bad("Concat stack cast byte extent overflow"))?;
        }
        (output_dtype, ConcatLowering::Stack)
    } else {
        let mut offsets = 0usize;
        let paddings = shapes
            .iter()
            .zip(&dtypes)
            .map(|(shape, dtype)| {
                let before = offsets;
                offsets = offsets
                    .checked_add(shape.dims()[axis])
                    .ok_or_else(|| bad("Concat axis extent overflow"))?;
                let after = axis_extent
                    .checked_sub(offsets)
                    .ok_or_else(|| bad("Concat axis extent underflow"))?;
                let padding = shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, _)| {
                        if dimension == axis {
                            (before, after)
                        } else {
                            (0, 0)
                        }
                    })
                    .collect::<Vec<_>>();
                // `Tensor.pad` preserves the input storage dtype and fills
                // with its source-typed zero before `usum` starts.
                output_shape
                    .numel()?
                    .checked_mul(dtype.itemsize())
                    .ok_or_else(|| bad("Concat pad byte extent overflow"))?;
                Ok(padding)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut output_dtype = dtypes[0];
        for dtype in dtypes.iter().copied().skip(1) {
            let prior_dtype = output_dtype;
            output_dtype = clip_dtype(prior_dtype, dtype);
            // Each `usum` ADD casts both source-order operands to this stage
            // dtype, then stores a full output-shaped intermediate.
            for operand_dtype in [prior_dtype, output_dtype, dtype, output_dtype] {
                output_shape
                    .numel()?
                    .checked_mul(operand_dtype.itemsize())
                    .ok_or_else(|| bad("Concat usum byte extent overflow"))?;
            }
        }
        (output_dtype, ConcatLowering::PadSum { paddings })
    };
    Ok(ConcatPlan {
        identity: inputs.len() == 1,
        inputs,
        axis,
        output_shape,
        output_dtype,
        lowering,
    })
}

/// Fully resolved source descriptor for ONNX `Where`.  tinygrad requires a
/// Bool condition, promotes the two branches through its least-upper lattice,
/// and only then broadcasts the condition over their common shape.
struct WherePlan {
    condition: NodeId,
    on_true: NodeId,
    on_false: NodeId,
    output_shape: Shape,
    output_dtype: DType,
}

fn where_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<WherePlan> {
    if ins.len() != 3 || !attrs.is_empty() {
        return Err(bad("Where requires exactly three inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let descriptors = inputs
        .iter()
        .map(|&input| {
            let shape = g.shape(input)?.clone();
            let dtype = g.dtype(input)?;
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| bad("Where input byte extent overflow"))?;
            Ok((shape, dtype))
        })
        .collect::<Result<Vec<_>>>()?;
    if descriptors[0].1 != DType::Bool {
        return Err(bad("Where condition must be Bool"));
    }
    let branch_shape = descriptors[1].0.broadcast_with(&descriptors[2].0)?;
    let output_shape = descriptors[0].0.broadcast_with(&branch_shape)?;
    let output_dtype = prelu_dtype(descriptors[1].1, descriptors[2].1);
    for (shape, what) in [
        (&descriptors[1].0, "true branch cast"),
        (&descriptors[2].0, "false branch cast"),
        (&output_shape, "output"),
    ] {
        shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| bad(format!("Where {what} byte extent overflow")))?;
    }
    output_shape
        .numel()?
        .checked_mul(DType::Bool.itemsize())
        .ok_or_else(|| bad("Where condition broadcast byte extent overflow"))?;
    Ok(WherePlan {
        condition: inputs[0],
        on_true: inputs[1],
        on_false: inputs[2],
        output_shape,
        output_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `Equal`.  tinygrad implements
/// equality as promoted `ne(...).logical_not()`, so its observable comparison
/// dtype is the common branch dtype even though the result itself is Bool.
struct EqualPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    comparison_dtype: DType,
}

fn equal_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<EqualPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Equal requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Equal input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let comparison_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [(&lhs_shape, "left cast"), (&rhs_shape, "right cast")] {
        shape
            .numel()?
            .checked_mul(comparison_dtype.itemsize())
            .ok_or_else(|| bad(format!("Equal {what} byte extent overflow")))?;
    }
    output_shape
        .numel()?
        .checked_mul(DType::Bool.itemsize())
        .ok_or_else(|| bad("Equal output byte extent overflow"))?;
    Ok(EqualPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        comparison_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `Less`.  Like tinygrad's
/// literal `x < y`, comparison occurs after the same common-dtype broadcast
/// used by all binary tensor elementwise operators.
struct LessPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    comparison_dtype: DType,
}

fn less_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<LessPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Less requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Less input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let comparison_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [(&lhs_shape, "left cast"), (&rhs_shape, "right cast")] {
        shape
            .numel()?
            .checked_mul(comparison_dtype.itemsize())
            .ok_or_else(|| bad(format!("Less {what} byte extent overflow")))?;
    }
    output_shape
        .numel()?
        .checked_mul(DType::Bool.itemsize())
        .ok_or_else(|| bad("Less output byte extent overflow"))?;
    Ok(LessPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        comparison_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `Greater`.  tinygrad reverses
/// its literal CMPLT operands (`y < x`) after the same common-dtype promotion
/// used by `Less`; the externally visible predicate is therefore `x > y`.
struct GreaterPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    comparison_dtype: DType,
}

fn greater_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<GreaterPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Greater requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Greater input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let comparison_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [(&lhs_shape, "left cast"), (&rhs_shape, "right cast")] {
        shape
            .numel()?
            .checked_mul(comparison_dtype.itemsize())
            .ok_or_else(|| bad(format!("Greater {what} byte extent overflow")))?;
    }
    output_shape
        .numel()?
        .checked_mul(DType::Bool.itemsize())
        .ok_or_else(|| bad("Greater output byte extent overflow"))?;
    Ok(GreaterPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        comparison_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `LessOrEqual`.  The source is
/// not direct LE: it promotes `x > y` and then logically negates that Bool
/// result, which deliberately turns an unordered NaN comparison into true.
struct LessOrEqualPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    comparison_dtype: DType,
}

fn less_or_equal_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<LessOrEqualPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad(
            "LessOrEqual requires exactly two inputs and no attributes",
        ));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("LessOrEqual input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let comparison_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [(&lhs_shape, "left cast"), (&rhs_shape, "right cast")] {
        shape
            .numel()?
            .checked_mul(comparison_dtype.itemsize())
            .ok_or_else(|| bad(format!("LessOrEqual {what} byte extent overflow")))?;
    }
    for what in ["comparison", "output"] {
        output_shape
            .numel()?
            .checked_mul(DType::Bool.itemsize())
            .ok_or_else(|| bad(format!("LessOrEqual {what} byte extent overflow")))?;
    }
    Ok(LessOrEqualPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        comparison_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `GreaterOrEqual`.  tinygrad
/// spells this as `logical_not(x < y)`, retaining true for unordered NaN
/// comparisons after its common-dtype binary promotion.
struct GreaterOrEqualPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    comparison_dtype: DType,
}

fn greater_or_equal_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<GreaterOrEqualPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad(
            "GreaterOrEqual requires exactly two inputs and no attributes",
        ));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("GreaterOrEqual input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let comparison_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [(&lhs_shape, "left cast"), (&rhs_shape, "right cast")] {
        shape
            .numel()?
            .checked_mul(comparison_dtype.itemsize())
            .ok_or_else(|| bad(format!("GreaterOrEqual {what} byte extent overflow")))?;
    }
    for what in ["comparison", "output"] {
        output_shape
            .numel()?
            .checked_mul(DType::Bool.itemsize())
            .ok_or_else(|| bad(format!("GreaterOrEqual {what} byte extent overflow")))?;
    }
    Ok(GreaterOrEqualPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        comparison_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `Add`.  tinygrad promotes both
/// operands through `_broadcasted` before storage-width addition.
struct AddPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    output_dtype: DType,
}

fn add_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<AddPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Add requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Add input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let output_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [
        (&lhs_shape, "left cast"),
        (&rhs_shape, "right cast"),
        (&output_shape, "output"),
    ] {
        shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| bad(format!("Add {what} byte extent overflow")))?;
    }
    Ok(AddPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        output_dtype,
    })
}

/// Descriptor-only lowering for tinygrad's optional-presence query.  An
/// omitted slot is Python `None`; a bound zero-extent tensor is distinctly
/// present but still returns false because the source checks `numel() > 0`.
fn optional_has_element_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<TensorData> {
    if ins.len() > 1 || !attrs.is_empty() {
        return Err(bad(
            "OptionalHasElement accepts zero or one input and no attributes",
        ));
    }
    let present = match ins.first().filter(|name| !name.is_empty()) {
        None => false,
        Some(name) => {
            let input = values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing OptionalHasElement input"))?;
            let shape = g.shape(input)?.clone();
            let dtype = g.dtype(input)?;
            let numel = shape.numel()?;
            numel
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| bad("OptionalHasElement input byte extent overflow"))?;
            numel > 0
        }
    };
    let output = TensorData::scalar_with_dtype(Scalar::Bool(present), DType::Bool);
    output
        .shape()
        .numel()?
        .checked_mul(output.dtype().itemsize())
        .ok_or_else(|| bad("OptionalHasElement output byte extent overflow"))?;
    Ok(output)
}

enum OptionalGetElementPlan {
    Alias(NodeId),
    Empty(TensorData),
}

/// Complete static contract for tinygrad's optional unwrap.  Presence is
/// determined by slot syntax, not tensor numel: zero-extent inputs remain
/// aliases, whereas omitted and explicit-empty slots produce `Tensor([])`.
fn optional_get_element_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<OptionalGetElementPlan> {
    if ins.len() > 1 || !attrs.is_empty() {
        return Err(bad(
            "OptionalGetElement accepts zero or one input and no attributes",
        ));
    }
    match ins.first().filter(|name| !name.is_empty()) {
        Some(name) => {
            let input = values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing OptionalGetElement input"))?;
            let shape = g.shape(input)?.clone();
            let dtype = g.dtype(input)?;
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| bad("OptionalGetElement input byte extent overflow"))?;
            Ok(OptionalGetElementPlan::Alias(input))
        }
        None => {
            // `dtypes.from_py([])` is tinygrad's default float. The literal
            // empty list has rank one and a zero extent, not scalar shape.
            let empty = TensorData::zeros_with_dtype([0], DType::F32)?;
            empty
                .shape()
                .numel()?
                .checked_mul(empty.dtype().itemsize())
                .ok_or_else(|| bad("OptionalGetElement fallback byte extent overflow"))?;
            Ok(OptionalGetElementPlan::Empty(empty))
        }
    }
}

/// Fully resolved source descriptor for ONNX `Sub`.  tinygrad performs
/// source-common dtype casting before ordered storage-width subtraction.
struct SubPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    output_dtype: DType,
}

fn sub_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<SubPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Sub requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Sub input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let output_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [
        (&lhs_shape, "left cast"),
        (&rhs_shape, "right cast"),
        (&output_shape, "output"),
    ] {
        shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| bad(format!("Sub {what} byte extent overflow")))?;
    }
    Ok(SubPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        output_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `Mul`.  tinygrad promotes both
/// operands through `_broadcasted` before storage-width multiplication.
struct MulPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    output_dtype: DType,
}

fn mul_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<MulPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Mul requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Mul input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let output_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    for (shape, what) in [
        (&lhs_shape, "left cast"),
        (&rhs_shape, "right cast"),
        (&output_shape, "output"),
    ] {
        shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| bad(format!("Mul {what} byte extent overflow")))?;
    }
    Ok(MulPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        output_dtype,
    })
}

/// Fully resolved source descriptor for ONNX integer bitwise binary operators.
/// tinygrad delegates each handler to `_broadcasted`, which commits both
/// operands to their least-upper dtype before the Bool/integer ALU operation.
struct BitwiseBinaryPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    output_dtype: DType,
}

fn bitwise_binary_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
    operator: &str,
) -> Result<BitwiseBinaryPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad(format!(
            "{operator} requires exactly two inputs and no attributes"
        )));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("{operator} input byte extent overflow")))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let output_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    if output_dtype.is_float() {
        return Err(bad(format!("{operator} requires Bool or integer operands")));
    }
    for (shape, what) in [
        (&lhs_shape, "left cast"),
        (&rhs_shape, "right cast"),
        (&output_shape, "output"),
    ] {
        shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| bad(format!("{operator} {what} byte extent overflow")))?;
    }
    Ok(BitwiseBinaryPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        output_dtype,
    })
}

/// Fully resolved source descriptor for ONNX `Div`.  tinygrad's adapter uses
/// integer CDIV only when its promoted operands remain integer; every other
/// path is `a * reciprocal(b)`, with an additional truncation when the
/// original left operand was integer.
struct DivPlan {
    lhs: NodeId,
    rhs: NodeId,
    output_shape: Shape,
    work_dtype: DType,
    integer_division: bool,
    truncate: bool,
}

fn div_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<DivPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Div requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let lhs_shape = g.shape(inputs[0])?.clone();
    let rhs_shape = g.shape(inputs[1])?.clone();
    let lhs_dtype = g.dtype(inputs[0])?;
    let rhs_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&lhs_shape, lhs_dtype), (&rhs_shape, rhs_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Div input byte extent overflow"))?;
    }
    let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let promoted_dtype = prelu_dtype(lhs_dtype, rhs_dtype);
    let integer_division = lhs_dtype.is_integer() && promoted_dtype.is_integer();
    let work_dtype = if integer_division || promoted_dtype.is_float() {
        promoted_dtype
    } else {
        DType::F32
    };
    let truncate = lhs_dtype.is_integer() && !integer_division;
    for (shape, what) in [(&lhs_shape, "left cast"), (&rhs_shape, "right cast")] {
        shape
            .numel()?
            .checked_mul(work_dtype.itemsize())
            .ok_or_else(|| bad(format!("Div {what} byte extent overflow")))?;
    }
    let operation_count = if integer_division {
        1
    } else if truncate {
        3
    } else {
        2
    };
    for _ in 0..operation_count {
        output_shape
            .numel()?
            .checked_mul(work_dtype.itemsize())
            .ok_or_else(|| bad("Div intermediate byte extent overflow"))?;
    }
    Ok(DivPlan {
        lhs: inputs[0],
        rhs: inputs[1],
        output_shape,
        work_dtype,
        integer_division,
        truncate,
    })
}

/// Fully resolved source descriptor for ONNX `Pow`.  The Tensor operation
/// promotes base/exponent before POW; the ONNX adapter then rounds and casts
/// back only when the original base dtype is integer.
struct PowPlan {
    base: NodeId,
    exponent: NodeId,
    output_shape: Shape,
    work_dtype: DType,
    output_dtype: DType,
    integer_base: bool,
}

fn pow_plan(
    g: &Graph,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    values: &BTreeMap<String, NodeId>,
) -> Result<PowPlan> {
    if ins.len() != 2 || !attrs.is_empty() {
        return Err(bad("Pow requires exactly two inputs and no attributes"));
    }
    let inputs = ins
        .iter()
        .map(|name| {
            values
                .get(*name)
                .copied()
                .ok_or_else(|| bad("missing ONNX input"))
        })
        .collect::<Result<Vec<_>>>()?;
    let base_shape = g.shape(inputs[0])?.clone();
    let exponent_shape = g.shape(inputs[1])?.clone();
    let base_dtype = g.dtype(inputs[0])?;
    let exponent_dtype = g.dtype(inputs[1])?;
    for (shape, dtype) in [(&base_shape, base_dtype), (&exponent_shape, exponent_dtype)] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Pow input byte extent overflow"))?;
    }
    let output_shape = base_shape.broadcast_with(&exponent_shape)?;
    let work_dtype = prelu_dtype(base_dtype, exponent_dtype);
    let integer_base = base_dtype.is_integer();
    let output_dtype = if integer_base { base_dtype } else { work_dtype };
    for (shape, dtype, what) in [
        (&base_shape, work_dtype, "base cast"),
        (&exponent_shape, work_dtype, "exponent cast"),
        (&output_shape, work_dtype, "power"),
        (&output_shape, work_dtype, "round"),
        (&output_shape, output_dtype, "output"),
    ] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Pow {what} byte extent overflow")))?;
    }
    Ok(PowPlan {
        base: inputs[0],
        exponent: inputs[1],
        output_shape,
        work_dtype,
        output_dtype,
        integer_base,
    })
}

/// Complete source descriptor for ONNX `LeakyRelu`. tinygrad spells the
/// activation as strict `(x < 0).where(alpha * x, x)` with a weak FLOAT alpha
/// that rounds at the input floating storage width.
struct LeakyReluPlan {
    input: NodeId,
    shape: Shape,
    input_dtype: DType,
    output_dtype: DType,
    comparison_zero: TensorData,
    alpha: TensorData,
}

fn leaky_relu_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<LeakyReluPlan> {
    if ins.len() != 1 || attrs.keys().any(|name| name != "alpha") {
        return Err(bad("LeakyRelu requires one input and only alpha"));
    }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(0.01);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("LeakyRelu input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    // `x < 0` remains at x's source width; Bool is compared through the
    // existing Bool scalar semantics, while arithmetic follows weak-FLOAT
    // promotion to F32 for every nonfloating input.
    let comparison_zero = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
    let alpha = TensorData::scalar_with_dtype(Scalar::F(alpha as f64), output_dtype);
    if shape.broadcast_with(comparison_zero.shape())? != shape
        || shape.broadcast_with(alpha.shape())? != shape
    {
        return Err(bad("LeakyRelu scalar broadcast mismatch"));
    }
    for (dtype, what) in [
        (DType::Bool, "comparison"),
        (output_dtype, "input cast"),
        (output_dtype, "scaled branch"),
        (output_dtype, "output"),
    ] {
        numel
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("LeakyRelu {what} byte extent overflow")))?;
    }
    Ok(LeakyReluPlan {
        input,
        shape,
        input_dtype,
        output_dtype,
        comparison_zero,
        alpha,
    })
}

/// Fully validated ONNX `Cast` descriptor.  The source adapter accepts the
/// FP8-only `saturate` attribute but has no observable use for it among the
/// locally supported dtype codes.
struct CastPlan {
    input: NodeId,
    input_dtype: DType,
    output_dtype: DType,
}

fn cast_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<CastPlan> {
    if ins.len() != 1 || attrs.keys().any(|name| name != "to" && name != "saturate") {
        return Err(bad("Cast requires one input and only to or saturate"));
    }
    let raw_to = strict_typed_scalar_i64_attr(n, "to")?.ok_or_else(|| bad("Cast needs to"))?;
    // tinygrad accepts but ignores `saturate`: it applies only to FP8 targets,
    // none of which are present in RustGrad's checked ONNX dtype mapping.
    let _ = strict_typed_scalar_i64_attr(n, "saturate")?;
    let output_dtype =
        onnx_dtype(u64::try_from(raw_to).map_err(|_| bad("unsupported ONNX dtype"))?)?;
    let input_dtype = g.dtype(input)?;
    let numel = g.shape(input)?.numel()?;
    numel
        .checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("Cast input byte extent overflow"))?;
    numel
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Cast output byte extent overflow"))?;
    Ok(CastPlan {
        input,
        input_dtype,
        output_dtype,
    })
}

/// Reads a single explicitly typed Constant AttributeProto payload.  Constant
/// has several concrete payload forms, so the generic normalized attribute
/// bytes are insufficient to prove the declared field and AttributeProto type
/// agree before private TensorData construction.
fn strict_constant_attr(
    n: &Msg<'_>,
    wanted: &str,
    expected_type: u64,
    expected_field: u32,
    expected_wire: u8,
) -> Result<Option<Vec<u8>>> {
    let mut out = None;
    for raw in n.bytes(5)? {
        let attribute = Msg::new(raw);
        if attribute.string(1)? != Some(wanted) {
            continue;
        }
        if out.is_some() {
            return Err(bad("duplicate ONNX attribute"));
        }
        let fields = attribute.fields()?;
        let types: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| *id == 20 && *wire == 0)
            .collect();
        let [(_, _, raw_type)] = types.as_slice() else {
            return Err(bad("Constant attribute must declare its type"));
        };
        let mut at = 0;
        if var(raw_type, &mut at)? != expected_type || at != raw_type.len() {
            return Err(bad("Constant attribute has the wrong declared type"));
        }
        let values: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| {
                (*id == 2 && *wire == 5)
                    || (*id == 3 && *wire == 0)
                    || ((*id == 4 || *id == 5 || *id == 7 || *id == 8) && *wire == 2)
            })
            .collect();
        let [(field, wire, value)] = values.as_slice() else {
            return Err(bad("Constant attribute must have one payload"));
        };
        if *field != expected_field || *wire != expected_wire {
            return Err(bad("Constant attribute has the wrong payload field"));
        }
        out = Some(value.to_vec());
    }
    Ok(out)
}

/// Builds the source-supported ONNX Constant payload privately.  String and
/// sparse forms are intentionally rejected because tinygrad's dispatcher
/// explicitly raises for them and RustGrad has no corresponding dtype/storage.
fn constant_plan(
    n: &Msg<'_>,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<TensorData> {
    if !ins.is_empty() || attrs.len() != 1 {
        return Err(bad(
            "Constant requires zero inputs and one payload attribute",
        ));
    }
    let name = attrs
        .keys()
        .next()
        .expect("Constant attribute count checked");
    let data = match name.as_str() {
        "value" => {
            let raw = strict_constant_attr(n, "value", 4, 5, 2)?
                .ok_or_else(|| bad("Constant needs value"))?;
            tensor_data(Msg::new(&raw))?
        }
        "value_float" => {
            let raw = strict_constant_attr(n, "value_float", 1, 2, 5)?
                .ok_or_else(|| bad("Constant needs value_float"))?;
            let value: [u8; 4] = raw
                .as_slice()
                .try_into()
                .map_err(|_| bad("Constant value_float must be f32"))?;
            TensorData::scalar(f32::from_le_bytes(value))
        }
        "value_floats" => {
            let raw = strict_constant_attr(n, "value_floats", 6, 7, 2)?
                .ok_or_else(|| bad("Constant needs value_floats"))?;
            if raw.len() % 4 != 0 {
                return Err(bad("Constant value_floats has invalid byte length"));
            }
            let count = raw.len() / 4;
            count
                .checked_mul(DType::F32.itemsize())
                .ok_or_else(|| bad("Constant value_floats byte extent overflow"))?;
            let values = raw
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("exact chunks")))
                .collect();
            TensorData::new([count], values)?
        }
        "value_int" => {
            let raw = strict_constant_attr(n, "value_int", 2, 3, 0)?
                .ok_or_else(|| bad("Constant needs value_int"))?;
            let mut at = 0;
            let value = var(&raw, &mut at)?;
            if at != raw.len() {
                return Err(bad("Constant value_int is not a single INT"));
            }
            TensorData::from_scalars([], DType::I64, [Scalar::I(value as i64)])?
        }
        "value_ints" => {
            let raw = strict_constant_attr(n, "value_ints", 7, 8, 2)?
                .ok_or_else(|| bad("Constant needs value_ints"))?;
            let values = packed_i64(&raw)?;
            values
                .len()
                .checked_mul(DType::I64.itemsize())
                .ok_or_else(|| bad("Constant value_ints byte extent overflow"))?;
            TensorData::from_scalars(
                [values.len()],
                DType::I64,
                values.into_iter().map(Scalar::I),
            )?
        }
        "value_string" | "value_strings" | "sparse_value" => {
            return Err(bad("unsupported ONNX Constant payload"));
        }
        _ => return Err(bad("unsupported ONNX Constant attribute")),
    };
    data.shape().numel()?;
    data.shape()
        .numel()?
        .checked_mul(data.dtype().itemsize())
        .ok_or_else(|| bad("Constant output byte extent overflow"))?;
    Ok(data)
}

/// Source promotion used by Tensor's binary batch-normalization composition.
/// The public Graph lattice deliberately differs only for I64/U64, where
/// tinygrad's least-upper dtype remains its default F32 width.
fn batch_norm_dtype(lhs: DType, rhs: DType) -> DType {
    if matches!(
        (lhs, rhs),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

struct BatchNormPlan {
    input: NodeId,
    scale: NodeId,
    bias: NodeId,
    mean: NodeId,
    variance: NodeId,
    channel_shape: Shape,
    variance_is_vector: bool,
    centered_dtype: DType,
    scaled_dtype: DType,
    variance_dtype: DType,
    normalized_dtype: DType,
    output_dtype: DType,
    output_shape: Shape,
    epsilon: TensorData,
}

fn batch_norm_plan(
    g: &Graph,
    inputs: [NodeId; 5],
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<BatchNormPlan> {
    if ins.len() != 5
        || attrs.keys().any(|name| {
            !matches!(
                name.as_str(),
                "epsilon" | "momentum" | "training_mode" | "spatial" | "is_test"
            )
        })
    {
        return Err(bad("unsupported BatchNormalization inputs or attributes"));
    }
    let epsilon = typed_scalar_f32_attr(n, "epsilon")?.unwrap_or(1e-5);
    // These source attributes are live only on the training branch.  Decode
    // them strictly even for inference, so aliases never become accepted.
    let _ = typed_scalar_f32_attr(n, "momentum")?;
    let training = strict_typed_scalar_i64_attr(n, "training_mode")?.unwrap_or(0);
    let _ = strict_typed_scalar_i64_attr(n, "spatial")?;
    let _ = strict_typed_scalar_i64_attr(n, "is_test")?;
    if training != 0 {
        return Err(bad("BatchNormalization training mode is unsupported"));
    }

    let input_shape = g.shape(inputs[0])?.clone();
    if input_shape.rank() < 2 {
        return Err(bad("BatchNormalization X must have rank at least two"));
    }
    let channels = input_shape.dims()[1];
    let channel_shape = Shape::new(
        input_shape
            .dims()
            .iter()
            .enumerate()
            .map(|(axis, &extent)| if axis == 1 { extent } else { 1 })
            .collect::<Vec<_>>(),
    );
    let input_dtype = g.dtype(inputs[0])?;
    let mut dtypes = [input_dtype; 5];
    let mut shapes = Vec::with_capacity(5);
    for (index, input) in inputs.into_iter().enumerate() {
        let shape = g.shape(input)?.clone();
        let dtype = g.dtype(input)?;
        shape.numel()?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("BatchNormalization input byte extent overflow"))?;
        dtypes[index] = dtype;
        shapes.push(shape);
    }
    for shape in [&shapes[1], &shapes[2], &shapes[3]] {
        if shape.numel()? != channels {
            return Err(bad(
                "BatchNormalization scale, bias, and mean must contain C values",
            ));
        }
    }
    let variance_shape = shapes[4].clone();
    let variance_is_vector = variance_shape.rank() == 1;
    let epsilon_dtype = if dtypes[4].is_float() {
        dtypes[4]
    } else {
        DType::F32
    };
    let variance_dtype = batch_norm_dtype(dtypes[4], epsilon_dtype);
    let invstd_dtype = variance_dtype;
    let centered_dtype = batch_norm_dtype(dtypes[0], dtypes[3]);
    let scaled_dtype = batch_norm_dtype(centered_dtype, dtypes[1]);
    let invstd_shape = if variance_is_vector {
        channel_shape.clone()
    } else {
        variance_shape.clone()
    };
    let normalized_shape = input_shape.broadcast_with(&invstd_shape)?;
    let normalized_dtype = batch_norm_dtype(scaled_dtype, invstd_dtype);
    let output_shape = normalized_shape.broadcast_with(&channel_shape)?;
    let output_dtype = batch_norm_dtype(normalized_dtype, dtypes[2]);
    for (shape, dtype, what) in [
        (&channel_shape, dtypes[1], "channel reshape"),
        (&channel_shape, dtypes[2], "bias reshape"),
        (&channel_shape, dtypes[3], "mean reshape"),
        (&variance_shape, variance_dtype, "variance plus epsilon"),
        (&invstd_shape, invstd_dtype, "inverse standard deviation"),
        (&input_shape, centered_dtype, "centered input"),
        (&input_shape, scaled_dtype, "scaled input"),
        (&normalized_shape, normalized_dtype, "normalized output"),
        (&output_shape, output_dtype, "output"),
    ] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("BatchNormalization {what} byte extent overflow")))?;
    }
    Ok(BatchNormPlan {
        input: inputs[0],
        scale: inputs[1],
        bias: inputs[2],
        mean: inputs[3],
        variance: inputs[4],
        channel_shape,
        variance_is_vector,
        centered_dtype,
        scaled_dtype,
        variance_dtype,
        normalized_dtype,
        output_dtype,
        output_shape,
        epsilon: TensorData::scalar_with_dtype(Scalar::F(f64::from(epsilon)), epsilon_dtype),
    })
}

fn flatten_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<FlattenPlan> {
    if ins.len() != 1 || attrs.keys().any(|name| name != "axis") {
        return Err(bad("Flatten requires one input and only axis"));
    }
    let source_shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    let total = source_shape.numel()?;
    total
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Flatten input byte extent overflow"))?;
    let rank = i64::try_from(source_shape.rank()).map_err(|_| bad("Flatten rank overflow"))?;
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(1);
    // Equivalent to Python's `shape[0:raw_axis]` stop normalization.
    let axis = if raw_axis < 0 {
        raw_axis.saturating_add(rank).clamp(0, rank)
    } else {
        raw_axis.min(rank)
    };
    let axis = usize::try_from(axis).map_err(|_| bad("Flatten axis overflow"))?;
    let leading = source_shape.dims()[..axis]
        .iter()
        .try_fold(1usize, |product, extent| product.checked_mul(*extent))
        .ok_or_else(|| bad("Flatten leading extent overflow"))?;
    // tinygrad delegates `-1` to Tensor.reshape. If the explicit leading
    // product is zero, that inference divides by zero and must fail before a
    // view node is appended.
    if leading == 0 || total % leading != 0 {
        return Err(bad("invalid Flatten inferred dimension"));
    }
    let output_shape = Shape::new([leading, total / leading]);
    output_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Flatten output byte extent overflow"))?;
    Ok(FlattenPlan {
        identity: output_shape == source_shape,
        output_shape,
        dtype,
    })
}

fn reshape_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReshapePlan> {
    if ins.len() != 2 || attrs.keys().any(|name| name != "allowzero") {
        return Err(bad("Reshape requires two inputs and only allowzero"));
    }
    let allowzero = strict_typed_scalar_i64_attr(n, "allowzero")?.unwrap_or(0) != 0;
    let source_shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    let source_numel = source_shape.numel()?;
    source_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Reshape input byte extent overflow"))?;
    let requested = static_slice_control(constants, ins[1], "shape")?;
    let mut output = Vec::with_capacity(requested.len());
    let mut inferred = None;
    let mut known = 1usize;
    for (axis, requested) in requested.into_iter().enumerate() {
        let extent = match requested {
            0 if !allowzero => *source_shape
                .dims()
                .get(axis)
                .ok_or_else(|| bad("Reshape zero axis out of range"))?,
            -1 => {
                if inferred.replace(axis).is_some() {
                    return Err(bad("multiple Reshape -1 dimensions"));
                }
                output.push(1);
                continue;
            }
            value => usize::try_from(value).map_err(|_| bad("negative Reshape dimension"))?,
        };
        known = known
            .checked_mul(extent)
            .ok_or_else(|| bad("Reshape extent overflow"))?;
        output.push(extent);
    }
    if let Some(axis) = inferred {
        // Tensor.reshape infers with integer division by the product that
        // still contains -1. A literal zero therefore raises rather than
        // inventing an ambiguous zero-sized inferred dimension.
        if known == 0 || source_numel % known != 0 {
            return Err(bad("invalid Reshape inferred dimension"));
        }
        output[axis] = source_numel / known;
    } else if source_numel != known {
        return Err(bad("Reshape element count mismatch"));
    }
    let output_shape = Shape::new(output);
    let output_numel = output_shape.numel()?;
    output_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Reshape output byte extent overflow"))?;
    Ok(ReshapePlan {
        identity: output_shape == source_shape,
        output_shape,
        dtype,
    })
}

/// Every fallible fact in tinygrad's static ONNX Slice adapter.  In
/// particular, Python's `slices[axis] = ...` overwrites repeats in source
/// order, so only the final slice on each axis is normalized.
struct SlicePlan {
    slices: Vec<Slice>,
    output_shape: Shape,
    dtype: DType,
}

fn slice_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<SlicePlan> {
    if !(3..=5).contains(&ins.len()) || !attrs.is_empty() {
        return Err(bad("Slice requires three to five inputs and no attributes"));
    }
    let shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    shape.numel()?;
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Slice input byte extent overflow"))?;
    let starts = static_slice_control(constants, ins[1], "starts")?;
    let ends = static_slice_control(constants, ins[2], "ends")?;
    let rank = shape.rank();
    let axes = if ins.len() >= 4 && !ins[3].is_empty() {
        let axes = static_slice_control(constants, ins[3], "axes")?;
        if axes.is_empty() {
            (0..rank).map(|axis| axis as i64).collect()
        } else {
            axes
        }
    } else {
        (0..rank).map(|axis| axis as i64).collect()
    };
    let steps = if ins.len() == 5 && !ins[4].is_empty() {
        let steps = static_slice_control(constants, ins[4], "steps")?;
        if steps.is_empty() {
            vec![1; rank]
        } else {
            steps
        }
    } else {
        vec![1; rank]
    };
    if starts.len() < axes.len() || ends.len() < axes.len() || steps.len() < axes.len() {
        return Err(bad("Slice controls are shorter than axes"));
    }
    let mut slices = vec![
        Slice {
            start: None,
            stop: None,
            step: 1,
        };
        rank
    ];
    for index in 0..axes.len() {
        // Python assignment validates each axis immediately, even if a later
        // duplicate replaces its range.
        let axis = axes_usize(&[axes[index]], rank)?[0];
        slices[axis] = Slice {
            start: Some(isize::try_from(starts[index]).map_err(|_| bad("Slice start overflow"))?),
            stop: Some(isize::try_from(ends[index]).map_err(|_| bad("Slice end overflow"))?),
            step: isize::try_from(steps[index]).map_err(|_| bad("Slice step overflow"))?,
        };
    }
    let output_shape = Shape::new(
        slices
            .iter()
            .zip(shape.dims())
            .enumerate()
            .map(|(axis, (slice, dim))| crate::ir::normalized_slice(*dim, *slice, axis))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|(_, _, _, length)| length)
            .collect::<Vec<_>>(),
    );
    output_shape.numel()?;
    output_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Slice output byte extent overflow"))?;
    Ok(SlicePlan {
        slices,
        output_shape,
        dtype,
    })
}

/// All static facts required by the coupled ONNX TopK value/index result.
///
/// The graph's stable-sort selectors are already an atomic producer pair, but
/// ONNX adds a static K input and publishes both selectors.  Keep every
/// fallible source descriptor here so neither selector nor either value-map
/// binding can be exposed following malformed input.
struct TopKPlan {
    k: usize,
    axis: isize,
    largest: bool,
}

/// Fully resolved source sections for ONNX Split.  Both explicit sections and
/// the omitted-input `num_outputs` form become the same checked ordered list
/// before Graph::split creates its first shrink view.
struct SplitPlan {
    sections: Vec<usize>,
    axis: isize,
}

fn static_split_sections(
    constants: &BTreeMap<String, TensorData>,
    name: &str,
) -> Result<Vec<usize>> {
    let value = constants
        .get(name)
        .ok_or_else(|| bad("Split sections must be a constant initializer"))?;
    if !matches!(value.dtype(), DType::I32 | DType::I64) {
        return Err(bad("Split sections must be I32 or I64"));
    }
    let shape = value.shape();
    let numel = shape.numel()?;
    numel
        .checked_mul(value.dtype().itemsize())
        .ok_or_else(|| bad("Split sections byte extent overflow"))?;
    if shape.rank() != 1 {
        return Err(bad("Split sections must be rank one"));
    }
    (0..value.len())
        .map(|index| {
            usize::try_from(value.scalar_at(index).as_i64())
                .map_err(|_| bad("Split sections must be nonnegative"))
        })
        .collect()
}

fn split_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    outs: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<SplitPlan> {
    if !(ins.len() == 1 || ins.len() == 2) {
        return Err(bad("Split requires data and optional sections input"));
    }
    if attrs
        .keys()
        .any(|key| !matches!(key.as_str(), "axis" | "num_outputs"))
    {
        return Err(bad("unsupported Split attribute"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    let input_numel = shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("Split input byte extent overflow"))?;
    let rank = i64::try_from(shape.rank()).map_err(|_| bad("Split rank overflow"))?;
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(0);
    let raw_num_outputs = strict_typed_scalar_i64_attr(n, "num_outputs")?;
    let axis = if raw_axis < 0 {
        raw_axis
            .checked_add(rank)
            .ok_or_else(|| bad("invalid Split axis"))?
    } else {
        raw_axis
    };
    if axis < 0 || axis >= rank {
        return Err(bad("invalid Split axis"));
    }
    let axis = usize::try_from(axis).map_err(|_| bad("invalid Split axis"))?;
    let axis_len = shape.dims()[axis];

    let explicit = ins.get(1).filter(|name| !name.is_empty());
    let sections = if let Some(name) = explicit {
        let sections = static_split_sections(constants, name)?;
        if sections.len() != outs.len() {
            return Err(bad("Split section count must match outputs"));
        }
        sections
    } else {
        // tinygrad's runner supplies the node output count when this
        // attribute is absent. An explicit attribute wins, exactly as the
        // source call-site's `if 'num_outputs' not in opts` does.
        let default_count =
            i64::try_from(outs.len()).map_err(|_| bad("Split output count overflow"))?;
        let count = raw_num_outputs.unwrap_or(default_count);
        let count =
            usize::try_from(count).map_err(|_| bad("Split num_outputs must be positive"))?;
        if count == 0 {
            return Err(bad("Split num_outputs must be positive"));
        }
        if count != outs.len() {
            // Source reaches strict tuple/output zip after materializing its
            // views. Keep this mismatch fail-closed before graph mutation.
            return Err(bad("Split num_outputs must match outputs"));
        }
        let base = axis_len / count;
        let remainder = axis_len % count;
        (0..count)
            .map(|index| base + usize::from(index < remainder))
            .collect()
    };
    let total = sections.iter().try_fold(0usize, |total, &section| {
        total
            .checked_add(section)
            .ok_or_else(|| bad("Split section extent overflow"))
    })?;
    if total != axis_len {
        return Err(bad("Split sections must cover the selected axis"));
    }

    // Every future Shrink output descriptor (including zero-sized sections)
    // is checked before `Graph::split` appends a single view.
    for &section in &sections {
        let mut output_dims = shape.dims().to_vec();
        output_dims[axis] = section;
        let output_numel = Shape::new(output_dims).numel()?;
        output_numel
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("Split output byte extent overflow"))?;
    }
    Ok(SplitPlan {
        sections,
        axis: isize::try_from(axis).map_err(|_| bad("Split axis overflow"))?,
    })
}

fn topk_plan(
    g: &Graph,
    input: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<TopKPlan> {
    if ins.len() != 2 {
        return Err(bad("TopK requires data and K inputs"));
    }
    if attrs
        .keys()
        .any(|key| !matches!(key.as_str(), "axis" | "largest" | "sorted"))
    {
        return Err(bad("unsupported TopK attribute"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    let input_numel = shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("TopK input byte extent overflow"))?;
    input_numel
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| bad("TopK stable indices byte extent overflow"))?;
    if shape.rank() == 0 {
        return Err(bad("TopK requires an input rank of at least one"));
    }

    let raw_k = static_i32_i64_scalar(constants, ins[1], "TopK K")?;
    let k = usize::try_from(raw_k).map_err(|_| bad("TopK K must be nonnegative"))?;
    let rank = i64::try_from(shape.rank()).map_err(|_| bad("TopK rank overflow"))?;
    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(-1);
    let axis = if raw_axis < 0 {
        raw_axis
            .checked_add(rank)
            .ok_or_else(|| bad("invalid TopK axis"))?
    } else {
        raw_axis
    };
    if axis < 0 || axis >= rank {
        return Err(bad("invalid TopK axis"));
    }
    let axis = usize::try_from(axis).map_err(|_| bad("invalid TopK axis"))?;
    let axis_extent = shape.dims()[axis];
    if axis_extent > i32::MAX as usize {
        return Err(bad("TopK axis extent exceeds stable I32 index range"));
    }
    if k > axis_extent {
        return Err(bad("TopK K exceeds selected axis extent"));
    }
    let largest = strict_typed_scalar_i64_attr(n, "largest")?.unwrap_or(1) != 0;
    let sorted = strict_typed_scalar_i64_attr(n, "sorted")?.unwrap_or(1) != 0;
    if !sorted {
        // This is the source adapter's own Tensor.topk gate, not a narrowed
        // importer policy: checked-in tinygrad raises for sorted_=False.
        return Err(bad("TopK sorted=false is unsupported by source"));
    }

    let mut output_dims = shape.dims().to_vec();
    output_dims[axis] = k;
    let output_shape = Shape::new(output_dims);
    let output_numel = output_shape.numel()?;
    output_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("TopK values byte extent overflow"))?;
    // Sort's temporary index selector is I32; ONNX's adapter then casts it
    // to I64 for the published indices result.
    output_numel
        .checked_mul(DType::I32.itemsize())
        .ok_or_else(|| bad("TopK temporary indices byte extent overflow"))?;
    output_numel
        .checked_mul(DType::I64.itemsize())
        .ok_or_else(|| bad("TopK indices byte extent overflow"))?;
    Ok(TopKPlan {
        k,
        axis: isize::try_from(axis).map_err(|_| bad("TopK axis overflow"))?,
        largest,
    })
}

fn cumsum_zero(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(0),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U(0),
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(0.0),
    }
}

fn cumsum_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<CumSumPlan> {
    if attrs
        .keys()
        .any(|key| key != "exclusive" && key != "reverse")
    {
        return Err(bad("unsupported CumSum attribute"));
    }
    let exclusive = typed_scalar_i64_attr(n, "exclusive")?.unwrap_or(0) != 0;
    let reverse = typed_scalar_i64_attr(n, "reverse")?.unwrap_or(0) != 0;
    let shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    shape.numel()?;
    let raw_axis = static_i32_i64_scalar(constants, ins[1], "CumSum axis")?;
    let rank = shape.rank();
    let axis = if rank == 0 {
        if !matches!(raw_axis, -1 | 0) {
            return Err(bad("invalid CumSum scalar axis"));
        }
        if exclusive || reverse {
            // tinygrad reaches flip/pad before its scalar cumsum fast path,
            // so these requests fail instead of constructing a scalar result.
            return Err(bad("CumSum scalar does not support exclusive or reverse"));
        }
        raw_axis as isize
    } else {
        let rank_i64 = i64::try_from(rank).map_err(|_| bad("CumSum rank overflow"))?;
        if raw_axis < -rank_i64 || raw_axis >= rank_i64 {
            return Err(bad("invalid CumSum axis"));
        }
        let normalized = if raw_axis < 0 {
            raw_axis
                .checked_add(rank_i64)
                .ok_or_else(|| bad("invalid CumSum axis"))?
        } else {
            raw_axis
        };
        isize::try_from(normalized).map_err(|_| bad("CumSum axis overflow"))?
    };

    // Graph::cumsum is a prefix-Sum composition.  Establish its exact typed
    // output and every prefix/reduction bound before invoking it.
    let sum_dtypes = ReductionDType::sum_default(dtype);
    if rank == 0 || shape.dims().contains(&0) {
        shape.numel()?;
    } else {
        let axis = axis as usize;
        let mut concat_extent = 0usize;
        for end in 0..shape.dims()[axis] {
            let prefix = Shape::new(
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(
                        |(dimension, &extent)| {
                            if dimension == axis {
                                end + 1
                            } else {
                                extent
                            }
                        },
                    )
                    .collect::<Vec<_>>(),
            );
            prefix.numel()?;
            let reduced = Shape::new(
                prefix
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &extent)| if dimension == axis { 1 } else { extent })
                    .collect::<Vec<_>>(),
            );
            reduced.numel()?;
            concat_extent = concat_extent
                .checked_add(reduced.dims()[axis])
                .ok_or_else(|| bad("CumSum concat extent overflow"))?;
        }
        if concat_extent != shape.dims()[axis] {
            return Err(bad("CumSum prefix extent mismatch"));
        }
        shape.numel()?;
    }
    let _output_dtype = sum_dtypes.output;

    let (padding, shrink) = if exclusive {
        let axis = axis as usize;
        let mut padded_dims = shape.dims().to_vec();
        padded_dims[axis] = padded_dims[axis]
            .checked_add(1)
            .ok_or_else(|| bad("CumSum exclusive pad extent overflow"))?;
        let padded = Shape::new(padded_dims);
        padded.numel()?;
        let shrink = shape
            .dims()
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        Shape::new(shrink.iter().map(|(start, end)| end - start).collect()).numel()?;
        let padding = (0..rank)
            .map(|dimension| if dimension == axis { (1, 0) } else { (0, 0) })
            .collect();
        (Some(padding), Some(shrink))
    } else {
        (None, None)
    };
    Ok(CumSumPlan {
        axis,
        reverse,
        exclusive,
        padding,
        shrink,
        fill: cumsum_zero(dtype),
    })
}

/// Closed lowerings for tinygrad's `Trilu`: the normal interior mask, or one
/// of its source-observable saturated forms.  Saturation is needed because
/// tinygrad's Python integer diagonal has no I64 overflow boundary.
enum TriluLowering {
    Identity,
    Zero(TensorData),
    Upper(i64),
    Lower(i64),
}

fn trilu_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<TriluLowering> {
    if attrs.keys().any(|key| key != "upper") {
        return Err(bad("unsupported Trilu attribute"));
    }
    let upper = typed_scalar_i64_attr(n, "upper")?.unwrap_or(1) != 0;
    let diagonal = match ins.get(1).filter(|name| !name.is_empty()) {
        Some(name) => static_i32_i64_scalar(constants, name, "Trilu k")?,
        None => 0,
    };
    let shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    let output_numel = shape.numel()?;
    if shape.rank() < 2 {
        return Err(bad("Trilu input must have rank at least two"));
    }
    let rows = shape.dims()[shape.rank() - 2];
    let columns = shape.dims()[shape.rank() - 1];
    let rows_i64 = i64::try_from(rows).map_err(|_| bad("Trilu row extent exceeds I64"))?;
    let columns_i64 = i64::try_from(columns).map_err(|_| bad("Trilu column extent exceeds I64"))?;
    let mask_shape = Shape::new([rows, columns]);
    mask_shape.numel()?;
    if mask_shape.broadcast_with(&shape).as_ref() != Ok(&shape) {
        return Err(bad("Trilu mask cannot broadcast to input"));
    }
    // The source has no populated output for an empty tensor.  Returning the
    // same NodeId is observationally exact and avoids constructing aranges
    // whose unused matrix extent could be enormous.
    if output_numel == 0 {
        return Ok(TriluLowering::Identity);
    }

    let zero = || TensorData::zeros_with_dtype(shape.clone(), dtype);
    if upper {
        // `row + k <= column`: all rows retain for k <= 1 - rows; all mask
        // false for k >= columns. Both comparisons are safe with rows > 0.
        let all_input = 1i64
            .checked_sub(rows_i64)
            .ok_or_else(|| bad("Trilu upper diagonal threshold overflow"))?;
        if diagonal <= all_input {
            return Ok(TriluLowering::Identity);
        }
        if diagonal >= columns_i64 {
            return Ok(TriluLowering::Zero(zero()?));
        }
        (rows_i64 - 1)
            .checked_add(diagonal)
            .ok_or_else(|| bad("Trilu upper diagonal overflow"))?;
        Ok(TriluLowering::Upper(diagonal))
    } else {
        // `row + (k + 1) <= column` is the zero branch of tril.  Thus k <=
        // -rows masks everything, while k >= columns - 1 retains everything.
        let all_zero = rows_i64
            .checked_neg()
            .ok_or_else(|| bad("Trilu lower diagonal threshold overflow"))?;
        let all_input = columns_i64
            .checked_sub(1)
            .ok_or_else(|| bad("Trilu lower diagonal threshold overflow"))?;
        if diagonal <= all_zero {
            return Ok(TriluLowering::Zero(zero()?));
        }
        if diagonal >= all_input {
            return Ok(TriluLowering::Identity);
        }
        let shift = diagonal
            .checked_add(1)
            .ok_or_else(|| bad("Trilu lower diagonal overflow"))?;
        (rows_i64 - 1)
            .checked_add(shift)
            .ok_or_else(|| bad("Trilu lower diagonal overflow"))?;
        Ok(TriluLowering::Lower(diagonal))
    }
}

/// Read-only ONNX opset-13 reduction planning shared by the supported
/// reductions and their source-level compositions. Every shape, axis, and
/// accumulator fact is established before a caller appends its first node.
struct ReducePlan {
    axes: Vec<isize>,
    keepdims: bool,
    noop: bool,
    output_shape: Shape,
    sum_dtypes: ReductionDType,
}

fn reduce_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReducePlan> {
    let input_shape = g.shape(x)?.clone();
    let input_dtype = g.dtype(x)?;
    input_shape.numel()?;
    let keepdims = attrs
        .get("keepdims")
        .map(|x| scalar_i64(x))
        .transpose()?
        .unwrap_or(1);
    let noop = attrs
        .get("noop_with_empty_axes")
        .map(|x| scalar_i64(x))
        .transpose()?
        .unwrap_or(0);
    if !matches!(keepdims, 0 | 1) || !matches!(noop, 0 | 1) {
        return Err(bad("Reduce boolean attributes must be 0 or 1"));
    }
    let axes = if ins.len() == 2 && !ins[1].is_empty() {
        let axes = const_i64(constants, ins[1])?;
        if constants
            .get(ins[1])
            .expect("constant axes checked by const_i64")
            .shape()
            .rank()
            != 1
        {
            return Err(bad("Reduce axes constant must be rank-1"));
        }
        axes
    } else {
        Vec::new()
    };
    let noop = axes.is_empty() && noop == 1;
    let axes = if noop {
        Vec::new()
    } else if axes.is_empty() {
        (0..input_shape.rank()).map(|axis| axis as isize).collect()
    } else {
        let axes = axes_usize(&axes, input_shape.rank())?;
        if axes
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != axes.len()
        {
            return Err(bad("duplicate Reduce axis"));
        }
        axes.into_iter().map(|axis| axis as isize).collect()
    };
    let output_shape = if noop {
        input_shape.clone()
    } else {
        let normalized = axes.iter().map(|&axis| axis as usize).collect::<Vec<_>>();
        reduction_shape(&input_shape, &normalized, keepdims == 1)
    };
    // The source square is shape/dtype preserving, so its extent and the
    // ensuing Sum output are both fully known before it is constructed.
    input_shape.numel()?;
    output_shape.numel()?;
    Ok(ReducePlan {
        axes,
        keepdims: keepdims == 1,
        noop,
        output_shape,
        sum_dtypes: ReductionDType::sum_default(input_dtype),
    })
}

/// Full source-level contract for `ReduceMean`. Tensor.mean first casts to
/// its Sum accumulator, reduces, performs true division as reciprocal/mul,
/// then narrows only at its declared result dtype. That differs from
/// `ReduceKind::Mean`, especially for F16/BF16.
struct ReduceMeanPlan {
    reduction: ReducePlan,
    sum_dtypes: ReductionDType,
    division_dtype: DType,
    output_dtype: DType,
    divisor: TensorData,
}

fn reduce_mean_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceMeanPlan> {
    let source_shape = g.shape(x)?.clone();
    let source_dtype = g.dtype(x)?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    let count = reduction.axes.iter().try_fold(1usize, |count, axis| {
        count
            .checked_mul(source_shape.dims()[*axis as usize])
            .ok_or_else(|| bad("ReduceMean reduction extent overflow"))
    })?;
    // Tensor.mean explicitly casts before calling sum, so its narrow-float
    // Sum remains F32 instead of Tensor.sum's usual post-Sum narrowing.
    let default_sum = ReductionDType::sum_default(source_dtype);
    let sum_dtypes = ReductionDType::new(default_sum.accumulator, default_sum.accumulator);
    let division_dtype = if sum_dtypes.output.is_float() {
        sum_dtypes.output
    } else {
        DType::F32
    };
    let output_dtype = if source_dtype.is_float() {
        source_dtype
    } else {
        DType::F32
    };
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("ReduceMean {what} byte extent overflow")))
    };
    extent(&source_shape, source_dtype, "input")?;
    extent(&source_shape, sum_dtypes.accumulator, "accumulator cast")?;
    extent(
        &reduction.output_shape,
        sum_dtypes.accumulator,
        "Sum output",
    )?;
    extent(&reduction.output_shape, division_dtype, "true division")?;
    extent(&reduction.output_shape, output_dtype, "output")?;
    let divisor = TensorData::scalar_with_dtype(Scalar::F(count as f64), division_dtype);
    if divisor.dtype() != division_dtype
        || reduction.output_shape.broadcast_with(divisor.shape())? != reduction.output_shape
        || division_dtype.promote(divisor.dtype()) != division_dtype
    {
        return Err(bad("ReduceMean divisor promotion mismatch"));
    }
    Ok(ReduceMeanPlan {
        reduction,
        sum_dtypes,
        division_dtype,
        output_dtype,
        divisor,
    })
}

/// Full source-level contract for `ReduceSum`. Tensor.sum accumulates at
/// `sum_acc_dtype`, narrowing only after the reduction for F16/BF16, and it
/// retains that accumulator result even when an empty axes list is a shape
/// no-op for integral inputs.
struct ReduceSumPlan {
    reduction: ReducePlan,
}

fn reduce_sum_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceSumPlan> {
    let source_shape = g.shape(x)?.clone();
    let source_dtype = g.dtype(x)?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("ReduceSum {what} byte extent overflow")))
    };
    extent(&source_shape, source_dtype, "input")?;
    extent(
        &source_shape,
        reduction.sum_dtypes.accumulator,
        "accumulator cast",
    )?;
    extent(
        &reduction.output_shape,
        reduction.sum_dtypes.accumulator,
        "accumulator reduction",
    )?;
    extent(
        &reduction.output_shape,
        reduction.sum_dtypes.output,
        "output",
    )?;
    Ok(ReduceSumPlan { reduction })
}

/// Full source-level descriptor for `ReduceProd`. Tensor.prod explicitly
/// casts to its own dtype, so Product's accumulator and result are identical;
/// the plan exists to establish every byte extent before construction.
struct ReduceProdPlan {
    reduction: ReducePlan,
    dtypes: ReductionDType,
}

fn reduce_prod_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceProdPlan> {
    let source_shape = g.shape(x)?.clone();
    let source_dtype = g.dtype(x)?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    let dtypes = ReductionDType::product_default(source_dtype);
    if dtypes.accumulator != source_dtype || dtypes.output != source_dtype {
        return Err(bad("ReduceProd dtype contract mismatch"));
    }
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("ReduceProd {what} byte extent overflow")))
    };
    extent(&source_shape, source_dtype, "input")?;
    extent(
        &reduction.output_shape,
        dtypes.accumulator,
        "accumulator reduction",
    )?;
    extent(&reduction.output_shape, dtypes.output, "output")?;
    Ok(ReduceProdPlan { reduction, dtypes })
}

/// Fully resolved ONNX `ReduceMin` lowering. Graph::reduce intentionally
/// rejects populated outputs with an empty extrema domain, but tinygrad's
/// Tensor.min is inverse(max(inverse(x))) and supplies dtype.max there.
enum ReduceMinLowering {
    Identity,
    Empty,
    IdentityValue,
    Reduce,
}

struct ReduceMinPlan {
    reduction: ReducePlan,
    dtype: DType,
    lowering: ReduceMinLowering,
}

fn reduce_min_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceMinPlan> {
    let source_shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    let extent = |shape: &Shape, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("ReduceMin {what} byte extent overflow")))
    };
    extent(&source_shape, "input")?;
    extent(&reduction.output_shape, "output")?;
    let lowering = if reduction.noop {
        ReduceMinLowering::Identity
    } else {
        let output_numel = reduction.output_shape.numel()?;
        let empty_domain = reduction
            .axes
            .iter()
            .any(|&axis| source_shape.dims()[axis as usize] == 0);
        if output_numel == 0 {
            // An unreduced zero extent leaves no output values to populate.
            ReduceMinLowering::Empty
        } else if empty_domain {
            // Tensor.min is inverse(max(inverse(x))), hence dtype.max.
            ReduceMinLowering::IdentityValue
        } else {
            ReduceMinLowering::Reduce
        }
    };
    Ok(ReduceMinPlan {
        reduction,
        dtype,
        lowering,
    })
}

/// Fully resolved ONNX `ReduceMax` lowering. Graph::reduce intentionally
/// rejects populated outputs with an empty extrema domain, but tinygrad's
/// Tensor.max supplies the dtype-min identity for precisely that source form.
enum ReduceMaxLowering {
    Identity,
    Empty,
    IdentityValue,
    Reduce,
}

struct ReduceMaxPlan {
    reduction: ReducePlan,
    dtype: DType,
    lowering: ReduceMaxLowering,
}

fn reduce_max_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceMaxPlan> {
    let source_shape = g.shape(x)?.clone();
    let dtype = g.dtype(x)?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    let extent = |shape: &Shape, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("ReduceMax {what} byte extent overflow")))
    };
    extent(&source_shape, "input")?;
    extent(&reduction.output_shape, "output")?;
    let lowering = if reduction.noop {
        ReduceMaxLowering::Identity
    } else {
        let output_numel = reduction.output_shape.numel()?;
        let empty_domain = reduction
            .axes
            .iter()
            .any(|&axis| source_shape.dims()[axis as usize] == 0);
        if output_numel == 0 {
            // An unreduced zero extent leaves no output values to populate.
            ReduceMaxLowering::Empty
        } else if empty_domain {
            // Tensor._rop MAX has the static dtype.min identity here.
            ReduceMaxLowering::IdentityValue
        } else {
            ReduceMaxLowering::Reduce
        }
    };
    Ok(ReduceMaxPlan {
        reduction,
        dtype,
        lowering,
    })
}

/// Fully resolved source contract for tinygrad's ONNX
/// `LogSoftmax(X, axis) = m - log(sum(exp(m)))`, where
/// `m = X - detach(max(X, axis, keepdim=True))`.  The source implements exp
/// and log through Exp2/Log2 with storage-width constants, so this plan keeps
/// each intermediate dtype and extent explicit before graph mutation.
struct LogSoftmaxPlan {
    source_dtype: DType,
    output_dtype: DType,
    axis: Option<isize>,
    exp_work_dtype: DType,
    exp_dtype: DType,
    sum_dtypes: ReductionDType,
    inv_ln2: TensorData,
    ln2: TensorData,
    empty: bool,
}

fn log_softmax_plan(
    g: &Graph,
    x: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<LogSoftmaxPlan> {
    if attrs.keys().any(|key| key != "axis") {
        return Err(bad("unsupported LogSoftmax attribute"));
    }
    let shape = g.shape(x)?.clone();
    let source_dtype = g.dtype(x)?;
    let numel = shape.numel()?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("LogSoftmax {what} byte extent overflow")))
    };
    extent(&shape, source_dtype, "input")?;

    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(-1);
    let axis = if shape.rank() == 0 {
        if !matches!(raw_axis, -1 | 0) {
            return Err(bad("invalid LogSoftmax scalar axis"));
        }
        None
    } else {
        let rank = i64::try_from(shape.rank()).map_err(|_| bad("LogSoftmax rank overflow"))?;
        let axis = if raw_axis < 0 {
            raw_axis
                .checked_add(rank)
                .ok_or_else(|| bad("invalid LogSoftmax axis"))?
        } else {
            raw_axis
        };
        if axis < 0 || axis >= rank {
            return Err(bad("invalid LogSoftmax axis"));
        }
        Some(isize::try_from(axis).map_err(|_| bad("invalid LogSoftmax axis"))?)
    };
    let max_shape = match axis {
        None => shape.clone(),
        Some(axis) => Shape::new(
            shape
                .dims()
                .iter()
                .enumerate()
                .map(|(index, &dimension)| {
                    (index == axis as usize).then_some(1).unwrap_or(dimension)
                })
                .collect::<Vec<_>>(),
        ),
    };
    extent(&max_shape, source_dtype, "Max")?;
    if shape.broadcast_with(&max_shape)? != shape {
        return Err(bad("LogSoftmax Max cannot broadcast to input"));
    }
    extent(&shape, source_dtype, "centered")?;

    // Tensor.exp first promotes narrow and exact storage to F32, computes
    // `exp2(x * (1 / ln(2)))`, then restores the original floating storage.
    let exp_work_dtype = if source_dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let exp_dtype = if source_dtype.is_float() {
        source_dtype
    } else {
        DType::F32
    };
    let inv_ln2 =
        TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LOG2_E), exp_work_dtype);
    if shape.broadcast_with(inv_ln2.shape())? != shape
        || exp_work_dtype.promote(inv_ln2.dtype()) != exp_work_dtype
    {
        return Err(bad("LogSoftmax Exp2 scalar promotion mismatch"));
    }
    extent(&shape, exp_work_dtype, "Exp2 work")?;
    extent(&shape, exp_dtype, "Exp")?;

    let sum_dtypes = ReductionDType::sum_default(exp_dtype);
    extent(&max_shape, sum_dtypes.accumulator, "Sum accumulator")?;
    extent(&max_shape, sum_dtypes.output, "Sum output")?;
    let log_dtype = sum_dtypes.output;
    let ln2 = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LN_2), log_dtype);
    if max_shape.broadcast_with(ln2.shape())? != max_shape
        || log_dtype.promote(ln2.dtype()) != log_dtype
    {
        return Err(bad("LogSoftmax Log2 scalar promotion mismatch"));
    }
    extent(&max_shape, log_dtype, "Log")?;
    let output_dtype = source_dtype.promote(log_dtype);
    if shape.broadcast_with(&max_shape)? != shape {
        return Err(bad("LogSoftmax final subtraction cannot broadcast"));
    }
    extent(&shape, output_dtype, "output")?;

    Ok(LogSoftmaxPlan {
        source_dtype,
        output_dtype,
        axis,
        exp_work_dtype,
        exp_dtype,
        sum_dtypes,
        inv_ln2,
        ln2,
        empty: numel == 0,
    })
}

/// Fully resolved source contract for tinygrad's
/// `Softmax(X, axis) = exp(m) * reciprocal(sum(exp(m)))`, where
/// `m = X - detach(max(X, axis, keepdim=True))`.  This remains separate from
/// the LogSoftmax plan because the source's final reciprocal/multiply storage
/// rounding is observably different from a logarithm composition.
struct SoftmaxPlan {
    source_dtype: DType,
    output_dtype: DType,
    axis: Option<isize>,
    exp_work_dtype: DType,
    exp_dtype: DType,
    sum_dtypes: ReductionDType,
    inv_ln2: TensorData,
    empty: bool,
}

fn softmax_plan(
    g: &Graph,
    x: NodeId,
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<SoftmaxPlan> {
    if attrs.keys().any(|key| key != "axis") {
        return Err(bad("unsupported Softmax attribute"));
    }
    let shape = g.shape(x)?.clone();
    let source_dtype = g.dtype(x)?;
    let numel = shape.numel()?;
    let extent = |shape: &Shape, dtype: DType, what: &str| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad(format!("Softmax {what} byte extent overflow")))
    };
    extent(&shape, source_dtype, "input")?;

    let raw_axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(-1);
    let axis = if shape.rank() == 0 {
        if !matches!(raw_axis, -1 | 0) {
            return Err(bad("invalid Softmax scalar axis"));
        }
        None
    } else {
        let rank = i64::try_from(shape.rank()).map_err(|_| bad("Softmax rank overflow"))?;
        let axis = if raw_axis < 0 {
            raw_axis
                .checked_add(rank)
                .ok_or_else(|| bad("invalid Softmax axis"))?
        } else {
            raw_axis
        };
        if axis < 0 || axis >= rank {
            return Err(bad("invalid Softmax axis"));
        }
        Some(isize::try_from(axis).map_err(|_| bad("invalid Softmax axis"))?)
    };
    let max_shape = match axis {
        None => shape.clone(),
        Some(axis) => Shape::new(
            shape
                .dims()
                .iter()
                .enumerate()
                .map(|(index, &dimension)| {
                    (index == axis as usize).then_some(1).unwrap_or(dimension)
                })
                .collect::<Vec<_>>(),
        ),
    };
    extent(&max_shape, source_dtype, "Max")?;
    if shape.broadcast_with(&max_shape)? != shape {
        return Err(bad("Softmax Max cannot broadcast to input"));
    }
    extent(&shape, source_dtype, "centered")?;

    let exp_work_dtype = if source_dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let exp_dtype = if source_dtype.is_float() {
        source_dtype
    } else {
        DType::F32
    };
    let inv_ln2 =
        TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LOG2_E), exp_work_dtype);
    if shape.broadcast_with(inv_ln2.shape())? != shape
        || exp_work_dtype.promote(inv_ln2.dtype()) != exp_work_dtype
    {
        return Err(bad("Softmax Exp2 scalar promotion mismatch"));
    }
    extent(&shape, exp_work_dtype, "Exp2 work")?;
    extent(&shape, exp_dtype, "Exp")?;
    let sum_dtypes = ReductionDType::sum_default(exp_dtype);
    extent(&max_shape, sum_dtypes.accumulator, "Sum accumulator")?;
    extent(&max_shape, sum_dtypes.output, "Sum output")?;
    extent(&max_shape, sum_dtypes.output, "Reciprocal")?;
    if shape.broadcast_with(&max_shape)? != shape
        || exp_dtype.promote(sum_dtypes.output) != exp_dtype
    {
        return Err(bad("Softmax reciprocal cannot broadcast to exponentials"));
    }
    extent(&shape, exp_dtype, "output")?;

    Ok(SoftmaxPlan {
        source_dtype,
        output_dtype: exp_dtype,
        axis,
        exp_work_dtype,
        exp_dtype,
        sum_dtypes,
        inv_ln2,
        empty: numel == 0,
    })
}

/// Read-only dtype and scalar planning for tinygrad's ReduceLogSum
/// composition: typed Sum, then `log2 * ln(2)` at the concrete log width.
struct ReduceLogSumPlan {
    reduction: ReducePlan,
    ln2: TensorData,
}

fn reduce_log_sum_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceLogSumPlan> {
    let source_dtype = g.dtype(x)?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    let sum_dtype = if reduction.noop {
        source_dtype
    } else {
        reduction.sum_dtypes.output
    };
    // tinygrad's LOG2 lifts integers to its default F32, while concrete
    // floating storage widths—including F16/BF16—remain unchanged.
    let log_dtype = if sum_dtype.is_float() {
        sum_dtype
    } else {
        DType::F32
    };
    // tinygrad commits its weak mathematical literal at this exact concrete
    // multiplication width before rendering.
    let ln2 = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LN_2), log_dtype);
    let mul_shape = reduction.output_shape.broadcast_with(ln2.shape())?;
    mul_shape.numel()?;
    if ln2.dtype() != log_dtype
        || log_dtype.promote(ln2.dtype()) != log_dtype
        || mul_shape != reduction.output_shape
    {
        return Err(bad("ReduceLogSum scalar promotion mismatch"));
    }
    Ok(ReduceLogSumPlan { reduction, ln2 })
}

/// Read-only planning for tinygrad ONNX ReduceLogSumExp: `exp`, typed Sum,
/// then Tensor.log's concrete `log2 * ln(2)` composition.  This deliberately
/// differs from the numerically stable Graph::logsumexp helper.
struct ReduceLogSumExpPlan {
    reduction: ReducePlan,
    exp_dtype: DType,
    sum_dtypes: ReductionDType,
    ln2: TensorData,
}

fn reduce_log_sum_exp_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceLogSumExpPlan> {
    let source_dtype = g.dtype(x)?;
    let source_shape = g.shape(x)?.clone();
    source_shape.numel()?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    // Tensor.exp retains every concrete floating storage width and lifts all
    // integer/Bool inputs to tinygrad's default F32 width.
    let exp_dtype = if source_dtype.is_float() {
        source_dtype
    } else {
        DType::F32
    };
    source_shape.numel()?;
    let sum_dtypes = ReductionDType::sum_default(exp_dtype);
    let sum_dtype = if reduction.noop {
        exp_dtype
    } else {
        sum_dtypes.output
    };
    // Sum follows exp, so this is already floating; keep the general unary
    // rule explicit to preflight the final Graph::log2 result.
    let log_dtype = if sum_dtype.is_float() {
        sum_dtype
    } else {
        DType::F32
    };
    let ln2 = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LN_2), log_dtype);
    let mul_shape = reduction.output_shape.broadcast_with(ln2.shape())?;
    mul_shape.numel()?;
    if ln2.dtype() != log_dtype
        || log_dtype.promote(ln2.dtype()) != log_dtype
        || mul_shape != reduction.output_shape
    {
        return Err(bad("ReduceLogSumExp scalar promotion mismatch"));
    }
    Ok(ReduceLogSumExpPlan {
        reduction,
        exp_dtype,
        sum_dtypes,
        ln2,
    })
}

/// Fully resolved source contract for tinygrad's attribute-free
/// `GlobalMaxPool(X) = X.max(range(2, X.ndim), keepdim=True)`.
struct GlobalMaxPoolPlan {
    axes: Vec<isize>,
    output_shape: Shape,
    dtype: DType,
    output_numel: usize,
    empty_spatial: bool,
    max_identity: Scalar,
}

fn max_identity(dtype: DType) -> Scalar {
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

fn min_identity(dtype: DType) -> Scalar {
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

fn global_max_pool_plan(g: &Graph, x: NodeId) -> Result<GlobalMaxPoolPlan> {
    let dtype = g.dtype(x)?;
    let shape = g.shape(x)?.clone();
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("GlobalMaxPool input byte extent overflow"))?;
    let axes = (2..shape.rank())
        .map(|axis| axis as isize)
        .collect::<Vec<_>>();
    let empty_spatial = axes.iter().any(|&axis| shape.dims()[axis as usize] == 0);
    let output_shape = if axes.is_empty() {
        shape
    } else {
        Shape::new(
            g.shape(x)?
                .dims()
                .iter()
                .enumerate()
                .map(|(axis, &extent)| if axis >= 2 { 1 } else { extent })
                .collect::<Vec<_>>(),
        )
    };
    let output_numel = output_shape.numel()?;
    output_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("GlobalMaxPool output byte extent overflow"))?;
    Ok(GlobalMaxPoolPlan {
        axes,
        output_shape,
        dtype,
        output_numel,
        empty_spatial,
        max_identity: max_identity(dtype),
    })
}

/// Read-only dtype planning for tinygrad's ReduceL2 composition. Narrow
/// floats are widened *before* the square and Sum, then narrowed only after
/// sqrt; all other dtypes retain their source work width until sqrt's normal
/// unary promotion.
struct ReduceL2Plan {
    reduction: ReducePlan,
    source_dtype: DType,
    work_dtype: DType,
    sum_dtypes: ReductionDType,
    sqrt_dtype: DType,
}

fn reduce_l2_plan(
    g: &Graph,
    x: NodeId,
    ins: &[&str],
    attrs: &BTreeMap<String, Vec<u8>>,
    constants: &BTreeMap<String, TensorData>,
) -> Result<ReduceL2Plan> {
    let source_dtype = g.dtype(x)?;
    let reduction = reduce_plan(g, x, ins, attrs, constants)?;
    let work_dtype = match source_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        _ => source_dtype,
    };
    // tinygrad's explicit narrow-float cast means this Sum is over F32, not
    // over a narrow value with a post-reduction narrowing contract.
    let sum_dtypes = ReductionDType::sum_default(work_dtype);
    let sqrt_input_dtype = if reduction.noop {
        work_dtype
    } else {
        sum_dtypes.output
    };
    let sqrt_dtype = if sqrt_input_dtype.is_float() {
        sqrt_input_dtype
    } else {
        DType::F32
    };
    Ok(ReduceL2Plan {
        reduction,
        source_dtype,
        work_dtype,
        sum_dtypes,
        sqrt_dtype,
    })
}

#[derive(Clone)]
enum QuantParameterPlan {
    Keep,
    Reshape(Shape),
    Repeat {
        repeats: isize,
        axis: isize,
        shape: Shape,
    },
}

struct DequantizeLinearPlan {
    x: NodeId,
    scale: NodeId,
    zero: Option<NodeId>,
    scale_plan: QuantParameterPlan,
    zero_plan: Option<QuantParameterPlan>,
    subtract_dtype: DType,
    multiply_dtype: DType,
    output_dtype: DType,
    shape: Shape,
}

fn dequantize_dtype(lhs: DType, rhs: DType) -> DType {
    // `x.int()` is I32. tinygrad's weak quantization lattice deliberately
    // chooses its default float width for the otherwise-unrepresentable U64
    // pairing, rather than RustGrad's generic F64 promotion.
    if matches!(
        (lhs, rhs),
        (DType::I32, DType::U64) | (DType::U64, DType::I32)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

fn quant_parameter_plan(
    g: &Graph,
    parameter: NodeId,
    data_shape: &Shape,
    axis: i64,
    block_size: i64,
) -> Result<QuantParameterPlan> {
    let shape = g.shape(parameter)?.clone();
    let numel = shape.numel()?;
    numel
        .checked_mul(g.dtype(parameter)?.itemsize())
        .ok_or_else(|| bad("DequantizeLinear parameter byte extent overflow"))?;
    if numel == 1 {
        return Ok(QuantParameterPlan::Keep);
    }
    let rank = data_shape.rank();
    let normalized = if axis < 0 {
        axis.checked_add(i64::try_from(rank).map_err(|_| bad("DequantizeLinear rank overflow"))?)
            .ok_or_else(|| bad("invalid DequantizeLinear axis"))?
    } else {
        axis
    };
    let axis = usize::try_from(normalized)
        .ok()
        .filter(|axis| *axis < rank)
        .ok_or_else(|| bad("invalid DequantizeLinear axis"))?;
    let prepared = if block_size == 0 {
        let target = Shape::new(
            (0..rank)
                .map(|dim| {
                    if dim == axis {
                        data_shape.dims()[dim]
                    } else {
                        1
                    }
                })
                .collect::<Vec<_>>(),
        );
        if numel != target.numel()? {
            return Err(bad(
                "DequantizeLinear per-axis parameter cardinality mismatch",
            ));
        }
        QuantParameterPlan::Reshape(target)
    } else {
        let repeats =
            isize::try_from(block_size).map_err(|_| bad("invalid DequantizeLinear block_size"))?;
        if repeats < 0 || axis >= shape.rank() {
            return Err(bad("invalid DequantizeLinear blocked parameter"));
        }
        let mut dims = shape.dims().to_vec();
        dims[axis] = dims[axis]
            .checked_mul(
                usize::try_from(repeats).map_err(|_| bad("invalid DequantizeLinear block_size"))?,
            )
            .ok_or_else(|| bad("DequantizeLinear blocked extent overflow"))?;
        QuantParameterPlan::Repeat {
            repeats,
            axis: axis as isize,
            shape: Shape::new(dims),
        }
    };
    let prepared_shape = match &prepared {
        QuantParameterPlan::Keep => shape,
        QuantParameterPlan::Reshape(s) => s.clone(),
        QuantParameterPlan::Repeat { shape, .. } => shape.clone(),
    };
    prepared_shape
        .numel()?
        .checked_mul(g.dtype(parameter)?.itemsize())
        .ok_or_else(|| bad("DequantizeLinear prepared parameter byte extent overflow"))?;
    if prepared_shape.broadcast_with(data_shape)? != data_shape.clone() {
        return Err(bad("DequantizeLinear parameter cannot broadcast to X"));
    }
    Ok(prepared)
}

fn dequantize_linear_plan(
    g: &Graph,
    inputs: &[NodeId],
    ins: &[&str],
    n: &Msg<'_>,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<DequantizeLinearPlan> {
    if !(2..=3).contains(&inputs.len())
        || ins.len() != inputs.len()
        || attrs.keys().any(|x| x != "axis" && x != "block_size")
    {
        return Err(bad("unsupported DequantizeLinear inputs or attributes"));
    }
    let axis = strict_typed_scalar_i64_attr(n, "axis")?.unwrap_or(1);
    let block_size = strict_typed_scalar_i64_attr(n, "block_size")?.unwrap_or(0);
    let x = inputs[0];
    let scale = inputs[1];
    let zero = inputs.get(2).copied();
    let shape = g.shape(x)?.clone();
    let sd = g.dtype(scale)?;
    for id in inputs {
        let s = g.shape(*id)?;
        s.numel()?
            .checked_mul(g.dtype(*id)?.itemsize())
            .ok_or_else(|| bad("DequantizeLinear input byte extent overflow"))?;
    }
    let scale_plan = quant_parameter_plan(g, scale, &shape, axis, block_size)?;
    let zero_plan = zero
        .map(|z| quant_parameter_plan(g, z, &shape, axis, block_size))
        .transpose()?;
    let subtract_dtype = dequantize_dtype(
        DType::I32,
        zero.map(|z| g.dtype(z)).transpose()?.unwrap_or(DType::I32),
    );
    let multiply_dtype = dequantize_dtype(subtract_dtype, sd);
    for (s, d) in [
        (&shape, DType::I32),
        (&shape, subtract_dtype),
        (&shape, multiply_dtype),
        (&shape, sd),
    ] {
        s.numel()?
            .checked_mul(d.itemsize())
            .ok_or_else(|| bad("DequantizeLinear output byte extent overflow"))?;
    }
    Ok(DequantizeLinearPlan {
        x,
        scale,
        zero,
        scale_plan,
        zero_plan,
        subtract_dtype,
        multiply_dtype,
        output_dtype: sd,
        shape,
    })
}

fn emit_quant_parameter(
    g: &mut Graph,
    parameter: NodeId,
    plan: QuantParameterPlan,
) -> Result<NodeId> {
    match plan {
        QuantParameterPlan::Keep => Ok(parameter),
        QuantParameterPlan::Reshape(shape) => g.reshape(parameter, shape),
        QuantParameterPlan::Repeat { repeats, axis, .. } => {
            g.repeat_interleave(parameter, repeats, Some(axis))
        }
    }
}

pub(super) fn lower(
    g: &mut Graph,
    n: Msg<'_>,
    values: &mut BTreeMap<String, NodeId>,
    constants: &mut BTreeMap<String, TensorData>,
) -> Result<()> {
    if !n.string(7)?.unwrap_or("").is_empty() {
        return Err(bad("ONNX custom domains and attributes are unsupported"));
    }
    let op = n.string(4)?.ok_or_else(|| bad("ONNX node lacks op_type"))?;
    let ins = n.strings(1)?;
    let outs = n.strings(2)?;
    if op == "MaxPool" && outs.len() == 2 {
        return Err(bad("MaxPool indices output is unsupported"));
    }
    let topk_outputs = op == "TopK";
    let split_outputs = op == "Split";
    let outputs_are_valid = if topk_outputs || split_outputs {
        let expected = if topk_outputs { Some(2) } else { None };
        expected.is_none_or(|count| outs.len() == count)
            && !outs.is_empty()
            && outs.iter().all(|output| !output.is_empty())
            && outs.iter().enumerate().all(|(index, output)| {
                !values.contains_key(*output) && !outs[..index].contains(output)
            })
    } else {
        outs.len() == 1 && !outs[0].is_empty() && !values.contains_key(outs[0])
    };
    if !outputs_are_valid {
        return Err(bad("invalid or duplicate ONNX node output"));
    }
    let get = |i: usize| -> Result<NodeId> {
        ins.get(i)
            .and_then(|x| values.get(*x))
            .copied()
            .ok_or_else(|| bad("missing ONNX node input"))
    };
    let attrs = attrs(&n)?;
    if topk_outputs {
        let input = get(0)?;
        let plan = topk_plan(g, input, &ins, &n, &attrs, constants)?;
        // `topk` itself preflights its stable Sort pair and both Shrink views
        // before it can append either selector. The final cast has only the
        // already-validated output shape. With both names checked above, the
        // two map insertions below cannot fail or expose a half-pair.
        let (top_values, top_indices) = g.topk(input, plan.k, plan.axis, plan.largest, true)?;
        let top_indices = g.cast(top_indices, DType::I64)?;
        values.insert(outs[0].to_owned(), top_values);
        values.insert(outs[1].to_owned(), top_indices);
        return Ok(());
    }
    if split_outputs {
        let input = get(0)?;
        let plan = split_plan(g, input, &ins, &outs, &n, &attrs, constants)?;
        // `Graph::split` establishes the full coverage/range list before its
        // first Shrink. The plan above independently proves every emitted
        // descriptor, and output names were all reserved before construction.
        let outputs = g.split(
            input,
            crate::SplitSections::Explicit(plan.sections),
            plan.axis,
        )?;
        debug_assert_eq!(outputs.len(), outs.len());
        for (name, output) in outs.iter().zip(outputs) {
            values.insert((*name).to_owned(), output);
        }
        return Ok(());
    }
    let out = match op {
        "Identity" if ins.len() == 1 && attrs.is_empty() => get(0)?,
        "EyeLike" if ins.len() == 1 => {
            let plan = eye_like_plan(g, get(0)?, &n, &attrs)?;
            g.constant(plan.data)
        }
        "SpaceToDepth" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = space_to_depth_plan(g, input, &n, &attrs)?;
            if plan.identity {
                // The source rearrange is observationally an identity at a
                // unit block after all descriptor checks have completed.
                input
            } else {
                let first = g.reshape(input, plan.first_shape)?;
                let permuted = g.permute(first, [0, 3, 5, 1, 2, 4])?;
                let output = g.reshape(permuted, plan.output_shape)?;
                debug_assert_eq!(
                    g.dtype(output).expect("SpaceToDepth dtype preflighted"),
                    g.dtype(input).expect("SpaceToDepth input dtype")
                );
                output
            }
        }
        "DepthToSpace" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = depth_to_space_plan(g, input, &n, &attrs)?;
            if plan.identity {
                input
            } else {
                let first = g.reshape(input, plan.first_shape)?;
                let permuted = g.permute(first, plan.permutation)?;
                let output = g.reshape(permuted, plan.output_shape)?;
                debug_assert_eq!(
                    g.dtype(output).expect("DepthToSpace dtype preflighted"),
                    g.dtype(input).expect("DepthToSpace input dtype")
                );
                output
            }
        }
        "CenterCropPad" if ins.len() == 2 => {
            let input = get(0)?;
            let plan = center_crop_pad_plan(g, input, &ins, &n, &attrs, constants)?;
            let cropped = match plan.shrink {
                Some(bounds) => g.shrink(input, bounds)?,
                None => input,
            };
            match plan.padding {
                Some(padding) => g.pad(cropped, padding, plan.fill)?,
                None => cropped,
            }
        }
        "LRN" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = lrn_plan(g, input, &n, &attrs)?;
            if plan.empty {
                // Tinygrad's empty arithmetic has no values to normalize;
                // only its true-division result dtype remains observable.
                if g.dtype(input)? == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                }
            } else {
                // `x ** 2` is source-storage multiplication: it intentionally
                // wraps integral values and rounds F16/BF16 before mean's
                // separate accumulator widening.
                let squared = g.mul(input, input)?;
                let reshaped = g.reshape(squared, plan.reshaped.clone())?;
                let padded = g.pad(
                    reshaped,
                    plan.padding,
                    center_crop_pad_zero(plan.input_dtype),
                )?;
                let padded = if plan.input_dtype == plan.sum_dtypes.accumulator {
                    padded
                } else {
                    g.cast(padded, plan.sum_dtypes.accumulator)?
                };
                let mut windows = Vec::with_capacity(plan.windows.len());
                for slices in plan.windows {
                    windows.push(g.stride(padded, slices)?);
                }
                let stacked = g.stack(windows, -1)?;
                let summed = g.reduce_with_dtypes(
                    stacked,
                    ReduceKind::Sum,
                    Some(vec![-1]),
                    false,
                    ReductionDType::new(plan.sum_dtypes.accumulator, plan.sum_dtypes.accumulator),
                )?;
                let summed = if plan.pool_dtype == plan.sum_dtypes.accumulator {
                    summed
                } else {
                    g.cast(summed, plan.pool_dtype)?
                };
                let divisor = g.constant(plan.divisor);
                let mut pooled = g.div(summed, divisor)?;
                if plan.narrow_pool {
                    pooled = g.cast(pooled, plan.output_dtype)?;
                }
                let pooled = g.reshape(pooled, plan.output_shape.clone())?;
                let alpha = g.constant(plan.alpha);
                let bias = g.constant(plan.bias);
                let beta = g.constant(plan.beta);
                let denominator = g.add(g.mul(pooled, alpha)?, bias)?;
                let denominator = g.pow(denominator, beta)?;
                let numerator = if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                };
                let output = g.div(numerator, denominator)?;
                debug_assert_eq!(
                    g.shape(output).expect("LRN shape preflighted"),
                    &plan.output_shape
                );
                debug_assert_eq!(
                    g.dtype(output).expect("LRN dtype preflighted"),
                    plan.output_dtype
                );
                output
            }
        }
        "Gelu" if ins.len() == 1 => {
            let input = get(0)?;
            if attrs.keys().any(|key| key != "approximate") {
                return Err(bad("unsupported Gelu attribute"));
            }
            // tinygrad's dedicated ONNX handler maps an omitted attribute to
            // exact `none` (unlike Tensor.gelu's public tanh default).  The
            // shared helper then validates and lowers the complete selected
            // literal composition before publishing any constants or nodes.
            let mode =
                strict_typed_string_attr(&n, "approximate")?.unwrap_or_else(|| "none".into());
            if mode != "none" && mode != "tanh" {
                return Err(bad("unsupported Gelu approximation"));
            }
            g.gelu(input, &mode)?
        }
        "BiasGelu" if ins.len() == 2 => {
            let plan = bias_gelu_plan(g, &ins, &n, &attrs, values)?;
            let gelu_input = g.add(plan.add.lhs, plan.add.rhs)?;
            debug_assert_eq!(
                g.shape(gelu_input).expect("BiasGelu add preflighted"),
                &plan.add.output_shape
            );
            debug_assert_eq!(
                g.dtype(gelu_input).expect("BiasGelu add preflighted"),
                plan.add.output_dtype
            );
            let output = g.gelu(gelu_input, &plan.mode)?;
            debug_assert_eq!(
                g.shape(output).expect("BiasGelu GELU preflighted"),
                &plan.add.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("BiasGelu GELU preflighted"),
                plan.output_dtype
            );
            output
        }
        "OptionalHasElement" if ins.len() <= 1 => {
            // The source returns a fresh Bool scalar, never the optional
            // payload: resolve absent/empty/present wholly before publication.
            g.constant(optional_has_element_plan(g, &ins, &attrs, values)?)
        }
        "OptionalGetElement" if ins.len() <= 1 => {
            match optional_get_element_plan(g, &ins, &attrs, values)? {
                OptionalGetElementPlan::Alias(input) => input,
                OptionalGetElementPlan::Empty(data) => g.constant(data),
            }
        }
        "FastGelu" if (1..=2).contains(&ins.len()) => {
            let plan = fast_gelu_plan(g, &ins, &attrs, values)?;
            let gelu_input = if let Some(bias) = plan.bias {
                // The source owns this source-LUB cast/broadcast boundary
                // before taking the public tanh GELU path.
                g.add(plan.input, bias)?
            } else {
                plan.input
            };
            debug_assert_eq!(
                g.shape(gelu_input).expect("FastGelu add preflighted"),
                &plan.gelu_input_shape
            );
            debug_assert_eq!(
                g.dtype(gelu_input).expect("FastGelu add preflighted"),
                plan.gelu_input_dtype
            );
            let output = g.gelu(gelu_input, "tanh")?;
            debug_assert_eq!(
                g.shape(output).expect("FastGelu shape preflighted"),
                &plan.gelu_input_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("FastGelu dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Elu" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = elu_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                }
            } else {
                let x = if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                };
                let zero = g.constant(plan.zero);
                let one = g.constant(plan.one);
                let alpha = g.constant(plan.alpha);
                // Match Tensor.elu literally: each ReLU is a strict select.
                // This intentionally maps NaN through the false branch.
                let positive = g.select(g.gt(x, zero)?, x, zero)?;
                let exp_x = g.exp(x)?;
                let negative_raw = g.sub(one, exp_x)?;
                let negative = g.select(g.gt(negative_raw, zero)?, negative_raw, zero)?;
                let output = g.sub(positive, g.mul(alpha, negative)?)?;
                debug_assert_eq!(g.shape(output).expect("Elu shape preflighted"), &plan.shape);
                debug_assert_eq!(
                    g.dtype(output).expect("Elu dtype preflighted"),
                    plan.output_dtype
                );
                output
            }
        }
        "Celu" if ins.len() == 1 => {
            let input = get(0)?;
            if attrs.keys().any(|key| key != "alpha") {
                return Err(bad("unsupported Celu attribute"));
            }
            // ONNX's declared FLOAT is the same weak Python scalar accepted
            // by parameterless Tensor.celu.  The shared scalar helper proves
            // all source-order descriptors before it publishes alpha or any
            // extrema/division nodes.
            let alpha = typed_scalar_f32_attr(&n, "alpha")?.unwrap_or(1.0);
            g.celu_scalar(input, f64::from(alpha))?
        }
        "Selu" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = selu_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                }
            } else {
                let x = if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                };
                let zero = g.constant(plan.zero);
                let one = g.constant(plan.one);
                let alpha = g.constant(plan.alpha);
                let gamma = g.constant(plan.gamma);
                // SELU's >= condition deliberately keeps both signed zeroes
                // on the X branch; NaN takes the exponential branch.
                let condition = g.ge(x, zero)?;
                let negative = g.mul(alpha, g.sub(g.exp(x)?, one)?)?;
                let branch = g.select(condition, x, negative)?;
                let output = g.mul(gamma, branch)?;
                debug_assert_eq!(
                    g.shape(output).expect("Selu shape preflighted"),
                    &plan.shape
                );
                debug_assert_eq!(
                    g.dtype(output).expect("Selu dtype preflighted"),
                    plan.output_dtype
                );
                output
            }
        }
        "Swish" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = swish_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                }
            } else {
                let x = if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                };
                let alpha = g.constant(plan.alpha);
                let one = g.constant(plan.one);
                let neg_inv_ln2 = g.constant(plan.neg_inv_ln2);
                let scaled = g.mul(x, alpha)?;
                let exponent = g.mul(scaled, neg_inv_ln2)?;
                let sigmoid = g.reciprocal(g.add(one, g.exp2(exponent)?)?)?;
                let output = g.mul(x, sigmoid)?;
                debug_assert_eq!(
                    g.shape(output).expect("Swish shape preflighted"),
                    &plan.shape
                );
                debug_assert_eq!(
                    g.dtype(output).expect("Swish dtype preflighted"),
                    plan.output_dtype
                );
                output
            }
        }
        "Softplus" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = softplus_plan(g, input, &attrs)?;
            // Do not simplify away either default-beta boundary: tinygrad's
            // parameterless ONNX dispatch invokes the literal public
            // `(1/beta) * (x*beta).logaddexp(0)` composition.
            let output = g.softplus(input, g.constant(plan.beta))?;
            debug_assert_eq!(
                g.shape(output).expect("Softplus shape preflighted"),
                &plan.shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Softplus dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Softsign" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = softsign_plan(g, input, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                }
            } else {
                let one = g.constant(plan.one);
                // Keep tinygrad's literal `x / (1 + x.abs())` decomposition:
                // abs is `x * sign(x)` and true division is reciprocal then
                // multiply. Unary Abs and Graph::softsign erase those
                // source-visible storage and signed-zero boundaries.
                let absolute = g.mul(input, g.sign(input)?)?;
                let denominator = g.add(one, absolute)?;
                let output = g.mul(input, g.reciprocal(denominator)?)?;
                debug_assert_eq!(
                    g.shape(output).expect("Softsign shape preflighted"),
                    &plan.shape
                );
                debug_assert_eq!(
                    g.dtype(output).expect("Softsign dtype preflighted"),
                    plan.output_dtype
                );
                output
            }
        }
        "Mod" if ins.len() == 2 => {
            let lhs = get(0)?;
            let rhs = get(1)?;
            let plan = mod_plan(g, lhs, rhs, ins[1], &n, &attrs, constants)?;
            let output = if plan.fmod {
                g.fmod(lhs, rhs)?
            } else {
                g.modulo(lhs, rhs)?
            };
            debug_assert_eq!(g.shape(output).expect("Mod shape preflighted"), &plan.shape);
            debug_assert_eq!(g.dtype(output).expect("Mod dtype preflighted"), plan.dtype);
            output
        }
        "OneHot" if ins.len() == 3 => {
            let indices = get(0)?;
            let values_input = get(2)?;
            let plan = one_hot_plan(g, indices, values_input, &ins, &attrs, constants)?;

            // This follows the source adapter literally rather than using
            // Graph::one_hot: I32 conversion, one negative adjustment, an
            // arbitrary inserted class axis, and live off/on values.
            let indices = g.cast(indices, DType::I32)?;
            let zero = g.constant(plan.index_zero);
            let depth = g.constant(plan.index_depth);
            let negative = g.lt(indices, zero)?;
            let shifted = g.add(indices, depth)?;
            let indices = g.select(negative, shifted, indices)?;
            let indices = g.unsqueeze(indices, plan.axis)?;
            let classes = g.constant(plan.classes);
            let classes = g.reshape(classes, plan.class_shape)?;
            let mask = g.eq(indices, classes)?;
            let off = g.squeeze(g.shrink(values_input, plan.off_bounds)?, Some(0))?;
            let on = g.squeeze(g.shrink(values_input, plan.on_bounds)?, Some(0))?;
            let output = g.select(mask, on, off)?;
            debug_assert_eq!(
                g.shape(output).expect("OneHot shape preflighted"),
                &plan.result_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("OneHot dtype preflighted"),
                plan.result_dtype
            );
            output
        }
        "Hardmax" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = hardmax_plan(g, input, &n, &attrs)?;
            if plan.empty {
                // Restoring a zero-sized class axis has no observable values;
                // retain the source tensor identity after complete planning.
                input
            } else {
                let indices = g.argmax(input, Some(plan.axis), false)?;
                let first = g.squeeze(g.shrink(input, plan.first_bounds)?, Some(plan.axis))?;
                let leading_nan = g.isnan(first)?;
                let sentinel = g.constant(plan.sentinel);
                let indices = g.select(leading_nan, sentinel, indices)?;
                let indices = g.unsqueeze(indices, plan.axis)?;
                let classes = g.constant(plan.classes);
                let classes = g.reshape(classes, plan.class_shape)?;
                let mask = g.eq(indices, classes)?;
                let output = g.cast(mask, plan.output_dtype)?;
                debug_assert_eq!(
                    g.shape(output).expect("Hardmax shape preflighted"),
                    &plan.output_shape
                );
                output
            }
        }
        "Shrink" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = shrink_activation_plan(g, input, &n, &attrs)?;

            // Keep tinygrad's two products intact. In particular, neither
            // false mask may erase NaN or infinity from its arithmetic branch,
            // and a negative lambda can intentionally enable both products.
            let work = if g.dtype(input)? == plan.work_dtype {
                input
            } else {
                g.cast(input, plan.work_dtype)?
            };
            let negative_lambda = g.constant(plan.negative_lambda);
            let lambda = g.constant(plan.lambda);
            let bias = g.constant(plan.bias);
            let lower_mask = g.lt(work, negative_lambda)?;
            let upper_mask = g.gt(work, lambda)?;
            let mut lower_branch = g.add(work, bias)?;
            let mut upper_branch = g.sub(work, bias)?;
            if plan.narrow {
                // tinygrad rounds each storage-width branch before the Bool
                // mask product, rather than deferring both rounds to the end.
                lower_branch = g.cast(lower_branch, plan.output_dtype)?;
                upper_branch = g.cast(upper_branch, plan.output_dtype)?;
            }
            let lower_product = g.mul(lower_mask, lower_branch)?;
            let upper_product = g.mul(upper_mask, upper_branch)?;
            let output = g.add(lower_product, upper_product)?;
            debug_assert_eq!(
                g.shape(output).expect("Shrink shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Shrink dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Relu" => {
            let input = get(0)?;
            let plan = relu_plan(g, input, &ins, &attrs)?;
            let output = g.relu(input)?;
            debug_assert_eq!(
                g.shape(output).expect("Relu shape preflighted"),
                &plan.shape
            );
            debug_assert_eq!(g.dtype(output).expect("Relu dtype preflighted"), plan.dtype);
            output
        }
        "Sigmoid" => {
            if ins.len() != 1 || !attrs.is_empty() {
                return Err(bad("Sigmoid requires exactly one input and no attributes"));
            }
            // tinygrad dispatches directly to its parameterless Tensor.sigmoid
            // composition; Graph::sigmoid owns its typed full-operation plan.
            g.sigmoid(get(0)?)?
        }
        "Tanh" => {
            if ins.len() != 1 || !attrs.is_empty() {
                return Err(bad("Tanh requires exactly one input and no attributes"));
            }
            // tinygrad dispatches directly to its parameterless Tensor.tanh
            // composition; Graph::tanh owns its typed full-operation plan.
            g.tanh(get(0)?)?
        }
        "Add" if ins.len() == 2 => {
            let plan = add_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("Add lhs preflighted") == plan.output_dtype {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.output_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("Add rhs preflighted") == plan.output_dtype {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.output_dtype)?
            };
            let output = g.add(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("Add shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Add dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Max" => {
            let plan = variadic_max_plan(g, &ins, &attrs, values)?;
            let mut maximum = plan.first;
            for fold in plan.folds {
                // Tensor.maximum casts both sides before comparing.  Do not
                // use Graph::maximum here: its extrema primitive has a
                // deliberately different cross-evaluator NaN/tie policy.
                let lhs = if g.dtype(maximum)? == fold.dtype {
                    maximum
                } else {
                    g.cast(maximum, fold.dtype)?
                };
                let rhs = if g.dtype(fold.input)? == fold.dtype {
                    fold.input
                } else {
                    g.cast(fold.input, fold.dtype)?
                };
                let condition = g.lt(lhs, rhs)?;
                maximum = g.select(condition, rhs, lhs)?;
                debug_assert_eq!(
                    g.shape(maximum).expect("Max shape preflighted"),
                    &fold.shape
                );
                debug_assert_eq!(g.dtype(maximum).expect("Max dtype preflighted"), fold.dtype);
            }
            debug_assert_eq!(
                g.shape(maximum).expect("Max output shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(maximum).expect("Max output dtype preflighted"),
                plan.output_dtype
            );
            maximum
        }
        "Min" => {
            let plan = variadic_min_plan(g, &ins, &attrs, values)?;
            let mut minimum = plan.first;
            for fold in plan.folds {
                // Tensor.minimum's negated/bias-transformed Max path is
                // observably `lhs > rhs ? rhs : lhs`.  Keep that ordered
                // comparison instead of Graph::minimum's extrema policy.
                let lhs = if g.dtype(minimum)? == fold.dtype {
                    minimum
                } else {
                    g.cast(minimum, fold.dtype)?
                };
                let rhs = if g.dtype(fold.input)? == fold.dtype {
                    fold.input
                } else {
                    g.cast(fold.input, fold.dtype)?
                };
                let condition = g.gt(lhs, rhs)?;
                minimum = g.select(condition, rhs, lhs)?;
                debug_assert_eq!(
                    g.shape(minimum).expect("Min shape preflighted"),
                    &fold.shape
                );
                debug_assert_eq!(g.dtype(minimum).expect("Min dtype preflighted"), fold.dtype);
            }
            debug_assert_eq!(
                g.shape(minimum).expect("Min output shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(minimum).expect("Min output dtype preflighted"),
                plan.output_dtype
            );
            minimum
        }
        "Sum" => {
            let plan = variadic_sum_plan(g, &ins, &attrs, values)?;
            lower_variadic_sum_plan(g, plan)?
        }
        "Mean" => {
            let plan = variadic_mean_plan(g, &ins, &attrs, values)?;
            let sum = lower_variadic_sum_plan(g, plan.sum)?;
            let sum = if g.dtype(sum).expect("Mean sum preflighted") == plan.division_dtype {
                sum
            } else {
                g.cast(sum, plan.division_dtype)?
            };
            let divisor = g.constant(plan.divisor);
            let output = g.div(sum, divisor)?;
            debug_assert_eq!(
                g.shape(output).expect("Mean shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Mean dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Sub" if ins.len() == 2 => {
            let plan = sub_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("Sub lhs preflighted") == plan.output_dtype {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.output_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("Sub rhs preflighted") == plan.output_dtype {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.output_dtype)?
            };
            let output = g.sub(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("Sub shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Sub dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Mul" if ins.len() == 2 => {
            let plan = mul_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("Mul lhs preflighted") == plan.output_dtype {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.output_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("Mul rhs preflighted") == plan.output_dtype {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.output_dtype)?
            };
            let output = g.mul(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("Mul shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Mul dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "BitwiseAnd" if ins.len() == 2 => {
            let plan = bitwise_binary_plan(g, &ins, &attrs, values, "BitwiseAnd")?;
            let lhs = if g.dtype(plan.lhs).expect("BitwiseAnd lhs preflighted") == plan.output_dtype
            {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.output_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("BitwiseAnd rhs preflighted") == plan.output_dtype
            {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.output_dtype)?
            };
            let output = g.bit_and(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("BitwiseAnd shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("BitwiseAnd dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "BitwiseOr" if ins.len() == 2 => {
            let plan = bitwise_binary_plan(g, &ins, &attrs, values, "BitwiseOr")?;
            let lhs = if g.dtype(plan.lhs).expect("BitwiseOr lhs preflighted") == plan.output_dtype
            {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.output_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("BitwiseOr rhs preflighted") == plan.output_dtype
            {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.output_dtype)?
            };
            let output = g.bit_or(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("BitwiseOr shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("BitwiseOr dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "BitwiseXor" if ins.len() == 2 => {
            let plan = bitwise_binary_plan(g, &ins, &attrs, values, "BitwiseXor")?;
            let lhs = if g.dtype(plan.lhs).expect("BitwiseXor lhs preflighted") == plan.output_dtype
            {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.output_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("BitwiseXor rhs preflighted") == plan.output_dtype
            {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.output_dtype)?
            };
            let output = g.bit_xor(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("BitwiseXor shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("BitwiseXor dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "BitwiseNot" if ins.len() == 1 => {
            if !attrs.is_empty() {
                return Err(bad(
                    "BitwiseNot requires exactly one input and no attributes",
                ));
            }
            g.bitwise_not(
                *values
                    .get(ins[0])
                    .ok_or_else(|| bad("missing ONNX input"))?,
            )?
        }
        "Div" if ins.len() == 2 => {
            let plan = div_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("Div lhs preflighted") == plan.work_dtype {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.work_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("Div rhs preflighted") == plan.work_dtype {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.work_dtype)?
            };
            let output = if plan.integer_division {
                g.trunc_div(lhs, rhs)?
            } else {
                let quotient = g.mul(lhs, g.reciprocal(rhs)?)?;
                if plan.truncate {
                    g.trunc(quotient)?
                } else {
                    quotient
                }
            };
            debug_assert_eq!(
                g.shape(output).expect("Div shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Div dtype preflighted"),
                plan.work_dtype
            );
            output
        }
        "MatMul" if ins.len() == 2 && attrs.is_empty() => {
            let lhs = get(0)?;
            let rhs = get(1)?;
            g.matmul(lhs, rhs)?
        }
        "Cast" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = cast_plan(g, input, &ins, &n, &attrs)?;
            // Tensor.cast is an exact identity when the concrete dtype already
            // matches.  Avoid publishing RustGrad's otherwise redundant Cast.
            if plan.input_dtype == plan.output_dtype {
                plan.input
            } else {
                g.cast(plan.input, plan.output_dtype)?
            }
        }
        "CastLike" if ins.len() == 2 => {
            if attrs.keys().any(|name| name != "saturate") {
                return Err(bad("unsupported CastLike attribute"));
            }
            // tinygrad accepts the opset's scalar saturate attribute but, as
            // documented beside its adapter, it applies only to FP8 types
            // outside RustGrad's supported dtype set.
            if let Some(saturate) = attrs.get("saturate") {
                scalar_i64(saturate)?;
            }
            let input = get(0)?;
            let target = get(1)?;
            let input_dtype = g.dtype(input)?;
            let target_dtype = g.dtype(target)?;
            let numel = g.shape(input)?.numel()?;
            numel
                .checked_mul(input_dtype.itemsize())
                .ok_or_else(|| bad("CastLike input byte extent overflow"))?;
            numel
                .checked_mul(target_dtype.itemsize())
                .ok_or_else(|| bad("CastLike output byte extent overflow"))?;
            // tinygrad derives CastLike entirely from target_type.dtype; its
            // values and shape have no effect on the result.
            if input_dtype == target_dtype {
                input
            } else {
                g.cast(input, target_dtype)?
            }
        }
        "Constant" if ins.is_empty() => {
            let data = constant_plan(&n, &ins, &attrs)?;
            let output = g.constant(data.clone());
            constants.insert(outs[0].to_owned(), data);
            output
        }
        "Reshape" if ins.len() == 2 => {
            let x = get(0)?;
            let plan = reshape_plan(g, x, &ins, &n, &attrs, constants)?;
            let output = if plan.identity {
                x
            } else {
                g.reshape(x, plan.output_shape.clone())?
            };
            debug_assert_eq!(
                g.shape(output).expect("Reshape shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Reshape dtype preflighted"),
                plan.dtype
            );
            output
        }
        "Transpose" if ins.len() == 1 => {
            let x = get(0)?;
            let plan = transpose_plan(g, x, &ins, &n, &attrs)?;
            let output = if plan.identity {
                x
            } else {
                g.permute(x, plan.axes)?
            };
            debug_assert_eq!(
                g.shape(output).expect("Transpose shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Transpose dtype preflighted"),
                plan.dtype
            );
            output
        }
        "Flatten" if ins.len() == 1 => {
            let x = get(0)?;
            let plan = flatten_plan(g, x, &ins, &n, &attrs)?;
            let output = if plan.identity {
                x
            } else {
                g.reshape(x, plan.output_shape.clone())?
            };
            debug_assert_eq!(
                g.shape(output).expect("Flatten shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Flatten dtype preflighted"),
                plan.dtype
            );
            output
        }
        "Squeeze" if (1..=2).contains(&ins.len()) => {
            let x = get(0)?;
            let plan = squeeze_plan(g, x, &ins, &attrs, constants)?;
            let output = if plan.identity {
                x
            } else {
                g.reshape(x, plan.output_shape.clone())?
            };
            debug_assert_eq!(
                g.shape(output).expect("Squeeze shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Squeeze dtype preflighted"),
                plan.dtype
            );
            output
        }
        "Unsqueeze" if ins.len() == 2 => {
            let x = get(0)?;
            let plan = unsqueeze_plan(g, x, &ins, &attrs, constants)?;
            let output = if plan.identity {
                x
            } else {
                g.reshape(x, plan.output_shape.clone())?
            };
            debug_assert_eq!(
                g.shape(output).expect("Unsqueeze shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Unsqueeze dtype preflighted"),
                plan.dtype
            );
            output
        }
        "Concat" if !ins.is_empty() => {
            let plan = concat_plan(g, &ins, &n, &attrs, values)?;
            let output = if plan.identity {
                plan.inputs[0]
            } else {
                match plan.lowering {
                    ConcatLowering::Stack => {
                        let inputs = plan
                            .inputs
                            .iter()
                            .map(|&input| {
                                if g.dtype(input).expect("Concat stack input preflighted")
                                    == plan.output_dtype
                                {
                                    Ok(input)
                                } else {
                                    g.cast(input, plan.output_dtype)
                                }
                            })
                            .collect::<Result<Vec<_>>>()?;
                        g.concat(inputs, plan.axis)?
                    }
                    ConcatLowering::PadSum { paddings } => {
                        let mut padded = plan
                            .inputs
                            .iter()
                            .copied()
                            .zip(paddings)
                            .map(|(input, padding)| {
                                let dtype = g.dtype(input).expect("Concat pad input preflighted");
                                g.pad(input, padding, center_crop_pad_zero(dtype))
                            })
                            .collect::<Result<Vec<_>>>()?
                            .into_iter();
                        let mut output = padded.next().expect("Concat plan has input");
                        for input in padded {
                            output = g.add(output, input)?;
                        }
                        output
                    }
                }
            };
            debug_assert_eq!(
                g.shape(output).expect("Concat shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Concat dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Softmax" if ins.len() == 1 => {
            let x = get(0)?;
            let plan = softmax_plan(g, x, &n, &attrs)?;
            if plan.empty {
                if plan.output_dtype == plan.source_dtype {
                    x
                } else {
                    g.cast(x, plan.output_dtype)?
                }
            } else {
                let maximum = if let Some(axis) = plan.axis {
                    g.reduce(x, ReduceKind::Max, Some(vec![axis]), true)?
                } else {
                    x
                };
                // `_softmax` detaches only Max, then carries the centered
                // Exp/Sum branch into the source's reciprocal multiplication.
                let centered = g.sub(x, g.detach(maximum)?)?;
                let exp_work = if plan.exp_work_dtype == plan.source_dtype {
                    centered
                } else {
                    g.cast(centered, plan.exp_work_dtype)?
                };
                let inv_ln2 = g.constant(plan.inv_ln2);
                let exponentials = g.exp2(g.mul(exp_work, inv_ln2)?)?;
                let exponentials = if plan.exp_dtype == plan.exp_work_dtype {
                    exponentials
                } else {
                    g.cast(exponentials, plan.exp_dtype)?
                };
                let sum = if let Some(axis) = plan.axis {
                    g.reduce_with_dtypes(
                        exponentials,
                        ReduceKind::Sum,
                        Some(vec![axis]),
                        true,
                        plan.sum_dtypes,
                    )?
                } else {
                    exponentials
                };
                let reciprocal = g.reciprocal(sum)?;
                g.mul(exponentials, reciprocal)?
            }
        }
        "LogSoftmax" if ins.len() == 1 => {
            let x = get(0)?;
            let plan = log_softmax_plan(g, x, &n, &attrs)?;
            // An empty source has no populated Max/Sum reduction domain in
            // tinygrad. Its observable result is the same typed empty value;
            // exact storage is preserved for floats while exact input dtypes
            // follow the public transcendental F32 promotion.
            if plan.empty {
                if plan.output_dtype == plan.source_dtype {
                    x
                } else {
                    g.cast(x, plan.output_dtype)?
                }
            } else {
                let maximum = if let Some(axis) = plan.axis {
                    g.reduce(x, ReduceKind::Max, Some(vec![axis]), true)?
                } else {
                    x
                };
                // The source detaches only the maximum branch before the
                // centering subtraction; the Exp/Sum/Log path remains live.
                let centered = g.sub(x, g.detach(maximum)?)?;
                let exp_work = if plan.exp_work_dtype == plan.source_dtype {
                    centered
                } else {
                    g.cast(centered, plan.exp_work_dtype)?
                };
                let inv_ln2 = g.constant(plan.inv_ln2);
                let exponentials = g.exp2(g.mul(exp_work, inv_ln2)?)?;
                let exponentials = if plan.exp_dtype == plan.exp_work_dtype {
                    exponentials
                } else {
                    g.cast(exponentials, plan.exp_dtype)?
                };
                let sum = if let Some(axis) = plan.axis {
                    g.reduce_with_dtypes(
                        exponentials,
                        ReduceKind::Sum,
                        Some(vec![axis]),
                        true,
                        plan.sum_dtypes,
                    )?
                } else {
                    exponentials
                };
                let log2 = g.log2(sum)?;
                let ln2 = g.constant(plan.ln2);
                let logged = g.mul(log2, ln2)?;
                g.sub(centered, logged)?
            }
        }
        "Gemm" if ins.len() == 2 || ins.len() == 3 => {
            if attrs
                .keys()
                .any(|name| !matches!(name.as_str(), "alpha" | "beta" | "transA" | "transB"))
            {
                return Err(bad("unsupported Gemm attribute"));
            }
            let alpha = attrs
                .get("alpha")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(1.);
            let beta = attrs
                .get("beta")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(1.);
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(bad("Gemm alpha/beta must be finite"));
            }
            let transpose_attr = |name: &str| -> Result<bool> {
                match attrs.get(name) {
                    None => Ok(false),
                    Some(value) => match scalar_i64(value)? {
                        0 => Ok(false),
                        1 => Ok(true),
                        _ => Err(bad(format!("Gemm {name} must be 0 or 1"))),
                    },
                }
            };
            let trans_a = transpose_attr("transA")?;
            let trans_b = transpose_attr("transB")?;
            let transpose_shape = |shape: &Shape, on: bool| -> Result<Shape> {
                if !on {
                    return Ok(shape.clone());
                }
                let rank = shape.rank();
                if rank < 2 {
                    return Err(bad("Gemm transpose needs rank >= 2"));
                }
                let mut dims = shape.dims().to_vec();
                dims.swap(rank - 1, rank - 2);
                Ok(Shape::new(dims))
            };
            let a = get(0)?;
            let b = get(1)?;
            let c = (ins.len() == 3).then(|| get(2)).transpose()?;
            let a_shape = transpose_shape(g.shape(a)?, trans_a)?;
            let b_shape = transpose_shape(g.shape(b)?, trans_b)?;
            let output_shape = crate::ir::matmul_shape(&a_shape, &b_shape)
                .ok_or_else(|| bad("invalid Gemm matrix shapes"))?;
            output_shape.numel()?;
            if let Some(c) = c {
                output_shape.broadcast_with(g.shape(c)?)?;
            }
            let transpose = |g: &mut Graph, n: NodeId, on: bool| -> Result<NodeId> {
                if !on {
                    return Ok(n);
                }
                let rank = g.shape(n)?.rank();
                if rank < 2 {
                    return Err(bad("Gemm transpose needs rank >= 2"));
                }
                let mut p: Vec<usize> = (0..rank).collect();
                p.swap(rank - 1, rank - 2);
                g.permute(n, p)
            };
            let a = transpose(g, a, trans_a)?;
            let b = transpose(g, b, trans_b)?;
            let y = g.matmul(a, b)?;
            let y = if alpha == 1. {
                y
            } else {
                let scale = g.constant(TensorData::scalar(alpha));
                g.mul(y, scale)?
            };
            if let Some(c) = c {
                let c = if beta == 1. {
                    c
                } else {
                    let scale = g.constant(TensorData::scalar(beta));
                    g.mul(c, scale)?
                };
                g.add(y, c)?
            } else {
                y
            }
        }
        "Equal" if ins.len() == 2 => {
            let plan = equal_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("Equal lhs preflighted") == plan.comparison_dtype
            {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.comparison_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("Equal rhs preflighted") == plan.comparison_dtype
            {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.comparison_dtype)?
            };
            let output = g.eq(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("Equal shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Equal dtype preflighted"),
                DType::Bool
            );
            output
        }
        "Less" if ins.len() == 2 => {
            let plan = less_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("Less lhs preflighted") == plan.comparison_dtype {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.comparison_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("Less rhs preflighted") == plan.comparison_dtype {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.comparison_dtype)?
            };
            let output = g.lt(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("Less shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Less dtype preflighted"),
                DType::Bool
            );
            output
        }
        "LessOrEqual" if ins.len() == 2 => {
            let plan = less_or_equal_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("LessOrEqual lhs preflighted")
                == plan.comparison_dtype
            {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.comparison_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("LessOrEqual rhs preflighted")
                == plan.comparison_dtype
            {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.comparison_dtype)?
            };
            // Tensor.__le__ is `(self > rhs).logical_not()`, not a direct
            // ordered LE operation.  This retains tinygrad's unordered-NaN
            // truth value and its literal nondifferentiable graph structure.
            let output = g.logical_not(g.gt(lhs, rhs)?)?;
            debug_assert_eq!(
                g.shape(output).expect("LessOrEqual shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("LessOrEqual dtype preflighted"),
                DType::Bool
            );
            output
        }
        "Greater" if ins.len() == 2 => {
            let plan = greater_plan(g, &ins, &attrs, values)?;
            let lhs =
                if g.dtype(plan.lhs).expect("Greater lhs preflighted") == plan.comparison_dtype {
                    plan.lhs
                } else {
                    g.cast(plan.lhs, plan.comparison_dtype)?
                };
            let rhs =
                if g.dtype(plan.rhs).expect("Greater rhs preflighted") == plan.comparison_dtype {
                    plan.rhs
                } else {
                    g.cast(plan.rhs, plan.comparison_dtype)?
                };
            let output = g.gt(lhs, rhs)?;
            debug_assert_eq!(
                g.shape(output).expect("Greater shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Greater dtype preflighted"),
                DType::Bool
            );
            output
        }
        "GreaterOrEqual" if ins.len() == 2 => {
            let plan = greater_or_equal_plan(g, &ins, &attrs, values)?;
            let lhs = if g.dtype(plan.lhs).expect("GreaterOrEqual lhs preflighted")
                == plan.comparison_dtype
            {
                plan.lhs
            } else {
                g.cast(plan.lhs, plan.comparison_dtype)?
            };
            let rhs = if g.dtype(plan.rhs).expect("GreaterOrEqual rhs preflighted")
                == plan.comparison_dtype
            {
                plan.rhs
            } else {
                g.cast(plan.rhs, plan.comparison_dtype)?
            };
            let output = g.logical_not(g.lt(lhs, rhs)?)?;
            debug_assert_eq!(
                g.shape(output).expect("GreaterOrEqual shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("GreaterOrEqual dtype preflighted"),
                DType::Bool
            );
            output
        }
        "Where" if ins.len() == 3 => {
            let plan = where_plan(g, &ins, &attrs, values)?;
            let on_true = if g
                .dtype(plan.on_true)
                .expect("Where true branch preflighted")
                == plan.output_dtype
            {
                plan.on_true
            } else {
                g.cast(plan.on_true, plan.output_dtype)?
            };
            let on_false = if g
                .dtype(plan.on_false)
                .expect("Where false branch preflighted")
                == plan.output_dtype
            {
                plan.on_false
            } else {
                g.cast(plan.on_false, plan.output_dtype)?
            };
            let output = g.select(plan.condition, on_true, on_false)?;
            debug_assert_eq!(
                g.shape(output).expect("Where shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Where dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Not" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            // The public helper owns tinygrad's one Cast(Bool) followed by
            // comparison to true.  Pre-casting here would duplicate that
            // source-visible boundary for every ONNX Not root.
            g.logical_not(x)?
        }
        "IsInf" if ins.len() == 1 => {
            if attrs
                .keys()
                .any(|name| name != "detect_positive" && name != "detect_negative")
            {
                return Err(bad("unsupported IsInf attribute"));
            }
            // ONNX declares both IsInf selectors as AttributeProto INT.
            // Decode the original fields rather than the normalized raw map,
            // so a FLOAT/TENSOR wire payload or an omitted declared type
            // cannot be mistaken for an enabled selector.
            let detect_positive = strict_typed_scalar_i64_attr(&n, "detect_positive")?
                .map(|value| value != 0)
                .unwrap_or(true);
            let detect_negative = strict_typed_scalar_i64_attr(&n, "detect_negative")?
                .map(|value| value != 0)
                .unwrap_or(true);
            let x = get(0)?;
            g.dtype(x)?;
            g.shape(x)?.numel()?;
            g.isinf_with_signs(x, detect_positive, detect_negative)?
        }
        "IsNaN" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            g.dtype(x)?;
            g.shape(x)?.numel()?;
            g.isnan(x)?
        }
        "Xor" if ins.len() == 2 && attrs.is_empty() => {
            let lhs = get(0)?;
            let rhs = get(1)?;
            let lhs_dtype = g.dtype(lhs)?;
            let rhs_dtype = g.dtype(rhs)?;
            let lhs_shape = g.shape(lhs)?.clone();
            let rhs_shape = g.shape(rhs)?.clone();
            let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
            let extent = |shape: &Shape, dtype: DType, what: &str| {
                shape
                    .numel()?
                    .checked_mul(dtype.itemsize())
                    .ok_or_else(|| bad(format!("Xor {what} byte extent overflow")))
            };
            // Tinygrad casts both operands to Bool before the bitwise XOR.
            // Prove both source and cast descriptors, plus the broadcast
            // result, before either Cast can become observable.
            extent(&lhs_shape, lhs_dtype, "left input")?;
            extent(&rhs_shape, rhs_dtype, "right input")?;
            extent(&lhs_shape, DType::Bool, "left Bool cast")?;
            extent(&rhs_shape, DType::Bool, "right Bool cast")?;
            extent(&output_shape, DType::Bool, "output")?;
            // tinygrad explicitly converts both Xor operands to Bool before
            // applying its bitwise xor operation.
            let lhs = g.cast(lhs, DType::Bool)?;
            let rhs = g.cast(rhs, DType::Bool)?;
            g.bit_xor(lhs, rhs)?
        }
        "And" if ins.len() == 2 && attrs.is_empty() => {
            let lhs = get(0)?;
            let rhs = get(1)?;
            let lhs_dtype = g.dtype(lhs)?;
            g.dtype(rhs)?;
            let lhs_shape = g.shape(lhs)?.clone();
            let rhs_shape = g.shape(rhs)?.clone();
            lhs_shape.numel()?;
            rhs_shape.numel()?;
            let comparison_shape = lhs_shape.broadcast_with(&rhs_shape)?;
            let value_shape = lhs_shape.broadcast_with(&Shape::new([]))?;
            let output_shape = comparison_shape.broadcast_with(&value_shape)?;
            output_shape.numel()?;
            if lhs_dtype.promote(DType::Bool) != lhs_dtype {
                return Err(bad("And false scalar cannot preserve lhs dtype"));
            }
            // tinygrad defines And as `(x == y).where(x, False)`. Its weak
            // false scalar promotes to x's dtype; RustGrad's Bool promotion
            // has that same result for the select value branches.
            let equal = g.eq(lhs, rhs)?;
            let false_value = g.full_with_dtype([], Scalar::Bool(false), DType::Bool)?;
            g.select(equal, lhs, false_value)?
        }
        "Or" if ins.len() == 2 && attrs.is_empty() => {
            let lhs = get(0)?;
            let rhs = get(1)?;
            let lhs_dtype = g.dtype(lhs)?;
            g.dtype(rhs)?;
            let lhs_shape = g.shape(lhs)?.clone();
            let rhs_shape = g.shape(rhs)?.clone();
            lhs_shape.numel()?;
            rhs_shape.numel()?;
            let comparison_shape = lhs_shape.broadcast_with(&rhs_shape)?;
            let value_shape = lhs_shape.broadcast_with(&Shape::new([]))?;
            let output_shape = comparison_shape.broadcast_with(&value_shape)?;
            output_shape.numel()?;
            if lhs_dtype.promote(DType::Bool) != lhs_dtype {
                return Err(bad("Or true scalar cannot preserve lhs dtype"));
            }
            // tinygrad defines Or as `(x == y).where(x, True)`. Its weak
            // true scalar promotes to x's dtype; RustGrad's Bool promotion
            // has that same result for the select value branches.
            let equal = g.eq(lhs, rhs)?;
            let truth = g.full_with_dtype([], Scalar::Bool(true), DType::Bool)?;
            g.select(equal, lhs, truth)?
        }
        "Reciprocal" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            g.dtype(x)?;
            g.shape(x)?.numel()?;
            g.reciprocal(x)?
        }
        "Pow" if ins.len() == 2 => {
            let plan = pow_plan(g, &ins, &attrs, values)?;
            let base = if g.dtype(plan.base).expect("Pow base preflighted") == plan.work_dtype {
                plan.base
            } else {
                g.cast(plan.base, plan.work_dtype)?
            };
            let exponent =
                if g.dtype(plan.exponent).expect("Pow exponent preflighted") == plan.work_dtype {
                    plan.exponent
                } else {
                    g.cast(plan.exponent, plan.work_dtype)?
                };
            let value = g.pow(base, exponent)?;
            let output = if plan.integer_base {
                g.cast(g.round(value)?, plan.output_dtype)?
            } else {
                value
            };
            debug_assert_eq!(
                g.shape(output).expect("Pow shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Pow dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "Sqrt" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Sqrt preserves the input shape and applies Graph's established
            // floating promotion, so reject an invalid static output extent
            // before appending its unary node.
            g.shape(input)?.numel()?;
            g.sqrt(input)?
        }
        "Sin" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Sin preserves the input shape and applies Graph's established
            // floating promotion, so validate its static output extent before
            // appending the unary node.
            g.shape(input)?.numel()?;
            g.sin(input)?
        }
        "Cos" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Cos preserves the input shape and applies Graph's established
            // floating promotion, so validate its static output extent before
            // appending the unary node.
            g.shape(input)?.numel()?;
            g.cos(input)?
        }
        "Tan" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Tan preserves the input shape and applies Graph's established
            // floating promotion, so validate its static output extent before
            // appending the unary node.
            g.shape(input)?.numel()?;
            g.tan(input)?
        }
        "Asin" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Asin preserves the input shape and applies Graph's established
            // floating promotion, so validate its static output extent before
            // appending the unary node.
            g.shape(input)?.numel()?;
            g.asin(input)?
        }
        "Acos" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Acos preserves the input shape and applies Graph's established
            // floating promotion, so validate its static output extent before
            // appending the unary node.
            g.shape(input)?.numel()?;
            g.acos(input)?
        }
        "Atan" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Atan preserves the input shape and applies Graph's established
            // floating promotion, so validate its static output extent before
            // appending the unary node.
            g.shape(input)?.numel()?;
            g.atan(input)?
        }
        "Exp" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Exp preserves the input shape and applies Graph's established
            // floating promotion, so reject an invalid static output extent
            // before appending its unary node.
            g.shape(input)?.numel()?;
            g.exp(input)?
        }
        "Log" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Log preserves the input shape and applies Graph's established
            // floating promotion, so reject an invalid static output extent
            // before appending its unary node.
            g.shape(input)?.numel()?;
            g.log(input)?
        }
        "Floor" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Floor preserves the input shape and Graph defines both its
            // floating and exact integer paths. Validate the static output
            // extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.floor(input)?
        }
        "Ceil" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Ceil preserves the input shape and Graph defines both its
            // floating and exact integer paths. Validate the static output
            // extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.ceil(input)?
        }
        "Sign" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Sign preserves its input shape and Graph carries the checked
            // tinygrad NaN and signed-zero contract. Validate the static
            // output extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.sign(input)?
        }
        "Round" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Round preserves the input shape and Graph defines ties-to-even
            // together with exact integer and signed-zero behavior. Validate
            // the static output extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.round(input)?
        }
        "Erf" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Erf preserves its shape while Graph applies the established
            // floating promotion and deterministic A&S approximation. Check
            // the static output extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.erf(input)?
        }
        "Sinh" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Sinh preserves its shape while Graph applies its established
            // floating promotion and special-value behavior. Check the
            // static output extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.sinh(input)?
        }
        "Cosh" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Cosh preserves its shape while Graph applies its established
            // floating promotion and special-value behavior. Check the
            // static output extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.cosh(input)?
        }
        "Asinh" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Asinh preserves its shape while Graph applies its established
            // floating promotion and special-value behavior. Check the
            // static output extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.asinh(input)?
        }
        "Acosh" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Acosh preserves its shape while Graph applies its established
            // floating promotion and domain behavior. Check the static output
            // extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.acosh(input)?
        }
        "Atanh" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            g.dtype(input)?;
            // Atanh preserves its shape while Graph applies its established
            // floating promotion and domain behavior. Check the static output
            // extent before appending the unary node.
            g.shape(input)?.numel()?;
            g.atanh(input)?
        }
        "Abs" => {
            let input = get(0)?;
            let plan = abs_plan(g, input, &ins, &attrs)?;
            // tinygrad defines abs as source-storage `x * x.sign()`, not a
            // unary magnitude operation.  In particular, -0 * +0 is -0 and
            // signed-minimum multiplication wraps at the input width.
            let sign = g.sign(input)?;
            let output = g.mul(input, sign)?;
            debug_assert_eq!(g.shape(output).expect("Abs shape preflighted"), &plan.shape);
            debug_assert_eq!(g.dtype(output).expect("Abs dtype preflighted"), plan.dtype);
            output
        }
        "Neg" => {
            let input = get(0)?;
            let plan = neg_plan(g, input, &ins, &attrs)?;
            let output = g.neg(input)?;
            debug_assert_eq!(g.shape(output).expect("Neg shape preflighted"), &plan.shape);
            debug_assert_eq!(g.dtype(output).expect("Neg dtype preflighted"), plan.dtype);
            output
        }
        "LeakyRelu" if ins.len() == 1 => {
            let x = get(0)?;
            let plan = leaky_relu_plan(g, x, &ins, &n, &attrs)?;
            let zero = g.constant(plan.comparison_zero);
            let negative = g.lt(plan.input, zero)?;
            let value = if plan.input_dtype == plan.output_dtype {
                plan.input
            } else {
                g.cast(plan.input, plan.output_dtype)?
            };
            let alpha = g.constant(plan.alpha);
            let scaled = g.mul(alpha, value)?;
            let output = g.select(negative, scaled, value)?;
            debug_assert_eq!(
                g.shape(output).expect("LeakyRelu shape preflighted"),
                &plan.shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("LeakyRelu dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "HardSwish" if ins.len() == 1 && attrs.is_empty() => {
            // tinygrad dispatches ONNX HardSwish directly to Tensor.hardswish:
            // `x * (x + 3).relu6() * (1/6)`.  Graph::hardswish owns that
            // literal strict-Select plan and preflights all of its typed
            // intermediate descriptors before it publishes its first scalar.
            g.hardswish(get(0)?)?
        }
        "Mish" if ins.len() == 1 && attrs.is_empty() => {
            // tinygrad dispatches ONNX Mish directly to Tensor.mish:
            // `x * x.softplus().tanh()`. Graph::mish owns the source-default
            // beta and verifies its complete typed composite before publish.
            g.mish(get(0)?)?
        }
        "HardSigmoid" if ins.len() == 1 => {
            if attrs.keys().any(|name| name != "alpha" && name != "beta") {
                return Err(bad("unsupported HardSigmoid attribute"));
            }
            // tinygrad's adapter accepts IEEE FLOAT attribute values and
            // computes `(alpha*x + beta).clip(0, 1)`; it does not impose a
            // finite-attribute policy.
            let alpha = typed_scalar_f32_attr(&n, "alpha")?.unwrap_or(0.2);
            let beta = typed_scalar_f32_attr(&n, "beta")?.unwrap_or(0.5);
            let x = get(0)?;
            let input_shape = g.shape(x)?.clone();
            let input_dtype = g.dtype(x)?;
            let extent = |shape: &Shape, dtype: DType, what: &str| {
                shape
                    .numel()?
                    .checked_mul(dtype.itemsize())
                    .ok_or_else(|| bad(format!("Binarizer {what} byte extent overflow")))
                    .map(|_| ())
            };
            extent(&input_shape, input_dtype, "input")?;

            // Weak ONNX FLOAT scalars lift Bool/integer inputs to F32, retain
            // F32/F64 directly, and round each narrow-float arithmetic result
            // back to its storage width before the next expression.
            let output_dtype = match input_dtype {
                DType::F16 | DType::BF16 => input_dtype,
                DType::F64 => DType::F64,
                _ => DType::F32,
            };
            let work_dtype = if output_dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            };
            let narrow = matches!(output_dtype, DType::F16 | DType::BF16);
            let scalar_shape = Shape::new([]);
            let arithmetic_shape = input_shape.broadcast_with(&scalar_shape)?;
            arithmetic_shape.numel()?;
            let arithmetic_dtype = work_dtype.promote(DType::F32);
            if arithmetic_dtype != work_dtype {
                return Err(bad("HardSigmoid scalar promotion mismatch"));
            }
            // Simulate both strict comparisons and both Select value branches
            // before any constant or operation can become visible.
            let clamp_shape = arithmetic_shape.broadcast_with(&scalar_shape)?;
            clamp_shape.numel()?;
            let select_dtype = output_dtype.promote(output_dtype);
            if select_dtype != output_dtype {
                return Err(bad("HardSigmoid select promotion mismatch"));
            }

            let x = if input_dtype == work_dtype {
                x
            } else {
                g.cast(x, work_dtype)?
            };
            let alpha = g.constant(TensorData::scalar_with_dtype(
                Scalar::F(f64::from(alpha)),
                DType::F32,
            ));
            let mut value = g.mul(alpha, x)?;
            if narrow {
                value = g.cast(value, output_dtype)?;
                value = g.cast(value, DType::F32)?;
            }
            let beta = g.constant(TensorData::scalar_with_dtype(
                Scalar::F(f64::from(beta)),
                DType::F32,
            ));
            value = g.add(value, beta)?;
            if narrow {
                value = g.cast(value, output_dtype)?;
            }
            let zero = g.constant(TensorData::scalar_with_dtype(Scalar::I(0), output_dtype));
            let one = g.constant(TensorData::scalar_with_dtype(Scalar::I(1), output_dtype));
            // Do not use Graph::clip: tinygrad's clamp is ordered strict
            // comparisons plus selects, which retains NaNs and exact ties.
            let below = g.lt(value, zero)?;
            let lower_clamped = g.select(below, zero, value)?;
            let above = g.gt(lower_clamped, one)?;
            g.select(above, one, lower_clamped)?
        }
        "ThresholdedRelu" if ins.len() == 1 => {
            if attrs.keys().any(|name| name != "alpha") {
                return Err(bad("unsupported ThresholdedRelu attribute"));
            }
            // tinygrad accepts the IEEE payload of its Python FLOAT alpha
            // directly: `(X > alpha).where(X, 0)`.  In particular, NaN and
            // infinities are not validation errors.
            let alpha = typed_scalar_f32_attr(&n, "alpha")?.unwrap_or(1.0);
            let x = get(0)?;
            let input_shape = g.shape(x)?.clone();
            let input_dtype = g.dtype(x)?;
            let extent = |shape: &Shape, dtype: DType, what: &str| {
                shape
                    .numel()?
                    .checked_mul(dtype.itemsize())
                    .ok_or_else(|| bad(format!("ThresholdedRelu {what} byte extent overflow")))
                    .map(|_| ())
            };
            extent(&input_shape, input_dtype, "input")?;

            // A weak Python FLOAT comparison resolves at F32 unless the
            // source is F64.  The false Python integer literal is a separate
            // weak value: it preserves every non-Bool X dtype, but Bool plus
            // that literal resolves to tinygrad's default I32.
            let comparison_dtype = if input_dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            };
            let extent = |shape: &Shape, dtype: DType, what: &str| {
                shape
                    .numel()?
                    .checked_mul(dtype.itemsize())
                    .ok_or_else(|| bad(format!("Binarizer {what} byte extent overflow")))
                    .map(|_| ())
            };
            let output_dtype = if input_dtype == DType::Bool {
                DType::I32
            } else {
                input_dtype
            };
            let scalar_shape = Shape::new([]);
            let comparison_shape = input_shape.broadcast_with(&scalar_shape)?;
            if comparison_dtype.promote(comparison_dtype) != comparison_dtype {
                return Err(bad("ThresholdedRelu comparison promotion mismatch"));
            }
            let branch_shape = input_shape.broadcast_with(&scalar_shape)?;
            let output_shape = comparison_shape.broadcast_with(&branch_shape)?;
            if output_dtype.promote(output_dtype) != output_dtype {
                return Err(bad("ThresholdedRelu select promotion mismatch"));
            }
            // Every source value is scalar-broadcast to X, but their logical
            // storage widths differ.  Prove both casts, the Bool predicate,
            // each select branch, and its output before publishing alpha,
            // zero, or the first Cast.
            extent(&comparison_shape, comparison_dtype, "comparison input")?;
            extent(&comparison_shape, DType::Bool, "predicate")?;
            extent(&branch_shape, output_dtype, "true branch")?;
            extent(&branch_shape, output_dtype, "false branch")?;
            extent(&output_shape, output_dtype, "output")?;
            let alpha_value =
                TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), comparison_dtype);
            let zero_value = TensorData::scalar_with_dtype(Scalar::I(0), output_dtype);
            extent(alpha_value.shape(), alpha_value.dtype(), "alpha scalar")?;
            extent(zero_value.shape(), zero_value.dtype(), "zero scalar")?;
            if comparison_shape != input_shape
                || branch_shape != input_shape
                || output_shape != input_shape
                || input_shape.broadcast_with(alpha_value.shape())? != input_shape
                || input_shape.broadcast_with(zero_value.shape())? != input_shape
            {
                return Err(bad("ThresholdedRelu scalar broadcast mismatch"));
            }

            let comparison_x = if input_dtype == comparison_dtype {
                x
            } else {
                g.cast(x, comparison_dtype)?
            };
            let condition = g.gt(comparison_x, g.constant(alpha_value))?;
            let on_true = if input_dtype == output_dtype {
                x
            } else {
                g.cast(x, output_dtype)?
            };
            g.select(condition, on_true, g.constant(zero_value))?
        }
        "Binarizer" if ins.len() == 1 => {
            if attrs.keys().any(|name| name != "threshold") {
                return Err(bad("unsupported Binarizer attribute"));
            }
            // tinygrad's source definition is `(x > threshold).float()`.
            // Its Python FLOAT attribute permits all IEEE payloads.
            let threshold = typed_scalar_f32_attr(&n, "threshold")?.unwrap_or(0.0);
            let x = get(0)?;
            let input_shape = g.shape(x)?.clone();
            let input_dtype = g.dtype(x)?;
            input_shape.numel()?;

            // A weak FLOAT comparison uses F32 unless X is F64; unlike the
            // preceding ThresholdedRelu, the resulting Bool is then always
            // explicitly converted by tinygrad's `.float()` to F32.
            let comparison_dtype = if input_dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            };
            let scalar_shape = Shape::new([]);
            let comparison_shape = input_shape.broadcast_with(&scalar_shape)?;
            // The explicit cast, typed weak threshold, ordered predicate, and
            // final `.float()` all exist in the source graph.  Prove each
            // storage extent before the first cast or constant is published.
            extent(&comparison_shape, comparison_dtype, "comparison input")?;
            extent(&scalar_shape, comparison_dtype, "threshold")?;
            if comparison_dtype.promote(comparison_dtype) != comparison_dtype {
                return Err(bad("Binarizer comparison promotion mismatch"));
            }
            let output_shape = comparison_shape.broadcast_with(&scalar_shape)?;
            extent(&output_shape, DType::Bool, "predicate")?;
            extent(&output_shape, DType::F32, "output")?;
            if DType::F32.promote(DType::F32) != DType::F32 {
                return Err(bad("Binarizer cast promotion mismatch"));
            }

            let comparison_x = if input_dtype == comparison_dtype {
                x
            } else {
                g.cast(x, comparison_dtype)?
            };
            let threshold = g.constant(TensorData::scalar_with_dtype(
                Scalar::F(f64::from(threshold)),
                comparison_dtype,
            ));
            let condition = g.gt(comparison_x, threshold)?;
            g.cast(condition, DType::F32)?
        }
        "PRelu" if ins.len() == 2 && attrs.is_empty() => {
            let x = get(0)?;
            let slope = get(1)?;
            let x_shape = g.shape(x)?.clone();
            let slope_shape = g.shape(slope)?.clone();
            let x_dtype = g.dtype(x)?;
            let slope_dtype = g.dtype(slope)?;
            let extent = |shape: &Shape, dtype: DType, what: &str| {
                shape
                    .numel()?
                    .checked_mul(dtype.itemsize())
                    .ok_or_else(|| bad(format!("PRelu {what} byte extent overflow")))
                    .map(|_| ())
            };
            extent(&x_shape, x_dtype, "input")?;
            extent(&slope_shape, slope_dtype, "slope")?;
            let scaled_shape = x_shape.broadcast_with(&slope_shape)?;
            let output_dtype = prelu_dtype(x_dtype, slope_dtype);
            let scalar_shape = Shape::new([]);
            let condition_shape = x_shape.broadcast_with(&scalar_shape)?;
            let output_shape = condition_shape.broadcast_with(&scaled_shape)?;
            let exceptional_promotion = output_dtype == DType::F32
                && matches!(
                    (x_dtype, slope_dtype),
                    (DType::U64, DType::I64) | (DType::I64, DType::U64)
                );
            let scaled_dtype = if exceptional_promotion {
                DType::F32.promote(DType::F32)
            } else {
                x_dtype.promote(slope_dtype)
            };
            let selected_x_dtype = if exceptional_promotion {
                DType::F32
            } else {
                x_dtype
            };
            if scaled_dtype != output_dtype
                || selected_x_dtype.promote(scaled_dtype) != output_dtype
            {
                return Err(bad("PRelu promotion mismatch"));
            }
            // Validate all source-LUB casts, the strict zero predicate, both
            // select branches, and the final three-way broadcast before the
            // first constant or Cast can be published.
            extent(&x_shape, output_dtype, "input cast")?;
            extent(&slope_shape, output_dtype, "slope cast")?;
            extent(&scaled_shape, output_dtype, "scaled branch")?;
            extent(&condition_shape, x_dtype, "comparison input")?;
            extent(&condition_shape, DType::Bool, "predicate")?;
            extent(&x_shape, output_dtype, "true branch")?;
            extent(&output_shape, output_dtype, "output")?;
            let zero_value = TensorData::scalar_with_dtype(Scalar::I(0), x_dtype);
            extent(zero_value.shape(), zero_value.dtype(), "zero scalar")?;
            if condition_shape != x_shape
                || output_shape != scaled_shape
                || x_shape.broadcast_with(zero_value.shape())? != x_shape
            {
                return Err(bad("PRelu scalar broadcast mismatch"));
            }

            // tinygrad deliberately uses `X > 0`: zero and NaN take the
            // scaled branch, unlike Graph::leaky_relu's `< 0` helper.
            let condition = g.gt(x, g.constant(zero_value))?;
            let (x_value, slope) = if exceptional_promotion {
                (g.cast(x, DType::F32)?, g.cast(slope, DType::F32)?)
            } else {
                (x, slope)
            };
            let scaled = g.mul(x_value, slope)?;
            g.select(condition, x_value, scaled)?
        }
        "Clip" => {
            let x = get(0)?;
            let plan = clip_plan(g, x, &ins, &attrs, values)?;
            let mut value = x;
            if let Some(stage) = plan.min {
                let lhs = if g.dtype(value)? == stage.dtype {
                    value
                } else {
                    g.cast(value, stage.dtype)?
                };
                let bound = if g.dtype(stage.bound)? == stage.dtype {
                    stage.bound
                } else {
                    g.cast(stage.bound, stage.dtype)?
                };
                // Source clamp is `(value < min).where(min, value)`: ties
                // and NaNs retain the prior value rather than using extrema.
                value = g.select(g.lt(lhs, bound)?, bound, lhs)?;
                debug_assert_eq!(
                    g.shape(value).expect("Clip minimum preflighted"),
                    &stage.shape
                );
                debug_assert_eq!(
                    g.dtype(value).expect("Clip minimum dtype preflighted"),
                    stage.dtype
                );
            }
            if let Some(stage) = plan.max {
                let lhs = if g.dtype(value)? == stage.dtype {
                    value
                } else {
                    g.cast(value, stage.dtype)?
                };
                let bound = if g.dtype(stage.bound)? == stage.dtype {
                    stage.bound
                } else {
                    g.cast(stage.bound, stage.dtype)?
                };
                // Source then applies `(value > max).where(max, value)`.
                value = g.select(g.gt(lhs, bound)?, bound, lhs)?;
                debug_assert_eq!(
                    g.shape(value).expect("Clip maximum preflighted"),
                    &stage.shape
                );
                debug_assert_eq!(
                    g.dtype(value).expect("Clip maximum dtype preflighted"),
                    stage.dtype
                );
            }
            debug_assert_eq!(
                g.shape(value).expect("Clip output preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(value).expect("Clip output dtype preflighted"),
                plan.output_dtype
            );
            value
        }
        "Dropout" if (1..=3).contains(&ins.len()) => {
            let x = get(0)?;
            let plan = dropout_plan(g, x, &ins, &n, &attrs, constants)?;
            debug_assert_eq!(g.shape(x).expect("Dropout shape preflighted"), &plan.shape);
            debug_assert_eq!(g.dtype(x).expect("Dropout dtype preflighted"), plan.dtype);
            x
        }
        "Shape" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "start" && x != "end") {
                return Err(bad("unsupported Shape attribute"));
            }
            let dims = g.shape(get(0)?)?.dims();
            let rank = i64::try_from(dims.len()).map_err(|_| bad("Shape rank overflow"))?;
            // tinygrad's ONNX parser first resolves AttributeProto through
            // its declared AttributeType.  Do not let a raw varint-shaped
            // payload from another form masquerade as Shape's INT endpoint.
            let start = strict_typed_scalar_i64_attr(&n, "start")?.unwrap_or(0);
            let end = strict_typed_scalar_i64_attr(&n, "end")?.unwrap_or(rank);
            // tinygrad delegates these attributes directly to Python tuple
            // slicing: signed endpoints clamp to the closed rank interval and
            // a reversed interval produces an empty shape tensor.
            let normalize = |x: i64| -> Result<usize> {
                let x = if x < 0 {
                    x.saturating_add(rank).max(0)
                } else {
                    x.min(rank)
                };
                usize::try_from(x).map_err(|_| bad("Shape endpoint overflow"))
            };
            let (start, end) = (normalize(start)?, normalize(end)?);
            let dims = if start < end { &dims[start..end] } else { &[] };
            let values = dims
                .iter()
                .map(|&dim| {
                    i64::try_from(dim)
                        .map(Scalar::I)
                        .map_err(|_| bad("Shape dimension exceeds I64"))
                })
                .collect::<Result<Vec<_>>>()?;
            let data = TensorData::from_scalars([values.len()], DType::I64, values)?;
            constants.insert(outs[0].to_owned(), data.clone());
            g.constant(data)
        }
        "Size" if ins.len() == 1 && attrs.is_empty() => {
            let input = get(0)?;
            let shape = g.shape(input)?;
            let dtype = g.dtype(input)?;
            let numel = shape.numel()?;
            numel
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| bad("Size input byte extent overflow"))?;
            let numel = i64::try_from(numel).map_err(|_| bad("Size exceeds I64"))?;
            let data = TensorData::from_scalars([], DType::I64, [Scalar::I(numel)])?;
            constants.insert(outs[0].to_owned(), data.clone());
            g.constant(data)
        }
        "Expand" if ins.len() == 2 && attrs.is_empty() => {
            let x = get(0)?;
            let shape_data = constants
                .get(ins[1])
                .ok_or_else(|| bad("Expand shape must be a constant initializer"))?;
            if shape_data.dtype() != DType::I64 || shape_data.shape().rank() != 1 {
                return Err(bad("Expand shape must be a rank-one I64 constant"));
            }
            let shape = const_i64(constants, ins[1])?
                .into_iter()
                .map(|x| usize::try_from(x).map_err(|_| bad("Expand shape must be nonnegative")))
                .collect::<Result<Vec<_>>>()?;
            // tinygrad's ONNX adapter expands to the broadcast of the input
            // and requested shape. In particular, a shorter requested shape
            // may be a no-op on leading input dimensions.
            let shape = g.shape(x)?.broadcast_with(&Shape::new(shape))?;
            shape.numel()?;
            g.expand(x, shape)?
        }
        "Tile" if ins.len() == 2 && attrs.is_empty() => {
            let x = get(0)?;
            let repeats_data = constants
                .get(ins[1])
                .ok_or_else(|| bad("Tile repeats must be a constant initializer"))?;
            if repeats_data.dtype() != DType::I64 || repeats_data.shape().rank() != 1 {
                return Err(bad("Tile repeats must be a rank-one I64 constant"));
            }
            let repeats = const_i64(constants, ins[1])?;
            let input_shape = g.shape(x)?.clone();
            if repeats.len() != input_shape.rank() || repeats.iter().any(|&x| x < 0) {
                return Err(bad("Tile repeats must be nonnegative and match rank"));
            }
            if repeats.is_empty() {
                // tinygrad's scalar `repeat(())` is an identity. Graph::tile
                // deliberately rejects an empty public repeat list, so retain
                // that API boundary while lowering this static ONNX scalar.
                x
            } else {
                let repeats = repeats
                    .into_iter()
                    .map(|repeat| {
                        isize::try_from(repeat).map_err(|_| bad("Tile repeat extent overflow"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                g.tile(x, &repeats)?
            }
        }
        "Gather" if ins.len() == 2 => {
            if attrs.keys().any(|x| x != "axis") {
                return Err(bad("unsupported Gather attribute"));
            }
            let x = get(0)?;
            let input_shape = g.shape(x)?.clone();
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let rank = input_shape.rank();
            let axis = axes_usize(&[axis], rank)?[0];
            let name = ins[1];
            let data = constants
                .get(name)
                .ok_or_else(|| bad("Gather indices must be constant"))?;
            if !matches!(data.dtype(), DType::I32 | DType::I64) {
                return Err(bad("Gather indices must be constant I32/I64"));
            }
            if data.shape().rank() == 0 {
                // tinygrad's constant Gather fast path accepts scalar indices
                // and normalizes a negative value against the selected axis.
                // Materialize the equivalent fixed-rank index only after all
                // bounds and resulting-view extents have been checked.
                let dim = input_shape.dims()[axis];
                let dim_i64 =
                    i64::try_from(dim).map_err(|_| bad("Gather scalar axis extent exceeds I64"))?;
                let raw = data.scalar_at(0).as_i64();
                let index = if raw < 0 {
                    raw.checked_add(dim_i64)
                        .ok_or_else(|| bad("Gather scalar index is out of bounds"))?
                } else {
                    raw
                };
                if index < 0 || usize::try_from(index).ok().filter(|&i| i < dim).is_none() {
                    return Err(bad("Gather scalar index is out of bounds"));
                }
                let mut index_dims = input_shape.dims().to_vec();
                index_dims[axis] = 1;
                let index_shape = Shape::new(index_dims);
                let output_shape = Shape::new(
                    input_shape
                        .dims()
                        .iter()
                        .enumerate()
                        .filter_map(|(dim, &extent)| (dim != axis).then_some(extent))
                        .collect::<Vec<_>>(),
                );
                let index_len = index_shape.numel()?;
                if output_shape.numel()? != index_len {
                    return Err(bad("Gather scalar output extent mismatch"));
                }
                let index = TensorData::from_scalars(
                    index_shape,
                    data.dtype(),
                    std::iter::repeat(Scalar::I(index)).take(index_len),
                )?;
                let index = g.constant(index);
                let gathered = g.gather(x, index, axis)?;
                g.reshape(gathered, output_shape)?
            } else {
                if data.shape() != &input_shape {
                    return Err(bad("Gather requires same-rank constant I32/I64 indices"));
                }
                if (0..data.len()).any(|i| data.scalar_at(i).as_i64() < 0) {
                    return Err(bad("Gather negative indices are unsupported"));
                }
                g.gather(x, get(1)?, axis)?
            }
        }
        "Slice" if (3..=5).contains(&ins.len()) && attrs.is_empty() => {
            let x = get(0)?;
            let plan = slice_plan(g, x, &ins, &attrs, constants)?;
            let output = g.stride(x, plan.slices)?;
            debug_assert_eq!(
                g.shape(output).expect("Slice shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Slice dtype preflighted"),
                plan.dtype
            );
            output
        }
        "Pad" if (2..=3).contains(&ins.len()) => {
            if attrs.keys().any(|x| x != "mode") {
                return Err(bad("unsupported Pad attribute"));
            }
            if attrs.get("mode").map(Vec::as_slice).unwrap_or(b"constant") != b"constant" {
                return Err(bad("only constant Pad mode is supported"));
            }
            let x = get(0)?;
            let rank = g.shape(x)?.rank();
            let pads_data = constants
                .get(ins[1])
                .ok_or_else(|| bad("Pad pads must be a constant initializer"))?;
            if pads_data.dtype() != DType::I64 || pads_data.shape().rank() != 1 {
                return Err(bad("Pad pads must be a rank-one I64 constant"));
            }
            let pads = const_i64(constants, ins[1])?;
            if pads.len() != 2 * rank {
                return Err(bad("Pad pads must contain begin/end values for every axis"));
            }
            let fill = if ins.len() == 3 && !ins[2].is_empty() {
                let value = constants
                    .get(ins[2])
                    .ok_or_else(|| bad("Pad constant_value must be constant"))?;
                if value.len() != 1 || value.dtype() != g.dtype(x)? {
                    return Err(bad("Pad constant_value must be a same-dtype scalar"));
                }
                value.scalar_at(0)
            } else {
                Scalar::I(0)
            };
            let padding = (0..rank)
                .map(|i| (pads[i], pads[rank + i]))
                .collect::<Vec<_>>();
            g.pad_signed(x, padding, fill)?
        }
        "ConstantOfShape" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "value") {
                return Err(bad("unsupported ConstantOfShape attribute"));
            }
            let dims_data = constants
                .get(ins[0])
                .ok_or_else(|| bad("ConstantOfShape shape must be a constant initializer"))?;
            if dims_data.dtype() != DType::I64 || dims_data.shape().rank() != 1 {
                return Err(bad("ConstantOfShape shape must be a rank-one I64 constant"));
            }
            let dims = const_i64(constants, ins[0])?
                .into_iter()
                .map(|x| {
                    usize::try_from(x)
                        .map_err(|_| bad("ConstantOfShape dimensions must be nonnegative"))
                })
                .collect::<Result<Vec<_>>>()?;
            let shape = Shape::new(dims);
            shape.numel()?;
            let (value, dtype) = match attrs.get("value") {
                Some(bytes) => {
                    let value = tensor_data(Msg::new(bytes))?;
                    if value.len() != 1 {
                        return Err(bad("ConstantOfShape value must contain one element"));
                    }
                    // tinygrad obtains the result by expanding this tensor.
                    // Its explicit [0] special case is the sole empty shape
                    // that bypasses that broadcast check.
                    if shape.dims() != &[0]
                        && value.shape().broadcast_with(&shape).as_ref() != Ok(&shape)
                    {
                        return Err(bad(
                            "ConstantOfShape value shape cannot broadcast to output shape",
                        ));
                    }
                    (value.scalar_at(0), value.dtype())
                }
                None => (Scalar::F(0.0), DType::F32),
            };
            g.full_with_dtype(shape, value, dtype)?
        }
        "ReduceMin" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_min_plan(g, x, &ins, &attrs, constants)?;
            let output_shape = plan.reduction.output_shape.clone();
            let axes = plan.reduction.axes.clone();
            let keepdims = plan.reduction.keepdims;
            let output = match plan.lowering {
                ReduceMinLowering::Identity => x,
                ReduceMinLowering::Empty => g.constant(TensorData::zeros_with_dtype(
                    output_shape.clone(),
                    plan.dtype,
                )?),
                ReduceMinLowering::IdentityValue => g.constant(TensorData::full_with_dtype(
                    output_shape.clone(),
                    min_identity(plan.dtype),
                    plan.dtype,
                )?),
                ReduceMinLowering::Reduce => g.reduce(x, ReduceKind::Min, Some(axes), keepdims)?,
            };
            debug_assert_eq!(
                g.shape(output).expect("ReduceMin shape preflighted"),
                &output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("ReduceMin dtype preflighted"),
                plan.dtype
            );
            output
        }
        "ReduceMax" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_max_plan(g, x, &ins, &attrs, constants)?;
            let output_shape = plan.reduction.output_shape.clone();
            let axes = plan.reduction.axes.clone();
            let keepdims = plan.reduction.keepdims;
            let output = match plan.lowering {
                ReduceMaxLowering::Identity => x,
                ReduceMaxLowering::Empty => g.constant(TensorData::zeros_with_dtype(
                    output_shape.clone(),
                    plan.dtype,
                )?),
                ReduceMaxLowering::IdentityValue => g.constant(TensorData::full_with_dtype(
                    output_shape.clone(),
                    max_identity(plan.dtype),
                    plan.dtype,
                )?),
                ReduceMaxLowering::Reduce => g.reduce(x, ReduceKind::Max, Some(axes), keepdims)?,
            };
            debug_assert_eq!(
                g.shape(output).expect("ReduceMax shape preflighted"),
                &output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("ReduceMax dtype preflighted"),
                plan.dtype
            );
            output
        }
        "ReduceProd" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_prod_plan(g, x, &ins, &attrs, constants)?;
            let output = if plan.reduction.noop {
                // Tensor.prod's explicit same-dtype cast is an exact identity
                // for the importer-supported type inventory.
                x
            } else {
                g.reduce_with_dtypes(
                    x,
                    ReduceKind::Product,
                    Some(plan.reduction.axes),
                    plan.reduction.keepdims,
                    plan.dtypes,
                )?
            };
            debug_assert_eq!(
                g.shape(output).expect("ReduceProd shape preflighted"),
                &plan.reduction.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("ReduceProd dtype preflighted"),
                plan.dtypes.output
            );
            output
        }
        "ReduceSum" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_sum_plan(g, x, &ins, &attrs, constants)?;
            let output = if plan.reduction.noop {
                // A source empty-axis Sum has no movement, but it still
                // applies Tensor.sum's accumulator and narrow-output casts.
                let accumulator = if g.dtype(x)? == plan.reduction.sum_dtypes.accumulator {
                    x
                } else {
                    g.cast(x, plan.reduction.sum_dtypes.accumulator)?
                };
                if plan.reduction.sum_dtypes.accumulator == plan.reduction.sum_dtypes.output {
                    accumulator
                } else {
                    g.cast(accumulator, plan.reduction.sum_dtypes.output)?
                }
            } else {
                g.reduce_with_dtypes(
                    x,
                    ReduceKind::Sum,
                    Some(plan.reduction.axes),
                    plan.reduction.keepdims,
                    plan.reduction.sum_dtypes,
                )?
            };
            debug_assert_eq!(
                g.shape(output).expect("ReduceSum shape preflighted"),
                &plan.reduction.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("ReduceSum dtype preflighted"),
                plan.reduction.sum_dtypes.output
            );
            output
        }
        "ReduceMean" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_mean_plan(g, x, &ins, &attrs, constants)?;
            // An empty axis list leaves the shape unchanged, but Tensor.mean
            // still applies its explicit accumulator cast and true division
            // by one. Keep its nonfloat promotion and autograd boundary.
            let sum = if plan.reduction.noop {
                if g.dtype(x)? == plan.sum_dtypes.accumulator {
                    x
                } else {
                    g.cast(x, plan.sum_dtypes.accumulator)?
                }
            } else {
                g.reduce_with_dtypes(
                    x,
                    ReduceKind::Sum,
                    Some(plan.reduction.axes),
                    plan.reduction.keepdims,
                    plan.sum_dtypes,
                )?
            };
            let division_input = if g.dtype(sum)? == plan.division_dtype {
                sum
            } else {
                g.cast(sum, plan.division_dtype)?
            };
            // Tensor.div is reciprocal then multiplication, not Graph::div.
            let divisor = g.constant(plan.divisor);
            let mean = g.mul(division_input, g.reciprocal(divisor)?)?;
            let output = if plan.division_dtype == plan.output_dtype {
                mean
            } else {
                g.cast(mean, plan.output_dtype)?
            };
            debug_assert_eq!(
                g.shape(output).expect("ReduceMean shape preflighted"),
                &plan.reduction.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("ReduceMean dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "ReduceSumSquare" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_plan(g, x, &ins, &attrs, constants)?;
            // tinygrad defines square as `x * x`, not as a distinct unary
            // runtime operation. Keeping the binary form also retains the
            // existing generic renderer contract.
            let squared = g.mul(x, x)?;
            if plan.noop {
                squared
            } else {
                g.reduce_with_dtypes(
                    squared,
                    ReduceKind::Sum,
                    Some(plan.axes),
                    plan.keepdims,
                    plan.sum_dtypes,
                )?
            }
        }
        "ReduceL1" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_plan(g, x, &ins, &attrs, constants)?;
            // tinygrad defines abs as `x * x.sign()`. Do not use UnaryOp::Abs:
            // its hardware-style implementation clears negative zero, whereas
            // the source composition preserves it (including the noop path).
            let sign = g.sign(x)?;
            let absolute = g.mul(x, sign)?;
            if plan.noop {
                absolute
            } else {
                g.reduce_with_dtypes(
                    absolute,
                    ReduceKind::Sum,
                    Some(plan.axes),
                    plan.keepdims,
                    plan.sum_dtypes,
                )?
            }
        }
        "ReduceL2" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_l2_plan(g, x, &ins, &attrs, constants)?;
            let work = if plan.work_dtype == plan.source_dtype {
                x
            } else {
                g.cast(x, plan.work_dtype)?
            };
            // tinygrad's `square()` is exactly `work * work`.
            let squared = g.mul(work, work)?;
            let sum = if plan.reduction.noop {
                squared
            } else {
                g.reduce_with_dtypes(
                    squared,
                    ReduceKind::Sum,
                    Some(plan.reduction.axes),
                    plan.reduction.keepdims,
                    plan.sum_dtypes,
                )?
            };
            let root = g.sqrt(sum)?;
            if plan.sqrt_dtype == plan.source_dtype {
                root
            } else {
                g.cast(root, plan.source_dtype)?
            }
        }
        "ReduceLogSum" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_log_sum_plan(g, x, &ins, &attrs, constants)?;
            let sum = if plan.reduction.noop {
                x
            } else {
                g.reduce_with_dtypes(
                    x,
                    ReduceKind::Sum,
                    Some(plan.reduction.axes),
                    plan.reduction.keepdims,
                    plan.reduction.sum_dtypes,
                )?
            };
            // tinygrad's Tensor.log is `log2() * math.log(2)`, not ln().
            let log2 = g.log2(sum)?;
            let ln2 = g.constant(plan.ln2);
            g.mul(log2, ln2)?
        }
        "ReduceLogSumExp" if (1..=2).contains(&ins.len()) => {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_log_sum_exp_plan(g, x, &ins, &attrs, constants)?;
            // The ONNX adapter deliberately uses the direct dispatcher form,
            // not the stable Graph::logsumexp max-shift helper.
            let exponentials = g.exp(x)?;
            debug_assert_eq!(g.dtype(exponentials).ok(), Some(plan.exp_dtype));
            let sum = if plan.reduction.noop {
                exponentials
            } else {
                g.reduce_with_dtypes(
                    exponentials,
                    ReduceKind::Sum,
                    Some(plan.reduction.axes),
                    plan.reduction.keepdims,
                    plan.sum_dtypes,
                )?
            };
            let log2 = g.log2(sum)?;
            let ln2 = g.constant(plan.ln2);
            g.mul(log2, ln2)?
        }
        "ArgMax" if ins.len() == 1 => {
            let x = get(0)?;
            let plan = argmax_plan(g, x, &n, &attrs)?;
            if let Some(data) = plan.empty_axis_result {
                g.constant(data)
            } else {
                // The source's truthy last-index form flips before argmax;
                // sample that same flipped first lane for its NaN sentinel.
                let source = if plan.select_last {
                    g.flip(x, [plan.axis])?
                } else {
                    x
                };
                let indices = g.argmax(source, Some(plan.axis), plan.keepdims)?;
                let first = g.shrink(source, plan.first_bounds)?;
                let first = if plan.keepdims {
                    first
                } else {
                    g.squeeze(first, Some(plan.axis))?
                };
                let leading_nan = g.isnan(first)?;
                let sentinel = g.constant(plan.sentinel);
                let selected = g.select(leading_nan, sentinel, indices)?;
                let selected = if let Some(offset) = plan.last_offset {
                    let offset = g.constant(offset);
                    g.sub(offset, selected)?
                } else {
                    selected
                };
                g.cast(selected, DType::I64)?
            }
        }
        "ArgMin" if ins.len() == 1 => {
            let x = get(0)?;
            let plan = argmin_plan(g, x, &n, &attrs)?;
            if let Some(data) = plan.empty_axis_result {
                g.constant(data)
            } else {
                // Keep literal source order: ArgMin first negates its input,
                // then invokes the complete ArgMax adapter on that result.
                let negated = g.neg(x)?;
                let source = if plan.select_last {
                    g.flip(negated, [plan.axis])?
                } else {
                    negated
                };
                let indices = g.argmax(source, Some(plan.axis), plan.keepdims)?;
                let first = g.shrink(source, plan.first_bounds)?;
                let first = if plan.keepdims {
                    first
                } else {
                    g.squeeze(first, Some(plan.axis))?
                };
                let leading_nan = g.isnan(first)?;
                let sentinel = g.constant(plan.sentinel);
                let selected = g.select(leading_nan, sentinel, indices)?;
                let selected = if let Some(offset) = plan.last_offset {
                    let offset = g.constant(offset);
                    g.sub(offset, selected)?
                } else {
                    selected
                };
                g.cast(selected, DType::I64)?
            }
        }
        "BatchNormalization" if ins.len() == 5 => {
            let inputs = [get(0)?, get(1)?, get(2)?, get(3)?, get(4)?];
            let plan = batch_norm_plan(g, inputs, &ins, &n, &attrs)?;
            let cast = |g: &mut Graph, input: NodeId, dtype: DType| -> Result<NodeId> {
                if g.dtype(input)? == dtype {
                    Ok(input)
                } else {
                    g.cast(input, dtype)
                }
            };
            // Tensor.batchnorm is literal: center, apply scale, then apply
            // the separately-rounded inverse standard deviation, then bias.
            let mean = g.reshape(plan.mean, plan.channel_shape.clone())?;
            let input = cast(g, plan.input, plan.centered_dtype)?;
            let mean = cast(g, mean, plan.centered_dtype)?;
            let centered = g.sub(input, mean)?;
            let scale = g.reshape(plan.scale, plan.channel_shape.clone())?;
            let centered = cast(g, centered, plan.scaled_dtype)?;
            let scale = cast(g, scale, plan.scaled_dtype)?;
            let scaled = g.mul(centered, scale)?;
            let epsilon = g.constant(plan.epsilon);
            let variance = cast(g, plan.variance, plan.variance_dtype)?;
            let epsilon = cast(g, epsilon, plan.variance_dtype)?;
            let variance = g.add(variance, epsilon)?;
            // tinygrad spells rsqrt as sqrt followed by reciprocal, so narrow
            // storage rounds between these two existing primitive nodes.
            let sqrt = g.sqrt(variance)?;
            let invstd = g.reciprocal(sqrt)?;
            let invstd = if plan.variance_is_vector {
                g.reshape(invstd, plan.channel_shape.clone())?
            } else {
                invstd
            };
            let scaled = cast(g, scaled, plan.normalized_dtype)?;
            let invstd = cast(g, invstd, plan.normalized_dtype)?;
            let normalized = g.mul(scaled, invstd)?;
            let bias = g.reshape(plan.bias, plan.channel_shape.clone())?;
            let normalized = cast(g, normalized, plan.output_dtype)?;
            let bias = cast(g, bias, plan.output_dtype)?;
            let output = g.add(normalized, bias)?;
            debug_assert_eq!(
                g.shape(output)
                    .expect("BatchNormalization shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output)
                    .expect("BatchNormalization dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "LayerNormalization" if (2..=3).contains(&ins.len()) => {
            let input = get(0)?;
            let scale = get(1)?;
            let bias = ins
                .get(2)
                .filter(|name| !name.is_empty())
                .map(|_| get(2))
                .transpose()?;
            let plan = layer_normalization_plan(g, input, scale, bias, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                }
            } else {
                let count = g.constant(plan.count);
                let epsilon = g.constant(plan.epsilon);
                // Match the source literally: first cast to F32, use typed
                // F32 sums/divisions for both moments, then restore X before
                // the live scale/bias promotion boundary.
                let x32 = g.cast(input, DType::F32)?;
                let mean_sum = g.reduce_with_dtypes(
                    x32,
                    ReduceKind::Sum,
                    Some(plan.axes.clone()),
                    true,
                    plan.sum_dtypes,
                )?;
                let mean = g.mul(mean_sum, g.reciprocal(count)?)?;
                let centered = g.sub(x32, mean)?;
                let variance_sum = g.reduce_with_dtypes(
                    g.mul(centered, centered)?,
                    ReduceKind::Sum,
                    Some(plan.axes),
                    true,
                    plan.sum_dtypes,
                )?;
                let variance = g.mul(variance_sum, g.reciprocal(count)?)?;
                let inv_std_dev = g.rsqrt(g.add(variance, epsilon)?)?;
                let normalized = g.mul(centered, inv_std_dev)?;
                let restored = g.cast(normalized, plan.input_dtype)?;
                let output = g.mul(restored, scale)?;
                let output = if let Some(bias) = bias {
                    g.add(output, bias)?
                } else {
                    output
                };
                debug_assert_eq!(
                    g.shape(output)
                        .expect("LayerNormalization shape preflighted"),
                    &plan.shape
                );
                debug_assert_eq!(
                    g.dtype(output)
                        .expect("LayerNormalization dtype preflighted"),
                    plan.output_dtype
                );
                output
            }
        }
        "RMSNormalization" if ins.len() == 2 => {
            let input = get(0)?;
            let scale = get(1)?;
            let plan = rms_normalization_plan(g, input, scale, &n, &attrs)?;
            let count = g.constant(plan.count);
            let epsilon = g.constant(plan.epsilon);
            let x32 = g.cast(input, DType::F32)?;
            let squares = g.mul(x32, x32)?;
            let sum = g.reduce_with_dtypes(
                squares,
                ReduceKind::Sum,
                Some(plan.axes),
                true,
                ReductionDType::new(DType::F32, DType::F32),
            )?;
            let mean = g.mul(sum, g.reciprocal(count)?)?;
            let norm = g.rsqrt(g.add(mean, epsilon)?)?;
            let normalized = g.mul(input, norm)?;
            let output = g.mul(normalized, scale)?;
            debug_assert_eq!(
                g.shape(output).expect("RMSNormalization shape preflighted"),
                &plan.shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("RMSNormalization dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "MeanVarianceNormalization" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = mean_variance_normalization_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.work_dtype {
                    input
                } else {
                    g.cast(input, plan.work_dtype)?
                }
            } else {
                let count = g.constant(plan.count);
                let epsilon = g.constant(plan.epsilon);
                let sum_dtypes = ReductionDType::new(plan.sum_dtype, plan.sum_dtype);
                // The source computes X.mean twice: once for the numerator
                // and again inside std(correction=0). Keep their narrowing
                // points separate rather than reusing a generic variance.
                let mean_sum = g.reduce_with_dtypes(
                    input,
                    ReduceKind::Sum,
                    Some(plan.axes.clone()),
                    true,
                    sum_dtypes,
                )?;
                let mean = g.cast(g.mul(mean_sum, g.reciprocal(count)?)?, plan.work_dtype)?;
                let numerator = g.sub(input, mean)?;
                let variance_mean_sum = g.reduce_with_dtypes(
                    input,
                    ReduceKind::Sum,
                    Some(plan.axes.clone()),
                    true,
                    sum_dtypes,
                )?;
                let variance_mean = g.cast(
                    g.mul(variance_mean_sum, g.reciprocal(count)?)?,
                    plan.work_dtype,
                )?;
                let deviations = g.sub(input, variance_mean)?;
                let squares = g.square(deviations)?;
                // `Tensor.var` explicitly recasts the storage-width squares
                // to sum_acc_dtype(original X), including its integer paths.
                let squares = g.cast(squares, plan.sum_dtype)?;
                let variance_sum = g.reduce_with_dtypes(
                    squares,
                    ReduceKind::Sum,
                    Some(plan.axes),
                    true,
                    sum_dtypes,
                )?;
                let variance =
                    g.cast(g.mul(variance_sum, g.reciprocal(count)?)?, plan.work_dtype)?;
                let denominator = g.add(g.sqrt(variance)?, epsilon)?;
                let output = g.mul(numerator, g.reciprocal(denominator)?)?;
                debug_assert_eq!(
                    g.shape(output)
                        .expect("MeanVarianceNormalization shape preflighted"),
                    &plan.shape
                );
                debug_assert_eq!(
                    g.dtype(output)
                        .expect("MeanVarianceNormalization dtype preflighted"),
                    plan.work_dtype
                );
                output
            }
        }
        "LpNormalization" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = lp_normalization_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype {
                    input
                } else {
                    g.cast(input, plan.output_dtype)?
                }
            } else {
                let base = if plan.l1 {
                    // Tensor.abs is `x * sign(x)`, retaining -0 and wrapping
                    // signed minima before the source Sum contract.
                    g.mul(input, g.sign(input)?)?
                } else {
                    // Tensor.square is literally x*x, not UnaryOp::Square.
                    g.mul(input, input)?
                };
                let summed = g.reduce_with_dtypes(
                    base,
                    ReduceKind::Sum,
                    Some(plan.axes),
                    true,
                    plan.sum_dtypes,
                )?;
                let denominator = if plan.l1 { summed } else { g.sqrt(summed)? };
                debug_assert_eq!(
                    g.dtype(denominator)
                        .expect("LpNormalization denominator preflighted"),
                    plan.denominator_dtype
                );
                let output = g.mul(input, g.reciprocal(denominator)?)?;
                debug_assert_eq!(
                    g.shape(output).expect("LpNormalization shape preflighted"),
                    &plan.shape
                );
                debug_assert_eq!(
                    g.dtype(output).expect("LpNormalization dtype preflighted"),
                    plan.output_dtype
                );
                output
            }
        }
        "Einsum" if !ins.is_empty() => {
            let inputs = (0..ins.len()).map(get).collect::<Result<Vec<_>>>()?;
            let plan = einsum_plan(g, &inputs, &n, &attrs)?;
            let output = g.einsum(&plan.equation, &plan.inputs)?;
            debug_assert_eq!(
                g.shape(output).expect("Einsum shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("Einsum dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "GlobalAveragePool" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            let plan = global_average_pool_plan(g, x)?;
            // Do not special-case `axes=[]` as an identity.  The checked-in
            // handler calls `Tensor.mean`, which still commits the source to
            // `sum_acc_dtype`, reduces the empty tuple, and divides by the
            // weak scalar one before its final output cast.
            let summed = g.reduce_with_dtypes(
                x,
                ReduceKind::Sum,
                Some(plan.axes),
                true,
                ReductionDType::new(plan.sum_dtypes.accumulator, plan.sum_dtypes.accumulator),
            )?;
            let summed = if plan.work_dtype == plan.sum_dtypes.accumulator {
                summed
            } else {
                g.cast(summed, plan.work_dtype)?
            };
            let average = g.div(summed, g.constant(plan.divisor))?;
            let output = if plan.output_dtype == plan.work_dtype {
                average
            } else {
                g.cast(average, plan.output_dtype)?
            };
            debug_assert_eq!(
                g.shape(output)
                    .expect("GlobalAveragePool shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output)
                    .expect("GlobalAveragePool dtype preflighted"),
                plan.output_dtype
            );
            output
        }
        "GlobalMaxPool" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            let plan = global_max_pool_plan(g, x)?;
            let output = if plan.axes.is_empty() {
                // tinygrad's max over an empty axis tuple is an identity.
                x
            } else if plan.empty_spatial && plan.output_numel != 0 {
                // tinygrad lowers a zero-sized MAX reduction to dtype.min.
                // Do not call Graph::reduce here: it correctly fail-closes
                // generic empty extrema, whereas this source form has an
                // explicit identity contract.
                g.full_with_dtype(plan.output_shape, plan.max_identity, plan.dtype)?
            } else {
                // A zero retained N/C extent stays an empty result rather
                // than becoming a populated identity tensor.
                g.reduce(x, ReduceKind::Max, Some(plan.axes), true)?
            };
            debug_assert_eq!(
                g.shape(output).expect("GlobalMaxPool shape preflighted"),
                &plan.output_shape
            );
            debug_assert_eq!(
                g.dtype(output).expect("GlobalMaxPool dtype preflighted"),
                plan.dtype
            );
            output
        }
        "CumSum" if ins.len() == 2 => {
            let x = get(0)?;
            let plan = cumsum_plan(g, x, &ins, &n, &attrs, constants)?;
            let reversed = if plan.reverse {
                g.flip(x, vec![plan.axis])?
            } else {
                x
            };
            let shifted = if plan.exclusive {
                let padded = g.pad(
                    reversed,
                    plan.padding.expect("exclusive CumSum padding preflighted"),
                    plan.fill,
                )?;
                g.shrink(
                    padded,
                    plan.shrink.expect("exclusive CumSum shrink preflighted"),
                )?
            } else {
                reversed
            };
            let summed = g.cumsum(shifted, plan.axis)?;
            if plan.reverse {
                g.flip(summed, vec![plan.axis])?
            } else {
                summed
            }
        }
        "Trilu" if (1..=2).contains(&ins.len()) => {
            let x = get(0)?;
            match trilu_plan(g, x, &ins, &n, &attrs, constants)? {
                TriluLowering::Identity => x,
                TriluLowering::Zero(data) => g.constant(data),
                TriluLowering::Upper(diagonal) => g.triu(x, diagonal)?,
                TriluLowering::Lower(diagonal) => g.tril(x, diagonal)?,
            }
        }
        "MaxPool" if ins.len() == 1 => {
            let x = get(0)?;
            if g.shape(x)?.rank() != 4 || !g.dtype(x)?.is_float() {
                return Err(bad("MaxPool requires a rank-4 float NCHW tensor"));
            }
            let options = onnx_pool_options(&attrs, true, g.shape(x)?.dims())?;
            g.max_pool(x, options)?
        }
        "AveragePool" if ins.len() == 1 => {
            let x = get(0)?;
            if g.shape(x)?.rank() != 4 || !g.dtype(x)?.is_float() {
                return Err(bad("AveragePool requires a rank-4 float NCHW tensor"));
            }
            let options = onnx_pool_options(&attrs, false, g.shape(x)?.dims())?;
            g.avg_pool(x, options)?
        }
        "DequantizeLinear" if (2..=3).contains(&ins.len()) => {
            let inputs = (0..ins.len()).map(|i| get(i)).collect::<Result<Vec<_>>>()?;
            let plan = dequantize_linear_plan(g, &inputs, &ins, &n, &attrs)?;
            let DequantizeLinearPlan {
                x: plan_x,
                scale: plan_scale,
                zero: plan_zero,
                scale_plan,
                zero_plan,
                subtract_dtype,
                multiply_dtype,
                output_dtype,
                shape,
            } = plan;
            let cast = |g: &mut Graph, id: NodeId, dtype: DType| -> Result<NodeId> {
                if g.dtype(id)? == dtype {
                    Ok(id)
                } else {
                    g.cast(id, dtype)
                }
            };
            let scale = emit_quant_parameter(g, plan_scale, scale_plan)?;
            let zero = match (plan_zero, zero_plan) {
                (Some(z), Some(parameter_plan)) => emit_quant_parameter(g, z, parameter_plan)?,
                (None, None) => g.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::I32)),
                _ => unreachable!("DequantizeLinear plan pairs zero with its preparation"),
            };
            let x = cast(g, plan_x, DType::I32)?;
            let x = cast(g, x, subtract_dtype)?;
            let zero = cast(g, zero, subtract_dtype)?;
            let difference = g.sub(x, zero)?;
            let difference = cast(g, difference, multiply_dtype)?;
            let scale = cast(g, scale, multiply_dtype)?;
            let output = g.mul(difference, scale)?;
            let output = cast(g, output, output_dtype)?;
            debug_assert_eq!(
                g.shape(output).expect("DequantizeLinear shape preflighted"),
                &shape
            );
            output
        }
        "Conv" if ins.len() == 2 || ins.len() == 3 => {
            if attrs.keys().any(|name| {
                !matches!(
                    name.as_str(),
                    "auto_pad" | "dilations" | "group" | "kernel_shape" | "pads" | "strides"
                )
            }) {
                return Err(bad("unsupported Conv attribute"));
            }
            let x = get(0)?;
            let w = get(1)?;
            if attrs.contains_key("kernel_shape") {
                let kernel = conv_pair(&attrs, "kernel_shape", [0, 0], false)?;
                let weight = g.shape(w)?.dims();
                if weight.len() != 4 || weight[2..] != kernel {
                    return Err(bad(
                        "Conv kernel_shape must match weight spatial dimensions",
                    ));
                }
            }
            let strides = conv_pair(&attrs, "strides", [1, 1], false)?;
            let dilations = conv_pair(&attrs, "dilations", [1, 1], false)?;
            let groups = attrs
                .get("group")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            let groups = usize::try_from(groups)
                .ok()
                .filter(|&x| x != 0)
                .ok_or_else(|| bad("Conv group must be positive"))?;
            let explicit_pads = attrs.contains_key("pads");
            let pads = conv_pads(&attrs)?;
            let auto_pad = attrs
                .get("auto_pad")
                .map(Vec::as_slice)
                .unwrap_or(b"NOTSET");
            if auto_pad != b"NOTSET" && explicit_pads {
                return Err(bad("Conv pads conflicts with auto_pad"));
            }
            let padding = match auto_pad {
                b"NOTSET" => pads,
                b"VALID" => [0; 4],
                b"SAME_UPPER" => conv_same_padding(
                    g.shape(x)?.dims(),
                    g.shape(w)?.dims(),
                    strides,
                    dilations,
                    false,
                )?,
                b"SAME_LOWER" => conv_same_padding(
                    g.shape(x)?.dims(),
                    g.shape(w)?.dims(),
                    strides,
                    dilations,
                    true,
                )?,
                _ => return Err(bad("unsupported Conv auto_pad")),
            };
            g.conv2d(
                x,
                w,
                if ins.len() == 3 { Some(get(2)?) } else { None },
                Conv2dOptions {
                    groups,
                    stride: strides,
                    dilation: dilations,
                    padding,
                },
            )?
        }
        _ => return Err(bad(format!("unsupported ONNX opset-13 operator {op}"))),
    };
    values.insert(outs[0].to_owned(), out);
    Ok(())
}
