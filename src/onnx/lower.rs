//! Graph lowering for the validated static ONNX subset.

use super::{
    bad,
    schema::{
        attrs, axes_usize, const_i64, conv_pads, conv_pair, conv_same_padding, onnx_pool_options,
        packed_i64, reshape_dims, scalar_f32, scalar_i64, strict_typed_scalar_i64_attr,
        strict_typed_packed_i64_attr, strict_typed_string_attr, typed_scalar_f32_attr,
        typed_scalar_i64_attr,
    },
    tensor::{onnx_dtype, tensor_data},
    wire::{Msg, var},
};
use crate::{
    ir::reduction_shape, Conv2dOptions, DType, Graph, NodeId, ReduceKind, ReductionDType,
    Result, Scalar, Shape, Slice, TensorData,
};
use std::collections::BTreeMap;

fn prelu_dtype(x: DType, slope: DType) -> DType {
    // tinygrad's weak binary lowering resolves the only supported lattice
    // disagreement, U64 mixed with I64, at its default F32 width. RustGrad's
    // generic promotion intentionally chooses F64 for that pair.
    if matches!((x, slope), (DType::U64, DType::I64) | (DType::I64, DType::U64)) {
        DType::F32
    } else {
        x.promote(slope)
    }
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
fn static_one_hot_depth(
    constants: &BTreeMap<String, TensorData>,
    name: &str,
) -> Result<i64> {
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
        Scalar::U(value) => i64::try_from(value)
            .map_err(|_| bad("OneHot depth is not representable by arange")),
        Scalar::F(value) => {
            // Python rejects non-finite float-to-int conversion.  The upper
            // bound is exclusive because `i64::MAX as f64` rounds to 2^63.
            let value = value.trunc();
            if !value.is_finite()
                || value < i64::MIN as f64
                || value >= 9_223_372_036_854_775_808.0
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
        return Err(bad("OneHot depth below -1 is unsupported by source reshape"));
    }
    let classes = if raw_depth <= 0 {
        0usize
    } else {
        usize::try_from(raw_depth)
            .map_err(|_| bad("OneHot depth is not representable by shape"))?
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
    let index_depth = TensorData::scalar_with_dtype(Scalar::I(i64::from(raw_depth as i32)), DType::I32);
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

fn hardmax_plan(
    g: &Graph,
    input: NodeId,
    attrs: &BTreeMap<String, Vec<u8>>,
) -> Result<HardmaxPlan> {
    if attrs.keys().any(|key| key != "axis") {
        return Err(bad("unsupported Hardmax attribute"));
    }
    let shape = g.shape(input)?.clone();
    let dtype = g.dtype(input)?;
    let input_numel = shape.numel()?;
    let rank = shape.rank();
    if rank == 0 {
        // tinygrad's explicit `argmax(axis=-1)` path indexes the scalar
        // shape after resolving its axis, rather than using argmax(None).
        return Err(bad("Hardmax does not support scalar input"));
    }
    let rank_i64 = i64::try_from(rank).map_err(|_| bad("Hardmax rank overflow"))?;
    let raw_axis = attrs
        .get("axis")
        .map(|raw| scalar_i64(raw))
        .transpose()?
        .unwrap_or(-1);
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
    let sentinel = i32::try_from(axis_extent)
        .map_err(|_| bad("Hardmax axis extent exceeds I32 indices"))?;

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
    let mut first_bounds = Vec::with_capacity(rank);
    for (dimension, &extent) in shape.dims().iter().enumerate() {
        first_bounds.push(if dimension == axis { (0, 1) } else { (0, extent) });
    }
    let first_shape = Shape::new(
        first_bounds
            .iter()
            .map(|(start, end)| end - start)
            .collect::<Vec<_>>(),
    );
    first_shape.numel()?;
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
    let sentinel_data = TensorData::scalar_with_dtype(Scalar::I(i64::from(sentinel)), DType::I32);
    if arg_shape.broadcast_with(sentinel_data.shape())? != arg_shape {
        return Err(bad("Hardmax NaN sentinel cannot broadcast to argmax"));
    }
    let mut restored_dims = arg_shape.dims().to_vec();
    restored_dims.insert(axis, 1);
    let restored_index_shape = Shape::new(restored_dims);
    restored_index_shape.numel()?;

    let mut class_dims = vec![1; rank];
    class_dims[axis] = axis_extent;
    let class_shape = Shape::new(class_dims);
    class_shape.numel()?;
    if class_shape.broadcast_with(&restored_index_shape)? != shape {
        return Err(bad("Hardmax classes cannot broadcast to input"));
    }
    let classes = TensorData::arange(0, i64::try_from(axis_extent).map_err(|_| {
        bad("Hardmax axis extent exceeds arange range")
    })?, 1)?
    .cast(DType::I32);
    if classes.shape() != &Shape::new([axis_extent]) || classes.dtype() != DType::I32 {
        return Err(bad("Hardmax class range does not match validated axis"));
    }
    // The final compare restores the original shape and bool-to-source cast
    // preserves the exact input dtype.
    shape.numel()?;
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

struct GeluPlan {
    mode: String,
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    half: TensorData,
    one: TensorData,
    two: TensorData,
    root_two: TensorData,
    root_two_over_pi: TensorData,
    coefficient: TensorData,
    three: TensorData,
    neg_inv_ln2: TensorData,
    empty: bool,
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

struct ModPlan { fmod: bool, shape: Shape, dtype: DType }

struct GlobalAveragePoolPlan {
    axes: Vec<isize>,
    sum_dtypes: ReductionDType,
    work_dtype: DType,
    output_dtype: DType,
    divisor: TensorData,
    output_shape: Shape,
}

struct SoftplusPlan { input_dtype: DType, output_dtype: DType, shape: Shape, zero: TensorData, empty: bool }

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

/// Full descriptor plan for tinygrad's CELU composition.  In particular,
/// maximum/minimum use their source comparison decomposition instead of the
/// cross-backend extrema helpers, whose NaN and signed-zero ties are not this
/// operator's contract.
struct CeluPlan {
    input_dtype: DType,
    output_dtype: DType,
    shape: Shape,
    input_zero: TensorData,
    negative_work_zero: TensorData,
    one: TensorData,
    alpha: TensorData,
    empty: bool,
}

fn celu_plan(g: &Graph, input: NodeId, n: &Msg<'_>, attrs: &BTreeMap<String, Vec<u8>>) -> Result<CeluPlan> {
    if attrs.keys().any(|key| key != "alpha") { return Err(bad("unsupported Celu attribute")); }
    // The source passes the ONNX FLOAT through as a weak scalar: it admits
    // every IEEE F32 payload, including zero, NaN, and infinities.
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1.0);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize()).ok_or_else(|| bad("Celu input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    numel.checked_mul(output_dtype.itemsize()).ok_or_else(|| bad("Celu output byte extent overflow"))?;
    let input_zero = TensorData::scalar_with_dtype(Scalar::F(0.0), input_dtype);
    let negative_work_zero = TensorData::scalar_with_dtype(Scalar::F(-0.0), output_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
    let alpha = TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), output_dtype);
    for scalar in [&input_zero, &negative_work_zero, &one, &alpha] {
        if shape.broadcast_with(scalar.shape())? != shape { return Err(bad("Celu scalar broadcast mismatch")); }
    }
    if input_zero.dtype() != input_dtype
        || negative_work_zero.dtype() != output_dtype
        || one.dtype() != output_dtype
        || alpha.dtype() != output_dtype
        || input_dtype.promote(input_dtype) != input_dtype
        || input_dtype.promote(output_dtype) != output_dtype
        || output_dtype.promote(output_dtype) != output_dtype
    {
        return Err(bad("Celu scalar promotion mismatch"));
    }
    Ok(CeluPlan {
        input_dtype, output_dtype, shape, input_zero, negative_work_zero,
        one, alpha, empty: numel == 0,
    })
}

fn softsign_plan(g: &Graph, input: NodeId, attrs: &BTreeMap<String, Vec<u8>>) -> Result<SoftsignPlan> {
    if !attrs.is_empty() { return Err(bad("unsupported Softsign attribute")); }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize()).ok_or_else(|| bad("Softsign input byte extent overflow"))?;

    // `1 + x.abs()` stays at X's concrete storage dtype. Tensor.div then
    // lowers literally to `x * reciprocal(denominator)`, so exact storage
    // becomes F32 only at reciprocal for Bool/integer inputs.
    let reciprocal_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    let output_dtype = input_dtype.promote(reciprocal_dtype);
    numel.checked_mul(output_dtype.itemsize()).ok_or_else(|| bad("Softsign output byte extent overflow"))?;
    let one = TensorData::scalar_with_dtype(Scalar::I(1), input_dtype);
    if one.dtype() != input_dtype
        || shape.broadcast_with(one.shape())? != shape
        || input_dtype.promote(input_dtype) != input_dtype
        || input_dtype.promote(reciprocal_dtype) != output_dtype
    {
        return Err(bad("Softsign scalar promotion mismatch"));
    }
    Ok(SoftsignPlan { input_dtype, output_dtype, shape, one, empty: numel == 0 })
}

fn softplus_plan(g: &Graph, input: NodeId, attrs: &BTreeMap<String, Vec<u8>>) -> Result<SoftplusPlan> {
    if !attrs.is_empty() { return Err(bad("unsupported Softplus attribute")); }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize()).ok_or_else(|| bad("Softplus input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    numel.checked_mul(output_dtype.itemsize()).ok_or_else(|| bad("Softplus output byte extent overflow"))?;
    let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), output_dtype);
    if shape.broadcast_with(zero.shape())? != shape || zero.dtype() != output_dtype { return Err(bad("Softplus scalar promotion mismatch")); }
    Ok(SoftplusPlan { input_dtype, output_dtype, shape, zero, empty: numel == 0 })
}

fn global_average_pool_plan(g: &Graph, input: NodeId) -> Result<GlobalAveragePoolPlan> {
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize()).ok_or_else(|| bad("GlobalAveragePool input byte extent overflow"))?;
    let axes = (2..shape.rank()).map(|axis| axis as isize).collect::<Vec<_>>();
    let count = shape.dims()[2..].iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or_else(|| bad("GlobalAveragePool divisor overflow"))?;
    let sum_dtypes = ReductionDType::sum_default(input_dtype);
    let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    let work_dtype = if input_dtype.is_float() { sum_dtypes.accumulator } else { DType::F32 };
    let mut output_dims = shape.dims().to_vec();
    for dim in output_dims.iter_mut().skip(2) { *dim = 1; }
    let output_shape = Shape::new(output_dims);
    output_shape.numel()?.checked_mul(output_dtype.itemsize()).ok_or_else(|| bad("GlobalAveragePool output byte extent overflow"))?;
    let divisor = TensorData::scalar_with_dtype(Scalar::F(count as f64), work_dtype);
    if output_shape.broadcast_with(divisor.shape())? != output_shape || output_dtype.promote(output_dtype) != output_dtype {
        return Err(bad("GlobalAveragePool scalar promotion mismatch"));
    }
    Ok(GlobalAveragePoolPlan { axes, sum_dtypes, work_dtype, output_dtype, divisor, output_shape })
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
    if attrs.keys().any(|key| key != "fmod") { return Err(bad("unsupported Mod attribute")); }
    let fmod = strict_typed_scalar_i64_attr(n, "fmod")?.unwrap_or(0) != 0;
    let lhs_shape = g.shape(lhs)?.clone();
    let rhs_shape = g.shape(rhs)?.clone();
    let lhs_dtype = g.dtype(lhs)?;
    let rhs_dtype = g.dtype(rhs)?;
    lhs_shape.numel()?.checked_mul(lhs_dtype.itemsize()).ok_or_else(|| bad("Mod lhs byte extent overflow"))?;
    rhs_shape.numel()?.checked_mul(rhs_dtype.itemsize()).ok_or_else(|| bad("Mod rhs byte extent overflow"))?;
    let dtype = lhs_dtype.promote(rhs_dtype);
    let shape = lhs_shape.broadcast_with(&rhs_shape)?;
    shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| bad("Mod output byte extent overflow"))?;
    if dtype.is_integer() {
        if let Some(value) = constants.get(rhs_name) {
            if value.dtype().is_integer() && (0..value.len()).any(|i| value.scalar_at(i).as_i64() == 0) {
                return Err(bad("Mod integer divisor constant contains zero"));
            }
        }
    }
    Ok(ModPlan { fmod, shape, dtype })
}

fn swish_plan(g: &Graph, input: NodeId, n: &Msg<'_>, attrs: &BTreeMap<String, Vec<u8>>) -> Result<SwishPlan> {
    if attrs.keys().any(|key| key != "alpha") { return Err(bad("unsupported Swish attribute")); }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1.0);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize()).ok_or_else(|| bad("Swish input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    numel.checked_mul(output_dtype.itemsize()).ok_or_else(|| bad("Swish output byte extent overflow"))?;
    let alpha = TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), output_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
    let neg_inv_ln2 = TensorData::scalar_with_dtype(Scalar::F(-1.0 / std::f64::consts::LN_2), output_dtype);
    for scalar in [&alpha, &one, &neg_inv_ln2] {
        if scalar.dtype() != output_dtype || shape.broadcast_with(scalar.shape())? != shape { return Err(bad("Swish scalar promotion mismatch")); }
    }
    if output_dtype.promote(output_dtype) != output_dtype { return Err(bad("Swish output promotion mismatch")); }
    Ok(SwishPlan { input_dtype, output_dtype, shape, alpha, one, neg_inv_ln2, empty: numel == 0 })
}

fn selu_plan(g: &Graph, input: NodeId, n: &Msg<'_>, attrs: &BTreeMap<String, Vec<u8>>) -> Result<SeluPlan> {
    if attrs.keys().any(|key| key != "alpha" && key != "gamma") {
        return Err(bad("unsupported Selu attribute"));
    }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1.67326);
    let gamma = typed_scalar_f32_attr(n, "gamma")?.unwrap_or(1.0507);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize()).ok_or_else(|| bad("Selu input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    numel.checked_mul(output_dtype.itemsize()).ok_or_else(|| bad("Selu output byte extent overflow"))?;
    let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), output_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
    let alpha = TensorData::scalar_with_dtype(Scalar::F(f64::from(alpha)), output_dtype);
    let gamma = TensorData::scalar_with_dtype(Scalar::F(f64::from(gamma)), output_dtype);
    for scalar in [&zero, &one, &alpha, &gamma] {
        if scalar.dtype() != output_dtype || shape.broadcast_with(scalar.shape())? != shape {
            return Err(bad("Selu scalar promotion mismatch"));
        }
    }
    if output_dtype.promote(output_dtype) != output_dtype { return Err(bad("Selu output promotion mismatch")); }
    Ok(SeluPlan { input_dtype, output_dtype, shape, zero, one, alpha, gamma, empty: numel == 0 })
}

fn elu_plan(g: &Graph, input: NodeId, n: &Msg<'_>, attrs: &BTreeMap<String, Vec<u8>>) -> Result<EluPlan> {
    if attrs.keys().any(|key| key != "alpha") {
        return Err(bad("unsupported Elu attribute"));
    }
    let alpha = typed_scalar_f32_attr(n, "alpha")?.unwrap_or(1.0);
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize()).ok_or_else(|| bad("Elu input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    numel.checked_mul(output_dtype.itemsize()).ok_or_else(|| bad("Elu output byte extent overflow"))?;
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
    Ok(EluPlan { input_dtype, output_dtype, shape, zero, one, alpha, empty: numel == 0 })
}

fn gelu_plan(g: &Graph, input: NodeId, n: &Msg<'_>, attrs: &BTreeMap<String, Vec<u8>>) -> Result<GeluPlan> {
    if attrs.keys().any(|key| key != "approximate") {
        return Err(bad("unsupported Gelu attribute"));
    }
    let mode = strict_typed_string_attr(n, "approximate")?.unwrap_or_else(|| "none".into());
    if mode != "none" && mode != "tanh" {
        return Err(bad("unsupported Gelu approximation"));
    }
    let shape = g.shape(input)?.clone();
    let input_dtype = g.dtype(input)?;
    let numel = shape.numel()?;
    numel.checked_mul(input_dtype.itemsize())
        .ok_or_else(|| bad("Gelu input byte extent overflow"))?;
    let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    numel.checked_mul(output_dtype.itemsize())
        .ok_or_else(|| bad("Gelu output byte extent overflow"))?;
    let scalar = |value| TensorData::scalar_with_dtype(Scalar::F(value), output_dtype);
    let half = scalar(0.5);
    let one = scalar(1.0);
    let two = scalar(2.0);
    let root_two = scalar(std::f64::consts::SQRT_2);
    let root_two_over_pi = scalar((2.0 / std::f64::consts::PI).sqrt());
    let coefficient = scalar(0.044_715);
    let three = scalar(3.0);
    let neg_inv_ln2 = scalar(-1.0 / std::f64::consts::LN_2);
    let scalar_shape = Shape::new([]);
    for value in [&half, &one, &two, &root_two, &root_two_over_pi, &coefficient, &three, &neg_inv_ln2] {
        if value.dtype() != output_dtype || shape.broadcast_with(value.shape())? != shape {
            return Err(bad("Gelu scalar promotion mismatch"));
        }
    }
    if output_dtype.promote(output_dtype) != output_dtype {
        return Err(bad("Gelu output promotion mismatch"));
    }
    let _ = scalar_shape;
    Ok(GeluPlan { mode, input_dtype, output_dtype, shape, half, one, two, root_two, root_two_over_pi, coefficient, three, neg_inv_ln2, empty: numel == 0 })
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
    let raw_size = strict_typed_scalar_i64_attr(n, "size")?
        .ok_or_else(|| bad("LRN requires size"))?;
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
            Slice { start: None, stop: None, step: 1 },
            Slice { start: None, stop: None, step: 1 },
            Slice { start: Some(start), stop: Some(end), step: 1 },
            Slice { start: None, stop: None, step: 1 },
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

fn center_crop_pad_zero(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(0),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U(0),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(0.0),
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

    let shrink_shape = Shape::new(bounds.iter().map(|(start, end)| end - start).collect::<Vec<_>>());
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
    let shrink = (bounds != input_shape.dims().iter().map(|&dimension| (0, dimension)).collect::<Vec<_>>())
        .then_some(bounds);
    let padding = padding.iter().any(|&(before, after)| before != 0 || after != 0).then_some(padding);
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
        return Err(bad("DepthToSpace source reshape rejects empty batch or spatial extent"));
    }
    let input_numel = input_shape.numel()?;
    input_numel
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| bad("DepthToSpace input byte extent overflow"))?;
    let block_area = blocksize
        .checked_mul(blocksize)
        .ok_or_else(|| bad("DepthToSpace block area overflow"))?;
    if channels % block_area != 0 {
        return Err(bad("DepthToSpace channels must be divisible by blocksize squared"));
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
        return Err(bad("DepthToSpace intermediate reshape changes element count"));
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
        return Err(bad("SpaceToDepth spatial dimensions must be divisible by blocksize"));
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
        return Err(bad("SpaceToDepth intermediate reshape changes element count"));
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
    let rows = shape.dims()[0];
    let columns = shape.dims()[1];
    let rows_i64 = i64::try_from(rows).map_err(|_| bad("EyeLike row extent overflow"))?;
    let columns_i64 =
        i64::try_from(columns).map_err(|_| bad("EyeLike column extent overflow"))?;
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
    input_shape.numel()?;
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
    Ok(ShrinkActivationPlan {
        work_dtype,
        output_dtype,
        narrow,
        // Unary negation happens on the source FLOAT payload before weak
        // promotion, preserving signed zero and every IEEE special payload.
        negative_lambda: TensorData::scalar_with_dtype(
            Scalar::F(f64::from(-lambd)),
            work_dtype,
        ),
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

fn cumsum_zero(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(0),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U(0),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(0.0),
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
    if attrs.keys().any(|key| key != "exclusive" && key != "reverse") {
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
                    .map(|(dimension, &extent)| {
                        if dimension == axis { end + 1 } else { extent }
                    })
                    .collect(),
            );
            prefix.numel()?;
            let reduced = Shape::new(
                prefix
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &extent)| if dimension == axis { 1 } else { extent })
                    .collect(),
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
    let columns_i64 =
        i64::try_from(columns).map_err(|_| bad("Trilu column extent exceeds I64"))?;
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
    Ok(ReduceLogSumPlan {
        reduction,
        ln2,
    })
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
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(f64::NEG_INFINITY),
    }
}

fn global_max_pool_plan(g: &Graph, x: NodeId) -> Result<GlobalMaxPoolPlan> {
    let dtype = g.dtype(x)?;
    let shape = g.shape(x)?.clone();
    shape.numel()?;
    let axes = (2..shape.rank()).map(|axis| axis as isize).collect::<Vec<_>>();
    let empty_spatial = axes
        .iter()
        .any(|&axis| shape.dims()[axis as usize] == 0);
    let output_shape = if axes.is_empty() {
        shape
    } else {
        Shape::new(
            g.shape(x)?
                .dims()
                .iter()
                .enumerate()
                .map(|(axis, &extent)| if axis >= 2 { 1 } else { extent })
                .collect(),
        )
    };
    let output_numel = output_shape.numel()?;
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
    if outs.len() != 1 || outs[0].is_empty() || values.contains_key(outs[0]) {
        return Err(bad("invalid or duplicate ONNX node output"));
    }
    let get = |i: usize| -> Result<NodeId> {
        ins.get(i)
            .and_then(|x| values.get(*x))
            .copied()
            .ok_or_else(|| bad("missing ONNX node input"))
    };
    let attrs = attrs(&n)?;
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
                let padded = g.pad(reshaped, plan.padding, center_crop_pad_zero(plan.input_dtype))?;
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
            let plan = gelu_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? }
            } else {
                let mode = plan.mode.clone();
                let x = if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? };
                let half = g.constant(plan.half);
                let one = g.constant(plan.one);
                let two = g.constant(plan.two);
                let value = match mode.as_str() {
                    "none" => {
                        let root_two = g.constant(plan.root_two);
                        let scaled = g.div(x, root_two)?;
                        let erf = g.erf(scaled)?;
                        let left = g.mul(x, half)?;
                        let right = g.add(one, erf)?;
                        g.mul(left, right)?
                    }
                    "tanh" => {
                        let three = g.constant(plan.three);
                        let coefficient = g.constant(plan.coefficient);
                        let scale = g.constant(plan.root_two_over_pi);
                        let neg_inv_ln2 = g.constant(plan.neg_inv_ln2);
                        // Keep `x ** 3` as Pow, then expand Tensor.tanh through
                        // its source sigmoid rather than using a unary shortcut.
                        let cube = g.pow(x, three)?;
                        let inner = g.add(x, g.mul(coefficient, cube)?)?;
                        let z = g.mul(scale, inner)?;
                        let doubled = g.mul(two, z)?;
                        let exponent = g.mul(doubled, neg_inv_ln2)?;
                        let sigmoid = g.reciprocal(g.add(one, g.exp2(exponent)?)?)?;
                        let tanh = g.sub(g.mul(two, sigmoid)?, one)?;
                        g.mul(g.mul(half, x)?, g.add(one, tanh)?)?
                    }
                    _ => unreachable!("Gelu plan validated mode"),
                };
                debug_assert_eq!(g.shape(value).expect("Gelu shape preflighted"), &plan.shape);
                debug_assert_eq!(g.dtype(value).expect("Gelu dtype preflighted"), plan.output_dtype);
                value
            }
        }
        "Elu" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = elu_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? }
            } else {
                let x = if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? };
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
                debug_assert_eq!(g.dtype(output).expect("Elu dtype preflighted"), plan.output_dtype);
                output
            }
        }
        "Celu" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = celu_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? }
            } else {
                let input_zero = g.constant(plan.input_zero);
                let negative_work_zero = g.constant(plan.negative_work_zero);
                let one = g.constant(plan.one);
                let alpha = g.constant(plan.alpha);
                // Tensor.celu is literally
                // `x.maximum(0) + (alpha * ((x / alpha).exp() - 1)).minimum(0)`.
                // A source Max decomposes as `(lhs < rhs).where(rhs, lhs)`.
                let positive = g.select(g.lt(input, input_zero)?, input_zero, input)?;
                // Tensor.div is reciprocal then multiply, rather than a
                // hardware divide. This keeps alpha's source-width rounding.
                let scaled = g.mul(input, g.reciprocal(alpha)?)?;
                let scaled_negative = g.mul(alpha, g.sub(g.exp(scaled)?, one)?)?;
                // Float minimum is `-((-lhs).maximum(-rhs))`; spelling its
                // comparison form keeps NaNs and equal signed zeroes on the
                // first operand exactly as the checked-in source does.
                let negated = g.neg(scaled_negative)?;
                let selected = g.select(g.lt(negated, negative_work_zero)?, negative_work_zero, negated)?;
                let negative = g.neg(selected)?;
                let output = g.add(positive, negative)?;
                debug_assert_eq!(g.shape(output).expect("Celu shape preflighted"), &plan.shape);
                debug_assert_eq!(g.dtype(output).expect("Celu dtype preflighted"), plan.output_dtype);
                output
            }
        }
        "Selu" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = selu_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? }
            } else {
                let x = if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? };
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
                debug_assert_eq!(g.shape(output).expect("Selu shape preflighted"), &plan.shape);
                debug_assert_eq!(g.dtype(output).expect("Selu dtype preflighted"), plan.output_dtype);
                output
            }
        }
        "Swish" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = swish_plan(g, input, &n, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? }
            } else {
                let x = if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? };
                let alpha = g.constant(plan.alpha);
                let one = g.constant(plan.one);
                let neg_inv_ln2 = g.constant(plan.neg_inv_ln2);
                let scaled = g.mul(x, alpha)?;
                let exponent = g.mul(scaled, neg_inv_ln2)?;
                let sigmoid = g.reciprocal(g.add(one, g.exp2(exponent)?)?)?;
                let output = g.mul(x, sigmoid)?;
                debug_assert_eq!(g.shape(output).expect("Swish shape preflighted"), &plan.shape);
                debug_assert_eq!(g.dtype(output).expect("Swish dtype preflighted"), plan.output_dtype);
                output
            }
        }
        "Softplus" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = softplus_plan(g, input, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? }
            } else {
                let x = if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? };
                // ONNX supplies tinygrad's default beta=1.  Keeping zero at
                // X's concrete storage width makes Graph::logaddexp exactly
                // the source `(x*1).logaddexp(0) * 1` stable composition.
                let output = g.logaddexp(x, g.constant(plan.zero))?;
                debug_assert_eq!(g.shape(output).expect("Softplus shape preflighted"), &plan.shape);
                debug_assert_eq!(g.dtype(output).expect("Softplus dtype preflighted"), plan.output_dtype);
                output
            }
        }
        "Softsign" if ins.len() == 1 => {
            let input = get(0)?;
            let plan = softsign_plan(g, input, &attrs)?;
            if plan.empty {
                if plan.input_dtype == plan.output_dtype { input } else { g.cast(input, plan.output_dtype)? }
            } else {
                let one = g.constant(plan.one);
                // Keep tinygrad's literal `x / (1 + x.abs())` decomposition:
                // abs is `x * sign(x)` and true division is reciprocal then
                // multiply. Unary Abs and Graph::softsign erase those
                // source-visible storage and signed-zero boundaries.
                let absolute = g.mul(input, g.sign(input)?)?;
                let denominator = g.add(one, absolute)?;
                let output = g.mul(input, g.reciprocal(denominator)?)?;
                debug_assert_eq!(g.shape(output).expect("Softsign shape preflighted"), &plan.shape);
                debug_assert_eq!(g.dtype(output).expect("Softsign dtype preflighted"), plan.output_dtype);
                output
            }
        }
        "Mod" if ins.len() == 2 => {
            let lhs = get(0)?;
            let rhs = get(1)?;
            let plan = mod_plan(g, lhs, rhs, ins[1], &n, &attrs, constants)?;
            let output = if plan.fmod { g.fmod(lhs, rhs)? } else { g.modulo(lhs, rhs)? };
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
            let plan = hardmax_plan(g, input, &attrs)?;
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
        "Relu" if ins.len() == 1 => g.relu(get(0)?)?,
        "Sigmoid" if ins.len() == 1 => g.sigmoid(get(0)?)?,
        "Tanh" if ins.len() == 1 => g.tanh(get(0)?)?,
        "Add" if ins.len() == 2 => g.add(get(0)?, get(1)?)?,
        "Sum" if !ins.is_empty() && attrs.is_empty() => {
            // tinygrad lowers variadic Sum through functools.reduce(Tensor.add,
            // data_0), so one input is an identity and all later operands are
            // folded in source order. Simulate every Add contract before
            // appending the first graph node.
            let first = get(0)?;
            let mut inputs = vec![first];
            let mut output_shape = g.shape(first)?.clone();
            let mut output_dtype = g.dtype(first)?;
            output_shape.numel()?;
            for index in 1..ins.len() {
                let input = get(index)?;
                let shape = g.shape(input)?.clone();
                let dtype = g.dtype(input)?;
                shape.numel()?;
                output_shape = output_shape.broadcast_with(&shape)?;
                output_shape.numel()?;
                output_dtype = output_dtype.promote(dtype);
                inputs.push(input);
            }
            // Add has no additional dtype restriction beyond the promotion
            // lattice. Retain the computed dtype as an explicit preflight
            // fact while preserving its existing node construction path.
            let _output_dtype = output_dtype;
            let mut sum = first;
            for input in inputs.into_iter().skip(1) {
                sum = g.add(sum, input)?;
            }
            sum
        }
        "Mean" if !ins.is_empty() && attrs.is_empty() => {
            // tinygrad defines variadic Mean as `Sum(*data_0) / len(data_0)`.
            // Its true-division path lifts integer and Bool numerators to the
            // default float, while a floating sum retains its dtype. Compute
            // the entire source-order Add and final scalar-Div contract before
            // creating the first fold, cast, constant, or division node.
            let first = get(0)?;
            let mut inputs = vec![first];
            let mut output_shape = g.shape(first)?.clone();
            let mut sum_dtype = g.dtype(first)?;
            output_shape.numel()?;
            for index in 1..ins.len() {
                let input = get(index)?;
                let shape = g.shape(input)?.clone();
                let dtype = g.dtype(input)?;
                shape.numel()?;
                output_shape = output_shape.broadcast_with(&shape)?;
                output_shape.numel()?;
                sum_dtype = sum_dtype.promote(dtype);
                inputs.push(input);
            }
            let division_dtype = if sum_dtype.is_float() {
                sum_dtype
            } else {
                DType::F32
            };
            let divisor_shape = Shape::new([]);
            let output_shape = output_shape.broadcast_with(&divisor_shape)?;
            output_shape.numel()?;
            let division_output_dtype = division_dtype.promote(division_dtype);
            let divisor = TensorData::scalar_with_dtype(
                Scalar::F(ins.len() as f64),
                division_output_dtype,
            );
            let mut sum = first;
            for input in inputs.into_iter().skip(1) {
                sum = g.add(sum, input)?;
            }
            let sum = if sum_dtype == division_output_dtype {
                sum
            } else {
                g.cast(sum, division_output_dtype)?
            };
            let divisor = g.constant(divisor);
            g.div(sum, divisor)?
        }
        "Sub" if ins.len() == 2 => g.sub(get(0)?, get(1)?)?,
        "Mul" if ins.len() == 2 => g.mul(get(0)?, get(1)?)?,
        "Div" if ins.len() == 2 && attrs.is_empty() => {
            // tinygrad dispatches Div with no attributes. Resolve and validate
            // both operands before lowering so malformed broadcasts cannot
            // append a partial node or silently ignore an attribute.
            let lhs = get(0)?;
            let rhs = get(1)?;
            g.shape(lhs)?.broadcast_with(g.shape(rhs)?)?;
            g.div(lhs, rhs)?
        }
        "MatMul" if ins.len() == 2 && attrs.is_empty() => {
            let lhs = get(0)?;
            let rhs = get(1)?;
            g.matmul(lhs, rhs)?
        }
        "Cast" if ins.len() == 1 && attrs.len() == 1 => {
            let x = attrs.get("to").ok_or_else(|| bad("Cast needs to"))?;
            let mut at = 0;
            g.cast(get(0)?, onnx_dtype(var(x, &mut at)?)?)?
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
            g.shape(input)?.numel()?;
            // tinygrad derives CastLike entirely from target_type.dtype; its
            // values and shape have no effect on the result.
            if input_dtype == target_dtype {
                input
            } else {
                g.cast(input, target_dtype)?
            }
        }
        "Constant" if ins.is_empty() && attrs.len() == 1 => {
            let data = tensor_data(Msg::new(
                attrs
                    .get("value")
                    .ok_or_else(|| bad("Constant needs value"))?,
            ))?;
            constants.insert(outs[0].to_owned(), data.clone());
            g.constant(data)
        }
        "Reshape" if ins.len() == 2 => {
            if attrs.keys().any(|name| name != "allowzero") {
                return Err(bad("unsupported Reshape attribute"));
            }
            match attrs
                .get("allowzero")
                .map(|value| scalar_i64(value))
                .transpose()?
            {
                None | Some(0) => {}
                Some(1) => return Err(bad("Reshape allowzero=1 is unsupported")),
                Some(_) => return Err(bad("Reshape allowzero must be 0 or 1")),
            }
            let shape = const_i64(constants, ins[1])?;
            let source = g.shape(get(0)?)?.dims().to_vec();
            g.reshape(get(0)?, reshape_dims(&source, &shape)?)?
        }
        "Transpose" if ins.len() == 1 => {
            if attrs.keys().any(|name| name != "perm") {
                return Err(bad("unsupported Transpose attribute"));
            }
            let rank = g.shape(get(0)?)?.rank();
            let axes = attrs
                .get("perm")
                .map(|x| packed_i64(x))
                .transpose()?
                .unwrap_or_else(|| (0..rank).rev().map(|x| x as i64).collect());
            g.permute(get(0)?, axes_usize(&axes, rank)?)?
        }
        "Flatten" if ins.len() == 1 => {
            if attrs.keys().any(|name| name != "axis") {
                return Err(bad("unsupported Flatten attribute"));
            }
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            let rank = g.shape(get(0)?)?.rank() as i64;
            g.flatten(
                get(0)?,
                isize::try_from(axis).map_err(|_| bad("Flatten axis overflow"))?,
                isize::try_from(rank - 1).map_err(|_| bad("Flatten rank overflow"))?,
            )?
        }
        "Squeeze" if ins.len() == 2 => {
            if !attrs.is_empty() {
                return Err(bad("unsupported Squeeze attribute"));
            }
            let mut axes = const_i64(constants, ins[1])?
                .into_iter()
                .map(|axis| isize::try_from(axis).map_err(|_| bad("Squeeze axis overflow")))
                .collect::<Result<Vec<_>>>()?;
            axes.sort_unstable_by(|left, right| right.cmp(left));
            let input = get(0)?;
            let mut shape = g.shape(input)?.clone();
            for &axis in &axes {
                let rank = isize::try_from(shape.rank()).map_err(|_| bad("Squeeze rank overflow"))?;
                let axis = if axis < 0 {
                    axis.checked_add(rank)
                        .ok_or_else(|| bad("invalid Squeeze axis"))?
                } else {
                    axis
                };
                if axis < 0 || axis >= rank {
                    return Err(bad("invalid Squeeze axis"));
                }
                if shape.dims()[axis as usize] == 1 {
                    let mut dims = shape.dims().to_vec();
                    dims.remove(axis as usize);
                    shape = Shape::new(dims);
                    shape.numel()?;
                }
            }
            let mut out = input;
            for axis in axes {
                out = g.squeeze(out, Some(axis))?;
            }
            out
        }
        "Unsqueeze" if ins.len() == 2 => {
            if !attrs.is_empty() {
                return Err(bad("unsupported Unsqueeze attribute"));
            }
            let mut axes = const_i64(constants, ins[1])?
                .into_iter()
                .map(|axis| isize::try_from(axis).map_err(|_| bad("Unsqueeze axis overflow")))
                .collect::<Result<Vec<_>>>()?;
            axes.sort_unstable();
            let input = get(0)?;
            let mut shape = g.shape(input)?.clone();
            for &axis in &axes {
                let rank = shape
                    .rank()
                    .checked_add(1)
                    .and_then(|rank| isize::try_from(rank).ok())
                    .ok_or_else(|| bad("Unsqueeze rank overflow"))?;
                let axis = if axis < 0 {
                    axis.checked_add(rank)
                        .ok_or_else(|| bad("invalid Unsqueeze axis"))?
                } else {
                    axis
                };
                if axis < 0 || axis >= rank {
                    return Err(bad("invalid Unsqueeze axis"));
                }
                let mut dims = shape.dims().to_vec();
                dims.insert(axis as usize, 1);
                shape = Shape::new(dims);
                shape.numel()?;
            }
            let mut out = input;
            for axis in axes {
                out = g.unsqueeze(out, axis)?;
            }
            out
        }
        "Concat" if ins.len() >= 2 => {
            if attrs.len() != 1 || !attrs.contains_key("axis") {
                return Err(bad("Concat requires only an axis attribute"));
            }
            let axis = scalar_i64(attrs.get("axis").ok_or_else(|| bad("Concat needs axis"))?)?;
            let rank = g.shape(get(0)?)?.rank();
            g.concat(
                ins.iter()
                    .map(|x| {
                        values
                            .get(*x)
                            .copied()
                            .ok_or_else(|| bad("missing ONNX input"))
                    })
                    .collect::<Result<Vec<_>>>()?,
                axes_usize(&[axis], rank)?[0],
            )?
        }
        "Softmax" if ins.len() == 1 => {
            if attrs.keys().any(|name| name != "axis") {
                return Err(bad("unsupported Softmax attribute"));
            }
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(-1);
            g.softmax(
                get(0)?,
                isize::try_from(axis).map_err(|_| bad("Softmax axis overflow"))?,
                None,
            )?
        }
        "LogSoftmax" if ins.len() == 1 => {
            if attrs.keys().any(|name| name != "axis") {
                return Err(bad("unsupported LogSoftmax attribute"));
            }
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(-1);
            g.log_softmax(
                get(0)?,
                isize::try_from(axis).map_err(|_| bad("LogSoftmax axis overflow"))?,
                None,
            )?
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
        "Equal" if ins.len() == 2 && attrs.is_empty() => g.eq(get(0)?, get(1)?)?,
        "Less" if ins.len() == 2 && attrs.is_empty() => g.lt(get(0)?, get(1)?)?,
        "LessOrEqual" if ins.len() == 2 && attrs.is_empty() => g.le(get(0)?, get(1)?)?,
        "Greater" if ins.len() == 2 && attrs.is_empty() => g.gt(get(0)?, get(1)?)?,
        "GreaterOrEqual" if ins.len() == 2 && attrs.is_empty() => g.ge(get(0)?, get(1)?)?,
        "Where" if ins.len() == 3 && attrs.is_empty() => g.select(get(0)?, get(1)?, get(2)?)?,
        "Not" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            // tinygrad implements logical_not as a Bool cast followed by
            // comparison to true, so retain its non-Bool input behavior.
            g.dtype(x)?;
            g.shape(x)?.numel()?;
            let boolean = g.cast(x, DType::Bool)?;
            g.logical_not(boolean)?
        }
        "IsInf" if ins.len() == 1 => {
            if attrs
                .keys()
                .any(|name| name != "detect_positive" && name != "detect_negative")
            {
                return Err(bad("unsupported IsInf attribute"));
            }
            let detect_positive = attrs
                .get("detect_positive")
                .map(|value| scalar_i64(value).map(|value| value != 0))
                .transpose()?
                .unwrap_or(true);
            let detect_negative = attrs
                .get("detect_negative")
                .map(|value| scalar_i64(value).map(|value| value != 0))
                .transpose()?
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
            g.dtype(lhs)?;
            g.dtype(rhs)?;
            let output_shape = g.shape(lhs)?.broadcast_with(g.shape(rhs)?)?;
            output_shape.numel()?;
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
            let false = g.full_with_dtype([], Scalar::Bool(false), DType::Bool)?;
            g.select(equal, lhs, false)?
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
        "Pow" if ins.len() == 2 && attrs.is_empty() => {
            // tinygrad's ONNX adapter restores an integer base dtype after
            // rounding the promoted power result. Fetch and validate both
            // operands before composing that post-processing so malformed
            // broadcasts cannot append any partial graph nodes.
            let base = get(0)?;
            let exponent = get(1)?;
            let base_dtype = g.dtype(base)?;
            g.shape(base)?.broadcast_with(g.shape(exponent)?)?;
            let value = g.pow(base, exponent)?;
            if base_dtype.is_integer() {
                g.cast(g.round(value)?, base_dtype)?
            } else {
                value
            }
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
        "Abs" if ins.len() == 1 && attrs.is_empty() => g.abs(get(0)?)?,
        "Neg" if ins.len() == 1 && attrs.is_empty() => g.neg(get(0)?)?,
        "LeakyRelu" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "alpha") {
                return Err(bad("unsupported LeakyRelu attribute"));
            }
            let alpha = attrs
                .get("alpha")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(0.01);
            if !alpha.is_finite() {
                return Err(bad("LeakyRelu alpha must be finite"));
            }
            let x = get(0)?;
            // Keep alpha at the local F32 scalar dtype. tinygrad's weak
            // floating scalar promotes integral inputs for this composition;
            // narrowing it to X would otherwise turn fractional slopes into
            // zero and return an integer result.
            let slope = g.constant(TensorData::scalar(alpha));
            g.leaky_relu(x, slope)?
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
            input_shape.numel()?;

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
            input_shape.numel()?;

            // A weak Python FLOAT comparison resolves at F32 unless the
            // source is F64.  The false Python integer literal is a separate
            // weak value: it preserves every non-Bool X dtype, but Bool plus
            // that literal resolves to tinygrad's default I32.
            let comparison_dtype = if input_dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            };
            let output_dtype = if input_dtype == DType::Bool {
                DType::I32
            } else {
                input_dtype
            };
            let scalar_shape = Shape::new([]);
            let comparison_shape = input_shape.broadcast_with(&scalar_shape)?;
            comparison_shape.numel()?;
            if comparison_dtype.promote(comparison_dtype) != comparison_dtype {
                return Err(bad("ThresholdedRelu comparison promotion mismatch"));
            }
            let branch_shape = input_shape.broadcast_with(&scalar_shape)?;
            branch_shape.numel()?;
            let output_shape = comparison_shape.broadcast_with(&branch_shape)?;
            output_shape.numel()?;
            if output_dtype.promote(output_dtype) != output_dtype {
                return Err(bad("ThresholdedRelu select promotion mismatch"));
            }

            let comparison_x = if input_dtype == comparison_dtype {
                x
            } else {
                g.cast(x, comparison_dtype)?
            };
            let alpha = g.constant(TensorData::scalar_with_dtype(
                Scalar::F(f64::from(alpha)),
                comparison_dtype,
            ));
            let condition = g.gt(comparison_x, alpha)?;
            let on_true = if input_dtype == output_dtype {
                x
            } else {
                g.cast(x, output_dtype)?
            };
            let zero = g.constant(TensorData::scalar_with_dtype(Scalar::I(0), output_dtype));
            g.select(condition, on_true, zero)?
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
            comparison_shape.numel()?;
            if comparison_dtype.promote(comparison_dtype) != comparison_dtype {
                return Err(bad("Binarizer comparison promotion mismatch"));
            }
            let output_shape = comparison_shape.broadcast_with(&scalar_shape)?;
            output_shape.numel()?;
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
            x_shape.numel()?;
            slope_shape.numel()?;
            let scaled_shape = x_shape.broadcast_with(&slope_shape)?;
            scaled_shape.numel()?;
            let output_dtype = prelu_dtype(x_dtype, slope_dtype);
            let scalar_shape = Shape::new([]);
            let condition_shape = x_shape.broadcast_with(&scalar_shape)?;
            condition_shape.numel()?;
            let output_shape = condition_shape.broadcast_with(&scaled_shape)?;
            output_shape.numel()?;
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
            let selected_x_dtype = if exceptional_promotion { DType::F32 } else { x_dtype };
            if scaled_dtype != output_dtype
                || selected_x_dtype.promote(scaled_dtype) != output_dtype
            {
                return Err(bad("PRelu promotion mismatch"));
            }

            let zero = g.constant(TensorData::scalar_with_dtype(Scalar::I(0), x_dtype));
            // tinygrad deliberately uses `X > 0`: zero and NaN take the
            // scaled branch, unlike Graph::leaky_relu's `< 0` helper.
            let condition = g.gt(x, zero)?;
            let (x_value, slope) = if exceptional_promotion {
                (g.cast(x, DType::F32)?, g.cast(slope, DType::F32)?)
            } else {
                (x, slope)
            };
            let scaled = g.mul(x_value, slope)?;
            g.select(condition, x_value, scaled)?
        }
        "Clip" if (1..=3).contains(&ins.len()) && attrs.is_empty() => {
            let x = get(0)?;
            let bound = |i: usize| -> Result<Option<NodeId>> {
                let Some(name) = ins.get(i).filter(|x| !x.is_empty()) else {
                    return Ok(None);
                };
                let data = constants
                    .get(*name)
                    .ok_or_else(|| bad("Clip bounds must be constant initializers"))?;
                if data.len() != 1 || data.dtype() != g.dtype(x)? {
                    return Err(bad("Clip bounds must be same-dtype scalar tensors"));
                }
                Ok(Some(get(i)?))
            };
            match (bound(1)?, bound(2)?) {
                (None, None) => x,
                (min, max) => g.clamp(x, min, max)?,
            }
        }
        "Dropout" if (1..=3).contains(&ins.len()) && attrs.is_empty() => {
            let x = get(0)?;
            if let Some(name) = ins.get(1).filter(|x| !x.is_empty()) {
                let value = constants
                    .get(*name)
                    .ok_or_else(|| bad("Dropout ratio must be constant"))?;
                if value.len() != 1
                    || !value.dtype().is_float()
                    || value.scalar_at(0).as_f64() != 0.0
                {
                    return Err(bad("only inference Dropout with zero ratio is supported"));
                }
            }
            if let Some(name) = ins.get(2).filter(|x| !x.is_empty()) {
                let value = constants
                    .get(*name)
                    .ok_or_else(|| bad("Dropout training_mode must be constant"))?;
                if value.len() != 1 || value.dtype() != DType::Bool || value.scalar_at(0).as_bool()
                {
                    return Err(bad(
                        "only inference Dropout with training_mode=false is supported",
                    ));
                }
            }
            x
        }
        "Shape" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "start" && x != "end") {
                return Err(bad("unsupported Shape attribute"));
            }
            let dims = g.shape(get(0)?)?.dims();
            let rank = i64::try_from(dims.len()).map_err(|_| bad("Shape rank overflow"))?;
            let start = attrs
                .get("start")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let end = attrs
                .get("end")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(rank);
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
            let data = TensorData::from_scalars(
                [values.len()],
                DType::I64,
                values,
            )?;
            constants.insert(outs[0].to_owned(), data.clone());
            g.constant(data)
        }
        "Size" if ins.len() == 1 && attrs.is_empty() => {
            let numel = g.shape(get(0)?)?.numel()?;
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
                let dim_i64 = i64::try_from(dim)
                    .map_err(|_| bad("Gather scalar axis extent exceeds I64"))?;
                let raw = data.scalar_at(0).as_i64();
                let index = if raw < 0 {
                    raw.checked_add(dim_i64)
                        .ok_or_else(|| bad("Gather scalar index is out of bounds"))?
                } else {
                    raw
                };
                if index < 0
                    || usize::try_from(index)
                        .ok()
                        .filter(|&i| i < dim)
                        .is_none()
                {
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
            let starts = const_i64(constants, ins[1])?;
            let ends = const_i64(constants, ins[2])?;
            if starts.len() != ends.len() {
                return Err(bad("Slice starts/ends length mismatch"));
            }
            let axes = if ins.len() >= 4 && !ins[3].is_empty() {
                const_i64(constants, ins[3])?
            } else {
                (0..starts.len()).map(|x| x as i64).collect()
            };
            let steps = if ins.len() == 5 && !ins[4].is_empty() {
                const_i64(constants, ins[4])?
            } else {
                vec![1; starts.len()]
            };
            if axes.len() != starts.len() || steps.len() != starts.len() {
                return Err(bad("Slice control lengths mismatch"));
            }
            let rank = g.shape(x)?.rank();
            let axes = axes_usize(&axes, rank)?;
            let mut seen = vec![false; rank];
            let mut slices = vec![
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1
                };
                rank
            ];
            for ((axis, start), (end, step)) in axes
                .into_iter()
                .zip(starts)
                .zip(ends.into_iter().zip(steps))
            {
                if step == 0 {
                    return Err(bad("Slice step must be nonzero"));
                }
                let step = isize::try_from(step).map_err(|_| bad("Slice step overflow"))?;
                let start = isize::try_from(start).map_err(|_| bad("Slice start overflow"))?;
                let end = isize::try_from(end).map_err(|_| bad("Slice end overflow"))?;
                if std::mem::replace(&mut seen[axis], true) {
                    return Err(bad("duplicate Slice axis"));
                }
                slices[axis] = crate::Slice {
                    start: Some(start),
                    stop: Some(end),
                    step,
                };
            }
            g.stride(x, slices)?
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
        op @ ("ReduceSum" | "ReduceMean" | "ReduceProd" | "ReduceMin" | "ReduceMax")
            if (1..=2).contains(&ins.len()) =>
        {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let plan = reduce_plan(g, x, &ins, &attrs, constants)?;
            if plan.noop {
                x
            } else {
                let kind = match op {
                    "ReduceSum" => ReduceKind::Sum,
                    "ReduceMean" => ReduceKind::Mean,
                    "ReduceProd" => ReduceKind::Product,
                    "ReduceMin" => ReduceKind::Min,
                    "ReduceMax" => ReduceKind::Max,
                    _ => unreachable!(),
                };
                g.reduce(x, kind, Some(plan.axes), plan.keepdims)?
            }
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
        op @ ("ArgMax" | "ArgMin") if ins.len() == 1 => {
            if attrs
                .keys()
                .any(|x| !matches!(x.as_str(), "axis" | "keepdims" | "select_last_index"))
            {
                return Err(bad("unsupported Arg attribute"));
            }
            if attrs
                .get("select_last_index")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0)
                != 0
            {
                return Err(bad("Arg select_last_index is unsupported"));
            }
            let x = get(0)?;
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let axis = axes_usize(&[axis], g.shape(x)?.rank())?[0] as isize;
            let keepdims = attrs
                .get("keepdims")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            if !matches!(keepdims, 0 | 1) {
                return Err(bad("Arg keepdims must be 0 or 1"));
            }
            let value = if op == "ArgMax" {
                g.argmax(x, Some(axis), keepdims == 1)?
            } else {
                g.argmin(x, Some(axis), keepdims == 1)?
            };
            g.cast(value, DType::I64)?
        }
        "BatchNormalization" if ins.len() == 5 => {
            if attrs.keys().any(|x| {
                !matches!(
                    x.as_str(),
                    "epsilon" | "training_mode" | "momentum" | "spatial"
                )
            }) {
                return Err(bad("unsupported BatchNormalization attribute"));
            }
            if attrs.contains_key("momentum") || attrs.contains_key("spatial") {
                return Err(bad(
                    "BatchNormalization momentum/spatial attributes are unsupported",
                ));
            }
            if attrs
                .get("training_mode")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0)
                != 0
            {
                return Err(bad("BatchNormalization training mode is unsupported"));
            }
            let epsilon = attrs
                .get("epsilon")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(1e-5);
            if !epsilon.is_finite() || epsilon < 0. {
                return Err(bad(
                    "BatchNormalization epsilon must be finite and nonnegative",
                ));
            }
            let x = get(0)?;
            let shape = g.shape(x)?.clone();
            let dtype = g.dtype(x)?;
            if shape.rank() < 2 || !dtype.is_float() {
                return Err(bad("BatchNormalization X must be a rank >= 2 float tensor"));
            }
            let channels = shape.dims()[1];
            let param_shape = Shape::new([channels]);
            let mut broadcast = vec![1; shape.rank()];
            broadcast[1] = channels;
            let params = [get(1)?, get(2)?, get(3)?, get(4)?];
            for param in params {
                if g.dtype(param)? != dtype || g.shape(param)? != &param_shape {
                    return Err(bad(
                        "BatchNormalization parameters must be same-dtype [C] tensors",
                    ));
                }
            }
            let scale = g.reshape(params[0], broadcast.clone())?;
            let bias = g.reshape(params[1], broadcast.clone())?;
            let mean = g.reshape(params[2], broadcast.clone())?;
            let variance = g.reshape(params[3], broadcast)?;
            let epsilon = g.constant(TensorData::scalar(epsilon));
            let epsilon = g.cast(epsilon, dtype)?;
            let centered = g.sub(x, mean)?;
            let variance = g.add(variance, epsilon)?;
            let inv_std = g.sqrt(variance)?;
            let normalized = g.div(centered, inv_std)?;
            let scaled = g.mul(normalized, scale)?;
            g.add(scaled, bias)?
        }
        "GlobalAveragePool" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            let plan = global_average_pool_plan(g, x)?;
            if plan.axes.is_empty() {
                if g.dtype(x)? == plan.output_dtype { x } else { g.cast(x, plan.output_dtype)? }
            } else {
                let summed = g.reduce_with_dtypes(
                    x,
                    ReduceKind::Sum,
                    Some(plan.axes),
                    true,
                    ReductionDType::new(plan.sum_dtypes.accumulator, plan.sum_dtypes.accumulator),
                )?;
                let summed = if plan.work_dtype == plan.sum_dtypes.accumulator { summed } else { g.cast(summed, plan.work_dtype)? };
                let average = g.div(summed, g.constant(plan.divisor))?;
                let output = if plan.output_dtype == plan.work_dtype { average } else { g.cast(average, plan.output_dtype)? };
                debug_assert_eq!(g.shape(output).expect("GlobalAveragePool shape preflighted"), &plan.output_shape);
                output
            }
        }
        "GlobalMaxPool" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            let plan = global_max_pool_plan(g, x)?;
            if plan.axes.is_empty() {
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
            }
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
