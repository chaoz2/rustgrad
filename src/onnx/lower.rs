//! Graph lowering for the validated static ONNX subset.

use super::{
    bad,
    schema::{
        attrs, axes_usize, const_i64, conv_pads, conv_pair, conv_same_padding, onnx_pool_options,
        packed_i64, reshape_dims, scalar_f32, scalar_i64, typed_scalar_f32_attr,
    },
    tensor::{onnx_dtype, tensor_data},
    wire::{Msg, var},
};
use crate::{
    ir::reduction_shape, Conv2dOptions, DType, Graph, NodeId, ReduceKind, ReductionDType,
    Result, Scalar, Shape, TensorData,
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
            let rank = g.shape(x)?.rank();
            if rank < 3 || !g.dtype(x)?.is_float() {
                return Err(bad("GlobalAveragePool requires a rank >= 3 float tensor"));
            }
            g.reduce(
                x,
                ReduceKind::Mean,
                Some((2..rank).map(|x| x as isize).collect()),
                true,
            )?
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
