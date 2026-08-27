//! Graph lowering for the validated static ONNX subset.

use super::{
    bad,
    schema::{
        attrs, axes_usize, const_i64, conv_pads, conv_pair, conv_same_padding, onnx_pool_options,
        packed_i64, reshape_dims, scalar_f32, scalar_i64,
    },
    tensor::{onnx_dtype, tensor_data},
    wire::{Msg, var},
};
use crate::{Conv2dOptions, DType, Graph, NodeId, ReduceKind, Result, Scalar, Shape, TensorData};
use std::collections::BTreeMap;

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
        "Identity" if ins.len() == 1 => get(0)?,
        "Relu" if ins.len() == 1 => g.relu(get(0)?)?,
        "Sigmoid" if ins.len() == 1 => g.sigmoid(get(0)?)?,
        "Tanh" if ins.len() == 1 => g.tanh(get(0)?)?,
        "Add" if ins.len() == 2 => g.add(get(0)?, get(1)?)?,
        "Sub" if ins.len() == 2 => g.sub(get(0)?, get(1)?)?,
        "Mul" if ins.len() == 2 => g.mul(get(0)?, get(1)?)?,
        "Div" if ins.len() == 2 => g.div(get(0)?, get(1)?)?,
        "MatMul" if ins.len() == 2 => g.matmul(get(0)?, get(1)?)?,
        "Cast" if ins.len() == 1 && attrs.len() == 1 => {
            let x = attrs.get("to").ok_or_else(|| bad("Cast needs to"))?;
            let mut at = 0;
            g.cast(get(0)?, onnx_dtype(var(x, &mut at)?)?)?
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
        "Pow" if ins.len() == 2 && attrs.is_empty() => g.pow(get(0)?, get(1)?)?,
        "Sqrt" if ins.len() == 1 && attrs.is_empty() => g.sqrt(get(0)?)?,
        "Exp" if ins.len() == 1 && attrs.is_empty() => g.exp(get(0)?)?,
        "Log" if ins.len() == 1 && attrs.is_empty() => g.log(get(0)?)?,
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
            let dtype = g.dtype(x)?;
            let slope = g.constant(TensorData::scalar(alpha));
            let slope = g.cast(slope, dtype)?;
            g.leaky_relu(x, slope)?
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
            let start = attrs
                .get("start")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let end = attrs
                .get("end")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(dims.len() as i64);
            let normalize = |x: i64| -> Result<usize> {
                usize::try_from(if x < 0 { x + dims.len() as i64 } else { x })
                    .ok()
                    .filter(|&x| x <= dims.len())
                    .ok_or_else(|| bad("invalid Shape start/end"))
            };
            let (start, end) = (normalize(start)?, normalize(end)?);
            if start > end {
                return Err(bad("Shape start exceeds end"));
            }
            let data = TensorData::from_scalars(
                [end - start],
                DType::I64,
                dims[start..end].iter().map(|&x| Scalar::I(x as i64)),
            )?;
            constants.insert(outs[0].to_owned(), data.clone());
            g.constant(data)
        }
        "Expand" if ins.len() == 2 && attrs.is_empty() => {
            let x = get(0)?;
            let shape = const_i64(constants, ins[1])?
                .into_iter()
                .map(|x| usize::try_from(x).map_err(|_| bad("Expand shape must be nonnegative")))
                .collect::<Result<Vec<_>>>()?;
            g.expand(x, Shape::new(shape))?
        }
        "Tile" if ins.len() == 2 && attrs.is_empty() => {
            let x = get(0)?;
            let repeats = const_i64(constants, ins[1])?;
            if repeats.len() != g.shape(x)?.rank() || repeats.iter().any(|&x| x < 0) {
                return Err(bad("Tile repeats must be nonnegative and match rank"));
            }
            g.tile(
                x,
                &repeats.into_iter().map(|x| x as isize).collect::<Vec<_>>(),
            )?
        }
        "Gather" if ins.len() == 2 => {
            if attrs.keys().any(|x| x != "axis") {
                return Err(bad("unsupported Gather attribute"));
            }
            let x = get(0)?;
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let rank = g.shape(x)?.rank();
            let axis = axes_usize(&[axis], rank)?[0];
            let name = ins[1];
            let data = constants
                .get(name)
                .ok_or_else(|| bad("Gather indices must be constant"))?;
            if !matches!(data.dtype(), DType::I32 | DType::I64) || data.shape() != g.shape(x)? {
                return Err(bad("Gather requires same-rank constant I32/I64 indices"));
            }
            if (0..data.len()).any(|i| data.scalar_at(i).as_i64() < 0) {
                return Err(bad("Gather negative indices are unsupported"));
            }
            g.gather(x, get(1)?, axis)?
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
            let pads = const_i64(constants, ins[1])?;
            if pads.len() != 2 * rank {
                return Err(bad("Pad pads must contain begin/end values for every axis"));
            }
            if pads.iter().any(|&x| x < 0) {
                return Err(bad("negative ONNX Pad cropping is unsupported"));
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
                .map(|i| {
                    Ok((
                        usize::try_from(pads[i]).map_err(|_| bad("Pad overflow"))?,
                        usize::try_from(pads[rank + i]).map_err(|_| bad("Pad overflow"))?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            g.pad(x, padding, fill)?
        }
        "ConstantOfShape" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "value") {
                return Err(bad("unsupported ConstantOfShape attribute"));
            }
            let dims = const_i64(constants, ins[0])?
                .into_iter()
                .map(|x| {
                    usize::try_from(x)
                        .map_err(|_| bad("ConstantOfShape dimensions must be nonnegative"))
                })
                .collect::<Result<Vec<_>>>()?;
            let (value, dtype) = match attrs.get("value") {
                Some(bytes) => {
                    let value = tensor_data(Msg::new(bytes))?;
                    if value.len() != 1 {
                        return Err(bad("ConstantOfShape value must contain one element"));
                    }
                    (value.scalar_at(0), value.dtype())
                }
                None => (Scalar::F(0.0), DType::F32),
            };
            g.full_with_dtype(Shape::new(dims), value, dtype)?
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
            if axes.is_empty() && noop == 1 {
                x
            } else {
                let rank = g.shape(x)?.rank();
                let axes = if axes.is_empty() {
                    (0..rank).map(|x| x as isize).collect()
                } else {
                    let axes = axes_usize(&axes, rank)?;
                    if axes
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        != axes.len()
                    {
                        return Err(bad("duplicate Reduce axis"));
                    }
                    axes.into_iter().map(|x| x as isize).collect()
                };
                let kind = match op {
                    "ReduceSum" => ReduceKind::Sum,
                    "ReduceMean" => ReduceKind::Mean,
                    "ReduceProd" => ReduceKind::Product,
                    "ReduceMin" => ReduceKind::Min,
                    "ReduceMax" => ReduceKind::Max,
                    _ => unreachable!(),
                };
                g.reduce(x, kind, Some(axes), keepdims == 1)?
            }
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
