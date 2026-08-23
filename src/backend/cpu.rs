use super::Backend;
use crate::index::DenseIndex;
use crate::{
    BinaryOp, CompareOp, DType, Error, Graph, LogicalOp, NodeId, Op, Result, Scalar, Shape,
    TensorData, UnaryOp, ir::normalized_slice,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn execute(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<TensorData> {
        graph.node(output)?;
        let mut values: Vec<TensorData> = Vec::with_capacity(output.index() + 1);
        for node in &graph.nodes[..=output.index()] {
            let value = match &node.op {
                Op::Input { name } => {
                    let value = inputs
                        .get(name)
                        .ok_or_else(|| Error::MissingInput(name.clone()))?;
                    if value.shape() != &node.shape {
                        return Err(Error::InputShape {
                            name: name.clone(),
                            expected: node.shape.clone(),
                            actual: value.shape().clone(),
                        });
                    }
                    if value.dtype() != node.dtype {
                        return Err(Error::InputDType {
                            name: name.clone(),
                            expected: node.dtype,
                            actual: value.dtype(),
                        });
                    }
                    value.clone()
                }
                Op::Constant(data) => data.clone(),
                Op::Cast { input, dtype } => values[input.index()].cast(*dtype),
                Op::Unary { op, input } => unary(&values[input.index()], *op)?,
                Op::Binary { op, lhs, rhs } => binary(&values, *lhs, *rhs, &node.shape, *op)?,
                Op::Compare { op, lhs, rhs } => compare(&values, *lhs, *rhs, &node.shape, *op)?,
                Op::Logical { op, lhs, rhs } => logical(&values, *lhs, *rhs, &node.shape, *op)?,
                Op::Select {
                    condition,
                    on_true,
                    on_false,
                } => select(
                    &values,
                    *condition,
                    *on_true,
                    *on_false,
                    &node.shape,
                    node.dtype,
                )?,
                Op::Reduce {
                    input,
                    kind,
                    axes,
                    keepdim,
                } => reduce(&values[input.index()], *kind, axes, *keepdim, node.dtype)?,
                Op::ArgReduce {
                    input,
                    max,
                    axis,
                    keepdim,
                } => arg_reduce(&values[input.index()], *max, *axis, *keepdim)?,
                Op::ReduceGrad {
                    input,
                    upstream,
                    kind,
                    axes,
                    keepdim,
                } => reduce_grad(
                    &values[input.index()],
                    &values[upstream.index()],
                    *kind,
                    axes,
                    *keepdim,
                )?,
                Op::SumTo { input, shape } => sum_to(&values[input.index()], shape)?,
                Op::Reshape { input, shape } => TensorData::from_scalars(
                    shape.clone(),
                    values[input.index()].dtype(),
                    (0..values[input.index()].len()).map(|i| values[input.index()].scalar_at(i)),
                )?,
                Op::Permute { input, axes } => permute(&values[input.index()], axes)?,
                Op::Expand { input, shape } => expand(&values[input.index()], shape)?,
                Op::Shrink { input, bounds } => shrink(&values[input.index()], bounds)?,
                Op::Pad {
                    input,
                    padding,
                    fill,
                } => pad(&values[input.index()], padding, *fill)?,
                Op::Stride { input, slices } => stride(&values[input.index()], slices)?,
                Op::Concat { inputs, axis } => {
                    concat(&values, inputs, *axis, &node.shape, node.dtype)?
                }
                Op::Scatter {
                    base,
                    index,
                    updates,
                    axis,
                    add,
                } => indexed_scatter(
                    &values[base.index()],
                    &values[index.index()],
                    &values[updates.index()],
                    *axis,
                    *add,
                    node.dtype,
                )?,
                Op::ScatterPositions {
                    input,
                    shape,
                    starts,
                    steps,
                } => scatter(&values[input.index()], shape, starts, steps)?,
                Op::Gather { input, index, axis } => {
                    gather(&values[input.index()], &values[index.index()], *axis)?
                }
                Op::MaskedSelect {
                    input,
                    mask,
                    size,
                    fill,
                } => masked_select(&values[input.index()], &values[mask.index()], *size, *fill)?,
                Op::Matmul { lhs, rhs } => matmul(&values[lhs.index()], &values[rhs.index()])?,
                Op::MatmulGrad {
                    upstream,
                    lhs,
                    rhs,
                    lhs_gradient,
                } => matmul_grad(
                    &values[upstream.index()],
                    &values[lhs.index()],
                    &values[rhs.index()],
                    *lhs_gradient,
                )?,
            };
            debug_assert_eq!(value.shape(), &node.shape);
            values.push(value);
        }
        values.pop().ok_or(Error::UnknownNode(output))
    }
}

fn binary(
    values: &[TensorData],
    lhs: NodeId,
    rhs: NodeId,
    output_shape: &Shape,
    op: BinaryOp,
) -> Result<TensorData> {
    let lhs = values.get(lhs.index()).ok_or(Error::UnknownNode(lhs))?;
    let rhs = values.get(rhs.index()).ok_or(Error::UnknownNode(rhs))?;
    let output_len = output_shape.numel()?;
    let dtype = lhs.dtype().promote(rhs.dtype());
    let mut data = Vec::with_capacity(output_len);
    for linear in 0..output_len {
        let lhs_offset = broadcast_offset(linear, output_shape, lhs.shape());
        let rhs_offset = broadcast_offset(linear, output_shape, rhs.shape());
        data.push(binary_scalar(
            lhs.scalar_at(lhs_offset),
            rhs.scalar_at(rhs_offset),
            dtype,
            op,
        ));
    }
    TensorData::from_scalars(output_shape.clone(), dtype, data)
}

fn compare(
    values: &[TensorData],
    lhs: NodeId,
    rhs: NodeId,
    output_shape: &Shape,
    op: CompareOp,
) -> Result<TensorData> {
    let lhs = values.get(lhs.index()).ok_or(Error::UnknownNode(lhs))?;
    let rhs = values.get(rhs.index()).ok_or(Error::UnknownNode(rhs))?;
    let data = (0..output_shape.numel()?).map(|linear| {
        let lhs = lhs.scalar_at(broadcast_offset(linear, output_shape, lhs.shape()));
        let rhs = rhs.scalar_at(broadcast_offset(linear, output_shape, rhs.shape()));
        Scalar::Bool(compare_scalar(lhs, rhs, op))
    });
    TensorData::from_scalars(output_shape.clone(), DType::Bool, data)
}

fn compare_scalar(lhs: Scalar, rhs: Scalar, op: CompareOp) -> bool {
    use std::cmp::Ordering;
    let ordering = match (lhs, rhs) {
        (Scalar::F(lhs), rhs) => lhs.partial_cmp(&rhs.as_f64()),
        (lhs, Scalar::F(rhs)) => lhs.as_f64().partial_cmp(&rhs),
        (Scalar::I(lhs), Scalar::I(rhs)) => Some(lhs.cmp(&rhs)),
        (Scalar::U(lhs), Scalar::U(rhs)) => Some(lhs.cmp(&rhs)),
        (Scalar::I(lhs), Scalar::U(rhs)) => {
            if lhs < 0 {
                Some(Ordering::Less)
            } else {
                Some((lhs as u64).cmp(&rhs))
            }
        }
        (Scalar::U(lhs), Scalar::I(rhs)) => {
            if rhs < 0 {
                Some(Ordering::Greater)
            } else {
                Some(lhs.cmp(&(rhs as u64)))
            }
        }
        (Scalar::Bool(lhs), Scalar::Bool(rhs)) => Some(lhs.cmp(&rhs)),
        (Scalar::Bool(lhs), rhs) => Some((lhs as u8 as i64).cmp(&rhs.as_i64())),
        (lhs, Scalar::Bool(rhs)) => Some(lhs.as_i64().cmp(&(rhs as u8 as i64))),
    };
    match op {
        CompareOp::Eq => ordering == Some(Ordering::Equal),
        CompareOp::Ne => ordering != Some(Ordering::Equal),
        CompareOp::Lt => ordering == Some(Ordering::Less),
        CompareOp::Le => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
        CompareOp::Gt => ordering == Some(Ordering::Greater),
        CompareOp::Ge => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
    }
}

fn logical(
    values: &[TensorData],
    lhs: NodeId,
    rhs: Option<NodeId>,
    output_shape: &Shape,
    op: LogicalOp,
) -> Result<TensorData> {
    let lhs = values.get(lhs.index()).ok_or(Error::UnknownNode(lhs))?;
    let rhs = rhs
        .map(|id| values.get(id.index()).ok_or(Error::UnknownNode(id)))
        .transpose()?;
    let data = (0..output_shape.numel()?).map(|linear| {
        let lhs = lhs
            .scalar_at(broadcast_offset(linear, output_shape, lhs.shape()))
            .as_bool();
        let value = match (op, rhs) {
            (LogicalOp::Not, None) => !lhs,
            (LogicalOp::And, Some(rhs)) => {
                lhs && rhs
                    .scalar_at(broadcast_offset(linear, output_shape, rhs.shape()))
                    .as_bool()
            }
            (LogicalOp::Or, Some(rhs)) => {
                lhs || rhs
                    .scalar_at(broadcast_offset(linear, output_shape, rhs.shape()))
                    .as_bool()
            }
            _ => unreachable!("graph validates logical operands"),
        };
        Scalar::Bool(value)
    });
    TensorData::from_scalars(output_shape.clone(), DType::Bool, data)
}

fn select(
    values: &[TensorData],
    condition: NodeId,
    on_true: NodeId,
    on_false: NodeId,
    output_shape: &Shape,
    dtype: DType,
) -> Result<TensorData> {
    let condition = values
        .get(condition.index())
        .ok_or(Error::UnknownNode(condition))?;
    let on_true = values
        .get(on_true.index())
        .ok_or(Error::UnknownNode(on_true))?;
    let on_false = values
        .get(on_false.index())
        .ok_or(Error::UnknownNode(on_false))?;
    let data = (0..output_shape.numel()?).map(|linear| {
        let condition = condition
            .scalar_at(broadcast_offset(linear, output_shape, condition.shape()))
            .as_bool();
        if condition {
            on_true.scalar_at(broadcast_offset(linear, output_shape, on_true.shape()))
        } else {
            on_false.scalar_at(broadcast_offset(linear, output_shape, on_false.shape()))
        }
    });
    TensorData::from_scalars(output_shape.clone(), dtype, data)
}

fn binary_scalar(lhs: Scalar, rhs: Scalar, dtype: DType, op: BinaryOp) -> Scalar {
    if dtype.is_float() {
        let (lhs, rhs) = (lhs.as_f64(), rhs.as_f64());
        return Scalar::F(match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        });
    }
    if matches!(dtype, DType::Bool) {
        let (lhs, rhs) = (lhs.as_bool(), rhs.as_bool());
        return Scalar::Bool(match op {
            BinaryOp::Add => lhs || rhs,
            BinaryOp::Sub => lhs ^ rhs,
            BinaryOp::Mul => lhs && rhs,
            BinaryOp::Div => lhs && rhs,
        });
    }
    if matches!(dtype.category(), crate::DTypeCategory::Unsigned) {
        let (lhs, rhs) = (lhs.as_u64(), rhs.as_u64());
        return Scalar::U(match op {
            BinaryOp::Add => lhs.wrapping_add(rhs),
            BinaryOp::Sub => lhs.wrapping_sub(rhs),
            BinaryOp::Mul => lhs.wrapping_mul(rhs),
            BinaryOp::Div => lhs / rhs,
        });
    }
    let (lhs, rhs) = (lhs.as_i64(), rhs.as_i64());
    Scalar::I(match op {
        BinaryOp::Add => lhs.wrapping_add(rhs),
        BinaryOp::Sub => lhs.wrapping_sub(rhs),
        BinaryOp::Mul => lhs.wrapping_mul(rhs),
        BinaryOp::Div => lhs / rhs,
    })
}

fn unary(input: &TensorData, op: UnaryOp) -> Result<TensorData> {
    let values = (0..input.len()).map(|index| {
        let value = input.scalar_at(index).as_f64();
        Scalar::F(match op {
            UnaryOp::Neg => -value,
            UnaryOp::Exp => value.exp(),
            UnaryOp::Log => value.ln(),
            UnaryOp::Relu => value.max(0.0),
            UnaryOp::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        })
    });
    TensorData::from_scalars(input.shape().clone(), input.dtype(), values)
}

fn reduce(
    input: &TensorData,
    kind: crate::ReduceKind,
    axes: &[usize],
    keepdim: bool,
    dtype: DType,
) -> Result<TensorData> {
    let dims = input.shape().dims();
    let output_shape = Shape::new(
        dims.iter()
            .enumerate()
            .filter_map(|(i, d)| {
                if axes.contains(&i) {
                    keepdim.then_some(1)
                } else {
                    Some(*d)
                }
            })
            .collect::<Vec<_>>(),
    );
    let input_index = DenseIndex::new(input.shape().clone())?;
    let output_index = DenseIndex::new(output_shape.clone())?;
    let identity = match kind {
        crate::ReduceKind::Sum | crate::ReduceKind::Mean => Scalar::I(0),
        crate::ReduceKind::Product => Scalar::I(1),
        crate::ReduceKind::Max => Scalar::F(f64::NEG_INFINITY),
        crate::ReduceKind::Min => Scalar::F(f64::INFINITY),
    };
    let mut out = vec![identity; output_index.len()];
    let mut counts = vec![0usize; output_index.len()];
    for linear in 0..input_index.len() {
        let coords = input_index.coords(linear)?;
        let oc = coords
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                if axes.contains(&i) {
                    keepdim.then_some(0)
                } else {
                    Some(*c)
                }
            })
            .collect::<Vec<_>>();
        let o = output_index.offset(&oc)?;
        let v = input.scalar_at(linear);
        counts[o] += 1;
        out[o] = match kind {
            crate::ReduceKind::Sum | crate::ReduceKind::Mean => {
                binary_scalar(out[o], v, dtype, BinaryOp::Add)
            }
            crate::ReduceKind::Product => binary_scalar(out[o], v, dtype, BinaryOp::Mul),
            crate::ReduceKind::Max => {
                if !v.as_f64().is_nan() && v.as_f64() > out[o].as_f64() {
                    v
                } else {
                    out[o]
                }
            }
            crate::ReduceKind::Min => {
                if !v.as_f64().is_nan() && v.as_f64() < out[o].as_f64() {
                    v
                } else {
                    out[o]
                }
            }
        };
    }
    if matches!(kind, crate::ReduceKind::Mean) {
        for (v, c) in out.iter_mut().zip(counts) {
            *v = Scalar::F(v.as_f64() / c as f64);
        }
    }
    TensorData::from_scalars(output_shape, dtype, out)
}
fn reduce_grad(
    input: &TensorData,
    upstream: &TensorData,
    kind: crate::ReduceKind,
    axes: &[usize],
    keepdim: bool,
) -> Result<TensorData> {
    let ii = DenseIndex::new(input.shape().clone())?;
    let mut out = vec![Scalar::I(0); ii.len()];
    let reduced = reduce(input, kind, axes, true, input.dtype())?;
    let ri = DenseIndex::new(reduced.shape().clone())?;
    let mut zero = vec![0usize; ri.len()];
    let mut nonzero = vec![Scalar::I(1); ri.len()];
    for l in 0..ii.len() {
        let c = ii.coords(l)?;
        let rc = c
            .iter()
            .enumerate()
            .map(|(i, x)| if axes.contains(&i) { 0 } else { *x })
            .collect::<Vec<_>>();
        let r = ri.offset(&rc)?;
        let v = input.scalar_at(l);
        if v.as_f64() == 0. {
            zero[r] += 1
        } else {
            nonzero[r] = binary_scalar(nonzero[r], v, input.dtype(), BinaryOp::Mul);
        }
    }
    for (l, slot) in out.iter_mut().enumerate() {
        let c = ii.coords(l)?;
        let rc = c
            .iter()
            .enumerate()
            .map(|(i, x)| if axes.contains(&i) { 0 } else { *x })
            .collect::<Vec<_>>();
        let r = ri.offset(&rc)?;
        let uc = if keepdim {
            rc.clone()
        } else {
            c.iter()
                .enumerate()
                .filter_map(|(i, x)| (!axes.contains(&i)).then_some(*x))
                .collect()
        };
        let u = upstream.scalar_at(DenseIndex::new(upstream.shape().clone())?.offset(&uc)?);
        let v = input.scalar_at(l);
        *slot = match kind {
            crate::ReduceKind::Product => {
                if zero[r] == 0 {
                    Scalar::F(u.as_f64() * reduced.scalar_at(r).as_f64() / v.as_f64())
                } else if zero[r] == 1 && v.as_f64() == 0. {
                    Scalar::F(u.as_f64() * nonzero[r].as_f64())
                } else {
                    Scalar::F(0.)
                }
            }
            crate::ReduceKind::Max | crate::ReduceKind::Min => {
                let val = reduced.scalar_at(r).as_f64();
                if val.is_nan() {
                    // tinygrad's equality mask is all-false for NaN, then
                    // divides that mask by its zero tie-count: every member
                    // of the reduction group receives NaN.
                    Scalar::F(f64::NAN)
                } else {
                    let ties = (0..ii.len())
                        .filter(|q| {
                            let qc = ii.coords(*q).unwrap();
                            qc.iter()
                                .enumerate()
                                .all(|(i, x)| axes.contains(&i) || *x == c[i])
                                && input.scalar_at(*q).as_f64() == val
                        })
                        .count();
                    if v.as_f64() == val {
                        Scalar::F(u.as_f64() / ties as f64)
                    } else {
                        Scalar::F(0.)
                    }
                }
            }
            _ => Scalar::F(0.),
        };
    }
    TensorData::from_scalars(input.shape().clone(), upstream.dtype(), out)
}
fn arg_reduce(
    input: &TensorData,
    max: bool,
    axis: Option<usize>,
    keepdim: bool,
) -> Result<TensorData> {
    let axes: Vec<_> = axis.map_or_else(|| (0..input.shape().rank()).collect(), |a| vec![a]);
    let output_shape = Shape::new(
        input
            .shape()
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                if axes.contains(&i) {
                    keepdim.then_some(1)
                } else {
                    Some(*d)
                }
            })
            .collect::<Vec<_>>(),
    );
    let ii = DenseIndex::new(input.shape().clone())?;
    let oi = DenseIndex::new(output_shape.clone())?;
    let mut values = vec![Scalar::I(0); oi.len()];
    let mut best = vec![None::<f64>; oi.len()];
    for linear in 0..ii.len() {
        let c = ii.coords(linear)?;
        let oc = c
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if axes.contains(&i) {
                    keepdim.then_some(0)
                } else {
                    Some(*x)
                }
            })
            .collect::<Vec<_>>();
        let o = oi.offset(&oc)?;
        let v = input.scalar_at(linear).as_f64();
        if best[o].is_none()
            || if max {
                v > best[o].unwrap()
            } else {
                v < best[o].unwrap()
            }
        {
            best[o] = Some(v);
            values[o] = Scalar::I(axis.map_or(linear, |a| c[a]) as i64);
        }
    }
    TensorData::from_scalars(output_shape, DType::I32, values)
}

fn expand(input: &TensorData, output_shape: &Shape) -> Result<TensorData> {
    let output: Vec<_> = (0..output_shape.numel()?)
        .map(|linear| input.scalar_at(broadcast_offset(linear, output_shape, input.shape())))
        .collect();
    TensorData::from_scalars(output_shape.clone(), input.dtype(), output)
}

fn sum_to(input: &TensorData, output_shape: &Shape) -> Result<TensorData> {
    let input_shape = input.shape();
    let input_strides = input_shape.contiguous_strides();
    let output_strides = output_shape.contiguous_strides();
    let padding = input_shape.rank() - output_shape.rank();
    let mut output = vec![Scalar::I(0); output_shape.numel()?];
    for linear in 0..input.len() {
        let mut output_offset = 0;
        for (output_axis, output_stride) in output_strides.iter().enumerate() {
            let input_axis = output_axis + padding;
            let coordinate = (linear / input_strides[input_axis]) % input_shape.dims()[input_axis];
            if output_shape.dims()[output_axis] != 1 {
                output_offset += coordinate * output_stride;
            }
        }
        output[output_offset] = binary_scalar(
            output[output_offset],
            input.scalar_at(linear),
            input.dtype(),
            BinaryOp::Add,
        );
    }
    TensorData::from_scalars(output_shape.clone(), input.dtype(), output)
}

#[allow(unreachable_code)]
fn broadcast_offset(linear: usize, output_shape: &Shape, input_shape: &Shape) -> usize {
    // Graph construction already validated broadcast compatibility; this shared
    // index map centralizes row-major coordinate handling for future views.
    let output = DenseIndex::new(output_shape.clone()).expect("validated shape");
    let input = DenseIndex::new(input_shape.clone()).expect("validated shape");
    debug_assert!(linear < output.len());
    let coords = output.coords(linear).expect("in-bounds output offset");
    return input
        .broadcast_offset(&output, &coords)
        .expect("validated broadcast");
    let output_strides = output_shape.contiguous_strides();
    let input_strides = input_shape.contiguous_strides();
    let padding = output_shape.rank() - input_shape.rank();
    output_strides
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis >= padding)
        .map(|(axis, output_stride)| {
            let input_axis = axis - padding;
            if input_shape.dims()[input_axis] == 1 {
                0
            } else {
                let coordinate = (linear / output_stride) % output_shape.dims()[axis];
                coordinate * input_strides[input_axis]
            }
        })
        .sum()
}

fn permute(input: &TensorData, axes: &[usize]) -> Result<TensorData> {
    let output_shape = Shape::new(
        axes.iter()
            .map(|axis| input.shape().dims()[*axis])
            .collect::<Vec<_>>(),
    );
    let output_strides = output_shape.contiguous_strides();
    let input_strides = input.shape().contiguous_strides();
    let mut output = vec![Scalar::I(0); input.len()];
    for (linear, slot) in output.iter_mut().enumerate() {
        let input_offset = axes
            .iter()
            .enumerate()
            .map(|(output_axis, input_axis)| {
                let coordinate =
                    (linear / output_strides[output_axis]) % output_shape.dims()[output_axis];
                coordinate * input_strides[*input_axis]
            })
            .sum::<usize>();
        *slot = input.scalar_at(input_offset);
    }
    TensorData::from_scalars(output_shape, input.dtype(), output)
}

fn shrink(input: &TensorData, bounds: &[(usize, usize)]) -> Result<TensorData> {
    let output_shape = Shape::new(
        bounds
            .iter()
            .map(|(start, end)| end - start)
            .collect::<Vec<_>>(),
    );
    let source_index = DenseIndex::new(input.shape().clone())?;
    let output_index = DenseIndex::new(output_shape.clone())?;
    let values = (0..output_index.len())
        .map(|linear| {
            let coords = output_index.coords(linear)?;
            let source = coords
                .iter()
                .zip(bounds)
                .map(|(coord, (start, _))| coord + start)
                .collect::<Vec<_>>();
            Ok(input.scalar_at(source_index.offset(&source)?))
        })
        .collect::<Result<Vec<_>>>()?;
    TensorData::from_scalars(output_shape, input.dtype(), values)
}

fn pad(input: &TensorData, padding: &[(usize, usize)], fill: Scalar) -> Result<TensorData> {
    let dims = input
        .shape()
        .dims()
        .iter()
        .zip(padding)
        .map(|(dim, (before, after))| {
            dim.checked_add(*before)
                .and_then(|x| x.checked_add(*after))
                .ok_or_else(|| Error::ShapeOverflow(input.shape().clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    let output_shape = Shape::new(dims);
    let source_index = DenseIndex::new(input.shape().clone())?;
    let output_index = DenseIndex::new(output_shape.clone())?;
    let values = (0..output_index.len())
        .map(|linear| {
            let coords = output_index.coords(linear)?;
            let inside =
                coords.iter().zip(padding).zip(input.shape().dims()).all(
                    |((coord, (before, _)), dim)| *coord >= *before && *coord - *before < *dim,
                );
            if !inside {
                Ok(fill)
            } else {
                let source = coords
                    .iter()
                    .zip(padding)
                    .map(|(coord, (before, _))| coord - before)
                    .collect::<Vec<_>>();
                Ok(input.scalar_at(source_index.offset(&source)?))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    TensorData::from_scalars(output_shape, input.dtype(), values)
}

fn stride(input: &TensorData, slices: &[crate::Slice]) -> Result<TensorData> {
    let normalized = slices
        .iter()
        .zip(input.shape().dims())
        .enumerate()
        .map(|(axis, (slice, dim))| normalized_slice(*dim, *slice, axis))
        .collect::<Result<Vec<_>>>()?;
    let output_shape = Shape::new(
        normalized
            .iter()
            .map(|(_, _, _, length)| *length)
            .collect::<Vec<_>>(),
    );
    let source_index = DenseIndex::new(input.shape().clone())?;
    let output_index = DenseIndex::new(output_shape.clone())?;
    let values = (0..output_index.len())
        .map(|linear| {
            let coords = output_index.coords(linear)?;
            let source = coords
                .iter()
                .zip(&normalized)
                .map(|(coord, (start, _, step, _))| {
                    usize::try_from(
                        start
                            .checked_add(
                                isize::try_from(*coord)
                                    .ok()
                                    .and_then(|n| n.checked_mul(*step))
                                    .ok_or(Error::InvalidIndex)?,
                            )
                            .ok_or(Error::InvalidIndex)?,
                    )
                    .map_err(|_| Error::InvalidIndex)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(input.scalar_at(source_index.offset(&source)?))
        })
        .collect::<Result<Vec<_>>>()?;
    TensorData::from_scalars(output_shape, input.dtype(), values)
}

fn concat(
    values: &[TensorData],
    inputs: &[NodeId],
    axis: usize,
    output_shape: &Shape,
    dtype: DType,
) -> Result<TensorData> {
    let tensors = inputs
        .iter()
        .map(|id| values.get(id.index()).ok_or(Error::UnknownNode(*id)))
        .collect::<Result<Vec<_>>>()?;
    let output_index = DenseIndex::new(output_shape.clone())?;
    let mut ends = Vec::with_capacity(tensors.len());
    let mut total = 0usize;
    for tensor in &tensors {
        total = total
            .checked_add(tensor.shape().dims()[axis])
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        ends.push(total);
    }
    let data = (0..output_index.len())
        .map(|linear| {
            let mut coords = output_index.coords(linear)?;
            let tensor_index = ends
                .iter()
                .position(|end| coords[axis] < *end)
                .ok_or(Error::InvalidIndex)?;
            let prior = if tensor_index == 0 {
                0
            } else {
                ends[tensor_index - 1]
            };
            coords[axis] -= prior;
            let index = DenseIndex::new(tensors[tensor_index].shape().clone())?;
            Ok(tensors[tensor_index].scalar_at(index.offset(&coords)?))
        })
        .collect::<Result<Vec<_>>>()?;
    TensorData::from_scalars(output_shape.clone(), dtype, data)
}

fn scatter(
    input: &TensorData,
    output_shape: &Shape,
    starts: &[isize],
    steps: &[isize],
) -> Result<TensorData> {
    let input_index = DenseIndex::new(input.shape().clone())?;
    let output_index = DenseIndex::new(output_shape.clone())?;
    let mut output = vec![Scalar::I(0); output_index.len()];
    for linear in 0..input_index.len() {
        let coords = input_index.coords(linear)?;
        let destination = coords
            .iter()
            .zip(starts)
            .zip(steps)
            .map(|((coord, start), step)| {
                let scaled = isize::try_from(*coord)
                    .ok()
                    .and_then(|x| x.checked_mul(*step))
                    .ok_or(Error::InvalidIndex)?;
                usize::try_from(start.checked_add(scaled).ok_or(Error::InvalidIndex)?)
                    .map_err(|_| Error::InvalidIndex)
            })
            .collect::<Result<Vec<_>>>()?;
        let offset = output_index.offset(&destination)?;
        output[offset] = input.scalar_at(linear);
    }
    TensorData::from_scalars(output_shape.clone(), input.dtype(), output)
}

/// Maps every index coordinate to its source/destination coordinate. Gather
/// and both scatter variants share this checked row-major mapping.
fn indexed_coordinates(
    input: &TensorData,
    index: &TensorData,
    axis: usize,
) -> Result<Vec<(Vec<usize>, usize)>> {
    let input_index = DenseIndex::new(input.shape().clone())?;
    let index_index = DenseIndex::new(index.shape().clone())?;
    let mut mapped = Vec::with_capacity(index.len());
    for linear in 0..index_index.len() {
        let mut coords = index_index.coords(linear)?;
        let selected = integer_index(index.scalar_at(linear), axis, input.shape().dims()[axis])?;
        coords[axis] = selected;
        input_index.offset(&coords)?;
        mapped.push((coords, linear));
    }
    Ok(mapped)
}

fn integer_index(value: Scalar, axis: usize, dim: usize) -> Result<usize> {
    let value = match value {
        Scalar::I(value) => value,
        Scalar::U(value) => i64::try_from(value).map_err(|_| Error::IndexOutOfBounds {
            axis,
            index: i64::MAX,
            dim,
        })?,
        _ => {
            return Err(Error::InvalidIndexDType {
                op: "indexed operation",
                actual: DType::Bool,
            });
        }
    };
    let index = usize::try_from(value).map_err(|_| Error::IndexOutOfBounds {
        axis,
        index: value,
        dim,
    })?;
    if index >= dim {
        return Err(Error::IndexOutOfBounds {
            axis,
            index: value,
            dim,
        });
    }
    Ok(index)
}

fn gather(input: &TensorData, index: &TensorData, axis: usize) -> Result<TensorData> {
    let map = indexed_coordinates(input, index, axis)?;
    let source_index = DenseIndex::new(input.shape().clone())?;
    let mut values = vec![Scalar::I(0); index.len()];
    for (coords, linear) in map {
        values[linear] = input.scalar_at(source_index.offset(&coords)?);
    }
    TensorData::from_scalars(index.shape().clone(), input.dtype(), values)
}

fn indexed_scatter(
    base: &TensorData,
    index: &TensorData,
    updates: &TensorData,
    axis: usize,
    add: bool,
    dtype: DType,
) -> Result<TensorData> {
    let base_index = DenseIndex::new(base.shape().clone())?;
    let update_index = DenseIndex::new(updates.shape().clone())?;
    let mut output = (0..base.len())
        .map(|linear| base.scalar_at(linear))
        .collect::<Vec<_>>();
    for (destination, update_linear) in indexed_coordinates(base, index, axis)? {
        let update_coords = DenseIndex::new(index.shape().clone())?.coords(update_linear)?;
        let update_value = updates.scalar_at(update_index.offset(&update_coords)?);
        let destination = base_index.offset(&destination)?;
        output[destination] = if add {
            binary_scalar(output[destination], update_value, dtype, BinaryOp::Add)
        } else {
            update_value
        };
    }
    TensorData::from_scalars(base.shape().clone(), dtype, output)
}

fn masked_select(
    input: &TensorData,
    mask: &TensorData,
    size: usize,
    fill: Scalar,
) -> Result<TensorData> {
    let input_index = DenseIndex::new(input.shape().clone())?;
    let mask_index = DenseIndex::new(mask.shape().clone())?;
    let mut output = Vec::with_capacity(size);
    for linear in 0..input_index.len() {
        let coords = input_index.coords(linear)?;
        let mask_offset = mask_index.broadcast_offset(&input_index, &coords)?;
        if mask.scalar_at(mask_offset).as_bool() && output.len() < size {
            output.push(input.scalar_at(linear));
        }
    }
    output.resize(size, fill);
    TensorData::from_scalars([size], input.dtype(), output)
}

fn matmul(lhs: &TensorData, rhs: &TensorData) -> Result<TensorData> {
    let shape =
        crate::ir::matmul_shape(lhs.shape(), rhs.shape()).ok_or_else(|| Error::InvalidMatmul {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        })?;
    let dtype = lhs.dtype().promote(rhs.dtype());
    let output_index = DenseIndex::new(shape.clone())?;
    let lhs_index = DenseIndex::new(lhs.shape().clone())?;
    let rhs_index = DenseIndex::new(rhs.shape().clone())?;
    let mut output = vec![Scalar::I(0); output_index.len()];
    let k = *lhs
        .shape()
        .dims()
        .last()
        .ok_or_else(|| Error::InvalidMatmul {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        })?;
    for (linear, value) in output.iter_mut().enumerate() {
        let coords = output_index.coords(linear)?;
        for inner in 0..k {
            let product = binary_scalar(
                lhs.scalar_at(matmul_lhs_offset(
                    &lhs_index,
                    &coords,
                    inner,
                    lhs.shape().rank() == 1,
                    rhs.shape().rank() == 1,
                )?),
                rhs.scalar_at(matmul_rhs_offset(
                    &rhs_index,
                    &coords,
                    inner,
                    lhs.shape().rank() == 1,
                    rhs.shape().rank() == 1,
                )?),
                dtype,
                BinaryOp::Mul,
            );
            *value = binary_scalar(*value, product, dtype, BinaryOp::Add);
        }
    }
    TensorData::from_scalars(shape, dtype, output)
}

fn matmul_grad(
    upstream: &TensorData,
    lhs: &TensorData,
    rhs: &TensorData,
    lhs_gradient: bool,
) -> Result<TensorData> {
    let output_shape =
        crate::ir::matmul_shape(lhs.shape(), rhs.shape()).ok_or_else(|| Error::InvalidMatmul {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        })?;
    if upstream.shape() != &output_shape {
        return Err(Error::ShapeMismatch {
            op: "matmul gradient",
            lhs: upstream.shape().clone(),
            rhs: output_shape,
        });
    }
    let target = if lhs_gradient { lhs } else { rhs };
    let dtype = target.dtype();
    let target_index = DenseIndex::new(target.shape().clone())?;
    let output_index = DenseIndex::new(upstream.shape().clone())?;
    let lhs_index = DenseIndex::new(lhs.shape().clone())?;
    let rhs_index = DenseIndex::new(rhs.shape().clone())?;
    let mut result = vec![Scalar::I(0); target_index.len()];
    let k = *lhs.shape().dims().last().ok_or(Error::InvalidIndex)?;
    for out_linear in 0..output_index.len() {
        let coords = output_index.coords(out_linear)?;
        let up = upstream.scalar_at(out_linear);
        for inner in 0..k {
            let lhs_offset = matmul_lhs_offset(
                &lhs_index,
                &coords,
                inner,
                lhs.shape().rank() == 1,
                rhs.shape().rank() == 1,
            )?;
            let rhs_offset = matmul_rhs_offset(
                &rhs_index,
                &coords,
                inner,
                lhs.shape().rank() == 1,
                rhs.shape().rank() == 1,
            )?;
            let (target_offset, other) = if lhs_gradient {
                (lhs_offset, rhs.scalar_at(rhs_offset))
            } else {
                (rhs_offset, lhs.scalar_at(lhs_offset))
            };
            let contribution = binary_scalar(up, other, dtype, BinaryOp::Mul);
            result[target_offset] =
                binary_scalar(result[target_offset], contribution, dtype, BinaryOp::Add);
        }
    }
    TensorData::from_scalars(target.shape().clone(), dtype, result)
}

fn matmul_lhs_offset(
    index: &DenseIndex,
    output_coords: &[usize],
    inner: usize,
    lhs_vector: bool,
    rhs_vector: bool,
) -> Result<usize> {
    let dims = index.shape().dims();
    if dims.len() == 1 {
        return index.offset(&[inner]);
    }
    let batch_rank = dims.len() - 2;
    let out_batch_len = output_coords.len() - usize::from(!lhs_vector) - usize::from(!rhs_vector);
    let out_batch = &output_coords[..out_batch_len];
    let pad = out_batch.len() - batch_rank;
    let mut coords = Vec::with_capacity(dims.len());
    for (axis, dim) in dims[..batch_rank].iter().enumerate() {
        coords.push(if *dim == 1 { 0 } else { out_batch[axis + pad] });
    }
    let row = output_coords[out_batch.len()];
    coords.extend([row, inner]);
    index.offset(&coords)
}

fn matmul_rhs_offset(
    index: &DenseIndex,
    output_coords: &[usize],
    inner: usize,
    lhs_vector: bool,
    rhs_vector: bool,
) -> Result<usize> {
    let dims = index.shape().dims();
    if dims.len() == 1 {
        return index.offset(&[inner]);
    }
    let batch_rank = dims.len() - 2;
    let out_batch_len = output_coords.len() - usize::from(!lhs_vector) - usize::from(!rhs_vector);
    let out_batch = &output_coords[..out_batch_len];
    let pad = out_batch.len() - batch_rank;
    let mut coords = Vec::with_capacity(dims.len());
    for (axis, dim) in dims[..batch_rank].iter().enumerate() {
        coords.push(if *dim == 1 { 0 } else { out_batch[axis + pad] });
    }
    let column = output_coords[out_batch_len + usize::from(!lhs_vector)];
    coords.extend([inner, column]);
    index.offset(&coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn evaluates_elementwise_and_reduction_graph() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2]);
        let y = graph.input("y", [2, 2]);
        let product = graph.mul(x, y).unwrap();
        let shifted = graph.add(product, y).unwrap();
        let output = graph.sum(shifted, 1).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2, 2], &[1., 2., 3., 4.])),
            ("y".into(), data([2, 2], &[5., 6., 7., 8.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([2], &[28., 68.])
        );
    }

    #[test]
    fn trace_is_inspectable() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let two = graph.constant(data([2], &[2., 2.]));
        let output = graph.mul(x, two).unwrap();
        assert_eq!(
            graph.trace(output).unwrap().to_string(),
            "%0 = input(\"x\") : [2]\n%1 = constant : [2]\n%2 = mul(%0, %1) : [2]\nreturn %2"
        );
    }

    #[test]
    fn broadcasts_trailing_dimensions_and_scalars() {
        let mut graph = Graph::new();
        let matrix = graph.input("matrix", [2, 3]);
        let row = graph.input("row", [3]);
        let scalar = graph.constant(TensorData::scalar(2.0));
        let shifted = graph.add(matrix, row).unwrap();
        let output = graph.mul(shifted, scalar).unwrap();
        let inputs = HashMap::from([
            ("matrix".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("row".into(), data([3], &[10., 20., 30.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([2, 3], &[22., 44., 66., 28., 50., 72.])
        );
    }

    #[test]
    fn reshapes_and_permutes_without_changing_values() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let transposed = graph.permute(input, [1, 0]).unwrap();
        let output = graph.reshape(transposed, [6]).unwrap();
        let inputs = HashMap::from([("x".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.]))]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([6], &[1., 4., 2., 5., 3., 6.])
        );
    }

    #[test]
    fn multiplies_rank_two_matrices() {
        let mut graph = Graph::new();
        let lhs = graph.input("lhs", [2, 3]);
        let rhs = graph.input("rhs", [3, 2]);
        let output = graph.matmul(lhs, rhs).unwrap();
        let inputs = HashMap::from([
            ("lhs".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("rhs".into(), data([3, 2], &[7., 8., 9., 10., 11., 12.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([2, 2], &[58., 64., 139., 154.])
        );
    }

    #[test]
    fn rejects_invalid_movement_and_matmul_shapes() {
        let mut graph = Graph::new();
        let matrix = graph.input("matrix", [2, 3]);
        let other = graph.input("other", [4, 2]);
        assert!(matches!(
            graph.reshape(matrix, [5]),
            Err(Error::InvalidReshape { .. })
        ));
        assert!(matches!(
            graph.permute(matrix, [0, 0]),
            Err(Error::InvalidPermutation { .. })
        ));
        assert!(matches!(
            graph.matmul(matrix, other),
            Err(Error::InvalidMatmul { .. })
        ));
    }

    #[test]
    fn generalized_matmul_handles_vectors_and_broadcast_batches() {
        let mut graph = Graph::new();
        let vector = graph.input("vector", [3]);
        let matrix = graph.input("matrix", [3, 2]);
        let product = graph.matmul(vector, matrix).unwrap();
        assert_eq!(graph.shape(product).unwrap(), &Shape::from([2]));
        let inputs = HashMap::from([
            ("vector".into(), data([3], &[1., 2., 3.])),
            ("matrix".into(), data([3, 2], &[1., 2., 3., 4., 5., 6.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, product, &inputs).unwrap(),
            data([2], &[22., 28.])
        );

        let mut graph = Graph::new();
        let lhs = graph.input("lhs", [2, 1, 2, 3]);
        let rhs = graph.input("rhs", [1, 4, 3, 2]);
        let product = graph.matmul(lhs, rhs).unwrap();
        assert_eq!(graph.shape(product).unwrap(), &Shape::from([2, 4, 2, 2]));
        let inputs = HashMap::from([
            ("lhs".into(), data([2, 1, 2, 3], &[1.; 12])),
            ("rhs".into(), data([1, 4, 3, 2], &[1.; 24])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, product, &inputs).unwrap(),
            data([2, 4, 2, 2], &[3.; 32])
        );
    }

    #[test]
    fn movement_maps_coordinates_and_preserves_exact_storage() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2, 3], DType::Bool);
        let shrunk = graph.shrink(x, [(0, 2), (1, 3)]).unwrap();
        let padded = graph
            .pad(shrunk, [(1, 0), (1, 1)], Scalar::Bool(true))
            .unwrap();
        let flipped = graph
            .stride(
                padded,
                [
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                    crate::Slice {
                        start: Some(3),
                        stop: Some(0),
                        step: -2,
                    },
                ],
            )
            .unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [2, 3],
                DType::Bool,
                [
                    Scalar::Bool(false),
                    Scalar::Bool(true),
                    Scalar::Bool(false),
                    Scalar::Bool(true),
                    Scalar::Bool(false),
                    Scalar::Bool(true),
                ],
            )
            .unwrap(),
        )]);
        assert_eq!(
            CpuBackend
                .execute(&graph, flipped, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![true, false, true, true, true, true])
        );
        assert!(
            graph
                .trace(flipped)
                .unwrap()
                .to_string()
                .contains("stride(")
        );
    }

    #[test]
    fn concat_promotes_dtype_on_an_arbitrary_axis() {
        let mut graph = Graph::new();
        let a = graph.input_dtype("a", [2, 1], DType::I8);
        let b = graph.input_dtype("b", [2, 2], DType::U8);
        let output = graph.concat([a, b], 1).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 3]));
        assert_eq!(graph.dtype(output).unwrap(), DType::I16);
        let inputs = HashMap::from([
            (
                "a".into(),
                TensorData::from_scalars([2, 1], DType::I8, [Scalar::I(-2), Scalar::I(3)]).unwrap(),
            ),
            (
                "b".into(),
                TensorData::from_scalars(
                    [2, 2],
                    DType::U8,
                    [Scalar::U(4), Scalar::U(5), Scalar::U(6), Scalar::U(7)],
                )
                .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, output, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I16(vec![-2, 4, 5, 3, 6, 7])
        );
    }

    #[test]
    fn movement_accepts_scalars_and_empty_dimensions_and_rejects_invalid_inputs() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let scalar_shrunk = graph.shrink(scalar, []).unwrap();
        let scalar_output = graph.pad(scalar_shrunk, [], Scalar::F(0.0)).unwrap();
        let empty = graph.input("empty", [2, 0]);
        let empty_output = graph
            .stride(
                empty,
                [
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                ],
            )
            .unwrap();
        let inputs = HashMap::from([
            ("scalar".into(), data([], &[7.])),
            ("empty".into(), data([2, 0], &[])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, scalar_output, &inputs).unwrap(),
            data([], &[7.])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, empty_output, &inputs)
                .unwrap()
                .shape(),
            &Shape::from([2, 0])
        );
        assert!(matches!(
            graph.shrink(empty, [(0, 3), (0, 0)]),
            Err(Error::InvalidBounds { .. })
        ));
        assert!(matches!(
            graph.stride(
                empty,
                [
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 0
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1
                    }
                ]
            ),
            Err(Error::InvalidSliceStep { .. })
        ));
        assert!(matches!(
            graph.concat([empty, empty], 2),
            Err(Error::InvalidAxis { .. })
        ));
    }

    #[test]
    fn gathers_and_scatters_with_checked_shared_index_mapping() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::Bool);
        let index = graph.input_dtype("index", [2, 2], DType::I32);
        let gathered = graph.gather(input, index, 1).unwrap();
        let base = graph.input_dtype("base", [2, 3], DType::U8);
        let updates = graph.input_dtype("updates", [2, 2], DType::I8);
        let replaced = graph.scatter(base, index, updates, 1).unwrap();
        let added = graph.scatter_add(base, index, updates, 1).unwrap();
        let inputs = HashMap::from([
            (
                "input".into(),
                TensorData::from_scalars(
                    [2, 3],
                    DType::Bool,
                    [
                        Scalar::Bool(false),
                        Scalar::Bool(true),
                        Scalar::Bool(true),
                        Scalar::Bool(true),
                        Scalar::Bool(false),
                        Scalar::Bool(true),
                    ],
                )
                .unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_scalars(
                    [2, 2],
                    DType::I32,
                    [Scalar::I(2), Scalar::I(0), Scalar::I(1), Scalar::I(1)],
                )
                .unwrap(),
            ),
            (
                "base".into(),
                TensorData::from_scalars([2, 3], DType::U8, [Scalar::U(1); 6]).unwrap(),
            ),
            (
                "updates".into(),
                TensorData::from_scalars(
                    [2, 2],
                    DType::I8,
                    [Scalar::I(10), Scalar::I(20), Scalar::I(-2), Scalar::I(4)],
                )
                .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, gathered, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![true, false, false, false])
        );
        assert_eq!(graph.dtype(replaced).unwrap(), DType::I16);
        assert_eq!(
            CpuBackend
                .execute(&graph, replaced, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I16(vec![20, 1, 10, 1, 4, 1])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, added, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I16(vec![21, 1, 11, 1, 3, 1])
        );
        assert!(
            graph
                .trace(added)
                .unwrap()
                .to_string()
                .contains("scatter_add")
        );

        let mut axes_graph = Graph::new();
        let cube = axes_graph.input("cube", [2, 2, 3]);
        let axis_zero_index = axes_graph.input_dtype("axis_zero_index", [1, 2, 2], DType::I32);
        let axis_two_index = axes_graph.input_dtype("axis_two_index", [2, 2, 2], DType::I32);
        let axis_zero = axes_graph.gather(cube, axis_zero_index, 0).unwrap();
        let axis_two = axes_graph.gather(cube, axis_two_index, 2).unwrap();
        let axes_inputs = HashMap::from([
            (
                "cube".into(),
                data([2, 2, 3], &(0..12).map(|x| x as f32).collect::<Vec<_>>()),
            ),
            (
                "axis_zero_index".into(),
                TensorData::from_scalars(
                    [1, 2, 2],
                    DType::I32,
                    [Scalar::I(1), Scalar::I(0), Scalar::I(1), Scalar::I(0)],
                )
                .unwrap(),
            ),
            (
                "axis_two_index".into(),
                TensorData::from_scalars(
                    [2, 2, 2],
                    DType::I32,
                    [
                        Scalar::I(2),
                        Scalar::I(0),
                        Scalar::I(1),
                        Scalar::I(2),
                        Scalar::I(0),
                        Scalar::I(1),
                        Scalar::I(2),
                        Scalar::I(0),
                    ],
                )
                .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&axes_graph, axis_zero, &axes_inputs)
                .unwrap(),
            data([1, 2, 2], &[6., 1., 9., 4.])
        );
        assert_eq!(
            CpuBackend
                .execute(&axes_graph, axis_two, &axes_inputs)
                .unwrap(),
            data([2, 2, 2], &[2., 0., 4., 5., 6., 7., 11., 9.])
        );
    }

    #[test]
    fn indexed_operations_reject_bad_dtype_and_bounds_and_support_fixed_mask_selection() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let bad = graph.input_dtype("bad", [2, 3], DType::Bool);
        assert!(matches!(
            graph.gather(x, bad, 1),
            Err(Error::InvalidIndexDType { .. })
        ));
        let index = graph.input_dtype("index", [2, 1], DType::I32);
        let gathered = graph.gather(x, index, 1).unwrap();
        let mask = graph.input_dtype("mask", [3], DType::Bool);
        let selected = graph.masked_select(x, mask, 5, Scalar::F(-1.0)).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            (
                "bad".into(),
                TensorData::from_scalars([2, 3], DType::Bool, [Scalar::Bool(false); 6]).unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_scalars([2, 1], DType::I32, [Scalar::I(0), Scalar::I(2)]).unwrap(),
            ),
            (
                "mask".into(),
                TensorData::from_scalars(
                    [3],
                    DType::Bool,
                    [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
                )
                .unwrap(),
            ),
        ]);
        let mut invalid_inputs = inputs.clone();
        invalid_inputs.insert(
            "index".into(),
            TensorData::from_scalars([2, 1], DType::I32, [Scalar::I(-1), Scalar::I(3)]).unwrap(),
        );
        assert!(matches!(
            CpuBackend.execute(&graph, gathered, &invalid_inputs),
            Err(Error::IndexOutOfBounds { .. })
        ));
        assert_eq!(
            CpuBackend.execute(&graph, selected, &inputs).unwrap(),
            data([5], &[1., 3., 4., 6., -1.])
        );
        assert!(
            graph
                .trace(selected)
                .unwrap()
                .to_string()
                .contains("masked_select")
        );

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input_dtype("empty", [0], DType::I32);
        let empty_index = empty_graph.input_dtype("empty_index", [0], DType::I32);
        let empty_gather = empty_graph.gather(empty, empty_index, 0).unwrap();
        let empty_inputs = HashMap::from([
            (
                "empty".into(),
                TensorData::from_scalars([0], DType::I32, []).unwrap(),
            ),
            (
                "empty_index".into(),
                TensorData::from_scalars([0], DType::I32, []).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&empty_graph, empty_gather, &empty_inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I32(vec![])
        );
    }

    #[test]
    fn evaluates_unary_and_binary_alu() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let y = graph.input("y", [3]);
        let quotient = graph.div(x, y).unwrap();
        let shifted = graph.sub(quotient, y).unwrap();
        let negated = graph.neg(shifted).unwrap();
        let output = graph.relu(negated).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([3], &[2., 8., -3.])),
            ("y".into(), data([3], &[2., 4., 1.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([3], &[1., 2., 4.])
        );
    }

    #[test]
    fn exp_and_log_round_trip_positive_values() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let logged = graph.log(x).unwrap();
        let output = graph.exp(logged).unwrap();
        let inputs = HashMap::from([("x".into(), data([3], &[0.5, 1.0, 4.0]))]);
        let actual = CpuBackend.execute(&graph, output, &inputs).unwrap();
        for (actual, expected) in actual.values().iter().zip([0.5, 1.0, 4.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn promotes_and_executes_mixed_exact_integer_storage() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], DType::I8);
        let rhs = graph.input_dtype("rhs", [2], DType::U8);
        let output = graph.add(lhs, rhs).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::I16);
        let inputs = HashMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([2], DType::I8, [Scalar::I(-2), Scalar::I(100)]).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([2], DType::U8, [Scalar::U(3), Scalar::U(200)]).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, output, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I16(vec![1, 300])
        );
    }

    #[test]
    fn cast_nodes_and_input_dtypes_are_checked() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], DType::I64);
        let output = graph.cast(input, DType::F32).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(7), Scalar::I(-3)]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap().dtype(),
            DType::F32
        );
        let wrong = HashMap::from([("x".into(), TensorData::new([2], vec![7.0, -3.0]).unwrap())]);
        assert!(matches!(
            CpuBackend.execute(&graph, output, &wrong),
            Err(Error::InputDType { .. })
        ));
    }

    #[test]
    fn predicates_logic_and_select_broadcast_exact_storage() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 1], DType::I64);
        let rhs = graph.input_dtype("rhs", [2], DType::U64);
        let condition = graph.lt(lhs, rhs).unwrap();
        assert_eq!(graph.dtype(condition).unwrap(), DType::Bool);
        let selected = graph.select(condition, lhs, rhs).unwrap();
        assert_eq!(graph.shape(selected).unwrap(), &Shape::from([2, 2]));
        assert_eq!(graph.dtype(selected).unwrap(), DType::F64);
        let inputs = HashMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([2, 1], DType::I64, [Scalar::I(-1), Scalar::I(5)])
                    .unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([2], DType::U64, [Scalar::U(0), Scalar::U(4)]).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, selected, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::F64(vec![-1.0, -1.0, 0.0, 4.0])
        );

        let mut logical_graph = Graph::new();
        let a = logical_graph.constant(
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(true), Scalar::Bool(false)])
                .unwrap(),
        );
        let b = logical_graph.logical_not(a).unwrap();
        let both = logical_graph.logical_and(a, b).unwrap();
        assert_eq!(
            CpuBackend
                .execute(&logical_graph, both, &HashMap::new())
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![false, false])
        );
    }

    #[test]
    fn comparisons_define_nan_and_invalid_logical_contracts() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let y = graph.input("y", [2]);
        let equal = graph.eq(x, y).unwrap();
        let unequal = graph.ne(x, y).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2], &[f32::NAN, 2.0])),
            ("y".into(), data([2], &[f32::NAN, 2.0])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, equal, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![false, true])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, unequal, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![true, false])
        );
        assert!(matches!(
            graph.logical_not(x),
            Err(Error::InvalidLogicalDType { .. })
        ));
        assert!(matches!(
            graph.select(x, x, y),
            Err(Error::InvalidLogicalDType { .. })
        ));
        assert!(
            graph
                .trace(equal)
                .unwrap()
                .to_string()
                .contains("eq(%0, %1)")
        );
    }
}
