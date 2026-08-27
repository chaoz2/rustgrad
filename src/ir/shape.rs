use super::*;
use crate::{DType, Error, Result, Shape};

pub(crate) fn conv_vjp_shape(
    graph: &Graph,
    upstream: NodeId,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    wrt: u8,
) -> Result<Shape> {
    match wrt {
        0 => Ok(graph.node(upstream)?.shape.clone()),
        1 => Ok(graph.node(input)?.shape.clone()),
        2 => Ok(graph.node(weight)?.shape.clone()),
        3 => Ok(graph.node(bias.ok_or(Error::InvalidIndex)?)?.shape.clone()),
        _ => Err(Error::InvalidIndex),
    }
}

/// The scalar dtype contract for unary ALU operations.
///
/// Tinygrad's public transcendental helpers lift non-floats to the default
/// float. RustGrad has no configurable default dtype, so that type is F32.
/// Narrow floats retain their storage dtype and are quantized at the CPU
/// result boundary. Predicates always produce bool; discrete operations retain
/// their input dtype so integer paths never travel through floating point.
pub(crate) fn unary_dtype(op: UnaryOp, input: DType) -> DType {
    if matches!(op, UnaryOp::IsNan | UnaryOp::IsInf | UnaryOp::IsFinite) {
        return DType::Bool;
    }
    if matches!(
        op,
        UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Reciprocal
            | UnaryOp::Sqrt
            | UnaryOp::Rsqrt
            | UnaryOp::Exp2
            | UnaryOp::Log2
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tan
            | UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tanh
            | UnaryOp::Erf
            | UnaryOp::Erfc
            | UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh
    ) && !input.is_float()
    {
        DType::F32
    } else {
        input
    }
}

/// Infers NumPy-style matmul shape.  Vectors are temporarily treated as a
/// leading (lhs) or trailing (rhs) matrix axis, then that artificial axis is
/// removed from the result.  All preceding axes broadcast normally.
pub(crate) fn matmul_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
    if lhs.rank() == 0 || rhs.rank() == 0 {
        return None;
    }
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let lhs_vector = lhs.rank() == 1;
    let rhs_vector = rhs.rank() == 1;
    let k_lhs = *lhs_dims.last()?;
    let k_rhs = if rhs_vector {
        rhs_dims[0]
    } else {
        rhs_dims[rhs.rank() - 2]
    };
    if k_lhs != k_rhs {
        return None;
    }
    let lhs_batch = if lhs_vector {
        &[][..]
    } else {
        &lhs_dims[..lhs.rank() - 2]
    };
    let rhs_batch = if rhs_vector {
        &[][..]
    } else {
        &rhs_dims[..rhs.rank() - 2]
    };
    let rank = lhs_batch.len().max(rhs_batch.len());
    let mut result = Vec::with_capacity(rank + 2);
    for axis in 0..rank {
        let lhs_axis = axis
            .checked_sub(rank - lhs_batch.len())
            .and_then(|i| lhs_batch.get(i))
            .copied()
            .unwrap_or(1);
        let rhs_axis = axis
            .checked_sub(rank - rhs_batch.len())
            .and_then(|i| rhs_batch.get(i))
            .copied()
            .unwrap_or(1);
        if lhs_axis != rhs_axis && lhs_axis != 1 && rhs_axis != 1 {
            return None;
        }
        result.push(lhs_axis.max(rhs_axis));
    }
    if !lhs_vector {
        result.push(lhs_dims[lhs.rank() - 2]);
    }
    if !rhs_vector {
        result.push(rhs_dims[rhs.rank() - 1]);
    }
    Some(Shape::new(result))
}

pub(crate) fn conv_transpose2d_shape(
    input: &Shape,
    weight: &Shape,
    options: ConvTranspose2dOptions,
) -> Result<Shape> {
    if input.rank() != 4
        || weight.rank() != 4
        || options.groups == 0
        || options.stride.contains(&0)
        || options.dilation.contains(&0)
        || options.output_padding[0] >= options.stride[0]
        || options.output_padding[1] >= options.stride[1]
        || input.dims()[1] != weight.dims()[0]
        || weight.dims()[0] % options.groups != 0
    {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "invalid transpose convolution geometry",
        });
    }
    let oc = weight.dims()[1]
        .checked_mul(options.groups)
        .ok_or_else(|| Error::ShapeOverflow(weight.clone()))?;
    let dim = |n: usize, k: usize, s: usize, d: usize, b: usize, a: usize, op: usize| {
        n.checked_sub(1)
            .and_then(|x| x.checked_mul(s))
            .and_then(|x| x.checked_add(d.checked_mul(k.checked_sub(1)?)?))
            .and_then(|x| x.checked_add(op))
            .and_then(|x| x.checked_add(1))
            .and_then(|x| x.checked_sub(b))
            .and_then(|x| x.checked_sub(a))
    };
    let h = dim(
        input.dims()[2],
        weight.dims()[2],
        options.stride[0],
        options.dilation[0],
        options.padding[0],
        options.padding[1],
        options.output_padding[0],
    )
    .ok_or_else(|| Error::InvalidConv2d {
        input: input.clone(),
        weight: weight.clone(),
        reason: "invalid transpose output shape",
    })?;
    let w = dim(
        input.dims()[3],
        weight.dims()[3],
        options.stride[1],
        options.dilation[1],
        options.padding[2],
        options.padding[3],
        options.output_padding[1],
    )
    .ok_or_else(|| Error::InvalidConv2d {
        input: input.clone(),
        weight: weight.clone(),
        reason: "invalid transpose output shape",
    })?;
    let output = Shape::new([input.dims()[0], oc, h, w]);
    output.numel()?;
    Ok(output)
}
pub(crate) fn conv2d_shape(input: &Shape, weight: &Shape, options: Conv2dOptions) -> Result<Shape> {
    if input.rank() != 4 || weight.rank() != 4 {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "input and weight must be rank 4",
        });
    }
    if options.groups == 0 || options.stride.contains(&0) || options.dilation.contains(&0) {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "groups, stride, and dilation must be positive",
        });
    }
    let i = input.dims();
    let w = weight.dims();
    if w[0] % options.groups != 0
        || i[1]
            != w[1]
                .checked_mul(options.groups)
                .ok_or_else(|| Error::ShapeOverflow(input.clone()))?
    {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "channel/group geometry",
        });
    }
    let spatial = |size: usize,
                   kernel: usize,
                   before: usize,
                   after: usize,
                   stride: usize,
                   dilation: usize|
     -> Result<usize> {
        let extent = kernel
            .checked_sub(1)
            .and_then(|x| x.checked_mul(dilation))
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| Error::ShapeOverflow(input.clone()))?;
        let padded = size
            .checked_add(before)
            .and_then(|x| x.checked_add(after))
            .ok_or_else(|| Error::ShapeOverflow(input.clone()))?;
        if padded < extent {
            return Err(Error::InvalidConv2d {
                input: input.clone(),
                weight: weight.clone(),
                reason: "kernel exceeds padded input",
            });
        }
        Ok((padded - extent) / stride + 1)
    };
    let output = Shape::from([
        i[0],
        w[0],
        spatial(
            i[2],
            w[2],
            options.padding[0],
            options.padding[1],
            options.stride[0],
            options.dilation[0],
        )?,
        spatial(
            i[3],
            w[3],
            options.padding[2],
            options.padding[3],
            options.stride[1],
            options.dilation[1],
        )?,
    ]);
    output.numel()?;
    Ok(output)
}

/// Returns normalized `(start, stop, step, output_length)` with the same
/// endpoint clipping rules as Rust's/Python's signed slicing model.
pub(crate) fn normalized_slice(
    dim: usize,
    slice: Slice,
    axis: usize,
) -> Result<(isize, isize, isize, usize)> {
    if slice.step == 0 {
        return Err(Error::InvalidSliceStep { axis });
    }
    let dim =
        isize::try_from(dim).map_err(|_| Error::ShapeOverflow(Shape::new(vec![usize::MAX])))?;
    let step = slice.step;
    let clamp = |value: isize, lo: isize, hi: isize| value.clamp(lo, hi);
    let (start, stop) = if step > 0 {
        let start = match slice.start {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, 0, dim),
            None => 0,
        };
        let stop = match slice.stop {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, 0, dim),
            None => dim,
        };
        (start, stop)
    } else {
        let start = match slice.start {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, -1, dim - 1),
            None => dim - 1,
        };
        // An omitted negative-step stop is the sentinel -1, not an index.
        let stop = match slice.stop {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, -1, dim - 1),
            None => -1,
        };
        (start, stop)
    };
    let length = if step > 0 {
        if start >= stop {
            0
        } else {
            usize::try_from((stop - start - 1) / step + 1).unwrap_or(0)
        }
    } else if start <= stop {
        0
    } else {
        usize::try_from((start - stop - 1) / (-step) + 1).unwrap_or(0)
    };
    Ok((start, stop, step, length))
}

pub(crate) fn validate_indexed(
    op: &'static str,
    input: &Node,
    index: &Node,
    axis: usize,
) -> Result<()> {
    if !index.dtype.is_integer() {
        return Err(Error::InvalidIndexDType {
            op,
            actual: index.dtype,
        });
    }
    if axis >= input.shape.rank() {
        return Err(Error::InvalidAxis {
            node: NodeId(usize::MAX),
            axis,
            rank: input.shape.rank(),
        });
    }
    if input.shape.rank() != index.shape.rank()
        || input
            .shape
            .dims()
            .iter()
            .zip(index.shape.dims())
            .enumerate()
            .any(|(dim, (input, index))| dim != axis && index > input)
    {
        return Err(Error::InvalidIndexedShape {
            op,
            input: input.shape.clone(),
            index: index.shape.clone(),
        });
    }
    Ok(())
}
pub(crate) fn normalize_axes(
    node: NodeId,
    rank: usize,
    axes: Option<Vec<isize>>,
) -> Result<Vec<usize>> {
    let mut axes = axes.unwrap_or_else(|| (0..rank).map(|x| x as isize).collect());
    for axis in &mut axes {
        if *axis < 0 {
            *axis += rank as isize;
        }
    }
    if axes.iter().any(|axis| *axis < 0 || *axis >= rank as isize) {
        return Err(Error::InvalidReductionAxes {
            node,
            axes: axes
                .iter()
                .map(|x| usize::try_from(*x).unwrap_or(usize::MAX))
                .collect(),
            rank,
        });
    }
    let mut normalized = axes.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    normalized.sort_unstable();
    if normalized.windows(2).any(|x| x[0] == x[1]) {
        return Err(Error::InvalidReductionAxes {
            node,
            axes: normalized,
            rank,
        });
    }
    Ok(normalized)
}
pub(crate) fn reduction_shape(shape: &Shape, axes: &[usize], keepdim: bool) -> Shape {
    Shape::new(
        shape
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if axes.contains(&i) {
                    keepdim.then_some(1)
                } else {
                    Some(*dim)
                }
            })
            .collect::<Vec<_>>(),
    )
}
pub(crate) fn has_empty_reduction_domain(input: &Shape, output: &Shape, axes: &[usize]) -> bool {
    matches!(output.numel(), Ok(numel) if numel > 0)
        && axes.iter().any(|axis| input.dims()[*axis] == 0)
}
pub(crate) fn sum_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => dtype,
        DType::Bool => DType::I32,
        DType::I8 | DType::I16 => DType::I32,
        DType::U8 | DType::U16 => DType::U32,
        _ => dtype,
    }
}
