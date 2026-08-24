use super::Backend;
use crate::engine::{DynamicGradient, DynamicRealized, RuntimeShape};
use crate::index::DenseIndex;
use crate::ir::{DynamicNodeId, DynamicOp};
use crate::{
    BinaryOp, CompareOp, DType, Error, Graph, LogicalOp, NodeId, Op, Result, Scalar, Shape,
    TensorData, UnaryOp,
    ir::{RandomKind, normalized_slice},
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn unit(seed: u64, index: u64) -> f64 {
    ((splitmix64(seed.wrapping_add(index)) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn random(shape: Shape, dtype: DType, kind: RandomKind, seed: u64) -> Result<TensorData> {
    let values = (0..shape.numel()?).map(|index| match kind {
        RandomKind::Uniform { low, high } => {
            Scalar::F(low + (high - low) * unit(seed, index as u64))
        }
        RandomKind::Normal { mean, std } => {
            let index = (index as u64).wrapping_mul(2);
            let u1 = unit(seed, index).max(f64::MIN_POSITIVE);
            let u2 = unit(seed, index.wrapping_add(1));
            Scalar::F(mean + std * (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos())
        }
        RandomKind::RandInt { low, high } => Scalar::I(
            low + (splitmix64(seed.wrapping_add(index as u64)) % (high - low) as u64) as i64,
        ),
    });
    TensorData::from_scalars(shape, dtype, values)
}

fn random_permutation(shape: Shape, dtype: DType, seed: u64) -> Result<TensorData> {
    let count = shape.numel()?;
    let mut values: Vec<i64> = (0..count).map(|value| value as i64).collect();
    for index in (1..count).rev() {
        let swap = (splitmix64(seed.wrapping_add(index as u64)) % (index as u64 + 1)) as usize;
        values.swap(index, swap);
    }
    TensorData::from_scalars(shape, dtype, values.into_iter().map(Scalar::I))
}

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
                Op::Random { kind, seed } => random(node.shape.clone(), node.dtype, *kind, *seed)?,
                Op::RandomPermutation { seed } => {
                    random_permutation(node.shape.clone(), node.dtype, *seed)?
                }
                Op::Cast { input, dtype } => values[input.index()].cast(*dtype),
                Op::Detach { input } => values[input.index()].clone(),
                Op::Unary { op, input } => unary(&values[input.index()], *op, node.dtype)?,
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
                Op::ReduceGradVjp {
                    cotangent,
                    input,
                    upstream,
                    kind,
                    axes,
                    keepdim,
                    wrt,
                } => reduce_grad_vjp(
                    &values[cotangent.index()],
                    &values[input.index()],
                    &values[upstream.index()],
                    *kind,
                    axes,
                    *keepdim,
                    *wrt,
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
                Op::ScatterPositionsVjp {
                    cotangent,
                    input_shape,
                    starts,
                    steps,
                } => scatter_positions_vjp(&values[cotangent.index()], input_shape, starts, steps)?,
                Op::Gather { input, index, axis } => {
                    gather(&values[input.index()], &values[index.index()], *axis)?
                }
                Op::StaticIndex { input, plan } => static_index(&values[input.index()], plan)?,
                Op::StaticIndexGrad {
                    cotangent,
                    input_shape,
                    plan,
                } => static_index_grad(&values[cotangent.index()], input_shape, plan)?,
                Op::MaskedSelect {
                    input,
                    mask,
                    size,
                    fill,
                } => masked_select(&values[input.index()], &values[mask.index()], *size, *fill)?,
                Op::Matmul { lhs, rhs } => matmul(&values[lhs.index()], &values[rhs.index()])?,
                Op::Einsum { inputs, plan } => einsum(&values, inputs, plan, node.dtype)?,
                Op::EinsumGrad {
                    upstream,
                    inputs,
                    plan,
                    target,
                } => einsum_grad(&values, *upstream, inputs, plan, *target, node.dtype)?,
                Op::EinsumGradVjp {
                    cotangent,
                    upstream,
                    inputs,
                    plan,
                    target,
                    wrt,
                } => einsum_grad_vjp(
                    &values, *cotangent, *upstream, inputs, plan, *target, *wrt, node.dtype,
                )?,
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
                Op::MatmulGradVjp {
                    cotangent,
                    upstream,
                    lhs,
                    rhs,
                    lhs_gradient,
                    wrt,
                } => matmul_grad_vjp(
                    &values[cotangent.index()],
                    &values[upstream.index()],
                    &values[lhs.index()],
                    &values[rhs.index()],
                    *lhs_gradient,
                    *wrt,
                )?,
                Op::Conv2d {
                    input,
                    weight,
                    bias,
                    options,
                } => conv2d(
                    &values[input.index()],
                    &values[weight.index()],
                    bias.map(|id| &values[id.index()]),
                    *options,
                )?,
                Op::Conv2dGrad {
                    upstream,
                    input,
                    weight,
                    bias,
                    options,
                    target,
                } => conv2d_grad(
                    &values[upstream.index()],
                    &values[input.index()],
                    &values[weight.index()],
                    bias.map(|id| &values[id.index()]),
                    *options,
                    *target,
                )?,
                Op::Conv2dGradVjp {
                    cotangent,
                    upstream,
                    input,
                    weight,
                    bias,
                    options,
                    target,
                    wrt,
                } => conv2d_grad_vjp(
                    &values[cotangent.index()],
                    &values[upstream.index()],
                    &values[input.index()],
                    &values[weight.index()],
                    bias.map(|id| &values[id.index()]),
                    *options,
                    *target,
                    *wrt,
                )?,
                Op::ConvTranspose2d {
                    input,
                    weight,
                    bias,
                    options,
                } => conv_transpose2d(
                    &values[input.index()],
                    &values[weight.index()],
                    bias.map(|id| &values[id.index()]),
                    *options,
                )?,
                Op::ConvTranspose2dGrad {
                    upstream,
                    input,
                    weight,
                    bias,
                    options,
                    target,
                } => conv_transpose2d_grad(
                    &values[upstream.index()],
                    &values[input.index()],
                    &values[weight.index()],
                    bias.map(|id| &values[id.index()]),
                    *options,
                    *target,
                )?,
                Op::ConvTranspose2dGradVjp {
                    cotangent,
                    upstream,
                    input,
                    weight,
                    bias,
                    options,
                    target,
                    wrt,
                } => conv_transpose2d_grad_vjp(
                    &values[cotangent.index()],
                    &values[upstream.index()],
                    &values[input.index()],
                    &values[weight.index()],
                    bias.map(|id| &values[id.index()]),
                    *options,
                    *target,
                    *wrt,
                )?,
            };
            debug_assert_eq!(value.shape(), &node.shape);
            values.push(value);
        }
        values.pop().ok_or(Error::UnknownNode(output))
    }
}

impl CpuBackend {
    /// Realizes a typed dynamic result through the CPU semantic oracle.
    pub fn execute_dynamic(
        &self,
        graph: &Graph,
        output: DynamicNodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<DynamicRealized> {
        let node = graph.dynamic_node(output)?;
        let value = self.dynamic_value(graph, output, inputs)?;
        node.output.validate(value.shape())?;
        if value.dtype() != node.dtype {
            return Err(Error::InvalidIndex);
        }
        Ok(DynamicRealized {
            shape: RuntimeShape::new(node.output.rank(), value.shape().clone())
                .map_err(|_| Error::InvalidIndex)?,
            output: value,
        })
    }

    /// Executes a scalar dynamic loss and its first-order gradient with
    /// respect to one static floating source node.
    pub fn execute_dynamic_gradient(
        &self,
        graph: &Graph,
        loss: DynamicNodeId,
        wrt: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<DynamicGradient> {
        let realized = self.execute_dynamic(graph, loss, inputs)?;
        if realized.output.shape().numel()? != 1 || !realized.output.dtype().is_float() {
            return Err(Error::NonScalarLoss(realized.output.shape().clone()));
        }
        let seed = TensorData::from_scalars([], realized.output.dtype(), [Scalar::F(1.0)])?;
        let gradient = self.dynamic_vjp(graph, loss, &seed, wrt, inputs)?;
        Ok(DynamicGradient {
            loss: realized,
            gradient,
        })
    }

    fn dynamic_value(
        &self,
        graph: &Graph,
        output: DynamicNodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<TensorData> {
        match graph.dynamic_node(output)?.op {
            DynamicOp::Nonzero { input } => nonzero(&self.execute(graph, input, inputs)?),
            DynamicOp::MaskedSelect { input, mask } => dynamic_masked_select(
                &self.execute(graph, input, inputs)?,
                &self.execute(graph, mask, inputs)?,
            ),
            DynamicOp::Sum { input } => dynamic_sum(&self.dynamic_value(graph, input, inputs)?),
        }
    }

    fn dynamic_vjp(
        &self,
        graph: &Graph,
        output: DynamicNodeId,
        upstream: &TensorData,
        wrt: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<TensorData> {
        match graph.dynamic_node(output)?.op {
            DynamicOp::Sum { input } => {
                let value = self.dynamic_value(graph, input, inputs)?;
                if upstream.shape().numel()? != 1 || upstream.dtype() != value.dtype() {
                    return Err(Error::InvalidIndex);
                }
                let expanded = TensorData::from_scalars(
                    value.shape().clone(),
                    value.dtype(),
                    (0..value.len()).map(|_| upstream.scalar_at(0)),
                )?;
                self.dynamic_vjp(graph, input, &expanded, wrt, inputs)
            }
            DynamicOp::MaskedSelect { input, mask } if input == wrt => {
                let source = self.execute(graph, input, inputs)?;
                if !source.dtype().is_float() {
                    return Err(Error::NonDifferentiableTarget(input));
                }
                dynamic_masked_select_vjp(&source, &self.execute(graph, mask, inputs)?, upstream)
            }
            DynamicOp::MaskedSelect { .. } => Err(Error::NonDifferentiableTarget(wrt)),
            DynamicOp::Nonzero { .. } => Err(Error::NonDifferentiableIndexing("dynamic nonzero")),
        }
    }

    /// First-order VJP executor for a realized dynamic masked selection.
    /// The upstream must have the exact realized `[selected_count]` shape.
    pub fn execute_dynamic_masked_select_vjp(
        &self,
        graph: &Graph,
        output: DynamicNodeId,
        upstream: &TensorData,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<TensorData> {
        let node = graph.dynamic_node(output)?;
        let DynamicOp::MaskedSelect { input, mask } = node.op else {
            return Err(Error::NonDifferentiableIndexing("dynamic nonzero"));
        };
        let input = self.execute(graph, input, inputs)?;
        if !input.dtype().is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "dynamic masked_select input",
            ));
        }
        let mask = self.execute(graph, mask, inputs)?;
        dynamic_masked_select_vjp(&input, &mask, upstream)
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
    if matches!(
        op,
        BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::TruncDiv | BinaryOp::Mod | BinaryOp::FMod
    ) && dtype.is_integer()
    {
        for linear in 0..output_len {
            let divisor = rhs.scalar_at(broadcast_offset(linear, output_shape, rhs.shape()));
            if (matches!(dtype.category(), crate::DTypeCategory::Unsigned) && divisor.as_u64() == 0)
                || (matches!(dtype.category(), crate::DTypeCategory::Signed)
                    && divisor.as_i64() == 0)
            {
                return Err(Error::DivisionByZero { op: op.name() });
            }
        }
    }
    if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        for linear in 0..output_len {
            let count = rhs.scalar_at(broadcast_offset(linear, output_shape, rhs.shape()));
            let invalid = match rhs.dtype().category() {
                crate::DTypeCategory::Signed => {
                    count.as_i64() < 0 || count.as_i64() as u64 >= dtype.bits() as u64
                }
                crate::DTypeCategory::Unsigned => count.as_u64() >= dtype.bits() as u64,
                crate::DTypeCategory::Bool | crate::DTypeCategory::Float => true,
            };
            if invalid {
                return Err(Error::InvalidShiftCount {
                    // The public error predates U64. Saturation keeps its diagnostic
                    // deterministic instead of reinterpreting a large count as negative.
                    count: count.as_u64().min(i64::MAX as u64) as i64,
                    bits: dtype.bits(),
                });
            }
        }
    }
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
            BinaryOp::Pow => lhs.powf(rhs),
            BinaryOp::Maximum => lhs.max(rhs),
            BinaryOp::Minimum => lhs.min(rhs),
            BinaryOp::FloorDiv => (lhs / rhs).floor(),
            BinaryOp::TruncDiv => (lhs / rhs).trunc(),
            BinaryOp::Mod => lhs - (lhs / rhs).floor() * rhs,
            BinaryOp::FMod => lhs % rhs,
            BinaryOp::Atan2 => lhs.atan2(rhs),
            // tinygrad defines copysign with comparisons plus reciprocal so
            // -0 selects a negative result while either NaN sign selects +.
            BinaryOp::Copysign => {
                let magnitude = lhs.abs();
                if rhs < 0.0 || rhs.recip() < 0.0 {
                    -magnitude
                } else {
                    magnitude
                }
            }
            _ => f64::NAN,
        });
    }
    if matches!(dtype, DType::Bool) {
        let (lhs, rhs) = (lhs.as_bool(), rhs.as_bool());
        return Scalar::Bool(match op {
            BinaryOp::Add => lhs || rhs,
            BinaryOp::Sub => lhs ^ rhs,
            BinaryOp::Mul => lhs && rhs,
            BinaryOp::Div => lhs && rhs,
            BinaryOp::BitAnd => lhs && rhs,
            BinaryOp::BitOr => lhs || rhs,
            BinaryOp::BitXor => lhs ^ rhs,
            BinaryOp::Maximum => lhs || rhs,
            BinaryOp::Minimum => lhs && rhs,
            BinaryOp::Atan2 | BinaryOp::Copysign => lhs,
            _ => false,
        });
    }
    if matches!(dtype.category(), crate::DTypeCategory::Unsigned) {
        let (lhs, rhs) = (lhs.as_u64(), rhs.as_u64());
        return Scalar::U(match op {
            BinaryOp::Add => lhs.wrapping_add(rhs),
            BinaryOp::Sub => lhs.wrapping_sub(rhs),
            BinaryOp::Mul => lhs.wrapping_mul(rhs),
            BinaryOp::Div => lhs / rhs,
            BinaryOp::Pow => lhs.wrapping_pow(rhs as u32),
            BinaryOp::Maximum => lhs.max(rhs),
            BinaryOp::Minimum => lhs.min(rhs),
            BinaryOp::FloorDiv | BinaryOp::TruncDiv => {
                if rhs == 0 {
                    0
                } else {
                    lhs / rhs
                }
            }
            BinaryOp::Mod | BinaryOp::FMod => {
                if rhs == 0 {
                    0
                } else {
                    lhs % rhs
                }
            }
            BinaryOp::BitAnd => lhs & rhs,
            BinaryOp::BitOr => lhs | rhs,
            BinaryOp::BitXor => lhs ^ rhs,
            BinaryOp::Shl => lhs.wrapping_shl(rhs as u32),
            BinaryOp::Shr => lhs.wrapping_shr(rhs as u32),
            BinaryOp::Atan2 => lhs,
            BinaryOp::Copysign => lhs,
        });
    }
    let (lhs, rhs) = (lhs.as_i64(), rhs.as_i64());
    Scalar::I(match op {
        BinaryOp::Add => lhs.wrapping_add(rhs),
        BinaryOp::Sub => lhs.wrapping_sub(rhs),
        BinaryOp::Mul => lhs.wrapping_mul(rhs),
        BinaryOp::Div => lhs.wrapping_div(rhs),
        BinaryOp::Pow => lhs.wrapping_pow(rhs as u32),
        BinaryOp::Maximum => lhs.max(rhs),
        BinaryOp::Minimum => lhs.min(rhs),
        BinaryOp::FloorDiv => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_div_euclid(rhs)
            }
        }
        BinaryOp::TruncDiv => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_div(rhs)
            }
        }
        BinaryOp::Mod => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_rem_euclid(rhs)
            }
        }
        BinaryOp::FMod => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_rem(rhs)
            }
        }
        BinaryOp::BitAnd => lhs & rhs,
        BinaryOp::BitOr => lhs | rhs,
        BinaryOp::BitXor => lhs ^ rhs,
        BinaryOp::Shl => lhs.wrapping_shl(rhs as u32),
        BinaryOp::Shr => lhs.wrapping_shr(rhs as u32),
        BinaryOp::Atan2 => lhs,
        BinaryOp::Copysign => {
            let magnitude = lhs.wrapping_abs();
            if rhs < 0 {
                magnitude.wrapping_neg()
            } else {
                magnitude
            }
        }
    })
}

fn unary(input: &TensorData, op: UnaryOp, dtype: DType) -> Result<TensorData> {
    if !input.dtype().is_float()
        && (dtype == input.dtype()
            || matches!(op, UnaryOp::IsNan | UnaryOp::IsInf | UnaryOp::IsFinite))
    {
        return unary_exact(input, op);
    }
    let values = (0..input.len()).map(|index| {
        let value = input.scalar_at(index).as_f64();
        let result = match op {
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
            UnaryOp::Abs => value.abs(),
            UnaryOp::Reciprocal => value.recip(),
            UnaryOp::Square => value * value,
            UnaryOp::Sqrt => value.sqrt(),
            UnaryOp::Rsqrt => value.sqrt().recip(),
            UnaryOp::Exp2 => value.exp2(),
            UnaryOp::Log2 => value.log2(),
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Tan => value.tan(),
            UnaryOp::Sinh => value.sinh(),
            UnaryOp::Cosh => value.cosh(),
            UnaryOp::Tanh => value.tanh(),
            UnaryOp::Erf => erf(value),
            UnaryOp::Erfc => 1.0 - erf(value),
            UnaryOp::Asin => value.asin(),
            UnaryOp::Acos => value.acos(),
            UnaryOp::Atan => value.atan(),
            UnaryOp::Asinh => value.asinh(),
            UnaryOp::Acosh => value.acosh(),
            UnaryOp::Atanh => value.atanh(),
            UnaryOp::Floor => value.floor(),
            UnaryOp::Ceil => value.ceil(),
            UnaryOp::Trunc => value.trunc(),
            UnaryOp::Round => value.round_ties_even(),
            UnaryOp::Sign => {
                if value.is_nan() {
                    f64::NAN
                } else {
                    value.signum()
                }
            }
            UnaryOp::IsNan | UnaryOp::IsInf | UnaryOp::IsFinite => value,
        };
        match op {
            UnaryOp::IsNan => Scalar::Bool(value.is_nan()),
            UnaryOp::IsInf => Scalar::Bool(value.is_infinite()),
            UnaryOp::IsFinite => Scalar::Bool(value.is_finite()),
            _ => Scalar::F(result),
        }
    });
    TensorData::from_scalars(input.shape().clone(), dtype, values)
}

/// The checked-in tinygrad error-function approximation (A&S 7.1.26).
/// Keeping it here, rather than depending on a host C library, makes the CPU
/// oracle deterministic across supported platforms and matches tinygrad's
/// compositional implementation.
fn erf(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    let t = 1.0 / (1.0 + 0.327_591_1 * value.abs());
    let polynomial =
        ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    value.signum() * (1.0 - polynomial * (-value * value).exp())
}

/// Executes non-floating unary operations without converting exact values to
/// f32/f64. Only the float-only family reaches the floating implementation.
fn unary_exact(input: &TensorData, op: UnaryOp) -> Result<TensorData> {
    let dtype = input.dtype();
    let values = (0..input.len()).map(|index| {
        let value = input.scalar_at(index);
        match op {
            UnaryOp::IsNan => Scalar::Bool(false),
            UnaryOp::IsInf => Scalar::Bool(false),
            UnaryOp::IsFinite => Scalar::Bool(true),
            _ if dtype == DType::Bool => {
                let value = value.as_bool();
                Scalar::Bool(match op {
                    UnaryOp::Neg => !value,
                    UnaryOp::Relu
                    | UnaryOp::Step
                    | UnaryOp::Abs
                    | UnaryOp::Square
                    | UnaryOp::Sign => value,
                    UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc | UnaryOp::Round => value,
                    _ => value,
                })
            }
            _ if matches!(dtype.category(), crate::DTypeCategory::Unsigned) => {
                let value = value.as_u64();
                Scalar::U(match op {
                    UnaryOp::Neg => 0u64.wrapping_sub(value),
                    UnaryOp::Relu
                    | UnaryOp::Step
                    | UnaryOp::Abs
                    | UnaryOp::Floor
                    | UnaryOp::Ceil
                    | UnaryOp::Trunc
                    | UnaryOp::Round => value,
                    UnaryOp::Square => value.wrapping_mul(value),
                    UnaryOp::Sign => u64::from(value != 0),
                    _ => value,
                })
            }
            _ => {
                let value = value.as_i64();
                Scalar::I(match op {
                    UnaryOp::Neg => value.wrapping_neg(),
                    UnaryOp::Relu => value.max(0),
                    UnaryOp::Step => i64::from(value > 0),
                    UnaryOp::Abs => value.wrapping_abs(),
                    UnaryOp::Square => value.wrapping_mul(value),
                    UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc | UnaryOp::Round => value,
                    UnaryOp::Sign => value.signum(),
                    _ => value,
                })
            }
        }
    });
    let output_dtype = if matches!(op, UnaryOp::IsNan | UnaryOp::IsInf | UnaryOp::IsFinite) {
        DType::Bool
    } else {
        dtype
    };
    TensorData::from_scalars(input.shape().clone(), output_dtype, values)
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
            *v = Scalar::F(if c == 0 {
                f64::NAN
            } else {
                v.as_f64() / c as f64
            });
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

fn reduce_grad_vjp(
    cotangent: &TensorData,
    input: &TensorData,
    upstream: &TensorData,
    kind: crate::ReduceKind,
    axes: &[usize],
    keepdim: bool,
    wrt: u8,
) -> Result<TensorData> {
    if cotangent.shape() != input.shape() {
        return Err(Error::GradientShape {
            output: input.shape().clone(),
            upstream: cotangent.shape().clone(),
        });
    }
    let ii = DenseIndex::new(input.shape().clone())?;
    let reduced = reduce(input, kind, axes, true, input.dtype())?;
    let ri = DenseIndex::new(reduced.shape().clone())?;
    let ui = DenseIndex::new(upstream.shape().clone())?;
    let output_shape = match wrt {
        0 => input.shape().clone(),
        1 => upstream.shape().clone(),
        _ => return Err(Error::InvalidIndex),
    };
    let oi = DenseIndex::new(output_shape.clone())?;
    let mut zero = vec![0usize; ri.len()];
    let mut nonzero = vec![Scalar::I(1); ri.len()];
    let mut ties = vec![0usize; ri.len()];
    for l in 0..ii.len() {
        let c = ii.coords(l)?;
        let r = ri.offset(&reduce_group_coords(&c, axes))?;
        let value = input.scalar_at(l);
        if value.as_f64() == 0. {
            zero[r] += 1;
        } else {
            nonzero[r] = binary_scalar(nonzero[r], value, input.dtype(), BinaryOp::Mul);
        }
        if matches!(kind, crate::ReduceKind::Max | crate::ReduceKind::Min)
            && value.as_f64() == reduced.scalar_at(r).as_f64()
        {
            ties[r] += 1;
        }
    }
    let mut out = vec![Scalar::I(0); oi.len()];
    for i in 0..ii.len() {
        let coords_i = ii.coords(i)?;
        let r = ri.offset(&reduce_group_coords(&coords_i, axes))?;
        let ucoords = if keepdim {
            reduce_group_coords(&coords_i, axes)
        } else {
            coords_i
                .iter()
                .enumerate()
                .filter_map(|(axis, value)| (!axes.contains(&axis)).then_some(*value))
                .collect()
        };
        let uoffset = ui.offset(&ucoords)?;
        let c_i = cotangent.scalar_at(i);
        match kind {
            crate::ReduceKind::Product => {
                if wrt == 1 {
                    let local = if zero[r] == 0 {
                        reduced.scalar_at(r).as_f64() / input.scalar_at(i).as_f64()
                    } else if zero[r] == 1 && input.scalar_at(i).as_f64() == 0. {
                        nonzero[r].as_f64()
                    } else {
                        0.
                    };
                    out[uoffset] = binary_scalar(
                        out[uoffset],
                        Scalar::F(c_i.as_f64() * local),
                        upstream.dtype(),
                        BinaryOp::Add,
                    );
                } else if zero[r] == 0 {
                    for (j, slot) in out.iter_mut().enumerate() {
                        let coords_j = ii.coords(j)?;
                        if ri.offset(&reduce_group_coords(&coords_j, axes))? == r && j != i {
                            let value = c_i.as_f64()
                                * upstream.scalar_at(uoffset).as_f64()
                                * reduced.scalar_at(r).as_f64()
                                / input.scalar_at(i).as_f64()
                                / input.scalar_at(j).as_f64();
                            *slot = binary_scalar(
                                *slot,
                                Scalar::F(value),
                                input.dtype(),
                                BinaryOp::Add,
                            );
                        }
                    }
                } else if zero[r] == 1 && input.scalar_at(i).as_f64() == 0. {
                    for (j, slot) in out.iter_mut().enumerate() {
                        let coords_j = ii.coords(j)?;
                        if ri.offset(&reduce_group_coords(&coords_j, axes))? == r
                            && input.scalar_at(j).as_f64() != 0.
                        {
                            let value = c_i.as_f64()
                                * upstream.scalar_at(uoffset).as_f64()
                                * nonzero[r].as_f64()
                                / input.scalar_at(j).as_f64();
                            *slot = binary_scalar(
                                *slot,
                                Scalar::F(value),
                                input.dtype(),
                                BinaryOp::Add,
                            );
                        }
                    }
                }
            }
            crate::ReduceKind::Max | crate::ReduceKind::Min => {
                if wrt == 1 {
                    let local = if reduced.scalar_at(r).as_f64().is_nan() {
                        f64::NAN
                    } else if input.scalar_at(i).as_f64() == reduced.scalar_at(r).as_f64() {
                        1.0 / ties[r] as f64
                    } else {
                        0.0
                    };
                    out[uoffset] = binary_scalar(
                        out[uoffset],
                        Scalar::F(c_i.as_f64() * local),
                        upstream.dtype(),
                        BinaryOp::Add,
                    );
                }
            }
            _ => return Err(Error::InvalidIndex),
        }
    }
    TensorData::from_scalars(
        output_shape,
        if wrt == 0 {
            input.dtype()
        } else {
            upstream.dtype()
        },
        out,
    )
}

fn reduce_group_coords(coords: &[usize], axes: &[usize]) -> Vec<usize> {
    coords
        .iter()
        .enumerate()
        .map(|(axis, value)| if axes.contains(&axis) { 0 } else { *value })
        .collect()
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

fn scatter_positions_vjp(
    cotangent: &TensorData,
    input_shape: &Shape,
    starts: &[isize],
    steps: &[isize],
) -> Result<TensorData> {
    if starts.len() != input_shape.rank()
        || steps.len() != input_shape.rank()
        || cotangent.shape().rank() != input_shape.rank()
    {
        return Err(Error::InvalidMovementRank {
            op: "scatter vjp",
            expected: input_shape.rank(),
            actual: starts.len().min(steps.len()).min(cotangent.shape().rank()),
        });
    }
    let input_index = DenseIndex::new(input_shape.clone())?;
    let cotangent_index = DenseIndex::new(cotangent.shape().clone())?;
    let mut output = Vec::with_capacity(input_index.len());
    for linear in 0..input_index.len() {
        let coords = input_index.coords(linear)?;
        let source = coords
            .iter()
            .zip(starts)
            .zip(steps)
            .map(|((coord, start), step)| {
                let scaled = isize::try_from(*coord)
                    .ok()
                    .and_then(|value| value.checked_mul(*step))
                    .ok_or(Error::InvalidIndex)?;
                usize::try_from(start.checked_add(scaled).ok_or(Error::InvalidIndex)?)
                    .map_err(|_| Error::InvalidIndex)
            })
            .collect::<Result<Vec<_>>>()?;
        output.push(cotangent.scalar_at(cotangent_index.offset(&source)?));
    }
    TensorData::from_scalars(input_shape.clone(), cotangent.dtype(), output)
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

fn static_index(
    input: &TensorData,
    plan: &crate::ir::indexing::StaticIndexPlan,
) -> Result<TensorData> {
    let source = DenseIndex::new(input.shape().clone())?;
    let output = DenseIndex::new(plan.output_shape().clone())?;
    let mut values = Vec::with_capacity(output.len());
    for linear in 0..output.len() {
        let coords = plan.source_coords(&output.coords(linear)?)?;
        values.push(input.scalar_at(source.offset(&coords)?));
    }
    TensorData::from_scalars(plan.output_shape().clone(), input.dtype(), values)
}

fn static_index_grad(
    cotangent: &TensorData,
    input_shape: &Shape,
    plan: &crate::ir::indexing::StaticIndexPlan,
) -> Result<TensorData> {
    if cotangent.shape() != plan.output_shape() {
        return Err(Error::InvalidIndex);
    }
    let input = DenseIndex::new(input_shape.clone())?;
    let output = DenseIndex::new(plan.output_shape().clone())?;
    let mut values = vec![Scalar::I(0); input.len()];
    for linear in 0..output.len() {
        let destination = input.offset(&plan.source_coords(&output.coords(linear)?)?)?;
        values[destination] = binary_scalar(
            values[destination],
            cotangent.scalar_at(linear),
            cotangent.dtype(),
            BinaryOp::Add,
        );
    }
    TensorData::from_scalars(input_shape.clone(), cotangent.dtype(), values)
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

/// Checked once and consumed by both dynamic count and materialization.
fn masked_positions(input: &TensorData, mask: &TensorData) -> Result<Vec<usize>> {
    let input_index = DenseIndex::new(input.shape().clone())?;
    let mask_index = DenseIndex::new(mask.shape().clone())?;
    let mut positions = Vec::new();
    for linear in 0..input_index.len() {
        let coords = input_index.coords(linear)?;
        if mask
            .scalar_at(mask_index.broadcast_offset(&input_index, &coords)?)
            .as_bool()
        {
            positions.push(linear);
        }
    }
    Ok(positions)
}

fn dynamic_masked_select(input: &TensorData, mask: &TensorData) -> Result<TensorData> {
    let positions = masked_positions(input, mask)?;
    let count = positions.len();
    let values = positions
        .into_iter()
        .map(|position| input.scalar_at(position));
    TensorData::from_scalars([count], input.dtype(), values)
}

fn dynamic_masked_select_vjp(
    input: &TensorData,
    mask: &TensorData,
    upstream: &TensorData,
) -> Result<TensorData> {
    let positions = masked_positions(input, mask)?;
    if upstream.shape() != &Shape::from([positions.len()]) || upstream.dtype() != input.dtype() {
        return Err(Error::InvalidIndex);
    }
    let mut output = vec![Scalar::I(0); input.len()];
    for (upstream_index, position) in positions.into_iter().enumerate() {
        output[position] = binary_scalar(
            output[position],
            upstream.scalar_at(upstream_index),
            input.dtype(),
            BinaryOp::Add,
        );
    }
    TensorData::from_scalars(input.shape().clone(), input.dtype(), output)
}

fn dynamic_sum(input: &TensorData) -> Result<TensorData> {
    let value = (0..input.len()).fold(Scalar::I(0), |sum, index| {
        binary_scalar(sum, input.scalar_at(index), input.dtype(), BinaryOp::Add)
    });
    TensorData::from_scalars([], input.dtype(), [value])
}

fn nonzero(input: &TensorData) -> Result<TensorData> {
    let index = DenseIndex::new(input.shape().clone())?;
    let mut coordinates = Vec::new();
    for linear in 0..index.len() {
        if input.scalar_at(linear).as_bool() {
            coordinates.push(index.coords(linear)?);
        }
    }
    let count = coordinates.len();
    let values = coordinates
        .into_iter()
        .flatten()
        .map(|value| Scalar::I(value as i64));
    TensorData::from_scalars([count, input.shape().rank()], DType::I64, values)
}

fn einsum(
    values: &[TensorData],
    inputs: &[NodeId],
    plan: &crate::EinsumPlan,
    dtype: DType,
) -> Result<TensorData> {
    let tensors = inputs
        .iter()
        .map(|id| values.get(id.index()).ok_or(Error::UnknownNode(*id)))
        .collect::<Result<Vec<_>>>()?;
    let output_shape = plan.output_shape();
    let output_index = DenseIndex::new(output_shape.clone())?;
    let contraction_shape = Shape::new(
        plan.contracted_labels
            .iter()
            .map(|label| plan.label_extents[label])
            .collect::<Vec<_>>(),
    );
    let contraction_index = DenseIndex::new(contraction_shape)?;
    let indices = tensors
        .iter()
        .map(|tensor| DenseIndex::new(tensor.shape().clone()))
        .collect::<Result<Vec<_>>>()?;
    let mut output = Vec::with_capacity(output_index.len());
    for linear in 0..output_index.len() {
        let output_coords = output_index.coords(linear)?;
        let mut coordinates = std::collections::BTreeMap::new();
        for (label, coordinate) in plan.output_labels.iter().zip(output_coords) {
            coordinates.insert(label.clone(), coordinate);
        }
        let mut sum = Scalar::I(0);
        for contracted_linear in 0..contraction_index.len() {
            let contracted_coords = contraction_index.coords(contracted_linear)?;
            for (label, coordinate) in plan.contracted_labels.iter().zip(contracted_coords) {
                coordinates.insert(label.clone(), coordinate);
            }
            let mut product = Scalar::I(1);
            for ((tensor, index), labels) in tensors.iter().zip(&indices).zip(&plan.operand_labels)
            {
                let input_coords = labels
                    .iter()
                    .zip(tensor.shape().dims())
                    .map(|(label, extent)| {
                        let coordinate = coordinates[label];
                        if *extent == 1 { 0 } else { coordinate }
                    })
                    .collect::<Vec<_>>();
                product = binary_scalar(
                    product,
                    tensor.scalar_at(index.offset(&input_coords)?),
                    dtype,
                    BinaryOp::Mul,
                );
            }
            sum = binary_scalar(sum, product, dtype, BinaryOp::Add);
        }
        output.push(sum);
    }
    TensorData::from_scalars(output_shape, dtype, output)
}

fn einsum_grad(
    values: &[TensorData],
    upstream: NodeId,
    inputs: &[NodeId],
    plan: &crate::EinsumPlan,
    target: usize,
    dtype: DType,
) -> Result<TensorData> {
    let upstream = values
        .get(upstream.index())
        .ok_or(Error::UnknownNode(upstream))?;
    if upstream.shape() != &plan.output_shape() {
        return Err(Error::ShapeMismatch {
            op: "einsum gradient",
            lhs: upstream.shape().clone(),
            rhs: plan.output_shape(),
        });
    }
    let tensors = inputs
        .iter()
        .map(|id| values.get(id.index()).ok_or(Error::UnknownNode(*id)))
        .collect::<Result<Vec<_>>>()?;
    let target_tensor = *tensors.get(target).ok_or(Error::InvalidIndex)?;
    let target_index = DenseIndex::new(target_tensor.shape().clone())?;
    let output_index = DenseIndex::new(upstream.shape().clone())?;
    let contraction_index = DenseIndex::new(Shape::new(
        plan.contracted_labels
            .iter()
            .map(|label| plan.label_extents[label])
            .collect::<Vec<_>>(),
    ))?;
    let indices = tensors
        .iter()
        .map(|tensor| DenseIndex::new(tensor.shape().clone()))
        .collect::<Result<Vec<_>>>()?;
    let mut result = vec![Scalar::I(0); target_index.len()];
    for output_linear in 0..output_index.len() {
        let mut coordinates = std::collections::BTreeMap::new();
        for (label, coordinate) in plan
            .output_labels
            .iter()
            .zip(output_index.coords(output_linear)?)
        {
            coordinates.insert(label.clone(), coordinate);
        }
        for contracted_linear in 0..contraction_index.len() {
            for (label, coordinate) in plan
                .contracted_labels
                .iter()
                .zip(contraction_index.coords(contracted_linear)?)
            {
                coordinates.insert(label.clone(), coordinate);
            }
            let mut contribution = upstream.scalar_at(output_linear);
            for (operand, ((tensor, index), labels)) in tensors
                .iter()
                .zip(&indices)
                .zip(&plan.operand_labels)
                .enumerate()
            {
                if operand == target {
                    continue;
                }
                let coords = labels
                    .iter()
                    .zip(tensor.shape().dims())
                    .map(
                        |(label, extent)| {
                            if *extent == 1 { 0 } else { coordinates[label] }
                        },
                    )
                    .collect::<Vec<_>>();
                contribution = binary_scalar(
                    contribution,
                    tensor.scalar_at(index.offset(&coords)?),
                    dtype,
                    BinaryOp::Mul,
                );
            }
            let labels = &plan.operand_labels[target];
            let coords = labels
                .iter()
                .zip(target_tensor.shape().dims())
                .map(
                    |(label, extent)| {
                        if *extent == 1 { 0 } else { coordinates[label] }
                    },
                )
                .collect::<Vec<_>>();
            let offset = target_index.offset(&coords)?;
            result[offset] = binary_scalar(result[offset], contribution, dtype, BinaryOp::Add);
        }
    }
    TensorData::from_scalars(target_tensor.shape().clone(), dtype, result)
}

#[allow(clippy::too_many_arguments)]
fn einsum_grad_vjp(
    values: &[TensorData],
    cotangent: NodeId,
    upstream: NodeId,
    inputs: &[NodeId],
    plan: &crate::EinsumPlan,
    target: usize,
    wrt: usize,
    dtype: DType,
) -> Result<TensorData> {
    let cotangent = values
        .get(cotangent.index())
        .ok_or(Error::UnknownNode(cotangent))?;
    let upstream = values
        .get(upstream.index())
        .ok_or(Error::UnknownNode(upstream))?;
    let tensors = inputs
        .iter()
        .map(|id| values.get(id.index()).ok_or(Error::UnknownNode(*id)))
        .collect::<Result<Vec<_>>>()?;
    let target_tensor = *tensors.get(target).ok_or(Error::InvalidIndex)?;
    if cotangent.shape() != target_tensor.shape() {
        return Err(Error::GradientShape {
            output: target_tensor.shape().clone(),
            upstream: cotangent.shape().clone(),
        });
    }
    let output_shape = if wrt == inputs.len() {
        plan.output_shape()
    } else {
        tensors.get(wrt).ok_or(Error::InvalidIndex)?.shape().clone()
    };
    if wrt == target {
        return TensorData::from_scalars(
            output_shape.clone(),
            dtype,
            (0..output_shape.numel()?).map(|_| Scalar::I(0)),
        );
    }
    let output_index = DenseIndex::new(plan.output_shape())?;
    let contract_shape = Shape::new(
        plan.contracted_labels
            .iter()
            .map(|label| plan.label_extents[label])
            .collect::<Vec<_>>(),
    );
    let contract_index = DenseIndex::new(contract_shape)?;
    let indices = tensors
        .iter()
        .map(|tensor| DenseIndex::new(tensor.shape().clone()))
        .collect::<Result<Vec<_>>>()?;
    let result_index = DenseIndex::new(output_shape.clone())?;
    let mut result = vec![Scalar::I(0); result_index.len()];
    for out_linear in 0..output_index.len() {
        let mut coordinates = std::collections::BTreeMap::new();
        for (label, coordinate) in plan
            .output_labels
            .iter()
            .zip(output_index.coords(out_linear)?)
        {
            coordinates.insert(label.clone(), coordinate);
        }
        for contracted_linear in 0..contract_index.len() {
            for (label, coordinate) in plan
                .contracted_labels
                .iter()
                .zip(contract_index.coords(contracted_linear)?)
            {
                coordinates.insert(label.clone(), coordinate);
            }
            let target_offset = operand_offset(
                &indices[target],
                target_tensor,
                &plan.operand_labels[target],
                &coordinates,
            )?;
            let c = cotangent.scalar_at(target_offset);
            let mut contribution = if wrt == inputs.len() {
                c
            } else {
                binary_scalar(c, upstream.scalar_at(out_linear), dtype, BinaryOp::Mul)
            };
            for (operand, ((tensor, index), labels)) in tensors
                .iter()
                .zip(&indices)
                .zip(&plan.operand_labels)
                .enumerate()
            {
                if operand != target && operand != wrt {
                    contribution = binary_scalar(
                        contribution,
                        tensor.scalar_at(operand_offset(index, tensor, labels, &coordinates)?),
                        dtype,
                        BinaryOp::Mul,
                    );
                }
            }
            let result_offset = if wrt == inputs.len() {
                out_linear
            } else {
                operand_offset(
                    &indices[wrt],
                    tensors[wrt],
                    &plan.operand_labels[wrt],
                    &coordinates,
                )?
            };
            result[result_offset] =
                binary_scalar(result[result_offset], contribution, dtype, BinaryOp::Add);
        }
    }
    TensorData::from_scalars(output_shape, dtype, result)
}

fn operand_offset(
    index: &DenseIndex,
    tensor: &TensorData,
    labels: &[crate::EinsumLabel],
    coordinates: &std::collections::BTreeMap<crate::EinsumLabel, usize>,
) -> Result<usize> {
    let coords = labels
        .iter()
        .zip(tensor.shape().dims())
        .map(|(label, extent)| {
            let coordinate = coordinates[label];
            if *extent == 1 { 0 } else { coordinate }
        })
        .collect::<Vec<_>>();
    index.offset(&coords)
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
    let output_index = DenseIndex::new(upstream.shape().clone())?;
    let lhs_index = DenseIndex::new(lhs.shape().clone())?;
    let rhs_index = DenseIndex::new(rhs.shape().clone())?;
    let target_index = DenseIndex::new(target.shape().clone())?;
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

fn matmul_grad_vjp(
    cotangent: &TensorData,
    upstream: &TensorData,
    lhs: &TensorData,
    rhs: &TensorData,
    lhs_gradient: bool,
    wrt: u8,
) -> Result<TensorData> {
    let output_shape =
        crate::ir::matmul_shape(lhs.shape(), rhs.shape()).ok_or_else(|| Error::InvalidMatmul {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        })?;
    let target = if lhs_gradient { lhs } else { rhs };
    if cotangent.shape() != target.shape() {
        return Err(Error::GradientShape {
            output: target.shape().clone(),
            upstream: cotangent.shape().clone(),
        });
    }
    let result_shape = match wrt {
        0 => output_shape,
        1 => lhs.shape().clone(),
        2 => rhs.shape().clone(),
        _ => return Err(Error::InvalidIndex),
    };
    let dtype = cotangent.dtype();
    let result_index = DenseIndex::new(result_shape.clone())?;
    let output_index = DenseIndex::new(upstream.shape().clone())?;
    let lhs_index = DenseIndex::new(lhs.shape().clone())?;
    let rhs_index = DenseIndex::new(rhs.shape().clone())?;
    let mut result = vec![Scalar::I(0); result_index.len()];
    let k = *lhs.shape().dims().last().ok_or(Error::InvalidIndex)?;
    for out_linear in 0..output_index.len() {
        let coords = output_index.coords(out_linear)?;
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
            let (target_offset, other, other_offset) = if lhs_gradient {
                (lhs_offset, rhs.scalar_at(rhs_offset), rhs_offset)
            } else {
                (rhs_offset, lhs.scalar_at(lhs_offset), lhs_offset)
            };
            let c = cotangent.scalar_at(target_offset);
            let u = upstream.scalar_at(out_linear);
            match wrt {
                0 => {
                    result[out_linear] = binary_scalar(
                        result[out_linear],
                        binary_scalar(c, other, dtype, BinaryOp::Mul),
                        dtype,
                        BinaryOp::Add,
                    )
                }
                1 if !lhs_gradient => {
                    result[other_offset] = binary_scalar(
                        result[other_offset],
                        binary_scalar(c, u, dtype, BinaryOp::Mul),
                        dtype,
                        BinaryOp::Add,
                    )
                }
                2 if lhs_gradient => {
                    result[other_offset] = binary_scalar(
                        result[other_offset],
                        binary_scalar(c, u, dtype, BinaryOp::Mul),
                        dtype,
                        BinaryOp::Add,
                    )
                }
                _ => {}
            }
        }
    }
    TensorData::from_scalars(result_shape, dtype, result)
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

fn conv2d(
    input: &TensorData,
    weight: &TensorData,
    bias: Option<&TensorData>,
    options: crate::Conv2dOptions,
) -> Result<TensorData> {
    let shape = crate::ir::conv2d_shape(input.shape(), weight.shape(), options)?;
    let dtype = bias.map_or(input.dtype().promote(weight.dtype()), |b| {
        input.dtype().promote(weight.dtype()).promote(b.dtype())
    });
    let out = DenseIndex::new(shape.clone())?;
    let xi = DenseIndex::new(input.shape().clone())?;
    let wi = DenseIndex::new(weight.shape().clone())?;
    let cpg = weight.shape().dims()[1];
    let opg = weight.shape().dims()[0] / options.groups;
    let mut values = vec![Scalar::I(0); out.len()];
    for (n, value) in values.iter_mut().enumerate() {
        let c = out.coords(n)?;
        let group = c[1] / opg;
        for ic in 0..cpg {
            for kh in 0..weight.shape().dims()[2] {
                for kw in 0..weight.shape().dims()[3] {
                    let y = c[2] * options.stride[0] + kh * options.dilation[0];
                    let x = c[3] * options.stride[1] + kw * options.dilation[1];
                    if y >= options.padding[0] && x >= options.padding[2] {
                        let y = y - options.padding[0];
                        let x = x - options.padding[2];
                        if y < input.shape().dims()[2] && x < input.shape().dims()[3] {
                            let a = input.scalar_at(xi.offset(&[c[0], group * cpg + ic, y, x])?);
                            let b = weight.scalar_at(wi.offset(&[c[1], ic, kh, kw])?);
                            *value = binary_scalar(
                                *value,
                                binary_scalar(a, b, dtype, BinaryOp::Mul),
                                dtype,
                                BinaryOp::Add,
                            );
                        }
                    }
                }
            }
        }
        if let Some(b) = bias {
            *value = binary_scalar(*value, b.scalar_at(c[1]), dtype, BinaryOp::Add);
        }
    }
    TensorData::from_scalars(shape, dtype, values)
}

fn conv_transpose2d(
    input: &TensorData,
    weight: &TensorData,
    bias: Option<&TensorData>,
    options: crate::ConvTranspose2dOptions,
) -> Result<TensorData> {
    let shape = crate::ir::conv_transpose2d_shape(input.shape(), weight.shape(), options)?;
    let dtype = bias.map_or(input.dtype().promote(weight.dtype()), |b| {
        input.dtype().promote(weight.dtype()).promote(b.dtype())
    });
    let out = DenseIndex::new(shape.clone())?;
    let xi = DenseIndex::new(input.shape().clone())?;
    let wi = DenseIndex::new(weight.shape().clone())?;
    let icpg = input.shape().dims()[1] / options.groups;
    let ocpg = weight.shape().dims()[1];
    let mut values = vec![Scalar::I(0); out.len()];
    for n in 0..input.shape().dims()[0] {
        for g in 0..options.groups {
            for ic in 0..icpg {
                for iy in 0..input.shape().dims()[2] {
                    for ix in 0..input.shape().dims()[3] {
                        let a = input.scalar_at(xi.offset(&[n, g * icpg + ic, iy, ix])?);
                        for oc in 0..ocpg {
                            for kh in 0..weight.shape().dims()[2] {
                                for kw in 0..weight.shape().dims()[3] {
                                    let y = iy * options.stride[0] + kh * options.dilation[0];
                                    let x = ix * options.stride[1] + kw * options.dilation[1];
                                    if y >= options.padding[0] && x >= options.padding[2] {
                                        let y = y - options.padding[0];
                                        let x = x - options.padding[2];
                                        if y < shape.dims()[2] && x < shape.dims()[3] {
                                            let o = out.offset(&[n, g * ocpg + oc, y, x])?;
                                            let b = weight.scalar_at(wi.offset(&[
                                                g * icpg + ic,
                                                oc,
                                                kh,
                                                kw,
                                            ])?);
                                            values[o] = binary_scalar(
                                                values[o],
                                                binary_scalar(a, b, dtype, BinaryOp::Mul),
                                                dtype,
                                                BinaryOp::Add,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(b) = bias {
        for (i, value) in values.iter_mut().enumerate() {
            let c = out.coords(i)?[1];
            *value = binary_scalar(*value, b.scalar_at(c), dtype, BinaryOp::Add);
        }
    }
    TensorData::from_scalars(shape, dtype, values)
}

fn conv_transpose2d_grad(
    upstream: &TensorData,
    input: &TensorData,
    weight: &TensorData,
    bias: Option<&TensorData>,
    options: crate::ConvTranspose2dOptions,
    target: u8,
) -> Result<TensorData> {
    let shape = crate::ir::conv_transpose2d_shape(input.shape(), weight.shape(), options)?;
    if upstream.shape() != &shape {
        return Err(Error::ShapeMismatch {
            op: "conv_transpose2d gradient",
            lhs: upstream.shape().clone(),
            rhs: shape,
        });
    };
    let (target_shape, dtype) = match target {
        0 => (input.shape().clone(), input.dtype()),
        1 => (weight.shape().clone(), weight.dtype()),
        2 => (
            bias.ok_or(Error::NonDifferentiableIndexing("missing transpose bias"))?
                .shape()
                .clone(),
            bias.unwrap().dtype(),
        ),
        _ => return Err(Error::InvalidIndex),
    };
    if !dtype.is_float() {
        return Err(Error::NonDifferentiableIndexing(
            "transpose convolution gradients require floating point tensors",
        ));
    };
    let oi = DenseIndex::new(shape)?;
    let xi = DenseIndex::new(input.shape().clone())?;
    let wi = DenseIndex::new(weight.shape().clone())?;
    let ti = DenseIndex::new(target_shape.clone())?;
    let icpg = input.shape().dims()[1] / options.groups;
    let ocpg = weight.shape().dims()[1];
    let mut result = vec![Scalar::I(0); ti.len()];
    for n in 0..input.shape().dims()[0] {
        for g in 0..options.groups {
            for ic in 0..icpg {
                for iy in 0..input.shape().dims()[2] {
                    for ix in 0..input.shape().dims()[3] {
                        let xv = input.scalar_at(xi.offset(&[n, g * icpg + ic, iy, ix])?);
                        for oc in 0..ocpg {
                            for kh in 0..weight.shape().dims()[2] {
                                for kw in 0..weight.shape().dims()[3] {
                                    let y = iy * options.stride[0] + kh * options.dilation[0];
                                    let x = ix * options.stride[1] + kw * options.dilation[1];
                                    if y >= options.padding[0] && x >= options.padding[2] {
                                        let y = y - options.padding[0];
                                        let x = x - options.padding[2];
                                        if y < upstream.shape().dims()[2]
                                            && x < upstream.shape().dims()[3]
                                        {
                                            let up = upstream.scalar_at(oi.offset(&[
                                                n,
                                                g * ocpg + oc,
                                                y,
                                                x,
                                            ])?);
                                            let wo = wi.offset(&[g * icpg + ic, oc, kh, kw])?;
                                            let xo = xi.offset(&[n, g * icpg + ic, iy, ix])?;
                                            if target == 0 {
                                                result[xo] = binary_scalar(
                                                    result[xo],
                                                    binary_scalar(
                                                        up,
                                                        weight.scalar_at(wo),
                                                        dtype,
                                                        BinaryOp::Mul,
                                                    ),
                                                    dtype,
                                                    BinaryOp::Add,
                                                )
                                            } else if target == 1 {
                                                result[wo] = binary_scalar(
                                                    result[wo],
                                                    binary_scalar(up, xv, dtype, BinaryOp::Mul),
                                                    dtype,
                                                    BinaryOp::Add,
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if target == 2 {
        for i in 0..oi.len() {
            let c = oi.coords(i)?[1];
            result[c] = binary_scalar(result[c], upstream.scalar_at(i), dtype, BinaryOp::Add)
        }
    }
    TensorData::from_scalars(target_shape, dtype, result)
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_grad_vjp(
    c: &TensorData,
    u: &TensorData,
    x: &TensorData,
    w: &TensorData,
    b: Option<&TensorData>,
    o: crate::ConvTranspose2dOptions,
    target: u8,
    wrt: u8,
) -> Result<TensorData> {
    let shape = match wrt {
        0 => u.shape().clone(),
        1 => x.shape().clone(),
        2 => w.shape().clone(),
        3 => b.ok_or(Error::InvalidIndex)?.shape().clone(),
        _ => return Err(Error::InvalidIndex),
    };
    let expected = match target {
        0 => x.shape(),
        1 => w.shape(),
        2 => b.ok_or(Error::InvalidIndex)?.shape(),
        _ => return Err(Error::InvalidIndex),
    };
    if c.shape() != expected {
        return Err(Error::GradientShape {
            output: expected.clone(),
            upstream: c.shape().clone(),
        });
    }
    let oi = DenseIndex::new(u.shape().clone())?;
    let xi = DenseIndex::new(x.shape().clone())?;
    let wi = DenseIndex::new(w.shape().clone())?;
    let ri = DenseIndex::new(shape.clone())?;
    let mut out = vec![Scalar::I(0); ri.len()];
    let icpg = x.shape().dims()[1] / o.groups;
    let ocpg = w.shape().dims()[1];
    if target == 2 {
        if wrt == 0 {
            for (n, slot) in out.iter_mut().enumerate() {
                let ch = oi.coords(n)?[1];
                *slot = c.scalar_at(ch);
            }
        }
        return TensorData::from_scalars(shape, c.dtype(), out);
    }
    for n in 0..x.shape().dims()[0] {
        for g in 0..o.groups {
            for ic in 0..icpg {
                for iy in 0..x.shape().dims()[2] {
                    for ix in 0..x.shape().dims()[3] {
                        for oc in 0..ocpg {
                            for kh in 0..w.shape().dims()[2] {
                                for kw in 0..w.shape().dims()[3] {
                                    let yy = iy * o.stride[0] + kh * o.dilation[0];
                                    let xx = ix * o.stride[1] + kw * o.dilation[1];
                                    if yy >= o.padding[0] && xx >= o.padding[2] {
                                        let yy = yy - o.padding[0];
                                        let xx = xx - o.padding[2];
                                        if yy < u.shape().dims()[2] && xx < u.shape().dims()[3] {
                                            let no = oi.offset(&[n, g * ocpg + oc, yy, xx])?;
                                            let xo = xi.offset(&[n, g * icpg + ic, iy, ix])?;
                                            let wo = wi.offset(&[g * icpg + ic, oc, kh, kw])?;
                                            let to = if target == 0 { xo } else { wo };
                                            let cv = c.scalar_at(to);
                                            let up = u.scalar_at(no);
                                            match wrt {
                                                0 => {
                                                    out[no] = binary_scalar(
                                                        out[no],
                                                        binary_scalar(
                                                            cv,
                                                            if target == 0 {
                                                                w.scalar_at(wo)
                                                            } else {
                                                                x.scalar_at(xo)
                                                            },
                                                            c.dtype(),
                                                            BinaryOp::Mul,
                                                        ),
                                                        c.dtype(),
                                                        BinaryOp::Add,
                                                    )
                                                }
                                                1 if target == 1 => {
                                                    out[xo] = binary_scalar(
                                                        out[xo],
                                                        binary_scalar(
                                                            cv,
                                                            up,
                                                            c.dtype(),
                                                            BinaryOp::Mul,
                                                        ),
                                                        c.dtype(),
                                                        BinaryOp::Add,
                                                    )
                                                }
                                                2 if target == 0 => {
                                                    out[wo] = binary_scalar(
                                                        out[wo],
                                                        binary_scalar(
                                                            cv,
                                                            up,
                                                            c.dtype(),
                                                            BinaryOp::Mul,
                                                        ),
                                                        c.dtype(),
                                                        BinaryOp::Add,
                                                    )
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    TensorData::from_scalars(shape, c.dtype(), out)
}

fn conv2d_grad(
    upstream: &TensorData,
    input: &TensorData,
    weight: &TensorData,
    bias: Option<&TensorData>,
    options: crate::Conv2dOptions,
    target: u8,
) -> Result<TensorData> {
    let shape = crate::ir::conv2d_shape(input.shape(), weight.shape(), options)?;
    if upstream.shape() != &shape {
        return Err(Error::ShapeMismatch {
            op: "conv2d gradient",
            lhs: upstream.shape().clone(),
            rhs: shape,
        });
    }
    let (target_shape, dtype) = match target {
        0 => (input.shape().clone(), input.dtype()),
        1 => (weight.shape().clone(), weight.dtype()),
        2 => {
            let b = bias.ok_or(Error::NonDifferentiableIndexing("missing conv2d bias"))?;
            (b.shape().clone(), b.dtype())
        }
        _ => return Err(Error::InvalidIndex),
    };
    if !dtype.is_float() {
        return Err(Error::NonDifferentiableIndexing(
            "conv2d gradients require floating point tensors",
        ));
    }
    let out = DenseIndex::new(upstream.shape().clone())?;
    let xi = DenseIndex::new(input.shape().clone())?;
    let wi = DenseIndex::new(weight.shape().clone())?;
    let ti = DenseIndex::new(target_shape.clone())?;
    let cpg = weight.shape().dims()[1];
    let opg = weight.shape().dims()[0] / options.groups;
    let mut result = vec![Scalar::I(0); ti.len()];
    for n in 0..out.len() {
        let c = out.coords(n)?;
        let up = upstream.scalar_at(n);
        let group = c[1] / opg;
        if target == 2 {
            result[c[1]] = binary_scalar(result[c[1]], up, dtype, BinaryOp::Add);
            continue;
        }
        for ic in 0..cpg {
            for kh in 0..weight.shape().dims()[2] {
                for kw in 0..weight.shape().dims()[3] {
                    let y = c[2] * options.stride[0] + kh * options.dilation[0];
                    let x = c[3] * options.stride[1] + kw * options.dilation[1];
                    if y >= options.padding[0] && x >= options.padding[2] {
                        let y = y - options.padding[0];
                        let x = x - options.padding[2];
                        if y < input.shape().dims()[2] && x < input.shape().dims()[3] {
                            let xo = xi.offset(&[c[0], group * cpg + ic, y, x])?;
                            let wo = wi.offset(&[c[1], ic, kh, kw])?;
                            let (offset, other) = if target == 0 {
                                (xo, weight.scalar_at(wo))
                            } else {
                                (wo, input.scalar_at(xo))
                            };
                            result[offset] = binary_scalar(
                                result[offset],
                                binary_scalar(up, other, dtype, BinaryOp::Mul),
                                dtype,
                                BinaryOp::Add,
                            );
                        }
                    }
                }
            }
        }
    }
    TensorData::from_scalars(target_shape, dtype, result)
}

#[allow(clippy::too_many_arguments)]
fn conv2d_grad_vjp(
    c: &TensorData,
    u: &TensorData,
    x: &TensorData,
    w: &TensorData,
    b: Option<&TensorData>,
    o: crate::Conv2dOptions,
    target: u8,
    wrt: u8,
) -> Result<TensorData> {
    let shape = match wrt {
        0 => u.shape().clone(),
        1 => x.shape().clone(),
        2 => w.shape().clone(),
        3 => b.ok_or(Error::InvalidIndex)?.shape().clone(),
        _ => return Err(Error::InvalidIndex),
    };
    let oi = DenseIndex::new(u.shape().clone())?;
    let xi = DenseIndex::new(x.shape().clone())?;
    let wi = DenseIndex::new(w.shape().clone())?;
    let ri = DenseIndex::new(shape.clone())?;
    let expected = match target {
        0 => x.shape(),
        1 => w.shape(),
        2 => b.ok_or(Error::InvalidIndex)?.shape(),
        _ => return Err(Error::InvalidIndex),
    };
    if c.shape() != expected {
        return Err(Error::GradientShape {
            output: expected.clone(),
            upstream: c.shape().clone(),
        });
    }
    let mut out = vec![Scalar::I(0); ri.len()];
    let cpg = w.shape().dims()[1];
    let opg = w.shape().dims()[0] / o.groups;
    for n in 0..oi.len() {
        let co = oi.coords(n)?;
        let group = co[1] / opg;
        let up = u.scalar_at(n);
        if target == 2 {
            if wrt == 0 {
                out[n] = c.scalar_at(co[1]);
            }
            continue;
        }
        for ic in 0..cpg {
            for kh in 0..w.shape().dims()[2] {
                for kw in 0..w.shape().dims()[3] {
                    let yy = co[2] * o.stride[0] + kh * o.dilation[0];
                    let xx = co[3] * o.stride[1] + kw * o.dilation[1];
                    if yy < o.padding[0] || xx < o.padding[2] {
                        continue;
                    }
                    let yy = yy - o.padding[0];
                    let xx = xx - o.padding[2];
                    if yy >= x.shape().dims()[2] || xx >= x.shape().dims()[3] {
                        continue;
                    }
                    let xo = xi.offset(&[co[0], group * cpg + ic, yy, xx])?;
                    let wo = wi.offset(&[co[1], ic, kh, kw])?;
                    let to = if target == 0 { xo } else { wo };
                    let cv = c.scalar_at(to);
                    match wrt {
                        0 => {
                            out[n] = binary_scalar(
                                out[n],
                                binary_scalar(
                                    cv,
                                    if target == 0 {
                                        w.scalar_at(wo)
                                    } else {
                                        x.scalar_at(xo)
                                    },
                                    c.dtype(),
                                    BinaryOp::Mul,
                                ),
                                c.dtype(),
                                BinaryOp::Add,
                            )
                        }
                        1 if target == 1 => {
                            out[xo] = binary_scalar(
                                out[xo],
                                binary_scalar(cv, up, c.dtype(), BinaryOp::Mul),
                                c.dtype(),
                                BinaryOp::Add,
                            )
                        }
                        2 if target == 0 => {
                            out[wo] = binary_scalar(
                                out[wo],
                                binary_scalar(cv, up, c.dtype(), BinaryOp::Mul),
                                c.dtype(),
                                BinaryOp::Add,
                            )
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    TensorData::from_scalars(shape, c.dtype(), out)
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
            "%0 = input(\"x\") : [2] F32\n%1 = constant : [2] F32\n%2 = mul(%0, %1) : [2] F32\nreturn %2"
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
    fn conv2d_oracle_handles_padding_stride_groups_and_bias() {
        let mut graph = Graph::new();
        let x = graph.input("x", [1, 2, 3, 3]);
        let w = graph.input("w", [2, 1, 2, 2]);
        let b = graph.input("b", [2]);
        let y = graph
            .conv2d(
                x,
                w,
                Some(b),
                crate::Conv2dOptions {
                    groups: 2,
                    stride: [2, 1],
                    dilation: [1, 1],
                    padding: [1, 0, 1, 0],
                },
            )
            .unwrap();
        assert_eq!(graph.shape(y).unwrap(), &Shape::from([1, 2, 2, 3]));
        let inputs = HashMap::from([
            (
                "x".into(),
                data(
                    [1, 2, 3, 3],
                    &[
                        1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                    ],
                ),
            ),
            (
                "w".into(),
                data([2, 1, 2, 2], &[1., 1., 1., 1., 2., 0., 0., 0.]),
            ),
            ("b".into(), data([2], &[0., 1.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, y, &inputs).unwrap(),
            data(
                [1, 2, 2, 3],
                &[1., 3., 5., 11., 24., 28., 1., 1., 1., 1., 3., 3.]
            )
        );
        assert!(graph.trace(y).unwrap().to_string().contains("groups=2"));
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

    #[test]
    fn extended_elementwise_ops_cover_float_predicates_and_exact_integer_bits() {
        let mut graph = Graph::new();
        let floats = graph.input_dtype("floats", [3], DType::F64);
        let ints = graph.input_dtype("ints", [3], DType::U64);
        let sin = graph.sin(floats).unwrap();
        let root = graph.sqrt(floats).unwrap();
        let finite = graph.isfinite(floats).unwrap();
        let ones = graph.constant(
            TensorData::from_scalars([3], DType::U64, [Scalar::U(1), Scalar::U(1), Scalar::U(1)])
                .unwrap(),
        );
        let bits = graph.bit_xor(ints, ones).unwrap();
        let shift =
            graph.constant(TensorData::from_scalars([], DType::U64, [Scalar::U(1)]).unwrap());
        let shifted = graph.shl(bits, shift).unwrap();
        let inputs = HashMap::from([
            (
                "floats".into(),
                TensorData::from_scalars(
                    [3],
                    DType::F64,
                    [Scalar::F(0.0), Scalar::F(4.0), Scalar::F(f64::INFINITY)],
                )
                .unwrap(),
            ),
            (
                "ints".into(),
                TensorData::from_scalars(
                    [3],
                    DType::U64,
                    [Scalar::U(u64::MAX), Scalar::U(2), Scalar::U(0)],
                )
                .unwrap(),
            ),
        ]);
        let sin_values = CpuBackend.execute(&graph, sin, &inputs).unwrap();
        assert_eq!(sin_values.scalar_at(0).as_f64(), 0.0);
        assert_eq!(sin_values.scalar_at(1).as_f64(), 4.0_f64.sin());
        assert!(sin_values.scalar_at(2).as_f64().is_nan());
        assert_eq!(
            CpuBackend.execute(&graph, root, &inputs).unwrap().storage(),
            &crate::Storage::F64(vec![0.0, 2.0, f64::INFINITY])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, finite, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![true, true, false])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, shifted, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::U64(vec![u64::MAX - 3, 6, 2])
        );
        assert!(graph.trace(shifted).unwrap().to_string().contains("lshift"));
    }

    #[test]
    fn elementwise_dtype_and_edge_contract_matrix() {
        let all = [
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
        ];
        for dtype in all {
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", [1], dtype);
            let finite = graph.isfinite(x).unwrap();
            assert_eq!(graph.dtype(finite).unwrap(), DType::Bool);
            let expected = if dtype.is_float() { dtype } else { DType::F32 };
            let sin = graph.sin(x).unwrap();
            let round = graph.round(x).unwrap();
            assert_eq!(graph.dtype(sin).unwrap(), expected);
            assert_eq!(graph.dtype(round).unwrap(), dtype);
        }

        let mut graph = Graph::new();
        let bools = graph.input_dtype("b", [2], DType::Bool);
        let bool_bits = graph.bit_xor(bools, bools).unwrap();
        assert_eq!(graph.dtype(bool_bits).unwrap(), DType::Bool);
        assert!(matches!(
            graph.shl(bools, bools),
            Err(Error::InvalidElementwiseDType {
                actual: DType::Bool,
                ..
            })
        ));
        let floats = graph.input_dtype("f", [1], DType::F32);
        assert!(matches!(
            graph.bit_and(floats, floats),
            Err(Error::InvalidElementwiseDType {
                actual: DType::F32,
                ..
            })
        ));
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    bool_bits,
                    &HashMap::from([(
                        "b".into(),
                        TensorData::from_scalars(
                            [2],
                            DType::Bool,
                            [Scalar::Bool(true), Scalar::Bool(false)]
                        )
                        .unwrap()
                    )]),
                )
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![false, false]),
        );

        let mut graph = Graph::new();
        let signed = graph.input_dtype("signed", [1], DType::I64);
        let unsigned = graph.input_dtype("unsigned", [1], DType::U64);
        let mixed = graph.add(signed, unsigned).unwrap();
        assert_eq!(graph.dtype(mixed).unwrap(), DType::F64);
        let small_signed = graph.input_dtype("small_signed", [1], DType::I8);
        let small_unsigned = graph.input_dtype("small_unsigned", [1], DType::U8);
        let mixed_bits = graph.bit_or(small_signed, small_unsigned).unwrap();
        assert_eq!(graph.dtype(mixed_bits).unwrap(), DType::I16);
        let narrow = graph.input_dtype("narrow", [1], DType::I8);
        let neg = graph.neg(narrow).unwrap();
        let abs = graph.abs(narrow).unwrap();
        let input = HashMap::from([
            (
                "signed".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
            ),
            (
                "unsigned".into(),
                TensorData::from_scalars([1], DType::U64, [Scalar::U(0)]).unwrap(),
            ),
            (
                "small_signed".into(),
                TensorData::from_scalars([1], DType::I8, [Scalar::I(0)]).unwrap(),
            ),
            (
                "small_unsigned".into(),
                TensorData::from_scalars([1], DType::U8, [Scalar::U(0)]).unwrap(),
            ),
            (
                "narrow".into(),
                TensorData::from_scalars([1], DType::I8, [Scalar::I(i8::MIN as i64)]).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, neg, &input).unwrap().storage(),
            &crate::Storage::I8(vec![i8::MIN])
        );
        assert_eq!(
            CpuBackend.execute(&graph, abs, &input).unwrap().storage(),
            &crate::Storage::I8(vec![i8::MIN])
        );
    }

    #[test]
    fn integer_failures_and_shift_bounds_are_errors_not_panics() {
        for op in [
            BinaryOp::Div,
            BinaryOp::FloorDiv,
            BinaryOp::TruncDiv,
            BinaryOp::Mod,
            BinaryOp::FMod,
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1], DType::I64);
            let rhs = graph.input_dtype("rhs", [1], DType::I64);
            let output = graph.binary(op, lhs, rhs).unwrap();
            let inputs = HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars([1], DType::I64, [Scalar::I(i64::MIN)]).unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap(),
                ),
            ]);
            // MIN/-1 is defined through the wrapping host operation.
            assert!(CpuBackend.execute(&graph, output, &inputs).is_ok());
            let zero = HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
                ),
            ]);
            assert!(matches!(
                CpuBackend.execute(&graph, output, &zero),
                Err(Error::DivisionByZero { op: _ })
            ));
        }
        for count in [-1, 8, 9] {
            let mut graph = Graph::new();
            let value = graph.input_dtype("value", [1], DType::I8);
            let shift = graph.input_dtype("shift", [1], DType::I8);
            let output = graph.shl(value, shift).unwrap();
            let inputs = HashMap::from([
                (
                    "value".into(),
                    TensorData::from_scalars([1], DType::I8, [Scalar::I(1)]).unwrap(),
                ),
                (
                    "shift".into(),
                    TensorData::from_scalars([1], DType::I8, [Scalar::I(count)]).unwrap(),
                ),
            ]);
            assert!(matches!(
                CpuBackend.execute(&graph, output, &inputs),
                Err(Error::InvalidShiftCount { bits: 8, .. })
            ));
        }
    }

    #[test]
    fn narrow_float_quantization_and_special_values_survive_elementwise_paths() {
        let half = TensorData::from_scalars(
            [5],
            DType::F16,
            [
                Scalar::F(-0.0),
                Scalar::F(f64::INFINITY),
                Scalar::F(f64::NAN),
                Scalar::F(2f64.powi(-24)),
                Scalar::F(2f64.powi(-25)),
            ],
        )
        .unwrap();
        let crate::Storage::F16(bits) = half.storage() else {
            panic!("expected f16 storage")
        };
        assert_eq!(bits[0], 0x8000);
        assert_eq!(bits[1], 0x7c00);
        assert_ne!(bits[2] & 0x03ff, 0); // NaN has a nonzero mantissa.
        assert_eq!(bits[3], 1);
        assert_eq!(bits[4], 0); // exact halfway rounds to even.
        assert_eq!(half.scalar_at(3).as_f64(), 2f64.powi(-24));
        let bf = TensorData::from_scalars(
            [2],
            DType::BF16,
            [
                Scalar::F(-0.0),
                Scalar::F(f32::from_bits(0x0001_0000) as f64),
            ],
        )
        .unwrap();
        assert_eq!(bf.storage(), &crate::Storage::BF16(vec![0x8000, 1]));

        let mut graph = Graph::new();
        let a = graph.input_dtype("a", [2], DType::F16);
        let b = graph.input_dtype("b", [2], DType::BF16);
        let sum = graph.add(a, b).unwrap();
        let rounded = graph.round(a).unwrap();
        let reduced = graph.sum(a, 0).unwrap();
        assert_eq!(graph.dtype(sum).unwrap(), DType::F32);
        assert_eq!(graph.dtype(rounded).unwrap(), DType::F16);
        assert_eq!(graph.dtype(reduced).unwrap(), DType::F16);
        let inputs = HashMap::from([
            (
                "a".into(),
                TensorData::from_scalars([2], DType::F16, [Scalar::F(-0.0), Scalar::F(1.5)])
                    .unwrap(),
            ),
            (
                "b".into(),
                TensorData::from_scalars([2], DType::BF16, [Scalar::F(0.0), Scalar::F(1.0)])
                    .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, sum, &inputs).unwrap().storage(),
            &crate::Storage::F32(vec![0.0, 2.5])
        );
        let result = CpuBackend.execute(&graph, rounded, &inputs).unwrap();
        let crate::Storage::F16(bits) = result.storage() else {
            panic!("expected f16 storage")
        };
        assert_eq!(bits[0], 0x8000);
        assert_eq!(
            CpuBackend
                .execute(&graph, reduced, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![1.5]
        );
    }

    #[test]
    fn transpose_conv_oracle_handles_stride_padding_and_bias() {
        let mut graph = Graph::new();
        let x = graph.input("x", [1, 1, 2, 2]);
        let w = graph.input("w", [1, 1, 2, 2]);
        let b = graph.input("b", [1]);
        let y = graph
            .conv_transpose2d(
                x,
                w,
                Some(b),
                crate::ConvTranspose2dOptions {
                    stride: [2, 2],
                    output_padding: [1, 1],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(graph.shape(y).unwrap(), &Shape::new([1, 1, 5, 5]));
        let values = HashMap::from([
            ("x".into(), data([1, 1, 2, 2], &[1., 2., 3., 4.])),
            ("w".into(), data([1, 1, 2, 2], &[1., 1., 1., 1.])),
            ("b".into(), data([1], &[1.])),
        ]);
        let output = CpuBackend.execute(&graph, y, &values).unwrap();
        assert_eq!(
            output.to_vec_f64(),
            vec![
                2., 2., 3., 3., 1., 2., 2., 3., 3., 1., 4., 4., 5., 5., 1., 4., 4., 5., 5., 1., 1.,
                1., 1., 1., 1.
            ]
        );
    }

    #[test]
    fn transpose_conv1d_lowers_through_the_2d_oracle() {
        let mut graph = Graph::new();
        let x = graph.input("x", [1, 1, 2]);
        let w = graph.input("w", [1, 1, 2]);
        let y = graph
            .conv_transpose1d(
                x,
                w,
                None,
                crate::ConvTranspose1dOptions {
                    stride: 2,
                    output_padding: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(graph.shape(y).unwrap(), &Shape::new([1, 1, 5]));
        let values = HashMap::from([
            ("x".into(), data([1, 1, 2], &[1., 2.])),
            ("w".into(), data([1, 1, 2], &[1., 1.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, y, &values).unwrap().to_vec_f64(),
            vec![1., 1., 2., 2., 0.]
        );
        let loss = graph
            .reduce(y, crate::ReduceKind::Sum, None, false)
            .unwrap();
        let gradient = graph.grad(loss, w).unwrap();
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &values)
                .unwrap()
                .to_vec_f64(),
            vec![3., 3.]
        );
    }
}
