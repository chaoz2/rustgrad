use super::Backend;
use super::float8_reduce;
use crate::engine::{DynamicGradient, DynamicRealized, RuntimeShape};
use crate::engine::dynamic::MixedMaterializationMap;
use crate::index::DenseIndex;
use crate::ir::{DynamicAllocationTarget, DynamicInput, DynamicNodeId, DynamicOp};
use crate::schedule::dynamic::{
    MixedSchedule, ScheduledOutputDesc, schedule_dynamic, schedule_dynamic_unary,
};
use crate::random::threefry2x32;
use crate::{
    BinaryOp, CompareOp, DType, Error, Float8Storage, Graph, LogicalOp, NodeId, Op, Result, Scalar,
    Shape, Storage, TensorData, UnaryOp,
    ir::{RandomKind, RandomStream, normalized_slice},
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

fn random(
    shape: Shape,
    dtype: DType,
    kind: RandomKind,
    stream: RandomStream,
) -> Result<TensorData> {
    crate::random::plan::RandomKernelPlan::new(
        crate::NodeId::from_index(0),
        shape,
        dtype,
        kind,
        stream,
    )?
    .execute()
}

fn random_permutation(shape: Shape, dtype: DType, stream: RandomStream) -> Result<TensorData> {
    let count = shape.numel()?;
    // tinygrad defines randperm as `rand(n).argsort()`. Reusing the typed
    // random plan keeps its word packing and captured reservation identical.
    let random = crate::random::plan::RandomKernelPlan::new(
        crate::NodeId::from_index(0),
        shape.clone(),
        DType::F32,
        RandomKind::Uniform {
            low: 0.0,
            high: 1.0,
        },
        stream,
    )?
    .execute()?;
    let mut indices: Vec<_> = (0..count).collect();
    indices.sort_by(|left, right| {
        random
            .scalar_at(*left)
            .as_f64()
            .total_cmp(&random.scalar_at(*right).as_f64())
            .then(left.cmp(right))
    });
    TensorData::from_scalars(
        shape,
        dtype,
        indices.into_iter().map(|index| Scalar::I(index as i64)),
    )
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
            if let Op::Cast { input, dtype } = node.op
                && (graph.nodes[input.index()].dtype.is_float8() || dtype.is_float8())
                && !(graph.nodes[input.index()].dtype.is_float() && dtype.is_float())
            {
                return Err(Error::UnsupportedDType { dtype: node.dtype });
            }
            if float8_reaches_node(graph, node) && !float8_cpu_capability(&node.op) {
                return Err(Error::UnsupportedDType { dtype: node.dtype });
            }
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
                Op::Random { kind, stream } => {
                    random(node.shape.clone(), node.dtype, *kind, *stream)?
                }
                Op::RandomPermutation { stream } => {
                    random_permutation(node.shape.clone(), node.dtype, *stream)?
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
                } => {
                    let input = &values[input.index()];
                    if input.dtype().is_float8() {
                        float8_reduce::reduce(input, *kind, axes, *keepdim)?
                    } else {
                        reduce(input, *kind, axes, *keepdim, node.dtype)?
                    }
                }
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
                Op::Reshape { input, shape } => {
                    let input = &values[input.index()];
                    input.reorder_raw(shape.clone(), &(0..input.len()).collect::<Vec<_>>())?
                }
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
                Op::StaticIndexUpdate { base, value, plan } => {
                    static_index_update(&values[base.index()], &values[value.index()], plan)?
                }
                Op::StaticIndexUpdateGrad {
                    cotangent,
                    base_shape,
                    value_shape,
                    plan,
                    wrt,
                } => static_index_update_grad(
                    &values[cotangent.index()],
                    base_shape,
                    value_shape,
                    plan,
                    *wrt,
                )?,
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

/// C2's intentionally narrow float8 CPU-oracle surface. Keeping this table
/// beside execution makes unsupported graph operations fail before they reach
/// legacy scalar, reduction, or accelerator-oriented paths.
fn float8_cpu_capability(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. }
            | Op::Constant(_)
            | Op::Detach { .. }
            | Op::Cast { .. }
            | Op::Unary {
                op: UnaryOp::Neg
                    | UnaryOp::Abs
                    | UnaryOp::IsNan
                    | UnaryOp::IsInf
                    | UnaryOp::IsFinite,
                ..
            }
            | Op::Binary {
                op: BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Maximum
                    | BinaryOp::Minimum,
                ..
            }
            | Op::Compare { .. }
            | Op::Reduce { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
            | Op::Expand { .. }
            | Op::Shrink { .. }
            | Op::Stride { .. }
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::StaticIndex { .. }
            | Op::StaticIndexUpdate { .. }
            | Op::MaskedSelect { .. }
            | Op::Select { .. }
            | Op::Scatter { add: false, .. }
            | Op::Matmul { .. }
            | Op::Einsum { .. }
            | Op::Conv2d { .. }
    )
}

fn float8_reaches_node(graph: &Graph, node: &crate::ir::Node) -> bool {
    node.dtype.is_float8()
        || node
            .op
            .value_inputs()
            .iter()
            .any(|input| graph.nodes[input.index()].dtype.is_float8())
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
        self.dynamic_value_memo(graph, output, inputs, &mut HashMap::new())
    }
    fn dynamic_value_memo(
        &self,
        graph: &Graph,
        output: DynamicNodeId,
        inputs: &HashMap<String, TensorData>,
        memo: &mut HashMap<DynamicNodeId, TensorData>,
    ) -> Result<TensorData> {
        if let Some(value) = memo.get(&output) {
            return Ok(value.clone());
        }
        let value = match graph.dynamic_node(output)?.op {
            DynamicOp::Nonzero { input } => nonzero(&self.execute(graph, input, inputs)?),
            DynamicOp::MaskedSelect { input, mask } => {
                let schedule = schedule_dynamic(graph, output).map_err(|error| {
                    Error::DynamicAllocation {
                        reason: error.to_string(),
                    }
                })?;
                schedule
                    .runtime()
                    .plan()
                    .validate_target(DynamicAllocationTarget::CpuInterpreter)
                    .map_err(|error| Error::DynamicAllocation {
                        reason: error.to_string(),
                    })?;
                let input_value = self.execute(graph, input, inputs)?;
                let mask_value = self.execute(graph, mask, inputs)?;
                dynamic_masked_select(&schedule, &input_value, &mask_value)
            }
            DynamicOp::Sum { input } => {
                match &graph.dynamic_node(input)?.op {
                    DynamicOp::MaskedSelect { input: source, mask } => {
                        let schedule = crate::schedule::dynamic::schedule_dynamic_sum(graph, output)
                            .map_err(|error| Error::DynamicAllocation {
                                reason: error.to_string(),
                            })?;
                        dynamic_masked_select_to_reduction(
                            &schedule,
                            &self.execute(graph, *source, inputs)?,
                            &self.execute(graph, *mask, inputs)?,
                            crate::ReduceKind::Sum,
                        )
                    }
                    DynamicOp::Unary { input: selected, .. } => {
                        let DynamicOp::MaskedSelect {
                            input: source,
                            mask,
                        } = &graph.dynamic_node(*selected)?.op
                        else {
                            return Err(Error::DynamicAllocation {
                                reason: "unsupported dynamic sum runtime producer".into(),
                            });
                        };
                        let schedule = crate::schedule::dynamic::schedule_dynamic_sum(graph, output)
                            .map_err(|error| Error::DynamicAllocation {
                                reason: error.to_string(),
                            })?;
                        dynamic_masked_select_to_reduction(
                            &schedule,
                            &self.execute(graph, *source, inputs)?,
                            &self.execute(graph, *mask, inputs)?,
                            crate::ReduceKind::Sum,
                        )
                    }
                    _ => dynamic_sum(&self.dynamic_value_memo(graph, input, inputs, memo)?),
                }
            }
            DynamicOp::Mean { input } => {
                let (source, mask) = match &graph.dynamic_node(input)?.op {
                    DynamicOp::MaskedSelect { input: source, mask } => (*source, *mask),
                    DynamicOp::Unary { input: selected, .. } => {
                        let DynamicOp::MaskedSelect {
                            input: source,
                            mask,
                        } = &graph.dynamic_node(*selected)?.op
                        else {
                            return Err(Error::DynamicAllocation {
                                reason: "unsupported dynamic mean runtime producer".into(),
                            });
                        };
                        (*source, *mask)
                    }
                    _ => {
                        return Err(Error::DynamicAllocation {
                            reason: "unsupported dynamic mean runtime producer".into(),
                        });
                    }
                };
                let schedule = crate::schedule::dynamic::schedule_dynamic_mean(graph, output)
                    .map_err(|error| Error::DynamicAllocation {
                        reason: error.to_string(),
                    })?;
                dynamic_masked_select_to_reduction(
                    &schedule,
                    &self.execute(graph, source, inputs)?,
                    &self.execute(graph, mask, inputs)?,
                    crate::ReduceKind::Mean,
                )
            }
            DynamicOp::Unary { op, input } => {
                if let DynamicOp::MaskedSelect { input: source, mask } = &graph.dynamic_node(input)?.op {
                    let schedule = schedule_dynamic_unary(graph, output).map_err(|error| {
                        Error::DynamicAllocation { reason: error.to_string() }
                    })?;
                    let input_value = self.execute(graph, *source, inputs)?;
                    let mask_value = self.execute(graph, *mask, inputs)?;
                    dynamic_masked_select_unary(&schedule, &input_value, &mask_value, op)
                } else {
                    unary(
                        &self.dynamic_value_memo(graph, input, inputs, memo)?,
                        op,
                        graph.dynamic_node(output)?.dtype,
                    )
                }
            }
            DynamicOp::Binary { op, lhs, rhs } => {
                let lhs = dynamic_operand(self, graph, lhs, inputs, memo)?;
                let rhs = dynamic_operand(self, graph, rhs, inputs, memo)?;
                dynamic_binary(&lhs, &rhs, graph.dynamic_node(output)?.dtype, op)
            }
        }?;
        memo.insert(output, value.clone());
        Ok(value)
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
            DynamicOp::Mean { .. } => Err(Error::NonDifferentiableIndexing(
                "dynamic mean autograd is not implemented",
            )),
            DynamicOp::MaskedSelect { input, mask } if input == wrt => {
                let source = self.execute(graph, input, inputs)?;
                if !source.dtype().is_float() {
                    return Err(Error::NonDifferentiableTarget(input));
                }
                dynamic_masked_select_vjp(&source, &self.execute(graph, mask, inputs)?, upstream)
            }
            DynamicOp::MaskedSelect { .. } => Err(Error::NonDifferentiableTarget(wrt)),
            DynamicOp::Unary { op, input } => {
                let value = self.dynamic_value(graph, input, inputs)?;
                let local = match op {
                    UnaryOp::Neg => TensorData::from_scalars(
                        value.shape().clone(),
                        value.dtype(),
                        (0..value.len()).map(|_| Scalar::F(-1.0)),
                    )?,
                    UnaryOp::Square => TensorData::from_scalars(
                        value.shape().clone(),
                        value.dtype(),
                        (0..value.len()).map(|i| Scalar::F(2.0 * value.scalar_at(i).as_f64())),
                    )?,
                    _ => {
                        return Err(Error::NonDifferentiableIndexing(
                            "unsupported dynamic unary",
                        ));
                    }
                };
                let chained = dynamic_binary(upstream, &local, value.dtype(), BinaryOp::Mul)?;
                self.dynamic_vjp(graph, input, &chained, wrt, inputs)
            }
            DynamicOp::Binary { op, lhs, rhs } => {
                let lhs_value = dynamic_operand(self, graph, lhs, inputs, &mut HashMap::new())?;
                let rhs_value = dynamic_operand(self, graph, rhs, inputs, &mut HashMap::new())?;
                let mut result = None;
                for (operand, local) in [
                    (
                        lhs,
                        match op {
                            BinaryOp::Add => Some(TensorData::from_scalars(
                                upstream.shape().clone(),
                                upstream.dtype(),
                                (0..upstream.len()).map(|i| upstream.scalar_at(i)),
                            )?),
                            BinaryOp::Sub => Some(TensorData::from_scalars(
                                upstream.shape().clone(),
                                upstream.dtype(),
                                (0..upstream.len()).map(|i| upstream.scalar_at(i)),
                            )?),
                            BinaryOp::Mul => Some(dynamic_binary(
                                upstream,
                                &rhs_value,
                                upstream.dtype(),
                                BinaryOp::Mul,
                            )?),
                            _ => None,
                        },
                    ),
                    (
                        rhs,
                        match op {
                            BinaryOp::Add => Some(TensorData::from_scalars(
                                upstream.shape().clone(),
                                upstream.dtype(),
                                (0..upstream.len()).map(|i| upstream.scalar_at(i)),
                            )?),
                            BinaryOp::Sub => Some(TensorData::from_scalars(
                                upstream.shape().clone(),
                                upstream.dtype(),
                                (0..upstream.len()).map(|_| Scalar::F(-1.0)),
                            )?),
                            BinaryOp::Mul => Some(dynamic_binary(
                                upstream,
                                &lhs_value,
                                upstream.dtype(),
                                BinaryOp::Mul,
                            )?),
                            _ => None,
                        },
                    ),
                ] {
                    if let (DynamicInput::Dynamic(id), Some(local)) = (operand, local) {
                        let grad = self.dynamic_vjp(graph, id, &local, wrt, inputs)?;
                        result = Some(match result {
                            None => grad,
                            Some(old) => dynamic_binary(&old, &grad, old.dtype(), BinaryOp::Add)?,
                        });
                    }
                }
                result.ok_or(Error::NonDifferentiableTarget(wrt))
            }
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
    if lhs.dtype().is_float8() || rhs.dtype().is_float8() {
        return float8_binary(lhs, rhs, output_shape, dtype, op);
    }
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
    if lhs.dtype().is_float8() || rhs.dtype().is_float8() {
        return float8_compare(lhs, rhs, output_shape, op);
    }
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

/// Decodes a tagged float8 lane through its format codec. This boundary keeps
/// float8 distinct from U8 storage even when broadcasting selects one lane
/// repeatedly.
fn float8_scalar(data: &TensorData, index: usize) -> Scalar {
    match data.storage() {
        crate::Storage::Float8(values) => Scalar::F(values.format().decode(values.as_raw()[index])),
        _ => data.scalar_at(index),
    }
}

fn float8_binary(
    lhs: &TensorData,
    rhs: &TensorData,
    output_shape: &Shape,
    dtype: DType,
    op: BinaryOp,
) -> Result<TensorData> {
    let data = (0..output_shape.numel()?).map(|linear| {
        let lhs = float8_scalar(lhs, broadcast_offset(linear, output_shape, lhs.shape())).as_f64();
        let rhs = float8_scalar(rhs, broadcast_offset(linear, output_shape, rhs.shape())).as_f64();
        let value = match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
            // tinygrad's Python oracle uses max and implements minimum as
            // negated max, both left-biased on ties and NaNs.
            BinaryOp::Maximum => {
                if rhs > lhs {
                    rhs
                } else {
                    lhs
                }
            }
            BinaryOp::Minimum => {
                if rhs < lhs {
                    rhs
                } else {
                    lhs
                }
            }
            _ => unreachable!("float8 capability table excludes {op:?}"),
        };
        Scalar::F(value)
    });
    TensorData::from_scalars(output_shape.clone(), dtype, data)
}

fn float8_compare(
    lhs: &TensorData,
    rhs: &TensorData,
    output_shape: &Shape,
    op: CompareOp,
) -> Result<TensorData> {
    let data = (0..output_shape.numel()?).map(|linear| {
        let lhs = float8_scalar(lhs, broadcast_offset(linear, output_shape, lhs.shape())).as_f64();
        let rhs = float8_scalar(rhs, broadcast_offset(linear, output_shape, rhs.shape())).as_f64();
        // Match tinygrad's public construction: <= and >= are logical-not of
        // the opposite strict comparison, so they are true for NaN operands.
        let value = match op {
            CompareOp::Eq => lhs == rhs,
            CompareOp::Ne => lhs != rhs,
            CompareOp::Lt => lhs < rhs,
            CompareOp::Le => rhs.partial_cmp(&lhs) != Some(std::cmp::Ordering::Less),
            CompareOp::Gt => rhs < lhs,
            CompareOp::Ge => lhs.partial_cmp(&rhs) != Some(std::cmp::Ordering::Less),
        };
        Scalar::Bool(value)
    });
    TensorData::from_scalars(output_shape.clone(), DType::Bool, data)
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
    if dtype.is_float8() && on_true.dtype() == dtype && on_false.dtype() == dtype {
        let (Storage::Float8(true_storage), Storage::Float8(false_storage)) =
            (on_true.storage(), on_false.storage())
        else {
            unreachable!("float8 dtype has float8 storage");
        };
        let raw = (0..output_shape.numel()?)
            .map(|linear| {
                if condition
                    .scalar_at(broadcast_offset(linear, output_shape, condition.shape()))
                    .as_bool()
                {
                    true_storage.as_raw()[broadcast_offset(linear, output_shape, on_true.shape())]
                } else {
                    false_storage.as_raw()[broadcast_offset(linear, output_shape, on_false.shape())]
                }
            })
            .collect::<Vec<_>>();
        return TensorData::from_storage(
            output_shape.clone(),
            Storage::Float8(Float8Storage::from_raw(true_storage.format(), raw)),
        );
    }
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
            // tinygrad's MAX is `rhs if lhs < rhs else lhs`: an unordered
            // (NaN) comparison keeps the left operand. minimum is derived
            // from MAX and has the corresponding left-biased contract.
            BinaryOp::Maximum => {
                if rhs > lhs {
                    rhs
                } else {
                    lhs
                }
            }
            BinaryOp::Minimum => {
                if rhs < lhs {
                    rhs
                } else {
                    lhs
                }
            }
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
    if input.dtype().is_float8() {
        return float8_unary(input, op, dtype);
    }
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
            // tinygrad composes sign from `ne(0)` and `lt(0)`: both signed
            // zeroes become +0, and an unordered NaN is nonzero but not less
            // than zero, therefore +1.
            UnaryOp::Sign => {
                if value == 0.0 {
                    0.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    1.0
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

fn float8_unary(input: &TensorData, op: UnaryOp, dtype: DType) -> Result<TensorData> {
    let values = (0..input.len()).map(|index| {
        let value = float8_scalar(input, index).as_f64();
        match op {
            UnaryOp::Neg => Scalar::F(-value),
            UnaryOp::Abs => Scalar::F(value.abs()),
            UnaryOp::IsNan => Scalar::Bool(value.is_nan()),
            UnaryOp::IsInf => Scalar::Bool(value.is_infinite()),
            UnaryOp::IsFinite => Scalar::Bool(value.is_finite()),
            _ => unreachable!("float8 capability table excludes {op:?}"),
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
        crate::ReduceKind::Any => Scalar::Bool(false),
        crate::ReduceKind::All => Scalar::Bool(true),
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
            crate::ReduceKind::Any => Scalar::Bool(out[o].as_bool() || v.as_bool()),
            crate::ReduceKind::All => Scalar::Bool(out[o].as_bool() && v.as_bool()),
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
    let offsets = (0..output_shape.numel()?)
        .map(|linear| broadcast_offset(linear, output_shape, input.shape()))
        .collect::<Vec<_>>();
    input.reorder_raw(output_shape.clone(), &offsets)
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
    let offsets = (0..input.len())
        .map(|linear| {
            axes.iter()
                .enumerate()
                .map(|(output_axis, input_axis)| {
                    let coordinate =
                        (linear / output_strides[output_axis]) % output_shape.dims()[output_axis];
                    coordinate * input_strides[*input_axis]
                })
                .sum::<usize>()
        })
        .collect::<Vec<_>>();
    input.reorder_raw(output_shape, &offsets)
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
    let offsets = (0..output_index.len())
        .map(|linear| {
            let coords = output_index.coords(linear)?;
            let source = coords
                .iter()
                .zip(bounds)
                .map(|(coord, (start, _))| coord + start)
                .collect::<Vec<_>>();
            source_index.offset(&source)
        })
        .collect::<Result<Vec<_>>>()?;
    input.reorder_raw(output_shape, &offsets)
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
    let offsets = (0..output_index.len())
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
            source_index.offset(&source)
        })
        .collect::<Result<Vec<_>>>()?;
    input.reorder_raw(output_shape, &offsets)
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
    if dtype.is_float8() && tensors.iter().all(|tensor| tensor.dtype() == dtype) {
        let format = dtype.float8_format().expect("float8 dtype has a format");
        let raw = (0..output_index.len())
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
                let offset =
                    DenseIndex::new(tensors[tensor_index].shape().clone())?.offset(&coords)?;
                match tensors[tensor_index].storage() {
                    Storage::Float8(values) => Ok(values.as_raw()[offset]),
                    _ => unreachable!("float8 dtype has float8 storage"),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        return TensorData::from_storage(
            output_shape.clone(),
            Storage::Float8(Float8Storage::from_raw(format, raw)),
        );
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
    let mut offsets = vec![0; index.len()];
    for (coords, linear) in map {
        offsets[linear] = source_index.offset(&coords)?;
    }
    input.reorder_raw(index.shape().clone(), &offsets)
}

fn static_index(
    input: &TensorData,
    plan: &crate::ir::indexing::StaticIndexPlan,
) -> Result<TensorData> {
    let source = DenseIndex::new(input.shape().clone())?;
    let output = DenseIndex::new(plan.output_shape().clone())?;
    let mut offsets = Vec::with_capacity(output.len());
    for linear in 0..output.len() {
        let coords = plan.source_coords(&output.coords(linear)?)?;
        offsets.push(source.offset(&coords)?);
    }
    input.reorder_raw(plan.output_shape().clone(), &offsets)
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
    let mut values = vec![Scalar::I(0); input.len()];
    for (linear, destination) in plan.source_offsets()?.into_iter().enumerate() {
        values[destination] = binary_scalar(
            values[destination],
            cotangent.scalar_at(linear),
            cotangent.dtype(),
            BinaryOp::Add,
        );
    }
    TensorData::from_scalars(input_shape.clone(), cotangent.dtype(), values)
}

fn static_index_update(
    input: &TensorData,
    value: &TensorData,
    plan: &crate::ir::indexing::StaticIndexPlan,
) -> Result<TensorData> {
    let mut output = input.clone();
    output.static_index_update_from(plan, value)?;
    Ok(output)
}

fn static_index_update_grad(
    cotangent: &TensorData,
    base_shape: &Shape,
    value_shape: &Shape,
    plan: &crate::ir::indexing::StaticIndexPlan,
    wrt: crate::StaticIndexUpdateWrt,
) -> Result<TensorData> {
    if cotangent.dtype() != DType::F32 || cotangent.shape() != base_shape {
        return Err(Error::NonDifferentiableIndexing(
            "static index update gradients require F32 base cotangent",
        ));
    }
    let selected = DenseIndex::new(plan.output_shape().clone())?;
    let value = DenseIndex::new(value_shape.clone())?;
    let mut final_writer = vec![None; base_shape.numel()?];
    for (linear, target) in plan.source_offsets()?.into_iter().enumerate() {
        final_writer[target] = Some(linear);
    }
    match wrt {
        crate::StaticIndexUpdateWrt::Base => {
            let values: Vec<Scalar> = final_writer
                .iter()
                .enumerate()
                .map(|(offset, writer)| match writer {
                    Some(_) => Scalar::F(0.0),
                    None => cotangent.scalar_at(offset),
                })
                .collect();
            TensorData::from_scalars(base_shape.clone(), DType::F32, values)
        }
        crate::StaticIndexUpdateWrt::Value => {
            let mut values = vec![Scalar::F(0.0); value.len()];
            for (target, writer) in final_writer.into_iter().enumerate() {
                let Some(selected_linear) = writer else {
                    continue;
                };
                let coords = selected.coords(selected_linear)?;
                let value_offset = value.broadcast_offset(&selected, &coords)?;
                let current = values[value_offset];
                values[value_offset] = binary_scalar(
                    current,
                    cotangent.scalar_at(target),
                    DType::F32,
                    BinaryOp::Add,
                );
            }
            TensorData::from_scalars(value_shape.clone(), DType::F32, values)
        }
    }
}

fn indexed_scatter(
    base: &TensorData,
    index: &TensorData,
    updates: &TensorData,
    axis: usize,
    add: bool,
    dtype: DType,
) -> Result<TensorData> {
    if !add && base.dtype().is_float8() && base.dtype() == updates.dtype() && dtype == base.dtype()
    {
        let base_index = DenseIndex::new(base.shape().clone())?;
        let update_index = DenseIndex::new(updates.shape().clone())?;
        let index_index = DenseIndex::new(index.shape().clone())?;
        let mut destinations = Vec::with_capacity(index.len());
        let mut sources = Vec::with_capacity(index.len());
        for (destination, update_linear) in indexed_coordinates(base, index, axis)? {
            destinations.push(base_index.offset(&destination)?);
            sources.push(update_index.offset(&index_index.coords(update_linear)?)?);
        }
        let mut output = base.clone();
        output.replace_raw_offsets(updates, &destinations, &sources)?;
        return Ok(output);
    }
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
    let mut positions = Vec::with_capacity(size);
    for linear in 0..input_index.len() {
        let coords = input_index.coords(linear)?;
        let mask_offset = mask_index.broadcast_offset(&input_index, &coords)?;
        if mask.scalar_at(mask_offset).as_bool() && positions.len() < size {
            positions.push(linear);
        }
    }
    if let Storage::Float8(values) = input.storage() {
        let mut raw = positions
            .into_iter()
            .map(|offset| values.as_raw()[offset])
            .collect::<Vec<_>>();
        raw.resize(size, values.format().encode(fill.as_f64()));
        return TensorData::from_storage(
            [size],
            Storage::Float8(Float8Storage::from_raw(values.format(), raw)),
        );
    }
    let mut output = positions
        .into_iter()
        .map(|offset| input.scalar_at(offset))
        .collect::<Vec<_>>();
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

fn dynamic_masked_select(
    schedule: &MixedSchedule,
    input: &TensorData,
    mask: &TensorData,
) -> Result<TensorData> {
    schedule
        .runtime()
        .plan()
        .validate_target(DynamicAllocationTarget::CpuInterpreter)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    schedule
        .runtime()
        .plan()
        .validate_bindings(input, mask)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let positions = masked_positions(input, mask)?;
    let mut materializations = MixedMaterializationMap::new(schedule).map_err(|error| {
        Error::DynamicAllocation {
            reason: error.to_string(),
        }
    })?;
    let allocation_item = schedule
        .items
        .iter()
        .find(|item| {
            matches!(
                item.kind,
                crate::schedule::dynamic::MixedScheduleItemKind::MaterializeMaskedSelect
            )
        })
        .ok_or_else(|| Error::DynamicAllocation {
            reason: "mixed runtime schedule has no materialization item".into(),
        })?;
    let runtime_output = schedule
        .runtime_output(allocation_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    materializations
        .allocate_after_count(schedule, positions.len())
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let allocation = materializations
        .allocation_for_consumer(schedule, allocation_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    if allocation.dtype != runtime_output.dtype || allocation.shape.rank() != runtime_output.rank {
        return Err(Error::DynamicAllocation {
            reason: "runtime allocation does not match mixed output descriptor".into(),
        });
    }
    let values = positions
        .into_iter()
        .map(|position| input.scalar_at(position));
    TensorData::from_scalars(allocation.shape.clone(), allocation.dtype, values)
}

/// Executes the bounded count→allocate→materialize→allocate→unary chain.
/// The two descriptors are allocated separately before either result is
/// materialized, so the source can remain live through the unary consumer.
fn dynamic_masked_select_unary(
    schedule: &MixedSchedule,
    input: &TensorData,
    mask: &TensorData,
    op: UnaryOp,
) -> Result<TensorData> {
    schedule
        .runtime()
        .plan()
        .validate_target(DynamicAllocationTarget::CpuInterpreter)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    schedule
        .runtime()
        .plan()
        .validate_bindings(input, mask)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let positions = masked_positions(input, mask)?;
    let mut materializations =
        MixedMaterializationMap::new(schedule).map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let source_item = schedule
        .items
        .iter()
        .find(|item| {
            matches!(
                item.kind,
                crate::schedule::dynamic::MixedScheduleItemKind::MaterializeMaskedSelect
            )
        })
    .ok_or_else(|| Error::DynamicAllocation {
        reason: "mixed runtime schedule has no masked-select materialization".into(),
    })?;
    let unary_item = schedule
        .items
        .iter()
        .find(|item| {
            matches!(
                item.kind,
                crate::schedule::dynamic::MixedScheduleItemKind::DynamicUnary { .. }
            )
        })
    .ok_or_else(|| Error::DynamicAllocation {
        reason: "mixed runtime schedule has no dynamic unary item".into(),
    })?;
    let source_descriptor = schedule
        .runtime_output(source_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let unary_descriptor = schedule
        .runtime_output(unary_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    materializations
        .allocate_after_count(schedule, positions.len())
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let source_allocation = materializations
        .allocation_for_consumer(schedule, source_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    if source_allocation.dtype != source_descriptor.dtype
        || source_allocation.shape.rank() != source_descriptor.rank
    {
        return Err(Error::DynamicAllocation {
            reason: "runtime source allocation does not match descriptor".into(),
        });
    }
    let selected = TensorData::from_scalars(
        source_allocation.shape.clone(),
        source_allocation.dtype,
        positions.into_iter().map(|position| input.scalar_at(position)),
    )?;
    materializations
        .allocate_item_output_after_count(schedule, unary_item.id, selected.len())
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let output_allocation = materializations
        .allocation_for_item_output(schedule, unary_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    if output_allocation.dtype != unary_descriptor.dtype
        || output_allocation.shape.rank() != unary_descriptor.rank
    {
        return Err(Error::DynamicAllocation {
            reason: "runtime unary allocation does not match descriptor".into(),
        });
    }
    let result = unary(&selected, op, unary_descriptor.dtype)?;
    if result.shape() != &output_allocation.shape || result.dtype() != output_allocation.dtype {
        return Err(Error::DynamicAllocation {
            reason: "dynamic unary result does not match exact output allocation".into(),
        });
    }
    Ok(result)
}

/// Materializes the validated runtime chain once, then consumes its final
/// runtime buffer through the sole permitted fixed scalar reduction bridge.
fn dynamic_masked_select_to_reduction(
    schedule: &MixedSchedule,
    input: &TensorData,
    mask: &TensorData,
    kind: crate::ReduceKind,
) -> Result<TensorData> {
    schedule
        .runtime()
        .plan()
        .validate_target(DynamicAllocationTarget::CpuInterpreter)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    schedule
        .runtime()
        .plan()
        .validate_bindings(input, mask)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let positions = masked_positions(input, mask)?;
    let mut materializations =
        MixedMaterializationMap::new(schedule).map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let source_item = schedule
        .items
        .iter()
        .find(|item| {
            matches!(
                item.kind,
                crate::schedule::dynamic::MixedScheduleItemKind::MaterializeMaskedSelect
            )
        })
        .ok_or_else(|| Error::DynamicAllocation {
            reason: "mixed runtime schedule has no masked-select materialization".into(),
        })?;
    materializations
        .allocate_after_count(schedule, positions.len())
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let source_allocation = materializations
        .allocation_for_consumer(schedule, source_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let mut value = TensorData::from_scalars(
        source_allocation.shape.clone(),
        source_allocation.dtype,
        positions.into_iter().map(|position| input.scalar_at(position)),
    )?;
    if let Some(unary_item) = schedule.items.iter().find(|item| {
        matches!(
            item.kind,
            crate::schedule::dynamic::MixedScheduleItemKind::DynamicUnary { .. }
        )
    }) {
        let crate::schedule::dynamic::MixedScheduleItemKind::DynamicUnary { op } = &unary_item.kind else {
            unreachable!("dynamic unary item kind was checked")
        };
        materializations
            .allocate_item_output_after_count(schedule, unary_item.id, value.len())
            .map_err(|error| Error::DynamicAllocation {
                reason: error.to_string(),
            })?;
        let allocation = materializations
            .allocation_for_item_output(schedule, unary_item.id)
            .map_err(|error| Error::DynamicAllocation {
                reason: error.to_string(),
            })?;
        value = unary(&value, *op, allocation.dtype)?;
        if value.shape() != &allocation.shape || value.dtype() != allocation.dtype {
            return Err(Error::DynamicAllocation {
                reason: "dynamic unary result does not match exact output allocation".into(),
            });
        }
    }
    let reduction_item = schedule
        .items
        .iter()
        .find(|item| {
            matches!(
                item.kind,
                crate::schedule::dynamic::MixedScheduleItemKind::DynamicReduceSum
                    | crate::schedule::dynamic::MixedScheduleItemKind::DynamicReduceMean
            )
        })
        .ok_or_else(|| Error::DynamicAllocation {
            reason: "mixed runtime schedule has no dynamic reduction item".into(),
        })?;
    materializations
        .allocation_for_consumer(schedule, reduction_item.id)
        .map_err(|error| Error::DynamicAllocation {
            reason: error.to_string(),
        })?;
    let ScheduledOutputDesc::Fixed(output) = &reduction_item.output else {
        return Err(Error::DynamicAllocation {
            reason: "dynamic reduction item lacks a fixed scalar descriptor".into(),
        });
    };
    if !matches!(
        (kind, &reduction_item.kind),
        (
            crate::ReduceKind::Sum,
            crate::schedule::dynamic::MixedScheduleItemKind::DynamicReduceSum
        ) | (
            crate::ReduceKind::Mean,
            crate::schedule::dynamic::MixedScheduleItemKind::DynamicReduceMean
        )
    ) {
        return Err(Error::DynamicAllocation {
            reason: "dynamic reduction operation does not match typed schedule item".into(),
        });
    }
    let result = reduce(&value, kind, &[0], false, output.dtype)?;
    if result.shape() != &output.shape || result.dtype() != output.dtype {
        return Err(Error::DynamicAllocation {
            reason: "dynamic reduction result does not match fixed descriptor".into(),
        });
    }
    Ok(result)
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

fn dynamic_operand(
    backend: &CpuBackend,
    graph: &Graph,
    input: DynamicInput,
    bindings: &HashMap<String, TensorData>,
    memo: &mut HashMap<DynamicNodeId, TensorData>,
) -> Result<TensorData> {
    match input {
        DynamicInput::Dynamic(id) => backend.dynamic_value_memo(graph, id, bindings, memo),
        DynamicInput::StaticScalar(id) => backend.execute(graph, id, bindings),
    }
}
fn dynamic_binary(
    lhs: &TensorData,
    rhs: &TensorData,
    dtype: DType,
    op: BinaryOp,
) -> Result<TensorData> {
    let len = lhs.len().max(rhs.len());
    if !((lhs.len() == len || lhs.len() == 1) && (rhs.len() == len || rhs.len() == 1)) {
        return Err(Error::InvalidIndex);
    }
    TensorData::from_scalars(
        [len],
        dtype,
        (0..len).map(|i| {
            binary_scalar(
                lhs.scalar_at(if lhs.len() == 1 { 0 } else { i }),
                rhs.scalar_at(if rhs.len() == 1 { 0 } else { i }),
                dtype,
                op,
            )
        }),
    )
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
    if inputs.len() == 1
        && plan.contracted_labels.is_empty()
        && plan.operand_labels[0]
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == plan.operand_labels[0].len()
    {
        let axes = plan
            .output_labels
            .iter()
            .map(|label| {
                plan.operand_labels[0]
                    .iter()
                    .position(|axis| axis == label)
                    .ok_or(Error::InvalidIndex)
            })
            .collect::<Result<Vec<_>>>()?;
        return permute(tensors[0], &axes);
    }
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
    let float8_contract = crate::backend::float8_contract::einsum_policy(dtype);
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
            let mut product_f32 = 1.0f32;
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
                let value = tensor.scalar_at(index.offset(&input_coords)?);
                if float8_contract.is_some() {
                    product_f32 *= value.as_f64() as f32;
                } else {
                    product = binary_scalar(product, value, dtype, BinaryOp::Mul);
                }
            }
            if let Some(policy) = float8_contract {
                sum = policy.accumulate(sum, Scalar::F(f64::from(product_f32)), Scalar::I(1));
            } else {
                sum = binary_scalar(sum, product, dtype, BinaryOp::Add);
            }
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
    // tinygrad's dot lowers to multiply followed by sum: float8 uses the
    // established F32 reduction accumulator and narrows once at the result.
    let float8_contract = crate::backend::float8_contract::matmul_policy(dtype);
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
            let left = lhs.scalar_at(matmul_lhs_offset(
                &lhs_index,
                &coords,
                inner,
                lhs.shape().rank() == 1,
                rhs.shape().rank() == 1,
            )?);
            let right = rhs.scalar_at(matmul_rhs_offset(
                &rhs_index,
                &coords,
                inner,
                lhs.shape().rank() == 1,
                rhs.shape().rank() == 1,
            )?);
            if let Some(policy) = float8_contract {
                *value = policy.accumulate(*value, left, right);
                continue;
            }
            let product = binary_scalar(left, right, dtype, BinaryOp::Mul);
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
    let float8_contract = crate::backend::float8_contract::conv2d_policy(dtype);
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
                            if let Some(policy) = float8_contract {
                                *value = policy.accumulate(*value, a, b);
                            } else {
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
    use crate::{Float8Storage, ReduceKind, Storage, ir::indexing::StaticIndex};

    type Float8C2Build = fn(&mut Graph, NodeId) -> Result<NodeId>;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    fn float8_data() -> TensorData {
        TensorData::from_storage(
            Shape::from([2]),
            crate::Storage::Float8(crate::Float8Storage::from_raw(
                crate::Float8Format::E4M3,
                vec![0x80, 0xff],
            )),
        )
        .unwrap()
    }

    #[test]
    fn float8_matmul_accumulates_once_then_narrows() {
        let formats = [
            (crate::DType::F8E4M3, crate::Float8Format::E4M3),
            (crate::DType::F8E5M2, crate::Float8Format::E5M2),
            (crate::DType::F8E4M3FNUZ, crate::Float8Format::E4M3FNUZ),
            (crate::DType::F8E5M2FNUZ, crate::Float8Format::E5M2FNUZ),
        ];
        for (dtype, format) in formats {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1, 2], dtype);
            let rhs = graph.input_dtype("rhs", [2, 1], dtype);
            let out = graph.matmul(lhs, rhs).unwrap();
            let inputs = HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_storage(
                        Shape::from([1, 2]),
                        Storage::Float8(Float8Storage::from_raw(
                            format,
                            vec![format.encode(1.0), format.encode(2.0)],
                        )),
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_storage(
                        Shape::from([2, 1]),
                        Storage::Float8(Float8Storage::from_raw(
                            format,
                            vec![format.encode(3.0), format.encode(4.0)],
                        )),
                    )
                    .unwrap(),
                ),
            ]);
            let actual = CpuBackend.execute(&graph, out, &inputs).unwrap();
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual.scalar_at(0).as_f64(),
                format.decode(format.encode(11.0))
            );
        }
    }

    #[test]
    fn float8_conv2d_uses_the_contraction_policy() {
        for (dtype, format) in [
            (DType::F8E4M3, crate::Float8Format::E4M3),
            (DType::F8E5M2, crate::Float8Format::E5M2),
            (DType::F8E4M3FNUZ, crate::Float8Format::E4M3FNUZ),
            (DType::F8E5M2FNUZ, crate::Float8Format::E5M2FNUZ),
        ] {
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", [1, 1, 1, 2], dtype);
            let w = graph.input_dtype("w", [1, 1, 1, 2], dtype);
            let out = graph
                .conv2d(x, w, None, crate::Conv2dOptions::default())
                .unwrap();
            let inputs = HashMap::from([
                (
                    "x".into(),
                    TensorData::from_storage(
                        [1, 1, 1, 2],
                        Storage::Float8(Float8Storage::from_raw(
                            format,
                            vec![format.encode(1.0), format.encode(2.0)],
                        )),
                    )
                    .unwrap(),
                ),
                (
                    "w".into(),
                    TensorData::from_storage(
                        [1, 1, 1, 2],
                        Storage::Float8(Float8Storage::from_raw(
                            format,
                            vec![format.encode(3.0), format.encode(4.0)],
                        )),
                    )
                    .unwrap(),
                ),
            ]);
            let actual = CpuBackend.execute(&graph, out, &inputs).unwrap();
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual.scalar_at(0).as_f64(),
                format.decode(format.encode(11.0))
            );
        }
    }

    fn float8_values(dtype: DType, values: impl IntoIterator<Item = f64>) -> TensorData {
        let values = values.into_iter().collect::<Vec<_>>();
        TensorData::from_storage(
            Shape::from([values.len()]),
            crate::Storage::Float8(crate::Float8Storage::from_f64(
                dtype.float8_format().expect("float8 dtype"),
                values,
            )),
        )
        .unwrap()
    }

    fn float8_raw(dtype: DType, bytes: Vec<u8>) -> TensorData {
        TensorData::from_storage(
            Shape::from([bytes.len()]),
            crate::Storage::Float8(crate::Float8Storage::from_raw(
                dtype.float8_format().expect("float8 dtype"),
                bytes,
            )),
        )
        .unwrap()
    }

    fn float8_bytes(data: &TensorData) -> Vec<u8> {
        match data.storage() {
            Storage::Float8(values) => values.as_raw().to_vec(),
            _ => panic!("expected float8 storage"),
        }
    }

    #[test]
    fn float8_constants_inputs_and_unsupported_nodes_fail_closed() {
        let mut transport = Graph::new();
        let constant = transport.constant(float8_data());
        let detached = transport.detach(constant).unwrap();
        assert_eq!(
            CpuBackend
                .execute(&transport, detached, &HashMap::new())
                .unwrap(),
            float8_data()
        );

        let mut input_graph = Graph::new();
        let input = input_graph.input_dtype("x", [2], DType::F8E4M3);
        assert_eq!(
            CpuBackend
                .execute(
                    &input_graph,
                    input,
                    &HashMap::from([("x".into(), float8_data())])
                )
                .unwrap(),
            float8_data()
        );

        let cast = transport.cast(constant, DType::F32).unwrap();
        assert_eq!(
            CpuBackend
                .execute(&transport, cast, &HashMap::new())
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            [0, 0, 0, 0x80, 0, 0, 0xc0, 0x7f]
        );

        let cases: [fn(&mut Graph, NodeId) -> Result<NodeId>; 1] =
            [|graph, value| graph.exp(value)];
        for build in cases {
            let mut graph = Graph::new();
            let value = graph.constant(float8_data());
            let output = build(&mut graph, value).unwrap();
            assert!(matches!(
                CpuBackend.execute(&graph, output, &HashMap::new()),
                Err(Error::UnsupportedDType { .. })
            ));
        }
    }

    #[test]
    fn float8_c2_alu_and_comparisons_quantize_through_typed_storage() {
        let raw = TensorData::from_storage(
            Shape::from([2]),
            crate::Storage::Float8(crate::Float8Storage::from_raw(
                crate::Float8Format::E4M3,
                vec![0x38, 0x40],
            )),
        )
        .unwrap();
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F8E4M3);
        let doubled = graph.add(x, x).unwrap();
        let negated = graph.neg(x).unwrap();
        let absolute = graph.abs(negated).unwrap();
        let ordered = graph.lt(x, doubled).unwrap();
        let inputs = HashMap::from([("x".into(), raw)]);
        assert_eq!(
            CpuBackend
                .execute(&graph, doubled, &inputs)
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            vec![0x40, 0x48]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, absolute, &inputs)
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            vec![0x38, 0x40]
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, ordered, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![true, true])
        );
        assert!(graph.trace(doubled).unwrap().to_string().contains("add"));
    }

    #[test]
    fn float8_c2_all_families_same_format_alu_quantizes_once() {
        let formats = [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ];
        for dtype in formats {
            let input = float8_values(dtype, [1.0, 2.0]);
            let format = dtype.float8_format().unwrap();
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", [2], dtype);
            let negated = graph.neg(x).unwrap();
            let absolute = graph.abs(negated).unwrap();
            let outputs = [
                (graph.add(x, x).unwrap(), vec![2.0, 4.0]),
                (graph.sub(x, x).unwrap(), vec![0.0, 0.0]),
                (graph.mul(x, x).unwrap(), vec![1.0, 4.0]),
                (graph.div(x, x).unwrap(), vec![1.0, 1.0]),
                (graph.maximum(x, x).unwrap(), vec![1.0, 2.0]),
                (graph.minimum(x, x).unwrap(), vec![1.0, 2.0]),
                (negated, vec![-1.0, -2.0]),
                (absolute, vec![1.0, 2.0]),
            ];
            for (output, expected) in outputs {
                assert_eq!(graph.dtype(output).unwrap(), dtype, "{dtype:?}");
                assert_eq!(
                    CpuBackend
                        .execute(
                            &graph,
                            output,
                            &HashMap::from([("x".into(), input.clone())])
                        )
                        .unwrap()
                        .to_le_bytes()
                        .unwrap(),
                    expected
                        .into_iter()
                        .map(|value| format.encode(value))
                        .collect::<Vec<_>>(),
                    "{dtype:?} {}",
                    graph.trace(output).unwrap().steps.last().unwrap().operation
                );
            }
        }
    }

    #[test]
    fn float8_c2_promotes_cross_format_and_wider_operands() {
        let lhs = float8_values(DType::F8E4M3, [1.0, 2.0]);
        let rhs = float8_values(DType::F8E5M2, [0.5, 1.0]);
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F8E4M3);
        let y = graph.input_dtype("y", [2], DType::F8E5M2);
        let cross = graph.add(x, y).unwrap();
        assert_eq!(graph.dtype(cross).unwrap(), DType::F16);
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    cross,
                    &HashMap::from([("x".into(), lhs.clone()), ("y".into(), rhs)])
                )
                .unwrap()
                .to_vec_f64(),
            vec![1.5, 3.0]
        );

        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F8E4M3);
        let wide = graph.constant(TensorData::new([2], vec![0.25f32, 0.5]).unwrap());
        let f32_output = graph.add(x, wide).unwrap();
        assert_eq!(graph.dtype(f32_output).unwrap(), DType::F32);
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    f32_output,
                    &HashMap::from([("x".into(), lhs.clone())])
                )
                .unwrap()
                .to_vec_f64(),
            vec![1.25, 2.5]
        );

        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F8E4M3);
        let integer = graph.constant(
            TensorData::from_scalars([2], DType::I32, [Scalar::I(1), Scalar::I(2)]).unwrap(),
        );
        let narrow = graph.add(x, integer).unwrap();
        assert_eq!(graph.dtype(narrow).unwrap(), DType::F8E4M3);
        assert_eq!(
            CpuBackend
                .execute(&graph, narrow, &HashMap::from([("x".into(), lhs)]))
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            vec![0x40, 0x48]
        );
    }

    #[test]
    fn float8_c2_predicates_comparisons_and_extrema_audit_special_values() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let nan = if dtype.is_float8() && matches!(dtype, DType::F8E4M3FNUZ | DType::F8E5M2FNUZ)
            {
                0x80
            } else {
                0x7f
            };
            let zero = if matches!(dtype, DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) {
                0x00
            } else {
                0x80
            };
            let x = float8_raw(dtype, vec![nan, zero, 0x38]);
            let y = float8_raw(dtype, vec![nan, 0x00, 0x40]);
            let mut graph = Graph::new();
            let left = graph.input_dtype("x", [3], dtype);
            let right = graph.input_dtype("y", [3], dtype);
            let comparisons = [
                (graph.eq(left, right).unwrap(), vec![false, true, false]),
                (graph.ne(left, right).unwrap(), vec![true, false, true]),
                (graph.lt(left, right).unwrap(), vec![false, false, true]),
                (graph.le(left, right).unwrap(), vec![true, true, true]),
                (graph.gt(left, right).unwrap(), vec![false, false, false]),
                (graph.ge(left, right).unwrap(), vec![true, true, false]),
            ];
            for (output, expected) in comparisons {
                assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
                assert_eq!(
                    CpuBackend
                        .execute(
                            &graph,
                            output,
                            &HashMap::from([("x".into(), x.clone()), ("y".into(), y.clone())])
                        )
                        .unwrap()
                        .storage(),
                    &crate::Storage::Bool(expected),
                    "{dtype:?} {}",
                    graph.trace(output).unwrap().steps.last().unwrap().operation
                );
            }
            let predicates = [
                (graph.isnan(left).unwrap(), vec![true, false, false]),
                (graph.isinf(left).unwrap(), vec![false, false, false]),
                (graph.isfinite(left).unwrap(), vec![false, true, true]),
            ];
            for (output, expected) in predicates {
                assert_eq!(
                    CpuBackend
                        .execute(
                            &graph,
                            output,
                            &HashMap::from([("x".into(), x.clone()), ("y".into(), y.clone())])
                        )
                        .unwrap()
                        .storage(),
                    &crate::Storage::Bool(expected),
                    "{dtype:?} {}",
                    graph.trace(output).unwrap().steps.last().unwrap().operation
                );
            }
            let maximum = graph.maximum(left, right).unwrap();
            let minimum = graph.minimum(left, right).unwrap();
            let inputs = HashMap::from([("x".into(), x), ("y".into(), y)]);
            assert!(
                CpuBackend
                    .execute(&graph, maximum, &inputs)
                    .unwrap()
                    .to_vec_f64()[0]
                    .is_nan()
            );
            assert!(
                CpuBackend
                    .execute(&graph, minimum, &inputs)
                    .unwrap()
                    .to_vec_f64()[0]
                    .is_nan()
            );
        }
    }

    #[test]
    fn float8_c2_broadcasts_and_preserves_empty_output_dtype() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2, 1], DType::F8E4M3);
        let y = graph.input_dtype("y", [2], DType::F8E4M3);
        let output = graph.add(x, y).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 2]));
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        (
                            "x".into(),
                            TensorData::from_storage(
                                [2, 1],
                                float8_values(DType::F8E4M3, [1.0, 2.0]).storage().clone(),
                            )
                            .unwrap(),
                        ),
                        ("y".into(), float8_values(DType::F8E4M3, [0.5, 1.0])),
                    ]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![1.5, 2.0, 2.5, 3.0]
        );
        let mut graph = Graph::new();
        let empty = graph.input_dtype("x", [0], DType::F8E5M2);
        let output = graph.neg(empty).unwrap();
        let realized = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("x".into(), float8_raw(DType::F8E5M2, vec![]))]),
            )
            .unwrap();
        assert_eq!(realized.dtype(), DType::F8E5M2);
        assert!(realized.is_empty());
    }

    #[test]
    fn float8_c4_movement_reorders_all_raw_lanes_without_codec_round_trips() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let raw = (0..=u8::MAX).collect::<Vec<_>>();
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [2, 2, 64], dtype);
            let permuted = graph.permute(input, [2, 0, 1]).unwrap();
            let reshaped = graph.reshape(permuted, [4, 64]).unwrap();
            let flipped = graph
                .stride(
                    reshaped,
                    [
                        crate::Slice {
                            start: None,
                            stop: None,
                            step: -1,
                        },
                        crate::Slice {
                            start: None,
                            stop: None,
                            step: 1,
                        },
                    ],
                )
                .unwrap();
            let output = graph.shrink(flipped, [(1, 4), (0, 64)]).unwrap();
            let actual = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([(
                        "x".into(),
                        TensorData::from_storage(
                            [2, 2, 64],
                            Storage::Float8(Float8Storage::from_raw(
                                dtype.float8_format().unwrap(),
                                raw.clone(),
                            )),
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap();
            let mut permuted = Vec::with_capacity(256);
            for k in 0..64 {
                for i in 0..2 {
                    for j in 0..2 {
                        permuted.push(raw[i * 128 + j * 64 + k]);
                    }
                }
            }
            let expected = permuted
                .chunks_exact(64)
                .rev()
                .skip(1)
                .flat_map(|chunk| chunk.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(float8_bytes(&actual), expected, "{dtype:?}");

            let scalar = graph.input_dtype("scalar", [], dtype);
            let expanded = graph.expand(scalar, [2, 0, 3]).unwrap();
            let expanded = CpuBackend
                .execute(
                    &graph,
                    expanded,
                    &HashMap::from([
                        (
                            "x".into(),
                            TensorData::from_storage(
                                [2, 2, 64],
                                Storage::Float8(Float8Storage::from_raw(
                                    dtype.float8_format().unwrap(),
                                    raw,
                                )),
                            )
                            .unwrap(),
                        ),
                        (
                            "scalar".into(),
                            TensorData::from_storage(
                                [],
                                Storage::Float8(Float8Storage::from_raw(
                                    dtype.float8_format().unwrap(),
                                    vec![0xff],
                                )),
                            )
                            .unwrap(),
                        ),
                    ]),
                )
                .unwrap();
            assert!(float8_bytes(&expanded).is_empty(), "{dtype:?}");
        }
    }

    #[test]
    fn float8_c4_index_update_mask_and_select_preserve_raw_winners() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let mut graph = Graph::new();
            let base = graph.input_dtype("base", [2], dtype);
            let value = graph.input_dtype("value", [3], dtype);
            let updated = graph
                .static_index_update(
                    base,
                    &[StaticIndex::Advanced {
                        shape: [3].into(),
                        values: vec![1, 1, 0],
                    }],
                    value,
                )
                .unwrap();
            let indexed = graph
                .static_index(
                    updated,
                    &[StaticIndex::Advanced {
                        shape: [3].into(),
                        values: vec![1, 0, 1],
                    }],
                )
                .unwrap();
            let mask = graph.input_dtype("mask", [3], DType::Bool);
            let selected = graph
                .masked_select(indexed, mask, 5, Scalar::F(-1.0))
                .unwrap();
            let actual = CpuBackend
                .execute(
                    &graph,
                    selected,
                    &HashMap::from([
                        ("base".into(), float8_raw(dtype, vec![0x01, 0x02])),
                        ("value".into(), float8_raw(dtype, vec![0x90, 0x91, 0x92])),
                        (
                            "mask".into(),
                            TensorData::from_scalars(
                                [3],
                                DType::Bool,
                                [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
                            )
                            .unwrap(),
                        ),
                    ]),
                )
                .unwrap();
            let format = dtype.float8_format().unwrap();
            assert_eq!(
                float8_bytes(&actual),
                vec![
                    0x91,
                    0x91,
                    format.encode(-1.0),
                    format.encode(-1.0),
                    format.encode(-1.0)
                ]
            );

            let condition = graph.input_dtype("condition", [2, 1], DType::Bool);
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [2], dtype);
            let output = graph.select(condition, lhs, rhs).unwrap();
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            let actual = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        ("base".into(), float8_raw(dtype, vec![0x01, 0x02])),
                        ("value".into(), float8_raw(dtype, vec![0x90, 0x91, 0x92])),
                        (
                            "mask".into(),
                            TensorData::from_scalars([3], DType::Bool, [Scalar::Bool(true); 3])
                                .unwrap(),
                        ),
                        (
                            "condition".into(),
                            TensorData::from_scalars(
                                [2, 1],
                                DType::Bool,
                                [Scalar::Bool(true), Scalar::Bool(false)],
                            )
                            .unwrap(),
                        ),
                        (
                            "lhs".into(),
                            TensorData::from_storage(
                                [2, 1],
                                Storage::Float8(Float8Storage::from_raw(format, vec![0xfe, 0xfd])),
                            )
                            .unwrap(),
                        ),
                        (
                            "rhs".into(),
                            TensorData::from_storage(
                                [2],
                                Storage::Float8(Float8Storage::from_raw(format, vec![0xfc, 0xfb])),
                            )
                            .unwrap(),
                        ),
                    ]),
                )
                .unwrap();
            assert_eq!(
                float8_bytes(&actual),
                vec![0xfe, 0xfe, 0xfc, 0xfb],
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn float8_c4_concat_and_replacement_scatter_keep_raw_storage() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [2, 1], dtype);
            let concatenated = graph.concat([lhs, rhs], 1).unwrap();
            let index = graph.input_dtype("index", [2, 2], DType::I32);
            let updates = graph.input_dtype("updates", [2, 2], dtype);
            let output = graph.scatter(concatenated, index, updates, 1).unwrap();
            let actual = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        (
                            "lhs".into(),
                            TensorData::from_storage(
                                [2, 1],
                                float8_raw(dtype, vec![0x10, 0x11]).storage().clone(),
                            )
                            .unwrap(),
                        ),
                        (
                            "rhs".into(),
                            TensorData::from_storage(
                                [2, 1],
                                float8_raw(dtype, vec![0x20, 0x21]).storage().clone(),
                            )
                            .unwrap(),
                        ),
                        (
                            "index".into(),
                            TensorData::from_scalars(
                                [2, 2],
                                DType::I32,
                                [Scalar::I(1), Scalar::I(1), Scalar::I(0), Scalar::I(0)],
                            )
                            .unwrap(),
                        ),
                        (
                            "updates".into(),
                            TensorData::from_storage(
                                [2, 2],
                                float8_raw(dtype, vec![0x90, 0x91, 0x92, 0x93])
                                    .storage()
                                    .clone(),
                            )
                            .unwrap(),
                        ),
                    ]),
                )
                .unwrap();
            assert_eq!(
                float8_bytes(&actual),
                vec![0x10, 0x91, 0x93, 0x21],
                "{dtype:?}"
            );
            assert!(graph.trace(output).unwrap().to_string().contains("scatter"));
        }
    }

    #[test]
    fn float8_c4_cross_format_select_promotes_before_materialization() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let mut graph = Graph::new();
            let condition = graph.input_dtype("condition", [2], DType::Bool);
            let narrow = graph.input_dtype("narrow", [2], dtype);
            let wide = graph.input_dtype("wide", [2], DType::F16);
            let output = graph.select(condition, narrow, wide).unwrap();
            assert_eq!(graph.dtype(output).unwrap(), DType::F16);
            let actual = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        (
                            "condition".into(),
                            TensorData::from_scalars(
                                [2],
                                DType::Bool,
                                [Scalar::Bool(true), Scalar::Bool(false)],
                            )
                            .unwrap(),
                        ),
                        ("narrow".into(), float8_values(dtype, [1.0, 3.0])),
                        (
                            "wide".into(),
                            TensorData::from_scalars(
                                [2],
                                DType::F16,
                                [Scalar::F(9.0), Scalar::F(2.0)],
                            )
                            .unwrap(),
                        ),
                    ]),
                )
                .unwrap();
            assert_eq!(actual.dtype(), DType::F16);
            assert_eq!(actual.to_vec_f64(), vec![1.0, 2.0], "{dtype:?}");
        }
    }

    #[test]
    fn float8_c5_rejects_outside_cpu_oracle_table() {
        let cases: [(&str, Float8C2Build); 3] = [
            ("exp", |graph, x| graph.exp(x)),
            ("argmax", |graph, x| graph.argmax(x, Some(0), false)),
            ("pad", |graph, x| {
                graph.pad(x, [(0, 0), (0, 0)], Scalar::F(0.0))
            }),
        ];
        for (name, build) in cases {
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", [1, 1], DType::F8E4M3);
            let output = build(&mut graph, x).unwrap();
            assert!(
                matches!(
                    CpuBackend.execute(
                        &graph,
                        output,
                        &HashMap::from([(
                            "x".into(),
                            TensorData::from_storage(
                                [1, 1],
                                float8_values(DType::F8E4M3, [1.0]).storage().clone(),
                            )
                            .unwrap(),
                        )]),
                    ),
                    Err(Error::UnsupportedDType { .. })
                ),
                "{name}"
            );
        }

        let mut graph = Graph::new();
        let random = graph.rand([1], DType::F8E4M3, 7).unwrap();
        assert!(matches!(
            CpuBackend.execute(&graph, random, &HashMap::new()),
            Err(Error::UnsupportedDType {
                dtype: DType::F8E4M3
            })
        ));

        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1], DType::F8E4M3);
        let loss = graph.add(x, x).unwrap();
        assert!(matches!(
            graph.grad(loss, x),
            Err(Error::UnsupportedDType {
                dtype: DType::F8E4M3
            })
        ));

        let mut graph = Graph::new();
        let base = graph.input_dtype("base", [1], DType::F8E4M3);
        let index = graph.input_dtype("index", [1], DType::I32);
        let update = graph.input_dtype("update", [1], DType::F8E4M3);
        let scatter_add = graph.scatter_add(base, index, update, 0).unwrap();
        assert!(matches!(
            CpuBackend.execute(
                &graph,
                scatter_add,
                &HashMap::from([
                    ("base".into(), float8_raw(DType::F8E4M3, vec![0x38])),
                    (
                        "index".into(),
                        TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap(),
                    ),
                    ("update".into(), float8_raw(DType::F8E4M3, vec![0x38])),
                ]),
            ),
            Err(Error::UnsupportedDType {
                dtype: DType::F8E4M3
            })
        ));
    }

    #[test]
    fn float8_c7_einsum_uses_f32_contraction_and_raw_reorder() {
        for (dtype, format) in [
            (DType::F8E4M3, crate::Float8Format::E4M3),
            (DType::F8E5M2, crate::Float8Format::E5M2),
            (DType::F8E4M3FNUZ, crate::Float8Format::E4M3FNUZ),
            (DType::F8E5M2FNUZ, crate::Float8Format::E5M2FNUZ),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1, 2], dtype);
            let rhs = graph.input_dtype("rhs", [2, 1], dtype);
            let dot = graph.einsum("ij,jk->ik", &[lhs, rhs]).unwrap();
            let transpose = graph.einsum("ij->ji", &[lhs]).unwrap();
            let inputs = HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_storage(
                        [1, 2],
                        float8_raw(dtype, vec![format.encode(1.0), format.encode(2.0)])
                            .storage()
                            .clone(),
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_storage(
                        [2, 1],
                        float8_raw(dtype, vec![format.encode(3.0), format.encode(4.0)])
                            .storage()
                            .clone(),
                    )
                    .unwrap(),
                ),
            ]);
            let actual = CpuBackend.execute(&graph, dot, &inputs).unwrap();
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual.scalar_at(0).as_f64(),
                format.decode(format.encode(11.0))
            );
            let raw = CpuBackend.execute(&graph, transpose, &inputs).unwrap();
            assert_eq!(
                float8_bytes(&raw),
                vec![format.encode(1.0), format.encode(2.0)]
            );
        }
    }

    #[test]
    fn float8_c3_reductions_use_the_source_audited_policy_table() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let format = dtype.float8_format().unwrap();
            let first = format.decode(0x38);
            let second = format.decode(0x30);
            let third = format.decode(0x80);
            let fourth = format.decode(0x01);
            let source = TensorData::from_storage(
                [2, 2],
                Storage::Float8(Float8Storage::from_raw(
                    format,
                    vec![0x38, 0x30, 0x80, 0x01],
                )),
            )
            .unwrap();
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", [2, 2], dtype);
            let sum = graph
                .reduce(x, ReduceKind::Sum, Some(vec![1]), true)
                .unwrap();
            let mean = graph
                .reduce(x, ReduceKind::Mean, Some(vec![1]), false)
                .unwrap();
            let product = graph
                .reduce(x, ReduceKind::Product, Some(vec![1]), false)
                .unwrap();
            let maximum = graph
                .reduce(x, ReduceKind::Max, Some(vec![1]), false)
                .unwrap();
            let minimum = graph
                .reduce(x, ReduceKind::Min, Some(vec![1]), false)
                .unwrap();
            let inputs = HashMap::from([("x".into(), source)]);
            for output in [sum, mean, product, maximum, minimum] {
                assert_eq!(graph.dtype(output).unwrap(), dtype, "{dtype:?}");
                assert!(matches!(
                    CpuBackend
                        .execute(&graph, output, &inputs)
                        .unwrap()
                        .storage(),
                    Storage::Float8(_)
                ));
            }
            // Sum/mean use the decoded lanes in F32; product re-quantizes each step.
            assert_eq!(
                CpuBackend
                    .execute(&graph, sum, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                vec![
                    format.encode(f64::from(first as f32 + second as f32)),
                    format.encode(f64::from(third as f32 + fourth as f32))
                ]
            );
            assert_eq!(
                CpuBackend
                    .execute(&graph, mean, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                vec![
                    format.encode(f64::from((first as f32 + second as f32) / 2.0)),
                    format.encode(f64::from((third as f32 + fourth as f32) / 2.0))
                ]
            );
            assert_eq!(
                CpuBackend
                    .execute(&graph, product, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                vec![
                    format.encode(format.decode(format.encode(first * second))),
                    format.encode(format.decode(format.encode(third * fourth)))
                ]
            );
            // Strict comparisons retain the first tied signed-zero byte and ignore NaNs.
            assert_eq!(
                CpuBackend
                    .execute(&graph, maximum, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap()[0],
                0x38
            );
            assert_eq!(
                CpuBackend
                    .execute(&graph, minimum, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap()[1],
                if matches!(dtype, DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) {
                    0x01
                } else {
                    0x80
                }
            );
            assert!(graph.trace(sum).unwrap().to_string().contains("Sum"));

            let empty = TensorData::from_storage(
                [2, 0],
                Storage::Float8(Float8Storage::from_raw(format, vec![])),
            )
            .unwrap();
            let mut empty_graph = Graph::new();
            let empty_x = empty_graph.input_dtype("x", [2, 0], dtype);
            for kind in [ReduceKind::Sum, ReduceKind::Mean, ReduceKind::Product] {
                let output = empty_graph
                    .reduce(empty_x, kind, Some(vec![1]), false)
                    .unwrap();
                let bytes = CpuBackend
                    .execute(
                        &empty_graph,
                        output,
                        &HashMap::from([("x".into(), empty.clone())]),
                    )
                    .unwrap()
                    .to_le_bytes()
                    .unwrap();
                let expected = match kind {
                    ReduceKind::Sum => format.encode(0.0),
                    ReduceKind::Mean => format.encode(f64::NAN),
                    ReduceKind::Product => format.encode(1.0),
                    _ => unreachable!(),
                };
                assert_eq!(bytes, vec![expected; 2], "{dtype:?} {kind:?}");
            }
            for (kind, name) in [(ReduceKind::Max, "max"), (ReduceKind::Min, "min")] {
                assert!(
                    matches!(empty_graph.reduce(empty_x, kind, Some(vec![1]), false), Err(Error::EmptyReduction { op, .. }) if op == name)
                );
            }

            let mut scalar_graph = Graph::new();
            let scalar = scalar_graph.input_dtype("x", [], dtype);
            let scalar_sum = scalar_graph
                .reduce(scalar, ReduceKind::Sum, None, false)
                .unwrap();
            assert_eq!(scalar_graph.shape(scalar_sum).unwrap(), &Shape::new([]));
            assert_eq!(
                CpuBackend
                    .execute(
                        &scalar_graph,
                        scalar_sum,
                        &HashMap::from([(
                            "x".into(),
                            TensorData::from_storage(
                                [],
                                Storage::Float8(Float8Storage::from_raw(
                                    format,
                                    vec![format.encode(-0.0)],
                                )),
                            )
                            .unwrap(),
                        )]),
                    )
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                vec![format.encode(0.0)]
            );
        }
    }

    #[test]
    fn float8_cast_matrix_remains_a_typed_cpu_boundary() {
        let float8 = [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ];
        let floating = [DType::F16, DType::BF16, DType::F32, DType::F64];
        let scalar_cases = [0.0, -0.0, 1.0625, f64::INFINITY, f64::NAN];

        for source in float8.into_iter().chain(floating) {
            for target in float8.into_iter().chain(floating) {
                let source_data = TensorData::from_scalars(
                    [scalar_cases.len()],
                    source,
                    scalar_cases.into_iter().map(Scalar::F),
                )
                .unwrap();
                let mut graph = Graph::new();
                let input = graph.input_dtype("x", [scalar_cases.len()], source);
                let cast = graph.cast(input, target).unwrap();
                let output = CpuBackend
                    .execute(&graph, cast, &HashMap::from([("x".into(), source_data)]))
                    .unwrap();
                assert_eq!(output.dtype(), target, "{source:?} -> {target:?}");
                assert!(graph.trace(cast).unwrap().to_string().contains("cast"));
            }
        }

        let exact = [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
        ];
        for float8 in float8 {
            for exact_dtype in exact {
                for (source, target) in [(float8, exact_dtype), (exact_dtype, float8)] {
                    let data = TensorData::from_scalars([1], source, [Scalar::F(1.0)]).unwrap();
                    let mut graph = Graph::new();
                    let input = graph.input_dtype("x", [1], source);
                    let cast = graph.cast(input, target).unwrap();
                    assert!(
                        matches!(
                            CpuBackend.execute(&graph, cast, &HashMap::from([("x".into(), data)])),
                            Err(Error::UnsupportedDType { .. })
                        ),
                        "{source:?} -> {target:?}"
                    );
                }
            }
        }
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
    fn float_extrema_follow_tinygrad_left_biased_nan_and_tie_rules() {
        let cases: &[(&str, f32, f32, bool, bool)] = &[
            ("lhs_nan", f32::NAN, 2.0, true, true),
            ("rhs_nan", 2.0, f32::NAN, false, false),
            ("tie", -0.0, 0.0, false, false),
        ];
        for &(name, lhs_value, rhs_value, max_nan, min_nan) in cases {
            let mut graph = Graph::new();
            let lhs = graph.input("lhs", []);
            let rhs = graph.input("rhs", []);
            let maximum = graph.maximum(lhs, rhs).unwrap();
            let minimum = graph.minimum(lhs, rhs).unwrap();
            let inputs = HashMap::from([
                ("lhs".into(), data([], &[lhs_value])),
                ("rhs".into(), data([], &[rhs_value])),
            ]);
            let max = CpuBackend
                .execute(&graph, maximum, &inputs)
                .unwrap()
                .values()[0];
            let min = CpuBackend
                .execute(&graph, minimum, &inputs)
                .unwrap()
                .values()[0];
            assert_eq!(max.is_nan(), max_nan, "{name} maximum");
            assert_eq!(min.is_nan(), min_nan, "{name} minimum");
            if name == "tie" {
                assert_eq!(max.to_bits(), (-0.0f32).to_bits());
                assert_eq!(min.to_bits(), (-0.0f32).to_bits());
            }
            assert!(
                graph
                    .trace(maximum)
                    .unwrap()
                    .to_string()
                    .contains("maximum")
            );
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
        assert_eq!(graph.dtype(reduced).unwrap(), DType::F32);
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

    #[test]
    fn dynamic_elementwise_chain_has_exact_forward_and_vjp() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let mask = graph.input_dtype("mask", [3], DType::Bool);
        let scalar = graph.constant(TensorData::scalar(2.0));
        let selected = graph.masked_select_dynamic(x, mask).unwrap();
        let square = graph.dynamic_unary(selected, UnaryOp::Square).unwrap();
        let product = graph
            .dynamic_binary(
                square,
                crate::ir::DynamicInput::StaticScalar(scalar),
                BinaryOp::Mul,
            )
            .unwrap();
        let loss = graph.dynamic_sum(product).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([3], &[1., 2., 3.])),
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
        let result = CpuBackend
            .execute_dynamic_gradient(&graph, loss, x, &inputs)
            .unwrap();
        assert_eq!(result.loss.output.to_vec_f64(), vec![20.]);
        assert_eq!(result.gradient.to_vec_f64(), vec![4., 0., 12.]);
    }

    #[test]
    fn float_sign_uses_tinygrad_comparison_contract() {
        let cases: &[(&str, f32, f32)] = &[
            ("negative", -2.0, -1.0),
            ("positive", 2.0, 1.0),
            ("negative_zero", -0.0, 0.0),
            ("positive_zero", 0.0, 0.0),
            ("nan", f32::NAN, 1.0),
            ("infinity", f32::INFINITY, 1.0),
        ];
        for &(name, value, expected) in cases {
            let mut graph = Graph::new();
            let input = graph.input("x", []);
            let output = graph.sign(input).unwrap();
            let result = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("x".into(), data([], &[value]))]),
                )
                .unwrap();
            let actual = result.values()[0];
            assert_eq!(actual.to_bits(), expected.to_bits(), "{name}");
            assert!(graph.trace(output).unwrap().to_string().contains("sign"));
        }
    }

    #[test]
    fn masked_select_dynamic_unary_uses_exact_distinct_runtime_outputs() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let mask = graph.input_dtype("mask", [3], DType::Bool);
        let selected = graph.masked_select_dynamic(x, mask).unwrap();
        let negated = graph.dynamic_unary(selected, UnaryOp::Neg).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([3], &[1., -2., 3.])),
            ("mask".into(), TensorData::from_scalars(
                [3], DType::Bool,
                [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
            ).unwrap()),
        ]);
        assert_eq!(
            CpuBackend.execute_dynamic(&graph, negated, &inputs).unwrap().output.to_vec_f64(),
            vec![-1., -3.],
        );
    }

    #[test]
    fn masked_select_dynamic_unary_preserves_zero_domain() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let mask = graph.input_dtype("mask", [2], DType::Bool);
        let selected = graph.masked_select_dynamic(x, mask).unwrap();
        let squared = graph.dynamic_unary(selected, UnaryOp::Square).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2], &[1., 2.])),
            ("mask".into(), TensorData::from_scalars(
                [2], DType::Bool, [Scalar::Bool(false), Scalar::Bool(false)],
            ).unwrap()),
        ]);
        let output = CpuBackend.execute_dynamic(&graph, squared, &inputs).unwrap().output;
        assert_eq!(output.shape(), &Shape::from([0]));
        assert_eq!(output.dtype(), DType::F32);
    }

    #[test]
    fn masked_select_dynamic_sum_uses_scalar_identity_for_empty_domain() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let mask = graph.input_dtype("mask", [3], DType::Bool);
        let selected = graph.masked_select_dynamic(x, mask).unwrap();
        let sum = graph.dynamic_sum(selected).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([3], &[1., 2., 3.])),
            (
                "mask".into(),
                TensorData::from_scalars(
                    [3],
                    DType::Bool,
                    [Scalar::Bool(false), Scalar::Bool(false), Scalar::Bool(false)],
                )
                .unwrap(),
            ),
        ]);
        let output = CpuBackend.execute_dynamic(&graph, sum, &inputs).unwrap().output;
        assert_eq!(output.shape(), &Shape::from([]));
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.to_vec_f64(), vec![0.]);
    }

    #[test]
    fn masked_select_dynamic_unary_sum_preserves_canonical_reduction_values() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let mask = graph.input_dtype("mask", [3], DType::Bool);
        let selected = graph.masked_select_dynamic(x, mask).unwrap();
        let squared = graph.dynamic_unary(selected, UnaryOp::Square).unwrap();
        let sum = graph.dynamic_sum(squared).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([3], &[2., 3., 4.])),
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
        assert_eq!(
            CpuBackend.execute_dynamic(&graph, sum, &inputs)
                .unwrap()
                .output
                .to_vec_f64(),
            vec![20.],
        );
    }

    #[test]
    fn dynamic_diamond_accumulates_and_cross_graph_rejects() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let mask = graph.input_dtype("m", [2], DType::Bool);
        let selected = graph.masked_select_dynamic(x, mask).unwrap();
        let diamond = graph
            .dynamic_binary(
                selected,
                crate::ir::DynamicInput::Dynamic(selected),
                BinaryOp::Add,
            )
            .unwrap();
        let loss = graph.dynamic_sum(diamond).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2], &[3., 4.])),
            (
                "m".into(),
                TensorData::from_scalars(
                    [2],
                    DType::Bool,
                    [Scalar::Bool(true), Scalar::Bool(true)],
                )
                .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute_dynamic_gradient(&graph, loss, x, &inputs)
                .unwrap()
                .gradient
                .to_vec_f64(),
            vec![2., 2.]
        );
        let mut other = Graph::new();
        let y = other.input("y", [1]);
        let foreign = other.nonzero(y).unwrap();
        assert!(graph.dynamic_sum(foreign).is_err());
    }
}
