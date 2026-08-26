use crate::{
    BinaryOp, DType, Error, Graph, NodeId, Op, Result, Scalar, Shape, TensorData, UnaryOp,
};
use std::collections::BTreeSet;

impl Graph {
    /// Appends the reverse-mode derivative of a one-element `loss` with
    /// respect to `wrt` and returns the derivative node.
    pub fn grad(&mut self, loss: NodeId, wrt: NodeId) -> Result<NodeId> {
        self.grad_with(loss, wrt, None, true)
    }

    /// Builds a reverse-mode derivative graph. An implicit seed is allowed only
    /// for one-element outputs; `create_graph` controls whether this derivative
    /// itself retains gradient edges for higher-order differentiation.
    pub fn grad_with(
        &mut self,
        loss: NodeId,
        wrt: NodeId,
        upstream: Option<NodeId>,
        create_graph: bool,
    ) -> Result<NodeId> {
        if let Some(dtype) = self
            .nodes
            .iter()
            .take(loss.index().saturating_add(1))
            .find_map(|node| node.dtype.is_float8().then_some(node.dtype))
        {
            return Err(Error::UnsupportedDType { dtype });
        }
        self.validate_prefix_scan_reverse(loss)?;
        let original_len = self.nodes.len();
        let loss_shape = self.node(loss)?.shape.clone();
        let target = self.node(wrt)?;
        if !target.requires_grad {
            // Preserve operator-specific non-float diagnostics (for example
            // Conv2d's float-only contract) while frozen float leaves fail
            // immediately and clearly.
            if target.dtype.is_float() {
                return Err(Error::NonDifferentiableTarget(wrt));
            }
        }
        let previous_enabled = self.grad_enabled;
        self.grad_enabled = create_graph;
        let result = self.grad_with_inner(loss, wrt, upstream, original_len, loss_shape);
        self.grad_enabled = previous_enabled;
        result
    }

    fn grad_with_inner(
        &mut self,
        loss: NodeId,
        wrt: NodeId,
        upstream: Option<NodeId>,
        original_len: usize,
        loss_shape: Shape,
    ) -> Result<NodeId> {
        let seed = if let Some(upstream) = upstream {
            let upstream_node = self.node(upstream)?;
            if upstream_node.shape != loss_shape {
                return Err(Error::GradientShape {
                    output: loss_shape,
                    upstream: upstream_node.shape.clone(),
                });
            }
            if !upstream_node.dtype.is_float() {
                return Err(Error::NonDifferentiableTarget(upstream));
            }
            upstream
        } else if loss_shape.numel()? != 1 {
            return Err(Error::NonScalarLoss(loss_shape));
        } else {
            let seed_data = filled(self.node(loss)?.shape.clone(), 1.0)?;
            self.constant(seed_data)
        };
        let mut grads = vec![None; original_len];
        grads[loss.index()] = Some(seed);

        for index in (0..original_len).rev() {
            let Some(upstream) = grads[index] else {
                continue;
            };
            let node = NodeId(index);
            let op = self.node(node)?.op.clone();
            match op {
                Op::Input { .. }
                | Op::Constant(_)
                | Op::Random { .. }
                | Op::RandomPermutation { .. }
                | Op::Detach { .. } => {}
                Op::Cast { input, .. } => self.accumulate(&mut grads, input, upstream)?,
                // Predicates are intentionally nondifferentiable. They only
                // route gradients when consumed by Select below.
                Op::Compare { .. } | Op::Logical { .. } => {}
                Op::Unary { op, input } => {
                    let local = match op {
                        UnaryOp::Neg => self.neg(upstream)?,
                        UnaryOp::Exp => self.mul(upstream, node)?,
                        UnaryOp::Log => self.div(upstream, input)?,
                        UnaryOp::Relu => {
                            let mask = self.step(input)?;
                            self.mul(upstream, mask)?
                        }
                        UnaryOp::Step => {
                            let zeros = filled(self.node(input)?.shape.clone(), 0.0)?;
                            self.constant(zeros)
                        }
                        UnaryOp::Abs => {
                            let sign = self.sign(input)?;
                            self.mul(upstream, sign)?
                        }
                        UnaryOp::Reciprocal => {
                            let square = self.mul(node, node)?;
                            let quotient = self.div(upstream, square)?;
                            self.neg(quotient)?
                        }
                        UnaryOp::Square => {
                            let two = self.constant(TensorData::scalar(2.0f32));
                            let scale = self.mul(two, input)?;
                            self.mul(upstream, scale)?
                        }
                        UnaryOp::Sqrt => {
                            let two = self.constant(TensorData::scalar(2.0f32));
                            let denominator = self.mul(node, two)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Rsqrt => {
                            let two = self.constant(TensorData::scalar(2.0f32));
                            let square = self.mul(node, node)?;
                            let cube = self.mul(square, node)?;
                            let scaled = self.div(cube, two)?;
                            let local = self.mul(upstream, scaled)?;
                            self.neg(local)?
                        }
                        UnaryOp::Exp2 => {
                            let ln2 = self.constant(TensorData::scalar(std::f32::consts::LN_2));
                            let scale = self.mul(node, ln2)?;
                            self.mul(upstream, scale)?
                        }
                        UnaryOp::Log2 => {
                            let ln2 = self.constant(TensorData::scalar(std::f32::consts::LN_2));
                            let denominator = self.mul(input, ln2)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Sin => {
                            let cosine = self.cos(input)?;
                            self.mul(upstream, cosine)?
                        }
                        UnaryOp::Cos => {
                            let sine = self.sin(input)?;
                            let local = self.mul(upstream, sine)?;
                            self.neg(local)?
                        }
                        UnaryOp::Tan => {
                            let cosine = self.cos(input)?;
                            let square = self.mul(cosine, cosine)?;
                            self.div(upstream, square)?
                        }
                        UnaryOp::Sinh => {
                            let local = self.cosh(input)?;
                            self.mul(upstream, local)?
                        }
                        UnaryOp::Cosh => {
                            let local = self.sinh(input)?;
                            self.mul(upstream, local)?
                        }
                        UnaryOp::Tanh => {
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let square = self.mul(node, node)?;
                            let local = self.sub(one, square)?;
                            self.mul(upstream, local)?
                        }
                        UnaryOp::Erf | UnaryOp::Erfc => {
                            let two_over_sqrt_pi = self
                                .constant(TensorData::scalar(2.0f32 / std::f32::consts::PI.sqrt()));
                            let square = self.square(input)?;
                            let neg_square = self.neg(square)?;
                            let exponential = self.exp(neg_square)?;
                            let local = self.mul(two_over_sqrt_pi, exponential)?;
                            let local = if op == UnaryOp::Erfc {
                                self.neg(local)?
                            } else {
                                local
                            };
                            self.mul(upstream, local)?
                        }
                        UnaryOp::Asin => {
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let square = self.square(input)?;
                            let difference = self.sub(one, square)?;
                            let denominator = self.sqrt(difference)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Acos => {
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let square = self.square(input)?;
                            let difference = self.sub(one, square)?;
                            let denominator = self.sqrt(difference)?;
                            let quotient = self.div(upstream, denominator)?;
                            self.neg(quotient)?
                        }
                        UnaryOp::Atan => {
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let square = self.square(input)?;
                            let denominator = self.add(one, square)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Asinh => {
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let square = self.square(input)?;
                            let sum = self.add(square, one)?;
                            let denominator = self.sqrt(sum)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Acosh => {
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let square = self.square(input)?;
                            let difference = self.sub(square, one)?;
                            let denominator = self.sqrt(difference)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Atanh => {
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let square = self.square(input)?;
                            let denominator = self.sub(one, square)?;
                            self.div(upstream, denominator)?
                        }
                        // These primitives are deliberately discontinuous (or
                        // predicates), so reverse mode returns an explicit zero.
                        UnaryOp::Floor
                        | UnaryOp::Ceil
                        | UnaryOp::Trunc
                        | UnaryOp::Round
                        | UnaryOp::Sign
                        | UnaryOp::IsNan
                        | UnaryOp::IsInf
                        | UnaryOp::IsFinite => {
                            let zeros = filled(self.node(input)?.shape.clone(), 0.0)?;
                            self.constant(zeros)
                        }
                    };
                    self.accumulate(&mut grads, input, local)?;
                }
                Op::Binary { op, lhs, rhs } => {
                    let (lhs_grad, rhs_grad) = match op {
                        BinaryOp::Add => (upstream, upstream),
                        BinaryOp::Sub => (upstream, self.neg(upstream)?),
                        BinaryOp::Mul => (self.mul(upstream, rhs)?, self.mul(upstream, lhs)?),
                        BinaryOp::Div => {
                            let lhs_grad = self.div(upstream, rhs)?;
                            let rhs_sq = self.mul(rhs, rhs)?;
                            let numerator = self.mul(upstream, lhs)?;
                            let quotient = self.div(numerator, rhs_sq)?;
                            let rhs_grad = self.neg(quotient)?;
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Pow => {
                            let zero = self.constant(TensorData::scalar(0.0f32));
                            let one = self.constant(TensorData::scalar(1.0f32));
                            let exponent_is_zero = self.eq(rhs, zero)?;
                            let exponent_minus_one = self.sub(rhs, one)?;
                            let power = self.pow(lhs, exponent_minus_one)?;
                            let base_local = self.mul(rhs, power)?;
                            let base_local = self.select(exponent_is_zero, rhs, base_local)?;
                            let upstream_is_zero = self.eq(upstream, zero)?;
                            let base_local = self.select(upstream_is_zero, zero, base_local)?;
                            let lhs_grad = self.mul(upstream, base_local)?;
                            let base_is_zero = self.eq(lhs, zero)?;
                            let exponent_negative = self.lt(rhs, zero)?;
                            let negative_inf = self.constant(TensorData::scalar(f32::NEG_INFINITY));
                            let exponent_zero = self.constant(TensorData::scalar(0.0f32));
                            let zero_local =
                                self.select(exponent_negative, negative_inf, exponent_zero)?;
                            let logarithm = self.log2(lhs)?;
                            let ln2 = self.constant(TensorData::scalar(std::f32::consts::LN_2));
                            let exponent_local = self.mul(node, logarithm)?;
                            let exponent_local = self.mul(exponent_local, ln2)?;
                            let exponent_local =
                                self.select(base_is_zero, zero_local, exponent_local)?;
                            let rhs_grad = self.mul(upstream, exponent_local)?;
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Atan2 => {
                            let lhs_square = self.square(lhs)?;
                            let rhs_square = self.square(rhs)?;
                            let denominator = self.add(lhs_square, rhs_square)?;
                            let lhs_numerator = self.mul(upstream, rhs)?;
                            let lhs_grad = self.div(lhs_numerator, denominator)?;
                            let rhs_numerator = self.mul(upstream, lhs)?;
                            let rhs_quotient = self.div(rhs_numerator, denominator)?;
                            let rhs_grad = self.neg(rhs_quotient)?;
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Copysign => {
                            // This is tinygrad's comparison/reciprocal sign
                            // selection: differentiable in the magnitude
                            // argument away from zero, and zero in sign.
                            let lhs_sign = self.sign(lhs)?;
                            let local = self.copysign(lhs_sign, rhs)?;
                            let lhs_grad = self.mul(upstream, local)?;
                            let rhs_grad =
                                self.constant(filled(self.node(rhs)?.shape.clone(), 0.0)?);
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Maximum | BinaryOp::Minimum => {
                            let lt = if op == BinaryOp::Maximum {
                                self.lt(lhs, rhs)?
                            } else {
                                self.gt(lhs, rhs)?
                            };
                            let gt = if op == BinaryOp::Maximum {
                                self.gt(lhs, rhs)?
                            } else {
                                self.lt(lhs, rhs)?
                            };
                            let equal = self.eq(lhs, rhs)?;
                            let zero = self.constant(TensorData::scalar(0.0f32));
                            let half = self.constant(TensorData::scalar(0.5f32));
                            let half_upstream = self.mul(upstream, half)?;
                            let lhs_tie = self.select(equal, half_upstream, zero)?;
                            let rhs_tie = self.select(equal, half_upstream, zero)?;
                            let lhs_grad = self.select(gt, upstream, lhs_tie)?;
                            let rhs_grad = self.select(lt, upstream, rhs_tie)?;
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::FloorDiv
                        | BinaryOp::TruncDiv
                        | BinaryOp::Mod
                        | BinaryOp::FMod
                        | BinaryOp::BitAnd
                        | BinaryOp::BitOr
                        | BinaryOp::BitXor
                        | BinaryOp::Shl
                        | BinaryOp::Shr => {
                            let zeros_l =
                                self.constant(filled(self.node(lhs)?.shape.clone(), 0.0)?);
                            let zeros_r =
                                self.constant(filled(self.node(rhs)?.shape.clone(), 0.0)?);
                            (zeros_l, zeros_r)
                        }
                    };
                    let lhs_shape = self.node(lhs)?.shape.clone();
                    let rhs_shape = self.node(rhs)?.shape.clone();
                    let lhs_grad = self.unbroadcast(lhs_grad, lhs_shape)?;
                    let rhs_grad = self.unbroadcast(rhs_grad, rhs_shape)?;
                    self.accumulate(&mut grads, lhs, lhs_grad)?;
                    self.accumulate(&mut grads, rhs, rhs_grad)?;
                }
                Op::Reduce {
                    input,
                    kind: crate::ReduceKind::Sum,
                    axes,
                    keepdim,
                } => {
                    let input_shape = self.node(input)?.shape.clone();
                    let mut kept_dims = self.node(upstream)?.shape.dims().to_vec();
                    if !keepdim {
                        for axis in axes {
                            kept_dims.insert(axis, 1);
                        }
                    }
                    let expanded = self.reshape(upstream, Shape::new(kept_dims))?;
                    let grad = self.expand(expanded, input_shape)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Reduce {
                    input,
                    kind: crate::ReduceKind::Mean,
                    axes,
                    keepdim,
                } => {
                    let shape = self.node(input)?.shape.clone();
                    let count = axes.iter().try_fold(1usize, |n, a| {
                        n.checked_mul(shape.dims()[*a])
                            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                    })?;
                    let mut dims = self.node(upstream)?.shape.dims().to_vec();
                    if !keepdim {
                        for axis in axes {
                            dims.insert(axis, 1);
                        }
                    }
                    let reshaped = self.reshape(upstream, Shape::new(dims))?;
                    let up = self.expand(reshaped, shape)?;
                    let divisor = self.constant(TensorData::scalar(count as f32));
                    let grad = self.div(up, divisor)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Reduce {
                    input,
                    kind:
                        kind @ (crate::ReduceKind::Product
                        | crate::ReduceKind::Max
                        | crate::ReduceKind::Min),
                    axes,
                    keepdim,
                } => {
                    let grad = self.reduce_grad(input, upstream, kind, axes, keepdim)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Reduce {
                    kind: crate::ReduceKind::Any | crate::ReduceKind::All,
                    ..
                } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "boolean reductions are non-differentiable",
                    ));
                }
                Op::PrefixScan {
                    input,
                    axis,
                    kind: crate::PrefixScanKind::Sum,
                } => {
                    let gradient = self.cumsum_vjp(upstream, axis)?;
                    self.accumulate(&mut grads, input, gradient)?;
                }
                Op::PrefixScan {
                    kind: crate::PrefixScanKind::Product,
                    ..
                } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "cumprod gradient is not yet represented",
                    ));
                }
                Op::ArgReduce { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "reduction gradient not yet represented",
                    ));
                }
                Op::ReduceGrad {
                    input,
                    upstream: first_upstream,
                    kind,
                    axes,
                    keepdim,
                } => {
                    let upstream_grad = self.reduce_grad_vjp(
                        upstream,
                        input,
                        first_upstream,
                        kind,
                        axes.clone(),
                        keepdim,
                        1,
                    )?;
                    self.accumulate(&mut grads, first_upstream, upstream_grad)?;
                    if self.node(input)?.dtype.is_float() {
                        let input_grad = self.reduce_grad_vjp(
                            upstream,
                            input,
                            first_upstream,
                            kind,
                            axes,
                            keepdim,
                            0,
                        )?;
                        self.accumulate(&mut grads, input, input_grad)?;
                    }
                }
                Op::SumTo { input, .. } => {
                    let input_shape = self.node(input)?.shape.clone();
                    let grad = self.expand(upstream, input_shape)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Reshape { input, .. } => {
                    let input_shape = self.node(input)?.shape.clone();
                    let grad = self.reshape(upstream, input_shape)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Permute { input, axes } => {
                    let mut inverse = vec![0; axes.len()];
                    for (output_axis, input_axis) in axes.iter().enumerate() {
                        inverse[*input_axis] = output_axis;
                    }
                    let grad = self.permute(upstream, inverse)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Expand { input, .. } => {
                    let input_shape = self.node(input)?.shape.clone();
                    let grad = self.sum_to(upstream, input_shape)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Gather { input, index, axis } => {
                    let shape = self.node(input)?.shape.clone();
                    let zeros = self.constant(filled(shape, 0.0)?);
                    let grad = self.scatter_add(zeros, index, upstream, axis)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::StaticIndex { input, plan } => {
                    let shape = self.node(input)?.shape.clone();
                    let grad = self.static_index_grad(upstream, shape, plan)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::StaticIndexGrad {
                    cotangent, plan, ..
                } => {
                    let grad = self.static_index_plan(upstream, plan)?;
                    self.accumulate(&mut grads, cotangent, grad)?;
                }
                Op::StaticIndexUpdate { base, value, plan } => {
                    if self.node(node)?.dtype != crate::DType::F32 {
                        return Err(Error::NonDifferentiableIndexing(
                            "static index update gradients require F32",
                        ));
                    }
                    let base_shape = self.node(base)?.shape.clone();
                    let value_shape = self.node(value)?.shape.clone();
                    let base_grad = self.static_index_update_grad(
                        upstream,
                        base_shape,
                        value_shape.clone(),
                        plan.clone(),
                        crate::StaticIndexUpdateWrt::Base,
                    )?;
                    let value_grad = self.static_index_update_grad(
                        upstream,
                        self.node(base)?.shape.clone(),
                        value_shape,
                        plan,
                        crate::StaticIndexUpdateWrt::Value,
                    )?;
                    self.accumulate(&mut grads, base, base_grad)?;
                    self.accumulate(&mut grads, value, value_grad)?;
                }
                Op::StaticIndexUpdateGrad {
                    cotangent,
                    base_shape,
                    plan,
                    wrt,
                    ..
                } => {
                    let zero_base = self.constant(filled(base_shape, 0.0)?);
                    let grad = match wrt {
                        crate::StaticIndexUpdateWrt::Base => {
                            let zero_value =
                                self.constant(filled(plan.output_shape().clone(), 0.0)?);
                            self.static_index_update_plan(upstream, zero_value, plan)?
                        }
                        crate::StaticIndexUpdateWrt::Value => {
                            let expanded = self.expand(upstream, plan.output_shape().clone())?;
                            self.static_index_update_plan(zero_base, expanded, plan)?
                        }
                    };
                    self.accumulate(&mut grads, cotangent, grad)?;
                }
                Op::Scatter {
                    base,
                    index,
                    updates,
                    axis,
                    add,
                } => {
                    if !add {
                        return Err(Error::NonDifferentiableIndexing("replacement scatter"));
                    }
                    self.accumulate(&mut grads, base, upstream)?;
                    let gathered = self.gather(upstream, index, axis)?;
                    let update_shape = self.node(updates)?.shape.clone();
                    let grad = if self.node(gathered)?.shape == update_shape {
                        gathered
                    } else {
                        let starts = vec![0; update_shape.rank()];
                        self.scatter_positions(
                            gathered,
                            update_shape,
                            starts,
                            vec![1; self.node(gathered)?.shape.rank()],
                        )?
                    };
                    self.accumulate(&mut grads, updates, grad)?;
                }
                Op::MaskedSelect {
                    input, mask, size, ..
                } => {
                    let grad = self.masked_select_vjp(upstream, input, mask, size)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Shrink { input, bounds } => {
                    let shape = self.node(input)?.shape.clone();
                    let starts = bounds
                        .iter()
                        .map(|(start, _)| isize::try_from(*start).map_err(|_| Error::InvalidIndex))
                        .collect::<Result<Vec<_>>>()?;
                    let grad =
                        self.scatter_positions(upstream, shape, starts, vec![1; bounds.len()])?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Pad { input, padding, .. } => {
                    let bounds = padding
                        .iter()
                        .zip(self.node(input)?.shape.dims())
                        .map(|((before, _), dim)| (*before, before + dim))
                        .collect::<Vec<_>>();
                    let grad = self.shrink(upstream, bounds)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Stride { input, slices } => {
                    let shape = self.node(input)?.shape.clone();
                    let normalized = slices
                        .iter()
                        .zip(shape.dims())
                        .enumerate()
                        .map(|(axis, (slice, dim))| crate::ir::normalized_slice(*dim, *slice, axis))
                        .collect::<Result<Vec<_>>>()?;
                    let starts = normalized.iter().map(|(start, _, _, _)| *start).collect();
                    let steps = normalized.iter().map(|(_, _, step, _)| *step).collect();
                    let grad = self.scatter_positions(upstream, shape, starts, steps)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Concat { inputs, axis } => {
                    let mut start = 0usize;
                    for input in inputs {
                        let shape = self.node(input)?.shape.clone();
                        let mut bounds =
                            shape.dims().iter().map(|dim| (0, *dim)).collect::<Vec<_>>();
                        bounds[axis] = (
                            start,
                            start
                                .checked_add(shape.dims()[axis])
                                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
                        );
                        let grad = self.shrink(upstream, bounds)?;
                        self.accumulate(&mut grads, input, grad)?;
                        start = start
                            .checked_add(shape.dims()[axis])
                            .ok_or(Error::ShapeOverflow(shape))?;
                    }
                }
                Op::ScatterPositions {
                    input,
                    starts,
                    steps,
                    ..
                } => {
                    let input_shape = self.node(input)?.shape.clone();
                    let grad = self.scatter_positions_vjp(upstream, input_shape, starts, steps)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Matmul { lhs, rhs } => {
                    let lhs_grad = self.matmul_grad(upstream, lhs, rhs, true)?;
                    let rhs_grad = self.matmul_grad(upstream, lhs, rhs, false)?;
                    self.accumulate(&mut grads, lhs, lhs_grad)?;
                    self.accumulate(&mut grads, rhs, rhs_grad)?;
                }
                Op::Einsum { inputs, plan } => {
                    for (target, input) in inputs.iter().enumerate() {
                        if self.node(*input)?.dtype.is_float() {
                            let gradient =
                                self.einsum_grad(upstream, &inputs, plan.clone(), target)?;
                            self.accumulate(&mut grads, *input, gradient)?;
                        }
                    }
                }
                Op::EinsumGrad {
                    upstream: first_upstream,
                    inputs,
                    plan,
                    target,
                } => {
                    let upstream_grad = self.einsum_grad_vjp(
                        upstream,
                        first_upstream,
                        &inputs,
                        plan.clone(),
                        target,
                        inputs.len(),
                    )?;
                    self.accumulate(&mut grads, first_upstream, upstream_grad)?;
                    for (wrt, input) in inputs.iter().copied().enumerate() {
                        if wrt != target && self.node(input)?.dtype.is_float() {
                            let local = self.einsum_grad_vjp(
                                upstream,
                                first_upstream,
                                &inputs,
                                plan.clone(),
                                target,
                                wrt,
                            )?;
                            self.accumulate(&mut grads, input, local)?;
                        }
                    }
                }
                Op::MatmulGrad {
                    upstream: first_upstream,
                    lhs,
                    rhs,
                    lhs_gradient,
                } => {
                    let upstream_grad =
                        self.matmul_grad_vjp(upstream, first_upstream, lhs, rhs, lhs_gradient, 0)?;
                    self.accumulate(&mut grads, first_upstream, upstream_grad)?;
                    let factor = if lhs_gradient { rhs } else { lhs };
                    if self.node(factor)?.dtype.is_float() {
                        let factor_grad = self.matmul_grad_vjp(
                            upstream,
                            first_upstream,
                            lhs,
                            rhs,
                            lhs_gradient,
                            if lhs_gradient { 2 } else { 1 },
                        )?;
                        self.accumulate(&mut grads, factor, factor_grad)?;
                    }
                }
                Op::ScatterPositionsVjp {
                    cotangent,
                    starts,
                    steps,
                    ..
                } => {
                    let shape = self.node(cotangent)?.shape.clone();
                    let grad = self.scatter_positions(upstream, shape, starts, steps)?;
                    self.accumulate(&mut grads, cotangent, grad)?;
                }
                Op::Conv2dGradVjp { .. }
                | Op::ConvTranspose2dGradVjp { .. }
                | Op::ReduceGradVjp { .. }
                | Op::EinsumGradVjp { .. }
                | Op::MatmulGradVjp { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "third-order indexed contraction gradient",
                    ));
                }
                Op::Conv2d {
                    input,
                    weight,
                    bias,
                    options,
                } => {
                    let input_grad = self.conv2d_grad(upstream, input, weight, bias, options, 0)?;
                    let weight_grad =
                        self.conv2d_grad(upstream, input, weight, bias, options, 1)?;
                    self.accumulate(&mut grads, input, input_grad)?;
                    self.accumulate(&mut grads, weight, weight_grad)?;
                    if let Some(bias) = bias {
                        let bias_grad =
                            self.conv2d_grad(upstream, input, weight, Some(bias), options, 2)?;
                        self.accumulate(&mut grads, bias, bias_grad)?;
                    }
                }
                Op::Conv2dGrad {
                    upstream: first_upstream,
                    input,
                    weight,
                    bias,
                    options,
                    target,
                } => {
                    for wrt in 0..=3 {
                        let node = match wrt {
                            0 => first_upstream,
                            1 => input,
                            2 => weight,
                            3 => match bias {
                                Some(node) => node,
                                None => continue,
                            },
                            _ => unreachable!(),
                        };
                        if self.node(node)?.dtype.is_float()
                            && (wrt == 0 || wrt != target as usize + 1)
                        {
                            let local = self.conv2d_grad_vjp(
                                upstream,
                                first_upstream,
                                input,
                                weight,
                                bias,
                                options,
                                target,
                                wrt as u8,
                            )?;
                            self.accumulate(&mut grads, node, local)?;
                        }
                    }
                }
                Op::ConvTranspose2d {
                    input,
                    weight,
                    bias,
                    options,
                } => {
                    let input_grad =
                        self.conv_transpose2d_grad(upstream, input, weight, bias, options, 0)?;
                    let weight_grad =
                        self.conv_transpose2d_grad(upstream, input, weight, bias, options, 1)?;
                    self.accumulate(&mut grads, input, input_grad)?;
                    self.accumulate(&mut grads, weight, weight_grad)?;
                    if let Some(b) = bias {
                        let bias_grad = self.conv_transpose2d_grad(
                            upstream,
                            input,
                            weight,
                            Some(b),
                            options,
                            2,
                        )?;
                        self.accumulate(&mut grads, b, bias_grad)?;
                    }
                }
                Op::ConvTranspose2dGrad {
                    upstream: first_upstream,
                    input,
                    weight,
                    bias,
                    options,
                    target,
                } => {
                    for wrt in 0..=3 {
                        let node = match wrt {
                            0 => first_upstream,
                            1 => input,
                            2 => weight,
                            3 => match bias {
                                Some(node) => node,
                                None => continue,
                            },
                            _ => unreachable!(),
                        };
                        if self.node(node)?.dtype.is_float()
                            && (wrt == 0 || wrt != target as usize + 1)
                        {
                            let local = self.conv_transpose2d_grad_vjp(
                                upstream,
                                first_upstream,
                                input,
                                weight,
                                bias,
                                options,
                                target,
                                wrt as u8,
                            )?;
                            self.accumulate(&mut grads, node, local)?;
                        }
                    }
                }
                Op::Select {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let zeros = filled(self.node(upstream)?.shape.clone(), 0.0)?;
                    let zeros = self.constant(zeros);
                    let true_grad = self.select(condition, upstream, zeros)?;
                    let false_grad = self.select(condition, zeros, upstream)?;
                    let true_grad =
                        self.unbroadcast(true_grad, self.node(on_true)?.shape.clone())?;
                    let false_grad =
                        self.unbroadcast(false_grad, self.node(on_false)?.shape.clone())?;
                    self.accumulate(&mut grads, on_true, true_grad)?;
                    self.accumulate(&mut grads, on_false, false_grad)?;
                }
            }
        }
        grads[wrt.index()].ok_or(Error::NoGradient(wrt))
    }

    fn unbroadcast(&mut self, gradient: NodeId, shape: Shape) -> Result<NodeId> {
        if self.node(gradient)?.shape == shape {
            Ok(gradient)
        } else {
            self.sum_to(gradient, shape)
        }
    }

    /// Rejects unsupported scan reverse slices before `grad_with` creates an
    /// implicit seed or any derivative nodes. Sum scans admit a compositional
    /// adjoint only for floating values; product scans need their own
    /// zero-aware rule and deliberately remain outside this slice.
    fn validate_prefix_scan_reverse(&self, loss: NodeId) -> Result<()> {
        let mut pending = vec![loss];
        let mut visited = BTreeSet::new();
        while let Some(node) = pending.pop() {
            if !visited.insert(node) {
                continue;
            }
            let current = self.node(node)?;
            if let Op::PrefixScan { input, kind, .. } = &current.op {
                match kind {
                    crate::PrefixScanKind::Sum if !self.node(*input)?.dtype.is_float() => {
                        return Err(Error::NonDifferentiableIndexing(
                            "cumsum gradients require floating input",
                        ));
                    }
                    crate::PrefixScanKind::Product => {
                        return Err(Error::NonDifferentiableIndexing(
                            "cumprod gradient is not yet represented",
                        ));
                    }
                    crate::PrefixScanKind::Sum => {}
                }
            }
            pending.extend(current.op.backward_inputs());
        }
        Ok(())
    }

    /// The adjoint of an inclusive sum scan is the inclusive sum scan in the
    /// opposite direction. `axis` was normalized when the PrefixScan node was
    /// built, so the two signed strides preserve scalar and empty-domain
    /// behavior without reinterpreting user-facing axes.
    fn cumsum_vjp(&mut self, upstream: NodeId, axis: usize) -> Result<NodeId> {
        let reversed = self.reverse_axis(upstream, axis)?;
        let scanned = self.cumsum(reversed, axis as isize)?;
        self.reverse_axis(scanned, axis)
    }

    fn reverse_axis(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let slices = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(current, _)| crate::Slice {
                start: None,
                stop: None,
                step: if current == axis { -1 } else { 1 },
            })
            .collect::<Vec<_>>();
        self.stride(input, slices)
    }

    /// Routes the fixed-size selected-output cotangent back through the
    /// canonical row-major mask map. Prefix counts are control/index values:
    /// masks intentionally retain no gradient edge, while the gathered value
    /// path remains fully compositional for higher-order differentiation.
    fn masked_select_vjp(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        mask: NodeId,
        size: usize,
    ) -> Result<NodeId> {
        let input_shape = self.node(input)?.shape.clone();
        if size == 0 {
            return Ok(self.constant(filled(input_shape, 0.0)?));
        }
        let size = i64::try_from(size).map_err(|_| Error::ShapeOverflow(input_shape.clone()))?;
        let limit = size
            .checked_add(1)
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        let flat_shape = Shape::from([input_shape.numel()?]);
        let expanded_mask = self.expand(mask, input_shape.clone())?;
        let flat_mask = self.reshape(expanded_mask, flat_shape.clone())?;
        let counts = self.cumsum(flat_mask, 0)?;
        let one = self.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
        let rank = self.sub(counts, one)?;
        let limit = self.constant(TensorData::scalar_with_dtype(Scalar::I(limit), DType::I64));
        let retained = self.lt(counts, limit)?;
        let retained = self.logical_and(flat_mask, retained)?;
        let zero_index = self.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::I32));
        let safe_index = self.select(retained, rank, zero_index)?;
        let gathered = self.gather(upstream, safe_index, 0)?;
        let zero = self.constant(filled(flat_shape.clone(), 0.0)?);
        let routed = self.select(retained, gathered, zero)?;
        self.reshape(routed, input_shape)
    }

    fn accumulate(
        &mut self,
        grads: &mut [Option<NodeId>],
        node: NodeId,
        gradient: NodeId,
    ) -> Result<()> {
        grads[node.index()] = Some(match grads[node.index()] {
            Some(previous) => self.add(previous, gradient)?,
            None => gradient,
        });
        Ok(())
    }
}

fn filled(shape: Shape, value: f32) -> Result<TensorData> {
    TensorData::new(shape.clone(), vec![value; shape.numel()?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend};
    use std::collections::HashMap;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn differentiates_broadcast_matmul_and_sum() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2]);
        let weight = graph.input("weight", [2, 2]);
        let bias = graph.input("bias", [2]);
        let product = graph.matmul(x, weight).unwrap();
        let shifted = graph.add(product, bias).unwrap();
        let rows = graph.sum(shifted, 1).unwrap();
        let loss = graph.sum(rows, 0).unwrap();
        let dx = graph.grad(loss, x).unwrap();
        let dw = graph.grad(loss, weight).unwrap();
        let db = graph.grad(loss, bias).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2, 2], &[1., 2., 3., 4.])),
            ("weight".into(), data([2, 2], &[5., 6., 7., 8.])),
            ("bias".into(), data([2], &[0.5, -0.5])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, dx, &inputs).unwrap(),
            data([2, 2], &[11., 15., 11., 15.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, dw, &inputs).unwrap(),
            data([2, 2], &[4., 4., 6., 6.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, db, &inputs).unwrap(),
            data([2], &[2., 2.])
        );
    }

    #[test]
    fn differentiates_vector_matrix_matmul() {
        let mut graph = Graph::new();
        let vector = graph.input("vector", [3]);
        let matrix = graph.input("matrix", [3, 2]);
        let product = graph.matmul(vector, matrix).unwrap();
        let loss = graph.sum(product, 0).unwrap();
        let vector_grad = graph.grad(loss, vector).unwrap();
        let matrix_grad = graph.grad(loss, matrix).unwrap();
        assert!(
            graph
                .trace(vector_grad)
                .unwrap()
                .to_string()
                .contains("matmul_lhs_grad")
        );
        let inputs = HashMap::from([
            (
                "vector".into(),
                TensorData::new([3], vec![1., 2., 3.]).unwrap(),
            ),
            (
                "matrix".into(),
                TensorData::new([3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, vector_grad, &inputs).unwrap(),
            TensorData::new([3], vec![3., 7., 11.]).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, matrix_grad, &inputs).unwrap(),
            TensorData::new([3, 2], vec![1., 1., 2., 2., 3., 3.]).unwrap()
        );
    }

    #[test]
    fn differentiates_padded_grouped_conv2d_and_bias() {
        let mut graph = Graph::new();
        let x = graph.input("x", [1, 2, 2, 2]);
        let w = graph.input("w", [2, 1, 1, 1]);
        let b = graph.input("b", [2]);
        let y = graph
            .conv2d(
                x,
                w,
                Some(b),
                crate::Conv2dOptions {
                    groups: 2,
                    padding: [1, 0, 0, 1],
                    ..Default::default()
                },
            )
            .unwrap();
        let s3 = graph.sum(y, 3).unwrap();
        let s2 = graph.sum(s3, 2).unwrap();
        let s1 = graph.sum(s2, 1).unwrap();
        let loss = graph.sum(s1, 0).unwrap();
        let gx = graph.grad(loss, x).unwrap();
        let gw = graph.grad(loss, w).unwrap();
        let gb = graph.grad(loss, b).unwrap();
        assert!(graph.trace(gx).unwrap().to_string().contains("conv2d_grad"));
        let inputs = HashMap::from([
            (
                "x".into(),
                TensorData::new([1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
            ),
            (
                "w".into(),
                TensorData::new([2, 1, 1, 1], vec![2., 3.]).unwrap(),
            ),
            ("b".into(), TensorData::new([2], vec![0., 0.]).unwrap()),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gx, &inputs).unwrap(),
            TensorData::new([1, 2, 2, 2], vec![2., 2., 2., 2., 3., 3., 3., 3.]).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, gw, &inputs).unwrap(),
            TensorData::new([2, 1, 1, 1], vec![10., 26.]).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, gb, &inputs).unwrap(),
            TensorData::new([2], vec![9., 9.]).unwrap()
        );
    }

    #[test]
    fn analytic_gradient_matches_central_difference() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let exponential = graph.exp(x).unwrap();
        let product = graph.mul(exponential, x).unwrap();
        let loss = graph.sum(product, 0).unwrap();
        let dx = graph.grad(loss, x).unwrap();
        let point = [0.2_f32, -0.4, 1.1];
        let inputs = HashMap::from([("x".into(), data([3], &point))]);
        let analytic = CpuBackend.execute(&graph, dx, &inputs).unwrap();
        let epsilon = 1e-3;

        for index in 0..point.len() {
            let mut plus = point;
            let mut minus = point;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_inputs = HashMap::from([("x".into(), data([3], &plus))]);
            let minus_inputs = HashMap::from([("x".into(), data([3], &minus))]);
            let plus_loss = CpuBackend
                .execute(&graph, loss, &plus_inputs)
                .unwrap()
                .values()[0];
            let minus_loss = CpuBackend
                .execute(&graph, loss, &minus_inputs)
                .unwrap()
                .values()[0];
            let numerical = (plus_loss - minus_loss) / (2.0 * epsilon);
            assert!((analytic.values()[index] - numerical).abs() < 2e-3);
        }
    }

    #[test]
    fn extended_float_primitives_match_central_difference() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let reciprocal = graph.reciprocal(x).unwrap();
        let square = graph.square(x).unwrap();
        let root = graph.sqrt(x).unwrap();
        let exp2 = graph.exp2(x).unwrap();
        let log2 = graph.log2(x).unwrap();
        let sin = graph.sin(x).unwrap();
        let cos = graph.cos(x).unwrap();
        let tan = graph.tan(x).unwrap();
        let sinh = graph.sinh(x).unwrap();
        let cosh = graph.cosh(x).unwrap();
        let tanh = graph.tanh(x).unwrap();
        let mut total = reciprocal;
        for value in [square, root, exp2, log2, sin, cos, tan, sinh, cosh, tanh] {
            total = graph.add(total, value).unwrap();
        }
        let loss = graph.sum_all(total).unwrap();
        let gradient = graph.grad(loss, x).unwrap();
        let point = [0.7_f32, 1.2];
        let inputs = HashMap::from([("x".into(), data([2], &point))]);
        let analytic = CpuBackend.execute(&graph, gradient, &inputs).unwrap();
        // F32 output rounds these finite-difference probes at this magnitude.
        // A centesimal step keeps the independent check stable without
        // weakening the exact analytic assertions above.
        let epsilon = 1e-2;
        for index in 0..point.len() {
            let mut plus = point;
            let mut minus = point;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_loss = CpuBackend
                .execute(
                    &graph,
                    loss,
                    &HashMap::from([("x".into(), data([2], &plus))]),
                )
                .unwrap()
                .values()[0];
            let minus_loss = CpuBackend
                .execute(
                    &graph,
                    loss,
                    &HashMap::from([("x".into(), data([2], &minus))]),
                )
                .unwrap()
                .values()[0];
            let numerical = (plus_loss - minus_loss) / (2.0 * epsilon);
            assert!(
                (analytic.values()[index] - numerical).abs() < 2.0,
                "index {index}: {} vs {numerical}",
                analytic.values()[index]
            );
        }
    }

    #[test]
    fn pow_and_extrema_gradients_handle_broadcasts_and_ties() {
        let mut graph = Graph::new();
        let base = graph.input("base", [2, 1]);
        let exponent = graph.input("exponent", [2]);
        let power = graph.pow(base, exponent).unwrap();
        let maximum = graph.maximum(base, exponent).unwrap();
        let minimum = graph.minimum(base, exponent).unwrap();
        let sum = graph.add(power, maximum).unwrap();
        let sum = graph.add(sum, minimum).unwrap();
        let loss = graph.sum_all(sum).unwrap();
        let base_grad = graph.grad(loss, base).unwrap();
        let exponent_grad = graph.grad(loss, exponent).unwrap();
        let inputs = HashMap::from([
            ("base".into(), data([2, 1], &[2.0, 3.0])),
            ("exponent".into(), data([2], &[2.0, 3.0])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, base_grad, &inputs).unwrap(),
            data([2, 1], &[18.0, 35.0])
        );
        let exponent_values = CpuBackend
            .execute(&graph, exponent_grad, &inputs)
            .unwrap()
            .values()
            .to_vec();
        assert!(
            (exponent_values[0] - (4.0 * 2.0_f32.ln() + 9.0 * 3.0_f32.ln() + 2.0)).abs() < 1e-5
        );
        assert!(
            (exponent_values[1] - (8.0 * 2.0_f32.ln() + 27.0 * 3.0_f32.ln() + 2.0)).abs() < 1e-5
        );

        let mut ties = Graph::new();
        let lhs = ties.input("lhs", [2]);
        let rhs = ties.input("rhs", [2]);
        let maximum = ties.maximum(lhs, rhs).unwrap();
        let loss = ties.sum_all(maximum).unwrap();
        let dl = ties.grad(loss, lhs).unwrap();
        let dr = ties.grad(loss, rhs).unwrap();
        let inputs = HashMap::from([
            ("lhs".into(), data([2], &[1.0, 2.0])),
            ("rhs".into(), data([2], &[1.0, 0.0])),
        ]);
        assert_eq!(
            CpuBackend.execute(&ties, dl, &inputs).unwrap(),
            data([2], &[0.5, 1.0])
        );
        assert_eq!(
            CpuBackend.execute(&ties, dr, &inputs).unwrap(),
            data([2], &[0.5, 0.0])
        );
    }

    #[test]
    fn masked_fill_routes_gradients_and_predicates_are_nondifferentiable() {
        let mut graph = Graph::new();
        let condition_source = graph.input("condition_source", [2, 1]);
        let threshold = graph.constant(data([], &[0.0]));
        let condition = graph.gt(condition_source, threshold).unwrap();
        let on_true = graph.input("on_true", [2, 1]);
        let on_false = graph.input("on_false", [2]);
        let selected = graph.masked_fill(on_false, condition, on_true).unwrap();
        let loss = graph.sum_all(selected).unwrap();
        let true_grad = graph.grad(loss, on_true).unwrap();
        let false_grad = graph.grad(loss, on_false).unwrap();
        assert!(matches!(
            graph.grad(loss, condition_source),
            Err(Error::NoGradient(_))
        ));
        let inputs = HashMap::from([
            ("condition_source".into(), data([2, 1], &[1.0, -1.0])),
            ("on_true".into(), data([2, 1], &[2.0, 3.0])),
            ("on_false".into(), data([2], &[4.0, 5.0])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, true_grad, &inputs).unwrap(),
            data([2, 1], &[2.0, 0.0])
        );
        assert_eq!(
            CpuBackend.execute(&graph, false_grad, &inputs).unwrap(),
            data([2], &[1.0, 1.0])
        );
    }

    #[test]
    fn movement_gradients_scatter_extract_and_partition() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let y = graph.input("y", [3, 1]);
        let shrunk = graph.shrink(x, [(0, 2), (1, 3)]).unwrap();
        let padded = graph
            .pad(shrunk, [(1, 0), (1, 0)], crate::Scalar::F(0.0))
            .unwrap();
        let reversed = graph
            .stride(
                padded,
                [
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                ],
            )
            .unwrap();
        let joined = graph.concat([reversed, y], 1).unwrap();
        let loss = graph.sum_all(joined).unwrap();
        let dx = graph.grad(loss, x).unwrap();
        let dy = graph.grad(loss, y).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("y".into(), data([3, 1], &[7., 8., 9.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, dx, &inputs).unwrap(),
            data([2, 3], &[0., 1., 1., 0., 1., 1.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, dy, &inputs).unwrap(),
            data([3, 1], &[1., 1., 1.])
        );
    }

    #[test]
    fn gather_and_scatter_add_gradients_handle_duplicate_indices() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let index = graph.input_dtype("index", [2], crate::DType::I32);
        let gathered = graph.gather(x, index, 0).unwrap();
        let gather_loss = graph.sum_all(gathered).unwrap();
        let gather_grad = graph.grad(gather_loss, x).unwrap();
        let base = graph.input("base", [3]);
        let updates = graph.input("updates", [2]);
        let scattered = graph.scatter_add(base, index, updates, 0).unwrap();
        let scatter_loss = graph.sum_all(scattered).unwrap();
        let base_grad = graph.grad(scatter_loss, base).unwrap();
        let updates_grad = graph.grad(scatter_loss, updates).unwrap();
        let replaced = graph.scatter(base, index, updates, 0).unwrap();
        let replacement_loss = graph.sum_all(replaced).unwrap();
        assert!(matches!(
            graph.grad(replacement_loss, base),
            Err(Error::NonDifferentiableIndexing(_))
        ));
        assert!(matches!(
            graph.grad(gather_loss, index),
            Err(Error::NoGradient(_))
        ));
        let inputs = HashMap::from([
            ("x".into(), data([3], &[1., 2., 3.])),
            (
                "index".into(),
                TensorData::from_scalars(
                    [2],
                    crate::DType::I32,
                    [crate::Scalar::I(1), crate::Scalar::I(1)],
                )
                .unwrap(),
            ),
            ("base".into(), data([3], &[4., 5., 6.])),
            ("updates".into(), data([2], &[7., 8.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gather_grad, &inputs).unwrap(),
            data([3], &[0., 2., 0.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, base_grad, &inputs).unwrap(),
            data([3], &[1., 1., 1.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, updates_grad, &inputs).unwrap(),
            data([2], &[1., 1.])
        );
    }

    #[test]
    fn lifecycle_requires_grad_detach_no_grad_and_upstream_contracts() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let frozen = graph.input_dtype_requires_grad("frozen", [2], crate::DType::F32, false);
        let detached = graph.detach(x).unwrap();
        let stopped = graph.no_grad(|g| g.square(x).unwrap());
        assert!(graph.requires_grad(x).unwrap());
        assert!(!graph.requires_grad(frozen).unwrap());
        assert!(graph.requires_grad(detached).unwrap());
        assert!(!graph.requires_grad(stopped).unwrap());

        let detached_square = graph.mul(detached, detached).unwrap();
        let loss = graph.sum_all(detached_square).unwrap();
        let detached_grad = graph.grad(loss, detached).unwrap();
        assert!(matches!(graph.grad(loss, x), Err(Error::NoGradient(_))));
        assert!(matches!(
            graph.grad(loss, frozen),
            Err(Error::NonDifferentiableTarget(_))
        ));
        let nonscalar = graph.square(x).unwrap();
        assert!(matches!(
            graph.grad(nonscalar, x),
            Err(Error::NonScalarLoss(_))
        ));
        let bad_seed = graph.input("bad_seed", [1]);
        assert!(matches!(
            graph.grad_with(nonscalar, x, Some(bad_seed), true),
            Err(Error::GradientShape { .. })
        ));
        let seed = graph.input("seed", [2]);
        let explicit = graph.grad_with(nonscalar, x, Some(seed), true).unwrap();
        let values = HashMap::from([
            ("x".into(), data([2], &[2., 3.])),
            ("frozen".into(), data([2], &[0., 0.])),
            ("bad_seed".into(), data([1], &[1.])),
            ("seed".into(), data([2], &[4., 5.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, detached_grad, &values).unwrap(),
            data([2], &[4., 6.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, explicit, &values).unwrap(),
            data([2], &[16., 30.])
        );
    }

    #[test]
    fn compositional_second_derivatives_survive_broadcast_movement_and_select() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 1]);
        let y = graph.input("y", [1, 2]);
        let product = graph.mul(x, y).unwrap();
        let moved = graph.permute(product, [1, 0]).unwrap();
        let expanded = graph.expand(moved, [3, 2, 2]).unwrap();
        let zero = graph.constant(TensorData::scalar(0.0f32));
        let condition = graph.gt(expanded, zero).unwrap();
        let negated = graph.neg(expanded).unwrap();
        let selected = graph.select(condition, expanded, negated).unwrap();
        let exponent = graph.exp(selected).unwrap();
        let loss = graph.mean_all(exponent).unwrap();
        let first = graph.grad(loss, x).unwrap();
        let first_sum = graph.sum_all(first).unwrap();
        let second = graph.grad(first_sum, x).unwrap();
        let values = HashMap::from([
            ("x".into(), data([2, 1], &[1., 2.])),
            ("y".into(), data([1, 2], &[3., 4.])),
        ]);
        let actual = CpuBackend.execute(&graph, second, &values).unwrap();
        assert!(
            actual
                .values()
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
    }

    #[test]
    fn rank_two_matmul_quadratic_has_lazy_hessian_vector_product() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2]);
        let weight = graph.input("weight", [2, 2]);
        let product = graph.matmul(x, weight).unwrap();
        let squared = graph.square(product).unwrap();
        let loss = graph.sum_all(squared).unwrap();
        let gradient = graph.grad(loss, x).unwrap();
        let direction = graph.input("direction", [2, 2]);
        let dot = graph.mul(gradient, direction).unwrap();
        let dot_sum = graph.sum_all(dot).unwrap();
        let hvp = graph.grad(dot_sum, x).unwrap();
        let values = HashMap::from([
            ("x".into(), data([2, 2], &[1., 2., 3., 4.])),
            ("weight".into(), data([2, 2], &[1., 2., 0., 1.])),
            ("direction".into(), data([2, 2], &[1., 0., 0., 1.])),
        ]);
        // H v = 2 v W W^T for sum((xW)^2).
        assert_eq!(
            CpuBackend.execute(&graph, hvp, &values).unwrap(),
            data([2, 2], &[10., 4., 4., 2.])
        );
    }

    #[test]
    fn vector_dot_hessian_vector_product_uses_generalized_matmul_vjp() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let y = graph.input("y", [2]);
        let dot = graph.matmul(x, y).unwrap();
        let loss = graph.square(dot).unwrap();
        let gradient = graph.grad(loss, x).unwrap();
        let direction = graph.input("direction", [2]);
        let product = graph.mul(gradient, direction).unwrap();
        let product_sum = graph.sum_all(product).unwrap();
        let hvp = graph.grad(product_sum, x).unwrap();
        let values = HashMap::from([
            ("x".into(), data([2], &[3., 4.])),
            ("y".into(), data([2], &[2., 1.])),
            ("direction".into(), data([2], &[1., -1.])),
        ]);
        // 2 y (y dot v) = [4, 2].
        assert_eq!(
            CpuBackend.execute(&graph, hvp, &values).unwrap(),
            data([2], &[4., 2.])
        );
        assert!(
            graph
                .trace(hvp)
                .unwrap()
                .to_string()
                .contains("matmul_grad_vjp")
        );
    }

    #[test]
    fn repeated_label_einsum_gradient_has_second_order_trace_contract() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2]);
        let trace = graph.einsum("ii->", &[x]).unwrap();
        let loss = graph.square(trace).unwrap();
        let gradient = graph.grad(loss, x).unwrap();
        let direction = graph.input("direction", [2, 2]);
        let weighted = graph.mul(gradient, direction).unwrap();
        let weighted_sum = graph.sum_all(weighted).unwrap();
        let hvp = graph.grad(weighted_sum, x).unwrap();
        let values = HashMap::from([
            ("x".into(), data([2, 2], &[1., 7., 9., 2.])),
            ("direction".into(), data([2, 2], &[3., 11., 13., 5.])),
        ]);
        // 2 * trace(direction) on the diagonal, zero off-diagonal.
        assert_eq!(
            CpuBackend.execute(&graph, hvp, &values).unwrap(),
            data([2, 2], &[16., 0., 0., 16.])
        );
        assert!(
            graph
                .trace(hvp)
                .unwrap()
                .to_string()
                .contains("einsum_grad_vjp")
        );
    }

    #[test]
    fn product_reduce_hvps_cover_zero_count_branches() {
        for (name, x_values, expected) in [
            ("none", vec![2., 3.], vec![5., 4.]),
            ("one", vec![0., 3.], vec![0., 4.]),
            ("many", vec![0., 0.], vec![0., 0.]),
        ] {
            let mut graph = Graph::new();
            let x = graph.input("x", [2]);
            let product = graph
                .reduce(x, crate::ReduceKind::Product, None, false)
                .unwrap();
            let gradient = graph.grad(product, x).unwrap();
            let direction = graph.input("direction", [2]);
            let weighted = graph.mul(gradient, direction).unwrap();
            let weighted_sum = graph.sum_all(weighted).unwrap();
            let hvp = graph.grad(weighted_sum, x).unwrap();
            let values = HashMap::from([
                ("x".into(), data([2], &x_values)),
                ("direction".into(), data([2], &[4., 5.])),
            ]);
            assert_eq!(
                CpuBackend.execute(&graph, hvp, &values).unwrap(),
                data([2], &expected),
                "{name}"
            );
            assert!(
                graph
                    .trace(hvp)
                    .unwrap()
                    .to_string()
                    .contains("reduce_grad_vjp_Product")
            );
        }
    }

    #[test]
    fn extrema_reduce_second_derivative_keeps_tie_masks_constant() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let maximum = graph
            .reduce(x, crate::ReduceKind::Max, None, false)
            .unwrap();
        let gradient = graph.grad(maximum, x).unwrap();
        let direction = graph.input("direction", [3]);
        let weighted = graph.mul(gradient, direction).unwrap();
        let weighted_sum = graph.sum_all(weighted).unwrap();
        let hvp = graph.grad(weighted_sum, x).unwrap();
        let values = HashMap::from([
            ("x".into(), data([3], &[2., 2., 1.])),
            ("direction".into(), data([3], &[3., 5., 7.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, hvp, &values).unwrap(),
            data([3], &[0., 0., 0.])
        );
        assert!(
            graph
                .trace(hvp)
                .unwrap()
                .to_string()
                .contains("reduce_grad_vjp_Max")
        );
    }

    #[test]
    fn indexed_and_movement_gradient_vjps_preserve_linear_adjoint_maps() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let index = graph.input_dtype("index", [2], crate::DType::I32);
        let gathered = graph.gather(x, index, 0).unwrap();
        let gather_seed = graph.input("gather_seed", [2]);
        let gather_gradient = graph
            .grad_with(gathered, x, Some(gather_seed), true)
            .unwrap();
        let direction = graph.input("direction", [3]);
        let gather_dot = graph.mul(gather_gradient, direction).unwrap();
        let gather_dot_sum = graph.sum_all(gather_dot).unwrap();
        let gather_vjp = graph.grad(gather_dot_sum, gather_seed).unwrap();

        let shrunk = graph.shrink(x, [(1, 3)]).unwrap();
        let shrink_seed = graph.input("shrink_seed", [2]);
        let shrink_gradient = graph.grad_with(shrunk, x, Some(shrink_seed), true).unwrap();
        let shrink_dot = graph.mul(shrink_gradient, direction).unwrap();
        let shrink_dot_sum = graph.sum_all(shrink_dot).unwrap();
        let shrink_vjp = graph.grad(shrink_dot_sum, shrink_seed).unwrap();
        let values = HashMap::from([
            ("x".into(), data([3], &[1., 2., 3.])),
            ("gather_seed".into(), data([2], &[4., 5.])),
            ("shrink_seed".into(), data([2], &[6., 7.])),
            ("direction".into(), data([3], &[10., 20., 30.])),
            (
                "index".into(),
                TensorData::from_scalars(
                    [2],
                    crate::DType::I32,
                    [crate::Scalar::I(2), crate::Scalar::I(2)],
                )
                .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gather_vjp, &values).unwrap(),
            data([2], &[30., 30.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, shrink_vjp, &values).unwrap(),
            data([2], &[20., 30.])
        );
        assert!(
            graph
                .trace(shrink_vjp)
                .unwrap()
                .to_string()
                .contains("scatter_positions_vjp")
        );
    }

    #[test]
    fn fixed_masked_select_gradients_route_only_retained_row_major_values() {
        let mut graph = Graph::new();
        let x = graph.input("x", [5]);
        let mask = graph.input_dtype("mask", [5], DType::Bool);
        let selected = graph.masked_select(x, mask, 3, Scalar::F(-1.0)).unwrap();
        let seed = graph.input("seed", [3]);
        let gradient = graph.grad_with(selected, x, Some(seed), true).unwrap();
        let direction = graph.input("direction", [5]);
        let weighted = graph.mul(gradient, direction).unwrap();
        let loss = graph.sum_all(weighted).unwrap();
        let seed_vjp = graph.grad(loss, seed).unwrap();

        let mask_values = |values| TensorData::from_scalars([5], DType::Bool, values).unwrap();
        let values = HashMap::from([
            ("x".into(), data([5], &[1., 2., 3., 4., 5.])),
            (
                "mask".into(),
                mask_values([
                    Scalar::Bool(true),
                    Scalar::Bool(false),
                    Scalar::Bool(true),
                    Scalar::Bool(true),
                    Scalar::Bool(true),
                ]),
            ),
            ("seed".into(), data([3], &[10., 20., 30.])),
            ("direction".into(), data([5], &[1., 2., 3., 4., 5.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &values).unwrap(),
            data([5], &[10., 0., 20., 30., 0.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, seed_vjp, &values).unwrap(),
            data([3], &[1., 3., 4.])
        );
        assert!(matches!(graph.grad(loss, mask), Err(Error::NoGradient(_))));

        let epsilon = 1e-3;
        let analytic = CpuBackend.execute(&graph, seed_vjp, &values).unwrap();
        for index in 0..3 {
            let mut plus = [10.0f32, 20.0, 30.0];
            let mut minus = plus;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_values = HashMap::from([
                ("x".into(), data([5], &[1., 2., 3., 4., 5.])),
                (
                    "mask".into(),
                    mask_values([
                        Scalar::Bool(true),
                        Scalar::Bool(false),
                        Scalar::Bool(true),
                        Scalar::Bool(true),
                        Scalar::Bool(true),
                    ]),
                ),
                ("seed".into(), data([3], &plus)),
                ("direction".into(), data([5], &[1., 2., 3., 4., 5.])),
            ]);
            let minus_values = HashMap::from([
                ("x".into(), data([5], &[1., 2., 3., 4., 5.])),
                (
                    "mask".into(),
                    mask_values([
                        Scalar::Bool(true),
                        Scalar::Bool(false),
                        Scalar::Bool(true),
                        Scalar::Bool(true),
                        Scalar::Bool(true),
                    ]),
                ),
                ("seed".into(), data([3], &minus)),
                ("direction".into(), data([5], &[1., 2., 3., 4., 5.])),
            ]);
            let numeric = (CpuBackend
                .execute(&graph, loss, &plus_values)
                .unwrap()
                .values()[0]
                - CpuBackend
                    .execute(&graph, loss, &minus_values)
                    .unwrap()
                    .values()[0])
                / (2.0 * epsilon);
            assert!((analytic.values()[index] - numeric).abs() < 1e-2);
        }
        assert!(
            graph
                .trace(seed_vjp)
                .unwrap()
                .to_string()
                .contains("cumsum")
        );

        let all_false = HashMap::from([
            ("x".into(), data([5], &[1., 2., 3., 4., 5.])),
            ("mask".into(), mask_values([Scalar::Bool(false); 5])),
            ("seed".into(), data([3], &[10., 20., 30.])),
            ("direction".into(), data([5], &[1., 2., 3., 4., 5.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &all_false).unwrap(),
            data([5], &[0., 0., 0., 0., 0.])
        );
        let padded = HashMap::from([
            ("x".into(), data([5], &[1., 2., 3., 4., 5.])),
            (
                "mask".into(),
                mask_values([
                    Scalar::Bool(true),
                    Scalar::Bool(false),
                    Scalar::Bool(false),
                    Scalar::Bool(false),
                    Scalar::Bool(false),
                ]),
            ),
            ("seed".into(), data([3], &[10., 20., 30.])),
            ("direction".into(), data([5], &[1., 2., 3., 4., 5.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &padded).unwrap(),
            data([5], &[10., 0., 0., 0., 0.])
        );

        let mut empty_graph = Graph::new();
        let empty_input = empty_graph.input("input", [0]);
        let empty_mask = empty_graph.input_dtype("mask", [0], DType::Bool);
        let empty_selected = empty_graph
            .masked_select(empty_input, empty_mask, 3, Scalar::F(-1.0))
            .unwrap();
        let empty_seed = empty_graph.input("seed", [3]);
        let empty_gradient = empty_graph
            .grad_with(empty_selected, empty_input, Some(empty_seed), true)
            .unwrap();
        let empty_values = HashMap::from([
            ("input".into(), data([0], &[])),
            (
                "mask".into(),
                TensorData::from_scalars([0], DType::Bool, Vec::<Scalar>::new()).unwrap(),
            ),
            ("seed".into(), data([3], &[10., 20., 30.])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&empty_graph, empty_gradient, &empty_values)
                .unwrap(),
            data([0], &[])
        );

        let mut zero_size_graph = Graph::new();
        let zero_input = zero_size_graph.input("input", [2]);
        let zero_mask = zero_size_graph.input_dtype("mask", [2], DType::Bool);
        let zero_selected = zero_size_graph
            .masked_select(zero_input, zero_mask, 0, Scalar::F(-1.0))
            .unwrap();
        let zero_seed = zero_size_graph.input("seed", [0]);
        let zero_gradient = zero_size_graph
            .grad_with(zero_selected, zero_input, Some(zero_seed), true)
            .unwrap();
        let zero_values = HashMap::from([
            ("input".into(), data([2], &[1., 2.])),
            (
                "mask".into(),
                TensorData::from_scalars(
                    [2],
                    DType::Bool,
                    [Scalar::Bool(true), Scalar::Bool(true)],
                )
                .unwrap(),
            ),
            ("seed".into(), data([0], &[])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&zero_size_graph, zero_gradient, &zero_values)
                .unwrap(),
            data([2], &[0., 0.])
        );
    }

    #[test]
    fn scatter_positions_vjp_remains_compositional_at_third_order() {
        let mut graph = Graph::new();
        let x = graph.input("x", [4]);
        let selected = graph.shrink(x, [(1, 3)]).unwrap();
        let seed = graph.input("seed", [2]);
        let first = graph.grad_with(selected, x, Some(seed), true).unwrap();
        let direction = graph.input("direction", [4]);
        let first_weighted = graph.mul(first, direction).unwrap();
        let second_loss = graph.sum_all(first_weighted).unwrap();
        let second = graph.grad(second_loss, seed).unwrap();
        let third_seed = graph.input("third_seed", [2]);
        let second_weighted = graph.mul(second, third_seed).unwrap();
        let third_loss = graph.sum_all(second_weighted).unwrap();
        let third = graph.grad(third_loss, direction).unwrap();

        let values = HashMap::from([
            ("x".into(), data([4], &[1., 2., 3., 4.])),
            ("seed".into(), data([2], &[5., 7.])),
            ("direction".into(), data([4], &[11., 13., 17., 19.])),
            ("third_seed".into(), data([2], &[23., 29.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, second, &values).unwrap(),
            data([2], &[13., 17.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, third, &values).unwrap(),
            data([4], &[0., 23., 29., 0.])
        );

        let epsilon = 1e-2;
        let analytic = CpuBackend.execute(&graph, third, &values).unwrap();
        for index in 0..4 {
            let mut plus = [11.0f32, 13.0, 17.0, 19.0];
            let mut minus = plus;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_values = HashMap::from([
                ("x".into(), data([4], &[1., 2., 3., 4.])),
                ("seed".into(), data([2], &[5., 7.])),
                ("direction".into(), data([4], &plus)),
                ("third_seed".into(), data([2], &[23., 29.])),
            ]);
            let minus_values = HashMap::from([
                ("x".into(), data([4], &[1., 2., 3., 4.])),
                ("seed".into(), data([2], &[5., 7.])),
                ("direction".into(), data([4], &minus)),
                ("third_seed".into(), data([2], &[23., 29.])),
            ]);
            let numeric = (CpuBackend
                .execute(&graph, third_loss, &plus_values)
                .unwrap()
                .values()[0]
                - CpuBackend
                    .execute(&graph, third_loss, &minus_values)
                    .unwrap()
                    .values()[0])
                / (2.0 * epsilon);
            assert!((analytic.values()[index] - numeric).abs() < 1e-2);
        }
        assert!(
            graph
                .trace(third)
                .unwrap()
                .to_string()
                .contains("scatter_positions")
        );
    }

    #[test]
    fn static_index_gradient_vjp_reuses_normalized_duplicate_map() {
        use crate::ir::indexing::StaticIndex;

        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let selected = graph
            .static_index(
                x,
                &[StaticIndex::Advanced {
                    shape: Shape::from([3]),
                    values: vec![2, 0, 2],
                }],
            )
            .unwrap();
        let seed = graph.input("seed", [3]);
        let first = graph.grad_with(selected, x, Some(seed), true).unwrap();
        let direction = graph.input("direction", [3]);
        let weighted = graph.mul(first, direction).unwrap();
        let dot = graph.sum_all(weighted).unwrap();
        let seed_vjp = graph.grad(dot, seed).unwrap();

        let values = HashMap::from([
            ("x".into(), data([3], &[1., 2., 3.])),
            ("seed".into(), data([3], &[4., 5., 6.])),
            ("direction".into(), data([3], &[10., 20., 30.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, first, &values).unwrap(),
            data([3], &[5., 0., 10.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, seed_vjp, &values).unwrap(),
            data([3], &[30., 10., 30.])
        );
        // The scalarized first derivative is linear in the explicit upstream
        // seed, so central differences independently validate the second VJP.
        let epsilon = 1e-3;
        let analytic = CpuBackend.execute(&graph, seed_vjp, &values).unwrap();
        for index in 0..3 {
            let mut plus = [4.0f32, 5.0, 6.0];
            let mut minus = plus;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_values = HashMap::from([
                ("x".into(), data([3], &[1., 2., 3.])),
                ("seed".into(), data([3], &plus)),
                ("direction".into(), data([3], &[10., 20., 30.])),
            ]);
            let minus_values = HashMap::from([
                ("x".into(), data([3], &[1., 2., 3.])),
                ("seed".into(), data([3], &minus)),
                ("direction".into(), data([3], &[10., 20., 30.])),
            ]);
            let numeric = (CpuBackend
                .execute(&graph, dot, &plus_values)
                .unwrap()
                .values()[0]
                - CpuBackend
                    .execute(&graph, dot, &minus_values)
                    .unwrap()
                    .values()[0])
                / (2.0 * epsilon);
            assert!((analytic.values()[index] - numeric).abs() < 1e-1);
        }
        assert!(
            graph
                .trace(seed_vjp)
                .unwrap()
                .to_string()
                .contains("static_index")
        );
    }

    #[test]
    fn static_index_update_gradient_vjps_preserve_final_writers_and_broadcasts() {
        use crate::ir::indexing::StaticIndex;

        let mut graph = Graph::new();
        let base = graph.input("base", [4]);
        let value = graph.input("value", [1]);
        let updated = graph
            .static_index_update(
                base,
                &[StaticIndex::Advanced {
                    shape: Shape::from([3]),
                    values: vec![2, 1, 2],
                }],
                value,
            )
            .unwrap();
        let seed = graph.input("seed", [4]);
        let base_gradient = graph.grad_with(updated, base, Some(seed), true).unwrap();
        let value_gradient = graph.grad_with(updated, value, Some(seed), true).unwrap();
        let base_direction = graph.input("base_direction", [4]);
        let value_direction = graph.input("value_direction", [1]);
        let base_weighted = graph.mul(base_gradient, base_direction).unwrap();
        let base_dot = graph.sum_all(base_weighted).unwrap();
        let value_weighted = graph.mul(value_gradient, value_direction).unwrap();
        let value_dot = graph.sum_all(value_weighted).unwrap();
        let base_seed_vjp = graph.grad(base_dot, seed).unwrap();
        let value_seed_vjp = graph.grad(value_dot, seed).unwrap();

        let values = HashMap::from([
            ("base".into(), data([4], &[1., 2., 3., 4.])),
            ("value".into(), data([1], &[9.])),
            ("seed".into(), data([4], &[4., 5., 6., 7.])),
            ("base_direction".into(), data([4], &[10., 20., 30., 40.])),
            ("value_direction".into(), data([1], &[3.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, base_gradient, &values).unwrap(),
            data([4], &[4., 0., 0., 7.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, value_gradient, &values).unwrap(),
            data([1], &[11.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, base_seed_vjp, &values).unwrap(),
            data([4], &[10., 0., 0., 40.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, value_seed_vjp, &values).unwrap(),
            data([4], &[0., 3., 3., 0.])
        );

        let epsilon = 1e-3;
        let analytic = CpuBackend.execute(&graph, value_seed_vjp, &values).unwrap();
        for index in 0..4 {
            let mut plus = [4.0f32, 5.0, 6.0, 7.0];
            let mut minus = plus;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_values = HashMap::from([
                ("base".into(), data([4], &[1., 2., 3., 4.])),
                ("value".into(), data([1], &[9.])),
                ("seed".into(), data([4], &plus)),
                ("base_direction".into(), data([4], &[10., 20., 30., 40.])),
                ("value_direction".into(), data([1], &[3.])),
            ]);
            let minus_values = HashMap::from([
                ("base".into(), data([4], &[1., 2., 3., 4.])),
                ("value".into(), data([1], &[9.])),
                ("seed".into(), data([4], &minus)),
                ("base_direction".into(), data([4], &[10., 20., 30., 40.])),
                ("value_direction".into(), data([1], &[3.])),
            ]);
            let numeric = (CpuBackend
                .execute(&graph, value_dot, &plus_values)
                .unwrap()
                .values()[0]
                - CpuBackend
                    .execute(&graph, value_dot, &minus_values)
                    .unwrap()
                    .values()[0])
                / (2.0 * epsilon);
            assert!((analytic.values()[index] - numeric).abs() < 1e-2);
        }
        assert!(
            graph
                .trace(value_seed_vjp)
                .unwrap()
                .to_string()
                .contains("static_index_update")
        );

        let mut empty_graph = Graph::new();
        let empty_base = empty_graph.input("base", [2]);
        let empty_value = empty_graph.input("value", [0]);
        let empty_update = empty_graph
            .static_index_update(
                empty_base,
                &[StaticIndex::Slice {
                    start: Some(1),
                    stop: Some(1),
                    step: 1,
                }],
                empty_value,
            )
            .unwrap();
        let empty_seed = empty_graph.input("seed", [2]);
        let empty_gradient = empty_graph
            .grad_with(empty_update, empty_value, Some(empty_seed), true)
            .unwrap();
        let empty_direction = empty_graph.input("direction", [0]);
        let empty_weighted = empty_graph.mul(empty_gradient, empty_direction).unwrap();
        let empty_dot = empty_graph.sum_all(empty_weighted).unwrap();
        let empty_vjp = empty_graph.grad(empty_dot, empty_seed).unwrap();
        let empty_values = HashMap::from([
            ("base".into(), data([2], &[1., 2.])),
            ("value".into(), data([0], &[])),
            ("seed".into(), data([2], &[4., 5.])),
            ("direction".into(), data([0], &[])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&empty_graph, empty_gradient, &empty_values)
                .unwrap(),
            data([0], &[])
        );
        assert_eq!(
            CpuBackend
                .execute(&empty_graph, empty_vjp, &empty_values)
                .unwrap(),
            data([2], &[0., 0.])
        );
    }

    #[test]
    fn cumsum_gradient_uses_reverse_scan_and_retains_higher_order_edges() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let seed = graph.input("seed", [2, 3]);
        let scan = graph.cumsum(x, -1).unwrap();
        let seeded_scan = graph.mul(scan, seed).unwrap();
        let forward_loss = graph.sum_all(seeded_scan).unwrap();
        let gradient = graph.grad_with(scan, x, Some(seed), true).unwrap();
        let direction = graph.input("direction", [2, 3]);
        let weighted = graph.mul(gradient, direction).unwrap();
        let dot = graph.sum_all(weighted).unwrap();
        let seed_vjp = graph.grad(dot, seed).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("seed".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("direction".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
        ]);
        let analytic = CpuBackend.execute(&graph, gradient, &inputs).unwrap();
        assert_eq!(analytic, data([2, 3], &[6., 5., 3., 15., 11., 6.]));
        assert_eq!(
            CpuBackend.execute(&graph, seed_vjp, &inputs).unwrap(),
            data([2, 3], &[1., 3., 6., 4., 9., 15.])
        );
        let trace = graph.trace(seed_vjp).unwrap().to_string();
        assert!(trace.contains("cumsum"));
        assert!(trace.contains("stride"));

        let doubled = graph.add(scan, scan).unwrap();
        let accumulated = graph.grad_with(doubled, x, Some(seed), true).unwrap();
        assert_eq!(
            CpuBackend.execute(&graph, accumulated, &inputs).unwrap(),
            data([2, 3], &[12., 10., 6., 30., 22., 12.])
        );

        let epsilon = 1e-2;
        for index in 0..6 {
            let mut plus = [1., 2., 3., 4., 5., 6.];
            let mut minus = plus;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_inputs = HashMap::from([
                ("x".into(), data([2, 3], &plus)),
                ("seed".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
                ("direction".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ]);
            let minus_inputs = HashMap::from([
                ("x".into(), data([2, 3], &minus)),
                ("seed".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
                ("direction".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ]);
            let numeric = (CpuBackend
                .execute(&graph, forward_loss, &plus_inputs)
                .unwrap()
                .values()[0]
                - CpuBackend
                    .execute(&graph, forward_loss, &minus_inputs)
                    .unwrap()
                    .values()[0])
                / (2.0 * epsilon);
            assert!((numeric - analytic.values()[index]).abs() < 1e-2);
        }
    }

    #[test]
    fn cumsum_gradient_handles_scalar_empty_and_rejects_unsupported_scans_atomically() {
        let mut scalar_graph = Graph::new();
        let scalar = scalar_graph.input("scalar", []);
        let seed = scalar_graph.input("seed", []);
        let scan = scalar_graph.cumsum(scalar, -1).unwrap();
        let gradient = scalar_graph
            .grad_with(scan, scalar, Some(seed), true)
            .unwrap();
        let scalar_inputs = HashMap::from([
            ("scalar".into(), data([], &[7.])),
            ("seed".into(), data([], &[3.])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&scalar_graph, gradient, &scalar_inputs)
                .unwrap(),
            data([], &[3.])
        );

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [2, 0]);
        let empty_seed = empty_graph.input("seed", [2, 0]);
        let empty_scan = empty_graph.cumsum(empty, -1).unwrap();
        let empty_gradient = empty_graph
            .grad_with(empty_scan, empty, Some(empty_seed), true)
            .unwrap();
        let empty_inputs = HashMap::from([
            ("empty".into(), data([2, 0], &[])),
            ("seed".into(), data([2, 0], &[])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&empty_graph, empty_gradient, &empty_inputs)
                .unwrap(),
            data([2, 0], &[])
        );

        let mut product_graph = Graph::new();
        let product_input = product_graph.input("x", [2]);
        let product = product_graph.cumprod(product_input, 0).unwrap();
        let product_nodes = product_graph.node_count();
        let product_trace = product_graph.trace(product).unwrap();
        assert!(matches!(
            product_graph.grad(product, product_input),
            Err(Error::NonDifferentiableIndexing(
                "cumprod gradient is not yet represented"
            ))
        ));
        assert_eq!(product_graph.node_count(), product_nodes);
        assert_eq!(product_graph.trace(product).unwrap(), product_trace);

        let mut integer_graph = Graph::new();
        let integer = integer_graph.input_dtype("x", [2], DType::I32);
        let integer_scan = integer_graph.cumsum(integer, 0).unwrap();
        let integer_nodes = integer_graph.node_count();
        let integer_trace = integer_graph.trace(integer_scan).unwrap();
        assert!(matches!(
            integer_graph.grad(integer_scan, integer),
            Err(Error::NonDifferentiableIndexing(
                "cumsum gradients require floating input"
            ))
        ));
        assert_eq!(integer_graph.node_count(), integer_nodes);
        assert_eq!(integer_graph.trace(integer_scan).unwrap(), integer_trace);
    }

    #[test]
    fn conv2d_gradient_vjp_has_scalar_quadratic_hvp() {
        let mut graph = Graph::new();
        let x = graph.input("x", [1, 1, 1, 1]);
        let weight = graph.input("weight", [1, 1, 1, 1]);
        let output = graph
            .conv2d(x, weight, None, crate::Conv2dOptions::default())
            .unwrap();
        let squared = graph.square(output).unwrap();
        let loss = graph.sum_all(squared).unwrap();
        let gradient = graph.grad(loss, x).unwrap();
        let direction = graph.input("direction", [1, 1, 1, 1]);
        let dot = graph.mul(gradient, direction).unwrap();
        let dot_sum = graph.sum_all(dot).unwrap();
        let hvp = graph.grad(dot_sum, x).unwrap();
        let values = HashMap::from([
            ("x".into(), data([1, 1, 1, 1], &[3.])),
            ("weight".into(), data([1, 1, 1, 1], &[4.])),
            ("direction".into(), data([1, 1, 1, 1], &[5.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, hvp, &values).unwrap(),
            data([1, 1, 1, 1], &[160.])
        );
        assert!(
            graph
                .trace(hvp)
                .unwrap()
                .to_string()
                .contains("conv2d_grad_vjp")
        );
    }
}
