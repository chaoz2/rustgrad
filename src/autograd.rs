use crate::{BinaryOp, Error, Graph, NodeId, Op, Result, Shape, TensorData, UnaryOp};

impl Graph {
    /// Appends the reverse-mode derivative of a one-element `loss` with
    /// respect to `wrt` and returns the derivative node.
    pub fn grad(&mut self, loss: NodeId, wrt: NodeId) -> Result<NodeId> {
        let original_len = self.nodes.len();
        let loss_shape = self.node(loss)?.shape.clone();
        self.node(wrt)?;
        if loss_shape.numel()? != 1 {
            return Err(Error::NonScalarLoss(loss_shape));
        }

        let seed_data = filled(self.node(loss)?.shape.clone(), 1.0)?;
        let seed = self.constant(seed_data);
        let mut grads = vec![None; original_len];
        grads[loss.index()] = Some(seed);

        for index in (0..original_len).rev() {
            let Some(upstream) = grads[index] else {
                continue;
            };
            let node = NodeId(index);
            let op = self.node(node)?.op.clone();
            match op {
                Op::Input { .. } | Op::Constant(_) => {}
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
                Op::ArgReduce { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "reduction gradient not yet represented",
                    ));
                }
                Op::ReduceGrad { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "higher-order reduction gradient",
                    ));
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
                Op::MaskedSelect { .. } => {
                    return Err(Error::NonDifferentiableIndexing("masked_select"));
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
                Op::ScatterPositions { input, .. } => {
                    // Scatter only exists in backwards graphs and is not a public
                    // differentiable operation in this milestone.
                    self.accumulate(&mut grads, input, upstream)?;
                }
                Op::Matmul { lhs, rhs } => {
                    let lhs_grad = self.matmul_grad(upstream, lhs, rhs, true)?;
                    let rhs_grad = self.matmul_grad(upstream, lhs, rhs, false)?;
                    self.accumulate(&mut grads, lhs, lhs_grad)?;
                    self.accumulate(&mut grads, rhs, rhs_grad)?;
                }
                Op::MatmulGrad { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "higher-order matmul gradient",
                    ));
                }
                Op::Conv2d { .. } | Op::Conv2dGrad { .. } => {
                    return Err(Error::NonDifferentiableIndexing("conv2d gradient pending"));
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
    fn select_routes_gradients_and_predicates_are_nondifferentiable() {
        let mut graph = Graph::new();
        let condition_source = graph.input("condition_source", [2, 1]);
        let threshold = graph.constant(data([], &[0.0]));
        let condition = graph.gt(condition_source, threshold).unwrap();
        let on_true = graph.input("on_true", [2, 1]);
        let on_false = graph.input("on_false", [2]);
        let selected = graph.select(condition, on_true, on_false).unwrap();
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
}
