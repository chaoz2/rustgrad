use crate::{
    BinaryOp, DType, Error, Graph, NodeId, Op, ReductionDType, Result, Scalar, Shape, TensorData,
    UnaryOp,
};
use std::collections::BTreeSet;

/// Immutable reverse traversal frontier for one root and an ordered target
/// request. The bitmap marks exactly the nodes whose local reverse rule must
/// run to reach at least one requested target; target nodes themselves remain
/// gradient destinations unless another requested target lies below them.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ReverseTraversal {
    original_len: usize,
    active_inputs: Vec<Vec<NodeId>>,
}

impl ReverseTraversal {
    fn new(graph: &Graph, root: NodeId, targets: &[NodeId]) -> Result<Self> {
        let original_len = graph.node_count();
        graph.node(root)?;
        let target_set = targets
            .iter()
            .copied()
            .map(|target| graph.node(target).map(|_| target))
            .collect::<Result<BTreeSet<_>>>()?;

        let mut ancestors = vec![false; original_len];
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            if ancestors[node.index()] {
                continue;
            }
            ancestors[node.index()] = true;
            pending.extend(graph.reverse_inputs(node)?);
        }

        // Graph NodeIds are topological. Track whether each root ancestor
        // contains a requested target, then retain only rules needed to reach
        // one. This is tinygrad's `_deepwalk` target-path policy expressed over
        // RustGrad's typed reverse-edge projection.
        let mut contains_target = vec![false; original_len];
        let mut active_inputs = vec![vec![]; original_len];
        for index in 0..original_len {
            if !ancestors[index] {
                continue;
            }
            let node = NodeId(index);
            let inputs = graph.reverse_inputs(node)?;
            active_inputs[index] = inputs
                .iter()
                .copied()
                .filter(|input| contains_target[input.index()])
                .collect();
            let reaches_target = !active_inputs[index].is_empty();
            contains_target[index] = target_set.contains(&node) || reaches_target;
        }
        Ok(Self {
            original_len,
            active_inputs,
        })
    }

    fn contains(&self, index: usize) -> bool {
        !self.active_inputs[index].is_empty()
    }

    fn wants(&self, index: usize, input: NodeId) -> bool {
        self.active_inputs[index].contains(&input)
    }

    fn active_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.active_inputs
            .iter()
            .enumerate()
            .filter_map(|(index, inputs)| (!inputs.is_empty()).then_some(NodeId(index)))
    }
}

/// Fully resolved descriptor for tinygrad's Pow VJP. This deliberately leaves
/// Pow forward semantics alone: it only proves that the already-built Pow
/// node can receive tinygrad's literal, weak-constant backward expansion.
struct PowVjpPlan {
    zero_lhs: TensorData,
    zero_rhs: TensorData,
    one_rhs: TensorData,
    negative_inf: TensorData,
    zero_output: TensorData,
    ln2: TensorData,
}

/// Fully resolved local derivative for a direct tinygrad SQRT node. The
/// source spells this as `upstream / (result * 2)`, where the weak integer
/// literal adopts the result storage dtype rather than being an F32 scalar.
struct SqrtVjpPlan {
    two: TensorData,
}

fn sqrt_vjp_plan(
    graph: &Graph,
    node: NodeId,
    input: NodeId,
    upstream: NodeId,
) -> Result<SqrtVjpPlan> {
    let input_data = graph.node(input)?;
    let input_shape = input_data.shape.clone();
    let input_dtype = input_data.dtype;
    let result_data = graph.node(node)?;
    let result_shape = result_data.shape.clone();
    let result_dtype = result_data.dtype;
    let upstream_data = graph.node(upstream)?;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let fail = |actual| Error::InvalidElementwiseDType {
        op: "sqrt vjp source promotion",
        actual,
    };

    let expected_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    if result_shape != input_shape
        || result_dtype != expected_dtype
        || upstream_data.shape != result_shape
        || !result_dtype.is_float()
        || !upstream_data.dtype.is_float()
    {
        return Err(fail(result_dtype));
    }
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (&result_shape, result_dtype),
        (&upstream_data.shape, upstream_data.dtype),
    ] {
        extent(shape, dtype)?;
    }

    let two = TensorData::scalar_with_dtype(Scalar::I(2), result_dtype);
    extent(two.shape(), two.dtype())?;
    let denominator_shape = result_shape.broadcast_with(two.shape())?;
    let denominator_dtype = result_dtype.promote(two.dtype());
    let gradient_shape = upstream_data.shape.broadcast_with(&denominator_shape)?;
    let gradient_dtype = upstream_data.dtype.promote(denominator_dtype);
    if two.dtype() != result_dtype
        || denominator_shape != result_shape
        || denominator_dtype != result_dtype
        || gradient_shape != result_shape
        || !gradient_dtype.is_float()
    {
        return Err(fail(gradient_dtype));
    }
    for (shape, dtype) in [
        (&denominator_shape, denominator_dtype),
        (&gradient_shape, gradient_dtype),
        (&input_shape, gradient_dtype),
    ] {
        extent(shape, dtype)?;
    }
    Ok(SqrtVjpPlan { two })
}

/// Fully resolved literal tinygrad SIN derivative. tinygrad deliberately
/// writes `(pi/2 - x).sin() * ctx`, so pi/2 is a weak source scalar at the
/// input storage width rather than an F32 constant or a raw COS node.
struct SinVjpPlan {
    half_pi: TensorData,
}

fn sin_vjp_plan(
    graph: &Graph,
    node: NodeId,
    input: NodeId,
    upstream: NodeId,
) -> Result<SinVjpPlan> {
    let input_data = graph.node(input)?;
    let input_shape = input_data.shape.clone();
    let input_dtype = input_data.dtype;
    let result_data = graph.node(node)?;
    let result_shape = result_data.shape.clone();
    let result_dtype = result_data.dtype;
    let upstream_data = graph.node(upstream)?;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let fail = |actual| Error::InvalidElementwiseDType {
        op: "sin vjp source promotion",
        actual,
    };

    let expected_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    if result_shape != input_shape
        || result_dtype != expected_dtype
        || upstream_data.shape != result_shape
        || !input_dtype.is_float()
        || !upstream_data.dtype.is_float()
    {
        return Err(fail(result_dtype));
    }
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (&result_shape, result_dtype),
        (&upstream_data.shape, upstream_data.dtype),
    ] {
        extent(shape, dtype)?;
    }

    let half_pi =
        TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::FRAC_PI_2), input_dtype);
    extent(half_pi.shape(), half_pi.dtype())?;
    let phase_shape = half_pi.shape().broadcast_with(&input_shape)?;
    let phase_dtype = half_pi.dtype().promote(input_dtype);
    let sine_dtype = if phase_dtype.is_float() {
        phase_dtype
    } else {
        DType::F32
    };
    let gradient_shape = phase_shape.broadcast_with(&upstream_data.shape)?;
    let gradient_dtype = sine_dtype.promote(upstream_data.dtype);
    if half_pi.dtype() != input_dtype
        || phase_shape != input_shape
        || phase_dtype != input_dtype
        || sine_dtype != result_dtype
        || gradient_shape != result_shape
        || !gradient_dtype.is_float()
    {
        return Err(fail(gradient_dtype));
    }
    for (shape, dtype) in [
        (&phase_shape, phase_dtype),
        (&phase_shape, sine_dtype),
        (&gradient_shape, gradient_dtype),
        (&input_shape, gradient_dtype),
    ] {
        extent(shape, dtype)?;
    }
    Ok(SinVjpPlan { half_pi })
}

fn pow_vjp_plan(
    graph: &Graph,
    node: NodeId,
    lhs: NodeId,
    rhs: NodeId,
    upstream: NodeId,
    needed: [bool; 2],
) -> Result<PowVjpPlan> {
    let lhs_node = graph.node(lhs)?;
    let lhs_shape = lhs_node.shape.clone();
    let lhs_dtype = lhs_node.dtype;
    let rhs_node = graph.node(rhs)?;
    let rhs_shape = rhs_node.shape.clone();
    let rhs_dtype = rhs_node.dtype;
    let node_data = graph.node(node)?;
    let output_shape = node_data.shape.clone();
    let output_dtype = node_data.dtype;
    let upstream_data = graph.node(upstream)?;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let source_promote = |left: DType, right: DType| {
        if matches!(
            (left, right),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            left.promote(right)
        }
    };
    let fail = |actual| Error::InvalidElementwiseDType {
        op: "pow vjp source promotion",
        actual,
    };

    // The forward node must retain the raw Pow descriptor it was built with;
    // this phase intentionally does not reinterpret or repair it.
    let pow_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let pow_dtype = lhs_dtype.promote(rhs_dtype);
    if output_shape != pow_shape || output_dtype != pow_dtype || upstream_data.shape != output_shape
    {
        return Err(fail(output_dtype));
    }
    for (shape, dtype) in [
        (&lhs_shape, lhs_dtype),
        (&rhs_shape, rhs_dtype),
        (&output_shape, output_dtype),
        (&upstream_data.shape, upstream_data.dtype),
    ] {
        extent(shape, dtype)?;
    }
    if !output_dtype.is_float() || !upstream_data.dtype.is_float() {
        return Err(fail(output_dtype));
    }

    // Python literals in the source are weak: comparison/subtraction literals
    // adopt their lhs storage, while ret.const_like and the final ln(2) tail
    // use the result/tail storage.
    let zero_lhs = TensorData::scalar_with_dtype(Scalar::I(0), lhs_dtype);
    let zero_rhs = TensorData::scalar_with_dtype(Scalar::I(0), rhs_dtype);
    let one_rhs = TensorData::scalar_with_dtype(Scalar::I(1), rhs_dtype);
    let negative_inf = TensorData::scalar_with_dtype(Scalar::F(f64::NEG_INFINITY), output_dtype);
    let zero_output = TensorData::scalar_with_dtype(Scalar::I(0), output_dtype);
    let log_dtype = if lhs_dtype.is_float() {
        lhs_dtype
    } else {
        DType::F32
    };
    let log_shape = lhs_shape.clone();
    let tail_dtype = source_promote(output_dtype, log_dtype);
    let ln2 = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LN_2), tail_dtype);
    if needed[0] {
        for scalar in [&zero_lhs, &zero_rhs, &one_rhs] {
            extent(scalar.shape(), scalar.dtype())?;
        }
        let rhs_minus_one_shape = rhs_shape.broadcast_with(one_rhs.shape())?;
        let rhs_minus_one_dtype = source_promote(rhs_dtype, one_rhs.dtype());
        let power_shape = lhs_shape.broadcast_with(&rhs_minus_one_shape)?;
        let power_dtype = lhs_dtype.promote(rhs_minus_one_dtype);
        let base_local_shape = rhs_shape.broadcast_with(&power_shape)?;
        let base_local_dtype = source_promote(rhs_dtype, power_dtype);
        let lhs_grad_dtype = source_promote(upstream_data.dtype, base_local_dtype);
        if zero_lhs.dtype() != lhs_dtype
            || zero_rhs.dtype() != rhs_dtype
            || one_rhs.dtype() != rhs_dtype
            || rhs_minus_one_shape != rhs_shape
            || rhs_minus_one_dtype != rhs_dtype
            || power_shape != output_shape
            || power_dtype != output_dtype
            || base_local_shape != output_shape
            || base_local_dtype != output_dtype
            || lhs_shape.broadcast_with(&output_shape)? != output_shape
        {
            return Err(fail(output_dtype));
        }
        for (shape, dtype) in [
            (&rhs_minus_one_shape, rhs_minus_one_dtype),
            (&power_shape, power_dtype),
            (&base_local_shape, base_local_dtype),
            (&output_shape, lhs_grad_dtype),
            (&lhs_shape, lhs_grad_dtype),
            (&output_shape, DType::Bool),
            (&lhs_shape, DType::Bool),
            (&rhs_shape, DType::Bool),
        ] {
            extent(shape, dtype)?;
        }
    }
    if needed[1] {
        for scalar in [&zero_lhs, &zero_rhs, &negative_inf, &zero_output, &ln2] {
            extent(scalar.shape(), scalar.dtype())?;
        }
        let zero_local_shape = rhs_shape
            .broadcast_with(negative_inf.shape())?
            .broadcast_with(zero_output.shape())?;
        let zero_local_dtype = source_promote(output_dtype, output_dtype);
        let tail_shape = output_shape.broadcast_with(&log_shape)?;
        let final_shape = lhs_shape
            .broadcast_with(&zero_local_shape)?
            .broadcast_with(&tail_shape)?;
        let final_dtype = source_promote(zero_local_dtype, tail_dtype);
        let rhs_grad_dtype = source_promote(upstream_data.dtype, final_dtype);
        if zero_lhs.dtype() != lhs_dtype
            || zero_rhs.dtype() != rhs_dtype
            || negative_inf.dtype() != output_dtype
            || zero_output.dtype() != output_dtype
            || ln2.dtype() != tail_dtype
            || zero_local_dtype != output_dtype
            || tail_shape != output_shape
            || final_shape != output_shape
            || rhs_shape.broadcast_with(&output_shape)? != output_shape
        {
            return Err(fail(output_dtype));
        }
        for (shape, dtype) in [
            (&zero_local_shape, zero_local_dtype),
            (&log_shape, log_dtype),
            (&tail_shape, tail_dtype),
            (&final_shape, final_dtype),
            (&output_shape, rhs_grad_dtype),
            (&rhs_shape, rhs_grad_dtype),
            (&output_shape, DType::Bool),
            (&lhs_shape, DType::Bool),
            (&rhs_shape, DType::Bool),
        ] {
            extent(shape, dtype)?;
        }
    }
    Ok(PowVjpPlan {
        zero_lhs,
        zero_rhs,
        one_rhs,
        negative_inf,
        zero_output,
        ln2,
    })
}

impl Graph {
    /// Appends the reverse-mode derivative of a one-element `loss` with
    /// respect to `wrt` and returns the derivative node.
    pub fn grad(&mut self, loss: NodeId, wrt: NodeId) -> Result<NodeId> {
        self.grad_with(loss, wrt, None, true)
    }

    /// Builds a reverse-mode derivative graph. An implicit seed is allowed only
    /// for one-element outputs; `create_graph` controls whether this derivative
    /// itself retains gradient edges for higher-order differentiation. Only the
    /// graph-checked root-to-`wrt` value slice is transformed, and the complete
    /// candidate commits atomically after every local rule succeeds.
    pub fn grad_with(
        &mut self,
        loss: NodeId,
        wrt: NodeId,
        upstream: Option<NodeId>,
        create_graph: bool,
    ) -> Result<NodeId> {
        // Construct the complete derivative in isolation. A late local VJP,
        // shape, dtype, or allocation-geometry failure cannot publish a seed
        // or a partial derivative graph into the caller's arena.
        let mut candidate = self.clone();
        let previous_enabled = candidate.grad_enabled;
        candidate.grad_enabled = create_graph;
        let result = candidate.grad_with_inner(loss, wrt, upstream);
        candidate.grad_enabled = previous_enabled;
        let result = result?;
        *self = candidate;
        Ok(result)
    }

    fn grad_with_inner(
        &mut self,
        loss: NodeId,
        wrt: NodeId,
        upstream: Option<NodeId>,
    ) -> Result<NodeId> {
        let loss_node = self.node(loss)?;
        if !loss_node.dtype.is_float() {
            return Err(Error::NonDifferentiableTarget(loss));
        }
        let loss_shape = loss_node.shape.clone();
        let target = self.node(wrt)?;
        if !target.dtype.is_float() || !target.requires_grad {
            return Err(Error::NonDifferentiableTarget(wrt));
        }
        let traversal = ReverseTraversal::new(self, loss, &[wrt])?;
        self.validate_reverse_path(&traversal)?;
        if let Some(upstream) = upstream {
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
        }
        self.reverse_with_inner(loss, upstream, &traversal, loss_shape)
            .and_then(|grads| grads[wrt.index()].ok_or(Error::NoGradient(wrt)))
    }

    /// Computes tinygrad's public multi-target `Tensor.gradient` transform.
    ///
    /// All targets are differentiated in one reverse traversal. Repeated
    /// targets therefore return the same graph node, while a disconnected
    /// floating target receives an independent typed `const_like(0)` value.
    /// An explicit seed is normalized by the established equal-element-count
    /// reshape or broadcast-to-loss expansion policy.
    /// Unlike [`Graph::grad_with`], this source-facing transform deliberately
    /// does not use RustGrad's leaf `requires_grad` policy for admission:
    /// tinygrad accepts any concrete strong floating target.
    pub fn gradient(
        &mut self,
        loss: NodeId,
        targets: &[NodeId],
        gradient: Option<NodeId>,
    ) -> Result<Vec<NodeId>> {
        // Build the complete reverse graph, including every disconnected zero
        // fallback, in one private candidate. Commit that candidate only after
        // every seed, local derivative, accumulation, and fallback succeeds.
        // Preserve the caller's graph-local no-grad mode while forcing the
        // returned derivative itself to retain ordinary higher-order edges.
        let mut candidate = self.clone();
        let previous_enabled = candidate.grad_enabled;
        candidate.grad_enabled = true;
        let result = candidate.gradient_source_inner(loss, targets, gradient);
        candidate.grad_enabled = previous_enabled;
        let result = result?;
        *self = candidate;
        Ok(result)
    }

    /// Source-default `Tensor.gradient(*targets)` with an implicit scalar one
    /// seed.
    pub fn gradient_default(&mut self, loss: NodeId, targets: &[NodeId]) -> Result<Vec<NodeId>> {
        self.gradient(loss, targets, None)
    }

    fn gradient_source_inner(
        &mut self,
        loss: NodeId,
        targets: &[NodeId],
        gradient: Option<NodeId>,
    ) -> Result<Vec<NodeId>> {
        let loss_node = self.node(loss)?;
        if !loss_node.dtype.is_float() {
            return Err(Error::NonDifferentiableTarget(loss));
        }
        let loss_shape = loss_node.shape.clone();
        for &target in targets {
            if !self.node(target)?.dtype.is_float() {
                return Err(Error::NonDifferentiableTarget(target));
            }
        }

        if targets.is_empty() {
            if let Some(gradient) = gradient {
                // Preserve graph ownership/unknown-node admission for an
                // explicitly supplied handle, but it has no consumer and no
                // shape contract when no derivative was requested.
                self.node(gradient)?;
            } else if loss_shape.rank() != 0 {
                return Err(Error::NonScalarLoss(loss_shape));
            }
            return Ok(vec![]);
        }

        let traversal = ReverseTraversal::new(self, loss, targets)?;
        self.validate_reverse_path(&traversal)?;

        let gradient = match gradient {
            Some(gradient) => {
                let shape = self.node(gradient)?.shape.clone();
                if shape == loss_shape {
                    gradient
                } else if shape.numel()? == loss_shape.numel()? {
                    self.reshape(gradient, loss_shape.clone())?
                } else if shape.broadcast_with(&loss_shape).as_ref() == Ok(&loss_shape) {
                    self.expand(gradient, loss_shape.clone())?
                } else {
                    return Err(Error::GradientShape {
                        output: loss_shape,
                        upstream: shape,
                    });
                }
            }
            None => {
                // tinygrad tests the public shape tuple, not merely numel, so
                // rank-one `[1]` outputs still require an explicit gradient.
                if loss_shape.rank() != 0 {
                    return Err(Error::NonScalarLoss(loss_shape));
                }
                self.const_like(loss, Scalar::F(1.0), None)?
            }
        };

        let previous_enabled = self.grad_enabled;
        self.grad_enabled = true;
        let result = self.reverse_with_inner(loss, Some(gradient), &traversal, loss_shape);
        self.grad_enabled = previous_enabled;
        let grads = result?;

        let mut zeros = std::collections::BTreeMap::new();
        targets
            .iter()
            .map(|&target| {
                if let Some(gradient) = grads[target.index()] {
                    Ok(gradient)
                } else if let Some(&zero) = zeros.get(&target) {
                    Ok(zero)
                } else {
                    let zero = self.const_like(target, Scalar::F(0.0), None)?;
                    zeros.insert(target, zero);
                    Ok(zero)
                }
            })
            .collect()
    }

    fn reverse_with_inner(
        &mut self,
        loss: NodeId,
        upstream: Option<NodeId>,
        traversal: &ReverseTraversal,
        loss_shape: Shape,
    ) -> Result<Vec<Option<NodeId>>> {
        let seed = if let Some(upstream) = upstream {
            upstream
        } else if loss_shape.numel()? != 1 {
            return Err(Error::NonScalarLoss(loss_shape));
        } else {
            let loss_node = self.node(loss)?;
            let seed_data = filled(loss_node.shape.clone(), 1.0, loss_node.dtype)?;
            self.constant(seed_data)
        };
        let mut grads = vec![None; traversal.original_len];
        grads[loss.index()] = Some(seed);

        for index in (0..traversal.original_len).rev() {
            if !traversal.contains(index) {
                continue;
            }
            let Some(upstream) = grads[index] else {
                continue;
            };
            let node = NodeId(index);
            let op = self.node(node)?.op.clone();
            let wants = |input| traversal.wants(index, input);
            match op {
                Op::Input { .. }
                | Op::Constant(_)
                | Op::Random { .. }
                | Op::RandomPermutation { .. }
                | Op::Threefry { .. }
                | Op::Detach { .. }
                | Op::Bitcast { .. } => {}
                Op::Contiguous { input } => {
                    self.accumulate(&mut grads, input, upstream)?;
                }
                Op::ContiguousBackward { input } => {
                    let local = self.contiguous(upstream)?;
                    self.accumulate(&mut grads, input, local)?;
                }
                Op::Cast { input, .. } => {
                    // A floating cast has an identity local derivative, but
                    // its cotangent belongs to the source storage dtype. This
                    // matches tinygrad's CAST rule and keeps mixed-precision
                    // accumulation type-stable at the differentiated leaf.
                    let input_dtype = self.node(input)?.dtype;
                    let output_dtype = self.node(node)?.dtype;
                    if input_dtype.is_float() && output_dtype.is_float() {
                        let local = if self.node(upstream)?.dtype == input_dtype {
                            upstream
                        } else {
                            self.cast(upstream, input_dtype)?
                        };
                        self.accumulate(&mut grads, input, local)?;
                    }
                }
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
                            let input_node = self.node(input)?;
                            let zeros = filled(input_node.shape.clone(), 0.0, input_node.dtype)?;
                            self.constant(zeros)
                        }
                        UnaryOp::Abs => {
                            let sign = self.sign(input)?;
                            self.mul(upstream, sign)?
                        }
                        UnaryOp::Reciprocal => {
                            // tinygrad's literal rule is `-ctx * ret * ret`.
                            // Preserve that order as well as using the
                            // reciprocal result rather than the source input.
                            let local = self.neg(upstream)?;
                            let local = self.mul(local, node)?;
                            self.mul(local, node)?
                        }
                        UnaryOp::Square => {
                            let dtype = self.node(node)?.dtype;
                            let two =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(2), dtype));
                            let scale = self.mul(two, input)?;
                            self.mul(upstream, scale)?
                        }
                        UnaryOp::Sqrt => {
                            let plan = sqrt_vjp_plan(self, node, input, upstream)?;
                            let two = self.constant(plan.two);
                            let denominator = self.mul(node, two)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Rsqrt => {
                            let dtype = self.node(node)?.dtype;
                            let two =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(2), dtype));
                            let square = self.mul(node, node)?;
                            let cube = self.mul(square, node)?;
                            let scaled = self.div(cube, two)?;
                            let local = self.mul(upstream, scaled)?;
                            self.neg(local)?
                        }
                        UnaryOp::Exp2 => {
                            // tinygrad's weak ln(2) adopts the Exp2 storage
                            // dtype. This matters both for the F64 path used
                            // by Tensor.exp and for narrow direct Exp2 VJPs.
                            let dtype = self.node(node)?.dtype;
                            let ln2 = self.constant(TensorData::scalar_with_dtype(
                                Scalar::F(std::f64::consts::LN_2),
                                dtype,
                            ));
                            let scale = self.mul(node, ln2)?;
                            self.mul(upstream, scale)?
                        }
                        UnaryOp::Log2 => {
                            // tinygrad's weak ln(2) adopts the Log2 storage
                            // dtype, including narrow and F64 VJP paths.
                            let dtype = self.node(node)?.dtype;
                            let ln2 = self.constant(TensorData::scalar_with_dtype(
                                Scalar::F(std::f64::consts::LN_2),
                                dtype,
                            ));
                            let denominator = self.mul(input, ln2)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Sin => {
                            let plan = sin_vjp_plan(self, node, input, upstream)?;
                            let half_pi = self.constant(plan.half_pi);
                            let phase = self.sub(half_pi, input)?;
                            let local = self.sin(phase)?;
                            self.mul(local, upstream)?
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
                            let dtype = self.node(node)?.dtype;
                            let one =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(1), dtype));
                            let square = self.mul(node, node)?;
                            let local = self.sub(one, square)?;
                            self.mul(upstream, local)?
                        }
                        UnaryOp::Erf | UnaryOp::Erfc => {
                            let dtype = self.node(node)?.dtype;
                            let two_over_sqrt_pi = self.constant(TensorData::scalar_with_dtype(
                                Scalar::F(2.0 / std::f64::consts::PI.sqrt()),
                                dtype,
                            ));
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
                            let dtype = self.node(node)?.dtype;
                            let one =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(1), dtype));
                            let square = self.square(input)?;
                            let difference = self.sub(one, square)?;
                            let denominator = self.sqrt(difference)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Acos => {
                            let dtype = self.node(node)?.dtype;
                            let one =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(1), dtype));
                            let square = self.square(input)?;
                            let difference = self.sub(one, square)?;
                            let denominator = self.sqrt(difference)?;
                            let quotient = self.div(upstream, denominator)?;
                            self.neg(quotient)?
                        }
                        UnaryOp::Atan => {
                            let dtype = self.node(node)?.dtype;
                            let one =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(1), dtype));
                            let square = self.square(input)?;
                            let denominator = self.add(one, square)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Asinh => {
                            let dtype = self.node(node)?.dtype;
                            let one =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(1), dtype));
                            let square = self.square(input)?;
                            let sum = self.add(square, one)?;
                            let denominator = self.sqrt(sum)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Acosh => {
                            let dtype = self.node(node)?.dtype;
                            let one =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(1), dtype));
                            let square = self.square(input)?;
                            let difference = self.sub(square, one)?;
                            let denominator = self.sqrt(difference)?;
                            self.div(upstream, denominator)?
                        }
                        UnaryOp::Atanh => {
                            let dtype = self.node(node)?.dtype;
                            let one =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(1), dtype));
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
                            let input_node = self.node(input)?;
                            let zeros = filled(input_node.shape.clone(), 0.0, input_node.dtype)?;
                            self.constant(zeros)
                        }
                    };
                    self.accumulate(&mut grads, input, local)?;
                }
                Op::Binary { op, lhs, rhs } => {
                    let lhs_needed = wants(lhs);
                    let rhs_needed = wants(rhs);
                    let (lhs_grad, rhs_grad) = match op {
                        BinaryOp::Add => (
                            lhs_needed.then_some(upstream),
                            rhs_needed.then_some(upstream),
                        ),
                        BinaryOp::Sub => (
                            lhs_needed.then_some(upstream),
                            if rhs_needed {
                                Some(self.neg(upstream)?)
                            } else {
                                None
                            },
                        ),
                        BinaryOp::Mul => (
                            if lhs_needed {
                                Some(self.mul(upstream, rhs)?)
                            } else {
                                None
                            },
                            if rhs_needed {
                                Some(self.mul(upstream, lhs)?)
                            } else {
                                None
                            },
                        ),
                        BinaryOp::Div => {
                            let lhs_grad = if lhs_needed {
                                Some(self.div(upstream, rhs)?)
                            } else {
                                None
                            };
                            let rhs_grad = if rhs_needed {
                                let rhs_sq = self.mul(rhs, rhs)?;
                                let numerator = self.mul(upstream, lhs)?;
                                let quotient = self.div(numerator, rhs_sq)?;
                                Some(self.neg(quotient)?)
                            } else {
                                None
                            };
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Pow => {
                            // Validate each requested source-literal branch
                            // before its first constant. Unrequested operand
                            // derivatives are neither planned nor published.
                            let plan = pow_vjp_plan(
                                self,
                                node,
                                lhs,
                                rhs,
                                upstream,
                                [lhs_needed, rhs_needed],
                            )?;
                            let lhs_grad = if lhs_needed {
                                let zero_lhs = self.constant(plan.zero_lhs.clone());
                                let zero_rhs = self.constant(plan.zero_rhs.clone());
                                let one_rhs = self.constant(plan.one_rhs.clone());
                                let exponent_is_zero = self.eq(rhs, zero_rhs)?;
                                let exponent_minus_one = self.sub(rhs, one_rhs)?;
                                let power = self.pow(lhs, exponent_minus_one)?;
                                let base_local = self.mul(rhs, power)?;
                                let base_local = self.select(exponent_is_zero, rhs, base_local)?;
                                let upstream_is_zero = self.eq(upstream, zero_lhs)?;
                                let base_local =
                                    self.select(upstream_is_zero, zero_lhs, base_local)?;
                                Some(self.mul(upstream, base_local)?)
                            } else {
                                None
                            };
                            let rhs_grad = if rhs_needed {
                                let zero_lhs = self.constant(plan.zero_lhs);
                                let zero_rhs = self.constant(plan.zero_rhs);
                                let base_is_zero = self.eq(lhs, zero_lhs)?;
                                let exponent_negative = self.lt(rhs, zero_rhs)?;
                                let negative_inf = self.constant(plan.negative_inf);
                                let exponent_zero = self.constant(plan.zero_output);
                                let zero_local =
                                    self.select(exponent_negative, negative_inf, exponent_zero)?;
                                let logarithm = self.log2(lhs)?;
                                let ln2 = self.constant(plan.ln2);
                                let exponent_local = self.mul(node, logarithm)?;
                                let exponent_local = self.mul(exponent_local, ln2)?;
                                let exponent_local =
                                    self.select(base_is_zero, zero_local, exponent_local)?;
                                Some(self.mul(upstream, exponent_local)?)
                            } else {
                                None
                            };
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Atan2 => {
                            let lhs_square = self.square(lhs)?;
                            let rhs_square = self.square(rhs)?;
                            let denominator = self.add(lhs_square, rhs_square)?;
                            let lhs_grad = if lhs_needed {
                                let numerator = self.mul(upstream, rhs)?;
                                Some(self.div(numerator, denominator)?)
                            } else {
                                None
                            };
                            let rhs_grad = if rhs_needed {
                                let numerator = self.mul(upstream, lhs)?;
                                let quotient = self.div(numerator, denominator)?;
                                Some(self.neg(quotient)?)
                            } else {
                                None
                            };
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Copysign => {
                            // This is tinygrad's comparison/reciprocal sign
                            // selection: differentiable in the magnitude
                            // argument away from zero, and zero in sign.
                            let lhs_grad = if lhs_needed {
                                let lhs_sign = self.sign(lhs)?;
                                let local = self.copysign(lhs_sign, rhs)?;
                                Some(self.mul(upstream, local)?)
                            } else {
                                None
                            };
                            let rhs_grad = if rhs_needed {
                                let rhs_node = self.node(rhs)?;
                                Some(self.constant(filled(
                                    rhs_node.shape.clone(),
                                    0.0,
                                    rhs_node.dtype,
                                )?))
                            } else {
                                None
                            };
                            (lhs_grad, rhs_grad)
                        }
                        BinaryOp::Maximum | BinaryOp::Minimum => {
                            let equal = self.eq(lhs, rhs)?;
                            let dtype = self.node(upstream)?.dtype;
                            let zero =
                                self.constant(TensorData::scalar_with_dtype(Scalar::I(0), dtype));
                            let half =
                                self.constant(TensorData::scalar_with_dtype(Scalar::F(0.5), dtype));
                            let half_upstream = self.mul(upstream, half)?;
                            let lhs_grad = if lhs_needed {
                                let winner = if op == BinaryOp::Maximum {
                                    self.gt(lhs, rhs)?
                                } else {
                                    self.lt(lhs, rhs)?
                                };
                                let tie = self.select(equal, half_upstream, zero)?;
                                Some(self.select(winner, upstream, tie)?)
                            } else {
                                None
                            };
                            let rhs_grad = if rhs_needed {
                                let winner = if op == BinaryOp::Maximum {
                                    self.lt(lhs, rhs)?
                                } else {
                                    self.gt(lhs, rhs)?
                                };
                                let tie = self.select(equal, half_upstream, zero)?;
                                Some(self.select(winner, upstream, tie)?)
                            } else {
                                None
                            };
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
                            let zeros_l = if lhs_needed {
                                let lhs_node = self.node(lhs)?;
                                Some(self.constant(filled(
                                    lhs_node.shape.clone(),
                                    0.0,
                                    lhs_node.dtype,
                                )?))
                            } else {
                                None
                            };
                            let zeros_r = if rhs_needed {
                                let rhs_node = self.node(rhs)?;
                                Some(self.constant(filled(
                                    rhs_node.shape.clone(),
                                    0.0,
                                    rhs_node.dtype,
                                )?))
                            } else {
                                None
                            };
                            (zeros_l, zeros_r)
                        }
                    };
                    if let Some(lhs_grad) = lhs_grad {
                        let lhs_shape = self.node(lhs)?.shape.clone();
                        let lhs_grad = self.unbroadcast(lhs_grad, lhs_shape)?;
                        self.accumulate(&mut grads, lhs, lhs_grad)?;
                    }
                    if let Some(rhs_grad) = rhs_grad {
                        let rhs_shape = self.node(rhs)?.shape.clone();
                        let rhs_grad = self.unbroadcast(rhs_grad, rhs_shape)?;
                        self.accumulate(&mut grads, rhs, rhs_grad)?;
                    }
                }
                Op::Reduce {
                    input,
                    kind: crate::ReduceKind::Sum,
                    axes,
                    keepdim,
                    ..
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
                    let input_dtype = self.node(input)?.dtype;
                    let grad = if self.node(grad)?.dtype == input_dtype {
                        grad
                    } else {
                        self.cast(grad, input_dtype)?
                    };
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Reduce {
                    input,
                    kind: crate::ReduceKind::Mean,
                    axes,
                    keepdim,
                    ..
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
                    let dtype = self.node(up)?.dtype;
                    let divisor = self.constant(TensorData::scalar_with_dtype(
                        Scalar::U(count as u64),
                        dtype,
                    ));
                    let grad = self.div(up, divisor)?;
                    let input_dtype = self.node(input)?.dtype;
                    let grad = if self.node(grad)?.dtype == input_dtype {
                        grad
                    } else {
                        self.cast(grad, input_dtype)?
                    };
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
                    ..
                } => {
                    let grad = self.reduce_grad(input, upstream, kind, axes, keepdim)?;
                    let input_dtype = self.node(input)?.dtype;
                    let grad = if self.node(grad)?.dtype == input_dtype {
                        grad
                    } else {
                        self.cast(grad, input_dtype)?
                    };
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Reduce {
                    kind: crate::ReduceKind::Any | crate::ReduceKind::All,
                    ..
                } => {}
                Op::ArgReduce { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "reduction gradient not yet represented",
                    ));
                }
                Op::PrefixScan {
                    input,
                    axis,
                    kind: crate::PrefixScanKind::Sum,
                    ..
                } => {
                    let gradient = self.cumsum_vjp(upstream, axis)?;
                    let input_dtype = self.node(input)?.dtype;
                    let gradient = if self.node(gradient)?.dtype == input_dtype {
                        gradient
                    } else {
                        self.cast(gradient, input_dtype)?
                    };
                    self.accumulate(&mut grads, input, gradient)?;
                }
                Op::PrefixScan {
                    input,
                    axis,
                    kind: crate::PrefixScanKind::Product,
                    ..
                } => {
                    let gradient = self.cumprod_vjp(upstream, input, axis)?;
                    let input_dtype = self.node(input)?.dtype;
                    let gradient = if self.node(gradient)?.dtype == input_dtype {
                        gradient
                    } else {
                        self.cast(gradient, input_dtype)?
                    };
                    self.accumulate(&mut grads, input, gradient)?;
                }
                Op::PrefixScan {
                    input,
                    axis,
                    kind: crate::PrefixScanKind::Max | crate::PrefixScanKind::Min,
                    output: crate::PrefixScanOutput::Values,
                } => {
                    let gradient = self.cumulative_extrema_vjp(upstream, input, node, axis)?;
                    let input_dtype = self.node(input)?.dtype;
                    let gradient = if self.node(gradient)?.dtype == input_dtype {
                        gradient
                    } else {
                        self.cast(gradient, input_dtype)?
                    };
                    self.accumulate(&mut grads, input, gradient)?;
                }
                Op::PrefixScan {
                    kind: crate::PrefixScanKind::Max | crate::PrefixScanKind::Min,
                    output: crate::PrefixScanOutput::Indices,
                    ..
                } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "cumulative extrema indices",
                    ));
                }
                Op::Sort {
                    input,
                    axis,
                    descending,
                    pair,
                    output: crate::SortOutput::Values,
                } => {
                    let indices = self.sort_indices_sibling(node, input, axis, descending, pair)?;
                    let (source_shape, source_dtype) = {
                        let source = self.node(input)?;
                        (source.shape.clone(), source.dtype)
                    };
                    let grad = if source_shape.rank() == 0 {
                        if self.node(upstream)?.dtype == source_dtype {
                            upstream
                        } else {
                            self.cast(upstream, source_dtype)?
                        }
                    } else {
                        let upstream = if self.node(upstream)?.dtype == source_dtype {
                            upstream
                        } else {
                            self.cast(upstream, source_dtype)?
                        };
                        let zeros = self.constant(filled(source_shape, 0.0, source_dtype)?);
                        self.scatter_add(zeros, indices, upstream, axis)?
                    };
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Sort {
                    output: crate::SortOutput::Indices,
                    ..
                } => {
                    return Err(Error::NonDifferentiableIndexing("sort indices"));
                }
                Op::TensorGuard { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "tensor guard gradient is not represented",
                    ));
                }
                Op::ReduceGrad {
                    input,
                    upstream: first_upstream,
                    kind,
                    axes,
                    keepdim,
                } => {
                    if wants(first_upstream) {
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
                    }
                    if wants(input) && self.node(input)?.dtype.is_float() {
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
                    let grad = self.unbroadcast(upstream, input_shape)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::Gather { input, index, axis } => {
                    let shape = self.node(input)?.shape.clone();
                    let dtype = self.node(input)?.dtype;
                    let zeros = self.constant(filled(shape, 0.0, dtype)?);
                    let grad = self.scatter_add(zeros, index, upstream, axis)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::StaticIndex { input, plan } => {
                    let shape = self.node(input)?.shape.clone();
                    let grad = self.static_index_grad(upstream, shape, plan)?;
                    self.accumulate(&mut grads, input, grad)?;
                }
                Op::StaticIndexGrad { .. } => {
                    return Err(Error::NonDifferentiableIndexing("static index gradient"));
                }
                Op::StaticIndexUpdate { base, value, plan } => {
                    if self.node(node)?.dtype != crate::DType::F32 {
                        return Err(Error::NonDifferentiableIndexing(
                            "static index update gradients require F32",
                        ));
                    }
                    if wants(base) {
                        let base_grad = self.static_index_update_grad(
                            upstream,
                            self.node(base)?.shape.clone(),
                            self.node(value)?.shape.clone(),
                            plan.clone(),
                            crate::StaticIndexUpdateWrt::Base,
                        )?;
                        self.accumulate(&mut grads, base, base_grad)?;
                    }
                    if wants(value) {
                        let value_grad = self.static_index_update_grad(
                            upstream,
                            self.node(base)?.shape.clone(),
                            self.node(value)?.shape.clone(),
                            plan,
                            crate::StaticIndexUpdateWrt::Value,
                        )?;
                        self.accumulate(&mut grads, value, value_grad)?;
                    }
                }
                Op::StaticIndexUpdateGrad { .. } => {
                    return Err(Error::NonDifferentiableIndexing(
                        "static index update gradient",
                    ));
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
                    if wants(base) {
                        self.accumulate(&mut grads, base, upstream)?;
                    }
                    if wants(updates) {
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
                        if wants(input) {
                            let grad = self.shrink(upstream, bounds)?;
                            self.accumulate(&mut grads, input, grad)?;
                        }
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
                    if wants(lhs) {
                        let lhs_grad = self.matmul_grad(upstream, lhs, rhs, true)?;
                        self.accumulate(&mut grads, lhs, lhs_grad)?;
                    }
                    if wants(rhs) {
                        let rhs_grad = self.matmul_grad(upstream, lhs, rhs, false)?;
                        self.accumulate(&mut grads, rhs, rhs_grad)?;
                    }
                }
                Op::Einsum { inputs, plan, .. } => {
                    for (target, input) in inputs.iter().enumerate() {
                        if wants(*input) && self.node(*input)?.dtype.is_float() {
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
                    if wants(first_upstream) {
                        let upstream_grad = self.einsum_grad_vjp(
                            upstream,
                            first_upstream,
                            &inputs,
                            plan.clone(),
                            target,
                            inputs.len(),
                        )?;
                        self.accumulate(&mut grads, first_upstream, upstream_grad)?;
                    }
                    for (wrt, input) in inputs.iter().copied().enumerate() {
                        if wants(input) && wrt != target && self.node(input)?.dtype.is_float() {
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
                    if wants(first_upstream) {
                        let upstream_grad = self.matmul_grad_vjp(
                            upstream,
                            first_upstream,
                            lhs,
                            rhs,
                            lhs_gradient,
                            0,
                        )?;
                        self.accumulate(&mut grads, first_upstream, upstream_grad)?;
                    }
                    let factor = if lhs_gradient { rhs } else { lhs };
                    if wants(factor) && self.node(factor)?.dtype.is_float() {
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
                Op::Conv2dGradVjp { .. }
                | Op::ConvTranspose2dGradVjp { .. }
                | Op::ScatterPositionsVjp { .. }
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
                    if wants(input) {
                        let input_grad =
                            self.conv2d_grad(upstream, input, weight, bias, options, 0)?;
                        self.accumulate(&mut grads, input, input_grad)?;
                    }
                    if wants(weight) {
                        let weight_grad =
                            self.conv2d_grad(upstream, input, weight, bias, options, 1)?;
                        self.accumulate(&mut grads, weight, weight_grad)?;
                    }
                    if let Some(bias) = bias.filter(|bias| wants(*bias)) {
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
                        if wants(node)
                            && self.node(node)?.dtype.is_float()
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
                    if wants(input) {
                        let input_grad =
                            self.conv_transpose2d_grad(upstream, input, weight, bias, options, 0)?;
                        self.accumulate(&mut grads, input, input_grad)?;
                    }
                    if wants(weight) {
                        let weight_grad =
                            self.conv_transpose2d_grad(upstream, input, weight, bias, options, 1)?;
                        self.accumulate(&mut grads, weight, weight_grad)?;
                    }
                    if let Some(b) = bias.filter(|bias| wants(*bias)) {
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
                        if wants(node)
                            && self.node(node)?.dtype.is_float()
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
                    let upstream_node = self.node(upstream)?;
                    let zeros = filled(upstream_node.shape.clone(), 0.0, upstream_node.dtype)?;
                    let zeros = self.constant(zeros);
                    if wants(on_true) {
                        let true_grad = self.select(condition, upstream, zeros)?;
                        let true_grad =
                            self.unbroadcast(true_grad, self.node(on_true)?.shape.clone())?;
                        self.accumulate(&mut grads, on_true, true_grad)?;
                    }
                    if wants(on_false) {
                        let false_grad = self.select(condition, zeros, upstream)?;
                        let false_grad =
                            self.unbroadcast(false_grad, self.node(on_false)?.shape.clone())?;
                        self.accumulate(&mut grads, on_false, false_grad)?;
                    }
                }
            }
        }
        Ok(grads)
    }

    fn validate_reverse_path(&self, traversal: &ReverseTraversal) -> Result<()> {
        for node in traversal.active_nodes() {
            if let Op::PrefixScan { input, kind, .. } = &self.node(node)?.op {
                match kind {
                    crate::PrefixScanKind::Sum
                        if !prefix_scan_gradient_dtype_supported(self.node(*input)?.dtype) =>
                    {
                        return Err(Error::NonDifferentiableIndexing(
                            "cumsum gradients require floating input",
                        ));
                    }
                    crate::PrefixScanKind::Product
                        if !prefix_scan_gradient_dtype_supported(self.node(*input)?.dtype) =>
                    {
                        return Err(Error::NonDifferentiableIndexing(
                            "cumprod gradients require floating input",
                        ));
                    }
                    crate::PrefixScanKind::Max | crate::PrefixScanKind::Min
                        if !prefix_scan_gradient_dtype_supported(self.node(*input)?.dtype) =>
                    {
                        return Err(Error::NonDifferentiableIndexing(
                            "cumulative extrema gradients require floating input",
                        ));
                    }
                    crate::PrefixScanKind::Sum
                    | crate::PrefixScanKind::Product
                    | crate::PrefixScanKind::Max
                    | crate::PrefixScanKind::Min => {}
                }
            }
        }
        Ok(())
    }

    pub(crate) fn unbroadcast(&mut self, gradient: NodeId, shape: Shape) -> Result<NodeId> {
        let (source, dtype) = {
            let gradient = self.node(gradient)?;
            (gradient.shape.clone(), gradient.dtype)
        };
        if source == shape {
            Ok(gradient)
        } else {
            // tinygrad reduces every shaped reverse edge by casting to
            // `sum_acc_dtype`, summing the canonical broadcast axes, reshaping
            // to the source descriptor, and casting once back to the incoming
            // gradient storage. Keep raw Graph::sum_to available for its
            // independent same-storage contract, but do not publish that
            // CPU-only operation from ordinary reverse mode.
            if shape.broadcast_with(&source).as_ref() != Ok(&source) {
                return Err(Error::InvalidSumTo {
                    from: source,
                    to: shape,
                });
            }
            let padding =
                source
                    .rank()
                    .checked_sub(shape.rank())
                    .ok_or_else(|| Error::InvalidSumTo {
                        from: source.clone(),
                        to: shape.clone(),
                    })?;
            let mut axes = (0..padding)
                .map(|axis| isize::try_from(axis).map_err(|_| Error::ShapeOverflow(source.clone())))
                .collect::<Result<Vec<_>>>()?;
            axes.extend(
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .filter_map(|(target_axis, target_dim)| {
                        let source_axis = target_axis + padding;
                        (*target_dim == 1 && source.dims()[source_axis] != 1).then_some(source_axis)
                    })
                    .map(|axis| {
                        isize::try_from(axis).map_err(|_| Error::ShapeOverflow(source.clone()))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            let accumulator = dtype.sum_accumulator_dtype();
            let reduced = self.reduce_with_dtypes(
                gradient,
                crate::ReduceKind::Sum,
                Some(axes),
                true,
                ReductionDType::new(accumulator, accumulator),
            )?;
            let reshaped = self.reshape(reduced, shape)?;
            if accumulator == dtype {
                Ok(reshaped)
            } else {
                self.cast(reshaped, dtype)
            }
        }
    }

    fn cumsum_vjp(&mut self, upstream: NodeId, axis: usize) -> Result<NodeId> {
        let reversed = self.reverse_axis(upstream, axis)?;
        let summed = self.cumsum(reversed, axis as isize)?;
        self.reverse_axis(summed, axis)
    }

    fn cumprod_vjp(&mut self, upstream: NodeId, input: NodeId, axis: usize) -> Result<NodeId> {
        let dtype = self.node(input)?.dtype;
        let zero_value = self.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), dtype));
        let one_value = self.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), dtype));
        let zero_mask = self.eq(input, zero_value)?;
        let safe_input = self.select(zero_mask, one_value, input)?;
        let zero_mask_i32 = self.cast(zero_mask, DType::I32)?;
        let zero_count = self.cumsum(zero_mask_i32, axis as isize)?;
        let safe_product = self.cumprod(safe_input, axis as isize)?;
        let weighted = self.mul(upstream, safe_product)?;
        let count_zero = self.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::I32));
        let count_one = self.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
        let zero = self.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), dtype));
        let no_zero = self.eq(zero_count, count_zero)?;
        let one_zero = self.eq(zero_count, count_one)?;
        let ordinary = self.select(no_zero, weighted, zero)?;
        let zero_lane = self.select(one_zero, weighted, zero)?;
        let ordinary_sum = self.cumsum_vjp(ordinary, axis)?;
        let ordinary = self.div(ordinary_sum, safe_input)?;
        let zero_lane = self.cumsum_vjp(zero_lane, axis)?;
        self.select(zero_mask, zero_lane, ordinary)
    }

    fn cumulative_extrema_vjp(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        values: NodeId,
        axis: usize,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let shape = input_node.shape.clone();
        if shape.rank() == 0 || shape.numel()? == 0 {
            return Ok(upstream);
        }
        let extent = shape.dims()[axis];
        let end = i64::try_from(extent).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let mut order = (0..shape.rank())
            .filter(|candidate| *candidate != axis)
            .collect::<Vec<_>>();
        order.push(axis);
        let mut inverse = vec![0usize; shape.rank()];
        for (position, original) in order.iter().copied().enumerate() {
            inverse[original] = position;
        }

        let input = self.permute(input, order.clone())?;
        let values = self.permute(values, order.clone())?;
        let upstream = self.permute(upstream, order)?;
        let input = self.unsqueeze(input, -1)?;
        let values = self.unsqueeze(values, -2)?;
        let winners = self.eq(input, values)?;

        let coordinates = self.lazy_arange_default_int(0, end, 1)?;
        let rows = self.reshape(coordinates, Shape::from([extent, 1]))?;
        let columns = self.reshape(coordinates, Shape::from([1, extent]))?;
        let prefix = self.le(rows, columns)?;
        let winners = self.logical_and(winners, prefix)?;
        let upstream_dtype = self.node(upstream)?.dtype;
        let winners = self.cast(winners, upstream_dtype)?;
        let winner_axis =
            isize::try_from(shape.rank() - 1).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let counts = self.reduce_with_output_dtype(
            winners,
            crate::ReduceKind::Sum,
            Some(vec![winner_axis]),
            true,
            upstream_dtype,
        )?;
        let upstream = self.unsqueeze(upstream, -2)?;
        let contributions = self.mul(winners, upstream)?;
        let contributions = self.div(contributions, counts)?;
        let prefix_axis =
            isize::try_from(shape.rank()).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let gradient = self.reduce_with_output_dtype(
            contributions,
            crate::ReduceKind::Sum,
            Some(vec![prefix_axis]),
            false,
            upstream_dtype,
        )?;
        self.permute(gradient, inverse)
    }

    fn reverse_axis(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let slices: Vec<crate::Slice> = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(current, _)| crate::Slice {
                start: None,
                stop: None,
                step: if current == axis { -1 } else { 1 },
            })
            .collect();
        self.stride(input, slices)
    }

    fn sort_indices_sibling(
        &self,
        values: NodeId,
        input: NodeId,
        axis: usize,
        descending: bool,
        pair: u64,
    ) -> Result<NodeId> {
        let values_node = self.node(values)?;
        let matches = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| match &node.op {
                Op::Sort {
                    input: candidate_input,
                    axis: candidate_axis,
                    descending: candidate_descending,
                    pair: candidate_pair,
                    output: crate::SortOutput::Indices,
                } if *candidate_input == input
                    && *candidate_axis == axis
                    && *candidate_descending == descending
                    && *candidate_pair == pair
                    && node.shape == values_node.shape
                    && node.dtype == crate::DType::I32 =>
                {
                    Some(NodeId(index))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(Error::NonDifferentiableIndexing(
                "stable sort pair is missing or ambiguous",
            ));
        }
        Ok(matches[0])
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

fn filled(shape: Shape, value: f64, dtype: DType) -> Result<TensorData> {
    let elements = shape.numel()?;
    TensorData::from_scalars(
        shape,
        dtype,
        std::iter::repeat_n(Scalar::F(value), elements),
    )
}

fn prefix_scan_gradient_dtype_supported(dtype: DType) -> bool {
    dtype.is_float() && !dtype.is_float8()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, Scalar};
    use std::collections::HashMap;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    fn data_f64(shape: impl Into<Shape>, values: &[f64]) -> TensorData {
        TensorData::from_scalars(shape, DType::F64, values.iter().copied().map(Scalar::F)).unwrap()
    }

    fn typed_data(shape: impl Into<Shape>, dtype: DType, values: &[f64]) -> TensorData {
        TensorData::from_scalars(shape, dtype, values.iter().copied().map(Scalar::F)).unwrap()
    }

    #[test]
    fn stable_sort_values_vjp_uses_the_paired_stable_indices() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let (values, indices) = graph.sort(input, -1, true).unwrap();
        let weights = graph.constant(data([2, 3], &[1., 2., 4., 8., 16., 32.]));
        let weighted = graph.mul(values, weights).unwrap();
        let loss = graph.sum_all(weighted).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        assert_eq!(graph.dtype(gradient).unwrap(), DType::F32);
        assert!(matches!(
            graph.grad(indices, input),
            Err(Error::NonDifferentiableTarget(node)) if node == indices
        ));
        let inputs = HashMap::from([(
            "input".into(),
            data([2, 3], &[1., 1., f32::NAN, -0.0, 0.0, -0.0]),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs).unwrap(),
            data([2, 3], &[5., 2., 0., 8., 16., 32.])
        );
    }

    #[test]
    fn stable_sort_values_vjp_handles_ascending_reuse_scalar_and_empty_inputs() {
        let mut graph = Graph::new();
        let input = graph.input("input", [3]);
        let (values, _) = graph.sort(input, 0, false).unwrap();
        let weights = graph.constant(data([3], &[1., 2., 4.]));
        let weighted = graph.mul(values, weights).unwrap();
        let total = graph.add(weighted, values).unwrap();
        let loss = graph.sum_all(total).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    gradient,
                    &HashMap::from([("input".into(), data([3], &[2., 1., 1.]))])
                )
                .unwrap(),
            data([3], &[5., 2., 3.])
        );

        let mut scalar = Graph::new();
        let source = scalar.input("scalar", []);
        let nodes = scalar.node_count();
        assert!(scalar.sort(source, -1, false).is_err());
        assert_eq!(scalar.node_count(), nodes);

        let mut empty = Graph::new();
        let source = empty.input("empty", [0]);
        let (values, indices) = empty.sort(source, 0, false).unwrap();
        let loss = empty.sum_all(values).unwrap();
        let gradient = empty.grad(loss, source).unwrap();
        assert_eq!(empty.shape(gradient).unwrap(), &Shape::new([0]));
        assert!(!empty.requires_grad(indices).unwrap());
    }

    #[test]
    fn topk_values_vjp_uses_the_stable_sort_indices_and_indices_remain_closed() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let (values, indices) = graph.topk(input, 2, -1, true, true).unwrap();
        let weights = graph.constant(data([2, 2], &[1., 2., 4., 8.]));
        let weighted = graph.mul(values, weights).unwrap();
        let loss = graph.sum_all(weighted).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        assert_eq!(graph.dtype(gradient).unwrap(), DType::F32);
        assert!(matches!(
            graph.grad(indices, input),
            Err(Error::NonDifferentiableTarget(node)) if node == indices
        ));
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    gradient,
                    &HashMap::from([(
                        "input".into(),
                        data([2, 3], &[1., 1., f32::NAN, -0.0, 0.0, -0.0])
                    )]),
                )
                .unwrap(),
            data([2, 3], &[1., 2., 0., 4., 8., 0.])
        );
    }

    #[test]
    fn floating_cast_vjp_restores_source_dtype_and_accumulates() {
        let mut graph = Graph::new();
        let narrow_source = graph.input("narrow_source", [2]);
        let wide_left = graph.cast(narrow_source, DType::F64).unwrap();
        let wide_right = graph.cast(narrow_source, DType::F64).unwrap();
        let wide_sum = graph.add(wide_left, wide_right).unwrap();
        let wide_loss = graph.sum_all(wide_sum).unwrap();
        let narrow_grad = graph.grad(wide_loss, narrow_source).unwrap();
        assert_eq!(graph.dtype(narrow_grad).unwrap(), DType::F32);

        let wide_source = graph.input_dtype("wide_source", [1, 2], DType::F64);
        let narrowed = graph.cast(wide_source, DType::F32).unwrap();
        let expanded = graph.expand(narrowed, [3, 2]).unwrap();
        let narrow_loss = graph.sum_all(expanded).unwrap();
        let wide_grad = graph.grad(narrow_loss, wide_source).unwrap();
        assert_eq!(graph.dtype(wide_grad).unwrap(), DType::F64);

        let inputs = HashMap::from([
            ("narrow_source".into(), data([2], &[1.0, -2.0])),
            ("wide_source".into(), data_f64([1, 2], &[1.5, -2.5])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, narrow_grad, &inputs).unwrap(),
            data([2], &[2.0, 2.0])
        );
        assert_eq!(
            CpuBackend.execute(&graph, wide_grad, &inputs).unwrap(),
            data_f64([1, 2], &[3.0, 3.0])
        );
    }

    #[test]
    fn typed_unbroadcast_uses_canonical_axes_and_never_emits_sum_to() {
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let target = graph.input_dtype("target", [1, 3], dtype);
            let other = graph.input_dtype("other", [2, 4, 3], dtype);
            let output = graph.add(target, other).unwrap();
            let seed = graph
                .lazy_full_with_dtype([2, 4, 3], Scalar::F(1.0), dtype)
                .unwrap();
            let first_vjp_node = graph.node_count();
            let gradient = graph.grad_with(output, target, Some(seed), true).unwrap();

            assert_eq!(graph.shape(gradient).unwrap(), &Shape::from([1, 3]));
            assert_eq!(graph.dtype(gradient).unwrap(), dtype);
            assert!(
                (first_vjp_node..graph.node_count())
                    .map(NodeId)
                    .all(|node| !matches!(graph.op(node).unwrap(), Op::SumTo { .. }))
            );

            let reductions = (first_vjp_node..graph.node_count())
                .map(NodeId)
                .filter(|node| {
                    matches!(
                        graph.op(*node).unwrap(),
                        Op::Reduce {
                            kind: crate::ReduceKind::Sum,
                            axes,
                            keepdim: true,
                            ..
                        } if axes.as_slice() == [0, 1]
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(reductions.len(), 1);
            let accumulator = dtype.sum_accumulator_dtype();
            assert_eq!(graph.dtype(reductions[0]).unwrap(), accumulator);
            let Op::Reduce {
                accumulator: committed,
                ..
            } = graph.op(reductions[0]).unwrap()
            else {
                unreachable!()
            };
            assert_eq!(*committed, accumulator);
            if matches!(dtype, DType::F16 | DType::BF16) {
                assert!(
                    matches!(graph.op(gradient).unwrap(), Op::Cast { dtype: actual, .. } if *actual == dtype)
                );
                assert!(
                    (first_vjp_node..graph.node_count())
                        .map(NodeId)
                        .any(|node| matches!(
                            graph.op(node).unwrap(),
                            Op::Cast {
                                dtype: DType::F32,
                                ..
                            }
                        ))
                );
            }

            let actual = CpuBackend
                .execute(&graph, gradient, &HashMap::new())
                .unwrap();
            assert_eq!(actual, typed_data([1, 3], dtype, &[8.0, 8.0, 8.0]));
        }
    }

    #[test]
    fn narrow_unbroadcast_accumulates_in_f32_then_narrows_once() {
        for (dtype, magnitude) in [(DType::F16, 2048.0), (DType::BF16, 256.0)] {
            let mut graph = Graph::new();
            let target = graph.input_dtype("target", [1], dtype);
            let other = graph.input_dtype("other", [3], dtype);
            let output = graph.add(target, other).unwrap();
            let seed = graph.constant(typed_data([3], dtype, &[magnitude, 1.0, -magnitude]));
            let gradient = graph.grad_with(output, target, Some(seed), true).unwrap();

            // Per-step narrow accumulation would lose the middle one. The
            // source contract widens the complete sum and narrows only here.
            assert_eq!(
                CpuBackend
                    .execute(&graph, gradient, &HashMap::new())
                    .unwrap(),
                typed_data([1], dtype, &[1.0])
            );
            let Op::Cast {
                input: reshaped,
                dtype: final_dtype,
            } = graph.op(gradient).unwrap()
            else {
                panic!("narrow unbroadcast must end in one storage cast")
            };
            assert_eq!(*final_dtype, dtype);
            let reduced = match graph.op(*reshaped).unwrap() {
                Op::Reshape { input, .. } => *input,
                // A keepdim reduction already has the target descriptor for
                // this rank-one fixture, so the checked reshape is identity.
                Op::Reduce { .. } => *reshaped,
                _ => panic!("narrow unbroadcast must retain its widened reduction"),
            };
            assert!(matches!(
                graph.op(reduced).unwrap(),
                Op::Reduce {
                    kind: crate::ReduceKind::Sum,
                    accumulator: DType::F32,
                    ..
                }
            ));
            assert_eq!(graph.dtype(reduced).unwrap(), DType::F32);
        }
    }

    #[test]
    fn raw_sum_to_retains_its_same_storage_compatibility_contract() {
        let mut graph = Graph::new();
        let input = graph.constant(typed_data([3], DType::F16, &[2048.0, 1.0, -2048.0]));
        let reduced = graph.sum_to(input, [1]).unwrap();
        assert!(matches!(graph.op(reduced).unwrap(), Op::SumTo { .. }));
        assert_eq!(graph.dtype(reduced).unwrap(), DType::F16);
        assert_eq!(
            CpuBackend
                .execute(&graph, reduced, &HashMap::new())
                .unwrap(),
            typed_data([1], DType::F16, &[1.0])
        );
    }

    #[test]
    fn typed_unbroadcast_preserves_the_incoming_cotangent_storage() {
        let mut identity_graph = Graph::new();
        let identity_target = identity_graph.input_dtype("target", [3], DType::F16);
        let identity_other = identity_graph.input_dtype("other", [3], DType::F16);
        let identity_output = identity_graph.add(identity_target, identity_other).unwrap();
        let identity_seed = identity_graph.input_dtype("seed", [3], DType::I32);
        assert_eq!(
            identity_graph
                .gradient(identity_output, &[identity_target], Some(identity_seed))
                .unwrap(),
            vec![identity_seed]
        );

        let before = identity_graph.node_count();
        assert!(matches!(
            identity_graph.grad_with(
                identity_output,
                identity_target,
                Some(identity_seed),
                true
            ),
            Err(Error::NonDifferentiableTarget(node)) if node == identity_seed
        ));
        assert_eq!(identity_graph.node_count(), before);

        let mut graph = Graph::new();
        let target = graph.input_dtype("target", [1], DType::F16);
        let other = graph.input_dtype("other", [2, 3], DType::F16);
        let output = graph.add(target, other).unwrap();
        let seed = graph.constant(typed_data(
            [2, 3],
            DType::I8,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let gradient = graph.gradient(output, &[target], Some(seed)).unwrap()[0];
        assert_eq!(graph.dtype(gradient).unwrap(), DType::I8);
        let Op::Cast {
            input: reshaped,
            dtype: DType::I8,
        } = graph.op(gradient).unwrap()
        else {
            panic!("I8 unbroadcast must narrow its I32 accumulation once")
        };
        let Op::Reshape { input: reduced, .. } = graph.op(*reshaped).unwrap() else {
            panic!("I8 unbroadcast must reshape its I32 reduction")
        };
        assert_eq!(graph.dtype(*reduced).unwrap(), DType::I32);
        assert!(matches!(
            graph.op(*reduced).unwrap(),
            Op::Reduce {
                kind: crate::ReduceKind::Sum,
                accumulator: DType::I32,
                ..
            }
        ));
        assert_eq!(
            CpuBackend
                .execute(&graph, gradient, &HashMap::new())
                .unwrap(),
            typed_data([1], DType::I8, &[21.0])
        );
    }

    #[test]
    fn typed_unbroadcast_preserves_scalar_zero_duplicate_and_higher_order_edges() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let scalar_squared = graph.square(scalar).unwrap();
        let scalar_gradient = graph.grad(scalar_squared, scalar).unwrap();
        assert_eq!(graph.shape(scalar_gradient).unwrap(), &Shape::from([]));

        let empty = graph.input("empty", [1, 0]);
        let expanded_empty = graph.expand(empty, [2, 0]).unwrap();
        let empty_loss = graph.sum_all(expanded_empty).unwrap();
        let empty_gradient = graph.grad(empty_loss, empty).unwrap();
        assert_eq!(graph.shape(empty_gradient).unwrap(), &Shape::from([1, 0]));
        assert_eq!(
            CpuBackend
                .execute(&graph, empty_gradient, &HashMap::new())
                .unwrap(),
            data([1, 0], &[])
        );

        let value = graph.input("value", [1]);
        let expanded = graph.expand(value, [3]).unwrap();
        let repeated = graph.add(expanded, expanded).unwrap();
        let squared = graph.square(repeated).unwrap();
        let loss = graph.sum_all(squared).unwrap();
        let first = graph.grad(loss, value).unwrap();
        let first_sum = graph.sum_all(first).unwrap();
        let second = graph.grad(first_sum, value).unwrap();
        // The exact second derivative is the constant 24, so its value edge
        // no longer needs the original input even though it remains fully
        // graph-constructible and executable.
        assert!(!graph.backward_slice_contains(second, value).unwrap());
        let bindings = HashMap::from([
            ("scalar".into(), data([], &[2.0])),
            ("empty".into(), data([1, 0], &[])),
            ("value".into(), data([1], &[2.0])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, first, &bindings).unwrap(),
            data([1], &[48.0])
        );
        assert_eq!(
            CpuBackend.execute(&graph, second, &bindings).unwrap(),
            data([1], &[24.0])
        );
        assert!(
            (0..graph.node_count())
                .map(NodeId)
                .all(|node| !matches!(graph.op(node).unwrap(), Op::SumTo { .. }))
        );
    }

    #[test]
    fn float8_unbroadcast_uses_f32_accumulation_for_every_format() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let mut graph = Graph::new();
            let target = graph.input_dtype("target", [1], dtype);
            let other = graph.input_dtype("other", [2, 2], dtype);
            let output = graph.add(target, other).unwrap();
            let seed = graph.constant(typed_data([2, 2], dtype, &[1.0, 2.0, 3.0, 4.0]));
            let first_vjp_node = graph.node_count();
            let gradient = graph.grad_with(output, target, Some(seed), true).unwrap();

            assert_eq!(graph.dtype(gradient).unwrap(), dtype);
            let Op::Cast {
                input: reshaped,
                dtype: final_dtype,
            } = graph.op(gradient).unwrap()
            else {
                panic!("Float8 unbroadcast must encode its F32 sum once")
            };
            assert_eq!(*final_dtype, dtype);
            let Op::Reshape { input: reduced, .. } = graph.op(*reshaped).unwrap() else {
                panic!("Float8 unbroadcast must retain the target descriptor")
            };
            assert!(matches!(
                graph.op(*reduced).unwrap(),
                Op::Reduce {
                    kind: crate::ReduceKind::Sum,
                    accumulator: DType::F32,
                    ..
                }
            ));
            assert!(
                (first_vjp_node..graph.node_count())
                    .map(NodeId)
                    .any(|node| matches!(
                        graph.op(node).unwrap(),
                        Op::Cast {
                            dtype: DType::F32,
                            ..
                        }
                    ))
            );
            assert_eq!(
                CpuBackend
                    .execute(&graph, gradient, &HashMap::new())
                    .unwrap(),
                typed_data([1], dtype, &[10.0])
            );
        }
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
        assert!((0..graph.node_count()).map(NodeId).all(|node| !matches!(
            graph.op(node).unwrap(),
            Op::MatmulGrad { .. } | Op::MatmulGradVjp { .. }
        )));
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
    fn generalized_matmul_vjp_handles_broadcast_batches_and_duplicate_roles() {
        let mut vectors = Graph::new();
        let left = vectors.input("left", [3]);
        let right = vectors.input("right", [3]);
        let dot = vectors.matmul(left, right).unwrap();
        let gradients = vectors.gradient_default(dot, &[left, right]).unwrap();
        let bindings = HashMap::from([
            ("left".into(), data([3], &[2., 3., 5.])),
            ("right".into(), data([3], &[7., 11., 13.])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&vectors, gradients[0], &bindings)
                .unwrap(),
            data([3], &[7., 11., 13.])
        );
        assert_eq!(
            CpuBackend
                .execute(&vectors, gradients[1], &bindings)
                .unwrap(),
            data([3], &[2., 3., 5.])
        );

        let mut matrix_vector = Graph::new();
        let matrix = matrix_vector.input("matrix", [2, 3]);
        let vector = matrix_vector.input("vector", [3]);
        let product = matrix_vector.matmul(matrix, vector).unwrap();
        let loss = matrix_vector.sum_all(product).unwrap();
        let gradients = matrix_vector
            .gradient_default(loss, &[matrix, vector])
            .unwrap();
        let bindings = HashMap::from([
            ("matrix".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("vector".into(), data([3], &[7., 11., 13.])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&matrix_vector, gradients[0], &bindings)
                .unwrap(),
            data([2, 3], &[7., 11., 13., 7., 11., 13.])
        );
        assert_eq!(
            CpuBackend
                .execute(&matrix_vector, gradients[1], &bindings)
                .unwrap(),
            data([3], &[5., 7., 9.])
        );

        let mut graph = Graph::new();
        let lhs = graph.input("lhs", [2, 1, 2, 3]);
        let rhs = graph.input("rhs", [1, 2, 3, 2]);
        let product = graph.matmul(lhs, rhs).unwrap();
        let loss = graph.sum_all(product).unwrap();
        let gradients = graph.gradient_default(loss, &[lhs, rhs]).unwrap();
        assert_eq!(
            graph.shape(gradients[0]).unwrap(),
            &Shape::from([2, 1, 2, 3])
        );
        assert_eq!(
            graph.shape(gradients[1]).unwrap(),
            &Shape::from([1, 2, 3, 2])
        );
        assert!((0..graph.node_count()).map(NodeId).all(|node| !matches!(
            graph.op(node).unwrap(),
            Op::MatmulGrad { .. } | Op::MatmulGradVjp { .. }
        )));

        let bindings = HashMap::from([
            (
                "lhs".into(),
                data(
                    [2, 1, 2, 3],
                    &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                ),
            ),
            (
                "rhs".into(),
                data(
                    [1, 2, 3, 2],
                    &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                ),
            ),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gradients[0], &bindings).unwrap(),
            data(
                [2, 1, 2, 3],
                &[18., 26., 34., 18., 26., 34., 18., 26., 34., 18., 26., 34.],
            )
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradients[1], &bindings).unwrap(),
            data(
                [1, 2, 3, 2],
                &[22., 22., 26., 26., 30., 30., 22., 22., 26., 26., 30., 30.],
            )
        );

        let mut shared = Graph::new();
        let x = shared.input("x", [2, 2]);
        let square = shared.matmul(x, x).unwrap();
        let loss = shared.sum_all(square).unwrap();
        let gradient = shared.gradient_default(loss, &[x, x]).unwrap();
        assert_eq!(gradient[0], gradient[1], "duplicate targets must alias");
        assert_eq!(
            CpuBackend
                .execute(
                    &shared,
                    gradient[0],
                    &HashMap::from([("x".into(), data([2, 2], &[1., 2., 3., 4.]))]),
                )
                .unwrap(),
            data([2, 2], &[7., 11., 9., 13.])
        );
    }

    #[test]
    fn generalized_matmul_vjp_is_target_sliced_higher_order_and_zero_safe() {
        let mut graph = Graph::new();
        let vector = graph.input_dtype("vector", [3], DType::F64);
        let matrix = graph.input_dtype("matrix", [3, 2], DType::F64);
        let product = graph.matmul(vector, matrix).unwrap();
        let loss = graph.sum_all(product).unwrap();
        let before = graph.node_count();
        let vector_gradient = graph.gradient_default(loss, &[vector]).unwrap()[0];
        assert_eq!(
            (before..graph.node_count())
                .map(NodeId)
                .filter(|node| matches!(
                    graph.op(*node).unwrap(),
                    Op::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ))
                .count(),
            1,
            "only the requested local Matmul edge is constructed"
        );
        let first_sum = graph.sum_all(vector_gradient).unwrap();
        let second = graph.grad(first_sum, matrix).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    second,
                    &HashMap::from([
                        ("vector".into(), data_f64([3], &[2., 3., 5.])),
                        ("matrix".into(), data_f64([3, 2], &[1., 2., 3., 4., 5., 6.]),),
                    ]),
                )
                .unwrap(),
            data_f64([3, 2], &[1., 1., 1., 1., 1., 1.])
        );

        let mut zero = Graph::new();
        let lhs = zero.input("lhs", [2, 0]);
        let rhs = zero.input("rhs", [0, 3]);
        let output = zero.matmul(lhs, rhs).unwrap();
        let loss = zero.sum_all(output).unwrap();
        let gradients = zero.gradient_default(loss, &[lhs, rhs]).unwrap();
        let bindings = HashMap::from([
            ("lhs".into(), data([2, 0], &[])),
            ("rhs".into(), data([0, 3], &[])),
        ]);
        assert_eq!(
            CpuBackend.execute(&zero, gradients[0], &bindings).unwrap(),
            data([2, 0], &[])
        );
        assert_eq!(
            CpuBackend.execute(&zero, gradients[1], &bindings).unwrap(),
            data([0, 3], &[])
        );
    }

    #[test]
    fn generalized_matmul_vjp_preflights_atomically_and_preserves_narrow_fallback() {
        let mut malformed = Graph::new();
        let lhs = malformed.input("lhs", [2, 3]);
        let rhs = malformed.input("rhs", [3, 4]);
        let upstream = malformed.input("upstream", [2, 3]);
        let before = malformed.node_count();
        assert!(matches!(
            malformed.matmul_grad(upstream, lhs, rhs, true),
            Err(Error::ShapeMismatch {
                op: "matmul gradient",
                ..
            })
        ));
        assert_eq!(malformed.node_count(), before);

        let mut overflow = Graph::new();
        let lhs = overflow.input_dtype("lhs", [usize::MAX, 1], DType::F64);
        let rhs = overflow.input_dtype("rhs", [1, 2], DType::F64);
        let upstream = overflow.input_dtype("upstream", [usize::MAX, 2], DType::F64);
        let before = overflow.node_count();
        assert!(matches!(
            overflow.matmul_grad(upstream, lhs, rhs, true),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);

        let mut narrow = Graph::new();
        let lhs = narrow.input_dtype("lhs", [2, 2], DType::F16);
        let rhs = narrow.input_dtype("rhs", [2, 2], DType::F16);
        let upstream = narrow.input_dtype("upstream", [2, 2], DType::F16);
        let gradient = narrow.matmul_grad(upstream, lhs, rhs, true).unwrap();
        assert!(matches!(
            narrow.op(gradient).unwrap(),
            Op::MatmulGrad { .. }
        ));

        let mut mixed = Graph::new();
        let lhs = mixed.input_dtype("lhs", [2, 2], DType::F32);
        let rhs = mixed.input_dtype("rhs", [2, 2], DType::F64);
        let upstream = mixed.input_dtype("upstream", [2, 2], DType::F64);
        let lhs_gradient = mixed.matmul_grad(upstream, lhs, rhs, true).unwrap();
        let rhs_gradient = mixed.matmul_grad(upstream, lhs, rhs, false).unwrap();
        assert!(matches!(
            mixed.op(lhs_gradient).unwrap(),
            Op::MatmulGrad {
                lhs_gradient: true,
                ..
            }
        ));
        assert!(matches!(
            mixed.op(rhs_gradient).unwrap(),
            Op::MatmulGrad {
                lhs_gradient: false,
                ..
            }
        ));
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
        assert!(!graph.trace(gx).unwrap().to_string().contains("conv2d_grad"));
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
    fn source_gradient_batches_targets_preserves_aliases_and_zero_fallbacks() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let y = graph.input_dtype_requires_grad("y", [3], DType::F32, false);
        let unused = graph.input_dtype_requires_grad("unused", [3], DType::F32, false);
        let product = graph.mul(x, y).unwrap();
        let loss = graph.sum_all(product).unwrap();

        let gradients = graph.gradient_default(loss, &[x, y, x, unused]).unwrap();
        assert_eq!(gradients[0], gradients[2], "duplicate targets must alias");
        assert_eq!(graph.shape(gradients[0]).unwrap(), &Shape::new([3]));
        assert_eq!(graph.shape(gradients[1]).unwrap(), &Shape::new([3]));
        assert_eq!(graph.shape(gradients[3]).unwrap(), &Shape::new([3]));
        assert_eq!(graph.dtype(gradients[3]).unwrap(), DType::F32);
        assert!(matches!(graph.op(gradients[3]).unwrap(), Op::Expand { .. }));

        let bindings = HashMap::from([
            ("x".into(), data([3], &[2.0, 3.0, 5.0])),
            ("y".into(), data([3], &[7.0, 11.0, 13.0])),
            ("unused".into(), data([3], &[17.0, 19.0, 23.0])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gradients[0], &bindings).unwrap(),
            data([3], &[7.0, 11.0, 13.0])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradients[1], &bindings).unwrap(),
            data([3], &[2.0, 3.0, 5.0])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradients[3], &bindings).unwrap(),
            data([3], &[0.0, 0.0, 0.0])
        );

        // The source-style transform is independent of RustGrad's leaf
        // tracking bit, including for higher-order composition.
        let gx_sum = graph.sum_all(gradients[0]).unwrap();
        let d_gx_dy = graph.gradient_default(gx_sum, &[y]).unwrap()[0];
        assert_eq!(
            CpuBackend.execute(&graph, d_gx_dy, &bindings).unwrap(),
            data([3], &[1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn reverse_edge_projection_excludes_predicate_index_and_detach_edges() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let y = graph.input("y", [3]);
        let condition = graph.input_dtype("condition", [3], DType::Bool);
        let index = graph.input_dtype("index", [3], DType::I32);

        let selected = graph.select(condition, x, y).unwrap();
        assert_eq!(graph.op(selected).unwrap().backward_inputs(), vec![x, y]);

        let gathered = graph.gather(x, index, 0).unwrap();
        assert_eq!(graph.op(gathered).unwrap().backward_inputs(), vec![x]);

        let scattered = graph.scatter(x, index, y, 0).unwrap();
        assert_eq!(graph.op(scattered).unwrap().backward_inputs(), vec![x, y]);

        let compacted = graph
            .masked_select(x, condition, 3, Scalar::F(0.0))
            .unwrap();
        assert_eq!(graph.op(compacted).unwrap().backward_inputs(), vec![x]);

        let detached = graph.detach(x).unwrap();
        assert!(graph.op(detached).unwrap().backward_inputs().is_empty());

        let (_, cumulative_indices) = graph.cummax(x, 0).unwrap();
        assert!(
            graph
                .op(cumulative_indices)
                .unwrap()
                .backward_inputs()
                .is_empty()
        );
        let (_, sort_indices) = graph.sort(x, 0, false).unwrap();
        assert!(graph.op(sort_indices).unwrap().backward_inputs().is_empty());
    }

    #[test]
    fn grad_with_prunes_unrequested_failures_and_commits_atomically() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let y = graph.input("y", [3]);
        let index = graph.input_dtype("index", [3], DType::I32);

        let x_squared = graph.square(x).unwrap();
        let x_loss = graph.sum_all(x_squared).unwrap();
        let replacement = graph.scatter(y, index, y, 0).unwrap();
        let replacement_squared = graph.square(replacement).unwrap();
        let replacement_loss = graph.sum_all(replacement_squared).unwrap();
        let loss = graph.add(x_loss, replacement_loss).unwrap();

        // Like tinygrad's target-filtered `_deepwalk`, the reverse transform
        // never visits the unrelated replacement-scatter rule.
        let x_gradient = graph.no_grad(|candidate| {
            assert!(!candidate.grad_enabled());
            let gradient = candidate.grad(loss, x).unwrap();
            assert!(!candidate.grad_enabled());
            gradient
        });
        assert!(graph.grad_enabled());
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    x_gradient,
                    &HashMap::from([
                        ("x".into(), data([3], &[2.0, 3.0, 5.0])),
                        ("y".into(), data([3], &[7.0, 11.0, 13.0])),
                        (
                            "index".into(),
                            TensorData::from_scalars(
                                [3],
                                DType::I32,
                                [Scalar::I(0), Scalar::I(1), Scalar::I(2)],
                            )
                            .unwrap(),
                        ),
                    ]),
                )
                .unwrap(),
            data([3], &[4.0, 6.0, 10.0])
        );

        let source_gradient = graph.no_grad(|candidate| {
            assert!(!candidate.grad_enabled());
            let gradient = candidate.gradient_default(loss, &[x]).unwrap()[0];
            assert!(!candidate.grad_enabled());
            gradient
        });
        assert!(graph.grad_enabled());
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    source_gradient,
                    &HashMap::from([
                        ("x".into(), data([3], &[2.0, 3.0, 5.0])),
                        ("y".into(), data([3], &[7.0, 11.0, 13.0])),
                        (
                            "index".into(),
                            TensorData::from_scalars(
                                [3],
                                DType::I32,
                                [Scalar::I(0), Scalar::I(1), Scalar::I(2)],
                            )
                            .unwrap(),
                        ),
                    ]),
                )
                .unwrap(),
            data([3], &[4.0, 6.0, 10.0])
        );

        // The active replacement-scatter path first builds derivatives for
        // Add, Sum, and Square, then rejects at Scatter. None of those private
        // candidate nodes, nor the temporary create_graph mode, may publish.
        graph.no_grad(|candidate| {
            let before = candidate.node_count();
            assert!(!candidate.grad_enabled());
            assert!(matches!(
                candidate.grad_with(loss, y, None, false),
                Err(Error::NonDifferentiableIndexing("replacement scatter"))
            ));
            assert_eq!(candidate.node_count(), before);
            assert!(!candidate.grad_enabled());
        });
        assert!(graph.grad_enabled());

        graph.no_grad(|candidate| {
            let before = candidate.node_count();
            assert!(!candidate.grad_enabled());
            assert!(matches!(
                candidate.gradient_default(loss, &[y]),
                Err(Error::NonDifferentiableIndexing("replacement scatter"))
            ));
            assert_eq!(candidate.node_count(), before);
            assert!(!candidate.grad_enabled());
        });
        assert!(graph.grad_enabled());
    }

    #[test]
    fn target_slice_builds_only_requested_edges_and_preserves_duplicate_roles() {
        let mut graph = Graph::new();
        let condition = graph.input_dtype("condition", [2], DType::Bool);
        let x = graph.input("x", [2]);
        let y = graph.input("y", [2]);
        let selected = graph.select(condition, x, y).unwrap();
        let loss = graph.sum_all(selected).unwrap();
        let before = graph.node_count();
        graph.grad(loss, x).unwrap();
        assert_eq!(
            (before..graph.node_count())
                .map(NodeId)
                .filter(|node| matches!(graph.op(*node).unwrap(), Op::Select { .. }))
                .count(),
            1,
            "the unrequested false branch must not build a local derivative"
        );

        let lhs = graph.input("lhs", [2, 3]);
        let rhs = graph.input("rhs", [3, 2]);
        let product = graph.matmul(lhs, rhs).unwrap();
        let cotangent = graph.input("cotangent", [2, 2]);
        let before = graph.node_count();
        let lhs_gradient = graph
            .grad_with(product, lhs, Some(cotangent), true)
            .unwrap();
        // The requested lhs role is one reshape/transpose/Mul/typed-Sum
        // composition. It consumes the exact cotangent and rhs, but never the
        // differentiated lhs that only the unrequested rhs role would need.
        let Op::Reshape { input: reduced, .. } = graph.op(lhs_gradient).unwrap() else {
            panic!("the lhs gradient must restore the original lhs descriptor")
        };
        let Op::Reduce {
            input: local_product,
            kind: crate::ReduceKind::Sum,
            axes,
            keepdim: true,
            accumulator: DType::F32,
        } = graph.op(*reduced).unwrap()
        else {
            panic!("the broadcast batch axis must use one typed Sum")
        };
        assert_eq!(axes, &[1]);
        assert!(matches!(
            graph.op(*local_product).unwrap(),
            Op::Binary {
                op: BinaryOp::Mul,
                ..
            }
        ));
        assert_eq!(graph.dtype(*reduced).unwrap(), DType::F32);
        assert!(
            graph
                .backward_slice_contains(lhs_gradient, cotangent)
                .unwrap()
        );
        assert!(!graph.backward_slice_contains(lhs_gradient, lhs).unwrap());
        assert!(graph.backward_slice_contains(lhs_gradient, rhs).unwrap());
        assert_eq!(
            (before..graph.node_count())
                .map(NodeId)
                .filter(|node| matches!(
                    graph.op(*node).unwrap(),
                    Op::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ))
                .count(),
            1,
            "the unrequested rhs-gradient role must not be built"
        );
        assert!(
            (before..graph.node_count())
                .map(NodeId)
                .all(|node| !graph.backward_slice_contains(node, lhs).unwrap())
        );

        let repeated = graph.select(condition, x, x).unwrap();
        let repeated_loss = graph.sum_all(repeated).unwrap();
        let repeated_gradient = graph.grad(repeated_loss, x).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    repeated_gradient,
                    &HashMap::from([
                        (
                            "condition".into(),
                            TensorData::from_scalars(
                                [2],
                                DType::Bool,
                                [Scalar::Bool(true), Scalar::Bool(false)],
                            )
                            .unwrap()
                        ),
                        ("x".into(), data([2], &[2.0, 3.0])),
                    ]),
                )
                .unwrap(),
            data([2], &[1.0, 1.0])
        );
        assert!(matches!(
            graph.op(repeated_gradient).unwrap(),
            Op::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn checked_cast_edges_stop_at_nonfloating_storage() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2]);
        let integer = graph.cast(input, DType::I32).unwrap();
        let recast = graph.cast(integer, DType::F32).unwrap();
        let loss = graph.sum_all(recast).unwrap();

        assert!(!graph.backward_slice_contains(loss, input).unwrap());
        let before = graph.node_count();
        assert!(matches!(graph.grad(loss, input), Err(Error::NoGradient(node)) if node == input));
        assert_eq!(graph.node_count(), before);

        let source_gradient = graph.gradient_default(loss, &[input]).unwrap()[0];
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    source_gradient,
                    &HashMap::from([("input".into(), data([2], &[1.25, -3.5]))]),
                )
                .unwrap(),
            data([2], &[0.0, 0.0])
        );

        let widened = graph.cast(input, DType::F64).unwrap();
        let widened_loss = graph.sum_all(widened).unwrap();
        assert!(graph.backward_slice_contains(widened_loss, input).unwrap());
        let widened_gradient = graph.grad(widened_loss, input).unwrap();
        assert_eq!(graph.dtype(widened_gradient).unwrap(), DType::F32);
    }

    #[test]
    fn cumulative_extrema_value_vjp_splits_prefix_ties_and_preserves_axes() {
        let mut graph = Graph::new();
        let maximum_input = graph.input("maximum_input", [3]);
        let (maximum_values, maximum_indices) = graph.cummax(maximum_input, 0).unwrap();
        let maximum_seed = graph.constant(data([3], &[10.0, 20.0, 30.0]));
        let maximum_gradient = graph
            .grad_with(maximum_values, maximum_input, Some(maximum_seed), true)
            .unwrap();
        assert!(matches!(
            graph.grad(maximum_indices, maximum_input),
            Err(Error::NonDifferentiableTarget(node)) if node == maximum_indices
        ));

        let minimum_input = graph.input("minimum_input", [3]);
        let (minimum_values, _) = graph.cummin(minimum_input, 0).unwrap();
        let minimum_seed = graph.constant(data([3], &[10.0, 20.0, 30.0]));
        let minimum_gradient = graph
            .grad_with(minimum_values, minimum_input, Some(minimum_seed), true)
            .unwrap();

        let axis_input = graph.input("axis_input", [3, 2]);
        let (axis_values, _) = graph.cummax(axis_input, 0).unwrap();
        let axis_seed = graph.constant(data([3, 2], &[1.0; 6]));
        let axis_gradient = graph
            .grad_with(axis_values, axis_input, Some(axis_seed), true)
            .unwrap();

        let zero_input = graph.input("zero_input", [2]);
        let (zero_values, _) = graph.cummax(zero_input, 0).unwrap();
        let zero_seed = graph.constant(data([2], &[2.0, 4.0]));
        let zero_gradient = graph
            .grad_with(zero_values, zero_input, Some(zero_seed), true)
            .unwrap();

        let bindings = HashMap::from([
            ("maximum_input".into(), data([3], &[2.0, 2.0, 1.0])),
            ("minimum_input".into(), data([3], &[1.0, 1.0, 2.0])),
            (
                "axis_input".into(),
                data([3, 2], &[2.0, 1.0, 2.0, 3.0, 1.0, 3.0]),
            ),
            ("zero_input".into(), data([2], &[-0.0, 0.0])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, maximum_gradient, &bindings)
                .unwrap(),
            data([3], &[35.0, 25.0, 0.0])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, minimum_gradient, &bindings)
                .unwrap(),
            data([3], &[35.0, 25.0, 0.0])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, axis_gradient, &bindings)
                .unwrap(),
            data([3, 2], &[2.0, 1.0, 1.0, 1.5, 0.0, 0.5])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, zero_gradient, &bindings)
                .unwrap(),
            data([2], &[4.0, 2.0])
        );
    }

    #[test]
    fn source_gradient_returns_typed_zeros_below_every_nonvalue_edge() {
        let mut graph = Graph::new();
        let values = graph.input("values", [3]);
        let other = graph.input("other", [3]);
        let zero = graph.constant(TensorData::scalar(0.0));

        let predicate_source = graph.input("predicate_source", [3]);
        let predicate = graph.gt(predicate_source, zero).unwrap();
        let selected = graph.select(predicate, values, other).unwrap();
        let selected_loss = graph.sum_all(selected).unwrap();
        let predicate_gradient = graph
            .gradient_default(selected_loss, &[predicate_source])
            .unwrap()[0];

        let index_source = graph.input("index_source", [3]);
        let index = graph.cast(index_source, DType::I32).unwrap();
        let gathered = graph.gather(values, index, 0).unwrap();
        let base = graph.constant(data([3], &[0.0, 0.0, 0.0]));
        let scattered = graph.scatter_add(base, index, other, 0).unwrap();
        let indexed = graph.add(gathered, scattered).unwrap();
        let indexed_loss = graph.sum_all(indexed).unwrap();
        let index_gradient = graph
            .gradient_default(indexed_loss, &[index_source])
            .unwrap()[0];

        let mask_source = graph.input("mask_source", [3]);
        let mask = graph.gt(mask_source, zero).unwrap();
        let compacted = graph
            .masked_select(values, mask, 3, Scalar::F(0.0))
            .unwrap();
        let compacted_loss = graph.sum_all(compacted).unwrap();
        let mask_gradient = graph
            .gradient_default(compacted_loss, &[mask_source])
            .unwrap()[0];

        let detach_source = graph.input("detach_source", [3]);
        let detached = graph.detach(detach_source).unwrap();
        let detached_squared = graph.square(detached).unwrap();
        let detached_loss = graph.sum_all(detached_squared).unwrap();
        let detach_gradient = graph
            .gradient_default(detached_loss, &[detach_source])
            .unwrap()[0];

        let bindings = HashMap::from([
            ("values".into(), data([3], &[2.0, 3.0, 5.0])),
            ("other".into(), data([3], &[7.0, 11.0, 13.0])),
            ("predicate_source".into(), data([3], &[1.0, -1.0, 1.0])),
            ("index_source".into(), data([3], &[0.0, 1.0, 2.0])),
            ("mask_source".into(), data([3], &[1.0, -1.0, 1.0])),
            ("detach_source".into(), data([3], &[17.0, 19.0, 23.0])),
        ]);
        for gradient in [
            predicate_gradient,
            index_gradient,
            mask_gradient,
            detach_gradient,
        ] {
            assert_eq!(graph.dtype(gradient).unwrap(), DType::F32);
            assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([3]));
            assert_eq!(
                CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
                data([3], &[0.0, 0.0, 0.0])
            );
        }
    }

    #[test]
    fn cumulative_extrema_value_vjp_handles_scalar_empty_nan_and_higher_order() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let (scalar_values, _) = graph.cummax(scalar, 0).unwrap();
        let scalar_gradient = graph.grad(scalar_values, scalar).unwrap();

        let empty = graph.input("empty", [0, 2]);
        let (empty_values, _) = graph.cummin(empty, 0).unwrap();
        let empty_seed = graph.constant(data([0, 2], &[]));
        let empty_gradient = graph
            .grad_with(empty_values, empty, Some(empty_seed), true)
            .unwrap();

        let nan = graph.input("nan", [2]);
        let (nan_values, _) = graph.cummax(nan, 0).unwrap();
        let nan_seed = graph.constant(data([2], &[1.0, 1.0]));
        let nan_gradient = graph
            .grad_with(nan_values, nan, Some(nan_seed), true)
            .unwrap();

        let first_sum = graph.sum_all(nan_gradient).unwrap();
        let second = graph.gradient_default(first_sum, &[nan]).unwrap()[0];
        let bindings = HashMap::from([
            ("scalar".into(), data([], &[7.0])),
            ("empty".into(), data([0, 2], &[])),
            ("nan".into(), data([2], &[f32::NAN, 1.0])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, scalar_gradient, &bindings)
                .unwrap(),
            data([], &[1.0])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, empty_gradient, &bindings)
                .unwrap(),
            data([0, 2], &[])
        );
        let nan_gradient = CpuBackend.execute(&graph, nan_gradient, &bindings).unwrap();
        assert!(nan_gradient.values().iter().all(|value| value.is_nan()));
        assert_eq!(
            CpuBackend.execute(&graph, second, &bindings).unwrap(),
            data([2], &[0.0, 0.0])
        );

        for dtype in [DType::F16, DType::BF16, DType::F64] {
            let input = graph.input_dtype(format!("{dtype:?}_input"), [2], dtype);
            let (values, _) = graph.cummin(input, 0).unwrap();
            let seed = graph
                .lazy_full_with_dtype([2], Scalar::F(1.0), dtype)
                .unwrap();
            let first_vjp_node = graph.node_count();
            let gradient = graph.grad_with(values, input, Some(seed), true).unwrap();
            assert_eq!(graph.dtype(gradient).unwrap(), dtype);
            if matches!(dtype, DType::F16 | DType::BF16) {
                let sum_reductions = (first_vjp_node..graph.node_count())
                    .map(NodeId)
                    .filter(|node| {
                        matches!(
                            graph.op(*node).unwrap(),
                            Op::Reduce {
                                kind: crate::ReduceKind::Sum,
                                ..
                            }
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(sum_reductions.len(), 2);
                assert!(
                    sum_reductions
                        .iter()
                        .all(|node| graph.dtype(*node).unwrap() == dtype)
                );
                assert!(
                    (first_vjp_node..graph.node_count())
                        .map(NodeId)
                        .all(|node| graph.dtype(node).unwrap() != DType::F32)
                );
            }
        }

        let float8 = graph.input_dtype("float8", [2], DType::F8E4M3);
        let (float8_values, _) = graph.cummax(float8, 0).unwrap();
        let float8_seed = graph
            .lazy_full_with_dtype([2], Scalar::F(1.0), DType::F8E4M3)
            .unwrap();
        let before = graph.node_count();
        assert!(matches!(
            graph.gradient(float8_values, &[float8], Some(float8_seed)),
            Err(Error::NonDifferentiableIndexing(_))
        ));
        assert_eq!(graph.node_count(), before);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn source_cumulative_extrema_vjp_discards_late_descriptor_failure() {
        let mut graph = Graph::new();
        let extent = 1usize << 32;
        let input = graph.input("input", [extent]);
        let (values, _) = graph.cummax(input, 0).unwrap();
        let seed = graph
            .lazy_full_with_dtype([extent], Scalar::F(1.0), DType::F32)
            .unwrap();
        let before = graph.node_count();
        assert!(matches!(
            graph.gradient(values, &[input], Some(seed)),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn source_gradient_broadcasts_custom_seed_and_prunes_unrequested_paths() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let y = graph.input("y", [3]);
        let product = graph.mul(x, y).unwrap();
        let product_sum = graph.sum_all(product).unwrap();
        let unrelated = graph.input("unrelated", [3]);
        let (cumulative, cumulative_indices) = graph.cummax(unrelated, 0).unwrap();
        let cumulative_sum = graph.sum_all(cumulative).unwrap();
        let cumulative_indices = graph.cast(cumulative_indices, DType::F32).unwrap();
        let cumulative_indices_sum = graph.sum_all(cumulative_indices).unwrap();
        let cumulative_branch = graph.add(cumulative_sum, cumulative_indices_sum).unwrap();
        let loss = graph.add(product_sum, cumulative_branch).unwrap();
        let seed = graph.constant(data([1], &[3.0]));
        let indices_only_gradient = graph
            .gradient_default(cumulative_indices_sum, &[unrelated])
            .unwrap()[0];

        // `[1]` is the concrete public representation accepted by tinygrad's
        // scalar custom-gradient example. The established source-facing seed
        // contract reshapes its one element to the scalar loss descriptor.
        let gradient = graph.gradient(loss, &[x], Some(seed)).unwrap()[0];
        let bindings = HashMap::from([
            ("x".into(), data([3], &[2.0, 3.0, 5.0])),
            ("y".into(), data([3], &[7.0, 11.0, 13.0])),
            ("unrelated".into(), data([3], &[1.0, 4.0, 2.0])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            data([3], &[21.0, 33.0, 39.0])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, indices_only_gradient, &bindings)
                .unwrap(),
            data([3], &[0.0, 0.0, 0.0])
        );

        let unrelated_gradient = graph.gradient_default(loss, &[unrelated]).unwrap()[0];
        assert_eq!(
            CpuBackend
                .execute(&graph, unrelated_gradient, &bindings)
                .unwrap(),
            data([3], &[1.0, 2.0, 0.0])
        );
    }

    #[test]
    fn source_gradient_rejects_invalid_requests_before_publication() {
        let mut graph = Graph::new();
        let vector = graph.input("vector", [1]);
        let before = graph.node_count();
        assert!(matches!(
            graph.gradient_default(vector, &[vector]),
            Err(Error::NonScalarLoss(shape)) if shape == Shape::new([1])
        ));
        assert_eq!(graph.node_count(), before);

        let integer = graph.input_dtype("integer", [1], DType::I32);
        let before = graph.node_count();
        assert!(matches!(
            graph.gradient(vector, &[integer], Some(vector)),
            Err(Error::NonDifferentiableTarget(node)) if node == integer
        ));
        assert_eq!(graph.node_count(), before);

        let before = graph.node_count();
        assert!(matches!(
            graph.gradient(vector, &[NodeId(usize::MAX)], Some(vector)),
            Err(Error::UnknownNode(NodeId(usize::MAX)))
        ));
        assert_eq!(graph.node_count(), before);

        let bad_seed = graph.input("bad_seed", [2]);
        let before = graph.node_count();
        assert!(matches!(
            graph.gradient(vector, &[vector], Some(bad_seed)),
            Err(Error::GradientShape { .. })
        ));
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn source_gradient_identity_and_empty_target_requests_publish_exactly() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let explicit = graph.input("explicit", []);
        let before = graph.node_count();
        assert_eq!(
            graph.gradient(scalar, &[scalar], Some(explicit)).unwrap(),
            vec![explicit]
        );
        assert_eq!(graph.node_count(), before);
        assert_eq!(
            graph
                .grad_with(scalar, scalar, Some(explicit), false)
                .unwrap(),
            explicit
        );
        assert_eq!(graph.node_count(), before);

        let implicit = graph.gradient_default(scalar, &[scalar]).unwrap()[0];
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    implicit,
                    &HashMap::from([("scalar".into(), TensorData::scalar(7.0))]),
                )
                .unwrap(),
            TensorData::scalar(1.0)
        );

        let before = graph.node_count();
        assert!(graph.gradient_default(scalar, &[]).unwrap().is_empty());
        assert_eq!(graph.node_count(), before);

        let vector = graph.input("vector", [2]);
        let unused_bad_shape = graph.input("unused_bad_shape", [3]);
        let before = graph.node_count();
        assert!(
            graph
                .gradient(vector, &[], Some(unused_bad_shape))
                .unwrap()
                .is_empty()
        );
        assert_eq!(graph.node_count(), before);

        assert!(matches!(
            graph.gradient_default(vector, &[]),
            Err(Error::NonScalarLoss(shape)) if shape == Shape::new([2])
        ));
        assert_eq!(graph.node_count(), before);
        assert!(matches!(
            graph.gradient(vector, &[], Some(NodeId(usize::MAX))),
            Err(Error::UnknownNode(NodeId(usize::MAX)))
        ));
        assert_eq!(graph.node_count(), before);

        let float8_scalar = graph.input_dtype("float8_scalar", [], DType::F8E5M2);
        let before = graph.node_count();
        assert!(
            graph
                .gradient_default(float8_scalar, &[])
                .unwrap()
                .is_empty()
        );
        assert_eq!(graph.node_count(), before);

        let integer = graph.input_dtype("integer", [], DType::I32);
        let before = graph.node_count();
        assert!(matches!(
            graph.gradient_default(integer, &[]),
            Err(Error::NonDifferentiableTarget(node)) if node == integer
        ));
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn source_gradient_restores_equal_numel_and_broadcast_seed_normalization() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let output = graph.square(input).unwrap();
        let row_seed = graph.input("row_seed", [3]);
        let row_gradient = graph.gradient(output, &[input], Some(row_seed)).unwrap()[0];

        let scalar_seed = graph.input("scalar_seed", [1]);
        let scalar_gradient = graph.gradient(output, &[input], Some(scalar_seed)).unwrap()[0];

        let bindings = HashMap::from([
            (
                "input".into(),
                data([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            ("row_seed".into(), data([3], &[10.0, 20.0, 30.0])),
            ("scalar_seed".into(), data([1], &[2.0])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, row_gradient, &bindings).unwrap(),
            data([2, 3], &[20.0, 80.0, 180.0, 80.0, 200.0, 360.0])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, scalar_gradient, &bindings)
                .unwrap(),
            data([2, 3], &[4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
        );

        let flat_seed = graph.input("flat_seed", [6]);
        let flat_gradient = graph.gradient(output, &[input], Some(flat_seed)).unwrap()[0];
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    flat_gradient,
                    &HashMap::from([
                        (
                            "input".into(),
                            data([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                        ),
                        (
                            "flat_seed".into(),
                            data([6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                        ),
                    ]),
                )
                .unwrap(),
            data([2, 3], &[2.0, 8.0, 18.0, 32.0, 50.0, 72.0])
        );

        let incompatible = graph.input("incompatible", [2, 2]);
        let before = graph.node_count();
        assert!(matches!(
            graph.gradient(output, &[input], Some(incompatible)),
            Err(Error::GradientShape { output, upstream })
                if output == Shape::new([2, 3]) && upstream == Shape::new([2, 2])
        ));
        assert_eq!(graph.node_count(), before);

        // The legacy single-target API intentionally remains descriptor-exact.
        assert!(matches!(
            graph.grad_with(output, input, Some(row_seed), true),
            Err(Error::GradientShape { output, upstream })
                if output == Shape::new([2, 3]) && upstream == Shape::new([3])
        ));
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn reciprocal_vjp_multiplies_by_the_squared_result() {
        let mut graph = Graph::new();
        let input = graph.input("x", [3]);
        let reciprocal = graph.reciprocal(input).unwrap();
        let loss = graph.sum_all(reciprocal).unwrap();
        let gradient = graph.grad(loss, input).unwrap();

        let Op::Binary {
            op: BinaryOp::Mul,
            lhs: local,
            rhs: final_reciprocal,
        } = graph.op(gradient).unwrap()
        else {
            panic!("reciprocal gradient must end in the second source Mul");
        };
        assert_eq!(*final_reciprocal, reciprocal);
        let Op::Binary {
            op: BinaryOp::Mul,
            lhs: negated_upstream,
            rhs: first_reciprocal,
        } = graph.op(*local).unwrap()
        else {
            panic!("reciprocal gradient must multiply by the result twice");
        };
        assert_eq!(*first_reciprocal, reciprocal);
        assert!(matches!(
            graph.op(*negated_upstream).unwrap(),
            Op::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    gradient,
                    &HashMap::from([("x".into(), data([3], &[0.5, 1.0, 2.0]))]),
                )
                .unwrap(),
            data([3], &[-4.0, -1.0, -0.25]),
        );
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
        let epsilon = 1e-3;
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
    fn pow_vjp_uses_source_width_weak_constants_and_preflights() {
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let base = graph.input_dtype("base", [2, 1], dtype);
            let exponent = graph.input_dtype("exponent", [2], dtype);
            let output = graph.pow(base, exponent).unwrap();
            let loss = graph.sum_all(output).unwrap();
            let base_grad = graph.grad(loss, base).unwrap();
            let exponent_grad = graph.grad(loss, exponent).unwrap();
            assert_eq!(graph.shape(base_grad).unwrap(), &Shape::new([2, 1]));
            assert_eq!(graph.shape(exponent_grad).unwrap(), &Shape::new([2]));
            let scalar_dtypes = (0..graph.node_count())
                .filter_map(
                    |index| match &graph.node(NodeId::from_index(index)).unwrap().op {
                        Op::Constant(data) if data.shape().rank() == 0 => Some(data.dtype()),
                        _ => None,
                    },
                )
                .collect::<Vec<_>>();
            // The seed/reduction may introduce additional F32 constants, but
            // every Pow-local weak literal must exist at the source storage
            // width: lhs zero; rhs zero/one; result -inf/zero; and ln(2).
            assert!(
                scalar_dtypes
                    .iter()
                    .filter(|&&actual| actual == dtype)
                    .count()
                    >= 6
            );
        }

        let mut graph = Graph::new();
        let base = graph.input_dtype("base", [3], DType::F64);
        let exponent = graph.input_dtype("exponent", [3], DType::F64);
        let output = graph.pow(base, exponent).unwrap();
        let loss = graph.sum_all(output).unwrap();
        let exponent_grad = graph.grad(loss, exponent).unwrap();
        let values = CpuBackend
            .execute(
                &graph,
                exponent_grad,
                &HashMap::from([
                    (
                        "base".into(),
                        TensorData::from_scalars(
                            [3],
                            DType::F64,
                            [Scalar::F(0.0), Scalar::F(2.0), Scalar::F(f64::NAN)],
                        )
                        .unwrap(),
                    ),
                    (
                        "exponent".into(),
                        TensorData::from_scalars(
                            [3],
                            DType::F64,
                            [Scalar::F(-1.0), Scalar::F(0.0), Scalar::F(1.0)],
                        )
                        .unwrap(),
                    ),
                ]),
            )
            .unwrap();
        assert_eq!(values.scalar_at(0).as_f64(), f64::NEG_INFINITY);
        assert_eq!(values.scalar_at(1).as_f64(), 2.0f64.ln());
        assert!(values.scalar_at(2).as_f64().is_nan());

        let mut overflow = Graph::new();
        let lhs = overflow.input_dtype("lhs", [usize::MAX, 2], DType::F64);
        let rhs = overflow.input_dtype("rhs", [usize::MAX, 2], DType::F64);
        let output = overflow.pow(lhs, rhs).unwrap();
        let node_count = overflow.node_count();
        assert!(matches!(
            pow_vjp_plan(&overflow, output, lhs, rhs, output, [true, true]),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), node_count);
    }

    #[test]
    fn shared_pow_operand_retains_both_reverse_roles() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2]);
        let power = graph.pow(input, input).unwrap();
        let traversal = ReverseTraversal::new(&graph, power, &[input]).unwrap();
        assert_eq!(
            traversal.active_inputs[power.index()],
            vec![input, input],
            "duplicate operand roles are part of the checked frontier"
        );
        // Each role has an independently checked preflight, and the shared
        // operand requests both without collapsing duplicate edge roles.
        pow_vjp_plan(&graph, power, input, input, power, [true, false]).unwrap();
        pow_vjp_plan(&graph, power, input, input, power, [false, true]).unwrap();
        pow_vjp_plan(&graph, power, input, input, power, [true, true]).unwrap();

        let loss = graph.sum_all(power).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        assert!(matches!(
            graph.op(gradient).unwrap(),
            Op::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        let realized = CpuBackend
            .execute(
                &graph,
                gradient,
                &HashMap::from([("input".into(), data([2], &[2.0, 3.0]))]),
            )
            .unwrap();
        for (actual, expected) in realized
            .values()
            .iter()
            .zip([4.0 * (1.0 + 2.0f32.ln()), 27.0 * (1.0 + 3.0f32.ln())])
        {
            assert!((actual - expected).abs() < 1e-5);
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
            Err(Error::NonDifferentiableTarget(node)) if node == index
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
    fn signed_gather_preserves_tinygrad_order_and_preflights_before_nodes() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let index = graph.input_dtype("index", [2, 2], crate::DType::I32);
        let gathered = graph.gather_signed(input, index, -1).unwrap();
        let loss = graph.sum_all(gathered).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([
            ("input".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            (
                "index".into(),
                TensorData::from_scalars(
                    [2, 2],
                    crate::DType::I32,
                    [
                        crate::Scalar::I(2),
                        crate::Scalar::I(0),
                        crate::Scalar::I(1),
                        crate::Scalar::I(1),
                    ],
                )
                .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, gathered, &bindings).unwrap(),
            data([2, 2], &[3., 1., 5., 5.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            data([2, 3], &[1., 0., 1., 0., 2., 0.])
        );
        assert!(matches!(
            graph.grad(loss, index),
            Err(Error::NonDifferentiableTarget(node)) if node == index
        ));

        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 3]);
        let float_index = malformed.input("float_index", [2, 2]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.gather_signed(input, float_index, -1),
            Err(Error::InvalidIndexDType { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let index = malformed.input_dtype("shape", [2], crate::DType::I32);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.gather_signed(input, index, -1),
            Err(Error::InvalidIndexedShape { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.gather_signed(input, index, isize::MIN),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let scalar = malformed.input("scalar", []);
        let scalar_index = malformed.input_dtype("scalar_index", [], crate::DType::I32);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.gather_signed(scalar, scalar_index, 0),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut empty = Graph::new();
        let input = empty.input("input", [0]);
        let index = empty.input_dtype("index", [0], crate::DType::I32);
        let gathered = empty.gather_signed(input, index, 0).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &empty,
                    gathered,
                    &HashMap::from([
                        ("input".into(), data([0], &[])),
                        (
                            "index".into(),
                            TensorData::from_scalars([0], crate::DType::I32, []).unwrap(),
                        ),
                    ]),
                )
                .unwrap()
                .to_vec_f64(),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn fixed_nonzero_matches_tinygrad_row_major_padding_and_static_boundaries() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let indices = graph.nonzero_fixed(input, 4, crate::Scalar::I(-1)).unwrap();
        assert_eq!(graph.shape(indices).unwrap(), &Shape::new([4, 2]));
        assert_eq!(graph.dtype(indices).unwrap(), crate::DType::I32);
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    indices,
                    &HashMap::from([("input".into(), data([2, 3], &[0., 2., 0., 3., 4., 0.]))]),
                )
                .unwrap(),
            TensorData::from_scalars(
                [4, 2],
                crate::DType::I32,
                [
                    crate::Scalar::I(0),
                    crate::Scalar::I(1),
                    crate::Scalar::I(1),
                    crate::Scalar::I(0),
                    crate::Scalar::I(1),
                    crate::Scalar::I(1),
                    crate::Scalar::I(-1),
                    crate::Scalar::I(-1),
                ],
            )
            .unwrap()
        );
        let loss = graph.sum_all(indices).unwrap();
        assert!(matches!(
            graph.grad(loss, input),
            Err(Error::NonDifferentiableTarget(node)) if node == loss
        ));

        let mut scalar = Graph::new();
        let input = scalar.input("input", []);
        let indices = scalar
            .nonzero_fixed(input, 2, crate::Scalar::I(-1))
            .unwrap();
        assert_eq!(scalar.shape(indices).unwrap(), &Shape::new([2, 0]));
        assert_eq!(
            CpuBackend
                .execute(
                    &scalar,
                    indices,
                    &HashMap::from([("input".into(), TensorData::scalar(1.))]),
                )
                .unwrap()
                .to_vec_f64(),
            Vec::<f64>::new()
        );

        let mut empty = Graph::new();
        let input = empty.input("input", [0, 2]);
        let indices = empty.nonzero_fixed(input, 2, crate::Scalar::I(-1)).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &empty,
                    indices,
                    &HashMap::from([("input".into(), data([0, 2], &[]))]),
                )
                .unwrap(),
            TensorData::from_scalars(
                [2, 2],
                crate::DType::I32,
                [
                    crate::Scalar::I(-1),
                    crate::Scalar::I(-1),
                    crate::Scalar::I(-1),
                    crate::Scalar::I(-1),
                ],
            )
            .unwrap()
        );

        let mut malformed = Graph::new();
        let input = malformed.input("input", [1, 1]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.nonzero_fixed(input, usize::MAX, crate::Scalar::I(0)),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let input = malformed.input("wide", [usize::MAX, 2]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.nonzero_fixed(input, 1, crate::Scalar::I(0)),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), original_nodes);
    }

    #[test]
    fn signed_constant_pad_crops_then_pads_without_partial_movement_nodes() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let padded = graph
            .pad_signed(input, [(-1, 0), (1, -1)], crate::Scalar::F(-1.))
            .unwrap();
        let loss = graph.sum_all(padded).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([("input".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.]))]);
        assert_eq!(
            CpuBackend.execute(&graph, padded, &bindings).unwrap(),
            data([1, 3], &[-1., 4., 5.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            data([2, 3], &[0., 0., 0., 1., 1., 0.])
        );

        let mut empty = Graph::new();
        let input = empty.input("input", [0, 2]);
        let padded = empty
            .pad_signed(input, [(1, 1), (0, 0)], crate::Scalar::F(7.))
            .unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &empty,
                    padded,
                    &HashMap::from([("input".into(), data([0, 2], &[]))]),
                )
                .unwrap(),
            data([2, 2], &[7., 7., 7., 7.])
        );

        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 3]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.pad_signed(input, [(0, 0)], crate::Scalar::F(0.)),
            Err(Error::InvalidMovementRank { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.pad_signed(input, [(-3, 0), (0, 0)], crate::Scalar::F(0.)),
            Err(Error::InvalidBounds { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        let wide = malformed.input("wide", [usize::MAX]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.pad_signed(wide, [(i64::MAX, 0)], crate::Scalar::F(0.)),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), original_nodes);
    }

    #[test]
    fn signed_transpose_matches_tinygrad_axis_swap_and_view_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let transposed = graph.transpose(input, -1, -2).unwrap();
        let selected = graph.shrink(transposed, [(0, 1), (0, 2)]).unwrap();
        let loss = graph.sum_all(selected).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([("input".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.]))]);
        assert_eq!(
            CpuBackend.execute(&graph, transposed, &bindings).unwrap(),
            data([3, 2], &[1., 4., 2., 5., 3., 6.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            data([2, 3], &[1., 0., 0., 1., 0., 0.])
        );

        let mut empty = Graph::new();
        let input = empty.input("input", [0, 2]);
        let transposed = empty.transpose(input, 0, 1).unwrap();
        assert_eq!(empty.shape(transposed).unwrap(), &Shape::new([2, 0]));
        assert_eq!(
            CpuBackend
                .execute(
                    &empty,
                    transposed,
                    &HashMap::from([("input".into(), data([0, 2], &[]))]),
                )
                .unwrap()
                .to_vec_f64(),
            Vec::<f64>::new()
        );

        let mut malformed = Graph::new();
        let scalar = malformed.input("scalar", []);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.transpose(scalar, 0, 0),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let input = malformed.input("input", [2, 3]);
        let original_nodes = malformed.node_count();
        assert_eq!(malformed.transpose(input, 1, 1).unwrap(), input);
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.transpose(input, isize::MIN, 0),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
    }

    #[test]
    fn unflatten_prevalidates_concrete_and_inferred_extents_before_reshape() {
        use crate::UnflattenExtent::{Exact, Infer};

        let mut graph = Graph::new();
        let input = graph.input("input", [2, 4]);
        let concrete = graph.unflatten(input, -1, [Exact(2), Exact(2)]).unwrap();
        let inferred = graph.unflatten(input, -1, [Infer, Exact(2)]).unwrap();
        let selected = graph.shrink(inferred, [(0, 2), (0, 1), (0, 2)]).unwrap();
        let loss = graph.sum_all(selected).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            data([2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]),
        )]);
        assert_eq!(graph.shape(concrete).unwrap(), &Shape::new([2, 2, 2]));
        assert_eq!(graph.shape(inferred).unwrap(), &Shape::new([2, 2, 2]));
        assert_eq!(
            CpuBackend.execute(&graph, concrete, &bindings).unwrap(),
            data([2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            data([2, 4], &[1., 1., 0., 0., 1., 1., 0., 0.])
        );

        let mut empty = Graph::new();
        let input = empty.input("input", [2, 0]);
        let unflattened = empty.unflatten(input, -1, [Infer, Exact(2)]).unwrap();
        assert_eq!(empty.shape(unflattened).unwrap(), &Shape::new([2, 0, 2]));
        assert_eq!(
            CpuBackend
                .execute(
                    &empty,
                    unflattened,
                    &HashMap::from([("input".into(), data([2, 0], &[]))]),
                )
                .unwrap()
                .to_vec_f64(),
            Vec::<f64>::new()
        );

        let mut malformed = Graph::new();
        let scalar = malformed.input("scalar", []);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.unflatten(scalar, 0, [Exact(1)]),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let input = malformed.input("input", [4]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.unflatten(input, 0, [Infer, Infer]),
            Err(Error::InvalidRandom { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.unflatten(input, 0, [Infer, Exact(0)]),
            Err(Error::InvalidRandom { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.unflatten(input, 0, [Exact(3)]),
            Err(Error::InvalidReshape { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.unflatten(input, isize::MIN, [Exact(4)]),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.unflatten(input, 0, [Exact(usize::MAX), Exact(2)]),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        let zero_elsewhere = malformed.input("zero_elsewhere", [0, 6]);
        let zero_elsewhere_nodes = malformed.node_count();
        assert!(matches!(
            malformed.unflatten(zero_elsewhere, 1, [Infer, Exact(4)]),
            Err(Error::InvalidReshape { .. })
        ));
        assert_eq!(malformed.node_count(), zero_elsewhere_nodes);
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
    fn vector_dot_hessian_vector_product_uses_compositional_matmul_vjp() {
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
            !graph
                .trace(hvp)
                .unwrap()
                .to_string()
                .contains("matmul_grad")
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
        let before = graph.node_count();
        let third_order = graph
            .grad_with(shrink_vjp, direction, Some(shrink_seed), true)
            .unwrap_err();
        assert!(
            third_order
                .to_string()
                .contains("third-order indexed contraction gradient")
        );
        assert_eq!(graph.node_count(), before);
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
            !graph
                .trace(hvp)
                .unwrap()
                .to_string()
                .contains("conv2d_grad")
        );
    }
}
