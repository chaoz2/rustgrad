use super::{Graph, NodeId, RandomKind, RandomStream, ReduceKind};
use crate::{DType, Error, Result, Shape, TensorData};

#[derive(Clone, Debug)]
pub(crate) struct MultinomialPlan {
    pub(crate) axis: usize,
    samples: usize,
    replacement: bool,
    output_shape: Shape,
    pub(crate) random_shape: Shape,
    pub(crate) dtype: DType,
}

impl MultinomialPlan {
    pub(crate) fn new(graph: &Graph, input: NodeId, samples: usize, axis: isize, replacement: bool) -> Result<Self> {
        let source = graph.node(input)?;
        if !source.dtype.is_float() {
            return Err(Error::InvalidRandom { reason: "multinomial requires floating probabilities" });
        }
        let rank = source.shape.rank();
        if !(1..=2).contains(&rank) {
            return Err(Error::InvalidRandom { reason: "multinomial requires rank one or two probabilities" });
        }
        let normalized = if axis < 0 { axis + rank as isize } else { axis };
        if normalized < 0 || normalized >= rank as isize {
            return Err(Error::InvalidAxis { node: input, axis: usize::MAX, rank });
        }
        let axis = normalized as usize;
        let extent = source.shape.dims()[axis];
        if extent == 0 {
            return Err(Error::InvalidRandom { reason: "multinomial category axis must be nonempty" });
        }
        if !replacement && samples > extent {
            return Err(Error::InvalidBounds { axis, start: 0, end: samples, dim: extent });
        }
        let mut output_dims = source.shape.dims().to_vec();
        output_dims[axis] = samples;
        let output_shape = Shape::new(output_dims);
        output_shape.numel()?;
        let random_shape = if replacement { output_shape.clone() } else { source.shape.clone() };
        random_shape.numel()?;
        Ok(Self { axis, samples, replacement, output_shape, random_shape, dtype: source.dtype })
    }
}

impl Graph {
    /// Samples I32 category indices from floating weights using an explicit,
    /// replayable Threefry stream. Values are validated by TensorGuard before
    /// the composed sampling result is realized.
    pub fn multinomial(
        &mut self,
        input: NodeId,
        samples: usize,
        axis: isize,
        replacement: bool,
        stream: RandomStream,
    ) -> Result<NodeId> {
        let plan = MultinomialPlan::new(self, input, samples, axis, replacement)?;
        let guarded = self.tensor_guard_distribution(input, plan.axis as isize)?;
        let uniform = self.random_stream(
            plan.random_shape.clone(),
            plan.dtype,
            RandomKind::Uniform { low: 0.0, high: 1.0 },
            stream,
        )?;
        self.multinomial_from_uniform(guarded, uniform, &plan)
    }

    /// Composes multinomial output from a TensorGuard-validated weight node and
    /// an already-reserved ordinary uniform RandomStream node. This is the
    /// session continuation seam; it introduces no additional stream state.
    pub(crate) fn multinomial_from_uniform(
        &mut self,
        guarded: NodeId,
        uniform: NodeId,
        plan: &MultinomialPlan,
    ) -> Result<NodeId> {
        if !matches!(self.op(guarded)?, super::Op::TensorGuard { .. }) {
            return Err(Error::InvalidRandom { reason: "multinomial requires a TensorGuard input" });
        }
        if self.shape(uniform)? != &plan.random_shape || self.dtype(uniform)? != plan.dtype {
            return Err(Error::InvalidRandom { reason: "multinomial uniform stream shape or dtype does not match request" });
        }
        let total = self.reduce(guarded, ReduceKind::Sum, Some(vec![plan.axis as isize]), true)?;
        let weights = self.div(guarded, total)?;
        if !plan.replacement {
            let one = self.constant(TensorData::scalar_with_dtype(crate::Scalar::F(1.0), plan.dtype));
            let inverse = self.div(one, weights)?;
            let keys = self.pow(uniform, inverse)?;
            let output = self.topk(keys, plan.samples, plan.axis as isize, true)?.1;
            debug_assert_eq!(self.shape(output)?, &plan.output_shape);
            return Ok(output);
        }
        let cdf = self.cumsum(weights, plan.axis as isize)?;
        let cdf = self.unsqueeze(cdf, (plan.axis + 1) as isize)?;
        let uniform = self.unsqueeze(uniform, plan.axis as isize)?;
        let before = self.le(cdf, uniform)?;
        let count = self.reduce(before, ReduceKind::Sum, Some(vec![plan.axis as isize]), false)?;
        let output = self.cast(count, DType::I32)?;
        debug_assert_eq!(self.shape(output)?, &plan.output_shape);
        Ok(output)
    }
}
