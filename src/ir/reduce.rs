use super::{shape::normalize_axes, Graph, NodeId};
use crate::{DType, Error, Result, TensorData};

impl Graph {
    /// Product reduction over optional signed axes.
    pub fn prod(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        self.reduce(input, crate::ReduceKind::Product, axes, keepdim)
    }

    /// Boolean all-reduction over optional signed axes.
    ///
    /// This is the checked `bool().prod(...)` composition used by tinygrad:
    /// every nonzero value (including NaN) is true, and an empty product is
    /// true. Axis normalization completes before a cast or reduction node is
    /// appended.
    pub fn all(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let (rank, dtype) = {
            let source = self.node(input)?;
            (source.shape.rank(), source.dtype)
        };
        let axes = normalize_axes(input, rank, axes)?;
        let boolean = if dtype == DType::Bool {
            input
        } else {
            self.cast(input, DType::Bool)?
        };
        self.reduce(
            boolean,
            crate::ReduceKind::Product,
            Some(axes.into_iter().map(|axis| axis as isize).collect()),
            keepdim,
        )
    }

    /// Reduces multiple axes. Axes refer to the original input rank.
    pub fn sum_axes(
        &mut self,
        input: NodeId,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        let mut axes = axes.into_iter().collect::<Vec<_>>();
        axes.sort_unstable();
        if axes.iter().any(|axis| *axis >= rank) || axes.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes,
                rank,
            });
        }

        let mut output = input;
        for axis in axes.into_iter().rev() {
            output = self.sum(output, axis)?;
        }
        Ok(output)
    }

    pub fn sum_all(&mut self, input: NodeId) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        self.sum_axes(input, 0..rank)
    }

    pub fn mean(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let count = *shape.dims().get(axis).ok_or(Error::InvalidAxis {
            node: input,
            axis,
            rank: shape.rank(),
        })?;
        let sum = self.sum(input, axis)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }

    pub fn mean_axes(
        &mut self,
        input: NodeId,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let mut axes = axes.into_iter().collect::<Vec<_>>();
        axes.sort_unstable();
        if axes.iter().any(|axis| *axis >= shape.rank())
            || axes.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes,
                rank: shape.rank(),
            });
        }
        let count = axes
            .iter()
            .try_fold(1usize, |count, axis| count.checked_mul(shape.dims()[*axis]))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let sum = self.sum_axes(input, axes)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }

    pub fn mean_all(&mut self, input: NodeId) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let count = shape.numel()?;
        let sum = self.sum_all(input)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Shape};
    use std::collections::HashMap;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn multi_axis_sum_and_mean_are_composable_and_differentiable() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2, 2]);
        let sum = graph.sum_axes(x, [0, 2]).unwrap();
        let mean = graph.mean_all(x).unwrap();
        let dx = graph.grad(mean, x).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            data([2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]),
        )]);

        assert_eq!(
            CpuBackend.execute(&graph, sum, &inputs).unwrap(),
            data([2], &[14., 22.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, mean, &inputs).unwrap(),
            TensorData::scalar(4.5)
        );
        assert_eq!(
            CpuBackend.execute(&graph, dx, &inputs).unwrap(),
            data([2, 2, 2], &[0.125; 8])
        );
    }

    #[test]
    fn rejects_duplicate_reduction_axes() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        assert_eq!(
            graph.sum_axes(x, [1, 1]),
            Err(Error::InvalidReductionAxes {
                node: x,
                axes: vec![1, 1],
                rank: 2
            })
        );
    }

    #[test]
    fn mean_all_preflights_total_extent_before_sum_lowering() {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [usize::MAX, 2]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.mean_all(input),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut valid = Graph::new();
        let input = valid.input("input", [2]);
        let output = valid.mean_all(input).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &valid,
                    output,
                    &HashMap::from([("input".into(), data([2], &[2., 6.]))]),
                )
                .unwrap(),
            TensorData::scalar(4.)
        );
    }

    #[test]
    fn generalized_reductions_normalize_axes_keep_dimensions_and_arg_ties() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let sum = graph
            .reduce(x, crate::ReduceKind::Sum, Some(vec![-1]), true)
            .unwrap();
        let product = graph
            .reduce(x, crate::ReduceKind::Product, Some(vec![0]), false)
            .unwrap();
        let maximum = graph
            .reduce(x, crate::ReduceKind::Max, None, false)
            .unwrap();
        let minimum = graph
            .reduce(x, crate::ReduceKind::Min, Some(vec![1]), false)
            .unwrap();
        let argmax = graph.argmax(x, Some(-1), false).unwrap();
        let inputs = HashMap::from([("x".into(), data([2, 3], &[1., 3., 3., 2., 0., -1.]))]);
        assert_eq!(
            CpuBackend.execute(&graph, sum, &inputs).unwrap(),
            data([2, 1], &[7., 1.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, product, &inputs).unwrap(),
            data([3], &[2., 0., -3.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, maximum, &inputs).unwrap(),
            TensorData::scalar(3.)
        );
        assert_eq!(
            CpuBackend.execute(&graph, minimum, &inputs).unwrap(),
            data([2], &[1., -1.])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, argmax, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I32(vec![1, 0])
        );
        assert!(graph.trace(argmax).unwrap().to_string().contains("argmax"));
    }

    #[test]
    fn prod_forwards_signed_axes_to_validated_product_reduction() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let original_nodes = graph.node_count();
        assert!(matches!(
            graph.prod(input, Some(vec![-1, 1]), false),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(graph.node_count(), original_nodes);

        let product = graph.prod(input, Some(vec![0, -1]), true).unwrap();
        let gradient = graph.grad(product, input).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            data([2, 3], &[1., 2., 3., 4., 5., 6.]),
        )]);
        assert_eq!(graph.shape(product).unwrap(), &Shape::new([1, 1]));
        assert_eq!(
            CpuBackend.execute(&graph, product, &bindings).unwrap(),
            data([1, 1], &[720.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            data([2, 3], &[720., 360., 240., 180., 144., 120.])
        );

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [2, 0]);
        let reduced = empty_graph.prod(empty, Some(vec![-1]), false).unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &empty_graph,
                    reduced,
                    &HashMap::from([("empty".into(), data([2, 0], &[]))]),
                )
                .unwrap(),
            data([2], &[1., 1.])
        );
    }

    #[test]
    fn all_is_boolean_nondifferentiable_and_preflights_axes() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let original_nodes = graph.node_count();
        assert!(matches!(
            graph.all(input, Some(vec![-1, 1]), false),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(graph.node_count(), original_nodes);

        let all = graph.all(input, Some(vec![0, -1]), true).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            data([2, 2], &[1., -2., f32::NAN, 4.]),
        )]);
        let output = CpuBackend.execute(&graph, all, &bindings).unwrap();
        assert_eq!(graph.shape(all).unwrap(), &Shape::new([1, 1]));
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![1.]);
        assert!(matches!(graph.grad(all, input), Err(Error::NoGradient(_))));

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [2, 0]);
        let reduced = empty_graph.all(empty, Some(vec![-1]), false).unwrap();
        let output = CpuBackend
            .execute(
                &empty_graph,
                reduced,
                &HashMap::from([("empty".into(), data([2, 0], &[]))]),
            )
            .unwrap();
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![1., 1.]);
    }
}
