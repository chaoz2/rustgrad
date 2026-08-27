use super::{shape::normalize_axes, Graph, NodeId};
use crate::{DType, Error, ReduceKind, ReductionDType, Result, TensorData};

impl Graph {
    /// Runs a Sum or Product through an explicit, source-validated
    /// accumulator/output dtype contract.
    ///
    /// The whole contract, axis list, and source extent are validated before
    /// the accumulator cast, reduction, or final narrowing cast is appended.
    pub fn reduce_with_dtypes(
        &mut self,
        input: NodeId,
        kind: ReduceKind,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        dtypes: ReductionDType,
    ) -> Result<NodeId> {
        let (shape, input_dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        let axes = normalize_axes(input, shape.rank(), axes)?;
        if !valid_reduction_dtypes(kind, input_dtype, dtypes) {
            return Err(Error::InvalidElementwiseDType {
                op: "reduce_with_dtypes",
                actual: dtypes.accumulator,
            });
        }
        let accumulator = if input_dtype == dtypes.accumulator {
            input
        } else {
            self.cast(input, dtypes.accumulator)?
        };
        let reduced = self.reduce(
            accumulator,
            kind,
            Some(axes.into_iter().map(|axis| axis as isize).collect()),
            keepdim,
        )?;
        if dtypes.output == dtypes.accumulator {
            Ok(reduced)
        } else {
            self.cast(reduced, dtypes.output)
        }
    }

    fn boolean_reduction_input(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
    ) -> Result<(NodeId, Vec<usize>)> {
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
        Ok((boolean, axes))
    }

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
        let (boolean, axes) = self.boolean_reduction_input(input, axes)?;
        self.reduce(
            boolean,
            crate::ReduceKind::Product,
            Some(axes.into_iter().map(|axis| axis as isize).collect()),
            keepdim,
        )
    }

    /// Boolean any-reduction over optional signed axes.
    ///
    /// `any` is the Boolean dual of [`Self::all`]: `!all(!bool(input))`.
    /// This gives the exact false empty identity without routing through an
    /// integer accumulator, and preflights axes before appending any nodes.
    pub fn any(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let (boolean, axes) = self.boolean_reduction_input(input, axes)?;
        let inverted = self.logical_not(boolean)?;
        let all_inverted = self.reduce(
            inverted,
            crate::ReduceKind::Product,
            Some(axes.into_iter().map(|axis| axis as isize).collect()),
            keepdim,
        )?;
        self.logical_not(all_inverted)
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

fn valid_reduction_dtypes(kind: ReduceKind, input: DType, dtypes: ReductionDType) -> bool {
    match kind {
        ReduceKind::Sum => {
            dtypes == ReductionDType::sum_default(input)
                || dtypes.accumulator == dtypes.output
        }
        ReduceKind::Product => dtypes.accumulator == dtypes.output,
        ReduceKind::Mean | ReduceKind::Max | ReduceKind::Min => false,
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

    #[test]
    fn any_is_boolean_nondifferentiable_and_preflights_axes() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let original_nodes = graph.node_count();
        assert!(matches!(
            graph.any(input, Some(vec![-1, 1]), false),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(graph.node_count(), original_nodes);

        let any = graph.any(input, Some(vec![0, -1]), true).unwrap();
        let bindings = HashMap::from([(
            "input".into(),
            data([2, 2], &[0., 0., f32::NAN, 0.]),
        )]);
        let output = CpuBackend.execute(&graph, any, &bindings).unwrap();
        assert_eq!(graph.shape(any).unwrap(), &Shape::new([1, 1]));
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![1.]);
        assert!(matches!(graph.grad(any, input), Err(Error::NoGradient(_))));

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [2, 0]);
        let reduced = empty_graph.any(empty, Some(vec![-1]), false).unwrap();
        let output = CpuBackend
            .execute(
                &empty_graph,
                reduced,
                &HashMap::from([("empty".into(), data([2, 0], &[]))]),
            )
            .unwrap();
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.to_vec_f64(), vec![0., 0.]);
    }

    #[test]
    fn typed_reduction_accumulation_preflights_and_preserves_source_contracts() {
        let mut malformed = Graph::new();
        let input = malformed.input_dtype("input", [2, 2], DType::F16);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                Some(vec![-1, 1]),
                false,
                ReductionDType::sum_default(DType::F16),
            ),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                Some(vec![isize::MIN]),
                false,
                ReductionDType::sum_default(DType::F16),
            ),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                Some(vec![-1]),
                false,
                ReductionDType::new(DType::F64, DType::F32),
            ),
            Err(Error::InvalidElementwiseDType { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut graph = Graph::new();
        let narrow = graph.input_dtype("narrow", [2, 2], DType::F16);
        let narrowed_sum = graph
            .reduce_with_dtypes(
                narrow,
                ReduceKind::Sum,
                Some(vec![-1]),
                false,
                ReductionDType::sum_default(DType::F16),
            )
            .unwrap();
        assert_eq!(graph.dtype(narrowed_sum).unwrap(), DType::F16);
        let reduced = match &graph.nodes[narrowed_sum.index()].op {
            crate::Op::Cast {
                input,
                dtype: DType::F16,
            } => *input,
            op => panic!("expected final F16 cast, got {op:?}"),
        };
        assert_eq!(graph.dtype(reduced).unwrap(), DType::F32);
        let narrow_data = TensorData::from_scalars(
            [2, 2],
            DType::F16,
            [1.5, 2.25, 3.5, 4.75].map(crate::Scalar::F),
        )
        .unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    narrowed_sum,
                    &HashMap::from([("narrow".into(), narrow_data.clone())]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![3.75, 8.25]
        );
        let narrow_loss = graph.sum_all(narrowed_sum).unwrap();
        let narrow_gradient = graph.grad(narrow_loss, narrow).unwrap();
        assert_eq!(graph.dtype(narrow_gradient).unwrap(), DType::F16);
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    narrow_gradient,
                    &HashMap::from([("narrow".into(), narrow_data)]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![1.; 4]
        );

        let bfloat = graph.input_dtype("bfloat", [2], DType::BF16);
        let bfloat_sum = graph
            .reduce_with_dtypes(
                bfloat,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::BF16),
            )
            .unwrap();
        let reduced = match &graph.nodes[bfloat_sum.index()].op {
            crate::Op::Cast {
                input,
                dtype: DType::BF16,
            } => *input,
            op => panic!("expected final BF16 cast, got {op:?}"),
        };
        assert_eq!(graph.dtype(reduced).unwrap(), DType::F32);

        let single = graph.input("single", [2]);
        let single_sum = graph
            .reduce_with_dtypes(
                single,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::F32),
            )
            .unwrap();
        assert_eq!(graph.dtype(single_sum).unwrap(), DType::F32);
        let widened_product = graph
            .reduce_with_dtypes(
                single,
                ReduceKind::Product,
                None,
                false,
                ReductionDType::new(DType::F64, DType::F64),
            )
            .unwrap();
        assert_eq!(graph.dtype(widened_product).unwrap(), DType::F64);

        let wide = graph.input_dtype("wide", [2], DType::F64);
        let wide_sum = graph
            .reduce_with_dtypes(
                wide,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::F64),
            )
            .unwrap();
        assert_eq!(graph.dtype(wide_sum).unwrap(), DType::F64);

        let integer = graph.input_dtype("integer", [2], DType::I8);
        let integer_sum = graph
            .reduce_with_dtypes(
                integer,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::I8),
            )
            .unwrap();
        assert_eq!(graph.dtype(integer_sum).unwrap(), DType::I32);

        let legacy = graph.input_dtype("legacy", [2], DType::F16);
        let legacy_sum = graph.reduce(legacy, ReduceKind::Sum, None, false).unwrap();
        assert_eq!(graph.dtype(legacy_sum).unwrap(), DType::F16);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 2]);
        let original_nodes = overflow.node_count();
        assert!(matches!(
            overflow.reduce_with_dtypes(
                input,
                ReduceKind::Sum,
                None,
                false,
                ReductionDType::sum_default(DType::F32),
            ),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), original_nodes);
    }
}
