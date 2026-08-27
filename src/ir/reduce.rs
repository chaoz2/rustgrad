use super::{shape::normalize_axes, Graph, NodeId};
use crate::{DType, Error, ReduceKind, ReductionDType, Result, Scalar, TensorData, VarianceCorrection};

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

    /// Cumulative sum along one signed axis.
    ///
    /// This is the checked static composition used by tinygrad's `cumsum`:
    /// each inclusive prefix is reduced with tinygrad's default Sum
    /// accumulator/output contract, then the prefix results are concatenated.
    /// Axis, source extent, every prefix bound, and the empty/scalar result
    /// dtype are all resolved before the first movement or reduction node.
    pub fn cumsum(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        let dtypes = ReductionDType::sum_default(dtype);
        if shape.rank() == 0 {
            if !matches!(axis, -1 | 0) {
                return Err(Error::InvalidAxis {
                    node: input,
                    axis: usize::MAX,
                    rank: 0,
                });
            }
            return self.cast(input, dtypes.output);
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        if shape.dims().contains(&0) {
            return self.cast(input, dtypes.output);
        }

        let prefixes = (0..shape.dims()[axis])
            .map(|end| {
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &extent)| {
                        if dimension == axis {
                            (0, end + 1)
                        } else {
                            (0, extent)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let values = prefixes
            .into_iter()
            .map(|bounds| {
                let prefix = self.shrink(input, bounds)?;
                self.reduce_with_dtypes(
                    prefix,
                    crate::ReduceKind::Sum,
                    Some(vec![axis as isize]),
                    true,
                    dtypes,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.concat(values, axis)
    }

    /// Cumulative product along one signed axis.
    ///
    /// This is the checked static composition used by tinygrad's `cumprod`:
    /// each inclusive prefix is reduced with tinygrad's default Product
    /// accumulator/output contract, then the prefix results are concatenated.
    /// Axis, source extent, and every prefix bound are resolved before the
    /// first movement or reduction node.
    pub fn cumprod(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        let dtypes = ReductionDType::product_default(dtype);
        if shape.rank() == 0 {
            if !matches!(axis, -1 | 0) {
                return Err(Error::InvalidAxis {
                    node: input,
                    axis: usize::MAX,
                    rank: 0,
                });
            }
            return Ok(input);
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        if shape.dims().contains(&0) {
            return Ok(input);
        }

        let prefixes = (0..shape.dims()[axis])
            .map(|end| {
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &extent)| {
                        if dimension == axis {
                            (0, end + 1)
                        } else {
                            (0, extent)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let values = prefixes
            .into_iter()
            .map(|bounds| {
                let prefix = self.shrink(input, bounds)?;
                self.reduce_with_dtypes(
                    prefix,
                    crate::ReduceKind::Product,
                    Some(vec![axis as isize]),
                    true,
                    dtypes,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.concat(values, axis)
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

    /// Variance with tinygrad's signed `correction` contract.
    ///
    /// The numerator is accumulated through the source dtype's Sum
    /// accumulator, while the public result is the floating input dtype (or
    /// F32 for nonfloating inputs).  As in tinygrad, the denominator is
    /// `max(n - correction, 0)`, including for empty reductions.
    pub fn var(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: Option<VarianceCorrection>,
    ) -> Result<NodeId> {
        let (shape, input_dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        let axes = normalize_axes(input, shape.rank(), axes)?;
        let count = axes.iter().try_fold(1usize, |count, axis| {
            count
                .checked_mul(shape.dims()[*axis])
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        })?;
        let correction = correction.unwrap_or(VarianceCorrection::UNBIASED);
        let denominator = variance_denominator(count, correction, &shape)?;
        let accumulation = ReductionDType::sum_default(input_dtype).accumulator;
        let output_dtype = if input_dtype.is_float() {
            input_dtype
        } else {
            DType::F32
        };
        let accumulation_contract = ReductionDType::new(accumulation, accumulation);
        let normalized_axes = Some(axes.iter().map(|axis| *axis as isize).collect());

        // `mean` first accumulates in the Sum accumulator and only then casts
        // to the public dtype, matching tinygrad's explicit cast/sum/div/cast
        // composition for narrow floats.
        let mean_sum = self.reduce_with_dtypes(
            input,
            ReduceKind::Sum,
            normalized_axes.clone(),
            true,
            accumulation_contract,
        )?;
        let mean_divisor = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(count as f64),
            output_dtype,
        ));
        let mean = self.div(mean_sum, mean_divisor)?;
        let mean = if self.dtype(mean)? == output_dtype {
            mean
        } else {
            self.cast(mean, output_dtype)?
        };
        let deviations = self.sub(input, mean)?;
        let squares = self.square(deviations)?;
        let numerator = self.reduce_with_dtypes(
            squares,
            ReduceKind::Sum,
            normalized_axes,
            keepdim,
            accumulation_contract,
        )?;
        let divisor = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(denominator as f64),
            output_dtype,
        ));
        let variance = self.div(numerator, divisor)?;
        if self.dtype(variance)? == output_dtype {
            Ok(variance)
        } else {
            self.cast(variance, output_dtype)
        }
    }

    /// Standard deviation, defined exactly as `sqrt(var(...))`.
    pub fn std(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: Option<VarianceCorrection>,
    ) -> Result<NodeId> {
        let variance = self.var(input, axes, keepdim, correction)?;
        self.sqrt(variance)
    }

    /// L2-normalizes a floating tensor along one signed axis.
    ///
    /// This is the closed `p = 2` form of tinygrad's public `normalize`
    /// composition: `x / max(sqrt(sum(abs(x)^2, keepdim=true)), eps)`. The
    /// axis, source extent, and dtype are all validated before any cast,
    /// reduction, constant, or elementwise node is appended. `eps` remains a
    /// plain floating scalar because tinygrad applies no finiteness or sign
    /// validation before its `maximum` composition.
    pub fn normalize_l2(&mut self, input: NodeId, axis: isize, eps: f64) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        if !dtype.is_float() {
            return Err(Error::InvalidElementwiseDType {
                op: "normalize_l2",
                actual: dtype,
            });
        }
        shape.numel()?;
        let axes = normalize_axes(input, shape.rank(), Some(vec![axis]))?;
        let accumulation = ReductionDType::sum_default(dtype);
        let normalized_axes = Some(axes.into_iter().map(|axis| axis as isize).collect());

        // For real dtypes, `square` is the exact closed p=2 instance of
        // tinygrad's `abs().pow(2.0)` composition, including its normal
        // floating nonfinite propagation.
        let squares = self.square(input)?;
        let sum = self.reduce_with_dtypes(
            squares,
            ReduceKind::Sum,
            normalized_axes,
            true,
            accumulation,
        )?;
        let magnitude = self.sqrt(sum)?;
        let epsilon = self.constant(TensorData::scalar_with_dtype(Scalar::F(eps), dtype));
        let denominator = self.maximum(magnitude, epsilon)?;
        self.div(input, denominator)
    }
}

fn variance_denominator(
    count: usize,
    correction: VarianceCorrection,
    shape: &crate::Shape,
) -> Result<usize> {
    let correction = correction.value();
    if correction >= 0 {
        let correction = usize::try_from(correction).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        Ok(count.saturating_sub(correction))
    } else {
        let magnitude = usize::try_from(correction.unsigned_abs())
            .map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        count
            .checked_add(magnitude)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
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
    fn cumsum_matches_tinygrad_signed_axis_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let cumulative = graph.cumsum(input, -1).unwrap();
        let loss = graph.sum_all(cumulative).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([(
            "input".into(),
            TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, cumulative, &inputs).unwrap().to_vec_f64(),
            vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs).unwrap().to_vec_f64(),
            vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0]
        );

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::I8);
        let cumulative = scalar.cumsum(input, 0).unwrap();
        assert_eq!(scalar.dtype(cumulative).unwrap(), DType::I32);
        assert_eq!(
            CpuBackend
                .execute(
                    &scalar,
                    cumulative,
                    &HashMap::from([(
                        "input".into(),
                        TensorData::from_scalars([], DType::I8, [Scalar::I(5)]).unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![5.0]
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0], DType::F16);
        let cumulative = empty.cumsum(input, -1).unwrap();
        assert_eq!(empty.dtype(cumulative).unwrap(), DType::F16);
        assert!(CpuBackend
            .execute(
                &empty,
                cumulative,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64()
            .is_empty());

        let mut invalid = Graph::new();
        let input = invalid.input("input", [2]);
        let before_nodes = invalid.node_count();
        assert!(invalid.cumsum(input, 1).is_err());
        assert_eq!(invalid.node_count(), before_nodes);
    }

    #[test]
    fn cumprod_matches_tinygrad_signed_axis_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let cumulative = graph.cumprod(input, -1).unwrap();
        let loss = graph.sum_all(cumulative).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([(
            "input".into(),
            TensorData::new([2, 3], vec![2.0, 3.0, 4.0, 2.0, 0.0, -3.0]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, cumulative, &inputs).unwrap().to_vec_f64(),
            vec![2.0, 6.0, 24.0, 2.0, 0.0, 0.0]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs).unwrap().to_vec_f64(),
            vec![16.0, 10.0, 6.0, 1.0, -4.0, 0.0]
        );

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::I8);
        let cumulative = scalar.cumprod(input, 0).unwrap();
        assert_eq!(scalar.dtype(cumulative).unwrap(), DType::I8);
        assert_eq!(
            CpuBackend
                .execute(
                    &scalar,
                    cumulative,
                    &HashMap::from([(
                        "input".into(),
                        TensorData::from_scalars([], DType::I8, [Scalar::I(-5)]).unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![-5.0]
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0], DType::F16);
        let cumulative = empty.cumprod(input, -1).unwrap();
        assert_eq!(empty.dtype(cumulative).unwrap(), DType::F16);
        assert!(CpuBackend
            .execute(
                &empty,
                cumulative,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64()
            .is_empty());

        let mut invalid = Graph::new();
        let input = invalid.input("input", [2]);
        let before_nodes = invalid.node_count();
        assert!(invalid.cumprod(input, 1).is_err());
        assert_eq!(invalid.node_count(), before_nodes);
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

    #[test]
    fn variance_and_std_match_tinygrad_correction_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("input", [3]);
        let default_variance = graph.var(input, None, false, None).unwrap();
        let population_variance = graph
            .var(
                input,
                None,
                false,
                Some(VarianceCorrection::new(0)),
            )
            .unwrap();
        let negative_correction = graph
            .var(
                input,
                None,
                false,
                Some(VarianceCorrection::new(-1)),
            )
            .unwrap();
        let standard_deviation = graph
            .std(
                input,
                Some(vec![-1]),
                true,
                Some(VarianceCorrection::new(0)),
            )
            .unwrap();
        let default_gradient = graph.grad(default_variance, input).unwrap();
        let gradient = graph.grad(population_variance, input).unwrap();
        let inputs = HashMap::from([("input".into(), data([3], &[1., 3., 5.]))]);

        assert_eq!(
            CpuBackend.execute(&graph, default_variance, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![4.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, population_variance, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![(8.0f32 / 3.0) as f64]
        );
        assert_eq!(
            CpuBackend.execute(&graph, negative_correction, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![2.]
        );
        assert_eq!(graph.shape(standard_deviation).unwrap(), &Shape::new([1]));
        assert_eq!(
            CpuBackend
                .execute(&graph, standard_deviation, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![(8.0f32 / 3.0).sqrt() as f64]
        );
        assert_eq!(
            CpuBackend.execute(&graph, default_gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![-2., 0., 2.]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![(-4.0f32 / 3.0) as f64, 0., (4.0f32 / 3.0) as f64]
        );

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", [2], DType::F16);
        let bf16 = narrow.input_dtype("bf16", [2], DType::BF16);
        let f16_variance = narrow.var(f16, None, false, None).unwrap();
        let bf16_variance = narrow.var(bf16, None, false, None).unwrap();
        assert_eq!(narrow.dtype(f16_variance).unwrap(), DType::F16);
        assert_eq!(narrow.dtype(bf16_variance).unwrap(), DType::BF16);
        assert!(narrow.nodes.iter().any(|node| {
            matches!(&node.op, crate::Op::Reduce { kind: ReduceKind::Sum, .. })
                && node.dtype == DType::F32
        }));
        let f16_data = TensorData::from_scalars([2], DType::F16, [Scalar::F(1.5), Scalar::F(2.5)])
            .unwrap();
        assert_eq!(
            CpuBackend
                .execute(
                    &narrow,
                    f16_variance,
                    &HashMap::from([("f16".into(), f16_data)]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![0.5]
        );

        let mut integer = Graph::new();
        let values = integer.input_dtype("values", [2], DType::I32);
        let variance = integer
            .var(values, None, false, Some(VarianceCorrection::new(0)))
            .unwrap();
        assert_eq!(integer.dtype(variance).unwrap(), DType::F32);
        assert_eq!(
            CpuBackend
                .execute(
                    &integer,
                    variance,
                    &HashMap::from([(
                        "values".into(),
                        TensorData::from_scalars(
                            [2],
                            DType::I32,
                            [Scalar::I(1), Scalar::I(3)],
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![1.]
        );
    }

    #[test]
    fn variance_preflights_axes_extents_and_preserves_empty_zero_denominator_policy() {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2, 3]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.var(input, Some(vec![-1, 1]), false, None),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);
        assert!(matches!(
            malformed.var(input, Some(vec![isize::MIN]), false, None),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX]);
        let original_nodes = overflow.node_count();
        assert!(matches!(
            overflow.var(
                input,
                None,
                false,
                Some(VarianceCorrection::new(-1)),
            ),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), original_nodes);

        let mut empty = Graph::new();
        let input = empty.input("input", [0]);
        let variance = empty.var(input, None, false, None).unwrap();
        let output = CpuBackend
            .execute(
                &empty,
                variance,
                &HashMap::from([("input".into(), data([0], &[]))]),
            )
            .unwrap()
            .to_vec_f64();
        assert!(output[0].is_nan());

        let mut singleton = Graph::new();
        let input = singleton.input("input", []);
        let variance = singleton.var(input, None, false, None).unwrap();
        let output = CpuBackend
            .execute(
                &singleton,
                variance,
                &HashMap::from([("input".into(), TensorData::scalar(7.))]),
            )
            .unwrap()
            .to_vec_f64();
        assert!(output[0].is_nan());
    }

    #[test]
    fn l2_normalize_matches_the_closed_tinygrad_default_composition() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let normalized = graph.normalize_l2(input, -1, 1e-12).unwrap();
        let loss = graph.sum_all(normalized).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([("input".into(), data([2, 2], &[3., 4., 5., 12.]))]);

        assert_eq!(graph.shape(normalized).unwrap(), &Shape::new([2, 2]));
        assert_eq!(
            CpuBackend.execute(&graph, normalized, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![
                0.6f32 as f64,
                0.8f32 as f64,
                (5.0f32 / 13.0) as f64,
                (12.0f32 / 13.0) as f64,
            ]
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &inputs)
                .unwrap()
                .to_vec_f64(),
            vec![
                0.032f32 as f64,
                -0.024f32 as f64,
                (84.0f32 / 2197.0) as f64,
                (-35.0f32 / 2197.0) as f64,
            ]
        );

        let mut narrow = Graph::new();
        let input = narrow.input_dtype("input", [2], DType::F16);
        let normalized = narrow.normalize_l2(input, 0, 1e-12).unwrap();
        assert_eq!(narrow.dtype(normalized).unwrap(), DType::F16);
        assert!(narrow.nodes.iter().any(|node| {
            matches!(&node.op, crate::Op::Reduce { kind: ReduceKind::Sum, .. })
                && node.dtype == DType::F32
        }));
        assert_eq!(
            CpuBackend
                .execute(
                    &narrow,
                    normalized,
                    &HashMap::from([(
                        "input".into(),
                        TensorData::from_scalars(
                            [2],
                            DType::F16,
                            [Scalar::F(3.), Scalar::F(4.)],
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            vec![0.60009765625, 0.7998046875]
        );
    }

    #[test]
    fn l2_normalize_preflights_dtype_axis_and_empty_scalar_boundaries() {
        let mut malformed = Graph::new();
        let integer = malformed.input_dtype("integer", [2], DType::I32);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.normalize_l2(integer, 0, 1e-12),
            Err(Error::InvalidElementwiseDType { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let floating = malformed.input("floating", [2]);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.normalize_l2(floating, 1, 1e-12),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let scalar = malformed.input("scalar", []);
        let original_nodes = malformed.node_count();
        assert!(matches!(
            malformed.normalize_l2(scalar, 0, 1e-12),
            Err(Error::InvalidReductionAxes { .. })
        ));
        assert_eq!(malformed.node_count(), original_nodes);

        let mut empty = Graph::new();
        let input = empty.input("input", [2, 0]);
        let normalized = empty.normalize_l2(input, -1, 1e-12).unwrap();
        assert_eq!(empty.shape(normalized).unwrap(), &Shape::new([2, 0]));
        assert_eq!(
            CpuBackend
                .execute(
                    &empty,
                    normalized,
                    &HashMap::from([("input".into(), data([2, 0], &[]))]),
                )
                .unwrap()
                .to_vec_f64(),
            Vec::<f64>::new()
        );
    }
}
