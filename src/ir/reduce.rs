use super::{normalize_axes, Graph, NodeId, ReduceKind};
use crate::{DType, Error, Result, Scalar, TensorData};

#[derive(Clone, Debug)]
struct VariancePlan {
    axes: Vec<usize>,
    accumulator_dtype: DType,
    output_dtype: DType,
    denominator: usize,
}

impl VariancePlan {
    /// Validates the complete public variance request before graph construction.
    ///
    /// tinygrad defines variance through mean, square, and sum.  Float8's
    /// released CPU boundary intentionally does not include those elementwise
    /// operations, so reject it before adding a partial expression.
    fn new(
        graph: &Graph,
        input: NodeId,
        axes: Option<Vec<isize>>,
        correction: usize,
    ) -> Result<Self> {
        let source = graph.node(input)?;
        let axes = normalize_axes(input, source.shape.rank(), axes)?;
        if source.dtype.is_float8() {
            return Err(Error::InvalidElementwiseDType {
                op: "var",
                actual: source.dtype,
            });
        }
        let count = axes
            .iter()
            .try_fold(1usize, |count, axis| {
                count.checked_mul(source.shape.dims()[*axis])
            })
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        let accumulator_dtype = if source.dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        Ok(Self {
            axes,
            accumulator_dtype,
            output_dtype: if source.dtype.is_float() {
                source.dtype
            } else {
                DType::F32
            },
            denominator: count.saturating_sub(correction),
        })
    }

    fn signed_axes(&self) -> Vec<isize> {
        self.axes.iter().map(|&axis| axis as isize).collect()
    }
}

impl Graph {
    /// Computes tinygrad-compatible variance over `axes`.
    ///
    /// `correction` is Bessel's correction.  A reduction with zero effective
    /// denominator follows tinygrad's `max(n - correction, 0)` contract and
    /// therefore preserves IEEE division results (NaN for an empty numerator,
    /// infinity for a nonzero numerator).
    pub fn var(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: usize,
    ) -> Result<NodeId> {
        let plan = VariancePlan::new(self, input, axes, correction)?;
        self.var_with_plan(input, &plan, keepdim)
    }

    /// Computes variance and mean from the same public reduction parameters.
    pub fn var_mean(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: usize,
    ) -> Result<(NodeId, NodeId)> {
        let plan = VariancePlan::new(self, input, axes, correction)?;
        let variance = self.var_with_plan(input, &plan, keepdim)?;
        let mean = self.reduce(input, ReduceKind::Mean, Some(plan.signed_axes()), keepdim)?;
        Ok((variance, mean))
    }

    /// Computes the square root of [`Graph::var`].
    pub fn std(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: usize,
    ) -> Result<NodeId> {
        let variance = self.var(input, axes, keepdim, correction)?;
        self.sqrt(variance)
    }

    /// Computes standard deviation and mean from the same public parameters.
    pub fn std_mean(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        correction: usize,
    ) -> Result<(NodeId, NodeId)> {
        let (variance, mean) = self.var_mean(input, axes, keepdim, correction)?;
        Ok((self.sqrt(variance)?, mean))
    }

    fn var_with_plan(
        &mut self,
        input: NodeId,
        plan: &VariancePlan,
        keepdim: bool,
    ) -> Result<NodeId> {
        let axes = plan.signed_axes();
        let mean = self.reduce(input, ReduceKind::Mean, Some(axes.clone()), true)?;
        let centered = self.sub(input, mean)?;
        let squares = self.square(centered)?;
        // tinygrad explicitly casts squared residuals to sum_acc_dtype before
        // summing.  This is deliberately independent from sum's public result
        // dtype (notably for F16/BF16).
        let numerator = self.reduce(
            self.cast(squares, plan.accumulator_dtype)?,
            ReduceKind::Sum,
            Some(axes),
            keepdim,
        )?;
        let divisor = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(plan.denominator as f64),
            plan.accumulator_dtype,
        ));
        let variance = self.div(numerator, divisor)?;
        if plan.output_dtype == plan.accumulator_dtype {
            Ok(variance)
        } else {
            self.cast(variance, plan.output_dtype)
        }
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
        let sum = self.sum_all(input)?;
        let divisor = self.constant(TensorData::scalar(shape.numel()? as f32));
        self.div(sum, divisor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, Scalar, Shape};
    use std::collections::HashMap;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    fn typed_data(shape: impl Into<Shape>, dtype: DType, values: &[f64]) -> TensorData {
        TensorData::from_scalars(
            shape,
            dtype,
            values.iter().copied().map(Scalar::F),
        )
        .unwrap()
    }

    #[test]
    fn variance_family_matches_tinygrad_axis_keepdim_and_dtype_contract() {
        struct Case {
            axes: Option<Vec<isize>>,
            keepdim: bool,
            correction: usize,
            shape: Shape,
            expected: Vec<f64>,
        }

        let cases = [
            Case {
                axes: Some(vec![-1]),
                keepdim: false,
                correction: 1,
                shape: Shape::new(vec![2]),
                expected: vec![1.0, 1.0],
            },
            Case {
                axes: Some(vec![0]),
                keepdim: false,
                correction: 1,
                shape: Shape::new(vec![3]),
                expected: vec![4.5, 4.5, 4.5],
            },
            Case {
                axes: Some(vec![0, 1]),
                keepdim: true,
                correction: 0,
                shape: Shape::new(vec![1, 1]),
                expected: vec![35.0 / 12.0],
            },
        ];

        for case in cases {
            let mut graph = Graph::new();
            let x = graph.input("x", [2, 3]);
            let variance = graph
                .var(x, case.axes.clone(), case.keepdim, case.correction)
                .unwrap();
            let standard_deviation = graph
                .std(x, case.axes, case.keepdim, case.correction)
                .unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                data([2, 3], &[1., 2., 3., 4., 5., 6.]),
            )]);

            let actual = CpuBackend.execute(&graph, variance, &inputs).unwrap();
            assert_eq!(actual.shape(), &case.shape);
            assert_eq!(actual.dtype(), DType::F32);
            for (actual, expected) in actual.to_vec_f64().iter().zip(&case.expected) {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
            let actual_std = CpuBackend
                .execute(&graph, standard_deviation, &inputs)
                .unwrap()
                .to_vec_f64();
            for (actual, expected) in actual_std.iter().zip(&case.expected) {
                assert!((actual - expected.sqrt()).abs() < 1e-6, "{actual} != sqrt({expected})");
            }
        }

        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2, 3], DType::I32);
        let (variance, mean) = graph.var_mean(x, Some(vec![-1]), false, 1).unwrap();
        assert_eq!(graph.dtype(variance).unwrap(), DType::F32);
        assert_eq!(graph.dtype(mean).unwrap(), DType::F32);
        let inputs = HashMap::from([(
            "x".into(),
            typed_data([2, 3], DType::I32, &[1., 2., 3., 4., 5., 6.]),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, variance, &inputs).unwrap(),
            data([2], &[1., 1.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, mean, &inputs).unwrap(),
            data([2], &[2., 5.])
        );

        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let (standard_deviation, mean) = graph.std_mean(x, Some(vec![-1]), false, 1).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            data([2, 3], &[1., 2., 3., 4., 5., 6.]),
        )]);
        assert_eq!(
            CpuBackend
                .execute(&graph, standard_deviation, &inputs)
                .unwrap(),
            data([2], &[1., 1.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, mean, &inputs).unwrap(),
            data([2], &[2., 5.])
        );
    }

    #[test]
    fn variance_family_preserves_tinygrad_scalar_and_empty_ieee_cases() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let population = graph.var(scalar, None, false, 0).unwrap();
        let sample = graph.var(scalar, None, false, 1).unwrap();
        let inputs = HashMap::from([("scalar".into(), TensorData::scalar(4.0))]);
        assert_eq!(
            CpuBackend.execute(&graph, population, &inputs).unwrap(),
            TensorData::scalar(0.0)
        );
        assert!(CpuBackend
            .execute(&graph, sample, &inputs)
            .unwrap()
            .to_vec_f64()[0]
            .is_nan());

        let mut graph = Graph::new();
        let empty = graph.input("empty", [2, 0]);
        let variance = graph.var(empty, Some(vec![-1]), false, 1).unwrap();
        let actual = CpuBackend
            .execute(
                &graph,
                variance,
                &HashMap::from([("empty".into(), data([2, 0], &[]))]),
            )
            .unwrap();
        assert_eq!(actual.shape(), &Shape::new(vec![2]));
        assert!(actual.to_vec_f64().iter().all(|value| value.is_nan()));

        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let variance = graph.var(x, None, false, 3).unwrap();
        let inputs = HashMap::from([("x".into(), data([2], &[1., 3.]))]);
        assert!(CpuBackend
            .execute(&graph, variance, &inputs)
            .unwrap()
            .to_vec_f64()[0]
            .is_infinite());

        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let variance = graph.var(x, None, false, 0).unwrap();
        let inputs = HashMap::from([("x".into(), data([2], &[f32::NAN, 3.]))]);
        assert!(CpuBackend
            .execute(&graph, variance, &inputs)
            .unwrap()
            .to_vec_f64()[0]
            .is_nan());

        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F16);
        let variance = graph.var(x, None, false, 0).unwrap();
        assert_eq!(graph.dtype(variance).unwrap(), DType::F16);
        let inputs = HashMap::from([(
            "x".into(),
            typed_data([2], DType::F16, &[1., 3.]),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, variance, &inputs).unwrap().to_vec_f64(),
            vec![1.0]
        );
    }

    #[test]
    fn variance_validates_before_graph_mutation_and_trace_exposes_components() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let before = graph.trace(x).unwrap();
        assert_eq!(
            graph.var(x, Some(vec![1, -1]), false, 1),
            Err(Error::InvalidReductionAxes {
                node: x,
                axes: vec![1, 1],
                rank: 2,
            })
        );
        assert_eq!(graph.trace(x).unwrap(), before);

        let float8 = graph.input_dtype("float8", [2], DType::F8E4M3);
        let before = graph.trace(float8).unwrap();
        assert_eq!(
            graph.var(float8, None, false, 1),
            Err(Error::InvalidElementwiseDType {
                op: "var",
                actual: DType::F8E4M3,
            })
        );
        assert_eq!(graph.trace(float8).unwrap(), before);

        let variance = graph.var(x, Some(vec![-1]), true, 1).unwrap();
        let trace = graph.trace(variance).unwrap().to_string();
        assert!(trace.contains("Mean(%"));
        assert!(trace.contains("square(%"));
        assert!(trace.contains("Sum(%"));
        assert!(trace.contains("div(%"));
        assert!(trace.contains("[2, 1] F32"));
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
}
