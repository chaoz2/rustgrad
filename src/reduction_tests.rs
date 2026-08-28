use crate::{
    Backend, CpuBackend, DType, Error, Graph, ReduceKind, Scalar, Shape, Storage, TensorData,
};
use std::collections::HashMap;

fn f32_data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
    TensorData::new(shape, values.to_vec()).unwrap()
}

fn typed_empty(dtype: DType) -> TensorData {
    TensorData::from_scalars([2, 0], dtype, []).unwrap()
}

fn execute(graph: &Graph, output: crate::NodeId, input: TensorData) -> TensorData {
    CpuBackend
        .execute(graph, output, &HashMap::from([("x".into(), input)]))
        .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 2e-4,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn arg_reduce_uses_typed_integer_ordering_and_preserves_float_first_ties() {
    let assert_index = |dtype, data: Vec<Scalar>, max, expected| {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [data.len()], dtype);
        let output = if max {
            graph.argmax(x, Some(0), false)
        } else {
            graph.argmin(x, Some(0), false)
        }
        .unwrap();
        let input = TensorData::from_scalars([data.len()], dtype, data).unwrap();
        assert_eq!(execute(&graph, output, input).scalar_at(0).as_i64(), expected);
    };

    // These adjacent representable integer lanes collapse to equal f64s, but
    // ArgMax/ArgMin must still observe their native I64/U64 order.
    let two_to_53 = 1_i64 << 53;
    assert_index(
        DType::I64,
        vec![Scalar::I(two_to_53), Scalar::I(two_to_53 + 1)],
        true,
        1,
    );
    assert_index(
        DType::I64,
        vec![Scalar::I(-two_to_53), Scalar::I(-two_to_53 - 1)],
        false,
        1,
    );
    assert_index(
        DType::U64,
        vec![Scalar::U(two_to_53 as u64), Scalar::U(two_to_53 as u64 + 1)],
        true,
        1,
    );
    assert_index(
        DType::U64,
        vec![Scalar::U(two_to_53 as u64 + 1), Scalar::U(two_to_53 as u64)],
        false,
        1,
    );
    assert_index(
        DType::I64,
        vec![Scalar::I(i64::MIN), Scalar::I(i64::MAX)],
        true,
        1,
    );
    assert_index(
        DType::I64,
        vec![Scalar::I(i64::MIN), Scalar::I(i64::MAX)],
        false,
        0,
    );
    assert_index(
        DType::U64,
        vec![Scalar::U(0), Scalar::U(u64::MAX)],
        true,
        1,
    );
    assert_index(
        DType::U64,
        vec![Scalar::U(0), Scalar::U(u64::MAX)],
        false,
        0,
    );

    // Float behavior stays deliberately strict/partial: both zero signs are
    // equal first ties, while leading and later NaNs retain the existing CPU
    // ArgReduce positions.
    for max in [true, false] {
        assert_index(DType::F64, vec![Scalar::F(-0.0), Scalar::F(0.0)], max, 0);
        assert_index(
            DType::F64,
            vec![Scalar::F(f64::NAN), Scalar::F(3.0)],
            max,
            0,
        );
        assert_index(
            DType::F64,
            vec![Scalar::F(3.0), Scalar::F(f64::NAN)],
            max,
            0,
        );
    }
}

#[test]
fn extrema_ignore_nan_in_every_position_and_exclude_it_from_gradients() {
    for kind in [ReduceKind::Max, ReduceKind::Min] {
        for nan_index in 0..3 {
            let mut graph = Graph::new();
            let x = graph.input("x", [3]);
            let reduced = graph.reduce(x, kind, None, false).unwrap();
            let gradient = graph.grad(reduced, x).unwrap();
            let mut values = vec![1.0, 3.0, -2.0];
            values[nan_index] = f32::NAN;
            let input = f32_data([3], &values);
            let forward = execute(&graph, reduced, input.clone())
                .scalar_at(0)
                .as_f64();
            let expected = match kind {
                ReduceKind::Max => {
                    if nan_index == 1 {
                        1.
                    } else {
                        3.
                    }
                }
                ReduceKind::Min => {
                    if nan_index == 2 {
                        1.
                    } else {
                        -2.
                    }
                }
                _ => unreachable!(),
            };
            assert_eq!(forward, expected, "{kind:?} with NaN at {nan_index}");
            let gradients = execute(&graph, gradient, input).to_vec_f64();
            assert_eq!(gradients[nan_index], 0.);
            assert_eq!(gradients.iter().filter(|&&value| value == 1.).count(), 1);
        }
    }
}

#[test]
fn extrema_split_ties_evenly_including_all_equal_groups() {
    let mut graph = Graph::new();
    let x = graph.input("x", [3]);
    let maximum = graph.reduce(x, ReduceKind::Max, None, false).unwrap();
    let minimum = graph.reduce(x, ReduceKind::Min, None, false).unwrap();
    let max_gradient = graph.grad(maximum, x).unwrap();
    let min_gradient = graph.grad(minimum, x).unwrap();

    assert_close(
        &execute(&graph, max_gradient, f32_data([3], &[4., 4., 1.])).to_vec_f64(),
        &[0.5, 0.5, 0.0],
    );
    assert_close(
        &execute(&graph, min_gradient, f32_data([3], &[2., 2., 2.])).to_vec_f64(),
        &[1.0 / 3.0; 3],
    );
}

#[test]
fn product_gradient_handles_zero_counts() {
    for (input, expected) in [
        (vec![2., 3., 4.], vec![12., 8., 6.]),
        (vec![2., 0., 4.], vec![0., 8., 0.]),
        (vec![2., 0., 0.], vec![0., 0., 0.]),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let product = graph.reduce(x, ReduceKind::Product, None, false).unwrap();
        let gradient = graph.grad(product, x).unwrap();
        assert_close(
            &execute(&graph, gradient, f32_data([3], &input)).to_vec_f64(),
            &expected,
        );
    }
}

#[test]
fn reduction_gradients_cover_multi_axis_and_keepdim() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2, 2]);
    let product = graph
        .reduce(x, ReduceKind::Product, Some(vec![0, 2]), true)
        .unwrap();
    let product_loss = graph.reduce(product, ReduceKind::Sum, None, false).unwrap();
    let maximum = graph
        .reduce(x, ReduceKind::Max, Some(vec![0, 2]), true)
        .unwrap();
    let maximum_loss = graph.reduce(maximum, ReduceKind::Sum, None, false).unwrap();
    let product_gradient = graph.grad(product_loss, x).unwrap();
    let maximum_gradient = graph.grad(maximum_loss, x).unwrap();
    let input = f32_data([2, 2, 2], &[2., 0., 3., 4., 5., 6., 7., 8.]);

    assert_eq!(graph.shape(product).unwrap(), &Shape::new([1, 2, 1]));
    assert_close(
        &execute(&graph, product_gradient, input.clone()).to_vec_f64(),
        &[0., 60., 224., 168., 0., 0., 96., 84.],
    );
    assert_close(
        &execute(&graph, maximum_gradient, input).to_vec_f64(),
        &[0., 0., 0., 0., 0., 1., 0., 1.],
    );
}

#[test]
fn empty_reduction_contract_is_typed_and_explicit() {
    for dtype in [DType::F32, DType::Bool, DType::I32, DType::U32] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2, 0], dtype);
        let sum = graph
            .reduce(x, ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let mean = graph
            .reduce(x, ReduceKind::Mean, Some(vec![1]), false)
            .unwrap();
        let product = graph
            .reduce(x, ReduceKind::Product, Some(vec![1]), false)
            .unwrap();
        let input = typed_empty(dtype);
        assert_eq!(
            execute(&graph, sum, input.clone()).to_vec_f64(),
            vec![0.; 2]
        );
        assert!(
            execute(&graph, mean, input.clone())
                .to_vec_f64()
                .iter()
                .all(|value| value.is_nan())
        );
        assert_eq!(execute(&graph, product, input).to_vec_f64(), vec![1.; 2]);

        for (kind, op) in [(ReduceKind::Max, "max"), (ReduceKind::Min, "min")] {
            assert_eq!(
                graph.reduce(x, kind, Some(vec![1]), false),
                Err(Error::EmptyReduction {
                    op,
                    shape: Shape::new([2, 0]),
                    axes: vec![1]
                })
            );
        }
        for (max, op) in [(true, "argmax"), (false, "argmin")] {
            let result = if max {
                graph.argmax(x, Some(1), false)
            } else {
                graph.argmin(x, Some(1), false)
            };
            assert_eq!(
                result,
                Err(Error::EmptyReduction {
                    op,
                    shape: Shape::new([2, 0]),
                    axes: vec![1]
                })
            );
        }
    }
}

#[test]
fn finite_difference_matches_product_and_unique_extrema_gradients() {
    for (kind, values) in [
        (ReduceKind::Product, vec![1.2, -0.7, 2.3]),
        (ReduceKind::Max, vec![1.2, -0.7, 2.3]),
        (ReduceKind::Min, vec![1.2, -0.7, 2.3]),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let loss = graph.reduce(x, kind, None, false).unwrap();
        let gradient = graph.grad(loss, x).unwrap();
        let analytic = execute(&graph, gradient, f32_data([3], &values)).to_vec_f64();
        let epsilon = 1e-3f32;
        let numeric = (0..values.len())
            .map(|index| {
                let mut positive = values.clone();
                let mut negative = values.clone();
                positive[index] += epsilon;
                negative[index] -= epsilon;
                (execute(&graph, loss, f32_data([3], &positive))
                    .scalar_at(0)
                    .as_f64()
                    - execute(&graph, loss, f32_data([3], &negative))
                        .scalar_at(0)
                        .as_f64())
                    / f64::from(2. * epsilon)
            })
            .collect::<Vec<_>>();
        assert_close(&analytic, &numeric);
    }
}

#[test]
fn reduce_grad_has_inspectable_label_shape_and_dtype() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let product = graph
        .reduce(x, ReduceKind::Product, Some(vec![1]), false)
        .unwrap();
    let loss = graph.reduce(product, ReduceKind::Sum, None, false).unwrap();
    let gradient = graph.grad(loss, x).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([2, 2]));
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F32);
    let trace = graph.trace(gradient).unwrap();
    assert!(
        trace
            .steps
            .iter()
            .any(|step| step.operation.contains("reduce_grad_Product"))
    );
}

#[test]
fn empty_unreduced_output_has_no_invalid_extrema_domain() {
    let mut graph = Graph::new();
    let x = graph.input("x", [0, 2]);
    let maximum = graph
        .reduce(x, ReduceKind::Max, Some(vec![1]), false)
        .unwrap();
    assert_eq!(
        execute(&graph, maximum, f32_data([0, 2], &[])).storage(),
        &Storage::F32(vec![])
    );
}
