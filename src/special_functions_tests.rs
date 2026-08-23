use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape, TensorData};
use std::collections::HashMap;

type UnaryGraphOp = fn(&mut Graph, crate::NodeId) -> crate::Result<crate::NodeId>;
type UnaryCase = (&'static str, UnaryGraphOp, &'static [f64], &'static [f64]);

fn typed(dtype: DType, values: &[f64]) -> TensorData {
    TensorData::from_scalars(
        Shape::new([values.len()]),
        dtype,
        values.iter().copied().map(Scalar::F),
    )
    .unwrap()
}

fn execute(graph: &Graph, output: crate::NodeId, dtype: DType, values: &[f64]) -> TensorData {
    CpuBackend
        .execute(
            graph,
            output,
            &HashMap::from([("x".into(), typed(dtype, values))]),
        )
        .unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected} (tol {tolerance})"
    );
}

#[test]
fn special_unaries_have_typed_known_values_and_domain_behavior() {
    let cases: &[UnaryCase] = &[
        (
            "erf",
            Graph::erf,
            &[-1., 0., 1.],
            &[-0.84270079, 0., 0.84270079],
        ),
        (
            "erfc",
            Graph::erfc,
            &[-1., 0., 1.],
            &[1.84270079, 1., 0.15729921],
        ),
        (
            "asin",
            Graph::asin,
            &[-1., 0., 1.],
            &[
                -std::f64::consts::FRAC_PI_2,
                0.,
                std::f64::consts::FRAC_PI_2,
            ],
        ),
        (
            "acos",
            Graph::acos,
            &[-1., 0., 1.],
            &[std::f64::consts::PI, std::f64::consts::FRAC_PI_2, 0.],
        ),
        (
            "atan",
            Graph::atan,
            &[-1., 0., 1.],
            &[
                -std::f64::consts::FRAC_PI_4,
                0.,
                std::f64::consts::FRAC_PI_4,
            ],
        ),
        (
            "asinh",
            Graph::asinh,
            &[-1., 0., 1.],
            &[-0.88137359, 0., 0.88137359],
        ),
        (
            "acosh",
            Graph::acosh,
            &[1., 2., 3.],
            &[0., 1.3169579, 1.7627472],
        ),
        (
            "atanh",
            Graph::atanh,
            &[-0.5, 0., 0.5],
            &[-0.54930614, 0., 0.54930614],
        ),
    ];
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        for &(name, operation, values, expected) in cases {
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", [values.len()], dtype);
            let output = operation(&mut graph, x).unwrap();
            assert_eq!(graph.dtype(output).unwrap(), dtype, "{name} {dtype:?}");
            for (actual, expected) in execute(&graph, output, dtype, values)
                .to_vec_f64()
                .into_iter()
                .zip(expected)
            {
                close(
                    actual,
                    *expected,
                    if dtype == DType::F16 || dtype == DType::BF16 {
                        0.02
                    } else {
                        2e-5
                    },
                );
            }
        }
    }
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [3], DType::F32);
    let asin = graph.asin(x).unwrap();
    let acosh = graph.acosh(x).unwrap();
    let atanh = graph.atanh(x).unwrap();
    assert!(execute(&graph, asin, DType::F32, &[2., 0., -2.]).to_vec_f64()[0].is_nan());
    assert!(execute(&graph, acosh, DType::F32, &[0., 1., 2.]).to_vec_f64()[0].is_nan());
    assert!(execute(&graph, atanh, DType::F32, &[2., 0., -2.]).to_vec_f64()[0].is_nan());
}

#[test]
fn special_functions_promote_non_floats_and_atan2_broadcasts_quadrants() {
    for dtype in [DType::Bool, DType::I32, DType::U64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], dtype);
        let output = graph.erf(x).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        let angle = graph.atan2(x, x).unwrap();
        assert_eq!(graph.dtype(angle).unwrap(), DType::F32);
    }
    let mut graph = Graph::new();
    let magnitude = graph.input_dtype("x", [2], DType::I32);
    let sign = graph.constant(
        TensorData::from_scalars([2], DType::I32, [Scalar::I(-1), Scalar::I(1)]).unwrap(),
    );
    let signed = graph.copysign(magnitude, sign).unwrap();
    assert_eq!(graph.dtype(signed).unwrap(), DType::I32);
    assert_eq!(
        execute(&graph, signed, DType::I32, &[2., -3.]).to_vec_f64(),
        vec![-2., 3.]
    );
    let mut graph = Graph::new();
    let y = graph.input("x", [2, 1]);
    let x = graph.constant(TensorData::new([2], vec![1.0f32, -1.0]).unwrap());
    let output = graph.atan2(y, x).unwrap();
    let result = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 1], vec![1.0f32, -1.0]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(
        graph.trace(output).unwrap().steps.last().unwrap().operation,
        "atan2(%0, %1)"
    );
    let values = result.to_vec_f64();
    close(values[0], std::f64::consts::FRAC_PI_4, 1e-6);
    close(values[1], 3.0 * std::f64::consts::FRAC_PI_4, 1e-6);
    close(values[2], -std::f64::consts::FRAC_PI_4, 1e-6);
    close(values[3], -3.0 * std::f64::consts::FRAC_PI_4, 1e-6);
}

#[test]
fn copysign_preserves_tinygrad_signed_zero_and_nan_contract() {
    let mut graph = Graph::new();
    let magnitude = graph.input("x", [4]);
    let sign = graph.constant(TensorData::new([4], vec![-0.0f32, 0.0, f32::NAN, -1.0]).unwrap());
    let output = graph.copysign(magnitude, sign).unwrap();
    let values = execute(&graph, output, DType::F32, &[2., -2., -0., f64::NAN]).to_vec_f64();
    assert!(values[0].is_sign_negative() && values[0] == 0.0 + -2.0);
    assert_eq!(values[1], 2.0);
    assert_eq!(values[2], 0.0); // NaN sign is positive under tinygrad's predicate contract.
    assert!(values[3].is_nan());
}

#[test]
fn special_function_gradients_match_central_differences() {
    let operations: &[UnaryGraphOp] = &[
        Graph::erf,
        Graph::erfc,
        Graph::asin,
        Graph::acos,
        Graph::atan,
        Graph::asinh,
        Graph::acosh,
        Graph::atanh,
    ];
    let points = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.4, 0.2];
    for (&operation, &point) in operations.iter().zip(&points) {
        let mut graph = Graph::new();
        let x = graph.input("x", [1]);
        let value = operation(&mut graph, x).unwrap();
        let output = graph.sum(value, 0).unwrap();
        let gradient = graph.grad(output, x).unwrap();
        let analytic = execute(&graph, gradient, DType::F32, &[point]).to_vec_f64()[0];
        let epsilon = 1e-3;
        let plus = execute(&graph, output, DType::F32, &[point + epsilon]).to_vec_f64()[0];
        let minus = execute(&graph, output, DType::F32, &[point - epsilon]).to_vec_f64()[0];
        close(analytic, (plus - minus) / (2.0 * epsilon), 2e-3);
    }
    let mut graph = Graph::new();
    let y = graph.input("x", []);
    let x = graph.input("z", []);
    let output = graph.atan2(y, x).unwrap();
    let dy = graph.grad(output, y).unwrap();
    let dx = graph.grad(output, x).unwrap();
    let inputs = HashMap::from([
        ("x".into(), TensorData::scalar(2.0f32)),
        ("z".into(), TensorData::scalar(3.0f32)),
    ]);
    close(
        CpuBackend
            .execute(&graph, dy, &inputs)
            .unwrap()
            .to_vec_f64()[0],
        3.0 / 13.0,
        1e-6,
    );
    close(
        CpuBackend
            .execute(&graph, dx, &inputs)
            .unwrap()
            .to_vec_f64()[0],
        -2.0 / 13.0,
        1e-6,
    );
}

#[test]
fn gelu_modes_are_distinct_and_exact_mode_uses_erf() {
    let mut graph = Graph::new();
    let x = graph.input("x", [1]);
    let exact = graph.gelu(x, "none").unwrap();
    let tanh = graph.gelu(x, "tanh").unwrap();
    let input = TensorData::new([1], vec![1.0f32]).unwrap();
    let inputs = HashMap::from([("x".into(), input)]);
    close(
        CpuBackend
            .execute(&graph, exact, &inputs)
            .unwrap()
            .to_vec_f64()[0],
        0.84134475,
        2e-5,
    );
    assert!(
        (CpuBackend
            .execute(&graph, tanh, &inputs)
            .unwrap()
            .to_vec_f64()[0]
            - 0.84134475)
            .abs()
            < 3e-4
    );
}
