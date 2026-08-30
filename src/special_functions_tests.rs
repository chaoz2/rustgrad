use crate::{Backend, BinaryOp, CompareOp, CpuBackend, DType, Graph, Op, Scalar, Shape, TensorData, UnaryOp};
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
fn copysign_is_the_source_literal_predicate_reciprocal_select_graph() {
    // Tensor.copysign is not raw BinaryOp::Copysign: source commits both
    // values to one `_broadcasted` dtype, detects -0 through reciprocal, and
    // selects between the typed abs branches. Keep the structural acceptance
    // separate from the raw binary primitive retained for lower-level users.
    let mut graph = Graph::new();
    let magnitude = graph.input_dtype("magnitude", [2, 1], DType::I64);
    let sign = graph.input_dtype("sign", [3], DType::U64);
    let output = graph.copysign(magnitude, sign).unwrap();

    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    let operations: Vec<_> = graph
        .trace(output)
        .unwrap()
        .steps
        .into_iter()
        .map(|step| step.operation)
        .collect();
    assert!(operations.iter().any(|operation| operation.starts_with("reciprocal(")));
    assert!(operations.iter().any(|operation| operation.starts_with("logical_or(")));
    assert!(operations.iter().any(|operation| operation.starts_with("sign(")));
    assert!(operations.iter().any(|operation| operation.starts_with("neg(")));
    assert!(!operations.iter().any(|operation| operation.starts_with("copysign(")));
}

#[test]
fn copysign_plan_keeps_scalar_and_empty_descriptors_concrete() {
    let mut scalar = Graph::new();
    let magnitude = scalar.input_dtype("magnitude", [], DType::BF16);
    let sign = scalar.input_dtype("sign", [], DType::BF16);
    let output = scalar.copysign(magnitude, sign).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(scalar.dtype(output).unwrap(), DType::BF16);

    let mut empty = Graph::new();
    let magnitude = empty.input_dtype("magnitude", [0, 1], DType::F16);
    let sign = empty.input_dtype("sign", [3], DType::F16);
    let output = empty.copysign(magnitude, sign).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);
    assert!(matches!(empty.op(output).unwrap(), Op::Select { .. }));
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
    let exact_loss = graph.sum_all(exact).unwrap();
    let exact_gradient = graph.grad(exact_loss, x).unwrap();
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
    assert_eq!(graph.dtype(exact).unwrap(), DType::F32);
    close(
        CpuBackend
            .execute(&graph, exact_gradient, &inputs)
            .unwrap()
            .to_vec_f64()[0],
        0.84134475 + (-0.5f64).exp() / (2.0 * std::f64::consts::PI).sqrt(),
        2e-4,
    );

    let mut extreme = Graph::new();
    let x = extreme.input("x", [2]);
    let output = extreme.gelu(x, "none").unwrap();
    let values = execute(&extreme, output, DType::F32, &[f64::INFINITY, f64::NAN]).to_vec_f64();
    assert!(values[0].is_infinite() && values[0].is_sign_positive());
    assert!(values[1].is_nan());

    let mut empty = Graph::new();
    let x = empty.input("x", [0]);
    let output = empty.gelu(x, "tanh").unwrap();
    assert_eq!(empty.shape(output).unwrap(), &crate::Shape::new([0]));
    assert_eq!(
        execute(&empty, output, DType::F32, &[]).to_vec_f64(),
        Vec::<f64>::new()
    );

    let mut malformed = Graph::new();
    let node_count = malformed.node_count();
    assert!(matches!(
        malformed.gelu(crate::NodeId(usize::MAX), "tanh"),
        Err(crate::Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), node_count);
}

#[test]
fn gelu_uses_source_width_pow_erf_and_compositional_tanh() {
    for mode in ["none", "tanh"] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [5], DType::F64);
        let output = graph.gelu(input, mode).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F64);
        let loss = graph.sum_all(output).unwrap();
        let input_gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [5],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert!(values.scalar_at(0).as_f64().is_nan());
        assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
        assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
        assert!(values.scalar_at(3).as_f64().is_nan());
        assert_eq!(values.scalar_at(4).as_f64(), f64::INFINITY);
        let gradient = CpuBackend
            .execute(&graph, input_gradient, &bindings)
            .unwrap()
            .to_vec_f64();
        close(gradient[1], 0.5, 1e-12);
        close(gradient[2], 0.5, 1e-12);

        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let mut narrow = Graph::new();
            let x = narrow.input_dtype("x", [], dtype);
            let output = narrow.gelu(x, mode).unwrap();
            assert_eq!(narrow.dtype(output).unwrap(), dtype);
            assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
        }
    }

    let mut nonfloat = Graph::new();
    let x = nonfloat.input_dtype("x", [0], DType::I32);
    let output = nonfloat.gelu(x, "tanh").unwrap();
    assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
    assert_eq!(nonfloat.shape(output).unwrap(), &Shape::new([0]));

    let mut malformed = Graph::new();
    let x = malformed.input("x", [1]);
    let nodes = malformed.node_count();
    assert!(malformed.gelu(x, "unsupported").is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn elu_uses_strict_source_relu_branches_and_live_alpha_promotion() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [6], DType::F64);
    let alpha = graph.input_dtype("alpha", [], DType::F64);
    let output = graph.elu(input, alpha).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([
        (
            "x".into(),
            TensorData::from_scalars(
                [6],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-1.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "alpha".into(),
            TensorData::scalar_with_dtype(Scalar::F(1.5), DType::F64),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), -1.5);
    close(values.scalar_at(1).as_f64(), 1.5 * ((-1.0f64).exp() - 1.0), 1e-12);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(4).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(5).as_f64().is_infinite());
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(gradient[0], 0.0);
    close(gradient[1], 1.5 * (-1.0f64).exp(), 1e-12);
    assert_eq!(gradient[2], 0.0);
    assert_eq!(gradient[3], 0.0);
    assert_eq!(gradient[4], 0.0);
    assert_eq!(gradient[5], 1.0);

    let mut infinite_alpha = Graph::new();
    let x = infinite_alpha.input_dtype("x", [], DType::F64);
    let alpha = infinite_alpha.constant(TensorData::scalar_with_dtype(
        Scalar::F(f64::INFINITY),
        DType::F64,
    ));
    let output = infinite_alpha.elu(x, alpha).unwrap();
    let result = CpuBackend
        .execute(
            &infinite_alpha,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F64),
            )]),
        )
        .unwrap();
    assert!(result.scalar_at(0).as_f64().is_nan());

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let alpha = narrow.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), dtype));
        let output = narrow.elu(x, alpha).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    let mut exact = Graph::new();
    let x = exact.input_dtype("x", [0], DType::I32);
    let alpha = exact.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
    let output = exact.elu(x, alpha).unwrap();
    assert_eq!(exact.dtype(output).unwrap(), DType::F32);
    assert_eq!(exact.shape(output).unwrap(), &Shape::new([0]));
}

#[test]
fn elu_scalar_matches_tinygrad_weak_default_without_changing_live_alpha_api() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [5], DType::F64);
    let output = graph.elu_default(input).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([5]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.op, Op::Constant(data)
            if data.dtype() == DType::F64 && data.shape() == &Shape::new([])
              && data.scalar_at(0).as_f64().to_bits() == 1.0f64.to_bits())
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [5],
                    DType::F64,
                    [
                        Scalar::F(-1.0),
                        Scalar::F(-0.0),
                        Scalar::F(f64::NAN),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(2.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    close(values.scalar_at(0).as_f64(), (-1.0f64).exp() - 1.0, 1e-12);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(3).as_f64().is_infinite());
    assert_eq!(values.scalar_at(4).as_f64(), 2.0);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let output = narrow.elu_scalar(x, 0.125).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert!(narrow.nodes.iter().any(|node| {
            matches!(&node.op, Op::Constant(data)
                if data.dtype() == dtype && data.shape() == &Shape::new([]))
        }));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut nonfloat = Graph::new();
        let x = nonfloat.input_dtype("x", [], dtype);
        let output = nonfloat.elu_scalar(x, 0.125).unwrap();
        assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
        assert!(nonfloat.nodes.iter().any(|node| {
            matches!(&node.op, Op::Constant(data) if data.dtype() == DType::F32)
        }));
    }

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::BF16);
    let output = empty.elu_scalar(x, -0.5).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed.elu_default(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX], DType::F32);
    let before = malformed.node_count();
    assert!(malformed.elu_default(overflow).is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn elu_with_scalar_preserves_tinygrad_untyped_alpha_staging() {
    for (input_dtype, alpha, expected_dtype) in [
        (DType::Bool, Scalar::Bool(true), DType::F32),
        (DType::Bool, Scalar::I(-1), DType::F32),
        (DType::I8, Scalar::U(1), DType::F32),
        (DType::I16, Scalar::Bool(true), DType::F32),
        (DType::I32, Scalar::F(-0.5), DType::F32),
        (DType::I64, Scalar::U(1), DType::F32),
        (DType::U8, Scalar::I(-1), DType::F32),
        (DType::U16, Scalar::Bool(false), DType::F32),
        (DType::U32, Scalar::F(0.5), DType::F32),
        (DType::U64, Scalar::I(1), DType::F32),
        (DType::F16, Scalar::Bool(true), DType::F16),
        (DType::BF16, Scalar::I(-1), DType::BF16),
        (DType::F32, Scalar::U(1), DType::F32),
        (DType::F64, Scalar::F(f64::NAN), DType::F64),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], input_dtype);
        let output = graph.elu_with_scalar(input, alpha).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), expected_dtype);
        let Op::Binary { op: BinaryOp::Add, rhs, .. } = graph.op(output).unwrap() else {
            panic!("ELU must retain its source final subtraction composition");
        };
        let Op::Unary { op: UnaryOp::Neg, input: negative } = graph.op(*rhs).unwrap() else {
            panic!("ELU final subtraction must negate its alpha-scaled branch");
        };
        let Op::Binary { op: BinaryOp::Mul, lhs, .. } = graph.op(*negative).unwrap() else {
            panic!("ELU negative branch must retain alpha-left Mul");
        };
        assert!(matches!(graph.op(*lhs).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == expected_dtype));
    }

    // A live U64 alpha meets Exp's F32 result rather than creating an I64/U64
    // arithmetic bridge; concrete scalars follow that identical staging.
    let mut bridge = Graph::new();
    let x = bridge.input_dtype("x", [], DType::I64);
    let alpha = bridge.input_dtype("alpha", [], DType::U64);
    assert_eq!(bridge.dtype(bridge.elu(x, alpha).unwrap()).unwrap(), DType::F32);

    let mut scalar = Graph::new();
    let x = scalar.input_dtype("x", [], DType::F64);
    let output = scalar.elu_with_scalar(x, Scalar::F(f64::NAN)).unwrap();
    let loss = scalar.sum_all(output).unwrap();
    assert_eq!(scalar.shape(scalar.grad(loss, x).unwrap()).unwrap(), &Shape::new([]));

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::BF16);
    let output = empty.elu_with_scalar(x, Scalar::I(-1)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed
        .elu_with_scalar(crate::NodeId(usize::MAX), Scalar::Bool(true))
        .is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(malformed
        .elu_with_scalar(overflow, Scalar::F(-0.5))
        .is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn selu_uses_ge_source_branch_and_live_parameter_promotion() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [6], DType::F64);
    let alpha = graph.input_dtype("alpha", [], DType::F64);
    let gamma = graph.input_dtype("gamma", [], DType::F64);
    let output = graph.selu(input, alpha, gamma).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([
        (
            "x".into(),
            TensorData::from_scalars(
                [6],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-1.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "alpha".into(),
            TensorData::scalar_with_dtype(Scalar::F(1.5), DType::F64),
        ),
        (
            "gamma".into(),
            TensorData::scalar_with_dtype(Scalar::F(2.0), DType::F64),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), -3.0);
    close(
        values.scalar_at(1).as_f64(),
        3.0 * ((-1.0f64).exp() - 1.0),
        1e-12,
    );
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert_eq!(values.scalar_at(5).as_f64(), f64::INFINITY);
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(gradient[0], 0.0);
    close(gradient[1], 3.0 * (-1.0f64).exp(), 1e-12);
    assert_eq!(gradient[2], 2.0);
    assert_eq!(gradient[3], 2.0);
    assert!(gradient[4].is_nan());
    assert_eq!(gradient[5], 2.0);

    let mut infinite_gamma = Graph::new();
    let x = infinite_gamma.input_dtype("x", [], DType::F64);
    let alpha = infinite_gamma.constant(TensorData::scalar_with_dtype(
        Scalar::F(1.0),
        DType::F64,
    ));
    let gamma = infinite_gamma.constant(TensorData::scalar_with_dtype(
        Scalar::F(f64::INFINITY),
        DType::F64,
    ));
    let output = infinite_gamma.selu(x, alpha, gamma).unwrap();
    let result = CpuBackend
        .execute(
            &infinite_gamma,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::scalar_with_dtype(Scalar::F(-0.0), DType::F64),
            )]),
        )
        .unwrap();
    assert!(result.scalar_at(0).as_f64().is_nan());

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let alpha = narrow.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), dtype));
        let gamma = narrow.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), dtype));
        let output = narrow.selu(x, alpha, gamma).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    let mut exact = Graph::new();
    let x = exact.input_dtype("x", [0], DType::I32);
    let alpha = exact.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
    let gamma = exact.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
    let output = exact.selu(x, alpha, gamma).unwrap();
    assert_eq!(exact.dtype(output).unwrap(), DType::F32);
    assert_eq!(exact.shape(output).unwrap(), &Shape::new([0]));
}

#[test]
fn sigmoid_uses_typed_exp2_reciprocal_composition_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [5], DType::F64);
    let output = graph.sigmoid(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [5],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64(), 0.5);
    assert_eq!(values.scalar_at(2).as_f64(), 0.5);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    close(gradient[1], 0.25, 1e-12);
    close(gradient[2], 0.25, 1e-12);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let output = narrow.sigmoid(x).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    let mut nonfloat = Graph::new();
    let x = nonfloat.input_dtype("x", [0], DType::I32);
    let output = nonfloat.sigmoid(x).unwrap();
    assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
    assert_eq!(nonfloat.shape(output).unwrap(), &Shape::new([0]));

    let mut malformed = Graph::new();
    let nodes = malformed.node_count();
    assert!(matches!(
        malformed.sigmoid(crate::NodeId(usize::MAX)),
        Err(crate::Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn tanh_uses_typed_doubled_sigmoid_composition_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [5], DType::F64);
    let output = graph.tanh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [5],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), -1.0);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    close(gradient[1], 1.0, 1e-12);
    close(gradient[2], 1.0, 1e-12);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let output = narrow.tanh(x).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    let mut nonfloat = Graph::new();
    let x = nonfloat.input_dtype("x", [0], DType::I32);
    let output = nonfloat.tanh(x).unwrap();
    assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
    assert_eq!(nonfloat.shape(output).unwrap(), &Shape::new([0]));

    let mut malformed = Graph::new();
    let nodes = malformed.node_count();
    assert!(matches!(
        malformed.tanh(crate::NodeId(usize::MAX)),
        Err(crate::Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn softplus_uses_live_beta_stable_logaddexp_and_typed_reciprocal() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [5], DType::F64);
    let beta = graph.input_dtype("beta", [], DType::F64);
    let output = graph.softplus(input, beta).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([
        (
            "x".into(),
            TensorData::from_scalars(
                [5],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "beta".into(),
            TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F64),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    close(values.scalar_at(1).as_f64(), std::f64::consts::LN_2, 1e-12);
    close(values.scalar_at(2).as_f64(), std::f64::consts::LN_2, 1e-12);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), f64::INFINITY);
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    close(gradient[1], 0.5, 1e-12);
    close(gradient[2], 0.5, 1e-12);

    let mut zero_beta = Graph::new();
    let x = zero_beta.input_dtype("x", [], DType::F64);
    let beta = zero_beta.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F64));
    let output = zero_beta.softplus(x, beta).unwrap();
    let value = CpuBackend
        .execute(
            &zero_beta,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F64),
            )]),
        )
        .unwrap();
    assert_eq!(value.scalar_at(0).as_f64(), f64::INFINITY);

    let mut broadcast = Graph::new();
    let x = broadcast.input_dtype("x", [2, 3], DType::F16);
    let beta = broadcast.input_dtype("beta", [1, 3], DType::F32);
    let output = broadcast.softplus(x, beta).unwrap();
    assert_eq!(broadcast.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(broadcast.dtype(output).unwrap(), DType::F32);

    let mut nonfloat = Graph::new();
    let x = nonfloat.input_dtype("x", [0], DType::I32);
    let beta = nonfloat.input_dtype("beta", [], DType::I32);
    let output = nonfloat.softplus(x, beta).unwrap();
    assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
    assert_eq!(nonfloat.shape(output).unwrap(), &Shape::new([0]));

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2, 3]);
    let beta = malformed.input("beta", [2, 2]);
    let nodes = malformed.node_count();
    assert!(malformed.softplus(x, beta).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn softplus_scalar_and_default_preserve_tinygrad_literal_staging() {
    for (input_dtype, beta, expected_dtype) in [
        (DType::Bool, Scalar::Bool(true), DType::F32),
        (DType::I8, Scalar::I(-1), DType::F32),
        (DType::I16, Scalar::U(1), DType::F32),
        (DType::I32, Scalar::F(0.5), DType::F32),
        (DType::I64, Scalar::Bool(true), DType::F32),
        (DType::U8, Scalar::I(-1), DType::F32),
        (DType::U16, Scalar::U(1), DType::F32),
        (DType::U32, Scalar::F(-0.5), DType::F32),
        (DType::U64, Scalar::Bool(true), DType::F32),
        (DType::F16, Scalar::I(1), DType::F16),
        (DType::BF16, Scalar::Bool(true), DType::BF16),
        (DType::F32, Scalar::U(1), DType::F32),
        (DType::F64, Scalar::F(f64::NAN), DType::F64),
    ] {
        let mut graph=Graph::new(); let x=graph.input_dtype("x",[2],input_dtype);
        let output=graph.softplus_with_scalar(x,beta).unwrap();
        assert_eq!(graph.shape(output).unwrap(),&Shape::new([2])); assert_eq!(graph.dtype(output).unwrap(),expected_dtype);
        let Op::Binary { op: BinaryOp::Mul, lhs, .. }=graph.op(output).unwrap() else { panic!("Softplus needs reciprocal-left final Mul") };
        assert!(matches!(graph.op(*lhs).unwrap(),Op::Binary { op: BinaryOp::Mul, .. }));
    }
    let mut default=Graph::new(); let x=default.input_dtype("x",[],DType::F16);
    assert_eq!(default.dtype(default.softplus_default(x).unwrap()).unwrap(),DType::F16);
    let mut bridge=Graph::new(); let x=bridge.input_dtype("x",[],DType::I64); let beta=bridge.input_dtype("beta",[],DType::U64);
    assert_eq!(bridge.dtype(bridge.softplus(x,beta).unwrap()).unwrap(),DType::F32);
    let mut empty=Graph::new(); let x=empty.input_dtype("x",[0,2],DType::BF16);
    assert_eq!(empty.shape(empty.softplus_with_scalar(x,Scalar::I(1)).unwrap()).unwrap(),&Shape::new([0,2]));
    let mut malformed=Graph::new(); let before=malformed.node_count();
    assert!(malformed.softplus_with_scalar(crate::NodeId(usize::MAX),Scalar::F(1.0)).is_err()); assert_eq!(malformed.node_count(),before);
    let overflow=malformed.input_dtype("overflow",[usize::MAX,2],DType::F64); let before=malformed.node_count();
    assert!(malformed.softplus_with_scalar(overflow,Scalar::F(1.0)).is_err()); assert_eq!(malformed.node_count(),before);
}

#[test]
fn softsign_uses_literal_sign_reciprocal_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [6], DType::F64);
    let output = graph.softsign(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [6],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-1.0),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert!(values.scalar_at(0).as_f64().is_nan());
    assert_eq!(values.scalar_at(1).as_f64(), -0.5);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_nan());
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    close(gradient[2], 1.0, 1e-12);
    close(gradient[3], 1.0, 1e-12);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let output = narrow.softsign(x).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    let mut signed_min = Graph::new();
    let x = signed_min.input_dtype("x", [], DType::I64);
    let output = signed_min.softsign(x).unwrap();
    assert_eq!(signed_min.dtype(output).unwrap(), DType::F32);
    assert_eq!(signed_min.shape(output).unwrap(), &Shape::new([]));
    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0], DType::I32);
    let output = empty.softsign(x).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F32);
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));

    let mut malformed = Graph::new();
    let nodes = malformed.node_count();
    assert!(matches!(
        malformed.softsign(crate::NodeId(usize::MAX)),
        Err(crate::Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn selu_scalar_matches_tinygrad_defaults_and_preflights_before_constants() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2], DType::F64);
    let output = graph.selu_default(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
    assert_eq!(graph.nodes.iter().filter(|node| matches!(&node.op, Op::Constant(data) if data.dtype() == DType::F64)).count(), 4);
    let values = CpuBackend.execute(&graph, output, &HashMap::from([("x".into(), TensorData::from_scalars([2], DType::F64, [Scalar::F(-1.0), Scalar::F(1.0)]).unwrap())])).unwrap();
    close(values.scalar_at(0).as_f64(), 1.0507 * 1.67326 * ((-1.0f64).exp() - 1.0), 1e-12);
    close(values.scalar_at(1).as_f64(), 1.0507, 1e-12);
    for dtype in [DType::F16,DType::BF16,DType::F32,DType::F64] { let mut g=Graph::new(); let x=g.input_dtype("x", [], dtype); let out=g.selu_scalar(x, 0.125, 0.5).unwrap(); assert_eq!(g.dtype(out).unwrap(), dtype); }
    for dtype in [DType::Bool,DType::I8,DType::I16,DType::I32,DType::I64,DType::U8,DType::U16,DType::U32,DType::U64] { let mut g=Graph::new(); let x=g.input_dtype("x", [], dtype); let out=g.selu_scalar(x,0.125,0.5).unwrap(); assert_eq!(g.dtype(out).unwrap(),DType::F32); }
    let mut malformed=Graph::new(); let before=malformed.node_count(); assert!(malformed.selu_default(crate::NodeId(usize::MAX)).is_err()); assert_eq!(malformed.node_count(),before);
    let overflow=malformed.input_dtype("overflow",[usize::MAX],DType::F32); let before=malformed.node_count(); assert!(malformed.selu_default(overflow).is_err()); assert_eq!(malformed.node_count(),before);
}

#[test]
fn selu_with_scalars_preserves_tinygrad_untyped_parameter_consumers() {
    for (input_dtype, alpha, gamma, expected_dtype) in [
        (DType::Bool, Scalar::Bool(true), Scalar::I(-1), DType::F32),
        (DType::I8, Scalar::U(1), Scalar::Bool(false), DType::F32),
        (DType::I16, Scalar::Bool(true), Scalar::F(-0.5), DType::F32),
        (DType::I32, Scalar::F(-0.5), Scalar::U(1), DType::F32),
        (DType::I64, Scalar::U(1), Scalar::I(-1), DType::F32),
        (DType::U8, Scalar::I(-1), Scalar::Bool(true), DType::F32),
        (DType::U16, Scalar::Bool(false), Scalar::F(0.5), DType::F32),
        (DType::U32, Scalar::F(0.5), Scalar::I(-1), DType::F32),
        (DType::U64, Scalar::I(1), Scalar::U(1), DType::F32),
        (DType::F16, Scalar::Bool(true), Scalar::I(-1), DType::F16),
        (DType::BF16, Scalar::I(-1), Scalar::Bool(true), DType::BF16),
        (DType::F32, Scalar::U(1), Scalar::F(-0.5), DType::F32),
        (DType::F64, Scalar::F(f64::NAN), Scalar::F(f64::INFINITY), DType::F64),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], input_dtype);
        let output = graph.selu_with_scalars(input, alpha, gamma).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), expected_dtype);
        let Op::Binary { op: BinaryOp::Mul, lhs: gamma_node, rhs: branch } = graph.op(output).unwrap() else {
            panic!("SELU must retain gamma-left final Mul");
        };
        assert!(matches!(graph.op(*gamma_node).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == expected_dtype));
        let Op::Select { on_false, .. } = graph.op(*branch).unwrap() else {
            panic!("SELU must retain its inclusive predicate Select");
        };
        let Op::Binary { op: BinaryOp::Mul, lhs: alpha_node, .. } = graph.op(*on_false).unwrap() else {
            panic!("SELU false branch must retain alpha-left Mul");
        };
        assert!(matches!(graph.op(*alpha_node).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == expected_dtype));
    }

    let mut bridge = Graph::new();
    let x = bridge.input_dtype("x", [], DType::I64);
    let alpha = bridge.input_dtype("alpha", [], DType::U64);
    let gamma = bridge.input_dtype("gamma", [], DType::U64);
    assert_eq!(bridge.dtype(bridge.selu(x, alpha, gamma).unwrap()).unwrap(), DType::F32);

    let mut scalar = Graph::new();
    let x = scalar.input_dtype("x", [], DType::F64);
    let output = scalar.selu_with_scalars(x, Scalar::F(f64::NAN), Scalar::F(f64::INFINITY)).unwrap();
    let loss = scalar.sum_all(output).unwrap();
    assert_eq!(scalar.shape(scalar.grad(loss, x).unwrap()).unwrap(), &Shape::new([]));

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::BF16);
    let output = empty.selu_with_scalars(x, Scalar::I(-1), Scalar::Bool(true)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed
        .selu_with_scalars(crate::NodeId(usize::MAX), Scalar::Bool(true), Scalar::I(1))
        .is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(malformed
        .selu_with_scalars(overflow, Scalar::F(-0.5), Scalar::F(1.0))
        .is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn hardsigmoid_supports_source_defaults_and_live_strict_relu_parameters() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [7], DType::F64);
    let alpha = graph.input_dtype("alpha", [], DType::F64);
    let beta = graph.input_dtype("beta", [], DType::F64);
    let output = graph.hardsigmoid_with(input, alpha, beta).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([
        (
            "x".into(),
            TensorData::from_scalars(
                [7],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-1.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(3.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "alpha".into(),
            TensorData::scalar_with_dtype(Scalar::F(0.25), DType::F64),
        ),
        (
            "beta".into(),
            TensorData::scalar_with_dtype(Scalar::F(0.25), DType::F64),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(2).as_f64(), 0.25);
    assert_eq!(values.scalar_at(3).as_f64(), 0.25);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert_eq!(values.scalar_at(5).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(6).as_f64(), 1.0);
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(gradient[1], 0.0);
    close(gradient[2], 0.25, 1e-12);
    close(gradient[3], 0.25, 1e-12);
    close(gradient[4], 0.25, 1e-12);

    let mut defaults = Graph::new();
    let x = defaults.input_dtype("x", [], DType::F64);
    let output = defaults.hardsigmoid(x).unwrap();
    let value = CpuBackend
        .execute(
            &defaults,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F64),
            )]),
        )
        .unwrap();
    assert_eq!(value.scalar_at(0).as_f64(), 0.5);

    let mut broadcast = Graph::new();
    let x = broadcast.input_dtype("x", [2, 3], DType::F16);
    let alpha = broadcast.input_dtype("alpha", [1, 3], DType::F32);
    let beta = broadcast.input_dtype("beta", [], DType::F32);
    let output = broadcast.hardsigmoid_with(x, alpha, beta).unwrap();
    assert_eq!(broadcast.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(broadcast.dtype(output).unwrap(), DType::F32);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let output = narrow.hardsigmoid(x).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0], DType::I32);
    let output = empty.hardsigmoid(x).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F32);
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2, 3]);
    let alpha = malformed.input("alpha", [2, 2]);
    let beta = malformed.input("beta", []);
    let nodes = malformed.node_count();
    assert!(malformed.hardsigmoid_with(x, alpha, beta).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn hardswish_uses_tinygrad_strict_relu6_arithmetic() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [8], DType::F64);
    let output = graph.hardswish(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [8],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-4.0),
                Scalar::F(-3.0),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(3.0),
                Scalar::F(f64::INFINITY),
                Scalar::F(f64::NAN),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    // Unlike a clamp formulation, strict ReLU6 makes both infinities reach
    // an infinity-minus-infinity lane before the outer product.
    assert!(values.scalar_at(0).as_f64().is_nan());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(4).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(5).as_f64(), 3.0);
    assert!(values.scalar_at(6).as_f64().is_nan());
    assert!(values.scalar_at(7).as_f64().is_nan());
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(gradient[1], 0.0);
    assert_eq!(gradient[2], 0.0);
    assert_eq!(gradient[4], 0.5);
    assert_eq!(gradient[5], 1.5);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let input = narrow.input_dtype("x", [], dtype);
        let output = narrow.hardswish(input).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut promoted = Graph::new();
        let input = promoted.input_dtype("x", [], dtype);
        let output = promoted.hardswish(input).unwrap();
        assert_eq!(promoted.dtype(output).unwrap(), DType::F32);
    }
    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0], DType::F16);
    let output = empty.hardswish(input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);

    let mut malformed = Graph::new();
    let nodes = malformed.node_count();
    assert!(malformed.hardswish(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn relu6_uses_tinygrad_strict_relu_subtraction() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [7], DType::F64);
    let output = graph.relu6(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [7],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-1.0),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(6.0),
                Scalar::F(f64::INFINITY),
                Scalar::F(f64::NAN),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    for index in 0..4 {
        assert_eq!(values.scalar_at(index).as_f64().to_bits(), 0.0f64.to_bits());
    }
    assert_eq!(values.scalar_at(4).as_f64(), 6.0);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert_eq!(values.scalar_at(6).as_f64().to_bits(), 0.0f64.to_bits());
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(gradient[1], 0.0);
    assert_eq!(gradient[2], 0.0);
    assert_eq!(gradient[3], 0.0);
    assert_eq!(gradient[4], 1.0);

    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut typed = Graph::new();
        let input = typed.input_dtype("x", [], dtype);
        let output = typed.relu6(input).unwrap();
        assert_eq!(typed.dtype(output).unwrap(), dtype);
        assert_eq!(typed.shape(output).unwrap(), &Shape::new([]));
    }
    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0], DType::F16);
    let output = empty.relu6(input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);

    let mut malformed = Graph::new();
    let nodes = malformed.node_count();
    assert!(malformed.relu6(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn leaky_relu_uses_tinygrad_ordered_live_slope_select() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [7], DType::F64);
    let slope = graph.input_dtype("slope", [], DType::F64);
    let output = graph.leaky_relu(input, slope).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([7]));
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([
        (
            "x".into(),
            TensorData::from_scalars(
                [7],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-1.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(1.0),
                ],
            )
            .unwrap(),
        ),
        (
            "slope".into(),
            TensorData::scalar_with_dtype(Scalar::F(-0.5), DType::F64),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert!(values.scalar_at(0).as_f64().is_infinite());
    assert!(values.scalar_at(0).as_f64().is_sign_positive());
    assert_eq!(values.scalar_at(1).as_f64(), 0.5);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_infinite());
    assert_eq!(values.scalar_at(6).as_f64(), 1.0);
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(gradient[1], -0.5);
    assert_eq!(gradient[2], 1.0);
    assert_eq!(gradient[3], 1.0);
    assert_eq!(gradient[4], 1.0);
    assert_eq!(gradient[6], 1.0);

    // A NaN slope is unselected on a nonnegative input because the source
    // predicate is strict `x < 0`, rather than a branch arithmetic shortcut.
    let mut unordered = Graph::new();
    let x = unordered.input_dtype("x", [], DType::F64);
    let slope = unordered.input_dtype("slope", [], DType::F64);
    let output = unordered.leaky_relu(x, slope).unwrap();
    let value = CpuBackend
        .execute(
            &unordered,
            output,
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F64),
                ),
                (
                    "slope".into(),
                    TensorData::scalar_with_dtype(Scalar::F(f64::NAN), DType::F64),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(value.scalar_at(0).as_f64(), 1.0);

    let mut broadcast = Graph::new();
    let x = broadcast.input_dtype("x", [2, 3], DType::F16);
    let slope = broadcast.input_dtype("slope", [1, 3], DType::F32);
    let output = broadcast.leaky_relu(x, slope).unwrap();
    assert_eq!(broadcast.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(broadcast.dtype(output).unwrap(), DType::F32);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let slope = narrow.input_dtype("slope", [], dtype);
        let output = narrow.leaky_relu(x, slope).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut exact = Graph::new();
        let x = exact.input_dtype("x", [], dtype);
        let slope = exact.input_dtype("slope", [], dtype);
        let output = exact.leaky_relu(x, slope).unwrap();
        assert_eq!(exact.dtype(output).unwrap(), dtype);
    }
    let mut wide = Graph::new();
    let x = wide.input_dtype("x", [], DType::I64);
    let slope = wide.input_dtype("slope", [], DType::U64);
    let output = wide.leaky_relu(x, slope).unwrap();
    assert_eq!(wide.dtype(output).unwrap(), DType::F32);

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0], DType::I32);
    let slope = empty.input_dtype("slope", [], DType::F32);
    let output = empty.leaky_relu(x, slope).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F32);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2, 3]);
    let slope = malformed.input("slope", [2, 2]);
    let nodes = malformed.node_count();
    assert!(malformed.leaky_relu(x, slope).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn leaky_relu_scalar_matches_tinygrad_weak_default_without_changing_live_slope_api() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [5], DType::F64);
    let output = graph.leaky_relu_default(input).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([5]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.op, Op::Constant(data)
            if data.dtype() == DType::F64 && data.shape() == &Shape::new([])
              && data.scalar_at(0).as_f64().to_bits() == 0.01f64.to_bits())
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [5],
                    DType::F64,
                    [
                        Scalar::F(-2.0),
                        Scalar::F(-0.0),
                        Scalar::F(f64::NAN),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(3.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    close(values.scalar_at(0).as_f64(), -0.02, 1e-15);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_infinite());
    assert_eq!(values.scalar_at(4).as_f64(), 3.0);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let output = narrow.leaky_relu_scalar(x, 0.125).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert!(narrow.nodes.iter().any(|node| {
            matches!(&node.op, Op::Constant(data)
                if data.dtype() == dtype && data.shape() == &Shape::new([]))
        }));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut nonfloat = Graph::new();
        let x = nonfloat.input_dtype("x", [], dtype);
        let output = nonfloat.leaky_relu_scalar(x, 0.125).unwrap();
        assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
        assert!(nonfloat.nodes.iter().any(|node| {
            matches!(&node.op, Op::Constant(data) if data.dtype() == DType::F32)
        }));
    }

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::BF16);
    let output = empty.leaky_relu_scalar(x, -0.5).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed.leaky_relu_default(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX], DType::F32);
    let before = malformed.node_count();
    assert!(malformed.leaky_relu_default(overflow).is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn leaky_relu_with_scalar_preserves_tinygrad_untyped_slope_surface() {
    for (input_dtype, slope, expected_dtype) in [
        (DType::Bool, Scalar::Bool(true), DType::Bool),
        (DType::Bool, Scalar::I(-1), DType::I32),
        (DType::I8, Scalar::U(1), DType::I8),
        (DType::I16, Scalar::Bool(true), DType::I16),
        (DType::I32, Scalar::F(-0.5), DType::F32),
        (DType::I64, Scalar::U(1), DType::I64),
        (DType::U8, Scalar::I(-1), DType::U8),
        (DType::U16, Scalar::Bool(false), DType::U16),
        (DType::U32, Scalar::F(0.5), DType::F32),
        (DType::U64, Scalar::I(1), DType::U64),
        (DType::F16, Scalar::Bool(true), DType::F16),
        (DType::BF16, Scalar::I(-1), DType::BF16),
        (DType::F32, Scalar::U(1), DType::F32),
        (DType::F64, Scalar::F(f64::NAN), DType::F64),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], input_dtype);
        let output = graph.leaky_relu_with_scalar(input, slope).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), expected_dtype);
        let Op::Select { condition, on_true, on_false } = graph.op(output).unwrap() else {
            panic!("untyped scalar leaky_relu must lower to Select");
        };
        assert!(matches!(graph.op(*condition).unwrap(), Op::Compare { op: CompareOp::Lt, .. }));
        let Op::Binary { op: BinaryOp::Mul, lhs, .. } = graph.op(*on_true).unwrap() else {
            panic!("negative branch must remain slope-left Mul");
        };
        assert!(matches!(graph.op(*lhs).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == expected_dtype));
        assert_eq!(graph.dtype(*on_false).unwrap(), expected_dtype);
    }

    // The only I64/U64 bridge is still the live two-tensor surface; concrete
    // weak scalars commit to their tensor reference width before multiplication.
    let mut bridge = Graph::new();
    let x = bridge.input_dtype("x", [], DType::I64);
    let slope = bridge.input_dtype("slope", [], DType::U64);
    assert_eq!(bridge.dtype(bridge.leaky_relu(x, slope).unwrap()).unwrap(), DType::F32);

    let mut scalar = Graph::new();
    let x = scalar.input_dtype("x", [], DType::F64);
    let output = scalar.leaky_relu_with_scalar(x, Scalar::F(f64::NAN)).unwrap();
    let loss = scalar.sum_all(output).unwrap();
    assert_eq!(scalar.shape(scalar.grad(loss, x).unwrap()).unwrap(), &Shape::new([]));

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::BF16);
    let output = empty.leaky_relu_with_scalar(x, Scalar::I(-1)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed
        .leaky_relu_with_scalar(crate::NodeId(usize::MAX), Scalar::Bool(true))
        .is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(malformed
        .leaky_relu_with_scalar(overflow, Scalar::F(-0.5))
        .is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn celu_uses_source_ordered_extrema_and_reciprocal_division() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [7], DType::F64);
    let alpha = graph.input_dtype("alpha", [], DType::F64);
    let output = graph.celu(input, alpha).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([7]));
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([
        (
            "x".into(),
            TensorData::from_scalars(
                [7],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-1.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(1.0),
                ],
            )
            .unwrap(),
        ),
        (
            "alpha".into(),
            TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F64),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), -1.0);
    close(values.scalar_at(1).as_f64(), (-1.0f64).exp() - 1.0, 1e-12);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_infinite());
    assert_eq!(values.scalar_at(6).as_f64(), 1.0);
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    close(gradient[1], (-1.0f64).exp(), 1e-12);
    assert_eq!(gradient[2], 1.0);
    assert_eq!(gradient[3], 1.0);
    assert_eq!(gradient[6], 1.0);

    // The source formula evaluates its negative term even on positive lanes:
    // zero and nonfinite alpha therefore remain observable rather than being
    // hidden behind a conventional conditional activation helper.
    for alpha_value in [0.0, f64::NAN, f64::INFINITY] {
        let mut special = Graph::new();
        let x = special.input_dtype("x", [], DType::F64);
        let alpha = special.input_dtype("alpha", [], DType::F64);
        let output = special.celu(x, alpha).unwrap();
        let value = CpuBackend
            .execute(
                &special,
                output,
                &HashMap::from([
                    (
                        "x".into(),
                        TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F64),
                    ),
                    (
                        "alpha".into(),
                        TensorData::scalar_with_dtype(Scalar::F(alpha_value), DType::F64),
                    ),
                ]),
            )
            .unwrap();
        assert!(value.scalar_at(0).as_f64().is_nan());
    }

    let mut negative_alpha = Graph::new();
    let x = negative_alpha.input_dtype("x", [], DType::F64);
    let alpha = negative_alpha.input_dtype("alpha", [], DType::F64);
    let output = negative_alpha.celu(x, alpha).unwrap();
    let value = CpuBackend
        .execute(
            &negative_alpha,
            output,
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::scalar_with_dtype(Scalar::F(-1.0), DType::F64),
                ),
                (
                    "alpha".into(),
                    TensorData::scalar_with_dtype(Scalar::F(-1.0), DType::F64),
                ),
            ]),
        )
        .unwrap();
    close(value.scalar_at(0).as_f64(), 1.0 - std::f64::consts::E, 1e-12);

    let mut broadcast = Graph::new();
    let x = broadcast.input_dtype("x", [2, 3], DType::F16);
    let alpha = broadcast.input_dtype("alpha", [1, 3], DType::F32);
    let output = broadcast.celu(x, alpha).unwrap();
    assert_eq!(broadcast.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(broadcast.dtype(output).unwrap(), DType::F32);
    let mut source_common = Graph::new();
    let x = source_common.input_dtype("x", [], DType::F16);
    let alpha = source_common.input_dtype("alpha", [], DType::I32);
    let output = source_common.celu(x, alpha).unwrap();
    assert_eq!(source_common.dtype(output).unwrap(), DType::F16);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let alpha = narrow.input_dtype("alpha", [], dtype);
        let output = narrow.celu(x, alpha).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut promoted = Graph::new();
        let x = promoted.input_dtype("x", [], dtype);
        let alpha = promoted.input_dtype("alpha", [], dtype);
        let output = promoted.celu(x, alpha).unwrap();
        assert_eq!(promoted.dtype(output).unwrap(), DType::F32);
    }
    let mut wide = Graph::new();
    let x = wide.input_dtype("x", [], DType::I64);
    let alpha = wide.input_dtype("alpha", [], DType::U64);
    let output = wide.celu(x, alpha).unwrap();
    assert_eq!(wide.dtype(output).unwrap(), DType::F32);
    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0], DType::I32);
    let alpha = empty.input_dtype("alpha", [], DType::F32);
    let output = empty.celu(x, alpha).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F32);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2, 3]);
    let alpha = malformed.input("alpha", [2, 2]);
    let nodes = malformed.node_count();
    assert!(malformed.celu(x, alpha).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn celu_scalar_matches_tinygrad_weak_default_without_changing_live_alpha_api() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [5], DType::F64);
    let output = graph.celu_default(input).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([5]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.op, Op::Constant(data)
            if data.dtype() == DType::F64 && data.shape() == &Shape::new([])
              && data.scalar_at(0).as_f64().to_bits() == 1.0f64.to_bits())
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [5],
                    DType::F64,
                    [
                        Scalar::F(-1.0),
                        Scalar::F(-0.0),
                        Scalar::F(f64::NAN),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(2.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    close(values.scalar_at(0).as_f64(), (-1.0f64).exp() - 1.0, 1e-12);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_infinite());
    assert_eq!(values.scalar_at(4).as_f64(), 2.0);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let x = narrow.input_dtype("x", [], dtype);
        let output = narrow.celu_scalar(x, 0.125).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert!(narrow.nodes.iter().any(|node| {
            matches!(&node.op, Op::Constant(data)
                if data.dtype() == dtype && data.shape() == &Shape::new([]))
        }));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut nonfloat = Graph::new();
        let x = nonfloat.input_dtype("x", [], dtype);
        let output = nonfloat.celu_scalar(x, 0.125).unwrap();
        assert_eq!(nonfloat.dtype(output).unwrap(), DType::F32);
        assert!(nonfloat.nodes.iter().any(|node| {
            matches!(&node.op, Op::Constant(data) if data.dtype() == DType::F32)
        }));
    }

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::BF16);
    let output = empty.celu_scalar(x, -0.5).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed.celu_default(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX], DType::F32);
    let before = malformed.node_count();
    assert!(malformed.celu_default(overflow).is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn celu_with_scalar_preserves_tinygrad_per_consumer_alpha_commitment() {
    for (input_dtype, alpha, expected_dtype) in [
        (DType::Bool, Scalar::Bool(true), DType::F32),
        (DType::Bool, Scalar::I(-1), DType::F32),
        (DType::I8, Scalar::U(1), DType::F32),
        (DType::I16, Scalar::Bool(true), DType::F32),
        (DType::I32, Scalar::F(-0.5), DType::F32),
        (DType::I64, Scalar::U(1), DType::F32),
        (DType::U8, Scalar::I(-1), DType::F32),
        (DType::U16, Scalar::Bool(false), DType::F32),
        (DType::U32, Scalar::F(0.5), DType::F32),
        (DType::U64, Scalar::I(1), DType::F32),
        (DType::F16, Scalar::Bool(true), DType::F16),
        (DType::BF16, Scalar::I(-1), DType::BF16),
        (DType::F32, Scalar::U(1), DType::F32),
        (DType::F64, Scalar::F(f64::NAN), DType::F64),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], input_dtype);
        let output = graph.celu_with_scalar(input, alpha).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), expected_dtype);
        let Op::Binary { op: BinaryOp::Add, rhs, .. } = graph.op(output).unwrap() else {
            panic!("CELU must retain its source final Add");
        };
        let Op::Binary { op: BinaryOp::Minimum, lhs: scaled, .. } = graph.op(*rhs).unwrap() else {
            panic!("CELU negative branch must retain its ordered minimum");
        };
        let Op::Binary { op: BinaryOp::Mul, lhs: alpha_node, .. } = graph.op(*scaled).unwrap() else {
            panic!("CELU negative branch must retain alpha-left Mul");
        };
        assert!(matches!(graph.op(*alpha_node).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == expected_dtype));
    }

    // An integer alpha is wrapped once against x for division and once
    // against the post-Exp F32 branch for multiplication.
    let mut staged = Graph::new();
    let x = staged.input_dtype("x", [], DType::I16);
    let output = staged.celu_with_scalar(x, Scalar::I(-1)).unwrap();
    assert_eq!(staged.dtype(output).unwrap(), DType::F32);
    assert!(staged.nodes.iter().any(|node| matches!(&node.op, Op::Constant(data)
        if data.dtype() == DType::I16 && data.scalar_at(0).as_i64() == -1)));
    assert!(staged.nodes.iter().any(|node| matches!(&node.op, Op::Constant(data)
        if data.dtype() == DType::F32 && data.scalar_at(0).as_i64() == -1)));

    let mut bridge = Graph::new();
    let x = bridge.input_dtype("x", [], DType::I64);
    let alpha = bridge.input_dtype("alpha", [], DType::U64);
    assert_eq!(bridge.dtype(bridge.celu(x, alpha).unwrap()).unwrap(), DType::F32);

    let mut scalar = Graph::new();
    let x = scalar.input_dtype("x", [], DType::F64);
    let output = scalar.celu_with_scalar(x, Scalar::F(f64::NAN)).unwrap();
    let loss = scalar.sum_all(output).unwrap();
    assert_eq!(scalar.shape(scalar.grad(loss, x).unwrap()).unwrap(), &Shape::new([]));

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::BF16);
    let output = empty.celu_with_scalar(x, Scalar::I(-1)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed
        .celu_with_scalar(crate::NodeId(usize::MAX), Scalar::Bool(true))
        .is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(malformed
        .celu_with_scalar(overflow, Scalar::F(-0.5))
        .is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn swish_and_silu_share_tinygrad_sigmoid_outer_multiply() {
    for helper in [Graph::swish as UnaryGraphOp, Graph::silu as UnaryGraphOp] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [5], DType::F64);
        let output = helper(&mut graph, input).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F64);
        let loss = graph.sum_all(output).unwrap();
        let input_gradient = graph.grad(loss, input).unwrap();
        let bindings = HashMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [5],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        )]);
        let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
        // The outer product is observable: -infinity times sigmoid(-infinity)
        // is NaN, while signed zeros retain their sign through multiplication
        // by the source-width 0.5 sigmoid value.
        assert!(values.scalar_at(0).as_f64().is_nan());
        assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
        assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
        assert!(values.scalar_at(3).as_f64().is_nan());
        assert!(values.scalar_at(4).as_f64().is_infinite());
        assert!(values.scalar_at(4).as_f64().is_sign_positive());
        let gradient = CpuBackend
            .execute(&graph, input_gradient, &bindings)
            .unwrap()
            .to_vec_f64();
        assert_eq!(gradient[1], 0.5);
        assert_eq!(gradient[2], 0.5);
    }

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [], dtype);
        let output = graph.swish(input).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), dtype);
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [], dtype);
        let output = graph.silu(input).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    }
    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0], DType::F16);
    let output = empty.swish(input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);

    let mut malformed = Graph::new();
    let nodes = malformed.node_count();
    assert!(malformed.swish(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn quick_gelu_uses_tinygrad_typed_sigmoid_composition_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [5], DType::F64);
    let output = graph.quick_gelu(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [5],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    // The outer multiply is literal and observable: infinity times the
    // sigmoid zero is NaN, while zeros retain their input sign.
    assert!(values.scalar_at(0).as_f64().is_nan());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_infinite());
    assert!(values.scalar_at(4).as_f64().is_sign_positive());
    let gradient = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(gradient[1], 0.5);
    assert_eq!(gradient[2], 0.5);

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut typed = Graph::new();
        let x = typed.input_dtype("x", [], dtype);
        let output = typed.quick_gelu(x).unwrap();
        assert_eq!(typed.dtype(output).unwrap(), dtype);
        assert_eq!(typed.shape(output).unwrap(), &Shape::new([]));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut promoted = Graph::new();
        let x = promoted.input_dtype("x", [], dtype);
        assert_eq!(
            promoted.dtype(promoted.quick_gelu(x).unwrap()).unwrap(),
            DType::F32
        );
    }
    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0], DType::F16);
    let output = empty.quick_gelu(x).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);

    let mut malformed = Graph::new();
    let nodes = malformed.node_count();
    assert!(malformed.quick_gelu(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn parameterized_composite_activations_preflight_broadcasts() {
    let mut leaky = Graph::new();
    let input = leaky.input("x", [2, 3]);
    let bad_slope = leaky.input("slope", [2, 2]);
    let nodes = leaky.node_count();
    assert!(leaky.leaky_relu(input, bad_slope).is_err());
    assert_eq!(leaky.node_count(), nodes);

    let mut elu = Graph::new();
    let input = elu.input("x", [2, 3]);
    let bad_alpha = elu.input("alpha", [2, 2]);
    let nodes = elu.node_count();
    assert!(elu.elu(input, bad_alpha).is_err());
    assert_eq!(elu.node_count(), nodes);

    let mut celu = Graph::new();
    let input = celu.input("x", [2, 3]);
    let bad_alpha = celu.input("alpha", [2, 2]);
    let nodes = celu.node_count();
    assert!(celu.celu(input, bad_alpha).is_err());
    assert_eq!(celu.node_count(), nodes);

    let mut selu = Graph::new();
    let input = selu.input("x", [2, 1]);
    let alpha = selu.input("alpha", [1, 3]);
    let bad_gamma = selu.input("gamma", [2, 2]);
    let nodes = selu.node_count();
    assert!(selu.selu(input, alpha, bad_gamma).is_err());
    assert_eq!(selu.node_count(), nodes);

    let mut valid = Graph::new();
    let input = valid.input("x", [2]);
    let alpha = valid.constant(TensorData::scalar(1.0f32));
    let output = valid.elu(input, alpha).unwrap();
    let values = execute(&valid, output, DType::F32, &[-1.0, 2.0]).to_vec_f64();
    close(values[0], (-1.0f64).exp() - 1.0, 1e-6);
    assert_eq!(values[1], 2.0);
}

#[test]
fn parameterized_hardsigmoid_matches_tinygrad_relu_difference() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [5], dtype);
        let alpha = graph
            .constant(TensorData::from_scalars(Shape::new([]), dtype, [Scalar::F(0.25)]).unwrap());
        let beta = graph
            .constant(TensorData::from_scalars(Shape::new([]), dtype, [Scalar::F(0.5)]).unwrap());
        let output = graph.hardsigmoid_with(x, alpha, beta).unwrap();
        assert_eq!(
            graph.dtype(output).unwrap(),
            if dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            }
        );
        let values = execute(&graph, output, dtype, &[-4.0, -2.0, 0.0, 2.0, 4.0]).to_vec_f64();
        for (actual, expected) in values.into_iter().zip([0.0, 0.0, 0.5, 1.0, 1.0]) {
            close(
                actual,
                expected,
                if dtype == DType::F64 { 1e-12 } else { 0.01 },
            );
        }
    }

    for dtype in [DType::Bool, DType::I32, DType::U64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1], dtype);
        let alpha = graph.constant(TensorData::scalar(0.25f32));
        let beta = graph.constant(TensorData::scalar(0.5f32));
        let output = graph.hardsigmoid_with(x, alpha, beta).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [4], DType::F64);
    let output = graph.hardsigmoid(x).unwrap();
    let values = execute(
        &graph,
        output,
        DType::F64,
        &[f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -0.0],
    )
    .to_vec_f64();
    assert_eq!(values[0], 0.0);
    assert!(values[1].is_nan());
    assert_eq!(values[2], 0.0);
    close(values[3], 0.5, 1e-12);
    let operations = graph
        .trace(output)
        .unwrap()
        .steps
        .into_iter()
        .map(|step| step.operation)
        .collect::<Vec<_>>();
    assert_eq!(
        operations
            .iter()
            .filter(|operation| operation.starts_with("relu("))
            .count(),
        2
    );
}

#[test]
fn parameterized_hardsigmoid_gradient_matches_central_difference() {
    let point = 0.4;
    let epsilon = 1e-4;
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1], DType::F64);
    let alpha = graph.constant(TensorData::scalar_with_dtype(Scalar::F(0.25), DType::F64));
    let beta = graph.constant(TensorData::scalar_with_dtype(Scalar::F(0.5), DType::F64));
    let value = graph.hardsigmoid_with(x, alpha, beta).unwrap();
    let output = graph.sum(value, 0).unwrap();
    let gradient = graph.grad(output, x).unwrap();
    let analytic = execute(&graph, gradient, DType::F64, &[point]).to_vec_f64()[0];
    let plus = execute(&graph, output, DType::F64, &[point + epsilon]).to_vec_f64()[0];
    let minus = execute(&graph, output, DType::F64, &[point - epsilon]).to_vec_f64()[0];
    close(analytic, (plus - minus) / (2.0 * epsilon), 2e-6);
}

#[test]
fn hardswish_matches_tinygrad_relu6_composition() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [6], dtype);
        let output = graph.hardswish(x).unwrap();
        assert_eq!(
            graph.dtype(output).unwrap(),
            if dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            }
        );
        let values = execute(&graph, output, dtype, &[-4.0, -1.0, 0.0, 1.0, 3.0, 4.0]).to_vec_f64();
        let expected = [0.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 3.0, 4.0];
        for (actual, expected) in values.into_iter().zip(expected) {
            close(
                actual,
                expected,
                if dtype == DType::F64 { 1e-12 } else { 0.01 },
            );
        }
    }

    for dtype in [DType::Bool, DType::I32, DType::U64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1], dtype);
        let output = graph.hardswish(x).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [5], DType::F64);
    let output = graph.hardswish(x).unwrap();
    let values = execute(
        &graph,
        output,
        DType::F64,
        &[f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -0.0, 0.0],
    )
    .to_vec_f64();
    assert!(values[0].is_nan());
    assert!(values[1].is_nan());
    assert!(values[2].is_nan());
    assert_eq!(values[3], 0.0);
    assert!(values[3].is_sign_negative());
    assert_eq!(values[4], 0.0);
    assert!(values[4].is_sign_positive());
    let operations = graph
        .trace(output)
        .unwrap()
        .steps
        .into_iter()
        .map(|step| step.operation)
        .collect::<Vec<_>>();
    assert_eq!(
        operations
            .iter()
            .filter(|operation| operation.starts_with("relu("))
            .count(),
        2
    );
}

#[test]
fn hardswish_gradient_matches_central_difference_away_from_kinks() {
    let point = 1.0;
    let epsilon = 1e-4;
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1], DType::F64);
    let value = graph.hardswish(x).unwrap();
    let output = graph.sum(value, 0).unwrap();
    let gradient = graph.grad(output, x).unwrap();
    let analytic = execute(&graph, gradient, DType::F64, &[point]).to_vec_f64()[0];
    let plus = execute(&graph, output, DType::F64, &[point + epsilon]).to_vec_f64()[0];
    let minus = execute(&graph, output, DType::F64, &[point - epsilon]).to_vec_f64()[0];
    close(analytic, (plus - minus) / (2.0 * epsilon), 2e-6);
}

#[test]
fn relu6_matches_tinygrad_relu_difference() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [7], dtype);
        let output = graph.relu6(x).unwrap();
        assert_eq!(
            graph.dtype(output).unwrap(),
            if dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            }
        );
        let values = execute(
            &graph,
            output,
            dtype,
            &[-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0],
        )
        .to_vec_f64();
        for (actual, expected) in values.into_iter().zip([0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 6.0]) {
            close(
                actual,
                expected,
                if dtype == DType::F64 { 1e-12 } else { 0.01 },
            );
        }
    }

    for dtype in [DType::Bool, DType::I32, DType::U64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1], dtype);
        let output = graph.relu6(x).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [5], DType::F64);
    let output = graph.relu6(x).unwrap();
    let values = execute(
        &graph,
        output,
        DType::F64,
        &[f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -0.0, 0.0],
    )
    .to_vec_f64();
    assert_eq!(values[0], 0.0);
    assert!(values[1].is_nan());
    assert_eq!(values[2], 0.0);
    assert_eq!(values[3], 0.0);
    // tinygrad specifies the ReLU-difference composition, not a sign choice
    // for an equal-zero floating maximum; Rust's `max` is platform-dependent
    // at that tie, while both signed zeros are exact numeric results.
    assert_eq!(values[4], 0.0);
    assert!(values[4].is_sign_positive());
    let operations = graph
        .trace(output)
        .unwrap()
        .steps
        .into_iter()
        .map(|step| step.operation)
        .collect::<Vec<_>>();
    assert_eq!(
        operations
            .iter()
            .filter(|operation| operation.starts_with("relu("))
            .count(),
        2
    );
}

#[test]
fn relu6_gradient_matches_central_difference_away_from_kinks() {
    let point = 2.0;
    let epsilon = 1e-4;
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1], DType::F64);
    let value = graph.relu6(x).unwrap();
    let output = graph.sum(value, 0).unwrap();
    let gradient = graph.grad(output, x).unwrap();
    let analytic = execute(&graph, gradient, DType::F64, &[point]).to_vec_f64()[0];
    let plus = execute(&graph, output, DType::F64, &[point + epsilon]).to_vec_f64()[0];
    let minus = execute(&graph, output, DType::F64, &[point - epsilon]).to_vec_f64()[0];
    close(analytic, (plus - minus) / (2.0 * epsilon), 2e-6);
}

#[test]
fn stable_softplus_family_matches_tinygrad_logaddexp_definition() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [3], dtype);
        let beta = graph
            .constant(TensorData::from_scalars(Shape::new([]), dtype, [Scalar::F(1.0)]).unwrap());
        let output = graph.softplus(x, beta).unwrap();
        // RustGrad's scalar literals follow the existing activation-helper
        // promotion contract: F16/BF16 and non-floats lift through F32,
        // while F32/F64 retain their widened floating dtype.
        assert_eq!(
            graph.dtype(output).unwrap(),
            if dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            }
        );
        let values = execute(&graph, output, dtype, &[-1000.0, 0.0, 1000.0]).to_vec_f64();
        close(values[0], 0.0, 1e-5);
        close(
            values[1],
            std::f64::consts::LN_2,
            if matches!(dtype, DType::F16 | DType::BF16) {
                0.01
            } else {
                1e-6
            },
        );
        close(values[2], 1000.0, 1e-4);
    }

    for dtype in [DType::Bool, DType::I32, DType::U64] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1], dtype);
        let beta = graph.constant(TensorData::scalar(1.0f32));
        let output = graph.softplus(x, beta).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [4], DType::F64);
    let beta = graph.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F64));
    let softplus = graph.softplus(x, beta).unwrap();
    let logsigmoid = graph.logsigmoid(x).unwrap();
    let softplus_values = execute(
        &graph,
        softplus,
        DType::F64,
        &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0],
    )
    .to_vec_f64();
    assert!(softplus_values[0].is_infinite() && softplus_values[0].is_sign_positive());
    assert_eq!(softplus_values[1], 0.0);
    assert!(softplus_values[2].is_nan());
    close(softplus_values[3], std::f64::consts::LN_2, 1e-12);
    let logsigmoid_values = execute(
        &graph,
        logsigmoid,
        DType::F64,
        &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0],
    )
    .to_vec_f64();
    assert_eq!(logsigmoid_values[0], 0.0);
    assert!(logsigmoid_values[0].is_sign_negative());
    assert!(logsigmoid_values[1].is_infinite() && logsigmoid_values[1].is_sign_negative());
    assert!(logsigmoid_values[2].is_nan());
    close(logsigmoid_values[3], -std::f64::consts::LN_2, 1e-12);
    let operations = graph
        .trace(softplus)
        .unwrap()
        .steps
        .into_iter()
        .map(|step| step.operation)
        .collect::<Vec<_>>();
    assert!(
        operations
            .iter()
            .any(|operation| operation.starts_with("maximum("))
    );
    assert!(
        operations
            .iter()
            .any(|operation| operation.starts_with("abs("))
    );
}

#[test]
fn stable_softplus_family_gradients_match_central_differences() {
    let point = 0.75;
    let epsilon = 1e-4;
    for logsigmoid in [false, true] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1], DType::F64);
        let value = if logsigmoid {
            graph.logsigmoid(x).unwrap()
        } else {
            let beta = graph.constant(TensorData::scalar_with_dtype(Scalar::F(1.25), DType::F64));
            graph.softplus(x, beta).unwrap()
        };
        let output = graph.sum(value, 0).unwrap();
        let gradient = graph.grad(output, x).unwrap();
        let analytic = execute(&graph, gradient, DType::F64, &[point]).to_vec_f64()[0];
        let plus = execute(&graph, output, DType::F64, &[point + epsilon]).to_vec_f64()[0];
        let minus = execute(&graph, output, DType::F64, &[point - epsilon]).to_vec_f64()[0];
        close(analytic, (plus - minus) / (2.0 * epsilon), 2e-6);
    }
}
