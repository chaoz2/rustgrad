use crate::{
    Backend, Conv2dOptions, CpuBackend, DType, Error, Graph, ReduceKind, Scalar, Shape, TensorData,
};
use std::collections::HashMap;

fn f32_data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
    TensorData::new(shape, values.to_vec()).unwrap()
}

fn typed_data(
    shape: impl Into<Shape>,
    dtype: DType,
    values: impl IntoIterator<Item = Scalar>,
) -> TensorData {
    TensorData::from_scalars(shape, dtype, values).unwrap()
}

fn execute(
    graph: &Graph,
    output: crate::NodeId,
    x: TensorData,
    w: TensorData,
    b: Option<TensorData>,
) -> TensorData {
    let mut inputs = HashMap::from([("x".into(), x), ("w".into(), w)]);
    if let Some(b) = b {
        inputs.insert("b".into(), b);
    }
    CpuBackend.execute(graph, output, &inputs).unwrap()
}

fn all_sum(graph: &mut Graph, input: crate::NodeId) -> crate::NodeId {
    graph.reduce(input, ReduceKind::Sum, None, false).unwrap()
}

fn assert_close(actual: &TensorData, expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, expected) in expected.iter().enumerate() {
        let actual = actual.scalar_at(index).as_f64() as f32;
        assert!(
            (actual - expected).abs() < 2e-3,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn conv2d_forward_fixture_matrix() {
    struct Fixture {
        name: &'static str,
        x_shape: [usize; 4],
        x: &'static [f32],
        w_shape: [usize; 4],
        w: &'static [f32],
        bias: Option<&'static [f32]>,
        options: Conv2dOptions,
        shape: [usize; 4],
        expected: &'static [f32],
    }
    let fixtures = [
        Fixture {
            name: "plain without bias",
            x_shape: [1, 1, 3, 3],
            x: &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            w_shape: [1, 1, 2, 2],
            w: &[1., 1., 1., 1.],
            bias: None,
            options: Conv2dOptions::default(),
            shape: [1, 1, 2, 2],
            expected: &[12., 16., 24., 28.],
        },
        Fixture {
            name: "asymmetric padding with bias",
            x_shape: [1, 1, 2, 2],
            x: &[1., 2., 3., 4.],
            w_shape: [1, 1, 2, 2],
            w: &[1., 1., 1., 1.],
            bias: Some(&[10.]),
            options: Conv2dOptions {
                padding: [1, 0, 0, 1],
                ..Default::default()
            },
            shape: [1, 1, 2, 2],
            expected: &[13., 12., 20., 16.],
        },
        Fixture {
            name: "stride greater than one",
            x_shape: [1, 1, 4, 4],
            x: &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            w_shape: [1, 1, 2, 2],
            w: &[1., 1., 1., 1.],
            bias: None,
            options: Conv2dOptions {
                stride: [2, 2],
                ..Default::default()
            },
            shape: [1, 1, 2, 2],
            expected: &[14., 22., 46., 54.],
        },
        Fixture {
            name: "dilation greater than one",
            x_shape: [1, 1, 3, 3],
            x: &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            w_shape: [1, 1, 2, 2],
            w: &[1., 1., 1., 1.],
            bias: None,
            options: Conv2dOptions {
                dilation: [2, 2],
                ..Default::default()
            },
            shape: [1, 1, 1, 1],
            expected: &[20.],
        },
        Fixture {
            name: "grouped convolution",
            x_shape: [1, 2, 2, 2],
            x: &[1., 2., 3., 4., 5., 6., 7., 8.],
            w_shape: [2, 1, 1, 1],
            w: &[2., 3.],
            bias: None,
            options: Conv2dOptions {
                groups: 2,
                ..Default::default()
            },
            shape: [1, 2, 2, 2],
            expected: &[2., 4., 6., 8., 15., 18., 21., 24.],
        },
        Fixture {
            name: "depthwise convolution",
            x_shape: [1, 3, 1, 1],
            x: &[2., 3., 4.],
            w_shape: [3, 1, 1, 1],
            w: &[5., 6., 7.],
            bias: None,
            options: Conv2dOptions {
                groups: 3,
                ..Default::default()
            },
            shape: [1, 3, 1, 1],
            expected: &[10., 18., 28.],
        },
    ];

    for fixture in fixtures {
        let mut graph = Graph::new();
        let x = graph.input("x", fixture.x_shape);
        let w = graph.input("w", fixture.w_shape);
        let b = fixture.bias.map(|_| graph.input("b", [fixture.w_shape[0]]));
        let y = graph.conv2d(x, w, b, fixture.options).unwrap();
        assert_eq!(
            graph.shape(y).unwrap(),
            &Shape::from(fixture.shape),
            "{}",
            fixture.name
        );
        assert_close(
            &execute(
                &graph,
                y,
                f32_data(fixture.x_shape, fixture.x),
                f32_data(fixture.w_shape, fixture.w),
                fixture
                    .bias
                    .map(|values| f32_data([fixture.w_shape[0]], values)),
            ),
            fixture.expected,
        );
    }
}

#[test]
fn conv2d_promotes_mixed_integers_and_preserves_bool_logic() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1, 1, 2, 2], DType::I16);
    let w = graph.input_dtype("w", [1, 1, 1, 1], DType::U8);
    let b = graph.input_dtype("b", [1], DType::I8);
    let y = graph
        .conv2d(x, w, Some(b), Conv2dOptions::default())
        .unwrap();
    assert_eq!(graph.dtype(y).unwrap(), DType::I32);
    assert_eq!(
        execute(
            &graph,
            y,
            typed_data(
                [1, 1, 2, 2],
                DType::I16,
                [Scalar::I(1), Scalar::I(2), Scalar::I(3), Scalar::I(4)]
            ),
            typed_data([1, 1, 1, 1], DType::U8, [Scalar::U(2)]),
            Some(typed_data([1], DType::I8, [Scalar::I(-1)])),
        ),
        typed_data(
            [1, 1, 2, 2],
            DType::I32,
            [Scalar::I(1), Scalar::I(3), Scalar::I(5), Scalar::I(7)]
        ),
    );

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1, 1, 1, 2], DType::Bool);
    let w = graph.input_dtype("w", [1, 1, 1, 2], DType::Bool);
    let y = graph.conv2d(x, w, None, Conv2dOptions::default()).unwrap();
    assert_eq!(graph.dtype(y).unwrap(), DType::I32);
    assert_eq!(
        execute(
            &graph,
            y,
            typed_data(
                [1, 1, 1, 2],
                DType::Bool,
                [Scalar::Bool(true), Scalar::Bool(false)]
            ),
            typed_data(
                [1, 1, 1, 2],
                DType::Bool,
                [Scalar::Bool(false), Scalar::Bool(true)]
            ),
            None,
        ),
        typed_data([1, 1, 1, 1], DType::I32, [Scalar::I(0)]),
    );
}

#[test]
fn conv2d_validation_matrix() {
    let mut graph = Graph::new();
    let bad_rank = graph.input("bad_rank", [1, 1, 2]);
    let weight = graph.input("weight", [1, 1, 1, 1]);
    assert!(matches!(
        graph.conv2d(bad_rank, weight, None, Conv2dOptions::default()),
        Err(Error::InvalidConv2d {
            input: actual_input,
            weight: actual_weight,
            reason: "input and weight must be rank 4",
        }) if actual_input == Shape::from([1, 1, 2])
            && actual_weight == Shape::from([1, 1, 1, 1])
    ));

    let mut graph = Graph::new();
    let input = graph.input("x", [1, 3, 2, 2]);
    let weight = graph.input("w", [3, 1, 1, 1]);
    assert!(matches!(
        graph.conv2d(
            input,
            weight,
            None,
            Conv2dOptions {
                groups: 2,
                ..Default::default()
            }
        ),
        Err(Error::InvalidConv2d {
            reason: "channel/group geometry",
            ..
        })
    ));

    let mut graph = Graph::new();
    let input = graph.input("x", [1, 4, 2, 2]);
    let weight = graph.input("w", [2, 1, 1, 1]);
    assert!(matches!(
        graph.conv2d(
            input,
            weight,
            None,
            Conv2dOptions {
                groups: 2,
                ..Default::default()
            }
        ),
        Err(Error::InvalidConv2d {
            reason: "channel/group geometry",
            ..
        })
    ));

    for options in [
        Conv2dOptions {
            groups: 0,
            ..Default::default()
        },
        Conv2dOptions {
            stride: [0, 1],
            ..Default::default()
        },
        Conv2dOptions {
            dilation: [1, 0],
            ..Default::default()
        },
    ] {
        let mut graph = Graph::new();
        let input = graph.input("x", [1, 1, 2, 2]);
        let weight = graph.input("w", [1, 1, 1, 1]);
        assert!(matches!(
            graph.conv2d(input, weight, None, options),
            Err(Error::InvalidConv2d {
                reason: "groups, stride, and dilation must be positive",
                ..
            })
        ));
    }

    let mut graph = Graph::new();
    let input = graph.input("x", [1, 1, 1, 1]);
    let weight = graph.input("w", [1, 1, 2, 2]);
    assert!(matches!(
        graph.conv2d(input, weight, None, Conv2dOptions::default()),
        Err(Error::InvalidConv2d {
            reason: "kernel exceeds padded input",
            ..
        })
    ));
    assert!(matches!(
        graph.conv2d(
            input,
            weight,
            None,
            Conv2dOptions {
                padding: [usize::MAX, 1, 0, 0],
                ..Default::default()
            }
        ),
        Err(Error::ShapeOverflow(_))
    ));

    let mut graph = Graph::new();
    let input = graph.input("x", [1, 1, 2, 2]);
    let weight = graph.input("w", [2, 1, 1, 1]);
    let bias = graph.input("b", [1, 2]);
    assert!(matches!(
        graph.conv2d(input, weight, Some(bias), Conv2dOptions::default()),
        Err(Error::InvalidConv2d {
            reason: "bias must be [output_channels]",
            ..
        })
    ));
}

#[test]
fn transpose_conv1d_preflights_delegated_geometry_before_reshaping() {
    let mut malformed = Graph::new();
    let input = malformed.input("x", [1, 1, 2]);
    let weight = malformed.input("w", [1, 1, 2]);
    let original_nodes = malformed.node_count();
    assert!(matches!(
        malformed.conv_transpose1d(
            input,
            weight,
            None,
            crate::ConvTranspose1dOptions {
                groups: 0,
                ..Default::default()
            },
        ),
        Err(Error::InvalidConv2d {
            reason: "invalid transpose convolution geometry",
            ..
        })
    ));
    assert_eq!(malformed.node_count(), original_nodes);

    let mut valid = Graph::new();
    let input = valid.input("x", [1, 1, 2]);
    let weight = valid.input("w", [1, 1, 2]);
    let output = valid
        .conv_transpose1d(
            input,
            weight,
            None,
            crate::ConvTranspose1dOptions {
                stride: 2,
                output_padding: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        CpuBackend
            .execute(
                &valid,
                output,
                &HashMap::from([
                    ("x".into(), f32_data([1, 1, 2], &[1., 2.])),
                    ("w".into(), f32_data([1, 1, 2], &[1., 1.])),
                ]),
            )
            .unwrap()
            .to_vec_f64(),
        vec![1., 1., 2., 2., 0.]
    );
}

#[test]
fn conv2d_zero_batch_and_spatial_contract() {
    let mut graph = Graph::new();
    let x = graph.input("x", [0, 1, 2, 2]);
    let w = graph.input("w", [1, 1, 1, 1]);
    let y = graph.conv2d(x, w, None, Conv2dOptions::default()).unwrap();
    let output = execute(
        &graph,
        y,
        f32_data([0, 1, 2, 2], &[]),
        f32_data([1, 1, 1, 1], &[1.]),
        None,
    );
    assert_eq!(output.shape(), &Shape::from([0, 1, 2, 2]));
    assert!(output.is_empty());

    // tinygrad's checked-in `_pool` asserts that the effective kernel fits the
    // padded extent; an unpadded zero-spatial input therefore rejects.
    let mut graph = Graph::new();
    let x = graph.input("x", [1, 1, 0, 0]);
    let w = graph.input("w", [1, 1, 1, 1]);
    assert!(matches!(
        graph.conv2d(x, w, None, Conv2dOptions::default()),
        Err(Error::InvalidConv2d {
            reason: "kernel exceeds padded input",
            ..
        })
    ));

    let mut graph = Graph::new();
    let x = graph.input("x", [1, 1, 0, 0]);
    let w = graph.input("w", [1, 1, 1, 1]);
    let b = graph.input("b", [1]);
    let y = graph
        .conv2d(
            x,
            w,
            Some(b),
            Conv2dOptions {
                padding: [1, 1, 1, 1],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        execute(
            &graph,
            y,
            f32_data([1, 1, 0, 0], &[]),
            f32_data([1, 1, 1, 1], &[5.]),
            Some(f32_data([1], &[3.]))
        ),
        f32_data([1, 1, 2, 2], &[3., 3., 3., 3.]),
    );
}

fn analytic_gradients(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    b: &[f32],
    options: Conv2dOptions,
) -> (TensorData, TensorData, TensorData) {
    let mut graph = Graph::new();
    let input = graph.input("x", x_shape);
    let weight = graph.input("w", w_shape);
    let bias = graph.input("b", [w_shape[0]]);
    let output = graph.conv2d(input, weight, Some(bias), options).unwrap();
    let loss = all_sum(&mut graph, output);
    let gx = graph.grad(loss, input).unwrap();
    let gw = graph.grad(loss, weight).unwrap();
    let gb = graph.grad(loss, bias).unwrap();
    (
        execute(
            &graph,
            gx,
            f32_data(x_shape, x),
            f32_data(w_shape, w),
            Some(f32_data([w_shape[0]], b)),
        ),
        execute(
            &graph,
            gw,
            f32_data(x_shape, x),
            f32_data(w_shape, w),
            Some(f32_data([w_shape[0]], b)),
        ),
        execute(
            &graph,
            gb,
            f32_data(x_shape, x),
            f32_data(w_shape, w),
            Some(f32_data([w_shape[0]], b)),
        ),
    )
}

fn loss_value(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    b: &[f32],
    options: Conv2dOptions,
) -> f32 {
    let mut graph = Graph::new();
    let input = graph.input("x", x_shape);
    let weight = graph.input("w", w_shape);
    let bias = graph.input("b", [w_shape[0]]);
    let y = graph.conv2d(input, weight, Some(bias), options).unwrap();
    let loss = all_sum(&mut graph, y);
    execute(
        &graph,
        loss,
        f32_data(x_shape, x),
        f32_data(w_shape, w),
        Some(f32_data([w_shape[0]], b)),
    )
    .scalar_at(0)
    .as_f64() as f32
}

fn assert_finite_difference(analytic: &TensorData, values: &[f32], loss: impl Fn(&[f32]) -> f32) {
    let epsilon = 1e-3;
    for index in 0..values.len() {
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let numeric = (loss(&plus) - loss(&minus)) / (2. * epsilon);
        let actual = analytic.scalar_at(index).as_f64() as f32;
        assert!(
            (actual - numeric).abs() < 2e-2,
            "index {index}: analytic {actual}, numeric {numeric}"
        );
    }
}

fn finite_difference_case(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    b: &[f32],
    options: Conv2dOptions,
) {
    let (gx, gw, gb) = analytic_gradients(x, x_shape, w, w_shape, b, options);
    assert_finite_difference(&gx, x, |candidate| {
        loss_value(candidate, x_shape, w, w_shape, b, options)
    });
    assert_finite_difference(&gw, w, |candidate| {
        loss_value(x, x_shape, candidate, w_shape, b, options)
    });
    assert_finite_difference(&gb, b, |candidate| {
        loss_value(x, x_shape, w, w_shape, candidate, options)
    });
}

#[test]
fn conv2d_plain_gradients_match_central_differences() {
    finite_difference_case(
        &[0.2, -0.1, 0.4, 0.3, 0.5, -0.2, 0.1, 0.6, -0.4],
        [1, 1, 3, 3],
        &[0.7, -0.3, 0.2, 0.5],
        [1, 1, 2, 2],
        &[0.1],
        Conv2dOptions::default(),
    );
}

#[test]
fn conv2d_grouped_and_asymmetric_strided_dilated_gradients_match_central_differences() {
    finite_difference_case(
        &[0.2, -0.1, 0.4, 0.3, 0.5, -0.2, 0.1, 0.6],
        [1, 2, 2, 2],
        &[0.7, -0.3],
        [2, 1, 1, 1],
        &[0.1, -0.2],
        Conv2dOptions {
            groups: 2,
            ..Default::default()
        },
    );
    finite_difference_case(
        &[0.2, -0.1, 0.4, 0.3, 0.5, -0.2, 0.1, 0.6, -0.4],
        [1, 1, 3, 3],
        &[0.7, -0.3, 0.2, 0.5],
        [1, 1, 2, 2],
        &[0.1],
        Conv2dOptions {
            stride: [2, 2],
            dilation: [2, 2],
            padding: [1, 2, 1, 2],
            groups: 1,
        },
    );
}

#[test]
fn conv2d_trace_exposes_labels_shapes_and_dtypes() {
    let mut graph = Graph::new();
    let x = graph.input("x", [1, 1, 2, 2]);
    let w = graph.input("w", [1, 1, 1, 1]);
    let b = graph.input("b", [1]);
    let y = graph
        .conv2d(x, w, Some(b), Conv2dOptions::default())
        .unwrap();
    let forward_nodes = graph.node_count();
    let loss = all_sum(&mut graph, y);
    let gradient = graph.grad(loss, x).unwrap();
    let trace = graph.trace(gradient).unwrap();
    assert!(
        !trace
            .steps
            .iter()
            .any(|step| step.operation.starts_with("conv2d"))
    );
    assert!(trace.steps.iter().any(|step| {
        step.node.index() >= forward_nodes && step.operation.starts_with("expand(")
    }));
    assert!(
        !trace
            .steps
            .iter()
            .any(|step| step.operation.starts_with("reduce_grad"))
    );
    assert!(trace.to_string().contains("F32"));
}

#[test]
fn conv2d_rejects_non_float_gradients_and_runtime_bias_dtype_mismatches() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1, 1, 1, 1], DType::I32);
    let w = graph.input_dtype("w", [1, 1, 1, 1], DType::I32);
    let y = graph.conv2d(x, w, None, Conv2dOptions::default()).unwrap();
    let loss = all_sum(&mut graph, y);
    assert!(matches!(
        graph.grad(loss, x),
        Err(Error::NonDifferentiableTarget(node)) if node == loss
    ));

    let mut graph = Graph::new();
    let x = graph.input("x", [1, 1, 1, 1]);
    let w = graph.input("w", [1, 1, 1, 1]);
    let b = graph.input_dtype("b", [1], DType::F32);
    let y = graph
        .conv2d(x, w, Some(b), Conv2dOptions::default())
        .unwrap();
    let inputs = HashMap::from([
        ("x".into(), f32_data([1, 1, 1, 1], &[1.])),
        ("w".into(), f32_data([1, 1, 1, 1], &[2.])),
        ("b".into(), typed_data([1], DType::I32, [Scalar::I(3)])),
    ]);
    assert!(
        matches!(CpuBackend.execute(&graph, y, &inputs), Err(Error::InputDType { name, expected: DType::F32, actual: DType::I32 }) if name == "b")
    );
}
