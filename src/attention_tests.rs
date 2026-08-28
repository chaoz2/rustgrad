use crate::{
    AttentionOptions, Backend, CpuBackend, DType, Error, Graph, ReduceKind, Scalar, Shape,
    TensorData,
};
use std::collections::HashMap;

fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
    TensorData::new(shape, values.to_vec()).unwrap()
}

fn execute(
    graph: &Graph,
    output: crate::NodeId,
    inputs: HashMap<String, TensorData>,
) -> TensorData {
    CpuBackend.execute(graph, output, &inputs).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn logsumexp_is_stable_multi_axis_signed_and_differentiable() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let along_last = graph.logsumexp(x, Some(vec![-1]), true).unwrap();
    let all = graph.logsumexp(x, None, false).unwrap();
    let dx = graph.grad(all, x).unwrap();
    let input = data([2, 2], &[1000., 999., -1000., -1001.]);
    assert_close(
        &execute(
            &graph,
            along_last,
            HashMap::from([("x".into(), input.clone())]),
        )
        .to_vec_f64(),
        &[1000.31323, -999.68677],
        2e-3,
    );
    assert_close(
        &execute(&graph, all, HashMap::from([("x".into(), input.clone())])).to_vec_f64(),
        &[1000.31323],
        2e-3,
    );
    assert_close(
        &execute(&graph, dx, HashMap::from([("x".into(), input)])).to_vec_f64(),
        &[0.73106, 0.26894, 0., 0.],
        2e-3,
    );
    let values = vec![0.2, -0.4, 1.1, 0.7];
    let analytic = execute(
        &graph,
        dx,
        HashMap::from([("x".into(), data([2, 2], &values))]),
    )
    .to_vec_f64();
    let epsilon = 1e-3;
    for index in 0..values.len() {
        let mut positive = values.clone();
        let mut negative = values.clone();
        positive[index] += epsilon;
        negative[index] -= epsilon;
        let numeric = (execute(
            &graph,
            all,
            HashMap::from([("x".into(), data([2, 2], &positive))]),
        )
        .scalar_at(0)
        .as_f64()
            - execute(
                &graph,
                all,
                HashMap::from([("x".into(), data([2, 2], &negative))]),
            )
            .scalar_at(0)
            .as_f64())
            / f64::from(2. * epsilon);
        assert!((analytic[index] - numeric).abs() < 2e-3);
    }
}

#[test]
fn logsumexp_preflights_axes_and_keeps_established_nonfinite_boundaries() {
    let mut malformed = Graph::new();
    let input = malformed.input("input", [2, 0]);
    let original_nodes = malformed.node_count();
    assert!(matches!(
        malformed.logsumexp(input, Some(vec![-1, 1]), false),
        Err(Error::InvalidReductionAxes { .. })
    ));
    assert_eq!(malformed.node_count(), original_nodes);
    assert!(matches!(
        malformed.logsumexp(input, Some(vec![isize::MIN]), false),
        Err(Error::InvalidReductionAxes { .. })
    ));
    assert_eq!(malformed.node_count(), original_nodes);
    assert!(matches!(
        malformed.logsumexp(input, Some(vec![-1]), false),
        Err(Error::EmptyReduction { op: "max", .. })
    ));
    assert_eq!(malformed.node_count(), original_nodes);

    let mut graph = Graph::new();
    let values = graph.input("values", [2]);
    let output = graph.logsumexp(values, Some(vec![-1]), false).unwrap();
    assert!(execute(
        &graph,
        output,
        HashMap::from([("values".into(), data([2], &[f32::NEG_INFINITY; 2]))]),
    )
    .scalar_at(0)
    .as_f64()
    .is_nan());
    assert!(execute(
        &graph,
        output,
        HashMap::from([("values".into(), data([2], &[f32::NAN, 0.]))]),
    )
    .scalar_at(0)
    .as_f64()
    .is_nan());

    let mut integer_graph = Graph::new();
    let integer = integer_graph.input_dtype("integer", [2], DType::I32);
    let output = integer_graph
        .logsumexp(integer, Some(vec![-1]), false)
        .unwrap();
    assert_eq!(integer_graph.dtype(output).unwrap(), DType::F32);
}

#[test]
fn softmax_and_log_softmax_are_stable_and_promote_requested_dtype() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::F16);
    let softmax = graph.softmax(x, -1, Some(DType::F32)).unwrap();
    let log_softmax = graph.log_softmax(x, -1, Some(DType::F32)).unwrap();
    assert_eq!(graph.dtype(softmax).unwrap(), DType::F32);
    let input = TensorData::from_scalars(
        [2, 3],
        DType::F16,
        [1000., 999., 998., 1., 1., 1.].map(crate::Scalar::F),
    )
    .unwrap();
    assert_close(
        &execute(
            &graph,
            softmax,
            HashMap::from([("x".into(), input.clone())]),
        )
        .to_vec_f64(),
        &[0.66524, 0.24473, 0.09003, 1. / 3., 1. / 3., 1. / 3.],
        2e-3,
    );
    assert_close(
        &execute(&graph, log_softmax, HashMap::from([("x".into(), input)])).to_vec_f64(),
        &[-0.40761, -1.40761, -2.40761, -1.09861, -1.09861, -1.09861],
        2e-3,
    );
}

#[test]
fn softmax_preflights_requested_dtype_before_stable_lowering() {
    let mut requested_integer = Graph::new();
    let input = requested_integer.input("input", [2]);
    let output = requested_integer.softmax(input, -1, Some(DType::I32)).unwrap();
    // Tinygrad permits a requested exact dtype, then Exp lifts that storage to
    // F32. It is not a rejection and the final probabilities are F32.
    assert_eq!(requested_integer.dtype(output).unwrap(), DType::F32);

    let mut valid = Graph::new();
    let input = valid.input("input", [2]);
    let output = valid.softmax(input, -1, Some(DType::F32)).unwrap();
    assert_close(
        &execute(
            &valid,
            output,
            HashMap::from([("input".into(), data([2], &[0., 1.]))]),
        )
        .to_vec_f64(),
        &[0.26894, 0.73106],
        2e-3,
    );
}

#[test]
fn softmax_uses_detached_typed_exp2_sum_reciprocal_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F64);
    let output = graph.softmax(input, -1, None).unwrap();
    let trace = graph.trace(output).unwrap();
    assert!(trace.steps.iter().any(|step| step.operation.starts_with("detach(")));
    assert!(trace.steps.iter().any(|step| step.operation.starts_with("exp2(")));
    assert!(trace.steps.iter().any(|step| step.operation.starts_with("reciprocal(")));
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [2, 2],
            DType::F64,
            [
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap(),
    )]);
    let values = execute(&graph, output, bindings.clone());
    assert_eq!(values.scalar_at(0).as_f64(), 0.5);
    assert_eq!(values.scalar_at(1).as_f64(), 0.5);
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_nan());
    let gradients = execute(&graph, gradient, bindings).to_vec_f64();
    assert_eq!(&gradients[..2], &[0., 0.]);
    assert!(gradients[2].is_nan() && gradients[3].is_nan());

    for (input_dtype, requested, expected) in [
        (DType::F16, None, DType::F16),
        (DType::BF16, None, DType::BF16),
        (DType::F32, None, DType::F32),
        (DType::F64, None, DType::F64),
        (DType::I64, None, DType::F32),
        (DType::U64, None, DType::F32),
        (DType::F16, Some(DType::F64), DType::F64),
        (DType::F32, Some(DType::I32), DType::F32),
    ] {
        let mut typed = Graph::new();
        let x = typed.input_dtype("x", [], input_dtype);
        let output = typed.softmax(x, -1, requested).unwrap();
        assert_eq!(typed.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(typed.dtype(output).unwrap(), expected);
    }
    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [2, 0], DType::F16);
    let nodes = empty.node_count();
    let output = empty.softmax(x, -1, None).unwrap();
    assert_eq!(output, x);
    assert_eq!(empty.node_count(), nodes);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2]);
    let nodes = malformed.node_count();
    assert!(malformed.softmax(x, 1, None).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn log_softmax_uses_detached_typed_exp2_log2_composition_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F64);
    let output = graph.log_softmax(input, -1, None).unwrap();
    let trace = graph.trace(output).unwrap();
    assert!(trace.steps.iter().any(|step| step.operation.starts_with("detach(")));
    assert!(trace.steps.iter().any(|step| step.operation.starts_with("exp2(")));
    assert!(trace.steps.iter().any(|step| step.operation.starts_with("log2(")));
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [2, 2],
            DType::F64,
            [
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap(),
    )]);
    let values = execute(&graph, output, bindings.clone());
    assert_close(&values.to_vec_f64()[..2], &[-std::f64::consts::LN_2; 2], 1e-12);
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_nan());
    let gradients = execute(&graph, gradient, bindings).to_vec_f64();
    assert_eq!(&gradients[..2], &[0., 0.]);
    assert!(gradients[2].is_nan() && gradients[3].is_nan());

    for (input_dtype, requested, expected) in [
        (DType::F16, None, DType::F16),
        (DType::BF16, None, DType::BF16),
        (DType::F32, None, DType::F32),
        (DType::F64, None, DType::F64),
        (DType::I64, None, DType::F32),
        (DType::U64, None, DType::F32),
        (DType::F16, Some(DType::F64), DType::F64),
        (DType::F32, Some(DType::I32), DType::F32),
    ] {
        let mut typed = Graph::new();
        let x = typed.input_dtype("x", [], input_dtype);
        let output = typed.log_softmax(x, -1, requested).unwrap();
        assert_eq!(typed.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(typed.dtype(output).unwrap(), expected);
    }
    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [2, 0], DType::F16);
    let nodes = empty.node_count();
    let output = empty.log_softmax(x, -1, None).unwrap();
    assert_eq!(output, x);
    assert_eq!(empty.node_count(), nodes);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2]);
    let nodes = malformed.node_count();
    assert!(malformed.log_softmax(x, 1, None).is_err());
    assert_eq!(malformed.node_count(), nodes);
}

#[test]
fn attention_causal_boolean_and_additive_masks_match_fixtures() {
    let mut graph = Graph::new();
    let q = graph.input("q", [1, 1, 2, 2]);
    let k = graph.input("k", [1, 1, 2, 2]);
    let v = graph.input("v", [1, 1, 2, 1]);
    let mask = graph.input_dtype("mask", [2, 2], DType::Bool);
    let additive = graph.input("additive", [2, 2]);
    let causal = graph
        .scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            AttentionOptions {
                is_causal: true,
                ..Default::default()
            },
        )
        .unwrap();
    let boolean = graph
        .scaled_dot_product_attention(q, k, v, Some(mask), AttentionOptions::default())
        .unwrap();
    let biased = graph
        .scaled_dot_product_attention(
            q,
            k,
            v,
            Some(additive),
            AttentionOptions {
                scale: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
    let inputs = HashMap::from([
        ("q".into(), data([1, 1, 2, 2], &[1., 0., 0., 1.])),
        ("k".into(), data([1, 1, 2, 2], &[1., 0., 0., 1.])),
        ("v".into(), data([1, 1, 2, 1], &[10., 20.])),
        (
            "mask".into(),
            TensorData::from_scalars(
                [2, 2],
                DType::Bool,
                [true, false, true, true].map(crate::Scalar::Bool),
            )
            .unwrap(),
        ),
        ("additive".into(), data([2, 2], &[0., -10., 0., 0.])),
    ]);
    assert_close(
        &execute(&graph, causal, inputs.clone()).to_vec_f64(),
        &[10., 16.6976],
        2e-3,
    );
    assert_close(
        &execute(&graph, boolean, inputs.clone()).to_vec_f64(),
        &[10., 16.6976],
        2e-3,
    );
    assert_close(
        &execute(&graph, biased, inputs).to_vec_f64(),
        &[10.00017, 17.3106],
        2e-3,
    );
}

#[test]
fn attention_supports_grouped_query_and_qkv_gradients() {
    let mut graph = Graph::new();
    let q = graph.input("q", [1, 2, 1, 2]);
    let k = graph.input("k", [1, 1, 2, 2]);
    let v = graph.input("v", [1, 1, 2, 1]);
    let output = graph
        .scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            AttentionOptions {
                enable_gqa: true,
                ..Default::default()
            },
        )
        .unwrap();
    let loss = graph.reduce(output, ReduceKind::Sum, None, false).unwrap();
    let dq = graph.grad(loss, q).unwrap();
    let dk = graph.grad(loss, k).unwrap();
    let dv = graph.grad(loss, v).unwrap();
    let inputs = HashMap::from([
        ("q".into(), data([1, 2, 1, 2], &[1., 0., 0., 1.])),
        ("k".into(), data([1, 1, 2, 2], &[1., 0., 0., 1.])),
        ("v".into(), data([1, 1, 2, 1], &[1., 3.])),
    ]);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([1, 2, 1, 1]));
    for grad in [dq, dk, dv] {
        assert!(
            execute(&graph, grad, inputs.clone())
                .to_vec_f64()
                .iter()
                .all(|value| value.is_finite())
        );
    }
    let epsilon = 1e-3;
    for (name, gradient, shape, values) in [
        ("q", dq, [1, 2, 1, 2], vec![1., 0., 0., 1.]),
        ("k", dk, [1, 1, 2, 2], vec![1., 0., 0., 1.]),
        ("v", dv, [1, 1, 2, 1], vec![1., 3.]),
    ] {
        let analytic = execute(&graph, gradient, inputs.clone()).to_vec_f64();
        for index in 0..values.len() {
            let mut positive = values.clone();
            let mut negative = values.clone();
            positive[index] += epsilon;
            negative[index] -= epsilon;
            let mut positive_inputs = inputs.clone();
            positive_inputs.insert(name.into(), data(shape, &positive));
            let mut negative_inputs = inputs.clone();
            negative_inputs.insert(name.into(), data(shape, &negative));
            let numeric = (execute(&graph, loss, positive_inputs).scalar_at(0).as_f64()
                - execute(&graph, loss, negative_inputs).scalar_at(0).as_f64())
                / f64::from(2. * epsilon);
            assert!(
                (analytic[index] - numeric).abs() < 3e-3,
                "{name}[{index}] analytic={}, numeric={numeric}",
                analytic[index]
            );
        }
    }
    let trace = graph.trace(output).unwrap().to_string();
    assert!(trace.contains("matmul") && trace.contains("Max") && trace.contains("exp"));
}

#[test]
fn softmax_gradient_matches_central_difference() {
    let mut graph = Graph::new();
    let x = graph.input("x", [3]);
    let y = graph.softmax(x, -1, None).unwrap();
    let weights = graph.constant(data([3], &[1., -2., 0.5]));
    let weighted = graph.mul(y, weights).unwrap();
    let loss = graph
        .reduce(weighted, ReduceKind::Sum, None, false)
        .unwrap();
    let gradient = graph.grad(loss, x).unwrap();
    let values = vec![0.2, -0.4, 1.1];
    let analytic = execute(
        &graph,
        gradient,
        HashMap::from([("x".into(), data([3], &values))]),
    )
    .to_vec_f64();
    let epsilon = 1e-3;
    for index in 0..values.len() {
        let mut positive = values.clone();
        let mut negative = values.clone();
        positive[index] += epsilon;
        negative[index] -= epsilon;
        let numeric = (execute(
            &graph,
            loss,
            HashMap::from([("x".into(), data([3], &positive))]),
        )
        .scalar_at(0)
        .as_f64()
            - execute(
                &graph,
                loss,
                HashMap::from([("x".into(), data([3], &negative))]),
            )
            .scalar_at(0)
            .as_f64())
            / f64::from(2. * epsilon);
        assert!(
            (analytic[index] - numeric).abs() < 2e-3,
            "index {index}: analytic={}, numeric={numeric}",
            analytic[index]
        );
    }
}

#[test]
fn attention_rejects_invalid_contracts_and_nonzero_dropout() {
    let mut graph = Graph::new();
    let q = graph.input("q", [1, 2]);
    let k = graph.input("k", [1, 2]);
    let v = graph.input("v", [1, 2]);
    assert_eq!(
        graph.scaled_dot_product_attention(q, k, v, None, AttentionOptions::default()),
        Err(Error::InvalidAttention {
            reason: "query, key, and value need rank at least three"
        })
    );

    let mut graph = Graph::new();
    let q = graph.input("q", [1, 1, 1, 2]);
    let k = graph.input("k", [1, 1, 1, 2]);
    let v = graph.input("v", [1, 1, 1, 2]);
    assert_eq!(
        graph.scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            AttentionOptions {
                dropout_p: 0.25,
                training: true,
                ..Default::default()
            }
        ),
        Err(Error::InvalidAttention {
            reason: "training dropout requires an explicit dropout_seed"
        })
    );
    assert!(
        graph
            .scaled_dot_product_attention(
                q,
                k,
                v,
                Some(q),
                AttentionOptions {
                    is_causal: true,
                    ..Default::default()
                }
            )
            .is_err()
    );
}

#[test]
fn attention_preflights_mask_geometry_before_lowering() {
    let mut malformed = Graph::new();
    let query = malformed.input("query", [1, 1, 2, 1]);
    let key = malformed.input("key", [1, 1, 2, 1]);
    let value = malformed.input("value", [1, 1, 2, 1]);
    let mask = malformed.input_dtype("mask", [3], DType::Bool);
    let original_nodes = malformed.node_count();
    assert_eq!(
        malformed.scaled_dot_product_attention(
            query,
            key,
            value,
            Some(mask),
            AttentionOptions::default(),
        ),
        Err(Error::InvalidAttention {
            reason: "attn_mask must broadcast to attention scores"
        })
    );
    assert_eq!(malformed.node_count(), original_nodes);

    let mut valid = Graph::new();
    let query = valid.input("query", [1, 1, 1, 1]);
    let key = valid.input("key", [1, 1, 2, 1]);
    let value = valid.input("value", [1, 1, 2, 1]);
    let mask = valid.input_dtype("mask", [1, 2], DType::Bool);
    let output = valid
        .scaled_dot_product_attention(
            query,
            key,
            value,
            Some(mask),
            AttentionOptions::default(),
        )
        .unwrap();
    assert_close(
        &execute(
            &valid,
            output,
            HashMap::from([
                ("query".into(), data([1, 1, 1, 1], &[1.])),
                ("key".into(), data([1, 1, 2, 1], &[1., 0.])),
                ("value".into(), data([1, 1, 2, 1], &[3., 7.])),
                (
                    "mask".into(),
                    TensorData::from_scalars(
                        [1, 2],
                        DType::Bool,
                        [crate::Scalar::Bool(true), crate::Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .to_vec_f64(),
        &[3.],
        2e-3,
    );
}

#[test]
fn attention_dropout_is_seeded_inverted_and_differentiable() {
    let mut graph = Graph::new();
    let q = graph.input("q", [1, 1, 1, 1]);
    let k = graph.input("k", [1, 1, 2, 1]);
    let v = graph.input("v", [1, 1, 2, 1]);
    let options = AttentionOptions {
        dropout_p: 0.5,
        training: true,
        dropout_seed: Some(99),
        ..Default::default()
    };
    let output = graph
        .scaled_dot_product_attention(q, k, v, None, options)
        .unwrap();
    let loss = graph.reduce(output, ReduceKind::Sum, None, false).unwrap();
    let dv = graph.grad(loss, v).unwrap();
    let inputs = HashMap::from([
        ("q".into(), data([1, 1, 1, 1], &[1.])),
        ("k".into(), data([1, 1, 2, 1], &[1., 0.])),
        ("v".into(), data([1, 1, 2, 1], &[3., 7.])),
    ]);
    let forward = execute(&graph, output, inputs.clone());
    assert_eq!(forward, execute(&graph, output, inputs.clone()));
    assert!(forward.to_vec_f64().iter().all(|value| value.is_finite()));
    assert!(
        execute(&graph, dv, inputs)
            .to_vec_f64()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        graph
            .trace(output)
            .unwrap()
            .to_string()
            .contains("random_Uniform")
    );
}
