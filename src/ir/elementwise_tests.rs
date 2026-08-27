use super::*;
use crate::{Backend, CpuBackend, Error, Scalar};
use std::collections::HashMap;

fn bool_data(shape: impl Into<Shape>, values: impl IntoIterator<Item = bool>) -> TensorData {
    TensorData::from_scalars(shape, DType::Bool, values.into_iter().map(Scalar::Bool)).unwrap()
}

#[test]
fn masked_fill_matches_select_broadcasts_and_vjp() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let mask = graph.constant(bool_data([3], [true, false, true]));
    let fill = graph.constant(TensorData::scalar(-4.0));
    let output = graph.masked_fill(input, mask, fill).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    )]);

    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-4., 2., -4., -4., 5., -4.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0., 1., 0., 0., 1., 0.]
    );
}

#[test]
fn masked_fill_rejects_nonboolean_mask_without_allocating_a_node() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2]);
    let nonboolean_mask = graph.input_dtype("mask", [2], DType::I32);
    let fill = graph.constant(TensorData::scalar(0.0));
    let node_count = graph.node_count();

    assert!(matches!(
        graph.masked_fill(input, nonboolean_mask, fill),
        Err(Error::InvalidLogicalDType {
            op: "select",
            actual: DType::I32,
        })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn clip_is_a_clamp_alias_with_the_existing_vjp() {
    let mut graph = Graph::new();
    let input = graph.input("x", [3]);
    let min = graph.constant(TensorData::scalar(-1.0));
    let max = graph.constant(TensorData::scalar(1.0));
    let output = graph.clip(input, Some(min), Some(max)).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([3], vec![-2., 0.5, 3.]).unwrap(),
    )]);

    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1., 0.5, 1.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0., 1., 0.]
    );
}

#[test]
fn clip_preflights_both_bounds_before_graph_growth() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let valid_min = graph.constant(TensorData::scalar(-1.0));
    let incompatible_max = graph.constant(TensorData::new([2, 2], vec![1.; 4]).unwrap());
    let node_count = graph.node_count();

    assert!(matches!(
        graph.clip(input, Some(valid_min), Some(incompatible_max)),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn clip_rejects_bounds_that_only_conflict_with_each_other_without_graph_growth() {
    let mut graph = Graph::new();
    let input = graph.input("x", [1]);
    let min = graph.constant(TensorData::new([2], vec![-1., -2.]).unwrap());
    let max = graph.constant(TensorData::new([3], vec![1., 2., 3.]).unwrap());
    let node_count = graph.node_count();

    assert!(matches!(
        graph.clip(input, Some(min), Some(max)),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn squeeze_of_a_nonunit_axis_is_a_tinygrad_style_noop() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let node_count = graph.node_count();

    let output = graph.squeeze(input, Some(-1)).unwrap();
    assert_eq!(output, input);
    assert_eq!(graph.node_count(), node_count);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.; 6]
    );
}

#[test]
fn isinf_sign_selection_preserves_tinygrad_predicate_contract() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let positive = graph.isinf_with_signs(input, true, false).unwrap();
    let negative = graph.isinf_with_signs(input, false, true).unwrap();
    let neither = graph.isinf_with_signs(input, false, false).unwrap();
    let both = graph.isinf_with_signs(input, true, true).unwrap();
    let scalar = graph.input_dtype("scalar", [], DType::F32);
    let scalar_positive = graph.isinf_with_signs(scalar, true, false).unwrap();
    let integers = graph.input_dtype("integers", [2], DType::I32);
    let integer_positive = graph.isinf_with_signs(integers, true, false).unwrap();
    let bindings = HashMap::from([
        (
            "input".into(),
            TensorData::from_scalars(
                [5],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(f64::NAN),
                ],
            )
            .unwrap(),
        ),
        ("scalar".into(), TensorData::scalar(f32::INFINITY)),
        (
            "integers".into(),
            TensorData::from_scalars([2], DType::I32, [Scalar::I(-1), Scalar::I(0)]).unwrap(),
        ),
    ]);
    for (node, expected) in [
        (positive, vec![false, false, false, true, false]),
        (negative, vec![true, false, false, false, false]),
        (neither, vec![false; 5]),
        (both, vec![true, false, false, true, false]),
        (integer_positive, vec![false; 2]),
    ] {
        let output = CpuBackend.execute(&graph, node, &bindings).unwrap();
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.storage(), &crate::Storage::Bool(expected));
    }
    assert_eq!(
        CpuBackend.execute(&graph, scalar_positive, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![true])
    );

    assert!(matches!(graph.grad(positive, input), Err(Error::NoGradient(_))));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F32);
    let output = empty.isinf_with_signs(input, false, true).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![])
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.isinf_with_signs(NodeId(usize::MAX), true, false),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn sign_uses_tinygrad_ordered_nan_and_signed_zero_contract() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.sign(input).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let scalar = graph.input_dtype("scalar", [], DType::F32);
    let scalar_output = graph.sign(scalar).unwrap();
    let integers = graph.input_dtype("integers", [3], DType::I32);
    let integer_output = graph.sign(integers).unwrap();
    let bindings = HashMap::from([
        (
            "input".into(),
            TensorData::from_scalars(
                [5],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(f64::NAN),
                ],
            )
            .unwrap(),
        ),
        ("scalar".into(), TensorData::scalar(-0.0)),
        (
            "integers".into(),
            TensorData::from_scalars(
                [3],
                DType::I32,
                [Scalar::I(-3), Scalar::I(0), Scalar::I(4)],
            )
            .unwrap(),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64();
    assert_eq!(values, vec![-1.0, 0.0, 0.0, 1.0, 1.0]);
    assert!(values[1].is_sign_positive());
    assert!(values[2].is_sign_positive());
    let scalar_value = CpuBackend
        .execute(&graph, scalar_output, &bindings)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    assert_eq!(scalar_value, 0.0);
    assert!(scalar_value.is_sign_positive());
    assert_eq!(
        CpuBackend
            .execute(&graph, integer_output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0, 0.0, 1.0]
    );
    assert_eq!(
        CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
        vec![0.0; 5]
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F32);
    let output = empty.sign(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F32);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.sign(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
}
