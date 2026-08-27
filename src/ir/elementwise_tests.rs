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
