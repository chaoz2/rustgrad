use crate::{Backend, CpuBackend, Graph, Shape, TensorData};
use std::collections::HashMap;

#[test]
fn dot_keeps_leading_axes_as_an_outer_product() {
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [2, 2]);
    let rhs = graph.input("rhs", [3, 2, 1]);
    let output = graph.dot(lhs, rhs).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 3, 1]));
    let values = HashMap::from([
        (
            "lhs".into(),
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::new([3, 2, 1], vec![1., 1., 2., 2., 3., 3.]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, output, &values).unwrap(),
        TensorData::new([2, 3, 1], vec![3., 6., 9., 7., 14., 21.]).unwrap()
    );
    assert!(graph.trace(output).unwrap().to_string().contains("einsum"));
}

#[test]
fn dot_rejects_scalars_and_contract_mismatch() {
    let mut graph = Graph::new();
    let scalar = graph.input("scalar", []);
    let vector = graph.input("vector", [2]);
    assert!(graph.dot(scalar, vector).is_err());
    let other = graph.input("other", [3]);
    assert!(graph.dot(vector, other).is_err());
}
