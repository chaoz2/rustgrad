use crate::{Backend, CpuBackend, DType, Error, Graph, Scalar, TensorData};
use std::collections::HashMap;

fn execute(graph: &Graph, output: crate::NodeId, value: TensorData) -> crate::Result<TensorData> {
    CpuBackend.execute(graph, output, &HashMap::from([("x".into(), value)]))
}

#[test]
fn tensor_guard_preserves_valid_distribution_storage() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F32);
    let guard = graph.tensor_guard_distribution(input, -1).unwrap();
    let value = TensorData::from_scalars(
        [2, 2], DType::F32,
        [Scalar::F(0.0), Scalar::F(1.0), Scalar::F(2.0), Scalar::F(3.0)],
    ).unwrap();
    assert_eq!(execute(&graph, guard, value.clone()).unwrap().storage(), value.storage());
    assert!(graph.trace(guard).unwrap().to_string().contains("tensor_guard"));
}

#[test]
fn tensor_guard_reports_first_invalid_lane_deterministically() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F32);
    let guard = graph.tensor_guard_distribution(input, 1).unwrap();
    let nan = TensorData::from_scalars(
        [2, 2], DType::F32,
        [Scalar::F(1.0), Scalar::F(f64::NAN), Scalar::F(-1.0), Scalar::F(2.0)],
    ).unwrap();
    assert!(matches!(execute(&graph, guard, nan), Err(Error::TensorGuard { reason: "value is not finite", row: 0, index: 1 })));
    let zero = TensorData::from_scalars(
        [2, 2], DType::F32,
        [Scalar::F(0.0), Scalar::F(0.0), Scalar::F(1.0), Scalar::F(0.0)],
    ).unwrap();
    assert!(matches!(execute(&graph, guard, zero), Err(Error::TensorGuard { reason: "row has nonpositive total", row: 0, index: 0 })));
}

#[test]
fn tensor_guard_preflight_is_atomic() {
    let mut graph = Graph::new();
    let integer = graph.input_dtype("x", [2], DType::I32);
    let before = graph.node_count();
    assert!(graph.tensor_guard_distribution(integer, 0).is_err());
    assert_eq!(graph.node_count(), before);
    let float = graph.input_dtype("f", [2], DType::F32);
    let before = graph.node_count();
    assert!(graph.tensor_guard_distribution(float, 2).is_err());
    assert_eq!(graph.node_count(), before);
}
