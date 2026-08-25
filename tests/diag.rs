use rustgrad::{Backend, CpuBackend, DType, Error, Graph, Scalar, Shape, TensorData};
use std::collections::HashMap;

fn f32_data(shape: impl Into<Shape>, values: impl IntoIterator<Item = f32>) -> TensorData {
    TensorData::new(shape, values.into_iter().collect()).unwrap()
}

fn execute(graph: &Graph, output: rustgrad::NodeId, input: TensorData) -> TensorData {
    CpuBackend
        .execute(graph, output, &HashMap::from([("input".into(), input)]))
        .unwrap()
}

#[test]
fn diag_matches_tinygrad_values_and_composition_trace() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let output = graph.diag(input).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::from([3, 3]));
    assert_eq!(
        execute(&graph, output, f32_data([3], [1., 2., 3.])),
        f32_data([3, 3], [1., 0., 0., 0., 2., 0., 0., 0., 3.])
    );
    let trace = graph.trace(output).unwrap().to_string();
    assert!(trace.contains("pad"));
    assert!(trace.contains("shrink"));
}

#[test]
fn diag_preserves_typed_values_and_zero_extent_contract() {
    let mut integer_graph = Graph::new();
    let input = integer_graph.input_dtype("input", [2], DType::U64);
    let output = integer_graph.diag(input).unwrap();
    assert_eq!(
        execute(
            &integer_graph,
            output,
            TensorData::from_scalars([2], DType::U64, [Scalar::U(u64::MAX), Scalar::U(7)],)
                .unwrap(),
        ),
        TensorData::from_scalars(
            [2, 2],
            DType::U64,
            [
                Scalar::U(u64::MAX),
                Scalar::U(0),
                Scalar::U(0),
                Scalar::U(7)
            ],
        )
        .unwrap()
    );

    let mut empty_graph = Graph::new();
    let empty = empty_graph.input("input", [0]);
    let empty_diag = empty_graph.diag(empty).unwrap();
    assert_eq!(empty_graph.shape(empty_diag).unwrap(), &Shape::from([0, 0]));
    assert_eq!(
        execute(&empty_graph, empty_diag, f32_data([0], [])),
        f32_data([0, 0], [])
    );
}

#[test]
fn diag_rejects_non_vectors_before_graph_mutation() {
    let mut graph = Graph::new();
    let scalar = graph.input("scalar", []);
    let matrix = graph.input("matrix", [2, 2]);
    let before = graph.node_count();
    for input in [scalar, matrix] {
        assert!(matches!(
            graph.diag(input),
            Err(Error::InvalidDiagonal { .. })
        ));
    }
    assert_eq!(graph.node_count(), before);
}
