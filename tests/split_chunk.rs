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
fn split_matches_tinygrad_scalar_and_explicit_section_tables() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 5]);
    let scalar = graph.split(input, 2usize, -1).unwrap();
    let explicit = graph.split(input, vec![1, 0, 4], 1).unwrap();
    let values = f32_data([2, 5], (0..10).map(|value| value as f32));

    let scalar_expected = [
        f32_data([2, 2], [0., 1., 5., 6.]),
        f32_data([2, 2], [2., 3., 7., 8.]),
        f32_data([2, 1], [4., 9.]),
    ];
    assert_eq!(scalar.len(), scalar_expected.len());
    for (output, expected) in scalar.iter().zip(scalar_expected) {
        assert_eq!(execute(&graph, *output, values.clone()), expected);
        assert!(graph.trace(*output).unwrap().to_string().contains("shrink"));
    }

    let explicit_expected = [
        f32_data([2, 1], [0., 5.]),
        f32_data([2, 0], []),
        f32_data([2, 4], [1., 2., 3., 4., 6., 7., 8., 9.]),
    ];
    assert_eq!(explicit.len(), explicit_expected.len());
    for (output, expected) in explicit.iter().zip(explicit_expected) {
        assert_eq!(execute(&graph, *output, values.clone()), expected);
    }
}

#[test]
fn chunk_matches_tinygrad_partial_and_zero_axis_contracts() {
    let mut graph = Graph::new();
    let input = graph.input("input", [13]);
    let chunks = graph.chunk(input, 6, -1).unwrap();
    let values = f32_data([13], (0..13).map(|value| value as f32));
    let expected = [
        f32_data([3], [0., 1., 2.]),
        f32_data([3], [3., 4., 5.]),
        f32_data([3], [6., 7., 8.]),
        f32_data([3], [9., 10., 11.]),
        f32_data([1], [12.]),
    ];
    assert_eq!(chunks.len(), expected.len());
    for (output, expected) in chunks.iter().zip(expected) {
        assert_eq!(execute(&graph, *output, values.clone()), expected);
    }

    let mut zero_graph = Graph::new();
    let empty = zero_graph.input("input", [0, 2]);
    let zero_chunks = zero_graph.chunk(empty, 3, 0).unwrap();
    assert_eq!(zero_chunks.len(), 3);
    for output in zero_chunks {
        assert_eq!(
            execute(&zero_graph, output, f32_data([0, 2], [])),
            f32_data([0, 2], [])
        );
    }

    // `Tensor([].reshape(0)).split(0)` is an empty tuple in tinygrad.
    let split_empty = zero_graph.split(empty, 0usize, 0).unwrap();
    assert!(split_empty.is_empty());
}

#[test]
fn split_preserves_bool_payloads_and_rejects_malformed_specs_before_graph_mutation() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::Bool);
    let pieces = graph.split(input, vec![1, 2, 1], 0).unwrap();
    let values = TensorData::from_scalars(
        [4],
        DType::Bool,
        [true, false, true, false].map(Scalar::Bool),
    )
    .unwrap();
    assert_eq!(
        execute(&graph, pieces[1], values),
        TensorData::from_scalars([2], DType::Bool, [false, true].map(Scalar::Bool)).unwrap()
    );

    let before = graph.node_count();
    assert!(matches!(
        graph.split(input, 0usize, 0),
        Err(Error::InvalidSplit { .. })
    ));
    assert!(matches!(
        graph.split(input, vec![1, 1], 0),
        Err(Error::InvalidSplit { .. })
    ));
    assert!(matches!(
        graph.chunk(input, 0, 0),
        Err(Error::InvalidSplit { .. })
    ));
    assert!(matches!(
        graph.split(input, 1usize, -2),
        Err(Error::InvalidAxis { .. })
    ));
    assert_eq!(graph.node_count(), before);
}
