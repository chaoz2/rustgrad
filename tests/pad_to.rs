use rustgrad::{Backend, CpuBackend, DType, Error, Graph, Scalar, Shape, TensorData};
use std::collections::HashMap;

fn data(shape: impl Into<Shape>, values: impl IntoIterator<Item = f32>) -> TensorData {
    TensorData::new(shape, values.into_iter().collect()).unwrap()
}

fn execute(graph: &Graph, output: rustgrad::NodeId, input: TensorData) -> TensorData {
    CpuBackend
        .execute(graph, output, &HashMap::from([("input".into(), input)]))
        .unwrap()
}

#[test]
fn pad_to_right_pads_to_tinygrad_target_shape_and_keeps_trace_transparent() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 2]);
    let output = graph.pad_to(input, [3, 4], Scalar::F(-1.0)).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::from([3, 4]));
    assert!(graph.trace(output).unwrap().to_string().contains("pad"));
    assert_eq!(
        execute(&graph, output, data([2, 2], [1., 2., 3., 4.])),
        data(
            [3, 4],
            [1., 2., -1., -1., 3., 4., -1., -1., -1., -1., -1., -1.],
        )
    );
}

#[test]
fn pad_to_preserves_dtype_zero_domains_and_matching_shape_identity() {
    let mut bool_graph = Graph::new();
    let input = bool_graph.input_dtype("input", [0, 1], DType::Bool);
    let output = bool_graph.pad_to(input, [0, 3], Scalar::Bool(true)).unwrap();
    assert_eq!(
        execute(
            &bool_graph,
            output,
            TensorData::from_scalars([0, 1], DType::Bool, []).unwrap(),
        ),
        TensorData::from_scalars([0, 3], DType::Bool, []).unwrap()
    );

    let mut identity_graph = Graph::new();
    let input = identity_graph.input("input", [2, 3]);
    let before = identity_graph.node_count();
    assert_eq!(
        identity_graph.pad_to(input, [2, 3], Scalar::F(0.0)).unwrap(),
        input
    );
    assert_eq!(identity_graph.node_count(), before);
}

#[test]
fn pad_to_rejects_rank_or_shrinking_targets_before_graph_mutation() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let before = graph.node_count();
    assert!(matches!(
        graph.pad_to(input, [2], Scalar::F(0.0)),
        Err(Error::InvalidMovementRank { op: "pad_to", .. })
    ));
    assert!(matches!(
        graph.pad_to(input, [1, 3], Scalar::F(0.0)),
        Err(Error::InvalidReshape { .. })
    ));
    assert_eq!(graph.node_count(), before);
}

#[test]
fn pad_to_inherits_pad_reverse_mode_for_the_original_extent() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let padded = graph.pad_to(input, [4], Scalar::F(0.0)).unwrap();
    let loss = graph.sum_all(padded).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(
        execute(&graph, gradient, data([2], [3., 7.])),
        data([2], [1., 1.])
    );
}
