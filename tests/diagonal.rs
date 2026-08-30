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
fn diagonal_matches_tinygrad_square_rectangular_and_offset_tables() {
    let cases = [
        (vec![3usize, 3], 0, 0, 1, Shape::from([3]), vec![0., 4., 8.]),
        (
            vec![3usize, 5],
            2,
            0,
            1,
            Shape::from([3]),
            vec![2., 8., 14.],
        ),
        (
            vec![5usize, 5],
            -1,
            0,
            1,
            Shape::from([4]),
            vec![5., 11., 17., 23.],
        ),
    ];
    for (shape, offset, dim1, dim2, output_shape, expected) in cases {
        let mut graph = Graph::new();
        let input = graph.input("input", shape.clone());
        let output = graph.diagonal(input, offset, dim1, dim2).unwrap();
        let input_len = shape.iter().product::<usize>();
        assert_eq!(graph.shape(output).unwrap(), &output_shape);
        assert_eq!(
            execute(
                &graph,
                output,
                f32_data(shape, (0..input_len).map(|x| x as f32))
            ),
            f32_data(output_shape, expected)
        );
        let trace = graph.trace(output).unwrap().to_string();
        assert!(trace.contains("pad"));
        assert!(!trace.contains("static_index"));
    }
}

#[test]
fn diagonal_preserves_batch_axis_order_empty_domains_and_bool_storage() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3, 4]);
    let output = graph.diagonal(input, 0, -2, -1).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 3]));
    assert_eq!(
        execute(
            &graph,
            output,
            f32_data([2, 3, 4], (0..24).map(|x| x as f32)),
        ),
        f32_data([2, 3], [0., 5., 10., 12., 17., 22.])
    );

    let mut zero_graph = Graph::new();
    let zero = zero_graph.input("input", [5, 0, 3]);
    let zero_output = zero_graph.diagonal(zero, 0, -2, -1).unwrap();
    assert_eq!(zero_graph.shape(zero_output).unwrap(), &Shape::from([5, 0]));
    assert_eq!(
        execute(&zero_graph, zero_output, f32_data([5, 0, 3], [])),
        f32_data([5, 0], [])
    );

    let mut bool_graph = Graph::new();
    let bools = bool_graph.input_dtype("input", [3, 3], DType::Bool);
    let diagonal = bool_graph.diagonal(bools, 0, 0, 1).unwrap();
    assert_eq!(
        execute(
            &bool_graph,
            diagonal,
            TensorData::from_scalars(
                [3, 3],
                DType::Bool,
                [true, false, false, false, true, false, false, false, true].map(Scalar::Bool),
            )
            .unwrap(),
        ),
        TensorData::from_scalars([3], DType::Bool, [true, true, true].map(Scalar::Bool)).unwrap()
    );
}

#[test]
fn diagonal_rejects_axis_errors_before_graph_mutation() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3, 3]);
    let before = graph.node_count();
    assert!(matches!(
        graph.diagonal(input, 0, 0, 0),
        Err(Error::InvalidDiagonal { .. })
    ));
    assert!(matches!(
        graph.diagonal(input, 0, 2, 1),
        Err(Error::InvalidAxis { .. })
    ));
    assert_eq!(graph.node_count(), before);
}
