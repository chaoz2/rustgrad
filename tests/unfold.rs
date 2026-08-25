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
fn unfold_matches_tinygrad_window_tables_and_trace() {
    let cases = [
        (
            vec![8usize],
            0,
            2,
            1,
            Shape::from([7, 2]),
            vec![0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7.],
        ),
        (
            vec![8usize],
            0,
            2,
            2,
            Shape::from([4, 2]),
            vec![0., 1., 2., 3., 4., 5., 6., 7.],
        ),
        (
            vec![8usize],
            0,
            7,
            3,
            Shape::from([1, 7]),
            vec![0., 1., 2., 3., 4., 5., 6.],
        ),
        (
            vec![3usize, 3, 3],
            2,
            2,
            8,
            Shape::from([3, 3, 1, 2]),
            vec![
                0., 1., 3., 4., 6., 7., 9., 10., 12., 13., 15., 16., 18., 19., 21., 22., 24., 25.,
            ],
        ),
    ];
    for (shape, dim, size, step, output_shape, expected) in cases {
        let mut graph = Graph::new();
        let input = graph.input("input", shape.clone());
        let output = graph.unfold(input, dim, size, step).unwrap();
        let input_len = shape.iter().product::<usize>();
        assert_eq!(graph.shape(output).unwrap(), &output_shape);
        assert_eq!(
            execute(
                &graph,
                output,
                f32_data(shape, (0..input_len).map(|x| x as f32)),
            ),
            f32_data(output_shape, expected)
        );
        assert!(
            graph
                .trace(output)
                .unwrap()
                .to_string()
                .contains("static_index")
        );
    }
}

#[test]
fn unfold_keeps_axis_placement_zero_windows_and_bool_payloads() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let output = graph.unfold(input, -1, 2, 1).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 2, 2]));
    assert_eq!(
        execute(&graph, output, f32_data([2, 3], [0., 1., 2., 3., 4., 5.])),
        f32_data([2, 2, 2], [0., 1., 1., 2., 3., 4., 4., 5.])
    );

    let mut empty_graph = Graph::new();
    let empty = empty_graph.input("input", [0]);
    let empty_window = empty_graph.unfold(empty, 0, 0, 1).unwrap();
    assert_eq!(
        empty_graph.shape(empty_window).unwrap(),
        &Shape::from([1, 0])
    );
    assert_eq!(
        execute(&empty_graph, empty_window, f32_data([0], [])),
        f32_data([1, 0], [])
    );

    let nonempty_zero_window = graph.unfold(input, 1, 0, 8).unwrap();
    assert_eq!(
        graph.shape(nonempty_zero_window).unwrap(),
        &Shape::from([2, 1, 0, 3])
    );
    assert_eq!(
        execute(
            &graph,
            nonempty_zero_window,
            f32_data([2, 3], [0., 1., 2., 3., 4., 5.])
        ),
        f32_data([2, 1, 0, 3], [])
    );

    let mut bool_graph = Graph::new();
    let bools = bool_graph.input_dtype("input", [4], DType::Bool);
    let bool_windows = bool_graph.unfold(bools, 0, 3, 2).unwrap();
    assert_eq!(
        execute(
            &bool_graph,
            bool_windows,
            TensorData::from_scalars(
                [4],
                DType::Bool,
                [true, false, true, false].map(Scalar::Bool),
            )
            .unwrap(),
        ),
        TensorData::from_scalars([1, 3], DType::Bool, [true, false, true].map(Scalar::Bool),)
            .unwrap()
    );
}

#[test]
fn unfold_rejects_bad_static_contracts_without_graph_mutation() {
    let mut graph = Graph::new();
    let input = graph.input("input", [8]);
    let before = graph.node_count();
    for (dim, size, step) in [(0, -1, 1), (0, 1, 0), (0, 9, 1)] {
        assert!(matches!(
            graph.unfold(input, dim, size, step),
            Err(Error::InvalidUnfold { .. })
        ));
    }
    assert!(matches!(
        graph.unfold(input, 1, 1, 1),
        Err(Error::InvalidAxis { .. })
    ));
    assert_eq!(graph.node_count(), before);
}
