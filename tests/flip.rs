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
fn flip_matches_tinygrad_axis_set_and_signed_axis_tables() {
    let cases = [
        (vec![0isize], f32_data([2, 3], [3., 4., 5., 0., 1., 2.])),
        (vec![-1isize], f32_data([2, 3], [2., 1., 0., 5., 4., 3.])),
        (vec![0isize, 1], f32_data([2, 3], [5., 4., 3., 2., 1., 0.])),
    ];
    for (axes, expected) in cases {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let output = graph.flip(input, &axes).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 3]));
        assert_eq!(
            execute(&graph, output, f32_data([2, 3], (0..6).map(|x| x as f32))),
            expected
        );
        assert!(graph.trace(output).unwrap().to_string().contains("stride"));
    }
}

#[test]
fn flip_preserves_bool_payloads_zero_domains_and_empty_axis_identity() {
    let mut bool_graph = Graph::new();
    let input = bool_graph.input_dtype("input", [3], DType::Bool);
    let output = bool_graph.flip(input, &[0]).unwrap();
    assert_eq!(
        execute(
            &bool_graph,
            output,
            TensorData::from_scalars([3], DType::Bool, [true, false, true].map(Scalar::Bool))
                .unwrap(),
        ),
        TensorData::from_scalars([3], DType::Bool, [true, false, true].map(Scalar::Bool)).unwrap()
    );

    let mut zero_graph = Graph::new();
    let zero = zero_graph.input("input", [0, 3]);
    let flipped = zero_graph.flip(zero, &[0]).unwrap();
    assert_eq!(
        execute(&zero_graph, flipped, f32_data([0, 3], [])),
        f32_data([0, 3], [])
    );

    let mut scalar_graph = Graph::new();
    let scalar = scalar_graph.input("input", []);
    let before = scalar_graph.node_count();
    assert_eq!(scalar_graph.flip(scalar, &[]).unwrap(), scalar);
    assert_eq!(scalar_graph.node_count(), before);
}

#[test]
fn flip_rejects_duplicate_and_invalid_axes_before_graph_mutation() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3, 4]);
    let before = graph.node_count();
    assert!(matches!(
        graph.flip(input, &[0, 0]),
        Err(Error::InvalidFlip { .. })
    ));
    assert!(matches!(
        graph.flip(input, &[1, -1]),
        Err(Error::InvalidFlip { .. })
    ));
    assert!(matches!(
        graph.flip(input, &[2]),
        Err(Error::InvalidAxis { .. })
    ));
    assert_eq!(graph.node_count(), before);
}
