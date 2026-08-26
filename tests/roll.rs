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
fn roll_matches_tinygrad_signed_axis_and_flattened_tables() {
    let cases = [
        (
            Some(vec![0isize]),
            vec![1isize],
            f32_data([2, 4], [4., 5., 6., 7., 0., 1., 2., 3.]),
        ),
        (
            Some(vec![-1isize]),
            vec![-1isize],
            f32_data([2, 4], [1., 2., 3., 0., 5., 6., 7., 4.]),
        ),
        (
            None,
            vec![1isize],
            f32_data([2, 4], [7., 0., 1., 2., 3., 4., 5., 6.]),
        ),
    ];
    for (dims, shifts, expected) in cases {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 4]);
        let output = graph.roll(input, &shifts, dims.as_deref()).unwrap();
        assert_eq!(execute(&graph, output, f32_data([2, 4], (0..8).map(|x| x as f32))), expected);
        assert!(graph.trace(output).unwrap().to_string().contains("shrink"));
    }
}

#[test]
fn roll_preserves_bool_payloads_and_zero_domains() {
    let mut bool_graph = Graph::new();
    let input = bool_graph.input_dtype("input", [4], DType::Bool);
    let output = bool_graph.roll(input, &[1], Some(&[0])).unwrap();
    assert_eq!(
        execute(
            &bool_graph,
            output,
            TensorData::from_scalars([4], DType::Bool, [true, false, false, true].map(Scalar::Bool)).unwrap(),
        ),
        TensorData::from_scalars([4], DType::Bool, [true, true, false, false].map(Scalar::Bool)).unwrap()
    );

    let mut empty_graph = Graph::new();
    let empty = empty_graph.input("input", [2, 0, 3]);
    let before = empty_graph.node_count();
    let output = empty_graph.roll(empty, &[1], Some(&[1])).unwrap();
    assert_eq!(output, empty);
    assert_eq!(empty_graph.node_count(), before);
    assert_eq!(execute(&empty_graph, output, f32_data([2, 0, 3], [])), f32_data([2, 0, 3], []));

    let mut zero_shift_graph = Graph::new();
    let input = zero_shift_graph.input("input", [2, 3]);
    let before = zero_shift_graph.node_count();
    let output = zero_shift_graph.roll(input, &[0], Some(&[1])).unwrap();
    assert_eq!(output, input);
    assert_eq!(zero_shift_graph.node_count(), before);
}

#[test]
fn roll_composes_multiple_static_axes_in_tinygrad_order() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3, 4]);
    let output = graph.roll(input, &[2, -3], Some(&[0, 2])).unwrap();
    assert_eq!(
        execute(
            &graph,
            output,
            f32_data([2, 3, 4], (0..24).map(|x| x as f32)),
        ),
        f32_data(
            [2, 3, 4],
            [
                3., 0., 1., 2., 7., 4., 5., 6., 11., 8., 9., 10., 15., 12., 13., 14., 19.,
                16., 17., 18., 23., 20., 21., 22.,
            ],
        )
    );
}

#[test]
fn roll_rejects_malformed_contracts_before_graph_mutation() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 4]);
    let before = graph.node_count();
    assert!(matches!(
        graph.roll(input, &[1, 2], Some(&[0])),
        Err(Error::InvalidRoll { .. })
    ));
    assert!(matches!(
        graph.roll(input, &[1, 2], None),
        Err(Error::InvalidRoll { .. })
    ));
    assert!(matches!(
        graph.roll(input, &[1], Some(&[2])),
        Err(Error::InvalidAxis { .. })
    ));
    assert_eq!(graph.node_count(), before);
}
