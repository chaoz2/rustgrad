use rustgrad::{Backend, CpuBackend, DType, Graph, Scalar, TensorData, ir::indexing::StaticIndex};
use std::collections::HashMap;

fn f32_data(
    shape: impl Into<rustgrad::Shape>,
    values: impl IntoIterator<Item = f32>,
) -> TensorData {
    TensorData::new(shape, values.into_iter().collect()).unwrap()
}

#[test]
fn graph_executes_mixed_static_advanced_indexing_and_traces_it() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3, 4]);
    let output = graph
        .static_index(
            input,
            &[
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                StaticIndex::Advanced {
                    shape: [2].into(),
                    values: vec![1, 0],
                },
                StaticIndex::Advanced {
                    shape: [2].into(),
                    values: vec![-1, 1],
                },
            ],
        )
        .unwrap();
    let values = f32_data([2, 3, 4], (0..24).map(|value| value as f32));
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &HashMap::from([("input".into(), values)]))
            .unwrap(),
        f32_data([2, 2], [7., 1., 19., 13.])
    );
    assert!(
        graph
            .trace(output)
            .unwrap()
            .steps
            .last()
            .unwrap()
            .operation
            .starts_with("static_index")
    );
}

#[test]
fn static_index_keeps_exact_bool_storage_and_duplicate_gradients_accumulate() {
    let mut graph = Graph::new();
    let bools = graph.input_dtype("bools", [2, 3], DType::Bool);
    let selected = graph
        .static_index(bools, &[StaticIndex::Ellipsis, StaticIndex::Integer(-1)])
        .unwrap();
    let bool_data = TensorData::from_scalars(
        [2, 3],
        DType::Bool,
        [false, true, false, true, false, true].map(Scalar::Bool),
    )
    .unwrap();
    assert_eq!(
        CpuBackend
            .execute(
                &graph,
                selected,
                &HashMap::from([("bools".into(), bool_data)])
            )
            .unwrap(),
        TensorData::from_scalars([2], DType::Bool, [false, true].map(Scalar::Bool)).unwrap()
    );

    let mut graph = Graph::new();
    let x = graph.input("x", [2, 3]);
    let gathered = graph
        .static_index(
            x,
            &[
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                StaticIndex::Advanced {
                    shape: [2].into(),
                    values: vec![1, 1],
                },
            ],
        )
        .unwrap();
    let row_sums = graph.sum(gathered, 1).unwrap();
    let loss = graph.sum(row_sums, 0).unwrap();
    let gradient = graph.grad(loss, x).unwrap();
    assert_eq!(
        CpuBackend
            .execute(
                &graph,
                gradient,
                &HashMap::from([("x".into(), f32_data([2, 3], [0., 0., 0., 0., 0., 0.]))])
            )
            .unwrap(),
        f32_data([2, 3], [0., 2., 0., 0., 2., 0.])
    );
}

#[test]
fn static_index_rejects_invalid_static_specs() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 3]);
    assert!(graph.static_index(x, &[StaticIndex::Integer(2)]).is_err());
    assert!(
        graph
            .static_index(
                x,
                &[StaticIndex::Advanced {
                    shape: [2].into(),
                    values: vec![0, 3]
                }]
            )
            .is_err()
    );
    assert!(
        graph
            .static_index(
                x,
                &[StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 0
                }]
            )
            .is_err()
    );
}
