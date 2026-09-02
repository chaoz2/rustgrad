use rustgrad::{
    Backend, CpuBackend, DType, Graph, Scalar, Storage, TensorData, ir::indexing::StaticIndex,
};
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
            .iter()
            .any(|step| step.operation.starts_with("gather(") && step.operation.contains("axis=0"))
    );
    assert!(
        graph
            .trace(output)
            .unwrap()
            .steps
            .iter()
            .all(|step| !step.operation.starts_with("static_index"))
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
    let before = graph.node_count();
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
    assert_eq!(graph.node_count(), before);
}

#[test]
fn compositional_static_index_preserves_scalar_empty_and_higher_order_routes() {
    let mut scalar = Graph::new();
    let input = scalar.input("input", []);
    let output = scalar.static_index(input, &[]).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &rustgrad::Shape::new([]));
    assert_eq!(
        CpuBackend
            .execute(
                &scalar,
                output,
                &HashMap::from([("input".into(), f32_data([], [7.0]))]),
            )
            .unwrap(),
        f32_data([], [7.0])
    );

    let mut empty = Graph::new();
    let input = empty.input("input", [0, 3]);
    let output = empty
        .static_index(
            input,
            &[
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                StaticIndex::Integer(1),
            ],
        )
        .unwrap();
    assert_eq!(empty.shape(output).unwrap(), &rustgrad::Shape::from([0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), f32_data([0, 3], []))]),
            )
            .unwrap(),
        f32_data([0], [])
    );

    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let selected = graph
        .static_index(
            input,
            &[StaticIndex::Advanced {
                shape: [2].into(),
                values: vec![1, 1],
            }],
        )
        .unwrap();
    let seed = graph.input("seed", [2]);
    let first = graph.grad_with(selected, input, Some(seed), true).unwrap();
    let first_sum = graph.sum(first, 0).unwrap();
    let second = graph.grad(first_sum, seed).unwrap();
    let bindings = HashMap::from([
        ("input".into(), f32_data([3], [0.0, 0.0, 0.0])),
        ("seed".into(), f32_data([2], [2.0, 3.0])),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, first, &bindings).unwrap(),
        f32_data([3], [0.0, 5.0, 0.0])
    );
    assert_eq!(
        CpuBackend.execute(&graph, second, &bindings).unwrap(),
        f32_data([2], [1.0, 1.0])
    );
}

#[test]
fn functional_static_update_is_snapshot_based_and_last_write_wins() {
    let mut graph = Graph::new();
    let base = graph.input("base", [3]);
    let value = graph.input("value", []);
    let output = graph
        .static_index_update(
            base,
            &[StaticIndex::Advanced {
                shape: [2].into(),
                values: vec![1, 1],
            }],
            value,
        )
        .unwrap();
    assert_eq!(
        CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([
                    ("base".into(), f32_data([3], [1., 2., 3.])),
                    ("value".into(), f32_data([], [9.])),
                ])
            )
            .unwrap(),
        f32_data([3], [1., 9., 3.])
    );
}

#[test]
fn functional_static_update_vjp_credits_only_final_duplicate_writer() {
    let mut graph = Graph::new();
    let base = graph.input("base", [3]);
    let value = graph.input("value", [2]);
    let output = graph
        .static_index_update(
            base,
            &[StaticIndex::Advanced {
                shape: [2].into(),
                values: vec![1, 1],
            }],
            value,
        )
        .unwrap();
    let loss = graph.sum(output, 0).unwrap();
    let base_grad = graph.grad(loss, base).unwrap();
    let value_grad = graph.grad(loss, value).unwrap();
    let bindings = HashMap::from([
        ("base".into(), f32_data([3], [1., 2., 3.])),
        ("value".into(), f32_data([2], [7., 9.])),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, base_grad, &bindings).unwrap(),
        f32_data([3], [1., 0., 1.])
    );
    assert_eq!(
        CpuBackend.execute(&graph, value_grad, &bindings).unwrap(),
        f32_data([2], [0., 1.])
    );
    assert!(
        graph
            .trace(value_grad)
            .unwrap()
            .steps
            .iter()
            .any(|step| step.operation.contains("static_index_update_grad_Value"))
    );
    assert!(graph.grad(value_grad, value).is_err());
}

#[test]
fn functional_static_update_vjp_accumulates_rhs_broadcast_offsets() {
    let mut graph = Graph::new();
    let base = graph.input("base", [2, 3]);
    let value = graph.input("value", [2, 1]);
    let output = graph
        .static_index_update(
            base,
            &[
                StaticIndex::Ellipsis,
                StaticIndex::Advanced {
                    shape: [2].into(),
                    values: vec![0, 2],
                },
            ],
            value,
        )
        .unwrap();
    let row_sum = graph.sum(output, 0).unwrap();
    let loss = graph.sum(row_sum, 0).unwrap();
    let base_grad = graph.grad(loss, base).unwrap();
    let value_grad = graph.grad(loss, value).unwrap();
    let bindings = HashMap::from([
        ("base".into(), f32_data([2, 3], [0., 1., 2., 3., 4., 5.])),
        ("value".into(), f32_data([2, 1], [10., 20.])),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, base_grad, &bindings).unwrap(),
        f32_data([2, 3], [0., 1., 0., 0., 1., 0.])
    );
    assert_eq!(
        CpuBackend.execute(&graph, value_grad, &bindings).unwrap(),
        f32_data([2, 1], [2., 2.])
    );
}

#[test]
fn functional_static_update_preserves_exact_raw_storage_classes() {
    let cases = [
        (
            DType::Bool,
            Storage::Bool(vec![true, false]),
            Storage::Bool(vec![false]),
        ),
        (
            DType::U64,
            Storage::U64(vec![1, 0xfeed_face_dead_beef]),
            Storage::U64(vec![u64::MAX]),
        ),
        (
            DType::F16,
            Storage::F16(vec![0x3c00, 0x8001]),
            Storage::F16(vec![0x7e01]),
        ),
        (
            DType::BF16,
            Storage::BF16(vec![0x3f80, 0x8001]),
            Storage::BF16(vec![0x7fc1]),
        ),
        (
            DType::F32,
            Storage::F32(vec![1.0, f32::from_bits(0x8000_0000)]),
            Storage::F32(vec![f32::from_bits(0x7fc0_1234)]),
        ),
        (
            DType::F64,
            Storage::F64(vec![1.0, f64::from_bits(0x8000_0000_0000_0000)]),
            Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_1234)]),
        ),
    ];
    for (dtype, base_value, update_value) in cases {
        let mut graph = Graph::new();
        let base = graph.input_dtype("base", [2], dtype);
        let value = graph.input_dtype("value", [], dtype);
        let output = graph
            .static_index_update(base, &[StaticIndex::Integer(1)], value)
            .unwrap();
        let actual = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([
                    (
                        "base".into(),
                        TensorData::from_storage([2], base_value).unwrap(),
                    ),
                    (
                        "value".into(),
                        TensorData::from_storage([], update_value).unwrap(),
                    ),
                ]),
            )
            .unwrap();
        match actual.storage() {
            Storage::Bool(values) => assert_eq!(values, &[true, false]),
            Storage::U64(values) => assert_eq!(values, &[1, u64::MAX]),
            Storage::F16(values) => assert_eq!(values, &[0x3c00, 0x7e01]),
            Storage::BF16(values) => assert_eq!(values, &[0x3f80, 0x7fc1]),
            Storage::F32(values) => assert_eq!(
                values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                vec![1.0f32.to_bits(), 0x7fc0_1234]
            ),
            Storage::F64(values) => assert_eq!(
                values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                vec![1.0f64.to_bits(), 0x7ff8_0000_0000_1234]
            ),
            other => panic!("unexpected storage {other:?}"),
        }
    }
}

#[test]
fn functional_static_update_rejects_mismatched_rhs_contracts() {
    let mut graph = Graph::new();
    let base = graph.input_dtype("base", [2, 2], DType::F32);
    let wrong_dtype = graph.input_dtype("wrong_dtype", [], DType::I32);
    let wrong_shape = graph.input_dtype("wrong_shape", [3], DType::F32);
    assert!(
        graph
            .static_index_update(base, &[StaticIndex::Integer(0)], wrong_dtype)
            .is_err()
    );
    assert!(
        graph
            .static_index_update(base, &[StaticIndex::Integer(0)], wrong_shape)
            .is_err()
    );
    assert!(
        graph
            .static_index_update(base, &[StaticIndex::Integer(2)], base)
            .is_err()
    );
}
