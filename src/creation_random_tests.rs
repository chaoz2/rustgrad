use crate::{Backend, CpuBackend, DType, Error, Graph, Op, RandomStream, Shape, TensorData};
use std::collections::HashMap;

fn run(graph: &Graph, output: crate::NodeId) -> TensorData {
    CpuBackend.execute(graph, output, &HashMap::new()).unwrap()
}

fn stream(graph: &Graph, output: crate::NodeId) -> RandomStream {
    match graph.nodes[output.index()].op {
        Op::Random { stream, .. } => stream,
        ref op => panic!("expected random node, got {op:?}"),
    }
}

#[test]
fn creation_helpers_cover_scalar_empty_ranges_and_dtypes() {
    assert_eq!(
        TensorData::empty([], DType::I32).unwrap().to_vec_f64(),
        vec![0.]
    );
    assert_eq!(
        TensorData::linspace(-1., 1., 3, DType::F64)
            .unwrap()
            .to_vec_f64(),
        vec![-1., 0., 1.]
    );
    assert_eq!(
        TensorData::linspace(1., 2., 0, DType::F32).unwrap().shape(),
        &Shape::new([0])
    );
    assert_eq!(
        TensorData::eye(2, Some(3), DType::Bool)
            .unwrap()
            .to_vec_f64(),
        vec![1., 0., 0., 0., 1., 0.]
    );
    assert_eq!(
        TensorData::linspace(0., 1., -1, DType::F32),
        Err(Error::InvalidLinspace { steps: -1 })
    );
}

#[test]
fn seeded_random_nodes_replay_exactly_and_are_typed() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut graph = Graph::new();
        let first = graph.uniform([2, 3], -2., 4., dtype, 7).unwrap();
        let second = graph.uniform([2, 3], -2., 4., dtype, 7).unwrap();
        let distinct = graph.uniform([2, 3], -2., 4., dtype, 8).unwrap();
        let a = run(&graph, first);
        assert_eq!(a, run(&graph, second));
        assert_ne!(a, run(&graph, distinct));
        assert_eq!(a.dtype(), dtype);
        assert!(
            a.to_vec_f64()
                .iter()
                .all(|value| (-2. ..4.).contains(value))
        );
    }
    let mut graph = Graph::new();
    let empty = graph.rand([0, 2], DType::F32, 1).unwrap();
    assert_eq!(run(&graph, empty).shape(), &Shape::new([0, 2]));
    assert!(
        graph
            .trace(empty)
            .unwrap()
            .to_string()
            .contains("random_Uniform")
    );
}

#[test]
fn normal_and_randint_have_deterministic_sane_static_contracts() {
    let mut graph = Graph::new();
    let normal = graph.normal([128], 2., 0.5, DType::F32, 42).unwrap();
    let integers = graph.randint([128], -3, 5, DType::I16, 42).unwrap();
    let normal_values = run(&graph, normal).to_vec_f64();
    let mean = normal_values.iter().sum::<f64>() / normal_values.len() as f64;
    assert!((mean - 2.).abs() < 0.2);
    assert!(
        run(&graph, integers)
            .to_vec_f64()
            .iter()
            .all(|value| (-3. ..5.).contains(value))
    );
    assert!(graph.randint([1], 0, 1, DType::F32, 0).is_err());
    assert!(graph.uniform([1], 1., 1., DType::F32, 0).is_err());
    assert!(graph.normal([1], 0., -1., DType::F32, 0).is_err());
}

#[test]
fn like_global_seed_randperm_and_initializers_are_replayable() {
    let mut graph = Graph::new();
    let source = graph.input_dtype("x", [2, 3], DType::F16);
    let ones = graph.ones_like(source, None).unwrap();
    let zeros = graph.zeros_like(source, None).unwrap();
    let full = graph
        .full_like(source, crate::Scalar::I(2), Some(DType::I32))
        .unwrap();
    let random = graph.randn_like(source, None, 1).unwrap();
    assert_eq!(graph.shape(ones).unwrap(), &Shape::new([2, 3]));
    assert_eq!(graph.dtype(zeros).unwrap(), DType::F16);
    assert_eq!(graph.dtype(full).unwrap(), DType::I32);
    assert_eq!(graph.shape(random).unwrap(), &Shape::new([2, 3]));

    Graph::manual_seed(19);
    let mut first = Graph::new();
    let a = first.rand_implicit([4], DType::F32).unwrap();
    let b = first.rand_implicit([4], DType::F32).unwrap();
    assert_ne!(run(&first, a), run(&first, b));
    Graph::manual_seed(19);
    let mut replay = Graph::new();
    let again = replay.rand_implicit([4], DType::F32).unwrap();
    assert_eq!(run(&first, a), run(&replay, again));

    let permutation = first.randperm(8, DType::I32, 7).unwrap();
    let mut values = run(&first, permutation).to_vec_f64();
    values.sort_by(f64::total_cmp);
    assert_eq!(values, (0..8).map(f64::from).collect::<Vec<_>>());
    assert!(first.randperm(2, DType::F32, 0).is_err());
    let initialized = first.kaiming_uniform([3, 4], 0.01, DType::F32, 4).unwrap();
    assert!(
        run(&first, initialized)
            .to_vec_f64()
            .iter()
            .all(|value| value.abs() <= 1.23)
    );
}

#[test]
fn implicit_streams_reserve_packed_words_reset_and_isolate_devices() {
    Graph::manual_seed(1337);
    let mut graph = Graph::new();
    let first = graph.rand_implicit([1], DType::F16).unwrap();
    let second = graph.rand_implicit([1], DType::F16).unwrap();
    let empty = graph.rand_implicit([0], DType::F16).unwrap();
    let third = graph.rand_implicit([1], DType::F64).unwrap();
    let other = graph.rand_implicit_on_device([1], DType::F16, 1).unwrap();
    assert_eq!(stream(&graph, first).counter, [0, 0]);
    assert_eq!(stream(&graph, second).counter, [1, 0]);
    assert_eq!(stream(&graph, empty).counter, [2, 0]);
    assert_eq!(stream(&graph, third).counter, [2, 0]);
    assert_eq!(stream(&graph, other).counter, [0, 0]);
    assert_ne!(stream(&graph, first).key, stream(&graph, other).key);
    assert_ne!(run(&graph, first), run(&graph, second));

    Graph::manual_seed(1337);
    let mut replay = Graph::new();
    let replayed = replay.rand_implicit([1], DType::F16).unwrap();
    assert_eq!(stream(&graph, first), stream(&replay, replayed));
    assert_eq!(run(&graph, first), run(&replay, replayed));
}

#[test]
fn invalid_implicit_randperm_does_not_reserve_or_append() {
    Graph::manual_seed(411);
    let mut graph = Graph::new();
    let original_nodes = graph.node_count();
    assert!(matches!(
        graph.randperm_implicit(8, DType::F32),
        Err(Error::InvalidRandom { .. })
    ));
    assert_eq!(graph.node_count(), original_nodes);

    let first = graph.rand_implicit([1], DType::F32).unwrap();
    assert_eq!(stream(&graph, first).counter, [0, 0]);
    Graph::manual_seed(411);
    let mut replay = Graph::new();
    let expected = replay.rand_implicit([1], DType::F32).unwrap();
    assert_eq!(stream(&graph, first), stream(&replay, expected));
    assert_eq!(run(&graph, first), run(&replay, expected));
}

#[test]
fn randint_uses_float_uniform_scaling_then_storage_cast() {
    let mut graph = Graph::new();
    let uniform = graph.uniform([16], -3.0, 5.0, DType::F32, 23).unwrap();
    let integers = graph.randint([16], -3, 5, DType::I32, 23).unwrap();
    let expected: Vec<_> = run(&graph, uniform)
        .to_vec_f64()
        .into_iter()
        .map(|value| value as i64 as f64)
        .collect();
    assert_eq!(run(&graph, integers).to_vec_f64(), expected);
}

#[test]
fn rank_stack_one_hot_and_meshgrid_compose_through_existing_ops() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let y = graph.input("y", [2]);
    let stacked = graph.stack(vec![x, y], -1).unwrap();
    assert_eq!(graph.shape(stacked).unwrap(), &Shape::new([2, 2]));
    let loss = graph
        .reduce(stacked, crate::ReduceKind::Sum, None, false)
        .unwrap();
    let dx = graph.grad(loss, x).unwrap();
    let inputs = HashMap::from([
        ("x".into(), TensorData::new([2], vec![1., 2.]).unwrap()),
        ("y".into(), TensorData::new([2], vec![3., 4.]).unwrap()),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, dx, &inputs)
            .unwrap()
            .to_vec_f64(),
        vec![1., 1.]
    );
    let indices = graph.input_dtype("i", [2], DType::I32);
    let hot = graph.one_hot(indices, 3).unwrap();
    assert_eq!(
        CpuBackend
            .execute(
                &graph,
                hot,
                &HashMap::from([
                    ("x".into(), TensorData::new([2], vec![1., 2.]).unwrap()),
                    ("y".into(), TensorData::new([2], vec![3., 4.]).unwrap()),
                    (
                        "i".into(),
                        TensorData::from_scalars(
                            [2],
                            DType::I32,
                            [crate::Scalar::I(0), crate::Scalar::I(3)]
                        )
                        .unwrap()
                    ),
                ])
            )
            .unwrap()
            .to_vec_f64(),
        vec![1., 0., 0., 0., 0., 0.]
    );
    let gx = graph.input("gx", [2]);
    let gy = graph.input("gy", [3]);
    let grids = graph.meshgrid(vec![gx, gy], "xy").unwrap();
    assert_eq!(graph.shape(grids[0]).unwrap(), &Shape::new([3, 2]));
    assert!(graph.trace(stacked).unwrap().to_string().contains("concat"));
}
