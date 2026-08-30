use crate::{Backend, CpuBackend, DType, Error, Graph, Op, RandomStream, Shape, TensorData};
use std::collections::HashMap;

fn run(graph: &Graph, output: crate::NodeId) -> TensorData {
    CpuBackend.execute(graph, output, &HashMap::new()).unwrap()
}

fn stream(graph: &Graph, output: crate::NodeId) -> RandomStream {
    match graph.nodes[output.index()].op {
        Op::Random { stream, .. } | Op::RandomPermutation { stream } => stream,
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

    let mut graph = Graph::new();
    let ones = graph.ones_with_dtype([], DType::U16).unwrap();
    assert_eq!(run(&graph, ones).dtype(), DType::U16);
    assert_eq!(run(&graph, ones).to_vec_f64(), vec![1.]);
}

#[test]
fn graph_arange_preflights_zero_step_and_keeps_terminal_i64_values() {
    let mut graph = Graph::new();
    let original_nodes = graph.node_count();
    assert!(matches!(
        graph.arange(0, 4, 0),
        Err(Error::InvalidArange { .. })
    ));
    assert_eq!(graph.node_count(), original_nodes);

    let upper = graph.arange(i64::MAX - 1, i64::MAX, 2).unwrap();
    let lower = graph.arange(i64::MIN + 1, i64::MIN, -2).unwrap();
    assert_eq!(
        run(&graph, upper).to_vec_f64(),
        vec![(i64::MAX - 1) as f64]
    );
    assert_eq!(
        run(&graph, lower).to_vec_f64(),
        vec![(i64::MIN + 1) as f64]
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
fn like_creation_helpers_preserve_metadata_and_const_values() {
    let cases = [
        (crate::Scalar::Bool(true), None, DType::BF16, vec![1., 1.]),
        (
            crate::Scalar::I(-3),
            Some(DType::I16),
            DType::I16,
            vec![-3., -3.],
        ),
    ];
    for (value, dtype, expected_dtype, expected_values) in cases {
        let mut graph = Graph::new();
        let source = graph.input_dtype("source", [2], DType::BF16);
        let output = graph.const_like(source, value, dtype).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), expected_dtype);
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([(
                        "source".into(),
                        TensorData::from_scalars(
                            [2],
                            DType::BF16,
                            [crate::Scalar::F(0.0), crate::Scalar::F(0.0)],
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap()
                .to_vec_f64(),
            expected_values,
        );
    }

    let mut graph = Graph::new();
    let source = graph.input_dtype("source", [0, 2], DType::BF16);
    let uniform = graph.rand_like_implicit(source, Some(DType::F32)).unwrap();
    let normal = graph.randn_like_implicit(source, None).unwrap();
    assert_eq!(graph.shape(uniform).unwrap(), &Shape::new([0, 2]));
    assert_eq!(graph.dtype(uniform).unwrap(), DType::F32);
    assert_eq!(graph.shape(normal).unwrap(), &Shape::new([0, 2]));
    assert_eq!(graph.dtype(normal).unwrap(), DType::BF16);
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
fn initializer_fan_overflow_rejects_before_random_node_construction() {
    let mut graph = Graph::new();
    let original_nodes = graph.node_count();

    assert!(matches!(
        graph.glorot_uniform([usize::MAX, 1], DType::F32, 9),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.node_count(), original_nodes);
    assert!(matches!(
        graph.kaiming_uniform([1, usize::MAX, 2], 0.01, DType::F32, 9),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.node_count(), original_nodes);
    assert!(matches!(
        graph.kaiming_normal([1, usize::MAX, 2], 0.01, DType::F32, 9),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.node_count(), original_nodes);

    let glorot = graph.glorot_uniform([2, 3], DType::F32, 9).unwrap();
    let uniform = graph
        .uniform(
            [2, 3],
            -(6.0_f64 / 5.0).sqrt(),
            (6.0_f64 / 5.0).sqrt(),
            DType::F32,
            9,
        )
        .unwrap();
    assert_eq!(run(&graph, glorot), run(&graph, uniform));
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
fn randperm_uses_captured_threefry_reservations_and_random_ordering() {
    Graph::manual_seed(1337);
    let mut graph = Graph::new();
    let first = graph.randperm_implicit(20, DType::I32).unwrap();
    let empty = graph.randperm_implicit(0, DType::U64).unwrap();
    let next = graph.randperm_implicit(1, DType::I16).unwrap();
    let other_device = graph.randperm_implicit_on_device(1, DType::I32, 1).unwrap();
    assert_eq!(stream(&graph, first).counter, [0, 0]);
    assert_eq!(stream(&graph, empty).counter, [20, 0]);
    assert_eq!(stream(&graph, next).counter, [20, 0]);
    assert_eq!(stream(&graph, other_device).counter, [0, 0]);
    assert_ne!(stream(&graph, first).key, stream(&graph, other_device).key);
    // Checked-in tinygrad's Tensor.rand(20).argsort() fixture after manual_seed(1337).
    assert_eq!(
        run(&graph, first).to_vec_f64(),
        vec![
            11., 2., 16., 19., 17., 14., 10., 8., 0., 15., 6., 13., 1., 4., 5., 3., 12., 18., 9.,
            7.,
        ]
    );
    assert_eq!(run(&graph, empty).to_vec_f64(), Vec::<f64>::new());
    assert_eq!(graph.dtype(next).unwrap(), DType::I16);
    assert!(
        graph
            .trace(first)
            .unwrap()
            .to_string()
            .contains("randperm(device=0")
    );

    Graph::manual_seed(1337);
    let mut replay = Graph::new();
    let replayed = replay.randperm_implicit(20, DType::I32).unwrap();
    assert_eq!(stream(&graph, first), stream(&replay, replayed));
    assert_eq!(run(&graph, first), run(&replay, replayed));

    Graph::manual_seed(5);
    let mut validation = Graph::new();
    assert!(validation.randperm_implicit(2, DType::F32).is_err());
    let valid = validation.randperm_implicit(1, DType::I32).unwrap();
    assert_eq!(stream(&validation, valid).counter, [0, 0]);
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

#[test]
fn meshgrid_matches_tinygrad_flattened_input_xy_dtype_and_vjp_contracts() {
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [2, 2]);
    let rhs = graph.input_dtype("rhs", [3], DType::I8);
    let grids = graph.meshgrid(vec![lhs, rhs], "xy").unwrap();
    assert_eq!(graph.shape(grids[0]).unwrap(), &Shape::new([3, 4]));
    assert_eq!(graph.shape(grids[1]).unwrap(), &Shape::new([3, 4]));
    assert_eq!(graph.dtype(grids[0]).unwrap(), DType::F32);
    assert_eq!(graph.dtype(grids[1]).unwrap(), DType::I8);
    let loss = graph.sum_all(grids[0]).unwrap();
    let gradient = graph.grad(loss, lhs).unwrap();
    let inputs = HashMap::from([
        ("lhs".into(), TensorData::new([2, 2], vec![0., 1., 2., 3.]).unwrap()),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [3],
                DType::I8,
                [crate::Scalar::I(10), crate::Scalar::I(20), crate::Scalar::I(30)],
            )
            .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, grids[0], &inputs)
            .unwrap()
            .to_vec_f64(),
        vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, grids[1], &inputs)
            .unwrap()
            .to_vec_f64(),
        vec![10., 10., 10., 10., 20., 20., 20., 20., 30., 30., 30., 30.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &inputs)
            .unwrap()
            .to_vec_f64(),
        vec![3.; 4]
    );

    let mut singleton = Graph::new();
    let input = singleton.input_dtype("input", [2, 2], DType::BF16);
    assert_eq!(singleton.meshgrid(vec![input], "ij").unwrap(), vec![input]);

    let mut empty = Graph::new();
    let lhs = empty.input("lhs", [0]);
    let rhs = empty.input("rhs", [2]);
    let grids = empty.meshgrid(vec![lhs, rhs], "ij").unwrap();
    assert_eq!(empty.shape(grids[0]).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.shape(grids[1]).unwrap(), &Shape::new([0, 2]));
}

#[test]
fn meshgrid_preflights_every_descriptor_before_graph_growth() {
    let mut empty = Graph::new();
    let nodes = empty.node_count();
    assert!(empty.meshgrid(Vec::new(), "ij").is_err());
    assert_eq!(empty.node_count(), nodes);

    let mut overflow = Graph::new();
    let large = overflow.input("large", [usize::MAX, 2]);
    let small = overflow.input("small", [1]);
    let nodes = overflow.node_count();
    assert!(overflow.meshgrid(vec![large, small], "ij").is_err());
    assert_eq!(overflow.node_count(), nodes);
}

#[test]
fn meshgrid_default_is_source_ij_identity_and_inherits_atomic_preflight() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2], DType::F16);
    let y = graph.input_dtype("y", [3], DType::I16);
    let grids = graph.meshgrid_default(vec![x, y]).unwrap();
    assert_eq!(graph.shape(grids[0]).unwrap(), &Shape::new([2, 3]));
    assert_eq!(graph.shape(grids[1]).unwrap(), &Shape::new([2, 3]));
    assert_eq!(graph.dtype(grids[0]).unwrap(), DType::F16);
    assert_eq!(graph.dtype(grids[1]).unwrap(), DType::I16);

    let scalar = graph.input_dtype("scalar", [], DType::BF16);
    assert_eq!(graph.meshgrid_default(vec![scalar]).unwrap(), vec![scalar]);

    let mut invalid = Graph::new();
    let before = invalid.node_count();
    assert!(invalid.meshgrid_default(Vec::new()).is_err());
    assert_eq!(invalid.node_count(), before);
}

#[test]
fn one_hot_preflights_unrepresentable_class_counts_before_creating_nodes() {
    if usize::BITS < i64::BITS {
        return;
    }
    let mut graph = Graph::new();
    let indices = graph.input_dtype("indices", [1], DType::I32);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.one_hot(indices, usize::MAX),
        Err(Error::InvalidRandom {
            reason: "one_hot class count exceeds the supported i64 range",
        })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn one_hot_uses_a_scalar_backed_default_integer_range_and_preflights_the_full_graph() {
    let mut graph = Graph::new();
    let indices = graph.input_dtype("indices", [2, 0], DType::I16);
    let output = graph.one_hot(indices, 3).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 0, 3]));
    assert_eq!(graph.dtype(output).unwrap(), DType::I32);
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    assert!(graph.nodes.iter().filter_map(|node| match &node.op {
        Op::Constant(data) => Some(data.len()),
        _ => None,
    }).all(|len| len == 1));
    assert!(graph.nodes.iter().any(|node| matches!(&node.op, Op::Reduce {
        kind: crate::ReduceKind::Sum, ..
    })));

    let scalar = graph.input_dtype("scalar", [], DType::I32);
    let scalar_output = graph.one_hot(scalar, 2).unwrap();
    assert_eq!(graph.shape(scalar_output).unwrap(), &Shape::new([2]));
    let empty = graph.input_dtype("empty", [0], DType::U8);
    let empty_output = graph.one_hot(empty, 0).unwrap();
    assert_eq!(graph.shape(empty_output).unwrap(), &Shape::new([0, 0]));

    let mut invalid = Graph::new();
    let float = invalid.input("float", [1]);
    let before = invalid.node_count();
    assert!(invalid.one_hot(float, 1).is_err());
    assert_eq!(invalid.node_count(), before);
    let large = invalid.input_dtype("large", [usize::MAX / 2 + 1], DType::I8);
    let before = invalid.node_count();
    assert!(matches!(invalid.one_hot(large, 3), Err(Error::ShapeOverflow(_))));
    assert_eq!(invalid.node_count(), before);
}
