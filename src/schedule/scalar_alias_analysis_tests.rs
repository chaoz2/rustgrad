use super::*;
use crate::{Backend, Operation, Scalar, TensorData, UType};

fn affine(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let shared = graph.permute(producer, [1, 0]).unwrap();
    let first = graph.reshape(shared, [1, 3, 2]).unwrap();
    let second = graph.reshape(shared, [3, 2, 1]).unwrap();
    let second = graph.permute(second, [2, 0, 1]).unwrap();
    graph.add(first, second).unwrap()
}

fn projected(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [0, 2, 1, 3]).unwrap();
    let view = graph.reshape(view, [1, 2, 4]).unwrap();
    let lhs = graph.relu(view).unwrap();
    let rhs = graph.square(view).unwrap();
    graph.add(lhs, rhs).unwrap()
}

fn padded(graph: &mut Graph, input: NodeId) -> NodeId {
    let pad = graph.pad(input, [(1, 2)], Scalar::F(0.0)).unwrap();
    let view = graph
        .reshape(pad, [1, graph.shape(pad).unwrap().dims()[0]])
        .unwrap();
    graph.square(view).unwrap()
}

fn reversed(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph
        .stride(
            producer,
            [
                crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    let lhs = graph.relu(view).unwrap();
    let rhs = graph.square(view).unwrap();
    graph.add(lhs, rhs).unwrap()
}

fn reduced(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [1, 0]).unwrap();
    graph.sum(view, 1).unwrap()
}

fn empty(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [1, 0]).unwrap();
    graph.square(view).unwrap()
}

fn compare_schedules(graph: &Graph, requested: &[NodeId], actual: &Schedule, legacy: &Schedule) {
    actual.validate().unwrap();
    legacy.validate().unwrap();
    assert_eq!(actual.items.len(), legacy.items.len());
    assert_eq!(actual.requested_passthroughs, legacy.requested_passthroughs);
    for (actual, legacy) in actual.items.iter().zip(&legacy.items) {
        assert_eq!(actual.node, legacy.node);
        assert_eq!(actual.kernel, legacy.kernel);
        assert_eq!(
            actual.kernel.topological().unwrap(),
            legacy.kernel.topological().unwrap()
        );
        assert_eq!(actual.cache_key, legacy.cache_key);
        assert_eq!(actual.input_bindings, legacy.input_bindings);
        assert_eq!(actual.inputs, legacy.inputs);
        assert_eq!(actual.dependencies, legacy.dependencies);
        assert_eq!(actual.outputs, legacy.outputs);
        assert_eq!(actual.boundary, legacy.boundary);
    }
    let capture = |schedule| crate::CapturedSchedule::capture(graph, schedule, requested).unwrap();
    assert_eq!(
        capture(actual).to_bytes().unwrap(),
        capture(legacy).to_bytes().unwrap()
    );
}

#[test]
fn fixed_point_reuse_matches_previous_discovery_and_cpu() {
    type Builder = fn(&mut Graph, NodeId) -> NodeId;
    let cases: &[(&str, &[usize], Builder)] = &[
        ("affine aliases", &[2, 3], affine),
        ("projected aliases", &[1, 2, 2, 2], projected),
        ("predicated pad", &[2], padded),
        ("empty source pad", &[0], padded),
        ("signed stride", &[2, 3], reversed),
        ("reduction iteration", &[2, 3], reduced),
        ("empty output", &[0, 3], empty),
    ];
    let mut exercised_candidate_reuse = false;
    let mut exercised_projection_reuse = false;
    for &(name, shape, build) in cases {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::new(shape.to_vec()));
        let output = build(&mut graph, input);
        let (cached, counts) = with_mode(true, || schedule(&graph, output));
        let (legacy, previous) = with_mode(false, || schedule(&graph, output));
        let cached = cached.unwrap();
        let legacy = legacy.unwrap();
        assert_eq!(
            counts.fixed_point_passes, previous.fixed_point_passes,
            "{name}"
        );
        assert!(
            counts.candidate_collections <= previous.candidate_collections,
            "{name}"
        );
        assert!(
            counts.projection_builds <= previous.projection_builds,
            "{name}"
        );
        assert_eq!(previous.candidate_hits, 0, "{name}");
        assert_eq!(previous.projection_hits, 0, "{name}");
        if counts.candidate_hits > 0 {
            assert!(counts.fixed_point_passes >= 2, "{name}");
            assert!(
                counts.candidate_collections < previous.candidate_collections,
                "{name}"
            );
            exercised_candidate_reuse = true;
        }
        if counts.projection_hits > 0 {
            assert!(
                counts.projection_builds < previous.projection_builds,
                "{name}"
            );
            exercised_projection_reuse = true;
        }
        compare_schedules(&graph, &[output], &cached, &legacy);
        let data = TensorData::new(
            Shape::new(shape.to_vec()),
            (0..graph.shape(input).unwrap().numel().unwrap())
                .map(|i| (i as f32 - 2.0) * 0.5)
                .collect(),
        )
        .unwrap();
        let oracle = crate::CpuBackend
            .execute(
                &graph,
                output,
                &std::collections::HashMap::from([("input".into(), data.clone())]),
            )
            .unwrap();
        let inputs = BTreeMap::from([("input".into(), data)]);
        for schedule in [&cached, &legacy] {
            let capture = crate::CapturedSchedule::capture(&graph, schedule, &[output]).unwrap();
            let result = capture.replay(&inputs).unwrap();
            assert_eq!(result[0].shape(), oracle.shape(), "{name}");
            assert_eq!(result[0].storage(), oracle.storage(), "{name}");
        }
    }
    assert!(
        exercised_candidate_reuse,
        "no repeated candidate discovery was exercised"
    );
    assert!(
        exercised_projection_reuse,
        "no repeated projection query was exercised"
    );
}

#[test]
fn queries_preserve_guard_classification_full_shapes_and_exact_errors() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let alias = graph.reshape(input, [1, 2, 3]).unwrap();
    let ((), counts) = with_mode(true, || {
        let mut queries = ProjectionQueries::new(&graph);
        let shape = Shape::from([1, 2, 3]);
        // A guarded-query miss is not a failed projection. Its later
        // unconditional lookup must find the source, not reuse that miss.
        assert_eq!(
            queries.predicated_source(alias, &shape),
            Err(RangeifyError::Unsupported(alias))
        );
        assert_eq!(queries.projected_source(alias, &shape), Ok(input));
        assert_eq!(
            queries.projected_source(alias, &Shape::from([2, 3, 1])),
            Err(RangeifyError::Invalid)
        );
        assert_eq!(
            queries.projected_source(alias, &Shape::from([2, 3, 1])),
            Err(RangeifyError::Invalid)
        );
        assert_eq!(queries.results[&alias].len(), 2);
        for bad in [
            NodeId::from_index(graph.node_count()),
            NodeId::from_index(usize::MAX),
        ] {
            assert_eq!(
                queries.projected_source(bad, &shape),
                Err(RangeifyError::Invalid)
            );
            assert_eq!(
                queries.predicated_source(bad, &shape),
                Err(RangeifyError::Invalid)
            );
        }
    });
    assert_eq!(counts.projection_builds, 4);
    assert_eq!(counts.projection_hits, 4);

    for length in [0, 2] {
        let mut graph = Graph::new();
        let input = graph.input("input", [length]);
        let pad = graph.pad(input, [(1, 2)], Scalar::F(0.0)).unwrap();
        let shape = graph.shape(pad).unwrap();
        let mut queries = ProjectionQueries::new(&graph);
        assert_eq!(queries.predicated_source(pad, shape), Ok(input));
        assert_eq!(queries.projected_source(pad, shape), Ok(input));
        assert_eq!(
            queries.classification(pad, shape),
            Ok(ProjectionSource::Predicated(input))
        );
    }
}

#[test]
fn candidate_discovery_is_scoped_to_its_borrowed_frontier() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let producer = graph.square(input).unwrap();
    let alias = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.square(alias).unwrap();
    let roots = BTreeSet::from([producer.index(), alias.index(), output.index()]);
    let none = BTreeSet::new();
    let external = BTreeSet::from([alias.index()]);
    let requested = BTreeSet::from([alias.index(), output.index()]);
    let mut internal = ScalarAliasAnalysis::new(&graph, &roots, &none, &none);
    assert!(
        internal
            .candidates(output)
            .unwrap()
            .unwrap()
            .affine_maps
            .contains_key(&producer.index())
    );
    for (external, requested) in [(&external, &none), (&none, &requested)] {
        let mut observed = ScalarAliasAnalysis::new(&graph, &roots, external, requested);
        let candidates = observed.candidates(output).unwrap().unwrap();
        assert!(candidates.affine_maps.is_empty());
        assert!(candidates.projected_view_roots.is_empty());
        assert_eq!(candidates.direct_roots, BTreeSet::from([alias.index()]));
    }
    assert!(
        internal
            .candidates(output)
            .unwrap()
            .unwrap()
            .affine_maps
            .contains_key(&producer.index())
    );
    let unknown = NodeId::from_index(graph.node_count());
    assert!(
        matches!(internal.candidates(unknown), Err(ScheduleError::Graph(crate::Error::UnknownNode(n))) if n == unknown)
    );
    assert!(!internal.candidates.contains_key(&unknown));
}

#[test]
fn observed_frontiers_match_previous_discovery() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let producer = graph.square(input).unwrap();
    let alias = graph.permute(producer, [1, 0]).unwrap();
    let lhs = graph.square(alias).unwrap();
    let rhs = graph.neg(alias).unwrap();
    let output = graph.add(lhs, rhs).unwrap();
    for (requested, external) in [
        (vec![alias, output], vec![]),
        (vec![output], vec![producer]),
        (vec![output], vec![alias]),
    ] {
        let (cached, _) = with_mode(true, || {
            schedule_with_external_materializations(&graph, &requested, &external)
        });
        let (legacy, _) = with_mode(false, || {
            schedule_with_external_materializations(&graph, &requested, &external)
        });
        compare_schedules(&graph, &requested, &cached.unwrap(), &legacy.unwrap());
    }
}

#[test]
fn source_discovery_does_not_admit_a_concrete_range_or_address() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let alias = graph.permute(input, [1, 0]).unwrap();
    let shape = Shape::from([3, 2]);
    let mut queries = ProjectionQueries::new(&graph);
    assert_eq!(queries.projected_source(alias, &shape), Ok(input));
    let integer = UType::scalar(DType::I64);
    assert_eq!(
        crate::rangeify::projected_view(&graph, alias, &shape, &UOp::constant(6, integer)),
        Err(RangeifyError::Invalid)
    );

    let range = UOp::from_operation(
        Operation::Range(0),
        Some(integer),
        vec![UOp::constant(6, integer)],
    );
    let projected = crate::rangeify::projected_view(&graph, alias, &shape, &range).unwrap();
    let value = UType::scalar(DType::F32);
    let address = UOp::from_operation(
        Operation::DefineGlobal(crate::uop::AddressValue {
            space: crate::AddressSpace::Global,
            name: "source".into(),
            element: value,
        }),
        Some(value),
        vec![],
    );
    let invalid = UOp::from_operation(
        Operation::Index(crate::IndexValue::Buffer {
            buffer: input.index() as u64,
            elements: 6,
            input_shape: Shape::from([2, 3]),
            output_shape: shape.clone(),
            addressing: crate::IndexAddressing::Projected,
        }),
        Some(value),
        vec![
            address,
            UOp::binary(
                crate::uop::Binary::Add,
                projected.expression,
                UOp::constant(6, integer),
            ),
        ],
    );
    assert!(matches!(
        crate::projected_index::ProjectedIndexPlan::from_index(&invalid),
        Err(UOpError::InvalidIndex)
    ));
    assert_eq!(queries.projected_source(alias, &shape), Ok(input));
}

#[test]
fn scheduling_failures_keep_the_previous_error_contract() {
    let graph = Graph::new();
    let unknown = NodeId::from_index(0);
    let (cached, _) = with_mode(true, || schedule(&graph, unknown));
    let (legacy, _) = with_mode(false, || schedule(&graph, unknown));
    let expected = ScheduleError::Graph(crate::Error::UnknownNode(unknown));
    assert_eq!(cached.unwrap_err(), expected);
    assert_eq!(legacy.unwrap_err(), expected);

    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [usize::MAX, 2], DType::F32);
    // An external input alone deliberately produces no scheduled item and
    // does not enter kernel extent validation. Preserve that contract too.
    for reuse in [true, false] {
        let (passthrough, counts) = with_mode(reuse, || schedule(&graph, input));
        assert!(passthrough.unwrap().items.is_empty());
        assert_eq!(counts.candidate_collections, 0);
    }
    // Inject a malformed producer directly: the public graph constructor
    // would reject its overflowing descriptor before scheduling can observe it.
    let output = graph.push(
        Op::Unary {
            op: crate::UnaryOp::Neg,
            input,
        },
        Shape::from([usize::MAX, 2]),
        DType::F32,
    );
    for reuse in [true, false] {
        let (result, counts) = with_mode(reuse, || schedule(&graph, output));
        assert_eq!(
            result.unwrap_err(),
            ScheduleError::UOp(UOpError::InvalidArgument)
        );
        assert!(counts.candidate_collections > 0);
    }
}
