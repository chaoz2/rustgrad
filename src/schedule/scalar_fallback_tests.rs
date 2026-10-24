use super::*;

// Released normalization followed by an independent second traversal.
fn released_kernel_loads(kernel: &UOp) -> Result<BTreeSet<usize>, ScheduleError> {
    let normalized = crate::uop::normalize_kernel(kernel).map_err(ScheduleError::UOp)?;
    let nodes = normalized.topological().map_err(ScheduleError::UOp)?;
    let mut loads = BTreeSet::new();
    for node in nodes {
        if !matches!(node.operation(), crate::Operation::Load) {
            continue;
        }
        let index = node
            .sources()
            .first()
            .ok_or_else(|| ScheduleError::Binding("ordinary scalar Load index is absent".into()))?;
        let buffer = match index.operation() {
            crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
            | crate::Operation::Index(crate::IndexValue::View { buffer, .. }) => *buffer,
            _ => {
                return Err(ScheduleError::Binding(
                    "ordinary scalar Load index is not a buffer descriptor".into(),
                ));
            }
        };
        loads.insert(usize::try_from(buffer).map_err(|_| ScheduleError::Overflow)?);
    }
    Ok(loads)
}

#[test]
fn fallback_loads_reuse_the_final_normalization_order() {
    for shape in [vec![], vec![0, 3], vec![2, 3]] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", shape, crate::DType::F32);
        let producer = graph.square(input).unwrap();
        let output = graph.neg(producer).unwrap();
        let reduction = graph
            .reduce(output, crate::ReduceKind::Sum, None, false)
            .unwrap();
        for (name, node, external, expected) in [
            ("elementwise", output, BTreeSet::new(), input),
            ("reduction", reduction, BTreeSet::new(), input),
            (
                "external producer",
                output,
                BTreeSet::from([producer.index()]),
                producer,
            ),
        ] {
            let roots = BTreeSet::from([node.index()]);
            let materialized = scalar_alias_materialized(node, &roots, &external, &BTreeSet::new());
            let kernel = match graph.op(node).unwrap() {
                Op::Reduce { .. } => crate::kernel::lower_graph_reduction_with_materialized(
                    &graph,
                    node,
                    &materialized,
                ),
                _ => crate::kernel::lower_graph_elementwise_with_materialized(
                    &graph,
                    node,
                    &materialized,
                ),
            }
            .unwrap();
            let before = crate::uop::canonical_traversal_count();
            let actual = scalar_fallback_kernel_loads(&kernel).unwrap();
            let retained_walks = crate::uop::canonical_traversal_count() - before;
            let before = crate::uop::canonical_traversal_count();
            let reference = released_kernel_loads(&kernel).unwrap();
            let released_walks = crate::uop::canonical_traversal_count() - before;
            assert_eq!(actual, BTreeSet::from([expected.index()]), "{name}");
            assert_eq!(actual, reference, "{name}");
            assert_eq!(released_walks, retained_walks + 1, "{name}");
            assert_eq!(
                ordinary_scalar_fallback_loads(
                    &graph,
                    node,
                    &roots,
                    &external,
                    &mut crate::rangeify::ProjectedForestCache::new(&graph),
                )
                .unwrap(),
                actual,
                "graph path: {name}"
            );
        }
    }
}

#[test]
fn fallback_order_preserves_normalization_errors_and_empty_load_sets() {
    let ty = crate::UType::scalar(crate::DType::I64);
    let malformed = UOp::from_operation(
        crate::Operation::Unary(crate::uop::Unary::Neg),
        Some(ty),
        vec![],
    );
    let range = UOp::from_operation(
        crate::Operation::Range(0),
        Some(ty),
        vec![UOp::constant(2, ty)],
    );
    let condition = UOp::scalar_constant(
        crate::DType::Bool,
        1,
        crate::UType::scalar(crate::DType::Bool),
    );
    let if_ = UOp::from_operation(crate::Operation::If, None, vec![condition]);
    let end_if = UOp::from_operation(crate::Operation::EndIf, None, vec![if_]);
    for (name, kernel) in [
        ("empty", UOp::sink(vec![])),
        ("constant", UOp::sink(vec![UOp::constant(7, ty)])),
        ("malformed", UOp::sink(vec![malformed])),
        ("open control", UOp::sink(vec![range])),
        ("closed conditional", UOp::sink(vec![end_if])),
        ("not a kernel", UOp::constant(7, ty)),
    ] {
        let before = crate::uop::canonical_traversal_count();
        let actual = scalar_fallback_kernel_loads(&kernel);
        let retained_walks = crate::uop::canonical_traversal_count() - before;
        let before = crate::uop::canonical_traversal_count();
        let reference = released_kernel_loads(&kernel);
        let released_walks = crate::uop::canonical_traversal_count() - before;
        assert_eq!(actual, reference, "{name}");
        assert_eq!(
            released_walks,
            retained_walks + usize::from(actual.is_ok()),
            "{name}"
        );
        if name == "empty" || name == "constant" {
            assert_eq!(actual.unwrap(), BTreeSet::new(), "{name}");
        } else {
            assert!(actual.is_err(), "{name}");
        }
    }
}

fn captured_bytes(graph: &Graph, schedule: &Schedule, requested: &[NodeId]) -> Vec<u8> {
    crate::CapturedSchedule::capture(graph, schedule, requested)
        .unwrap()
        .to_bytes()
        .unwrap()
}

#[test]
fn reuse_removes_duplicate_lowering_without_schedule_or_capture_drift() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], crate::DType::F32);
    let ordinary = graph.neg(input).unwrap();
    let producer = graph.square(input).unwrap();
    let shared = graph.permute(producer, [1, 0]).unwrap();
    let first = graph.reshape(shared, [1, 3, 2]).unwrap();
    let second = graph.reshape(shared, [3, 2, 1]).unwrap();
    let second = graph.permute(second, [2, 0, 1]).unwrap();
    let fused = graph.add(first, second).unwrap();
    let requested = [ordinary, fused];

    let (cached, cached_counts) = with_rehearsal_mode(true, || schedule_many(&graph, &requested));
    let (reference, reference_counts) =
        with_rehearsal_mode(false, || schedule_many(&graph, &requested));
    let cached = cached.unwrap();
    let reference = reference.unwrap();
    cached.validate().unwrap();
    reference.validate().unwrap();

    assert!(cached_counts.fixed_point_passes >= 2);
    assert_eq!(
        cached_counts.fixed_point_passes,
        reference_counts.fixed_point_passes
    );
    assert_eq!(cached_counts.queries, reference_counts.queries);
    assert!(cached_counts.cache_hits > 0);
    assert_eq!(reference_counts.cache_hits, 0);
    assert!(cached_counts.lowerings < reference_counts.lowerings);
    assert!(
        [producer, shared, first, second]
            .into_iter()
            .all(|removed| cached.items.iter().all(|item| item.node != removed))
    );

    assert_eq!(
        captured_bytes(&graph, &cached, &requested),
        captured_bytes(&graph, &reference, &requested)
    );
}

#[test]
fn rehearsals_are_scoped_to_one_immutable_frontier() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], crate::DType::F32);
    let producer = graph.square(input).unwrap();
    let output = graph.neg(producer).unwrap();
    let roots = BTreeSet::from([output.index()]);
    let no_external = BTreeSet::new();

    let mut fused = OrdinaryScalarFallbackRehearsals::new(&graph, &roots, &no_external);
    let mut projections = crate::rangeify::ProjectedForestCache::new(&graph);
    assert_eq!(
        fused.load_nodes(output, &mut projections).unwrap(),
        &[input.index()]
    );

    let external = BTreeSet::from([producer.index()]);
    let mut materialized = OrdinaryScalarFallbackRehearsals::new(&graph, &roots, &external);
    assert_eq!(
        materialized.load_nodes(output, &mut projections).unwrap(),
        &[producer.index()]
    );
    assert_eq!(
        fused.load_nodes(output, &mut projections).unwrap(),
        &[input.index()]
    );
}

#[test]
fn failed_queries_preserve_the_error_and_are_not_cached() {
    let graph = Graph::new();
    let roots = BTreeSet::new();
    let external = BTreeSet::new();
    let unknown = NodeId::from_index(7);
    let mut rehearsals = OrdinaryScalarFallbackRehearsals::new(&graph, &roots, &external);
    let mut projections = crate::rangeify::ProjectedForestCache::new(&graph);

    let first = rehearsals
        .load_nodes(unknown, &mut projections)
        .unwrap_err();
    let second = rehearsals
        .load_nodes(unknown, &mut projections)
        .unwrap_err();
    assert!(matches!(
        first,
        ScheduleError::Graph(crate::Error::UnknownNode(node)) if node == unknown
    ));
    assert!(matches!(
        second,
        ScheduleError::Graph(crate::Error::UnknownNode(node)) if node == unknown
    ));
    assert!(rehearsals.rehearsals.is_empty());

    let mut malformed = Graph::new();
    let input = malformed.input_dtype("input", [2, 3], crate::DType::F32);
    let output = malformed.dropout(input, 0.5, true, Some(1)).unwrap();
    let roots = BTreeSet::from([output.index()]);
    let mut rehearsals = OrdinaryScalarFallbackRehearsals::new(&malformed, &roots, &external);
    let mut projections = crate::rangeify::ProjectedForestCache::new(&malformed);
    assert!(matches!(
        rehearsals.load_nodes(output, &mut projections),
        Err(ScheduleError::UOp(UOpError::InvalidArgument))
    ));
    assert!(matches!(
        rehearsals.load_nodes(output, &mut projections),
        Err(ScheduleError::UOp(UOpError::InvalidArgument))
    ));
    assert!(rehearsals.rehearsals.is_empty());
}

#[test]
fn external_materialization_matches_the_uncached_reference() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], crate::DType::F32);
    let producer = graph.square(input).unwrap();
    let output = graph.neg(producer).unwrap();
    let requested = [output];

    let (cached, cached_counts) = with_rehearsal_mode(true, || {
        schedule_with_external_materializations(&graph, &requested, &[producer])
    });
    let (reference, reference_counts) = with_rehearsal_mode(false, || {
        schedule_with_external_materializations(&graph, &requested, &[producer])
    });
    let cached = cached.unwrap();
    let reference = reference.unwrap();
    cached.validate().unwrap();
    reference.validate().unwrap();
    assert!(cached_counts.cache_hits > 0);
    assert_eq!(reference_counts.cache_hits, 0);
    assert!(cached_counts.lowerings < reference_counts.lowerings);
    assert_eq!(cached.items[0].external_materializations, vec![producer]);
    assert_eq!(
        captured_bytes(&graph, &cached, &requested),
        captured_bytes(&graph, &reference, &requested)
    );
}
