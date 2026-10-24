use super::*;

#[derive(Debug, Eq, PartialEq)]
struct Snapshot {
    operation: Operation,
    ty: Option<UType>,
    tag: Option<UOpTag>,
    children: Vec<usize>,
}

fn snapshot(projection: &RangeifiedProjection, caller: &UOp) -> (Vec<Snapshot>, Vec<Vec<bool>>) {
    enum Frame {
        Visit(UOp),
        Finish(UOp),
    }
    let mut identities = HashMap::new();
    let mut nodes = Vec::<(usize, Snapshot)>::new();
    let mut occurrences = Vec::new();
    for root in projection
        .predicate
        .iter()
        .chain(std::iter::once(&projection.expression))
        .chain(std::iter::once(caller))
    {
        let mut pending = vec![Frame::Visit(root.clone())];
        while let Some(frame) = pending.pop() {
            match frame {
                Frame::Visit(node) => {
                    occurrences.push(node.node_identity());
                    if identities.contains_key(&node.node_identity()) {
                        continue;
                    }
                    pending.push(Frame::Finish(node.clone()));
                    pending.extend(node.sources().iter().rev().cloned().map(Frame::Visit));
                }
                Frame::Finish(node) => {
                    if identities.contains_key(&node.node_identity()) {
                        continue;
                    }
                    let children = node
                        .sources()
                        .iter()
                        .map(|source| identities[&source.node_identity()])
                        .collect();
                    let index = nodes.len();
                    identities.insert(node.node_identity(), index);
                    nodes.push((
                        node.node_identity(),
                        Snapshot {
                            operation: node.operation().clone(),
                            ty: node.ty(),
                            tag: node.tag().cloned(),
                            children,
                        },
                    ));
                }
            }
        }
    }
    let matrix = occurrences
        .iter()
        .map(|lhs| occurrences.iter().map(|rhs| lhs == rhs).collect())
        .collect();
    (nodes.into_iter().map(|(_, node)| node).collect(), matrix)
}

fn range_owners(root: &UOp) -> BTreeMap<usize, ()> {
    let mut owners = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        if !seen.insert(node.node_identity()) {
            continue;
        }
        if matches!(node.operation(), Operation::Range(_)) {
            owners.insert(node.node_identity(), ());
        }
        pending.extend(node.sources());
    }
    owners
}

fn projection_owners(projection: &RangeifiedProjection) -> std::collections::BTreeSet<usize> {
    let mut owners = std::collections::BTreeSet::new();
    let mut pending = projection
        .predicate
        .iter()
        .chain(std::iter::once(&projection.expression))
        .collect::<Vec<_>>();
    while let Some(node) = pending.pop() {
        if !owners.insert(node.node_identity()) {
            continue;
        }
        pending.extend(node.sources());
    }
    owners
}

#[test]
fn reemission_preserves_complete_partitions_and_two_fresh_uses() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 2], DType::F32);
    let pad = graph
        .pad(input, [(1, 0), (0, 1)], crate::Scalar::I(0))
        .unwrap();
    let node = graph.permute(pad, [1, 0]).unwrap();
    let shape = graph.shape(node).unwrap().clone();
    let make_range = || ProjectedForestCache::range(&shape).unwrap();
    let first_range = make_range();
    let mut cache = ProjectedForestCache::new(&graph);
    let first = cache
        .projection(&graph, node, &shape, &first_range)
        .unwrap();
    let second_range = make_range();
    let second = cache
        .projection(&graph, node, &shape, &second_range)
        .unwrap();
    let third_range = make_range();
    let third = cache
        .projection(&graph, node, &shape, &third_range)
        .unwrap();
    let expected = snapshot(&first, &first_range);
    assert_eq!(snapshot(&second, &second_range), expected);
    assert_eq!(snapshot(&third, &third_range), expected);
    assert!(!first.expression.shares_node_with(&second.expression));
    assert!(!second.expression.shares_node_with(&third.expression));
    let first_owners = projection_owners(&first);
    let second_owners = projection_owners(&second);
    let third_owners = projection_owners(&third);
    assert!(first_owners.is_disjoint(&second_owners));
    assert!(first_owners.is_disjoint(&third_owners));
    assert!(second_owners.is_disjoint(&third_owners));
    for (projection, caller) in [
        (&first, &first_range),
        (&second, &second_range),
        (&third, &third_range),
    ] {
        let address = range_owners(&projection.expression);
        let predicate = range_owners(projection.predicate.as_ref().unwrap());
        assert!(!address.is_empty());
        assert!(!predicate.is_empty());
        assert!(
            address
                .keys()
                .all(|identity| !predicate.contains_key(identity))
        );
        assert!(!address.contains_key(&caller.node_identity()));
        assert!(!predicate.contains_key(&caller.node_identity()));
    }
}

#[test]
fn cache_matches_released_oracle_across_projected_movement_families() {
    let mut graph = Graph::new();
    let source = graph.input_dtype("source", [2, 2], DType::F32);
    let movement = graph.permute(source, [1, 0]).unwrap();
    let reshaped = graph.reshape(movement, [4]).unwrap();
    let shrunk = graph.shrink(source, [(0, 1), (1, 2)]).unwrap();
    let reversed = graph
        .stride(
            source,
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
    let scalar = graph.input_dtype("scalar", [], DType::F32);
    let padded = graph
        .pad(source, [(1, 0), (0, 1)], crate::Scalar::F(0.0))
        .unwrap();
    let repeated = graph.repeat(source, &[2, 3]).unwrap();
    let broadcast_source = graph.input_dtype("broadcast", [1, 2], DType::F32);
    let broadcast = graph.expand(broadcast_source, [3, 2]).unwrap();
    let empty_source = graph.input_dtype("empty", [0, 2], DType::F32);
    let zero_domain = graph
        .pad(empty_source, [(1, 0), (0, 0)], crate::Scalar::I(0))
        .unwrap();
    let mut cache = ProjectedForestCache::new(&graph);
    for node in [
        movement,
        reshaped,
        shrunk,
        reversed,
        scalar,
        padded,
        repeated,
        broadcast,
        empty_source,
        zero_domain,
    ] {
        let shape = graph.shape(node).unwrap().clone();
        let range = ProjectedForestCache::range(&shape).unwrap();
        let oracle = projected_view(&graph, node, &shape, &range).unwrap();
        let first = cache.projection(&graph, node, &shape, &range).unwrap();
        let second_range = ProjectedForestCache::range(&shape).unwrap();
        let second = cache
            .projection(&graph, node, &shape, &second_range)
            .unwrap();
        let oracle_snapshot = snapshot(&oracle, &range);
        assert_eq!(first, oracle);
        assert_eq!(second, oracle);
        assert_eq!(snapshot(&first, &range), oracle_snapshot);
        assert_eq!(snapshot(&second, &second_range), oracle_snapshot);
        assert_eq!(graph.dtype(first.source), graph.dtype(node));
        let oracle_index = index_for(&graph, &oracle, &shape);
        let oracle_plan =
            crate::projected_index::ProjectedIndexPlan::from_index(&oracle_index).unwrap();
        let emitted_index = index_for(&graph, &second, &shape);
        let emitted_plan =
            crate::projected_index::ProjectedIndexPlan::from_index(&emitted_index).unwrap();
        for lane in 0..shape.numel().unwrap() {
            let valid = oracle_plan.valid(lane).unwrap();
            assert_eq!(emitted_plan.valid(lane).unwrap(), valid);
            if valid {
                assert_eq!(
                    emitted_plan.offset(lane).unwrap(),
                    oracle_plan.offset(lane).unwrap()
                );
            }
        }
    }
}

#[test]
fn coordinate_reuse_preserves_index_storage_types_and_lane_maps() {
    for dtype in [
        DType::Bool,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let source = graph.input_dtype("source", [2, 2], dtype);
        let node = graph.permute(source, [1, 0]).unwrap();
        let shape = graph.shape(node).unwrap().clone();
        let range = ProjectedForestCache::range(&shape).unwrap();
        let mut cache = ProjectedForestCache::new(&graph);
        for _ in 0..2 {
            let projection = cache.projection(&graph, node, &shape, &range).unwrap();
            let index = index_for(&graph, &projection, &shape);
            assert_eq!(index.ty(), Some(UType::scalar(dtype)), "{dtype:?}");
            let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).unwrap();
            assert_eq!(
                (0..4)
                    .map(|lane| plan.offset(lane).unwrap())
                    .collect::<Vec<_>>(),
                [0, 2, 1, 3],
                "{dtype:?}"
            );
        }
    }
}

#[test]
fn index_admission_keeps_exact_depth_combined_occurrence_and_byte_gates() {
    use crate::projected_index::{
        MAX_PROJECTED_INDEX_DEPTH, MAX_PROJECTED_INDEX_NODES, ProjectedIndexPlan,
    };
    use crate::uop::{Binary, UOpError};
    let ty = UType::scalar(DType::I64);
    let range = ProjectedForestCache::range(&Shape::from([1])).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [1], DType::F32);
    let shape = Shape::from([1]);
    for depth in [MAX_PROJECTED_INDEX_DEPTH, MAX_PROJECTED_INDEX_DEPTH + 1] {
        let mut expression = range.clone();
        for _ in 0..depth {
            expression = UOp::from_operation(
                Operation::Binary(Binary::Add),
                Some(ty),
                vec![expression, UOp::constant(0, ty)],
            );
        }
        let original = RangeifiedProjection {
            source: input,
            expression,
            predicate: None,
        };
        let emitted = ProjectedForest::extract(&original, &range).emit(&range);
        for projection in [&original, &emitted] {
            let result = ProjectedIndexPlan::from_index(&index_for(&graph, projection, &shape));
            if depth == MAX_PROJECTED_INDEX_DEPTH {
                assert_eq!(result.unwrap().offset(0).unwrap(), 0);
            } else {
                assert!(matches!(result, Err(UOpError::InvalidIndex)));
            }
        }
    }
    // Each root is independently well below the ordinary 4096-occurrence
    // limit. Their shared diamonds remain shared in the descriptor, but that
    // never exempts the assembled Index from the expanded combined budget.
    assert_eq!(MAX_PROJECTED_INDEX_NODES, 4096);
    let mut expression = UOp::constant(0, ty);
    let mut predicate = UOp::scalar_constant(DType::Bool, 1, UType::scalar(DType::Bool));
    for _ in 0..10 {
        expression = UOp::from_operation(
            Operation::Binary(Binary::Add),
            Some(ty),
            vec![expression.clone(), expression],
        );
        predicate = UOp::from_operation(
            Operation::GraphLogical(crate::LogicalOp::And),
            Some(UType::scalar(DType::Bool)),
            vec![predicate.clone(), predicate],
        );
    }
    let expression_only = RangeifiedProjection {
        source: input,
        expression: expression.clone(),
        predicate: None,
    };
    assert!(ProjectedIndexPlan::from_index(&index_for(&graph, &expression_only, &shape)).is_ok());
    for _ in 0..2 {
        predicate = UOp::from_operation(
            Operation::GraphLogical(crate::LogicalOp::Not),
            Some(UType::scalar(DType::Bool)),
            vec![predicate],
        );
    }
    let predicate_only = RangeifiedProjection {
        source: input,
        expression: UOp::constant(0, ty),
        predicate: Some(predicate.clone()),
    };
    assert!(ProjectedIndexPlan::from_index(&index_for(&graph, &predicate_only, &shape)).is_ok());
    for over_limit in [false, true] {
        let predicate = if over_limit {
            UOp::from_operation(
                Operation::GraphLogical(crate::LogicalOp::Not),
                Some(UType::scalar(DType::Bool)),
                vec![predicate.clone()],
            )
        } else {
            predicate.clone()
        };
        let original = RangeifiedProjection {
            source: input,
            expression: expression.clone(),
            predicate: Some(predicate),
        };
        let emitted = ProjectedForest::extract(&original, &range).emit(&range);
        for projection in [&original, &emitted] {
            let result = ProjectedIndexPlan::from_index(&index_for(&graph, projection, &shape));
            if over_limit {
                assert!(matches!(result, Err(UOpError::InvalidIndex)));
            } else {
                assert_eq!(result.unwrap().offset(0).unwrap(), 0);
            }
        }
    }
    let extent = usize::MAX / DType::F64.itemsize() + 1;
    let mut graph = Graph::new();
    let source = graph.input_dtype("huge", [extent], DType::F64);
    let shape = Shape::from([extent]);
    let range = ProjectedForestCache::range(&shape).unwrap();
    let mut cache = ProjectedForestCache::new(&graph);
    for _ in 0..2 {
        let projection = cache.projection(&graph, source, &shape, &range).unwrap();
        assert!(matches!(
            ProjectedIndexPlan::from_index(&index_for(&graph, &projection, &shape)),
            Err(UOpError::InvalidIndex)
        ));
    }
    assert_eq!(cache.retained_forests, 1);
}

#[test]
fn predicate_depth_and_independent_byte_overflows_remain_final_index_gates() {
    use crate::projected_index::{MAX_PROJECTED_INDEX_DEPTH, ProjectedIndexPlan};
    use crate::uop::UOpError;
    let integer = UType::scalar(DType::I64);
    let boolean = UType::scalar(DType::Bool);
    let mut graph = Graph::new();
    let small = graph.input_dtype("small", [1], DType::F64);
    let shape = Shape::from([1]);
    let range = ProjectedForestCache::range(&shape).unwrap();
    for depth in [MAX_PROJECTED_INDEX_DEPTH, MAX_PROJECTED_INDEX_DEPTH + 1] {
        let mut predicate = UOp::scalar_constant(DType::Bool, 1, boolean);
        for _ in 0..depth {
            predicate = UOp::from_operation(
                Operation::GraphLogical(crate::LogicalOp::Not),
                Some(boolean),
                vec![predicate],
            );
        }
        let original = RangeifiedProjection {
            source: small,
            expression: UOp::constant(0, integer),
            predicate: Some(predicate),
        };
        let emitted = ProjectedForest::extract(&original, &range).emit(&range);
        for projection in [&original, &emitted] {
            let result = ProjectedIndexPlan::from_index(&index_for(&graph, projection, &shape));
            if depth == MAX_PROJECTED_INDEX_DEPTH {
                assert!(result.unwrap().valid(0).unwrap());
            } else {
                assert!(matches!(result, Err(UOpError::InvalidIndex)));
            }
        }
    }

    let extent = usize::MAX / DType::F64.itemsize() + 1;
    let huge_shape = Shape::from([extent]);
    let huge = graph.input_dtype("huge", huge_shape.clone(), DType::F64);
    for (name, source, output_shape) in [
        ("input only", huge, &shape),
        ("output only", small, &huge_shape),
    ] {
        let caller = ProjectedForestCache::range(output_shape).unwrap();
        let original = RangeifiedProjection {
            source,
            expression: UOp::constant(0, integer),
            predicate: None,
        };
        let emitted = ProjectedForest::extract(&original, &caller).emit(&caller);
        for projection in [&original, &emitted] {
            assert!(
                matches!(
                    ProjectedIndexPlan::from_index(&index_for(&graph, projection, output_shape)),
                    Err(UOpError::InvalidIndex)
                ),
                "{name}"
            );
        }
    }
    // A valid one-element broadcast can still overflow output storage bytes.
    // Successful cached coordinate construction must not admit that Index.
    let caller = ProjectedForestCache::range(&huge_shape).unwrap();
    let mut cache = ProjectedForestCache::new(&graph);
    for _ in 0..2 {
        let projection = cache
            .projection(&graph, small, &huge_shape, &caller)
            .unwrap();
        assert!(matches!(
            ProjectedIndexPlan::from_index(&index_for(&graph, &projection, &huge_shape)),
            Err(UOpError::InvalidIndex)
        ));
    }
    assert_eq!(cache.retained_forests, 1);
}

#[test]
fn held_forest_releases_projector_owners() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let node = graph.pad(input, [(1, 1)], crate::Scalar::I(0)).unwrap();
    let shape = graph.shape(node).unwrap().clone();
    let range = ProjectedForestCache::range(&shape).unwrap();
    let projection = projected_view(&graph, node, &shape, &range).unwrap();
    let mut caller_owned = std::collections::BTreeSet::new();
    let mut caller_pending = vec![range.clone()];
    while let Some(value) = caller_pending.pop() {
        if !caller_owned.insert(value.node_identity()) {
            continue;
        }
        caller_pending.extend(value.sources().iter().cloned());
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut pending = projection
        .predicate
        .iter()
        .chain(std::iter::once(&projection.expression))
        .cloned()
        .collect::<Vec<_>>();
    let mut owners = Vec::new();
    while let Some(value) = pending.pop() {
        if !seen.insert(value.node_identity()) {
            continue;
        }
        if !caller_owned.contains(&value.node_identity()) {
            owners.push(value.weak_owner());
        }
        pending.extend(value.sources().iter().cloned());
    }
    let forest = ProjectedForest::extract(&projection, &range);
    drop(projection);
    assert!(owners.iter().all(crate::uop::WeakUOp::is_released));
    assert_eq!(forest.emit(&range).source, input);
}

#[test]
fn descriptor_preserves_distinct_equal_children_tags_and_cross_root_sharing() {
    let ty = UType::scalar(DType::I64);
    let equal_left = UOp::constant(3, ty);
    let equal_right = UOp::constant(3, ty);
    assert!(!equal_left.shares_node_with(&equal_right));
    let shared = UOp::constant(1, ty).retag(Some(UOpTag::Text("shared".into())));
    let equal_pair = UOp::from_operation(
        Operation::Binary(crate::uop::Binary::Add),
        Some(ty),
        vec![equal_left, equal_right],
    );
    let expression = UOp::from_operation(
        Operation::Binary(crate::uop::Binary::Add),
        Some(ty),
        vec![equal_pair, shared.clone()],
    );
    let predicate = UOp::from_operation(
        Operation::GraphCompare(crate::CompareOp::Eq),
        Some(UType::scalar(DType::Bool)),
        vec![shared, UOp::constant(1, ty)],
    );
    let projection = RangeifiedProjection {
        source: NodeId::from_index(0),
        expression,
        predicate: Some(predicate),
    };
    let caller = UOp::from_operation(Operation::Range(0), Some(ty), vec![UOp::constant(4, ty)]);
    let expected = snapshot(&projection, &caller);
    let forest = ProjectedForest::extract(&projection, &caller);
    let emitted = forest.emit(&caller);
    assert_eq!(snapshot(&emitted, &caller), expected);
}

#[test]
fn source_queries_do_not_emit_and_noncanonical_ranges_bypass_cache() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let node = graph.pad(input, [(1, 1)], crate::Scalar::I(0)).unwrap();
    let shape = graph.shape(node).unwrap().clone();
    let ((first, second), counts) = with_projected_forest_mode(true, || {
        let mut cache = ProjectedForestCache::new(&graph);
        (
            cache.classification(&graph, node, &shape),
            cache.classification(&graph, node, &shape),
        )
    });
    assert_eq!(first, Ok(ProjectionSource::Predicated(input)));
    assert_eq!(second, first);
    assert_eq!(
        super::super::view::predicated_source(&graph, node, &shape),
        Ok(input)
    );
    assert_eq!(
        super::super::view::projected_source(&graph, node, &shape),
        Ok(input)
    );
    assert_eq!(
        (counts.extractions, counts.hits, counts.emissions),
        (1, 1, 0)
    );

    let range = UOp::from_operation(
        Operation::Range(0),
        None,
        vec![UOp::constant(
            i64::try_from(shape.numel().unwrap()).unwrap(),
            UType::scalar(DType::I64),
        )],
    );
    let (_, counts) = with_projected_forest_mode(true, || {
        let mut cache = ProjectedForestCache::new(&graph);
        [
            cache.projection(&graph, node, &shape, &range),
            cache.projection(&graph, node, &shape, &range),
        ]
    });
    assert_eq!(
        (counts.hits, counts.extractions, counts.emissions),
        (0, 0, 0)
    );
    assert_eq!(counts.released_builds, 2);

    let tagged_range = ProjectedForestCache::range(&shape)
        .unwrap()
        .retag(Some(UOpTag::Text("valid noncanonical caller".into())));
    let oracle = projected_view(&graph, node, &shape, &tagged_range).unwrap();
    let (results, counts) = with_projected_forest_mode(true, || {
        let mut cache = ProjectedForestCache::new(&graph);
        let results = [
            cache
                .projection(&graph, node, &shape, &tagged_range)
                .unwrap(),
            cache
                .projection(&graph, node, &shape, &tagged_range)
                .unwrap(),
        ];
        assert_eq!(cache.retained_forests, 0);
        results
    });
    for result in results {
        assert_eq!(
            snapshot(&result, &tagged_range),
            snapshot(&oracle, &tagged_range)
        );
        assert!(
            crate::projected_index::ProjectedIndexPlan::from_index(&index_for(
                &graph, &result, &shape
            ))
            .is_ok()
        );
    }
    assert_eq!(
        (counts.hits, counts.extractions, counts.emissions),
        (0, 0, 0)
    );
    assert_eq!(counts.released_builds, 2);

    let unknown = NodeId::from_index(graph.node_count());
    let (errors, counts) = with_projected_forest_mode(true, || {
        let mut cache = ProjectedForestCache::new(&graph);
        [
            cache.classification(&graph, unknown, &shape),
            cache.classification(&graph, unknown, &shape),
        ]
    });
    assert_eq!(
        errors,
        [Err(RangeifyError::Invalid), Err(RangeifyError::Invalid)]
    );
    assert_eq!(
        (counts.hits, counts.extractions, counts.emissions),
        (0, 0, 0)
    );
    assert_eq!(counts.released_builds, 2);
}

#[test]
fn caller_range_and_bound_keep_their_complete_external_sharing_class() {
    let ty = UType::scalar(DType::I64);
    let make_caller =
        || UOp::from_operation(Operation::Range(0), Some(ty), vec![UOp::constant(4, ty)]);
    let caller = make_caller();
    let bound = caller.sources()[0].clone();
    let projection = RangeifiedProjection {
        source: NodeId::from_index(0),
        expression: UOp::from_operation(
            Operation::Binary(crate::uop::Binary::Add),
            Some(ty),
            vec![caller.clone(), bound.clone()],
        ),
        predicate: Some(UOp::from_operation(
            Operation::GraphCompare(crate::CompareOp::Lt),
            Some(UType::scalar(DType::Bool)),
            vec![caller.clone(), bound],
        )),
    };
    let forest = ProjectedForest::extract(&projection, &caller);
    let fresh_caller = make_caller();
    let emitted = forest.emit(&fresh_caller);
    assert_eq!(
        snapshot(&emitted, &fresh_caller),
        snapshot(&projection, &caller)
    );
    assert!(emitted.expression.sources()[0].shares_node_with(&fresh_caller));
    assert!(emitted.expression.sources()[1].shares_node_with(&fresh_caller.sources()[0]));
    assert!(
        emitted.predicate.as_ref().unwrap().sources()[1]
            .shares_node_with(&fresh_caller.sources()[0])
    );
}

fn index_for(graph: &Graph, projection: &RangeifiedProjection, shape: &Shape) -> UOp {
    use crate::uop::{AddressSpace, AddressValue, IndexAddressing, IndexValue};
    let input_shape = graph.shape(projection.source).unwrap().clone();
    let ty = UType::scalar(graph.dtype(projection.source).unwrap());
    let buffer = projection.source.index() as u64;
    let address = UOp::from_operation(
        Operation::DefineGlobal(AddressValue {
            space: AddressSpace::Global,
            name: format!("b{buffer}"),
            element: ty,
        }),
        Some(ty),
        vec![],
    );
    let mut sources = vec![address, projection.expression.clone()];
    sources.extend(projection.predicate.clone());
    UOp::from_operation(
        Operation::Index(IndexValue::Buffer {
            buffer,
            elements: input_shape.numel().unwrap(),
            input_shape,
            output_shape: shape.clone(),
            addressing: if projection.predicate.is_some() {
                IndexAddressing::Predicated
            } else {
                IndexAddressing::Projected
            },
        }),
        Some(ty),
        sources,
    )
}

#[test]
fn foreign_graph_with_coincident_ids_and_shapes_never_uses_cached_semantics() {
    let mut first_graph = Graph::new();
    let input = first_graph.input_dtype("input", [2, 2], DType::F32);
    let first_node = first_graph.permute(input, [1, 0]).unwrap();
    let mut other_graph = Graph::new();
    let other_input = other_graph.input_dtype("input", [1, 2], DType::F32);
    let other_node = other_graph
        .pad(other_input, [(1, 0), (0, 0)], crate::Scalar::I(0))
        .unwrap();
    assert_eq!(first_node, other_node);
    let shape = first_graph.shape(first_node).unwrap().clone();
    assert_eq!(other_graph.shape(other_node).unwrap(), &shape);
    let range = ProjectedForestCache::range(&shape).unwrap();
    let mut cache = ProjectedForestCache::new(&first_graph);
    let ((first, foreign, classification, repeated), counts) =
        with_projected_forest_mode(true, || {
            (
                cache
                    .projection(&first_graph, first_node, &shape, &range)
                    .unwrap(),
                cache
                    .projection(&other_graph, other_node, &shape, &range)
                    .unwrap(),
                cache
                    .classification(&other_graph, other_node, &shape)
                    .unwrap(),
                cache
                    .projection(&first_graph, first_node, &shape, &range)
                    .unwrap(),
            )
        });
    let oracle = projected_view(&other_graph, other_node, &shape, &range).unwrap();
    assert_eq!(snapshot(&foreign, &range), snapshot(&oracle, &range));
    assert_eq!(snapshot(&first, &range), snapshot(&repeated, &range));
    assert_ne!(first.expression, foreign.expression);
    assert_eq!(classification, ProjectionSource::Predicated(other_input));
    assert!(first.predicate.is_none());
    assert_eq!(
        (counts.extractions, counts.hits, counts.emissions),
        (1, 1, 1)
    );
    let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index_for(
        &other_graph,
        &foreign,
        &shape,
    ))
    .unwrap();
    assert_eq!(
        (0..4)
            .map(|lane| plan.valid(lane).unwrap())
            .collect::<Vec<_>>(),
        [false, false, true, true]
    );
}

#[test]
fn aggregate_retention_limits_fall_back_without_changing_projection_admission() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let first_node = graph.pad(input, [(1, 1)], crate::Scalar::I(0)).unwrap();
    let second_node = graph.pad(input, [(2, 0)], crate::Scalar::I(0)).unwrap();
    let shape = graph.shape(first_node).unwrap().clone();
    let range = ProjectedForestCache::range(&shape).unwrap();
    let oracle = projected_view(&graph, first_node, &shape, &range).unwrap();
    let cost = ProjectedForest::extract(&oracle, &range)
        .retention_units(&shape)
        .unwrap();
    for (name, limit, expected_hits, expected_bypasses) in [
        ("exact units", cost, 1, 0),
        ("one unit short", cost - 1, 0, 2),
    ] {
        let mut cache = ProjectedForestCache::new(&graph);
        cache.unit_limit = limit;
        let (results, counts) = with_projected_forest_mode(true, || {
            [
                cache
                    .projection(&graph, first_node, &shape, &range)
                    .unwrap(),
                cache
                    .projection(&graph, first_node, &shape, &range)
                    .unwrap(),
            ]
        });
        for result in results {
            assert_eq!(
                snapshot(&result, &range),
                snapshot(&oracle, &range),
                "{name}"
            );
            assert!(
                crate::projected_index::ProjectedIndexPlan::from_index(&index_for(
                    &graph, &result, &shape
                ))
                .is_ok(),
                "{name}"
            );
        }
        assert_eq!(counts.hits, expected_hits, "{name}");
        assert_eq!(counts.retention_bypasses, expected_bypasses, "{name}");
        assert!(cache.retained_units <= limit, "{name}");
    }
    let mut cache = ProjectedForestCache::new(&graph);
    cache.forest_limit = 1;
    cache
        .projection(&graph, first_node, &shape, &range)
        .unwrap();
    let ((), counts) = with_projected_forest_mode(true, || {
        for _ in 0..2 {
            let result = cache
                .projection(&graph, second_node, &shape, &range)
                .unwrap();
            let oracle = projected_view(&graph, second_node, &shape, &range).unwrap();
            assert_eq!(snapshot(&result, &range), snapshot(&oracle, &range));
        }
    });
    assert_eq!(cache.retained_forests, 1);
    assert_eq!((counts.hits, counts.retention_bypasses), (0, 2));
}
