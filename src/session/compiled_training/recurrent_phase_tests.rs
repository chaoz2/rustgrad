use super::*;

fn legacy_requested_aliases(graph: &Graph, requested: &[NodeId]) -> Result<BTreeSet<NodeId>> {
    Ok(schedule_many(graph, requested)
        .map_err(schedule_error)?
        .requested_passthroughs
        .iter()
        .map(|alias| alias.requested)
        .collect())
}

fn legacy_unowned_requests(graph: &Graph, requested: &[NodeId]) -> Result<BTreeSet<NodeId>> {
    let preview = schedule_many(graph, requested).map_err(schedule_error)?;
    let owners = preview
        .items
        .iter()
        .flat_map(|item| item.outputs.iter())
        .map(|output| output.id)
        .collect::<BTreeSet<_>>();
    Ok(requested
        .iter()
        .copied()
        .filter(|node| !owners.contains(&(node.index() as u64)))
        .collect())
}

fn legacy_public_aliases(graph: &mut Graph, requested: &[NodeId]) -> Result<Vec<NodeId>> {
    let aliases = legacy_requested_aliases(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                graph.contiguous(*node)
            } else {
                Ok(*node)
            }
        })
        .collect()
}

fn legacy_recurrent_public_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
    state_links: &[InferenceStateLink],
) -> Result<Vec<NodeId>> {
    let requested = legacy_public_aliases(graph, requested)?;
    let unowned = legacy_unowned_requests(graph, &requested)?;
    let state_nodes = state_links
        .iter()
        .flat_map(|link| [link.input(), link.output()])
        .collect::<BTreeSet<_>>();
    requested
        .into_iter()
        .map(|node| {
            if state_nodes.contains(&node) || unowned.contains(&node) {
                materialize_compiled_output_alias(graph, node)
            } else {
                Ok(node)
            }
        })
        .collect()
}

fn legacy_state_aliases(graph: &mut Graph, requested: &[NodeId]) -> Result<Vec<NodeId>> {
    let aliases = legacy_unowned_requests(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                materialize_compiled_output_alias(graph, *node)
            } else {
                Ok(*node)
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum CaptureSerializationExpectation {
    ExactBytes,
    StaticSortUnsupported,
}

fn assert_final_schedule_and_capture_match(
    legacy_graph: &Graph,
    legacy_requested: &[NodeId],
    preview_graph: &Graph,
    preview_requested: &[NodeId],
    serialization: CaptureSerializationExpectation,
) {
    let legacy_schedule = schedule_many(legacy_graph, legacy_requested).unwrap();
    let preview_schedule = schedule_many(preview_graph, preview_requested).unwrap();
    let legacy_inventory = legacy_schedule
        .items
        .iter()
        .map(|item| {
            (
                item.node,
                item.outputs
                    .iter()
                    .map(|output| output.id)
                    .collect::<Vec<_>>(),
                item.cache_key,
            )
        })
        .collect::<Vec<_>>();
    let preview_inventory = preview_schedule
        .items
        .iter()
        .map(|item| {
            (
                item.node,
                item.outputs
                    .iter()
                    .map(|output| output.id)
                    .collect::<Vec<_>>(),
                item.cache_key,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(preview_inventory, legacy_inventory);
    assert_eq!(
        preview_schedule.requested_materializations,
        legacy_schedule.requested_materializations
    );
    assert_eq!(
        preview_schedule.requested_passthroughs,
        legacy_schedule.requested_passthroughs
    );
    let legacy_capture =
        crate::CapturedSchedule::capture(legacy_graph, &legacy_schedule, legacy_requested).unwrap();
    let preview_capture =
        crate::CapturedSchedule::capture(preview_graph, &preview_schedule, preview_requested)
            .unwrap();
    match serialization {
        CaptureSerializationExpectation::ExactBytes => {
            assert_eq!(
                preview_capture.to_bytes().unwrap(),
                legacy_capture.to_bytes().unwrap()
            );
        }
        CaptureSerializationExpectation::StaticSortUnsupported => {
            let expected = crate::ReplayError::Unsupported(
                "static sort capture serialization is unsupported".into(),
            );
            assert_eq!(legacy_capture.to_bytes().unwrap_err(), expected);
            assert_eq!(preview_capture.to_bytes().unwrap_err(), expected);
        }
    }
}

fn assert_legacy_and_preview_materialization_match(
    graph: &Graph,
    requested: &[NodeId],
    state_links: &[InferenceStateLink],
    serialization: CaptureSerializationExpectation,
) {
    let mut legacy = graph.clone();
    let mut preview = graph.clone();
    let legacy_requested =
        legacy_recurrent_public_aliases(&mut legacy, requested, state_links).unwrap();
    let preview_requested =
        materialize_compiled_recurrent_public_aliases(&mut preview, requested, state_links)
            .unwrap();
    assert_eq!(preview_requested, legacy_requested);
    assert_eq!(preview.node_count(), legacy.node_count());
    assert_final_schedule_and_capture_match(
        &legacy,
        &legacy_requested,
        &preview,
        &preview_requested,
        serialization,
    );

    let mut legacy = graph.clone();
    let mut preview = graph.clone();
    let legacy_requested = legacy_state_aliases(&mut legacy, requested).unwrap();
    let preview_requested = materialize_compiled_state_aliases(&mut preview, requested).unwrap();
    assert_eq!(preview_requested, legacy_requested);
    assert_eq!(preview.node_count(), legacy.node_count());
    assert_final_schedule_and_capture_match(
        &legacy,
        &legacy_requested,
        &preview,
        &preview_requested,
        serialization,
    );
}

#[test]
fn planned_ownership_preserves_compiled_materialization_and_capture_bytes() {
    let empty = Graph::new();
    let preview = crate::schedule::requested_schedule_ownership(&empty, &[]).unwrap();
    assert!(preview.scheduled().is_empty());
    assert!(preview.passthroughs().is_empty());

    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 2], DType::F32);
    let constant = graph.constant(TensorData::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let lazy_full = graph
        .lazy_full_with_dtype([2, 2], crate::Scalar::F(3.0), DType::F32)
        .unwrap();
    let static_view = graph.permute(input, [1, 0]).unwrap();
    let producer = graph.square(input).unwrap();
    let computed_view = graph.permute(producer, [1, 0]).unwrap();
    let backward_view = graph.contiguous_backward(computed_view).unwrap();
    let product = graph.matmul(computed_view, constant).unwrap();
    let copied = graph.contiguous(computed_view).unwrap();
    assert_legacy_and_preview_materialization_match(
        &graph,
        &[
            input,
            constant,
            lazy_full,
            static_view,
            computed_view,
            backward_view,
            product,
            copied,
            backward_view,
        ],
        &[],
        CaptureSerializationExpectation::ExactBytes,
    );

    let mut sort = Graph::new();
    let input = sort.input_dtype("input", [2, 2], DType::F32);
    let (values, indices) = sort.sort(input, 1, false).unwrap();
    let requested_values = sort.contiguous_backward(values).unwrap();
    assert_legacy_and_preview_materialization_match(
        &sort,
        &[requested_values, indices, requested_values],
        &[],
        CaptureSerializationExpectation::StaticSortUnsupported,
    );

    let mut zero = Graph::new();
    let input = zero.input_dtype("input", [0, 2], DType::F32);
    let producer = zero.square(input).unwrap();
    let view = zero.permute(producer, [1, 0]).unwrap();
    assert_legacy_and_preview_materialization_match(
        &zero,
        &[view, view],
        &[],
        CaptureSerializationExpectation::ExactBytes,
    );

    let mut recurrent = Graph::new();
    let state = recurrent.input_dtype("state", [2], DType::F32);
    let next = recurrent.square(state).unwrap();
    let public = recurrent.neg(next).unwrap();
    let links = [InferenceStateLink::new(state, next)];
    assert_legacy_and_preview_materialization_match(
        &recurrent,
        &[next, public],
        &links,
        CaptureSerializationExpectation::ExactBytes,
    );
    let mut collision = recurrent.clone();
    let collision_requested =
        materialize_compiled_recurrent_public_aliases(&mut collision, &[next, public], &links)
            .unwrap();
    assert_ne!(collision_requested[0], next);
    assert!(matches!(
        collision.op(collision_requested[0]),
        Ok(crate::Op::Contiguous { input }) if *input == next
    ));
}
