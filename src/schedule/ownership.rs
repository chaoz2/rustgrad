use super::*;

/// The requested-output ownership facts available before fusion rehearsal or
/// executable lowering. Callers must still build and validate the final
/// schedule before publishing an executable plan.
#[derive(Debug)]
pub(crate) struct RequestedScheduleOwnership {
    scheduled: BTreeSet<NodeId>,
    passthroughs: BTreeSet<NodeId>,
}

impl RequestedScheduleOwnership {
    pub(crate) fn scheduled(&self) -> &BTreeSet<NodeId> {
        &self.scheduled
    }

    pub(crate) fn passthroughs(&self) -> &BTreeSet<NodeId> {
        &self.passthroughs
    }
}

pub(super) struct ScheduleOwnershipPlan {
    pub(super) outputs: Vec<NodeId>,
    pub(super) external: BTreeSet<usize>,
    pub(super) consumers: Vec<usize>,
    pub(super) requested: BTreeSet<usize>,
    pub(super) requested_passthroughs: Vec<RequestedPassthrough>,
    pub(super) direct_payload_operands: BTreeSet<usize>,
    pub(super) movement_operand_owners: BTreeMap<usize, BTreeSet<usize>>,
    pub(super) movement_operands: BTreeSet<usize>,
    pub(super) requested_passthrough_sources: BTreeSet<usize>,
    pub(super) roots: BTreeSet<usize>,
}

impl ScheduleOwnershipPlan {
    pub(super) fn build(
        graph: &Graph,
        outputs: &[NodeId],
        external: &BTreeSet<usize>,
    ) -> Result<Self, ScheduleError> {
        let outputs = outputs
            .iter()
            .map(|requested| {
                graph
                    .contiguous_backward_owner(*requested)
                    .map_err(ScheduleError::Graph)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let external = external
            .iter()
            .map(|index| {
                graph
                    .contiguous_backward_owner(NodeId::from_index(*index))
                    .map(NodeId::index)
                    .map_err(ScheduleError::Graph)
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut needed = BTreeSet::new();
        let mut consumers = vec![0usize; graph.node_count()];
        for output in &outputs {
            graph.op(*output).map_err(ScheduleError::Graph)?;
            mark_needed(graph, *output, &mut needed, &mut consumers, &external)?;
        }
        // Sort selectors are one coupled producer. Preserve the requested
        // selector while making its sibling available to the same owner.
        let marked = needed.iter().copied().collect::<Vec<_>>();
        for index in marked {
            if let Some(sibling) = sort_sibling(graph, NodeId::from_index(index))? {
                needed.insert(sibling.index());
            }
        }
        let requested = outputs
            .iter()
            .copied()
            .map(NodeId::index)
            .collect::<BTreeSet<_>>();
        let mut requested_passthroughs = Vec::new();
        let mut requested_passthrough_ids = BTreeSet::new();
        for &requested_node in &outputs {
            if requested_passthrough_ids.contains(&requested_node.index())
                || matches!(
                    graph.op(requested_node).map_err(ScheduleError::Graph)?,
                    Op::Input { .. } | Op::Constant(_)
                )
            {
                continue;
            }
            let Ok(rangeified) = crate::rangeify::static_view(graph, requested_node)
                .or_else(|_| crate::rangeify::computed_view(graph, requested_node))
            else {
                continue;
            };
            if rangeified.source == requested_node {
                continue;
            }
            let requested_shape = graph.shape(requested_node).map_err(ScheduleError::Graph)?;
            let requested_dtype = graph.dtype(requested_node).map_err(ScheduleError::Graph)?;
            let source_dtype = graph
                .dtype(rangeified.source)
                .map_err(ScheduleError::Graph)?;
            if rangeified.view.logical_shape != *requested_shape || requested_dtype != source_dtype
            {
                return Err(ScheduleError::Binding(
                    "requested passthrough graph descriptor is invalid".into(),
                ));
            }
            let mut desc = buffer(graph, rangeified.source, true)?;
            desc.view = Some(rangeified.view);
            let passthrough = RequestedPassthrough {
                requested: requested_node,
                source: rangeified.source,
                desc,
            };
            passthrough.validate_against_graph(graph)?;
            requested_passthrough_ids.insert(requested_node.index());
            requested_passthroughs.push(passthrough);
        }
        // Typed payloads name dense operand identities directly and therefore
        // protect those operands from alias-only requested ownership.
        let direct_payload_operands = needed
            .iter()
            .map(|index| {
                let op = graph
                    .op(NodeId::from_index(*index))
                    .map_err(ScheduleError::Graph)?;
                op_direct_payload_operands(graph, op)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .map(NodeId::index)
            .filter(|index| {
                !matches!(
                    graph.op(NodeId::from_index(*index)),
                    Ok(Op::Input { .. } | Op::Constant(_))
                )
            })
            .collect::<BTreeSet<_>>();
        // Movement plans likewise own an exact pointer ABI.
        let mut movement_operand_owners = BTreeMap::<usize, BTreeSet<usize>>::new();
        for index in &needed {
            let id = NodeId::from_index(*index);
            let plan = match crate::MovementKernelPlan::from_scheduled_graph(graph, id) {
                Ok(plan) => plan,
                Err(crate::MovementPlanError::NotMovement) => continue,
                Err(error) => return Err(ScheduleError::Binding(error.to_string())),
            };
            for input in plan.input_operands() {
                if !matches!(graph.op(input.node), Ok(Op::Input { .. } | Op::Constant(_))) {
                    movement_operand_owners
                        .entry(input.node.index())
                        .or_default()
                        .insert(*index);
                }
            }
        }
        let movement_operands = movement_operand_owners
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        let computed_view_sources = needed
            .iter()
            .filter_map(|index| {
                let id = NodeId::from_index(*index);
                matches!(
                    graph.op(id),
                    Ok(Op::Shrink { .. }
                        | Op::Reshape { .. }
                        | Op::Permute { .. }
                        | Op::Expand { .. }
                        | Op::Stride { .. })
                )
                .then(|| {
                    crate::rangeify::computed_view(graph, id)
                        .map(|view| view.source)
                        .or_else(|_| {
                            let shape = graph
                                .shape(id)
                                .map_err(|_| crate::rangeify::RangeifyError::Invalid)?;
                            crate::rangeify::projected_source(graph, id, shape)
                        })
                        .ok()
                })
                .flatten()
                .map(NodeId::index)
            })
            .collect::<BTreeSet<_>>();
        let external_view_aliases = needed
            .iter()
            .filter(|index| !requested.contains(index))
            .filter(|index| {
                !direct_payload_operands.contains(index) && !movement_operands.contains(index)
            })
            .filter_map(|index| {
                let id = NodeId::from_index(*index);
                crate::rangeify::computed_view(graph, id)
                    .map(|view| view.source)
                    .or_else(|_| {
                        let shape = graph
                            .shape(id)
                            .map_err(|_| crate::rangeify::RangeifyError::Invalid)?;
                        crate::rangeify::projected_source(graph, id, shape)
                    })
                    .ok()
                    .filter(|source| external.contains(&source.index()))
                    .map(|_| *index)
            })
            .collect::<BTreeSet<_>>();
        requested_passthroughs.retain(|passthrough| {
            let id = passthrough.requested.index();
            !direct_payload_operands.contains(&id) && !movement_operands.contains(&id)
        });
        requested_passthrough_ids = requested_passthroughs
            .iter()
            .map(|passthrough| passthrough.requested.index())
            .collect();
        let requested_passthrough_sources = requested_passthroughs
            .iter()
            .map(|passthrough| passthrough.source.index())
            .collect::<BTreeSet<_>>();
        let roots = needed
            .iter()
            .copied()
            .filter(|index| {
                let id = NodeId::from_index(*index);
                !external.contains(index)
                    && !external_view_aliases.contains(index)
                    && !requested_passthrough_ids.contains(index)
                    && !matches!(graph.op(id), Ok(Op::Input { .. } | Op::Constant(_)))
                    && !matches!(
                        graph.op(id),
                        Ok(Op::Sort {
                            output: crate::SortOutput::Indices,
                            ..
                        })
                    )
                    && (requested.contains(index)
                        || direct_payload_operands.contains(index)
                        || movement_operands.contains(index)
                        || computed_view_sources.contains(index)
                        || (consumers[*index] > 1
                            && !matches!(graph.op(id), Ok(Op::Input { .. } | Op::Constant(_))))
                        || matches!(
                            graph.op(id),
                            Ok(Op::Random { .. }
                                | Op::ShapeIota { .. }
                                | Op::Threefry { .. }
                                | Op::Reduce { .. }
                                | Op::PrefixScan { .. }
                                | Op::Sort { .. }
                                | Op::Matmul { .. }
                                | Op::Conv2d { .. }
                                | Op::Bitcast { .. }
                                | Op::Contiguous { .. }
                                | Op::Pad { .. }
                                | Op::Concat { .. }
                                | Op::Gather { .. }
                                | Op::Scatter { .. }
                                | Op::ScatterPositions { .. }
                                | Op::ScatterPositionsVjp { .. })
                        )
                        || !matches!(graph.op(id), Ok(op) if supported(op)))
            })
            .collect();
        Ok(Self {
            outputs,
            external,
            consumers,
            requested,
            requested_passthroughs,
            direct_payload_operands,
            movement_operand_owners,
            movement_operands,
            requested_passthrough_sources,
            roots,
        })
    }

    pub(super) fn requested_ownership(
        &self,
        graph: &Graph,
    ) -> Result<RequestedScheduleOwnership, ScheduleError> {
        let mut scheduled_candidates = self
            .roots
            .iter()
            .copied()
            .map(NodeId::from_index)
            .collect::<BTreeSet<_>>();
        for root in &self.roots {
            if let Some(sibling) = sort_sibling(graph, NodeId::from_index(*root))? {
                scheduled_candidates.insert(sibling);
            }
        }
        let scheduled = scheduled_candidates
            .into_iter()
            .filter(|node| self.requested.contains(&node.index()))
            .collect();
        let passthroughs = self
            .requested_passthroughs
            .iter()
            .map(|passthrough| passthrough.requested)
            .collect::<BTreeSet<_>>();
        Ok(RequestedScheduleOwnership {
            scheduled,
            passthroughs,
        })
    }
}

pub(super) fn sort_sibling(graph: &Graph, id: NodeId) -> Result<Option<NodeId>, ScheduleError> {
    let Op::Sort { pair, output, .. } = graph.op(id).map_err(ScheduleError::Graph)? else {
        return Ok(None);
    };
    let want = match output {
        crate::SortOutput::Values => crate::SortOutput::Indices,
        crate::SortOutput::Indices => crate::SortOutput::Values,
    };
    (0..graph.node_count())
        .map(NodeId::from_index)
        .find(|candidate| {
            matches!(
                graph.op(*candidate),
                Ok(Op::Sort { pair: candidate_pair, output: candidate_output, .. })
                    if candidate_pair == pair && *candidate_output == want
            )
        })
        .map(Some)
        .ok_or_else(|| ScheduleError::Binding("sort pair sibling is absent".into()))
}

/// Graph operands whose scheduled payload ABI names the exact dense NodeId.
fn op_direct_payload_operands(graph: &Graph, op: &Op) -> Result<Vec<NodeId>, ScheduleError> {
    let operands = match op {
        Op::Matmul { lhs, rhs } => vec![*lhs, *rhs],
        Op::PrefixScan { input, .. } | Op::Sort { input, .. } | Op::TensorGuard { input, .. } => {
            vec![*input]
        }
        Op::Threefry { counter, key } => vec![*counter, *key],
        Op::Conv2d {
            input,
            weight,
            bias,
            ..
        } => [Some(*input), Some(*weight), *bias]
            .into_iter()
            .flatten()
            .collect(),
        _ => Vec::new(),
    };
    operands
        .into_iter()
        .map(|node| {
            graph
                .contiguous_backward_owner(node)
                .map_err(ScheduleError::Graph)
        })
        .collect()
}

pub(super) fn mark_needed(
    graph: &Graph,
    output: NodeId,
    needed: &mut BTreeSet<usize>,
    consumers: &mut [usize],
    external: &BTreeSet<usize>,
) -> Result<(), ScheduleError> {
    enum Frame {
        Node(NodeId),
        Edge(NodeId),
    }

    let mut stack = vec![Frame::Node(output)];
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Node(node) => {
                if !needed.insert(node.index()) || external.contains(&node.index()) {
                    continue;
                }
                let op = graph.op(node).map_err(ScheduleError::Graph)?;
                let children = if supported(op) {
                    op.value_inputs()
                } else {
                    Vec::new()
                };
                stack.extend(children.into_iter().rev().map(Frame::Edge));
            }
            Frame::Edge(child) => {
                let child = graph
                    .contiguous_backward_owner(child)
                    .map_err(ScheduleError::Graph)?;
                consumers[child.index()] += 1;
                stack.push(Frame::Node(child));
            }
        }
    }
    Ok(())
}
