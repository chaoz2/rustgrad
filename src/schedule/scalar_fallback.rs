use super::*;

/// One ordinary scalar root's exact load inventory under the immutable
/// pre-fusion roots/external frontier. Kernels are deliberately discarded:
/// later alias removal and Contiguous redirection can change the final leaf
/// frontier, while this compact inventory is reused only inside that frontier's
/// fixed-point rehearsal.
struct OrdinaryScalarFallbackRehearsal {
    load_nodes: Box<[usize]>,
}

pub(super) struct OrdinaryScalarFallbackRehearsals<'a> {
    graph: &'a Graph,
    roots: &'a BTreeSet<usize>,
    external: &'a BTreeSet<usize>,
    rehearsals: BTreeMap<usize, OrdinaryScalarFallbackRehearsal>,
}

impl<'a> OrdinaryScalarFallbackRehearsals<'a> {
    pub(super) fn new(
        graph: &'a Graph,
        roots: &'a BTreeSet<usize>,
        external: &'a BTreeSet<usize>,
    ) -> Self {
        Self {
            graph,
            roots,
            external,
            rehearsals: BTreeMap::new(),
        }
    }

    pub(super) fn begin_fixed_point_pass(&self) {
        #[cfg(test)]
        record_rehearsal(|counts| counts.fixed_point_passes += 1);
    }

    pub(super) fn load_nodes(&mut self, output: NodeId) -> Result<&[usize], ScheduleError> {
        #[cfg(test)]
        record_rehearsal(|counts| counts.queries += 1);
        if rehearsal_reuse_enabled() && self.rehearsals.contains_key(&output.index()) {
            #[cfg(test)]
            record_rehearsal(|counts| counts.cache_hits += 1);
            return Ok(self.rehearsals[&output.index()].load_nodes.as_ref());
        }
        let load_nodes =
            ordinary_scalar_fallback_loads(self.graph, output, self.roots, self.external)?
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice();
        self.rehearsals.insert(
            output.index(),
            OrdinaryScalarFallbackRehearsal { load_nodes },
        );
        Ok(self.rehearsals[&output.index()].load_nodes.as_ref())
    }
}

fn ordinary_scalar_fallback_loads(
    graph: &Graph,
    output: NodeId,
    roots: &BTreeSet<usize>,
    external: &BTreeSet<usize>,
) -> Result<BTreeSet<usize>, ScheduleError> {
    if !scalar_alias_output(graph.op(output).map_err(ScheduleError::Graph)?) {
        return Ok(BTreeSet::new());
    }
    #[cfg(test)]
    record_rehearsal(|counts| counts.lowerings += 1);
    let materialized = scalar_alias_materialized(output, roots, external, &BTreeSet::new());
    let kernel = match graph.op(output).map_err(ScheduleError::Graph)? {
        Op::Reduce { .. } => {
            crate::kernel::lower_graph_reduction_with_materialized(graph, output, &materialized)
        }
        _ => crate::kernel::lower_graph_elementwise_with_materialized(graph, output, &materialized),
    }
    .map_err(ScheduleError::UOp)?;
    let kernel = crate::uop::normalize_kernel(&kernel).map_err(ScheduleError::UOp)?;
    let topology = kernel.topological().map_err(ScheduleError::UOp)?;
    let mut loads = BTreeSet::new();
    for value in topology {
        if !matches!(value.operation(), crate::Operation::Load) {
            continue;
        }
        let index = value
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

#[cfg(not(test))]
const fn rehearsal_reuse_enabled() -> bool {
    true
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RehearsalCounts {
    queries: usize,
    lowerings: usize,
    cache_hits: usize,
    fixed_point_passes: usize,
}

#[cfg(test)]
std::thread_local! {
    static REHEARSAL_REUSE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static REHEARSAL_COUNTS: std::cell::Cell<RehearsalCounts> =
        const { std::cell::Cell::new(RehearsalCounts {
            queries: 0,
            lowerings: 0,
            cache_hits: 0,
            fixed_point_passes: 0,
        }) };
}

#[cfg(test)]
fn rehearsal_reuse_enabled() -> bool {
    REHEARSAL_REUSE.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_rehearsal(update: impl FnOnce(&mut RehearsalCounts)) {
    REHEARSAL_COUNTS.with(|counts| {
        let mut next = counts.get();
        update(&mut next);
        counts.set(next);
    });
}

#[cfg(test)]
fn with_rehearsal_mode<T>(reuse: bool, f: impl FnOnce() -> T) -> (T, RehearsalCounts) {
    struct Restore {
        reuse: bool,
        counts: RehearsalCounts,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            REHEARSAL_REUSE.with(|reuse| reuse.set(self.reuse));
            REHEARSAL_COUNTS.with(|counts| counts.set(self.counts));
        }
    }

    let restore = Restore {
        reuse: REHEARSAL_REUSE.with(|enabled| enabled.replace(reuse)),
        counts: REHEARSAL_COUNTS.with(|counts| counts.replace(RehearsalCounts::default())),
    };
    let value = f();
    let counts = REHEARSAL_COUNTS.with(std::cell::Cell::get);
    drop(restore);
    (value, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let (cached, cached_counts) =
            with_rehearsal_mode(true, || schedule_many(&graph, &requested));
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
        assert_eq!(fused.load_nodes(output).unwrap(), &[input.index()]);

        let external = BTreeSet::from([producer.index()]);
        let mut materialized = OrdinaryScalarFallbackRehearsals::new(&graph, &roots, &external);
        assert_eq!(
            materialized.load_nodes(output).unwrap(),
            &[producer.index()]
        );
        assert_eq!(fused.load_nodes(output).unwrap(), &[input.index()]);
    }

    #[test]
    fn failed_queries_preserve_the_error_and_are_not_cached() {
        let graph = Graph::new();
        let roots = BTreeSet::new();
        let external = BTreeSet::new();
        let unknown = NodeId::from_index(7);
        let mut rehearsals = OrdinaryScalarFallbackRehearsals::new(&graph, &roots, &external);

        let first = rehearsals.load_nodes(unknown).unwrap_err();
        let second = rehearsals.load_nodes(unknown).unwrap_err();
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
        assert!(matches!(
            rehearsals.load_nodes(output),
            Err(ScheduleError::UOp(UOpError::InvalidArgument))
        ));
        assert!(matches!(
            rehearsals.load_nodes(output),
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
}
