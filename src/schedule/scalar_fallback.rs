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

    pub(super) fn load_nodes(
        &mut self,
        output: NodeId,
        projections: &mut crate::rangeify::ProjectedForestCache<'_>,
    ) -> Result<&[usize], ScheduleError> {
        #[cfg(test)]
        record_rehearsal(|counts| counts.queries += 1);
        if rehearsal_reuse_enabled() && self.rehearsals.contains_key(&output.index()) {
            #[cfg(test)]
            record_rehearsal(|counts| counts.cache_hits += 1);
            return Ok(self.rehearsals[&output.index()].load_nodes.as_ref());
        }
        let load_nodes = ordinary_scalar_fallback_loads(
            self.graph,
            output,
            self.roots,
            self.external,
            projections,
        )?
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
    projections: &mut crate::rangeify::ProjectedForestCache<'_>,
) -> Result<BTreeSet<usize>, ScheduleError> {
    if !scalar_alias_output(graph.op(output).map_err(ScheduleError::Graph)?) {
        return Ok(BTreeSet::new());
    }
    #[cfg(test)]
    record_rehearsal(|counts| counts.lowerings += 1);
    let materialized = scalar_alias_materialized(output, roots, external, &BTreeSet::new());
    let kernel = match graph.op(output).map_err(ScheduleError::Graph)? {
        Op::Reduce { .. } => {
            crate::kernel::lower_graph_reduction_with_materialized_and_projections(
                graph,
                output,
                &materialized,
                projections,
            )
        }
        _ => crate::kernel::lower_graph_elementwise_with_materialized_and_projections(
            graph,
            output,
            &materialized,
            projections,
        ),
    }
    .map_err(ScheduleError::UOp)?;
    scalar_fallback_kernel_loads(&kernel)
}

fn scalar_fallback_kernel_loads(kernel: &UOp) -> Result<BTreeSet<usize>, ScheduleError> {
    let order = crate::uop::normalize_kernel_with_order(kernel).map_err(ScheduleError::UOp)?;
    let mut loads = BTreeSet::new();
    for value in order.nodes() {
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
#[path = "scalar_fallback_tests.rs"]
mod tests;
