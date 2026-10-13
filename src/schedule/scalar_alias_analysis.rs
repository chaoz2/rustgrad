//! Immutable scalar-alias discovery for one pre-fusion frontier. Acceptance
//! and exact kernel rehearsals remain in the changing ownership fixed point.
use super::*;

/// Roots, requested values, and external materializations are immutable for
/// this object's lifetime. Protection/load reservations are intentionally not
/// retained: each fixed-point pass must reevaluate them and lower its trials.
pub(super) struct ScalarAliasAnalysis<'a> {
    graph: &'a Graph,
    roots: &'a BTreeSet<usize>,
    external: &'a BTreeSet<usize>,
    requested: &'a BTreeSet<usize>,
    candidates: BTreeMap<NodeId, ScalarAliasCandidates>,
}

impl<'a> ScalarAliasAnalysis<'a> {
    pub(super) fn new(
        graph: &'a Graph,
        roots: &'a BTreeSet<usize>,
        external: &'a BTreeSet<usize>,
        requested: &'a BTreeSet<usize>,
    ) -> Self {
        Self {
            graph,
            roots,
            external,
            requested,
            candidates: BTreeMap::new(),
        }
    }

    pub(super) fn begin_fixed_point_pass(&self) {
        #[cfg(test)]
        record(|counts| counts.fixed_point_passes += 1);
    }

    pub(super) fn candidates(
        &mut self,
        output: NodeId,
        projections: &mut crate::rangeify::ProjectedForestCache<'a>,
    ) -> Result<Option<&ScalarAliasCandidates>, ScheduleError> {
        if !scalar_alias_output(self.graph.op(output).map_err(ScheduleError::Graph)?) {
            return Ok(None);
        }
        if reuse_enabled() && self.candidates.contains_key(&output) {
            #[cfg(test)]
            record(|counts| counts.candidate_hits += 1);
            return Ok(self.candidates.get(&output));
        }
        #[cfg(test)]
        record(|counts| counts.candidate_collections += 1);
        let candidates = ScalarAliasCandidates::collect(
            self.graph,
            output,
            self.roots,
            self.external,
            self.requested,
            projections,
        )?;
        self.candidates.insert(output, candidates);
        Ok(self.candidates.get(&output))
    }
}

#[cfg(not(test))]
const fn reuse_enabled() -> bool {
    true
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Counts {
    fixed_point_passes: usize,
    candidate_collections: usize,
    candidate_hits: usize,
}

#[cfg(test)]
std::thread_local! {
    static REUSE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static COUNTS: std::cell::Cell<Counts> = const { std::cell::Cell::new(Counts {
        fixed_point_passes: 0,
        candidate_collections: 0,
        candidate_hits: 0,
    }) };
}

#[cfg(test)]
fn reuse_enabled() -> bool {
    REUSE.get()
}

#[cfg(test)]
fn record(update: impl FnOnce(&mut Counts)) {
    let mut counts = COUNTS.get();
    update(&mut counts);
    COUNTS.set(counts);
}

#[cfg(test)]
fn with_mode<T>(reuse: bool, run: impl FnOnce() -> T) -> (T, Counts) {
    struct Restore(bool, Counts);
    impl Drop for Restore {
        fn drop(&mut self) {
            REUSE.set(self.0);
            COUNTS.set(self.1);
        }
    }
    let restore = Restore(REUSE.replace(reuse), COUNTS.replace(Counts::default()));
    let result = run();
    let counts = COUNTS.get();
    drop(restore);
    (result, counts)
}

#[cfg(test)]
#[path = "scalar_alias_analysis_tests.rs"]
mod tests;
