//! Immutable scalar-alias discovery for one pre-fusion frontier. Acceptance
//! and exact kernel rehearsals remain in the changing ownership fixed point.
use super::*;
use crate::rangeify::{ProjectionSource, RangeifyError};

/// Holds only terminal source classification, never synthetic address UOps.
/// The bound graph cannot mutate and the full iteration shape is part of the
/// semantic key. Both successes and failures come from the existing projector.
pub(super) struct ProjectionQueries<'a> {
    graph: &'a Graph,
    results: BTreeMap<NodeId, BTreeMap<Shape, Result<ProjectionSource, RangeifyError>>>,
}

impl<'a> ProjectionQueries<'a> {
    pub(super) fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            results: BTreeMap::new(),
        }
    }

    fn classification(
        &mut self,
        node: NodeId,
        shape: &Shape,
    ) -> Result<ProjectionSource, RangeifyError> {
        if let Some(result) = self.results.get(&node).and_then(|shapes| shapes.get(shape)) {
            #[cfg(test)]
            record(|counts| counts.projection_hits += 1);
            return result.clone();
        }
        #[cfg(test)]
        record(|counts| counts.projection_builds += 1);
        let result = crate::rangeify::projection_source(self.graph, node, shape);
        self.results
            .entry(node)
            .or_default()
            .insert(shape.clone(), result.clone());
        result
    }

    pub(super) fn projected_source(
        &mut self,
        node: NodeId,
        shape: &Shape,
    ) -> Result<NodeId, RangeifyError> {
        #[cfg(test)]
        if !reuse_enabled() {
            record(|counts| counts.projection_builds += 1);
            return crate::rangeify::projected_source(self.graph, node, shape);
        }
        self.classification(node, shape)
            .map(ProjectionSource::source)
    }

    pub(super) fn predicated_source(
        &mut self,
        node: NodeId,
        shape: &Shape,
    ) -> Result<NodeId, RangeifyError> {
        #[cfg(test)]
        if !reuse_enabled() {
            record(|counts| counts.projection_builds += 1);
            return crate::rangeify::predicated_source(self.graph, node, shape);
        }
        match self.classification(node, shape)? {
            ProjectionSource::Predicated(source) => Ok(source),
            ProjectionSource::Unconditional(_) => Err(RangeifyError::Unsupported(node)),
        }
    }
}

/// Roots, requested values, and external materializations are immutable for
/// this object's lifetime. Protection/load reservations are intentionally not
/// retained: each fixed-point pass must reevaluate them and lower its trials.
pub(super) struct ScalarAliasAnalysis<'a> {
    graph: &'a Graph,
    roots: &'a BTreeSet<usize>,
    external: &'a BTreeSet<usize>,
    requested: &'a BTreeSet<usize>,
    queries: ProjectionQueries<'a>,
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
            queries: ProjectionQueries::new(graph),
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
            &mut self.queries,
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
    projection_builds: usize,
    projection_hits: usize,
}

#[cfg(test)]
std::thread_local! {
    static REUSE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static COUNTS: std::cell::Cell<Counts> = const { std::cell::Cell::new(Counts {
        fixed_point_passes: 0,
        candidate_collections: 0,
        candidate_hits: 0,
        projection_builds: 0,
        projection_hits: 0,
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
