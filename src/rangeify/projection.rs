//! Schedule-pass-local descriptor reuse for successful concrete projections.
//! The released projector in `view` remains the semantic and sharing oracle.

use super::view::{ProjectionSource, RangeifiedProjection, RangeifyError, projected_view};
use crate::uop::{LiteralValue, Operation, UOp, UOpTag};
use crate::{DType, Graph, NodeId, Shape, UType};
use std::collections::{BTreeMap, HashMap};

// These are retention limits, not projection/admission limits. A full cache
// returns the released result without keeping another descriptor. Units count
// nodes, edges, key dimensions and an entry; they are not physical-memory bytes.
const MAX_RETAINED_FORESTS: usize = 1024;
const MAX_RETAINED_UNITS: usize = 65_536;

#[derive(Clone)]
struct ForestNode {
    operation: Operation,
    ty: Option<UType>,
    tag: Option<UOpTag>,
    children: Box<[usize]>,
}

struct ProjectedForest {
    source: NodeId,
    nodes: Box<[ForestNode]>,
    caller_range: Option<usize>,
    caller_bound: Option<usize>,
    expression: usize,
    predicate: Option<usize>,
}

impl ProjectedForest {
    fn retention_units(&self, shape: &Shape) -> Option<usize> {
        self.nodes
            .iter()
            .try_fold(shape.rank().checked_add(1)?, |units, node| {
                units.checked_add(1)?.checked_add(node.children.len())
            })
    }

    fn extract(projection: &RangeifiedProjection, caller: &UOp) -> Self {
        enum Frame {
            Visit(UOp),
            Finish(UOp),
        }

        let mut identities = HashMap::<usize, usize>::new();
        let mut nodes = Vec::new();
        let roots = projection
            .predicate
            .iter()
            .chain(std::iter::once(&projection.expression));
        for root in roots {
            let mut pending = vec![Frame::Visit(root.clone())];
            while let Some(frame) = pending.pop() {
                match frame {
                    Frame::Visit(node) => {
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
                            .collect::<Vec<_>>()
                            .into_boxed_slice();
                        let index = nodes.len();
                        nodes.push(ForestNode {
                            operation: node.operation().clone(),
                            ty: node.ty(),
                            tag: node.tag().cloned(),
                            children,
                        });
                        identities.insert(node.node_identity(), index);
                    }
                }
            }
        }
        let expression = identities[&projection.expression.node_identity()];
        let predicate = projection
            .predicate
            .as_ref()
            .map(|value| identities[&value.node_identity()]);
        // Equal Range operations are not caller aliases. Only the exact
        // allocation supplied to the released projector may bind this slot.
        let caller_range = identities.get(&caller.node_identity()).copied();
        // Canonical callers have one leaf bound. If that allocation appears
        // separately in either root, retain its relation to the caller too;
        // rebinding only the Range would split the original sharing class.
        let caller_bound = caller
            .sources()
            .first()
            .and_then(|bound| identities.get(&bound.node_identity()).copied());
        Self {
            source: projection.source,
            nodes: nodes.into_boxed_slice(),
            caller_range,
            caller_bound,
            expression,
            predicate,
        }
    }

    fn emit(&self, caller: &UOp) -> RangeifiedProjection {
        let mut emitted = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            if self.caller_range == Some(index) {
                emitted.push(caller.clone());
                continue;
            }
            if self.caller_bound == Some(index) {
                emitted.push(caller.sources()[0].clone());
                continue;
            }
            let sources = node
                .children
                .iter()
                .map(|child| emitted[*child].clone())
                .collect();
            emitted.push(UOp::from_parts(
                node.operation.clone(),
                node.ty,
                sources,
                node.tag.clone(),
            ));
        }
        RangeifiedProjection {
            source: self.source,
            expression: emitted[self.expression].clone(),
            predicate: self.predicate.map(|index| emitted[index].clone()),
        }
    }

    fn classification(&self) -> ProjectionSource {
        if self.predicate.is_some() {
            ProjectionSource::Predicated(self.source)
        } else {
            ProjectionSource::Unconditional(self.source)
        }
    }
}

/// One schedule-pass-local success-only cache. It borrows the immutable graph
/// and owns descriptors, never UOps produced by the released projector.
pub(crate) struct ProjectedForestCache<'a> {
    graph: &'a Graph,
    forests: BTreeMap<NodeId, BTreeMap<Shape, ProjectedForest>>,
    retained_forests: usize,
    retained_units: usize,
    forest_limit: usize,
    unit_limit: usize,
}

impl<'a> ProjectedForestCache<'a> {
    pub(crate) fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            forests: BTreeMap::new(),
            retained_forests: 0,
            retained_units: 0,
            forest_limit: MAX_RETAINED_FORESTS,
            unit_limit: MAX_RETAINED_UNITS,
        }
    }

    fn retain(&mut self, node: NodeId, shape: &Shape, forest: ProjectedForest) {
        let units = forest
            .retention_units(shape)
            .and_then(|units| self.retained_units.checked_add(units));
        if self.retained_forests >= self.forest_limit
            || units.is_none_or(|units| units > self.unit_limit)
        {
            #[cfg(test)]
            record(|counts| counts.retention_bypasses += 1);
            return;
        }
        self.forests
            .entry(node)
            .or_default()
            .insert(shape.clone(), forest);
        self.retained_forests += 1;
        self.retained_units = units.expect("the checked retention budget is present");
    }

    fn canonical_range(shape: &Shape, range: &UOp) -> bool {
        let Ok(extent) = shape.numel() else {
            return false;
        };
        let Ok(extent) = i64::try_from(extent) else {
            return false;
        };
        matches!(range.operation(), Operation::Range(0))
            && range.ty() == Some(UType::scalar(DType::I64))
            && range.tag().is_none()
            && matches!(range.sources(), [bound]
                if matches!(bound.operation(), Operation::Const(LiteralValue::Int(value)) if *value == extent)
                    && bound.ty() == Some(UType::scalar(DType::I64))
                    && bound.tag().is_none()
                    && bound.sources().is_empty())
    }

    fn range(shape: &Shape) -> Result<UOp, RangeifyError> {
        let extent = shape.numel().map_err(|_| RangeifyError::Invalid)?;
        let extent = i64::try_from(extent).map_err(|_| RangeifyError::Invalid)?;
        Ok(UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![UOp::constant(extent, UType::scalar(DType::I64))],
        ))
    }

    pub(crate) fn projection(
        &mut self,
        graph: &Graph,
        node: NodeId,
        shape: &Shape,
        range: &UOp,
    ) -> Result<RangeifiedProjection, RangeifyError> {
        #[cfg(test)]
        record(|counts| counts.queries += 1);
        if !std::ptr::eq(self.graph, graph) {
            #[cfg(test)]
            record(|counts| counts.released_builds += 1);
            return projected_view(graph, node, shape, range);
        }
        #[cfg(test)]
        if !reuse_enabled() {
            record(|counts| counts.released_builds += 1);
            return projected_view(self.graph, node, shape, range);
        }
        if !Self::canonical_range(shape, range) {
            #[cfg(test)]
            record(|counts| counts.released_builds += 1);
            return projected_view(self.graph, node, shape, range);
        }
        if let Some(forest) = self.forests.get(&node).and_then(|shapes| shapes.get(shape)) {
            #[cfg(test)]
            record(|counts| {
                counts.hits += 1;
                counts.emissions += 1;
            });
            return Ok(forest.emit(range));
        }
        #[cfg(test)]
        record(|counts| counts.released_builds += 1);
        let projection = projected_view(self.graph, node, shape, range)?;
        let forest = ProjectedForest::extract(&projection, range);
        #[cfg(test)]
        record(|counts| counts.extractions += 1);
        self.retain(node, shape, forest);
        Ok(projection)
    }

    pub(crate) fn classification(
        &mut self,
        graph: &Graph,
        node: NodeId,
        shape: &Shape,
    ) -> Result<ProjectionSource, RangeifyError> {
        #[cfg(test)]
        record(|counts| counts.source_queries += 1);
        if !std::ptr::eq(self.graph, graph) {
            #[cfg(test)]
            record(|counts| counts.released_builds += 1);
            return super::view::projection_source(graph, node, shape);
        }
        #[cfg(test)]
        if !reuse_enabled() {
            record(|counts| counts.released_builds += 1);
            let range = Self::range(shape)?;
            let projection = projected_view(self.graph, node, shape, &range)?;
            return Ok(if projection.predicate.is_some() {
                ProjectionSource::Predicated(projection.source)
            } else {
                ProjectionSource::Unconditional(projection.source)
            });
        }
        if let Some(forest) = self.forests.get(&node).and_then(|shapes| shapes.get(shape)) {
            #[cfg(test)]
            record(|counts| counts.hits += 1);
            return Ok(forest.classification());
        }
        let range = Self::range(shape)?;
        #[cfg(test)]
        record(|counts| counts.released_builds += 1);
        let projection = projected_view(self.graph, node, shape, &range)?;
        let classification = if projection.predicate.is_some() {
            ProjectionSource::Predicated(projection.source)
        } else {
            ProjectionSource::Unconditional(projection.source)
        };
        let forest = ProjectedForest::extract(&projection, &range);
        #[cfg(test)]
        record(|counts| counts.extractions += 1);
        self.retain(node, shape, forest);
        Ok(classification)
    }

    pub(crate) fn projected_source(
        &mut self,
        graph: &Graph,
        node: NodeId,
        shape: &Shape,
    ) -> Result<NodeId, RangeifyError> {
        self.classification(graph, node, shape)
            .map(ProjectionSource::source)
    }

    pub(crate) fn predicated_source(
        &mut self,
        graph: &Graph,
        node: NodeId,
        shape: &Shape,
    ) -> Result<NodeId, RangeifyError> {
        match self.classification(graph, node, shape)? {
            ProjectionSource::Predicated(source) => Ok(source),
            ProjectionSource::Unconditional(_) => Err(RangeifyError::Unsupported(node)),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProjectedForestCounts {
    pub(crate) queries: usize,
    pub(crate) source_queries: usize,
    pub(crate) hits: usize,
    pub(crate) extractions: usize,
    pub(crate) emissions: usize,
    pub(crate) released_builds: usize,
    pub(crate) retention_bypasses: usize,
}

#[cfg(test)]
std::thread_local! {
    static REUSE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static COUNTS: std::cell::Cell<ProjectedForestCounts> = const {
        std::cell::Cell::new(ProjectedForestCounts {
            queries: 0,
            source_queries: 0,
            hits: 0,
            extractions: 0,
            emissions: 0,
            released_builds: 0,
            retention_bypasses: 0,
        })
    };
}

#[cfg(test)]
fn reuse_enabled() -> bool {
    REUSE.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record(update: impl FnOnce(&mut ProjectedForestCounts)) {
    COUNTS.with(|counts| {
        let mut next = counts.get();
        update(&mut next);
        counts.set(next);
    });
}

#[cfg(test)]
pub(crate) fn with_projected_forest_mode<T>(
    reuse: bool,
    run: impl FnOnce() -> T,
) -> (T, ProjectedForestCounts) {
    struct Restore(bool, ProjectedForestCounts);
    impl Drop for Restore {
        fn drop(&mut self) {
            REUSE.with(|reuse| reuse.set(self.0));
            COUNTS.with(|counts| counts.set(self.1));
        }
    }
    let restore = Restore(
        REUSE.with(|enabled| enabled.replace(reuse)),
        COUNTS.with(|counts| counts.replace(ProjectedForestCounts::default())),
    );
    let value = run();
    let counts = COUNTS.with(std::cell::Cell::get);
    drop(restore);
    (value, counts)
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
