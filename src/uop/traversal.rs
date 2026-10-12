use super::{TopologicalOrder, UOp, UOpError, UOpNode};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, btree_map::Entry};
use std::sync::{Arc, Weak};

/// Child IDs make equality shallow while preserving every field of UOpNode.
/// The representative retains payload ownership without copying large plans.
struct NodeKey {
    representative: UOp,
    sources: Vec<usize>,
}

impl Ord for NodeKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.representative
            .operation()
            .cmp(other.representative.operation())
            .then_with(|| self.representative.ty().cmp(&other.representative.ty()))
            .then_with(|| self.sources.cmp(&other.sources))
            .then_with(|| self.representative.tag().cmp(&other.representative.tag()))
    }
}

impl PartialOrd for NodeKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for NodeKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for NodeKey {}

enum Frame {
    Enter(UOp),
    Finish(UOp),
}

/// Equality-only identities for one rewrite invocation, not allocation IDs or
/// semantic admission. They never select a replacement UOp representative.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct RewriteNodeId(usize);

/// Dynamic rewrite results can outlive the original input DAG. Weak tokens
/// pin every observed allocation's control block, while canonical keys retain
/// one representative per complete structure. Both owners drop with the call.
#[derive(Default)]
pub(super) struct RewriteNodeIds {
    allocations: HashMap<usize, (Weak<UOpNode>, RewriteNodeId)>,
    canonical: BTreeMap<NodeKey, RewriteNodeId>,
}

impl RewriteNodeIds {
    fn cached(&self, node: &UOp) -> Option<RewriteNodeId> {
        let (allocation, id) = self.allocations.get(&node.node_identity())?;
        allocation
            .upgrade()
            .filter(|owner| Arc::ptr_eq(owner, &node.0))
            .map(|_| *id)
    }

    pub(super) fn id(&mut self, root: &UOp) -> Result<RewriteNodeId, UOpError> {
        if let Some(id) = self.cached(root) {
            return Ok(id);
        }
        let mut pending = vec![Frame::Enter(root.clone())];
        let mut active = BTreeSet::new();
        while let Some(frame) = pending.pop() {
            match frame {
                Frame::Enter(node) => {
                    if self.cached(&node).is_some() {
                        continue;
                    }
                    if !active.insert(node.node_identity()) {
                        return Err(UOpError::Cycle);
                    }
                    pending.push(Frame::Finish(node.clone()));
                    pending.extend(node.sources().iter().rev().cloned().map(Frame::Enter));
                }
                Frame::Finish(node) => {
                    let sources = node
                        .sources()
                        .iter()
                        .map(|source| {
                            self.cached(source)
                                .map(|id| id.0)
                                .ok_or(UOpError::UseBeforeDefinition)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let next = RewriteNodeId(self.canonical.len());
                    let id = *self
                        .canonical
                        .entry(NodeKey {
                            representative: node.clone(),
                            sources,
                        })
                        .or_insert(next);
                    self.allocations
                        .insert(node.node_identity(), (Arc::downgrade(&node.0), id));
                    active.remove(&node.node_identity());
                }
            }
        }
        self.cached(root).ok_or(UOpError::UseBeforeDefinition)
    }

    pub(super) fn same_sources(&mut self, lhs: &[UOp], rhs: &[UOp]) -> Result<bool, UOpError> {
        if lhs.len() != rhs.len() {
            return Ok(false);
        }
        for (lhs, rhs) in lhs.iter().zip(rhs) {
            if self.id(lhs)? != self.id(rhs)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Encoding-only adjacency for the canonical first representatives.
/// The temporary allocation memo is discarded before this owner is returned.
pub(crate) struct CanonicalNodeTable {
    order: TopologicalOrder,
    offsets: Vec<usize>,
    source_ids: Vec<usize>,
}

impl CanonicalNodeTable {
    pub(crate) fn new(root: &UOp) -> Result<Self, UOpError> {
        let mut offsets = vec![0];
        let mut source_ids = Vec::new();
        let nodes = with_sources(root, |sources| {
            source_ids.extend_from_slice(sources);
            offsets.push(source_ids.len());
        })?;
        Ok(Self {
            order: TopologicalOrder {
                root: root.clone(),
                nodes,
            },
            offsets,
            source_ids,
        })
    }

    pub(crate) fn order(&self) -> &TopologicalOrder {
        &self.order
    }

    pub(crate) fn into_order(self) -> TopologicalOrder {
        self.order
    }

    pub(crate) fn sources(&self, node: usize) -> Option<&[usize]> {
        let start = *self.offsets.get(node)?;
        let end = *self.offsets.get(node.checked_add(1)?)?;
        self.source_ids.get(start..end)
    }

    #[cfg(test)]
    pub(crate) fn from_test_parts(
        order: TopologicalOrder,
        offsets: Vec<usize>,
        source_ids: Vec<usize>,
    ) -> Self {
        Self {
            order,
            offsets,
            source_ids,
        }
    }
}

pub(super) fn topological(root: &UOp) -> Result<Vec<UOp>, UOpError> {
    with_sources(root, |_| {})
}

fn with_sources(
    root: &UOp,
    mut publish_sources: impl FnMut(&[usize]),
) -> Result<Vec<UOp>, UOpError> {
    let mut pending = vec![Frame::Enter(root.clone())];
    let mut active = BTreeSet::new();
    let mut allocations = HashMap::new();
    let mut canonical = BTreeMap::new();
    let mut out = Vec::new();
    // The borrowed root retains the complete immutable allocation graph for
    // this call. Addresses cannot be reused and never select output ordering
    // or structural equivalence; only the canonical key below does that.
    while let Some(frame) = pending.pop() {
        match frame {
            Frame::Enter(node) => {
                let identity = node.node_identity();
                if allocations.contains_key(&identity) {
                    continue;
                }
                if !active.insert(identity) {
                    return Err(UOpError::Cycle);
                }
                pending.push(Frame::Finish(node.clone()));
                pending.extend(node.sources().iter().rev().cloned().map(Frame::Enter));
            }
            Frame::Finish(node) => {
                let sources = node
                    .sources()
                    .iter()
                    .map(|source| {
                        allocations
                            .get(&source.node_identity())
                            .copied()
                            .ok_or(UOpError::UseBeforeDefinition)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let key = NodeKey {
                    representative: node.clone(),
                    sources,
                };
                let id = match canonical.entry(key) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let id = out.len();
                        publish_sources(&entry.key().sources);
                        entry.insert(id);
                        out.push(node.clone());
                        id
                    }
                };
                let identity = node.node_identity();
                allocations.insert(identity, id);
                active.remove(&identity);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "traversal_tests.rs"]
mod tests;
