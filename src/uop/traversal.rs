use super::{UOp, UOpError};
use std::collections::{BTreeSet, HashMap, hash_map::Entry};
use std::hash::{BuildHasher, Hash, Hasher, RandomState};

/// Child IDs make equality shallow while preserving every field of UOpNode.
/// The representative retains payload ownership without copying large plans.
struct NodeKey {
    representative: UOp,
    sources: Vec<usize>,
}

impl PartialEq for NodeKey {
    fn eq(&self, other: &Self) -> bool {
        self.representative.operation() == other.representative.operation()
            && self.representative.ty() == other.representative.ty()
            && self.sources == other.sources
            && self.representative.tag() == other.representative.tag()
    }
}

impl Eq for NodeKey {}

impl Hash for NodeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Never hash the representative UOp: its public Hash recursively
        // expands sources, while these call-local child IDs are already
        // canonical and preserve the same structural distinctions.
        self.representative.operation().hash(state);
        self.representative.ty().hash(state);
        self.sources.hash(state);
        self.representative.tag().hash(state);
    }
}

enum Frame {
    Enter(UOp),
    Finish(UOp),
}

pub(super) fn topological(root: &UOp) -> Result<Vec<UOp>, UOpError> {
    topological_with_hasher(root, RandomState::new())
}

fn topological_with_hasher<S: BuildHasher>(root: &UOp, hasher: S) -> Result<Vec<UOp>, UOpError> {
    let mut pending = vec![Frame::Enter(root.clone())];
    let mut active = BTreeSet::new();
    let mut allocations = HashMap::new();
    let mut canonical = HashMap::with_hasher(hasher);
    let mut out = Vec::new();
    // The borrowed root retains the complete immutable allocation graph for
    // this call. Addresses cannot be reused and never select output ordering
    // or structural equivalence; only the canonical key below does that.
    // The interner is never iterated: IDs and first representatives follow
    // the work stack, independently of hash seeds, buckets, or collisions.
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
