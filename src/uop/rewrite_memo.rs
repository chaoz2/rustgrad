//! Completed results for one rewrite invocation, with structural fallback.

use super::UOp;
use std::collections::{BTreeMap, HashMap, btree_map::Entry};

struct CompletedRewrite {
    input: UOp,
    output: UOp,
}

/// Allocation lookups accelerate only keys already retained by the structural
/// memo. Equal-but-distinct lookup inputs are deliberately never retained.
/// Addresses cannot recycle while their exact input owners remain in this call.
#[derive(Default)]
pub(super) struct RewriteMemo {
    structural: BTreeMap<UOp, UOp>,
    allocations: HashMap<usize, CompletedRewrite>,
}

impl RewriteMemo {
    pub(super) fn get(&self, input: &UOp) -> Option<UOp> {
        #[cfg(test)]
        record_lookup(|counts| counts.queries += 1);
        if let Some(completed) = self.allocations.get(&input.node_identity()) {
            debug_assert!(completed.input.shares_node_with(input));
            #[cfg(test)]
            record_lookup(|counts| counts.allocation_hits += 1);
            return Some(completed.output.clone());
        }
        #[cfg(test)]
        record_lookup(|counts| counts.structural_queries += 1);
        // Keep the released structural semantics and alias lifetimes. Do not
        // turn a structural hit into another allocation-owned memo entry.
        self.structural.get(input).cloned()
    }

    pub(super) fn insert(&mut self, input: UOp, output: UOp) {
        let retained_input = match self.structural.entry(input) {
            Entry::Vacant(entry) => {
                let input = entry.key().clone();
                entry.insert(output.clone());
                input
            }
            Entry::Occupied(mut entry) => {
                // BTreeMap keeps its original key on replacement. Couple the
                // fast result to that same allocation, not the incoming alias.
                let input = entry.key().clone();
                entry.insert(output.clone());
                input
            }
        };
        self.allocations.insert(
            retained_input.node_identity(),
            CompletedRewrite {
                input: retained_input,
                output,
            },
        );
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LookupCounts {
    queries: usize,
    allocation_hits: usize,
    structural_queries: usize,
}

#[cfg(test)]
std::thread_local! {
    static LOOKUPS: std::cell::Cell<Option<LookupCounts>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn record_lookup(update: impl FnOnce(&mut LookupCounts)) {
    LOOKUPS.with(|counts| {
        if let Some(mut value) = counts.get() {
            update(&mut value);
            counts.set(Some(value));
        }
    });
}

#[cfg(test)]
fn with_lookup_counts<T>(run: impl FnOnce() -> T) -> (T, LookupCounts) {
    struct Reset(Option<LookupCounts>);
    impl Drop for Reset {
        fn drop(&mut self) {
            LOOKUPS.with(|counts| counts.set(self.0));
        }
    }
    let _reset = Reset(LOOKUPS.with(|counts| counts.replace(Some(LookupCounts::default()))));
    let result = run();
    (result, LOOKUPS.with(|counts| counts.get().unwrap()))
}

#[cfg(test)]
#[path = "rewrite_memo_tests.rs"]
mod tests;
