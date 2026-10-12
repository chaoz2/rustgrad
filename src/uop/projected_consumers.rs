//! Load-only admission for explicit projected and predicated index nodes.
use super::{Operation, UOp, UOpError};
use crate::projected_index::ProjectedIndexPlan;
use std::collections::HashMap;

pub(super) fn validate(nodes: &[UOp]) -> Result<(), UOpError> {
    let mut consumers = HashMap::<UOp, bool>::new();
    for consumer in nodes {
        let is_load = matches!(consumer.operation(), Operation::Load);
        for source in consumer
            .sources()
            .iter()
            .filter(|source| ProjectedIndexPlan::is_projected(source))
        {
            // Structurally equal nodes necessarily have the same projected
            // operation. Unrelated keys cannot affect this conjunction, and
            // hashing them would recursively expand ordinary shared DAGs.
            consumers
                .entry(source.clone())
                .and_modify(|only_loads| *only_loads &= is_load)
                .or_insert(is_load);
        }
    }
    for index in nodes
        .iter()
        .filter(|node| ProjectedIndexPlan::is_projected(node))
    {
        if consumers.get(index) != Some(&true) {
            return Err(UOpError::InvalidIndex);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "projected_consumers_tests.rs"]
mod tests;
