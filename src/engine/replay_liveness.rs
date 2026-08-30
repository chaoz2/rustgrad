//! Conservative reverse-demand analysis for strict-native captured replay.
//!
//! This is intentionally an engine-only plan: artifacts and schedules retain
//! their complete immutable topology, while an execution can omit pure work
//! whose only demanded consumer is an exact zero-domain result.
use super::capture::{CapturedSchedule, ReplayError};
use crate::BufferDesc;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default)]
pub(crate) struct ReplayLivenessPlan {
    pruned: BTreeMap<u64, BufferDesc>,
    materialized_zeros: BTreeSet<u64>,
}

impl ReplayLivenessPlan {
    pub(crate) fn analyze(capture: &CapturedSchedule) -> Result<Self, ReplayError> {
        let mut demanded = BTreeSet::new();
        let requested = capture.requested.iter().copied().collect::<BTreeSet<_>>();
        for item in &capture.items {
            // Requested values, boundaries, and effects are externally
            // observable or have ordering semantics. Keep them as roots even
            // when their output domain is empty.
            if requested.contains(&item.primary_output().id)
                || item.boundary.is_some()
                || item.is_effect()
            {
                demanded.insert(item.id);
            }
        }
        let source_values = capture
            .inputs
            .iter()
            .map(|input| input.desc.id)
            .chain(capture.constants.keys().copied())
            .collect::<BTreeSet<_>>();
        let produced_requested = capture
            .items
            .iter()
            .filter(|item| requested.contains(&item.primary_output().id))
            .count();
        if requested
            .iter()
            .filter(|id| !source_values.contains(id))
            .count()
            != produced_requested
        {
            return Err(ReplayError::Corrupt(
                "requested value has no unique captured owner".into(),
            ));
        }

        let mut pending = demanded.iter().copied().collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            let item = capture.items.get(id as usize).ok_or_else(|| {
                ReplayError::Corrupt("captured liveness dependency is absent".into())
            })?;
            if item.id != id {
                return Err(ReplayError::Corrupt(
                    "captured liveness item IDs are not contiguous".into(),
                ));
            }
            let zero_domain = item
                .output
                .shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?
                == 0;
            if zero_domain && item.boundary.is_none() && !item.is_effect() {
                // Exact typed zeros can be produced without reading any
                // operands. Consequently, pure producers reachable only from
                // this value have no observable demand.
                continue;
            }
            for dependency in &item.dependencies {
                if demanded.insert(*dependency) {
                    pending.push(*dependency);
                }
            }
        }

        let mut plan = Self::default();
        for item in &capture.items {
            let zero_domain = item
                .output
                .shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?
                == 0;
            if demanded.contains(&item.id) {
                if zero_domain && item.boundary.is_none() && !item.is_effect() {
                    plan.materialized_zeros.insert(item.id);
                }
            } else if item.boundary.is_none() && !item.is_effect() {
                plan.pruned.insert(item.id, item.primary_output().clone());
            }
        }
        Ok(plan)
    }

    pub(crate) fn is_pruned(&self, item: u64) -> Option<&BufferDesc> {
        self.pruned.get(&item)
    }

    pub(crate) fn materializes_zero(&self, item: u64) -> bool {
        self.materialized_zeros.contains(&item)
    }

    pub(crate) fn pruned_item_count(&self) -> usize {
        self.pruned.len()
    }

    pub(crate) fn materialized_zero_item_count(&self) -> usize {
        self.materialized_zeros.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_a_unique_requested_producer() {
        let capture = CapturedSchedule {
            items: vec![],
            inputs: vec![],
            constants: Default::default(),
            quantized_constants: Default::default(),
            requested: vec![7],
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        assert!(matches!(
            ReplayLivenessPlan::analyze(&capture),
            Err(ReplayError::Corrupt(_))
        ));
    }

    #[test]
    fn accepts_source_owned_requested_values_without_producers() {
        let capture = CapturedSchedule {
            items: vec![],
            inputs: vec![super::capture::ReplayInput {
                name: "input".into(),
                node: crate::NodeId::from_index(3),
                desc: crate::BufferDesc {
                    id: 3,
                    shape: crate::Shape::from([2]),
                    dtype: crate::DType::F32,
                    bytes: 8,
                    alignment: 4,
                    read_only: true,
                    view: None,
                },
            }],
            constants: std::collections::BTreeMap::from([(
                4,
                crate::TensorData::zeros_with_dtype([2], crate::DType::F32).unwrap(),
            )]),
            quantized_constants: Default::default(),
            requested: vec![3, 4],
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        let plan = ReplayLivenessPlan::analyze(&capture).unwrap();
        assert_eq!(plan.pruned_item_count(), 0);
        assert_eq!(plan.materialized_zero_item_count(), 0);
    }
}
