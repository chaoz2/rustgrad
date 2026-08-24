//! Immutable caller-selected persistent-state namespace for mixed replay.
//!
//! RGSM/RGMB artifacts retain their captured logical names.  A rebinding is a
//! replay-local schema only; it never mutates or serializes an artifact and it
//! contains no runtime resource identity or storage bytes.
use crate::ReplayError;
use std::collections::{BTreeMap, BTreeSet};

/// A total, one-to-one mapping from the states named by one capture to caller
/// runtime logical buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedStateRebinding {
    map: BTreeMap<u64, u64>,
}

impl MixedStateRebinding {
    /// Constructs a deterministic mapping.  Completeness is checked against a
    /// concrete capture at replay, while duplicate destinations are rejected
    /// here rather than being silently treated as aliases.
    pub fn new(map: BTreeMap<u64, u64>) -> Result<Self, ReplayError> {
        let mut destinations = BTreeSet::new();
        for &destination in map.values() {
            if !destinations.insert(destination) {
                return Err(ReplayError::Descriptor(
                    "mixed state rebinding has duplicate destination".into(),
                ));
            }
        }
        Ok(Self { map })
    }

    /// Returns the immutable logical mapping, ordered by captured buffer ID.
    pub fn mappings(&self) -> &BTreeMap<u64, u64> {
        &self.map
    }

    pub(crate) fn destination(&self, captured: u64) -> Result<u64, ReplayError> {
        self.map
            .get(&captured)
            .copied()
            .ok_or_else(|| ReplayError::Missing(format!("rebinding for state {captured}")))
    }

    pub(crate) fn mapped(&self, captured: u64) -> Option<u64> {
        self.map.get(&captured).copied()
    }

    pub(crate) fn validate_exact(&self, referenced: &BTreeSet<u64>) -> Result<(), ReplayError> {
        let supplied = self.map.keys().copied().collect::<BTreeSet<_>>();
        if &supplied != referenced {
            return Err(ReplayError::Descriptor(
                "mixed state rebinding must exactly cover captured states".into(),
            ));
        }
        Ok(())
    }

    /// Logical schema key suitable for replay traces.  It deliberately omits
    /// runtime slot/generation/pointer/current-byte information.
    pub(crate) fn schema_key(&self) -> u64 {
        self.map
            .iter()
            .fold(0xcbf29ce484222325u64, |hash, (&from, &to)| {
                hash.wrapping_mul(0x100000001b3)
                    .wrapping_add(from)
                    .wrapping_mul(0x100000001b3)
                    .wrapping_add(to)
            })
    }
}
