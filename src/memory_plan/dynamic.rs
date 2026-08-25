//! Lifetime metadata for exact runtime-sized schedule allocations.
//!
//! Fixed `MemoryPlan` records continue to model only concrete `BufferDesc`
//! values. This compact companion records an unknown-shape buffer's ordering
//! facts without pretending its byte size is known before its count stage.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RuntimeAllocationLifetime {
    pub buffer_id: u64,
    pub allocation_item: u64,
    pub final_consumer: u64,
    pub identity: u64,
}

impl RuntimeAllocationLifetime {
    pub(crate) fn new(buffer_id: u64, allocation_item: u64, final_consumer: u64) -> Self {
        let mut value = Self {
            buffer_id,
            allocation_item,
            final_consumer,
            identity: 0,
        };
        let mut hasher = DefaultHasher::new();
        value.buffer_id.hash(&mut hasher);
        value.allocation_item.hash(&mut hasher);
        value.final_consumer.hash(&mut hasher);
        value.identity = hasher.finish();
        value
    }

    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.final_consumer < self.allocation_item {
            return Err("runtime allocation final consumer precedes allocation");
        }
        let expected = Self::new(self.buffer_id, self.allocation_item, self.final_consumer);
        if self.identity != expected.identity {
            return Err("runtime allocation lifetime identity mismatch");
        }
        Ok(())
    }
}
