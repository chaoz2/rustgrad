//! In-memory batch schema for ordered mixed captures.
//!
//! This module intentionally reuses RGSM and [`crate::EffectBatch`]; it owns
//! no serialization format and never records runtime resource identities.
use super::mixed_capture::CapturedMixedSchedule;
use crate::ReplayError;

/// Ordered, immutable logical batch of decoded mixed captures.
#[derive(Clone, Debug)]
pub struct CapturedMixedBatch {
    captures: Vec<CapturedMixedSchedule>,
    identity: u64,
}

impl CapturedMixedBatch {
    /// Validates every constituent RGSM envelope and assigns a stable identity
    /// over its ordered logical bytes. Runtime slots, generations, pointers,
    /// and current storage never participate.
    pub fn new(captures: Vec<CapturedMixedSchedule>) -> Result<Self, ReplayError> {
        if captures.is_empty() {
            return Err(ReplayError::Corrupt("empty mixed batch".into()));
        }
        let mut hash = 0xcbf29ce484222325u64;
        for capture in &captures {
            let bytes = capture.to_bytes()?;
            for byte in bytes {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        Ok(Self {
            captures,
            identity: hash,
        })
    }

    pub fn captures(&self) -> &[CapturedMixedSchedule] {
        &self.captures
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }
}
