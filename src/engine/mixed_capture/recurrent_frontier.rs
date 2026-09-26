//! Capture-bound authenticated recurrent frontier metadata.

use super::*;
use std::sync::Arc;

/// Immutable proof of one mixed capture's canonical identity and version-zero
/// recurrent schema. Owning the capture allocation prevents this metadata from
/// being paired with a different program by construction.
pub(crate) struct AuthenticatedRecurrentFrontier {
    // Ownership, rather than a second freely pairable hash, binds this schema
    // to the exact immutable program allocation that was authenticated.
    capture: Arc<CapturedMixedSchedule>,
    capture_identity: u64,
    initial_frontier: Box<[BufferState]>,
}

impl AuthenticatedRecurrentFrontier {
    pub(crate) fn authenticate(capture: Arc<CapturedMixedSchedule>) -> Result<Self, ReplayError> {
        #[cfg(test)]
        record_prepared_replay_validation(|counts| {
            counts.recurrent_frontier_authentications += 1;
        });
        validate(capture.as_ref(), true)?;
        let capture_identity = identity(capture.as_ref())?;
        let initial_frontier = recurrent_initial_frontier(capture.as_ref())?.into_boxed_slice();
        Ok(Self {
            capture,
            capture_identity,
            initial_frontier,
        })
    }

    pub(crate) const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub(crate) fn capture(&self) -> &CapturedMixedSchedule {
        self.capture.as_ref()
    }

    pub(crate) fn capture_arc(&self) -> Arc<CapturedMixedSchedule> {
        self.capture.clone()
    }

    pub(crate) fn initial_frontier(&self) -> &[BufferState] {
        &self.initial_frontier
    }

    pub(crate) fn initial_cursor(&self) -> MixedReplayCursor {
        MixedReplayCursor {
            capture_identity: self.capture_identity,
            frontier: self.initial_frontier.to_vec(),
        }
    }

    pub(crate) fn resume_cursor(
        &self,
        frontier: impl IntoIterator<Item = BufferState>,
    ) -> Result<MixedReplayCursor, ReplayError> {
        let cursor = MixedReplayCursor {
            capture_identity: self.capture_identity,
            frontier: canonical_frontier(frontier)?,
        };
        if cursor.frontier.len() != self.initial_frontier.len() {
            return Err(ReplayError::Descriptor(
                "recurrent cursor state frontier is incomplete".into(),
            ));
        }
        for (actual, initial) in cursor.frontier.iter().zip(self.initial_frontier.iter()) {
            if actual.buffer != initial.buffer
                || actual.version < initial.version
                || actual.shape != initial.shape
                || actual.dtype != initial.dtype
                || actual.bytes != initial.bytes
            {
                return Err(ReplayError::Descriptor(
                    "recurrent cursor state descriptor mismatch".into(),
                ));
            }
        }
        Ok(cursor)
    }

    #[cfg(test)]
    pub(crate) fn capture_allocation_identity(&self) -> usize {
        Arc::as_ptr(&self.capture) as usize
    }
}
