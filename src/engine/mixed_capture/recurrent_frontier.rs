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
        validate(capture.as_ref(), true)?;
        let capture_identity = identity(capture.as_ref())?;
        Self::from_admitted_capture(capture, capture_identity)
    }

    /// Retains decoder admission without exposing a mutable capture between
    /// validation and immutable frontier ownership. Legacy migration and all
    /// envelope checks remain owned by the canonical decoder.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        let capture = CapturedMixedSchedule::from_bytes(bytes)?;
        let capture_identity = capture.schedule.identity;
        Self::from_admitted_capture(Arc::new(capture), capture_identity)
    }

    // Both callers establish canonical structural admission and identity before
    // reaching this private constructor; no unchecked caller-owned path exists.
    fn from_admitted_capture(
        capture: Arc<CapturedMixedSchedule>,
        capture_identity: u64,
    ) -> Result<Self, ReplayError> {
        #[cfg(test)]
        record_prepared_replay_validation(|counts| {
            counts.recurrent_frontier_authentications += 1;
        });
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

    pub(super) fn validate_cursor(&self, cursor: &MixedReplayCursor) -> Result<(), ReplayError> {
        if cursor.capture_identity != self.capture_identity {
            return Err(ReplayError::Descriptor(
                "recurrent cursor belongs to a different mixed capture".into(),
            ));
        }
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
        Ok(())
    }

    pub(crate) fn resume_cursor(
        &self,
        frontier: impl IntoIterator<Item = BufferState>,
    ) -> Result<MixedReplayCursor, ReplayError> {
        let cursor = MixedReplayCursor {
            capture_identity: self.capture_identity,
            frontier: canonical_frontier(frontier)?,
        };
        self.validate_cursor(&cursor)?;
        Ok(cursor)
    }

    pub(crate) fn preflight_native(
        &self,
        runtime: &crate::EffectRuntime,
        cursor: &MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        vectorized: bool,
    ) -> Result<RecurrentNativePreparation, ReplayError> {
        self.preflight_native_impl(runtime, cursor, provided, vectorized, false)
    }

    pub(crate) fn preflight_native_retaining_unchanged(
        &self,
        runtime: &crate::EffectRuntime,
        cursor: &MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        vectorized: bool,
    ) -> Result<RecurrentNativePreparation, ReplayError> {
        self.preflight_native_impl(runtime, cursor, provided, vectorized, true)
    }

    fn preflight_native_impl(
        &self,
        runtime: &crate::EffectRuntime,
        cursor: &MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        vectorized: bool,
        retain_unchanged: bool,
    ) -> Result<RecurrentNativePreparation, ReplayError> {
        self.validate_cursor(cursor)?;
        let capture = self.capture();
        let starts = recurrent_rebase_starts(capture, cursor)?;
        let replacements = PreparedRecurrentReplacementPlan::from_authenticated(self)?;
        let candidates = recurrent_preflight_candidates(runtime, cursor)?;
        let bound = BoundMixedCapture::bind_authenticated(self, &candidates, starts, provided)?;
        capture.finish_recurrent_native_preflight(
            bound.inputs,
            replacements,
            || self.native_replay_trace(vectorized),
            retain_unchanged,
        )
    }

    fn native_replay_trace(&self, vectorized: bool) -> Result<NativeMixedReplayTrace, ReplayError> {
        self.capture
            .native_replay_trace_from_identity(self.capture_identity, vectorized)
    }

    #[cfg(test)]
    pub(crate) fn capture_allocation_identity(&self) -> usize {
        Arc::as_ptr(&self.capture) as usize
    }
}
