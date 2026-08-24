//! Compiler-visible STORE/AFTER form for graph-adjacent effects.
use super::{BufferState, EffectCommit, EffectError, EffectGraph};
use std::{
    collections::{BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EffectUOpKind {
    Store,
    After,
}

/// Typed immutable STORE payload. `snapshot` is the target state read before
/// writing, so overlap cannot silently become write-after-read.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectPayload {
    pub step: u64,
    pub target: BufferState,
    pub source: BufferState,
    pub snapshot: BufferState,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectUOp {
    pub kind: EffectUOpKind,
    pub payload: EffectPayload,
    pub after: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectSchedule {
    pub uops: Vec<EffectUOp>,
    pub cache_key: u64,
}

impl EffectSchedule {
    pub fn lower(graph: &EffectGraph) -> Result<Self, EffectError> {
        let plan = graph.plan();
        plan.validate()?;
        let mut uops = Vec::with_capacity(plan.steps.len() * 2);
        for step in &plan.steps {
            let snapshot = step
                .reads
                .first()
                .cloned()
                .ok_or(EffectError::UseBeforeState {
                    step: step.id,
                    buffer: step.write.buffer,
                    version: step.write.version,
                })?;
            let source = step
                .reads
                .get(1)
                .cloned()
                .ok_or(EffectError::UseBeforeState {
                    step: step.id,
                    buffer: step.write.buffer,
                    version: step.write.version,
                })?;
            let payload = EffectPayload {
                step: step.id,
                target: step.write.clone(),
                source,
                snapshot,
            };
            uops.push(EffectUOp {
                kind: EffectUOpKind::Store,
                payload: payload.clone(),
                after: vec![],
            });
            uops.push(EffectUOp {
                kind: EffectUOpKind::After,
                payload,
                after: step.after.clone(),
            });
        }
        let mut schedule = Self { uops, cache_key: 0 };
        schedule.validate()?;
        schedule.cache_key = schedule_key(&schedule);
        Ok(schedule)
    }
    pub fn validate(&self) -> Result<(), EffectError> {
        let mut stores = BTreeSet::new();
        let mut after = BTreeSet::new();
        for pair in self.uops.chunks_exact(2) {
            let (store, next) = (&pair[0], &pair[1]);
            if store.kind != EffectUOpKind::Store
                || next.kind != EffectUOpKind::After
                || store.payload != next.payload
                || !store.after.is_empty()
            {
                return Err(EffectError::EffectCycle);
            }
            if !stores.insert(store.payload.step) || !after.insert(next.payload.step) {
                return Err(EffectError::DuplicateStep(store.payload.step));
            }
            if store.payload.target.version
                != store
                    .payload
                    .snapshot
                    .version
                    .checked_add(1)
                    .ok_or(EffectError::Overflow)?
                || store.payload.target.buffer != store.payload.snapshot.buffer
            {
                return Err(EffectError::InvalidVersion {
                    buffer: store.payload.target.buffer,
                    previous: store.payload.snapshot.version,
                    next: store.payload.target.version,
                });
            }
            if store.payload.target.dtype != store.payload.source.dtype {
                return Err(EffectError::DescriptorMismatch {
                    buffer: store.payload.target.buffer,
                    version: store.payload.target.version,
                });
            }
            for predecessor in &next.after {
                if *predecessor >= store.payload.step || !after.contains(predecessor) {
                    return Err(EffectError::MissingAfter {
                        step: store.payload.step,
                        after: *predecessor,
                    });
                }
            }
        }
        if self.uops.len() % 2 != 0 {
            return Err(EffectError::EffectCycle);
        }
        Ok(())
    }
    /// Preflights the typed schedule, then delegates staging/commit to its
    /// source graph. The deterministic hook models a failed native stage.
    pub fn execute(
        &self,
        graph: &EffectGraph,
        fail_at: Option<u64>,
    ) -> Result<EffectCommit, EffectError> {
        self.validate()?;
        if let Some(step) = fail_at
            && self
                .uops
                .iter()
                .any(|u| u.kind == EffectUOpKind::Store && u.payload.step == step)
        {
            return Err(EffectError::TransactionFailed { step });
        }
        graph.execute()
    }
}
fn schedule_key(schedule: &EffectSchedule) -> u64 {
    let mut h = DefaultHasher::new();
    schedule.uops.hash(&mut h);
    h.finish()
}
