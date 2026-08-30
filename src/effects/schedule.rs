//! Compiler-visible STORE/AFTER form for graph-adjacent effects.
use super::{BufferState, EffectCommit, EffectError, EffectGraph};
use std::{
    collections::{BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

/// Typed immutable effect-assignment payload. `snapshot` is the target state
/// read before writing, so overlap cannot silently become write-after-read.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectPayload {
    pub step: u64,
    pub target: BufferState,
    pub source: BufferState,
    pub snapshot: BufferState,
    pub target_view: Option<crate::AffineView>,
    pub index_plan: Option<crate::ir::indexing::StaticIndexPlan>,
}

/// One validated state transition in an effect schedule.
///
/// The logical assignment payload is stored once. Its compiler-visible
/// `EffectStore` and `After` operations are derived together, so callers
/// cannot pair the wrong payload, predecessor list, or UOp kind.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectScheduleNode {
    payload: EffectPayload,
    predecessors: Vec<u64>,
}

impl EffectScheduleNode {
    /// Creates one descriptor-valid assignment node. Predecessor existence and
    /// source order are schedule-level invariants checked by
    /// [`EffectSchedule::validate`].
    pub fn assignment(payload: EffectPayload, predecessors: Vec<u64>) -> Result<Self, EffectError> {
        super::validate_effect_payload(&payload)?;
        Ok(Self {
            payload,
            predecessors,
        })
    }

    pub fn payload(&self) -> &EffectPayload {
        &self.payload
    }

    pub fn predecessors(&self) -> &[u64] {
        &self.predecessors
    }

    /// Synthesizes the immutable STORE source for this assignment.
    pub fn store_uop(&self) -> crate::UOp {
        crate::UOp::from_operation(
            crate::Operation::EffectStore(Box::new(self.payload.clone())),
            None,
            vec![],
        )
    }

    /// Synthesizes the ordered AFTER root and its matching STORE source.
    pub fn after_uop(&self) -> crate::UOp {
        crate::UOp::from_operation(
            crate::Operation::After(Box::new(self.payload.clone())),
            None,
            vec![self.store_uop()],
        )
    }

    fn validate(&self) -> Result<(), EffectError> {
        super::validate_effect_payload(&self.payload)?;
        self.after_uop()
            .validate()
            .map_err(|_| EffectError::EffectCycle)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectSchedule {
    nodes: Vec<EffectScheduleNode>,
    pub cache_key: u64,
}

impl EffectSchedule {
    pub fn lower(graph: &EffectGraph) -> Result<Self, EffectError> {
        let plan = graph.plan();
        plan.validate()?;
        let mut nodes = Vec::with_capacity(plan.steps.len());
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
                target_view: step.target_view.clone(),
                index_plan: step.index_plan.clone(),
            };
            nodes.push(EffectScheduleNode::assignment(payload, step.after.clone())?);
        }
        let mut schedule = Self {
            nodes,
            cache_key: 0,
        };
        schedule.validate()?;
        schedule.cache_key = schedule_key(&schedule);
        Ok(schedule)
    }

    pub fn nodes(&self) -> &[EffectScheduleNode] {
        &self.nodes
    }

    pub fn validate(&self) -> Result<(), EffectError> {
        let mut completed = BTreeSet::new();
        for node in &self.nodes {
            node.validate()?;
            let payload = node.payload();
            if !completed.insert(payload.step) {
                return Err(EffectError::DuplicateStep(payload.step));
            }
            for predecessor in node.predecessors() {
                if *predecessor >= payload.step || !completed.contains(predecessor) {
                    return Err(EffectError::MissingAfter {
                        step: payload.step,
                        after: *predecessor,
                    });
                }
            }
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
            && self.nodes.iter().any(|node| node.payload.step == step)
        {
            return Err(EffectError::TransactionFailed { step });
        }
        graph.execute()
    }
}
fn schedule_key(schedule: &EffectSchedule) -> u64 {
    let mut h = DefaultHasher::new();
    schedule.nodes.hash(&mut h);
    h.finish()
}
