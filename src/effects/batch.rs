//! Ordered rebasing of independently constructed effect plans.
//!
//! An [`EffectPlan`] deliberately has a local version-zero namespace. This
//! module maps those immutable plans onto explicit persistent starting states
//! without serializing runtime slots, generations, or values.
use super::{BufferState, EffectError, EffectPlan, EffectStep};
use crate::TensorData;
use std::collections::{BTreeMap, BTreeSet};

/// Identifies one plan-local STORE source override in an ordered batch.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectBatchSource {
    pub entry: usize,
    pub step: u64,
}

/// Identifies one rebased STORE for deterministic failure injection.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectBatchStep {
    pub entry: usize,
    pub step: u64,
}

/// One existing plan and its explicit persistent state namespace.
#[derive(Clone, Debug)]
pub struct EffectBatchEntry {
    /// The plan remains immutable and locally validated.
    pub plan: EffectPlan,
    /// Persistent states corresponding to this plan's local version-zero
    /// buffers. Every non-overridden external read and every target requires
    /// an exact descriptor-preserving entry here.
    pub starts: BTreeMap<u64, BufferState>,
    /// Owned source values supplied by pure execution for plan-local stores.
    pub sources: BTreeMap<u64, TensorData>,
}

/// A deterministic, ordered, all-or-nothing effect transaction.
#[derive(Clone, Debug, Default)]
pub struct EffectBatch {
    pub entries: Vec<EffectBatchEntry>,
}

#[derive(Clone, Debug)]
pub(crate) struct RebasedEffectStep {
    pub id: EffectBatchStep,
    pub step: EffectStep,
    pub source: Option<TensorData>,
}

impl EffectBatch {
    pub fn new(entries: Vec<EffectBatchEntry>) -> Result<Self, EffectError> {
        let batch = Self { entries };
        batch.rebased_steps()?;
        Ok(batch)
    }

    /// Validates and maps every local plan state onto its supplied persistent
    /// state. A buffer previously written by an earlier entry must begin the
    /// later entry at that exact successor, preventing accidental resets.
    pub(crate) fn rebased_steps(&self) -> Result<Vec<RebasedEffectStep>, EffectError> {
        if self.entries.is_empty() {
            return Err(EffectError::EffectCycle);
        }
        let mut produced = BTreeMap::<u64, BufferState>::new();
        let mut output = Vec::new();
        for (entry_index, entry) in self.entries.iter().enumerate() {
            entry.plan.validate()?;
            if entry.plan.steps.is_empty() {
                return Err(EffectError::EffectCycle);
            }
            let mut required = BTreeSet::new();
            for step in &entry.plan.steps {
                required.insert(step.write.buffer);
                for read in &step.reads {
                    if !entry.sources.contains_key(&step.id) || read != &step.reads[1] {
                        required.insert(read.buffer);
                    }
                }
            }
            for buffer in required {
                let start = entry
                    .starts
                    .get(&buffer)
                    .ok_or(EffectError::MissingBuffer(buffer))?;
                super::validate_buffer_state(start)?;
                if let Some(previous) = produced.get(&buffer)
                    && previous != start
                {
                    return Err(EffectError::InvalidVersion {
                        buffer,
                        previous: previous.version,
                        next: start.version,
                    });
                }
            }
            for step in &entry.plan.steps {
                let rebased = rebase_step(step, &entry.starts)?;
                if let Some(source) = entry.sources.get(&step.id)
                    && (source.shape() != &rebased.reads[1].shape
                        || source.dtype() != rebased.reads[1].dtype)
                {
                    return Err(EffectError::DescriptorMismatch {
                        buffer: rebased.reads[1].buffer,
                        version: rebased.reads[1].version,
                    });
                }
                produced.insert(rebased.write.buffer, rebased.write.clone());
                output.push(RebasedEffectStep {
                    id: EffectBatchStep {
                        entry: entry_index,
                        step: step.id,
                    },
                    step: rebased,
                    source: entry.sources.get(&step.id).cloned(),
                });
            }
        }
        Ok(output)
    }
}

fn rebase_step(
    step: &EffectStep,
    starts: &BTreeMap<u64, BufferState>,
) -> Result<EffectStep, EffectError> {
    let rebase = |state: &BufferState| -> Result<BufferState, EffectError> {
        let start = starts
            .get(&state.buffer)
            .ok_or(EffectError::MissingBuffer(state.buffer))?;
        if start.shape != state.shape || start.dtype != state.dtype || start.bytes != state.bytes {
            return Err(EffectError::DescriptorMismatch {
                buffer: state.buffer,
                version: state.version,
            });
        }
        Ok(BufferState {
            buffer: state.buffer,
            version: start
                .version
                .checked_add(state.version)
                .ok_or(EffectError::Overflow)?,
            shape: state.shape.clone(),
            dtype: state.dtype,
            bytes: state.bytes,
        })
    };
    Ok(EffectStep {
        id: step.id,
        reads: step.reads.iter().map(rebase).collect::<Result<_, _>>()?,
        write: rebase(&step.write)?,
        target_view: step.target_view.clone(),
        index_plan: step.index_plan.clone(),
        after: step.after.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EffectGraph, Shape, Storage};

    fn data(value: i32) -> TensorData {
        TensorData::from_storage(Shape::from([1]), Storage::I32(vec![value])).unwrap()
    }

    #[test]
    fn later_entry_cannot_reset_a_produced_buffer_to_version_zero() {
        let mut first = EffectGraph::default();
        let target = first.insert(1, data(1)).unwrap();
        let source = first.insert(2, data(2)).unwrap();
        let next = first.assign(&target, &source).unwrap();
        let mut second = EffectGraph::default();
        let target_two = second.insert(1, data(0)).unwrap();
        let source_two = second.insert(3, data(3)).unwrap();
        second.assign(&target_two, &source_two).unwrap();
        let batch = EffectBatch {
            entries: vec![
                EffectBatchEntry {
                    plan: first.plan(),
                    starts: BTreeMap::from([
                        (1, target.state().clone()),
                        (2, source.state().clone()),
                    ]),
                    sources: BTreeMap::new(),
                },
                EffectBatchEntry {
                    plan: second.plan(),
                    starts: BTreeMap::from([
                        (1, target_two.state().clone()),
                        (3, source_two.state().clone()),
                    ]),
                    sources: BTreeMap::new(),
                },
            ],
        };
        assert!(matches!(
            batch.rebased_steps(),
            Err(EffectError::InvalidVersion { .. })
        ));
        assert_eq!(next.state().version, 1);
    }
}
