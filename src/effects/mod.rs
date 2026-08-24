//! Typed immutable buffer-state dependencies for effectful schedule work.
//!
//! Pure graph values remain `NodeId`s.  This module represents only observable
//! buffer state transitions, so a future STORE/AFTER lowering cannot hide
//! write ordering in ordinary dataflow edges.
use crate::{DType, Shape};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BufferState {
    pub buffer: u64,
    pub version: u64,
    pub shape: Shape,
    pub dtype: DType,
    pub bytes: usize,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectStep {
    pub id: u64,
    pub reads: Vec<BufferState>,
    pub write: BufferState,
    /// Explicit predecessor effects; these are not inferred from labels.
    pub after: Vec<u64>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct EffectPlan {
    pub steps: Vec<EffectStep>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectError {
    Overflow,
    DuplicateStep(u64),
    DuplicateWrite {
        buffer: u64,
        version: u64,
    },
    InvalidBytes {
        buffer: u64,
    },
    InvalidVersion {
        buffer: u64,
        previous: u64,
        next: u64,
    },
    DescriptorMismatch {
        buffer: u64,
        version: u64,
    },
    UseBeforeState {
        step: u64,
        buffer: u64,
        version: u64,
    },
    MissingAfter {
        step: u64,
        after: u64,
    },
    EffectCycle,
}
impl fmt::Display for EffectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "effect plan error: {self:?}")
    }
}
impl std::error::Error for EffectError {}

impl EffectPlan {
    /// Validates the stable, source-order state transition contract. Version
    /// zero is an external/base state; each write must advance one version of
    /// exactly one descriptor-preserving logical buffer.
    pub fn validate(&self) -> Result<(), EffectError> {
        let mut steps = BTreeSet::new();
        let mut states: BTreeMap<u64, BufferState> = BTreeMap::new();
        let mut completed = BTreeSet::new();
        for step in &self.steps {
            if !steps.insert(step.id) {
                return Err(EffectError::DuplicateStep(step.id));
            }
            for dependency in &step.after {
                if !completed.contains(dependency) {
                    return Err(EffectError::MissingAfter {
                        step: step.id,
                        after: *dependency,
                    });
                }
            }
            validate_state(&step.write)?;
            let previous = states.get(&step.write.buffer);
            if let Some(previous) = previous {
                if previous.shape != step.write.shape
                    || previous.dtype != step.write.dtype
                    || previous.bytes != step.write.bytes
                {
                    return Err(EffectError::DescriptorMismatch {
                        buffer: step.write.buffer,
                        version: step.write.version,
                    });
                }
                let want = previous
                    .version
                    .checked_add(1)
                    .ok_or(EffectError::Overflow)?;
                if step.write.version != want {
                    return Err(EffectError::InvalidVersion {
                        buffer: step.write.buffer,
                        previous: previous.version,
                        next: step.write.version,
                    });
                }
            } else if step.write.version != 1 {
                return Err(EffectError::InvalidVersion {
                    buffer: step.write.buffer,
                    previous: 0,
                    next: step.write.version,
                });
            }
            for read in &step.reads {
                validate_state(read)?;
                match states.get(&read.buffer) {
                    Some(current) if current == read => {}
                    _ if read.version == 0 => {}
                    _ => {
                        return Err(EffectError::UseBeforeState {
                            step: step.id,
                            buffer: read.buffer,
                            version: read.version,
                        });
                    }
                }
            }
            states.insert(step.write.buffer, step.write.clone());
            completed.insert(step.id);
        }
        Ok(())
    }
}

fn validate_state(state: &BufferState) -> Result<(), EffectError> {
    let bytes = state
        .shape
        .numel()
        .map_err(|_| EffectError::Overflow)?
        .checked_mul(state.dtype.itemsize())
        .ok_or(EffectError::Overflow)?;
    if bytes != state.bytes {
        return Err(EffectError::InvalidBytes {
            buffer: state.buffer,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(buffer: u64, version: u64) -> BufferState {
        BufferState {
            buffer,
            version,
            shape: Shape::new([2]),
            dtype: DType::I32,
            bytes: 8,
        }
    }
    #[test]
    fn chained_states_are_deterministic_and_reject_ambiguity() {
        let plan = EffectPlan {
            steps: vec![
                EffectStep {
                    id: 3,
                    reads: vec![state(1, 0)],
                    write: state(1, 1),
                    after: vec![],
                },
                EffectStep {
                    id: 4,
                    reads: vec![state(1, 1)],
                    write: state(1, 2),
                    after: vec![3],
                },
            ],
        };
        assert!(plan.validate().is_ok());
        let mut bad = plan.clone();
        bad.steps[1].write.version = 3;
        assert!(matches!(
            bad.validate(),
            Err(EffectError::InvalidVersion { .. })
        ));
        let mut before = plan;
        before.steps[1].after = vec![99];
        assert!(matches!(
            before.validate(),
            Err(EffectError::MissingAfter { .. })
        ));
    }
}
