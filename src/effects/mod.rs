//! Typed immutable buffer-state dependencies for effectful schedule work.
//!
//! Pure graph values remain `NodeId`s.  This module represents only observable
//! buffer state transitions, so a future STORE/AFTER lowering cannot hide
//! write ordering in ordinary dataflow edges.
use crate::{DType, Shape, TensorData};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};
pub mod runtime;
pub mod schedule;
pub use runtime::{
    EffectRuntime, PersistentRuntimeStats, PersistentSlotIdentity, PersistentSnapshot, RuntimeError,
};
pub use schedule::{EffectPayload, EffectSchedule, EffectUOp, EffectUOpKind};

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
    /// Optional logical affine target inside `write`'s base storage.
    pub target_view: Option<crate::AffineView>,
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
    MissingRead {
        step: u64,
    },
    EffectCycle,
    MissingBuffer(u64),
    ValueDescriptor(u64),
    TransactionFailed {
        step: u64,
    },
    CaptureUnsupported,
    AutogradUnsupported,
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
            if let Some(view) = &step.target_view {
                validate_target_view(&step.write, view)?;
            }
            if step.reads.len() != 2 {
                return Err(EffectError::MissingRead { step: step.id });
            }
            let snapshot = &step.reads[0];
            validate_state(snapshot)?;
            if snapshot.buffer != step.write.buffer {
                return Err(EffectError::DescriptorMismatch {
                    buffer: step.write.buffer,
                    version: step.write.version,
                });
            }
            if snapshot.shape != step.write.shape
                || snapshot.dtype != step.write.dtype
                || snapshot.bytes != step.write.bytes
            {
                return Err(EffectError::DescriptorMismatch {
                    buffer: step.write.buffer,
                    version: step.write.version,
                });
            }
            if step.write.version
                != snapshot
                    .version
                    .checked_add(1)
                    .ok_or(EffectError::Overflow)?
            {
                return Err(EffectError::InvalidVersion {
                    buffer: step.write.buffer,
                    previous: snapshot.version,
                    next: step.write.version,
                });
            }
            let source = &step.reads[1];
            validate_state(source)?;
            if source.dtype != step.write.dtype {
                return Err(EffectError::DescriptorMismatch {
                    buffer: step.write.buffer,
                    version: step.write.version,
                });
            }
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

fn validate_target_view(state: &BufferState, view: &crate::AffineView) -> Result<(), EffectError> {
    if view.source_shape != state.shape || view.strides.len() != view.logical_shape.rank() {
        return Err(EffectError::DescriptorMismatch {
            buffer: state.buffer,
            version: state.version,
        });
    }
    view.validate()
        .map_err(|_| EffectError::DescriptorMismatch {
            buffer: state.buffer,
            version: state.version,
        })?;
    let mut seen = BTreeSet::new();
    for index in 0..view
        .logical_shape
        .numel()
        .map_err(|_| EffectError::Overflow)?
    {
        let offset = view
            .element_offset(index)
            .map_err(|_| EffectError::DescriptorMismatch {
                buffer: state.buffer,
                version: state.version,
            })?;
        let offset = usize::try_from(offset).map_err(|_| EffectError::DescriptorMismatch {
            buffer: state.buffer,
            version: state.version,
        })?;
        if !seen.insert(offset) {
            return Err(EffectError::DescriptorMismatch {
                buffer: state.buffer,
                version: state.version,
            });
        }
    }
    Ok(())
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

/// Typed graph-adjacent handle. It cannot be confused with a pure `NodeId`.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StateHandle(BufferState);
impl StateHandle {
    pub fn state(&self) -> &BufferState {
        &self.0
    }
}

#[derive(Clone, Debug)]
struct Assignment {
    step: EffectStep,
    source: BufferState,
}

/// Constructs a deterministic CPU effect graph without changing the pure Graph
/// DAG or giving effectful values accidental autograd semantics.
#[derive(Clone, Debug, Default)]
pub struct EffectGraph {
    initial: BTreeMap<u64, TensorData>,
    states: BTreeMap<u64, BufferState>,
    assignments: Vec<Assignment>,
}

#[derive(Clone, Debug)]
pub struct EffectCommit {
    pub values: BTreeMap<u64, TensorData>,
    pub states: BTreeMap<u64, BufferState>,
    pub trace: Vec<u64>,
}

impl EffectGraph {
    pub fn insert(&mut self, buffer: u64, value: TensorData) -> Result<StateHandle, EffectError> {
        if self.states.contains_key(&buffer) {
            return Err(EffectError::DuplicateWrite { buffer, version: 0 });
        }
        let state = state_for(buffer, 0, &value)?;
        self.initial.insert(buffer, value);
        self.states.insert(buffer, state.clone());
        Ok(StateHandle(state))
    }
    /// Adds one whole-buffer/broadcast assignment. Both operands are frozen
    /// states: later writes cannot change this source read.
    pub fn assign(
        &mut self,
        target: &StateHandle,
        source: &StateHandle,
    ) -> Result<StateHandle, EffectError> {
        self.assign_inner(target, source, None)
    }

    /// Assigns into a statically proven injective logical region of a base
    /// state. `ViewMap` is the shared rangeification/view ABI, not a new view
    /// representation.
    pub fn assign_view(
        &mut self,
        target: &StateHandle,
        source: &StateHandle,
        view: crate::ViewMap,
    ) -> Result<StateHandle, EffectError> {
        self.assign_inner(target, source, Some(view.into()))
    }

    /// Assigns through the shared signed affine descriptor.
    pub fn assign_affine_view(
        &mut self,
        target: &StateHandle,
        source: &StateHandle,
        view: crate::AffineView,
    ) -> Result<StateHandle, EffectError> {
        self.assign_inner(target, source, Some(view))
    }

    fn assign_inner(
        &mut self,
        target: &StateHandle,
        source: &StateHandle,
        target_view: Option<crate::AffineView>,
    ) -> Result<StateHandle, EffectError> {
        let current = self
            .states
            .get(&target.0.buffer)
            .ok_or(EffectError::MissingBuffer(target.0.buffer))?;
        if current != &target.0 {
            return Err(EffectError::UseBeforeState {
                step: self.assignments.len() as u64,
                buffer: target.0.buffer,
                version: target.0.version,
            });
        }
        let source_current = self
            .states
            .get(&source.0.buffer)
            .ok_or(EffectError::MissingBuffer(source.0.buffer))?;
        if source_current.version < source.0.version {
            return Err(EffectError::UseBeforeState {
                step: self.assignments.len() as u64,
                buffer: source.0.buffer,
                version: source.0.version,
            });
        }
        if current.dtype != source.0.dtype {
            return Err(EffectError::DescriptorMismatch {
                buffer: target.0.buffer,
                version: target.0.version,
            });
        }
        // Assignment broadcasting is checked by the transactional executor;
        // preflight values exist here for a deterministic construction error.
        let mut probe = self
            .initial
            .get(&target.0.buffer)
            .ok_or(EffectError::MissingBuffer(target.0.buffer))?
            .clone();
        let source_value = self
            .initial
            .get(&source.0.buffer)
            .ok_or(EffectError::MissingBuffer(source.0.buffer))?;
        if let Some(view) = &target_view {
            validate_target_view(current, view)?;
            probe.assign_view_from(view, source_value)
        } else {
            probe.assign_from(source_value)
        }
        .map_err(|_| EffectError::DescriptorMismatch {
            buffer: target.0.buffer,
            version: target.0.version,
        })?;
        let next = BufferState {
            buffer: current.buffer,
            version: current
                .version
                .checked_add(1)
                .ok_or(EffectError::Overflow)?,
            shape: current.shape.clone(),
            dtype: current.dtype,
            bytes: current.bytes,
        };
        let id = self.assignments.len() as u64;
        self.assignments.push(Assignment {
            step: EffectStep {
                id,
                reads: vec![target.0.clone(), source.0.clone()],
                write: next.clone(),
                target_view,
                after: self
                    .assignments
                    .last()
                    .map(|last| last.step.id)
                    .into_iter()
                    .collect(),
            },
            source: source.0.clone(),
        });
        self.states.insert(next.buffer, next.clone());
        Ok(StateHandle(next))
    }
    pub fn plan(&self) -> EffectPlan {
        EffectPlan {
            steps: self
                .assignments
                .iter()
                .map(|assignment| assignment.step.clone())
                .collect(),
        }
    }

    /// Effect artifacts are deliberately gated until a complete state-store
    /// replay ABI exists; callers cannot serialize or silently drop effects.
    pub fn capture(&self) -> Result<(), EffectError> {
        Err(EffectError::CaptureUnsupported)
    }

    /// Effects have no VJP contract yet.  Rejecting at the state API keeps
    /// mutation from accidentally participating in the pure graph gradient.
    pub fn grad(&self) -> Result<(), EffectError> {
        Err(EffectError::AutogradUnsupported)
    }
    /// Executes all assignments as a single transaction. Every source is read
    /// from a staged snapshot; no visible value changes if any preflight fails.
    pub fn execute(&self) -> Result<EffectCommit, EffectError> {
        self.plan().validate()?;
        let mut staged = self.initial.clone();
        // Versioned snapshots are distinct from the externally visible latest
        // state.  This gives overlap assignments read-before-write semantics.
        let mut snapshots = self
            .initial
            .iter()
            .map(|(buffer, value)| ((*buffer, 0_u64), value.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut states = self
            .initial
            .iter()
            .map(|(buffer, value)| state_for(*buffer, 0, value).map(|state| (*buffer, state)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let mut trace = Vec::with_capacity(self.assignments.len());
        for assignment in &self.assignments {
            let target_state = &assignment.step.reads[0];
            let target = snapshots
                .get(&(target_state.buffer, target_state.version))
                .ok_or(EffectError::UseBeforeState {
                    step: assignment.step.id,
                    buffer: target_state.buffer,
                    version: target_state.version,
                })?
                .clone();
            let source = snapshots
                .get(&(assignment.source.buffer, assignment.source.version))
                .ok_or(EffectError::UseBeforeState {
                    step: assignment.step.id,
                    buffer: assignment.source.buffer,
                    version: assignment.source.version,
                })?
                .clone();
            let mut candidate = target;
            if let Some(view) = &assignment.step.target_view {
                candidate.assign_view_from(view, &source)
            } else {
                candidate.assign_from(&source)
            }
            .map_err(|_| EffectError::TransactionFailed {
                step: assignment.step.id,
            })?;
            snapshots.insert(
                (assignment.step.write.buffer, assignment.step.write.version),
                candidate.clone(),
            );
            staged.insert(assignment.step.write.buffer, candidate);
            states.insert(assignment.step.write.buffer, assignment.step.write.clone());
            trace.push(assignment.step.id);
        }
        Ok(EffectCommit {
            values: staged,
            states,
            trace,
        })
    }
}

fn state_for(buffer: u64, version: u64, value: &TensorData) -> Result<BufferState, EffectError> {
    let bytes = value
        .len()
        .checked_mul(value.dtype().itemsize())
        .ok_or(EffectError::Overflow)?;
    Ok(BufferState {
        buffer,
        version,
        shape: value.shape().clone(),
        dtype: value.dtype(),
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;
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
                    reads: vec![state(1, 0), state(1, 0)],
                    write: state(1, 1),
                    target_view: None,
                    after: vec![],
                },
                EffectStep {
                    id: 4,
                    reads: vec![state(1, 1), state(1, 1)],
                    write: state(1, 2),
                    target_view: None,
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

    #[test]
    fn affine_assignment_preserves_untouched_raw_base_lanes() {
        let mut graph = EffectGraph::default();
        let base = graph
            .insert(
                1,
                TensorData::from_storage([2, 3], Storage::F16(vec![1, 2, 3, 4, 5, 6])).unwrap(),
            )
            .unwrap();
        let source = graph
            .insert(
                2,
                TensorData::from_storage([1, 2], Storage::F16(vec![0x7e01, 0x8000])).unwrap(),
            )
            .unwrap();
        let view = crate::ViewMap::identity(Shape::from([2, 3]))
            .shrink(&[(1, 2), (1, 3)])
            .unwrap();
        let next = graph.assign_view(&base, &source, view).unwrap();
        let committed = graph.execute().unwrap();
        assert_eq!(
            committed.values[&1].storage(),
            &Storage::F16(vec![1, 2, 3, 4, 0x7e01, 0x8000])
        );
        assert_eq!(next.state().version, 1);
    }

    #[test]
    fn affine_views_are_injective_and_versioned_in_plan_order() {
        let mut graph = EffectGraph::default();
        let base = graph
            .insert(
                10,
                TensorData::from_storage([2, 3], Storage::U64(vec![9; 6])).unwrap(),
            )
            .unwrap();
        let first_source = graph
            .insert(
                11,
                TensorData::from_storage([2, 1], Storage::U64(vec![u64::MAX, 7])).unwrap(),
            )
            .unwrap();
        let second_source = graph
            .insert(
                12,
                TensorData::from_storage([1, 2], Storage::U64(vec![1, 2])).unwrap(),
            )
            .unwrap();
        let first_view = crate::ViewMap::identity(Shape::from([2, 3]))
            .shrink(&[(0, 2), (1, 2)])
            .unwrap();
        let first = graph.assign_view(&base, &first_source, first_view).unwrap();
        let second_view = crate::ViewMap::identity(Shape::from([2, 3]))
            .permute(&[1, 0])
            .unwrap()
            .shrink(&[(1, 2), (0, 2)])
            .unwrap();
        let second = graph
            .assign_view(&first, &second_source, second_view)
            .unwrap();
        let committed = graph.execute().unwrap();
        assert_eq!(
            committed.values[&10].storage(),
            &Storage::U64(vec![9, 1, 9, 9, 2, 9])
        );
        assert_eq!(second.state().version, 2);
        let expanded = crate::ViewMap::identity(Shape::from([1]))
            .expand(Shape::from([2]))
            .unwrap();
        assert!(matches!(
            graph.assign_view(&second, &first_source, expanded),
            Err(EffectError::DescriptorMismatch { .. })
        ));
    }

    #[test]
    fn signed_affine_flip_preserves_snapshot_and_base_bytes() {
        let mut graph = EffectGraph::default();
        let base = graph
            .insert(
                20,
                TensorData::from_storage([4], Storage::BF16(vec![1, 2, 3, 4])).unwrap(),
            )
            .unwrap();
        let source = graph
            .insert(
                21,
                TensorData::from_storage([4], Storage::BF16(vec![0x7fc1, 0x8000, 7, 8])).unwrap(),
            )
            .unwrap();
        let flip = crate::AffineView::identity(Shape::from([4]))
            .flip(0)
            .unwrap();
        let next = graph.assign_affine_view(&base, &source, flip).unwrap();
        assert_eq!(
            graph.execute().unwrap().values[&20].storage(),
            &Storage::BF16(vec![8, 7, 0x8000, 0x7fc1])
        );
        assert_eq!(next.state().version, 1);
    }

    #[test]
    fn effect_graph_stages_snapshot_assignments_before_commit() {
        let mut graph = EffectGraph::default();
        let target = graph
            .insert(
                1,
                TensorData::from_storage([2, 2], crate::Storage::U64(vec![0; 4])).unwrap(),
            )
            .unwrap();
        let source = graph
            .insert(
                2,
                TensorData::from_storage(
                    [2, 2],
                    crate::Storage::U64(vec![u64::MAX, 7, u64::MAX, 7]),
                )
                .unwrap(),
            )
            .unwrap();
        let first = graph.assign(&target, &source).unwrap();
        let second = graph.assign(&source, &first).unwrap();
        let commit = graph.execute().unwrap();
        assert_eq!(commit.trace, vec![0, 1]);
        assert_eq!(
            commit.values[&1].storage(),
            &crate::Storage::U64(vec![u64::MAX, 7, u64::MAX, 7])
        );
        assert_eq!(
            commit.values[&2].storage(),
            &crate::Storage::U64(vec![u64::MAX, 7, u64::MAX, 7])
        );
        assert_eq!(second.state().version, 1);
    }

    #[test]
    fn store_after_schedule_has_stable_order_and_failure_is_retryable() {
        let mut graph = EffectGraph::default();
        let a = graph
            .insert(
                1,
                TensorData::from_storage([1], crate::Storage::F32(vec![0.0])).unwrap(),
            )
            .unwrap();
        let b = graph
            .insert(
                2,
                TensorData::from_storage([1], crate::Storage::F32(vec![3.0])).unwrap(),
            )
            .unwrap();
        let c = graph.assign(&a, &b).unwrap();
        graph.assign(&b, &c).unwrap();
        let schedule = EffectSchedule::lower(&graph).unwrap();
        assert_eq!(schedule.uops.len(), 4);
        assert_eq!(
            EffectSchedule::lower(&graph).unwrap().cache_key,
            schedule.cache_key
        );
        assert!(matches!(
            crate::uop::artifact::encode(&schedule.uops[0].uop),
            Err(crate::uop::artifact::ArtifactError::Unsupported)
        ));
        assert!(matches!(
            schedule.execute(&graph, Some(1)),
            Err(EffectError::TransactionFailed { step: 1 })
        ));
        assert!(matches!(
            graph.capture(),
            Err(EffectError::CaptureUnsupported)
        ));
        assert_eq!(
            schedule.execute(&graph, None).unwrap().values[&1].storage(),
            &crate::Storage::F32(vec![3.0])
        );
    }

    #[test]
    fn normal_schedule_effect_items_are_transactional_and_not_capturable() {
        let mut graph = EffectGraph::default();
        let a = graph
            .insert(
                10,
                TensorData::from_storage([2], crate::Storage::U64(vec![0, 0])).unwrap(),
            )
            .unwrap();
        let b = graph
            .insert(
                20,
                TensorData::from_storage([2], crate::Storage::U64(vec![9, 9])).unwrap(),
            )
            .unwrap();
        let first = graph.assign(&a, &b).unwrap();
        graph.assign(&b, &first).unwrap();
        let schedule = crate::schedule_effects(&graph).unwrap();
        assert_eq!(schedule.items.len(), 2);
        assert!(schedule.items.iter().all(crate::ScheduleItem::is_effect));
        assert_eq!(schedule.items[1].dependencies, vec![0]);
        assert_eq!(schedule.items[0].consumers, vec![1]);
        assert!(schedule.items[0].validate_input_bindings().is_ok());
        assert!(matches!(
            crate::CapturedSchedule::capture(&crate::Graph::new(), &schedule, &[]),
            Err(crate::ReplayError::Unsupported(_))
        ));
        assert!(matches!(
            graph.grad(),
            Err(EffectError::AutogradUnsupported)
        ));
        assert!(matches!(
            crate::engine::realize_effects(&graph, &schedule, Some(1)),
            Err(crate::engine::RealizationError::Execution(_))
        ));
        let committed = crate::engine::realize_effects(&graph, &schedule, None).unwrap();
        assert_eq!(committed.trace, vec![0, 1]);
        assert_eq!(
            committed.values[&10].storage(),
            &crate::Storage::U64(vec![9, 9])
        );
        assert_eq!(
            committed.values[&20].storage(),
            &crate::Storage::U64(vec![9, 9])
        );
    }
}
