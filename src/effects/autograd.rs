//! Pure-graph provenance checks for graph-adjacent STORE operations.
//!
//! The token in this module is intentionally immutable and bound to one
//! `BufferState`. It does not add an effect edge to `Graph`, retain a mutable
//! alias registry, or claim an effect VJP.

use super::{BufferState, EffectError, EffectGraph, EffectSourceBridge, StateHandle};
use crate::{DType, Graph, NodeId, TensorData};
use std::collections::BTreeSet;

/// Why a graph-derived mutation permit is safe for its exact state snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MutationSafety {
    /// The pure value is not gradient tracked.
    NoGrad,
    /// Value provenance reaches the loss only through a `Detach` boundary.
    Detached,
    /// The target only reaches the loss through a nondifferentiable control
    /// edge such as a predicate or index. It is not retained by reverse mode.
    NonDifferentiableUse,
    /// The target is outside both the value and reverse slices of the loss.
    Unrelated,
}

/// Capability authorizing a single graph-adjacent state mutation.
///
/// Construct it with [`EffectMutationPermit::from_graph`], then pass it to a
/// guarded `EffectGraph` assignment method. It is tied to the graph identity,
/// requested loss/target analysis, and exact pre-write buffer generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectMutationPermit {
    graph: u64,
    loss: NodeId,
    target: NodeId,
    state: BufferState,
    safety: MutationSafety,
}

impl EffectMutationPermit {
    /// Analyzes a requested first-order reverse slice without appending a
    /// derivative graph. If the old target value can be read by that slice,
    /// returns a typed error before any STORE is constructed or executed.
    pub fn from_graph(
        graph: &Graph,
        loss: NodeId,
        target: NodeId,
        state: &StateHandle,
    ) -> Result<Self, EffectError> {
        let target_requires_grad = graph
            .requires_grad(target)
            .map_err(|_| EffectError::MutationUnknownNode(target.index()))?;
        graph
            .node(loss)
            .map_err(|_| EffectError::MutationUnknownNode(loss.index()))?;
        if target_requires_grad
            && graph
                .backward_slice_contains(loss, target)
                .map_err(|_| EffectError::MutationUnknownNode(target.index()))?
        {
            return Err(EffectError::MutationWouldInvalidateBackward {
                buffer: state.state().buffer,
                version: state.state().version,
                graph: graph.id(),
                loss: loss.index(),
                target: target.index(),
            });
        }
        let safety = if !target_requires_grad {
            MutationSafety::NoGrad
        } else if graph
            .value_slice_contains(loss, target)
            .map_err(|_| EffectError::MutationUnknownNode(target.index()))?
        {
            if graph
                .value_slice_contains_detach(loss, target)
                .map_err(|_| EffectError::MutationUnknownNode(target.index()))?
            {
                MutationSafety::Detached
            } else {
                MutationSafety::NonDifferentiableUse
            }
        } else {
            MutationSafety::Unrelated
        };
        Ok(Self {
            graph: graph.id(),
            loss,
            target,
            state: state.state().clone(),
            safety,
        })
    }

    pub fn safety(&self) -> MutationSafety {
        self.safety
    }

    pub fn graph_id(&self) -> u64 {
        self.graph
    }

    pub fn loss(&self) -> NodeId {
        self.loss
    }

    pub fn target(&self) -> NodeId {
        self.target
    }

    pub(super) fn permits(&self, target: &StateHandle) -> Result<(), EffectError> {
        if self.state != target.0 {
            return Err(EffectError::MutationPermitMismatch {
                buffer: target.0.buffer,
                version: target.0.version,
            });
        }
        Ok(())
    }
}

/// Immutable first-order mutation-local provenance and assignment map.
#[derive(Clone, Debug)]
pub struct MutationTapeRecord {
    pre_write: BufferState,
    rhs: NodeId,
    form: MutationAssignmentForm,
    after: Vec<u64>,
}

#[derive(Clone, Debug)]
enum MutationAssignmentForm {
    Whole,
    Affine(crate::AffineView),
    Indexed(crate::ir::indexing::StaticIndexPlan),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MutationVjp {
    pub pre_write: TensorData,
    pub rhs_output: TensorData,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MutationVjpError {
    Effect(EffectError),
    NonF32,
    UpstreamShape,
}

impl From<EffectError> for MutationVjpError {
    fn from(value: EffectError) -> Self {
        Self::Effect(value)
    }
}

impl MutationTapeRecord {
    /// Captures only frozen plan/provenance metadata; it does not mutate state.
    pub fn from_bridge(
        bridge: &EffectSourceBridge,
        effects: &EffectGraph,
    ) -> Result<Self, MutationVjpError> {
        let (binding, _) = bridge.provenance();
        let step = effects
            .plan()
            .steps
            .into_iter()
            .find(|step| step.id == binding.step)
            .ok_or(EffectError::MissingAfter {
                step: binding.step,
                after: binding.step,
            })?;
        let form = match (step.target_view, step.index_plan) {
            (Some(view), None) => MutationAssignmentForm::Affine(view),
            (None, Some(plan)) => MutationAssignmentForm::Indexed(plan),
            (None, None) => MutationAssignmentForm::Whole,
            _ => {
                return Err(EffectError::DescriptorMismatch {
                    buffer: step.write.buffer,
                    version: step.write.version,
                }
                .into());
            }
        };
        Ok(Self {
            pre_write: step.reads[0].clone(),
            rhs: binding.output,
            form,
            after: step.after,
        })
    }

    /// Computes the local first-order adjoint for an F32 replacement write.
    pub fn vjp(&self, upstream: &TensorData) -> Result<MutationVjp, MutationVjpError> {
        if self.pre_write.dtype != DType::F32 || upstream.dtype() != DType::F32 {
            return Err(MutationVjpError::NonF32);
        }
        if upstream.shape() != &self.pre_write.shape {
            return Err(MutationVjpError::UpstreamShape);
        }
        let mut old = vec![0.0; upstream.len()];
        let rhs_shape = match &self.form {
            MutationAssignmentForm::Whole => self.pre_write.shape.clone(),
            MutationAssignmentForm::Affine(view) => view.logical_shape.clone(),
            MutationAssignmentForm::Indexed(plan) => plan.output_shape().clone(),
        };
        let mut rhs = vec![
            0.0;
            rhs_shape
                .numel()
                .map_err(|_| MutationVjpError::Effect(EffectError::Overflow))?
        ];
        match &self.form {
            MutationAssignmentForm::Whole => {
                rhs.copy_from_slice(upstream.values());
            }
            MutationAssignmentForm::Affine(view) => {
                view.validate_write().map_err(|_| {
                    MutationVjpError::Effect(EffectError::DescriptorMismatch {
                        buffer: self.pre_write.buffer,
                        version: self.pre_write.version,
                    })
                })?;
                let mut written = BTreeSet::new();
                for (lane, slot) in rhs.iter_mut().enumerate() {
                    let offset = usize::try_from(
                        view.element_offset(lane)
                            .map_err(|_| MutationVjpError::Effect(EffectError::Overflow))?,
                    )
                    .map_err(|_| MutationVjpError::Effect(EffectError::Overflow))?;
                    written.insert(offset);
                    *slot = upstream.values()[offset];
                }
                for (lane, slot) in old.iter_mut().enumerate() {
                    if !written.contains(&lane) {
                        *slot = upstream.values()[lane];
                    }
                }
            }
            MutationAssignmentForm::Indexed(plan) => {
                let offsets = plan
                    .source_offsets()
                    .map_err(|_| MutationVjpError::Effect(EffectError::Overflow))?;
                let mut final_writer = std::collections::BTreeMap::new();
                for (lane, offset) in offsets.iter().enumerate() {
                    final_writer.insert(*offset, lane);
                }
                for (lane, slot) in old.iter_mut().enumerate() {
                    if !final_writer.contains_key(&lane) {
                        *slot = upstream.values()[lane];
                    }
                }
                for (offset, lane) in final_writer {
                    rhs[lane] = upstream.values()[offset];
                }
            }
        }
        Ok(MutationVjp {
            pre_write: TensorData::new(self.pre_write.shape.clone(), old)
                .map_err(|_| MutationVjpError::Effect(EffectError::Overflow))?,
            rhs_output: TensorData::new(rhs_shape, rhs)
                .map_err(|_| MutationVjpError::Effect(EffectError::Overflow))?,
        })
    }

    pub fn rhs_output(&self) -> NodeId {
        self.rhs
    }
    pub fn after(&self) -> &[u64] {
        &self.after
    }
    pub fn pre_write(&self) -> &BufferState {
        &self.pre_write
    }
}
