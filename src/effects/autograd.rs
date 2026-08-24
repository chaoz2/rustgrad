//! Pure-graph provenance checks for graph-adjacent STORE operations.
//!
//! The token in this module is intentionally immutable and bound to one
//! `BufferState`. It does not add an effect edge to `Graph`, retain a mutable
//! alias registry, or claim an effect VJP.

use super::{BufferState, EffectError, StateHandle};
use crate::{Graph, NodeId};

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
