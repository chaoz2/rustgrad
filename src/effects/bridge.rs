//! Graph-to-effect source binding for host interpreter execution.
//!
//! This is a sidecar to a frozen [`super::EffectGraph`] plan: it owns neither
//! `NodeId`s nor effect states, and records no new effect IR.  It only binds
//! exact persistent snapshots to pure graph inputs and one pure output to an
//! existing STORE source position.

use super::{BufferState, EffectError, EffectGraph, EffectRuntime, RuntimeError};
use crate::{Backend, CpuBackend, Graph, NodeId, Op};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One exact persistent state injected into one pure graph input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentInputBinding {
    pub input: NodeId,
    pub state: BufferState,
}

/// One pure output substituted for one existing effect STORE source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PureEffectBinding {
    pub step: u64,
    pub output: NodeId,
}

/// Immutable, graph-local provenance sidecar for a host pure-to-effect step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectSourceBridge {
    graph: u64,
    effect: PureEffectBinding,
    pre_write: BufferState,
    source: BufferState,
    after: Vec<u64>,
    inputs: Vec<PersistentInputBinding>,
}

impl EffectSourceBridge {
    /// Validates a single pure output against one frozen effect source slot.
    pub fn new(
        graph: &Graph,
        effects: &EffectGraph,
        effect: PureEffectBinding,
        mut inputs: Vec<PersistentInputBinding>,
    ) -> Result<Self, EffectError> {
        let output = graph
            .node(effect.output)
            .map_err(|_| EffectError::MutationUnknownNode(effect.output.index()))?;
        let step = effects
            .plan()
            .steps
            .into_iter()
            .find(|step| step.id == effect.step)
            .ok_or(EffectError::MissingAfter {
                step: effect.step,
                after: effect.step,
            })?;
        let source = &step.reads[1];
        if output.shape != source.shape || output.dtype != source.dtype {
            return Err(EffectError::DescriptorMismatch {
                buffer: source.buffer,
                version: source.version,
            });
        }
        inputs.sort_by_key(|binding| binding.input.index());
        let mut seen = BTreeSet::new();
        for binding in &inputs {
            if !seen.insert(binding.input) {
                return Err(EffectError::MutationUnknownNode(binding.input.index()));
            }
            let node = graph
                .node(binding.input)
                .map_err(|_| EffectError::MutationUnknownNode(binding.input.index()))?;
            if node.shape != binding.state.shape || node.dtype != binding.state.dtype {
                return Err(EffectError::DescriptorMismatch {
                    buffer: binding.state.buffer,
                    version: binding.state.version,
                });
            }
            if !matches!(node.op, Op::Input { .. }) {
                return Err(EffectError::MutationUnknownNode(binding.input.index()));
            }
        }
        Ok(Self {
            graph: graph.id(),
            effect,
            pre_write: step.reads[0].clone(),
            source: step.reads[1].clone(),
            after: step.after.clone(),
            inputs,
        })
    }

    /// Executes the pure prefix against exact runtime snapshots, then commits
    /// the frozen effect plan once through its existing source-override path.
    pub fn execute(
        &self,
        graph: &Graph,
        effects: &EffectGraph,
        runtime: &mut EffectRuntime,
        injected_store_failure: Option<u64>,
    ) -> Result<Vec<BufferState>, RuntimeError> {
        if graph.id() != self.graph {
            return Err(RuntimeError::Effect(EffectError::MutationPermitMismatch {
                buffer: 0,
                version: 0,
            }));
        }
        let mut pure_inputs = HashMap::new();
        for binding in &self.inputs {
            let snapshot = runtime.snapshot(&binding.state)?;
            let name = match &graph
                .node(binding.input)
                .map_err(|_| {
                    RuntimeError::Effect(EffectError::MutationUnknownNode(binding.input.index()))
                })?
                .op
            {
                Op::Input { name } => name.clone(),
                _ => {
                    return Err(RuntimeError::Effect(EffectError::MutationUnknownNode(
                        binding.input.index(),
                    )));
                }
            };
            pure_inputs.insert(name, snapshot.tensor().clone());
        }
        let output = CpuBackend
            .execute(graph, self.effect.output, &pure_inputs)
            .map_err(|_| {
                RuntimeError::Effect(EffectError::TransactionFailed {
                    step: self.effect.step,
                })
            })?;
        let sources = BTreeMap::from([(self.effect.step, output)]);
        runtime.execute_with_sources(&effects.plan(), &sources, injected_store_failure)
    }

    pub fn provenance(&self) -> (&PureEffectBinding, &[PersistentInputBinding]) {
        (&self.effect, &self.inputs)
    }

    pub(crate) const fn graph_id(&self) -> u64 {
        self.graph
    }

    pub(crate) fn assignment_provenance(&self) -> (&BufferState, &BufferState, &[u64]) {
        (&self.pre_write, &self.source, &self.after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, TensorData};

    #[test]
    fn pure_rhs_snapshot_overrides_one_store_source_atomically() {
        let target_value = TensorData::new([2], vec![1.0f32, 2.0]).unwrap();
        let source_value = TensorData::new([2], vec![3.0f32, 4.0]).unwrap();
        let mut effects = EffectGraph::default();
        let target = effects.insert(1, target_value.clone()).unwrap();
        let source = effects.insert(2, source_value.clone()).unwrap();
        let next = effects.assign(&target, &source).unwrap();
        let mut runtime = EffectRuntime::new();
        let target_state = runtime.register(1, target_value).unwrap();
        let source_state = runtime.register(2, source_value).unwrap();

        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let two = graph.constant(TensorData::scalar(2.0f32));
        let output = graph.mul(x, two).unwrap();
        let bridge = EffectSourceBridge::new(
            &graph,
            &effects,
            PureEffectBinding { step: 0, output },
            vec![PersistentInputBinding {
                input: x,
                state: source_state,
            }],
        )
        .unwrap();
        let states = bridge
            .execute(&graph, &effects, &mut runtime, None)
            .unwrap();
        assert_eq!(states, vec![next.state().clone()]);
        assert_eq!(
            runtime.snapshot(&next.state().clone()).unwrap().tensor(),
            &TensorData::new([2], vec![6.0f32, 8.0]).unwrap()
        );
        assert!(matches!(
            runtime.snapshot(&target_state),
            Err(RuntimeError::StaleState {
                buffer: 1,
                version: 0
            })
        ));
    }

    #[test]
    fn mutation_tape_whole_vjp_reduces_to_the_pure_rhs_descriptor() {
        let base = TensorData::new([2, 2], vec![1.0f32; 4]).unwrap();
        let rhs = TensorData::scalar(4.0f32);
        let mut effects = EffectGraph::default();
        let target = effects.insert(1, base).unwrap();
        let source = effects.insert(2, rhs).unwrap();
        effects.assign(&target, &source).unwrap();
        let mut graph = Graph::new();
        let x = graph.input("x", []);
        let bridge = EffectSourceBridge::new(
            &graph,
            &effects,
            PureEffectBinding { step: 0, output: x },
            vec![],
        )
        .unwrap();
        let tape = crate::effects::MutationTapeRecord::from_bridge(&bridge, &effects).unwrap();
        let vjp = tape
            .vjp(&TensorData::new([2, 2], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap())
            .unwrap();
        assert_eq!(
            vjp.pre_write,
            TensorData::new([2, 2], vec![0.0f32; 4]).unwrap()
        );
        assert_eq!(vjp.rhs_output, TensorData::scalar(10.0f32));
        assert_eq!(tape.rhs_output(), x);
    }

    #[test]
    fn mutation_tape_affine_and_indexed_vjps_preserve_assignment_adjoint() {
        let base = TensorData::new([3], vec![1.0f32, 2.0, 3.0]).unwrap();
        let mut effects = EffectGraph::default();
        let target = effects.insert(1, base).unwrap();
        let source = effects
            .insert(2, TensorData::new([3], vec![4.0f32, 5.0, 6.0]).unwrap())
            .unwrap();
        let flip = crate::AffineView::identity(crate::Shape::from([3]))
            .flip(0)
            .unwrap();
        effects.assign_affine_view(&target, &source, flip).unwrap();
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let bridge = EffectSourceBridge::new(
            &graph,
            &effects,
            PureEffectBinding { step: 0, output: x },
            vec![],
        )
        .unwrap();
        let tape = crate::effects::MutationTapeRecord::from_bridge(&bridge, &effects).unwrap();
        let vjp = tape
            .vjp(&TensorData::new([3], vec![7.0f32, 8.0, 9.0]).unwrap())
            .unwrap();
        assert_eq!(
            vjp.pre_write,
            TensorData::new([3], vec![0.0f32; 3]).unwrap()
        );
        assert_eq!(
            vjp.rhs_output,
            TensorData::new([3], vec![9.0f32, 8.0, 7.0]).unwrap()
        );

        use crate::ir::indexing::{StaticIndex, StaticIndexPlan};
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(3, TensorData::new([3], vec![1.0f32, 2.0, 3.0]).unwrap())
            .unwrap();
        let source = effects
            .insert(4, TensorData::new([3], vec![5.0f32; 3]).unwrap())
            .unwrap();
        let plan = StaticIndexPlan::new(
            crate::Shape::from([3]),
            &[StaticIndex::Advanced {
                shape: crate::Shape::from([3]),
                values: vec![1, 1, -1],
            }],
        )
        .unwrap();
        effects.static_index_assign(&target, &source, plan).unwrap();
        let indexed_output = graph.input("indexed", [3]);
        let bridge = EffectSourceBridge::new(
            &graph,
            &effects,
            PureEffectBinding {
                step: 0,
                output: indexed_output,
            },
            vec![],
        )
        .unwrap();
        let tape = crate::effects::MutationTapeRecord::from_bridge(&bridge, &effects).unwrap();
        let vjp = tape
            .vjp(&TensorData::new([3], vec![10.0f32, 20.0, 30.0]).unwrap())
            .unwrap();
        assert_eq!(
            vjp.pre_write,
            TensorData::new([3], vec![10.0f32, 0.0, 0.0]).unwrap()
        );
        // The first duplicate write is overwritten and receives no credit.
        assert_eq!(
            vjp.rhs_output,
            TensorData::new([3], vec![0.0f32, 20.0, 30.0]).unwrap()
        );
    }

    #[test]
    fn mutation_tape_graph_vjp_uses_the_explicit_rhs_seed() {
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(1, TensorData::new([2], vec![0.0f32; 2]).unwrap())
            .unwrap();
        let source = effects
            .insert(2, TensorData::new([2], vec![0.0f32; 2]).unwrap())
            .unwrap();
        effects.assign(&target, &source).unwrap();
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let rhs = graph.mul(x, x).unwrap();
        let bridge = EffectSourceBridge::new(
            &graph,
            &effects,
            PureEffectBinding {
                step: 0,
                output: rhs,
            },
            vec![],
        )
        .unwrap();
        let tape = crate::effects::MutationTapeRecord::from_bridge(&bridge, &effects).unwrap();
        let seed = TensorData::new([2], vec![3.0f32, 4.0]).unwrap();
        let derivative = tape.graph_vjp(&mut graph, x, seed, false).unwrap();
        let values = HashMap::from([(
            "x".to_owned(),
            TensorData::new([2], vec![2.0f32, 3.0]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, derivative, &values).unwrap(),
            TensorData::new([2], vec![12.0f32, 24.0]).unwrap()
        );

        let disconnected = graph.input("disconnected", [2]);
        let before = graph.node_count();
        assert!(matches!(
            tape.graph_vjp(
                &mut graph,
                disconnected,
                TensorData::new([2], vec![1.0f32, 1.0]).unwrap(),
                false,
            ),
            Err(crate::effects::MutationVjpError::GraphNode(node)) if node == disconnected
        ));
        assert_eq!(
            graph.node_count(),
            before,
            "failed mutation VJP must not publish its explicit seed"
        );
    }
}
