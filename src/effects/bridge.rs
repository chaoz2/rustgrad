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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorData;

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
    fn mutation_tape_whole_and_indexed_vjps_keep_frozen_provenance() {
        let base = TensorData::new([3], vec![1.0f32, 2.0, 3.0]).unwrap();
        let rhs = TensorData::new([3], vec![4.0f32, 5.0, 6.0]).unwrap();
        let mut effects = EffectGraph::default();
        let target = effects.insert(1, base).unwrap();
        let source = effects.insert(2, rhs).unwrap();
        effects.assign(&target, &source).unwrap();
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
            TensorData::new([3], vec![7.0f32, 8.0, 9.0]).unwrap()
        );
        assert_eq!(tape.rhs_output(), x);
    }
}
