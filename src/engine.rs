//! Deterministic realization of scheduled UOp items.
use crate::{
    BufferRole, CpuJitBackend, Graph, JitFallback, KernelBindings, KernelBufferDesc, NodeId, Op,
    Schedule, TensorData,
};
use std::{collections::HashMap, fmt};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealizationPolicy {
    Interpreter,
    CpuJit { fallback_to_interpreter: bool },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ItemBackend {
    Interpreter,
    NativeJit,
    JitFallback,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ItemTrace {
    pub item: u64,
    pub dependencies: Vec<u64>,
    pub backend: ItemBackend,
    pub cache_key: u64,
    pub materialized_buffer: u64,
    /// Stable schedule item at which this owned buffer has its final consumer.
    /// A future allocator can reuse only after this point.
    pub last_consumer: Option<u64>,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RealizationTrace {
    pub items: Vec<ItemTrace>,
}
#[derive(Clone, Debug)]
pub struct Realized {
    pub outputs: Vec<TensorData>,
    pub trace: RealizationTrace,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RealizationError {
    Schedule(String),
    MissingBuffer(u64),
    Unsupported(String),
    Execution(String),
}
impl fmt::Display for RealizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "realization error: {self:?}")
    }
}
impl std::error::Error for RealizationError {}

pub fn realize(
    graph: &Graph,
    schedule: &Schedule,
    requested: &[NodeId],
    inputs: &HashMap<String, TensorData>,
    policy: RealizationPolicy,
) -> Result<Realized, RealizationError> {
    let mut values: HashMap<u64, TensorData> = HashMap::new();
    let mut trace = RealizationTrace::default();
    let jit = matches!(policy, RealizationPolicy::CpuJit { .. })
        .then(|| CpuJitBackend::new(JitFallback::Error));
    for item in &schedule.items {
        if item.boundary.is_some() {
            return Err(RealizationError::Unsupported(format!(
                "item {} has boundary {:?}",
                item.id, item.boundary
            )));
        }
        if item
            .dependencies
            .iter()
            .any(|dependency| !trace.items.iter().any(|entry| entry.item == *dependency))
        {
            return Err(RealizationError::Schedule(format!(
                "item {} uses a future dependency",
                item.id
            )));
        }
        let mut backend = ItemBackend::Interpreter;
        let value = if let Some(jit) = &jit {
            let native_eligible = item.dependencies.is_empty()
                && item.inputs.iter().all(|buffer| {
                    matches!(
                        graph.op(NodeId::from_index(buffer.id as usize)),
                        Ok(Op::Input { .. } | Op::Constant(_))
                    )
                });
            if native_eligible {
                match jit.execute_native(graph, item.node, inputs) {
                    Ok((value, _)) => {
                        backend = ItemBackend::NativeJit;
                        value
                    }
                    Err(error)
                        if matches!(
                            policy,
                            RealizationPolicy::CpuJit {
                                fallback_to_interpreter: true
                            }
                        ) =>
                    {
                        backend = ItemBackend::JitFallback;
                        interpret_item(graph, item, inputs, &values)
                            .map_err(|e| RealizationError::Execution(format!("{error}; {e}")))?
                    }
                    Err(error) => return Err(RealizationError::Execution(error.to_string())),
                }
            } else if matches!(
                policy,
                RealizationPolicy::CpuJit {
                    fallback_to_interpreter: true
                }
            ) {
                backend = ItemBackend::JitFallback;
                interpret_item(graph, item, inputs, &values).map_err(RealizationError::Execution)?
            } else {
                return Err(RealizationError::Unsupported(format!(
                    "item {} cannot use native CPU JIT with materialized dependencies",
                    item.id
                )));
            }
        } else {
            interpret_item(graph, item, inputs, &values).map_err(RealizationError::Execution)?
        };
        values.insert(item.output.id, value);
        trace.items.push(ItemTrace {
            item: item.id,
            dependencies: item.dependencies.clone(),
            backend,
            cache_key: item.cache_key,
            materialized_buffer: item.output.id,
            last_consumer: item.consumers.last().copied(),
        });
    }
    let outputs = requested
        .iter()
        .map(|node| {
            values
                .get(&(node.index() as u64))
                .cloned()
                .ok_or(RealizationError::MissingBuffer(node.index() as u64))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Realized { outputs, trace })
}

/// Convenience entry point for the internal lazy path. Scheduling is repeated
/// for each call so symbolic bindings remain concrete in both the schedule
/// descriptors and the executable kernel cache identity.
pub fn realize_graph(
    graph: &Graph,
    requested: &[NodeId],
    inputs: &HashMap<String, TensorData>,
    policy: RealizationPolicy,
) -> Result<Realized, RealizationError> {
    let schedule = crate::schedule_many(graph, requested)
        .map_err(|error| RealizationError::Schedule(error.to_string()))?;
    realize(graph, &schedule, requested, inputs, policy)
}

fn interpret_item(
    graph: &Graph,
    item: &crate::ScheduleItem,
    inputs: &HashMap<String, TensorData>,
    values: &HashMap<u64, TensorData>,
) -> Result<TensorData, String> {
    if matches!(graph.op(item.node), Ok(Op::Reduce { .. })) && item.dependencies.is_empty() {
        return crate::execute_elementwise(graph, item.node, inputs).map_err(|e| e.to_string());
    }
    let mut bindings = KernelBindings::default();
    for desc in &item.inputs {
        let id = NodeId::from_index(desc.id as usize);
        let value = if let Some(value) = values.get(&desc.id) {
            value.clone()
        } else {
            match graph.op(id).map_err(|e| e.to_string())? {
                Op::Input { name } => inputs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("missing input {name}"))?,
                Op::Constant(value) => value.clone(),
                _ => return Err(format!("missing materialized buffer {}", desc.id)),
            }
        };
        let role = if matches!(graph.op(id), Ok(Op::Constant(_))) {
            BufferRole::Constant
        } else {
            BufferRole::Input
        };
        let kernel_desc =
            KernelBufferDesc::concrete(desc.id, role, desc.shape.clone(), desc.dtype, false)
                .map_err(|e| e.to_string())?;
        bindings
            .insert(&kernel_desc, value)
            .map_err(|e| e.to_string())?;
    }
    crate::kernel::execute_lowered_elementwise(&item.kernel, &bindings).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, ReduceKind, Scalar, Shape, TensorData};

    #[test]
    fn realizes_reduction_boundary_without_recomputing_shared_producers() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2, 3]), DType::F32);
        let y = graph.input_dtype("y", Shape::from([1, 3]), DType::F32);
        let producer = graph.add(x, y).unwrap();
        let sum = graph
            .reduce(producer, ReduceKind::Mean, Some(vec![1]), false)
            .unwrap();
        let two = graph.constant(TensorData::scalar(2.0));
        let output = graph.mul(sum, two).unwrap();
        let inputs = HashMap::from([
            (
                "x".into(),
                TensorData::from_scalars(
                    [2, 3],
                    DType::F32,
                    (0..6).map(|value| Scalar::F(value as f64)),
                )
                .unwrap(),
            ),
            (
                "y".into(),
                TensorData::from_scalars(
                    [1, 3],
                    DType::F32,
                    [Scalar::F(1.0), Scalar::F(2.0), Scalar::F(3.0)],
                )
                .unwrap(),
            ),
        ]);
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        assert_eq!(schedule.items.len(), 2);
        assert_eq!(schedule.items[1].dependencies, vec![schedule.items[0].id]);
        let actual = realize(
            &graph,
            &schedule,
            &[output],
            &inputs,
            RealizationPolicy::Interpreter,
        )
        .unwrap();
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        assert_eq!(actual.outputs[0].storage(), expected.storage());
        assert_eq!(actual.trace.items.len(), 2);
        assert!(
            actual
                .trace
                .items
                .iter()
                .all(|entry| entry.backend == ItemBackend::Interpreter)
        );
        let fallback = realize(
            &graph,
            &schedule,
            &[output],
            &inputs,
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: true,
            },
        )
        .unwrap();
        assert_eq!(fallback.outputs[0].storage(), expected.storage());
        assert_eq!(fallback.trace.items[1].backend, ItemBackend::JitFallback);
    }

    #[test]
    fn diamond_is_materialized_once_and_native_jit_is_explicit() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2]));
        let producer = graph.square(x).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(producer, one).unwrap();
        let right = graph.mul(producer, one).unwrap();
        let inputs = HashMap::from([("x".into(), TensorData::new([2], vec![2.0, 3.0]).unwrap())]);
        let schedule = crate::schedule_many(&graph, &[left, right]).unwrap();
        assert_eq!(schedule.items.len(), 3);
        assert_eq!(schedule.items[0].consumers.len(), 2);
        assert_eq!(schedule.internal_temporaries(&[left, right]).len(), 1);
        let actual = realize(
            &graph,
            &schedule,
            &[left, right],
            &inputs,
            RealizationPolicy::Interpreter,
        )
        .unwrap();
        for (output, expected_node) in actual.outputs.iter().zip([left, right]) {
            assert_eq!(
                output.storage(),
                CpuBackend
                    .execute(&graph, expected_node, &inputs)
                    .unwrap()
                    .storage()
            );
        }

        let direct = crate::schedule(&graph, producer).unwrap();
        let native = realize(
            &graph,
            &direct,
            &[producer],
            &inputs,
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
        )
        .unwrap();
        assert_eq!(native.trace.items[0].backend, ItemBackend::NativeJit);
    }

    #[test]
    fn malformed_dependencies_and_bindings_fail_before_silent_execution() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2]));
        let y = graph.neg(x).unwrap();
        let inputs = HashMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
        let mut schedule = crate::schedule(&graph, y).unwrap();
        schedule.items[0].dependencies.push(99);
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[y],
                &inputs,
                RealizationPolicy::Interpreter
            ),
            Err(RealizationError::Schedule(_))
        ));

        let mut schedule = crate::schedule(&graph, y).unwrap();
        schedule.items[0].inputs[0].shape = Shape::from([3]);
        assert!(matches!(
            realize(
                &graph,
                &schedule,
                &[y],
                &inputs,
                RealizationPolicy::Interpreter
            ),
            Err(RealizationError::Execution(_))
        ));
    }
}
