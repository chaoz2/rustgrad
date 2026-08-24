//! Immutable schedule capture and backend-neutral interpreter replay.
use crate::{
    BufferRole, Graph, KernelBindings, KernelBufferDesc, NodeId, Op, Schedule, ScheduleItem,
    TensorData,
};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReplayInput {
    pub name: String,
    pub node: NodeId,
    pub desc: crate::BufferDesc,
}
#[derive(Clone, Debug)]
pub struct CapturedSchedule {
    pub items: Vec<ScheduleItem>,
    pub inputs: Vec<ReplayInput>,
    pub constants: BTreeMap<u64, TensorData>,
    pub requested: Vec<u64>,
    pub identity: u64,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    Missing(String),
    Extra(String),
    Descriptor(String),
    Corrupt(String),
    Execute(String),
}
impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "replay error: {self:?}")
    }
}
impl std::error::Error for ReplayError {}
impl CapturedSchedule {
    pub fn capture(
        graph: &Graph,
        schedule: &Schedule,
        requested: &[NodeId],
    ) -> Result<Self, ReplayError> {
        let mut inputs = BTreeMap::new();
        let mut constants = BTreeMap::new();
        for item in &schedule.items {
            item.validate_input_bindings()
                .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
            for b in item.ordered_inputs() {
                match graph
                    .op(b.input_node)
                    .map_err(|e| ReplayError::Corrupt(e.to_string()))?
                {
                    Op::Input { name } => {
                        inputs.entry(name.clone()).or_insert(ReplayInput {
                            name: name.clone(),
                            node: b.input_node,
                            desc: b.desc.clone(),
                        });
                    }
                    Op::Constant(v) => {
                        constants.insert(b.desc.id, v.clone());
                    }
                    _ => {}
                }
            }
        }
        let inputs = inputs.into_values().collect::<Vec<_>>();
        let mut h = DefaultHasher::new();
        for i in &schedule.items {
            i.cache_key.hash(&mut h);
            i.dependencies.hash(&mut h);
            i.input_bindings.hash(&mut h);
        }
        inputs.hash(&mut h);
        requested.hash(&mut h);
        Ok(Self {
            items: schedule.items.clone(),
            inputs,
            constants,
            requested: requested.iter().map(|n| n.index() as u64).collect(),
            identity: h.finish(),
        })
    }
    pub fn replay(
        &self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<Vec<TensorData>, ReplayError> {
        let expected = self
            .inputs
            .iter()
            .map(|i| i.name.as_str())
            .collect::<BTreeSet<_>>();
        if provided.keys().any(|n| !expected.contains(n.as_str())) {
            return Err(ReplayError::Extra("input".into()));
        }
        let mut values = self.constants.clone();
        let mut done = BTreeSet::new();
        for item in &self.items {
            if item.dependencies.iter().any(|d| !done.contains(d)) {
                return Err(ReplayError::Corrupt(format!("dependency for {}", item.id)));
            }
            let mut bindings = KernelBindings::default();
            for b in item.ordered_inputs() {
                let value = values
                    .get(&b.desc.id)
                    .cloned()
                    .or_else(|| {
                        self.inputs
                            .iter()
                            .find(|i| i.node == b.input_node)
                            .and_then(|i| provided.get(&i.name).cloned())
                    })
                    .ok_or_else(|| ReplayError::Missing(b.desc.id.to_string()))?;
                if value.shape() != &b.desc.shape || value.dtype() != b.desc.dtype {
                    return Err(ReplayError::Descriptor(b.desc.id.to_string()));
                }
                let role = if self.constants.contains_key(&b.desc.id) {
                    BufferRole::Constant
                } else {
                    BufferRole::Input
                };
                let desc = KernelBufferDesc::concrete(
                    b.desc.id,
                    role,
                    b.desc.shape.clone(),
                    b.desc.dtype,
                    false,
                )
                .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
                bindings
                    .insert(&desc, value)
                    .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
            }
            let value = crate::kernel::execute_lowered_elementwise(&item.kernel, &bindings)
                .map_err(|e| ReplayError::Execute(e.to_string()))?;
            values.insert(item.output.id, value);
            done.insert(item.id);
        }
        self.requested
            .iter()
            .map(|id| {
                values
                    .get(id)
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing(id.to_string()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape};
    #[test]
    fn capture_replays_without_graph_traversal() {
        let mut g = Graph::new();
        let x = g.input_dtype("x", Shape::from([3]), DType::F32);
        let y = g.square(x).unwrap();
        let s = crate::schedule(&g, y).unwrap();
        let c = CapturedSchedule::capture(&g, &s, &[y]).unwrap();
        let a = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [Scalar::F(1.), Scalar::F(2.), Scalar::F(3.)],
            )
            .unwrap(),
        )]);
        let out = c.replay(&a).unwrap();
        let oracle = CpuBackend
            .execute(
                &g,
                y,
                &a.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            )
            .unwrap();
        assert_eq!(out[0].storage(), oracle.storage());
        assert!(matches!(
            c.replay(&BTreeMap::new()),
            Err(ReplayError::Missing(_))
        ));
    }
}
