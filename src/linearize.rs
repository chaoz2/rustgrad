//! Typed late linearization of ranged UOps for portable lane renderers.
use crate::{DType, Shape, UArg, UOp, UOpKind};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LinearAccess {
    ContiguousVector,
    ScalarSplat,
    ScalarOnly(String),
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LinearBuffer {
    pub buffer: u64,
    pub dtype: DType,
    pub elements: usize,
    pub input_shape: Shape,
    pub byte_offset: usize,
    pub byte_stride: usize,
    pub alignment: usize,
    pub mutable: bool,
    pub access: LinearAccess,
}
#[derive(Clone, Debug)]
pub struct LinearKernel {
    /// Retained immutable source DAG; scalar UOp meaning is unchanged.
    pub source: UOp,
    pub output_buffer: u64,
    pub output_shape: Shape,
    pub dtype: DType,
    pub elements: usize,
    pub lanes: usize,
    pub vector_main: usize,
    pub scalar_tail: usize,
    pub tail_mask: Vec<bool>,
    pub buffers: Vec<LinearBuffer>,
    pub enabled: bool,
    pub reason: String,
    pub cache_key: u64,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinearizeError {
    MissingStore,
    Untyped,
    Overflow,
    Invalid(String),
}
impl fmt::Display for LinearizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "linearize error: {self:?}")
    }
}
impl std::error::Error for LinearizeError {}

impl LinearKernel {
    pub fn from_uop(source: &UOp) -> Result<Self, LinearizeError> {
        let nodes = source
            .topological()
            .map_err(|e| LinearizeError::Invalid(e.to_string()))?;
        let store = source
            .sources()
            .iter()
            .find(|node| matches!(node.kind(), UOpKind::Store))
            .ok_or(LinearizeError::MissingStore)?;
        let output = store
            .sources()
            .first()
            .ok_or(LinearizeError::MissingStore)?;
        let (output_buffer, elements, output_shape) = match output.arg() {
            UArg::BufferIndex {
                buffer,
                elements,
                output_shape,
                ..
            } => (*buffer, *elements, output_shape.clone()),
            _ => return Err(LinearizeError::MissingStore),
        };
        let dtype = output.ty().ok_or(LinearizeError::Untyped)?.scalar;
        let lanes = (16 / dtype.itemsize()).max(1);
        let mut enabled = lanes > 1;
        let mut reason = if enabled {
            "contiguous portable lane plan".to_string()
        } else {
            "64-bit scalar policy".to_string()
        };
        if nodes.iter().any(|node| {
            matches!(
                node.kind(),
                UOpKind::ReduceInit
                    | UOpKind::ReduceAccumulate
                    | UOpKind::ReduceFinalize
                    | UOpKind::Barrier
            )
        }) {
            enabled = false;
            reason = "reduction or effect requires scalar path".into();
        }
        let mut buffers = BTreeMap::new();
        for node in &nodes {
            let Some(ty) = node.ty() else { continue };
            let (buffer, count, input_shape, indexed_output, offset, contiguous) = match node.arg()
            {
                UArg::BufferIndex {
                    buffer,
                    elements,
                    input_shape,
                    output_shape,
                } => (
                    *buffer,
                    *elements,
                    input_shape.clone(),
                    output_shape.clone(),
                    0usize,
                    true,
                ),
                UArg::ViewBufferIndex {
                    buffer,
                    elements,
                    input_shape,
                    output_shape,
                    view,
                } => {
                    let contiguous = view.strides == view.logical_shape.contiguous_strides();
                    (
                        *buffer,
                        *elements,
                        input_shape.clone(),
                        output_shape.clone(),
                        view.offset,
                        contiguous,
                    )
                }
                _ => continue,
            };
            let byte_offset = offset
                .checked_mul(ty.scalar.itemsize())
                .ok_or(LinearizeError::Overflow)?;
            let access = if buffer == output_buffer {
                LinearAccess::ContiguousVector
            } else if count == 1 {
                LinearAccess::ScalarSplat
            } else if indexed_output != output_shape || input_shape != output_shape || !contiguous {
                enabled = false;
                reason = "varying broadcast, view, or non-contiguous index".into();
                LinearAccess::ScalarOnly(reason.clone())
            } else if byte_offset % (lanes * ty.scalar.itemsize()) != 0 {
                enabled = false;
                reason = "misaligned view byte offset".into();
                LinearAccess::ScalarOnly(reason.clone())
            } else {
                LinearAccess::ContiguousVector
            };
            buffers.entry(buffer).or_insert(LinearBuffer {
                buffer,
                dtype: ty.scalar,
                elements: count,
                input_shape,
                byte_offset,
                byte_stride: ty.scalar.itemsize(),
                alignment: ty.scalar.itemsize().max(1),
                mutable: buffer == output_buffer,
                access,
            });
        }
        let vector_main = if enabled { elements / lanes * lanes } else { 0 };
        let scalar_tail = elements
            .checked_sub(vector_main)
            .ok_or(LinearizeError::Overflow)?;
        let tail_mask = (0..lanes)
            .map(|lane| lane < scalar_tail)
            .collect::<Vec<_>>();
        let buffers = buffers.into_values().collect::<Vec<_>>();
        let mut h = DefaultHasher::new();
        output_buffer.hash(&mut h);
        output_shape.hash(&mut h);
        dtype.hash(&mut h);
        elements.hash(&mut h);
        lanes.hash(&mut h);
        vector_main.hash(&mut h);
        scalar_tail.hash(&mut h);
        tail_mask.hash(&mut h);
        buffers.hash(&mut h);
        enabled.hash(&mut h);
        reason.hash(&mut h);
        Ok(Self {
            source: source.clone(),
            output_buffer,
            output_shape,
            dtype,
            elements,
            lanes,
            vector_main,
            scalar_tail,
            tail_mask,
            buffers,
            enabled,
            reason,
            cache_key: h.finish(),
        })
    }
    pub fn validate(&self) -> Result<(), LinearizeError> {
        if self.lanes == 0 || self.tail_mask.len() != self.lanes {
            return Err(LinearizeError::Invalid("invalid lane mask".into()));
        }
        if self.vector_main.checked_add(self.scalar_tail) != Some(self.elements) {
            return Err(LinearizeError::Overflow);
        }
        if self.enabled && self.vector_main % self.lanes != 0 {
            return Err(LinearizeError::Invalid(
                "vector main is not lane aligned".into(),
            ));
        }
        if self.buffers.iter().filter(|buffer| buffer.mutable).count() != 1 {
            return Err(LinearizeError::Invalid(
                "requires exactly one mutable output".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape};
    #[test]
    fn snapshots_contiguous_and_varying_broadcast_plans() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([5]));
        let out = graph.square(x).unwrap();
        let plan =
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&graph, out).unwrap()).unwrap();
        plan.validate().unwrap();
        assert!(plan.enabled);
        assert_eq!((plan.lanes, plan.vector_main, plan.scalar_tail), (4, 4, 1));
        let mut broadcast = Graph::new();
        let a = broadcast.input("a", Shape::from([2, 3]));
        let b = broadcast.input("b", Shape::from([1, 3]));
        let out = broadcast.add(a, b).unwrap();
        let plan =
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&broadcast, out).unwrap())
                .unwrap();
        assert!(!plan.enabled);
        assert!(plan.reason.contains("varying"));

        let mut views = Graph::new();
        let x = views.input("x", Shape::from([8]));
        let aligned = views.shrink(x, vec![(4, 8)]).unwrap();
        let out = views.neg(aligned).unwrap();
        assert!(
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&views, out).unwrap())
                .unwrap()
                .enabled
        );
        let misaligned = views.shrink(x, vec![(1, 5)]).unwrap();
        let out = views.neg(misaligned).unwrap();
        let plan =
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&views, out).unwrap()).unwrap();
        assert!(!plan.enabled);
        assert!(plan.reason.contains("misaligned"));
    }
}
