//! Typed graph nodes whose concrete output extent is known only at realization.

use super::{BinaryOp, Graph, NodeId, UnaryOp};
use crate::{DType, Error, Result, Shape, TensorData};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

/// Identifier in a graph's dynamic-result arena. It cannot be used where a
/// static [`NodeId`] is required.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DynamicNodeId {
    pub(crate) graph: u64,
    pub(crate) index: usize,
}

/// The static part of a data-dependent output shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynamicOutputShape {
    rank: usize,
}
/// A dynamic operand is either another arena result or a scalar static node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DynamicInput {
    Dynamic(DynamicNodeId),
    StaticScalar(NodeId),
}

/// One statically described value consumed by a dynamic cardinality count
/// stage.  The node identity is graph-local; its descriptor is the complete
/// static ABI expected before counting or allocating.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DynamicBinding {
    pub node: NodeId,
    pub shape: Shape,
    pub dtype: DType,
    pub bytes: usize,
}

/// The first count stage supported by the exact dynamic allocation contract.
/// More dynamic graph operators remain CPU-oracle-only until they have an
/// equally explicit allocation and lowering contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DynamicCountStage {
    MaskedSelect { input: NodeId, mask: NodeId },
}

/// Target requested for a dynamic allocation plan.  Only the CPU interpreter
/// owns a concrete execution path in this first exact-cardinality slice.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DynamicAllocationTarget {
    CpuInterpreter,
    RuntimeSchedule,
    NativeCpuJit,
    Device,
    Schedule,
    Capture,
    Artifact,
    Replay,
}

/// Checked exact allocation metadata. It describes an owned dense result and
/// never encodes a maximum capacity, sentinel bound, or placeholder storage.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DynamicAllocation {
    pub shape: Shape,
    pub dtype: DType,
    pub elements: usize,
    pub bytes: usize,
}

/// Immutable, graph-owned plan for one exact runtime-cardinality output.
///
/// The plan separates count production from dense result allocation. It is not
/// a second scheduler: its ordered bindings are the static values the count
/// stage needs, while the CPU semantic executor supplies their realized bytes.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DynamicAllocationPlan {
    output: DynamicNodeId,
    count_stage: DynamicCountStage,
    bindings: Vec<DynamicBinding>,
    output_dtype: DType,
    output_rank: usize,
    identity: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DynamicAllocationError {
    UnsupportedOutput { output: DynamicNodeId },
    InvalidBinding {
        node: NodeId,
        expected_shape: Shape,
        actual_shape: Shape,
        expected_dtype: DType,
        actual_dtype: DType,
    },
    AllocationOverflow { elements: usize, dtype: DType },
    UnsupportedTarget(DynamicAllocationTarget),
}

impl fmt::Display for DynamicAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dynamic allocation error: {self:?}")
    }
}
impl std::error::Error for DynamicAllocationError {}

impl DynamicAllocationPlan {
    pub(crate) fn for_output(
        graph: &Graph,
        output: DynamicNodeId,
    ) -> std::result::Result<Self, DynamicAllocationError> {
        let node = graph
            .dynamic_node(output)
            .map_err(|_| DynamicAllocationError::UnsupportedOutput { output })?;
        let DynamicOp::MaskedSelect { input, mask } = &node.op else {
            return Err(DynamicAllocationError::UnsupportedOutput { output });
        };
        let bindings = [*input, *mask]
            .into_iter()
            .map(|source| {
                let value = graph
                    .node(source)
                    .map_err(|_| DynamicAllocationError::UnsupportedOutput { output })?;
                let elements = value
                    .shape
                    .numel()
                    .map_err(|_| DynamicAllocationError::UnsupportedOutput { output })?;
                let bytes = elements.checked_mul(value.dtype.itemsize()).ok_or(
                    DynamicAllocationError::AllocationOverflow {
                        elements,
                        dtype: value.dtype,
                    },
                )?;
                Ok(DynamicBinding {
                    node: source,
                    shape: value.shape.clone(),
                    dtype: value.dtype,
                    bytes,
                })
            })
            .collect::<std::result::Result<Vec<_>, DynamicAllocationError>>()?;
        let mut plan = Self {
            output,
            count_stage: DynamicCountStage::MaskedSelect {
                input: *input,
                mask: *mask,
            },
            bindings,
            output_dtype: node.dtype,
            output_rank: node.output.rank(),
            identity: 0,
        };
        plan.identity = plan.logical_identity();
        Ok(plan)
    }

    pub fn output(&self) -> DynamicNodeId {
        self.output
    }

    pub fn count_stage(&self) -> DynamicCountStage {
        self.count_stage
    }

    pub fn bindings(&self) -> &[DynamicBinding] {
        &self.bindings
    }

    pub fn output_dtype(&self) -> DType {
        self.output_dtype
    }

    pub fn output_rank(&self) -> usize {
        self.output_rank
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Rejects routes that cannot retain an exact runtime extent before they
    /// schedule, capture, compile, submit, or allocate output storage.
    pub fn validate_target(
        &self,
        target: DynamicAllocationTarget,
    ) -> std::result::Result<(), DynamicAllocationError> {
        if matches!(
            target,
            DynamicAllocationTarget::CpuInterpreter | DynamicAllocationTarget::RuntimeSchedule
        ) {
            Ok(())
        } else {
            Err(DynamicAllocationError::UnsupportedTarget(target))
        }
    }

    /// Validates the statically described count-stage inputs before counting.
    pub fn validate_bindings(
        &self,
        input: &TensorData,
        mask: &TensorData,
    ) -> std::result::Result<(), DynamicAllocationError> {
        for (binding, value) in self.bindings.iter().zip([input, mask]) {
            if value.shape() != &binding.shape || value.dtype() != binding.dtype {
                return Err(DynamicAllocationError::InvalidBinding {
                    node: binding.node,
                    expected_shape: binding.shape.clone(),
                    actual_shape: value.shape().clone(),
                    expected_dtype: binding.dtype,
                    actual_dtype: value.dtype(),
                });
            }
        }
        Ok(())
    }

    /// Converts a validated exact count into the owned dense result ABI.
    pub fn allocation_for_count(
        &self,
        elements: usize,
    ) -> std::result::Result<DynamicAllocation, DynamicAllocationError> {
        let bytes = elements.checked_mul(self.output_dtype.itemsize()).ok_or(
            DynamicAllocationError::AllocationOverflow {
                elements,
                dtype: self.output_dtype,
            },
        )?;
        let shape = Shape::from([elements]);
        debug_assert_eq!(shape.rank(), self.output_rank);
        Ok(DynamicAllocation {
            shape,
            dtype: self.output_dtype,
            elements,
            bytes,
        })
    }

    fn logical_identity(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.output.index.hash(&mut hasher);
        self.count_stage.hash(&mut hasher);
        self.bindings.hash(&mut hasher);
        self.output_dtype.hash(&mut hasher);
        self.output_rank.hash(&mut hasher);
        hasher.finish()
    }
}

impl DynamicOutputShape {
    pub const fn new(rank: usize) -> Self {
        Self { rank }
    }
    pub const fn rank(self) -> usize {
        self.rank
    }
    pub fn validate(self, shape: &Shape) -> Result<()> {
        if shape.rank() == self.rank {
            Ok(())
        } else {
            Err(Error::InvalidIndex)
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum DynamicOp {
    Nonzero {
        input: NodeId,
    },
    MaskedSelect {
        input: NodeId,
        mask: NodeId,
    },
    Sum {
        input: DynamicNodeId,
    },
    Unary {
        op: UnaryOp,
        input: DynamicNodeId,
    },
    Binary {
        op: BinaryOp,
        lhs: DynamicInput,
        rhs: DynamicInput,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicNode {
    pub op: DynamicOp,
    pub output: DynamicOutputShape,
    pub dtype: DType,
}

impl DynamicNode {
    pub(crate) fn nonzero(input: NodeId) -> Self {
        Self {
            op: DynamicOp::Nonzero { input },
            output: DynamicOutputShape::new(2),
            dtype: DType::I64,
        }
    }

    pub(crate) fn masked_select(input: NodeId, mask: NodeId, dtype: DType) -> Self {
        Self {
            op: DynamicOp::MaskedSelect { input, mask },
            output: DynamicOutputShape::new(1),
            dtype,
        }
    }

    pub(crate) fn sum(input: DynamicNodeId, dtype: DType) -> Self {
        Self {
            op: DynamicOp::Sum { input },
            output: DynamicOutputShape::new(0),
            dtype,
        }
    }
    pub(crate) fn unary(op: UnaryOp, input: DynamicNodeId, dtype: DType) -> Self {
        Self {
            op: DynamicOp::Unary { op, input },
            output: DynamicOutputShape::new(1),
            dtype,
        }
    }
    pub(crate) fn binary(op: BinaryOp, lhs: DynamicInput, rhs: DynamicInput, dtype: DType) -> Self {
        Self {
            op: DynamicOp::Binary { op, lhs, rhs },
            output: DynamicOutputShape::new(1),
            dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scalar, TensorData};

    fn masked_select_plan() -> (Graph, DynamicNodeId) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
        let output = graph.masked_select_dynamic(input, mask).unwrap();
        (graph, output)
    }

    #[test]
    fn masked_select_allocation_plan_is_ordered_and_graph_identity_independent() {
        let (graph, output) = masked_select_plan();
        let plan = graph.dynamic_allocation_plan(output).unwrap();
        assert_eq!(
            plan.count_stage(),
            DynamicCountStage::MaskedSelect {
                input: plan.bindings()[0].node,
                mask: plan.bindings()[1].node,
            }
        );
        assert_eq!(plan.bindings().len(), 2);
        assert_eq!(plan.bindings()[0].shape, Shape::from([2, 2]));
        assert_eq!(plan.bindings()[1].shape, Shape::from([1, 2]));
        assert_eq!(plan.output_dtype(), DType::F32);
        assert_eq!(plan.output_rank(), 1);
        assert_eq!(plan.allocation_for_count(0).unwrap().bytes, 0);
        assert_eq!(plan.allocation_for_count(3).unwrap().shape, Shape::from([3]));

        let (equivalent, equivalent_output) = masked_select_plan();
        assert_eq!(
            plan.identity(),
            equivalent.dynamic_allocation_plan(equivalent_output).unwrap().identity()
        );
    }

    #[test]
    fn allocation_plan_rejects_non_cpu_targets_and_invalid_bindings_before_counting() {
        let (graph, output) = masked_select_plan();
        let plan = graph.dynamic_allocation_plan(output).unwrap();
        assert_eq!(
            plan.validate_target(DynamicAllocationTarget::Capture),
            Err(DynamicAllocationError::UnsupportedTarget(
                DynamicAllocationTarget::Capture
            ))
        );
        let input = TensorData::from_scalars([2, 2], DType::F32, [Scalar::F(0.0); 4]).unwrap();
        let wrong_mask = TensorData::from_scalars([2, 2], DType::Bool, [Scalar::Bool(true); 4])
            .unwrap();
        assert!(matches!(
            plan.validate_bindings(&input, &wrong_mask),
            Err(DynamicAllocationError::InvalidBinding { .. })
        ));
        assert!(matches!(
            plan.allocation_for_count(usize::MAX),
            Err(DynamicAllocationError::AllocationOverflow { .. })
        ));
    }
}
