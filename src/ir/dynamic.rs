//! Typed graph nodes whose concrete output extent is known only at realization.

use super::{BinaryOp, Graph, NodeId, ReduceKind, ReductionDType, UnaryOp};
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

/// The statically known expression for one data-dependent output shape.
///
/// The producing graph node is the count provenance; the concrete count value
/// remains absent until realization. The variant states how that tagged count
/// maps to a concrete tensor shape once available.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DynamicOutputShape {
    Scalar,
    Count1d { count: DynamicNodeId },
    CountRows { count: DynamicNodeId, width: usize },
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

/// One graph-owned count stage for an exact dynamic allocation contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum DynamicCountStage {
    Nonzero {
        input: DynamicBinding,
    },
    MaskedSelect {
        input: DynamicBinding,
        mask: DynamicBinding,
    },
}

#[derive(Hash)]
enum LegacyMaskedSelectCountStage {
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
    output_dtype: DType,
    output_shape: DynamicOutputShape,
    identity: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DynamicAllocationError {
    UnsupportedOutput {
        output: DynamicNodeId,
    },
    InvalidBinding {
        node: NodeId,
        expected_shape: Shape,
        actual_shape: Shape,
        expected_dtype: DType,
        actual_dtype: DType,
    },
    AllocationOverflow {
        elements: usize,
        dtype: DType,
    },
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
        let binding = |source| {
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
        };
        let count_stage = match &node.operation {
            DynamicOperation::Nonzero { input } => DynamicCountStage::Nonzero {
                input: binding(*input)?,
            },
            DynamicOperation::MaskedSelect { input, mask } => DynamicCountStage::MaskedSelect {
                input: binding(*input)?,
                mask: binding(*mask)?,
            },
            _ => return Err(DynamicAllocationError::UnsupportedOutput { output }),
        };
        let mut plan = Self {
            output,
            count_stage,
            output_dtype: node.dtype,
            output_shape: node.output,
            identity: 0,
        };
        plan.identity = plan.logical_identity();
        Ok(plan)
    }

    pub fn output(&self) -> DynamicNodeId {
        self.output
    }

    pub fn count_stage(&self) -> &DynamicCountStage {
        &self.count_stage
    }

    pub fn bindings(&self) -> Vec<&DynamicBinding> {
        match &self.count_stage {
            DynamicCountStage::Nonzero { input } => vec![input],
            DynamicCountStage::MaskedSelect { input, mask } => vec![input, mask],
        }
    }

    pub fn output_dtype(&self) -> DType {
        self.output_dtype
    }

    pub fn output_rank(&self) -> usize {
        self.output_shape.rank()
    }

    pub fn output_shape(&self) -> DynamicOutputShape {
        self.output_shape
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
        values: &[&TensorData],
    ) -> std::result::Result<(), DynamicAllocationError> {
        let bindings = self.bindings();
        if values.len() != bindings.len() {
            return Err(DynamicAllocationError::UnsupportedOutput {
                output: self.output,
            });
        }
        for (binding, value) in bindings.into_iter().zip(values.iter().copied()) {
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
        let shape = self.output_shape.resolve(elements).map_err(|_| {
            DynamicAllocationError::AllocationOverflow {
                elements,
                dtype: self.output_dtype,
            }
        })?;
        let output_elements =
            shape
                .numel()
                .map_err(|_| DynamicAllocationError::AllocationOverflow {
                    elements,
                    dtype: self.output_dtype,
                })?;
        let bytes = output_elements
            .checked_mul(self.output_dtype.itemsize())
            .ok_or(DynamicAllocationError::AllocationOverflow {
                elements: output_elements,
                dtype: self.output_dtype,
            })?;
        Ok(DynamicAllocation {
            shape,
            dtype: self.output_dtype,
            elements: output_elements,
            bytes,
        })
    }

    fn logical_identity(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.output.index.hash(&mut hasher);
        match &self.count_stage {
            DynamicCountStage::MaskedSelect { input, mask } => {
                LegacyMaskedSelectCountStage::MaskedSelect {
                    input: input.node,
                    mask: mask.node,
                }
                .hash(&mut hasher);
                vec![input.clone(), mask.clone()].hash(&mut hasher);
            }
            DynamicCountStage::Nonzero { input } => {
                "runtime-nonzero-plan-v1".hash(&mut hasher);
                input.hash(&mut hasher);
            }
        }
        self.output_dtype.hash(&mut hasher);
        self.output_shape.rank().hash(&mut hasher);
        hasher.finish()
    }
}

impl DynamicOutputShape {
    pub const fn scalar() -> Self {
        Self::Scalar
    }
    pub const fn count_1d(count: DynamicNodeId) -> Self {
        Self::Count1d { count }
    }
    pub const fn count_rows(count: DynamicNodeId, width: usize) -> Self {
        Self::CountRows { count, width }
    }
    pub const fn rank(self) -> usize {
        match self {
            Self::Scalar => 0,
            Self::Count1d { .. } => 1,
            Self::CountRows { .. } => 2,
        }
    }
    pub(crate) fn resolve(self, count: usize) -> Result<Shape> {
        let shape = match self {
            Self::Scalar => Shape::from([]),
            Self::Count1d { .. } => Shape::from([count]),
            Self::CountRows { width, .. } => Shape::from([count, width]),
        };
        shape.numel()?;
        Ok(shape)
    }
    pub fn validate(self, shape: &Shape) -> Result<()> {
        let valid = match self {
            Self::Scalar => shape.dims().is_empty(),
            Self::Count1d { .. } => shape.rank() == 1,
            Self::CountRows { width, .. } => {
                shape.rank() == 2 && shape.dims().get(1) == Some(&width)
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidIndex)
        }
    }

    /// Validates both the static expression and its exact realized count.
    pub fn validate_for_count(self, count: usize, shape: &Shape) -> Result<()> {
        match self.resolve(count) {
            Ok(expected) if &expected == shape => Ok(()),
            _ => Err(Error::InvalidIndex),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum DynamicOperation {
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
    Mean {
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

pub(crate) fn dynamic_reduction_dtypes(input: DType, op: ReduceKind) -> Option<ReductionDType> {
    match op {
        ReduceKind::Sum => Some(ReductionDType::sum_default(input)),
        ReduceKind::Mean => {
            let output = if input.is_float() { input } else { DType::F32 };
            Some(ReductionDType::new(input.sum_accumulator_dtype(), output))
        }
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicNode {
    pub operation: DynamicOperation,
    pub output: DynamicOutputShape,
    pub dtype: DType,
}

impl DynamicNode {
    pub(crate) fn nonzero(id: DynamicNodeId, input: NodeId, rank: usize, dtype: DType) -> Self {
        Self {
            operation: DynamicOperation::Nonzero { input },
            output: DynamicOutputShape::count_rows(id, rank),
            dtype,
        }
    }

    pub(crate) fn masked_select(
        id: DynamicNodeId,
        input: NodeId,
        mask: NodeId,
        dtype: DType,
    ) -> Self {
        Self {
            operation: DynamicOperation::MaskedSelect { input, mask },
            output: DynamicOutputShape::count_1d(id),
            dtype,
        }
    }

    pub(crate) fn sum(input: DynamicNodeId, dtype: DType) -> Self {
        Self {
            operation: DynamicOperation::Sum { input },
            output: DynamicOutputShape::scalar(),
            dtype,
        }
    }
    pub(crate) fn mean(input: DynamicNodeId, dtype: DType) -> Self {
        Self {
            operation: DynamicOperation::Mean { input },
            output: DynamicOutputShape::scalar(),
            dtype,
        }
    }
    pub(crate) fn unary(
        op: UnaryOp,
        input: DynamicNodeId,
        output: DynamicOutputShape,
        dtype: DType,
    ) -> Self {
        Self {
            operation: DynamicOperation::Unary { op, input },
            output,
            dtype,
        }
    }
    pub(crate) fn binary(
        op: BinaryOp,
        lhs: DynamicInput,
        rhs: DynamicInput,
        output: DynamicOutputShape,
        dtype: DType,
    ) -> Self {
        Self {
            operation: DynamicOperation::Binary { op, lhs, rhs },
            output,
            dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Op, Scalar, TensorData};

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
        assert!(matches!(
            plan.count_stage(),
            DynamicCountStage::MaskedSelect { input, mask }
                if input.node == plan.bindings()[0].node && mask.node == plan.bindings()[1].node
        ));
        assert_eq!(plan.bindings().len(), 2);
        assert_eq!(plan.bindings()[0].shape, Shape::from([2, 2]));
        assert_eq!(plan.bindings()[1].shape, Shape::from([1, 2]));
        assert_eq!(plan.output_dtype(), DType::F32);
        assert_eq!(plan.output_rank(), 1);
        assert_eq!(plan.allocation_for_count(0).unwrap().bytes, 0);
        assert_eq!(
            plan.allocation_for_count(3).unwrap().shape,
            Shape::from([3])
        );
        assert!(
            plan.output_shape()
                .validate_for_count(3, &Shape::from([3]))
                .is_ok()
        );
        assert!(
            plan.output_shape()
                .validate_for_count(2, &Shape::from([3]))
                .is_err()
        );

        let (equivalent, equivalent_output) = masked_select_plan();
        assert_eq!(
            plan.identity(),
            equivalent
                .dynamic_allocation_plan(equivalent_output)
                .unwrap()
                .identity()
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
        let wrong_mask =
            TensorData::from_scalars([2, 2], DType::Bool, [Scalar::Bool(true); 4]).unwrap();
        assert!(matches!(
            plan.validate_bindings(&[&input, &wrong_mask]),
            Err(DynamicAllocationError::InvalidBinding { .. })
        ));
        assert!(matches!(
            plan.allocation_for_count(usize::MAX),
            Err(DynamicAllocationError::AllocationOverflow { .. })
        ));
    }

    #[test]
    fn nonzero_coordinate_width_checks_final_i64_extent_atomically() {
        let Ok(maximum_i64_extent) = usize::try_from(i64::MAX) else {
            return;
        };
        let Some(final_valid_extent) = maximum_i64_extent.checked_add(1) else {
            return;
        };
        let Some(first_invalid_extent) = final_valid_extent.checked_add(1) else {
            return;
        };

        let mut graph = Graph::new();
        let valid = graph.input_dtype("valid", [final_valid_extent, 0], DType::F32);
        let output = graph.nonzero(valid).unwrap();
        assert_eq!(graph.dynamic_node(output).unwrap().dtype, DType::I64);
        assert_eq!(graph.dynamic_nodes.len(), 1);

        let invalid = graph.input_dtype("invalid", [first_invalid_extent, 0], DType::F32);
        let before = graph.dynamic_nodes.len();
        assert!(matches!(
            graph.nonzero(invalid),
            Err(Error::ShapeOverflow(shape))
                if shape == Shape::from([first_invalid_extent, 0])
        ));
        assert_eq!(graph.dynamic_nodes.len(), before);

        let mut fixed = Graph::new();
        let valid = fixed.input_dtype("fixed_valid", [final_valid_extent, 0], DType::F32);
        let before = fixed.node_count();
        let descriptor_only_size = 1usize << 26;
        let output = fixed
            .nonzero_fixed(valid, descriptor_only_size, Scalar::I(-7))
            .unwrap();
        assert_eq!(
            fixed.shape(output).unwrap(),
            &Shape::from([descriptor_only_size, 2])
        );
        assert_eq!(fixed.dtype(output).unwrap(), DType::I64);
        let Op::Expand {
            input: scalar,
            shape,
        } = fixed.op(output).unwrap()
        else {
            panic!("zero-domain fixed nonzero must remain a lazy scalar expansion");
        };
        assert_eq!(shape, &Shape::from([descriptor_only_size, 2]));
        let Op::Constant(value) = fixed.op(*scalar).unwrap() else {
            panic!("lazy fixed-nonzero fill must be backed by one scalar constant");
        };
        assert_eq!(value.shape(), &Shape::from([]));
        assert_eq!(value.dtype(), DType::I64);
        assert_eq!(value.scalar_at(0), Scalar::I(-7));
        assert_eq!(fixed.node_count(), before + 2);

        let mut scalar_fixed = Graph::new();
        let scalar = scalar_fixed.input_dtype("scalar", [], DType::F32);
        let before = scalar_fixed.node_count();
        let scalar_output = scalar_fixed.nonzero_fixed(scalar, 3, Scalar::I(9)).unwrap();
        assert_eq!(
            scalar_fixed.shape(scalar_output).unwrap(),
            &Shape::from([3, 0])
        );
        assert_eq!(scalar_fixed.dtype(scalar_output).unwrap(), DType::I32);
        assert!(matches!(
            scalar_fixed.op(scalar_output).unwrap(),
            Op::Expand { input, shape }
                if shape == &Shape::from([3, 0])
                    && matches!(scalar_fixed.op(*input).unwrap(), Op::Constant(value)
                        if value.shape() == &Shape::from([]))
        ));
        assert_eq!(scalar_fixed.node_count(), before + 2);

        let invalid = fixed.input_dtype("fixed_invalid", [first_invalid_extent, 0], DType::F32);
        let before = fixed.node_count();
        assert!(matches!(
            fixed.nonzero_fixed(invalid, 1, Scalar::I(-7)),
            Err(Error::ShapeOverflow(shape))
                if shape == Shape::from([first_invalid_extent, 0])
        ));
        assert_eq!(fixed.node_count(), before);
    }
}
