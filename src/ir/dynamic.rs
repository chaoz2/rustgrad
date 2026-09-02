//! Typed graph nodes whose concrete output extent is known only at realization.

use super::{BinaryOp, Graph, NodeId, ReduceKind, ReductionDType, UnaryOp};
use crate::{DType, Error, Result, Scalar, Shape, TensorData};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
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

/// Immutable descriptor contract for one first-order dynamic-result VJP.
///
/// The upstream retains the exact runtime shape expression of `output`; no
/// maximum extent or static placeholder is introduced. The target is an
/// ordinary static graph value because a masked-selection VJP scatters the
/// compacted cotangent back into that value's fixed descriptor.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DynamicVjpPlan {
    output: DynamicNodeId,
    upstream_shape: DynamicOutputShape,
    upstream_dtype: DType,
    target: DynamicBinding,
    identity: u64,
}

/// Checked reverse rule for one runtime-cardinality Mean.
///
/// The concrete denominator deliberately remains absent: execution derives it
/// from the realized `input`. Keeping the canonical reduction dtype pair here
/// makes graph preflight and the CPU executor agree on the cast/divide/cast
/// boundary without adding another dynamic operation or schedule payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DynamicMeanVjpRule {
    input: DynamicNodeId,
    source_dtype: DType,
    dtypes: ReductionDType,
}

/// Checked graph composition for scattering one exact compacted cotangent
/// back into its fixed source descriptor.
///
/// The dynamic arena owns only the count provenance. Once an upstream with
/// that exact realized count is available, this rule lowers the inverse
/// compaction into ordinary graph operations. That keeps row-major placement,
/// broadcast-mask semantics, and higher-order edges in the existing indexing
/// and autograd contracts instead of duplicating them in a host-only loop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DynamicCompactionVjpRule {
    output: DynamicNodeId,
    input: DynamicBinding,
    mask: DynamicBinding,
}

pub(crate) struct DynamicCompactionVjpGraph {
    graph: Graph,
    gradient: NodeId,
    upstream: NodeId,
}

impl DynamicCompactionVjpGraph {
    pub(crate) fn into_parts(self) -> (Graph, NodeId, NodeId) {
        (self.graph, self.gradient, self.upstream)
    }
}

impl DynamicCompactionVjpRule {
    fn for_output(graph: &Graph, output: DynamicNodeId) -> Result<Self> {
        let node = graph.dynamic_node(output)?;
        let DynamicOperation::MaskedSelect { input, mask } = &node.operation else {
            return Err(Error::DynamicVjp {
                reason: "dynamic compaction VJP requires masked_select",
            });
        };
        let input = DynamicBinding::from_graph(graph, *input)?;
        let mask = DynamicBinding::from_graph(graph, *mask)?;
        if node.output != DynamicOutputShape::count_1d(output)
            || node.dtype != input.dtype
            || mask.dtype != DType::Bool
            || mask.shape.broadcast_with(&input.shape).as_ref() != Ok(&input.shape)
        {
            return Err(Error::DynamicVjp {
                reason: "dynamic compaction descriptor is not canonical",
            });
        }
        Ok(Self {
            output,
            input,
            mask,
        })
    }

    pub(crate) fn input(&self) -> NodeId {
        self.input.node
    }

    pub(crate) fn lower(
        &self,
        graph: &Graph,
        upstream: &TensorData,
        target: NodeId,
    ) -> Result<DynamicCompactionVjpGraph> {
        if Self::for_output(graph, self.output)? != *self
            || upstream.shape().rank() != 1
            || upstream.dtype() != self.input.dtype
        {
            return Err(Error::DynamicVjp {
                reason: "dynamic compaction upstream descriptor mismatch",
            });
        }
        let elements = self.input.shape.numel()?;
        let end =
            i64::try_from(elements).map_err(|_| Error::ShapeOverflow(self.input.shape.clone()))?;
        let selected = upstream.shape().dims()[0];
        if selected > elements {
            return Err(Error::DynamicVjp {
                reason: "dynamic compaction count exceeds source extent",
            });
        }

        // Rehearse the complete inverse compaction and any static-boundary VJP
        // on a clone. A late index, shape, or reverse-rule failure cannot
        // publish a partial graph into the caller's arena.
        let mut candidate = graph.clone();
        let indices = candidate.lazy_arange_default_int(0, end, 1)?;
        let indices = candidate.reshape(indices, self.input.shape.clone())?;
        let positions = candidate.masked_select(indices, self.mask.node, selected, Scalar::I(0))?;
        let upstream = candidate.constant(upstream.clone());
        let zeros = candidate.zeros_with_dtype([elements], self.input.dtype)?;
        let scattered = candidate.scatter_add(zeros, positions, upstream, 0)?;
        let local = candidate.reshape(scattered, self.input.shape.clone())?;
        let gradient = if self.input.node == target {
            local
        } else {
            candidate.grad_with(self.input.node, target, Some(local), true)?
        };
        Ok(DynamicCompactionVjpGraph {
            graph: candidate,
            gradient,
            upstream,
        })
    }
}

impl DynamicMeanVjpRule {
    pub(crate) fn input(self) -> DynamicNodeId {
        self.input
    }

    pub(crate) fn source_dtype(self) -> DType {
        self.source_dtype
    }

    pub(crate) fn dtypes(self) -> ReductionDType {
        self.dtypes
    }
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

impl DynamicBinding {
    fn from_graph(graph: &Graph, node: NodeId) -> Result<Self> {
        let value = graph.node(node)?;
        let elements = value.shape.numel()?;
        let bytes = elements
            .checked_mul(value.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(value.shape.clone()))?;
        Ok(Self {
            node,
            shape: value.shape.clone(),
            dtype: value.dtype,
            bytes,
        })
    }
}

impl DynamicVjpPlan {
    pub(crate) fn for_output(graph: &Graph, output: DynamicNodeId, target: NodeId) -> Result<Self> {
        let output_node = graph.dynamic_node(output)?;
        let target_node = graph.node(target)?;
        if !output_node.dtype.is_float() || output_node.dtype.is_float8() {
            return Err(Error::NonDifferentiableTarget(target));
        }
        if !target_node.dtype.is_float()
            || target_node.dtype.is_float8()
            || !target_node.requires_grad
        {
            return Err(Error::NonDifferentiableTarget(target));
        }
        let mut memo = BTreeMap::new();
        if !graph.validate_dynamic_vjp_path(output, target, &mut memo)? {
            return Err(Error::NonDifferentiableTarget(target));
        }
        let mut plan = Self {
            output,
            upstream_shape: output_node.output,
            upstream_dtype: output_node.dtype,
            target: DynamicBinding::from_graph(graph, target)?,
            identity: 0,
        };
        plan.identity = plan.logical_identity();
        Ok(plan)
    }

    pub fn output(&self) -> DynamicNodeId {
        self.output
    }

    pub fn upstream_shape(&self) -> DynamicOutputShape {
        self.upstream_shape
    }

    pub fn upstream_dtype(&self) -> DType {
        self.upstream_dtype
    }

    pub fn target(&self) -> &DynamicBinding {
        &self.target
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub(crate) fn validate_against(&self, graph: &Graph) -> Result<()> {
        if Self::for_output(graph, self.output, self.target.node)?.eq(self) {
            Ok(())
        } else {
            Err(Error::DynamicVjp {
                reason: "plan does not match graph",
            })
        }
    }

    pub(crate) fn validate_realized(
        &self,
        output: &TensorData,
        upstream: &TensorData,
    ) -> Result<()> {
        if output.dtype() != self.upstream_dtype
            || self.upstream_shape.validate(output.shape()).is_err()
        {
            return Err(Error::DynamicVjp {
                reason: "realized output descriptor mismatch",
            });
        }
        if upstream.shape() != output.shape() || upstream.dtype() != self.upstream_dtype {
            return Err(Error::DynamicVjp {
                reason: "upstream descriptor mismatch",
            });
        }
        Ok(())
    }

    fn logical_identity(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        "dynamic-vjp-plan-v1".hash(&mut hasher);
        self.output.index.hash(&mut hasher);
        match self.upstream_shape {
            DynamicOutputShape::Scalar => 0_u8.hash(&mut hasher),
            DynamicOutputShape::Count1d { count } => {
                1_u8.hash(&mut hasher);
                count.index.hash(&mut hasher);
            }
            DynamicOutputShape::CountRows { count, width } => {
                2_u8.hash(&mut hasher);
                count.index.hash(&mut hasher);
                width.hash(&mut hasher);
            }
        }
        self.upstream_dtype.hash(&mut hasher);
        self.target.hash(&mut hasher);
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

impl Graph {
    /// Builds the immutable descriptor contract for one first-order VJP of an
    /// exact-cardinality dynamic result into a static graph target.
    ///
    /// This is a read-only preflight: supported dynamic rules are checked and
    /// participating static boundaries rehearse `grad_with` on a private graph
    /// clone. The caller's arenas are unchanged on success or failure. Mask
    /// predicates are value-only cardinality inputs and never participate in
    /// the derivative path.
    pub fn dynamic_vjp_plan(
        &self,
        output: DynamicNodeId,
        target: NodeId,
    ) -> Result<DynamicVjpPlan> {
        DynamicVjpPlan::for_output(self, output, target)
    }

    pub(crate) fn dynamic_compaction_vjp_rule(
        &self,
        output: DynamicNodeId,
    ) -> Result<DynamicCompactionVjpRule> {
        DynamicCompactionVjpRule::for_output(self, output)
    }

    pub(crate) fn dynamic_backward_slice_contains(
        &self,
        output: DynamicNodeId,
        target: NodeId,
    ) -> Result<bool> {
        self.validate_dynamic_vjp_path(output, target, &mut BTreeMap::new())
    }

    fn validate_dynamic_input_vjp_path(
        &self,
        input: DynamicInput,
        target: NodeId,
        memo: &mut BTreeMap<usize, bool>,
    ) -> Result<bool> {
        match input {
            DynamicInput::Dynamic(input) => self.validate_dynamic_vjp_path(input, target, memo),
            DynamicInput::StaticScalar(input) => self.validate_static_vjp_boundary(input, target),
        }
    }

    fn validate_static_vjp_boundary(&self, boundary: NodeId, target: NodeId) -> Result<bool> {
        if !self.backward_slice_contains(boundary, target)? {
            return Ok(false);
        }
        let boundary_node = self.node(boundary)?;
        let shape = boundary_node.shape.clone();
        let dtype = boundary_node.dtype;
        let mut rehearsal = self.clone();
        let seed = rehearsal.lazy_full_with_dtype(shape, Scalar::F(1.0), dtype)?;
        rehearsal.grad_with(boundary, target, Some(seed), false)?;
        Ok(true)
    }

    fn validate_dynamic_vjp_path(
        &self,
        output: DynamicNodeId,
        target: NodeId,
        memo: &mut BTreeMap<usize, bool>,
    ) -> Result<bool> {
        let operation = self.dynamic_node(output)?.operation.clone();
        if let Some(contains) = memo.get(&output.index) {
            return Ok(*contains);
        }
        let contains = match operation {
            DynamicOperation::Nonzero { .. } => false,
            DynamicOperation::MaskedSelect { input, .. } => {
                self.validate_static_vjp_boundary(input, target)?
            }
            DynamicOperation::Sum { input } => {
                self.validate_dynamic_vjp_path(input, target, memo)?
            }
            DynamicOperation::Unary { op, input } => {
                if !matches!(op, UnaryOp::Neg | UnaryOp::Square) {
                    return Err(Error::DynamicVjp {
                        reason: "unsupported dynamic unary VJP",
                    });
                }
                self.validate_dynamic_vjp_path(input, target, memo)?
            }
            DynamicOperation::Mean { .. } => {
                let rule = self.dynamic_mean_vjp_rule(output)?;
                self.validate_dynamic_vjp_path(rule.input(), target, memo)?
            }
            DynamicOperation::Binary { op, lhs, rhs } => {
                if !matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) {
                    return Err(Error::DynamicVjp {
                        reason: "unsupported dynamic binary VJP",
                    });
                }
                let lhs = self.validate_dynamic_input_vjp_path(lhs, target, memo)?;
                let rhs = self.validate_dynamic_input_vjp_path(rhs, target, memo)?;
                lhs || rhs
            }
        };
        memo.insert(output.index, contains);
        Ok(contains)
    }

    /// Reconstructs and validates the only supported runtime-cardinality Mean
    /// reverse rule. The denominator is intentionally realized later.
    pub(crate) fn dynamic_mean_vjp_rule(
        &self,
        output: DynamicNodeId,
    ) -> Result<DynamicMeanVjpRule> {
        let node = self.dynamic_node(output)?;
        let DynamicOperation::Mean { input } = &node.operation else {
            return Err(Error::DynamicVjp {
                reason: "dynamic VJP node is not Mean",
            });
        };
        let input = *input;
        let source = self.dynamic_node(input)?;
        let dtypes =
            dynamic_reduction_dtypes(source.dtype, ReduceKind::Mean).ok_or(Error::DynamicVjp {
                reason: "dynamic Mean dtype policy is unsupported",
            })?;
        if node.output != DynamicOutputShape::Scalar || node.dtype != dtypes.output {
            return Err(Error::DynamicVjp {
                reason: "dynamic Mean descriptor is not canonical",
            });
        }
        Ok(DynamicMeanVjpRule {
            input,
            source_dtype: source.dtype,
            dtypes,
        })
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
    use crate::{Backend, CpuBackend, Op, Scalar, TensorData};
    use std::collections::HashMap;

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

    fn dynamic_vjp_graph() -> (Graph, NodeId, NodeId, DynamicNodeId) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let mask = graph.input_dtype("mask", [1, 3], DType::Bool);
        let output = graph.masked_select_dynamic(input, mask).unwrap();
        (graph, input, mask, output)
    }

    #[test]
    fn dynamic_vjp_plan_owns_exact_runtime_and_static_descriptors() {
        let (graph, input, _, output) = dynamic_vjp_graph();
        let plan = graph.dynamic_vjp_plan(output, input).unwrap();
        assert_eq!(plan.output(), output);
        assert_eq!(plan.upstream_dtype(), DType::F32);
        assert_eq!(plan.upstream_shape(), DynamicOutputShape::count_1d(output));
        assert_eq!(plan.target().node, input);
        assert_eq!(plan.target().shape, Shape::from([2, 3]));
        assert_eq!(plan.target().bytes, 6 * DType::F32.itemsize());

        let selected = TensorData::from_scalars(
            [4],
            DType::F32,
            [
                Scalar::F(1.0),
                Scalar::F(2.0),
                Scalar::F(3.0),
                Scalar::F(4.0),
            ],
        )
        .unwrap();
        assert!(plan.validate_realized(&selected, &selected).is_ok());
        assert!(
            plan.validate_realized(
                &selected,
                &TensorData::from_scalars([3], DType::F32, [Scalar::F(1.0); 3]).unwrap()
            )
            .is_err()
        );

        let (equivalent, equivalent_input, _, equivalent_output) = dynamic_vjp_graph();
        assert_eq!(
            plan.identity(),
            equivalent
                .dynamic_vjp_plan(equivalent_output, equivalent_input)
                .unwrap()
                .identity()
        );

        let mut tampered = plan.clone();
        tampered.identity ^= 1;
        assert!(tampered.validate_against(&graph).is_err());
    }

    #[test]
    fn compaction_vjp_lowers_to_duplicate_free_graph_scatter_with_higher_order_edges() {
        let (graph, input, _, output) = dynamic_vjp_graph();
        let rule = graph.dynamic_compaction_vjp_rule(output).unwrap();
        let upstream = TensorData::new([4], vec![2.0, 3.0, 5.0, 7.0]).unwrap();
        let lowered = rule.lower(&graph, &upstream, input).unwrap();
        let (mut derivative, gradient, upstream_node) = lowered.into_parts();
        let bindings = HashMap::from([
            (
                "input".into(),
                TensorData::new([2, 3], vec![11.0, 13.0, 17.0, 19.0, 23.0, 29.0]).unwrap(),
            ),
            (
                "mask".into(),
                TensorData::from_scalars(
                    [1, 3],
                    DType::Bool,
                    [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
                )
                .unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&derivative, gradient, &bindings)
                .unwrap(),
            TensorData::new([2, 3], vec![2.0, 0.0, 3.0, 5.0, 0.0, 7.0]).unwrap()
        );

        // The compacted cotangent is an ordinary graph value after count
        // realization. Target slicing therefore differentiates only through
        // Scatter updates; the Bool mask/index route remains non-value data.
        let sum = derivative.sum_all(gradient).unwrap();
        let second = derivative.gradient_default(sum, &[upstream_node]).unwrap()[0];
        assert_eq!(
            CpuBackend.execute(&derivative, second, &bindings).unwrap(),
            TensorData::new([4], vec![1.0; 4]).unwrap()
        );
    }

    #[test]
    fn compaction_vjp_preserves_supported_float_storage_and_rejects_impossible_counts() {
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [3], dtype);
            let mask = graph.input_dtype("mask", [], DType::Bool);
            let output = graph.masked_select_dynamic(input, mask).unwrap();
            let rule = graph.dynamic_compaction_vjp_rule(output).unwrap();
            let upstream = TensorData::from_scalars(
                [3],
                dtype,
                [Scalar::F(-0.0), Scalar::F(2.0), Scalar::F(f64::NAN)],
            )
            .unwrap();
            let (derivative, gradient, _) =
                rule.lower(&graph, &upstream, input).unwrap().into_parts();
            let actual = CpuBackend
                .execute(
                    &derivative,
                    gradient,
                    &HashMap::from([
                        (
                            "input".into(),
                            TensorData::from_scalars([3], dtype, [Scalar::F(1.0); 3]).unwrap(),
                        ),
                        (
                            "mask".into(),
                            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)])
                                .unwrap(),
                        ),
                    ]),
                )
                .unwrap();
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(actual.shape(), &Shape::from([3]));
            assert_eq!(actual.scalar_at(1).as_f64(), 2.0);
            assert!(actual.scalar_at(2).as_f64().is_nan());
        }

        let (graph, input, _, output) = dynamic_vjp_graph();
        let rule = graph.dynamic_compaction_vjp_rule(output).unwrap();
        let too_many = TensorData::from_scalars([7], DType::F32, [Scalar::F(1.0); 7]).unwrap();
        assert!(matches!(
            rule.lower(&graph, &too_many, input),
            Err(Error::DynamicVjp {
                reason: "dynamic compaction count exceeds source extent"
            })
        ));
    }

    #[test]
    fn dynamic_vjp_plan_prunes_masks_and_checks_runtime_mean_atomically() {
        let (mut graph, input, mask, output) = dynamic_vjp_graph();
        let before_static = graph.node_count();
        let before_dynamic = graph.dynamic_nodes.len();
        assert!(matches!(
            graph.dynamic_vjp_plan(output, mask),
            Err(Error::NonDifferentiableTarget(node)) if node == mask
        ));
        assert_eq!(graph.node_count(), before_static);
        assert_eq!(graph.dynamic_nodes.len(), before_dynamic);

        let cast_source = graph.input_dtype("cast_source", [2, 3], DType::F32);
        let integer = graph.cast(cast_source, DType::I32).unwrap();
        let recast = graph.cast(integer, DType::F32).unwrap();
        let cast_output = graph.masked_select_dynamic(recast, mask).unwrap();
        let before_static = graph.node_count();
        let before_dynamic = graph.dynamic_nodes.len();
        assert!(matches!(
            graph.dynamic_vjp_plan(cast_output, cast_source),
            Err(Error::NonDifferentiableTarget(node)) if node == cast_source
        ));
        assert_eq!(graph.node_count(), before_static);
        assert_eq!(graph.dynamic_nodes.len(), before_dynamic);

        let guarded = graph.tensor_guard_distribution(input, 1).unwrap();
        let guarded_output = graph.masked_select_dynamic(guarded, mask).unwrap();
        let before_static = graph.node_count();
        let before_dynamic = graph.dynamic_nodes.len();
        assert!(matches!(
            graph.dynamic_vjp_plan(guarded_output, input),
            Err(Error::NonDifferentiableIndexing(
                "tensor guard gradient is not represented"
            ))
        ));
        assert_eq!(graph.node_count(), before_static);
        assert_eq!(graph.dynamic_nodes.len(), before_dynamic);

        let mask_source = graph.input_dtype("mask_source", [1, 3], DType::F32);
        let zero = graph.constant(TensorData::scalar(0.0));
        let derived_mask = graph.gt(mask_source, zero).unwrap();
        let derived_output = graph.masked_select_dynamic(input, derived_mask).unwrap();
        let before_static = graph.node_count();
        let before_dynamic = graph.dynamic_nodes.len();
        assert!(matches!(
            graph.dynamic_vjp_plan(derived_output, mask_source),
            Err(Error::NonDifferentiableTarget(node)) if node == mask_source
        ));
        assert_eq!(graph.node_count(), before_static);
        assert_eq!(graph.dynamic_nodes.len(), before_dynamic);

        let unrelated = graph.input_dtype("unrelated", [2, 3], DType::F32);
        let before_static = graph.node_count();
        assert!(matches!(
            graph.dynamic_vjp_plan(output, unrelated),
            Err(Error::NonDifferentiableTarget(node)) if node == unrelated
        ));
        assert_eq!(graph.node_count(), before_static);
        assert_eq!(graph.dynamic_nodes.len(), before_dynamic);
        let mean = graph.dynamic_mean(output).unwrap();
        let before_dynamic = graph.dynamic_nodes.len();
        let mean_plan = graph.dynamic_vjp_plan(mean, input).unwrap();
        assert_eq!(mean_plan.output(), mean);
        assert_eq!(mean_plan.upstream_shape(), DynamicOutputShape::Scalar);
        let rule = graph.dynamic_mean_vjp_rule(mean).unwrap();
        assert_eq!(rule.input(), output);
        assert_eq!(rule.source_dtype(), DType::F32);
        assert_eq!(rule.dtypes(), ReductionDType::new(DType::F32, DType::F32));
        assert_eq!(graph.node_count(), before_static);
        assert_eq!(graph.dynamic_nodes.len(), before_dynamic);

        graph.dynamic_nodes[mean.index].dtype = DType::F64;
        assert!(matches!(
            graph.dynamic_vjp_plan(mean, input),
            Err(Error::DynamicVjp {
                reason: "dynamic Mean descriptor is not canonical"
            })
        ));
        assert_eq!(graph.node_count(), before_static);
        assert_eq!(graph.dynamic_nodes.len(), before_dynamic);

        for dtype in [DType::I32, DType::F8E5M2] {
            let mut unsupported = Graph::new();
            let input = unsupported.input_dtype("input", [2], dtype);
            let mask = unsupported.input_dtype("mask", [2], DType::Bool);
            let selected = unsupported.masked_select_dynamic(input, mask).unwrap();
            let mean = unsupported.dynamic_mean(selected).unwrap();
            let before_static = unsupported.node_count();
            let before_dynamic = unsupported.dynamic_nodes.len();
            assert!(matches!(
                unsupported.dynamic_vjp_plan(mean, input),
                Err(Error::NonDifferentiableTarget(node)) if node == input
            ));
            assert_eq!(unsupported.node_count(), before_static);
            assert_eq!(unsupported.dynamic_nodes.len(), before_dynamic);
        }
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
