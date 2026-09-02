//! Backend-neutral universal operations. This layer is below the tensor graph
//! and above future scheduling/rendering; it deliberately does not execute.
use crate::{DType, Shape, SymbolicExpr};
pub mod artifact;
mod operation;
pub use operation::{
    AddressValue, IndexAddressing, IndexValue, LiteralValue, MatmulValue, MovementValue, Operation,
    PrefixScanValue, ReductionValue, SortValue, TensorGuardValue, ThreefryValue, VariableValue,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AddressSpace {
    Global,
    Local,
    Register,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UType {
    pub scalar: DType,
    pub lanes: u16,
}
impl UType {
    pub fn scalar(scalar: DType) -> Self {
        Self { scalar, lanes: 1 }
    }
    pub fn vector(scalar: DType, lanes: u16) -> Result<Self, UOpError> {
        if lanes == 0 {
            Err(UOpError::InvalidLaneWidth)
        } else {
            Ok(Self { scalar, lanes })
        }
    }
    pub fn is_bool(self) -> bool {
        self.scalar == DType::Bool
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Unary {
    Neg,
    Not,
    Abs,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    Min,
    Max,
    Eq,
    Lt,
    Le,
    And,
    Or,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Ternary {
    Where,
}
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ViewMap {
    pub source_shape: Shape,
    pub logical_shape: Shape,
    pub strides: Vec<usize>,
    pub offset: usize,
}

/// Canonical signed affine logical-to-physical map. `ViewMap` remains the
/// lossless unsigned renderer/artifact adapter; effectful targets use this
/// descriptor when a flip needs a negative stride.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AffineView {
    pub source_shape: Shape,
    pub logical_shape: Shape,
    pub strides: Vec<i64>,
    pub offset: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NormalizedAffineRead {
    pub offset: usize,
    pub axes: Vec<NormalizedAffineReadAxis>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NormalizedAffineReadAxis {
    pub stride: usize,
    pub reversed: bool,
}

impl From<ViewMap> for AffineView {
    fn from(view: ViewMap) -> Self {
        Self {
            source_shape: view.source_shape,
            logical_shape: view.logical_shape,
            strides: view
                .strides
                .into_iter()
                .map(|stride| stride as i64)
                .collect(),
            offset: view.offset as i64,
        }
    }
}

impl AffineView {
    pub fn identity(shape: Shape) -> Self {
        ViewMap::identity(shape).into()
    }
    /// Checked adapter for legacy unsigned renderers. Signed addresses must be
    /// rejected by those renderers rather than silently reinterpreted.
    pub fn as_unsigned(&self) -> Result<ViewMap, UOpError> {
        self.validate_read()?;
        if self.offset < 0 || self.strides.iter().any(|stride| *stride < 0) {
            return Err(UOpError::InvalidIndex);
        }
        Ok(ViewMap {
            source_shape: self.source_shape.clone(),
            logical_shape: self.logical_shape.clone(),
            strides: self
                .strides
                .iter()
                .map(|stride| usize::try_from(*stride).map_err(|_| UOpError::InvalidIndex))
                .collect::<Result<Vec<_>, _>>()?,
            offset: usize::try_from(self.offset).map_err(|_| UOpError::InvalidIndex)?,
        })
    }
    pub fn flip(&self, axis: usize) -> Result<Self, UOpError> {
        if axis >= self.logical_shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let mut out = self.clone();
        let dim =
            i64::try_from(out.logical_shape.dims()[axis]).map_err(|_| UOpError::InvalidIndex)?;
        out.offset = out
            .offset
            .checked_add(
                (dim.saturating_sub(1))
                    .checked_mul(out.strides[axis])
                    .ok_or(UOpError::InvalidIndex)?,
            )
            .ok_or(UOpError::InvalidIndex)?;
        out.strides[axis] = out.strides[axis]
            .checked_neg()
            .ok_or(UOpError::InvalidIndex)?;
        out.validate()?;
        Ok(out)
    }
    /// Restricts each logical axis to a half-open interval without losing a
    /// signed source stride. This is a read map, so zero strides remain valid.
    pub fn shrink(&self, bounds: &[(usize, usize)]) -> Result<Self, UOpError> {
        if bounds.len() != self.logical_shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let mut out = self.clone();
        let mut logical = Vec::with_capacity(bounds.len());
        for (axis, ((start, end), dim)) in bounds.iter().zip(self.logical_shape.dims()).enumerate()
        {
            if start > end || *end > *dim {
                return Err(UOpError::InvalidIndex);
            }
            let start_index = *start;
            let start = i64::try_from(start_index).map_err(|_| UOpError::InvalidIndex)?;
            out.offset = out
                .offset
                .checked_add(
                    start
                        .checked_mul(out.strides[axis])
                        .ok_or(UOpError::InvalidIndex)?,
                )
                .ok_or(UOpError::InvalidIndex)?;
            logical.push(end - start_index);
        }
        out.logical_shape = Shape::new(logical);
        out.validate_read()?;
        Ok(out)
    }
    /// Reorders logical axes while retaining their signed source strides.
    pub fn permute(&self, axes: &[usize]) -> Result<Self, UOpError> {
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        if sorted != (0..self.logical_shape.rank()).collect::<Vec<_>>() {
            return Err(UOpError::InvalidIndex);
        }
        let out = Self {
            source_shape: self.source_shape.clone(),
            logical_shape: Shape::new(
                axes.iter()
                    .map(|axis| self.logical_shape.dims()[*axis])
                    .collect::<Vec<_>>(),
            ),
            strides: axes.iter().map(|axis| self.strides[*axis]).collect(),
            offset: self.offset,
        };
        out.validate_read()?;
        Ok(out)
    }
    /// Reshapes a read map when the physical addressing remains provably
    /// affine. Dense positive-stride maps retain the general `ViewMap`
    /// reshape contract. Non-contiguous and signed maps may only insert or
    /// remove singleton axes, which never changes a logical coordinate or its
    /// source address.
    pub fn reshape_read(&self, shape: Shape) -> Result<Self, UOpError> {
        if self.logical_shape.numel().ok() != shape.numel().ok() {
            return Err(UOpError::InvalidIndex);
        }
        if let Ok(view) = self.as_unsigned()
            && let Ok(reshaped) = view.reshape(shape.clone())
        {
            return Ok(reshaped.into());
        }

        let source_axes = self
            .logical_shape
            .dims()
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .filter(|(dim, _)| *dim != 1)
            .collect::<Vec<_>>();
        let target_axes = shape
            .dims()
            .iter()
            .copied()
            .filter(|dim| *dim != 1)
            .collect::<Vec<_>>();
        if source_axes.iter().map(|(dim, _)| *dim).collect::<Vec<_>>() != target_axes {
            return Err(UOpError::InvalidIndex);
        }
        let mut source_axes = source_axes.into_iter();
        let strides = shape
            .dims()
            .iter()
            .map(|dim| {
                if *dim == 1 {
                    Ok(0)
                } else {
                    source_axes
                        .next()
                        .map(|(_, stride)| stride)
                        .ok_or(UOpError::InvalidIndex)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if source_axes.next().is_some() {
            return Err(UOpError::InvalidIndex);
        }
        let out = Self {
            source_shape: self.source_shape.clone(),
            logical_shape: shape,
            strides,
            offset: self.offset,
        };
        out.validate_read()?;
        Ok(out)
    }
    /// Broadcasts singleton logical dimensions. Reads may deliberately become
    /// noninjective, but write-target validation remains separate.
    pub fn expand(&self, shape: Shape) -> Result<Self, UOpError> {
        if self.logical_shape.rank() > shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let pad = shape.rank() - self.logical_shape.rank();
        let mut strides = vec![0; pad];
        for ((input, output), stride) in self
            .logical_shape
            .dims()
            .iter()
            .zip(&shape.dims()[pad..])
            .zip(&self.strides)
        {
            if input == output {
                strides.push(*stride);
            } else if *input == 1 {
                strides.push(0);
            } else {
                return Err(UOpError::InvalidIndex);
            }
        }
        let out = Self {
            source_shape: self.source_shape.clone(),
            logical_shape: shape,
            strides,
            offset: self.offset,
        };
        out.validate_read()?;
        Ok(out)
    }
    /// Validates a logical-to-physical read map. Broadcast dimensions may have
    /// a zero stride and therefore intentionally alias source elements.
    pub fn validate_read(&self) -> Result<(), UOpError> {
        self.normalized_read().map(|_| ())
    }

    /// Converts a proved signed read map into nonnegative addressing metadata.
    /// Negative axes reverse their logical coordinate around the last element;
    /// singleton axes are discarded before magnitude conversion.
    pub(crate) fn normalized_read(&self) -> Result<NormalizedAffineRead, UOpError> {
        if self.strides.len() != self.logical_shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let logical_numel = self
            .logical_shape
            .numel()
            .map_err(|_| UOpError::InvalidIndex)?;
        let source_numel = self
            .source_shape
            .numel()
            .map_err(|_| UOpError::InvalidIndex)?;
        if logical_numel == 0 {
            let axes = self
                .logical_shape
                .dims()
                .iter()
                .zip(&self.strides)
                .map(|(&dimension, &stride)| {
                    if dimension > 1 {
                        let magnitude = i128::from(stride).abs();
                        let axis_span = i128::try_from(dimension - 1)
                            .map_err(|_| UOpError::InvalidIndex)?
                            .checked_mul(magnitude)
                            .ok_or(UOpError::InvalidIndex)?;
                        if axis_span > i128::from(i64::MAX) {
                            return Err(UOpError::InvalidIndex);
                        }
                    }
                    Ok(NormalizedAffineReadAxis {
                        stride: 0,
                        reversed: false,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(NormalizedAffineRead { offset: 0, axes });
        }
        let mut minimum = i128::from(self.offset);
        let mut span = 0i128;
        let mut axes = Vec::with_capacity(self.strides.len());
        for (&dimension, &stride) in self.logical_shape.dims().iter().zip(&self.strides) {
            if dimension <= 1 {
                axes.push(NormalizedAffineReadAxis {
                    stride: 0,
                    reversed: false,
                });
                continue;
            }
            let magnitude = i128::from(stride).abs();
            let axis_span = i128::try_from(dimension - 1)
                .map_err(|_| UOpError::InvalidIndex)?
                .checked_mul(magnitude)
                .ok_or(UOpError::InvalidIndex)?;
            if axis_span > i128::from(i64::MAX) {
                return Err(UOpError::InvalidIndex);
            }
            let reversed = stride < 0;
            if reversed {
                minimum = minimum
                    .checked_sub(axis_span)
                    .ok_or(UOpError::InvalidIndex)?;
            }
            span = span.checked_add(axis_span).ok_or(UOpError::InvalidIndex)?;
            axes.push(NormalizedAffineReadAxis {
                stride: usize::try_from(magnitude).map_err(|_| UOpError::InvalidIndex)?,
                reversed,
            });
        }
        let maximum = minimum.checked_add(span).ok_or(UOpError::InvalidIndex)?;
        if minimum < 0
            || minimum > i128::from(i64::MAX)
            || maximum > i128::from(i64::MAX)
            || (logical_numel != 0
                && maximum >= i128::try_from(source_numel).map_err(|_| UOpError::InvalidIndex)?)
        {
            return Err(UOpError::InvalidIndex);
        }
        let _: usize = usize::try_from(maximum).map_err(|_| UOpError::InvalidIndex)?;
        Ok(NormalizedAffineRead {
            offset: usize::try_from(minimum).map_err(|_| UOpError::InvalidIndex)?,
            axes,
        })
    }
    /// Validates a writable affine target. Unlike reads, every logical lane
    /// must identify a distinct physical element so an effect has one meaning.
    pub fn validate_write(&self) -> Result<(), UOpError> {
        self.validate_read()?;
        let numel = self
            .logical_shape
            .numel()
            .map_err(|_| UOpError::InvalidIndex)?;
        let mut seen = std::collections::BTreeSet::new();
        for index in 0..numel {
            if !seen.insert(self.element_offset(index)?) {
                return Err(UOpError::InvalidIndex);
            }
        }
        Ok(())
    }
    /// Backward-compatible validation for the already-effectful caller.
    pub fn validate(&self) -> Result<(), UOpError> {
        self.validate_write()
    }
    pub fn element_offset(&self, logical_linear: usize) -> Result<i64, UOpError> {
        if logical_linear
            >= self
                .logical_shape
                .numel()
                .map_err(|_| UOpError::InvalidIndex)?
        {
            return Err(UOpError::InvalidIndex);
        }
        let mut linear = logical_linear;
        let mut offset = self.offset;
        for axis in (0..self.logical_shape.rank()).rev() {
            let dim = self.logical_shape.dims()[axis];
            if dim != 0 {
                let coordinate = i64::try_from(linear % dim).map_err(|_| UOpError::InvalidIndex)?;
                linear /= dim;
                offset = offset
                    .checked_add(
                        coordinate
                            .checked_mul(self.strides[axis])
                            .ok_or(UOpError::InvalidIndex)?,
                    )
                    .ok_or(UOpError::InvalidIndex)?;
            }
        }
        Ok(offset)
    }
}
impl ViewMap {
    pub fn identity(shape: Shape) -> Self {
        Self {
            strides: shape.contiguous_strides(),
            logical_shape: shape.clone(),
            source_shape: shape,
            offset: 0,
        }
    }
    pub fn shrink(&self, bounds: &[(usize, usize)]) -> Result<Self, UOpError> {
        if bounds.len() != self.logical_shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let mut offset = self.offset;
        let mut logical = Vec::with_capacity(bounds.len());
        for ((start, end), (dim, stride)) in bounds
            .iter()
            .zip(self.logical_shape.dims().iter().zip(&self.strides))
        {
            if start > end || *end > *dim {
                return Err(UOpError::InvalidIndex);
            }
            offset = offset
                .checked_add(start.checked_mul(*stride).ok_or(UOpError::InvalidIndex)?)
                .ok_or(UOpError::InvalidIndex)?;
            logical.push(end - start);
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: Shape::new(logical),
            strides: self.strides.clone(),
            offset,
        })
    }
    pub fn permute(&self, axes: &[usize]) -> Result<Self, UOpError> {
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        if sorted != (0..self.logical_shape.rank()).collect::<Vec<_>>() {
            return Err(UOpError::InvalidIndex);
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: Shape::new(
                axes.iter()
                    .map(|axis| self.logical_shape.dims()[*axis])
                    .collect::<Vec<_>>(),
            ),
            strides: axes.iter().map(|axis| self.strides[*axis]).collect(),
            offset: self.offset,
        })
    }
    pub fn reshape(&self, shape: Shape) -> Result<Self, UOpError> {
        if self.logical_shape.numel().ok() != shape.numel().ok()
            || self.strides != self.logical_shape.contiguous_strides()
        {
            return Err(UOpError::InvalidIndex);
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            strides: shape.contiguous_strides(),
            logical_shape: shape,
            offset: self.offset,
        })
    }
    pub fn expand(&self, shape: Shape) -> Result<Self, UOpError> {
        if self.logical_shape.rank() > shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let pad = shape.rank() - self.logical_shape.rank();
        let mut strides = vec![0; pad];
        for ((input, output), stride) in self
            .logical_shape
            .dims()
            .iter()
            .zip(&shape.dims()[pad..])
            .zip(&self.strides)
        {
            if input == output {
                strides.push(*stride);
            } else if *input == 1 {
                strides.push(0);
            } else {
                return Err(UOpError::InvalidIndex);
            }
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: shape,
            strides,
            offset: self.offset,
        })
    }
    /// Applies already-normalized positive-stride slices as
    /// `(start, step, output_length)` tuples. Negative strides require signed
    /// address metadata and are intentionally outside this affine map.
    pub fn stride_positive(&self, slices: &[(usize, usize, usize)]) -> Result<Self, UOpError> {
        if slices.len() != self.logical_shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let mut offset = self.offset;
        let mut logical = Vec::with_capacity(slices.len());
        let mut strides = Vec::with_capacity(slices.len());
        for ((start, step, length), (dim, stride)) in slices
            .iter()
            .zip(self.logical_shape.dims().iter().zip(&self.strides))
        {
            if *step == 0 || (*length != 0 && *start >= *dim) {
                return Err(UOpError::InvalidIndex);
            }
            offset = offset
                .checked_add(start.checked_mul(*stride).ok_or(UOpError::InvalidIndex)?)
                .ok_or(UOpError::InvalidIndex)?;
            logical.push(*length);
            strides.push(stride.checked_mul(*step).ok_or(UOpError::InvalidIndex)?);
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: Shape::new(logical),
            strides,
            offset,
        })
    }
    pub fn element_offset(&self, logical_linear: usize) -> Result<usize, UOpError> {
        if logical_linear
            >= self
                .logical_shape
                .numel()
                .map_err(|_| UOpError::InvalidIndex)?
        {
            return Err(UOpError::InvalidIndex);
        }
        let mut linear = logical_linear;
        let mut offset = self.offset;
        for axis in (0..self.logical_shape.rank()).rev() {
            let dim = self.logical_shape.dims()[axis];
            if dim != 0 {
                let coord = linear % dim;
                linear /= dim;
                offset = offset
                    .checked_add(
                        coord
                            .checked_mul(self.strides[axis])
                            .ok_or(UOpError::InvalidIndex)?,
                    )
                    .ok_or(UOpError::InvalidIndex)?;
            }
        }
        let source = self
            .source_shape
            .numel()
            .map_err(|_| UOpError::InvalidIndex)?;
        if offset >= source && source != 0 {
            return Err(UOpError::InvalidIndex);
        }
        Ok(offset)
    }
}
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct UOpNode {
    operation: Operation,
    ty: Option<UType>,
    sources: Vec<UOp>,
}
/// Immutable and structurally hashable. Cloning preserves DAG sharing.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UOp(Arc<UOpNode>);
impl UOp {
    pub fn from_operation(operation: Operation, ty: Option<UType>, sources: Vec<UOp>) -> Self {
        Self(Arc::new(UOpNode {
            operation,
            ty,
            sources,
        }))
    }
    pub(crate) fn shares_node_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn node_identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
    pub fn operation(&self) -> &Operation {
        &self.0.operation
    }
    pub fn ty(&self) -> Option<UType> {
        self.0.ty
    }
    pub fn sources(&self) -> &[UOp] {
        &self.0.sources
    }
    pub fn constant(value: i64, ty: UType) -> Self {
        Self::from_operation(Operation::Const(LiteralValue::Int(value)), Some(ty), vec![])
    }
    pub fn scalar_constant(dtype: DType, bits: u64, ty: UType) -> Self {
        Self::from_operation(
            Operation::Const(LiteralValue::Scalar { dtype, bits }),
            Some(ty),
            vec![],
        )
    }
    pub fn unary(op: Unary, x: UOp) -> Self {
        Self::from_operation(Operation::Unary(op), x.ty(), vec![x])
    }
    pub fn binary(op: Binary, a: UOp, b: UOp) -> Self {
        let ty = if matches!(op, Binary::Eq | Binary::Lt | Binary::Le) {
            Some(UType::scalar(DType::Bool))
        } else {
            a.ty()
        };
        Self::from_operation(Operation::Binary(op), ty, vec![a, b])
    }
    pub fn cast(x: UOp, to: UType) -> Self {
        Self::from_operation(Operation::Cast, Some(to), vec![x])
    }
    pub fn sink(sources: Vec<UOp>) -> Self {
        Self::from_operation(Operation::Sink, None, sources)
    }
    pub fn topological(&self) -> Result<Vec<UOp>, UOpError> {
        fn visit(
            n: &UOp,
            seen: &mut BTreeSet<UOp>,
            active: &mut BTreeSet<UOp>,
            out: &mut Vec<UOp>,
        ) -> Result<(), UOpError> {
            if seen.contains(n) {
                return Ok(());
            }
            if !active.insert(n.clone()) {
                return Err(UOpError::Cycle);
            }
            for s in n.sources() {
                visit(s, seen, active, out)?
            }
            active.remove(n);
            seen.insert(n.clone());
            out.push(n.clone());
            Ok(())
        }
        let mut out = vec![];
        visit(self, &mut BTreeSet::new(), &mut BTreeSet::new(), &mut out)?;
        Ok(out)
    }
    pub fn is_pure(&self) -> bool {
        !matches!(
            self.operation(),
            Operation::EndRange
                | Operation::If
                | Operation::EndIf
                | Operation::Store
                | Operation::EffectStore(_)
                | Operation::After(_)
                | Operation::Barrier
                | Operation::Sink
        )
    }
    pub fn validate(&self) -> Result<(), UOpError> {
        let nodes = self.topological()?;
        let mut ranges = BTreeSet::new();
        let mut ifs = Vec::new();
        for n in &nodes {
            validate_one(n, &mut ranges, &mut ifs)?
        }
        validate_projected_index_consumers(&nodes)?;
        if !ifs.is_empty() || !ranges.is_empty() {
            return Err(UOpError::UnclosedControl);
        }
        validate_reduction_topology(self)?;
        Ok(())
    }
}

fn validate_projected_index_consumers(nodes: &[UOp]) -> Result<(), UOpError> {
    let mut consumers = HashMap::<UOp, bool>::new();
    for consumer in nodes {
        let is_load = matches!(consumer.operation(), Operation::Load);
        for source in consumer.sources() {
            consumers
                .entry(source.clone())
                .and_modify(|only_loads| *only_loads &= is_load)
                .or_insert(is_load);
        }
    }
    for index in nodes
        .iter()
        .filter(|node| crate::projected_index::ProjectedIndexPlan::is_projected(node))
    {
        if consumers.get(index) != Some(&true) {
            return Err(UOpError::InvalidIndex);
        }
    }
    Ok(())
}

#[cfg(test)]
mod projected_consumer_tests {
    use super::*;

    #[test]
    fn large_projected_consumer_summary_uses_one_edge_pass() {
        let value_type = UType::scalar(DType::F32);
        let index_type = UType::scalar(DType::I64);
        let shape = Shape::from([1]);
        let mut nodes = Vec::with_capacity(8193);
        let mut first = None;
        for buffer in 0..4096_u64 {
            let address = UOp::from_operation(
                Operation::DefineGlobal(AddressValue {
                    space: AddressSpace::Global,
                    name: format!("b{buffer}"),
                    element: value_type,
                }),
                Some(value_type),
                vec![],
            );
            let index = UOp::from_operation(
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements: 1,
                    input_shape: shape.clone(),
                    output_shape: shape.clone(),
                    addressing: IndexAddressing::Projected,
                }),
                Some(value_type),
                vec![address, UOp::constant(0, index_type)],
            );
            if first.is_none() {
                first = Some(index.clone());
            }
            nodes.push(index.clone());
            nodes.push(UOp::from_operation(
                Operation::Load,
                Some(value_type),
                vec![index],
            ));
        }
        assert_eq!(validate_projected_index_consumers(&nodes), Ok(()));

        nodes.push(UOp::from_operation(
            Operation::Store,
            None,
            vec![
                first.unwrap(),
                UOp::scalar_constant(DType::F32, 0, value_type),
            ],
        ));
        assert_eq!(
            validate_projected_index_consumers(&nodes),
            Err(UOpError::InvalidIndex)
        );
    }
}

fn validate_reduction_topology(root: &UOp) -> Result<(), UOpError> {
    fn visit(node: &UOp, seen: &mut BTreeSet<usize>, out: &mut Vec<UOp>) {
        if !seen.insert(node.node_identity()) {
            return;
        }
        for source in node.sources() {
            visit(source, seen, out);
        }
        out.push(node.clone());
    }

    // `topological` deliberately deduplicates structurally equal UOps for
    // canonical ordering and cache identity. Reduction ownership instead
    // follows actual Arc nodes and source edges so two equal-but-distinct
    // chains cannot hide a duplicated Init/Finalize use.
    let mut nodes = Vec::new();
    visit(root, &mut BTreeSet::new(), &mut nodes);
    let mut structural_reductions = BTreeMap::<UOp, usize>::new();
    for node in &nodes {
        if matches!(
            node.operation(),
            Operation::ReduceInit(_) | Operation::ReduceAccumulate | Operation::ReduceFinalize
        ) && structural_reductions
            .insert(node.clone(), node.node_identity())
            .is_some_and(|identity| identity != node.node_identity())
        {
            // Artifact ordering and source indices use structural UOp
            // identity. Reject Arc-distinct reduction nodes that would
            // collapse during serialization and change ownership.
            return Err(UOpError::InvalidArgument);
        }
    }
    let mut consumers = BTreeMap::<usize, Vec<&UOp>>::new();
    for node in &nodes {
        for source in node.sources() {
            consumers
                .entry(source.node_identity())
                .or_default()
                .push(node);
        }
    }
    for node in &nodes {
        let uses = consumers
            .get(&node.node_identity())
            .map(Vec::as_slice)
            .unwrap_or_default();
        match node.operation() {
            Operation::ReduceInit(_) => {
                if uses.len() != 1
                    || !matches!(uses[0].operation(), Operation::ReduceAccumulate)
                    || !uses[0]
                        .sources()
                        .first()
                        .is_some_and(|source| source.shares_node_with(node))
                {
                    return Err(UOpError::InvalidArgument);
                }
            }
            Operation::ReduceAccumulate => {
                if uses.len() != 1
                    || !matches!(uses[0].operation(), Operation::ReduceFinalize)
                    || !uses[0]
                        .sources()
                        .first()
                        .is_some_and(|source| source.shares_node_with(node))
                {
                    return Err(UOpError::InvalidArgument);
                }
            }
            Operation::ReduceFinalize => {
                if uses.len() > 1 {
                    return Err(UOpError::InvalidArgument);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Returns whether raw scalar storage metadata can faithfully inhabit `ty`.
///
/// `LiteralValue::Scalar` is a storage literal rather than an integer
/// expression: its
/// dtype is part of the immutable UOp ABI and its high bits must be absent for
/// narrow storage.  Keep this check in the universal layer so direct UOp
/// construction, artifact decoding, and rewrite guards share one contract.
pub(crate) fn scalar_literal_is_valid(ty: Option<UType>, dtype: DType, bits: u64) -> bool {
    ty.is_some_and(|node_ty| node_ty.scalar == dtype)
        && (dtype.bits() == 64 || bits >> usize::from(dtype.bits()) == 0)
}

impl fmt::Display for UOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.operation())?;
        if let Some(t) = self.ty() {
            write!(f, ":{:?}x{}", t.scalar, t.lanes)?
        }
        if !self.sources().is_empty() {
            write!(f, "[")?;
            for (i, s) in self.sources().iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?
                }
                write!(f, "{s}")?
            }
            write!(f, "]")?
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UOpError {
    InvalidArity {
        expected: &'static str,
        actual: usize,
    },
    InvalidDType,
    InvalidLaneWidth,
    InvalidArgument,
    InvalidIndex,
    Cycle,
    UseBeforeDefinition,
    ControlMismatch,
    UnclosedControl,
    EffectRewrite,
    RewriteCycle,
    RewriteStepLimit,
}
impl fmt::Display for UOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UOp validation error: {self:?}")
    }
}
impl std::error::Error for UOpError {}
fn validate_operation_arity(n: &UOp) -> Result<(), UOpError> {
    let actual = n.sources().len();
    let (accepted, expected) = match n.operation() {
        Operation::Const(_)
        | Operation::VConst(_)
        | Operation::DefineVar(_)
        | Operation::DefineGlobal(_)
        | Operation::DefineLocal(_)
        | Operation::DefineRegister(_)
        | Operation::Special(_)
        | Operation::Matmul(_)
        | Operation::Conv2d(_)
        | Operation::Movement(_)
        | Operation::Random(_)
        | Operation::Threefry(_)
        | Operation::PrefixScan(_)
        | Operation::Sort(_)
        | Operation::TensorGuard(_)
        | Operation::ReduceInit(_)
        | Operation::EffectStore(_)
        | Operation::Barrier => (actual == 0, "no sources"),
        Operation::Range(_)
        | Operation::EndRange
        | Operation::If
        | Operation::EndIf
        | Operation::Unary(_)
        | Operation::GraphUnary(_)
        | Operation::ReduceFinalize
        | Operation::Cast
        | Operation::Bitcast
        | Operation::Gep(_)
        | Operation::Load
        | Operation::After(_) => (actual == 1, "one source"),
        Operation::Binary(_)
        | Operation::GraphBinary(_)
        | Operation::GraphCompare(_)
        | Operation::ReduceAccumulate
        | Operation::Index(_)
        | Operation::Store => (actual == 2, "two sources"),
        Operation::GraphLogical(crate::LogicalOp::Not) => (actual == 1, "one source"),
        Operation::GraphLogical(crate::LogicalOp::And | crate::LogicalOp::Or) => {
            (actual == 2, "two sources")
        }
        Operation::Ternary(_) => (actual == 3, "three sources"),
        Operation::Vectorize => (actual != 0, "one or more sources"),
        Operation::Sink => (true, "any source count"),
    };
    if !accepted {
        Err(UOpError::InvalidArity { expected, actual })
    } else {
        Ok(())
    }
}
fn same(n: &UOp) -> bool {
    n.sources().iter().all(|s| s.ty() == n.ty())
}
fn graph_unary_type_is_valid(
    op: crate::UnaryOp,
    input: Option<UType>,
    output: Option<UType>,
) -> bool {
    input.zip(output).is_some_and(|(input, output)| {
        input.lanes == output.lanes && crate::ir::unary_dtype(op, input.scalar) == output.scalar
    })
}
fn validate_one(n: &UOp, ranges: &mut BTreeSet<u32>, ifs: &mut Vec<UOp>) -> Result<(), UOpError> {
    validate_operation_arity(n)?;
    match n.operation() {
        Operation::Const(value) | Operation::VConst(value) => match value {
            // Integer literals are structural/index values interpreted by the
            // node type. Scalar literals retain exact storage metadata.
            LiteralValue::Int(_) if n.ty().is_some() => {}
            LiteralValue::Scalar { dtype, bits }
                if scalar_literal_is_valid(n.ty(), *dtype, *bits) => {}
            LiteralValue::Int(_) | LiteralValue::Scalar { .. } => {
                return Err(UOpError::InvalidDType);
            }
        },
        Operation::DefineVar(_) | Operation::Special(_) => {}
        Operation::DefineGlobal(value) if value.space == AddressSpace::Global => {}
        Operation::DefineLocal(value) if value.space == AddressSpace::Local => {}
        Operation::DefineRegister(value) if value.space == AddressSpace::Register => {}
        Operation::DefineGlobal(_) | Operation::DefineLocal(_) | Operation::DefineRegister(_) => {
            return Err(UOpError::InvalidArgument);
        }
        Operation::Range(axis) => {
            if !n.sources()[0].ty().is_some_and(|t| t.scalar.is_integer()) {
                return Err(UOpError::InvalidDType);
            }
            ranges.insert(*axis);
        }
        Operation::EndRange => {
            let Operation::Range(axis) = n.sources()[0].operation() else {
                return Err(UOpError::ControlMismatch);
            };
            if !ranges.remove(axis) {
                return Err(UOpError::ControlMismatch);
            }
        }
        Operation::If => {
            if !n.sources()[0].ty().is_some_and(UType::is_bool) {
                return Err(UOpError::InvalidDType);
            }
            ifs.push(n.clone())
        }
        Operation::EndIf => {
            if !matches!(n.sources()[0].operation(), Operation::If)
                || ifs.pop().as_ref() != Some(&n.sources()[0])
            {
                return Err(UOpError::ControlMismatch);
            }
        }
        Operation::Unary(_) => {
            if !same(n) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::GraphUnary(op) => {
            if !graph_unary_type_is_valid(*op, n.sources()[0].ty(), n.ty()) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Binary(op) => {
            if !matches!(
                op,
                crate::uop::Binary::Eq | crate::uop::Binary::Lt | crate::uop::Binary::Le
            ) && !same(n)
            {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::GraphBinary(_) => {
            if n.ty().is_none() {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::GraphCompare(_) => {
            if n.ty() != Some(UType::scalar(DType::Bool)) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::GraphLogical(_) => {
            if n.ty() != Some(UType::scalar(DType::Bool))
                || n.sources()
                    .iter()
                    .any(|s| !s.ty().is_some_and(UType::is_bool))
            {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Threefry(value) => {
            value.validate()?;
            if n.ty() != Some(UType::scalar(DType::U64)) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Matmul(value) => match value {
            MatmulValue::Serial(plan) => {
                plan.validate().map_err(|_| UOpError::InvalidArgument)?;
                if n.ty() != Some(UType::scalar(plan.dtype)) {
                    return Err(UOpError::InvalidDType);
                }
            }
            MatmulValue::Tiled(payload) => {
                payload.validate().map_err(|_| UOpError::InvalidArgument)?;
                if n.ty() != Some(UType::scalar(payload.matmul.dtype)) {
                    return Err(UOpError::InvalidDType);
                }
            }
            MatmulValue::TensorCore(payload) => {
                payload.validate().map_err(|_| UOpError::InvalidArgument)?;
                if n.ty() != Some(UType::scalar(payload.matmul.dtype)) {
                    return Err(UOpError::InvalidDType);
                }
            }
            MatmulValue::Quantized(plan) => {
                plan.validate().map_err(|_| UOpError::InvalidArgument)?;
                if n.ty() != Some(UType::scalar(plan.output_dtype)) {
                    return Err(UOpError::InvalidDType);
                }
            }
        },
        Operation::Conv2d(plan) => {
            plan.validate().map_err(|_| UOpError::InvalidArgument)?;
            if n.ty() != Some(UType::scalar(DType::F32)) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Movement(value) => match value {
            MovementValue::Plan(plan) => {
                plan.validate().map_err(|_| UOpError::InvalidArgument)?;
                if n.ty() != Some(UType::scalar(plan.dtype)) {
                    return Err(UOpError::InvalidDType);
                }
            }
            MovementValue::QuantizedRowGather(plan) => {
                plan.validate().map_err(|_| UOpError::InvalidArgument)?;
                if n.ty() != Some(UType::scalar(plan.output_dtype)) {
                    return Err(UOpError::InvalidDType);
                }
            }
        },
        Operation::Random(plan) => {
            plan.validate().map_err(|_| UOpError::InvalidArgument)?;
            if n.ty() != Some(UType::scalar(plan.dtype)) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::PrefixScan(value) => {
            let PrefixScanValue {
                input,
                destination,
                input_shape,
                output_shape,
                axis,
                kind,
                output,
                input_dtype,
                dtype,
                ..
            } = value;
            if input_shape != output_shape
                || input == destination
                || (input_shape.rank() != 0 && *axis >= input_shape.rank())
                || (input_shape.rank() == 0 && *axis != 0)
                || (*kind == crate::PrefixScanKind::Sum && *dtype == DType::Bool)
                || crate::ir::prefix_scan_output_dtype(*input_dtype, *kind, *output) != Some(*dtype)
                || (matches!(
                    kind,
                    crate::PrefixScanKind::Sum | crate::PrefixScanKind::Product
                ) && *output != crate::PrefixScanOutput::Values)
                || (matches!(
                    kind,
                    crate::PrefixScanKind::Max | crate::PrefixScanKind::Min
                ) && *output == crate::PrefixScanOutput::Indices
                    && *dtype != DType::I32)
            {
                return Err(UOpError::InvalidArgument);
            }
            if n.ty() != Some(UType::scalar(*dtype)) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Sort(value) => {
            let SortValue {
                input_shape,
                axis,
                values,
                indices,
                dtype,
                ..
            } = value;
            if (input_shape.rank() != 0 && *axis >= input_shape.rank())
                || (input_shape.rank() == 0 && *axis != 0)
                || values == indices
            {
                return Err(UOpError::InvalidArgument);
            }
            if n.ty() != Some(UType::scalar(*dtype)) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::TensorGuard(value) => {
            let TensorGuardValue {
                input_shape,
                axis,
                dtype,
                ..
            } = value;
            if !(1..=2).contains(&input_shape.rank())
                || *axis >= input_shape.rank()
                || !dtype.is_float()
                || n.ty() != Some(UType::scalar(*dtype))
            {
                return Err(UOpError::InvalidArgument);
            }
        }
        Operation::ReduceInit(_) => {
            if n.ty().is_none() {
                return Err(UOpError::InvalidArgument);
            }
        }
        Operation::ReduceAccumulate => {
            if n.ty().is_none()
                || !matches!(n.sources()[0].operation(), Operation::ReduceInit(_))
                || n.sources()[0].ty() != n.ty()
                || n.sources()[1].ty().is_none()
            {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::ReduceFinalize => {
            crate::reduction_native::NativeReductionPlan::from_finalize(n)
                .map_err(|_| UOpError::InvalidDType)?;
        }
        Operation::Ternary(crate::uop::Ternary::Where) => {
            if !n.sources()[0].ty().is_some_and(UType::is_bool)
                || n.sources()[1].ty() != n.sources()[2].ty()
                || n.ty() != n.sources()[1].ty()
            {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Cast | Operation::Bitcast => {
            if n.ty().is_none() {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Vectorize => {
            if n.sources().is_empty() || !same(n) {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::Gep(_) => {}
        Operation::Index(value) => {
            if !matches!(
                n.sources()[0].operation(),
                Operation::DefineGlobal(_)
                    | Operation::DefineLocal(_)
                    | Operation::DefineRegister(_)
            ) || !n.sources()[1].ty().is_some_and(|t| t.scalar.is_integer())
            {
                return Err(UOpError::InvalidIndex);
            }
            let (elements, input_shape, output_shape) = match value {
                IndexValue::Buffer {
                    elements,
                    input_shape,
                    output_shape,
                    ..
                }
                | IndexValue::View {
                    elements,
                    input_shape,
                    output_shape,
                    ..
                } => (elements, input_shape, output_shape),
            };
            if input_shape.numel().ok() != Some(*elements) {
                return Err(UOpError::InvalidIndex);
            }
            let projected = crate::projected_index::ProjectedIndexPlan::is_projected(n);
            if projected {
                crate::projected_index::ProjectedIndexPlan::from_index(n)?;
            } else if input_shape.rank() > output_shape.rank()
                || !input_shape
                    .dims()
                    .iter()
                    .rev()
                    .zip(output_shape.dims().iter().rev())
                    .all(|(input, output)| *input == 1 || input == output)
            {
                return Err(UOpError::InvalidIndex);
            }
            if let IndexValue::View {
                view, input_shape, ..
            } = value
                && (&view.logical_shape != input_shape || view.validate_read().is_err())
            {
                return Err(UOpError::InvalidIndex);
            }
        }
        Operation::Load => {
            if !matches!(n.sources()[0].operation(), Operation::Index(_)) {
                return Err(UOpError::InvalidIndex);
            }
        }
        Operation::Store => {
            if !matches!(n.sources()[0].operation(), Operation::Index(_)) {
                return Err(UOpError::InvalidIndex);
            }
        }
        Operation::EffectStore(payload) => {
            if payload.target.buffer != payload.snapshot.buffer
                || payload.target.version
                    != payload
                        .snapshot
                        .version
                        .checked_add(1)
                        .ok_or(UOpError::InvalidArgument)?
                || payload.target.dtype != payload.source.dtype
                || payload.target.shape != payload.snapshot.shape
                || payload.target.bytes != payload.snapshot.bytes
            {
                return Err(UOpError::InvalidArgument);
            }
        }
        Operation::After(_) => {
            if !matches!(n.sources()[0].operation(), Operation::EffectStore(_)) {
                return Err(UOpError::ControlMismatch);
            }
        }
        Operation::Barrier => {}
        Operation::Sink => {
            if n.ty().is_some() {
                return Err(UOpError::InvalidDType);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct Captures(BTreeMap<String, UOp>);
impl Captures {
    pub fn get(&self, name: &str) -> Option<&UOp> {
        self.0.get(name)
    }
}

#[derive(Clone, Debug)]
enum SourceConstraint {
    Exact(Vec<UPat>),
    Prefix(Vec<UPat>),
    Each(Box<UPat>),
    Permuted(Vec<UPat>),
}

type EarlyRejectPredicates = Vec<fn(&Operation) -> bool>;

#[derive(Clone, Debug)]
pub struct UPat {
    alternatives: Option<Vec<UPat>>,
    operations: Option<BTreeSet<Operation>>,
    operation_predicate: Option<fn(&Operation) -> bool>,
    ty: Option<UType>,
    type_predicates: Vec<fn(Option<UType>) -> bool>,
    sources: Option<SourceConstraint>,
    name: Option<String>,
    custom_early_reject: Option<EarlyRejectPredicates>,
}
impl UPat {
    pub fn any() -> Self {
        Self {
            alternatives: None,
            operations: None,
            operation_predicate: None,
            ty: None,
            type_predicates: vec![],
            sources: None,
            name: None,
            custom_early_reject: None,
        }
    }
    /// Matches the first successful alternative in declaration order. Rewrite
    /// callbacks may reject that capture candidate, in which case the driver
    /// continues with the next successful alternative.
    pub fn any_of(alternatives: impl IntoIterator<Item = UPat>) -> Result<Self, UOpError> {
        let alternatives = alternatives.into_iter().collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(UOpError::InvalidArgument);
        }
        let mut x = Self::any();
        x.alternatives = Some(alternatives);
        Ok(x)
    }
    pub fn op(operation: Operation) -> Self {
        let mut x = Self::any();
        x.operations = Some([operation].into());
        x
    }
    pub fn ops(operations: impl IntoIterator<Item = Operation>) -> Self {
        let mut x = Self::any();
        x.operations = Some(operations.into_iter().collect());
        x
    }
    pub fn operation_predicate(predicate: fn(&Operation) -> bool) -> Self {
        let mut x = Self::any();
        x.operation_predicate = Some(predicate);
        x
    }
    pub fn dtype(mut self, ty: UType) -> Self {
        self.ty = Some(ty);
        self
    }
    /// Adds a predicate over the complete optional UOp result type. This is
    /// intentionally a function pointer: patterns remain cloneable,
    /// deterministic data and cannot retain mutable matching state.
    pub fn type_predicate(mut self, predicate: fn(Option<UType>) -> bool) -> Self {
        self.type_predicates.push(predicate);
        self
    }
    pub fn sources(mut self, s: Vec<UPat>) -> Self {
        self.sources = Some(SourceConstraint::Exact(s));
        self
    }
    /// Matches the ordered source prefix and permits any number of trailing
    /// sources. The named parent pattern can inspect the complete source list
    /// in its rewrite callback.
    pub fn sources_prefix(mut self, prefix: Vec<UPat>) -> Self {
        self.sources = Some(SourceConstraint::Prefix(prefix));
        self
    }
    /// Applies one pattern to every source, including an empty source list.
    /// A name on the repeated child therefore requires every source to be the
    /// same structural UOp, matching ordinary named-capture semantics.
    pub fn sources_varargs(mut self, pattern: UPat) -> Self {
        self.sources = Some(SourceConstraint::Each(Box::new(pattern)));
        self
    }
    /// Matches every declaration-ordered permutation of an exact source list.
    /// Callers use this only when the operation's operand roles are
    /// semantically interchangeable. Permutations are generated lazily, and an
    /// all-identical pattern list has only one candidate.
    pub fn sources_permuted(mut self, patterns: Vec<UPat>) -> Self {
        self.sources = Some(SourceConstraint::Permuted(patterns));
        self
    }
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Overrides inferred direct-source requirements used by the rewrite
    /// driver to reject impossible candidates before recursive matching. An
    /// empty list explicitly disables inference. This metadata does not
    /// change [`UPat::matches`] semantics.
    pub fn custom_early_reject(
        mut self,
        predicates: impl IntoIterator<Item = fn(&Operation) -> bool>,
    ) -> Self {
        self.custom_early_reject = Some(predicates.into_iter().collect());
        self
    }
    pub fn matches(&self, node: &UOp) -> Option<Captures> {
        let mut found = None;
        self.visit_matches(node, &Captures::default(), &mut |captures| {
            found = Some(captures);
            true
        });
        found
    }
    fn visit_matches(
        &self,
        n: &UOp,
        captures: &Captures,
        visitor: &mut dyn FnMut(Captures) -> bool,
    ) -> bool {
        if self
            .operations
            .as_ref()
            .is_some_and(|operations| !operations.contains(n.operation()))
            || self
                .operation_predicate
                .is_some_and(|predicate| !predicate(n.operation()))
        {
            return false;
        }
        if self.ty.is_some_and(|x| n.ty() != Some(x)) {
            return false;
        }
        if self
            .type_predicates
            .iter()
            .any(|predicate| !predicate(n.ty()))
        {
            return false;
        }

        if let Some(alternatives) = &self.alternatives {
            for alternative in alternatives {
                let mut after_alternative =
                    |candidate| self.visit_sources_and_capture(n, candidate, visitor);
                if alternative.visit_matches(n, captures, &mut after_alternative) {
                    return true;
                }
            }
            false
        } else {
            self.visit_sources_and_capture(n, captures.clone(), visitor)
        }
    }
    fn visit_sources_and_capture(
        &self,
        n: &UOp,
        captures: Captures,
        visitor: &mut dyn FnMut(Captures) -> bool,
    ) -> bool {
        let mut finish = |candidate| match self.with_capture(n, candidate) {
            Some(candidate) => visitor(candidate),
            None => false,
        };
        match &self.sources {
            None => finish(captures),
            Some(SourceConstraint::Exact(patterns)) => {
                patterns.len() == n.sources().len()
                    && Self::visit_ordered_sources(patterns, n.sources(), 0, captures, &mut finish)
            }
            Some(SourceConstraint::Prefix(patterns)) => {
                patterns.len() <= n.sources().len()
                    && Self::visit_ordered_sources(patterns, n.sources(), 0, captures, &mut finish)
            }
            Some(SourceConstraint::Each(pattern)) => {
                Self::visit_repeated_source(pattern, n.sources(), 0, captures, &mut finish)
            }
            Some(SourceConstraint::Permuted(patterns)) => {
                if patterns.len() != n.sources().len() {
                    return false;
                }
                if patterns.first().is_none_or(|first| {
                    patterns
                        .iter()
                        .all(|pattern| pattern.match_equivalent(first))
                }) {
                    return Self::visit_ordered_sources(
                        patterns,
                        n.sources(),
                        0,
                        captures,
                        &mut finish,
                    );
                }
                let mut used_patterns = vec![false; patterns.len()];
                Self::visit_permuted_sources(
                    patterns,
                    n.sources(),
                    0,
                    &mut used_patterns,
                    captures,
                    &mut finish,
                )
            }
        }
    }
    fn visit_ordered_sources(
        patterns: &[UPat],
        sources: &[UOp],
        index: usize,
        captures: Captures,
        visitor: &mut dyn FnMut(Captures) -> bool,
    ) -> bool {
        if index == patterns.len() {
            return visitor(captures);
        }
        patterns[index].visit_matches(&sources[index], &captures, &mut |candidate| {
            Self::visit_ordered_sources(patterns, sources, index + 1, candidate, visitor)
        })
    }
    fn visit_repeated_source(
        pattern: &UPat,
        sources: &[UOp],
        index: usize,
        captures: Captures,
        visitor: &mut dyn FnMut(Captures) -> bool,
    ) -> bool {
        if index == sources.len() {
            return visitor(captures);
        }
        pattern.visit_matches(&sources[index], &captures, &mut |candidate| {
            Self::visit_repeated_source(pattern, sources, index + 1, candidate, visitor)
        })
    }
    fn visit_permuted_sources(
        patterns: &[UPat],
        sources: &[UOp],
        source_index: usize,
        used_patterns: &mut [bool],
        captures: Captures,
        visitor: &mut dyn FnMut(Captures) -> bool,
    ) -> bool {
        if source_index == sources.len() {
            return visitor(captures);
        }
        for pattern_index in 0..patterns.len() {
            if used_patterns[pattern_index] {
                continue;
            }
            used_patterns[pattern_index] = true;
            let matched = patterns[pattern_index].visit_matches(
                &sources[source_index],
                &captures,
                &mut |candidate| {
                    Self::visit_permuted_sources(
                        patterns,
                        sources,
                        source_index + 1,
                        used_patterns,
                        candidate,
                        visitor,
                    )
                },
            );
            used_patterns[pattern_index] = false;
            if matched {
                return true;
            }
        }
        false
    }
    fn with_capture(&self, n: &UOp, mut candidate: Captures) -> Option<Captures> {
        if let Some(name) = &self.name {
            if let Some(old) = candidate.0.get(name) {
                if old != n {
                    return None;
                }
            } else {
                candidate.0.insert(name.clone(), n.clone());
            }
        }
        Some(candidate)
    }
    /// Equality of observable matching behavior. Prepared-only early-reject
    /// hints are deliberately excluded: they may skip an impossible rule but
    /// cannot create another permutation candidate.
    fn match_equivalent(&self, other: &Self) -> bool {
        fn same_operation_predicate(
            left: Option<fn(&Operation) -> bool>,
            right: Option<fn(&Operation) -> bool>,
        ) -> bool {
            match (left, right) {
                (None, None) => true,
                (Some(left), Some(right)) => std::ptr::fn_addr_eq(left, right),
                _ => false,
            }
        }
        fn same_type_predicate(
            left: fn(Option<UType>) -> bool,
            right: fn(Option<UType>) -> bool,
        ) -> bool {
            std::ptr::fn_addr_eq(left, right)
        }
        fn same_sources(left: &SourceConstraint, right: &SourceConstraint) -> bool {
            match (left, right) {
                (SourceConstraint::Exact(left), SourceConstraint::Exact(right))
                | (SourceConstraint::Prefix(left), SourceConstraint::Prefix(right))
                | (SourceConstraint::Permuted(left), SourceConstraint::Permuted(right)) => {
                    same_patterns(left, right)
                }
                (SourceConstraint::Each(left), SourceConstraint::Each(right)) => {
                    left.match_equivalent(right)
                }
                _ => false,
            }
        }
        fn same_patterns(left: &[UPat], right: &[UPat]) -> bool {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| left.match_equivalent(right))
        }

        match (&self.alternatives, &other.alternatives) {
            (None, None) => {}
            (Some(left), Some(right)) if same_patterns(left, right) => {}
            _ => return false,
        }
        self.operations == other.operations
            && same_operation_predicate(self.operation_predicate, other.operation_predicate)
            && self.ty == other.ty
            && self.type_predicates.len() == other.type_predicates.len()
            && self
                .type_predicates
                .iter()
                .copied()
                .zip(other.type_predicates.iter().copied())
                .all(|(left, right)| same_type_predicate(left, right))
            && match (&self.sources, &other.sources) {
                (None, None) => true,
                (Some(left), Some(right)) => same_sources(left, right),
                _ => false,
            }
            && self.name == other.name
    }

    fn possible_root_operations(&self) -> Option<BTreeSet<Operation>> {
        if let Some(operations) = &self.operations {
            return Some(operations.clone());
        }
        let alternatives = self.alternatives.as_ref()?;
        let mut operations = BTreeSet::new();
        for alternative in alternatives {
            operations.extend(alternative.possible_root_operations()?);
        }
        Some(operations)
    }

    fn direct_source_requirements(&self) -> Vec<DirectSourceRequirement> {
        if let Some(predicates) = &self.custom_early_reject {
            return predicates
                .iter()
                .copied()
                .map(DirectSourceRequirement::Predicate)
                .collect();
        }
        let patterns = match &self.sources {
            Some(SourceConstraint::Exact(patterns))
            | Some(SourceConstraint::Prefix(patterns))
            | Some(SourceConstraint::Permuted(patterns)) => patterns,
            // Repeated-source patterns accept zero sources, so no direct
            // source operation is structurally mandatory.
            None | Some(SourceConstraint::Each(_)) => return vec![],
        };
        patterns
            .iter()
            .filter_map(UPat::possible_root_operations)
            .map(DirectSourceRequirement::AnyOf)
            .collect()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Walk {
    BottomUp,
    TopDown,
}
pub type RewriteFn = fn(&Captures, &UOp) -> Option<UOp>;
#[derive(Clone)]
pub struct RewriteRule {
    pub name: &'static str,
    pub priority: i32,
    pub pattern: UPat,
    pub apply: RewriteFn,
}
#[derive(Clone, Debug)]
pub struct RewriteTrace {
    pub rules: Vec<&'static str>,
}

#[derive(Clone)]
enum DirectSourceRequirement {
    AnyOf(BTreeSet<Operation>),
    Predicate(fn(&Operation) -> bool),
}
impl DirectSourceRequirement {
    fn is_satisfied_by(&self, sources: &[UOp]) -> bool {
        sources.iter().any(|source| match self {
            Self::AnyOf(operations) => operations.contains(source.operation()),
            Self::Predicate(predicate) => predicate(source.operation()),
        })
    }
}

struct PreparedRewriteRule<'a> {
    rule: &'a RewriteRule,
    direct_source_requirements: Vec<DirectSourceRequirement>,
}
impl<'a> PreparedRewriteRule<'a> {
    fn new(rule: &'a RewriteRule) -> Self {
        Self {
            direct_source_requirements: rule.pattern.direct_source_requirements(),
            rule,
        }
    }

    fn admits(&self, node: &UOp) -> bool {
        self.direct_source_requirements
            .iter()
            .all(|requirement| requirement.is_satisfied_by(node.sources()))
    }
}

pub fn rewrite(
    root: &UOp,
    rules: &mut [RewriteRule],
    walk: Walk,
) -> Result<(UOp, RewriteTrace), UOpError> {
    const STEP_LIMIT: usize = 128;

    rules.sort_by_key(|r| r.priority);
    let rules = rules
        .iter()
        .map(PreparedRewriteRule::new)
        .collect::<Vec<_>>();
    let mut trace = RewriteTrace { rules: vec![] };
    let mut memo = BTreeMap::new();
    let mut active = BTreeSet::new();
    fn go(
        n: &UOp,
        r: &[PreparedRewriteRule<'_>],
        w: Walk,
        m: &mut BTreeMap<UOp, UOp>,
        active: &mut BTreeSet<UOp>,
        t: &mut RewriteTrace,
    ) -> Result<UOp, UOpError> {
        if let Some(x) = m.get(n) {
            return Ok(x.clone());
        }
        if !active.insert(n.clone()) {
            return Err(UOpError::RewriteCycle);
        }

        let mut x = n.clone();
        let mut seen = BTreeSet::new();
        for _ in 0..STEP_LIMIT {
            if !seen.insert(x.clone()) {
                active.remove(n);
                return Err(UOpError::RewriteCycle);
            }
            let before = x.clone();
            if w == Walk::BottomUp {
                let sources = x
                    .sources()
                    .iter()
                    .map(|source| go(source, r, w, m, active, t))
                    .collect::<Result<Vec<_>, _>>()?;
                if sources.as_slice() != x.sources() {
                    x = UOp::from_operation(x.operation().clone(), x.ty(), sources);
                }
            }

            for rule in r {
                if !rule.admits(&x) {
                    continue;
                }
                let mut next = None;
                rule.rule
                    .pattern
                    .visit_matches(
                        &x,
                        &Captures::default(),
                        &mut |captures| match (rule.rule.apply)(&captures, &x) {
                            Some(candidate) => {
                                if candidate != x {
                                    next = Some(candidate);
                                }
                                true
                            }
                            None => false,
                        },
                    );
                if let Some(next) = next {
                    if !x.is_pure() || !next.is_pure() {
                        active.remove(n);
                        return Err(UOpError::EffectRewrite);
                    }
                    t.rules.push(rule.rule.name);
                    x = next;
                    break;
                }
            }

            if w == Walk::TopDown {
                let sources = x
                    .sources()
                    .iter()
                    .map(|source| go(source, r, w, m, active, t))
                    .collect::<Result<Vec<_>, _>>()?;
                if sources.as_slice() != x.sources() {
                    x = UOp::from_operation(x.operation().clone(), x.ty(), sources);
                }
            }

            if x == before {
                active.remove(n);
                m.insert(n.clone(), x.clone());
                return Ok(x);
            }
        }
        active.remove(n);
        Err(UOpError::RewriteStepLimit)
    }
    let x = go(root, &rules, walk, &mut memo, &mut active, &mut trace)?;
    Ok((x, trace))
}

/// Produces the canonical pure scalar subexpressions used by schedule ABI and
/// cache construction. Artifact decoding deliberately does not call this:
/// historical encoded UOp tables retain their exact node and byte identity.
pub(crate) fn normalize_kernel(root: &UOp) -> Result<UOp, UOpError> {
    root.validate()?;
    let nodes = root.topological()?;
    if !matches!(root.operation(), Operation::Sink)
        || nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::If
                    | Operation::EndIf
                    | Operation::EffectStore(_)
                    | Operation::After(_)
                    | Operation::Barrier
            )
        })
    {
        return Err(UOpError::EffectRewrite);
    }
    if nodes.iter().any(|node| {
        matches!(
            node.operation(),
            Operation::ReduceInit(_)
                | Operation::ReduceAccumulate
                | Operation::ReduceFinalize
                | Operation::TensorGuard(_)
        )
    }) {
        return Ok(root.clone());
    }
    let (normalized, _) = rewrite(root, &mut builtin_rules(), Walk::BottomUp)?;
    normalized.validate()?;
    Ok(normalized)
}

/// Returns whether `literal` is an exact raw scalar identity for the result
/// type of `operation`. Floating identities intentionally stay out of this
/// rewrite set: even `x + 0` can change signed-zero or NaN payload behavior.
fn exact_integral_literal(operation: &UOp, literal: &UOp, bits: u64) -> bool {
    let Some(ty) = operation.ty() else {
        return false;
    };
    if !(ty.scalar.is_integer() || ty.scalar == DType::Bool) || literal.ty() != Some(ty) {
        return false;
    }
    match literal.operation() {
        Operation::Const(LiteralValue::Scalar { dtype, bits: raw }) => {
            *dtype == ty.scalar && *raw == bits
        }
        Operation::Const(LiteralValue::Int(value)) => {
            (*value == 0 && bits == 0) || (*value == 1 && bits == 1)
        }
        _ => false,
    }
}

fn exact_bool_literal(literal: &UOp, value: bool) -> bool {
    matches!(
        (literal.operation(), literal.ty()),
        (
            Operation::Const(LiteralValue::Scalar { dtype, bits }),
            Some(UType { scalar: DType::Bool, lanes: 1 }),
        ) if *dtype == DType::Bool && *bits == u64::from(value)
    )
}

fn is_const_operation(operation: &Operation) -> bool {
    matches!(operation, Operation::Const(_))
}

fn is_add_operation(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Binary(Binary::Add) | Operation::GraphBinary(crate::BinaryOp::Add)
    )
}

fn is_mul_operation(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Binary(Binary::Mul) | Operation::GraphBinary(crate::BinaryOp::Mul)
    )
}

fn is_sub_operation(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Binary(Binary::Sub) | Operation::GraphBinary(crate::BinaryOp::Sub)
    )
}

fn is_foldable_integral_unary_operation(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::GraphUnary(
            crate::UnaryOp::Neg
                | crate::UnaryOp::Abs
                | crate::UnaryOp::Relu
                | crate::UnaryOp::Step
                | crate::UnaryOp::Square
                | crate::UnaryOp::Floor
                | crate::UnaryOp::Ceil
                | crate::UnaryOp::Trunc
                | crate::UnaryOp::Round
                | crate::UnaryOp::Sign
                | crate::UnaryOp::IsNan
                | crate::UnaryOp::IsInf
                | crate::UnaryOp::IsFinite
        )
    )
}

fn is_foldable_integral_binary_operation(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::GraphBinary(
            crate::BinaryOp::Add
                | crate::BinaryOp::Sub
                | crate::BinaryOp::Mul
                | crate::BinaryOp::Maximum
                | crate::BinaryOp::Minimum
                | crate::BinaryOp::BitAnd
                | crate::BinaryOp::BitOr
                | crate::BinaryOp::BitXor
        )
    )
}

fn is_graph_compare_operation(operation: &Operation) -> bool {
    matches!(operation, Operation::GraphCompare(_))
}

fn is_scalar_integral_type(ty: Option<UType>) -> bool {
    ty.is_some_and(|ty| ty.lanes == 1 && (ty.scalar.is_integer() || ty.scalar == DType::Bool))
}

fn exact_storage_carrier(dtype: DType) -> Option<DType> {
    match dtype.itemsize() {
        1 => Some(DType::U8),
        2 => Some(DType::U16),
        4 => Some(DType::U32),
        8 => Some(DType::U64),
        _ => None,
    }
}

fn storage_scalar_literal(node: &UOp) -> Option<(DType, crate::Scalar)> {
    let ty = node.ty()?;
    if ty.lanes != 1 || (!ty.scalar.is_integer() && ty.scalar != DType::Bool) {
        return None;
    }
    let Operation::Const(LiteralValue::Scalar { dtype, bits }) = node.operation() else {
        return None;
    };
    if *dtype != ty.scalar || !scalar_literal_is_valid(Some(ty), *dtype, *bits) {
        return None;
    }
    let carrier = exact_storage_carrier(*dtype)?;
    let value = carrier
        .bitcast_scalar(crate::Scalar::U(*bits), *dtype)
        .ok()?;
    Some((*dtype, value))
}

fn scalar_literal(dtype: DType, value: crate::Scalar) -> Option<UOp> {
    if !dtype.is_integer() && dtype != DType::Bool {
        return None;
    }
    let carrier = exact_storage_carrier(dtype)?;
    let value = dtype.commit_scalar(value);
    let bits = dtype.bitcast_scalar(value, carrier).ok()?.as_u64();
    Some(UOp::scalar_constant(dtype, bits, UType::scalar(dtype)))
}

fn fold_integral_unary(captures: &Captures, node: &UOp) -> Option<UOp> {
    let Operation::GraphUnary(operation) = node.operation() else {
        return None;
    };
    let (dtype, value) = storage_scalar_literal(captures.get("x")?)?;
    let output = node.ty()?;
    if output.lanes != 1
        || !graph_unary_type_is_valid(*operation, Some(UType::scalar(dtype)), Some(output))
    {
        return None;
    }
    if !is_foldable_integral_unary_operation(node.operation()) {
        return None;
    }
    let value = crate::kernel::evaluate_constant_unary(value, dtype, *operation).ok()?;
    scalar_literal(output.scalar, value)
}

fn fold_integral_binary(captures: &Captures, node: &UOp) -> Option<UOp> {
    let Operation::GraphBinary(operation) = node.operation() else {
        return None;
    };
    let (lhs_dtype, lhs) = storage_scalar_literal(captures.get("lhs")?)?;
    let (rhs_dtype, rhs) = storage_scalar_literal(captures.get("rhs")?)?;
    let output = node.ty()?;
    if output.lanes != 1
        || lhs_dtype != output.scalar
        || rhs_dtype != output.scalar
        || (!output.scalar.is_integer() && output.scalar != DType::Bool)
    {
        return None;
    }
    if !is_foldable_integral_binary_operation(node.operation()) {
        return None;
    }
    let value =
        crate::kernel::evaluate_constant_binary(lhs, rhs, output.scalar, *operation).ok()?;
    scalar_literal(output.scalar, value)
}

fn fold_integral_compare(captures: &Captures, node: &UOp) -> Option<UOp> {
    let Operation::GraphCompare(operation) = node.operation() else {
        return None;
    };
    let (lhs_dtype, lhs) = storage_scalar_literal(captures.get("lhs")?)?;
    let (rhs_dtype, rhs) = storage_scalar_literal(captures.get("rhs")?)?;
    if lhs_dtype != rhs_dtype || node.ty() != Some(UType::scalar(DType::Bool)) {
        return None;
    }
    let value = crate::kernel::evaluate_constant_compare(lhs, rhs, *operation);
    scalar_literal(DType::Bool, crate::Scalar::Bool(value))
}

fn fold_bool_logical(captures: &Captures, node: &UOp) -> Option<UOp> {
    let Operation::GraphLogical(operation) = node.operation() else {
        return None;
    };
    let (lhs_dtype, lhs) = storage_scalar_literal(captures.get("lhs")?)?;
    if lhs_dtype != DType::Bool || node.ty() != Some(UType::scalar(DType::Bool)) {
        return None;
    }
    let rhs = match operation {
        crate::LogicalOp::Not => None,
        crate::LogicalOp::And | crate::LogicalOp::Or => {
            let (dtype, value) = storage_scalar_literal(captures.get("rhs")?)?;
            if dtype != DType::Bool {
                return None;
            }
            Some(value)
        }
    };
    let value = crate::kernel::evaluate_constant_logical(lhs, *operation, || {
        rhs.ok_or(crate::Error::InvalidIndex)
    })
    .ok()?;
    scalar_literal(DType::Bool, value)
}

pub fn builtin_rules() -> Vec<RewriteRule> {
    vec![
        RewriteRule {
            name: "add-zero",
            priority: 0,
            pattern: UPat::operation_predicate(is_add_operation)
                .sources(vec![UPat::any().named("x"), UPat::any().named("zero")]),
            apply: |c, n| {
                let x = c.get("x")?;
                let zero = c.get("zero")?;
                n.ty()
                    .filter(|ty| !ty.scalar.is_float())
                    .is_some_and(|ty| typed_positive_zero(zero, Some(ty)))
                    .then(|| x.clone())
                    .filter(|x| x.ty() == n.ty())
            },
        },
        RewriteRule {
            name: "add-zero-left",
            priority: 1,
            pattern: UPat::operation_predicate(is_add_operation)
                .sources(vec![UPat::any().named("zero"), UPat::any().named("x")]),
            apply: |c, n| {
                let x = c.get("x")?;
                let zero = c.get("zero")?;
                n.ty()
                    .filter(|ty| !ty.scalar.is_float())
                    .is_some_and(|ty| typed_positive_zero(zero, Some(ty)))
                    .then(|| x.clone())
                    .filter(|x| x.ty() == n.ty())
            },
        },
        RewriteRule {
            name: "add-zero-untyped-int",
            priority: 2,
            pattern: UPat::operation_predicate(is_add_operation).sources(vec![
                UPat::any().named("x"),
                UPat::op(Operation::Const(LiteralValue::Int(0))),
            ]),
            // Do not turn a floating `-0.0 + 0` into `-0.0`: only the exact
            // non-float domain has this untyped literal identity.
            apply: |c, n| {
                n.ty()
                    .filter(|ty| !ty.scalar.is_float())
                    .and_then(|_| c.get("x").cloned())
            },
        },
        RewriteRule {
            name: "cast-same",
            priority: 2,
            pattern: UPat::op(Operation::Cast).sources(vec![UPat::any().named("x")]),
            apply: |c, n| c.get("x").filter(|x| x.ty() == n.ty()).cloned(),
        },
        RewriteRule {
            name: "where-same",
            priority: 3,
            pattern: UPat::op(Operation::Ternary(Ternary::Where))
                .sources(vec![
                    // A nonconstant condition can have observable failure or
                    // binding behavior even when both arms are the same value.
                    // Keep only the dependency-free, total constant condition.
                    UPat::operation_predicate(is_const_operation),
                    UPat::any().named("x"),
                    UPat::any().named("x"),
                ])
                .custom_early_reject([is_const_operation as fn(&Operation) -> bool]),
            apply: |c, _| c.get("x").cloned(),
        },
        RewriteRule {
            name: "typed-add-zero-right",
            priority: 3,
            pattern: UPat::operation_predicate(is_add_operation)
                .sources(vec![
                    UPat::any().named("x"),
                    UPat::operation_predicate(is_const_operation).named("zero"),
                ])
                .custom_early_reject([is_const_operation as fn(&Operation) -> bool]),
            apply: |c, operation| {
                c.get("x")
                    .filter(|_| {
                        c.get("zero")
                            .is_some_and(|zero| exact_integral_literal(operation, zero, 0))
                    })
                    .cloned()
            },
        },
        RewriteRule {
            name: "typed-add-zero-left",
            priority: 4,
            pattern: UPat::operation_predicate(is_add_operation)
                .sources(vec![
                    UPat::operation_predicate(is_const_operation).named("zero"),
                    UPat::any().named("x"),
                ])
                .custom_early_reject([is_const_operation as fn(&Operation) -> bool]),
            apply: |c, operation| {
                c.get("x")
                    .filter(|_| {
                        c.get("zero")
                            .is_some_and(|zero| exact_integral_literal(operation, zero, 0))
                    })
                    .cloned()
            },
        },
        RewriteRule {
            name: "typed-mul-one",
            priority: 5,
            pattern: UPat::operation_predicate(is_mul_operation)
                .sources(vec![
                    UPat::any().named("x"),
                    UPat::operation_predicate(is_const_operation).named("one"),
                ])
                .custom_early_reject([is_const_operation as fn(&Operation) -> bool]),
            apply: |c, operation| {
                c.get("x")
                    .filter(|x| {
                        x.ty() == operation.ty()
                            && c.get("one")
                                .is_some_and(|one| exact_integral_literal(operation, one, 1))
                    })
                    .cloned()
            },
        },
        RewriteRule {
            name: "typed-mul-one-left",
            priority: 5,
            pattern: UPat::operation_predicate(is_mul_operation)
                .sources(vec![
                    UPat::operation_predicate(is_const_operation).named("one"),
                    UPat::any().named("x"),
                ])
                .custom_early_reject([is_const_operation as fn(&Operation) -> bool]),
            apply: |c, operation| {
                c.get("x")
                    .filter(|x| {
                        x.ty() == operation.ty()
                            && c.get("one")
                                .is_some_and(|one| exact_integral_literal(operation, one, 1))
                    })
                    .cloned()
            },
        },
        RewriteRule {
            name: "typed-sub-zero",
            priority: 5,
            pattern: UPat::operation_predicate(is_sub_operation)
                .sources(vec![
                    UPat::any().named("x"),
                    UPat::operation_predicate(is_const_operation).named("zero"),
                ])
                .custom_early_reject([is_const_operation as fn(&Operation) -> bool]),
            apply: |c, operation| {
                c.get("x")
                    .filter(|x| {
                        x.ty() == operation.ty()
                            && c.get("zero")
                                .is_some_and(|zero| exact_integral_literal(operation, zero, 0))
                    })
                    .cloned()
            },
        },
        RewriteRule {
            name: "typed-where-const",
            priority: 6,
            pattern: UPat::op(Operation::Ternary(Ternary::Where))
                .sources(vec![
                    UPat::operation_predicate(is_const_operation).named("gate"),
                    UPat::any().named("on_true"),
                    UPat::any().named("on_false"),
                ])
                .custom_early_reject([is_const_operation as fn(&Operation) -> bool]),
            apply: |c, _| {
                let gate = c.get("gate")?;
                if exact_bool_literal(gate, true) {
                    c.get("on_true").cloned()
                } else if exact_bool_literal(gate, false) {
                    c.get("on_false").cloned()
                } else {
                    None
                }
            },
        },
        RewriteRule {
            name: "fold-integral-unary",
            priority: 7,
            pattern: UPat::operation_predicate(is_foldable_integral_unary_operation)
                .type_predicate(is_scalar_integral_type)
                .sources(vec![
                    UPat::any()
                        .type_predicate(is_scalar_integral_type)
                        .named("x"),
                ]),
            apply: fold_integral_unary,
        },
        RewriteRule {
            name: "fold-integral-binary",
            priority: 7,
            pattern: UPat::operation_predicate(is_foldable_integral_binary_operation)
                .type_predicate(is_scalar_integral_type)
                .sources(vec![
                    UPat::any()
                        .type_predicate(is_scalar_integral_type)
                        .named("lhs"),
                    UPat::any()
                        .type_predicate(is_scalar_integral_type)
                        .named("rhs"),
                ]),
            apply: fold_integral_binary,
        },
        RewriteRule {
            name: "fold-integral-compare",
            priority: 7,
            pattern: UPat::operation_predicate(is_graph_compare_operation)
                .sources(vec![UPat::any().named("lhs"), UPat::any().named("rhs")]),
            apply: fold_integral_compare,
        },
        RewriteRule {
            name: "fold-bool-logical-not",
            priority: 7,
            pattern: UPat::op(Operation::GraphLogical(crate::LogicalOp::Not))
                .sources(vec![UPat::any().named("lhs")]),
            apply: fold_bool_logical,
        },
        RewriteRule {
            name: "fold-bool-logical-and",
            priority: 7,
            pattern: UPat::op(Operation::GraphLogical(crate::LogicalOp::And))
                .sources(vec![UPat::any().named("lhs"), UPat::any().named("rhs")]),
            apply: fold_bool_logical,
        },
        RewriteRule {
            name: "fold-bool-logical-or",
            priority: 7,
            pattern: UPat::op(Operation::GraphLogical(crate::LogicalOp::Or))
                .sources(vec![UPat::any().named("lhs"), UPat::any().named("rhs")]),
            apply: fold_bool_logical,
        },
    ]
}

/// Addition identities are exact only for a canonical positive raw zero of
/// the operand's own dtype. In particular, this deliberately does not fold a
/// floating `-0.0`, whose signed-zero result can be observable.
fn typed_positive_zero(node: &UOp, ty: Option<UType>) -> bool {
    match node.operation() {
        Operation::Const(LiteralValue::Int(0)) => node.ty() == ty,
        Operation::Const(LiteralValue::Scalar { dtype, bits }) => {
            *bits == 0 && node.ty() == ty && scalar_literal_is_valid(node.ty(), *dtype, *bits)
        }
        _ => false,
    }
}

/// Lowers a scalar-expression pilot from the high-level graph. It is
/// inspectable metadata only; execution remains with the CPU backend.
/// Returns the exact storage payload of a graph-owned scalar constant.  This
/// is shared by scalar metadata lowering and fused elementwise lowering so a
/// constant never loses its F16/BF16/float NaN or signed-zero bits at either
/// boundary.
pub(crate) fn raw_literal_bits(data: &crate::TensorData) -> Result<u64, UOpError> {
    if data.len() != 1 {
        return Err(UOpError::InvalidArgument);
    }
    Ok(match data.storage() {
        crate::Storage::Bool(v) => u64::from(v[0]),
        crate::Storage::I8(v) => v[0] as u8 as u64,
        crate::Storage::U8(v) => v[0] as u64,
        crate::Storage::Float8(v) => v.as_raw()[0] as u64,
        crate::Storage::I16(v) => v[0] as u16 as u64,
        crate::Storage::U16(v) | crate::Storage::F16(v) | crate::Storage::BF16(v) => v[0] as u64,
        crate::Storage::I32(v) => v[0] as u32 as u64,
        crate::Storage::U32(v) => v[0] as u64,
        crate::Storage::I64(v) => v[0] as u64,
        crate::Storage::U64(v) => v[0],
        crate::Storage::F32(v) => v[0].to_bits() as u64,
        crate::Storage::F64(v) => v[0].to_bits(),
    })
}
pub fn lower_graph_scalar(graph: &crate::Graph, output: crate::NodeId) -> Result<UOp, UOpError> {
    fn lower(
        graph: &crate::Graph,
        id: crate::NodeId,
        memo: &mut HashMap<crate::NodeId, UOp>,
    ) -> Result<UOp, UOpError> {
        if let Some(x) = memo.get(&id) {
            return Ok(x.clone());
        }
        if graph
            .shape(id)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            .numel()
            .map_err(|_| UOpError::InvalidArgument)?
            != 1
        {
            return Err(UOpError::InvalidArgument);
        }
        let ty = UType::scalar(graph.dtype(id).map_err(|_| UOpError::UseBeforeDefinition)?);
        let x = match graph.op(id).map_err(|_| UOpError::UseBeforeDefinition)? {
            crate::Op::Input { name } => UOp::from_operation(
                Operation::DefineVar(VariableValue {
                    name: name.clone(),
                    bounds: SymbolicExpr::constant(0),
                }),
                Some(ty),
                vec![],
            ),
            crate::Op::Constant(data) => {
                UOp::scalar_constant(data.dtype(), raw_literal_bits(data)?, ty)
            }
            crate::Op::Cast { input, .. } => UOp::cast(lower(graph, *input, memo)?, ty),
            crate::Op::Bitcast { input, .. } => {
                let source = lower(graph, *input, memo)?;
                if source.ty().map(|source_ty| source_ty.scalar.itemsize())
                    != Some(ty.scalar.itemsize())
                {
                    return Err(UOpError::InvalidArgument);
                }
                UOp::from_operation(Operation::Bitcast, Some(ty), vec![source])
            }
            crate::Op::Contiguous { .. } => return Err(UOpError::InvalidArgument),
            crate::Op::ContiguousBackward { input } => lower(graph, *input, memo)?,
            crate::Op::Unary { op, input } => {
                let u = match op {
                    crate::UnaryOp::Neg => Unary::Neg,
                    crate::UnaryOp::Abs => Unary::Abs,
                    _ => return Err(UOpError::InvalidArgument),
                };
                UOp::unary(u, lower(graph, *input, memo)?)
            }
            crate::Op::Binary { op, lhs, rhs }
                if matches!(*op, crate::BinaryOp::Maximum | crate::BinaryOp::Minimum) =>
            {
                // tinygrad extrema are ordered selects, rather than host max/min
                // intrinsics: MAX is lhs < rhs ? rhs : lhs, while MIN is its
                // inverse predicate.  Keep this lowering in Compare/Where form
                // so scalar UOp renderers cannot reintroduce platform extrema
                // behavior for NaNs or signed-zero ties.
                let lhs = lower(graph, *lhs, memo)?;
                let rhs = lower(graph, *rhs, memo)?;
                let condition = UOp::from_operation(
                    Operation::GraphCompare(crate::CompareOp::Lt),
                    Some(UType::scalar(DType::Bool)),
                    if *op == crate::BinaryOp::Maximum {
                        vec![lhs.clone(), rhs.clone()]
                    } else {
                        vec![rhs.clone(), lhs.clone()]
                    },
                );
                UOp::from_operation(
                    Operation::Ternary(Ternary::Where),
                    Some(ty),
                    vec![condition, rhs, lhs],
                )
            }
            crate::Op::Binary { op, lhs, rhs } => {
                let b = match op {
                    crate::BinaryOp::Add => Binary::Add,
                    crate::BinaryOp::Sub => Binary::Sub,
                    crate::BinaryOp::Mul => Binary::Mul,
                    crate::BinaryOp::FloorDiv => Binary::FloorDiv,
                    crate::BinaryOp::Mod => Binary::Mod,
                    _ => return Err(UOpError::InvalidArgument),
                };
                UOp::binary(b, lower(graph, *lhs, memo)?, lower(graph, *rhs, memo)?)
            }
            crate::Op::Compare { op, lhs, rhs } => {
                let b = match op {
                    crate::CompareOp::Eq => Binary::Eq,
                    crate::CompareOp::Lt => Binary::Lt,
                    crate::CompareOp::Le => Binary::Le,
                    _ => return Err(UOpError::InvalidArgument),
                };
                UOp::binary(b, lower(graph, *lhs, memo)?, lower(graph, *rhs, memo)?)
            }
            crate::Op::Select {
                condition,
                on_true,
                on_false,
            } => UOp::from_operation(
                Operation::Ternary(Ternary::Where),
                Some(ty),
                vec![
                    lower(graph, *condition, memo)?,
                    lower(graph, *on_true, memo)?,
                    lower(graph, *on_false, memo)?,
                ],
            ),
            _ => return Err(UOpError::InvalidArgument),
        };
        memo.insert(id, x.clone());
        Ok(x)
    }
    lower(graph, output, &mut HashMap::new())
}
