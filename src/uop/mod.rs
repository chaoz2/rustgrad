//! Backend-neutral universal operations. This layer is below the tensor graph
//! and above future scheduling/rendering; it deliberately does not execute.
use crate::{DType, Shape, SymbolicExpr};
pub mod artifact;
mod operation;
pub use operation::{
    AddressValue, IndexValue, LiteralValue, MatmulValue, MovementValue, Operation, PrefixScanValue,
    ReductionValue, SortValue, TensorGuardValue, VariableValue,
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
        if self.strides.len() != self.logical_shape.rank() {
            return Err(UOpError::InvalidIndex);
        }
        let numel = self
            .logical_shape
            .numel()
            .map_err(|_| UOpError::InvalidIndex)?;
        if numel == 0 {
            return Ok(());
        }
        let extent = i64::try_from(
            self.source_shape
                .numel()
                .map_err(|_| UOpError::InvalidIndex)?,
        )
        .map_err(|_| UOpError::InvalidIndex)?;
        for index in 0..numel {
            let offset = self.element_offset(index)?;
            if offset < 0 || offset >= extent {
                return Err(UOpError::InvalidIndex);
            }
        }
        Ok(())
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
        for n in nodes {
            validate_one(&n, &mut ranges, &mut ifs)?
        }
        if !ifs.is_empty() || !ranges.is_empty() {
            return Err(UOpError::UnclosedControl);
        }
        Ok(())
    }
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
        Operation::DefineVar(_)
        | Operation::DefineGlobal(_)
        | Operation::DefineLocal(_)
        | Operation::DefineRegister(_)
        | Operation::Special(_) => {}
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
            let valid = if matches!(
                op,
                crate::UnaryOp::IsNan | crate::UnaryOp::IsInf | crate::UnaryOp::IsFinite
            ) {
                matches!(
                    (n.ty(), n.sources()[0].ty()),
                    (Some(output), Some(input))
                        if output.scalar == DType::Bool && output.lanes == input.lanes
                )
            } else {
                same(n)
            };
            if !valid {
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
                input_shape,
                output_shape,
                axis,
                kind,
                output,
                dtype,
                ..
            } = value;
            if input_shape != output_shape
                || (input_shape.rank() != 0 && *axis >= input_shape.rank())
                || (input_shape.rank() == 0 && *axis != 0)
                || (*kind == crate::PrefixScanKind::Sum && *dtype == DType::Bool)
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
        Operation::ReduceInit(_) => {}
        Operation::ReduceAccumulate => {
            if n.ty().is_none() {
                return Err(UOpError::InvalidDType);
            }
        }
        Operation::ReduceFinalize => {
            if n.ty().is_none() {
                return Err(UOpError::InvalidDType);
            }
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
            if input_shape.numel().ok() != Some(*elements)
                || input_shape.rank() > output_shape.rank()
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
pub struct UPat {
    operations: Option<BTreeSet<Operation>>,
    operation_predicate: Option<fn(&Operation) -> bool>,
    ty: Option<UType>,
    sources: Option<Vec<UPat>>,
    name: Option<String>,
    any: bool,
}
impl UPat {
    pub fn any() -> Self {
        Self {
            operations: None,
            operation_predicate: None,
            ty: None,
            sources: None,
            name: None,
            any: true,
        }
    }
    pub fn op(operation: Operation) -> Self {
        let mut x = Self::any();
        x.operations = Some([operation].into());
        x.any = false;
        x
    }
    pub fn ops(operations: impl IntoIterator<Item = Operation>) -> Self {
        let mut x = Self::any();
        x.operations = Some(operations.into_iter().collect());
        x.any = false;
        x
    }
    pub fn operation_predicate(predicate: fn(&Operation) -> bool) -> Self {
        let mut x = Self::any();
        x.operation_predicate = Some(predicate);
        x.any = false;
        x
    }
    pub fn dtype(mut self, ty: UType) -> Self {
        self.ty = Some(ty);
        self
    }
    pub fn sources(mut self, s: Vec<UPat>) -> Self {
        self.sources = Some(s);
        self
    }
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn matches(&self, node: &UOp) -> Option<Captures> {
        let mut c = Captures::default();
        self.match_into(node, &mut c).then_some(c)
    }
    fn match_into(&self, n: &UOp, c: &mut Captures) -> bool {
        if !self.any
            && (self
                .operations
                .as_ref()
                .is_some_and(|operations| !operations.contains(n.operation()))
                || self
                    .operation_predicate
                    .is_some_and(|predicate| !predicate(n.operation())))
        {
            return false;
        }
        if self.ty.is_some_and(|x| n.ty() != Some(x)) {
            return false;
        }
        if let Some(ps) = &self.sources {
            if ps.len() != n.sources().len() {
                return false;
            }
            for (p, s) in ps.iter().zip(n.sources()) {
                if !p.match_into(s, c) {
                    return false;
                }
            }
        }
        if let Some(name) = &self.name {
            if let Some(old) = c.0.get(name) {
                if old != n {
                    return false;
                }
            } else {
                c.0.insert(name.clone(), n.clone());
            }
        }
        true
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
pub fn rewrite(
    root: &UOp,
    rules: &mut [RewriteRule],
    walk: Walk,
) -> Result<(UOp, RewriteTrace), UOpError> {
    rules.sort_by_key(|r| r.priority);
    let mut trace = RewriteTrace { rules: vec![] };
    let mut memo = BTreeMap::new();
    fn go(
        n: &UOp,
        r: &[RewriteRule],
        w: Walk,
        m: &mut BTreeMap<UOp, UOp>,
        t: &mut RewriteTrace,
    ) -> Result<UOp, UOpError> {
        if let Some(x) = m.get(n) {
            return Ok(x.clone());
        }
        let mut x = n.clone();
        if w == Walk::BottomUp {
            x = UOp::from_operation(
                x.operation().clone(),
                x.ty(),
                x.sources()
                    .iter()
                    .map(|s| go(s, r, w, m, t))
                    .collect::<Result<_, _>>()?,
            )
        }
        for rule in r {
            if let Some(c) = rule.pattern.matches(&x)
                && let Some(next) = (rule.apply)(&c, &x)
            {
                if !x.is_pure() || !next.is_pure() {
                    return Err(UOpError::EffectRewrite);
                }
                t.rules.push(rule.name);
                x = next;
                break;
            }
        }
        if w == Walk::TopDown {
            x = UOp::from_operation(
                x.operation().clone(),
                x.ty(),
                x.sources()
                    .iter()
                    .map(|s| go(s, r, w, m, t))
                    .collect::<Result<_, _>>()?,
            )
        }
        m.insert(n.clone(), x.clone());
        Ok(x)
    }
    let x = go(root, rules, walk, &mut memo, &mut trace)?;
    Ok((x, trace))
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
    matches!(
        literal.operation(),
        Operation::Const(LiteralValue::Scalar { dtype, bits: raw })
            if *dtype == ty.scalar && *raw == bits
    )
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

pub fn builtin_rules() -> Vec<RewriteRule> {
    vec![
        RewriteRule {
            name: "add-zero",
            priority: 0,
            pattern: UPat::op(Operation::Binary(Binary::Add))
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
            pattern: UPat::op(Operation::Binary(Binary::Add))
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
            pattern: UPat::op(Operation::Binary(Binary::Add)).sources(vec![
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
            pattern: UPat::op(Operation::Ternary(Ternary::Where)).sources(vec![
                // A nonconstant condition can have observable failure or
                // binding behavior even when both arms are the same value.
                // Keep only the dependency-free, total constant condition.
                UPat::operation_predicate(is_const_operation),
                UPat::any().named("x"),
                UPat::any().named("x"),
            ]),
            apply: |c, _| c.get("x").cloned(),
        },
        RewriteRule {
            name: "typed-add-zero-right",
            priority: 3,
            pattern: UPat::op(Operation::Binary(Binary::Add)).sources(vec![
                UPat::any().named("x"),
                UPat::operation_predicate(is_const_operation).named("zero"),
            ]),
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
            pattern: UPat::op(Operation::Binary(Binary::Add)).sources(vec![
                UPat::operation_predicate(is_const_operation).named("zero"),
                UPat::any().named("x"),
            ]),
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
            pattern: UPat::op(Operation::Binary(Binary::Mul)).sources(vec![
                UPat::any().named("x"),
                UPat::operation_predicate(is_const_operation).named("one"),
            ]),
            apply: |c, operation| {
                c.get("x")
                    .filter(|_| {
                        c.get("one")
                            .is_some_and(|one| exact_integral_literal(operation, one, 1))
                    })
                    .cloned()
            },
        },
        RewriteRule {
            name: "typed-where-const",
            priority: 6,
            pattern: UPat::op(Operation::Ternary(Ternary::Where)).sources(vec![
                UPat::operation_predicate(is_const_operation).named("gate"),
                UPat::any().named("on_true"),
                UPat::any().named("on_false"),
            ]),
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
