//! Typed metadata and bounded fault reconstruction for transactional kernels.
use super::OpenClError;
use crate::{BinaryOp, CompareOp, DType, LogicalOp, Operation, Scalar, Shape, UArgRef, UOp};
use std::collections::BTreeMap;

pub const OPENCL_TRANSACTION_ABI_VERSION: u32 = 3;
pub const CLEAN_STATUS: u32 = u32::MAX;

/// Guarded integer operation encoded in the staged OpenCL ABI.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GuardedIntegerOp {
    Div,
    FloorDiv,
    TruncDiv,
    Mod,
    FMod,
    Shl,
    Shr,
}

impl GuardedIntegerOp {
    pub(crate) fn from_binary(op: BinaryOp) -> Option<Self> {
        Some(match op {
            BinaryOp::Div => Self::Div,
            BinaryOp::FloorDiv => Self::FloorDiv,
            BinaryOp::TruncDiv => Self::TruncDiv,
            BinaryOp::Mod => Self::Mod,
            BinaryOp::FMod => Self::FMod,
            BinaryOp::Shl => Self::Shl,
            BinaryOp::Shr => Self::Shr,
            _ => return None,
        })
    }

    pub(crate) fn is_shift(self) -> bool {
        matches!(self, Self::Shl | Self::Shr)
    }
}

/// One potentially failing operation in deterministic producer-first order.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenClGuard {
    pub id: u32,
    pub operation: GuardedIntegerOp,
    pub dtype: DType,
    #[doc(hidden)]
    pub expression: UOp,
    #[doc(hidden)]
    pub rhs: UOp,
}

/// Logical ordering domain used by the bounded atomic fault status.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum OpenClGuardDomain {
    /// Elementwise guards are ordered by output logical index.
    Elementwise { shape: Shape },
    /// Fused reduction guards are ordered by their original source index.
    ReductionSource { shape: Shape },
}

impl OpenClGuardDomain {
    pub(crate) fn extent(&self) -> Result<usize, OpenClError> {
        match self {
            Self::Elementwise { shape } | Self::ReductionSource { shape } => {
                shape.numel().map_err(|_| OpenClError::Overflow)
            }
        }
    }
}

/// Complete deterministic metadata for all failing operations in one kernel.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenClTransactionAbi {
    pub version: u32,
    pub output_abi_index: usize,
    pub domain: OpenClGuardDomain,
    pub guards: Vec<OpenClGuard>,
    #[doc(hidden)]
    pub evaluation_root: UOp,
}

impl OpenClTransactionAbi {
    pub(crate) fn analyze(
        value: &UOp,
        output_abi_index: usize,
        domain: OpenClGuardDomain,
    ) -> Result<Option<Self>, OpenClError> {
        let mut guards = Vec::new();
        for node in value
            .topological()
            .map_err(|error| OpenClError::Unsupported(error.to_string()))?
        {
            let Operation::GraphBinary(op) = node.operation() else {
                continue;
            };
            let Some(operation) = GuardedIntegerOp::from_binary(op) else {
                continue;
            };
            let dtype = node
                .ty()
                .ok_or_else(|| OpenClError::Unsupported("untyped guarded expression".into()))?
                .scalar;
            if !matches!(dtype, DType::I32 | DType::U32 | DType::I64 | DType::U64) {
                continue;
            }
            let rhs = node
                .sources()
                .get(1)
                .ok_or_else(|| OpenClError::Unsupported("guarded operation lacks RHS".into()))?
                .clone();
            let id = u32::try_from(guards.len()).map_err(|_| OpenClError::Overflow)?;
            guards.push(OpenClGuard {
                id,
                operation,
                dtype,
                expression: node,
                rhs,
            });
        }
        if guards.is_empty() {
            return Ok(None);
        }
        let count = u32::try_from(guards.len()).map_err(|_| OpenClError::Overflow)?;
        let extent = domain.extent()?;
        if extent != 0 {
            let last_index = u32::try_from(extent - 1).map_err(|_| {
                OpenClError::Unsupported("transaction status index exceeds u32".into())
            })?;
            let last = last_index
                .checked_mul(count)
                .and_then(|value| value.checked_add(count - 1))
                .ok_or(OpenClError::Overflow)?;
            if last == CLEAN_STATUS {
                return Err(OpenClError::Unsupported(
                    "transaction status key collides with clean sentinel".into(),
                ));
            }
        }
        Ok(Some(Self {
            version: OPENCL_TRANSACTION_ABI_VERSION,
            output_abi_index,
            domain,
            guards,
            evaluation_root: value.clone(),
        }))
    }

    pub(crate) fn guard_count(&self) -> u32 {
        self.guards.len() as u32
    }

    #[cfg(test)]
    pub(crate) fn key(&self, index: usize, guard_id: u32) -> Result<u32, OpenClError> {
        u32::try_from(index)
            .map_err(|_| OpenClError::Overflow)?
            .checked_mul(self.guard_count())
            .and_then(|value| value.checked_add(guard_id))
            .ok_or(OpenClError::Overflow)
    }

    pub(crate) fn decode(&self, key: u32) -> Result<(usize, &OpenClGuard), OpenClError> {
        let count = self.guard_count();
        if count == 0 || key == CLEAN_STATUS {
            return Err(OpenClError::InvalidBinding(
                "invalid transactional status key".into(),
            ));
        }
        let id = key % count;
        let guard = self
            .guards
            .get(id as usize)
            .filter(|guard| guard.id == id)
            .ok_or_else(|| OpenClError::InvalidBinding("unknown guard id".into()))?;
        Ok(((key / count) as usize, guard))
    }

    pub(crate) fn guard_ids(&self) -> BTreeMap<UOp, u32> {
        self.guards
            .iter()
            .map(|guard| (guard.expression.clone(), guard.id))
            .collect()
    }
}

enum Evaluated {
    Value(Scalar),
    Fault(u32),
}

#[cfg(test)]
pub(super) fn first_fault_at<F>(
    transaction: &OpenClTransactionAbi,
    logical: usize,
    mut load: F,
) -> Result<Option<u32>, OpenClError>
where
    F: FnMut(UArgRef<'_>, DType, usize) -> Result<Scalar, OpenClError>,
{
    match eval(
        &transaction.evaluation_root,
        logical,
        &transaction.guard_ids(),
        &mut load,
    )? {
        Evaluated::Value(_) => Ok(None),
        Evaluated::Fault(id) => Ok(Some(id)),
    }
}

pub(super) fn detail_rhs_at<F>(
    transaction: &OpenClTransactionAbi,
    guard: &OpenClGuard,
    logical: usize,
    mut load: F,
) -> Result<Scalar, OpenClError>
where
    F: FnMut(UArgRef<'_>, DType, usize) -> Result<Scalar, OpenClError>,
{
    match eval(&guard.rhs, logical, &transaction.guard_ids(), &mut load)? {
        Evaluated::Value(value) => Ok(value),
        Evaluated::Fault(id) => Err(OpenClError::InvalidBinding(format!(
            "guard {id} failed while reconstructing a later diagnostic"
        ))),
    }
}

fn eval<F>(
    node: &UOp,
    logical: usize,
    guard_ids: &BTreeMap<UOp, u32>,
    load: &mut F,
) -> Result<Evaluated, OpenClError>
where
    F: FnMut(UArgRef<'_>, DType, usize) -> Result<Scalar, OpenClError>,
{
    let dtype = node
        .ty()
        .ok_or_else(|| OpenClError::InvalidBinding("untyped detail expression".into()))?
        .scalar;
    let value = match node.operation() {
        Operation::Const => match node.arg() {
            UArgRef::Scalar { dtype, bits } => scalar_from_bits(*dtype, *bits),
            UArgRef::Int(value) => Scalar::I(*value),
            _ => {
                return Err(OpenClError::InvalidBinding(
                    "invalid detail constant".into(),
                ));
            }
        },
        Operation::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::InvalidBinding("detail load lacks index".into()))?;
            load(index.arg(), dtype, logical)?
        }
        Operation::Cast => {
            let source = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            cast(source, dtype)
        }
        Operation::GraphUnary(op) => {
            let source = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            match (op, dtype) {
                (crate::UnaryOp::Neg, DType::I32) => {
                    Scalar::I((source.as_i64() as i32).wrapping_neg() as i64)
                }
                (crate::UnaryOp::Neg, DType::I64) => Scalar::I(source.as_i64().wrapping_neg()),
                _ => return Err(OpenClError::Unsupported("detail unary expression".into())),
            }
        }
        Operation::GraphBinary(op) => {
            let lhs = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            let rhs = match eval(&node.sources()[1], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            if let Some(id) = guard_ids.get(node).copied()
                && invalid_guard(op, dtype, rhs)
            {
                return Ok(Evaluated::Fault(id));
            }
            integer_binary(op, dtype, lhs, rhs)?
        }
        Operation::GraphCompare(op) => {
            let lhs = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            let rhs = match eval(&node.sources()[1], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            Scalar::Bool(compare(lhs, rhs, op))
        }
        Operation::GraphLogical(op) => {
            let lhs = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value.as_bool(),
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            match op {
                LogicalOp::Not => Scalar::Bool(!lhs),
                LogicalOp::And if !lhs => Scalar::Bool(false),
                LogicalOp::Or if lhs => Scalar::Bool(true),
                LogicalOp::And | LogicalOp::Or => {
                    let rhs = match eval(&node.sources()[1], logical, guard_ids, load)? {
                        Evaluated::Value(value) => value.as_bool(),
                        Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
                    };
                    Scalar::Bool(rhs)
                }
            }
        }
        Operation::Ternary(crate::uop::Ternary::Where) => {
            let condition = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            let branch = if condition.as_bool() { 1 } else { 2 };
            return eval(&node.sources()[branch], logical, guard_ids, load);
        }
        _ => return Err(OpenClError::Unsupported("detail expression kind".into())),
    };
    Ok(Evaluated::Value(value))
}

fn scalar_from_bits(dtype: DType, bits: u64) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(bits != 0),
        DType::I32 => Scalar::I(bits as i32 as i64),
        DType::U32 => Scalar::U(bits as u32 as u64),
        DType::I64 => Scalar::I(bits as i64),
        DType::U64 => Scalar::U(bits),
        DType::F32 => Scalar::F(f32::from_bits(bits as u32) as f64),
        DType::F64 => Scalar::F(f64::from_bits(bits)),
        _ => Scalar::U(bits),
    }
}

fn cast(value: Scalar, dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(value.as_bool()),
        DType::I32 => Scalar::I(value.as_i64() as i32 as i64),
        DType::U32 => Scalar::U(value.as_u64() as u32 as u64),
        DType::I64 => Scalar::I(value.as_i64()),
        DType::U64 => Scalar::U(value.as_u64()),
        DType::F32 => Scalar::F(value.as_f64() as f32 as f64),
        DType::F64 => Scalar::F(value.as_f64()),
        _ => value,
    }
}

fn invalid_guard(op: BinaryOp, dtype: DType, rhs: Scalar) -> bool {
    if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        return match dtype {
            DType::I32 | DType::I64 => rhs.as_i64() < 0 || rhs.as_u64() >= dtype.bits() as u64,
            _ => rhs.as_u64() >= dtype.bits() as u64,
        };
    }
    rhs.as_u64() == 0
}

fn integer_binary(
    op: BinaryOp,
    dtype: DType,
    lhs: Scalar,
    rhs: Scalar,
) -> Result<Scalar, OpenClError> {
    if matches!(dtype, DType::U32 | DType::U64) {
        let (lhs, rhs) = (lhs.as_u64(), rhs.as_u64());
        let value = match op {
            BinaryOp::Add => lhs.wrapping_add(rhs),
            BinaryOp::Sub => lhs.wrapping_sub(rhs),
            BinaryOp::Mul => lhs.wrapping_mul(rhs),
            BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::TruncDiv => lhs / rhs,
            BinaryOp::Mod | BinaryOp::FMod => lhs % rhs,
            BinaryOp::Shl => lhs.wrapping_shl(rhs as u32),
            BinaryOp::Shr => lhs.wrapping_shr(rhs as u32),
            _ => return Err(OpenClError::Unsupported("detail binary expression".into())),
        };
        return Ok(Scalar::U(if dtype == DType::U32 {
            value as u32 as u64
        } else {
            value
        }));
    }
    if matches!(dtype, DType::I32 | DType::I64) {
        let (lhs, rhs) = (lhs.as_i64(), rhs.as_i64());
        let value = match op {
            BinaryOp::Add => lhs.wrapping_add(rhs),
            BinaryOp::Sub => lhs.wrapping_sub(rhs),
            BinaryOp::Mul => lhs.wrapping_mul(rhs),
            BinaryOp::Div | BinaryOp::TruncDiv => lhs.wrapping_div(rhs),
            BinaryOp::FloorDiv => lhs.wrapping_div_euclid(rhs),
            BinaryOp::Mod => lhs.wrapping_rem_euclid(rhs),
            BinaryOp::FMod => lhs.wrapping_rem(rhs),
            BinaryOp::Shl => lhs.wrapping_shl(rhs as u32),
            BinaryOp::Shr => lhs.wrapping_shr(rhs as u32),
            _ => return Err(OpenClError::Unsupported("detail binary expression".into())),
        };
        return Ok(Scalar::I(if dtype == DType::I32 {
            value as i32 as i64
        } else {
            value
        }));
    }
    Err(OpenClError::Unsupported(
        "guarded detail requires exact integer dtype".into(),
    ))
}

fn compare(lhs: Scalar, rhs: Scalar, op: CompareOp) -> bool {
    let ordering = match (lhs, rhs) {
        (Scalar::I(lhs), Scalar::I(rhs)) => lhs.cmp(&rhs),
        (Scalar::U(lhs), Scalar::U(rhs)) => lhs.cmp(&rhs),
        _ => lhs.as_i64().cmp(&rhs.as_i64()),
    };
    match op {
        CompareOp::Eq => ordering.is_eq(),
        CompareOp::Ne => !ordering.is_eq(),
        CompareOp::Lt => ordering.is_lt(),
        CompareOp::Le => ordering.is_le(),
        CompareOp::Gt => ordering.is_gt(),
        CompareOp::Ge => ordering.is_ge(),
    }
}

pub(super) fn logical_offset(arg: UArgRef<'_>, logical: usize) -> Result<usize, OpenClError> {
    let (input, output, view) = match arg {
        UArgRef::BufferIndex {
            input_shape,
            output_shape,
            ..
        } => (input_shape, output_shape, None),
        UArgRef::ViewBufferIndex {
            input_shape,
            output_shape,
            view,
            ..
        } => (input_shape, output_shape, Some(view)),
        _ => return Err(OpenClError::InvalidBinding("detail index mismatch".into())),
    };
    let output_strides = output.contiguous_strides();
    let input_strides = input.contiguous_strides();
    let rank_delta = output
        .rank()
        .checked_sub(input.rank())
        .ok_or(OpenClError::Bounds)?;
    let mut input_offset = 0usize;
    for axis in 0..input.rank() {
        let coordinate =
            (logical / output_strides[axis + rank_delta]) % output.dims()[axis + rank_delta];
        if input.dims()[axis] != 1 {
            input_offset = input_offset
                .checked_add(
                    coordinate
                        .checked_mul(input_strides[axis])
                        .ok_or(OpenClError::Overflow)?,
                )
                .ok_or(OpenClError::Overflow)?;
        }
    }
    match view {
        Some(view) => view
            .element_offset(input_offset)
            .map_err(|_| OpenClError::Bounds)
            .and_then(|offset| usize::try_from(offset).map_err(|_| OpenClError::Bounds)),
        None => Ok(input_offset),
    }
}
