//! Typed transaction metadata and bounded integer-fault reconstruction.
use super::{MetalError, renderer::unsigned_view};
use crate::{BinaryOp, CompareOp, DType, LogicalOp, Scalar, Shape, UArg, UOp, UOpKind};
use std::collections::BTreeMap;

pub const METAL_TRANSACTION_ABI_VERSION: u32 = 1;
pub(super) const CLEAN_STATUS: u32 = u32::MAX;

/// Guarded integer operation encoded in the staged Metal ABI.
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
    pub(super) fn from_binary(op: BinaryOp) -> Option<Self> {
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

    pub(super) fn is_shift(self) -> bool {
        matches!(self, Self::Shl | Self::Shr)
    }
}

/// One potentially failing operation in producer order.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MetalGuard {
    /// Producer-order identifier within the kernel.
    pub id: u32,
    /// Exact guarded arithmetic operation.
    pub operation: GuardedIntegerOp,
    /// I32 or U32 arithmetic dtype.
    pub dtype: DType,
    #[doc(hidden)]
    pub expression: UOp,
    #[doc(hidden)]
    pub rhs: UOp,
}

/// Deterministic metadata for a transactional elementwise kernel.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MetalTransactionAbi {
    /// Transaction ABI schema version.
    pub version: u32,
    /// Position of the logical output pointer in the ordinary pointer ABI.
    pub output_abi_index: usize,
    /// Static logical output domain.
    pub shape: Shape,
    /// Guards in dependency/producer order.
    pub guards: Vec<MetalGuard>,
    #[doc(hidden)]
    pub evaluation_root: UOp,
}

impl MetalTransactionAbi {
    pub(super) fn analyze(
        value: &UOp,
        output_abi_index: usize,
        shape: Shape,
    ) -> Result<Option<Self>, MetalError> {
        let mut guards = Vec::new();
        for node in value
            .topological()
            .map_err(|error| MetalError::Unsupported(error.to_string()))?
        {
            let UOpKind::GraphBinary(op) = node.kind() else {
                continue;
            };
            let Some(operation) = GuardedIntegerOp::from_binary(*op) else {
                continue;
            };
            let dtype = node
                .ty()
                .ok_or_else(|| MetalError::Unsupported("untyped guarded expression".into()))?
                .scalar;
            if !matches!(dtype, DType::I32 | DType::U32) {
                continue;
            }
            let id = u32::try_from(guards.len()).map_err(|_| MetalError::Overflow)?;
            let rhs = node
                .sources()
                .get(1)
                .ok_or_else(|| MetalError::Unsupported("guarded operation lacks RHS".into()))?
                .clone();
            guards.push(MetalGuard {
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
        let count = u32::try_from(guards.len()).map_err(|_| MetalError::Overflow)?;
        let extent = shape.numel().map_err(|_| MetalError::Overflow)?;
        if extent != 0 {
            let last_index = u32::try_from(extent - 1).map_err(|_| {
                MetalError::Unsupported("transaction status index exceeds u32".into())
            })?;
            let last = last_index
                .checked_mul(count)
                .and_then(|value| value.checked_add(count - 1))
                .ok_or(MetalError::Overflow)?;
            if last == CLEAN_STATUS {
                return Err(MetalError::Unsupported(
                    "transaction status collides with clean sentinel".into(),
                ));
            }
        }
        Ok(Some(Self {
            version: METAL_TRANSACTION_ABI_VERSION,
            output_abi_index,
            shape,
            guards,
            evaluation_root: value.clone(),
        }))
    }

    pub(super) fn guard_count(&self) -> u32 {
        self.guards.len() as u32
    }

    #[cfg(test)]
    pub(super) fn key(&self, index: usize, guard_id: u32) -> Result<u32, MetalError> {
        u32::try_from(index)
            .map_err(|_| MetalError::Overflow)?
            .checked_mul(self.guard_count())
            .and_then(|value| value.checked_add(guard_id))
            .ok_or(MetalError::Overflow)
    }

    pub(super) fn decode(&self, key: u32) -> Result<(usize, &MetalGuard), MetalError> {
        let count = self.guard_count();
        if count == 0 || key == CLEAN_STATUS {
            return Err(MetalError::InvalidBinding(
                "invalid transactional status key".into(),
            ));
        }
        let id = key % count;
        let guard = self
            .guards
            .get(id as usize)
            .filter(|guard| guard.id == id)
            .ok_or_else(|| MetalError::InvalidBinding("unknown guard id".into()))?;
        Ok(((key / count) as usize, guard))
    }

    fn guard_ids(&self) -> BTreeMap<UOp, u32> {
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
    transaction: &MetalTransactionAbi,
    logical: usize,
    mut load: F,
) -> Result<Option<u32>, MetalError>
where
    F: FnMut(&UArg, DType, usize) -> Result<Scalar, MetalError>,
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
    transaction: &MetalTransactionAbi,
    guard: &MetalGuard,
    logical: usize,
    mut load: F,
) -> Result<Scalar, MetalError>
where
    F: FnMut(&UArg, DType, usize) -> Result<Scalar, MetalError>,
{
    match eval(&guard.rhs, logical, &transaction.guard_ids(), &mut load)? {
        Evaluated::Value(value) => Ok(value),
        Evaluated::Fault(id) => Err(MetalError::InvalidBinding(format!(
            "guard {id} failed while reconstructing a later diagnostic"
        ))),
    }
}

fn eval<F>(
    node: &UOp,
    logical: usize,
    guard_ids: &BTreeMap<UOp, u32>,
    load: &mut F,
) -> Result<Evaluated, MetalError>
where
    F: FnMut(&UArg, DType, usize) -> Result<Scalar, MetalError>,
{
    let dtype = node
        .ty()
        .ok_or_else(|| MetalError::InvalidBinding("untyped detail expression".into()))?
        .scalar;
    let value = match node.kind() {
        UOpKind::Const => match node.arg() {
            UArg::Scalar { dtype, bits } => scalar_from_bits(*dtype, *bits)?,
            _ => return Err(MetalError::InvalidBinding("invalid detail constant".into())),
        },
        UOpKind::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| MetalError::InvalidBinding("detail load lacks index".into()))?;
            load(index.arg(), dtype, logical)?
        }
        UOpKind::Cast => {
            let source = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            cast(source, dtype)?
        }
        UOpKind::GraphBinary(op) => {
            let lhs = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            let rhs = match eval(&node.sources()[1], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            if let Some(id) = guard_ids.get(node).copied()
                && invalid_guard(*op, dtype, rhs)
            {
                return Ok(Evaluated::Fault(id));
            }
            integer_binary(*op, dtype, lhs, rhs)?
        }
        UOpKind::GraphCompare(op) => {
            let lhs = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            let rhs = match eval(&node.sources()[1], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            Scalar::Bool(compare(lhs, rhs, *op))
        }
        UOpKind::GraphLogical(op) => {
            let lhs = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value.as_bool(),
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            match op {
                LogicalOp::Not => Scalar::Bool(!lhs),
                LogicalOp::And if !lhs => Scalar::Bool(false),
                LogicalOp::Or if lhs => Scalar::Bool(true),
                LogicalOp::And | LogicalOp::Or => {
                    match eval(&node.sources()[1], logical, guard_ids, load)? {
                        Evaluated::Value(value) => Scalar::Bool(value.as_bool()),
                        Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
                    }
                }
            }
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            let condition = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value.as_bool(),
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            return eval(
                &node.sources()[if condition { 1 } else { 2 }],
                logical,
                guard_ids,
                load,
            );
        }
        _ => return Err(MetalError::Unsupported("detail expression kind".into())),
    };
    Ok(Evaluated::Value(value))
}

fn scalar_from_bits(dtype: DType, bits: u64) -> Result<Scalar, MetalError> {
    Ok(match dtype {
        DType::Bool => Scalar::Bool(bits != 0),
        DType::I32 => Scalar::I(bits as i32 as i64),
        DType::U32 => Scalar::U(bits as u32 as u64),
        _ => return Err(MetalError::Unsupported("transaction scalar dtype".into())),
    })
}

fn cast(value: Scalar, dtype: DType) -> Result<Scalar, MetalError> {
    Ok(match dtype {
        DType::Bool => Scalar::Bool(value.as_bool()),
        DType::I32 => Scalar::I(value.as_i64() as i32 as i64),
        DType::U32 => Scalar::U(value.as_u64() as u32 as u64),
        _ => return Err(MetalError::Unsupported("transaction cast dtype".into())),
    })
}

fn invalid_guard(op: BinaryOp, dtype: DType, rhs: Scalar) -> bool {
    if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        return match dtype {
            DType::I32 => rhs.as_i64() < 0 || rhs.as_u64() >= 32,
            DType::U32 => rhs.as_u64() >= 32,
            _ => true,
        };
    }
    rhs.as_u64() == 0
}

fn integer_binary(
    op: BinaryOp,
    dtype: DType,
    lhs: Scalar,
    rhs: Scalar,
) -> Result<Scalar, MetalError> {
    match dtype {
        DType::U32 => {
            let (lhs, rhs) = (lhs.as_u64() as u32, rhs.as_u64() as u32);
            Ok(Scalar::U(match op {
                BinaryOp::Add => lhs.wrapping_add(rhs),
                BinaryOp::Sub => lhs.wrapping_sub(rhs),
                BinaryOp::Mul => lhs.wrapping_mul(rhs),
                BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::TruncDiv => lhs / rhs,
                BinaryOp::Mod | BinaryOp::FMod => lhs % rhs,
                BinaryOp::Shl => lhs.wrapping_shl(rhs),
                BinaryOp::Shr => lhs.wrapping_shr(rhs),
                _ => return Err(MetalError::Unsupported("detail binary operation".into())),
            } as u64))
        }
        DType::I32 => {
            let (lhs, rhs) = (lhs.as_i64() as i32, rhs.as_i64() as i32);
            Ok(Scalar::I(match op {
                BinaryOp::Add => lhs.wrapping_add(rhs),
                BinaryOp::Sub => lhs.wrapping_sub(rhs),
                BinaryOp::Mul => lhs.wrapping_mul(rhs),
                BinaryOp::Div | BinaryOp::TruncDiv => lhs.wrapping_div(rhs),
                BinaryOp::FloorDiv => lhs.wrapping_div_euclid(rhs),
                BinaryOp::Mod => lhs.wrapping_rem_euclid(rhs),
                BinaryOp::FMod => lhs.wrapping_rem(rhs),
                BinaryOp::Shl => lhs.wrapping_shl(rhs as u32),
                BinaryOp::Shr => lhs.wrapping_shr(rhs as u32),
                _ => return Err(MetalError::Unsupported("detail binary operation".into())),
            } as i64))
        }
        _ => Err(MetalError::Unsupported("transaction integer dtype".into())),
    }
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

pub(super) fn logical_offset(arg: &UArg, logical: usize) -> Result<usize, MetalError> {
    let (input, output, view) = match arg {
        UArg::BufferIndex {
            input_shape,
            output_shape,
            ..
        } => (input_shape, output_shape, None),
        UArg::ViewBufferIndex {
            input_shape,
            output_shape,
            view,
            ..
        } => (input_shape, output_shape, Some(view)),
        _ => return Err(MetalError::InvalidBinding("detail index mismatch".into())),
    };
    let output_strides = output.contiguous_strides();
    let input_strides = input.contiguous_strides();
    let rank_delta = output
        .rank()
        .checked_sub(input.rank())
        .ok_or(MetalError::Bounds)?;
    let mut input_offset = 0usize;
    for axis in 0..input.rank() {
        let coordinate =
            (logical / output_strides[axis + rank_delta]) % output.dims()[axis + rank_delta];
        if input.dims()[axis] != 1 {
            input_offset = input_offset
                .checked_add(
                    coordinate
                        .checked_mul(input_strides[axis])
                        .ok_or(MetalError::Overflow)?,
                )
                .ok_or(MetalError::Overflow)?;
        }
    }
    match view {
        Some(view) => unsigned_view(view)?
            .element_offset(input_offset)
            .map_err(|_| MetalError::Bounds),
        None => Ok(input_offset),
    }
}
