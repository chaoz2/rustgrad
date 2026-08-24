//! Typed WebGPU transaction metadata and bounded fault reconstruction.
use super::{WebGpuError, renderer::unsigned_view};
use crate::{BinaryOp, CompareOp, DType, LogicalOp, Scalar, Shape, UArg, UOp, UOpKind};
use std::collections::BTreeMap;

/// Schema version for guarded WebGPU candidate/status execution.
pub const WEBGPU_TRANSACTION_ABI_VERSION: u32 = 1;
pub(super) const CLEAN_STATUS: u32 = u32::MAX;

/// Potentially failing integer operation encoded in the WebGPU status ABI.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GuardedIntegerOp {
    /// Integer division with truncating signed semantics.
    Div,
    /// Euclidean signed division and ordinary unsigned division.
    FloorDiv,
    /// Explicit truncating division.
    TruncDiv,
    /// Euclidean signed remainder and ordinary unsigned remainder.
    Mod,
    /// Truncating signed remainder and ordinary unsigned remainder.
    FMod,
    /// Checked left shift.
    Shl,
    /// Checked right shift.
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

/// One guarded expression in dependency/producer order.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WebGpuGuard {
    /// Producer-order identifier within this kernel.
    pub id: u32,
    /// Exact guarded operation.
    pub operation: GuardedIntegerOp,
    /// I32 or U32 operation dtype.
    pub dtype: DType,
    #[doc(hidden)]
    pub expression: UOp,
    #[doc(hidden)]
    pub rhs: UOp,
}

/// Deterministic metadata for one transactional elementwise kernel.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WebGpuTransactionAbi {
    /// Transaction schema version.
    pub version: u32,
    /// Position of the logical output in the ordinary ordered buffer ABI.
    pub output_abi_index: usize,
    /// Static logical output shape.
    pub shape: Shape,
    /// Guards in dependency/producer order.
    pub guards: Vec<WebGpuGuard>,
    #[doc(hidden)]
    pub evaluation_root: UOp,
}

impl WebGpuTransactionAbi {
    pub(super) fn analyze(
        value: &UOp,
        output_abi_index: usize,
        shape: Shape,
    ) -> Result<Option<Self>, WebGpuError> {
        let mut guards = Vec::new();
        for node in value
            .topological()
            .map_err(|error| WebGpuError::Unsupported(error.to_string()))?
        {
            let UOpKind::GraphBinary(op) = node.kind() else {
                continue;
            };
            let Some(operation) = GuardedIntegerOp::from_binary(*op) else {
                continue;
            };
            let dtype = node
                .ty()
                .ok_or_else(|| WebGpuError::Unsupported("untyped guarded expression".into()))?
                .scalar;
            if !matches!(dtype, DType::I32 | DType::U32) {
                return Err(WebGpuError::Unsupported(format!(
                    "guarded {operation:?} requires I32 or U32, got {dtype:?}"
                )));
            }
            let id = u32::try_from(guards.len()).map_err(|_| WebGpuError::Overflow)?;
            let rhs = node
                .sources()
                .get(1)
                .ok_or_else(|| WebGpuError::Unsupported("guarded operation lacks RHS".into()))?
                .clone();
            guards.push(WebGpuGuard {
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
        let guard_count = u32::try_from(guards.len()).map_err(|_| WebGpuError::Overflow)?;
        let extent = shape.numel().map_err(|_| WebGpuError::Overflow)?;
        if extent != 0 {
            let last_index = u32::try_from(extent - 1).map_err(|_| {
                WebGpuError::Unsupported("transaction status index exceeds u32".into())
            })?;
            let last = last_index
                .checked_mul(guard_count)
                .and_then(|value| value.checked_add(guard_count - 1))
                .ok_or(WebGpuError::Overflow)?;
            if last == CLEAN_STATUS {
                return Err(WebGpuError::Unsupported(
                    "transaction status collides with clean sentinel".into(),
                ));
            }
        }
        let transaction = Self {
            version: WEBGPU_TRANSACTION_ABI_VERSION,
            output_abi_index,
            shape,
            guards,
            evaluation_root: value.clone(),
        };
        transaction.validate_launch(extent, output_abi_index)?;
        Ok(Some(transaction))
    }

    pub(super) fn validate_launch(
        &self,
        extent: usize,
        output_abi_index: usize,
    ) -> Result<(), WebGpuError> {
        if self.version != WEBGPU_TRANSACTION_ABI_VERSION
            || self.output_abi_index != output_abi_index
            || self.shape.numel().map_err(|_| WebGpuError::Overflow)? != extent
            || self.guards.is_empty()
        {
            return Err(WebGpuError::InvalidBinding(
                "transaction metadata identity mismatch".into(),
            ));
        }
        let count = u32::try_from(self.guards.len()).map_err(|_| WebGpuError::Overflow)?;
        for (position, guard) in self.guards.iter().enumerate() {
            let expected_id = u32::try_from(position).map_err(|_| WebGpuError::Overflow)?;
            let UOpKind::GraphBinary(op) = guard.expression.kind() else {
                return Err(WebGpuError::InvalidBinding(
                    "transaction guard expression mismatch".into(),
                ));
            };
            if guard.id != expected_id
                || GuardedIntegerOp::from_binary(*op) != Some(guard.operation)
                || !matches!(guard.dtype, DType::I32 | DType::U32)
                || guard.expression.ty().map(|ty| ty.scalar) != Some(guard.dtype)
                || guard.expression.sources().get(1) != Some(&guard.rhs)
            {
                return Err(WebGpuError::InvalidBinding(
                    "transaction guard metadata mismatch".into(),
                ));
            }
        }
        if extent != 0 {
            let last_index = u32::try_from(extent - 1).map_err(|_| WebGpuError::Overflow)?;
            let last = last_index
                .checked_mul(count)
                .and_then(|value| value.checked_add(count - 1))
                .ok_or(WebGpuError::Overflow)?;
            if last == CLEAN_STATUS {
                return Err(WebGpuError::InvalidBinding(
                    "transaction status collides with clean sentinel".into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn guard_count(&self) -> u32 {
        self.guards.len() as u32
    }

    #[cfg(test)]
    pub(super) fn key(&self, index: usize, guard_id: u32) -> Result<u32, WebGpuError> {
        u32::try_from(index)
            .map_err(|_| WebGpuError::Overflow)?
            .checked_mul(self.guard_count())
            .and_then(|value| value.checked_add(guard_id))
            .ok_or(WebGpuError::Overflow)
    }

    pub(super) fn decode(&self, key: u32) -> Result<(usize, &WebGpuGuard), WebGpuError> {
        let count = self.guard_count();
        if count == 0 || key == CLEAN_STATUS {
            return Err(WebGpuError::InvalidBinding(
                "invalid transactional status key".into(),
            ));
        }
        let id = key % count;
        let guard = self
            .guards
            .get(id as usize)
            .filter(|guard| guard.id == id)
            .ok_or_else(|| WebGpuError::InvalidBinding("unknown guard id".into()))?;
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
    transaction: &WebGpuTransactionAbi,
    logical: usize,
    mut load: F,
) -> Result<Option<u32>, WebGpuError>
where
    F: FnMut(&UArg, DType, usize) -> Result<Scalar, WebGpuError>,
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
    transaction: &WebGpuTransactionAbi,
    guard: &WebGpuGuard,
    logical: usize,
    mut load: F,
) -> Result<Scalar, WebGpuError>
where
    F: FnMut(&UArg, DType, usize) -> Result<Scalar, WebGpuError>,
{
    match eval(&guard.rhs, logical, &transaction.guard_ids(), &mut load)? {
        Evaluated::Value(value) => Ok(value),
        Evaluated::Fault(id) => Err(WebGpuError::InvalidBinding(format!(
            "guard {id} failed while reconstructing a later diagnostic"
        ))),
    }
}

fn eval<F>(
    node: &UOp,
    logical: usize,
    guard_ids: &BTreeMap<UOp, u32>,
    load: &mut F,
) -> Result<Evaluated, WebGpuError>
where
    F: FnMut(&UArg, DType, usize) -> Result<Scalar, WebGpuError>,
{
    let dtype = node
        .ty()
        .ok_or_else(|| WebGpuError::InvalidBinding("untyped detail expression".into()))?
        .scalar;
    let value = match node.kind() {
        UOpKind::Const => match node.arg() {
            UArg::Scalar { dtype, bits } => scalar_from_bits(*dtype, *bits)?,
            _ => {
                return Err(WebGpuError::InvalidBinding(
                    "invalid detail constant".into(),
                ));
            }
        },
        UOpKind::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| WebGpuError::InvalidBinding("detail load lacks index".into()))?;
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
        UOpKind::Binary(op) => {
            let lhs = match eval(&node.sources()[0], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            let rhs = match eval(&node.sources()[1], logical, guard_ids, load)? {
                Evaluated::Value(value) => value,
                Evaluated::Fault(id) => return Ok(Evaluated::Fault(id)),
            };
            use crate::uop::Binary::{Add, Eq, Le, Lt, Mul, Sub};
            match op {
                Add => integer_binary(BinaryOp::Add, dtype, lhs, rhs)?,
                Sub => integer_binary(BinaryOp::Sub, dtype, lhs, rhs)?,
                Mul => integer_binary(BinaryOp::Mul, dtype, lhs, rhs)?,
                Eq => Scalar::Bool(compare(lhs, rhs, CompareOp::Eq)),
                Lt => Scalar::Bool(compare(lhs, rhs, CompareOp::Lt)),
                Le => Scalar::Bool(compare(lhs, rhs, CompareOp::Le)),
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "transaction detail core binary operation".into(),
                    ));
                }
            }
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
        _ => {
            return Err(WebGpuError::Unsupported(
                "transaction detail expression kind".into(),
            ));
        }
    };
    Ok(Evaluated::Value(value))
}

fn scalar_from_bits(dtype: DType, bits: u64) -> Result<Scalar, WebGpuError> {
    Ok(match dtype {
        DType::Bool => Scalar::Bool(bits != 0),
        DType::I32 => Scalar::I(bits as i32 as i64),
        DType::U32 => Scalar::U(bits as u32 as u64),
        _ => {
            return Err(WebGpuError::Unsupported("transaction scalar dtype".into()));
        }
    })
}

fn cast(value: Scalar, dtype: DType) -> Result<Scalar, WebGpuError> {
    Ok(match dtype {
        DType::Bool => Scalar::Bool(value.as_bool()),
        DType::I32 => Scalar::I(value.as_i64() as i32 as i64),
        DType::U32 => Scalar::U(value.as_u64() as u32 as u64),
        _ => {
            return Err(WebGpuError::Unsupported("transaction cast dtype".into()));
        }
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
) -> Result<Scalar, WebGpuError> {
    match dtype {
        DType::Bool => {
            let (lhs, rhs) = (lhs.as_bool(), rhs.as_bool());
            Ok(Scalar::Bool(match op {
                BinaryOp::Add => lhs || rhs,
                BinaryOp::Sub => lhs ^ rhs,
                BinaryOp::Mul => lhs && rhs,
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "transaction detail bool operation".into(),
                    ));
                }
            }))
        }
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
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "transaction detail binary operation".into(),
                    ));
                }
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
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "transaction detail binary operation".into(),
                    ));
                }
            } as i64))
        }
        _ => Err(WebGpuError::Unsupported("transaction integer dtype".into())),
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

pub(super) fn logical_offset(arg: &UArg, logical: usize) -> Result<usize, WebGpuError> {
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
        _ => {
            return Err(WebGpuError::InvalidBinding(
                "transaction detail index mismatch".into(),
            ));
        }
    };
    let output_strides = output.contiguous_strides();
    let input_strides = input.contiguous_strides();
    let rank_delta = output
        .rank()
        .checked_sub(input.rank())
        .ok_or(WebGpuError::Bounds)?;
    let mut input_offset = 0usize;
    for (axis, (&input_dim, &input_stride)) in input.dims().iter().zip(&input_strides).enumerate() {
        let output_axis = axis + rank_delta;
        let output_dim = output.dims()[output_axis];
        if output_dim == 0 {
            return Err(WebGpuError::Bounds);
        }
        let coordinate = (logical / output_strides[output_axis]) % output_dim;
        if input_dim != 1 {
            input_offset = input_offset
                .checked_add(
                    coordinate
                        .checked_mul(input_stride)
                        .ok_or(WebGpuError::Overflow)?,
                )
                .ok_or(WebGpuError::Overflow)?;
        }
    }
    match view {
        Some(view) => unsigned_view(view)?
            .element_offset(input_offset)
            .map_err(|_| WebGpuError::Bounds),
        None => Ok(input_offset),
    }
}
