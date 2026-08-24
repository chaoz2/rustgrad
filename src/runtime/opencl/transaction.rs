//! Typed metadata for OpenCL kernels whose integer faults require staged commit.
use crate::{BinaryOp, DType, UArg};

pub const OPENCL_TRANSACTION_ABI_VERSION: u32 = 1;
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

/// Complete deterministic metadata needed to interpret the first failing lane.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenClTransactionAbi {
    /// Status/scratch argument layout version.
    pub version: u32,
    /// The sole guarded operation in this kernel.
    pub operation: GuardedIntegerOp,
    /// Promoted result and operand dtype.
    pub dtype: DType,
    /// ABI slot replaced by provisional storage during compute.
    pub output_abi_index: usize,
    /// ABI slot read to reconstruct an invalid divisor or shift count.
    pub rhs_abi_index: usize,
    /// Exact broadcast/view mapping for the retained right operand.
    pub rhs_index: UArg,
}
