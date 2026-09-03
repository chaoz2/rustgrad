//! SDK-free Apple Metal runtime and static MSL lowering.
//!
//! Safe resources are thread-confined and never expose Objective-C handles.
//! The native adapter dynamically loads the system frameworks on macOS; the
//! default test suite uses an injected byte-accurate dispatch instead.

mod buffer;
mod dispatch;
mod ffi;
mod guard;
mod prepared;
mod random;
mod renderer;
mod resource;
mod scoreboard;
mod session;
mod transaction;

pub use buffer::MetalBuffer;
pub use dispatch::{MetalCapabilities, MetalDeviceInfo};
pub use prepared::{CapturedMetalPrefix, MetalPrefixPlan, PreparedMetalPrefix};
pub use renderer::{
    METAL_ABI_VERSION, METAL_APPEND_STATE_RENDERER_VERSION, METAL_RENDERER_VERSION,
    MetalAppendStateAbi, MetalBufferAbi, MetalRenderer, RenderedMetal,
};
pub use resource::{
    MetalCache, MetalCommand, MetalCommandQueue, MetalCompletion, MetalDevice, MetalDiscovery,
    MetalLibrary, MetalPipeline, MetalRuntime, MetalTransaction,
};
pub use scoreboard::{
    METAL_SESSION_SCOREBOARD_FORMAT_VERSION, MetalHostWallTimeSummary, MetalScoreboardContext,
    MetalScoreboardError, MetalScoreboardInput, MetalScoreboardInputKind, MetalScoreboardRun,
    MetalSessionScoreboard, MetalSessionScoreboardReport,
};
pub use session::{
    MetalAppendStateInferencePlan, MetalDevicePreparationReport, MetalDeviceRun,
    MetalDeviceRunReport, MetalDeviceSession, MetalDeviceSessionPlan, MetalDeviceSessionSummary,
    MetalInferencePlan, MetalPlanOptions, MetalStatefulInferencePlan,
};
pub use transaction::{
    GuardedIntegerOp, METAL_INDEXED_MOVEMENT_ABI_VERSION, METAL_TRANSACTION_ABI_VERSION,
    MetalGuard, MetalIndexedMovementAbi, MetalTransactionAbi,
};

use std::fmt;

/// Structured Metal errors. Native diagnostics are bounded and safe APIs do
/// not include raw Objective-C object addresses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalError {
    PlatformUnsupported,
    FrameworkUnavailable {
        framework: &'static str,
        detail: String,
    },
    MissingSymbol(&'static str),
    NoDevices,
    Driver {
        operation: &'static str,
        detail: String,
    },
    Build {
        diagnostic: String,
    },
    InvalidArgument(&'static str),
    InvalidBinding(String),
    Unsupported(String),
    OwnerMismatch,
    StaleGeneration {
        expected: u64,
        actual: u64,
    },
    IntegerFault {
        operation: GuardedIntegerOp,
        index: usize,
        count: Option<i64>,
        bits: usize,
    },
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        value: i32,
        dim: usize,
    },
    Closed(&'static str),
    Bounds,
    Overflow,
    Utf8,
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlatformUnsupported => write!(f, "Metal is available only on macOS"),
            Self::FrameworkUnavailable { framework, detail } => {
                write!(f, "Metal framework {framework} is unavailable: {detail}")
            }
            Self::MissingSymbol(symbol) => write!(f, "missing Metal symbol {symbol}"),
            Self::NoDevices => write!(f, "Metal reported no devices"),
            Self::Driver { operation, detail } => {
                write!(f, "Metal {operation} failed: {detail}")
            }
            Self::Build { diagnostic } => {
                write!(f, "Metal source compilation failed: {diagnostic}")
            }
            Self::InvalidArgument(reason) => write!(f, "invalid Metal argument: {reason}"),
            Self::InvalidBinding(reason) => write!(f, "invalid Metal binding: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported Metal kernel: {reason}"),
            Self::OwnerMismatch => write!(f, "Metal resource owner mismatch"),
            Self::StaleGeneration { expected, actual } => write!(
                f,
                "stale Metal buffer generation {expected}; visible generation is {actual}"
            ),
            Self::IntegerFault {
                operation,
                index,
                count,
                bits,
            } => write!(
                f,
                "Metal guarded integer {operation:?} failed at logical index {index} (count {count:?}, {bits} bits)"
            ),
            Self::IndexOutOfBounds {
                axis,
                index,
                value,
                dim,
            } => write!(
                f,
                "Metal indexed movement axis {axis} has value {value} at logical index {index}, outside [0, {dim})"
            ),
            Self::Closed(resource) => write!(f, "Metal {resource} is closed"),
            Self::Bounds => write!(f, "Metal buffer range is out of bounds"),
            Self::Overflow => write!(f, "Metal size arithmetic overflow"),
            Self::Utf8 => write!(f, "Metal text is not valid UTF-8"),
        }
    }
}

impl std::error::Error for MetalError {}

#[cfg(test)]
mod tests;
