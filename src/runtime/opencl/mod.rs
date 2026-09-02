//! Dynamically loaded OpenCL 1.2 runtime foundation.
//!
//! This subsystem intentionally owns its ICD boundary rather than pretending
//! CUDA and OpenCL have identical resource contracts.  Runtime resources are
//! thread-confined (`!Send`/`!Sync`); the injected [`Dispatch`] itself is
//! thread-safe so deterministic mocks can be shared by independent contexts.

mod buffer;
mod dispatch;
mod ffi;
mod guard;
mod narrow;
mod prepared;
mod random;
mod reduction;
mod renderer;
mod resource;
mod transaction;
mod view;

pub use buffer::OpenClBuffer;
pub use dispatch::{
    BufferCopyRegion, BuildInfo, DeviceInfo, Dispatch, OpenClCapabilities, RawBuffer, RawContext,
    RawDevice, RawEvent, RawKernel, RawPlatform, RawProgram, RawQueue,
};
pub(crate) use prepared::OpenClPrefixPlan;
pub use prepared::{CapturedOpenClPrefix, PreparedOpenClPrefix};
pub use renderer::{OpenClBufferAbi, OpenClRenderer, RenderedOpenCl};
pub use resource::{
    OpenClCache, OpenClContext, OpenClDevice, OpenClEvent, OpenClIcd, OpenClKernel, OpenClPlatform,
    OpenClQueue, OpenClTransaction,
};
pub use transaction::{
    GuardedIntegerOp, OPENCL_TRANSACTION_ABI_VERSION, OpenClGuard, OpenClGuardDomain,
    OpenClTransactionAbi,
};

use std::fmt;

/// Structured OpenCL errors. Driver status codes are preserved without
/// requiring an OpenCL SDK at build time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenClError {
    LibraryNotFound {
        tried: Vec<String>,
        detail: String,
    },
    MissingSymbol(&'static str),
    Driver {
        operation: &'static str,
        code: i32,
    },
    Build {
        code: i32,
        log: String,
    },
    InvalidArgument(&'static str),
    InvalidBinding(String),
    IntegerFault {
        operation: GuardedIntegerOp,
        index: usize,
        count: Option<i64>,
        bits: u8,
    },
    Unsupported(String),
    OwnerMismatch,
    StaleGeneration {
        expected: u64,
        actual: u64,
    },
    Closed(&'static str),
    Bounds,
    Overflow,
    NoPlatforms,
    NoDevices,
    Utf8,
}

impl fmt::Display for OpenClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound { tried, detail } => {
                write!(f, "OpenCL library not found (tried {tried:?}): {detail}")
            }
            Self::MissingSymbol(name) => write!(f, "missing OpenCL symbol {name}"),
            Self::Driver { operation, code } => {
                write!(f, "OpenCL {operation} failed with status {code}")
            }
            Self::Build { code, log } => {
                write!(f, "OpenCL program build failed with status {code}: {log}")
            }
            Self::InvalidArgument(reason) => write!(f, "invalid OpenCL argument: {reason}"),
            Self::InvalidBinding(reason) => write!(f, "invalid OpenCL binding: {reason}"),
            Self::IntegerFault {
                operation,
                index,
                count,
                bits,
            } => write!(
                f,
                "OpenCL guarded integer {operation:?} failed at logical index {index} (count {count:?}, {bits} bits)"
            ),
            Self::Unsupported(reason) => write!(f, "unsupported OpenCL kernel: {reason}"),
            Self::OwnerMismatch => write!(f, "OpenCL resource owner mismatch"),
            Self::StaleGeneration { expected, actual } => write!(
                f,
                "stale OpenCL buffer generation {expected}; visible generation is {actual}"
            ),
            Self::Closed(resource) => write!(f, "OpenCL {resource} is closed"),
            Self::Bounds => write!(f, "OpenCL buffer range is out of bounds"),
            Self::Overflow => write!(f, "OpenCL size arithmetic overflow"),
            Self::NoPlatforms => write!(f, "OpenCL ICD reported no platforms"),
            Self::NoDevices => write!(f, "OpenCL platform reported no devices"),
            Self::Utf8 => write!(f, "OpenCL text is not valid UTF-8"),
        }
    }
}

impl std::error::Error for OpenClError {}

#[cfg(test)]
pub(crate) mod tests;
