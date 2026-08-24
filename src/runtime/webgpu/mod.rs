//! Portable WebGPU ownership and deterministic static WGSL lowering.
//!
//! The safe API is deliberately handle-free and thread-confined. Native C ABI
//! loading is isolated in `ffi`; deterministic tests inject the private typed
//! dispatch seam and execute retained lowered UOps rather than `CpuBackend`.

mod buffer;
mod dispatch;
mod ffi;
mod guard;
mod narrow;
mod renderer;
mod resource;
mod transaction;

pub use buffer::WebGpuBuffer;
pub use dispatch::{WebGpuAdapterInfo, WebGpuBackend, WebGpuCapabilities};
pub use narrow::WEBGPU_NARROW_ABI_VERSION;
pub use renderer::{
    RenderedWgsl, WEBGPU_ABI_VERSION, WEBGPU_STATUS_VERSION, WGSL_RENDERER_VERSION, WgslBufferAbi,
    WgslRenderer,
};
pub use resource::{
    WebGpuAdapter, WebGpuCache, WebGpuCommand, WebGpuCompletion, WebGpuDevice, WebGpuInstance,
    WebGpuPipeline, WebGpuQueue, WebGpuRuntime, WebGpuShader, WebGpuTransaction,
};
pub use transaction::{
    GuardedIntegerOp, WEBGPU_TRANSACTION_ABI_VERSION, WebGpuGuard, WebGpuTransactionAbi,
};

use std::fmt;

/// Structured WebGPU failures without native handles or unbounded diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WebGpuError {
    /// No candidate dynamic WebGPU library could be loaded.
    LibraryUnavailable {
        /// Candidate names attempted in deterministic order.
        tried: Vec<String>,
    },
    /// A library was found but its unversioned callback ABI cannot be used safely.
    NativeAbiUnsupported {
        /// Bounded explanation of the incompatible provider ABI.
        detail: String,
    },
    /// The selected library lacks a required C entry point.
    MissingSymbol(&'static str),
    /// No adapter matched discovery.
    NoAdapters,
    /// A typed dispatch operation failed.
    Driver {
        /// Stable typed operation name.
        operation: &'static str,
        /// Bounded provider or injected-dispatch detail.
        detail: String,
    },
    /// WGSL compilation or validation failed.
    Build {
        /// Bounded shader validation or compilation diagnostic.
        diagnostic: String,
    },
    /// A scalar, size, or geometry argument is invalid.
    InvalidArgument(&'static str),
    /// Ordered resource metadata does not match the compiled ABI.
    InvalidBinding(String),
    /// The requested semantic is outside this exact subset.
    Unsupported(String),
    /// A guarded integer operation failed without exposing candidate output.
    IntegerFault {
        /// Exact failing operation.
        operation: GuardedIntegerOp,
        /// Earliest logical output index.
        index: usize,
        /// Reconstructed invalid shift count; absent for division/remainder.
        count: Option<i64>,
        /// Operation width in bits.
        bits: usize,
    },
    /// Resources belong to different logical devices.
    OwnerMismatch,
    /// A submitted physical generation is no longer visible.
    StaleGeneration {
        /// Generation captured at submission.
        expected: u64,
        /// Generation visible during collection.
        actual: u64,
    },
    /// A resource was used after logical closure.
    Closed(&'static str),
    /// A checked byte or element range is outside its buffer.
    Bounds,
    /// Checked size or indexing arithmetic overflowed.
    Overflow,
}

impl fmt::Display for WebGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryUnavailable { tried } => {
                write!(
                    f,
                    "WebGPU native library unavailable (tried {})",
                    tried.join(", ")
                )
            }
            Self::NativeAbiUnsupported { detail } => {
                write!(f, "WebGPU native C ABI is unsupported: {detail}")
            }
            Self::MissingSymbol(symbol) => write!(f, "missing WebGPU symbol {symbol}"),
            Self::NoAdapters => write!(f, "WebGPU reported no adapters"),
            Self::Driver { operation, detail } => {
                write!(f, "WebGPU {operation} failed: {detail}")
            }
            Self::Build { diagnostic } => {
                write!(f, "WGSL compilation failed: {diagnostic}")
            }
            Self::InvalidArgument(reason) => write!(f, "invalid WebGPU argument: {reason}"),
            Self::InvalidBinding(reason) => write!(f, "invalid WebGPU binding: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported WebGPU kernel: {reason}"),
            Self::IntegerFault {
                operation,
                index,
                count,
                bits,
            } => write!(
                f,
                "WebGPU guarded integer {operation:?} failed at logical index {index} (count {count:?}, {bits} bits)"
            ),
            Self::OwnerMismatch => write!(f, "WebGPU resource owner mismatch"),
            Self::StaleGeneration { expected, actual } => write!(
                f,
                "stale WebGPU buffer generation {expected}; visible generation is {actual}"
            ),
            Self::Closed(resource) => write!(f, "WebGPU {resource} is closed"),
            Self::Bounds => write!(f, "WebGPU buffer range is out of bounds"),
            Self::Overflow => write!(f, "WebGPU size arithmetic overflow"),
        }
    }
}

impl std::error::Error for WebGpuError {}

#[cfg(test)]
mod tests;
