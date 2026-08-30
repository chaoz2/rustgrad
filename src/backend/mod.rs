pub(crate) mod cpu;
mod float8_contract;
mod float8_reduce;
mod jit;

use crate::{Graph, NodeId, Result, TensorData};
use std::collections::HashMap;

pub use cpu::CpuBackend;
pub(crate) use cpu::stable_sort_pair;
pub use jit::{CpuJitBackend, JitBackendError, JitExecution, JitFallback};
pub(crate) use jit::{PreparedScheduleItem, TensorValueStore};

/// A deliberately thin execution boundary. CUDA-specific capabilities will
/// be exposed by extension traits rather than erased from this common core.
pub trait Backend {
    fn execute(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<TensorData>;
}
