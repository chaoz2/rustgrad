//! Public CPU-first tensor workflow built on the inspectable [`crate::Graph`].
//!
//! This module does not add a second IR or a global default graph. A
//! [`CpuSession`] owns exactly one graph and its explicit input bindings; each
//! [`Tensor`] handle carries that session identity and is rejected elsewhere.

mod cpu;
mod train;

pub use cpu::{CpuSession, SessionDevice, Tensor};
pub use train::{CpuModuleTrainer, ModuleCrossEntropy, ModuleStepResult};
