//! Public CPU-first tensor workflow built on the inspectable [`crate::Graph`].
//!
//! This module does not add a second IR or a global default graph. A
//! [`CpuSession`] owns exactly one graph and its explicit input bindings; each
//! [`Tensor`] handle carries that session identity and is rejected elsewhere.

mod classification;
mod cpu;
mod inference;
mod train;

pub use classification::{
    BinaryClassificationSummary, ClassificationSummary, summarize_binary_classification,
    summarize_classification,
};
pub use cpu::{
    CpuSession, DynamicTensor, MetalSessionResult, MetalSessionTrace, SessionDevice, Tensor,
};
pub use inference::{
    ModuleInferenceResult, NativeModuleExecutionReport, NativeModuleInferenceResult,
    NativeModuleInferenceTrace, ReportedNativeModuleInferenceResult, infer_module_cpu,
    infer_module_native_cpu, infer_module_native_cpu_with_report,
};
pub use train::{
    CpuBinaryModuleTrainer, CpuModeModuleTrainer, CpuModuleTrainer, ModuleBinaryCrossEntropy,
    ModuleCrossEntropy, ModuleStepResult,
};
