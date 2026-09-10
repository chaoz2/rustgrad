//! Public CPU-first tensor workflow built on the inspectable [`crate::Graph`].
//!
//! This module does not add a second IR or a global default graph. A
//! [`CpuSession`] owns exactly one graph and its explicit input bindings; each
//! [`Tensor`] handle carries that session identity and is rejected elsewhere.

mod classification;
mod compiled_training;
mod cpu;
mod inference;
mod target;
mod train;

pub use classification::{
    BinaryClassificationSummary, ClassificationSummary, summarize_binary_classification,
    summarize_classification,
};
pub use compiled_training::{
    CompiledAdamWCheckpoint, CompiledAdamWCheckpointInfo, CompiledAdamWConfig, CompiledAdamWFlush,
    CompiledAdamWFlushResult, CompiledAdamWFlushRuntime, CompiledAdamWPlan, CompiledAdamWRuntime,
    CompiledAdamWStep, CompiledAdamWStepResult, CompiledAdamWZeroGradResult,
    CompiledCheckpointRuntime, CompiledDropoutConfig, CompiledDropoutKey, CompiledEvaluation,
    CompiledEvaluationResult, CompiledEvaluationRuntime, CompiledInputBatch, CompiledInputSpec,
    CompiledModuleAdamWCompileError, CompiledModuleAdamWEvaluationError,
    CompiledModuleAdamWFinishError, CompiledModuleAdamWPlan, CompiledModuleAdamWPrepareError,
    CompiledModuleAdamWRestoreError, CompiledModuleAdamWSession, CompiledMomentumSgdConfig,
    CompiledMomentumSgdStepResult, CompiledMultiStepLr, CompiledScheduledAdamWRuntime,
    CompiledTrainingRuntime, CompiledTrainingStep, CompiledTrainingStepResult, CpuCompiledAdamW,
    CpuCompiledMomentumSgd, MetalCompiledAdamW, MetalCompiledAdamWCommitResult,
    MetalCompiledAdamWFlushResult, MetalCompiledAdamWPlan, MetalCompiledAdamWStepResult,
    MetalCompiledEvaluationResult, NativeCpuCompiledAdamW, NativeCpuCompiledAdamWFlushResult,
    NativeCpuCompiledAdamWPreparationReport, NativeCpuCompiledAdamWStepResult,
    NativeCpuCompiledEvaluationResult, NativeCpuProgramPreparationReport, NativeCpuRunReport,
    TrainingParameterInit,
};
pub use cpu::{
    CpuGradientStore, CpuSession, DynamicTensor, MaskedSelectOutput, MetalSessionResult,
    MetalSessionTrace, SessionDevice, Tensor,
};
pub use inference::{
    CapturedAppendStateInference, CapturedInference, CapturedInferenceError,
    CapturedStatefulInference, InferenceAppendStateLink, InferenceStateLink, ModuleInferenceResult,
    NativeModuleExecutionReport, NativeModuleInferenceResult, NativeModuleInferenceTrace,
    ReportedNativeModuleInferenceResult, infer_module_cpu, infer_module_native_cpu,
    infer_module_native_cpu_with_report,
};
pub(crate) use inference::{
    CapturedHostGather, CapturedHostIndexedMovement, CapturedHostIndexedMovementKind,
};
pub use target::{
    ConfiguredCpuSessionTarget, CpuNonFinitePolicy, CpuSessionTarget, MetalSessionTarget,
    NativeCpuSessionTarget, SessionTarget,
};
pub use train::{
    CpuBinaryModuleTrainer, CpuModeModuleTrainer, CpuModuleTrainer, ModuleBinaryCrossEntropy,
    ModuleCrossEntropy, ModuleStepResult,
};
