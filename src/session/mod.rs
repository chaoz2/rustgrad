//! Public CPU-first tensor workflow built on the inspectable [`crate::Graph`].
//!
//! This module does not add a second IR or a global default graph. A
//! [`CpuSession`] owns exactly one graph and its explicit input bindings; each
//! [`Tensor`] handle carries that session identity and is rejected elsewhere.

mod classification;
mod compiled_training;
mod cpu;
mod inference;
mod native_training_scoreboard;
mod target;
mod train;

pub use classification::{
    BinaryClassificationSummary, ClassificationSummary, summarize_binary_classification,
    summarize_classification,
};
pub use compiled_training::{
    CompiledAdamWCheckpoint, CompiledAdamWCheckpointInfo, CompiledAdamWClipReport,
    CompiledAdamWCommitOnlyRuntime, CompiledAdamWConfig, CompiledAdamWFlush,
    CompiledAdamWFlushResult, CompiledAdamWFlushRuntime, CompiledAdamWGraph,
    CompiledAdamWIgnoreIndexContext, CompiledAdamWObjective, CompiledAdamWPlan,
    CompiledAdamWProgramArtifact, CompiledAdamWProgramArtifactFileError,
    CompiledAdamWProgramArtifactInfo, CompiledAdamWResumeBundle,
    CompiledAdamWResumeBundleFileError, CompiledAdamWRuntime, CompiledAdamWStep,
    CompiledAdamWStepResult, CompiledAdamWWindowLossReport, CompiledAdamWZeroGradResult,
    CompiledCheckpointParameterSnapshot, CompiledCheckpointRestoreRuntime,
    CompiledCheckpointRuntime, CompiledDropoutConfig, CompiledDropoutKey, CompiledEvaluation,
    CompiledEvaluationResult, CompiledEvaluationRuntime, CompiledInputBatch, CompiledInputSpec,
    CompiledModuleAdamWArtifactRestoreError, CompiledModuleAdamWCheckpoint,
    CompiledModuleAdamWCompileError, CompiledModuleAdamWEvaluationError,
    CompiledModuleAdamWFinishError, CompiledModuleAdamWPlan, CompiledModuleAdamWPrepareError,
    CompiledModuleAdamWRestoreError, CompiledModuleAdamWSession,
    CompiledModuleMomentumSgdCompileError, CompiledModuleMomentumSgdPlan,
    CompiledModuleMomentumSgdPrepareError, CompiledModuleMomentumSgdRestoreError,
    CompiledModuleTrainingFinishError, CompiledModuleTrainingSession,
    CompiledMomentumSgdCheckpoint, CompiledMomentumSgdConfig, CompiledMomentumSgdPlan,
    CompiledMomentumSgdStepResult, CompiledMultiStepLr, CompiledScheduledAdamWCommitOnlyRuntime,
    CompiledScheduledAdamWRuntime, CompiledTrainingCommitOnlyRuntime,
    CompiledTrainingRatePolicyCommitOnlyRuntime, CompiledTrainingRatePolicyRuntime,
    CompiledTrainingRatePolicyWindowCommitRuntime, CompiledTrainingRuntime, CompiledTrainingStep,
    CompiledTrainingStepResult, CompiledTrainingWindowCommit, CompiledTrainingWindowCommitRuntime,
    CompiledTrainingWindowReset, CompiledTrainingWindowResetRuntime, CompiledTrainingWindowRuntime,
    CompiledTrainingWindowStep, CpuCompiledAdamW, CpuCompiledMomentumSgd, MetalCompiledAdamW,
    MetalCompiledAdamWCommitResult, MetalCompiledAdamWFlushResult, MetalCompiledAdamWPlan,
    MetalCompiledAdamWStepResult, MetalCompiledEvaluationResult, NativeCpuCompiledAdamW,
    NativeCpuCompiledAdamWFlushResult, NativeCpuCompiledAdamWPreparationReport,
    NativeCpuCompiledAdamWStepResult, NativeCpuCompiledEvaluationResult,
    NativeCpuDispatchSegmentation, NativeCpuPreparationPhases, NativeCpuPreparationWork,
    NativeCpuProgramPreparationReport, NativeCpuRenderCapsuleDiagnostic,
    NativeCpuRenderCapsuleLoadStatus, NativeCpuRenderCapsuleProgramRole,
    NativeCpuRenderCapsuleStoreStatus, NativeCpuReplayTraffic, NativeCpuRunReport,
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
pub use native_training_scoreboard::{
    CompiledAdamWInspection, CompiledTrainingCompileObservation,
    CompiledTrainingCompilePhaseObservation, NATIVE_TRAINING_REPORT_FORMAT_VERSION,
    NativeTrainingCompilePhase, NativeTrainingCompilePhaseReport, NativeTrainingFirstStepReport,
    NativeTrainingPreparationTiming, NativeTrainingProgramReport, NativeTrainingReplayTiming,
    NativeTrainingReport, NativeTrainingScoreboard, NativeTrainingStepPhase,
    NativeTrainingStepPhaseReport, NativeTrainingWarmStepReport,
};
pub use target::{
    ConfiguredCpuSessionTarget, CpuNonFinitePolicy, CpuSessionTarget, MetalSessionTarget,
    NativeCpuSessionTarget, SessionTarget,
};
pub use train::{
    CpuBinaryModuleTrainer, CpuModeModuleTrainer, CpuModuleTrainer, ModuleBinaryCrossEntropy,
    ModuleCrossEntropy, ModuleStepResult,
};
