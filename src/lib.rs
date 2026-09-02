//! RustGrad is an inspectable tensor compiler inspired by tinygrad and built
//! around Rust's explicit ownership and backend traits.

pub mod autograd;
pub mod backend;
pub mod collective;
mod collective_inspection;
pub mod conv2d_plan;
pub mod cpu_jit;
pub mod cpu_stable_sort;
pub mod cuda;
mod cuda_profile;
pub mod datasets;
pub mod effects;
pub mod einsum;
pub mod engine;
pub mod error;
pub mod fuzz;
pub mod gguf;
pub mod gradcheck;
mod host_buffer;
mod index;
pub mod interop;
pub mod ir;
pub mod kernel;
pub mod linearize;
pub mod linked_exp;
pub mod loss;
pub mod matmul;
pub mod random;
mod source_einsum;
/// Compatibility facade for the original normalized matmul-plan module path.
pub mod matmul_plan {
    pub use crate::matmul::{MatmulKernelPlan, MatmulPlanError};
}
pub mod memory_plan;
pub mod memory_space;
pub mod models;
pub mod movement_plan;
pub mod nn;
pub mod onnx;
pub mod optim;
mod portable_sort;
mod portable_threefry;
pub mod ptx;
mod rangeify;
pub mod runtime;
pub mod safetensors;
pub mod schedule;
pub mod session;
pub mod sharded_cuda_execute;
pub mod sharded_cuda_plan;
pub mod sharded_graph;
pub mod sharding;
pub mod symbolic;
pub mod symbolic_shape;
pub mod tensor;
#[cfg(test)]
mod tensor_guard_tests;
pub mod tokenizer;
pub mod torch;
pub mod trace;
pub mod training_checkpoint;
pub mod uop;
pub mod vector_ir;
pub mod viz;

pub use backend::{Backend, CpuBackend, CpuJitBackend, JitFallback};
pub use collective::{
    CollectiveAction, CollectiveExecutor, CollectiveKind, CollectivePlan, CollectivePlanner,
    CollectiveRequest, CudaCollectiveGroup, CudaCollectiveTrace, DeviceGroup,
    InMemoryCollectiveExecutor, LogicalRange, Reduction as CollectiveReduction, StreamLane,
};
pub use collective_inspection::{CollectivePlanInspection, CollectivePlanInspectionError};
pub use conv2d_plan::{StaticConv2dPlan, StaticConv2dPlanError};
pub use cpu_jit::{
    CpuJit, JitBuffer, JitError, JitKernel, KernelAbi, KernelPointerAbi, QuantizedBufferAbi,
    RenderedC, VectorPlan,
};
pub use cpu_stable_sort::{
    BoundCpuStableSortPlan, CpuStableSortDescriptor, CpuStableSortExecutionError,
    CpuStableSortPlan, CpuStableSortPlanError,
};
pub use cuda::{
    BufferLease, BufferView, Capability, Capture, Context, ContextGuard, CudaAllocator, CudaError,
    CudaGraph, CudaModule, Device, DeviceBuffer, DeviceId, Driver, Event, Function, GraphExec,
    LaunchConfig, LinkInput, LinkInputKind, LinkInputResourceDescriptor, LinkedModuleIdentity,
    ModuleLoadMetadata, ModuleLoadOptions, NvvmExportContract, NvvmProducerContract, NvvmPrototype,
    PeerAccess, PeerTransfer, PinnedHostBuffer, PrimaryBlock, PrimaryBufferLease, PrimaryContext,
    PrimaryContextGuard, PrimaryCudaAllocator, PrimaryEventFence, PrimaryLinkedKernel,
    PrimaryLinkedKernelCache, PrimaryLinkedModule, PrimaryLinkedModuleCache, PrimaryOwner,
    PrimaryPoolStats, Stream, Transfer, linked_module_identity,
};
pub use datasets::{
    BatchIter, Cifar10, Cifar10FileError, Cifar10ReadLimits, ClassificationBatch,
    ClassificationFeatureLayout, MnistIdx, MnistIdxFileError, MnistIdxReadLimits,
    load_cifar10_files, load_cifar10_files_with_limits, load_mnist_idx_files,
    load_mnist_idx_files_with_limits, materialize_classification_batch, parse_cifar10,
    parse_mnist_idx,
};
pub use effects::{
    BufferState, EffectBatch, EffectBatchEntry, EffectBatchSource, EffectBatchStep, EffectCommit,
    EffectError, EffectGraph, EffectMutationPermit, EffectPayload, EffectPlan, EffectRuntime,
    EffectSchedule, EffectScheduleNode, EffectStep, MutationSafety, PersistentRuntimeStats,
    PersistentSlotIdentity, PersistentSnapshot, RuntimeError, StateHandle,
};
pub use einsum::{EinsumLabel, EinsumPlan};
pub use engine::capture::{CapturedSchedule, ReplayError, ReplayInput};
pub use engine::mixed_batch::{
    CapturedMixedBatch, CapturedMixedStateBundle, InstantiatedMixedBatch, MixedBatchArtifactError,
    MixedStateBundleError, NativeMixedBatchResult, NativeMixedBatchTrace, PortableMixedState,
};
pub use engine::mixed_capture::{
    CapturedMixedSchedule, MixedReplayCursor, MixedReplayResult, NativeMixedReplayTrace,
};
pub use engine::mixed_rebinding::MixedStateRebinding;
pub use engine::{
    CapturedBackendPolicy, CapturedBatch, CapturedBatchResult, CapturedInvocation,
    CapturedItemTrace, CapturedReplayExecutor, CapturedReplayOptions, CapturedReplayResult,
    CapturedReplayTrace, CapturedSpecialization, CapturedSpecializationTrace, ItemBackend,
    ItemTrace, MemoryReuse, RealizationError, RealizationOptions, RealizationPolicy,
    RealizationTrace, Realized, SymbolicCaptureSpec, SymbolicGuard, SymbolicParameter, realize,
    realize_effects_persistent, realize_graph, realize_graph_with_options, realize_mixed_effects,
    realize_with_options,
};
pub use error::{Error, Result, ShardedCudaCompositionErrorKind, ShardedCudaCompositionField};
pub use fuzz::{
    FuzzArtifactError, FuzzBinaryOp, FuzzCampaign, FuzzCase, FuzzCompareOp, FuzzComparison,
    FuzzComparisonPolicy, FuzzConfig, FuzzFailureArtifact, FuzzLogicalOp, FuzzOutcome, FuzzPath,
    FuzzReduction, FuzzScatterOp, FuzzSlice, FuzzTensor, FuzzUnaryOp, generate_case, minimize_case,
    regression_cases, replay_failure, run_campaign, run_case,
};
pub use gguf::{
    GgmlLayout, GgmlType, QuantizedBufferDesc, QuantizedError, QuantizedRowGatherError,
    QuantizedRowGatherPlan, QuantizedTensorData,
};
pub use gradcheck::{
    GradcheckConfig, GradcheckError, GradcheckMismatch, GradcheckReport, gradcheck_cpu,
};
pub use host_buffer::HostBufferError;
pub use ir::pool::MaxPool2dOutput;
pub use ir::{
    AttentionOptions, BinaryOp, CompareOp, Conv2dOptions, ConvTranspose1dOptions,
    ConvTranspose2dOptions, ConvolutionSpec, DynamicAllocation, DynamicAllocationError,
    DynamicAllocationPlan, DynamicAllocationTarget, DynamicBinding, DynamicCountStage,
    DynamicInput, DynamicNodeId, DynamicOutputShape, DynamicVjpPlan, ExpandExtent, Graph,
    LogicalOp, NodeId, Op, PadMode, Pool2dOptions, PoolOptions, PrefixScanKind, PrefixScanOutput,
    RandomKind, RandomStream, ReduceKind, ReductionDType, ReshapeExtent, RollDims, RollShifts,
    ShrinkRange, Slice, SortOutput, SpatialWindow, SpatialWindowError, SplitSections, SplitSizes,
    StaticIndexUpdateWrt, TransposedConvolutionSpec, UnaryOp, UnflattenExtent, VarianceCorrection,
};
pub use ir::{ScatterMode, ScatterReduceKind, ScatterSource};
pub use kernel::{
    BufferRole, IterationPlan, KernelBindings, KernelBufferDesc, KernelShape, ReductionPlan,
    execute_elementwise, execute_with_memory_plan, lower_graph_elementwise, lower_graph_matmul,
    lower_graph_movement, lower_graph_prefix_scan, lower_graph_reduction,
    lower_graph_static_conv2d,
};
pub use linearize::{
    AddressRef, IndexRef, LaneInstruction, LaneInstructionView, LaneProgramInstruction,
    LaneSourceRecord, LinearAccess, LinearBuffer, LinearKernel, LinearProgram, LinearizeError,
    LiveInterval, RegisterAssignment, RegisterClass, TypedValue, allocate,
};
pub use linked_exp::*;
pub use loss::{
    LossOptions, Reduction, SparseCategoricalCrossEntropyOptions, binary_cross_entropy,
    binary_cross_entropy_with_logits, cross_entropy, nll_loss, sparse_categorical_cross_entropy,
    sparse_categorical_cross_entropy_tinygrad,
};
pub use matmul::{
    MatmulBarrierKind, MatmulBarrierPhase, MatmulKernelPlan, MatmulPlanError,
    MatmulResourceEstimate, MatmulTargetCaps, MmaFragmentLayout, MmaInstruction,
    QuantizedMatmulError, QuantizedMatmulOrientation, QuantizedMatmulPlan, SharedTileLayout,
    TensorCoreMatmulError, TensorCoreMatmulPayload, TensorCoreMatmulPlan, TensorCoreOutputPolicy,
    TensorCoreTailPolicy, TiledMatmulError, TiledMatmulPayload, TiledMatmulPlan, TiledMatmulTails,
};
pub use memory_plan::{
    AliasLifetime, AliasLivenessPlan, AllocationRequest, MemoryAddressSpace, MemoryPlan,
    MemoryPlanError, TemporaryAllocation,
};
pub use memory_space::{
    BarrierPoint, BarrierScope, GlobalAccess, MemorySpace, MemorySpaceError, MemorySpacePlan,
    PromotionDecision, RegisterBinding, SpaceAllocation, plan_tensor_core_matmul_promotion,
    plan_tiled_matmul_promotion,
};
pub use movement_plan::{
    MovementExecutionError, MovementKernelKind, MovementKernelPlan, MovementOperand,
    MovementPlanError,
};
pub use nn::{
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool1d, BatchNorm, BatchNorm2d, BatchNorm3d,
    BatchNormOutput, CastPolicy, ConvTranspose1d, ConvTranspose2d, Flatten, GroupNorm,
    InstanceNorm, LiveStateDict, LoadReport, MaxPool1d, Mode, ModeForwardOutput, ModeModuleForward,
    ModeSequential, Module, ModuleForward, Parameter, ParameterId, ParameterSnapshot,
    PendingBatchNormStats, PendingModeEffects, ReLU, RealizedBatchNormStats,
    StateDict as ModuleStateDict, StrictStateLoadLimits, get_parameters, get_state_dict,
};
pub use onnx::{
    NativeOnnxInferenceResult, NativeOnnxInferenceTrace, OnnxModel, import_onnx,
    run_onnx_files_native,
};
pub use optim::{AdamConfig, Gradient, Optimizer, OptimizerKind, ParameterGroup, SgdConfig};
pub use ptx::{
    ConcurrentPtxCache, LINKED_F32_EXP_RENDERER_CONTRACT_VERSION, LinkedF32ExpRequest,
    PrimaryLinkedRenderedKernel, PrimaryLinkedRenderedKernelCache, PrimaryPtxKernel, PtxBinding,
    PtxCache, PtxError, PtxKernel, PtxLaunchGeometry, PtxRenderer, RenderedPtx,
};
pub use runtime::cuda_graph::{CudaGraphPrefixPlan, PreparedCudaGraphPrefix};
pub use runtime::mapped::{MappedBackingId, MappedTensor, MappedTensorError, MappedTensorPolicy};
pub use runtime::mapped_mut::{MutableMappedFile, MutableMappedFileError};
pub use safetensors::{
    Metadata, OwnedSafetensorsMetadata, SafetensorsFileError, SafetensorsMetadata,
    SafetensorsReadLimits, StateDict, inspect_safetensors_metadata,
    inspect_safetensors_metadata_file, load_safetensors, load_safetensors_file,
    load_safetensors_file_with_limits, load_safetensors_state_only,
    load_safetensors_state_only_file, save_safetensors, save_safetensors_file,
    save_safetensors_file_with_json_metadata, save_safetensors_with_json_metadata,
};
pub use schedule::{
    BufferDesc, ExecutionPlanItemSummary, ExecutionPlanSummary, ExecutionPlanSummaryError,
    QuantizedScheduleInputBinding, RequestedPassthrough, Schedule, ScheduleBoundary, ScheduleError,
    ScheduleInputBinding, ScheduleItem, ScheduleStateBinding, ScheduleValueBinding,
    ScheduledOutputs, bind_schedule_states, combine_mixed_schedules, plan_temporary_reuse,
    schedule, schedule_effects, schedule_many, schedule_with_external_materializations,
};
pub use session::{
    BinaryClassificationSummary, ClassificationSummary, CompiledMomentumSgdConfig,
    CompiledMomentumSgdStepResult, CpuBinaryModuleTrainer, CpuCompiledMomentumSgd,
    CpuGradientStore, CpuModeModuleTrainer, CpuModuleTrainer, CpuSession, DynamicTensor,
    MetalSessionResult, MetalSessionTrace, ModuleBinaryCrossEntropy, ModuleCrossEntropy,
    ModuleInferenceResult, ModuleStepResult, NativeModuleExecutionReport,
    NativeModuleInferenceResult, NativeModuleInferenceTrace, ReportedNativeModuleInferenceResult,
    SessionDevice, Tensor, TrainingParameterInit, infer_module_cpu, infer_module_native_cpu,
    infer_module_native_cpu_with_report, summarize_binary_classification, summarize_classification,
};
pub use sharded_cuda_execute::{
    BufferSubstitution, ShardedCudaExecutionEnvironment, ShardedCudaExecutionResult,
    ShardedCudaExecutionTrace, ShardedCudaPlanComposition,
};
pub use sharded_cuda_plan::{
    CollectiveCandidateDescriptor, CollectiveCommitRecord, CollectiveConsumerDescriptor,
    CollectiveDownstreamConsumerAbi, CollectiveDownstreamOutputArtifact,
    CollectiveDownstreamOutputCommitRecord, CollectiveDownstreamOutputDescriptor,
    CollectiveGraphResultBinding, CollectiveGraphUnaryOutputArtifact,
    CollectiveGraphUnaryOutputComponents, CollectiveLifecycleMaterialization,
    CollectiveLifecycleMaterializationArtifact, CollectiveMaterializationArtifact,
    CollectiveMaterializationLifecycle, CollectiveResultMaterialization,
    CollectiveTransactionArtifact, CudaPlanBinding, CudaPlanDiagnostic, CudaPlanStage,
    CudaTransferRoute, ExecutableBuffer, ExecutableBufferRole,
    ExecutableCollectiveDownstreamOutput, ExecutableCollectiveGraphUnaryOutput,
    ExecutableCollectiveLifecycleMaterialization, ExecutableCollectiveMaterialization,
    ExecutableCollectiveTransaction, ExecutableShardedCudaPlan, ShardedCudaPlan,
    ShardedCudaPlanArtifact, ShardedCudaPlanner,
};
pub use sharded_graph::{
    CollectiveBoundaryLifecycle, CollectiveBoundaryProvenance, LocalInputProvenance,
    LocalOperandProvenance, ShardGraphTrace, ShardGraphTraceStep, ShardedGraphTensor,
};
pub use sharding::{
    DeviceShard, LayoutTransform, MovementDecision, ShardDistribution, ShardExecutionPlan,
    ShardLayout, ShardRange, ShardedTensorData,
};
pub use symbolic::{
    Bounds as SymbolicBounds, Simplified as SimplifiedSymbolicExpr, SymbolicError, SymbolicExpr,
    SymbolicVar,
};
pub use symbolic_shape::{SymbolicDim, SymbolicShape};
pub use tensor::{
    DType, DTypeCategory, Float8Format, Float8Storage, LiteralScalar, Scalar, Shape, Storage,
    TensorData, TensorDataReader, TensorList,
};
pub use torch::{
    TorchStateFileError, TorchStateLimit, TorchStateReadLimits, extract_tar_files,
    load_legacy_torch_state_dict, load_torch_state_dict, load_torch_state_dict_strict,
    load_torch_state_dict_strict_with_limits, load_torch_state_dict_with_limits,
    load_torch_state_file, load_torch_state_file_strict, load_torch_state_file_strict_with_limits,
    load_torch_state_file_with_limits,
};
pub use trace::{CompileTrace, TraceStep};
pub use training_checkpoint::{PortableTrainingCheckpoint, TrainingCheckpoint};
pub use uop::{
    AddressSpace, AddressValue, AffineView, Binary as UBinary, IndexValue, LiteralValue,
    MatmulValue, MovementValue, Operation, PrefixScanValue, ReductionValue, SortValue,
    TensorGuardValue, ThreefryValue, UOp, UOpError, UPat, UType, VariableValue, ViewMap,
};
pub use vector_ir::{VectorIrError, VectorOperand, VectorProgram};
pub use viz::{
    VizEdge, VizError, VizGraph, VizNode, captured_replay_trace_viz, captured_schedule_viz,
    captured_specialization_trace_viz, compile_trace_viz, cuda_collective_trace_viz, graph_viz,
    linear_viz, memory_space_viz, native_mixed_batch_trace_viz, native_mixed_replay_trace_viz,
    realization_trace_viz, schedule_viz, sharded_cuda_execution_trace_viz, uop_viz, vector_viz,
};

#[cfg(test)]
mod attention_tests;
#[cfg(test)]
mod conv2d_tests;
#[cfg(test)]
mod creation_random_tests;
#[cfg(test)]
mod dot_tests;
#[cfg(test)]
mod einsum_tests;
#[cfg(test)]
mod multinomial_tests;
mod prefix_scan_native;
#[cfg(test)]
mod prefix_scan_tests;
#[cfg(test)]
mod rearrange_tests;
mod reduction_native;
#[cfg(test)]
mod reduction_tests;
#[cfg(test)]
mod schedule_tests;
#[cfg(test)]
mod sort_tests;
#[cfg(test)]
mod special_functions_tests;
#[cfg(test)]
mod svd_tests;
#[cfg(test)]
mod symbolic_tests;
#[cfg(test)]
mod topk_tests;
#[cfg(test)]
mod uop_tests;
