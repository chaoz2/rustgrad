//! Graph-free CPU replay for static training programs with recurrent state.

mod adamw_checkpoint;
mod adamw_contract;
mod adamw_plan;
mod adamw_plan_restore;
mod capture;
mod cpu_adamw_capabilities;
mod cpu_adamw_runtime;
mod cpu_training_program;
mod cpu_training_step;
mod delegation;
mod dropout;
mod exchange;
mod module_adamw_checkpoint;
mod module_plan;
mod module_session;
mod module_state;
mod momentum_module_plan;
mod momentum_plan;
mod native_cpu_evidence;
mod native_cpu_preparation;
mod native_cpu_programs;
mod objective;
mod observation;
mod optimizer_lowering;
mod policy;
mod program_artifact;
mod recurrent_phase;
mod resume_bundle;
mod runtime;
mod state_schema;
mod training_plan;
mod validation;
mod window_progress;

#[cfg(test)]
use self::adamw_checkpoint::{
    ADAMW_CHECKPOINT_FORMAT_V1, ADAMW_CHECKPOINT_FORMAT_V2, ADAMW_CHECKPOINT_FORMAT_V3,
    ADAMW_CHECKPOINT_FORMAT_V4, ADAMW_CHECKPOINT_FORMAT_V8, ADAMW_CHECKPOINT_FORMAT_V9,
};
use self::adamw_checkpoint::{
    AdamWCheckpointProgress, AdamWCheckpointTensors, DecodedAdamWCheckpoint,
    decode_adamw_checkpoint, encode_adamw_checkpoint,
};
pub use self::adamw_checkpoint::{CompiledAdamWCheckpoint, CompiledAdamWCheckpointInfo};
use self::adamw_contract::{CompiledAdamWContract, MetalAdamWContract};
pub use self::adamw_plan::CompiledAdamWPlan;
#[cfg(test)]
use self::adamw_plan_restore::adamw_checkpoint_restore_counts;
use self::capture::{CompiledEvaluationCapture, CompiledRecurrentCapture};
#[cfg(test)]
use self::capture::{
    canonical_recurrent_capture_counts, canonical_recurrent_capture_delta,
    with_canonical_recurrent_reference,
};
pub use self::cpu_adamw_runtime::{CpuCompiledAdamW, NativeCpuCompiledAdamW};
use self::cpu_adamw_runtime::{
    NativeCpuEvaluationPreparation, PreparedNativeCpuEvaluation, PreparedNativeCpuProgram,
    PreparedNativeEvaluationParameterInput, adamw_step_result,
};
use self::cpu_training_program::CpuCompiledTrainingProgram;
use self::cpu_training_step::{CompiledStepOutputSelection, CompiledStepReplayRequest};
pub use self::dropout::{CompiledDropoutConfig, CompiledDropoutKey};
use self::dropout::{CompiledDropoutState, CompiledDropoutStream, expected_dropout_counter};
pub use self::exchange::*;
use self::exchange::{CompiledAdamWWindowLossValue, CompiledInputPolicy};
pub use self::module_adamw_checkpoint::CompiledModuleAdamWCheckpoint;
use self::module_adamw_checkpoint::encode_module_adamw_checkpoint;
pub use self::module_state::TrainingParameterInit;
use self::module_state::{CompiledModuleSeal, ModuleParameterPlan};
pub use self::momentum_module_plan::{
    CompiledModuleMomentumSgdPlan, CompiledModuleMomentumSgdPrepareError,
    CompiledModuleMomentumSgdRestoreError,
};
pub use self::momentum_plan::CompiledMomentumSgdPlan;
use self::native_cpu_evidence::native_preparation_wall_time;
pub use self::native_cpu_evidence::*;
use self::native_cpu_programs::*;
pub use self::objective::{
    CompiledAdamWGraph, CompiledAdamWIgnoreIndexContext, CompiledAdamWObjective,
};
use self::objective::{
    lower_compiled_adamw_objective, lower_compiled_adamw_objective_for_ignore_index_policy,
    lower_compiled_adamw_objective_for_policy,
    lower_compiled_adamw_objective_with_ignore_index_nodes, lower_ignore_index_nodes,
    lower_ignore_index_nodes_for_policy, lower_token_batch_count, lower_token_mean_loss,
    reject_token_weighted_scalar_loss, safe_token_count_divisor, token_mean_loss_descriptor,
    validate_retained_token_count, validate_token_weight, validate_token_weight_policy,
};
use self::observation::{
    CompiledAdamWAuxiliaryOutputSchema, CompiledAdamWAuxiliaryReports,
    CompiledAdamWWindowLossNodes, CompiledTrainingLossOutput, CompiledTrainingObservationSchema,
    CompiledTrainingPhaseOutputSchema, adamw_observation_nodes, adamw_observation_schema,
    validate_adamw_observation_schema, validate_observation_value_descriptor,
    validate_staged_observations,
};
use self::optimizer_lowering::{
    AdamWProgram, CompiledOptimizerLowering, CompiledOptimizerLoweringContext,
    CompiledOptimizerProgram, MomentumProgram, RecurrentStoreGroupSpec,
    adamw_recurrent_store_group_specs, clip_gradients_by_global_norm, lower_adamw_learning_rate,
    lower_adamw_update_candidates, scalar_f32,
};
pub use self::policy::{CompiledAdamWConfig, CompiledMomentumSgdConfig, CompiledMultiStepLr};
use self::policy::{
    CompiledLearningRatePolicy, CompiledTokenWeightPolicy, CompiledTrainingWindowTopology,
};
pub use self::program_artifact::{
    CompiledAdamWProgramArtifact, CompiledAdamWProgramArtifactFileError,
    CompiledAdamWProgramArtifactInfo, CompiledModuleAdamWArtifactRestoreError,
};
use self::recurrent_phase::*;
pub use self::resume_bundle::{CompiledAdamWResumeBundle, CompiledAdamWResumeBundleFileError};
pub use self::runtime::*;
#[cfg(test)]
use self::state_schema::INTERNAL_PREFIX;
use self::state_schema::{AdamWGlobalState, AdamWParameterState, RecurrentStateKey, StateSpec};
use self::training_plan::*;
use self::validation::{
    canonical_parameters, checked_bytes, checked_descriptor, checked_recurrent_state_extent,
    collect_state_bindings, effect_states, has_non_finite_f32, state_for,
    validate_evaluation_inputs, validate_external_binding_ownership, validate_finite_tensors,
    validate_learning_rate, validate_learning_rate_for_policy, validate_loss, validate_outputs,
    validate_staged_transition, validate_step_inputs, validate_training_inputs, validate_user_name,
    validate_weight_decay_exclusion_names, value_binding,
};
#[cfg(test)]
use self::window_progress::validate_training_window_progress;
use self::window_progress::{
    CompiledTrainingWindowProgress, adamw_window_progress, validate_adamw_progress,
};
use super::inference::{PortableCapturedInferenceRecipe, PortableInferenceHostPolicy};
use super::native_training_scoreboard::{
    CompiledAdamWInspection, CompiledTrainingCompileObservation,
    CompiledTrainingCompilePhaseObservation,
};
use super::target::{
    ConfiguredCpuSessionTarget, CpuNonFinitePolicy, CpuSessionTarget, MetalSessionTarget,
    NativeCpuSessionTarget, SessionTarget,
};
use crate::effects::runtime::RecurrentTransactionError;
use crate::engine::mixed_capture::{
    NativeReplayContext, PreparedRecurrentCursorProjection, PreparedRecurrentNativeReplay,
    ProjectedRecurrentCursor, RecurrentCursorProjectionError, RecurrentNativePreparation,
};
use crate::engine::{NativeReplayTraffic, PlannedNativeItems};
use crate::nn::TrainingDropoutProvider;
use crate::runtime::metal::{
    MetalDevice, MetalDeviceRun, MetalDeviceRunReport, MetalDeviceSession,
    MetalDeviceSessionSummary, MetalError, MetalFixedStateReadPlan, MetalFixedStateReadSession,
    MetalFixedStateTransitionPlan, MetalFixedStateTransitionSession, MetalRenderer,
    MetalScoreboardContext, MetalScoreboardError, MetalScoreboardObserver, MetalSessionScoreboard,
    MetalSessionScoreboardReport, MetalStatefulInferencePlan, RenderedMetal,
};
use crate::{
    BufferState, CapturedMixedSchedule, CapturedReplayExecutor, CapturedSchedule,
    CapturedStatefulInference, CompareOp, DType, EffectGraph, EffectRuntime, Error,
    ExecutionPlanSummary, Graph, InferenceStateLink, LoadReport, MixedReplayCursor, Module,
    NativeMixedReplayTrace, NodeId, ReplayError, Result, Scalar, Shape, TensorData,
    bind_schedule_states, combine_mixed_schedules, schedule_effects, schedule_many,
};
#[cfg(test)]
use crate::{load_safetensors, save_safetensors};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

const LEARNING_RATE_INPUT: &str = "__rustgrad_compiled_training_learning_rate";
const STATE_BUFFER_BASE: u64 = 1_u64 << 62;
const MAX_EXACT_F32_INTEGER_COUNT: u64 = 1_u64 << 24;

/// One compiled momentum-SGD training program.
pub struct CpuCompiledMomentumSgd {
    inner: CpuCompiledTrainingProgram,
}

/// Resource-free AdamW plan paired with the exact module value used to build it.
///
/// The module is not exposed while the plan or its prepared session exists.
/// This prevents ordinary callers from accidentally treating its stale host
/// parameters as the active training frontier. Successful
/// [`CompiledModuleAdamWSession::finish`] publishes before returning it; the
/// explicit abort path returns the sealed host state without publication.
pub struct CompiledModuleAdamWPlan<M> {
    module: M,
    plan: CompiledAdamWPlan,
    seal: CompiledModuleSeal,
    required_evaluation_capture_identity: Option<u64>,
}

/// Recoverable momentum-SGD owned-module compilation failure.
///
/// Compilation and checkpoint validation never publish into the supplied
/// module. The unchanged module is returned to callers on every failure.
pub struct CompiledModuleMomentumSgdCompileError<M> {
    module: M,
    source: Error,
}

impl<M> CompiledModuleMomentumSgdCompileError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_module(self) -> M {
        self.module
    }

    pub fn into_parts(self) -> (M, Error) {
        (self.module, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleMomentumSgdCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleMomentumSgdCompileError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleMomentumSgdCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled momentum-SGD compilation failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleMomentumSgdCompileError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Prepared compiled training session that owns its source module for the
/// complete replay lifecycle.
///
/// Optimizer policy remains on `R`. This owner only seals the source module,
/// forwards the runtime's supported capabilities, and publishes one exact
/// detached parameter frontier when the session finishes.
pub struct CompiledModuleTrainingSession<M, R> {
    module: M,
    runtime: R,
    seal: CompiledModuleSeal,
    evaluation_capture_identity: Option<u64>,
}

/// Source-compatible compiled AdamW owner around the optimizer-neutral session.
pub struct CompiledModuleAdamWSession<M, R> {
    training: CompiledModuleTrainingSession<M, R>,
}

/// Recoverable compilation failure retaining the exact uncompiled module.
pub struct CompiledModuleAdamWCompileError<M> {
    module: M,
    source: Error,
}

/// Recoverable checkpoint-restore failure retaining the complete owned plan.
pub struct CompiledModuleAdamWRestoreError<M> {
    plan: Box<CompiledModuleAdamWPlan<M>>,
    source: Error,
}

impl<M> CompiledModuleAdamWRestoreError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn plan(&self) -> &CompiledModuleAdamWPlan<M> {
        &self.plan
    }

    pub fn into_plan(self) -> CompiledModuleAdamWPlan<M> {
        *self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWPlan<M>, Error) {
        (*self.plan, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleAdamWRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWRestoreError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleAdamWRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW checkpoint restore failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleAdamWRestoreError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<M> CompiledModuleAdamWCompileError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_module(self) -> M {
        self.module
    }

    pub fn into_parts(self) -> (M, Error) {
        (self.module, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleAdamWCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWCompileError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleAdamWCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW compilation failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleAdamWCompileError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Recoverable target-preparation failure retaining the unconsumed owned plan.
pub struct CompiledModuleAdamWPrepareError<M, E> {
    plan: CompiledModuleAdamWPlan<M>,
    source: E,
}

/// Recoverable evaluation-capture failure retaining the complete owned plan.
pub struct CompiledModuleAdamWEvaluationError<M> {
    plan: Box<CompiledModuleAdamWPlan<M>>,
    source: Error,
}

impl<M> CompiledModuleAdamWEvaluationError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_plan(self) -> CompiledModuleAdamWPlan<M> {
        *self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWPlan<M>, Error) {
        (*self.plan, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleAdamWEvaluationError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWEvaluationError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleAdamWEvaluationError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW evaluation capture failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleAdamWEvaluationError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<M, E> CompiledModuleAdamWPrepareError<M, E> {
    pub fn source_error(&self) -> &E {
        &self.source
    }

    pub fn into_plan(self) -> CompiledModuleAdamWPlan<M> {
        self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWPlan<M>, E) {
        (self.plan, self.source)
    }
}

impl<M, E: fmt::Debug> fmt::Debug for CompiledModuleAdamWPrepareError<M, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWPrepareError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M, E: fmt::Display> fmt::Display for CompiledModuleAdamWPrepareError<M, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW preparation failed: {}",
            self.source
        )
    }
}

impl<M, E: std::error::Error + 'static> std::error::Error
    for CompiledModuleAdamWPrepareError<M, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Finalization failure retaining the intact owned module/session pair.
///
/// This covers both parameter-only [`CompiledModuleTrainingSession::finish`] and
/// checkpointed [`CompiledModuleTrainingSession::finish_with_checkpoint`]
/// finalization.
/// The retained session remains available for inspection, retry, or recovery
/// without publication.
pub struct CompiledModuleTrainingFinishError<M, R> {
    session: Box<CompiledModuleTrainingSession<M, R>>,
    source: Error,
}

/// Compiled AdamW compatibility failure retaining its intact owned session.
pub struct CompiledModuleAdamWFinishError<M, R> {
    session: Box<CompiledModuleAdamWSession<M, R>>,
    source: Error,
}

impl<M, R> CompiledModuleTrainingFinishError<M, R> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn session(&self) -> &CompiledModuleTrainingSession<M, R> {
        &self.session
    }

    pub fn into_session(self) -> CompiledModuleTrainingSession<M, R> {
        *self.session
    }

    pub fn into_parts(self) -> (CompiledModuleTrainingSession<M, R>, Error) {
        (*self.session, self.source)
    }

    /// Discards the failed runtime frontier and returns the sealed host module
    /// exactly as it currently exists, without attempting publication again.
    pub fn into_module_without_publication(self) -> M {
        (*self.session).into_module_without_publication()
    }
}

impl<M, R> fmt::Debug for CompiledModuleTrainingFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleTrainingFinishError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M, R> fmt::Display for CompiledModuleTrainingFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled training finalization failed: {}",
            self.source
        )
    }
}

impl<M, R> CompiledModuleAdamWFinishError<M, R> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn session(&self) -> &CompiledModuleAdamWSession<M, R> {
        &self.session
    }

    pub fn into_session(self) -> CompiledModuleAdamWSession<M, R> {
        *self.session
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWSession<M, R>, Error) {
        (*self.session, self.source)
    }

    /// Discards the failed runtime frontier and returns the sealed host module
    /// exactly as it currently exists, without attempting publication again.
    pub fn into_module_without_publication(self) -> M {
        (*self.session).into_module_without_publication()
    }
}

impl<M, R> fmt::Debug for CompiledModuleAdamWFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWFinishError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M, R> fmt::Display for CompiledModuleAdamWFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW finalization failed: {}",
            self.source
        )
    }
}

impl<M, R> std::error::Error for CompiledModuleAdamWFinishError<M, R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<M, R> std::error::Error for CompiledModuleTrainingFinishError<M, R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Resource-free Metal rendering of one compiled AdamW plan. Preparing it
/// uploads the plan's parameter, moment, and optimizer-step frontier into the
/// existing epoch-swapped Metal runtime.
pub struct MetalCompiledAdamWPlan {
    inner: MetalCompiledTrainingPlan,
    accumulation_capture_identity: Option<u64>,
    partial_flush: Option<MetalFixedStateTransitionPlan>,
    progress: CompiledTrainingWindowProgress,
    flush_capture_identity: Option<u64>,
    contract: MetalAdamWContract,
}

/// Optimizer-neutral strict-Metal rendering of one compiled training program.
/// Optimizer facades retain only their policy and progress around this shared
/// recurrent execution core.
struct MetalCompiledTrainingPlan {
    inner: MetalStatefulInferencePlan,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    program_identity: u64,
    evaluation: Option<(MetalFixedStateReadPlan, Vec<String>, u64)>,
}

/// Device-resident AdamW training session backed by one fixed Metal capture.
/// Parameters and optimizer slots remain in the double-buffered device state
/// frontier between calls. Batch inputs and learning rate cross the host
/// boundary on every step; requested outputs cross only when the caller uses
/// the observed [`MetalCompiledAdamW::step`] path.
pub struct MetalCompiledAdamW {
    inner: MetalCompiledTrainingProgram,
    accumulation_capture_identity: Option<u64>,
    partial_flush: Option<MetalFixedStateTransitionSession>,
    progress: CompiledTrainingWindowProgress,
    flush_capture_identity: Option<u64>,
    contract: MetalAdamWContract,
}

/// Optimizer-neutral owner of one prepared strict-Metal training program.
struct MetalCompiledTrainingProgram {
    session: MetalDeviceSession,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    program_identity: u64,
    scoreboard: Option<MetalScoreboardObserver>,
    evaluation: Option<(MetalFixedStateReadSession, Vec<String>, u64)>,
}

struct MetalCompiledTrainingRun {
    loss: TensorData,
    outputs: BTreeMap<String, TensorData>,
    report: MetalDeviceRunReport,
}

/// One committed Metal AdamW step plus its exact device execution report.
pub struct MetalCompiledAdamWStepResult {
    inner: CompiledAdamWStepResult,
    report: MetalDeviceRunReport,
}

/// One committed Metal AdamW step whose loss and named outputs remained on the
/// device. The exact replay and optimizer progress plus device report remain
/// available without manufacturing an observed [`CompiledTrainingStep`].
pub struct MetalCompiledAdamWCommitResult {
    progress: CompiledTrainingWindowProgress,
    capture_identity: u64,
    report: MetalDeviceRunReport,
}

/// One committed strict-Metal partial-window flush and its exact device report.
pub struct MetalCompiledAdamWFlushResult {
    inner: CompiledAdamWFlushResult,
    report: Option<MetalDeviceRunReport>,
}

impl MetalCompiledAdamWFlushResult {
    pub fn flushed_microbatches(&self) -> u64 {
        self.inner.flushed_microbatches()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    /// Exact device report for a committed update. Empty flushes execute no
    /// device invocation and therefore return `None`.
    pub fn report(&self) -> Option<&MetalDeviceRunReport> {
        self.report.as_ref()
    }
}

impl CompiledAdamWFlush for MetalCompiledAdamWFlushResult {
    fn flushed_microbatches(&self) -> u64 {
        MetalCompiledAdamWFlushResult::flushed_microbatches(self)
    }

    fn did_update(&self) -> bool {
        MetalCompiledAdamWFlushResult::did_update(self)
    }

    fn optimizer_step(&self) -> u64 {
        MetalCompiledAdamWFlushResult::optimizer_step(self)
    }
}

impl MetalCompiledAdamWCommitResult {
    pub fn step(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn optimizer_step(&self) -> u64 {
        self.progress.optimizer_step
    }

    pub fn accumulation_index(&self) -> u64 {
        self.progress.accumulation_index
    }

    pub fn did_update(&self) -> bool {
        self.progress.accumulation_index == 0
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl MetalCompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    pub fn accumulation_index(&self) -> u64 {
        self.inner.accumulation_index()
    }

    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl CompiledTrainingStep for MetalCompiledAdamWStepResult {
    fn loss(&self) -> &TensorData {
        MetalCompiledAdamWStepResult::loss(self)
    }

    fn loss_aggregation_weight(&self) -> u64 {
        MetalCompiledAdamWStepResult::loss_weight(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        MetalCompiledAdamWStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        MetalCompiledAdamWStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        MetalCompiledAdamWStepResult::capture_identity(self)
    }
}

impl CompiledAdamWStep for MetalCompiledAdamWStepResult {
    fn optimizer_step(&self) -> u64 {
        MetalCompiledAdamWStepResult::optimizer_step(self)
    }

    fn accumulation_index(&self) -> u64 {
        MetalCompiledAdamWStepResult::accumulation_index(self)
    }

    fn loss_weight(&self) -> u64 {
        MetalCompiledAdamWStepResult::loss_weight(self)
    }
}

fn zero_inputs(inputs: &BTreeMap<String, (Shape, DType)>) -> Result<BTreeMap<String, TensorData>> {
    inputs
        .iter()
        .map(|(name, (shape, dtype))| {
            Ok((
                name.clone(),
                TensorData::zeros_with_dtype(shape.clone(), *dtype)?,
            ))
        })
        .collect()
}

fn native_cpu_identity(
    capture_identity: u64,
    vectorized: bool,
    cache_keys: impl IntoIterator<Item = u64>,
) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in crate::cpu_jit::RENDERER_VERSION
        .as_bytes()
        .iter()
        .chain(std::env::consts::ARCH.as_bytes())
        .chain(std::env::consts::OS.as_bytes())
    {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    for value in std::iter::once(capture_identity)
        .chain(std::iter::once(if vectorized { 1 } else { 0 }))
        .chain(cache_keys)
    {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn native_cpu_run_report(
    capture_identity: u64,
    trace: &NativeMixedReplayTrace,
    traffic: NativeReplayTraffic,
    executor_wall_time: Duration,
    successful_invocation: u64,
    wall_time: Duration,
) -> NativeCpuRunReport {
    let report = NativeCpuRunReport {
        capture_identity,
        native_identity: trace.identity,
        vectorized: trace.vectorized,
        successful_invocation,
        native_item_count: trace.pure_item_cache_keys.len(),
        executed_native_item_count: traffic.executed_native_item_count,
        module_dispatch_count: traffic.module_dispatch_count,
        module_dispatched_native_item_count: traffic.module_dispatched_native_item_count,
        skipped_output_clear_count: traffic.skipped_output_clear_count,
        schedule_cache_keys: trace.pure_item_cache_keys.clone(),
        native_dispatcher_wall_time: traffic.native_dispatcher_wall_time,
        traffic: native_cpu_replay_traffic(traffic),
        executor_wall_time,
        wall_time,
    };
    debug_assert!(validate_native_cpu_run_report(&report).is_ok());
    report
}

fn validate_native_cpu_run_report(report: &NativeCpuRunReport) -> Result<()> {
    if report.module_dispatch_count > report.module_dispatched_native_item_count
        || report.module_dispatched_native_item_count > report.executed_native_item_count
        || report.executed_native_item_count > report.native_item_count
        || report.skipped_output_clear_count > report.native_item_count
    {
        return Err(training(
            "native CPU physical execution evidence exceeds logical coverage",
        ));
    }
    report
        .executor_wall_time
        .checked_sub(report.native_dispatcher_wall_time)
        .ok_or_else(|| training("native CPU dispatcher time exceeds executor time"))?;
    report
        .wall_time
        .checked_sub(report.executor_wall_time)
        .ok_or_else(|| training("native CPU executor time exceeds replay time"))?;
    Ok(())
}

const fn native_cpu_replay_traffic(traffic: NativeReplayTraffic) -> NativeCpuReplayTraffic {
    NativeCpuReplayTraffic::new(
        traffic.external_input_import_count,
        traffic.external_input_import_bytes,
        traffic.borrowed_recurrent_input_bytes,
        traffic.borrowed_recurrent_output_bytes,
    )
    .with_recurrent_inventory(
        traffic.retained_recurrent_state_count,
        traffic.retained_recurrent_state_bytes,
        traffic.replaced_recurrent_state_count,
        traffic.replaced_recurrent_state_bytes,
    )
    .with_materialized_egress(
        traffic.materialized_egress_count,
        traffic.materialized_egress_bytes,
    )
}

fn evaluation_result(
    values: Vec<TensorData>,
    output_names: &[String],
    loss_weight: u64,
    capture_identity: u64,
) -> Result<CompiledEvaluationResult> {
    if values.len() != 1 + output_names.len() {
        return Err(training("compiled evaluation output inventory differs"));
    }
    let mut values = values.into_iter();
    let loss = values
        .next()
        .expect("compiled evaluation output cardinality was checked");
    Ok(CompiledEvaluationResult {
        loss,
        outputs: output_names.iter().cloned().zip(values).collect(),
        loss_weight,
        capture_identity,
    })
}

fn take_compiled_clip_report(
    values: &mut impl Iterator<Item = TensorData>,
    enabled: bool,
) -> Option<CompiledAdamWClipReport> {
    enabled.then(|| {
        let norm = values
            .next()
            .expect("compiled clip-report norm cardinality was authenticated");
        let scale = values
            .next()
            .expect("compiled clip-report scale cardinality was authenticated");
        debug_assert_eq!(norm.shape(), &Shape::from([]));
        debug_assert_eq!(norm.dtype(), DType::F32);
        debug_assert_eq!(scale.shape(), &Shape::from([]));
        debug_assert_eq!(scale.dtype(), DType::F32);
        CompiledAdamWClipReport::new(norm.values()[0], scale.values()[0])
    })
}

fn take_compiled_window_loss_value(
    values: &mut impl Iterator<Item = TensorData>,
    enabled: bool,
) -> Option<CompiledAdamWWindowLossValue> {
    enabled.then(|| {
        let mean_loss = values
            .next()
            .expect("compiled window-loss mean cardinality was authenticated");
        let loss_weight = values
            .next()
            .expect("compiled window-loss weight cardinality was authenticated");
        debug_assert_eq!(mean_loss.shape(), &Shape::from([]));
        debug_assert_eq!(mean_loss.dtype(), DType::F32);
        debug_assert_eq!(loss_weight.shape(), &Shape::from([]));
        debug_assert_eq!(loss_weight.dtype(), DType::U64);
        CompiledAdamWWindowLossValue {
            mean_loss_bits: mean_loss.values()[0].to_bits(),
            loss_weight: loss_weight.scalar_at(0).as_u64(),
        }
    })
}

impl CpuCompiledMomentumSgd {
    pub fn compile<F>(
        config: CompiledMomentumSgdConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledMomentumSgdPlan::compile(config, parameters, build)?.prepare_cpu()
    }

    /// Compiles an ordinary module forward against optimizer-owned parameter
    /// and momentum state without taking ownership of the host module.
    pub fn compile_module<M, F>(
        config: CompiledMomentumSgdConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledMomentumSgdPlan::compile_module(config, module, build)?.prepare_cpu()
    }

    /// Recompiles a matching program and restores its exact momentum frontier
    /// before the fresh runtime is returned.
    pub fn compile_from_checkpoint<F>(
        config: CompiledMomentumSgdConfig,
        checkpoint: &CompiledMomentumSgdCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledMomentumSgdPlan::compile_from_checkpoint(config, checkpoint, build)?.prepare_cpu()
    }

    /// Recompiles a module-bound program and restores its exact parameter and
    /// momentum frontier without mutating the host module.
    pub fn compile_module_from_checkpoint<M, F>(
        config: CompiledMomentumSgdConfig,
        module: &M,
        checkpoint: &CompiledMomentumSgdCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledMomentumSgdPlan::compile_module_from_checkpoint(config, module, checkpoint, build)?
            .prepare_cpu()
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner.step(inputs, learning_rate)
    }

    /// Commits one replay while omitting only graph-named outputs.
    pub fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner.step_commit_only(inputs, learning_rate)
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn momentum_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.momentum_snapshots()
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn momentum_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.momentum_versions()
    }

    /// Snapshots the exact persistent parameter/momentum frontier once.
    pub fn checkpoint(&self) -> Result<CompiledMomentumSgdCheckpoint> {
        let plan = self.inner.plan()?;
        let capture_identity = plan.capture_identity()?;
        let step = plan.step;
        let mut parameters = BTreeMap::new();
        let mut momenta = BTreeMap::new();
        let mut parameter_versions = BTreeMap::new();
        let mut momentum_versions = BTreeMap::new();
        for (key, value) in plan.state_values {
            let version = plan.state_versions[&key];
            if let Some(name) = key.parameter_name() {
                parameters.insert(name.to_owned(), value);
                parameter_versions.insert(name.to_owned(), version);
            } else if let Some(name) = key.momentum_parameter_name() {
                momenta.insert(name.to_owned(), value);
                momentum_versions.insert(name.to_owned(), version);
            } else {
                return Err(training(
                    "compiled momentum-SGD checkpoint contains unexpected state",
                ));
            }
        }
        if parameters.keys().ne(momenta.keys())
            || parameters.keys().ne(parameter_versions.keys())
            || parameters.keys().ne(momentum_versions.keys())
        {
            return Err(training(
                "compiled momentum-SGD checkpoint state names mismatch",
            ));
        }
        Ok(CompiledMomentumSgdCheckpoint {
            capture_identity,
            step,
            parameters,
            momenta,
            parameter_versions,
            momentum_versions,
        })
    }

    fn restored_candidate(&self, checkpoint: &CompiledMomentumSgdCheckpoint) -> Result<Self> {
        CompiledMomentumSgdPlan::from_inner(self.inner.plan()?)?
            .restore_checkpoint_owned(checkpoint)?
            .prepare_cpu()
    }

    /// Validates and restores a checkpoint atomically. A rejected checkpoint
    /// leaves the live parameter and momentum frontier unchanged.
    pub fn restore_checkpoint_in_place(
        &mut self,
        checkpoint: &CompiledMomentumSgdCheckpoint,
    ) -> Result<()> {
        let restored = self.restored_candidate(checkpoint)?;
        *self = restored;
        Ok(())
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner
            .step_inner(inputs, learning_rate, injected_failure)
    }

    #[cfg(test)]
    fn commit_step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner
            .step_commit_only_inner(inputs, learning_rate, injected_failure)
    }
}

impl CompiledTrainingRuntime for CpuCompiledMomentumSgd {
    type Step = CompiledMomentumSgdStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledMomentumSgd::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        CpuCompiledMomentumSgd::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        CpuCompiledMomentumSgd::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledMomentumSgd::parameter_snapshots(self)
    }
}

impl CompiledTrainingCommitOnlyRuntime for CpuCompiledMomentumSgd {
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledMomentumSgd::commit_step(self, inputs, learning_rate)
    }
}

impl CompiledCheckpointRuntime for CpuCompiledMomentumSgd {
    type Checkpoint = CompiledMomentumSgdCheckpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        CpuCompiledMomentumSgd::checkpoint(self)
    }
}

impl CompiledCheckpointRestoreRuntime for CpuCompiledMomentumSgd {
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        CpuCompiledMomentumSgd::restore_checkpoint_in_place(self, checkpoint)
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for MetalSessionTarget {
    type Session = MetalCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        let rendered = plan.metal_plan(self.renderer().clone())?;
        match self.scoreboard_context() {
            Some(context) => {
                rendered.prepare_with_scoreboard(self.device().clone(), context.clone())
            }
            None => rendered.prepare(self.device().clone()),
        }
    }
}

impl MetalCompiledAdamWPlan {
    pub fn deployment_identity(&self) -> u64 {
        self.inner.inner.deployment_identity()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.program_identity
    }

    /// Stable identity of the exact mixed transition executed by CPU and
    /// represented by this strict-Metal state-only plan.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.contract.gradient_accumulation_steps
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.contract.max_gradient_norm
    }

    pub fn loss_scale(&self) -> f32 {
        self.contract.loss_scale
    }

    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.contract.dropout.map(|dropout| dropout.config)
    }

    pub fn dropout_blocks_per_replay(&self) -> Option<u64> {
        self.contract
            .dropout
            .map(|dropout| dropout.blocks_per_replay)
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.inner.summary()
    }

    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.inner.rendered_items()
    }

    /// Creates all native resources and uploads the captured recurrent
    /// frontier once. No training step is executed during preparation.
    pub fn prepare(self, device: MetalDevice) -> Result<MetalCompiledAdamW> {
        self.prepare_inner(device, None)
    }

    /// Creates the persistent training session and binds an epoch-state
    /// scoreboard before the first step can execute.
    pub fn prepare_with_scoreboard(
        self,
        device: MetalDevice,
        context: MetalScoreboardContext,
    ) -> Result<MetalCompiledAdamW> {
        let recorder = MetalSessionScoreboard::new_epoch_state(context, &self.inner.inner);
        self.prepare_inner(device, Some(recorder))
    }

    fn prepare_inner(
        self,
        device: MetalDevice,
        recorder: Option<MetalSessionScoreboard>,
    ) -> Result<MetalCompiledAdamW> {
        let inner = self.inner.prepare(device.clone(), recorder)?;
        let partial_flush = self
            .partial_flush
            .map(|plan| {
                plan.prepare(device, &inner.session)
                    .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledAdamW {
            inner,
            accumulation_capture_identity: self.accumulation_capture_identity,
            partial_flush,
            progress: self.progress,
            flush_capture_identity: self.flush_capture_identity,
            contract: self.contract,
        })
    }
}

impl MetalCompiledTrainingPlan {
    fn prepare(
        self,
        device: MetalDevice,
        recorder: Option<MetalSessionScoreboard>,
    ) -> Result<MetalCompiledTrainingProgram> {
        let session = self
            .inner
            .prepare(device.clone())
            .map_err(metal_training_error)?;
        let evaluation = self
            .evaluation
            .map(|(plan, output_names, capture_identity)| {
                plan.prepare(device.clone(), &session)
                    .map(|session| (session, output_names, capture_identity))
                    .map_err(metal_training_error)
            })
            .transpose()?;
        let scoreboard = recorder
            .map(|recorder| {
                MetalScoreboardObserver::bind(recorder, &session)
                    .map_err(|error| training(format!("compiled Metal scoreboard: {error}")))
            })
            .transpose()?;
        Ok(MetalCompiledTrainingProgram {
            session,
            inputs: self.inputs,
            output_names: self.output_names,
            state_input_keys: self.state_input_keys,
            program_identity: self.program_identity,
            scoreboard,
            evaluation,
        })
    }
}

impl MetalCompiledTrainingProgram {
    fn prepare_inputs(
        &self,
        mut inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<BTreeMap<String, TensorData>> {
        validate_step_inputs(&self.inputs, &inputs, &learning_rate)?;
        inputs.insert(LEARNING_RATE_INPUT.into(), learning_rate);
        Ok(inputs)
    }

    fn observe_committed_step(&mut self, run: &MetalDeviceRun) {
        if let Some(scoreboard) = &mut self.scoreboard {
            scoreboard.observe(run);
        }
    }

    fn run(&mut self, provided: &BTreeMap<String, TensorData>) -> Result<MetalCompiledTrainingRun> {
        let run = self.session.run(provided).map_err(metal_training_error)?;
        self.observe_committed_step(&run);
        let (outputs, report) = run.into_parts();
        debug_assert_eq!(outputs.len(), 1 + self.output_names.len());
        let mut outputs = outputs.into_iter();
        let loss = outputs
            .next()
            .expect("compiled Metal output cardinality was authenticated before preparation");
        let outputs = self.output_names.iter().cloned().zip(outputs).collect();
        Ok(MetalCompiledTrainingRun {
            loss,
            outputs,
            report,
        })
    }

    fn run_without_host_outputs(
        &mut self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<MetalDeviceRunReport> {
        let run = self
            .session
            .run_epoch_without_host_outputs(provided)
            .map_err(metal_training_error)?;
        debug_assert!(run.outputs().is_empty());
        debug_assert_eq!(run.report().output_count, 0);
        debug_assert_eq!(run.report().retained_d2h_calls, 0);
        debug_assert_eq!(run.report().retained_d2h_bytes, 0);
        self.observe_committed_step(&run);
        let (_, report) = run.into_parts();
        Ok(report)
    }

    fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<MetalCompiledEvaluationResult> {
        validate_evaluation_inputs(&self.inputs, &inputs)?;
        let (evaluation, output_names, capture_identity) = self
            .evaluation
            .as_mut()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        let run = evaluation
            .run(self.session.state_epoch(), &inputs)
            .map_err(metal_training_error)?;
        let (values, report) = run.into_parts();
        let inner = evaluation_result(values, output_names, 1, *capture_identity)?;
        Ok(MetalCompiledEvaluationResult { inner, report })
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation
            .as_ref()
            .map(|(_, _, capture_identity)| *capture_identity)
    }

    fn state_snapshots(&self) -> Result<BTreeMap<RecurrentStateKey, TensorData>> {
        let snapshots = self
            .session
            .state_snapshots()
            .map_err(metal_training_error)?;
        if snapshots.len() != self.state_input_keys.len()
            || snapshots.keys().ne(self.state_input_keys.keys())
        {
            return Err(training("compiled Metal state inventory mismatch"));
        }
        snapshots
            .into_iter()
            .map(|(input, value)| {
                let key = self
                    .state_input_keys
                    .get(&input)
                    .cloned()
                    .ok_or_else(|| training("compiled Metal state key is absent"))?;
                Ok((key, value))
            })
            .collect()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        let state_inputs = self
            .session
            .state_inputs()
            .iter()
            .map(|input| (input.name.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        if state_inputs.len() != self.state_input_keys.len()
            || state_inputs
                .keys()
                .copied()
                .ne(self.state_input_keys.keys().map(String::as_str))
        {
            return Err(training("compiled Metal state inventory mismatch"));
        }
        let expected = self
            .state_input_keys
            .iter()
            .filter_map(|(input, key)| {
                key.parameter_name()
                    .map(|name| (input.clone(), name.to_owned()))
            })
            .collect::<BTreeMap<_, _>>();
        if expected.is_empty() {
            return Err(training("compiled Metal parameter inventory is empty"));
        }
        let requested = expected
            .keys()
            .map(|name| state_inputs[name.as_str()].desc.id)
            .collect::<BTreeSet<_>>();
        if requested.len() != expected.len() {
            return Err(training("compiled Metal parameter state identities repeat"));
        }
        let snapshots = self
            .session
            .state_snapshot_subset(&requested)
            .map_err(metal_training_error)?;
        if snapshots.len() != expected.len() || snapshots.keys().ne(expected.keys()) {
            return Err(training(
                "compiled Metal parameter snapshot inventory mismatch",
            ));
        }
        snapshots
            .into_iter()
            .map(|(input, value)| {
                let name = expected
                    .get(&input)
                    .cloned()
                    .ok_or_else(|| training("compiled Metal parameter snapshot is unknown"))?;
                Ok((name, value))
            })
            .collect()
    }
}

impl MetalCompiledAdamW {
    fn prepare_step(
        &self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<(CompiledTrainingWindowProgress, BTreeMap<String, TensorData>)> {
        let inputs = self.inner.prepare_inputs(inputs, learning_rate)?;
        let next = adamw_window_progress(
            self.progress
                .advance_replay(self.contract.gradient_accumulation_steps),
        )?;
        if let Some(dropout) = self.contract.dropout {
            expected_dropout_counter(dropout, next.replay_step)?;
        }
        Ok((next, inputs))
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWStepResult> {
        let (next, provided) = self.prepare_step(inputs, learning_rate)?;
        let MetalCompiledTrainingRun {
            loss,
            outputs,
            report,
        } = self.inner.run(&provided)?;
        self.progress = next;
        let inner = adamw_step_result(
            CompiledTrainingStepResult {
                loss,
                loss_aggregation_weight: 1,
                outputs,
                step: self.progress.replay_step,
                capture_identity: self.inner.program_identity,
                observations: Vec::new(),
            },
            self.progress,
            1,
            self.contract.gradient_accumulation_steps,
            false,
            false,
        );
        Ok(MetalCompiledAdamWStepResult { inner, report })
    }

    /// Executes and commits the identical captured training program while
    /// leaving its loss and named outputs on the device. Batch inputs and the
    /// learning rate are still staged, the complete inactive state bank is
    /// produced, and successful replay/optimizer progress advances normally.
    /// Use [`Self::step`] whenever the caller needs to observe loss or outputs.
    pub fn step_without_host_outputs(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWCommitResult> {
        let (next, provided) = self.prepare_step(inputs, learning_rate)?;
        let report = self.inner.run_without_host_outputs(&provided)?;
        self.progress = next;
        Ok(MetalCompiledAdamWCommitResult {
            progress: self.progress,
            capture_identity: self.inner.program_identity,
            report,
        })
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<MetalCompiledEvaluationResult> {
        self.inner.evaluate(inputs)
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }

    /// Returns preparation evidence for both stateless evaluators sharing the
    /// training session's physical parameter banks. Imported trainable state
    /// contributes zero resident or initial-state uploads.
    pub fn evaluation_preparation_reports(
        &self,
    ) -> Option<[&crate::runtime::metal::MetalDevicePreparationReport; 2]> {
        self.inner
            .evaluation
            .as_ref()
            .map(|(evaluation, _, _)| evaluation.preparation_reports())
    }

    pub fn evaluation_summaries(&self) -> Option<[&MetalDeviceSessionSummary; 2]> {
        self.inner
            .evaluation
            .as_ref()
            .map(|(evaluation, _, _)| evaluation.summaries())
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.contract.gradient_accumulation_steps
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.contract.max_gradient_norm
    }

    pub fn loss_scale(&self) -> f32 {
        self.contract.loss_scale
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.program_identity
    }

    pub fn metal_session(&self) -> &MetalDeviceSession {
        &self.inner.session
    }

    /// Returns the opt-in successful-step recorder, when preparation enabled it.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.inner
            .scoreboard
            .as_ref()
            .map(MetalScoreboardObserver::recorder)
    }

    /// Returns a deterministic snapshot of all successfully observed steps.
    pub fn execution_scoreboard_report(
        &self,
    ) -> std::result::Result<Option<MetalSessionScoreboardReport>, MetalScoreboardError> {
        self.execution_scoreboard()
            .map(MetalSessionScoreboard::report)
            .transpose()
    }

    /// Returns the first fail-soft measurement error, if recording froze.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.inner
            .scoreboard
            .as_ref()
            .and_then(MetalScoreboardObserver::first_error)
    }

    /// Downloads every currently committed recurrent value once and returns
    /// it under the optimizer's semantic state keys.
    fn state_snapshots(&self) -> Result<BTreeMap<RecurrentStateKey, TensorData>> {
        self.inner.state_snapshots()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(self.state_snapshots()?, AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(self.state_snapshots()?, AdamWParameterState::SecondMoment)
    }

    pub fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(
            self.state_snapshots()?,
            AdamWParameterState::GradientAccumulator,
        )
    }

    pub fn accumulation_index(&self) -> Result<u64> {
        Ok(self.progress.accumulation_index)
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        Ok(self.progress.optimizer_step)
    }

    /// Explicit diagnostic download of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.contract
            .dropout
            .map(|_| {
                Ok(self
                    .state_snapshots()?
                    .get(&RecurrentStateKey::dropout_counter())
                    .ok_or_else(|| training("compiled Metal dropout counter is absent"))?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()
    }

    /// Clears a retained partial window entirely inside the epoch-swapped
    /// device frontier. No training run or host gradient download is performed.
    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        let (next, reset) = adamw_window_progress(
            self.progress
                .cancel(self.contract.gradient_accumulation_steps),
        )?;
        if !reset.did_discard() {
            return Ok(reset);
        }
        let state_inputs = self
            .inner
            .session
            .state_inputs()
            .iter()
            .map(|input| (input.name.as_str(), &input.desc))
            .collect::<BTreeMap<_, _>>();
        let mut replacements = BTreeMap::new();
        for (input, key) in &self.inner.state_input_keys {
            if key.is_accumulation_reset_state() {
                let desc = state_inputs
                    .get(input.as_str())
                    .ok_or_else(|| training("compiled Metal reset state is absent"))?;
                replacements.insert(
                    input.clone(),
                    TensorData::zeros_with_dtype(desc.shape.clone(), desc.dtype)?,
                );
            }
        }
        let expected = self
            .inner
            .state_input_keys
            .values()
            .filter(|key| key.is_accumulation_reset_state())
            .count();
        if replacements.len() != expected
            || !replacements
                .values()
                .any(|value| value.dtype() == DType::U64)
        {
            return Err(training("compiled Metal reset state inventory mismatch"));
        }
        self.inner
            .session
            .replace_fixed_state(replacements)
            .map_err(metal_training_error)?;
        self.progress = next;
        Ok(reset)
    }

    /// Commits a retained partial window through the separately rendered
    /// state-only capture while sharing the live epoch banks and queue.
    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWFlushResult> {
        validate_step_inputs(&BTreeMap::new(), &BTreeMap::new(), &learning_rate)?;
        let (next, flush) = adamw_window_progress(
            self.progress
                .flush_partial(self.contract.gradient_accumulation_steps),
        )?;
        if !flush.did_update() {
            return Ok(MetalCompiledAdamWFlushResult {
                inner: flush.into_adamw_result(),
                report: None,
            });
        }
        let result = flush.into_adamw_result();
        let transition = self
            .partial_flush
            .as_mut()
            .ok_or_else(|| training("compiled Metal partial flush transition is absent"))?;
        let inputs = BTreeMap::from([(LEARNING_RATE_INPUT.to_owned(), learning_rate)]);
        let run = transition
            .run(&mut self.inner.session, &inputs)
            .map_err(metal_training_error)?;
        let (outputs, report) = run.into_parts();
        debug_assert!(outputs.is_empty());
        self.progress = next;
        Ok(MetalCompiledAdamWFlushResult {
            inner: result,
            report: Some(report),
        })
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    /// Preparation evidence for the state-only transition. Imported recurrent
    /// state must contribute zero initialization uploads.
    #[cfg(test)]
    pub(crate) fn flush_preparation_report(
        &self,
    ) -> Option<&crate::runtime::metal::MetalDevicePreparationReport> {
        self.partial_flush
            .as_ref()
            .map(MetalFixedStateTransitionSession::preparation_report)
    }

    /// Downloads one coherent active state bank and encodes the same portable
    /// checkpoint format accepted by [`CpuCompiledAdamW::compile_from_checkpoint`].
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        let states = self.state_snapshots()?;
        let topology = CompiledTrainingWindowTopology::from_validated_parts(
            self.contract.gradient_accumulation_steps,
            false,
            false,
        );
        let optimizer_step = states
            .get(&RecurrentStateKey::adamw_global(AdamWGlobalState::Step))
            .ok_or_else(|| training("compiled Metal optimizer step is absent"))?
            .scalar_at(0)
            .as_u64();
        let accumulation_index = if !topology.accumulating() {
            0
        } else {
            states
                .get(&RecurrentStateKey::adamw_global(
                    AdamWGlobalState::AccumulationIndex,
                ))
                .ok_or_else(|| training("compiled Metal accumulation index is absent"))?
                .scalar_at(0)
                .as_u64()
        };
        validate_adamw_progress(self.progress, self.contract.gradient_accumulation_steps)?;
        if optimizer_step != self.progress.optimizer_step
            || accumulation_index != self.progress.accumulation_index
        {
            return Err(training("compiled Metal AdamW progress state mismatch"));
        }
        let dropout_block_counter = self
            .contract
            .dropout
            .map(|dropout| {
                let counter = states
                    .get(&RecurrentStateKey::dropout_counter())
                    .ok_or_else(|| training("compiled Metal dropout counter is absent"))?
                    .scalar_at(0)
                    .as_u64();
                if counter != expected_dropout_counter(dropout, self.progress.replay_step)? {
                    return Err(training(
                        "compiled Metal dropout counter and replay progress diverged",
                    ));
                }
                Ok(counter)
            })
            .transpose()?;
        let parameters = metal_parameter_snapshots(&states);
        let first_moments =
            metal_adamw_state_snapshots(states.clone(), AdamWParameterState::FirstMoment)?;
        let second_moments =
            metal_adamw_state_snapshots(states.clone(), AdamWParameterState::SecondMoment)?;
        let gradient_accumulators =
            metal_adamw_state_snapshots(states, AdamWParameterState::GradientAccumulator)?;
        CompiledAdamWCheckpoint::from_bytes(encode_adamw_checkpoint(
            AdamWCheckpointProgress {
                capture_identity: self.inner.program_identity,
                accumulation_capture_identity: self.accumulation_capture_identity,
                replay_step: self.progress.replay_step,
                optimizer_step: self.progress.optimizer_step,
                accumulation_steps: self.contract.gradient_accumulation_steps,
                accumulation_index: self.progress.accumulation_index,
                discarded_microbatches: self.progress.discarded_microbatches,
                flushed_window_count: self.progress.flushed_window_count,
                flushed_microbatch_count: self.progress.flushed_microbatch_count,
                flush_capture_identity: self.flush_capture_identity,
                dropout_block_counter,
                accumulated_token_count: None,
                window_loss_report: false,
                reset_transition_count: 0,
                reset_capture_identity: None,
            },
            AdamWCheckpointTensors {
                parameters,
                first_moments,
                second_moments,
                gradient_accumulators,
                accumulated_loss_numerator: None,
            },
        )?)
    }
}

impl CompiledTrainingRuntime for MetalCompiledAdamW {
    type Step = MetalCompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        MetalCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        MetalCompiledAdamW::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        MetalCompiledAdamW::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::parameter_snapshots(self)
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.parameter_snapshots()?,
            &self.contract.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for MetalCompiledAdamW {
    type Evaluation = MetalCompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        MetalCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::evaluation_capture_identity(self)
    }
}

impl CompiledCheckpointRuntime for MetalCompiledAdamW {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        MetalCompiledAdamW::checkpoint(self)
    }
}

impl CompiledAdamWRuntime for MetalCompiledAdamW {
    fn gradient_accumulation_steps(&self) -> u64 {
        MetalCompiledAdamW::gradient_accumulation_steps(self)
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        MetalCompiledAdamW::max_gradient_norm(self)
    }

    fn loss_scale(&self) -> f32 {
        MetalCompiledAdamW::loss_scale(self)
    }

    fn optimizer_step(&self) -> Result<u64> {
        MetalCompiledAdamW::optimizer_step(self)
    }

    fn accumulation_index(&self) -> Result<u64> {
        MetalCompiledAdamW::accumulation_index(self)
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        MetalCompiledAdamW::zero_grad(self)
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::first_moment_snapshots(self)
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::second_moment_snapshots(self)
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::gradient_accumulator_snapshots(self)
    }
}

impl CompiledTrainingWindowCommitRuntime for MetalCompiledAdamW {
    type WindowCommit = MetalCompiledAdamWFlushResult;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        MetalCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::flush_capture_identity(self)
    }
}

impl CompiledAdamWFlushRuntime for MetalCompiledAdamW {
    type Flush = MetalCompiledAdamWFlushResult;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        CompiledTrainingWindowCommitRuntime::commit_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::flush_capture_identity(self)
    }
}

fn metal_parameter_snapshots(
    states: &BTreeMap<RecurrentStateKey, TensorData>,
) -> BTreeMap<String, TensorData> {
    states
        .iter()
        .filter_map(|(key, value)| {
            key.parameter_name()
                .map(|name| (name.to_owned(), value.clone()))
        })
        .collect()
}

fn metal_adamw_state_snapshots(
    states: BTreeMap<RecurrentStateKey, TensorData>,
    state: AdamWParameterState,
) -> Result<BTreeMap<String, TensorData>> {
    Ok(states
        .into_iter()
        .filter_map(|(key, value)| {
            key.parameter_for_adamw_state(state)
                .map(|name| (name.to_owned(), value))
        })
        .collect())
}

fn validate_cpu_adamw_state(
    inner: &CpuCompiledTrainingProgram,
    progress: CompiledTrainingWindowProgress,
    accumulation_steps: u64,
) -> Result<()> {
    let optimizer_step = inner
        .global_snapshot(AdamWGlobalState::Step)?
        .scalar_at(0)
        .as_u64();
    let topology =
        CompiledTrainingWindowTopology::from_validated_parts(accumulation_steps, false, false);
    let accumulation_index = if !topology.accumulating() {
        0
    } else {
        inner
            .global_snapshot(AdamWGlobalState::AccumulationIndex)?
            .scalar_at(0)
            .as_u64()
    };
    if optimizer_step != progress.optimizer_step
        || accumulation_index != progress.accumulation_index
    {
        return Err(training("compiled CPU AdamW progress state mismatch"));
    }
    Ok(())
}

fn schedule_error(error: impl std::fmt::Display) -> Error {
    training(format!("compiled schedule: {error}"))
}

fn replay_error(error: ReplayError) -> Error {
    training(format!("compiled replay: {error:?}"))
}

fn cursor_projection_error(error: RecurrentCursorProjectionError) -> Error {
    match error {
        RecurrentCursorProjectionError::IncompleteSourceFrontier => {
            training("compiled auxiliary state frontier is incomplete")
        }
        RecurrentCursorProjectionError::VersionOverflow => {
            training("compiled auxiliary state version would overflow")
        }
        RecurrentCursorProjectionError::InvalidSource(error) => replay_error(error),
    }
}

fn captured_inference_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled recurrent capture: {error:?}"))
}

fn metal_training_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled Metal runtime: {error:?}"))
}

fn runtime_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled persistent runtime: {error:?}"))
}

fn effect_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled effect graph: {error:?}"))
}

fn training(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "compiled_training/tests.rs"]
mod tests;
