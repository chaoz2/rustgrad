//! Graph-free CPU replay for static training programs with recurrent state.

mod adamw_checkpoint;
mod adamw_contract;
mod dropout;
mod exchange;
mod module_adamw_checkpoint;
mod module_state;
mod native_cpu_evidence;
mod observation;
mod optimizer_lowering;
mod policy;
mod program_artifact;
mod resume_bundle;
mod runtime;
mod state_schema;
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
pub use self::dropout::{CompiledDropoutConfig, CompiledDropoutKey};
use self::dropout::{CompiledDropoutState, CompiledDropoutStream, expected_dropout_counter};
pub use self::exchange::*;
use self::exchange::{CompiledAdamWWindowLossValue, CompiledInputPolicy};
pub use self::module_adamw_checkpoint::CompiledModuleAdamWCheckpoint;
use self::module_adamw_checkpoint::encode_module_adamw_checkpoint;
pub use self::module_state::TrainingParameterInit;
use self::module_state::{CompiledModuleSeal, ModuleParameterPlan};
use self::native_cpu_evidence::native_preparation_wall_time;
pub use self::native_cpu_evidence::*;
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
pub use self::resume_bundle::{CompiledAdamWResumeBundle, CompiledAdamWResumeBundleFileError};
pub use self::runtime::*;
use self::state_schema::{
    AdamWGlobalState, AdamWParameterState, INTERNAL_PREFIX, RecurrentStateKey, StateSpec,
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
    NativeMixedReplayTrace, NodeId, ReplayError, Result, Scalar, Schedule, ScheduleStateBinding,
    ScheduleValueBinding, Shape, TensorData, bind_schedule_states, combine_mixed_schedules,
    schedule_effects, schedule_many,
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

struct PreparedNativeCpuProgram {
    report: NativeCpuProgramPreparationReport,
    replay: PreparedRecurrentNativeReplay,
}

struct PreparedNativeCpuEvaluation {
    report: NativeCpuProgramPreparationReport,
    plan: PlannedNativeItems,
    parameter_inputs: Vec<PreparedNativeEvaluationParameterInput>,
}

struct NativeCpuEvaluationPreparation {
    inputs: Option<BTreeMap<String, TensorData>>,
    parameter_inputs: Vec<PreparedNativeEvaluationParameterInput>,
    residual_wall_time: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedNativeEvaluationParameterInput {
    parameter: String,
    input: String,
    buffer: u64,
    shape: Shape,
    dtype: DType,
    bytes: usize,
}

impl PreparedNativeCpuEvaluation {
    fn validate(
        &self,
        capture_identity: u64,
        capture: &CapturedSchedule,
        parameter_buffers: &BTreeMap<String, u64>,
    ) -> Result<()> {
        self.plan
            .validate_structure(capture)
            .map_err(replay_error)?;
        let native_identity = native_cpu_identity(
            capture_identity,
            self.plan.vectorized(),
            self.plan.schedule_cache_keys().iter().copied(),
        );
        if self.report.capture_identity != capture_identity
            || self.report.native_identity != native_identity
            || self.report.native_item_count != self.plan.item_count()
            || self.report.cache_hit_count != self.plan.cache_hit_count()
            || self.report.cache_miss_count != self.plan.cache_miss_count()
            || self.report.work
                != NativeCpuPreparationWork::from_module(self.plan.module_preparation())
            || self.report.vectorized != self.plan.vectorized()
            || capture.items.iter().map(|item| item.cache_key).ne(self
                .plan
                .schedule_cache_keys()
                .iter()
                .copied())
        {
            return Err(training(
                "compiled native CPU evaluation preparation identity mismatch",
            ));
        }
        self.report.validate_work()?;
        if parameter_buffers.len() != self.parameter_inputs.len() {
            return Err(training(
                "compiled native CPU evaluation parameter mapping mismatch",
            ));
        }
        let mut parameters = BTreeSet::new();
        let mut inputs = BTreeSet::new();
        let mut buffers = BTreeSet::new();
        for binding in &self.parameter_inputs {
            let input = capture
                .inputs
                .iter()
                .find(|input| input.name == binding.input)
                .ok_or_else(|| {
                    training("compiled native CPU evaluation parameter input is absent")
                })?;
            if !parameters.insert(binding.parameter.as_str())
                || parameter_buffers.get(&binding.parameter) != Some(&binding.buffer)
                || !inputs.insert(binding.input.as_str())
                || !buffers.insert(binding.buffer)
                || input.desc.shape != binding.shape
                || input.desc.dtype != binding.dtype
                || input.desc.bytes != binding.bytes
            {
                return Err(training(
                    "compiled native CPU evaluation parameter mapping mismatch",
                ));
            }
        }
        Ok(())
    }

    fn active_parameter_states(&self, frontier: &[BufferState]) -> Result<Vec<BufferState>> {
        let mut frontier_by_buffer = BTreeMap::new();
        for state in frontier {
            if frontier_by_buffer.insert(state.buffer, state).is_some() {
                return Err(training(
                    "compiled native CPU recurrent frontier contains duplicate buffers",
                ));
            }
        }
        self.parameter_inputs
            .iter()
            .map(|binding| {
                let state = frontier_by_buffer.get(&binding.buffer).ok_or_else(|| {
                    training("compiled native CPU evaluation parameter state is absent")
                })?;
                if state.shape != binding.shape
                    || state.dtype != binding.dtype
                    || state.bytes != binding.bytes
                {
                    return Err(training(
                        "compiled native CPU evaluation parameter state descriptor mismatch",
                    ));
                }
                Ok((*state).clone())
            })
            .collect()
    }
}

/// One compiled momentum-SGD training program.
pub struct CpuCompiledMomentumSgd {
    inner: CpuCompiledTrainingProgram,
}

/// Resource-free compiled AdamW program ready for a concrete runtime.
///
/// Compilation owns graph construction, differentiation, scheduling, capture,
/// recurrent-state admission, and optional checkpoint restoration. Preparing
/// the plan then chooses CPU replay or strict Metal rendering without changing
/// the authenticated program or optimizer frontier.
#[derive(Clone)]
pub struct CompiledAdamWPlan {
    inner: CompiledTrainingPlan,
    partial_flush: Option<CompiledAdamWAuxiliaryPlan>,
    zero_grad: Option<CompiledAdamWAuxiliaryPlan>,
    program_identity: u64,
    contract: CompiledAdamWContract,
    progress: CompiledTrainingWindowProgress,
    evaluation: Option<CompiledEvaluationPlan>,
    compile_phases: Option<CompiledTrainingCompileObservation>,
}

struct ValidatedAdamWCheckpointFrontier {
    replay_step: u64,
    values: BTreeMap<RecurrentStateKey, TensorData>,
    versions: BTreeMap<RecurrentStateKey, u64>,
    progress: CompiledTrainingWindowProgress,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AdamWCheckpointRestoreCounts {
    borrowed_plan_clones: usize,
    consumed_plan_restores: usize,
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct AdamWPlanCaptureAllocations {
    main: (usize, usize, usize),
    accumulation: Option<(usize, usize, usize)>,
    partial_flush: Option<(usize, usize, usize)>,
    zero_grad: Option<(usize, usize, usize)>,
    evaluation: Option<(usize, usize, usize)>,
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct AdamWPlanTopologyAllocations {
    main: (usize, usize),
    accumulation: Option<(usize, usize, usize)>,
    partial_flush: Option<(usize, usize, usize)>,
    zero_grad: Option<(usize, usize, usize)>,
    evaluation: Option<(usize, usize)>,
}

#[cfg(test)]
std::thread_local! {
    static ADAMW_CHECKPOINT_RESTORE_COUNTS: std::cell::Cell<AdamWCheckpointRestoreCounts> =
        const { std::cell::Cell::new(AdamWCheckpointRestoreCounts {
            borrowed_plan_clones: 0,
            consumed_plan_restores: 0,
        }) };
}

#[cfg(test)]
fn record_adamw_checkpoint_restore(update: impl FnOnce(&mut AdamWCheckpointRestoreCounts)) {
    ADAMW_CHECKPOINT_RESTORE_COUNTS.with(|counts| {
        let mut next = counts.get();
        update(&mut next);
        counts.set(next);
    });
}

#[cfg(test)]
fn adamw_checkpoint_restore_counts() -> AdamWCheckpointRestoreCounts {
    ADAMW_CHECKPOINT_RESTORE_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn captured_schedule_allocation(capture: &CapturedSchedule) -> (usize, usize, usize) {
    (
        capture.items.as_ptr() as usize,
        capture.items.len(),
        capture.items.capacity(),
    )
}

/// Explicit scalar or token-mean objective returned by a compiled module
/// training or evaluation builder.
///
/// [`Scalar`](Self::Scalar) is the already-normalized scalar loss used by the
/// ordinary compiled AdamW policy. [`TokenMean`](Self::TokenMean) is a
/// fixed-shape F32 tensor of per-token losses; compilation combines it with
/// the explicit mask or target-derived ignore-index policy configured on
/// [`CompiledAdamWConfig`] and owns the resulting masked mean as both the
/// public loss and differentiation root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledAdamWObjective {
    Scalar(NodeId),
    TokenMean(NodeId),
}

impl CompiledAdamWObjective {
    pub const fn scalar(loss: NodeId) -> Self {
        Self::Scalar(loss)
    }

    pub const fn token_mean(losses: NodeId) -> Self {
        Self::TokenMean(losses)
    }

    pub const fn node(self) -> NodeId {
        match self {
            Self::Scalar(node) | Self::TokenMean(node) => node,
        }
    }
}

/// Compact result of building one compiled AdamW module training or evaluation
/// graph.
///
/// The objective makes scalar-loss versus compiler-owned token-mean policy
/// explicit at the builder boundary. Named outputs retain their existing
/// replay behavior and capture identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAdamWGraph {
    objective: CompiledAdamWObjective,
    outputs: BTreeMap<String, NodeId>,
}

/// Compiler-derived graph nodes for one configured ignore-index token policy.
///
/// All three nodes have the configured fixed target shape. [`Self::targets`] is
/// the declared I32 target input, [`Self::validity`] is the Bool result of the
/// exact `target != ignore_index` comparison, and [`Self::weight`] is that same
/// validity cast to F32. Context-aware builders may reshape the Bool node for a
/// model's attention policy while compilation reuses the F32 node for the
/// token-mean objective and optimizer window accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWIgnoreIndexContext {
    targets: NodeId,
    validity: NodeId,
    weight: NodeId,
}

impl CompiledAdamWIgnoreIndexContext {
    pub const fn targets(self) -> NodeId {
        self.targets
    }

    pub const fn validity(self) -> NodeId {
        self.validity
    }

    pub const fn weight(self) -> NodeId {
        self.weight
    }
}

impl CompiledAdamWGraph {
    pub fn new(objective: CompiledAdamWObjective, outputs: BTreeMap<String, NodeId>) -> Self {
        Self { objective, outputs }
    }

    pub fn scalar(loss: NodeId, outputs: BTreeMap<String, NodeId>) -> Self {
        Self::new(CompiledAdamWObjective::Scalar(loss), outputs)
    }

    pub fn token_mean(losses: NodeId, outputs: BTreeMap<String, NodeId>) -> Self {
        Self::new(CompiledAdamWObjective::TokenMean(losses), outputs)
    }

    pub const fn objective(&self) -> CompiledAdamWObjective {
        self.objective
    }

    pub fn outputs(&self) -> &BTreeMap<String, NodeId> {
        &self.outputs
    }

    pub fn into_parts(self) -> (CompiledAdamWObjective, BTreeMap<String, NodeId>) {
        (self.objective, self.outputs)
    }
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

/// One compiled AdamW training program with recurrent first/second moments, a
/// graph-owned step counter, and capture-authenticated gradient policies.
pub struct CpuCompiledAdamW {
    inner: CpuCompiledTrainingProgram,
    partial_flush: Option<CompiledAdamWAuxiliaryPlan>,
    zero_grad: Option<CompiledAdamWAuxiliaryPlan>,
    contract: CompiledAdamWContract,
    progress: CompiledTrainingWindowProgress,
    evaluation: Option<CpuCompiledEvaluation>,
    non_finite_policy: CpuNonFinitePolicy,
}

/// Strict-native CPU AdamW session prepared from the same authenticated plan
/// as [`CpuCompiledAdamW`]. Optimizer, progress, checkpoint, accumulation,
/// dropout, and evaluation ownership remain in the shared CPU core; only pure
/// schedule execution is replaced with strict native JIT replay.
pub struct NativeCpuCompiledAdamW<'a> {
    inner: CpuCompiledAdamW,
    executor: &'a CapturedReplayExecutor,
    main_replay: PreparedRecurrentNativeReplay,
    accumulation_replay: Option<PreparedRecurrentNativeReplay>,
    partial_flush_replay: Option<PreparedRecurrentNativeReplay>,
    zero_grad_replay: Option<PreparedRecurrentNativeReplay>,
    evaluation_replay: Option<PreparedNativeCpuEvaluation>,
    preparation: NativeCpuCompiledAdamWPreparationReport,
    successful_steps: u64,
    successful_flushes: u64,
    successful_zero_grads: u64,
    successful_evaluations: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum NativeCpuTrainingProgramRole {
    Main,
    Accumulation,
    PartialFlush,
    ZeroGrad,
    Evaluation,
}

impl NativeCpuTrainingProgramRole {
    const fn name(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Accumulation => "accumulation",
            Self::PartialFlush => "partial-flush",
            Self::ZeroGrad => "zero-grad",
            Self::Evaluation => "evaluation",
        }
    }

    const fn render_capsule_role(self) -> NativeCpuRenderCapsuleProgramRole {
        match self {
            Self::Main => NativeCpuRenderCapsuleProgramRole::Main,
            Self::Accumulation => NativeCpuRenderCapsuleProgramRole::Accumulation,
            Self::PartialFlush => NativeCpuRenderCapsuleProgramRole::PartialFlush,
            Self::ZeroGrad => NativeCpuRenderCapsuleProgramRole::ZeroGrad,
            Self::Evaluation => NativeCpuRenderCapsuleProgramRole::Evaluation,
        }
    }
}

struct NativeCpuTrainingProgramDrafts<D> {
    by_role: BTreeMap<NativeCpuTrainingProgramRole, D>,
}

impl<D> NativeCpuTrainingProgramDrafts<D> {
    fn new() -> Self {
        Self {
            by_role: BTreeMap::new(),
        }
    }

    fn insert(&mut self, role: NativeCpuTrainingProgramRole, draft: D) -> Result<()> {
        if self.by_role.insert(role, draft).is_some() {
            return Err(training(format!(
                "compiled native CPU {} planning draft is duplicated",
                role.name()
            )));
        }
        Ok(())
    }

    fn take(&mut self, role: NativeCpuTrainingProgramRole) -> Result<D> {
        self.by_role.remove(&role).ok_or_else(|| {
            training(format!(
                "compiled native CPU {} planning draft is absent",
                role.name()
            ))
        })
    }

    fn is_empty(&self) -> bool {
        self.by_role.is_empty()
    }
}

struct NativeCpuTrainingProgramBatch<C, D> {
    programs: Vec<(NativeCpuTrainingProgramRole, C, D)>,
}

impl<C, D> NativeCpuTrainingProgramBatch<C, D> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            programs: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, role: NativeCpuTrainingProgramRole, capture: C, draft: D) -> Result<()> {
        match self.programs.last() {
            None if role != NativeCpuTrainingProgramRole::Main => {
                return Err(training("compiled native CPU main program is absent"));
            }
            Some((previous, _, _)) if *previous >= role => {
                return Err(training("compiled native CPU program role order differs"));
            }
            _ => {}
        }
        self.programs.push((role, capture, draft));
        Ok(())
    }

    fn into_planning_inputs(self) -> (Vec<NativeCpuTrainingProgramRole>, Vec<(C, D)>) {
        let mut roles = Vec::with_capacity(self.programs.len());
        let mut programs = Vec::with_capacity(self.programs.len());
        for (role, capture, draft) in self.programs {
            roles.push(role);
            programs.push((capture, draft));
        }
        (roles, programs)
    }
}

struct NativeCpuTrainingPrograms<P> {
    main: P,
    accumulation: Option<P>,
    partial_flush: Option<P>,
    zero_grad: Option<P>,
    evaluation: Option<P>,
}

impl<P> NativeCpuTrainingPrograms<P> {
    fn from_ordered(roles: Vec<NativeCpuTrainingProgramRole>, plans: Vec<P>) -> Result<Self> {
        if roles.len() != plans.len() {
            return Err(training("compiled native CPU plan inventory differs"));
        }
        let mut main = None;
        let mut accumulation = None;
        let mut partial_flush = None;
        let mut zero_grad = None;
        let mut evaluation = None;
        for (role, plan) in roles.into_iter().zip(plans) {
            let slot = match role {
                NativeCpuTrainingProgramRole::Main => &mut main,
                NativeCpuTrainingProgramRole::Accumulation => &mut accumulation,
                NativeCpuTrainingProgramRole::PartialFlush => &mut partial_flush,
                NativeCpuTrainingProgramRole::ZeroGrad => &mut zero_grad,
                NativeCpuTrainingProgramRole::Evaluation => &mut evaluation,
            };
            if slot.replace(plan).is_some() {
                return Err(training("compiled native CPU program role is duplicated"));
            }
        }
        Ok(Self {
            main: main.ok_or_else(|| training("compiled native CPU main plan is absent"))?,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
        })
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

/// Resource-free output of compiled training graph construction.
#[derive(Clone)]
struct CompiledTrainingPlan {
    capture: Arc<CapturedMixedSchedule>,
    recurrent_capture: Arc<CompiledRecurrentCapture>,
    inputs: BTreeMap<String, (Shape, DType)>,
    phase_outputs: CompiledTrainingPhaseOutputSchema,
    parameter_buffers: BTreeMap<String, u64>,
    optimizer_buffers: BTreeMap<RecurrentStateKey, u64>,
    workload_buffers: BTreeMap<RecurrentStateKey, u64>,
    state_input_buffers: BTreeMap<String, u64>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    state_values: BTreeMap<RecurrentStateKey, TensorData>,
    state_versions: BTreeMap<RecurrentStateKey, u64>,
    recurrent_store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
    frozen_parameter_nodes: BTreeSet<NodeId>,
    step: u64,
    accumulation: Option<CompiledTrainingSiblingPlan>,
}

#[derive(Clone)]
struct CompiledTrainingSiblingPlan {
    phase: CompiledRecurrentPhasePlan,
}

#[derive(Clone)]
enum CompiledRecurrentPhaseAdmission {
    RetainUnchanged,
    Replace {
        store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
    },
}

#[derive(Clone)]
struct CompiledRecurrentPhasePlan {
    capture: Arc<CapturedMixedSchedule>,
    recurrent_capture: CompiledRecurrentCapture,
    state_buffers: BTreeMap<RecurrentStateKey, u64>,
    cursor_projection: Arc<PreparedRecurrentCursorProjection>,
    capture_identity: u64,
    admission: CompiledRecurrentPhaseAdmission,
}

#[derive(Clone)]
struct CompiledAdamWAuxiliaryPlan {
    phase: CompiledRecurrentPhasePlan,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    outputs: CompiledAdamWAuxiliaryOutputSchema,
}

impl CompiledRecurrentPhasePlan {
    fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    fn store_groups(&self) -> &[crate::engine::RecurrentStoreGroupManifest] {
        match &self.admission {
            CompiledRecurrentPhaseAdmission::RetainUnchanged => &[],
            CompiledRecurrentPhaseAdmission::Replace { store_groups } => store_groups,
        }
    }

    fn retains_unchanged(&self) -> bool {
        matches!(
            &self.admission,
            CompiledRecurrentPhaseAdmission::RetainUnchanged
        )
    }
}

impl CompiledTrainingSiblingPlan {
    fn phase(&self) -> &CompiledRecurrentPhasePlan {
        &self.phase
    }
}

impl CompiledAdamWAuxiliaryPlan {
    fn phase(&self) -> &CompiledRecurrentPhasePlan {
        &self.phase
    }
}

#[derive(Clone)]
struct CompiledEvaluationPlan {
    inference: CompiledEvaluationCapture,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    parameter_inputs: BTreeMap<String, String>,
    loss_weight_policy: Option<CompiledTokenWeightPolicy>,
    allow_zero_valid_token_microbatches: bool,
    capture_identity: u64,
}

#[derive(Clone, Debug)]
struct CompiledRecurrentCapture {
    stateful: Option<CapturedStatefulInference>,
    portable: Option<Arc<PortableCapturedInferenceRecipe>>,
    execution_plan: Arc<ExecutionPlanSummary>,
}

impl CompiledRecurrentCapture {
    fn from_stateful(stateful: CapturedStatefulInference) -> Self {
        Self {
            execution_plan: Arc::new(stateful.execution_plan().clone()),
            stateful: Some(stateful),
            portable: None,
        }
    }

    fn from_artifact(
        capture: &CapturedMixedSchedule,
        portable: Option<PortableCapturedInferenceRecipe>,
    ) -> Result<Self> {
        let (pure, execution_plan) = artifact_recurrent_execution_plan(capture)?;
        if let Some(portable) = &portable
            && portable.capture_bytes() != pure.to_bytes().map_err(replay_error)?
        {
            return Err(training(
                "compiled program artifact Metal recipe capture differs",
            ));
        }
        Ok(Self {
            stateful: None,
            portable: portable.map(Arc::new),
            execution_plan: Arc::new(execution_plan),
        })
    }

    fn from_canonical_mixed(
        graph: &Graph,
        capture: &CapturedMixedSchedule,
        public_requested: &[NodeId],
        state_links: &[InferenceStateLink],
        initial_state: BTreeMap<String, TensorData>,
    ) -> Result<Self> {
        let prefix = AuthenticatedRecurrentPrefix::from_mixed(capture)?;
        #[cfg(test)]
        let reference_initial_state = initial_state.clone();
        let stateful = CapturedStatefulInference::from_captured_graph(
            graph,
            prefix.capture,
            prefix.execution_plan,
            public_requested,
            state_links,
            initial_state,
        )
        .map_err(captured_inference_error)?;
        #[cfg(test)]
        {
            record_canonical_recurrent_capture();
            if canonical_recurrent_reference_enabled() {
                let reference = CapturedStatefulInference::from_graph(
                    graph,
                    public_requested,
                    state_links,
                    reference_initial_state,
                )
                .map_err(captured_inference_error)?;
                let stateful_recipe = stateful
                    .portable_recipe(PortableInferenceHostPolicy::None)
                    .and_then(|recipe| recipe.to_bytes())
                    .map_err(captured_inference_error)?;
                let reference_recipe = reference
                    .portable_recipe(PortableInferenceHostPolicy::None)
                    .and_then(|recipe| recipe.to_bytes())
                    .map_err(captured_inference_error)?;
                if stateful.capture().to_bytes().map_err(replay_error)?
                    != reference.capture().to_bytes().map_err(replay_error)?
                    || stateful.execution_plan() != reference.execution_plan()
                    || stateful.public_output_count() != reference.public_output_count()
                    || stateful.state_links().ne(reference.state_links())
                    || stateful.initial_state() != reference.initial_state()
                    || stateful.deployment_identity() != reference.deployment_identity()
                    || stateful_recipe != reference_recipe
                {
                    return Err(training(
                        "canonical recurrent capture differs from reference construction",
                    ));
                }
                record_reference_recurrent_capture();
            }
        }
        Ok(Self::from_stateful(stateful))
    }

    fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.execution_plan.as_ref()
    }

    fn stateful(
        &self,
        initial_state: BTreeMap<String, TensorData>,
    ) -> Result<CapturedStatefulInference> {
        if let Some(stateful) = &self.stateful {
            return stateful
                .clone()
                .with_initial_state(initial_state)
                .map_err(captured_inference_error);
        }
        self.portable
            .as_ref()
            .ok_or_else(|| {
                training("compiled AdamW program artifacts currently prepare on CPU only")
            })?
            .instantiate_stateful(initial_state)
            .map_err(captured_inference_error)
    }

    fn portable_recipe(
        &self,
        policy: PortableInferenceHostPolicy,
    ) -> Result<PortableCapturedInferenceRecipe> {
        if let Some(recipe) = &self.portable {
            return Ok(recipe.as_ref().clone());
        }
        self.stateful
            .as_ref()
            .ok_or_else(|| training("compiled recurrent capture is absent"))?
            .portable_recipe(policy)
            .map_err(captured_inference_error)
    }

    fn portable_training_recipe(
        &self,
        host_token_inputs: &BTreeMap<String, Shape>,
        frozen_parameter_nodes: &BTreeSet<NodeId>,
    ) -> Result<PortableCapturedInferenceRecipe> {
        if let Some(recipe) = &self.portable {
            return Ok(recipe.as_ref().clone());
        }
        self.stateful
            .as_ref()
            .ok_or_else(|| training("compiled recurrent capture is absent"))?
            .clone()
            .with_authenticated_training_host_indices(host_token_inputs, frozen_parameter_nodes)
            .map_err(captured_inference_error)?
            .portable_recipe(if host_token_inputs.is_empty() {
                PortableInferenceHostPolicy::None
            } else {
                PortableInferenceHostPolicy::Training
            })
            .map_err(captured_inference_error)
    }
}

fn artifact_recurrent_execution_plan(
    capture: &CapturedMixedSchedule,
) -> Result<(CapturedSchedule, ExecutionPlanSummary)> {
    #[cfg(test)]
    program_artifact::record_recurrent_execution_plan();
    let prefix = AuthenticatedRecurrentPrefix::from_mixed(capture)?;
    Ok((prefix.capture, prefix.execution_plan))
}

struct AuthenticatedRecurrentPrefix {
    capture: CapturedSchedule,
    execution_plan: ExecutionPlanSummary,
}

impl AuthenticatedRecurrentPrefix {
    fn from_mixed(capture: &CapturedMixedSchedule) -> Result<Self> {
        let split = capture
            .schedule
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| training("compiled artifact mixed capture has no effects"))?;
        if capture.schedule.items[split..]
            .iter()
            .any(|item| !item.is_effect())
        {
            return Err(training(
                "compiled artifact mixed capture does not have an ordered effect suffix",
            ));
        }
        let mut pure = capture.schedule.clone();
        pure.items.truncate(split);
        let split =
            u64::try_from(split).map_err(|_| training("compiled artifact split overflows"))?;
        for item in &mut pure.items {
            item.consumers.retain(|consumer| *consumer < split);
        }
        pure.requested.extend(
            capture
                .value_bindings
                .iter()
                .map(|binding| binding.producer_output.id),
        );
        let specialization = pure
            .specialized_from
            .as_ref()
            .map(|source| (source.source_identity, source.bindings.as_slice()));
        crate::schedule::rekey_schedule_items(&mut pure.items, &[], specialization)
            .map_err(schedule_error)?;
        pure.identity = crate::schedule::artifact::identity(&pure)
            .map_err(|error| training(format!("compiled artifact capture identity: {error}")))?;
        let pure = CapturedSchedule::from_bytes(&pure.to_bytes().map_err(replay_error)?)
            .map_err(replay_error)?;
        let execution_plan = ExecutionPlanSummary::from_capture(&pure, true)
            .map_err(|error| training(format!("compiled artifact execution summary: {error}")))?;
        Ok(Self {
            capture: pure,
            execution_plan,
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CanonicalRecurrentCaptureCounts {
    canonical: usize,
    reference: usize,
}

#[cfg(test)]
std::thread_local! {
    static CANONICAL_RECURRENT_CAPTURE_COUNTS: std::cell::Cell<CanonicalRecurrentCaptureCounts> =
        const { std::cell::Cell::new(CanonicalRecurrentCaptureCounts {
            canonical: 0,
            reference: 0,
        }) };
    static CANONICAL_RECURRENT_REFERENCE_ENABLED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
struct CanonicalRecurrentReferenceGuard(bool);

#[cfg(test)]
impl Drop for CanonicalRecurrentReferenceGuard {
    fn drop(&mut self) {
        CANONICAL_RECURRENT_REFERENCE_ENABLED.with(|enabled| enabled.set(self.0));
    }
}

#[cfg(test)]
fn with_canonical_recurrent_reference<T>(f: impl FnOnce() -> T) -> T {
    let previous = CANONICAL_RECURRENT_REFERENCE_ENABLED.with(|enabled| enabled.replace(true));
    let _guard = CanonicalRecurrentReferenceGuard(previous);
    f()
}

#[cfg(test)]
fn canonical_recurrent_capture_counts() -> CanonicalRecurrentCaptureCounts {
    CANONICAL_RECURRENT_CAPTURE_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_canonical_recurrent_capture() {
    CANONICAL_RECURRENT_CAPTURE_COUNTS.with(|counts| {
        let mut next = counts.get();
        next.canonical += 1;
        counts.set(next);
    });
}

#[cfg(test)]
fn canonical_recurrent_reference_enabled() -> bool {
    CANONICAL_RECURRENT_REFERENCE_ENABLED.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_reference_recurrent_capture() {
    CANONICAL_RECURRENT_CAPTURE_COUNTS.with(|counts| {
        let mut next = counts.get();
        next.reference += 1;
        counts.set(next);
    });
}

#[cfg(test)]
fn canonical_recurrent_capture_delta(
    before: CanonicalRecurrentCaptureCounts,
    after: CanonicalRecurrentCaptureCounts,
) -> CanonicalRecurrentCaptureCounts {
    CanonicalRecurrentCaptureCounts {
        canonical: after.canonical - before.canonical,
        reference: after.reference - before.reference,
    }
}

#[derive(Clone, Debug)]
struct CompiledEvaluationCapture {
    inference: Option<crate::CapturedInference>,
    portable: Option<Arc<PortableCapturedInferenceRecipe>>,
    capture: Arc<CapturedSchedule>,
    execution_plan: Arc<ExecutionPlanSummary>,
}

impl CompiledEvaluationCapture {
    fn from_inference(inference: crate::CapturedInference) -> Self {
        Self {
            capture: Arc::new(inference.capture().clone()),
            execution_plan: Arc::new(inference.execution_plan().clone()),
            inference: Some(inference),
            portable: None,
        }
    }

    fn from_artifact(
        capture: Arc<CapturedSchedule>,
        portable: Option<PortableCapturedInferenceRecipe>,
    ) -> Result<Self> {
        #[cfg(test)]
        program_artifact::record_evaluation_execution_plan();
        let execution_plan = ExecutionPlanSummary::from_capture(capture.as_ref(), true)
            .map_err(|error| training(format!("compiled evaluation artifact summary: {error}")))?;
        if let Some(portable) = &portable
            && portable.capture_bytes() != capture.to_bytes().map_err(replay_error)?
        {
            return Err(training(
                "compiled evaluation artifact Metal recipe capture differs",
            ));
        }
        Ok(Self {
            inference: None,
            portable: portable.map(Arc::new),
            capture,
            execution_plan: Arc::new(execution_plan),
        })
    }

    fn capture(&self) -> &CapturedSchedule {
        self.capture.as_ref()
    }

    fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.execution_plan.as_ref()
    }

    fn inference(
        &self,
        resident_bindings: BTreeMap<String, TensorData>,
    ) -> Result<crate::CapturedInference> {
        if let Some(inference) = &self.inference {
            return Ok(inference.clone());
        }
        self.portable
            .as_ref()
            .ok_or_else(|| {
                training("compiled AdamW program artifacts currently prepare on CPU only")
            })?
            .instantiate(resident_bindings)
            .map_err(captured_inference_error)
    }

    fn portable_recipe(&self) -> Result<PortableCapturedInferenceRecipe> {
        if let Some(recipe) = &self.portable {
            return Ok(recipe.as_ref().clone());
        }
        self.inference
            .as_ref()
            .ok_or_else(|| training("compiled evaluation capture is absent"))?
            .portable_recipe(PortableInferenceHostPolicy::FixedGathers)
            .map_err(captured_inference_error)
    }
}

#[derive(Clone)]
struct CpuCompiledEvaluation {
    plan: CompiledEvaluationPlan,
}

#[derive(Clone, Copy)]
enum CompiledStepOutputSelection {
    All,
    CommitOnly,
}

impl CompiledStepOutputSelection {
    const fn includes_named_outputs(self) -> bool {
        matches!(self, Self::All)
    }
}

struct CompiledStepReplayRequest {
    inputs: BTreeMap<String, TensorData>,
    learning_rate: Option<TensorData>,
    non_finite_policy: CpuNonFinitePolicy,
    output_selection: CompiledStepOutputSelection,
    injected_failure: Option<u64>,
}

struct AdmittedTrainingStep {
    inputs: BTreeMap<String, TensorData>,
    learning_rate: Option<TensorData>,
    transaction: TrainingStepTransaction,
}

struct TrainingStepTransaction {
    next_step: u64,
    non_finite_policy: CpuNonFinitePolicy,
    output_selection: CompiledStepOutputSelection,
    injected_failure: Option<u64>,
}

struct AuthenticatedTrainingStep {
    transaction: TrainingStepTransaction,
    selected_requested: Option<Vec<u64>>,
    named_output_count: usize,
    observations: Option<CompiledTrainingObservationSchema>,
    include_observations: bool,
    validate_commit_observations: bool,
}

struct CompletedTrainingStep {
    next_step: u64,
    result: CompiledTrainingStepResult,
}

impl AdmittedTrainingStep {
    fn into_main_bindings(mut self) -> (BTreeMap<String, TensorData>, TrainingStepTransaction) {
        if let Some(learning_rate) = self.learning_rate {
            self.inputs
                .insert(LEARNING_RATE_INPUT.to_string(), learning_rate);
        }
        (self.inputs, self.transaction)
    }

    fn into_accumulation_bindings(self) -> (BTreeMap<String, TensorData>, TrainingStepTransaction) {
        (self.inputs, self.transaction)
    }
}

impl TrainingStepTransaction {
    fn authenticate_outputs(
        self,
        capture: &CapturedMixedSchedule,
        outputs: &CompiledTrainingPhaseOutputSchema,
        include_observations: bool,
        validate_commit_observations: bool,
    ) -> Result<AuthenticatedTrainingStep> {
        let expected = outputs.selected_len(true, include_observations)?;
        if capture.schedule.requested.len() != expected {
            return Err(training(
                "compiled requested output layout does not match its authenticated capture",
            ));
        }
        let selected_requested = if self.output_selection.includes_named_outputs() {
            None
        } else {
            let mut selected =
                Vec::with_capacity(outputs.selected_len(false, include_observations)?);
            selected.push(capture.schedule.requested[0]);
            selected.extend(
                capture
                    .schedule
                    .requested
                    .iter()
                    .skip(1 + outputs.named_outputs.len())
                    .copied(),
            );
            Some(selected)
        };
        Ok(AuthenticatedTrainingStep {
            transaction: self,
            selected_requested,
            named_output_count: outputs.named_outputs.len(),
            observations: include_observations.then(|| outputs.observations.clone()),
            include_observations,
            validate_commit_observations,
        })
    }
}

impl AuthenticatedTrainingStep {
    fn next_step(&self) -> u64 {
        self.transaction.next_step
    }

    fn selected_requested(&self) -> Option<&[u64]> {
        self.selected_requested.as_deref()
    }

    fn injected_failure(&self) -> Option<u64> {
        self.transaction.injected_failure
    }

    fn validate_transition<'a>(
        &self,
        values: &[TensorData],
        successors: impl IntoIterator<Item = &'a TensorData>,
    ) -> std::result::Result<(), String> {
        validate_staged_transition(values, successors, self.transaction.non_finite_policy, true)?;
        if let Some(observations) = &self.observations {
            validate_staged_observations(
                values,
                1 + usize::from(self.transaction.output_selection.includes_named_outputs())
                    * self.named_output_count,
                observations,
                self.validate_commit_observations,
                self.transaction.non_finite_policy,
            )?;
        }
        Ok(())
    }

    fn complete(
        self,
        values: Vec<TensorData>,
        outputs: &CompiledTrainingPhaseOutputSchema,
        capture_identity: u64,
    ) -> CompletedTrainingStep {
        debug_assert_eq!(
            values.len(),
            outputs
                .selected_len(
                    self.transaction.output_selection.includes_named_outputs(),
                    self.include_observations,
                )
                .expect("compiled output schema was authenticated before replay")
        );
        let next_step = self.transaction.next_step;
        let outputs = outputs.take(
            values,
            self.transaction.output_selection,
            self.include_observations,
        );
        CompletedTrainingStep {
            next_step,
            result: CompiledTrainingStepResult {
                loss: outputs.loss,
                loss_aggregation_weight: 1,
                outputs: outputs.named_outputs,
                step: next_step,
                capture_identity,
                observations: outputs.observations,
            },
        }
    }
}

struct PendingAdamWStep {
    request: CompiledStepReplayRequest,
    next_progress: CompiledTrainingWindowProgress,
    loss_weight: u64,
}

/// One static CPU training program with runtime-owned recurrent state.
struct CpuCompiledTrainingProgram {
    capture: Arc<CapturedMixedSchedule>,
    recurrent_capture: Arc<CompiledRecurrentCapture>,
    runtime: EffectRuntime,
    cursor: MixedReplayCursor,
    inputs: BTreeMap<String, (Shape, DType)>,
    phase_outputs: CompiledTrainingPhaseOutputSchema,
    parameter_buffers: BTreeMap<String, u64>,
    optimizer_buffers: BTreeMap<RecurrentStateKey, u64>,
    workload_buffers: BTreeMap<RecurrentStateKey, u64>,
    state_input_buffers: BTreeMap<String, u64>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    recurrent_store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
    frozen_parameter_nodes: BTreeSet<NodeId>,
    step: u64,
    accumulation: Option<CompiledTrainingSiblingPlan>,
}

struct CpuAuxiliaryReplay {
    cursor: ProjectedRecurrentCursor,
    provided: BTreeMap<String, TensorData>,
}

/// Gives one public observation a distinct scheduled owner while preserving
/// its exact descriptor and raw value. Mixed capture rejects duplicate public
/// request identities even when two outputs intentionally report one value.
fn materialize_compiled_output_alias(graph: &mut Graph, node: NodeId) -> Result<NodeId> {
    let shape = graph.shape(node)?.clone();
    let dtype = graph.dtype(node)?;
    checked_descriptor(&shape, dtype)?;
    Ok(graph.push(crate::Op::Contiguous { input: node }, shape, dtype))
}

/// Gives only terminal public aliases a concrete schedule owner before mixed
/// capture. RGSM stays owner-only, while ordinary already-materialized losses
/// and named outputs retain their existing graph and capture identities.
fn materialize_compiled_public_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
) -> Result<Vec<NodeId>> {
    let aliases = compiled_requested_aliases(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                graph.contiguous(*node)
            } else {
                Ok(*node)
            }
        })
        .collect()
}

/// Gives ownerless public values and values that coincide with recurrent
/// inputs or successors distinct storage. Mixed capture requires every public
/// request to name a scheduled owner and deliberately rejects shared recurrent
/// node identity even when the value is otherwise already materialized.
fn materialize_compiled_recurrent_public_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
    state_links: &[InferenceStateLink],
) -> Result<Vec<NodeId>> {
    let requested = materialize_compiled_public_aliases(graph, requested)?;
    let unowned = compiled_unowned_requests(graph, &requested)?;
    let state_nodes = state_links
        .iter()
        .flat_map(|link| [link.input(), link.output()])
        .collect::<BTreeSet<_>>();
    requested
        .into_iter()
        .map(|node| {
            if state_nodes.contains(&node) || unowned.contains(&node) {
                let shape = graph.shape(node)?.clone();
                let dtype = graph.dtype(node)?;
                checked_descriptor(&shape, dtype)?;
                Ok(graph.push(crate::Op::Contiguous { input: node }, shape, dtype))
            } else {
                Ok(node)
            }
        })
        .collect()
}

fn compiled_requested_aliases(graph: &Graph, requested: &[NodeId]) -> Result<BTreeSet<NodeId>> {
    Ok(schedule_many(graph, requested)
        .map_err(schedule_error)?
        .requested_passthroughs
        .iter()
        .map(|alias| alias.requested)
        .collect())
}

fn compiled_unowned_requests(graph: &Graph, requested: &[NodeId]) -> Result<BTreeSet<NodeId>> {
    let preview = schedule_many(graph, requested).map_err(schedule_error)?;
    let owners = preview
        .items
        .iter()
        .flat_map(|item| item.outputs.iter())
        .map(|output| output.id)
        .collect::<BTreeSet<_>>();
    Ok(requested
        .iter()
        .copied()
        .filter(|node| !owners.contains(&(node.index() as u64)))
        .collect())
}

/// Recurrent outputs must name storage produced by the captured transition,
/// even when their value is a constant or an existing buffer alias. Insert an
/// explicit copy only for those passthroughs; ordinary computed successors
/// retain their original identity and schedule.
fn materialize_compiled_state_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
) -> Result<Vec<NodeId>> {
    let aliases = compiled_unowned_requests(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                let shape = graph.shape(*node)?.clone();
                let dtype = graph.dtype(*node)?;
                checked_descriptor(&shape, dtype)?;
                Ok(graph.push(crate::Op::Contiguous { input: *node }, shape, dtype))
            } else {
                Ok(*node)
            }
        })
        .collect()
}

/// Builds an exact zero that retains the old recurrent value as an
/// authenticated dependency. Unlike subtraction, this remains zero for
/// non-finite floating-point state.
fn state_dependent_zero(graph: &mut Graph, input: NodeId) -> Result<NodeId> {
    let shape = graph.shape(input)?.clone();
    let dtype = graph.dtype(input)?;
    let zero_value = if dtype == DType::F32 {
        Scalar::F(0.0)
    } else if dtype == DType::U64 {
        Scalar::U(0)
    } else {
        return Err(training(
            "compiled recurrent zero state dtype is unsupported",
        ));
    };
    let zero = graph.lazy_full_with_dtype(shape, zero_value, dtype)?;
    let false_condition = if dtype == DType::F32 {
        // F32 addition by +0 deliberately remains a graph operation because
        // it changes -0 to +0. Every ordered input compares equal to the
        // shifted value, while NaN makes ordered Lt false.
        let shifted = graph.add(input, zero)?;
        graph.compare(CompareOp::Lt, input, shifted)?
    } else if dtype == DType::U64 {
        // U64 subtraction is defined modulo 2^64. Casting its exact zero
        // avoids a native C self-comparison rejected by Apple Clang.
        let difference = graph.sub(input, input)?;
        graph.cast(difference, DType::Bool)?
    } else {
        unreachable!("state-dependent zero dtype was preflighted")
    };
    graph.select(false_condition, input, zero)
}

fn resolve_recurrent_store_groups(
    groups: &[RecurrentStoreGroupSpec],
    updates: &BTreeMap<RecurrentStateKey, NodeId>,
    state_buffers: &BTreeMap<RecurrentStateKey, u64>,
) -> Result<Vec<crate::engine::RecurrentStoreGroupManifest>> {
    groups
        .iter()
        .map(|group| {
            let members = group
                .members
                .iter()
                .map(|key| {
                    let output = updates
                        .get(key)
                        .ok_or_else(|| training("compiled recurrent store successor is absent"))?;
                    let state_buffer = state_buffers
                        .get(key)
                        .ok_or_else(|| training("compiled recurrent store state is absent"))?;
                    Ok(crate::engine::RecurrentStoreGroupMember {
                        output: output.index() as u64,
                        state_buffer: *state_buffer,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(crate::engine::RecurrentStoreGroupManifest { members })
        })
        .collect()
}

struct CompiledTrainingPhaseCapture {
    capture: CapturedMixedSchedule,
    recurrent_capture: CompiledRecurrentCapture,
    state_buffers: BTreeMap<RecurrentStateKey, u64>,
    recurrent_store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
}

struct CompiledTrainingPhaseInput<'a> {
    specs: &'a [StateSpec],
    state_nodes: &'a BTreeMap<RecurrentStateKey, NodeId>,
    state_values: &'a [(u64, TensorData)],
    state_by_input: &'a BTreeMap<NodeId, BufferState>,
    recurrent_store_groups: &'a [RecurrentStoreGroupSpec],
    updates: &'a BTreeMap<RecurrentStateKey, NodeId>,
    public_requested: &'a [NodeId],
    external_input_names: &'a [String],
    materialize_state_passthroughs: bool,
}

fn capture_training_phase(
    graph: &mut Graph,
    input: CompiledTrainingPhaseInput<'_>,
) -> Result<CompiledTrainingPhaseCapture> {
    let CompiledTrainingPhaseInput {
        specs,
        state_nodes,
        state_values,
        state_by_input,
        recurrent_store_groups,
        updates,
        public_requested,
        external_input_names,
        materialize_state_passthroughs,
    } = input;
    if updates.len() != specs.len() || specs.iter().any(|spec| !updates.contains_key(&spec.key)) {
        return Err(training("compiled optimizer successor set mismatch"));
    }
    let ordered_updates = specs
        .iter()
        .map(|spec| updates[&spec.key])
        .collect::<Vec<_>>();
    let ordered_updates = if materialize_state_passthroughs {
        materialize_compiled_state_aliases(graph, &ordered_updates)?
    } else {
        ordered_updates
    };
    let updates = specs
        .iter()
        .zip(ordered_updates)
        .map(|(spec, update)| (spec.key.clone(), update))
        .collect::<BTreeMap<_, _>>();
    let state_links = specs
        .iter()
        .map(|spec| InferenceStateLink::new(state_nodes[&spec.key], updates[&spec.key]))
        .collect::<Vec<_>>();
    let public_requested =
        materialize_compiled_recurrent_public_aliases(graph, public_requested, &state_links)?;
    let initial_state = specs
        .iter()
        .map(|spec| (spec.input_name.clone(), spec.value.clone()))
        .collect();
    let public_output_count = public_requested.len();
    let mut requested = Vec::with_capacity(public_output_count + updates.len());
    requested.extend(public_requested.iter().copied());
    for spec in specs {
        requested.push(updates[&spec.key]);
    }
    for node in &requested {
        checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
    }
    let pure = schedule_many(graph, &requested).map_err(schedule_error)?;
    if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
        return Err(training(format!(
            "compiled pure prefix has an unsupported boundary at node {}",
            item.node.index()
        )));
    }
    let state_buffers = specs
        .iter()
        .zip(state_values)
        .map(|(spec, (buffer, _))| (spec.key.clone(), *buffer))
        .collect::<BTreeMap<_, _>>();
    let recurrent_store_groups =
        resolve_recurrent_store_groups(recurrent_store_groups, &updates, &state_buffers)?;
    let mut captured = CapturedSchedule::capture(graph, &pure, &requested[..public_output_count])
        .map_err(replay_error)?;
    if captured.requested.len() != public_output_count {
        return Err(training("compiled capture output count mismatch"));
    }

    let state_bindings = collect_state_bindings(&pure, state_by_input)?;
    let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
    let mut effects = EffectGraph::default();
    let mut effect_bindings = Vec::with_capacity(updates.len());
    for (ordinal, spec) in specs.iter().enumerate() {
        let next = updates[&spec.key];
        if next.index() as u64 >= STATE_BUFFER_BASE {
            return Err(training(
                "graph node identity overlaps persistent state namespace",
            ));
        }
        let buffer = state_values[ordinal].0;
        let destination = effects
            .insert(buffer, spec.value.clone())
            .map_err(effect_error)?;
        let source = effects
            .insert(
                next.index() as u64,
                TensorData::zeros_with_dtype(spec.value.shape().clone(), spec.value.dtype())?,
            )
            .map_err(effect_error)?;
        effects
            .assign(&destination, &source)
            .map_err(effect_error)?;
        let effect_index = u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?;
        effect_bindings.push(value_binding(&pure, next, effect_index)?);
    }
    let mixed = combine_mixed_schedules(
        pure,
        schedule_effects(&effects).map_err(schedule_error)?,
        effect_bindings,
    )
    .map_err(schedule_error)?;
    captured.items = mixed.items.clone();
    let states = effect_states(&effects)?;
    let capture =
        CapturedMixedSchedule::from_parts(captured, &mixed, states).map_err(replay_error)?;
    validate_external_binding_ownership(&capture, external_input_names.iter())?;
    let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
        graph,
        &capture,
        &public_requested,
        &state_links,
        initial_state,
    )?;
    Ok(CompiledTrainingPhaseCapture {
        capture,
        recurrent_capture,
        state_buffers,
        recurrent_store_groups,
    })
}

impl CompiledTrainingPlan {
    /// Compiles one exact static training program.
    ///
    /// `build` receives the declared external inputs and detached parameter
    /// graph inputs. It returns one scalar F32 loss and deterministically named
    /// detached outputs. All parameter gradients are constructed by exactly
    /// one [`Graph::gradient_default`] traversal.
    fn compile<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_observed(optimizer, parameters, build).map(|(plan, _)| plan)
    }

    fn compile_observed<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<(Self, CompiledTrainingCompileObservation)>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_with_workload_observed(
            optimizer,
            parameters,
            None,
            |graph, inputs, parameters, _| {
                let (loss, outputs) = build(graph, inputs, parameters)?;
                Ok((loss, outputs, None, None))
            },
        )
    }

    fn compile_with_token_weight_observed<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<(Self, CompiledTrainingCompileObservation)>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>, NodeId)>,
    {
        Self::compile_with_workload_observed(
            optimizer,
            parameters,
            None,
            |graph, inputs, parameters, _| {
                let (loss, outputs, token_weight) = build(graph, inputs, parameters)?;
                Ok((loss, outputs, None, Some(token_weight)))
            },
        )
    }

    fn compile_with_workload_observed<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        workload: Option<StateSpec>,
        build: F,
    ) -> Result<(Self, CompiledTrainingCompileObservation)>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
            Option<NodeId>,
        ) -> Result<(
            NodeId,
            BTreeMap<String, NodeId>,
            Option<NodeId>,
            Option<NodeId>,
        )>,
    {
        let parameters = canonical_parameters(parameters)?;
        if parameters.is_empty() {
            return Err(training(format!(
                "compiled {} needs at least one parameter",
                optimizer.name()
            )));
        }
        if parameters
            .keys()
            .any(|name| optimizer.inputs().contains_key(name))
        {
            return Err(training(
                "compiled parameter and input names must be globally unique",
            ));
        }

        let mut graph = Graph::new();
        let inputs = optimizer
            .inputs()
            .iter()
            .map(|(name, (shape, dtype))| {
                (
                    name.clone(),
                    graph.input_dtype_requires_grad(name.clone(), shape.clone(), *dtype, false),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let learning_rate = graph.input_dtype_requires_grad(
            LEARNING_RATE_INPUT,
            Shape::from([]),
            DType::F32,
            false,
        );

        let mut specs = optimizer.state_specs(&parameters)?;
        let optimizer_spec_count = specs.len();
        if let Some(workload) = workload {
            if specs.iter().any(|spec| spec.key == workload.key) {
                return Err(training("compiled workload state key repeats"));
            }
            specs.push(workload);
        }
        let mut parameter_nodes = BTreeMap::new();
        let mut state_nodes = BTreeMap::new();
        let mut state_values = Vec::with_capacity(specs.len());
        let mut state_by_input = BTreeMap::new();
        let mut parameter_buffers = BTreeMap::new();
        let mut optimizer_buffers = BTreeMap::new();
        let mut workload_buffers = BTreeMap::new();
        let mut state_input_buffers = BTreeMap::new();
        let mut state_input_keys = BTreeMap::new();
        let optimizer_spec_count_u64 = u64::try_from(optimizer_spec_count)
            .map_err(|_| training("optimizer state count overflow"))?;
        for (ordinal, spec) in specs.iter().enumerate() {
            let ordinal = u64::try_from(ordinal).map_err(|_| training("parameter overflow"))?;
            let parameter_buffer = STATE_BUFFER_BASE
                .checked_add(ordinal)
                .ok_or_else(|| training("parameter buffer overflow"))?;
            let node = graph.input_dtype_requires_grad(
                spec.input_name.clone(),
                spec.value.shape().clone(),
                spec.value.dtype(),
                spec.requires_grad,
            );
            let state = state_for(parameter_buffer, &spec.value)?;
            state_nodes.insert(spec.key.clone(), node);
            state_by_input.insert(node, state);
            state_values.push((parameter_buffer, spec.value.clone()));
            state_input_buffers.insert(spec.input_name.clone(), parameter_buffer);
            state_input_keys.insert(spec.input_name.clone(), spec.key.clone());
            if let Some(name) = spec.key.parameter_name() {
                parameter_nodes.insert(name.to_string(), node);
                parameter_buffers.insert(name.to_string(), parameter_buffer);
            } else if ordinal < optimizer_spec_count_u64 {
                optimizer_buffers.insert(spec.key.clone(), parameter_buffer);
            } else {
                workload_buffers.insert(spec.key.clone(), parameter_buffer);
            }
        }
        if parameter_nodes.len() != parameters.len() {
            return Err(training("compiled optimizer omitted parameter state"));
        }

        let workload_node = specs
            .get(optimizer_spec_count)
            .map(|spec| state_nodes[&spec.key]);
        let objective_started = Instant::now();
        let (loss, outputs, workload_successor, token_weight) =
            build(&mut graph, &inputs, &parameter_nodes, workload_node)?;
        let objective_forward = CompiledTrainingCompilePhaseObservation::graph(
            objective_started.elapsed(),
            graph.node_count(),
        );
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            optimizer.inputs().keys().chain(parameters.keys()),
        )?;

        let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
        let autograd_started = Instant::now();
        let gradients = optimizer.gradients(&mut graph, loss, &targets)?;
        let autograd = CompiledTrainingCompilePhaseObservation::graph(
            autograd_started.elapsed(),
            graph.node_count(),
        );
        if gradients.len() != targets.len() {
            return Err(training("compiled gradient target count mismatch"));
        }
        let gradients = parameter_nodes
            .keys()
            .cloned()
            .zip(gradients)
            .collect::<BTreeMap<_, _>>();
        let optimizer_lowering_started = Instant::now();
        let CompiledOptimizerLowering {
            mut updates,
            mut sibling_updates,
            recurrent_store_groups,
            observations,
        } = optimizer.lower_updates(
            &mut graph,
            CompiledOptimizerLoweringContext {
                loss,
                learning_rate,
                token_weight,
                inputs: &inputs,
                parameters: &parameter_nodes,
                gradients: &gradients,
                states: &state_nodes,
            },
        )?;
        match (specs.get(optimizer_spec_count), workload_successor) {
            (Some(spec), Some(successor)) => {
                updates.insert(spec.key.clone(), successor);
                if let Some(sibling_updates) = &mut sibling_updates {
                    sibling_updates.insert(spec.key.clone(), successor);
                }
            }
            (None, None) => {}
            _ => return Err(training("compiled workload successor set mismatch")),
        }
        let observation_schema =
            CompiledTrainingObservationSchema::from_nodes(&graph, &observations)?;
        let optimizer_lowering = CompiledTrainingCompilePhaseObservation::graph(
            optimizer_lowering_started.elapsed(),
            graph.node_count(),
        );
        let main_public_requested = std::iter::once(loss)
            .chain(outputs.values().copied())
            .chain(observations.iter().map(|observation| observation.node))
            .collect::<Vec<_>>();
        let external_input_names = optimizer.inputs().keys().cloned().collect::<Vec<_>>();
        let main_started = Instant::now();
        let main = capture_training_phase(
            &mut graph,
            CompiledTrainingPhaseInput {
                specs: &specs,
                state_nodes: &state_nodes,
                state_values: &state_values,
                state_by_input: &state_by_input,
                recurrent_store_groups: &recurrent_store_groups,
                updates: &updates,
                public_requested: &main_public_requested,
                external_input_names: &external_input_names,
                materialize_state_passthroughs: false,
            },
        )?;
        let main_capture = CompiledTrainingCompilePhaseObservation::schedule(
            main_started.elapsed(),
            main.recurrent_capture.execution_plan().schedule_item_count,
        );
        let (accumulation, accumulation_capture) = if let Some(updates) = sibling_updates {
            let accumulation_started = Instant::now();
            let public_requested = std::iter::once(loss)
                .chain(outputs.values().copied())
                .collect::<Vec<_>>();
            let phase = capture_training_phase(
                &mut graph,
                CompiledTrainingPhaseInput {
                    specs: &specs,
                    state_nodes: &state_nodes,
                    state_values: &state_values,
                    state_by_input: &state_by_input,
                    recurrent_store_groups: &recurrent_store_groups,
                    updates: &updates,
                    public_requested: &public_requested,
                    external_input_names: &external_input_names,
                    materialize_state_passthroughs: true,
                },
            )?;
            let cursor_projection = PreparedRecurrentCursorProjection::prepare(
                &main.capture,
                &phase.capture,
                phase.state_buffers.values().copied(),
            )
            .map_err(replay_error)?;
            let capture_identity = cursor_projection.target_capture_identity();
            let schedule_item_count = phase.recurrent_capture.execution_plan().schedule_item_count;
            let plan = CompiledTrainingSiblingPlan {
                phase: CompiledRecurrentPhasePlan {
                    capture: Arc::new(phase.capture),
                    recurrent_capture: phase.recurrent_capture,
                    state_buffers: phase.state_buffers,
                    cursor_projection: Arc::new(cursor_projection),
                    capture_identity,
                    admission: CompiledRecurrentPhaseAdmission::RetainUnchanged,
                },
            };
            (
                Some(plan),
                Some(CompiledTrainingCompilePhaseObservation::schedule(
                    accumulation_started.elapsed(),
                    schedule_item_count,
                )),
            )
        } else {
            (None, None)
        };
        if let Some(accumulation) = &accumulation {
            let main_identity = main
                .capture
                .initial_recurrent_cursor()
                .map_err(replay_error)?
                .capture_identity();
            if accumulation.phase().capture_identity == main_identity {
                return Err(training(
                    "compiled accumulation capture identity is not distinct",
                ));
            }
        }

        let phase_outputs = CompiledTrainingPhaseOutputSchema {
            loss: CompiledTrainingLossOutput::ScalarF32,
            named_outputs: outputs.keys().cloned().collect(),
            observations: observation_schema,
        };
        let observation = CompiledTrainingCompileObservation::new(
            objective_forward,
            autograd,
            optimizer_lowering,
            main_capture,
            accumulation_capture,
        );
        Ok((
            Self {
                capture: Arc::new(main.capture),
                recurrent_capture: Arc::new(main.recurrent_capture),
                inputs: optimizer.inputs().clone(),
                phase_outputs,
                parameter_buffers,
                optimizer_buffers,
                workload_buffers,
                state_input_buffers,
                state_input_keys,
                state_values: specs
                    .iter()
                    .map(|spec| (spec.key.clone(), spec.value.clone()))
                    .collect(),
                state_versions: specs.iter().map(|spec| (spec.key.clone(), 0)).collect(),
                recurrent_store_groups: main.recurrent_store_groups,
                frozen_parameter_nodes: BTreeSet::new(),
                step: 0,
                accumulation,
            },
            observation,
        ))
    }

    fn capture_identity(&self) -> Result<u64> {
        Ok(self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .capture_identity())
    }

    fn recurrent_capture(&self) -> Result<CapturedStatefulInference> {
        let initial_state = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                let value = self
                    .state_values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| training("compiled plan state value is absent"))?;
                Ok((input.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.recurrent_capture.stateful(initial_state)
    }

    fn metal_plan(
        &self,
        renderer: MetalRenderer,
        host_token_inputs: &BTreeMap<String, Shape>,
        evaluation: Option<CompiledEvaluationPlan>,
    ) -> Result<MetalCompiledTrainingPlan> {
        let recurrent = self.recurrent_capture()?;
        let recurrent = if self.recurrent_capture.portable.is_some() {
            recurrent
        } else {
            recurrent
                .with_authenticated_training_host_indices(
                    host_token_inputs,
                    &self.frozen_parameter_nodes,
                )
                .map_err(captured_inference_error)?
        };
        let inner = MetalStatefulInferencePlan::new(recurrent.clone(), renderer.clone()).map_err(
            |error| {
                let detail = if matches!(&error, MetalError::Unsupported(_)) {
                    recurrent
                        .capture()
                        .items
                        .iter()
                        .find_map(|item| {
                            renderer.render(&item.kernel).err().map(|item_error| {
                                format!(
                                    " at schedule item {} (node {}, {:?}): {item_error}",
                                    item.id,
                                    item.node.index(),
                                    item.kernel.operation()
                                )
                            })
                        })
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                training(format!("compiled Metal runtime: {error:?}{detail}"))
            },
        )?;
        let evaluation = evaluation
            .map(|evaluation| {
                let parameter_names = evaluation
                    .parameter_inputs
                    .values()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                let resident_bindings = evaluation
                    .parameter_inputs
                    .iter()
                    .map(|(parameter, input)| {
                        let value = self
                            .state_values
                            .get(&RecurrentStateKey::parameter(parameter))
                            .cloned()
                            .ok_or_else(|| {
                                training("compiled evaluation parameter value is absent")
                            })?;
                        Ok((input.clone(), value))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                MetalFixedStateReadPlan::new(
                    evaluation.inference.inference(resident_bindings)?,
                    renderer,
                    &inner,
                    &parameter_names,
                )
                .map(|plan| (plan, evaluation.output_names, evaluation.capture_identity))
                .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledTrainingPlan {
            inner,
            inputs: self.inputs.clone(),
            output_names: self.phase_outputs.named_outputs.clone(),
            state_input_keys: self.state_input_keys.clone(),
            program_identity: self.capture_identity()?,
            evaluation,
        })
    }

    #[cfg(test)]
    fn restore_frontier(
        self,
        step: u64,
        values: BTreeMap<RecurrentStateKey, TensorData>,
    ) -> Result<Self> {
        let versions = values.keys().cloned().map(|key| (key, step)).collect();
        self.restore_frontier_with_versions(step, values, versions)
    }

    fn restore_frontier_with_versions(
        mut self,
        step: u64,
        values: BTreeMap<RecurrentStateKey, TensorData>,
        versions: BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<Self> {
        let expected = self
            .parameter_buffers
            .keys()
            .map(RecurrentStateKey::parameter)
            .chain(self.optimizer_buffers.keys().cloned())
            .chain(self.workload_buffers.keys().cloned())
            .collect::<BTreeSet<_>>();
        if values.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(training("compiled checkpoint state names mismatch"));
        }
        if versions.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(training("compiled checkpoint state versions mismatch"));
        }
        for (key, value) in &values {
            let current = self
                .state_values
                .get(key)
                .ok_or_else(|| training("compiled checkpoint state is absent"))?;
            if value.shape() != current.shape() || value.dtype() != current.dtype() {
                return Err(training("compiled checkpoint state descriptor mismatch"));
            }
            checked_bytes(value)?;
        }
        self.state_values = values;
        self.state_versions = versions;
        self.step = step;
        let frontier = self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .frontier()
            .iter()
            .cloned()
            .map(|mut state| {
                let key = self
                    .state_input_buffers
                    .iter()
                    .find_map(|(input, buffer)| {
                        (*buffer == state.buffer).then(|| self.state_input_keys[input].clone())
                    })
                    .ok_or_else(|| training("compiled checkpoint state buffer is absent"))?;
                state.version = self.state_versions[&key];
                Ok(state)
            })
            .collect::<Result<Vec<_>>>()?;
        MixedReplayCursor::resume(&self.capture, frontier).map_err(replay_error)?;
        Ok(self)
    }

    fn prepare_cpu(&self) -> Result<CpuCompiledTrainingProgram> {
        self.prepare_cpu_with_non_finite_policy(CpuNonFinitePolicy::Propagate)
    }

    fn prepare_cpu_with_non_finite_policy(
        &self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> Result<CpuCompiledTrainingProgram> {
        if non_finite_policy == CpuNonFinitePolicy::RejectTransition {
            validate_finite_tensors(self.state_values.values(), "prepared recurrent state")?;
        }
        let initial_states = self
            .state_input_buffers
            .iter()
            .map(|(input, buffer)| {
                let key = self
                    .state_input_keys
                    .get(input)
                    .ok_or_else(|| training("compiled plan state key is absent"))?;
                let value = self
                    .state_values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| training("compiled plan state value is absent"))?;
                Ok((*buffer, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_states(initial_states)
            .map_err(runtime_error)?;
        let cursor = self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?;
        let mut program = CpuCompiledTrainingProgram {
            capture: self.capture.clone(),
            recurrent_capture: self.recurrent_capture.clone(),
            runtime,
            cursor,
            inputs: self.inputs.clone(),
            phase_outputs: self.phase_outputs.clone(),
            parameter_buffers: self.parameter_buffers.clone(),
            optimizer_buffers: self.optimizer_buffers.clone(),
            workload_buffers: self.workload_buffers.clone(),
            state_input_buffers: self.state_input_buffers.clone(),
            state_input_keys: self.state_input_keys.clone(),
            recurrent_store_groups: self.recurrent_store_groups.clone(),
            frozen_parameter_nodes: self.frozen_parameter_nodes.clone(),
            step: 0,
            accumulation: self.accumulation.clone(),
        };
        if self.step != 0 || self.state_versions.values().any(|version| *version != 0) {
            program.restore_frontier(self.step, &self.state_values, &self.state_versions)?;
        }
        Ok(program)
    }
}

impl CompiledAdamWAuxiliaryPlan {
    fn compile_partial_flush(
        training_plan: &CompiledTrainingPlan,
        config: &CompiledAdamWConfig,
    ) -> Result<Self> {
        let topology = CompiledTrainingWindowTopology::from_config(config);
        if !topology.accumulating() {
            return Err(training(
                "compiled AdamW partial flush requires gradient accumulation",
            ));
        }

        let state_buffers = training_plan
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(
                training_plan
                    .optimizer_buffers
                    .iter()
                    .map(|(key, buffer)| (key.clone(), *buffer)),
            )
            .collect::<BTreeMap<_, _>>();
        let mut graph = Graph::new();
        let learning_rate = graph.input_dtype_requires_grad(
            LEARNING_RATE_INPUT,
            Shape::from([]),
            DType::F32,
            false,
        );
        let mut state_nodes = BTreeMap::new();
        let mut state_by_input = BTreeMap::new();
        let mut specs = Vec::with_capacity(state_buffers.len());
        for (input_name, key) in &training_plan.state_input_keys {
            let Some(buffer) = state_buffers.get(key).copied() else {
                continue;
            };
            let value = training_plan
                .state_values
                .get(key)
                .cloned()
                .ok_or_else(|| training("compiled partial flush state value is absent"))?;
            let node = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                false,
            );
            state_nodes.insert(key.clone(), node);
            state_by_input.insert(node, state_for(buffer, &value)?);
            specs.push((input_name.clone(), key.clone(), value, node, buffer));
        }
        if specs.len() != state_buffers.len() {
            return Err(training("compiled partial flush state schema differs"));
        }

        let parameters = state_nodes
            .iter()
            .filter_map(|(key, node)| key.parameter_name().map(|name| (name.to_owned(), *node)))
            .collect::<BTreeMap<_, _>>();
        if parameters.len() != training_plan.parameter_buffers.len() {
            return Err(training("compiled partial flush parameter schema differs"));
        }
        let accumulation_index_key =
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex);
        let index = state_nodes
            .get(&accumulation_index_key)
            .copied()
            .ok_or_else(|| training("compiled partial flush accumulation index is absent"))?;
        let token_count_key =
            topology
                .retains_token_count()
                .then_some(RecurrentStateKey::adamw_global(
                    AdamWGlobalState::AccumulatedTokenCount,
                ));
        let divisor = match &token_count_key {
            Some(key) => {
                let count = state_nodes.get(key).copied().ok_or_else(|| {
                    training("compiled partial flush accumulated token count is absent")
                })?;
                graph.cast(count, DType::F32)?
            }
            None => graph.cast(index, DType::F32)?,
        };
        let divisor = safe_token_count_divisor(
            &mut graph,
            divisor,
            config.allow_zero_valid_token_microbatches,
        )?;
        let window_loss_report = if config.window_loss_report {
            let numerator_key =
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator);
            let numerator = state_nodes.get(&numerator_key).copied().ok_or_else(|| {
                training("compiled partial flush accumulated loss numerator is absent")
            })?;
            let loss_weight = match &token_count_key {
                Some(key) => state_nodes[key],
                None => index,
            };
            Some((
                numerator_key,
                CompiledAdamWWindowLossNodes {
                    mean_loss: graph.div(numerator, divisor)?,
                    loss_weight,
                },
            ))
        } else {
            None
        };
        let mut gradients = BTreeMap::new();
        for name in parameters.keys() {
            let accumulator_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulator = state_nodes
                .get(&accumulator_key)
                .copied()
                .ok_or_else(|| training("compiled partial flush accumulator is absent"))?;
            gradients.insert(name.clone(), graph.div(accumulator, divisor)?);
        }
        let clipped = clip_gradients_by_global_norm(config, &mut graph, &gradients)?;
        let learning_rate =
            lower_adamw_learning_rate(config, &mut graph, learning_rate, &state_nodes)?;
        let mut updates = lower_adamw_update_candidates(
            config,
            &mut graph,
            learning_rate,
            &parameters,
            &clipped.gradients,
            &state_nodes,
        )?;
        // Token-weighted flush divides by the retained count instead of the
        // accumulation index. Derive the exact reset from the old index so
        // the strict recurrent capture still owns every state input.
        let zero_index = if token_count_key.is_some() {
            graph.sub(index, index)?
        } else {
            graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?
        };
        updates.insert(accumulation_index_key, zero_index);
        if let Some(key) = token_count_key {
            let zero_count = graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?;
            updates.insert(key, zero_count);
        }
        if let Some((key, _)) = &window_loss_report {
            updates.insert(key.clone(), scalar_f32(&mut graph, 0.0)?);
        }
        for name in parameters.keys() {
            let key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulator = state_nodes
                .get(&key)
                .copied()
                .ok_or_else(|| training("compiled partial flush accumulator is absent"))?;
            let zero = state_dependent_zero(&mut graph, accumulator)?;
            updates.insert(key, zero);
        }
        let successor_keys = specs
            .iter()
            .map(|(_, key, ..)| key.clone())
            .collect::<Vec<_>>();
        let successors = successor_keys
            .iter()
            .map(|key| updates[key])
            .collect::<Vec<_>>();
        let successors = materialize_compiled_state_aliases(&mut graph, &successors)?;
        for (key, successor) in successor_keys.into_iter().zip(successors) {
            updates.insert(key, successor);
        }
        if updates.len() != specs.len()
            || specs.iter().any(|(_, key, ..)| !updates.contains_key(key))
        {
            return Err(training("compiled partial flush successor schema differs"));
        }

        let state_links = specs
            .iter()
            .map(|(_, key, _, node, _)| InferenceStateLink::new(*node, updates[key]))
            .collect::<Vec<_>>();
        let initial_state = specs
            .iter()
            .map(|(input, _, value, _, _)| (input.clone(), value.clone()))
            .collect();
        let observations = adamw_observation_nodes(
            clipped.report,
            window_loss_report.as_ref().map(|(_, report)| *report),
        );
        let outputs = CompiledAdamWAuxiliaryOutputSchema::from_nodes(&graph, &observations)?;
        let public_requested = outputs.node_ids(&observations)?;
        let public_requested = materialize_compiled_recurrent_public_aliases(
            &mut graph,
            &public_requested,
            &state_links,
        )?;
        let public_output_count = public_requested.len();
        let mut requested = public_requested.clone();
        requested.extend(specs.iter().map(|(_, key, ..)| updates[key]));
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled partial flush has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured =
            CapturedSchedule::capture(&graph, &pure, &requested[..public_output_count])
                .map_err(replay_error)?;
        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(specs.len());
        for (ordinal, (_, key, value, _, buffer)) in specs.iter().enumerate() {
            let next = updates[key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let destination = effects
                .insert(*buffer, value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(value.shape().clone(), value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            effect_bindings.push(value_binding(
                &pure,
                next,
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?,
            )?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let capture = CapturedMixedSchedule::from_parts(captured, &mixed, effect_states(&effects)?)
            .map_err(replay_error)?;
        validate_external_binding_ownership(&capture, std::iter::empty::<&String>())?;
        let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
            &graph,
            &capture,
            &public_requested,
            &state_links,
            initial_state,
        )?;
        let cursor_projection = PreparedRecurrentCursorProjection::prepare(
            &training_plan.capture,
            &capture,
            state_buffers.values().copied(),
        )
        .map_err(replay_error)?;
        let capture_identity = cursor_projection.target_capture_identity();
        let recurrent_store_groups = resolve_recurrent_store_groups(
            &adamw_recurrent_store_group_specs(parameters.keys(), topology),
            &updates,
            &state_buffers,
        )?;
        Ok(Self {
            phase: CompiledRecurrentPhasePlan {
                capture: Arc::new(capture),
                recurrent_capture,
                state_buffers,
                cursor_projection: Arc::new(cursor_projection),
                capture_identity,
                admission: CompiledRecurrentPhaseAdmission::Replace {
                    store_groups: recurrent_store_groups,
                },
            },
            state_input_keys: specs
                .into_iter()
                .map(|(input, key, ..)| (input, key))
                .collect(),
            outputs,
        })
    }

    fn compile_zero_grad(
        training_plan: &CompiledTrainingPlan,
        topology: CompiledTrainingWindowTopology,
    ) -> Result<Self> {
        if !topology.accumulating() {
            return Err(training(
                "compiled AdamW zero-grad requires gradient accumulation",
            ));
        }
        let state_buffers = training_plan
            .optimizer_buffers
            .iter()
            .filter(|(key, _)| key.is_accumulation_reset_state())
            .map(|(key, buffer)| (key.clone(), *buffer))
            .collect::<BTreeMap<_, _>>();
        if state_buffers.is_empty() {
            return Err(training(
                "compiled AdamW zero-grad requires gradient accumulation",
            ));
        }

        Ok(Self {
            phase: CompiledRecurrentPhasePlan::compile_state_only_reset(
                training_plan,
                state_buffers,
            )?,
            state_input_keys: training_plan
                .state_input_keys
                .iter()
                .filter(|(_, key)| key.is_accumulation_reset_state())
                .map(|(input, key)| (input.clone(), key.clone()))
                .collect(),
            outputs: CompiledAdamWAuxiliaryOutputSchema::from_report_flags(false, false),
        })
    }
}

impl CompiledRecurrentPhasePlan {
    fn compile_state_only_reset(
        training_plan: &CompiledTrainingPlan,
        state_buffers: BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<Self> {
        let mut graph = Graph::new();
        let mut state_by_input = BTreeMap::new();
        let mut specs = Vec::with_capacity(state_buffers.len());
        let mut successors = BTreeMap::new();
        for (input_name, key) in &training_plan.state_input_keys {
            let Some(buffer) = state_buffers.get(key).copied() else {
                continue;
            };
            let value = training_plan
                .state_values
                .get(key)
                .cloned()
                .ok_or_else(|| training("compiled zero-grad state value is absent"))?;
            let input = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                false,
            );
            let successor = state_dependent_zero(&mut graph, input)?;
            state_by_input.insert(input, state_for(buffer, &value)?);
            successors.insert(key.clone(), successor);
            specs.push((input_name.clone(), key.clone(), value, input, buffer));
        }
        if specs.len() != state_buffers.len() {
            return Err(training("compiled zero-grad state schema differs"));
        }

        let successor_keys = specs
            .iter()
            .map(|(_, key, ..)| key.clone())
            .collect::<Vec<_>>();
        let materialized = materialize_compiled_state_aliases(
            &mut graph,
            &successor_keys
                .iter()
                .map(|key| successors[key])
                .collect::<Vec<_>>(),
        )?;
        for (key, successor) in successor_keys.into_iter().zip(materialized) {
            successors.insert(key, successor);
        }
        let state_links = specs
            .iter()
            .map(|(_, key, _, input, _)| InferenceStateLink::new(*input, successors[key]))
            .collect::<Vec<_>>();
        let initial_state = specs
            .iter()
            .map(|(input, _, value, _, _)| (input.clone(), value.clone()))
            .collect();
        let requested = specs
            .iter()
            .map(|(_, key, ..)| successors[key])
            .collect::<Vec<_>>();
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled zero-grad has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured = CapturedSchedule::capture(&graph, &pure, &[]).map_err(replay_error)?;
        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(specs.len());
        for (ordinal, (_, key, value, _, buffer)) in specs.iter().enumerate() {
            let next = successors[key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let destination = effects
                .insert(*buffer, value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(value.shape().clone(), value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            effect_bindings.push(value_binding(
                &pure,
                next,
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?,
            )?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let capture = CapturedMixedSchedule::from_parts(captured, &mixed, effect_states(&effects)?)
            .map_err(replay_error)?;
        validate_external_binding_ownership(&capture, std::iter::empty::<&String>())?;
        let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
            &graph,
            &capture,
            &[],
            &state_links,
            initial_state,
        )?;
        let cursor_projection = PreparedRecurrentCursorProjection::prepare(
            &training_plan.capture,
            &capture,
            state_buffers.values().copied(),
        )
        .map_err(replay_error)?;
        let capture_identity = cursor_projection.target_capture_identity();
        Ok(Self {
            capture: Arc::new(capture),
            recurrent_capture,
            state_buffers,
            cursor_projection: Arc::new(cursor_projection),
            capture_identity,
            admission: CompiledRecurrentPhaseAdmission::Replace {
                store_groups: Vec::new(),
            },
        })
    }
}

impl CompiledAdamWAuxiliaryPlan {
    fn capture_identity(&self) -> u64 {
        self.phase.capture_identity()
    }

    fn with_frontier(mut self, values: &BTreeMap<RecurrentStateKey, TensorData>) -> Result<Self> {
        let initial_state = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                Ok((
                    input.clone(),
                    values
                        .get(key)
                        .cloned()
                        .ok_or_else(|| training("compiled auxiliary frontier is absent"))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        if let Some(stateful) = self.phase.recurrent_capture.stateful.take() {
            self.phase.recurrent_capture.stateful = Some(
                stateful
                    .with_initial_state(initial_state)
                    .map_err(captured_inference_error)?,
            );
        }
        Ok(self)
    }
}

impl CompiledEvaluationPlan {
    fn compile_with_parameter_plan<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
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
        Self::compile_with_parameter_plan_inner(
            module,
            training_plan,
            parameter_plan,
            |module, graph, inputs| {
                let (loss, outputs) = build(module, graph, inputs)?;
                Ok((loss, outputs, None))
            },
        )
    }

    fn compile_graph_with_parameter_plan<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        Self::compile_with_parameter_plan_inner(
            module,
            training_plan,
            parameter_plan,
            |module, graph, inputs| {
                let (objective, outputs) = build(module, graph, inputs)?.into_parts();
                let loss_weight_policy = match objective {
                    CompiledAdamWObjective::Scalar(_) => None,
                    CompiledAdamWObjective::TokenMean(_) => {
                        training_plan.contract.token_weight_policy.clone()
                    }
                };
                let loss = lower_compiled_adamw_objective_for_policy(
                    graph,
                    inputs,
                    objective,
                    training_plan.contract.token_weight_policy.as_ref(),
                    &training_plan.inner.inputs,
                    training_plan.contract.allow_zero_valid_token_microbatches,
                )?;
                Ok((loss, outputs, loss_weight_policy))
            },
        )
    }

    fn compile_graph_with_ignore_index_parameter_plan<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        Self::compile_with_parameter_plan_inner(
            module,
            training_plan,
            parameter_plan,
            |module, graph, inputs| {
                let nodes = lower_ignore_index_nodes_for_policy(
                    training_plan.contract.token_weight_policy.as_ref(),
                    &training_plan.inner.inputs,
                    graph,
                    inputs,
                )?;
                let (objective, outputs) = build(module, graph, inputs, nodes)?.into_parts();
                let loss = lower_compiled_adamw_objective_for_ignore_index_policy(
                    graph,
                    objective,
                    nodes,
                    training_plan.contract.token_weight_policy.as_ref(),
                    &training_plan.inner.inputs,
                    training_plan.contract.allow_zero_valid_token_microbatches,
                )?;
                Ok((
                    loss,
                    outputs,
                    training_plan.contract.token_weight_policy.clone(),
                ))
            },
        )
    }

    fn compile_with_parameter_plan_inner<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(
            NodeId,
            BTreeMap<String, NodeId>,
            Option<CompiledTokenWeightPolicy>,
        )>,
    {
        let mut graph = Graph::new();
        let inputs = training_plan
            .inner
            .inputs
            .iter()
            .map(|(name, (shape, dtype))| {
                (
                    name.clone(),
                    graph.input_dtype_requires_grad(name.clone(), shape.clone(), *dtype, false),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut parameters = BTreeMap::new();
        let mut parameter_inputs = BTreeMap::new();
        let mut residents = BTreeMap::new();
        for init in parameter_plan.initial_parameters()? {
            let key = RecurrentStateKey::parameter(init.name());
            let input_name = training_plan
                .inner
                .state_input_keys
                .iter()
                .find_map(|(input, candidate)| (candidate == &key).then(|| input.clone()))
                .ok_or_else(|| training("compiled evaluation parameter state is absent"))?;
            let value = training_plan
                .inner
                .state_values
                .get(&key)
                .cloned()
                .ok_or_else(|| training("compiled evaluation parameter value is absent"))?;
            let node = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                true,
            );
            parameters.insert(init.name().to_owned(), node);
            parameter_inputs.insert(init.name().to_owned(), input_name.clone());
            residents.insert(input_name, (node, value));
        }
        let mut loss_weight_policy = None;
        let (loss, outputs) = parameter_plan.lower(&mut graph, &parameters, |graph| {
            let (loss, outputs, mask_input) = build(module, graph, &inputs)?;
            loss_weight_policy = mask_input;
            Ok((loss, outputs))
        })?;
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            training_plan
                .inner
                .inputs
                .keys()
                .chain(parameter_inputs.keys()),
        )?;
        let requested = std::iter::once(loss)
            .chain(outputs.values().copied())
            .collect::<Vec<_>>();
        let requested = materialize_compiled_public_aliases(&mut graph, &requested)?;
        let inference =
            crate::CapturedInference::from_graph_residents(&graph, &requested, residents, &[])
                .map_err(captured_inference_error)?
                .with_authenticated_fixed_host_gathers(&training_plan.contract.host_token_inputs)
                .map_err(captured_inference_error)?;
        let transient_names = inference
            .transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        if transient_names
            != training_plan
                .inner
                .inputs
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        {
            return Err(training("compiled evaluation input inventory differs"));
        }
        let capture_identity = inference.capture().identity;
        Ok(Self {
            inference: CompiledEvaluationCapture::from_inference(inference),
            inputs: training_plan.inner.inputs.clone(),
            output_names: outputs.keys().cloned().collect(),
            parameter_inputs,
            loss_weight_policy,
            allow_zero_valid_token_microbatches: training_plan
                .contract
                .allow_zero_valid_token_microbatches,
            capture_identity,
        })
    }

    fn bind(
        &self,
        mut inputs: BTreeMap<String, TensorData>,
        parameters: BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<String, TensorData>> {
        validate_evaluation_inputs(&self.inputs, &inputs)?;
        if parameters.keys().ne(self.parameter_inputs.keys()) {
            return Err(training("compiled evaluation parameter inventory differs"));
        }
        for (name, value) in parameters {
            let input = self
                .parameter_inputs
                .get(&name)
                .ok_or_else(|| training("compiled evaluation parameter is absent"))?;
            inputs.insert(input.clone(), value);
        }
        Ok(inputs)
    }

    fn validate_loss_weight(&self, inputs: &BTreeMap<String, TensorData>) -> Result<u64> {
        validate_evaluation_inputs(&self.inputs, inputs)?;
        validate_token_weight(
            inputs,
            self.loss_weight_policy.as_ref(),
            self.allow_zero_valid_token_microbatches,
        )
    }

    fn evaluate(
        &self,
        inputs: BTreeMap<String, TensorData>,
        parameters: BTreeMap<String, TensorData>,
    ) -> Result<CompiledEvaluationResult> {
        let loss_weight = self.validate_loss_weight(&inputs)?;
        let inputs = self.bind(inputs, parameters)?;
        let values = self
            .inference
            .capture()
            .replay(&inputs)
            .map_err(replay_error)?;
        evaluation_result(
            values,
            &self.output_names,
            loss_weight,
            self.capture_identity,
        )
    }

    fn preflight_native(
        &self,
        parameters: BTreeMap<String, TensorData>,
        parameter_buffers: &BTreeMap<String, u64>,
    ) -> Result<NativeCpuEvaluationPreparation> {
        let started = Instant::now();
        if parameter_buffers.keys().ne(self.parameter_inputs.keys()) {
            return Err(training(
                "compiled native CPU evaluation parameter buffer inventory differs",
            ));
        }
        let capture = self.inference.capture();
        let parameter_inputs = self
            .parameter_inputs
            .iter()
            .map(|(parameter, input_name)| {
                let value = parameters.get(parameter).ok_or_else(|| {
                    training("compiled native CPU evaluation parameter value is absent")
                })?;
                let input = capture
                    .inputs
                    .iter()
                    .find(|input| input.name == *input_name)
                    .ok_or_else(|| {
                        training("compiled native CPU evaluation parameter input is absent")
                    })?;
                let bytes = value
                    .len()
                    .checked_mul(value.dtype().itemsize())
                    .ok_or_else(|| {
                        training("compiled native CPU evaluation parameter bytes overflow")
                    })?;
                if value.shape() != &input.desc.shape
                    || value.dtype() != input.desc.dtype
                    || bytes != input.desc.bytes
                {
                    return Err(training(
                        "compiled native CPU evaluation parameter descriptor mismatch",
                    ));
                }
                Ok(PreparedNativeEvaluationParameterInput {
                    parameter: parameter.clone(),
                    input: input_name.clone(),
                    buffer: parameter_buffers[parameter],
                    shape: value.shape().clone(),
                    dtype: value.dtype(),
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let inputs = self.bind(zero_inputs(&self.inputs)?, parameters)?;
        Ok(NativeCpuEvaluationPreparation {
            inputs: Some(inputs),
            parameter_inputs,
            residual_wall_time: started.elapsed(),
        })
    }

    fn finish_native(
        &self,
        preparation: NativeCpuEvaluationPreparation,
        parameter_buffers: &BTreeMap<String, u64>,
        plan: PlannedNativeItems,
    ) -> Result<PreparedNativeCpuEvaluation> {
        let capture = self.inference.capture();
        let module_preparation = plan.module_preparation();
        let work = NativeCpuPreparationWork::from_module(module_preparation);
        let execution_plan = ExecutionPlanSummary::from_capture(capture, true)
            .map_err(|error| training(format!("compiled native CPU summary: {error}")))?;
        let wall_time =
            native_preparation_wall_time(module_preparation, preparation.residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: self.capture_identity,
            native_identity: native_cpu_identity(
                self.capture_identity,
                plan.vectorized(),
                capture.items.iter().map(|item| item.cache_key),
            ),
            vectorized: plan.vectorized(),
            native_item_count: plan.item_count(),
            cache_hit_count: plan.cache_hit_count(),
            cache_miss_count: plan.cache_miss_count(),
            work,
            phases: NativeCpuPreparationPhases::from_module(module_preparation, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                plan.dispatch_segmentation(),
            )?,
            execution_plan,
            wall_time,
        };
        let prepared = PreparedNativeCpuEvaluation {
            report,
            plan,
            parameter_inputs: preparation.parameter_inputs,
        };
        prepared.validate(self.capture_identity, capture, parameter_buffers)?;
        Ok(prepared)
    }

    fn preflight_native_borrowed(
        &self,
        inputs: &BTreeMap<String, TensorData>,
        frontier: &[BufferState],
        prepared: &PreparedNativeCpuEvaluation,
        parameter_buffers: &BTreeMap<String, u64>,
    ) -> Result<(u64, Vec<BufferState>)> {
        let loss_weight = self.validate_loss_weight(inputs)?;
        let capture = self.inference.capture();
        prepared.validate(self.capture_identity, capture, parameter_buffers)?;
        let parameter_inputs = prepared
            .parameter_inputs
            .iter()
            .map(|binding| binding.input.as_str())
            .collect::<BTreeSet<_>>();
        for input in &capture.inputs {
            if parameter_inputs.contains(input.name.as_str()) {
                continue;
            }
            let value = inputs
                .get(&input.name)
                .ok_or_else(|| training("compiled evaluation input is absent"))?;
            crate::engine::validate_input_value(capture, input, value).map_err(replay_error)?;
        }
        Ok((loss_weight, prepared.active_parameter_states(frontier)?))
    }

    fn evaluate_native_borrowed(
        &self,
        inputs: &BTreeMap<String, TensorData>,
        reads: &[crate::host_buffer::HostBufferRead<'_>],
        loss_weight: u64,
        executor: &CapturedReplayExecutor,
        prepared: &mut PreparedNativeCpuEvaluation,
    ) -> Result<(CompiledEvaluationResult, NativeCpuRunReport)> {
        if reads.len() != prepared.parameter_inputs.len() {
            return Err(training(
                "compiled native CPU evaluation active parameter cardinality mismatch",
            ));
        }
        let mut recurrent = BTreeMap::new();
        for (binding, read) in prepared.parameter_inputs.iter().zip(reads) {
            if read.ordinal() != recurrent.len() || read.buffer_id() != binding.buffer {
                return Err(training(
                    "compiled native CPU evaluation active parameter mapping mismatch",
                ));
            }
            if recurrent
                .insert(binding.input.clone(), read.tensor())
                .is_some()
            {
                return Err(training(
                    "compiled native CPU evaluation active parameter input is duplicated",
                ));
            }
        }
        let started = Instant::now();
        let capture = self.inference.capture();
        let executor_started = Instant::now();
        let executed = executor.execute_planned_native_items_with_recurrent_inputs(
            capture,
            inputs,
            &recurrent,
            &mut prepared.plan,
        );
        let executor_wall_time = executor_started.elapsed();
        let (values, traffic) = executed.map_err(replay_error)?;
        let outputs = values.requested(&capture.requested).map_err(replay_error)?;
        let schedule_cache_keys = prepared.plan.schedule_cache_keys().to_vec();
        let wall_time = started.elapsed();
        let report = NativeCpuRunReport {
            capture_identity: self.capture_identity,
            native_identity: prepared.report.native_identity,
            vectorized: prepared.plan.vectorized(),
            successful_invocation: 0,
            native_item_count: prepared.plan.item_count(),
            executed_native_item_count: traffic.executed_native_item_count,
            module_dispatch_count: traffic.module_dispatch_count,
            module_dispatched_native_item_count: traffic.module_dispatched_native_item_count,
            skipped_output_clear_count: traffic.skipped_output_clear_count,
            schedule_cache_keys,
            native_dispatcher_wall_time: traffic.native_dispatcher_wall_time,
            traffic: native_cpu_replay_traffic(traffic),
            executor_wall_time,
            wall_time,
        };
        debug_assert!(validate_native_cpu_run_report(&report).is_ok());
        Ok((
            evaluation_result(
                outputs,
                &self.output_names,
                loss_weight,
                self.capture_identity,
            )?,
            report,
        ))
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

impl CpuCompiledTrainingProgram {
    fn preflight_native(
        &self,
        vectorized: bool,
        external_learning_rate: bool,
    ) -> Result<(RecurrentNativePreparation, Duration)> {
        let started = Instant::now();
        let mut provided = zero_inputs(&self.inputs)?;
        if external_learning_rate {
            provided.insert(
                LEARNING_RATE_INPUT.to_owned(),
                TensorData::zeros_with_dtype(Shape::from([]), DType::F32)?,
            );
        }
        let preparation = self
            .capture
            .preflight_recurrent_native(&self.runtime, &self.cursor, &provided, vectorized)
            .map_err(replay_error)?;
        Ok((preparation, started.elapsed()))
    }

    fn finish_native(
        &self,
        preparation: RecurrentNativePreparation,
        plan: PlannedNativeItems,
        residual_wall_time: Duration,
    ) -> Result<PreparedNativeCpuProgram> {
        let replay = preparation.finish(plan).map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let wall_time = native_preparation_wall_time(trace.module, residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: self.capture_identity(),
            native_identity: trace.replay.identity,
            vectorized: trace.replay.vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            work: NativeCpuPreparationWork::from_module(trace.module),
            phases: NativeCpuPreparationPhases::from_module(trace.module, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                trace.dispatch_segmentation,
            )?,
            execution_plan: self.recurrent_capture.execution_plan().clone(),
            wall_time,
        };
        report.validate_work()?;
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    fn admit_training_step(
        &self,
        request: CompiledStepReplayRequest,
    ) -> Result<AdmittedTrainingStep> {
        let CompiledStepReplayRequest {
            inputs,
            learning_rate,
            non_finite_policy,
            output_selection,
            injected_failure,
        } = request;
        validate_training_inputs(&self.inputs, &inputs)?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, non_finite_policy)?;
        }
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| training("compiled training step overflow"))?;
        Ok(AdmittedTrainingStep {
            inputs,
            learning_rate,
            transaction: TrainingStepTransaction {
                next_step,
                non_finite_policy,
                output_selection,
                injected_failure,
            },
        })
    }

    fn publish_main_step(
        &mut self,
        completed: CompletedTrainingStep,
    ) -> CompiledTrainingStepResult {
        self.step = completed.next_step;
        completed.result
    }

    fn publish_accumulation_step(
        &mut self,
        completed: CompletedTrainingStep,
        cursor: ProjectedRecurrentCursor,
    ) -> CompiledTrainingStepResult {
        cursor.publish(&mut self.cursor);
        self.step = completed.next_step;
        completed.result
    }

    /// Executes one graph-free replay and atomically publishes every recurrent
    /// successor. The learning rate is an explicit rank-zero F32 input.
    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_inner(inputs, learning_rate, None)
    }

    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_with_output_selection(
            inputs,
            learning_rate,
            CompiledStepOutputSelection::All,
            injected_failure,
        )
    }

    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_commit_only_inner(inputs, learning_rate, None)
    }

    fn step_commit_only_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_with_output_selection(
            inputs,
            learning_rate,
            CompiledStepOutputSelection::CommitOnly,
            injected_failure,
        )
    }

    fn step_with_output_selection(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        output_selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_inner_with_learning_rate(
            CompiledStepReplayRequest {
                inputs,
                learning_rate: Some(learning_rate),
                non_finite_policy: CpuNonFinitePolicy::Propagate,
                output_selection,
                injected_failure,
            },
            true,
        )
    }

    fn step_inner_with_learning_rate(
        &mut self,
        request: CompiledStepReplayRequest,
        validate_commit_observations: bool,
    ) -> Result<CompiledTrainingStepResult> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_main_bindings();
        let transaction = transaction.authenticate_outputs(
            &self.capture,
            &self.phase_outputs,
            true,
            validate_commit_observations,
        )?;
        let replay = self
            .capture
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                &mut self.cursor,
                &provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| transaction.validate_transition(outputs, successors),
            )
            .map_err(replay_error)?;
        let completed = transaction.complete(
            replay.outputs,
            &self.phase_outputs,
            self.cursor.capture_identity(),
        );
        Ok(self.publish_main_step(completed))
    }

    fn step_native_inner_with_learning_rate(
        &mut self,
        request: CompiledStepReplayRequest,
        validate_commit_observations: bool,
        native: NativeReplayContext<'_>,
    ) -> Result<(CompiledTrainingStepResult, NativeCpuRunReport)> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_main_bindings();
        let started = Instant::now();
        let transaction = transaction.authenticate_outputs(
            &self.capture,
            &self.phase_outputs,
            true,
            validate_commit_observations,
        )?;
        let next_step = transaction.next_step();
        let replay = native
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                &mut self.cursor,
                &provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| {
                    transaction.validate_transition(outputs, successors.iter().copied())
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let executor_wall_time = replay.executor_wall_time;
        let replay = replay.replay;
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            self.capture_identity(),
            native,
            traffic,
            executor_wall_time,
            next_step,
            started.elapsed(),
        );
        let completed = transaction.complete(
            replay.outputs,
            &self.phase_outputs,
            self.cursor.capture_identity(),
        );
        Ok((self.publish_main_step(completed), report))
    }

    fn step_accumulation_inner_with_learning_rate(
        &mut self,
        transition: &CompiledTrainingSiblingPlan,
        request: CompiledStepReplayRequest,
    ) -> Result<CompiledTrainingStepResult> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_accumulation_bindings();
        let mut prepared =
            self.prepare_phase_replay(&transition.phase().cursor_projection, provided)?;
        let transaction = transaction.authenticate_outputs(
            &transition.phase().capture,
            &self.phase_outputs,
            false,
            false,
        )?;
        let replay = transition
            .phase()
            .capture
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| transaction.validate_transition(outputs, successors),
            )
            .map_err(replay_error)?;
        let completed =
            transaction.complete(replay.outputs, &self.phase_outputs, self.capture_identity());
        Ok(self.publish_accumulation_step(completed, prepared.cursor))
    }

    fn step_accumulation_native_inner_with_learning_rate(
        &mut self,
        transition: &CompiledTrainingSiblingPlan,
        request: CompiledStepReplayRequest,
        native: NativeReplayContext<'_>,
    ) -> Result<(CompiledTrainingStepResult, NativeCpuRunReport)> {
        let admitted = self.admit_training_step(request)?;
        let (provided, transaction) = admitted.into_accumulation_bindings();
        let started = Instant::now();
        let mut prepared =
            self.prepare_phase_replay(&transition.phase().cursor_projection, provided)?;
        let transaction = transaction.authenticate_outputs(
            &transition.phase().capture,
            &self.phase_outputs,
            false,
            false,
        )?;
        let replay = native
            .replay_recurrent_selected_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                transaction.selected_requested(),
                transaction.injected_failure(),
                |outputs, successors| {
                    transaction.validate_transition(outputs, successors.iter().copied())
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let executor_wall_time = replay.executor_wall_time;
        let replay = replay.replay;
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            transition.phase().capture_identity,
            native,
            traffic,
            executor_wall_time,
            0,
            started.elapsed(),
        );
        let completed =
            transaction.complete(replay.outputs, &self.phase_outputs, self.capture_identity());
        Ok((
            self.publish_accumulation_step(completed, prepared.cursor),
            report,
        ))
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn capture_identity(&self) -> u64 {
        self.cursor.capture_identity()
    }

    fn plan(&self) -> Result<CompiledTrainingPlan> {
        let state_frontier = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                let buffer = self
                    .state_input_buffers
                    .get(input)
                    .ok_or_else(|| training("compiled runtime state buffer is absent"))?;
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((key.clone(), (value, state.version)))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(CompiledTrainingPlan {
            capture: self.capture.clone(),
            recurrent_capture: self.recurrent_capture.clone(),
            inputs: self.inputs.clone(),
            phase_outputs: self.phase_outputs.clone(),
            parameter_buffers: self.parameter_buffers.clone(),
            optimizer_buffers: self.optimizer_buffers.clone(),
            workload_buffers: self.workload_buffers.clone(),
            state_input_buffers: self.state_input_buffers.clone(),
            state_input_keys: self.state_input_keys.clone(),
            state_values: state_frontier
                .iter()
                .map(|(key, (value, _))| (key.clone(), value.clone()))
                .collect(),
            state_versions: state_frontier
                .into_iter()
                .map(|(key, (_, version))| (key, version))
                .collect(),
            recurrent_store_groups: self.recurrent_store_groups.clone(),
            frozen_parameter_nodes: self.frozen_parameter_nodes.clone(),
            step: self.step,
            accumulation: self.accumulation.clone(),
        })
    }

    /// Returns independent owned parameter snapshots in canonical name order.
    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.snapshots(&self.parameter_buffers)
    }

    /// Current logical parameter versions. Every successful step advances all
    /// parameter and optimizer-state buffers exactly once.
    fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.versions(&self.parameter_buffers)
    }

    fn adamw_state_snapshots(
        &self,
        state: AdamWParameterState,
    ) -> Result<BTreeMap<String, TensorData>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.parameter_for_adamw_state(state)
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.snapshots(&buffers)
    }

    fn adamw_state_versions(&self, state: AdamWParameterState) -> Result<BTreeMap<String, u64>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.parameter_for_adamw_state(state)
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.versions(&buffers)
    }

    fn momentum_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.momentum_parameter_name()
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.snapshots(&buffers)
    }

    fn momentum_versions(&self) -> Result<BTreeMap<String, u64>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.momentum_parameter_name()
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.versions(&buffers)
    }

    fn global_snapshot(&self, state: AdamWGlobalState) -> Result<TensorData> {
        let buffer = self
            .optimizer_buffers
            .get(&RecurrentStateKey::adamw_global(state))
            .ok_or_else(|| training("compiled global optimizer state is absent"))?;
        let state = self.current_state(*buffer)?;
        Ok(self
            .runtime
            .snapshot(state)
            .map_err(runtime_error)?
            .tensor()
            .clone())
    }

    fn workload_snapshot(&self, key: &RecurrentStateKey) -> Result<TensorData> {
        let buffer = self
            .workload_buffers
            .get(key)
            .ok_or_else(|| training("compiled workload state is absent"))?;
        let state = self.current_state(*buffer)?;
        Ok(self
            .runtime
            .snapshot(state)
            .map_err(runtime_error)?
            .tensor()
            .clone())
    }

    fn restore_frontier(
        &mut self,
        step: u64,
        values: &BTreeMap<RecurrentStateKey, TensorData>,
        versions: &BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<()> {
        let buffers = self
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(
                self.optimizer_buffers
                    .iter()
                    .map(|(name, buffer)| (name.clone(), *buffer)),
            )
            .chain(
                self.workload_buffers
                    .iter()
                    .map(|(name, buffer)| (name.clone(), *buffer)),
            )
            .collect::<BTreeMap<_, _>>();
        if values.len() != buffers.len() || values.keys().ne(buffers.keys()) {
            return Err(training("compiled checkpoint state names mismatch"));
        }
        if versions.len() != buffers.len() || versions.keys().ne(buffers.keys()) {
            return Err(training("compiled checkpoint state versions mismatch"));
        }

        let mut snapshots = Vec::with_capacity(buffers.len());
        for (name, buffer) in buffers {
            let value = &values[&name];
            let current = self.current_state(buffer)?;
            if value.shape() != &current.shape || value.dtype() != current.dtype {
                return Err(training("compiled checkpoint state descriptor mismatch"));
            }
            checked_bytes(value)?;
            let mut state = current.clone();
            state.version = versions[&name];
            snapshots.push((state, value.clone()));
        }

        let frontier = snapshots
            .iter()
            .map(|(state, _)| state.clone())
            .collect::<Vec<_>>();
        let cursor = MixedReplayCursor::resume(&self.capture, frontier).map_err(replay_error)?;
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_snapshots(snapshots)
            .map_err(runtime_error)?;
        self.runtime = runtime;
        self.cursor = cursor;
        self.step = step;
        Ok(())
    }

    #[cfg(test)]
    fn replace_state_values(
        &mut self,
        step: u64,
        replacements: BTreeMap<RecurrentStateKey, TensorData>,
    ) -> Result<()> {
        let plan = self.plan()?;
        let mut values = plan.state_values;
        for (key, value) in replacements {
            let current = values
                .get(&key)
                .ok_or_else(|| training("compiled replacement state is absent"))?;
            if value.shape() != current.shape() || value.dtype() != current.dtype() {
                return Err(training("compiled replacement state descriptor mismatch"));
            }
            values.insert(key, value);
        }
        self.restore_frontier(step, &values, &plan.state_versions)
    }

    fn prepare_auxiliary_replay(
        &self,
        transition: &CompiledRecurrentPhasePlan,
        learning_rate: Option<TensorData>,
    ) -> Result<CpuAuxiliaryReplay> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let mut provided = BTreeMap::new();
        if let Some(learning_rate) = learning_rate {
            provided.insert(LEARNING_RATE_INPUT.to_owned(), learning_rate);
        }
        self.prepare_phase_replay(&transition.cursor_projection, provided)
    }

    fn prepare_phase_replay(
        &self,
        projection: &PreparedRecurrentCursorProjection,
        provided: BTreeMap<String, TensorData>,
    ) -> Result<CpuAuxiliaryReplay> {
        Ok(CpuAuxiliaryReplay {
            cursor: projection
                .project(&self.cursor)
                .map_err(cursor_projection_error)?,
            provided,
        })
    }

    fn replay_auxiliary_transition(
        &mut self,
        transition: &CompiledAdamWAuxiliaryPlan,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWAuxiliaryReports> {
        let mut prepared = self.prepare_auxiliary_replay(transition.phase(), learning_rate)?;
        let output_schema = transition.outputs.clone();
        let mut reports = None;
        let _replay = transition
            .phase()
            .capture
            .replay_recurrent_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(outputs, successors, non_finite_policy, false)?;
                    reports = Some(output_schema.validate_and_decode(outputs, non_finite_policy)?);
                    Ok(())
                },
            )
            .map_err(replay_error)?;
        let reports = reports.expect("compiled auxiliary outputs were authenticated before commit");
        #[cfg(debug_assertions)]
        {
            let mut committed = _replay.committed.clone();
            committed.sort_by_key(|state| state.buffer);
            debug_assert_eq!(committed, prepared.cursor.cursor().frontier());
        }
        prepared.cursor.publish(&mut self.cursor);
        Ok(reports)
    }

    fn preflight_native_auxiliary_transition(
        &self,
        transition: &CompiledRecurrentPhasePlan,
        vectorized: bool,
        external_learning_rate: bool,
    ) -> Result<(RecurrentNativePreparation, Duration)> {
        debug_assert!(!transition.retains_unchanged());
        let started = Instant::now();
        let learning_rate = external_learning_rate
            .then(|| TensorData::zeros_with_dtype(Shape::from([]), DType::F32))
            .transpose()?;
        let prepared = self.prepare_auxiliary_replay(transition, learning_rate)?;
        let preparation = transition
            .capture
            .preflight_recurrent_native(
                &self.runtime,
                prepared.cursor.cursor(),
                &prepared.provided,
                vectorized,
            )
            .map_err(replay_error)?;
        Ok((preparation, started.elapsed()))
    }

    fn preflight_native_accumulation(
        &self,
        transition: &CompiledTrainingSiblingPlan,
        vectorized: bool,
    ) -> Result<(RecurrentNativePreparation, Duration)> {
        debug_assert!(transition.phase().retains_unchanged());
        let started = Instant::now();
        let prepared = self.prepare_phase_replay(
            &transition.phase().cursor_projection,
            zero_inputs(&self.inputs)?,
        )?;
        let preparation = transition
            .phase()
            .capture
            .preflight_recurrent_native_retaining_unchanged(
                &self.runtime,
                prepared.cursor.cursor(),
                &prepared.provided,
                vectorized,
            )
            .map_err(replay_error)?;
        Ok((preparation, started.elapsed()))
    }

    fn finish_native_auxiliary_transition(
        &self,
        transition: &CompiledRecurrentPhasePlan,
        preparation: RecurrentNativePreparation,
        plan: PlannedNativeItems,
        residual_wall_time: Duration,
    ) -> Result<PreparedNativeCpuProgram> {
        let replay = preparation.finish(plan).map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let wall_time = native_preparation_wall_time(trace.module, residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: transition.capture_identity(),
            native_identity: trace.replay.identity,
            vectorized: trace.replay.vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            work: NativeCpuPreparationWork::from_module(trace.module),
            phases: NativeCpuPreparationPhases::from_module(trace.module, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                trace.dispatch_segmentation,
            )?,
            execution_plan: transition.recurrent_capture.execution_plan().clone(),
            wall_time,
        };
        report.validate_work()?;
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    fn finish_native_accumulation(
        &self,
        transition: &CompiledTrainingSiblingPlan,
        preparation: RecurrentNativePreparation,
        plan: PlannedNativeItems,
        residual_wall_time: Duration,
    ) -> Result<PreparedNativeCpuProgram> {
        let replay = preparation.finish(plan).map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let wall_time = native_preparation_wall_time(trace.module, residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: transition.phase().capture_identity,
            native_identity: trace.replay.identity,
            vectorized: trace.replay.vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            work: NativeCpuPreparationWork::from_module(trace.module),
            phases: NativeCpuPreparationPhases::from_module(trace.module, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                trace.dispatch_segmentation,
            )?,
            execution_plan: transition
                .phase()
                .recurrent_capture
                .execution_plan()
                .clone(),
            wall_time,
        };
        report.validate_work()?;
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    fn replay_auxiliary_transition_native(
        &mut self,
        transition: &CompiledAdamWAuxiliaryPlan,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        native: NativeReplayContext<'_>,
        successful_invocation: u64,
        injected_failure: Option<u64>,
    ) -> Result<(CompiledAdamWAuxiliaryReports, NativeCpuRunReport)> {
        let started = Instant::now();
        let mut prepared = self.prepare_auxiliary_replay(transition.phase(), learning_rate)?;
        let output_schema = transition.outputs.clone();
        let mut reports = None;
        let replay = native
            .replay_recurrent_checked(
                &mut self.runtime,
                prepared.cursor.cursor_mut(),
                &prepared.provided,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(
                        outputs,
                        successors.iter().copied(),
                        non_finite_policy,
                        false,
                    )?;
                    reports = Some(output_schema.validate_and_decode(outputs, non_finite_policy)?);
                    Ok(())
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let executor_wall_time = replay.executor_wall_time;
        let replay = replay.replay;
        let reports = reports.expect("compiled auxiliary outputs were authenticated before commit");
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            transition.capture_identity(),
            native,
            traffic,
            executor_wall_time,
            successful_invocation,
            started.elapsed(),
        );
        #[cfg(debug_assertions)]
        {
            let mut committed = replay.committed.clone();
            committed.sort_by_key(|state| state.buffer);
            debug_assert_eq!(committed, prepared.cursor.cursor().frontier());
        }
        prepared.cursor.publish(&mut self.cursor);
        Ok((reports, report))
    }

    fn snapshots(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, TensorData>> {
        buffers
            .iter()
            .map(|(name, buffer)| {
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((name.clone(), value))
            })
            .collect()
    }

    fn versions(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, u64>> {
        buffers
            .iter()
            .map(|(name, buffer)| Ok((name.clone(), self.current_state(*buffer)?.version)))
            .collect()
    }

    fn current_state(&self, buffer: u64) -> Result<&BufferState> {
        self.cursor
            .frontier()
            .iter()
            .find(|state| state.buffer == buffer)
            .ok_or_else(|| training("compiled persistent state is absent"))
    }
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
        let plan = CompiledTrainingPlan::compile(MomentumProgram { config }, parameters, build)?;
        Ok(Self {
            inner: plan.prepare_cpu()?,
        })
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
        let parameter_plan = ModuleParameterPlan::new(module, &BTreeSet::new())?;
        let parameters = parameter_plan.initial_parameters()?;
        Self::compile(config, parameters, |graph, inputs, parameters| {
            parameter_plan.lower(graph, parameters, |graph| build(module, graph, inputs))
        })
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
        let parameters = checkpoint
            .parameters
            .iter()
            .map(|(name, value)| TrainingParameterInit::new(name.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()?;
        let mut runtime = Self::compile(config, parameters, build)?;
        runtime.restore_checkpoint_in_place(checkpoint)?;
        Ok(runtime)
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
        let parameter_plan = ModuleParameterPlan::new(module, &BTreeSet::new())?;
        let destination_parameters = parameter_plan.initial_parameters()?;
        if destination_parameters.len() != checkpoint.parameters.len()
            || destination_parameters.iter().any(|parameter| {
                checkpoint
                    .parameters
                    .get(parameter.name())
                    .is_none_or(|saved| {
                        saved.shape() != parameter.value().shape()
                            || saved.dtype() != parameter.value().dtype()
                    })
            })
        {
            return Err(training(
                "compiled momentum-SGD checkpoint parameter schema mismatch",
            ));
        }
        let parameters = checkpoint
            .parameters
            .iter()
            .map(|(name, value)| TrainingParameterInit::new(name.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()?;
        let mut runtime = Self::compile(config, parameters, |graph, inputs, parameters| {
            parameter_plan.lower(graph, parameters, |graph| build(module, graph, inputs))
        })?;
        runtime.restore_checkpoint_in_place(checkpoint)?;
        Ok(runtime)
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
        if self.capture_identity() != checkpoint.capture_identity {
            return Err(training(
                "compiled momentum-SGD checkpoint capture identity mismatch",
            ));
        }
        if checkpoint.parameters.keys().ne(checkpoint.momenta.keys())
            || checkpoint
                .parameters
                .keys()
                .ne(checkpoint.parameter_versions.keys())
            || checkpoint
                .parameters
                .keys()
                .ne(checkpoint.momentum_versions.keys())
        {
            return Err(training(
                "compiled momentum-SGD checkpoint state names mismatch",
            ));
        }
        let values = checkpoint
            .parameters
            .iter()
            .map(|(name, value)| (RecurrentStateKey::parameter(name), value.clone()))
            .chain(
                checkpoint
                    .momenta
                    .iter()
                    .map(|(name, value)| (RecurrentStateKey::momentum(name), value.clone())),
            )
            .collect();
        let versions = checkpoint
            .parameter_versions
            .iter()
            .map(|(name, version)| (RecurrentStateKey::parameter(name), *version))
            .chain(
                checkpoint
                    .momentum_versions
                    .iter()
                    .map(|(name, version)| (RecurrentStateKey::momentum(name), *version)),
            )
            .collect();
        let plan =
            self.inner
                .plan()?
                .restore_frontier_with_versions(checkpoint.step, values, versions)?;
        Ok(Self {
            inner: plan.prepare_cpu()?,
        })
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

impl CompiledAdamWPlan {
    pub fn compile<F>(
        config: CompiledAdamWConfig,
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
        if !config.frozen_parameters.is_empty() {
            return Err(training(
                "compiled AdamW raw parameters cannot resolve frozen parameter names",
            ));
        }
        Self::compile_parameters(config, parameters, build)
    }

    fn compile_parameters<F>(
        config: CompiledAdamWConfig,
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
        reject_token_weighted_scalar_loss(&config)?;
        Self::compile_parameters_with_lowered_loss(config, parameters, build)
    }

    fn compile_parameters_with_lowered_loss<F>(
        config: CompiledAdamWConfig,
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
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        validate_weight_decay_exclusion_names(
            &config,
            parameters.iter().map(TrainingParameterInit::name),
        )?;
        let (inner, compile_phases) = CompiledTrainingPlan::compile_observed(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            build,
        )?;
        Self::from_compiled_inner(config, inner, compile_phases)
    }

    fn compile_parameters_with_ignore_index<F>(
        config: CompiledAdamWConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>, NodeId)>,
    {
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        validate_weight_decay_exclusion_names(
            &config,
            parameters.iter().map(TrainingParameterInit::name),
        )?;
        let (inner, compile_phases) = CompiledTrainingPlan::compile_with_token_weight_observed(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            build,
        )?;
        Self::from_compiled_inner(config, inner, compile_phases)
    }

    fn from_compiled_inner(
        config: CompiledAdamWConfig,
        inner: CompiledTrainingPlan,
        mut compile_phases: CompiledTrainingCompileObservation,
    ) -> Result<Self> {
        let topology = CompiledTrainingWindowTopology::from_config(&config);
        let (partial_flush, partial_flush_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_partial_flush(&inner, &config)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        let (zero_grad, zero_grad_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_zero_grad(&inner, topology)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        compile_phases.set_auxiliary(partial_flush_phase, zero_grad_phase);
        let program_identity = inner.capture_identity()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract: CompiledAdamWContract::from_config(&config, None),
            progress: CompiledTrainingWindowProgress::INITIAL,
            evaluation: None,
            compile_phases: Some(compile_phases),
        })
    }

    /// Compiles an ordinary module forward without preparing a runtime.
    pub fn compile_module<M, F>(config: CompiledAdamWConfig, module: &M, build: F) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan =
            Self::compile_parameters(config, parameters, |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| build(module, graph, inputs),
                )
            })?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    /// Compiles a module through one explicit scalar-or-token-mean objective
    /// facade without preparing a runtime.
    ///
    /// The objective must agree with the configuration: ordinary configs
    /// accept [`CompiledAdamWObjective::Scalar`], while token-weighted configs
    /// accept [`CompiledAdamWObjective::TokenMean`]. The selected objective is
    /// lowered through the same capture path as the compatibility constructors.
    pub fn compile_module_graph<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        Self::compile_module_graph_parameters(config, module, parameter_plan, build)
    }

    /// Compiles an ignore-index token-mean module while exposing the exact
    /// compiler-owned target validity nodes to the graph builder.
    ///
    /// This opt-in surface is available only for
    /// [`CompiledAdamWConfig::with_token_weighted_ignore_index`]. Existing graph
    /// constructors retain their capture topology and bytes.
    pub fn compile_module_graph_with_ignore_index<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        Self::compile_module_graph_with_ignore_index_parameters(
            config,
            module,
            parameter_plan,
            build,
        )
    }

    fn compile_module_graph_parameters<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan = Self::compile_parameters_with_lowered_loss(
            config,
            parameters,
            |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| {
                        let built = build(module, graph, inputs)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective(
                            &objective_config,
                            graph,
                            inputs,
                            objective,
                        )?;
                        Ok((loss, outputs))
                    },
                )
            },
        )?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    fn compile_module_graph_with_ignore_index_parameters<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan = Self::compile_parameters_with_ignore_index(
            config,
            parameters,
            |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| {
                        let nodes = lower_ignore_index_nodes(&objective_config, graph, inputs)?;
                        let built = build(module, graph, inputs, nodes)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective_with_ignore_index_nodes(
                            &objective_config,
                            graph,
                            objective,
                            nodes,
                        )?;
                        Ok((loss, outputs, nodes.weight))
                    },
                )
            },
        )?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    /// Compiles module-bound AdamW with one device-resident Threefry block
    /// counter shared by the module's explicit residual-dropout calls.
    pub fn compile_module_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        reject_token_weighted_scalar_loss(&config)?;
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            build,
        )
    }

    /// Compiles a module with recurrent dropout through the unified explicit
    /// scalar-or-token-mean objective facade.
    pub fn compile_module_graph_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let built = build(module, graph, inputs, dropout)?;
                let (objective, outputs) = built.into_parts();
                let loss =
                    lower_compiled_adamw_objective(&objective_config, graph, inputs, objective)?;
                Ok((loss, outputs))
            },
        )
    }

    /// Dropout counterpart of [`Self::compile_module_graph_with_ignore_index`].
    pub fn compile_module_graph_with_dropout_and_ignore_index<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        Self::compile_module_with_dropout_parameters_inner(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let nodes = lower_ignore_index_nodes(&objective_config, graph, inputs)?;
                let built = build(module, graph, inputs, nodes, dropout)?;
                let (objective, outputs) = built.into_parts();
                let loss = lower_compiled_adamw_objective_with_ignore_index_nodes(
                    &objective_config,
                    graph,
                    objective,
                    nodes,
                )?;
                Ok((loss, outputs, Some(nodes.weight)))
            },
        )
    }

    /// Compiles module-bound AdamW and derives its scalar differentiation root
    /// from fixed-shape per-token F32 losses and the configured token mask.
    ///
    /// The returned loss node must have exactly the mask input's descriptor.
    /// Capture owns `sum(mask * losses) / sum(mask)` as both the public loss and
    /// differentiation root, while the existing replay guard rejects an empty
    /// or malformed mask before recurrent state can advance.
    pub fn compile_token_mean_module_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let (mask_input, mask_shape) = token_mean_loss_descriptor(&config)?;
        let allow_zero_valid_token_microbatches = config.allow_zero_valid_token_microbatches;
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let (losses, outputs) = build(module, graph, inputs, dropout)?;
                let loss = lower_token_mean_loss(
                    graph,
                    losses,
                    inputs[mask_input.as_str()],
                    &mask_shape,
                    allow_zero_valid_token_microbatches,
                )?;
                Ok((loss, outputs))
            },
        )
    }

    fn compile_module_with_dropout_parameters<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_module_with_dropout_parameters_inner(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let (loss, outputs) = build(module, graph, inputs, dropout)?;
                Ok((loss, outputs, None))
            },
        )
    }

    fn compile_module_with_dropout_parameters_inner<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>, Option<NodeId>)>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let topology = CompiledTrainingWindowTopology::from_config(&config);
        let workload = StateSpec::dropout_counter()?;
        let mut dropout_state = None;
        let mut frozen_parameter_nodes = BTreeSet::new();
        let (mut inner, mut compile_phases) = CompiledTrainingPlan::compile_with_workload_observed(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            Some(workload),
            |graph, inputs, parameters, counter| {
                let counter =
                    counter.ok_or_else(|| training("compiled dropout state is absent"))?;
                let mut provider = CompiledDropoutStream::new(counter, dropout);
                let (loss, outputs, token_weight) = parameter_plan
                    .lower_with_frozen_parameter_nodes(
                        graph,
                        parameters,
                        &mut frozen_parameter_nodes,
                        |graph| build(module, graph, inputs, &mut provider),
                    )?;
                let (successor, state) = provider.finish(graph)?;
                dropout_state = Some(state);
                Ok((loss, outputs, Some(successor), token_weight))
            },
        )?;
        inner.frozen_parameter_nodes = frozen_parameter_nodes;
        let dropout = dropout_state
            .ok_or_else(|| training("compiled dropout configuration produced no state"))?;
        let (partial_flush, partial_flush_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_partial_flush(&inner, &config)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        let (zero_grad, zero_grad_phase) = if topology.accumulating() {
            let started = Instant::now();
            let plan = CompiledAdamWAuxiliaryPlan::compile_zero_grad(&inner, topology)?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                plan.phase()
                    .recurrent_capture
                    .execution_plan()
                    .schedule_item_count,
            );
            (Some(plan), Some(phase))
        } else {
            (None, None)
        };
        compile_phases.set_auxiliary(partial_flush_phase, zero_grad_phase);
        let program_identity = inner.capture_identity()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract: CompiledAdamWContract::from_config(&config, Some(dropout)),
            progress: CompiledTrainingWindowProgress::INITIAL,
            evaluation: None,
            compile_phases: Some(compile_phases),
        })
    }

    /// Returns an independent plan whose recurrent frontier is restored from
    /// one checkpoint without rebuilding the Graph, derivatives, schedules,
    /// captures, accumulation/partial-flush transitions, or attached evaluation
    /// program.
    ///
    /// The checkpoint must authenticate this exact compiled program and its
    /// accumulation, dropout, frozen-parameter, input, clipping, loss-scaling,
    /// and partial-flush policies. The source plan remains unchanged on both
    /// success and failure, and each returned plan may be prepared or restored
    /// independently.
    pub fn restore_checkpoint(&self, checkpoint: &CompiledAdamWCheckpoint) -> Result<Self> {
        let decoded = checkpoint.decoded();
        let frontier = self.validate_checkpoint_frontier(decoded)?;
        #[cfg(test)]
        record_adamw_checkpoint_restore(|counts| counts.borrowed_plan_clones += 1);
        self.clone().apply_checkpoint_frontier(frontier)
    }

    fn restore_checkpoint_owned(self, checkpoint: &CompiledAdamWCheckpoint) -> Result<Self> {
        let frontier = self.validate_checkpoint_frontier(checkpoint.decoded())?;
        #[cfg(test)]
        record_adamw_checkpoint_restore(|counts| counts.consumed_plan_restores += 1);
        self.apply_checkpoint_frontier(frontier)
    }

    fn validate_checkpoint_frontier(
        &self,
        decoded: &DecodedAdamWCheckpoint,
    ) -> Result<ValidatedAdamWCheckpointFrontier> {
        let topology = CompiledTrainingWindowTopology::from_contract(&self.contract);
        if self.contract.gradient_accumulation_steps != decoded.accumulation_steps {
            return Err(training(
                "compiled AdamW checkpoint accumulation policy mismatch",
            ));
        }
        if self.contract.window_loss_report != decoded.window_loss_report {
            return Err(training(
                "compiled AdamW checkpoint window-loss reporting policy mismatch",
            ));
        }
        match (
            topology.retains_token_count(),
            decoded.accumulated_token_count,
        ) {
            (true, Some(count)) => validate_retained_token_count(
                &self.inner.inputs,
                self.contract
                    .token_weight_policy
                    .as_ref()
                    .expect("retained token counts require token weighting"),
                decoded.accumulation_index,
                count,
                self.contract.allow_zero_valid_token_microbatches,
            )?,
            (false, None) => {}
            _ => {
                return Err(training(
                    "compiled AdamW checkpoint token-weighting policy mismatch",
                ));
            }
        }
        if self.capture_identity() != decoded.capture_identity {
            return Err(training(
                "compiled AdamW checkpoint capture identity mismatch",
            ));
        }
        if decoded.accumulation_capture_identity.is_some()
            && self.accumulation_capture_identity() != decoded.accumulation_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint accumulation capture identity mismatch",
            ));
        }
        if decoded.flush_capture_identity.is_some()
            && self.flush_capture_identity() != decoded.flush_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint partial flush capture identity mismatch",
            ));
        }
        if decoded.reset_capture_identity.is_some()
            && self.zero_grad_capture_identity() != decoded.reset_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint zero-grad capture identity mismatch",
            ));
        }
        match (self.contract.dropout, decoded.dropout_block_counter) {
            (None, None) => {}
            (None, Some(_)) => {
                return Err(training(
                    "compiled AdamW dropout checkpoint requires dropout restore",
                ));
            }
            (Some(_), None) => {
                return Err(training(
                    "compiled AdamW checkpoint has no dropout block counter",
                ));
            }
            (Some(dropout), Some(counter)) => {
                let expected = decoded
                    .replay_step
                    .checked_mul(dropout.blocks_per_replay)
                    .ok_or_else(|| training("compiled dropout counter progress overflows"))?;
                if counter != expected {
                    return Err(training(
                        "compiled dropout counter and replay progress diverged",
                    ));
                }
            }
        }

        let mut values = decoded
            .parameters
            .iter()
            .map(|(name, value)| (RecurrentStateKey::parameter(name.clone()), value.clone()))
            .collect::<BTreeMap<_, _>>();
        for (name, value) in &decoded.first_moments {
            values.insert(
                RecurrentStateKey::adamw_parameter(name.clone(), AdamWParameterState::FirstMoment),
                value.clone(),
            );
        }
        for (name, value) in &decoded.second_moments {
            values.insert(
                RecurrentStateKey::adamw_parameter(name.clone(), AdamWParameterState::SecondMoment),
                value.clone(),
            );
        }
        for (name, value) in &decoded.gradient_accumulators {
            values.insert(
                RecurrentStateKey::adamw_parameter(
                    name.clone(),
                    AdamWParameterState::GradientAccumulator,
                ),
                value.clone(),
            );
        }
        values.insert(
            RecurrentStateKey::adamw_global(AdamWGlobalState::Step),
            TensorData::from_scalars(
                Shape::from([]),
                DType::U64,
                [Scalar::U(decoded.optimizer_step)],
            )?,
        );
        if topology.accumulating() {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex),
                TensorData::from_scalars(
                    Shape::from([]),
                    DType::U64,
                    [Scalar::U(decoded.accumulation_index)],
                )?,
            );
        }
        if let Some(count) = decoded.accumulated_token_count {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedTokenCount),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(count)])?,
            );
        }
        if let Some(numerator) = &decoded.accumulated_loss_numerator {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator),
                numerator.clone(),
            );
        }
        if let Some(counter) = decoded.dropout_block_counter {
            values.insert(
                RecurrentStateKey::dropout_counter(),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(counter)])?,
            );
        }

        let progress = CompiledTrainingWindowProgress {
            replay_step: decoded.replay_step,
            optimizer_step: decoded.optimizer_step,
            accumulation_index: decoded.accumulation_index,
            discarded_microbatches: decoded.discarded_microbatches,
            flushed_window_count: decoded.flushed_window_count,
            flushed_microbatch_count: decoded.flushed_microbatch_count,
            reset_transition_count: decoded.reset_transition_count,
        };
        let optimizer_version = decoded
            .replay_step
            .checked_add(decoded.flushed_window_count)
            .ok_or_else(|| training("compiled AdamW checkpoint state version overflows"))?;
        let reset_version = optimizer_version
            .checked_add(decoded.reset_transition_count)
            .ok_or_else(|| training("compiled AdamW checkpoint reset state version overflows"))?;
        let versions = values
            .keys()
            .cloned()
            .map(|key| {
                let version = if self.inner.workload_buffers.contains_key(&key) {
                    decoded.replay_step
                } else if key.is_accumulation_reset_state() {
                    reset_version
                } else {
                    optimizer_version
                };
                (key, version)
            })
            .collect();

        Ok(ValidatedAdamWCheckpointFrontier {
            replay_step: decoded.replay_step,
            values,
            versions,
            progress,
        })
    }

    fn apply_checkpoint_frontier(self, frontier: ValidatedAdamWCheckpointFrontier) -> Result<Self> {
        let Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract,
            progress: _,
            evaluation,
            compile_phases,
        } = self;
        let inner = inner.restore_frontier_with_versions(
            frontier.replay_step,
            frontier.values,
            frontier.versions,
        )?;
        let partial_flush = partial_flush
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        let zero_grad = zero_grad
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            contract,
            progress: frontier.progress,
            evaluation,
            compile_phases,
        })
    }

    #[cfg(test)]
    fn capture_allocations(&self) -> AdamWPlanCaptureAllocations {
        AdamWPlanCaptureAllocations {
            main: captured_schedule_allocation(&self.inner.capture.schedule),
            accumulation: self
                .inner
                .accumulation
                .as_ref()
                .map(|plan| captured_schedule_allocation(&plan.phase().capture.schedule)),
            partial_flush: self
                .partial_flush
                .as_ref()
                .map(|plan| captured_schedule_allocation(&plan.phase().capture.schedule)),
            zero_grad: self
                .zero_grad
                .as_ref()
                .map(|plan| captured_schedule_allocation(&plan.phase().capture.schedule)),
            evaluation: self
                .evaluation
                .as_ref()
                .map(|plan| captured_schedule_allocation(plan.inference.capture())),
        }
    }

    #[cfg(test)]
    fn topology_allocations(&self) -> AdamWPlanTopologyAllocations {
        let phase = |phase: &CompiledRecurrentPhasePlan| {
            (
                Arc::as_ptr(&phase.capture) as usize,
                Arc::as_ptr(&phase.recurrent_capture.execution_plan) as usize,
                Arc::as_ptr(&phase.cursor_projection) as usize,
            )
        };
        AdamWPlanTopologyAllocations {
            main: (
                Arc::as_ptr(&self.inner.capture) as usize,
                Arc::as_ptr(&self.inner.recurrent_capture.execution_plan) as usize,
            ),
            accumulation: self
                .inner
                .accumulation
                .as_ref()
                .map(|plan| phase(plan.phase())),
            partial_flush: self.partial_flush.as_ref().map(|plan| phase(plan.phase())),
            zero_grad: self.zero_grad.as_ref().map(|plan| phase(plan.phase())),
            evaluation: self.evaluation.as_ref().map(|plan| {
                (
                    Arc::as_ptr(&plan.inference.capture) as usize,
                    Arc::as_ptr(&plan.inference.execution_plan) as usize,
                )
            }),
        }
    }

    /// Compatibility constructor that compiles an exact program and then
    /// restores its portable AdamW frontier before runtime preparation.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        if !config.frozen_parameters.is_empty() {
            return Err(training(
                "compiled AdamW raw parameters cannot resolve frozen parameter names",
            ));
        }
        let decoded = checkpoint.decoded();
        let parameters = decoded
            .parameters
            .iter()
            .map(|(name, value)| TrainingParameterInit::new(name.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()?;
        Self::compile(config, parameters, build)?.restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor that compiles a module-bound program and then
    /// restores its portable frontier without preparing a runtime.
    pub fn compile_module_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
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
        Self::compile_module(config, module, build)?.restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor that compiles the explicit residual-dropout
    /// program and then restores its optimizer and Threefry-counter frontier.
    pub fn compile_module_with_dropout_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_module_with_dropout(config, dropout, module, build)?
            .restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor for a compiler-owned token-mean loss that
    /// restores its portable optimizer and Threefry-counter frontier.
    pub fn compile_token_mean_module_with_dropout_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_token_mean_module_with_dropout(config, dropout, module, build)?
            .restore_checkpoint(checkpoint)
    }

    /// Prepares graph-free CPU replay from this plan's exact frontier.
    pub fn prepare_cpu(&self) -> Result<CpuCompiledAdamW> {
        self.prepare_cpu_with_non_finite_policy(CpuNonFinitePolicy::Propagate)
    }

    fn prepare_cpu_with_non_finite_policy(
        &self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> Result<CpuCompiledAdamW> {
        validate_adamw_observation_schema(
            &self.inner.phase_outputs.observations,
            self.contract.clip_report,
            self.contract.window_loss_report,
        )?;
        if let Some(partial_flush) = &self.partial_flush {
            partial_flush.outputs.validate_report_flags(
                self.contract.clip_report,
                self.contract.window_loss_report,
            )?;
        }
        if let Some(zero_grad) = &self.zero_grad {
            zero_grad.outputs.validate_report_flags(false, false)?;
        }
        Ok(CpuCompiledAdamW {
            inner: self
                .inner
                .prepare_cpu_with_non_finite_policy(non_finite_policy)?,
            partial_flush: self.partial_flush.clone(),
            zero_grad: self.zero_grad.clone(),
            contract: self.contract.clone(),
            progress: self.progress,
            evaluation: self
                .evaluation
                .clone()
                .map(|plan| CpuCompiledEvaluation { plan }),
            non_finite_policy,
        })
    }

    /// Prepares strict-native CPU replay and compiles every attached pure
    /// program before exposing mutable session state.
    pub fn prepare_native_cpu<'a>(
        &self,
        target: &NativeCpuSessionTarget<'a>,
    ) -> Result<NativeCpuCompiledAdamW<'a>> {
        let inner = self.prepare_cpu_with_non_finite_policy(target.non_finite_policy())?;
        NativeCpuCompiledAdamW::prepare(inner, target.executor(), target.is_vectorized())
    }

    /// Prepares this authenticated plan through a concrete session target.
    ///
    /// The target's associated session and error keep backend-specific
    /// diagnostics statically available without a runtime backend enum or CPU
    /// fallback.
    pub fn prepare<'a, T>(
        &'a self,
        target: &T,
    ) -> std::result::Result<
        <T as SessionTarget<&'a Self>>::Session,
        <T as SessionTarget<&'a Self>>::Error,
    >
    where
        T: SessionTarget<&'a Self>,
    {
        target.prepare(self)
    }

    /// Renders the compiled program for strict Metal admission without
    /// creating device resources.
    pub fn metal_plan(&self, renderer: MetalRenderer) -> Result<MetalCompiledAdamWPlan> {
        let contract = self.contract.metal()?;
        let inner = self.inner.metal_plan(
            renderer.clone(),
            &self.contract.host_token_inputs,
            self.evaluation.clone(),
        )?;
        let partial_flush = self
            .partial_flush
            .as_ref()
            .map(|transition| {
                let initial_state = transition
                    .state_input_keys
                    .iter()
                    .map(|(input, key)| {
                        let value = self.inner.state_values.get(key).cloned().ok_or_else(|| {
                            training("compiled partial-flush state value is absent")
                        })?;
                        Ok((input.clone(), value))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                MetalFixedStateTransitionPlan::new(
                    transition
                        .phase()
                        .recurrent_capture
                        .stateful(initial_state)?,
                    renderer,
                    &inner.inner,
                )
                .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledAdamWPlan {
            inner,
            accumulation_capture_identity: self.accumulation_capture_identity(),
            partial_flush,
            progress: self.progress,
            flush_capture_identity: self.flush_capture_identity(),
            contract,
        })
    }

    pub fn capture_identity(&self) -> u64 {
        self.program_identity
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.contract.gradient_accumulation_steps
    }

    pub fn token_weighted_gradient_accumulation_mask(&self) -> Option<&str> {
        match &self.contract.token_weight_policy {
            Some(CompiledTokenWeightPolicy::ExplicitMask(name)) => Some(name),
            _ => None,
        }
    }

    /// I32 target input and sentinel used for compiler-owned token weighting.
    pub fn token_weighted_ignore_index(&self) -> Option<(&str, i32)> {
        match &self.contract.token_weight_policy {
            Some(CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            }) => Some((target_input, *value)),
            _ => None,
        }
    }

    pub fn zero_valid_token_microbatches_enabled(&self) -> bool {
        self.contract.allow_zero_valid_token_microbatches
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.contract.max_gradient_norm
    }

    pub fn clip_report_enabled(&self) -> bool {
        self.contract.clip_report
    }

    /// Whether completed-window loss aggregation is captured and reported.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.contract.window_loss_report
    }

    pub fn loss_scale(&self) -> f32 {
        self.contract.loss_scale
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        match &self.contract.learning_rate {
            CompiledLearningRatePolicy::External => None,
            CompiledLearningRatePolicy::MultiStep(schedule) => Some(schedule),
        }
    }

    /// Returns immutable logical work and recurrent-state facts without
    /// preparing a runtime or exposing the raw mixed capture.
    pub fn inspection(&self) -> Result<CompiledAdamWInspection> {
        let recurrent_state = checked_recurrent_state_extent(
            self.inner
                .state_values
                .values()
                .map(checked_bytes)
                .collect::<Result<Vec<_>>>()?,
        )?;
        let main = (
            self.capture_identity(),
            self.inner.recurrent_capture.execution_plan().clone(),
        );
        let accumulation = self.inner.accumulation.as_ref().map(|transition| {
            (
                transition.phase().capture_identity,
                transition
                    .phase()
                    .recurrent_capture
                    .execution_plan()
                    .clone(),
            )
        });
        let partial_flush = self.partial_flush.as_ref().map(|transition| {
            (
                transition.capture_identity(),
                transition
                    .phase()
                    .recurrent_capture
                    .execution_plan()
                    .clone(),
            )
        });
        let zero_grad = self.zero_grad.as_ref().map(|transition| {
            (
                transition.capture_identity(),
                transition
                    .phase()
                    .recurrent_capture
                    .execution_plan()
                    .clone(),
            )
        });
        let evaluation = self.evaluation.as_ref().map(|evaluation| {
            (
                evaluation.capture_identity,
                evaluation.inference.execution_plan().clone(),
            )
        });
        Ok(CompiledAdamWInspection::new(
            self.step_count(),
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
            recurrent_state,
        )
        .with_compile_phases(self.compile_phases.clone()))
    }

    /// Backend-neutral graph/autograd/lowering/capture observations retained
    /// by the freshly compiled plan. Restored runtime snapshots deliberately
    /// carry no synthetic compilation evidence.
    pub fn compile_phases(&self) -> Option<&CompiledTrainingCompileObservation> {
        self.compile_phases.as_ref()
    }

    /// Returns the explicit compiled dropout policy, when present.
    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.contract.dropout.map(|dropout| dropout.config)
    }

    /// Number of Threefry U64 blocks reserved by each successful replay.
    pub fn dropout_blocks_per_replay(&self) -> Option<u64> {
        self.contract
            .dropout
            .map(|dropout| dropout.blocks_per_replay)
    }

    /// Stable identity of the private accumulation-only sibling capture.
    pub fn accumulation_capture_identity(&self) -> Option<u64> {
        self.inner
            .accumulation
            .as_ref()
            .map(|transition| transition.phase().capture_identity)
    }

    /// Stable identity of the state-only flush capture, when accumulation is
    /// enabled.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.partial_flush
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    /// Stable identity of the captured state-only accumulation reset.
    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.zero_grad
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }
}

impl<M: Module> CompiledModuleAdamWPlan<M> {
    fn build_owned<F>(
        module: M,
        frozen_parameters: &BTreeSet<String>,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M) -> Result<CompiledAdamWPlan>,
    {
        let result: Result<(CompiledAdamWPlan, CompiledModuleSeal)> = (|| {
            let seal = CompiledModuleSeal::capture(&module, frozen_parameters)?;
            let plan = build(&module)?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal))
        })();
        match result {
            Ok((plan, seal)) => Ok(Self {
                module,
                plan,
                seal,
                required_evaluation_capture_identity: None,
            }),
            Err(source) => Err(CompiledModuleAdamWCompileError { module, source }),
        }
    }

    fn build_owned_from_module_checkpoint<F>(
        config: &CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, ModuleParameterPlan) -> Result<CompiledAdamWPlan>,
    {
        let result: Result<(CompiledAdamWPlan, CompiledModuleSeal, Option<u64>)> = (|| {
            let decoded = checkpoint.decoded();
            let required_evaluation_capture_identity = decoded.evaluation_capture_identity;
            let mut seal = CompiledModuleSeal::capture(&module, &config.frozen_parameters)?;
            let immutable_values = seal.apply_module_checkpoint(decoded)?;
            let parameter_plan = ModuleParameterPlan::new(&module, &config.frozen_parameters)?
                .with_immutable_values(&immutable_values)?;
            let plan = build(&module, parameter_plan)?
                .restore_checkpoint(checkpoint.optimizer_checkpoint())?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal, required_evaluation_capture_identity))
        })();
        match result {
            Ok((plan, seal, required_evaluation_capture_identity)) => Ok(Self {
                module,
                plan,
                seal,
                required_evaluation_capture_identity,
            }),
            Err(source) => Err(CompiledModuleAdamWCompileError { module, source }),
        }
    }

    fn authenticate_restored_evaluation(&self, evaluation: &CompiledEvaluationPlan) -> Result<()> {
        if let Some(expected) = self.required_evaluation_capture_identity
            && evaluation.capture_identity != expected
        {
            return Err(training(
                "compiled module checkpoint evaluation capture identity mismatch",
            ));
        }
        Ok(())
    }

    fn validate_ready_for_preparation(&self) -> Result<()> {
        self.seal.validate_unchanged(&self.module)?;
        if self.required_evaluation_capture_identity.is_some() {
            return Err(training(
                "compiled module checkpoint requires its authenticated evaluation capture",
            ));
        }
        Ok(())
    }

    /// Compiles AdamW from, and takes ownership of, one exact module value.
    pub fn compile<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module(config, module, build)
        })
    }

    /// Compiles the explicit recurrent-dropout workload while taking ownership
    /// of its exact module value.
    pub fn compile_with_dropout<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_with_dropout(config, dropout, module, build)
        })
    }

    /// Compiles and owns a module through the unified explicit objective
    /// facade.
    pub fn compile_graph<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph(config, module, build)
        })
    }

    /// Owned-module counterpart of
    /// [`CompiledAdamWPlan::compile_module_graph_with_ignore_index`].
    pub fn compile_graph_with_ignore_index<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph_with_ignore_index(config, module, build)
        })
    }

    /// Compiles and owns a recurrent-dropout module through the unified
    /// explicit objective facade.
    pub fn compile_graph_with_dropout<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph_with_dropout(config, dropout, module, build)
        })
    }

    /// Owned-module dropout counterpart of
    /// [`CompiledAdamWPlan::compile_module_graph_with_dropout_and_ignore_index`].
    pub fn compile_graph_with_dropout_and_ignore_index<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph_with_dropout_and_ignore_index(
                config, dropout, module, build,
            )
        })
    }

    /// Recompiles a unified owned module program from a complete module
    /// checkpoint without first mutating the destination module.
    ///
    /// Saved frozen parameters and buffers are used as capture constants.
    /// Destination topology, ties, kinds, and source trainability must match;
    /// optimizer and immutable values are published together only by finish. A
    /// v2 checkpoint carrying an evaluator identity must attach that exact
    /// evaluator before target preparation.
    pub fn compile_graph_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                CompiledAdamWPlan::compile_module_graph_parameters(
                    objective_config,
                    module,
                    parameter_plan,
                    build,
                )
            },
        )
    }

    /// Ignore-index-context counterpart of
    /// [`Self::compile_graph_from_module_checkpoint`].
    pub fn compile_graph_with_ignore_index_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                CompiledAdamWPlan::compile_module_graph_with_ignore_index_parameters(
                    objective_config,
                    module,
                    parameter_plan,
                    build,
                )
            },
        )
    }

    /// Recompiles a recurrent-dropout unified owned module program from a
    /// complete module checkpoint without mutating the destination module.
    /// A v2 checkpoint carrying an evaluator identity must attach that exact
    /// evaluator before target preparation.
    pub fn compile_graph_with_dropout_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                let parameters = parameter_plan.initial_parameters()?;
                let lower_config = objective_config.clone();
                CompiledAdamWPlan::compile_module_with_dropout_parameters(
                    objective_config,
                    dropout,
                    module,
                    parameter_plan,
                    parameters,
                    move |module, graph, inputs, dropout| {
                        let built = build(module, graph, inputs, dropout)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective(
                            &lower_config,
                            graph,
                            inputs,
                            objective,
                        )?;
                        Ok((loss, outputs))
                    },
                )
            },
        )
    }

    /// Ignore-index-context counterpart of
    /// [`Self::compile_graph_with_dropout_from_module_checkpoint`].
    pub fn compile_graph_with_dropout_and_ignore_index_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                let parameters = parameter_plan.initial_parameters()?;
                let lower_config = objective_config.clone();
                CompiledAdamWPlan::compile_module_with_dropout_parameters_inner(
                    objective_config,
                    dropout,
                    module,
                    parameter_plan,
                    parameters,
                    move |module, graph, inputs, dropout| {
                        let nodes = lower_ignore_index_nodes(&lower_config, graph, inputs)?;
                        let built = build(module, graph, inputs, nodes, dropout)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective_with_ignore_index_nodes(
                            &lower_config,
                            graph,
                            objective,
                            nodes,
                        )?;
                        Ok((loss, outputs, Some(nodes.weight)))
                    },
                )
            },
        )
    }

    /// Compatibility constructor that compiles an owned module program and
    /// then restores its authenticated AdamW frontier before preparation.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile(config, module, build).and_then(|plan| {
            plan.restore_checkpoint(checkpoint).map_err(|error| {
                let (plan, source) = error.into_parts();
                CompiledModuleAdamWCompileError {
                    module: plan.module,
                    source,
                }
            })
        })
    }

    /// Compatibility constructor that compiles the owned recurrent-dropout
    /// workload and then restores its complete optimizer/dropout frontier.
    pub fn compile_with_dropout_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_with_dropout(config, dropout, module, build).and_then(|plan| {
            plan.restore_checkpoint(checkpoint).map_err(|error| {
                let (plan, source) = error.into_parts();
                CompiledModuleAdamWCompileError {
                    module: plan.module,
                    source,
                }
            })
        })
    }

    /// Restores a checkpoint onto this already compiled owned program without
    /// rebuilding the Graph, derivatives, schedules, captures, partial-flush
    /// transition, or attached evaluation program.
    ///
    /// The returned owner contains an independent restored plan while keeping
    /// the same sealed module value. Failure retains this complete owner for
    /// inspection, retry, or preparation of its unchanged frontier.
    pub fn restore_checkpoint(
        mut self,
        checkpoint: &CompiledAdamWCheckpoint,
    ) -> std::result::Result<Self, CompiledModuleAdamWRestoreError<M>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleAdamWRestoreError {
                plan: Box::new(self),
                source,
            });
        }
        match self.plan.restore_checkpoint(checkpoint) {
            Ok(plan) => {
                self.plan = plan;
                Ok(self)
            }
            Err(source) => Err(CompiledModuleAdamWRestoreError {
                plan: Box::new(self),
                source,
            }),
        }
    }

    /// Attaches one read-only evaluation capture to this exact owned plan.
    /// The evaluator reuses the training input schema and live canonical
    /// trainable frontier; frozen parameters and buffers remain capture-owned
    /// constants. When fresh-module restoration requires a v2-authenticated
    /// evaluator, a different capture identity rejects without consuming the
    /// plan. Failure retains the unconsumed plan for retry or recovery.
    pub fn with_evaluation<F>(
        mut self,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWEvaluationError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let result = (|| {
            if self.plan.evaluation.is_some() {
                return Err(training("compiled evaluation is already attached"));
            }
            self.seal.validate_unchanged(&self.module)?;
            let parameter_plan = self.seal.parameter_plan(&self.module)?;
            let started = Instant::now();
            let evaluation = CompiledEvaluationPlan::compile_with_parameter_plan(
                &self.module,
                &self.plan,
                parameter_plan,
                build,
            )?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                evaluation.inference.execution_plan().schedule_item_count,
            );
            self.seal.validate_unchanged(&self.module)?;
            self.authenticate_restored_evaluation(&evaluation)?;
            Ok((evaluation, phase))
        })();
        match result {
            Ok((evaluation, phase)) => {
                self.plan.evaluation = Some(evaluation);
                if let Some(observation) = &mut self.plan.compile_phases {
                    observation.set_evaluation(phase);
                }
                self.required_evaluation_capture_identity = None;
                Ok(self)
            }
            Err(source) => Err(CompiledModuleAdamWEvaluationError {
                plan: Box::new(self),
                source,
            }),
        }
    }

    /// Attaches a read-only evaluator using the training plan's authenticated
    /// scalar or compiler-owned token-mean objective policy.
    ///
    /// Token-mean evaluation reuses the configured F32 mask input, owns masked
    /// normalization inside the captured graph, and reports the exact validated
    /// token count for weighted aggregation. The zero-token training opt-in also
    /// admits an empty evaluation batch with exact-zero loss and weight. Invalid
    /// masks fail before replay. The legacy [`Self::with_evaluation`] scalar
    /// surface remains behavior-compatible, including on token-weighted plans.
    /// A v2-authenticated fresh-module restore accepts only the saved evaluator
    /// identity and retains the plan for retry on mismatch.
    pub fn with_evaluation_graph<F>(
        mut self,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWEvaluationError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let result = (|| {
            if self.plan.evaluation.is_some() {
                return Err(training("compiled evaluation is already attached"));
            }
            self.seal.validate_unchanged(&self.module)?;
            let parameter_plan = self.seal.parameter_plan(&self.module)?;
            let started = Instant::now();
            let evaluation = CompiledEvaluationPlan::compile_graph_with_parameter_plan(
                &self.module,
                &self.plan,
                parameter_plan,
                build,
            )?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                evaluation.inference.execution_plan().schedule_item_count,
            );
            self.seal.validate_unchanged(&self.module)?;
            self.authenticate_restored_evaluation(&evaluation)?;
            Ok((evaluation, phase))
        })();
        match result {
            Ok((evaluation, phase)) => {
                self.plan.evaluation = Some(evaluation);
                if let Some(observation) = &mut self.plan.compile_phases {
                    observation.set_evaluation(phase);
                }
                self.required_evaluation_capture_identity = None;
                Ok(self)
            }
            Err(source) => Err(CompiledModuleAdamWEvaluationError {
                plan: Box::new(self),
                source,
            }),
        }
    }

    /// Attaches a read-only token-mean evaluator while exposing the exact
    /// compiler-owned ignore-index nodes to its graph builder.
    pub fn with_evaluation_graph_and_ignore_index<F>(
        mut self,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWEvaluationError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        let result = (|| {
            if self.plan.evaluation.is_some() {
                return Err(training("compiled evaluation is already attached"));
            }
            self.seal.validate_unchanged(&self.module)?;
            let parameter_plan = self.seal.parameter_plan(&self.module)?;
            let started = Instant::now();
            let evaluation =
                CompiledEvaluationPlan::compile_graph_with_ignore_index_parameter_plan(
                    &self.module,
                    &self.plan,
                    parameter_plan,
                    build,
                )?;
            let phase = CompiledTrainingCompilePhaseObservation::schedule(
                started.elapsed(),
                evaluation.inference.execution_plan().schedule_item_count,
            );
            self.seal.validate_unchanged(&self.module)?;
            self.authenticate_restored_evaluation(&evaluation)?;
            Ok((evaluation, phase))
        })();
        match result {
            Ok((evaluation, phase)) => {
                self.plan.evaluation = Some(evaluation);
                if let Some(observation) = &mut self.plan.compile_phases {
                    observation.set_evaluation(phase);
                }
                self.required_evaluation_capture_identity = None;
                Ok(self)
            }
            Err(source) => Err(CompiledModuleAdamWEvaluationError {
                plan: Box::new(self),
                source,
            }),
        }
    }

    /// Consumes this owner into a target-specific session. A preparation error
    /// retains the complete plan and module for inspection or retry.
    pub fn prepare<T>(
        self,
        target: &T,
    ) -> std::result::Result<<T as SessionTarget<Self>>::Session, <T as SessionTarget<Self>>::Error>
    where
        T: SessionTarget<Self>,
    {
        target.prepare(self)
    }

    pub fn capture_identity(&self) -> u64 {
        self.plan.capture_identity()
    }

    /// Returns the private accumulation-only sibling capture identity when
    /// gradient accumulation is enabled for this owned plan.
    pub fn accumulation_capture_identity(&self) -> Option<u64> {
        self.plan.accumulation_capture_identity()
    }

    /// Returns the owned plan's immutable logical work and recurrent-state
    /// inspection without exposing its sealed module.
    pub fn inspection(&self) -> Result<CompiledAdamWInspection> {
        self.plan.inspection()
    }

    pub fn step_count(&self) -> u64 {
        self.plan.step_count()
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.plan.flush_capture_identity()
    }

    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.plan.dropout_config()
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.plan.captured_multi_step_lr()
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.plan
            .evaluation
            .as_ref()
            .map(|evaluation| evaluation.capture_identity)
    }

    /// Inspects strict Metal admission without exposing an independently
    /// preparable runtime path or releasing the owned module. Resource
    /// preparation still consumes this owner through [`Self::prepare`].
    pub fn metal_summary(&self, renderer: MetalRenderer) -> Result<MetalDeviceSessionSummary> {
        Ok(self.plan.metal_plan(renderer)?.summary().clone())
    }
}

impl<M, R> CompiledModuleTrainingSession<M, R> {
    /// Read-only access to optimizer- or backend-specific diagnostics while the
    /// neutral owner retains exclusive control of module publication.
    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    /// Discards the compiled runtime frontier and returns the sealed host
    /// module without publishing any trained parameter values.
    pub fn into_module_without_publication(self) -> M {
        self.module
    }
}

impl<M: Module> CompiledModuleTrainingSession<M, CpuCompiledMomentumSgd> {
    fn compile_owned_momentum<F>(
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(&M) -> Result<CpuCompiledMomentumSgd>,
    {
        let result: Result<(CpuCompiledMomentumSgd, CompiledModuleSeal)> = (|| {
            let seal = CompiledModuleSeal::capture(&module, &BTreeSet::new())?;
            let runtime = build(&module)?;
            seal.validate_unchanged(&module)?;
            Ok((runtime, seal))
        })();
        match result {
            Ok((runtime, seal)) => Ok(Self {
                module,
                runtime,
                seal,
                evaluation_capture_identity: None,
            }),
            Err(source) => Err(CompiledModuleMomentumSgdCompileError { module, source }),
        }
    }

    /// Compiles CPU momentum-SGD while taking exclusive ownership of the
    /// module for the complete replay and publication lifecycle.
    pub fn compile_momentum_sgd<F>(
        config: CompiledMomentumSgdConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_owned_momentum(module, |module| {
            CpuCompiledMomentumSgd::compile_module(config, module, build)
        })
    }

    /// Compiles against a fresh module identity and restores an authenticated
    /// parameter/momentum frontier before returning the owned session.
    pub fn compile_momentum_sgd_from_checkpoint<F>(
        config: CompiledMomentumSgdConfig,
        module: M,
        checkpoint: &CompiledMomentumSgdCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_owned_momentum(module, |module| {
            CpuCompiledMomentumSgd::compile_module_from_checkpoint(
                config, module, checkpoint, build,
            )
        })
    }
}

impl<M, R> CompiledModuleAdamWSession<M, R> {
    /// Read-only access to the retained AdamW runtime for diagnostics.
    pub fn runtime(&self) -> &R {
        self.training.runtime()
    }

    /// Removes the AdamW compatibility surface while retaining the exact owned
    /// module, runtime, seal, and prepared-session frontier.
    pub fn into_training_session(self) -> CompiledModuleTrainingSession<M, R> {
        self.training
    }

    pub fn into_module_without_publication(self) -> M {
        self.training.into_module_without_publication()
    }
}

impl<M: Module, R: CompiledScheduledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.training.runtime.captured_multi_step_lr()
    }

    pub fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<R::Step> {
        self.training.runtime.step_scheduled(inputs)
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training.runtime.step_batch_scheduled(batch)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<R::ScheduledFlush> {
        self.training.runtime.flush_partial_window_scheduled()
    }
}

impl<M: Module, R: CompiledAdamWCommitOnlyRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<R::Step> {
        self.training
            .runtime
            .step_commit_only(inputs, learning_rate)
    }

    pub fn step_batch_commit_only<B>(&mut self, batch: B, learning_rate: f32) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training
            .runtime
            .step_batch_commit_only(batch, learning_rate)
    }
}

impl<M: Module, R: CompiledTrainingCommitOnlyRuntime> CompiledModuleTrainingSession<M, R> {
    pub fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<R::Step> {
        self.runtime.commit_step(inputs, learning_rate)
    }

    pub fn commit_step_batch<B>(&mut self, batch: B, learning_rate: f32) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.runtime.commit_step_batch(batch, learning_rate)
    }
}

impl<M: Module, R: CompiledTrainingCommitOnlyRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<R::Step> {
        self.training.runtime.commit_step(inputs, learning_rate)
    }

    pub fn commit_step_batch<B>(&mut self, batch: B, learning_rate: f32) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training
            .runtime
            .commit_step_batch(batch, learning_rate)
    }
}

impl<M: Module, R: CompiledScheduledAdamWCommitOnlyRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<R::Step> {
        self.training.runtime.step_commit_only_scheduled(inputs)
    }

    pub fn step_batch_commit_only_scheduled<B>(&mut self, batch: B) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.training
            .runtime
            .step_batch_commit_only_scheduled(batch)
    }
}

impl<M: Module, R: CompiledTrainingRuntime> CompiledModuleTrainingSession<M, R> {
    /// Atomically publishes the runtime's exact trainable frontier and returns
    /// the owned module. The complete module topology, identities, versions,
    /// descriptors, and frozen/buffer bytes must still match the compile seal.
    /// A failure retains the intact session and can be recovered with
    /// [`CompiledModuleTrainingFinishError::into_session`].
    pub fn finish(self) -> std::result::Result<M, CompiledModuleTrainingFinishError<M, R>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let parameters = match self.runtime.parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { module, .. } = self;
        Ok(module)
    }
}

impl<M: Module, R> CompiledModuleTrainingSession<M, R>
where
    R: CompiledCheckpointRuntime,
    R::Checkpoint: CompiledCheckpointParameterSnapshot,
{
    /// Atomically publishes and returns the exact checkpointed training
    /// frontier.
    ///
    /// The module seal is validated before snapshot work. One coherent
    /// checkpoint snapshot supplies both the returned resumable state and the
    /// parameter values published into the owned module, so a backend never
    /// performs a second parameter-only read. A checkpoint, decode, or
    /// publication failure retains the intact session for inspection or retry.
    pub fn finish_with_checkpoint(
        self,
    ) -> std::result::Result<(M, R::Checkpoint), CompiledModuleTrainingFinishError<M, R>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let checkpoint = match self.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let parameters = match checkpoint.checkpoint_parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleTrainingFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleTrainingFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { module, .. } = self;
        Ok((module, checkpoint))
    }
}

impl<M: Module, R: CompiledTrainingRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn finish(self) -> std::result::Result<M, CompiledModuleAdamWFinishError<M, R>> {
        match self.training.finish() {
            Ok(module) => Ok(module),
            Err(error) => {
                let (training, source) = error.into_parts();
                Err(CompiledModuleAdamWFinishError {
                    session: Box::new(Self { training }),
                    source,
                })
            }
        }
    }
}

impl<M: Module, R: CompiledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    /// Source-compatible AdamW finalization over the optimizer-neutral owner.
    pub fn finish_with_checkpoint(
        self,
    ) -> std::result::Result<(M, CompiledAdamWCheckpoint), CompiledModuleAdamWFinishError<M, R>>
    {
        match self.training.finish_with_checkpoint() {
            Ok(finished) => Ok(finished),
            Err(error) => {
                let (training, source) = error.into_parts();
                Err(CompiledModuleAdamWFinishError {
                    session: Box::new(Self { training }),
                    source,
                })
            }
        }
    }
}

impl<M: Module, R: CompiledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    /// Snapshots the exact optimizer frontier together with the owned
    /// module's canonical immutable state and topology.
    ///
    /// This does not publish into or release the sealed host module. The
    /// embedded optimizer checkpoint is reused byte-for-byte across v1--v9.
    /// Programs without the private accumulation sibling retain their existing
    /// v1--v8 formats; multi-replay accumulation emits v9. The module envelope
    /// remains v1 when no evaluator is attached and uses v2 only to authenticate
    /// an attached evaluator's capture identity.
    pub fn module_checkpoint(&self) -> Result<CompiledModuleAdamWCheckpoint> {
        self.training
            .seal
            .validate_unchanged(&self.training.module)?;
        let optimizer = self.training.runtime.checkpoint()?;
        self.training
            .seal
            .validate_unchanged(&self.training.module)?;
        let (states, visits) = self.training.seal.checkpoint_inventory();
        encode_module_adamw_checkpoint(
            &optimizer,
            self.training.evaluation_capture_identity,
            &states,
            &visits,
        )
    }

    /// Atomically publishes and returns one complete module checkpoint built
    /// from the exact AdamW snapshot used for publication.
    ///
    /// The checkpoint retains canonical module topology, ties, frozen
    /// parameters, and buffers in addition to the optimizer frontier. The
    /// runtime is checkpointed exactly once; encoding and publication both use
    /// that same snapshot. A seal, checkpoint, encoding, decode, or publication
    /// failure retains the intact session in [`CompiledModuleAdamWFinishError`]
    /// for inspection or retry.
    pub fn finish_with_module_checkpoint(
        self,
    ) -> std::result::Result<(M, CompiledModuleAdamWCheckpoint), CompiledModuleAdamWFinishError<M, R>>
    {
        if let Err(source) = self.training.seal.validate_unchanged(&self.training.module) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let optimizer = match self.training.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let (states, visits) = self.training.seal.checkpoint_inventory();
        let checkpoint = match encode_module_adamw_checkpoint(
            &optimizer,
            self.training.evaluation_capture_identity,
            &states,
            &visits,
        ) {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let parameters = match decode_adamw_checkpoint(optimizer.as_bytes()) {
            Ok(decoded) => decoded.parameters,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self
            .training
            .seal
            .publish(&self.training.module, &parameters)
        {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { training } = self;
        let CompiledModuleTrainingSession { module, .. } = training;
        Ok((module, checkpoint))
    }
}

impl<M> CompiledModuleAdamWSession<M, CpuCompiledAdamW> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.training.runtime.dropout_block_counter()
    }
}

impl<'a, M> CompiledModuleAdamWSession<M, NativeCpuCompiledAdamW<'a>> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.training.runtime.dropout_block_counter()
    }

    /// Returns strict-native CPU preparation evidence without exposing the
    /// sealed module or mutable runtime internals.
    pub fn native_cpu_preparation_report(&self) -> &NativeCpuCompiledAdamWPreparationReport {
        self.training.runtime.preparation_report()
    }
}

impl<M: Module> CompiledModuleAdamWSession<M, MetalCompiledAdamW> {
    /// Strict Metal replay that commits the complete device state frontier
    /// without downloading loss or named outputs.
    pub fn step_without_host_outputs(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWCommitResult> {
        self.training
            .runtime
            .step_without_host_outputs(inputs, learning_rate)
    }

    /// Returns the sealed runtime's read-only Metal session evidence without
    /// exposing the owned module or mutable backend internals.
    pub fn metal_session(&self) -> &MetalDeviceSession {
        self.training.runtime.metal_session()
    }

    /// Returns the opt-in successful-step recorder attached during target
    /// preparation, when present.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.training.runtime.execution_scoreboard()
    }

    /// Snapshots the owned Metal runtime's successfully recorded prefix.
    pub fn execution_scoreboard_report(
        &self,
    ) -> std::result::Result<Option<MetalSessionScoreboardReport>, MetalScoreboardError> {
        self.training.runtime.execution_scoreboard_report()
    }

    /// Returns the first fail-soft scoreboard recording error, when recording
    /// has frozen.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.training.runtime.scoreboard_recording_error()
    }

    /// Returns preparation evidence for the two read-only active-bank
    /// evaluators, when evaluation was attached before preparation.
    pub fn evaluation_preparation_reports(
        &self,
    ) -> Option<[&crate::runtime::metal::MetalDevicePreparationReport; 2]> {
        self.training.runtime.evaluation_preparation_reports()
    }

    /// Returns deterministic resource/execution summaries for both read-only
    /// physical-bank evaluators.
    pub fn evaluation_summaries(&self) -> Option<[&MetalDeviceSessionSummary; 2]> {
        self.training.runtime.evaluation_summaries()
    }
}

impl CpuCompiledAdamW {
    fn admit_step(
        &self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<PendingAdamWStep> {
        validate_training_inputs(&self.inner.inputs, &inputs)?;
        let loss_weight = validate_token_weight(
            &inputs,
            self.contract.token_weight_policy.as_ref(),
            self.contract.allow_zero_valid_token_microbatches,
        )?;
        let next_progress = adamw_window_progress(
            self.progress
                .advance_replay(self.contract.gradient_accumulation_steps),
        )?;
        self.validate_completed_token_window(next_progress, loss_weight)?;
        if let Some(dropout) = self.contract.dropout {
            expected_dropout_counter(dropout, next_progress.replay_step)?;
        }
        Ok(PendingAdamWStep {
            request: CompiledStepReplayRequest {
                inputs,
                learning_rate,
                non_finite_policy: self.non_finite_policy,
                output_selection: selection,
                injected_failure,
            },
            next_progress,
            loss_weight,
        })
    }

    fn publish_step(
        &mut self,
        next_progress: CompiledTrainingWindowProgress,
        loss_weight: u64,
        mut result: CompiledTrainingStepResult,
    ) -> CompiledAdamWStepResult {
        result.step = next_progress.replay_step;
        self.progress = next_progress;
        adamw_step_result(
            result,
            next_progress,
            loss_weight,
            self.contract.gradient_accumulation_steps,
            self.contract.clip_report,
            self.contract.window_loss_report,
        )
    }

    fn validate_completed_token_window(
        &self,
        next: CompiledTrainingWindowProgress,
        loss_weight: u64,
    ) -> Result<()> {
        if !self.contract.allow_zero_valid_token_microbatches
            || self.contract.token_weight_policy.is_none()
            || next.accumulation_index != 0
        {
            return Ok(());
        }
        let topology = CompiledTrainingWindowTopology::from_contract(&self.contract);
        let retained = if topology.retains_token_count() {
            self.inner
                .global_snapshot(AdamWGlobalState::AccumulatedTokenCount)?
                .scalar_at(0)
                .as_u64()
        } else {
            0
        };
        let total = retained
            .checked_add(loss_weight)
            .ok_or_else(|| training("compiled AdamW completed token count overflows"))?;
        if total == 0 {
            return Err(training(
                "compiled AdamW completed token window must contain at least one valid token",
            ));
        }
        Ok(())
    }

    fn validate_partial_token_window(&self) -> Result<()> {
        if !self.contract.allow_zero_valid_token_microbatches
            || self.contract.token_weight_policy.is_none()
        {
            return Ok(());
        }
        let retained = self
            .inner
            .global_snapshot(AdamWGlobalState::AccumulatedTokenCount)?
            .scalar_at(0)
            .as_u64();
        if retained == 0 {
            return Err(training(
                "compiled AdamW partial token window must contain at least one valid token",
            ));
        }
        Ok(())
    }

    pub fn compile<F>(
        config: CompiledAdamWConfig,
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
        CompiledAdamWPlan::compile(config, parameters, build)?.prepare_cpu()
    }

    /// Compiles an ordinary module forward against optimizer-owned recurrent
    /// parameter state.
    ///
    /// The builder receives only declared batch inputs: calls to
    /// [`crate::nn::Parameter::bind`] inside `module` resolve automatically to
    /// the compiled state frontier. Frozen parameters and buffers are captured
    /// as immutable constants, while tied parameter handles share one graph
    /// node and one AdamW state tuple. Names selected by
    /// [`CompiledAdamWConfig::with_frozen_parameters`] receive that same
    /// constant treatment without changing their host trainable flags.
    pub fn compile_module<M, F>(config: CompiledAdamWConfig, module: &M, build: F) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile_module(config, module, build)?.prepare_cpu()
    }

    /// Recompiles an exact program and restores its saved recurrent frontier.
    /// The build/configuration must reproduce the checkpoint's capture
    /// identity; all state is validated before the fresh runtime is replaced.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile_from_checkpoint(config, checkpoint, build)?.prepare_cpu()
    }

    /// Recompiles a module-bound program and restores its exact AdamW state.
    /// The module topology, frozen values, builder, and input descriptors must
    /// reproduce the authenticated capture identity before any restored state
    /// becomes visible.
    pub fn compile_module_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
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
        CompiledAdamWPlan::compile_module_from_checkpoint(config, module, checkpoint, build)?
            .prepare_cpu()
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            None,
        )
    }

    /// Commits one external-rate replay while omitting graph-named outputs.
    ///
    /// Loss and enabled clip/window reports remain captured, validated, and
    /// returned. Only the user-named output range is excluded from CPU egress;
    /// recurrent state and checkpoint identity are unchanged.
    pub fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            None,
        )
    }

    /// Replays one batch using the immutable MultiStep rate captured in the
    /// program. This method accepts no host learning-rate value.
    pub fn step_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::All, None)
    }

    /// Scheduled-rate counterpart of [`Self::step_commit_only`].
    pub fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledAdamWStepResult> {
        self.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::CommitOnly, None)
    }

    fn step_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        let PendingAdamWStep {
            request,
            next_progress,
            loss_weight,
        } = self.admit_step(inputs, learning_rate, selection, injected_failure)?;
        let result = if next_progress.accumulation_index == 0 {
            self.inner.step_inner_with_learning_rate(request, true)?
        } else {
            let transition = self
                .inner
                .accumulation
                .clone()
                .ok_or_else(|| training("compiled accumulation replay is absent"))?;
            self.inner
                .step_accumulation_inner_with_learning_rate(&transition, request)?
        };
        Ok(self.publish_step(next_progress, loss_weight, result))
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<CompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    pub fn step_batch_commit_only<B>(
        &mut self,
        batch: B,
        learning_rate: f32,
    ) -> Result<CompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }

    pub fn step_batch_commit_only_scheduled<B>(
        &mut self,
        batch: B,
    ) -> Result<CompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only_scheduled(batch.into_compiled_inputs()?)
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledEvaluationResult> {
        let evaluation = self
            .evaluation
            .as_ref()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        evaluation
            .plan
            .evaluate(inputs, self.inner.parameter_snapshots()?)
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation
            .as_ref()
            .map(|evaluation| evaluation.plan.capture_identity)
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.contract.gradient_accumulation_steps
    }

    pub fn token_weighted_gradient_accumulation_mask(&self) -> Option<&str> {
        match &self.contract.token_weight_policy {
            Some(CompiledTokenWeightPolicy::ExplicitMask(name)) => Some(name),
            _ => None,
        }
    }

    /// I32 target input and sentinel used for compiler-owned token weighting.
    pub fn token_weighted_ignore_index(&self) -> Option<(&str, i32)> {
        match &self.contract.token_weight_policy {
            Some(CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            }) => Some((target_input, *value)),
            _ => None,
        }
    }

    pub fn zero_valid_token_microbatches_enabled(&self) -> bool {
        self.contract.allow_zero_valid_token_microbatches
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.contract.max_gradient_norm
    }

    pub fn clip_report_enabled(&self) -> bool {
        self.contract.clip_report
    }

    /// Whether completed-window loss aggregation is captured and reported.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.contract.window_loss_report
    }

    pub fn loss_scale(&self) -> f32 {
        self.contract.loss_scale
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        match &self.contract.learning_rate {
            CompiledLearningRatePolicy::External => None,
            CompiledLearningRatePolicy::MultiStep(schedule) => Some(schedule),
        }
    }

    /// CPU-only admission policy selected when this runtime was prepared.
    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.non_finite_policy
    }

    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.contract
            .dropout
            .map(|_| {
                Ok(self
                    .inner
                    .workload_snapshot(&RecurrentStateKey::dropout_counter())?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        Ok(self.progress.optimizer_step)
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::SecondMoment)
    }

    /// Partial F32 gradient sums retained between microbatches. The map is
    /// empty when accumulation is disabled (`steps == 1`).
    pub fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::GradientAccumulator)
    }

    /// Number of microbatches currently retained toward the next update.
    pub fn accumulation_index(&self) -> Result<u64> {
        Ok(self.progress.accumulation_index)
    }

    /// Atomically clears a retained partial accumulation window. Parameters,
    /// moments, optimizer progress, and successful replay count are preserved.
    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_inner(None)
    }

    fn zero_grad_inner(
        &mut self,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWZeroGradResult> {
        let (next, reset) = adamw_window_progress(
            self.progress
                .cancel(self.contract.gradient_accumulation_steps),
        )?;
        if !reset.did_discard() {
            return Ok(reset);
        }
        let next = adamw_window_progress(next.record_reset_transition())?;
        let transition = self
            .zero_grad
            .as_ref()
            .ok_or_else(|| training("compiled AdamW zero-grad capture is absent"))?;
        let reports = self.inner.replay_auxiliary_transition(
            transition,
            None,
            CpuNonFinitePolicy::Propagate,
            injected_failure,
        )?;
        debug_assert!(reports.clip_report.is_none());
        debug_assert!(reports.window_loss.is_none());
        self.progress = next;
        Ok(reset)
    }

    #[cfg(test)]
    fn zero_grad_with_injected_failure(
        &mut self,
        injected_failure: u64,
    ) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_inner(Some(injected_failure))
    }

    /// Atomically commits a nonempty partial accumulation window through its
    /// separately authenticated state-only capture. Replay/dropout progress
    /// and all workload state remain unchanged.
    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWFlushResult> {
        self.contract.learning_rate.require_external()?;
        self.flush_partial_window_with_learning_rate(Some(learning_rate), None)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<CompiledAdamWFlushResult> {
        self.contract.learning_rate.require_scheduled()?;
        self.flush_partial_window_with_learning_rate(None, None)
    }

    fn flush_partial_window_with_learning_rate(
        &mut self,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWFlushResult> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let (next, flush) = adamw_window_progress(
            self.progress
                .flush_partial(self.contract.gradient_accumulation_steps),
        )?;
        if !flush.did_update() {
            return Ok(flush.into_adamw_result());
        }
        let mut result = flush.into_adamw_result();
        self.validate_partial_token_window()?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, self.non_finite_policy)?;
        }
        let transition = self
            .partial_flush
            .as_ref()
            .ok_or_else(|| training("compiled AdamW partial flush capture is absent"))?;
        let reports = self.inner.replay_auxiliary_transition(
            transition,
            learning_rate,
            self.non_finite_policy,
            injected_failure,
        )?;
        result.clip_report = reports.clip_report;
        result.window_loss_report = reports
            .window_loss
            .map(|value| CompiledAdamWWindowLossReport::new(value, result.flushed_microbatches));
        self.progress = next;
        Ok(result)
    }

    /// Stable identity of the state-only flush capture, when accumulation is
    /// enabled.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.partial_flush
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.zero_grad
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn first_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner
            .adamw_state_versions(AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner
            .adamw_state_versions(AdamWParameterState::SecondMoment)
    }

    fn snapshot_plan(&self) -> Result<CompiledAdamWPlan> {
        let inner = self.inner.plan()?;
        let partial_flush = self
            .partial_flush
            .clone()
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        let zero_grad = self
            .zero_grad
            .clone()
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        Ok(CompiledAdamWPlan {
            inner,
            partial_flush,
            zero_grad,
            program_identity: self.capture_identity(),
            contract: self.contract.clone(),
            progress: self.progress,
            evaluation: self
                .evaluation
                .as_ref()
                .map(|evaluation| evaluation.plan.clone()),
            compile_phases: None,
        })
    }

    fn restored_candidate(&self, checkpoint: &CompiledAdamWCheckpoint) -> Result<Self> {
        self.snapshot_plan()?
            .restore_checkpoint(checkpoint)?
            .prepare_cpu_with_non_finite_policy(self.non_finite_policy)
    }

    /// Renders the identical loss/backward/AdamW capture for Metal, seeded
    /// from this session's currently committed recurrent state. Planning is
    /// resource-free; unsupported kernels fail before a device is touched.
    pub fn metal_plan(&self, renderer: MetalRenderer) -> Result<MetalCompiledAdamWPlan> {
        self.snapshot_plan()?.metal_plan(renderer)
    }

    /// Captures parameter values, both moment sets, the graph-owned optimizer
    /// step, and the exact compiled capture identity into deterministic bytes.
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        let topology = CompiledTrainingWindowTopology::from_contract(&self.contract);
        validate_adamw_progress(self.progress, self.contract.gradient_accumulation_steps)?;
        validate_cpu_adamw_state(
            &self.inner,
            self.progress,
            self.contract.gradient_accumulation_steps,
        )?;
        let dropout_block_counter = self
            .contract
            .dropout
            .map(|dropout| {
                let counter = self
                    .inner
                    .workload_snapshot(&RecurrentStateKey::dropout_counter())?
                    .scalar_at(0)
                    .as_u64();
                if counter != expected_dropout_counter(dropout, self.progress.replay_step)? {
                    return Err(training(
                        "compiled CPU dropout counter and replay progress diverged",
                    ));
                }
                Ok(counter)
            })
            .transpose()?;
        let accumulated_token_count = topology
            .retains_token_count()
            .then_some(())
            .map(|_| {
                Ok(self
                    .inner
                    .global_snapshot(AdamWGlobalState::AccumulatedTokenCount)?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()?;
        if let (Some(policy), Some(count)) = (
            self.contract.token_weight_policy.as_ref(),
            accumulated_token_count,
        ) {
            validate_retained_token_count(
                &self.inner.inputs,
                policy,
                self.progress.accumulation_index,
                count,
                self.contract.allow_zero_valid_token_microbatches,
            )?;
        }
        let accumulated_loss_numerator = topology
            .retains_window_numerator()
            .then(|| {
                self.inner
                    .global_snapshot(AdamWGlobalState::AccumulatedLossNumerator)
            })
            .transpose()?;
        let bytes = encode_adamw_checkpoint(
            AdamWCheckpointProgress {
                capture_identity: self.capture_identity(),
                accumulation_capture_identity: self
                    .inner
                    .accumulation
                    .as_ref()
                    .map(|transition| transition.phase().capture_identity),
                replay_step: self.progress.replay_step,
                optimizer_step: self.progress.optimizer_step,
                accumulation_steps: self.contract.gradient_accumulation_steps,
                accumulation_index: self.progress.accumulation_index,
                discarded_microbatches: self.progress.discarded_microbatches,
                flushed_window_count: self.progress.flushed_window_count,
                flushed_microbatch_count: self.progress.flushed_microbatch_count,
                flush_capture_identity: self.flush_capture_identity(),
                dropout_block_counter,
                accumulated_token_count,
                window_loss_report: self.contract.window_loss_report,
                reset_transition_count: self.progress.reset_transition_count,
                reset_capture_identity: (self.progress.reset_transition_count != 0)
                    .then(|| self.zero_grad_capture_identity())
                    .flatten(),
            },
            AdamWCheckpointTensors {
                parameters: self.parameter_snapshots()?,
                first_moments: self.first_moment_snapshots()?,
                second_moments: self.second_moment_snapshots()?,
                gradient_accumulators: self.gradient_accumulator_snapshots()?,
                accumulated_loss_numerator,
            },
        )?;
        CompiledAdamWCheckpoint::from_bytes(bytes)
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            injected_failure,
        )
    }

    #[cfg(test)]
    fn step_commit_only_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            injected_failure,
        )
    }

    #[cfg(test)]
    fn flush_partial_window_inner(
        &mut self,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWFlushResult> {
        self.flush_partial_window_with_learning_rate(Some(learning_rate), injected_failure)
    }
}

impl<'a> NativeCpuCompiledAdamW<'a> {
    fn prepare(
        inner: CpuCompiledAdamW,
        executor: &'a CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<Self> {
        let external_learning_rate = matches!(
            &inner.contract.learning_rate,
            CompiledLearningRatePolicy::External
        );
        let program_count = 1
            + usize::from(inner.inner.accumulation.is_some())
            + usize::from(inner.partial_flush.is_some())
            + usize::from(inner.zero_grad.is_some())
            + usize::from(inner.evaluation.is_some());
        let mut drafts = NativeCpuTrainingProgramDrafts::new();
        let (mut main_preparation, main_residual) = inner
            .inner
            .preflight_native(vectorized, external_learning_rate)?;
        {
            let (pure, inputs) = main_preparation.pure_and_inputs();
            let draft = executor
                .preflight_native_items_with_store_groups(
                    pure,
                    inputs,
                    &inner.inner.recurrent_store_groups,
                )
                .map_err(replay_error)?;
            drafts.insert(NativeCpuTrainingProgramRole::Main, draft)?;
        }
        main_preparation.release_input_witnesses();
        let accumulation_preparation = match inner.inner.accumulation.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) = inner
                    .inner
                    .preflight_native_accumulation(transition, vectorized)?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items_with_recurrent_retention(
                            pure,
                            inputs,
                            preparation.retained_recurrent_states(),
                        )
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::Accumulation, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let partial_flush_preparation = match inner.partial_flush.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) =
                    inner.inner.preflight_native_auxiliary_transition(
                        transition.phase(),
                        vectorized,
                        external_learning_rate,
                    )?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items_with_store_groups(
                            pure,
                            inputs,
                            transition.phase().store_groups(),
                        )
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::PartialFlush, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let zero_grad_preparation = match inner.zero_grad.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) = inner
                    .inner
                    .preflight_native_auxiliary_transition(transition.phase(), vectorized, false)?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items(pure, inputs)
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::ZeroGrad, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let evaluation_preparation = match inner.evaluation.as_ref() {
            Some(evaluation) => {
                let mut preparation = evaluation.plan.preflight_native(
                    inner.parameter_snapshots()?,
                    &inner.inner.parameter_buffers,
                )?;
                let draft = executor
                    .preflight_native_items(
                        evaluation.plan.inference.capture(),
                        preparation
                            .inputs
                            .as_ref()
                            .expect("native evaluation input witnesses are present"),
                    )
                    .map_err(replay_error)?;
                drafts.insert(NativeCpuTrainingProgramRole::Evaluation, draft)?;
                preparation.inputs = None;
                Some(preparation)
            }
            None => None,
        };
        let mut programs = NativeCpuTrainingProgramBatch::with_capacity(program_count);
        programs.push(
            NativeCpuTrainingProgramRole::Main,
            main_preparation.pure(),
            drafts.take(NativeCpuTrainingProgramRole::Main)?,
        )?;
        if let Some((preparation, _)) = &accumulation_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::Accumulation,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::Accumulation)?,
            )?;
        }
        if let Some((preparation, _)) = &partial_flush_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::PartialFlush,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::PartialFlush)?,
            )?;
        }
        if let Some((preparation, _)) = &zero_grad_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::ZeroGrad,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::ZeroGrad)?,
            )?;
        }
        if let Some(evaluation) = inner.evaluation.as_ref() {
            programs.push(
                NativeCpuTrainingProgramRole::Evaluation,
                evaluation.plan.inference.capture(),
                drafts.take(NativeCpuTrainingProgramRole::Evaluation)?,
            )?;
        }
        if !drafts.is_empty() {
            return Err(training(
                "compiled native CPU planning draft inventory is excessive",
            ));
        }
        let (roles, programs) = programs.into_planning_inputs();
        let (plans, compilation) = executor
            .plan_native_item_drafts(programs, vectorized)
            .map_err(replay_error)?;
        let render_capsule_diagnostics = NativeCpuRenderCapsuleDiagnostic::from_ordered_native(
            &roles,
            compilation.render_capsule_diagnostics,
        )?;
        let NativeCpuTrainingPrograms {
            main: main_plan,
            accumulation: accumulation_plan,
            partial_flush: partial_flush_plan,
            zero_grad: zero_grad_plan,
            evaluation: evaluation_plan,
        } = NativeCpuTrainingPrograms::from_ordered(roles, plans)?;
        let main = inner
            .inner
            .finish_native(main_preparation, main_plan, main_residual)?;
        let accumulation = match (
            inner.inner.accumulation.as_ref(),
            accumulation_preparation,
            accumulation_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => Some(
                inner
                    .inner
                    .finish_native_accumulation(transition, preparation, plan, residual)?,
            ),
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU accumulation preparation differs",
                ));
            }
        };
        let partial_flush = match (
            inner.partial_flush.as_ref(),
            partial_flush_preparation,
            partial_flush_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => {
                Some(inner.inner.finish_native_auxiliary_transition(
                    transition.phase(),
                    preparation,
                    plan,
                    residual,
                )?)
            }
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU partial-flush preparation differs",
                ));
            }
        };
        let zero_grad = match (
            inner.zero_grad.as_ref(),
            zero_grad_preparation,
            zero_grad_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => {
                Some(inner.inner.finish_native_auxiliary_transition(
                    transition.phase(),
                    preparation,
                    plan,
                    residual,
                )?)
            }
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU zero-grad preparation differs",
                ));
            }
        };
        let evaluation = match (
            inner.evaluation.as_ref(),
            evaluation_preparation,
            evaluation_plan,
        ) {
            (Some(evaluation), Some(preparation), Some(plan)) => Some(
                evaluation
                    .plan
                    .finish_native(preparation, &inner.inner.parameter_buffers, plan)?,
            ),
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU evaluation preparation differs",
                ));
            }
        };
        let (recurrent_state_count, recurrent_state_bytes) = checked_recurrent_state_extent(
            inner
                .inner
                .cursor
                .frontier()
                .iter()
                .map(|state| state.bytes),
        )?;
        let PreparedNativeCpuProgram {
            report: main_report,
            replay: main_replay,
        } = main;
        let (accumulation_report, accumulation_replay) = accumulation
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let (partial_flush_report, partial_flush_replay) = partial_flush
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let (zero_grad_report, zero_grad_replay) = zero_grad
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let evaluation_report = evaluation.as_ref().map(|prepared| prepared.report.clone());
        Ok(Self {
            inner,
            executor,
            main_replay,
            accumulation_replay,
            partial_flush_replay,
            zero_grad_replay,
            evaluation_replay: evaluation,
            preparation: NativeCpuCompiledAdamWPreparationReport {
                main: main_report,
                accumulation: accumulation_report,
                partial_flush: partial_flush_report,
                zero_grad: zero_grad_report,
                evaluation: evaluation_report,
                recurrent_state_count,
                recurrent_state_bytes,
                render_capsule_hit_count: compilation.render_capsule_hit_count,
                render_capsule_miss_count: compilation.render_capsule_miss_count,
                local_render_job_count: compilation.local_render_job_count,
                parallel_render_overlap_wall_time: compilation.parallel_render_overlap_wall_time,
                max_parallel_render_job_count: compilation.max_parallel_render_job_count,
                parallel_module_overlap_wall_time: compilation.parallel_work_overlap_wall_time,
                compiler_process_overlap_wall_time: compilation.compiler_process_overlap_wall_time,
                compiler_process_count: compilation.compiler_process_count,
                max_parallel_compiler_process_count: compilation
                    .max_parallel_compiler_process_count,
                compiler_process_timings: compilation
                    .compiler_process_timings
                    .into_iter()
                    .map(NativeCpuCompilerProcessTiming::from_native)
                    .collect(),
                module_overlaps: compilation
                    .module_overlaps
                    .into_iter()
                    .map(NativeCpuModuleOverlap::from_native)
                    .collect(),
                program_pair_overlaps: compilation
                    .program_pair_overlaps
                    .into_iter()
                    .map(NativeCpuProgramPairOverlap::from_native)
                    .collect(),
                translation_units: compilation
                    .translation_units
                    .into_iter()
                    .map(NativeCpuTranslationUnitEvidence::from_native)
                    .collect(),
                render_capsule_diagnostics,
            },
            successful_steps: 0,
            successful_flushes: 0,
            successful_zero_grads: 0,
            successful_evaluations: 0,
        })
    }

    pub fn preparation_report(&self) -> &NativeCpuCompiledAdamWPreparationReport {
        &self.preparation
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            None,
        )
    }

    /// Strict-native CPU replay that omits only graph-named output egress.
    /// Loss and enabled report scalars retain their ordinary validation and
    /// result semantics.
    pub fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_external()?;
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            None,
        )
    }

    pub fn step_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::All, None)
    }

    pub fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.contract.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, CompiledStepOutputSelection::CommitOnly, None)
    }

    fn step_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        selection: CompiledStepOutputSelection,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        let PendingAdamWStep {
            request,
            next_progress,
            loss_weight,
        } = self
            .inner
            .admit_step(inputs, learning_rate, selection, injected_failure)?;
        let successful_invocation = self
            .successful_steps
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU run count overflow"))?;
        let (result, mut report) = if next_progress.accumulation_index == 0 {
            self.inner.inner.step_native_inner_with_learning_rate(
                request,
                true,
                NativeReplayContext::new(self.executor, &mut self.main_replay),
            )?
        } else {
            let transition = self
                .inner
                .inner
                .accumulation
                .clone()
                .ok_or_else(|| training("compiled accumulation replay is absent"))?;
            let replay = self
                .accumulation_replay
                .as_mut()
                .ok_or_else(|| training("compiled native CPU accumulation replay is absent"))?;
            self.inner
                .inner
                .step_accumulation_native_inner_with_learning_rate(
                    &transition,
                    request,
                    NativeReplayContext::new(self.executor, replay),
                )?
        };
        report.successful_invocation = successful_invocation;
        let inner = self.inner.publish_step(next_progress, loss_weight, result);
        self.successful_steps = successful_invocation;
        Ok(NativeCpuCompiledAdamWStepResult { inner, report })
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<NativeCpuCompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    pub fn step_batch_commit_only<B>(
        &mut self,
        batch: B,
        learning_rate: f32,
    ) -> Result<NativeCpuCompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }

    pub fn step_batch_commit_only_scheduled<B>(
        &mut self,
        batch: B,
    ) -> Result<NativeCpuCompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_commit_only_scheduled(batch.into_compiled_inputs()?)
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::All,
            injected_failure,
        )
    }

    #[cfg(test)]
    fn step_commit_only_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.step_with_learning_rate(
            inputs,
            Some(learning_rate),
            CompiledStepOutputSelection::CommitOnly,
            injected_failure,
        )
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledEvaluationResult> {
        let NativeCpuCompiledAdamW {
            inner,
            executor,
            evaluation_replay,
            successful_evaluations,
            ..
        } = self;
        let successful_invocation = (*successful_evaluations)
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU evaluation count overflow"))?;
        let executor = *executor;
        let evaluation = inner
            .evaluation
            .as_ref()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        let prepared = evaluation_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU evaluation preparation is absent"))?;
        let program = &mut inner.inner;
        let (loss_weight, parameter_states) = evaluation.plan.preflight_native_borrowed(
            &inputs,
            program.cursor.frontier(),
            prepared,
            &program.parameter_buffers,
        )?;
        let evaluated = program
            .runtime
            .with_active_state_tensors(&parameter_states, |reads| {
                evaluation.plan.evaluate_native_borrowed(
                    &inputs,
                    reads,
                    loss_weight,
                    executor,
                    prepared,
                )
            });
        let (inner, mut report) = match evaluated {
            Ok(evaluated) => evaluated,
            Err(RecurrentTransactionError::Runtime(error)) => return Err(runtime_error(error)),
            Err(RecurrentTransactionError::Stage(error)) => return Err(error),
            Err(RecurrentTransactionError::Contract(reason)) => return Err(training(reason)),
        };
        report.successful_invocation = successful_invocation;
        *successful_evaluations = successful_invocation;
        Ok(NativeCpuCompiledEvaluationResult { inner, report })
    }

    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.inner.contract.learning_rate.require_external()?;
        self.flush_partial_window_impl(Some(learning_rate), None)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.inner.contract.learning_rate.require_scheduled()?;
        self.flush_partial_window_impl(None, None)
    }

    fn flush_partial_window_impl(
        &mut self,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let (next, flush) = adamw_window_progress(
            self.inner
                .progress
                .flush_partial(self.inner.contract.gradient_accumulation_steps),
        )?;
        if !flush.did_update() {
            return Ok(NativeCpuCompiledAdamWFlushResult {
                inner: flush.into_adamw_result(),
                report: None,
            });
        }
        let mut result = flush.into_adamw_result();
        self.inner.validate_partial_token_window()?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, self.inner.non_finite_policy)?;
        }
        let successful_invocation = self
            .successful_flushes
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU flush count overflow"))?;
        let transition = self
            .inner
            .partial_flush
            .as_ref()
            .ok_or_else(|| training("compiled AdamW partial flush capture is absent"))?;
        let prepared = self
            .partial_flush_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU partial flush preparation is absent"))?;
        let (reports, report) = self.inner.inner.replay_auxiliary_transition_native(
            transition,
            learning_rate,
            self.inner.non_finite_policy,
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            injected_failure,
        )?;
        result.clip_report = reports.clip_report;
        result.window_loss_report = reports
            .window_loss
            .map(|value| CompiledAdamWWindowLossReport::new(value, result.flushed_microbatches));
        self.inner.progress = next;
        self.successful_flushes = successful_invocation;
        Ok(NativeCpuCompiledAdamWFlushResult {
            inner: result,
            report: Some(report),
        })
    }

    #[cfg(test)]
    fn flush_partial_window_with_injected_failure(
        &mut self,
        learning_rate: TensorData,
        injected_failure: u64,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.flush_partial_window_impl(Some(learning_rate), Some(injected_failure))
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }

    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.inner.zero_grad_capture_identity()
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.inner.captured_multi_step_lr()
    }

    /// CPU-only admission policy selected when this runtime was prepared.
    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.inner.non_finite_policy
    }

    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.inner.dropout_block_counter()
    }

    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        self.inner.checkpoint()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_impl(None)
    }

    fn zero_grad_impl(
        &mut self,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWZeroGradResult> {
        let (next, reset) = adamw_window_progress(
            self.inner
                .progress
                .cancel(self.inner.contract.gradient_accumulation_steps),
        )?;
        if !reset.did_discard() {
            return Ok(reset);
        }
        let next = adamw_window_progress(next.record_reset_transition())?;
        let successful_invocation = self
            .successful_zero_grads
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU zero-grad count overflow"))?;
        let transition = self
            .inner
            .zero_grad
            .as_ref()
            .ok_or_else(|| training("compiled AdamW zero-grad capture is absent"))?;
        let prepared = self
            .zero_grad_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU zero-grad preparation is absent"))?;
        let (reports, _) = self.inner.inner.replay_auxiliary_transition_native(
            transition,
            None,
            CpuNonFinitePolicy::Propagate,
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            injected_failure,
        )?;
        debug_assert!(reports.clip_report.is_none());
        debug_assert!(reports.window_loss.is_none());
        self.inner.progress = next;
        self.successful_zero_grads = successful_invocation;
        Ok(reset)
    }

    #[cfg(test)]
    fn zero_grad_with_injected_failure(
        &mut self,
        injected_failure: u64,
    ) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_impl(Some(injected_failure))
    }
}

impl CompiledTrainingRuntime for CpuCompiledAdamW {
    type Step = CompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        CpuCompiledAdamW::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        CpuCompiledAdamW::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::parameter_snapshots(self)
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.parameter_snapshots()?,
            &self.contract.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for CpuCompiledAdamW {
    type Evaluation = CompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        CpuCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::evaluation_capture_identity(self)
    }
}

impl CompiledCheckpointRuntime for CpuCompiledAdamW {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        CpuCompiledAdamW::checkpoint(self)
    }
}

impl CompiledCheckpointRestoreRuntime for CpuCompiledAdamW {
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        let restored = self.restored_candidate(checkpoint)?;
        *self = restored;
        Ok(())
    }
}

impl CompiledAdamWRuntime for CpuCompiledAdamW {
    fn gradient_accumulation_steps(&self) -> u64 {
        CpuCompiledAdamW::gradient_accumulation_steps(self)
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        CpuCompiledAdamW::max_gradient_norm(self)
    }

    fn loss_scale(&self) -> f32 {
        CpuCompiledAdamW::loss_scale(self)
    }

    fn window_loss_report_enabled(&self) -> bool {
        CpuCompiledAdamW::window_loss_report_enabled(self)
    }

    fn optimizer_step(&self) -> Result<u64> {
        CpuCompiledAdamW::optimizer_step(self)
    }

    fn accumulation_index(&self) -> Result<u64> {
        CpuCompiledAdamW::accumulation_index(self)
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        CpuCompiledAdamW::zero_grad(self)
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::zero_grad_capture_identity(self)
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::first_moment_snapshots(self)
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::second_moment_snapshots(self)
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::gradient_accumulator_snapshots(self)
    }
}

impl CompiledAdamWCommitOnlyRuntime for CpuCompiledAdamW {
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingCommitOnlyRuntime for CpuCompiledAdamW {
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingWindowCommitRuntime for CpuCompiledAdamW {
    type WindowCommit = CompiledAdamWFlushResult;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        CpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::flush_capture_identity(self)
    }
}

impl CompiledAdamWFlushRuntime for CpuCompiledAdamW {
    type Flush = CompiledAdamWFlushResult;

    fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWFlushResult> {
        CompiledTrainingWindowCommitRuntime::commit_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::flush_capture_identity(self)
    }
}

impl CompiledTrainingRatePolicyWindowCommitRuntime for CpuCompiledAdamW {
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        CpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWRuntime for CpuCompiledAdamW {
    type ScheduledFlush = CompiledAdamWFlushResult;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        CpuCompiledAdamW::captured_multi_step_lr(self)
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        CpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        CpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWCommitOnlyRuntime for CpuCompiledAdamW {
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only_scheduled(self, inputs)
    }
}

impl CompiledTrainingRuntime for NativeCpuCompiledAdamW<'_> {
    type Step = NativeCpuCompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.inner.parameter_snapshots()?,
            &self.inner.contract.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for NativeCpuCompiledAdamW<'_> {
    type Evaluation = NativeCpuCompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        NativeCpuCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }
}

impl CompiledCheckpointRuntime for NativeCpuCompiledAdamW<'_> {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.inner.checkpoint()
    }
}

impl CompiledCheckpointRestoreRuntime for NativeCpuCompiledAdamW<'_> {
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        let restored = self.inner.restored_candidate(checkpoint)?;
        self.inner = restored;
        Ok(())
    }
}

impl CompiledAdamWRuntime for NativeCpuCompiledAdamW<'_> {
    fn gradient_accumulation_steps(&self) -> u64 {
        self.inner.gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.inner.max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.inner.loss_scale()
    }

    fn window_loss_report_enabled(&self) -> bool {
        self.inner.window_loss_report_enabled()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.inner.optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.inner.accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        NativeCpuCompiledAdamW::zero_grad(self)
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.inner.zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.gradient_accumulator_snapshots()
    }
}

impl CompiledAdamWCommitOnlyRuntime for NativeCpuCompiledAdamW<'_> {
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingCommitOnlyRuntime for NativeCpuCompiledAdamW<'_> {
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }
}

impl CompiledTrainingWindowCommitRuntime for NativeCpuCompiledAdamW<'_> {
    type WindowCommit = NativeCpuCompiledAdamWFlushResult;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        NativeCpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }
}

impl CompiledAdamWFlushRuntime for NativeCpuCompiledAdamW<'_> {
    type Flush = NativeCpuCompiledAdamWFlushResult;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        CompiledTrainingWindowCommitRuntime::commit_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }
}

impl CompiledTrainingRatePolicyWindowCommitRuntime for NativeCpuCompiledAdamW<'_> {
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        NativeCpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWRuntime for NativeCpuCompiledAdamW<'_> {
    type ScheduledFlush = NativeCpuCompiledAdamWFlushResult;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.inner.captured_multi_step_lr()
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        NativeCpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledScheduledAdamWCommitOnlyRuntime for NativeCpuCompiledAdamW<'_> {
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only_scheduled(self, inputs)
    }
}

impl<M, R> CompiledTrainingRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingRuntime,
{
    type Step = R::Step;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.runtime.step(inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.runtime.step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.runtime.capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime.parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        self.runtime.publish_parameters(module)
    }
}

impl<M, R> CompiledCheckpointRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledCheckpointRuntime,
{
    type Checkpoint = R::Checkpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.runtime.checkpoint()
    }
}

impl<M, R> CompiledCheckpointRestoreRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledCheckpointRestoreRuntime,
{
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.runtime.restore_checkpoint_in_place(checkpoint)
    }
}

impl<M, R> CompiledEvaluationRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledEvaluationRuntime,
{
    type Evaluation = R::Evaluation;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        self.runtime.evaluate(inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.runtime.evaluation_capture_identity()
    }
}

impl<M, R> CompiledTrainingRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingRuntime,
{
    type Step = R::Step;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.training.step(inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.training.step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.training.capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.training.parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        self.training.publish_parameters(module)
    }
}

impl<M, R> CompiledCheckpointRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledCheckpointRuntime,
{
    type Checkpoint = R::Checkpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.training.checkpoint()
    }
}

impl<M, R> CompiledCheckpointRestoreRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledCheckpointRestoreRuntime,
{
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.training.restore_checkpoint_in_place(checkpoint)
    }
}

impl<M, R> CompiledEvaluationRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledEvaluationRuntime,
{
    type Evaluation = R::Evaluation;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        self.training.evaluate(inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.training.evaluation_capture_identity()
    }
}

impl<M, R> CompiledAdamWRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWRuntime,
{
    fn gradient_accumulation_steps(&self) -> u64 {
        self.training.runtime.gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.training.runtime.max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.training.runtime.loss_scale()
    }

    fn window_loss_report_enabled(&self) -> bool {
        self.training.runtime.window_loss_report_enabled()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.training.runtime.optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.training.runtime.accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.training.runtime.zero_grad()
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.training.runtime.zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.training.runtime.first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.training.runtime.second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.training.runtime.gradient_accumulator_snapshots()
    }
}

impl<M, R> CompiledAdamWCommitOnlyRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWCommitOnlyRuntime,
{
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.training
            .runtime
            .step_commit_only(inputs, learning_rate)
    }
}

impl<M, R> CompiledTrainingCommitOnlyRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingCommitOnlyRuntime,
{
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.runtime.commit_step(inputs, learning_rate)
    }
}

impl<M, R> CompiledTrainingWindowResetRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingWindowResetRuntime,
{
    fn reset_gradient_window(&mut self) -> Result<CompiledTrainingWindowReset> {
        self.runtime.reset_gradient_window()
    }

    fn gradient_window_reset_capture_identity(&self) -> Option<u64> {
        self.runtime.gradient_window_reset_capture_identity()
    }
}

impl<M, R> CompiledTrainingWindowRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingWindowRuntime,
{
    fn gradient_window_size(&self) -> u64 {
        self.runtime.gradient_window_size()
    }

    fn pending_microbatch_count(&self) -> Result<u64> {
        self.runtime.pending_microbatch_count()
    }
}

impl<M, R> CompiledTrainingWindowCommitRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingWindowCommitRuntime,
{
    type WindowCommit = R::WindowCommit;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        self.runtime.commit_partial_window(learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        self.runtime.partial_window_commit_capture_identity()
    }
}

impl<M, R> CompiledTrainingRatePolicyRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingRatePolicyRuntime,
{
    fn step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.runtime.step_with_rate_policy(inputs)
    }
}

impl<M, R> CompiledTrainingRatePolicyCommitOnlyRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingRatePolicyCommitOnlyRuntime,
{
    fn commit_step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.runtime.commit_step_with_rate_policy(inputs)
    }
}

impl<M, R> CompiledTrainingRatePolicyWindowCommitRuntime for CompiledModuleTrainingSession<M, R>
where
    R: CompiledTrainingRatePolicyWindowCommitRuntime,
{
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        self.runtime.commit_partial_window_with_rate_policy()
    }
}

impl<M, R> CompiledTrainingCommitOnlyRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingCommitOnlyRuntime,
{
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.training.runtime.commit_step(inputs, learning_rate)
    }
}

impl<M, R> CompiledTrainingWindowCommitRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingWindowCommitRuntime + CompiledAdamWRuntime,
{
    type WindowCommit = R::WindowCommit;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        self.training.runtime.commit_partial_window(learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        self.training
            .runtime
            .partial_window_commit_capture_identity()
    }
}

impl<M, R> CompiledAdamWFlushRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWFlushRuntime,
{
    type Flush = R::Flush;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        self.training.runtime.flush_partial_window(learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        self.training.runtime.flush_capture_identity()
    }
}

impl<M, R> CompiledTrainingRatePolicyWindowCommitRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingRatePolicyWindowCommitRuntime + CompiledScheduledAdamWRuntime,
{
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        self.training
            .runtime
            .commit_partial_window_with_rate_policy()
    }
}

impl<M, R> CompiledScheduledAdamWRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledScheduledAdamWRuntime,
{
    type ScheduledFlush = R::ScheduledFlush;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.training.runtime.captured_multi_step_lr()
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        self.training.runtime.step_scheduled(inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        self.training.runtime.flush_partial_window_scheduled()
    }
}

impl<M, R> CompiledScheduledAdamWCommitOnlyRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledScheduledAdamWCommitOnlyRuntime,
{
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.training.runtime.step_commit_only_scheduled(inputs)
    }
}

fn adamw_step_result(
    mut inner: CompiledTrainingStepResult,
    progress: CompiledTrainingWindowProgress,
    loss_weight: u64,
    gradient_accumulation_steps: u64,
    clip_report_enabled: bool,
    window_loss_report_enabled: bool,
) -> CompiledAdamWStepResult {
    inner.loss_aggregation_weight = loss_weight;
    let expected = if progress.accumulation_index == 0 {
        adamw_observation_schema(clip_report_enabled, window_loss_report_enabled)
    } else {
        CompiledTrainingObservationSchema::default()
    };
    debug_assert_eq!(inner.observations.len(), expected.len());
    debug_assert!(
        inner
            .observations
            .iter()
            .map(|observation| observation.key)
            .eq(expected.entries.iter().map(|spec| spec.key))
    );
    for (observation, spec) in inner.observations.iter().zip(&expected.entries) {
        debug_assert!(
            validate_observation_value_descriptor(&observation.value, spec.constraint).is_ok()
        );
    }
    let mut observations = std::mem::take(&mut inner.observations)
        .into_iter()
        .map(|observation| observation.value);
    let clip_report = take_compiled_clip_report(
        &mut observations,
        clip_report_enabled && progress.accumulation_index == 0,
    );
    let window_loss = take_compiled_window_loss_value(
        &mut observations,
        window_loss_report_enabled && progress.accumulation_index == 0,
    );
    debug_assert!(observations.next().is_none());
    let window_loss_report = window_loss
        .map(|value| CompiledAdamWWindowLossReport::new(value, gradient_accumulation_steps));
    CompiledAdamWStepResult {
        inner,
        optimizer_step: progress.optimizer_step,
        accumulation_index: progress.accumulation_index,
        clip_report,
        window_loss_report,
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for CpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu()
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for ConfiguredCpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu_with_non_finite_policy(self.non_finite_policy())
    }
}

impl<'executor> SessionTarget<&CompiledAdamWPlan> for NativeCpuSessionTarget<'executor> {
    type Session = NativeCpuCompiledAdamW<'executor>;
    type Error = Error;

    fn prepare(&self, plan: &CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_native_cpu(self)
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

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for CpuSessionTarget {
    type Session = CompiledModuleAdamWSession<M, CpuCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan.plan.prepare_cpu() {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for ConfiguredCpuSessionTarget {
    type Session = CompiledModuleAdamWSession<M, CpuCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan
            .plan
            .prepare_cpu_with_non_finite_policy(self.non_finite_policy())
        {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
    }
}

impl<'executor, M: Module> SessionTarget<CompiledModuleAdamWPlan<M>>
    for NativeCpuSessionTarget<'executor>
{
    type Session = CompiledModuleAdamWSession<M, NativeCpuCompiledAdamW<'executor>>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match plan.plan.prepare_native_cpu(self) {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for MetalSessionTarget {
    type Session = CompiledModuleAdamWSession<M, MetalCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.validate_ready_for_preparation() {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let evaluation_capture_identity = plan.evaluation_capture_identity();
        let runtime = match <Self as SessionTarget<&CompiledAdamWPlan>>::prepare(self, &plan.plan) {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
            required_evaluation_capture_identity: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            training: CompiledModuleTrainingSession {
                module,
                runtime,
                seal,
                evaluation_capture_identity,
            },
        })
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

fn canonical_parameters(
    parameters: impl IntoIterator<Item = TrainingParameterInit>,
) -> Result<BTreeMap<String, TensorData>> {
    let mut values = BTreeMap::new();
    for parameter in parameters {
        validate_user_name(&parameter.name, "parameter")?;
        if parameter.value.dtype() != DType::F32 {
            return Err(training("compiled training parameters must be F32"));
        }
        checked_bytes(&parameter.value)?;
        if values.insert(parameter.name, parameter.value).is_some() {
            return Err(training("duplicate compiled parameter name"));
        }
    }
    Ok(values)
}

fn validate_weight_decay_exclusion_names<'a>(
    config: &CompiledAdamWConfig,
    parameter_names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    if config.weight_decay_exclusions.is_empty() {
        return Ok(());
    }
    let parameter_names = parameter_names.into_iter().collect::<BTreeSet<_>>();
    if let Some(name) = config
        .weight_decay_exclusions
        .iter()
        .find(|name| !parameter_names.contains(name.as_str()))
    {
        return Err(training(format!(
            "compiled AdamW weight-decay exclusion name {name:?} is unknown"
        )));
    }
    Ok(())
}

fn validate_user_name(name: &str, kind: &str) -> Result<()> {
    if name.is_empty() || name == "loss" || name.starts_with(INTERNAL_PREFIX) {
        return Err(training(format!("invalid compiled {kind} name")));
    }
    Ok(())
}

fn checked_bytes(value: &TensorData) -> Result<usize> {
    value
        .len()
        .checked_mul(value.dtype().itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

fn checked_recurrent_state_extent(
    bytes: impl IntoIterator<Item = usize>,
) -> Result<(usize, usize)> {
    bytes
        .into_iter()
        .try_fold((0usize, 0usize), |(count, total), bytes| {
            Ok((
                count
                    .checked_add(1)
                    .ok_or_else(|| training("compiled recurrent state count overflows"))?,
                total
                    .checked_add(bytes)
                    .ok_or_else(|| training("compiled recurrent state bytes overflow"))?,
            ))
        })
}

fn checked_descriptor(shape: &Shape, dtype: DType) -> Result<usize> {
    shape
        .numel()
        .map_err(|_| training("compiled tensor element extent overflow"))?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

fn state_for(buffer: u64, value: &TensorData) -> Result<BufferState> {
    Ok(BufferState {
        buffer,
        version: 0,
        shape: value.shape().clone(),
        dtype: value.dtype(),
        bytes: checked_bytes(value)?,
    })
}

fn validate_loss(graph: &Graph, loss: NodeId) -> Result<()> {
    if graph.dtype(loss)? != DType::F32 || graph.shape(loss)? != &Shape::from([]) {
        return Err(training(
            "compiled training loss must be a rank-zero F32 scalar",
        ));
    }
    Ok(())
}

fn validate_evaluation_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    provided: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if expected.len() != provided.len() || expected.keys().ne(provided.keys()) {
        return Err(training("compiled evaluation input names mismatch"));
    }
    for (name, (shape, dtype)) in expected {
        let value = &provided[name];
        if value.shape() != shape || value.dtype() != *dtype {
            return Err(training(format!(
                "compiled evaluation input {name:?} descriptor mismatch"
            )));
        }
        checked_bytes(value)?;
    }
    Ok(())
}

fn validate_outputs<'a>(
    loss: NodeId,
    outputs: &BTreeMap<String, NodeId>,
    reserved_user_names: impl IntoIterator<Item = &'a String>,
) -> Result<()> {
    let mut nodes = BTreeSet::from([loss]);
    let reserved_user_names = reserved_user_names
        .into_iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for (name, node) in outputs {
        validate_user_name(name, "output")?;
        if name == "loss" || reserved_user_names.contains(name.as_str()) {
            return Err(training(
                "compiled output name collides with another user name",
            ));
        }
        if !nodes.insert(*node) {
            return Err(training("duplicate compiled output node"));
        }
    }
    Ok(())
}

fn validate_external_binding_ownership<'a>(
    capture: &CapturedMixedSchedule,
    configured_inputs: impl IntoIterator<Item = &'a String>,
) -> Result<()> {
    let mut external = configured_inputs
        .into_iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    external.insert(LEARNING_RATE_INPUT);
    for binding in &capture.state_bindings {
        let input = capture
            .schedule
            .inputs
            .iter()
            .find(|input| input.node == binding.input_node)
            .ok_or_else(|| training("compiled state input ABI is absent"))?;
        if external.contains(input.name.as_str()) {
            return Err(training(
                "compiled external input shadows persistent state binding",
            ));
        }
    }
    Ok(())
}

fn collect_state_bindings(
    schedule: &Schedule,
    states: &BTreeMap<NodeId, BufferState>,
) -> Result<Vec<ScheduleStateBinding>> {
    let mut bindings = Vec::new();
    let mut seen = BTreeSet::new();
    let mut bound_nodes = BTreeSet::new();
    for item in &schedule.items {
        for binding in &item.input_bindings {
            let Some(state) = states.get(&binding.input_node) else {
                continue;
            };
            if !seen.insert((item.id, binding.input_node)) {
                return Err(training("duplicate compiled state input binding"));
            }
            bound_nodes.insert(binding.input_node);
            bindings.push(ScheduleStateBinding {
                state: state.clone(),
                view: None,
                consumer_item: item.id,
                consumer_node: item.node,
                input_node: binding.input_node,
                desc: binding.desc.clone(),
                abi_index: binding.abi_index,
            });
        }
    }
    if bindings.is_empty() || states.keys().any(|node| !bound_nodes.contains(node)) {
        return Err(training("compiled state input is not reachable"));
    }
    Ok(bindings)
}

fn value_binding(
    schedule: &Schedule,
    node: NodeId,
    effect_item: u64,
) -> Result<ScheduleValueBinding> {
    let (producer_item, producer) = schedule
        .items
        .iter()
        .enumerate()
        .find(|(_, item)| item.primary_output().id == node.index() as u64)
        .ok_or_else(|| training("compiled update output is not materialized"))?;
    Ok(ScheduleValueBinding {
        producer_item: u64::try_from(producer_item)
            .map_err(|_| training("compiled producer index overflow"))?,
        producer_node: node,
        producer_output: producer.primary_output().clone(),
        abi_index: 0,
        effect_item,
        source_position: 0,
    })
}

fn effect_states(effects: &EffectGraph) -> Result<Vec<BufferState>> {
    let plan = effects.plan();
    plan.validate().map_err(effect_error)?;
    let mut states = BTreeMap::new();
    for step in plan.steps {
        for state in step.reads.into_iter().chain([step.write]) {
            states.insert((state.buffer, state.version), state);
        }
    }
    Ok(states.into_values().collect())
}

fn validate_step_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    actual: &BTreeMap<String, TensorData>,
    learning_rate: &TensorData,
) -> Result<()> {
    validate_training_inputs(expected, actual)?;
    validate_learning_rate(learning_rate)
}

fn validate_training_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    actual: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if actual.len() != expected.len() || actual.keys().ne(expected.keys()) {
        return Err(training("compiled training input names do not match"));
    }
    for (name, value) in actual {
        let (shape, dtype) = &expected[name];
        if value.shape() != shape || value.dtype() != *dtype {
            return Err(training("compiled training input descriptor mismatch"));
        }
        checked_bytes(value)?;
    }
    Ok(())
}

fn validate_token_weight_policy(
    inputs: &BTreeMap<String, (Shape, DType)>,
    policy: &CompiledTokenWeightPolicy,
    accumulation_steps: u64,
) -> Result<()> {
    if accumulation_steps == 0 {
        return Err(training(
            "compiled AdamW gradient accumulation steps must be positive",
        ));
    }
    let (shape, dtype) = policy.expected_descriptor(inputs)?;
    let token_elements = shape.numel()?;
    let valid_dtype = match policy {
        CompiledTokenWeightPolicy::ExplicitMask(_) => *dtype == DType::F32,
        CompiledTokenWeightPolicy::IgnoreIndex { .. } => *dtype == DType::I32,
    };
    if !valid_dtype || token_elements == 0 {
        let message = match policy {
            CompiledTokenWeightPolicy::ExplicitMask(_) => {
                "compiled AdamW token-weight mask must be nonempty fixed-shape F32"
            }
            CompiledTokenWeightPolicy::IgnoreIndex { .. } => {
                "compiled AdamW ignore-index target must be nonempty fixed-shape I32"
            }
        };
        return Err(training(message));
    }
    let token_elements = u64::try_from(token_elements)
        .map_err(|_| training("compiled AdamW token-weight element count overflows"))?;
    let maximum_count = token_elements
        .checked_mul(accumulation_steps)
        .ok_or_else(|| training("compiled AdamW token-weight count bound overflows"))?;
    if maximum_count > MAX_EXACT_F32_INTEGER_COUNT {
        return Err(training(
            "compiled AdamW token-weight count must remain exactly representable in F32",
        ));
    }
    Ok(())
}

fn reject_token_weighted_scalar_loss(config: &CompiledAdamWConfig) -> Result<()> {
    if config.token_weight_policy.is_some() {
        return Err(training(
            "compiled AdamW token-weighted accumulation requires the token-mean-loss compile surface",
        ));
    }
    Ok(())
}

fn lower_compiled_adamw_objective(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    objective: CompiledAdamWObjective,
) -> Result<NodeId> {
    lower_compiled_adamw_objective_for_policy(
        graph,
        inputs,
        objective,
        config.token_weight_policy.as_ref(),
        &config.inputs,
        config.allow_zero_valid_token_microbatches,
    )
}

fn lower_compiled_adamw_objective_with_ignore_index_nodes(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    objective: CompiledAdamWObjective,
    nodes: CompiledAdamWIgnoreIndexContext,
) -> Result<NodeId> {
    lower_compiled_adamw_objective_for_ignore_index_policy(
        graph,
        objective,
        nodes,
        config.token_weight_policy.as_ref(),
        &config.inputs,
        config.allow_zero_valid_token_microbatches,
    )
}

fn lower_compiled_adamw_objective_for_ignore_index_policy(
    graph: &mut Graph,
    objective: CompiledAdamWObjective,
    nodes: CompiledAdamWIgnoreIndexContext,
    policy: Option<&CompiledTokenWeightPolicy>,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    let policy = require_ignore_index_policy(policy)?;
    let (token_shape, _) = policy.expected_descriptor(input_descriptors)?;
    match objective {
        CompiledAdamWObjective::Scalar(_) => Err(training(
            "compiled AdamW token-weighted accumulation requires the token-mean-loss compile surface",
        )),
        CompiledAdamWObjective::TokenMean(losses) => lower_token_mean_loss(
            graph,
            losses,
            nodes.weight,
            token_shape,
            allow_zero_valid_token_microbatches,
        ),
    }
}

fn lower_compiled_adamw_objective_for_policy(
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    objective: CompiledAdamWObjective,
    token_weight_policy: Option<&CompiledTokenWeightPolicy>,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    match objective {
        CompiledAdamWObjective::Scalar(loss) => {
            if token_weight_policy.is_some() {
                return Err(training(
                    "compiled AdamW token-weighted accumulation requires the token-mean-loss compile surface",
                ));
            }
            Ok(loss)
        }
        CompiledAdamWObjective::TokenMean(losses) => {
            let policy = token_weight_policy.ok_or_else(|| {
                training("compiled AdamW token-mean-loss compilation requires token weighting")
            })?;
            let (token_shape, _) = policy.expected_descriptor(input_descriptors)?;
            let mask = lower_token_weight_mask(graph, inputs, policy)?;
            lower_token_mean_loss(
                graph,
                losses,
                mask,
                token_shape,
                allow_zero_valid_token_microbatches,
            )
        }
    }
}

fn token_mean_loss_descriptor(config: &CompiledAdamWConfig) -> Result<(String, Shape)> {
    let mask_input = match config.token_weight_policy.as_ref().ok_or_else(|| {
        training("compiled AdamW token-mean-loss compilation requires token weighting")
    })? {
        CompiledTokenWeightPolicy::ExplicitMask(name) => name,
        CompiledTokenWeightPolicy::IgnoreIndex { .. } => {
            return Err(training(
                "compiled AdamW ignore-index weighting requires the unified token-mean compile surface",
            ));
        }
    };
    let (shape, dtype) = config
        .inputs
        .get(mask_input)
        .ok_or_else(|| training("compiled AdamW token-weight mask must name an existing input"))?;
    if *dtype != DType::F32 {
        return Err(training(
            "compiled AdamW token-weight mask must be nonempty fixed-shape F32",
        ));
    }
    Ok((mask_input.clone(), shape.clone()))
}

fn lower_token_weight_mask(
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    policy: &CompiledTokenWeightPolicy,
) -> Result<NodeId> {
    match policy {
        CompiledTokenWeightPolicy::ExplicitMask(mask_input) => inputs
            .get(mask_input)
            .copied()
            .ok_or_else(|| training("compiled AdamW token-weight mask input is absent")),
        CompiledTokenWeightPolicy::IgnoreIndex {
            target_input,
            value,
        } => {
            let targets = inputs
                .get(target_input)
                .copied()
                .ok_or_else(|| training("compiled AdamW ignore-index target input is absent"))?;
            let ignored =
                graph.full_with_dtype(Shape::from([]), Scalar::I(i64::from(*value)), DType::I32)?;
            let keep = graph.compare(CompareOp::Ne, targets, ignored)?;
            graph.cast(keep, DType::F32)
        }
    }
}

#[derive(Clone, Copy)]
struct CompiledTokenBatchCount {
    float: NodeId,
    exact: NodeId,
}

fn lower_token_batch_count(
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    token_weight: Option<NodeId>,
    policy: &CompiledTokenWeightPolicy,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
) -> Result<CompiledTokenBatchCount> {
    let weight = match token_weight {
        Some(weight) => weight,
        None => lower_token_weight_mask(graph, inputs, policy)?,
    };
    validate_token_weight_node(graph, weight, policy, input_descriptors)?;
    let float = graph.sum_all(weight)?;
    let exact = graph.cast(float, DType::U64)?;
    Ok(CompiledTokenBatchCount { float, exact })
}

fn lower_ignore_index_nodes(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<CompiledAdamWIgnoreIndexContext> {
    lower_ignore_index_nodes_for_policy(
        config.token_weight_policy.as_ref(),
        &config.inputs,
        graph,
        inputs,
    )
}

fn lower_ignore_index_nodes_for_policy(
    policy: Option<&CompiledTokenWeightPolicy>,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<CompiledAdamWIgnoreIndexContext> {
    let (policy, target_input, value) = match policy {
        Some(
            policy @ CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            },
        ) => (policy, target_input, value),
        _ => {
            return Err(training(
                "compiled AdamW ignore-index graph context requires ignore-index weighting",
            ));
        }
    };
    let targets = inputs
        .get(target_input)
        .copied()
        .ok_or_else(|| training("compiled AdamW ignore-index target input is absent"))?;
    let ignored =
        graph.full_with_dtype(Shape::from([]), Scalar::I(i64::from(*value)), DType::I32)?;
    let validity = graph.compare(CompareOp::Ne, targets, ignored)?;
    let weight = graph.cast(validity, DType::F32)?;
    let nodes = CompiledAdamWIgnoreIndexContext {
        targets,
        validity,
        weight,
    };
    validate_ignore_index_nodes(policy, input_descriptors, graph, nodes)?;
    Ok(nodes)
}

fn require_ignore_index_policy(
    policy: Option<&CompiledTokenWeightPolicy>,
) -> Result<&CompiledTokenWeightPolicy> {
    match policy {
        Some(policy @ CompiledTokenWeightPolicy::IgnoreIndex { .. }) => Ok(policy),
        _ => Err(training(
            "compiled AdamW ignore-index graph context requires ignore-index weighting",
        )),
    }
}

fn validate_ignore_index_nodes(
    policy: &CompiledTokenWeightPolicy,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    graph: &Graph,
    nodes: CompiledAdamWIgnoreIndexContext,
) -> Result<()> {
    let (shape, dtype) = policy.expected_descriptor(input_descriptors)?;
    if *dtype != DType::I32
        || graph.shape(nodes.targets)? != shape
        || graph.dtype(nodes.targets)? != DType::I32
        || graph.shape(nodes.validity)? != shape
        || graph.dtype(nodes.validity)? != DType::Bool
        || graph.shape(nodes.weight)? != shape
        || graph.dtype(nodes.weight)? != DType::F32
    {
        return Err(training(
            "compiled AdamW ignore-index graph context descriptor differs",
        ));
    }
    Ok(())
}

fn validate_token_weight_node(
    graph: &Graph,
    node: NodeId,
    policy: &CompiledTokenWeightPolicy,
    inputs: &BTreeMap<String, (Shape, DType)>,
) -> Result<()> {
    let (shape, _) = policy.expected_descriptor(inputs)?;
    if graph.shape(node)? != shape || graph.dtype(node)? != DType::F32 {
        return Err(training(
            "compiled AdamW token-weight graph context descriptor differs",
        ));
    }
    Ok(())
}

fn lower_token_mean_loss(
    graph: &mut Graph,
    losses: NodeId,
    mask: NodeId,
    expected_shape: &Shape,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    if graph.dtype(losses)? != DType::F32 || graph.shape(losses)? != expected_shape {
        return Err(training(
            "compiled AdamW per-token losses must exactly match the token-weight mask descriptor",
        ));
    }
    if graph.dtype(mask)? != DType::F32 || graph.shape(mask)? != expected_shape {
        return Err(training(
            "compiled AdamW token-weight mask descriptor changed during compilation",
        ));
    }
    let weighted = if allow_zero_valid_token_microbatches {
        let zero = scalar_f32(graph, 0.0)?;
        let keep = graph.compare(CompareOp::Gt, mask, zero)?;
        graph.select(keep, losses, zero)?
    } else {
        graph.mul(losses, mask)?
    };
    let numerator = graph.sum_all(weighted)?;
    let denominator = graph.sum_all(mask)?;
    let denominator =
        safe_token_count_divisor(graph, denominator, allow_zero_valid_token_microbatches)?;
    graph.div(numerator, denominator)
}

fn safe_token_count_divisor(
    graph: &mut Graph,
    count: NodeId,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    if !allow_zero_valid_token_microbatches {
        return Ok(count);
    }
    let zero = scalar_f32(graph, 0.0)?;
    let one = scalar_f32(graph, 1.0)?;
    let positive = graph.compare(CompareOp::Gt, count, zero)?;
    graph.select(positive, count, one)
}

fn validate_token_weight(
    inputs: &BTreeMap<String, TensorData>,
    policy: Option<&CompiledTokenWeightPolicy>,
    allow_zero_valid_token_microbatches: bool,
) -> Result<u64> {
    let Some(policy) = policy else {
        return Ok(1);
    };
    let mut valid_tokens = 0_u64;
    match policy {
        CompiledTokenWeightPolicy::ExplicitMask(mask_input) => {
            let mask = inputs
                .get(mask_input)
                .ok_or_else(|| training("compiled AdamW token-weight mask input is absent"))?;
            for index in 0..mask.shape().numel()? {
                let value = mask.scalar_at(index).as_f64();
                if !value.is_finite() || (value != 0.0 && value != 1.0) {
                    return Err(training(
                        "compiled AdamW token-weight mask must contain finite binary values",
                    ));
                }
                valid_tokens = valid_tokens
                    .checked_add(u64::from(value == 1.0))
                    .ok_or_else(|| training("compiled AdamW token-weight count overflows"))?;
            }
        }
        CompiledTokenWeightPolicy::IgnoreIndex {
            target_input,
            value,
        } => {
            let targets = inputs
                .get(target_input)
                .ok_or_else(|| training("compiled AdamW ignore-index target input is absent"))?;
            for index in 0..targets.shape().numel()? {
                valid_tokens = valid_tokens
                    .checked_add(u64::from(
                        targets.scalar_at(index).as_i64() != i64::from(*value),
                    ))
                    .ok_or_else(|| training("compiled AdamW token-weight count overflows"))?;
            }
        }
    }
    if valid_tokens == 0 && !allow_zero_valid_token_microbatches {
        return Err(training(
            "compiled AdamW token-weight mask must contain at least one valid token",
        ));
    }
    Ok(valid_tokens)
}

fn validate_retained_token_count(
    inputs: &BTreeMap<String, (Shape, DType)>,
    policy: &CompiledTokenWeightPolicy,
    accumulation_index: u64,
    count: u64,
    allow_zero_valid_token_microbatches: bool,
) -> Result<()> {
    let (shape, _) = policy.expected_descriptor(inputs)?;
    let token_elements = u64::try_from(shape.numel()?)
        .map_err(|_| training("compiled AdamW token-weight element count overflows"))?;
    let maximum_count = token_elements
        .checked_mul(accumulation_index)
        .ok_or_else(|| training("compiled AdamW retained token count bound overflows"))?;
    if (!allow_zero_valid_token_microbatches && count < accumulation_index) || count > maximum_count
    {
        return Err(training(
            "compiled AdamW retained token count is inconsistent with progress",
        ));
    }
    Ok(())
}

fn validate_learning_rate(learning_rate: &TensorData) -> Result<()> {
    if learning_rate.shape() != &Shape::from([]) || learning_rate.dtype() != DType::F32 {
        return Err(training(
            "compiled training learning rate must be rank-zero F32",
        ));
    }
    checked_bytes(learning_rate)?;
    Ok(())
}

fn validate_learning_rate_for_policy(
    learning_rate: &TensorData,
    policy: CpuNonFinitePolicy,
) -> Result<()> {
    validate_learning_rate(learning_rate)?;
    if policy == CpuNonFinitePolicy::RejectTransition {
        validate_finite_tensors(std::iter::once(learning_rate), "external learning rate")?;
    }
    Ok(())
}

fn validate_staged_transition<'a>(
    outputs: &[TensorData],
    successors: impl IntoIterator<Item = &'a TensorData>,
    policy: CpuNonFinitePolicy,
    require_loss: bool,
) -> std::result::Result<(), String> {
    if policy == CpuNonFinitePolicy::Propagate {
        return Ok(());
    }
    if require_loss {
        let loss = outputs
            .first()
            .ok_or_else(|| "compiled CPU transition loss is absent".to_owned())?;
        if loss.shape() != &Shape::from([]) || loss.dtype() != DType::F32 {
            return Err("compiled CPU transition loss must be rank-zero F32".to_owned());
        }
        if has_non_finite_f32(std::iter::once(loss)) {
            return Err("compiled CPU transition has a non-finite loss".to_owned());
        }
    }
    if has_non_finite_f32(successors) {
        return Err("compiled CPU transition has a non-finite recurrent successor".to_owned());
    }
    Ok(())
}

fn validate_finite_tensors<'a>(
    tensors: impl IntoIterator<Item = &'a TensorData>,
    role: &str,
) -> Result<()> {
    if has_non_finite_f32(tensors) {
        return Err(training(format!(
            "compiled CPU transition has a non-finite {role}"
        )));
    }
    Ok(())
}

fn has_non_finite_f32<'a>(tensors: impl IntoIterator<Item = &'a TensorData>) -> bool {
    tensors.into_iter().any(|tensor| {
        tensor.dtype() == DType::F32 && tensor.values().iter().any(|value| !value.is_finite())
    })
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
