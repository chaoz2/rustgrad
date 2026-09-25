//! Graph-free CPU replay for static training programs with recurrent state.

mod adamw_checkpoint;
mod adamw_contract;
mod adamw_plan;
mod adamw_plan_restore;
mod capture;
mod cpu_adamw_capabilities;
mod cpu_adamw_runtime;
mod cpu_momentum_runtime;
mod cpu_training_program;
mod cpu_training_step;
mod delegation;
mod dropout;
mod exchange;
mod module_adamw_checkpoint;
mod module_checkpoint;
mod module_momentum_checkpoint;
mod module_owner;
mod module_plan;
mod module_session;
mod module_state;
mod momentum_checkpoint;
mod momentum_module_plan;
mod momentum_plan;
mod native_cpu_adamw_runtime;
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
pub use self::cpu_adamw_runtime::CpuCompiledAdamW;
use self::cpu_adamw_runtime::{PendingAdamWStep, adamw_step_result};
pub use self::cpu_momentum_runtime::CpuCompiledMomentumSgd;
use self::cpu_training_program::CpuCompiledTrainingProgram;
use self::cpu_training_step::{CompiledStepOutputSelection, CompiledStepReplayRequest};
pub use self::dropout::{CompiledDropoutConfig, CompiledDropoutKey};
use self::dropout::{CompiledDropoutState, CompiledDropoutStream, expected_dropout_counter};
pub use self::exchange::*;
use self::exchange::{CompiledAdamWWindowLossValue, CompiledInputPolicy};
pub use self::module_adamw_checkpoint::CompiledModuleAdamWCheckpoint;
pub use self::module_checkpoint::{CompiledModuleCheckpoint, CompiledModuleCheckpointPayload};
use self::module_checkpoint::{DecodedModuleCheckpoint, encode_complete_module_checkpoint};
pub use self::module_momentum_checkpoint::CompiledModuleMomentumSgdCheckpoint;
pub use self::module_owner::{
    CompiledModuleAdamWCompileError, CompiledModuleAdamWEvaluationError,
    CompiledModuleAdamWFinishError, CompiledModuleAdamWPlan, CompiledModuleAdamWPrepareError,
    CompiledModuleAdamWRestoreError, CompiledModuleAdamWSession,
    CompiledModuleCheckpointFinishResult, CompiledModuleCompileError,
    CompiledModuleMomentumSgdCompileError, CompiledModuleMomentumSgdPlan,
    CompiledModuleMomentumSgdPrepareError, CompiledModuleMomentumSgdRestoreError,
    CompiledModulePlanError, CompiledModuleTrainingFinishError, CompiledModuleTrainingPlan,
    CompiledModuleTrainingSession,
};
use self::module_owner::{
    adamw_compile_error, adamw_evaluation_error, adamw_prepare_error, adamw_restore_error,
    momentum_sgd_compile_error, momentum_sgd_prepare_error, momentum_sgd_restore_error,
};
pub use self::module_state::TrainingParameterInit;
use self::module_state::{CompiledModuleSeal, ModuleParameterPlan};
pub use self::momentum_checkpoint::CompiledMomentumSgdCheckpoint;
pub use self::momentum_plan::CompiledMomentumSgdPlan;
pub use self::native_cpu_adamw_runtime::NativeCpuCompiledAdamW;
use self::native_cpu_adamw_runtime::{
    NativeCpuEvaluationPreparation, PreparedNativeCpuEvaluation, PreparedNativeCpuProgram,
    PreparedNativeEvaluationParameterInput,
};
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

mod metal_training_core;
mod metal_training_runtime;
use self::metal_training_core::{
    MetalCompiledTrainingPlan, MetalCompiledTrainingProgram, MetalCompiledTrainingRun,
};
pub use self::metal_training_runtime::{
    MetalCompiledAdamW, MetalCompiledAdamWCommitResult, MetalCompiledAdamWFlushResult,
    MetalCompiledAdamWPlan, MetalCompiledAdamWStepResult,
};

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
