//! Optimizer-neutral contracts for compiled training runtimes.
//!
//! Concrete CPU and Metal implementations remain in the parent module. This
//! module owns the public behavioral seams shared by training loops,
//! checkpointing, accumulation windows, and captured learning-rate policies.

use super::*;

/// Shared execution contract for one compiled training program.
///
/// Optimizer and backend implementations retain their concrete policy,
/// checkpoint, and device evidence through extension traits and inherent APIs.
/// This base interface owns only the common replay contract needed by a generic
/// training loop.
pub trait CompiledTrainingRuntime {
    type Step: CompiledTrainingStep;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step>;

    /// Replays one workload-owned batch with a rank-zero F32 learning rate.
    ///
    /// Batch conversion and the existing complete input validation both finish
    /// before CPU or device execution can mutate recurrent state.
    fn step_batch<B>(&mut self, batch: B, learning_rate: f32) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.step(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }

    fn step_count(&self) -> u64;

    fn capture_identity(&self) -> u64;

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;

    /// Explicitly publishes the current detached trainable-parameter frontier
    /// into any live module with the exact same canonical trainable schema.
    ///
    /// This does not synchronize frozen state, buffers, optimizer state, replay
    /// progress, or checkpoints. The source runtime remains unchanged if the
    /// snapshot or the module's one atomic replacement transaction fails.
    /// Optimizer runtimes with an authenticated compile-time freeze policy
    /// publish only their effective trainable frontier and preserve those
    /// policy-frozen host identities as well.
    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        let parameters = self.parameter_snapshots()?;
        module.load_trainable_parameters_exact(&parameters)
    }
}

pub(super) fn publish_parameters_with_freeze_policy(
    module: &dyn Module,
    parameters: BTreeMap<String, TensorData>,
    frozen_parameters: &BTreeSet<String>,
) -> Result<LoadReport> {
    if frozen_parameters.is_empty() {
        return module.load_trainable_parameters_exact(&parameters);
    }
    CompiledModuleSeal::capture(module, frozen_parameters)?.publish(module, &parameters)
}

/// Portable checkpoint capability for a compiled training runtime.
///
/// Keeping persistence separate lets non-checkpointable optimizers implement
/// the common execution contract without inventing an empty checkpoint type.
pub trait CompiledCheckpointRuntime: CompiledTrainingRuntime {
    type Checkpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint>;
}

/// Authenticated trainable-parameter frontier carried by one checkpoint value.
///
/// Owned module finalization uses this capability after taking exactly one
/// checkpoint. The parameters must be decoded from that same immutable snapshot;
/// implementations may not read a second live runtime frontier.
pub trait CompiledCheckpointParameterSnapshot {
    /// Returns the canonical trainable parameter map embedded in this snapshot.
    fn checkpoint_parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;
}

impl CompiledCheckpointParameterSnapshot for CompiledAdamWCheckpoint {
    fn checkpoint_parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        Ok(decode_adamw_checkpoint(self.as_bytes())?.parameters)
    }
}

impl CompiledCheckpointParameterSnapshot for CompiledMomentumSgdCheckpoint {
    fn checkpoint_parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        Ok(self.parameters.clone())
    }
}

/// In-place checkpoint restoration for an already prepared training runtime.
///
/// Implementations validate and prepare a detached candidate before replacing
/// live state. A failed restore leaves the runtime unchanged.
pub trait CompiledCheckpointRestoreRuntime: CompiledCheckpointRuntime {
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()>;
}

/// AdamW-specific policy and recurrent-state inspection.
///
/// CPU and Metal AdamW sessions share this extension and the same checkpoint
/// type while retaining their concrete step-result and device-report types.
pub trait CompiledAdamWRuntime:
    CompiledTrainingRuntime<Step: CompiledAdamWStep>
    + CompiledCheckpointRuntime<Checkpoint = CompiledAdamWCheckpoint>
{
    fn gradient_accumulation_steps(&self) -> u64;

    fn max_gradient_norm(&self) -> Option<f32>;

    fn loss_scale(&self) -> f32;

    /// Whether completed-window loss aggregation is captured and reported.
    fn window_loss_report_enabled(&self) -> bool {
        false
    }

    fn optimizer_step(&self) -> Result<u64>;

    fn accumulation_index(&self) -> Result<u64>;

    /// Atomically discards a retained partial gradient window. Empty windows
    /// are exact no-ops and successful microbatch replay progress never rewinds.
    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        Err(training(
            "compiled AdamW runtime does not support accumulation cancellation",
        ))
    }

    /// Stable identity of a separately captured accumulation-reset transition.
    /// Runtimes that retain historical host-side reset semantics return `None`.
    fn zero_grad_capture_identity(&self) -> Option<u64> {
        None
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;
}

/// Optimizer-neutral capability for discarding a retained gradient window.
///
/// A reset preserves successful replay progress and optimizer state. Empty
/// windows are exact no-ops. The distinct method names keep this capability
/// unambiguous beside the source-compatible AdamW extension.
pub trait CompiledTrainingWindowResetRuntime: CompiledTrainingRuntime {
    fn reset_gradient_window(&mut self) -> Result<CompiledTrainingWindowReset>;

    /// Stable identity of a separately captured state-reset transition.
    /// Runtimes with no distinct captured transition return `None`.
    fn gradient_window_reset_capture_identity(&self) -> Option<u64> {
        None
    }
}

/// Optimizer-neutral accumulation-window configuration and live progress.
///
/// The reset supertrait makes cancellation part of the same structural
/// capability without coupling the loop to optimizer policy or checkpointing.
pub trait CompiledTrainingWindowRuntime:
    CompiledTrainingWindowResetRuntime<Step: CompiledTrainingWindowStep>
{
    /// Positive number of microbatches in one complete gradient window.
    fn gradient_window_size(&self) -> u64;

    /// Microbatches currently retained toward the next complete window.
    fn pending_microbatch_count(&self) -> Result<u64>;
}

/// Optimizer-neutral capability for committing a retained partial gradient
/// window with an external learning rate.
///
/// The transition consumes no workload batch and leaves replay/dropout progress
/// unchanged. A nonempty call commits the retained window atomically; an empty
/// call is an exact no-op. Optimizer-specific extensions may expose additional
/// update, clipping, loss, or backend execution evidence on the result.
pub trait CompiledTrainingWindowCommitRuntime: CompiledTrainingWindowRuntime {
    type WindowCommit: CompiledTrainingWindowCommit;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit>;

    /// Stable identity of the separately captured partial-window transition.
    fn partial_window_commit_capture_identity(&self) -> Option<u64>;
}

/// Optimizer-neutral capability for committing replay state without returning
/// graph-named outputs.
///
/// Loss and enabled diagnostic reports remain part of the result and retain
/// their ordinary transition validation. This capability changes observation
/// only; it does not define a distinct capture or checkpoint identity.
pub trait CompiledTrainingCommitOnlyRuntime: CompiledTrainingRuntime {
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step>;

    fn commit_step_batch<B>(&mut self, batch: B, learning_rate: f32) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.commit_step(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }
}

/// CPU AdamW capability for committing replay state without returning
/// graph-named outputs.
///
/// Loss and enabled diagnostic reports remain part of the result and retain
/// their ordinary transition validation. This capability changes observation
/// only; it does not define a distinct capture or checkpoint identity.
pub trait CompiledAdamWCommitOnlyRuntime: CompiledAdamWRuntime {
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step>;

    fn step_batch_commit_only<B>(&mut self, batch: B, learning_rate: f32) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.step_commit_only(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }
}

/// AdamW compatibility extension for committing an incomplete gradient window.
///
/// The transition consumes only the live parameter, moment, accumulator, and
/// optimizer-cursor frontier plus the explicit learning rate. It cannot reach
/// a training batch, forward/backward graph, or recurrent dropout state.
/// CPU executes the exact retained mixed capture. Strict Metal reuses its
/// authenticated state-only projection against the live epoch banks without
/// changing user code or staging gradients through the host. Generic loops use
/// [`CompiledTrainingWindowCommitRuntime`]; this extension retains AdamW update,
/// clipping, and window-loss diagnostics.
pub trait CompiledAdamWFlushRuntime: CompiledAdamWRuntime {
    type Flush: CompiledAdamWFlush;

    /// Atomically averages the currently retained `k < N` gradients by `k`,
    /// clips once, performs one AdamW update, and resets the partial window.
    /// An empty window is an exact no-op.
    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush>;

    /// Stable identity of the authenticated auxiliary state transition.
    fn flush_capture_identity(&self) -> Option<u64>;
}

/// Optimizer-neutral replay driven by a runtime-owned learning-rate policy.
///
/// The configured policy supplies the rate, so each call binds only the
/// workload inputs. External-rate replay remains available through
/// [`CompiledTrainingRuntime`].
pub trait CompiledTrainingRatePolicyRuntime: CompiledTrainingRuntime {
    fn step_with_rate_policy(&mut self, inputs: BTreeMap<String, TensorData>)
    -> Result<Self::Step>;

    fn step_batch_with_rate_policy<B>(&mut self, batch: B) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.step_with_rate_policy(batch.into_compiled_inputs()?)
    }
}

/// Partial-window commit driven by the same runtime-owned learning-rate policy
/// as [`CompiledTrainingRatePolicyRuntime`].
///
/// This is intentionally separate from external-rate window commit so a backend
/// can support one policy boundary without claiming the other.
pub trait CompiledTrainingRatePolicyWindowCommitRuntime:
    CompiledTrainingRatePolicyRuntime + CompiledTrainingWindowCommitRuntime
{
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit>;
}

/// Commit-only replay driven by a runtime-owned learning-rate policy.
pub trait CompiledTrainingRatePolicyCommitOnlyRuntime: CompiledTrainingRatePolicyRuntime {
    fn commit_step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step>;

    fn commit_step_batch_with_rate_policy<B>(&mut self, batch: B) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.commit_step_with_rate_policy(batch.into_compiled_inputs()?)
    }
}

/// CPU capability for replaying AdamW with a captured learning-rate policy.
///
/// Metal deliberately does not implement this capability until it can admit
/// the same policy without weakening strict planning.
pub trait CompiledScheduledAdamWRuntime: CompiledAdamWRuntime {
    type ScheduledFlush: CompiledAdamWFlush;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr>;

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step>;

    fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush>;
}

/// Captured-rate counterpart of [`CompiledAdamWCommitOnlyRuntime`].
pub trait CompiledScheduledAdamWCommitOnlyRuntime:
    CompiledAdamWCommitOnlyRuntime + CompiledScheduledAdamWRuntime
{
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step>;

    fn step_batch_commit_only_scheduled<B>(&mut self, batch: B) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.step_commit_only_scheduled(batch.into_compiled_inputs()?)
    }
}
