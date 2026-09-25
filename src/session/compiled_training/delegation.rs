//! Sealed delegation from module-owning sessions to their compiled runtimes.

use super::*;

/// Private interface shared by wrappers that own a compiled runtime.
///
/// Keeping this sealed lets the public runtime traits delegate once without
/// exposing wrapper layout or generating one implementation per wrapper.
pub trait RuntimeDelegate {
    type Runtime;

    fn runtime(&self) -> &Self::Runtime;

    fn runtime_mut(&mut self) -> &mut Self::Runtime;
}

impl<M, R> RuntimeDelegate for CompiledModuleTrainingSession<M, R> {
    type Runtime = R;

    fn runtime(&self) -> &Self::Runtime {
        &self.runtime
    }

    fn runtime_mut(&mut self) -> &mut Self::Runtime {
        &mut self.runtime
    }
}

impl<T> CompiledTrainingRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingRuntime,
{
    type Step = <T::Runtime as CompiledTrainingRuntime>::Step;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.runtime_mut().step(inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.runtime().step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.runtime().capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime().parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        self.runtime().publish_parameters(module)
    }
}

impl<T> CompiledCheckpointRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledCheckpointRuntime,
{
    type Checkpoint = <T::Runtime as CompiledCheckpointRuntime>::Checkpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.runtime().checkpoint()
    }
}

impl<T> CompiledCheckpointRestoreRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledCheckpointRestoreRuntime,
{
    fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.runtime_mut().restore_checkpoint_in_place(checkpoint)
    }
}

impl<T> CompiledEvaluationRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledEvaluationRuntime,
{
    type Evaluation = <T::Runtime as CompiledEvaluationRuntime>::Evaluation;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        self.runtime_mut().evaluate(inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.runtime().evaluation_capture_identity()
    }
}

impl<T> CompiledTrainingCommitOnlyRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingCommitOnlyRuntime,
{
    fn commit_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.runtime_mut().commit_step(inputs, learning_rate)
    }
}

impl<T> CompiledTrainingWindowResetRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingWindowResetRuntime,
{
    fn reset_gradient_window(&mut self) -> Result<CompiledTrainingWindowReset> {
        self.runtime_mut().reset_gradient_window()
    }

    fn gradient_window_reset_capture_identity(&self) -> Option<u64> {
        self.runtime().gradient_window_reset_capture_identity()
    }
}

impl<T> CompiledTrainingWindowRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingWindowRuntime,
{
    fn gradient_window_size(&self) -> u64 {
        self.runtime().gradient_window_size()
    }

    fn pending_microbatch_count(&self) -> Result<u64> {
        self.runtime().pending_microbatch_count()
    }
}

impl<T> CompiledTrainingWindowCommitRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingWindowCommitRuntime,
{
    type WindowCommit = <T::Runtime as CompiledTrainingWindowCommitRuntime>::WindowCommit;

    fn commit_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::WindowCommit> {
        self.runtime_mut().commit_partial_window(learning_rate)
    }

    fn partial_window_commit_capture_identity(&self) -> Option<u64> {
        self.runtime().partial_window_commit_capture_identity()
    }
}

impl<T> CompiledTrainingRatePolicyRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingRatePolicyRuntime,
{
    fn step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.runtime_mut().step_with_rate_policy(inputs)
    }
}

impl<T> CompiledTrainingRatePolicyCommitOnlyRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingRatePolicyCommitOnlyRuntime,
{
    fn commit_step_with_rate_policy(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.runtime_mut().commit_step_with_rate_policy(inputs)
    }
}

impl<T> CompiledTrainingRatePolicyWindowCommitRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledTrainingRatePolicyWindowCommitRuntime,
{
    fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
        self.runtime_mut().commit_partial_window_with_rate_policy()
    }
}

impl<T> CompiledAdamWRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledAdamWRuntime,
{
    fn gradient_accumulation_steps(&self) -> u64 {
        self.runtime().gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.runtime().max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.runtime().loss_scale()
    }

    fn window_loss_report_enabled(&self) -> bool {
        self.runtime().window_loss_report_enabled()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.runtime().optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.runtime().accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.runtime_mut().zero_grad()
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.runtime().zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime().first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime().second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime().gradient_accumulator_snapshots()
    }
}

impl<T> CompiledAdamWCommitOnlyRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledAdamWCommitOnlyRuntime,
{
    fn step_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.runtime_mut().step_commit_only(inputs, learning_rate)
    }
}

impl<T> CompiledAdamWFlushRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledAdamWFlushRuntime,
{
    type Flush = <T::Runtime as CompiledAdamWFlushRuntime>::Flush;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        self.runtime_mut().flush_partial_window(learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        self.runtime().flush_capture_identity()
    }
}

impl<T> CompiledScheduledAdamWRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledScheduledAdamWRuntime,
{
    type ScheduledFlush = <T::Runtime as CompiledScheduledAdamWRuntime>::ScheduledFlush;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.runtime().captured_multi_step_lr()
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        self.runtime_mut().step_scheduled(inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        self.runtime_mut().flush_partial_window_scheduled()
    }
}

impl<T> CompiledScheduledAdamWCommitOnlyRuntime for T
where
    T: RuntimeDelegate,
    T::Runtime: CompiledScheduledAdamWCommitOnlyRuntime,
{
    fn step_commit_only_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        self.runtime_mut().step_commit_only_scheduled(inputs)
    }
}
