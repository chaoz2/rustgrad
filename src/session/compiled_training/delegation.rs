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
