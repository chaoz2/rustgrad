//! CPU ownership and replay for a compiled momentum-SGD program.

use super::cpu_training_program::CpuCompiledTrainingProgram;
use super::{
    CompiledCheckpointRestoreRuntime, CompiledCheckpointRuntime, CompiledMomentumSgdCheckpoint,
    CompiledMomentumSgdConfig, CompiledMomentumSgdPlan, CompiledMomentumSgdStepResult,
    CompiledTrainingCommitOnlyRuntime, CompiledTrainingRuntime, TrainingParameterInit, training,
};
use crate::{Graph, Module, NodeId, Result, TensorData};
use std::collections::BTreeMap;

/// One compiled momentum-SGD training program.
pub struct CpuCompiledMomentumSgd {
    pub(super) inner: CpuCompiledTrainingProgram,
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
    pub(super) fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner
            .step_inner(inputs, learning_rate, injected_failure)
    }

    #[cfg(test)]
    pub(super) fn commit_step_inner(
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
