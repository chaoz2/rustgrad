//! Owned-module momentum-SGD planning and target preparation.

use super::*;

/// Resource-free momentum-SGD plan paired with the exact module used to build it.
///
/// The module remains sealed and unavailable while the plan or its prepared
/// session exists. Compilation and checkpoint restoration allocate no runtime;
/// a concrete [`SessionTarget`] consumes the owner when execution resources are
/// required.
pub struct CompiledModuleMomentumSgdPlan<M> {
    module: M,
    plan: CompiledMomentumSgdPlan,
    seal: CompiledModuleSeal,
}

impl<M: Module> CompiledModuleMomentumSgdPlan<M> {
    fn build_owned<F>(
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(&M) -> Result<CompiledMomentumSgdPlan>,
    {
        let result: Result<(CompiledMomentumSgdPlan, CompiledModuleSeal)> = (|| {
            let seal = CompiledModuleSeal::capture(&module, &BTreeSet::new())?;
            let plan = build(&module)?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal))
        })();
        match result {
            Ok((plan, seal)) => Ok(Self { module, plan, seal }),
            Err(source) => Err(CompiledModuleMomentumSgdCompileError { module, source }),
        }
    }

    /// Compiles momentum-SGD from, and takes ownership of, one module value.
    pub fn compile<F>(
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
        Self::build_owned(module, |module| {
            CompiledMomentumSgdPlan::compile_module(config, module, build)
        })
    }

    /// Recompiles the same module topology and restores an authenticated
    /// parameter/momentum frontier before any runtime is allocated.
    pub fn compile_from_checkpoint<F>(
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
        Self::build_owned(module, |module| {
            CompiledMomentumSgdPlan::compile_module_from_checkpoint(
                config, module, checkpoint, build,
            )
        })
    }

    /// Restores an authenticated checkpoint without rebuilding the graph,
    /// gradients, schedules, or capture.
    ///
    /// Failure retains this complete owner for retry or preparation of its
    /// unchanged frontier.
    pub fn restore_checkpoint(
        mut self,
        checkpoint: &CompiledMomentumSgdCheckpoint,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdRestoreError<M>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleMomentumSgdRestoreError {
                plan: Box::new(self),
                source,
            });
        }
        match self.plan.restore_checkpoint(checkpoint) {
            Ok(plan) => {
                self.plan = plan;
                Ok(self)
            }
            Err(source) => Err(CompiledModuleMomentumSgdRestoreError {
                plan: Box::new(self),
                source,
            }),
        }
    }

    /// Consumes this owner into a target-specific training session.
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

    pub fn step_count(&self) -> u64 {
        self.plan.step_count()
    }

    pub(super) fn into_module(self) -> M {
        self.module
    }

    fn prepare_cpu(
        self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> std::result::Result<
        CompiledModuleTrainingSession<M, CpuCompiledMomentumSgd>,
        CompiledModuleMomentumSgdPrepareError<M, Error>,
    > {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleMomentumSgdPrepareError {
                plan: Box::new(self),
                source,
            });
        }
        let runtime = match self
            .plan
            .prepare_cpu_with_non_finite_policy(non_finite_policy)
        {
            Ok(runtime) => runtime,
            Err(source) => {
                return Err(CompiledModuleMomentumSgdPrepareError {
                    plan: Box::new(self),
                    source,
                });
            }
        };
        let Self { module, seal, .. } = self;
        Ok(CompiledModuleTrainingSession {
            module,
            runtime,
            seal,
            evaluation_capture_identity: None,
        })
    }
}

/// Recoverable checkpoint-restore failure retaining the complete owned plan.
pub struct CompiledModuleMomentumSgdRestoreError<M> {
    plan: Box<CompiledModuleMomentumSgdPlan<M>>,
    source: Error,
}

impl<M> CompiledModuleMomentumSgdRestoreError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_plan(self) -> CompiledModuleMomentumSgdPlan<M> {
        *self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleMomentumSgdPlan<M>, Error) {
        (*self.plan, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleMomentumSgdRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleMomentumSgdRestoreError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleMomentumSgdRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled momentum-SGD checkpoint restore failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleMomentumSgdRestoreError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Recoverable target-preparation failure retaining the unconsumed owned plan.
pub struct CompiledModuleMomentumSgdPrepareError<M, E> {
    plan: Box<CompiledModuleMomentumSgdPlan<M>>,
    source: E,
}

impl<M, E> CompiledModuleMomentumSgdPrepareError<M, E> {
    pub fn source_error(&self) -> &E {
        &self.source
    }

    pub fn into_plan(self) -> CompiledModuleMomentumSgdPlan<M> {
        *self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleMomentumSgdPlan<M>, E) {
        (*self.plan, self.source)
    }
}

impl<M, E: fmt::Debug> fmt::Debug for CompiledModuleMomentumSgdPrepareError<M, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleMomentumSgdPrepareError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M, E: fmt::Display> fmt::Display for CompiledModuleMomentumSgdPrepareError<M, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled momentum-SGD preparation failed: {}",
            self.source
        )
    }
}

impl<M, E: std::error::Error + 'static> std::error::Error
    for CompiledModuleMomentumSgdPrepareError<M, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<M: Module> SessionTarget<CompiledModuleMomentumSgdPlan<M>> for CpuSessionTarget {
    type Session = CompiledModuleTrainingSession<M, CpuCompiledMomentumSgd>;
    type Error = CompiledModuleMomentumSgdPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleMomentumSgdPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        plan.prepare_cpu(CpuNonFinitePolicy::Propagate)
    }
}

impl<M: Module> SessionTarget<CompiledModuleMomentumSgdPlan<M>> for ConfiguredCpuSessionTarget {
    type Session = CompiledModuleTrainingSession<M, CpuCompiledMomentumSgd>;
    type Error = CompiledModuleMomentumSgdPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleMomentumSgdPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        plan.prepare_cpu(self.non_finite_policy())
    }
}
