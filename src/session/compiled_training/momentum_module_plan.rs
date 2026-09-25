//! Owned-module momentum-SGD planning and target preparation.

use super::*;

impl<M: Module> CompiledModuleTrainingPlan<M, CompiledMomentumSgdPlan> {
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
            Ok((plan, seal)) => Ok(Self {
                module,
                plan,
                seal,
                attachment: (),
            }),
            Err(source) => Err(momentum_sgd_compile_error(module, source)),
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

    /// Recompiles a fresh module from one complete topology-authenticated
    /// momentum-SGD checkpoint without publishing into that module.
    pub fn compile_from_module_checkpoint<F>(
        config: CompiledMomentumSgdConfig,
        module: M,
        checkpoint: &CompiledModuleMomentumSgdCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleMomentumSgdCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let result: Result<(CompiledMomentumSgdPlan, CompiledModuleSeal)> = (|| {
            let decoded = checkpoint.decoded();
            let mut seal = CompiledModuleSeal::capture(&module, &BTreeSet::new())?;
            let immutable_values =
                seal.apply_module_checkpoint(decoded, decoded.optimizer.parameters())?;
            let parameter_plan = ModuleParameterPlan::new(&module, &BTreeSet::new())?
                .with_immutable_values(&immutable_values)?;
            let plan = CompiledMomentumSgdPlan::compile_module_from_parameter_plan_checkpoint(
                config,
                &module,
                parameter_plan,
                checkpoint.optimizer_checkpoint(),
                build,
            )?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal))
        })();
        match result {
            Ok((plan, seal)) => Ok(Self {
                module,
                plan,
                seal,
                attachment: (),
            }),
            Err(source) => Err(momentum_sgd_compile_error(module, source)),
        }
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
            return Err(momentum_sgd_restore_error(self, source));
        }
        match self.plan.restore_checkpoint(checkpoint) {
            Ok(plan) => {
                self.plan = plan;
                Ok(self)
            }
            Err(source) => Err(momentum_sgd_restore_error(self, source)),
        }
    }

    pub fn capture_identity(&self) -> u64 {
        self.plan.capture_identity()
    }

    pub fn step_count(&self) -> u64 {
        self.plan.step_count()
    }

    fn prepare_cpu(
        self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> std::result::Result<
        CompiledModuleTrainingSession<M, CpuCompiledMomentumSgd>,
        CompiledModuleMomentumSgdPrepareError<M, Error>,
    > {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(momentum_sgd_prepare_error(self, source));
        }
        let runtime = match self
            .plan
            .prepare_cpu_with_non_finite_policy(non_finite_policy)
        {
            Ok(runtime) => runtime,
            Err(source) => {
                return Err(momentum_sgd_prepare_error(self, source));
            }
        };
        let Self { module, seal, .. } = self;
        Ok(CompiledModuleTrainingSession::training(
            module, runtime, seal, None,
        ))
    }

    fn prepare_native_cpu<'executor>(
        self,
        target: &NativeCpuSessionTarget<'executor>,
    ) -> std::result::Result<
        CompiledModuleTrainingSession<M, NativeCpuCompiledMomentumSgd<'executor>>,
        CompiledModuleMomentumSgdPrepareError<M, Error>,
    > {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(momentum_sgd_prepare_error(self, source));
        }
        let runtime = match self.plan.prepare_native_cpu(target) {
            Ok(runtime) => runtime,
            Err(source) => return Err(momentum_sgd_prepare_error(self, source)),
        };
        let Self { module, seal, .. } = self;
        Ok(CompiledModuleTrainingSession::training(
            module, runtime, seal, None,
        ))
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

impl<'executor, M: Module> SessionTarget<CompiledModuleMomentumSgdPlan<M>>
    for NativeCpuSessionTarget<'executor>
{
    type Session = CompiledModuleTrainingSession<M, NativeCpuCompiledMomentumSgd<'executor>>;
    type Error = CompiledModuleMomentumSgdPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleMomentumSgdPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        plan.prepare_native_cpu(self)
    }
}
