//! Resource-free momentum-SGD compilation and checkpoint restoration.

use super::*;

/// Resource-free compiled momentum-SGD program ready for CPU preparation.
///
/// Compilation owns graph construction, differentiation, scheduling, capture,
/// and recurrent-state admission. The immutable plan can prepare independent
/// runtimes or restore a checkpoint frontier without rebuilding that program.
#[derive(Clone)]
pub struct CompiledMomentumSgdPlan {
    pub(super) inner: CompiledTrainingPlan,
    program_identity: u64,
    compile_phases: Option<CompiledTrainingCompileObservation>,
}

impl CompiledMomentumSgdPlan {
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
        let (inner, compile_phases) =
            CompiledTrainingPlan::compile_observed(MomentumProgram { config }, parameters, build)?;
        Self::from_compiled_inner(inner, compile_phases)
    }

    /// Compiles a module forward without allocating a runtime or taking
    /// ownership of the host module.
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

    /// Recompiles a matching program and restores its exact checkpoint
    /// frontier without allocating a runtime.
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
        Self::compile(config, parameters, build)?.restore_checkpoint_owned(checkpoint)
    }

    /// Recompiles a module-bound program and restores its exact checkpoint
    /// frontier without mutating the host module or allocating a runtime.
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
        Self::compile_module_from_parameter_plan_checkpoint(
            config,
            module,
            parameter_plan,
            checkpoint,
            build,
        )
    }

    pub(super) fn compile_module_from_parameter_plan_checkpoint<M, F>(
        config: CompiledMomentumSgdConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
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
        validate_module_checkpoint_schema(&parameter_plan, checkpoint)?;
        let parameters = checkpoint
            .parameters
            .iter()
            .map(|(name, value)| TrainingParameterInit::new(name.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()?;
        Self::compile(config, parameters, |graph, inputs, parameters| {
            parameter_plan.lower(graph, parameters, |graph| build(module, graph, inputs))
        })?
        .restore_checkpoint_owned(checkpoint)
    }

    /// Returns an independent plan restored to `checkpoint` without rebuilding
    /// the graph, derivatives, schedules, or capture. The source plan remains
    /// unchanged on success and failure.
    pub fn restore_checkpoint(&self, checkpoint: &CompiledMomentumSgdCheckpoint) -> Result<Self> {
        self.clone().restore_checkpoint_owned(checkpoint)
    }

    pub(super) fn restore_checkpoint_owned(
        self,
        checkpoint: &CompiledMomentumSgdCheckpoint,
    ) -> Result<Self> {
        if self.program_identity != checkpoint.capture_identity {
            return Err(training(
                "compiled momentum-SGD checkpoint capture identity mismatch",
            ));
        }
        validate_checkpoint_names(checkpoint)?;
        let values = checkpoint
            .parameters
            .iter()
            .map(|(name, value)| (RecurrentStateKey::parameter(name), value.clone()))
            .chain(
                checkpoint
                    .momenta
                    .iter()
                    .map(|(name, value)| (momentum_key(name), value.clone())),
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
                    .map(|(name, version)| (momentum_key(name), *version)),
            )
            .collect();
        Ok(Self {
            inner: self
                .inner
                .restore_frontier_with_versions(checkpoint.step, values, versions)?,
            program_identity: self.program_identity,
            compile_phases: self.compile_phases,
        })
    }

    /// Prepares an independent graph-free CPU runtime from this plan's exact
    /// recurrent frontier.
    pub fn prepare_cpu(&self) -> Result<CpuCompiledMomentumSgd> {
        Ok(CpuCompiledMomentumSgd {
            inner: self.inner.prepare_cpu()?,
            non_finite_policy: CpuNonFinitePolicy::Propagate,
        })
    }

    pub(super) fn prepare_cpu_with_non_finite_policy(
        &self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> Result<CpuCompiledMomentumSgd> {
        Ok(CpuCompiledMomentumSgd {
            inner: self
                .inner
                .prepare_cpu_with_non_finite_policy(non_finite_policy)?,
            non_finite_policy,
        })
    }

    /// Prepares strict-native CPU replay before exposing mutable momentum state.
    pub fn prepare_native_cpu<'a>(
        &self,
        target: &NativeCpuSessionTarget<'a>,
    ) -> Result<NativeCpuCompiledMomentumSgd<'a>> {
        let inner = self.prepare_cpu_with_non_finite_policy(target.non_finite_policy())?;
        NativeCpuCompiledMomentumSgd::prepare(inner, target.executor(), target.is_vectorized())
    }

    /// Prepares this plan through a concrete target without introducing a
    /// runtime backend enum.
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

    pub const fn capture_identity(&self) -> u64 {
        self.program_identity
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step
    }

    /// Returns immutable logical work and recurrent-state facts without
    /// preparing a runtime or exposing the raw mixed capture.
    pub fn inspection(&self) -> Result<CompiledTrainingInspection> {
        let recurrent_state = checked_recurrent_state_extent(
            self.inner
                .state_values
                .values()
                .map(checked_bytes)
                .collect::<Result<Vec<_>>>()?,
        )?;
        Ok(CompiledTrainingInspection::new(
            self.step_count(),
            (
                self.capture_identity(),
                self.inner.recurrent_capture.execution_plan().clone(),
            ),
            None,
            None,
            None,
            None,
            recurrent_state,
        )
        .with_compile_phases(self.compile_phases.clone()))
    }

    /// Backend-neutral graph/autograd/lowering/capture observations retained
    /// by the freshly compiled plan. Artifact-restored plans deliberately
    /// carry no synthetic compilation evidence.
    pub fn compile_phases(&self) -> Option<&CompiledTrainingCompileObservation> {
        self.compile_phases.as_ref()
    }

    fn from_compiled_inner(
        inner: CompiledTrainingPlan,
        compile_phases: CompiledTrainingCompileObservation,
    ) -> Result<Self> {
        let mut plan = Self::from_inner(inner)?;
        plan.compile_phases = Some(compile_phases);
        Ok(plan)
    }

    pub(super) fn from_inner(inner: CompiledTrainingPlan) -> Result<Self> {
        let program_identity = inner.capture_identity()?;
        Ok(Self {
            inner,
            program_identity,
            compile_phases: None,
        })
    }
}

impl<'a> SessionTarget<&'a CompiledMomentumSgdPlan> for CpuSessionTarget {
    type Session = CpuCompiledMomentumSgd;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledMomentumSgdPlan) -> Result<Self::Session> {
        plan.prepare_cpu()
    }
}

impl<'a> SessionTarget<&'a CompiledMomentumSgdPlan> for ConfiguredCpuSessionTarget {
    type Session = CpuCompiledMomentumSgd;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledMomentumSgdPlan) -> Result<Self::Session> {
        plan.prepare_cpu_with_non_finite_policy(self.non_finite_policy())
    }
}

impl<'executor> SessionTarget<&CompiledMomentumSgdPlan> for NativeCpuSessionTarget<'executor> {
    type Session = NativeCpuCompiledMomentumSgd<'executor>;
    type Error = Error;

    fn prepare(&self, plan: &CompiledMomentumSgdPlan) -> Result<Self::Session> {
        plan.prepare_native_cpu(self)
    }
}

fn validate_checkpoint_names(checkpoint: &CompiledMomentumSgdCheckpoint) -> Result<()> {
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
    Ok(())
}

fn validate_module_checkpoint_schema(
    parameter_plan: &ModuleParameterPlan,
    checkpoint: &CompiledMomentumSgdCheckpoint,
) -> Result<()> {
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
    Ok(())
}
