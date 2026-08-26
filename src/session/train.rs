//! Fresh-graph CPU bridge for one-input classification modules.

use crate::nn::{
    Mode, ModeModuleForward, Module, ModuleForward, Parameter, PendingModeEffects,
    RealizedBatchNormStats,
};
use crate::optim::{Gradient, LearningRateScheduler, Optimizer};
use crate::{
    Backend, CompileTrace, CpuBackend, DType, Error, Graph, LossOptions, Reduction, Result,
    TensorData, cross_entropy,
};
use std::collections::{BTreeMap, BTreeSet};

/// Existing sparse categorical cross-entropy configured for one module step.
///
/// `Reduction::None` is rejected because first-order reverse mode requires a
/// one-element loss. The graph loss implementation owns class-axis, target
/// shape, target dtype, ignore-index, and label-smoothing validation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModuleCrossEntropy {
    pub options: LossOptions,
}

/// Inspectable result of one completed module train or evaluation request.
#[derive(Clone, Debug)]
pub struct ModuleStepResult {
    loss: f64,
    logits: TensorData,
    trace: CompileTrace,
    parameter_versions: BTreeMap<String, u64>,
    optimizer_step: u64,
    scheduler_epoch: u64,
}

impl ModuleStepResult {
    pub const fn loss(&self) -> f64 {
        self.loss
    }
    pub fn logits(&self) -> &TensorData {
        &self.logits
    }
    pub fn trace(&self) -> &CompileTrace {
        &self.trace
    }
    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }
    pub const fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }
    pub const fn scheduler_epoch(&self) -> u64 {
        self.scheduler_epoch
    }
}

/// A small CPU-only bridge from a graph-composed module to the established
/// evaluated-gradient optimizer and scheduler APIs.
///
/// It never retains a graph. Each call snapshots current parameter values into
/// a new graph, evaluates all loss/gradient nodes through [`CpuBackend`], then
/// invokes the existing optimizer and scheduler. This makes an old graph
/// incapable of observing a later parameter replacement.
pub struct CpuModuleTrainer<'a, M: ModuleForward + ?Sized> {
    module: &'a M,
    optimizer: &'a mut Optimizer,
    scheduler: &'a mut LearningRateScheduler,
    loss: ModuleCrossEntropy,
}

impl<'a, M: ModuleForward + ?Sized> CpuModuleTrainer<'a, M> {
    /// Connects a module to existing optimizer/scheduler ownership.
    ///
    /// Parameter names must exactly equal the module's deterministic canonical
    /// traversal names. This rejects a missing, extra, or differently named
    /// optimizer parameter before any graph execution or update.
    pub fn new(
        module: &'a M,
        optimizer: &'a mut Optimizer,
        scheduler: &'a mut LearningRateScheduler,
        loss: ModuleCrossEntropy,
    ) -> Result<Self> {
        if loss.options.reduction == Reduction::None {
            return Err(training("module training needs a scalar loss"));
        }
        let parameters = training_parameters(module)?;
        let actual = optimizer
            .parameter_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        let expected = parameters.keys().cloned().collect::<BTreeSet<_>>();
        if actual != expected || actual.len() != expected.len() {
            return Err(training(
                "optimizer parameter names do not match module traversal",
            ));
        }
        // This is a no-mutation preflight: metric-only schedulers cannot be
        // advanced by `train_step`, so reject them before an optimizer update.
        scheduler.validate_step_without_metric()?;
        Ok(Self {
            module,
            optimizer,
            scheduler,
            loss,
        })
    }

    /// Builds and executes a fresh static F32 graph, then updates parameters
    /// through the existing version-checked optimizer and advances scheduler.
    pub fn train_step(
        &mut self,
        input: TensorData,
        target: TensorData,
    ) -> Result<ModuleStepResult> {
        let planned = self.plan(input, target, true)?;
        // Every CPU execution happens before any visible parameter/optimizer
        // mutation. Existing Optimizer::step rechecks parameter identities and
        // versions, and scheduler preflight in `new` ensures `step` is valid.
        self.optimizer.step(&planned.gradients)?;
        self.scheduler.step(self.optimizer)?;
        self.result(planned.loss, planned.logits, planned.trace)
    }

    /// Builds and executes a fresh static F32 graph without changing module,
    /// optimizer, scheduler, random state, or checkpoint identity.
    pub fn evaluate(&self, input: TensorData, target: TensorData) -> Result<ModuleStepResult> {
        let planned = self.plan(input, target, false)?;
        self.result(planned.loss, planned.logits, planned.trace)
    }

    pub fn module(&self) -> &M {
        self.module
    }
    pub fn optimizer(&self) -> &Optimizer {
        self.optimizer
    }
    pub fn scheduler(&self) -> &LearningRateScheduler {
        self.scheduler
    }

    fn plan(&self, input: TensorData, target: TensorData, gradients: bool) -> Result<PlannedStep> {
        if input.dtype() != DType::F32 {
            return Err(training("module CPU step input must have dtype F32"));
        }
        if !target.dtype().is_integer() {
            return Err(training(
                "module CPU step target must have an integer dtype",
            ));
        }
        let parameters = training_parameters(self.module)?;
        let mut graph = Graph::new();
        let input_node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
        let target_node =
            graph.input_dtype("module_target", target.shape().clone(), target.dtype());
        let logits = self.module.forward(&mut graph, input_node)?;
        if graph.dtype(logits)? != DType::F32 {
            return Err(training("module CPU step logits must have dtype F32"));
        }
        let loss = cross_entropy(&mut graph, logits, target_node, self.loss.options)?;
        let gradient_nodes = if gradients {
            parameters
                .iter()
                .map(|(name, parameter)| {
                    Ok((name.clone(), graph.grad(loss, parameter.node(&graph)?)?))
                })
                .collect::<Result<BTreeMap<_, _>>>()?
        } else {
            BTreeMap::new()
        };
        let mut bindings = self.module.input_bindings(&graph)?;
        bindings.insert("module_input".to_string(), input);
        bindings.insert("module_target".to_string(), target);
        let cpu = CpuBackend;
        let logits_data = cpu.execute(&graph, logits, &bindings)?;
        let loss_data = cpu.execute(&graph, loss, &bindings)?;
        let gradients = gradient_nodes
            .into_iter()
            .map(|(name, node)| {
                let parameter = &parameters[&name];
                Ok((
                    name,
                    Gradient::for_parameter(parameter, cpu.execute(&graph, node, &bindings)?)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(PlannedStep {
            loss: loss_data.scalar_at(0).as_f64(),
            logits: logits_data,
            trace: graph.trace(loss)?,
            gradients,
        })
    }

    fn result(
        &self,
        loss: f64,
        logits: TensorData,
        trace: CompileTrace,
    ) -> Result<ModuleStepResult> {
        let parameter_versions = training_parameters(self.module)?
            .into_iter()
            .map(|(name, parameter)| Ok((name, parameter.version()?)))
            .collect::<Result<_>>()?;
        Ok(ModuleStepResult {
            loss,
            logits,
            trace,
            parameter_versions,
            optimizer_step: self.optimizer.step_count(),
            scheduler_epoch: self.scheduler.epoch(),
        })
    }
}

/// CPU-only one-input classification workflow for an explicitly mode-aware
/// module.  Unlike [`CpuModuleTrainer`], this type never infers a mode and
/// never hides stateful work: training realizes every pending-stat node and
/// atomically commits those buffers with the prepared optimizer replacement.
pub struct CpuModeModuleTrainer<'a, M: ModeModuleForward + ?Sized> {
    module: &'a M,
    optimizer: &'a mut Optimizer,
    scheduler: &'a mut LearningRateScheduler,
    loss: ModuleCrossEntropy,
}

impl<'a, M: ModeModuleForward + ?Sized> CpuModeModuleTrainer<'a, M> {
    /// Connects a mode-aware module to the existing host optimizer and
    /// scheduler ownership.  This performs the same no-mutation traversal and
    /// scheduler preflight as the stateless trainer.
    pub fn new(
        module: &'a M,
        optimizer: &'a mut Optimizer,
        scheduler: &'a mut LearningRateScheduler,
        loss: ModuleCrossEntropy,
    ) -> Result<Self> {
        if loss.options.reduction == Reduction::None {
            return Err(training("module training needs a scalar loss"));
        }
        let parameters = training_parameters(module)?;
        let actual = optimizer
            .parameter_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        let expected = parameters.keys().cloned().collect::<BTreeSet<_>>();
        if actual != expected || actual.len() != expected.len() {
            return Err(training(
                "optimizer parameter names do not match module traversal",
            ));
        }
        scheduler.validate_step_without_metric()?;
        Ok(Self {
            module,
            optimizer,
            scheduler,
            loss,
        })
    }

    /// Realizes output, loss, gradients, and all pending BatchNorm statistics
    /// before preparing any visible state transition.  Optimizer and scheduler
    /// successors are computed on clones; their parameter replacements then
    /// commit together with running buffers under one existing all-lock
    /// transaction.  Assigning the precomputed host candidates cannot fail.
    pub fn train_step(
        &mut self,
        input: TensorData,
        target: TensorData,
    ) -> Result<ModuleStepResult> {
        let planned = self.plan(input, target, Mode::Training, true)?;
        let mut prepared_optimizer = self.optimizer.prepare_step(&planned.gradients)?;
        let mut next_scheduler = self.scheduler.clone();
        next_scheduler.step(prepared_optimizer.optimizer_mut())?;
        let (next_optimizer, restores) = prepared_optimizer.into_parts();
        planned
            .pending
            .commit_batchnorm_with(planned.statistics, restores)?;
        *self.optimizer = next_optimizer;
        *self.scheduler = next_scheduler;
        self.result(planned.loss, planned.logits, planned.trace)
    }

    /// Evaluation selects [`Mode::Eval`] explicitly and rejects any accidental
    /// pending effect rather than mutating running buffers implicitly.
    pub fn evaluate(&self, input: TensorData, target: TensorData) -> Result<ModuleStepResult> {
        let planned = self.plan(input, target, Mode::Eval, false)?;
        if !planned.pending.is_empty() {
            return Err(training("evaluation produced pending mode effects"));
        }
        self.result(planned.loss, planned.logits, planned.trace)
    }

    pub fn module(&self) -> &M {
        self.module
    }

    pub fn optimizer(&self) -> &Optimizer {
        self.optimizer
    }

    pub fn scheduler(&self) -> &LearningRateScheduler {
        self.scheduler
    }

    fn plan<'b>(
        &'b self,
        input: TensorData,
        target: TensorData,
        mode: Mode,
        gradients: bool,
    ) -> Result<PlannedModeStep<'b>> {
        if input.dtype() != DType::F32 {
            return Err(training("module CPU step input must have dtype F32"));
        }
        if !target.dtype().is_integer() {
            return Err(training(
                "module CPU step target must have an integer dtype",
            ));
        }
        let parameters = training_parameters(self.module)?;
        let mut graph = Graph::new();
        let input_node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
        let target_node =
            graph.input_dtype("module_target", target.shape().clone(), target.dtype());
        let mode_output = self.module.forward_mode(&mut graph, input_node, mode)?;
        if graph.dtype(mode_output.output)? != DType::F32 {
            return Err(training("module CPU step logits must have dtype F32"));
        }
        let loss = cross_entropy(&mut graph, mode_output.output, target_node, self.loss.options)?;
        let gradient_nodes = if gradients {
            parameters
                .iter()
                .map(|(name, parameter)| {
                    Ok((name.clone(), graph.grad(loss, parameter.node(&graph)?)?))
                })
                .collect::<Result<BTreeMap<_, _>>>()?
        } else {
            BTreeMap::new()
        };
        let stat_nodes = mode_output.pending.batchnorm_stat_nodes();
        let mut bindings = self.module.input_bindings(&graph)?;
        bindings.insert("module_input".to_string(), input);
        bindings.insert("module_target".to_string(), target);
        let cpu = CpuBackend;
        let logits_data = cpu.execute(&graph, mode_output.output, &bindings)?;
        let loss_data = cpu.execute(&graph, loss, &bindings)?;
        let gradients = gradient_nodes
            .into_iter()
            .map(|(name, node)| {
                let parameter = &parameters[&name];
                Ok((
                    name,
                    Gradient::for_parameter(parameter, cpu.execute(&graph, node, &bindings)?)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let statistics = stat_nodes
            .into_iter()
            .map(|(mean, variance)| {
                Ok(RealizedBatchNormStats {
                    mean: cpu.execute(&graph, mean, &bindings)?,
                    variance: cpu.execute(&graph, variance, &bindings)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PlannedModeStep {
            loss: loss_data.scalar_at(0).as_f64(),
            logits: logits_data,
            trace: graph.trace(loss)?,
            gradients,
            pending: mode_output.pending,
            statistics,
        })
    }

    fn result(
        &self,
        loss: f64,
        logits: TensorData,
        trace: CompileTrace,
    ) -> Result<ModuleStepResult> {
        let parameter_versions = training_parameters(self.module)?
            .into_iter()
            .map(|(name, parameter)| Ok((name, parameter.version()?)))
            .collect::<Result<_>>()?;
        Ok(ModuleStepResult {
            loss,
            logits,
            trace,
            parameter_versions,
            optimizer_step: self.optimizer.step_count(),
            scheduler_epoch: self.scheduler.epoch(),
        })
    }
}

struct PlannedStep {
    loss: f64,
    logits: TensorData,
    trace: CompileTrace,
    gradients: BTreeMap<String, Gradient>,
}

struct PlannedModeStep<'a> {
    loss: f64,
    logits: TensorData,
    trace: CompileTrace,
    gradients: BTreeMap<String, Gradient>,
    pending: PendingModeEffects<'a>,
    statistics: Vec<RealizedBatchNormStats>,
}

/// Materializes the module-owned canonical traversal where this bridge needs
/// name lookups for gradients and reported versions. Filtering, lock snapshots,
/// tied-identity collapse, and duplicate-key rejection remain owned by
/// `Module::trainable_parameters`.
fn training_parameters<M: Module + ?Sized>(module: &M) -> Result<BTreeMap<String, Parameter>> {
    Ok(module.trainable_parameters()?.into_iter().collect())
}

fn training(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{
        AdaptiveAvgPool2d, BatchNorm, Conv2d, Flatten, ModeSequential, ReLU, StateKind,
    };
    use crate::optim::SgdConfig;
    use crate::{Conv2dOptions, Scalar};

    fn mode_chain() -> ModeSequential {
        let mut init = Graph::new();
        let mut chain = ModeSequential::default();
        chain.push(
            Conv2d::new_static(1, 2, [1, 1], Conv2dOptions::default(), true, 401).unwrap(),
        );
        chain.push(BatchNorm::new(&mut init, 2, 1e-5, true, true, 0.1).unwrap());
        chain.push(ReLU);
        chain.push(AdaptiveAvgPool2d::new([Some(1), Some(1)]));
        chain.push(Flatten::new(1));
        chain.push(crate::nn::Linear::new_static(2, 2, true, 402).unwrap());
        chain
    }

    fn mode_input() -> TensorData {
        TensorData::new([2, 1, 2, 2], vec![1., -2., 3., 4., 2., 1., 0., -1.]).unwrap()
    }

    fn mode_target() -> TensorData {
        TensorData::from_scalars(
            crate::Shape::from([2]),
            DType::I64,
            [Scalar::I(0), Scalar::I(1)],
        )
        .unwrap()
    }

    #[test]
    fn poisoned_module_parameter_rejects_trainer_construction_without_panic() {
        let mut graph = Graph::new();
        let model = Linear::new(&mut graph, 2, 2, true, 71).unwrap();
        let mut optimizer = Optimizer::sgd(
            vec![
                ("weight".into(), model.weight.clone()),
                ("bias".into(), model.bias.clone().unwrap()),
            ],
            SgdConfig::default(),
        )
        .unwrap();
        let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.).unwrap();
        model.weight.poison_for_test();
        assert!(matches!(
            CpuModuleTrainer::new(
                &model,
                &mut optimizer,
                &mut scheduler,
                ModuleCrossEntropy::default()
            ),
            Err(Error::ParameterLockPoisoned { .. })
        ));
    }

    struct FrozenLinear {
        linear: Linear,
        frozen: Parameter,
    }

    impl Module for FrozenLinear {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            self.linear.visit(prefix, visitor);
            let name = if prefix.is_empty() {
                "frozen".to_string()
            } else {
                format!("{prefix}.frozen")
            };
            visitor(name, &self.frozen, StateKind::Parameter);
        }
    }

    impl ModuleForward for FrozenLinear {
        fn forward(&self, graph: &mut Graph, input: crate::NodeId) -> Result<crate::NodeId> {
            self.linear.forward(graph, input)
        }
    }

    #[test]
    fn trainer_and_module_optimizer_share_trainable_traversal() {
        let model = FrozenLinear {
            linear: Linear::new_static(2, 2, true, 72).unwrap(),
            frozen: Parameter::new(TensorData::new([1], vec![1.]).unwrap(), false),
        };
        let mut optimizer = Optimizer::sgd_for_module(&model, SgdConfig::default()).unwrap();
        let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.).unwrap();
        let trainer = CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )
        .unwrap();
        assert_eq!(
            trainer.optimizer().parameter_names(),
            vec!["bias", "weight"]
        );
        assert_eq!(model.frozen.version().unwrap(), 0);
    }

    #[test]
    fn mode_trainer_keeps_eval_read_only_and_commits_batchnorm_with_optimizer() {
        let model = mode_chain();
        assert_eq!(
            model
                .state_dict()
                .unwrap()
                .tensors()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                "0.bias",
                "0.weight",
                "1.bias",
                "1.num_batches_tracked",
                "1.running_mean",
                "1.running_var",
                "1.weight",
                "5.bias",
                "5.weight",
            ]
        );
        let mut optimizer = Optimizer::sgd_for_module(&model, SgdConfig::default()).unwrap();
        let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.).unwrap();
        let before = model.state_dict().unwrap();
        let mut trainer = CpuModeModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )
        .unwrap();

        let first_eval = trainer.evaluate(mode_input(), mode_target()).unwrap();
        let second_eval = trainer.evaluate(mode_input(), mode_target()).unwrap();
        assert_eq!(first_eval.logits(), second_eval.logits());
        assert_eq!(model.state_dict().unwrap(), before);

        let trained = trainer.train_step(mode_input(), mode_target()).unwrap();
        assert_eq!(trained.optimizer_step(), 1);
        assert_eq!(trained.scheduler_epoch(), 1);
        assert_ne!(model.state_dict().unwrap(), before);
        assert_eq!(
            model
                .state_dict()
                .unwrap()
                .tensors()
                .get("1.num_batches_tracked")
                .unwrap()
                .scalar_at(0)
                .as_u64(),
            1
        );
    }

    #[test]
    fn mode_trainer_rejects_invalid_input_without_state_mutation() {
        let model = mode_chain();
        let mut optimizer = Optimizer::sgd_for_module(&model, SgdConfig::default()).unwrap();
        let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.).unwrap();
        let before_model = model.state_dict().unwrap();
        let before_optimizer = optimizer.state_dict().unwrap();
        let before_scheduler = scheduler.state_dict().unwrap();
        let mut trainer = CpuModeModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )
        .unwrap();
        assert!(trainer
            .train_step(
                TensorData::from_scalars(
                    crate::Shape::from([2, 1, 2, 2]),
                    DType::I32,
                    std::iter::repeat_n(Scalar::I(0), 8),
                )
                .unwrap(),
                mode_target(),
            )
            .is_err());
        assert_eq!(model.state_dict().unwrap(), before_model);
        assert_eq!(trainer.optimizer().state_dict().unwrap(), before_optimizer);
        assert_eq!(trainer.scheduler().state_dict().unwrap(), before_scheduler);
    }
}
