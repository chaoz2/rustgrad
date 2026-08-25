//! Fresh-graph CPU bridge for one-input classification modules.

use crate::nn::{Module, ModuleForward, Parameter};
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
        let parameters = canonical_parameters(module)?;
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
        let parameters = canonical_parameters(self.module)?;
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
        let parameter_versions = canonical_parameters(self.module)?
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

fn canonical_parameters<M: Module + ?Sized>(module: &M) -> Result<BTreeMap<String, Parameter>> {
    let mut parameters = BTreeMap::new();
    let mut identities = BTreeSet::new();
    let mut error = None;
    module.visit("", &mut |name, parameter, _| {
        if identities.insert(parameter.identity()) {
            if let Err(err) = parameter.snapshot() {
                error = Some(err);
            } else {
                parameters.insert(name, parameter.clone());
            }
        }
    });
    if let Some(error) = error {
        Err(error)
    } else {
        Ok(parameters)
    }
}

fn training(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::optim::SgdConfig;

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
}
