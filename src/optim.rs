//! Imperative dense optimizers for versioned [`crate::Parameter`] leaves.
//!
//! Evaluate graph gradient nodes with `Module::input_bindings`, wrap each dense
//! result with [`Gradient::for_parameter`], then call [`Optimizer::step`]. A
//! step checks the captured parameter versions before replacement, so callers
//! must rebuild/evaluate the next graph cycle after an update.

use crate::nn::StateDict;
use crate::{
    DType, Error, Module, Parameter, ParameterId, Result, Scalar, Shape, TensorData,
    load_safetensors, save_safetensors,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Index;

#[derive(Clone, Debug)]
pub struct Gradient {
    pub data: TensorData,
    identity: ParameterId,
    version: u64,
}
impl Gradient {
    pub fn for_parameter(parameter: &Parameter, data: TensorData) -> Result<Self> {
        let snapshot = parameter.snapshot()?;
        Ok(Self {
            data,
            identity: snapshot.identity,
            version: snapshot.version,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SgdConfig {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub weight_decay: f64,
}
impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: false,
            weight_decay: 0.0,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct LarsConfig {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub classic: bool,
    pub pre_wd: bool,
    pub tcoef: f64,
}
impl Default for LarsConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            momentum: 0.9,
            weight_decay: 1e-4,
            nesterov: false,
            classic: true,
            pre_wd: true,
            tcoef: 0.001,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct LambConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub adam: bool,
}
/// CPU Muon configuration matching tinygrad's non-fused Muon constructor.
#[derive(Clone, Debug)]
pub struct MuonConfig {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub ns_steps: usize,
    pub ns_coefficients: Vec<f64>,
    pub nesterov: bool,
}
impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            momentum: 0.95,
            weight_decay: 0.1,
            ns_steps: 5,
            ns_coefficients: vec![3.4445, -4.775, 2.0315],
            nesterov: true,
        }
    }
}
impl Default for LambConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.,
            adam: false,
        }
    }
}
impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}
#[derive(Clone, Debug)]
pub enum OptimizerKind {
    Sgd(SgdConfig),
    Adam(AdamConfig),
    AdamW(AdamConfig),
    Lars(LarsConfig),
    Lamb(LambConfig),
    Muon(MuonConfig),
}

pub struct ParameterGroup {
    pub parameters: Vec<(String, Parameter)>,
    pub kind: OptimizerKind,
}
impl ParameterGroup {
    pub fn new(parameters: Vec<(String, Parameter)>, kind: OptimizerKind) -> Self {
        Self { parameters, kind }
    }
}
#[derive(Clone)]
struct Entry {
    name: String,
    parameter: Parameter,
    version: u64,
    group: usize,
    first_step: bool,
}
#[derive(Clone)]
enum Slots {
    Sgd(Vec<Vec<f64>>),
    Adam {
        mean: Vec<Vec<f64>>,
        variance: Vec<Vec<f64>>,
    },
}

/// Deterministically ordered, dense CPU optimizer state. It accepts only
/// explicit, already-evaluated gradients; it never owns a graph or global tape.
#[derive(Clone)]
pub struct Optimizer {
    entries: Vec<Entry>,
    groups: Vec<OptimizerKind>,
    learning_rates: Vec<f64>,
    slots: Vec<Slots>,
    step: u64,
}

/// Ordered composition of independent evaluated-gradient optimizers.
///
/// RustGrad gradients are caller-owned, so this has no tensor scheduling API:
/// [`Self::step`] routes one evaluated gradient map to every child. Construction
/// rejects duplicate parameter names and identities, avoiding ambiguous routing
/// and silent double updates.
pub struct OptimizerGroup {
    optimizers: Vec<Optimizer>,
}
impl OptimizerGroup {
    pub fn new(optimizers: Vec<Optimizer>) -> Result<Self> {
        if optimizers.is_empty() {
            return Err(invalid("optimizer group needs at least one child"));
        }
        let mut names = BTreeSet::new();
        let mut identities = BTreeSet::new();
        for optimizer in &optimizers {
            for entry in &optimizer.entries {
                if !names.insert(entry.name.clone()) {
                    return Err(invalid("optimizer group has duplicate parameter name"));
                }
                if !identities.insert(entry.parameter.identity()) {
                    return Err(invalid(
                        "optimizer group has overlapping parameter identity",
                    ));
                }
            }
        }
        Ok(Self { optimizers })
    }
    pub fn len(&self) -> usize {
        self.optimizers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.optimizers.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&Optimizer> {
        self.optimizers.get(index)
    }
    /// Current mutable learning rates in deterministic child/group order.
    pub fn learning_rates(&self) -> Vec<Vec<f64>> {
        self.optimizers
            .iter()
            .map(|optimizer| optimizer.learning_rates.clone())
            .collect()
    }
    pub fn set_child_learning_rates(
        &mut self,
        child: usize,
        learning_rates: Vec<f64>,
    ) -> Result<()> {
        self.optimizers
            .get_mut(child)
            .ok_or_else(|| invalid("optimizer group child index"))?
            .set_learning_rates(learning_rates)
    }
    /// Gradients are caller-owned; this fans out the current no-op child hook.
    pub fn zero_grad(&self) {
        for optimizer in &self.optimizers {
            optimizer.zero_grad();
        }
    }
    /// Validates every child before any child can replace a parameter.
    pub fn step(&mut self, gradients: &BTreeMap<String, Gradient>) -> Result<()> {
        for optimizer in &self.optimizers {
            optimizer.validate_step(gradients)?;
        }
        for optimizer in &mut self.optimizers {
            optimizer.step(gradients)?;
        }
        Ok(())
    }
    pub fn state_dict(&self) -> Result<StateDict> {
        let mut state = StateDict::default();
        state.insert(
            "optimizer_group.config",
            TensorData::from_scalars(
                Shape::new([self.config_fingerprint().len()]),
                DType::U8,
                self.config_fingerprint()
                    .into_iter()
                    .map(|x| Scalar::U(x as u64)),
            )?,
        );
        for (index, optimizer) in self.optimizers.iter().enumerate() {
            for (key, value) in optimizer.state_dict()?.into_tensors() {
                state.insert(format!("optimizer_group.{index}.{key}"), value);
            }
        }
        Ok(state)
    }
    pub fn load_state_dict(&mut self, state: &StateDict) -> Result<()> {
        if state
            .tensors()
            .get("optimizer_group.config")
            .is_none_or(|value| {
                value.dtype() != DType::U8
                    || value.shape() != &Shape::new([self.config_fingerprint().len()])
                    || to_u8(value) != self.config_fingerprint()
            })
        {
            return Err(invalid("optimizer group config fingerprint mismatch"));
        }
        let expected = self.expected_state_keys();
        let actual = state.tensors().keys().cloned().collect::<BTreeSet<_>>();
        if let Some(key) = expected.difference(&actual).next() {
            return Err(invalid(&format!("optimizer group state missing key {key}")));
        }
        if let Some(key) = actual.difference(&expected).next() {
            return Err(invalid(&format!(
                "optimizer group state unexpected key {key}"
            )));
        }
        let child_states = self.child_states(state)?;
        for (optimizer, child) in self.optimizers.iter().zip(&child_states) {
            optimizer.validate_state_dict(child)?;
        }
        for (optimizer, child) in self.optimizers.iter_mut().zip(&child_states) {
            optimizer.load_state_dict(child)?;
        }
        Ok(())
    }
    fn expected_state_keys(&self) -> BTreeSet<String> {
        let mut keys = BTreeSet::from(["optimizer_group.config".into()]);
        for (index, optimizer) in self.optimizers.iter().enumerate() {
            for key in optimizer.expected_state_keys() {
                keys.insert(format!("optimizer_group.{index}.{key}"));
            }
        }
        keys
    }
    fn child_states(&self, state: &StateDict) -> Result<Vec<StateDict>> {
        self.optimizers
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let prefix = format!("optimizer_group.{index}.");
                let tensors: BTreeMap<String, TensorData> = state
                    .tensors()
                    .iter()
                    .filter_map(|(key, value)| {
                        key.strip_prefix(&prefix)
                            .map(|key| (key.to_string(), value.clone()))
                    })
                    .collect();
                Ok(StateDict::from(tensors))
            })
            .collect()
    }
    fn config_fingerprint(&self) -> Vec<u8> {
        let mut out = b"rustgrad-optimizer-group\0\x01".to_vec();
        out.extend_from_slice(&(self.optimizers.len() as u64).to_le_bytes());
        for optimizer in &self.optimizers {
            let fingerprint = optimizer.config_fingerprint();
            out.extend_from_slice(&(fingerprint.len() as u64).to_le_bytes());
            out.extend_from_slice(&fingerprint);
        }
        out
    }
}
impl Index<usize> for OptimizerGroup {
    type Output = Optimizer;
    fn index(&self, index: usize) -> &Self::Output {
        &self.optimizers[index]
    }
}

/// Shared host-side epoch counter. Scheduler steps compute from the current
/// counter and advance it afterwards, matching tinygrad's static helpers.
#[derive(Clone, Debug, Default)]
pub struct LrSchedulerState {
    epoch: u64,
}
impl LrSchedulerState {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlateauMode {
    Min,
    Max,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThresholdMode {
    Relative,
    Absolute,
}
#[derive(Clone, Debug)]
pub enum LearningRateScheduler {
    MultiStep {
        state: LrSchedulerState,
        milestones: Vec<u64>,
        gamma: f64,
    },
    ReduceLROnPlateau {
        state: LrSchedulerState,
        mode: PlateauMode,
        factor: f64,
        patience: u64,
        threshold: f64,
        threshold_mode: ThresholdMode,
        best: f64,
        bad_epochs: u64,
    },
    CosineAnnealing {
        state: LrSchedulerState,
        t_max: u64,
        eta_min: f64,
        eta_max: Vec<f64>,
    },
    OneCycle {
        state: LrSchedulerState,
        max_lr: f64,
        initial_lr: f64,
        min_lr: f64,
        total_steps: u64,
        pct_start: f64,
    },
}
impl LearningRateScheduler {
    pub fn multi_step(milestones: Vec<u64>, gamma: f64) -> Result<Self> {
        if !gamma.is_finite() || gamma < 0. {
            return Err(invalid("invalid MultiStepLR gamma"));
        }
        Ok(Self::MultiStep {
            state: LrSchedulerState::default(),
            milestones,
            gamma,
        })
    }
    pub fn reduce_on_plateau(
        mode: PlateauMode,
        factor: f64,
        patience: u64,
        threshold: f64,
        threshold_mode: ThresholdMode,
    ) -> Result<Self> {
        if !factor.is_finite() || factor < 0. || !threshold.is_finite() {
            return Err(invalid("invalid ReduceLROnPlateau configuration"));
        }
        let best = if mode == PlateauMode::Min {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        Ok(Self::ReduceLROnPlateau {
            state: LrSchedulerState::default(),
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            best,
            bad_epochs: 0,
        })
    }
    pub fn cosine_annealing(optimizer: &Optimizer, t_max: u64, eta_min: f64) -> Result<Self> {
        if t_max == 0 || !eta_min.is_finite() {
            return Err(invalid("invalid CosineAnnealingLR configuration"));
        }
        Ok(Self::CosineAnnealing {
            state: LrSchedulerState::default(),
            t_max,
            eta_min,
            eta_max: optimizer.learning_rates.clone(),
        })
    }
    pub fn one_cycle(
        optimizer: &mut Optimizer,
        max_lr: f64,
        div_factor: f64,
        final_div_factor: f64,
        total_steps: u64,
        pct_start: f64,
    ) -> Result<Self> {
        if !max_lr.is_finite()
            || max_lr < 0.
            || !div_factor.is_finite()
            || div_factor <= 0.
            || !final_div_factor.is_finite()
            || final_div_factor <= 0.
            || total_steps == 0
            || !(0. < pct_start && pct_start < 1.)
        {
            return Err(invalid("invalid OneCycleLR configuration"));
        }
        let initial_lr = max_lr / div_factor;
        optimizer.set_learning_rate(initial_lr)?;
        Ok(Self::OneCycle {
            state: LrSchedulerState::default(),
            max_lr,
            initial_lr,
            min_lr: initial_lr / final_div_factor,
            total_steps,
            pct_start,
        })
    }
    pub fn epoch(&self) -> u64 {
        match self {
            Self::MultiStep { state, .. }
            | Self::ReduceLROnPlateau { state, .. }
            | Self::CosineAnnealing { state, .. }
            | Self::OneCycle { state, .. } => state.epoch,
        }
    }
    pub fn step(&mut self, optimizer: &mut Optimizer) -> Result<()> {
        self.step_metric(optimizer, None)
    }
    pub fn step_metric(&mut self, optimizer: &mut Optimizer, metric: Option<f64>) -> Result<()> {
        match self {
            Self::MultiStep {
                state,
                milestones,
                gamma,
            } => {
                if milestones.contains(&state.epoch) {
                    optimizer.set_learning_rates(
                        optimizer
                            .learning_rates
                            .iter()
                            .map(|lr| lr * *gamma)
                            .collect(),
                    )?;
                }
                state.epoch += 1;
            }
            Self::ReduceLROnPlateau {
                state,
                mode,
                factor,
                patience,
                threshold,
                threshold_mode,
                best,
                bad_epochs,
            } => {
                let value = metric.ok_or_else(|| invalid("ReduceLROnPlateau needs a metric"))?;
                if !value.is_finite() {
                    return Err(invalid("invalid ReduceLROnPlateau metric"));
                }
                let boundary = match threshold_mode {
                    ThresholdMode::Relative => {
                        *best
                            * (1.
                                + if *mode == PlateauMode::Min {
                                    -*threshold
                                } else {
                                    *threshold
                                })
                    }
                    ThresholdMode::Absolute => {
                        *best
                            + if *mode == PlateauMode::Min {
                                -*threshold
                            } else {
                                *threshold
                            }
                    }
                };
                let better = if *mode == PlateauMode::Min {
                    value < boundary
                } else {
                    value > boundary
                };
                if better {
                    *best = value;
                    *bad_epochs = 0;
                } else {
                    *bad_epochs += 1;
                }
                if *bad_epochs > *patience {
                    optimizer.set_learning_rates(
                        optimizer
                            .learning_rates
                            .iter()
                            .map(|lr| lr * *factor)
                            .collect(),
                    )?;
                    *bad_epochs = 0;
                }
                state.epoch += 1;
            }
            Self::CosineAnnealing {
                state,
                t_max,
                eta_min,
                eta_max,
            } => {
                let ratio = state.epoch as f64 / *t_max as f64;
                optimizer.set_learning_rates(
                    eta_max
                        .iter()
                        .map(|max| {
                            *eta_min
                                + 0.5
                                    * (*max - *eta_min)
                                    * (1. + (ratio * std::f64::consts::PI).cos())
                        })
                        .collect(),
                )?;
                state.epoch += 1;
            }
            Self::OneCycle {
                state,
                max_lr,
                initial_lr,
                min_lr,
                total_steps,
                pct_start,
            } => {
                let split = *total_steps as f64 * *pct_start;
                let epoch = state.epoch as f64;
                let lr = if epoch < split {
                    *initial_lr + (*max_lr - *initial_lr) * epoch / split
                } else {
                    *max_lr
                        + (*min_lr - *max_lr) * (epoch - split)
                            / (*total_steps as f64 * (1. - *pct_start))
                };
                optimizer.set_learning_rate(lr)?;
                state.epoch += 1;
            }
        }
        Ok(())
    }
    pub fn state_dict(&self) -> Result<StateDict> {
        let mut state = StateDict::default();
        state.insert(
            "scheduler.config",
            TensorData::from_scalars(
                Shape::new([self.config_fingerprint().len()]),
                DType::U8,
                self.config_fingerprint()
                    .into_iter()
                    .map(|x| Scalar::U(x as u64)),
            )?,
        );
        state.insert(
            "scheduler.epoch",
            TensorData::scalar_with_dtype(Scalar::U(self.epoch()), DType::U64),
        );
        if let Self::ReduceLROnPlateau {
            best, bad_epochs, ..
        } = self
        {
            state.insert(
                "scheduler.best",
                TensorData::scalar_with_dtype(Scalar::F(*best), DType::F64),
            );
            state.insert(
                "scheduler.bad_epochs",
                TensorData::scalar_with_dtype(Scalar::U(*bad_epochs), DType::U64),
            );
        }
        Ok(state)
    }
    pub fn load_state_dict(&mut self, state: &StateDict) -> Result<()> {
        self.validate_state_dict(state)?;
        self.apply_state_dict(state)
    }
    fn validate_state_dict(&self, state: &StateDict) -> Result<()> {
        let mut expected = BTreeSet::from([
            "scheduler.config".to_string(),
            "scheduler.epoch".to_string(),
        ]);
        if matches!(self, Self::ReduceLROnPlateau { .. }) {
            expected.extend(["scheduler.best".into(), "scheduler.bad_epochs".into()]);
        }
        let actual = state.tensors().keys().cloned().collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(invalid("scheduler state keys mismatch"));
        }
        let config = &state.tensors()["scheduler.config"];
        if config.dtype() != DType::U8
            || config.shape() != &Shape::new([self.config_fingerprint().len()])
            || to_u8(config) != self.config_fingerprint()
        {
            return Err(invalid("scheduler config fingerprint mismatch"));
        }
        if state.tensors()["scheduler.epoch"].dtype() != DType::U64
            || state.tensors()["scheduler.epoch"].len() != 1
        {
            return Err(invalid("invalid scheduler epoch"));
        }
        if matches!(self, Self::ReduceLROnPlateau { .. })
            && (state.tensors()["scheduler.best"].dtype() != DType::F64
                || state.tensors()["scheduler.best"].len() != 1
                || state.tensors()["scheduler.bad_epochs"].dtype() != DType::U64
                || state.tensors()["scheduler.bad_epochs"].len() != 1)
        {
            return Err(invalid("invalid plateau scheduler state"));
        }
        Ok(())
    }
    fn apply_state_dict(&mut self, state: &StateDict) -> Result<()> {
        let epoch = state.tensors()["scheduler.epoch"].scalar_at(0).as_u64();
        match self {
            Self::MultiStep { state, .. }
            | Self::CosineAnnealing { state, .. }
            | Self::OneCycle { state, .. } => state.epoch = epoch,
            Self::ReduceLROnPlateau {
                state: scheduler_state,
                best,
                bad_epochs,
                ..
            } => {
                scheduler_state.epoch = epoch;
                *best = state.tensors()["scheduler.best"].scalar_at(0).as_f64();
                *bad_epochs = state.tensors()["scheduler.bad_epochs"]
                    .scalar_at(0)
                    .as_u64();
            }
        }
        Ok(())
    }
    fn config_fingerprint(&self) -> Vec<u8> {
        let mut out = b"rustgrad-lr-scheduler\0\x01".to_vec();
        match self {
            Self::MultiStep {
                milestones, gamma, ..
            } => {
                out.push(0);
                out.extend_from_slice(&(milestones.len() as u64).to_le_bytes());
                for value in milestones {
                    out.extend_from_slice(&value.to_le_bytes());
                }
                out.extend_from_slice(&gamma.to_le_bytes());
            }
            Self::ReduceLROnPlateau {
                mode,
                factor,
                patience,
                threshold,
                threshold_mode,
                ..
            } => {
                out.push(1);
                out.extend_from_slice(&[*mode as u8, *threshold_mode as u8]);
                out.extend_from_slice(&factor.to_le_bytes());
                out.extend_from_slice(&patience.to_le_bytes());
                out.extend_from_slice(&threshold.to_le_bytes());
            }
            Self::CosineAnnealing {
                t_max,
                eta_min,
                eta_max,
                ..
            } => {
                out.push(2);
                out.extend_from_slice(&t_max.to_le_bytes());
                out.extend_from_slice(&eta_min.to_le_bytes());
                for value in eta_max {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::OneCycle {
                max_lr,
                initial_lr,
                min_lr,
                total_steps,
                pct_start,
                ..
            } => {
                out.push(3);
                for value in [*max_lr, *initial_lr, *min_lr, *pct_start] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
                out.extend_from_slice(&total_steps.to_le_bytes());
            }
        }
        out
    }
}
/// Validates both independent state dictionaries before applying either one.
pub fn load_optimizer_scheduler_state(
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
    optimizer_state: &StateDict,
    scheduler_state: &StateDict,
) -> Result<()> {
    optimizer.validate_state_dict(optimizer_state)?;
    scheduler.validate_state_dict(scheduler_state)?;
    optimizer.apply_state_dict(optimizer_state)?;
    scheduler.apply_state_dict(scheduler_state)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParameterCheckpointStamp {
    identity: ParameterId,
    version: u64,
    trainable: bool,
}

/// Exact in-process training checkpoint for a module whose host parameters stay alive.
///
/// Module tensors are serialized through safetensors. Resume deliberately rejects
/// replacement parameters even when their names and shapes match: preserving the
/// original [`ParameterId`] values prevents optimizer slots from being attached to
/// unrelated host state. Parameter values and versions must still equal the capture;
/// fresh graphs, optimizer objects, and scheduler objects are then safe to create.
#[derive(Clone, Debug)]
pub struct TrainingCheckpoint {
    module_safetensors: Vec<u8>,
    optimizer_state: StateDict,
    scheduler_state: StateDict,
    parameter_stamps: BTreeMap<String, ParameterCheckpointStamp>,
    optimizer_ownership: BTreeMap<String, ParameterId>,
}

impl TrainingCheckpoint {
    pub fn capture(
        module: &(impl Module + ?Sized),
        optimizer: &Optimizer,
        scheduler: &LearningRateScheduler,
    ) -> Result<Self> {
        let (module_state, parameter_stamps) = checkpoint_module_state(module)?;
        let optimizer_ownership = optimizer.checkpoint_ownership();
        validate_optimizer_ownership(&parameter_stamps, &optimizer_ownership)?;
        let module_safetensors = save_safetensors(module_state.tensors(), &BTreeMap::new())?;
        Ok(Self {
            module_safetensors,
            optimizer_state: optimizer.state_dict()?,
            scheduler_state: scheduler.state_dict()?,
            parameter_stamps,
            optimizer_ownership,
        })
    }

    /// Atomically validates every checkpoint part before applying optimizer and
    /// scheduler state. The module itself is verified, not rewritten, so its
    /// identities and monotonically increasing versions remain unchanged.
    pub fn resume(
        &self,
        module: &(impl Module + ?Sized),
        optimizer: &mut Optimizer,
        scheduler: &mut LearningRateScheduler,
    ) -> Result<()> {
        let (raw_module, metadata) = load_safetensors(&self.module_safetensors)?;
        if !metadata.is_empty() {
            return Err(invalid("training checkpoint module metadata must be empty"));
        }
        let serialized_module = StateDict::from(raw_module);
        let (current_module, current_stamps) = checkpoint_module_state(module)?;
        if current_stamps != self.parameter_stamps {
            return Err(invalid(
                "training checkpoint parameter identity or version mismatch",
            ));
        }
        if current_module != serialized_module {
            return Err(invalid("training checkpoint module value mismatch"));
        }
        let current_ownership = optimizer.checkpoint_ownership();
        if current_ownership != self.optimizer_ownership {
            return Err(invalid("training checkpoint optimizer ownership mismatch"));
        }
        validate_optimizer_ownership(&current_stamps, &current_ownership)?;
        optimizer.validate_state_dict(&self.optimizer_state)?;
        scheduler.validate_state_dict(&self.scheduler_state)?;
        let mut next_optimizer = optimizer.clone();
        let mut next_scheduler = scheduler.clone();
        next_optimizer.apply_state_dict(&self.optimizer_state)?;
        next_scheduler.apply_state_dict(&self.scheduler_state)?;
        *optimizer = next_optimizer;
        *scheduler = next_scheduler;
        Ok(())
    }

    pub fn module_safetensors(&self) -> &[u8] {
        &self.module_safetensors
    }

    pub fn optimizer_state(&self) -> &StateDict {
        &self.optimizer_state
    }

    pub fn scheduler_state(&self) -> &StateDict {
        &self.scheduler_state
    }

    pub fn parameter_versions(&self) -> BTreeMap<String, u64> {
        self.parameter_stamps
            .iter()
            .map(|(name, stamp)| (name.clone(), stamp.version))
            .collect()
    }
}

fn checkpoint_module_state(
    module: &(impl Module + ?Sized),
) -> Result<(StateDict, BTreeMap<String, ParameterCheckpointStamp>)> {
    let mut tensors = BTreeMap::new();
    let mut stamps = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut error = None;
    module.visit("", &mut |name, parameter, _| {
        if seen.insert(parameter.id()) {
            match parameter.snapshot() {
                Ok(snapshot) => {
                    tensors.insert(name.clone(), snapshot.data);
                    stamps.insert(
                        name,
                        ParameterCheckpointStamp {
                            identity: snapshot.identity,
                            version: snapshot.version,
                            trainable: snapshot.trainable,
                        },
                    );
                }
                Err(err) => error = Some(err),
            }
        }
    });
    match error {
        Some(err) => Err(err),
        None => Ok((StateDict::from(tensors), stamps)),
    }
}

fn validate_optimizer_ownership(
    stamps: &BTreeMap<String, ParameterCheckpointStamp>,
    ownership: &BTreeMap<String, ParameterId>,
) -> Result<()> {
    for (name, identity) in ownership {
        let stamp = stamps
            .get(name)
            .ok_or_else(|| invalid("optimizer parameter is absent from module checkpoint"))?;
        if !stamp.trainable || stamp.identity != *identity {
            return Err(invalid("optimizer parameter identity mismatch"));
        }
    }
    Ok(())
}
/// Ordered scheduler fan-out for the matching ordered [`OptimizerGroup`].
pub struct LrSchedulerGroup {
    schedulers: Vec<LearningRateScheduler>,
}
impl LrSchedulerGroup {
    pub fn new(schedulers: Vec<LearningRateScheduler>) -> Self {
        Self { schedulers }
    }
    pub fn step(&mut self, optimizers: &mut OptimizerGroup) -> Result<()> {
        if self.schedulers.len() != optimizers.len() {
            return Err(invalid("scheduler group child count mismatch"));
        }
        for (scheduler, optimizer) in self.schedulers.iter_mut().zip(&mut optimizers.optimizers) {
            scheduler.step(optimizer)?;
        }
        Ok(())
    }
}
impl Optimizer {
    pub(crate) fn checkpoint_ownership(&self) -> BTreeMap<String, ParameterId> {
        self.entries
            .iter()
            .map(|entry| (entry.name.clone(), entry.parameter.id()))
            .collect()
    }
    pub fn new(groups: Vec<ParameterGroup>) -> Result<Self> {
        if groups.is_empty() {
            return Err(invalid("optimizer needs at least one parameter group"));
        }
        let mut entries = Vec::new();
        let mut kinds = Vec::new();
        let mut seen = BTreeSet::new();
        for (group_index, group) in groups.into_iter().enumerate() {
            validate(&group.kind)?;
            kinds.push(group.kind);
            for (name, parameter) in group.parameters {
                if !parameter.is_trainable() {
                    continue;
                }
                let snapshot = parameter.snapshot()?;
                if !snapshot.dtype.is_float() {
                    return Err(invalid("optimizer parameters must have float dtype"));
                }
                if matches!(&kinds[group_index], OptimizerKind::Muon(_))
                    && (snapshot.shape.rank() < 2 || snapshot.data.is_empty())
                {
                    return Err(invalid(
                        "Muon parameters must have rank at least two and nonzero size",
                    ));
                }
                if !seen.insert(parameter.identity()) {
                    continue;
                }
                entries.push(Entry {
                    name,
                    version: snapshot.version,
                    parameter,
                    group: group_index,
                    first_step: true,
                });
            }
        }
        if entries.is_empty() {
            return Err(invalid("optimizer needs at least one trainable parameter"));
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let slots = kinds
            .iter()
            .map(|kind| match kind {
                OptimizerKind::Sgd(_) | OptimizerKind::Lars(_) | OptimizerKind::Muon(_) => {
                    Slots::Sgd(Vec::new())
                }
                OptimizerKind::Adam(_) | OptimizerKind::AdamW(_) | OptimizerKind::Lamb(_) => {
                    Slots::Adam {
                        mean: Vec::new(),
                        variance: Vec::new(),
                    }
                }
            })
            .collect();
        let mut optimizer = Self {
            entries,
            learning_rates: kinds.iter().map(kind_learning_rate).collect(),
            groups: kinds,
            slots,
            step: 0,
        };
        optimizer.allocate_slots()?;
        Ok(optimizer)
    }
    pub fn sgd(parameters: Vec<(String, Parameter)>, config: SgdConfig) -> Result<Self> {
        Self::new(vec![ParameterGroup::new(
            parameters,
            OptimizerKind::Sgd(config),
        )])
    }
    pub fn adam(parameters: Vec<(String, Parameter)>, config: AdamConfig) -> Result<Self> {
        Self::new(vec![ParameterGroup::new(
            parameters,
            OptimizerKind::Adam(config),
        )])
    }
    pub fn adamw(parameters: Vec<(String, Parameter)>, config: AdamConfig) -> Result<Self> {
        Self::new(vec![ParameterGroup::new(
            parameters,
            OptimizerKind::AdamW(config),
        )])
    }
    pub fn lars(parameters: Vec<(String, Parameter)>, config: LarsConfig) -> Result<Self> {
        Self::new(vec![ParameterGroup::new(
            parameters,
            OptimizerKind::Lars(config),
        )])
    }
    pub fn lamb(parameters: Vec<(String, Parameter)>, config: LambConfig) -> Result<Self> {
        Self::new(vec![ParameterGroup::new(
            parameters,
            OptimizerKind::Lamb(config),
        )])
    }
    /// Constructs tinygrad-compatible non-fused CPU Muon state.
    pub fn muon(parameters: Vec<(String, Parameter)>, config: MuonConfig) -> Result<Self> {
        Self::new(vec![ParameterGroup::new(
            parameters,
            OptimizerKind::Muon(config),
        )])
    }
    pub fn step_count(&self) -> u64 {
        self.step
    }
    /// Current mutable learning rates, one per deterministic parameter group.
    pub fn learning_rates(&self) -> &[f64] {
        &self.learning_rates
    }
    pub fn set_learning_rates(&mut self, learning_rates: Vec<f64>) -> Result<()> {
        if learning_rates.len() != self.learning_rates.len()
            || learning_rates.iter().any(|lr| !lr.is_finite() || *lr < 0.)
        {
            return Err(invalid("invalid optimizer learning rates"));
        }
        self.learning_rates = learning_rates;
        Ok(())
    }
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        self.set_learning_rates(vec![learning_rate; self.learning_rates.len()])
    }
    pub fn parameter_names(&self) -> Vec<&str> {
        self.entries.iter().map(|x| x.name.as_str()).collect()
    }
    pub fn zero_grad(&self) { /* gradients are caller-owned and never retained */
    }
    fn validate_step(&self, gradients: &BTreeMap<String, Gradient>) -> Result<()> {
        let snapshots = self
            .entries
            .iter()
            .map(|entry| entry.parameter.snapshot())
            .collect::<Result<Vec<_>>>()?;
        for (entry, snapshot) in self.entries.iter().zip(&snapshots) {
            let gradient = gradients
                .get(&entry.name)
                .ok_or_else(|| invalid("missing gradient"))?;
            validate_gradient(snapshot, gradient)?;
            if gradient.version != entry.version || snapshot.version != entry.version {
                return Err(invalid("stale gradient parameter version"));
            }
        }
        Ok(())
    }
    fn allocate_slots(&mut self) -> Result<()> {
        for (group, slot) in self.slots.iter_mut().enumerate() {
            let lens = self
                .entries
                .iter()
                .filter(|x| x.group == group)
                .map(|x| x.parameter.snapshot().map(|snapshot| snapshot.data.len()))
                .collect::<Result<Vec<_>>>()?;
            match slot {
                Slots::Sgd(values) => *values = lens.into_iter().map(|n| vec![0.; n]).collect(),
                Slots::Adam { mean, variance } => {
                    *mean = lens.iter().map(|&n| vec![0.; n]).collect();
                    *variance = lens.into_iter().map(|n| vec![0.; n]).collect()
                }
            }
        }
        Ok(())
    }
    pub fn step(&mut self, gradients: &BTreeMap<String, Gradient>) -> Result<()> {
        // Snapshot every parameter before mutating any parameter or optimizer slot.
        // This keeps graph/optimizer computation lock-free and writes one-at-a-time.
        let snapshots = self
            .entries
            .iter()
            .map(|entry| entry.parameter.snapshot())
            .collect::<Result<Vec<_>>>()?;
        for (entry, snapshot) in self.entries.iter().zip(&snapshots) {
            let gradient = gradients
                .get(&entry.name)
                .ok_or_else(|| invalid("missing gradient"))?;
            validate_gradient(snapshot, gradient)?;
            if gradient.version != entry.version || snapshot.version != entry.version {
                return Err(invalid("stale gradient parameter version"));
            }
        }
        let mut positions = vec![0usize; self.groups.len()];
        let next_step = self.step.wrapping_add(1);
        for (entry, snapshot) in self.entries.iter_mut().zip(snapshots) {
            let gradient = &gradients[&entry.name];
            let values = to_f64(&snapshot.data);
            let grad = to_f64(&gradient.data);
            let pos = positions[entry.group];
            positions[entry.group] += 1;
            let learning_rate = self.learning_rates[entry.group];
            let updated = match (
                self.groups[entry.group].clone(),
                &mut self.slots[entry.group],
            ) {
                (OptimizerKind::Sgd(mut config), Slots::Sgd(momentum)) => {
                    config.lr = learning_rate;
                    sgd(values, grad, &mut momentum[pos], entry.first_step, config)
                }
                (OptimizerKind::Adam(mut config), Slots::Adam { mean, variance }) => {
                    config.lr = learning_rate;
                    adam(
                        values,
                        grad,
                        &mut mean[pos],
                        &mut variance[pos],
                        next_step,
                        config,
                        false,
                    )
                }
                (OptimizerKind::AdamW(mut config), Slots::Adam { mean, variance }) => {
                    config.lr = learning_rate;
                    adam(
                        values,
                        grad,
                        &mut mean[pos],
                        &mut variance[pos],
                        next_step,
                        config,
                        true,
                    )
                }
                (OptimizerKind::Lars(mut config), Slots::Sgd(momentum)) => {
                    config.lr = learning_rate;
                    lars(values, grad, &mut momentum[pos], entry.first_step, config)
                }
                (OptimizerKind::Lamb(mut config), Slots::Adam { mean, variance }) => {
                    config.lr = learning_rate;
                    lamb(
                        values,
                        grad,
                        &mut mean[pos],
                        &mut variance[pos],
                        next_step,
                        config,
                    )
                }
                (OptimizerKind::Muon(mut config), Slots::Sgd(momentum)) => {
                    config.lr = learning_rate;
                    muon(
                        values,
                        grad,
                        &mut momentum[pos],
                        snapshot.shape.dims(),
                        config,
                    )
                }
                _ => return Err(invalid("internal optimizer state mismatch")),
            }?;
            entry.parameter.replace_expected(
                from_f64(snapshot.shape, snapshot.dtype, updated)?,
                Some(snapshot.version),
            )?;
            entry.version = snapshot.version.wrapping_add(1);
            entry.first_step = false;
        }
        self.step = next_step;
        Ok(())
    }
    pub fn state_dict(&self) -> Result<StateDict> {
        let mut state = StateDict::default();
        state.insert(
            "optimizer.config",
            TensorData::from_scalars(
                Shape::new([self.config_fingerprint().len()]),
                DType::U8,
                self.config_fingerprint()
                    .into_iter()
                    .map(|x| Scalar::U(x as u64)),
            )?,
        );
        state.insert(
            "optimizer.learning_rates",
            f64_tensor(
                Shape::new([self.learning_rates.len()]),
                &self.learning_rates,
            ),
        );
        state.insert(
            "optimizer.step",
            TensorData::scalar_with_dtype(Scalar::U(self.step), DType::U64),
        );
        let mut positions = vec![0usize; self.groups.len()];
        for entry in &self.entries {
            let pos = positions[entry.group];
            positions[entry.group] += 1;
            match &self.slots[entry.group] {
                Slots::Sgd(momentum) => state.insert(
                    format!("optimizer.{}.momentum", entry.name),
                    f64_tensor(entry.parameter.snapshot()?.shape, &momentum[pos]),
                ),
                Slots::Adam { mean, variance } => {
                    state.insert(
                        format!("optimizer.{}.exp_avg", entry.name),
                        f64_tensor(entry.parameter.snapshot()?.shape, &mean[pos]),
                    );
                    state.insert(
                        format!("optimizer.{}.exp_avg_sq", entry.name),
                        f64_tensor(entry.parameter.snapshot()?.shape, &variance[pos]),
                    );
                }
            }
        }
        Ok(state)
    }
    pub fn load_state_dict(&mut self, state: &StateDict) -> Result<()> {
        self.validate_state_dict(state)?;
        self.apply_state_dict(state)
    }
    fn validate_state_dict(&self, state: &StateDict) -> Result<()> {
        let expected = self.expected_state_keys();
        let actual = state.tensors().keys().cloned().collect::<BTreeSet<_>>();
        if let Some(key) = expected.difference(&actual).next() {
            return Err(invalid(&format!("optimizer state missing key {key}")));
        }
        if let Some(key) = actual.difference(&expected).next() {
            return Err(invalid(&format!("optimizer state unexpected key {key}")));
        }
        let config = state
            .tensors()
            .get("optimizer.config")
            .ok_or_else(|| invalid("legacy optimizer state lacks config fingerprint"))?;
        if config.dtype() != DType::U8
            || config.shape() != &Shape::new([self.config_fingerprint().len()])
            || to_u8(config) != self.config_fingerprint()
        {
            return Err(invalid("optimizer config fingerprint mismatch"));
        }
        let step = state
            .tensors()
            .get("optimizer.step")
            .expect("expected-key validation");
        if step.dtype() != DType::U64 || step.len() != 1 {
            return Err(invalid("invalid optimizer step"));
        };
        let learning_rates = state
            .tensors()
            .get("optimizer.learning_rates")
            .expect("expected-key validation");
        if learning_rates.dtype() != DType::F64
            || learning_rates.shape() != &Shape::new([self.learning_rates.len()])
            || to_f64(learning_rates)
                .iter()
                .any(|lr| !lr.is_finite() || *lr < 0.)
        {
            return Err(invalid("invalid optimizer learning rates"));
        }
        for entry in &self.entries {
            let suffixes: &[&str] = match self.slots[entry.group] {
                Slots::Sgd(_) => &["momentum"],
                Slots::Adam { .. } => &["exp_avg", "exp_avg_sq"],
            };
            for suffix in suffixes {
                let value = state
                    .tensors()
                    .get(&format!("optimizer.{}.{}", entry.name, suffix))
                    .expect("expected-key validation");
                if value.dtype() != DType::F64
                    || value.shape() != &entry.parameter.snapshot()?.shape
                {
                    return Err(invalid("optimizer state shape mismatch"));
                }
            }
        }
        Ok(())
    }
    fn apply_state_dict(&mut self, state: &StateDict) -> Result<()> {
        let next_step = state
            .tensors()
            .get("optimizer.step")
            .expect("validated optimizer step")
            .scalar_at(0)
            .as_u64();
        let next_learning_rates = to_f64(
            state
                .tensors()
                .get("optimizer.learning_rates")
                .expect("validated optimizer learning rates"),
        );
        let mut next_slots = self.slots.clone();
        let mut next_versions = Vec::new();
        let mut positions = vec![0usize; self.groups.len()];
        for entry in &self.entries {
            let pos = positions[entry.group];
            positions[entry.group] += 1;
            let load = |suffix: &str| -> Result<Vec<f64>> {
                let value = state
                    .tensors()
                    .get(&format!("optimizer.{}.{}", entry.name, suffix))
                    .expect("expected-key validation");
                if value.dtype() != DType::F64
                    || value.shape() != &entry.parameter.snapshot()?.shape
                {
                    return Err(invalid("optimizer state shape mismatch"));
                };
                Ok(to_f64(value))
            };
            match &mut next_slots[entry.group] {
                Slots::Sgd(momentum) => momentum[pos] = load("momentum")?,
                Slots::Adam { mean, variance } => {
                    mean[pos] = load("exp_avg")?;
                    variance[pos] = load("exp_avg_sq")?
                }
            };
            next_versions.push(entry.parameter.snapshot()?.version);
        }
        self.slots = next_slots;
        self.learning_rates = next_learning_rates;
        self.step = next_step;
        for (entry, version) in self.entries.iter_mut().zip(next_versions) {
            entry.version = version;
        }
        Ok(())
    }
    fn expected_state_keys(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::from([
            "optimizer.config".into(),
            "optimizer.step".into(),
            "optimizer.learning_rates".into(),
        ]);
        for entry in &self.entries {
            match self.slots[entry.group] {
                Slots::Sgd(_) => {
                    out.insert(format!("optimizer.{}.momentum", entry.name));
                }
                Slots::Adam { .. } => {
                    out.insert(format!("optimizer.{}.exp_avg", entry.name));
                    out.insert(format!("optimizer.{}.exp_avg_sq", entry.name));
                }
            }
        }
        out
    }
    fn config_fingerprint(&self) -> Vec<u8> {
        let mut out = b"rustgrad-optimizer\0\x02".to_vec();
        out.extend_from_slice(&(self.groups.len() as u64).to_le_bytes());
        for (index, kind) in self.groups.iter().enumerate() {
            out.extend_from_slice(
                &(self.entries.iter().filter(|e| e.group == index).count() as u64).to_le_bytes(),
            );
            match kind {
                OptimizerKind::Sgd(c) => {
                    out.push(0);
                    for x in [c.momentum, c.dampening, c.weight_decay] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                    out.push(c.nesterov as u8)
                }
                OptimizerKind::Adam(c) | OptimizerKind::AdamW(c) => {
                    out.push(if matches!(kind, OptimizerKind::Adam(_)) {
                        1
                    } else {
                        2
                    });
                    for x in [c.beta1, c.beta2, c.eps, c.weight_decay] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                }
                OptimizerKind::Lars(c) => {
                    out.push(3);
                    for x in [c.momentum, c.weight_decay, c.tcoef] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                    out.extend_from_slice(&[c.nesterov as u8, c.classic as u8, c.pre_wd as u8])
                }
                OptimizerKind::Lamb(c) => {
                    out.push(4);
                    for x in [c.beta1, c.beta2, c.eps, c.weight_decay] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                    out.push(c.adam as u8)
                }
                OptimizerKind::Muon(c) => {
                    out.push(5);
                    for x in [c.momentum, c.weight_decay] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                    out.extend_from_slice(&(c.ns_steps as u64).to_le_bytes());
                    out.extend_from_slice(&(c.ns_coefficients.len() as u64).to_le_bytes());
                    for coefficient in &c.ns_coefficients {
                        out.extend_from_slice(&coefficient.to_le_bytes());
                    }
                    out.push(c.nesterov as u8);
                }
            }
        }
        out
    }
}
fn to_u8(data: &TensorData) -> Vec<u8> {
    (0..data.len())
        .map(|i| data.scalar_at(i).as_u64() as u8)
        .collect()
}
fn invalid(reason: &str) -> Error {
    Error::Serialization {
        reason: format!("optimizer: {reason}"),
    }
}
fn validate(kind: &OptimizerKind) -> Result<()> {
    let (lr, wd) = match kind {
        OptimizerKind::Sgd(c) => (c.lr, c.weight_decay),
        OptimizerKind::Adam(c) | OptimizerKind::AdamW(c) => (c.lr, c.weight_decay),
        OptimizerKind::Lars(c) => (c.lr, c.weight_decay),
        OptimizerKind::Lamb(c) => (c.lr, c.weight_decay),
        OptimizerKind::Muon(c) => (c.lr, c.weight_decay),
    };
    if !lr.is_finite() || lr < 0. || !wd.is_finite() || wd < 0. {
        return Err(invalid(
            "learning rate and weight decay must be finite and nonnegative",
        ));
    }
    match kind {
        OptimizerKind::Sgd(c) => {
            if !c.momentum.is_finite()
                || c.momentum < 0.
                || !c.dampening.is_finite()
                || c.dampening < 0.
                || c.nesterov && c.momentum <= 0.
            {
                Err(invalid("invalid SGD momentum, dampening, or nesterov"))
            } else {
                Ok(())
            }
        }
        OptimizerKind::Adam(c) | OptimizerKind::AdamW(c) => {
            if !(0. <= c.beta1
                && c.beta1 < 1.
                && 0. <= c.beta2
                && c.beta2 < 1.
                && c.eps.is_finite())
                || c.eps <= 0.
            {
                Err(invalid("invalid Adam beta or epsilon"))
            } else {
                Ok(())
            }
        }
        OptimizerKind::Lars(c) => {
            if c.momentum < 0. || !c.tcoef.is_finite() || c.tcoef < 0. {
                Err(invalid("invalid LARS momentum or trust coefficient"))
            } else {
                Ok(())
            }
        }
        OptimizerKind::Lamb(c) => {
            if !(0. <= c.beta1 && c.beta1 < 1. && 0. <= c.beta2 && c.beta2 < 1.)
                || c.eps <= 0.
                || !c.eps.is_finite()
            {
                Err(invalid("invalid LAMB beta or epsilon"))
            } else {
                Ok(())
            }
        }
        OptimizerKind::Muon(c) => {
            if !c.momentum.is_finite()
                || c.momentum < 0.
                || c.ns_coefficients.is_empty()
                || c.ns_coefficients.iter().any(|x| !x.is_finite())
            {
                Err(invalid(
                    "invalid Muon momentum or Newton-Schulz coefficients",
                ))
            } else {
                Ok(())
            }
        }
    }
}
fn kind_learning_rate(kind: &OptimizerKind) -> f64 {
    match kind {
        OptimizerKind::Sgd(c) => c.lr,
        OptimizerKind::Adam(c) | OptimizerKind::AdamW(c) => c.lr,
        OptimizerKind::Lars(c) => c.lr,
        OptimizerKind::Lamb(c) => c.lr,
        OptimizerKind::Muon(c) => c.lr,
    }
}
fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn lars(
    mut p: Vec<f64>,
    mut g: Vec<f64>,
    b: &mut [f64],
    _first: bool,
    c: LarsConfig,
) -> Result<Vec<f64>> {
    let r = if c.tcoef != 0. {
        let a = norm(&p);
        let z = norm(&g);
        if a > 0. && z > 0. {
            c.tcoef * a / (z + c.weight_decay * a)
        } else {
            1.
        }
    } else {
        1.
    };
    if c.pre_wd {
        for i in 0..g.len() {
            g[i] += c.weight_decay * p[i];
        }
    }
    if c.classic {
        for v in &mut g {
            *v *= r * c.lr;
        }
    }
    if c.momentum != 0. {
        for i in 0..g.len() {
            b[i] = c.momentum * b[i] + g[i];
            g[i] = if c.nesterov {
                g[i] + c.momentum * b[i]
            } else {
                b[i]
            };
        }
    }
    if !c.classic {
        for v in &mut g {
            *v *= r * c.lr;
        }
    }
    if !c.pre_wd {
        for v in &mut p {
            *v *= 1. - c.weight_decay * c.lr;
        }
    }
    Ok(p.into_iter().zip(g).map(|(a, b)| a - b).collect())
}
fn muon(
    mut p: Vec<f64>,
    mut g: Vec<f64>,
    momentum: &mut [f64],
    shape: &[usize],
    c: MuonConfig,
) -> Result<Vec<f64>> {
    let rows = *shape
        .first()
        .ok_or_else(|| invalid("Muon parameters must have rank at least two"))?;
    let columns = shape[1..].iter().try_fold(1usize, |product, dim| {
        product
            .checked_mul(*dim)
            .ok_or_else(|| invalid("Muon matrix dimensions overflow"))
    })?;
    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(g.len()) {
        return Err(invalid(
            "Muon parameters must have nonzero rectangular size",
        ));
    }
    for index in 0..g.len() {
        momentum[index] = c.momentum * momentum[index] + g[index];
        g[index] = if c.nesterov {
            g[index] + c.momentum * momentum[index]
        } else {
            momentum[index]
        };
    }
    g = newton_schulz(g, rows, columns, c.ns_steps, &c.ns_coefficients)?;
    for value in &mut p {
        *value *= 1. - c.weight_decay * c.lr;
    }
    Ok(p.into_iter()
        .zip(g)
        .map(|(parameter, update)| parameter - c.lr * update)
        .collect())
}

/// Tinygrad's Newton-Schulz odd-polynomial iteration on a row-major matrix.
fn newton_schulz(
    matrix: Vec<f64>,
    rows: usize,
    columns: usize,
    steps: usize,
    coefficients: &[f64],
) -> Result<Vec<f64>> {
    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(matrix.len()) {
        return Err(invalid("Muon Newton-Schulz matrix shape"));
    }
    if coefficients.is_empty() || coefficients.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid Muon Newton-Schulz coefficients"));
    }
    if rows > columns {
        return transpose(
            &newton_schulz(
                transpose(&matrix, rows, columns)?,
                columns,
                rows,
                steps,
                coefficients,
            )?,
            columns,
            rows,
        );
    }
    let scale = norm(&matrix) + 1e-7;
    let mut current = matrix.into_iter().map(|x| x / scale).collect::<Vec<_>>();
    for _ in 0..steps {
        let gram = matmul_transpose_right(&current, rows, columns)?;
        let mut next = vec![0.; current.len()];
        let mut term = current.clone();
        for coefficient in coefficients {
            for (out, value) in next.iter_mut().zip(&term) {
                *out += coefficient * value;
            }
            term = matmul(&gram, rows, rows, &term, columns)?;
        }
        if next.iter().any(|x| !x.is_finite()) {
            return Err(invalid("Muon Newton-Schulz produced non-finite values"));
        }
        current = next;
    }
    Ok(current)
}
fn transpose(matrix: &[f64], rows: usize, columns: usize) -> Result<Vec<f64>> {
    if rows.checked_mul(columns) != Some(matrix.len()) {
        return Err(invalid("Muon transpose matrix shape"));
    }
    let mut out = vec![0.; matrix.len()];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = matrix[row * columns + column];
        }
    }
    Ok(out)
}
fn matmul_transpose_right(matrix: &[f64], rows: usize, columns: usize) -> Result<Vec<f64>> {
    if rows.checked_mul(columns) != Some(matrix.len()) {
        return Err(invalid("Muon Gram matrix shape"));
    }
    let mut out = vec![
        0.;
        rows.checked_mul(rows)
            .ok_or_else(|| invalid("Muon matrix overflow"))?
    ];
    for left in 0..rows {
        for right in 0..rows {
            let mut value = 0.;
            for column in 0..columns {
                value += matrix[left * columns + column] * matrix[right * columns + column];
            }
            out[left * rows + right] = value;
        }
    }
    Ok(out)
}
fn matmul(
    lhs: &[f64],
    lhs_rows: usize,
    lhs_columns: usize,
    rhs: &[f64],
    rhs_columns: usize,
) -> Result<Vec<f64>> {
    if lhs_rows.checked_mul(lhs_columns) != Some(lhs.len())
        || lhs_columns.checked_mul(rhs_columns) != Some(rhs.len())
    {
        return Err(invalid("Muon matrix multiplication shape"));
    }
    let mut out = vec![
        0.;
        lhs_rows
            .checked_mul(rhs_columns)
            .ok_or_else(|| invalid("Muon matrix overflow"))?
    ];
    for row in 0..lhs_rows {
        for column in 0..rhs_columns {
            let mut value = 0.;
            for middle in 0..lhs_columns {
                value += lhs[row * lhs_columns + middle] * rhs[middle * rhs_columns + column];
            }
            out[row * rhs_columns + column] = value;
        }
    }
    Ok(out)
}
fn lamb(
    p: Vec<f64>,
    g: Vec<f64>,
    m: &mut [f64],
    v: &mut [f64],
    step: u64,
    c: LambConfig,
) -> Result<Vec<f64>> {
    let mut up = Vec::new();
    for i in 0..p.len() {
        m[i] = c.beta1 * m[i] + (1. - c.beta1) * g[i];
        v[i] = c.beta2 * v[i] + (1. - c.beta2) * g[i] * g[i];
        up.push(
            m[i] / (1. - c.beta1.powi(step as i32))
                / ((v[i] / (1. - c.beta2.powi(step as i32))).sqrt() + c.eps)
                + c.weight_decay * p[i],
        );
    }
    let r = if c.adam || norm(&p) == 0. || norm(&up) == 0. {
        1.
    } else {
        norm(&p) / norm(&up)
    };
    Ok(p.into_iter()
        .zip(up)
        .map(|(a, b)| a - c.lr * r * b)
        .collect())
}
fn validate_gradient(snapshot: &crate::ParameterSnapshot, gradient: &Gradient) -> Result<()> {
    if gradient.identity != snapshot.identity {
        return Err(invalid("gradient parameter identity mismatch"));
    }
    if gradient.data.shape() != &snapshot.shape {
        return Err(invalid("gradient shape mismatch"));
    }
    if !gradient.data.dtype().is_float() {
        return Err(invalid("gradient dtype must be float"));
    }
    Ok(())
}
fn to_f64(data: &TensorData) -> Vec<f64> {
    (0..data.len())
        .map(|i| data.scalar_at(i).as_f64())
        .collect()
}
fn from_f64(shape: Shape, dtype: DType, values: Vec<f64>) -> Result<TensorData> {
    TensorData::from_scalars(shape, dtype, values.into_iter().map(Scalar::F))
}
fn f64_tensor(shape: Shape, values: &[f64]) -> TensorData {
    TensorData::from_scalars(shape, DType::F64, values.iter().copied().map(Scalar::F))
        .expect("slot shape")
}
fn sgd(
    mut p: Vec<f64>,
    mut g: Vec<f64>,
    buffer: &mut [f64],
    first: bool,
    c: SgdConfig,
) -> Result<Vec<f64>> {
    for i in 0..p.len() {
        g[i] += c.weight_decay * p[i];
        if c.momentum != 0. {
            buffer[i] = c.momentum * buffer[i]
                + if first {
                    g[i]
                } else {
                    (1. - c.dampening) * g[i]
                };
            g[i] = if c.nesterov {
                g[i] + c.momentum * buffer[i]
            } else {
                buffer[i]
            };
        }
        p[i] -= c.lr * g[i];
    }
    Ok(p)
}
fn adam(
    mut p: Vec<f64>,
    mut g: Vec<f64>,
    m: &mut [f64],
    v: &mut [f64],
    step: u64,
    c: AdamConfig,
    decoupled: bool,
) -> Result<Vec<f64>> {
    for i in 0..p.len() {
        if !decoupled {
            g[i] += c.weight_decay * p[i]
        }
        m[i] = c.beta1 * m[i] + (1. - c.beta1) * g[i];
        v[i] = c.beta2 * v[i] + (1. - c.beta2) * g[i] * g[i];
        let update = (m[i] / (1. - c.beta1.powi(step as i32)))
            / (v[i] / (1. - c.beta2.powi(step as i32)))
                .sqrt()
                .mul_add(1., c.eps);
        if decoupled {
            p[i] *= 1. - c.lr * c.weight_decay
        }
        p[i] -= c.lr * update;
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Graph, Module, Storage};

    fn parameter(_graph: &mut Graph, values: Vec<f32>) -> Parameter {
        Parameter::new(TensorData::new([values.len()], values).unwrap(), true)
    }
    fn values(parameter: &Parameter) -> Vec<f32> {
        match parameter.value().unwrap().storage() {
            Storage::F32(v) => v.clone(),
            _ => unreachable!(),
        }
    }
    fn gradient(parameter: &Parameter, values: Vec<f32>) -> Gradient {
        Gradient::for_parameter(parameter, TensorData::new([values.len()], values).unwrap())
            .unwrap()
    }

    #[test]
    fn sgd_variants_have_known_updates_and_tied_parameters_are_once_only() {
        let mut graph = Graph::new();
        let parameter = parameter(&mut graph, vec![1.]);
        let mut optimizer = Optimizer::sgd(
            vec![
                ("a".into(), parameter.clone()),
                ("b".into(), parameter.clone()),
            ],
            SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.,
                nesterov: false,
                weight_decay: 0.,
            },
        )
        .unwrap();
        assert_eq!(optimizer.parameter_names(), vec!["a"]);
        let mut gradients = BTreeMap::new();
        gradients.insert("a".into(), gradient(&parameter, vec![2.]));
        optimizer.step(&gradients).unwrap();
        assert!((values(&parameter)[0] - 0.8).abs() < 1e-6);
        gradients.insert("a".into(), gradient(&parameter, vec![2.]));
        optimizer.step(&gradients).unwrap();
        assert!((values(&parameter)[0] - 0.42).abs() < 1e-6);
        let mut nesterov = Optimizer::sgd(
            vec![("a".into(), parameter.clone())],
            SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.,
                nesterov: true,
                weight_decay: 0.,
            },
        )
        .unwrap();
        gradients.insert("a".into(), gradient(&parameter, vec![1.]));
        nesterov.step(&gradients).unwrap();
        assert!((values(&parameter)[0] - 0.23).abs() < 1e-6);
    }
    #[test]
    fn adam_and_adamw_match_one_step_oracle_and_reject_stale_gradients() {
        let mut graph = Graph::new();
        let adam_parameter = parameter(&mut graph, vec![1.]);
        let config = AdamConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
        };
        let mut adam = Optimizer::adam(vec![("p".into(), adam_parameter.clone())], config).unwrap();
        let mut gradients = BTreeMap::new();
        gradients.insert("p".into(), gradient(&adam_parameter, vec![1.]));
        adam.step(&gradients).unwrap();
        assert!((values(&adam_parameter)[0] - 0.9).abs() < 1e-5);
        let adamw_parameter = parameter(&mut graph, vec![1.]);
        let mut adamw =
            Optimizer::adamw(vec![("p".into(), adamw_parameter.clone())], config).unwrap();
        let mut gradients = BTreeMap::new();
        gradients.insert("p".into(), gradient(&adamw_parameter, vec![1.]));
        adamw.step(&gradients).unwrap();
        assert!((values(&adamw_parameter)[0] - 0.89).abs() < 1e-5);
        let stale = gradient(&adamw_parameter, vec![1.]);
        adamw_parameter
            .replace(TensorData::new([1], vec![2.]).unwrap())
            .unwrap();
        gradients.insert("p".into(), stale);
        assert!(adamw.step(&gradients).is_err());
    }

    #[test]
    fn gradient_stamps_reject_another_parameter_with_matching_shape_and_version() {
        let mut graph = Graph::new();
        let target = parameter(&mut graph, vec![1.]);
        let other = parameter(&mut graph, vec![1.]);
        let mut optimizer =
            Optimizer::sgd(vec![("p".into(), target)], SgdConfig::default()).unwrap();
        let gradients = BTreeMap::from([("p".into(), gradient(&other, vec![1.]))]);
        assert!(optimizer.step(&gradients).is_err());
    }
    #[test]
    fn checkpoint_resume_matches_uninterrupted_adam() {
        let mut first_graph = Graph::new();
        let first = parameter(&mut first_graph, vec![1., -1.]);
        let config = AdamConfig::default();
        let mut uninterrupted =
            Optimizer::adamw(vec![("weight".into(), first.clone())], config).unwrap();
        for _ in 0..2 {
            let mut gradients = BTreeMap::new();
            gradients.insert("weight".into(), gradient(&first, vec![0.5, -0.25]));
            uninterrupted.step(&gradients).unwrap();
        }
        let mut second_graph = Graph::new();
        let second = parameter(&mut second_graph, vec![1., -1.]);
        let mut saved = Optimizer::adamw(vec![("weight".into(), second.clone())], config).unwrap();
        let mut gradients = BTreeMap::new();
        gradients.insert("weight".into(), gradient(&second, vec![0.5, -0.25]));
        saved.step(&gradients).unwrap();
        let checkpoint = saved.state_dict().unwrap();
        let value = second.value().unwrap();
        let resumed = Parameter::new(value, true);
        let mut resumed_optimizer =
            Optimizer::adamw(vec![("weight".into(), resumed.clone())], config).unwrap();
        resumed_optimizer.load_state_dict(&checkpoint).unwrap();
        let mut gradients = BTreeMap::new();
        gradients.insert("weight".into(), gradient(&resumed, vec![0.5, -0.25]));
        resumed_optimizer.step(&gradients).unwrap();
        assert_eq!(values(&first), values(&resumed));
    }
    #[test]
    fn explicit_graph_gradients_drive_a_linear_training_step() {
        let mut graph = Graph::new();
        let linear = crate::nn::Linear::new(&mut graph, 1, 1, false, 1).unwrap();
        linear
            .weight
            .replace(TensorData::new([1, 1], vec![0.]).unwrap())
            .unwrap();
        let mut optimizer = Optimizer::sgd(
            vec![("weight".into(), linear.weight.clone())],
            SgdConfig {
                lr: 0.1,
                ..SgdConfig::default()
            },
        )
        .unwrap();
        let x = graph.input("x", [1, 1]);
        let prediction = linear.forward(&mut graph, x).unwrap();
        let target = graph.constant(TensorData::new([1, 1], vec![2.]).unwrap());
        let error = graph.sub(prediction, target).unwrap();
        let squared = graph.square(error).unwrap();
        let loss = graph
            .reduce(squared, crate::ReduceKind::Mean, None, false)
            .unwrap();
        let grad = graph
            .grad(loss, linear.weight.node(&graph).unwrap())
            .unwrap();
        let mut bindings = linear.input_bindings(&graph).unwrap();
        bindings.insert("x".into(), TensorData::new([1, 1], vec![1.]).unwrap());
        let cpu = CpuBackend;
        let before = cpu
            .execute(&graph, loss, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64();
        let gradient = cpu.execute(&graph, grad, &bindings).unwrap();
        let mut gradients = BTreeMap::new();
        gradients.insert(
            "weight".into(),
            Gradient::for_parameter(&linear.weight, gradient).unwrap(),
        );
        optimizer.step(&gradients).unwrap();
        let mut next_graph = Graph::new();
        let x = next_graph.input("x", [1, 1]);
        let prediction = linear.forward(&mut next_graph, x).unwrap();
        let target = next_graph.constant(TensorData::new([1, 1], vec![2.]).unwrap());
        let error = next_graph.sub(prediction, target).unwrap();
        let squared = next_graph.square(error).unwrap();
        let next_loss = next_graph
            .reduce(squared, crate::ReduceKind::Mean, None, false)
            .unwrap();
        let mut bindings = linear.input_bindings(&next_graph).unwrap();
        bindings.insert("x".into(), TensorData::new([1, 1], vec![1.]).unwrap());
        let after = cpu
            .execute(&next_graph, next_loss, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64();
        assert!(after < before);
    }

    #[test]
    fn optimizer_paths_propagate_parameter_lock_poisoning() {
        let mut graph = Graph::new();
        let parameter = parameter(&mut graph, vec![1.]);
        let mut optimizer =
            Optimizer::sgd(vec![("p".into(), parameter.clone())], SgdConfig::default()).unwrap();
        let mut gradients = BTreeMap::new();
        gradients.insert("p".into(), gradient(&parameter, vec![1.]));
        parameter.poison_for_test();
        assert!(matches!(
            Gradient::for_parameter(&parameter, TensorData::new([1], vec![1.]).unwrap()),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert!(matches!(
            optimizer.state_dict(),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert!(matches!(
            optimizer.step(&gradients),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert!(matches!(
            Optimizer::sgd(vec![("p".into(), parameter)], SgdConfig::default()),
            Err(Error::ParameterLockPoisoned { .. })
        ));
    }

    #[test]
    fn checkpoint_load_rejects_mutations_atomically() {
        let config = AdamConfig {
            lr: 0.02,
            weight_decay: 0.1,
            ..AdamConfig::default()
        };
        let mut source_graph = Graph::new();
        let source = parameter(&mut source_graph, vec![1., -2.]);
        let mut source_opt = Optimizer::adamw(vec![("p".into(), source.clone())], config).unwrap();
        source_opt
            .step(&BTreeMap::from([(
                "p".into(),
                gradient(&source, vec![0.3, -0.2]),
            )]))
            .unwrap();
        let good = source_opt.state_dict().unwrap();
        let mut target_graph = Graph::new();
        let target = parameter(&mut target_graph, vec![1., -2.]);
        let mut target_opt = Optimizer::adamw(vec![("p".into(), target)], config).unwrap();
        target_opt.load_state_dict(&good).unwrap();
        let before = target_opt.state_dict().unwrap();
        enum Change {
            Remove(&'static str),
            Add,
            BadConfig,
            BadSlot,
        }
        for change in [
            Change::Remove("optimizer.config"),
            Change::Remove("optimizer.step"),
            Change::Remove("optimizer.p.exp_avg_sq"),
            Change::Add,
            Change::BadConfig,
            Change::BadSlot,
        ] {
            let mut raw = good.clone().into_tensors();
            match change {
                Change::Remove(k) => {
                    raw.remove(k);
                }
                Change::Add => {
                    raw.insert("extra".into(), TensorData::scalar(1.));
                }
                Change::BadConfig => {
                    raw.insert(
                        "optimizer.config".into(),
                        TensorData::new([1], vec![1.]).unwrap(),
                    );
                }
                Change::BadSlot => {
                    raw.insert(
                        "optimizer.p.exp_avg_sq".into(),
                        TensorData::new([1], vec![1.]).unwrap(),
                    );
                }
            }
            assert!(target_opt.load_state_dict(&StateDict::from(raw)).is_err());
            assert_eq!(target_opt.state_dict().unwrap(), before);
        }
    }

    #[test]
    fn lars_matches_independent_one_step_variant_table() {
        fn reference(p: &[f64], g: &[f64], b: &[f64], c: LarsConfig) -> (Vec<f64>, Vec<f64>) {
            let n = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let r = if c.tcoef != 0. && n(p) > 0. && n(g) > 0. {
                c.tcoef * n(p) / (n(g) + c.weight_decay * n(p))
            } else {
                1.
            };
            let mut u = g.to_vec();
            if c.pre_wd {
                for i in 0..u.len() {
                    u[i] += c.weight_decay * p[i];
                }
            }
            if c.classic {
                for x in &mut u {
                    *x *= r * c.lr;
                }
            }
            let mut nb = b.to_vec();
            if c.momentum != 0. {
                for i in 0..u.len() {
                    nb[i] = c.momentum * nb[i] + u[i];
                    u[i] = if c.nesterov {
                        u[i] + c.momentum * nb[i]
                    } else {
                        nb[i]
                    };
                }
            }
            if !c.classic {
                for x in &mut u {
                    *x *= r * c.lr;
                }
            }
            (p.iter().zip(&u).map(|(a, x)| a - x).collect(), nb)
        }
        let cases = [
            ("default", LarsConfig::default()),
            (
                "popular",
                LarsConfig {
                    classic: false,
                    ..LarsConfig::default()
                },
            ),
            (
                "post",
                LarsConfig {
                    pre_wd: false,
                    ..LarsConfig::default()
                },
            ),
            (
                "nesterov",
                LarsConfig {
                    nesterov: true,
                    ..LarsConfig::default()
                },
            ),
            (
                "zero",
                LarsConfig {
                    momentum: 0.,
                    ..LarsConfig::default()
                },
            ),
        ];
        for (name, c) in cases {
            let (expected, _) = reference(&[1., -2.], &[0.3, -0.2], &[0., 0.], c);
            let mut g = Graph::new();
            let p = parameter(&mut g, vec![1., -2.]);
            let mut o = Optimizer::lars(vec![("p".into(), p.clone())], c).unwrap();
            o.step(&BTreeMap::from([(
                "p".into(),
                gradient(&p, vec![0.3, -0.2]),
            )]))
            .unwrap();
            for (a, b) in values(&p).iter().zip(expected) {
                assert!((*a as f64 - b).abs() < 1e-6, "{name}");
            }
        }
    }

    #[test]
    fn lars_two_step_checkpoint_resume_and_config_fingerprint() {
        let c = LarsConfig {
            nesterov: true,
            momentum: 0.8,
            ..LarsConfig::default()
        };
        let grads = [vec![0.2, -0.1], vec![-0.3, 0.4]];
        let mut a_graph = Graph::new();
        let a = parameter(&mut a_graph, vec![1., -2.]);
        let mut uninterrupted = Optimizer::lars(vec![("p".into(), a.clone())], c).unwrap();
        for g in &grads {
            uninterrupted
                .step(&BTreeMap::from([(
                    "p".into(),
                    gradient(&a, g.iter().map(|x| *x as f32).collect()),
                )]))
                .unwrap();
        }
        let mut b_graph = Graph::new();
        let b = parameter(&mut b_graph, vec![1., -2.]);
        let mut saved = Optimizer::lars(vec![("p".into(), b.clone())], c).unwrap();
        saved
            .step(&BTreeMap::from([(
                "p".into(),
                gradient(&b, grads[0].iter().map(|x| *x as f32).collect()),
            )]))
            .unwrap();
        let checkpoint = saved.state_dict().unwrap();
        let mut r_graph = Graph::new();
        let r = parameter(&mut r_graph, values(&b));
        let mut resumed = Optimizer::lars(vec![("p".into(), r.clone())], c).unwrap();
        resumed.load_state_dict(&checkpoint).unwrap();
        resumed
            .step(&BTreeMap::from([(
                "p".into(),
                gradient(&r, grads[1].iter().map(|x| *x as f32).collect()),
            )]))
            .unwrap();
        assert_eq!(values(&a), values(&r));
        assert_eq!(
            uninterrupted.state_dict().unwrap(),
            resumed.state_dict().unwrap()
        );
        for bad in [
            LarsConfig { momentum: 0.7, ..c },
            LarsConfig {
                weight_decay: 0.2,
                ..c
            },
            LarsConfig {
                nesterov: false,
                ..c
            },
            LarsConfig {
                classic: false,
                ..c
            },
            LarsConfig { pre_wd: false, ..c },
            LarsConfig { tcoef: 0.2, ..c },
        ] {
            let mut g = Graph::new();
            let p = parameter(&mut g, values(&b));
            let mut target = Optimizer::lars(vec![("p".into(), p)], bad).unwrap();
            let before = target.state_dict().unwrap();
            assert!(target.load_state_dict(&checkpoint).is_err());
            assert_eq!(target.state_dict().unwrap(), before);
        }
    }

    #[test]
    fn lamb_default_one_step_matches_independent_reference() {
        let c = LambConfig {
            lr: 0.02,
            beta1: 0.8,
            beta2: 0.9,
            eps: 1e-6,
            weight_decay: 0.1,
            adam: false,
        };
        let p = vec![1.5f64, -0.5];
        let grad = [0.3f64, -0.2];
        let m: Vec<f64> = grad.iter().map(|x| (1. - c.beta1) * x).collect();
        let v: Vec<f64> = grad.iter().map(|x| (1. - c.beta2) * x * x).collect();
        let update: Vec<f64> = p
            .iter()
            .enumerate()
            .map(|(i, x)| {
                m[i] / (1. - c.beta1) / ((v[i] / (1. - c.beta2)).sqrt() + c.eps)
                    + c.weight_decay * x
            })
            .collect();
        let norm = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let trust = norm(&p) / norm(&update);
        assert!((trust - 1.).abs() > 1e-3);
        let expected: Vec<f64> = p
            .iter()
            .zip(&update)
            .map(|(x, u)| x - c.lr * trust * u)
            .collect();
        let mut g = Graph::new();
        let p_handle = parameter(&mut g, p.iter().map(|x| *x as f32).collect());
        let mut opt = Optimizer::lamb(vec![("p".into(), p_handle.clone())], c).unwrap();
        opt.step(&BTreeMap::from([(
            "p".into(),
            gradient(&p_handle, grad.iter().map(|x| *x as f32).collect()),
        )]))
        .unwrap();
        for (a, b) in values(&p_handle).iter().zip(expected) {
            assert!((*a as f64 - b).abs() < 2e-5, "actual={a} expected={b}");
        }
        let state = opt.state_dict().unwrap();
        assert_eq!(
            state
                .tensors()
                .get("optimizer.step")
                .unwrap()
                .scalar_at(0)
                .as_u64(),
            1
        );
        for (key, want) in [("optimizer.p.exp_avg", &m), ("optimizer.p.exp_avg_sq", &v)] {
            for (i, x) in want.iter().enumerate() {
                assert!((state.tensors().get(key).unwrap().scalar_at(i).as_f64() - x).abs() < 1e-8);
            }
        }
        let mut zero_graph = Graph::new();
        let zero = parameter(&mut zero_graph, vec![0.]);
        let mut zero_opt = Optimizer::lamb(vec![("z".into(), zero.clone())], c).unwrap();
        zero_opt
            .step(&BTreeMap::from([("z".into(), gradient(&zero, vec![0.]))]))
            .unwrap();
        assert_eq!(values(&zero), vec![0.]);
    }

    #[test]
    fn lamb_one_step_variants_match_independent_reference() {
        let base = LambConfig {
            lr: 0.02,
            beta1: 0.8,
            beta2: 0.9,
            eps: 1e-6,
            weight_decay: 0.1,
            adam: false,
        };
        struct Case {
            name: &'static str,
            p: Vec<f64>,
            g: Vec<f64>,
            c: LambConfig,
        }
        let cases = vec![
            Case {
                name: "adam trust bypass",
                p: vec![1.5, -0.5],
                g: vec![0.3, -0.2],
                c: LambConfig { adam: true, ..base },
            },
            Case {
                name: "no decay",
                p: vec![1.5, -0.5],
                g: vec![0.3, -0.2],
                c: LambConfig {
                    weight_decay: 0.,
                    ..base
                },
            },
            Case {
                name: "decay altered beta epsilon",
                p: vec![1.5, -0.5],
                g: vec![0.3, -0.2],
                c: LambConfig {
                    beta1: 0.6,
                    beta2: 0.7,
                    eps: 1e-4,
                    ..base
                },
            },
            Case {
                name: "zero parameter norm",
                p: vec![0., 0.],
                g: vec![0.3, -0.2],
                c: base,
            },
            Case {
                name: "zero update guard",
                p: vec![1., -2.],
                g: vec![0., 0.],
                c: LambConfig {
                    weight_decay: 0.,
                    ..base
                },
            },
        ];
        let reference = |p: &[f64], g: &[f64], c: LambConfig| {
            let m = g.iter().map(|x| (1. - c.beta1) * x).collect::<Vec<_>>();
            let v = g.iter().map(|x| (1. - c.beta2) * x * x).collect::<Vec<_>>();
            let update = p
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    m[i] / (1. - c.beta1) / ((v[i] / (1. - c.beta2)).sqrt() + c.eps)
                        + c.weight_decay * x
                })
                .collect::<Vec<_>>();
            let norm = |x: &[f64]| x.iter().map(|x| x * x).sum::<f64>().sqrt();
            let trust = if c.adam || norm(p) == 0. || norm(&update) == 0. {
                1.
            } else {
                norm(p) / norm(&update)
            };
            (
                p.iter()
                    .zip(&update)
                    .map(|(p, u)| p - c.lr * trust * u)
                    .collect::<Vec<_>>(),
                m,
                v,
                trust,
            )
        };
        for case in cases {
            let (expected, m, v, _) = reference(&case.p, &case.g, case.c);
            let mut graph = Graph::new();
            let parameter = parameter(&mut graph, case.p.iter().map(|x| *x as f32).collect());
            let mut optimizer =
                Optimizer::lamb(vec![("p".into(), parameter.clone())], case.c).unwrap();
            optimizer
                .step(&BTreeMap::from([(
                    "p".into(),
                    gradient(&parameter, case.g.iter().map(|x| *x as f32).collect()),
                )]))
                .unwrap();
            for (actual, expected) in values(&parameter).iter().zip(expected) {
                assert!(
                    (*actual as f64 - expected).abs() < 2e-5,
                    "{} parameter",
                    case.name
                );
            }
            let state = optimizer.state_dict().unwrap();
            assert_eq!(
                state
                    .tensors()
                    .get("optimizer.step")
                    .unwrap()
                    .scalar_at(0)
                    .as_u64(),
                1,
                "{} step",
                case.name
            );
            for (key, expected) in [("optimizer.p.exp_avg", &m), ("optimizer.p.exp_avg_sq", &v)] {
                for (i, expected) in expected.iter().enumerate() {
                    assert!(
                        (state.tensors().get(key).unwrap().scalar_at(i).as_f64() - expected).abs()
                            < 1e-8,
                        "{} {key}",
                        case.name
                    );
                }
            }
        }
        let (_, _, _, trusted) = reference(&[1.5, -0.5], &[0.3, -0.2], base);
        let (_, _, _, adam) = reference(
            &[1.5, -0.5],
            &[0.3, -0.2],
            LambConfig { adam: true, ..base },
        );
        assert!((trusted - 1.).abs() > 1e-3 && adam == 1.);
    }

    #[test]
    fn lamb_two_step_checkpoint_resume_and_config_fingerprint() {
        let c = LambConfig {
            lr: 0.03,
            beta1: 0.7,
            beta2: 0.85,
            eps: 1e-5,
            weight_decay: 0.12,
            adam: false,
        };
        let grads = [vec![0.2, -0.4], vec![-0.3, 0.1]];
        let reference = |mut p: Vec<f64>| {
            let mut m = vec![0.; p.len()];
            let mut v = vec![0.; p.len()];
            for (step, g) in grads.iter().enumerate() {
                for i in 0..p.len() {
                    m[i] = c.beta1 * m[i] + (1. - c.beta1) * g[i];
                    v[i] = c.beta2 * v[i] + (1. - c.beta2) * g[i] * g[i];
                }
                let update = (0..p.len())
                    .map(|i| {
                        m[i] / (1. - c.beta1.powi((step + 1) as i32))
                            / ((v[i] / (1. - c.beta2.powi((step + 1) as i32))).sqrt() + c.eps)
                            + c.weight_decay * p[i]
                    })
                    .collect::<Vec<_>>();
                let norm = |x: &[f64]| x.iter().map(|x| x * x).sum::<f64>().sqrt();
                let trust = if norm(&p) == 0. || norm(&update) == 0. {
                    1.
                } else {
                    norm(&p) / norm(&update)
                };
                for (x, u) in p.iter_mut().zip(update) {
                    *x -= c.lr * trust * u;
                }
            }
            (p, m, v)
        };
        let (expected, expected_m, expected_v) = reference(vec![1.2, -0.8]);
        let mut a_graph = Graph::new();
        let a = parameter(&mut a_graph, vec![1.2, -0.8]);
        let mut uninterrupted = Optimizer::lamb(vec![("p".into(), a.clone())], c).unwrap();
        for g in &grads {
            uninterrupted
                .step(&BTreeMap::from([(
                    "p".into(),
                    gradient(&a, g.iter().map(|x| *x as f32).collect()),
                )]))
                .unwrap();
        }
        let mut b_graph = Graph::new();
        let b = parameter(&mut b_graph, vec![1.2, -0.8]);
        let mut saved = Optimizer::lamb(vec![("p".into(), b.clone())], c).unwrap();
        saved
            .step(&BTreeMap::from([(
                "p".into(),
                gradient(&b, grads[0].iter().map(|x| *x as f32).collect()),
            )]))
            .unwrap();
        let checkpoint = saved.state_dict().unwrap();
        let mut r_graph = Graph::new();
        let r = parameter(&mut r_graph, values(&b));
        let mut resumed = Optimizer::lamb(vec![("p".into(), r.clone())], c).unwrap();
        resumed.load_state_dict(&checkpoint).unwrap();
        resumed
            .step(&BTreeMap::from([(
                "p".into(),
                gradient(&r, grads[1].iter().map(|x| *x as f32).collect()),
            )]))
            .unwrap();
        for (actual, want) in values(&a).iter().zip(&expected) {
            assert!((*actual as f64 - *want).abs() < 3e-5);
        }
        assert_eq!(values(&a), values(&r));
        assert_eq!(
            uninterrupted.state_dict().unwrap(),
            resumed.state_dict().unwrap()
        );
        let state = resumed.state_dict().unwrap();
        assert_eq!(
            state
                .tensors()
                .get("optimizer.step")
                .unwrap()
                .scalar_at(0)
                .as_u64(),
            2
        );
        for (key, want) in [
            ("optimizer.p.exp_avg", expected_m),
            ("optimizer.p.exp_avg_sq", expected_v),
        ] {
            for (i, want) in want.iter().enumerate() {
                assert!(
                    (state.tensors().get(key).unwrap().scalar_at(i).as_f64() - want).abs() < 1e-8
                );
            }
        }
        for bad in [
            LambConfig { beta1: 0.6, ..c },
            LambConfig { beta2: 0.8, ..c },
            LambConfig { eps: 1e-4, ..c },
            LambConfig {
                weight_decay: 0.,
                ..c
            },
            LambConfig { adam: true, ..c },
        ] {
            let mut g = Graph::new();
            let p = parameter(&mut g, values(&b));
            let mut target = Optimizer::lamb(vec![("p".into(), p)], bad).unwrap();
            let before = target.state_dict().unwrap();
            assert!(target.load_state_dict(&checkpoint).is_err());
            assert_eq!(target.state_dict().unwrap(), before);
        }
    }

    fn matrix_parameter(
        _graph: &mut Graph,
        shape: &[usize],
        dtype: DType,
        values: &[f64],
    ) -> Parameter {
        Parameter::new(
            TensorData::from_scalars(
                Shape::new(shape.to_vec()),
                dtype,
                values.iter().copied().map(Scalar::F),
            )
            .unwrap(),
            true,
        )
    }
    fn matrix_gradient(parameter: &Parameter, shape: &[usize], values: &[f64]) -> Gradient {
        Gradient::for_parameter(
            parameter,
            TensorData::from_scalars(
                Shape::new(shape.to_vec()),
                DType::F64,
                values.iter().copied().map(Scalar::F),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn independent_ns(
        matrix: &[f64],
        rows: usize,
        cols: usize,
        steps: usize,
        c: &[f64],
    ) -> Vec<f64> {
        fn t(x: &[f64], r: usize, q: usize) -> Vec<f64> {
            let mut out = vec![0.; x.len()];
            for i in 0..r {
                for j in 0..q {
                    out[j * r + i] = x[i * q + j];
                }
            }
            out
        }
        fn mm(a: &[f64], ar: usize, ac: usize, b: &[f64], bc: usize) -> Vec<f64> {
            let mut out = vec![0.; ar * bc];
            for i in 0..ar {
                for j in 0..bc {
                    for k in 0..ac {
                        out[i * bc + j] += a[i * ac + k] * b[k * bc + j];
                    }
                }
            }
            out
        }
        if rows > cols {
            return t(
                &independent_ns(&t(matrix, rows, cols), cols, rows, steps, c),
                cols,
                rows,
            );
        }
        let scale = matrix.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-7;
        let mut current = matrix.iter().map(|x| x / scale).collect::<Vec<_>>();
        for _ in 0..steps {
            let gram = mm(&current, rows, cols, &t(&current, rows, cols), rows);
            let mut next = vec![0.; current.len()];
            let mut term = current.clone();
            for coefficient in c {
                for (out, value) in next.iter_mut().zip(&term) {
                    *out += coefficient * value;
                }
                term = mm(&gram, rows, rows, &term, cols);
            }
            current = next;
        }
        current
    }
    fn independent_muon(
        p: &[f64],
        g: &[f64],
        momentum: &[f64],
        shape: &[usize],
        c: &MuonConfig,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut slot = momentum.to_vec();
        let mut update = g.to_vec();
        for i in 0..update.len() {
            slot[i] = c.momentum * slot[i] + update[i];
            update[i] = if c.nesterov {
                update[i] + c.momentum * slot[i]
            } else {
                slot[i]
            };
        }
        let columns = shape[1..].iter().product();
        update = independent_ns(&update, shape[0], columns, c.ns_steps, &c.ns_coefficients);
        (
            p.iter()
                .zip(update)
                .map(|(x, u)| x * (1. - c.weight_decay * c.lr) - c.lr * u)
                .collect(),
            slot,
        )
    }

    #[test]
    fn muon_matches_independent_square_tall_wide_and_custom_oracles() {
        let cases = [
            (
                "square",
                vec![2, 2],
                vec![1., -2., 0.5, 3.],
                vec![0.2, -0.4, 0.7, 0.1],
                MuonConfig {
                    lr: 0.03,
                    momentum: 0.8,
                    weight_decay: 0.02,
                    ns_steps: 2,
                    ns_coefficients: vec![2., -1.5, 0.5],
                    nesterov: true,
                },
            ),
            (
                "tall",
                vec![3, 2],
                vec![1., 2., -1., 0.5, 2., -2.],
                vec![0.3, -0.2, 0.4, 0.1, -0.5, 0.6],
                MuonConfig {
                    ns_steps: 1,
                    ns_coefficients: vec![2., -1.5, 0.5],
                    ..MuonConfig::default()
                },
            ),
            (
                "wide",
                vec![2, 3],
                vec![1., 2., -1., 0.5, 2., -2.],
                vec![0.3, -0.2, 0.4, 0.1, -0.5, 0.6],
                MuonConfig {
                    ns_steps: 1,
                    ns_coefficients: vec![2., -1.5, 0.5],
                    ..MuonConfig::default()
                },
            ),
        ];
        for (name, shape, initial, grad, config) in cases {
            let (expected, slot) =
                independent_muon(&initial, &grad, &vec![0.; initial.len()], &shape, &config);
            let mut graph = Graph::new();
            let parameter = matrix_parameter(&mut graph, &shape, DType::F32, &initial);
            let mut optimizer =
                Optimizer::muon(vec![("p".into(), parameter.clone())], config).unwrap();
            optimizer
                .step(&BTreeMap::from([(
                    "p".into(),
                    matrix_gradient(&parameter, &shape, &grad),
                )]))
                .unwrap();
            let actual = to_f64(&parameter.snapshot().unwrap().data);
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "{name}: {actual} != {expected}"
                );
                assert!(actual.is_finite(), "{name}");
            }
            let state = optimizer.state_dict().unwrap();
            for (i, expected) in slot.iter().enumerate() {
                assert!(
                    (state.tensors()["optimizer.p.momentum"]
                        .scalar_at(i)
                        .as_f64()
                        - expected)
                        .abs()
                        < 1e-10
                );
            }
        }
    }

    #[test]
    fn muon_resume_requantization_and_validation_are_strict() {
        let c = MuonConfig {
            lr: 0.02,
            momentum: 0.8,
            weight_decay: 0.03,
            ns_steps: 2,
            ns_coefficients: vec![2., -1.5, 0.5],
            nesterov: false,
        };
        let shape = [2, 2];
        let first = [0.3, -0.2, 0.4, 0.1];
        let second = [-0.1, 0.5, 0.2, -0.4];
        let mut a_graph = Graph::new();
        let a = matrix_parameter(&mut a_graph, &shape, DType::F32, &[1., -2., 0.5, 3.]);
        let mut uninterrupted = Optimizer::muon(vec![("p".into(), a.clone())], c.clone()).unwrap();
        for gradient_data in [&first[..], &second[..]] {
            uninterrupted
                .step(&BTreeMap::from([(
                    "p".into(),
                    matrix_gradient(&a, &shape, gradient_data),
                )]))
                .unwrap();
        }
        let mut b_graph = Graph::new();
        let b = matrix_parameter(&mut b_graph, &shape, DType::F32, &[1., -2., 0.5, 3.]);
        let mut saved = Optimizer::muon(vec![("p".into(), b.clone())], c.clone()).unwrap();
        saved
            .step(&BTreeMap::from([(
                "p".into(),
                matrix_gradient(&b, &shape, &first),
            )]))
            .unwrap();
        let checkpoint = saved.state_dict().unwrap();
        let resumed_parameter = Parameter::new(b.snapshot().unwrap().data, true);
        let mut resumed =
            Optimizer::muon(vec![("p".into(), resumed_parameter.clone())], c.clone()).unwrap();
        resumed.load_state_dict(&checkpoint).unwrap();
        resumed
            .step(&BTreeMap::from([(
                "p".into(),
                matrix_gradient(&resumed_parameter, &shape, &second),
            )]))
            .unwrap();
        assert_eq!(
            a.snapshot().unwrap().data,
            resumed_parameter.snapshot().unwrap().data
        );
        assert_eq!(
            uninterrupted.state_dict().unwrap(),
            resumed.state_dict().unwrap()
        );
        let mut wrong = Optimizer::muon(
            vec![("p".into(), resumed_parameter.clone())],
            MuonConfig {
                ns_steps: 1,
                ..c.clone()
            },
        )
        .unwrap();
        let before = wrong.state_dict().unwrap();
        assert!(wrong.load_state_dict(&checkpoint).is_err());
        assert_eq!(wrong.state_dict().unwrap(), before);
        let mut bf16_graph = Graph::new();
        let bf16 = matrix_parameter(&mut bf16_graph, &shape, DType::BF16, &[1., -2., 0.5, 3.]);
        let mut bf16_optimizer = Optimizer::muon(vec![("p".into(), bf16.clone())], c).unwrap();
        bf16_optimizer
            .step(&BTreeMap::from([(
                "p".into(),
                matrix_gradient(&bf16, &shape, &first),
            )]))
            .unwrap();
        assert_eq!(bf16.snapshot().unwrap().dtype, DType::BF16);
        assert!(
            to_f64(&bf16.snapshot().unwrap().data)
                .iter()
                .all(|x| x.is_finite())
        );
        let mut invalid_graph = Graph::new();
        let vector = parameter(&mut invalid_graph, vec![1.]);
        assert!(Optimizer::muon(vec![("v".into(), vector)], MuonConfig::default()).is_err());
        let empty = matrix_parameter(&mut invalid_graph, &[0, 2], DType::F32, &[]);
        assert!(Optimizer::muon(vec![("e".into(), empty)], MuonConfig::default()).is_err());
        assert!(
            Optimizer::muon(
                vec![("x".into(), a.clone())],
                MuonConfig {
                    ns_coefficients: vec![],
                    ..MuonConfig::default()
                }
            )
            .is_err()
        );
        let integer = matrix_parameter(&mut invalid_graph, &[2, 2], DType::I32, &[1., 2., 3., 4.]);
        assert!(Optimizer::muon(vec![("i".into(), integer)], MuonConfig::default()).is_err());
        let zero = matrix_parameter(&mut invalid_graph, &[2, 2], DType::F32, &[0., 0., 0., 0.]);
        let mut zero_optimizer =
            Optimizer::muon(vec![("z".into(), zero.clone())], MuonConfig::default()).unwrap();
        zero_optimizer
            .step(&BTreeMap::from([(
                "z".into(),
                matrix_gradient(&zero, &[2, 2], &[0., 0., 0., 0.]),
            )]))
            .unwrap();
        assert_eq!(to_f64(&zero.snapshot().unwrap().data), vec![0.; 4]);
        let tied = matrix_parameter(&mut invalid_graph, &[2, 2], DType::F32, &[1., 1., 1., 1.]);
        let mut tied_optimizer = Optimizer::muon(
            vec![("a".into(), tied.clone()), ("b".into(), tied.clone())],
            MuonConfig {
                ns_steps: 0,
                ..MuonConfig::default()
            },
        )
        .unwrap();
        assert_eq!(tied_optimizer.parameter_names(), vec!["a"]);
        let stale = matrix_gradient(&tied, &[2, 2], &[0.1, 0.2, 0.3, 0.4]);
        tied.replace(TensorData::new([2, 2], vec![2., 2., 2., 2.]).unwrap())
            .unwrap();
        assert!(
            tied_optimizer
                .step(&BTreeMap::from([("a".into(), stale)]))
                .is_err()
        );
    }

    fn skip_list_group(first: Parameter, second: Parameter) -> OptimizerGroup {
        OptimizerGroup::new(vec![
            Optimizer::lars(
                vec![("lars".into(), first)],
                LarsConfig {
                    lr: 0.1,
                    momentum: 0.,
                    weight_decay: 0.,
                    tcoef: 0.,
                    ..LarsConfig::default()
                },
            )
            .unwrap(),
            Optimizer::sgd(
                vec![("sgd".into(), second)],
                SgdConfig {
                    lr: 0.1,
                    ..SgdConfig::default()
                },
            )
            .unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn optimizer_group_routes_skip_list_children_and_rejects_overlap() {
        let mut graph = Graph::new();
        let lars = parameter(&mut graph, vec![1.]);
        let sgd_parameter = parameter(&mut graph, vec![2.]);
        let mut group = skip_list_group(lars.clone(), sgd_parameter.clone());
        assert_eq!(group.len(), 2);
        assert_eq!(group[0].parameter_names(), vec!["lars"]);
        assert_eq!(group.get(1).unwrap().parameter_names(), vec!["sgd"]);
        group.zero_grad();
        group
            .step(&BTreeMap::from([
                ("lars".into(), gradient(&lars, vec![1.])),
                ("sgd".into(), gradient(&sgd_parameter, vec![2.])),
            ]))
            .unwrap();
        assert!((values(&lars)[0] - 0.9).abs() < 1e-6);
        assert!((values(&sgd_parameter)[0] - 1.8).abs() < 1e-6);
        let before = (values(&lars), values(&sgd_parameter));
        assert!(
            group
                .step(&BTreeMap::from([(
                    "lars".into(),
                    gradient(&lars, vec![1.])
                )]))
                .is_err()
        );
        assert_eq!((values(&lars), values(&sgd_parameter)), before);
        let overlap =
            Optimizer::sgd(vec![("other".into(), lars.clone())], SgdConfig::default()).unwrap();
        assert!(
            OptimizerGroup::new(vec![
                Optimizer::sgd(vec![("lars".into(), lars)], SgdConfig::default()).unwrap(),
                overlap,
            ])
            .is_err()
        );
    }

    #[test]
    fn optimizer_group_prevalidates_and_resumes_namespaced_state() {
        let mut graph = Graph::new();
        let left = parameter(&mut graph, vec![1.]);
        let right = parameter(&mut graph, vec![2.]);
        let mut group = skip_list_group(left.clone(), right.clone());
        let stale_right = gradient(&right, vec![1.]);
        right
            .replace(TensorData::new([1], vec![3.]).unwrap())
            .unwrap();
        let before = values(&left);
        assert!(
            group
                .step(&BTreeMap::from([
                    ("lars".into(), gradient(&left, vec![1.])),
                    ("sgd".into(), stale_right),
                ]))
                .is_err()
        );
        assert_eq!(values(&left), before);

        let mut a_graph = Graph::new();
        let a_left = parameter(&mut a_graph, vec![1.]);
        let a_right = parameter(&mut a_graph, vec![2.]);
        let mut uninterrupted = skip_list_group(a_left.clone(), a_right.clone());
        for (lars_grad, sgd_grad) in [(1., 2.), (-0.5, 0.25)] {
            uninterrupted
                .step(&BTreeMap::from([
                    ("lars".into(), gradient(&a_left, vec![lars_grad])),
                    ("sgd".into(), gradient(&a_right, vec![sgd_grad])),
                ]))
                .unwrap();
        }
        let mut saved_graph = Graph::new();
        let saved_left = parameter(&mut saved_graph, vec![1.]);
        let saved_right = parameter(&mut saved_graph, vec![2.]);
        let mut saved = skip_list_group(saved_left.clone(), saved_right.clone());
        saved
            .step(&BTreeMap::from([
                ("lars".into(), gradient(&saved_left, vec![1.])),
                ("sgd".into(), gradient(&saved_right, vec![2.])),
            ]))
            .unwrap();
        let checkpoint = saved.state_dict().unwrap();
        assert!(
            checkpoint
                .tensors()
                .contains_key("optimizer_group.0.optimizer.lars.momentum")
        );
        assert!(
            checkpoint
                .tensors()
                .contains_key("optimizer_group.1.optimizer.sgd.momentum")
        );
        let mut resumed_graph = Graph::new();
        let resumed_left = parameter(&mut resumed_graph, values(&saved_left));
        let resumed_right = parameter(&mut resumed_graph, values(&saved_right));
        let mut resumed = skip_list_group(resumed_left.clone(), resumed_right.clone());
        resumed.load_state_dict(&checkpoint).unwrap();
        resumed
            .step(&BTreeMap::from([
                ("lars".into(), gradient(&resumed_left, vec![-0.5])),
                ("sgd".into(), gradient(&resumed_right, vec![0.25])),
            ]))
            .unwrap();
        assert_eq!(values(&a_left), values(&resumed_left));
        assert_eq!(values(&a_right), values(&resumed_right));
        assert_eq!(
            uninterrupted.state_dict().unwrap(),
            resumed.state_dict().unwrap()
        );
        let mut wrong = OptimizerGroup::new(vec![
            Optimizer::lars(
                vec![("lars".into(), resumed_left.clone())],
                LarsConfig {
                    lr: 0.2,
                    ..LarsConfig::default()
                },
            )
            .unwrap(),
            Optimizer::sgd(
                vec![("sgd".into(), resumed_right.clone())],
                SgdConfig {
                    lr: 0.1,
                    ..SgdConfig::default()
                },
            )
            .unwrap(),
        ])
        .unwrap();
        let before = wrong.state_dict().unwrap();
        assert!(wrong.load_state_dict(&checkpoint).is_err());
        assert_eq!(wrong.state_dict().unwrap(), before);
        let mut raw = checkpoint.clone().into_tensors();
        raw.remove("optimizer_group.1.optimizer.sgd.momentum");
        let before = resumed.state_dict().unwrap();
        assert!(resumed.load_state_dict(&StateDict::from(raw)).is_err());
        assert_eq!(resumed.state_dict().unwrap(), before);
    }

    #[test]
    fn host_schedulers_match_static_formulas_and_group_order() {
        let mut graph = Graph::new();
        let p = parameter(&mut graph, vec![1.]);
        let mut optimizer = Optimizer::sgd(
            vec![("p".into(), p)],
            SgdConfig {
                lr: 1.,
                ..SgdConfig::default()
            },
        )
        .unwrap();
        let mut multi = LearningRateScheduler::multi_step(vec![1, 2], 0.1).unwrap();
        multi.step(&mut optimizer).unwrap();
        assert_eq!(optimizer.learning_rates(), &[1.]);
        multi.step(&mut optimizer).unwrap();
        assert_eq!(optimizer.learning_rates(), &[0.1]);
        multi.step(&mut optimizer).unwrap();
        assert!((optimizer.learning_rates()[0] - 0.01).abs() < 1e-12);
        let mut cosine = LearningRateScheduler::cosine_annealing(&optimizer, 4, 0.).unwrap();
        cosine.step(&mut optimizer).unwrap();
        assert!((optimizer.learning_rates()[0] - 0.01).abs() < 1e-12);
        cosine.step(&mut optimizer).unwrap();
        assert!(
            (optimizer.learning_rates()[0] - 0.005 * (1. + std::f64::consts::FRAC_1_SQRT_2)).abs()
                < 1e-12
        );
        let mut plateau = LearningRateScheduler::reduce_on_plateau(
            PlateauMode::Min,
            0.5,
            1,
            0.1,
            ThresholdMode::Relative,
        )
        .unwrap();
        plateau.step_metric(&mut optimizer, Some(2.)).unwrap();
        plateau.step_metric(&mut optimizer, Some(1.95)).unwrap();
        plateau.step_metric(&mut optimizer, Some(1.95)).unwrap();
        assert!(
            (optimizer.learning_rates()[0] - 0.5 * 0.005 * (1. + std::f64::consts::FRAC_1_SQRT_2))
                .abs()
                < 1e-12
        );
        let mut one_cycle =
            LearningRateScheduler::one_cycle(&mut optimizer, 1., 10., 10., 4, 0.5).unwrap();
        assert!((optimizer.learning_rates()[0] - 0.1).abs() < 1e-12);
        one_cycle.step(&mut optimizer).unwrap();
        one_cycle.step(&mut optimizer).unwrap();
        one_cycle.step(&mut optimizer).unwrap();
        assert!((optimizer.learning_rates()[0] - 1.).abs() < 1e-12);

        let left = parameter(&mut graph, vec![1.]);
        let right = parameter(&mut graph, vec![1.]);
        let mut group = skip_list_group(left, right);
        let mut scheduler_group = LrSchedulerGroup::new(vec![
            LearningRateScheduler::multi_step(vec![0], 0.5).unwrap(),
            LearningRateScheduler::multi_step(vec![0], 0.25).unwrap(),
        ]);
        scheduler_group.step(&mut group).unwrap();
        assert_eq!(group[0].learning_rates(), &[0.05]);
        assert_eq!(group[1].learning_rates(), &[0.025]);
    }

    #[test]
    fn scheduler_checkpoint_resume_is_atomic_with_optimizer_state() {
        let mut graph = Graph::new();
        let p = parameter(&mut graph, vec![1.]);
        let mut source = Optimizer::sgd(
            vec![("p".into(), p.clone())],
            SgdConfig {
                lr: 0.2,
                ..SgdConfig::default()
            },
        )
        .unwrap();
        let mut scheduler =
            LearningRateScheduler::one_cycle(&mut source, 0.2, 2., 10., 4, 0.5).unwrap();
        source
            .step(&BTreeMap::from([("p".into(), gradient(&p, vec![1.]))]))
            .unwrap();
        scheduler.step(&mut source).unwrap();
        let optimizer_state = source.state_dict().unwrap();
        let scheduler_state = scheduler.state_dict().unwrap();
        let mut resumed_graph = Graph::new();
        let resumed_p = parameter(&mut resumed_graph, values(&p));
        let mut resumed = Optimizer::sgd(
            vec![("p".into(), resumed_p.clone())],
            SgdConfig {
                lr: 9.,
                ..SgdConfig::default()
            },
        )
        .unwrap();
        let mut resumed_scheduler =
            LearningRateScheduler::one_cycle(&mut resumed, 0.2, 2., 10., 4, 0.5).unwrap();
        load_optimizer_scheduler_state(
            &mut resumed,
            &mut resumed_scheduler,
            &optimizer_state,
            &scheduler_state,
        )
        .unwrap();
        source
            .step(&BTreeMap::from([("p".into(), gradient(&p, vec![1.]))]))
            .unwrap();
        scheduler.step(&mut source).unwrap();
        resumed
            .step(&BTreeMap::from([(
                "p".into(),
                gradient(&resumed_p, vec![1.]),
            )]))
            .unwrap();
        resumed_scheduler.step(&mut resumed).unwrap();
        assert_eq!(values(&p), values(&resumed_p));
        assert_eq!(source.state_dict().unwrap(), resumed.state_dict().unwrap());
        assert_eq!(
            scheduler.state_dict().unwrap(),
            resumed_scheduler.state_dict().unwrap()
        );
        let before_optimizer = resumed.state_dict().unwrap();
        let before_scheduler = resumed_scheduler.state_dict().unwrap();
        let mut bad_scheduler = scheduler_state.into_tensors();
        bad_scheduler.remove("scheduler.epoch");
        assert!(
            load_optimizer_scheduler_state(
                &mut resumed,
                &mut resumed_scheduler,
                &optimizer_state,
                &StateDict::from(bad_scheduler)
            )
            .is_err()
        );
        assert_eq!(resumed.state_dict().unwrap(), before_optimizer);
        assert_eq!(resumed_scheduler.state_dict().unwrap(), before_scheduler);
    }

    #[test]
    fn training_checkpoint_rejects_each_mismatched_part_atomically() {
        let mut construction_graph = Graph::new();
        let linear = crate::nn::Linear::new(&mut construction_graph, 1, 1, false, 5).unwrap();
        let config = SgdConfig {
            lr: 0.2,
            momentum: 0.9,
            ..SgdConfig::default()
        };
        let mut source =
            Optimizer::sgd(vec![("weight".into(), linear.weight.clone())], config).unwrap();
        source
            .step(&BTreeMap::from([(
                "weight".into(),
                Gradient::for_parameter(
                    &linear.weight,
                    TensorData::new([1, 1], vec![0.5]).unwrap(),
                )
                .unwrap(),
            )]))
            .unwrap();
        let mut source_scheduler = LearningRateScheduler::multi_step(vec![0], 0.5).unwrap();
        source_scheduler.step(&mut source).unwrap();
        let checkpoint = TrainingCheckpoint::capture(&linear, &source, &source_scheduler).unwrap();

        let mut target =
            Optimizer::sgd(vec![("weight".into(), linear.weight.clone())], config).unwrap();
        let mut target_scheduler = LearningRateScheduler::multi_step(vec![0], 0.5).unwrap();
        let before_module = checkpoint_module_state(&linear).unwrap();
        let before_optimizer = target.state_dict().unwrap();
        let before_scheduler = target_scheduler.state_dict().unwrap();

        let assert_unchanged = |target: &Optimizer, scheduler: &LearningRateScheduler| {
            assert_eq!(checkpoint_module_state(&linear).unwrap(), before_module);
            assert_eq!(target.state_dict().unwrap(), before_optimizer);
            assert_eq!(scheduler.state_dict().unwrap(), before_scheduler);
        };

        let mut bad_module = checkpoint.clone();
        let mut module_tensors = linear.state_dict().unwrap().into_tensors();
        module_tensors.insert(
            "weight".into(),
            TensorData::new([1, 1], vec![123.]).unwrap(),
        );
        bad_module.module_safetensors =
            save_safetensors(&module_tensors, &BTreeMap::new()).unwrap();
        assert!(
            bad_module
                .resume(&linear, &mut target, &mut target_scheduler)
                .is_err()
        );
        assert_unchanged(&target, &target_scheduler);

        let mut bad_optimizer = checkpoint.clone();
        let mut optimizer_tensors = bad_optimizer.optimizer_state.into_tensors();
        optimizer_tensors.remove("optimizer.step");
        bad_optimizer.optimizer_state = StateDict::from(optimizer_tensors);
        assert!(
            bad_optimizer
                .resume(&linear, &mut target, &mut target_scheduler)
                .is_err()
        );
        assert_unchanged(&target, &target_scheduler);

        let mut bad_scheduler = checkpoint.clone();
        let mut scheduler_tensors = bad_scheduler.scheduler_state.into_tensors();
        scheduler_tensors.remove("scheduler.epoch");
        bad_scheduler.scheduler_state = StateDict::from(scheduler_tensors);
        assert!(
            bad_scheduler
                .resume(&linear, &mut target, &mut target_scheduler)
                .is_err()
        );
        assert_unchanged(&target, &target_scheduler);

        let mut other_graph = Graph::new();
        let other = crate::nn::Linear::new(&mut other_graph, 1, 1, false, 5).unwrap();
        let mut other_optimizer =
            Optimizer::sgd(vec![("weight".into(), other.weight.clone())], config).unwrap();
        let mut other_scheduler = LearningRateScheduler::multi_step(vec![0], 0.5).unwrap();
        let other_module_before = checkpoint_module_state(&other).unwrap();
        let other_optimizer_before = other_optimizer.state_dict().unwrap();
        let other_scheduler_before = other_scheduler.state_dict().unwrap();
        assert!(
            checkpoint
                .resume(&other, &mut other_optimizer, &mut other_scheduler)
                .is_err()
        );
        assert_eq!(
            checkpoint_module_state(&other).unwrap(),
            other_module_before
        );
        assert_eq!(
            other_optimizer.state_dict().unwrap(),
            other_optimizer_before
        );
        assert_eq!(
            other_scheduler.state_dict().unwrap(),
            other_scheduler_before
        );

        checkpoint
            .resume(&linear, &mut target, &mut target_scheduler)
            .unwrap();
        assert_eq!(target.state_dict().unwrap(), source.state_dict().unwrap());
        assert_eq!(
            target_scheduler.state_dict().unwrap(),
            source_scheduler.state_dict().unwrap()
        );
    }
}
