//! Imperative dense optimizers for versioned [`crate::Parameter`] leaves.
//!
//! Evaluate graph gradient nodes with `Module::input_bindings`, wrap each dense
//! result with [`Gradient::for_parameter`], then call [`Optimizer::step`]. A
//! step checks the captured parameter versions before replacement, so callers
//! must rebuild/evaluate the next graph cycle after an update.

use crate::nn::StateDict;
use crate::{DType, Error, Parameter, Result, Scalar, Shape, TensorData};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct Gradient {
    pub data: TensorData,
    version: u64,
}
impl Gradient {
    pub fn for_parameter(parameter: &Parameter, data: TensorData) -> Result<Self> {
        Ok(Self {
            data,
            version: parameter.snapshot()?.version,
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
#[derive(Clone, Copy, Debug)]
pub enum OptimizerKind {
    Sgd(SgdConfig),
    Adam(AdamConfig),
    AdamW(AdamConfig),
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
struct Entry {
    name: String,
    parameter: Parameter,
    version: u64,
    group: usize,
    first_step: bool,
}
enum Slots {
    Sgd(Vec<Vec<f64>>),
    Adam {
        mean: Vec<Vec<f64>>,
        variance: Vec<Vec<f64>>,
    },
}

/// Deterministically ordered, dense CPU optimizer state. It accepts only
/// explicit, already-evaluated gradients; it never owns a graph or global tape.
pub struct Optimizer {
    entries: Vec<Entry>,
    groups: Vec<OptimizerKind>,
    slots: Vec<Slots>,
    step: u64,
}
impl Optimizer {
    pub fn new(groups: Vec<ParameterGroup>) -> Result<Self> {
        if groups.is_empty() {
            return Err(invalid("optimizer needs at least one parameter group"));
        }
        let mut entries = Vec::new();
        let mut kinds = Vec::new();
        let mut seen = BTreeSet::new();
        for (group_index, group) in groups.into_iter().enumerate() {
            validate(group.kind)?;
            kinds.push(group.kind);
            for (name, parameter) in group.parameters {
                if !parameter.is_trainable() {
                    continue;
                }
                let snapshot = parameter.snapshot()?;
                if !snapshot.dtype.is_float() {
                    return Err(invalid("optimizer parameters must have float dtype"));
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
                OptimizerKind::Sgd(_) => Slots::Sgd(Vec::new()),
                OptimizerKind::Adam(_) | OptimizerKind::AdamW(_) => Slots::Adam {
                    mean: Vec::new(),
                    variance: Vec::new(),
                },
            })
            .collect();
        let mut optimizer = Self {
            entries,
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
    pub fn step_count(&self) -> u64 {
        self.step
    }
    pub fn parameter_names(&self) -> Vec<&str> {
        self.entries.iter().map(|x| x.name.as_str()).collect()
    }
    pub fn zero_grad(&self) { /* gradients are caller-owned and never retained */
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
            let updated = match (self.groups[entry.group], &mut self.slots[entry.group]) {
                (OptimizerKind::Sgd(config), Slots::Sgd(momentum)) => {
                    sgd(values, grad, &mut momentum[pos], entry.first_step, config)
                }
                (OptimizerKind::Adam(config), Slots::Adam { mean, variance }) => adam(
                    values,
                    grad,
                    &mut mean[pos],
                    &mut variance[pos],
                    next_step,
                    config,
                    false,
                ),
                (OptimizerKind::AdamW(config), Slots::Adam { mean, variance }) => adam(
                    values,
                    grad,
                    &mut mean[pos],
                    &mut variance[pos],
                    next_step,
                    config,
                    true,
                ),
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
        let step = state
            .tensors()
            .get("optimizer.step")
            .ok_or_else(|| invalid("optimizer state missing step"))?;
        if step.len() != 1 {
            return Err(invalid("invalid optimizer step"));
        };
        self.step = step.scalar_at(0).as_u64();
        let mut positions = vec![0usize; self.groups.len()];
        for entry in &mut self.entries {
            let pos = positions[entry.group];
            positions[entry.group] += 1;
            let load = |suffix: &str| -> Result<Vec<f64>> {
                let value = state
                    .tensors()
                    .get(&format!("optimizer.{}.{}", entry.name, suffix))
                    .ok_or_else(|| invalid("optimizer state missing slot"))?;
                if value.shape() != &entry.parameter.snapshot()?.shape {
                    return Err(invalid("optimizer state shape mismatch"));
                };
                Ok(to_f64(value))
            };
            match &mut self.slots[entry.group] {
                Slots::Sgd(momentum) => momentum[pos] = load("momentum")?,
                Slots::Adam { mean, variance } => {
                    mean[pos] = load("exp_avg")?;
                    variance[pos] = load("exp_avg_sq")?
                }
            };
            entry.version = entry.parameter.snapshot()?.version;
        }
        Ok(())
    }
}
fn invalid(reason: &str) -> Error {
    Error::Serialization {
        reason: format!("optimizer: {reason}"),
    }
}
fn validate(kind: OptimizerKind) -> Result<()> {
    let (lr, wd) = match kind {
        OptimizerKind::Sgd(c) => (c.lr, c.weight_decay),
        OptimizerKind::Adam(c) | OptimizerKind::AdamW(c) => (c.lr, c.weight_decay),
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
    }
}
fn validate_gradient(snapshot: &crate::ParameterSnapshot, gradient: &Gradient) -> Result<()> {
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

    fn parameter(graph: &mut Graph, values: Vec<f32>) -> Parameter {
        Parameter::new(
            graph,
            TensorData::new([values.len()], values).unwrap(),
            true,
        )
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
        let mut resume_graph = Graph::new();
        let resumed = Parameter::new(&mut resume_graph, value, true);
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
        let mut bindings = linear.input_bindings().unwrap();
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
        let mut bindings = linear.input_bindings().unwrap();
        bindings.insert("x".into(), TensorData::new([1, 1], vec![1.]).unwrap());
        let after = cpu
            .execute(&graph, loss, &bindings)
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
}
