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
#[derive(Clone, Copy, Debug)]
pub enum OptimizerKind {
    Sgd(SgdConfig),
    Adam(AdamConfig),
    AdamW(AdamConfig),
    Lars(LarsConfig),
    Lamb(LambConfig),
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
                OptimizerKind::Sgd(_) | OptimizerKind::Lars(_) => Slots::Sgd(Vec::new()),
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
                (OptimizerKind::Lars(config), Slots::Sgd(momentum)) => {
                    lars(values, grad, &mut momentum[pos], entry.first_step, config)
                }
                (OptimizerKind::Lamb(config), Slots::Adam { mean, variance }) => lamb(
                    values,
                    grad,
                    &mut mean[pos],
                    &mut variance[pos],
                    next_step,
                    config,
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
        let next_step = step.scalar_at(0).as_u64();
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
        self.step = next_step;
        for (entry, version) in self.entries.iter_mut().zip(next_versions) {
            entry.version = version;
        }
        Ok(())
    }
    fn expected_state_keys(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::from(["optimizer.config".into(), "optimizer.step".into()]);
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
        let mut out = b"rustgrad-optimizer\0\x01".to_vec();
        out.extend_from_slice(&(self.groups.len() as u64).to_le_bytes());
        for (index, kind) in self.groups.iter().enumerate() {
            out.extend_from_slice(
                &(self.entries.iter().filter(|e| e.group == index).count() as u64).to_le_bytes(),
            );
            match kind {
                OptimizerKind::Sgd(c) => {
                    out.push(0);
                    for x in [c.lr, c.momentum, c.dampening, c.weight_decay] {
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
                    for x in [c.lr, c.beta1, c.beta2, c.eps, c.weight_decay] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                }
                OptimizerKind::Lars(c) => {
                    out.push(3);
                    for x in [c.lr, c.momentum, c.weight_decay, c.tcoef] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                    out.extend_from_slice(&[c.nesterov as u8, c.classic as u8, c.pre_wd as u8])
                }
                OptimizerKind::Lamb(c) => {
                    out.push(4);
                    for x in [c.lr, c.beta1, c.beta2, c.eps, c.weight_decay] {
                        out.extend_from_slice(&x.to_le_bytes())
                    }
                    out.push(c.adam as u8)
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
fn validate(kind: OptimizerKind) -> Result<()> {
    let (lr, wd) = match kind {
        OptimizerKind::Sgd(c) => (c.lr, c.weight_decay),
        OptimizerKind::Adam(c) | OptimizerKind::AdamW(c) => (c.lr, c.weight_decay),
        OptimizerKind::Lars(c) => (c.lr, c.weight_decay),
        OptimizerKind::Lamb(c) => (c.lr, c.weight_decay),
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
            LarsConfig { lr: 0.2, ..c },
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
            LambConfig { lr: 0.04, ..c },
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
}
