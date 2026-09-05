//! Graph-free CPU replay for static training programs with recurrent state.

use crate::nn::StateKind;
use crate::runtime::metal::{
    MetalDevice, MetalDeviceRunReport, MetalDeviceSession, MetalDeviceSessionSummary, MetalError,
    MetalRenderer, MetalStatefulInferencePlan, RenderedMetal,
};
use crate::{
    BufferState, CapturedMixedSchedule, CapturedSchedule, CapturedStatefulInference, DType,
    EffectGraph, EffectRuntime, Error, Graph, InferenceStateLink, Metadata, MixedReplayCursor,
    Module, NodeId, ParameterId, ReplayError, Result, Scalar, Schedule, ScheduleStateBinding,
    ScheduleValueBinding, Shape, StateDict, TensorData, bind_schedule_states,
    combine_mixed_schedules, load_safetensors, save_safetensors, schedule_effects, schedule_many,
};
use std::collections::{BTreeMap, BTreeSet};

const INTERNAL_PREFIX: &str = "__rustgrad_compiled_training_";
const LEARNING_RATE_INPUT: &str = "__rustgrad_compiled_training_learning_rate";
const STATE_BUFFER_BASE: u64 = 1_u64 << 62;

/// Detached initial value for one compiled training parameter.
///
/// Construction does not create an [`crate::nn::Parameter`] or retain a live
/// module handle. The value is consumed by compilation and subsequently owned
/// only by the compiled session's [`EffectRuntime`].
#[derive(Clone, Debug)]
pub struct TrainingParameterInit {
    name: String,
    value: TensorData,
}

impl TrainingParameterInit {
    pub fn new(name: impl Into<String>, value: TensorData) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "parameter")?;
        if value.dtype() != DType::F32 {
            return Err(training("compiled training parameters must be F32"));
        }
        checked_bytes(&value)?;
        Ok(Self { name, value })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> &TensorData {
        &self.value
    }
}

#[derive(Clone, Debug)]
struct ModuleParameterEntry {
    identity: ParameterId,
    name: String,
    value: TensorData,
    trainable: bool,
}

/// Frozen snapshot of one module's parameter topology for compilation.
///
/// Trainable identities become recurrent optimizer-owned inputs. Frozen
/// parameters and buffers become immutable capture constants. Tied traversal
/// entries keep one identity and therefore resolve to exactly one graph node.
#[derive(Clone, Debug)]
struct ModuleParameterPlan {
    entries: Vec<ModuleParameterEntry>,
}

impl ModuleParameterPlan {
    fn new(module: &(impl Module + ?Sized)) -> Result<Self> {
        let mut entries = Vec::<ModuleParameterEntry>::new();
        let mut identities = BTreeMap::<ParameterId, usize>::new();
        let mut names = BTreeSet::new();
        let mut error = None;
        module.visit("", &mut |name, parameter, kind| {
            if error.is_some() {
                return;
            }
            let identity = parameter.id();
            if let Some(&index) = identities.get(&identity) {
                let is_parameter = matches!(kind, StateKind::Parameter);
                if entries[index].trainable != (parameter.is_trainable() && is_parameter) {
                    error = Some(training(
                        "tied compiled module state has inconsistent trainability",
                    ));
                }
                return;
            }
            if !names.insert(name.clone()) {
                error = Some(training("compiled module state names repeat"));
                return;
            }
            match parameter.snapshot() {
                Ok(snapshot) => {
                    let trainable = snapshot.trainable && matches!(kind, StateKind::Parameter);
                    identities.insert(identity, entries.len());
                    entries.push(ModuleParameterEntry {
                        identity,
                        name,
                        value: snapshot.data,
                        trainable,
                    });
                }
                Err(err) => error = Some(err),
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        if entries.iter().all(|entry| !entry.trainable) {
            return Err(training(
                "compiled module needs at least one trainable parameter",
            ));
        }
        Ok(Self { entries })
    }

    fn initial_parameters(&self) -> Result<Vec<TrainingParameterInit>> {
        self.entries
            .iter()
            .filter(|entry| entry.trainable)
            .map(|entry| TrainingParameterInit::new(entry.name.clone(), entry.value.clone()))
            .collect()
    }

    fn lower<T>(
        &self,
        graph: &mut Graph,
        parameters: &BTreeMap<String, NodeId>,
        build: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        let mut overrides = BTreeMap::new();
        for entry in &self.entries {
            let node = if entry.trainable {
                parameters
                    .get(&entry.name)
                    .copied()
                    .ok_or_else(|| training("compiled module parameter set mismatch"))?
            } else {
                graph.constant(entry.value.clone())
            };
            if graph.shape(node)? != entry.value.shape()
                || graph.dtype(node)? != entry.value.dtype()
                || graph.requires_grad(node)? != entry.trainable
            {
                return Err(training("compiled module parameter descriptor mismatch"));
            }
            overrides.insert(entry.identity, node);
        }
        graph.with_parameter_overrides(overrides, build)
    }
}

/// Static compilation policy for [`CpuCompiledMomentumSgd`].
#[derive(Clone, Debug)]
pub struct CompiledMomentumSgdConfig {
    momentum: f32,
    inputs: BTreeMap<String, (Shape, DType)>,
}

impl CompiledMomentumSgdConfig {
    /// Creates the source-style momentum rule `v = momentum*v + grad`.
    /// Tinygrad rejects only ordered negative momentum, so NaN and infinity
    /// remain ordinary graph constants rather than receiving an invented
    /// finite-value restriction here.
    pub fn new(momentum: f32) -> Result<Self> {
        if momentum < 0.0 {
            return Err(training(
                "compiled momentum-SGD momentum must be nonnegative",
            ));
        }
        Ok(Self {
            momentum,
            inputs: BTreeMap::new(),
        })
    }

    /// Adds one exact external input descriptor. Names are deterministic and
    /// may not overlap the session's private state/LR namespace.
    pub fn with_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, dtype)?;
        if self.inputs.insert(name, (shape, dtype)).is_some() {
            return Err(training("duplicate compiled training input name"));
        }
        Ok(self)
    }

    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&str, &Shape, DType)> {
        self.inputs
            .iter()
            .map(|(name, (shape, dtype))| (name.as_str(), shape, *dtype))
    }
}

/// Static compilation policy for [`CpuCompiledAdamW`].
#[derive(Clone, Debug)]
pub struct CompiledAdamWConfig {
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    inputs: BTreeMap<String, (Shape, DType)>,
}

impl CompiledAdamWConfig {
    pub fn new(beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Result<Self> {
        if !(0.0..1.0).contains(&beta1)
            || !(0.0..1.0).contains(&beta2)
            || !eps.is_finite()
            || eps <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return Err(training(
                "compiled AdamW requires beta1/beta2 in [0,1), positive finite epsilon, and finite nonnegative weight decay",
            ));
        }
        Ok(Self {
            beta1,
            beta2,
            eps,
            weight_decay,
            inputs: BTreeMap::new(),
        })
    }

    pub fn with_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, dtype)?;
        if self.inputs.insert(name, (shape, dtype)).is_some() {
            return Err(training("duplicate compiled training input name"));
        }
        Ok(self)
    }

    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&str, &Shape, DType)> {
        self.inputs
            .iter()
            .map(|(name, (shape, dtype))| (name.as_str(), shape, *dtype))
    }
}

/// Detached result of one successfully committed compiled training step.
#[derive(Clone, Debug)]
pub struct CompiledTrainingStepResult {
    loss: TensorData,
    outputs: BTreeMap<String, TensorData>,
    step: u64,
    capture_identity: u64,
}

impl CompiledTrainingStepResult {
    pub fn loss(&self) -> &TensorData {
        &self.loss
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        &self.outputs
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs.get(name)
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }
}

pub type CompiledMomentumSgdStepResult = CompiledTrainingStepResult;
pub type CompiledAdamWStepResult = CompiledTrainingStepResult;

const ADAMW_CHECKPOINT_FORMAT: &str = "rustgrad-compiled-adamw-v1";

/// Deterministic, portable state for one exact [`CpuCompiledAdamW`] program.
///
/// The safetensors payload contains parameter and moment tensors. String
/// metadata authenticates the format, ordered parameter names, compiled
/// capture identity, and logical step. It never serializes executable code,
/// graphs, runtime slots, or host pointers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAdamWCheckpoint {
    bytes: Vec<u8>,
}

impl CompiledAdamWCheckpoint {
    /// Validates and owns deterministic checkpoint bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        decode_adamw_checkpoint(&bytes)?;
        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

struct DecodedAdamWCheckpoint {
    capture_identity: u64,
    step: u64,
    parameters: BTreeMap<String, TensorData>,
    first_moments: BTreeMap<String, TensorData>,
    second_moments: BTreeMap<String, TensorData>,
}

#[derive(Clone, Debug)]
struct StateSpec {
    key: String,
    input_name: String,
    value: TensorData,
    requires_grad: bool,
}

trait CompiledOptimizerProgram {
    fn name(&self) -> &'static str;
    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)>;
    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>>;
    fn lower_updates(
        &self,
        graph: &mut Graph,
        learning_rate: NodeId,
        parameters: &BTreeMap<String, NodeId>,
        gradients: &BTreeMap<String, NodeId>,
        states: &BTreeMap<String, NodeId>,
    ) -> Result<BTreeMap<String, NodeId>>;
}

struct MomentumProgram {
    config: CompiledMomentumSgdConfig,
}

struct AdamWProgram {
    config: CompiledAdamWConfig,
}

fn parameter_key(name: &str) -> String {
    format!("parameter:{name}")
}

fn slot_key(name: &str, slot: &str) -> String {
    format!("slot:{name}:{slot}")
}

impl CompiledOptimizerProgram for MomentumProgram {
    fn name(&self) -> &'static str {
        "momentum-SGD"
    }

    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)> {
        &self.config.inputs
    }

    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>> {
        let mut specs = Vec::with_capacity(parameters.len() * 2);
        for (ordinal, (name, value)) in parameters.iter().enumerate() {
            specs.push(StateSpec {
                key: parameter_key(name),
                input_name: format!("{INTERNAL_PREFIX}parameter_{ordinal}"),
                value: value.clone(),
                requires_grad: true,
            });
            specs.push(StateSpec {
                key: slot_key(name, "momentum"),
                input_name: format!("{INTERNAL_PREFIX}momentum_{ordinal}"),
                value: TensorData::zeros_with_dtype(value.shape().clone(), DType::F32)?,
                requires_grad: false,
            });
        }
        Ok(specs)
    }

    fn lower_updates(
        &self,
        graph: &mut Graph,
        learning_rate: NodeId,
        parameters: &BTreeMap<String, NodeId>,
        gradients: &BTreeMap<String, NodeId>,
        states: &BTreeMap<String, NodeId>,
    ) -> Result<BTreeMap<String, NodeId>> {
        let momentum = scalar_f32(graph, self.config.momentum)?;
        let mut updates = BTreeMap::new();
        for (name, parameter) in parameters {
            let slot = states[&slot_key(name, "momentum")];
            let retained = graph.mul(momentum, slot)?;
            let next_momentum = graph.add(retained, gradients[name])?;
            let scaled = graph.mul(learning_rate, next_momentum)?;
            let next_parameter = graph.sub(*parameter, scaled)?;
            validate_parameter_update(graph, *parameter, next_momentum)?;
            validate_parameter_update(graph, *parameter, next_parameter)?;
            updates.insert(slot_key(name, "momentum"), next_momentum);
            updates.insert(parameter_key(name), next_parameter);
        }
        Ok(updates)
    }
}

impl CompiledOptimizerProgram for AdamWProgram {
    fn name(&self) -> &'static str {
        "AdamW"
    }

    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)> {
        &self.config.inputs
    }

    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>> {
        let mut specs = Vec::with_capacity(parameters.len() * 3 + 1);
        for (ordinal, (name, value)) in parameters.iter().enumerate() {
            specs.push(StateSpec {
                key: parameter_key(name),
                input_name: format!("{INTERNAL_PREFIX}parameter_{ordinal}"),
                value: value.clone(),
                requires_grad: true,
            });
            for slot in ["first_moment", "second_moment"] {
                specs.push(StateSpec {
                    key: slot_key(name, slot),
                    input_name: format!("{INTERNAL_PREFIX}{slot}_{ordinal}"),
                    value: TensorData::zeros_with_dtype(value.shape().clone(), DType::F32)?,
                    requires_grad: false,
                });
            }
        }
        specs.push(StateSpec {
            key: "global:step".into(),
            input_name: format!("{INTERNAL_PREFIX}adamw_step"),
            value: TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(0)])?,
            requires_grad: false,
        });
        Ok(specs)
    }

    fn lower_updates(
        &self,
        graph: &mut Graph,
        learning_rate: NodeId,
        parameters: &BTreeMap<String, NodeId>,
        gradients: &BTreeMap<String, NodeId>,
        states: &BTreeMap<String, NodeId>,
    ) -> Result<BTreeMap<String, NodeId>> {
        let one_u64 = graph.full_with_dtype(Shape::from([]), Scalar::U(1), DType::U64)?;
        let next_step = graph.add(states["global:step"], one_u64)?;
        let step_f32 = graph.cast(next_step, DType::F32)?;
        let one = scalar_f32(graph, 1.0)?;
        let beta1 = scalar_f32(graph, self.config.beta1)?;
        let beta2 = scalar_f32(graph, self.config.beta2)?;
        let one_minus_beta1 = scalar_f32(graph, 1.0 - self.config.beta1)?;
        let one_minus_beta2 = scalar_f32(graph, 1.0 - self.config.beta2)?;
        let eps = scalar_f32(graph, self.config.eps)?;
        let weight_decay = scalar_f32(graph, self.config.weight_decay)?;
        let beta1_power = graph.pow(beta1, step_f32)?;
        let beta2_power = graph.pow(beta2, step_f32)?;
        let first_correction = graph.sub(one, beta1_power)?;
        let second_correction = graph.sub(one, beta2_power)?;
        let decay = graph.mul(learning_rate, weight_decay)?;
        let decay_factor = graph.sub(one, decay)?;

        let mut updates = BTreeMap::from([("global:step".into(), next_step)]);
        for (name, parameter) in parameters {
            let gradient = gradients[name];
            let first_key = slot_key(name, "first_moment");
            let second_key = slot_key(name, "second_moment");
            let retained_first = graph.mul(beta1, states[&first_key])?;
            let fresh_first = graph.mul(one_minus_beta1, gradient)?;
            let next_first = graph.add(retained_first, fresh_first)?;
            let retained_second = graph.mul(beta2, states[&second_key])?;
            let gradient_squared = graph.mul(gradient, gradient)?;
            let fresh_second = graph.mul(one_minus_beta2, gradient_squared)?;
            let next_second = graph.add(retained_second, fresh_second)?;
            let corrected_first = graph.div(next_first, first_correction)?;
            let corrected_second = graph.div(next_second, second_correction)?;
            let root = graph.sqrt(corrected_second)?;
            let denominator = graph.add(root, eps)?;
            let normalized = graph.div(corrected_first, denominator)?;
            let decayed = graph.mul(*parameter, decay_factor)?;
            let scaled = graph.mul(learning_rate, normalized)?;
            let next_parameter = graph.sub(decayed, scaled)?;
            validate_parameter_update(graph, *parameter, next_first)?;
            validate_parameter_update(graph, *parameter, next_second)?;
            validate_parameter_update(graph, *parameter, next_parameter)?;
            updates.insert(first_key, next_first);
            updates.insert(second_key, next_second);
            updates.insert(parameter_key(name), next_parameter);
        }
        Ok(updates)
    }
}

fn scalar_f32(graph: &mut Graph, value: f32) -> Result<NodeId> {
    graph.full_with_dtype(Shape::from([]), Scalar::F(value as f64), DType::F32)
}

fn validate_parameter_update(graph: &Graph, parameter: NodeId, update: NodeId) -> Result<()> {
    if graph.shape(update)? != graph.shape(parameter)? || graph.dtype(update)? != DType::F32 {
        return Err(training("compiled optimizer update descriptor mismatch"));
    }
    Ok(())
}

/// One compiled momentum-SGD training program.
pub struct CpuCompiledMomentumSgd {
    inner: CpuCompiledTrainingProgram,
}

/// One compiled AdamW training program with recurrent first/second moments and
/// a graph-owned step counter.
pub struct CpuCompiledAdamW {
    inner: CpuCompiledTrainingProgram,
}

/// Resource-free Metal rendering of the same recurrent program owned by a
/// [`CpuCompiledAdamW`] session. Preparing it uploads the session's current
/// parameter, moment, and optimizer-step frontier into the existing
/// epoch-swapped Metal runtime.
pub struct MetalCompiledAdamWPlan {
    inner: MetalStatefulInferencePlan,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    state_input_keys: BTreeMap<String, String>,
    program_identity: u64,
    step: u64,
}

/// Device-resident AdamW training session backed by one fixed Metal capture.
/// Parameters and optimizer slots remain in the double-buffered device state
/// frontier between calls; only batch inputs, learning rate, and requested
/// outputs cross the host boundary on each step.
pub struct MetalCompiledAdamW {
    session: MetalDeviceSession,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    state_input_keys: BTreeMap<String, String>,
    program_identity: u64,
    step: u64,
}

/// One committed Metal AdamW step plus its exact device execution report.
pub struct MetalCompiledAdamWStepResult {
    inner: CompiledTrainingStepResult,
    report: MetalDeviceRunReport,
}

impl MetalCompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

/// One static CPU momentum-SGD program with runtime-owned recurrent state.
///
/// Compilation builds one pure graph and one mixed capture. The graph is then
/// dropped: every later step is graph-free interpreter replay through the
/// capture, and [`EffectRuntime`] is the sole owner of parameter/momentum
/// bytes. This type deliberately has no live-module synchronization surface.
struct CpuCompiledTrainingProgram {
    capture: CapturedMixedSchedule,
    recurrent_capture: CapturedStatefulInference,
    runtime: EffectRuntime,
    cursor: MixedReplayCursor,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    parameter_buffers: BTreeMap<String, u64>,
    optimizer_buffers: BTreeMap<String, u64>,
    state_input_buffers: BTreeMap<String, u64>,
    state_input_keys: BTreeMap<String, String>,
    step: u64,
}

impl CpuCompiledTrainingProgram {
    /// Compiles one exact static training program.
    ///
    /// `build` receives the declared external inputs and detached parameter
    /// graph inputs. It returns one scalar F32 loss and deterministically named
    /// detached outputs. All parameter gradients are constructed by exactly
    /// one [`Graph::gradient_default`] traversal.
    fn compile<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let parameters = canonical_parameters(parameters)?;
        if parameters.is_empty() {
            return Err(training(format!(
                "compiled {} needs at least one parameter",
                optimizer.name()
            )));
        }
        if parameters
            .keys()
            .any(|name| optimizer.inputs().contains_key(name))
        {
            return Err(training(
                "compiled parameter and input names must be globally unique",
            ));
        }

        let mut graph = Graph::new();
        let inputs = optimizer
            .inputs()
            .iter()
            .map(|(name, (shape, dtype))| {
                (
                    name.clone(),
                    graph.input_dtype_requires_grad(name.clone(), shape.clone(), *dtype, false),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let learning_rate = graph.input_dtype_requires_grad(
            LEARNING_RATE_INPUT,
            Shape::from([]),
            DType::F32,
            false,
        );

        let specs = optimizer.state_specs(&parameters)?;
        let mut parameter_nodes = BTreeMap::new();
        let mut state_nodes = BTreeMap::new();
        let mut state_values = Vec::with_capacity(specs.len());
        let mut state_by_input = BTreeMap::new();
        let mut parameter_buffers = BTreeMap::new();
        let mut optimizer_buffers = BTreeMap::new();
        let mut state_input_buffers = BTreeMap::new();
        let mut state_input_keys = BTreeMap::new();
        for (ordinal, spec) in specs.iter().enumerate() {
            let ordinal = u64::try_from(ordinal).map_err(|_| training("parameter overflow"))?;
            let parameter_buffer = STATE_BUFFER_BASE
                .checked_add(ordinal)
                .ok_or_else(|| training("parameter buffer overflow"))?;
            let node = graph.input_dtype_requires_grad(
                spec.input_name.clone(),
                spec.value.shape().clone(),
                spec.value.dtype(),
                spec.requires_grad,
            );
            let state = state_for(parameter_buffer, &spec.value)?;
            state_nodes.insert(spec.key.clone(), node);
            state_by_input.insert(node, state);
            state_values.push((parameter_buffer, spec.value.clone()));
            state_input_buffers.insert(spec.input_name.clone(), parameter_buffer);
            state_input_keys.insert(spec.input_name.clone(), spec.key.clone());
            if let Some(name) = spec.key.strip_prefix("parameter:") {
                parameter_nodes.insert(name.to_string(), node);
                parameter_buffers.insert(name.to_string(), parameter_buffer);
            } else {
                optimizer_buffers.insert(spec.key.clone(), parameter_buffer);
            }
        }
        if parameter_nodes.len() != parameters.len() {
            return Err(training("compiled optimizer omitted parameter state"));
        }

        let (loss, outputs) = build(&mut graph, &inputs, &parameter_nodes)?;
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            optimizer.inputs().keys().chain(parameters.keys()),
        )?;

        let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
        let gradients = graph.gradient_default(loss, &targets)?;
        if gradients.len() != targets.len() {
            return Err(training("compiled gradient target count mismatch"));
        }
        let gradients = parameter_nodes
            .keys()
            .cloned()
            .zip(gradients)
            .collect::<BTreeMap<_, _>>();
        let updates = optimizer.lower_updates(
            &mut graph,
            learning_rate,
            &parameter_nodes,
            &gradients,
            &state_nodes,
        )?;
        if updates.len() != specs.len() || specs.iter().any(|spec| !updates.contains_key(&spec.key))
        {
            return Err(training("compiled optimizer successor set mismatch"));
        }

        let public_requested = std::iter::once(loss)
            .chain(outputs.values().copied())
            .collect::<Vec<_>>();
        let state_links = specs
            .iter()
            .map(|spec| InferenceStateLink::new(state_nodes[&spec.key], updates[&spec.key]))
            .collect::<Vec<_>>();
        let initial_state = specs
            .iter()
            .map(|spec| (spec.input_name.clone(), spec.value.clone()))
            .collect();
        let recurrent_capture = CapturedStatefulInference::from_graph(
            &graph,
            &public_requested,
            &state_links,
            initial_state,
        )
        .map_err(captured_inference_error)?;

        let mut requested = Vec::with_capacity(1 + outputs.len() + updates.len());
        requested.extend(public_requested);
        for spec in &specs {
            requested.push(updates[&spec.key]);
        }
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled pure prefix has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured =
            CapturedSchedule::capture(&graph, &pure, &requested[..1 + outputs.len()])
                .map_err(replay_error)?;
        if captured.requested.len() != 1 + outputs.len() {
            return Err(training("compiled capture output count mismatch"));
        }

        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(updates.len());
        for (ordinal, spec) in specs.iter().enumerate() {
            let next = updates[&spec.key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let buffer = state_values[ordinal].0;
            let destination = effects
                .insert(buffer, spec.value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(spec.value.shape().clone(), spec.value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            let effect_index =
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?;
            effect_bindings.push(value_binding(&pure, next, effect_index)?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let states = effect_states(&effects)?;
        let capture =
            CapturedMixedSchedule::from_parts(captured, &mixed, states).map_err(replay_error)?;
        validate_external_binding_ownership(&capture, optimizer.inputs().keys())?;

        // Runtime ownership is published only after every graph, schedule,
        // effect, capture, and descriptor check above has succeeded.
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_states(state_values)
            .map_err(runtime_error)?;
        let cursor = capture.initial_recurrent_cursor().map_err(replay_error)?;
        let output_names = outputs.keys().cloned().collect();
        Ok(Self {
            capture,
            recurrent_capture,
            runtime,
            cursor,
            inputs: optimizer.inputs().clone(),
            output_names,
            parameter_buffers,
            optimizer_buffers,
            state_input_buffers,
            state_input_keys,
            step: 0,
        })
    }

    /// Executes one graph-free replay and atomically publishes every recurrent
    /// successor. The learning rate is an explicit rank-zero F32 input.
    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_inner(inputs, learning_rate, None)
    }

    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        validate_step_inputs(&self.inputs, &inputs, &learning_rate)?;
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| training("compiled training step overflow"))?;
        let mut provided = inputs;
        provided.insert(LEARNING_RATE_INPUT.to_string(), learning_rate);
        let replay = self
            .capture
            .replay_recurrent(
                &mut self.runtime,
                &mut self.cursor,
                &provided,
                injected_failure,
            )
            .map_err(replay_error)?;
        debug_assert_eq!(replay.outputs.len(), 1 + self.output_names.len());
        let mut outputs = replay.outputs.into_iter();
        let loss = outputs
            .next()
            .expect("compiled output cardinality was validated before publication");
        let outputs = self
            .output_names
            .iter()
            .cloned()
            .zip(outputs)
            .collect::<BTreeMap<_, _>>();
        self.step = next_step;
        Ok(CompiledTrainingStepResult {
            loss,
            outputs,
            step: self.step,
            capture_identity: self.cursor.capture_identity(),
        })
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn capture_identity(&self) -> u64 {
        self.cursor.capture_identity()
    }

    fn recurrent_capture(&self) -> Result<CapturedStatefulInference> {
        let initial_state = self
            .state_input_buffers
            .iter()
            .map(|(name, buffer)| {
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((name.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.recurrent_capture
            .clone()
            .with_initial_state(initial_state)
            .map_err(captured_inference_error)
    }

    /// Returns independent owned parameter snapshots in canonical name order.
    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.snapshots(&self.parameter_buffers)
    }

    /// Current logical parameter versions. Every successful step advances all
    /// parameter and optimizer-state buffers exactly once.
    fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.versions(&self.parameter_buffers)
    }

    fn slot_snapshots(&self, slot: &str) -> Result<BTreeMap<String, TensorData>> {
        let suffix = format!(":{slot}");
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.strip_prefix("slot:")
                    .and_then(|key| key.strip_suffix(&suffix))
                    .map(|name| (name.to_string(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.snapshots(&buffers)
    }

    fn slot_versions(&self, slot: &str) -> Result<BTreeMap<String, u64>> {
        let suffix = format!(":{slot}");
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.strip_prefix("slot:")
                    .and_then(|key| key.strip_suffix(&suffix))
                    .map(|name| (name.to_string(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.versions(&buffers)
    }

    fn global_snapshot(&self, name: &str) -> Result<TensorData> {
        let buffer = self
            .optimizer_buffers
            .get(&format!("global:{name}"))
            .ok_or_else(|| training("compiled global optimizer state is absent"))?;
        let state = self.current_state(*buffer)?;
        Ok(self
            .runtime
            .snapshot(state)
            .map_err(runtime_error)?
            .tensor()
            .clone())
    }

    fn restore_frontier(&mut self, step: u64, values: &BTreeMap<String, TensorData>) -> Result<()> {
        let buffers = self
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (parameter_key(name), *buffer))
            .chain(
                self.optimizer_buffers
                    .iter()
                    .map(|(name, buffer)| (name.clone(), *buffer)),
            )
            .collect::<BTreeMap<_, _>>();
        if values.len() != buffers.len() || values.keys().ne(buffers.keys()) {
            return Err(training("compiled checkpoint state names mismatch"));
        }

        let mut snapshots = Vec::with_capacity(buffers.len());
        for (name, buffer) in buffers {
            let value = &values[&name];
            let current = self.current_state(buffer)?;
            if value.shape() != &current.shape || value.dtype() != current.dtype {
                return Err(training("compiled checkpoint state descriptor mismatch"));
            }
            checked_bytes(value)?;
            let mut state = current.clone();
            state.version = step;
            snapshots.push((state, value.clone()));
        }

        let frontier = snapshots
            .iter()
            .map(|(state, _)| state.clone())
            .collect::<Vec<_>>();
        let cursor = MixedReplayCursor::resume(&self.capture, frontier).map_err(replay_error)?;
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_snapshots(snapshots)
            .map_err(runtime_error)?;
        self.runtime = runtime;
        self.cursor = cursor;
        self.step = step;
        Ok(())
    }

    fn snapshots(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, TensorData>> {
        buffers
            .iter()
            .map(|(name, buffer)| {
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((name.clone(), value))
            })
            .collect()
    }

    fn versions(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, u64>> {
        buffers
            .iter()
            .map(|(name, buffer)| Ok((name.clone(), self.current_state(*buffer)?.version)))
            .collect()
    }

    fn current_state(&self, buffer: u64) -> Result<&BufferState> {
        self.cursor
            .frontier()
            .iter()
            .find(|state| state.buffer == buffer)
            .ok_or_else(|| training("compiled persistent state is absent"))
    }
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
        Ok(Self {
            inner: CpuCompiledTrainingProgram::compile(
                MomentumProgram { config },
                parameters,
                build,
            )?,
        })
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner.step(inputs, learning_rate)
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
        self.inner.slot_snapshots("momentum")
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn momentum_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.slot_versions("momentum")
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner
            .step_inner(inputs, learning_rate, injected_failure)
    }
}

impl CpuCompiledAdamW {
    pub fn compile<F>(
        config: CompiledAdamWConfig,
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
        Ok(Self {
            inner: CpuCompiledTrainingProgram::compile(AdamWProgram { config }, parameters, build)?,
        })
    }

    /// Compiles an ordinary module forward against optimizer-owned recurrent
    /// parameter state.
    ///
    /// The builder receives only declared batch inputs: calls to
    /// [`crate::nn::Parameter::bind`] inside `module` resolve automatically to
    /// the compiled state frontier. Frozen parameters and buffers are captured
    /// as immutable constants, while tied parameter handles share one graph
    /// node and one AdamW state tuple.
    pub fn compile_module<M, F>(config: CompiledAdamWConfig, module: &M, build: F) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let plan = ModuleParameterPlan::new(module)?;
        let parameters = plan.initial_parameters()?;
        Self::compile(config, parameters, move |graph, inputs, parameters| {
            plan.lower(graph, parameters, |graph| build(module, graph, inputs))
        })
    }

    /// Recompiles an exact program and restores its saved recurrent frontier.
    /// The build/configuration must reproduce the checkpoint's capture
    /// identity; all state is validated before the fresh runtime is replaced.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let decoded = decode_adamw_checkpoint(checkpoint.as_bytes())?;
        let parameters = decoded
            .parameters
            .iter()
            .map(|(name, value)| TrainingParameterInit::new(name.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()?;
        let mut compiled = Self::compile(config, parameters, build)?;
        if compiled.capture_identity() != decoded.capture_identity {
            return Err(training(
                "compiled AdamW checkpoint capture identity mismatch",
            ));
        }
        let mut values = decoded
            .parameters
            .into_iter()
            .map(|(name, value)| (parameter_key(&name), value))
            .collect::<BTreeMap<_, _>>();
        for (name, value) in decoded.first_moments {
            values.insert(slot_key(&name, "first_moment"), value);
        }
        for (name, value) in decoded.second_moments {
            values.insert(slot_key(&name, "second_moment"), value);
        }
        values.insert(
            "global:step".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(decoded.step)])?,
        );
        compiled.inner.restore_frontier(decoded.step, &values)?;
        Ok(compiled)
    }

    /// Recompiles a module-bound program and restores its exact AdamW state.
    /// The module topology, frozen values, builder, and input descriptors must
    /// reproduce the authenticated capture identity before any restored state
    /// becomes visible.
    pub fn compile_module_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
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
        let plan = ModuleParameterPlan::new(module)?;
        Self::compile_from_checkpoint(config, checkpoint, move |graph, inputs, parameters| {
            plan.lower(graph, parameters, |graph| build(module, graph, inputs))
        })
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWStepResult> {
        self.inner.step(inputs, learning_rate)
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        Ok(self.inner.global_snapshot("step")?.scalar_at(0).as_u64())
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.slot_snapshots("first_moment")
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.slot_snapshots("second_moment")
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn first_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.slot_versions("first_moment")
    }

    pub fn second_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.slot_versions("second_moment")
    }

    /// Renders the identical loss/backward/AdamW capture for Metal, seeded
    /// from this session's currently committed recurrent state. Planning is
    /// resource-free; unsupported kernels fail before a device is touched.
    pub fn metal_plan(&self, renderer: MetalRenderer) -> Result<MetalCompiledAdamWPlan> {
        let recurrent = self.inner.recurrent_capture()?;
        let inner = MetalStatefulInferencePlan::new(recurrent.clone(), renderer.clone()).map_err(
            |error| {
                let detail = if matches!(&error, MetalError::Unsupported(_)) {
                    recurrent
                        .capture()
                        .items
                        .iter()
                        .find_map(|item| {
                            renderer.render(&item.kernel).err().map(|item_error| {
                                format!(
                                    " at schedule item {} (node {}, {:?}): {item_error}",
                                    item.id,
                                    item.node.index(),
                                    item.kernel.operation()
                                )
                            })
                        })
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                training(format!("compiled Metal runtime: {error:?}{detail}"))
            },
        )?;
        Ok(MetalCompiledAdamWPlan {
            inner,
            inputs: self.inner.inputs.clone(),
            output_names: self.inner.output_names.clone(),
            state_input_keys: self.inner.state_input_keys.clone(),
            program_identity: self.capture_identity(),
            step: self.step_count(),
        })
    }

    /// Captures parameter values, both moment sets, the graph-owned optimizer
    /// step, and the exact compiled capture identity into deterministic bytes.
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        let step = self.optimizer_step()?;
        if step != self.step_count() {
            return Err(training("compiled AdamW host and graph steps diverged"));
        }
        let bytes = encode_adamw_checkpoint(
            self.capture_identity(),
            step,
            self.parameter_snapshots()?,
            self.first_moment_snapshots()?,
            self.second_moment_snapshots()?,
        )?;
        CompiledAdamWCheckpoint::from_bytes(bytes)
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        self.inner
            .step_inner(inputs, learning_rate, injected_failure)
    }
}

impl MetalCompiledAdamWPlan {
    pub fn deployment_identity(&self) -> u64 {
        self.inner.deployment_identity()
    }

    pub fn capture_identity(&self) -> u64 {
        self.program_identity
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    /// Creates all native resources and uploads the captured recurrent
    /// frontier once. No training step is executed during preparation.
    pub fn prepare(self, device: MetalDevice) -> Result<MetalCompiledAdamW> {
        let session = self.inner.prepare(device).map_err(metal_training_error)?;
        Ok(MetalCompiledAdamW {
            session,
            inputs: self.inputs,
            output_names: self.output_names,
            state_input_keys: self.state_input_keys,
            program_identity: self.program_identity,
            step: self.step,
        })
    }
}

impl MetalCompiledAdamW {
    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWStepResult> {
        validate_step_inputs(&self.inputs, &inputs, &learning_rate)?;
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| training("compiled training step overflow"))?;
        let mut provided = inputs;
        provided.insert(LEARNING_RATE_INPUT.into(), learning_rate);
        let run = self.session.run(&provided).map_err(metal_training_error)?;
        let (outputs, report) = run.into_parts();
        debug_assert_eq!(outputs.len(), 1 + self.output_names.len());
        let mut outputs = outputs.into_iter();
        let loss = outputs
            .next()
            .expect("compiled Metal output cardinality was authenticated before preparation");
        let outputs = self.output_names.iter().cloned().zip(outputs).collect();
        self.step = next_step;
        Ok(MetalCompiledAdamWStepResult {
            inner: CompiledTrainingStepResult {
                loss,
                outputs,
                step: self.step,
                capture_identity: self.program_identity,
            },
            report,
        })
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn capture_identity(&self) -> u64 {
        self.program_identity
    }

    pub fn metal_session(&self) -> &MetalDeviceSession {
        &self.session
    }

    /// Downloads every currently committed recurrent value once and returns
    /// it under the optimizer's semantic state keys.
    fn state_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        let snapshots = self
            .session
            .state_snapshots()
            .map_err(metal_training_error)?;
        if snapshots.len() != self.state_input_keys.len()
            || snapshots.keys().ne(self.state_input_keys.keys())
        {
            return Err(training("compiled Metal state inventory mismatch"));
        }
        snapshots
            .into_iter()
            .map(|(input, value)| {
                let key = self
                    .state_input_keys
                    .get(&input)
                    .cloned()
                    .ok_or_else(|| training("compiled Metal state key is absent"))?;
                Ok((key, value))
            })
            .collect()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        Ok(self
            .state_snapshots()?
            .into_iter()
            .filter_map(|(key, value)| {
                key.strip_prefix("parameter:")
                    .map(|name| (name.to_owned(), value))
            })
            .collect())
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_slot_snapshots(self.state_snapshots()?, "first_moment")
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_slot_snapshots(self.state_snapshots()?, "second_moment")
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        let states = self.state_snapshots()?;
        let step = states
            .get("global:step")
            .ok_or_else(|| training("compiled Metal optimizer step is absent"))?;
        Ok(step.scalar_at(0).as_u64())
    }

    /// Downloads one coherent active state bank and encodes the same portable
    /// checkpoint format accepted by [`CpuCompiledAdamW::compile_from_checkpoint`].
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        let states = self.state_snapshots()?;
        let optimizer_step = states
            .get("global:step")
            .ok_or_else(|| training("compiled Metal optimizer step is absent"))?
            .scalar_at(0)
            .as_u64();
        if optimizer_step != self.step {
            return Err(training("compiled Metal host and graph steps diverged"));
        }
        let parameters = metal_parameter_snapshots(&states);
        let first_moments = metal_slot_snapshots(states.clone(), "first_moment")?;
        let second_moments = metal_slot_snapshots(states, "second_moment")?;
        CompiledAdamWCheckpoint::from_bytes(encode_adamw_checkpoint(
            self.program_identity,
            optimizer_step,
            parameters,
            first_moments,
            second_moments,
        )?)
    }
}

fn metal_parameter_snapshots(
    states: &BTreeMap<String, TensorData>,
) -> BTreeMap<String, TensorData> {
    states
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("parameter:")
                .map(|name| (name.to_owned(), value.clone()))
        })
        .collect()
}

fn metal_slot_snapshots(
    states: BTreeMap<String, TensorData>,
    slot: &str,
) -> Result<BTreeMap<String, TensorData>> {
    let suffix = format!(":{slot}");
    Ok(states
        .into_iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("slot:")
                .and_then(|key| key.strip_suffix(&suffix))
                .map(|name| (name.to_owned(), value))
        })
        .collect())
}

fn encode_adamw_checkpoint(
    capture_identity: u64,
    step: u64,
    parameters: BTreeMap<String, TensorData>,
    first_moments: BTreeMap<String, TensorData>,
    second_moments: BTreeMap<String, TensorData>,
) -> Result<Vec<u8>> {
    validate_adamw_checkpoint_maps(&parameters, &first_moments, &second_moments)?;
    let names = parameters.keys().cloned().collect::<Vec<_>>();
    let mut tensors = StateDict::default();
    for (ordinal, name) in names.iter().enumerate() {
        tensors.insert(format!("parameter.{ordinal}"), parameters[name].clone());
        tensors.insert(
            format!("first_moment.{ordinal}"),
            first_moments[name].clone(),
        );
        tensors.insert(
            format!("second_moment.{ordinal}"),
            second_moments[name].clone(),
        );
    }
    let metadata = Metadata::from([
        ("format".into(), ADAMW_CHECKPOINT_FORMAT.into()),
        ("capture_identity".into(), capture_identity.to_string()),
        ("step".into(), step.to_string()),
        (
            "parameter_names".into(),
            serde_json::to_string(&names)
                .map_err(|error| training(format!("checkpoint names: {error}")))?,
        ),
    ]);
    save_safetensors(&tensors, &metadata)
}

fn decode_adamw_checkpoint(bytes: &[u8]) -> Result<DecodedAdamWCheckpoint> {
    let (state, metadata) = load_safetensors(bytes)?;
    let expected_metadata =
        BTreeSet::from(["format", "capture_identity", "step", "parameter_names"]);
    if metadata.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_metadata {
        return Err(training("compiled AdamW checkpoint metadata mismatch"));
    }
    if metadata["format"] != ADAMW_CHECKPOINT_FORMAT {
        return Err(training("compiled AdamW checkpoint format mismatch"));
    }
    let capture_identity = metadata["capture_identity"]
        .parse::<u64>()
        .map_err(|_| training("compiled AdamW checkpoint capture identity is invalid"))?;
    let step = metadata["step"]
        .parse::<u64>()
        .map_err(|_| training("compiled AdamW checkpoint step is invalid"))?;
    let names = serde_json::from_str::<Vec<String>>(&metadata["parameter_names"])
        .map_err(|_| training("compiled AdamW checkpoint parameter names are invalid"))?;
    if names.is_empty() {
        return Err(training("compiled AdamW checkpoint has no parameters"));
    }
    let mut unique = BTreeSet::new();
    for name in &names {
        validate_user_name(name, "checkpoint parameter")?;
        if !unique.insert(name.clone()) {
            return Err(training("compiled AdamW checkpoint parameter names repeat"));
        }
    }

    let mut tensors = state;
    let mut parameters = BTreeMap::new();
    let mut first_moments = BTreeMap::new();
    let mut second_moments = BTreeMap::new();
    for (ordinal, name) in names.into_iter().enumerate() {
        let parameter = tensors
            .remove(&format!("parameter.{ordinal}"))
            .ok_or_else(|| training("compiled AdamW checkpoint parameter is absent"))?;
        let first = tensors
            .remove(&format!("first_moment.{ordinal}"))
            .ok_or_else(|| training("compiled AdamW checkpoint first moment is absent"))?;
        let second = tensors
            .remove(&format!("second_moment.{ordinal}"))
            .ok_or_else(|| training("compiled AdamW checkpoint second moment is absent"))?;
        parameters.insert(name.clone(), parameter);
        first_moments.insert(name.clone(), first);
        second_moments.insert(name, second);
    }
    if !tensors.is_empty() {
        return Err(training("compiled AdamW checkpoint tensor set mismatch"));
    }
    validate_adamw_checkpoint_maps(&parameters, &first_moments, &second_moments)?;
    Ok(DecodedAdamWCheckpoint {
        capture_identity,
        step,
        parameters,
        first_moments,
        second_moments,
    })
}

fn validate_adamw_checkpoint_maps(
    parameters: &BTreeMap<String, TensorData>,
    first_moments: &BTreeMap<String, TensorData>,
    second_moments: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if parameters.is_empty()
        || parameters.keys().ne(first_moments.keys())
        || parameters.keys().ne(second_moments.keys())
    {
        return Err(training("compiled AdamW checkpoint state names mismatch"));
    }
    for (name, parameter) in parameters {
        validate_user_name(name, "checkpoint parameter")?;
        let first = &first_moments[name];
        let second = &second_moments[name];
        if parameter.dtype() != DType::F32
            || first.dtype() != DType::F32
            || second.dtype() != DType::F32
            || first.shape() != parameter.shape()
            || second.shape() != parameter.shape()
        {
            return Err(training("compiled AdamW checkpoint descriptor mismatch"));
        }
        checked_bytes(parameter)?;
        checked_bytes(first)?;
        checked_bytes(second)?;
    }
    Ok(())
}

fn canonical_parameters(
    parameters: impl IntoIterator<Item = TrainingParameterInit>,
) -> Result<BTreeMap<String, TensorData>> {
    let mut values = BTreeMap::new();
    for parameter in parameters {
        validate_user_name(&parameter.name, "parameter")?;
        if parameter.value.dtype() != DType::F32 {
            return Err(training("compiled training parameters must be F32"));
        }
        checked_bytes(&parameter.value)?;
        if values.insert(parameter.name, parameter.value).is_some() {
            return Err(training("duplicate compiled parameter name"));
        }
    }
    Ok(values)
}

fn validate_user_name(name: &str, kind: &str) -> Result<()> {
    if name.is_empty() || name == "loss" || name.starts_with(INTERNAL_PREFIX) {
        return Err(training(format!("invalid compiled {kind} name")));
    }
    Ok(())
}

fn checked_bytes(value: &TensorData) -> Result<usize> {
    value
        .len()
        .checked_mul(value.dtype().itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

fn checked_descriptor(shape: &Shape, dtype: DType) -> Result<usize> {
    shape
        .numel()
        .map_err(|_| training("compiled tensor element extent overflow"))?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

fn state_for(buffer: u64, value: &TensorData) -> Result<BufferState> {
    Ok(BufferState {
        buffer,
        version: 0,
        shape: value.shape().clone(),
        dtype: value.dtype(),
        bytes: checked_bytes(value)?,
    })
}

fn validate_loss(graph: &Graph, loss: NodeId) -> Result<()> {
    if graph.dtype(loss)? != DType::F32 || graph.shape(loss)? != &Shape::from([]) {
        return Err(training(
            "compiled training loss must be a rank-zero F32 scalar",
        ));
    }
    Ok(())
}

fn validate_outputs<'a>(
    loss: NodeId,
    outputs: &BTreeMap<String, NodeId>,
    reserved_user_names: impl IntoIterator<Item = &'a String>,
) -> Result<()> {
    let mut nodes = BTreeSet::from([loss]);
    let reserved_user_names = reserved_user_names
        .into_iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for (name, node) in outputs {
        validate_user_name(name, "output")?;
        if name == "loss" || reserved_user_names.contains(name.as_str()) {
            return Err(training(
                "compiled output name collides with another user name",
            ));
        }
        if !nodes.insert(*node) {
            return Err(training("duplicate compiled output node"));
        }
    }
    Ok(())
}

fn validate_external_binding_ownership<'a>(
    capture: &CapturedMixedSchedule,
    configured_inputs: impl IntoIterator<Item = &'a String>,
) -> Result<()> {
    let mut external = configured_inputs
        .into_iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    external.insert(LEARNING_RATE_INPUT);
    for binding in &capture.state_bindings {
        let input = capture
            .schedule
            .inputs
            .iter()
            .find(|input| input.node == binding.input_node)
            .ok_or_else(|| training("compiled state input ABI is absent"))?;
        if external.contains(input.name.as_str()) {
            return Err(training(
                "compiled external input shadows persistent state binding",
            ));
        }
    }
    Ok(())
}

fn collect_state_bindings(
    schedule: &Schedule,
    states: &BTreeMap<NodeId, BufferState>,
) -> Result<Vec<ScheduleStateBinding>> {
    let mut bindings = Vec::new();
    let mut seen = BTreeSet::new();
    let mut bound_nodes = BTreeSet::new();
    for item in &schedule.items {
        for binding in &item.input_bindings {
            let Some(state) = states.get(&binding.input_node) else {
                continue;
            };
            if !seen.insert((item.id, binding.input_node)) {
                return Err(training("duplicate compiled state input binding"));
            }
            bound_nodes.insert(binding.input_node);
            bindings.push(ScheduleStateBinding {
                state: state.clone(),
                view: None,
                consumer_item: item.id,
                consumer_node: item.node,
                input_node: binding.input_node,
                desc: binding.desc.clone(),
                abi_index: binding.abi_index,
            });
        }
    }
    if bindings.is_empty() || states.keys().any(|node| !bound_nodes.contains(node)) {
        return Err(training("compiled state input is not reachable"));
    }
    Ok(bindings)
}

fn value_binding(
    schedule: &Schedule,
    node: NodeId,
    effect_item: u64,
) -> Result<ScheduleValueBinding> {
    let (producer_item, producer) = schedule
        .items
        .iter()
        .enumerate()
        .find(|(_, item)| item.primary_output().id == node.index() as u64)
        .ok_or_else(|| training("compiled update output is not materialized"))?;
    Ok(ScheduleValueBinding {
        producer_item: u64::try_from(producer_item)
            .map_err(|_| training("compiled producer index overflow"))?,
        producer_node: node,
        producer_output: producer.primary_output().clone(),
        abi_index: 0,
        effect_item,
        source_position: 0,
    })
}

fn effect_states(effects: &EffectGraph) -> Result<Vec<BufferState>> {
    let plan = effects.plan();
    plan.validate().map_err(effect_error)?;
    let mut states = BTreeMap::new();
    for step in plan.steps {
        for state in step.reads.into_iter().chain([step.write]) {
            states.insert((state.buffer, state.version), state);
        }
    }
    Ok(states.into_values().collect())
}

fn validate_step_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    actual: &BTreeMap<String, TensorData>,
    learning_rate: &TensorData,
) -> Result<()> {
    if actual.len() != expected.len() || actual.keys().ne(expected.keys()) {
        return Err(training("compiled training input names do not match"));
    }
    for (name, value) in actual {
        let (shape, dtype) = &expected[name];
        if value.shape() != shape || value.dtype() != *dtype {
            return Err(training("compiled training input descriptor mismatch"));
        }
        checked_bytes(value)?;
    }
    if learning_rate.shape() != &Shape::from([]) || learning_rate.dtype() != DType::F32 {
        return Err(training(
            "compiled training learning rate must be rank-zero F32",
        ));
    }
    checked_bytes(learning_rate)?;
    Ok(())
}

fn schedule_error(error: impl std::fmt::Display) -> Error {
    training(format!("compiled schedule: {error}"))
}

fn replay_error(error: ReplayError) -> Error {
    training(format!("compiled replay: {error:?}"))
}

fn captured_inference_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled recurrent capture: {error:?}"))
}

fn metal_training_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled Metal runtime: {error:?}"))
}

fn runtime_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled persistent runtime: {error:?}"))
}

fn effect_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled effect graph: {error:?}"))
}

fn training(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, LossOptions, Parameter, cross_entropy};
    use std::collections::HashMap;

    struct TiedFrozenModule {
        shared: Parameter,
        frozen: Parameter,
    }

    impl TiedFrozenModule {
        fn new(frozen: [f32; 2]) -> Self {
            Self {
                shared: Parameter::new(TensorData::new([2], vec![0.25, -0.5]).unwrap(), true),
                frozen: Parameter::new(TensorData::new([2], frozen.to_vec()).unwrap(), false),
            }
        }
    }

    impl Module for TiedFrozenModule {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            assert!(prefix.is_empty());
            visitor("shared".into(), &self.shared, StateKind::Parameter);
            visitor("shared_alias".into(), &self.shared, StateKind::Parameter);
            visitor("frozen".into(), &self.frozen, StateKind::Parameter);
        }
    }

    fn module_config() -> CompiledAdamWConfig {
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_input("x", [2], DType::F32)
            .unwrap()
    }

    fn build_tied_frozen(
        module: &TiedFrozenModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let shared = module.shared.bind(graph)?;
        assert_eq!(module.shared.bind(graph)?, shared);
        assert_eq!(module.shared.node(graph)?, shared);
        let frozen = module.frozen.bind(graph)?;
        assert!(!graph.requires_grad(frozen)?);
        let scaled = graph.mul(inputs["x"], shared)?;
        let tied = graph.add(scaled, shared)?;
        let output = graph.add(tied, frozen)?;
        let squared = graph.square(output)?;
        let loss = graph.reduce(squared, crate::ReduceKind::Mean, None, false)?;
        Ok((loss, BTreeMap::from([("output".into(), output)])))
    }

    fn initial_parameters() -> Vec<TrainingParameterInit> {
        vec![
            TrainingParameterInit::new(
                "w1",
                TensorData::new(
                    [2, 4],
                    vec![0.20, -0.10, 0.05, 0.30, -0.25, 0.15, 0.40, -0.20],
                )
                .unwrap(),
            )
            .unwrap(),
            TrainingParameterInit::new(
                "w2",
                TensorData::new(
                    [4, 2],
                    vec![0.10, -0.20, 0.30, 0.05, -0.15, 0.25, 0.20, -0.10],
                )
                .unwrap(),
            )
            .unwrap(),
        ]
    }

    fn config() -> CompiledMomentumSgdConfig {
        CompiledMomentumSgdConfig::new(0.9)
            .unwrap()
            .with_input("x", [4, 2], DType::F32)
            .unwrap()
            .with_input("target", [4], DType::I64)
            .unwrap()
    }

    fn build_tinybob(
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        parameters: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let hidden = graph.matmul(inputs["x"], parameters["w1"])?;
        let hidden = graph.relu(hidden)?;
        let logits = graph.matmul(hidden, parameters["w2"])?;
        let loss = cross_entropy(graph, logits, inputs["target"], LossOptions::default())?;
        Ok((loss, BTreeMap::from([("logits".into(), logits)])))
    }

    fn compiled() -> CpuCompiledMomentumSgd {
        CpuCompiledMomentumSgd::compile(config(), initial_parameters(), build_tinybob).unwrap()
    }

    fn adamw_config() -> CompiledAdamWConfig {
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)
            .unwrap()
            .with_input("x", [4, 2], DType::F32)
            .unwrap()
            .with_input("target", [4], DType::I64)
            .unwrap()
    }

    fn compiled_adamw() -> CpuCompiledAdamW {
        CpuCompiledAdamW::compile(adamw_config(), initial_parameters(), build_tinybob).unwrap()
    }

    fn batch() -> BTreeMap<String, TensorData> {
        BTreeMap::from([
            (
                "target".into(),
                TensorData::from_scalars(
                    [4],
                    DType::I64,
                    [Scalar::I(0), Scalar::I(1), Scalar::I(1), Scalar::I(0)],
                )
                .unwrap(),
            ),
            (
                "x".into(),
                TensorData::new([4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5]).unwrap(),
            ),
        ])
    }

    fn lr() -> TensorData {
        TensorData::scalar(0.05)
    }

    fn fresh_oracle_step(
        parameters: &BTreeMap<String, TensorData>,
        momentum: &BTreeMap<String, TensorData>,
    ) -> (
        TensorData,
        TensorData,
        BTreeMap<String, TensorData>,
        BTreeMap<String, TensorData>,
    ) {
        let values = batch();
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [4, 2], DType::F32);
        let target = graph.input_dtype("target", [4], DType::I64);
        let learning_rate = graph.input_dtype("lr", Shape::from([]), DType::F32);
        let mut parameter_nodes = BTreeMap::new();
        let mut momentum_nodes = BTreeMap::new();
        let mut bindings = HashMap::from([
            ("x".into(), values["x"].clone()),
            ("target".into(), values["target"].clone()),
            ("lr".into(), lr()),
        ]);
        for name in ["w1", "w2"] {
            let parameter_name = format!("parameter_{name}");
            let momentum_name = format!("momentum_{name}");
            let parameter = graph.input_dtype(
                parameter_name.clone(),
                parameters[name].shape().clone(),
                DType::F32,
            );
            let velocity = graph.input_dtype(
                momentum_name.clone(),
                momentum[name].shape().clone(),
                DType::F32,
            );
            parameter_nodes.insert(name.to_string(), parameter);
            momentum_nodes.insert(name.to_string(), velocity);
            bindings.insert(parameter_name, parameters[name].clone());
            bindings.insert(momentum_name, momentum[name].clone());
        }
        let hidden = graph.matmul(x, parameter_nodes["w1"]).unwrap();
        let hidden = graph.relu(hidden).unwrap();
        let logits = graph.matmul(hidden, parameter_nodes["w2"]).unwrap();
        let loss = cross_entropy(&mut graph, logits, target, LossOptions::default()).unwrap();
        let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
        let gradients = graph.gradient_default(loss, &targets).unwrap();
        let retained = graph
            .full_with_dtype(Shape::from([]), Scalar::F(0.9), DType::F32)
            .unwrap();
        let mut next_parameters = BTreeMap::new();
        let mut next_momentum = BTreeMap::new();
        let mut update_nodes = Vec::new();
        for ((name, parameter), gradient) in parameter_nodes.iter().zip(gradients) {
            let velocity = momentum_nodes[name];
            let velocity = graph
                .mul(retained, velocity)
                .and_then(|value| graph.add(value, gradient))
                .unwrap();
            let update = graph.mul(learning_rate, velocity).unwrap();
            let parameter = graph.sub(*parameter, update).unwrap();
            update_nodes.push((name.clone(), velocity, parameter));
        }
        let cpu = CpuBackend;
        for (name, velocity, parameter) in update_nodes {
            next_momentum.insert(
                name.clone(),
                cpu.execute(&graph, velocity, &bindings).unwrap(),
            );
            next_parameters.insert(name, cpu.execute(&graph, parameter, &bindings).unwrap());
        }
        (
            cpu.execute(&graph, loss, &bindings).unwrap(),
            cpu.execute(&graph, logits, &bindings).unwrap(),
            next_parameters,
            next_momentum,
        )
    }

    #[test]
    fn tinybob_three_step_compiled_replay_matches_fresh_cpu_training() {
        let mut compiled = compiled();
        let identity = compiled.capture_identity();
        let mut parameters = compiled.parameter_snapshots().unwrap();
        let mut momentum = compiled.momentum_snapshots().unwrap();
        for step in 1..=3 {
            let (loss, logits, next_parameters, next_momentum) =
                fresh_oracle_step(&parameters, &momentum);
            let result = compiled.step(batch(), lr()).unwrap();
            assert_eq!(result.loss().storage(), loss.storage());
            assert_eq!(result.output("logits").unwrap().storage(), logits.storage());
            assert_eq!(result.step(), step);
            assert_eq!(result.capture_identity(), identity);
            assert_eq!(compiled.parameter_snapshots().unwrap(), next_parameters);
            assert_eq!(compiled.momentum_snapshots().unwrap(), next_momentum);
            assert_eq!(
                compiled.parameter_versions().unwrap(),
                BTreeMap::from([("w1".into(), step), ("w2".into(), step)])
            );
            assert_eq!(
                compiled.momentum_versions().unwrap(),
                BTreeMap::from([("w1".into(), step), ("w2".into(), step)])
            );
            parameters = next_parameters;
            momentum = next_momentum;
        }
    }

    #[test]
    fn compile_identity_is_stable_and_initial_values_are_detached() {
        let original = initial_parameters();
        let before = original
            .iter()
            .map(|parameter| (parameter.name().to_string(), parameter.value().clone()))
            .collect::<BTreeMap<_, _>>();
        let first = CpuCompiledMomentumSgd::compile(config(), original, build_tinybob).unwrap();
        let second = compiled();
        assert_eq!(first.capture_identity(), second.capture_identity());
        assert_eq!(first.parameter_snapshots().unwrap(), before);
        let mut detached = first.parameter_snapshots().unwrap();
        detached
            .get_mut("w1")
            .unwrap()
            .assign(&TensorData::zeros([2, 4]).unwrap())
            .unwrap();
        assert_ne!(detached, first.parameter_snapshots().unwrap());
    }

    #[test]
    fn step_inputs_exclude_every_persistent_state_binding() {
        let compiled = compiled();
        let external = compiled
            .inner
            .inputs
            .keys()
            .map(String::as_str)
            .chain([LEARNING_RATE_INPUT])
            .collect::<BTreeSet<_>>();
        assert_eq!(
            external,
            BTreeSet::from(["target", "x", LEARNING_RATE_INPUT])
        );
        let persistent = compiled
            .inner
            .capture
            .state_bindings
            .iter()
            .map(|binding| {
                compiled
                    .inner
                    .capture
                    .schedule
                    .inputs
                    .iter()
                    .find(|input| input.node == binding.input_node)
                    .unwrap()
                    .name
                    .as_str()
            })
            .collect::<BTreeSet<_>>();
        assert!(!persistent.is_empty());
        assert!(
            persistent
                .iter()
                .all(|name| name.starts_with(INTERNAL_PREFIX))
        );
        assert!(external.is_disjoint(&persistent));
        let mut consumer_views = BTreeMap::<NodeId, BTreeSet<bool>>::new();
        for binding in &compiled.inner.capture.state_bindings {
            consumer_views
                .entry(binding.input_node)
                .or_default()
                .insert(binding.desc.view.is_some());
        }
        assert!(
            consumer_views
                .values()
                .any(|views| views == &BTreeSet::from([false, true]))
        );
        let pure_items = compiled
            .inner
            .capture
            .schedule
            .items
            .iter()
            .take_while(|item| !item.is_effect())
            .collect::<Vec<_>>();
        assert!(!pure_items.is_empty());
        assert!(pure_items.iter().all(|item| item.boundary.is_none()));
    }

    #[test]
    fn malformed_and_duplicate_inputs_fail_before_state_publication() {
        let duplicate = TrainingParameterInit::new(
            "w1",
            TensorData::zeros_with_dtype([2, 4], DType::F32).unwrap(),
        )
        .unwrap();
        assert!(
            CpuCompiledMomentumSgd::compile(
                config(),
                initial_parameters().into_iter().chain([duplicate]),
                build_tinybob,
            )
            .is_err()
        );
        assert!(
            CompiledMomentumSgdConfig::new(0.9)
                .unwrap()
                .with_input(INTERNAL_PREFIX, [1], DType::F32)
                .is_err()
        );
        assert!(
            CompiledMomentumSgdConfig::new(0.9)
                .unwrap()
                .with_input("x", [1], DType::F32)
                .unwrap()
                .with_input("x", [1], DType::F32)
                .is_err()
        );
        assert!(
            TrainingParameterInit::new(
                "bad",
                TensorData::zeros_with_dtype([1], DType::I32).unwrap(),
            )
            .is_err()
        );

        let mut compiled = compiled();
        let before = compiled.parameter_snapshots().unwrap();
        let mut missing = batch();
        missing.remove("target");
        assert!(compiled.step(missing, lr()).is_err());
        assert!(
            compiled
                .step(batch(), TensorData::new([1], vec![0.05]).unwrap())
                .is_err()
        );
        assert_eq!(compiled.step_count(), 0);
        assert_eq!(compiled.parameter_snapshots().unwrap(), before);
    }

    #[test]
    fn cross_namespace_names_reject_before_the_private_graph_builder_runs() {
        let invoked = std::cell::Cell::new(false);
        let conflicting = CompiledMomentumSgdConfig::new(0.9)
            .unwrap()
            .with_input("w1", [4, 2], DType::F32)
            .unwrap();
        let result =
            CpuCompiledMomentumSgd::compile(conflicting, initial_parameters(), |_, _, _| {
                invoked.set(true);
                Err(training("builder should not run"))
            });
        assert!(result.is_err());
        assert!(!invoked.get());
    }

    #[test]
    fn injected_and_stale_replay_failures_preserve_runtime_cursor_and_step() {
        let mut compiled = compiled();
        let initial_parameters = compiled.parameter_snapshots().unwrap();
        let initial_momentum = compiled.momentum_snapshots().unwrap();
        let initial_cursor = compiled.inner.cursor.clone();
        assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
        assert_eq!(compiled.step_count(), 0);
        assert_eq!(compiled.inner.cursor, initial_cursor);
        assert_eq!(compiled.parameter_snapshots().unwrap(), initial_parameters);
        assert_eq!(compiled.momentum_snapshots().unwrap(), initial_momentum);

        compiled.step(batch(), lr()).unwrap();
        let advanced = compiled.inner.cursor.clone();
        let advanced_parameters = compiled.parameter_snapshots().unwrap();
        compiled.inner.cursor = initial_cursor;
        assert!(compiled.step(batch(), lr()).is_err());
        assert_eq!(compiled.step_count(), 1);
        compiled.inner.cursor = advanced;
        assert_eq!(compiled.parameter_snapshots().unwrap(), advanced_parameters);
    }

    #[test]
    fn adamw_replays_one_capture_with_graph_owned_state() {
        let mut compiled = compiled_adamw();
        let identity = compiled.capture_identity();
        let zeros = initial_parameters()
            .into_iter()
            .map(|parameter| {
                (
                    parameter.name().to_string(),
                    TensorData::zeros_with_dtype(parameter.value().shape().clone(), DType::F32)
                        .unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(compiled.optimizer_step().unwrap(), 0);
        assert_eq!(compiled.first_moment_snapshots().unwrap(), zeros);
        assert_eq!(compiled.second_moment_snapshots().unwrap(), zeros);

        let first = compiled.step(batch(), lr()).unwrap();
        assert_eq!(first.step(), 1);
        assert_eq!(first.capture_identity(), identity);
        assert_eq!(compiled.optimizer_step().unwrap(), 1);
        assert_eq!(
            compiled
                .parameter_versions()
                .unwrap()
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            compiled
                .first_moment_versions()
                .unwrap()
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            compiled
                .second_moment_versions()
                .unwrap()
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_ne!(compiled.first_moment_snapshots().unwrap(), zeros);
        assert_ne!(compiled.second_moment_snapshots().unwrap(), zeros);

        let second = compiled.step(batch(), lr()).unwrap();
        assert_eq!(second.step(), 2);
        assert_eq!(second.capture_identity(), identity);
        assert_eq!(compiled.optimizer_step().unwrap(), 2);
    }

    #[test]
    fn adamw_failure_preserves_parameters_moments_step_and_cursor() {
        let mut compiled = compiled_adamw();
        let parameters = compiled.parameter_snapshots().unwrap();
        let first = compiled.first_moment_snapshots().unwrap();
        let second = compiled.second_moment_snapshots().unwrap();
        let cursor = compiled.inner.cursor.clone();

        assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
        assert_eq!(compiled.step_count(), 0);
        assert_eq!(compiled.optimizer_step().unwrap(), 0);
        assert_eq!(compiled.inner.cursor, cursor);
        assert_eq!(compiled.parameter_snapshots().unwrap(), parameters);
        assert_eq!(compiled.first_moment_snapshots().unwrap(), first);
        assert_eq!(compiled.second_moment_snapshots().unwrap(), second);
    }

    #[test]
    fn adamw_module_binding_owns_trainable_state_and_preserves_ties_and_freezing() {
        let module = TiedFrozenModule::new([1.0, -1.0]);
        let mut compiled =
            CpuCompiledAdamW::compile_module(module_config(), &module, build_tied_frozen).unwrap();
        assert_eq!(
            compiled
                .parameter_snapshots()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["shared"]
        );

        // Host state is only an initialization source. Once compiled, the
        // recurrent runtime is the sole owner of the trainable value.
        module
            .shared
            .replace(TensorData::new([2], vec![9.0, 9.0]).unwrap())
            .unwrap();
        let input = BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
        let first = compiled
            .step(input.clone(), TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(first.step(), 1);
        assert_eq!(compiled.optimizer_step().unwrap(), 1);
        assert_eq!(compiled.parameter_versions().unwrap()["shared"], 1);

        let checkpoint = compiled.checkpoint().unwrap();
        let resumed_module = TiedFrozenModule::new([1.0, -1.0]);
        let mut resumed = CpuCompiledAdamW::compile_module_from_checkpoint(
            module_config(),
            &resumed_module,
            &checkpoint,
            build_tied_frozen,
        )
        .unwrap();
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            compiled.parameter_snapshots().unwrap()
        );
        assert_eq!(
            resumed.first_moment_snapshots().unwrap(),
            compiled.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.second_moment_snapshots().unwrap(),
            compiled.second_moment_snapshots().unwrap()
        );
        assert_eq!(resumed.step_count(), 1);
        assert_eq!(
            resumed
                .step(input, TensorData::scalar(0.01))
                .unwrap()
                .step(),
            2
        );

        let changed_frozen = TiedFrozenModule::new([2.0, -1.0]);
        assert!(
            CpuCompiledAdamW::compile_module_from_checkpoint(
                module_config(),
                &changed_frozen,
                &checkpoint,
                build_tied_frozen,
            )
            .is_err()
        );
    }

    #[test]
    fn adamw_config_rejects_invalid_hyperparameters_before_build() {
        for config in [
            CompiledAdamWConfig::new(-0.1, 0.999, 1e-8, 0.0),
            CompiledAdamWConfig::new(1.0, 0.999, 1e-8, 0.0),
            CompiledAdamWConfig::new(0.9, 1.0, 1e-8, 0.0),
            CompiledAdamWConfig::new(0.9, 0.999, 0.0, 0.0),
            CompiledAdamWConfig::new(0.9, 0.999, f32::NAN, 0.0),
            CompiledAdamWConfig::new(0.9, 0.999, 1e-8, -0.1),
        ] {
            assert!(config.is_err());
        }
    }

    #[test]
    fn adamw_checkpoint_resume_matches_uninterrupted_replay_exactly() {
        let mut uninterrupted = compiled_adamw();
        let mut saved = compiled_adamw();
        for _ in 0..2 {
            uninterrupted.step(batch(), lr()).unwrap();
            saved.step(batch(), lr()).unwrap();
        }
        let checkpoint = saved.checkpoint().unwrap();
        assert_eq!(checkpoint, saved.checkpoint().unwrap());
        assert_eq!(
            CompiledAdamWCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
            checkpoint
        );

        let mut resumed =
            CpuCompiledAdamW::compile_from_checkpoint(adamw_config(), &checkpoint, build_tinybob)
                .unwrap();
        assert_eq!(resumed.step_count(), 2);
        assert_eq!(resumed.optimizer_step().unwrap(), 2);
        assert_eq!(resumed.capture_identity(), saved.capture_identity());
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            saved.parameter_snapshots().unwrap()
        );
        assert_eq!(
            resumed.first_moment_snapshots().unwrap(),
            saved.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.second_moment_snapshots().unwrap(),
            saved.second_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.parameter_versions().unwrap(),
            saved.parameter_versions().unwrap()
        );
        assert_eq!(
            resumed.first_moment_versions().unwrap(),
            saved.first_moment_versions().unwrap()
        );
        assert_eq!(
            resumed.second_moment_versions().unwrap(),
            saved.second_moment_versions().unwrap()
        );

        let expected = uninterrupted.step(batch(), lr()).unwrap();
        let actual = resumed.step(batch(), lr()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            uninterrupted.parameter_snapshots().unwrap()
        );
        assert_eq!(
            resumed.first_moment_snapshots().unwrap(),
            uninterrupted.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.second_moment_snapshots().unwrap(),
            uninterrupted.second_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.parameter_versions().unwrap(),
            uninterrupted.parameter_versions().unwrap()
        );
    }

    #[test]
    fn adamw_checkpoint_rejects_corruption_and_wrong_program_identity() {
        let checkpoint = compiled_adamw().checkpoint().unwrap();
        let mut corrupt = checkpoint.as_bytes().to_vec();
        corrupt.pop();
        assert!(CompiledAdamWCheckpoint::from_bytes(corrupt).is_err());

        let wrong = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.02)
            .unwrap()
            .with_input("x", [4, 2], DType::F32)
            .unwrap()
            .with_input("target", [4], DType::I64)
            .unwrap();
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(wrong, &checkpoint, build_tinybob).is_err()
        );
    }
}
