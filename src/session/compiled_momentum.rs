//! Bounded graph-free CPU replay for static momentum-SGD training.

use crate::{
    BufferState, CapturedMixedSchedule, CapturedSchedule, DType, EffectGraph, EffectRuntime, Error,
    Graph, MixedReplayCursor, NodeId, ReplayError, Result, Scalar, Schedule, ScheduleStateBinding,
    ScheduleValueBinding, Shape, TensorData, bind_schedule_states, combine_mixed_schedules,
    schedule_effects, schedule_many,
};
use std::collections::{BTreeMap, BTreeSet};

const INTERNAL_PREFIX: &str = "__rustgrad_compiled_sgd_";
const LEARNING_RATE_INPUT: &str = "__rustgrad_compiled_sgd_learning_rate";
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
            return Err(training("compiled momentum-SGD parameters must be F32"));
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

/// Detached result of one successfully committed compiled training step.
#[derive(Clone, Debug)]
pub struct CompiledMomentumSgdStepResult {
    loss: TensorData,
    outputs: BTreeMap<String, TensorData>,
    step: u64,
    capture_identity: u64,
}

impl CompiledMomentumSgdStepResult {
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

/// One static CPU momentum-SGD program with runtime-owned recurrent state.
///
/// Compilation builds one pure graph and one mixed capture. The graph is then
/// dropped: every later step is graph-free interpreter replay through the
/// capture, and [`EffectRuntime`] is the sole owner of parameter/momentum
/// bytes. This type deliberately has no live-module synchronization surface.
pub struct CpuCompiledMomentumSgd {
    capture: CapturedMixedSchedule,
    runtime: EffectRuntime,
    cursor: MixedReplayCursor,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    parameter_buffers: BTreeMap<String, u64>,
    momentum_buffers: BTreeMap<String, u64>,
    step: u64,
}

impl CpuCompiledMomentumSgd {
    /// Compiles one exact static training program.
    ///
    /// `build` receives the declared external inputs and detached parameter
    /// graph inputs. It returns one scalar F32 loss and deterministically named
    /// detached outputs. All parameter gradients are constructed by exactly
    /// one [`Graph::gradient_default`] traversal.
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
        let parameters = canonical_parameters(parameters)?;
        if parameters.is_empty() {
            return Err(training(
                "compiled momentum-SGD needs at least one parameter",
            ));
        }
        if parameters
            .keys()
            .any(|name| config.inputs.contains_key(name))
        {
            return Err(training(
                "compiled parameter and input names must be globally unique",
            ));
        }

        let mut graph = Graph::new();
        let inputs = config
            .inputs
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

        let mut parameter_nodes = BTreeMap::new();
        let mut momentum_nodes = BTreeMap::new();
        let mut state_values = Vec::with_capacity(parameters.len() * 2);
        let mut state_by_input = BTreeMap::new();
        let mut parameter_buffers = BTreeMap::new();
        let mut momentum_buffers = BTreeMap::new();
        for (ordinal, (name, value)) in parameters.iter().enumerate() {
            let ordinal = u64::try_from(ordinal).map_err(|_| training("parameter overflow"))?;
            let parameter_buffer = STATE_BUFFER_BASE
                .checked_add(
                    ordinal
                        .checked_mul(2)
                        .ok_or_else(|| training("parameter buffer overflow"))?,
                )
                .ok_or_else(|| training("parameter buffer overflow"))?;
            let momentum_buffer = parameter_buffer
                .checked_add(1)
                .ok_or_else(|| training("momentum buffer overflow"))?;
            let parameter_input_name = format!("{INTERNAL_PREFIX}parameter_{ordinal}");
            let momentum_input_name = format!("{INTERNAL_PREFIX}momentum_{ordinal}");
            let parameter = graph.input_dtype_requires_grad(
                parameter_input_name,
                value.shape().clone(),
                DType::F32,
                true,
            );
            let momentum = graph.input_dtype_requires_grad(
                momentum_input_name,
                value.shape().clone(),
                DType::F32,
                false,
            );
            let zeros = TensorData::zeros_with_dtype(value.shape().clone(), DType::F32)?;
            let parameter_state = state_for(parameter_buffer, value)?;
            let momentum_state = state_for(momentum_buffer, &zeros)?;
            parameter_nodes.insert(name.clone(), parameter);
            momentum_nodes.insert(name.clone(), momentum);
            parameter_buffers.insert(name.clone(), parameter_buffer);
            momentum_buffers.insert(name.clone(), momentum_buffer);
            state_by_input.insert(parameter, parameter_state.clone());
            state_by_input.insert(momentum, momentum_state.clone());
            state_values.push((parameter_buffer, value.clone()));
            state_values.push((momentum_buffer, zeros));
        }

        let (loss, outputs) = build(&mut graph, &inputs, &parameter_nodes)?;
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            config.inputs.keys().chain(parameters.keys()),
        )?;

        let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
        let gradients = graph.gradient_default(loss, &targets)?;
        if gradients.len() != targets.len() {
            return Err(training("compiled gradient target count mismatch"));
        }
        let momentum_constant = graph.full_with_dtype(
            Shape::from([]),
            Scalar::F(config.momentum as f64),
            DType::F32,
        )?;
        let mut updates = Vec::with_capacity(parameters.len());
        for ((name, parameter), gradient) in parameter_nodes.iter().zip(gradients) {
            let momentum = momentum_nodes[name];
            let retained = graph.mul(momentum_constant, momentum)?;
            let next_momentum = graph.add(retained, gradient)?;
            let scaled = graph.mul(learning_rate, next_momentum)?;
            let next_parameter = graph.sub(*parameter, scaled)?;
            if graph.shape(next_momentum)? != graph.shape(*parameter)?
                || graph.dtype(next_momentum)? != DType::F32
                || graph.shape(next_parameter)? != graph.shape(*parameter)?
                || graph.dtype(next_parameter)? != DType::F32
            {
                return Err(training("compiled momentum update descriptor mismatch"));
            }
            updates.push((name.clone(), next_momentum, next_parameter));
        }

        let mut requested = Vec::with_capacity(1 + outputs.len() + updates.len() * 2);
        requested.push(loss);
        requested.extend(outputs.values().copied());
        for (_, momentum, parameter) in &updates {
            requested.extend([*momentum, *parameter]);
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
        let mut effect_bindings = Vec::with_capacity(updates.len() * 2);
        for (ordinal, (name, next_momentum, next_parameter)) in updates.iter().enumerate() {
            if next_momentum.index() as u64 >= STATE_BUFFER_BASE
                || next_parameter.index() as u64 >= STATE_BUFFER_BASE
            {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let parameter_value = &parameters[name];
            let zeros = TensorData::zeros_with_dtype(parameter_value.shape().clone(), DType::F32)?;
            let parameter = effects
                .insert(parameter_buffers[name], parameter_value.clone())
                .map_err(effect_error)?;
            let momentum = effects
                .insert(momentum_buffers[name], zeros.clone())
                .map_err(effect_error)?;
            let momentum_source = effects
                .insert(next_momentum.index() as u64, zeros)
                .map_err(effect_error)?;
            let parameter_source = effects
                .insert(
                    next_parameter.index() as u64,
                    TensorData::zeros_with_dtype(parameter_value.shape().clone(), DType::F32)?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&momentum, &momentum_source)
                .map_err(effect_error)?;
            effects
                .assign(&parameter, &parameter_source)
                .map_err(effect_error)?;
            let effect_index = u64::try_from(ordinal)
                .map_err(|_| training("effect index overflow"))?
                .checked_mul(2)
                .ok_or_else(|| training("effect index overflow"))?;
            effect_bindings.push(value_binding(&pure, *next_momentum, effect_index)?);
            effect_bindings.push(value_binding(
                &pure,
                *next_parameter,
                effect_index
                    .checked_add(1)
                    .ok_or_else(|| training("effect index overflow"))?,
            )?);
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
        validate_external_binding_ownership(&capture, config.inputs.keys())?;

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
            runtime,
            cursor,
            inputs: config.inputs,
            output_names,
            parameter_buffers,
            momentum_buffers,
            step: 0,
        })
    }

    /// Executes one graph-free replay and atomically publishes all parameter
    /// and momentum successors. The LR is an explicit rank-zero F32 input.
    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.step_inner(inputs, learning_rate, None)
    }

    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledMomentumSgdStepResult> {
        validate_step_inputs(&self.inputs, &inputs, &learning_rate)?;
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| training("compiled momentum-SGD step overflow"))?;
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
        Ok(CompiledMomentumSgdStepResult {
            loss,
            outputs,
            step: self.step,
            capture_identity: self.cursor.capture_identity(),
        })
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn capture_identity(&self) -> u64 {
        self.cursor.capture_identity()
    }

    /// Returns independent owned parameter snapshots in canonical name order.
    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.snapshots(&self.parameter_buffers)
    }

    /// Returns independent owned momentum snapshots in canonical name order.
    pub fn momentum_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.snapshots(&self.momentum_buffers)
    }

    /// Current logical parameter versions. Every successful step advances all
    /// parameter and momentum buffers exactly once.
    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.versions(&self.parameter_buffers)
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

fn canonical_parameters(
    parameters: impl IntoIterator<Item = TrainingParameterInit>,
) -> Result<BTreeMap<String, TensorData>> {
    let mut values = BTreeMap::new();
    for parameter in parameters {
        validate_user_name(&parameter.name, "parameter")?;
        if parameter.value.dtype() != DType::F32 {
            return Err(training("compiled momentum-SGD parameters must be F32"));
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
            "compiled momentum-SGD loss must be a rank-zero F32 scalar",
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
            "compiled momentum-SGD learning rate must be rank-zero F32",
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
    use crate::{Backend, CpuBackend, LossOptions, cross_entropy};
    use std::collections::HashMap;

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
                compiled.versions(&compiled.momentum_buffers).unwrap(),
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
            .capture
            .state_bindings
            .iter()
            .map(|binding| {
                compiled
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
        for binding in &compiled.capture.state_bindings {
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
        let initial_cursor = compiled.cursor.clone();
        assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
        assert_eq!(compiled.step_count(), 0);
        assert_eq!(compiled.cursor, initial_cursor);
        assert_eq!(compiled.parameter_snapshots().unwrap(), initial_parameters);
        assert_eq!(compiled.momentum_snapshots().unwrap(), initial_momentum);

        compiled.step(batch(), lr()).unwrap();
        let advanced = compiled.cursor.clone();
        let advanced_parameters = compiled.parameter_snapshots().unwrap();
        compiled.cursor = initial_cursor;
        assert!(compiled.step(batch(), lr()).is_err());
        assert_eq!(compiled.step_count(), 1);
        compiled.cursor = advanced;
        assert_eq!(compiled.parameter_snapshots().unwrap(), advanced_parameters);
    }
}
