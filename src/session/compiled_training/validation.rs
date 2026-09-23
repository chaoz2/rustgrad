//! Descriptor, binding, and transition validation for compiled training programs.

use super::state_schema::INTERNAL_PREFIX;
use super::{
    CompiledAdamWConfig, CpuNonFinitePolicy, LEARNING_RATE_INPUT, TrainingParameterInit,
    effect_error, training,
};
use crate::{
    BufferState, CapturedMixedSchedule, DType, EffectGraph, Graph, NodeId, Result, Schedule,
    ScheduleStateBinding, ScheduleValueBinding, Shape, TensorData,
};
use std::collections::{BTreeMap, BTreeSet};

// Parameter and descriptor admission.

pub(super) fn canonical_parameters(
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

pub(super) fn validate_weight_decay_exclusion_names<'a>(
    config: &CompiledAdamWConfig,
    parameter_names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    if config.weight_decay_exclusions.is_empty() {
        return Ok(());
    }
    let parameter_names = parameter_names.into_iter().collect::<BTreeSet<_>>();
    if let Some(name) = config
        .weight_decay_exclusions
        .iter()
        .find(|name| !parameter_names.contains(name.as_str()))
    {
        return Err(training(format!(
            "compiled AdamW weight-decay exclusion name {name:?} is unknown"
        )));
    }
    Ok(())
}

pub(super) fn validate_user_name(name: &str, kind: &str) -> Result<()> {
    if name.is_empty() || name == "loss" || name.starts_with(INTERNAL_PREFIX) {
        return Err(training(format!("invalid compiled {kind} name")));
    }
    Ok(())
}

pub(super) fn checked_bytes(value: &TensorData) -> Result<usize> {
    value
        .len()
        .checked_mul(value.dtype().itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

pub(super) fn checked_recurrent_state_extent(
    bytes: impl IntoIterator<Item = usize>,
) -> Result<(usize, usize)> {
    bytes
        .into_iter()
        .try_fold((0usize, 0usize), |(count, total), bytes| {
            Ok((
                count
                    .checked_add(1)
                    .ok_or_else(|| training("compiled recurrent state count overflows"))?,
                total
                    .checked_add(bytes)
                    .ok_or_else(|| training("compiled recurrent state bytes overflow"))?,
            ))
        })
}

pub(super) fn checked_descriptor(shape: &Shape, dtype: DType) -> Result<usize> {
    shape
        .numel()
        .map_err(|_| training("compiled tensor element extent overflow"))?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

pub(super) fn state_for(buffer: u64, value: &TensorData) -> Result<BufferState> {
    Ok(BufferState {
        buffer,
        version: 0,
        shape: value.shape().clone(),
        dtype: value.dtype(),
        bytes: checked_bytes(value)?,
    })
}

pub(super) fn validate_loss(graph: &Graph, loss: NodeId) -> Result<()> {
    if graph.dtype(loss)? != DType::F32 || graph.shape(loss)? != &Shape::from([]) {
        return Err(training(
            "compiled training loss must be a rank-zero F32 scalar",
        ));
    }
    Ok(())
}

pub(super) fn validate_evaluation_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    provided: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if expected.len() != provided.len() || expected.keys().ne(provided.keys()) {
        return Err(training("compiled evaluation input names mismatch"));
    }
    for (name, (shape, dtype)) in expected {
        let value = &provided[name];
        if value.shape() != shape || value.dtype() != *dtype {
            return Err(training(format!(
                "compiled evaluation input {name:?} descriptor mismatch"
            )));
        }
        checked_bytes(value)?;
    }
    Ok(())
}

pub(super) fn validate_outputs<'a>(
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

// Persistent-state capture ABI validation.

pub(super) fn validate_external_binding_ownership<'a>(
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

pub(super) fn collect_state_bindings(
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

pub(super) fn value_binding(
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

pub(super) fn effect_states(effects: &EffectGraph) -> Result<Vec<BufferState>> {
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

// Runtime input and transition validation.

pub(super) fn validate_step_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    actual: &BTreeMap<String, TensorData>,
    learning_rate: &TensorData,
) -> Result<()> {
    validate_training_inputs(expected, actual)?;
    validate_learning_rate(learning_rate)
}

pub(super) fn validate_training_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    actual: &BTreeMap<String, TensorData>,
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
    Ok(())
}

pub(super) fn validate_learning_rate(learning_rate: &TensorData) -> Result<()> {
    if learning_rate.shape() != &Shape::from([]) || learning_rate.dtype() != DType::F32 {
        return Err(training(
            "compiled training learning rate must be rank-zero F32",
        ));
    }
    checked_bytes(learning_rate)?;
    Ok(())
}

pub(super) fn validate_learning_rate_for_policy(
    learning_rate: &TensorData,
    policy: CpuNonFinitePolicy,
) -> Result<()> {
    validate_learning_rate(learning_rate)?;
    if policy == CpuNonFinitePolicy::RejectTransition {
        validate_finite_tensors(std::iter::once(learning_rate), "external learning rate")?;
    }
    Ok(())
}

pub(super) fn validate_staged_transition<'a>(
    outputs: &[TensorData],
    successors: impl IntoIterator<Item = &'a TensorData>,
    policy: CpuNonFinitePolicy,
    require_loss: bool,
) -> std::result::Result<(), String> {
    if policy == CpuNonFinitePolicy::Propagate {
        return Ok(());
    }
    if require_loss {
        let loss = outputs
            .first()
            .ok_or_else(|| "compiled CPU transition loss is absent".to_owned())?;
        if loss.shape() != &Shape::from([]) || loss.dtype() != DType::F32 {
            return Err("compiled CPU transition loss must be rank-zero F32".to_owned());
        }
        if has_non_finite_f32(std::iter::once(loss)) {
            return Err("compiled CPU transition has a non-finite loss".to_owned());
        }
    }
    if has_non_finite_f32(successors) {
        return Err("compiled CPU transition has a non-finite recurrent successor".to_owned());
    }
    Ok(())
}

pub(super) fn validate_finite_tensors<'a>(
    tensors: impl IntoIterator<Item = &'a TensorData>,
    role: &str,
) -> Result<()> {
    if has_non_finite_f32(tensors) {
        return Err(training(format!(
            "compiled CPU transition has a non-finite {role}"
        )));
    }
    Ok(())
}

pub(super) fn has_non_finite_f32<'a>(tensors: impl IntoIterator<Item = &'a TensorData>) -> bool {
    tensors.into_iter().any(|tensor| {
        tensor.dtype() == DType::F32 && tensor.values().iter().any(|value| !value.is_finite())
    })
}
