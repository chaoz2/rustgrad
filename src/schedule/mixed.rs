//! Typed ABI edges from materialized pure schedule outputs to effect STOREs.
use super::{BufferDesc, Schedule, ScheduleError};
use crate::NodeId;

/// Immutable typed edge from one pure materialization to one STORE source.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ScheduleValueBinding {
    pub producer_item: u64,
    pub producer_node: NodeId,
    pub producer_output: BufferDesc,
    pub abi_index: usize,
    pub effect_item: u64,
    pub source_position: usize,
}

/// Immutable ABI edge from one persistent state version to a pure item input.
/// The state identity is explicit; no buffer/node ordering is inferred.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ScheduleStateBinding {
    pub state: crate::BufferState,
    pub view: Option<crate::AffineView>,
    pub consumer_item: u64,
    pub consumer_node: NodeId,
    pub input_node: NodeId,
    pub desc: BufferDesc,
    pub abi_index: usize,
}

impl ScheduleStateBinding {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.desc.read_only || self.desc.dtype != self.state.dtype {
            return Err("state binding descriptor mismatch".into());
        }
        if let Some(view) = &self.view {
            view.validate_read().map_err(|_| "state binding view")?;
            if view.source_shape != self.state.shape || view.logical_shape != self.desc.shape {
                return Err("state binding view descriptor mismatch".into());
            }
        } else if self.desc.shape != self.state.shape {
            return Err("state binding shape mismatch".into());
        }
        let expected_bytes = self
            .desc
            .shape
            .numel()
            .map_err(|_| "state binding shape overflow")?
            .checked_mul(self.desc.dtype.itemsize())
            .ok_or("state binding byte overflow")?;
        if self.desc.bytes != expected_bytes {
            return Err("state binding byte mismatch".into());
        }
        Ok(())
    }
}

/// Attaches explicit persistent-version reads to an otherwise ordinary pure or
/// mixed schedule. The ABI descriptor remains the Graph input descriptor;
/// `state` supplies the independent persistent identity and version.
pub fn bind_states(
    mut schedule: Schedule,
    bindings: Vec<ScheduleStateBinding>,
) -> Result<Schedule, ScheduleError> {
    if !schedule.state_bindings.is_empty() {
        return Err(ScheduleError::Binding(
            "schedule already has state bindings".into(),
        ));
    }
    schedule.state_bindings = bindings;
    schedule.validate()?;
    super::rekey_schedule_items(&mut schedule.items, &schedule.state_bindings, None)?;
    Ok(schedule)
}

impl ScheduleValueBinding {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.producer_output.id != self.producer_node.index() as u64 {
            return Err("value binding producer node/output mismatch".into());
        }
        if self.source_position != 0 {
            return Err("effect STORE has exactly one value source position".into());
        }
        Ok(())
    }
}

/// Builds one ordered mixed DAG from independently-valid pure and effect
/// schedules. Bindings are typed identities, never labels or inferred buffer
/// matches. Effect identifiers in `bindings` are relative to `effects`.
pub fn combine(
    pure: Schedule,
    mut effects: Schedule,
    mut bindings: Vec<ScheduleValueBinding>,
) -> Result<Schedule, ScheduleError> {
    pure.validate()?;
    effects.validate()?;
    if pure.items.iter().any(|item| !item.outputs.is_single()) {
        return Err(ScheduleError::Binding(
            "mixed schedules do not yet execute multi-output producers".into(),
        ));
    }
    if pure.items.iter().any(|item| item.is_effect())
        || effects.items.iter().any(|item| !item.is_effect())
        || !pure.value_bindings.is_empty()
        || !effects.value_bindings.is_empty()
        || !effects.requested_passthroughs.is_empty()
    {
        return Err(ScheduleError::Binding(
            "mixed construction requires disjoint canonical schedules".into(),
        ));
    }
    let offset = u64::try_from(pure.items.len()).map_err(|_| ScheduleError::Overflow)?;
    for item in &mut effects.items {
        item.id = item.id.checked_add(offset).ok_or(ScheduleError::Overflow)?;
        for edge in &mut item.dependencies {
            *edge = edge.checked_add(offset).ok_or(ScheduleError::Overflow)?;
        }
        for edge in &mut item.consumers {
            *edge = edge.checked_add(offset).ok_or(ScheduleError::Overflow)?;
        }
    }
    for binding in &mut bindings {
        let producer = pure
            .items
            .get(binding.producer_item as usize)
            .ok_or_else(|| ScheduleError::Binding("value binding producer is absent".into()))?;
        if producer.node != binding.producer_node
            || producer.primary_output() != &binding.producer_output
            || binding.abi_index != 0
        {
            return Err(ScheduleError::Binding(
                "value binding producer identity mismatch".into(),
            ));
        }
        let relative = binding.effect_item;
        let effect = effects
            .items
            .get(relative as usize)
            .ok_or_else(|| ScheduleError::Binding("value binding effect is absent".into()))?;
        if binding.source_position != 0
            || effect
                .inputs
                .first()
                .map(|desc| (&desc.shape, desc.dtype, desc.bytes))
                != Some((
                    &binding.producer_output.shape,
                    binding.producer_output.dtype,
                    binding.producer_output.bytes,
                ))
        {
            return Err(ScheduleError::Binding(
                "value binding payload descriptor mismatch".into(),
            ));
        }
        binding.effect_item = relative
            .checked_add(offset)
            .ok_or(ScheduleError::Overflow)?;
        let effect = effects
            .items
            .get_mut(relative as usize)
            .expect("checked effect");
        effect.inputs[0] = binding.producer_output.clone();
        effect.input_bindings[0] = super::ScheduleInputBinding {
            input_node: binding.producer_node,
            desc: binding.producer_output.clone(),
            abi_index: binding.abi_index,
        };
        if !effect.dependencies.contains(&binding.producer_item) {
            effect.dependencies.push(binding.producer_item);
        }
    }
    let requested_materializations = pure.requested_materializations;
    let requested_passthroughs = pure.requested_passthroughs;
    let state_bindings = pure.state_bindings;
    let mut items = pure.items;
    items.extend(effects.items);
    for item in &mut items {
        item.consumers.clear();
        item.dependencies.sort_unstable();
    }
    for position in 0..items.len() {
        let consumer = items[position].id;
        let dependencies = items[position].dependencies.clone();
        for dependency in dependencies {
            let producer = items
                .get_mut(dependency as usize)
                .expect("validated mixed dependency");
            producer.consumers.push(consumer);
        }
    }
    for item in &mut items {
        item.consumers.sort_unstable();
    }
    super::rekey_schedule_items(&mut items, &state_bindings, None)?;
    let schedule = Schedule {
        items,
        requested_materializations,
        requested_passthroughs,
        value_bindings: bindings,
        state_bindings,
    };
    schedule.validate()?;
    Ok(schedule)
}
