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
    if pure.items.iter().any(|item| item.is_effect())
        || effects.items.iter().any(|item| !item.is_effect())
        || !pure.value_bindings.is_empty()
        || !effects.value_bindings.is_empty()
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
            || producer.output != binding.producer_output
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
    let mut items = pure.items;
    items.extend(effects.items);
    for binding in &bindings {
        let producer = items
            .get_mut(binding.producer_item as usize)
            .expect("checked producer");
        if !producer.consumers.contains(&binding.effect_item) {
            producer.consumers.push(binding.effect_item);
        }
    }
    for item in &mut items {
        item.consumers.sort_unstable();
        item.dependencies.sort_unstable();
    }
    for item in &mut items {
        item.cache_key = super::item_cache_key(item);
    }
    let schedule = Schedule {
        items,
        value_bindings: bindings,
    };
    schedule.validate()?;
    Ok(schedule)
}
