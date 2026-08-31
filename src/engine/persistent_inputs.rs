use crate::{BufferDesc, BufferState, NodeId, ReplayInput, ScheduleStateBinding, TensorData};
use std::collections::BTreeMap;

fn same_physical_descriptor(lhs: &BufferDesc, rhs: &BufferDesc) -> bool {
    lhs.id == rhs.id
        && lhs.shape == rhs.shape
        && lhs.dtype == rhs.dtype
        && lhs.bytes == rhs.bytes
        && lhs.alignment == rhs.alignment
        && lhs.read_only == rhs.read_only
}

/// Binds each logical persistent Graph input exactly once. Individual
/// consumers retain their own `BufferDesc::view` in the scheduled ABI, so a
/// normal read and a transposed read of the same state legitimately have
/// different consumer descriptors. Every physical field, the persistent
/// identity, and the optional state-to-input view must still agree.
pub(crate) fn bind_persistent_inputs<E>(
    capture_inputs: &[ReplayInput],
    bindings: &[ScheduleStateBinding],
    provided: &BTreeMap<String, TensorData>,
    mut resolve: impl FnMut(&ScheduleStateBinding) -> Result<TensorData, E>,
    mut corrupt: impl FnMut(&'static str) -> E,
    mut descriptor: impl FnMut(&'static str) -> E,
) -> Result<BTreeMap<String, TensorData>, E> {
    let mut inputs = provided.clone();
    let mut injected =
        BTreeMap::<NodeId, (BufferState, Option<crate::AffineView>, BufferDesc)>::new();
    for binding in bindings {
        let input = capture_inputs
            .iter()
            .find(|input| input.node == binding.input_node)
            .ok_or_else(|| corrupt("state input ABI is absent"))?;
        if provided.contains_key(&input.name) {
            return Err(descriptor(
                "external input shadows persistent state binding",
            ));
        }
        if !same_physical_descriptor(&binding.desc, &input.desc) {
            return Err(corrupt(
                "persistent state consumer has incompatible base descriptor",
            ));
        }
        if let Some((state, view, desc)) = injected.get(&binding.input_node) {
            if state != &binding.state
                || view != &binding.view
                || !same_physical_descriptor(desc, &binding.desc)
            {
                return Err(corrupt(
                    "persistent state input has conflicting consumer bindings",
                ));
            }
            continue;
        }
        let value = resolve(binding)?;
        if value.shape() != &input.desc.shape
            || value.dtype() != input.desc.dtype
            || value.len().checked_mul(value.dtype().itemsize()) != Some(input.desc.bytes)
        {
            return Err(descriptor("persistent state input descriptor mismatch"));
        }
        if inputs.insert(input.name.clone(), value).is_some() {
            return Err(corrupt("persistent state input name is duplicated"));
        }
        injected.insert(
            binding.input_node,
            (
                binding.state.clone(),
                binding.view.clone(),
                binding.desc.clone(),
            ),
        );
    }
    Ok(inputs)
}
