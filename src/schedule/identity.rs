//! Stable logical identities shared by live schedules and persisted artifacts.
//!
//! These bytes are an ABI: never derive a durable key through `Hash` or a
//! process-selected hasher. Add a new domain version when the logical fields
//! change, while leaving runtime-only cache maps free to choose local keys.

use super::{BufferDesc, QuantizedScheduleInputBinding, ScheduleBoundary, ScheduleInputBinding};
use crate::uop::artifact::{
    ArtifactError, Writer, dtype_tag, encode_schedule_identity, write_affine_view,
    write_buffer_state, write_shape,
};
use crate::{ScheduleItem, ScheduleStateBinding};

const ITEM_DOMAIN: &[u8] = b"rustgrad-schedule-item";
const SPECIALIZED_DOMAIN: &[u8] = b"rustgrad-schedule-specialization";
const STATE_DOMAIN: &[u8] = b"rustgrad-schedule-state-binding";
const KEY_VERSION: u8 = 1;

pub(super) fn item_key(item: &ScheduleItem) -> Result<u64, ArtifactError> {
    Ok(fnv1a64(&item_bytes(item)?))
}

pub(super) fn specialized_item_key(
    item: &ScheduleItem,
    source_identity: u64,
    bindings: &[(u64, i64)],
) -> Result<u64, ArtifactError> {
    let item = item_bytes(item)?;
    let mut writer = domain_writer(SPECIALIZED_DOMAIN)?;
    write_blob(&mut writer, &item)?;
    writer.u64(source_identity)?;
    write_len(&mut writer, bindings.len())?;
    for (variable, value) in bindings {
        writer.u64(*variable)?;
        writer.i64(*value)?;
    }
    Ok(fnv1a64(&writer.out))
}

pub(super) fn state_bound_item_key(
    source_key: u64,
    bindings: &[&ScheduleStateBinding],
) -> Result<u64, ArtifactError> {
    let mut writer = domain_writer(STATE_DOMAIN)?;
    writer.u64(source_key)?;
    write_len(&mut writer, bindings.len())?;
    let mut bindings = bindings.to_vec();
    bindings.sort_by_key(|binding| {
        (
            binding.consumer_item,
            binding.abi_index,
            binding.input_node.index(),
        )
    });
    for binding in bindings {
        write_state_binding(&mut writer, binding)?;
    }
    Ok(fnv1a64(&writer.out))
}

#[cfg(test)]
pub(super) fn canonical_item_bytes(item: &ScheduleItem) -> Result<Vec<u8>, ArtifactError> {
    item_bytes(item)
}

fn item_bytes(item: &ScheduleItem) -> Result<Vec<u8>, ArtifactError> {
    let mut writer = domain_writer(ITEM_DOMAIN)?;
    writer.u64(item.id)?;
    writer.u64(item.node.index() as u64)?;
    write_u64s(&mut writer, &item.dependencies)?;
    write_len(&mut writer, item.inputs.len())?;
    for desc in &item.inputs {
        write_desc(&mut writer, desc)?;
    }
    write_len(&mut writer, item.outputs.len())?;
    for desc in item.outputs.iter() {
        write_desc(&mut writer, desc)?;
    }
    write_boundary(&mut writer, item.boundary.as_ref())?;
    let kernel = encode_schedule_identity(&item.kernel)?;
    write_blob(&mut writer, &kernel)?;
    write_len(&mut writer, item.external_materializations.len())?;
    for node in &item.external_materializations {
        writer.u64(node.index() as u64)?;
    }
    write_len(&mut writer, item.input_bindings.len())?;
    for binding in &item.input_bindings {
        write_input_binding(&mut writer, binding)?;
    }
    write_len(&mut writer, item.quantized_input_bindings.len())?;
    for binding in &item.quantized_input_bindings {
        write_quantized_input_binding(&mut writer, binding)?;
    }
    Ok(writer.out)
}

fn domain_writer(domain: &[u8]) -> Result<Writer, ArtifactError> {
    let mut writer = Writer::new();
    write_blob(&mut writer, domain)?;
    writer.u8(KEY_VERSION)?;
    Ok(writer)
}

fn write_desc(writer: &mut Writer, desc: &BufferDesc) -> Result<(), ArtifactError> {
    writer.u64(desc.id)?;
    write_shape(writer, &desc.shape)?;
    writer.u8(dtype_tag(desc.dtype))?;
    writer.usize(desc.bytes)?;
    writer.usize(desc.alignment)?;
    writer.bool(desc.read_only)?;
    writer.bool(desc.view.is_some())?;
    if let Some(view) = &desc.view {
        write_affine_view(writer, view)?;
    }
    Ok(())
}

fn write_input_binding(
    writer: &mut Writer,
    binding: &ScheduleInputBinding,
) -> Result<(), ArtifactError> {
    writer.u64(binding.input_node.index() as u64)?;
    write_desc(writer, &binding.desc)?;
    writer.usize(binding.abi_index)
}

fn write_quantized_input_binding(
    writer: &mut Writer,
    binding: &QuantizedScheduleInputBinding,
) -> Result<(), ArtifactError> {
    writer.u64(binding.input_node.index() as u64)?;
    writer.u32(binding.desc.ggml_type.raw())?;
    write_shape(writer, &binding.desc.logical_shape)?;
    writer.usize(binding.desc.block_elements)?;
    writer.usize(binding.desc.block_bytes)?;
    writer.usize(binding.desc.bytes)?;
    writer.usize(binding.desc.alignment)?;
    writer.u64(binding.desc.identity)?;
    writer.usize(binding.abi_index)
}

fn write_state_binding(
    writer: &mut Writer,
    binding: &ScheduleStateBinding,
) -> Result<(), ArtifactError> {
    write_buffer_state(writer, &binding.state)?;
    writer.bool(binding.view.is_some())?;
    if let Some(view) = &binding.view {
        write_affine_view(writer, view)?;
    }
    writer.u64(binding.consumer_item)?;
    writer.u64(binding.consumer_node.index() as u64)?;
    writer.u64(binding.input_node.index() as u64)?;
    write_desc(writer, &binding.desc)?;
    writer.usize(binding.abi_index)
}

fn write_boundary(
    writer: &mut Writer,
    boundary: Option<&ScheduleBoundary>,
) -> Result<(), ArtifactError> {
    match boundary {
        None => writer.u8(0),
        Some(ScheduleBoundary::Unsupported(reason)) => {
            writer.u8(1)?;
            writer.string(reason)
        }
        Some(ScheduleBoundary::NonScalarUOpBridge) => writer.u8(2),
        Some(ScheduleBoundary::Effect) => writer.u8(3),
    }
}

fn write_len(writer: &mut Writer, len: usize) -> Result<(), ArtifactError> {
    writer.u64(len as u64)
}

fn write_u64s(writer: &mut Writer, values: &[u64]) -> Result<(), ArtifactError> {
    write_len(writer, values.len())?;
    for value in values {
        writer.u64(*value)?;
    }
    Ok(())
}

fn write_blob(writer: &mut Writer, bytes: &[u8]) -> Result<(), ArtifactError> {
    write_len(writer, bytes.len())?;
    writer.bytes(bytes)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BufferState, DType, NodeId, ScheduleItem, ScheduleStateBinding, ScheduledOutputs, Shape,
        UOp,
    };

    fn decode_hex(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn scalar_sink() -> ScheduleItem {
        let output = BufferDesc {
            id: 0,
            shape: Shape::new(vec![]),
            dtype: DType::F32,
            bytes: DType::F32.itemsize(),
            alignment: DType::F32.itemsize(),
            read_only: false,
            view: None,
        };
        ScheduleItem {
            id: 0,
            node: NodeId::from_index(0),
            dependencies: vec![],
            consumers: vec![],
            inputs: vec![],
            input_bindings: vec![],
            quantized_input_bindings: vec![],
            external_materializations: vec![],
            outputs: ScheduledOutputs::single(output),
            kernel: UOp::sink(vec![]),
            boundary: None,
            cache_key: 0,
        }
    }

    #[test]
    fn canonical_item_key_has_frozen_versioned_bytes() {
        let item = scalar_sink();
        let expected = decode_hex(concat!(
            "160000000000000072757374677261642d7363686564756c652d6974656d01",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "01000000000000000000000000000000000000000b04000000000000000400000000000000",
            "0000001c0000000000000052475541120100000000000000000000001d000000000000d6a96d7c",
            "000000000000000000000000000000000000000000000000"
        ));
        let bytes = canonical_item_bytes(&item).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(item_key(&item).unwrap(), 0x087a_c53b_3e9b_f53f);
        assert_eq!(item_key(&item).unwrap(), item_key(&item).unwrap());
    }

    #[test]
    fn mixed_state_keys_are_binding_order_independent_and_version_sensitive() {
        let binding = |buffer, version, abi_index| ScheduleStateBinding {
            state: BufferState {
                buffer,
                version,
                shape: Shape::from([2]),
                dtype: DType::F32,
                bytes: 2 * DType::F32.itemsize(),
            },
            view: None,
            consumer_item: 0,
            consumer_node: NodeId::from_index(2),
            input_node: NodeId::from_index(abi_index),
            desc: BufferDesc {
                id: abi_index as u64,
                shape: Shape::from([2]),
                dtype: DType::F32,
                bytes: 2 * DType::F32.itemsize(),
                alignment: DType::F32.itemsize(),
                read_only: true,
                view: None,
            },
            abi_index,
        };
        let first = binding(10, 1, 0);
        let second = binding(11, 1, 1);
        assert_eq!(
            state_bound_item_key(7, &[&first, &second]).unwrap(),
            state_bound_item_key(7, &[&second, &first]).unwrap()
        );
        let changed = binding(10, 2, 0);
        assert_ne!(
            state_bound_item_key(7, &[&first, &second]).unwrap(),
            state_bound_item_key(7, &[&changed, &second]).unwrap()
        );
    }
}
