//! Portable executable schedule descriptors and bindings.
use super::{
    BufferDesc, QuantizedScheduleInputBinding, ScheduleBoundary, ScheduleInputBinding,
    ScheduleItem, ScheduledOutputs,
};
use crate::engine::symbolic::{
    SpecializedFrom, SymbolicGuard, SymbolicItemDomain, SymbolicParameter, SymbolicSchema,
};
use crate::engine::symbolic_projected::SymbolicProjectedIndexMap;
use crate::engine::symbolic_view::SymbolicViewMap;
use crate::projected_index::ProjectedExpr;
use crate::tensor::artifact as tensor_artifact;
use crate::uop::artifact::{
    ArtifactError, Reader, Writer, checksum, decode as decode_uop, dtype, dtype_tag,
    encode_schedule_identity, read_affine_view, read_shape, read_symbolic, read_view,
    write_affine_view, write_shape, write_symbolic, write_view,
};
use crate::{
    CapturedSchedule, GgmlType, NodeId, QuantizedTensorData, ReplayInput, SymbolicDim,
    SymbolicExpr, SymbolicShape, SymbolicVar,
};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 4] = b"RGSA";
/// v7 replaces implementation-selected `DefaultHasher` item/specialization
/// keys with the canonical versioned schedule-key codec.
/// v8 adds authenticated zero-kernel requested affine passthroughs while
/// preserving every v1-v7 decoder and payload.
/// v9 removes the nonsemantic symbolic Reduction input-buffer word while
/// retaining authenticated v8 decode and deterministic current re-encoding.
/// v10 authenticates binding-independent projected-index expressions.
const VERSION: u8 = 10;
const LAST_OPAQUE_KEY_VERSION: u8 = 6;
const HEADER_LEN: usize = MAGIC.len() + 1 + std::mem::size_of::<u64>();
// The executable envelope supports ordered outputs; the inspection-only
// multi-output envelope remains intentionally separate.
/// Inspection-only scheduled-output envelope. This deliberately has a
/// distinct magic and identity domain from the released single-output
/// executable artifact above.
const MULTI_MAGIC: &[u8; 4] = b"RGSO";
const MULTI_VERSION: u8 = 5;
const MAX_ARTIFACT_BYTES: usize = 64 << 20;
const MAX_ITEMS: usize = 1 << 16;
const MAX_BINDINGS: usize = 1 << 16;

pub fn encode(capture: &CapturedSchedule) -> Result<Vec<u8>, ArtifactError> {
    validate(capture, true)?;
    let identity = identity(capture)?;
    let mut w = Writer::new();
    w.bytes(MAGIC)?;
    w.u8(VERSION)?;
    w.u64(identity)?;
    write_payload(&mut w, capture)?;
    if w.out
        .len()
        .checked_add(4)
        .is_none_or(|len| len > MAX_ARTIFACT_BYTES)
    {
        return Err(ArtifactError::Format("schedule length"));
    }
    let sum = checksum(&w.out);
    w.u32(sum)?;
    Ok(w.out)
}

pub fn decode(bytes: &[u8]) -> Result<CapturedSchedule, ArtifactError> {
    if bytes.len() < 17 || bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(ArtifactError::Format("schedule length"));
    }
    let body = bytes.len() - 4;
    let got = u32::from_le_bytes(bytes[body..].try_into().unwrap());
    if checksum(&bytes[..body]) != got {
        return Err(ArtifactError::Checksum);
    }
    let mut r = Reader::new(&bytes[..body]);
    if r.take(4)? != MAGIC {
        return Err(ArtifactError::Format("schedule magic"));
    }
    let version = r.u8()?;
    if !(1..=VERSION).contains(&version) {
        return Err(ArtifactError::Format("schedule version"));
    }
    let stored_identity = r.u64()?;
    let mut capture = read_payload(&mut r, stored_identity, version)?;
    if !r.done() {
        return Err(ArtifactError::Format("schedule trailing bytes"));
    }
    if version <= LAST_OPAQUE_KEY_VERSION {
        let decoded_identity = fnv1a64(&bytes[HEADER_LEN..body]);
        if decoded_identity != stored_identity {
            return Err(ArtifactError::Format("schedule identity"));
        }
        upgrade_legacy_storeless_sinks(&mut capture);
        // Historical item keys are authenticated bytes, not values the
        // current process can reproduce. Validate every other executable
        // field first while deliberately skipping only current-key equality.
        validate(&capture, false)?;
    } else {
        validate(&capture, true)?;
        let decoded_identity = fnv1a64(&bytes[HEADER_LEN..body]);
        if decoded_identity != stored_identity {
            return Err(ArtifactError::Format("schedule identity"));
        }
    }
    if version <= LAST_OPAQUE_KEY_VERSION {
        // Historical item keys were opaque `DefaultHasher` outputs. The
        // original envelope identity above authenticates those exact stored
        // bytes, while structural validation proves every executable field.
        // Never claim a current process can reproduce the old key state:
        // discard it and derive the current canonical identities instead.
        rekey_current(&mut capture)?;
    }
    if version < VERSION {
        // The stored envelope identity authenticates the exact historical
        // payload above.  Current validation must instead use the identity
        // of the upgraded payload, including fields appended by newer
        // versions such as the v8 requested-passthrough sidecar.
        capture.identity = identity(&capture)?;
    }
    validate_capture(&capture)?;
    Ok(capture)
}

/// Encodes an inspection-only capture whose items may carry an ordered output
/// collection. It is intentionally not accepted by the executable replay
/// validation path: coupled producers have not been introduced yet.
pub fn encode_scheduled_outputs(capture: &CapturedSchedule) -> Result<Vec<u8>, ArtifactError> {
    validate_scheduled_outputs(capture, true)?;
    let identity = scheduled_outputs_identity(capture)?;
    let mut w = Writer::new();
    w.bytes(MULTI_MAGIC)?;
    w.u8(MULTI_VERSION)?;
    w.u64(identity)?;
    write_scheduled_outputs_payload(&mut w, capture)?;
    if w.out
        .len()
        .checked_add(4)
        .is_none_or(|len| len > MAX_ARTIFACT_BYTES)
    {
        return Err(ArtifactError::Format("schedule length"));
    }
    let sum = checksum(&w.out);
    w.u32(sum)?;
    Ok(w.out)
}

/// Decodes the inspection-only scheduled-output envelope. Callers may inspect
/// its logical descriptors, but normal capture/replay validation continues to
/// reject every multi-output item before live work.
pub fn decode_scheduled_outputs(bytes: &[u8]) -> Result<CapturedSchedule, ArtifactError> {
    if bytes.len() < 17 || bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(ArtifactError::Format("schedule length"));
    }
    let body = bytes.len() - 4;
    let got = u32::from_le_bytes(bytes[body..].try_into().unwrap());
    if checksum(&bytes[..body]) != got {
        return Err(ArtifactError::Checksum);
    }
    let mut r = Reader::new(&bytes[..body]);
    if r.take(4)? != MULTI_MAGIC {
        return Err(ArtifactError::Format("scheduled-output magic"));
    }
    let version = r.u8()?;
    if !(1..=MULTI_VERSION).contains(&version) {
        return Err(ArtifactError::Format("scheduled-output version"));
    }
    let stored_identity = r.u64()?;
    let mut capture = read_scheduled_outputs_payload(&mut r, stored_identity, version)?;
    if !r.done() {
        return Err(ArtifactError::Format("schedule trailing bytes"));
    }
    validate_scheduled_outputs(&capture, version == MULTI_VERSION)?;
    if fnv1a64(&bytes[HEADER_LEN..body]) != stored_identity {
        return Err(ArtifactError::Format("scheduled-output identity"));
    }
    if version < MULTI_VERSION {
        rekey_current(&mut capture)?;
        capture.identity = scheduled_outputs_identity(&capture)?;
        validate_scheduled_outputs(&capture, true)?;
    }
    Ok(capture)
}

pub(crate) fn identity(capture: &CapturedSchedule) -> Result<u64, ArtifactError> {
    let mut w = Writer::new();
    write_payload(&mut w, capture)?;
    if w.out
        .len()
        .checked_add(17)
        .is_none_or(|len| len > MAX_ARTIFACT_BYTES)
    {
        return Err(ArtifactError::Format("schedule length"));
    }
    Ok(fnv1a64(&w.out))
}

pub(crate) fn scheduled_outputs_identity(capture: &CapturedSchedule) -> Result<u64, ArtifactError> {
    let mut w = Writer::new();
    write_scheduled_outputs_payload(&mut w, capture)?;
    if w.out
        .len()
        .checked_add(17)
        .is_none_or(|len| len > MAX_ARTIFACT_BYTES)
    {
        return Err(ArtifactError::Format("schedule length"));
    }
    Ok(fnv1a64(&w.out))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn write_payload(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_base(w, c, true, true, true)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    write_requested_passthroughs(w, &c.requested_passthroughs)?;
    Ok(())
}

#[cfg(test)]
fn write_payload_v5(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_base(w, c, true, true, false)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema_v8(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    Ok(())
}

#[cfg(test)]
fn write_payload_v7(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_base(w, c, true, true, true)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema_v8(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    Ok(())
}

#[cfg(test)]
fn write_payload_v8(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_base(w, c, true, true, true)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema_v8(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    write_requested_passthroughs(w, &c.requested_passthroughs)
}

#[cfg(test)]
fn write_payload_v9(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_base(w, c, true, true, true)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema_v9(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    write_requested_passthroughs(w, &c.requested_passthroughs)
}

fn write_requested_passthroughs(
    w: &mut Writer,
    passthroughs: &[crate::RequestedPassthrough],
) -> Result<(), ArtifactError> {
    write_len(w, passthroughs.len(), MAX_BINDINGS)?;
    for passthrough in passthroughs {
        w.u64(passthrough.requested.index() as u64)?;
        w.u64(passthrough.source.index() as u64)?;
        write_desc_inner(w, &passthrough.desc, true)?;
    }
    Ok(())
}

#[cfg(test)]
fn write_payload_v2(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_payload_v1(w, c)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema_v2(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    Ok(())
}

#[cfg(test)]
fn write_payload_v1(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_base(w, c, false, false, false)
}

fn write_base(
    w: &mut Writer,
    c: &CapturedSchedule,
    quantized_items: bool,
    output_lists: bool,
    affine_views: bool,
) -> Result<(), ArtifactError> {
    write_len(w, c.items.len(), MAX_ITEMS)?;
    for item in &c.items {
        if quantized_items {
            if output_lists {
                write_item(w, item, affine_views)?;
            } else {
                write_item_inner(w, item, false)?;
            }
        } else {
            if !item.quantized_input_bindings.is_empty() {
                return Err(ArtifactError::Unsupported);
            }
            write_item_v3(w, item)?;
        }
    }
    write_len(w, c.inputs.len(), MAX_BINDINGS)?;
    for input in &c.inputs {
        w.string(&input.name)?;
        w.u64(input.node.index() as u64)?;
        write_desc_inner(w, &input.desc, affine_views)?;
    }
    write_len(w, c.constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.constants {
        w.u64(*id)?;
        tensor_artifact::encode_into(w, value)?;
    }
    write_u64s(w, &c.requested)
}

fn write_scheduled_outputs_payload(
    w: &mut Writer,
    c: &CapturedSchedule,
) -> Result<(), ArtifactError> {
    write_len(w, c.items.len(), MAX_ITEMS)?;
    for item in &c.items {
        write_scheduled_outputs_item(w, item)?;
    }
    write_len(w, c.inputs.len(), MAX_BINDINGS)?;
    for input in &c.inputs {
        w.string(&input.name)?;
        w.u64(input.node.index() as u64)?;
        write_desc_inner(w, &input.desc, false)?;
    }
    write_len(w, c.constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.constants {
        w.u64(*id)?;
        tensor_artifact::encode_into(w, value)?;
    }
    write_u64s(w, &c.requested)?;
    // Symbolic specialization has an explicitly single-output ABI in this
    // migration phase. Keep its established codec fields so a single-output
    // inspection artifact can describe the same immutable capture.
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    write_requested_passthroughs(w, &c.requested_passthroughs)?;
    Ok(())
}

#[cfg(test)]
fn write_scheduled_outputs_payload_v2(
    w: &mut Writer,
    c: &CapturedSchedule,
) -> Result<(), ArtifactError> {
    write_len(w, c.items.len(), MAX_ITEMS)?;
    for item in &c.items {
        write_scheduled_outputs_item(w, item)?;
    }
    write_len(w, c.inputs.len(), MAX_BINDINGS)?;
    for input in &c.inputs {
        w.string(&input.name)?;
        w.u64(input.node.index() as u64)?;
        write_desc_inner(w, &input.desc, false)?;
    }
    write_len(w, c.constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.constants {
        w.u64(*id)?;
        tensor_artifact::encode_into(w, value)?;
    }
    write_u64s(w, &c.requested)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema_v8(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    Ok(())
}

#[cfg(test)]
fn write_scheduled_outputs_payload_v3(
    w: &mut Writer,
    c: &CapturedSchedule,
) -> Result<(), ArtifactError> {
    write_scheduled_outputs_payload_v2(w, c)?;
    write_requested_passthroughs(w, &c.requested_passthroughs)
}

#[cfg(test)]
fn write_scheduled_outputs_payload_v4(
    w: &mut Writer,
    c: &CapturedSchedule,
) -> Result<(), ArtifactError> {
    write_len(w, c.items.len(), MAX_ITEMS)?;
    for item in &c.items {
        write_scheduled_outputs_item(w, item)?;
    }
    write_len(w, c.inputs.len(), MAX_BINDINGS)?;
    for input in &c.inputs {
        w.string(&input.name)?;
        w.u64(input.node.index() as u64)?;
        write_desc_inner(w, &input.desc, false)?;
    }
    write_len(w, c.constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.constants {
        w.u64(*id)?;
        tensor_artifact::encode_into(w, value)?;
    }
    write_u64s(w, &c.requested)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema_v9(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    write_len(w, c.quantized_constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.quantized_constants {
        w.u64(*id)?;
        write_quantized_data(w, value)?;
    }
    write_requested_passthroughs(w, &c.requested_passthroughs)
}

fn read_payload(
    r: &mut Reader<'_>,
    identity: u64,
    version: u8,
) -> Result<CapturedSchedule, ArtifactError> {
    let n = r.count(MAX_ITEMS)?;
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        items.push(read_item(r, version)?);
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        inputs.push(ReplayInput {
            name: r.string()?,
            node: node(r.u64()?)?,
            desc: read_desc_inner(r, version >= 6)?,
        });
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut constants = BTreeMap::new();
    for _ in 0..n {
        let id = r.u64()?;
        if constants
            .insert(id, tensor_artifact::decode_from(r)?)
            .is_some()
        {
            return Err(ArtifactError::Format("duplicate constant"));
        }
    }
    let requested = read_u64s(r)?;
    let symbolic = if version >= 2 && r.bool()? {
        Some(read_symbolic_schema(
            r,
            version,
            version <= 8,
            version >= 10,
        )?)
    } else {
        None
    };
    let specialized_from = if version >= 2 && r.bool()? {
        Some(read_specialized_from(r)?)
    } else {
        None
    };
    let mut quantized_constants = BTreeMap::new();
    if version >= 4 {
        let n = r.count(MAX_BINDINGS)?;
        for _ in 0..n {
            let id = r.u64()?;
            if quantized_constants
                .insert(id, read_quantized_data(r)?)
                .is_some()
            {
                return Err(ArtifactError::Format("duplicate quantized constant"));
            }
        }
    }
    let requested_passthroughs = if version >= 8 {
        read_requested_passthroughs(r)?
    } else {
        Vec::new()
    };
    Ok(CapturedSchedule {
        items,
        inputs,
        constants,
        quantized_constants,
        requested_passthroughs,
        requested,
        identity,
        symbolic,
        specialized_from,
    })
}

fn read_scheduled_outputs_payload(
    r: &mut Reader<'_>,
    identity: u64,
    version: u8,
) -> Result<CapturedSchedule, ArtifactError> {
    let n = r.count(MAX_ITEMS)?;
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        items.push(read_scheduled_outputs_item(r)?);
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        inputs.push(ReplayInput {
            name: r.string()?,
            node: node(r.u64()?)?,
            desc: read_desc_inner(r, false)?,
        });
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut constants = BTreeMap::new();
    for _ in 0..n {
        let id = r.u64()?;
        if constants
            .insert(id, tensor_artifact::decode_from(r)?)
            .is_some()
        {
            return Err(ArtifactError::Format("duplicate constant"));
        }
    }
    let requested = read_u64s(r)?;
    let symbolic = if r.bool()? {
        Some(read_symbolic_schema(r, 4, version <= 3, version >= 5)?)
    } else {
        None
    };
    let specialized_from = if r.bool()? {
        Some(read_specialized_from(r)?)
    } else {
        None
    };
    let n = r.count(MAX_BINDINGS)?;
    let mut quantized_constants = BTreeMap::new();
    for _ in 0..n {
        let id = r.u64()?;
        if quantized_constants
            .insert(id, read_quantized_data(r)?)
            .is_some()
        {
            return Err(ArtifactError::Format("duplicate quantized constant"));
        }
    }
    let requested_passthroughs = if version >= 3 {
        read_requested_passthroughs(r)?
    } else {
        Vec::new()
    };
    Ok(CapturedSchedule {
        items,
        inputs,
        constants,
        quantized_constants,
        requested_passthroughs,
        requested,
        identity,
        symbolic,
        specialized_from,
    })
}

fn read_requested_passthroughs(
    r: &mut Reader<'_>,
) -> Result<Vec<crate::RequestedPassthrough>, ArtifactError> {
    let n = r.count(MAX_BINDINGS)?;
    let mut passthroughs = Vec::with_capacity(n);
    for _ in 0..n {
        passthroughs.push(crate::RequestedPassthrough {
            requested: node(r.u64()?)?,
            source: node(r.u64()?)?,
            desc: read_desc_inner(r, true)?,
        });
    }
    Ok(passthroughs)
}

#[cfg(test)]
fn identity_v1(capture: &CapturedSchedule) -> Result<u64, ArtifactError> {
    let mut writer = Writer::new();
    write_payload_v1(&mut writer, capture)?;
    if writer
        .out
        .len()
        .checked_add(17)
        .is_none_or(|len| len > MAX_ARTIFACT_BYTES)
    {
        return Err(ArtifactError::Format("schedule length"));
    }
    Ok(fnv1a64(&writer.out))
}

#[cfg(test)]
fn identity_v2(capture: &CapturedSchedule) -> Result<u64, ArtifactError> {
    let mut writer = Writer::new();
    write_payload_v2(&mut writer, capture)?;
    if writer
        .out
        .len()
        .checked_add(17)
        .is_none_or(|len| len > MAX_ARTIFACT_BYTES)
    {
        return Err(ArtifactError::Format("schedule length"));
    }
    Ok(fnv1a64(&writer.out))
}

#[cfg(test)]
fn identity_v5(capture: &CapturedSchedule) -> Result<u64, ArtifactError> {
    let mut writer = Writer::new();
    write_payload_v5(&mut writer, capture)?;
    if writer
        .out
        .len()
        .checked_add(17)
        .is_none_or(|len| len > MAX_ARTIFACT_BYTES)
    {
        return Err(ArtifactError::Format("schedule length"));
    }
    Ok(fnv1a64(&writer.out))
}

fn write_item(w: &mut Writer, x: &ScheduleItem, affine_views: bool) -> Result<(), ArtifactError> {
    w.u64(x.id)?;
    w.u64(x.node.index() as u64)?;
    write_u64s(w, &x.dependencies)?;
    write_u64s(w, &x.consumers)?;
    write_len(w, x.inputs.len(), MAX_BINDINGS)?;
    for desc in &x.inputs {
        write_desc_inner(w, desc, affine_views)?;
    }
    write_len(w, x.input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_desc_inner(w, &binding.desc, affine_views)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.quantized_input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.quantized_input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_quantized_desc(w, &binding.desc)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.external_materializations.len(), MAX_BINDINGS)?;
    for id in &x.external_materializations {
        w.u64(id.index() as u64)?;
    }
    write_len(w, x.outputs.len(), MAX_BINDINGS)?;
    for output in x.outputs.iter() {
        write_desc_inner(w, output, affine_views)?;
    }
    let kernel = encode_schedule_identity(&x.kernel)?;
    write_len(w, kernel.len(), MAX_ARTIFACT_BYTES)?;
    w.bytes(&kernel)?;
    write_boundary(w, x.boundary.as_ref())?;
    w.u64(x.cache_key)
}

fn write_scheduled_outputs_item(w: &mut Writer, x: &ScheduleItem) -> Result<(), ArtifactError> {
    w.u64(x.id)?;
    w.u64(x.node.index() as u64)?;
    write_u64s(w, &x.dependencies)?;
    write_u64s(w, &x.consumers)?;
    write_len(w, x.inputs.len(), MAX_BINDINGS)?;
    for desc in &x.inputs {
        write_desc_inner(w, desc, false)?;
    }
    write_len(w, x.input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_desc_inner(w, &binding.desc, false)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.quantized_input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.quantized_input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_quantized_desc(w, &binding.desc)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.external_materializations.len(), MAX_BINDINGS)?;
    for id in &x.external_materializations {
        w.u64(id.index() as u64)?;
    }
    // Keep the legacy primary descriptor explicit in the new envelope so a
    // decoder can reject a list whose projection was tampered independently.
    write_desc_inner(w, x.primary_output(), false)?;
    write_len(w, x.outputs.len(), MAX_BINDINGS)?;
    for output in x.outputs.iter() {
        write_desc_inner(w, output, false)?;
    }
    let kernel = encode_schedule_identity(&x.kernel)?;
    write_len(w, kernel.len(), MAX_ARTIFACT_BYTES)?;
    w.bytes(&kernel)?;
    write_boundary(w, x.boundary.as_ref())?;
    w.u64(x.cache_key)
}

/// RGSM's typed item stream shares every ordinary field codec but permits the
/// one explicit `Effect` boundary and v11 UOp payloads.
pub(crate) fn write_effect_item(w: &mut Writer, x: &ScheduleItem) -> Result<(), ArtifactError> {
    write_item_inner(w, x, true)
}

fn write_item_inner(w: &mut Writer, x: &ScheduleItem, effects: bool) -> Result<(), ArtifactError> {
    if !x.outputs.is_single() {
        return Err(ArtifactError::Unsupported);
    }
    w.u64(x.id)?;
    w.u64(x.node.index() as u64)?;
    write_u64s(w, &x.dependencies)?;
    write_u64s(w, &x.consumers)?;
    write_len(w, x.inputs.len(), MAX_BINDINGS)?;
    for desc in &x.inputs {
        write_desc_inner(w, desc, effects)?;
    }
    write_len(w, x.input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_desc_inner(w, &binding.desc, effects)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.quantized_input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.quantized_input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_quantized_desc(w, &binding.desc)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.external_materializations.len(), MAX_BINDINGS)?;
    for id in &x.external_materializations {
        w.u64(id.index() as u64)?;
    }
    write_desc_inner(w, x.primary_output(), effects)?;
    let kernel = encode_schedule_identity(&x.kernel)?;
    write_len(w, kernel.len(), MAX_ARTIFACT_BYTES)?;
    w.bytes(&kernel)?;
    write_boundary_inner(w, x.boundary.as_ref(), effects)?;
    w.u64(x.cache_key)
}

fn write_item_v3(w: &mut Writer, x: &ScheduleItem) -> Result<(), ArtifactError> {
    w.u64(x.id)?;
    w.u64(x.node.index() as u64)?;
    write_u64s(w, &x.dependencies)?;
    write_u64s(w, &x.consumers)?;
    write_len(w, x.inputs.len(), MAX_BINDINGS)?;
    for desc in &x.inputs {
        write_desc_inner(w, desc, false)?;
    }
    write_len(w, x.input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_desc_inner(w, &binding.desc, false)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.external_materializations.len(), MAX_BINDINGS)?;
    for id in &x.external_materializations {
        w.u64(id.index() as u64)?;
    }
    write_desc_inner(w, x.primary_output(), false)?;
    let kernel = encode_schedule_identity(&x.kernel)?;
    write_len(w, kernel.len(), MAX_ARTIFACT_BYTES)?;
    w.bytes(&kernel)?;
    write_boundary(w, x.boundary.as_ref())?;
    w.u64(x.cache_key)
}

fn read_item(r: &mut Reader<'_>, version: u8) -> Result<ScheduleItem, ArtifactError> {
    read_item_inner(r, version, false)
}

fn read_scheduled_outputs_item(r: &mut Reader<'_>) -> Result<ScheduleItem, ArtifactError> {
    let id = r.u64()?;
    let item_node = node(r.u64()?)?;
    let dependencies = read_u64s(r)?;
    let consumers = read_u64s(r)?;
    let n = r.count(MAX_BINDINGS)?;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        inputs.push(read_desc_inner(r, false)?);
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut input_bindings = Vec::with_capacity(n);
    for _ in 0..n {
        input_bindings.push(ScheduleInputBinding {
            input_node: node(r.u64()?)?,
            desc: read_desc_inner(r, false)?,
            abi_index: r.usize()?,
        });
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut quantized_input_bindings = Vec::with_capacity(n);
    for _ in 0..n {
        quantized_input_bindings.push(QuantizedScheduleInputBinding {
            input_node: node(r.u64()?)?,
            desc: read_quantized_desc(r)?,
            abi_index: r.usize()?,
        });
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut external_materializations = Vec::with_capacity(n);
    for _ in 0..n {
        external_materializations.push(node(r.u64()?)?);
    }
    let legacy_primary = read_desc_inner(r, false)?;
    let n = r.count(MAX_BINDINGS)?;
    let mut output_descs = Vec::with_capacity(n);
    for _ in 0..n {
        output_descs.push(read_desc_inner(r, false)?);
    }
    let outputs = ScheduledOutputs::new(output_descs)
        .map_err(|_| ArtifactError::Format("scheduled outputs"))?;
    if outputs.primary() != &legacy_primary {
        return Err(ArtifactError::Format("scheduled-output projection"));
    }
    let kernel_len = r.count(MAX_ARTIFACT_BYTES)?;
    let kernel = decode_uop(r.take(kernel_len)?)?;
    let boundary = read_boundary_inner(r, false)?;
    let cache_key = r.u64()?;
    Ok(ScheduleItem {
        id,
        node: item_node,
        dependencies,
        consumers,
        inputs,
        input_bindings,
        quantized_input_bindings,
        external_materializations,
        outputs,
        kernel,
        boundary,
        cache_key,
    })
}

pub(crate) fn read_effect_item(r: &mut Reader<'_>) -> Result<ScheduleItem, ArtifactError> {
    read_item_inner(r, 4, true)
}

fn read_item_inner(
    r: &mut Reader<'_>,
    version: u8,
    effects: bool,
) -> Result<ScheduleItem, ArtifactError> {
    let affine_views = effects || version >= 6;
    let id = r.u64()?;
    let item_node = node(r.u64()?)?;
    let dependencies = read_u64s(r)?;
    let consumers = read_u64s(r)?;
    let n = r.count(MAX_BINDINGS)?;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        inputs.push(read_desc_inner(r, affine_views)?);
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut input_bindings = Vec::with_capacity(n);
    for _ in 0..n {
        input_bindings.push(ScheduleInputBinding {
            input_node: node(r.u64()?)?,
            desc: read_desc_inner(r, affine_views)?,
            abi_index: r.usize()?,
        });
    }
    let mut quantized_input_bindings = Vec::new();
    if version >= 4 {
        let n = r.count(MAX_BINDINGS)?;
        quantized_input_bindings.reserve(n);
        for _ in 0..n {
            quantized_input_bindings.push(QuantizedScheduleInputBinding {
                input_node: node(r.u64()?)?,
                desc: read_quantized_desc(r)?,
                abi_index: r.usize()?,
            });
        }
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut external_materializations = Vec::with_capacity(n);
    for _ in 0..n {
        external_materializations.push(node(r.u64()?)?);
    }
    let outputs = if version >= 5 {
        let n = r.count(MAX_BINDINGS)?;
        let mut outputs = Vec::with_capacity(n);
        for _ in 0..n {
            outputs.push(read_desc_inner(r, affine_views)?);
        }
        ScheduledOutputs::new(outputs).map_err(|_| ArtifactError::Format("scheduled outputs"))?
    } else {
        ScheduledOutputs::single(read_desc_inner(r, affine_views)?)
    };
    let kernel_len = r.count(MAX_ARTIFACT_BYTES)?;
    let kernel = decode_uop(r.take(kernel_len)?)?;
    let boundary = read_boundary_inner(r, effects)?;
    let cache_key = r.u64()?;
    Ok(ScheduleItem {
        id,
        node: item_node,
        dependencies,
        consumers,
        inputs,
        input_bindings,
        quantized_input_bindings,
        external_materializations,
        outputs,
        kernel,
        boundary,
        cache_key,
    })
}

pub(crate) fn write_effect_desc(w: &mut Writer, x: &BufferDesc) -> Result<(), ArtifactError> {
    write_desc_inner(w, x, true)
}

fn write_desc_inner(
    w: &mut Writer,
    x: &BufferDesc,
    affine_views: bool,
) -> Result<(), ArtifactError> {
    validate_desc(x)?;
    w.u64(x.id)?;
    write_shape(w, &x.shape)?;
    w.u8(dtype_tag(x.dtype))?;
    w.usize(x.bytes)?;
    w.usize(x.alignment)?;
    w.bool(x.read_only)?;
    w.bool(x.view.is_some())?;
    if let Some(view) = &x.view {
        if affine_views {
            write_affine_view(w, view)?;
        } else {
            write_view(
                w,
                &view.as_unsigned().map_err(|_| ArtifactError::Unsupported)?,
            )?;
        }
    }
    Ok(())
}
pub(crate) fn read_effect_desc(r: &mut Reader<'_>) -> Result<BufferDesc, ArtifactError> {
    read_desc_inner(r, true)
}

fn read_desc_inner(r: &mut Reader<'_>, affine_views: bool) -> Result<BufferDesc, ArtifactError> {
    let x = BufferDesc {
        id: r.u64()?,
        shape: read_shape(r)?,
        dtype: dtype(r.u8()?)?,
        bytes: r.usize()?,
        alignment: r.usize()?,
        read_only: r.bool()?,
        view: if r.bool()? {
            Some(if affine_views {
                read_affine_view(r)?
            } else {
                read_view(r)?.into()
            })
        } else {
            None
        },
    };
    validate_desc(&x)?;
    Ok(x)
}
fn validate_desc(x: &BufferDesc) -> Result<(), ArtifactError> {
    super::validate_buffer_desc(x).map_err(|_| ArtifactError::Format("buffer descriptor"))
}

fn write_quantized_desc(
    w: &mut Writer,
    desc: &crate::QuantizedBufferDesc,
) -> Result<(), ArtifactError> {
    desc.validate_metadata()
        .map_err(|_| ArtifactError::Format("quantized descriptor"))?;
    w.u32(desc.ggml_type.raw())?;
    write_shape(w, &desc.logical_shape)?;
    w.usize(desc.block_elements)?;
    w.usize(desc.block_bytes)?;
    w.usize(desc.bytes)?;
    w.usize(desc.alignment)?;
    w.u64(desc.identity)
}

fn read_quantized_desc(r: &mut Reader<'_>) -> Result<crate::QuantizedBufferDesc, ArtifactError> {
    let ggml_type = match r.u32()? {
        2 => GgmlType::Q4_0,
        8 => GgmlType::Q8_0,
        12 => GgmlType::Q4K,
        14 => GgmlType::Q6K,
        _ => return Err(ArtifactError::Format("quantized type")),
    };
    let desc = crate::QuantizedBufferDesc {
        ggml_type,
        logical_shape: read_shape(r)?,
        block_elements: r.usize()?,
        block_bytes: r.usize()?,
        bytes: r.usize()?,
        alignment: r.usize()?,
        identity: r.u64()?,
    };
    desc.validate_metadata()
        .map_err(|_| ArtifactError::Format("quantized descriptor"))?;
    Ok(desc)
}

fn write_quantized_data(w: &mut Writer, value: &QuantizedTensorData) -> Result<(), ArtifactError> {
    value
        .validate()
        .map_err(|_| ArtifactError::Format("quantized constant"))?;
    write_quantized_desc(w, value.descriptor())?;
    write_len(w, value.bytes().len(), MAX_ARTIFACT_BYTES)?;
    w.bytes(value.bytes())
}

fn read_quantized_data(r: &mut Reader<'_>) -> Result<QuantizedTensorData, ArtifactError> {
    let desc = read_quantized_desc(r)?;
    let len = r.count(MAX_ARTIFACT_BYTES)?;
    let value = QuantizedTensorData::from_aligned_bytes(
        desc.ggml_type,
        desc.logical_shape.clone(),
        r.take(len)?.to_vec(),
        desc.alignment,
        0,
    )
    .map_err(|_| ArtifactError::Format("quantized constant"))?;
    if value.descriptor() != &desc {
        return Err(ArtifactError::Format("quantized constant identity"));
    }
    Ok(value)
}

fn validate_requested_ids(
    requested: &[u64],
    available: &BTreeSet<u64>,
) -> Result<(), ArtifactError> {
    // `requested` is a logical ordered projection, not an ownership set.
    // Repeated IDs are authenticated by the payload and materialized once.
    if requested.iter().any(|id| !available.contains(id)) {
        return Err(ArtifactError::Format("requested value"));
    }
    Ok(())
}

fn validate(c: &CapturedSchedule, validate_keys: bool) -> Result<(), ArtifactError> {
    if c.items.len() > MAX_ITEMS
        || c.inputs.len() > MAX_BINDINGS
        || c.constants.len() > MAX_BINDINGS
        || c.quantized_constants.len() > MAX_BINDINGS
        || c.requested_passthroughs.len() > MAX_BINDINGS
    {
        return Err(ArtifactError::Format("schedule limit"));
    }
    let count = c.items.len() as u64;
    let mut output_ids = BTreeSet::new();
    for (index, item) in c.items.iter().enumerate() {
        if item.id != index as u64 || item.node.index() as u64 != item.primary_output().id {
            return Err(ArtifactError::Format("item identity"));
        }
        for output in item.outputs.iter() {
            validate_desc(output)?;
            if !output_ids.insert(output.id) {
                return Err(ArtifactError::Format("item outputs"));
            }
        }
        if item.dependencies.windows(2).any(|x| x[0] >= x[1])
            || item.dependencies.iter().any(|x| *x >= item.id)
        {
            return Err(ArtifactError::Format("item dependencies"));
        }
        if item.consumers.windows(2).any(|x| x[0] >= x[1])
            || item.consumers.iter().any(|x| *x <= item.id || *x >= count)
        {
            return Err(ArtifactError::Format("item consumers"));
        }
        for dependency in &item.dependencies {
            if !c.items[*dependency as usize].consumers.contains(&item.id) {
                return Err(ArtifactError::Format("dependency edge"));
            }
        }
        for consumer in &item.consumers {
            if !c.items[*consumer as usize].dependencies.contains(&item.id) {
                return Err(ArtifactError::Format("consumer edge"));
            }
        }
        let mut external = BTreeSet::new();
        if item
            .external_materializations
            .iter()
            .any(|x| !external.insert(x.index()))
        {
            return Err(ArtifactError::Format("external materialization"));
        }
        item.validate_input_bindings()
            .map_err(|_| ArtifactError::Format("input bindings"))?;
        item.kernel
            .validate()
            .map_err(|_| ArtifactError::Format("kernel"))?;
        super::validate_item_output_bindings(item)
            .map_err(|_| ArtifactError::Format("kernel outputs"))?;
        let mut inventory = BTreeSet::new();
        for desc in &item.inputs {
            validate_desc(desc)?;
            if !inventory.insert(desc.id) {
                return Err(ArtifactError::Format("input inventory"));
            }
        }
        if item.boundary.is_none()
            && super::input_bindings(&item.kernel, &item.inputs, item.primary_output())
                .map_err(|_| ArtifactError::Format("kernel resources"))?
                != item.input_bindings
        {
            return Err(ArtifactError::Format("kernel bindings"));
        }
        if item.boundary.is_none()
            && super::quantized_input_bindings(&item.kernel)
                .map_err(|_| ArtifactError::Format("quantized kernel resources"))?
                != item.quantized_input_bindings
        {
            return Err(ArtifactError::Format("quantized kernel bindings"));
        }
        if validate_keys {
            let expected_cache_key = if let Some(provenance) = &c.specialized_from {
                super::specialized_item_cache_key(
                    item,
                    provenance.source_identity,
                    &provenance.bindings,
                )
            } else {
                super::item_cache_key(item)
            }
            .map_err(|_| ArtifactError::Format("item cache identity"))?;
            if item.cache_key != expected_cache_key {
                return Err(ArtifactError::Format("item cache identity"));
            }
        }
    }
    let mut names = BTreeSet::new();
    let mut input_ids = BTreeSet::new();
    for input in &c.inputs {
        validate_desc(&input.desc)?;
        if input.name.is_empty()
            || !names.insert(&input.name)
            || input.node.index() as u64 != input.desc.id
            || !input_ids.insert(input.desc.id)
        {
            return Err(ArtifactError::Format("replay input"));
        }
    }
    let mut passthrough_requested = BTreeSet::new();
    let mut passthrough_sources = BTreeSet::new();
    for passthrough in &c.requested_passthroughs {
        passthrough
            .validate()
            .map_err(|_| ArtifactError::Format("requested passthrough"))?;
        let requested = passthrough.requested.index() as u64;
        let source = passthrough.source.index() as u64;
        if !passthrough_requested.insert(requested)
            || !c.requested.contains(&requested)
            || output_ids.contains(&requested)
            || output_ids.contains(&source)
        {
            return Err(ArtifactError::Format("requested passthrough ownership"));
        }
        passthrough_sources.insert(source);
        let mut physical = passthrough.desc.clone();
        physical.view = None;
        let input_owner = c
            .inputs
            .iter()
            .filter(|input| input.desc.id == source)
            .count();
        let constant_owner = if c.constants.contains_key(&source) {
            1
        } else {
            0
        };
        if input_owner + constant_owner != 1
            || c.inputs
                .iter()
                .find(|input| input.desc.id == source)
                .is_some_and(|input| {
                    input.desc.id != physical.id
                        || input.desc.shape != physical.shape
                        || input.desc.dtype != physical.dtype
                        || input.desc.bytes != physical.bytes
                        || input.desc.alignment != physical.alignment
                        || !input.desc.read_only
                })
            || c.constants.get(&source).is_some_and(|value| {
                value.shape() != &physical.shape || value.dtype() != physical.dtype
            })
        {
            return Err(ArtifactError::Format("requested passthrough source"));
        }
    }
    for (id, value) in &c.constants {
        if value.shape().numel().is_err() || value.len() != value.shape().numel().unwrap() {
            return Err(ArtifactError::Format("constant tensor"));
        }
        let desc = c
            .items
            .iter()
            .flat_map(|x| &x.input_bindings)
            .find(|x| x.desc.id == *id)
            .map(|x| &x.desc);
        if let Some(desc) = desc {
            if value.shape() != &desc.shape || value.dtype() != desc.dtype {
                return Err(ArtifactError::Format("constant descriptor"));
            }
        } else if !c.requested.contains(id) && !passthrough_sources.contains(id) {
            return Err(ArtifactError::Format("unbound constant"));
        }
    }
    for (id, value) in &c.quantized_constants {
        value
            .validate()
            .map_err(|_| ArtifactError::Format("quantized constant"))?;
        let binding = c
            .items
            .iter()
            .flat_map(|item| &item.quantized_input_bindings)
            .find(|binding| binding.input_node.index() as u64 == *id)
            .ok_or(ArtifactError::Format("unbound quantized constant"))?;
        if value.descriptor() != &binding.desc {
            return Err(ArtifactError::Format("quantized constant descriptor"));
        }
    }
    let mut available = input_ids.clone();
    available.extend(c.constants.keys().copied());
    available.extend(c.quantized_constants.keys().copied());
    for item in &c.items {
        if item
            .input_bindings
            .iter()
            .any(|x| !available.contains(&x.desc.id))
            || item
                .quantized_input_bindings
                .iter()
                .any(|binding| !available.contains(&(binding.input_node.index() as u64)))
        {
            return Err(ArtifactError::Format("unavailable binding"));
        }
        available.extend(item.outputs.iter().map(|output| output.id));
    }
    let used = c
        .items
        .iter()
        .flat_map(|x| x.input_bindings.iter().map(|x| x.desc.id))
        .collect::<BTreeSet<_>>();
    let mut used = used;
    used.extend(c.items.iter().flat_map(|item| {
        item.quantized_input_bindings
            .iter()
            .map(|binding| binding.input_node.index() as u64)
    }));
    used.extend(c.requested.iter().copied());
    used.extend(passthrough_sources.iter().copied());
    if c.inputs.iter().any(|x| !used.contains(&x.desc.id)) {
        return Err(ArtifactError::Format("unused replay input"));
    }
    if c.quantized_constants.keys().any(|id| !used.contains(id)) {
        return Err(ArtifactError::Format("unused quantized constant"));
    }
    let outputs = c
        .items
        .iter()
        .flat_map(|item| item.outputs.iter().map(|output| output.id))
        .collect::<BTreeSet<_>>();
    let replay_values = input_ids
        .iter()
        .copied()
        .chain(c.constants.keys().copied())
        .chain(outputs.iter().copied())
        .chain(passthrough_requested.iter().copied())
        .collect::<BTreeSet<_>>();
    validate_requested_ids(&c.requested, &replay_values)?;
    if !c.requested_passthroughs.is_empty()
        && (c.symbolic.is_some() || c.specialized_from.is_some())
    {
        return Err(ArtifactError::Format("symbolic requested passthrough"));
    }
    if c.symbolic.is_some() && c.specialized_from.is_some() {
        return Err(ArtifactError::Format("symbolic specialization state"));
    }
    if let Some(schema) = &c.symbolic {
        schema
            .validate_against(c)
            .map_err(|_| ArtifactError::Format("symbolic schema"))?;
    }
    if let Some(provenance) = &c.specialized_from
        && (provenance.source_identity == 0
            || provenance.bindings.is_empty()
            || provenance
                .bindings
                .windows(2)
                .any(|pair| pair[0].0 >= pair[1].0))
    {
        return Err(ArtifactError::Format("specialization provenance"));
    }
    Ok(())
}

fn rekey_current(capture: &mut CapturedSchedule) -> Result<(), ArtifactError> {
    let provenance = capture.specialized_from.clone();
    for item in &mut capture.items {
        item.cache_key = if let Some(provenance) = &provenance {
            super::specialized_item_cache_key(
                item,
                provenance.source_identity,
                &provenance.bindings,
            )
        } else {
            super::item_cache_key(item)
        }
        .map_err(|_| ArtifactError::Format("item cache identity"))?;
    }
    Ok(())
}

/// Historical RGSA v1-v6 authenticated a boundary-free empty Sink. It never
/// produced its declared output, so upgrade it to the existing explicit
/// materialization boundary rather than granting current executable meaning.
fn upgrade_legacy_storeless_sinks(capture: &mut CapturedSchedule) {
    for item in &mut capture.items {
        if item.boundary.is_none()
            && item.outputs.is_single()
            && matches!(item.kernel.operation(), crate::Operation::Sink)
            && item.kernel.sources().is_empty()
        {
            item.boundary = Some(ScheduleBoundary::Unsupported(
                "operation requires materialization",
            ));
        }
    }
}

/// Validates the distinct inspection envelope without weakening the released
/// executable artifact invariant. This routine validates the complete output
/// inventory and then projects each item to its canonical primary descriptor
/// before applying the established single-output rules.
fn validate_scheduled_outputs(
    c: &CapturedSchedule,
    validate_keys: bool,
) -> Result<(), ArtifactError> {
    if c.items.len() > MAX_ITEMS
        || c.inputs.len() > MAX_BINDINGS
        || c.constants.len() > MAX_BINDINGS
        || c.quantized_constants.len() > MAX_BINDINGS
        || c.requested_passthroughs.len() > MAX_BINDINGS
    {
        return Err(ArtifactError::Format("schedule limit"));
    }
    if c.symbolic.is_some() && c.items.iter().any(|item| !item.outputs.is_single()) {
        return Err(ArtifactError::Unsupported);
    }
    if c.specialized_from.is_some() && c.items.iter().any(|item| !item.outputs.is_single()) {
        return Err(ArtifactError::Unsupported);
    }

    let mut output_ids = BTreeSet::new();
    for item in &c.items {
        if item.node.index() as u64 != item.primary_output().id {
            return Err(ArtifactError::Format("item identity"));
        }
        for output in item.outputs.iter() {
            validate_desc(output)?;
            if !output_ids.insert(output.id) {
                return Err(ArtifactError::Format("scheduled-output ownership"));
            }
        }
        if !item.is_effect()
            && item.input_bindings.iter().any(|binding| {
                item.outputs
                    .iter()
                    .any(|output| output.id == binding.desc.id)
            })
        {
            return Err(ArtifactError::Format("scheduled-output binding"));
        }
        if validate_keys {
            let expected = if let Some(provenance) = &c.specialized_from {
                super::specialized_item_cache_key(
                    item,
                    provenance.source_identity,
                    &provenance.bindings,
                )
            } else {
                super::item_cache_key(item)
            }
            .map_err(|_| ArtifactError::Format("item cache identity"))?;
            if item.cache_key != expected {
                return Err(ArtifactError::Format("item cache identity"));
            }
        }
    }

    let source_ids = c
        .inputs
        .iter()
        .map(|input| input.desc.id)
        .chain(c.constants.keys().copied())
        .collect::<BTreeSet<_>>();
    let passthrough_ids = c
        .requested_passthroughs
        .iter()
        .map(|passthrough| passthrough.requested.index() as u64)
        .collect::<BTreeSet<_>>();
    let requested_values = output_ids
        .iter()
        .copied()
        .chain(source_ids.iter().copied())
        .chain(passthrough_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    validate_requested_ids(&c.requested, &requested_values)?;
    if c.inputs
        .iter()
        .any(|input| output_ids.contains(&input.desc.id))
        || c.constants.keys().any(|id| output_ids.contains(id))
        || c.quantized_constants
            .keys()
            .any(|id| output_ids.contains(id))
    {
        return Err(ArtifactError::Format("scheduled-output external ownership"));
    }

    // Retain every established descriptor ABI, reciprocal DAG, state/effect,
    // constant, and symbolic validation rule through an exact primary-only
    // projection. Its cache keys must be projected too because legacy keys are
    // intentionally byte-for-byte unchanged for singleton items.
    let mut projected = c.clone();
    let primary_ids = projected
        .items
        .iter()
        .map(|item| item.primary_output().id)
        .collect::<BTreeSet<_>>();
    projected.requested.retain(|id| {
        primary_ids.contains(id) || source_ids.contains(id) || passthrough_ids.contains(id)
    });
    let provenance = projected.specialized_from.clone();
    for item in &mut projected.items {
        let primary = item.primary_output().clone();
        item.outputs = ScheduledOutputs::single(primary);
        item.cache_key = if let Some(provenance) = &provenance {
            super::specialized_item_cache_key(
                item,
                provenance.source_identity,
                &provenance.bindings,
            )
        } else {
            super::item_cache_key(item)
        }
        .map_err(|_| ArtifactError::Format("item cache identity"))?;
    }
    validate(&projected, true)?;

    let input_ids = c
        .inputs
        .iter()
        .map(|input| input.desc.id)
        .collect::<BTreeSet<_>>();
    let mut available = input_ids;
    available.extend(c.constants.keys().copied());
    available.extend(c.quantized_constants.keys().copied());
    for item in &c.items {
        if item
            .input_bindings
            .iter()
            .any(|binding| !available.contains(&binding.desc.id))
            || item
                .quantized_input_bindings
                .iter()
                .any(|binding| !available.contains(&(binding.input_node.index() as u64)))
        {
            return Err(ArtifactError::Format("unavailable binding"));
        }
        available.extend(item.outputs.iter().map(|output| output.id));
    }
    Ok(())
}

pub(crate) fn validate_for_replay(c: &CapturedSchedule) -> Result<(), ArtifactError> {
    validate_capture(c)?;
    if c.items
        .iter()
        .any(|item| matches!(item.kernel.operation(), crate::Operation::TensorGuard(_)))
    {
        return Err(ArtifactError::Unsupported);
    }
    if c.symbolic.is_some() {
        return Err(ArtifactError::Unsupported);
    }
    if c.items.iter().any(|x| x.boundary.is_some()) {
        return Err(ArtifactError::Unsupported);
    }
    if c.items.iter().any(|item| !item.outputs.is_single()) {
        return Err(ArtifactError::Unsupported);
    }
    Ok(())
}

pub(crate) fn validate_capture(c: &CapturedSchedule) -> Result<(), ArtifactError> {
    validate(c, true)?;
    if identity(c)? != c.identity {
        return Err(ArtifactError::Format("schedule identity"));
    }
    Ok(())
}

fn write_symbolic_schema(w: &mut Writer, schema: &SymbolicSchema) -> Result<(), ArtifactError> {
    write_symbolic_schema_core(w, schema, None)?;
    write_symbolic_schema_sidecars(w, schema, true)
}

#[cfg(test)]
fn write_symbolic_schema_v8(w: &mut Writer, schema: &SymbolicSchema) -> Result<(), ArtifactError> {
    write_symbolic_schema_core(w, schema, Some(0xfeed_face_dead_beef))?;
    write_symbolic_schema_sidecars(w, schema, false)
}

#[cfg(test)]
fn write_symbolic_schema_v9(w: &mut Writer, schema: &SymbolicSchema) -> Result<(), ArtifactError> {
    write_symbolic_schema_core(w, schema, None)?;
    write_symbolic_schema_sidecars(w, schema, false)
}

fn write_symbolic_schema_sidecars(
    w: &mut Writer,
    schema: &SymbolicSchema,
    projected: bool,
) -> Result<(), ArtifactError> {
    write_len(w, schema.views.len(), MAX_BINDINGS)?;
    for ((item, buffer), view) in &schema.views {
        w.u64(*item)?;
        w.u64(*buffer)?;
        write_symbolic_shape(w, &view.source_shape)?;
        write_symbolic_shape(w, &view.logical_shape)?;
        write_len(w, view.strides.len(), MAX_BINDINGS)?;
        for stride in &view.strides {
            write_symbolic(w, stride, 0)?;
        }
        write_symbolic(w, &view.offset, 0)?;
    }
    write_len(w, schema.splat_constants.len(), MAX_BINDINGS)?;
    for buffer in &schema.splat_constants {
        w.u64(*buffer)?;
    }
    if projected {
        write_len(w, schema.projected.len(), MAX_BINDINGS)?;
        for ((item, ordinal), map) in &schema.projected {
            w.u64(*item)?;
            w.u32(*ordinal)?;
            write_symbolic_shape(w, &map.source_shape)?;
            write_symbolic_shape(w, &map.output_shape)?;
            let mut nodes = 0;
            write_projected_expr(w, &map.expression, 0, &mut nodes)?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn write_symbolic_schema_v2(w: &mut Writer, schema: &SymbolicSchema) -> Result<(), ArtifactError> {
    write_symbolic_schema_core(w, schema, Some(0xfeed_face_dead_beef))
}

fn write_symbolic_schema_core(
    w: &mut Writer,
    schema: &SymbolicSchema,
    legacy_reduction_buffer: Option<u64>,
) -> Result<(), ArtifactError> {
    write_len(w, schema.parameters.len(), MAX_BINDINGS)?;
    for (parameter, template) in schema.parameters.iter().zip(&schema.template_values) {
        let variable = parameter.variable();
        let (min, max) = variable.bounds();
        w.u64(variable.id())?;
        w.string(variable.name())?;
        w.i64(min)?;
        w.i64(max)?;
        w.u8(dtype_tag(parameter.dtype()))?;
        w.i64(*template)?;
    }
    write_len(w, schema.guards.len(), MAX_BINDINGS)?;
    for guard in &schema.guards {
        match guard {
            SymbolicGuard::Equal { left, right } => {
                w.u8(0)?;
                write_symbolic(w, left, 0)?;
                write_symbolic(w, right, 0)?;
            }
            SymbolicGuard::Divisible { value, divisor } => {
                w.u8(1)?;
                write_symbolic(w, value, 0)?;
                w.u64(*divisor)?;
            }
        }
    }
    write_len(w, schema.buffer_shapes.len(), MAX_BINDINGS)?;
    for (buffer, shape) in &schema.buffer_shapes {
        w.u64(*buffer)?;
        write_symbolic_shape(w, shape)?;
    }
    write_len(w, schema.item_domains.len(), MAX_ITEMS)?;
    for (item, domain) in &schema.item_domains {
        w.u64(*item)?;
        match domain {
            SymbolicItemDomain::Elementwise { output } => {
                w.u8(0)?;
                write_symbolic_shape(w, output)?;
            }
            SymbolicItemDomain::Reduction {
                input,
                output,
                reduction,
            } => {
                w.u8(1)?;
                if let Some(buffer) = legacy_reduction_buffer {
                    w.u64(buffer)?;
                }
                write_symbolic_shape(w, input)?;
                write_symbolic_shape(w, output)?;
                write_symbolic_shape(w, reduction)?;
            }
            SymbolicItemDomain::Matmul {
                lhs_buffer,
                rhs_buffer,
                output,
                batch,
                m,
                n,
                k,
            } => {
                w.u8(2)?;
                w.u64(*lhs_buffer)?;
                w.u64(*rhs_buffer)?;
                write_symbolic_shape(w, output)?;
                write_symbolic_shape(w, batch)?;
                write_symbolic(w, m, 0)?;
                write_symbolic(w, n, 0)?;
                write_symbolic(w, k, 0)?;
            }
        }
    }
    Ok(())
}

fn read_symbolic_schema(
    r: &mut Reader<'_>,
    version: u8,
    legacy_reduction_buffer: bool,
    projected_sidecar: bool,
) -> Result<SymbolicSchema, ArtifactError> {
    let count = r.count(MAX_BINDINGS)?;
    let mut parameters = Vec::with_capacity(count);
    let mut template_values = Vec::with_capacity(count);
    for _ in 0..count {
        let variable = SymbolicVar::from_artifact(r.u64()?, r.string()?, r.i64()?, r.i64()?)
            .map_err(|_| ArtifactError::Format("symbolic parameter"))?;
        parameters.push(SymbolicParameter {
            variable,
            dtype: dtype(r.u8()?)?,
        });
        template_values.push(r.i64()?);
    }
    let count = r.count(MAX_BINDINGS)?;
    let mut guards = Vec::with_capacity(count);
    for _ in 0..count {
        guards.push(match r.u8()? {
            0 => SymbolicGuard::Equal {
                left: read_symbolic(r, 0)?,
                right: read_symbolic(r, 0)?,
            },
            1 => SymbolicGuard::Divisible {
                value: read_symbolic(r, 0)?,
                divisor: r.u64()?,
            },
            _ => return Err(ArtifactError::Format("symbolic guard tag")),
        });
    }
    let count = r.count(MAX_BINDINGS)?;
    let mut buffer_shapes = BTreeMap::new();
    for _ in 0..count {
        let id = r.u64()?;
        if buffer_shapes.insert(id, read_symbolic_shape(r)?).is_some() {
            return Err(ArtifactError::Format("duplicate symbolic buffer"));
        }
    }
    let count = r.count(MAX_ITEMS)?;
    let mut item_domains = BTreeMap::new();
    for _ in 0..count {
        let id = r.u64()?;
        let domain = match r.u8()? {
            0 => SymbolicItemDomain::Elementwise {
                output: read_symbolic_shape(r)?,
            },
            1 => {
                if legacy_reduction_buffer {
                    let _legacy_input_buffer = r.u64()?;
                }
                SymbolicItemDomain::Reduction {
                    input: read_symbolic_shape(r)?,
                    output: read_symbolic_shape(r)?,
                    reduction: read_symbolic_shape(r)?,
                }
            }
            2 => SymbolicItemDomain::Matmul {
                lhs_buffer: r.u64()?,
                rhs_buffer: r.u64()?,
                output: read_symbolic_shape(r)?,
                batch: read_symbolic_shape(r)?,
                m: read_symbolic(r, 0)?,
                n: read_symbolic(r, 0)?,
                k: read_symbolic(r, 0)?,
            },
            _ => return Err(ArtifactError::Format("symbolic item tag")),
        };
        if item_domains.insert(id, domain).is_some() {
            return Err(ArtifactError::Format("duplicate symbolic item"));
        }
    }
    let mut views = BTreeMap::new();
    let mut splat_constants = BTreeSet::new();
    let mut projected = BTreeMap::new();
    if version >= 3 {
        let count = r.count(MAX_BINDINGS)?;
        let mut previous = None;
        for _ in 0..count {
            let key = (r.u64()?, r.u64()?);
            if previous.is_some_and(|previous| key <= previous) {
                return Err(ArtifactError::Format("symbolic view order"));
            }
            previous = Some(key);
            let source_shape = read_symbolic_shape(r)?;
            let logical_shape = read_symbolic_shape(r)?;
            let stride_count = r.count(MAX_BINDINGS)?;
            let strides = (0..stride_count)
                .map(|_| read_symbolic(r, 0))
                .collect::<Result<Vec<_>, _>>()?;
            let view = SymbolicViewMap {
                source_shape,
                logical_shape,
                strides,
                offset: read_symbolic(r, 0)?,
            };
            if views.insert(key, view).is_some() {
                return Err(ArtifactError::Format("duplicate symbolic view"));
            }
        }
        let count = r.count(MAX_BINDINGS)?;
        let mut previous = None;
        for _ in 0..count {
            let buffer = r.u64()?;
            if previous.is_some_and(|previous| buffer <= previous) {
                return Err(ArtifactError::Format("symbolic constant order"));
            }
            previous = Some(buffer);
            splat_constants.insert(buffer);
        }
    }
    if projected_sidecar {
        let count = r.count(MAX_BINDINGS)?;
        let mut previous = None;
        for _ in 0..count {
            let key = (r.u64()?, r.u32()?);
            if previous.is_some_and(|previous| key <= previous) {
                return Err(ArtifactError::Format("symbolic projected order"));
            }
            previous = Some(key);
            let mut nodes = 0;
            let map = SymbolicProjectedIndexMap {
                source_shape: read_symbolic_shape(r)?,
                output_shape: read_symbolic_shape(r)?,
                expression: read_projected_expr(r, 0, &mut nodes)?,
            };
            if projected.insert(key, map).is_some() {
                return Err(ArtifactError::Format("duplicate symbolic projected index"));
            }
        }
    }
    Ok(SymbolicSchema {
        parameters,
        template_values,
        guards,
        buffer_shapes,
        item_domains,
        views,
        projected,
        splat_constants,
    })
}

fn write_projected_expr(
    w: &mut Writer,
    expression: &ProjectedExpr<SymbolicExpr>,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), ArtifactError> {
    *nodes = nodes
        .checked_add(1)
        .ok_or(ArtifactError::Format("symbolic projected nodes"))?;
    if depth > crate::projected_index::MAX_PROJECTED_INDEX_DEPTH
        || *nodes > crate::projected_index::MAX_PROJECTED_INDEX_NODES
    {
        return Err(ArtifactError::Format("symbolic projected depth"));
    }
    match expression {
        ProjectedExpr::Linear => w.u8(0),
        ProjectedExpr::Constant(value) => {
            w.u8(1)?;
            write_symbolic(w, value, 0)
        }
        ProjectedExpr::Binary {
            operation,
            lhs,
            rhs,
        } => {
            w.u8(2)?;
            w.u8(crate::uop::artifact::binary_tag(*operation))?;
            write_projected_expr(w, lhs, depth + 1, nodes)?;
            write_projected_expr(w, rhs, depth + 1, nodes)
        }
    }
}

fn read_projected_expr(
    r: &mut Reader<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<ProjectedExpr<SymbolicExpr>, ArtifactError> {
    *nodes = nodes
        .checked_add(1)
        .ok_or(ArtifactError::Format("symbolic projected nodes"))?;
    if depth > crate::projected_index::MAX_PROJECTED_INDEX_DEPTH
        || *nodes > crate::projected_index::MAX_PROJECTED_INDEX_NODES
    {
        return Err(ArtifactError::Format("symbolic projected depth"));
    }
    match r.u8()? {
        0 => Ok(ProjectedExpr::Linear),
        1 => Ok(ProjectedExpr::Constant(read_symbolic(r, 0)?)),
        2 => ProjectedExpr::binary(
            crate::uop::artifact::binary_from_tag(r.u8()?)?,
            read_projected_expr(r, depth + 1, nodes)?,
            read_projected_expr(r, depth + 1, nodes)?,
        )
        .map_err(|_| ArtifactError::Format("symbolic projected operation")),
        _ => Err(ArtifactError::Format("symbolic projected tag")),
    }
}

fn write_symbolic_shape(w: &mut Writer, shape: &SymbolicShape) -> Result<(), ArtifactError> {
    write_len(w, shape.rank(), MAX_BINDINGS)?;
    for dim in shape.dims() {
        write_symbolic(w, dim.expression(), 0)?;
    }
    Ok(())
}

fn read_symbolic_shape(r: &mut Reader<'_>) -> Result<SymbolicShape, ArtifactError> {
    let count = r.count(MAX_BINDINGS)?;
    let mut dims = Vec::with_capacity(count);
    for _ in 0..count {
        dims.push(SymbolicDim::new(read_symbolic(r, 0)?));
    }
    Ok(SymbolicShape::new(dims))
}

fn write_specialized_from(w: &mut Writer, value: &SpecializedFrom) -> Result<(), ArtifactError> {
    w.u64(value.source_identity)?;
    write_len(w, value.bindings.len(), MAX_BINDINGS)?;
    for (id, binding) in &value.bindings {
        w.u64(*id)?;
        w.i64(*binding)?;
    }
    Ok(())
}

fn read_specialized_from(r: &mut Reader<'_>) -> Result<SpecializedFrom, ArtifactError> {
    let source_identity = r.u64()?;
    let count = r.count(MAX_BINDINGS)?;
    let mut bindings = Vec::with_capacity(count);
    for _ in 0..count {
        bindings.push((r.u64()?, r.i64()?));
    }
    Ok(SpecializedFrom {
        source_identity,
        bindings,
    })
}

fn write_boundary(w: &mut Writer, x: Option<&ScheduleBoundary>) -> Result<(), ArtifactError> {
    write_boundary_inner(w, x, false)
}

fn write_boundary_inner(
    w: &mut Writer,
    x: Option<&ScheduleBoundary>,
    effects: bool,
) -> Result<(), ArtifactError> {
    match x {
        None => w.u8(0),
        Some(ScheduleBoundary::NonScalarUOpBridge) => w.u8(1),
        Some(ScheduleBoundary::Unsupported(s)) => {
            w.u8(2)?;
            w.string(s)
        }
        // Effects have no replay artifact contract yet.  Keep this fail-closed
        // even when a caller bypasses `CapturedSchedule::capture`.
        Some(ScheduleBoundary::Effect) if effects => w.u8(3),
        Some(ScheduleBoundary::Effect) => Err(ArtifactError::Unsupported),
    }
}
fn read_boundary_inner(
    r: &mut Reader<'_>,
    effects: bool,
) -> Result<Option<ScheduleBoundary>, ArtifactError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(ScheduleBoundary::NonScalarUOpBridge),
        2 => Some(ScheduleBoundary::Unsupported(match r.string()?.as_str() {
            "operation requires materialization" => "operation requires materialization",
            "shrink of a computed value requires materialization" => {
                "shrink of a computed value requires materialization"
            }
            "view of a computed value requires materialization" => {
                "view of a computed value requires materialization"
            }
            "product reductions are outside sum/mean lowering" => {
                "product reductions are outside sum/mean lowering"
            }
            "min/max reductions are outside sum/mean lowering" => {
                "min/max reductions are outside sum/mean lowering"
            }
            "operation is outside phase-one elementwise lowering" => {
                "operation is outside phase-one elementwise lowering"
            }
            _ => return Err(ArtifactError::Format("schedule boundary")),
        })),
        3 if effects => Some(ScheduleBoundary::Effect),
        _ => return Err(ArtifactError::Format("boundary tag")),
    })
}
fn write_len(w: &mut Writer, n: usize, max: usize) -> Result<(), ArtifactError> {
    if n > max || n > u32::MAX as usize {
        Err(ArtifactError::Format("schedule count"))
    } else {
        w.u32(n as u32)
    }
}
fn write_u64s(w: &mut Writer, xs: &[u64]) -> Result<(), ArtifactError> {
    write_len(w, xs.len(), MAX_BINDINGS)?;
    for x in xs {
        w.u64(*x)?;
    }
    Ok(())
}
fn read_u64s(r: &mut Reader<'_>) -> Result<Vec<u64>, ArtifactError> {
    let n = r.count(MAX_BINDINGS)?;
    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        xs.push(r.u64()?);
    }
    Ok(xs)
}
fn node(x: u64) -> Result<NodeId, ArtifactError> {
    Ok(NodeId::from_index(
        usize::try_from(x).map_err(|_| ArtifactError::Format("node id"))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Graph, Scalar, Shape, Slice, TensorData};

    fn decode_hex(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn unchecked(capture: &CapturedSchedule) -> Vec<u8> {
        unchecked_with_identity(capture, identity(capture).unwrap())
    }

    fn unchecked_with_identity(capture: &CapturedSchedule, stored_identity: u64) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(MAGIC).unwrap();
        w.u8(VERSION).unwrap();
        w.u64(stored_identity).unwrap();
        write_payload(&mut w, capture).unwrap();
        let sum = checksum(&w.out);
        w.u32(sum).unwrap();
        w.out
    }

    fn legacy_scheduled_outputs_v2(capture: &CapturedSchedule) -> Vec<u8> {
        let mut payload = Writer::new();
        write_scheduled_outputs_payload_v2(&mut payload, capture).unwrap();
        let identity = fnv1a64(&payload.out);
        let mut writer = Writer::new();
        writer.bytes(MULTI_MAGIC).unwrap();
        writer.u8(2).unwrap();
        writer.u64(identity).unwrap();
        writer.bytes(&payload.out).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_scheduled_outputs_v3(capture: &CapturedSchedule) -> Vec<u8> {
        let mut payload = Writer::new();
        write_scheduled_outputs_payload_v3(&mut payload, capture).unwrap();
        let identity = fnv1a64(&payload.out);
        let mut writer = Writer::new();
        writer.bytes(MULTI_MAGIC).unwrap();
        writer.u8(3).unwrap();
        writer.u64(identity).unwrap();
        writer.bytes(&payload.out).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_scheduled_outputs_v4(capture: &CapturedSchedule) -> Vec<u8> {
        let mut payload = Writer::new();
        write_scheduled_outputs_payload_v4(&mut payload, capture).unwrap();
        let identity = fnv1a64(&payload.out);
        let mut writer = Writer::new();
        writer.bytes(MULTI_MAGIC).unwrap();
        writer.u8(4).unwrap();
        writer.u64(identity).unwrap();
        writer.bytes(&payload.out).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_v1(capture: &CapturedSchedule) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.bytes(MAGIC).unwrap();
        writer.u8(1).unwrap();
        writer.u64(identity_v1(capture).unwrap()).unwrap();
        write_payload_v1(&mut writer, capture).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_v2(capture: &CapturedSchedule) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.bytes(MAGIC).unwrap();
        writer.u8(2).unwrap();
        writer.u64(identity_v2(capture).unwrap()).unwrap();
        write_payload_v2(&mut writer, capture).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_v5(capture: &CapturedSchedule) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.bytes(MAGIC).unwrap();
        writer.u8(5).unwrap();
        writer.u64(identity_v5(capture).unwrap()).unwrap();
        write_payload_v5(&mut writer, capture).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_v7(capture: &CapturedSchedule) -> Vec<u8> {
        let mut payload = Writer::new();
        write_payload_v7(&mut payload, capture).unwrap();
        let identity = fnv1a64(&payload.out);
        let mut writer = Writer::new();
        writer.bytes(MAGIC).unwrap();
        writer.u8(7).unwrap();
        writer.u64(identity).unwrap();
        writer.bytes(&payload.out).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_v8(capture: &CapturedSchedule) -> Vec<u8> {
        let mut payload = Writer::new();
        write_payload_v8(&mut payload, capture).unwrap();
        let identity = fnv1a64(&payload.out);
        let mut writer = Writer::new();
        writer.bytes(MAGIC).unwrap();
        writer.u8(8).unwrap();
        writer.u64(identity).unwrap();
        writer.bytes(&payload.out).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn legacy_v9(capture: &CapturedSchedule) -> Vec<u8> {
        let mut payload = Writer::new();
        write_payload_v9(&mut payload, capture).unwrap();
        let identity = fnv1a64(&payload.out);
        let mut writer = Writer::new();
        writer.bytes(MAGIC).unwrap();
        writer.u8(9).unwrap();
        writer.u64(identity).unwrap();
        writer.bytes(&payload.out).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        writer.out
    }

    fn fixture() -> CapturedSchedule {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2]), DType::F32);
        let y = graph.square(x).unwrap();
        let schedule = crate::schedule(&graph, y).unwrap();
        CapturedSchedule::capture(&graph, &schedule, &[y]).unwrap()
    }

    fn symbolic_fixture() -> CapturedSchedule {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let output = graph.square(input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([(
            input,
            SymbolicShape::new(vec![extent.clone().into()]),
        )]))
        .with_guard(crate::SymbolicGuard::divisible(extent, 2).unwrap());
        CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &spec,
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap()
    }

    fn symbolic_view_fixture() -> CapturedSchedule {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3, 4], DType::F32);
        let view = graph.shrink(input, [(0, 3), (1, 4)]).unwrap();
        let output = graph.neg(view).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![extent.into(), 4usize.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 3)]),
        )
        .unwrap()
    }

    fn symbolic_reduction_fixture() -> CapturedSchedule {
        let rows = crate::SymbolicExpr::variable("rows", 0, 8).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 4], DType::F32);
        let output = graph
            .reduce(input, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                SymbolicShape::new(vec![rows.into(), 4usize.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap()
    }

    #[test]
    fn captured_elementwise_bytes_are_deterministic() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2]), DType::F32);
        let y = graph.square(x).unwrap();
        let schedule = crate::schedule(&graph, y).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[y]).unwrap();
        let bytes = encode(&capture).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(bytes, encode(&decoded).unwrap());
        assert_eq!(decoded.requested, capture.requested);
        let provided = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars([2], DType::F32, [Scalar::F(2.0), Scalar::F(-3.0)]).unwrap(),
        )]);
        assert_eq!(
            decoded.replay(&provided).unwrap()[0].storage(),
            capture.replay(&provided).unwrap()[0].storage()
        );
        let upgraded = decode(&legacy_v1(&capture)).unwrap();
        assert!(!upgraded.is_symbolic());
        assert_eq!(
            upgraded.replay(&provided).unwrap()[0].storage(),
            decoded.replay(&provided).unwrap()[0].storage()
        );
        assert_eq!(encode(&upgraded).unwrap()[4], VERSION);
        let upgraded_v2 = decode(&legacy_v2(&capture)).unwrap();
        assert_eq!(encode(&upgraded_v2).unwrap()[4], VERSION);
        let upgraded_v5 = decode(&legacy_v5(&capture)).unwrap();
        assert_eq!(encode(&upgraded_v5).unwrap()[4], VERSION);
        let upgraded_v7 = decode(&legacy_v7(&capture)).unwrap();
        assert!(upgraded_v7.requested_passthroughs.is_empty());
        assert_eq!(upgraded_v7.identity, identity(&upgraded_v7).unwrap());
        let reencoded_v7 = encode(&upgraded_v7).unwrap();
        assert_eq!(reencoded_v7[4], VERSION);
        assert_eq!(
            encode(&decode(&reencoded_v7).unwrap()).unwrap(),
            reencoded_v7
        );
    }

    #[test]
    fn frozen_v6_opaque_keys_authenticate_then_upgrade_to_current_identity() {
        // Frozen independently from the v6 writer. The historical cache key
        // 0x8877665544332211 is deliberately not reproducible by the current
        // codec; it remains covered by the v6 envelope identity and checksum.
        let bytes = decode_hex(concat!(
            "524753410618f1debba4c2a6f201000000000000000000000000000000000000",
            "0000000000000000000000000000000000000000000000000001000000000000",
            "0000000000000000000b0400000000000000040000000000000000001c000000",
            "52475541110100000000000000000000001d0000000000003d6ee4a600112233",
            "44556677880000000000000000010000000000000000000000000000000000e3",
            "12ed6b"
        ));
        let upgraded = decode(&bytes).unwrap();
        assert_eq!(upgraded.items.len(), 1);
        assert_eq!(
            upgraded.items[0].boundary,
            Some(ScheduleBoundary::Unsupported(
                "operation requires materialization"
            ))
        );
        assert!(upgraded.replay(&BTreeMap::new()).is_err());
        assert_ne!(upgraded.items[0].cache_key, 0x8877_6655_4433_2211);
        assert_eq!(
            upgraded.items[0].cache_key,
            super::super::item_cache_key(&upgraded.items[0]).unwrap()
        );
        assert_ne!(upgraded.identity, 0xf2a6_c2a4_bbde_f118);

        let current = encode(&upgraded).unwrap();
        assert_eq!(current[4], VERSION);
        let decoded = decode(&current).unwrap();
        assert_eq!(decoded.identity, upgraded.identity);
        assert_eq!(decoded.items[0].cache_key, upgraded.items[0].cache_key);
        assert_eq!(decoded.items[0].boundary, upgraded.items[0].boundary);
        assert!(decoded.replay(&BTreeMap::new()).is_err());
        assert_eq!(encode(&decoded).unwrap(), current);

        let mut forged = bytes;
        forged[5] ^= 1;
        let body = forged.len() - 4;
        let sum = checksum(&forged[..body]);
        forged[body..].copy_from_slice(&sum.to_le_bytes());
        assert!(matches!(
            decode(&forged),
            Err(ArtifactError::Format("schedule identity"))
        ));
    }

    #[test]
    fn signed_affine_views_round_trip_in_executable_artifacts() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::I64);
        let output = graph
            .stride(
                input,
                [
                    Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                    Slice {
                        start: None,
                        stop: None,
                        step: 2,
                    },
                ],
            )
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let bytes = encode(&capture).unwrap();
        assert_eq!(bytes[4], VERSION);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(encode(&decoded).unwrap(), bytes);
        assert!(decoded.items.is_empty());
        let view = decoded.requested_passthroughs[0]
            .desc
            .view
            .as_ref()
            .expect("stride passthrough retains its affine view");
        assert_eq!(view.logical_shape.dims(), &[2, 2]);
        assert_eq!(view.strides, vec![-3, 2]);
        assert_eq!(view.offset, 3);
        assert!(validate_for_replay(&decoded).is_ok());
    }

    #[test]
    fn requested_passthrough_ownership_remains_authenticated() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::I32);
        let alternate = graph.input_dtype("alternate", [2, 3], DType::I32);
        let output = graph.permute(input, [1, 0]).unwrap();
        let requested = [output, input, alternate];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &requested).unwrap();
        let bytes = encode(&capture).unwrap();
        assert_eq!(bytes[4], VERSION);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(
            decoded.requested_passthroughs,
            capture.requested_passthroughs
        );
        assert_eq!(encode(&decoded).unwrap(), bytes);
        let ordered = encode_scheduled_outputs(&capture).unwrap();
        assert_eq!(ordered[4], MULTI_VERSION);
        let ordered = decode_scheduled_outputs(&ordered).unwrap();
        assert_eq!(
            ordered.requested_passthroughs,
            capture.requested_passthroughs
        );

        let mut stale_ownership = capture.clone();
        stale_ownership.requested_passthroughs[0].source = alternate;
        stale_ownership.requested_passthroughs[0].desc.id = alternate.index() as u64;
        let tampered = unchecked_with_identity(&stale_ownership, capture.identity);
        assert!(matches!(
            decode(&tampered),
            Err(ArtifactError::Format("schedule identity"))
        ));

        let mut out_of_bounds = capture;
        out_of_bounds.requested_passthroughs[0]
            .desc
            .view
            .as_mut()
            .unwrap()
            .offset = i64::MAX;
        assert!(matches!(
            identity(&out_of_bounds),
            Err(ArtifactError::Format("buffer descriptor"))
        ));
        assert!(encode(&out_of_bounds).is_err());
    }

    #[test]
    fn ordered_outputs_round_trip_but_replay_fails_closed() {
        let mut capture = fixture();
        let item = &mut capture.items[0];
        let primary = item.primary_output().clone();
        let mut secondary = primary.clone();
        secondary.id = 99;
        item.outputs = ScheduledOutputs::new(vec![primary, secondary.clone()]).unwrap();
        item.cache_key = crate::schedule::item_cache_key(item).unwrap();
        capture.identity = identity(&capture).unwrap();

        let decoded = decode(&encode(&capture).unwrap()).unwrap();
        assert_eq!(decoded.items[0].outputs.iter().count(), 2);
        assert_eq!(decoded.items[0].outputs.iter().nth(1), Some(&secondary));
        assert!(validate_for_replay(&decoded).is_err());
    }

    #[test]
    fn malformed_dependencies_bindings_and_limits_are_rejected_on_decode() {
        let capture = fixture();

        let mut dependency = capture.clone();
        dependency.items[0].dependencies.push(0);
        assert!(decode(&unchecked(&dependency)).is_err());

        let mut binding = capture.clone();
        binding.items[0].input_bindings[0].abi_index = 7;
        assert!(decode(&unchecked(&binding)).is_err());

        // Serialize a valid artifact first, then corrupt its output descriptor
        // in place. The canonical writer correctly refuses malformed
        // descriptors, so routing this case through `unchecked` would test
        // encode-time rejection rather than decoder validation.
        let mut descriptor = encode(&capture).unwrap();
        let body = descriptor.len() - 4;
        let mut output = Writer::new();
        write_desc_inner(&mut output, capture.items[0].primary_output(), true).unwrap();
        let output_start = descriptor[..body]
            .windows(output.out.len())
            .rposition(|window| window == output.out)
            .unwrap();
        let bytes = capture.items[0].primary_output().bytes.to_le_bytes();
        let bytes_offset = output
            .out
            .windows(bytes.len())
            .position(|window| window == bytes)
            .unwrap();
        descriptor[output_start + bytes_offset] ^= 1;
        let sum = checksum(&descriptor[..body]);
        descriptor[body..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&descriptor).is_err());

        let mut unused = capture;
        let mut extra = unused.inputs[0].clone();
        extra.name = "unused".into();
        extra.node = NodeId::from_index(999);
        extra.desc.id = 999;
        unused.inputs.push(extra);
        assert!(decode(&unchecked(&unused)).is_err());

        let mut limit = Writer::new();
        limit.bytes(MAGIC).unwrap();
        limit.u8(VERSION).unwrap();
        limit.u64(1).unwrap();
        limit.u32(MAX_ITEMS as u32 + 1).unwrap();
        let sum = checksum(&limit.out);
        limit.u32(sum).unwrap();
        assert!(decode(&limit.out).is_err());
    }

    #[test]
    fn malformed_symbolic_schema_is_rejected_during_decode() {
        let capture = symbolic_fixture();
        let bytes = encode(&capture).unwrap();
        assert_eq!(bytes, encode(&decode(&bytes).unwrap()).unwrap());

        let mut wrong_dtype = capture.clone();
        wrong_dtype.symbolic.as_mut().unwrap().parameters[0].dtype = DType::F32;
        assert!(decode(&unchecked(&wrong_dtype)).is_err());

        let mut bad_template = capture.clone();
        bad_template.symbolic.as_mut().unwrap().template_values[0] = 3;
        assert!(decode(&unchecked(&bad_template)).is_err());

        let mut missing_shape = capture.clone();
        let output = missing_shape.items[0].primary_output().id;
        missing_shape
            .symbolic
            .as_mut()
            .unwrap()
            .buffer_shapes
            .remove(&output);
        assert!(decode(&unchecked(&missing_shape)).is_err());

        let mut zero_divisor = capture;
        let crate::SymbolicGuard::Divisible { divisor, .. } =
            &mut zero_divisor.symbolic.as_mut().unwrap().guards[0]
        else {
            unreachable!()
        };
        *divisor = 0;
        assert!(decode(&unchecked(&zero_divisor)).is_err());

        let legacy = symbolic_fixture();
        let upgraded = decode(&legacy_v2(&legacy)).unwrap();
        assert!(upgraded.is_symbolic());
        assert_eq!(encode(&upgraded).unwrap()[4], VERSION);
    }

    #[test]
    fn historical_v8_symbolic_reduction_discards_buffer_word_and_reencodes_current() {
        let capture = symbolic_reduction_fixture();
        let historical = legacy_v8(&capture);
        assert_eq!(historical[4], 8);
        let upgraded = decode(&historical).unwrap();
        assert_eq!(upgraded.symbolic, capture.symbolic);
        let current = encode(&upgraded).unwrap();
        assert_eq!(current[4], VERSION);
        assert_eq!(encode(&decode(&current).unwrap()).unwrap(), current);
        assert_ne!(historical, current);
    }

    #[test]
    fn historical_v9_symbolic_schema_rekeys_and_reencodes_v10() {
        let capture = symbolic_fixture();
        let historical = legacy_v9(&capture);
        assert_eq!(historical[4], 9);
        let upgraded = decode(&historical).unwrap();
        assert_eq!(upgraded.symbolic, capture.symbolic);
        assert_ne!(
            upgraded.identity,
            u64::from_le_bytes(historical[5..13].try_into().unwrap())
        );
        let current = encode(&upgraded).unwrap();
        assert_eq!(current[4], VERSION);
        assert_eq!(encode(&decode(&current).unwrap()).unwrap(), current);
    }

    #[test]
    fn historical_rgso_v3_symbolic_reduction_rekeys_and_reencodes_current() {
        let capture = symbolic_reduction_fixture();
        let historical = legacy_scheduled_outputs_v3(&capture);
        assert_eq!(historical[4], 3);
        let historical_identity = u64::from_le_bytes(historical[5..13].try_into().unwrap());
        let upgraded = decode_scheduled_outputs(&historical).unwrap();
        assert_eq!(upgraded.symbolic, capture.symbolic);
        assert_eq!(upgraded.items[0].cache_key, capture.items[0].cache_key);
        assert_ne!(upgraded.identity, historical_identity);

        let current = encode_scheduled_outputs(&upgraded).unwrap();
        assert_eq!(current[4], MULTI_VERSION);
        assert_eq!(
            u64::from_le_bytes(current[5..13].try_into().unwrap()),
            upgraded.identity
        );
        assert_eq!(
            encode_scheduled_outputs(&decode_scheduled_outputs(&current).unwrap()).unwrap(),
            current
        );
    }

    #[test]
    fn historical_rgso_v4_symbolic_schema_rekeys_and_reencodes_v5() {
        let capture = symbolic_fixture();
        let historical = legacy_scheduled_outputs_v4(&capture);
        assert_eq!(historical[4], 4);
        let upgraded = decode_scheduled_outputs(&historical).unwrap();
        assert_eq!(upgraded.symbolic, capture.symbolic);
        let current = encode_scheduled_outputs(&upgraded).unwrap();
        assert_eq!(current[4], MULTI_VERSION);
        assert_eq!(
            encode_scheduled_outputs(&decode_scheduled_outputs(&current).unwrap()).unwrap(),
            current
        );
    }

    #[test]
    fn symbolic_specialization_keys_are_canonical_and_binding_sensitive() {
        let symbolic = symbolic_fixture();
        let variable = symbolic.symbolic_parameters()[0].variable().id();
        let first =
            crate::engine::symbolic::specialize_capture(&symbolic, &[(variable, 4)]).unwrap();
        let repeated =
            crate::engine::symbolic::specialize_capture(&symbolic, &[(variable, 4)]).unwrap();
        let different =
            crate::engine::symbolic::specialize_capture(&symbolic, &[(variable, 6)]).unwrap();
        assert_eq!(first.items[0].cache_key, repeated.items[0].cache_key);
        assert_ne!(first.items[0].cache_key, different.items[0].cache_key);
        assert_eq!(first.identity, repeated.identity);
        assert_ne!(first.identity, different.identity);
        let bytes = encode(&first).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.identity, first.identity);
        assert_eq!(decoded.items[0].cache_key, first.items[0].cache_key);
        assert_eq!(encode(&decoded).unwrap(), bytes);
    }

    #[test]
    fn malformed_symbolic_views_and_constant_policies_fail_closed() {
        let capture = symbolic_view_fixture();
        let bytes = encode(&capture).unwrap();
        assert_eq!(bytes, encode(&decode(&bytes).unwrap()).unwrap());

        let mut missing = capture.clone();
        missing.symbolic.as_mut().unwrap().views.clear();
        assert!(decode(&unchecked(&missing)).is_err());

        let mut offset = capture.clone();
        offset
            .symbolic
            .as_mut()
            .unwrap()
            .views
            .values_mut()
            .next()
            .unwrap()
            .offset = crate::SymbolicExpr::constant(i64::MAX);
        assert!(decode(&unchecked(&offset)).is_err());

        let mut stride = capture.clone();
        stride
            .symbolic
            .as_mut()
            .unwrap()
            .views
            .values_mut()
            .next()
            .unwrap()
            .strides[0] = crate::SymbolicExpr::constant(i64::MAX);
        assert!(decode(&unchecked(&stride)).is_err());

        let mut extra_symbol = capture.clone();
        extra_symbol
            .symbolic
            .as_mut()
            .unwrap()
            .views
            .values_mut()
            .next()
            .unwrap()
            .offset = crate::SymbolicExpr::variable("unknown", 0, 0).unwrap();
        assert!(decode(&unchecked(&extra_symbol)).is_err());

        let mut unknown_constant = capture;
        unknown_constant
            .symbolic
            .as_mut()
            .unwrap()
            .splat_constants
            .insert(999);
        assert!(decode(&unchecked(&unknown_constant)).is_err());
    }

    #[test]
    fn scheduled_output_envelope_is_distinct_canonical_and_inspection_only() {
        let capture = fixture();
        let legacy = encode(&capture).unwrap();
        let single = encode_scheduled_outputs(&capture).unwrap();
        assert_eq!(single, encode_scheduled_outputs(&capture).unwrap());
        assert_ne!(single, legacy);
        assert_eq!(&single[..4], MULTI_MAGIC);
        assert!(decode(&single).is_err());
        assert!(decode_scheduled_outputs(&legacy).is_err());
        let decoded = decode_scheduled_outputs(&single).unwrap();
        assert_eq!(single, encode_scheduled_outputs(&decoded).unwrap());
        assert_eq!(legacy, encode(&capture).unwrap());

        let mut opaque = capture.clone();
        opaque.items[0].cache_key = 0x8877_6655_4433_2211;
        let v2 = legacy_scheduled_outputs_v2(&opaque);
        let upgraded = decode_scheduled_outputs(&v2).unwrap();
        assert_eq!(
            encode_scheduled_outputs(&upgraded).unwrap()[4],
            MULTI_VERSION
        );
        assert_ne!(upgraded.items[0].cache_key, opaque.items[0].cache_key);

        let mut multi = capture.clone();
        let mut secondary = multi.items[0].primary_output().clone();
        secondary.id = secondary.id.checked_add(1).unwrap();
        multi.items[0].outputs =
            ScheduledOutputs::new(vec![multi.items[0].primary_output().clone(), secondary])
                .unwrap();
        multi.items[0].cache_key = super::super::item_cache_key(&multi.items[0]).unwrap();
        let bytes = encode_scheduled_outputs(&multi).unwrap();
        let decoded = decode_scheduled_outputs(&bytes).unwrap();
        assert_eq!(bytes, encode_scheduled_outputs(&decoded).unwrap());
        assert_eq!(decoded.items[0].outputs.len(), 2);
        assert!(decoded.replay(&BTreeMap::new()).is_err());
    }

    #[test]
    fn scheduled_output_envelope_rejects_projection_and_identity_tampering() {
        let mut capture = fixture();
        let mut secondary = capture.items[0].primary_output().clone();
        secondary.id = secondary.id.checked_add(1).unwrap();
        capture.items[0].outputs =
            ScheduledOutputs::new(vec![capture.items[0].primary_output().clone(), secondary])
                .unwrap();
        capture.items[0].cache_key = super::super::item_cache_key(&capture.items[0]).unwrap();

        // RGSO retains the historical primary descriptor only inside its
        // codec. Corrupt that private projection independently of the
        // canonical output list and prove the decoder still rejects it.
        let mut bad_projection = encode_scheduled_outputs(&capture).unwrap();
        let body = bad_projection.len() - 4;
        let mut marker = Writer::new();
        write_desc_inner(&mut marker, capture.items[0].primary_output(), false).unwrap();
        write_len(&mut marker, capture.items[0].outputs.len(), MAX_BINDINGS).unwrap();
        write_desc_inner(&mut marker, capture.items[0].primary_output(), false).unwrap();
        let projection_start = bad_projection[..body]
            .windows(marker.out.len())
            .position(|window| window == marker.out)
            .unwrap();
        bad_projection[projection_start] ^= 1;
        let sum = checksum(&bad_projection[..body]);
        bad_projection[body..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode_scheduled_outputs(&bad_projection).is_err());

        assert!(
            ScheduledOutputs::new(vec![
                capture.items[0].primary_output().clone(),
                capture.items[0].primary_output().clone(),
            ])
            .is_err()
        );

        let mut bad_identity = encode_scheduled_outputs(&capture).unwrap();
        bad_identity[5..13].copy_from_slice(&0u64.to_le_bytes());
        let body = bad_identity.len() - 4;
        let sum = checksum(&bad_identity[..body]);
        bad_identity[body..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode_scheduled_outputs(&bad_identity).is_err());
    }
}
