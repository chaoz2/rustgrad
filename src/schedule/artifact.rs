//! Portable executable schedule descriptors and bindings.
use super::{BufferDesc, ScheduleBoundary, ScheduleInputBinding, ScheduleItem};
use crate::engine::symbolic::{
    SpecializedFrom, SymbolicGuard, SymbolicItemDomain, SymbolicParameter, SymbolicSchema,
};
use crate::tensor::artifact as tensor_artifact;
use crate::uop::artifact::{
    ArtifactError, Reader, Writer, checksum, decode as decode_uop, dtype, dtype_tag,
    encode as encode_uop, read_shape, read_symbolic, read_view, validate_view, write_shape,
    write_symbolic, write_view,
};
use crate::{CapturedSchedule, NodeId, ReplayInput, SymbolicDim, SymbolicShape, SymbolicVar};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 4] = b"RGSA";
const VERSION: u8 = 2;
const MAX_ARTIFACT_BYTES: usize = 64 << 20;
const MAX_ITEMS: usize = 1 << 16;
const MAX_BINDINGS: usize = 1 << 16;

pub fn encode(capture: &CapturedSchedule) -> Result<Vec<u8>, ArtifactError> {
    validate(capture, false)?;
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
    if !matches!(version, 1 | VERSION) {
        return Err(ArtifactError::Format("schedule version"));
    }
    let stored_identity = r.u64()?;
    let mut capture = read_payload(&mut r, stored_identity, version)?;
    if !r.done() {
        return Err(ArtifactError::Format("schedule trailing bytes"));
    }
    validate(&capture, true)?;
    let decoded_identity = if version == 1 {
        identity_v1(&capture)?
    } else {
        identity(&capture)?
    };
    if decoded_identity != stored_identity {
        return Err(ArtifactError::Format("schedule identity"));
    }
    if version == 1 {
        capture.identity = identity(&capture)?;
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
    Ok(w.out.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    }))
}

fn write_payload(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_payload_v1(w, c)?;
    w.bool(c.symbolic.is_some())?;
    if let Some(schema) = &c.symbolic {
        write_symbolic_schema(w, schema)?;
    }
    w.bool(c.specialized_from.is_some())?;
    if let Some(provenance) = &c.specialized_from {
        write_specialized_from(w, provenance)?;
    }
    Ok(())
}

fn write_payload_v1(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
    write_len(w, c.items.len(), MAX_ITEMS)?;
    for item in &c.items {
        write_item(w, item)?;
    }
    write_len(w, c.inputs.len(), MAX_BINDINGS)?;
    for input in &c.inputs {
        w.string(&input.name)?;
        w.u64(input.node.index() as u64)?;
        write_desc(w, &input.desc)?;
    }
    write_len(w, c.constants.len(), MAX_BINDINGS)?;
    for (id, value) in &c.constants {
        w.u64(*id)?;
        tensor_artifact::encode_into(w, value)?;
    }
    write_u64s(w, &c.requested)
}

fn read_payload(
    r: &mut Reader<'_>,
    identity: u64,
    version: u8,
) -> Result<CapturedSchedule, ArtifactError> {
    let n = r.count(MAX_ITEMS)?;
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        items.push(read_item(r)?);
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        inputs.push(ReplayInput {
            name: r.string()?,
            node: node(r.u64()?)?,
            desc: read_desc(r)?,
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
        Some(read_symbolic_schema(r)?)
    } else {
        None
    };
    let specialized_from = if version >= 2 && r.bool()? {
        Some(read_specialized_from(r)?)
    } else {
        None
    };
    Ok(CapturedSchedule {
        items,
        inputs,
        constants,
        requested,
        identity,
        symbolic,
        specialized_from,
    })
}

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
    Ok(writer.out.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    }))
}

fn write_item(w: &mut Writer, x: &ScheduleItem) -> Result<(), ArtifactError> {
    w.u64(x.id)?;
    w.u64(x.node.index() as u64)?;
    write_u64s(w, &x.dependencies)?;
    write_u64s(w, &x.consumers)?;
    write_len(w, x.inputs.len(), MAX_BINDINGS)?;
    for desc in &x.inputs {
        write_desc(w, desc)?;
    }
    write_len(w, x.input_bindings.len(), MAX_BINDINGS)?;
    for binding in &x.input_bindings {
        w.u64(binding.input_node.index() as u64)?;
        write_desc(w, &binding.desc)?;
        w.usize(binding.abi_index)?;
    }
    write_len(w, x.external_materializations.len(), MAX_BINDINGS)?;
    for id in &x.external_materializations {
        w.u64(id.index() as u64)?;
    }
    write_desc(w, &x.output)?;
    let kernel = encode_uop(&x.kernel)?;
    write_len(w, kernel.len(), MAX_ARTIFACT_BYTES)?;
    w.bytes(&kernel)?;
    write_boundary(w, x.boundary.as_ref())?;
    w.u64(x.cache_key)
}

fn read_item(r: &mut Reader<'_>) -> Result<ScheduleItem, ArtifactError> {
    let id = r.u64()?;
    let item_node = node(r.u64()?)?;
    let dependencies = read_u64s(r)?;
    let consumers = read_u64s(r)?;
    let n = r.count(MAX_BINDINGS)?;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        inputs.push(read_desc(r)?);
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut input_bindings = Vec::with_capacity(n);
    for _ in 0..n {
        input_bindings.push(ScheduleInputBinding {
            input_node: node(r.u64()?)?,
            desc: read_desc(r)?,
            abi_index: r.usize()?,
        });
    }
    let n = r.count(MAX_BINDINGS)?;
    let mut external_materializations = Vec::with_capacity(n);
    for _ in 0..n {
        external_materializations.push(node(r.u64()?)?);
    }
    let output = read_desc(r)?;
    let kernel_len = r.count(MAX_ARTIFACT_BYTES)?;
    let kernel = decode_uop(r.take(kernel_len)?)?;
    let boundary = read_boundary(r)?;
    let cache_key = r.u64()?;
    Ok(ScheduleItem {
        id,
        node: item_node,
        dependencies,
        consumers,
        inputs,
        input_bindings,
        external_materializations,
        output,
        kernel,
        boundary,
        cache_key,
    })
}

fn write_desc(w: &mut Writer, x: &BufferDesc) -> Result<(), ArtifactError> {
    validate_desc(x)?;
    w.u64(x.id)?;
    write_shape(w, &x.shape)?;
    w.u8(dtype_tag(x.dtype))?;
    w.usize(x.bytes)?;
    w.usize(x.alignment)?;
    w.bool(x.read_only)?;
    w.bool(x.view.is_some())?;
    if let Some(view) = &x.view {
        write_view(w, view)?;
    }
    Ok(())
}
fn read_desc(r: &mut Reader<'_>) -> Result<BufferDesc, ArtifactError> {
    let x = BufferDesc {
        id: r.u64()?,
        shape: read_shape(r)?,
        dtype: dtype(r.u8()?)?,
        bytes: r.usize()?,
        alignment: r.usize()?,
        read_only: r.bool()?,
        view: if r.bool()? { Some(read_view(r)?) } else { None },
    };
    validate_desc(&x)?;
    Ok(x)
}
fn validate_desc(x: &BufferDesc) -> Result<(), ArtifactError> {
    let bytes = x
        .shape
        .numel()
        .ok()
        .and_then(|n| n.checked_mul(x.dtype.itemsize()))
        .ok_or(ArtifactError::Format("buffer size"))?;
    if bytes != x.bytes || x.alignment == 0 || !x.alignment.is_power_of_two() {
        return Err(ArtifactError::Format("buffer descriptor"));
    }
    if let Some(view) = &x.view {
        validate_view(view)?;
        if view.source_shape != x.shape {
            return Err(ArtifactError::Format("buffer view"));
        }
    }
    Ok(())
}

fn validate(c: &CapturedSchedule, decoded: bool) -> Result<(), ArtifactError> {
    if c.items.len() > MAX_ITEMS
        || c.inputs.len() > MAX_BINDINGS
        || c.constants.len() > MAX_BINDINGS
    {
        return Err(ArtifactError::Format("schedule limit"));
    }
    let count = c.items.len() as u64;
    let mut output_ids = BTreeSet::new();
    for (index, item) in c.items.iter().enumerate() {
        if item.id != index as u64
            || item.node.index() as u64 != item.output.id
            || !output_ids.insert(item.output.id)
        {
            return Err(ArtifactError::Format("item identity"));
        }
        validate_desc(&item.output)?;
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
        let mut inventory = BTreeSet::new();
        for desc in &item.inputs {
            validate_desc(desc)?;
            if !inventory.insert(desc.id) {
                return Err(ArtifactError::Format("input inventory"));
            }
        }
        if item.boundary.is_none()
            && super::input_bindings(&item.kernel, &item.inputs, &item.output)
                .map_err(|_| ArtifactError::Format("kernel resources"))?
                != item.input_bindings
        {
            return Err(ArtifactError::Format("kernel bindings"));
        }
        let expected_cache_key = if let Some(provenance) = &c.specialized_from {
            super::specialized_item_cache_key(
                item,
                provenance.source_identity,
                &provenance.bindings,
            )
        } else {
            super::item_cache_key(item)
        };
        if item.cache_key != expected_cache_key {
            return Err(ArtifactError::Format("item cache identity"));
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
    for (id, value) in &c.constants {
        if value.shape().numel().is_err() || value.len() != value.shape().numel().unwrap() {
            return Err(ArtifactError::Format("constant tensor"));
        }
        let desc = c
            .items
            .iter()
            .flat_map(|x| &x.input_bindings)
            .find(|x| x.desc.id == *id)
            .map(|x| &x.desc)
            .ok_or(ArtifactError::Format("unbound constant"))?;
        if value.shape() != &desc.shape || value.dtype() != desc.dtype {
            return Err(ArtifactError::Format("constant descriptor"));
        }
    }
    let mut available = input_ids;
    available.extend(c.constants.keys().copied());
    for item in &c.items {
        if item
            .input_bindings
            .iter()
            .any(|x| !available.contains(&x.desc.id))
        {
            return Err(ArtifactError::Format("unavailable binding"));
        }
        available.insert(item.output.id);
    }
    let used = c
        .items
        .iter()
        .flat_map(|x| x.input_bindings.iter().map(|x| x.desc.id))
        .collect::<BTreeSet<_>>();
    if c.inputs.iter().any(|x| !used.contains(&x.desc.id)) {
        return Err(ArtifactError::Format("unused replay input"));
    }
    let mut requested = BTreeSet::new();
    let outputs = c.items.iter().map(|x| x.output.id).collect::<BTreeSet<_>>();
    if c.requested
        .iter()
        .any(|x| !requested.insert(*x) || !outputs.contains(x))
    {
        return Err(ArtifactError::Format("requested output"));
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
    let _ = decoded;
    Ok(())
}

pub(crate) fn validate_for_replay(c: &CapturedSchedule) -> Result<(), ArtifactError> {
    validate_capture(c)?;
    if c.symbolic.is_some() {
        return Err(ArtifactError::Unsupported);
    }
    if c.items.iter().any(|x| x.boundary.is_some()) {
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
                input_buffer,
                input,
                output,
                reduction,
            } => {
                w.u8(1)?;
                w.u64(*input_buffer)?;
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

fn read_symbolic_schema(r: &mut Reader<'_>) -> Result<SymbolicSchema, ArtifactError> {
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
            1 => SymbolicItemDomain::Reduction {
                input_buffer: r.u64()?,
                input: read_symbolic_shape(r)?,
                output: read_symbolic_shape(r)?,
                reduction: read_symbolic_shape(r)?,
            },
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
    Ok(SymbolicSchema {
        parameters,
        template_values,
        guards,
        buffer_shapes,
        item_domains,
    })
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
    match x {
        None => w.u8(0),
        Some(ScheduleBoundary::NonScalarUOpBridge) => w.u8(1),
        Some(ScheduleBoundary::Unsupported(s)) => {
            w.u8(2)?;
            w.string(s)
        }
    }
}
fn read_boundary(r: &mut Reader<'_>) -> Result<Option<ScheduleBoundary>, ArtifactError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(ScheduleBoundary::NonScalarUOpBridge),
        2 => Some(ScheduleBoundary::Unsupported(match r.string()?.as_str() {
            "operation requires materialization" => "operation requires materialization",
            "shrink of a computed value requires materialization" => {
                "shrink of a computed value requires materialization"
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
    use crate::{DType, Graph, Scalar, Shape, TensorData};

    fn unchecked(capture: &CapturedSchedule) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(MAGIC).unwrap();
        w.u8(VERSION).unwrap();
        w.u64(identity(capture).unwrap()).unwrap();
        write_payload(&mut w, capture).unwrap();
        let sum = checksum(&w.out);
        w.u32(sum).unwrap();
        w.out
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
        let output = missing_shape.items[0].output.id;
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
    }
}
