//! Portable executable schedule descriptors and bindings.
use super::{BufferDesc, ScheduleBoundary, ScheduleInputBinding, ScheduleItem};
use crate::tensor::artifact as tensor_artifact;
use crate::uop::artifact::{
    ArtifactError, Reader, Writer, checksum, decode as decode_uop, dtype, dtype_tag,
    encode as encode_uop, read_shape, read_view, validate_view, write_shape, write_view,
};
use crate::{CapturedSchedule, NodeId, ReplayInput};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 4] = b"RGSA";
const VERSION: u8 = 1;
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
    if r.u8()? != VERSION {
        return Err(ArtifactError::Format("schedule version"));
    }
    let stored_identity = r.u64()?;
    let capture = read_payload(&mut r, stored_identity)?;
    if !r.done() {
        return Err(ArtifactError::Format("schedule trailing bytes"));
    }
    validate(&capture, true)?;
    if identity(&capture)? != stored_identity {
        return Err(ArtifactError::Format("schedule identity"));
    }
    Ok(capture)
}

pub(crate) fn identity(capture: &CapturedSchedule) -> Result<u64, ArtifactError> {
    let mut w = Writer::new();
    write_payload(&mut w, capture)?;
    Ok(w.out.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    }))
}

fn write_payload(w: &mut Writer, c: &CapturedSchedule) -> Result<(), ArtifactError> {
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

fn read_payload(r: &mut Reader<'_>, identity: u64) -> Result<CapturedSchedule, ArtifactError> {
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
    Ok(CapturedSchedule {
        items,
        inputs,
        constants,
        requested: read_u64s(r)?,
        identity,
    })
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
    let _ = decoded;
    Ok(())
}

pub(crate) fn validate_for_replay(c: &CapturedSchedule) -> Result<(), ArtifactError> {
    validate_capture(c)?;
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

    fn fixture() -> CapturedSchedule {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2]), DType::F32);
        let y = graph.square(x).unwrap();
        let schedule = crate::schedule(&graph, y).unwrap();
        CapturedSchedule::capture(&graph, &schedule, &[y]).unwrap()
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
}
