//! Immutable capture boundary for mixed pure/effect schedules.
//!
//! This type intentionally owns only logical schedule/state metadata. Runtime
//! leases, slot generations, pointers, and current bytes remain caller-owned.
use crate::uop::artifact::{ArtifactError, Reader, Writer, checksum};
use crate::{
    BufferDesc, BufferState, CapturedSchedule, NodeId, ReplayError, ReplayInput, Schedule,
    ScheduleStateBinding, ScheduleValueBinding,
};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 4] = b"RGSM";
const VERSION: u8 = 1;
const MAX_BYTES: usize = 64 << 20;
const MAX_ITEMS: usize = 1 << 16;
const MAX_BINDINGS: usize = 1 << 16;

/// Graph-free mixed-schedule descriptor. The ordinary capture remains its
/// canonical typed UOp/item payload; persistent identities are stored beside
/// it so an effect runtime must prove the declared versions at replay time.
#[derive(Clone, Debug)]
pub struct CapturedMixedSchedule {
    pub schedule: CapturedSchedule,
    pub value_bindings: Vec<ScheduleValueBinding>,
    pub state_bindings: Vec<ScheduleStateBinding>,
    pub states: Vec<BufferState>,
}

impl CapturedMixedSchedule {
    /// Constructs the logical mixed boundary after complete schedule
    /// validation. Serialization/replay are deliberately separate so neither
    /// can consult a Graph or an EffectRuntime during capture.
    pub fn from_parts(
        schedule: CapturedSchedule,
        mixed: &Schedule,
        states: Vec<BufferState>,
    ) -> Result<Self, ReplayError> {
        mixed
            .validate()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        if !mixed.items.iter().any(crate::ScheduleItem::is_effect) {
            return Err(ReplayError::Unsupported(
                "mixed capture has no effects".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for state in &states {
            if !ids.insert((state.buffer, state.version)) {
                return Err(ReplayError::Corrupt(
                    "duplicate logical state version".into(),
                ));
            }
        }
        Ok(Self {
            schedule,
            value_bindings: mixed.value_bindings.clone(),
            state_bindings: mixed.state_bindings.clone(),
            states,
        })
    }

    /// Bounded deterministic RGSM encoding. It intentionally excludes every
    /// runtime lease, slot, pointer, generation, and current buffer byte.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        validate(self)?;
        let mut w = Writer::new();
        w.bytes(MAGIC).map_err(codec)?;
        w.u8(VERSION).map_err(codec)?;
        let identity = identity(self)?;
        w.u64(identity).map_err(codec)?;
        write_len(&mut w, self.schedule.items.len())?;
        for item in &self.schedule.items {
            crate::schedule::artifact::write_effect_item(&mut w, item).map_err(codec)?;
        }
        write_inputs(&mut w, &self.schedule.inputs)?;
        write_constants(&mut w, &self.schedule.constants)?;
        write_u64s(&mut w, &self.schedule.requested)?;
        write_value_bindings(&mut w, &self.value_bindings)?;
        write_state_bindings(&mut w, &self.state_bindings)?;
        write_states(&mut w, &self.states)?;
        if w.out.len().checked_add(4).is_none_or(|n| n > MAX_BYTES) {
            return Err(ReplayError::Corrupt("RGSM byte limit".into()));
        }
        let sum = checksum(&w.out);
        w.u32(sum).map_err(codec)?;
        Ok(w.out)
    }

    /// Decodes and validates every logical relationship before returning a
    /// replayable descriptor. No EffectRuntime is touched at this boundary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        if bytes.len() < 17 || bytes.len() > MAX_BYTES {
            return Err(ReplayError::Corrupt("RGSM length".into()));
        }
        let body = bytes.len() - 4;
        let got = u32::from_le_bytes(
            bytes[body..]
                .try_into()
                .map_err(|_| ReplayError::Corrupt("RGSM checksum".into()))?,
        );
        if checksum(&bytes[..body]) != got {
            return Err(ReplayError::Corrupt("RGSM checksum".into()));
        }
        let mut r = Reader::new(&bytes[..body]);
        if r.take(4).map_err(codec)? != MAGIC {
            return Err(ReplayError::Corrupt("RGSM magic".into()));
        }
        if r.u8().map_err(codec)? != VERSION {
            return Err(ReplayError::Corrupt("RGSM version".into()));
        }
        let stored_identity = r.u64().map_err(codec)?;
        let count = r.count(MAX_ITEMS).map_err(codec)?;
        if count == 0 {
            return Err(ReplayError::Corrupt("RGSM item count".into()));
        }
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(crate::schedule::artifact::read_effect_item(&mut r).map_err(codec)?);
        }
        let inputs = read_inputs(&mut r)?;
        let constants = read_constants(&mut r)?;
        let requested = read_u64s(&mut r)?;
        let value_bindings = read_value_bindings(&mut r)?;
        let state_bindings = read_state_bindings(&mut r)?;
        let states = read_states(&mut r)?;
        if !r.done() {
            return Err(ReplayError::Corrupt("RGSM trailing bytes".into()));
        }
        let mut schedule = CapturedSchedule {
            items,
            inputs,
            constants,
            quantized_constants: BTreeMap::new(),
            requested,
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        let decoded = Self {
            schedule: schedule.clone(),
            value_bindings,
            state_bindings,
            states,
        };
        validate(&decoded)?;
        let actual = identity(&decoded)?;
        if actual != stored_identity {
            return Err(ReplayError::Corrupt("RGSM identity".into()));
        }
        schedule.identity = actual;
        Ok(Self {
            schedule,
            ..decoded
        })
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::{DType, EffectGraph, Shape, Storage, TensorData, schedule_effects};

    fn captured_effect() -> CapturedMixedSchedule {
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                40,
                TensorData::from_storage([2], Storage::F16(vec![0x8000, 0x7e01])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                41,
                TensorData::from_storage([2], Storage::F16(vec![1, 2])).unwrap(),
            )
            .unwrap();
        let next = effects.assign(&target, &source).unwrap();
        let schedule = schedule_effects(&effects).unwrap();
        let capture = CapturedSchedule {
            items: schedule.items,
            inputs: vec![],
            constants: BTreeMap::new(),
            quantized_constants: BTreeMap::new(),
            requested: vec![],
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        CapturedMixedSchedule::from_parts(
            capture,
            &schedule_effects(&effects).unwrap(),
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn rgsm_round_trips_typed_store_after_payloads() {
        let captured = captured_effect();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(bytes, captured.to_bytes().unwrap());
        let decoded = CapturedMixedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.schedule.items.len(), 1);
        assert!(decoded.schedule.items[0].is_effect());
        assert_eq!(decoded.states, captured.states);
        assert!(matches!(
            decoded.schedule.items[0].kernel.sources()[0].kind(),
            crate::UOpKind::EffectStore
        ));
        assert!(crate::uop::artifact::encode(&decoded.schedule.items[0].kernel).is_err());
        let _ = (DType::F16, Shape::from([2]));
    }

    #[test]
    fn rgsm_rejects_corrupt_envelope_before_decode() {
        let bytes = captured_effect().to_bytes().unwrap();
        for mut bad in [
            {
                let mut x = bytes.clone();
                x[0] ^= 1;
                x
            },
            {
                let mut x = bytes.clone();
                x[4] = 0;
                x
            },
            {
                let mut x = bytes.clone();
                let last = x.len() - 1;
                x[last] ^= 1;
                x
            },
            bytes[..bytes.len() - 1].to_vec(),
        ] {
            assert!(CapturedMixedSchedule::from_bytes(&bad).is_err());
            bad.clear();
        }
    }

    #[test]
    fn rgsm_round_trips_value_and_versioned_state_sidecars() {
        use crate::{
            AffineView, BinaryOp, Graph, ScheduleStateBinding, ScheduleValueBinding,
            bind_schedule_states, combine_mixed_schedules, schedule,
        };
        let mut graph = Graph::new();
        let state_input = graph.input_dtype("state", [2], DType::F32);
        let bias = graph.input_dtype("bias", [2], DType::F32);
        let sum = graph.binary(BinaryOp::Add, state_input, bias).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let capture = CapturedSchedule::capture(&graph, &pure, &[sum]).unwrap();
        let state_input_binding = pure.items[0]
            .input_bindings
            .iter()
            .find(|binding| binding.input_node == state_input)
            .unwrap()
            .clone();
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                100,
                TensorData::from_storage([2], Storage::F32(vec![0.0; 2])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                sum.index() as u64,
                TensorData::from_storage([2], Storage::F32(vec![0.0; 2])).unwrap(),
            )
            .unwrap();
        let next = effects.assign(&target, &source).unwrap();
        let pure = bind_schedule_states(
            pure,
            vec![ScheduleStateBinding {
                state: target.state().clone(),
                view: Some(AffineView::identity(Shape::from([2])).flip(0).unwrap()),
                consumer_item: 0,
                consumer_node: sum,
                input_node: state_input,
                desc: state_input_binding.desc,
                abi_index: state_input_binding.abi_index,
            }],
        )
        .unwrap();
        let binding = ScheduleValueBinding {
            producer_item: 0,
            producer_node: sum,
            producer_output: pure.items[0].output.clone(),
            abi_index: 0,
            effect_item: 0,
            source_position: 0,
        };
        let mixed =
            combine_mixed_schedules(pure, schedule_effects(&effects).unwrap(), vec![binding])
                .unwrap();
        let mut capture = capture;
        capture.items = mixed.items.clone();
        let value = CapturedMixedSchedule::from_parts(
            capture,
            &mixed,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap();
        let decoded = CapturedMixedSchedule::from_bytes(&value.to_bytes().unwrap()).unwrap();
        assert_eq!(decoded.value_bindings, mixed.value_bindings);
        assert_eq!(decoded.state_bindings, mixed.state_bindings);
    }
}

fn codec(error: ArtifactError) -> ReplayError {
    ReplayError::Corrupt(error.to_string())
}

fn write_len(w: &mut Writer, n: usize) -> Result<(), ReplayError> {
    if n > MAX_BINDINGS || n > u32::MAX as usize {
        return Err(ReplayError::Corrupt("RGSM count".into()));
    }
    w.u32(n as u32).map_err(codec)
}
fn node(raw: u64) -> Result<NodeId, ReplayError> {
    Ok(NodeId::from_index(
        usize::try_from(raw).map_err(|_| ReplayError::Corrupt("RGSM node".into()))?,
    ))
}
fn write_inputs(w: &mut Writer, inputs: &[ReplayInput]) -> Result<(), ReplayError> {
    write_len(w, inputs.len())?;
    for input in inputs {
        w.string(&input.name).map_err(codec)?;
        w.u64(input.node.index() as u64).map_err(codec)?;
        crate::schedule::artifact::write_effect_desc(w, &input.desc).map_err(codec)?;
    }
    Ok(())
}
fn read_inputs(r: &mut Reader<'_>) -> Result<Vec<ReplayInput>, ReplayError> {
    let n = r.count(MAX_BINDINGS).map_err(codec)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ReplayInput {
            name: r.string().map_err(codec)?,
            node: node(r.u64().map_err(codec)?)?,
            desc: crate::schedule::artifact::read_effect_desc(r).map_err(codec)?,
        });
    }
    Ok(out)
}
fn write_constants(
    w: &mut Writer,
    constants: &BTreeMap<u64, crate::TensorData>,
) -> Result<(), ReplayError> {
    write_len(w, constants.len())?;
    for (id, value) in constants {
        w.u64(*id).map_err(codec)?;
        crate::tensor::artifact::encode_into(w, value).map_err(codec)?;
    }
    Ok(())
}
fn read_constants(r: &mut Reader<'_>) -> Result<BTreeMap<u64, crate::TensorData>, ReplayError> {
    let n = r.count(MAX_BINDINGS).map_err(codec)?;
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let id = r.u64().map_err(codec)?;
        if out
            .insert(id, crate::tensor::artifact::decode_from(r).map_err(codec)?)
            .is_some()
        {
            return Err(ReplayError::Corrupt("RGSM duplicate constant".into()));
        }
    }
    Ok(out)
}
fn write_u64s(w: &mut Writer, values: &[u64]) -> Result<(), ReplayError> {
    write_len(w, values.len())?;
    for value in values {
        w.u64(*value).map_err(codec)?;
    }
    Ok(())
}
fn read_u64s(r: &mut Reader<'_>) -> Result<Vec<u64>, ReplayError> {
    let n = r.count(MAX_BINDINGS).map_err(codec)?;
    (0..n).map(|_| r.u64().map_err(codec)).collect()
}
fn write_desc(w: &mut Writer, desc: &BufferDesc) -> Result<(), ReplayError> {
    crate::schedule::artifact::write_effect_desc(w, desc).map_err(codec)
}
fn read_desc(r: &mut Reader<'_>) -> Result<BufferDesc, ReplayError> {
    crate::schedule::artifact::read_effect_desc(r).map_err(codec)
}
fn write_value_bindings(w: &mut Writer, xs: &[ScheduleValueBinding]) -> Result<(), ReplayError> {
    write_len(w, xs.len())?;
    for x in xs {
        w.u64(x.producer_item).map_err(codec)?;
        w.u64(x.producer_node.index() as u64).map_err(codec)?;
        write_desc(w, &x.producer_output)?;
        w.usize(x.abi_index).map_err(codec)?;
        w.u64(x.effect_item).map_err(codec)?;
        w.usize(x.source_position).map_err(codec)?;
    }
    Ok(())
}
fn read_value_bindings(r: &mut Reader<'_>) -> Result<Vec<ScheduleValueBinding>, ReplayError> {
    let n = r.count(MAX_BINDINGS).map_err(codec)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ScheduleValueBinding {
            producer_item: r.u64().map_err(codec)?,
            producer_node: node(r.u64().map_err(codec)?)?,
            producer_output: read_desc(r)?,
            abi_index: r.usize().map_err(codec)?,
            effect_item: r.u64().map_err(codec)?,
            source_position: r.usize().map_err(codec)?,
        });
    }
    Ok(out)
}
fn write_state_bindings(w: &mut Writer, xs: &[ScheduleStateBinding]) -> Result<(), ReplayError> {
    write_len(w, xs.len())?;
    for x in xs {
        crate::uop::artifact::write_buffer_state(w, &x.state).map_err(codec)?;
        w.bool(x.view.is_some()).map_err(codec)?;
        if let Some(view) = &x.view {
            crate::uop::artifact::write_affine_view(w, view).map_err(codec)?;
        }
        w.u64(x.consumer_item).map_err(codec)?;
        w.u64(x.consumer_node.index() as u64).map_err(codec)?;
        w.u64(x.input_node.index() as u64).map_err(codec)?;
        write_desc(w, &x.desc)?;
        w.usize(x.abi_index).map_err(codec)?;
    }
    Ok(())
}
fn read_state_bindings(r: &mut Reader<'_>) -> Result<Vec<ScheduleStateBinding>, ReplayError> {
    let n = r.count(MAX_BINDINGS).map_err(codec)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ScheduleStateBinding {
            state: crate::uop::artifact::read_buffer_state(r).map_err(codec)?,
            view: if r.bool().map_err(codec)? {
                Some(crate::uop::artifact::read_affine_view(r).map_err(codec)?)
            } else {
                None
            },
            consumer_item: r.u64().map_err(codec)?,
            consumer_node: node(r.u64().map_err(codec)?)?,
            input_node: node(r.u64().map_err(codec)?)?,
            desc: read_desc(r)?,
            abi_index: r.usize().map_err(codec)?,
        });
    }
    Ok(out)
}
fn write_states(w: &mut Writer, xs: &[BufferState]) -> Result<(), ReplayError> {
    write_len(w, xs.len())?;
    for x in xs {
        crate::uop::artifact::write_buffer_state(w, x).map_err(codec)?;
    }
    Ok(())
}
fn read_states(r: &mut Reader<'_>) -> Result<Vec<BufferState>, ReplayError> {
    let n = r.count(MAX_BINDINGS).map_err(codec)?;
    (0..n)
        .map(|_| crate::uop::artifact::read_buffer_state(r).map_err(codec))
        .collect()
}

fn validate(value: &CapturedMixedSchedule) -> Result<(), ReplayError> {
    let schedule = Schedule {
        items: value.schedule.items.clone(),
        value_bindings: value.value_bindings.clone(),
        state_bindings: value.state_bindings.clone(),
    };
    schedule
        .validate()
        .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
    if !schedule.items.iter().any(crate::ScheduleItem::is_effect) {
        return Err(ReplayError::Unsupported(
            "mixed capture has no effects".into(),
        ));
    }
    if value.schedule.items.len() > MAX_ITEMS
        || value.value_bindings.len() > MAX_BINDINGS
        || value.state_bindings.len() > MAX_BINDINGS
        || value.states.len() > MAX_BINDINGS
    {
        return Err(ReplayError::Corrupt("RGSM count".into()));
    }
    let mut states = BTreeSet::new();
    for state in &value.states {
        crate::effects::validate_buffer_state(state)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        if !states.insert((state.buffer, state.version)) {
            return Err(ReplayError::Corrupt(
                "duplicate logical state version".into(),
            ));
        }
    }
    if value
        .state_bindings
        .iter()
        .any(|binding| !states.contains(&(binding.state.buffer, binding.state.version)))
    {
        return Err(ReplayError::Corrupt("unlisted state binding".into()));
    }
    if value
        .schedule
        .inputs
        .iter()
        .any(|input| input.name.is_empty())
    {
        return Err(ReplayError::Corrupt("empty replay input".into()));
    }
    Ok(())
}
fn identity(value: &CapturedMixedSchedule) -> Result<u64, ReplayError> {
    let mut clone = value.clone();
    clone.schedule.identity = 0;
    let bytes = clone.to_bytes_without_identity()?;
    Ok(bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    }))
}
impl CapturedMixedSchedule {
    fn to_bytes_without_identity(&self) -> Result<Vec<u8>, ReplayError> {
        let mut w = Writer::new();
        write_len(&mut w, self.schedule.items.len())?;
        for item in &self.schedule.items {
            crate::schedule::artifact::write_effect_item(&mut w, item).map_err(codec)?;
        }
        write_inputs(&mut w, &self.schedule.inputs)?;
        write_constants(&mut w, &self.schedule.constants)?;
        write_u64s(&mut w, &self.schedule.requested)?;
        write_value_bindings(&mut w, &self.value_bindings)?;
        write_state_bindings(&mut w, &self.state_bindings)?;
        write_states(&mut w, &self.states)?;
        Ok(w.out)
    }
}
