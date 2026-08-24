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
const VERSION: u8 = 2;
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

/// Detached pure outputs and committed logical states from one graph-free
/// mixed replay. No runtime lease or view is returned to the caller.
#[derive(Clone, Debug)]
pub struct MixedReplayResult {
    pub outputs: Vec<crate::TensorData>,
    pub committed: Vec<BufferState>,
    /// Present only when the pure prefix ran through strict native replay.
    /// This is a logical cache/trace identity: it deliberately contains no
    /// runtime slot, generation, pointer, or current storage byte.
    pub native_trace: Option<NativeMixedReplayTrace>,
}

/// Stable logical identity of a strict-native mixed replay. The native JIT
/// retains ownership of compiled-item reuse; this trace binds that reuse to
/// the decoded RGSM schema without creating a second cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMixedReplayTrace {
    pub identity: u64,
    pub artifact_identity: u64,
    pub vectorized: bool,
    pub pure_item_cache_keys: Vec<u64>,
}

impl CapturedMixedSchedule {
    pub(crate) fn initial_states(&self) -> impl Iterator<Item = &BufferState> {
        self.states.iter().filter(|state| state.version == 0)
    }

    /// Stages this capture against caller-owned detached candidates. This is
    /// deliberately the shared batch seam: it never observes a runtime lease
    /// and never commits a persistent write.
    pub(crate) fn stage_interpreter(
        &self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        provided: &BTreeMap<String, crate::TensorData>,
    ) -> Result<crate::EffectBatchEntry, ReplayError> {
        self.stage(candidates, starts, provided, None)
    }

    pub(crate) fn stage_native(
        &self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        provided: &BTreeMap<String, crate::TensorData>,
        executor: &super::captured_replay::CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<crate::EffectBatchEntry, ReplayError> {
        self.stage(candidates, starts, provided, Some((executor, vectorized)))
    }

    fn stage(
        &self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        provided: &BTreeMap<String, crate::TensorData>,
        native: Option<(&super::captured_replay::CapturedReplayExecutor, bool)>,
    ) -> Result<crate::EffectBatchEntry, ReplayError> {
        validate(self)?;
        let schedule = Schedule {
            items: self.schedule.items.clone(),
            value_bindings: self.value_bindings.clone(),
            state_bindings: self.state_bindings.clone(),
        };
        let split = schedule
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
        if schedule.items[split..].iter().any(|item| !item.is_effect()) {
            return Err(ReplayError::Unsupported(
                "mixed replay requires ordered pure then effect items".into(),
            ));
        }
        let rebase = |state: &BufferState| -> Result<BufferState, ReplayError> {
            let start = starts
                .get(&state.buffer)
                .ok_or_else(|| ReplayError::Missing(state.buffer.to_string()))?;
            if start.shape != state.shape
                || start.dtype != state.dtype
                || start.bytes != state.bytes
            {
                return Err(ReplayError::Descriptor(
                    "batch state descriptor mismatch".into(),
                ));
            }
            Ok(BufferState {
                buffer: state.buffer,
                version: start
                    .version
                    .checked_add(state.version)
                    .ok_or_else(|| ReplayError::Corrupt("batch version overflow".into()))?,
                shape: state.shape.clone(),
                dtype: state.dtype,
                bytes: state.bytes,
            })
        };
        let mut pure = self.schedule.clone();
        pure.items.truncate(split);
        pure.requested = self
            .value_bindings
            .iter()
            .map(|b| b.producer_output.id)
            .collect();
        pure.identity = 0;
        let mut inputs = provided.clone();
        for binding in &self.state_bindings {
            let input = self
                .schedule
                .inputs
                .iter()
                .find(|x| x.node == binding.input_node)
                .ok_or_else(|| ReplayError::Corrupt("state input ABI is absent".into()))?;
            if inputs.contains_key(&input.name) {
                return Err(ReplayError::Descriptor(
                    "external input shadows persistent state binding".into(),
                ));
            }
            let state = rebase(&binding.state)?;
            let value = candidates
                .get(&state)
                .ok_or_else(|| ReplayError::Missing("batch state candidate".into()))?;
            let value = match &binding.view {
                Some(view) => value.affine_read(view),
                None => Ok(value.clone()),
            }
            .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
            if value.shape() != &binding.desc.shape || value.dtype() != binding.desc.dtype {
                return Err(ReplayError::Descriptor(
                    "batch state input descriptor mismatch".into(),
                ));
            }
            inputs.insert(input.name.clone(), value);
        }
        let values = match native {
            Some((executor, vectorized)) => {
                super::captured_replay::replay_native_items(&pure, &inputs, executor, vectorized)?
            }
            None => super::captured_replay::replay_interpreter_items(&pure, &inputs)?,
        };
        let plan = effect_plan(&schedule)?;
        let mut sources = BTreeMap::new();
        for binding in &self.value_bindings {
            let payload = effect_payload(&schedule.items[binding.effect_item as usize])?;
            sources.insert(
                payload.step,
                values
                    .get(&binding.producer_output.id)
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing(binding.producer_output.id.to_string()))?,
            );
        }
        let entry = crate::EffectBatchEntry {
            plan,
            starts,
            sources,
        };
        let batch = crate::EffectBatch::new(vec![entry.clone()])
            .map_err(|e| ReplayError::Execute(format!("batch stage: {e:?}")))?;
        for rebased in batch
            .rebased_steps()
            .map_err(|e| ReplayError::Execute(format!("batch stage: {e:?}")))?
        {
            let target = candidates
                .get(&rebased.step.reads[0])
                .cloned()
                .ok_or_else(|| ReplayError::Missing("batch target candidate".into()))?;
            let source = match rebased.source {
                Some(value) => value,
                None => candidates
                    .get(&rebased.step.reads[1])
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing("batch source candidate".into()))?,
            };
            let mut next = target;
            if let Some(view) = &rebased.step.target_view {
                next.assign_view_from(view, &source)
            } else if let Some(plan) = &rebased.step.index_plan {
                next.static_index_update_from(plan, &source)
            } else {
                next.assign_from(&source)
            }
            .map_err(|e| ReplayError::Execute(format!("batch stage: {e}")))?;
            candidates.insert(rebased.step.write, next);
        }
        Ok(entry)
    }
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
        let version = r.u8().map_err(codec)?;
        if !(1..=VERSION).contains(&version) {
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
        // v1's effect-aware UOp stream predates the optional normalized index
        // payload flag. Its bytes cannot be re-emitted byte-for-byte by v2,
        // so retain the validated v1 envelope while assigning the upgraded
        // canonical identity on decode. v2 remains self-authenticating.
        if version == VERSION && actual != stored_identity {
            return Err(ReplayError::Corrupt("RGSM identity".into()));
        }
        schedule.identity = actual;
        Ok(Self {
            schedule,
            ..decoded
        })
    }

    /// Replays a decoded mixed artifact against caller-owned persistent state.
    /// All input/state/topology checks happen before the pool-wide commit; the
    /// interpreter owns only temporary pure values and cannot rebind runtime
    /// identities or consult a mutable graph/registry.
    pub fn replay(
        &self,
        runtime: &mut crate::EffectRuntime,
        provided: &BTreeMap<String, crate::TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<MixedReplayResult, ReplayError> {
        validate(self)?;
        let schedule = Schedule {
            items: self.schedule.items.clone(),
            value_bindings: self.value_bindings.clone(),
            state_bindings: self.state_bindings.clone(),
        };
        let split = schedule
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
        if schedule.items[split..].iter().any(|item| !item.is_effect()) {
            return Err(ReplayError::Unsupported(
                "mixed replay requires ordered pure then effect items".into(),
            ));
        }
        let mut pure_capture = self.schedule.clone();
        pure_capture.items.truncate(split);
        pure_capture.requested = self
            .value_bindings
            .iter()
            .map(|binding| binding.producer_output.id)
            .collect();
        pure_capture.identity = 0;

        let mut inputs = provided.clone();
        for binding in &self.state_bindings {
            let input = self
                .schedule
                .inputs
                .iter()
                .find(|input| input.node == binding.input_node)
                .ok_or_else(|| ReplayError::Corrupt("state input ABI is absent".into()))?;
            if provided.contains_key(&input.name) {
                return Err(ReplayError::Descriptor(
                    "external input shadows persistent state binding".into(),
                ));
            }
            let snapshot = runtime.snapshot(&binding.state).map_err(|error| {
                ReplayError::Execute(format!("persistent state preflight: {error:?}"))
            })?;
            let value = match &binding.view {
                Some(view) => snapshot.tensor().affine_read(view),
                None => Ok(snapshot.tensor().clone()),
            }
            .map_err(|error| ReplayError::Descriptor(format!("persistent affine read: {error}")))?;
            if value.shape() != &binding.desc.shape
                || value.dtype() != binding.desc.dtype
                || value.len().checked_mul(value.dtype().itemsize()) != Some(binding.desc.bytes)
            {
                return Err(ReplayError::Descriptor(
                    "persistent state input descriptor mismatch".into(),
                ));
            }
            inputs.insert(input.name.clone(), value);
        }
        let values = super::captured_replay::replay_interpreter_items(&pure_capture, &inputs)?;
        let outputs = self
            .schedule
            .requested
            .iter()
            .map(|id| {
                values
                    .get(id)
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing(id.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = effect_plan(&schedule)?;
        let pure_sources = self
            .value_bindings
            .iter()
            .map(|binding| binding.effect_item)
            .collect::<BTreeSet<_>>();
        let mut required_states = self
            .state_bindings
            .iter()
            .map(|binding| binding.state.clone())
            .collect::<BTreeSet<_>>();
        for item in schedule.items.iter().filter(|item| item.is_effect()) {
            let payload = effect_payload(item)?;
            required_states.insert(payload.snapshot.clone());
            if !pure_sources.contains(&item.id) {
                required_states.insert(payload.source.clone());
            }
        }
        for state in &required_states {
            runtime.snapshot(state).map_err(|error| {
                ReplayError::Execute(format!("persistent state preflight: {error:?}"))
            })?;
        }
        let mut sources = BTreeMap::new();
        for binding in &self.value_bindings {
            let payload = effect_payload(&schedule.items[binding.effect_item as usize])?;
            let value = values
                .get(&binding.producer_output.id)
                .ok_or_else(|| ReplayError::Missing(binding.producer_output.id.to_string()))?
                .clone();
            sources.insert(payload.step, value);
        }
        let committed = runtime
            .execute_with_sources(&plan, &sources, injected_failure)
            .map_err(|error| ReplayError::Execute(format!("persistent mixed replay: {error:?}")))?;
        Ok(MixedReplayResult {
            outputs,
            committed,
            native_trace: None,
        })
    }

    /// Replays every pure prefix through strict native CPU JIT, then commits
    /// the resulting detached tensors through the same single EffectRuntime
    /// transaction as interpreter replay. No native failure can mutate state.
    pub fn replay_native(
        &self,
        runtime: &mut crate::EffectRuntime,
        provided: &BTreeMap<String, crate::TensorData>,
        executor: &super::captured_replay::CapturedReplayExecutor,
        vectorized: bool,
        injected_failure: Option<u64>,
    ) -> Result<MixedReplayResult, ReplayError> {
        validate(self)?;
        let native_trace = self.native_replay_trace(vectorized)?;
        let schedule = Schedule {
            items: self.schedule.items.clone(),
            value_bindings: self.value_bindings.clone(),
            state_bindings: self.state_bindings.clone(),
        };
        let split = schedule
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
        if schedule.items[split..].iter().any(|item| !item.is_effect()) {
            return Err(ReplayError::Unsupported(
                "mixed replay requires ordered pure then effect items".into(),
            ));
        }
        let mut pure_capture = self.schedule.clone();
        pure_capture.items.truncate(split);
        pure_capture.requested = self
            .value_bindings
            .iter()
            .map(|binding| binding.producer_output.id)
            .collect();
        pure_capture.identity = 0;
        let mut inputs = provided.clone();
        for binding in &self.state_bindings {
            let input = self
                .schedule
                .inputs
                .iter()
                .find(|input| input.node == binding.input_node)
                .ok_or_else(|| ReplayError::Corrupt("state input ABI is absent".into()))?;
            if provided.contains_key(&input.name) {
                return Err(ReplayError::Descriptor(
                    "external input shadows persistent state binding".into(),
                ));
            }
            let snapshot = runtime.snapshot(&binding.state).map_err(|error| {
                ReplayError::Execute(format!("persistent state preflight: {error:?}"))
            })?;
            let value = match &binding.view {
                Some(view) => snapshot.tensor().affine_read(view),
                None => Ok(snapshot.tensor().clone()),
            }
            .map_err(|error| ReplayError::Descriptor(format!("persistent affine read: {error}")))?;
            if value.shape() != &binding.desc.shape
                || value.dtype() != binding.desc.dtype
                || value.len().checked_mul(value.dtype().itemsize()) != Some(binding.desc.bytes)
            {
                return Err(ReplayError::Descriptor(
                    "persistent state input descriptor mismatch".into(),
                ));
            }
            inputs.insert(input.name.clone(), value);
        }
        preflight_effect_states(self, &schedule, runtime)?;
        let values = super::captured_replay::replay_native_items(
            &pure_capture,
            &inputs,
            executor,
            vectorized,
        )?;
        let outputs = self
            .schedule
            .requested
            .iter()
            .map(|id| {
                values
                    .get(id)
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing(id.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = effect_plan(&schedule)?;
        let mut sources = BTreeMap::new();
        for binding in &self.value_bindings {
            let payload = effect_payload(&schedule.items[binding.effect_item as usize])?;
            sources.insert(
                payload.step,
                values
                    .get(&binding.producer_output.id)
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing(binding.producer_output.id.to_string()))?,
            );
        }
        let committed = runtime
            .execute_with_sources(&plan, &sources, injected_failure)
            .map_err(|error| ReplayError::Execute(format!("persistent mixed replay: {error:?}")))?;
        Ok(MixedReplayResult {
            outputs,
            committed,
            native_trace: Some(native_trace),
        })
    }

    /// Computes the native replay trace identity before any native code or
    /// persistent state mutation. The RGSM identity carries decoded item and
    /// value/state ABI schema; the remaining fields bind native policy and
    /// the exact pure-item cache entries.
    pub fn native_replay_trace(
        &self,
        vectorized: bool,
    ) -> Result<NativeMixedReplayTrace, ReplayError> {
        validate(self)?;
        let artifact_identity = identity(self)?;
        let pure_item_cache_keys = self
            .schedule
            .items
            .iter()
            .take_while(|item| !item.is_effect())
            .map(|item| item.cache_key)
            .collect::<Vec<_>>();
        let mut bytes = self.to_bytes_without_identity()?;
        bytes.extend_from_slice(&artifact_identity.to_le_bytes());
        bytes.extend_from_slice(crate::cpu_jit::RENDERER_VERSION.as_bytes());
        bytes.extend_from_slice(std::env::consts::ARCH.as_bytes());
        bytes.extend_from_slice(std::env::consts::OS.as_bytes());
        bytes.push(u8::from(vectorized));
        for key in &pure_item_cache_keys {
            bytes.extend_from_slice(&key.to_le_bytes());
        }
        Ok(NativeMixedReplayTrace {
            identity: fnv1a(&bytes),
            artifact_identity,
            vectorized,
            pure_item_cache_keys,
        })
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    use crate::{
        BinaryOp, CapturedReplayExecutor, DType, EffectGraph, EffectRuntime, Graph,
        ScheduleValueBinding, Storage, TensorData, combine_mixed_schedules, schedule,
        schedule_effects,
    };

    #[test]
    fn decoded_rgsm_replays_pure_value_into_one_atomic_persistent_commit() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], DType::F32);
        let y = graph.input_dtype("y", [2], DType::F32);
        let sum = graph.binary(BinaryOp::Add, x, y).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let mut capture = CapturedSchedule::capture(&graph, &pure, &[sum]).unwrap();
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                100,
                TensorData::from_storage([2], Storage::F32(vec![0.0, 0.0])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                sum.index() as u64,
                TensorData::from_storage([2], Storage::F32(vec![0.0, 0.0])).unwrap(),
            )
            .unwrap();
        let next = effects.assign(&target, &source).unwrap();
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
        capture.items = mixed.items.clone();
        let artifact = CapturedMixedSchedule::from_parts(
            capture,
            &mixed,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap();
        let decoded = CapturedMixedSchedule::from_bytes(&artifact.to_bytes().unwrap()).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                100,
                TensorData::from_storage([2], Storage::F32(vec![9.0, 9.0])).unwrap(),
            )
            .unwrap();
        let inputs = BTreeMap::from([
            (
                "x".into(),
                TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap(),
            ),
            (
                "y".into(),
                TensorData::from_storage([2], Storage::F32(vec![3.0, 4.0])).unwrap(),
            ),
        ]);
        let native = CapturedReplayExecutor::default();
        let expected_trace = decoded.native_replay_trace(false).unwrap();
        assert!(
            decoded
                .replay_native(&mut runtime, &inputs, &native, false, Some(0))
                .is_err()
        );
        assert_eq!(
            runtime.snapshot(target.state()).unwrap().tensor().storage(),
            &Storage::F32(vec![9.0, 9.0])
        );
        let result = decoded
            .replay_native(&mut runtime, &inputs, &native, false, None)
            .unwrap();
        assert_eq!(result.native_trace, Some(expected_trace));
        // The injected commit failure still compiled the detached pure item;
        // retry must reuse that exact strict-native compilation.
        assert_eq!(native.compile_cache_len(false), 1);
        assert_eq!(result.outputs[0].storage(), &Storage::F32(vec![4.0, 6.0]));
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::F32(vec![4.0, 6.0])
        );
    }
}

fn effect_payload(item: &crate::ScheduleItem) -> Result<&crate::EffectPayload, ReplayError> {
    match item.kernel.arg() {
        crate::UArg::Effect(payload) => Ok(payload),
        _ => Err(ReplayError::Corrupt("effect payload is absent".into())),
    }
}

fn effect_plan(schedule: &Schedule) -> Result<crate::EffectPlan, ReplayError> {
    let mut steps = Vec::new();
    for item in schedule.items.iter().filter(|item| item.is_effect()) {
        let after = effect_payload(item)?;
        let store = item
            .kernel
            .sources()
            .first()
            .ok_or_else(|| ReplayError::Corrupt("effect AFTER lacks STORE".into()))?;
        let crate::UArg::Effect(store_payload) = store.arg() else {
            return Err(ReplayError::Corrupt(
                "effect STORE payload is absent".into(),
            ));
        };
        if store_payload.as_ref() != after {
            return Err(ReplayError::Corrupt(
                "effect STORE/AFTER payload mismatch".into(),
            ));
        }
        let predecessors = item
            .dependencies
            .iter()
            .filter(|id| schedule.items[**id as usize].is_effect())
            .map(|id| effect_payload(&schedule.items[*id as usize]).map(|payload| payload.step))
            .collect::<Result<Vec<_>, _>>()?;
        steps.push(crate::EffectStep {
            id: after.step,
            reads: vec![after.snapshot.clone(), after.source.clone()],
            write: after.target.clone(),
            target_view: after.target_view.clone(),
            index_plan: after.index_plan.clone(),
            after: predecessors,
        });
    }
    let plan = crate::EffectPlan { steps };
    plan.validate()
        .map_err(|error| ReplayError::Corrupt(format!("effect plan: {error}")))?;
    Ok(plan)
}

/// Validates every persistent state touched by the effect suffix before a
/// strict-native pure prefix is allowed to run. Detached native outputs never
/// enter the runtime, but stale state remains a complete-artifact preflight
/// error rather than a late commit error.
fn preflight_effect_states(
    capture: &CapturedMixedSchedule,
    schedule: &Schedule,
    runtime: &crate::EffectRuntime,
) -> Result<(), ReplayError> {
    let pure_sources = capture
        .value_bindings
        .iter()
        .map(|binding| binding.effect_item)
        .collect::<BTreeSet<_>>();
    let mut required_states = capture
        .state_bindings
        .iter()
        .map(|binding| binding.state.clone())
        .collect::<BTreeSet<_>>();
    for item in schedule.items.iter().filter(|item| item.is_effect()) {
        let payload = effect_payload(item)?;
        required_states.insert(payload.snapshot.clone());
        if !pure_sources.contains(&item.id) {
            required_states.insert(payload.source.clone());
        }
    }
    for state in &required_states {
        runtime.snapshot(state).map_err(|error| {
            ReplayError::Execute(format!("persistent state preflight: {error:?}"))
        })?;
    }
    Ok(())
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::{
        CapturedReplayExecutor, DType, EffectGraph, EffectRuntime, Shape, Storage, TensorData,
        schedule_effects,
    };

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
    fn rgsm_replays_indexed_effect_store_with_duplicates_and_raw_bits() {
        use crate::ir::indexing::{StaticIndex, StaticIndexPlan};
        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                70,
                TensorData::from_storage([3], Storage::F16(vec![1, 0x8000, 3])).unwrap(),
            )
            .unwrap();
        let source = effects
            .insert(
                71,
                TensorData::from_storage([3], Storage::F16(vec![0x7e01, 7, 0x8000])).unwrap(),
            )
            .unwrap();
        let plan = StaticIndexPlan::new(
            Shape::from([3]),
            &[StaticIndex::Advanced {
                shape: Shape::from([3]),
                values: vec![1, 1, -1],
            }],
        )
        .unwrap();
        let next = effects.static_index_assign(&target, &source, plan).unwrap();
        let schedule = schedule_effects(&effects).unwrap();
        let captured = CapturedMixedSchedule::from_parts(
            CapturedSchedule {
                items: schedule.items.clone(),
                inputs: vec![],
                constants: BTreeMap::new(),
                quantized_constants: BTreeMap::new(),
                requested: vec![],
                identity: 0,
                symbolic: None,
                specialized_from: None,
            },
            &schedule,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(bytes, captured.to_bytes().unwrap());
        let decoded = CapturedMixedSchedule::from_bytes(&bytes).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                70,
                TensorData::from_storage([3], Storage::F16(vec![1, 0x8000, 3])).unwrap(),
            )
            .unwrap();
        runtime
            .register(
                71,
                TensorData::from_storage([3], Storage::F16(vec![0x7e01, 7, 0x8000])).unwrap(),
            )
            .unwrap();
        let native = CapturedReplayExecutor::default();
        assert!(
            decoded
                .replay_native(&mut runtime, &BTreeMap::new(), &native, false, Some(0))
                .is_err()
        );
        assert_eq!(
            runtime.snapshot(target.state()).unwrap().tensor().storage(),
            &Storage::F16(vec![1, 0x8000, 3])
        );
        let result = decoded
            .replay_native(&mut runtime, &BTreeMap::new(), &native, false, None)
            .unwrap();
        assert!(result.native_trace.is_some());
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::F16(vec![1, 7, 0x8000])
        );
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
    fn rgsm_v1_envelope_upgrades_to_the_canonical_v2_identity() {
        let mut bytes = captured_effect().to_bytes().unwrap();
        bytes[4] = 1;
        let checksum_at = bytes.len() - 4;
        let sum = checksum(&bytes[..checksum_at]).to_le_bytes();
        bytes[checksum_at..].copy_from_slice(&sum);
        let decoded = CapturedMixedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap()[4], VERSION);
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
        let mut runtime = crate::EffectRuntime::new();
        runtime
            .register(
                100,
                TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap(),
            )
            .unwrap();
        let native = CapturedReplayExecutor::default();
        let replay = decoded
            .replay_native(
                &mut runtime,
                &BTreeMap::from([(
                    "bias".into(),
                    TensorData::from_storage([2], Storage::F32(vec![10.0, 20.0])).unwrap(),
                )]),
                &native,
                false,
                None,
            )
            .unwrap();
        assert!(replay.native_trace.is_some());
        assert_eq!(replay.outputs[0].storage(), &Storage::F32(vec![12.0, 21.0]));
        assert_eq!(
            runtime.snapshot(next.state()).unwrap().tensor().storage(),
            &Storage::F32(vec![12.0, 21.0])
        );
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
    let mut names = BTreeSet::new();
    let mut input_ids = BTreeSet::new();
    for input in &value.schedule.inputs {
        if !names.insert(&input.name)
            || !input_ids.insert(input.desc.id)
            || input.node.index() as u64 != input.desc.id
        {
            return Err(ReplayError::Corrupt("duplicate replay input".into()));
        }
    }
    let outputs = value
        .schedule
        .items
        .iter()
        .map(|item| item.output.id)
        .collect::<BTreeSet<_>>();
    let mut requested = BTreeSet::new();
    if value
        .schedule
        .requested
        .iter()
        .any(|id| !requested.insert(*id) || !outputs.contains(id))
    {
        return Err(ReplayError::Corrupt("invalid requested output".into()));
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
