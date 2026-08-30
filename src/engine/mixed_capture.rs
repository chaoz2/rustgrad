//! Immutable capture boundary for mixed pure/effect schedules.
//!
//! This type intentionally owns only logical schedule/state metadata. Runtime
//! leases, slot generations, pointers, and current bytes remain caller-owned.
use crate::uop::artifact::{ArtifactError, Reader, Writer, checksum};
use crate::{
    BufferDesc, BufferState, CapturedSchedule, EffectPayload, MixedStateRebinding, NodeId,
    Operation, ReplayError, ReplayInput, Schedule, ScheduleStateBinding, ScheduleValueBinding, UOp,
};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 4] = b"RGSM";
/// v3 adopts canonical schedule item/state-binding keys. v1-v2 retain opaque
/// historical keys and are upgraded only after their stored envelope passes.
const VERSION: u8 = 3;
const HEADER_LEN: usize = MAGIC.len() + 1 + std::mem::size_of::<u64>();
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

/// Validated, detached input binding for one mixed capture. It has no runtime
/// lease and performs neither pure execution nor persistent mutation.
#[allow(dead_code)]
pub(crate) struct BoundMixedCapture<'a> {
    capture: &'a CapturedMixedSchedule,
    inputs: BTreeMap<String, crate::TensorData>,
    starts: BTreeMap<u64, BufferState>,
}

/// Strict-native compilation of one already-bound pure prefix.
#[allow(dead_code)]
pub(crate) struct PlannedBoundMixedCapture<'a> {
    bound: BoundMixedCapture<'a>,
    plan: super::captured_replay::PlannedNativeItems,
}

#[allow(dead_code)]
impl<'a> BoundMixedCapture<'a> {
    pub(crate) fn bind(
        capture: &'a CapturedMixedSchedule,
        candidates: &BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        provided: &BTreeMap<String, crate::TensorData>,
    ) -> Result<Self, ReplayError> {
        validate(capture, true)?;
        let mut inputs = provided.clone();
        for binding in &capture.state_bindings {
            let input = capture
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
            let start = starts
                .get(&binding.state.buffer)
                .ok_or_else(|| ReplayError::Missing(binding.state.buffer.to_string()))?;
            let state = BufferState {
                version: start
                    .version
                    .checked_add(binding.state.version)
                    .ok_or_else(|| ReplayError::Corrupt("batch version overflow".into()))?,
                ..binding.state.clone()
            };
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
        Ok(Self {
            capture,
            inputs,
            starts,
        })
    }

    pub(crate) fn inputs(&self) -> &BTreeMap<String, crate::TensorData> {
        &self.inputs
    }
    pub(crate) fn starts(&self) -> &BTreeMap<u64, BufferState> {
        &self.starts
    }
    pub(crate) fn capture(&self) -> &'a CapturedMixedSchedule {
        self.capture
    }

    pub(crate) fn plan_native(
        self,
        executor: &super::captured_replay::CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<PlannedBoundMixedCapture<'a>, ReplayError> {
        let mut pure = self.capture.schedule.clone();
        let split = pure
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
        pure.items.truncate(split);
        pure.requested = self
            .capture
            .value_bindings
            .iter()
            .map(|x| x.producer_output.id)
            .collect();
        pure.identity = 0;
        let plan = executor.plan_native_items(&pure, &self.inputs, vectorized)?;
        Ok(PlannedBoundMixedCapture { bound: self, plan })
    }
}

#[allow(dead_code)]
impl<'a> PlannedBoundMixedCapture<'a> {
    /// Deterministic ABI schema identity for a bound capture. This deliberately
    /// describes names and tensor descriptors only: runtime state resources
    /// and input bytes must never influence a native batch cache/trace key.
    pub(crate) fn binding_schema_key(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for (name, tensor) in &self.bound.inputs {
            for byte in name.as_bytes() {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
            }
            hash = (hash ^ tensor.dtype() as u64).wrapping_mul(0x100000001b3);
            for &dimension in tensor.shape().dims() {
                hash = (hash ^ dimension as u64).wrapping_mul(0x100000001b3);
            }
        }
        hash
    }

    pub(crate) fn cache_keys(&self) -> Vec<u64> {
        self.bound
            .capture
            .schedule
            .items
            .iter()
            .take_while(|item| !item.is_effect())
            .map(|item| item.cache_key)
            .collect()
    }
    pub(crate) fn execute(
        &self,
        executor: &super::captured_replay::CapturedReplayExecutor,
    ) -> Result<super::captured_replay::ReplayValues, ReplayError> {
        let mut pure = self.bound.capture.schedule.clone();
        let split = pure
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
        pure.items.truncate(split);
        pure.identity = 0;
        executor.execute_planned_native_items(&pure, &self.bound.inputs, &self.plan)
    }

    pub(crate) fn execute_stage(
        self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        executor: &super::captured_replay::CapturedReplayExecutor,
    ) -> Result<crate::EffectBatchEntry, ReplayError> {
        let values = self.execute(executor)?;
        self.bound
            .capture
            .stage_values(candidates, self.bound.starts.clone(), values)
    }
}

impl CapturedMixedSchedule {
    /// Builds a replay-local state namespace without changing this RGSM's
    /// encoded bytes or identity.  The caller mapping is required to cover
    /// every persistent state referenced by the captured Store/After and
    /// state-input ABI exactly once.
    pub fn rebound(&self, rebinding: &MixedStateRebinding) -> Result<Self, ReplayError> {
        validate(self, true)?;
        let referenced = referenced_buffers(self)?;
        rebinding.validate_exact(&referenced)?;
        let map_state = |state: &BufferState| -> Result<BufferState, ReplayError> {
            Ok(BufferState {
                buffer: rebinding.destination(state.buffer)?,
                ..state.clone()
            })
        };
        let mut value = self.clone();
        value.states = self
            .states
            .iter()
            .map(map_state)
            .collect::<Result<_, _>>()?;
        value.state_bindings = self
            .state_bindings
            .iter()
            .cloned()
            .map(|mut binding| {
                binding.state = map_state(&binding.state)?;
                Ok(binding)
            })
            .collect::<Result<_, ReplayError>>()?;
        for item in &mut value.schedule.items {
            if !item.is_effect() {
                continue;
            }
            let crate::Operation::After(after) = item.kernel.operation() else {
                return Err(ReplayError::Corrupt("effect item missing payload".into()));
            };
            let payload = rebind_payload(after, &map_state)?;
            let store = item
                .kernel
                .sources()
                .first()
                .ok_or_else(|| ReplayError::Corrupt("effect item missing store".into()))?;
            let store_uop = UOp::from_operation(
                Operation::EffectStore(Box::new(payload.clone())),
                store.ty(),
                vec![],
            );
            item.kernel = UOp::from_operation(
                Operation::After(Box::new(payload)),
                item.kernel.ty(),
                vec![store_uop],
            );
            let pure_source = self
                .value_bindings
                .iter()
                .any(|binding| binding.effect_item == item.id);
            if !pure_source {
                for desc in &mut item.inputs {
                    if let Some(mapped) = rebinding.mapped(desc.id) {
                        desc.id = mapped;
                    }
                }
            }
            // Effect boundaries are not callable pure-kernel ABIs.  Their
            // original graph NodeId bindings cannot be renamed with a runtime
            // logical state, so retain no misleading binding table; the typed
            // Store/After payload and inventory carry the replay contract.
            item.input_bindings.clear();
            let outputs = item
                .outputs
                .iter()
                .cloned()
                .map(|mut output| {
                    if let Some(mapped) = rebinding.mapped(output.id) {
                        output.id = mapped;
                    }
                    output
                })
                .collect::<Vec<_>>();
            item.outputs = crate::ScheduledOutputs::new(outputs)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
            item.output = item.primary_output().clone();
        }
        let specialization = value
            .schedule
            .specialized_from
            .as_ref()
            .map(|source| (source.source_identity, source.bindings.as_slice()));
        crate::schedule::rekey_schedule_items(
            &mut value.schedule.items,
            &value.state_bindings,
            specialization,
        )
        .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        validate(&value, true)?;
        Ok(value)
    }

    /// Interpreter replay with a caller-selected persistent namespace.
    pub fn replay_with_rebinding(
        &self,
        runtime: &mut crate::EffectRuntime,
        provided: &BTreeMap<String, crate::TensorData>,
        rebinding: &MixedStateRebinding,
        injected_failure: Option<u64>,
    ) -> Result<MixedReplayResult, ReplayError> {
        self.rebound(rebinding)?
            .replay(runtime, provided, injected_failure)
    }

    /// Strict-native replay with a caller-selected persistent namespace.
    pub fn replay_native_with_rebinding(
        &self,
        runtime: &mut crate::EffectRuntime,
        provided: &BTreeMap<String, crate::TensorData>,
        rebinding: &MixedStateRebinding,
        executor: &super::captured_replay::CapturedReplayExecutor,
        vectorized: bool,
        injected_failure: Option<u64>,
    ) -> Result<MixedReplayResult, ReplayError> {
        let schema_key = rebinding.schema_key();
        let mut result = self.rebound(rebinding)?.replay_native(
            runtime,
            provided,
            executor,
            vectorized,
            injected_failure,
        )?;
        if let Some(trace) = &mut result.native_trace {
            trace.identity ^= schema_key;
        }
        Ok(result)
    }
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

    fn stage(
        &self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        provided: &BTreeMap<String, crate::TensorData>,
        native: Option<(&super::captured_replay::CapturedReplayExecutor, bool)>,
    ) -> Result<crate::EffectBatchEntry, ReplayError> {
        validate(self, true)?;
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
        self.stage_values(candidates, starts, values)
    }

    pub(crate) fn stage_values(
        &self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        values: super::captured_replay::ReplayValues,
    ) -> Result<crate::EffectBatchEntry, ReplayError> {
        let schedule = Schedule {
            items: self.schedule.items.clone(),
            value_bindings: self.value_bindings.clone(),
            state_bindings: self.state_bindings.clone(),
        };
        let plan = effect_plan(&schedule)?;
        let mut sources = BTreeMap::new();
        for binding in &self.value_bindings {
            let payload = effect_payload(&schedule.items[binding.effect_item as usize])?;
            sources.insert(
                payload.step,
                values
                    .tensor(binding.producer_output.id, "effect source")
                    .cloned()?,
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
        let value = Self {
            schedule,
            value_bindings: mixed.value_bindings.clone(),
            state_bindings: mixed.state_bindings.clone(),
            states,
        };
        validate(&value, true)?;
        Ok(value)
    }

    /// Bounded deterministic RGSM encoding. It intentionally excludes every
    /// runtime lease, slot, pointer, generation, and current buffer byte.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        validate(self, true)?;
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
        let legacy = version < VERSION;
        // Current envelopes must already carry canonical item identities.
        // Historical versions authenticate their opaque keys first and only
        // then derive the current representation below.
        validate(&decoded, !legacy)?;
        let actual = fnv1a(&bytes[HEADER_LEN..body]);
        if actual != stored_identity {
            return Err(ReplayError::Corrupt("RGSM identity".into()));
        }
        if legacy {
            crate::schedule::rekey_schedule_items(
                &mut schedule.items,
                &decoded.state_bindings,
                None,
            )
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        }
        let mut upgraded = Self {
            schedule,
            ..decoded
        };
        upgraded.schedule.identity = identity(&upgraded)?;
        validate(&upgraded, true)?;
        Ok(upgraded)
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
        validate(self, true)?;
        if self
            .schedule
            .items
            .iter()
            .any(|item| matches!(item.kernel.operation(), crate::Operation::TensorGuard(_)))
        {
            return Err(ReplayError::Unsupported(
                "tensor guard mixed replay is unsupported".into(),
            ));
        }
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
            .map(|id| values.tensor(*id, "requested mixed output").cloned())
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
                .tensor(binding.producer_output.id, "effect source")?
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
        validate(self, true)?;
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
            .map(|id| values.tensor(*id, "requested mixed output").cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let plan = effect_plan(&schedule)?;
        let mut sources = BTreeMap::new();
        for binding in &self.value_bindings {
            let payload = effect_payload(&schedule.items[binding.effect_item as usize])?;
            sources.insert(
                payload.step,
                values
                    .tensor(binding.producer_output.id, "effect source")
                    .cloned()?,
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
        validate(self, true)?;
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
            producer_output: pure.items[0].primary_output().clone(),
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

fn referenced_buffers(value: &CapturedMixedSchedule) -> Result<BTreeSet<u64>, ReplayError> {
    let mut buffers = value
        .states
        .iter()
        .map(|state| state.buffer)
        .collect::<BTreeSet<_>>();
    buffers.extend(
        value
            .state_bindings
            .iter()
            .map(|binding| binding.state.buffer),
    );
    for item in value.schedule.items.iter().filter(|item| item.is_effect()) {
        let payload = effect_payload(item)?;
        buffers.extend([
            payload.target.buffer,
            payload.source.buffer,
            payload.snapshot.buffer,
        ]);
    }
    Ok(buffers)
}

fn rebind_payload(
    payload: &EffectPayload,
    map: &impl Fn(&BufferState) -> Result<BufferState, ReplayError>,
) -> Result<EffectPayload, ReplayError> {
    Ok(EffectPayload {
        step: payload.step,
        target: map(&payload.target)?,
        source: map(&payload.source)?,
        snapshot: map(&payload.snapshot)?,
        target_view: payload.target_view.clone(),
        index_plan: payload.index_plan.clone(),
    })
}

fn effect_payload(item: &crate::ScheduleItem) -> Result<&crate::EffectPayload, ReplayError> {
    match item.kernel.operation() {
        crate::Operation::EffectStore(payload) | crate::Operation::After(payload) => Ok(payload),
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
        let crate::Operation::EffectStore(store_payload) = store.operation() else {
            return Err(ReplayError::Corrupt(
                "effect STORE payload is absent".into(),
            ));
        };
        if store_payload != after {
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
            decoded.schedule.items[0].kernel.sources()[0].operation(),
            crate::Operation::EffectStore(_)
        ));
        assert!(crate::uop::artifact::encode(&decoded.schedule.items[0].kernel).is_err());
        let _ = (DType::F16, Shape::from([2]));
    }

    #[test]
    fn rgsm_rejects_unserialized_symbolic_and_specialization_metadata() {
        let base = captured_effect();
        let mixed = Schedule {
            items: base.schedule.items.clone(),
            value_bindings: base.value_bindings.clone(),
            state_bindings: base.state_bindings.clone(),
        };
        let states = base.states.clone();

        let mut symbolic = base.schedule.clone();
        symbolic.symbolic = Some(crate::engine::symbolic::SymbolicSchema {
            parameters: vec![],
            template_values: vec![],
            guards: vec![],
            buffer_shapes: BTreeMap::new(),
            item_domains: BTreeMap::new(),
            views: BTreeMap::new(),
            splat_constants: BTreeSet::new(),
        });
        let symbolic_value = CapturedMixedSchedule {
            schedule: symbolic.clone(),
            value_bindings: base.value_bindings.clone(),
            state_bindings: base.state_bindings.clone(),
            states: base.states.clone(),
        };
        assert!(matches!(
            symbolic_value.to_bytes(),
            Err(ReplayError::Unsupported(message))
                if message == "RGSM does not encode symbolic schemas or specialization provenance"
        ));
        assert!(matches!(
            CapturedMixedSchedule::from_parts(symbolic, &mixed, states.clone()),
            Err(ReplayError::Unsupported(message))
                if message == "RGSM does not encode symbolic schemas or specialization provenance"
        ));

        let mut specialized = base.schedule.clone();
        specialized.specialized_from = Some(crate::engine::symbolic::SpecializedFrom {
            source_identity: 17,
            bindings: vec![(3, 5)],
        });
        assert!(matches!(
            CapturedMixedSchedule::from_parts(specialized.clone(), &mixed, states),
            Err(ReplayError::Unsupported(message))
                if message == "RGSM does not encode symbolic schemas or specialization provenance"
        ));
        let invalid = CapturedMixedSchedule {
            schedule: specialized,
            value_bindings: base.value_bindings,
            state_bindings: base.state_bindings,
            states: base.states,
        };
        assert!(matches!(
            invalid.to_bytes(),
            Err(ReplayError::Unsupported(message))
                if message == "RGSM does not encode symbolic schemas or specialization provenance"
        ));
    }

    #[test]
    fn replay_local_rebinding_preserves_artifact_and_raw_state_contract() {
        let captured = captured_effect();
        let bytes = captured.to_bytes().unwrap();
        let rebinding = MixedStateRebinding::new(BTreeMap::from([(40, 140), (41, 141)])).unwrap();
        let rebound = captured.rebound(&rebinding).unwrap();
        assert_eq!(captured.to_bytes().unwrap(), bytes);
        assert_eq!(rebound.states[0].buffer, 140);
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                140,
                TensorData::from_storage([2], Storage::F16(vec![0x8000, 0x7e01])).unwrap(),
            )
            .unwrap();
        runtime
            .register(
                141,
                TensorData::from_storage([2], Storage::F16(vec![1, 2])).unwrap(),
            )
            .unwrap();
        rebound
            .replay(&mut runtime, &BTreeMap::new(), None)
            .unwrap();
        assert_eq!(
            runtime
                .snapshot(&BufferState {
                    buffer: 140,
                    version: 1,
                    shape: Shape::from([2]),
                    dtype: DType::F16,
                    bytes: 4
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F16(vec![1, 2])
        );
        for bad in [
            BTreeMap::from([(40, 140)]),
            BTreeMap::from([(40, 140), (41, 140)]),
            BTreeMap::from([(40, 140), (41, 141), (99, 199)]),
        ] {
            assert!(
                MixedStateRebinding::new(bad.clone())
                    .and_then(|value| captured.rebound(&value))
                    .is_err()
            );
        }
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
        let rebinding = MixedStateRebinding::new(BTreeMap::from([(70, 170), (71, 171)])).unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                170,
                TensorData::from_storage([3], Storage::F16(vec![1, 0x8000, 3])).unwrap(),
            )
            .unwrap();
        runtime
            .register(
                171,
                TensorData::from_storage([3], Storage::F16(vec![0x7e01, 7, 0x8000])).unwrap(),
            )
            .unwrap();
        let native = CapturedReplayExecutor::default();
        assert!(
            decoded
                .replay_native_with_rebinding(
                    &mut runtime,
                    &BTreeMap::new(),
                    &rebinding,
                    &native,
                    false,
                    Some(0),
                )
                .is_err()
        );
        assert_eq!(
            runtime
                .snapshot(&BufferState {
                    buffer: 170,
                    ..target.state().clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F16(vec![1, 0x8000, 3])
        );
        let result = decoded
            .replay_native_with_rebinding(
                &mut runtime,
                &BTreeMap::new(),
                &rebinding,
                &native,
                false,
                None,
            )
            .unwrap();
        assert!(result.native_trace.is_some());
        assert_eq!(
            runtime
                .snapshot(&BufferState {
                    buffer: 170,
                    ..next.state().clone()
                })
                .unwrap()
                .tensor()
                .storage(),
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
    fn released_rgsm_v1_v2_layout_fixture_remains_decodable() {
        // Preserve the historical fixture used before the identity migration:
        // v1 and v2 share the released typed field layout, differing only in
        // envelope policy. Updating only the version/checksum must therefore
        // remain a supported decode-and-upgrade path.
        for legacy_version in [1, 2] {
            let mut bytes = captured_effect().to_bytes().unwrap();
            bytes[4] = legacy_version;
            let checksum_at = bytes.len() - 4;
            let sum = checksum(&bytes[..checksum_at]).to_le_bytes();
            bytes[checksum_at..].copy_from_slice(&sum);
            let decoded = CapturedMixedSchedule::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.to_bytes().unwrap()[4], VERSION);
        }
    }

    #[test]
    fn legacy_rgsm_envelopes_upgrade_to_the_canonical_v3_identity() {
        for legacy_version in [1, 2] {
            let mut legacy = captured_effect();
            for (index, item) in legacy.schedule.items.iter_mut().enumerate() {
                item.cache_key = 0x8877_6655_4433_2200 + index as u64;
            }
            let opaque = legacy
                .schedule
                .items
                .iter()
                .map(|item| item.cache_key)
                .collect::<Vec<_>>();
            let payload = legacy.to_bytes_without_identity().unwrap();
            let mut writer = Writer::new();
            writer.bytes(MAGIC).unwrap();
            writer.u8(legacy_version).unwrap();
            writer.u64(fnv1a(&payload)).unwrap();
            writer.bytes(&payload).unwrap();
            let sum = checksum(&writer.out);
            writer.u32(sum).unwrap();
            let decoded = CapturedMixedSchedule::from_bytes(&writer.out).unwrap();
            assert_eq!(decoded.to_bytes().unwrap()[4], VERSION);
            assert_ne!(
                decoded
                    .schedule
                    .items
                    .iter()
                    .map(|item| item.cache_key)
                    .collect::<Vec<_>>(),
                opaque
            );
        }
    }

    #[test]
    fn current_rgsm_rejects_authenticated_noncanonical_item_keys() {
        let mut forged = captured_effect();
        forged.schedule.items[0].cache_key ^= 1;
        let payload = forged.to_bytes_without_identity().unwrap();
        let mut writer = Writer::new();
        writer.bytes(MAGIC).unwrap();
        writer.u8(VERSION).unwrap();
        writer.u64(fnv1a(&payload)).unwrap();
        writer.bytes(&payload).unwrap();
        let sum = checksum(&writer.out);
        writer.u32(sum).unwrap();
        assert!(matches!(
            CapturedMixedSchedule::from_bytes(&writer.out),
            Err(ReplayError::Corrupt(message)) if message.contains("cache identity")
        ));
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
            producer_output: pure.items[0].primary_output().clone(),
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

fn validate(value: &CapturedMixedSchedule, validate_keys: bool) -> Result<(), ReplayError> {
    if value.schedule.symbolic.is_some() || value.schedule.specialized_from.is_some() {
        return Err(ReplayError::Unsupported(
            "RGSM does not encode symbolic schemas or specialization provenance".into(),
        ));
    }
    let schedule = Schedule {
        items: value.schedule.items.clone(),
        value_bindings: value.value_bindings.clone(),
        state_bindings: value.state_bindings.clone(),
    };
    schedule
        .validate()
        .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
    if validate_keys {
        let mut expected = schedule.items.clone();
        let specialization = value
            .schedule
            .specialized_from
            .as_ref()
            .map(|source| (source.source_identity, source.bindings.as_slice()));
        crate::schedule::rekey_schedule_items(
            &mut expected,
            &schedule.state_bindings,
            specialization,
        )
        .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        if expected
            .iter()
            .zip(&schedule.items)
            .any(|(expected, actual)| expected.cache_key != actual.cache_key)
        {
            return Err(ReplayError::Corrupt("RGSM item cache identity".into()));
        }
    }
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
        .flat_map(|item| item.outputs.iter().map(|output| output.id))
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
